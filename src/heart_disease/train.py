"""Training CLI: tune, cross-validate tuned models, evaluate, and save pipeline.

Usage::

    python -m heart_disease.train
"""

from __future__ import annotations

import json
import logging
import platform
import warnings
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from heart_disease.config import Config, get_project_root, load_config
from heart_disease.data import (
    clean_data,
    get_site_grouped_data,
    load_and_prepare,
    load_raw_data,
)
from heart_disease.evaluate import (
    compare_calibration_methods,
    compute_bootstrap_ci,
    cross_validate_model_with_cv,
    evaluate_leave_one_site_out,
    evaluate_model,
    evaluate_subgroups_by_sex,
    find_optimal_threshold_oof,
    generate_site_confounding_report,
    plot_calibration_curve,
    plot_comparison_table,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    run_feature_ablation,
)
from heart_disease.explain import plot_shap_summary
from heart_disease.features import build_preprocessor
from heart_disease.logging_utils import setup_logging
from heart_disease.models import create_pipeline, get_available_models
from heart_disease.schema import ALL_FEATURES, RAW_CLINICAL_FEATURES

# Suppress harmless BLAS floating-point warnings on Apple Silicon during L-BFGS optimization
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*(divide by zero|overflow|invalid value) encountered in matmul.*",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------


def cross_validate_model(
    pipeline: Any,
    x: pd.DataFrame,
    y: pd.Series,
    config: Config,
) -> dict[str, float]:
    """Run repeated stratified k-fold cross-validation using config settings."""
    cv = RepeatedStratifiedKFold(
        n_splits=config.training.cv_folds,
        n_repeats=config.training.cv_repeats,
        random_state=config.project.random_seed,
    )
    return cross_validate_model_with_cv(pipeline, x, y, cv)


# ---------------------------------------------------------------------------
# Model Selection (Non-Self-Referential)
# ---------------------------------------------------------------------------


def select_best_model(
    cv_results: dict[str, dict[str, float]],
    config: Config,
) -> tuple[str, str]:
    """Select the best model according to the documented selection rule.

    Rule:
    1. Primary metric: ROC-AUC.
    2. 1-fold std rule: Identify all candidate models whose mean ROC-AUC is
       within one fold standard deviation of the top model
       (score >= max_score - 1.0 * max_std). Differences smaller than one fold
       std are treated as statistically indistinguishable.
    3. Tie-breaking among candidates:
       a. Model simplicity: logistic_regression > random_forest > xgboost.
       b. Lower standard deviation (higher stability across CV folds).

    Args:
        cv_results: Cross-validation results dictionary for all models.
        config: Project configuration.

    Returns:
        Tuple of ``(selected_model_name, selection_rationale)``.
    """
    primary = config.selection.primary_metric
    tol_factor = config.selection.tolerance_std_factor
    simplicity = config.selection.simplicity_order

    mean_key = f"{primary}_mean"
    std_key = f"{primary}_std"

    # Step 1: Find model with highest mean score
    top_model = max(cv_results, key=lambda m: cv_results[m][mean_key])
    top_mean = cv_results[top_model][mean_key]
    top_std = cv_results[top_model][std_key]
    cutoff = top_mean - (tol_factor * top_std)

    # Step 2: Candidates within 1 fold std
    candidates = [m for m, scores in cv_results.items() if scores[mean_key] >= cutoff]

    model_summaries = [
        f"'{m}' ({cv_results[m][mean_key]:.4f} ± {cv_results[m][std_key]:.4f})"
        for m in sorted(cv_results.keys())
    ]
    models_text = ", ".join(model_summaries)

    logger.info(
        "Model Selection: Top model on %s is '%s' (%.4f ± %.4f). Cutoff: %.4f. Candidates: %s",
        primary,
        top_model,
        top_mean,
        top_std,
        cutoff,
        candidates,
    )

    if len(candidates) == 1:
        rationale = (
            f"Evaluated 3 models ({models_text}). Ranked #1 on {primary} is '{top_model}' "
            f"({top_mean:.4f} ± {top_std:.4f}), strictly outperforming all competing models "
            f"beyond the {tol_factor}-fold std tolerance cutoff ({cutoff:.4f})."
        )
        return candidates[0], rationale

    # Step 3: Tie-breaking (simplicity first for Occam's razor, then lower fold std)
    def tie_break_key(model: str) -> tuple[int, float]:
        simp_idx = simplicity.index(model) if model in simplicity else 99
        m_std = cv_results[model][std_key]
        return (simp_idx, m_std)

    selected = sorted(candidates, key=tie_break_key)[0]
    sel_mean = cv_results[selected][mean_key]
    sel_std = cv_results[selected][std_key]
    diff = top_mean - sel_mean

    if selected == top_model:
        rationale = (
            f"Evaluated 3 models ({models_text}). With a 1-fold std tolerance cutoff of "
            f"{cutoff:.4f}, all competing models are statistically indistinguishable. "
            f"Selected '{selected}' ({top_mean:.4f} ± {top_std:.4f}) based on simplicity and "
            f"clinical interpretability (linear log-odds parsimony) over complex tree ensembles "
            f"when scores are within one std."
        )
    else:
        std_note = (
            "with lower fold std and higher clinical simplicity"
            if sel_std < top_std
            else "based on simplicity and clinical interpretability (linear log-odds parsimony)"
        )
        rationale = (
            f"Evaluated 3 models ({models_text}). Top model on {primary} is '{top_model}' "
            f"({top_mean:.4f} ± {top_std:.4f}) with a 1-fold std tolerance cutoff of "
            f"{cutoff:.4f}. Selected '{selected}' ({sel_mean:.4f} ± {sel_std:.4f}) as score "
            f"difference ({diff:.4f}) is within 1 fold std ({top_std:.4f}); choice is {std_note} "
            f"when discriminative performance is statistically indistinguishable."
        )

    logger.info("Selected model: '%s' | Rationale: %s", selected, rationale)
    return selected, rationale


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune_model(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    config: Config,
) -> tuple[Any, dict[str, Any], float]:
    """Tune a model with ``RandomizedSearchCV``.

    Args:
        model_name: Registered model name.
        x: Training features.
        y: Training target.
        config: Project configuration.

    Returns:
        Tuple of ``(best_pipeline, best_params, best_score)``.
    """
    pipeline = create_pipeline(model_name, config)

    model_config = config.models.get(model_name)
    if model_config is None:
        logger.warning("No param grid for %s; fitting with defaults", model_name)
        pipeline.fit(x, y)
        return pipeline, {}, 0.0

    param_grid = model_config.param_grid

    cv = StratifiedKFold(
        n_splits=config.training.cv_folds,
        shuffle=True,
        random_state=config.project.random_seed,
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=config.training.n_iter,
        scoring=config.training.scoring,
        cv=cv,
        random_state=config.project.random_seed,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    search.fit(x, y)

    best_score = float(search.best_score_)
    logger.info(
        "%s best %s: %.4f | best params: %s",
        model_name,
        config.training.scoring,
        best_score,
        search.best_params_,
    )

    return search.best_estimator_, search.best_params_, best_score


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(
    pipeline: Any,
    x: pd.DataFrame,
    y: pd.Series,
    config: Config,
    method: str = "none",
) -> Any:
    """Prepare final fitted pipeline with optional calibration.

    Args:
        pipeline: Base sklearn Pipeline.
        x: Training features.
        y: Training target.
        config: Project configuration.
        method: Calibration method ('none', 'sigmoid', or 'isotonic').

    Returns:
        Fitted pipeline (either uncalibrated or wrapped in CalibratedClassifierCV).
    """
    if method == "none":
        fitted = clone(pipeline)
        fitted.fit(x, y)
        logger.info("Fitted uncalibrated pipeline on full training data (method='none')")
        return fitted

    calibrated = CalibratedClassifierCV(
        clone(pipeline),
        method=method,
        cv=config.training.cv_folds,
    )
    calibrated.fit(x, y)
    logger.info("Model calibrated with %s regression (cv=%d)", method, config.training.cv_folds)
    return calibrated


# ---------------------------------------------------------------------------
# Missingness & Imputation Documentation Helper
# ---------------------------------------------------------------------------


def get_feature_missingness_report(raw_df: pd.DataFrame, config: Config) -> list[dict[str, Any]]:
    """Compute missing rate and documented imputation strategy for each feature."""
    df = raw_df.copy()
    if "thalch" in df.columns and "thalach" not in df.columns:
        df = df.rename(columns={"thalch": "thalach"})

    for marker in config.data.missing_markers:
        df.replace(marker, np.nan, inplace=True)  # noqa: PD002

    # Physiological zeros: chol=0 and trestbps=0 are converted to NaN (treated as missing)
    for col in ("chol", "trestbps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0, col] = np.nan

    total = len(df)
    report: list[dict[str, Any]] = []

    for feat in ALL_FEATURES:
        if feat in df.columns:
            cnt = int(df[feat].isna().sum())
            pct = round(cnt / total * 100, 2)
        else:
            cnt, pct = 0, 0.0

        if feat in config.features.numeric:
            strategy = "SimpleImputer(strategy='median')"
            kind = "numeric"
        else:
            strategy = "SimpleImputer(strategy='most_frequent')"
            kind = "categorical"

        report.append(
            {
                "feature": feat,
                "type": kind,
                "missing_count": cnt,
                "missing_pct": pct,
                "imputation_strategy": strategy,
            }
        )

    return report


# ---------------------------------------------------------------------------
# Parameter Serialization Helper
# ---------------------------------------------------------------------------


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Make hyperparameters JSON-serialisable."""
    serialized: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            serialized[k] = int(v)
        elif isinstance(v, np.floating):
            serialized[k] = float(v)
        elif isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        else:
            serialized[k] = v
    return serialized


# ---------------------------------------------------------------------------
# Save Pipeline and Complete Metadata
# ---------------------------------------------------------------------------


def save_pipeline(
    pipeline: Any,
    metadata: dict[str, Any],
    config: Config,
) -> None:
    """Save the trained pipeline and comprehensive metadata to disk.

    Args:
        pipeline: Trained (and calibrated) pipeline.
        metadata: Metadata dictionary to serialize.
        config: Project configuration.
    """
    root = get_project_root()
    models_dir = root / config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline artifact
    pipeline_path = models_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved pipeline artifact to %s", pipeline_path)

    # Save metadata JSON
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata to %s", metrics_path)


# ---------------------------------------------------------------------------
# Main Pipeline Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete end-to-end training and evaluation pipeline."""
    setup_logging()
    config = load_config()

    np.random.seed(config.project.random_seed)

    logger.info("=" * 60)
    logger.info("Heart Disease ML — Training Pipeline")
    logger.info("=" * 60)

    # ---- Confirm feature safety ----
    logger.info("Auditing features: Confirming 'id' and 'dataset' are excluded...")
    assert "id" not in ALL_FEATURES, "'id' must not be in ALL_FEATURES"
    assert "dataset" not in ALL_FEATURES, "'dataset' must not be in ALL_FEATURES"
    assert "id" not in config.features.numeric and "id" not in config.features.categorical
    assert (
        "dataset" not in config.features.numeric and "dataset" not in config.features.categorical
    )
    logger.info("Final feature list (%d features): %s", len(ALL_FEATURES), ALL_FEATURES)

    # ---- Load & prepare data ----
    raw_df = load_raw_data(config)
    x_train, x_test, y_train, y_test = load_and_prepare(config)

    total_rows = len(raw_df)
    train_size = len(x_train)
    test_size = len(x_test)

    class_balance_overall = {
        "0": int((raw_df["num"] == 0).sum())
        if "num" in raw_df.columns
        else int((y_train == 0).sum() + (y_test == 0).sum()),
        "1": int((raw_df["num"] >= 1).sum())
        if "num" in raw_df.columns
        else int((y_train == 1).sum() + (y_test == 1).sum()),
    }
    class_balance_train = {
        "0": int((y_train == 0).sum()),
        "1": int((y_train == 1).sum()),
        "disease_prevalence": round(float(y_train.mean()), 4),
    }
    class_balance_test = {
        "0": int((y_test == 0).sum()),
        "1": int((y_test == 1).sum()),
        "disease_prevalence": round(float(y_test.mean()), 4),
    }

    # Generate missingness and imputation report
    feature_missing_report = get_feature_missingness_report(raw_df, config)

    root = get_project_root()
    reports_dir = root / config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate multi-site demographic and confounding report
    logger.info("Generating multi-site demographic and confounding report...")
    generate_site_confounding_report(raw_df, config, reports_dir)

    # ---- Phase 1: Hyperparameter Tuning via RandomizedSearchCV (all models) ----
    logger.info("Phase 1: Hyperparameter Tuning (all models)")
    tuned_models: dict[str, tuple[Any, dict[str, Any]]] = {}
    tuning_details: dict[str, Any] = {}

    for model_name in get_available_models():
        logger.info("Tuning %s …", model_name)
        best_pipe, best_p, best_s = tune_model(model_name, x_train, y_train, config)
        tuned_models[model_name] = (best_pipe, best_p)
        model_cfg = config.models.get(model_name)
        tuning_details[model_name] = {
            "search_space": model_cfg.param_grid if model_cfg else {},
            "best_params": _serialize_params(best_p),
            "best_cv_score": round(best_s, 4),
            "scoring": config.training.scoring,
        }

    # ---- Phase 2: Repeated Stratified Cross-Validation (5 Folds x 10 Repeats = 50 Splits) ----
    logger.info(
        "Phase 2: Repeated Stratified CV of Tuned Models (%d folds x %d repeats = %d splits)",
        config.training.cv_folds,
        config.training.cv_repeats,
        config.training.cv_folds * config.training.cv_repeats,
    )
    cv_results: dict[str, dict[str, float]] = {}
    cv_hyperparameters: dict[str, dict[str, Any]] = {}
    eval_seed = config.project.random_seed + 10  # Fresh seed for evaluation

    for model_name, (best_pipe, best_p) in tuned_models.items():
        eval_pipeline = clone(best_pipe)
        cv = RepeatedStratifiedKFold(
            n_splits=config.training.cv_folds,
            n_repeats=config.training.cv_repeats,
            random_state=eval_seed,
        )
        scores = cross_validate_model_with_cv(eval_pipeline, x_train, y_train, cv)
        cv_results[model_name] = scores
        cv_hyperparameters[model_name] = _serialize_params(best_p)
        logger.info(
            "%s (tuned, 5x10 repeated CV) | ROC-AUC: %.4f ± %.4f | "
            "PR-AUC: %.4f ± %.4f | Brier: %.4f ± %.4f",
            model_name,
            scores["roc_auc_mean"],
            scores["roc_auc_std"],
            scores["pr_auc_mean"],
            scores["pr_auc_std"],
            scores["brier_score_mean"],
            scores["brier_score_std"],
        )

    plot_comparison_table(cv_results, reports_dir, model_params=cv_hyperparameters)

    # ---- Phase 3: Model Selection via Documented Rule ----
    logger.info("Phase 3: Model Selection (Documented Rule on Tuned Models)")
    best_model_name, selection_rationale = select_best_model(cv_results, config)
    best_pipeline, best_params = tuned_models[best_model_name]

    # ---- Phase 4: Out-of-Fold Calibration Comparison on Training Set ----
    logger.info("Phase 4: Out-of-Fold Calibration Comparison ('none', 'sigmoid', 'isotonic')")
    (
        chosen_calibration,
        calibration_rationale,
        calibration_metrics,
        oof_predictions_dict,
    ) = compare_calibration_methods(best_pipeline, x_train, y_train, config, reports_dir)

    # ---- Phase 5: Out-of-Fold Threshold Analysis on Chosen Model's OOF Predictions ----
    logger.info(
        "Phase 5: Out-of-Fold Threshold Analysis on Chosen Model ('%s') Training Predictions",
        chosen_calibration,
    )
    if chosen_calibration == "none":
        thresh_pipeline = best_pipeline
    else:
        thresh_pipeline = CalibratedClassifierCV(
            estimator=clone(best_pipeline),
            method=chosen_calibration,
            cv=config.training.cv_folds,
        )

    optimal_threshold, oof_metrics = find_optimal_threshold_oof(
        thresh_pipeline, x_train, y_train, config, reports_dir
    )

    # ---- Phase 5b: Fit Final Model Pipeline on Full Training Set ----
    logger.info(
        "Phase 5b: Fitting Final Pipeline on Full Training Set (Calibration: '%s')",
        chosen_calibration,
    )
    final_pipeline = calibrate_model(
        best_pipeline, x_train, y_train, config, method=chosen_calibration
    )

    # ---- Phase 6: Final Evaluation on Held-Out Test Set ----
    logger.info("Phase 6: Final Evaluation on Held-Out Test Set")
    test_metrics = evaluate_model(final_pipeline, x_test, y_test, threshold=optimal_threshold)
    y_test_prob = final_pipeline.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

    # 95% Bootstrap Confidence Intervals (1000 resamples)
    logger.info("Computing 95% bootstrap confidence intervals (1,000 resamples)...")
    test_cis = compute_bootstrap_ci(
        y_test,
        y_test_prob,
        threshold=optimal_threshold,
        n_bootstraps=1000,
        random_state=config.project.random_seed,
    )

    cm = confusion_matrix(y_test, y_test_pred).tolist()

    logger.info("Test set results (threshold=%.3f):", optimal_threshold)
    for m, val in test_metrics.items():
        ci_str = f" [95% CI: {test_cis[m][0]:.4f} - {test_cis[m][1]:.4f}]" if m in test_cis else ""
        logger.info("  %-12s: %.4f%s", m, val, ci_str)
    logger.info(
        "Test Confusion Matrix:\n  TN=%d, FP=%d\n  FN=%d, TP=%d",
        cm[0][0],
        cm[0][1],
        cm[1][0],
        cm[1][1],
    )

    # Save evaluation plots
    all_pipelines = {name: pipe for name, (pipe, _) in tuned_models.items()}
    plot_roc_curve(all_pipelines, x_test, y_test, reports_dir)
    plot_pr_curve(all_pipelines, x_test, y_test, reports_dir)
    plot_confusion_matrix(final_pipeline, x_test, y_test, optimal_threshold, reports_dir)
    plot_calibration_curve(final_pipeline, x_test, y_test, reports_dir)

    # SHAP summary on the raw best pipeline
    plot_shap_summary(best_pipeline, x_train, config, reports_dir)

    # ---- Phase 7: Leave-One-Site-Out (LOSO) Cross-Validation ----
    logger.info("Phase 7: Multi-Center Leave-One-Site-Out (LOSO) Evaluation")
    x_site, y_site, sites = get_site_grouped_data(config)
    sites_train = sites.loc[x_train.index]
    evaluate_leave_one_site_out(
        pipeline_factory=lambda: create_pipeline(best_model_name, config),
        x=x_site,
        y=y_site,
        sites=sites,
        output_dir=reports_dir,
        threshold=optimal_threshold,
    )

    # ---- Phase 7b: Feature Ablation Study ((A), (B), (C) via Repeated CV + LOSO) ----
    logger.info("Phase 7b: Feature Ablation Study across Sets (A), (B), and (C)")
    clean_all = clean_data(raw_df, config, preserve_columns=["dataset"])
    avail_ablation_cols = [c for c in RAW_CLINICAL_FEATURES if c in clean_all.columns]
    x_ablation_train = clean_all.loc[x_train.index, avail_ablation_cols]
    x_ablation_full = clean_all[avail_ablation_cols]

    def ablation_builder(nf: list[str], cf: list[str]) -> Any:
        return create_pipeline(
            best_model_name,
            config,
            preprocessor=build_preprocessor(config, nf, cf),
            **best_params,
        )

    run_feature_ablation(
        pipeline_factory_builder=ablation_builder,
        x_train=x_ablation_train,
        y_train=y_train,
        x_full=x_ablation_full,
        y_full=y_site,
        sites_full=sites,
        config=config,
        output_dir=reports_dir,
        threshold=optimal_threshold,
        sites_train=sites_train,
    )

    # ---- Phase 7c: Subgroup Performance Evaluation by Biological Sex ----
    logger.info("Phase 7c: Subgroup Performance Evaluation by Sex (Test Set & Pooled LOSO)")
    subgroup_metrics = evaluate_subgroups_by_sex(
        pipeline=final_pipeline,
        x_test=x_test,
        y_test=y_test,
        threshold=optimal_threshold,
        output_dir=reports_dir,
        x_site=x_site,
        y_site=y_site,
        sites=sites,
        pipeline_factory=lambda: create_pipeline(best_model_name, config, **best_params),
        n_bootstraps=1000,
        random_state=config.project.random_seed,
    )

    # ---- Phase 8: Save Pipeline & Comprehensive Metadata ----
    metadata: dict[str, Any] = {
        "version": config.project.version,
        "model": best_model_name,
        "decision_threshold": optimal_threshold,
        "random_seed": config.project.random_seed,
        "zero_as_missing_rule": (
            "In clinical physiology, a serum cholesterol of 0 mg/dL (chol=0) and resting blood "
            "pressure of 0 mm Hg (trestbps=0) are biologically impossible and represent missing "
            "clinical data or database null/zero-fill encodings in historical cohorts. "
            "clean_data() converts chol=0 and trestbps=0 to NaN prior to pipeline "
            "median imputation."
        ),
        "dataset_summary": {
            "raw_row_count": total_rows,
            "cleaned_row_count": len(x_site),
            "class_balance_overall": class_balance_overall,
            "class_balance_train": class_balance_train,
            "class_balance_test": class_balance_test,
            "split_sizes": {"train": train_size, "test": test_size},
        },
        "features": {
            "numeric": config.features.numeric,
            "categorical": config.features.categorical,
            "all": ALL_FEATURES,
            "count": len(ALL_FEATURES),
        },
        "feature_imputation_and_missingness": feature_missing_report,
        "calibration": {
            "selected_method": chosen_calibration,
            "selection_rule": (
                "Prefer simpler option ('none' > 'sigmoid' > 'isotonic') unless an alternative "
                "achieves ΔBrier > 0.005 and ΔLogLoss > 0.010 on out-of-fold training predictions."
            ),
            "selection_rationale": calibration_rationale,
            "comparison_oof_metrics": calibration_metrics,
        },
        "selected_model": {
            "name": best_model_name,
            "selection_rule": (
                "Primary: ROC-AUC. 1-fold std rule (diff < 1.0 * std). "
                "Tie-breakers: model simplicity (Occam's razor), then lower fold std."
            ),
            "selection_reason": selection_rationale,
            "best_params": _serialize_params(best_params),
            "decision_threshold": optimal_threshold,
            "oof_recall_at_threshold": oof_metrics["recall"],
            "oof_metrics_at_threshold": oof_metrics,
        },
        "cv_hyperparameters_used": cv_hyperparameters,
        "tuning_details_all_models": tuning_details,
        "metrics": {
            "cv_results": cv_results,
            "test_metrics": test_metrics,
            "test_metrics_ci_95": test_cis,
            "test_confusion_matrix": cm,
            "oof_metrics_at_threshold": oof_metrics,
            "subgroup_analysis": subgroup_metrics,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "platform": platform.platform(),
        },
    }

    save_pipeline(final_pipeline, metadata, config)

    logger.info("=" * 60)
    logger.info("Training pipeline complete! Best model: %s", best_model_name)
    logger.info("Decision threshold: %.3f", optimal_threshold)
    logger.info(
        "OOF recall at threshold: %.4f (target: %.2f)",
        oof_metrics["recall"],
        config.training.target_recall,
    )
    logger.info(
        "Test recall: %.4f [95%% CI: %.4f - %.4f]",
        test_metrics["recall"],
        test_cis["recall"][0],
        test_cis["recall"][1],
    )
    logger.info(
        "Test precision: %.4f [95%% CI: %.4f - %.4f]",
        test_metrics["precision"],
        test_cis["precision"][0],
        test_cis["precision"][1],
    )
    logger.info(
        "Test ROC-AUC: %.4f [95%% CI: %.4f - %.4f]",
        test_metrics["roc_auc"],
        test_cis["roc_auc"][0],
        test_cis["roc_auc"][1],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
