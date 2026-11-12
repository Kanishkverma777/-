"""Tests for evaluation metrics and threshold analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from heart_disease.config import Config
from heart_disease.evaluate import (
    evaluate_model,
    plot_calibration_curve,
    plot_comparison_table,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from heart_disease.models import create_pipeline


def test_evaluate_model_returns_all_metrics(
    sample_dataframe: pd.DataFrame, config: Config
) -> None:
    """evaluate_model computes all expected metrics within [0, 1]."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    metrics = evaluate_model(pipeline, x, y, threshold=0.5)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "threshold",
    }
    assert set(metrics.keys()) == expected_keys

    for k in (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ):
        val = metrics[k]
        assert 0.0 <= val <= 1.0, f"Metric {k} out of bounds: {val}"


def test_run_threshold_analysis(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """run_threshold_analysis selects an optimal threshold and creates a plot."""
    from heart_disease.evaluate import find_optimal_threshold_oof

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    threshold, oof_metrics = find_optimal_threshold_oof(pipeline, x, y, config, tmp_path)
    assert 0.01 <= threshold <= 0.99
    assert isinstance(oof_metrics, dict)
    assert "recall" in oof_metrics
    # Assert threshold is rounded to at most 3 decimal places
    assert round(threshold, 3) == threshold
    assert (tmp_path / "threshold_analysis.png").exists()


def test_compute_bootstrap_ci() -> None:
    """compute_bootstrap_ci returns valid 95% confidence intervals."""
    import numpy as np

    from heart_disease.evaluate import compute_bootstrap_ci

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0] * 5)
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.05] * 5)

    cis = compute_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstraps=50, random_state=42)

    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"):
        assert metric in cis
        low, high = cis[metric]
        assert 0.0 <= low <= high <= 1.0


def test_evaluate_leave_one_site_out(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """evaluate_leave_one_site_out runs multi-center evaluation and saves reports."""
    from heart_disease.evaluate import evaluate_leave_one_site_out

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]
    # 2 sites for test
    sites = pd.Series(["Cleveland"] * 25 + ["Hungary"] * 25, name="dataset")

    df_loso = evaluate_leave_one_site_out(
        pipeline_factory=lambda: create_pipeline("logistic_regression", config),
        x=x,
        y=y,
        sites=sites,
        output_dir=tmp_path,
        threshold=0.5,
    )

    assert len(df_loso) == 4  # 2 sites + 1 macro-average row + 1 pooled row
    assert "specificity" in df_loso.columns
    assert (tmp_path / "leave_one_site_out.csv").exists()
    assert (tmp_path / "leave_one_site_out.md").exists()


def test_plot_comparison_table(tmp_path: Path) -> None:
    """plot_comparison_table writes markdown and csv tables."""
    cv_results = {
        "lr": {
            "accuracy_mean": 0.85,
            "accuracy_std": 0.02,
            "recall_mean": 0.80,
            "recall_std": 0.03,
        },
        "rf": {
            "accuracy_mean": 0.82,
            "accuracy_std": 0.04,
            "recall_mean": 0.76,
            "recall_std": 0.05,
        },
    }
    plot_comparison_table(cv_results, tmp_path)
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "model_comparison.md").exists()


def test_evaluation_plots(sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path) -> None:
    """plot_roc_curve, plot_pr_curve, confusion matrix and calibration curve save files."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    pipelines = {"lr": pipeline}
    plot_roc_curve(pipelines, x, y, tmp_path)
    assert (tmp_path / "roc_curve.png").exists()

    plot_pr_curve(pipelines, x, y, tmp_path)
    assert (tmp_path / "pr_curve.png").exists()

    plot_confusion_matrix(pipeline, x, y, 0.5, tmp_path)
    assert (tmp_path / "confusion_matrix.png").exists()

    plot_calibration_curve(pipeline, x, y, tmp_path)
    assert (tmp_path / "calibration_curve.png").exists()


def test_run_feature_ablation(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """run_feature_ablation generates ablation report and csv."""
    from heart_disease.evaluate import run_feature_ablation
    from heart_disease.features import build_preprocessor

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]
    sites = pd.Series(["Cleveland"] * 25 + ["Hungary"] * 25, name="dataset")

    test_config = config.model_copy(deep=True)
    test_config.training.cv_folds = 2
    test_config.training.cv_repeats = 2

    def builder(nf: list[str], cf: list[str]) -> Any:
        return create_pipeline(
            "logistic_regression",
            test_config,
            preprocessor=build_preprocessor(test_config, nf, cf),
        )

    df_ablation = run_feature_ablation(
        pipeline_factory_builder=builder,
        x_train=x,
        y_train=y,
        x_full=x,
        y_full=y,
        sites_full=sites,
        config=test_config,
        output_dir=tmp_path,
        threshold=0.5,
    )

    assert len(df_ablation) == 4
    assert (tmp_path / "ablation.csv").exists()
    assert (tmp_path / "ablation.md").exists()


def test_compare_calibration_methods(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """compare_calibration_methods computes Brier/log-loss for all 3 methods and saves reports."""
    from heart_disease.evaluate import compare_calibration_methods

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    test_config = config.model_copy(deep=True)
    test_config.training.cv_folds = 2

    pipe = create_pipeline("logistic_regression", test_config)
    selected, rationale, metrics, oof_probs = compare_calibration_methods(
        pipe, x, y, test_config, tmp_path
    )

    assert selected in ("none", "sigmoid", "isotonic")
    assert isinstance(rationale, str)
    assert "none" in metrics
    assert "sigmoid" in metrics
    assert "isotonic" in metrics
    assert (tmp_path / "calibration_comparison.png").exists()
    assert (tmp_path / "calibration_comparison.md").exists()


def test_generate_site_confounding_report(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """generate_site_confounding_report generates markdown and csv reports."""
    from heart_disease.evaluate import generate_site_confounding_report

    raw_df = sample_dataframe.copy()
    raw_df["dataset"] = ["Cleveland"] * 25 + ["Hungary"] * 25
    df_demo, df_miss = generate_site_confounding_report(raw_df, config, tmp_path)

    assert len(df_demo) == 3  # 2 sites + Total row
    assert len(df_miss) >= 8
    assert (tmp_path / "site_confounding.md").exists()
    assert (tmp_path / "site_confounding.csv").exists()


def test_evaluate_subgroups_by_sex(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path
) -> None:
    """evaluate_subgroups_by_sex calculates sex subgroup metrics and flags small cohorts."""
    from heart_disease.evaluate import evaluate_subgroups_by_sex

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]
    pipe = create_pipeline("logistic_regression", config)
    pipe.fit(x, y)

    res = evaluate_subgroups_by_sex(
        pipeline=pipe,
        x_test=x,
        y_test=y,
        threshold=0.5,
        output_dir=tmp_path,
        n_bootstraps=50,
        random_state=42,
    )

    assert "test_set" in res
    assert "female" in res["test_set"]
    assert "male" in res["test_set"]
    assert "recall" in res["test_set"]["female"]
    assert "specificity" in res["test_set"]["female"]
    assert "confidence_status" in res["test_set"]["female"]
    assert (tmp_path / "subgroup_analysis.csv").exists()
    assert (tmp_path / "subgroup_analysis.md").exists()
