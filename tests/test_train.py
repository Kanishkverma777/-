"""Tests for training routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heart_disease.config import Config
from heart_disease.models import create_pipeline
from heart_disease.train import (
    _serialize_params,
    calibrate_model,
    cross_validate_model,
    save_pipeline,
    tune_model,
)


def test_serialize_params() -> None:
    """_serialize_params converts numpy types to standard python primitives."""
    raw = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "str": "saga",
    }
    s = _serialize_params(raw)
    assert s["int"] == 42
    assert isinstance(s["int"], int)
    assert s["float"] == 3.14
    assert isinstance(s["float"], float)
    assert s["array"] == [1, 2, 3]


def test_cross_validate_model(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """cross_validate_model returns mean and std for all target metrics."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    res = cross_validate_model(pipeline, x, y, config)

    assert "accuracy_mean" in res
    assert "recall_mean" in res
    assert "f1_mean" in res
    assert "roc_auc_mean" in res
    assert "pr_auc_mean" in res
    assert "brier_score_mean" in res


def test_select_best_model_rule(config: Config) -> None:
    """select_best_model adheres to ROC-AUC primary, 1-fold std rule, and tie-breakers."""
    from heart_disease.train import select_best_model

    # Scenario 1: Model A clearly beats B and C by > 1 std
    cv_res_clear = {
        "logistic_regression": {"roc_auc_mean": 0.80, "roc_auc_std": 0.02},
        "random_forest": {"roc_auc_mean": 0.90, "roc_auc_std": 0.02},  # Cutoff: 0.88 -> RF only
        "xgboost": {"roc_auc_mean": 0.85, "roc_auc_std": 0.03},
    }
    winner, _ = select_best_model(cv_res_clear, config)
    assert winner == "random_forest"

    # Scenario 2: RF has top mean 0.88 +/- 0.03 (cutoff: 0.85).
    # LR has 0.87 +/- 0.01 (within 1 std, lower std, simpler).
    # Selection rule must tie-break to logistic_regression!
    cv_res_tie = {
        "random_forest": {"roc_auc_mean": 0.88, "roc_auc_std": 0.03},
        "logistic_regression": {"roc_auc_mean": 0.87, "roc_auc_std": 0.01},
        "xgboost": {"roc_auc_mean": 0.80, "roc_auc_std": 0.04},
    }
    winner_tie, rationale = select_best_model(cv_res_tie, config)
    assert winner_tie == "logistic_regression"
    assert "simplicity" in rationale
    assert "lower fold std" in rationale

    # Scenario 3: RF top mean 0.88 +/- 0.02 (cutoff: 0.86).
    # LR has 0.87 +/- 0.03 (within 1 std, higher std, but simpler).
    # Must choose LR for simplicity without falsely claiming lower fold std!
    cv_res_tie_higher_std = {
        "random_forest": {"roc_auc_mean": 0.88, "roc_auc_std": 0.02},
        "logistic_regression": {"roc_auc_mean": 0.87, "roc_auc_std": 0.03},
        "xgboost": {"roc_auc_mean": 0.80, "roc_auc_std": 0.04},
    }
    winner_tie3, rationale3 = select_best_model(cv_res_tie_higher_std, config)
    assert winner_tie3 == "logistic_regression"
    assert "simplicity" in rationale3
    assert "lower fold std" not in rationale3

    # Scenario 4: LR has highest mean 0.88 +/- 0.02 (cutoff: 0.86).
    # RF has 0.87 +/- 0.02, XGB has 0.87 +/- 0.02.
    # LR selected based on simplicity and interpretability when scores within 1 std.
    cv_res_lr_top = {
        "logistic_regression": {"roc_auc_mean": 0.88, "roc_auc_std": 0.02},
        "random_forest": {"roc_auc_mean": 0.87, "roc_auc_std": 0.02},
        "xgboost": {"roc_auc_mean": 0.87, "roc_auc_std": 0.02},
    }
    winner_tie4, rationale4 = select_best_model(cv_res_lr_top, config)
    assert winner_tie4 == "logistic_regression"
    assert "simplicity" in rationale4
    assert "within one std" in rationale4


def test_get_feature_missingness_report(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """get_feature_missingness_report returns structured missingness for all active features."""
    from heart_disease.train import get_feature_missingness_report

    report = get_feature_missingness_report(sample_dataframe, config)
    assert len(report) == len(config.features.numeric) + len(config.features.categorical)
    for entry in report:
        assert "feature" in entry
        assert "missing_count" in entry
        assert "missing_pct" in entry
        assert "imputation_strategy" in entry


def test_calibrate_model(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """calibrate_model returns a fitted estimator (uncalibrated or calibrated)."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    fitted_none = calibrate_model(pipeline, x, y, config, method="none")
    assert hasattr(fitted_none, "predict_proba")
    assert fitted_none.predict_proba(x).shape == (len(x), 2)

    cfg = config.model_copy(deep=True)
    cfg.training.cv_folds = 2
    fitted_sig = calibrate_model(pipeline, x, y, cfg, method="sigmoid")
    assert hasattr(fitted_sig, "predict_proba")
    assert fitted_sig.predict_proba(x).shape == (len(x), 2)


def test_tune_model(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """tune_model executes hyperparameter search and returns tuned pipeline and score."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    cfg = config.model_copy(deep=True)
    cfg.training.n_iter = 1
    cfg.training.cv_folds = 2
    best_pipe, best_params, best_score = tune_model("logistic_regression", x, y, cfg)
    assert best_pipe is not None
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)


def test_save_pipeline(
    sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path, monkeypatch: Any
) -> None:
    """save_pipeline writes pipeline.joblib and comprehensive metrics.json."""
    monkeypatch.setattr("heart_disease.train.get_project_root", lambda: tmp_path)

    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    metadata = {
        "version": "1.0.0",
        "random_seed": 42,
        "dataset_summary": {"raw_row_count": 920},
        "features": {"all": ["age", "sex"]},
        "selected_model": {"name": "logistic_regression", "decision_threshold": 0.35},
        "tuning_details_all_models": {},
        "metrics": {"test_metrics": {"recall": 0.95}},
    }
    save_pipeline(pipeline, metadata, config)

    assert (tmp_path / config.paths.models_dir / "pipeline.joblib").exists()
    assert (tmp_path / config.paths.models_dir / "metrics.json").exists()
