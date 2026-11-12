"""Tests for SHAP explanation functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heart_disease.config import Config
from heart_disease.explain import (
    compute_shap_values,
    compute_single_shap,
    plot_shap_summary,
)
from heart_disease.models import create_pipeline


def test_compute_shap_values(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """compute_shap_values computes explanations for a batch of samples."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    shap_vals, feature_names = compute_shap_values(pipeline, x.iloc[:5], config)
    assert isinstance(shap_vals, np.ndarray)
    assert len(shap_vals) == 5
    assert len(feature_names) > 0


def test_compute_single_shap(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """compute_single_shap returns 1D array for a single sample."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    single_row = x.iloc[[0]]
    sv, feat_names = compute_single_shap(pipeline, single_row, config)

    assert sv.ndim == 1
    assert len(sv) == len(feat_names)


def test_plot_shap_summary(sample_dataframe: pd.DataFrame, config: Config, tmp_path: Path) -> None:
    """plot_shap_summary generates a PNG image file."""
    x = sample_dataframe.drop(columns=["num"])
    y = sample_dataframe["num"]

    pipeline = create_pipeline("logistic_regression", config)
    pipeline.fit(x, y)

    plot_shap_summary(pipeline, x, config, tmp_path)
    assert (tmp_path / "shap_summary.png").exists()
