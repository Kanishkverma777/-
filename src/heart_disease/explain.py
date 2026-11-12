"""SHAP explanation helpers for heart disease models.

Supports ``TreeExplainer`` (RF, XGBoost) and ``LinearExplainer`` (LR),
with a ``KernelExplainer`` fallback for unknown model types.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from heart_disease.config import Config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_shap_values(
    pipeline: Any,
    x: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for a batch of observations.

    Args:
        pipeline: Trained sklearn ``Pipeline`` (preprocessor + classifier).
        x: Feature DataFrame (multiple rows).
        config: Project configuration.

    Returns:
        Tuple of ``(shap_values_2d, feature_names)``.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    x_transformed = preprocessor.transform(x)
    feature_names = _get_feature_names(preprocessor)

    model_type = type(classifier).__name__

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(x_transformed)
        # Binary classification → take class-1 values
        if isinstance(sv, list):
            sv = sv[1]
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(classifier, x_transformed)
        sv = explainer.shap_values(x_transformed)
    else:
        logger.warning("Using KernelExplainer for %s (slow)", model_type)
        background = shap.sample(x_transformed, min(100, len(x_transformed)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        sv = explainer.shap_values(x_transformed)[1]

    return np.asarray(sv), feature_names


def compute_single_shap(
    pipeline: Any,
    x_single: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for a **single** prediction.

    Args:
        pipeline: Trained sklearn Pipeline.
        x_single: Single-row DataFrame.
        config: Project configuration.

    Returns:
        Tuple of ``(shap_values_1d, feature_names)``.
    """
    sv, feature_names = compute_shap_values(pipeline, x_single, config)
    return sv.flatten(), feature_names


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_shap_summary(
    pipeline: Any,
    x: pd.DataFrame,
    config: Config,
    output_dir: Path,
    max_display: int = 15,
) -> None:
    """Generate and save a SHAP bee-swarm / summary plot.

    Args:
        pipeline: Trained pipeline.
        x: Training features.
        config: Project configuration.
        output_dir: Directory to save the plot.
        max_display: Maximum number of features to display.
    """
    shap_values, feature_names = compute_shap_values(pipeline, x, config)

    preprocessor = pipeline.named_steps["preprocessor"]
    x_transformed = preprocessor.transform(x)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", output_dir / "shap_summary.png")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _get_feature_names(preprocessor: Any) -> list[str]:
    """Extract feature names from a fitted ``ColumnTransformer``."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        logger.warning("Could not get feature names from preprocessor")
        n: int = getattr(preprocessor, "n_features_in_", 0)
        return [f"feature_{i}" for i in range(n)]
