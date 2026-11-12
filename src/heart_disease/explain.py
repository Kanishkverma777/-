"""SHAP explanation helpers for heart disease models.

Supports ``TreeExplainer`` (RF, XGBoost) and ``LinearExplainer`` (LR),
with a ``KernelExplainer`` fallback for unknown model types.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from heart_disease.config import Config, get_project_root, load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level cache for background reference
_cached_background: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_shap_background(config: Config | None = None) -> np.ndarray | None:
    """Load cached training background data for SHAP explanations.

    Args:
        config: Project configuration. If None, loads default config.

    Returns:
        Transformed background matrix, or None if file not found.
    """
    global _cached_background  # noqa: PLW0603
    if _cached_background is not None:
        return _cached_background

    if config is None:
        config = load_config()

    root = get_project_root()
    bg_path = root / config.paths.models_dir / "shap_background.joblib"
    if bg_path.exists():
        try:
            _cached_background = joblib.load(bg_path)
            return _cached_background
        except Exception:
            logger.warning("Could not load SHAP background from %s", bg_path)

    return None


def compute_shap_values(
    pipeline: Any,
    x: pd.DataFrame,
    config: Config,
    background_x: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for a batch of observations or a single patient.

    Args:
        pipeline: Trained sklearn ``Pipeline`` (preprocessor + classifier).
        x: Feature DataFrame (one or multiple rows).
        config: Project configuration.
        background_x: Optional custom background dataset or transformed matrix.
            If None and len(x) == 1, loads cached training background from
            models/shap_background.joblib to provide population-relative attributions.

    Returns:
        Tuple of ``(shap_values_2d, feature_names)``.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    x_transformed = preprocessor.transform(x)
    feature_names = _get_feature_names(preprocessor)

    model_type = type(classifier).__name__

    # Determine background reference:
    # A single patient compared only against themselves results in identically zero
    # attributions because E[f(x)] = f(x). For single-sample inference, we require
    # a population background reference.
    bg_data: np.ndarray | None = None
    if background_x is not None:
        if isinstance(background_x, pd.DataFrame):
            bg_data = preprocessor.transform(background_x)
        else:
            bg_data = np.asarray(background_x)
    elif len(x) == 1:
        bg_data = load_shap_background(config)
        if bg_data is None:
            # Fallback to zero-baseline (population medians under RobustScaler)
            bg_data = np.zeros((2, x_transformed.shape[1]), dtype=float)

    if bg_data is None:
        bg_data = x_transformed

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(x_transformed)
        # Binary classification → take class-1 values
        if isinstance(sv, list):
            sv = sv[1]
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(classifier, bg_data)
        sv = explainer.shap_values(x_transformed)
    else:
        logger.warning("Using KernelExplainer for %s (slow)", model_type)
        background = shap.sample(bg_data, min(100, len(bg_data)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        sv = explainer.shap_values(x_transformed)[1]

    return np.asarray(sv), feature_names


def compute_single_shap(
    pipeline: Any,
    x_single: pd.DataFrame,
    config: Config,
    background_x: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for a **single** prediction.

    Args:
        pipeline: Trained sklearn Pipeline.
        x_single: Single-row DataFrame.
        config: Project configuration.
        background_x: Optional background dataset or matrix.

    Returns:
        Tuple of ``(shap_values_1d, feature_names)``.
    """
    sv, feature_names = compute_shap_values(pipeline, x_single, config, background_x=background_x)
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
