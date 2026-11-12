"""Prediction module: load pipeline → validate input → return probability.

This module is the **single entry point** for inference, shared by both
the training CLI and the Streamlit app.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from heart_disease.config import Config, get_project_root, load_config
from heart_disease.schema import PatientInput

logger = logging.getLogger(__name__)

# Module-level cache (cleared between tests when needed)
_cached_pipeline: Any | None = None
_cached_config: Config | None = None


def _train_fallback_pipeline(config: Config) -> Any:
    """Train a fresh pipeline in-memory from the tracked clean dataset.

    Used when models/pipeline.joblib is missing or incompatible with the runtime
    environment (e.g. cross-version unpickling on Streamlit Cloud).
    """
    import warnings

    from heart_disease.data import clean_data, load_raw_data
    from heart_disease.models import create_pipeline

    logger.warning("Training fresh fallback pipeline from local dataset...")
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    feature_cols = list(config.features.numeric) + list(config.features.categorical)
    x_train = clean_df[feature_cols]
    y_train = clean_df[config.data.target_column]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*(divide by zero|overflow|invalid value) encountered in matmul.*",
        )
        pipe = create_pipeline("logistic_regression", config)
        pipe.fit(x_train, y_train)

    return pipe


def load_pipeline(config: Config | None = None) -> Any:
    """Load the trained pipeline from disk (cached after first call).

    Falls back automatically to an in-memory fit if pipeline.joblib is missing
    or was pickled under an incompatible scikit-learn / Python version.

    Args:
        config: Project configuration. If ``None``, loads the default.

    Returns:
        Trained sklearn Pipeline (or ``CalibratedClassifierCV``).
    """
    global _cached_pipeline, _cached_config  # noqa: PLW0603

    if _cached_pipeline is not None:
        return _cached_pipeline

    if config is None:
        config = load_config()

    root = get_project_root()
    pipeline_path = root / config.paths.models_dir / "pipeline.joblib"

    if pipeline_path.exists():
        try:
            logger.info("Loading pipeline from %s", pipeline_path)
            loaded = joblib.load(pipeline_path)
            _cached_pipeline = loaded
            _cached_config = config
            return _cached_pipeline
        except Exception as exc:
            logger.warning(
                "Failed to unpickle pipeline from %s (%s). Falling back to in-memory fit.",
                pipeline_path,
                exc,
            )

    _cached_pipeline = _train_fallback_pipeline(config)
    _cached_config = config
    return _cached_pipeline


def validate_input(data: dict[str, Any]) -> PatientInput:
    """Validate patient input data against the schema.

    Args:
        data: Raw input dictionary.

    Returns:
        Validated ``PatientInput``.

    Raises:
        pydantic.ValidationError: If input violates schema constraints.
    """
    return PatientInput.model_validate(data)


def predict(patient: PatientInput, config: Config | None = None) -> dict[str, Any]:
    """Make a prediction for a **validated** patient input.

    Args:
        patient: Validated patient input.
        config: Project configuration.

    Returns:
        Dict with keys ``probability``, ``prediction``, ``risk_level``,
        ``threshold``.
    """
    if config is None:
        config = load_config()

    pipeline = load_pipeline(config)

    # Load persisted decision threshold
    threshold = _load_threshold(config)

    # Build single-row DataFrame from validated input
    feature_dict = patient.to_feature_dict()
    x = pd.DataFrame([feature_dict])

    # Cast numeric and categorical columns to match expected training dtypes
    for col in config.features.numeric:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)

    for col in config.features.categorical:
        if col in x.columns:
            x[col] = x[col].astype("object")

    # Predict with version-resilient fallback
    try:
        raw_prob = pipeline.predict_proba(x)[:, 1][0]
    except (AttributeError, ValueError, TypeError) as exc:
        logger.warning(
            "Pipeline predict_proba failed with %s. Falling back to fresh in-memory pipeline.",
            exc,
        )
        global _cached_pipeline  # noqa: PLW0603
        _cached_pipeline = _train_fallback_pipeline(config)
        pipeline = _cached_pipeline
        raw_prob = pipeline.predict_proba(x)[:, 1][0]

    probability = float(raw_prob)
    prediction = int(probability >= threshold)
    risk_level = _get_risk_level(probability, config)

    min_prob = config.app.display_min_probability
    max_prob = config.app.display_max_probability
    clipped_probability = float(np.clip(probability, min_prob, max_prob))
    display_probability = format_display_probability(probability, min_prob, max_prob)

    logger.info(
        "Prediction: prob=%.4f (disp=%s), pred=%d, risk=%s (threshold=%.3f)",
        probability,
        display_probability,
        prediction,
        risk_level,
        threshold,
    )

    return {
        "probability": probability,
        "display_probability": display_probability,
        "clipped_probability": clipped_probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": threshold,
    }


def format_display_probability(
    probability: float,
    min_prob: float = 0.01,
    max_prob: float = 0.99,
) -> str:
    """Format probability for display, clipping extremes to '<1%' or '>99%'.

    Args:
        probability: Raw probability between 0 and 1.
        min_prob: Lower threshold for displaying '<min%'.
        max_prob: Upper threshold for displaying '>max%'.

    Returns:
        Formatted string (e.g. '>99%', '<1%', or '45.2%').
    """
    if probability > max_prob:
        return f">{int(round(max_prob * 100))}%"
    if probability < min_prob:
        return f"<{int(round(min_prob * 100))}%"
    return f"{probability:.1%}"


def predict_from_dict(data: dict[str, Any], config: Config | None = None) -> dict[str, Any]:
    """Validate and predict from a raw dictionary (convenience wrapper).

    Args:
        data: Raw input dictionary.
        config: Project configuration.

    Returns:
        Prediction result dictionary.

    Raises:
        pydantic.ValidationError: If input is invalid.
    """
    patient = validate_input(data)
    return predict(patient, config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def get_model_decision_threshold(config: Config | None = None) -> float:
    """Return the active decision threshold loaded dynamically from metrics.json."""
    if config is None:
        config = load_config()
    return _load_threshold(config)


def _load_threshold(config: Config) -> float:
    """Load the decision threshold from ``metrics.json`` or config default."""
    root = get_project_root()
    metrics_path = root / config.paths.models_dir / "metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            metadata = json.load(f)
        if "decision_threshold" in metadata:
            return float(metadata["decision_threshold"])
        if "selected_model" in metadata and "decision_threshold" in metadata["selected_model"]:
            return float(metadata["selected_model"]["decision_threshold"])

    return config.training.decision_threshold


def _get_risk_level(probability: float, config: Config) -> str:
    """Map a probability to a human-readable risk-level label."""
    thresholds = config.app.risk_thresholds

    if probability < thresholds.get("low", 0.20):
        return "Low"
    elif probability < thresholds.get("moderate", 0.50):
        return "Moderate"
    elif probability < thresholds.get("elevated", 0.75):
        return "Elevated"
    else:
        return "High"
