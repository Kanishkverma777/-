"""Tests for the prediction module."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from heart_disease.config import Config
from heart_disease.predict import (
    _get_risk_level,
    _load_threshold,
    load_pipeline,
    predict,
    predict_from_dict,
    validate_input,
)
from heart_disease.schema import PatientInput


def test_validate_input_valid(sample_patient_data: dict[str, Any]) -> None:
    """Valid data passes validation."""
    patient = validate_input(sample_patient_data)
    assert isinstance(patient, PatientInput)


def test_validate_input_invalid_type() -> None:
    """Non-numeric age raises ``ValidationError``."""
    with pytest.raises(ValidationError):
        validate_input(
            {
                "age": "old",
                "sex": 1,
                "cp": 1,
                "trestbps": 130,
                "chol": 245,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            }
        )


def test_validate_input_missing_field() -> None:
    """Missing required fields raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        validate_input({"age": 55})


def test_patient_output_has_all_features(sample_patient_data: dict[str, Any]) -> None:
    """Validated patient dict has exactly the active feature keys."""
    from heart_disease.schema import ALL_FEATURES

    patient = validate_input(sample_patient_data)
    fd = patient.to_feature_dict()
    assert len(fd) == len(ALL_FEATURES)


def test_predict_without_optional_features(config: Config) -> None:
    """predict succeeds when ca, thal, slope, chol, and fbs are absent from input."""
    data = {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.5,
    }
    res = predict_from_dict(data, config)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)


def test_load_pipeline_success(config: Config) -> None:
    """Trained pipeline loads and can be retrieved."""
    pipeline = load_pipeline(config)
    assert pipeline is not None


def test_predict_output_structure_and_range(
    sample_patient_data: dict[str, Any], config: Config
) -> None:
    """Predict returns valid probability [0, 1] and prediction {0, 1}."""
    patient = validate_input(sample_patient_data)
    res = predict(patient, config)

    assert "probability" in res
    assert "prediction" in res
    assert "risk_level" in res
    assert "threshold" in res

    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)
    assert res["risk_level"] in ("Low", "Moderate", "Elevated", "High")
    assert 0.0 < res["threshold"] < 1.0


def test_predict_from_dict(sample_patient_data: dict[str, Any], config: Config) -> None:
    """predict_from_dict validates and predicts end-to-end."""
    res = predict_from_dict(sample_patient_data, config)
    assert 0.0 <= res["probability"] <= 1.0


def test_predict_with_optional_unknown_features(
    sample_patient_data: dict[str, Any], config: Config
) -> None:
    """predict succeeds when optional features (ca, thal, slope, etc.) are None (unknown)."""
    data_with_unknowns = sample_patient_data.copy()
    data_with_unknowns["ca"] = None
    data_with_unknowns["thal"] = None
    data_with_unknowns["slope"] = None
    data_with_unknowns["chol"] = None
    data_with_unknowns["trestbps"] = None

    res = predict_from_dict(data_with_unknowns, config)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)


def test_get_risk_level(config: Config) -> None:
    """Risk levels categorize probabilities according to configured thresholds."""
    assert _get_risk_level(0.10, config) == "Low"
    assert _get_risk_level(0.35, config) == "Moderate"
    assert _get_risk_level(0.65, config) == "Elevated"
    assert _get_risk_level(0.85, config) == "High"


def test_load_threshold(config: Config) -> None:
    """Threshold is loaded from metrics or config fallback."""
    from heart_disease.predict import get_model_decision_threshold

    thresh = _load_threshold(config)
    assert isinstance(thresh, float)
    assert 0.0 < thresh < 1.0
    assert get_model_decision_threshold(config) == thresh


def test_probability_strictly_between_zero_and_one_for_extreme_inputs(config: Config) -> None:
    """Predicted probability is strictly between 0 and 1 for extreme valid inputs."""
    extreme_low = {
        "age": 29,
        "sex": 0,
        "cp": 1,
        "trestbps": 94,
        "chol": 126,
        "fbs": 0,
        "restecg": 0,
        "thalach": 200,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }
    extreme_high = {
        "age": 77,
        "sex": 1,
        "cp": 4,
        "trestbps": 200,
        "chol": 564,
        "fbs": 1,
        "restecg": 2,
        "thalach": 71,
        "exang": 1,
        "oldpeak": 6.2,
        "slope": 2,
        "ca": 3,
        "thal": 7,
    }

    res_low = predict_from_dict(extreme_low, config)
    res_high = predict_from_dict(extreme_high, config)

    # Strictly between 0 and 1 (never exactly 0.0 or 1.0)
    assert 0.0 < res_low["probability"] < 1.0
    assert 0.0 < res_high["probability"] < 1.0

    # Display probability clipping checks
    assert res_low["display_probability"] in ("<1%", f"{res_low['probability']:.1%}")
    assert res_high["display_probability"] in (">99%", f"{res_high['probability']:.1%}")


def test_monotonicity_raising_oldpeak_ca_or_exang_never_lowers_probability(
    sample_patient_data: dict[str, Any], config: Config
) -> None:
    """Raising oldpeak, ca, or exang never lowers the predicted probability."""
    base = sample_patient_data.copy()
    base.update({"oldpeak": 1.0, "ca": 0, "exang": 0})
    p_base = predict_from_dict(base, config)["probability"]

    # 1. Raising oldpeak (1.0 -> 2.0 -> 4.0)
    p_oldpeak_2 = predict_from_dict({**base, "oldpeak": 2.0}, config)["probability"]
    p_oldpeak_4 = predict_from_dict({**base, "oldpeak": 4.0}, config)["probability"]
    assert p_oldpeak_2 >= p_base - 1e-6
    assert p_oldpeak_4 >= p_oldpeak_2 - 1e-6

    # 2. Raising ca (0 -> 1 -> 2 -> 3)
    p_ca_1 = predict_from_dict({**base, "ca": 1}, config)["probability"]
    p_ca_2 = predict_from_dict({**base, "ca": 2}, config)["probability"]
    p_ca_3 = predict_from_dict({**base, "ca": 3}, config)["probability"]
    assert p_ca_1 >= p_base - 1e-6
    assert p_ca_2 >= p_ca_1 - 1e-6
    assert p_ca_3 >= p_ca_2 - 1e-6

    # 3. Raising exang (0 -> 1)
    p_exang_1 = predict_from_dict({**base, "exang": 1}, config)["probability"]
    assert p_exang_1 >= p_base - 1e-6


def test_changing_cp_from_1_to_4_changes_output(
    sample_patient_data: dict[str, Any], config: Config
) -> None:
    """Changing chest pain type (cp) from 1 to 4 changes the model output."""
    patient_cp1 = sample_patient_data.copy()
    patient_cp1["cp"] = 1
    patient_cp4 = sample_patient_data.copy()
    patient_cp4["cp"] = 4

    res_cp1 = predict_from_dict(patient_cp1, config)
    res_cp4 = predict_from_dict(patient_cp4, config)

    # Output probabilities must differ significantly
    assert abs(res_cp4["probability"] - res_cp1["probability"]) > 0.01


def test_predict_with_screenshot_unknown_inputs(config: Config) -> None:
    """Predict succeeds when numeric features (trestbps, thalach, oldpeak) are None."""
    data = {
        "age": 55,
        "sex": 0,
        "cp": 1,
        "trestbps": None,
        "chol": None,
        "fbs": 0,
        "restecg": 1,
        "thalach": None,
        "exang": 1,
        "oldpeak": None,
        "slope": None,
        "ca": None,
        "thal": None,
    }
    patient = validate_input(data)
    res = predict(patient, config)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)
    assert res["risk_level"] in ("Low", "Moderate", "Elevated", "High")


def test_train_fallback_pipeline_resilience(config: Config) -> None:
    """In-memory fallback training succeeds and produces a working pipeline."""
    from heart_disease.predict import _train_fallback_pipeline

    pipe = _train_fallback_pipeline(config)
    assert pipe is not None
    assert hasattr(pipe, "predict_proba")
