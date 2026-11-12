"""Unit tests for the LLM clinical summary service and component."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from heart_disease.config import load_config
from heart_disease.llm import (
    format_patient_summary_text,
    generate_clinical_summary,
    get_groq_api_key,
)


@pytest.fixture
def sample_patient() -> dict[str, Any]:
    return {
        "age": 55,
        "sex": 0,
        "cp": 1,
        "trestbps": 130.0,
        "chol": 245.0,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150.0,
        "exang": 1,
        "oldpeak": 1.0,
    }


@pytest.fixture
def sample_prediction() -> dict[str, Any]:
    return {
        "probability": 0.418,
        "risk_level": "Moderate",
        "threshold": 0.360,
        "prediction": 1,
    }


@pytest.fixture
def sample_top_features() -> list[tuple[str, float]]:
    return [
        ("Exercise Angina: Yes", 0.52),
        ("Chest Pain: Atypical", 0.18),
        ("Sex: Male", -0.85),
    ]


def test_get_groq_api_key_from_env(sample_patient: dict[str, Any]) -> None:
    config = load_config()
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_env_key_123"}):
        key = get_groq_api_key(config)
        assert key == "test_env_key_123"


def test_format_patient_summary_text(sample_patient: dict[str, Any]) -> None:
    text = format_patient_summary_text(sample_patient)
    assert "Age: 55 years old" in text
    assert "Sex: Female" in text
    assert "Typical angina" in text
    assert "Resting Blood Pressure: 130.0 mm Hg" in text
    assert "Mild ST-T electrical changes" in text
    assert "chest discomfort occurred during physical activity" in text


def test_generate_clinical_summary_success(
    sample_patient: dict[str, Any],
    sample_prediction: dict[str, Any],
    sample_top_features: list[tuple[str, float]],
) -> None:
    config = load_config()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": "### 1. Clinical Assessment\nPatient has moderate risk CAD."}}
        ]
    }

    with (
        patch("requests.post", return_value=mock_resp),
        patch("heart_disease.llm.get_groq_api_key", return_value="dummy_key"),
    ):
        summary = generate_clinical_summary(
            sample_patient, sample_prediction, sample_top_features, config
        )
        assert "Clinical Assessment" in summary
        assert "moderate risk" in summary


def test_generate_clinical_summary_api_error(
    sample_patient: dict[str, Any],
    sample_prediction: dict[str, Any],
    sample_top_features: list[tuple[str, float]],
) -> None:
    config = load_config()

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Invalid API Key"

    with (
        patch("requests.post", return_value=mock_resp),
        patch("heart_disease.llm.get_groq_api_key", return_value="dummy_key"),
    ):
        summary = generate_clinical_summary(
            sample_patient, sample_prediction, sample_top_features, config
        )
        assert "AI Summary Generation Failed" in summary
        assert "401" in summary


def test_generate_clinical_summary_timeout(
    sample_patient: dict[str, Any],
    sample_prediction: dict[str, Any],
    sample_top_features: list[tuple[str, float]],
) -> None:
    config = load_config()

    with (
        patch("requests.post", side_effect=requests.Timeout),
        patch("heart_disease.llm.get_groq_api_key", return_value="dummy_key"),
    ):
        summary = generate_clinical_summary(
            sample_patient, sample_prediction, sample_top_features, config
        )
        assert "Request Timed Out" in summary


def test_generate_clinical_summary_missing_key(
    sample_patient: dict[str, Any],
    sample_prediction: dict[str, Any],
    sample_top_features: list[tuple[str, float]],
) -> None:
    config = load_config()

    with patch("heart_disease.llm.get_groq_api_key", return_value=""):
        summary = generate_clinical_summary(
            sample_patient, sample_prediction, sample_top_features, config
        )
        assert "Groq API Key Not Configured" in summary
