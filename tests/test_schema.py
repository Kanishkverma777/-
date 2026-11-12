"""Tests for the pydantic schema and validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from heart_disease.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PatientInput,
)


def test_valid_patient_input(sample_patient_data: dict[str, Any]) -> None:
    """Valid patient data is accepted and fields round-trip correctly."""
    patient = PatientInput.model_validate(sample_patient_data)
    assert patient.age == 55.0
    assert patient.sex == 1


def test_patient_to_feature_dict(sample_patient_data: dict[str, Any]) -> None:
    """``to_feature_dict`` returns a dict with all 13 feature keys."""
    patient = PatientInput.model_validate(sample_patient_data)
    fd = patient.to_feature_dict()
    assert set(fd.keys()) == set(ALL_FEATURES)


def test_age_out_of_range() -> None:
    """Age > 120 is rejected."""
    with pytest.raises(ValidationError):
        PatientInput(
            age=200,
            sex=1,
            cp=1,
            trestbps=130,
            chol=245,
            fbs=0,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=1.0,
            slope=2,
            ca=0,
            thal=3,
        )


def test_negative_age() -> None:
    """Negative age is rejected."""
    with pytest.raises(ValidationError):
        PatientInput(
            age=-5,
            sex=1,
            cp=1,
            trestbps=130,
            chol=245,
            fbs=0,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=1.0,
            slope=2,
            ca=0,
            thal=3,
        )


def test_invalid_thal_value() -> None:
    """thal value not in {3, 6, 7} is rejected."""
    with pytest.raises(ValidationError):
        PatientInput(
            age=55,
            sex=1,
            cp=1,
            trestbps=130,
            chol=245,
            fbs=0,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=1.0,
            slope=2,
            ca=0,
            thal=5,
        )


def test_invalid_sex_value() -> None:
    """sex value outside {0, 1} is rejected."""
    with pytest.raises(ValidationError):
        PatientInput(
            age=55,
            sex=2,
            cp=1,
            trestbps=130,
            chol=245,
            fbs=0,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=1.0,
            slope=2,
            ca=0,
            thal=3,
        )


def test_boundary_values_accepted() -> None:
    """Minimum boundary values are accepted."""
    patient = PatientInput(
        age=0,
        sex=0,
        cp=1,
        trestbps=50,
        chol=50,
        fbs=0,
        restecg=0,
        thalach=50,
        exang=0,
        oldpeak=-5.0,
        slope=1,
        ca=0,
        thal=3,
    )
    assert patient.age == 0
    assert patient.trestbps == 50


def test_feature_lists_consistent() -> None:
    """``ALL_FEATURES`` is the disjoint union of numeric + categorical."""
    assert set(ALL_FEATURES) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert len(ALL_FEATURES) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_optional_fields_accept_none() -> None:
    """Optional fields (ca, thal, slope, etc.) accept None."""
    patient = PatientInput(
        age=55,
        sex=1,
        cp=2,
        trestbps=None,
        chol=None,
        fbs=None,
        restecg=None,
        thalach=None,
        exang=None,
        oldpeak=None,
        slope=None,
        ca=None,
        thal=None,
    )
    assert patient.ca is None
    assert patient.thal is None
    assert patient.slope is None
    fd = patient.to_feature_dict()
    assert len(fd) == len(ALL_FEATURES)
    assert "ca" not in fd
    assert "thal" not in fd
    assert "slope" not in fd
    assert "chol" not in fd
    assert "fbs" not in fd
