"""Pydantic models and feature metadata for heart disease prediction.

This module defines:
- PatientInput: validated patient data with clinically valid ranges.
- Feature metadata: names, types, valid ranges, descriptions.
- Categorical label mappings for human-readable display.

Shared by training code and the Streamlit app (single source of truth).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age in years",
    "sex": "Sex (0 = female, 1 = male)",
    "cp": "Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg on admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0=no, 1=yes)",
    "restecg": "Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    "thalach": "Maximum heart rate achieved during exercise",
    "exang": "Exercise-induced angina (0=no, 1=yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)",
}

NUMERIC_FEATURES: list[str] = ["age", "trestbps", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "restecg", "exang"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OPTIONAL_FEATURES: list[str] = ["ca", "thal", "slope", "chol", "fbs"]
RAW_CLINICAL_FEATURES: list[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------


class PatientInput(BaseModel):
    """Validated patient input for heart disease prediction.

    Core clinical demographics and symptoms (age, sex, chest pain) are required.
    Diagnostic procedures and laboratory tests (ca, thal, slope, etc.) may be
    None when unknown or not performed, and are imputed using population statistics.
    """

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float | None = Field(
        default=None, ge=50, le=250, description="Resting blood pressure (mm Hg)"
    )
    chol: float | None = Field(
        default=None, ge=50, le=600, description="Serum cholesterol (mg/dl)"
    )
    fbs: int | None = Field(
        default=None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl"
    )
    restecg: int | None = Field(default=None, ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float | None = Field(
        default=None, ge=50, le=250, description="Maximum heart rate achieved"
    )
    exang: int | None = Field(default=None, ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float | None = Field(default=None, ge=-5.0, le=10.0, description="ST depression")
    slope: int | None = Field(
        default=None, ge=1, le=3, description="Slope of peak exercise ST segment"
    )
    ca: float | None = Field(default=None, ge=0, le=4, description="Number of major vessels (0-3)")
    thal: int | None = Field(default=None, description="Thalassemia type (3, 6, or 7)")

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, v: int | None) -> int | None:
        """Validate thal is one of the known categories when provided."""
        if v is not None and v not in (3, 6, 7):
            msg = f"thal must be 3, 6, or 7; got {v}"
            raise ValueError(msg)
        return v

    def to_feature_dict(self) -> dict[str, float | int | None]:
        """Convert to a dict suitable for DataFrame creation (active features only)."""
        data = self.model_dump()
        return {k: data[k] for k in ALL_FEATURES if k in data}


# ---------------------------------------------------------------------------
# Human-readable labels for categorical features
# ---------------------------------------------------------------------------

CATEGORICAL_LABELS: dict[str, dict[int, str]] = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},
}
