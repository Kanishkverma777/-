"""Patient input form component for the Streamlit app."""

from __future__ import annotations

from typing import Any

import streamlit as st

from heart_disease.schema import CATEGORICAL_LABELS


def render_input_form() -> dict[str, Any]:
    """Render the patient clinical-data form in the sidebar.

    Returns:
        Dictionary of feature-name → value, ready for validation.
    """
    data: dict[str, Any] = {}

    # --- Age ---
    data["age"] = st.sidebar.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=55,
        step=1,
    )

    # --- Sex ---
    sex_labels = CATEGORICAL_LABELS["sex"]
    data["sex"] = st.sidebar.selectbox(
        "Sex",
        options=list(sex_labels.keys()),
        format_func=lambda x: sex_labels[x],
    )

    # --- Chest Pain Type ---
    cp_labels = CATEGORICAL_LABELS["cp"]
    data["cp"] = st.sidebar.selectbox(
        "Chest Pain Type",
        options=list(cp_labels.keys()),
        format_func=lambda x: cp_labels[x],
    )

    # --- Resting Blood Pressure ---
    trestbps_unknown = st.sidebar.checkbox(
        "Blood pressure unknown",
        value=False,
    )
    if trestbps_unknown:
        data["trestbps"] = None
    else:
        data["trestbps"] = st.sidebar.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=130,
            step=1,
        )

    # --- Cholesterol ---
    chol_unknown = st.sidebar.checkbox(
        "Cholesterol unknown",
        value=False,
    )
    if chol_unknown:
        data["chol"] = None
    else:
        data["chol"] = st.sidebar.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=50,
            max_value=600,
            value=245,
            step=1,
        )

    # --- Fasting Blood Sugar ---
    fbs_labels = {-1: "Unknown / Not Tested", 0: "≤ 120 mg/dl (0)", 1: "> 120 mg/dl (1)"}
    fbs_val = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=list(fbs_labels.keys()),
        index=1,  # Default to 0 (<= 120 mg/dl)
        format_func=lambda x: fbs_labels[x],
    )
    data["fbs"] = None if fbs_val is None or fbs_val == -1 else int(fbs_val)

    # --- Resting ECG ---
    restecg_labels = {
        -1: "Unknown / Not Recorded",
        0: "Normal (0)",
        1: "ST-T Abnormality (1)",
        2: "LV Hypertrophy (2)",
    }
    restecg_val = st.sidebar.selectbox(
        "Resting ECG Results",
        options=list(restecg_labels.keys()),
        index=1,  # Default to Normal (0)
        format_func=lambda x: restecg_labels[x],
    )
    data["restecg"] = None if restecg_val is None or restecg_val == -1 else int(restecg_val)

    # --- Max Heart Rate ---
    thalach_unknown = st.sidebar.checkbox(
        "Max heart rate unknown",
        value=False,
    )
    if thalach_unknown:
        data["thalach"] = None
    else:
        data["thalach"] = st.sidebar.number_input(
            "Max Heart Rate Achieved",
            min_value=50,
            max_value=250,
            value=150,
            step=1,
        )

    # --- Exercise-Induced Angina ---
    exang_labels = {-1: "Unknown / Not Tested", 0: "No (0)", 1: "Yes (1)"}
    exang_val = st.sidebar.selectbox(
        "Exercise-Induced Angina",
        options=list(exang_labels.keys()),
        index=1,  # Default to No (0)
        format_func=lambda x: exang_labels[x],
    )
    data["exang"] = None if exang_val is None or exang_val == -1 else int(exang_val)

    # --- ST Depression ---
    oldpeak_unknown = st.sidebar.checkbox(
        "ST depression unknown",
        value=False,
    )
    if oldpeak_unknown:
        data["oldpeak"] = None
    else:
        data["oldpeak"] = st.sidebar.number_input(
            "ST Depression (oldpeak)",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    return data
