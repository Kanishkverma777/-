"""AI Clinical Summary component for Streamlit using Groq and GPT-OSS 120B."""

from __future__ import annotations

from typing import Any

import streamlit as st

from heart_disease.config import Config
from heart_disease.llm import generate_clinical_summary


def render_ai_summary_section(
    patient_data: dict[str, Any],
    prediction_result: dict[str, Any],
    top_features: list[tuple[str, float]],
    config: Config,
) -> None:
    """Render the AI clinical summary and recommendations in the blank space after the report.

    Args:
        patient_data: Raw input dictionary with patient clinical measurements.
        prediction_result: Dict containing probability, risk_level, threshold, prediction.
        top_features: List of (feature_display_name, shap_value) tuples sorted by impact.
        config: System configuration containing LLM parameters.
    """
    st.divider()

    st.subheader(
        "AI Clinical Summary & Recommendations",
        icon=":material/psychology:",
    )

    # Cache key with prompt version to prevent stale cache across prompt updates
    prob = prediction_result.get("probability", 0.0)
    cache_key = f"ai_summary_v5_plain_{hash(str(patient_data))}_{prob:.4f}"

    if cache_key not in st.session_state:
        with st.spinner(f"{config.llm.model}..."):
            summary_content = generate_clinical_summary(
                patient_data, prediction_result, top_features, config
            )
            st.session_state[cache_key] = summary_content

    summary_text = st.session_state[cache_key]

    with st.container(border=True):
        st.markdown(summary_text)
