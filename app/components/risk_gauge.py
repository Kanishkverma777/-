"""Risk gauge component for the Streamlit app."""

from __future__ import annotations

import streamlit as st

# Colour / emoji mapping
RISK_COLORS: dict[str, str] = {
    "Low": "🟢",
    "Moderate": "🟡",
    "Elevated": "🟠",
    "High": "🔴",
}

RISK_MESSAGES: dict[str, str] = {
    "Low": ("The model suggests a **low** probability of heart disease."),
    "Moderate": (
        "The model suggests a **moderate** probability of heart disease. "
        "Consider consulting a healthcare provider."
    ),
    "Elevated": (
        "The model suggests an **elevated** probability of heart disease. "
        "A medical evaluation is recommended."
    ),
    "High": (
        "The model suggests a **high** probability of heart disease. "
        "Please consult a healthcare provider promptly."
    ),
}


def get_risk_color(risk_level: str) -> str:
    """Return the emoji colour indicator for a risk level.

    Args:
        risk_level: One of ``'Low'``, ``'Moderate'``, ``'Elevated'``, ``'High'``.

    Returns:
        Emoji string.
    """
    return RISK_COLORS.get(risk_level, "⚪")


def get_risk_message(risk_level: str) -> str:
    """Return the descriptive message for a risk level.

    Args:
        risk_level: One of ``'Low'``, ``'Moderate'``, ``'Elevated'``, ``'High'``.

    Returns:
        Human-readable message.
    """
    return RISK_MESSAGES.get(risk_level, "Unable to determine risk level.")


def render_risk_gauge(
    probability: float,
    risk_level: str,
    threshold: float,
    display_probability: str | None = None,
) -> None:
    """Render the risk probability display.

    Args:
        probability: Predicted probability (0 to 1).
        risk_level: Risk-level label.
        threshold: Decision threshold used by the model.
        display_probability: Optional pre-formatted probability string (e.g. '>99%' or '<1%').
    """
    color = get_risk_color(risk_level)

    st.subheader("Prediction Results")

    # Format probability display if not pre-computed
    if display_probability is None:
        if probability > 0.99:
            disp_val = ">99%"
        elif probability < 0.01:
            disp_val = "<1%"
        else:
            disp_val = f"{probability:.1%}"
    else:
        disp_val = display_probability

    # Large probability metric (shows '>99%' or '<1%' at extremes)
    st.metric(label="Heart Disease Probability", value=disp_val)

    # Risk-level badge
    st.markdown(f"### {color} Risk Level: **{risk_level}**")

    # Progress bar (clipped to [0.01, 0.99] to prevent visual edge pinning)
    clipped_bar_val = max(0.01, min(probability, 0.99))
    st.progress(clipped_bar_val)

    # Descriptive message
    st.info(get_risk_message(risk_level))

    # Threshold footnote
    st.caption(f"Decision threshold: {threshold:.3f} (loaded from `models/metrics.json`)")
