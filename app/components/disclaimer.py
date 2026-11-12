"""Disclaimer component for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_disclaimer() -> None:
    """Render the medical disclaimer banner.

    Always visible at the top of the page.
    """
    st.warning(
        "⚠️ **DISCLAIMER — Educational Use Only**\n\n"
        "This tool is for **educational and research purposes only**. "
        "It is **NOT** a substitute for professional medical advice, diagnosis, "
        "or treatment. Always seek the advice of a qualified healthcare provider "
        "with any questions regarding a medical condition. Never disregard "
        "professional medical advice or delay seeking it because of information "
        "from this tool.\n\n"
        "The predictions are based on statistical patterns in historical data "
        "and may not reflect your individual health status."
    )
