"""Tests for Streamlit app component pure logic (no Streamlit dependency)."""

from __future__ import annotations

from app.components.risk_gauge import get_risk_color, get_risk_message
from app.components.shap_chart import _clean_feature_name


def test_risk_color_mapping() -> None:
    """Known risk levels map to the correct prebuilt library icon."""
    assert get_risk_color("Low") == ":material/check_circle:"
    assert get_risk_color("Moderate") == ":material/info:"
    assert get_risk_color("Elevated") == ":material/warning:"
    assert get_risk_color("High") == ":material/error:"
    assert get_risk_color("Unknown") == ":material/help:"  # fallback


def test_risk_message_content() -> None:
    """Risk messages contain the relevant keyword."""
    assert "low" in get_risk_message("Low").lower()
    assert "moderate" in get_risk_message("Moderate").lower()
    assert "elevated" in get_risk_message("Elevated").lower()
    assert "high" in get_risk_message("High").lower()


def test_clean_feature_name_removes_prefix() -> None:
    """Transformer prefixes are removed and labels are human-readable."""
    assert _clean_feature_name("num__age") == "Age"
    assert _clean_feature_name("cat__cp_1") == "Chest Pain: Typical Angina"
    assert _clean_feature_name("num__thalach") == "Max Heart Rate"
    assert _clean_feature_name("cat__restecg_0.0") == "ECG: Normal"
    assert _clean_feature_name("cat__exang_1.0") == "Exercise Angina: Yes"


def test_clean_feature_name_unknown() -> None:
    """Unknown feature names are title-cased with underscores replaced."""
    assert _clean_feature_name("some_unknown_feature") == "Some Unknown Feature"
