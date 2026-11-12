"""SHAP explanation chart component for the Streamlit app."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def render_shap_chart(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> None:
    """Render a horizontal bar chart of the top SHAP contributions.

    Args:
        shap_values: SHAP values for a single prediction (1-D).
        feature_names: Names of all transformed features.
        top_k: Number of top features to display.
    """
    st.subheader("🔍 Top Contributing Factors")

    # Sort by |SHAP|
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]

    top_names = [_clean_feature_name(feature_names[i]) for i in top_indices]
    top_values = shap_values[top_indices]

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.5)))

    colors = ["#ff4b4b" if v > 0 else "#0068c9" for v in top_values]

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Feature Contributions")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#ff4b4b", label="Increases risk"),
        Patch(facecolor="#0068c9", label="Decreases risk"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature-name cleanup
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, str] = {
    "age": "Age",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "thalach": "Max Heart Rate",
    "oldpeak": "ST Depression",
    "ca": "Major Vessels (ca)",
    "sex_0": "Sex: Female",
    "sex_1": "Sex: Male",
    "cp_1": "Chest Pain: Typical Angina",
    "cp_2": "Chest Pain: Atypical",
    "cp_3": "Chest Pain: Non-anginal",
    "cp_4": "Chest Pain: Asymptomatic",
    "fbs_0": "Fasting BS: Normal",
    "fbs_1": "Fasting BS: High",
    "restecg_0": "ECG: Normal",
    "restecg_1": "ECG: ST-T Abnormality",
    "restecg_2": "ECG: LV Hypertrophy",
    "exang_0": "Exercise Angina: No",
    "exang_1": "Exercise Angina: Yes",
    "slope_1": "Slope: Upsloping",
    "slope_2": "Slope: Flat",
    "slope_3": "Slope: Downsloping",
    "thal_3": "Thal: Normal",
    "thal_6": "Thal: Fixed Defect",
    "thal_7": "Thal: Reversible Defect",
}


def _clean_feature_name(name: str) -> str:
    """Clean up sklearn ``ColumnTransformer`` output names.

    Args:
        name: Raw feature name (e.g. ``'num__age'``, ``'cat__cp_2'``).

    Returns:
        Cleaned, human-readable label.
    """
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    return _LABEL_MAP.get(name, name.replace("_", " ").title())
