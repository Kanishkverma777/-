"""Streamlit app entry point for Heart Disease Prediction.

This is a **thin** entry point — all logic lives in ``app/components/``
and ``src/heart_disease/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

# Ensure project root is on sys.path so both ``heart_disease`` and ``app``
# packages are importable when running via ``streamlit run app/streamlit_app.py``.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from app.components.disclaimer import render_disclaimer  # noqa: E402
from app.components.input_form import render_input_form  # noqa: E402
from app.components.risk_gauge import render_risk_gauge  # noqa: E402
from app.components.shap_chart import render_shap_chart  # noqa: E402
from heart_disease.config import Config, load_config  # noqa: E402
from heart_disease.explain import compute_single_shap  # noqa: E402
from heart_disease.predict import load_pipeline, predict, validate_input  # noqa: E402

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_config_cached() -> Config:
    """Load and cache the project configuration."""
    return load_config()


@st.cache_resource
def _load_model_cached() -> Any:
    """Load and cache the trained pipeline."""
    return load_pipeline()


@st.cache_resource
def _load_metadata_cached() -> dict[str, Any]:
    """Load metadata from models/metrics.json."""
    import json

    from heart_disease.config import get_project_root

    metrics_path = get_project_root() / "models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return cast(dict[str, Any], json.load(f))
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap_pipeline(pipeline: Any) -> Any:
    """Unwrap a ``CalibratedClassifierCV`` to get the base pipeline for SHAP."""
    if hasattr(pipeline, "calibrated_classifiers_"):
        return pipeline.calibrated_classifiers_[0].estimator
    if hasattr(pipeline, "estimator"):
        return pipeline.estimator
    return pipeline


def _patient_to_df(data: dict[str, Any], config: Config) -> Any:
    """Convert patient dict to a single-row DataFrame with correct dtypes."""
    import pandas as pd

    df = pd.DataFrame([data])
    for col in config.features.categorical:
        if col in df.columns:
            df[col] = df[col].astype("object")
    return df


def _format_validation_error(error: Exception) -> str:
    """Format a pydantic ``ValidationError`` into user-friendly text."""
    if hasattr(error, "errors"):
        messages: list[str] = []
        for err in error.errors():  # type: ignore[union-attr]
            field = " → ".join(str(loc) for loc in err["loc"])
            messages.append(f"**{field}**: {err['msg']}")
        return "\n".join(messages)
    return str(error)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Heart Disease Risk Predictor — main Streamlit entry point."""
    st.set_page_config(
        page_title="Heart Disease Risk Predictor",
        page_icon="❤️",
        layout="wide",
    )

    st.title("❤️ Heart Disease Risk Predictor")
    render_disclaimer()

    # Load config
    try:
        config = _load_config_cached()
    except Exception:
        st.error("⚠️ Configuration error. Please check that `configs/config.yaml` exists.")
        return

    # Load model
    try:
        pipeline = _load_model_cached()
    except FileNotFoundError:
        st.error(
            "🔍 **Model not found.** Please train the model first:\n\n"
            "```bash\nmake train\n```\n\n"
            "or\n\n"
            "```bash\npython -m heart_disease.train\n```"
        )
        return
    except Exception:
        st.error("⚠️ Error loading the model. Please retrain with `make train`.")
        return

    # ---- Sidebar: patient input form ----
    metadata = _load_metadata_cached()
    active_thresh = float(metadata.get("decision_threshold", config.training.decision_threshold))
    model_name = metadata.get("model", "calibrated model")

    st.sidebar.header("Patient Clinical Data")
    st.sidebar.caption(
        f"Model: `{model_name}` | Threshold: `{active_thresh:.3f}` *(from `metrics.json`)*"
    )
    patient_data = render_input_form()

    predict_clicked = st.sidebar.button(
        "🔬 Predict Risk", type="primary", use_container_width=True
    )

    if predict_clicked:
        # Check for missing/unknown optional features and surface clinical warning
        active_features = set(config.features.numeric) | set(config.features.categorical)
        missing_inputs = [k for k, v in patient_data.items() if v is None and k in active_features]
        if missing_inputs:
            st.warning(
                f"⚠️ **Missing / Incomplete Clinical Workup**: The following diagnostic feature(s) "
                f"are marked as Unknown: `{', '.join(missing_inputs)}`. Values are automatically "
                f"imputed with population statistics (medians/modes) inside the pipeline. "
                f"Risk predictions for patients with omitted diagnostic procedures carry higher "
                f"clinical uncertainty."
            )

        # Validate
        try:
            patient = validate_input(patient_data)
        except Exception as exc:
            st.error(f"❌ **Invalid input:** {_format_validation_error(exc)}")
            return

        # Predict
        try:
            result = predict(patient, config)
        except Exception:
            st.error("⚠️ Prediction failed. Please check your inputs and try again.")
            return

        # ---- Display results ----
        col1, col2 = st.columns([1, 1])

        with col1:
            render_risk_gauge(
                result["probability"],
                result["risk_level"],
                result["threshold"],
                display_probability=result.get("display_probability"),
            )

        with col2:
            try:
                base_pipeline = _unwrap_pipeline(pipeline)
                shap_vals, feat_names = compute_single_shap(
                    base_pipeline,
                    _patient_to_df(patient_data, config),
                    config,
                )
                render_shap_chart(shap_vals, feat_names)
            except Exception:
                st.warning("⚠️ SHAP explanation is currently unavailable.")
    else:
        st.info("👈 Enter patient data in the sidebar and click **Predict Risk**.")


if __name__ == "__main__":
    main()
