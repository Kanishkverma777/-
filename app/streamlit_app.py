"""Streamlit app entry point for Heart Disease Prediction.

This is a **thin** entry point — all logic lives in ``app/components/``
and ``src/heart_disease/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

# Ensure project root is on sys.path so both ``heart_disease`` and ``app``
# packages are importable when running via ``streamlit run app/streamlit_app.py``.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.ai_summary import render_ai_summary_section  # noqa: E402
from app.components.input_form import render_input_form  # noqa: E402
from app.components.risk_gauge import render_risk_gauge  # noqa: E402
from app.components.shap_chart import _clean_feature_name, render_shap_chart  # noqa: E402
from heart_disease.config import Config, load_config  # noqa: E402
from heart_disease.explain import compute_single_shap  # noqa: E402
from heart_disease.predict import load_pipeline, predict, validate_input  # noqa: E402

logger = logging.getLogger(__name__)

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


def _load_metadata() -> dict[str, Any]:
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
    for col in config.features.numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
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
    """हृदय (Heart Disease Risk Predictor) — main Streamlit entry point."""
    st.set_page_config(
        page_title="हृदय (Heart Disease Risk Predictor)",
        page_icon=":material/cardiology:",
        layout="wide",
    )

    st.markdown(
        "<h1 style='font-size: 5.5rem; font-weight: 800; "
        "margin-bottom: 0.5rem; line-height: 1.15;'>"
        "हृदय "
        "<span style='font-size: 0.55em; font-weight: 500; opacity: 0.85;'>"
        "(Heart Disease Risk Predictor)"
        "</span>"
        "</h1>",
        unsafe_allow_html=True,
    )

    # Load config
    try:
        config = _load_config_cached()
    except Exception:
        st.error(
            "Configuration error. Please check that `configs/config.yaml` exists.",
            icon=":material/error:",
        )
        return

    # Load model
    try:
        _load_model_cached()
    except FileNotFoundError:
        st.error(
            "**Model not found.** Please train the model first:\n\n"
            "```bash\nmake train\n```\n\n"
            "or\n\n"
            "```bash\npython -m heart_disease.train\n```",
            icon=":material/search:",
        )
        return
    except Exception:
        st.error(
            "Error loading the model. Please retrain with `make train`.",
            icon=":material/error:",
        )
        return

    # ---- Sidebar: patient input form ----
    metadata = _load_metadata()
    active_thresh = float(metadata.get("decision_threshold", config.training.decision_threshold))
    model_name = metadata.get("model", "calibrated model")

    st.sidebar.header("Patient Clinical Data", icon=":material/person:")
    st.sidebar.caption(
        f"Model: `{model_name}` | Threshold: `{active_thresh:.3f}` *(from `metrics.json`)*"
    )
    patient_data = render_input_form()

    predict_clicked = st.sidebar.button(
        "Predict Risk",
        icon=":material/analytics:",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        # Clear any existing AI summary cache so fresh prompt and inputs are used
        for key in list(st.session_state.keys()):
            if str(key).startswith("ai_summary_"):
                del st.session_state[key]

        # Check for missing/unknown optional features and surface clinical warning
        active_features = set(config.features.numeric) | set(config.features.categorical)
        missing_inputs = [k for k, v in patient_data.items() if v is None and k in active_features]
        if missing_inputs:
            st.warning(
                f"**Missing / Incomplete Clinical Workup**: The following diagnostic feature(s) "
                f"are marked as Unknown: `{', '.join(missing_inputs)}`. Values are automatically "
                f"imputed with population statistics (medians/modes) inside the pipeline. "
                f"Risk predictions for patients with omitted diagnostic procedures carry higher "
                f"clinical uncertainty.",
                icon=":material/warning:",
            )

        # Validate
        try:
            patient = validate_input(patient_data)
        except Exception as exc:
            st.error(
                f"**Invalid input:** {_format_validation_error(exc)}",
                icon=":material/error:",
            )
            return

        # Predict
        try:
            result = predict(patient, config)
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            st.error(
                f"**Prediction failed:** {exc}. Please check your inputs and try again.",
                icon=":material/error:",
            )
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

        shap_vals = None
        feat_names: list[str] = []
        with col2:
            try:
                active_pipe = load_pipeline(config)
                base_pipeline = _unwrap_pipeline(active_pipe)
                shap_vals, feat_names = compute_single_shap(
                    base_pipeline,
                    _patient_to_df(patient_data, config),
                    config,
                )
                render_shap_chart(shap_vals, feat_names)
            except Exception as exc:
                logger.warning("SHAP explanation unavailable: %s", exc)
                st.warning(
                    "SHAP explanation is currently unavailable.",
                    icon=":material/info:",
                )

        # ---- AI Clinical Summary & Recommendations (GPT-OSS 120B) ----
        top_features: list[tuple[str, float]] = []
        if shap_vals is not None and len(feat_names) == len(shap_vals):
            sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:6]
            top_features = [
                (_clean_feature_name(feat_names[int(i)]), float(shap_vals[int(i)]))
                for i in sorted_indices
            ]

        try:
            render_ai_summary_section(patient_data, result, top_features, config)
        except Exception as exc:
            st.warning(
                f"AI summary is temporarily unavailable: {exc}",
                icon=":material/info:",
            )
    else:
        st.info(
            "Enter patient data in the sidebar and click Predict Risk.",
            icon=":material/arrow_back:",
        )


if __name__ == "__main__":
    main()
