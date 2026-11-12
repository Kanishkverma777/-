"""LLM clinical summary service using Groq and open-weights models (GPT-OSS 120B).

Generates evidence-based clinical summaries and prioritized actionable next steps
based on patient demographics, resting hemodynamics, exercise stress test responses,
and model-derived SHAP feature attributions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from heart_disease.config import Config, get_project_root
from heart_disease.schema import CATEGORICAL_LABELS

logger = logging.getLogger(__name__)


def get_groq_api_key(config: Config) -> str:
    """Retrieve Groq API key from environment, st.secrets, .env file, or config default."""
    env_var = config.llm.api_key_env_var

    # 1. Environment variable
    key = os.environ.get(env_var, "").strip()
    if key:
        return key

    # 2. Streamlit secrets (for cloud deployments)
    try:
        import streamlit as st

        if hasattr(st, "secrets") and env_var in st.secrets:
            sec_key = str(st.secrets[env_var]).strip()
            if sec_key:
                return sec_key
    except Exception:
        pass

    # 3. .env file in project root
    env_file = get_project_root() / ".env"
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception as exc:
            logger.debug("Could not read .env file: %s", exc)

    # 4. Config default
    if config.llm.default_api_key:
        return config.llm.default_api_key.strip()

    return ""


def format_patient_summary_text(patient_data: dict[str, Any]) -> str:
    """Format raw patient dictionary into a human-readable clinical profile string."""
    sex_str = CATEGORICAL_LABELS["sex"].get(patient_data.get("sex", -1), "Unknown")
    cp_raw = patient_data.get("cp", -1)
    cp_map = {
        1: "Typical angina (chest discomfort brought on by physical effort)",
        2: "Atypical chest discomfort",
        3: "Non-anginal chest pain",
        4: "No chest discomfort reported (asymptomatic)",
    }
    cp_str = cp_map.get(cp_raw, "Unknown")

    restecg_map = {
        0: "Normal resting heart rhythm",
        1: "Mild ST-T electrical changes",
        2: "Thickened heart muscle pattern (LV hypertrophy)",
    }
    restecg_val = patient_data.get("restecg")
    restecg_str = restecg_map.get(restecg_val, "Unknown") if restecg_val is not None else "Unknown"

    exang_val = patient_data.get("exang")
    exang_str = (
        "Yes (chest discomfort occurred during physical activity)"
        if exang_val == 1
        else (
            "No (no chest discomfort during physical activity)"
            if exang_val == 0
            else "Unknown / Not Tested"
        )
    )

    fbs_val = patient_data.get("fbs")
    fbs_str = (
        "Elevated (> 120 mg/dl)"
        if fbs_val == 1
        else ("Normal (≤ 120 mg/dl)" if fbs_val == 0 else "Unknown / Not Tested")
    )

    def _val_str(val: Any, unit: str = "") -> str:
        return f"{val} {unit}".strip() if val is not None else "Unknown / Not Recorded"

    oldpeak_val = patient_data.get("oldpeak")
    oldpeak_str = (
        f"{oldpeak_val} mm (sign of reduced blood flow to heart muscle during exercise)"
        if oldpeak_val is not None and oldpeak_val > 0
        else (f"{oldpeak_val} mm" if oldpeak_val is not None else "Unknown / Not Recorded")
    )

    lines = [
        f"- Age: {_val_str(patient_data.get('age'), 'years old')}",
        f"- Sex: {sex_str}",
        f"- Chest Discomfort Type: {cp_str}",
        f"- Resting Blood Pressure: {_val_str(patient_data.get('trestbps'), 'mm Hg')}",
        f"- Total Serum Cholesterol: {_val_str(patient_data.get('chol'), 'mg/dl')}",
        f"- Blood Sugar: {fbs_str}",
        f"- Resting ECG: {restecg_str}",
        f"- Max Exercise Heart Rate: {_val_str(patient_data.get('thalach'), 'bpm')}",
        f"- Exercise-Induced Chest Pain: {exang_str}",
        f"- Exercise ECG Changes (ST depression): {oldpeak_str}",
    ]
    return "\n".join(lines)


def generate_clinical_summary(
    patient_data: dict[str, Any],
    prediction_result: dict[str, Any],
    top_features: list[tuple[str, float]],
    config: Config,
) -> str:
    """Call Groq API using GPT-OSS 120B to generate a human-friendly clinical summary.

    Args:
        patient_data: Raw input dictionary with patient clinical measurements.
        prediction_result: Dict containing probability, risk_level, threshold, prediction.
        top_features: List of (feature_display_name, shap_value) tuples sorted by impact.
        config: System configuration containing LLM parameters.

    Returns:
        Structured Markdown text with human-friendly explanation and tailored actions.
    """
    api_key = get_groq_api_key(config)
    if not api_key:
        return (
            "> :material/warning: **Groq API Key Not Configured**: Please set the `GROQ_API_KEY` "
            "environment variable or add it to `.env` to enable GPT-OSS 120B clinical summaries."
        )

    patient_profile = format_patient_summary_text(patient_data)

    prob = prediction_result.get("probability", 0.0)
    risk_level = prediction_result.get("risk_level", "Unknown")
    threshold = prediction_result.get("threshold", 0.36)

    shap_increases = [f"{name} (impact: +{val:.3f})" for name, val in top_features if val > 0]
    shap_decreases = [f"{name} (impact: {val:.3f})" for name, val in top_features if val < 0]

    increases_str = "\n".join(f"  * {s}" for s in shap_increases) if shap_increases else "  * None"
    decreases_str = "\n".join(f"  * {s}" for s in shap_decreases) if shap_decreases else "  * None"

    system_prompt = (
        "You are a friendly, caring, and encouraging cardiovascular health advisor. "
        "Your mission is to explain heart health assessment results in warm, plain language "
        "that is very easy for any patient or family member to understand. "
        "Avoid heavy medical jargon. Do NOT use complex terms like 'subendocardial ischemia', "
        "'chronotropic response', 'epicardial stenosis', 'repolarization disturbance', or "
        "'microvascular dysfunction'. Instead, use simple, everyday phrases like 'how well blood "
        "and oxygen reach your heart muscle during exercise', 'cholesterol buildup in arteries', "
        "and 'how effectively your heart speeds up during physical activity'. "
        "Do NOT use any emoji characters. Use clean markdown formatting with bold headers and "
        "bullet points.\n\n"
        "You MUST structure your response into exactly three sections:\n"
        "1. **Your Health Summary in Plain English**: A warm, clear explanation of the "
        "estimated risk percentage and what the category (Low, Moderate, or High) means in "
        "simple everyday terms.\n"
        "2. **What Factors Influenced Your Result**: Plain-language points explaining what "
        "habits and test results helped protect the heart and what factors added risk.\n"
        "3. **Your Personalized Action Plan**:\n"
        "   - If the patient is at High or Moderate risk: Provide clear, prioritized 'Actions to "
        "Reduce Your Risk & Protect Your Heart', including following up with a healthcare "
        "provider, simple lifestyle changes to lower risk, and warning signs to never ignore.\n"
        "   - If the patient is at Low or Normal risk: Provide encouraging 'Actions to Stay "
        "Healthy & Keep Your Heart Strong', including maintaining regular physical activity, "
        "balanced nutrition, routine health check-ups, and habits to keep the heart strong."
    )

    risk_direction_str = (
        "Above the decision cutoff — elevated concern"
        if prob >= threshold
        else "Below the decision cutoff — low/normal risk profile"
    )

    action_goal = (
        "The patient is in an elevated or moderate risk category. In Section 3, focus on clear, "
        "practical 'Actions to Reduce Your Risk & Protect Your Heart' (talking with a doctor, "
        "recommended tests, and actionable steps to bring risk down)."
        if (prob >= threshold or risk_level in ("Elevated", "High", "Moderate"))
        else "The patient is in a low or normal risk category. In Section 3, focus on "
        "encouraging 'Actions to Stay Healthy & Keep Your Heart Strong' (maintaining regular "
        "physical activity, heart-healthy eating, and routine check-ups)."
    )

    user_prompt = (
        f"### Patient Health Information:\n{patient_profile}\n\n"
        f"### Assessment Scores:\n"
        f"- Estimated Risk Probability: {prob:.1%}\n"
        f"- Overall Category: {risk_level} Risk ({risk_direction_str})\n"
        f"- Decision Cutoff Threshold: {threshold:.3f}\n\n"
        f"### Factors That Influenced This Score:\n"
        f"- Factors Raising Risk:\n{increases_str}\n"
        f"- Factors Lowering/Protecting Risk:\n{decreases_str}\n\n"
        f"### Action Plan Guidance:\n{action_goal}\n\n"
        "Please provide the warm, plain-English summary following the 3-section structure."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "HeartDiseasePredictor/1.0",
    }

    payload: dict[str, Any] = {
        "model": config.llm.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
    }

    try:
        logger.info("Calling Groq API model %s for clinical summary...", config.llm.model)
        response = requests.post(
            config.llm.api_url,
            headers=headers,
            json=payload,
            timeout=25,
        )
        if response.status_code == 200:
            data = response.json()
            content = str(data["choices"][0]["message"]["content"])
            return content.strip()

        logger.error("Groq API error (%d): %s", response.status_code, response.text)
        error_snippet = response.text[:200]
        return (
            f"> :material/error: **AI Summary Generation Failed** "
            f"(Groq API HTTP {response.status_code}): `{error_snippet}`"
        )
    except requests.Timeout:
        logger.error("Groq API request timed out.")
        return (
            "> :material/warning: **AI Summary Request Timed Out**: "
            "The Groq API did not respond in time. Please try again."
        )
    except Exception as exc:
        logger.exception("Unexpected error during Groq API call: %s", exc)
        return f"> :material/error: **AI Summary Service Error**: `{exc}`"
