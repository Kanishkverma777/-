"""Typed configuration loader for the heart disease prediction project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    name: str = "heart-disease-ml"
    version: str = "1.0.0"
    random_seed: int = 42


class DataConfig(BaseModel):
    """Data loading and splitting configuration."""

    uci_dataset_id: int = 45
    kaggle_dataset_slug: str = "redwankarimsony/heart-disease-data"
    raw_dir: str = "data/raw"
    raw_file: str = "heart_disease_uci.csv"
    test_size: float = 0.2
    target_column: str = "num"
    missing_markers: list[str] = Field(default_factory=lambda: ["?", "-9"])


class FeaturesConfig(BaseModel):
    """Feature column definitions."""

    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    cv_folds: int = 5
    cv_repeats: int = 10
    n_iter: int = 50
    scoring: str = "roc_auc"
    target_recall: float = 0.90
    decision_threshold: float = 0.5


class SelectionConfig(BaseModel):
    """Model selection rule configuration."""

    primary_metric: str = "roc_auc"
    tolerance_std_factor: float = 1.0
    tie_breakers: list[str] = Field(default_factory=lambda: ["lower_std", "simplicity"])
    simplicity_order: list[str] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost"]
    )


class ModelParamGrid(BaseModel):
    """Hyperparameter grid for a single model."""

    param_grid: dict[str, list[Any]] = Field(default_factory=dict)


class PathsConfig(BaseModel):
    """Output paths."""

    models_dir: str = "models"
    reports_dir: str = "reports"


class AppConfig(BaseModel):
    """Streamlit app configuration."""

    risk_thresholds: dict[str, float] = Field(default_factory=dict)
    display_min_probability: float = 0.01
    display_max_probability: float = 0.99


class AblationConfig(BaseModel):
    """Feature ablation selection tolerance configuration."""

    cv_std_factor: float = Field(default=1.0, ge=0.0)
    loso_auc_tolerance: float = Field(default=0.01, ge=0.0)


class LLMConfig(BaseModel):
    """Configuration for LLM clinical summary generation."""

    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    temperature: float = 0.3
    max_tokens: int = 800
    api_key_env_var: str = "GROQ_API_KEY"
    default_api_key: str = ""


class Config(BaseModel):
    """Root configuration model."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    models: dict[str, ModelParamGrid] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: Path | None = None) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file. Defaults to configs/config.yaml.

    Returns:
        Validated Config object.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    logger.info("Loading config from %s", config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = Config.model_validate(raw)
    logger.info(
        "Config loaded successfully (project=%s, version=%s)",
        config.project.name,
        config.project.version,
    )
    return config


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
