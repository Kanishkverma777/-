"""Tests for the configuration loader."""

from __future__ import annotations

from heart_disease.config import Config, get_project_root, load_config


def test_load_config_returns_config() -> None:
    """Config loads successfully from the default path."""
    config = load_config()
    assert isinstance(config, Config)


def test_config_has_expected_sections(config: Config) -> None:
    """Config contains all expected top-level sections."""
    assert config.project.name == "heart-disease-ml"
    assert config.project.random_seed == 42
    assert config.data.test_size > 0
    assert len(config.features.numeric) > 0
    assert len(config.features.categorical) > 0
    assert config.selection.primary_metric == "roc_auc"
    assert config.training.target_recall == 0.90


def test_config_model_grids_present(config: Config) -> None:
    """Config has non-empty hyperparameter grids for all three models."""
    for name in ("logistic_regression", "random_forest", "xgboost"):
        assert name in config.models, f"Missing grid for {name}"
        assert len(config.models[name].param_grid) > 0, f"Empty grid for {name}"


def test_project_root_exists() -> None:
    """Project root directory and config file exist on disk."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "config.yaml").exists()
