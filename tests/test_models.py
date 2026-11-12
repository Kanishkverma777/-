"""Tests for model creation and registry."""

from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from heart_disease.config import Config
from heart_disease.models import create_pipeline, get_available_models


def test_get_available_models() -> None:
    """All three core models are registered."""
    models = get_available_models()
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models


def test_create_pipeline_valid(config: Config) -> None:
    """Pipelines are created with preprocessor and classifier steps."""
    for model_name in get_available_models():
        pipeline = create_pipeline(model_name, config)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_create_pipeline_unknown_raises(config: Config) -> None:
    """Unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        create_pipeline("non_existent_model", config)
