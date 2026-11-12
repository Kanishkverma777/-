"""Model factory for heart disease classifiers.

Provides a factory function that creates an sklearn ``Pipeline``
(preprocessor → classifier) for each supported model type.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from heart_disease.config import Config
from heart_disease.features import build_preprocessor

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def create_pipeline(
    model_name: str,
    config: Config,
    preprocessor: ColumnTransformer | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Create an sklearn Pipeline with preprocessing + classifier.

    Args:
        model_name: One of ``'logistic_regression'``, ``'random_forest'``,
            ``'xgboost'``.
        config: Project configuration.
        preprocessor: Optional custom preprocessor. If None, builds default from config.
        **kwargs: Extra keyword arguments forwarded to the classifier constructor.

    Returns:
        sklearn ``Pipeline`` with steps ``('preprocessor', ...)``,
        ``('classifier', ...)``.

    Raises:
        ValueError: If *model_name* is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)

    if preprocessor is None:
        preprocessor = build_preprocessor(config)
    seed = config.project.random_seed

    classifier_cls = MODEL_REGISTRY[model_name]

    # Sensible defaults per model family
    default_kwargs: dict[str, Any] = {"random_state": seed}

    if model_name == "logistic_regression":
        default_kwargs.update({"max_iter": 2000, "solver": "lbfgs"})
    elif model_name == "xgboost":
        default_kwargs.update(
            {
                "eval_metric": "logloss",
                "verbosity": 0,
            }
        )

    clean_kwargs = {
        (k.replace("classifier__", "") if k.startswith("classifier__") else k): v
        for k, v in kwargs.items()
    }
    default_kwargs.update(clean_kwargs)
    classifier = classifier_cls(**default_kwargs)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.info("Created pipeline: %s", model_name)
    return pipeline


def get_available_models() -> list[str]:
    """Return the list of registered model names."""
    return list(MODEL_REGISTRY.keys())
