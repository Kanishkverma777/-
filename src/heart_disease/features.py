"""Feature preprocessing pipeline builder.

Builds a scikit-learn ``ColumnTransformer`` that handles:

- **Numeric features**: median imputation → standard scaling.
- **Categorical features**: most-frequent imputation → one-hot encoding.

All transformations live inside the ``Pipeline`` to prevent data leakage.
"""

from __future__ import annotations

import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease.config import Config

logger = logging.getLogger(__name__)

FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"id", "dataset"})


def build_preprocessor(
    config: Config,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the feature preprocessing ``ColumnTransformer``.

    Args:
        config: Project configuration with feature column definitions.
        numeric_features: Optional custom list of numeric features. If None,
            uses ``config.features.numeric``.
        categorical_features: Optional custom list of categorical features. If None,
            uses ``config.features.categorical``.

    Returns:
        Un-fitted ``ColumnTransformer`` ready to be embedded in a ``Pipeline``.

    Raises:
        ValueError: If forbidden columns (e.g. 'id', 'dataset') are configured
            as features.
    """
    if numeric_features is None:
        numeric_features = list(config.features.numeric)
    if categorical_features is None:
        categorical_features = list(config.features.categorical)

    configured_features = set(numeric_features) | set(categorical_features)
    leaked_columns = configured_features & FORBIDDEN_COLUMNS
    if leaked_columns:
        msg = (
            f"Forbidden columns configured as features: {sorted(leaked_columns)}. "
            "Neither 'id' nor 'dataset' may be used as model features."
        )
        raise ValueError(msg)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    logger.info(
        "Built preprocessor: %d numeric features, %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )

    return preprocessor
