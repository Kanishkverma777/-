"""Tests for the feature preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heart_disease.config import Config
from heart_disease.data import clean_data, split_data
from heart_disease.features import build_preprocessor


def test_preprocessor_output_shape(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Preprocessor output has the correct number of rows."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, _, _, _ = split_data(cleaned, config)

    preprocessor = build_preprocessor(config)
    x_transformed = preprocessor.fit_transform(x_train)

    assert x_transformed.shape[0] == len(x_train)
    assert x_transformed.shape[1] > 0


def test_preprocessor_no_nan_output(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """No NaN values survive after imputation."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, _, _, _ = split_data(cleaned, config)

    preprocessor = build_preprocessor(config)
    x_transformed = preprocessor.fit_transform(x_train)

    assert not np.any(np.isnan(x_transformed))


def test_preprocessor_feature_names(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Preprocessor produces correctly prefixed feature names."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, _, _, _ = split_data(cleaned, config)

    preprocessor = build_preprocessor(config)
    preprocessor.fit(x_train)

    names = list(preprocessor.get_feature_names_out())
    assert len(names) > 0
    num_names = [n for n in names if n.startswith("num__")]
    assert len(num_names) == len(config.features.numeric)


def test_preprocessor_transform_consistency(
    sample_dataframe: pd.DataFrame, config: Config
) -> None:
    """Same input produces the same output (deterministic)."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, _, _, _ = split_data(cleaned, config)

    preprocessor = build_preprocessor(config)
    preprocessor.fit(x_train)

    out1 = preprocessor.transform(x_train)
    out2 = preprocessor.transform(x_train)
    np.testing.assert_array_equal(out1, out2)


def test_forbidden_columns_raise_error_when_configured(config: Config) -> None:
    """Configuring 'id' or 'dataset' as features raises a ValueError."""
    import pytest

    # Test 'id' in numeric
    bad_config = config.model_copy(deep=True)
    bad_config.features.numeric.append("id")
    with pytest.raises(ValueError, match="Forbidden columns configured as features"):
        build_preprocessor(bad_config)

    # Test 'dataset' in categorical
    bad_config2 = config.model_copy(deep=True)
    bad_config2.features.categorical.append("dataset")
    with pytest.raises(ValueError, match="Forbidden columns configured as features"):
        build_preprocessor(bad_config2)


def test_pipeline_strictly_excludes_id_and_dataset(
    sample_dataframe: pd.DataFrame, config: Config
) -> None:
    """Neither 'id' nor 'dataset' ever enters the transformed features."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, _, _, _ = split_data(cleaned, config)

    # Add id and dataset columns directly to the input DataFrame
    x_with_metadata = x_train.copy()
    x_with_metadata["id"] = range(1, len(x_with_metadata) + 1)
    x_with_metadata["dataset"] = ["Cleveland"] * len(x_with_metadata)

    preprocessor = build_preprocessor(config)
    preprocessor.fit(x_with_metadata)

    feature_names = list(preprocessor.get_feature_names_out())
    assert len(feature_names) > 0

    # Ensure neither id nor dataset is present anywhere in output feature names
    for name in feature_names:
        assert "id" not in name.lower().split("__")[-1].split("_")[0], (
            f"Found 'id' in feature {name}"
        )
        assert "dataset" not in name.lower(), f"Found 'dataset' in feature {name}"

    # Output dimension matches clean data without extra columns
    out_clean = preprocessor.transform(x_train)
    out_with_meta = preprocessor.transform(x_with_metadata)
    np.testing.assert_array_equal(out_clean, out_with_meta)


def test_preprocessor_custom_feature_subsets(config: Config) -> None:
    """build_preprocessor accepts custom subsets of numeric and categorical features."""
    custom_num = ["age", "trestbps"]
    custom_cat = ["sex", "cp"]
    prep = build_preprocessor(config, numeric_features=custom_num, categorical_features=custom_cat)
    assert prep is not None
