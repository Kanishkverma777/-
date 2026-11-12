"""Tests for data loading and cleaning."""

from __future__ import annotations

import pandas as pd

from heart_disease.config import Config
from heart_disease.data import clean_data, split_data


def test_clean_data_binarises_target(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Target column is binarised to {0, 1}."""
    df = sample_dataframe.copy()
    df["num"] = [0, 1, 2, 3, 4] * 10
    cleaned = clean_data(df, config)
    assert set(cleaned["num"].unique()) == {0, 1}


def test_clean_data_handles_missing_markers(config: Config) -> None:
    """Strings ``'?'`` and cholesterol == 0 become NaN after cleaning."""
    df = pd.DataFrame(
        {
            "age": [55, 60, "?"],
            "sex": [1, 0, 1],
            "cp": [1, 2, 3],
            "trestbps": [130, "?", 140],
            "chol": [245, 300, 0],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 2],
            "thalach": [150, 160, 170],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 3.0],
            "slope": [1, 2, 3],
            "ca": [0, 1, "?"],
            "thal": [3, 6, 7],
            "num": [0, 1, 2],
        }
    )
    cleaned = clean_data(df, config)

    assert cleaned["trestbps"].isna().sum() >= 1, "? marker not converted to NaN"
    assert cleaned["chol"].isna().sum() >= 1, "chol=0 not converted to NaN"


def test_zero_as_missing_rule(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """chol=0 and trestbps=0 are converted to NaN (treated as missing)."""
    df = sample_dataframe.copy()
    df.loc[0, "chol"] = 0
    df.loc[1, "trestbps"] = 0

    cleaned = clean_data(df, config)
    assert cleaned["chol"].isna().iloc[0], "chol=0 was not converted to NaN"
    assert cleaned["trestbps"].isna().iloc[1], "trestbps=0 was not converted to NaN"
    assert (cleaned["chol"] == 0).sum() == 0, "Zero cholesterol value survived cleaning"
    assert (cleaned["trestbps"] == 0).sum() == 0, "Zero trestbps value survived cleaning"


def test_clean_data_drops_duplicates(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Duplicate rows are removed."""
    df = pd.concat([sample_dataframe, sample_dataframe.iloc[:5]], ignore_index=True)
    cleaned = clean_data(df, config)
    assert len(cleaned) <= len(sample_dataframe)


def test_split_data_stratified(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Train/test split preserves class proportions."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, x_test, y_train, y_test = split_data(cleaned, config)

    assert len(x_train) + len(x_test) == len(cleaned)
    assert len(y_train) + len(y_test) == len(cleaned)
    assert config.data.target_column not in x_train.columns

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.15


def test_split_data_no_leakage(sample_dataframe: pd.DataFrame, config: Config) -> None:
    """Train and test indices are disjoint (no data leakage)."""
    cleaned = clean_data(sample_dataframe, config)
    x_train, x_test, _, _ = split_data(cleaned, config)
    assert len(set(x_train.index) & set(x_test.index)) == 0


def test_load_raw_data_and_prepare(config: Config) -> None:
    """load_raw_data and load_and_prepare load the database and return splits."""
    from heart_disease.data import load_and_prepare, load_raw_data

    raw = load_raw_data(config)
    assert len(raw) == 920
    assert "num" in raw.columns

    x_train, x_test, y_train, y_test = load_and_prepare(config)
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)


def test_clean_data_handles_kaggle_strings(config: Config) -> None:
    """clean_data handles text categoricals from the Kaggle dataset."""
    df_kaggle = pd.DataFrame(
        {
            "age": [63, 67],
            "sex": ["Male", "Female"],
            "cp": ["typical angina", "asymptomatic"],
            "trestbps": [145, 160],
            "chol": [233, 286],
            "fbs": [True, False],
            "restecg": ["normal", "lv hypertrophy"],
            "thalch": [150, 108],
            "exang": [False, True],
            "oldpeak": [2.3, 1.5],
            "slope": ["downsloping", "flat"],
            "ca": [0.0, 3.0],
            "thal": ["fixed defect", "normal"],
            "num": [0, 2],
        }
    )
    cleaned = clean_data(df_kaggle, config)
    assert "thalach" in cleaned.columns
    assert cleaned.loc[0, "sex"] == 1
    assert cleaned.loc[1, "sex"] == 0
    assert cleaned.loc[0, "fbs"] == 1
    assert cleaned.loc[1, "fbs"] == 0
    assert cleaned.loc[0, "num"] == 0
    assert cleaned.loc[1, "num"] == 1


def test_get_site_grouped_data(config: Config) -> None:
    """get_site_grouped_data returns clinical features X without dataset or id."""
    from heart_disease.data import get_site_grouped_data

    x, y, sites = get_site_grouped_data(config)
    assert len(x) == len(y) == len(sites)
    assert "dataset" not in x.columns
    assert "id" not in x.columns
    assert "num" not in x.columns
    assert set(sites.unique()) == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}


def test_validate_raw_data_valid(config: Config) -> None:
    """validate_raw_data passes on the verified 920-row 4-center dataset."""
    from heart_disease.data import load_raw_data, validate_raw_data

    raw = load_raw_data(config)
    # Should complete without error
    validate_raw_data(raw)


def test_validate_raw_data_invalid_rows(config: Config) -> None:
    """validate_raw_data rejects datasets with row count != 920."""
    import pytest

    from heart_disease.data import load_raw_data, validate_raw_data

    raw = load_raw_data(config)
    truncated = raw.iloc[:303].copy()
    with pytest.raises(ValueError, match="expected exactly 920 rows"):
        validate_raw_data(truncated)


def test_validate_raw_data_missing_dataset_column(config: Config) -> None:
    """validate_raw_data rejects datasets missing the 'dataset' grouping column."""
    import pytest

    from heart_disease.data import load_raw_data, validate_raw_data

    raw = load_raw_data(config)
    without_dataset = raw.drop(columns=["dataset"])
    with pytest.raises(ValueError, match="missing required grouping column"):
        validate_raw_data(without_dataset)


def test_validate_raw_data_wrong_site_count(config: Config) -> None:
    """validate_raw_data rejects datasets with fewer than 4 distinct clinical centers."""
    import pytest

    from heart_disease.data import load_raw_data, validate_raw_data

    raw = load_raw_data(config).copy()
    # Replace all sites with Cleveland and Hungary
    raw["dataset"] = ["Cleveland"] * 460 + ["Hungary"] * 460
    with pytest.raises(ValueError, match="Expected exactly 4 distinct clinical sites"):
        validate_raw_data(raw)


def test_validate_raw_data_missing_columns(config: Config) -> None:
    """validate_raw_data rejects datasets missing expected clinical feature columns."""
    import pytest

    from heart_disease.data import load_raw_data, validate_raw_data

    raw = load_raw_data(config).copy()
    without_chol = raw.drop(columns=["chol"])
    with pytest.raises(ValueError, match="missing required clinical columns"):
        validate_raw_data(without_chol)
