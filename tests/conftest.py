"""Shared test fixtures for the heart disease project."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from heart_disease.config import Config, load_config


@pytest.fixture()
def config() -> Config:
    """Load the project configuration."""
    return load_config()


@pytest.fixture()
def sample_patient_data() -> dict[str, Any]:
    """Return a valid patient data dictionary."""
    return {
        "age": 55.0,
        "sex": 1,
        "cp": 1,
        "trestbps": 130.0,
        "chol": 245.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0.0,
        "thal": 3,
    }


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """Return a small synthetic DataFrame mimicking the cleaned dataset."""
    rng = np.random.default_rng(42)
    n = 50
    data: dict[str, Any] = {
        "age": rng.uniform(30, 80, n),
        "sex": rng.choice([0, 1], n),
        "cp": rng.choice([1, 2, 3, 4], n),
        "trestbps": rng.uniform(90, 200, n),
        "chol": rng.uniform(100, 400, n),
        "fbs": rng.choice([0, 1], n),
        "restecg": rng.choice([0, 1, 2], n),
        "thalach": rng.uniform(70, 200, n),
        "exang": rng.choice([0, 1], n),
        "oldpeak": rng.uniform(0, 6, n),
        "slope": rng.choice([1, 2, 3], n),
        "ca": rng.choice([0, 1, 2, 3], n).astype(float),
        "thal": rng.choice([3, 6, 7], n),
        "num": rng.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    # Insert a few NaN values to exercise imputation
    df.loc[0, "trestbps"] = np.nan
    df.loc[1, "chol"] = np.nan
    df.loc[2, "ca"] = np.nan
    df.loc[3, "fbs"] = np.nan
    df.loc[4, "oldpeak"] = np.nan
    return cast(pd.DataFrame, df)
