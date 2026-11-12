"""Data loading, cleaning, and splitting for heart disease prediction.

Cleaning decisions (all documented inline):
1. Target binarization: 0 = no disease, ≥1 = disease (standard in literature).
2. Missing markers: ``?``, ``-9``, ``NaN``, and ``0`` (for cholesterol) → NaN.
3. Duplicate rows are dropped.
4. All imputation happens *inside* the sklearn Pipeline (no leakage).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart_disease.config import Config, get_project_root
from heart_disease.schema import ALL_FEATURES, RAW_CLINICAL_FEATURES

logger = logging.getLogger(__name__)


EXPECTED_DATASETS: set[str] = {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}
REQUIRED_RAW_COLUMNS: set[str] = {
    "id",
    "dataset",
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
}


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate that the loaded raw dataset conforms strictly to the 920-row 4-center schema.

    Checks:
    1. Exactly 920 rows (rejects truncated or 303-row single-center files).
    2. 'dataset' column present with exactly 4 distinct expected centers:
       Cleveland, Hungary, Switzerland, VA Long Beach.
    3. All expected raw clinical columns present.

    Args:
        df: Raw DataFrame to validate.

    Raises:
        ValueError: If row count != 920, dataset column missing or does not have exactly
            4 distinct expected clinical sites, or required columns are missing.
    """
    if len(df) != 920:
        raise ValueError(
            f"Invalid raw dataset row count: expected exactly 920 rows, got {len(df)}. "
            "Truncated or single-center datasets (e.g. 303-row Cleveland cache) are prohibited."
        )

    cols_lower = {str(c).lower().strip() for c in df.columns}
    if "dataset" not in cols_lower:
        raise ValueError("Raw dataset missing required grouping column: 'dataset'")

    missing_cols = REQUIRED_RAW_COLUMNS - cols_lower
    # Accept either thalch or thalach
    if "thalch" not in cols_lower and "thalach" not in cols_lower:
        missing_cols.add("thalach/thalch")

    if missing_cols:
        raise ValueError(
            f"Raw dataset is missing required clinical columns: {sorted(missing_cols)}"
        )

    # Find the actual case-preserved dataset column name
    dataset_col = next(c for c in df.columns if str(c).lower().strip() == "dataset")
    sites = set(df[dataset_col].dropna().unique())
    if len(sites) != 4:
        site_list = sorted(sites)
        msg = (
            f"Expected exactly 4 distinct clinical sites in 'dataset', "
            f"got {len(sites)}: {site_list}"
        )
        raise ValueError(msg)
    if sites != EXPECTED_DATASETS:
        msg = (
            f"Unexpected clinical sites in 'dataset': got {sorted(sites)}, "
            f"expected {sorted(EXPECTED_DATASETS)}"
        )
        raise ValueError(msg)


def load_raw_data(config: Config) -> pd.DataFrame:
    """Load and validate raw heart disease data from verified multi-center dataset.

    Primary mode loads from config.data.raw_file ('data/raw/heart_disease_uci.csv').
    If missing, attempts to download the verified 920-row Kaggle dataset via kagglehub.
    All single-center (303-row) fallbacks (ucimlrepo, processed.cleveland.data,
    heart_disease.csv) are permanently removed to eliminate silent dataset substitution.

    Args:
        config: Project configuration.

    Returns:
        Validated raw DataFrame with 920 rows across 4 clinical centers.

    Raises:
        RuntimeError: If data file does not exist and cannot be fetched.
        ValueError: If loaded data fails strict 920-row 4-center validation.
    """
    raw_dir = get_project_root() / config.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_file_path = raw_dir / config.data.raw_file

    if data_file_path.exists():
        logger.info("Loading data from raw dataset file: %s", data_file_path)
        df = pd.read_csv(data_file_path, na_values=config.data.missing_markers)
    else:
        # Download via kagglehub if primary file is missing
        try:
            import shutil

            import kagglehub  # type: ignore[import-untyped]

            logger.info(
                "Fetching 920-row multi-center dataset from Kaggle via kagglehub (%s)...",
                config.data.kaggle_dataset_slug,
            )
            kaggle_path = Path(kagglehub.dataset_download(config.data.kaggle_dataset_slug))
            csv_candidates = list(kaggle_path.glob("*.csv"))
            if csv_candidates:
                shutil.copy2(csv_candidates[0], data_file_path)
                logger.info("Cached Kaggle dataset to %s", data_file_path)
                df = pd.read_csv(data_file_path, na_values=config.data.missing_markers)
            else:
                raise FileNotFoundError(f"No CSV found in Kaggle download at {kaggle_path}")
        except Exception as exc:
            msg = (
                f"Could not load data. Place the verified 920-row multi-center CSV at "
                f"{data_file_path}. (ucimlrepo and 303-row fallbacks are disabled). Error: {exc}"
            )
            raise RuntimeError(msg) from exc

    validate_raw_data(df)
    return cast(pd.DataFrame, df)


def clean_data(
    df: pd.DataFrame,
    config: Config,
    preserve_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Clean the raw heart disease dataset.

    Steps:
    1. Standardise column names to lowercase.
    2. Normalize Kaggle column names (e.g. thalch -> thalach).
    3. Map text categorical values to canonical integer encodings.
    4. Replace missing markers with NaN.
    5. Replace cholesterol == 0 with NaN (clinically impossible).
    6. Drop duplicate rows.
    7. Binarise target: 0 → 0, ≥ 1 → 1.
    8. Keep only known feature + target columns (plus any explicitly preserved columns).
    9. Cast numeric columns to float, categorical to ``object``.

    Args:
        df: Raw DataFrame.
        config: Project configuration.
        preserve_columns: Optional list of non-feature columns to preserve
            (e.g. ['dataset'] for multi-center leave-one-site-out grouping).
            These are never included in the model feature matrix.

    Returns:
        Cleaned DataFrame ready for train/test split.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    logger.info("Raw data shape: %s", df.shape)

    # --- Normalize Kaggle column names & map text encodings ---
    if "thalch" in df.columns and "thalach" not in df.columns:
        df = df.rename(columns={"thalch": "thalach"})

    sex_map: dict[object, int] = {
        "male": 1,
        "female": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    cp_map: dict[object, int] = {
        "typical angina": 1,
        "atypical angina": 2,
        "non-anginal": 3,
        "asymptomatic": 4,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    }
    restecg_map: dict[object, int] = {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2,
        "0": 0,
        "1": 1,
        "2": 2,
        0: 0,
        1: 1,
        2: 2,
    }
    slope_map: dict[object, int] = {
        "upsloping": 1,
        "flat": 2,
        "downsloping": 3,
        "1": 1,
        "2": 2,
        "3": 3,
        1: 1,
        2: 2,
        3: 3,
    }
    thal_map: dict[object, int] = {
        "normal": 3,
        "fixed defect": 6,
        "reversable defect": 7,
        "reversible defect": 7,
        "3": 3,
        "6": 6,
        "7": 7,
        3: 3,
        6: 6,
        7: 7,
    }

    bool_map: dict[object, int] = {
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = df["sex"].astype(str).str.lower().map(sex_map)
    if "cp" in df.columns and df["cp"].dtype == "object":
        df["cp"] = df["cp"].astype(str).str.lower().map(cp_map)
    if "fbs" in df.columns and (df["fbs"].dtype == "bool" or df["fbs"].dtype == "object"):
        df["fbs"] = df["fbs"].map(bool_map)
    if "restecg" in df.columns and df["restecg"].dtype == "object":
        df["restecg"] = df["restecg"].astype(str).str.lower().map(restecg_map)
    if "exang" in df.columns and (df["exang"].dtype == "bool" or df["exang"].dtype == "object"):
        df["exang"] = df["exang"].map(bool_map)
    if "slope" in df.columns and df["slope"].dtype == "object":
        df["slope"] = df["slope"].astype(str).str.lower().map(slope_map)
    if "thal" in df.columns and df["thal"].dtype == "object":
        df["thal"] = df["thal"].astype(str).str.lower().map(thal_map)

    # --- Replace missing markers with NaN ---
    for marker in config.data.missing_markers:
        df.replace(marker, np.nan, inplace=True)  # noqa: PD002

    # --- Physiological zero check: chol=0 and trestbps=0 are clinically impossible → NaN ---
    for col in ("chol", "trestbps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            zero_count = int((df[col] == 0).sum())
            if zero_count > 0:
                logger.info(
                    "Replacing %d zero-values in '%s' with NaN (treated as missing)",
                    zero_count,
                    col,
                )
                df.loc[df[col] == 0, col] = np.nan

    # --- Drop duplicates ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d duplicate rows", n_dropped)

    # --- Binarise target ---
    target = config.data.target_column
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[target] = (df[target] >= 1).astype(int)
        logger.info(
            "Target distribution after binarisation:\n%s",
            df[target].value_counts().to_string(),
        )

    # --- Keep only known feature + target columns (plus preserved grouping cols) ---
    extra_cols = preserve_columns or []
    allowed_cols = list(
        dict.fromkeys(
            list(ALL_FEATURES) + RAW_CLINICAL_FEATURES + [target] + [c.lower() for c in extra_cols]
        )
    )
    keep_cols = [c for c in allowed_cols if c in df.columns]
    missing_cols = set(ALL_FEATURES) - set(df.columns)
    if missing_cols:
        logger.warning("Missing expected columns: %s", missing_cols)
    df = df[keep_cols]

    # --- Cast dtypes ---
    for col in config.features.numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config.features.categorical:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # --- Report missingness ---
    missing_report = df[ALL_FEATURES].isnull().sum()
    cols_with_missing = missing_report[missing_report > 0]
    if len(cols_with_missing) > 0:
        missing_pct = (cols_with_missing / len(df) * 100).round(1)
        logger.info(
            "Missing values:\n%s",
            pd.DataFrame({"count": cols_with_missing, "pct%": missing_pct}).to_string(),
        )

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification.

    Ensures the feature matrix X contains strictly the clinical features
    defined in ALL_FEATURES (never id, dataset, or metadata columns).

    Args:
        df: Cleaned DataFrame containing features + target.
        config: Project configuration.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)``.
    """
    target = config.data.target_column
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    x = df[feature_cols].copy()
    y = df[target].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.data.test_size,
        random_state=config.project.random_seed,
        stratify=y,
    )

    logger.info(
        "Train/test split: train=%d (disease=%.1f%%), test=%d (disease=%.1f%%)",
        len(x_train),
        y_train.mean() * 100,
        len(x_test),
        y_test.mean() * 100,
    )

    return x_train, x_test, y_train, y_test


def get_site_grouped_data(
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load cleaned features, target, and clinical site grouping variable.

    The site column is strictly for multi-center leave-one-site-out
    evaluation and is NEVER included in the feature matrix X.

    Args:
        config: Project configuration.

    Returns:
        Tuple of ``(X, y, sites)`` where X contains ONLY clinical features.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config, preserve_columns=["dataset"])

    feature_cols = [c for c in ALL_FEATURES if c in clean_df.columns]
    x = clean_df[feature_cols].copy()
    target = config.data.target_column
    y = clean_df[target].copy()

    if "dataset" in clean_df.columns:
        sites = clean_df["dataset"].copy()
    else:
        sites = pd.Series(["Cleveland"] * len(clean_df), index=clean_df.index, name="dataset")

    logger.info(
        "Loaded site-grouped data: %d samples across sites: %s",
        len(x),
        sites.value_counts().to_dict(),
    )
    return x, y, sites


def load_and_prepare(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Full data pipeline: load → clean → split.

    Args:
        config: Project configuration.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)``.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)
    return split_data(clean_df, config)
