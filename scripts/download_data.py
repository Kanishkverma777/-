"""Script to download and cache the raw UCI Heart Disease dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from heart_disease.config import load_config
from heart_disease.data import load_raw_data
from heart_disease.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def download_from_kaggle(output_dir: Path) -> Path:
    """Download the Kaggle mirror dataset using kagglehub."""
    try:
        import kagglehub  # type: ignore[import-untyped]
    except ImportError:
        msg = "kagglehub is not installed. Run 'pip install kagglehub' first."
        raise RuntimeError(msg) from None

    logger.info("Downloading redwankarimsony/heart-disease-data via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download("redwankarimsony/heart-disease-data"))
    logger.info("Kaggle dataset downloaded to: %s", dataset_path)

    # Copy the Kaggle CSV into output_dir
    src_file = dataset_path / "heart_disease_uci.csv"
    if src_file.exists():
        import shutil

        dest_file = output_dir / "heart_disease_uci.csv"
        shutil.copy2(src_file, dest_file)
        logger.info("Copied Kaggle dataset to %s", dest_file)
        return dest_file

    return dataset_path


def main() -> None:
    """Download and cache raw UCI heart disease dataset."""
    parser = argparse.ArgumentParser(description="Download heart disease dataset.")
    parser.add_argument(
        "--source",
        choices=["local", "kaggle", "uci"],
        default="local",
        help="Data source: 'local' (database file), 'kaggle' (kagglehub), or 'uci' (ucimlrepo).",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    raw_dir = Path(config.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        download_from_kaggle(raw_dir)
    else:
        logger.info("Loading dataset (source=%s)...", args.source)
        df = load_raw_data(config)
        logger.info(
            "Successfully loaded/cached dataset: %d rows, %d columns",
            len(df),
            len(df.columns),
        )


if __name__ == "__main__":
    main()
