"""Helpers for accessing local historical price data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Gold Futures Historical Data.csv"


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw historical CSV from disk."""

    csv_path = Path(path) if path else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {csv_path}. Place your CSV inside backend/data/raw."
        )
    return pd.read_csv(csv_path)
