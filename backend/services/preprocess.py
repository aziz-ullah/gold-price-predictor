"""Data preprocessing utilities for the gold price predictor project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = BASE_DIR / "data" / "raw" / "Gold Futures Historical Data.csv"
DEFAULT_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "gold_prices_processed.csv"


def preprocess_gold_data(
    raw_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Transform the raw historical CSV into a feature-rich dataset.

    Parameters
    ----------
    raw_path:
        Location of the raw CSV file. Defaults to ``backend/data/raw``.
    save_path:
        Where to write the processed CSV. Defaults to ``backend/data/processed``.

    Returns
    -------
    pandas.DataFrame
        Processed dataset containing engineered features and prediction target.
    """

    raw_csv = Path(raw_path) if raw_path else DEFAULT_RAW_PATH
    processed_csv = Path(save_path) if save_path else DEFAULT_PROCESSED_PATH

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_csv}. Place a CSV in backend/data/raw."
        )

    df = pd.read_csv(raw_csv)
    print(f"Loaded {len(df)} rows from {raw_csv}")

    df.columns = [col.strip().replace(".", "").replace(" ", "_") for col in df.columns]

    rename_map = {
        "Last": "Price",
        "Price_": "Price",
        "Vol": "Vol",
        "Vol_": "Vol",
        "Change": "Change_%",
    }
    df = df.rename(columns=rename_map)

    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Change_%" in df.columns:
        df["Change_%"] = df["Change_%"].astype(str).str.replace("%", "", regex=True)
        df["Change_%"] = pd.to_numeric(df["Change_%"], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    if "Open" not in df.columns:
        df["Open"] = df["Price"].shift(1).bfill()

    df["Price_Diff"] = df["Price"] - df["Open"]

    for lag in (1, 2, 3):
        df[f"Prev_Close_{lag}"] = df["Price"].shift(lag)

    for window in (3, 5, 7, 14, 30):
        df[f"MA_{window}"] = df["Price"].rolling(window=window).mean()

    df["Momentum_3"] = df["Price"] - df["Price"].shift(3)
    df["Momentum_7"] = df["Price"] - df["Price"].shift(7)

    df["Volatility_7"] = df["Price"].rolling(window=7).std()
    df["Volatility_14"] = df["Price"].rolling(window=14).std()

    df["EMA_7"] = df["Price"].ewm(span=7, adjust=False).mean()
    df["EMA_14"] = df["Price"].ewm(span=14, adjust=False).mean()

    df["Daily_Change_%"] = df["Price"].pct_change() * 100
    df["Target_Change_%"] = df["Price"].pct_change().shift(-1) * 100

    # Only drop rows where required features are missing (not all features)
    from backend.services.features import FEATURE_COLUMNS, TARGET_COLUMN
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    print(
        f"Processed dataset saved to {processed_csv} with {len(df)} rows and {len(df.columns)} columns"
    )

    return df


def ensure_processed_exists() -> Path:
    """Ensure the processed dataset exists on disk and return its path."""

    processed_csv = DEFAULT_PROCESSED_PATH
    if not processed_csv.exists():
        # If raw data doesn't exist, try to fetch some initial data
        if not DEFAULT_RAW_PATH.exists():
            try:
                from backend.services.fetch_today import fetch_and_process
                print("⚠️ Raw data file not found. Attempting to fetch initial data...")
                fetch_and_process()
                # If fetch succeeded, try preprocessing again
                if DEFAULT_RAW_PATH.exists():
                    preprocess_gold_data()
                else:
                    raise FileNotFoundError(
                        f"Raw data file not found at {DEFAULT_RAW_PATH}. "
                        "Place a CSV in backend/data/raw or ensure the API is accessible."
                    )
            except Exception as e:
                raise FileNotFoundError(
                    f"Raw data file not found at {DEFAULT_RAW_PATH}. "
                    f"Failed to fetch initial data: {e}. "
                    "Please place a CSV file in backend/data/raw."
                )
        else:
            preprocess_gold_data()
    return processed_csv


if __name__ == "__main__":
    processed = preprocess_gold_data()
    print(processed.head())
