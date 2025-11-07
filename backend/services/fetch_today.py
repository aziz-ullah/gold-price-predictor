"""Fetch the latest gold futures data and refresh the processed dataset."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from backend.services.preprocess import ensure_processed_exists


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "Gold Futures Historical Data.csv"
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "gold_prices_processed.csv"
API_URL = (
    "https://api.investing.com/api/financialdata/historical/8830"
    "?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false"
)
HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


def _utc_today() -> datetime:
    return datetime.utcnow()


def fetch_latest_payload(date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Fetch the most recent daily candle from the Investing.com API."""

    target_date = date or _utc_today()
    start = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
    end = target_date.strftime("%Y-%m-%d")
    url = API_URL.format(start=start, end=end)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"⚠️ Failed to fetch data: {exc}")
        return None

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        print(f"⚠️ Unable to parse API response: {exc}")
        return None

    rows = payload.get("data") or []
    if not rows:
        print("ℹ️ API returned no data for the requested window.")
        return None

    return rows[0]


def load_raw_dataframe() -> pd.DataFrame:
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)

    columns = ["Date", "Price", "Open", "High", "Low", "Vol."]
    return pd.DataFrame(columns=columns)


def load_processed_dataframe() -> pd.DataFrame:
    processed_path = ensure_processed_exists()
    return pd.read_csv(processed_path, parse_dates=["Date"])


def append_row_if_missing(raw_df: pd.DataFrame, payload: Dict[str, Any]) -> bool:
    target_date = payload.get("rowDateTimestamp", "")[:10]
    if not target_date:
        print("⚠️ Missing date in API payload.")
        return False

    if not raw_df.empty and str(raw_df.iloc[-1]["Date"])[:10] == target_date:
        print("ℹ️ Today's data already exists. Skipping append.")
        return False

    new_row = {
        "Date": target_date,
        "Price": float(payload.get("last_closeRaw", 0.0)),
        "Open": float(payload.get("last_openRaw", 0.0)),
        "High": float(payload.get("last_maxRaw", 0.0)),
        "Low": float(payload.get("last_minRaw", 0.0)),
        "Vol.": payload.get("volumeRaw", ""),
    }

    updated_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(RAW_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Appended raw data for {target_date}")
    return True


def run_preprocess() -> bool:
    try:
        subprocess.run(
            ["python", "backend/services/preprocess.py"],
            check=True,
            cwd=PROJECT_ROOT,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ Preprocessing failed: {exc}")
        return False


def fetch_and_process() -> None:
    payload = fetch_latest_payload()
    if payload is None:
        return

    raw_df = load_raw_dataframe()
    updated = append_row_if_missing(raw_df, payload)
    if not updated:
        return

    if run_preprocess():
        print("✅ Today’s gold data fetched and processed successfully!")


if __name__ == "__main__":
    fetch_and_process()
