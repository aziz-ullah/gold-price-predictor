"""Fetch historical gold price data in bulk to improve model accuracy."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "Gold Futures Historical Data.csv"
API_URL = (
    "https://api.investing.com/api/financialdata/historical/8830"
    "?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false"
)
HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


def fetch_date_range(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Fetch historical data for a date range."""
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    url = API_URL.format(start=start_str, end=end_str)

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") or []
        return rows
    except requests.RequestException as exc:
        print(f"[!] Failed to fetch data for {start_str} to {end_str}: {exc}")
        return []
    except json.JSONDecodeError as exc:
        print(f"[!] Unable to parse API response: {exc}")
        return []


def convert_payload_to_row(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert API payload to CSV row format."""
    date_str = payload.get("rowDateTimestamp", "")
    if not date_str:
        return None

    # Extract date (first 10 characters: YYYY-MM-DD)
    date_str = date_str[:10]

    return {
        "Date": date_str,
        "Price": float(payload.get("last_closeRaw", 0.0)),
        "Open": float(payload.get("last_openRaw", 0.0)),
        "High": float(payload.get("last_maxRaw", 0.0)),
        "Low": float(payload.get("last_minRaw", 0.0)),
        "Vol.": payload.get("volumeRaw", ""),
        "Change %": f"{payload.get('last_changePercentRaw', 0.0):.2f}%",
    }


def load_existing_data() -> pd.DataFrame:
    """Load existing raw data if it exists."""
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        # Convert Date column to datetime for comparison
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    return pd.DataFrame()


def merge_and_save(new_rows: List[Dict[str, Any]], existing_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows with existing data, avoiding duplicates."""
    if not new_rows:
        return existing_df

    new_df = pd.DataFrame(new_rows)
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")

    if existing_df.empty:
        merged_df = new_df
    else:
        # Remove duplicates based on Date, keeping existing data
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["Date"], keep="first")
        merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    # Convert Date back to string format for CSV
    merged_df["Date"] = merged_df["Date"].dt.strftime("%m/%d/%Y")
    
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(RAW_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    
    return merged_df


def fetch_historical_data(
    days_back: int = 365,
    chunk_size_days: int = 90,
    delay_seconds: float = 1.0,
) -> None:
    """Fetch historical gold price data in chunks.
    
    Parameters
    ----------
    days_back : int
        How many days of historical data to fetch (default: 365 = 1 year)
    chunk_size_days : int
        Size of each API request in days (default: 90 to avoid rate limits)
    delay_seconds : float
        Delay between API requests to avoid rate limiting
    """
    print(f"[*] Fetching {days_back} days of historical gold price data...")
    print(f"    Using chunks of {chunk_size_days} days with {delay_seconds}s delay\n")

    existing_df = load_existing_data()
    if not existing_df.empty:
        print(f"[*] Found {len(existing_df)} existing rows")
        # Find the earliest date in existing data
        earliest_existing = existing_df["Date"].min()
        if pd.notna(earliest_existing):
            print(f"    Earliest existing date: {earliest_existing.strftime('%Y-%m-%d')}\n")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    all_new_rows = []
    current_start = start_date

    chunk_num = 1
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size_days), end_date)
        
        print(f"[Chunk {chunk_num}] Fetching {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...", end=" ")
        
        rows = fetch_date_range(current_start, current_end)
        
        if rows:
            converted_rows = [convert_payload_to_row(row) for row in rows]
            converted_rows = [r for r in converted_rows if r is not None]
            all_new_rows.extend(converted_rows)
            print(f"Got {len(converted_rows)} rows")
        else:
            print("No data")
        
        current_start = current_end + timedelta(days=1)
        chunk_num += 1
        
        # Delay to avoid rate limiting
        if current_start < end_date:
            time.sleep(delay_seconds)

    if all_new_rows:
        print(f"\n[*] Merging {len(all_new_rows)} new rows with existing data...")
        merged_df = merge_and_save(all_new_rows, existing_df)
        print(f"[OK] Saved {len(merged_df)} total rows to {RAW_CSV}")
        print(f"     Date range: {merged_df['Date'].iloc[0]} to {merged_df['Date'].iloc[-1]}")
    else:
        print("\n[!] No new data fetched. Check your internet connection and API availability.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch historical gold price data to improve model accuracy"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365 = 1 year)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=90,
        help="Days per API request chunk (default: 90)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    fetch_historical_data(
        days_back=args.days,
        chunk_size_days=args.chunk_size,
        delay_seconds=args.delay,
    )

