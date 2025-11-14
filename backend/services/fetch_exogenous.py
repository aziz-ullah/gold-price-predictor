"""Fetch exogenous features for gold price prediction.

This module fetches external economic indicators, market indices, and other
factors that influence gold prices.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
EXOGENOUS_DATA_DIR = BASE_DIR / "data" / "exogenous"


def fetch_dxy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch USD Dollar Index (DXY) data from Yahoo Finance."""
    try:
        ticker = yf.Ticker("DX-Y.NYB")  # DXY ticker
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            # Try alternative ticker
            ticker = yf.Ticker("DX=F")
            df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "DXY"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch DXY data: {e}")
        return pd.DataFrame(columns=["Date", "DXY"])


def fetch_sp500_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch S&P 500 index data."""
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "SP500"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch S&P 500 data: {e}")
        return pd.DataFrame(columns=["Date", "SP500"])


def fetch_dow_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Dow Jones Industrial Average data."""
    try:
        ticker = yf.Ticker("^DJI")
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "DOW"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch Dow Jones data: {e}")
        return pd.DataFrame(columns=["Date", "DOW"])


def fetch_bond_yield_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch US 10-Year Treasury Bond Yield."""
    try:
        ticker = yf.Ticker("^TNX")  # 10-Year Treasury Note
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "Bond_Yield_10Y"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch bond yield data: {e}")
        return pd.DataFrame(columns=["Date", "Bond_Yield_10Y"])


def fetch_oil_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch WTI Crude Oil prices."""
    try:
        ticker = yf.Ticker("CL=F")  # WTI Crude Oil Futures
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "Oil_Price"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch oil data: {e}")
        return pd.DataFrame(columns=["Date", "Oil_Price"])


def fetch_silver_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Silver prices."""
    try:
        ticker = yf.Ticker("SI=F")  # Silver Futures
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "Silver_Price"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch silver data: {e}")
        return pd.DataFrame(columns=["Date", "Silver_Price"])


def fetch_platinum_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Platinum prices."""
    try:
        ticker = yf.Ticker("PL=F")  # Platinum Futures
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close"]].rename(columns={"Close": "Platinum_Price"})
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch platinum data: {e}")
        return pd.DataFrame(columns=["Date", "Platinum_Price"])


def fetch_gld_etf_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch GLD (Gold ETF) data."""
    try:
        ticker = yf.Ticker("GLD")
        df = ticker.history(start=start_date, end=end_date)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Close", "Volume"]].rename(
            columns={"Close": "GLD_Price", "Volume": "GLD_Volume"}
        )
        return df
    except Exception as e:
        print(f"[!] Warning: Could not fetch GLD ETF data: {e}")
        return pd.DataFrame(columns=["Date", "GLD_Price", "GLD_Volume"])


def fetch_cpi_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch CPI (Consumer Price Index) data from FRED API.
    
    Note: Requires FRED API key. Set FRED_API_KEY environment variable.
    Falls back to monthly data approximation if API unavailable.
    """
    try:
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            print("[!] Warning: FRED_API_KEY not set. Skipping CPI data.")
            return None
        
        # FRED API endpoint
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CPIAUCSL",  # CPI for All Urban Consumers
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": "m",  # Monthly
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "observations" not in data:
            return None
        
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"date": "Date", "value": "CPI"})
        df["Date"] = df["Date"].dt.date
        
        # Forward fill monthly data to daily
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df_daily = pd.DataFrame({"Date": date_range.date})
        df_daily = df_daily.merge(df, on="Date", how="left")
        df_daily["CPI"] = df_daily["CPI"].ffill()
        
        return df_daily[["Date", "CPI"]]
    except Exception as e:
        print(f"[!] Warning: Could not fetch CPI data: {e}")
        return None


def fetch_fed_rate_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch Federal Reserve Interest Rate data from FRED API.
    
    Note: Requires FRED API key. Set FRED_API_KEY environment variable.
    """
    try:
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            print("[!] Warning: FRED_API_KEY not set. Skipping Fed Rate data.")
            return None
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "FEDFUNDS",  # Federal Funds Rate
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": "d",
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "observations" not in data:
            return None
        
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"date": "Date", "value": "Fed_Rate"})
        df["Date"] = df["Date"].dt.date
        
        return df[["Date", "Fed_Rate"]]
    except Exception as e:
        print(f"[!] Warning: Could not fetch Fed Rate data: {e}")
        return None


def fetch_all_exogenous_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_to_file: bool = True,
) -> pd.DataFrame:
    """Fetch all exogenous features and merge them into a single DataFrame.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 8 years ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    save_to_file : bool
        Whether to save the merged data to a CSV file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Date and all exogenous features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=8 * 365)).strftime("%Y-%m-%d")
    
    print(f"[*] Fetching exogenous data from {start_date} to {end_date}...")
    
    # Fetch all data sources
    dataframes = []
    
    print("  - Fetching DXY (USD Index)...")
    df_dxy = fetch_dxy_data(start_date, end_date)
    if not df_dxy.empty:
        dataframes.append(df_dxy)
    time.sleep(0.5)  # Rate limiting
    
    print("  - Fetching S&P 500...")
    df_sp500 = fetch_sp500_data(start_date, end_date)
    if not df_sp500.empty:
        dataframes.append(df_sp500)
    time.sleep(0.5)
    
    print("  - Fetching Dow Jones...")
    df_dow = fetch_dow_data(start_date, end_date)
    if not df_dow.empty:
        dataframes.append(df_dow)
    time.sleep(0.5)
    
    print("  - Fetching Bond Yields...")
    df_bond = fetch_bond_yield_data(start_date, end_date)
    if not df_bond.empty:
        dataframes.append(df_bond)
    time.sleep(0.5)
    
    print("  - Fetching Oil prices...")
    df_oil = fetch_oil_data(start_date, end_date)
    if not df_oil.empty:
        dataframes.append(df_oil)
    time.sleep(0.5)
    
    print("  - Fetching Silver prices...")
    df_silver = fetch_silver_data(start_date, end_date)
    if not df_silver.empty:
        dataframes.append(df_silver)
    time.sleep(0.5)
    
    print("  - Fetching Platinum prices...")
    df_platinum = fetch_platinum_data(start_date, end_date)
    if not df_platinum.empty:
        dataframes.append(df_platinum)
    time.sleep(0.5)
    
    print("  - Fetching GLD ETF...")
    df_gld = fetch_gld_etf_data(start_date, end_date)
    if not df_gld.empty:
        dataframes.append(df_gld)
    time.sleep(0.5)
    
    print("  - Fetching CPI data...")
    df_cpi = fetch_cpi_data(start_date, end_date)
    if df_cpi is not None and not df_cpi.empty:
        dataframes.append(df_cpi)
    
    print("  - Fetching Fed Rate data...")
    df_fed = fetch_fed_rate_data(start_date, end_date)
    if df_fed is not None and not df_fed.empty:
        dataframes.append(df_fed)
    
    # Merge all dataframes on Date
    if not dataframes:
        print("[!] Warning: No exogenous data fetched!")
        return pd.DataFrame()
    
    print("[*] Merging exogenous data...")
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on="Date", how="outer")
    
    # Sort by date
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    
    # Forward fill missing values (for monthly data like CPI)
    merged_df = merged_df.ffill()
    
    if save_to_file:
        EXOGENOUS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXOGENOUS_DATA_DIR / "exogenous_features.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"[OK] Exogenous data saved to {output_path}")
        print(f"     Total rows: {len(merged_df)}, Columns: {list(merged_df.columns)}")
    
    return merged_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch exogenous features for gold prediction")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    fetch_all_exogenous_data(
        start_date=args.start_date,
        end_date=args.end_date,
    )

