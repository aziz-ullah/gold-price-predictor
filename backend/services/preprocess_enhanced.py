"""Enhanced preprocessing that integrates gold data with exogenous features and sentiment."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from backend.services.preprocess import preprocess_gold_data, DEFAULT_PROCESSED_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
EXOGENOUS_DATA_PATH = BASE_DIR / "data" / "exogenous" / "exogenous_features.csv"
SENTIMENT_DATA_PATH = BASE_DIR / "data" / "sentiment" / "news_sentiment.csv"
ENHANCED_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "gold_prices_enhanced.csv"


def load_exogenous_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load exogenous features data."""
    data_path = Path(path) if path else EXOGENOUS_DATA_PATH
    
    if not data_path.exists():
        print(f"[!] Warning: Exogenous data not found at {data_path}")
        print("    Run: python -m backend.services.fetch_exogenous")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def load_sentiment_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load news sentiment data."""
    data_path = Path(path) if path else SENTIMENT_DATA_PATH
    
    if not data_path.exists():
        print(f"[!] Warning: Sentiment data not found at {data_path}")
        print("    Run: python -m backend.services.news_sentiment")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from exogenous and sentiment data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with gold, exogenous, and sentiment data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with enhanced features.
    """
    # Lagged features for exogenous variables
    exogenous_cols = [
        "DXY", "SP500", "DOW", "Bond_Yield_10Y", "Oil_Price",
        "Silver_Price", "Platinum_Price", "GLD_Price", "GLD_Volume",
        "CPI", "Fed_Rate",
    ]
    
    for col in exogenous_cols:
        if col in df.columns:
            # Lag 1, 2, 3 days
            for lag in [1, 2, 3]:
                df[f"{col}_Lag_{lag}"] = df[col].shift(lag)
            
            # Percentage change
            df[f"{col}_Change_%"] = df[col].pct_change() * 100
            
            # Rolling statistics
            for window in [3, 7, 14]:
                df[f"{col}_MA_{window}"] = df[col].rolling(window=window).mean()
                df[f"{col}_Std_{window}"] = df[col].rolling(window=window).std()
    
    # Sentiment features
    if "sentiment_score" in df.columns:
        # Lagged sentiment
        for lag in [1, 2, 3, 7]:
            df[f"sentiment_score_Lag_{lag}"] = df["sentiment_score"].shift(lag)
        
        # Rolling sentiment
        for window in [3, 7, 14]:
            df[f"sentiment_MA_{window}"] = df["sentiment_score"].rolling(window=window).mean()
    
    # Cross-feature interactions
    if "DXY" in df.columns and "Price" in df.columns:
        df["Gold_DXY_Ratio"] = df["Price"] / (df["DXY"] + 1e-6)
        df["Gold_DXY_Correlation_7"] = df["Price"].rolling(7).corr(df["DXY"].rolling(7))
    
    if "Oil_Price" in df.columns and "Price" in df.columns:
        df["Gold_Oil_Ratio"] = df["Price"] / (df["Oil_Price"] + 1e-6)
    
    if "Silver_Price" in df.columns and "Price" in df.columns:
        df["Gold_Silver_Ratio"] = df["Price"] / (df["Silver_Price"] + 1e-6)
    
    # Volatility features
    if "Price" in df.columns:
        df["Price_Volatility_7"] = df["Price"].rolling(7).std()
        df["Price_Volatility_14"] = df["Price"].rolling(14).std()
        df["Price_Volatility_30"] = df["Price"].rolling(30).std()
    
    # Momentum features
    if "Price" in df.columns:
        for window in [5, 10, 20]:
            df[f"Price_Momentum_{window}"] = (
                df["Price"] - df["Price"].shift(window)
            ) / df["Price"].shift(window) * 100
    
    # RSI-like indicator
    if "Price" in df.columns:
        delta = df["Price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        df["RSI_14"] = 100 - (100 / (1 + rs))
    
    return df


def preprocess_enhanced(
    gold_data_path: Optional[Path] = None,
    exogenous_data_path: Optional[Path] = None,
    sentiment_data_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Preprocess gold data with exogenous features and sentiment.
    
    Parameters
    ----------
    gold_data_path : Path, optional
        Path to gold processed data. If None, uses default.
    exogenous_data_path : Path, optional
        Path to exogenous features CSV.
    sentiment_data_path : Path, optional
        Path to sentiment data CSV.
    save_path : Path, optional
        Where to save the enhanced processed data.
    
    Returns
    -------
    pd.DataFrame
        Enhanced processed dataset.
    """
    print("[*] Starting enhanced preprocessing...")
    
    # Load gold data
    print("  - Loading gold data...")
    if gold_data_path:
        df_gold = pd.read_csv(gold_data_path)
    else:
        # Use existing preprocessing
        df_gold = preprocess_gold_data()
    
    df_gold["Date"] = pd.to_datetime(df_gold["Date"]).dt.date
    
    # Load exogenous data
    print("  - Loading exogenous features...")
    df_exogenous = load_exogenous_data(exogenous_data_path)
    
    # Load sentiment data
    print("  - Loading sentiment data...")
    df_sentiment = load_sentiment_data(sentiment_data_path)
    
    # Merge all data
    print("  - Merging datasets...")
    df_merged = df_gold.copy()
    
    if not df_exogenous.empty:
        df_merged = df_merged.merge(df_exogenous, on="Date", how="left")
        print(f"     Merged {len(df_exogenous.columns) - 1} exogenous features")
    
    if not df_sentiment.empty:
        df_merged = df_merged.merge(df_sentiment, on="Date", how="left")
        print(f"     Merged {len(df_sentiment.columns) - 1} sentiment features")
    
    # Create enhanced features
    print("  - Creating enhanced features...")
    df_enhanced = create_enhanced_features(df_merged)
    
    # Ensure target column exists
    if "Target_Change_%" not in df_enhanced.columns:
        if "Price" in df_enhanced.columns:
            df_enhanced["Target_Change_%"] = df_enhanced["Price"].pct_change().shift(-1) * 100
    
    # Create classification target (Up/Down)
    if "Target_Change_%" in df_enhanced.columns:
        df_enhanced["Target_Direction"] = (df_enhanced["Target_Change_%"] > 0).astype(int)
    
    # Forward fill missing values (for monthly data like CPI)
    print("  - Handling missing values...")
    numeric_cols = df_enhanced.select_dtypes(include=["float64", "int64"]).columns
    df_enhanced[numeric_cols] = df_enhanced[numeric_cols].ffill().bfill()
    
    # Drop rows where target is missing
    df_enhanced = df_enhanced.dropna(subset=["Target_Change_%"]).reset_index(drop=True)
    
    # Save
    output_path = Path(save_path) if save_path else ENHANCED_PROCESSED_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enhanced.to_csv(output_path, index=False)
    
    print(f"[OK] Enhanced dataset saved to {output_path}")
    print(f"     Total rows: {len(df_enhanced)}, Columns: {len(df_enhanced.columns)}")
    print(f"     Date range: {df_enhanced['Date'].min()} to {df_enhanced['Date'].max()}")
    
    return df_enhanced


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess gold data with exogenous features")
    parser.add_argument("--gold-data", type=str, help="Path to gold processed data")
    parser.add_argument("--exogenous-data", type=str, help="Path to exogenous features")
    parser.add_argument("--sentiment-data", type=str, help="Path to sentiment data")
    parser.add_argument("--output", type=str, help="Output path")
    args = parser.parse_args()
    
    preprocess_enhanced(
        gold_data_path=Path(args.gold_data) if args.gold_data else None,
        exogenous_data_path=Path(args.exogenous_data) if args.exogenous_data else None,
        sentiment_data_path=Path(args.sentiment_data) if args.sentiment_data else None,
        save_path=Path(args.output) if args.output else None,
    )

