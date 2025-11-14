"""Enhanced feature definitions for the ML pipeline with exogenous features."""

from __future__ import annotations

# Original gold price features
GOLD_FEATURES = [
    "Price",
    "Open",
    "High",
    "Low",
    "Price_Diff",
    "Prev_Close_1",
    "Prev_Close_2",
    "Prev_Close_3",
    "MA_3",
    "MA_5",
    "MA_7",
    "Momentum_3",
    "Momentum_7",
    "Volatility_7",
    "Volatility_14",
    "EMA_7",
    "EMA_14",
    "Daily_Change_%",
]

# Exogenous features
EXOGENOUS_FEATURES = [
    "DXY",
    "SP500",
    "DOW",
    "Bond_Yield_10Y",
    "Oil_Price",
    "Silver_Price",
    "Platinum_Price",
    "GLD_Price",
    "GLD_Volume",
    "CPI",
    "Fed_Rate",
]

# Sentiment features
SENTIMENT_FEATURES = [
    "sentiment_score",
    "sentiment_count",
    "positive_ratio",
]

# All base features (will be expanded with lags, rolling stats, etc. in preprocessing)
BASE_FEATURES = GOLD_FEATURES + EXOGENOUS_FEATURES + SENTIMENT_FEATURES

# Target columns
TARGET_COLUMN = "Target_Change_%"
TARGET_DIRECTION = "Target_Direction"

# Feature categories for analysis
FEATURE_CATEGORIES = {
    "gold_price": GOLD_FEATURES,
    "exogenous": EXOGENOUS_FEATURES,
    "sentiment": SENTIMENT_FEATURES,
}

