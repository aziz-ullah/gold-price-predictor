"""Shared configuration such as feature definitions for the ML pipeline."""

from __future__ import annotations

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "Target_Change_%"
