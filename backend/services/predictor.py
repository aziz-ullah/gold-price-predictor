"""Prediction helpers shared across the API and CLI utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.base import RegressorMixin

from backend.services.features import FEATURE_COLUMNS


def load_trained_model(model_path: Path) -> RegressorMixin:
    """Load a previously trained scikit-learn model from disk."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def predict_change(model: RegressorMixin, features: pd.DataFrame) -> float:
    """Predict the percentage change for the next day."""

    if features.empty:
        raise ValueError("No feature rows supplied for prediction")

    missing = [col for col in FEATURE_COLUMNS if col not in features.columns]
    if missing:
        raise ValueError(f"Feature dataframe missing columns: {missing}")

    prediction = model.predict(features[FEATURE_COLUMNS])[0]
    return float(prediction)


def predict_tomorrow_price(
    model: RegressorMixin, features: pd.DataFrame
) -> Tuple[float, float]:
    """Return both the predicted percent change and tomorrow's projected price."""

    change_pct = predict_change(model, features)
    latest_price = float(features.iloc[-1]["Price"])
    tomorrow_price = latest_price * (1 + change_pct / 100)
    return change_pct, tomorrow_price
