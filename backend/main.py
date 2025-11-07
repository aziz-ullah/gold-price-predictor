"""FastAPI application exposing prediction endpoints for the gold model."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # Support both pydantic v1 and v2
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - v1 compatibility
    ConfigDict = None

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.exchange import get_usd_to_pkr_rate
from backend.services.fetch_today import load_processed_dataframe
from backend.services.preprocess import ensure_processed_exists
from backend.services.predictor import load_trained_model, predict_tomorrow_price


BASE_DIR = CURRENT_FILE.parent
MODEL_PATH = BASE_DIR / "models" / "gold_price_model.pkl"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "gold_prices_processed.csv"


class FeaturePayload(BaseModel):
    Price: float = Field(..., description="Most recent closing price")
    Open: float
    High: float
    Low: float
    Price_Diff: float
    Prev_Close_1: float
    Prev_Close_2: float
    Prev_Close_3: float
    MA_3: float
    MA_5: float
    MA_7: float
    Momentum_3: float
    Momentum_7: float
    Volatility_7: float
    Volatility_14: float
    EMA_7: float
    EMA_14: float
    Daily_Change_pct: float = Field(..., alias="Daily_Change_%")

    if ConfigDict is None:

        class Config:  # pragma: no cover - exercised in pydantic v1 only
            allow_population_by_field_name = True
            anystr_strip_whitespace = True

    else:  # pragma: no cover - exercised in pydantic v2 only
        model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)


app = FastAPI(title="Gold Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None


@app.on_event("startup")
def load_resources() -> None:
    """Load the trained model and ensure the processed data is available."""

    global model
    ensure_processed_exists()
    if MODEL_PATH.exists():
        model = load_trained_model(MODEL_PATH)
    else:
        model = None


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Welcome to the Gold Price Predictor API"}


def _predict_from_dataframe(feature_row: pd.DataFrame) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not trained. Run the trainer first."
        )

    prediction, tomorrow_price = predict_tomorrow_price(model, feature_row)

    return {
        "predicted_change_percent": round(prediction, 4),
        "predicted_tomorrow_price": round(tomorrow_price, 2),
        "success_probability": _calculate_success_probability(prediction),
    }


def _calculate_success_probability(change_percent: float) -> float:
    baseline = 85.0
    penalty = min(40.0, abs(change_percent) * 6)
    return round(max(40.0, baseline - penalty), 2)


@app.post("/predict")
def predict(payload: FeaturePayload) -> Dict[str, Any]:
    feature_row = pd.DataFrame([payload.dict(by_alias=True)])
    return _predict_from_dataframe(feature_row)


@app.get("/predict/latest")
def predict_latest() -> Dict[str, Any]:
    if not PROCESSED_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed dataset not found. Run preprocessing first.",
        )

    df = load_processed_dataframe()
    latest = df.iloc[-1:]
    result = _predict_from_dataframe(latest)

    date_value = latest["Date"].iloc[0] if "Date" in latest else None
    if pd.isna(date_value):
        dataset_date = None
    else:
        dataset_date = pd.to_datetime(date_value).date()

    today_display = datetime.utcnow().date()
    tomorrow_display = today_display + timedelta(days=1)

    usd_to_pkr = get_usd_to_pkr_rate()

    return {
        "date": dataset_date.isoformat() if dataset_date else None,
        "today_date": today_display.isoformat(),
        "tomorrow_date": tomorrow_display.isoformat(),
        "today_price": round(float(latest["Price"].iloc[0]), 2),
        **result,
        "usd_to_pkr_rate": round(usd_to_pkr, 4),
    }


@app.get("/history")
def price_history(limit: int = 30) -> Dict[str, Any]:
    if limit <= 0:
        raise HTTPException(status_code=400, detail="Limit must be greater than zero.")

    if not PROCESSED_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed dataset not found. Run preprocessing first.",
        )

    df = load_processed_dataframe()
    history_df = df.tail(limit)

    records = [
        {
            "date": (
                pd.to_datetime(row["Date"]).date().isoformat()
                if not pd.isna(row["Date"])
                else None
            ),
            "price": round(float(row["Price"]), 2),
        }
        for _, row in history_df.iterrows()
    ]

    return {"data": records}
