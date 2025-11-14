"""Enhanced prediction helpers that support multiple models and classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ENHANCED_DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_enhanced.csv"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns (exclude Date and target columns)."""
    exclude_cols = ["Date", "Target_Change_%", "Target_Direction"]
    return [col for col in df.columns if col not in exclude_cols]


def load_enhanced_model(model_name: str) -> Union[RegressorMixin, ClassifierMixin]:
    """Load a trained enhanced model.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'rf_regression', 'xgb_classification', 'ensemble').
    
    Returns
    -------
    Model object
        Trained model.
    """
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Available models: {list(MODELS_DIR.glob('*_model.pkl'))}"
        )
    
    return joblib.load(model_path)


def predict_enhanced(
    model_reg: RegressorMixin,
    model_clf: Optional[ClassifierMixin],
    features: pd.DataFrame,
    current_price: Optional[float] = None,
) -> Tuple[float, float, Optional[str]]:
    """Make prediction using enhanced models.
    
    Parameters
    ----------
    model_reg : RegressorMixin
        Regression model for price change prediction.
    model_clf : ClassifierMixin, optional
        Classification model for direction prediction.
    features : pd.DataFrame
        Feature dataframe with all required columns.
    current_price : float, optional
        Current gold price. If None, uses Price from features.
    
    Returns
    -------
    Tuple[float, float, Optional[str]]
        (predicted_change_percent, predicted_price, direction)
        direction is 'Up' or 'Down' or None if classification model not provided.
    """
    # Get feature columns
    feature_cols = get_feature_columns(features)
    
    # Ensure all features are present
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        # Fill missing with 0 (or forward fill if available)
        for col in missing:
            features[col] = 0.0
    
    # Prepare feature array
    X = features[feature_cols].fillna(0)
    
    # Regression prediction
    change_pct = float(model_reg.predict(X.iloc[[-1]])[0])
    
    # Calculate predicted price
    if current_price is None:
        if "Price" in features.columns:
            current_price = float(features.iloc[-1]["Price"])
        else:
            raise ValueError("current_price required if 'Price' not in features")
    
    predicted_price = current_price * (1 + change_pct / 100)
    
    # Classification prediction
    direction = None
    if model_clf is not None:
        direction_pred = model_clf.predict(X.iloc[[-1]])[0]
        direction = "Up" if direction_pred == 1 else "Down"
    
    return change_pct, predicted_price, direction


def predict_with_ensemble(
    features: pd.DataFrame,
    current_price: Optional[float] = None,
    use_classification: bool = True,
) -> Dict[str, Union[float, str]]:
    """Make prediction using ensemble of best models.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature dataframe.
    current_price : float, optional
        Current gold price.
    use_classification : bool
        Whether to include classification prediction.
    
    Returns
    -------
    Dict
        Dictionary with predictions from multiple models.
    """
    results = {}
    
    # Try to load ensemble model first
    try:
        ensemble = load_enhanced_model("ensemble")
        change_pct, price, direction = predict_enhanced(
            ensemble, None, features, current_price
        )
        results["ensemble"] = {
            "predicted_change_percent": change_pct,
            "predicted_price": price,
        }
    except FileNotFoundError:
        pass
    
    # Try Random Forest
    try:
        rf_reg = load_enhanced_model("rf_regression")
        change_pct, price, direction = predict_enhanced(
            rf_reg, None, features, current_price
        )
        results["random_forest"] = {
            "predicted_change_percent": change_pct,
            "predicted_price": price,
        }
        
        if use_classification:
            try:
                rf_clf = load_enhanced_model("rf_classification")
                _, _, direction = predict_enhanced(
                    rf_reg, rf_clf, features, current_price
                )
                results["random_forest"]["direction"] = direction
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    
    # Try XGBoost
    try:
        xgb_reg = load_enhanced_model("xgb_regression")
        change_pct, price, direction = predict_enhanced(
            xgb_reg, None, features, current_price
        )
        results["xgboost"] = {
            "predicted_change_percent": change_pct,
            "predicted_price": price,
        }
        
        if use_classification:
            try:
                xgb_clf = load_enhanced_model("xgb_classification")
                _, _, direction = predict_enhanced(
                    xgb_reg, xgb_clf, features, current_price
                )
                results["xgboost"]["direction"] = direction
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    
    return results


def get_latest_features() -> pd.DataFrame:
    """Get latest features from enhanced dataset for prediction.
    
    Returns
    -------
    pd.DataFrame
        Latest row of features.
    """
    if not ENHANCED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Enhanced data not found at {ENHANCED_DATA_PATH}. "
            "Run preprocess_enhanced first."
        )
    
    df = pd.read_csv(ENHANCED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    # Return last row
    return df.iloc[[-1]]


if __name__ == "__main__":
    # Example usage
    features = get_latest_features()
    results = predict_with_ensemble(features, use_classification=True)
    print("Predictions:", results)

