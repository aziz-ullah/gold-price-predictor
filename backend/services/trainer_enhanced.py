"""Enhanced model training with multiple algorithms and classification output.

Supports:
- Random Forest, XGBoost (ML models)
- LSTM (Deep Learning)
- ARIMA, Prophet (Time Series)
- Ensemble methods
- Classification (Up/Down) in addition to regression
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
ENHANCED_DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_enhanced.csv"
MODELS_DIR = BASE_DIR / "models"


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude Date and target columns)."""
    exclude_cols = ["Date", "Target_Change_%", "Target_Direction"]
    return [col for col in df.columns if col not in exclude_cols]


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "regression",
) -> Tuple[object, Dict[str, float]]:
    """Train XGBoost model."""
    try:
        import xgboost as xgb
        
        if task == "regression":
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {"MAE": mae, "R2": r2}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {"Accuracy": accuracy}
        
        return model, metrics
    except ImportError:
        print("[!] XGBoost not installed. Skipping XGBoost model.")
        return None, {}
    except Exception as e:
        print(f"[!] XGBoost training error: {e}")
        return None, {}


def train_lstm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sequence_length: int = 10,
) -> Tuple[Optional[object], Dict[str, float]]:
    """Train LSTM model for time series prediction."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Prepare sequences
        def create_sequences(data, targets, seq_len):
            X_seq, y_seq = [], []
            for i in range(seq_len, len(data)):
                X_seq.append(data[i - seq_len:i])
                y_seq.append(targets.iloc[i])
            return np.array(X_seq), np.array(y_seq)
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(
            pd.DataFrame(X_train_scaled), pd.Series(y_train_scaled), sequence_length
        )
        X_test_seq, y_test_seq = create_sequences(
            pd.DataFrame(X_test_scaled), pd.Series(y_test_scaled), sequence_length
        )
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1),
        ])
        
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        
        # Train
        model.fit(
            X_train_seq,
            y_train_seq,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0,
        )
        
        # Predict
        y_pred_scaled = model.predict(X_test_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        mae = mean_absolute_error(y_test.iloc[sequence_length:], y_pred)
        r2 = r2_score(y_test.iloc[sequence_length:], y_pred)
        
        # Store scalers with model
        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
        model.sequence_length = sequence_length
        
        return model, {"MAE": mae, "R2": r2}
    except ImportError:
        print("[!] TensorFlow not installed. Skipping LSTM model.")
        return None, {}
    except Exception as e:
        print(f"[!] LSTM training error: {e}")
        return None, {}


def train_prophet_model(
    df: pd.DataFrame,
    train_end_idx: int,
) -> Tuple[Optional[object], Dict[str, float]]:
    """Train Prophet model (requires Date and Price columns)."""
    try:
        from prophet import Prophet
        
        if "Date" not in df.columns or "Price" not in df.columns:
            print("[!] Prophet requires Date and Price columns")
            return None, {}
        
        # Prepare data for Prophet
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:].copy()
        
        prophet_data = train_df[["Date", "Price"]].copy()
        prophet_data.columns = ["ds", "y"]
        prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])
        
        # Train Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(prophet_data)
        
        # Predict
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        
        # Calculate metrics
        y_pred = forecast["yhat"].iloc[train_end_idx:].values
        y_test = test_df["Price"].values
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, {"MAE": mae, "R2": r2}
    except ImportError:
        print("[!] Prophet not installed. Skipping Prophet model.")
        return None, {}
    except Exception as e:
        print(f"[!] Prophet training error: {e}")
        return None, {}


def train_arima_model(
    df: pd.DataFrame,
    train_end_idx: int,
) -> Tuple[Optional[object], Dict[str, float]]:
    """Train ARIMA model."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        if "Price" not in df.columns:
            print("[!] ARIMA requires Price column")
            return None, {}
        
        train_data = df["Price"].iloc[:train_end_idx].values
        
        # Auto ARIMA (simplified - use fixed order for speed)
        model = ARIMA(train_data, order=(5, 1, 2))
        fitted_model = model.fit()
        
        # Predict
        test_data = df["Price"].iloc[train_end_idx:].values
        y_pred = fitted_model.forecast(steps=len(test_data))
        
        mae = mean_absolute_error(test_data, y_pred)
        r2 = r2_score(test_data, y_pred)
        
        return fitted_model, {"MAE": mae, "R2": r2}
    except ImportError:
        print("[!] statsmodels not installed. Skipping ARIMA model.")
        return None, {}
    except Exception as e:
        print(f"[!] ARIMA training error: {e}")
        return None, {}


def train_enhanced_models(
    data_path: Optional[Path] = None,
    models_to_train: Optional[List[str]] = None,
    enable_classification: bool = True,
) -> Dict[str, object]:
    """Train multiple models and return trained models with metrics.
    
    Parameters
    ----------
    data_path : Path, optional
        Path to enhanced processed data.
    models_to_train : List[str], optional
        List of models to train: ['rf', 'xgb', 'lstm', 'prophet', 'arima', 'ensemble'].
        If None, trains all available models.
    enable_classification : bool
        Whether to also train classification models.
    
    Returns
    -------
    Dict[str, object]
        Dictionary of trained models with their metrics.
    """
    if models_to_train is None:
        models_to_train = ["rf", "xgb", "lstm", "ensemble"]
    
    data_file = Path(data_path) if data_path else ENHANCED_DATA_PATH
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Enhanced data not found at {data_file}. "
            "Run preprocess_enhanced first."
        )
    
    print(f"[*] Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Get features
    feature_cols = get_feature_columns(df)
    print(f"[*] Using {len(feature_cols)} features")
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train_reg = train_df["Target_Change_%"]
    y_test_reg = test_df["Target_Change_%"]
    
    if enable_classification and "Target_Direction" in df.columns:
        y_train_clf = train_df["Target_Direction"]
        y_test_clf = test_df["Target_Direction"]
    else:
        y_train_clf = None
        y_test_clf = None
    
    print(f"[*] Training on {len(train_df)} samples, testing on {len(test_df)} samples")
    
    trained_models = {}
    model_metrics = {}
    
    # Train Random Forest (Regression)
    if "rf" in models_to_train:
        print("\n[*] Training Random Forest (Regression)...")
        rf_reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        rf_reg.fit(X_train, y_train_reg)
        y_pred = rf_reg.predict(X_test)
        
        mae = mean_absolute_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)
        model_metrics["rf_regression"] = {"MAE": mae, "R2": r2}
        trained_models["rf_regression"] = rf_reg
        print(f"     MAE: {mae:.4f}%, R2: {r2:.4f}")
        
        # Classification
        if enable_classification and y_train_clf is not None:
            print("[*] Training Random Forest (Classification)...")
            rf_clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            rf_clf.fit(X_train, y_train_clf)
            y_pred_clf = rf_clf.predict(X_test)
            accuracy = accuracy_score(y_test_clf, y_pred_clf)
            model_metrics["rf_classification"] = {"Accuracy": accuracy}
            trained_models["rf_classification"] = rf_clf
            print(f"     Accuracy: {accuracy:.4f}")
    
    # Train XGBoost
    if "xgb" in models_to_train:
        print("\n[*] Training XGBoost (Regression)...")
        xgb_model, xgb_metrics = train_xgboost_model(
            X_train, y_train_reg, X_test, y_test_reg, task="regression"
        )
        if xgb_model:
            trained_models["xgb_regression"] = xgb_model
            model_metrics["xgb_regression"] = xgb_metrics
            print(f"     {xgb_metrics}")
        
        if enable_classification and y_train_clf is not None:
            print("[*] Training XGBoost (Classification)...")
            xgb_clf, xgb_clf_metrics = train_xgboost_model(
                X_train, y_train_clf, X_test, y_test_clf, task="classification"
            )
            if xgb_clf:
                trained_models["xgb_classification"] = xgb_clf
                model_metrics["xgb_classification"] = xgb_clf_metrics
                print(f"     {xgb_clf_metrics}")
    
    # Train LSTM
    if "lstm" in models_to_train:
        print("\n[*] Training LSTM...")
        lstm_model, lstm_metrics = train_lstm_model(
            X_train, y_train_reg, X_test, y_test_reg
        )
        if lstm_model:
            trained_models["lstm"] = lstm_model
            model_metrics["lstm"] = lstm_metrics
            print(f"     {lstm_metrics}")
    
    # Train Prophet
    if "prophet" in models_to_train:
        print("\n[*] Training Prophet...")
        prophet_model, prophet_metrics = train_prophet_model(df, split_idx)
        if prophet_model:
            trained_models["prophet"] = prophet_model
            model_metrics["prophet"] = prophet_metrics
            print(f"     {prophet_metrics}")
    
    # Train ARIMA
    if "arima" in models_to_train:
        print("\n[*] Training ARIMA...")
        arima_model, arima_metrics = train_arima_model(df, split_idx)
        if arima_model:
            trained_models["arima"] = arima_model
            model_metrics["arima"] = arima_metrics
            print(f"     {arima_metrics}")
    
    # Ensemble
    if "ensemble" in models_to_train and len(trained_models) > 1:
        print("\n[*] Creating Ensemble...")
        reg_models = [
            (name, model) for name, model in trained_models.items()
            if "regression" in name or name == "lstm"
        ]
        
        if len(reg_models) >= 2:
            ensemble = VotingRegressor(reg_models[:3])  # Use top 3
            ensemble.fit(X_train, y_train_reg)
            y_pred_ens = ensemble.predict(X_test)
            
            mae_ens = mean_absolute_error(y_test_reg, y_pred_ens)
            r2_ens = r2_score(y_test_reg, y_pred_ens)
            model_metrics["ensemble"] = {"MAE": mae_ens, "R2": r2_ens}
            trained_models["ensemble"] = ensemble
            print(f"     MAE: {mae_ens:.4f}%, R2: {r2_ens:.4f}")
    
    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Saving models to {MODELS_DIR}...")
    
    for name, model in trained_models.items():
        model_path = MODELS_DIR / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"     Saved {name} to {model_path}")
    
    # Save metrics
    metrics_path = MODELS_DIR / "model_metrics.json"
    import json
    # Convert numpy types to Python types for JSON
    metrics_json = {
        k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
            for kk, vv in v.items()}
        for k, v in model_metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"     Saved metrics to {metrics_path}")
    
    print("\n[OK] Model training complete!")
    print("\nModel Performance Summary:")
    for name, metrics in model_metrics.items():
        print(f"  {name}: {metrics}")
    
    return trained_models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train enhanced models")
    parser.add_argument("--data", type=str, help="Path to enhanced processed data")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "xgb", "lstm", "prophet", "arima", "ensemble"],
        help="Models to train",
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Skip classification models",
    )
    args = parser.parse_args()
    
    train_enhanced_models(
        data_path=Path(args.data) if args.data else None,
        models_to_train=args.models,
        enable_classification=not args.no_classification,
    )

