"""Model training utilities for the gold price predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import numpy as np

from backend.services.features import FEATURE_COLUMNS, TARGET_COLUMN
from backend.services.preprocess import ensure_processed_exists, preprocess_gold_data


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_processed.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "gold_price_model.pkl"


def load_processed_dataset(data_path: Path | None = None) -> Tuple[pd.DataFrame, Path]:
    """Load the processed dataset, running preprocessing on demand if missing."""

    processed_path = Path(data_path) if data_path else ensure_processed_exists()
    df = pd.read_csv(processed_path)
    return df, processed_path


def _tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, float], float]:
    """Run a randomized hyperparameter search using time-series CV."""

    base_estimator = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    param_distributions = {
        "n_estimators": np.arange(200, 801, 100),
        "max_depth": [6, 8, 10, 12, 16, None],
        "max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": np.arange(1, 6),
        "bootstrap": [True, False],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        cv=tscv,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_mae = -search.best_score_

    return best_model, best_params, best_mae


def train_gold_model(
    data_path: Path | None = None,
    model_path: Path | None = None,
    force_recompute: bool = False,
    enable_tuning: bool = True,
) -> Tuple[RandomForestRegressor, float, float]:
    """Train the regression model and persist it alongside evaluation metrics."""

    processed_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if force_recompute or not processed_path.exists():
        preprocess_gold_data(save_path=processed_path)

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed dataset is empty. Check the raw data source.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Processed data is missing required features: {missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Processed data does not contain target column '{TARGET_COLUMN}'."
        )

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Filtered dataset is empty after dropping NA rows. Adjust preprocessing."
        )

    split_index = int(len(df) * 0.8)
    if split_index == 0 or split_index == len(df):
        raise ValueError(
            "Not enough data to create train/test split. Provide more historical rows."
        )

    X_train, X_test = (
        df.iloc[:split_index][FEATURE_COLUMNS],
        df.iloc[split_index:][FEATURE_COLUMNS],
    )
    y_train, y_test = (
        df.iloc[:split_index][TARGET_COLUMN],
        df.iloc[split_index:][TARGET_COLUMN],
    )

    if enable_tuning:
        print("➡️  Running hyperparameter search (this can take a few minutes)...")
        model, best_params, cv_mae = _tune_random_forest(X_train, y_train)
        print("Best cross-validated MAE: {:.4f}%".format(cv_mae))
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_output_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    print("Model training complete")
    print(f"Samples: train={len(X_train)}, test={len(X_test)}")
    print(f"MAE: {mae:.4f}%")
    print(f"R2: {r2:.4f}")
    print(f"Model saved to {model_output_path}")

    return model, mae, r2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the gold price prediction model"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning and use default RandomForest settings",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Recompute the processed dataset before training",
    )
    args = parser.parse_args()

    train_gold_model(
        force_recompute=args.force_preprocess, enable_tuning=not args.no_tune
    )
