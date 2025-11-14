# Deployment Workflow - How Training and Processing Run

## Overview

During deployment, the system automatically runs preprocessing and training before starting the API server. Here's how it works:

## Deployment Flow

### 1. Render.com Deployment (render.yaml)

```yaml
buildCommand: pip install -r backend/requirements.txt
preDeployCommand: |
  python -m backend.services.preprocess || echo "Preprocessing failed, continuing..."
  python -m backend.services.trainer --no-tune || echo "Training failed, continuing..."
startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

**Execution Order:**
1. **Build Phase**: Install dependencies from `backend/requirements.txt`
2. **Pre-Deploy Phase**: 
   - Run `preprocess.py` from project root (finds `backend.services.preprocess`)
   - Run `trainer.py` to train the model (without hyperparameter tuning for speed)
3. **Start Phase**: Start the FastAPI server with uvicorn on port from `$PORT` env var

### 2. Docker Deployment (Dockerfile)

```dockerfile
RUN python -m backend.services.preprocess || echo "Preprocessing skipped"
RUN python -m backend.services.trainer --no-tune || echo "Training skipped"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Execution Order:**
1. **Build Phase**: 
   - Install dependencies
   - Copy backend files
   - Create directories (`data/raw`, `data/processed`, `models`)
   - Run preprocessing (non-blocking)
   - Run training (non-blocking)
2. **Runtime**: Start the API server

## How the Scripts Work

### Preprocessing (`backend/services/preprocess.py`)

**Command**: `python -m backend.services.preprocess`

**What it does:**
1. Loads raw gold price data from `backend/data/raw/Gold Futures Historical Data.csv`
2. Creates features:
   - Price differences, moving averages, momentum
   - Volatility measures, EMA indicators
   - Lagged features (previous closes)
3. Calculates target: `Target_Change_%` (next day's price change)
4. Saves to: `backend/data/processed/gold_prices_processed.csv`

**If raw data is missing:**
- Tries to fetch from API (if available)
- If that fails, raises an error (but deployment continues due to `|| echo`)

**Module execution:**
```python
# When you run: python -m backend.services.preprocess
# Python executes the __main__ block:
if __name__ == "__main__":
    processed = preprocess_gold_data()
    print(processed.head())
```

### Training (`backend/services/trainer.py`)

**Command**: `python -m backend.services.trainer --no-tune`

**What it does:**
1. Loads processed data from `backend/data/processed/gold_prices_processed.csv`
2. Splits data: 80% train, 20% test
3. Trains Random Forest model (without hyperparameter tuning for speed)
4. Evaluates on test set (MAE, R²)
5. Saves model to: `backend/models/gold_price_model.pkl`

**Arguments:**
- `--no-tune`: Skip hyperparameter tuning (faster, less optimal)
- `--force-preprocess`: Re-run preprocessing before training

**Module execution:**
```python
# When you run: python -m backend.services.trainer --no-tune
# Python executes the __main__ block:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    train_gold_model(
        force_recompute=args.force_preprocess, 
        enable_tuning=not args.no_tune
    )
```

## Directory Structure

```
project-root/
├── backend/
│   ├── services/
│   │   ├── preprocess.py      # Preprocessing module
│   │   ├── trainer.py         # Training module
│   │   └── ...
│   ├── data/
│   │   ├── raw/               # Input: Raw gold price data
│   │   └── processed/         # Output: Processed features
│   ├── models/                # Output: Trained models
│   └── main.py                # FastAPI app
└── render.yaml                # Render deployment config
```

## Execution Context

### From Project Root (Correct)

```bash
# From: /path/to/gold-price-predictor/
python -m backend.services.preprocess
# ✅ Works: Python finds backend/ as a package
```

### From Backend Directory (Incorrect)

```bash
# From: /path/to/gold-price-predictor/backend/
python -m backend.services.preprocess
# ❌ Fails: Python can't find backend/ module
```

## Error Handling

Both deployment methods use `|| echo` to prevent failures from stopping deployment:

```bash
python -m backend.services.preprocess || echo "Preprocessing failed, continuing..."
```

**This means:**
- If preprocessing succeeds → Continue
- If preprocessing fails → Print message and continue (don't stop deployment)

The API will start even if:
- Raw data is missing
- Preprocessing fails
- Training fails
- Model file doesn't exist

In these cases, the API returns 503 errors for prediction endpoints with helpful messages.

## Manual Execution

You can run these commands manually:

```bash
# From project root
cd /path/to/gold-price-predictor

# Preprocessing
python -m backend.services.preprocess

# Training (fast, no tuning)
python -m backend.services.trainer --no-tune

# Training (slow, with tuning)
python -m backend.services.trainer

# Start API
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Issue: "No module named 'backend'"

**Cause**: Running from wrong directory

**Fix**: Run from project root, not from `backend/` directory

```bash
# Wrong
cd backend
python -m backend.services.preprocess

# Correct
cd /path/to/gold-price-predictor  # Project root
python -m backend.services.preprocess
```

### Issue: "Raw data file not found"

**Cause**: No gold price data in `backend/data/raw/`

**Fix**: Add `Gold Futures Historical Data.csv` to `backend/data/raw/`

### Issue: "Processed dataset is empty"

**Cause**: Not enough data after preprocessing (need at least 14-30 rows)

**Fix**: Add more historical data (see `IMPROVE_ACCURACY.md`)

### Issue: Training takes too long

**Cause**: Hyperparameter tuning enabled

**Fix**: Use `--no-tune` flag for faster training

```bash
python -m backend.services.trainer --no-tune
```

## Summary

1. **Pre-Deploy**: Runs preprocessing and training automatically
2. **Start**: API server starts with uvicorn
3. **Runtime**: API loads model on startup (if available)
4. **Graceful**: API works even if model/data is missing (returns 503 for predictions)

The deployment is designed to be resilient - the API will always start, even if training fails!
