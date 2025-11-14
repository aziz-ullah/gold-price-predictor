# Gold Price Predictor Enhancement Guide

This guide explains how to integrate exogenous features, news sentiment analysis, and advanced models into your existing gold price prediction project.

## Overview

The enhancement adds:
1. **Exogenous Features**: Economic indicators, stock indices, bond yields, oil prices, precious metals, ETF flows
2. **News Sentiment Analysis**: Daily sentiment scores from gold-related news
3. **Advanced Models**: LSTM, ARIMA, Prophet, XGBoost, and ensemble methods
4. **Classification Output**: Up/Down prediction in addition to price prediction

## Installation

### 1. Install New Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

**New dependencies:**
- `yfinance` - Fetch market data (DXY, S&P 500, etc.)
- `requests` - API requests
- `beautifulsoup4` - Web scraping for news
- `xgboost` - XGBoost model
- `tensorflow` - LSTM model
- `prophet` - Prophet time series model
- `statsmodels` - ARIMA model
- `vaderSentiment` - Sentiment analysis (optional)
- `textblob` - Sentiment analysis (optional)

### 2. Optional API Keys

For enhanced data fetching, set these environment variables (optional):

```powershell
# For CPI and Fed Rate data (FRED API)
$env:FRED_API_KEY = "your_fred_api_key"

# For news articles (NewsAPI)
$env:NEWSAPI_KEY = "your_newsapi_key"
```

**Getting API Keys:**
- **FRED API**: Free at https://fred.stlouisfed.org/docs/api/api_key.html
- **NewsAPI**: Free tier at https://newsapi.org/

## Step-by-Step Integration

### Step 1: Fetch Exogenous Features

Fetch economic indicators and market data:

```powershell
python -m backend.services.fetch_exogenous
```

This will:
- Fetch DXY (USD Index), S&P 500, Dow Jones
- Fetch bond yields, oil prices, silver, platinum
- Fetch GLD ETF data
- Fetch CPI and Fed Rate (if API key is set)
- Save to `backend/data/exogenous/exogenous_features.csv`

**Expected time**: 2-5 minutes depending on data range

### Step 2: Fetch News Sentiment

Fetch and analyze gold-related news:

```powershell
python -m backend.services.news_sentiment
```

This will:
- Fetch gold-related news articles
- Calculate daily sentiment scores
- Save to `backend/data/sentiment/news_sentiment.csv`

**Expected time**: 5-15 minutes (depends on date range and rate limits)

**Note**: If NewsAPI key is not set, it will fall back to web scraping (slower, less reliable).

### Step 3: Create Enhanced Dataset

Merge gold data with exogenous features and sentiment:

```powershell
python -m backend.services.preprocess_enhanced
```

This will:
- Load gold processed data
- Merge with exogenous features
- Merge with sentiment data
- Create enhanced features (lags, rolling stats, ratios, etc.)
- Save to `backend/data/processed/gold_prices_enhanced.csv`

**Output**: Enhanced dataset with 100+ features

### Step 4: Train Enhanced Models

Train multiple models:

```powershell
# Train all models
python -m backend.services.trainer_enhanced

# Train specific models
python -m backend.services.trainer_enhanced --models rf xgb lstm ensemble

# Skip classification models
python -m backend.services.trainer_enhanced --no-classification
```

**Available models:**
- `rf` - Random Forest (regression + classification)
- `xgb` - XGBoost (regression + classification)
- `lstm` - LSTM neural network
- `prophet` - Prophet time series model
- `arima` - ARIMA time series model
- `ensemble` - Voting ensemble of best models

**Expected time**: 
- Random Forest: 1-2 minutes
- XGBoost: 2-5 minutes
- LSTM: 10-30 minutes (depends on GPU)
- Prophet: 2-5 minutes
- ARIMA: 5-10 minutes
- Ensemble: 1 minute

**Output**: Models saved to `backend/models/` with metrics in `model_metrics.json`

## Integration with Existing Code

### Option 1: Use Enhanced Models in API

Update `backend/main.py` to use enhanced models:

```python
from backend.services.predictor_enhanced import (
    load_enhanced_model,
    predict_enhanced,
)

# In load_resources()
model_reg = load_enhanced_model("rf_regression")
model_clf = load_enhanced_model("rf_classification")

# In prediction endpoint
prediction, direction = predict_enhanced(
    model_reg, model_clf, features_df
)
```

### Option 2: Keep Existing Model (Backward Compatible)

The original model and preprocessing remain unchanged. You can:
- Continue using `backend/services/trainer.py` for the original model
- Use `backend/services/trainer_enhanced.py` for enhanced models
- Both can coexist

## Feature List

### Original Features (from gold data)
- Price, Open, High, Low
- Price_Diff, Prev_Close_1/2/3
- MA_3/5/7, Momentum_3/7
- Volatility_7/14, EMA_7/14
- Daily_Change_%

### New Exogenous Features
- **DXY** - USD Dollar Index
- **SP500** - S&P 500 Index
- **DOW** - Dow Jones Industrial Average
- **Bond_Yield_10Y** - 10-Year Treasury Yield
- **Oil_Price** - WTI Crude Oil
- **Silver_Price** - Silver Futures
- **Platinum_Price** - Platinum Futures
- **GLD_Price** - GLD ETF Price
- **GLD_Volume** - GLD ETF Volume
- **CPI** - Consumer Price Index (monthly)
- **Fed_Rate** - Federal Funds Rate

### Enhanced Features (derived)
- Lagged features (1, 2, 3 days) for all exogenous variables
- Rolling statistics (MA, Std) for all variables
- Cross-feature ratios (Gold/DXY, Gold/Oil, Gold/Silver)
- Volatility measures (7, 14, 30 day)
- Momentum indicators (5, 10, 20 day)
- RSI-14 indicator

### Sentiment Features
- **sentiment_score** - Daily sentiment (-1 to 1)
- **sentiment_count** - Number of articles analyzed
- **positive_ratio** - Ratio of positive articles
- Lagged and rolling sentiment features

## Model Comparison

After training, check `backend/models/model_metrics.json` for performance:

```json
{
  "rf_regression": {"MAE": 0.85, "R2": 0.72},
  "rf_classification": {"Accuracy": 0.68},
  "xgb_regression": {"MAE": 0.82, "R2": 0.75},
  "lstm": {"MAE": 0.88, "R2": 0.70},
  "ensemble": {"MAE": 0.80, "R2": 0.77}
}
```

**Expected improvements:**
- MAE reduction: 20-40% compared to gold-only model
- R² improvement: 0.5 → 0.7-0.8
- Classification accuracy: 60-70% (Up/Down prediction)

## Troubleshooting

### Issue: Missing exogenous data

**Solution**: Run `python -m backend.services.fetch_exogenous` first.

### Issue: News sentiment not working

**Solutions**:
1. Set `NEWSAPI_KEY` environment variable
2. Or wait for web scraping (slower)
3. Or skip sentiment: models will work without it

### Issue: LSTM training fails

**Solutions**:
1. Install TensorFlow: `pip install tensorflow`
2. Reduce sequence length in `train_lstm_model()`
3. Use CPU version: `pip install tensorflow-cpu`

### Issue: Prophet/ARIMA errors

**Solutions**:
1. Ensure sufficient data (at least 1 year)
2. Check for missing dates in data
3. Models will skip automatically if dependencies missing

### Issue: Memory errors

**Solutions**:
1. Reduce date range when fetching data
2. Train models one at a time
3. Use `--models rf xgb` to skip memory-intensive models

## Data Update Workflow

To keep models up-to-date:

```powershell
# 1. Fetch latest exogenous data (weekly/monthly)
python -m backend.services.fetch_exogenous --start-date 2024-01-01

# 2. Fetch latest sentiment (daily/weekly)
python -m backend.services.news_sentiment --start-date 2024-01-01

# 3. Recreate enhanced dataset
python -m backend.services.preprocess_enhanced

# 4. Retrain models
python -m backend.services.trainer_enhanced --models rf xgb ensemble
```

## File Structure

```
backend/
├── services/
│   ├── fetch_exogenous.py          # NEW: Fetch economic indicators
│   ├── news_sentiment.py            # NEW: News sentiment analysis
│   ├── preprocess_enhanced.py       # NEW: Enhanced preprocessing
│   ├── trainer_enhanced.py          # NEW: Multi-model training
│   ├── predictor_enhanced.py        # NEW: Enhanced prediction (to be created)
│   ├── preprocess.py                # EXISTING: Original preprocessing
│   └── trainer.py                   # EXISTING: Original trainer
├── data/
│   ├── raw/                         # Gold price data
│   ├── processed/                   # Processed datasets
│   ├── exogenous/                   # NEW: Exogenous features
│   └── sentiment/                   # NEW: Sentiment data
└── models/                          # Trained models
    ├── gold_price_model.pkl         # Original model
    ├── rf_regression_model.pkl      # NEW: Random Forest
    ├── rf_classification_model.pkl  # NEW: Classification
    └── model_metrics.json           # NEW: Performance metrics
```

## Next Steps

1. **Create predictor_enhanced.py**: Update prediction logic to use enhanced models
2. **Update API endpoints**: Add classification output to `/predict` endpoint
3. **Add model selection**: Allow choosing which model to use via API parameter
4. **Add feature importance**: Show which features matter most
5. **Add backtesting**: Test model performance on historical data

## Performance Tips

1. **Start small**: Test with 1 year of data first
2. **Use ensemble**: Usually performs best
3. **Update regularly**: Retrain monthly for best accuracy
4. **Monitor metrics**: Check `model_metrics.json` to choose best model
5. **Feature selection**: Remove low-importance features to speed up training

## Support

For issues or questions:
1. Check `model_metrics.json` for model performance
2. Verify data files exist in `backend/data/`
3. Check console output for warnings/errors
4. Ensure all dependencies are installed

