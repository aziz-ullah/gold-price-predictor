# How to Improve Model Accuracy

Your current model has only **10 rows** of training data, which is far too little for accurate predictions. For good accuracy, you need **at least 200-500 rows** (ideally 1-2 years of daily data).

## Quick Start: Fetch More Historical Data

### Option 1: Automated Fetch (Recommended)

Run the historical data fetcher to get 1 year of data:

```powershell
python -m backend.services.fetch_historical --days 365
```

This will:
- Fetch 365 days of historical gold price data
- Merge it with your existing data (avoiding duplicates)
- Save it to `backend/data/raw/Gold Futures Historical Data.csv`

**For even better accuracy, fetch 2 years:**

```powershell
python -m backend.services.fetch_historical --days 730
```

### Option 2: Manual Data Addition

1. Download historical gold price data from:
   - [Investing.com](https://www.investing.com/commodities/gold-historical-data)
   - [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF/history/)
   - [Alpha Vantage](https://www.alphavantage.co/) (requires free API key)

2. Export as CSV with columns: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`

3. Merge with your existing file at `backend/data/raw/Gold Futures Historical Data.csv`

## Step-by-Step: Complete Process

### 1. Fetch Historical Data

```powershell
# Fetch 1 year of data (recommended minimum)
python -m backend.services.fetch_historical --days 365

# Or fetch 2 years for better accuracy
python -m backend.services.fetch_historical --days 730
```

**Note:** The script fetches data in chunks to avoid rate limiting. It may take a few minutes.

### 2. Reprocess the Data

After fetching more data, regenerate the processed dataset:

```powershell
python -m backend.services.preprocess
```

You should see many more rows now (e.g., 200-500+ instead of 10).

### 3. Retrain the Model with Hyperparameter Tuning

Train with full hyperparameter tuning for best accuracy:

```powershell
python -m backend.services.trainer
```

**Note:** This will take 5-15 minutes depending on your data size, but will give much better results.

If you want faster training (but less optimal):

```powershell
python -m backend.services.trainer --no-tune
```

### 4. Verify the Results

Check the training output. You should see:
- **MAE (Mean Absolute Error)**: Lower is better (aim for < 1.0%)
- **R² Score**: Higher is better (aim for > 0.5, ideally > 0.7)

Good model metrics example:
```
MAE: 0.85%
R²: 0.72
```

## Expected Results by Data Size

| Data Size | Expected MAE | Expected R² | Training Time |
|-----------|--------------|-------------|---------------|
| 10 rows (current) | 2-5% | Negative | < 1 min |
| 100 rows | 1.5-2.5% | 0.3-0.5 | 2-5 min |
| 200 rows | 1.0-1.5% | 0.5-0.7 | 5-10 min |
| 500+ rows | 0.8-1.2% | 0.7-0.85 | 10-20 min |

## Troubleshooting

### API Rate Limiting

If you get errors, increase the delay between requests:

```powershell
python -m backend.services.fetch_historical --days 365 --delay 2.0
```

### Not Enough Data After Fetching

- Check your internet connection
- The API might be temporarily unavailable
- Try fetching in smaller chunks: `--chunk-size 30`

### Model Still Not Accurate

1. **Get more data**: Fetch 2 years instead of 1
2. **Check data quality**: Ensure dates are correct and prices are valid
3. **Retrain with tuning**: Always use `python -m backend.services.trainer` (without `--no-tune`)

## Advanced: Continuous Data Updates

To keep your model accurate over time, periodically fetch new data:

```powershell
# Fetch just the latest data (adds to existing)
python -m backend.services.fetch_today

# Then retrain
python -m backend.services.preprocess
python -m backend.services.trainer
```

## Summary

**Minimum for decent accuracy:**
- 200+ rows of historical data
- Full hyperparameter tuning
- Regular updates

**For best accuracy:**
- 500+ rows (1-2 years)
- Full hyperparameter tuning
- Weekly data updates

