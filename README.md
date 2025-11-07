# Gold Price Predictor

End-to-end machine learning project that transforms historical gold prices into
predictions for the next trading day. The stack is entirely offline and built
with FastAPI, scikit-learn, pandas, and a React dashboard.

## Project Structure

```
backend/
  main.py                # FastAPI application
  requirements.txt       # Python dependencies
  services/
    preprocess.py        # Raw -> processed feature engineering
    trainer.py           # Model training helpers
    fetch_today.py       # Latest processed row utilities
    predictor.py         # Shared prediction helpers
    features.py          # Feature definition constants
    fetch_data.py        # Raw data loader
  data/
    raw/                 # Place historical CSV here
    processed/           # Generated feature dataset
  models/                # Persisted scikit-learn models

frontend/
  react-app/             # React dashboard (Create React App)

run_project.bat          # Starts backend and frontend in separate terminals
```

## 1. Environment Setup

```powershell
cd backend
python -m venv venv            # optional but recommended
venv\Scripts\activate
pip install -r requirements.txt
```

Install frontend dependencies once:

```powershell
cd ..\frontend\react-app
npm install
```

## 2. Data Preparation

1. Place your historical CSV into `backend/data/raw/` and name it
   `Gold Futures Historical Data.csv` (or provide the path when calling the
   preprocessing function).
2. Run the preprocessing script to engineer features and targets:

   ```powershell
   cd backend
   python -m backend.services.preprocess
   ```

   The processed dataset is written to `backend/data/processed/`.

## 3. Model Training

Train the RandomForest regression model (with time-series hyperparameter search by default) and persist it with joblib:

```powershell
cd backend
python -m backend.services.trainer
```

Tuning uses a five-fold walk-forward split and may take several minutes on the first run. Console output includes the best parameter set, cross-validated MAE, and holdout MAE/R². The trained model is written to `backend/models/gold_price_model.pkl`.

For a quicker run you can skip tuning:

```powershell
python -m backend.services.trainer --no-tune
```

Use `--force-preprocess` if you need to regenerate the processed dataset before training.

## 4. Deploying for Free

### Backend (Render)
1. Fork/clone this repo and keep `render.yaml` at the project root.
2. In the Render dashboard, choose **New → Web Service**, pick this GitHub repo, and leave the branch on `main`.
3. Render reads `render.yaml`, installs dependencies, runs preprocessing + a quick training pass, and starts FastAPI with Uvicorn on the free tier.
4. After the first deployment, copy the public URL (e.g. `https://gold-predictor-api.onrender.com`).

When the raw data changes, push to `main` and Render will redeploy automatically. To speed things up you can temporarily drop the `preDeployCommand` in `render.yaml` or switch to `--no-tune`.

### Frontend (Vercel)
1. Visit [https://vercel.com/new](https://vercel.com/new) and import the repo.
2. Framework preset: **Create React App**. Build command: `npm install && npm run build`. Output directory: `frontend/react-app/build` (Vercel detects this automatically).
3. Add an environment variable `REACT_APP_API_BASE_URL` pointing to the Render backend URL.
4. Deploy and grab the Vercel URL; the dashboard will auto-refresh against the hosted API.

(Any other static host—Netlify, Cloudflare Pages, GitHub Pages—works as long as you set `REACT_APP_API_BASE_URL` to the backend URL.)

## 5. Run the Application

Launch both FastAPI and React via the batch script from the project root:

```powershell
run_project.bat
```

Prefer to start services manually? From the project root run:

```powershell
uvicorn backend.main:app --reload
```

and in another terminal:

```powershell
cd frontend\react-app
npm start
```

Services start on:

- Backend: <http://127.0.0.1:8000>
- Frontend: <http://localhost:3000>

## 5. API Overview

- `GET /` – Health and welcome message.
- `POST /predict` – Provide feature values (JSON keys follow the processed CSV)
  and receive the predicted percentage change and price for the next day.
- `GET /predict/latest` – Loads the last processed row and returns the
  prediction.
- `GET /history?limit=30` – Returns `limit` recent sessions for plotting.

## 6. React Dashboard

The dashboard automatically fetches the latest prediction and a rolling history
of recent sessions. A refresh button triggers both endpoints, and a line chart
visualises the recent trend.

> Configure a different backend URL by setting `REACT_APP_API_BASE_URL` before
> running `npm start`.

## 7. Retraining Workflow

When new historical data becomes available:

1. Append it to the raw CSV in `backend/data/raw/`.
2. Run the preprocessing script again to rebuild the processed dataset.
3. Retrain the model using the trainer module.

The FastAPI service reads the processed CSV on startup, so restarting the
backend is enough to serve updated predictions.

