# 🌫️ Karachi AQI Predictor

A **fully automated, cloud-ready Air Quality Index (AQI) forecasting system** for Karachi, Pakistan.
Predicts AQI for the **next 24 h, 48 h, and 72 h** using an end-to-end ML pipeline with hourly
automated data collection and daily model retraining — all driven by **GitHub Actions**.

---

## Architecture

```
OpenWeather API (Air Pollution + Weather)
        │
        ▼  (every hour via GitHub Actions)
  Feature Pipeline
  ├── fetch_openweather.py   — fetch & store raw data
  └── feature_engineering.py — compute lag/rolling/time features
        │
        ▼
  MongoDB Atlas (Feature Store)
  ├── raw_data collection    — raw API records
  ├── features collection    — engineered ML features
  └── model_registry collection — model metadata
        │
        ▼  (every day via GitHub Actions)
  Training Pipeline (train.py)
  ├── LinearRegression
  ├── RandomForestRegressor
  ├── GradientBoostingRegressor
  └── XGBRegressor
        │
        ▼
  Best Model saved as  models/best_model_<target>_<timestamp>.pkl
        │
        ▼
  Prediction Pipeline (predict.py)
  → 24h / 48h / 72h AQI forecasts
        │
        ▼
  Streamlit Dashboard (app/streamlit_app.py)
  → Real-time visualisation + SHAP explanations
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Data Source | OpenWeather Air Pollution API + Open-Meteo (historical weather fallback) |
| Feature Store | MongoDB Atlas |
| ML Models | scikit-learn (LR, RF, GB) + XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| CI/CD Automation | GitHub Actions |
| Model Serialisation | joblib |

---

## Project Structure

```
AQI-Predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly: fetch + feature engineering
│       └── training_pipeline.yml   # Daily:  train + save best model
├── app/
│   └── streamlit_app.py            # Interactive dashboard
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
├── src/
│   ├── config.py                   # Environment / settings loader
│   ├── database.py                 # MongoDB helper functions
│   ├── fetch_openweather.py        # API fetch + raw storage
│   ├── feature_engineering.py      # Lag, rolling, time features
│   ├── backfill.py                 # Historical data backfill (90 days)
│   ├── train.py                    # Training pipeline (4 models × 3 targets)
│   ├── predict.py                  # Inference pipeline
│   ├── model_registry.py           # Save / load / list models
│   └── utils.py                    # Shared utilities
├── models/                         # Saved .pkl model files (gitignored)
├── data/                           # Local data exports (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Dataset & Features

**Source:** OpenWeather [Air Pollution API](https://openweathermap.org/api/air-pollution)
and Open-Meteo historical weather archive.

**Raw fields stored per hour:**

| Field | Description |
|---|---|
| `aqi` | OpenWeather AQI index (1 Good → 5 Very Poor) |
| `pm2_5` | Fine particulate matter (μg/m³) |
| `pm10` | Coarse particulate matter (μg/m³) |
| `no2` | Nitrogen dioxide (μg/m³) |
| `o3` | Ozone (μg/m³) |
| `so2` | Sulphur dioxide (μg/m³) |
| `co` | Carbon monoxide (μg/m³) |
| `nh3` | Ammonia (μg/m³) |
| `temperature` | 2 m air temperature (°C) |
| `humidity` | Relative humidity (%) |
| `pressure` | Surface pressure (hPa) |
| `wind_speed` | 10 m wind speed (m/s) |
| `clouds` | Cloud cover (%) |

**Engineered features:**

| Feature | Description |
|---|---|
| `hour`, `day`, `month`, `weekday` | Time-of-day / calendar features |
| `is_weekend` | Binary weekend flag |
| `aqi_lag_1`, `aqi_lag_24`, `aqi_lag_48` | AQI 1h, 24h, 48h ago |
| `pm25_lag_24`, `pm10_lag_24` | PM lags |
| `aqi_rolling_24_mean` | 24-hour rolling mean AQI |
| `pm25_rolling_24_mean` | 24-hour rolling mean PM2.5 |
| `aqi_change_rate` | % change from 24h ago |

**Target variables:**
- `target_aqi_24h` — AQI 24 hours from now
- `target_aqi_48h` — AQI 48 hours from now
- `target_aqi_72h` — AQI 72 hours from now

---

## Model Training

Four models are trained for **each** of the three forecast horizons (12 total runs):

| Model | Notes |
|---|---|
| `LinearRegression` | With StandardScaler preprocessing |
| `RandomForestRegressor` | 200 trees, max_depth=12 |
| `GradientBoostingRegressor` | 200 estimators, lr=0.05 |
| `XGBRegressor` | 200 estimators, lr=0.05 |

**Train/test split:** time-ordered, last 20% held out — no random shuffling
to prevent look-ahead bias from lag features.

**Best model** (lowest RMSE) is saved for each horizon.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average magnitude of prediction error |
| RMSE | Root Mean Squared Error — penalises large errors more heavily |
| R² | Coefficient of determination — proportion of variance explained |

---

## Automation (GitHub Actions)

### Hourly Feature Pipeline
- Runs at `:00` every hour
- Fetches latest air pollution + weather for Karachi
- Recomputes engineered features
- Stores everything in MongoDB Atlas

### Daily Training Pipeline
- Runs at 02:00 UTC daily
- Loads all features from MongoDB
- Trains 4 models × 3 horizons
- Saves best models as GitHub Actions artifacts (30-day retention)
- Records metadata in MongoDB

---

## Quick Start — Local Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/AQI-Predictor.git
cd AQI-Predictor
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in your API key and MongoDB URI
```

### 3. Backfill historical data (run once)

```bash
python src/backfill.py
```

This fetches 90 days of hourly AQI + weather data and engineers features.

### 4. Train models

```bash
python src/train.py
```

### 5. Run predictions

```bash
python src/predict.py
```

### 6. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## GitHub Actions Secrets Setup

In your GitHub repository go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name | Value |
|---|---|
| `OPENWEATHER_API_KEY` | Your OpenWeather API key |
| `MONGODB_URI` | Your MongoDB Atlas connection string |
| `DB_NAME` | `aqi_predictor` (or your chosen DB name) |

---

## Deploy Streamlit Dashboard

### Option A — Streamlit Community Cloud (free)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Add your secrets under **Advanced settings → Secrets** in TOML format:

```toml
OPENWEATHER_API_KEY = "your_key"
MONGODB_URI = "your_uri"
DB_NAME = "aqi_predictor"
```

### Option B — Docker / Cloud Run / Railway

Build the Docker image and deploy to any container platform:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## EDA Notebook

Open `notebooks/01_eda.ipynb` in Jupyter or VS Code. Run all cells after completing the backfill step.

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebook covers:
- AQI trend over time
- PM2.5 & PM10 trends (with WHO guideline references)
- AQI by hour of day and weekday
- Correlation heatmap
- Pollutant vs AQI scatter plots
- Lag & rolling feature inspection
- Target distribution analysis

---

## Screenshots

> _Add screenshots of your running dashboard here._

| Dashboard | AQI Trend | SHAP Importance |
|---|---|---|
| ![Dashboard]() | ![Trend]() | ![SHAP]() |

---

## Future Improvements

- [ ] Add LSTM / temporal transformer model for sequence forecasting
- [ ] Integrate AQICN API as a secondary data source and validation check
- [ ] Add email/SMS alerts when predicted AQI exceeds threshold
- [ ] Multi-city support (Lahore, Islamabad, etc.)
- [ ] Serve predictions via a FastAPI REST endpoint
- [ ] Deploy to Vertex AI or Hopsworks as a managed feature store
- [ ] Add drift detection to trigger automated retraining

---

## License

MIT License — free to use, modify, and distribute.
