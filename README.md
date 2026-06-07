# Karachi AQI Predictor

Automated **pollutant forecasting + AQI calculation** for Karachi, Pakistan. The system ingests hourly pollution and weather data, engineers temporal features in MongoDB Atlas, trains **12 pollutant forecasting models** (4 pollutants × 3 horizons), and serves EPA-style 3-day AQI forecasts through a public Streamlit dashboard.

> **Detailed project report:** The full internship submission (architecture diagrams, EDA figures, model evaluation, automation, challenges) is at the repository root: **[final_report.pdf](final_report.pdf)**

**Detailed report:** [final_report.pdf](final_report.pdf)  
**Live dashboard:** [aqipredictorbyrashidhussain.streamlit.app](https://aqipredictorbyrashidhussain.streamlit.app/)  
**EDA notebook:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

---

## Overview

Karachi faces recurring air-quality stress from traffic, industry, construction dust, and seasonal weather. OpenWeather exposes a coarse **1–5 AQI category**, not a continuous EPA 0–500 score. This project therefore:

1. Forecasts **PM2.5, PM10, O₃, and NO₂** at **+24 h, +48 h, and +72 h**
2. Converts each concentration to an EPA **sub-index** via breakpoint interpolation
3. Reports **final AQI = max(sub-indices)** with the **dominant pollutant**


| Horizon | Output                                       |
| ------- | -------------------------------------------- |
| +24 h   | Predicted pollutant concentrations → EPA AQI |
| +48 h   | Predicted pollutant concentrations → EPA AQI |
| +72 h   | Predicted pollutant concentrations → EPA AQI |


---

## Architecture

Serverless end-to-end pipeline: OpenWeather + Open-Meteo → MongoDB Atlas → GitHub Actions → Hugging Face Hub → Streamlit.

```
OpenWeather API (+ Open-Meteo for historical weather)
        │
        ▼
Feature Pipeline (GitHub Actions — hourly via cron-job.org)
  hourly_pipeline.py
  ├── fetch_openweather.py
  ├── feature_engineering.py (incremental)
  └── catch-up (≤48 h gap fill)
        │
        ▼
MongoDB Atlas
  ├── raw_data          — hourly API records
  ├── features          — engineered rows (+ targets in batch rows)
  ├── predictions       — forecast documents (optional)
  └── model_registry    — active model metadata + HF paths
        │
        ▼
Training Pipeline (GitHub Actions — daily)
  train.py — 12 targets × 4 algorithms → best model per target
        │
        ▼
Hugging Face Model Registry
  models/<target>/<timestamp>/{model.pkl, metadata.json}
        │
        ▼
Inference (predict.py)
  12 regressors → EPA AQI via aqi_utils.py
        │
        ▼
Streamlit Dashboard
```


| Layer          | Technology                                        |
| -------------- | ------------------------------------------------- |
| Language       | Python 3.11                                       |
| Data sources   | OpenWeather Air Pollution API; Open-Meteo archive |
| Feature store  | MongoDB Atlas                                     |
| ML             | scikit-learn (Ridge, RF, GB), XGBoost             |
| Model registry | Hugging Face Hub + MongoDB metadata               |
| Explainability | SHAP                                              |
| Dashboard      | Streamlit, Plotly                                 |
| Automation     | GitHub Actions + cron-job.org                     |
| Serialisation  | joblib                                            |


---

## Dataset


| Metric                | Value                          |
| --------------------- | ------------------------------ |
| Raw MongoDB records   | 2,091                          |
| Feature rows          | 2,089                          |
| Training-ready rows   | 1,971                          |
| City                  | Karachi (24.8607°N, 67.0011°E) |
| Frequency             | Hourly                         |
| Historical window     | ~90 days                       |
| Prediction targets    | 12                             |
| Forecast horizons     | 24 h, 48 h, 72 h               |
| Pollutants forecasted | PM2.5, PM10, O₃, NO₂           |


Training-ready rows contain supervised target values generated during batch feature engineering. During training, each model uses rows where its specific target is available.

**Bootstrap:** `backfill.py` loads ~90 days of OpenWeather pollution history + Open-Meteo weather, upserts `raw_data`, then batch-engineers `features` with all 12 target columns.

**Live updates:** The feature pipeline runs after every 2 hours (cron-job.org → GitHub Actions). Each run optionally catches up missing hours (≤48 h), fetches the current snapshot, and upserts one incremental feature row.

**Raw fields:** PM2.5, PM10, O₃, NO₂, CO, SO₂, NH₃, OpenWeather `aqi_category` (1–5), plus temperature, humidity, pressure, wind speed, cloud cover.

---

## AQI calculation

```
ML prediction (per pollutant, per horizon)
   PM2.5, PM10, O3, NO2  @ +24h / +48h / +72h
              │
              ▼
   EPA breakpoint interpolation (aqi_utils.py)
              │
              ▼
   Final AQI = max(sub_indices)
   dominant_pollutant = argmax(sub_indices)
```

Breakpoint formula for concentration `C` in range `[C_low, C_high]` → AQI `[I_low, I_high]`:

```
AQI = ((I_high - I_low) / (C_high - C_low)) × (C - C_low) + I_low
```

Negative predicted concentrations are clipped to zero before sub-index calculation. PM2.5 drives computed AQI on ~73% of hours in EDA; PM10 on ~15%.

---

## Feature engineering

Two modes in `src/feature_engineering.py`:


| Mode            | Used by              | Targets                                 |
| --------------- | -------------------- | --------------------------------------- |
| **Batch**       | `backfill.py`        | 12 supervised target columns included   |
| **Incremental** | `hourly_pipeline.py` | Targets omitted (future values unknown) |


**Calendar:** `hour`, `day`, `month`, `weekday`, `is_weekend`

**Per pollutant (PM2.5, PM10, O₃, NO₂):**


| Type          | Features                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------- |
| Lags          | `{pollutant}_lag_1`, `_lag_24`, `_lag_48`                                                       |
| Rolling means | `{pollutant}_rolling_6_mean`, `_rolling_12_mean`, `_rolling_24_mean` (shifted to avoid leakage) |
| Momentum      | `{pollutant}_change_rate = (current − lag_24) / lag_24`                                         |


**Targets (batch rows):** `target_{pollutant}_{24|48|72}h` — 12 columns total.

---

## EDA highlights

Full analysis with all plots: `[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)`. Key findings:


| Insight                     | Implication                                          |
| --------------------------- | ---------------------------------------------------- |
| PM pollutants dominate AQI  | Prioritise PM2.5 / PM10 in modelling                 |
| Strong temporal persistence | Lag + rolling features essential                     |
| Diurnal morning peaks       | `hour` feature validated                             |
| Episodic spikes             | Compare regularized linear models and tree ensembles |
| Time-based split mandatory  | Random CV would leak future lag values               |


---

## Model training

**12 pollutant forecasting models** = 4 pollutants × 3 horizons.


| Pollutant            | Horizons            |
| -------------------- | ------------------- |
| PM2.5, PM10, O₃, NO₂ | +24 h, +48 h, +72 h |


**Algorithms evaluated per target (48 experiments total):**


| Algorithm                           | Notes                                |
| ----------------------------------- | ------------------------------------ |
| Ridge Regression (+ StandardScaler) | Linear baseline with correlated lags |
| Random Forest                       | Shallow ensemble, subsampling        |
| Gradient Boosting                   | Regularised boosting                 |
| XGBoost                             | L1/L2 penalties, tuned depth         |


Since the dataset contains ~2,000 hourly samples, **classical ML** was chosen over deep learning — tree-based and regularized linear models suit small-to-medium tabular data; deep learning typically needs much larger datasets.


| Setting          | Value                                                               |
| ---------------- | ------------------------------------------------------------------- |
| Train/test split | Time-ordered **80 / 20** (no shuffle)                               |
| Label filter     | Per-target `.dropna()`                                              |
| Overfitting rule | Flag when **both** test/train RMSE ratio > 1.5 **and** R² gap > 0.4 |
| Selection        | Prefer non-overfitting models; else lowest test RMSE                |


Winning models are saved locally, uploaded to Hugging Face, and registered in MongoDB with `active=True`.

---

## Model evaluation

Active models from the latest training run (`model_registry`):


| Target             | Best Model        | Test RMSE | Test MAE | Test R² |
| ------------------ | ----------------- | --------- | -------- | ------- |
| `target_pm2_5_24h` | Random Forest     | 5.3007    | 4.2833   | −0.4967 |
| `target_pm2_5_48h` | Random Forest     | 5.5129    | 3.9727   | −0.1322 |
| `target_pm2_5_72h` | Gradient Boosting | 4.8749    | 3.6338   | 0.3961  |
| `target_pm10_24h`  | Ridge Regression  | 38.3592   | 28.9942  | −1.7050 |
| `target_pm10_48h`  | Ridge Regression  | 35.5178   | 28.2349  | −1.0031 |
| `target_pm10_72h`  | Ridge Regression  | 35.8607   | 28.7841  | −0.6672 |
| `target_o3_24h`    | Ridge Regression  | 14.2729   | 12.0910  | −2.6419 |
| `target_o3_48h`    | Ridge Regression  | 13.1736   | 10.5458  | −2.2204 |
| `target_o3_72h`    | Ridge Regression  | 19.0673   | 15.4177  | −5.7218 |
| `target_no2_24h`   | Gradient Boosting | 0.0125    | 0.0099   | 0.6838  |
| `target_no2_48h`   | Random Forest     | 0.0182    | 0.0147   | 0.3057  |
| `target_no2_72h`   | Random Forest     | 0.0175    | 0.0153   | 0.3596  |


*Ridge is stored as `LinearRegression` in metadata (`StandardScaler` + `Ridge` pipeline).*

Negative R² on some targets reflects short-term spikes over a 90-day window; models are still selected by RMSE/MAE. Final AQI depends primarily on dominant pollutants (PM2.5 / PM10).

---

## Model registry

```
train.py → save_model()
  ├── joblib → models/ (local cache)
  ├── upload → Hugging Face Hub (model.pkl + metadata.json)
  └── insert → MongoDB model_registry (active=True)
```

MongoDB stores `target`, `model_name`, `metrics`, `feature_columns`, `hf_repo_id`, `hf_model_path`, `trained_at`. Streamlit Cloud and GitHub Actions download from the same HF repository.

---

## Automation

GitHub Actions run all pipelines. **cron-job.org** triggers `workflow_dispatch` on schedule to avoid GitHub `schedule` delays on free-tier runners.


| Workflow              | Schedule                    | Purpose                               |
| --------------------- | --------------------------- | ------------------------------------- |
| **Feature pipeline**  | Hourly (`0 * * * `* UTC)    | Catch-up, fetch, incremental features |
| **Training pipeline** | Daily 02:00 UTC (07:00 PKT) | Retrain 12 targets, HF upload         |


**Feature pipeline:** validate secrets → optional catch-up (≤48 h) → fetch current OpenWeather → incremental feature upsert.

**Training pipeline:** validate secrets → `train.py` (48 fits → 12 winners) → HF upload + MongoDB registry + 30-day Actions artefact backup.

---

## Dashboard

**URL:** [aqipredictorbyrashidhussain.streamlit.app](https://aqipredictorbyrashidhussain.streamlit.app/)

The app reads the latest MongoDB feature row, runs `predict_all_horizons()`, and renders current + forecast AQI. It does **not** predict AQI directly - pollutant concentrations are forecast first, then converted via EPA breakpoints.

**Example output:**


| Output              | Value        |
| ------------------- | ------------ |
| Current AQI         | 91           |
| 24 h forecast       | AQI 92       |
| 48 h forecast       | AQI 81       |
| 72 h forecast       | AQI 83       |
| Dominant pollutants | PM2.5 / PM10 |


**Sections:** current AQI, +24/+48/+72 h cards, pollutant breakdown, forecast timeline, historical trends, alert banners, SHAP importances for +24 h PM2.5.

---

## Explainability (SHAP)

The dashboard computes SHAP values for the active `target_pm2_5_24h` model using recent feature rows. Tree models use `TreeExplainer`; Ridge uses `LinearExplainer`. Top drivers align with EDA: current PM2.5, lags, rolling means, humidity, wind, hour.

---

## Project structure

```
├── .github/workflows/       # CI, feature (hourly), training (daily)
├── app/streamlit_app.py     # Dashboard
├── notebooks/01_eda.ipynb   # Full EDA
├── docs/EXTERNAL_CRON_SETUP.md
├── scripts/
│   ├── verify_latest_feature.py
│   └── flush_and_reset.py
├── src/
│   ├── config.py            # Schema and env constants
│   ├── database.py          # MongoDB layer
│   ├── fetch_openweather.py
│   ├── feature_engineering.py
│   ├── backfill.py
│   ├── hourly_pipeline.py
│   ├── train.py
│   ├── predict.py
│   ├── model_registry.py
│   ├── hf_model_registry.py
│   ├── aqi_utils.py
│   └── utils.py
├── models/                  # Local cache (gitignored)
├── requirements.txt         # Full local stack
├── requirements-ci.txt      # GitHub Actions deps
└── final_report.pdf         # Full project report (PDF)
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # OPENWEATHER_API_KEY, MONGODB_URI
python src/backfill.py          # ~90 days history + batch features
python src/train.py             # train 12 models locally
streamlit run app/streamlit_app.py
```

`scripts/verify_latest_feature.py` compares MongoDB state against `predict.py` output.  
`backfill.py --rebuild-features` regenerates features after schema changes.

---

## Configuration


| Variable              | Purpose                                  | Required in CI     |
| --------------------- | ---------------------------------------- | ------------------ |
| `OPENWEATHER_API_KEY` | OpenWeather API                          | Feature + training |
| `MONGODB_URI`         | MongoDB Atlas                            | Feature + training |
| `DB_NAME`             | Database name (default: `aqi_predictor`) | Optional           |
| `HF_TOKEN`            | Hugging Face write token                 | Training           |
| `HF_REPO_ID`          | Hugging Face model repo                  | Training           |


Local: `.env` (see `.env.example`). GitHub: **Settings → Secrets and variables → Actions**. Streamlit Cloud: same keys in app secrets (TOML).

---

## Deployment

- **Streamlit Community Cloud**: main file `app/streamlit_app.py`, secrets as above
- **Container platforms**:  Python 3.11, `requirements.txt`, port 8501, `streamlit run app/streamlit_app.py --server.address=0.0.0.0`

---

## Challenges addressed


| Problem                                | Solution                                                         |
| -------------------------------------- | ---------------------------------------------------------------- |
| OpenWeather 1–5 category only          | Pollutant forecasting + EPA AQI calculation                      |
| Missed hourly runs                     | Catch-up pipeline (≤48 h)                                        |
| Incremental rows lack targets          | Batch backfill builds targets; hourly path omits unknown futures |
| Windows paths break on Streamlit Cloud | HF download + filename-based local cache                         |
| GitHub schedule delays                 | cron-job.org external scheduler                                  |

See [final_report.pdf](final_report.pdf) for the full challenges list and detailed write-up.

---

## Future improvements

Multi-city support · LSTM / transformers · ground-station validation · email/SMS alerts · FastAPI endpoint · drift detection · regulatory averaging windows

---

