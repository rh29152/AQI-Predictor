# Karachi AQI Predictor

Automated **pollutant forecasting + AQI calculation** for Karachi, Pakistan. Hourly data flows through MongoDB Atlas; twelve ML models predict PM2.5, PM10, O₃, and NO₂ at 24 h / 48 h / 72 h; EPA breakpoints convert concentrations to AQI (dominant pollutant = max sub-index).

**Live dashboard:** [aqipredictorbyrashidhussain.streamlit.app](https://aqipredictorbyrashidhussain.streamlit.app/)  
 · **EDA:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

---

## Architecture

```
OpenWeather (+ Open-Meteo backfill)
        │
        ▼  hourly - GitHub Actions (triggered by cron-job.org)
  hourly_pipeline.py → MongoDB (raw_data, features)
        │
        ▼  daily - GitHub Actions
  train.py → 12 targets × 4 algorithms → Hugging Face + model_registry
        │
        ▼
  predict.py → EPA AQI  →  Streamlit dashboard
```

| Layer | Stack |
|---|---|
| Data | OpenWeather, Open-Meteo, MongoDB Atlas |
| ML | Ridge, Random Forest, Gradient Boosting, XGBoost |
| Registry | Hugging Face Hub + MongoDB metadata |
| UI | Streamlit, Plotly, SHAP |
| CI/CD | GitHub Actions + [cron-job.org setup](docs/EXTERNAL_CRON_SETUP.md) |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # add OPENWEATHER_API_KEY, MONGODB_URI
python src/backfill.py          # ~90 days history + batch features
python src/train.py             # train 12 models locally
streamlit run app/streamlit_app.py
```

---

## Automation

| Workflow | Schedule | Role |
|---|---|---|
| **Feature pipeline** | Hourly (UTC) | Fetch, catch-up (≤48 h), incremental features |
| **Training pipeline** | Daily 02:00 UTC | Retrain 12 models, upload to HF |
| **CI** | Push / PR | Import smoke test |

External scheduler [**cron-job.org**](docs/EXTERNAL_CRON_SETUP.md) triggers `workflow_dispatch` so runs stay on time (GitHub `schedule` can delay).

---

## Secrets

| Variable | Used by |
|---|---|
| `OPENWEATHER_API_KEY` | Feature pipeline, training, Streamlit |
| `MONGODB_URI` | All components |
| `DB_NAME` | Optional (default: `aqi_predictor`) |
| `HF_TOKEN`, `HF_REPO_ID` | Training upload + Streamlit model download |

Set locally in `.env`; in GitHub under **Settings → Secrets and variables → Actions**. Streamlit Cloud uses the same keys in app secrets.

---

## Project layout

```
src/           pipelines (fetch, features, train, predict, registry)
app/           Streamlit dashboard
notebooks/     EDA
.github/       CI + feature + training workflows
```

---

## License

MIT
