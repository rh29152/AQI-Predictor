"""Quick check: latest MongoDB raw/features vs dashboard predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_collection, ensure_indexes
from src.config import FEATURES_COLLECTION, RAW_COLLECTION, POLLUTANTS_TO_FORECAST
from src.aqi_utils import calculate_final_aqi
from src.predict import predict_all_horizons

ensure_indexes()

raw_col = get_collection(RAW_COLLECTION)
feat_col = get_collection(FEATURES_COLLECTION)

print(f"Collection counts: raw_data={raw_col.count_documents({})}, features={feat_col.count_documents({})}\n")

print("Latest 3 RAW rows:")
for r in raw_col.find({}, {"_id": 0}).sort("datetime", -1).limit(3):
    print(f"  {r['datetime']}  aqi={r.get('aqi')}  pm2_5={r.get('pm2_5')}  pm10={r.get('pm10')}")

print("\nLatest 3 FEATURE rows:")
for r in feat_col.find({}, {"_id": 0}).sort("datetime", -1).limit(3):
    has_lags = "pm2_5_lag_1" in r
    has_targets = any(k.startswith("target_") for k in r)
    print(
        f"  {r['datetime']}  pm2_5={r.get('pm2_5')}  pm10={r.get('pm10')}  "
        f"lags={has_lags}  targets={has_targets}"
    )

row = feat_col.find_one({}, {"_id": 0}, sort=[("datetime", -1)])
print("\n=== LATEST FEATURE (detail) ===")
print(f"datetime:      {row.get('datetime')}")
print(f"aqi_category:  {row.get('aqi_category')}")
for p in POLLUTANTS_TO_FORECAST:
    print(f"  {p}: {row.get(p)}")
print(f"  pm2_5_lag_1:           {row.get('pm2_5_lag_1')}")
print(f"  pm2_5_lag_24:          {row.get('pm2_5_lag_24')}")
print(f"  pm2_5_rolling_24_mean:  {row.get('pm2_5_rolling_24_mean')}")
print(f"  hour: {row.get('hour')}  day_of_week: {row.get('day_of_week')}")
print(f"  has target columns: {any(k.startswith('target_') for k in row)}")

concs = {p: float(row.get(p, 0) or 0) for p in POLLUTANTS_TO_FORECAST}
aqi_res = calculate_final_aqi(concs)
print("\n=== COMPUTED CURRENT AQI ===")
print(f"  AQI={aqi_res['aqi']}  category={aqi_res['category']}  dominant={aqi_res['dominant_pollutant']}")
print(f"  sub_indices: {aqi_res['sub_indices']}")

print("\n=== LIVE PREDICTIONS ===")
preds = predict_all_horizons()
for hz in ["24h", "48h", "72h"]:
    info = preds[hz]
    print(f"  +{hz}: AQI={info['predicted_aqi']}  dominant={info['dominant_pollutant']}  target={info['target_time']}")
    for p, v in info["predicted_pollutants"].items():
        print(f"         {p}={v}  sub={info['sub_indices'].get(p)}")
