import sys
import json
import os
import joblib
import pandas as pd
import warnings

# CRITICAL FIX: Redirect all stdout printing (from warnings, xgboost, etc) to stderr 
# to protect the pure JSON response payload back to the Node/JS dashboard.
_original_stdout = sys.stdout
sys.stdout = sys.stderr

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore")

# Adjust path and import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.live_pipeline import LivePipeline

def predict_batch(input_data):
    nn_path = config.MODEL_DIR / "optimized_temperature_nn.joblib"
    xgb_path = config.MODEL_DIR / "optimized_temperature_xgboost.joblib"
    scaler_path = config.MODEL_DIR / "scaler_temperature.joblib"
    
    if not (nn_path.exists() and scaler_path.exists()):
        return {"error": "Models not found on disk. Please wait for pipeline to finish."}
        
    model_nn = joblib.load(nn_path)
    scaler = joblib.load(scaler_path)
    
    # Gracefully handle XGBoost unpickling version crashes
    try:
        model_xgb = joblib.load(xgb_path)
    except Exception:
        model_xgb = None
    
    # The input_data is a dictionary containing a list of rows from the Javascript Open-Meteo pull
    # or just a requested basic date from Tab 2.
    
    pipeline = LivePipeline()
    
    # Check if we are doing a 'Live' request (last 30 days) or a deep historical audit
    is_historical = False
    try:
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        for req in input_data.get("requests", []):
            if pd.to_datetime(req.get("date")) < thirty_days_ago:
                is_historical = True
                break
    except Exception:
        is_historical = True

    if is_historical:
        # Deep Audit: Full read required (Slower)
        df = pd.read_csv(pipeline.features_file)
    else:
        # Live Interface: Optimized Tail-Read (Instant)
        try:
            f_path = pipeline.features_file
            size = os.path.getsize(f_path)
            with open(f_path, 'rb') as f:
                f.seek(max(0, size - 1500000)) # Last 1.5MB (covers ~150 regions*days)
                tail_data = f.readlines()[1:] 
            from io import BytesIO
            header = pd.read_csv(f_path, nrows=0).columns.tolist()
            df = pd.read_csv(BytesIO(b"".join(tail_data)), names=header)
        except Exception:
            df = pd.read_csv(pipeline.features_file)
    
    results = []
    
    # We will grab the absolute latest row of data from weather_features.csv for that region 
    # and simply inject the required DateTime to form an artificial "baseline prediction vector".
    # This guarantees the ML model operates cleanly using recent feature lag distributions.
    
    targets = ['temperature']
    non_feature_cols = ['datetime', 'region', 'season_class', 'Temperature (°C)'] + targets
    feature_names = [c for c in df.columns if c not in non_feature_cols]
    
    for req in input_data.get("requests", []):
        region = req.get("region", "London")
        date_str = req.get("date") # e.g. "2026-05-15" or "2026-05-15 14:00"
        
        df_region = df[df['region'] == region].copy()
        if df_region.empty:
            continue
            
        last_row = df_region.iloc[-1].copy()
        target_dt = pd.to_datetime(date_str)
        
        last_row['datetime'] = target_dt
        last_row['day_of_year'] = target_dt.dayofyear
        last_row['month'] = target_dt.month
        last_row['hour'] = target_dt.hour
        
        X = last_row[feature_names].to_frame().T
        X_scaled = scaler.transform(X)
        
        pred_nn = model_nn.predict(X_scaled)[0]
        
        if model_xgb:
            try:
                pred_xgb = model_xgb.predict(X)[0]
            except Exception:
                pred_xgb = pred_nn
        else:
            pred_xgb = pred_nn
            
        results.append({
            "region": region,
            "date": date_str,
            "nn_pred": round(float(pred_nn), 2),
            "xgb_pred": round(float(pred_xgb), 2)
        })
        
    return {"status": "success", "predictions": results}

def run_safe_inference():
    try:
        input_data = json.loads(sys.stdin.read())
        res = predict_batch(input_data)
        sys.stdout = _original_stdout
        print(json.dumps(res))
    except Exception as e:
        sys.stdout = _original_stdout
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    run_safe_inference()
