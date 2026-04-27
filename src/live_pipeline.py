import os
import sys
import time
import json
import logging
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Add parent dir and current dir to path to import config and local files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import weather_ml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class LivePipeline:
    def __init__(self):
        self.live_data_file = config.LIVE_DATA_DIR / "live_weather_data.csv"
        self.drift_file = config.DRIFT_HISTORY_FILE
        self.features_file = config.PROCESSED_DIR / "weather_features.csv"
        self.raw_file = config.RAW_DIR / "weather_master_dataset.csv"
        
        # Ensure drift log exists
        if not self.drift_file.exists():
            df = pd.DataFrame(columns=['date', 'xgb_mae', 'nn_mae'])
            df.to_csv(self.drift_file, index=False)
            
    def get_last_retrained_date(self):
        """Finds the last date the model was retrained on."""
        if not self.drift_file.exists():
            return pd.to_datetime(config.END_DATE) # 2024-12-31
            
        df = pd.read_csv(self.drift_file)
        if df.empty:
            return pd.to_datetime(config.END_DATE)
        
        last_date_str = df['date'].max()
        return pd.to_datetime(last_date_str)

    def fetch_missing_historical_data(self, start_date, end_date):
        """Fetches missing raw data for the catch-up gap."""
        logging.info(f"Fetching catch-up data from {start_date.date()} to {end_date.date()}...")
        
        all_data = []
        for city, coords in config.REGIONS.items():
            logging.info(f"Fetching catch-up data for {city}...")
            # Use Open-Meteo Archive API
            url = f"https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,wind_direction_10m,relative_humidity_2m,dew_point_2m,pressure_msl,cloud_cover"
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                df_hr = pd.DataFrame(data['hourly'])
                df_hr.rename(columns={'time': 'datetime'}, inplace=True)
                df_hr['region'] = city
                all_data.append(df_hr)
            except Exception as e:
                logging.error(f"Error fetching {city}: {e}")
                
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            new_df['datetime'] = pd.to_datetime(new_df['datetime'])
            return new_df
        return pd.DataFrame()

    def update_master_dataset(self, new_raw_df):
        """Appends new raw data and rebuilds the features CSV."""
        logging.info("Updating master dataset with new raw data...")
        master_raw = pd.read_csv(self.raw_file)
        master_raw['datetime'] = pd.to_datetime(master_raw['datetime'], format='mixed')
        
        # Deduplicate and append
        combined = pd.concat([master_raw, new_raw_df])
        combined.drop_duplicates(subset=['datetime', 'region'], keep='last', inplace=True)
        combined.sort_values(by=['region', 'datetime'], inplace=True)
        combined.to_csv(self.raw_file, index=False)
        logging.info(f"Raw dataset updated to {len(combined)} rows.")
        
        # Re-run feature engineering
        logging.info("Rebuilding feature vectors...")
        weather_ml.step_feature_engineering()
        logging.info("Features rebuilt successfully.")
        
    def _train_and_evaluate_day(self, df, target_date):
        """Performs full standard retraining for a specific day and calculates MAE on that day."""
        logging.info(f"--- Retraining for {target_date.date()} ---")
        
        # Train on EVERYTHING strictly prior to target_date
        train_df = df[df['datetime'] < target_date].copy()
        
        # Test on the target_date specifically (the day we are evaluating)
        # We define a day as from 00:00 to 23:59
        next_day = target_date + timedelta(days=1)
        test_df = df[(df['datetime'] >= target_date) & (df['datetime'] < next_day)].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            logging.warning(f"Insufficient data for {target_date.date()}. Skipping.")
            return None, None
            
        targets = ['temperature']
        non_feature_cols = ['datetime', 'region', 'season_class', 'Temperature (°C)'] + targets
        features = [c for c in train_df.columns if c not in non_feature_cols]
        
        X_train = train_df[features]
        y_train = train_df['temperature']
        X_test = test_df[features]
        y_test = test_df['temperature']
        
        # Neural Network Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Neural Network Normal Retrain
        model_nn = MLPRegressor(
            hidden_layer_sizes=(150,), activation='relu', solver='adam',
            alpha=0.001, learning_rate_init=0.001, max_iter=500,
            early_stopping=True, validation_fraction=0.1, random_state=42,
            n_iter_no_change=20
        )
        model_nn.fit(X_train_scaled, y_train)
        preds_nn = model_nn.predict(X_test_scaled)
        mae_nn = mean_absolute_error(y_test, preds_nn)
        
        # 2. XGBoost Normal Retrain
        xgb_params = {}
        tuned_path = config.METRICS_DIR / "tuned_params.json"
        if tuned_path.exists():
            with open(tuned_path, "r") as f:
                xgb_params = json.load(f).get("XGBoost", {})
                
        model_xgb = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
        model_xgb.fit(X_train, y_train)
        preds_xgb = model_xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, preds_xgb)
        
        logging.info(f"   [Neural Network] MAE: {mae_nn:.4f} | [XGBoost] MAE: {mae_xgb:.4f}")
        
        # Save newest models (Hot-Swap)
        joblib.dump(model_xgb, config.MODEL_DIR / f"optimized_temperature_xgboost.joblib")
        joblib.dump(model_nn, config.MODEL_DIR / f"optimized_temperature_nn.joblib")
        joblib.dump(scaler, config.MODEL_DIR / f"scaler_temperature.joblib")
        
        return mae_xgb, mae_nn

    def run_catch_up(self):
        """The Catch-Up Engine."""
        logging.info("Starting Historical Catch-Up Engine...")
        last_date = self.get_last_retrained_date()
        target_max_date = pd.to_datetime(config.CATCH_UP_MAX_DATE)
        
        if last_date >= target_max_date:
            logging.info("System is fully caught up to CATCH_UP_MAX_DATE. No catch-up needed.")
            return

        start_fetch_date = last_date + timedelta(days=1)
        
        # 1. Fetch missing raw data chunk
        new_raw_df = self.fetch_missing_historical_data(start_fetch_date, target_max_date)
        if not new_raw_df.empty:
            self.update_master_dataset(new_raw_df)
        else:
            logging.warning("No new data fetched for the catch-up gap.")
            
        # 2. Load the newly rebuilt feature dataset
        df = pd.read_csv(self.features_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        
        # Memory Optimization: Compress 64-bit floats to 32-bit to halve the 2.1M row RAM footprint
        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # 3. Simulate Daily Retraining
        current_retrain_date = start_fetch_date
        drift_logs = []
        
        import gc
        while current_retrain_date <= target_max_date:
            xgb_mae, nn_mae = self._train_and_evaluate_day(df, current_retrain_date)
            
            if xgb_mae is not None and nn_mae is not None:
                drift_logs.append({
                    'date': current_retrain_date.strftime("%Y-%m-%d"),
                    'xgb_mae': xgb_mae,
                    'nn_mae': nn_mae
                })
                
                # Append to file immediately in case it crashes midway
                pd.DataFrame([drift_logs[-1]]).to_csv(self.drift_file, mode='a', header=False, index=False)
                
            current_retrain_date += timedelta(days=1)
            
            # Explicit Garbage Collection: Forcefully flush the 660MB Neural Network scaler arrays from RAM
            gc.collect()
            
        logging.info(f"Catch-up completed successfully. Completed {len(drift_logs)} daily retrain cycles.")

    def start_polling_daemon(self):
        """Polls Open-Meteo every 5 minutes (Live Ingestion Phase)"""
        logging.info("Live Polling Daemon activated. Listening every 5 minutes...")
        
        # Ensure the webdata directory exists for the dashboard
        webdata_dir = config.OUTPUTS_DIR / "webdata"
        webdata_dir.mkdir(parents=True, exist_ok=True)
        live_csv_path = webdata_dir / "live_current_temp.csv"

        while True:
            try:
                # 1. Fetch live 15-min resolution micro-data for all 10 regions
                live_data = []
                for city, coords in config.REGIONS.items():
                    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,pressure_msl&timezone=Europe/London"
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        if "current" in data:
                            current = data["current"]
                            live_data.append({
                                "region": city,
                                "timestamp": current.get("time"),
                                "temperature_2m": current.get("temperature_2m"),
                                "relative_humidity_2m": current.get("relative_humidity_2m"),
                                "wind_speed_10m": current.get("wind_speed_10m"),
                                "pressure_msl": current.get("pressure_msl")
                            })
                
                # 2. Save micro-data queue to be consumed by the dashboard
                if live_data:
                    df_live = pd.DataFrame(live_data)
                    df_live.to_csv(live_csv_path, index=False)
                    
            except Exception as e:
                logging.error(f"Error in 5-minute live polling daemon: {e}")
                
            time.sleep(config.LIVE_FETCH_INTERVAL_SECONDS)

if __name__ == "__main__":
    import importlib
    import threading
    from datetime import datetime, timedelta
    
    pipeline = LivePipeline()
    
    # 1. Spawn the 5-Minute Ingestion Daemon in a Background Thread
    poller_thread = threading.Thread(target=pipeline.start_polling_daemon, daemon=True)
    poller_thread.start()
    
    # 2. Daemon loop for continuous autonomous daily retraining (Main Thread)
    while True:
        try:
            # Reload config dynamically to fetch the newest live date
            importlib.reload(config)
            
            pipeline.run_catch_up()
            
            # After catch-up, check if today moved ahead — if so, loop again immediately
            importlib.reload(config)
            last = pipeline.get_last_retrained_date()
            today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            
            if last < today:
                logging.info("New date detected. Continuing catch-up immediately...")
                continue  # Skip sleep, loop again
            
            # Fully caught up — sleep until midnight
            now = datetime.now()
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_seconds = (midnight - now).total_seconds()
            logging.info(f"System fully caught up to {today.date()}. Sleeping until midnight ({sleep_seconds/3600:.1f} hours)...")
            time.sleep(sleep_seconds)
            
        except Exception as e:
            logging.error(f"Error during Catch-Up Cycle: {e}")
            logging.info("Will retry in 5 minutes...")
            time.sleep(300)  # Retry in 5 minutes on error

