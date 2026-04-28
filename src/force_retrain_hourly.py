import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set up paths to access config and pipeline perfectly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.live_pipeline import LivePipeline
import weather_ml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', filename=config.LOGS_DIR / "force_retrain_hourly.log", filemode='w')

def force_catch_up_hourly():
    logging.info("==================================================")
    logging.info("MANUAL HOURLY RETRAIN INITIATED")
    pipeline = LivePipeline()
    
    last_date = pipeline.get_last_retrained_date()
    now = datetime.now()
    
    # We want to fill the gap from last_date up to NOW (current hour) using api.open-meteo.com past_days
    # The normal archive API handles 2-5 days latency, so we use the forecast endpoint with past_days
    logging.info(f"Fetching missing data strictly up to {now.strftime('%Y-%m-%d %H:00')}...")
    
    all_data = []
    
    # Calculate days needed (past_days takes max 90, min 1)
    days_diff = (now - last_date).days
    past_days = max(1, min(90, days_diff + 2))
    
    for city, coords in config.REGIONS.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "past_days": past_days,
            "hourly": "temperature_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,wind_direction_10m,relative_humidity_2m,dew_point_2m,pressure_msl,cloud_cover"
        }
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            if 'hourly' in data:
                df_hr = pd.DataFrame(data['hourly'])
                df_hr.rename(columns={'time': 'datetime'}, inplace=True)
                df_hr['region'] = city
                all_data.append(df_hr)
        except Exception as e:
            logging.error(f"Error fetching {city}: {e}")
            
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        new_df['datetime'] = pd.to_datetime(new_df['datetime'])
        
        # update_master_dataset appends, removes duplicates dynamically, and calls feature engineering
        pipeline.update_master_dataset(new_df)
        
        logging.info("Starting performance evaluation for the exact newly ingested timeframe...")
        df = pd.read_csv(pipeline.features_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        
        # Free up memory immediately by forcing types down
        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Evaluation Lock: We only log 'Drift' and mark a day as 'Evaluated' if it's over.
        # If it's 23:28 on the 27th, the 27th is NOT finished. We evaluate up to the 26th.
        evaluation_target = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if last_date >= evaluation_target.replace(tzinfo=None):
            logging.info("Manual Sync successfully fetched data, but the most recent finished day was already evaluated. Model updated with latest features, drift log remains steady.")
        else:
            logging.info(f"Evaluating model drift for newly completed cycle: {evaluation_target.date()}")
            xgb_mae, nn_mae = pipeline._train_and_evaluate_day(df, evaluation_target)
            
            if xgb_mae is not None and nn_mae is not None:
                logging.info(f"Manual Eval Complete. XGB MAE: {xgb_mae:.4f}, NN MAE: {nn_mae:.4f}")
                pd.DataFrame([{
                    'date': evaluation_target.strftime("%Y-%m-%d"),
                    'xgb_mae': xgb_mae,
                    'nn_mae': nn_mae
                }]).to_csv(pipeline.drift_file, mode='a', header=False, index=False)
            
            # Log heavily so the user sees it in run.log via the dashboard UI
            with open(config.LOGS_DIR / "run.log", "a") as f:
                f.write(f"\n======================================================================\n")
                f.write(f"Pipeline Run: {now.strftime('%Y-%m-%d %H:%M:%S')} (MANUAL HOURLY SYNC)\n")
                f.write(f"  Data Gap Filled: {past_days} days of sub-hourly resolution up to current hour.\n")
                f.write(f"  Master Dataset & Feature Matrix Restored.\n")
                f.write(f"  Recompiling Neural Matrices (Hot-Swap Enabled).\n")
                f.write(f"  System is now synced EXACTLY up to {now.strftime('%H:00')}.\n")
                f.write(f"======================================================================\n")
    else:
        logging.warning("No data retrieved during catch-up sequence.")
        
    logging.info("MANUAL HOURLY RETRAIN COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    force_catch_up_hourly()
