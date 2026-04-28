import os
import zipfile
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
WEB_DATA_DIR = BASE_DIR / "dashboard" / "data"

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, METRICS_DIR, LOGS_DIR, WEB_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def ensure_data_unzipped(csv_path):
    """Checks if a CSV exists. If not, looks for its .zip or multi-part .zips and extracts them."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return True
    
    # Check for single zip
    zip_path = csv_path.with_suffix('.zip')
    if zip_path.exists():
        logging.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(csv_path.parent)
        return True
    
    # Check for multi-part zips (e.g., pt1, pt2)
    base_name = csv_path.stem # e.g. weather_features
    parts = sorted(csv_path.parent.glob(f"{base_name}_pt*.zip"))
    if parts:
        logging.info(f"Detected {len(parts)} data parts. Extracting and merging...")
        all_dfs = []
        import pandas as pd
        for p in parts:
            with zipfile.ZipFile(p, 'r') as zf:
                # Assume one csv inside
                internal_csv = zf.namelist()[0]
                with zf.open(internal_csv) as f:
                    all_dfs.append(pd.read_csv(f))
        
        # Merge and save as the master CSV
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(csv_path, index=False)
        logging.info(f"Successfully reconstructed {csv_path.name} from parts.")
        return True
    
    return False

# Live Pipeline & Retraining Configuration
LIVE_DATA_DIR = DATA_DIR / "live"
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
RETRAIN_INTERVAL_HOURS = 24
LIVE_FETCH_INTERVAL_SECONDS = 300  # 5 minutes (polls 15-minute resolution data)

# CRITICAL LOGIC FIX: 'Just finished' day means Yesterday. 
# We target the last completed 24-hour cycle to prevent 400 errors and redundant processing.
CATCH_UP_MAX_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")   
DRIFT_HISTORY_FILE = METRICS_DIR / "drift_history.csv"

# 10 Regions for the Master Project
REGIONS = {
    "London": {"lat": 51.5085, "lon": -0.1257},
    "Manchester": {"lat": 53.4809, "lon": -2.2374},
    "Birmingham": {"lat": 52.4814, "lon": -1.8998},
    "Leeds": {"lat": 53.7964, "lon": -1.5478},
    "Glasgow": {"lat": 55.8651, "lon": -4.2576},
    "Southampton": {"lat": 50.9039, "lon": -1.4042},
    "Liverpool": {"lat": 53.4105, "lon": -2.9779},
    "Newcastle": {"lat": 54.9732, "lon": -1.6139},
    "Sheffield": {"lat": 53.3829, "lon": -1.4659},
    "Middlesbrough": {"lat": 54.5762, "lon": -1.2348}
}

# Data Range: 24 years (2001-2024) for ~2M+ hourly records across 10 regions
START_DATE = "2001-01-01"
END_DATE = "2024-12-31"

# Train/Test Split: 80/20
TRAIN_END_YEAR = 2019   # Train: 2001-2019 (19 years, ~80%)
TEST_START_YEAR = 2020  # Test:  2020-2024 (5 years, ~20%)
