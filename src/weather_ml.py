# Step 1: Import & Setup
import logging
import time
import pandas as pd
import numpy as np
import json
import joblib
import requests
from datetime import datetime, timedelta
import sys
import os
import glob

# ML Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

class StepTracker:
    """Tracks execution time and status of each Academic Pipeline step."""
    def __init__(self):
        self.steps = []
        
    def start(self, name):
        self.current_t = time.time()
        logging.info(f"{name}")
        
    def format_duration(self, dur):
        if dur < 60:
            return f"{dur:.2f}s"
        mins = int(dur // 60)
        secs = int(dur % 60)
        return f"{mins}m {secs}s"

    def end(self, name, status="OK"):
        dur = time.time() - self.current_t
        fmt_dur = self.format_duration(dur)
        logging.info(f"[{status}] {name} completed in {fmt_dur}\n")
        self.steps.append({"name": name, "duration": dur, "fmt_dur": fmt_dur, "status": status})
        
    def print_summary(self):
        print("\nFINAL PIPELINE EXECUTION SUMMARY")
        for i, step in enumerate(self.steps, 1):
            print(f"Step {i:02d} | {step['status']:<4} | {step['fmt_dur']:>8} | {step['name']}")
        print("\n")

    def get_text_summary(self):
        lines = ["FINAL PIPELINE EXECUTION SUMMARY"]
        lines.append("-" * 60)
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i:02d} | {step['status']:<4} | {step['fmt_dur']:>8} | {step['name']}")
        return "\n".join(lines)

def format_research_table(df, title="Metrics Overview"):
    """Formats a pandas DataFrame into the user's requested dissertation-style ASCII table."""
    width = 95
    double_sep = "=" * width
    single_sep = "-" * width
    
    lines = []
    lines.append(double_sep)
    lines.append(title.center(width))
    lines.append(double_sep)
    
    # Header logic
    cols = list(df.columns)
    index_name = df.index.name if df.index.name else "Model"
    header_line = f"{index_name:<30}"
    for col in cols:
        header_line += f" | {col:<15}"
    lines.append(header_line)
    lines.append(single_sep)
    
    # Rows logic
    for idx, row in df.iterrows():
        row_line = f"{str(idx):<30}"
        for col in cols:
            val = row[col]
            val_str = f"{val:.4f}" if isinstance(val, (float, np.float64, np.float32)) else str(val)
            row_line += f" | {val_str:<15}"
        lines.append(row_line)
    
    lines.append(double_sep)
    return "\n".join(lines)

tracker = StepTracker()

# Step 2: Data Fetching & Integration (Open-Meteo — Free, No API Key, No Limits)
def step_fetch_data():
    """Fetches hourly historical weather data from Open-Meteo API (ERA5 reanalysis).
    Downloads 24 years (2001-2024) of hourly data for 10 UK regions.
    Collects 10 weather variables for multi-output prediction.
    No API key required. No rate limits for reasonable use.
    """
    tracker.start("Step 2: Data Fetching & Integration")
    
    master_df_path = config.RAW_DIR / "weather_master_dataset.csv"
    
    start_year = int(config.START_DATE[:4])
    end_year = int(config.END_DATE[:4])
    expected_regions = list(config.REGIONS.keys())
    expected_years = list(range(start_year, end_year + 1))
    
    # Check if data already exists AND is complete
    existing_df = None
    missing_combos = []
    
    if master_df_path.exists():
        existing_df = pd.read_csv(master_df_path)
        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'], format='mixed')
        
        # Check which region+year combos are present
        existing_df['_year'] = existing_df['datetime'].dt.year
        for region in expected_regions:
            for year in expected_years:
                region_year = existing_df[(existing_df['region'] == region) & (existing_df['_year'] == year)]
                if len(region_year) < 8000:  # A full year has ~8760 hours
                    missing_combos.append((region, year))
        existing_df.drop(columns=['_year'], inplace=True)
        
        if not missing_combos:
            logging.info(f"   Dataset complete with {len(existing_df)} rows ({len(expected_regions)} regions × {len(expected_years)} years). Skipping download.")
            tracker.end("Step 2: Data Fetching & Integration")
            return
        else:
            logging.info(f"   Dataset exists but incomplete. Missing {len(missing_combos)} region-year combos. Fetching missing data...")
    else:
        # No file at all — fetch everything
        for region in expected_regions:
            for year in expected_years:
                missing_combos.append((region, year))
        logging.info(f"   No dataset found. Fetching all {len(missing_combos)} region-year combos...")
    
    # Weather variables to fetch from Open-Meteo
    hourly_vars = [
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_direction_10m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "cloud_cover"
    ]
    hourly_param = ",".join(hourly_vars)
    
    all_data = []
    
    for region, year in missing_combos:
        coords = config.REGIONS[region]
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={year_start}&end_date={year_end}"
            f"&hourly={hourly_param}"
            f"&timezone=Europe/London"
        )
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                logging.error(f"   X API Error {response.status_code} for {region} ({year}): {response.text}")
                continue
            
            data = response.json()
            hourly = data.get('hourly', {})
            
            times = hourly.get('time', [])
            
            for i in range(len(times)):
                all_data.append({
                    "datetime": times[i],
                    "region": region,
                    "temperature": hourly.get('temperature_2m', [0])[i] or 0,
                    "feels_like": hourly.get('apparent_temperature', [0])[i] or 0,
                    "precipitation": hourly.get('precipitation', [0])[i] or 0,
                    "rain": hourly.get('rain', [0])[i] or 0,
                    "snowfall": hourly.get('snowfall', [0])[i] or 0,
                    "wind_speed": hourly.get('wind_speed_10m', [0])[i] or 0,
                    "wind_direction": hourly.get('wind_direction_10m', [0])[i] or 0,
                    "humidity": hourly.get('relative_humidity_2m', [0])[i] or 0,
                    "dew_point": hourly.get('dew_point_2m', [0])[i] or 0,
                    "pressure": hourly.get('pressure_msl', [0])[i] or 0,
                    "cloud_cover": hourly.get('cloud_cover', [0])[i] or 0,
                })
            
            logging.info(f"      {region} {year}: {len(times)} hourly records downloaded")
            
        except Exception as e:
            logging.error(f"   X Failed to fetch {region} ({year}): {e}")
            continue
    
    if not all_data and existing_df is None:
        logging.error("   X No data collected!")
        tracker.end("Step 2: Data Fetching & Integration", status="FAIL")
        return
    
    # Combine existing data with newly fetched data
    new_df = pd.DataFrame(all_data) if all_data else pd.DataFrame()
    if existing_df is not None and not new_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif existing_df is not None:
        final_df = existing_df
    else:
        final_df = new_df
    
    final_df.to_csv(master_df_path, index=False)
    logging.info(f"   Saved {len(final_df)} hourly records ({len(final_df.columns)} variables) to {master_df_path.name}")
    tracker.end("Step 2: Data Fetching & Integration")




# Step 3: Validation & EDA
def step_validate_eda():
    tracker.start("Step 3: Validation & EDA")
    master_df_path = config.RAW_DIR / "weather_master_dataset.csv"
    if not master_df_path.exists():
        logging.error("   X Master dataset not found!")
        tracker.end("Step 3: Validation & EDA", status="FAIL")
        return
        
    df = pd.read_csv(master_df_path)
    logging.info(f"   Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    logging.info(f"   Columns: {list(df.columns)}")
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logging.info(f"   Handling {missing} missing values (forward fill).")
        df.ffill(inplace=True)
    else:
        logging.info("   No missing values found.")
        
    plt.style.use('ggplot')
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=['float64', 'number']).columns
    if not numeric_cols.empty:
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Weather Features Correlation Heatmap (Hourly, 11 Variables)")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "correlation_heatmap.png")
        logging.info("   Saved correlation_heatmap.png")
    plt.close()
    
    # Missing Values Bar Chart
    plt.figure(figsize=(10, 6))
    missing_counts = df.isnull().sum()
    ax = missing_counts.plot(kind='bar', color='coral')
    plt.title("Missing Values per Feature (Before Imputation)")
    max_missing = missing_counts.max()
    plt.ylim(0, max(5, max_missing + max_missing * 0.1))
    if max_missing == 0:
        plt.text(0.5, 0.5, '0 Missing Values Detected\n(Raw Dataset is Clean)', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax.transAxes, fontsize=14, color='green', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "missing_values_bar.png")
    plt.close()
    
    # Regional Temperature Trends
    try:
        temp_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(temp_df['datetime']):
            temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], format='mixed')
        plt.figure(figsize=(12, 6))
        for region in temp_df['region'].unique()[:3]:
            region_df = temp_df[temp_df['region'] == region].head(2000)
            plt.plot(region_df['datetime'], region_df['temperature'], label=region, alpha=0.6)
        plt.title("Hourly Temperature Trends (First 2000 records)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "temperature_trends_by_region.png")
        plt.close()
    except Exception as e:
        logging.error(f"   X Failed to plot trends: {e}")
    
    tracker.end("Step 3: Validation & EDA")

# Step 4: Feature Engineering
def step_feature_engineering():
    tracker.start("Step 4: Feature Engineering")
    master_df_path = config.RAW_DIR / "weather_master_dataset.csv"
    if not master_df_path.exists():
        tracker.end("Step 4: Feature Engineering", status="FAIL")
        return
        
    df = pd.read_csv(master_df_path)
    
    # Critically forward-fill and back-fill missing API columns (like 'feels_like')
    # This prevents the subsequent `dropna()` from destroying the entire 2.1 million row dataset!
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df.sort_values(by=['region', 'datetime'], inplace=True)
    
    # 1. Temporal Lags for hourly data
    # Lag 1 = 1 hour, Lag 6 = 6 hours, Lag 24 = 1 day
    lag_cols = ['temperature', 'precipitation', 'wind_speed', 'humidity', 'pressure']
    for col in lag_cols:
        df[f'{col}_lag1'] = df.groupby('region')[col].shift(1)
        df[f'{col}_lag6'] = df.groupby('region')[col].shift(6)
        df[f'{col}_lag24'] = df.groupby('region')[col].shift(24)
        
    # 2. Rolling averages (6 hour and 24 hour)
    df['temp_roll6'] = df.groupby('region')['temperature'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['temp_roll24'] = df.groupby('region')['temperature'].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df['wind_roll6'] = df.groupby('region')['wind_speed'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['precip_roll24'] = df.groupby('region')['precipitation'].transform(lambda x: x.rolling(24, min_periods=1).sum())
    
    # 3. Hour of Day & Day of Year
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    
    # 4. Season Classification
    df['season_class'] = (df['month'] % 12 // 3)
    
    df.dropna(inplace=True)
    
    # 5. Type Casting: Force all features to numeric to prevent XGBoost Categorical ValueErrors
    for col in df.columns:
        if col not in ['datetime', 'region']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df.dropna(inplace=True)
    
    processed_df_path = config.PROCESSED_DIR / "weather_features.csv"
    df.to_csv(processed_df_path, index=False)
    
    # Identify newly created features for reporting
    engineered_cols = [c for c in df.columns if any(x in c for x in ['_lag', 'roll', 'season_class', 'hour', 'day_of_year', 'month'])]
    
    logging.info(f"   Engineered {len(engineered_cols)} scientific features: {', '.join(engineered_cols)}")
    logging.info(f"   Saved {len(df)} ML-ready rows ({len(df.columns)} features) to weather_features.csv")
    tracker.end("Step 4: Feature Engineering")

# Step 5: Multi-Output Model Training
def step_model_training():
    """Trains baseline models for Temperature prediction (single-target)."""
    tracker.start("Step 5: Model Training & Split")
    
    base_metrics_path = config.METRICS_DIR / "base_regression_metrics.json"
    if base_metrics_path.exists():
        if os.environ.get('RETRAIN_BASE_MODELS', 'n').lower() != 'y':
            logging.info("   Skipping base model training (cached results found).")
            tracker.end("Step 5: Model Training & Split", status="SKIP")
            return
        logging.info("   User chose to retrain base models.")
        
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists():
        tracker.end("Step 5: Model Training & Split", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # 80/20 Split using config
    train_df = df[df['datetime'].dt.year <= config.TRAIN_END_YEAR].copy()
    test_df = df[df['datetime'].dt.year >= config.TEST_START_YEAR].copy()
    
    logging.info(f"   Train Split (2001-{config.TRAIN_END_YEAR}): {len(train_df)} rows")
    logging.info(f"   Test Split ({config.TEST_START_YEAR}-2024): {len(test_df)} rows")
    
    if len(train_df) == 0 or len(test_df) == 0:
        logging.warning("   ! Insufficient data for train/test split. Skipping model training.")
        tracker.end("Step 5: Model Training & Split", status="SKIP")
        return
    
    # Prediction targets
    targets = {
        'temperature': 'Temperature (°C)'
    }
    
    # Features = everything except identifiers, targets, and derived targets
    non_feature_cols = ['datetime', 'region', 'season_class'] + list(targets.keys())
    features = [c for c in df.columns if c not in non_feature_cols]
    
    all_metrics = {}
    for target_col, target_label in targets.items():
        logging.info(f"   Training models for {target_label}...")
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # Scale features for Neural Network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "RandomForest": (RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1), X_train, X_test),
            "NeuralNetwork": (MLPRegressor(hidden_layer_sizes=(150,), activation='relu', solver='adam', alpha=0.001, max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1), X_train_scaled, X_test_scaled),
            "XGBoost": (XGBRegressor(n_estimators=100, max_depth=12, learning_rate=0.1, random_state=42, n_jobs=-1), X_train, X_test),
        }
        
        target_metrics = {}
        for model_name, (model, X_tr, X_te) in models.items():
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            target_metrics[model_name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}
            logging.info(f"      [{model_name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
        
        all_metrics[target_col] = {"label": target_label, "models": target_metrics}
    
    with open(config.METRICS_DIR / "base_regression_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    tracker.end("Step 5: Model Training & Split")

# Step 6: Season Classification
def step_season_classification():
    tracker.start("Step 6: Season Classification")
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists():
        tracker.end("Step 6: Season Classification", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    train_df = df[df['datetime'].dt.year <= config.TRAIN_END_YEAR].copy()
    test_df = df[df['datetime'].dt.year >= config.TEST_START_YEAR].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        logging.warning("   ! Insufficient data for season classification. Skipping.")
        tracker.end("Step 6: Season Classification", status="SKIP")
        return
    
    # Downsample for speed
    train_df = train_df.sample(frac=0.3, random_state=42)
    test_df = test_df.sample(frac=0.3, random_state=42)
    
    drop_cols = ['datetime', 'region', 'season_class']
    features = [c for c in df.columns if c not in drop_cols]
    
    X_train = train_df[features]
    y_train = train_df['season_class']
    X_test = test_df[features]
    y_test = test_df['season_class']
    
    logging.info("   Training Season Classifier (Hourly features)...")
    clf = XGBClassifier(n_estimators=30, random_state=42, eval_metric='mlogloss', n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    logging.info(f"   Season Classification Accuracy: {acc*100:.2f}%")
    tracker.end("Step 6: Season Classification")

# Step 7: Hyperparameter Tuning
def step_hyperparameter_tuning():
    tracker.start("Step 7: Hyperparameter Tuning")
    
    tuned_path = config.METRICS_DIR / "tuned_params.json"
    if tuned_path.exists():
        if os.environ.get('RETUNE_MODELS', 'n').lower() != 'y':
            logging.info("   Skipping hyperparameter tuning (cached params found).")
            tracker.end("Step 7: Hyperparameter Tuning", status="SKIP")
            return
        logging.info("   User chose to re-tune hyperparameters.")
        
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists():
        tracker.end("Step 7: Hyperparameter Tuning", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    train_df = df[df['datetime'].dt.year <= config.TRAIN_END_YEAR].copy()
    
    if len(train_df) == 0:
        logging.warning("   ! Insufficient training data. Skipping tuning.")
        tracker.end("Step 7: Hyperparameter Tuning", status="SKIP")
        return
    
    # Downsample to 20% for speed during tuning (balances accuracy vs runtime)
    train_df = train_df.sample(frac=0.2, random_state=42)
    
    targets = ['temperature', 'rain', 'snowfall', 'wind_speed']
    non_feature_cols = ['datetime', 'region', 'temperature', 'season_class'] + targets
    drop_cols = list(set(non_feature_cols))
    features = [c for c in df.columns if c not in drop_cols]
    X_train = train_df[features]
    y_train = train_df['temperature']
    
    logging.info("   Tuning XGBoost hyperparameters using RandomizedSearchCV (3-fold CV)...")
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.05, 0.1]
    }
    search_xgb = RandomizedSearchCV(xgb, param_distributions=param_grid_xgb, n_iter=5, cv=3, random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')
    search_xgb.fit(X_train, y_train)
    
    best_params = {
        "XGBoost": search_xgb.best_params_
    }
    logging.info(f"   Best XGBoost params: {best_params}")
    
    with open(config.METRICS_DIR / "tuned_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    tracker.end("Step 7: Hyperparameter Tuning")

# Step 8: Optimized Multi-Output Model Training (XGBoost + Neural Network)
def step_optimized_model():
    """Trains Optimized XGBoost and Optimized Neural Network for single-target temperature prediction."""
    tracker.start("Step 8: Optimized Model Training")
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists():
        tracker.end("Step 8: Optimized Model Training", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    train_df = df[df['datetime'].dt.year <= config.TRAIN_END_YEAR].copy()
    test_df = df[df['datetime'].dt.year >= config.TEST_START_YEAR].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        logging.warning("   ! Insufficient data for optimized training. Skipping.")
        tracker.end("Step 8: Optimized Model Training", status="SKIP")
        return
    
    targets = {
        'temperature': 'Temperature (°C)'
    }
    
    non_feature_cols = ['datetime', 'region', 'season_class'] + list(targets.keys())
    features = [c for c in df.columns if c not in non_feature_cols]
    
    # Load tuned params from Step 7 (if available)
    xgb_params = {}
    tuned_path = config.METRICS_DIR / "tuned_params.json"
    if tuned_path.exists():
        with open(tuned_path, "r") as f:
            xgb_params = json.load(f).get("XGBoost", {})
        logging.info(f"   Using tuned XGBoost params: {xgb_params}")
    else:
        xgb_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        logging.info("   No tuned params found, using defaults.")
    
    all_preds = test_df[['datetime', 'region']].copy()
    optimized_metrics = {}
    
    # Store temperature predictions for plots
    temp_preds_xgb = None
    temp_preds_nn = None
    temp_y_test = None
    temp_model_xgb = None
    temp_features = features
    
    for target_col, target_label in targets.items():
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # Scale for Neural Network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- Optimized Neural Network ---
        model_nn_path = config.MODEL_DIR / f"optimized_{target_col}_nn.joblib"
        scaler_path = config.MODEL_DIR / f"scaler_{target_col}.joblib"
        target_architecture = (150,)
        cache_valid_nn = False
        
        if model_nn_path.exists():
            try:
                cached_nn = joblib.load(model_nn_path)
                # Verify strictly that the cached model has the exact structure we are currently requesting
                if hasattr(cached_nn, 'n_features_in_') and cached_nn.n_features_in_ == len(features):
                    if hasattr(cached_nn, 'hidden_layer_sizes') and cached_nn.hidden_layer_sizes == target_architecture:
                        cache_valid_nn = True
                        model_nn = cached_nn
                        logging.info(f"   [Cache Hit] Pre-trained NN strictly matches target architecture {target_architecture}. Instantly loading...")
                        if not scaler_path.exists():
                            joblib.dump(scaler, scaler_path)
                    else:
                        logging.info(f"   [Cache Miss] NN architecture mis-match. Found {cached_nn.hidden_layer_sizes}. Retraining...")
                else:
                    logging.info(f"   [Cache Miss] NN Feature shape mis-match. Retraining...")
            except Exception:
                pass

        if not cache_valid_nn:
            logging.info(f"   Training Optimized Neural Network for {target_label} with config {target_architecture}...")
            model_nn = MLPRegressor(
                hidden_layer_sizes=target_architecture, activation='relu', solver='adam',
                alpha=0.001, learning_rate_init=0.001, max_iter=500,
                early_stopping=True, validation_fraction=0.1, random_state=42,
                n_iter_no_change=20
            )
            model_nn.fit(X_train_scaled, y_train)
            joblib.dump(model_nn, model_nn_path)
            joblib.dump(scaler, scaler_path)
            
        preds_nn = model_nn.predict(X_test_scaled)
        mae_nn = mean_absolute_error(y_test, preds_nn)
        rmse_nn = np.sqrt(mean_squared_error(y_test, preds_nn))
        r2_nn = r2_score(y_test, preds_nn)
        logging.info(f"      [Optimized Neural Network] MAE: {mae_nn:.4f} | RMSE: {rmse_nn:.4f} | R2: {r2_nn:.4f}")
        
        # --- Optimized XGBoost ---
        model_xgb_path = config.MODEL_DIR / f"optimized_{target_col}_xgboost.joblib"
        cache_valid_xgb = False
        
        if model_xgb_path.exists():
            try:
                cached_xgb = joblib.load(model_xgb_path)
                # Verify hyperparameters and feature shapes match the current code
                if hasattr(cached_xgb, 'n_features_in_') and cached_xgb.n_features_in_ == len(features):
                    target_estimators = xgb_params.get('n_estimators', 100)
                    if hasattr(cached_xgb, 'n_estimators') and cached_xgb.n_estimators == target_estimators:
                        cache_valid_xgb = True
                        model_xgb = cached_xgb
                        logging.info(f"   [Cache Hit] Pre-trained XGBoost shape & params match. Instantly loading...")
                    else:
                        logging.info(f"   [Cache Miss] XGBoost params mis-match (needs {target_estimators}). Retraining...")
                else:
                    logging.info(f"   [Cache Miss] XGBoost Feature shape mis-match. Retraining...")
            except Exception:
                pass

        if not cache_valid_xgb:
            logging.info(f"   Training Optimized XGBoost for {target_label}...")
            model_xgb = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
            model_xgb.fit(X_train, y_train)
            joblib.dump(model_xgb, model_xgb_path)
            
        preds_xgb = model_xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, preds_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
        r2_xgb = r2_score(y_test, preds_xgb)
        logging.info(f"      [Optimized XGBoost] MAE: {mae_xgb:.4f} | RMSE: {rmse_xgb:.4f} | R2: {r2_xgb:.4f}")
        
        # Save predictions from both models
        all_preds[f'{target_col}_actual'] = y_test.values
        all_preds[f'{target_col}_nn_pred'] = preds_nn
        all_preds[f'{target_col}_xgb_pred'] = preds_xgb
        
        # Keep temperature data for plots
        if target_col == 'temperature':
            temp_preds_xgb = preds_xgb
            temp_preds_nn = preds_nn
            temp_y_test = y_test
            temp_model_xgb = model_xgb
        
        # Track which model won
        best = "Neural Network" if mae_nn < mae_xgb else "XGBoost"
        logging.info(f"      >> Best for {target_label}: {best}")
        
        optimized_metrics[target_col] = {
            "label": target_label,
            "Optimized Neural Network": {"MAE": round(mae_nn, 4), "RMSE": round(rmse_nn, 4), "R2": round(r2_nn, 4)},
            "Optimized XGBoost": {"MAE": round(mae_xgb, 4), "RMSE": round(rmse_xgb, 4), "R2": round(r2_xgb, 4)},
            "best_model": best
        }
    
    # Save main temperature model for SHAP
    if temp_model_xgb is not None:
        joblib.dump(temp_model_xgb, config.MODEL_DIR / "optimized_xgboost.joblib")
    
    with open(config.METRICS_DIR / "optimized_regression_metrics.json", "w") as f:
        json.dump(optimized_metrics, f, indent=4)
    
    all_preds.to_csv(config.REPORTS_DIR / "predictions.csv", index=False)
    logging.info(f"   Saved dual-model predictions for {len(targets)} targets (XGBoost + NN).")
    
    # --- Base vs Optimized Comparison Table ---
    try:
        with open(config.METRICS_DIR / "base_regression_metrics.json", "r") as f:
            base_metrics = json.load(f)
        temp_base = base_metrics.get("temperature", {}).get("models", {})
        temp_opt = optimized_metrics.get("temperature", {})
        
        metrics_table = {}
        for m in ["RandomForest", "NeuralNetwork", "XGBoost"]:
            if m in temp_base:
                metrics_table[f"{m} (Base)"] = temp_base[m]
        if "Optimized Neural Network" in temp_opt:
            metrics_table["NN (Optimized)"] = temp_opt["Optimized Neural Network"]
        if "Optimized XGBoost" in temp_opt:
            metrics_table["XGB (Optimized)"] = temp_opt["Optimized XGBoost"]
        
        if metrics_table:
            results_df = pd.DataFrame(metrics_table).T
            print("\n" + format_research_table(results_df, "Final Model Performance Table (Temperature)") + "\n")
    except Exception as e:
        logging.warning(f"   Could not print comparison table: {e}")
    
    # --- Performance Plots (Temperature) ---
    plt.style.use('ggplot')
    
    if temp_y_test is not None:
        # 1. Residual Distribution
        res_xgb = temp_y_test - temp_preds_xgb
        res_nn = temp_y_test - temp_preds_nn
        plt.figure(figsize=(10, 6))
        sns.kdeplot(res_xgb, label='XGBoost Residuals', fill=True, color='blue', alpha=0.3)
        sns.kdeplot(res_nn, label='Neural Network Residuals', fill=True, color='green', alpha=0.3)
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Residual Distribution (Prediction Error: Actual - Predicted)')
        plt.xlabel('Error Range')
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "residual_distribution_plot.png")
        plt.close()
        
        # 2. Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(temp_y_test, temp_preds_xgb, alpha=0.3, label='XGBoost', color='blue')
        plt.scatter(temp_y_test, temp_preds_nn, alpha=0.3, label='Neural Network', color='green')
        plt.plot([temp_y_test.min(), temp_y_test.max()], [temp_y_test.min(), temp_y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Actual vs Predicted Temperature Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "actual_vs_predicted_plot.png")
        plt.close()
    
    # 3. Feature Importance
    if temp_model_xgb is not None:
        plt.figure(figsize=(10, 8))
        plot_importance(temp_model_xgb, max_num_features=15, importance_type='weight', title='Feature Importance (Tuned XGBoost Model)')
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "feature_importance_plot.png")
        plt.close()
        
        # 3b. Native F-Score Feature Importance (separate plot for dashboard gallery)
        plt.figure(figsize=(10, 8))
        plot_importance(temp_model_xgb, max_num_features=15, importance_type='gain', title='Native Feature Importance (F-Score / Gain)')
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "feature_importance_native_fscore.png")
        plt.close()
    
    # 4. Model Comparison Plot
    try:
        temp_m = optimized_metrics.get("temperature", {})
        models_list = ["Optimized XGBoost", "Optimized Neural Network"]
        mae_vals = [temp_m.get(m, {}).get("MAE", 0) for m in models_list]
        r2_vals = [temp_m.get(m, {}).get("R2", 0) for m in models_list]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        bar_width = 0.35
        index = np.arange(len(models_list))
        ax1.bar(index, mae_vals, bar_width, label='MAE (Lower is Better)', color='skyblue', alpha=0.8)
        ax2.plot(index, r2_vals, color='darkred', marker='o', label='R2 Score (Higher is Better)')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('MAE')
        ax2.set_ylabel('R2 Score')
        plt.title('Final Performance Comparison: XGBoost vs Neural Network')
        ax1.set_xticks(index)
        ax1.set_xticklabels(models_list)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "model_comparison_plot.png")
        plt.close()
    except Exception as e:
        logging.warning(f"   Could not generate Comparison plot: {e}")
    
    tracker.end("Step 8: Optimized Model Training")


# Step 9: Explainable AI (SHAP)
def step_explainability():
    tracker.start("Step 9: Explainable AI (SHAP)")
    plt.style.use('default')
    
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists() or not (config.MODEL_DIR / "optimized_xgboost.joblib").exists():
        tracker.end("Step 9: Explainable AI (SHAP)", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    sample_df = df.sample(n=min(500, len(df)), random_state=42)
    
    targets = ['temperature']
    non_feature_cols = ['datetime', 'region', 'season_class'] + targets
    features = [c for c in sample_df.columns if c not in non_feature_cols]
    X_sample = sample_df[features]
    
    model = joblib.load(config.MODEL_DIR / "optimized_xgboost.joblib")
    
    logging.info("   Generating SHAP values for global explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_summary_plot.png")
    plt.close()
    
    tracker.end("Step 9: Explainable AI (SHAP)")

# Step 10: Anomaly Detection
def step_anomaly_detection():
    tracker.start("Step 10: Anomaly Detection")
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    if not features_path.exists():
        tracker.end("Step 10: Anomaly Detection", status="FAIL")
        return
        
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # Calculate Regional Z-Scores for strict anomalies
    mean_temp = df.groupby('region')['temperature'].transform('mean')
    std_temp = df.groupby('region')['temperature'].transform('std')
    
    df['temp_zscore'] = (df['temperature'] - mean_temp) / std_temp
    
    # Flag anomalies as |Z| > 3.0 (Extreme Events)
    df['is_anomaly'] = df['temp_zscore'].abs() > 3.0
    anomalies = df[df['is_anomaly']]
    
    logging.info(f"   Detected {len(anomalies)} severe weather anomalies across all regions.")
    
    # 10-City Subplot Grid (Dissertation Feature)
    plt.style.use('ggplot')
    regions = list(config.REGIONS.keys())
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        ax = axes[i]
        region_df = df[df['region'] == region]
        ax.plot(region_df['datetime'], region_df['temperature'], label='Temperature', color='blue', alpha=0.3)
        
        region_anomalies = region_df[region_df['is_anomaly']]
        if not region_anomalies.empty:
            ax.scatter(region_anomalies['datetime'], region_anomalies['temperature'], color='red', label='Anomaly', s=15, zorder=5)
        
        ax.set_title(f"{region} (|Z| > 3.0)")
        if i == 0: ax.legend()
            
    fig.suptitle("Multi-City Extreme Weather Anomalies (2001-2024)", fontsize=18, y=1.00)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "multi_city_anomalies.png")
    plt.close()
    
    # Middlesbrough Focused Plot (Per Rules)
    plt.figure(figsize=(12, 6))
    mboro_df = df[df['region'] == 'Middlesbrough']
    plt.plot(mboro_df['datetime'], mboro_df['temperature'], label='Temperature', color='blue', alpha=0.4)
    mboro_anomalies = mboro_df[mboro_df['is_anomaly']]
    if not mboro_anomalies.empty:
        plt.scatter(mboro_anomalies['datetime'], mboro_anomalies['temperature'], color='red', label='Anomaly', s=20, zorder=5)
    
    plt.title("Middlesbrough Temperature Anomaly Detection (|Z| > 3.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "middlesbrough_anomalies.png")
    plt.close()
    
    tracker.end("Step 10: Anomaly Detection")

# Step 11: Export & Summary
def step_export_summary():
    tracker.start("Step 11: Export & Summary")
    
    summary_data = {}
    
    # 1. Load base metrics
    try:
        with open(config.METRICS_DIR / "base_regression_metrics.json", "r") as f:
            base_metrics = json.load(f)
            temp_base = base_metrics.get("temperature", {}).get("models", {})
            if "RandomForest" in temp_base: summary_data["Base RandomForest"] = temp_base["RandomForest"]
            if "XGBoost" in temp_base: summary_data["Base XGBoost"] = temp_base["XGBoost"]
            if "NeuralNetwork" in temp_base: summary_data["Base NeuralNetwork"] = temp_base["NeuralNetwork"]
    except: pass
    
    # 2. Load optimized metrics
    winner_name = "XGBoost"
    try:
        with open(config.METRICS_DIR / "optimized_regression_metrics.json", "r") as f:
            opt = json.load(f)
            temp_opt = opt.get("temperature", {})
            if "Optimized XGBoost" in temp_opt: summary_data["Optimized XGBoost"] = temp_opt["Optimized XGBoost"]
            if "Optimized Neural Network" in temp_opt: summary_data["Optimized Neural Network"] = temp_opt["Optimized Neural Network"]
            winner_name = temp_opt.get("best_model", "XGBoost")
    except: pass
    
    final_df = None
    if summary_data:
        final_df = pd.DataFrame(summary_data).T
        
        # Restore the missing CSV export from the old project
        csv_path = config.REPORTS_DIR / "final_model_performance_summary.csv"
        final_df.to_csv(csv_path, index_label="Model")
        
    # 2.5 Generate Dashboard KPIs (to avoid loading 480MB CSV in JS)
    try:
        master_df_path = config.RAW_DIR / "weather_master_dataset.csv"
        if master_df_path.exists():
            kpi_data = pd.read_csv(master_df_path, usecols=['datetime'])
            kpis = {
                "total_records": len(kpi_data),
                "last_update": str(kpi_data['datetime'].max()),
                "region_count": len(config.REGIONS),
                "model_status": "Calibrated & Operational"
            }
            with open(config.METRICS_DIR / "dashboard_kpis.json", "w") as f:
                json.dump(kpis, f, indent=4)
            logging.info(f"   Generated dashboard_kpis.json for UI performance.")
    except Exception as e:
        logging.warning(f"   Could not generate Dashboard KPIs: {e}")
    
    # Copy KPIs to dashboard/data/ for direct access from the UI
    try:
        import shutil
        kpi_src = config.METRICS_DIR / "dashboard_kpis.json"
        kpi_dst = config.WEB_DATA_DIR / "dashboard_kpis.json"
        if kpi_src.exists():
            shutil.copy2(kpi_src, kpi_dst)
            logging.info(f"   Copied dashboard_kpis.json to {config.WEB_DATA_DIR.name}/")
    except Exception as e:
        logging.warning(f"   Could not copy KPIs to dashboard/data: {e}")
        
    # 3. Detailed Summary Export
    summary_txt = config.REPORTS_DIR / "project_summary.txt"
    plot_count = len(glob.glob(str(config.PLOTS_DIR / "*.png")))
    model_count = len(glob.glob(str(config.MODEL_DIR / "*.joblib")))
    report_files = [os.path.basename(f) for f in glob.glob(str(config.REPORTS_DIR / "*.csv"))]

    with open(summary_txt, "w") as f:
        f.write(f"Seasonal Weather Prediction System - Research Export\n")
        f.write(f"====================================================\n")
        f.write(f"Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GMT\n")
        f.write(f"Data Range: {config.START_DATE} to {config.END_DATE} (Hourly)\n")
        f.write(f"Prediction: Single-Target (Temperature)\n\n")
        
        if final_df is not None:
            f.write(" - Final Model Performance Table (Temperature):\n")
            for model_name, row in final_df.iterrows():
                f.write(f"   {model_name:<25} | MAE: {row['MAE']:.4f}, RMSE: {row['RMSE']:.4f}, R2: {row['R2']:.4f}\n")
                if "Optimized" in str(model_name):
                    f.write(f"   MODEL_METRIC::{model_name} | MAE={row['MAE']:.4f}\n")
        
        f.write(f"\n - Resource Inventory:\n")
        f.write(f"   Models:   {model_count} calibrated files saved to /models\n")
        f.write(f"   Figures:  {plot_count} research plots in /outputs/plots\n")
        f.write(f"   Reports:  {', '.join(report_files)}\n\n")
        
        f.write(f" - Feature Engineering Taxonomy:\n")
        f.write(f"   - Temporal Gaps (1h, 6h, 24h): Captures immediate weather memory.\n")
        f.write(f"   - Rolling Trends (6h, 24h): Smooths volatility to see regional gradients.\n")
        f.write(f"   - Season Class: Astronomical season mapping (0-Winter to 3-Autumn).\n\n")
        
        f.write(f" - Pipeline Success Log:\n")
        for step in tracker.steps:
            f.write(f"   - {step['name']}: {step['status']} ({step['fmt_dur']})\n")
        
        f.write(f"\nProject Execution Successful. Current Best Model: {winner_name}\n")
            
    # Persistent Run Log (detailed entry per run)
    log_file = config.LOGS_DIR / "run.log"
    with open(log_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Range: {config.START_DATE} to {config.END_DATE}\n")
        for step in tracker.steps:
            f.write(f"  {step['name']}: {step['status']} ({step['fmt_dur']})\n")
        if final_df is not None:
            f.write(f"Model Scores (Temperature):\n")
            for model_name, row in final_df.iterrows():
                f.write(f"  {model_name:<25} MAE: {row['MAE']:.4f} | RMSE: {row['RMSE']:.4f} | R2: {row['R2']:.4f}\n")
        f.write(f"Best Model: {winner_name}\n")
        f.write(f"{'='*70}\n")
        
    logging.info(f"   Saved research summary to {summary_txt.name}")
    logging.info("   The system is now fully prepared for the UI Dashboard Phase.")
    tracker.end("Step 11: Export & Summary")

def main():
    tracker.start("Step 1: Import & Setup")
    logging.info(f"   Project initialized at {config.BASE_DIR}")
    tracker.end("Step 1: Import & Setup")
    
    step_fetch_data() 
    step_validate_eda()
    step_feature_engineering()
    step_model_training()
    step_season_classification()
    step_hyperparameter_tuning()
    step_optimized_model()
    step_explainability()
    step_anomaly_detection()
    step_export_summary()
    tracker.print_summary()

if __name__ == "__main__":
    main()
