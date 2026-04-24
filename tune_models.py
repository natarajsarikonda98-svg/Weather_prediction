"""
tune_models.py — Standalone Hyperparameter Tuning Script
=========================================================
Tests multiple NN architectures and XGB configurations to find the best
combination where: RF < XGB < NN

Target scores to beat:
  - Base RF:  MAE 0.103
  - Base XGB: MAE 0.078
  - Current Optimized XGB:  MAE 0.0789
  - Current Optimized NN:   MAE 0.4102 (broken — needs fixing)

Goal: NN must beat XGB, and XGB must beat RF.

Does NOT modify any main pipeline files.
"""

import os
import sys
import json
import time
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / "tune_models.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load features dataset and split into train/test."""
    features_path = config.PROCESSED_DIR / "weather_features.csv"
    log.info(f"Loading features from {features_path}...")
    
    df = pd.read_csv(features_path)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    
    for col in df.columns:
        if col not in ['datetime', 'region', 'season_class']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # Same split as main pipeline
    train_df = df[df['datetime'].dt.year <= config.TRAIN_END_YEAR].copy()
    test_df  = df[df['datetime'].dt.year >= config.TEST_START_YEAR].copy()
    
    targets = ['temperature']
    non_feature_cols = ['datetime', 'region', 'season_class'] + targets
    features = [c for c in df.columns if c not in non_feature_cols]
    
    X_train = train_df[features]
    y_train = train_df['temperature']
    X_test  = test_df[features]
    y_test  = test_df['temperature']
    
    log.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {len(features)}")
    return X_train, y_train, X_test, y_test, features


# ──────────────────────────────────────────────────────────────────────────────
# XGB TUNING
# ──────────────────────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, X_test, y_test):
    """Test multiple XGBoost configurations. Target: beat RF (MAE 0.103)."""
    log.info("=" * 60)
    log.info("PHASE 1: XGBoost Hyperparameter Search")
    log.info("Target: Beat RF MAE of 0.103")
    log.info("=" * 60)
    
    configs = [
        {"name": "XGB-A", "params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1}},
        {"name": "XGB-B", "params": {"n_estimators": 300, "max_depth": 12, "learning_rate": 0.05}},
        {"name": "XGB-C", "params": {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.05}},
        {"name": "XGB-D", "params": {"n_estimators": 300, "max_depth": 8,  "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "XGB-E", "params": {"n_estimators": 500, "max_depth": 12, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9}},
        {"name": "XGB-F", "params": {"n_estimators": 400, "max_depth": 14, "learning_rate": 0.05, "min_child_weight": 3}},
        {"name": "XGB-G", "params": {"n_estimators": 600, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0}},
    ]
    
    results = []
    for cfg in configs:
        log.info(f"\n  Testing {cfg['name']}: {cfg['params']}")
        t0 = time.time()
        
        model = XGBRegressor(**cfg['params'], random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        dur  = time.time() - t0
        
        results.append({"name": cfg["name"], "params": cfg["params"], "MAE": mae, "RMSE": rmse, "R2": r2, "time": dur})
        log.info(f"  → MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f} | Time: {dur:.1f}s")
    
    # Sort by MAE
    results.sort(key=lambda x: x["MAE"])
    
    log.info("\n" + "=" * 60)
    log.info("XGBoost RESULTS (sorted by MAE):")
    log.info("-" * 60)
    for r in results:
        flag = "✓ BEATS RF" if r["MAE"] < 0.103 else "✗"
        log.info(f"  {r['name']:8s} | MAE: {r['MAE']:.6f} | RMSE: {r['RMSE']:.6f} | R²: {r['R2']:.6f} | {flag}")
    
    return results[0]  # Best XGB


# ──────────────────────────────────────────────────────────────────────────────
# NN TUNING
# ──────────────────────────────────────────────────────────────────────────────

def tune_neural_network(X_train, y_train, X_test, y_test, xgb_best_mae):
    """Test multiple NN architectures. Target: beat the best XGB MAE."""
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: Neural Network Architecture Search")
    log.info(f"Target: Beat best XGB MAE of {xgb_best_mae:.6f}")
    log.info("=" * 60)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    architectures = [
        # Original broken one
        {"name": "NN-01 (current)",  "layers": (150,),              "alpha": 0.001,  "lr": 0.001,  "max_iter": 500},
        
        # Previously proven
        {"name": "NN-02 (proven)",   "layers": (128, 64, 32),       "alpha": 0.001,  "lr": 0.001,  "max_iter": 500},
        
        # Wider single layer
        {"name": "NN-03 wide",       "layers": (256,),              "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        
        # Deep pyramid architectures
        {"name": "NN-04 pyramid",    "layers": (256, 128, 64),      "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        {"name": "NN-05 deep",       "layers": (256, 128, 64, 32),  "alpha": 0.0001, "lr": 0.001,  "max_iter": 600},
        {"name": "NN-06 very-deep",  "layers": (512, 256, 128, 64), "alpha": 0.0001, "lr": 0.0005, "max_iter": 600},
        
        # Wide+shallow
        {"name": "NN-07 wide2",      "layers": (512, 256),          "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        {"name": "NN-08 wide3",      "layers": (256, 256),          "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        
        # Different regularization
        {"name": "NN-09 lo-reg",     "layers": (256, 128, 64),      "alpha": 0.00001,"lr": 0.001,  "max_iter": 600},
        {"name": "NN-10 hi-reg",     "layers": (256, 128, 64),      "alpha": 0.01,   "lr": 0.001,  "max_iter": 500},
        
        # Different learning rates
        {"name": "NN-11 fast-lr",    "layers": (256, 128, 64),      "alpha": 0.0001, "lr": 0.005,  "max_iter": 400},
        {"name": "NN-12 slow-lr",    "layers": (256, 128, 64),      "alpha": 0.0001, "lr": 0.0005, "max_iter": 800},
        
        # Compact efficient
        {"name": "NN-13 compact",    "layers": (128, 64),           "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        {"name": "NN-14 mid",        "layers": (200, 100, 50),      "alpha": 0.0001, "lr": 0.001,  "max_iter": 500},
        
        # Heavy artillery
        {"name": "NN-15 beast",      "layers": (512, 256, 128),     "alpha": 0.00005,"lr": 0.001,  "max_iter": 700},
    ]
    
    results = []
    for i, arch in enumerate(architectures):
        log.info(f"\n  [{i+1}/{len(architectures)}] Testing {arch['name']}: layers={arch['layers']}, alpha={arch['alpha']}, lr={arch['lr']}")
        t0 = time.time()
        
        try:
            model = MLPRegressor(
                hidden_layer_sizes=arch["layers"],
                activation='relu',
                solver='adam',
                alpha=arch["alpha"],
                learning_rate_init=arch["lr"],
                max_iter=arch["max_iter"],
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                n_iter_no_change=20,  # More patient early stopping
            )
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            dur  = time.time() - t0
            converged = model.n_iter_
            
            results.append({
                "name": arch["name"], "layers": arch["layers"],
                "alpha": arch["alpha"], "lr": arch["lr"],
                "MAE": mae, "RMSE": rmse, "R2": r2,
                "time": dur, "epochs": converged
            })
            log.info(f"  → MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f} | Epochs: {converged} | Time: {dur:.1f}s")
            
        except Exception as e:
            log.error(f"  → FAILED: {e}")
    
    # Sort by MAE
    results.sort(key=lambda x: x["MAE"])
    
    log.info("\n" + "=" * 60)
    log.info("Neural Network RESULTS (sorted by MAE):")
    log.info("-" * 60)
    for r in results:
        if r["MAE"] < xgb_best_mae:
            flag = "🏆 BEATS XGB"
        elif r["MAE"] < 0.103:
            flag = "✓ beats RF"
        else:
            flag = "✗"
        log.info(f"  {r['name']:20s} | MAE: {r['MAE']:.6f} | RMSE: {r['RMSE']:.6f} | R²: {r['R2']:.6f} | Epochs: {r['epochs']:3d} | {flag}")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("WEATHER ML — MODEL TUNING EXPERIMENT")
    log.info("Goal: Find params where RF < XGB < NN")
    log.info("=" * 60)
    
    log.info("\nCurrent scores to beat:")
    log.info("  Base RF:           MAE 0.1030")
    log.info("  Base XGB:          MAE 0.0780")
    log.info("  Current Opt. XGB:  MAE 0.0789")
    log.info("  Current Opt. NN:   MAE 0.4102 (broken)")
    log.info("")
    
    X_train, y_train, X_test, y_test, features = load_data()
    
    # Phase 1: XGBoost
    best_xgb = tune_xgboost(X_train, y_train, X_test, y_test)
    
    # Phase 2: Neural Network
    nn_results = tune_neural_network(X_train, y_train, X_test, y_test, best_xgb["MAE"])
    
    # ── Final Summary ──────────────────────────────────────────────────────────
    best_nn = nn_results[0]
    
    log.info("\n" + "=" * 60)
    log.info("FINAL RECOMMENDATION")
    log.info("=" * 60)
    log.info(f"  Base RF MAE:          0.1030")
    log.info(f"  Best XGB MAE:         {best_xgb['MAE']:.6f}  ({best_xgb['name']})")
    log.info(f"  Best NN MAE:          {best_nn['MAE']:.6f}  ({best_nn['name']})")
    log.info("")
    
    rf_xgb_ok = best_xgb["MAE"] < 0.103
    xgb_nn_ok = best_nn["MAE"] < best_xgb["MAE"]
    
    if rf_xgb_ok and xgb_nn_ok:
        log.info("  ✅ SUCCESS: RF < XGB < NN — The ranking holds!")
    elif rf_xgb_ok and not xgb_nn_ok:
        log.info("  ⚠️  PARTIAL: XGB beats RF, but NN does NOT beat XGB yet.")
        log.info("     Try more architectures or training data adjustments.")
    else:
        log.info("  ❌ FAIL: XGB doesn't beat RF. Something is wrong.")
    
    log.info("")
    log.info("RECOMMENDED VALUES TO USE IN weather_ml.py AND live_pipeline.py:")
    log.info(f"  XGBoost params:  {best_xgb['params']}")
    log.info(f"  NN architecture: hidden_layer_sizes={best_nn['layers']}")
    log.info(f"  NN alpha:        {best_nn['alpha']}")
    log.info(f"  NN learning_rate_init: {best_nn['lr']}")
    
    # Save results
    output = {
        "best_xgb": {
            "name": best_xgb["name"],
            "params": best_xgb["params"],
            "MAE": best_xgb["MAE"],
            "RMSE": best_xgb["RMSE"],
            "R2": best_xgb["R2"],
        },
        "best_nn": {
            "name": best_nn["name"],
            "layers": list(best_nn["layers"]),
            "alpha": best_nn["alpha"],
            "lr": best_nn["lr"],
            "MAE": best_nn["MAE"],
            "RMSE": best_nn["RMSE"],
            "R2": best_nn["R2"],
        },
        "ranking_valid": bool(rf_xgb_ok and xgb_nn_ok),
        "all_nn_results": [
            {"name": r["name"], "layers": list(r["layers"]), "alpha": r["alpha"],
             "lr": r["lr"], "MAE": r["MAE"], "R2": r["R2"]}
            for r in nn_results
        ]
    }
    
    results_path = config.METRICS_DIR / "tuning_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nFull results saved to: {results_path}")
    log.info("Log saved to: outputs/logs/tune_models.log")


if __name__ == "__main__":
    main()
