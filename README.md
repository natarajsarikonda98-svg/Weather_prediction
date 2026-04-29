# MSc Computing Dissertation Project: Machine Learning–Based Seasonal Weather Prediction Across Regions

This repository contains a professional implementation for predicting regional weather patterns across the UK. The project evaluates **Optimized XGBoost** and **Optimized Neural Network** architectures and provides a high-fidelity dashboard for result visualization.


## 📂 File Inventory & Purpose

| File / Folder | Purpose |
| :--- | :--- |
| **`run.py`** | The main orchestrator. Launches the local API server, starts the ML pipeline, and opens the dashboard. |
| **`server.py`** | A lightweight HTTP bridge that routes JavaScript prediction requests into the Python ML inference engine (`predict_live.py`). |
| **`config.py`** | Central configuration for all file paths, API coordinates, and logical boundaries. |
| **`requirements.txt`** | List of all Python dependencies required to run the project. |
| **`src/`** | Contains the core ML engines: `weather_ml.py` (Base ML), `live_pipeline.py` (Daily Daemon), `predict_live.py` (O(1) Inference Engine), and `force_retrain_hourly.py` (Drift Evaluator). |
| **`models/`** | Contains optimized model weights (`.joblib`). Allows the system to warm-start instantly without retraining 24 years of data. |
| **`outputs/metrics/`** | Stores the `drift_history.csv` which tracks AI performance drift, and pre-calculated dashboard JSON KPIs. |
| **`dashboard/`** | Contains the premium glassmorphism frontend interface (HTML/V-JS/CSS). |

## 1. Installation

Ensure you are using Python 3.9+ and have an active virtual environment.

```bash
# To activate the virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

## 2. Running the Complete System

To launch the Live Pipeline, API Server, and Interactive Dashboard simultaneously:

```bash
python run.py
```

**System Behavior:**
1.  **Backend Web Target**: Starts the non-blocking `server.py` on `localhost:8000`.
2.  **Live Daemon**: The pipeline checks for "Yesterday's" data and handles any missing historical blocks.
3.  **Dashboard Launch**: Opens the live dashboard, which is now fully synced with the backend Python processes.

> [!IMPORTANT]
> **Startup Prompts:** When you run `python run.py`, the terminal will ask if you want to retrain the base models or re-tune hyperparameters. 
> - Type **`n`** (no) for both to boot the dashboard instantly using the pre-calculated models included in the repo. 
> - Type **`y`** (yes) if you wish to re-perform the 35-minute scientific training cycle from scratch.

## 3. Key Academic Features
*   **Explainable AI (SHAP):** Resolves the black-box problem by highlighting feature importance.
*   **Concept Drift Mitigation:** Autonomously tracks MAE degradation over time as climate patterns shift.
*   **Robust Exception Management:** The UI is protected from JSON crashes even during unexpected data interruptions, demonstrating professional error management.
*   **14-Day Scientific Validation:** Dynamic comparison of baseline ECMWF/GFS forecasts against Custom Neural Network inferences.
