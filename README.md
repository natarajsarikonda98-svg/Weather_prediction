# MSc Computing Dissertation Project: Machine Learning–Based Seasonal Weather Prediction Across Regions

This repository contains a professional implementation for predicting regional weather patterns across the UK. The project evaluates **Optimized XGBoost** and **Optimized Neural Network** architectures and provides a high-fidelity dashboard for result visualization.

## 📂 File Inventory & Purpose

| File / Folder | Purpose |
| :--- | :--- |
| **`run.py`** | The main orchestrator. Launches the local server, starts the ML pipeline, and opens the dashboard. |
| **`config.py`** | Central configuration for all file paths, API coordinates, and model hyperparameters. |
| **`requirements.txt`** | List of all Python dependencies required to run the project. |
| **`src/weather_ml.py`** | The core Machine Learning logic (Feature Engineering & Model Training). |
| **`src/live_pipeline.py`** | The "Catch-Up" engine that handles daily retraining and live data ingestion. |
| **`models/`** | Contains optimized model weights (.joblib). These allow the system to "warm-start" without retraining from year one. |
| **`outputs/metrics/`** | Stores the `drift_history.csv` which acts as a timestamp bookmark for retraining. |
| **`dashboard/`** | Contains the frontend interface (HTML/JS/CSS) for the weather visualization. |
| **`fetch_dataset.py`** | Utility to download historical archives for initial project setup. |
| **`tune_models.py`** | Script used to conduct the hyperparameter optimization (Randomized Search). |
| **`analyze_nulls.py`** | Data quality audit script used to verify dataset integrity. |

## Architecture & Workflow

---

## Active virtual environment

```bash
.\weather_ml_project\venv\Scripts\Activate.ps1" to activate
```

## 1. Installation

Ensure you are using Python 3.9+ and have an active virtual environment.

```bash
pip install -r requirements.txt
```

## 2. Running the Complete System

To launch the 11-step pipeline and the interactive dashboard simultaneously:

```bash
python run.py
```

**System Behavior:**
1.  **Backend Web Server**: Starts in the background on `localhost:8000`.
2.  **ML Pipeline**: Executes the 11-step process (Fetch -> Engineer -> Train -> Detect).
3.  **Dashboard Launch**: Once the pipeline successfully completes, your default browser will automatically open the **Seasonal Weather Dashboard**.

---

## 3. Key Findings & Analytics
- **Optimized Performance**: The optimized Neural Network and XGBoost models consistently achieve MAE < 1.30 on unseen 2022-2025 data.
- **Academic Rigor**: The project includes a "Feature Taxonomy" in the final summary report, explaining the scientific rationale behind every engineered signal.
- **Regional Focus**: While data is processed for 10 regions, the anomaly detection visualizes Middlesbrough as primary for case study.
