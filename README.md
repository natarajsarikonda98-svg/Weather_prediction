# MSc Computing Desertation Project: Machine Learning–Based Seasonal Weather Prediction Across Regions

This repository contains a implementation for predicting regional weather patterns across the UK. The project evaluates **Optimized XGBoost** and **Optimized Neural Network** architectures and provides a high-fidelity glassmorphism dashboard for result visualization.

## Architecture & Workflow

The project follows a rigorous Step wise Machine Learning Pipeline:
1.  **Data Acquisition**: Real-time fetching from the Open-Meteo Historical API (2015-2025).
2.  **Exploratory Data Analysis (EDA)**: Correlation heatmaps and missing value audits.
3.  **Feature Engineering**: Advanced temporal lags (Lag_1, Lag_7), rolling averages (Week/Month), and astronomical season classification.
4.  **Optimized Training**: Hyperparameter-tuned models with RandomizedSearchCV and 3-fold Cross-Validation.
5.  **Interpretability**: Full SHAP Analysis to explain model "Black-Box" decisions.
6.  **Anomaly Detection**: Statistical Z-Score detection (|Z| > 2.5) for extreme regional weather events (Focused on Middlesbrough for study purpose).

---

## Folder Structure
- `data/`: Raw API caches and processed ML-ready features.
- `models/`: Optimized production-ready model artifacts (.joblib).
- `src/weather_ml.py`: The core 11-Step ML Backend Engine.
- `dashboard/`: Frontend glassmorphism interface (HTML/JS/CSS).
- `outputs/`: 
    - `plots/`: All visual outputs (like SHAP, Trends, Anomalies, Feature Importance).
    - `reports/`: Academic summaries and performance CSVs.
- `run.py`: The main project orchestrator.
- `venv` : All the requirements to activate virtual environment.

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
