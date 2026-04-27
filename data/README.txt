       Time-Series Data Analysis and Trend Discovery in Pakistan's Crop Prices

Course      : Data Mining
Institution : FAST NUCES
Team Members:
              - Taha Nawaz    (23L-2644)
              - Ali Ahmad     (23L-2619)
              - Shahzeb Imran (23L-2506)

PROJECT OVERVIEW
================================================================================

This project performs end-to-end time-series analysis on historical crop price
data from Pakistan's agricultural markets. The goal is to uncover pricing
trends, forecast future prices, and detect anomalous market behaviour across
multiple crops and cities.

The pipeline is organized into three phases:

  PHASE 1 — Data Loading, Cleaning, EDA, Decomposition & Statistical Analysis
  PHASE 2 — Feature Engineering, Modeling & Evaluation
  PHASE 3 — Clustering, Anomaly Detection & Pattern Discovery

DATASET
================================================================================

Source  : Kaggle — Crop Prices Dataset of Pakistan
Link    : https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan

The dataset consists of 53 CSV files, each containing daily crop price records.

Required schema per file:
  - City    : Name of the city/market
  - Date    : Date of the recorded price
  - Crop    : Crop name
  - Price   : Recorded price (PKR)

All 53 CSV files must be placed in a single folder before running the notebook.
The DATA_DIR variable in the Configuration cell (Cell 4) must be updated to
point to that folder on your local machine.


PROJECT STRUCTURE
================================================================================

Time-Series-Data-Analysis-and-Trend-Discovery-in-Pakistan-Crop-Prices/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│	│
│	└──DM_Project_Final_Deliverable.ipynb 
│	│
│	└──DM_Project_Final_Deliverable.ipynb   ← Main notebook (all phases)
├── src/
│   ├── data_preprocessing.py
│   ├── modeling.py
│   ├── time_series.py
│   └── clustering.py
├── results/
│	│
│	└──outputs/
│		├──arima_results.csv
│    		├── hw_results.csv
│    		├── baseline_results.csv
│    		├── rf_tuned_results.csv
│    		├── xgb_tuned_results.csv
│    		└── Comparison/model_comparison.csv
│	└── figures/
│    		├── EDA/                             ← 11 exploratory analysis charts
│    		├── Decomposition/                   ← STL decomposition & ACF/PACF plots
│    		├── Comparison/                      ← Model comparison, global vs local
│    		├── CLUSTERING/                      ← K-Means, hierarchical, anomaly plots│
│		└──  Modeling/                        ← Baseline model output, ARIMA & Holt-Winters outputs,
│							Feature snapshot visualization, Feature importance plots (RF & XGBoost)
│							Residuals, RMSE heatmap, temporal error, Actual vs Predicted, forecast horizons
│	
└── data/
    ├── Dataset
    └── README.md

REQUIREMENTS
================================================================================

Python Version : 3.12+

Install all dependencies via pip:

  pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels psutil

Full list of libraries used:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn  (LinearRegression, RandomForest, StandardScaler, PCA, etc.)
  - xgboost
  - statsmodels   (ARIMA, Holt-Winters, ADF test, seasonal decomposition)
  - psutil        (memory monitoring)


HOW TO RUN
================================================================================

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan

2. Extract all 53 CSV files into a single folder on your machine.

3. Open DM_Project_Final_Deliverable.ipynb in Jupyter Notebook or JupyterLab.

4. In Cell 4 (Configuration), update the following paths:

     DATA_DIR    = r"path\to\your\csv_folder"
     MERGED_FILE = r"path\to\save\merged_output.csv"    # optional

5. Run all cells from top to bottom (Kernel → Restart & Run All).

   NOTE: Cells must be executed in order. Each phase depends on variables
   defined in the previous one. Do not skip any section.

6. All output figures are saved automatically to the outputs/ subdirectories.


METHODOLOGY SUMMARY
================================================================================

PHASE 1 — Data Preparation
  - Loaded and validated 53 CSV files against required schema
  - Removed zero/negative prices; flagged and Winsorized IQR outliers per
    (Crop, City) group — no rows dropped for outliers
  - Extracted calendar features (Year, Month, Quarter, Day-of-week)
  - Filtered series with fewer than 100 observations
  - 11 EDA visualizations: distributions, trends, seasonality, volatility,
    cross-crop correlations
  - STL seasonal decomposition + ADF stationarity tests per top (crop, city)pair.

PHASE 2 — Modeling & Evaluation
  - Baseline models: Naive (last value) and Seasonal Naive
  - Statistical models: ARIMA, Holt-Winters Exponential Smoothing
  - ML models: Linear Regression, Random Forest, XGBoost
  - Recursive forecasting at 1-month, 3-month, and 6-month horizons
  - Hyperparameter tuning via grid search on validation set (no leakage)
  - Evaluation metrics: MAE, RMSE, MAPE
  - Global vs Local XGBoost model comparison (scalability vs accuracy)
  - Feature importance analysis for Random Forest and XGBoost

PHASE 3 — Clustering & Anomaly Detection
  - Feature engineering per (Crop, City) pair: mean, std, OLS trend slope,
    coefficient of variation, seasonality spread, rolling volatility
  - K-Means clustering with Elbow + Silhouette for optimal K selection
  - Hierarchical/Agglomerative clustering for comparison (dendrogram)
  - PCA 2D projection for cluster visualization
  - Anomaly detection via Z-score, Rolling Deviation, and IQR methods
  - Integration: cluster type vs model RMSE, anomaly rate vs prediction error


OUTPUT FILES
================================================================================

CSV Results:
  arima_results.csv         — ARIMA forecast results per crop
  hw_results.csv            — Holt-Winters forecast results per crop
  baseline_results.csv      — Naive and Seasonal Naive results
  rf_tuned_results.csv      — Tuned Random Forest evaluation results
  xgb_tuned_results.csv     — Tuned XGBoost evaluation results
  model_comparison.csv      — Consolidated comparison across all models

Key Visualizations (inside outputs/):
  EDA/                      — Price distributions, trends, seasonality
  Decomposition/            — Per-crop trend/seasonal/residual components
  Comparison/               — Actual vs Predicted per crop-city pair
  CLUSTERING/               — Cluster profiles, PCA scatter, anomaly plots


NOTES
================================================================================

- The notebook is memory-conscious: dtype optimization and gc.collect() calls
  are used throughout to handle large merged datasets efficiently.
- Outliers are Winsorized (capped), not dropped, to preserve time-series
  continuity for forecasting.
- All train/val/test splits are strictly chronological — no shuffling.
- Figures are saved at 100 DPI. High-resolution export can be enabled by
  changing the dpi parameter in save_and_show().
