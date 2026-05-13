# 🌾 Time-Series Analysis & Trend Discovery in Pakistan's Crop Prices

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-brightgreen?style=flat-square)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%20%7C%20HW-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

**An end-to-end data mining pipeline for agricultural price forecasting, trend decomposition, and market anomaly detection across Pakistan's commodity markets (2008–2024).**

[Dataset](#-dataset) · [Methodology](#-methodology) · [Results](#-results--key-findings) · [How to Run](#-how-to-run) · [Reports](#-reports)

</div>

---

## 📌 Overview

This project presents a full-lifecycle data mining study on historical crop price records sourced from Pakistan's agricultural markets. Spanning **16 years (2008–2024)**, **138 cities**, and **76 crop varieties**, the dataset comprises approximately **7.99 million records** across 53 CSV files.

The pipeline is structured across three analytical phases:

| Phase | Focus |
|-------|-------|
| **Phase 1** | Data Ingestion, Cleaning, EDA, STL Decomposition & Stationarity Testing |
| **Phase 2** | Feature Engineering, Forecasting Models (Baseline → Statistical → ML), Evaluation |
| **Phase 3** | Clustering, Anomaly Detection & Pattern Discovery |

> **Course:** Data Mining · **Institution:** FAST NUCES, Lahore · **Team:** Ali Ahmad · Taha Nawaz · Shahzeb Imran

---

## 📂 Repository Structure

```
Time-Series-Data-Analysis-and-Trend-Discovery-in-Pakistan-Crop-Prices/
│
├── notebooks/
│   ├── DM_Project_Deliverable_1.ipynb        ← Phase 1 notebook
│   └── DM_Project_Final_Deliverable.ipynb    ← Main notebook (all phases)
│
├── reports/
│   ├── Data Mining Project Proposal.pdf
│   ├── DM_mid_semester_progress_report.pdf
│   └── DM_Final_Report.pdf
│
├── results/
│   ├── outputs/
│   │   ├── arima_results.csv
│   │   ├── hw_results.csv
│   │   ├── baseline_results.csv
│   │   ├── rf_tuned_results.csv
│   │   ├── xgb_tuned_results.csv
│   │   └── Comparison/model_comparison.csv
│   └── figures/
│       ├── EDA/                    ← 11 exploratory analysis charts
│       ├── Decomposition/          ← STL decomposition, ACF/PACF plots
│       ├── Modeling/               ← Forecasts, residuals, feature importance
│       ├── Comparison/             ← Actual vs. Predicted per crop-city pair
│       └── Clustering/             ← K-Means, hierarchical, PCA, anomaly plots
│
├── data/
│   ├── Dataset                     ← Kaggle source reference
│   └── README.txt                  ← Detailed data documentation
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [Kaggle — Crop Prices Dataset of Pakistan](https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan) |
| **Files** | 53 CSV files |
| **Records** | ~7.99 million rows |
| **Time Span** | 2008 – 2024 |
| **Cities** | 138 |
| **Crops** | 76 varieties |
| **Schema** | `City`, `Date`, `Crop`, `Price (PKR)` |

> ⚠️ The raw dataset is **not committed** to this repository due to size. Follow the [setup instructions](#-how-to-run) to download it from Kaggle.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | 1.24.4 | Numerical computing |
| `pandas` | 1.5.3 | Data manipulation |
| `matplotlib` | 3.7.1 | Visualization |
| `seaborn` | latest | Statistical plots |
| `scikit-learn` | 1.3.0 | ML models, PCA, preprocessing |
| `xgboost` | latest | Gradient boosted trees |
| `statsmodels` | 0.14.0 | ARIMA, Holt-Winters, ADF test |
| `scipy` | 1.10.1 | Statistical analysis |
| `psutil` | latest | Memory monitoring |

---

## 🔬 Methodology

### Phase 1 — Data Preparation & EDA

- Loaded and validated all 53 CSV files against the required schema
- Removed zero/negative price entries; applied **Winsorization** (IQR-based) per `(Crop, City)` group — no rows dropped, continuity preserved
- Extracted calendar features: Year, Month, Quarter, Day-of-week
- Filtered out series with fewer than 100 observations
- Conducted **11 EDA visualizations**: price distributions, yearly trends, monthly seasonality, crop volatility, cross-crop correlations

<p align="center">
  <img src="results/figures/EDA/04_overall_price_trend.png" width="49%" alt="Overall Price Trend"/>
  <img src="results/figures/EDA/05_monthly_seasonality.png" width="49%" alt="Monthly Seasonality"/>
</p>

<p align="center">
  <img src="results/figures/EDA/10_crop_volatility.png" width="49%" alt="Crop Volatility"/>
  <img src="results/figures/EDA/11_correlation_heatmap.png" width="49%" alt="Correlation Heatmap"/>
</p>

- **STL seasonal decomposition** + **ADF stationarity tests** per top `(Crop, City)` pair

<p align="center">
  <img src="results/figures/Decomposition/rolling_statistics_top_crops.png" width="80%" alt="Rolling Statistics"/>
</p>

---

### Phase 2 — Forecasting & Model Evaluation

**Feature Engineering:**
- Lag features (lags 1, 2, 3, 6, 12) capturing autocorrelation structure
- Rolling statistics: 7-day and 30-day mean/std
- Calendar encodings: month, quarter, year, day-of-week

**Model Hierarchy (complexity-ordered):**

| Tier | Models |
|------|--------|
| Baseline | Naive (last value), Seasonal Naive |
| Statistical | ARIMA, Holt-Winters Exponential Smoothing |
| Machine Learning | Linear Regression, Random Forest, XGBoost |

- **Recursive multi-horizon forecasting**: 1-month, 3-month, and 6-month horizons
- **Hyperparameter tuning** via grid search on validation set (strictly chronological — no data leakage)
- **Global vs. Local XGBoost** comparison: scalability vs. per-crop accuracy trade-off
- **Evaluation metrics**: MAE, RMSE, MAPE

<p align="center">
  <img src="results/figures/Modeling/actual_vs_predicted.png" width="80%" alt="Actual vs Predicted"/>
</p>

<p align="center">
  <img src="results/figures/Modeling/rmse_heatmap.png" width="49%" alt="RMSE Heatmap"/>
  <img src="results/figures/Modeling/forecast_horizons.png" width="49%" alt="Forecast Horizons"/>
</p>

<p align="center">
  <img src="results/figures/Modeling/feature_importance.png" width="49%" alt="Feature Importance"/>
  <img src="results/figures/Modeling/residuals.png" width="49%" alt="Residuals"/>
</p>

---

### Phase 3 — Clustering & Anomaly Detection

**Feature Engineering per `(Crop, City)` pair:**
- Mean price, standard deviation, OLS trend slope
- Coefficient of variation, seasonality spread, rolling volatility

**Clustering:**
- **K-Means**: Elbow method + Silhouette analysis for optimal K selection
- **Hierarchical/Agglomerative**: Dendrogram-based comparison
- **PCA 2D projection** for cluster visualization

**Anomaly Detection (three-method ensemble):**
- Z-Score · Rolling Deviation · IQR-based flagging

**Integration Analysis:**
- Cluster type vs. model RMSE correlation
- Anomaly rate vs. prediction error relationship

<p align="center">
  <img src="results/figures/Clustering/pca_scatter.png" width="49%" alt="PCA Cluster Scatter"/>
  <img src="results/figures/Clustering/cluster_profiles.png" width="49%" alt="Cluster Profiles"/>
</p>

<p align="center">
  <img src="results/figures/Clustering/anomaly_timeseries.png" width="49%" alt="Anomaly Time Series"/>
  <img src="results/figures/Clustering/dendrogram.png" width="49%" alt="Hierarchical Dendrogram"/>
</p>

---

## 📈 Results & Key Findings

### Forecasting Performance

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Naive Baseline | — | — | — |
| Seasonal Naive | — | — | — |
| ARIMA | — | — | — |
| Holt-Winters | — | — | — |
| Linear Regression | — | — | — |
| Random Forest (tuned) | — | — | — |
| **XGBoost (tuned)** | **—** | **—** | **—** |

> 📝 **Replace `—` with actual values from `results/outputs/Comparison/model_comparison.csv`**

### Clustering Summary

| Metric | Value |
|--------|-------|
| Optimal K (K-Means) | — |
| Silhouette Score | — |
| Anomaly Rate (Z-Score) | — |

### Key Insights

- **Seasonality detected** with dominant lag ≈ 12 months across major crop-city pairs (confirmed via ACF/PACF)
- **Significant inflationary trend** observed post-2019, with price acceleration across high-volatility crops
- **High-volatility cluster** identified: perishables (tomato, green chilli, onion) exhibit the lowest forecastability
- **Global XGBoost** model offers competitive RMSE with significantly reduced training time vs. per-crop local models
- **Winsorization strategy** proved more effective than outlier removal for preserving temporal continuity in forecasting

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/whozahm3d/Time-Series-Data-Analysis-and-Trend-Discovery-in-Pakistan-Crop-Prices.git
cd Time-Series-Data-Analysis-and-Trend-Discovery-in-Pakistan-Crop-Prices
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels scipy psutil
```

### 3. Download the Dataset

1. Go to [Kaggle — Crop Prices Dataset of Pakistan](https://www.kaggle.com/datasets/humairarana/crop-prices-dataset-of-pakistan)
2. Download and extract all **53 CSV files** into a single folder on your machine

### 4. Configure Paths

Open `notebooks/DM_Project_Final_Deliverable.ipynb` and update **Cell 4** (Configuration):

```python
DATA_DIR    = r"path\to\your\csv_folder"
MERGED_FILE = r"path\to\save\merged_output.csv"   # optional
```

### 5. Run the Notebook

```
Kernel → Restart & Run All
```

> ⚠️ **Important:** Cells must be executed in order. Each phase depends on variables set in the previous one. Do not skip sections.

All output figures and CSV results are saved automatically to `results/`.

---

## 📄 Reports

| Report | Description |
|--------|-------------|
| [Project Proposal](reports/Data%20Mining%20Project%20Proposal.pdf) | Initial scope, objectives, and methodology plan |
| [Mid-Semester Progress Report](reports/DM_mid_semester_progress_report.pdf) | Phase 1 findings and preliminary EDA results |
| [Final Report](reports/DM_Final_Report.pdf) | Complete methodology, results, and conclusions |

---

## 🧠 Engineering Notes

- **Memory-conscious design**: `dtype` optimization and `gc.collect()` calls throughout to handle large merged datasets (~7.99M rows)
- **No data leakage**: all train/val/test splits are strictly chronological — no shuffling at any stage
- **Outlier strategy**: Winsorization (capping) preferred over removal to preserve time-series continuity
- **Figure export**: all plots saved at 100 DPI via a unified `save_and_show()` helper; increase `dpi` parameter for high-resolution export

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Ali Ahmad | 23L-2619 |
| Taha Nawaz | 23L-2644 |
| Shahzeb Imran | 23L-2506 |

**FAST NUCES, Lahore — BS Data Science, Semester 6**
**Course: Data Mining (Spring 2026)**

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
