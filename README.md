# Predicting-Daily-Respiratory-Hospital-Admissions-from-Air-Pollution
# ML Final Project: Predicting Daily Respiratory Admissions from Air Pollution

This project tests whether daily air pollution + weather + mobility/policy signals can predict respiratory hospital admissions. Using a leakage-aware chronological split and a unified preprocessing pipeline, it benchmarks Ridge Regression, RBF-SVR, and a PyTorch MLP.

## Problem Statement
Goal: predict `respiratory_admissions` (daily count) from environmental + contextual features (supervised regression).

## Dataset
- Size: 3,000 rows × 26 columns  
- Target: `respiratory_admissions`  
- Key feature groups (examples):  
  - Air quality: `AQI`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`  
  - Weather: `temperature`, `humidity`, `wind_speed`, `precipitation`  
  - Mobility/policy: `mobility_index`, `school_closures`, `mask_usage_rate`, `lockdown_status`, etc.  
  - Categorical: `region`  
  - Time: `date`

Note: If the dataset is course-provided or restricted, do not upload it to GitHub. Keep it local under `data/` and ignore it via `.gitignore`.

## Methodology
### 1) Time-aware split (reduce leakage)
Data is sorted by `date`, then split chronologically:
- 70% train
- 15% validation
- 15% test

### 2) Preprocessing (single consistent pipeline)
- Numeric features: StandardScaler → PCA (95% explained variance)
- Categorical (`region`): OneHotEncoder
- Implemented with sklearn `ColumnTransformer` + `Pipeline`

### 3) Models
- Ridge Regression (tuned alpha)
- SVR (RBF kernel) (tuned `C` and `gamma`)
- PyTorch MLP
  - Architecture: `input_dim → 26 → 13 → 1` with ReLU
  - Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)
  - Loss: MSE
  - ~80 epochs with train/val monitoring

### 4) Metrics
- RMSE
- R²

## Results (Test Set)
| Model | Test RMSE | Test R² |
|------|-----------:|--------:|
| Ridge | 3.188 | -0.012 |
| SVR (RBF) | 3.208 | -0.025 |
| PyTorch MLP | 3.443 | -0.180 |

Interpretation: negative R² suggests performance slightly worse than predicting the mean, indicating limited predictive signal from the available covariates under these models/features.

## Repository Structure (recommended)
.
├── notebooks/
│ └── ML_Final_Project.ipynb
├── data/ # keep local (do not commit)
│ └── air_quality_health_dataset.csv
├── outputs/ # optional: exported figures/tables
├── requirements.txt
├── .gitignore
└── README.md
### Suggested `.gitignore`
data/
outputs/
.venv/
pycache/
.ipynb_checkpoints/
*.pyc

---

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```
## Minimal requirements.txt

```bash
pandas
numpy
scikit-learn
matplotlib
torch
jupyter
```

## How to Run

Put the dataset here:

data/air_quality_health_dataset.csv

Launch Jupyter and run the notebook:
```bash
jupyter notebook
```


Open:

notebooks/ML_Final_Project.ipynb

## Reproducibility Notes

Chronological splitting is used to reduce time leakage.

All preprocessing is done inside sklearn pipelines to ensure the same transforms are applied consistently across train/val/test.

## Limitations

Commit-level: RMSE/R² alone may not capture operational utility (e.g., threshold-based alerts).

The dataset may contain weak signal relative to daily admission variability; more informative features (lagged admissions, seasonal terms, local events) may be needed.

PCA can improve stability but may reduce interpretability of individual features.

## Future Work

Add time-series features (lags/rolling means), seasonality, and calendar effects.

Compare with stronger baselines: Gradient Boosting / XGBoost / LightGBM.

Use time-series cross-validation and perform more systematic hyperparameter tuning.

Add interpretability (permutation importance / SHAP) on the best-performing model.