# FYP Dengue Risk Prediction

Hybrid dengue risk modelling for Sri Lanka: time-series forecasting, outbreak classification, and spatial hotspot analysis.

## Project Structure

```
FYP_Dengue_Risk_Prediction/
├── data/
│   ├── raw/                       # Original inputs (do not edit)
│   │   ├── sri_lanka_dengue_data.csv
│   │   └── Weather/
│   ├── processed/                 # Clean weekly tables
│   └── features/                  # Engineered feature sets
├── src/                           # Source code
│   ├── config.py                  # Paths and constants
│   ├── utils.py                   # Shared helpers
│   ├── preprocessing/             # Data cleaning & feature engineering
│   ├── module1_forecasting/       # SARIMA + residual learner
│   ├── module2_classification/    # Outbreak classifier (Member 2)
│   ├── module3_spatial/           # KDE + spatial compensation
│   └── dashboard/                 # Final integration UI
├── notebooks/                     # Exploratory Jupyter notebooks
├── models/                        # Saved trained models
├── outputs/                       # Figures, metrics, reports
├── docs/                          # Project documentation
├── requirements.txt
├── run_all.py                     # Main pipeline entry point
└── README.md
```

> **Note:** Your workspace folder may still be named `Data Preprocessing`. You can rename it to `FYP_Dengue_Risk_Prediction` when convenient — all paths are resolved relative to the project root.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Use `python -m pip` instead of `pip` if your pip launcher points to an old Python install.

## Run Preprocessing

```powershell
python run_all.py --step preprocess
```

Or run the full pipeline (currently preprocessing only):

```powershell
python run_all.py
```

## Outputs

| Path | Description |
|------|-------------|
| `data/processed/weather_weekly_all_districts.csv` | Weekly weather, all districts |
| `data/processed/dengue_weekly_clean.csv` | Clean weekly dengue cases |
| `data/processed/merged_weekly_dengue_weather.csv` | Dengue + weather (no lags) |
| `data/features/engineered_features.csv` | Full feature set for modelling |
| `data/features/module2_features.csv` | Subset for outbreak classification |
| `outputs/reports/preprocessing_report.txt` | Validation and missingness report |

## Module 2 (Classification) — Starting Point

After preprocessing, begin Module 2 in `src/module2_classification/`:

1. `outbreak_label.py` — define outbreak labels from case counts
2. `stage1_base_classifier.py` — base classifier
3. `stage2_compensation.py` — compensation model
4. `train.py` / `evaluate.py` — training and evaluation

Input data: `data/features/module2_features.csv`

## Adding New Modules

Each module lives under `src/`. Import paths and file locations from `src/config.py` so everything stays consistent.
