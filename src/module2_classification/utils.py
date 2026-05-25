"""Module 2 paths, constants, and shared helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MERGED_WEEKLY_PATH, MODULE2_FEATURE_COLUMNS
from src.utils import ensure_dir, load_csv, save_csv

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
MODELS_DIR = MODULE_DIR / "models"
RESULTS_DIR = MODULE_DIR / "results"

MERGED_WITH_LABEL_PATH = DATA_DIR / "merged_with_outbreak_label.csv"
MODULE2_FEATURES_PATH = DATA_DIR / "module2_features.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "feature_sets.json"

BASE_MODEL_PATH = MODELS_DIR / "base_classifier.pkl"

FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_PATH = RESULTS_DIR / "metrics.txt"

ID_COLUMNS = ("district", "week_start_date")
TARGET_COLUMN = "outbreak_label"
CASES_COLUMN = "cases"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1

# District-wise percentile threshold: cases >= threshold -> outbreak (1).
OUTBREAK_PERCENTILE = 0.75
TEST_WEEK_FRACTION = 0.20
RANDOM_STATE = 42

# Stage 1: epidemiological + temporal only (no raw climate / anomalies).
STAGE1_FEATURES = [
    "rolling_mean_2",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "rolling_std_4",
    "rate_of_change",
    "year",
    "sin_week",
    "cos_week",
    "is_ne_monsoon",
]

# Stage 2 adds climate residuals on top of Stage 1 base probability.
STAGE2_EXTRA_FEATURES = [
    "rainfall_lag_6",
    "rainfall_lag_8",
    "rainfall_anomaly",
    "temperature_anomaly",
    "humidity_anomaly",
    "tmean",
    "humidity",
    "temp_lag_1",
]


def ensure_module_dirs() -> None:
    """Create module data, model, and results directories."""
    for path in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        ensure_dir(path)


def load_merged_source() -> pd.DataFrame:
    """Load the preprocessed merged weekly dataset from the project root."""
    return load_csv(MERGED_WEEKLY_PATH, parse_dates=["week_start_date"])


def add_outbreak_label(
    df: pd.DataFrame,
    percentile: float = OUTBREAK_PERCENTILE,
) -> pd.DataFrame:
    """Label weeks at or above a district-wise case percentile as outbreaks."""
    out = df.copy()
    thresholds = out.groupby("district")[CASES_COLUMN].transform(
        lambda s: s.quantile(percentile)
    )
    out[TARGET_COLUMN] = (out[CASES_COLUMN] >= thresholds).astype(int)
    return out


def outbreak_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Return normal/outbreak counts keyed by outbreak_label (0/1)."""
    counts = df[TARGET_COLUMN].value_counts()
    return counts.reindex([NEGATIVE_LABEL, POSITIVE_LABEL], fill_value=0)


def select_module2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the Module 2 feature subset plus identifiers and target."""
    label_cols = [TARGET_COLUMN] if TARGET_COLUMN in df.columns else []
    keep = list(dict.fromkeys(list(ID_COLUMNS) + list(MODULE2_FEATURE_COLUMNS) + label_cols))
    available = [c for c in keep if c in df.columns]
    return df[available].copy()


def save_feature_sets(stage1: list[str], stage2: list[str]) -> None:
    """Persist selected feature lists for downstream scripts."""
    ensure_module_dirs()
    payload = {"stage1_features": stage1, "stage2_features": stage2}
    FEATURE_SETS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_feature_sets() -> dict[str, list[str]]:
    """Load Stage 1 / Stage 2 feature lists."""
    if not FEATURE_SETS_PATH.exists():
        save_feature_sets(STAGE1_FEATURES, STAGE1_FEATURES + STAGE2_EXTRA_FEATURES)
    return json.loads(FEATURE_SETS_PATH.read_text(encoding="utf-8"))
