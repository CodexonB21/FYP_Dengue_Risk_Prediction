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

MERGED_WITH_LABEL_PATH = DATA_DIR / "merged_with_risk_label.csv"
MODULE2_FEATURES_PATH = DATA_DIR / "module2_features.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "feature_sets.json"

BASE_MODEL_PATH = MODELS_DIR / "base_classifier.pkl"

FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_PATH = RESULTS_DIR / "metrics.txt"

ID_COLUMNS = ("district", "week_start_date")
TARGET_COLUMN = "risk_label"
CASES_COLUMN = "cases"

# District-wise tertiles on weekly cases: low | medium | high.
RISK_CLASSES = ("low", "medium", "high")
RISK_LOW_PERCENTILE = 1 / 3
RISK_HIGH_PERCENTILE = 2 / 3
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


def add_risk_label(
    df: pd.DataFrame,
    low_percentile: float = RISK_LOW_PERCENTILE,
    high_percentile: float = RISK_HIGH_PERCENTILE,
) -> pd.DataFrame:
    """Assign low / medium / high risk from district-wise case tertiles."""
    out = df.copy()

    def _assign_group(group: pd.DataFrame) -> pd.Series:
        q_low = group[CASES_COLUMN].quantile(low_percentile)
        q_high = group[CASES_COLUMN].quantile(high_percentile)
        labels = pd.Series("medium", index=group.index, dtype="object")
        labels[group[CASES_COLUMN] < q_low] = "low"
        labels[group[CASES_COLUMN] >= q_high] = "high"
        return labels

    out[TARGET_COLUMN] = out.groupby("district", group_keys=False).apply(_assign_group)
    return out


def encode_risk_label(labels: pd.Series) -> pd.Series:
    """Map risk labels to integers 0=low, 1=medium, 2=high."""
    mapping = {label: idx for idx, label in enumerate(RISK_CLASSES)}
    encoded = labels.map(mapping)
    if encoded.isna().any():
        unknown = sorted(labels[encoded.isna()].unique())
        raise ValueError(f"Unknown risk labels: {unknown}")
    return encoded.astype(int)


def risk_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Return class counts ordered low -> medium -> high."""
    counts = df[TARGET_COLUMN].value_counts()
    return counts.reindex(RISK_CLASSES, fill_value=0)


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
