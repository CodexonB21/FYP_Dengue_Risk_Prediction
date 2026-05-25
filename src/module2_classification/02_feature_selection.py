"""
Step 2: Feature selection and temporal train/test split.

Scores predictors, confirms Stage 1 / Stage 2 feature sets, and writes
`train_data.csv` and `test_data.csv` under the module `data/` folder.

Run:
    python -m src.module2_classification.02_feature_selection
"""

from __future__ import annotations

import importlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from src.module2_classification.utils import (
    ID_COLUMNS,
    MERGED_WITH_LABEL_PATH,
    RISK_CLASSES,
    STAGE1_FEATURES,
    STAGE2_EXTRA_FEATURES,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_WEEK_FRACTION,
    TRAIN_DATA_PATH,
    encode_risk_label,
    ensure_module_dirs,
    risk_label_distribution,
    save_feature_sets,
)
from src.utils import load_csv, save_csv


def _candidate_features(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLUMNS) | {TARGET_COLUMN, "cases", "monsoon_season"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric predictors by correlation, MI, and RF importance."""
    features = _candidate_features(df)
    analysis = df[features + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    x = analysis[features]
    y = encode_risk_label(analysis[TARGET_COLUMN])

    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(imputer.fit_transform(x), columns=features)

    corr = x_imputed.corrwith(y).abs()
    mi = pd.Series(
        mutual_info_classif(x_imputed, y, random_state=42),
        index=features,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(x_imputed, y)
    rf_imp = pd.Series(rf.feature_importances_, index=features)

    scores = pd.DataFrame(
        {
            "feature": features,
            "correlation": corr.values,
            "mutual_info": mi.values,
            "rf_importance": rf_imp.values,
        }
    )
    for col in ("correlation", "mutual_info", "rf_importance"):
        scores[f"{col}_rank"] = scores[col].rank(ascending=False)
    scores["avg_rank"] = scores[
        ["correlation_rank", "mutual_info_rank", "rf_importance_rank"]
    ].mean(axis=1)
    return scores.sort_values("avg_rank")


def temporal_train_test_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar week so the test set is the most recent period."""
    work = df.copy()
    work["week_start_date"] = pd.to_datetime(work["week_start_date"])
    cutoff = work["week_start_date"].quantile(1 - test_fraction)
    train = work[work["week_start_date"] <= cutoff].copy()
    test = work[work["week_start_date"] > cutoff].copy()
    return train, test


def run_feature_selection() -> None:
    ensure_module_dirs()

    if not MERGED_WITH_LABEL_PATH.exists():
        step1 = importlib.import_module("src.module2_classification.01_create_outbreak_label")
        step1.run_create_outbreak_label()

    df = load_csv(MERGED_WITH_LABEL_PATH, parse_dates=["week_start_date"])
    scores = score_features(df)

    stage1 = [f for f in STAGE1_FEATURES if f in df.columns]
    stage2 = stage1 + [f for f in STAGE2_EXTRA_FEATURES if f in df.columns]
    save_feature_sets(stage1, stage2)

    train, test = temporal_train_test_split(df, TEST_WEEK_FRACTION)
    save_csv(train, TRAIN_DATA_PATH)
    save_csv(test, TEST_DATA_PATH)

    print(f"Stage 1 features ({len(stage1)}): {stage1}")
    print(f"Stage 2 features ({len(stage2)}): {stage2}")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print("Label distribution (all data):")
    for label in RISK_CLASSES:
        n = int(risk_label_distribution(df)[label])
        print(f"  {label:6s}: {n:,} ({n / len(df):.1%})")
    print("\nTop 10 predictors by average rank:")
    print(scores.head(10).to_string(index=False))


def main() -> None:
    run_feature_selection()


if __name__ == "__main__":
    main()
