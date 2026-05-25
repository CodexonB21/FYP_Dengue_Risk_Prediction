"""
Evaluate Module 2 classification models.

Reads Stage 1 outputs from `results/metrics.txt` and the saved model bundle.

Run:
    python -m src.module2_classification.evaluate
"""

from __future__ import annotations

import importlib
import pickle

import pandas as pd

from src.module2_classification.utils import (
    BASE_MODEL_PATH,
    METRICS_PATH,
    RISK_CLASSES,
    TEST_DATA_PATH,
    encode_risk_label,
    ensure_module_dirs,
)
from src.utils import load_csv


def run_evaluation() -> None:
    ensure_module_dirs()

    if not BASE_MODEL_PATH.exists():
        stage1 = importlib.import_module("src.module2_classification.03_stage1_base_model")
        stage1.run_stage1_base_model()

    with BASE_MODEL_PATH.open("rb") as fh:
        bundle = pickle.load(fh)

    test_df = load_csv(TEST_DATA_PATH, parse_dates=["week_start_date"])
    features = bundle["features"]
    x_test = test_df[features]
    y_test = encode_risk_label(test_df[bundle["target"]])
    x_test_imputed = bundle["imputer"].transform(x_test)
    y_pred = bundle["model"].predict(x_test_imputed)

    pred_counts = pd.Series(y_pred).value_counts().reindex(range(len(RISK_CLASSES)), fill_value=0)

    print("Module 2 — Stage 1 evaluation (low / medium / high)")
    print(f"Model: {BASE_MODEL_PATH}")
    print(f"Test rows: {len(test_df):,}")
    print("Predicted class counts:")
    for idx, label in enumerate(RISK_CLASSES):
        print(f"  {label:6s}: {int(pred_counts[idx]):,}")

    if METRICS_PATH.exists():
        print(f"\nDetailed metrics saved to:\n  {METRICS_PATH}")
        print("\n" + METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
