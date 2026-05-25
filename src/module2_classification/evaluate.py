"""
Evaluate Module 2 classification models.

Reads Stage 1 outputs from `results/metrics.txt` and the saved model bundle.

Run:
    python -m src.module2_classification.evaluate
"""

from __future__ import annotations

import importlib
import pickle

from src.module2_classification.utils import (
    BASE_MODEL_PATH,
    METRICS_PATH,
    TEST_DATA_PATH,
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
    y_test = test_df[bundle["target"]]
    x_test_imputed = bundle["imputer"].transform(x_test)
    y_pred = bundle["model"].predict(x_test_imputed)

    print("Module 2 — Stage 1 evaluation (binary outbreak)")
    print(f"Model: {BASE_MODEL_PATH}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Predicted outbreaks: {int(y_pred.sum()):,} / {len(y_pred):,}")

    if METRICS_PATH.exists():
        print(f"\nDetailed metrics saved to:\n  {METRICS_PATH}")
        print("\n" + METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
