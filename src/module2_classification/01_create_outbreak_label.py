"""
Step 1: Create risk labels and module-specific feature tables.

Reads the root preprocessed merge (`data/processed/merged_weekly_dengue_weather.csv`),
assigns district-wise low / medium / high risk labels from case tertiles, and writes
module-local copies under `data/`.

Run:
    python -m src.module2_classification.01_create_outbreak_label
"""

from __future__ import annotations

from src.module2_classification.utils import (
    MERGED_WITH_LABEL_PATH,
    MODULE2_FEATURES_PATH,
    RISK_CLASSES,
    RISK_HIGH_PERCENTILE,
    RISK_LOW_PERCENTILE,
    TARGET_COLUMN,
    add_risk_label,
    ensure_module_dirs,
    load_merged_source,
    risk_label_distribution,
    select_module2_columns,
)
from src.utils import save_csv


def run_create_outbreak_label() -> None:
    ensure_module_dirs()

    merged = load_merged_source()
    labeled = add_risk_label(
        merged,
        low_percentile=RISK_LOW_PERCENTILE,
        high_percentile=RISK_HIGH_PERCENTILE,
    )
    features = select_module2_columns(labeled)

    save_csv(labeled, MERGED_WITH_LABEL_PATH)
    save_csv(features, MODULE2_FEATURES_PATH)

    counts = risk_label_distribution(labeled)
    total = len(labeled)
    print(f"Saved labeled merge: {MERGED_WITH_LABEL_PATH} ({total:,} rows)")
    print(f"Saved Module 2 features: {MODULE2_FEATURES_PATH} ({len(features):,} rows)")
    print(
        f"Risk labels (district tertiles p{RISK_LOW_PERCENTILE:.0%}/p{RISK_HIGH_PERCENTILE:.0%}):"
    )
    for label in RISK_CLASSES:
        n = int(counts[label])
        print(f"  {label:6s}: {n:,} ({n / total:.1%})")


def main() -> None:
    run_create_outbreak_label()


if __name__ == "__main__":
    main()
