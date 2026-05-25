"""
Step 1: Create binary outbreak labels and module-specific feature tables.

Reads the root preprocessed merge (`data/processed/merged_weekly_dengue_weather.csv`),
assigns district-wise outbreak labels (cases >= 75th percentile), and writes
module-local copies under `data/`.

Run:
    python -m src.module2_classification.01_create_outbreak_label
"""

from __future__ import annotations

from src.module2_classification.utils import (
    MERGED_WITH_LABEL_PATH,
    MODULE2_FEATURES_PATH,
    NEGATIVE_LABEL,
    OUTBREAK_PERCENTILE,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    add_outbreak_label,
    ensure_module_dirs,
    load_merged_source,
    outbreak_label_distribution,
    select_module2_columns,
)
from src.utils import save_csv


def run_create_outbreak_label() -> None:
    ensure_module_dirs()

    merged = load_merged_source()
    labeled = add_outbreak_label(merged, percentile=OUTBREAK_PERCENTILE)
    features = select_module2_columns(labeled)

    save_csv(labeled, MERGED_WITH_LABEL_PATH)
    save_csv(features, MODULE2_FEATURES_PATH)

    counts = outbreak_label_distribution(labeled)
    total = len(labeled)
    normal = int(counts[NEGATIVE_LABEL])
    outbreak = int(counts[POSITIVE_LABEL])

    print(f"Saved labeled merge: {MERGED_WITH_LABEL_PATH} ({total:,} rows)")
    print(f"Saved Module 2 features: {MODULE2_FEATURES_PATH} ({len(features):,} rows)")
    print(f"Outbreak threshold: district-wise {OUTBREAK_PERCENTILE:.0%} percentile")
    print(f"  Normal (0):  {normal:,} ({normal / total:.1%})")
    print(f"  Outbreak (1): {outbreak:,} ({outbreak / total:.1%})")


def main() -> None:
    run_create_outbreak_label()


if __name__ == "__main__":
    main()
