"""
Main entry point for the FYP Dengue Risk Prediction pipeline.

Usage:
    python run_all.py                  # run all enabled steps
    python run_all.py --step preprocess
    python run_all.py --step module2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_preprocess() -> None:
    from src.preprocessing.preprocess import run_preprocessing

    run_preprocessing()


def run_module1() -> None:
    print("Module 1 (forecasting) is not yet implemented.")
    print("See src/module1_forecasting/ for planned scripts.")


def run_module2() -> None:
    import importlib

    step1 = importlib.import_module("src.module2_classification.01_create_outbreak_label")
    step2 = importlib.import_module("src.module2_classification.02_feature_selection")
    step3 = importlib.import_module("src.module2_classification.03_stage1_base_model")

    print("Module 2 — Step 1: outbreak labels")
    step1.run_create_outbreak_label()
    print("\nModule 2 — Step 2: feature selection + split")
    step2.run_feature_selection()
    print("\nModule 2 — Step 3: Stage 1 base classifier")
    step3.run_stage1_base_model()


def run_module3() -> None:
    print("Module 3 (spatial) is not yet implemented.")
    print("See src/module3_spatial/ for planned scripts.")


def run_dashboard() -> None:
    print("Dashboard is not yet implemented.")
    print("See src/dashboard/app.py when ready.")


STEPS = {
    "preprocess": run_preprocess,
    "module1": run_module1,
    "module2": run_module2,
    "module3": run_module3,
    "dashboard": run_dashboard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="FYP Dengue Risk Prediction pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        run_preprocess()
        # Uncomment as modules are implemented:
        # run_module1()
        # run_module2()
        # run_module3()
        # run_dashboard()
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
