"""Define outbreak labels from weekly case counts (district-wise thresholds)."""

from src.config import MODULE2_FEATURES_PATH
from src.utils import load_csv

# TODO: implement label logic (e.g. percentile-based or absolute threshold per district)


def load_module2_data():
    return load_csv(MODULE2_FEATURES_PATH)
