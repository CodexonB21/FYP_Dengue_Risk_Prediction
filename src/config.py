"""Project paths, constants, and shared parameters."""

from pathlib import Path

# Project root (FYP_Dengue_Risk_Prediction/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

RAW_WEATHER_DIR = RAW_DIR / "Weather"
RAW_DENGUE_PATH = RAW_DIR / "sri_lanka_dengue_data.csv"

WEATHER_WEEKLY_PATH = PROCESSED_DIR / "weather_weekly_all_districts.csv"
DENGUE_WEEKLY_PATH = PROCESSED_DIR / "dengue_weekly_clean.csv"
MERGED_WEEKLY_PATH = PROCESSED_DIR / "merged_weekly_dengue_weather.csv"

ENGINEERED_FEATURES_PATH = FEATURES_DIR / "engineered_features.csv"
MODULE2_FEATURES_PATH = FEATURES_DIR / "module2_features.csv"

# --- Output paths ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.txt"

# --- Model paths ---
MODELS_DIR = PROJECT_ROOT / "models"
MODULE2_BASE_MODEL_PATH = MODELS_DIR / "module2_base_model.pkl"
MODULE2_COMP_MODEL_PATH = MODELS_DIR / "module2_comp_model.pkl"

# --- Preprocessing constants ---
DATE_COLUMN_CANDIDATES = ("date", "time", "datetime", "timestamp")

WEATHER_COLUMN_PATTERNS: dict[str, list[str]] = {
    "tmax": [r"temperature_2m_max", r"temperature.*max", r"^tmax$", r"temp.*max"],
    "tmin": [r"temperature_2m_min", r"temperature.*min", r"^tmin$", r"temp.*min"],
    "app_tmax": [r"apparent_temperature_max", r"apparent.*max", r"app_temp.*max"],
    "app_tmin": [r"apparent_temperature_min", r"apparent.*min", r"app_temp.*min"],
    "rainfall": [r"rain_sum", r"rainfall", r"^rain$", r"rain.*sum"],
    "precipitation": [r"precipitation_sum", r"precipitation", r"^precip"],
    "wind_speed": [r"wind_speed", r"windspeed", r"wind.*speed"],
    "wind_gust": [r"wind_gust", r"windgust", r"wind.*gust"],
    "humidity": [
        r"relative_humidity_2m_mean",
        r"relative_humidity.*mean",
        r"relative_humidity",
        r"^humidity$",
        r"humidity.*mean",
    ],
}

DENGUE_DISTRICT_CANDIDATES = ("district", "location", "region", "area", "name")
DENGUE_DATE_CANDIDATES = (
    "week_start_date",
    "week_start",
    "week",
    "date",
    "start_date",
    "reporting_date",
)
DENGUE_CASES_CANDIDATES = (
    "number_of_cases",
    "cases",
    "dengue_cases",
    "case_count",
    "count",
    "no_of_cases",
)

WEATHER_MEAN_COLS = (
    "tmax",
    "tmin",
    "app_tmax",
    "app_tmin",
    "humidity",
    "wind_speed",
    "wind_gust",
)
WEATHER_SUM_COLS = ("rainfall", "precipitation")

# Columns exported for Module 2 (outbreak classification)
MODULE2_FEATURE_COLUMNS = [
    "district",
    "week_start_date",
    "cases",
    "year",
    "month",
    "week_of_year",
    "sin_week",
    "cos_week",
    "tmax",
    "tmin",
    "tmean",
    "humidity",
    "rainfall",
    "precipitation",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",
    "rainfall_lag_5",
    "rainfall_lag_6",
    "rainfall_lag_8",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "temp_lag_4",
    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "humidity_lag_4",
    "rolling_mean_2",
    "rolling_mean_4",
    "rolling_std_4",
    "rate_of_change",
    "percentage_change",
    "rainfall_anomaly",
    "temperature_anomaly",
    "humidity_anomaly",
    "is_sw_monsoon",
    "is_ne_monsoon",
    "is_inter_monsoon",
    "monsoon_season",
]
