"""
Weekly weather + dengue preprocessing pipeline.

Reads raw data from data/raw/, writes processed tables to data/processed/,
engineered features to data/features/, and report to outputs/reports/.
"""

from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DENGUE_CASES_CANDIDATES,
    DENGUE_DATE_CANDIDATES,
    DENGUE_DISTRICT_CANDIDATES,
    DENGUE_WEEKLY_PATH,
    ENGINEERED_FEATURES_PATH,
    FEATURES_DIR,
    MERGED_WEEKLY_PATH,
    MODULE2_FEATURE_COLUMNS,
    MODULE2_FEATURES_PATH,
    PREPROCESSING_REPORT_PATH,
    PROCESSED_DIR,
    RAW_DENGUE_PATH,
    RAW_WEATHER_DIR,
    WEATHER_COLUMN_PATTERNS,
    WEATHER_MEAN_COLS,
    WEATHER_SUM_COLS,
    WEATHER_WEEKLY_PATH,
    DATE_COLUMN_CANDIDATES,
)
from src.utils import ensure_dir, save_csv

REPORT: list[str] = []


def log(message: str) -> None:
    REPORT.append(message)
    print(message)


def normalize_column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(
    columns: list[str],
    candidates: tuple[str, ...] | None = None,
    patterns: list[str] | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    normalized = {normalize_column_key(c): c for c in columns}

    if candidates:
        for cand in candidates:
            key = normalize_column_key(cand)
            if key in normalized and normalized[key] not in exclude:
                return normalized[key]

    if patterns:
        for col in columns:
            if col in exclude:
                continue
            col_key = normalize_column_key(col)
            for pattern in patterns:
                if re.search(pattern, col_key, flags=re.IGNORECASE):
                    return col

    if candidates:
        for cand in candidates:
            key = normalize_column_key(cand)
            for col in columns:
                if col in exclude:
                    continue
                if key in normalize_column_key(col):
                    return col

    return None


def normalize_district_name(name: str) -> str:
    if pd.isna(name):
        return name
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    return cleaned.title()


def district_match_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_district_name(name).lower())


def parse_district_from_filename(filepath: Path) -> str:
    stem = filepath.stem
    match = re.match(r"open-meteo-[\d.NE]+m\s+(.+)$", stem, flags=re.IGNORECASE)
    if match:
        return normalize_district_name(match.group(1))
    return normalize_district_name(stem.rsplit(" ", 1)[-1])


def to_monday(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.dayofweek, unit="D")


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def load_weather_csv(filepath: Path) -> pd.DataFrame:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    header_idx = None
    for idx, line in enumerate(lines):
        first_cell = line.split(",")[0].strip().lower()
        if first_cell in DATE_COLUMN_CANDIDATES:
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"Could not locate date/time header row in {filepath.name}")

    df = pd.read_csv(StringIO("".join(lines[header_idx:])))
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)]
    df = df.dropna(axis=1, how="all")
    return df


def standardize_weather_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    date_col = find_column(list(df.columns), candidates=DATE_COLUMN_CANDIDATES)
    if date_col is None:
        raise ValueError("No date/time column found in weather data.")

    rename_map: dict[str, str] = {date_col: "date"}
    used_sources: set[str] = {date_col}

    for canonical, patterns in WEATHER_COLUMN_PATTERNS.items():
        match = find_column(list(df.columns), patterns=patterns, exclude=used_sources)
        if match:
            rename_map[match] = canonical
            used_sources.add(match)

    out = df.rename(columns=rename_map)
    keep_cols = ["date"] + [c for c in WEATHER_COLUMN_PATTERNS if c in out.columns]
    out = out[keep_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    for col in keep_cols:
        if col == "date":
            continue
        out[col] = parse_numeric_series(out[col])

    return out, rename_map


def aggregate_weather_to_weekly(df: pd.DataFrame, district: str) -> pd.DataFrame:
    daily = df.copy()
    daily["date"] = daily["date"].dt.normalize()

    daily_mean_cols = [c for c in WEATHER_MEAN_COLS if c in daily.columns]
    daily_sum_cols = [c for c in WEATHER_SUM_COLS if c in daily.columns]
    daily_agg: dict[str, str] = {c: "mean" for c in daily_mean_cols}
    daily_agg.update({c: "sum" for c in daily_sum_cols})

    if daily_agg:
        daily = daily.groupby("date", as_index=False).agg(daily_agg).sort_values("date")

    daily["week_start_date"] = to_monday(daily["date"])

    weekly_agg: dict[str, str] = {}
    for col in daily_mean_cols:
        weekly_agg[col] = "mean"
    for col in daily_sum_cols:
        weekly_agg[col] = "sum"

    if not weekly_agg:
        weekly = daily[["week_start_date"]].drop_duplicates().copy()
    else:
        weekly = (
            daily.groupby("week_start_date", as_index=False)
            .agg(weekly_agg)
            .sort_values("week_start_date")
        )

    weekly["district"] = district
    weekly["week_start_date"] = weekly["week_start_date"].dt.strftime("%Y-%m-%d")
    return weekly


def clean_weather_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    n_fixed = 0
    for col in ("rainfall", "precipitation"):
        if col in out.columns:
            mask = out[col] < 0
            n_fixed += int(mask.sum())
            out.loc[mask, col] = 0.0
    return out, n_fixed


def process_all_weather() -> tuple[pd.DataFrame, dict]:
    if not RAW_WEATHER_DIR.exists():
        raise FileNotFoundError(f"Weather directory not found: {RAW_WEATHER_DIR}")

    csv_files = sorted(RAW_WEATHER_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No weather CSV files in {RAW_WEATHER_DIR}")

    frames: list[pd.DataFrame] = []
    meta: dict = {"files": [], "negative_rainfall_fixed": 0}

    for path in csv_files:
        district = parse_district_from_filename(path)
        raw = load_weather_csv(path)
        standardized, mapping = standardize_weather_columns(raw)
        weekly = aggregate_weather_to_weekly(standardized, district)
        weekly, n_neg = clean_weather_values(weekly)
        meta["negative_rainfall_fixed"] += n_neg
        frames.append(weekly)
        meta["files"].append(
            {
                "file": path.name,
                "district": district,
                "raw_rows": len(raw),
                "weekly_rows": len(weekly),
                "column_mapping": mapping,
            }
        )
        log(f"  Weather: {district} - {len(raw)} daily rows -> {len(weekly)} weekly rows")

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values(["district", "week_start_date"]).reset_index(drop=True)
    meta["total_rows"] = len(weather)
    meta["districts"] = sorted(weather["district"].unique())
    return weather, meta


def build_district_reference(weather_districts: list[str]) -> dict[str, str]:
    return {district_match_key(d): d for d in weather_districts}


def align_district_to_weather(name: str, reference: dict[str, str]) -> str:
    key = district_match_key(name)
    if key in reference:
        return reference[key]
    normalized = normalize_district_name(name)
    log(f"  WARNING: dengue district '{name}' not matched in weather data; keeping '{normalized}'")
    return normalized


def detect_dengue_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = list(df.columns)
    district_col = find_column(cols, candidates=DENGUE_DISTRICT_CANDIDATES)
    date_col = find_column(cols, candidates=DENGUE_DATE_CANDIDATES)
    cases_col = find_column(cols, candidates=DENGUE_CASES_CANDIDATES)

    missing = [
        label
        for label, col in (
            ("district", district_col),
            ("date", date_col),
            ("cases", cases_col),
        )
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not detect dengue column(s): {missing}. Found columns: {cols}"
        )

    return district_col, date_col, cases_col


def process_dengue(weather_districts: list[str]) -> tuple[pd.DataFrame, dict]:
    if not RAW_DENGUE_PATH.exists():
        raise FileNotFoundError(f"Dengue file not found: {RAW_DENGUE_PATH}")

    raw = pd.read_csv(RAW_DENGUE_PATH)
    district_col, date_col, cases_col = detect_dengue_columns(raw)
    reference = build_district_reference(weather_districts)

    log(f"  Dengue columns detected: district='{district_col}', date='{date_col}', cases='{cases_col}'")

    df = raw[[district_col, date_col, cases_col]].copy()
    df.columns = ["district", "week_start_date", "cases"]

    df["district"] = df["district"].apply(lambda x: align_district_to_weather(x, reference))
    df["week_start_date"] = to_monday(df["week_start_date"])
    df["cases"] = parse_numeric_series(df["cases"])

    negative_cases = int((df["cases"] < 0).sum())
    if negative_cases:
        log(f"  WARNING: {negative_cases} negative case values set to NaN")
        df.loc[df["cases"] < 0, "cases"] = np.nan

    before_agg = len(df)
    df = (
        df.groupby(["district", "week_start_date"], as_index=False)["cases"]
        .sum(min_count=1)
        .sort_values(["district", "week_start_date"])
    )
    duplicates_dropped = before_agg - len(df)
    df["week_start_date"] = df["week_start_date"].dt.strftime("%Y-%m-%d")

    meta = {
        "raw_rows": len(raw),
        "weekly_rows": len(df),
        "duplicates_aggregated": duplicates_dropped,
        "negative_cases_set_nan": negative_cases,
        "districts": sorted(df["district"].unique()),
        "date_range": (df["week_start_date"].min(), df["week_start_date"].max()),
    }
    return df, meta


def missingness_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    rows = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        miss = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing": miss,
                "missing_pct": round(100 * miss / n, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def impute_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather_cols = [c for c in WEATHER_MEAN_COLS + WEATHER_SUM_COLS if c in df.columns]
    before = missingness_table(df, weather_cols)

    out = df.sort_values(["district", "week_start_date"]).copy()
    for col in weather_cols:
        out[col] = out.groupby("district", group_keys=False)[col].transform(
            lambda s: s.ffill().bfill()
        )
        district_median = out.groupby("district")[col].transform("median")
        out[col] = out[col].fillna(district_median)

    after = missingness_table(out, weather_cols)
    return out, before, after


def merge_datasets(dengue: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    merged = dengue.merge(
        weather,
        on=["district", "week_start_date"],
        how="left",
        suffixes=("", "_weather"),
        validate="m:1",
    )
    match_rate = 100.0 * merged["tmax"].notna().mean() if "tmax" in merged.columns else 0.0
    if "tmax" not in merged.columns:
        weather_cols = [c for c in WEATHER_MEAN_COLS + WEATHER_SUM_COLS if c in merged.columns]
        if weather_cols:
            match_rate = 100.0 * merged[weather_cols[0]].notna().mean()
    return merged, match_rate


def add_monsoon_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month = out["month"]

    out["is_sw_monsoon"] = month.isin([5, 6, 7, 8, 9]).astype(int)
    out["is_ne_monsoon"] = month.isin([12, 1, 2]).astype(int)
    out["is_inter_monsoon"] = month.isin([3, 4, 10, 11]).astype(int)

    conditions = [
        out["is_sw_monsoon"] == 1,
        out["is_ne_monsoon"] == 1,
        out["is_inter_monsoon"] == 1,
    ]
    choices = ["SW", "NE", "Inter"]
    out["monsoon_season"] = np.select(conditions, choices, default="Unknown")
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["district", "week_start_date"]).copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])

    out["year"] = out["week_start_date"].dt.year
    out["month"] = out["week_start_date"].dt.month
    out["week_of_year"] = out["week_start_date"].dt.isocalendar().week.astype(int)
    out["sin_week"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["cos_week"] = np.cos(2 * np.pi * out["week_of_year"] / 52)

    if "tmax" in out.columns and "tmin" in out.columns:
        out["tmean"] = (out["tmax"] + out["tmin"]) / 2.0
    elif "tmax" in out.columns:
        out["tmean"] = out["tmax"]
    elif "tmin" in out.columns:
        out["tmean"] = out["tmin"]
    else:
        out["tmean"] = np.nan

    if "rainfall" in out.columns:
        out["rainfall_feat"] = out["rainfall"]
    elif "precipitation" in out.columns:
        out["rainfall_feat"] = out["precipitation"]
    else:
        out["rainfall_feat"] = np.nan

    for lag in range(1, 5):
        out[f"cases_lag_{lag}"] = out.groupby("district")["cases"].shift(lag)

    for lag in (2, 3, 4, 5, 6, 8):
        out[f"rainfall_lag_{lag}"] = out.groupby("district")["rainfall_feat"].shift(lag)

    for lag in range(1, 5):
        out[f"temp_lag_{lag}"] = out.groupby("district")["tmean"].shift(lag)
        if "humidity" in out.columns:
            out[f"humidity_lag_{lag}"] = out.groupby("district")["humidity"].shift(lag)

    out["rolling_mean_2"] = out.groupby("district")["cases"].transform(
        lambda s: s.rolling(window=2, min_periods=1).mean()
    )
    out["rolling_mean_4"] = out.groupby("district")["cases"].transform(
        lambda s: s.rolling(window=4, min_periods=1).mean()
    )
    out["rolling_std_4"] = out.groupby("district")["cases"].transform(
        lambda s: s.rolling(window=4, min_periods=1).std()
    )
    out["rate_of_change"] = out["cases"] - out["cases_lag_1"]
    out["percentage_change"] = out.groupby("district")["cases"].pct_change()

    if "rainfall_feat" in out.columns:
        rain_baseline = out.groupby(["district", "month"])["rainfall_feat"].transform("mean")
        out["rainfall_anomaly"] = out["rainfall_feat"] - rain_baseline

    if "tmean" in out.columns:
        temp_baseline = out.groupby(["district", "month"])["tmean"].transform("mean")
        out["temperature_anomaly"] = out["tmean"] - temp_baseline

    if "humidity" in out.columns:
        hum_baseline = out.groupby(["district", "month"])["humidity"].transform("mean")
        out["humidity_anomaly"] = out["humidity"] - hum_baseline

    out = add_monsoon_features(out)
    out["week_start_date"] = out["week_start_date"].dt.strftime("%Y-%m-%d")
    return out


def build_module2_features(engineered: pd.DataFrame) -> pd.DataFrame:
    """Select columns required by Module 2 classification."""
    available = [c for c in MODULE2_FEATURE_COLUMNS if c in engineered.columns]
    missing = [c for c in MODULE2_FEATURE_COLUMNS if c not in engineered.columns]
    if missing:
        log(f"  WARNING: Module 2 columns not found and skipped: {missing}")
    return engineered[available].copy()


def weeks_per_district_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("district")["week_start_date"]
        .nunique()
        .reset_index(name="n_weeks")
        .sort_values("district")
    )


def write_report_file(
    weather_meta: dict,
    dengue_meta: dict,
    weather_before: pd.DataFrame,
    weather_after: pd.DataFrame,
    merged: pd.DataFrame,
    match_rate: float,
    engineered_cols: list[str],
) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DENGUE + WEATHER PREPROCESSING REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 72)
    lines.append("")

    lines.append("INPUTS")
    lines.append("-" * 40)
    lines.append(f"Weather directory: {RAW_WEATHER_DIR}")
    lines.append(f"Weather files processed: {len(weather_meta['files'])}")
    lines.append(f"Dengue file: {RAW_DENGUE_PATH}")
    lines.append(f"Dengue raw rows: {dengue_meta['raw_rows']}")
    lines.append("")

    lines.append("WEATHER PROCESSING")
    lines.append("-" * 40)
    lines.append(f"Weekly weather rows (all districts): {weather_meta['total_rows']}")
    lines.append(f"Districts ({len(weather_meta['districts'])}): {', '.join(weather_meta['districts'])}")
    lines.append(f"Negative rainfall/precipitation values clipped to 0: {weather_meta['negative_rainfall_fixed']}")
    lines.append("Wind speed/gust: not present in source CSVs; columns omitted from outputs.")
    lines.append("")
    for item in weather_meta["files"]:
        lines.append(f"  {item['district']}: {item['weekly_rows']} weeks (from {item['raw_rows']} daily rows)")
    lines.append("")

    lines.append("DENGUE PROCESSING")
    lines.append("-" * 40)
    lines.append(f"Weekly dengue rows: {dengue_meta['weekly_rows']}")
    lines.append(f"Districts ({len(dengue_meta['districts'])}): {', '.join(dengue_meta['districts'])}")
    lines.append(f"Date range: {dengue_meta['date_range'][0]} to {dengue_meta['date_range'][1]}")
    lines.append(f"Duplicate district-week rows aggregated: {dengue_meta['duplicates_aggregated']}")
    lines.append(f"Negative cases set to NaN: {dengue_meta['negative_cases_set_nan']}")
    lines.append("")
    lines.append("NOTE: Missing dengue cases are kept as NaN (not imputed).")
    lines.append("")

    lines.append("MISSINGNESS — WEATHER (before imputation)")
    lines.append("-" * 40)
    for _, row in weather_before.iterrows():
        lines.append(f"  {row['column']}: {row['missing']} ({row['missing_pct']}%)")
    lines.append("")

    lines.append("MISSINGNESS — WEATHER (after imputation)")
    lines.append("-" * 40)
    for _, row in weather_after.iterrows():
        lines.append(f"  {row['column']}: {row['missing']} ({row['missing_pct']}%)")
    lines.append("")

    lines.append("MERGE (left join: dengue -> weather)")
    lines.append("-" * 40)
    lines.append(f"Merged rows: {len(merged)}")
    lines.append(f"Weather match rate (dengue rows with weather): {match_rate:.2f}%")
    lines.append("")

    lines.append("VALIDATION SUMMARY")
    lines.append("-" * 40)
    wpd = weeks_per_district_summary(merged)
    lines.append(f"Number of districts in merged data: {merged['district'].nunique()}")
    lines.append(
        f"Weeks per district — min: {wpd['n_weeks'].min()}, max: {wpd['n_weeks'].max()}, mean: {wpd['n_weeks'].mean():.1f}"
    )
    lines.append("")

    lines.append("FEATURE ENGINEERING")
    lines.append("-" * 40)
    lines.append("Anomaly baseline: district-month mean (seasonal climatology per district).")
    lines.append("Monsoon encoding: binary flags + categorical monsoon_season {SW, NE, Inter}.")
    lines.append("")
    lines.append("Engineered columns added:")
    for col in engineered_cols:
        lines.append(f"  - {col}")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 40)
    lines.append(f"  {WEATHER_WEEKLY_PATH}")
    lines.append(f"  {DENGUE_WEEKLY_PATH}")
    lines.append(f"  {MERGED_WEEKLY_PATH}")
    lines.append(f"  {ENGINEERED_FEATURES_PATH}")
    lines.append(f"  {MODULE2_FEATURES_PATH}")
    lines.append(f"  {PREPROCESSING_REPORT_PATH}")
    lines.append("")

    ensure_dir(PREPROCESSING_REPORT_PATH.parent)
    PREPROCESSING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_preprocessing() -> None:
    """Run the full preprocessing pipeline."""
    global REPORT
    REPORT = []

    log("Starting dengue + weather preprocessing pipeline")
    ensure_dir(PROCESSED_DIR)
    ensure_dir(FEATURES_DIR)

    log("\n[1/5] Processing weather data...")
    weather_weekly, weather_meta = process_all_weather()

    log("\n[2/5] Processing dengue data...")
    dengue_weekly, dengue_meta = process_dengue(weather_meta["districts"])

    log("\n[3/5] Imputing missing weather values...")
    weather_imputed, miss_before, miss_after = impute_weather(weather_weekly)

    log("\n[4/5] Merging dengue with weather...")
    merged, match_rate = merge_datasets(dengue_weekly, weather_imputed)
    log(f"  Merge match rate: {match_rate:.2f}%")

    log("\n[5/5] Engineering features...")
    base_cols = set(merged.columns)
    engineered = engineer_features(merged)
    engineered_cols = sorted(set(engineered.columns) - base_cols)
    module2_features = build_module2_features(engineered)

    save_csv(weather_imputed, WEATHER_WEEKLY_PATH)
    save_csv(dengue_weekly, DENGUE_WEEKLY_PATH)
    save_csv(merged, MERGED_WEEKLY_PATH)
    save_csv(engineered, ENGINEERED_FEATURES_PATH)
    save_csv(module2_features, MODULE2_FEATURES_PATH)

    write_report_file(
        weather_meta=weather_meta,
        dengue_meta=dengue_meta,
        weather_before=miss_before,
        weather_after=miss_after,
        merged=engineered,
        match_rate=match_rate,
        engineered_cols=engineered_cols,
    )

    log("\nDone.")
    log(f"  Weather:   {WEATHER_WEEKLY_PATH}")
    log(f"  Dengue:    {DENGUE_WEEKLY_PATH}")
    log(f"  Merged:    {MERGED_WEEKLY_PATH}")
    log(f"  Features:  {ENGINEERED_FEATURES_PATH}")
    log(f"  Module 2:  {MODULE2_FEATURES_PATH}")
    log(f"  Report:    {PREPROCESSING_REPORT_PATH}")


if __name__ == "__main__":
    run_preprocessing()
