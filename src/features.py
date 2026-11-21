"""Weather lag feature pipeline for dengAlert.

All lag/rolling features are computed per ward_id on a weekly series, so no
information leaks across wards or from the future.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_XLSX = Path("data/raw/hackathon_preli_dataset.xlsx")
RAW_PARQUET = Path("data/processed/raw.parquet")
FEATURES_PARQUET = Path("data/processed/features.parquet")

STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
RAIN_LAGS = [1, 2, 3, 4, 5, 6]
CLIMATE_LAGS = [1, 2, 3, 4]
HORIZONS = [2, 3, 4]  # forecast weeks ahead


def load_raw() -> pd.DataFrame:
    """Load raw data (parquet if present, otherwise the xlsx) and normalise columns."""
    df = pd.read_parquet(RAW_PARQUET) if RAW_PARQUET.exists() else pd.read_excel(RAW_XLSX)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-week lag, rolling and seasonal features per ward."""
    df = df.sort_values(["ward_id", "date"]).copy()
    g = df.groupby("ward_id", sort=False)
    new = {}

    for k in RAIN_LAGS:
        new[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    for w in (2, 4, 6):
        new[f"rain_sum{w}w"] = g["rainfall_mm"].transform(lambda s: s.rolling(w).sum())
    # rainfall in the 2-4 week-ago window: the "vector breeding clock"
    new["rain_sum_lag2_4"] = sum(g["rainfall_mm"].shift(k) for k in (2, 3, 4))

    for col in ("avg_temp_c", "relative_humidity_pct"):
        for k in CLIMATE_LAGS:
            new[f"{col}_lag{k}"] = g[col].shift(k)
        new[f"{col}_mean4w"] = g[col].transform(lambda s: s.rolling(4).mean())

    woy = df["date"].dt.isocalendar().week.astype(int)
    new["week_sin"] = np.sin(2 * np.pi * woy / 52)
    new["week_cos"] = np.cos(2 * np.pi * woy / 52)
    new["month"] = df["date"].dt.month

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    df["city_code"] = df["city"].astype("category").cat.codes
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Future cases per ward: y_h2, y_h3, y_h4 and their 2-4 week sum."""
    g = df.groupby("ward_id", sort=False)["dengue_cases"]
    for h in HORIZONS:
        df[f"y_h{h}"] = g.shift(-h)
    df["y_2_4w"] = df[[f"y_h{h}" for h in HORIZONS]].sum(axis=1, min_count=len(HORIZONS))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    lag_cols = [c for c in df.columns if "_lag" in c or c.startswith(("rain_sum", "week_"))
                or c.endswith("mean4w")]
    return STATIC_COLS + ["city_code", "month"] + lag_cols


def run(save: bool = True) -> pd.DataFrame:
    df = add_targets(build_features(load_raw()))
    if save:
        FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_PARQUET, index=False)
    return df


if __name__ == "__main__":
    out = run()
    print(f"Saved {FEATURES_PARQUET}: {out.shape[0]} rows, {len(feature_columns(out))} features")