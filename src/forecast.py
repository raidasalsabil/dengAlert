"""Non-Streamlit loader so exports, tests and the app score the same way."""
import json
import joblib
import numpy as np
import pandas as pd
from src.config import ROOT
from src.risk import to_tier

MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


def load_scored():
    meta = json.loads(META_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    cols = meta["features"]
    ok = df[cols].notna().all(axis=1)
    df["pred"] = np.nan
    df.loc[ok, "pred"] = model.predict(df.loc[ok, cols])
    df["tier"] = None
    df.loc[ok, "tier"] = to_tier(df.loc[ok, "pred"], meta["tier_cutoffs"])
    return model, meta, df.dropna(subset=["pred"])


def snapshot(scored: pd.DataFrame, date) -> pd.DataFrame:
    """One row per ward for a forecast date, with pred_cases for the ops modules."""
    s = scored[scored["date"] == pd.Timestamp(date)].copy()
    s["pred_cases"] = s["pred"]
    return s.reset_index(drop=True)