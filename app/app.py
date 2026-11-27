"""dengAlert dashboard: ward-level dengue risk 2-4 weeks ahead.

Run from the project root:  python -m streamlit run app/app.py
Runs fully locally from a serialized model: no API calls.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.risk import TIERS, to_tier  # noqa: E402
from src.forecast import snapshot  # noqa: E402
from app.ops_ui import render_ops  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

TIER_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Critical": "#c62828"}
NICE = {
    "rain_sum2w": "Rain, last 2 weeks (mm)", "rain_sum4w": "Rain, last 4 weeks (mm)",
    "rain_sum6w": "Rain, last 6 weeks (mm)", "rain_sum_lag2_4": "Rain 2-4 weeks ago (mm)",
    "water_logging_risk_index": "Water-logging risk", "sanitation_score": "Sanitation score",
    "population_density_sqkm": "Population density", "week_sin": "Season (sin)",
    "week_cos": "Season (cos)", "month": "Month", "city_code": "City",
    "avg_temp_c_mean4w": "Avg temp, 4-week mean (C)",
    "relative_humidity_pct_mean4w": "Humidity, 4-week mean (%)",
}


def nice(name: str) -> str:
    if name in NICE:
        return NICE[name]
    for base, label in (("rain_lag", "Rain"), ("avg_temp_c_lag", "Temp"),
                        ("relative_humidity_pct_lag", "Humidity")):
        if name.startswith(base):
            return f"{label} {name[len(base):]} wk ago"
    return name


@st.cache_resource
def load_model():
    meta = json.loads(META_PATH.read_text())
    return joblib.load(MODEL_PATH), meta


@st.cache_data
def load_data():
    """Load features once and score every complete row in one batch."""
    model, meta = load_model()
    df = pd.read_parquet(FEATURES_PATH)
    cols = meta["features"]
    ok = df[cols].notna().all(axis=1)
    df["pred"] = np.nan
    df.loc[ok, "pred"] = model.predict(df.loc[ok, cols])
    df["tier"] = None
    df.loc[ok, "tier"] = to_tier(df.loc[ok, "pred"], meta["tier_cutoffs"])
    return df


def contributions(model, row: pd.DataFrame, cols):
    """Per-feature push on the (log) prediction, via XGBoost's built-in SHAP values."""
    dm = xgb.DMatrix(row[cols], feature_names=cols)
    best = getattr(model, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    c = model.get_booster().predict(dm, pred_contribs=True, iteration_range=rng)[0]
    return pd.Series(c[:-1], index=cols)


def badge(tier: str) -> str:
    return (f"<div style='background:{TIER_COLORS[tier]};color:white;padding:14px 18px;"
            f"border-radius:10px;font-size:28px;font-weight:700;text-align:center'>{tier}</div>")


st.set_page_config(page_title="dengueAlert", layout="wide")
st.title("dengueAlert: dengue early warning")
st.caption("Predicts dengue cases per ward in the 2-4 weeks after a chosen date, "
           "from lagged rainfall, temperature, humidity and ward vulnerability.")

if not (MODEL_PATH.exists() and FEATURES_PATH.exists()):
    st.error("Model or features missing. Run `python -m src.train` first.")
    st.stop()

model, meta = load_model()
cols = meta["features"]
sp = meta["split"]
cut = meta["tier_cutoffs"]
data = load_data()
scored = data.dropna(subset=["pred"])

# ---------------- Inputs ----------------
st.sidebar.header("Select area and time")
city = st.sidebar.selectbox("City", sorted(data["city"].unique()))
ward = st.sidebar.selectbox("Ward", sorted(data.loc[data["city"] == city, "ward_id"].unique()))
ward_rows = scored[scored["ward_id"] == ward].sort_values("date")
dates = ward_rows["date"].dt.date.tolist()
last_year = scored[scored["date"] > scored["date"].max() - pd.Timedelta(weeks=52)]
peak = last_year.groupby("date")["pred"].mean().idxmax().date()
sel_date = st.sidebar.select_slider("Forecast date (week starting)", options=dates,
                                    value=peak if peak in dates else dates[-1])
sel_ts = pd.Timestamp(sel_date)
st.sidebar.info("The forecast covers the 2-4 weeks AFTER this date, using only weather "
                "known up to this date.")
st.sidebar.caption(f"Risk tiers by predicted cases: Low < {cut[0]:.1f} | Medium < {cut[1]:.1f} | "
                   f"High < {cut[2]:.1f} | Critical above.")

row = ward_rows[ward_rows["date"] == sel_ts]
pred = float(row["pred"].iloc[0])
tier = str(row["tier"].iloc[0])

tab1, tab2, tab3, tab4 = st.tabs(["Ward forecast", "City overview", "Model accuracy",
                                  "Operations command"])
# ---------------- Tab 1: forecast + drivers ----------------
with tab1:
    start, end = sel_ts + pd.Timedelta(weeks=2), sel_ts + pd.Timedelta(weeks=4)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.markdown(badge(tier), unsafe_allow_html=True)
    c2.metric("Predicted cases (2-4 wks)", f"{pred:.1f}")
    c3.write(f"**{ward}** ({city})  \nWindow: **{start:%d %b %Y} to {end:%d %b %Y}**")
    actual = row["y_2_4w"].iloc[0]
    if pd.notna(actual):
        c3.write(f"Actual cases in that window: **{actual:.0f}**")
    if sel_ts <= pd.Timestamp(sp["train_end"]):
        st.caption("This date is in the training period, so the model has seen the outcome. "
                   "Pick a 2025 date for an honest forecast.")
    elif sel_ts <= pd.Timestamp(sp["val_end"]):
        st.caption("This date is in the validation period (2023-2024).")

    left, right = st.columns(2)
    with left:
        st.subheader("Main drivers of this forecast")
        contrib = contributions(model, row, cols)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)[::-1]
        vals = row[cols].iloc[0]
        labels = [f"{nice(k)} = {vals[k]:.2f}" for k in top.index]
        fig = go.Figure(go.Bar(x=top.values, y=labels, orientation="h",
                               marker_color=["#c62828" if v > 0 else "#1565c0" for v in top.values]))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Effect on risk (red raises, blue lowers)")
        st.plotly_chart(fig)
    with right:
        st.subheader("Weather lags used")
        lag_tbl = pd.DataFrame({
            "Weeks before date": [1, 2, 3, 4],
            "Rainfall (mm)": [row[f"rain_lag{k}"].iloc[0] for k in (1, 2, 3, 4)],
            "Avg temp (C)": [row[f"avg_temp_c_lag{k}"].iloc[0] for k in (1, 2, 3, 4)],
            "Humidity (%)": [row[f"relative_humidity_pct_lag{k}"].iloc[0] for k in (1, 2, 3, 4)],
        })
        st.dataframe(lag_tbl.round(2), hide_index=True)
        st.caption(f"Rain last 2 / 4 / 6 weeks: {row['rain_sum2w'].iloc[0]:.0f} / "
                   f"{row['rain_sum4w'].iloc[0]:.0f} / {row['rain_sum6w'].iloc[0]:.0f} mm")

    st.subheader("Recent cases and forecast")
    recent = data[(data["ward_id"] == ward) & (data["date"] <= sel_ts)].tail(26)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=recent["date"], y=recent["dengue_cases"], name="Weekly cases",
                              mode="lines+markers"))
    fig3.add_trace(go.Scatter(x=[start, end], y=[pred / 3, pred / 3], name="Forecast (avg per week)",
                              mode="lines+markers", line=dict(color=TIER_COLORS[tier], width=4)))
    fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Cases per week")
    st.plotly_chart(fig3)

# ---------------- Tab 2: all wards ----------------
with tab2:
    st.subheader(f"Ward risk ranking: forecast date {sel_date}")
    snap = scored[scored["date"] == sel_ts][["city", "ward_id", "pred", "tier"]].copy()
    snap["pred"] = snap["pred"].round(1)
    snap["tier"] = pd.Categorical(snap["tier"], TIERS, ordered=True)
    snap = snap.sort_values(["tier", "pred"], ascending=False).rename(
        columns={"pred": "predicted cases (2-4 wks)"})
    counts = snap["tier"].value_counts().reindex(TIERS).fillna(0).astype(int)
    for c, t in zip(st.columns(4), TIERS):
        c.markdown(f"<div style='color:{TIER_COLORS[t]};font-weight:700'>{t}</div>",
                   unsafe_allow_html=True)
        c.metric("wards", int(counts[t]), label_visibility="collapsed")
    st.dataframe(snap, hide_index=True)

    st.subheader(f"Risk heatmap: {city}, last 20 weeks")
    hm = scored[(scored["city"] == city) & (scored["date"] <= sel_ts)
                & (scored["date"] > sel_ts - pd.Timedelta(weeks=20))]
    pivot = hm.pivot(index="ward_id", columns="date", values="pred")
    fig4 = px.imshow(pivot.values, x=[d.strftime("%d %b %y") for d in pivot.columns], y=pivot.index,
                     color_continuous_scale="YlOrRd", aspect="auto",
                     labels=dict(color="Predicted cases"))
    fig4.update_layout(height=max(300, 24 * len(pivot) + 120), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig4)

# ---------------- Tab 3: validation ----------------
with tab3:
    st.subheader("Chronological validation (no shuffling)")
    st.write(f"Trained on data up to **{sp['train_end']}** (minus a {sp['purge_weeks']}-week purge), "
             f"tuned on 2023-2024, tested on **2025**, which the model never saw.")
    rows = []
    for split in ("val", "test"):
        for name, m in meta["metrics"][split].items():
            rows.append({"split": "Validation 2023-24" if split == "val" else "Test 2025",
                         "model": name.replace("_", " "),
                         "MAE": round(m["MAE"], 2), "RMSE": round(m["RMSE"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("MAE/RMSE are in cases per 2-4 week window per ward. Lower is better.")

    # ---------------- Tab 4: operations (Phase 2) ----------------
with tab4:
    render_ops(snapshot(scored, sel_ts), model, meta, str(sel_date))