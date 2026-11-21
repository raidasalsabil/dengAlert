"""Train the dengue early-warning model with a chronological split.

Target: total dengue cases in a ward over the 2-4 weeks AFTER the forecast date.
Run from the project root:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src import features as F
from src.risk import to_tier

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/meta.json")
PRED_PATH = Path("data/processed/eval_predictions.parquet")

TARGET = "y_2_4w"
TRAIN_END = "2022-12-31"   # train: 2016-2022
VAL_END = "2024-12-31"     # validation: 2023-2024, test: 2025
PURGE_WEEKS = 4            # target looks 4 weeks ahead, so drop the last 4 train weeks


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    df = F.run(save=True)
    cols = F.feature_columns(df)
    df = df.dropna(subset=cols + [TARGET]).copy()

    train = df[df["date"] <= TRAIN_END]
    train = train[train["date"] <= pd.Timestamp(TRAIN_END) - pd.Timedelta(weeks=PURGE_WEEKS)]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END]
    print(f"train {len(train)} | val {len(val)} | test {len(test)} rows")

    # Baselines for comparison
    clim = train.groupby(["ward_id", "month"])[TARGET].mean().rename("clim")
    def climatology(d):
        return d.join(clim, on=["ward_id", "month"])["clim"].fillna(train[TARGET].mean()).values
    def persistence(d):
        return d["dengue_cases"].values * 3  # this week's cases carried over 3 weeks

    # Poisson loss suits counts with many zeros
    model = XGBRegressor(
        objective="count:poisson", n_estimators=600, learning_rate=0.04, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        early_stopping_rounds=40, random_state=42, n_jobs=-1,
    )
    model.fit(train[cols], train[TARGET], eval_set=[(val[cols], val[TARGET])], verbose=False)
    print("best iteration:", model.best_iteration)

    metrics = {}
    for name, d in [("val", val), ("test", test)]:
        y = d[TARGET].values
        metrics[name] = {
            "model": {"MAE": float(mean_absolute_error(y, model.predict(d[cols]))),
                      "RMSE": rmse(y, model.predict(d[cols]))},
            "climatology_baseline": {"MAE": float(mean_absolute_error(y, climatology(d))),
                                     "RMSE": rmse(y, climatology(d))},
            "persistence_baseline": {"MAE": float(mean_absolute_error(y, persistence(d))),
                                     "RMSE": rmse(y, persistence(d))},
        }
    print(json.dumps(metrics, indent=2))

    # Risk tiers: cut-offs learned from TRAIN predictions only (no leakage)
    train_pred = model.predict(train[cols])
    cutoffs = [float(np.quantile(train_pred, q)) for q in (0.60, 0.85, 0.95)]
    print("tier cutoffs (predicted cases in 2-4 wks):", [round(c, 2) for c in cutoffs])

    ev = pd.concat([val.assign(split="val"), test.assign(split="test")])
    ev["predicted"] = model.predict(ev[cols])
    ev["tier"] = to_tier(ev["predicted"], cutoffs)
    ev[["date", "city", "ward_id", "split", TARGET, "predicted", "tier"]].to_parquet(PRED_PATH, index=False)

    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("top features:\n", imp.head(8).round(3).to_string())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "features": cols, "target": TARGET, "tier_cutoffs": cutoffs, "metrics": metrics,
        "split": {"train_end": TRAIN_END, "val_end": VAL_END, "purge_weeks": PURGE_WEEKS},
    }, indent=2))
    print("saved", MODEL_PATH, META_PATH)


if __name__ == "__main__":
    main()