"""Requirement 3: model-driven vector-control counterfactual.

Crews remove breeding sites, so we shrink the inputs that feed the breeding clock
(rainfall lags/sums and water-logging index), then RE-RUN the trained model.
"""
import numpy as np
import pandas as pd
from src.config import CREW_EFFECT, MAX_CREWS_PER_WARD, WEEK_SHARE
from src.resources import supplies_from_cases
from src.risk import to_tier


def _affected(F):
    return [c for c in F if "rain" in c.lower()] + (
        ["water_logging_risk_index"] if "water_logging_risk_index" in F else [])


def case_curve(model, F, rows: pd.DataFrame, kmax=MAX_CREWS_PER_WARD,
               effect=CREW_EFFECT) -> np.ndarray:
    """Predicted window cases for k = 0..kmax crews in every ward -> [wards, kmax+1]."""
    X0 = rows[F].astype(float)
    aff = _affected(F)
    out = np.zeros((len(rows), kmax + 1))
    for k in range(kmax + 1):
        X = X0.copy()
        X[aff] = X0[aff] * np.exp(-effect * k)
        out[:, k] = np.clip(model.predict(X), 0, None)
    return np.minimum.accumulate(out, axis=1)     # more crews never increase cases


def optimal_deployment(curve: np.ndarray, ward_ids, total_crews: int) -> dict:
    """Greedy: each next crew goes where the model says it prevents the most cases."""
    k = np.zeros(len(ward_ids), dtype=int)
    kmax = curve.shape[1] - 1
    for _ in range(int(total_crews)):
        nxt = np.where(k < kmax, curve[np.arange(len(k)), k] - curve[np.arange(len(k)),
                       np.minimum(k + 1, kmax)], -1)
        j = int(nxt.argmax())
        if nxt[j] <= 1e-9:
            break
        k[j] += 1
    return dict(zip(ward_ids, k.tolist()))


def simulate(curve, rows: pd.DataFrame, crews: dict, cutoffs) -> pd.DataFrame:
    ids = rows.ward_id.to_numpy()
    k = np.array([min(int(crews.get(w, 0)), curve.shape[1] - 1) for w in ids])
    base, sim = curve[:, 0], curve[np.arange(len(k)), k]
    prevented = base - sim
    out = pd.DataFrame({"ward_id": ids, "city": rows.city.to_numpy(), "crews": k,
                        "base_cases": base, "sim_cases": sim, "cases_prevented": prevented})
    for wk, share in WEEK_SHARE.items():      # split by how cases really fall across weeks
        out[f"prevented_w{wk}"] = prevented * share
    b, s = supplies_from_cases(base), supplies_from_cases(sim)
    out["beds_before"], out["beds_after"] = b["beds"], s["beds"]
    out["beds_saved"] = b["beds"] - s["beds"]
    out["saline_saved"] = b["saline"] - s["saline"]
    out["tier_before"] = np.asarray(to_tier(base, cutoffs))
    out["tier_after"] = np.asarray(to_tier(sim, cutoffs))
    out["roi"] = np.where(k > 0, prevented / np.maximum(k, 1), 0.0)   # cases prevented per crew
    return out.sort_values(["roi", "cases_prevented"], ascending=False)


def roi_ranking(curve, rows: pd.DataFrame) -> pd.DataFrame:
    """Rank every ward by cases prevented by its FIRST crew (marginal ROI)."""
    r = pd.DataFrame({"ward_id": rows.ward_id.to_numpy(), "city": rows.city.to_numpy(),
                      "base_cases": curve[:, 0], "roi_1st_crew": curve[:, 0] - curve[:, 1],
                      "roi_3_crews_avg": (curve[:, 0] - curve[:, 3]) / 3})
    return r.sort_values("roi_1st_crew", ascending=False).reset_index(drop=True)
