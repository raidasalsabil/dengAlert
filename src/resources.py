"""Requirement 1: constrained resource allocation with scarcity controls."""
import numpy as np
import pandas as pd
from src.config import (BED_SURGE_FACTOR, BEDS_PER_WEEKLY_CASE, KITS_PER_CASE,
                        SALINE_PER_CASE, TIER_CREWS, TIER_WEIGHT, WINDOW_WEEKS)


def supplies_from_cases(cases):
    """One conversion used by allocation AND the simulator, so they always agree."""
    c = np.clip(np.asarray(cases, dtype=float), 0, None)
    return {
        "saline": np.ceil(c * SALINE_PER_CASE).astype(int),
        "kits": np.ceil(c * KITS_PER_CASE).astype(int),
        "beds": np.ceil(c / WINDOW_WEEKS * BEDS_PER_WEEKLY_CASE * BED_SURGE_FACTOR).astype(int),
    }


def compute_demand(fc: pd.DataFrame) -> pd.DataFrame:
    """fc needs: ward_id, city, pred_cases, tier, population_density_sqkm."""
    d = fc[["ward_id", "city", "pred_cases", "tier", "population_density_sqkm"]].copy()
    d["tier"] = d["tier"].astype(str)
    s = supplies_from_cases(d.pred_cases)
    d["saline_bags_needed"] = s["saline"]
    d["diagnostic_kits_needed"] = s["kits"]
    d["hospital_beds_required"] = s["beds"]
    d["crews_needed"] = d.tier.map(TIER_CREWS).astype(int)
    dens = d.population_density_sqkm / d.population_density_sqkm.max()
    # risk x expected burden x crowding: high-risk, dense wards are served first
    d["priority"] = d.tier.map(TIER_WEIGHT) * d.pred_cases.clip(lower=0) * (0.5 + dens)
    return d


def _fill(need: pd.Series, order, pool: int) -> pd.Series:
    """Serve wards in priority order until the pool runs out (vectorised cumsum)."""
    n = need.loc[order].to_numpy()
    before = np.concatenate([[0], np.cumsum(n)[:-1]])
    got = np.clip(pool - before, 0, n).astype(int)
    return pd.Series(got, index=order).reindex(need.index)


def allocate(fc: pd.DataFrame, capacity: pd.DataFrame, stock_pct: float, total_crews: int):
    """stock_pct 30-100 scales ONLY the saline and kit pools; total_crews is the crew pool.
    Beds are NOT allocated here: bed need vs each ward's dengue_bed_capacity (from the
    capacity file) is handled by overflow routing and never depends on any slider."""
    d = compute_demand(fc).merge(capacity, on="ward_id", how="left")
    if d.dengue_bed_capacity.isna().any():
        missing = d.loc[d.dengue_bed_capacity.isna(), "ward_id"].tolist()
        raise ValueError(f"No bed capacity in the capacity file for wards: {missing}")
    s = stock_pct / 100.0
    d = d.sort_values("priority", ascending=False).reset_index(drop=True)
    order = d.index
    for need, got in (("saline_bags_needed", "saline_allocated"),
                      ("diagnostic_kits_needed", "kits_allocated")):
        d[got] = _fill(d[need], order, int(np.floor(d[need].sum() * s)))
    d["crews_assigned"] = _fill(d["crews_needed"], order, int(total_crews))

    d["saline_deficit"] = d.saline_bags_needed - d.saline_allocated
    d["kits_deficit"] = d.diagnostic_kits_needed - d.kits_allocated
    d["crew_deficit"] = d.crews_needed - d.crews_assigned
    short = (d.saline_deficit > 0) | (d.kits_deficit > 0) | (d.crew_deficit > 0)
    d["alert"] = np.where(short, "CRITICAL SUPPLY DEFICIT", "OK")
    items = []
    for r in d.itertuples():
        x = [n for n, v in (("saline", r.saline_deficit), ("kits", r.kits_deficit),
                            ("crews", r.crew_deficit)) if v > 0]
        items.append(", ".join(x))
    d["deficit_items"] = items
    return d

def allocate_regions(fc: pd.DataFrame, capacity: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """One region = one city. settings: {city: (stock_pct, crews)}.
    Each city allocates only from its OWN stock pool and its OWN crew pool."""
    parts = []
    for city, g in fc.groupby("city"):
        stock, crews = settings.get(city, (100, 0))
        parts.append(allocate(g.reset_index(drop=True), capacity, stock, crews))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["city", "priority"], ascending=[True, False]).reset_index(drop=True)
