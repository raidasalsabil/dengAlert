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
    """stock_pct 30-100.
    Saline/kits: regional pool = stock% of total demand.
    Beds: usable dengue beds = stock% of each ward's dengue_bed_capacity.
    Crews: a fixed pool of total_crews."""
    d = compute_demand(fc).merge(capacity, on="ward_id", how="left")
    d["dengue_bed_capacity"] = d.dengue_bed_capacity.fillna(1)
    s = stock_pct / 100.0
    d = d.sort_values("priority", ascending=False).reset_index(drop=True)
    order = d.index
    for need, got in (("saline_bags_needed", "saline_allocated"),
                      ("diagnostic_kits_needed", "kits_allocated")):
        d[got] = _fill(d[need], order, int(np.floor(d[need].sum() * s)))
    d["crews_assigned"] = _fill(d["crews_needed"], order, int(total_crews))
    d["bed_capacity_eff"] = np.floor(d.dengue_bed_capacity * s).astype(int)
    d["beds_allocated"] = np.minimum(d.hospital_beds_required, d.bed_capacity_eff)

    d["saline_deficit"] = d.saline_bags_needed - d.saline_allocated
    d["kits_deficit"] = d.diagnostic_kits_needed - d.kits_allocated
    d["crew_deficit"] = d.crews_needed - d.crews_assigned
    d["beds_deficit"] = d.hospital_beds_required - d.beds_allocated

    supply_short = (d.saline_deficit > 0) | (d.kits_deficit > 0) | (d.crew_deficit > 0)
    d["alert"] = np.select([supply_short, d.beds_deficit > 0],
                           ["CRITICAL SUPPLY DEFICIT", "BED SURGE - ROUTE OVERFLOW"], "OK")
    items = []
    for r in d.itertuples():
        x = [n for n, v in (("saline", r.saline_deficit), ("kits", r.kits_deficit),
                            ("crews", r.crew_deficit), ("beds", r.beds_deficit)) if v > 0]
        items.append(", ".join(x))
    d["deficit_items"] = items
    return d