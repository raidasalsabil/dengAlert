"""Requirement 2: intra-city overflow routing (never crosses a city boundary).

Bed need comes from the forecast; bed capacity is each ward's dengue_bed_capacity from the
capacity file. No slider affects any of this.
"""
import pandas as pd


def route_overflow(a: pd.DataFrame):
    """a: output of allocate(). Returns (ward_table, transfers, city_table)."""
    w = a[["city", "ward_id", "hospital_beds_required", "dengue_bed_capacity"]].set_index("ward_id").copy()
    w["net_deficit"] = (w.hospital_beds_required - w.dengue_bed_capacity).clip(lower=0)
    w["utilization_before_pct"] = (100 * w.hospital_beds_required / w.dengue_bed_capacity.clip(lower=1)).round(0)
    w["sent"] = 0
    w["received"] = 0
    w["unplaced"] = 0
    moves = []
    for city, g in w.groupby("city"):
        ids = list(g.index)
        load = {i: int(w.at[i, "hospital_beds_required"]) for i in ids}
        cap = {i: int(w.at[i, "dengue_bed_capacity"]) for i in ids}
        for src in g[g.net_deficit > 0].sort_values("net_deficit", ascending=False).index:
            need = int(w.at[src, "net_deficit"])
            while need > 0:
                # candidate wards: same city, not the sender, still has a free bed
                donors = [i for i in ids if i != src and load[i] < cap[i]]
                if not donors:
                    break
                # send to the ward with the LOWEST current utilisation (load / capacity);
                # ties go to the ward with more free beds, then by ward id
                best = min(donors, key=lambda i: (load[i] / max(cap[i], 1), -(cap[i] - load[i]), i))
                load[best] += 1
                need -= 1
                w.at[best, "received"] += 1
                w.at[src, "sent"] += 1
                moves.append((city, src, best))
            w.at[src, "unplaced"] = need
    tr = pd.DataFrame(moves, columns=["city", "from_ward", "to_ward"])
    if len(tr):
        tr = tr.groupby(["city", "from_ward", "to_ward"]).size().reset_index(name="patients")
    else:
        tr = pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients"])
    w["final_load"] = w.hospital_beds_required - w.sent + w.received
    w["headroom_left"] = (w.dengue_bed_capacity - w.final_load).clip(lower=0)
    w["utilization_after_pct"] = (100 * w.final_load / w.dengue_bed_capacity.clip(lower=1)).round(0)
    w = w.reset_index()
    c = w.groupby("city").agg(demand=("hospital_beds_required", "sum"),
                              capacity=("dengue_bed_capacity", "sum"),
                              overflow=("net_deficit", "sum"), routed=("sent", "sum"),
                              unplaced=("unplaced", "sum"), headroom_left=("headroom_left", "sum"))
    c["utilization_pct"] = (100 * c.demand / c.capacity.clip(lower=1)).round(1)
    return w, tr, c.reset_index()