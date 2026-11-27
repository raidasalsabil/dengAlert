"""Requirement 2: intra-city overflow routing (never crosses a city boundary)."""
import pandas as pd


def route_overflow(a: pd.DataFrame):
    """a: output of allocate(). Returns (ward_table, transfers, city_table)."""
    w = a[["city", "ward_id", "hospital_beds_required", "bed_capacity_eff"]].set_index("ward_id").copy()
    w["net_deficit"] = (w.hospital_beds_required - w.bed_capacity_eff).clip(lower=0)
    w["sent"] = 0
    w["received"] = 0
    w["unplaced"] = 0
    moves = []
    for city, g in w.groupby("city"):
        ids = list(g.index)
        load = {i: int(w.at[i, "hospital_beds_required"]) for i in ids}
        cap = {i: int(w.at[i, "bed_capacity_eff"]) for i in ids}
        for src in g[g.net_deficit > 0].sort_values("net_deficit", ascending=False).index:
            need = int(w.at[src, "net_deficit"])
            while need > 0:
                donors = [i for i in ids if i != src and load[i] < cap[i]]
                if not donors:
                    break
                # each patient goes to the donor ward with the lowest projected utilisation
                best = min(donors, key=lambda i: (load[i] + 1) / max(cap[i], 1))
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
    w["headroom_left"] = (w.bed_capacity_eff - w.final_load).clip(lower=0)
    w["utilization_pct"] = (100 * w.hospital_beds_required / w.bed_capacity_eff.clip(lower=1)).round(0)
    w = w.reset_index()
    c = w.groupby("city").agg(demand=("hospital_beds_required", "sum"),
                              capacity=("bed_capacity_eff", "sum"),
                              overflow=("net_deficit", "sum"), routed=("sent", "sum"),
                              unplaced=("unplaced", "sum"), headroom_left=("headroom_left", "sum"))
    c["utilization_pct"] = (100 * c.demand / c.capacity.clip(lower=1)).round(1)
    return w, tr, c.reset_index()