"""python -m src.make_exports  -> writes sample SMS/WhatsApp/HTML briefs to /exports."""
from src.capacity import load_capacity
from src.dispatch import export_all
from src.forecast import load_scored, snapshot
from src.overflow import route_overflow
from src.resources import allocate

if __name__ == "__main__":
    model, meta, sc = load_scored()
    s25 = sc[sc["date"] >= "2025-01-01"]
    peak = s25.groupby("date")["pred"].mean().idxmax()
    a = allocate(snapshot(sc, peak), load_capacity(), stock_pct=60, total_crews=40)
    _, transfers, _ = route_overflow(a)
    export_all(a, str(peak.date()), transfers)
    print("exports written for", peak.date())