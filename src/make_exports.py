"""python -m src.make_exports  -> writes sample SMS/WhatsApp/HTML briefs to /exports."""
from src.capacity import load_capacity
from src.dispatch import export_all
from src.forecast import load_scored, snapshot
from src.overflow import route_overflow
from src.resources import allocate_regions

if __name__ == "__main__":
    model, meta, sc = load_scored()
    s25 = sc[sc["date"] >= "2025-01-01"]
    peak = s25.groupby("date")["pred"].mean().idxmax()
    fc = snapshot(sc, peak)
    settings = {c: (60, 8) for c in fc.city.unique()}     # per-city stock % and crew pool
    a = allocate_regions(fc, load_capacity(), settings)
    _, transfers, _ = route_overflow(a)
    export_all(a, str(peak.date()), transfers)
    print("exports written for", peak.date())
