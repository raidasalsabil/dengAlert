import numpy as np
import pandas as pd
from src.resources import allocate
from src.overflow import route_overflow
from src.simulator import case_curve, optimal_deployment, simulate

CUT = [2, 5, 10]


def make(n=12):
    rng = np.random.default_rng(0)
    cities = ["Dhaka North"] * 6 + ["Gazipur"] * 6
    fc = pd.DataFrame({"ward_id": [f"W{i}" for i in range(n)], "city": cities,
                       "pred_cases": rng.uniform(0, 40, n),
                       "population_density_sqkm": rng.uniform(1e4, 6e4, n)})
    fc["tier"] = np.select([fc.pred_cases > 25, fc.pred_cases > 12, fc.pred_cases > 5],
                           ["Critical", "High", "Medium"], "Low")
    cap = pd.DataFrame({"ward_id": fc.ward_id, "total_hospital_beds": 50, "dengue_bed_capacity": 3})
    return fc, cap


def test_scarcity_never_over_allocates():
    fc, cap = make()
    for stock in (30, 60, 100):
        a = allocate(fc, cap, stock, 5)
        assert a.saline_allocated.sum() <= int(a.saline_bags_needed.sum() * stock / 100)
        assert a.crews_assigned.sum() <= 5
        assert (a.saline_allocated <= a.saline_bags_needed).all()


def test_full_stock_has_no_supply_deficit():
    fc, cap = make()
    a = allocate(fc, cap, 100, 999)
    assert (a.saline_deficit == 0).all() and (a.kits_deficit == 0).all()


def test_priority_served_first():
    fc, cap = make()
    a = allocate(fc, cap, 30, 3)
    assert a.iloc[0].crews_assigned == a.iloc[0].crews_needed


def test_overflow_stays_in_city_and_conserves_patients():
    fc, cap = make()
    a = allocate(fc, cap, 100, 10)
    w, tr, c = route_overflow(a)
    city = a.set_index("ward_id").city
    assert all(city[r.from_ward] == city[r.to_ward] for r in tr.itertuples())
    assert w.sent.sum() == w.received.sum() == tr.patients.sum()


class Dummy:
    def predict(self, X):
        return (X["rain_lag1"] * 0.1 + 1).to_numpy()


def test_simulator_monotone_and_ranked():
    fc, _ = make()
    fc["rain_lag1"] = np.linspace(10, 200, len(fc))
    cur = case_curve(Dummy(), ["rain_lag1"], fc)
    assert (np.diff(cur, axis=1) <= 1e-9).all()
    plan = optimal_deployment(cur, fc.ward_id.tolist(), 6)
    assert sum(plan.values()) <= 6
    res = simulate(cur, fc, plan, CUT)
    assert (res.cases_prevented >= -1e-9).all()
    assert res.roi.is_monotonic_decreasing

def test_each_region_uses_only_its_own_pool():
    from src.resources import allocate_regions
    fc, cap = make()
    settings = {"Dhaka North": (30, 2), "Gazipur": (100, 5)}
    a = allocate_regions(fc, cap, settings)
    for city, (stock, crews) in settings.items():
        g = a[a.city == city]
        assert g.saline_allocated.sum() <= int(g.saline_bags_needed.sum() * stock / 100)
        assert g.crews_assigned.sum() <= crews
    g = a[a.city == "Gazipur"]
    assert (g.saline_deficit == 0).all()          # 100% stock -> fully served
