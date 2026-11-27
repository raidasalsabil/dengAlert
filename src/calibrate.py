"""Derive every data-driven constant from the onsite dataset.

python -m src.calibrate            -> writes models/calibration.json
Ratios are ratio-of-sums over weeks with >= MIN_CASES cases, so rounding noise at tiny
counts does not bias them. Week shares come from how window cases are actually spread
across weeks +2, +3, +4 in the history.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "calibration.json"
MIN_CASES = 5


def find_dataset() -> Path:
    hits = sorted((ROOT / "data" / "raw").glob("*onsite*.xlsx"))
    if not hits:
        raise FileNotFoundError("No *onsite*.xlsx dataset found in data/raw")
    return hits[0]


def fit(df: pd.DataFrame) -> dict:
    df = df.sort_values(["ward_id", "date"])
    x = df[df.dengue_cases >= MIN_CASES]
    cases = x.dengue_cases.sum()
    g = df.groupby("ward_id").dengue_cases
    lead = pd.DataFrame({w: g.shift(-w) for w in (2, 3, 4)}).dropna()
    share = (lead.sum() / lead.to_numpy().sum()).to_dict()
    return {
        "saline_per_case": float(x.saline_bags_needed.sum() / cases),
        "kits_per_case": float(x.diagnostic_kits_needed.sum() / cases),
        "beds_per_weekly_case": float(x.hospital_beds_required.sum() / cases),
        "week_share": {str(k): float(v) for k, v in share.items()},
        "fit_rows": int(len(x)),
        "source": "ratio of sums over weeks with >= %d cases" % MIN_CASES,
    }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_dataset()
    cal = fit(pd.read_excel(path))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(cal, indent=2))
    print("source file:", path.name)
    print(json.dumps(cal, indent=2))