import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_XLSX = ROOT / "data" / "raw" / "bed_capacity_wardwise_scarce.xlsx"
CALIBRATION_JSON = ROOT / "models" / "calibration.json"

# ---- Data-derived constants: produced by `python -m src.calibrate` ----
if not CALIBRATION_JSON.exists():
    raise FileNotFoundError("models/calibration.json missing. Run: python -m src.calibrate")
_cal = json.loads(CALIBRATION_JSON.read_text())

WINDOW_WEEKS = 3                                   # forecast covers weeks +2, +3, +4
SALINE_PER_CASE = _cal["saline_per_case"]          # bags per case (window total)
KITS_PER_CASE = _cal["kits_per_case"]              # kits per case (window total)
BEDS_PER_WEEKLY_CASE = _cal["beds_per_weekly_case"]  # beds per weekly case (concurrent)
WEEK_SHARE = {int(k): v for k, v in _cal["week_share"].items()}  # share of window cases per week

# ---- Operating policy parameters (NOT in the dataset; exposed as scenario controls) ----
BED_SURGE_FACTOR = 1.0                              # >1 plans for the peak week
TIER_CREWS = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}   # crews a ward of this tier needs
TIER_WEIGHT = {"Low": 1, "Medium": 2, "High": 4, "Critical": 8}  # dispatch priority by tier
CREW_EFFECT = 0.15                                  # default per-crew breeding reduction (slider)
MAX_CREWS_PER_WARD = 5