import pandas as pd
from src.config import CAPACITY_XLSX


def load_capacity(path=CAPACITY_XLSX) -> pd.DataFrame:
    """ward_id, total_hospital_beds, dengue_bed_capacity (drops the Total row)."""
    c = pd.read_excel(path, sheet_name="bed_capacity").dropna(subset=["ward_id"])
    return c[["ward_id", "total_hospital_beds", "dengue_bed_capacity"]]