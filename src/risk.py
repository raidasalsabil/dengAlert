"""Risk tier logic shared by training and the dashboard."""
import numpy as np

TIERS = ["Low", "Medium", "High", "Critical"]


def to_tier(predicted_cases, cutoffs):
    """Map predicted cases (next 2-4 weeks) to Low/Medium/High/Critical.

    cutoffs = [medium_start, high_start, critical_start], learned from training data only.
    """
    idx = np.digitize(np.asarray(predicted_cases, dtype=float), cutoffs)
    return np.array(TIERS)[idx]