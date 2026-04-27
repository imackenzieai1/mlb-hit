"""Wrapper for stacked-isotonic-recalibration model.

Lives in the importable package (not in scripts/) so joblib can find this
class at unpickle time when load_model("xgb_v3_recal") is called from
anywhere in the codebase.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class StackedCalibratedModel:
    """A base classifier with a post-hoc isotonic head stacked on its
    predict_proba output.

    Used by `scripts/recalibrate_isotonic.py` to correct calibration drift
    without retraining the underlying XGBoost trees. predict_proba(X)
    returns a (n, 2) array compatible with sklearn's calibrated classifiers
    so the existing predict.py / score_today.py pipeline accepts it
    unchanged.
    """

    # Hard caps on recalibrated output. Without these, the isotonic step
    # function can return 1.0 for any input above the rightmost training
    # point if the tail bucket happened to be all-hits — this is overfit
    # to a tiny tail (often n<30) and inflates edges. 0.90 is the realistic
    # ceiling for a hit-prop probability (no batter hits 90% of the time
    # even in their best matchups). 0.05 prevents the symmetric pathology
    # at the bottom.
    P_FLOOR = 0.05
    P_CEILING = 0.90

    def __init__(
        self,
        base,
        isotonic: IsotonicRegression,
        recal_meta: dict | None = None,
    ):
        self.base = base
        self.iso = isotonic
        self.recal_meta = recal_meta or {}

    def predict_proba(self, X) -> np.ndarray:
        p = self.base.predict_proba(X)[:, 1]
        p_recal = self.iso.transform(p)
        # Cap at realistic hit-rate bounds. See class-level docstring.
        p_recal = np.clip(p_recal, self.P_FLOOR, self.P_CEILING)
        return np.column_stack([1.0 - p_recal, p_recal])
