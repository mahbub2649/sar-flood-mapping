"""Classical SAR water detection, the usual starting point before any learning.

Calm water reflects radar away from the sensor, so it looks dark, especially in VH.
Two variants:
- fixed: one dB threshold for every image, tuned on the training split
- otsu: a threshold chosen per image by Otsu's method on its VH histogram
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from .data import Chip, read_label, read_s1
from .metrics import Confusion


def predict_fixed(s1: np.ndarray, threshold_db: float) -> np.ndarray:
    return np.nan_to_num(s1[1], nan=0.0) < threshold_db


def predict_otsu(s1: np.ndarray) -> np.ndarray:
    vh = s1[1]
    good = vh[~np.isnan(vh)]
    if good.size < 100 or good.min() == good.max():
        return np.zeros(vh.shape, bool)
    return np.nan_to_num(vh, nan=0.0) < threshold_otsu(good)


def tune_fixed(chips: list[Chip], candidates=np.arange(-26, -12, 0.5)) -> float:
    data = [(read_s1(c.s1), read_label(c.label)) for c in chips]
    best = max(candidates, key=lambda t: sum((Confusion().update(predict_fixed(x, t), y) for x, y in data), Confusion()).scores()["iou"])
    return float(best)
