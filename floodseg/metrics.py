"""Segmentation metrics for the water class, accumulated over many images.

Scores are computed from summed counts over all pixels (not averaged per image),
which is how the Sen1Floods11 paper reports IoU. Ignored pixels never count.
"""

from __future__ import annotations

import numpy as np

from .data import IGNORE


class Confusion:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred: np.ndarray, y: np.ndarray) -> "Confusion":
        m = y != IGNORE
        p, t = pred[m].astype(bool), y[m] == 1
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())
        self.tn += int((~p & ~t).sum())
        return self

    def __add__(self, other: "Confusion") -> "Confusion":
        out = Confusion()
        for k in ("tp", "fp", "fn", "tn"):
            setattr(out, k, getattr(self, k) + getattr(other, k))
        return out

    def scores(self) -> dict:
        def div(a, b):
            return a / b if b else float("nan")

        precision, recall = div(self.tp, self.tp + self.fp), div(self.tp, self.tp + self.fn)
        return {
            "iou": div(self.tp, self.tp + self.fp + self.fn),
            "f1": div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": precision,
            "recall": recall,
            "accuracy": div(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn),
            "pixels": self.tp + self.tn + self.fp + self.fn,
            "water_fraction": div(self.tp + self.fn, self.tp + self.tn + self.fp + self.fn),
        }
