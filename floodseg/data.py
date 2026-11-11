"""Loading Sen1Floods11 hand-labelled chips.

Each chip is a 512 x 512 Sentinel-1 GRD image with two bands (VV, VH) in decibels,
and a label: 1 water, 0 not water, -1 not labelled. Unlabelled and missing
(NaN) pixels are excluded from training and from every metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent / "data"
IGNORE = -1

# Clipping ranges in dB, from the 1st and 99th percentiles of the training split (see README).
VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-35.0, -5.0)


@dataclass(frozen=True)
class Chip:
    s1: Path
    label: Path

    @property
    def region(self) -> str:
        return self.s1.name.split("_")[0]

    @property
    def name(self) -> str:
        return self.s1.name.removesuffix("_S1Hand.tif")


def read_split(split: str, root: Path = ROOT) -> list[Chip]:
    """split: train, valid, test or bolivia (the official CSVs)."""
    rows = pd.read_csv(root / "splits" / f"flood_{split}_data.csv", header=None).values
    return [Chip(root / "S1Hand" / s1, root / "LabelHand" / lab) for s1, lab in rows]


def all_chips(root: Path = ROOT) -> list[Chip]:
    return [c for s in ("train", "valid", "test", "bolivia") for c in read_split(s, root)]


def read_s1(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)  # (2, H, W), dB, NaN where missing


def read_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int64)


def to_features(s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale VV and VH to [0, 1] and add their difference (a cross-polarisation ratio in dB).

    Returns (features (3, H, W), valid mask (H, W)). Missing pixels become 0 and are marked invalid.
    """
    valid = ~np.isnan(s1).any(0)
    vv = np.clip(np.nan_to_num(s1[0], nan=VV_RANGE[0]), *VV_RANGE)
    vh = np.clip(np.nan_to_num(s1[1], nan=VH_RANGE[0]), *VH_RANGE)
    vv_n = (vv - VV_RANGE[0]) / (VV_RANGE[1] - VV_RANGE[0])
    vh_n = (vh - VH_RANGE[0]) / (VH_RANGE[1] - VH_RANGE[0])
    ratio = np.clip((vv - vh + 5) / 25, 0, 1)  # VV - VH typically spans about -5..20 dB
    feats = np.stack([vv_n, vh_n, ratio]).astype(np.float32)
    feats[:, ~valid] = 0
    return feats, valid


def target(label: np.ndarray, valid: np.ndarray) -> np.ndarray:
    y = label.copy()
    y[~valid] = IGNORE
    return y


class FloodChips(Dataset):
    """Chips in memory (the hand-labelled set is under 1 GB), with optional random crops and flips."""

    def __init__(self, chips: list[Chip], crop: int | None = None, augment: bool = False, seed: int = 0):
        self.chips, self.crop, self.augment = chips, crop, augment
        self.rng = np.random.default_rng(seed)
        self.items = []
        for c in chips:
            x, valid = to_features(read_s1(c.s1))
            self.items.append((x, target(read_label(c.label), valid)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        x, y = self.items[i]
        if self.crop:
            h, w = y.shape
            for _ in range(10):  # prefer crops with some labelled pixels
                r, c = self.rng.integers(0, h - self.crop + 1), self.rng.integers(0, w - self.crop + 1)
                yc = y[r:r + self.crop, c:c + self.crop]
                if (yc != IGNORE).mean() > 0.2:
                    break
            x, y = x[:, r:r + self.crop, c:c + self.crop], yc
        if self.augment:
            k = int(self.rng.integers(4))
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
