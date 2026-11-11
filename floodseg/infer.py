"""Inference on images of any size with an ONNX model, so serving needs no PyTorch.

Large images are cut into overlapping 512-pixel tiles; each output pixel is taken
from the tile where it lies furthest from the edge, which avoids seams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import to_features

TILE, OVERLAP = 512, 64


class Predictor:
    def __init__(self, onnx_path: Path, threshold: float):
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.threshold = threshold

    def _run(self, x: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {"image": x[None]})[0][0, 0]
        return 1 / (1 + np.exp(-logits))

    def proba(self, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """s1: (2, H, W) VV/VH in dB. Returns (water probability, valid mask)."""
        x, valid = to_features(s1)
        _, h, w = x.shape
        ph, pw = max(TILE, -(-h // 32) * 32), max(TILE, -(-w // 32) * 32)
        padded = np.zeros((3, ph, pw), np.float32)
        padded[:, :h, :w] = x
        if ph == TILE and pw == TILE:
            return self._run(padded)[:h, :w], valid
        out = np.zeros((ph, pw), np.float32)
        weight = np.zeros((ph, pw), np.float32)
        ramp = np.minimum(np.arange(TILE) + 1, np.arange(TILE)[::-1] + 1).astype(np.float32)
        win = np.minimum.outer(ramp, ramp)  # highest in the tile centre
        step = TILE - OVERLAP
        for r in sorted({*range(0, ph - TILE + 1, step), ph - TILE}):
            for c in sorted({*range(0, pw - TILE + 1, step), pw - TILE}):
                out[r:r + TILE, c:c + TILE] += self._run(np.ascontiguousarray(padded[:, r:r + TILE, c:c + TILE])) * win
                weight[r:r + TILE, c:c + TILE] += win
        return (out / weight)[:h, :w], valid

    def mask(self, s1: np.ndarray) -> np.ndarray:
        """uint8 mask: 1 water, 0 not water, 255 no data."""
        p, valid = self.proba(s1)
        m = (p >= self.threshold).astype(np.uint8)
        m[~valid] = 255
        return m
