"""HTTP service: upload a Sentinel-1 VV/VH GeoTIFF, get water statistics or a water-mask GeoTIFF.

    uvicorn floodseg.serve:app --port 8100
    curl -F file=@scene.tif localhost:8100/predict
    curl -F file=@scene.tif "localhost:8100/predict?format=geotiff" -o mask.tif
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from rasterio.io import MemoryFile

from .infer import Predictor

MODELS = Path(__file__).resolve().parent.parent / "models"
MAX_BYTES = 200 * 2**20
MAX_PIXELS = 8192 * 8192
app = FastAPI(title="Flood mapping from Sentinel-1")


@lru_cache
def predictor() -> tuple[Predictor, dict]:
    run = os.environ.get("FLOODSEG_MODEL", "unet_r34")
    meta = json.loads((MODELS / f"{run}.json").read_text())
    return Predictor(MODELS / f"{run}.onnx", meta["threshold"]), meta


def pixel_area_km2(src) -> np.ndarray | float:
    """Area of each pixel. Geographic grids get a per-row area that shrinks towards the poles."""
    a, e = src.transform.a, src.transform.e
    if src.crs and src.crs.is_geographic:
        rows = np.arange(src.height)
        lat = src.transform.f + (rows + 0.5) * e
        km_per_deg = 111.32
        return (abs(a) * km_per_deg * np.cos(np.radians(lat)) * abs(e) * km_per_deg)[:, None]
    unit = src.crs.linear_units_factor[1] if src.crs else 1.0
    return abs(a * e) * unit**2 / 1e6


@app.get("/health")
def health():
    _, meta = predictor()
    return {"ok": True, "model": meta["run"], "sha256": meta["sha256"][:12], "threshold": meta["threshold"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), format: str = Query("json", pattern="^(json|geotiff)$")):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File larger than 200 MB")
    try:
        with MemoryFile(raw) as mem, mem.open() as src:
            if src.count != 2:
                raise HTTPException(422, f"Expected 2 bands (VV, VH in dB), got {src.count}")
            if src.width * src.height > MAX_PIXELS:
                raise HTTPException(413, "Image larger than 8192 x 8192 pixels")
            s1 = src.read().astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                s1[s1 == src.nodata] = np.nan
            profile, area = src.profile, pixel_area_km2(src)
    except rasterio.errors.RasterioIOError:
        raise HTTPException(422, "Not a readable GeoTIFF")

    if np.nanmedian(s1[0]) > 1:  # linear backscatter instead of dB
        raise HTTPException(422, "Bands look like linear backscatter; convert to dB (10 * log10) first")
    pred, meta = predictor()
    mask = pred.mask(s1)
    if format == "geotiff":
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        buf = io.BytesIO()
        with MemoryFile() as mem:
            with mem.open(**profile) as dst:
                dst.write(mask, 1)
            buf.write(mem.read())
        return Response(buf.getvalue(), media_type="image/tiff", headers={"Content-Disposition": "attachment; filename=water_mask.tif"})

    area_grid = np.broadcast_to(area, mask.shape)
    valid = mask != 255
    return {
        "model": meta["run"], "threshold": meta["threshold"],
        "pixels": int(mask.size), "valid_pixels": int(valid.sum()),
        "water_fraction": float((mask == 1).sum() / max(valid.sum(), 1)),
        "water_km2": round(float(area_grid[mask == 1].sum()), 3),
        "observed_km2": round(float(area_grid[valid].sum()), 3),
    }
