import io
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodseg import baseline, data
from floodseg.evaluate import bootstrap_iou
from floodseg.metrics import Confusion


def test_confusion_ignores_unlabelled_pixels():
    y = np.array([[1, 0, -1], [1, 0, -1]])
    p = np.array([[1, 1, 1], [0, 0, 1]])
    s = Confusion().update(p, y).scores()
    assert s["pixels"] == 4
    assert s["iou"] == pytest.approx(1 / 3)          # tp 1, fp 1, fn 1
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_confusions_add_up():
    a = Confusion().update(np.ones((2, 2)), np.ones((2, 2), int))
    b = Confusion().update(np.zeros((2, 2)), np.ones((2, 2), int))
    assert (a + b).scores()["iou"] == 0.5


def test_features_handle_missing_pixels():
    s1 = np.full((2, 4, 4), -15.0, np.float32)
    s1[0, 0, 0] = np.nan
    x, valid = data.to_features(s1)
    assert x.shape == (3, 4, 4) and not valid[0, 0] and valid.sum() == 15
    assert np.all(x[:, 0, 0] == 0) and np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    y = data.target(np.ones((4, 4), int), valid)
    assert y[0, 0] == data.IGNORE


def test_baselines():
    s1 = np.full((2, 64, 64), -12.0, np.float32)
    s1[1, :32] = -28.0                                    # dark half = water
    assert baseline.predict_fixed(s1, -23.5)[:32].all() and not baseline.predict_fixed(s1, -23.5)[32:].any()
    assert baseline.predict_otsu(s1)[:32].all()
    assert not baseline.predict_otsu(np.full((2, 8, 8), np.nan, np.float32)).any()


def test_bootstrap_interval_contains_the_point_estimate():
    chips = [Confusion().update(np.random.default_rng(i).random((32, 32)) > 0.5, np.ones((32, 32), int)) for i in range(20)]
    lo, hi = bootstrap_iou(chips, n=500)
    total = sum(chips, Confusion()).scores()["iou"]
    assert lo <= total <= hi


def test_dataset_crops_and_augments(tmp_path):
    root = tmp_path
    (root / "S1Hand").mkdir(); (root / "LabelHand").mkdir(); (root / "splits").mkdir()
    prof = dict(driver="GTiff", width=64, height=64, crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4))
    with rasterio.open(root / "S1Hand" / "Spain_1_S1Hand.tif", "w", count=2, dtype="float32", **prof) as d:
        d.write(np.full((2, 64, 64), -15, np.float32))
    with rasterio.open(root / "LabelHand" / "Spain_1_LabelHand.tif", "w", count=1, dtype="int16", **prof) as d:
        d.write(np.ones((1, 64, 64), np.int16))
    (root / "splits" / "flood_train_data.csv").write_text("Spain_1_S1Hand.tif,Spain_1_LabelHand.tif\n")
    chips = data.read_split("train", root)
    assert chips[0].region == "Spain"
    x, y = data.FloodChips(chips, crop=32, augment=True)[0]
    assert tuple(x.shape) == (3, 32, 32) and tuple(y.shape) == (32, 32)


models = pytest.importorskip("onnxruntime")
MODEL = data.ROOT.parent / "models" / "unet_r34.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="export the model first")
def test_tiled_inference_matches_single_pass_and_handles_odd_sizes():
    from floodseg.infer import Predictor
    meta = json.loads(MODEL.with_suffix(".json").read_text())
    p = Predictor(MODEL, meta["threshold"])
    s1 = data.read_s1(data.read_split("test")[0].s1)
    big = np.concatenate([s1, s1[:, :, :300]], axis=2)       # 512 x 812: needs tiles and padding
    prob, valid = p.proba(big)
    assert prob.shape == (512, 812) and valid.shape == (512, 812)
    single, _ = p.proba(s1)
    assert np.mean((prob[:, :512] >= 0.5) == (single >= 0.5)) > 0.97
    m = p.mask(big)
    assert set(np.unique(m)) <= {0, 1, 255}


@pytest.mark.skipif(not MODEL.exists(), reason="export the model first")
def test_service_end_to_end():
    from fastapi.testclient import TestClient
    from floodseg.serve import app
    client = TestClient(app)
    assert client.get("/health").json()["ok"]
    chip = data.read_split("test")[0].s1
    with open(chip, "rb") as f:
        r = client.post("/predict", files={"file": ("x.tif", f.read())})
    body = r.json()
    assert r.status_code == 200 and 0 <= body["water_fraction"] <= 1 and body["observed_km2"] > 1
    with open(chip, "rb") as f:
        r = client.post("/predict?format=geotiff", files={"file": ("x.tif", f.read())})
    with rasterio.open(io.BytesIO(r.content)) as d:
        assert d.count == 1 and d.width == 512 and d.crs is not None
    bad = client.post("/predict", files={"file": ("x.tif", b"not a tiff")})
    assert bad.status_code == 422
