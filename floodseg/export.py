"""Export a trained run to ONNX and check it matches PyTorch.

    python -m floodseg.export --run unet_r34

Writes models/<run>.onnx and models/<run>.json (threshold, checksum, parity results).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data, model
from .train import RUNS

MODELS = Path(__file__).resolve().parent.parent / "models"


def export(run: str) -> dict:
    summary = json.loads((RUNS / run / "summary.json").read_text())
    net = model.build(summary["encoder"], pretrained=False)
    net.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    net.eval()
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"{run}.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, 512, 512), path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    diffs, agree, times = [], [], []
    for c in data.read_split("test")[:8]:
        x, _ = data.to_features(data.read_s1(c.s1))
        with torch.no_grad():
            ref = net(torch.from_numpy(x)[None]).numpy()
        t0 = time.perf_counter()
        got = sess.run(None, {"image": x[None]})[0]
        times.append(time.perf_counter() - t0)
        diffs.append(float(np.abs(ref - got).max()))
        th = np.log(summary["threshold"] / (1 - summary["threshold"]))
        agree.append(float(((ref >= th) == (got >= th)).mean()))
    meta = {
        "run": run, "threshold": summary["threshold"], "encoder": summary["encoder"], "input": "VV, VH in dB, see floodseg.data.to_features",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_mb": round(path.stat().st_size / 1e6, 1),
        "parity": {"max_abs_logit_diff": max(diffs), "min_mask_agreement": min(agree)},
        "cpu_ms_per_512_chip": round(1000 * float(np.median(times)), 1),
    }
    (MODELS / f"{run}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="unet_r34")
    print(json.dumps(export(ap.parse_args().run), indent=2))


if __name__ == "__main__":
    main()
