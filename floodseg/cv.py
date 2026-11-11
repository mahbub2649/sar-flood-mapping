"""Leave-one-region-out evaluation: how well does the model work in a country it has never seen?

The official test split contains chips from the same ten flood events as the
training split, sometimes from neighbouring areas, so it measures in-distribution
performance. Here each region in turn is held out completely: the model is trained
on the other nine regions (85% for training, 15% for validation and threshold
choice) and scored on every hand-labelled chip of the held-out region.

    python -m floodseg.cv --epochs 30
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import baseline, data, model
from .evaluate import REPORTS
from .metrics import Confusion
from .train import RUNS, predict_proba, train


def run(epochs: int, seed: int = 0) -> dict:
    chips = [c for c in data.all_chips() if c.region != "Bolivia"]
    regions = sorted({c.region for c in chips})
    out = {"epochs": epochs, "regions": {}}
    for region in regions:
        rest = [c for c in chips if c.region != region]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rest))
        n_val = int(0.15 * len(rest))
        valid = [rest[i] for i in order[:n_val]]
        train_chips = [rest[i] for i in order[n_val:]]
        held = [c for c in chips if c.region == region]

        summary = train(f"cv_{region}", train_chips, valid, epochs=epochs, seed=seed, quiet=True)
        t_fixed = baseline.tune_fixed(train_chips)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = model.build(pretrained=False).to(device)
        net.load_state_dict(torch.load(RUNS / f"cv_{region}" / "best.pt", map_location=device))
        m, f = Confusion(), Confusion()
        for c in held:
            s1 = data.read_s1(c.s1)
            x, ok = data.to_features(s1)
            y = data.target(data.read_label(c.label), ok)
            m.update(predict_proba(net, x, device) >= summary["threshold"], y)
            f.update(baseline.predict_fixed(s1, t_fixed), y)
        out["regions"][region] = {"chips": len(held), "unet_iou": m.scores()["iou"], "fixed_iou": f.scores()["iou"],
                                  "water_fraction": m.scores()["water_fraction"], "threshold": summary["threshold"]}
        print(f"{region:10s} chips {len(held):3d}  U-Net IoU {m.scores()['iou']:.3f}  fixed IoU {f.scores()['iou']:.3f}", flush=True)
        (REPORTS / "cv.json").write_text(json.dumps(out, indent=1))
    u = [v["unet_iou"] for v in out["regions"].values()]
    fx = [v["fixed_iou"] for v in out["regions"].values()]
    out["mean_unet_iou"], out["mean_fixed_iou"] = float(np.mean(u)), float(np.mean(fx))
    out["regions_where_unet_wins"] = int(sum(a > b for a, b in zip(u, fx)))
    (REPORTS / "cv.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    r = run(ap.parse_args().epochs)
    print(f"mean IoU over regions: U-Net {r['mean_unet_iou']:.3f}, fixed threshold {r['mean_fixed_iou']:.3f}; "
          f"U-Net better in {r['regions_where_unet_wins']} of {len(r['regions'])} regions")


if __name__ == "__main__":
    main()
