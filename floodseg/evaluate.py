"""Evaluate a trained run against the baselines on the test split and on Bolivia.

    python -m floodseg.evaluate --run unet_r34

Writes reports/<run>/results.json, results.md and examples.png.
Confidence intervals come from a bootstrap over chips (resampling whole images,
because pixels inside one image are not independent).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import baseline, data, model
from .metrics import Confusion
from .train import RUNS, predict_proba

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def bootstrap_iou(per_chip: list[Confusion], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    counts = np.array([[c.tp, c.fp, c.fn] for c in per_chip], dtype=np.float64)
    idx = rng.integers(0, len(counts), (n, len(counts)))
    s = counts[idx].sum(1)
    iou = s[:, 0] / np.maximum(s.sum(1), 1)
    return float(np.percentile(iou, 2.5)), float(np.percentile(iou, 97.5))


def evaluate(run: str, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = json.loads((RUNS / run / "summary.json").read_text())
    net = model.build(summary["encoder"], pretrained=False).to(device)
    net.load_state_dict(torch.load(RUNS / run / "best.pt", map_location=device))
    t_model = summary["threshold"]
    t_fixed = json.loads((REPORTS / "baseline.json").read_text())["fixed_threshold_db"]

    results = {"run": run, "threshold": t_model, "fixed_threshold_db": t_fixed, "splits": {}}
    examples = []
    for split in ("test", "bolivia"):
        per = {"model": [], "fixed": [], "otsu": []}
        regions = defaultdict(lambda: {"model": Confusion(), "fixed": Confusion()})
        for c in data.read_split(split):
            s1 = data.read_s1(c.s1)
            x, valid = data.to_features(s1)
            y = data.target(data.read_label(c.label), valid)
            prob = predict_proba(net, x, device)
            preds = {"model": prob >= t_model, "fixed": baseline.predict_fixed(s1, t_fixed), "otsu": baseline.predict_otsu(s1)}
            for k, p in preds.items():
                per[k].append(Confusion().update(p, y))
            regions[c.region]["model"].update(preds["model"], y)
            regions[c.region]["fixed"].update(preds["fixed"], y)
            if (y == 1).mean() > 0.08:
                examples.append((split, c.name, s1, y, preds["model"], preds["fixed"]))
        out = {}
        for k, chips in per.items():
            total = sum(chips, Confusion())
            out[k] = {**total.scores(), "iou_95ci": bootstrap_iou(chips)}
        out["per_region"] = {r: {k: v.scores()["iou"] for k, v in d.items()} for r, d in sorted(regions.items())}
        results["splits"][split] = out

    run_dir = REPORTS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    (run_dir / "results.md").write_text(to_markdown(results))
    figure(examples, run_dir / "examples.png")
    return results


def to_markdown(r: dict) -> str:
    lines = [f"# Results: {r['run']}", "", "Water IoU (95% bootstrap interval over chips), F1, precision and recall.", "",
             "| Split | Method | IoU | 95% CI | F1 | Precision | Recall |", "|---|---|---|---|---|---|---|"]
    names = {"model": "U-Net (this repo)", "fixed": f"Fixed VH threshold ({r['fixed_threshold_db']} dB)", "otsu": "Otsu per image"}
    for split, s in r["splits"].items():
        for k in ("model", "fixed", "otsu"):
            m = s[k]
            lines.append(f"| {split} | {names[k]} | {m['iou']:.3f} | {m['iou_95ci'][0]:.3f}-{m['iou_95ci'][1]:.3f} | "
                         f"{m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
    lines += ["", "## IoU by region (test split)", "", "| Region | U-Net | Fixed threshold |", "|---|---|---|"]
    for region, v in r["splits"]["test"]["per_region"].items():
        lines.append(f"| {region} | {v['model']:.3f} | {v['fixed']:.3f} |")
    return "\n".join(lines) + "\n"


def figure(examples, path: Path, n: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    pick = [examples[i] for i in sorted(rng.choice(len(examples), min(n, len(examples)), replace=False))]
    fig, axes = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)), facecolor="black")
    colours = np.array([[0, 0, 0], [0.9, 0.9, 0.85], [1, 0.37, 0.12], [0.23, 0.5, 0.79]])  # TN, TP, FP, FN

    def overlay(pred, y):
        img = np.zeros(y.shape + (3,))
        img[(pred == 1) & (y == 1)] = colours[1]
        img[(pred == 1) & (y == 0)] = colours[2]
        img[(pred == 0) & (y == 1)] = colours[3]
        img[y == -1] = 0.15
        return img

    for row, (split, name, s1, y, pm, pf) in zip(np.atleast_2d(axes), pick):
        vh = np.clip(np.nan_to_num(s1[1], nan=-35), -35, -5)
        panels = [(vh, f"{name} ({split})\nSentinel-1 VH"), (np.where(y == -1, 0.15, y == 1), "Hand label"),
                  (overlay(pm, y), "U-Net"), (overlay(pf, y), "Fixed threshold")]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")
    fig.text(0.5, 0.005, "white: water found   orange: false alarm   blue: water missed   grey: not labelled",
             ha="center", color="white", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="unet_r34")
    r = evaluate(ap.parse_args().run)
    print(to_markdown(r))


if __name__ == "__main__":
    main()
