"""Train the flood model.

    python -m floodseg.train --name unet_r34 --epochs 60

Writes runs/<name>/: best.pt (by validation IoU), history.json, config.json.
The decision threshold is chosen on the validation split, never on test.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, model
from .metrics import Confusion

RUNS = Path(__file__).resolve().parent.parent / "runs"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(net: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Water probability for one full chip (3, H, W); H and W must be multiples of 32."""
    net.eval()
    t = torch.from_numpy(x)[None].to(device)
    with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
        return torch.sigmoid(net(t)).float()[0, 0].cpu().numpy()


@torch.no_grad()
def probabilities(net, ds: data.FloodChips, device: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(predict_proba(net, x, device), y) for x, y in ds.items]


def best_threshold(probs, candidates=np.arange(0.2, 0.81, 0.05)) -> tuple[float, float]:
    scored = [(sum((Confusion().update(p >= t, y) for p, y in probs), Confusion()).scores()["iou"], float(t)) for t in candidates]
    iou, t = max(scored)
    return round(t, 2), iou


def train(name: str, train_chips, valid_chips, epochs: int = 60, batch: int = 8, crop: int = 256, lr: float = 3e-4,
          encoder: str = "resnet34", seed: int = 0, device: str | None = None, quiet: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"name": name, "epochs": epochs, "batch": batch, "crop": crop, "lr": lr, "encoder": encoder, "seed": seed,
           "train_chips": len(train_chips), "valid_chips": len(valid_chips), "device": device}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    train_ds = data.FloodChips(train_chips, crop=crop, augment=True, seed=seed)
    valid_ds = data.FloodChips(valid_chips)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    net = model.build(encoder).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    history, best = [], {"iou": -1}
    for epoch in range(1, epochs + 1):
        net.train()
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                logits = net(x)
            loss = model.loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        probs = probabilities(net, valid_ds, device)
        valid = sum((Confusion().update(p >= 0.5, y) for p, y in probs), Confusion()).scores()
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "valid_iou@0.5": valid["iou"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if valid["iou"] > best["iou"]:
            best = {"iou": valid["iou"], "epoch": epoch}
            torch.save(net.state_dict(), out / "best.pt")
        if not quiet:
            print(f"[{name}] epoch {epoch:3d} loss {row['loss']:.4f} valid IoU {valid['iou']:.4f} ({row['seconds']}s)", flush=True)

    net.load_state_dict(torch.load(out / "best.pt", map_location=device))
    threshold, valid_iou = best_threshold(probabilities(net, valid_ds, device))
    summary = {**cfg, "best_epoch": best["epoch"], "threshold": threshold, "valid_iou": valid_iou}
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet_r34")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ipv4", action="store_true", help="resolve hosts to IPv4 only when downloading weights")
    a = ap.parse_args()
    if a.ipv4:
        from .net import ipv4_only
        ipv4_only()
    s = train(a.name, data.read_split("train"), data.read_split("valid"), a.epochs, a.batch, a.crop, a.lr, a.encoder, a.seed)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
