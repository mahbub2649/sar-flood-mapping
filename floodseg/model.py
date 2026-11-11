"""The segmentation network: a U-Net with an ImageNet-pretrained ResNet-34 encoder."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from .data import IGNORE


def build(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def loss_fn(logits: torch.Tensor, y: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice on the water class, both ignoring unlabelled pixels.

    Dice counters the class imbalance (about 10% water); cross-entropy keeps gradients stable.
    """
    logits = logits.squeeze(1)
    mask = (y != IGNORE).float()
    t = (y == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, weight=mask, reduction="sum",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = bce / mask.sum().clamp(min=1)
    p = torch.sigmoid(logits) * mask
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * mask).sum() + 1)
    return bce + dice
