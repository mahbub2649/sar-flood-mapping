# Model card: unet_r34 (flood water from Sentinel-1)

## What it does
Marks open surface water, including flood water, in Sentinel-1 radar images. Input: a GeoTIFF
with two bands, VV and VH backscatter in decibels (Sentinel-1 GRD, terrain-corrected, about
10 m pixels). Output: a mask (1 water, 0 not water, 255 no data) or water area in km².

## Intended use
Rapid first-pass flood extent mapping when clouds or darkness block optical satellites, for an
analyst to review. Not for decisions about individual properties, and not without a person
checking the result.

## Training data
Sen1Floods11 hand-labelled chips (Bonafilia et al., 2020, CC BY 4.0): 446 chips of 512 x 512
pixels from 11 flood events in 11 countries. Official split: 252 train, 89 validation, 90 test,
15 Bolivia (held out entirely). The decision threshold (0.45) was chosen on validation only.

## Model
U-Net with an ImageNet-pretrained ResNet-34 encoder (segmentation-models-pytorch). Inputs:
scaled VV, VH and VV minus VH. Loss: masked binary cross-entropy plus soft Dice. 60 epochs on
256-pixel random crops with flips and rotations, AdamW with a one-cycle schedule. About
12 minutes on a laptop RTX 3050 Ti. Exported to ONNX (98 MB) and checked against PyTorch.

## Results
See README. On the official test split the model reaches 0.676 water IoU against 0.534 for a
tuned threshold. On Bolivia, a country never seen in training, the gap is smaller and not
statistically clear (0.590 against 0.543, overlapping 95% intervals).

## Known failure modes
- Large, smooth water bodies are sometimes missed (seen in India and Bolivia chips).
- River banks and wet fields are over-predicted as water.
- Things that look dark to radar but are not water: tarmac, sand, radar shadow behind hills.
- Water hidden under trees or between buildings is invisible to C-band radar and is missed.
- Only trained on 11 events; arid regions and cities are under-represented.
- Trained on Sentinel-1 C-band. Other radars (for example X-band from ICEYE, Capella or
  Umbra) have different resolution and backscatter, so the model should not be used on them
  without fine-tuning and a fresh evaluation.

## Ethical considerations
Flood maps can steer aid and insurance payouts. False "dry" results in populated areas are the
costly error, which is why recall is reported next to IoU and the README shows where the model
misses water.
