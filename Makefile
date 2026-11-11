PY := .venv/bin/python
RUN ?= unet_r34

.PHONY: setup data baseline train evaluate cv export serve test all

setup:          ## Create the environment (GPU build of PyTorch)
	python3 -m venv .venv
	.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	.venv/bin/pip install -r requirements.txt

data:           ## Download the hand-labelled Sen1Floods11 chips (~730 MB)
	python3 -I scripts/download_data.py

baseline:       ## Score the threshold baselines
	$(PY) -m scripts.baseline

train:          ## Train the U-Net (~12 min on an RTX 3050 Ti)
	$(PY) -m floodseg.train --name $(RUN) --epochs 60 --ipv4

evaluate:       ## Test split + Bolivia, with bootstrap intervals and figures
	$(PY) -m floodseg.evaluate --run $(RUN)

cv:             ## Leave-one-region-out (10 trainings, ~1 h)
	$(PY) -m floodseg.cv --epochs 30

export:         ## ONNX export with a parity check against PyTorch
	$(PY) -m floodseg.export --run $(RUN)

serve:          ## HTTP API on port 8100
	FLOODSEG_MODEL=$(RUN) .venv/bin/uvicorn floodseg.serve:app --port 8100

test:
	$(PY) -m pytest -q tests

all: data baseline train evaluate export test
