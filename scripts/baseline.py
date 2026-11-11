"""Score the threshold baselines on every split; writes reports/baseline.json."""

import json

import numpy as np

from floodseg import baseline, data
from floodseg.evaluate import REPORTS
from floodseg.metrics import Confusion

t = baseline.tune_fixed(data.read_split("train"))
res = {"fixed_threshold_db": t}
for split in ("valid", "test", "bolivia"):
    fx, ot = Confusion(), Confusion()
    for c in data.read_split(split):
        s1 = data.read_s1(c.s1)
        y = data.target(data.read_label(c.label), ~np.isnan(s1).any(0))
        fx.update(baseline.predict_fixed(s1, t), y)
        ot.update(baseline.predict_otsu(s1), y)
    res[split] = {"fixed": fx.scores(), "otsu": ot.scores()}
    print(f"{split:8s} fixed ({t} dB) IoU {fx.scores()['iou']:.3f}   Otsu IoU {ot.scores()['iou']:.3f}")
REPORTS.mkdir(exist_ok=True)
(REPORTS / "baseline.json").write_text(json.dumps(res, indent=1))
