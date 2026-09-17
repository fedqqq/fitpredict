from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitpredict import fit, predict


config = "examples/configs/regression.yaml"
result = fit(config)
predictions = predict(
    config, checkpoint=result.checkpoint_paths["best"], data="examples/data/predict.json"
)
print(predictions)
