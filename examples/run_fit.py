from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitpredict import fit


parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="?", default="examples/configs/regression.yaml")
args = parser.parse_args()

result = fit(args.config)
print("train_loss", result.history.train_loss[-1])
print("checkpoints", {name: str(path) for name, path in result.checkpoint_paths.items()})
