import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from fitpredict import FitResult, fit
from fitpredict.config import ExperimentConfig, load_config


class LinearRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.register_buffer("initial_weight", self.linear.weight.detach().clone())

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _write_json(path: Path) -> None:
    rows = [
        {"x1": 0.0, "x2": 0.0, "y": 0.0},
        {"x1": 1.0, "x2": 0.0, "y": 2.0},
        {"x1": 0.0, "x2": 1.0, "y": -1.0},
        {"x1": 1.0, "x2": 1.0, "y": 1.0},
        {"x1": 2.0, "x2": 1.0, "y": 3.0},
        {"x1": 1.0, "x2": 2.0, "y": 0.0},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _config(data_path: Path) -> dict:
    return {
        "data": {
            "path": str(data_path),
            "format": "json",
            "features": ["x1", "x2"],
            "targets": ["y"],
            "split": {
                "train": 1.0,
                "val": 0.0,
                "test": 0.0,
                "shuffle": False,
            },
        },
        "model": {
            "class": f"{__name__}:LinearRegressor",
            "params": {"input_dim": "${data.num_features}"},
            "inputs": {
                "x": {
                    "source": "features",
                    "dtype": "float32",
                }
            },
        },
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "shuffle": False,
            "device": "cpu",
            "optimizer": {
                "name": "SGD",
                "params": {"lr": 0.05},
            },
            "objectives": [
                {
                    "loss": {"name": "MSELoss"},
                    "bindings": {
                        "input": {"source": "outputs", "dtype": "float32"},
                        "target": {"source": "targets.y", "dtype": "float32"},
                    },
                    "weight": 1.0,
                }
            ],
        },
        "evaluation": {"metrics": []},
        "logging": {"console": False, "tensorboard": False, "mlflow": False},
        "saving": {"save_last": False},
    }


class FitTests(unittest.TestCase):
    def test_fit_trains_custom_model_from_json_path(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            config_path = Path(temp_dir) / "config.json"
            _write_json(data_path)
            config_path.write_text(json.dumps(_config(data_path)), encoding="utf-8")

            result = fit(config_path)

        self.assertIsInstance(result, FitResult)
        self.assertIsInstance(result.config, ExperimentConfig)
        self.assertEqual(len(result.history.train_loss), 3)
        self.assertEqual(len(result.history.batch_loss), 9)
        self.assertTrue(all(math.isfinite(loss) for loss in result.history.batch_loss))
        self.assertFalse(
            torch.equal(result.model.initial_weight, result.model.linear.weight.detach())
        )

    def test_fit_accepts_mapping_config(self):
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            _write_json(data_path)

            result = fit(_config(data_path))

        self.assertEqual(result.config.data.train_size, 6)
        self.assertTrue(math.isfinite(result.history.train_loss[-1]))

    def test_fit_accepts_experiment_config(self):
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            _write_json(data_path)
            config = load_config(_config(data_path))

            result = fit(config)

        self.assertIsInstance(result.config, ExperimentConfig)
        self.assertTrue(result.history.batch_loss)


if __name__ == "__main__":
    unittest.main()
