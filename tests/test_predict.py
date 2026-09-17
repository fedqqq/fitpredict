import json
import tempfile
import unittest
from pathlib import Path

import torch

from fitpredict import fit, predict


class OrderedFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[:, 0] * 10.0 + x[:, 1] + self.anchor * 0.0


class DictOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        score = x.reshape(x.shape[0], -1).sum(dim=1) + self.bias
        return {"score": score, "double": score * 2.0}


class WeightedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x.squeeze(-1) * self.weight


class ScalarOutputModel(torch.nn.Module):
    def forward(self, x):
        return x.sum()


def _config(data_path: Path, *, model_class: str, features=None, weights=None) -> dict:
    return {
        "data": {
            "path": str(data_path),
            "format": "json",
            "features": features or ["left", "right"],
            "targets": ["label"],
            "split": {
                "train": 1.0,
                "val": 0.0,
                "test": 0.0,
                "shuffle": False,
            },
        },
        "model": {
            "class": model_class,
            "weights": str(weights) if weights is not None else None,
            "inputs": {
                "x": {
                    "source": "features",
                    "dtype": "float32",
                }
            },
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "shuffle": True,
            "device": "cpu",
            "optimizer": {
                "name": "SGD",
                "params": {"lr": 0.1},
            },
            "objectives": [
                {
                    "loss": {"name": "MSELoss"},
                    "bindings": {
                        "input": {"source": "outputs", "dtype": "float32"},
                        "target": {"source": "targets.label", "dtype": "float32"},
                    },
                }
            ],
        },
        "evaluation": {"metrics": []},
        "logging": {"console": False},
        "saving": {"save_last": False},
    }


class PredictTests(unittest.TestCase):
    def test_predict_returns_tensor_without_target_columns_and_preserves_feature_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "predict.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"right": 2.0, "left": 1.0},
                        {"right": 4.0, "left": 3.0},
                    ]
                ),
                encoding="utf-8",
            )
            config = _config(
                data_path,
                model_class=f"{__name__}:OrderedFeatureModel",
                features=["left", "right"],
            )

            predictions = predict(config)

        self.assertIsInstance(predictions, torch.Tensor)
        self.assertFalse(predictions.requires_grad)
        self.assertEqual(predictions.device.type, "cpu")
        torch.testing.assert_close(predictions, torch.tensor([12.0, 34.0]))

    def test_predict_returns_dict_outputs_detached_on_cpu(self):
        rows = [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
            {"a": 5.0, "b": 6.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "unused.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = _config(
                data_path,
                model_class=f"{__name__}:DictOutputModel",
                features=["a", "b"],
            )

            predictions = predict(config, data=rows)

        self.assertEqual(set(predictions), {"score", "double"})
        for value in predictions.values():
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.device.type, "cpu")
        torch.testing.assert_close(predictions["score"], torch.tensor([3.0, 7.0, 11.0]))
        torch.testing.assert_close(predictions["double"], torch.tensor([6.0, 14.0, 22.0]))

    def test_predict_checkpoint_overrides_model_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "predict.json"
            data_path.write_text(
                json.dumps([{"x": 2.0}, {"x": 3.0}]),
                encoding="utf-8",
            )
            model_weights = temp_path / "model_weights.pt"
            checkpoint = temp_path / "checkpoint.pt"
            torch.save({"weight": torch.tensor([5.0])}, model_weights)
            torch.save({"model_state_dict": {"weight": torch.tensor([7.0])}}, checkpoint)
            config = _config(
                data_path,
                model_class=f"{__name__}:WeightedModel",
                features=["x"],
                weights=model_weights,
            )

            predictions = predict(config, checkpoint=checkpoint)

        torch.testing.assert_close(predictions, torch.tensor([14.0, 21.0]))

    def test_predict_runs_full_inference_lifecycle_from_trained_best_checkpoint(self):
        train_rows = [
            {"x": 1.0, "label": 2.0},
            {"x": 2.0, "label": 4.0},
        ]
        predict_rows = [
            {"x": 4.0},
            {"x": 5.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "train.json"
            output_dir = temp_path / "runs"
            data_path.write_text(json.dumps(train_rows), encoding="utf-8")
            config = _config(
                data_path,
                model_class=f"{__name__}:WeightedModel",
                features=["x"],
            )
            config["data"]["split"] = {
                "train": 0.5,
                "val": 0.5,
                "test": 0.0,
                "shuffle": False,
            }
            config["training"]["shuffle"] = False
            config["training"]["optimizer"] = {
                "name": "SGD",
                "params": {"lr": 0.5},
            }
            config["saving"] = {
                "save_last": True,
                "save_best": {"monitor": "val.loss", "mode": "min"},
                "output_dir": str(output_dir),
            }

            result = fit(config)
            predictions = predict(
                config,
                checkpoint=result.checkpoint_paths["best"],
                data=predict_rows,
            )

        torch.testing.assert_close(predictions, torch.tensor([8.0, 10.0]))

    def test_predict_empty_rows_return_none(self):
        rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "unused.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = _config(
                data_path,
                model_class=f"{__name__}:DictOutputModel",
                features=["a", "b"],
            )

            predictions = predict(config, data=rows)

        self.assertIsNone(predictions)

    def test_predict_rejects_scalar_tensor_outputs(self):
        rows = [{"a": 1.0, "b": 2.0}]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "unused.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = _config(
                data_path,
                model_class=f"{__name__}:ScalarOutputModel",
                features=["a", "b"],
            )

            with self.assertRaisesRegex(Exception, "leading batch dimension"):
                predict(config, data=rows)


if __name__ == "__main__":
    unittest.main()
