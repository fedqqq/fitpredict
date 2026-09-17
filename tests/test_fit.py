import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from fitpredict import FitResult, fit
from fitpredict.config import ConfigError, ExperimentConfig, load_config


class LinearRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.register_buffer("initial_weight", self.linear.weight.detach().clone())

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class EchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x.reshape(x.shape[0], -1).squeeze(-1) + self.anchor * 0.0


class DictLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        score = x.squeeze(-1) + self.anchor * 0.0
        return {
            "logits": torch.stack((score, -score), dim=-1),
            "aux": score + 100.0,
        }


class SingleWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x.squeeze(-1) * self.weight


class TracedSingleWeightModel(torch.nn.Module):
    val_calls = 0
    test_calls = 0

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        if bool((x == 2.0).all()):
            type(self).val_calls += 1
        if bool((x == 3.0).all()):
            type(self).test_calls += 1
        return x.squeeze(-1) * self.weight


class CountingBatchScheduler:
    calls = 0

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        type(self).calls += 1


class RecordingMetricScheduler:
    values = []

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, value):
        type(self).values.append(float(value))


class EvalSensitiveMSELoss(torch.nn.Module):
    def forward(self, input, target):
        penalty = 100.0 if self.training else 0.0
        return (input - target).square().mean() + penalty


def scaled_mse(input, target, scale=1.0):
    return (input - target).square().mean() * scale


def add_constant(input, value=0.0):
    return input + value


def tensor_mean(input):
    return input.float().mean()


def tensor_mean_raw(input):
    return input.float().mean()


def output_gap(left, right):
    selected = left[:, 0] if left.ndim > 1 else left
    return (selected - right).float().mean()


def subtract_outputs(left, right):
    return left[:, 0] - right


def explicit_dict_outputs(left, right):
    return {"a": left[:, 0], "b": right}


def exact_match(y_true, y_pred):
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(y_pred, torch.Tensor)
    assert y_true.device.type == "cpu"
    assert y_pred.device.type == "cpu"
    assert len(y_pred) == 2
    return (y_true == y_pred).float().mean()


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

    def test_fit_applies_multiple_weighted_objectives_to_gradient(self):
        rows = [{"x": 2.0, "y": 1.0}]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 1.0, "val": 0.0, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:SingleWeightModel",
                    "inputs": {"x": {"source": "features", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": f"{__name__}:scaled_mse", "params": {"scale": 1.0}},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                            "weight": 0.25,
                        },
                        {
                            "loss": {"name": f"{__name__}:scaled_mse", "params": {"scale": 2.0}},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                            "weight": 0.75,
                        },
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertAlmostEqual(float(result.model.weight.item()), 0.7, places=6)

    def test_fit_evaluates_dataset_metrics_with_sequential_local_transforms(self):
        rows = [
            {"x": 1.0, "y": 4.0},
            {"x": 2.0, "y": 5.0},
            {"x": 3.0, "y": 6.0},
            {"x": 4.0, "y": 7.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": f"{__name__}:scaled_mse"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "features.x", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:tensor_mean",
                            "transform": [
                                {"name": f"{__name__}:add_constant", "params": {"value": 10.0}},
                                {
                                    "name": f"{__name__}:add_constant",
                                    "params": {"value": 100.0},
                                    "bindings": {"input": {"source": "outputs"}},
                                },
                            ],
                        },
                        {
                            "name": f"{__name__}:tensor_mean_raw",
                            "bindings": {"input": {"source": "outputs"}},
                        },
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:tensor_mean"], 113.5)
        self.assertEqual(result.history.val_metrics[0][f"{__name__}:tensor_mean_raw"], 3.5)

    def test_fit_passes_full_cpu_dataset_to_metric_bindings(self):
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 2.0},
            {"x": 3.0, "y": 3.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "MSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "features.x", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:exact_match",
                            "bindings": {
                                "y_true": {"source": "targets.y", "dtype": "float32"},
                                "y_pred": {"source": "outputs", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:exact_match"], 1.0)

    def test_fit_applies_implicit_transform_to_named_metric_output(self):
        rows = [
            {"x": 1.0, "y": 0},
            {"x": -1.0, "y": 1},
            {"x": 2.0, "y": 0},
            {"x": -2.0, "y": 1},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:DictLogitsModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "CrossEntropyLoss"},
                            "bindings": {
                                "input": {"source": "outputs.logits", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "int64"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:exact_match",
                            "bindings": {
                                "y_true": {"source": "targets.y", "dtype": "int64"},
                                "y_pred": {"source": "outputs.logits", "dtype": "int64"},
                            },
                            "transform": [{"name": "argmax"}],
                        }
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:exact_match"], 1.0)

    def test_fit_allows_two_output_metric_without_transforms(self):
        rows = [
            {"x": 1.0, "y": 0},
            {"x": 2.0, "y": 0},
            {"x": 3.0, "y": 0},
            {"x": 4.0, "y": 0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:DictLogitsModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "CrossEntropyLoss"},
                            "bindings": {
                                "input": {"source": "outputs.logits", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "int64"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:output_gap",
                            "bindings": {
                                "left": {"source": "outputs.logits", "dtype": "float32"},
                                "right": {"source": "outputs.aux", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:output_gap"], -100.0)

    def test_fit_allows_explicit_transform_bindings_on_multiple_outputs(self):
        rows = [
            {"x": 1.0, "y": 0},
            {"x": 2.0, "y": 0},
            {"x": 3.0, "y": 0},
            {"x": 4.0, "y": 0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:DictLogitsModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "CrossEntropyLoss"},
                            "bindings": {
                                "input": {"source": "outputs.logits", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "int64"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:tensor_mean",
                            "bindings": {"input": {"source": "outputs", "dtype": "float32"}},
                            "transform": [
                                {
                                    "name": f"{__name__}:subtract_outputs",
                                    "bindings": {
                                        "left": {"source": "outputs.logits", "dtype": "float32"},
                                        "right": {"source": "outputs.aux", "dtype": "float32"},
                                    },
                                }
                            ],
                        }
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:tensor_mean"], -100.0)

    def test_fit_allows_explicit_dict_transform_with_two_output_metric_bindings(self):
        rows = [
            {"x": 1.0, "y": 0},
            {"x": 2.0, "y": 0},
            {"x": 3.0, "y": 0},
            {"x": 4.0, "y": 0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:DictLogitsModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "CrossEntropyLoss"},
                            "bindings": {
                                "input": {"source": "outputs.logits", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "int64"},
                            },
                        }
                    ],
                },
                "evaluation": {
                    "metrics": [
                        {
                            "name": f"{__name__}:output_gap",
                            "bindings": {
                                "left": {"source": "outputs.a", "dtype": "float32"},
                                "right": {"source": "outputs.b", "dtype": "float32"},
                            },
                            "transform": [
                                {
                                    "name": f"{__name__}:explicit_dict_outputs",
                                    "bindings": {
                                        "left": {"source": "outputs.logits", "dtype": "float32"},
                                        "right": {"source": "outputs.aux", "dtype": "float32"},
                                    },
                                }
                            ],
                        }
                    ],
                },
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_metrics[0][f"{__name__}:output_gap"], -100.0)

    def test_fit_weights_validation_loss_by_samples_for_uneven_batches(self):
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.0, "y": 1.0},
            {"x": 0.0, "y": 1.0},
            {"x": 0.0, "y": 10.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.25, "val": 0.75, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": "MSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_loss, [34.0])

    def test_fit_uses_eval_mode_for_loss_modules_during_validation(self):
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.0, "y": 0.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "objectives": [
                        {
                            "loss": {"name": f"{__name__}:EvalSensitiveMSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertEqual(result.history.val_loss, [0.0])

    def test_fit_steps_batch_and_metric_schedulers(self):
        CountingBatchScheduler.calls = 0
        RecordingMetricScheduler.values = []
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 2.0},
            {"x": 3.0, "y": 3.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 0.5, "val": 0.5, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
                    "scheduler": {
                        "name": f"{__name__}:CountingBatchScheduler",
                        "step_on": "batch",
                    },
                    "objectives": [
                        {
                            "loss": {"name": "MSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "features.x", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            fit(config)
            config["training"]["scheduler"] = {
                "name": f"{__name__}:RecordingMetricScheduler",
                "step_on": "metric",
                "monitor": "val.loss",
            }
            fit(config)

        self.assertEqual(CountingBatchScheduler.calls, 2)
        self.assertEqual(RecordingMetricScheduler.values, [0.0])

    def test_fit_steps_builtin_epoch_scheduler(self):
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {"train": 1.0, "val": 0.0, "test": 0.0, "shuffle": False},
                },
                "model": {
                    "class": f"{__name__}:EchoModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 0.2}},
                    "scheduler": {
                        "name": "StepLR",
                        "params": {"step_size": 1, "gamma": 0.5},
                    },
                    "objectives": [
                        {
                            "loss": {"name": "MSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {"save_last": False},
            }

            result = fit(config)

        self.assertAlmostEqual(result.optimizer.param_groups[0]["lr"], 0.05)

    def test_fit_runs_validation_checkpoints_and_final_test_from_best(self):
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            output_dir = Path(temp_dir) / "runs"
            _write_json(data_path)
            config = _config(data_path)
            config["data"]["split"] = {
                "train": 0.5,
                "val": 1 / 3,
                "test": 1 / 6,
                "shuffle": False,
            }
            config["saving"] = {
                "save_last": True,
                "save_best": {"monitor": "val.loss", "mode": "min"},
                "output_dir": str(output_dir),
            }

            result = fit(config)

            last_path = output_dir / "last.pt"
            best_path = output_dir / "best.pt"
            self.assertEqual(len(result.history.train_loss), 3)
            self.assertEqual(len(result.history.val_loss), 3)
            self.assertTrue(all(math.isfinite(loss) for loss in result.history.val_loss))
            self.assertIsNotNone(result.history.test_loss)
            self.assertTrue(math.isfinite(result.history.test_loss))
            self.assertEqual(result.checkpoint_paths["last"], last_path)
            self.assertEqual(result.checkpoint_paths["best"], best_path)
            self.assertTrue(last_path.exists())
            self.assertTrue(best_path.exists())

            last_checkpoint = torch.load(last_path, map_location="cpu")
            best_checkpoint = torch.load(best_path, map_location="cpu")
            self.assertIn("linear.weight", last_checkpoint)
            loaded_weight = best_checkpoint["linear.weight"]
            torch.testing.assert_close(result.model.linear.weight.detach(), loaded_weight)

    def test_fit_loads_early_best_once_before_final_test(self):
        TracedSingleWeightModel.val_calls = 0
        TracedSingleWeightModel.test_calls = 0
        rows = [
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 4.0},
            {"x": 3.0, "y": 6.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            output_dir = Path(temp_dir) / "runs"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            config = {
                "data": {
                    "path": str(data_path),
                    "format": "json",
                    "features": ["x"],
                    "targets": ["y"],
                    "split": {
                        "train": 1 / 3,
                        "val": 1 / 3,
                        "test": 1 / 3,
                        "shuffle": False,
                    },
                },
                "model": {
                    "class": f"{__name__}:TracedSingleWeightModel",
                    "inputs": {"x": {"source": "features.x", "dtype": "float32"}},
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 1,
                    "shuffle": False,
                    "device": "cpu",
                    "optimizer": {"name": "SGD", "params": {"lr": 1.0}},
                    "objectives": [
                        {
                            "loss": {"name": "MSELoss"},
                            "bindings": {
                                "input": {"source": "outputs", "dtype": "float32"},
                                "target": {"source": "targets.y", "dtype": "float32"},
                            },
                        }
                    ],
                },
                "evaluation": {"metrics": []},
                "logging": {"console": False, "tensorboard": False, "mlflow": False},
                "saving": {
                    "save_last": True,
                    "save_best": {"monitor": "val.loss", "mode": "min"},
                    "output_dir": str(output_dir),
                },
            }

            result = fit(config)

            best_weight = torch.load(output_dir / "best.pt", map_location="cpu")["weight"]
            last_weight = torch.load(output_dir / "last.pt", map_location="cpu")["weight"]

        self.assertEqual(TracedSingleWeightModel.val_calls, 2)
        self.assertEqual(TracedSingleWeightModel.test_calls, 1)
        torch.testing.assert_close(best_weight, torch.tensor([2.0]))
        torch.testing.assert_close(last_weight, torch.tensor([0.0]))
        torch.testing.assert_close(result.model.weight.detach(), best_weight)
        self.assertEqual(result.history.val_loss, [0.0, 16.0])
        self.assertEqual(result.history.test_loss, 0.0)

    def test_fit_rejects_test_split_without_best_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.json"
            _write_json(data_path)
            config = _config(data_path)
            config["data"]["split"] = {
                "train": 0.5,
                "val": 1 / 3,
                "test": 1 / 6,
                "shuffle": False,
            }

            with self.assertRaisesRegex(ConfigError, "test split requires saving.save_best"):
                fit(config)


if __name__ == "__main__":
    unittest.main()
