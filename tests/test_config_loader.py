import json
import tempfile
import unittest
from pathlib import Path

from fitpredict.config import ConfigError, ExperimentConfig, load_config, load_config_file, loads_config


def valid_config():
    return {
        "data": {
            "path": "dataset.parquet",
            "format": "parquet",
            "features": ["feature_1", "feature_2"],
            "targets": ["target"],
            "split": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1,
                "shuffle": True,
                "seed": 42,
            },
        },
        "model": {
            "class": "models.MyModel",
            "params": {"input_dim": "${data.num_features}"},
            "weights": "model.pt",
            "inputs": {"x": {"source": "features"}},
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "shuffle": True,
            "device": "cpu",
            "optimizer": {"name": "AdamW", "params": {"lr": 0.001}},
            "objectives": [
                {
                    "loss": {"name": "CrossEntropyLoss"},
                    "bindings": {
                        "input": {"source": "outputs.logits"},
                        "target": {"source": "targets.target"},
                    },
                    "weight": 1.0,
                }
            ],
        },
        "evaluation": {
            "metrics": [
                {
                    "name": "accuracy_score",
                    "bindings": {
                        "y_pred": {"source": "outputs.logits"},
                        "y_true": {"source": "targets.target"},
                    },
                }
            ]
        },
        "logging": {"console": True, "tensorboard": False, "mlflow": False},
        "saving": {"save_last": True, "save_best": {"monitor": "val.loss", "mode": "min"}},
    }


def valid_yaml_config():
    return """
data:
  path: dataset.parquet
  format: parquet
  features:
    - feature_1
    - feature_2
  targets:
    - target
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    shuffle: true
    seed: 42
model:
  class: models.MyModel
  params:
    input_dim: ${data.num_features}
  weights: model.pt
  inputs:
    x:
      source: features
training:
  epochs: 3
  batch_size: 16
  shuffle: true
  device: cpu
  optimizer:
    name: AdamW
    params:
      lr: 0.001
  objectives:
    - loss:
        name: CrossEntropyLoss
      bindings:
        input:
          source: outputs.logits
        target:
          source: targets.target
      weight: 1.0
evaluation:
  metrics:
    - name: accuracy_score
      bindings:
        y_pred:
          source: outputs.logits
        y_true:
          source: targets.target
logging:
  console: true
  tensorboard: false
  mlflow: false
saving:
  save_last: true
  save_best:
    monitor: val.loss
    mode: min
"""


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_json_into_typed_config(self):
        config = loads_config(json.dumps(valid_config()), format="json")

        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.model.class_path, "models.MyModel")
        self.assertEqual(config.training.optimizer.name, "AdamW")
        self.assertEqual(config.model.params["input_dim"], "${data.num_features}")

    def test_load_config_treats_json_string_as_content(self):
        config = load_config(json.dumps(valid_config()))

        self.assertEqual(config.model.class_path, "models.MyModel")

    def test_loads_mapping_into_typed_config(self):
        config = load_config(valid_config())

        self.assertEqual(config.data.features, ("feature_1", "feature_2"))
        self.assertEqual(config.training.objectives[0].bindings["input"].source.raw, "outputs.logits")

    def test_loads_json_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps(valid_config()), encoding="utf-8")

            config = load_config_file(path)

        self.assertEqual(config.saving.save_best.mode, "min")

    def test_missing_json_path_string_is_file_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"

            with self.assertRaisesRegex(ConfigError, "could not read config file"):
                load_config(str(missing))

    def test_missing_yaml_path_string_is_file_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.yaml"

            with self.assertRaisesRegex(ConfigError, "could not read config file"):
                load_config(str(missing))

    def test_relative_paths_are_kept_unresolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "config.json"
            path.parent.mkdir()
            path.write_text(json.dumps(valid_config()), encoding="utf-8")

            config = load_config_file(path)

        self.assertEqual(config.data.path, Path("dataset.parquet"))
        self.assertEqual(config.model.weights, Path("model.pt"))
        self.assertEqual(config.saving.output_dir, Path("runs"))

    def test_missing_section_is_diagnostic(self):
        raw = valid_config()
        del raw["logging"]

        with self.assertRaisesRegex(ConfigError, "config.logging section is required"):
            load_config(raw)

    def test_unknown_field_is_diagnostic(self):
        raw = valid_config()
        raw["training"]["learning_rate"] = 0.001

        with self.assertRaisesRegex(ConfigError, "training contains unknown field"):
            load_config(raw)

    def test_wrong_type_is_diagnostic(self):
        raw = valid_config()
        raw["training"]["epochs"] = "3"

        with self.assertRaisesRegex(ConfigError, "training.epochs must be an integer greater than 0"):
            load_config(raw)

    def test_malformed_json_is_diagnostic(self):
        with self.assertRaisesRegex(ConfigError, "malformed JSON"):
            loads_config("{", format="json")

    def test_duplicate_json_key_is_diagnostic(self):
        with self.assertRaisesRegex(ConfigError, "duplicate config key"):
            loads_config('{"data": {}, "data": {}}', format="json")

    def test_duplicate_json_key_is_diagnostic_in_auto_mode(self):
        with self.assertRaisesRegex(ConfigError, "duplicate config key"):
            load_config('{"data": {}, "data": {}}')

    def test_yaml_success_when_available_or_clear_error_when_unavailable(self):
        yaml_text = valid_yaml_config()
        try:
            import yaml  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(ConfigError, "PyYAML is not installed"):
                loads_config(yaml_text, format="yaml")
        else:
            config = loads_config(yaml_text, format="yaml")

            self.assertEqual(config.training.optimizer.name, "AdamW")

    def test_duplicate_yaml_key_is_diagnostic_when_yaml_available(self):
        try:
            import yaml  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(ConfigError, "PyYAML is not installed"):
                loads_config("data: {}\ndata: {}\n", format="yaml")
        else:
            with self.assertRaisesRegex(ConfigError, "duplicate config key 'data'"):
                loads_config("data: {}\ndata: {}\n", format="yaml")


if __name__ == "__main__":
    unittest.main()
