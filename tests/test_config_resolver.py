import unittest

from fitpredict.config import ConfigError, DataMetadata, FrozenDict, resolve_config


def valid_config():
    return {
        "data": {
            "path": "data/dataset.parquet",
            "format": "parquet",
            "features": ["f1", "f2", "f3"],
            "targets": ["target"],
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "model": {
            "class": "models.Model",
            "params": {
                "input_dim": "${data.num_features}",
                "epochs_label": "epochs-${training.epochs}",
                "steps": "${training.total_steps}",
            },
            "inputs": {"x": {"source": "features"}},
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "optimizer": {"name": "AdamW", "params": {"lr": 0.001}},
            "objectives": [
                {
                    "loss": {"name": "CrossEntropyLoss"},
                    "bindings": {
                        "input": {"source": "outputs.logits"},
                        "target": {"source": "targets.target"},
                    },
                }
            ],
        },
        "evaluation": {},
        "logging": {},
        "saving": {},
    }


class ConfigResolverTests(unittest.TestCase):
    def test_resolves_metadata_derived_values_and_references(self):
        config = resolve_config(
            valid_config(),
            data_metadata={
                "num_rows": 10,
                "columns": ["f1", "f2", "f3", "target"],
                "feature_dtypes": {"f1": "float32"},
            },
        )

        self.assertEqual(config.data.train_size, 6)
        self.assertEqual(config.data.val_size, 2)
        self.assertEqual(config.data.test_size, 2)
        self.assertEqual(config.training.steps_per_epoch, 2)
        self.assertEqual(config.training.total_steps, 6)
        self.assertEqual(config.model.params["input_dim"], 3)
        self.assertEqual(config.model.params["steps"], 6)
        self.assertEqual(config.model.params["epochs_label"], "epochs-3")
        self.assertEqual(config.data.metadata.feature_dtypes["f1"], "float32")

    def test_largest_remainder_split_rounding_keeps_total_rows(self):
        raw = valid_config()
        raw["data"]["split"] = {"train": 0.5, "val": 0.5, "test": 0.0}

        config = resolve_config(raw, data_metadata={"num_rows": 3, "columns": ["f1", "f2", "f3", "target"]})

        self.assertEqual(
            (config.data.train_size, config.data.val_size, config.data.test_size),
            (2, 1, 0),
        )

    def test_freezes_config_deeply(self):
        config = resolve_config(valid_config(), data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        self.assertIsInstance(config.model.params, FrozenDict)
        with self.assertRaises(TypeError):
            config.model.params["new"] = 1
        with self.assertRaises(AttributeError):
            config.training.objectives.append("extra")

    def test_rejects_unknown_reference(self):
        raw = valid_config()
        raw["model"]["params"]["bad"] = "${data.does_not_exist}"

        with self.assertRaisesRegex(ConfigError, "unknown config reference"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

    def test_rejects_reference_cycles(self):
        raw = valid_config()
        raw["model"]["params"]["a"] = "${model.params.b}"
        raw["model"]["params"]["b"] = "${model.params.a}"

        with self.assertRaisesRegex(ConfigError, "reference cycle detected"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

    def test_rejects_metadata_missing_configured_columns(self):
        with self.assertRaisesRegex(ConfigError, "missing configured feature"):
            resolve_config(valid_config(), data_metadata={"num_rows": 1, "columns": ["f1", "f2", "target"]})

    def test_requires_full_metadata_for_resolution(self):
        with self.assertRaisesRegex(ConfigError, "data_metadata.*required"):
            resolve_config(valid_config())

        with self.assertRaisesRegex(ConfigError, "data_metadata.columns is required"):
            resolve_config(valid_config(), data_metadata={"num_rows": 10})

    def test_rejects_non_public_and_method_references(self):
        raw = valid_config()
        raw["model"]["params"]["private"] = "${model.__class__}"

        with self.assertRaisesRegex(ConfigError, "non-public path segment"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        raw = valid_config()
        raw["model"]["params"]["method"] = "${model.from_mapping}"

        with self.assertRaisesRegex(ConfigError, "unknown config reference"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

    def test_rejects_invalid_type_after_reference_resolution(self):
        raw = valid_config()
        raw["model"]["class"] = "${training.epochs}"

        with self.assertRaisesRegex(ConfigError, "model.class"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

    def test_validates_typed_data_metadata(self):
        metadata = DataMetadata(columns=("f1", "f2", "f3", "target"), num_rows=-10)

        with self.assertRaisesRegex(ConfigError, "num_rows must be a non-negative integer"):
            resolve_config(valid_config(), data_metadata=metadata)

    def test_rejects_unknown_feature_and_target_bindings(self):
        raw = valid_config()
        raw["model"]["inputs"]["x"] = {"source": "features.missing"}

        with self.assertRaisesRegex(ConfigError, "unknown feature"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        raw = valid_config()
        raw["training"]["objectives"][0]["bindings"]["target"] = {"source": "targets.missing"}

        with self.assertRaisesRegex(ConfigError, "unknown target"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

    def test_resolver_derives_known_scheduler_step_on_from_helper(self):
        raw = valid_config()
        raw["training"]["scheduler"] = {"name": "StepLR", "params": {"step_size": 5}}

        config = resolve_config(
            raw,
            data_metadata={"num_rows": 8, "columns": ["f1", "f2", "f3", "target"]},
        )

        self.assertEqual(config.training.scheduler.step_on, "epoch")

    def test_validates_save_best_monitor(self):
        raw = valid_config()
        raw["saving"] = {"save_best": {"monitor": "test.accuracy", "mode": "max"}}

        with self.assertRaisesRegex(ConfigError, "cannot reference test"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        raw = valid_config()
        raw["evaluation"] = {"metrics": [{"name": "accuracy"}]}
        raw["saving"] = {"save_best": {"monitor": "val.unknown", "mode": "max"}}

        with self.assertRaisesRegex(ConfigError, "unknown validation metric"):
            resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        raw = valid_config()
        raw["evaluation"] = {"metrics": [{"name": "accuracy"}]}
        raw["saving"] = {"save_best": {"monitor": "val.accuracy", "mode": "max"}}

        config = resolve_config(raw, data_metadata={"num_rows": 1, "columns": ["f1", "f2", "f3", "target"]})

        self.assertEqual(config.saving.save_best.monitor, "val.accuracy")


if __name__ == "__main__":
    unittest.main()
