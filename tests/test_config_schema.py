import unittest

from fitpredict.config.schema import BindingSource, ConfigError, ExperimentConfig


def minimal_config():
    return {
        "data": {
            "path": "data/train.parquet",
            "format": "parquet",
            "features": ["age", "income", "score"],
            "targets": ["label"],
            "split": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1,
                "shuffle": True,
                "seed": 42,
            },
        },
        "model": {
            "class": "models.Classifier",
            "params": {"input_dim": "${data.num_features}"},
            "inputs": {"x": {"source": "features"}},
        },
        "training": {
            "epochs": 20,
            "batch_size": 128,
            "optimizer": {"name": "AdamW", "params": {"lr": 0.001}},
            "objectives": [
                {
                    "loss": {"name": "CrossEntropyLoss"},
                    "bindings": {
                        "input": {"source": "outputs.logits"},
                        "target": {"source": "targets.label"},
                    },
                }
            ],
        },
        "evaluation": {},
        "logging": {},
        "saving": {},
    }


class ConfigSchemaTests(unittest.TestCase):
    def test_parses_minimal_config_and_applies_defaults(self):
        config = ExperimentConfig.from_mapping(minimal_config())

        self.assertEqual(config.data.num_features, 3)
        self.assertEqual(config.data.num_targets, 1)
        self.assertEqual(config.data.features, ("age", "income", "score"))
        self.assertTrue(config.training.shuffle)
        self.assertEqual(config.training.device, "cpu")
        self.assertTrue(config.logging.console)
        self.assertFalse(config.logging.tensorboard)
        self.assertTrue(config.saving.save_last)
        self.assertEqual(config.saving.output_dir.name, "runs")

    def test_binding_source_protocol(self):
        cases = [
            ("features", "features", None, True, False),
            ("features.history", "features", "history", False, False),
            ("targets.label", "targets", "label", False, False),
            ("outputs.logits", "outputs", "logits", False, False),
            ("outputs", "outputs", None, False, True),
        ]

        for raw, namespace, name, is_feature_aggregate, is_single_output in cases:
            with self.subTest(raw=raw):
                source = BindingSource.parse(raw)

                self.assertEqual(source.namespace, namespace)
                self.assertEqual(source.name, name)
                self.assertIs(source.is_feature_aggregate, is_feature_aggregate)
                self.assertIs(source.is_single_output, is_single_output)

    def test_rejects_multiple_sources_in_one_binding(self):
        with self.assertRaisesRegex(ConfigError, "exactly one source"):
            BindingSource.parse("features.a,features.b")

    def test_rejects_invalid_split_sum(self):
        raw = minimal_config()
        raw["data"]["split"]["test"] = 0.2

        with self.assertRaisesRegex(ConfigError, "sum to 1"):
            ExperimentConfig.from_mapping(raw)

    def test_scheduler_metric_requires_monitor(self):
        raw = minimal_config()
        raw["training"]["scheduler"] = {
            "name": "ReduceLROnPlateau",
            "step_on": "metric",
        }

        with self.assertRaisesRegex(ConfigError, "monitor is required"):
            ExperimentConfig.from_mapping(raw)

    def test_known_scheduler_derives_step_on_and_tracks_explicit_flag(self):
        raw = minimal_config()
        raw["training"]["scheduler"] = {
            "name": "OneCycleLR",
            "params": {"max_lr": 0.1},
        }

        config = ExperimentConfig.from_mapping(raw)

        self.assertEqual(config.training.scheduler.step_on, "batch")
        self.assertFalse(config.training.scheduler.step_on_was_explicit)

    def test_custom_scheduler_requires_explicit_step_on(self):
        raw = minimal_config()
        raw["training"]["scheduler"] = {"name": "schedulers.MyScheduler"}

        with self.assertRaisesRegex(ConfigError, "step_on is required"):
            ExperimentConfig.from_mapping(raw)

    def test_known_scheduler_rejects_conflicting_step_on(self):
        raw = minimal_config()
        raw["training"]["scheduler"] = {
            "name": "CyclicLR",
            "step_on": "epoch",
        }

        with self.assertRaisesRegex(ConfigError, "must be batch"):
            ExperimentConfig.from_mapping(raw)

    def test_allows_feature_target_overlap(self):
        raw = minimal_config()
        raw["data"]["features"] = ["label", "score"]
        raw["data"]["targets"] = ["label"]

        config = ExperimentConfig.from_mapping(raw)

        self.assertEqual(config.data.features, ("label", "score"))
        self.assertEqual(config.data.targets, ("label",))

    def test_rejects_user_supplied_reserved_derived_fields(self):
        raw = minimal_config()
        raw["data"]["train_size"] = 100

        with self.assertRaisesRegex(ConfigError, "data.train_size"):
            ExperimentConfig.from_mapping(raw)

    def test_optional_fields_can_be_explicitly_disabled(self):
        raw = minimal_config()
        raw["logging"] = {"console": False, "tensorboard": False, "mlflow": False}
        raw["saving"] = {"save_last": False}

        config = ExperimentConfig.from_mapping(raw)

        self.assertFalse(config.logging.console)
        self.assertFalse(config.saving.save_last)


if __name__ == "__main__":
    unittest.main()
