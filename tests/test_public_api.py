import inspect
import unittest

import fitpredict
from fitpredict import (
    ConfigError,
    ExperimentConfig,
    fit,
    load_config,
    load_config_file,
    loads_config,
    predict,
    resolve_config,
)
from fitpredict.config.errors import ConfigError as ConfigErrorSource
from fitpredict.config.loader import load_config as load_config_source
from fitpredict.config.loader import load_config_file as load_config_file_source
from fitpredict.config.loader import loads_config as loads_config_source
from fitpredict.config.resolver import resolve_config as resolve_config_source
from fitpredict.config.schema import ExperimentConfig as ExperimentConfigSource
from fitpredict.prediction import predict as predict_source
from fitpredict.training import fit as fit_source


class PublicApiTests(unittest.TestCase):
    def test_root_public_api_is_explicit_and_stable(self):
        self.assertEqual(
            fitpredict.__all__,
            [
                "ConfigError",
                "ExperimentConfig",
                "FitHistory",
                "FitResult",
                "fit",
                "load_config",
                "load_config_file",
                "loads_config",
                "predict",
                "resolve_config",
            ],
        )

    def test_root_exports_user_entry_points(self):
        self.assertIs(fit, fit_source)
        self.assertIs(predict, predict_source)
        self.assertIs(load_config, load_config_source)
        self.assertIs(load_config_file, load_config_file_source)
        self.assertIs(loads_config, loads_config_source)
        self.assertIs(resolve_config, resolve_config_source)
        self.assertIs(ConfigError, ConfigErrorSource)
        self.assertIs(ExperimentConfig, ExperimentConfigSource)

    def test_star_import_exposes_only_public_names(self):
        namespace = {}
        exec("from fitpredict import *", namespace)

        exported = {name for name in namespace if not name.startswith("__")}
        self.assertEqual(exported, set(fitpredict.__all__))
        self.assertNotIn("ComponentResolver", exported)
        self.assertNotIn("BindingEngine", exported)
        self.assertNotIn("TabularDataset", exported)

    def test_fit_and_predict_signatures_are_small_user_contracts(self):
        self.assertEqual(tuple(inspect.signature(fit).parameters), ("config",))
        self.assertEqual(
            tuple(inspect.signature(predict).parameters),
            ("config", "checkpoint", "data"),
        )


if __name__ == "__main__":
    unittest.main()
