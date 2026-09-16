import sys
import types
import unittest

from fitpredict.config import (
    BuiltinComponent,
    ComponentResolver,
    ConfigError,
    resolve_builtin_scheduler_step_on,
)
from fitpredict.config.schema import ComponentConfig


class CustomComponent:
    def __init__(self, value=0):
        self.value = value


def custom_function():
    return "custom"


def fake_accuracy_score(y_true, y_pred):
    return sum(left == right for left, right in zip(y_true, y_pred)) / len(y_true)


class ComponentResolverTests(unittest.TestCase):
    def setUp(self):
        self.module_name = "fitpredict_test_components"
        self.module = types.ModuleType(self.module_name)
        self.module.CustomComponent = CustomComponent
        self.module.custom_function = custom_function
        self.module.Nested = types.SimpleNamespace(Component=CustomComponent)
        sys.modules[self.module_name] = self.module
        self.original_sklearn = sys.modules.get("sklearn")
        self.original_sklearn_metrics = sys.modules.get("sklearn.metrics")

    def tearDown(self):
        sys.modules.pop(self.module_name, None)
        if self.original_sklearn is None:
            sys.modules.pop("sklearn", None)
        else:
            sys.modules["sklearn"] = self.original_sklearn
        if self.original_sklearn_metrics is None:
            sys.modules.pop("sklearn.metrics", None)
        else:
            sys.modules["sklearn.metrics"] = self.original_sklearn_metrics

    def test_resolves_builtin_short_name_from_explicit_registry(self):
        resolver = ComponentResolver(
            builtins={
                "loss": {
                    "FakeLoss": BuiltinComponent(
                        self.module_name, "CustomComponent", "FakeFramework"
                    )
                }
            }
        )

        resolution = resolver.resolve("loss", "FakeLoss")

        self.assertIs(resolution.component, CustomComponent)
        self.assertEqual(resolution.source, "built-in")
        self.assertEqual(resolution.framework, "FakeFramework")
        self.assertEqual(resolution.module, self.module_name)
        self.assertEqual(resolution.attr, "CustomComponent")

    def test_builtin_missing_framework_has_contextual_error(self):
        resolver = ComponentResolver(
            builtins={
                "optimizer": {
                    "FakeAdam": BuiltinComponent(
                        "fitpredict_missing_framework.optim", "FakeAdam", "FakeFramework"
                    )
                }
            }
        )

        with self.assertRaisesRegex(ConfigError, "FakeFramework is not installed"):
            resolver.resolve("optimizer", "FakeAdam")

    def test_rejects_unknown_builtin_for_category(self):
        resolver = ComponentResolver(builtins={"loss": {}})

        with self.assertRaisesRegex(ConfigError, "unknown built-in loss component"):
            resolver.resolve("loss", "NotRegistered")

    def test_resolves_accuracy_score_metric_builtin(self):
        sklearn = types.ModuleType("sklearn")
        sklearn.__path__ = []
        metrics = types.ModuleType("sklearn.metrics")
        metrics.accuracy_score = fake_accuracy_score
        sklearn.metrics = metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = metrics

        resolution = ComponentResolver().resolve("metric", "accuracy_score")

        self.assertIs(resolution.component, fake_accuracy_score)
        self.assertEqual(resolution.framework, "scikit-learn")

    def test_resolves_custom_colon_import_path(self):
        resolver = ComponentResolver()

        resolution = resolver.resolve("model", f"{self.module_name}:CustomComponent")

        self.assertIs(resolution.component, CustomComponent)
        self.assertEqual(resolution.source, "custom")
        self.assertEqual(resolution.module, self.module_name)
        self.assertEqual(resolution.attr, "CustomComponent")

    def test_resolves_custom_dotted_import_path_with_nested_attribute(self):
        resolver = ComponentResolver()

        resolution = resolver.resolve("transform", f"{self.module_name}.Nested.Component")

        self.assertIs(resolution.component, CustomComponent)
        self.assertEqual(resolution.source, "custom")
        self.assertEqual(resolution.module, self.module_name)
        self.assertEqual(resolution.attr, "Nested.Component")

    def test_custom_import_reports_missing_attribute(self):
        resolver = ComponentResolver()

        with self.assertRaisesRegex(ConfigError, "has no attribute"):
            resolver.resolve("metric", f"{self.module_name}:Missing")

    def test_custom_import_rejects_non_callable_component(self):
        resolver = ComponentResolver()

        with self.assertRaisesRegex(ConfigError, "is not callable"):
            resolver.resolve("metric", "math.pi")

    def test_resolve_accepts_component_config_and_instantiate_uses_params(self):
        resolver = ComponentResolver()
        spec = ComponentConfig(
            name=f"{self.module_name}:CustomComponent",
            params={"value": 3},
        )

        resolution = resolver.resolve("loss", spec)
        instance = resolver.instantiate(resolution)

        self.assertIsInstance(instance, CustomComponent)
        self.assertEqual(instance.value, 3)

    def test_pure_scheduler_step_on_helper_does_not_import_component(self):
        self.assertEqual(resolve_builtin_scheduler_step_on("StepLR"), "epoch")
        self.assertEqual(resolve_builtin_scheduler_step_on("OneCycleLR"), "batch")
        self.assertEqual(resolve_builtin_scheduler_step_on("ReduceLROnPlateau"), "metric")
        self.assertIsNone(resolve_builtin_scheduler_step_on("custom.schedulers.MyScheduler"))

    def test_argmax_transform_defaults_to_last_axis_for_matrix_input(self):
        resolution = ComponentResolver().resolve("transform", "argmax")

        self.assertEqual(resolution.component([[0.1, 0.7, 0.2], [9, 1, 4]]), [1, 0])

    def test_threshold_transform_uses_value_param(self):
        resolution = ComponentResolver().resolve("transform", "threshold")

        self.assertEqual(
            resolution.component(input=[0.2, 0.7, 0.5], value=0.5),
            [False, True, True],
        )


if __name__ == "__main__":
    unittest.main()
