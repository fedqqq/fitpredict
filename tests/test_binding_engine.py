import unittest

import torch

from fitpredict.binding import BindingEngine, build_bound_kwargs
from fitpredict.config import BindingConfig, BindingSource, ConfigError
from fitpredict.data import RuntimeSourceContext


def binding(source: str, dtype: str | None = None) -> BindingConfig:
    return BindingConfig(source=BindingSource.parse(source), dtype=dtype)


class BindingEngineTests(unittest.TestCase):
    def test_builds_model_kwargs_from_feature_bindings(self):
        context = RuntimeSourceContext(
            features={
                "age": torch.tensor([20, 30], dtype=torch.int64),
                "height": torch.tensor([1.70, 1.80], dtype=torch.float32),
            },
            targets={},
        )
        engine = BindingEngine(context, features=("height", "age"), targets=())

        kwargs = engine.build_kwargs(
            {
                "x": binding("features", dtype="float32"),
                "age": binding("features.age"),
            },
            path="model.inputs",
        )

        self.assertEqual(set(kwargs), {"x", "age"})
        torch.testing.assert_close(
            kwargs["x"],
            torch.tensor([[1.70, 20.0], [1.80, 30.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(kwargs["age"], torch.tensor([20, 30], dtype=torch.int64))

    def test_builds_loss_kwargs_from_outputs_and_targets(self):
        logits = torch.randn(2, 3, requires_grad=True)
        labels = torch.tensor([1, 2])
        context = RuntimeSourceContext(
            features={},
            targets={"label": labels},
            outputs={"logits": logits},
        )

        kwargs = build_bound_kwargs(
            {
                "input": binding("outputs.logits"),
                "target": binding("targets.label"),
            },
            context,
            targets=("label",),
            path="training.objectives[0].bindings",
        )

        self.assertIs(kwargs["input"], logits)
        self.assertIs(kwargs["target"], labels)

    def test_preserves_autograd_for_single_output_bindings(self):
        output = torch.randn(2, 1, requires_grad=True)
        target = torch.ones(2, 1)
        context = RuntimeSourceContext(features={}, targets={"y": target}, outputs=output)
        engine = BindingEngine(context, targets=("y",))

        kwargs = engine.build_kwargs(
            {
                "prediction": binding("outputs"),
                "target": binding("targets.y"),
            }
        )
        loss = (kwargs["prediction"] - kwargs["target"]).square().mean()
        loss.backward()

        self.assertIs(kwargs["prediction"], output)
        self.assertIsNotNone(output.grad)

    def test_rejects_non_mapping_bindings(self):
        context = RuntimeSourceContext(features={}, targets={})
        engine = BindingEngine(context)

        with self.assertRaisesRegex(ConfigError, "bindings must be a mapping"):
            engine.build_kwargs([("x", binding("features.x"))])  # type: ignore[arg-type]

    def test_rejects_non_binding_values_with_argument_path(self):
        context = RuntimeSourceContext(features={}, targets={})
        engine = BindingEngine(context)

        with self.assertRaisesRegex(ConfigError, "metric.bindings.score must be a BindingConfig"):
            engine.build_kwargs({"score": "outputs"}, path="metric.bindings")  # type: ignore[dict-item]

    def test_source_errors_include_argument_path(self):
        context = RuntimeSourceContext(features={}, targets={}, outputs={"score": torch.tensor([1.0])})
        engine = BindingEngine(context)

        with self.assertRaisesRegex(ConfigError, 'metric.bindings.score source "outputs.missing"'):
            engine.build_kwargs({"score": binding("outputs.missing")}, path="metric.bindings")

    def test_builds_metric_kwargs_from_single_output_and_auxiliary_feature(self):
        output = torch.tensor([0.2, 0.8])
        segment = torch.tensor([1, 1])
        labels = torch.tensor([0, 1])
        context = RuntimeSourceContext(
            features={"segment": segment},
            targets={"label": labels},
            outputs=output,
        )

        kwargs = build_bound_kwargs(
            {
                "y_pred": binding("outputs"),
                "y_true": binding("targets.label"),
                "segment": binding("features.segment"),
            },
            context,
            features=("segment",),
            targets=("label",),
            path="evaluation.metrics[0].bindings",
        )

        self.assertIs(kwargs["y_pred"], output)
        self.assertIs(kwargs["y_true"], labels)
        self.assertIs(kwargs["segment"], segment)

    def test_rejects_single_output_binding_for_dict_output_model(self):
        context = RuntimeSourceContext(
            features={},
            targets={},
            outputs={"logits": torch.tensor([1.0])},
        )
        engine = BindingEngine(context)

        with self.assertRaisesRegex(ConfigError, "use outputs.<name>"):
            engine.build_kwargs({"prediction": binding("outputs")}, path="training.objectives[0].bindings")


if __name__ == "__main__":
    unittest.main()
