import unittest

import torch

from fitpredict.config import BindingConfig, BindingSource, ConfigError
from fitpredict.data import RuntimeSourceContext, SourceResolver, resolve_source


class SourceResolverTests(unittest.TestCase):
    def test_resolves_feature_target_and_feature_aggregate_sources(self):
        context = RuntimeSourceContext(
            features={
                "left": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                "right": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            },
            targets={"label": torch.tensor([1, 0])},
        )
        resolver = SourceResolver(context, features=("right", "left"), targets=("label",))

        torch.testing.assert_close(
            resolver.resolve("features.left"),
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        )
        torch.testing.assert_close(resolver.resolve("targets.label"), torch.tensor([1, 0]))
        torch.testing.assert_close(
            resolver.resolve("features"),
            torch.tensor(
                [[[10.0, 20.0], [1.0, 2.0]], [[30.0, 40.0], [3.0, 4.0]]],
                dtype=torch.float64,
            ),
        )

    def test_resolves_named_outputs_without_detaching_autograd(self):
        logits = torch.randn(2, 3, requires_grad=True)
        context = RuntimeSourceContext(features={}, targets={}, outputs={"logits": logits})

        resolved = resolve_source("outputs.logits", context)
        loss = resolved.square().sum()
        loss.backward()

        self.assertIs(resolved, logits)
        self.assertIsNotNone(logits.grad)

    def test_resolves_single_tensor_output_without_detaching_autograd(self):
        output = torch.randn(2, 1, requires_grad=True)
        context = RuntimeSourceContext(features={}, targets={}, outputs=output)

        resolved = resolve_source(BindingSource.parse("outputs"), context)
        resolved.sum().backward()

        self.assertIs(resolved, output)
        self.assertIsNotNone(output.grad)

    def test_supports_binding_config_dtype_conversion(self):
        output = torch.tensor([1.0, 2.0], requires_grad=True)
        binding = BindingConfig(source=BindingSource.parse("outputs"), dtype="float64")

        resolved = resolve_source(binding, RuntimeSourceContext(features={}, targets={}, outputs=output))
        resolved.sum().backward()

        self.assertEqual(resolved.dtype, torch.float64)
        self.assertIsNotNone(output.grad)

    def test_accepts_generic_batch_mapping_as_context(self):
        batch = {
            "features": {"x": torch.tensor([1.0, 2.0])},
            "targets": {"y": torch.tensor([0, 1])},
        }

        torch.testing.assert_close(resolve_source("features.x", batch), torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(resolve_source("targets.y", batch), torch.tensor([0, 1]))

    def test_rejects_unknown_declared_feature_source(self):
        context = RuntimeSourceContext(features={"x": torch.tensor([1])}, targets={})
        resolver = SourceResolver(context, features=("x",), targets=())

        with self.assertRaisesRegex(ConfigError, "not declared"):
            resolver.resolve("features.missing")

    def test_rejects_missing_runtime_feature_tensor(self):
        resolver = SourceResolver(
            RuntimeSourceContext(features={}, targets={}),
            features=("missing",),
            targets=(),
        )

        with self.assertRaisesRegex(ConfigError, "missing from runtime context"):
            resolver.resolve("features.missing")

    def test_rejects_named_outputs_when_model_returned_single_tensor(self):
        context = RuntimeSourceContext(features={}, targets={}, outputs=torch.ones(2, 1))

        with self.assertRaisesRegex(ConfigError, "requires model outputs to be a dictionary"):
            resolve_source("outputs.logits", context)

    def test_rejects_bare_outputs_when_model_returned_dictionary(self):
        context = RuntimeSourceContext(features={}, targets={}, outputs={"logits": torch.ones(2, 1)})

        with self.assertRaisesRegex(ConfigError, "use outputs.<name>"):
            resolve_source("outputs", context)

    def test_rejects_non_tensor_named_output_values(self):
        context = RuntimeSourceContext(features={}, targets={}, outputs={"logits": [1, 2]})

        with self.assertRaisesRegex(ConfigError, "must be a torch.Tensor"):
            resolve_source("outputs.logits", context)

    def test_runtime_context_validates_direct_dataclass_inputs(self):
        with self.assertRaisesRegex(ConfigError, "runtime_context.features.x must be a torch.Tensor"):
            RuntimeSourceContext(features={"x": 1}, targets={})


if __name__ == "__main__":
    unittest.main()
