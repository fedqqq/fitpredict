import unittest

import torch

from fitpredict.config import BindingConfig, BindingSource, ConfigError, DataConfig
from fitpredict.data import (
    aggregate_feature_tensors,
    tensorize_binding,
    tensorize_features,
    tensorize_source,
)


class DataTensorizerTests(unittest.TestCase):
    def test_single_sources_follow_batch_shape_contract(self):
        rows = [
            {"scalar": 1, "vector": [1.0, 2.0], "matrix": [[1, 2], [3, 4]], "label": True},
            {"scalar": 2, "vector": [3.0, 4.0], "matrix": [[5, 6], [7, 8]], "label": False},
        ]

        scalar = tensorize_source(rows, "features.scalar", features=("scalar",))
        vector = tensorize_source(rows, "features.vector", features=("vector",))
        matrix = tensorize_source(rows, "features.matrix", features=("matrix",))
        target = tensorize_source(rows, "targets.label", targets=("label",))

        self.assertEqual(tuple(scalar.tensor.shape), (2,))
        self.assertEqual(scalar.tensor.dtype, torch.int64)
        self.assertEqual(tuple(vector.tensor.shape), (2, 2))
        self.assertEqual(vector.tensor.dtype, torch.float32)
        self.assertEqual(tuple(matrix.tensor.shape), (2, 2, 2))
        self.assertEqual(matrix.tensor.dtype, torch.int64)
        self.assertEqual(tuple(target.tensor.shape), (2,))
        self.assertEqual(target.tensor.dtype, torch.bool)

    def test_feature_aggregate_stacks_declared_feature_order_on_dim_one(self):
        rows = [
            {"left": [1.0, 2.0], "right": [10.0, 20.0]},
            {"left": [3.0, 4.0], "right": [30.0, 40.0]},
        ]

        result = tensorize_source(
            rows,
            BindingSource.parse("features"),
            features=("right", "left"),
            targets=(),
        )

        self.assertEqual(result.columns, ("right", "left"))
        self.assertEqual(tuple(result.tensor.shape), (2, 2, 2))
        torch.testing.assert_close(
            result.tensor,
            torch.tensor([[[10.0, 20.0], [1.0, 2.0]], [[30.0, 40.0], [3.0, 4.0]]]),
        )

    def test_feature_aggregate_promotes_mixed_numeric_to_float64(self):
        rows = [{"age": 42, "income": 1000.5}, {"age": 30, "income": 850.0}]

        tensor = tensorize_features(rows, ("age", "income"))

        self.assertEqual(tuple(tensor.shape), (2, 2))
        self.assertEqual(tensor.dtype, torch.float64)

    def test_explicit_dtype_takes_priority(self):
        rows = [{"age": 42, "income": 1000.5}, {"age": 30, "income": 850.0}]
        binding = BindingConfig(source=BindingSource.parse("features"), dtype="float16")

        result = tensorize_binding(
            rows,
            binding,
            data=DataConfig.from_mapping(
                {
                    "path": "unused.jsonl",
                    "format": "jsonl",
                    "features": ["age", "income"],
                    "targets": ["label"],
                    "split": {"train": 1.0, "val": 0.0, "test": 0.0},
                }
            ),
        )

        self.assertEqual(result.tensor.dtype, torch.float16)

    def test_rejects_unsupported_explicit_dtype(self):
        with self.assertRaisesRegex(ConfigError, "dtype is unsupported"):
            BindingConfig.from_mapping(
                {"source": "features.age", "dtype": "float128"},
                "model.inputs.x",
            )

    def test_rejects_empty_row_batch_directly(self):
        with self.assertRaisesRegex(ConfigError, "row batch is empty"):
            tensorize_source([], "features.age", features=("age",), targets=())

    def test_aggregate_helper_accepts_already_batched_feature_tensors(self):
        tensors = {
            "left": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "right": torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32),
        }

        result = aggregate_feature_tensors(tensors, ("right", "left"))

        self.assertEqual(tuple(result.shape), (2, 2, 2))
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(
            result,
            torch.tensor(
                [[[1.5, 2.5], [1.0, 2.0]], [[3.5, 4.5], [3.0, 4.0]]],
                dtype=torch.float64,
            ),
        )

    def test_aggregate_helper_rejects_mismatched_batched_shapes(self):
        tensors = {
            "left": torch.ones((2, 2)),
            "right": torch.ones((2, 3)),
        }

        with self.assertRaisesRegex(ConfigError, "feature sample shapes differ"):
            aggregate_feature_tensors(tensors, ("left", "right"))

    def test_rejects_ragged_sample_shapes_without_padding(self):
        rows = [{"history": [1, 2]}, {"history": [3, 4, 5]}]

        with self.assertRaisesRegex(ConfigError, "automatic padding"):
            tensorize_source(rows, "features.history", features=("history",), targets=())

    def test_rejects_nested_ragged_values_without_padding(self):
        rows = [{"history": [[1], [2, 3]]}, {"history": [[4], [5, 6]]}]

        with self.assertRaisesRegex(ConfigError, "automatic padding"):
            tensorize_source(rows, "features.history", features=("history",), targets=())

    def test_rejects_incompatible_aggregate_shapes(self):
        rows = [{"scalar": 1, "vector": [2, 3]}, {"scalar": 4, "vector": [5, 6]}]

        with self.assertRaisesRegex(ConfigError, "feature sample shapes differ"):
            tensorize_source(rows, "features", features=("scalar", "vector"), targets=())

    def test_rejects_strings_as_unsupported_values(self):
        rows = [{"age": "42"}, {"age": "30"}]

        with self.assertRaisesRegex(ConfigError, "unsupported"):
            tensorize_source(rows, "features.age", features=("age",), targets=())

    def test_unsupported_nested_values_name_the_inner_problem(self):
        rows = [{"history": [1, "bad"]}, {"history": [2, 3]}]

        with self.assertRaisesRegex(ConfigError, "strings cannot be tensorized"):
            tensorize_source(rows, "features.history", features=("history",), targets=())

    def test_rejects_mixed_bool_and_numeric_vector_values(self):
        rows = [{"flags": [True, 1]}, {"flags": [False, 2]}]

        with self.assertRaisesRegex(ConfigError, "bool values cannot be mixed"):
            tensorize_source(rows, "features.flags", features=("flags",), targets=())

    def test_rejects_outputs_sources_at_tensorization_layer(self):
        with self.assertRaisesRegex(ConfigError, "cannot be tensorized"):
            tensorize_source([{"x": 1}], "outputs", features=("x",), targets=())


if __name__ == "__main__":
    unittest.main()
