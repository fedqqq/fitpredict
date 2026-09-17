import json
import tempfile
import unittest
from pathlib import Path

from fitpredict.config import ConfigError, ExperimentConfig, resolve_config
from fitpredict.data import load_data_metadata, load_tabular_data, metadata_from_rows


def config_for(path: Path, data_format: str = "csv"):
    return ExperimentConfig.from_mapping(
        {
            "data": {
                "path": str(path),
                "format": data_format,
                "features": ["age", "income"],
                "targets": ["label"],
                "split": {"train": 0.5, "val": 0.25, "test": 0.25},
            },
            "model": {
                "class": "models.Model",
                "params": {"input_dim": "${data.num_features}"},
                "inputs": {"x": {"source": "features"}},
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "optimizer": {"name": "AdamW"},
                "objectives": [
                    {
                        "loss": {"name": "MSELoss"},
                        "bindings": {
                            "input": {"source": "outputs"},
                            "target": {"source": "targets.label"},
                        },
                    }
                ],
            },
            "evaluation": {},
            "logging": {},
            "saving": {},
        }
    )


class DataLoaderTests(unittest.TestCase):
    def test_loads_csv_and_builds_resolver_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.csv"
            path.write_text(
                "age,income,label\n42,1000.5,1\n30,850.0,0\n25,725.5,1\n55,1100.0,0\n",
                encoding="utf-8",
            )
            config = config_for(path)

            metadata = load_data_metadata(config.data)
            resolved = resolve_config(config, data_metadata=metadata)

        self.assertEqual(metadata.num_rows, 4)
        self.assertEqual(metadata.columns, ("age", "income", "label"))
        self.assertEqual(metadata.feature_dtypes["age"], "str")
        self.assertEqual(metadata.feature_dtypes["income"], "str")
        self.assertEqual(metadata.target_dtypes["label"], "str")
        self.assertNotIn("age", metadata.feature_shapes)
        self.assertEqual(resolved.data.train_size, 2)
        self.assertEqual(resolved.training.steps_per_epoch, 1)

    def test_loads_jsonl_and_infers_vector_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            rows = [
                {"age": [1.0, 2.0], "income": 10, "label": True},
                {"age": [3.0, 4.0], "income": 20, "label": False},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            loaded = load_tabular_data(path, format="jsonl")
            metadata = metadata_from_rows(
                loaded.rows,
                columns=loaded.columns,
                features=("age", "income"),
                targets=("label",),
            )

        self.assertEqual(len(loaded), 2)
        self.assertEqual(metadata.feature_dtypes["age"], "list")
        self.assertEqual(metadata.feature_shapes["age"], (2,))
        self.assertEqual(metadata.target_dtypes["label"], "bool")

    def test_loads_json_list_of_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text(
                json.dumps(
                    [
                        {"age": 1, "income": 2, "label": 0},
                        {"age": 3, "income": 4, "label": 1},
                    ]
                ),
                encoding="utf-8",
            )
            config = config_for(path, "json")

            metadata = load_data_metadata(config.data)

        self.assertEqual(metadata.num_rows, 2)
        self.assertEqual(metadata.columns, ("age", "income", "label"))
        self.assertEqual(metadata.feature_dtypes["age"], "int64")

    def test_json_string_values_remain_string_dtype(self):
        metadata = metadata_from_rows(
            [{"age": "123", "label": "1"}],
            features=("age",),
            targets=("label",),
        )

        self.assertEqual(metadata.feature_dtypes["age"], "str")
        self.assertEqual(metadata.target_dtypes["label"], "str")

    def test_unknown_shapes_are_omitted_for_mixed_scalar_vector_values(self):
        metadata = metadata_from_rows(
            [{"x": 1, "label": 0}, {"x": [1, 2], "label": 1}],
            features=("x",),
            targets=("label",),
        )

        self.assertNotIn("x", metadata.feature_shapes)

    def test_unknown_shapes_are_omitted_for_ragged_vectors(self):
        metadata = metadata_from_rows(
            [{"x": [1, 2], "label": 0}, {"x": [1, 2, 3], "label": 1}],
            features=("x",),
            targets=("label",),
        )

        self.assertNotIn("x", metadata.feature_shapes)

    def test_unknown_shapes_are_omitted_for_nested_ragged_vectors(self):
        metadata = metadata_from_rows(
            [{"x": [[1, 2], [3, 4]], "label": 0}, {"x": [[1], [2, 3]], "label": 1}],
            features=("x",),
            targets=("label",),
        )

        self.assertNotIn("x", metadata.feature_shapes)

    def test_rejects_missing_configured_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.csv"
            path.write_text("age,label\n1,0\n", encoding="utf-8")
            config = config_for(path)

            with self.assertRaisesRegex(ConfigError, "missing configured feature"):
                load_data_metadata(config.data)

    def test_rejects_json_top_level_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text(json.dumps({"age": 1}), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "list of row objects"):
                load_tabular_data(path, format="json")

    def test_parquet_requires_optional_backend_or_loads_with_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.parquet"
            path.write_bytes(b"not a real parquet file")
            try:
                import pyarrow  # noqa: F401
            except ModuleNotFoundError:
                with self.assertRaisesRegex(ConfigError, "pyarrow is not installed"):
                    load_tabular_data(path, format="parquet")
            else:
                with self.assertRaisesRegex(ConfigError, "could not read parquet"):
                    load_tabular_data(path, format="parquet")

    def test_loads_real_parquet_when_pyarrow_is_available(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.parquet"
            table = pa.table({"age": [1, 2], "label": [0, 1]})
            pq.write_table(table, path)

            loaded = load_tabular_data(path, format="parquet")
            metadata = metadata_from_rows(
                loaded.rows,
                columns=loaded.columns,
                features=("age",),
                targets=("label",),
            )

        self.assertEqual(loaded.columns, ("age", "label"))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(metadata.feature_dtypes["age"], "int64")

    def test_loads_real_feather_when_pyarrow_is_available(self):
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.feather"
            table = pa.table({"age": [1, 2], "label": [0, 1]})
            feather.write_feather(table, path)

            loaded = load_tabular_data(path, format="feather")
            metadata = metadata_from_rows(
                loaded.rows,
                columns=loaded.columns,
                features=("age",),
                targets=("label",),
            )

        self.assertEqual(loaded.columns, ("age", "label"))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(metadata.target_dtypes["label"], "int64")


if __name__ == "__main__":
    unittest.main()
