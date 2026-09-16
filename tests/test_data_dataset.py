import unittest

import torch

from fitpredict.config import ConfigError, DataConfig
from fitpredict.data import TabularDataset, TabularRows, build_dataloader, build_datasets


def rows():
    return TabularRows(
        rows=(
            {"age": 42, "income": 1000.5, "history": [1.0, 2.0], "label": 1},
            {"age": 30, "income": 850.0, "history": [3.0, 4.0], "label": 0},
            {"age": 25, "income": 725.5, "history": [5.0, 6.0], "label": 1},
            {"age": 55, "income": 1100.0, "history": [7.0, 8.0], "label": 0},
        ),
        columns=("age", "income", "history", "label"),
    )


def config():
    return DataConfig.from_mapping(
        {
            "path": "unused.jsonl",
            "format": "jsonl",
            "features": ["age", "income", "history"],
            "targets": ["label"],
            "split": {"train": 0.5, "val": 0.25, "test": 0.25, "shuffle": False},
        }
    )


class DataDatasetTests(unittest.TestCase):
    def test_sample_exposes_generic_feature_and_target_maps(self):
        dataset = TabularDataset(rows(), data=config())

        sample = dataset[0]

        self.assertEqual(set(sample), {"features", "targets"})
        self.assertEqual(set(sample["features"]), {"age", "income", "history"})
        self.assertEqual(set(sample["targets"]), {"label"})
        self.assertEqual(sample["features"]["age"].shape, torch.Size([]))
        self.assertEqual(sample["features"]["history"].shape, torch.Size([2]))
        self.assertEqual(sample["targets"]["label"].shape, torch.Size([]))
        self.assertEqual(sample["features"]["age"].dtype, torch.int64)
        self.assertEqual(sample["features"]["income"].dtype, torch.float32)

    def test_dataloader_collates_nested_sample_contract(self):
        dataset = TabularDataset(rows(), data=config())
        loader = build_dataloader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(loader))

        self.assertEqual(tuple(batch["features"]["age"].shape), (2,))
        self.assertEqual(tuple(batch["features"]["history"].shape), (2, 2))
        self.assertEqual(tuple(batch["targets"]["label"].shape), (2,))
        torch.testing.assert_close(batch["features"]["age"], torch.tensor([42, 30]))
        torch.testing.assert_close(batch["targets"]["label"], torch.tensor([1, 0]))

    def test_build_datasets_uses_configured_split(self):
        datasets = build_datasets(rows(), config())

        self.assertEqual((len(datasets.train), len(datasets.val), len(datasets.test)), (2, 1, 1))
        self.assertEqual(datasets.split.diagnostics.train_size, 2)
        self.assertEqual(datasets.train[1]["features"]["age"].item(), 30)
        self.assertEqual(datasets.val[0]["features"]["age"].item(), 25)

    def test_empty_split_partition_is_valid_zero_length_dataset(self):
        data = DataConfig.from_mapping(
            {
                "path": "unused.jsonl",
                "format": "jsonl",
                "features": ["age"],
                "targets": ["label"],
                "split": {"train": 1.0, "val": 0.0, "test": 0.0, "shuffle": False},
            }
        )

        datasets = build_datasets(rows(), data)

        self.assertEqual(len(datasets.val), 0)
        self.assertFalse(datasets.val.diagnostics.tensorized)
        with self.assertRaises(IndexError):
            _ = datasets.val[0]

    def test_rejects_missing_declared_columns(self):
        with self.assertRaisesRegex(ConfigError, "missing configured feature"):
            TabularDataset(
                TabularRows(rows=({"age": 1, "label": 0},), columns=("age", "label")),
                features=("age", "income"),
                targets=("label",),
            )

    def test_rejects_invalid_batch_size(self):
        dataset = TabularDataset(rows(), data=config())

        for batch_size in (0, True, 1.5, "2"):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ConfigError, "positive integer"):
                    build_dataloader(dataset, batch_size=batch_size)

    def test_empty_dataset_dataloader_allows_shuffle_request(self):
        dataset = TabularDataset(
            TabularRows(rows=(), columns=("age", "label")),
            features=("age",),
            targets=("label",),
        )

        loader = build_dataloader(dataset, batch_size=2, shuffle=True)

        self.assertEqual(list(loader), [])


if __name__ == "__main__":
    unittest.main()
