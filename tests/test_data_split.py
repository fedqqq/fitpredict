import random
import unittest

from fitpredict.config import ConfigError, DataConfig, SplitConfig
from fitpredict.data import TabularRows, split_tabular_data


def rows(count):
    return TabularRows(
        rows=tuple(
            {"id": index, "label": "even" if index % 2 == 0 else "odd", "score": count - index}
            for index in range(count)
        ),
        columns=("id", "label", "score"),
    )


class DataSplitTests(unittest.TestCase):
    def test_largest_remainder_sizes_match_resolver_tie_order(self):
        result = split_tabular_data(
            rows(3),
            SplitConfig(train=0.5, val=0.5, test=0.0, shuffle=False),
        )

        self.assertEqual((len(result.train), len(result.val), len(result.test)), (2, 1, 0))
        self.assertEqual(result.diagnostics.train_size, 2)
        self.assertEqual([row["id"] for row in result.train], [0, 1])

    def test_sort_by_controls_order_without_shuffle(self):
        result = split_tabular_data(
            rows(5),
            SplitConfig(
                train=0.4,
                val=0.4,
                test=0.2,
                shuffle=False,
                sort_by="score",
                ascending=True,
            ),
        )

        self.assertEqual([row["id"] for row in result.train], [4, 3])
        self.assertEqual([row["id"] for row in result.val], [2, 1])
        self.assertEqual([row["id"] for row in result.test], [0])

    def test_shuffle_is_seeded_and_independent_from_global_rng(self):
        config = SplitConfig(train=0.5, val=0.25, test=0.25, shuffle=True, seed=7)
        first = split_tabular_data(rows(8), config)
        random.seed(123)
        before = random.random()
        second = split_tabular_data(rows(8), config)
        after = random.random()

        random.seed(123)
        expected_before = random.random()
        expected_after = random.random()

        self.assertEqual([row["id"] for row in first.train], [6, 7, 2, 4])
        self.assertEqual(
            [row["id"] for row in first.train],
            [row["id"] for row in second.train],
        )
        self.assertEqual(before, expected_before)
        self.assertEqual(after, expected_after)

    def test_stratify_keeps_global_sizes_and_reports_groups(self):
        result = split_tabular_data(
            rows(10),
            SplitConfig(
                train=0.6,
                val=0.2,
                test=0.2,
                shuffle=True,
                seed=11,
                stratify="label",
            ),
        )

        self.assertEqual((len(result.train), len(result.val), len(result.test)), (6, 2, 2))
        self.assertEqual(
            result.diagnostics.stratify_groups,
            {
                "'even'": {"train": 3, "val": 1, "test": 1},
                "'odd'": {"train": 3, "val": 1, "test": 1},
            },
        )

    def test_stratify_preserves_global_sorted_order_inside_partitions(self):
        result = split_tabular_data(
            rows(10),
            SplitConfig(
                train=0.6,
                val=0.2,
                test=0.2,
                shuffle=False,
                sort_by="score",
                ascending=True,
                stratify="label",
            ),
        )

        self.assertEqual([row["score"] for row in result.train], [1, 2, 3, 4, 5, 6])
        self.assertEqual([row["score"] for row in result.val], [7, 8])
        self.assertEqual([row["score"] for row in result.test], [9, 10])

    def test_sort_by_comparison_errors_are_config_errors(self):
        data = TabularRows(
            rows=(
                {"id": 0, "label": "a", "score": [1]},
                {"id": 1, "label": "b", "score": 1},
            ),
            columns=("id", "label", "score"),
        )

        with self.assertRaisesRegex(ConfigError, "data.split.sort_by"):
            split_tabular_data(
                data,
                SplitConfig(
                    train=1.0,
                    val=0.0,
                    test=0.0,
                    shuffle=False,
                    sort_by="score",
                ),
            )

    def test_zero_fraction_produces_empty_partition(self):
        result = split_tabular_data(
            rows(4),
            SplitConfig(train=1.0, val=0.0, test=0.0, shuffle=False),
        )

        self.assertEqual((len(result.train), len(result.val), len(result.test)), (4, 0, 0))
        self.assertEqual(result.val.rows, ())
        self.assertEqual(result.test.rows, ())

    def test_accepts_data_config(self):
        config = DataConfig.from_mapping(
            {
                "path": "data.csv",
                "format": "csv",
                "features": ["id", "score"],
                "targets": ["label"],
                "split": {"train": 0.5, "val": 0.25, "test": 0.25, "shuffle": False},
            }
        )

        result = split_tabular_data(rows(4), config)

        self.assertEqual((len(result.train), len(result.val), len(result.test)), (2, 1, 1))

    def test_rejects_missing_split_columns(self):
        with self.assertRaisesRegex(ConfigError, "sort_by column"):
            split_tabular_data(
                rows(2),
                SplitConfig(train=1.0, val=0.0, test=0.0, sort_by="missing"),
            )

        with self.assertRaisesRegex(ConfigError, "stratify column"):
            split_tabular_data(
                rows(2),
                SplitConfig(train=1.0, val=0.0, test=0.0, stratify="missing"),
            )


if __name__ == "__main__":
    unittest.main()
