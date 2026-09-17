"""Config-driven train/validation/test splitting for tabular rows."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import floor
import random
from typing import Any

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import DataConfig, SplitConfig
from fitpredict.data.loader import TabularRows


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitDiagnostics:
    """Execution details for a data split."""

    num_rows: int
    train_size: int
    val_size: int
    test_size: int
    shuffle: bool
    seed: int | None
    sort_by: str | None
    ascending: bool
    stratify: str | None
    stratify_groups: dict[str, dict[str, int]]


@dataclass(frozen=True)
class TabularSplit:
    """Train/validation/test row partitions plus diagnostics."""

    train: TabularRows
    val: TabularRows
    test: TabularRows
    diagnostics: SplitDiagnostics


def split_tabular_data(
    rows: TabularRows | Iterable[Mapping[str, Any]],
    split: DataConfig | SplitConfig,
    *,
    columns: Sequence[str] | None = None,
) -> TabularSplit:
    """Split row mappings into train, validation, and test partitions."""

    loaded = _normalize_rows(rows, columns)
    split_config = split.split if isinstance(split, DataConfig) else split
    _validate_split_columns(loaded.columns, split_config)

    indices = list(range(len(loaded.rows)))
    if split_config.sort_by is not None:
        sort_by = split_config.sort_by
        try:
            indices.sort(
                key=lambda index: _sort_key(loaded.rows[index], sort_by),
                reverse=not split_config.ascending,
            )
        except TypeError as exc:
            raise ConfigError(
                f"data.split.sort_by column {split_config.sort_by!r} contains "
                "values that cannot be compared."
            ) from exc

    rng = random.Random(split_config.seed)
    if split_config.shuffle:
        if split_config.stratify is None:
            rng.shuffle(indices)
        else:
            indices = _shuffle_within_strata(indices, loaded.rows, split_config.stratify, rng)

    sizes = _split_sizes(
        len(loaded.rows),
        (split_config.train, split_config.val, split_config.test),
    )
    partitions, stratify_groups = _partition_indices(indices, loaded.rows, split_config, sizes)

    train = _rows_for_indices(loaded, partitions[0])
    val = _rows_for_indices(loaded, partitions[1])
    test = _rows_for_indices(loaded, partitions[2])
    diagnostics = SplitDiagnostics(
        num_rows=len(loaded.rows),
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        shuffle=split_config.shuffle,
        seed=split_config.seed,
        sort_by=split_config.sort_by,
        ascending=split_config.ascending,
        stratify=split_config.stratify,
        stratify_groups=stratify_groups,
    )
    return TabularSplit(train=train, val=val, test=test, diagnostics=diagnostics)


def _normalize_rows(
    rows: TabularRows | Iterable[Mapping[str, Any]],
    columns: Sequence[str] | None,
) -> TabularRows:
    if isinstance(rows, TabularRows):
        if columns is not None and tuple(columns) != rows.columns:
            raise ConfigError("columns override conflicts with TabularRows.columns.")
        return rows

    materialized = tuple(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ConfigError(f"rows[{index}] must be a row mapping.")
    return TabularRows(
        rows=materialized,
        columns=tuple(columns) if columns is not None else _columns_from_rows(materialized),
    )


def _validate_split_columns(columns: Sequence[str], split: SplitConfig) -> None:
    available = set(columns)
    if split.sort_by is not None and split.sort_by not in available:
        raise ConfigError(f"data.split.sort_by column is missing from data: {split.sort_by}.")
    if split.stratify is not None and split.stratify not in available:
        raise ConfigError(f"data.split.stratify column is missing from data: {split.stratify}.")


def _sort_key(row: Mapping[str, Any], column: str) -> Any:
    try:
        return row[column]
    except KeyError as exc:
        raise ConfigError(f"row is missing data.split.sort_by column: {column}.") from exc


def _shuffle_within_strata(
    indices: list[int],
    rows: Sequence[Mapping[str, Any]],
    column: str,
    rng: random.Random,
) -> list[int]:
    groups = _group_indices(indices, rows, column)
    result: list[int] = []
    for group_indices in groups.values():
        shuffled = list(group_indices)
        rng.shuffle(shuffled)
        result.extend(shuffled)
    return result


def _partition_indices(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    split: SplitConfig,
    sizes: tuple[int, int, int],
) -> tuple[tuple[list[int], list[int], list[int]], dict[str, dict[str, int]]]:
    if split.stratify is None:
        train_size, val_size, _ = sizes
        return (
            (
                list(indices[:train_size]),
                list(indices[train_size : train_size + val_size]),
                list(indices[train_size + val_size :]),
            ),
            {},
        )

    return _stratified_partition(indices, rows, split, sizes)


def _stratified_partition(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    split: SplitConfig,
    sizes: tuple[int, int, int],
) -> tuple[tuple[list[int], list[int], list[int]], dict[str, dict[str, int]]]:
    groups = _group_indices(indices, rows, split.stratify or "")
    partitions: tuple[list[int], list[int], list[int]] = ([], [], [])
    remaining = list(sizes)
    diagnostics: dict[str, dict[str, int]] = {}

    for group_key, group_indices in groups.items():
        exact = [
            len(group_indices) * split.train,
            len(group_indices) * split.val,
            len(group_indices) * split.test,
        ]
        assigned_in_group = [0, 0, 0]
        for index in group_indices:
            candidates = [part for part, count in enumerate(remaining) if count > 0]
            if not candidates:
                raise ConfigError("data split assignment exhausted before all rows were assigned.")
            selected = max(
                candidates,
                key=lambda part: (
                    exact[part] - assigned_in_group[part],
                    remaining[part],
                    -part,
                ),
            )
            partitions[selected].append(index)
            remaining[selected] -= 1
            assigned_in_group[selected] += 1
        diagnostics[group_key] = {
            name: assigned_in_group[position] for position, name in enumerate(SPLIT_NAMES)
        }

    if remaining != [0, 0, 0]:
        raise ConfigError("data split assignment did not produce the requested split sizes.")
    order_position = {index: position for position, index in enumerate(indices)}
    for partition in partitions:
        partition.sort(key=order_position.__getitem__)
    return partitions, diagnostics


def _group_indices(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    column: str,
) -> "OrderedDict[str, list[int]]":
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index in indices:
        try:
            raw_key = rows[index][column]
        except KeyError as exc:
            raise ConfigError(f"row is missing data.split.stratify column: {column}.") from exc
        key = _diagnostic_key(raw_key)
        groups.setdefault(key, []).append(index)
    return groups


def _diagnostic_key(value: Any) -> str:
    if value is None:
        return "null"
    return repr(value)


def _rows_for_indices(rows: TabularRows, indices: Sequence[int]) -> TabularRows:
    return TabularRows(
        rows=tuple(rows.rows[index] for index in indices),
        columns=rows.columns,
    )


def _split_sizes(num_rows: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [num_rows * fraction for fraction in fractions]
    sizes = [floor(value) for value in exact]
    remaining = num_rows - sum(sizes)
    order = sorted(
        range(3),
        key=lambda index: (exact[index] - sizes[index], -index),
        reverse=True,
    )
    for index in order:
        if remaining == 0:
            break
        if fractions[index] > 0:
            sizes[index] += 1
            remaining -= 1
    return sizes[0], sizes[1], sizes[2]


def _columns_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if not isinstance(key, str) or not key:
                raise ConfigError("row keys must be non-empty strings.")
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return tuple(columns)
