"""Generic PyTorch dataset helpers for tabular fitpredict data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import DataConfig, DataMetadata
from fitpredict.data.loader import TabularRows
from fitpredict.data.split import TabularSplit, split_tabular_data
from fitpredict.data.tensorizer import tensorize_source


@dataclass(frozen=True)
class DatasetDiagnostics:
    """Execution details for a generic tabular dataset."""

    num_rows: int
    columns: tuple[str, ...]
    features: tuple[str, ...]
    targets: tuple[str, ...]
    feature_shapes: dict[str, tuple[int, ...]]
    target_shapes: dict[str, tuple[int, ...]]
    feature_dtypes: dict[str, str]
    target_dtypes: dict[str, str]
    tensorized: bool


@dataclass(frozen=True)
class DatasetSplits:
    """Train/validation/test datasets plus split diagnostics."""

    train: "TabularDataset"
    val: "TabularDataset"
    test: "TabularDataset"
    split: TabularSplit


class TabularDataset(Dataset):
    """Expose tabular rows as generic feature/target tensor samples.

    Each sample has the public shape ``{"features": {...}, "targets": {...}}``.
    The dataset does not know about model inputs, objectives, losses, or binding
    names; those layers can resolve sources against this generic structure.
    """

    def __init__(
        self,
        rows: TabularRows | Iterable[Mapping[str, Any]],
        *,
        data: DataConfig | None = None,
        features: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        metadata: DataMetadata | None = None,
    ) -> None:
        self._rows = _normalize_rows(rows, columns)
        self.features = _declared("features", data.features if data is not None else features)
        self.targets = _declared("targets", data.targets if data is not None else targets)
        self.metadata = metadata if metadata is not None else data.metadata if data is not None else None
        _validate_declared_columns(self._rows.columns, self.features, self.targets)

        self._feature_tensors = self._tensorize_namespace("features", self.features)
        self._target_tensors = self._tensorize_namespace("targets", self.targets)
        self.diagnostics = DatasetDiagnostics(
            num_rows=len(self._rows),
            columns=self._rows.columns,
            features=self.features,
            targets=self.targets,
            feature_shapes=_sample_shapes(self._feature_tensors),
            target_shapes=_sample_shapes(self._target_tensors),
            feature_dtypes=_tensor_dtypes(self._feature_tensors),
            target_dtypes=_tensor_dtypes(self._target_tensors),
            tensorized=len(self._rows) > 0,
        )

    @property
    def rows(self) -> TabularRows:
        """Return the source rows backing this dataset."""

        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("tabular dataset index out of range")
        return {
            "features": {
                name: tensor[index]
                for name, tensor in self._feature_tensors.items()
            },
            "targets": {
                name: tensor[index]
                for name, tensor in self._target_tensors.items()
            },
        }

    def _tensorize_namespace(
        self,
        namespace: str,
        names: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(self._rows) == 0:
            return {}
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            source = f"{namespace}.{name}"
            result = tensorize_source(
                self._rows.rows,
                source,
                features=self.features,
                targets=self.targets,
                metadata=self.metadata,
                path=f"dataset.{source}",
            )
            tensors[name] = result.tensor
        return tensors


def build_datasets(
    rows: TabularRows | Iterable[Mapping[str, Any]],
    data: DataConfig,
    *,
    columns: Sequence[str] | None = None,
) -> DatasetSplits:
    """Split rows with ``data.split`` and wrap each partition as a dataset."""

    split = split_tabular_data(rows, data, columns=columns)
    return DatasetSplits(
        train=TabularDataset(split.train, data=data),
        val=TabularDataset(split.val, data=data),
        test=TabularDataset(split.test, data=data),
        split=split,
    )


def build_dataloader(
    dataset: TabularDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """Create a PyTorch DataLoader for a ``TabularDataset``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigError("batch_size must be a positive integer.")
    selected_shuffle = False if len(dataset) == 0 else shuffle
    return DataLoader(dataset, batch_size=batch_size, shuffle=selected_shuffle, **kwargs)


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


def _declared(name: str, values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        raise ConfigError(f"data.{name} is required for dataset construction.")
    return tuple(values)


def _validate_declared_columns(
    columns: Sequence[str],
    features: Sequence[str],
    targets: Sequence[str],
) -> None:
    available = set(columns)
    missing_features = sorted(set(features) - available)
    missing_targets = sorted(set(targets) - available)
    if missing_features:
        raise ConfigError(
            "data columns are missing configured feature(s): "
            + ", ".join(missing_features)
        )
    if missing_targets:
        raise ConfigError(
            "data columns are missing configured target(s): "
            + ", ".join(missing_targets)
        )


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


def _sample_shapes(tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape[1:]) for name, tensor in tensors.items()}


def _tensor_dtypes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: str(tensor.dtype).removeprefix("torch.") for name, tensor in tensors.items()}
