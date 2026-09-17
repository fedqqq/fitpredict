"""Tensorization helpers for the common fitpredict tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import BindingConfig, BindingSource, DataConfig, DataMetadata


_DTYPES: Mapping[str, torch.dtype] = {
    "bool": torch.bool,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_NUMERIC_DTYPES = frozenset({"int64", "float16", "float32", "float64"})


@dataclass(frozen=True)
class TensorizedBatch:
    """A tensor plus the source columns used to build it."""

    tensor: torch.Tensor
    source: BindingSource
    columns: tuple[str, ...]


def tensorize_binding(
    rows: Sequence[Mapping[str, Any]],
    binding: BindingConfig,
    data: DataConfig,
    *,
    path: str = "binding",
) -> TensorizedBatch:
    """Tensorize one binding using declared data features and targets."""

    return tensorize_source(
        rows,
        binding.source,
        data=data,
        dtype=binding.dtype,
        path=path,
    )


def tensorize_source(
    rows: Sequence[Mapping[str, Any]],
    source: BindingSource | str,
    *,
    data: DataConfig | None = None,
    features: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    metadata: DataMetadata | None = None,
    dtype: str | None = None,
    path: str = "source",
) -> TensorizedBatch:
    """Tensorize a feature/target binding source into a batch tensor.

    The first tensor dimension is always batch size ``B``. Remaining dimensions
    are the sample shape ``S`` inferred from each row value.
    """

    parsed = source if isinstance(source, BindingSource) else BindingSource.parse(source, path)
    selected_metadata = (
        metadata if metadata is not None else data.metadata if data is not None else None
    )
    torch_dtype = _resolve_explicit_dtype(dtype, f"{path}.dtype")

    if parsed.is_feature_aggregate:
        declared_features = _declared("features", data.features if data is not None else features)
        if not declared_features:
            raise ConfigError(f'{path} source "features" requires at least one declared feature.')
        return _tensorize_feature_aggregate(
            rows,
            parsed,
            declared_features,
            dtype=torch_dtype,
            metadata=selected_metadata,
            path=path,
        )
    if parsed.namespace == "features":
        declared_features = _declared("features", data.features if data is not None else features)
        if parsed.name not in declared_features:
            raise ConfigError(f'{path} source "{parsed}" is not declared in data.features.')
        tensor = _tensorize_column(
            rows,
            parsed.name,
            dtype=torch_dtype,
            metadata_shape=(
                selected_metadata.feature_shapes.get(parsed.name) if selected_metadata else None
            ),
            metadata_dtype=(
                selected_metadata.feature_dtypes.get(parsed.name) if selected_metadata else None
            ),
            path=f'{path} source "{parsed}"',
        )
        return TensorizedBatch(tensor=tensor, source=parsed, columns=(parsed.name,))
    if parsed.namespace == "targets":
        declared_targets = _declared("targets", data.targets if data is not None else targets)
        if parsed.name not in declared_targets:
            raise ConfigError(f'{path} source "{parsed}" is not declared in data.targets.')
        tensor = _tensorize_column(
            rows,
            parsed.name,
            dtype=torch_dtype,
            metadata_shape=(
                selected_metadata.target_shapes.get(parsed.name) if selected_metadata else None
            ),
            metadata_dtype=(
                selected_metadata.target_dtypes.get(parsed.name) if selected_metadata else None
            ),
            path=f'{path} source "{parsed}"',
        )
        return TensorizedBatch(tensor=tensor, source=parsed, columns=(parsed.name,))
    raise ConfigError(f'{path} source "{parsed}" cannot be tensorized from data rows.')


def tensorize_features(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    dtype: str | None = None,
    metadata: DataMetadata | None = None,
    path: str = "features",
) -> torch.Tensor:
    """Tensorize all declared features as one aggregate tensor."""

    source = BindingSource.parse("features")
    return _tensorize_feature_aggregate(
        rows,
        source,
        tuple(features),
        dtype=_resolve_explicit_dtype(dtype, f"{path}.dtype"),
        metadata=metadata,
        path=path,
    ).tensor


def aggregate_feature_tensors(
    tensors: Mapping[str, torch.Tensor],
    features: Sequence[str],
    *,
    dtype: str | None = None,
    path: str = "features",
) -> torch.Tensor:
    """Aggregate already-batched feature tensors according to ``source: features``.

    Each input tensor must already follow the batch contract ``[B, *S]``. The
    returned tensor stacks declared features on dimension 1, yielding
    ``[B, F, *S]``.
    """

    declared_features = tuple(features)
    if not declared_features:
        raise ConfigError(f'{path} source "features" requires at least one declared feature.')
    missing = [feature for feature in declared_features if feature not in tensors]
    if missing:
        raise ConfigError(
            f'{path} source "features" is missing feature tensor(s): {", ".join(missing)}.'
        )

    batches = []
    for feature in declared_features:
        tensor = tensors[feature]
        if not isinstance(tensor, torch.Tensor):
            raise ConfigError(f'{path} source "features.{feature}" must be a torch.Tensor.')
        if tensor.ndim == 0:
            raise ConfigError(f'{path} source "features.{feature}" must include a batch dimension.')
        batches.append(tensor)

    batch_sizes = {int(batch.shape[0]) for batch in batches}
    if len(batch_sizes) != 1:
        observed = "\n".join(
            f"- {feature}: {tuple(batch.shape)}"
            for feature, batch in zip(declared_features, batches)
        )
        raise ConfigError(
            f'{path} source "features" could not be aggregated because batch sizes differ.\n'
            f"Observed batch shapes:\n{observed}"
        )
    sample_shapes = {tuple(batch.shape[1:]) for batch in batches}
    if len(sample_shapes) != 1:
        observed = "\n".join(
            f"- {feature}: {tuple(batch.shape[1:])}"
            for feature, batch in zip(declared_features, batches)
        )
        raise ConfigError(
            f'{path} source "features" could not be aggregated because feature sample shapes differ.\n'
            f"Observed sample shapes:\n{observed}"
        )

    explicit_dtype = _resolve_explicit_dtype(dtype, f"{path}.dtype")
    aggregate_dtype = explicit_dtype or _aggregate_dtype(batches, declared_features, path)
    return torch.stack([batch.to(dtype=aggregate_dtype) for batch in batches], dim=1)


def _tensorize_feature_aggregate(
    rows: Sequence[Mapping[str, Any]],
    source: BindingSource,
    features: tuple[str, ...],
    *,
    dtype: torch.dtype | None,
    metadata: DataMetadata | None,
    path: str,
) -> TensorizedBatch:
    batches = [
        _tensorize_column(
            rows,
            feature,
            dtype=dtype,
            metadata_shape=(metadata.feature_shapes.get(feature) if metadata else None),
            metadata_dtype=(metadata.feature_dtypes.get(feature) if metadata else None),
            path=f'{path} source "features.{feature}"',
        )
        for feature in features
    ]
    sample_shapes = {tuple(batch.shape[1:]) for batch in batches}
    if len(sample_shapes) != 1:
        observed = "\n".join(
            f"- {feature}: {tuple(batch.shape[1:])}" for feature, batch in zip(features, batches)
        )
        raise ConfigError(
            f'{path} source "features" could not be aggregated because feature sample shapes differ.\n'
            f"Observed sample shapes:\n{observed}"
        )

    aggregate_dtype = dtype or _aggregate_dtype(batches, features, path)
    converted = [batch.to(dtype=aggregate_dtype) for batch in batches]
    return TensorizedBatch(
        tensor=torch.stack(converted, dim=1),
        source=source,
        columns=features,
    )


def _tensorize_column(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    *,
    dtype: torch.dtype | None,
    metadata_shape: tuple[int, ...] | None,
    metadata_dtype: str | None,
    path: str,
) -> torch.Tensor:
    values = []
    shapes: list[tuple[int, ...]] = []
    inferred_dtypes: list[str] = []
    for index, row in enumerate(rows):
        if column not in row:
            raise ConfigError(f'{path} is missing column "{column}" at row {index}.')
        value = row[column]
        if value is None:
            raise ConfigError(
                f"{path} contains null at row {index}; null values cannot be tensorized."
            )
        shape = _sample_shape(value)
        if shape is None:
            raise ConfigError(
                f"{path} contains ragged nested values at row {index}; "
                "fitpredict does not perform automatic padding."
            )
        value_dtype = _value_dtype(value)
        if value_dtype == "unsupported":
            raise ConfigError(
                f"{path} contains unsupported value at row {index}: "
                f"{_unsupported_value_description(value)}."
            )
        shapes.append(shape)
        inferred_dtypes.append(value_dtype)
        values.append(value)

    if not values:
        raise ConfigError(f"{path} cannot be tensorized because the row batch is empty.")

    unique_shapes = set(shapes)
    if len(unique_shapes) != 1:
        observed = "\n".join(f"- row {index}: {shape}" for index, shape in enumerate(shapes))
        raise ConfigError(
            f"{path} could not be converted to a consistent batch tensor.\n"
            f"Observed sample shapes:\n{observed}\n"
            "fitpredict does not perform automatic padding."
        )
    sample_shape = shapes[0]
    if metadata_shape is not None and sample_shape != metadata_shape:
        raise ConfigError(
            f"{path} sample shape {sample_shape} does not match data metadata shape {metadata_shape}."
        )
    if metadata_dtype in {"str", "dict", "mixed"}:
        raise ConfigError(f'{path} has unsupported metadata dtype "{metadata_dtype}".')

    selected_dtype = dtype or _single_source_dtype(inferred_dtypes, path)
    try:
        tensor = torch.tensor(values, dtype=selected_dtype)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ConfigError(f"{path} could not be converted to a tensor: {exc}") from exc
    if tuple(tensor.shape[1:]) != sample_shape:
        raise ConfigError(
            f"{path} produced tensor shape {tuple(tensor.shape)}, expected batch shape "
            f"[{len(values)}, *{sample_shape}]."
        )
    return tensor


def _declared(name: str, values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        raise ConfigError(f"data.{name} is required for tensorization.")
    return tuple(values)


def _resolve_explicit_dtype(dtype: str | None, path: str) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return None
    try:
        return _DTYPES[dtype]
    except KeyError as exc:
        raise ConfigError(f"{path} is unsupported: {dtype}.") from exc


def _single_source_dtype(dtypes: Sequence[str], path: str) -> torch.dtype:
    unique = set(dtypes)
    if not unique <= (_NUMERIC_DTYPES | {"bool"}):
        unsupported = ", ".join(sorted(unique - (_NUMERIC_DTYPES | {"bool"})))
        raise ConfigError(f"{path} contains unsupported dtype(s): {unsupported}.")
    if "bool" in unique and len(unique) > 1:
        raise ConfigError(f"{path} mixes bool and numeric values, which is unsupported.")
    if unique == {"bool"}:
        return torch.bool
    if any(dtype in {"float16", "float32", "float64"} for dtype in unique):
        return torch.float32
    return torch.int64


def _aggregate_dtype(
    batches: Sequence[torch.Tensor],
    features: Sequence[str],
    path: str,
) -> torch.dtype:
    dtype_names = {_dtype_name(batch.dtype) for batch in batches}
    if dtype_names == {"bool"}:
        return torch.bool
    if "bool" in dtype_names:
        observed = ", ".join(
            f"{feature}: {_dtype_name(batch.dtype)}" for feature, batch in zip(features, batches)
        )
        raise ConfigError(
            f'{path} source "features" cannot aggregate bool and numeric feature tensors. '
            f"Observed dtypes: {observed}."
        )
    if not dtype_names <= _NUMERIC_DTYPES:
        observed = ", ".join(
            f"{feature}: {_dtype_name(batch.dtype)}" for feature, batch in zip(features, batches)
        )
        raise ConfigError(
            f'{path} source "features" contains unsupported feature tensor dtype(s): {observed}.'
        )
    if len(dtype_names) > 1:
        return torch.float64
    return batches[0].dtype


def _sample_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (bool, int, float)) and not isinstance(value, bool):
        return ()
    if isinstance(value, bool):
        return ()
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        maybe_child_shapes = [_sample_shape(item) for item in value]
        if any(shape is None for shape in maybe_child_shapes):
            return None
        child_shapes = [shape for shape in maybe_child_shapes if shape is not None]
        if len(set(child_shapes)) != 1:
            return None
        return (len(value),) + child_shapes[0]
    return ()


def _value_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    if isinstance(value, str) or isinstance(value, Mapping):
        return "unsupported"
    if isinstance(value, (list, tuple)):
        child_dtypes = {_value_dtype(item) for item in value}
        if "unsupported" in child_dtypes:
            return "unsupported"
        if "bool" in child_dtypes and len(child_dtypes) > 1:
            return "unsupported"
        if any(dtype.startswith("float") for dtype in child_dtypes):
            return "float64"
        if child_dtypes == {"bool"}:
            return "bool"
        if child_dtypes <= {"int64"}:
            return "int64"
    return "unsupported"


def _unsupported_value_description(value: Any) -> str:
    if isinstance(value, str):
        return "strings cannot be tensorized"
    if isinstance(value, Mapping):
        return f"{type(value).__name__} values cannot be tensorized"
    if isinstance(value, (list, tuple)):
        for item in value:
            if _value_dtype(item) == "unsupported":
                return _unsupported_value_description(item)
        if any(_value_dtype(item) == "bool" for item in value):
            return "bool values cannot be mixed with numeric values"
        return f"{type(value).__name__} values cannot be tensorized"
    return f"{type(value).__name__} values cannot be tensorized"


def _dtype_name(dtype: torch.dtype) -> str:
    for name, torch_dtype in _DTYPES.items():
        if dtype == torch_dtype:
            return name
    return str(dtype)
