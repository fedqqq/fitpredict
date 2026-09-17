"""Prediction entry point for resolved fitpredict experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from fitpredict.config import (
    ComponentResolver,
    DataMetadata,
    ExperimentConfig,
    load_config,
    resolve_config,
)
from fitpredict.config.errors import ConfigError
from fitpredict.data import (
    TabularRows,
    build_dataloader,
    load_tabular_data,
    metadata_from_rows,
)
from fitpredict.data.dataset import TabularDataset
from fitpredict.training import (
    _build_model_context,
    _concat_outputs,
    _detach_cpu,
    _load_checkpoint_model_state,
    _load_model_weights,
    _move_batch_to_device,
    _resolve_device,
)


def predict(
    config: str | Path | Mapping[str, Any] | ExperimentConfig,
    checkpoint: str | Path | None = None,
    data: str | Path | TabularRows | Iterable[Mapping[str, Any]] | None = None,
) -> Any:
    """Run model inference and return detached CPU predictions.

    The public prediction contract is intentionally small:

    - ``config`` is loaded and resolved with metadata from the prediction rows.
    - ``data`` may be omitted, a path using the config's data format/options, a
      ``TabularRows`` object, or an iterable of row mappings.
    - target columns are not required in prediction rows, even when the training
      config declares targets for losses or metrics.
    - empty prediction data returns ``None`` because no model outputs can be
      inferred without running a batch.
    - ``checkpoint`` overrides ``model.weights`` and is loaded as a model state
      dict or a checkpoint mapping containing ``state_dict`` or
      ``model_state_dict``; when neither is set, the configured model's initial
      parameters are used.
    - the model runs in ``eval`` mode under ``torch.no_grad()``.
    - return shape mirrors the model output: a single tensor stays a tensor, and
      a dict of tensors stays a dict of concatenated tensors.
    - tensor outputs must include a leading batch dimension. Scalar tensor
      outputs are rejected because they cannot be concatenated per input row.
    """

    typed_config = config if isinstance(config, ExperimentConfig) else load_config(config)
    rows = _load_prediction_rows(typed_config, data)
    metadata = _prediction_metadata(typed_config, rows)
    resolved = resolve_config(typed_config, data_metadata=metadata)

    dataset = TabularDataset(
        rows,
        features=resolved.data.features,
        targets=(),
        columns=rows.columns,
        metadata=resolved.data.metadata,
    )
    loader = build_dataloader(
        dataset,
        batch_size=resolved.training.batch_size,
        shuffle=False,
    )

    device = _resolve_device(resolved.training.device)
    model = ComponentResolver().instantiate("model", resolved.model).to(device)
    if checkpoint is not None:
        _load_checkpoint_model_state(model, Path(checkpoint), device)
    else:
        _load_model_weights(model, resolved.model.weights, device)

    model.eval()
    outputs: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            context = _build_model_context(
                model,
                batch,
                config=resolved,
                context_path="predict.batch",
            )
            _validate_prediction_output(context.outputs, path="predict.outputs")
            outputs.append(_detach_cpu(context.outputs))
    return _concat_outputs(outputs)


def _load_prediction_rows(
    config: ExperimentConfig,
    data: str | Path | TabularRows | Iterable[Mapping[str, Any]] | None,
) -> TabularRows:
    if data is None:
        return _with_empty_columns(load_tabular_data(config.data), config)
    if isinstance(data, (str, Path)):
        return _with_empty_columns(
            load_tabular_data(
                data,
                format=config.data.format,
                options=config.data.options,
            ),
            config,
        )
    if isinstance(data, TabularRows):
        return _with_empty_columns(data, config)
    if isinstance(data, Mapping):
        raise ConfigError("predict data must be a path, TabularRows, or iterable of row mappings.")
    materialized = tuple(data)
    rows = TabularRows(rows=materialized, columns=_columns_from_rows(materialized))
    return _with_empty_columns(rows, config)


def _prediction_metadata(config: ExperimentConfig, rows: TabularRows) -> DataMetadata:
    metadata = metadata_from_rows(
        rows.rows,
        columns=rows.columns,
        features=config.data.features,
        targets=(),
    )
    missing_targets = tuple(target for target in config.data.targets if target not in metadata.columns)
    if not missing_targets:
        return metadata
    return replace(
        metadata,
        columns=metadata.columns + missing_targets,
    )


def _with_empty_columns(rows: TabularRows, config: ExperimentConfig) -> TabularRows:
    if len(rows) > 0 or rows.columns:
        return rows
    return TabularRows(rows=rows.rows, columns=config.data.features)


def _validate_prediction_output(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ConfigError(f"{path} must include a leading batch dimension.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ConfigError(f"{path} keys must be non-empty strings.")
            _validate_prediction_output(item, path=f"{path}.{key}")
        return
    raise ConfigError(
        f"{path} must be a torch.Tensor or a mapping of torch.Tensor values."
    )


def _columns_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ConfigError(f"predict data rows[{index}] must be a row mapping.")
        for key in row:
            if not isinstance(key, str) or not key:
                raise ConfigError("predict data row keys must be non-empty strings.")
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return tuple(columns)
