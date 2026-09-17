"""Resolve binding sources against a runtime batch and model outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import BindingConfig, BindingSource
from fitpredict.data.tensorizer import aggregate_feature_tensors


_DTYPES: Mapping[str, torch.dtype] = {
    "bool": torch.bool,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class RuntimeSourceContext:
    """Runtime tensors available to the binding protocol."""

    features: Mapping[str, torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    outputs: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "features",
            _require_tensor_mapping(self.features, "runtime_context.features"),
        )
        object.__setattr__(
            self,
            "targets",
            _require_tensor_mapping(self.targets, "runtime_context.targets"),
        )

    @classmethod
    def from_batch(
        cls,
        batch: Mapping[str, Any],
        *,
        outputs: Any = None,
        path: str = "runtime_context",
    ) -> "RuntimeSourceContext":
        """Build a source context from a generic dataset batch."""

        if not isinstance(batch, Mapping):
            raise ConfigError(f"{path}.batch must be a mapping.")
        features = _require_tensor_mapping(batch.get("features"), f"{path}.features")
        targets = _require_tensor_mapping(batch.get("targets"), f"{path}.targets")
        selected_outputs = batch.get("outputs", outputs)
        return cls(features=features, targets=targets, outputs=selected_outputs)


class SourceResolver:
    """Resolve ``features.*``, ``targets.*``, ``outputs.*``, and aggregate sources."""

    def __init__(
        self,
        context: RuntimeSourceContext | Mapping[str, Any],
        *,
        features: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
        path: str = "runtime_context",
    ) -> None:
        self.context = _normalize_context(context, path)
        self.features = tuple(features) if features is not None else tuple(self.context.features)
        self.targets = tuple(targets) if targets is not None else tuple(self.context.targets)

    def resolve(
        self,
        source: BindingSource | BindingConfig | str,
        *,
        dtype: str | None = None,
        path: str = "source",
    ) -> torch.Tensor:
        """Resolve one binding source to a tensor.

        Returned tensors are never detached. If ``dtype`` is supplied, PyTorch's
        differentiable ``Tensor.to`` conversion is used for tensor outputs.
        """

        binding_dtype = None
        if isinstance(source, BindingConfig):
            binding_dtype = source.dtype
            parsed = source.source
        else:
            parsed = (
                source if isinstance(source, BindingSource) else BindingSource.parse(source, path)
            )
        selected_dtype = dtype if dtype is not None else binding_dtype

        if parsed.is_feature_aggregate:
            return _convert_dtype(
                aggregate_feature_tensors(
                    self.context.features,
                    self.features,
                    dtype=selected_dtype,
                    path=path,
                ),
                selected_dtype,
                f"{path}.dtype",
            )
        if parsed.namespace == "features":
            return _convert_dtype(
                _resolve_named_tensor(
                    self.context.features,
                    parsed.name,
                    declared=self.features,
                    namespace="features",
                    path=path,
                ),
                selected_dtype,
                f"{path}.dtype",
            )
        if parsed.namespace == "targets":
            return _convert_dtype(
                _resolve_named_tensor(
                    self.context.targets,
                    parsed.name,
                    declared=self.targets,
                    namespace="targets",
                    path=path,
                ),
                selected_dtype,
                f"{path}.dtype",
            )
        if parsed.namespace == "outputs":
            return _convert_dtype(
                _resolve_output(self.context.outputs, parsed, path),
                selected_dtype,
                f"{path}.dtype",
            )
        raise ConfigError(f'{path} source "{parsed}" is unsupported.')


def resolve_source(
    source: BindingSource | BindingConfig | str,
    context: RuntimeSourceContext | Mapping[str, Any],
    *,
    features: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    dtype: str | None = None,
    path: str = "source",
) -> torch.Tensor:
    """Resolve one source without explicitly constructing ``SourceResolver``."""

    return SourceResolver(context, features=features, targets=targets).resolve(
        source,
        dtype=dtype,
        path=path,
    )


def _normalize_context(
    context: RuntimeSourceContext | Mapping[str, Any],
    path: str,
) -> RuntimeSourceContext:
    if isinstance(context, RuntimeSourceContext):
        return context
    if not isinstance(context, Mapping):
        raise ConfigError(f"{path} must be a RuntimeSourceContext or mapping.")
    return RuntimeSourceContext.from_batch(context, path=path)


def _require_tensor_mapping(value: Any, path: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping of tensor names to torch.Tensor values.")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{path} keys must be non-empty strings.")
        if not isinstance(tensor, torch.Tensor):
            raise ConfigError(f"{path}.{name} must be a torch.Tensor.")
        result[name] = tensor
    return result


def _resolve_named_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str | None,
    *,
    declared: Sequence[str],
    namespace: str,
    path: str,
) -> torch.Tensor:
    if name is None:
        raise ConfigError(f'{path} source "{namespace}" must name a tensor.')
    if name not in declared:
        raise ConfigError(
            f'{path} source "{namespace}.{name}" is not declared in data.{namespace}.'
        )
    try:
        return tensors[name]
    except KeyError as exc:
        raise ConfigError(
            f'{path} source "{namespace}.{name}" is missing from runtime context.'
        ) from exc


def _resolve_output(outputs: Any, source: BindingSource, path: str) -> torch.Tensor:
    if source.is_single_output:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, Mapping):
            raise ConfigError(
                f'{path} source "outputs" requires a single tensor output; '
                "use outputs.<name> for dictionary outputs."
            )
        raise ConfigError(
            f'{path} source "outputs" requires a single tensor output, got {type(outputs).__name__}.'
        )

    if not isinstance(outputs, Mapping):
        raise ConfigError(f'{path} source "{source}" requires model outputs to be a dictionary.')
    if source.name not in outputs:
        raise ConfigError(f'{path} source "{source}" is missing from model outputs.')
    tensor = outputs[source.name]
    if not isinstance(tensor, torch.Tensor):
        raise ConfigError(f'{path} source "{source}" must be a torch.Tensor.')
    return tensor


def _convert_dtype(tensor: torch.Tensor, dtype: str | None, path: str) -> torch.Tensor:
    if dtype is None or dtype == "auto":
        return tensor
    try:
        torch_dtype = _DTYPES[dtype]
    except KeyError as exc:
        raise ConfigError(f"{path} is unsupported: {dtype}.") from exc
    return tensor.to(dtype=torch_dtype)
