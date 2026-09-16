"""Convert binding configs and runtime tensors into callable keyword arguments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import BindingConfig
from fitpredict.data.source_resolver import RuntimeSourceContext, SourceResolver


class BindingEngine:
    """Resolve a callable's binding mapping into ``callable(**kwargs)`` inputs."""

    def __init__(
        self,
        context: RuntimeSourceContext | Mapping[str, Any],
        *,
        features: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
        path: str = "runtime_context",
    ) -> None:
        self._resolver = SourceResolver(
            context,
            features=features,
            targets=targets,
            path=path,
        )

    def build_kwargs(
        self,
        bindings: Mapping[str, BindingConfig],
        *,
        path: str = "bindings",
    ) -> dict[str, torch.Tensor]:
        """Resolve every binding into a kwargs dictionary.

        The returned tensors are the resolver outputs directly. They are not
        detached, so autograd relationships from model outputs are preserved.
        """

        resolved_bindings = _require_binding_mapping(bindings, path)
        return {
            argument_name: self._resolver.resolve(
                binding,
                path=f"{path}.{argument_name}",
            )
            for argument_name, binding in resolved_bindings.items()
        }


def build_bound_kwargs(
    bindings: Mapping[str, BindingConfig],
    context: RuntimeSourceContext | Mapping[str, Any],
    *,
    features: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    path: str = "bindings",
    context_path: str = "runtime_context",
) -> dict[str, torch.Tensor]:
    """Resolve bindings without explicitly constructing ``BindingEngine``."""

    return BindingEngine(
        context,
        features=features,
        targets=targets,
        path=context_path,
    ).build_kwargs(bindings, path=path)


def _require_binding_mapping(
    bindings: Mapping[str, BindingConfig],
    path: str,
) -> Mapping[str, BindingConfig]:
    if not isinstance(bindings, Mapping):
        raise ConfigError(f"{path} must be a mapping of argument names to BindingConfig values.")

    result: dict[str, BindingConfig] = {}
    for argument_name, binding in bindings.items():
        if not isinstance(argument_name, str) or not argument_name:
            raise ConfigError(f"{path} keys must be non-empty argument names.")
        if not isinstance(binding, BindingConfig):
            raise ConfigError(f"{path}.{argument_name} must be a BindingConfig.")
        result[argument_name] = binding
    return result
