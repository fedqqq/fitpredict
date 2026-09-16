"""Resolve typed experiment configs into immutable executable configs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
import importlib
from math import exp, floor
import re
from types import MappingProxyType
from pathlib import Path
from typing import Any

from fitpredict.config.errors import ConfigError
from fitpredict.config.loader import load_config
from fitpredict.config.schema import (
    BindingConfig,
    BindingSource,
    ComponentConfig,
    DataMetadata,
    ExperimentConfig,
    MetricConfig,
    SaveBestConfig,
    SchedulerConfig,
    TransformConfig,
)


REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}")

COMPONENT_CATEGORIES = frozenset(
    {"model", "loss", "optimizer", "scheduler", "metric", "transform"}
)
BUILTIN_SCHEDULER_STEP_ON: Mapping[str, str] = {
    "CosineAnnealingLR": "epoch",
    "CyclicLR": "batch",
    "OneCycleLR": "batch",
    "ReduceLROnPlateau": "metric",
    "StepLR": "epoch",
}


@dataclass(frozen=True)
class BuiltinComponent:
    module: str
    attr: str
    framework: str


@dataclass(frozen=True)
class ComponentResolution:
    """Resolved component plus details useful for diagnostics and logging."""

    category: str
    requested: str
    source: str
    component: Any
    module: str
    attr: str
    framework: str | None = None
    params: Mapping[str, Any] | None = None


DEFAULT_BUILTINS: Mapping[str, Mapping[str, BuiltinComponent]] = {
    "model": {
        "Linear": BuiltinComponent("torch.nn", "Linear", "PyTorch"),
        "Sequential": BuiltinComponent("torch.nn", "Sequential", "PyTorch"),
        "ModuleList": BuiltinComponent("torch.nn", "ModuleList", "PyTorch"),
    },
    "loss": {
        "BCELoss": BuiltinComponent("torch.nn", "BCELoss", "PyTorch"),
        "BCEWithLogitsLoss": BuiltinComponent("torch.nn", "BCEWithLogitsLoss", "PyTorch"),
        "CrossEntropyLoss": BuiltinComponent("torch.nn", "CrossEntropyLoss", "PyTorch"),
        "L1Loss": BuiltinComponent("torch.nn", "L1Loss", "PyTorch"),
        "MSELoss": BuiltinComponent("torch.nn", "MSELoss", "PyTorch"),
        "NLLLoss": BuiltinComponent("torch.nn", "NLLLoss", "PyTorch"),
    },
    "optimizer": {
        "Adam": BuiltinComponent("torch.optim", "Adam", "PyTorch"),
        "AdamW": BuiltinComponent("torch.optim", "AdamW", "PyTorch"),
        "RMSprop": BuiltinComponent("torch.optim", "RMSprop", "PyTorch"),
        "SGD": BuiltinComponent("torch.optim", "SGD", "PyTorch"),
    },
    "scheduler": {
        "CosineAnnealingLR": BuiltinComponent(
            "torch.optim.lr_scheduler", "CosineAnnealingLR", "PyTorch"
        ),
        "CyclicLR": BuiltinComponent("torch.optim.lr_scheduler", "CyclicLR", "PyTorch"),
        "OneCycleLR": BuiltinComponent("torch.optim.lr_scheduler", "OneCycleLR", "PyTorch"),
        "ReduceLROnPlateau": BuiltinComponent(
            "torch.optim.lr_scheduler", "ReduceLROnPlateau", "PyTorch"
        ),
        "StepLR": BuiltinComponent("torch.optim.lr_scheduler", "StepLR", "PyTorch"),
    },
    "metric": {
        "accuracy_score": BuiltinComponent(
            "sklearn.metrics", "accuracy_score", "scikit-learn"
        ),
    },
    "transform": {
        "argmax": BuiltinComponent("fitpredict.config.resolver", "argmax_transform", "fitpredict"),
        "sigmoid": BuiltinComponent("fitpredict.config.resolver", "sigmoid_transform", "fitpredict"),
        "softmax": BuiltinComponent("fitpredict.config.resolver", "softmax_transform", "fitpredict"),
        "threshold": BuiltinComponent(
            "fitpredict.config.resolver", "threshold_transform", "fitpredict"
        ),
    },
}


class ComponentResolver:
    """Resolve built-in short names and custom import paths.

    This layer is intentionally separate from config validation and
    ``resolve_config``. Calling it may import runtime frameworks or user code.
    """

    def __init__(
        self,
        builtins: Mapping[str, Mapping[str, BuiltinComponent | tuple[str, str, str]]] | None = None,
    ) -> None:
        self._builtins = _normalize_builtin_registry(builtins or DEFAULT_BUILTINS)

    def resolve(
        self,
        category: str,
        spec: str | ComponentConfig | Mapping[str, Any] | Any = None,
        *,
        name: str | None = None,
        path: str | None = None,
    ) -> ComponentResolution:
        """Resolve a component spec to a class/callable and resolution metadata."""

        category = _normalize_category(category)
        requested, params = _component_name_and_params(category, spec, name=name, path=path)
        if _is_custom_component_path(requested):
            component, module_name, attr_path = self._resolve_custom(category, requested)
            _validate_component_callable(category, requested, component)
            return ComponentResolution(
                category=category,
                requested=requested,
                source="custom",
                component=component,
                module=module_name,
                attr=attr_path,
                params=params,
            )

        component, target = self._resolve_builtin(category, requested)
        _validate_component_callable(category, requested, component)
        return ComponentResolution(
            category=category,
            requested=requested,
            source="built-in",
            component=component,
            module=target.module,
            attr=target.attr,
            framework=target.framework,
            params=params,
        )

    def instantiate(
        self,
        category_or_resolution: str | ComponentResolution,
        spec: str | ComponentConfig | Mapping[str, Any] | Any = None,
        *args: Any,
        **params: Any,
    ) -> Any:
        """Instantiate a resolved component with config params plus overrides."""

        resolution = (
            category_or_resolution
            if isinstance(category_or_resolution, ComponentResolution)
            else self.resolve(category_or_resolution, spec)
        )
        init_params = dict(resolution.params or {})
        init_params.update(params)
        try:
            return resolution.component(*args, **init_params)
        except TypeError as exc:
            raise ConfigError(
                f"could not instantiate {resolution.category} component "
                f"{resolution.requested!r}: {exc}"
            ) from exc

    def scheduler_step_on(self, spec: str | ComponentConfig | Mapping[str, Any] | Any) -> str | None:
        """Return known scheduler timing without importing framework modules."""

        name, _ = _component_name_and_params("scheduler", spec)
        return resolve_builtin_scheduler_step_on(name)

    def _resolve_builtin(self, category: str, name: str) -> tuple[Any, BuiltinComponent]:
        builtins = self._builtins[category]
        if name not in builtins:
            known = ", ".join(sorted(builtins)) or "none"
            raise ConfigError(
                f"unknown built-in {category} component {name!r}. "
                f"Use one of: {known}; or provide a Python import path."
            )
        target = builtins[name]
        try:
            module = importlib.import_module(target.module)
        except ModuleNotFoundError as exc:
            raise ConfigError(
                f"cannot resolve built-in {category} component {name!r}: "
                f"{target.framework} is not installed or cannot be imported "
                f"(needed module {target.module!r})."
            ) from exc
        except Exception as exc:
            raise ConfigError(
                f"cannot resolve built-in {category} component {name!r} from "
                f"{target.module}: {exc}"
            ) from exc
        try:
            return getattr(module, target.attr), target
        except AttributeError as exc:
            raise ConfigError(
                f"cannot resolve built-in {category} component {name!r}: "
                f"{target.module} has no attribute {target.attr!r}."
            ) from exc

    def _resolve_custom(self, category: str, path: str) -> tuple[Any, str, str]:
        if ":" in path:
            module_name, attr_path = _split_colon_import_path(category, path)
            module = _import_custom_module(category, path, module_name)
            return _traverse_attr_path(category, path, module, module_name, attr_path)

        return _resolve_dotted_import_path(category, path)


def argmax_transform(input: Any, dim: int = -1) -> Any:
    if hasattr(input, "argmax"):
        try:
            return input.argmax(dim=dim)
        except TypeError:
            return input.argmax(axis=dim)
    return _argmax_list(input, dim)


def sigmoid_transform(input: Any) -> Any:
    if hasattr(input, "sigmoid"):
        return input.sigmoid()
    array_result = _try_numpy_unary("sigmoid", input)
    if array_result is not None:
        return array_result
    return _map_nested(input, _sigmoid_number)


def softmax_transform(input: Any, dim: int = -1) -> Any:
    if hasattr(input, "softmax"):
        return input.softmax(dim=dim)
    array_result = _try_numpy_softmax(input, dim)
    if array_result is not None:
        return array_result
    return _softmax_list(input, dim)


def threshold_transform(input: Any, value: float = 0.5) -> Any:
    try:
        return input >= value
    except TypeError:
        return _map_nested(input, lambda item: item >= value)


def resolve_builtin_scheduler_step_on(name: str) -> str | None:
    """Return known built-in scheduler step timing without importing components."""

    if _is_custom_component_path(name):
        return None
    return BUILTIN_SCHEDULER_STEP_ON.get(name)


def _normalize_builtin_registry(
    raw: Mapping[str, Mapping[str, BuiltinComponent | tuple[str, str, str]]],
) -> dict[str, dict[str, BuiltinComponent]]:
    normalized: dict[str, dict[str, BuiltinComponent]] = {}
    for category in COMPONENT_CATEGORIES:
        normalized[category] = {}
        for name, target in raw.get(category, {}).items():
            if isinstance(target, BuiltinComponent):
                normalized[category][name] = target
            else:
                module, attr, framework = target
                normalized[category][name] = BuiltinComponent(module, attr, framework)
    return normalized


def _normalize_category(category: str) -> str:
    if category not in COMPONENT_CATEGORIES:
        raise ConfigError(
            f"component category must be one of: {', '.join(sorted(COMPONENT_CATEGORIES))}."
        )
    return category


def _component_name_and_params(
    category: str,
    spec: str | ComponentConfig | Mapping[str, Any] | Any = None,
    *,
    name: str | None = None,
    path: str | None = None,
) -> tuple[str, Mapping[str, Any]]:
    selected = path if path is not None else name
    if selected is not None:
        if spec is not None:
            raise ConfigError("component spec cannot be combined with name= or path=.")
        return _require_component_name(selected, category), {}

    if isinstance(spec, str):
        return _require_component_name(spec, category), {}
    if isinstance(spec, ComponentConfig):
        return spec.name, dict(spec.params)
    if isinstance(spec, Mapping):
        if category == "model" and "class" in spec:
            key = "class"
        elif "name" in spec:
            key = "name"
        elif "path" in spec:
            key = "path"
        else:
            raise ConfigError(f"{category} component spec must include name, path, or class.")
        params = spec.get("params", {})
        if not isinstance(params, Mapping):
            raise ConfigError(f"{category} component params must be a mapping.")
        return _require_component_name(spec[key], category), dict(params)
    if category == "model" and hasattr(spec, "class_path"):
        return _require_component_name(getattr(spec, "class_path"), category), dict(
            getattr(spec, "params", {})
        )
    if hasattr(spec, "name"):
        return _require_component_name(getattr(spec, "name"), category), dict(
            getattr(spec, "params", {})
        )
    raise ConfigError(f"{category} component spec must be a string, mapping, or component config.")


def _require_component_name(value: Any, category: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{category} component name must be a non-empty string.")
    return value


def _validate_component_callable(category: str, requested: str, component: Any) -> None:
    if not callable(component):
        raise ConfigError(
            f"resolved {category} component {requested!r} is not callable "
            f"(got {type(component).__name__})."
        )


def _is_custom_component_path(name: str) -> bool:
    return ":" in name or "." in name


def _split_colon_import_path(category: str, path: str) -> tuple[str, str]:
    if path.count(":") != 1:
        raise ConfigError(
            f"{category} component import path {path!r} must contain exactly one ':'."
        )
    module_name, attr_path = path.split(":", 1)
    if not module_name or not attr_path or attr_path.startswith(".") or attr_path.endswith("."):
        raise ConfigError(
            f"{category} component import path {path!r} must use 'module:attribute'."
        )
    return module_name, attr_path


def _import_custom_module(category: str, requested: str, module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            raise ConfigError(
                f"cannot resolve custom {category} component {requested!r}: "
                f"module {module_name!r} could not be imported."
            ) from exc
        raise ConfigError(
            f"cannot resolve custom {category} component {requested!r}: "
            f"module {module_name!r} imports missing dependency {exc.name!r}."
        ) from exc
    except Exception as exc:
        raise ConfigError(
            f"cannot resolve custom {category} component {requested!r}: "
            f"module {module_name!r} raised {type(exc).__name__}: {exc}"
        ) from exc


def _resolve_dotted_import_path(category: str, path: str) -> tuple[Any, str, str]:
    parts = path.split(".")
    if len(parts) < 2 or any(part == "" for part in parts):
        raise ConfigError(
            f"{category} component import path {path!r} must use 'module.attribute'."
        )

    module_errors: list[str] = []
    attr_errors: list[str] = []
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        attr_path = ".".join(parts[index:])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                module_errors.append(module_name)
                continue
            raise ConfigError(
                f"cannot resolve custom {category} component {path!r}: "
                f"module {module_name!r} imports missing dependency {exc.name!r}."
            ) from exc
        except Exception as exc:
            raise ConfigError(
                f"cannot resolve custom {category} component {path!r}: "
                f"module {module_name!r} raised {type(exc).__name__}: {exc}"
            ) from exc

        try:
            return _traverse_attr_path(category, path, module, module_name, attr_path)
        except ConfigError as exc:
            attr_errors.append(str(exc))

    detail = attr_errors[-1] if attr_errors else f"no importable module prefix found in {path!r}"
    if module_errors and not attr_errors:
        detail = f"module {module_errors[0]!r} could not be imported"
    raise ConfigError(f"cannot resolve custom {category} component {path!r}: {detail}.")


def _traverse_attr_path(
    category: str,
    requested: str,
    module: Any,
    module_name: str,
    attr_path: str,
) -> tuple[Any, str, str]:
    current = module
    traversed: list[str] = []
    for attr in attr_path.split("."):
        if not attr:
            raise ConfigError(
                f"{category} component import path {requested!r} has an empty attribute segment."
            )
        traversed.append(attr)
        try:
            current = getattr(current, attr)
        except AttributeError as exc:
            dotted_attr = ".".join(traversed)
            raise ConfigError(
                f"{module_name!r} has no attribute {dotted_attr!r}"
            ) from exc
    return current, module_name, attr_path


def _try_numpy_unary(name: str, input: Any) -> Any:
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        return None
    if not hasattr(input, "__array__"):
        return None
    if name == "sigmoid":
        return 1 / (1 + numpy.exp(-input))
    return getattr(numpy, name)(input)


def _try_numpy_softmax(input: Any, dim: int) -> Any:
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        return None
    if not hasattr(input, "__array__"):
        return None
    shifted = input - numpy.max(input, axis=dim, keepdims=True)
    exponentials = numpy.exp(shifted)
    return exponentials / numpy.sum(exponentials, axis=dim, keepdims=True)


def _map_nested(value: Any, fn: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_map_nested(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_nested(item, fn) for item in value]
    return fn(value)


def _sigmoid_number(value: Any) -> float:
    return 1 / (1 + exp(-value))


def _argmax_list(input: Any, dim: int) -> Any:
    if not isinstance(input, (list, tuple)):
        raise ConfigError("built-in transform 'argmax' requires tensor-like or list input.")
    if not input:
        raise ConfigError("built-in transform 'argmax' cannot operate on an empty input.")
    if dim in {-1, 1} and all(isinstance(row, (list, tuple)) for row in input):
        return [_argmax_1d(row) for row in input]
    if dim in {0, -2} and all(isinstance(row, (list, tuple)) for row in input):
        row_lengths = {len(row) for row in input}
        if len(row_lengths) != 1:
            raise ConfigError("built-in transform 'argmax' requires rectangular list input.")
        return [_argmax_1d([row[index] for row in input]) for index in range(len(input[0]))]
    if dim in {-1, 0}:
        return _argmax_1d(input)
    raise ConfigError(f"built-in transform 'argmax' does not support dim={dim} for list input.")


def _argmax_1d(values: Any) -> int:
    if not values:
        raise ConfigError("built-in transform 'argmax' cannot operate on an empty input.")
    return max(range(len(values)), key=lambda index: values[index])


def _softmax_list(input: Any, dim: int) -> Any:
    if dim in {-1, 1} and isinstance(input, (list, tuple)) and input and all(
        isinstance(row, (list, tuple)) for row in input
    ):
        return [_softmax_1d(row) for row in input]
    if dim in {-1, 0} and isinstance(input, (list, tuple)):
        return _softmax_1d(input)
    raise ConfigError(f"built-in transform 'softmax' does not support dim={dim} for list input.")


def _softmax_1d(values: Any) -> list[float]:
    if not values:
        raise ConfigError("built-in transform 'softmax' cannot operate on an empty input.")
    max_value = max(values)
    exponentials = [exp(value - max_value) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _call_torch_function(name: str, *args: Any, **kwargs: Any) -> Any:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise ConfigError(f"built-in transform {name!r} requires PyTorch.") from exc
    return getattr(torch, name)(*args, **kwargs)


def _call_tensor_method(name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
    method = getattr(value, name, None)
    if method is None:
        raise ConfigError(f"built-in transform {name!r} requires an input with .{name}().")
    return method(*args, **kwargs)


class FrozenDict(Mapping[str, Any]):
    """Small immutable mapping used to deep-freeze resolved configs."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)
        self._proxy = MappingProxyType(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._proxy[key]

    def __iter__(self):
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def __repr__(self) -> str:
        return repr(self._proxy)


def resolve_config(
    config: ExperimentConfig | str | Path | Mapping[str, Any],
    *,
    data_metadata: DataMetadata | Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Validate, enrich, resolve references, and deep-freeze a config.

    ``data_metadata`` is supplied by the data-loading layer. The resolver never
    reads data files or imports experiment components.
    """

    typed = config if isinstance(config, ExperimentConfig) else load_config(config)
    enriched = _apply_metadata_and_derived_values(typed, data_metadata)
    resolved = _ReferenceResolver(enriched).resolve()
    resolved = _apply_post_reference_defaults(resolved)
    _validate_resolved_config(resolved)
    return _deep_freeze(resolved)


def _apply_metadata_and_derived_values(
    config: ExperimentConfig,
    metadata: DataMetadata | Mapping[str, Any] | None,
) -> ExperimentConfig:
    selected_metadata = metadata if metadata is not None else config.data.metadata
    data_metadata = _normalize_metadata(selected_metadata)
    _validate_metadata(config, data_metadata)

    train_size, val_size, test_size = _split_sizes(
        data_metadata.num_rows,
        (
            config.data.split.train,
            config.data.split.val,
            config.data.split.test,
        ),
    )
    steps_per_epoch = _ceil_div(train_size, config.training.batch_size)
    total_steps = steps_per_epoch * config.training.epochs

    data = replace(
        config.data,
        metadata=data_metadata,
        num_features=len(config.data.features),
        num_targets=len(config.data.targets),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    training = replace(
        config.training,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )
    return replace(config, data=data, training=training)


def _normalize_metadata(metadata: DataMetadata | Mapping[str, Any] | None) -> DataMetadata:
    if metadata is None:
        raise ConfigError(
            "data_metadata with num_rows and columns is required to resolve config."
        )
    if isinstance(metadata, DataMetadata):
        raw_metadata: Mapping[str, Any] = {
            "columns": metadata.columns,
            "feature_dtypes": metadata.feature_dtypes,
            "target_dtypes": metadata.target_dtypes,
            "feature_shapes": metadata.feature_shapes,
            "target_shapes": metadata.target_shapes,
            "num_rows": metadata.num_rows,
        }
    elif isinstance(metadata, Mapping):
        raw_metadata = metadata
    else:
        raise ConfigError("data_metadata must be a mapping or DataMetadata.")

    allowed = {
        "columns",
        "feature_dtypes",
        "target_dtypes",
        "feature_shapes",
        "target_shapes",
        "num_rows",
        "features",
        "targets",
        "num_features",
        "num_targets",
    }
    unknown = sorted(set(raw_metadata) - allowed)
    if unknown:
        raise ConfigError(f"data_metadata contains unknown field(s): {', '.join(unknown)}.")

    if "num_rows" not in raw_metadata or raw_metadata.get("num_rows") is None:
        raise ConfigError("data_metadata.num_rows is required to resolve config.")
    if "columns" not in raw_metadata:
        raise ConfigError("data_metadata.columns is required to resolve config.")

    num_rows = raw_metadata.get("num_rows")
    if num_rows is not None and (
        not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows < 0
    ):
        raise ConfigError("data_metadata.num_rows must be a non-negative integer.")

    columns = _string_tuple(raw_metadata.get("columns"), "data_metadata.columns")
    if not columns:
        raise ConfigError("data_metadata.columns must not be empty.")

    return DataMetadata(
        columns=columns,
        feature_dtypes=_string_mapping(raw_metadata.get("feature_dtypes"), "data_metadata.feature_dtypes"),
        target_dtypes=_string_mapping(raw_metadata.get("target_dtypes"), "data_metadata.target_dtypes"),
        feature_shapes=_shape_mapping(raw_metadata.get("feature_shapes"), "data_metadata.feature_shapes"),
        target_shapes=_shape_mapping(raw_metadata.get("target_shapes"), "data_metadata.target_shapes"),
        num_rows=num_rows,
    )


def _validate_metadata(config: ExperimentConfig, metadata: DataMetadata) -> None:
    columns = set(metadata.columns)
    if columns:
        missing_features = sorted(set(config.data.features) - columns)
        missing_targets = sorted(set(config.data.targets) - columns)
        if missing_features:
            raise ConfigError(
                "data_metadata.columns is missing configured feature(s): "
                + ", ".join(missing_features)
            )
        if missing_targets:
            raise ConfigError(
                "data_metadata.columns is missing configured target(s): "
                + ", ".join(missing_targets)
            )


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise ConfigError(f"{path} must be a list of strings.")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise ConfigError(f"{path} must be a list of non-empty strings.")
    return result


def _mapping_or_empty(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping.")
    return value


def _string_mapping(value: Any, path: str) -> dict[str, str]:
    raw = _mapping_or_empty(value, path)
    result: dict[str, str] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{path} keys must be non-empty strings.")
        if not isinstance(item, str) or not item:
            raise ConfigError(f"{path}.{key} must be a non-empty string.")
        result[key] = item
    return result


def _shape_mapping(value: Any, path: str) -> dict[str, tuple[int, ...]]:
    raw = _mapping_or_empty(value, path)
    result: dict[str, tuple[int, ...]] = {}
    for name, shape in raw.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{path} keys must be non-empty strings.")
        if not isinstance(shape, Iterable) or isinstance(shape, (str, bytes, Mapping)):
            raise ConfigError(f"{path}.{name} must be a list of non-negative integers.")
        normalized = tuple(shape)
        if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in normalized):
            raise ConfigError(f"{path}.{name} must be a list of non-negative integers.")
        result[name] = normalized
    return result


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


def _ceil_div(value: int, divisor: int) -> int:
    if value == 0:
        return 0
    return (value + divisor - 1) // divisor


class _ReferenceResolver:
    def __init__(self, root: ExperimentConfig) -> None:
        self.root = root
        self._resolving: set[str] = set()
        self._resolved_paths: dict[str, Any] = {}

    def resolve(self) -> ExperimentConfig:
        return self._resolve_value(self.root, "config")

    def _resolve_value(self, value: Any, path: str) -> Any:
        if isinstance(value, str):
            dotted_path = _config_path_to_dotted_path(path)
            if dotted_path is None or dotted_path in self._resolving:
                return self._resolve_string(value, path)
            self._resolving.add(dotted_path)
            try:
                resolved = self._resolve_string(value, path)
                self._resolved_paths[dotted_path] = resolved
                return resolved
            finally:
                self._resolving.remove(dotted_path)
        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field.name: self._resolve_value(getattr(value, field.name), f"{path}.{field.name}")
                for field in fields(value)
            }
            return replace(value, **updates)
        if isinstance(value, Mapping):
            return {
                key: self._resolve_value(item, f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._resolve_value(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                self._resolve_value(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        return value

    def _resolve_string(self, value: str, path: str) -> Any:
        matches = list(REFERENCE_RE.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return self._resolve_path(matches[0].group(1))

        def replace_match(match: re.Match[str]) -> str:
            return str(self._resolve_path(match.group(1)))

        return REFERENCE_RE.sub(replace_match, value)

    def _resolve_path(self, dotted_path: str) -> Any:
        if dotted_path in self._resolved_paths:
            return self._resolved_paths[dotted_path]
        if dotted_path in self._resolving:
            raise ConfigError(f"reference cycle detected at ${{{dotted_path}}}.")
        self._resolving.add(dotted_path)
        try:
            value = self._lookup_path(dotted_path)
            resolved = self._resolve_value(value, dotted_path)
            self._resolved_paths[dotted_path] = resolved
            return resolved
        finally:
            self._resolving.remove(dotted_path)

    def _lookup_path(self, dotted_path: str) -> Any:
        current: Any = self.root
        traversed: list[str] = []
        for part in dotted_path.split("."):
            if part.startswith("_"):
                raise ConfigError(f"config reference ${{{dotted_path}}} uses non-public path segment.")
            traversed.append(part)
            if is_dataclass(current) and not isinstance(current, type):
                field_names = {field.name for field in fields(current)}
                field_name = "class_path" if part == "class" and "class_path" in field_names else part
                if field_name not in field_names:
                    raise ConfigError(f"unknown config reference ${{{dotted_path}}}.")
                current = getattr(current, field_name)
            elif isinstance(current, Mapping):
                if part not in current:
                    raise ConfigError(f"unknown config reference ${{{dotted_path}}}.")
                current = current[part]
            else:
                joined = ".".join(traversed[:-1])
                raise ConfigError(f"config reference ${{{dotted_path}}} cannot traverse {joined}.")
        return current


def _apply_post_reference_defaults(config: ExperimentConfig) -> ExperimentConfig:
    scheduler = config.training.scheduler
    if scheduler is None:
        return config

    if not isinstance(scheduler.name, str) or not scheduler.name:
        raise ConfigError("training.scheduler.name must be a non-empty string after reference resolution.")

    expected_step_on = resolve_builtin_scheduler_step_on(scheduler.name)
    step_on = scheduler.step_on
    if expected_step_on is not None:
        if step_on is None:
            step_on = expected_step_on
        elif step_on != expected_step_on:
            raise ConfigError(
                f"training.scheduler.step_on for {scheduler.name} must be {expected_step_on}."
            )
    elif "." in scheduler.name and step_on is None:
        raise ConfigError("training.scheduler.step_on is required for custom schedulers.")

    if step_on == "metric" and scheduler.monitor is None:
        raise ConfigError("training.scheduler.monitor is required when step_on is metric.")

    resolved_scheduler = replace(scheduler, step_on=step_on)
    return replace(config, training=replace(config.training, scheduler=resolved_scheduler))


def _validate_resolved_config(config: ExperimentConfig) -> None:
    _require_instance(config.data.path, Path, "data.path")
    _require_non_empty_string(config.data.format, "data.format")
    _require_string_sequence(config.data.features, "data.features", allow_empty=False)
    _require_string_sequence(config.data.targets, "data.targets", allow_empty=False)
    _require_non_negative_int(config.data.num_features, "data.num_features")
    _require_non_negative_int(config.data.num_targets, "data.num_targets")
    _require_non_negative_int(config.data.train_size, "data.train_size")
    _require_non_negative_int(config.data.val_size, "data.val_size")
    _require_non_negative_int(config.data.test_size, "data.test_size")
    if config.data.train_size + config.data.val_size + config.data.test_size != config.data.metadata.num_rows:
        raise ConfigError("resolved split sizes must sum to data_metadata.num_rows.")

    _require_non_empty_string(config.model.class_path, "model.class")
    _require_mapping_instance(config.model.params, "model.params")
    if config.model.weights is not None:
        _require_instance(config.model.weights, Path, "model.weights")

    _require_positive_int_value(config.training.epochs, "training.epochs")
    _require_positive_int_value(config.training.batch_size, "training.batch_size")
    _validate_component_config(config.training.optimizer, "training.optimizer")
    for index, objective in enumerate(config.training.objectives):
        _validate_component_config(objective.loss, f"training.objectives[{index}].loss")
        _validate_bindings(objective.bindings, config, f"training.objectives[{index}].bindings")
        if not isinstance(objective.weight, (int, float)) or isinstance(objective.weight, bool):
            raise ConfigError(f"training.objectives[{index}].weight must be a finite number.")

    _validate_scheduler(config.training.scheduler)
    _validate_bindings(config.model.inputs, config, "model.inputs")
    for index, metric in enumerate(config.evaluation.metrics):
        _validate_metric_config(metric, config, f"evaluation.metrics[{index}]")
    _validate_save_best(config.saving.save_best, config)


def _validate_component_config(component: ComponentConfig, path: str) -> None:
    _require_non_empty_string(component.name, f"{path}.name")
    _require_mapping_instance(component.params, f"{path}.params")


def _validate_scheduler(scheduler: SchedulerConfig | None) -> None:
    if scheduler is None:
        return
    _require_non_empty_string(scheduler.name, "training.scheduler.name")
    if scheduler.step_on not in {"batch", "epoch", "metric"}:
        raise ConfigError("training.scheduler.step_on must be one of: batch, epoch, metric.")
    if scheduler.step_on == "metric":
        _require_non_empty_string(scheduler.monitor, "training.scheduler.monitor")


def _validate_metric_config(metric: MetricConfig, config: ExperimentConfig, path: str) -> None:
    _require_non_empty_string(metric.name, f"{path}.name")
    _require_mapping_instance(metric.params, f"{path}.params")
    _validate_bindings(metric.bindings, config, f"{path}.bindings")
    for index, transform in enumerate(metric.transform):
        _validate_transform_config(transform, config, f"{path}.transform[{index}]")


def _validate_transform_config(transform: TransformConfig, config: ExperimentConfig, path: str) -> None:
    _require_non_empty_string(transform.name, f"{path}.name")
    _require_mapping_instance(transform.params, f"{path}.params")
    _validate_bindings(transform.bindings, config, f"{path}.bindings")


def _validate_bindings(
    bindings: Mapping[str, BindingConfig],
    config: ExperimentConfig,
    path: str,
) -> None:
    _require_mapping_instance(bindings, path)
    for name, binding in bindings.items():
        _require_non_empty_string(name, f"{path} key")
        if not isinstance(binding, BindingConfig):
            raise ConfigError(f"{path}.{name} must be a BindingConfig.")
        _validate_binding_source(binding.source, config, f"{path}.{name}.source")


def _validate_binding_source(source: BindingSource, config: ExperimentConfig, path: str) -> None:
    if not isinstance(source, BindingSource):
        raise ConfigError(f"{path} must be a BindingSource.")
    if source.namespace == "features" and source.name is not None:
        if source.name not in config.data.features:
            raise ConfigError(f"{path} references unknown feature {source.name!r}.")
    if source.namespace == "targets":
        if source.name not in config.data.targets:
            raise ConfigError(f"{path} references unknown target {source.name!r}.")


def _validate_save_best(save_best: SaveBestConfig | None, config: ExperimentConfig) -> None:
    if save_best is None:
        return
    _require_non_empty_string(save_best.monitor, "saving.save_best.monitor")
    if save_best.monitor.startswith("test."):
        raise ConfigError("saving.save_best.monitor cannot reference test metrics.")
    if not save_best.monitor.startswith("val."):
        raise ConfigError("saving.save_best.monitor must reference val.loss or a configured val metric.")
    metric_name = save_best.monitor.removeprefix("val.")
    if metric_name == "loss":
        return
    configured_metrics = {metric.name for metric in config.evaluation.metrics}
    if metric_name not in configured_metrics:
        raise ConfigError(
            f"saving.save_best.monitor references unknown validation metric {metric_name!r}."
        )


def _require_instance(value: Any, expected_type: type, path: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigError(f"{path} must be {expected_type.__name__} after reference resolution.")


def _require_non_empty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string after reference resolution.")


def _require_mapping_instance(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping after reference resolution.")


def _require_string_sequence(value: Any, path: str, *, allow_empty: bool) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{path} must be a sequence of strings after reference resolution.")
    if not allow_empty and not value:
        raise ConfigError(f"{path} must not be empty after reference resolution.")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{path} must contain only non-empty strings after reference resolution.")


def _require_non_negative_int(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be a non-negative integer after reference resolution.")


def _require_positive_int_value(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be an integer greater than 0 after reference resolution.")


def _deep_freeze(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: _deep_freeze(getattr(value, field.name)) for field in fields(value)}
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _config_path_to_dotted_path(path: str) -> str | None:
    if not path.startswith("config.") or "[" in path:
        return None
    return path.removeprefix("config.")
