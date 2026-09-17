"""Typed configuration schema for fitpredict v1.

This module intentionally performs only schema-level validation. Checks that
need loaded data, imported components, or runtime model outputs belong to the
future resolver/data/binding layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from pathlib import Path
from typing import Any, ClassVar, Mapping

from fitpredict.config.errors import ConfigError


SUPPORTED_DATA_FORMATS = frozenset({"csv", "json", "jsonl", "parquet", "feather"})
SUPPORTED_DTYPES = frozenset({"bool", "int64", "float16", "float32", "float64", "auto"})
SUPPORTED_DEVICES = frozenset({"cpu", "cuda", "mps", "auto"})
SCHEDULER_STEP_ON = frozenset({"batch", "epoch", "metric"})
KNOWN_SCHEDULER_STEP_ON = {
    "ReduceLROnPlateau": "metric",
    "OneCycleLR": "batch",
    "CyclicLR": "batch",
}
SAVE_BEST_MODES = frozenset({"min", "max"})


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping.")
    return value


def _optional_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping.")
    return dict(value)


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string.")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, path)


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ConfigError(f"{path} must be a finite number.")
    return float(value)


def _require_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be an integer greater than 0.")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean.")
    return value


def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list of strings.")
    if not allow_empty and not value:
        raise ConfigError(f"{path} must not be empty.")
    result = [_require_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicate names.")
    return result


def _ensure_no_unknown_keys(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        joined = ", ".join(f"{path}.{key}" for key in unknown)
        raise ConfigError(f"{path} contains unknown field(s): {joined}.")


@dataclass(frozen=True, eq=False)
class BindingSource:
    """A parsed binding source from the common Binding Protocol.

    Supported user-defined source strings:
    - ``features``: aggregate all declared data features into one tensor.
    - ``features.<name>``: one declared feature column.
    - ``targets.<name>``: one declared target column.
    - ``outputs.<name>``: one named model output.
    - ``outputs``: the model's single unnamed tensor output.
    """

    raw: str
    namespace: str
    name: str | None = None

    VALID_NAMESPACES: ClassVar[frozenset[str]] = frozenset({"features", "targets", "outputs"})

    @classmethod
    def parse(cls, value: Any, path: str = "source") -> "BindingSource":
        raw = _require_string(value, path)
        if "," in raw or "[" in raw or "]" in raw:
            raise ConfigError(f"{path} must refer to exactly one source; lists are not supported.")

        parts = raw.split(".")
        if len(parts) > 2 or any(part == "" for part in parts):
            raise ConfigError(f"{path} has invalid binding source syntax: {raw}.")

        namespace = parts[0]
        if namespace not in cls.VALID_NAMESPACES:
            raise ConfigError(f"{path} must start with one of: features, targets, outputs.")

        name = parts[1] if len(parts) == 2 else None
        if namespace == "targets" and name is None:
            raise ConfigError(f'{path} "targets" must name a target column.')
        if namespace == "features" and name is None:
            return cls(raw=raw, namespace=namespace, name=None)
        if namespace == "outputs" and name is None:
            return cls(raw=raw, namespace=namespace, name=None)
        return cls(raw=raw, namespace=namespace, name=name)

    @property
    def is_feature_aggregate(self) -> bool:
        return self.namespace == "features" and self.name is None

    @property
    def is_single_output(self) -> bool:
        return self.namespace == "outputs" and self.name is None

    def __str__(self) -> str:
        return self.raw

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BindingSource):
            return self.raw == other.raw
        if isinstance(other, str):
            return self.raw == other
        return False

    def __hash__(self) -> int:
        return hash(self.raw)


@dataclass(frozen=True)
class BindingConfig:
    """Binding for one callable argument."""

    source: BindingSource
    dtype: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> "BindingConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"source", "dtype"}, path)
        if "source" not in data:
            raise ConfigError(f"{path}.source is required.")
        dtype = _optional_string(data.get("dtype"), f"{path}.dtype")
        if dtype is not None and dtype not in SUPPORTED_DTYPES:
            raise ConfigError(f"{path}.dtype is unsupported: {dtype}.")
        return cls(source=BindingSource.parse(data["source"], f"{path}.source"), dtype=dtype)


def _bindings_mapping(value: Any, path: str, *, required: bool) -> dict[str, BindingConfig]:
    if value is None:
        if required:
            raise ConfigError(f"{path} is required.")
        return {}
    raw = _require_mapping(value, path)
    if not raw and required:
        raise ConfigError(f"{path} must not be empty.")
    return {
        _require_string(name, f"{path} key"): BindingConfig.from_mapping(binding, f"{path}.{name}")
        for name, binding in raw.items()
    }


@dataclass(frozen=True)
class ComponentConfig:
    """Built-in short component name or user-defined Python import path."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> "ComponentConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"name", "params"}, path)
        if "name" not in data:
            raise ConfigError(f"{path}.name is required.")
        return cls(
            name=_require_string(data["name"], f"{path}.name"),
            params=_optional_mapping(data.get("params"), f"{path}.params"),
        )


@dataclass(frozen=True)
class TransformConfig:
    """Metric/output transform component, optionally using the Binding Protocol."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, BindingConfig] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> "TransformConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"name", "params", "bindings"}, path)
        if "name" not in data:
            raise ConfigError(f"{path}.name is required.")
        return cls(
            name=_require_string(data["name"], f"{path}.name"),
            params=_optional_mapping(data.get("params"), f"{path}.params"),
            bindings=_bindings_mapping(data.get("bindings"), f"{path}.bindings", required=False),
        )


@dataclass(frozen=True)
class SplitConfig:
    """User-defined data split semantics."""

    train: float
    val: float
    test: float
    shuffle: bool = True
    seed: int | None = None
    stratify: str | None = None
    sort_by: str | None = None
    ascending: bool = True

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "data.split") -> "SplitConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(
            data,
            {"train", "val", "test", "shuffle", "seed", "stratify", "sort_by", "ascending"},
            path,
        )
        for key in ("train", "val", "test"):
            if key not in data:
                raise ConfigError(f"{path}.{key} is required.")
        train = _require_number(data["train"], f"{path}.train")
        val = _require_number(data["val"], f"{path}.val")
        test = _require_number(data["test"], f"{path}.test")
        if train < 0 or val < 0 or test < 0:
            raise ConfigError(f"{path} fractions must be non-negative.")
        if not isclose(train + val + test, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError(f"{path} fractions must sum to 1.")

        seed = data.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ConfigError(f"{path}.seed must be an integer when provided.")

        return cls(
            train=train,
            val=val,
            test=test,
            shuffle=_require_bool(data.get("shuffle", True), f"{path}.shuffle"),
            seed=seed,
            stratify=_optional_string(data.get("stratify"), f"{path}.stratify"),
            sort_by=_optional_string(data.get("sort_by"), f"{path}.sort_by"),
            ascending=_require_bool(data.get("ascending", True), f"{path}.ascending"),
        )


@dataclass(frozen=True)
class DataMetadata:
    """Derived fields filled after loading data metadata."""

    columns: tuple[str, ...] = ()
    feature_dtypes: dict[str, str] = field(default_factory=dict)
    target_dtypes: dict[str, str] = field(default_factory=dict)
    feature_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    target_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    num_rows: int | None = None


@dataclass(frozen=True)
class DataConfig:
    """Data section.

    User-defined fields: path, format, options, features, targets, split.
    Derived fields: metadata, num_features, num_targets, train_size, val_size,
    test_size.
    """

    path: Path
    format: str
    features: tuple[str, ...]
    targets: tuple[str, ...]
    split: SplitConfig
    options: dict[str, Any] = field(default_factory=dict)
    metadata: DataMetadata = field(default_factory=DataMetadata)
    num_features: int | None = None
    num_targets: int | None = None
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "data") -> "DataConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(
            data,
            {
                "path",
                "format",
                "options",
                "features",
                "targets",
                "split",
            },
            path,
        )
        for key in ("path", "format", "features", "targets", "split"):
            if key not in data:
                raise ConfigError(f"{path}.{key} is required.")

        data_format = _require_string(data["format"], f"{path}.format").lower()
        if data_format not in SUPPORTED_DATA_FORMATS:
            raise ConfigError(
                f"{path}.format is unsupported: {data_format}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_DATA_FORMATS))}."
            )

        features = _string_list(data["features"], f"{path}.features")
        targets = _string_list(data["targets"], f"{path}.targets")
        return cls(
            path=Path(_require_string(data["path"], f"{path}.path")),
            format=data_format,
            options=_optional_mapping(data.get("options"), f"{path}.options"),
            features=tuple(features),
            targets=tuple(targets),
            split=SplitConfig.from_mapping(data["split"], f"{path}.split"),
            num_features=len(features),
            num_targets=len(targets),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model section.

    User-defined fields: class, params, weights, inputs. The ``class`` field is
    exposed as ``class_path`` because ``class`` is a Python keyword.
    """

    class_path: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, BindingConfig] = field(default_factory=dict)
    weights: Path | None = None

    @property
    def class_(self) -> str:
        """Compatibility alias for loader/resolver code that mirrors YAML."""

        return self.class_path

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "model") -> "ModelConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"class", "params", "weights", "inputs"}, path)
        if "class" not in data:
            raise ConfigError(f"{path}.class is required.")
        return cls(
            class_path=_require_string(data["class"], f"{path}.class"),
            params=_optional_mapping(data.get("params"), f"{path}.params"),
            weights=(
                Path(_require_string(data["weights"], f"{path}.weights"))
                if data.get("weights") is not None
                else None
            ),
            inputs=_bindings_mapping(data.get("inputs"), f"{path}.inputs", required=True),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """One weighted training objective."""

    loss: ComponentConfig
    bindings: dict[str, BindingConfig]
    weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> "ObjectiveConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"loss", "bindings", "weight"}, path)
        if "loss" not in data:
            raise ConfigError(f"{path}.loss is required.")
        return cls(
            loss=ComponentConfig.from_mapping(data["loss"], f"{path}.loss"),
            bindings=_bindings_mapping(data.get("bindings"), f"{path}.bindings", required=True),
            weight=_require_number(data.get("weight", 1.0), f"{path}.weight"),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """Optional scheduler component."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    step_on: str | None = None
    step_on_was_explicit: bool = False
    monitor: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "training.scheduler") -> "SchedulerConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"name", "params", "step_on", "monitor"}, path)
        if "name" not in data:
            raise ConfigError(f"{path}.name is required.")
        name = _require_string(data["name"], f"{path}.name")
        explicit_step_on = "step_on" in data
        step_on = (
            _require_string(data["step_on"], f"{path}.step_on")
            if explicit_step_on
            else KNOWN_SCHEDULER_STEP_ON.get(name)
        )
        if step_on is not None and step_on not in SCHEDULER_STEP_ON:
            raise ConfigError(f"{path}.step_on must be one of: batch, epoch, metric.")
        if "." in name and step_on is None:
            raise ConfigError(f"{path}.step_on is required for custom schedulers.")
        expected_step_on = KNOWN_SCHEDULER_STEP_ON.get(name)
        if expected_step_on is not None and step_on != expected_step_on:
            raise ConfigError(f"{path}.step_on for {name} must be {expected_step_on}.")
        monitor = _optional_string(data.get("monitor"), f"{path}.monitor")
        if step_on == "metric" and monitor is None:
            raise ConfigError(f"{path}.monitor is required when step_on is metric.")
        return cls(
            name=name,
            params=_optional_mapping(data.get("params"), f"{path}.params"),
            step_on=step_on,
            step_on_was_explicit=explicit_step_on,
            monitor=monitor,
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Training section.

    User-defined fields: epochs, batch_size, shuffle, device, optimizer,
    scheduler, objectives.
    Derived fields: steps_per_epoch, total_steps.
    """

    epochs: int
    batch_size: int
    optimizer: ComponentConfig
    objectives: list[ObjectiveConfig]
    shuffle: bool = True
    device: str = "cpu"
    scheduler: SchedulerConfig | None = None
    steps_per_epoch: int | None = None
    total_steps: int | None = None

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "training") -> "TrainingConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(
            data,
            {
                "epochs",
                "batch_size",
                "shuffle",
                "device",
                "optimizer",
                "scheduler",
                "objectives",
            },
            path,
        )
        for key in ("epochs", "batch_size", "optimizer", "objectives"):
            if key not in data:
                raise ConfigError(f"{path}.{key} is required.")

        device = _require_string(data.get("device", "cpu"), f"{path}.device")
        if device not in SUPPORTED_DEVICES:
            raise ConfigError(f"{path}.device is unsupported: {device}.")

        objectives_raw = data["objectives"]
        if not isinstance(objectives_raw, list) or not objectives_raw:
            raise ConfigError(f"{path}.objectives must be a non-empty list.")

        return cls(
            epochs=_require_positive_int(data["epochs"], f"{path}.epochs"),
            batch_size=_require_positive_int(data["batch_size"], f"{path}.batch_size"),
            shuffle=_require_bool(data.get("shuffle", True), f"{path}.shuffle"),
            device=device,
            optimizer=ComponentConfig.from_mapping(data["optimizer"], f"{path}.optimizer"),
            scheduler=(
                SchedulerConfig.from_mapping(data["scheduler"], f"{path}.scheduler")
                if data.get("scheduler") is not None
                else None
            ),
            objectives=[
                ObjectiveConfig.from_mapping(item, f"{path}.objectives[{index}]")
                for index, item in enumerate(objectives_raw)
            ],
        )


@dataclass(frozen=True)
class MetricConfig:
    """Dataset-level metric component."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, BindingConfig] = field(default_factory=dict)
    transform: list[TransformConfig] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> "MetricConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"name", "params", "bindings", "transform"}, path)
        if "name" not in data:
            raise ConfigError(f"{path}.name is required.")
        transform_raw = data.get("transform", [])
        if not isinstance(transform_raw, list):
            raise ConfigError(f"{path}.transform must be a list.")
        return cls(
            name=_require_string(data["name"], f"{path}.name"),
            params=_optional_mapping(data.get("params"), f"{path}.params"),
            bindings=_bindings_mapping(data.get("bindings"), f"{path}.bindings", required=False),
            transform=[
                TransformConfig.from_mapping(item, f"{path}.transform[{index}]")
                for index, item in enumerate(transform_raw)
            ],
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation section."""

    metrics: list[MetricConfig] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: Any | None, path: str = "evaluation") -> "EvaluationConfig":
        if raw is None:
            return cls()
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"metrics"}, path)
        metrics_raw = data.get("metrics", [])
        if not isinstance(metrics_raw, list):
            raise ConfigError(f"{path}.metrics must be a list.")
        return cls(
            metrics=[
                MetricConfig.from_mapping(item, f"{path}.metrics[{index}]")
                for index, item in enumerate(metrics_raw)
            ]
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Optional logging infrastructure section."""

    console: bool = True
    tensorboard: bool = False
    mlflow: bool = False

    @classmethod
    def from_mapping(cls, raw: Any | None, path: str = "logging") -> "LoggingConfig":
        if raw is None:
            return cls()
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"console", "tensorboard", "mlflow"}, path)
        return cls(
            console=_require_bool(data.get("console", True), f"{path}.console"),
            tensorboard=_require_bool(data.get("tensorboard", False), f"{path}.tensorboard"),
            mlflow=_require_bool(data.get("mlflow", False), f"{path}.mlflow"),
        )


@dataclass(frozen=True)
class SaveBestConfig:
    """Best-checkpoint selection config."""

    monitor: str
    mode: str

    @classmethod
    def from_mapping(cls, raw: Any, path: str = "saving.save_best") -> "SaveBestConfig":
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"monitor", "mode"}, path)
        for key in ("monitor", "mode"):
            if key not in data:
                raise ConfigError(f"{path}.{key} is required.")
        mode = _require_string(data["mode"], f"{path}.mode")
        if mode not in SAVE_BEST_MODES:
            raise ConfigError(f"{path}.mode must be one of: min, max.")
        return cls(
            monitor=_require_string(data["monitor"], f"{path}.monitor"),
            mode=mode,
        )


@dataclass(frozen=True)
class SavingConfig:
    """Saving section."""

    save_last: bool = True
    save_best: SaveBestConfig | None = None
    output_dir: Path = Path("runs")

    @classmethod
    def from_mapping(cls, raw: Any | None, path: str = "saving") -> "SavingConfig":
        if raw is None:
            return cls()
        data = _require_mapping(raw, path)
        _ensure_no_unknown_keys(data, {"save_last", "save_best", "output_dir"}, path)
        return cls(
            save_last=_require_bool(data.get("save_last", True), f"{path}.save_last"),
            save_best=(
                SaveBestConfig.from_mapping(data["save_best"], f"{path}.save_best")
                if data.get("save_best") is not None
                else None
            ),
            output_dir=Path(_require_string(data.get("output_dir", "runs"), f"{path}.output_dir")),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level v1 config with the six public sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)

    @classmethod
    def from_mapping(cls, raw: Any) -> "ExperimentConfig":
        data = _require_mapping(raw, "config")
        _ensure_no_unknown_keys(
            data,
            {"data", "model", "training", "evaluation", "logging", "saving"},
            "config",
        )
        for section in ("data", "model", "training", "evaluation", "logging", "saving"):
            if section not in data:
                raise ConfigError(f"config.{section} section is required.")
        return cls(
            data=DataConfig.from_mapping(data["data"]),
            model=ModelConfig.from_mapping(data["model"]),
            training=TrainingConfig.from_mapping(data["training"]),
            evaluation=EvaluationConfig.from_mapping(data["evaluation"]),
            logging=LoggingConfig.from_mapping(data["logging"]),
            saving=SavingConfig.from_mapping(data["saving"]),
        )


def parse_experiment_config(raw: Any) -> ExperimentConfig:
    """Parse a raw mapping into the typed schema.

    This small wrapper is the stable API used by ConfigLoader and future
    resolver code.
    """

    return ExperimentConfig.from_mapping(raw)
