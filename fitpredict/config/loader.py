"""Load JSON or YAML user config into the typed schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fitpredict.config.schema import ConfigError, ExperimentConfig

YAML_SUFFIXES = {".yaml", ".yml"}
JSON_SUFFIXES = {".json"}


def load_config(source: str | Path | Mapping[str, Any]) -> ExperimentConfig:
    """Load config from an existing path, raw JSON/YAML string, or mapping."""

    if isinstance(source, Mapping):
        return ExperimentConfig.from_mapping(source)
    if isinstance(source, Path):
        return load_config_file(source)
    if isinstance(source, str):
        if _looks_like_config_content(source):
            return loads_config(source)
        path = Path(source)
        if path.suffix.lower() in JSON_SUFFIXES | YAML_SUFFIXES:
            return load_config_file(path)
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if exists:
            return load_config_file(path)
        return loads_config(source)
    raise ConfigError(f"config source must be a path, string, or mapping, got {type(source).__name__}")


def load_config_file(path: str | Path) -> ExperimentConfig:
    """Load config from a .json, .yaml, or .yml file."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc

    suffix = config_path.suffix.lower()
    if suffix in JSON_SUFFIXES:
        return loads_config(text, format="json", source_name=str(config_path))
    if suffix in YAML_SUFFIXES:
        return loads_config(text, format="yaml", source_name=str(config_path))
    raise ConfigError(
        f"unsupported config file extension {suffix!r}; expected .json, .yaml, or .yml"
    )


def loads_config(
    text: str,
    *,
    format: str | None = None,
    source_name: str = "config string",
) -> ExperimentConfig:
    """Load config from raw JSON or YAML text."""

    raw = _parse_text(text, format=format, source_name=source_name)
    return ExperimentConfig.from_mapping(raw)


def _looks_like_config_content(source: str) -> bool:
    stripped = source.lstrip()
    return (
        stripped.startswith("{")
        or stripped.startswith("[")
        or stripped.startswith("---")
        or "\n" in source
        or "\r" in source
    )


def _parse_text(text: str, *, format: str | None, source_name: str) -> Mapping[str, Any]:
    if not isinstance(text, str):
        raise ConfigError(f"config text must be a string, got {type(text).__name__}")
    selected = format.lower() if format is not None else None
    if selected is None:
        try:
            return _parse_json(text, source_name)
        except ConfigError as json_error:
            try:
                return _parse_yaml(text, source_name)
            except ConfigError as yaml_error:
                raise ConfigError(
                    f"could not parse {source_name} as JSON or YAML: {json_error}; {yaml_error}"
                ) from yaml_error
    if selected == "json":
        return _parse_json(text, source_name)
    if selected in {"yaml", "yml"}:
        return _parse_yaml(text, source_name)
    raise ConfigError(f"unsupported config format {format!r}; expected 'json' or 'yaml'")


def _parse_json(text: str, source_name: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"malformed JSON in {source_name}: {exc.msg} at line {exc.lineno}") from exc
    if raw is None:
        raise ConfigError(f"{source_name} is empty; expected a top-level mapping")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{source_name} must contain a top-level mapping")
    return raw


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate config key {key!r}")
        result[key] = value
    return result


def _parse_yaml(text: str, source_name: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError(
            f"cannot load YAML config from {source_name}: PyYAML is not installed; "
            "install PyYAML or use JSON"
        ) from exc
    try:
        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
            loader.flatten_mapping(node)
            result: dict[Any, Any] = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in result:
                    raise ConfigError(f"duplicate config key {key!r} in {source_name}")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping,
        )
        raw = yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {source_name}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{source_name} is empty; expected a top-level mapping")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{source_name} must contain a top-level mapping")
    return raw
