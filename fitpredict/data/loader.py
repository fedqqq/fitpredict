"""Small stdlib-first tabular data loader for P0 config resolution.

This module only reads rows and derives metadata. Splitting, tensorization,
transforms, scientific validation, and training are intentionally outside this
layer.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import DataConfig, DataMetadata


@dataclass(frozen=True)
class TabularRows:
    """Loaded tabular rows plus their source columns."""

    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[str, ...]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def load_tabular_data(
    data: DataConfig | str | Path,
    *,
    format: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> TabularRows:
    """Load a supported tabular data file into row mappings.

    CSV, JSON, and JSONL are implemented with the standard library. Parquet and
    Feather are accepted schema formats and are loaded only when ``pyarrow`` is
    installed.
    """

    path, selected_format, selected_options = _data_source(data, format, options)
    if selected_format == "csv":
        return _load_csv(path, selected_options)
    if selected_format == "json":
        return _load_json(path, selected_options)
    if selected_format == "jsonl":
        return _load_jsonl(path, selected_options)
    if selected_format == "parquet":
        return _load_parquet(path, selected_options)
    if selected_format == "feather":
        return _load_feather(path, selected_options)
    raise ConfigError(f"data.format is unsupported by data loading: {selected_format}.")


def load_data_metadata(
    data: DataConfig | str | Path,
    *,
    format: str | None = None,
    features: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> DataMetadata:
    """Load tabular metadata required by ``ConfigResolver``.

    When ``data`` is a ``DataConfig``, declared features and targets are taken
    from the config. For path-based calls, pass ``features`` and ``targets``.
    """

    declared_features = (
        tuple(data.features) if isinstance(data, DataConfig) else tuple(features or ())
    )
    declared_targets = tuple(data.targets) if isinstance(data, DataConfig) else tuple(targets or ())
    loaded = load_tabular_data(data, format=format, options=options)
    return metadata_from_rows(
        loaded.rows,
        columns=loaded.columns,
        features=declared_features,
        targets=declared_targets,
    )


def metadata_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    features: Sequence[str] = (),
    targets: Sequence[str] = (),
) -> DataMetadata:
    """Derive resolver-compatible metadata from already loaded row mappings."""

    materialized_rows = tuple(rows)
    inferred_columns = (
        tuple(columns) if columns is not None else _columns_from_rows(materialized_rows)
    )
    _validate_declared_columns(inferred_columns, features, targets)

    feature_dtypes = {name: _infer_column_dtype(materialized_rows, name) for name in features}
    target_dtypes = {name: _infer_column_dtype(materialized_rows, name) for name in targets}
    feature_shapes = _known_column_shapes(materialized_rows, features)
    target_shapes = _known_column_shapes(materialized_rows, targets)
    return DataMetadata(
        columns=inferred_columns,
        feature_dtypes=feature_dtypes,
        target_dtypes=target_dtypes,
        feature_shapes=feature_shapes,
        target_shapes=target_shapes,
        num_rows=len(materialized_rows),
    )


def _data_source(
    data: DataConfig | str | Path,
    format: str | None,
    options: Mapping[str, Any] | None,
) -> tuple[Path, str, dict[str, Any]]:
    if isinstance(data, DataConfig):
        if format is not None and format.lower() != data.format:
            raise ConfigError(
                f"data.format override {format!r} conflicts with config format {data.format!r}."
            )
        if options is not None:
            selected_options = dict(data.options)
            selected_options.update(options)
        else:
            selected_options = dict(data.options)
        return data.path, data.format, selected_options

    if format is None:
        raise ConfigError("data format is required when loading from a path.")
    if not isinstance(data, (str, Path)):
        raise ConfigError(f"data source must be DataConfig or path, got {type(data).__name__}.")
    selected_format = format.lower()
    return Path(data), selected_format, dict(options or {})


def _read_text(path: Path, *, encoding: str) -> str:
    try:
        return path.read_text(encoding=encoding)
    except OSError as exc:
        raise ConfigError(f"could not read data file {path}: {exc}") from exc


def _load_csv(path: Path, options: Mapping[str, Any]) -> TabularRows:
    allowed = {"encoding", "delimiter", "quotechar"}
    _reject_unknown_options(options, allowed, "csv")
    encoding = _string_option(options, "encoding", default="utf-8")
    delimiter = _string_option(options, "delimiter", default=",")
    quotechar = _string_option(options, "quotechar", default='"')
    if len(delimiter) != 1:
        raise ConfigError("data.options.delimiter must be exactly one character for csv.")
    if len(quotechar) != 1:
        raise ConfigError("data.options.quotechar must be exactly one character for csv.")

    try:
        with path.open(newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter, quotechar=quotechar)
            if reader.fieldnames is None:
                raise ConfigError(f"csv data file {path} must contain a header row.")
            columns = tuple(_validate_column_names(reader.fieldnames, path))
            rows = tuple(dict(row) for row in reader)
    except UnicodeError as exc:
        raise ConfigError(f"could not decode csv data file {path}: {exc}") from exc
    except csv.Error as exc:
        raise ConfigError(f"malformed csv data file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read data file {path}: {exc}") from exc
    return TabularRows(rows=rows, columns=columns)


def _load_json(path: Path, options: Mapping[str, Any]) -> TabularRows:
    allowed = {"encoding"}
    _reject_unknown_options(options, allowed, "json")
    text = _read_text(path, encoding=_string_option(options, "encoding", default="utf-8"))
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"malformed json data file {path}: {exc.msg} at line {exc.lineno}"
        ) from exc
    if not isinstance(value, list):
        raise ConfigError(f"json data file {path} must contain a list of row objects.")
    rows = tuple(_require_row_mapping(row, f"{path}[{index}]") for index, row in enumerate(value))
    return TabularRows(rows=rows, columns=_columns_from_rows(rows))


def _load_jsonl(path: Path, options: Mapping[str, Any]) -> TabularRows:
    allowed = {"encoding"}
    _reject_unknown_options(options, allowed, "jsonl")
    text = _read_text(path, encoding=_string_option(options, "encoding", default="utf-8"))
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"malformed jsonl data file {path} at line {line_number}: {exc.msg}"
            ) from exc
        rows.append(_require_row_mapping(value, f"{path}:{line_number}"))
    return TabularRows(rows=tuple(rows), columns=_columns_from_rows(rows))


def _load_parquet(path: Path, options: Mapping[str, Any]) -> TabularRows:
    _reject_unknown_options(options, {"columns"}, "parquet")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "cannot load parquet data: pyarrow is not installed; install pyarrow "
            "or use csv/json/jsonl for stdlib-only loading."
        ) from exc
    try:
        table = pq.read_table(path, columns=options.get("columns"))
    except Exception as exc:
        raise ConfigError(f"could not read parquet data file {path}: {exc}") from exc
    return _rows_from_arrow_table(table)


def _load_feather(path: Path, options: Mapping[str, Any]) -> TabularRows:
    _reject_unknown_options(options, {"columns"}, "feather")
    try:
        import pyarrow.feather as feather
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "cannot load feather data: pyarrow is not installed; install pyarrow "
            "or use csv/json/jsonl for stdlib-only loading."
        ) from exc
    try:
        table = feather.read_table(path, columns=options.get("columns"))
    except Exception as exc:
        raise ConfigError(f"could not read feather data file {path}: {exc}") from exc
    return _rows_from_arrow_table(table)


def _rows_from_arrow_table(table: Any) -> TabularRows:
    columns = tuple(str(name) for name in table.column_names)
    _validate_unique_columns(columns, "arrow table")
    rows = tuple(dict(row) for row in table.to_pylist())
    return TabularRows(rows=rows, columns=columns)


def _require_row_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a row object.")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{path} row keys must be non-empty strings.")
        result[key] = item
    return result


def _columns_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return tuple(columns)


def _validate_column_names(columns: Iterable[str], path: Path) -> tuple[str, ...]:
    result = tuple(columns)
    if not all(isinstance(name, str) and name for name in result):
        raise ConfigError(f"data file {path} contains an empty column name.")
    _validate_unique_columns(result, f"data file {path}")
    return result


def _validate_unique_columns(columns: Sequence[str], source: str) -> None:
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ConfigError(f"{source} contains duplicate column(s): {', '.join(duplicates)}.")


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
            "data columns are missing configured feature(s): " + ", ".join(missing_features)
        )
    if missing_targets:
        raise ConfigError(
            "data columns are missing configured target(s): " + ", ".join(missing_targets)
        )


def _infer_column_dtype(rows: Sequence[Mapping[str, Any]], column: str) -> str:
    dtypes = {_infer_value_dtype(row.get(column)) for row in rows if row.get(column) is not None}
    dtypes.discard("null")
    if not dtypes:
        return "null"
    if len(dtypes) == 1:
        return next(iter(dtypes))
    numeric = {"int64", "float64"}
    if dtypes <= numeric:
        return "float64" if "float64" in dtypes else "int64"
    return "mixed"


def _infer_value_dtype(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _known_column_shapes(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for column in columns:
        shape = _infer_column_shape(rows, column)
        if shape is not None:
            result[column] = shape
    return result


def _infer_column_shape(rows: Sequence[Mapping[str, Any]], column: str) -> tuple[int, ...] | None:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    if not values or not all(isinstance(value, list) for value in values):
        return None
    shapes = {_infer_value_shape(value) for value in values}
    if None in shapes:
        return None
    if len(shapes) == 1:
        return next(iter(shapes))
    return None


def _infer_value_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = {_infer_value_shape(item) for item in value}
    if len(child_shapes) != 1 or None in child_shapes:
        return None
    child_shape = next(iter(child_shapes))
    if child_shape is None:
        return None
    return (len(value),) + child_shape


def _reject_unknown_options(
    options: Mapping[str, Any],
    allowed: set[str],
    format_name: str,
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ConfigError(
            f"data.options contains unsupported {format_name} option(s): " + ", ".join(unknown)
        )


def _string_option(options: Mapping[str, Any], name: str, *, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"data.options.{name} must be a non-empty string.")
    return value
