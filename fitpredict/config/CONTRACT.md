# Config Schema Contract

This document records the P0 schema assumptions that `ConfigLoader`,
`ConfigResolver`, `ComponentResolver`, data loading, tensorization, and binding
code should use.

## Top-level sections

All six public sections are required in the user config:

- `data`
- `model`
- `training`
- `evaluation`
- `logging`
- `saving`

Optional infrastructure is represented by defaults inside those sections, not
by omitting the section.

## Parser API

`fitpredict.config.schema.parse_experiment_config(raw)` is the stable parser
entry point for already-loaded mappings.

`fitpredict.config.load_config`, `load_config_file`, and `loads_config` are the
loader entry points and must return `ExperimentConfig`.

All schema and loader validation errors must raise
`fitpredict.config.errors.ConfigError`.

## Defaults

`data`:

- Required: `path`, `format`, `features`, `targets`, `split`.
- Optional: `options`.
- Defaults: `options={}`.
- Validation: `path` and `format` are non-empty strings; `format` is one of
  `csv`, `json`, `jsonl`, `parquet`, `feather`; `features` and `targets` are
  non-empty lists of unique strings inside each list.
- `features` and `targets` may contain the same column name because that is
  technically executable and the framework does not reject potential leakage.

`data.split`:

- Required: `train`, `val`, `test`.
- Optional: `shuffle`, `seed`, `stratify`, `sort_by`, `ascending`.
- Defaults: `shuffle=true`, `seed=null`, `stratify=null`, `sort_by=null`,
  `ascending=true`.
- Validation: fractions are finite non-negative numbers and sum to `1.0`
  within absolute tolerance `1e-9`; `seed` is an integer when provided;
  booleans must be actual booleans.

`model`:

- Required: `class`, `inputs`.
- Optional: `params`, `weights`.
- Defaults: `params={}`, `weights=null`.
- Validation: `class` is a non-empty string; `inputs` is a non-empty mapping of
  callable argument names to binding configs; `weights` is a string path when
  provided.

`training`:

- Required: `epochs`, `batch_size`, `optimizer`, `objectives`.
- Optional: `shuffle`, `device`, `scheduler`.
- Defaults: `shuffle=true`, `device=cpu`, `scheduler=null`.
- Validation: `epochs` and `batch_size` are integers greater than zero;
  `device` is one of `cpu`, `cuda`, `mps`, `auto`; `objectives` is a non-empty
  list.

`training.optimizer` and component configs:

- Required: `name`.
- Optional: `params`.
- Defaults: `params={}`.
- Validation: `name` is a non-empty string. Built-in lookup and import checks
  belong to `ComponentResolver`.

`training.objectives[]`:

- Required: `loss`, `bindings`.
- Optional: `weight`.
- Defaults: `weight=1.0`.
- Validation: `loss` is a component config; `bindings` is a non-empty mapping;
  `weight` is finite.

`training.scheduler`:

- Required: `name`.
- Optional: `params`, `step_on`, `monitor`.
- Defaults: `params={}`. `step_on` remains `null` unless the scheduler is a
  schema-known built-in with a fixed mode.
- Validation: `step_on`, when present, is one of `batch`, `epoch`, `metric`;
  custom scheduler import paths containing `.` must provide `step_on`;
  `step_on=metric` requires `monitor`.
- Schema-known built-in modes: `ReduceLROnPlateau=metric`,
  `OneCycleLR=batch`, `CyclicLR=batch`. A conflicting explicit `step_on` is an
  error. `SchedulerConfig.step_on_was_explicit` records whether the user set it.

`evaluation`:

- Required section.
- Optional: `metrics`.
- Defaults: `metrics=[]`.
- Validation: `metrics` is a list of metric configs.

`evaluation.metrics[]`:

- Required: `name`.
- Optional: `params`, `bindings`, `transform`.
- Defaults: `params={}`, `bindings={}`, `transform=[]`.
- Validation: `name` is a non-empty string; `bindings` follows the common
  Binding Protocol; `transform` is a list.

`evaluation.metrics[].transform[]`:

- Required: `name`.
- Optional: `params`, `bindings`.
- Defaults: `params={}`, `bindings={}`.
- Validation: same component and binding validation as other callables.

`logging`:

- Required section.
- Optional: `console`, `tensorboard`, `mlflow`.
- Defaults: `console=true`, `tensorboard=false`, `mlflow=false`.
- Validation: all fields are booleans.

`saving`:

- Required section.
- Optional: `save_last`, `save_best`, `output_dir`.
- Defaults: `save_last=true`, `save_best=null`, `output_dir=runs`.
- Validation: `save_last` is boolean; `output_dir` is a string path.

`saving.save_best`:

- Required when present: `monitor`, `mode`.
- Validation: `monitor` is a non-empty string; `mode` is `min` or `max`.

## User-defined and derived fields

User-defined fields:

- `data`: `path`, `format`, `options`, `features`, `targets`, `split`
- `model`: `class`, `params`, `weights`, `inputs`
- `training`: `epochs`, `batch_size`, `shuffle`, `device`, `optimizer`,
  `scheduler`, `objectives`
- `evaluation`: `metrics`
- `logging`: `console`, `tensorboard`, `mlflow`
- `saving`: `save_last`, `save_best`, `output_dir`

Derived fields reserved for `ConfigResolver`:

- `data.metadata`
- `data.num_features`
- `data.num_targets`
- `data.train_size`
- `data.val_size`
- `data.test_size`
- `training.steps_per_epoch`
- `training.total_steps`

The schema sets `data.num_features` and `data.num_targets` from declared column
names because those values do not require file metadata. Size and step fields
remain `None` until metadata and split resolution.

Reserved derived field names are rejected in user mappings. A future resolved
config parser can accept them explicitly, but the user-config parser must not
silently ignore them.

## Binding protocol

Every binding maps one callable argument to exactly one source.

Accepted source strings:

- `features`: aggregate all declared data features into one tensor.
- `features.<name>`: one declared feature column.
- `targets.<name>`: one declared target column.
- `outputs.<name>`: one named model output.
- `outputs`: the model's single unnamed tensor output.

`targets` by itself is not valid. Lists of sources in a single binding are not
valid. If multiple values are needed, the callable must expose multiple
arguments and the config must define one binding per argument.

`BindingConfig.source` stores a parsed `BindingSource`. It compares equal to
the raw source string for loader/resolver compatibility.

## Formats and rounding

Supported tabular `data.format` values for P0 schema validation are:

- `csv`
- `json`
- `jsonl`
- `parquet`
- `feather`

The exact backend dependency and option handling belongs to data loading.

`data.split.train + data.split.val + data.split.test` must equal `1.0` within
absolute tolerance `1e-9`. Converting fractions to row counts is left to
`ConfigResolver` or data splitting so one component owns the rounding rule.

`ConfigResolver` requires data metadata as a separate argument or pre-populated
`DataConfig.metadata`. It does not read data files. The required P0 metadata
fields are `columns` and `num_rows`. Optional metadata fields are
`feature_dtypes`, `target_dtypes`, `feature_shapes`, and `target_shapes`.
Metadata may also include `features`, `targets`, `num_features`, and
`num_targets` for producer compatibility; declared config values remain
authoritative.

With metadata, `ConfigResolver` converts split fractions to row counts with
largest-remainder rounding, in `train`, `val`, `test` tie order.
`training.steps_per_epoch` is `ceil(data.train_size / training.batch_size)`, and
`training.total_steps` is `steps_per_epoch * training.epochs`.

References are resolved after metadata and derived values are applied. A string
that is exactly `${some.dotted.path}` is replaced by the referenced value and
keeps that value's type. References embedded inside a longer string are
stringified. Reference paths traverse public dataclass fields and mapping keys
inside the resolved config only. Private/dunder path segments, methods, class
objects, `eval`, component imports, and file access are not allowed.

After references are resolved, `ConfigResolver` validates resolved field types,
known `features.*` and `targets.*` binding sources, known scheduler `step_on`
defaults, and `saving.save_best.monitor`. Runtime `outputs.*` bindings remain a
runtime validation responsibility.

## Component names

Component configs use:

- short built-in names such as `AdamW` or `CrossEntropyLoss`
- Python import paths such as `losses.my_loss.MyLoss`

The schema only stores names and params. `ComponentResolver` owns import and
built-in lookup validation.
