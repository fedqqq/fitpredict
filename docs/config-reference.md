# Config reference

A fitpredict config has six top-level sections.

## `data`

Defines the tabular input.

Required: `path`, `format`, `features`, `targets`, `split`.

Common fields:

- `format`: `csv`, `json`, `jsonl`, `parquet`, or `feather`.
- `features`: columns available through `features.*` or aggregated as `features`.
- `targets`: supervised target columns available through `targets.*`.
- `split`: ratios for `train`, `val`, `test`, plus optional `shuffle` and `seed`.

## `model`

Defines the user model and how inputs are bound.

Required: `class`, `inputs`.

Common fields:

- `class`: built-in short name or Python import path, for example `models:Regressor`.
- `params`: constructor kwargs. Config references like `${data.num_features}` are supported.
- `weights`: optional state dict path.
- `inputs`: mapping from model argument names to binding configs.

## `training`

Defines optimization.

Required: `epochs`, `batch_size`, `optimizer`, `objectives`.

Common fields:

- `device`: `cpu`, `cuda`, `mps`, or `auto`.
- `optimizer`: built-in/custom component plus params.
- `scheduler`: optional scheduler with `step_on: batch | epoch | metric`.
- `objectives`: one or more weighted loss definitions with bindings.

## `evaluation`

Defines validation/test metrics.

Common fields:

- `metrics`: list of metric components.
- `bindings`: metric argument bindings.
- `transform`: optional sequential transform pipeline before metric calculation.

## `logging`

Defines experiment logging.

Common fields:

- `console`: print progress and results.
- `tensorboard`: write TensorBoard scalars and hparams.
- `mlflow`: log params and metrics to the active MLflow tracking URI.

## `saving`

Defines output artifacts.

Common fields:

- `output_dir`: directory for config, metrics, result summary and checkpoints.
- `save_last`: write `last.pt`.
- `save_best`: monitor a validation metric, for example `{monitor: val.loss, mode: min}`.

`test.*` metrics are never valid for model selection.
