# Binding reference

Bindings map runtime data into keyword arguments for models, losses, metrics and transforms.

```yaml
argument_name:
  source: features.x1
  dtype: float32
```

## Canonical sources

- `features` — aggregate all configured feature columns in `data.features` order.
- `features.<name>` — one configured feature column.
- `targets.<name>` — one configured target column.
- `outputs` — single tensor model output.
- `outputs.<name>` — one value from a dictionary model output.

## `dtype`

Optional dtype conversion for tensor sources:

- `bool`
- `int64`
- `float16`
- `float32`
- `float64`
- `auto` or omitted

## Where bindings are used

- `model.inputs`: model `forward(**kwargs)` arguments.
- `training.objectives[].bindings`: loss call arguments.
- `evaluation.metrics[].bindings`: metric call arguments.
- `evaluation.metrics[].transform[].bindings`: transform call arguments.

## Output rules

A model may return either a single tensor or a dict of tensors. Use `outputs` for a single tensor. Use `outputs.<name>` for dict outputs.
