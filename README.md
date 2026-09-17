# fitpredict

fitpredict is a small declarative layer on top of PyTorch for tabular experiments.

The project goal is that a user writes a model and a YAML or JSON config, while
fitpredict owns the experiment plumbing: config validation, data loading,
splitting, tensorization, generic Dataset/DataLoader construction, and runtime
binding from data/model outputs into callables.

fitpredict is not AutoML. It checks whether a config can be executed; it does
not choose features, clean data, prevent leakage, tune hyperparameters, or
judge whether an experiment is scientifically sound.

## Current Status

P0 Milestone 0, P1 training lifecycle, and P2 experiment infrastructure are implemented.

Implemented:

- typed six-section config schema: `data`, `model`, `training`, `evaluation`,
  `logging`, `saving`
- JSON/YAML config loading
- config resolution with defaults, data metadata, derived values, references,
  component resolution, and freezing
- tabular data loading for `csv`, `json`, `jsonl`, plus `parquet` and `feather`
  when `pyarrow` is installed
- train/validation/test splitting from config
- tensorization and the generic sample contract
  `{"features": {...}, "targets": {...}}`
- source resolution for `features`, `features.<name>`, `targets.<name>`,
  `outputs`, and `outputs.<name>`
- binding engine that builds `callable(**kwargs)` arguments
- minimal `fit()` training pipeline: config -> data -> split -> Dataset ->
  DataLoader -> model -> binding -> forward -> loss -> backward -> optimizer
- validation after each epoch
- dataset-level validation/test metrics with transform pipelines
- `best.pt` / `last.pt` checkpoints, with best selected only by validation
  loss or validation metrics
- final test pass that loads `best.pt` once
- scheduler stepping on batch, epoch, or metric
- multiple weighted objectives
- console logging plus resolved config, metric, and result artifacts in
  `saving.output_dir`
- TensorBoard and MLflow parameter/metric logging when enabled
- minimal `predict()` inference pipeline:
  config -> metadata/resolve -> prediction data -> Dataset -> DataLoader ->
  model -> checkpoint or `model.weights` -> bound forward pass -> detached CPU
  predictions

TensorBoard and MLflow are optional runtime integrations. If either backend is
enabled in config but the package is not installed, `fit()` raises a clear
configuration error.

## Quickstart

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create a minimal config:

```yaml
data:
  path: data/train.jsonl
  format: jsonl
  features: [age, income]
  targets: [label]
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    shuffle: true
    seed: 42

model:
  class: models.Classifier
  params:
    input_dim: ${data.num_features}
  inputs:
    x:
      source: features
      dtype: float32

training:
  epochs: 5
  batch_size: 32
  device: cpu
  optimizer:
    name: AdamW
    params:
      lr: 0.001
  objectives:
    - loss:
        name: CrossEntropyLoss
      bindings:
        input:
          source: outputs.logits
        target:
          source: targets.label
          dtype: int64

evaluation:
  metrics: []
logging:
  console: true
  tensorboard: false
  mlflow: false
saving:
  output_dir: runs/example
  save_last: true
  save_best:
    monitor: val.loss
    mode: min
```

Make the configured model importable:

```python
# models.py
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return {"logits": self.linear(x)}
```

Run training:

```python
from fitpredict import fit

result = fit("config.yaml")
print(result.history.train_loss)
```

Run inference:

```python
from fitpredict import predict

predictions = predict("config.yaml", checkpoint="runs/example/best.pt", data="predict.jsonl")
print(predictions)
```

`predict()` returns the model output shape directly: a single tensor stays a
tensor, and a dictionary output stays a dictionary of tensors. Prediction rows
only need the configured feature columns; target columns used for training
losses or metrics are not required. Empty prediction data returns `None`.
Tensor outputs must include a leading batch dimension; scalar tensor outputs are
rejected. If neither `checkpoint` nor `model.weights` is set, prediction uses a
freshly initialized configured model.

## Tests

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run a syntax check:

```bash
.venv/bin/python -m compileall fitpredict tests
```
