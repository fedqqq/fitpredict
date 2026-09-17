# fitpredict

fitpredict is a small declarative layer on top of PyTorch for tabular machine learning experiments.

You write two things:

1. a normal `torch.nn.Module`;
2. a YAML or JSON config that describes data, training, evaluation, logging, and saving.

fitpredict handles the experiment plumbing: config validation, data loading, train/validation/test split, tensorization, generic `Dataset` / `DataLoader`, training loop, validation, metrics, checkpoints, logging, and prediction.

fitpredict is not AutoML. It checks that your config can run; it does not choose features, clean data, tune hyperparameters, or judge whether an experiment is scientifically correct.

## Install

From PyPI:

```bash
pip install fitpredict
```

For TestPyPI verification:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fitpredict==1.0.1
```

For local development from this repository:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

Supported Python versions: 3.11 and 3.12.

## First experiment

Create data as JSONL, CSV, JSON, parquet, or feather. Example `data/train.jsonl`:

```jsonl
{"age": 21, "income": 40000, "label": 0}
{"age": 42, "income": 90000, "label": 1}
```

Create a model importable from Python:

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

Create `config.yaml`:

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

Train:

```python
from fitpredict import fit

result = fit("config.yaml")
print(result.history.train_loss)
```

Run prediction:

```python
from fitpredict import predict

predictions = predict("config.yaml", checkpoint="runs/example/best.pt", data="data/predict.jsonl")
print(predictions)
```

Prediction rows only need the configured feature columns. Target columns are required for training losses or metrics, but not for inference.

## Public API

Use root package imports in application code:

```python
from fitpredict import (
    ConfigError,
    ExperimentConfig,
    fit,
    load_config,
    load_config_file,
    loads_config,
    predict,
    resolve_config,
)
```

Stable entry points:

- `fit(config)` trains an experiment from a path, raw JSON/YAML string, mapping, or `ExperimentConfig`.
- `predict(config, checkpoint=None, data=None)` runs inference for the configured model.
- `load_config`, `load_config_file`, and `loads_config` load typed config objects.
- `resolve_config` resolves defaults, data metadata, references, and components.
- `ConfigError` is the user-facing error type for configuration and runtime contract problems.

## Documentation and examples

- [Quickstart](docs/quickstart.md)
- [Config reference](docs/config-reference.md)
- [Binding reference](docs/bindings.md)
- [Runnable examples](examples/README.md)
- [Release process](docs/release.md)
- [Changelog](CHANGELOG.md)

Run bundled examples from the repository root:

```bash
python examples/run_fit.py examples/configs/classification.yaml
python examples/run_predict.py examples/configs/classification.yaml examples/data/predict.json
```

## Development checks

```bash
python -m ruff check fitpredict tests examples scripts
python -m ruff format --check fitpredict tests examples scripts
python -m mypy fitpredict
python -m pytest -q
python -m compileall -q fitpredict tests examples scripts
python -m build
python scripts/check_version.py
```
