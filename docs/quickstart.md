# Quickstart

## 1. Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install fitpredict
```

For local development from this repository:

```bash
pip install -e '.[dev]'
```

## 2. Write a model

```python
# models.py
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
```

## 3. Write a config

```yaml
data:
  path: data/train.json
  format: json
  features: [x1, x2]
  targets: [y]
  split: {train: 0.8, val: 0.1, test: 0.1, shuffle: true, seed: 42}
model:
  class: models:Regressor
  params: {input_dim: ${data.num_features}}
  inputs: {x: {source: features, dtype: float32}}
training:
  epochs: 5
  batch_size: 32
  optimizer: {name: AdamW, params: {lr: 0.001}}
  objectives:
    - loss: {name: MSELoss}
      bindings:
        input: {source: outputs, dtype: float32}
        target: {source: targets.y, dtype: float32}
evaluation: {metrics: []}
logging: {console: true, tensorboard: false, mlflow: false}
saving:
  output_dir: runs/example
  save_last: true
  save_best: {monitor: val.loss, mode: min}
```

## 4. Fit

```python
from fitpredict import fit

result = fit("config.yaml")
print(result.history.train_loss)
```

## 5. Predict

```python
from fitpredict import predict

predictions = predict("config.yaml", checkpoint="runs/example/best.pt", data="data/predict.json")
print(predictions)
```

See runnable examples in `examples/`.
