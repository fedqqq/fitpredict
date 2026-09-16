"""Minimal end-to-end training entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fitpredict.binding import BindingEngine
from fitpredict.config import (
    ComponentResolver,
    ExperimentConfig,
    load_config,
    resolve_config,
)
from fitpredict.config.errors import ConfigError
from fitpredict.data import (
    RuntimeSourceContext,
    build_dataloader,
    load_data_metadata,
    load_tabular_data,
    split_tabular_data,
)
from fitpredict.data.dataset import TabularDataset


@dataclass(frozen=True)
class FitHistory:
    """Small training history returned by ``fit``."""

    train_loss: list[float]
    batch_loss: list[float]


@dataclass(frozen=True)
class FitResult:
    """Result of a minimal fitpredict training run."""

    model: torch.nn.Module
    config: ExperimentConfig
    history: FitHistory
    optimizer: torch.optim.Optimizer


def fit(config: str | Path | Mapping[str, Any] | ExperimentConfig) -> FitResult:
    """Train a model from a fitpredict config.

    This P0 implementation covers the train-only path:
    config -> data -> split -> Dataset -> DataLoader -> model -> bindings ->
    forward -> objective loss -> backward -> optimizer.
    """

    typed_config = config if isinstance(config, ExperimentConfig) else load_config(config)
    metadata = load_data_metadata(typed_config.data)
    resolved = resolve_config(typed_config, data_metadata=metadata)

    loaded_rows = load_tabular_data(resolved.data)
    split = split_tabular_data(loaded_rows, resolved.data)
    train_dataset = TabularDataset(split.train, data=resolved.data)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=resolved.training.shuffle,
    )

    device = _resolve_device(resolved.training.device)
    component_resolver = ComponentResolver()
    model = component_resolver.instantiate("model", resolved.model).to(device)
    _load_model_weights(model, resolved.model.weights, device)

    optimizer = component_resolver.instantiate(
        "optimizer",
        resolved.training.optimizer,
        model.parameters(),
    )
    losses = [
        component_resolver.instantiate("loss", objective.loss)
        for objective in resolved.training.objectives
    ]

    epoch_losses: list[float] = []
    batch_losses: list[float] = []
    for _epoch in range(resolved.training.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            base_context = RuntimeSourceContext.from_batch(batch)
            model_kwargs = BindingEngine(
                base_context,
                features=resolved.data.features,
                targets=resolved.data.targets,
                path="train.batch",
            ).build_kwargs(resolved.model.inputs, path="model.inputs")
            outputs = model(**model_kwargs)
            loss_context = RuntimeSourceContext(
                features=base_context.features,
                targets=base_context.targets,
                outputs=outputs,
            )

            objective_losses = []
            for index, (objective, loss_fn) in enumerate(zip(resolved.training.objectives, losses)):
                loss_kwargs = BindingEngine(
                    loss_context,
                    features=resolved.data.features,
                    targets=resolved.data.targets,
                    path="train.batch",
                ).build_kwargs(
                    objective.bindings,
                    path=f"training.objectives[{index}].bindings",
                )
                objective_losses.append(loss_fn(**loss_kwargs) * objective.weight)
            loss = _sum_losses(objective_losses)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            batch_losses.append(loss_value)
            total_loss += loss_value
            num_batches += 1
        epoch_losses.append(total_loss / num_batches if num_batches else float("nan"))

    return FitResult(
        model=model,
        config=resolved,
        history=FitHistory(train_loss=epoch_losses, batch_loss=batch_losses),
        optimizer=optimizer,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _load_model_weights(
    model: torch.nn.Module,
    weights: Path | None,
    device: torch.device,
) -> None:
    if weights is None:
        return
    try:
        state = torch.load(weights, map_location=device)
    except OSError as exc:
        raise ConfigError(f"could not load model weights {weights}: {exc}") from exc
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    return value


def _sum_losses(losses: list[torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ConfigError("training.objectives must contain at least one objective.")
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total
