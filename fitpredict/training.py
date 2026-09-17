"""Minimal end-to-end training entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from fitpredict.binding import BindingEngine
from fitpredict.config import (
    ComponentResolution,
    ComponentResolver,
    ExperimentConfig,
    load_config,
    resolve_config,
)
from fitpredict.config.errors import ConfigError
from fitpredict.config.schema import MetricConfig, TransformConfig
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
    val_loss: list[float] = field(default_factory=list)
    val_metrics: list[dict[str, float]] = field(default_factory=list)
    test_loss: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)
    scheduler_metrics: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    """Result of a minimal fitpredict training run."""

    model: torch.nn.Module
    config: ExperimentConfig
    history: FitHistory
    optimizer: torch.optim.Optimizer
    checkpoint_paths: dict[str, Path] = field(default_factory=dict)


def fit(config: str | Path | Mapping[str, Any] | ExperimentConfig) -> FitResult:
    """Train a model from a fitpredict config.

    This entry point covers the minimal supervised lifecycle:
    config -> data -> split -> Dataset -> DataLoader -> model -> bindings ->
    train/validation -> best/last checkpoints -> one final best-checkpoint test.
    """

    typed_config = config if isinstance(config, ExperimentConfig) else load_config(config)
    metadata = load_data_metadata(typed_config.data)
    resolved = resolve_config(typed_config, data_metadata=metadata)
    _validate_lifecycle_config(resolved)

    loaded_rows = load_tabular_data(resolved.data)
    split = split_tabular_data(loaded_rows, resolved.data)
    train_dataset = TabularDataset(split.train, data=resolved.data)
    val_dataset = TabularDataset(split.val, data=resolved.data)
    test_dataset = TabularDataset(split.test, data=resolved.data)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=resolved.training.shuffle,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=False,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=False,
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
        _prepare_runtime_callable(
            component_resolver.resolve("loss", objective.loss),
            device=device,
            mode="train",
        )
        for objective in resolved.training.objectives
    ]
    metrics = _prepare_metrics(component_resolver, resolved.evaluation.metrics, device)
    scheduler = _build_scheduler(component_resolver, resolved, optimizer)

    checkpoint_paths: dict[str, Path] = {}
    output_dir = resolved.saving.output_dir
    if resolved.saving.save_last or resolved.saving.save_best is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    best_value: float | None = None

    epoch_losses: list[float] = []
    batch_losses: list[float] = []
    val_losses: list[float] = []
    val_metrics: list[dict[str, float]] = []
    scheduler_metrics: dict[str, list[float]] = {}
    for epoch in range(resolved.training.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = _batch_loss(
                model,
                batch,
                config=resolved,
                losses=losses,
                context_path="train.batch",
            )
            loss.backward()
            optimizer.step()
            _step_scheduler(scheduler, when="batch")

            loss_value = float(loss.detach().cpu().item())
            batch_losses.append(loss_value)
            total_loss += loss_value
            num_batches += 1
        epoch_losses.append(total_loss / num_batches if num_batches else float("nan"))
        _step_scheduler(scheduler, when="epoch")

        validation = _evaluate_model(
            model,
            val_loader,
            config=resolved,
            losses=losses,
            metrics=metrics,
            device=device,
            split_name="val",
        )
        val_losses.append(validation["loss"])
        val_metrics.append({name: value for name, value in validation.items() if name != "loss"})

        checkpoint_metrics = {
            "train.loss": epoch_losses[-1],
            **{f"val.{name}": value for name, value in validation.items()},
        }
        _record_scheduler_metrics(scheduler_metrics, checkpoint_metrics)
        _step_scheduler(scheduler, when="metric", values=checkpoint_metrics)
        if resolved.saving.save_last:
            _save_checkpoint(last_path, model=model)
            checkpoint_paths["last"] = last_path
        if resolved.saving.save_best is not None:
            monitor = resolved.saving.save_best.monitor
            monitor_value = checkpoint_metrics[monitor]
            if _is_better(
                monitor_value,
                best_value,
                mode=resolved.saving.save_best.mode,
            ):
                _save_checkpoint(best_path, model=model)
                best_value = monitor_value
                checkpoint_paths["best"] = best_path

    test_loss: float | None = None
    test_metrics: dict[str, float] = {}
    if len(test_dataset) > 0:
        if "best" not in checkpoint_paths:
            raise ConfigError("final test requires a selected best checkpoint.")
        _load_checkpoint_model_state(model, checkpoint_paths["best"], device)
        test = _evaluate_model(
            model,
            test_loader,
            config=resolved,
            losses=losses,
            metrics=metrics,
            device=device,
            split_name="test",
        )
        test_loss = test["loss"]
        test_metrics = {name: value for name, value in test.items() if name != "loss"}

    return FitResult(
        model=model,
        config=resolved,
        history=FitHistory(
            train_loss=epoch_losses,
            batch_loss=batch_losses,
            val_loss=val_losses,
            val_metrics=val_metrics,
            test_loss=test_loss,
            test_metrics=test_metrics,
            scheduler_metrics=scheduler_metrics,
        ),
        optimizer=optimizer,
        checkpoint_paths=checkpoint_paths,
    )


def _validate_lifecycle_config(config: ExperimentConfig) -> None:
    if config.saving.save_best is not None and config.data.val_size == 0:
        raise ConfigError("saving.save_best requires a non-empty validation split.")
    if config.data.test_size and config.saving.save_best is None:
        raise ConfigError("a non-empty test split requires saving.save_best.")
    scheduler = config.training.scheduler
    if scheduler is not None and scheduler.monitor is not None:
        if scheduler.monitor.startswith("test."):
            raise ConfigError("training.scheduler.monitor cannot reference test metrics.")
        if not scheduler.monitor.startswith(("train.", "val.")):
            raise ConfigError("training.scheduler.monitor must reference a train or val metric.")


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
    elif isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def _load_checkpoint_model_state(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> None:
    try:
        state = torch.load(checkpoint, map_location=device)
    except OSError as exc:
        raise ConfigError(f"could not load checkpoint {checkpoint}: {exc}") from exc
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


@dataclass(frozen=True)
class _ConfiguredCallable:
    fn: Any
    params: Mapping[str, Any]

    def __call__(self, **kwargs: Any) -> Any:
        call_kwargs = dict(self.params)
        call_kwargs.update(kwargs)
        return self.fn(**call_kwargs)


@dataclass(frozen=True)
class _RuntimeMetric:
    name: str
    callable: Any
    bindings: Mapping[str, Any]
    transforms: tuple["_RuntimeTransform", ...]


@dataclass(frozen=True)
class _RuntimeTransform:
    callable: Any
    bindings: Mapping[str, Any]


@dataclass(frozen=True)
class _RuntimeScheduler:
    scheduler: Any
    step_on: str
    monitor: str | None


def _prepare_runtime_callable(
    resolution: ComponentResolution,
    *,
    device: torch.device | None,
    mode: str,
) -> Any:
    callable_obj = (
        _instantiate_component(resolution)
        if inspect.isclass(resolution.component)
        else _ConfiguredCallable(resolution.component, dict(resolution.params or {}))
    )
    if device is not None and hasattr(callable_obj, "to"):
        callable_obj = callable_obj.to(device)
    if mode == "eval" and hasattr(callable_obj, "eval"):
        callable_obj.eval()
    if mode == "train" and hasattr(callable_obj, "train"):
        callable_obj.train()
    return callable_obj


def _instantiate_component(resolution: ComponentResolution, *args: Any) -> Any:
    try:
        return resolution.component(*args, **dict(resolution.params or {}))
    except TypeError as exc:
        raise ConfigError(
            f"could not instantiate {resolution.category} component "
            f"{resolution.requested!r}: {exc}"
        ) from exc


def _prepare_metrics(
    component_resolver: ComponentResolver,
    metric_configs: list[MetricConfig],
    device: torch.device,
) -> list[_RuntimeMetric]:
    runtime_metrics: list[_RuntimeMetric] = []
    for metric_config in metric_configs:
        metric = _prepare_runtime_callable(
            component_resolver.resolve(
                "metric",
                {"name": metric_config.name, "params": metric_config.params},
            ),
            device=None,
            mode="eval",
        )
        transforms = tuple(
            _prepare_transform(component_resolver, transform_config, torch.device("cpu"))
            for transform_config in metric_config.transform
        )
        runtime_metrics.append(
            _RuntimeMetric(
                name=metric_config.name,
                callable=metric,
                bindings=metric_config.bindings,
                transforms=transforms,
            )
        )
    return runtime_metrics


def _prepare_transform(
    component_resolver: ComponentResolver,
    transform_config: TransformConfig,
    device: torch.device,
) -> _RuntimeTransform:
    transform = _prepare_runtime_callable(
        component_resolver.resolve(
            "transform",
            {"name": transform_config.name, "params": transform_config.params},
        ),
        device=device,
        mode="eval",
    )
    return _RuntimeTransform(callable=transform, bindings=transform_config.bindings)


def _build_scheduler(
    component_resolver: ComponentResolver,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> _RuntimeScheduler | None:
    scheduler_config = config.training.scheduler
    if scheduler_config is None:
        return None
    resolution = component_resolver.resolve("scheduler", scheduler_config)
    return _RuntimeScheduler(
        scheduler=_instantiate_component(resolution, optimizer),
        step_on=scheduler_config.step_on or "epoch",
        monitor=scheduler_config.monitor,
    )


def _batch_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    losses: list[Any],
    context_path: str,
) -> torch.Tensor:
    context = _build_model_context(model, batch, config=config, context_path=context_path)
    return _loss_from_context(
        context,
        config=config,
        losses=losses,
        context_path=context_path,
    )


def _build_model_context(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    context_path: str,
) -> RuntimeSourceContext:
    base_context = RuntimeSourceContext.from_batch(batch)
    model_kwargs = BindingEngine(
        base_context,
        features=config.data.features,
        targets=config.data.targets,
        path=context_path,
    ).build_kwargs(config.model.inputs, path="model.inputs")
    outputs = model(**model_kwargs)
    return RuntimeSourceContext(
        features=base_context.features,
        targets=base_context.targets,
        outputs=outputs,
    )


def _loss_from_context(
    context: RuntimeSourceContext,
    *,
    config: ExperimentConfig,
    losses: list[Any],
    context_path: str,
) -> torch.Tensor:
    objective_losses = []
    for index, (objective, loss_fn) in enumerate(zip(config.training.objectives, losses)):
        loss_kwargs = _bound_kwargs(
            objective.bindings,
            context,
            config=config,
            context_path=context_path,
            path=f"training.objectives[{index}].bindings",
        )
        objective_losses.append(
            _call_runtime_callable(
                loss_fn,
                loss_kwargs,
                path=f"training.objectives[{index}].loss",
            )
            * objective.weight
        )
    return _sum_losses(objective_losses)


def _evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    *,
    config: ExperimentConfig,
    losses: list[Any],
    metrics: list[_RuntimeMetric],
    device: torch.device,
    split_name: str,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {"loss": float("nan")}
    was_training = model.training
    loss_modes = _set_callables_mode(losses, mode="eval")
    model.eval()
    total_loss = 0.0
    total_samples = 0
    accumulator = _DatasetAccumulator() if metrics else None
    try:
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                context_path = f"{split_name}.batch"
                context = _build_model_context(
                    model,
                    batch,
                    config=config,
                    context_path=context_path,
                )
                loss = _loss_from_context(
                    context,
                    config=config,
                    losses=losses,
                    context_path=context_path,
                )
                batch_size = _context_batch_size(context)
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_samples += batch_size
                if accumulator is not None:
                    accumulator.add(context)
    finally:
        _restore_callable_modes(losses, loss_modes)
        if was_training:
            model.train()

    result = {"loss": total_loss / total_samples if total_samples else float("nan")}
    if accumulator is not None:
        dataset_context = accumulator.context()
        for index, metric in enumerate(metrics):
            result[metric.name] = _evaluate_metric(
                metric,
                dataset_context,
                config=config,
                path=f"evaluation.metrics[{index}]",
            )
    return result


def _evaluate_metric(
    metric: _RuntimeMetric,
    context: RuntimeSourceContext,
    *,
    config: ExperimentConfig,
    path: str,
) -> float:
    selected_output = None
    if metric.transforms:
        selected_output = _infer_metric_output_selection(
            metric.bindings,
            context.outputs,
            path=f"{path}.bindings",
        )
    transformed_context = _apply_metric_transforms(
        context,
        metric.transforms,
        config=config,
        path=path,
        selected_output=selected_output,
    )
    kwargs = _bound_or_default_kwargs(
        metric.bindings,
        transformed_context,
        config=config,
        default_input=transformed_context.outputs,
        context_path=path,
        path=f"{path}.bindings",
    )
    return _as_float(
        _call_runtime_callable(metric.callable, kwargs, path=path),
        path=path,
    )


def _apply_metric_transforms(
    context: RuntimeSourceContext,
    transforms: tuple[_RuntimeTransform, ...],
    *,
    config: ExperimentConfig,
    path: str,
    selected_output: str | None,
) -> RuntimeSourceContext:
    current = context
    for index, transform in enumerate(transforms):
        if transform.bindings:
            kwargs = _bound_kwargs(
                transform.bindings,
                current,
                config=config,
                context_path=path,
                path=f"{path}.transform[{index}].bindings",
            )
        else:
            kwargs = {
                "input": _select_default_transform_input(
                    current.outputs,
                    selected_output,
                    path=f"{path}.transform[{index}]",
                )
            }
        transformed = _call_runtime_callable(
            transform.callable,
            kwargs,
            path=f"{path}.transform[{index}]",
        )
        current = RuntimeSourceContext(
            features=current.features,
            targets=current.targets,
            outputs=_replace_selected_output(
                current.outputs,
                transformed,
                selected_output,
            ),
        )
    return current


def _infer_metric_output_selection(
    bindings: Mapping[str, Any],
    outputs: Any,
    *,
    path: str,
) -> str | None:
    if bindings:
        selected: set[str | None] = set()
        for argument_name, binding in bindings.items():
            source = getattr(binding, "source", None)
            if getattr(source, "namespace", None) == "outputs":
                selected.add(getattr(source, "name", None))
        if len(selected) > 1:
            return None
        if selected:
            return next(iter(selected))

    if isinstance(outputs, Mapping):
        if len(outputs) == 1:
            return next(iter(outputs))
        if len(outputs) > 1:
            return None
    return None


def _select_default_transform_input(outputs: Any, selected_output: str | None, *, path: str) -> Any:
    if selected_output is None:
        if isinstance(outputs, Mapping) and len(outputs) > 1:
            raise ConfigError(
                f"{path} cannot infer which output to transform; bind the transform input explicitly."
            )
        if isinstance(outputs, Mapping):
            return next(iter(outputs.values()))
        return outputs
    if not isinstance(outputs, Mapping):
        if selected_output is None:
            return outputs
        raise ConfigError(f"{path} cannot select outputs.{selected_output} from a single output.")
    try:
        return outputs[selected_output]
    except KeyError as exc:
        raise ConfigError(f"{path} selected output {selected_output!r} is missing.") from exc


def _replace_selected_output(outputs: Any, transformed: Any, selected_output: str | None) -> Any:
    if selected_output is None:
        if isinstance(outputs, Mapping) and len(outputs) == 1:
            selected_output = next(iter(outputs))
        else:
            return transformed
    if isinstance(outputs, Mapping):
        updated = dict(outputs)
        updated[selected_output] = transformed
        return updated
    return transformed


class _DatasetAccumulator:
    def __init__(self) -> None:
        self._features: dict[str, list[torch.Tensor]] = {}
        self._targets: dict[str, list[torch.Tensor]] = {}
        self._outputs: list[Any] = []

    def add(self, context: RuntimeSourceContext) -> None:
        _append_tensor_mapping(self._features, context.features)
        _append_tensor_mapping(self._targets, context.targets)
        self._outputs.append(_detach_cpu(context.outputs))

    def context(self) -> RuntimeSourceContext:
        return RuntimeSourceContext(
            features=_concat_tensor_mapping(self._features),
            targets=_concat_tensor_mapping(self._targets),
            outputs=_concat_outputs(self._outputs),
        )


def _bound_kwargs(
    bindings: Mapping[str, Any],
    context: RuntimeSourceContext,
    *,
    config: ExperimentConfig,
    context_path: str,
    path: str,
) -> dict[str, torch.Tensor]:
    return BindingEngine(
        context,
        features=config.data.features,
        targets=config.data.targets,
        path=context_path,
    ).build_kwargs(bindings, path=path)


def _bound_or_default_kwargs(
    bindings: Mapping[str, Any],
    context: RuntimeSourceContext,
    *,
    config: ExperimentConfig,
    default_input: Any,
    context_path: str,
    path: str,
) -> dict[str, Any]:
    if not bindings:
        return {"input": default_input}
    return _bound_kwargs(
        bindings,
        context,
        config=config,
        context_path=context_path,
        path=path,
    )


def _call_runtime_callable(callable_obj: Any, kwargs: Mapping[str, Any], *, path: str) -> Any:
    try:
        return callable_obj(**dict(kwargs))
    except TypeError as exc:
        raise ConfigError(f"could not call {path}: {exc}") from exc


def _step_scheduler(
    scheduler: _RuntimeScheduler | None,
    *,
    when: str,
    values: Mapping[str, float] | None = None,
) -> None:
    if scheduler is None or scheduler.step_on != when:
        return
    if when == "metric":
        monitor = scheduler.monitor
        if monitor is None:
            raise ConfigError("training.scheduler.monitor is required when step_on is metric.")
        if values is None or monitor not in values:
            raise ConfigError(f"training.scheduler.monitor {monitor!r} was not produced.")
        scheduler.scheduler.step(values[monitor])
        return
    scheduler.scheduler.step()


def _record_scheduler_metrics(
    history: dict[str, list[float]],
    values: Mapping[str, float],
) -> None:
    for name, value in values.items():
        history.setdefault(name, []).append(value)


def _set_callables_mode(callables: list[Any], *, mode: str) -> dict[int, bool]:
    previous: dict[int, bool] = {}
    for index, callable_obj in enumerate(callables):
        if not hasattr(callable_obj, "training"):
            continue
        previous[index] = bool(callable_obj.training)
        if mode == "eval" and hasattr(callable_obj, "eval"):
            callable_obj.eval()
        elif mode == "train" and hasattr(callable_obj, "train"):
            callable_obj.train()
    return previous


def _restore_callable_modes(callables: list[Any], previous: Mapping[int, bool]) -> None:
    for index, was_training in previous.items():
        callable_obj = callables[index]
        if was_training and hasattr(callable_obj, "train"):
            callable_obj.train()
        elif not was_training and hasattr(callable_obj, "eval"):
            callable_obj.eval()


def _context_batch_size(context: RuntimeSourceContext) -> int:
    for tensors in (context.features, context.targets):
        for tensor in tensors.values():
            return int(tensor.shape[0])
    if isinstance(context.outputs, torch.Tensor):
        return int(context.outputs.shape[0])
    if isinstance(context.outputs, Mapping):
        for value in context.outputs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
    raise ConfigError("could not determine evaluation batch size.")


def _append_tensor_mapping(
    destination: dict[str, list[torch.Tensor]],
    values: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in values.items():
        destination.setdefault(name, []).append(tensor.detach().cpu())


def _concat_tensor_mapping(values: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(tensors, dim=0) for name, tensors in values.items()}


def _detach_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    return value


def _concat_outputs(values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        return {
            key: _concat_outputs([value[key] for value in values])
            for key in first
        }
    return values


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
) -> None:
    torch.save(model.state_dict(), path)


def _is_better(value: float, best: float | None, *, mode: str) -> bool:
    if not isfinite(value):
        return False
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ConfigError("saving.save_best.mode must be one of: min, max.")


def _as_float(value: Any, *, path: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ConfigError(f"{path} must return a scalar value.")
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ConfigError(f"{path} must return a scalar number.")


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
