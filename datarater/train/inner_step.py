from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping

import torch
from torch.nn.utils import stateless

from datarater.data.datamodule import SequenceBatch
from datarater.models.rater import DataRater
from datarater.models.tiny_lm import TinyLM
from datarater.utils.attention import sdpa_ctx


@dataclass(frozen=True)
class InnerLoopConfig:
    steps: int
    lr: float
    grad_clip: float | None = None


@dataclass
class InnerLoopResult:
    final_params: Dict[str, torch.Tensor]
    train_loss: torch.Tensor
    val_loss: torch.Tensor


def _functional_model_call(model: TinyLM, params: Mapping[str, torch.Tensor], batch: SequenceBatch) -> torch.Tensor:
    model_state: Dict[str, torch.Tensor] = {**params}
    for name, buffer in model.named_buffers():
        model_state[name] = buffer
    return stateless.functional_call(model, model_state, (batch.input_ids,))


def _apply_gradients(
    params: MutableMapping[str, torch.Tensor],
    grads: Iterable[torch.Tensor | None],
    lr: float,
    grad_clip: float | None,
) -> Dict[str, torch.Tensor]:
    updated: Dict[str, torch.Tensor] = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            raise RuntimeError(f"Gradient for parameter {name} is None")
        grad_to_use = grad
        if grad_clip is not None:
            grad_to_use = torch.clamp(grad_to_use, min=-grad_clip, max=grad_clip)
        updated[name] = param - lr * grad_to_use
    return updated


def run_inner_loop(
    model: TinyLM,
    rater: DataRater,
    train_batches: Iterator[SequenceBatch],
    val_batches: Iterable[SequenceBatch],
    config: InnerLoopConfig,
) -> InnerLoopResult:
    params: Dict[str, torch.Tensor] = {
        name: param.detach().clone().requires_grad_(True) for name, param in model.named_parameters()
    }
    device = next(model.parameters()).device

    train_losses: list[torch.Tensor] = []
    for _ in range(config.steps):
        try:
            batch = next(train_batches)
        except StopIteration as exc:  # pragma: no cover
            raise RuntimeError("Not enough training batches for inner loop") from exc
        batch = batch.to(device)
        # Use higher-order backend (MATH) for forward pass when create_graph=True will be used
        # This includes both rater and model since both are part of the computation graph
        with sdpa_ctx("higher_order"):
            weights = rater(batch.input_ids)
            logits = _functional_model_call(model, params, batch)
            per_sequence_loss = model.compute_sequence_loss(batch.input_ids, batch.target_ids, logits=logits)
        weighted_loss = torch.mean(weights * per_sequence_loss)
        grads = torch.autograd.grad(weighted_loss, params.values(), create_graph=True)
        params = _apply_gradients(params, grads, lr=config.lr, grad_clip=config.grad_clip)
        train_losses.append(weighted_loss.detach())

    val_loss_accum = torch.tensor(0.0, device=device)
    total_batches = 0
    for val_batch in val_batches:
        val_batch = val_batch.to(device)
        logits = _functional_model_call(model, params, val_batch)
        per_sequence_loss = model.compute_sequence_loss(val_batch.input_ids, val_batch.target_ids, logits=logits)
        val_loss_accum = val_loss_accum + per_sequence_loss.mean()
        total_batches += 1
    if total_batches == 0:
        raise ValueError("val_batches must not be empty")
    val_loss = val_loss_accum / total_batches
    mean_train_loss = torch.stack(train_losses).mean()
    return InnerLoopResult(final_params=params, train_loss=mean_train_loss, val_loss=val_loss)
