from __future__ import annotations

from typing import Tuple

import torch


def _validate_inputs(weights: torch.Tensor, losses: torch.Tensor) -> None:
    if weights.ndim != 1 or losses.ndim != 1:
        raise ValueError("weights and losses must be 1-D tensors")
    if weights.shape[0] != losses.shape[0]:
        raise ValueError("weights and losses must share the same length")


def filter_batch(
    weights: torch.Tensor,
    losses: torch.Tensor,
    keep_rate: float,
    mode: str = "filter",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter or reweight per-example losses according to rater scores."""

    if not 0 < keep_rate <= 1:
        raise ValueError("keep_rate must be in (0, 1]")
    mode_normalized = mode.lower()
    if mode_normalized not in {"filter", "reweight"}:
        raise ValueError("mode must be 'filter' or 'reweight'")

    _validate_inputs(weights, losses)

    if mode_normalized == "filter":
        keep_count = max(1, int(torch.ceil(torch.tensor(weights.numel() * keep_rate)).item()))
        _, sorted_indices = torch.sort(weights, descending=True)
        kept_indices = torch.sort(sorted_indices[:keep_count]).values
        return losses[kept_indices], kept_indices

    weight_mean = weights.mean()
    adjusted_losses = losses * (weights / weight_mean)
    indices = torch.arange(weights.numel(), device=weights.device)
    return adjusted_losses, indices
