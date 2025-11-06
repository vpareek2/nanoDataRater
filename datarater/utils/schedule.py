from __future__ import annotations

import math


def linear_warmup_cosine_decay(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Compute a learning rate with linear warmup followed by cosine decay."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * float(step) / float(warmup_steps)
    progress = (min(step, total_steps) - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine
