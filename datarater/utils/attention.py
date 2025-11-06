from __future__ import annotations

import contextlib
import os
import threading
from typing import Literal

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    # Fallback for older PyTorch versions
    sdpa_kernel = None
    SDPBackend = None

# Thread-local storage to track if we're in a higher-order context
_thread_local = threading.local()


def _is_higher_order_context() -> bool:
    """Check if we're currently in a higher-order gradient context."""
    return getattr(_thread_local, "higher_order", False)


def sdpa_ctx(mode: Literal["first_order", "higher_order"]) -> contextlib.AbstractContextManager:
    """
    Return SDPA kernel context manager optimized for the given gradient order.

    - "first_order": Uses FLASH_ATTENTION -> EFFICIENT_ATTENTION -> MATH fallback chain
      (maximizes performance for standard training/inference)
    - "higher_order": Uses MATH only (required for double backward with create_graph=True)

    When nested, inner "first_order" contexts automatically detect and respect outer
    "higher_order" contexts to ensure compatibility with double backward.

    Args:
        mode: Either "first_order" for standard training or "higher_order" for meta-learning
              with create_graph=True

    Returns:
        Context manager for SDPA kernel selection

    Examples:
        >>> with sdpa_ctx("first_order"):
        ...     output = model(inputs)  # Uses Flash Attention if available
        >>> with sdpa_ctx("higher_order"):
        ...     grads = torch.autograd.grad(loss, params, create_graph=True)  # Uses MATH backend
    """
    if sdpa_kernel is None:
        return contextlib.nullcontext()

    # Check for environment override
    env_override = os.environ.get("DATARATER_SDPA_FORCE", "").lower()
    if env_override == "math":
        return sdpa_kernel(backends=[SDPBackend.MATH])
    elif env_override == "firstorder":
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

    if mode == "higher_order":
        # Force MATH backend for higher-order gradients (double backward not supported in fused kernels)
        class HigherOrderContext:
            def __init__(self):
                self._ctx = sdpa_kernel(backends=[SDPBackend.MATH])

            def __enter__(self):
                _thread_local.higher_order = True
                return self._ctx.__enter__()

            def __exit__(self, *args):
                result = self._ctx.__exit__(*args)
                _thread_local.higher_order = False
                return result

        return HigherOrderContext()
    else:
        # First-order: prefer Flash Attention, fallback to Efficient, then MATH
        # But if we're already in a higher-order context, use MATH
        if _is_higher_order_context():
            return sdpa_kernel(backends=[SDPBackend.MATH])
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
