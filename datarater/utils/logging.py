from __future__ import annotations

import logging
from typing import Optional


def create_logger(name: str = "datarater", level: int = logging.INFO, stream: Optional[object] = None) -> logging.Logger:
    """Create a configured :class:`logging.Logger` for experiments."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
