"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, PyTorch CPU, and PyTorch CUDA RNGs.

    Args:
        seed: Seed value applied to all RNGs.
        deterministic_algorithms: If True, ask PyTorch to only use
            deterministic algorithm implementations. This can be slower
            and is off by default; enable it for exact-reproducibility
            experiments rather than default training runs.
    """
    # CPython requires PYTHONHASHSEED to be in [0; 4294967295] (uint32).
    # Bound the seed so spawned subprocesses inheriting os.environ do not fail startup.
    os.environ["PYTHONHASHSEED"] = str(seed & 0xFFFFFFFF)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
