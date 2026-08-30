"""Checkpoint save/reload for the fixed dense baseline.

A checkpoint bundles the model weights with enough context (optimizer
state, step, and the run config) to resume or audit a run, per the
`runs/<run>/checkpoint/` artifact convention in `README.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
) -> None:
    """Save model/optimizer state, step, and config to `path`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": config,
        },
        out_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by `save_checkpoint` into `model` (and
    `optimizer`, if given) in place. Returns the full checkpoint dict."""
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
