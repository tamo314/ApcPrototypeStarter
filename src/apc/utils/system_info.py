"""System, PyTorch/CUDA, and VRAM metadata capture for run artifacts."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import torch


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def get_system_info(seed: int | None = None) -> dict[str, Any]:
    """Collect system/device metadata for `runs/<run>/system.json`.

    CUDA fields are populated only when CUDA is available; unit tests must
    not require a GPU to pass, so callers should treat a missing/False
    `cuda_available` as an expected, valid state.
    """
    cuda_available = torch.cuda.is_available()

    info: dict[str, Any] = {
        "seed": seed,
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if cuda_available else None,
    }
    return info
