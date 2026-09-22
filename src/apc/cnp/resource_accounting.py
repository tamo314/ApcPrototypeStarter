"""Portable process-memory accounting for CNP runners."""

from __future__ import annotations

import importlib
from typing import Any


def peak_process_ram_bytes() -> int:
    """Return POSIX peak RSS in bytes, or zero when the platform cannot provide it."""

    try:
        resource: Any = importlib.import_module("resource")
    except ModuleNotFoundError:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
