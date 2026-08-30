"""CLI smoke check: print system/PyTorch/CUDA metadata as JSON.

Usage:
    python scripts/print_system_info.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.utils.system_info import get_system_info  # noqa: E402


def main() -> None:
    print(json.dumps(get_system_info(), indent=2))


if __name__ == "__main__":
    main()
