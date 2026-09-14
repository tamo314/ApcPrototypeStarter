"""CLI entry point for Task D-016's read-only SHIFT disorder x argument
paired factorial diagnostic (long sequences, lengths 6 and 10).

Loads only D-013's already-saved bundles (seeds 40-44); performs zero
training, zero optimizer construction, zero sealed access, and zero
D-013/D-014/D-015 artifact modification. See
``apc.evaluation.phase_d_d016_shift_disorder_argument_factorial`` for the
full method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.evaluation.phase_d_d016_shift_disorder_argument_factorial import run  # noqa: E402


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "task": result["task"],
                "result": result["result"],
                "reproducibility": result["reproducibility"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
