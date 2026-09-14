"""CLI entry point for Task D-014's read-only stepwise causal localization diagnostic.

Loads only D-013's already-saved FROZEN_PARENT/LOCAL_SORT_REPAIR bundles; performs
zero training, zero optimizer construction, zero sealed access, and zero D-013
artifact modification. See ``apc.evaluation.phase_d_d014_stepwise_causal_localization``
for the full method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.evaluation.phase_d_d014_stepwise_causal_localization import run  # noqa: E402


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "task": result["task"],
                "result": result["result"],
                "attribution_counts": result["attribution_counts"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
