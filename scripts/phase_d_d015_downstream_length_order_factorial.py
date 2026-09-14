"""CLI entry point for Task D-015's read-only downstream primitive
length x input-order paired factorial diagnostic.

Loads only D-013's already-saved LOCAL_SORT_REPAIR bundles (seeds 40-44);
performs zero training, zero optimizer construction, zero sealed access,
and zero D-013/D-014 artifact modification. See
``apc.evaluation.phase_d_d015_downstream_length_order_factorial`` for the
full method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.evaluation.phase_d_d015_downstream_length_order_factorial import (  # noqa: E402
    _to_jsonable,
    run,
)


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "task": result["task"],
                "result": result["result"],
                "reproducibility": _to_jsonable(result["reproducibility"]),
                "wall_clock_seconds": result["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
