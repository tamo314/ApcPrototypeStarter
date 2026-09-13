"""Run or dry-review the one preregistered Phase-D D-008 executor."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.evaluation.phase_d_executor import main  # noqa: E402

if __name__ == "__main__":
    main()
