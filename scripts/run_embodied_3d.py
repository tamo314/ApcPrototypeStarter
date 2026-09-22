"""Run from a checkout or an editable installation; see --help."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apc.embodied.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
