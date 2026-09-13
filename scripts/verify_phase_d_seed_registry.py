"""Run the D-005 static seed-registry audit without touching models or data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep this pre-execution audit runnable from a source checkout without an editable
# install.  The imported module itself performs only AST/JSON reads.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apc.evaluation.phase_d_seed_registry import (
    SeedRegistryAuditError,
    audit_phase_d_seed_registry,
)


def main() -> int:
    try:
        result = audit_phase_d_seed_registry()
    except SeedRegistryAuditError as exc:
        print(f"Phase D seed registry audit: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
