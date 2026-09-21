"""CNP command-line entry point.

Only static audit and dry-run planning are enabled by CNP-001.  Research modes
are deliberately rejected until the separately scoped CNP-002+ instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running a tracked script directly is supported in addition to an editable
# install.  This only locates this repository's own source tree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

def main() -> int:
    from apc.cnp.data import canonical_json_hash
    from apc.cnp.seed_registry import DEFAULT_REGISTRY_PATH, audit_cnp_seed_registry

    def config(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="CNP v1 static audit and future runner entry point"
    )
    parser.add_argument(
        "mode", choices=("audit", "dry-run", "develop", "confirm", "adapt", "report")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_REGISTRY_PATH)
    arguments = parser.parse_args()
    loaded_config = config(arguments.config)
    if arguments.mode == "audit":
        print(json.dumps(audit_cnp_seed_registry(arguments.config), sort_keys=True, indent=2))
        return 0
    if arguments.mode == "dry-run":
        audit = audit_cnp_seed_registry(arguments.config)
        print(
            json.dumps(
                {
                    "status": "PLAN_READY_NO_MODEL_OR_DATA_ACCESS",
                    "config_hash": canonical_json_hash(loaded_config),
                    "seed_audit": audit,
                    "enabled_modes": ["audit", "dry-run"],
                    "blocked_modes": ["develop", "confirm", "adapt", "report"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    raise SystemExit(
        f"CNP mode {arguments.mode!r} is not enabled by CNP-001; "
        "run the separately authorized CNP-002+ task first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
