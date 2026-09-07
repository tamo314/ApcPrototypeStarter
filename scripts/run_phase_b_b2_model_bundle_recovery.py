"""Explicit one-task dispatcher for the Phase B B2 model bundle recovery
series (Task B-C005REC-001..008).

Per `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` section 0 and
`docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md`: exactly one named
task runs per invocation, and there is no `--all` or implicit next-task
execution. B-C005REC-001 was a documentation/audit-only task (no `src/`
code, no dispatcher entry -- see ADR-0092); only B-C005REC-002 onward has a
dispatcher entry, and only once each is actually implemented.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.model_bundle_recovery import (
    ModelBundleContractConfig,
    run_model_bundle_contract_task,
)

_IMPLEMENTED_TASKS = ("B-C005REC-002",)


def _load_rec002_config(config_path: Path) -> ModelBundleContractConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = ModelBundleContractConfig()
    real_core_source = raw.get("real_core_source", defaults.real_core_source)
    return ModelBundleContractConfig(
        real_core_source=Path(real_core_source) if real_core_source else None,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B2 model bundle recovery -- explicit single-task dispatch"
    )
    parser.add_argument(
        "--task",
        required=True,
        help=f"Exact task ID to run. Implemented: {', '.join(_IMPLEMENTED_TASKS)}. "
        "No --all and no implicit next-task execution.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to this task's own configs/phase_b_b2_model_bundle_recovery_<task>.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.task not in _IMPLEMENTED_TASKS:
        print(
            f"ERROR: task {args.task!r} is not implemented by this dispatcher. "
            f"Only {_IMPLEMENTED_TASKS} run today; every later task in "
            "docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md requires its own "
            "explicit user instruction and its own implementation before it "
            "can be dispatched here.",
            file=sys.stderr,
        )
        return 2

    config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec002.yaml")
    config = _load_rec002_config(config_path)
    if args.output_dir is not None:
        config = dataclasses.replace(config, output_dir=args.output_dir)
    report = run_model_bundle_contract_task(config)
    next_blocked = "B-C005REC-003 onward"

    print(json.dumps(report["protocol"], indent=2))
    print(
        f"STOP: only {args.task} was executed. {next_blocked} and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    result = report["protocol"]["result"]
    return 0 if result == "RG1_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
