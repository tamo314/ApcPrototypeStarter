"""Explicit one-task dispatcher for the Phase B B2 post-D2 repair series.

Per `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md` Section 0: exactly one
named task runs per invocation, and there is no `--all` or implicit
next-task execution. Only `B-C005R3-001` is implemented so far; every other
task ID is rejected until it is explicitly implemented and wired in here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.post_d2_repair_benchmark import (
    PostD2ReproducibilityConfig,
    run_post_d2_reproducibility_task,
)

_IMPLEMENTED_TASKS = ("B-C005R3-001",)


def _load_r3_001_config(config_path: Path) -> PostD2ReproducibilityConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = PostD2ReproducibilityConfig()
    variants_raw = raw.get(
        "pythonhashseed_variants",
        ["<unset>" if v is None else v for v in defaults.pythonhashseed_variants],
    )
    variants = tuple(None if v in (None, "<unset>") else str(v) for v in variants_raw)
    return PostD2ReproducibilityConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        operations=tuple(raw.get("operations", defaults.operations)),
        splits=tuple(raw.get("splits", defaults.splits)),
        n_examples=raw.get("n_examples", defaults.n_examples),
        pythonhashseed_variants=variants,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B2 post-D2 repair -- explicit single-task dispatch"
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
        default=Path("configs/phase_b_b2_post_d2_reproducibility.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.task not in _IMPLEMENTED_TASKS:
        print(
            f"ERROR: task {args.task!r} is not implemented by this dispatcher. "
            f"Only {_IMPLEMENTED_TASKS} run today; every later task in "
            "docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md requires its own "
            "explicit user instruction and its own implementation before it "
            "can be dispatched here.",
            file=sys.stderr,
        )
        return 2

    config = _load_r3_001_config(args.config)
    if args.output_dir is not None:
        config = PostD2ReproducibilityConfig(
            seeds=config.seeds,
            operations=config.operations,
            splits=config.splits,
            n_examples=config.n_examples,
            pythonhashseed_variants=config.pythonhashseed_variants,
            output_dir=args.output_dir,
        )

    report = run_post_d2_reproducibility_task(config)
    print(json.dumps(report["protocol"], indent=2))
    print(
        "STOP: only B-C005R3-001 was executed. B-C005R3-002 onward and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    return 0 if report["protocol"]["result"] == "INFRASTRUCTURE_OR_PROTOCOL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
