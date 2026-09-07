"""Explicit one-task dispatcher for the Phase B B2 model bundle recovery
series (Task B-C005REC-001..008).

Per `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` section 0 and
`docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md`: exactly one named
task runs per invocation, and there is no `--all` or implicit next-task
execution. B-C005REC-001 was a documentation/audit-only task (no `src/`
code, no dispatcher entry -- see ADR-0092); only B-C005REC-002 onward has a
dispatcher entry, and only once each is actually implemented. B-C005REC-003
(ADR-0094) fixes the full 16-primitive build DAG and preregistered recovery
protocol on CPU tiny fixtures + real-registry structural checks -- it starts
no GPU training and no 5-model run.
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
    RECOVERY_PILOT_SEED,
    ModelBundleContractConfig,
    PilotRestoreBuildConfig,
    RecoveryBuildPlanConfig,
    run_model_bundle_contract_task,
    run_pilot_restore_build_task,
    run_recovery_build_plan_task,
)

_IMPLEMENTED_TASKS = ("B-C005REC-002", "B-C005REC-003", "B-C005REC-004")


def _load_rec002_config(config_path: Path) -> ModelBundleContractConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = ModelBundleContractConfig()
    real_core_source = raw.get("real_core_source", defaults.real_core_source)
    return ModelBundleContractConfig(
        real_core_source=Path(real_core_source) if real_core_source else None,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_rec003_config(config_path: Path) -> RecoveryBuildPlanConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = RecoveryBuildPlanConfig()
    seeds = raw.get("seeds", list(defaults.seeds))
    return RecoveryBuildPlanConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seeds=tuple(seeds),
    )


def _load_rec004_config(config_path: Path) -> PilotRestoreBuildConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = PilotRestoreBuildConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004's pilot seed is pre-registered as {RECOVERY_PILOT_SEED}; "
            f"config requested seed={seed}"
        )
    return PilotRestoreBuildConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        direct_query_examples_per_operation=raw.get(
            "direct_query_examples_per_operation", defaults.direct_query_examples_per_operation
        ),
        non_shift_floor=raw.get("non_shift_floor", defaults.non_shift_floor),
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

    if args.task == "B-C005REC-002":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec002.yaml")
        rec002_config = _load_rec002_config(config_path)
        if args.output_dir is not None:
            rec002_config = dataclasses.replace(rec002_config, output_dir=args.output_dir)
        report = run_model_bundle_contract_task(rec002_config)
        expected_result = "RG1_PASS"
    elif args.task == "B-C005REC-003":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec003.yaml")
        rec003_config = _load_rec003_config(config_path)
        if args.output_dir is not None:
            rec003_config = dataclasses.replace(rec003_config, output_dir=args.output_dir)
        report = run_recovery_build_plan_task(rec003_config)
        expected_result = "RG2_PASS"
    else:  # B-C005REC-004
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004.yaml")
        rec004_config = _load_rec004_config(config_path)
        if args.output_dir is not None:
            rec004_config = dataclasses.replace(rec004_config, output_dir=args.output_dir)
        report = run_pilot_restore_build_task(rec004_config)
        expected_result = "RG3_PASS"

    next_blocked = {
        "B-C005REC-002": "B-C005REC-003 onward",
        "B-C005REC-003": "B-C005REC-004 onward",
        "B-C005REC-004": "B-C005REC-005 onward",
    }[args.task]

    print(json.dumps(report["protocol"], indent=2))
    print(
        f"STOP: only {args.task} was executed. {next_blocked} and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    result = report["protocol"]["result"]
    return 0 if result == expected_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
