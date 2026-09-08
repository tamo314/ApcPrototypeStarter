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

from apc.evaluation.incremental_budget_calibration import (
    REC004A_RECHECK_QUERY_EXAMPLES,
    REC004A_STEP_LADDER,
    REC004A_VALIDATION_EXAMPLES,
    REC004A_VALIDATION_FLOOR,
    IncrementalBudgetCalibrationConfig,
    run_incremental_budget_calibration_task,
)
from apc.evaluation.mirror_position_bias_repair import (
    REC004D_CHECKPOINT_INTERVAL,
    REC004D_EXISTING_VALIDATION_EXAMPLES,
    REC004D_EXISTING_VALIDATION_FLOOR,
    REC004D_MAX_UPDATES_PER_RUN,
    REC004D_RECHECK_QUERY_EXAMPLES,
    MirrorPositionBiasRepairConfig,
    run_mirror_position_bias_repair_task,
)
from apc.evaluation.mirror_position_initialization_diagnostic import (
    REC004C_CHECKPOINT_INTERVAL,
    REC004C_DESCRIPTIVE_FLOOR,
    REC004C_MAX_UPDATES_PER_INIT,
    REC004C_SCHEDULE_VALIDATION_EXAMPLES,
    MirrorPositionInitializationDiagnosticConfig,
    run_mirror_position_initialization_diagnostic_task,
)
from apc.evaluation.mirror_schedule_comparison import (
    REC004B_CHECKPOINT_INTERVAL,
    REC004B_MAX_UPDATES,
    REC004B_RECHECK_QUERY_EXAMPLES,
    REC004B_SCHEDULE_VALIDATION_EXAMPLES,
    REC004B_SCHEDULE_VALIDATION_FLOOR,
    MirrorScheduleComparisonConfig,
    run_mirror_schedule_comparison_task,
)
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    ModelBundleContractConfig,
    PilotRestoreBuildConfig,
    RecoveryBuildPlanConfig,
    run_model_bundle_contract_task,
    run_pilot_restore_build_task,
    run_recovery_build_plan_task,
)

_IMPLEMENTED_TASKS = (
    "B-C005REC-002",
    "B-C005REC-003",
    "B-C005REC-004",
    "B-C005REC-004A",
    "B-C005REC-004B",
    "B-C005REC-004C",
    "B-C005REC-004D",
)


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


def _load_rec004a_config(config_path: Path) -> IncrementalBudgetCalibrationConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = IncrementalBudgetCalibrationConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004A's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004); config requested seed={seed}"
        )
    return IncrementalBudgetCalibrationConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        step_ladder=tuple(raw.get("step_ladder", REC004A_STEP_LADDER)),
        validation_examples=raw.get("validation_examples", REC004A_VALIDATION_EXAMPLES),
        validation_floor=raw.get("validation_floor", REC004A_VALIDATION_FLOOR),
        recheck_query_examples=raw.get(
            "recheck_query_examples", REC004A_RECHECK_QUERY_EXAMPLES
        ),
    )


def _load_rec004b_config(config_path: Path) -> MirrorScheduleComparisonConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorScheduleComparisonConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004B's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A); config requested seed={seed}"
        )
    return MirrorScheduleComparisonConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        schedule_validation_examples=raw.get(
            "schedule_validation_examples", REC004B_SCHEDULE_VALIDATION_EXAMPLES
        ),
        schedule_validation_floor=raw.get(
            "schedule_validation_floor", REC004B_SCHEDULE_VALIDATION_FLOOR
        ),
        recheck_query_examples=raw.get("recheck_query_examples", REC004B_RECHECK_QUERY_EXAMPLES),
        checkpoint_interval=raw.get("checkpoint_interval", REC004B_CHECKPOINT_INTERVAL),
        max_updates=raw.get("max_updates", REC004B_MAX_UPDATES),
    )


def _load_rec004c_config(config_path: Path) -> MirrorPositionInitializationDiagnosticConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorPositionInitializationDiagnosticConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004C's Core seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A/REC-004B); config requested seed={seed}"
        )
    return MirrorPositionInitializationDiagnosticConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        schedule_validation_examples=raw.get(
            "schedule_validation_examples", REC004C_SCHEDULE_VALIDATION_EXAMPLES
        ),
        checkpoint_interval=raw.get("checkpoint_interval", REC004C_CHECKPOINT_INTERVAL),
        max_updates_per_init=raw.get("max_updates_per_init", REC004C_MAX_UPDATES_PER_INIT),
        descriptive_floor=raw.get("descriptive_floor", REC004C_DESCRIPTIVE_FLOOR),
    )


def _load_rec004d_config(config_path: Path) -> MirrorPositionBiasRepairConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    defaults = MirrorPositionBiasRepairConfig()
    seed = raw.get("seed", defaults.seed)
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004D's Core seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A/REC-004C); config requested seed={seed}"
        )
    return MirrorPositionBiasRepairConfig(
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
        seed=seed,
        existing_validation_examples=raw.get(
            "existing_validation_examples", REC004D_EXISTING_VALIDATION_EXAMPLES
        ),
        existing_validation_floor=raw.get(
            "existing_validation_floor", REC004D_EXISTING_VALIDATION_FLOOR
        ),
        recheck_query_examples=raw.get("recheck_query_examples", REC004D_RECHECK_QUERY_EXAMPLES),
        checkpoint_interval=raw.get("checkpoint_interval", REC004D_CHECKPOINT_INTERVAL),
        max_updates_per_run=raw.get("max_updates_per_run", REC004D_MAX_UPDATES_PER_RUN),
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
    elif args.task == "B-C005REC-004":
        config_path = args.config or Path("configs/phase_b_b2_model_bundle_recovery_rec004.yaml")
        rec004_config = _load_rec004_config(config_path)
        if args.output_dir is not None:
            rec004_config = dataclasses.replace(rec004_config, output_dir=args.output_dir)
        report = run_pilot_restore_build_task(rec004_config)
        expected_result = "RG3_PASS"
    elif args.task == "B-C005REC-004A":
        config_path = args.config or Path(
            "configs/phase_b_b2_model_bundle_recovery_rec004a.yaml"
        )
        rec004a_config = _load_rec004a_config(config_path)
        if args.output_dir is not None:
            rec004a_config = dataclasses.replace(rec004a_config, output_dir=args.output_dir)
        rec004a_report = run_incremental_budget_calibration_task(rec004a_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004a_report["implementation_status"],
                    "calibration_status": rec004a_report["calibration_status"],
                    "rg3_recheck": rec004a_report["rg3_recheck"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004A was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004a_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1
    elif args.task == "B-C005REC-004B":
        config_path = args.config or Path(
            "configs/phase_b_b2_model_bundle_recovery_rec004b.yaml"
        )
        rec004b_config = _load_rec004b_config(config_path)
        if args.output_dir is not None:
            rec004b_config = dataclasses.replace(rec004b_config, output_dir=args.output_dir)
        rec004b_report = run_mirror_schedule_comparison_task(rec004b_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004b_report["implementation_status"],
                    "paired_comparison_status": rec004b_report["paired_comparison_status"],
                    "mirror_candidate_status": rec004b_report["mirror_candidate_status"],
                    "rg3_recheck": rec004b_report["rg3_recheck"],
                    "rec005_eligible": rec004b_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004B was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004b_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1
    elif args.task == "B-C005REC-004C":
        config_path = args.config or Path(
            "configs/phase_b_b2_model_bundle_recovery_rec004c.yaml"
        )
        rec004c_config = _load_rec004c_config(config_path)
        if args.output_dir is not None:
            rec004c_config = dataclasses.replace(rec004c_config, output_dir=args.output_dir)
        rec004c_report = run_mirror_position_initialization_diagnostic_task(rec004c_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004c_report["implementation_status"],
                    "source_audit_status": rec004c_report["source_audit_status"],
                    "execution_contract_status": rec004c_report["execution_contract_status"],
                    "initialization_experiment_status": (
                        rec004c_report["initialization_experiment_status"]
                    ),
                    "selected_init": rec004c_report["selected_init"],
                    "child_bundle": rec004c_report["child_bundle"],
                    "rg3_recheck": rec004c_report["rg3_recheck"],
                    "rec005_eligible": rec004c_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004C was executed (diagnostic only -- no candidate "
            "selected, no child bundle, no RG3 recheck). B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004c_report["execution_contract_status"] == "PASS" else 1
    else:  # B-C005REC-004D
        config_path = args.config or Path(
            "configs/phase_b_b2_model_bundle_recovery_rec004d.yaml"
        )
        rec004d_config = _load_rec004d_config(config_path)
        if args.output_dir is not None:
            rec004d_config = dataclasses.replace(rec004d_config, output_dir=args.output_dir)
        rec004d_report = run_mirror_position_bias_repair_task(rec004d_config)
        print(
            json.dumps(
                {
                    "implementation_status": rec004d_report["implementation_status"],
                    "source_audit_status": rec004d_report["source_audit_status"],
                    "candidate_status": rec004d_report["candidate_status"],
                    "rg3_recheck": rec004d_report["rg3_recheck"],
                    "rec005_eligible": rec004d_report["rec005_eligible"],
                },
                indent=2,
            )
        )
        print(
            "STOP: only B-C005REC-004D was executed. B-C005REC-005 onward and "
            "B-C006/Task Inference remain blocked pending an explicit next user "
            "instruction."
        )
        return 0 if rec004d_report["rg3_recheck"] == "RG3_RECHECK_PASS" else 1

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
