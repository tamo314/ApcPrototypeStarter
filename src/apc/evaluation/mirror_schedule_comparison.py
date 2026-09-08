"""B-C005REC-004B: MIRROR_HALVES Schedule Comparison & RG3 Recheck.

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md`. Inserted
between `B-C005REC-004A`'s `VALIDATION_TARGET_NOT_MET` (ADR-0096) and
`B-C005REC-005`: compares two LR-schedule conditions for MIRROR_HALVES only,
same shared initial weights, same 6000-step per-step training data stream,
differing ONLY in `CosineAnnealingLR`'s `T_max` --

  - ``A_FIXED_TMAX_1000``: T_max stays 1000 (REC-004A's own
    ``MECHANICAL_EXTENSION_FIXED_T_MAX`` recipe, replayed here as the
    control, from a *new* shared initial state -- not a replay of REC-004A's
    own run),
  - ``B_SINGLE_DECAY_6000``: T_max=6000, a single monotonic decay across all
    6000 steps.

CYCLE_FOUR/ROTATE_TRIPLETS/SWAP_ENDS are NOT retrained here -- their
REC-004A step=4000 checkpoints (read-only, `checkpoints/<op>_step4000.pt`
under REC-004A's own run directory) are fixed candidates, reused as-is. A
child bundle is only ever built and independently rechecked if MIRROR_HALVES
selects a passing condition at the decisive step=6000 checkpoint
(``B`` primary, ``A`` a preregistered fallback) -- this task's own D1 gate.

Stage lettering (A-E) matches the task doc:
  A. audit the real parent/candidate artifacts and recipe/contract facts,
     reconstruct the frozen parent runtime, and reproduce the 3 fixed
     candidates' reported step=4000 validation EM against the same
     validation split REC-004A used (source verification, not selection).
  B. fix the two-condition comparison protocol (LR-trace preflight against
     the closed-form CosineAnnealingLR formula, data roles, selection rule)
     before any new training.
  C. train MIRROR_HALVES under both conditions from one shared initial
     state, checkpointing every 500 steps up to 6000; build the paired
     A-vs-B delta at the decisive step.
  D. select a candidate by the preregistered B-primary/A-fallback rule and
     build a child `ModelBundleManifest` (parent = REC-004's own published
     bundle, since REC-004A never published a child) with at most 4 changed
     physical slots.
  E. independent `rec004b_recheck_query` (new namespace) + fresh-process RG3
     recheck across all 16 operations.

Stage D/E only run if MIRROR_HALVES selects a passing condition
(``mirror_candidate_status in {"B_SELECTED", "A_SELECTED"}``); otherwise the
run stops after Stage C with ``mirror_candidate_status ==
"VALIDATION_TARGET_NOT_MET"`` and ``rg3_recheck == "NOT_EXECUTED"``, per the
task doc's own STOP rule (section 8.4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_NAMESPACE_ROOT,
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _predict_tokens,
    _repo_root,
    _snapshot_forbidden_cache_hashes,
    frozen_evaluation,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _derive_local_seed,
    _generate_parameter_free_examples,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004B_TASK_ID",
    "REC004B_TARGET_OPERATION",
    "REC004B_FIXED_CANDIDATE_OPERATIONS",
    "REC004B_PROTECTED_OPERATIONS",
    "REC004B_PHYSICAL_IDS",
    "REC004B_CONDITIONS",
    "REC004B_CONDITION_T_MAX",
    "REC004B_MAX_UPDATES",
    "REC004B_CHECKPOINT_INTERVAL",
    "REC004B_PARENT_BUNDLE_ID",
    "MirrorScheduleComparisonConfig",
    "run_mirror_schedule_comparison_task",
]


# =============================================================================
# Constants -- pre-registered per the task doc sections 1/3/4/5, not chosen
# after seeing results. Reuses REC-004A's own recipe constants where the
# task doc says the recipe is unchanged (LR, weight decay, grad clip,
# eta_min, examples-per-step, protected-operation set, physical ids).
# =============================================================================

REC004B_TASK_ID: Final = "B-C005REC-004B"
REC004B_PARENT_TASK_ID: Final = "B-C005REC-004"  # NOT REC-004A -- it published no child.
REC004B_CANDIDATE_SOURCE_TASK_ID: Final = "B-C005REC-004A"

REC004B_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004B_FIXED_CANDIDATE_OPERATIONS: Final[tuple[str, ...]] = (
    "CYCLE_FOUR",
    "ROTATE_TRIPLETS",
    "SWAP_ENDS",
)
REC004B_PHYSICAL_IDS: Final[dict[str, int]] = dict(ibc.REC004A_PHYSICAL_IDS)
REC004B_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = ibc.REC004A_PROTECTED_OPERATIONS

REC004B_CONDITIONS: Final[tuple[str, ...]] = ("A_FIXED_TMAX_1000", "B_SINGLE_DECAY_6000")
REC004B_CONDITION_T_MAX: Final[dict[str, int]] = {
    "A_FIXED_TMAX_1000": 1000,
    "B_SINGLE_DECAY_6000": 6000,
}
REC004B_MAX_UPDATES: Final = 6000
REC004B_CHECKPOINT_INTERVAL: Final = 500
REC004B_DECISIVE_STEP: Final = 6000

REC004B_SCHEDULE_VALIDATION_EXAMPLES: Final = 1024
REC004B_SCHEDULE_VALIDATION_FLOOR: Final = 0.95
REC004B_RECHECK_QUERY_EXAMPLES: Final = 1024

# Reuses REC-004A's exact split so the two runs' schedule/budget validation
# sets are the same fixed 1024 MIRROR_HALVES examples (hash-checked in Stage
# A against the reported step=4000/6000 EM values, not re-derived).
REC004B_SCHEDULE_VALIDATION_SPLIT: Final = ibc.REC004A_BUDGET_VALIDATION_SPLIT
# New namespace for the final independent recheck -- never reused for
# selection (task doc section 3.4's "used_for_checkpoint_selection: False").
REC004B_RECHECK_QUERY_SPLIT: Final = "rec004b_recheck_query"
REC004B_TRAIN_SPLIT_LABEL: Final = ibc.REC004A_TRAIN_SPLIT_LABEL  # "train"

REC004B_PARENT_BUNDLE_ID: Final = ibc.REC004A_PARENT_BUNDLE_ID  # REC-004's own published bundle
REC004B_PARENT_MANIFEST_PATH: Final = ibc.REC004A_PARENT_MANIFEST_PATH

REC004B_FIXED_CANDIDATE_STEP: Final = 4000
REC004B_FIXED_CANDIDATE_SOURCE_DIR: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/checkpoints"
)
REC004B_FIXED_CANDIDATE_SELECTED_RECIPE_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/selected_recipe.json"
)

# Recipe values unchanged from REC-004/REC-004A -- reused, not re-derived.
REC004B_OPERATOR_LR: Final = ibc.REC004A_OPERATOR_LR
REC004B_OPERATOR_WEIGHT_DECAY: Final = ibc.REC004A_OPERATOR_WEIGHT_DECAY
REC004B_OPERATOR_GRAD_CLIP: Final = ibc.REC004A_OPERATOR_GRAD_CLIP
REC004B_SCHEDULER_ETA_MIN: Final = ibc.REC004A_SCHEDULER_ETA_MIN
REC004B_EXAMPLES_PER_STEP: Final = ibc.REC004A_EXAMPLES_PER_STEP
REC004B_TRAIN_FIT_STEP_COUNT: Final = ibc.REC004A_TRAIN_FIT_STEP_COUNT

REC004B_NAMESPACE_ROOT: Final = RECOVERY_NAMESPACE_ROOT
REC004B_VOCAB_SIZE: Final = 10
REC004B_SEQUENCE_LENGTH_RANGE: Final = (6, 10)


def _rec004b_namespace(seed: int) -> Path:
    return Path(REC004B_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004b"


@dataclass(frozen=True)
class MirrorScheduleComparisonConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004b")
    seed: int = RECOVERY_PILOT_SEED
    schedule_validation_examples: int = REC004B_SCHEDULE_VALIDATION_EXAMPLES
    schedule_validation_floor: float = REC004B_SCHEDULE_VALIDATION_FLOOR
    recheck_query_examples: int = REC004B_RECHECK_QUERY_EXAMPLES
    checkpoint_interval: int = REC004B_CHECKPOINT_INTERVAL
    max_updates: int = REC004B_MAX_UPDATES


# =============================================================================
# Stage A -- audit real artifacts/recipe/contract, reconstruct the frozen
# parent runtime (needed to actually reproduce the 3 fixed candidates'
# reported EM, not merely read it), and verify source candidates.
# =============================================================================


def _fixed_candidate_checkpoint_path(op: str) -> Path:
    return REC004B_FIXED_CANDIDATE_SOURCE_DIR / f"{op}_step{REC004B_FIXED_CANDIDATE_STEP}.pt"


def _check_recipe_matches_live_defaults() -> None:
    """B2's own RECIPE_MISMATCH guard -- same recipe values REC-004A fixed,
    cross-checked live rather than assumed unchanged."""
    live_defaults = UnifiedBenchmarkConfig()
    if live_defaults.operator_lr != REC004B_OPERATOR_LR:
        raise ValueError(
            "RECIPE_MISMATCH: live UnifiedBenchmarkConfig.operator_lr="
            f"{live_defaults.operator_lr} no longer matches the recorded recipe value "
            f"{REC004B_OPERATOR_LR}"
        )
    if live_defaults.operator_weight_decay != REC004B_OPERATOR_WEIGHT_DECAY:
        raise ValueError("RECIPE_MISMATCH: operator_weight_decay drifted from recipe")
    if live_defaults.operator_grad_clip != REC004B_OPERATOR_GRAD_CLIP:
        raise ValueError("RECIPE_MISMATCH: operator_grad_clip drifted from recipe")


def _check_mirror_halves_contract() -> dict[str, Any]:
    """Confirms MIRROR_HALVES's legal-length/split-point contract on a small
    fixture via the real oracle interpreter, rather than assuming a
    transform rule from the operation's name."""
    op_obj = get_operation(REC004B_TARGET_OPERATION)

    def _run(seq: tuple[int, ...]) -> tuple[int, ...]:
        prog = Program(steps=(ProgramStep(operation=REC004B_TARGET_OPERATION, params={}),))
        return run_program(prog, seq, max(seq) + 1).output_tokens

    even_input = (10, 20, 30, 40, 50, 60)
    even_expected = (30, 20, 10, 60, 50, 40)  # mid=3: reversed(seq[:3]) + reversed(seq[3:])
    even_actual = _run(even_input)

    odd_input = (1, 2, 3, 4, 5, 6, 7)
    odd_expected = (3, 2, 1, 7, 6, 5, 4)  # mid=3 (floor): reversed(seq[:3]) + reversed(seq[3:])
    odd_actual = _run(odd_input)

    return {
        "min_input_length": 2,
        "source": "src/apc/environments/operations.py:533-555",
        "is_parameter_free_live_check": not op_obj.required_argument_names,
        "output_length_is_identity_live_check": op_obj.output_length(7) == 7
        and op_obj.output_length(9) == 9,
        "even_length_fixture": {
            "input": list(even_input),
            "expected_output": list(even_expected),
            "actual_output": list(even_actual),
            "matches": even_actual == even_expected,
        },
        "odd_length_fixture": {
            "input": list(odd_input),
            "expected_output": list(odd_expected),
            "actual_output": list(odd_actual),
            "matches": odd_actual == odd_expected,
        },
        "padding_masking": (
            "sequences right-padded to batch max with PAD; loss uses IGNORE_INDEX "
            "over the full output_length (no operation-specific position exclusion)"
        ),
    }


def run_stage_a(
    config: MirrorScheduleComparisonConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_a")

    # A1: parent + 3 fixed candidates, real files, real hashes.
    parent_manifest, _parent_manifest_raw = ibc._load_parent_manifest()
    diagnostic_load = mb.load_bundle(
        parent_manifest, mode="diagnostic", expected_primitive_count=16
    )

    if not REC004B_FIXED_CANDIDATE_SELECTED_RECIPE_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {REC004B_FIXED_CANDIDATE_SELECTED_RECIPE_PATH} "
            "not found"
        )
    rec004a_selected_recipe = json.loads(
        REC004B_FIXED_CANDIDATE_SELECTED_RECIPE_PATH.read_text(encoding="utf-8")
    )

    fixed_candidate_files: dict[str, Any] = {}
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    for op in REC004B_FIXED_CANDIDATE_OPERATIONS:
        ckpt_path = _fixed_candidate_checkpoint_path(op)
        if not ckpt_path.is_file():
            raise mb.MissingArtifactError(f"SOURCE_CANDIDATE_UNVERIFIABLE: {ckpt_path} not found")
        sd = dict(mb.load_state_dict(ckpt_path))
        fixed_candidate_state_dicts[op] = sd
        reported = rec004a_selected_recipe["selection"][op]["all_checkpoint_validation_em"].get(
            "4000"
        )
        fixed_candidate_files[op] = {
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_state_hash": mb.canonical_state_hash(sd),
            "reported_step4000_validation_em": reported,
            "reported_step4000_selected_by_rec004a": (
                rec004a_selected_recipe["selection"][op]["selected_step"] == 4000
            ),
        }

    physical_ids_by_op = {p.operation_name: p.physical_id for p in parent_manifest.primitives}
    for op, expected_pid in REC004B_PHYSICAL_IDS.items():
        actual = physical_ids_by_op.get(op)
        if actual != expected_pid:
            raise mb.IncompleteBundleError(
                f"parent manifest declares physical_id={actual} for {op}, "
                f"expected the pre-registered {expected_pid}"
            )

    # A2: recipe/contract facts.
    _check_recipe_matches_live_defaults()
    operation_contract = _check_mirror_halves_contract()
    if not (
        operation_contract["even_length_fixture"]["matches"]
        and operation_contract["odd_length_fixture"]["matches"]
    ):
        raise ValueError(
            "MIRROR_HALVES contract fixture mismatch -- refusing to proceed with an "
            "unverified transform rule"
        )

    # A3: qualification scope (mirrors REC-004A's A5; same parent, same facts).
    if not ibc.REC004A_PARENT_QUALIFICATION_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {ibc.REC004A_PARENT_QUALIFICATION_PATH} not found"
        )
    parent_qualification = json.loads(
        ibc.REC004A_PARENT_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )
    capability_rejected = False
    try:
        mb.load_bundle(
            parent_manifest,
            mode="nominal",
            required_capabilities=frozenset({"non_shift_15_recovery_floor_certified"}),
            expected_primitive_count=16,
        )
    except mb.CapabilityNotQualifiedError:
        capability_rejected = True

    # Runtime reconstruction -- needed here (not deferred) so Stage A can
    # actually reproduce the 3 fixed candidates' reported EM against the
    # SAME core/bank/validation split, rather than trusting the report.
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )

    source_validation_replay: dict[str, Any] = {}
    original_slices = {
        op: {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}
        for op, pid in REC004B_PHYSICAL_IDS.items()
    }
    for op in REC004B_FIXED_CANDIDATE_OPERATIONS:
        pid = REC004B_PHYSICAL_IDS[op]
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in fixed_candidate_state_dicts[op].items()}, strict=True
        )
        eval_bank.get(pid).eval()
        row = _evaluate_one_operation(
            core, eval_bank, op_to_id, op,
            seed=config.seed, n_examples=config.schedule_validation_examples,
            split=REC004B_SCHEDULE_VALIDATION_SPLIT,
        )
        eval_bank.get(pid).load_state_dict(original_slices[op], strict=True)
        reported = fixed_candidate_files[op]["reported_step4000_validation_em"]
        source_validation_replay[op] = {
            "reproduced_correct_exact_match": row["correct_exact_match"],
            "reported_step4000_validation_em": reported,
            "matches_within_tolerance": (
                reported is not None
                and abs(row["correct_exact_match"] - reported) < 1e-9
            ),
        }
    eval_bank.eval()

    source_audit = {
        "task_id": REC004B_TASK_ID,
        "parent_task_id": REC004B_PARENT_TASK_ID,
        "candidate_source_task_id": REC004B_CANDIDATE_SOURCE_TASK_ID,
        "parent_manifest_path": str(REC004B_PARENT_MANIFEST_PATH.resolve()),
        "parent_bundle_id": parent_manifest.bundle_id,
        "parent_core_canonical_state_hash": parent_manifest.core.canonical_state_hash,
        "parent_self_load_diagnostic_checks_performed": list(diagnostic_load.checks_performed),
        "parent_unearned_capability_request_rejected": capability_rejected,
        "target_operation": REC004B_TARGET_OPERATION,
        "target_operation_physical_id": REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION],
        "fixed_candidate_operations": list(REC004B_FIXED_CANDIDATE_OPERATIONS),
        "fixed_candidate_step": REC004B_FIXED_CANDIDATE_STEP,
        "fixed_candidate_files": fixed_candidate_files,
        "source_validation_replay": source_validation_replay,
        "protected_operation_physical_ids": {
            op: physical_ids_by_op[op] for op in REC004B_PROTECTED_OPERATIONS
        },
        "operation_contract": {REC004B_TARGET_OPERATION: operation_contract},
        "qualification_scope": {
            "tier_1_diagnostic_loadable": True,
            "tier_2_non_shift_15_recovery_floor_passed": parent_qualification["criteria"][
                "non_shift_15_floor_passed"
            ],
            "tier_2_failures": parent_qualification["criteria"]["non_shift_floor_failures"],
            "tier_3_r3_reference_contract_nominal_certified": "NOT_ATTEMPTED",
            "unearned_full_capability_request_rejected": capability_rejected,
        },
        "recipe": {
            "optimizer": "torch.optim.AdamW",
            "operator_lr": REC004B_OPERATOR_LR,
            "operator_weight_decay": REC004B_OPERATOR_WEIGHT_DECAY,
            "operator_grad_clip": REC004B_OPERATOR_GRAD_CLIP,
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "scheduler_eta_min": REC004B_SCHEDULER_ETA_MIN,
            "examples_per_step": REC004B_EXAMPLES_PER_STEP,
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, default=str), encoding="utf-8"
    )

    all_candidates_verified = all(
        v["matches_within_tolerance"] for v in source_validation_replay.values()
    )
    if not all_candidates_verified:
        raise ValueError(
            "SOURCE_CANDIDATE_UNVERIFIABLE: reproduced step=4000 validation EM does not match "
            f"REC-004A's reported values: {source_validation_replay}"
        )

    return {
        "parent_manifest": parent_manifest,
        "core": core,
        "eval_bank": eval_bank,
        "op_to_id": op_to_id,
        "fixed_candidate_state_dicts": fixed_candidate_state_dicts,
        "source_audit": source_audit,
    }


# =============================================================================
# Stage B -- fix the two-condition comparison protocol, including a
# real (CPU) LR-trace preflight against the closed-form CosineAnnealingLR
# formula, before any new training.
# =============================================================================


def _closed_form_eta(u: int, t_max: int) -> float:
    return REC004B_SCHEDULER_ETA_MIN + (REC004B_OPERATOR_LR - REC004B_SCHEDULER_ETA_MIN) / 2 * (
        1 + math.cos(math.pi * u / t_max)
    )


def _verify_lr_trace(
    t_max: int, checkpoint_steps: list[int], *, tol: float = 1e-6
) -> dict[str, Any]:
    """Real (CPU, no gradients) scheduler dry run against the installed
    torch version -- confirms the actual `CosineAnnealingLR` LR sequence
    matches the closed-form formula this protocol's LR table is built from,
    rather than assuming the formula holds."""
    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy], lr=REC004B_OPERATOR_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=REC004B_SCHEDULER_ETA_MIN
    )
    checkpoint_set = set(checkpoint_steps)
    trace: dict[int, float] = {0: optimizer.param_groups[0]["lr"]}
    for u in range(1, max(checkpoint_steps) + 1):
        scheduler.step()
        if u in checkpoint_set:
            trace[u] = float(scheduler.get_last_lr()[0])
    mismatches = {
        u: {"actual": lr, "expected": _closed_form_eta(u, t_max)}
        for u, lr in trace.items()
        if abs(lr - _closed_form_eta(u, t_max)) > tol
    }
    return {
        "t_max": t_max, "trace": trace, "mismatches": mismatches, "contract_ok": not mismatches
    }


def build_protocol(
    config: MirrorScheduleComparisonConfig, stage_a: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_steps = list(range(0, config.max_updates + 1, config.checkpoint_interval))
    if checkpoint_steps[-1] != REC004B_DECISIVE_STEP:
        raise ValueError(
            "protocol construction requires the decisive step to fall on a checkpoint boundary"
        )

    lr_table: dict[str, dict[int, float]] = {}
    for condition_id, t_max in REC004B_CONDITION_T_MAX.items():
        check = _verify_lr_trace(t_max, checkpoint_steps)
        if not check["contract_ok"]:
            raise ValueError(
                f"LR_TRACE_CONTRACT_FAILURE: condition {condition_id} (T_max={t_max}) real "
                f"CosineAnnealingLR trace does not match the closed-form formula: "
                f"{check['mismatches']}"
            )
        lr_table[condition_id] = check["trace"]

    protocol = {
        "task_id": REC004B_TASK_ID,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "parent_core_canonical_state_hash": stage_a["parent_manifest"].core.canonical_state_hash,
        "model_seed": config.seed,
        "target_operation": REC004B_TARGET_OPERATION,
        "target_operation_physical_id": REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION],
        "fixed_candidates": {
            op: {
                "step": REC004B_FIXED_CANDIDATE_STEP,
                "source": str(_fixed_candidate_checkpoint_path(op).resolve()),
                "checkpoint_state_hash": stage_a["source_audit"]["fixed_candidate_files"][op][
                    "checkpoint_state_hash"
                ],
                "schedule_t_max_unchanged": 1000,
            }
            for op in REC004B_FIXED_CANDIDATE_OPERATIONS
        },
        "conditions": {
            "A_FIXED_TMAX_1000": {
                "scheduler": "CosineAnnealingLR",
                "t_max": 1000,
                "note": (
                    "MECHANICAL_EXTENSION_FIXED_T_MAX (same closed-form as REC-004A's own "
                    "recipe), but from a NEW shared initial state for this task -- not a "
                    "replay/continuation of REC-004A's own MIRROR_HALVES run."
                ),
            },
            "B_SINGLE_DECAY_6000": {
                "scheduler": "CosineAnnealingLR",
                "t_max": 6000,
                "note": "Single monotonic decay across all 6000 new updates.",
            },
        },
        "operator_lr": REC004B_OPERATOR_LR,
        "operator_weight_decay": REC004B_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004B_OPERATOR_GRAD_CLIP,
        "scheduler_eta_min": REC004B_SCHEDULER_ETA_MIN,
        "lr_definition": (
            "eta_T(u) = eta_min + (base_lr - eta_min)/2 * (1 + cos(pi*u/T)); u is the count of "
            "completed optimizer.step() calls; scheduler.step() is called once after each "
            "optimizer.step(). lr_used for update u is the LR in effect BEFORE that update's "
            "optimizer.step() (i.e. eta_T(u-1)); lr_after_scheduler is read after that update's "
            "scheduler.step() (i.e. eta_T(u))."
        ),
        "lr_trace_preflight_verified": True,
        "lr_table": lr_table,
        "max_new_updates_per_condition": config.max_updates,
        "max_new_updates_total": config.max_updates * len(REC004B_CONDITIONS),
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_steps": checkpoint_steps,
        "decisive_checkpoint": REC004B_DECISIVE_STEP,
        "examples_per_step": REC004B_EXAMPLES_PER_STEP,
        "data_roles": {
            "train": {
                "purpose": "online per-step training stream, shared identically by A and B",
                "seed_formula": f"_derive_local_seed(seed, step, '{REC004B_TRAIN_SPLIT_LABEL}:"
                f"{REC004B_TARGET_OPERATION}')",
                "used_for_checkpoint_selection": False,
            },
            "train_fit": {
                "purpose": "diagnostic only: fixed already-trained subset",
                "n_examples": REC004B_TRAIN_FIT_STEP_COUNT * REC004B_EXAMPLES_PER_STEP,
                "used_for_checkpoint_selection": False,
            },
            "schedule_validation": {
                "purpose": "A/B decisive-checkpoint selection ONLY",
                "split": REC004B_SCHEDULE_VALIDATION_SPLIT,
                "n_examples": config.schedule_validation_examples,
                "reused_from": (
                    "B-C005REC-004A budget_validation (same split/seed/n -- "
                    "hash-verified in Stage A)"
                ),
                "used_for_checkpoint_selection": True,
            },
            "source_validation_replay": {
                "purpose": "provenance confirmation of the 3 fixed candidates only",
                "used_for_checkpoint_selection": False,
            },
            "rec004b_recheck_query": {
                "purpose": "one-time independent RG3 recheck, post-selection only, new namespace",
                "split": REC004B_RECHECK_QUERY_SPLIT,
                "n_examples": config.recheck_query_examples,
                "used_for_checkpoint_selection": False,
            },
            "reference": {
                "purpose": "nominal reference-adequacy certification (not attempted here)",
                "used_for_checkpoint_selection": False,
            },
        },
        "selection_rule": (
            "if schedule_validation EM(B @ 6000) >= "
            f"{config.schedule_validation_floor}: select B; "
            "elif schedule_validation EM(A @ 6000) >= "
            f"{config.schedule_validation_floor}: select A "
            "(CONTROL_SELECTED_REPAIR_NOT_SUPPORTED); else: select null "
            "(VALIDATION_TARGET_NOT_MET). Only the step=6000 checkpoint is ever evaluated for "
            "selection; no intermediate checkpoint, no post-selection re-pick."
        ),
        "protected_operations": list(REC004B_PROTECTED_OPERATIONS),
        "forbidden_changes": [
            "Core retraining",
            "operator architecture/size change",
            "LR bounds change",
            "optimizer/loss/batch/dtype change",
            "warmup, curriculum, or data reweighting",
            "router recalibration",
            "argument scorer recalibration",
            "extending beyond 6000 new updates per condition",
            "retraining or re-selecting the 3 fixed candidates",
            "selecting from an intermediate checkpoint",
            "re-picking after the final query is scored",
        ],
    }
    protocol_for_hash = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_for_hash, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    protocol["protocol_hash"] = protocol_hash

    (config.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    return protocol


# =============================================================================
# Stage C -- shared initial state, both conditions, paired comparison.
# =============================================================================


def _new_primitive(core: Any) -> CrossPositionPrimitive:
    pid = REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION]
    return CrossPositionPrimitive(
        pid,
        CrossPositionPrimitiveConfig(
            operation=REC004B_TARGET_OPERATION,
            d_model=core.model.config.d_model,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=REC004B_VOCAB_SIZE,
            max_sequence_length=32,
        ),
        status=PrimitiveStatus.STABLE,
    )


def build_shared_initial_state(core: Any, seed: int) -> dict[str, torch.Tensor]:
    """Constructed ONCE from a fresh init draw (a new seed label, distinct
    from REC-004A's own `rec004a_init:MIRROR_HALVES`) and loaded identically
    into both conditions -- never REC-004A's own trained weights."""
    init_seed = _derive_local_seed(seed, 0, f"rec004b_shared_init:{REC004B_TARGET_OPERATION}")
    torch.manual_seed(init_seed)
    primitive = _new_primitive(core)
    return {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}


def _digest_examples(examples: list[Any]) -> str:
    payload = [[list(e.input_tokens), list(e.target_tokens)] for e in examples]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def run_one_condition(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    condition_id: str,
    initial_state: dict[str, torch.Tensor],
    config: MirrorScheduleComparisonConfig,
    output_dir: Path,
) -> dict[str, Any]:
    _guard_not_frozen(f"run_one_condition:{condition_id}")
    pid = REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION]
    device = core.device
    seed = config.seed
    t_max = REC004B_CONDITION_T_MAX[condition_id]

    primitive = _new_primitive(core)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in initial_state.items()}, strict=True)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004B_OPERATOR_LR, weight_decay=REC004B_OPERATOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=REC004B_SCHEDULER_ETA_MIN
    )

    cond_dir = output_dir / condition_id
    ckpt_dir = cond_dir / "checkpoints"
    state_dir = cond_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints: list[dict[str, Any]] = []
    lr_trace: list[dict[str, Any]] = []
    data_digests: list[str] = []
    cumulative_examples = 0
    t_start = time.time()
    diverged_at: int | None = None

    def _snapshot_and_eval(
        step: int, lr_used: float | None, lr_after: float
    ) -> dict[str, torch.Tensor]:
        state_snapshot = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
        torch.save(state_snapshot, ckpt_dir / f"step{step}.pt")
        torch.save(
            {
                "primitive_state_dict": state_snapshot,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "cumulative_updates": step,
                "condition_id": condition_id,
            },
            state_dir / f"step{step}.pt",
        )

        live_primitive = eval_bank.get(pid)
        live_primitive.load_state_dict(
            {k: v.to(device) for k, v in state_snapshot.items()}, strict=True
        )
        live_primitive.eval()
        schedule_validation = _evaluate_one_operation(
            core, eval_bank, op_to_id, REC004B_TARGET_OPERATION,
            seed=seed, n_examples=config.schedule_validation_examples,
            split=REC004B_SCHEDULE_VALIDATION_SPLIT,
        )
        train_fit = ibc._evaluate_train_fit(
            core, primitive, REC004B_TARGET_OPERATION, seed=seed,
            vocab_size=REC004B_VOCAB_SIZE, sequence_length_range=REC004B_SEQUENCE_LENGTH_RANGE,
        )
        length_strata = ibc._length_stratified_breakdown(
            core, primitive, REC004B_TARGET_OPERATION, seed=seed,
            n_examples=config.schedule_validation_examples,
            split=REC004B_SCHEDULE_VALIDATION_SPLIT,
        )
        primitive.train()

        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        checkpoints.append(
            {
                "step": step,
                "condition_id": condition_id,
                "cumulative_training_examples": cumulative_examples,
                "lr_used": lr_used,
                "lr_after_scheduler": lr_after,
                "schedule_validation": schedule_validation,
                "train_fit": train_fit,
                "length_stratified": length_strata,
                "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
                "cumulative_wall_clock_seconds": time.time() - t_start,
                "peak_vram_bytes": peak_vram,
            }
        )
        return state_snapshot

    initial_lr = optimizer.param_groups[0]["lr"]
    _snapshot_and_eval(0, lr_used=None, lr_after=initial_lr)

    for step in range(1, config.max_updates + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, REC004B_TARGET_OPERATION,
            vocab_size=REC004B_VOCAB_SIZE, sequence_length_range=REC004B_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(examples)
        data_digests.append(_digest_examples(examples))

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004B_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004B_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())
        lr_trace.append(
            {
                "u": step, "lr_used": lr_used, "lr_after_scheduler": lr_after,
                "train_loss": running_loss,
            }
        )

        if not math.isfinite(running_loss):
            diverged_at = step
            break

        if step % config.checkpoint_interval == 0:
            _snapshot_and_eval(step, lr_used, lr_after)

    for step in range(0, config.max_updates + 1, config.checkpoint_interval):
        if diverged_at is not None and step >= diverged_at and not any(
            c["step"] == step for c in checkpoints
        ):
            checkpoints.append(
                {
                    "step": step, "status": "NOT_EXECUTED",
                    "reason": f"loss diverged at step {diverged_at}",
                }
            )

    checkpoints.sort(key=lambda c: c["step"])
    primitive.eval()
    final_state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    return {
        "condition_id": condition_id,
        "t_max": t_max,
        "checkpoints": checkpoints,
        "lr_trace": lr_trace,
        "data_digests": data_digests,
        "diverged_at_step": diverged_at,
        "final_primitive_state_dict": final_state,
        "total_wall_clock_seconds": time.time() - t_start,
    }


def _em_at_step(outcome: dict[str, Any], step: int) -> float | None:
    for c in outcome["checkpoints"]:
        if c["step"] == step and "schedule_validation" in c:
            return c["schedule_validation"]["correct_exact_match"]
    return None


def paired_comparison_at_decisive_step(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    condition_outcomes: dict[str, dict[str, Any]],
    config: MirrorScheduleComparisonConfig,
) -> dict[str, Any]:
    pid = REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION]
    examples = _generate_parameter_free_examples(
        config.seed, config.schedule_validation_examples, operation=REC004B_TARGET_OPERATION,
        split=REC004B_SCHEDULE_VALIDATION_SPLIT, vocab_size=REC004B_VOCAB_SIZE,
        sequence_length_range=REC004B_SEQUENCE_LENGTH_RANGE,
    )
    original = {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}

    preds: dict[str, list[list[int]]] = {}
    for condition_id in REC004B_CONDITIONS:
        sd = condition_outcomes[condition_id]["final_primitive_state_dict"]
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in sd.items()}, strict=True
        )
        eval_bank.get(pid).eval()
        preds[condition_id] = _predict_tokens(
            core, eval_bank.get(pid), examples, REC004B_TARGET_OPERATION, None
        )
    eval_bank.get(pid).load_state_dict(original, strict=True)

    a_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds["A_FIXED_TMAX_1000"], examples, strict=True)
    ]
    b_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds["B_SINGLE_DECAY_6000"], examples, strict=True)
    ]
    both_correct = sum(1 for a, b in zip(a_correct, b_correct, strict=False) if a and b)
    a_only = sum(1 for a, b in zip(a_correct, b_correct, strict=False) if a and not b)
    b_only = sum(1 for a, b in zip(a_correct, b_correct, strict=False) if b and not a)
    both_wrong = sum(1 for a, b in zip(a_correct, b_correct, strict=False) if not a and not b)

    by_length: dict[int, dict[str, int]] = {}
    for ex, a, b in zip(examples, a_correct, b_correct, strict=False):
        bucket = by_length.setdefault(
            len(ex.input_tokens), {"n": 0, "a_correct": 0, "b_correct": 0}
        )
        bucket["n"] += 1
        bucket["a_correct"] += int(a)
        bucket["b_correct"] += int(b)

    em_a = sum(a_correct) / len(a_correct)
    em_b = sum(b_correct) / len(b_correct)
    return {
        "n_examples": len(examples),
        "pairing_valid": (
            len(preds["A_FIXED_TMAX_1000"])
            == len(preds["B_SINGLE_DECAY_6000"])
            == len(examples)
        ),
        "em_a": em_a,
        "em_b": em_b,
        "delta_b_minus_a": em_b - em_a,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "by_length": {str(k): v for k, v in sorted(by_length.items())},
    }


# =============================================================================
# Stage D -- selection (D1), child bundle (D2).
# =============================================================================


def select_candidate(
    condition_outcomes: dict[str, dict[str, Any]], floor: float
) -> dict[str, Any]:
    em_a = _em_at_step(condition_outcomes["A_FIXED_TMAX_1000"], REC004B_DECISIVE_STEP)
    em_b = _em_at_step(condition_outcomes["B_SINGLE_DECAY_6000"], REC004B_DECISIVE_STEP)
    a_diverged = condition_outcomes["A_FIXED_TMAX_1000"]["diverged_at_step"]
    b_diverged = condition_outcomes["B_SINGLE_DECAY_6000"]["diverged_at_step"]
    a_completed = a_diverged is None and em_a is not None
    b_completed = b_diverged is None and em_b is not None

    base = {"em_a_at_6000": em_a, "em_b_at_6000": em_b}
    if not (a_completed and b_completed) or em_a is None or em_b is None:
        return {**base, "selected_condition": None, "status": "PAIR_INCOMPLETE"}
    if em_b >= floor:
        return {**base, "selected_condition": "B_SINGLE_DECAY_6000", "status": "B_SELECTED"}
    if em_a >= floor:
        return {
            **base,
            "selected_condition": "A_FIXED_TMAX_1000",
            "status": "A_SELECTED",
            "note": "CONTROL_SELECTED_REPAIR_NOT_SUPPORTED",
        }
    return {**base, "selected_condition": None, "status": "VALIDATION_TARGET_NOT_MET"}


def build_child_bundle(
    config: MirrorScheduleComparisonConfig,
    parent_manifest: mb.ModelBundleManifest,
    selected_condition_id: str,
    condition_outcomes: dict[str, dict[str, Any]],
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]],
) -> tuple[mb.ModelBundleManifest, dict[str, Any]]:
    _guard_not_frozen("build_child_bundle")
    ns = _rec004b_namespace(config.seed)
    publish_dir = ns / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    core_copy = mb.legacy_import(
        Path(parent_manifest.core.file_path), publish_dir / "core",
        component_id=parent_manifest.core.component_id,
    )
    core_component = dataclasses.replace(core_copy, schema_hash=parent_manifest.core.schema_hash)
    vocab_copy = mb.legacy_import(
        Path(parent_manifest.vocabulary.file_path), publish_dir / "vocab",
        component_id=parent_manifest.vocabulary.component_id,
    )
    vocab_component = dataclasses.replace(
        vocab_copy, schema_hash=parent_manifest.vocabulary.schema_hash
    )

    router_dest = publish_dir / "router.pt"
    shutil.copy2(Path(parent_manifest.router.source_artifact), router_dest)
    router_component = dataclasses.replace(
        parent_manifest.router, source_artifact=str(router_dest.resolve())
    )

    scorer_dest = publish_dir / "argument_scorer.pt"
    shutil.copy2(Path(parent_manifest.argument_scorer.source_artifact), scorer_dest)
    scorer_component = dataclasses.replace(
        parent_manifest.argument_scorer, source_artifact=str(scorer_dest.resolve())
    )

    parent_bank_path = Path(ibc._single_bank_path(parent_manifest))
    parent_bank_sd = dict(mb.load_state_dict(parent_bank_path))
    merged_sd = dict(parent_bank_sd)

    selection_receipts: dict[str, str] = {}
    mirror_pid = REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION]
    mirror_sd = condition_outcomes[selected_condition_id]["final_primitive_state_dict"]
    for k, v in mirror_sd.items():
        merged_sd[f"_primitives.{mirror_pid}.{k}"] = v.cpu()
    mirror_em = _em_at_step(condition_outcomes[selected_condition_id], REC004B_DECISIVE_STEP)
    selection_receipts[REC004B_TARGET_OPERATION] = (
        f"REC-004B schedule comparison, condition={selected_condition_id}, "
        f"T_max={REC004B_CONDITION_T_MAX[selected_condition_id]}, step={REC004B_DECISIVE_STEP}, "
        f"schedule_validation_sequence_EM={mirror_em:.4f}"
    )

    for op in REC004B_FIXED_CANDIDATE_OPERATIONS:
        pid = REC004B_PHYSICAL_IDS[op]
        for k, v in fixed_candidate_state_dicts[op].items():
            merged_sd[f"_primitives.{pid}.{k}"] = v.cpu()
        selection_receipts[op] = (
            "REC-004A incremental budget calibration, FRESH_PAIRED_REBUILD trajectory, "
            f"imported read-only at step={REC004B_FIXED_CANDIDATE_STEP} (schedule T_max=1000 "
            f"unchanged), source={_fixed_candidate_checkpoint_path(op)}"
        )

    bank_path = (publish_dir / "primitive_bank_16.pt").resolve()
    torch.save(merged_sd, bank_path)

    replaced_ops = {REC004B_TARGET_OPERATION, *REC004B_FIXED_CANDIDATE_OPERATIONS}
    primitives = []
    for entry in parent_manifest.primitives:
        if entry.operation_name in replaced_ops:
            sliced = mb.primitive_state_dict(merged_sd, entry.physical_id)
            weights_hash = mb.canonical_state_hash(sliced)
            provenance = (
                mb.ProvenanceStatus.TRAINED_THIS_BUILD
                if entry.operation_name == REC004B_TARGET_OPERATION
                else mb.ProvenanceStatus.LEGACY_IMPORTED
            )
            primitives.append(
                dataclasses.replace(
                    entry,
                    version=f"{entry.version}-rec004b",
                    state_abi_hash=weights_hash,
                    weights_hash=weights_hash,
                    source_artifact=str(bank_path),
                    provenance_status=provenance,
                    training_receipt=selection_receipts[entry.operation_name],
                )
            )
        else:
            primitives.append(dataclasses.replace(entry, source_artifact=str(bank_path)))

    system_info = get_system_info(seed=config.seed)
    manifest = mb.build_manifest(
        schema_version=parent_manifest.schema_version,
        source_commit=system_info.get("git_commit") or "unknown",
        runtime_recipe_version="rec004b-v1",
        environment_record={
            "python_version": str(system_info["python_version"]),
            "torch_version": str(system_info["torch_version"]),
            "device_name": str(system_info.get("device_name")),
        },
        model_id=parent_manifest.model_id,
        model_seed=parent_manifest.model_seed,
        training_run_id=f"rec004b-seed{config.seed}-run-001",
        parent_bundle_ids=(parent_manifest.bundle_id,),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "diagnostic_only"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=tuple(primitives),
        router=router_component,
        argument_scorer=scorer_component,
        scoring_policy=parent_manifest.scoring_policy,
        build_recipe_hash=hashlib.sha256(b"B-C005REC-004B-mirror-schedule-repair-v1").hexdigest(),
        known_defects=("REC004B_AWAITING_RG3_RECHECK",),
        clean_build_exercised_stages=("MIRROR_SCHEDULE_COMPARISON",),
        qualification_refs=(REC004B_TASK_ID,),
    )

    self_load = mb.load_bundle(manifest, mode="nominal", expected_primitive_count=16)
    negative_test_passed = False
    try:
        mb.load_bundle(
            manifest, mode="nominal", required_capabilities=frozenset({"recovery_cohort_ready"}),
            expected_primitive_count=16,
        )
    except mb.CapabilityNotQualifiedError:
        negative_test_passed = True

    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles" / manifest.bundle_id
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = ibc._manifest_to_json_dict(manifest)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest_json, indent=2), encoding="utf-8"
    )

    manifest_paths_file = publish_dir / "manifest_paths.json"
    op_to_id = {p.operation_name: p.physical_id for p in manifest.primitives}
    manifest_paths_file.write_text(
        json.dumps({"op_to_id": op_to_id, "manifest": manifest_json}, indent=2), encoding="utf-8"
    )

    report = {
        "bundle_id": manifest.bundle_id,
        "parent_bundle_id": parent_manifest.bundle_id,
        "selected_condition": selected_condition_id,
        "content_manifest_digest": manifest.content_manifest_digest,
        "self_load_nominal_mode_checks_performed": list(self_load.checks_performed),
        "unearned_capability_request_rejected": negative_test_passed,
        "manifest_path": str(bundle_dir / "manifest.json"),
        "manifest_paths_json_for_fresh_process": str(manifest_paths_file.resolve()),
        "bank_path": str(bank_path),
        "core_path": core_component.file_path,
        "router_path": str(router_dest.resolve()),
        "scorer_path": str(scorer_dest.resolve()),
        "selection_receipts": selection_receipts,
    }
    return manifest, report


def run_embedding_consistency_check(
    config: MirrorScheduleComparisonConfig,
    core: Any,
    child_manifest: mb.ModelBundleManifest,
    selected_condition_id: str,
    condition_outcomes: dict[str, dict[str, Any]],
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """E2: confirms each of the 4 changed slots' standalone prediction
    matches its prediction inside the reconstructed child bank -- NOT a
    match against the (possibly-failing) parent prediction."""
    child_bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(child_manifest)))
    child_bank, child_op_to_id = ibc._reconstruct_16_op_bank_structure(core, config.seed)
    child_bank.to(core.device)
    child_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(child_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    child_bank.freeze_all()
    child_bank.eval()

    state_dicts_by_op = {
        REC004B_TARGET_OPERATION: (
            condition_outcomes[selected_condition_id]["final_primitive_state_dict"]
        ),
        **fixed_candidate_state_dicts,
    }
    results: dict[str, Any] = {}
    for op, sd in state_dicts_by_op.items():
        pid = REC004B_PHYSICAL_IDS[op]
        standalone = CrossPositionPrimitive(
            pid,
            CrossPositionPrimitiveConfig(
                operation=op, d_model=core.model.config.d_model, d_operator=32, n_head=4,
                d_operator_ff=64, vocab_size=REC004B_VOCAB_SIZE, max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        standalone.to(core.device)
        standalone.load_state_dict({k: v.to(core.device) for k, v in sd.items()}, strict=True)
        standalone.eval()
        examples = _generate_parameter_free_examples(
            config.seed, 64, operation=op, split="rec004b_embedding_consistency",
            vocab_size=REC004B_VOCAB_SIZE, sequence_length_range=REC004B_SEQUENCE_LENGTH_RANGE,
        )
        standalone_preds = _predict_tokens(core, standalone, examples, op, None)
        child_preds = _predict_tokens(core, child_bank.get(child_op_to_id[op]), examples, op, None)
        results[op] = {
            "standalone_matches_child_embedding": standalone_preds == child_preds,
            "n_examples": len(examples),
        }
    return {
        "task_id": REC004B_TASK_ID,
        "operations": results,
        "all_match": all(r["standalone_matches_child_embedding"] for r in results.values()),
    }


# =============================================================================
# Stage E -- independent recheck_query + fresh-process reproducibility.
# =============================================================================


def run_stage_e(
    config: MirrorScheduleComparisonConfig,
    core: Any,
    child_manifest: mb.ModelBundleManifest,
    manifest_paths_file: str,
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_e")
    op_to_id = {p.operation_name: p.physical_id for p in child_manifest.primitives}

    bank, fresh_op_to_id = ibc._reconstruct_16_op_bank_structure(core, config.seed)
    if fresh_op_to_id != op_to_id:
        raise mb.IncompleteBundleError(
            "child bank structure op_to_id mismatch during Stage E reload"
        )
    bank.to(core.device)
    bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(child_manifest)))
    bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    bank.freeze_all()
    bank.eval()

    with frozen_evaluation():
        rows = {
            op: _evaluate_one_operation(
                core, bank, op_to_id, op, seed=config.seed,
                n_examples=config.recheck_query_examples, split=REC004B_RECHECK_QUERY_SPLIT,
            )
            for op in sorted(op_to_id)
        }
        for op in rows:
            if op == "SHIFT":
                rows[op]["recovery_floor_status"] = "COHERENT_LIMITED"
                rows[op]["recovery_floor_passed"] = None
            else:
                rows[op]["recovery_floor_status"] = "EVALUATED"
                rows[op]["recovery_floor_passed"] = (
                    rows[op]["correct_exact_match"] >= config.schedule_validation_floor
                )

        in_process_predictions: dict[str, list[list[int]]] = {}
        for op in sorted(op_to_id):
            examples = _generate_parameter_free_examples(
                config.seed, config.recheck_query_examples, operation=op,
                split=REC004B_RECHECK_QUERY_SPLIT, vocab_size=REC004B_VOCAB_SIZE,
                sequence_length_range=REC004B_SEQUENCE_LENGTH_RANGE,
            )
            in_process_predictions[op] = _predict_tokens(
                core, bank.get(op_to_id[op]), examples, op, None
            )

    non_shift_ops = [op for op in rows if op != "SHIFT"]
    all_non_shift_floor_pass = all(bool(rows[op]["recovery_floor_passed"]) for op in non_shift_ops)

    all_primitive_execution = {
        "task_id": REC004B_TASK_ID,
        "non_shift_floor": config.schedule_validation_floor,
        "recheck_query_examples_per_operation": config.recheck_query_examples,
        "operations": rows,
        "all_16_operations_present": set(rows) == set(op_to_id) and len(rows) == 16,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
        "non_shift_floor_failures": [
            op for op in non_shift_ops if not rows[op]["recovery_floor_passed"]
        ],
        "shift_status": "COHERENT_LIMITED",
        "shift_correct_exact_match": rows.get("SHIFT", {}).get("correct_exact_match"),
    }

    script_path = _repo_root() / "scripts" / "rec004_fresh_process_check.py"
    scratch_cwd = Path(tempfile.gettempdir()) / f"apc_rec004b_fresh_process_cwd_seed{config.seed}"
    scratch_cwd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, str(script_path),
            "--manifest-paths", manifest_paths_file,
            "--seed", str(config.seed),
            "--sample-examples-per-op", str(config.recheck_query_examples),
            "--split", REC004B_RECHECK_QUERY_SPLIT,
        ],
        cwd=str(scratch_cwd), capture_output=True, text=True, timeout=1800, check=False,
        env=dict(os.environ),
    )
    wall = time.time() - t0

    if proc.returncode != 0:
        fresh_process_report = {
            "task_id": REC004B_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "subprocess_returncode": proc.returncode,
            "subprocess_stdout_tail": proc.stdout[-4000:],
            "subprocess_stderr_tail": proc.stderr[-4000:],
            "reproducibility_passed": False,
            "wall_clock_seconds": wall,
        }
    else:
        fresh = json.loads(proc.stdout.strip().splitlines()[-1])
        fresh_predictions = fresh["predictions_by_op"]
        mismatches = {
            op: {
                "in_process": in_process_predictions[op],
                "fresh_process": fresh_predictions.get(op),
            }
            for op in sorted(op_to_id)
            if in_process_predictions[op] != fresh_predictions.get(op)
        }
        fresh_process_report = {
            "task_id": REC004B_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "repo_root_cwd": str(_repo_root()),
            "different_working_directory_confirmed": str(scratch_cwd) != str(_repo_root()),
            "sample_examples_per_op": config.recheck_query_examples,
            "subprocess_returncode": proc.returncode,
            "no_builder_identifier_found_in_subprocess_script": fresh[
                "no_builder_identifier_found_in_this_script"
            ],
            "load_bundle_checks_performed_in_subprocess": fresh["checks_performed_by_load_bundle"],
            "bundle_id_matches": fresh["bundle_id"] == child_manifest.bundle_id,
            "per_operation_prediction_mismatches": mismatches,
            "predictions_byte_exact_for_all_16_ops": not mismatches,
            "exact_match_by_op_in_subprocess": fresh["exact_match_by_op"],
            "reproducibility_passed": (
                not mismatches
                and bool(fresh["no_builder_identifier_found_in_this_script"])
                and fresh["bundle_id"] == child_manifest.bundle_id
            ),
            "wall_clock_seconds": wall,
        }

    return {
        "all_primitive_execution": all_primitive_execution,
        "fresh_process_report": fresh_process_report,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
    }


def _protected_scope_hashes(eval_bank: Any, op_to_id: dict[str, int]) -> dict[str, str]:
    """Content hash of every protected primitive's slice (12 non-SHIFT +
    SHIFT) -- taken before and after this task's training loop so
    `freeze_audit.json` can prove (not merely assert) they never changed."""
    return {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004B_PROTECTED_OPERATIONS
    }


def _config_to_yaml_dict(config: MirrorScheduleComparisonConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "schedule_validation_examples": config.schedule_validation_examples,
        "schedule_validation_floor": config.schedule_validation_floor,
        "recheck_query_examples": config.recheck_query_examples,
        "checkpoint_interval": config.checkpoint_interval,
        "max_updates": config.max_updates,
    }


# =============================================================================
# Orchestration.
# =============================================================================


def run_mirror_schedule_comparison_task(
    config: MirrorScheduleComparisonConfig,
) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004B's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A); got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    stage_a = run_stage_a(config)
    protocol = build_protocol(config, stage_a)
    core, eval_bank, op_to_id = stage_a["core"], stage_a["eval_bank"], stage_a["op_to_id"]
    fixed_candidate_state_dicts = stage_a["fixed_candidate_state_dicts"]

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(protocol["data_roles"], indent=2, default=str), encoding="utf-8"
    )

    protected_hashes_before = _protected_scope_hashes(eval_bank, op_to_id)

    initial_state = build_shared_initial_state(core, seed)
    torch.save(initial_state, output_dir / "initial_state.pt")
    initial_state_hash = mb.canonical_state_hash(initial_state)

    mirror_pid = REC004B_PHYSICAL_IDS[REC004B_TARGET_OPERATION]
    original_mirror_slice = {
        k: v.detach().clone() for k, v in eval_bank.get(mirror_pid).state_dict().items()
    }

    condition_outcomes: dict[str, dict[str, Any]] = {}
    for condition_id in REC004B_CONDITIONS:
        eval_bank.get(mirror_pid).load_state_dict(original_mirror_slice, strict=True)
        condition_outcomes[condition_id] = run_one_condition(
            core, eval_bank, op_to_id, condition_id, initial_state, config, output_dir
        )
    eval_bank.get(mirror_pid).load_state_dict(original_mirror_slice, strict=True)

    data_stream_identical = (
        condition_outcomes["A_FIXED_TMAX_1000"]["data_digests"]
        == condition_outcomes["B_SINGLE_DECAY_6000"]["data_digests"]
    )
    if not data_stream_identical:
        raise ValueError(
            "PAIRING_CONTRACT_FAILURE: A and B did not receive identical per-step training data"
        )

    paired_comparison = paired_comparison_at_decisive_step(
        core, eval_bank, op_to_id, condition_outcomes, config
    )

    with (output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as fh:
        for condition_id, outcome in condition_outcomes.items():
            for row in outcome["lr_trace"]:
                fh.write(json.dumps({"condition_id": condition_id, **row}) + "\n")

    with (output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for condition_id, outcome in condition_outcomes.items():
            for ckpt in outcome["checkpoints"]:
                row = {k: v for k, v in ckpt.items() if k != "length_stratified"}
                fh.write(json.dumps({"condition_id": condition_id, **row}, default=str) + "\n")

    protected_hashes_after = _protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_after = {
        op: mb.canonical_state_hash(mb.load_state_dict(_fixed_candidate_checkpoint_path(op)))
        for op in REC004B_FIXED_CANDIDATE_OPERATIONS
    }
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004B_TASK_ID,
        "core_canonical_state_hash_before": stage_a["parent_manifest"].core.canonical_state_hash,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": (
            core_hash_after == stage_a["parent_manifest"].core.canonical_state_hash
        ),
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "fixed_candidate_files_hashes_before": {
            op: stage_a["source_audit"]["fixed_candidate_files"][op]["checkpoint_state_hash"]
            for op in REC004B_FIXED_CANDIDATE_OPERATIONS
        },
        "fixed_candidate_files_hashes_after": fixed_candidate_hashes_after,
        "fixed_candidate_files_unchanged": all(
            stage_a["source_audit"]["fixed_candidate_files"][op]["checkpoint_state_hash"]
            == fixed_candidate_hashes_after[op]
            for op in REC004B_FIXED_CANDIDATE_OPERATIONS
        ),
    }
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2, default=str), encoding="utf-8"
    )

    error_breakdown = {
        condition_id: [
            {"step": c["step"], "length_stratified": c.get("length_stratified")}
            for c in outcome["checkpoints"]
            if "length_stratified" in c
        ]
        for condition_id, outcome in condition_outcomes.items()
    }
    (output_dir / "error_breakdown.json").write_text(
        json.dumps(error_breakdown, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(paired_comparison, indent=2, default=str), encoding="utf-8"
    )

    selection = select_candidate(condition_outcomes, config.schedule_validation_floor)
    mirror_candidate_status = selection["status"]

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE",
        "paired_comparison_status": (
            "COMPLETE_VALID"
            if condition_outcomes["A_FIXED_TMAX_1000"]["diverged_at_step"] is None
            and condition_outcomes["B_SINGLE_DECAY_6000"]["diverged_at_step"] is None
            else "INVALID"
        ),
        "mirror_candidate_status": mirror_candidate_status,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "rec005_executed": False,
        "initial_state_hash": initial_state_hash,
        "data_stream_identical": data_stream_identical,
        "source_audit": stage_a["source_audit"],
        "protocol": protocol,
        "selection": selection,
        "paired_comparison": paired_comparison,
        "freeze_audit": freeze_audit,
        "wall_clock_seconds": time.time() - start,
    }

    fixed_candidate_manifest = {
        op: {
            "step": REC004B_FIXED_CANDIDATE_STEP,
            "checkpoint_state_hash": stage_a["source_audit"]["fixed_candidate_files"][op][
                "checkpoint_state_hash"
            ],
        }
        for op in REC004B_FIXED_CANDIDATE_OPERATIONS
    }
    (output_dir / "fixed_candidate_manifest.json").write_text(
        json.dumps(fixed_candidate_manifest, indent=2, default=str), encoding="utf-8"
    )

    if mirror_candidate_status not in ("B_SELECTED", "A_SELECTED"):
        after_hashes = _snapshot_forbidden_cache_hashes(seed)
        side_effect_audit = {
            "task_id": REC004B_TASK_ID,
            "before": before_hashes,
            "after": after_hashes,
            "shared_cache_unchanged": before_hashes == after_hashes,
        }
        (output_dir / "side_effect_audit.json").write_text(
            json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
        )
        result["side_effect_audit"] = side_effect_audit
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result

    selected_condition_id = selection["selected_condition"]
    with frozen_evaluation():
        child_manifest, publish_report = build_child_bundle(
            config, stage_a["parent_manifest"], selected_condition_id, condition_outcomes,
            fixed_candidate_state_dicts,
        )
        embedding_check = run_embedding_consistency_check(
            config, core, child_manifest, selected_condition_id, condition_outcomes,
            fixed_candidate_state_dicts,
        )
        protected_regression = ibc.run_protected_regression_check(
            ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), stage_a["parent_manifest"],
            child_manifest, core,
        )

    (output_dir / "recipe_revision.json").write_text(
        json.dumps(
            {
                "task_id": REC004B_TASK_ID,
                "parent_recipe": (
                    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/budget_protocol.json"
                ),
                "selected_condition": selected_condition_id,
                "mirror_halves_t_max": REC004B_CONDITION_T_MAX[selected_condition_id],
                "fixed_candidates_step": REC004B_FIXED_CANDIDATE_STEP,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "selected_recipe.json").write_text(
        json.dumps(
            {"task_id": REC004B_TASK_ID, "selection": selection, "publish_report": publish_report},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "bundle_lineage.json").write_text(
        json.dumps(
            {
                "child_bundle_id": child_manifest.bundle_id,
                "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
                "parent_bundle_ids_declared": list(child_manifest.parent_bundle_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "protected_regression.json").write_text(
        json.dumps(protected_regression, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "embedding_consistency.json").write_text(
        json.dumps(embedding_check, indent=2, default=str), encoding="utf-8"
    )

    stage_e = run_stage_e(
        config, core, child_manifest, publish_report["manifest_paths_json_for_fresh_process"]
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004B_TASK_ID,
        "before": before_hashes,
        "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }

    rg3_recheck_pass = all(
        [
            stage_e["all_primitive_execution"]["all_16_operations_present"],
            stage_e["all_non_shift_15_floor_passed"],
            stage_e["fresh_process_report"]["reproducibility_passed"],
            side_effect_audit["shared_cache_unchanged"],
            protected_regression["all_protected_unchanged"],
            embedding_check["all_match"],
        ]
    )
    rg3_recheck = "RG3_RECHECK_PASS" if rg3_recheck_pass else "RG3_RECHECK_FAIL"

    final_known_defects = () if rg3_recheck_pass else ("REC004B_RG3_RECHECK_FAIL",)
    final_manifest = dataclasses.replace(child_manifest, known_defects=final_known_defects)
    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles"
        / final_manifest.bundle_id
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(ibc._manifest_to_json_dict(final_manifest), indent=2), encoding="utf-8"
    )

    rg3_recheck_report = {
        "task_id": REC004B_TASK_ID,
        "result": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "selected_condition": selected_condition_id,
        "criteria": {
            "all_16_operations_coverage": (
                stage_e["all_primitive_execution"]["all_16_operations_present"]
            ),
            "non_shift_15_floor_passed": stage_e["all_non_shift_15_floor_passed"],
            "non_shift_floor_failures": (
                stage_e["all_primitive_execution"]["non_shift_floor_failures"]
            ),
            "fresh_process_reproducibility_passed": (
                stage_e["fresh_process_report"]["reproducibility_passed"]
            ),
            "shared_cache_unchanged": side_effect_audit["shared_cache_unchanged"],
            "protected_12_unchanged": protected_regression["all_protected_unchanged"],
            "embedding_consistency_all_match": embedding_check["all_match"],
        },
        "no_query_leakage": {
            "schedule_validation_split": REC004B_SCHEDULE_VALIDATION_SPLIT,
            "recheck_query_split": REC004B_RECHECK_QUERY_SPLIT,
            "distinct_splits": REC004B_SCHEDULE_VALIDATION_SPLIT != REC004B_RECHECK_QUERY_SPLIT,
        },
        "scope_note": (
            "This is a seed-10 recovery-pilot recheck with a revised MIRROR_HALVES LR schedule "
            "and 3 imported fixed candidates, not a 5-model cohort, unseen-family, or "
            "hard-negative routing result."
        ),
    }
    (output_dir / "rg3_recheck.json").write_text(
        json.dumps(rg3_recheck_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "all_primitive_execution.json").write_text(
        json.dumps(stage_e["all_primitive_execution"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "fresh_process_report.json").write_text(
        json.dumps(stage_e["fresh_process_report"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    qualification = {
        "task_id": REC004B_TASK_ID,
        "implementation_status": "COMPLETE",
        "mirror_candidate_status": mirror_candidate_status,
        "rg3_recheck": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "rec005_eligible": rg3_recheck_pass,
        "no_5_model_cohort_started_by_this_task": True,
    }
    (output_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2, default=str), encoding="utf-8"
    )

    result["rg3_recheck"] = rg3_recheck
    result["rec005_eligible"] = rg3_recheck_pass
    result["child_bundle_id"] = child_manifest.bundle_id
    result["publish_report"] = publish_report
    result["embedding_consistency"] = embedding_check
    result["protected_regression"] = protected_regression
    result["rg3_recheck_report"] = rg3_recheck_report
    result["side_effect_audit"] = side_effect_audit
    result["qualification"] = qualification
    result["wall_clock_seconds"] = time.time() - start

    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
