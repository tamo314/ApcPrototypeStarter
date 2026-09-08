"""B-C005REC-004C: MIRROR_HALVES Position Correspondence & Initialization
Diagnostic.

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_POSITION_INITIALIZATION_DIAGNOSTIC.md`.
Inserted between `B-C005REC-004B`'s `VALIDATION_TARGET_NOT_MET` (ADR-0097) and
`B-C005REC-005`. This is a **diagnostic-only** task -- it does not search for
a passing initialization, does not build a child bundle, does not run an RG3
recheck, and does not decide "the" mechanism behind MIRROR_HALVES's recovery
gap. It does two things:

  1. (Stages A-B, no new training) audits the real REC-004A/REC-004B source
     artifacts and MIRROR_HALVES's real operation contract (never guessed
     from the operation's name), and characterizes the *existing* saved
     checkpoints' position-level errors, padding/batching execution
     contract, and legal diagnostic-suite behavior.
  2. (Stages C-D, real GPU training) fixes 5 new initial states
     (`I01`..`I05`) for an otherwise-isolated MIRROR_HALVES primitive up
     front, then trains each for up to 6000 optimizer updates (30000 total)
     under the *same* recipe REC-004B's `A_FIXED_TMAX_1000` condition used
     (`CosineAnnealingLR(T_max=1000)`, mechanically extended past its own
     nominal length -- the *only* schedule this task uses, per the task
     doc's explicit instruction not to mix a schedule search into an
     initialization comparison), holding Core, the other 15 primitives,
     Router, ArgumentScorer, and all data-generation formulas fixed.

Stage lettering (A-E) matches the task doc:
  A. audit real source artifacts/manifests, reconstruct the frozen parent
     runtime, replay REC-004A's and REC-004B's own saved MIRROR_HALVES
     checkpoints against the same validation split those tasks used, and
     derive + verify MIRROR_HALVES's content-independent position map
     directly against the real oracle interpreter (never assumed from the
     operation's name).
  B. position-level error analysis of the 3 existing saved checkpoints
     (REC-004A step=6000, REC-004B A/B step=6000), a padding/batching
     metamorphic execution-contract check, a best-effort attention
     observation, and construction of 3 frozen diagnostic-suite fixtures
     (length-balanced, position-identifiable, content-counterfactual) --
     evaluated here against the existing checkpoints, and reused unchanged
     in Stage D against the 5 new inits.
  C. fix the 5-initialization protocol (`initialization_protocol.json`)
     before any new training: 5 distinct initial states derived from a
     namespace distinct from both REC-004A's and REC-004B's own init
     labels, a training-data stream identical across all 5 (never keyed by
     `init_id`), and REC-004B's `A_FIXED_TMAX_1000` recipe unchanged.
  D. train MIRROR_HALVES 5 times (one per init), 6000 updates each,
     checkpointing every 500 steps; run the 3 diagnostic suites and the
     existing Correct/Wrong-family/None causal controls at the decisive
     step=6000 checkpoint of every completed trial.
  E. aggregate cross-init statistics and write `final_diagnosis.json`.
     `selected_init`, `child_bundle`, and `rg3_recheck` are fixed at
     `null`/`null`/`"NOT_EXECUTED"` regardless of any individual init's
     validation EM -- this task never selects, publishes, or rechecks.

No child `ModelBundleManifest` is ever built by this task (the task doc's
own "旧parent全16slotと固定3候補はread-only" rule, section 2.3): unlike
REC-004B, there is no `build_child_bundle`/`run_stage_e` RG3-recheck
machinery here at all.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_schedule_comparison as msc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_NAMESPACE_ROOT,
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _predict_tokens,
    _snapshot_forbidden_cache_hashes,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004C_TASK_ID",
    "REC004C_TARGET_OPERATION",
    "REC004C_INIT_IDS",
    "REC004C_MAX_UPDATES_PER_INIT",
    "REC004C_TOTAL_MAX_UPDATES",
    "REC004C_CHECKPOINT_INTERVAL",
    "REC004C_T_MAX",
    "REC004C_DESCRIPTIVE_FLOOR",
    "mirror_halves_position_map",
    "MirrorPositionInitializationDiagnosticConfig",
    "run_mirror_position_initialization_diagnostic_task",
]


# =============================================================================
# Constants -- pre-registered per the task doc sections 1.4/2/5, not chosen
# after seeing results. Reuses REC-004A/REC-004B's own recipe constants
# where the task doc says the recipe is unchanged.
# =============================================================================

REC004C_TASK_ID: Final = "B-C005REC-004C"
REC004C_PARENT_TASK_ID: Final = "B-C005REC-004"  # same parent bundle REC-004A/B used
REC004C_REC004A_TASK_ID: Final = "B-C005REC-004A"
REC004C_REC004B_TASK_ID: Final = "B-C005REC-004B"

REC004C_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004C_PHYSICAL_IDS: Final[dict[str, int]] = dict(ibc.REC004A_PHYSICAL_IDS)
REC004C_TARGET_PHYSICAL_ID: Final = REC004C_PHYSICAL_IDS[REC004C_TARGET_OPERATION]

REC004C_INIT_IDS: Final[tuple[str, ...]] = ("I01", "I02", "I03", "I04", "I05")
REC004C_MAX_UPDATES_PER_INIT: Final = 6000
REC004C_TOTAL_MAX_UPDATES: Final = REC004C_MAX_UPDATES_PER_INIT * len(REC004C_INIT_IDS)
REC004C_CHECKPOINT_INTERVAL: Final = 500
REC004C_DECISIVE_STEP: Final = 6000
REC004C_T_MAX: Final = 1000  # REC-004B's A_FIXED_TMAX_1000 mechanical extension, ONLY.

# Descriptive reference value only (task doc section 6/D4): counted, never
# used to select/publish/re-gate anything in this task.
REC004C_DESCRIPTIVE_FLOOR: Final = 0.95

REC004C_SCHEDULE_VALIDATION_EXAMPLES: Final = 1024
REC004C_SCHEDULE_VALIDATION_SPLIT: Final = ibc.REC004A_BUDGET_VALIDATION_SPLIT  # same fixed 1024

REC004C_VOCAB_SIZE: Final = msc.REC004B_VOCAB_SIZE  # 10
REC004C_SEQUENCE_LENGTH_RANGE: Final = msc.REC004B_SEQUENCE_LENGTH_RANGE  # (6, 10)
REC004C_LEGAL_LENGTHS: Final[tuple[int, ...]] = tuple(
    range(REC004C_SEQUENCE_LENGTH_RANGE[0], REC004C_SEQUENCE_LENGTH_RANGE[1] + 1)
)

# Recipe values unchanged from REC-004A/REC-004B -- reused, not re-derived.
REC004C_OPERATOR_LR: Final = ibc.REC004A_OPERATOR_LR
REC004C_OPERATOR_WEIGHT_DECAY: Final = ibc.REC004A_OPERATOR_WEIGHT_DECAY
REC004C_OPERATOR_GRAD_CLIP: Final = ibc.REC004A_OPERATOR_GRAD_CLIP
REC004C_SCHEDULER_ETA_MIN: Final = ibc.REC004A_SCHEDULER_ETA_MIN
REC004C_EXAMPLES_PER_STEP: Final = ibc.REC004A_EXAMPLES_PER_STEP
REC004C_TRAIN_FIT_STEP_COUNT: Final = ibc.REC004A_TRAIN_FIT_STEP_COUNT
REC004C_TRAIN_SPLIT_LABEL: Final = ibc.REC004A_TRAIN_SPLIT_LABEL  # "train"

REC004C_PARENT_BUNDLE_ID: Final = ibc.REC004A_PARENT_BUNDLE_ID
REC004C_PARENT_MANIFEST_PATH: Final = ibc.REC004A_PARENT_MANIFEST_PATH

# Existing checkpoints this task reads read-only for the position/padding
# diagnostic (Stage B) and the historical replay (Stage A2).
REC004A_MIRROR_CHECKPOINT_DIR: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/checkpoints"
)
REC004A_MIRROR_STEP6000: Final = REC004A_MIRROR_CHECKPOINT_DIR / "MIRROR_HALVES_step6000.pt"
REC004A_SELECTED_RECIPE_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/selected_recipe.json"
)
REC004A_LEARNING_CURVE_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/learning_curve.jsonl"
)
REC004B_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004b/run_001")
REC004B_A_STEP6000: Final = REC004B_RUN_DIR / "A_FIXED_TMAX_1000" / "checkpoints" / "step6000.pt"
REC004B_B_STEP6000: Final = REC004B_RUN_DIR / "B_SINGLE_DECAY_6000" / "checkpoints" / "step6000.pt"
REC004B_LEARNING_CURVE_PATH: Final = REC004B_RUN_DIR / "learning_curve.jsonl"
REC004B_PAIRED_COMPARISON_PATH: Final = REC004B_RUN_DIR / "paired_comparison.json"

# Existing checkpoints characterized in Stage B -- (label, path, reported EM).
_EXISTING_CHECKPOINTS: Final[tuple[tuple[str, Path, float], ...]] = (
    ("REC004A_step6000", REC004A_MIRROR_STEP6000, 0.6875),
    ("REC004B_A_step6000", REC004B_A_STEP6000, 0.4716796875),
    ("REC004B_B_step6000", REC004B_B_STEP6000, 0.43359375),
)

# Frozen diagnostic-suite namespaces -- distinct from train/schedule_validation
# splits, hashed and locked in Stage B before any checkpoint sees them.
REC004C_LENGTH_BALANCED_SPLIT: Final = "rec004c_length_balanced_diagnostic"
REC004C_LENGTH_BALANCED_PER_LENGTH: Final = 256
REC004C_POSITION_IDENTIFIABLE_SPLIT: Final = "rec004c_position_identifiable_diagnostic"
REC004C_POSITION_IDENTIFIABLE_PER_LENGTH: Final = 64
REC004C_COUNTERFACTUAL_SPLIT: Final = "rec004c_content_counterfactual_diagnostic"
REC004C_COUNTERFACTUAL_BASES_PER_LENGTH: Final = 16
REC004C_PADDING_BATCH_SAMPLE_PER_LENGTH: Final = 16

REC004C_NAMESPACE_ROOT: Final = RECOVERY_NAMESPACE_ROOT


def _rec004c_namespace(seed: int) -> Path:
    return Path(REC004C_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004c"


@dataclass(frozen=True)
class MirrorPositionInitializationDiagnosticConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004c")
    seed: int = RECOVERY_PILOT_SEED
    schedule_validation_examples: int = REC004C_SCHEDULE_VALIDATION_EXAMPLES
    checkpoint_interval: int = REC004C_CHECKPOINT_INTERVAL
    max_updates_per_init: int = REC004C_MAX_UPDATES_PER_INIT
    descriptive_floor: float = REC004C_DESCRIPTIVE_FLOOR


# =============================================================================
# Position map -- derived directly from `MirrorHalvesOp.apply`
# (src/apc/environments/operations.py:551-555: mid = len(seq)//2;
# reversed(seq[:mid]) + reversed(seq[mid:])), never assumed from the
# operation's name. A pure, content-independent output<-input position
# permutation, self-inverse within each half.
# =============================================================================


def mirror_halves_position_map(n: int) -> tuple[int, ...]:
    """`pi_n(i)`: output position `i` reads input position `pi_n(i)`."""
    mid = n // 2
    return tuple((mid - 1 - i) if i < mid else (n + mid - 1 - i) for i in range(n))


def _verify_position_map_against_source(
    lengths: Sequence[int], *, trials_per_length: int = 64
) -> dict[str, Any]:
    """Confirms `mirror_halves_position_map` reproduces the real oracle
    interpreter's output for random legal sequences at every checked length
    -- not merely re-derived algebraically from the `apply` source read."""
    rng = random.Random(_local_seed(0, "position_map_source_verification"))
    per_length: dict[str, Any] = {}
    all_match = True
    for n in lengths:
        pi = mirror_halves_position_map(n)
        is_involution = all(pi[pi[i]] == i for i in range(n))
        n_checked = 0
        n_matched = 0
        for _ in range(trials_per_length):
            seq = tuple(rng.randrange(REC004C_VOCAB_SIZE) for _ in range(n))
            prog = Program(steps=(ProgramStep(operation=REC004C_TARGET_OPERATION, params={}),))
            out = run_program(prog, seq, REC004C_VOCAB_SIZE).output_tokens
            expected = tuple(seq[pi[i]] for i in range(n))
            n_checked += 1
            n_matched += int(out == expected)
        matched = n_matched == n_checked
        all_match = all_match and matched
        per_length[str(n)] = {
            "pi_n": list(pi),
            "is_involution": is_involution,
            "trials_checked": n_checked,
            "trials_matched": n_matched,
            "matches": matched,
        }
    return {
        "position_map_definition": (
            "pi_n(i) = mid-1-i for i<mid, else n+mid-1-i, where mid = n//2 (floor); "
            "y[i] = x[pi_n(i)] for all i in [0, n)"
        ),
        "source": "src/apc/environments/operations.py:533-555 (MirrorHalvesOp.apply)",
        "content_independent": True,
        "per_length": per_length,
        "all_lengths_matched": all_match,
        "applicability": "POSITION_MAP_APPLICABLE"
        if all_match
        else "POSITION_MAP_NOT_APPLICABLE_OR_UNRESOLVED",
    }


def _local_seed(step: int, label: str) -> int:
    """Same construction as `unified_oracle_causal_benchmark._derive_local_seed`,
    reused directly rather than reimplemented with a different formula."""
    from apc.evaluation.unified_oracle_causal_benchmark import _derive_local_seed

    return _derive_local_seed(RECOVERY_PILOT_SEED, step, label)


# =============================================================================
# Stage A -- source manifest, historical replay, operation-contract audit.
# No training, no new checkpoint of any kind.
# =============================================================================


def _check_recipe_matches_live_defaults() -> None:
    from apc.evaluation.unified_oracle_causal_benchmark import UnifiedBenchmarkConfig

    live = UnifiedBenchmarkConfig()
    if live.operator_lr != REC004C_OPERATOR_LR:
        raise ValueError(f"RECIPE_MISMATCH: operator_lr drifted from {REC004C_OPERATOR_LR}")
    if live.operator_weight_decay != REC004C_OPERATOR_WEIGHT_DECAY:
        raise ValueError("RECIPE_MISMATCH: operator_weight_decay drifted")
    if live.operator_grad_clip != REC004C_OPERATOR_GRAD_CLIP:
        raise ValueError("RECIPE_MISMATCH: operator_grad_clip drifted")


def _build_source_manifest() -> dict[str, Any]:
    """A1: existence + hash of every artifact this task depends on. Missing
    files raise `SOURCE_ARTIFACT_UNAVAILABLE` (new training must not
    proceed), UNLESS only REC-004A's own MIRROR_HALVES checkpoint is
    missing -- that specific historical comparison alone degrades to
    UNAVAILABLE (per section A1's explicit carve-out)."""
    required_hard: dict[str, Path] = {
        "parent_manifest": REC004C_PARENT_MANIFEST_PATH,
        "rec004a_budget_protocol": Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/budget_protocol.json"
        ),
        "rec004a_operation_contract_audit": Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/operation_contract_audit.json"
        ),
        "rec004a_selected_recipe": REC004A_SELECTED_RECIPE_PATH,
        "rec004b_source_audit": REC004B_RUN_DIR / "source_audit.json",
        "rec004b_protocol": REC004B_RUN_DIR / "protocol.json",
        "rec004b_config": REC004B_RUN_DIR / "config.yaml",
        "rec004b_data_manifest": REC004B_RUN_DIR / "data_manifest.json",
        "rec004b_initial_state": REC004B_RUN_DIR / "initial_state.pt",
        "rec004b_lr_trace": REC004B_RUN_DIR / "lr_trace.jsonl",
        "rec004b_learning_curve": REC004B_LEARNING_CURVE_PATH,
        "rec004b_error_breakdown": REC004B_RUN_DIR / "error_breakdown.json",
        "rec004b_fixed_candidate_manifest": REC004B_RUN_DIR / "fixed_candidate_manifest.json",
        "rec004b_a_step6000": REC004B_A_STEP6000,
        "rec004b_b_step6000": REC004B_B_STEP6000,
        "rec004b_paired_comparison": REC004B_PAIRED_COMPARISON_PATH,
    }
    missing_hard = [str(p) for p in required_hard.values() if not p.is_file()]
    if missing_hard:
        raise mb.MissingArtifactError(f"SOURCE_ARTIFACT_UNAVAILABLE: {missing_hard}")

    hashes: dict[str, str] = {}
    for name, path in required_hard.items():
        if path.suffix == ".pt":
            hashes[name] = mb.raw_file_sha256(path)
        else:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    rec004a_mirror_checkpoint_available = REC004A_MIRROR_STEP6000.is_file()
    if rec004a_mirror_checkpoint_available:
        hashes["rec004a_mirror_step6000"] = mb.raw_file_sha256(REC004A_MIRROR_STEP6000)

    return {
        "task_id": REC004C_TASK_ID,
        "required_artifact_paths": {k: str(v.resolve()) for k, v in required_hard.items()},
        "required_artifact_hashes": hashes,
        "rec004a_mirror_checkpoint_available": rec004a_mirror_checkpoint_available,
        "rec004a_mirror_step6000_path": str(REC004A_MIRROR_STEP6000.resolve()),
        "note": (
            "REC-004A's own MIRROR_HALVES step=6000 checkpoint is available in this "
            "repository state; its historical comparison is SOURCE_REPLAY_VERIFIED/"
            "MISMATCH below, not UNAVAILABLE."
            if rec004a_mirror_checkpoint_available
            else "REC-004A's own MIRROR_HALVES step=6000 checkpoint is missing; that one "
            "historical comparison is UNAVAILABLE, independent of REC-004B's own "
            "artifacts (all present)."
        ),
    }


def _historical_replay(core: Any, eval_bank: Any, op_to_id: dict[str, int]) -> dict[str, Any]:
    """A2: reload each existing saved checkpoint and reproduce its reported
    step=6000 validation EM against the SAME fixed validation split those
    tasks used -- not re-derived, not trusted from the report alone."""
    pid = REC004C_TARGET_PHYSICAL_ID
    original = {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}
    replay: dict[str, Any] = {}
    for label, path, reported_em in _EXISTING_CHECKPOINTS:
        if not path.is_file():
            replay[label] = {"status": "DESCRIPTIVE_ONLY", "reason": f"{path} not found"}
            continue
        sd = dict(mb.load_state_dict(path))
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in sd.items()}, strict=True
        )
        eval_bank.get(pid).eval()
        row = _evaluate_one_operation(
            core,
            eval_bank,
            op_to_id,
            REC004C_TARGET_OPERATION,
            seed=RECOVERY_PILOT_SEED,
            n_examples=REC004C_SCHEDULE_VALIDATION_EXAMPLES,
            split=REC004C_SCHEDULE_VALIDATION_SPLIT,
        )
        matches = abs(row["correct_exact_match"] - reported_em) < 1e-9
        replay[label] = {
            "checkpoint_path": str(path.resolve()),
            "checkpoint_state_hash": mb.canonical_state_hash(sd),
            "reproduced_correct_exact_match": row["correct_exact_match"],
            "reported_correct_exact_match": reported_em,
            "status": "SOURCE_REPLAY_VERIFIED" if matches else "SOURCE_REPLAY_MISMATCH",
            "full_row": row,
        }
    eval_bank.get(pid).load_state_dict(original, strict=True)
    eval_bank.get(pid).eval()

    any_mismatch = any(v.get("status") == "SOURCE_REPLAY_MISMATCH" for v in replay.values())

    # A2 continued: cross-run comparability table (REC-004A original vs
    # REC-004B's A condition, both nominally T_max=1000 mechanical
    # extension). Only what is independently confirmable from the real
    # artifacts is marked known; everything else is UNKNOWN.
    comparability = {
        "rec004a_original_vs_rec004b_a": {
            "training_data_stream_formula": {
                "value": "_derive_local_seed(seed, step, 'train:MIRROR_HALVES')",
                "status": "CONFIRMED_IDENTICAL",
                "evidence": "same formula read directly in both modules (ibc/msc share it)",
            },
            "lr_schedule_family": {
                "value": "CosineAnnealingLR(T_max=1000), mechanical extension",
                "status": "CONFIRMED_IDENTICAL",
            },
            "weight_initialization_seed_label": {
                "rec004a": "rec004a_init:MIRROR_HALVES",
                "rec004b_a": "rec004b_shared_init:MIRROR_HALVES",
                "status": "CONFIRMED_DIFFERENT",
                "evidence": (
                    "distinct seed labels by construction -- ADR-0097's own disclosed design choice"
                ),
            },
            "optimizer_dropout_batch_order_precision": {
                "status": "UNKNOWN",
                "reason": (
                    "neither run's artifacts record enough process-level state "
                    "(cuDNN determinism flags, thread count, etc.) to independently "
                    "confirm bitwise-identical non-weight training conditions"
                ),
            },
            "conclusion": (
                "Only the weight-initialization draw is a CONFIRMED difference between "
                "these two runs; other axes are UNKNOWN, not CONFIRMED_IDENTICAL. This "
                "task does not attribute REC-004A vs REC-004B's gap to initialization "
                "alone on this basis -- ADR-0097 already disclosed the same limitation."
            ),
        }
    }

    return {
        "task_id": REC004C_TASK_ID,
        "existing_checkpoints_replayed": replay,
        "any_mismatch": any_mismatch,
        "historical_comparability": comparability,
    }


def _check_mirror_halves_operation_contract() -> dict[str, Any]:
    """A3: reuses REC-004B's own source-verified fixture check, then adds
    the explicit position-map derivation/verification this task's own
    diagnostics depend on."""
    base = msc._check_mirror_halves_contract()
    position_map = _verify_position_map_against_source(REC004C_LEGAL_LENGTHS)
    op_obj = get_operation(REC004C_TARGET_OPERATION)
    return {
        **base,
        "min_input_length_live": op_obj.min_input_length,
        "output_length_equals_input_length_live_check": all(
            op_obj.output_length(n) == n for n in range(2, 20)
        ),
        "position_map": position_map,
    }


def run_stage_a(config: MirrorPositionInitializationDiagnosticConfig) -> dict[str, Any]:
    _guard_not_frozen("run_stage_a")

    source_manifest = _build_source_manifest()
    _check_recipe_matches_live_defaults()

    parent_manifest, _raw = ibc._load_parent_manifest()
    diagnostic_load = mb.load_bundle(
        parent_manifest, mode="diagnostic", expected_primitive_count=16
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

    physical_ids_by_op = {p.operation_name: p.physical_id for p in parent_manifest.primitives}
    for op, expected_pid in REC004C_PHYSICAL_IDS.items():
        if physical_ids_by_op.get(op) != expected_pid:
            raise mb.IncompleteBundleError(
                f"parent manifest declares physical_id={physical_ids_by_op.get(op)} for {op}, "
                f"expected pre-registered {expected_pid}"
            )
    non_target_ops = tuple(
        sorted(op for op in physical_ids_by_op if op != REC004C_TARGET_OPERATION)
    )
    if len(non_target_ops) != 15:
        raise mb.IncompleteBundleError(
            "expected exactly 15 non-target operations in a 16-op bundle, "
            f"got {len(non_target_ops)}"
        )

    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )

    historical_replay = _historical_replay(core, eval_bank, op_to_id)
    if historical_replay["any_mismatch"]:
        raise ValueError(
            "SOURCE_REPLAY_MISMATCH: a confirmed-identical historical checkpoint did not "
            f"reproduce its own reported EM -- {historical_replay['existing_checkpoints_replayed']}"
        )
    operation_contract = _check_mirror_halves_operation_contract()
    if not operation_contract["position_map"]["all_lengths_matched"]:
        raise ValueError(
            "POSITION_MAP contract fixture mismatch -- refusing to proceed with an "
            "unverified transform rule for MIRROR_HALVES"
        )
    if not (
        operation_contract["even_length_fixture"]["matches"]
        and operation_contract["odd_length_fixture"]["matches"]
    ):
        raise ValueError("MIRROR_HALVES base contract fixture mismatch")

    freeze_scope_manifest = {
        "task_id": REC004C_TASK_ID,
        "target_operation": REC004C_TARGET_OPERATION,
        "target_physical_id": REC004C_TARGET_PHYSICAL_ID,
        "non_target_operations": list(non_target_ops),
        "non_target_operation_count": len(non_target_ops),
        "legacy_12_protected_non_shift_plus_shift": list(ibc.REC004A_PROTECTED_OPERATIONS),
        "legacy_protected_count": len(ibc.REC004A_PROTECTED_OPERATIONS),
        "note": (
            "This task's own freeze rule (section 2.3): 15 non-target operations are "
            "read-only, not '12 protected + 3 fixed-candidate imports' -- no child bundle "
            "is built and no candidate is imported. The legacy 12+SHIFT set is cross-"
            "checked above for physical-id continuity only, not used as this task's own "
            "protected-set definition."
        ),
        "no_child_bundle_built_by_this_task": True,
    }

    source_audit = {
        "task_id": REC004C_TASK_ID,
        "parent_task_id": REC004C_PARENT_TASK_ID,
        "parent_manifest_path": str(REC004C_PARENT_MANIFEST_PATH.resolve()),
        "parent_bundle_id": parent_manifest.bundle_id,
        "parent_core_canonical_state_hash": parent_manifest.core.canonical_state_hash,
        "parent_self_load_diagnostic_checks_performed": list(diagnostic_load.checks_performed),
        "parent_unearned_capability_request_rejected": capability_rejected,
        "target_operation_physical_id": REC004C_TARGET_PHYSICAL_ID,
        "all_16_physical_ids": dict(sorted(physical_ids_by_op.items(), key=lambda kv: kv[1])),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("source_manifest", source_manifest),
        ("historical_comparability", historical_replay["historical_comparability"]),
        ("historical_replay", historical_replay),
        ("operation_contract_audit", {REC004C_TARGET_OPERATION: operation_contract}),
        ("freeze_scope_manifest", freeze_scope_manifest),
    ):
        (config.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    (config.output_dir / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, default=str), encoding="utf-8"
    )

    return {
        "parent_manifest": parent_manifest,
        "core": core,
        "eval_bank": eval_bank,
        "op_to_id": op_to_id,
        "non_target_operations": non_target_ops,
        "source_manifest": source_manifest,
        "historical_replay": historical_replay,
        "operation_contract": operation_contract,
        "freeze_scope_manifest": freeze_scope_manifest,
        "source_audit": source_audit,
    }


# =============================================================================
# Stage B -- position/padding diagnostics on the 3 existing saved
# checkpoints, and construction of the 3 frozen diagnostic-suite fixtures
# reused unchanged in Stage D. No new training of any kind.
# =============================================================================


def _make_example(seq: tuple[int, ...], split: str) -> Example:
    prog = Program(steps=(ProgramStep(operation=REC004C_TARGET_OPERATION, params={}),))
    res = run_program(prog, seq, REC004C_VOCAB_SIZE)
    return Example(
        input_tokens=seq,
        target_tokens=res.output_tokens,
        program=prog,
        operation_graph=res.graph,
        category="known",
        split=split,
        vocab_size=REC004C_VOCAB_SIZE,
        task_spec=TaskSpec.from_program(prog),
        oracle_metadata=OracleMetadata(label="K", primitive_operations=(REC004C_TARGET_OPERATION,)),
    )


def _generate_schedule_validation_examples() -> list[Example]:
    from apc.evaluation.unified_oracle_causal_benchmark import _generate_parameter_free_examples

    return _generate_parameter_free_examples(
        RECOVERY_PILOT_SEED,
        REC004C_SCHEDULE_VALIDATION_EXAMPLES,
        operation=REC004C_TARGET_OPERATION,
        split=REC004C_SCHEDULE_VALIDATION_SPLIT,
        vocab_size=REC004C_VOCAB_SIZE,
        sequence_length_range=REC004C_SEQUENCE_LENGTH_RANGE,
    )


def _late_learning_curve_audit() -> dict[str, Any]:
    """B1: re-reads (never re-derives or interpolates) the two existing
    checkpointed curves' 4000-6000 range."""
    rec004a_curve: dict[str, float] | None = None
    if REC004A_SELECTED_RECIPE_PATH.is_file():
        sel = json.loads(REC004A_SELECTED_RECIPE_PATH.read_text(encoding="utf-8"))
        rec004a_curve = sel["selection"][REC004C_TARGET_OPERATION]["all_checkpoint_validation_em"]

    rec004b_curve: dict[str, list[dict[str, Any]]] = {
        "A_FIXED_TMAX_1000": [],
        "B_SINGLE_DECAY_6000": [],
    }
    if REC004B_LEARNING_CURVE_PATH.is_file():
        for line in REC004B_LEARNING_CURVE_PATH.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["step"] >= 4000 and "schedule_validation" in row:
                rec004b_curve[row["condition_id"]].append(
                    {
                        "step": row["step"],
                        "validation_em": row["schedule_validation"]["correct_exact_match"],
                        "validation_token_acc": row["schedule_validation"][
                            "correct_token_accuracy"
                        ],
                        "train_fit_em": row["train_fit"]["sequence_exact_match"],
                        "lr_used": row.get("lr_used"),
                        "lr_after_scheduler": row.get("lr_after_scheduler"),
                    }
                )

    rec004a_late = (
        {k: v for k, v in rec004a_curve.items() if int(k) >= 4000} if rec004a_curve else None
    )
    return {
        "task_id": REC004C_TASK_ID,
        "rec004a_coarse_ladder_4000_to_6000": rec004a_late or "LATE_CURVE_UNAVAILABLE",
        "rec004b_fine_500_step_curve_4000_to_6000": rec004b_curve,
        "note": (
            "REC-004A recorded only 1000/2000/4000/6000 (no 500-step granularity); "
            "REC-004B recorded every 500 steps. Both read verbatim from their own saved "
            "logs -- neither reconstructed nor interpolated."
        ),
        "reobserved_points_within_task_limit": {
            "rec004a_points_used": len(rec004a_late) if rec004a_late else 0,
            "rec004b_points_used": sum(len(v) for v in rec004b_curve.values()),
            "limit_note": (
                "task doc section 4 B1: at most 13x2 (S1) + 4 (S2) previously-recorded points"
            ),
        },
    }


def _generate_length_balanced_diagnostic() -> list[Example]:
    examples: list[Example] = []
    for n in REC004C_LEGAL_LENGTHS:
        rng = random.Random(_local_seed(0, f"{REC004C_LENGTH_BALANCED_SPLIT}:len{n}"))
        for _ in range(REC004C_LENGTH_BALANCED_PER_LENGTH):
            seq = tuple(rng.randrange(REC004C_VOCAB_SIZE) for _ in range(n))
            examples.append(_make_example(seq, REC004C_LENGTH_BALANCED_SPLIT))
    return examples


def _generate_position_identifiable_diagnostic() -> dict[int, list[Example] | None]:
    """Only applicable where the legal vocabulary can make every position's
    value unique within an example (`n <= vocab_size`) -- true for every
    length in this task's 6-10 range against vocab_size=10, but checked
    explicitly rather than assumed."""
    result: dict[int, list[Example] | None] = {}
    for n in REC004C_LEGAL_LENGTHS:
        if n > REC004C_VOCAB_SIZE:
            result[n] = None
            continue
        rng = random.Random(_local_seed(0, f"{REC004C_POSITION_IDENTIFIABLE_SPLIT}:len{n}"))
        examples = []
        for _ in range(REC004C_POSITION_IDENTIFIABLE_PER_LENGTH):
            seq = tuple(rng.sample(range(REC004C_VOCAB_SIZE), n))
            examples.append(_make_example(seq, REC004C_POSITION_IDENTIFIABLE_SPLIT))
        result[n] = examples
    return result


def _generate_content_counterfactual_diagnostic() -> dict[int, list[dict[str, Any]]]:
    """Each valid input position of each base example is substituted with
    exactly one different legal token, one position at a time; the
    teacher's own (oracle) changed output positions are recorded alongside
    the position map's structural prediction, purely as a sanity check --
    this function never touches a model."""
    result: dict[int, list[dict[str, Any]]] = {}
    for n in REC004C_LEGAL_LENGTHS:
        rng = random.Random(_local_seed(0, f"{REC004C_COUNTERFACTUAL_SPLIT}:len{n}"))
        pi = mirror_halves_position_map(n)
        bases = []
        for _ in range(REC004C_COUNTERFACTUAL_BASES_PER_LENGTH):
            seq = tuple(rng.randrange(REC004C_VOCAB_SIZE) for _ in range(n))
            base_example = _make_example(seq, REC004C_COUNTERFACTUAL_SPLIT)
            variants = []
            for j in range(n):
                choices = [v for v in range(REC004C_VOCAB_SIZE) if v != seq[j]]
                new_val = rng.choice(choices)
                new_seq = seq[:j] + (new_val,) + seq[j + 1 :]
                variant_example = _make_example(new_seq, REC004C_COUNTERFACTUAL_SPLIT)
                expected_changed_output_positions = [i for i in range(n) if pi[i] == j]
                teacher_actual_changed = [
                    i
                    for i in range(n)
                    if base_example.target_tokens[i] != variant_example.target_tokens[i]
                ]
                variants.append(
                    {
                        "changed_input_position": j,
                        "original_value": seq[j],
                        "new_value": new_val,
                        "example": variant_example,
                        "expected_changed_output_positions": expected_changed_output_positions,
                        "teacher_actual_changed_output_positions": teacher_actual_changed,
                        "teacher_matches_position_map": (
                            teacher_actual_changed == expected_changed_output_positions
                        ),
                    }
                )
            bases.append({"base_example": base_example, "variants": variants})
        result[n] = bases
    return result


def _generate_padding_batch_sample() -> list[Example]:
    examples: list[Example] = []
    for n in REC004C_LEGAL_LENGTHS:
        rng = random.Random(_local_seed(0, f"rec004c_padding_batch_sample:len{n}"))
        for _ in range(REC004C_PADDING_BATCH_SAMPLE_PER_LENGTH):
            seq = tuple(rng.randrange(REC004C_VOCAB_SIZE) for _ in range(n))
            examples.append(_make_example(seq, "rec004c_padding_batch_sample"))
    return examples


def _digest_example_set(examples: Sequence[Example]) -> str:
    payload = sorted([[list(e.input_tokens), list(e.target_tokens)] for e in examples])
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def _diagnostic_data_manifest(
    length_balanced: list[Example],
    position_identifiable_by_length: dict[int, list[Example] | None],
    counterfactual_by_length: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    pos_id_flat = [ex for lst in position_identifiable_by_length.values() if lst for ex in lst]
    cf_flat: list[Example] = []
    for bases in counterfactual_by_length.values():
        for b in bases:
            cf_flat.append(b["base_example"])
            cf_flat.extend(v["example"] for v in b["variants"])

    train_digests: set[tuple[Any, Any]] = set()
    for step in range(1, REC004C_TRAIN_FIT_STEP_COUNT + 1):
        for e in ibc._generate_step_training_examples(
            RECOVERY_PILOT_SEED,
            step,
            REC004C_TARGET_OPERATION,
            vocab_size=REC004C_VOCAB_SIZE,
            sequence_length_range=REC004C_SEQUENCE_LENGTH_RANGE,
        ):
            train_digests.add((e.input_tokens, e.target_tokens))
    schedule_validation_digests = {
        (e.input_tokens, e.target_tokens) for e in _generate_schedule_validation_examples()
    }

    def _overlap(examples: Sequence[Example]) -> dict[str, Any]:
        digests = {(e.input_tokens, e.target_tokens) for e in examples}
        return {
            "n": len(examples),
            "vs_train_stream_overlap": len(digests & train_digests),
            "vs_schedule_validation_overlap": len(digests & schedule_validation_digests),
        }

    return {
        "task_id": REC004C_TASK_ID,
        "length_balanced_diagnostic": {
            "split": REC004C_LENGTH_BALANCED_SPLIT,
            "per_length_n": REC004C_LENGTH_BALANCED_PER_LENGTH,
            "digest": _digest_example_set(length_balanced),
            **_overlap(length_balanced),
        },
        "position_identifiable_diagnostic": {
            "split": REC004C_POSITION_IDENTIFIABLE_SPLIT,
            "per_length_n": REC004C_POSITION_IDENTIFIABLE_PER_LENGTH,
            "applicable_lengths": sorted(
                n for n, v in position_identifiable_by_length.items() if v is not None
            ),
            "not_applicable_lengths": sorted(
                n for n, v in position_identifiable_by_length.items() if v is None
            ),
            "digest": _digest_example_set(pos_id_flat),
            **_overlap(pos_id_flat),
        },
        "content_counterfactual_diagnostic": {
            "split": REC004C_COUNTERFACTUAL_SPLIT,
            "bases_per_length": REC004C_COUNTERFACTUAL_BASES_PER_LENGTH,
            "total_examples_incl_variants": len(cf_flat),
            "digest": _digest_example_set(cf_flat),
            **_overlap(cf_flat),
        },
    }


def _position_level_error_analysis(
    core: Any, primitive: Any, examples: list[Example]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """B2: returns `(position_error_summary, position_confusion)`."""
    preds = _predict_tokens(core, primitive, examples, REC004C_TARGET_OPERATION, None)

    by_position: dict[str, dict[str, int]] = {}
    first_error_positions: list[int | None] = []
    error_counts: list[int] = []
    structural = {"moved": {"n": 0, "correct": 0}, "fixed": {"n": 0, "correct": 0}}
    apparent = {"changed": {"n": 0, "correct": 0}, "unchanged": {"n": 0, "correct": 0}}
    j_hat_class = {"unique": 0, "ambiguous": 0, "predicted_token_not_in_input": 0}
    unique_outcomes = {"exact": 0, "off_by_one": 0, "other": 0, "total": 0}
    sequence_em = 0

    for ex, pred in zip(examples, preds, strict=True):
        n = len(ex.input_tokens)
        target = ex.target_tokens
        x = ex.input_tokens
        pi = mirror_halves_position_map(n)
        first_err: int | None = None
        n_err = 0
        for i in range(n):
            key = f"{n}:{i}"
            bucket = by_position.setdefault(key, {"n": 0, "correct": 0})
            bucket["n"] += 1
            is_correct = i < len(pred) and pred[i] == target[i]
            bucket["correct"] += int(is_correct)
            if not is_correct:
                n_err += 1
                if first_err is None:
                    first_err = i

            moved_key = "moved" if pi[i] != i else "fixed"
            structural[moved_key]["n"] += 1
            structural[moved_key]["correct"] += int(is_correct)

            apparent_key = "changed" if x[i] != target[i] else "unchanged"
            apparent[apparent_key]["n"] += 1
            apparent[apparent_key]["correct"] += int(is_correct)

            if i < len(pred):
                yhat = pred[i]
                j_hat = [j for j in range(n) if x[j] == yhat]
                if len(j_hat) == 1:
                    j_hat_class["unique"] += 1
                    unique_outcomes["total"] += 1
                    if j_hat[0] == pi[i]:
                        unique_outcomes["exact"] += 1
                    elif abs(j_hat[0] - pi[i]) == 1:
                        unique_outcomes["off_by_one"] += 1
                    else:
                        unique_outcomes["other"] += 1
                elif len(j_hat) == 0:
                    j_hat_class["predicted_token_not_in_input"] += 1
                else:
                    j_hat_class["ambiguous"] += 1
        first_error_positions.append(first_err)
        error_counts.append(n_err)
        sequence_em += int(n_err == 0)

    summary = {
        "task_id": REC004C_TASK_ID,
        "n_examples": len(examples),
        "sequence_exact_match": sequence_em / len(examples) if examples else None,
        "by_length_position": by_position,
        "first_error_position_distribution": {
            "null_count": sum(1 for p in first_error_positions if p is None),
            "by_position": {
                str(p): first_error_positions.count(p)
                for p in sorted({p for p in first_error_positions if p is not None})
            },
        },
        "errors_per_sequence_distribution": {
            str(k): error_counts.count(k) for k in sorted(set(error_counts))
        },
    }
    confusion = {
        "task_id": REC004C_TASK_ID,
        "structural_moved_vs_fixed": structural,
        "apparent_changed_vs_unchanged": apparent,
        "j_hat_source_token_classification": j_hat_class,
        "unique_j_hat_position_outcomes": unique_outcomes,
        "caveat": (
            "J_hat(i) is derived from output VALUE equality with input tokens -- an "
            "output-consistent-position analysis, not a proof of internal attended position."
        ),
    }
    return summary, confusion


def _padding_batch_metamorphic_check(
    core: Any, primitive: Any, examples: list[Example]
) -> dict[str, Any]:
    """B4: single vs. batched-with-fillers, and short- vs. long-filler
    padding width, execution equivalence -- both routes stay inside the
    public `collate_content_only_batch`/`_predict_tokens` API, so nothing
    here can construct an `UNSUPPORTED_METAMORPHIC_CASE` (mixed-length
    batching and variable right-padding are both natively supported)."""
    filler_rng = random.Random(_local_seed(0, "padding_batch_filler"))
    short_fillers = [
        _make_example(
            tuple(
                filler_rng.randrange(REC004C_VOCAB_SIZE)
                for _ in range(REC004C_SEQUENCE_LENGTH_RANGE[0])
            ),
            "rec004c_padding_filler_short",
        )
        for _ in range(3)
    ]
    long_fillers = [
        _make_example(
            tuple(
                filler_rng.randrange(REC004C_VOCAB_SIZE)
                for _ in range(REC004C_SEQUENCE_LENGTH_RANGE[1])
            ),
            "rec004c_padding_filler_long",
        )
        for _ in range(3)
    ]

    results = []
    for ex in examples:
        single_pred = _predict_tokens(core, primitive, [ex], REC004C_TARGET_OPERATION, None)[0]
        short_batch_pred = _predict_tokens(
            core, primitive, [ex, *short_fillers], REC004C_TARGET_OPERATION, None
        )[0]
        long_batch_pred = _predict_tokens(
            core, primitive, [ex, *long_fillers], REC004C_TARGET_OPERATION, None
        )[0]
        results.append(
            {
                "n": len(ex.input_tokens),
                "single_vs_short_batch_match": single_pred == short_batch_pred,
                "single_vs_long_batch_match": single_pred == long_batch_pred,
                "short_batch_vs_long_batch_match": short_batch_pred == long_batch_pred,
            }
        )

    all_match = all(
        r["single_vs_short_batch_match"]
        and r["single_vs_long_batch_match"]
        and r["short_batch_vs_long_batch_match"]
        for r in results
    )
    return {
        "task_id": REC004C_TASK_ID,
        "n_examples_checked": len(results),
        "conditions": [
            "single_vs_batched_with_short_fillers",
            "single_vs_batched_with_long_fillers",
            "short_padding_width_vs_long_padding_width_same_content",
        ],
        "unsupported_cases": [],
        "results": results,
        "all_match": all_match,
        "status": "PASS" if all_match else "EXECUTION_CONTRACT_FAILURE",
    }


def _attention_observation(core: Any, primitive: Any, examples: list[Example]) -> dict[str, Any]:
    """B5: best-effort, evaluation-only attention read via a parallel
    `need_weights=True` call through the SAME submodule/weights as the
    primitive's own `forward` -- never modifies `forward` itself. Confirms
    the hook's own prediction matches the normal (`need_weights=False`)
    forward before trusting any attention-position correlation."""
    try:
        device = core.device
        primitive.eval()
        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004C_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
        batch_input = collate_content_only_batch(examples, core.tokens, device=device)
        with torch.no_grad():
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
            batch, lmax, _ = h.shape
            out_max = max(output_lengths)
            content_position_ids = (
                torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
            )
            kv = primitive.content_in_proj(h) + primitive.content_position_embedding(
                content_position_ids
            )
            content_lengths_t = torch.tensor(content_lengths, device=device).unsqueeze(1)
            content_pad_mask = content_position_ids >= content_lengths_t
            query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
            query_slots = primitive.answer_query_embedding(query_ids)
            arg_token = h.new_zeros(batch, 1, primitive.d_operator)
            query = query_slots + arg_token
            attn_out, attn_weights = primitive.cross_attn(
                query,
                kv,
                kv,
                key_padding_mask=content_pad_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            hidden = primitive.attn_norm(query + attn_out)
            hidden = primitive.ffn_norm(hidden + primitive.ffn(hidden))
            logits_hooked = primitive.readout(hidden)
            preds_hooked_t = logits_hooked.argmax(dim=-1)
        preds_hooked = [preds_hooked_t[row, :n].tolist() for row, n in enumerate(output_lengths)]
        preds_normal = _predict_tokens(core, primitive, examples, REC004C_TARGET_OPERATION, None)
        hook_matches = preds_hooked == preds_normal

        n_checked = 0
        n_argmax_matches_pi = 0
        for row, ex in enumerate(examples):
            n = len(ex.input_tokens)
            pi = mirror_halves_position_map(n)
            for i in range(n):
                argmax_pos = int(attn_weights[row, i, :n].argmax().item())
                n_checked += 1
                n_argmax_matches_pi += int(argmax_pos == pi[i])

        return {
            "status": "OBSERVED",
            "hook_predictions_match_normal_forward": hook_matches,
            "n_positions_checked": n_checked,
            "n_attention_argmax_matches_position_map": n_argmax_matches_pi,
            "attention_argmax_match_rate": (n_argmax_matches_pi / n_checked) if n_checked else None,
            "caveat": (
                "attention-argmax agreement with the position map is descriptive only -- "
                "not a causal proof of internal attended position (task doc section 4 B5)"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - best-effort observation, never blocks the run
        return {"status": "INTERNAL_ATTENTION_NOT_OBSERVABLE", "reason": repr(exc)}


def _counterfactual_dependency_check(
    core: Any, primitive: Any, counterfactual_by_length: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    per_length: dict[str, Any] = {}
    total_variants = 0
    total_matches = 0
    for n, bases in counterfactual_by_length.items():
        n_variants = 0
        n_match = 0
        for base in bases:
            combined = [base["base_example"], *[v["example"] for v in base["variants"]]]
            preds = _predict_tokens(core, primitive, combined, REC004C_TARGET_OPERATION, None)
            base_pred = preds[0]
            for v, var_pred in zip(base["variants"], preds[1:], strict=True):
                model_changed = [
                    i
                    for i in range(n)
                    if i < len(base_pred) and i < len(var_pred) and base_pred[i] != var_pred[i]
                ]
                n_variants += 1
                n_match += int(model_changed == v["expected_changed_output_positions"])
        per_length[str(n)] = {
            "n_variants": n_variants,
            "n_matches_expected_single_position_change": n_match,
        }
        total_variants += n_variants
        total_matches += n_match
    return {
        "task_id": REC004C_TASK_ID,
        "per_length": per_length,
        "total_variants": total_variants,
        "total_matches_expected_single_position_change": total_matches,
        "match_rate": (total_matches / total_variants) if total_variants else None,
        "caveat": (
            "measures whether the MODEL's own output change matches the position map's "
            "structural prediction -- a functional-dependency measurement, not internal "
            "attention causal attribution (task doc section 4 B3)"
        ),
    }


def run_stage_b(
    config: MirrorPositionInitializationDiagnosticConfig, stage_a: dict[str, Any]
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_b")
    core, eval_bank = stage_a["core"], stage_a["eval_bank"]
    pid = REC004C_TARGET_PHYSICAL_ID
    original = {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}

    late_learning_curve_audit = _late_learning_curve_audit()

    length_balanced_examples = _generate_length_balanced_diagnostic()
    position_identifiable_by_length = _generate_position_identifiable_diagnostic()
    counterfactual_by_length = _generate_content_counterfactual_diagnostic()
    diagnostic_data_manifest = _diagnostic_data_manifest(
        length_balanced_examples, position_identifiable_by_length, counterfactual_by_length
    )

    schedule_validation_examples = _generate_schedule_validation_examples()
    padding_batch_sample = _generate_padding_batch_sample()
    attention_sample = [
        ex
        for n in REC004C_LEGAL_LENGTHS
        for ex in (position_identifiable_by_length.get(n) or [])[:8]
    ]

    position_error_summary: dict[str, Any] = {}
    position_confusion: dict[str, Any] = {}
    padding_batch_audit: dict[str, Any] = {}
    counterfactual_dependency: dict[str, Any] = {}
    attention_observation: dict[str, Any] = {}

    for label, path, _reported in _EXISTING_CHECKPOINTS:
        if not path.is_file():
            for d in (
                position_error_summary,
                position_confusion,
                padding_batch_audit,
                counterfactual_dependency,
                attention_observation,
            ):
                d[label] = {"status": "UNAVAILABLE"}
            continue
        sd = dict(mb.load_state_dict(path))
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in sd.items()}, strict=True
        )
        eval_bank.get(pid).eval()
        primitive = eval_bank.get(pid)

        summary, confusion = _position_level_error_analysis(
            core, primitive, schedule_validation_examples
        )
        position_error_summary[label] = summary
        position_confusion[label] = confusion
        padding_batch_audit[label] = _padding_batch_metamorphic_check(
            core, primitive, padding_batch_sample
        )
        counterfactual_dependency[label] = _counterfactual_dependency_check(
            core, primitive, counterfactual_by_length
        )
        attention_observation[label] = _attention_observation(core, primitive, attention_sample)

    eval_bank.get(pid).load_state_dict(original, strict=True)
    eval_bank.get(pid).eval()

    any_execution_contract_failure = any(
        v.get("status") == "EXECUTION_CONTRACT_FAILURE" for v in padding_batch_audit.values()
    )
    execution_contract_status = "FAILURE" if any_execution_contract_failure else "PASS"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("late_learning_curve_audit", late_learning_curve_audit),
        ("diagnostic_data_manifest", diagnostic_data_manifest),
        ("position_error_summary", position_error_summary),
        ("position_confusion", position_confusion),
        ("padding_batch_audit", padding_batch_audit),
        ("counterfactual_dependency", counterfactual_dependency),
    ):
        (config.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    return {
        "late_learning_curve_audit": late_learning_curve_audit,
        "diagnostic_data_manifest": diagnostic_data_manifest,
        "position_error_summary": position_error_summary,
        "position_confusion": position_confusion,
        "padding_batch_audit": padding_batch_audit,
        "counterfactual_dependency": counterfactual_dependency,
        "attention_observation": attention_observation,
        "execution_contract_status": execution_contract_status,
        "length_balanced_examples": length_balanced_examples,
        "position_identifiable_by_length": position_identifiable_by_length,
        "counterfactual_by_length": counterfactual_by_length,
        "schedule_validation_examples": schedule_validation_examples,
    }


# =============================================================================
# Stage C -- fix the 5-initialization protocol before any new training.
# =============================================================================


def _new_primitive(core: Any) -> CrossPositionPrimitive:
    return CrossPositionPrimitive(
        REC004C_TARGET_PHYSICAL_ID,
        CrossPositionPrimitiveConfig(
            operation=REC004C_TARGET_OPERATION,
            d_model=core.model.config.d_model,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=REC004C_VOCAB_SIZE,
            max_sequence_length=32,
        ),
        status=PrimitiveStatus.STABLE,
    )


def build_five_initial_states(core: Any, seed: int) -> dict[str, dict[str, torch.Tensor]]:
    """C2: 5 initial states from a namespace distinct from both REC-004A's
    (`rec004a_init:MIRROR_HALVES`) and REC-004B's (`rec004b_shared_init:
    MIRROR_HALVES`) own init labels. `init_id` never feeds the data-seed or
    Core-seed derivation anywhere in this module."""
    del seed  # Core/data seed stays RECOVERY_PILOT_SEED; init_id never derives from it.
    states: dict[str, dict[str, torch.Tensor]] = {}
    for init_id in REC004C_INIT_IDS:
        init_seed = _local_seed(0, f"rec004c_init:{REC004C_TARGET_OPERATION}:{init_id}")
        torch.manual_seed(init_seed)
        primitive = _new_primitive(core)
        states[init_id] = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    # Restore a FIXED, shared training-RNG state (never init_id-derived) before
    # any training starts, so torch-RNG consumption from the 5 init draws
    # never leaks into training. This primitive has no dropout/stochastic
    # forward path, so the restoration is a defensive no-op in practice, but
    # is performed and recorded per the task doc's explicit C2 instruction.
    torch.manual_seed(_local_seed(0, "rec004c_shared_training_rng"))
    return states


def build_initialization_protocol(
    config: MirrorPositionInitializationDiagnosticConfig, stage_a: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_steps = list(range(0, config.max_updates_per_init + 1, config.checkpoint_interval))
    if checkpoint_steps[-1] != REC004C_DECISIVE_STEP:
        raise ValueError("protocol requires the decisive step to fall on a checkpoint boundary")
    check = msc._verify_lr_trace(REC004C_T_MAX, checkpoint_steps)
    if not check["contract_ok"]:
        raise ValueError(f"LR_TRACE_CONTRACT_FAILURE: {check['mismatches']}")

    protocol = {
        "task_id": REC004C_TASK_ID,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "parent_core_canonical_state_hash": stage_a["parent_manifest"].core.canonical_state_hash,
        "model_seed": config.seed,
        "target_operation": REC004C_TARGET_OPERATION,
        "target_operation_physical_id": REC004C_TARGET_PHYSICAL_ID,
        "init_ids": list(REC004C_INIT_IDS),
        "init_seed_label_formula": f"rec004c_init:{REC004C_TARGET_OPERATION}:<init_id>",
        "shared_training_rng_label": "rec004c_shared_training_rng",
        "schedule": {
            "scheduler": "CosineAnnealingLR",
            "t_max": REC004C_T_MAX,
            "note": (
                "REC-004B's A_FIXED_TMAX_1000 mechanical extension (T_max stays 1000, "
                "the same scheduler object is stepped past its nominal length to 6000) -- "
                "the ONLY schedule this task uses. No T_max=6000 or any other condition."
            ),
        },
        "operator_lr": REC004C_OPERATOR_LR,
        "operator_weight_decay": REC004C_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004C_OPERATOR_GRAD_CLIP,
        "scheduler_eta_min": REC004C_SCHEDULER_ETA_MIN,
        "lr_definition": (
            "eta_T(u) = eta_min + (base_lr - eta_min)/2 * (1 + cos(pi*u/T)); u is the count of "
            "completed optimizer.step() calls; scheduler.step() is called once after each "
            "optimizer.step(); lr_used for update u is eta_T(u-1), lr_after_scheduler is eta_T(u)."
        ),
        "lr_trace_preflight_verified": True,
        "lr_table": check["trace"],
        "max_updates_per_init": config.max_updates_per_init,
        "max_updates_total": config.max_updates_per_init * len(REC004C_INIT_IDS),
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_steps": checkpoint_steps,
        "decisive_checkpoint": REC004C_DECISIVE_STEP,
        "examples_per_step": REC004C_EXAMPLES_PER_STEP,
        "data_roles": {
            "train": {
                "purpose": "online per-step training stream, IDENTICAL across all 5 inits",
                "seed_formula": (
                    f"_derive_local_seed(seed, step, 'train:{REC004C_TARGET_OPERATION}')"
                ),
                "init_id_included": False,
                "used_for_selection": False,
            },
            "train_fit": {
                "purpose": "diagnostic only: fixed already-trained subset",
                "n_examples": REC004C_TRAIN_FIT_STEP_COUNT * REC004C_EXAMPLES_PER_STEP,
                "used_for_selection": False,
            },
            "schedule_validation": {
                "split": REC004C_SCHEDULE_VALIDATION_SPLIT,
                "n_examples": config.schedule_validation_examples,
                "reused_from": (
                    "REC-004A/REC-004B's own fixed 1024 examples (hash-verified in Stage A)"
                ),
                "used_for_selection": False,
                "note": (
                    "recorded per checkpoint for descriptive/diagnostic purposes only -- "
                    "this task never selects on it"
                ),
            },
            "length_balanced_diagnostic": {
                "split": REC004C_LENGTH_BALANCED_SPLIT,
                "n_per_length": REC004C_LENGTH_BALANCED_PER_LENGTH,
                "used_for_selection": False,
            },
            "position_identifiable_diagnostic": {
                "split": REC004C_POSITION_IDENTIFIABLE_SPLIT,
                "n_per_length": REC004C_POSITION_IDENTIFIABLE_PER_LENGTH,
                "used_for_selection": False,
            },
            "content_counterfactual_diagnostic": {
                "split": REC004C_COUNTERFACTUAL_SPLIT,
                "bases_per_length": REC004C_COUNTERFACTUAL_BASES_PER_LENGTH,
                "used_for_selection": False,
            },
            "rg3_final_query_or_sealed_data": {
                "generated": False,
                "consumed": False,
                "note": "not generated and not consumed by this task, per section 1.4",
            },
        },
        "selection_rule": (
            "NONE. This task never selects, publishes, or re-gates on any measured EM -- "
            "selected_init/child_bundle/rg3_recheck are fixed to null/null/NOT_EXECUTED "
            "regardless of outcome, per the task doc's own D4/section-0 boundary."
        ),
        "non_target_operation_count": 15,
        "no_child_bundle_built": True,
        "forbidden_changes": [
            "Core retraining",
            "operator architecture/size change",
            "position bias addition",
            "oracle gather substitution",
            "LR bounds/optimizer/loss/batch/precision change",
            "warmup, curriculum, or data reweighting",
            "additional seed search beyond I01-I05",
            "fine-tuning or resuming from any prior trained MIRROR_HALVES weights",
            "router recalibration",
            "argument scorer recalibration",
            "extending beyond 6000 updates per init / 30000 total",
            "selecting/publishing/re-gating on any measured result",
        ],
        "descriptive_floor": config.descriptive_floor,
        "descriptive_floor_note": (
            "0.95 is recorded per-init as a descriptive reference count only (task doc "
            "section 6 D4) -- reaching it does not trigger selection, child assembly, or "
            "RG3 recheck in this task."
        ),
    }
    protocol_for_hash = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol_for_hash, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    (config.output_dir / "initialization_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    return protocol


# =============================================================================
# Stage D -- train MIRROR_HALVES 5 times (one per init), 6000 updates each.
# =============================================================================


def _digest_examples(examples: list[Example]) -> str:
    payload = [[list(e.input_tokens), list(e.target_tokens)] for e in examples]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def _run_diagnostic_suites_at_final_step(
    core: Any,
    primitive: Any,
    length_balanced_examples: list[Example],
    position_identifiable_by_length: dict[int, list[Example] | None],
    counterfactual_by_length: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    from apc.evaluation.unified_oracle_causal_benchmark import _evaluate_primitive_arm

    lb_exact, lb_tok = _evaluate_primitive_arm(
        core, primitive, length_balanced_examples, REC004C_TARGET_OPERATION, argument_provider=None
    )
    pos_id_results: dict[str, Any] = {}
    for n, examples in position_identifiable_by_length.items():
        if not examples:
            pos_id_results[str(n)] = "NOT_APPLICABLE"
            continue
        exact, tok = _evaluate_primitive_arm(
            core, primitive, examples, REC004C_TARGET_OPERATION, argument_provider=None
        )
        pos_id_results[str(n)] = {
            "n": len(examples),
            "sequence_exact_match": exact,
            "token_accuracy": tok,
        }
    counterfactual = _counterfactual_dependency_check(core, primitive, counterfactual_by_length)
    return {
        "length_balanced_diagnostic": {
            "n": len(length_balanced_examples),
            "sequence_exact_match": lb_exact,
            "token_accuracy": lb_tok,
        },
        "position_identifiable_diagnostic": pos_id_results,
        "content_counterfactual_diagnostic": counterfactual,
    }


def run_one_init(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    initial_state: dict[str, torch.Tensor],
    config: MirrorPositionInitializationDiagnosticConfig,
    output_dir: Path,
    length_balanced_examples: list[Example],
    position_identifiable_by_length: dict[int, list[Example] | None],
    counterfactual_by_length: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    _guard_not_frozen(f"run_one_init:{init_id}")
    pid = REC004C_TARGET_PHYSICAL_ID
    device = core.device
    seed = config.seed

    primitive = _new_primitive(core)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in initial_state.items()}, strict=True)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004C_OPERATOR_LR, weight_decay=REC004C_OPERATOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004C_T_MAX, eta_min=REC004C_SCHEDULER_ETA_MIN
    )

    init_dir = output_dir / init_id
    ckpt_dir = init_dir / "checkpoints"
    state_dir = init_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints: list[dict[str, Any]] = []
    data_digests: list[str] = []
    cumulative_examples = 0
    t_start = time.time()
    diverged_at: int | None = None

    def _snapshot_and_eval(
        step: int, lr_used: float | None, lr_after: float, *, final: bool
    ) -> None:
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
                "init_id": init_id,
            },
            state_dir / f"step{step}.pt",
        )

        live = eval_bank.get(pid)
        live.load_state_dict({k: v.to(device) for k, v in state_snapshot.items()}, strict=True)
        live.eval()
        schedule_validation = _evaluate_one_operation(
            core,
            eval_bank,
            op_to_id,
            REC004C_TARGET_OPERATION,
            seed=seed,
            n_examples=config.schedule_validation_examples,
            split=REC004C_SCHEDULE_VALIDATION_SPLIT,
        )
        train_fit = ibc._evaluate_train_fit(
            core,
            primitive,
            REC004C_TARGET_OPERATION,
            seed=seed,
            vocab_size=REC004C_VOCAB_SIZE,
            sequence_length_range=REC004C_SEQUENCE_LENGTH_RANGE,
        )
        length_strata = ibc._length_stratified_breakdown(
            core,
            primitive,
            REC004C_TARGET_OPERATION,
            seed=seed,
            n_examples=config.schedule_validation_examples,
            split=REC004C_SCHEDULE_VALIDATION_SPLIT,
        )
        primitive.train()

        row: dict[str, Any] = {
            "step": step,
            "init_id": init_id,
            "cumulative_training_examples": cumulative_examples,
            "lr_used": lr_used,
            "lr_after_scheduler": lr_after,
            "schedule_validation": schedule_validation,
            "train_fit": train_fit,
            "length_stratified": length_strata,
            "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
            "cumulative_wall_clock_seconds": time.time() - t_start,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        if final:
            row["diagnostic_suites"] = _run_diagnostic_suites_at_final_step(
                core,
                primitive,
                length_balanced_examples,
                position_identifiable_by_length,
                counterfactual_by_length,
            )
        checkpoints.append(row)

    initial_lr = optimizer.param_groups[0]["lr"]
    _snapshot_and_eval(0, lr_used=None, lr_after=initial_lr, final=False)

    for step in range(1, config.max_updates_per_init + 1):
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004C_TARGET_OPERATION,
            vocab_size=REC004C_VOCAB_SIZE,
            sequence_length_range=REC004C_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(examples)
        data_digests.append(_digest_examples(examples))

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004C_TARGET_OPERATION).output_length(n) for n in content_lengths
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004C_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())

        if not math.isfinite(running_loss):
            diverged_at = step
            break

        if step % config.checkpoint_interval == 0:
            _snapshot_and_eval(step, lr_used, lr_after, final=(step == REC004C_DECISIVE_STEP))

    for step in range(0, config.max_updates_per_init + 1, config.checkpoint_interval):
        if (
            diverged_at is not None
            and step >= diverged_at
            and not any(c["step"] == step for c in checkpoints)
        ):
            checkpoints.append(
                {
                    "step": step,
                    "init_id": init_id,
                    "status": "NOT_EXECUTED",
                    "reason": f"loss diverged at step {diverged_at}",
                }
            )

    checkpoints.sort(key=lambda c: c["step"])
    primitive.eval()
    final_state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    return {
        "init_id": init_id,
        "checkpoints": checkpoints,
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


def run_stage_d(
    config: MirrorPositionInitializationDiagnosticConfig,
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_d")
    core, eval_bank, op_to_id = stage_a["core"], stage_a["eval_bank"], stage_a["op_to_id"]
    pid = REC004C_TARGET_PHYSICAL_ID
    original = {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}

    initial_states = build_five_initial_states(core, config.seed)
    init_hashes = {k: mb.canonical_state_hash(v) for k, v in initial_states.items()}
    if len(set(init_hashes.values())) != len(REC004C_INIT_IDS):
        raise ValueError(f"5 initial states are not all distinct: {init_hashes}")

    init_state_dir = config.output_dir / "initial_states"
    init_state_dir.mkdir(parents=True, exist_ok=True)
    for init_id, sd in initial_states.items():
        torch.save(sd, init_state_dir / f"{init_id}.pt")

    outcomes: dict[str, dict[str, Any]] = {}
    for init_id in REC004C_INIT_IDS:
        eval_bank.get(pid).load_state_dict(original, strict=True)
        outcomes[init_id] = run_one_init(
            core,
            eval_bank,
            op_to_id,
            init_id,
            initial_states[init_id],
            config,
            config.output_dir,
            stage_b["length_balanced_examples"],
            stage_b["position_identifiable_by_length"],
            stage_b["counterfactual_by_length"],
        )
    eval_bank.get(pid).load_state_dict(original, strict=True)
    eval_bank.get(pid).eval()

    data_stream_digests = {
        init_id: outcomes[init_id]["data_digests"] for init_id in REC004C_INIT_IDS
    }
    reference = next(iter(data_stream_digests.values()))
    data_stream_identical_across_inits = all(d == reference for d in data_stream_digests.values())
    if not data_stream_identical_across_inits:
        raise ValueError("PAIRING_CONTRACT_FAILURE: training data streams diverged across inits")

    with (config.output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for _init_id, outcome in outcomes.items():
            for ckpt in outcome["checkpoints"]:
                row = {
                    k: v
                    for k, v in ckpt.items()
                    if k not in ("length_stratified", "diagnostic_suites")
                }
                fh.write(json.dumps(row, default=str) + "\n")

    return {
        "outcomes": outcomes,
        "initial_states": initial_states,
        "init_hashes": init_hashes,
        "data_stream_identical_across_inits": data_stream_identical_across_inits,
    }


# =============================================================================
# Stage E -- cross-init aggregation, freeze/side-effect audits, final
# diagnosis. selected_init/child_bundle/rg3_recheck are fixed regardless of
# any number below.
# =============================================================================


def build_initialization_summary(
    outcomes: dict[str, dict[str, Any]], descriptive_floor: float
) -> dict[str, Any]:
    completed_ids: list[str] = []
    diverged_ids: list[str] = []
    missing_ids: list[str] = []
    per_init: dict[str, Any] = {}
    final_ems: list[float] = []

    for init_id in REC004C_INIT_IDS:
        outcome = outcomes[init_id]
        em = _em_at_step(outcome, REC004C_DECISIVE_STEP)
        diverged = outcome["diverged_at_step"]
        step0 = next((c for c in outcome["checkpoints"] if c["step"] == 0), None)
        final_ckpt = next(
            (c for c in outcome["checkpoints"] if c["step"] == REC004C_DECISIVE_STEP), None
        )
        if diverged is not None:
            diverged_ids.append(init_id)
            status = "DIVERGED"
        elif em is None:
            missing_ids.append(init_id)
            status = "MISSING"
        else:
            completed_ids.append(init_id)
            final_ems.append(em)
            status = "COMPLETE"
        per_init[init_id] = {
            "status": status,
            "diverged_at_step": diverged,
            "initial_state_hash": step0.get("checkpoint_state_hash") if step0 else None,
            "final_state_hash": (final_ckpt or {}).get("checkpoint_state_hash"),
            "final_validation_em": em,
            "final_validation_token_accuracy": (
                (final_ckpt or {}).get("schedule_validation", {}).get("correct_token_accuracy")
                if final_ckpt
                else None
            ),
            "final_train_fit_em": (
                (final_ckpt or {}).get("train_fit", {}).get("sequence_exact_match")
                if final_ckpt
                else None
            ),
            "final_per_length_em": (
                (final_ckpt or {}).get("length_stratified", {}).get("by_length")
                if final_ckpt
                else None
            ),
            "final_diagnostic_suites": (final_ckpt or {}).get("diagnostic_suites"),
            "reaches_descriptive_floor": (em >= descriptive_floor) if em is not None else None,
        }

    n_completed = len(completed_ids)
    mean = sum(final_ems) / n_completed if n_completed else None
    if n_completed >= 2:
        assert mean is not None
        variance = sum((x - mean) ** 2 for x in final_ems) / (n_completed - 1)
        sample_sd = math.sqrt(variance)
    else:
        sample_sd = None
    n_reach_floor = sum(1 for em in final_ems if em >= descriptive_floor)

    return {
        "task_id": REC004C_TASK_ID,
        "planned": len(REC004C_INIT_IDS),
        "completed": n_completed,
        "diverged": len(diverged_ids),
        "missing": len(missing_ids),
        "completed_init_ids": completed_ids,
        "diverged_init_ids": diverged_ids,
        "missing_init_ids": missing_ids,
        "per_init": per_init,
        "final_validation_em_summary": {
            "n_completed": n_completed,
            "mean": mean,
            "sample_sd": sample_sd,
            "sample_sd_note": "null if fewer than 2 completed trials",
            "min": min(final_ems) if final_ems else None,
            "max": max(final_ems) if final_ems else None,
            "range": (max(final_ems) - min(final_ems)) if final_ems else None,
        },
        "n_reaching_descriptive_floor_of_completed": n_reach_floor,
        "n_reaching_descriptive_floor_of_planned_5": n_reach_floor,
        "descriptive_floor": descriptive_floor,
        "note": (
            "0.95 is counted here as a descriptive reference value only -- no selection, "
            "child assembly, or RG3 recheck follows from this count. All 5 trials are "
            "repeated draws on the SAME Core/data-stream/recipe, not 5 independent Cores "
            "or unseen task families; this sample size and shared-input constraint apply "
            "to every statistic in this file."
        ),
    }


def build_cross_init_error_overlap(
    core: Any,
    eval_bank: Any,
    outcomes: dict[str, dict[str, Any]],
    schedule_validation_examples: list[Example],
) -> dict[str, Any]:
    pid = REC004C_TARGET_PHYSICAL_ID
    original = {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}
    completed = [
        i
        for i in REC004C_INIT_IDS
        if outcomes[i]["diverged_at_step"] is None
        and _em_at_step(outcomes[i], REC004C_DECISIVE_STEP) is not None
    ]
    missing = [i for i in REC004C_INIT_IDS if i not in completed]

    preds_by_init: dict[str, list[list[int]]] = {}
    for init_id in completed:
        sd = outcomes[init_id]["final_primitive_state_dict"]
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in sd.items()}, strict=True
        )
        eval_bank.get(pid).eval()
        preds_by_init[init_id] = _predict_tokens(
            core, eval_bank.get(pid), schedule_validation_examples, REC004C_TARGET_OPERATION, None
        )
    eval_bank.get(pid).load_state_dict(original, strict=True)
    eval_bank.get(pid).eval()

    correct_by_init = {
        init_id: [
            tuple(p) == tuple(e.target_tokens)
            for p, e in zip(preds, schedule_validation_examples, strict=True)
        ]
        for init_id, preds in preds_by_init.items()
    }

    n_examples = len(schedule_validation_examples)
    per_example_wrong_count = [0] * n_examples
    for init_id in completed:
        for idx, is_correct in enumerate(correct_by_init[init_id]):
            per_example_wrong_count[idx] += int(not is_correct)
    wrong_count_distribution = {
        str(k): per_example_wrong_count.count(k) for k in range(0, len(completed) + 1)
    }

    pairwise: dict[str, Any] = {}
    for a, b in itertools.combinations(completed, 2):
        ca, cb = correct_by_init[a], correct_by_init[b]
        both_correct = sum(1 for x, y in zip(ca, cb, strict=True) if x and y)
        both_wrong = sum(1 for x, y in zip(ca, cb, strict=True) if not x and not y)
        a_only = sum(1 for x, y in zip(ca, cb, strict=True) if x and not y)
        b_only = sum(1 for x, y in zip(ca, cb, strict=True) if not x and y)
        wrong_a = {idx for idx, c in enumerate(ca) if not c}
        wrong_b = {idx for idx, c in enumerate(cb) if not c}
        union = wrong_a | wrong_b
        inter = wrong_a & wrong_b
        pairwise[f"{a}_vs_{b}"] = {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            f"{a}_only_correct": a_only,
            f"{b}_only_correct": b_only,
            "disagreement_rate": (a_only + b_only) / n_examples if n_examples else None,
            "wrong_set_jaccard": (len(inter) / len(union)) if union else None,
        }

    return {
        "task_id": REC004C_TASK_ID,
        "completed_init_ids": completed,
        "missing_init_ids": missing,
        "n_examples": n_examples,
        "per_example_wrong_count_distribution_0_to_n_completed": wrong_count_distribution,
        "pairwise": pairwise,
    }


def build_freeze_audit(
    stage_a: dict[str, Any],
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    non_target_hashes_before: dict[str, str],
    target_slot_hash_before: str,
) -> dict[str, Any]:
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    non_target_hashes_after = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in stage_a["non_target_operations"]
    }
    target_slot_hash_after = mb.canonical_state_hash(
        eval_bank.get(REC004C_TARGET_PHYSICAL_ID).state_dict()
    )
    return {
        "task_id": REC004C_TASK_ID,
        "core_canonical_state_hash_before": stage_a["parent_manifest"].core.canonical_state_hash,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == stage_a["parent_manifest"].core.canonical_state_hash,
        "non_target_operations_hashes_before": non_target_hashes_before,
        "non_target_operations_hashes_after": non_target_hashes_after,
        "non_target_operations_unchanged": non_target_hashes_before == non_target_hashes_after,
        "non_target_operation_count": len(stage_a["non_target_operations"]),
        "target_slot_restored_to_parent_original": {
            "before": target_slot_hash_before,
            "after": target_slot_hash_after,
            "unchanged": target_slot_hash_before == target_slot_hash_after,
            "note": (
                "the PARENT bank's own MIRROR_HALVES slot -- transiently overwritten many "
                "times during Stage B/D evaluation, always restored afterward; this task "
                "never leaves any new weights resident in the parent bank"
            ),
        },
    }


def build_final_diagnosis(
    *,
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
    stage_d: dict[str, Any] | None,
    initialization_summary: dict[str, Any] | None,
    cross_init: dict[str, Any] | None,
    freeze_audit: dict[str, Any] | None,
    execution_contract_status: str,
) -> dict[str, Any]:
    position_map_ok = stage_a["operation_contract"]["position_map"]["all_lengths_matched"]
    historical_all_verified = all(
        v.get("status") == "SOURCE_REPLAY_VERIFIED"
        for v in stage_a["historical_replay"]["existing_checkpoints_replayed"].values()
        if v.get("status") != "DESCRIPTIVE_ONLY"
    )
    source_audit_status = "VERIFIED" if historical_all_verified else "PARTIAL_HISTORY"
    if not stage_a["source_manifest"]["rec004a_mirror_checkpoint_available"]:
        source_audit_status = "PARTIAL_HISTORY"

    e1: dict[str, Any] = {
        "q1_mirror_halves_semantics": stage_a["operation_contract"]["position_map"],
        "q2_historical_replay": stage_a["historical_replay"],
        "q3_padding_batch_mask": {
            "execution_contract_status": execution_contract_status,
            "per_checkpoint": stage_b["padding_batch_audit"],
        },
        "q4_length_by_position_errors": {
            "per_checkpoint_files": "position_error_summary.json / position_confusion.json",
        },
        "q5_counterfactual_and_attention": {
            "counterfactual_per_checkpoint": stage_b["counterfactual_dependency"],
            "attention_observation_per_checkpoint": stage_b["attention_observation"],
        },
    }

    if stage_d is not None and initialization_summary is not None:
        e1["q6_init_only_difference_confirmed"] = {
            "data_stream_identical_across_inits": stage_d["data_stream_identical_across_inits"],
            "init_hashes_all_distinct": len(set(stage_d["init_hashes"].values()))
            == len(REC004C_INIT_IDS),
            "init_hashes": stage_d["init_hashes"],
        }
        e1["q7_cross_init_variance"] = initialization_summary["final_validation_em_summary"]
        e1["q7_cross_init_error_overlap"] = cross_init
        e1["q8_late_curve_trend"] = {
            init_id: [
                {
                    "step": c["step"],
                    "validation_em": c.get("schedule_validation", {}).get("correct_exact_match"),
                }
                for c in stage_d["outcomes"][init_id]["checkpoints"]
                if c.get("step", -1) >= 4000 and "schedule_validation" in c
            ]
            for init_id in REC004C_INIT_IDS
        }
        completed = initialization_summary["completed_init_ids"]
        n_completed = len(completed)
        n_floor = initialization_summary["n_reaching_descriptive_floor_of_completed"]
        rng = initialization_summary["final_validation_em_summary"]["range"]
        if n_completed == 0:
            init_signal = "NO_COMPLETED_TRIALS"
        elif rng is not None and rng >= 0.30:
            init_signal = "HIGH_CROSS_INIT_VARIANCE"
        elif n_floor == n_completed and n_completed > 0:
            init_signal = "ALL_COMPLETED_REACH_FLOOR"
        elif n_floor == 0:
            init_signal = "NONE_REACH_FLOOR"
        else:
            init_signal = "MIXED"
        e1["q9_supported_vs_unresolved"] = {
            "execution_contract": execution_contract_status,
            "position_map_applicable": position_map_ok,
            "cross_init_signal": init_signal,
        }
        if init_signal == "HIGH_CROSS_INIT_VARIANCE":
            next_mechanism = (
                "Initialization-sensitivity is the clearest signal (large spread in "
                f"final EM across {n_completed} completed trials, range={rng}). Next: a "
                "dedicated initialization-sensitivity study (weight-scale/variance at "
                "init vs. final EM correlation across a larger init sample), not a "
                "position-bias or capacity change."
            )
        elif init_signal in ("NONE_REACH_FLOOR",):
            next_mechanism = (
                "All completed trials stay below the descriptive floor under this fixed "
                "recipe/schedule. Next: examine whether the shared position-error pattern "
                "(position_confusion.json) is consistent across inits -- a common failure "
                "locus would motivate a position-scoring diagnostic before any capacity "
                "change, per this task's own STOP rule against jumping to capacity claims."
            )
        elif init_signal == "ALL_COMPLETED_REACH_FLOOR":
            next_mechanism = (
                "All completed trials individually reach the descriptive floor under this "
                "fixed recipe. Next: this is a single-Core/single-recipe finding (task doc "
                "D4) -- an explicit next task would need to test stability across seeds "
                "11-14 before any REC-005 continuation is considered, not this task."
            )
        else:
            next_mechanism = (
                "Mixed outcome across the 5 inits with moderate spread. Next: the position-"
                "level confusion pattern (position_confusion.json) shared across completed "
                "trials is the most direct next diagnostic -- see cross_init_error_overlap "
                "for which examples are consistently missed by ALL completed trials versus "
                "only some."
            )
        e1["q10_next_single_mechanism"] = next_mechanism
    else:
        e1["q6_init_only_difference_confirmed"] = "NOT_EXECUTED"
        e1["q7_cross_init_variance"] = "NOT_EXECUTED"
        e1["q8_late_curve_trend"] = "NOT_EXECUTED"
        e1["q9_supported_vs_unresolved"] = {
            "execution_contract": execution_contract_status,
            "position_map_applicable": position_map_ok,
            "cross_init_signal": "NOT_EXECUTED",
        }
        e1["q10_next_single_mechanism"] = (
            "UNRESOLVED -- Stage D did not run (see execution_contract_status / "
            "source_audit_status); resolving the Stage A/B STOP condition is the "
            "prerequisite next step, not a mechanism claim about MIRROR_HALVES itself."
        )

    return {
        "task_id": REC004C_TASK_ID,
        "e1_answers": e1,
        "source_audit_status": source_audit_status,
        "execution_contract_status": execution_contract_status,
        "freeze_audit": freeze_audit,
    }


def _config_to_yaml_dict(config: MirrorPositionInitializationDiagnosticConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "schedule_validation_examples": config.schedule_validation_examples,
        "checkpoint_interval": config.checkpoint_interval,
        "max_updates_per_init": config.max_updates_per_init,
        "descriptive_floor": config.descriptive_floor,
    }


# =============================================================================
# Orchestration.
# =============================================================================


def run_mirror_position_initialization_diagnostic_task(
    config: MirrorPositionInitializationDiagnosticConfig,
) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004C's Core seed is pre-registered as {RECOVERY_PILOT_SEED}; "
            f"got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )

    stage_a = run_stage_a(config)
    core, eval_bank, op_to_id = stage_a["core"], stage_a["eval_bank"], stage_a["op_to_id"]

    non_target_hashes_before = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in stage_a["non_target_operations"]
    }
    target_slot_hash_before = mb.canonical_state_hash(
        eval_bank.get(REC004C_TARGET_PHYSICAL_ID).state_dict()
    )

    stage_b = run_stage_b(config, stage_a)
    execution_contract_status = stage_b["execution_contract_status"]

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE",
        "source_audit_status": None,
        "historical_diagnostic_status": "COMPLETE",
        "execution_contract_status": execution_contract_status,
        "initialization_experiment_status": "NOT_EXECUTED",
        "selected_init": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }

    if execution_contract_status != "PASS":
        # Section 2.3/4 B4 STOP rule: an execution-contract failure halts
        # dependent (new-training) work; Stage C/D never run.
        freeze_audit = build_freeze_audit(
            stage_a, core, eval_bank, op_to_id, non_target_hashes_before, target_slot_hash_before
        )
        final_diagnosis = build_final_diagnosis(
            stage_a=stage_a,
            stage_b=stage_b,
            stage_d=None,
            initialization_summary=None,
            cross_init=None,
            freeze_audit=freeze_audit,
            execution_contract_status=execution_contract_status,
        )
        result["source_audit_status"] = final_diagnosis["source_audit_status"]
        after_hashes = _snapshot_forbidden_cache_hashes(seed)
        side_effect_audit = {
            "task_id": REC004C_TASK_ID,
            "before": before_hashes,
            "after": after_hashes,
            "shared_cache_unchanged": before_hashes == after_hashes,
        }
        for name, payload in (
            ("freeze_audit", freeze_audit),
            ("side_effect_audit", side_effect_audit),
            ("final_diagnosis", final_diagnosis),
        ):
            (output_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        result["freeze_audit"] = freeze_audit
        result["side_effect_audit"] = side_effect_audit
        result["final_diagnosis"] = final_diagnosis
        result["wall_clock_seconds"] = time.time() - start
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result

    protocol = build_initialization_protocol(config, stage_a)
    stage_d = run_stage_d(config, stage_a, stage_b)

    initialization_summary = build_initialization_summary(
        stage_d["outcomes"], config.descriptive_floor
    )
    cross_init = build_cross_init_error_overlap(
        core, eval_bank, stage_d["outcomes"], stage_b["schedule_validation_examples"]
    )
    freeze_audit = build_freeze_audit(
        stage_a, core, eval_bank, op_to_id, non_target_hashes_before, target_slot_hash_before
    )
    final_diagnosis = build_final_diagnosis(
        stage_a=stage_a,
        stage_b=stage_b,
        stage_d=stage_d,
        initialization_summary=initialization_summary,
        cross_init=cross_init,
        freeze_audit=freeze_audit,
        execution_contract_status=execution_contract_status,
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004C_TASK_ID,
        "before": before_hashes,
        "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }

    n_completed = initialization_summary["completed"]
    n_missing = initialization_summary["missing"]
    n_diverged = initialization_summary["diverged"]
    if n_completed == len(REC004C_INIT_IDS):
        init_status = "COMPLETE"
    elif n_diverged > 0 and n_completed > 0:
        init_status = "COMPLETE_WITH_NUMERICAL_FAILURES"
    elif n_completed > 0:
        init_status = "PARTIAL"
    else:
        init_status = (
            "PARTIAL" if (n_missing + n_diverged) < len(REC004C_INIT_IDS) else "NOT_EXECUTED"
        )

    result["source_audit_status"] = final_diagnosis["source_audit_status"]
    result["initialization_experiment_status"] = init_status
    result["mechanism_diagnosis"] = final_diagnosis["e1_answers"]["q9_supported_vs_unresolved"]
    result["protocol"] = protocol
    result["initialization_summary"] = initialization_summary
    result["cross_init_error_overlap"] = cross_init
    result["freeze_audit"] = freeze_audit
    result["side_effect_audit"] = side_effect_audit
    result["final_diagnosis"] = final_diagnosis
    result["wall_clock_seconds"] = time.time() - start

    for name, payload in (
        ("initialization_summary", initialization_summary),
        ("cross_init_error_overlap", cross_init),
        ("freeze_audit", freeze_audit),
        ("side_effect_audit", side_effect_audit),
        ("final_diagnosis", final_diagnosis),
    ):
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
