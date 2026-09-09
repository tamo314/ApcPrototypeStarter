"""B-C005REC-004H: MIRROR_HALVES Late-Stage Length-10 Audit & Conditional
Extension.

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LATE_PROGRESS_CONDITIONAL_EXTENSION.md`.
Source: REC-004G's (`mirror_budget_extension.py`, ADR-0102) own saved
P/I01-I05 checkpoints at step=8000..12000 (500-step interval) in
`runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/`. REC-004G itself
found all 5 inits improve substantially from 6000->12000 (mean EM 0.719 ->
0.910, concentrated at length 10), but only I04/I05 clear the 0.95
existing-validation floor; I01/I02/I03 remain below it.

This task does two things, in order, and stops:

1. **Audit (zero new optimizer updates).** Forward-only, on REC-004G's
   already-saved checkpoint weights: hash-verifies each checkpoint against
   REC-004G's own `learning_curve.jsonl`, reproduces REC-004G's own recorded
   overall EM and by-length sequence counts, and computes NEW length-10
   token-level statistics REC-004G never recorded (valid-token count, token
   error count, mean valid-token cross-entropy, per-output-position
   breakdown) at steps 8000/8500/.../12000 for all 5 inits. `EM_PROGRESS`/
   `SOFT_PROGRESS` predicates (this task's own new operational rule, not a
   statistical test) are then evaluated at the LR-phase-matched steps
   8000/10000/12000 for whichever inits are below the 0.95 floor at 12000.

2. **Conditional extension (only if EVERY below-floor init satisfies the
   predicate).** Resumes ALL 5 P/I01-I05 from their saved COMPLETE
   step=12000 training state (weights + AdamW optimizer state + scheduler
   state + CPU/CUDA RNG state) and continues training 6000 MORE updates each
   (30000 new optimizer updates total) under the exact same recipe REC-004G
   used, to a new cumulative 18000. If the predicate is not satisfied for
   every below-floor init, ZERO new optimizer updates run and the task stops
   with a report of which predicate component failed for which init.

Per this task's own charter, `selected_init`, `selected_intervention`, and
`child_bundle` are fixed `None`/`null` and `rg3_recheck` is fixed
`"NOT_EXECUTED"` regardless of outcome -- even a unanimous floor pass at
step=18000 does not, by itself, select a candidate, build a child bundle, or
run an RG3 recheck. `B-C005REC-005` (the separate, still-blocked five-model
cohort task) is not touched or consumed by this task.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as mbe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,  # noqa: F401 -- re-exported for tests, not used on the hot path
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
    frozen_evaluation,
)
from apc.evaluation.unified_oracle_causal_benchmark import _generate_parameter_free_examples
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004H_TASK_ID",
    "REC004H_SOURCE_TASK_ID",
    "REC004H_INIT_IDS",
    "REC004H_ARM",
    "REC004H_TARGET_OPERATION",
    "REC004H_AUDIT_STEPS",
    "REC004H_PHASE_MATCH_STEPS",
    "REC004H_TARGET_LENGTH",
    "REC004H_LOSS_EPS",
    "REC004H_SOURCE_STEP_FOR_EXTENSION",
    "REC004H_ADDITIONAL_UPDATES",
    "REC004H_TARGET_CUMULATIVE_STEP",
    "MirrorLateProgressConditionalExtensionConfig",
    "run_mirror_late_progress_conditional_extension_task",
]

# =============================================================================
# Constants -- reused from REC-004G/REC-004D wherever the recipe is
# unchanged, never retyped, so this module fails to import (rather than
# silently diverging) if REC-004G's own recipe constants ever change.
# =============================================================================

REC004H_TASK_ID: Final = "B-C005REC-004H"
REC004H_SOURCE_TASK_ID: Final = "B-C005REC-004G"
REC004H_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LATE_PROGRESS_CONDITIONAL_EXTENSION.md"
)

REC004H_INIT_IDS: Final[tuple[str, ...]] = mbe.REC004G_INIT_IDS  # ("I01", ..., "I05")
REC004H_ARM: Final = mbe.REC004G_ARM  # "P_LENGTH_POSITION_BIAS"
REC004H_TARGET_OPERATION: Final = mbe.REC004G_TARGET_OPERATION  # "MIRROR_HALVES"
REC004H_TARGET_PHYSICAL_ID: Final = mbe.REC004G_TARGET_PHYSICAL_ID

REC004H_SOURCE_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004g/run_001")
REC004H_SOURCE_LEARNING_CURVE_PATH: Final = REC004H_SOURCE_RUN_DIR / "learning_curve.jsonl"

# "Stored" observation points (REC-004G already checkpointed all of these).
REC004H_AUDIT_STEPS: Final[tuple[int, ...]] = (
    8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000,
)
# LR-phase-matched decisive points used by the continuation predicate.
REC004H_PHASE_MATCH_STEPS: Final[tuple[int, ...]] = (8000, 10000, 12000)

REC004H_TARGET_LENGTH: Final = 10

REC004H_EXISTING_VALIDATION_EXAMPLES: Final = mbe.REC004G_EXISTING_VALIDATION_EXAMPLES  # 1024
REC004H_EXISTING_VALIDATION_SPLIT: Final = mbe.REC004G_EXISTING_VALIDATION_SPLIT
REC004H_EXISTING_VALIDATION_FLOOR: Final = mbe.REC004G_EXISTING_VALIDATION_FLOOR  # 0.95
REC004H_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE
REC004H_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE

REC004H_SOURCE_REPLAY_EM_TOL: Final = mbe.REC004G_SOURCE_REPLAY_EM_TOL  # 1e-9
REC004H_LOSS_EPS: Final = 1e-6  # numerical comparison margin, NOT a significance level

REC004H_SOURCE_STEP_FOR_EXTENSION: Final = 12000
REC004H_ADDITIONAL_UPDATES: Final = 6000
REC004H_TARGET_CUMULATIVE_STEP: Final = (
    REC004H_SOURCE_STEP_FOR_EXTENSION + REC004H_ADDITIONAL_UPDATES
)  # 18000
REC004H_CHECKPOINT_INTERVAL: Final = mbe.REC004G_CHECKPOINT_INTERVAL  # 500
REC004H_MAX_UPDATES_TOTAL: Final = REC004H_ADDITIONAL_UPDATES * len(REC004H_INIT_IDS)  # 30000

REC004H_OPERATOR_LR: Final = mbe.REC004G_OPERATOR_LR
REC004H_OPERATOR_WEIGHT_DECAY: Final = mbe.REC004G_OPERATOR_WEIGHT_DECAY
REC004H_OPERATOR_GRAD_CLIP: Final = mbe.REC004G_OPERATOR_GRAD_CLIP
REC004H_SCHEDULER_ETA_MIN: Final = mbe.REC004G_SCHEDULER_ETA_MIN
REC004H_T_MAX: Final = mbe.REC004G_T_MAX  # 1000, mechanically continued (never reset)

REC004H_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = mbe.REC004G_PROTECTED_OPERATIONS
REC004H_FIXED_CANDIDATE_OPERATIONS: Final[tuple[str, ...]] = mbe.REC004G_FIXED_CANDIDATE_OPERATIONS

REC004H_MAX_PREDICTION_EXAMPLES_BUDGET: Final = 51200  # 50 * 1024, per the task's own cap


# =============================================================================
# Path / IO helpers -- all read-only against REC-004G's run_001 tree.
# =============================================================================


def _rec004g_checkpoint_path(init_id: str, step: int) -> Path:
    return REC004H_SOURCE_RUN_DIR / init_id / REC004H_ARM / "checkpoints" / f"step{step}.pt"


def _rec004g_training_state_path(init_id: str, step: int) -> Path:
    return REC004H_SOURCE_RUN_DIR / init_id / REC004H_ARM / "training_states" / f"step{step}.pt"


def _load_rec004g_learning_curve_rows() -> list[dict[str, Any]]:
    if not REC004H_SOURCE_LEARNING_CURVE_PATH.is_file():
        raise mb.MissingArtifactError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {REC004H_SOURCE_LEARNING_CURVE_PATH} not found"
        )
    rows = []
    for line in REC004H_SOURCE_LEARNING_CURVE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _rec004g_learning_curve_row(
    rows: list[dict[str, Any]], init_id: str, step: int
) -> dict[str, Any] | None:
    return next(
        (
            r for r in rows
            if r["init_id"] == init_id and r["arm"] == REC004H_ARM and r["step"] == step
        ),
        None,
    )


def _existing_validation_examples(seed: int) -> list[Any]:
    return _generate_parameter_free_examples(
        seed,
        REC004H_EXISTING_VALIDATION_EXAMPLES,
        operation=REC004H_TARGET_OPERATION,
        split=REC004H_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004H_VOCAB_SIZE,
        sequence_length_range=REC004H_SEQUENCE_LENGTH_RANGE,
    )


def _rec004g_source_file_hashes() -> dict[str, str]:
    """Raw-byte hashes of every REC-004G `run_001` file this task reads --
    proof this task never mutates its read-only source."""
    hashes = {"learning_curve.jsonl": mb.raw_file_sha256(REC004H_SOURCE_LEARNING_CURVE_PATH)}
    for init_id in REC004H_INIT_IDS:
        for step in REC004H_AUDIT_STEPS:
            path = _rec004g_checkpoint_path(init_id, step)
            if path.is_file():
                hashes[f"{init_id}/checkpoints/step{step}.pt"] = mb.raw_file_sha256(path)
        ext_path = _rec004g_training_state_path(init_id, REC004H_SOURCE_STEP_FOR_EXTENSION)
        if ext_path.is_file():
            key = f"{init_id}/training_states/step{REC004H_SOURCE_STEP_FOR_EXTENSION}.pt"
            hashes[key] = mb.raw_file_sha256(ext_path)
    return hashes


@dataclass(frozen=True)
class MirrorLateProgressConditionalExtensionConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004h")
    seed: int = RECOVERY_PILOT_SEED


def _config_to_yaml_dict(config: MirrorLateProgressConditionalExtensionConfig) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


# =============================================================================
# Stage A -- cheap hash-only precheck (NO forward pass) across every audit
# checkpoint, so `late_progress_protocol.json` can be locked before any B1
# additional forward runs, per the task's own sequencing requirement.
# =============================================================================


def _hash_only_precheck(init_id: str, step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = _rec004g_checkpoint_path(init_id, step)
    if not path.is_file():
        return {
            "init_id": init_id, "step": step,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE", "path": str(path),
        }
    state = torch.load(path, map_location="cpu", weights_only=False)
    computed_hash = mb.canonical_state_hash(state)
    row = _rec004g_learning_curve_row(rows, init_id, step)
    if row is None:
        return {
            "init_id": init_id, "step": step,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "reason": "no matching learning_curve.jsonl row",
        }
    recorded_hash = row["checkpoint_state_hash"]
    matches = computed_hash == recorded_hash
    return {
        "init_id": init_id, "step": step,
        "status": "VERIFIED" if matches else "SOURCE_REPLAY_MISMATCH",
        "computed_checkpoint_state_hash": computed_hash,
        "recorded_checkpoint_state_hash": recorded_hash,
    }


def build_late_progress_protocol(
    config: MirrorLateProgressConditionalExtensionConfig,
    precheck: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "task_id": REC004H_TASK_ID,
        "source_task_id": REC004H_SOURCE_TASK_ID,
        "locked_before": "B1 additional forward passes (length-10 token-level audit)",
        "audit_steps": list(REC004H_AUDIT_STEPS),
        "phase_match_steps": list(REC004H_PHASE_MATCH_STEPS),
        "target_length": REC004H_TARGET_LENGTH,
        "existing_validation_examples_total": REC004H_EXISTING_VALIDATION_EXAMPLES,
        "existing_validation_split": REC004H_EXISTING_VALIDATION_SPLIT,
        "existing_validation_floor": REC004H_EXISTING_VALIDATION_FLOOR,
        "below_floor_predicate": (
            "100 * sequence_correct_all < 95 * n_examples_all (integer comparison)"
        ),
        "loss_drop_eps": REC004H_LOSS_EPS,
        "loss_drop_definition": "loss_drop(a, b) := L_b < L_a - eps",
        "em_progress_definition": (
            "(C_12000 > C_10000) and (C_12000 > C_8000) and "
            "((E_12000 < E_10000) or loss_drop(L_10000, L_12000))"
        ),
        "soft_progress_definition": (
            "(C_12000 >= C_10000) and (C_12000 >= C_8000) and "
            "(E_12000 < E_10000) and (E_12000 < E_8000) and "
            "loss_drop(L_8000, L_12000) and loss_drop(L_10000, L_12000)"
        ),
        "init_progress_confirmed_definition": "EM_PROGRESS or SOFT_PROGRESS",
        "decision_rule_order": [
            "BLOCKED if source/data/phase/freeze contract fails",
            "EVIDENCE_INSUFFICIENT if required evidence is missing",
            "ALREADY_AT_FLOOR_AUDIT_ONLY if no init is below the overall floor",
            "EXTEND_ALL_FIVE_TO_18000 if every below-floor init has INIT_PROGRESS_CONFIRMED",
            "STOP_NO_EXTENSION otherwise",
        ],
        "source_replay_em_tolerance": REC004H_SOURCE_REPLAY_EM_TOL,
        "extension_additional_updates_per_init": REC004H_ADDITIONAL_UPDATES,
        "extension_target_cumulative_step": REC004H_TARGET_CUMULATIVE_STEP,
        "extension_max_new_updates_total": REC004H_MAX_UPDATES_TOTAL,
        "extension_recipe_unchanged": {
            "optimizer": "AdamW", "lr": REC004H_OPERATOR_LR,
            "weight_decay": REC004H_OPERATOR_WEIGHT_DECAY,
            "grad_clip": REC004H_OPERATOR_GRAD_CLIP,
            "scheduler": "CosineAnnealingLR", "T_max": REC004H_T_MAX,
            "eta_min": REC004H_SCHEDULER_ETA_MIN,
        },
        "precheck_hash_status": {
            i: {str(s): precheck[i][s]["status"] for s in REC004H_AUDIT_STEPS} for i in precheck
        },
        "locked_at_unix_time": time.time(),
        "note": (
            "This predicate is a NEW operational resource-decision rule introduced by "
            "this task, not a statistical significance test, not a re-derivation of the "
            "pre-registered 0.95 recovery floor, and not a guarantee of eventual "
            "convergence. loss_drop's eps is a numerical comparison margin only."
        ),
    }


# =============================================================================
# Stage B1 -- the substantive forward-only audit. ONE combined chunked
# forward pass per checkpoint over the full fixed 1024-example existing-
# validation set yields BOTH the source-replay reproduction (overall EM,
# by-length sequence counts -- values REC-004G already recorded) AND the new
# length-10 token-level statistics REC-004G never recorded, from the SAME
# logits -- no example is forwarded through the primitive twice.
# =============================================================================


def _run_full_checkpoint_forward(
    core: Any, primitive: Any, examples: list[Any], device: torch.device
) -> dict[str, Any]:
    primitive.eval()
    by_length: dict[int, dict[str, int]] = {}
    sequence_correct_all = 0
    per_example_correct: list[bool] = []

    target_len = REC004H_TARGET_LENGTH
    len10_valid_token_count = 0
    len10_token_error_count = 0
    len10_token_ce_sum = 0.0
    len10_position: dict[int, dict[str, int]] = {
        i: {"correct": 0, "errors": 0, "valid_count": 0} for i in range(target_len)
    }
    len10_structural = {"moved": {"n": 0, "correct": 0}, "fixed": {"n": 0, "correct": 0}}
    len10_sequence_correct = 0
    len10_n = 0
    len10_errors_per_sequence: list[int] = []
    len10_first_error_position: list[int | None] = []
    pi_10 = mpid.mirror_halves_position_map(target_len)

    with torch.no_grad():
        for start in range(0, len(examples), 128):
            chunk = examples[start : start + 128]
            content_lengths = [len(ex.input_tokens) for ex in chunk]
            output_lengths = [
                get_operation(REC004H_TARGET_OPERATION).output_length(n) for n in content_lengths
            ]
            out_max = max(output_lengths)
            labels = _labels_for_examples(chunk, output_lengths, out_max, device)
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
            logits = primitive(h, content_lengths, output_lengths, None)
            token_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="none",
            ).reshape(labels.shape)
            preds = logits.argmax(dim=-1)
            valid_mask = labels != IGNORE_INDEX
            token_correct_mask = (preds == labels) & valid_mask

            for row, ex in enumerate(chunk):
                n = output_lengths[row]
                pred = tuple(preds[row, :n].tolist())
                target = ex.target_tokens
                length = len(ex.input_tokens)
                is_exact = bool(len(pred) == len(target) and pred == target)
                bucket = by_length.setdefault(length, {"n": 0, "exact": 0})
                bucket["n"] += 1
                bucket["exact"] += int(is_exact)
                sequence_correct_all += int(is_exact)
                per_example_correct.append(is_exact)

                if length == target_len:
                    len10_n += 1
                    n_err = 0
                    first_err: int | None = None
                    for i in range(target_len):
                        if not bool(valid_mask[row, i]):
                            continue
                        c = bool(token_correct_mask[row, i])
                        len10_position[i]["valid_count"] += 1
                        if c:
                            len10_position[i]["correct"] += 1
                        else:
                            len10_position[i]["errors"] += 1
                            n_err += 1
                            if first_err is None:
                                first_err = i
                        key = "moved" if pi_10[i] != i else "fixed"
                        len10_structural[key]["n"] += 1
                        len10_structural[key]["correct"] += int(c)
                    len10_valid_token_count += int(valid_mask[row, :target_len].sum().item())
                    len10_token_error_count += n_err
                    len10_token_ce_sum += float(
                        token_ce[row, :target_len][valid_mask[row, :target_len]].sum().item()
                    )
                    len10_errors_per_sequence.append(n_err)
                    len10_first_error_position.append(first_err)
                    len10_sequence_correct += int(n_err == 0)

    n_all = len(examples)
    overall_em = sequence_correct_all / n_all if n_all else None
    len10_mean_ce = (
        len10_token_ce_sum / len10_valid_token_count if len10_valid_token_count > 0 else None
    )
    len10_token_accuracy = (
        (len10_valid_token_count - len10_token_error_count) / len10_valid_token_count
        if len10_valid_token_count > 0 else None
    )
    return {
        "n_examples": n_all,
        "sequence_correct_all": sequence_correct_all,
        "overall_exact_match": overall_em,
        "by_length": {str(k): v for k, v in sorted(by_length.items())},
        "per_example_correct": per_example_correct,
        "length10": {
            "length": target_len,
            "n_examples": len10_n,
            "sequence_correct": len10_sequence_correct,
            "sequence_EM": len10_sequence_correct / len10_n if len10_n else None,
            "valid_token_count": len10_valid_token_count,
            "token_error_count": len10_token_error_count,
            "token_accuracy": len10_token_accuracy,
            "unreduced_token_CE_sum": len10_token_ce_sum,
            "mean_valid_token_CE": len10_mean_ce,
            "per_output_position": {str(i): v for i, v in len10_position.items()},
            "structural_moved_vs_fixed": len10_structural,
            "errors_per_sequence": len10_errors_per_sequence,
            "first_error_position": len10_first_error_position,
        },
    }


def run_checkpoint_audit(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    step: int,
    rows: list[dict[str, Any]],
    examples: list[Any],
) -> dict[str, Any]:
    """Forward-only, zero-optimizer-update audit of ONE saved REC-004G
    checkpoint. Deliberately does NOT call `_guard_not_frozen` -- this
    function only loads weights and forwards; it is designed to run inside a
    `frozen_evaluation()` block (see the orchestrator below), which is the
    mechanical proof that nothing in the audit stage can build an optimizer."""
    path = _rec004g_checkpoint_path(init_id, step)
    if not path.is_file():
        return {
            "init_id": init_id, "step": step,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE", "path": str(path),
        }
    state = torch.load(path, map_location="cpu", weights_only=False)
    computed_hash = mb.canonical_state_hash(state)
    row = _rec004g_learning_curve_row(rows, init_id, step)
    if row is None:
        return {
            "init_id": init_id, "step": step,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "reason": "no matching learning_curve.jsonl row",
        }
    recorded_hash = row["checkpoint_state_hash"]
    if computed_hash != recorded_hash:
        return {
            "init_id": init_id, "step": step, "status": "SOURCE_REPLAY_MISMATCH",
            "computed_checkpoint_state_hash": computed_hash,
            "recorded_checkpoint_state_hash": recorded_hash,
        }

    device = core.device
    primitive = mpbr._new_arm_primitive(core, REC004H_ARM)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    primitive.eval()

    pid = REC004H_TARGET_PHYSICAL_ID
    original_slot = eval_bank.get(pid)
    eval_bank.replace_primitive(pid, primitive)
    try:
        forward_result = _run_full_checkpoint_forward(core, primitive, examples, device)
    finally:
        eval_bank.replace_primitive(pid, original_slot)

    recorded_em = row["existing_validation"]["correct_exact_match"]
    reproduced_em = forward_result["overall_exact_match"]
    em_matches = (
        reproduced_em is not None
        and abs(reproduced_em - recorded_em) < REC004H_SOURCE_REPLAY_EM_TOL
    )

    recorded_len10 = row["length_stratified"]["by_length"].get(str(REC004H_TARGET_LENGTH))
    computed_len10 = forward_result["by_length"].get(str(REC004H_TARGET_LENGTH))
    len10_count_matches = (
        recorded_len10 is not None and computed_len10 is not None
        and recorded_len10["exact"] == computed_len10["exact"]
        and recorded_len10["n"] == computed_len10["n"]
    )

    status = "VERIFIED" if (em_matches and len10_count_matches) else "SOURCE_REPLAY_MISMATCH"
    return {
        "init_id": init_id, "step": step, "status": status,
        "computed_checkpoint_state_hash": computed_hash,
        "recorded_checkpoint_state_hash": recorded_hash,
        "reproduced_overall_exact_match": reproduced_em,
        "recorded_overall_exact_match": recorded_em,
        "overall_em_matches_within_tolerance": em_matches,
        "recorded_length10_by_length": recorded_len10,
        "computed_length10_by_length": computed_len10,
        "length10_count_matches": len10_count_matches,
        "by_length": forward_result["by_length"],
        "length10_token_audit": forward_result["length10"],
        "per_example_correct": forward_result["per_example_correct"],
        "lr_used": row.get("lr_used"),
        "lr_after_scheduler": row.get("lr_after_scheduler"),
    }


# =============================================================================
# Stage B2/B3/B4 -- pure arithmetic on already-collected checkpoint audits
# (no additional forward passes).
# =============================================================================


def build_position_error_trajectory(
    checkpoint_audits: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for init_id in REC004H_INIT_IDS:
        steps: dict[str, Any] = {}
        for step in REC004H_AUDIT_STEPS:
            audit = checkpoint_audits[init_id].get(step)
            if audit is None or audit.get("status") != "VERIFIED":
                steps[str(step)] = {"status": (audit or {}).get("status", "MISSING")}
                continue
            token_audit = audit["length10_token_audit"]
            steps[str(step)] = {
                "status": "VERIFIED",
                "per_output_position": token_audit["per_output_position"],
                "structural_moved_vs_fixed": token_audit["structural_moved_vs_fixed"],
                "sequence_correct": token_audit["sequence_correct"],
                "n_examples": token_audit["n_examples"],
                "first_error_position_counts": {
                    str(p): token_audit["first_error_position"].count(p)
                    for p in set(token_audit["first_error_position"])
                },
            }
        result[init_id] = steps
    result["fixed_positions_length10"] = list(
        i for i in range(REC004H_TARGET_LENGTH)
        if mpid.mirror_halves_position_map(REC004H_TARGET_LENGTH)[i] == i
    )
    return result


def build_same_phase_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_init: dict[str, Any] = {}
    for init_id in REC004H_INIT_IDS:
        entries: dict[str, Any] = {}
        for step in REC004H_AUDIT_STEPS:
            row = _rec004g_learning_curve_row(rows, init_id, step)
            entries[str(step)] = (
                {"lr_used": row.get("lr_used"), "lr_after_scheduler": row.get("lr_after_scheduler")}
                if row is not None else None
            )
        phase_vals = [
            entries[str(s)]["lr_after_scheduler"]
            for s in REC004H_PHASE_MATCH_STEPS
            if entries.get(str(s)) is not None
        ]
        same_phase = len(phase_vals) == len(REC004H_PHASE_MATCH_STEPS) and (
            max(phase_vals) - min(phase_vals) < 1e-6
        )
        per_init[init_id] = {
            "all_audit_steps_lr": entries,
            "phase_match_steps_consistent": same_phase,
        }
    return {"task_id": REC004H_TASK_ID, "per_init": per_init}


def build_improvement_vs_residual_decomposition(
    checkpoint_audits: dict[str, dict[int, dict[str, Any]]],
    rec004g_paired_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for init_id in REC004H_INIT_IDS:
        entry: dict[str, Any] = {
            "step_6000_to_12000_reused_from_rec004g": (
                (rec004g_paired_comparison or {}).get(init_id)
            ),
        }
        by_length_by_step = {
            step: checkpoint_audits[init_id][step]["by_length"]
            for step in REC004H_PHASE_MATCH_STEPS
            if checkpoint_audits[init_id].get(step, {}).get("status") == "VERIFIED"
        }
        pair_results: dict[str, Any] = {}
        for a, b in ((8000, 10000), (10000, 12000), (8000, 12000)):
            if a not in by_length_by_step or b not in by_length_by_step:
                pair_results[f"{a}_to_{b}"] = {"status": "EVIDENCE_INSUFFICIENT"}
                continue
            bl_a, bl_b = by_length_by_step[a], by_length_by_step[b]
            lengths = sorted(set(bl_a) | set(bl_b), key=int)
            correct_a = sum(v["exact"] for v in bl_a.values())
            correct_b = sum(v["exact"] for v in bl_b.values())
            delta_all = correct_b - correct_a
            errors_all_b = sum(v["n"] - v["exact"] for v in bl_b.values())
            per_length: dict[str, Any] = {}
            for length in lengths:
                ea = bl_a.get(length, {}).get("exact", 0)
                eb = bl_b.get(length, {}).get("exact", 0)
                nb = bl_b.get(length, {}).get("n", 0)
                delta = eb - ea
                errors_b = nb - eb
                per_length[length] = {
                    "delta_correct": delta,
                    "contribution_to_total_delta": (delta / delta_all) if delta_all != 0 else None,
                    "errors_at_b": errors_b,
                    "error_share_of_all_errors_at_b": (
                        errors_b / errors_all_b if errors_all_b > 0 else None
                    ),
                }
            target_len_str = str(REC004H_TARGET_LENGTH)
            lengths_6_9_delta = sum(
                v["delta_correct"] for length, v in per_length.items() if length != target_len_str
            )
            pair_results[f"{a}_to_{b}"] = {
                "delta_correct_all": delta_all,
                "per_length": per_length,
                "lengths_6_9_combined_delta_correct": lengths_6_9_delta,
                "note": "the combined 6-9 figure is not a substitute for each length's own value",
            }
        entry["phase_match_steps_decomposition"] = pair_results
        result[init_id] = entry
    return result


# =============================================================================
# Stage C -- the continuation predicate (this task's own new operational
# resource-decision rule; not a statistical test, not a re-derivation of the
# 0.95 recovery floor).
# =============================================================================


def _below_floor_from_audit(audit: dict[str, Any], n_all: int) -> bool:
    correct_all = sum(v["exact"] for v in audit["by_length"].values())
    return 100 * correct_all < 95 * n_all


def compute_progress_flags(
    C: dict[int, int], E: dict[int, int], L: dict[int, float | None]
) -> dict[str, Any]:
    """C/E/L keyed by step in REC004H_PHASE_MATCH_STEPS (8000, 10000, 12000)."""
    missing = [
        s for s in REC004H_PHASE_MATCH_STEPS
        if s not in C or s not in E or s not in L or L.get(s) is None
    ]
    nonfinite = any(
        not math.isfinite(v)
        for s in REC004H_PHASE_MATCH_STEPS
        if (v := L.get(s)) is not None
    )
    if missing or nonfinite:
        return {"status": "EVIDENCE_INSUFFICIENT", "missing_steps": missing}

    C8, C10, C12 = C[8000], C[10000], C[12000]
    E8, E10, E12 = E[8000], E[10000], E[12000]
    L8_opt, L10_opt, L12_opt = L[8000], L[10000], L[12000]
    assert L8_opt is not None and L10_opt is not None and L12_opt is not None
    L8: float = L8_opt
    L10: float = L10_opt
    L12: float = L12_opt

    def loss_drop(a_val: float, b_val: float) -> bool:
        return b_val < a_val - REC004H_LOSS_EPS

    em_progress = (C12 > C10) and (C12 > C8) and ((E12 < E10) or loss_drop(L10, L12))
    soft_progress = (
        (C12 >= C10) and (C12 >= C8)
        and (E12 < E10) and (E12 < E8)
        and loss_drop(L10, L12) and loss_drop(L8, L12)
    )
    return {
        "status": "COMPUTED",
        "C_8000": C8, "C_10000": C10, "C_12000": C12,
        "E_8000": E8, "E_10000": E10, "E_12000": E12,
        "L_8000": L8, "L_10000": L10, "L_12000": L12,
        "loss_drop_10000_12000": loss_drop(L10, L12),
        "loss_drop_8000_12000": loss_drop(L8, L12),
        "EM_PROGRESS": em_progress,
        "SOFT_PROGRESS": soft_progress,
        "INIT_PROGRESS_CONFIRMED": em_progress or soft_progress,
    }


def build_continuation_decision(
    below_floor: dict[str, bool | None], per_init_progress: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if any(v is None for v in below_floor.values()):
        return {
            "decision": "EVIDENCE_INSUFFICIENT",
            "reason": "step=12000 audit not VERIFIED for at least one init",
            "below_floor_at_12000": below_floor,
        }
    below_floor_inits = [i for i in REC004H_INIT_IDS if below_floor[i]]
    if not below_floor_inits:
        return {
            "decision": "ALREADY_AT_FLOOR_AUDIT_ONLY",
            "below_floor_at_12000": below_floor,
            "below_floor_inits": below_floor_inits,
        }
    insufficient = [i for i in below_floor_inits if per_init_progress[i]["status"] != "COMPUTED"]
    if insufficient:
        return {
            "decision": "EVIDENCE_INSUFFICIENT",
            "below_floor_at_12000": below_floor,
            "below_floor_inits": below_floor_inits,
            "evidence_insufficient_inits": insufficient,
        }
    confirmed = {i: per_init_progress[i]["INIT_PROGRESS_CONFIRMED"] for i in below_floor_inits}
    all_confirmed = all(confirmed.values())
    return {
        "decision": "EXTEND_ALL_FIVE_TO_18000" if all_confirmed else "STOP_NO_EXTENSION",
        "below_floor_at_12000": below_floor,
        "below_floor_inits": below_floor_inits,
        "init_progress_confirmed": confirmed,
        "mixed_or_no_confirmed_late_progress": (
            "MIXED_OR_NO_CONFIRMED_LATE_PROGRESS" if not all_confirmed else None
        ),
    }


# =============================================================================
# Stage E -- extension (ONLY reached if EXTEND_ALL_FIVE_TO_18000). Mirrors
# `mirror_budget_extension.run_source_replay`/`run_one_extension` exactly,
# generalized from REC-004D's step=6000 source to REC-004G's own step=12000
# source and from target=12000 to target=18000. `_guard_not_frozen` is
# called at the top of every function here, matching REC-004G's own
# convention -- these functions must never run while `frozen_evaluation()`
# is active.
# =============================================================================


def load_source_training_state_for_extension(init_id: str) -> dict[str, Any]:
    path = _rec004g_training_state_path(init_id, REC004H_SOURCE_STEP_FOR_EXTENSION)
    if not path.is_file():
        raise mb.MissingArtifactError(f"SOURCE_ARTIFACT_UNAVAILABLE: {path} not found")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("cumulative_updates") != REC004H_SOURCE_STEP_FOR_EXTENSION:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {path} records cumulative_updates="
            f"{state.get('cumulative_updates')}, expected {REC004H_SOURCE_STEP_FOR_EXTENSION}"
        )
    if state.get("init_id") != init_id or state.get("arm") != REC004H_ARM:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {path} records init_id={state.get('init_id')!r}/"
            f"arm={state.get('arm')!r}, expected {init_id!r}/{REC004H_ARM!r}"
        )
    return state


def run_extension_source_replay(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    rows: list[dict[str, Any]],
    config: MirrorLateProgressConditionalExtensionConfig,
) -> dict[str, Any]:
    _guard_not_frozen(f"run_extension_source_replay:{init_id}")
    training_state = load_source_training_state_for_extension(init_id)
    primitive_state = {
        k: v.detach().clone().cpu() for k, v in training_state["primitive_state_dict"].items()
    }
    computed_hash = mb.canonical_state_hash(primitive_state)
    row = _rec004g_learning_curve_row(rows, init_id, REC004H_SOURCE_STEP_FOR_EXTENSION)
    if row is None:
        return {"init_id": init_id, "status": "SOURCE_ARTIFACT_UNAVAILABLE"}
    recorded_hash = row["checkpoint_state_hash"]
    if computed_hash != recorded_hash:
        return {
            "init_id": init_id, "status": "SOURCE_REPLAY_MISMATCH",
            "computed_checkpoint_state_hash": computed_hash,
            "recorded_checkpoint_state_hash": recorded_hash,
        }

    pid = REC004H_TARGET_PHYSICAL_ID
    device = core.device
    primitive = mpbr._new_arm_primitive(core, REC004H_ARM)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in primitive_state.items()}, strict=True)
    primitive.eval()
    original_slot = eval_bank.replace_primitive(pid, primitive)
    try:
        forward_result = _run_full_checkpoint_forward(
            core, primitive, _existing_validation_examples(config.seed), device
        )
    finally:
        eval_bank.replace_primitive(pid, original_slot)

    reproduced_em = forward_result["overall_exact_match"]
    recorded_em = row["existing_validation"]["correct_exact_match"]
    matches = (
        reproduced_em is not None
        and abs(reproduced_em - recorded_em) < REC004H_SOURCE_REPLAY_EM_TOL
    )
    return {
        "init_id": init_id,
        "status": "VERIFIED" if matches else "SOURCE_REPLAY_MISMATCH",
        "computed_checkpoint_state_hash": computed_hash,
        "recorded_checkpoint_state_hash": recorded_hash,
        "reproduced_existing_validation_em": reproduced_em,
        "recorded_existing_validation_em": recorded_em,
        "matches_within_tolerance": matches,
        "cumulative_updates": training_state.get("cumulative_updates"),
        "next_training_sample_step": REC004H_SOURCE_STEP_FOR_EXTENSION + 1,
        "training_state": training_state,
    }


def run_one_late_extension(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    replay: dict[str, Any],
    config: MirrorLateProgressConditionalExtensionConfig,
    output_dir: Path,
) -> dict[str, Any]:
    _guard_not_frozen(f"run_one_late_extension:{init_id}")
    pid = REC004H_TARGET_PHYSICAL_ID
    device = core.device
    seed = config.seed
    target_step = REC004H_TARGET_CUMULATIVE_STEP
    training_state = replay["training_state"]

    primitive = mpbr._new_arm_primitive(core, REC004H_ARM)
    primitive.to(device)
    primitive.load_state_dict(
        {k: v.to(device) for k, v in training_state["primitive_state_dict"].items()}, strict=True
    )
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004H_OPERATOR_LR, weight_decay=REC004H_OPERATOR_WEIGHT_DECAY
    )
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004H_T_MAX, eta_min=REC004H_SCHEDULER_ETA_MIN
    )
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    if scheduler.last_epoch != REC004H_SOURCE_STEP_FOR_EXTENSION:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: resumed scheduler.last_epoch="
            f"{scheduler.last_epoch}, expected {REC004H_SOURCE_STEP_FOR_EXTENSION}"
        )
    mbe._restore_rng_state(training_state, device)

    run_dir = output_dir / init_id / REC004H_ARM
    ckpt_dir = run_dir / "checkpoints"
    state_dir = run_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints: list[dict[str, Any]] = []
    lr_trace: list[dict[str, Any]] = []
    cumulative_examples = 0
    t_start = time.time()
    diverged_at: int | None = None

    def _snapshot_and_eval(step: int, lr_used: float | None, lr_after: float) -> None:
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
                "arm": REC004H_ARM,
            },
            state_dir / f"step{step}.pt",
        )

        primitive.eval()
        original_slot = eval_bank.replace_primitive(pid, primitive)
        try:
            forward_result = _run_full_checkpoint_forward(
                core, primitive, _existing_validation_examples(seed), device
            )
            existing_validation = {
                "actual_examples": forward_result["n_examples"],
                "correct_exact_match": forward_result["overall_exact_match"],
            }
        finally:
            eval_bank.replace_primitive(pid, original_slot)
        train_fit = ibc._evaluate_train_fit(
            core, primitive, REC004H_TARGET_OPERATION, seed=seed,
            vocab_size=REC004H_VOCAB_SIZE, sequence_length_range=REC004H_SEQUENCE_LENGTH_RANGE,
        )

        extras: dict[str, Any] | None = None
        if step == target_step:
            final_examples = _existing_validation_examples(seed)
            position_summary, position_confusion = mpbr._position_level_breakdown(
                core, primitive, final_examples
            )
            assert isinstance(primitive, mpbr.CrossPositionLengthBiasPrimitive)
            extras = {
                "position_summary": position_summary,
                "position_confusion": position_confusion,
                "bias_ablation": mpbr.run_bias_ablation(core, primitive, final_examples),
                "per_example_correct": forward_result["per_example_correct"],
                "sequence_correct_all": forward_result["sequence_correct_all"],
            }
        primitive.train()

        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        checkpoints.append(
            {
                "step": step, "init_id": init_id, "arm": REC004H_ARM,
                "cumulative_training_examples": cumulative_examples,
                "lr_used": lr_used, "lr_after_scheduler": lr_after,
                "existing_validation": existing_validation,
                "train_fit": train_fit,
                "by_length": forward_result["by_length"],
                "length10_token_audit": forward_result["length10"],
                "final_step_extras": extras,
                "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
                "cumulative_wall_clock_seconds": time.time() - t_start,
                "peak_vram_bytes": peak_vram,
            }
        )

    for step in range(REC004H_SOURCE_STEP_FOR_EXTENSION + 1, target_step + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, REC004H_TARGET_OPERATION,
            vocab_size=REC004H_VOCAB_SIZE, sequence_length_range=REC004H_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(examples)

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004H_TARGET_OPERATION).output_length(n) for n in content_lengths
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004H_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())
        lr_trace.append(
            {
                "u": step, "lr_used": lr_used,
                "lr_after_scheduler": lr_after, "train_loss": running_loss,
            }
        )

        if not math.isfinite(running_loss):
            diverged_at = step
            break

        if step % REC004H_CHECKPOINT_INTERVAL == 0:
            _snapshot_and_eval(step, lr_used, lr_after)

    if diverged_at is not None and not any(c["step"] == diverged_at for c in checkpoints):
        checkpoints.append(
            {
                "step": diverged_at, "status": "NOT_EXECUTED",
                "reason": f"loss diverged at step {diverged_at}",
            }
        )

    checkpoints.sort(key=lambda c: c["step"])
    primitive.eval()
    final_state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    new_updates = (
        (diverged_at - REC004H_SOURCE_STEP_FOR_EXTENSION - 1)
        if diverged_at is not None
        else (target_step - REC004H_SOURCE_STEP_FOR_EXTENSION)
    )
    return {
        "init_id": init_id, "arm": REC004H_ARM, "target_cumulative_step": target_step,
        "checkpoints": checkpoints, "lr_trace": lr_trace,
        "diverged_at_step": diverged_at,
        "new_optimizer_updates": max(new_updates, 0),
        "final_primitive_state_dict": final_state,
        "total_wall_clock_seconds": time.time() - t_start,
    }


def build_extension_protocol(
    config: MirrorLateProgressConditionalExtensionConfig,
) -> dict[str, Any]:
    return {
        "task_id": REC004H_TASK_ID,
        "source_task_id": REC004H_SOURCE_TASK_ID,
        "source_step": REC004H_SOURCE_STEP_FOR_EXTENSION,
        "target_cumulative_step": REC004H_TARGET_CUMULATIVE_STEP,
        "additional_updates_per_init": REC004H_ADDITIONAL_UPDATES,
        "max_new_updates_total": REC004H_MAX_UPDATES_TOTAL,
        "init_ids": list(REC004H_INIT_IDS),
        "recipe": {
            "optimizer": "AdamW", "lr": REC004H_OPERATOR_LR,
            "weight_decay": REC004H_OPERATOR_WEIGHT_DECAY,
            "grad_clip": REC004H_OPERATOR_GRAD_CLIP,
            "scheduler": "CosineAnnealingLR", "T_max": REC004H_T_MAX,
            "eta_min": REC004H_SCHEDULER_ETA_MIN,
        },
        "checkpoint_interval": REC004H_CHECKPOINT_INTERVAL,
        "locked_at_unix_time": time.time(),
        "note": (
            "All 5 inits proceed to the same terminal step=18000 regardless of "
            "individual step-12000 standing; no early stop at 95%, no candidate "
            "selection, no independent query generation."
        ),
    }


def _example_digest(ex: Any) -> str:
    payload = [list(ex.input_tokens), list(ex.target_tokens)]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def build_extension_data_manifest(seed: int, validation_examples: list[Any]) -> dict[str, Any]:
    val_digest_to_indices: dict[str, list[int]] = {}
    for idx, ex in enumerate(validation_examples):
        val_digest_to_indices.setdefault(_example_digest(ex), []).append(idx)
    overlap_count = 0
    overlap_val_indices: set[int] = set()
    overlap_rows: list[dict[str, Any]] = []
    total = 0
    for step in range(REC004H_SOURCE_STEP_FOR_EXTENSION + 1, REC004H_TARGET_CUMULATIVE_STEP + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, REC004H_TARGET_OPERATION,
            vocab_size=REC004H_VOCAB_SIZE, sequence_length_range=REC004H_SEQUENCE_LENGTH_RANGE,
        )
        for ex in examples:
            total += 1
            digest = _example_digest(ex)
            matched = val_digest_to_indices.get(digest)
            if matched:
                overlap_count += 1
                overlap_val_indices.update(matched)
                overlap_rows.append(
                    {
                        "train_step": step, "validation_indices": matched,
                        "length": len(ex.input_tokens),
                    }
                )
    return {
        "task_id": REC004H_TASK_ID,
        "total_new_training_examples": total,
        "existing_validation_examples": len(validation_examples),
        "overlap_count": overlap_count,
        "overlap_free": overlap_count == 0,
        "overlap_validation_indices": sorted(overlap_val_indices),
        "overlap_rows": overlap_rows,
        "note": (
            "Training-stream digests are (input_tokens, target_tokens) pairs, checked "
            "against the fixed existing-validation set's own digests -- this is a "
            "developer-side sample/content audit, not the generation of a new "
            "independent RG3 query or holdout set."
        ),
    }


def build_data_overlap_impact_audit(
    data_manifest: dict[str, Any],
    per_init_terminal_forward: dict[str, dict[str, Any]],
    floor: float,
) -> dict[str, Any]:
    """Post-hoc sensitivity audit: for each init's step=18000 forward result
    (must include `per_example_correct` and `sequence_correct_all`), computes
    the WORST-CASE terminal correct-count and floor status if every single
    overlap-covered validation example were removed from credit entirely
    (i.e. treated as if it could not have been answered without exposure
    during the extension training stream). This does not claim leakage
    caused any specific answer -- it bounds how much the disclosed overlap
    COULD have mattered."""
    overlap_indices = data_manifest["overlap_validation_indices"]
    per_init: dict[str, Any] = {}
    for init_id, fwd in per_init_terminal_forward.items():
        per_example_correct = fwd["per_example_correct"]
        n_all = len(per_example_correct)
        overall_correct = fwd["sequence_correct_all"]
        overlap_correct = [per_example_correct[i] for i in overlap_indices]
        n_overlap_correct = sum(overlap_correct)
        worst_case_correct = overall_correct - n_overlap_correct
        per_init[init_id] = {
            "overall_correct_count": overall_correct,
            "n_examples": n_all,
            "overlap_examples_answered_correctly": n_overlap_correct,
            "overlap_examples_total": len(overlap_indices),
            "worst_case_correct_if_overlap_examples_excluded": worst_case_correct,
            "actual_clears_floor": 100 * overall_correct >= 95 * n_all,
            "worst_case_clears_floor": 100 * worst_case_correct >= 95 * n_all,
            "floor_conclusion_robust_to_disclosed_overlap": (
                (100 * overall_correct >= 95 * n_all) == (100 * worst_case_correct >= 95 * n_all)
            ),
        }
    return {
        "task_id": REC004H_TASK_ID,
        "floor": floor,
        "overlap_validation_indices": overlap_indices,
        "overlap_count_in_training_stream": data_manifest["overlap_count"],
        "total_new_training_examples": data_manifest["total_new_training_examples"],
        "per_init": per_init,
        "caveat": (
            "This audit does not by itself prove leakage changed any specific answer -- "
            "the overlap-covered examples are short (length <= 7 in the observed run) and "
            "were answered correctly by every init, including ones with no borderline floor "
            "call, consistent with them being easy items rather than memorized ones. It "
            "reports a bound, not a causal attribution."
        ),
    }


def build_terminal_floor_status(
    outcomes: dict[str, dict[str, Any]], floor: float
) -> dict[str, Any]:
    em_at_18000: dict[str, float | None] = {}
    for init_id, o in outcomes.items():
        final = next(
            (c for c in o["checkpoints"] if c.get("step") == REC004H_TARGET_CUMULATIVE_STEP
             and "existing_validation" in c),
            None,
        )
        em_at_18000[init_id] = (
            final["existing_validation"]["correct_exact_match"] if final else None
        )
    clears = {
        i: (em is not None and em >= floor) for i, em in em_at_18000.items()
    }
    all_present = all(v is not None for v in em_at_18000.values())
    all_clear = all_present and all(clears.values())
    status = "ALL_FIVE_AT_18000_FLOOR" if all_clear else (
        "TARGET_NOT_MET_AT_18000" if all_present else "INCOMPLETE"
    )
    return {
        "task_id": REC004H_TASK_ID, "floor": floor,
        "existing_validation_em_at_18000": em_at_18000,
        "clears_floor": clears,
        "terminal_floor_status": status,
        "note": (
            "Development-set floor confirmation only. No candidate is selected, no "
            "child bundle is built, and no RG3 recheck runs regardless of this result."
        ),
        "selected_init": None, "selected_intervention": None, "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
    }


def _strip_training_state(replay: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        init_id: {k: v for k, v in r.items() if k != "training_state"}
        for init_id, r in replay.items()
    }


def run_extension_for_all_inits(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    config: MirrorLateProgressConditionalExtensionConfig,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pid = REC004H_TARGET_PHYSICAL_ID
    original_slot = eval_bank.get(pid)

    replay: dict[str, dict[str, Any]] = {}
    for init_id in REC004H_INIT_IDS:
        eval_bank.replace_primitive(pid, original_slot)
        replay[init_id] = run_extension_source_replay(
            core, eval_bank, op_to_id, init_id, rows, config
        )
    eval_bank.replace_primitive(pid, original_slot)

    blocked = [i for i, r in replay.items() if r["status"] != "VERIFIED"]
    if blocked:
        return {
            "status": "BLOCKED",
            "blocked_inits": blocked,
            "replay": _strip_training_state(replay),
        }

    outcomes: dict[str, dict[str, Any]] = {}
    for init_id in REC004H_INIT_IDS:
        eval_bank.replace_primitive(pid, original_slot)
        outcomes[init_id] = run_one_late_extension(
            core, eval_bank, op_to_id, init_id, replay[init_id], config, output_dir
        )
    eval_bank.replace_primitive(pid, original_slot)

    return {
        "status": "COMPLETE",
        "replay": _strip_training_state(replay),
        "outcomes": outcomes,
    }


# =============================================================================
# Orchestration
# =============================================================================


def build_cost_accounting(
    audit_forward_examples: int, extension_outcomes: dict[str, dict[str, Any]] | None
) -> dict[str, Any]:
    new_updates_by_init = (
        {i: o["new_optimizer_updates"] for i, o in extension_outcomes.items()}
        if extension_outcomes else {}
    )
    return {
        "task_id": REC004H_TASK_ID,
        "audit_stage_prediction_examples_used": audit_forward_examples,
        "audit_stage_prediction_examples_budget": REC004H_MAX_PREDICTION_EXAMPLES_BUDGET,
        "audit_stage_within_budget": (
            audit_forward_examples <= REC004H_MAX_PREDICTION_EXAMPLES_BUDGET
        ),
        "new_optimizer_updates_by_init": new_updates_by_init,
        "total_new_optimizer_updates": sum(new_updates_by_init.values()),
        "max_new_optimizer_updates_total": REC004H_MAX_UPDATES_TOTAL,
    }


def run_mirror_late_progress_conditional_extension_task(
    config: MirrorLateProgressConditionalExtensionConfig,
) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004H reads REC-004G artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; got seed={seed}"
        )

    if not REC004H_CONTRACT_FILE.is_file():
        return {
            "task_id": REC004H_TASK_ID,
            "implementation_status": "BLOCKED",
            "reason": f"AUTHORIZATION_ARTIFACT_MISSING: {REC004H_CONTRACT_FILE} not found",
            "new_optimizer_updates": 0,
            "selected_init": None, "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED", "rec005_eligible": False,
        }
    if not REC004H_SOURCE_RUN_DIR.is_dir():
        return {
            "task_id": REC004H_TASK_ID,
            "implementation_status": "BLOCKED",
            "reason": f"SOURCE_ARTIFACT_UNAVAILABLE: {REC004H_SOURCE_RUN_DIR} not found",
            "new_optimizer_updates": 0,
            "selected_init": None, "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED", "rec005_eligible": False,
        }

    before_hashes = _snapshot_forbidden_cache_hashes(seed)
    rec004g_source_files_before = _rec004g_source_file_hashes()

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mbe._protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_before = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004H_FIXED_CANDIDATE_OPERATIONS
    }

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )

    rows = _load_rec004g_learning_curve_rows()

    # --- Stage A: hash-only precheck (no forward) across all audit checkpoints ---
    precheck = {
        init_id: {step: _hash_only_precheck(init_id, step, rows) for step in REC004H_AUDIT_STEPS}
        for init_id in REC004H_INIT_IDS
    }
    precheck_failed = [
        (i, s) for i in REC004H_INIT_IDS for s in REC004H_AUDIT_STEPS
        if precheck[i][s]["status"] != "VERIFIED"
    ]
    (output_dir / "source_manifest.json").write_text(
        json.dumps(rec004g_source_files_before, indent=2, default=str), encoding="utf-8"
    )
    if precheck_failed:
        summary = {
            "task_id": REC004H_TASK_ID, "source_task_id": REC004H_SOURCE_TASK_ID,
            "implementation_status": "BLOCKED",
            "source_replay_status": "SOURCE_REPLAY_MISMATCH",
            "late_audit_status": "BLOCKED",
            "continuation_decision": "BLOCKED",
            "extension_status": "NOT_EXECUTED",
            "new_optimizer_updates": 0,
            "terminal_floor_status": "NOT_EVALUATED",
            "precheck_failed": [f"{i}:{s}" for i, s in precheck_failed],
            "selected_init": None, "selected_intervention": None, "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED", "rec005_eligible": False, "rec005_executed": False,
            "total_wall_clock_seconds": time.time() - start,
        }
        (output_dir / "source_replay.json").write_text(
            json.dumps(precheck, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    # --- Lock the predicate/protocol BEFORE any B1 additional forward pass ---
    protocol = build_late_progress_protocol(config, precheck)
    (output_dir / "late_progress_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )

    examples = _existing_validation_examples(seed)

    checkpoint_audits: dict[str, dict[int, dict[str, Any]]] = {i: {} for i in REC004H_INIT_IDS}
    continuation: dict[str, Any]
    below_floor: dict[str, bool | None]
    per_init_progress: dict[str, Any]

    with frozen_evaluation():
        for init_id in REC004H_INIT_IDS:
            for step in REC004H_AUDIT_STEPS:
                checkpoint_audits[init_id][step] = run_checkpoint_audit(
                    core, eval_bank, op_to_id, init_id, step, rows, examples
                )

        (output_dir / "source_replay.json").write_text(
            json.dumps(
                {
                    i: {str(s): checkpoint_audits[i][s] for s in REC004H_AUDIT_STEPS}
                    for i in REC004H_INIT_IDS
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

        late_checkpoint_index = {
            i: {
                str(s): {
                    "path": str(_rec004g_checkpoint_path(i, s)),
                    "status": checkpoint_audits[i][s]["status"],
                    "checkpoint_state_hash": (
                        checkpoint_audits[i][s].get("computed_checkpoint_state_hash")
                    ),
                }
                for s in REC004H_AUDIT_STEPS
            }
            for i in REC004H_INIT_IDS
        }
        (output_dir / "late_checkpoint_index.json").write_text(
            json.dumps(late_checkpoint_index, indent=2, default=str), encoding="utf-8"
        )

        with (output_dir / "late_length_metrics.jsonl").open("w", encoding="utf-8") as fh:
            for init_id in REC004H_INIT_IDS:
                for step in REC004H_AUDIT_STEPS:
                    audit = checkpoint_audits[init_id][step]
                    if audit["status"] != "VERIFIED":
                        skip_row = {"init_id": init_id, "step": step, "status": audit["status"]}
                        fh.write(json.dumps(skip_row) + "\n")
                        continue
                    row_out = {
                        "init_id": init_id, "step": step, "status": "VERIFIED",
                        "overall_exact_match": audit["reproduced_overall_exact_match"],
                        "by_length": audit["by_length"],
                        **audit["length10_token_audit"],
                    }
                    fh.write(json.dumps(row_out, default=str) + "\n")

        (output_dir / "position_error_trajectory.json").write_text(
            json.dumps(build_position_error_trajectory(checkpoint_audits), indent=2, default=str),
            encoding="utf-8",
        )

        rec004g_paired_comparison = None
        pc_path = REC004H_SOURCE_RUN_DIR / "paired_extension_comparison.json"
        if pc_path.is_file():
            rec004g_paired_comparison = json.loads(pc_path.read_text(encoding="utf-8"))
        improvement_decomposition = build_improvement_vs_residual_decomposition(
            checkpoint_audits, rec004g_paired_comparison
        )
        (output_dir / "improvement_vs_residual_decomposition.json").write_text(
            json.dumps(improvement_decomposition, indent=2, default=str), encoding="utf-8"
        )

        (output_dir / "same_phase_audit.json").write_text(
            json.dumps(build_same_phase_audit(rows), indent=2, default=str), encoding="utf-8"
        )

        below_floor = {}
        per_init_progress = {}
        for init_id in REC004H_INIT_IDS:
            audit_12000 = checkpoint_audits[init_id][12000]
            if audit_12000["status"] != "VERIFIED":
                below_floor[init_id] = None
                per_init_progress[init_id] = {"status": "EVIDENCE_INSUFFICIENT"}
                continue
            below_floor[init_id] = _below_floor_from_audit(audit_12000, len(examples))

            C: dict[int, int] = {}
            E: dict[int, int] = {}
            L: dict[int, float | None] = {}
            for s in REC004H_PHASE_MATCH_STEPS:
                a = checkpoint_audits[init_id][s]
                if a["status"] != "VERIFIED":
                    continue
                t = a["length10_token_audit"]
                C[s] = t["sequence_correct"]
                E[s] = t["token_error_count"]
                L[s] = t["mean_valid_token_CE"]
            per_init_progress[init_id] = compute_progress_flags(C, E, L)

        continuation = build_continuation_decision(below_floor, per_init_progress)
        (output_dir / "late_progress_decision.json").write_text(
            json.dumps(
                {
                    "below_floor_at_12000": below_floor,
                    "per_init_progress": per_init_progress,
                    "continuation": continuation,
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
    # --- end frozen_evaluation(): audit stage is over; training may now run ---

    audit_forward_examples = len(REC004H_INIT_IDS) * len(REC004H_AUDIT_STEPS) * len(examples)

    extension_result: dict[str, Any] | None = None
    terminal_floor_status: dict[str, Any] | None = None
    new_optimizer_updates = 0

    if continuation["decision"] == "EXTEND_ALL_FIVE_TO_18000":
        (output_dir / "extension_protocol.json").write_text(
            json.dumps(build_extension_protocol(config), indent=2, default=str), encoding="utf-8"
        )
        data_manifest = build_extension_data_manifest(seed, examples)
        (output_dir / "extension_data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2, default=str), encoding="utf-8"
        )
        if not data_manifest["overlap_free"]:
            # Per this task's own S5.4 protocol: do not silently proceed on a
            # disclosed training/validation content overlap -- stop with zero
            # new optimizer updates and report the contamination scope.
            extension_result = {
                "status": "BLOCKED_BY_DATA_OVERLAP",
                "overlap_count": data_manifest["overlap_count"],
                "overlap_validation_indices": data_manifest["overlap_validation_indices"],
            }
            (output_dir / "resume_state_audit.json").write_text(
                json.dumps({"status": "NOT_EXECUTED", "reason": "BLOCKED_BY_DATA_OVERLAP"}),
                encoding="utf-8",
            )
        else:
            extension_result = run_extension_for_all_inits(
                core, eval_bank, op_to_id, config, output_dir, rows
            )
            (output_dir / "resume_state_audit.json").write_text(
                json.dumps(
                    extension_result.get("replay", {}), indent=2, default=str
                ),
                encoding="utf-8",
            )
        if extension_result["status"] == "COMPLETE":
            outcomes = extension_result["outcomes"]
            new_optimizer_updates = sum(o["new_optimizer_updates"] for o in outcomes.values())
            with (output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
                for init_id, outcome in outcomes.items():
                    for ckpt in outcome["checkpoints"]:
                        row_out = {k: v for k, v in ckpt.items() if k != "final_step_extras"}
                        payload = {"init_id": init_id, "arm": REC004H_ARM, **row_out}
                        fh.write(json.dumps(payload, default=str) + "\n")
            with (output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as fh:
                for init_id, outcome in outcomes.items():
                    for r in outcome["lr_trace"]:
                        fh.write(json.dumps({"init_id": init_id, "arm": REC004H_ARM, **r}) + "\n")
            per_length_position_metrics = {
                init_id: next(
                    (
                        c["final_step_extras"]
                        for c in outcomes[init_id]["checkpoints"]
                        if c.get("final_step_extras")
                    ),
                    None,
                )
                for init_id in REC004H_INIT_IDS
            }
            (output_dir / "per_length_position_metrics.json").write_text(
                json.dumps(per_length_position_metrics, indent=2, default=str), encoding="utf-8"
            )
            terminal_floor_status = build_terminal_floor_status(
                outcomes, REC004H_EXISTING_VALIDATION_FLOOR
            )
            (output_dir / "terminal_floor_status.json").write_text(
                json.dumps(terminal_floor_status, indent=2, default=str), encoding="utf-8"
            )
            per_init_terminal_forward: dict[str, dict[str, Any]] = {
                init_id: extras
                for init_id in REC004H_INIT_IDS
                if (extras := per_length_position_metrics.get(init_id)) is not None
            }
            if data_manifest["overlap_count"] > 0 and per_init_terminal_forward:
                overlap_audit = build_data_overlap_impact_audit(
                    data_manifest, per_init_terminal_forward, REC004H_EXISTING_VALIDATION_FLOOR
                )
                (output_dir / "extension_data_overlap_impact_audit.json").write_text(
                    json.dumps(overlap_audit, indent=2, default=str), encoding="utf-8"
                )
        else:
            new_optimizer_updates = 0
    # else: STOP_NO_EXTENSION / EVIDENCE_INSUFFICIENT / ALREADY_AT_FLOOR_AUDIT_ONLY / BLOCKED
    # -> zero new optimizer updates, nothing below this line touches training.

    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_after = mbe._protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_after = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004H_FIXED_CANDIDATE_OPERATIONS
    }
    rec004g_source_files_after = _rec004g_source_file_hashes()
    freeze_audit = {
        "task_id": REC004H_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "fixed_candidate_hashes_before": fixed_candidate_hashes_before,
        "fixed_candidate_hashes_after": fixed_candidate_hashes_after,
        "fixed_candidates_unchanged": fixed_candidate_hashes_before == fixed_candidate_hashes_after,
        "rec004g_source_files_sha256_before": rec004g_source_files_before,
        "rec004g_source_files_sha256_after": rec004g_source_files_after,
        "rec004g_run_001_source_files_unchanged": (
            rec004g_source_files_before == rec004g_source_files_after
        ),
    }
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2, default=str), encoding="utf-8"
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004H_TASK_ID,
        "forbidden_cache_hashes_before": before_hashes,
        "forbidden_cache_hashes_after": after_hashes,
        "forbidden_cache_unchanged": before_hashes == after_hashes,
    }
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    cost_accounting = build_cost_accounting(
        audit_forward_examples, extension_result.get("outcomes") if extension_result else None
    )
    (output_dir / "cost_accounting.json").write_text(
        json.dumps(cost_accounting, indent=2, default=str), encoding="utf-8"
    )

    extension_status = "NOT_EXECUTED"
    if continuation["decision"] == "EXTEND_ALL_FIVE_TO_18000":
        if extension_result and extension_result["status"] == "COMPLETE":
            any_diverged = any(
                o["diverged_at_step"] is not None for o in extension_result["outcomes"].values()
            )
            extension_status = "COMPLETE_WITH_NUMERICAL_FAILURES" if any_diverged else "COMPLETE"
        else:
            # Covers BLOCKED_BY_DATA_OVERLAP and any other non-COMPLETE outcome:
            # the progress rule authorized extension, but it did not complete.
            extension_status = "PARTIAL"

    summary = {
        "task_id": REC004H_TASK_ID,
        "source_task_id": REC004H_SOURCE_TASK_ID,
        "implementation_status": "COMPLETE",
        "source_replay_status": "VERIFIED",
        "late_audit_status": "COMPLETE",
        "continuation_decision": continuation["decision"],
        "continuation_decision_detail": continuation,
        "extension_status": extension_status,
        "new_optimizer_updates": new_optimizer_updates,
        "max_new_optimizer_updates_total": REC004H_MAX_UPDATES_TOTAL,
        "terminal_floor_status": (
            terminal_floor_status["terminal_floor_status"]
            if terminal_floor_status else "NOT_EVALUATED"
        ),
        "freeze_audit_passed": (
            freeze_audit["core_unchanged"]
            and freeze_audit["protected_operations_unchanged"]
            and freeze_audit["fixed_candidates_unchanged"]
            and freeze_audit["rec004g_run_001_source_files_unchanged"]
        ),
        "side_effect_audit_passed": side_effect_audit["forbidden_cache_unchanged"],
        "selected_init": None,
        "selected_intervention": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "rec005_executed": False,
        "total_wall_clock_seconds": time.time() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    _write_report_md(
        output_dir, summary, checkpoint_audits, per_init_progress,
        extension_result, terminal_floor_status,
    )
    _write_next_step_recommendation(output_dir, summary, per_init_progress)

    return summary


def _write_report_md(
    output_dir: Path,
    summary: dict[str, Any],
    checkpoint_audits: dict[str, dict[int, dict[str, Any]]],
    per_init_progress: dict[str, Any],
    extension_result: dict[str, Any] | None,
    terminal_floor_status: dict[str, Any] | None,
) -> None:
    lines = [
        f"# {REC004H_TASK_ID} report",
        "",
        f"continuation_decision: {summary['continuation_decision']}",
        f"extension_status: {summary['extension_status']}",
        f"new_optimizer_updates: {summary['new_optimizer_updates']}",
        f"terminal_floor_status: {summary['terminal_floor_status']}",
        "",
        "## Length-10 audit at step=12000 (per init)",
        "",
        "| init | below_floor@12000 | C_8000 | C_10000 | C_12000 | E_8000 | E_10000 "
        "| E_12000 | EM_PROGRESS | SOFT_PROGRESS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    below_floor = summary["continuation_decision_detail"].get("below_floor_at_12000", {})
    for init_id in REC004H_INIT_IDS:
        p = per_init_progress.get(init_id, {})
        lines.append(
            f"| {init_id} | {below_floor.get(init_id)} | {p.get('C_8000')} | {p.get('C_10000')} | "
            f"{p.get('C_12000')} | {p.get('E_8000')} | {p.get('E_10000')} | {p.get('E_12000')} | "
            f"{p.get('EM_PROGRESS')} | {p.get('SOFT_PROGRESS')} |"
        )
    if extension_result and extension_result.get("status") == "COMPLETE" and terminal_floor_status:
        lines += [
            "", "## Extension to step=18000", "",
            "| init | EM@18000 | clears_floor |", "|---|---|---|",
        ]
        em18 = terminal_floor_status["existing_validation_em_at_18000"]
        clears = terminal_floor_status["clears_floor"]
        for init_id in REC004H_INIT_IDS:
            lines.append(f"| {init_id} | {em18.get(init_id)} | {clears.get(init_id)} |")
    lines += [
        "",
        "selected_init: null",
        "child_bundle: null",
        "rg3_recheck: NOT_EXECUTED",
        "rec005_eligible: false",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_step_recommendation(
    output_dir: Path, summary: dict[str, Any], per_init_progress: dict[str, Any]
) -> None:
    decision = summary["continuation_decision"]
    if decision == "STOP_NO_EXTENSION":
        failing = [
            i for i in REC004H_INIT_IDS
            if per_init_progress.get(i, {}).get("status") == "COMPUTED"
            and not per_init_progress[i]["INIT_PROGRESS_CONFIRMED"]
        ]
        text = (
            f"# Next step recommendation (proposal only -- NOT authorized)\n\n"
            f"continuation_decision was STOP_NO_EXTENSION. Init(s) without confirmed "
            f"late-stage length-10 progress at the 8000/10000/12000 phase-matched "
            f"points: {failing}.\n\n"
            "A possible single next step (not authorized by this task): audit whether "
            "the specific length-10 output positions where these init(s) still err at "
            "step=12000 are stable across the 8000/10000/12000 checkpoints (a diagnostic "
            "read of `position_error_trajectory.json`), before proposing any further "
            "training. This file's existence does not authorize implementing it.\n"
        )
    elif decision == "EXTEND_ALL_FIVE_TO_18000":
        text = (
            f"# Next step recommendation (proposal only -- NOT authorized)\n\n"
            f"continuation_decision was EXTEND_ALL_FIVE_TO_18000 and extension_status "
            f"is {summary['extension_status']}; terminal_floor_status is "
            f"{summary['terminal_floor_status']}. Per this task's own charter, no "
            "candidate is selected here regardless of the step=18000 result. Any "
            "candidate-adoption, independent-query, or REC-005 step requires a "
            "separate, explicit future user instruction.\n"
        )
    else:
        text = (
            f"# Next step recommendation (proposal only -- NOT authorized)\n\n"
            f"continuation_decision was {decision}. No new training was authorized or "
            "run by this task. See late_progress_decision.json for the per-init detail.\n"
        )
    (output_dir / "next_step_recommendation.md").write_text(text, encoding="utf-8")
