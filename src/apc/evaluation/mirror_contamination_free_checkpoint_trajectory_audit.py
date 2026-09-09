"""B-C005REC-004I: Contamination-Free Checkpoint Trajectory Audit &
Early-Stopping Feasibility.

Follows
`docs/CODEX_TASKS_PHASE_B_B2_MIRROR_CONTAMINATION_FREE_CHECKPOINT_TRAJECTORY_AUDIT.md`.
Source: every saved `P/I01-I05` `MIRROR_HALVES` checkpoint at step
6000..18000 (500-step interval) across REC-004D's, REC-004G's, and
REC-004H's own `run_001` trees.

**Zero new optimizer updates.** This task builds one new, disjoint
evaluation set (`clean_selection_validation_v2`), then forward-evaluates
already-saved checkpoints against it -- no training, no new checkpoint, no
selection. It answers a question prior to and separate from REC-004H's
step=18000 floor call: does the trajectory across the whole 6000-18000
window, on a set proven disjoint from every training example and every
other MIRROR_HALVES reference/sealed split this lineage has ever generated,
tell the same story the old `rec004a_budget_validation` set told -- or does
the old set's borderline-pass reading not reproduce?

Per this task's own charter, `selected_init`, `selected_step`, and
`child_bundle` are fixed `None`/`null` and `rg3_recheck` is fixed
`"NOT_EXECUTED"` regardless of outcome. `B-C005REC-005` (the separate,
still-blocked five-model cohort task) is not touched or consumed.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
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
from apc.evaluation import mirror_budget_extension as mbe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_schedule_comparison as msc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,  # noqa: F401 -- re-exported for tests, not used on the hot path
    _snapshot_forbidden_cache_hashes,
    frozen_evaluation,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    _derive_local_seed,
    _generate_parameter_free_examples,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004I_TASK_ID",
    "REC004I_SOURCE_TASK_IDS",
    "REC004I_INIT_IDS",
    "REC004I_ARM",
    "REC004I_TARGET_OPERATION",
    "REC004I_CHECKPOINT_STEPS",
    "REC004I_TARGET_LENGTHS",
    "REC004I_CLEAN_V2_SPLIT",
    "REC004I_CLEAN_V2_EXAMPLES",
    "REC004I_FLOOR",
    "MirrorContaminationFreeCheckpointTrajectoryAuditConfig",
    "run_mirror_contamination_free_checkpoint_trajectory_audit_task",
]

# =============================================================================
# Constants -- reused from REC-004D/REC-004G/REC-004H wherever the recipe is
# unchanged, never retyped, so this module fails to import (rather than
# silently diverging) if any upstream recipe constant ever changes.
# =============================================================================

REC004I_TASK_ID: Final = "B-C005REC-004I"
REC004I_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004D", "B-C005REC-004G", "B-C005REC-004H",
)
REC004I_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_MIRROR_CONTAMINATION_FREE_CHECKPOINT_TRAJECTORY_AUDIT.md"
)

REC004I_INIT_IDS: Final[tuple[str, ...]] = mbe.REC004G_INIT_IDS  # ("I01", ..., "I05")
REC004I_ARM: Final = mbe.REC004G_ARM  # "P_LENGTH_POSITION_BIAS"
REC004I_TARGET_OPERATION: Final = mbe.REC004G_TARGET_OPERATION  # "MIRROR_HALVES"
REC004I_TARGET_PHYSICAL_ID: Final = mbe.REC004G_TARGET_PHYSICAL_ID

REC004I_SOURCE_RUN_DIRS: Final[dict[str, Path]] = {
    "B-C005REC-004D": Path("runs/phase_b_b2_model_bundle_recovery/rec004d/run_001"),
    "B-C005REC-004G": Path("runs/phase_b_b2_model_bundle_recovery/rec004g/run_001"),
    "B-C005REC-004H": Path("runs/phase_b_b2_model_bundle_recovery/rec004h/run_001"),
}

REC004I_CHECKPOINT_STEPS: Final[tuple[int, ...]] = tuple(range(6000, 18000 + 1, 500))  # 25 points
REC004I_TARGET_LENGTHS: Final[tuple[int, ...]] = (6, 7, 8, 9, 10)
REC004I_TARGET_LENGTH_10: Final = 10

REC004I_EXISTING_VALIDATION_SPLIT: Final = mbe.REC004G_EXISTING_VALIDATION_SPLIT
REC004I_EXISTING_VALIDATION_EXAMPLES: Final = mbe.REC004G_EXISTING_VALIDATION_EXAMPLES  # 1024
REC004I_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE
REC004I_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE
REC004I_FLOOR: Final = mbe.REC004G_EXISTING_VALIDATION_FLOOR  # 0.95
REC004I_FLOOR_EPS: Final = 1e-6  # numerical comparison margin, NOT a significance level

REC004I_CLEAN_V2_SPLIT: Final = "clean_selection_validation_v2"
REC004I_CLEAN_V2_EXAMPLES: Final = 1024  # matches existing_validation's own size

REC004I_TRAIN_STREAM_FIRST_STEP: Final = 1
REC004I_TRAIN_STREAM_LAST_STEP: Final = 18000  # every step any init was ever updated on

REC004I_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = mbe.REC004G_PROTECTED_OPERATIONS
REC004I_FIXED_CANDIDATE_OPERATIONS: Final[tuple[str, ...]] = mbe.REC004G_FIXED_CANDIDATE_OPERATIONS

REC004I_MAX_PREDICTION_EXAMPLES_BUDGET: Final = (
    len(REC004I_INIT_IDS) * len(REC004I_CHECKPOINT_STEPS) * REC004I_CLEAN_V2_EXAMPLES
)  # 5 * 25 * 1024 = 128000, exact -- no other forward pass is run by this task


def _source_for_step(step: int) -> str:
    if step == 6000:
        return "B-C005REC-004D"
    if 6500 <= step <= 12000:
        return "B-C005REC-004G"
    if 12500 <= step <= 18000:
        return "B-C005REC-004H"
    raise ValueError(f"step={step} is outside the pre-registered 6000-18000 audit window")


def _checkpoint_path(init_id: str, step: int) -> Path:
    source = _source_for_step(step)
    run_dir = REC004I_SOURCE_RUN_DIRS[source]
    return run_dir / init_id / REC004I_ARM / "checkpoints" / f"step{step}.pt"


def _learning_curve_path(source: str) -> Path:
    return REC004I_SOURCE_RUN_DIRS[source] / "learning_curve.jsonl"


def _load_learning_curve_rows(source: str) -> list[dict[str, Any]]:
    path = _learning_curve_path(source)
    if not path.is_file():
        raise mb.MissingArtifactError(f"SOURCE_ARTIFACT_UNAVAILABLE: {path} not found")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _old_dev_metric_row(
    init_id: str, step: int, source_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    source = _source_for_step(step)
    row = next(
        (
            r for r in source_rows[source]
            if r["init_id"] == init_id and r["arm"] == REC004I_ARM and r["step"] == step
        ),
        None,
    )
    if row is None:
        return None
    by_length = row.get("by_length")
    if by_length is None:
        stratified = row.get("length_stratified")
        by_length = stratified.get("by_length") if stratified else None
    return {
        "source_task": source,
        "existing_validation_em": row["existing_validation"]["correct_exact_match"],
        "by_length": by_length,
        "checkpoint_state_hash": row.get("checkpoint_state_hash"),
    }


def _source_file_hashes() -> dict[str, str]:
    """Raw-byte hashes of every REC-004D/G/H `run_001` file this task reads --
    proof this task never mutates its read-only sources."""
    hashes: dict[str, str] = {}
    for source, run_dir in REC004I_SOURCE_RUN_DIRS.items():
        curve_path = run_dir / "learning_curve.jsonl"
        if curve_path.is_file():
            hashes[f"{source}/learning_curve.jsonl"] = mb.raw_file_sha256(curve_path)
    for init_id in REC004I_INIT_IDS:
        for step in REC004I_CHECKPOINT_STEPS:
            path = _checkpoint_path(init_id, step)
            if path.is_file():
                source = _source_for_step(step)
                key = f"{source}/{init_id}/checkpoints/step{step}.pt"
                hashes[key] = mb.raw_file_sha256(path)
    return hashes


def _existing_validation_examples(seed: int) -> list[Any]:
    return _generate_parameter_free_examples(
        seed,
        REC004I_EXISTING_VALIDATION_EXAMPLES,
        operation=REC004I_TARGET_OPERATION,
        split=REC004I_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004I_VOCAB_SIZE,
        sequence_length_range=REC004I_SEQUENCE_LENGTH_RANGE,
    )


@dataclass(frozen=True)
class MirrorContaminationFreeCheckpointTrajectoryAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004i")
    seed: int = RECOVERY_PILOT_SEED


def _config_to_yaml_dict(
    config: MirrorContaminationFreeCheckpointTrajectoryAuditConfig,
) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


# =============================================================================
# Digests and the protected (training-stream + reference/sealed) registry.
# Pure data generation -- no Core, no torch, no `_guard_not_frozen` concern.
# =============================================================================


def _digest(input_tokens: Any, target_tokens: Any) -> str:
    payload = [list(input_tokens), list(target_tokens)]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def _digest_example(ex: Any) -> str:
    return _digest(ex.input_tokens, ex.target_tokens)


def _digest_examples(examples: list[Any]) -> set[str]:
    return {_digest_example(ex) for ex in examples}


def build_protected_digest_registry(seed: int) -> tuple[set[str], dict[str, int]]:
    """Every `(input_tokens, target_tokens)` digest `clean_selection_validation_v2`
    must not collide with: the full training stream (steps 1-18000, the exact
    formula every init was actually updated on), `rec004a_budget_validation`
    (existing validation), and every other MIRROR_HALVES reference/sealed
    split this recovery lineage has ever generated (REC-004A/B/C/D recheck
    and diagnostic splits). Counts are per-source so the manifest states
    exactly what was protected against, not just a total."""
    counts: dict[str, int] = {}
    protected: set[str] = set()

    train_digests: set[str] = set()
    for step in range(REC004I_TRAIN_STREAM_FIRST_STEP, REC004I_TRAIN_STREAM_LAST_STEP + 1):
        for ex in ibc._generate_step_training_examples(
            seed, step, REC004I_TARGET_OPERATION,
            vocab_size=REC004I_VOCAB_SIZE, sequence_length_range=REC004I_SEQUENCE_LENGTH_RANGE,
        ):
            train_digests.add(_digest_example(ex))
    counts["training_stream_steps_1_to_18000"] = len(train_digests)
    protected |= train_digests

    existing_val_digests = _digest_examples(_existing_validation_examples(seed))
    counts["rec004a_budget_validation"] = len(existing_val_digests)
    protected |= existing_val_digests

    recheck_splits = (
        (
            "rec004a_recheck_query", ibc.REC004A_RECHECK_QUERY_SPLIT,
            ibc.REC004A_RECHECK_QUERY_EXAMPLES,
        ),
        (
            "rec004b_recheck_query", msc.REC004B_RECHECK_QUERY_SPLIT,
            ibc.REC004A_RECHECK_QUERY_EXAMPLES,
        ),
        (
            "rec004d_recheck_query", mpbr.REC004D_RECHECK_QUERY_SPLIT,
            mpbr.REC004D_RECHECK_QUERY_EXAMPLES,
        ),
    )
    for label, split, n in recheck_splits:
        digests = _digest_examples(
            _generate_parameter_free_examples(
                seed, n, operation=REC004I_TARGET_OPERATION, split=split,
                vocab_size=REC004I_VOCAB_SIZE, sequence_length_range=REC004I_SEQUENCE_LENGTH_RANGE,
            )
        )
        counts[label] = len(digests)
        protected |= digests

    length_balanced = mpid._generate_length_balanced_diagnostic()
    digests = _digest_examples(length_balanced)
    counts["rec004c_length_balanced_diagnostic"] = len(digests)
    protected |= digests

    position_identifiable = mpid._generate_position_identifiable_diagnostic()
    pi_flat = [ex for lst in position_identifiable.values() if lst for ex in lst]
    digests = _digest_examples(pi_flat)
    counts["rec004c_position_identifiable_diagnostic"] = len(digests)
    protected |= digests

    counterfactual = mpid._generate_content_counterfactual_diagnostic()
    cf_flat: list[Any] = []
    for bases in counterfactual.values():
        for b in bases:
            cf_flat.append(b["base_example"])
            cf_flat.extend(v["example"] for v in b["variants"])
    digests = _digest_examples(cf_flat)
    counts["rec004c_content_counterfactual_diagnostic"] = len(digests)
    protected |= digests

    padding_batch = mpid._generate_padding_batch_sample()
    digests = _digest_examples(padding_batch)
    counts["rec004c_padding_batch_sample"] = len(digests)
    protected |= digests

    return protected, counts


def build_clean_selection_validation_v2(
    seed: int, protected_digests: set[str], n: int = REC004I_CLEAN_V2_EXAMPLES
) -> tuple[list[Any], dict[str, Any]]:
    """Deterministically generates `n` MIRROR_HALVES examples disjoint from
    `protected_digests`. On a collision, does NOT reset or resample from a
    fresh draw -- it keeps drawing the NEXT candidate from the SAME
    continuing RNG stream until a non-colliding one appears, and uses that
    one in the colliding slot. This is a pure function of
    `(seed, protected_digests)`: it never inspects a checkpoint prediction
    before or during substitution."""
    seed_label = f"{REC004I_CLEAN_V2_SPLIT}:{REC004I_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed(seed, 0, seed_label))
    op_obj = get_operation(REC004I_TARGET_OPERATION)
    examples: list[Any] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq_len = rng.randint(*REC004I_SEQUENCE_LENGTH_RANGE)
            seq = tuple(rng.randrange(REC004I_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(rng, seq, REC004I_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004I_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004I_VOCAB_SIZE)
            digest = _digest(seq, res.output_tokens)
            if digest not in protected_digests:
                break
            rejected_digests.append(digest)
        if rejected_digests:
            substitutions.append(
                {"slot": slot, "rejected_digests": rejected_digests, "accepted_digest": digest}
            )
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=REC004I_CLEAN_V2_SPLIT,
                vocab_size=REC004I_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004I_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
    }
    return examples, detail


# =============================================================================
# Forward-only checkpoint evaluation, all of REC004I_TARGET_LENGTHS at once
# (unlike REC-004H, which only needed length-10 token statistics).
# =============================================================================


def _run_full_checkpoint_forward_all_lengths(
    core: Any, primitive: Any, examples: list[Any], device: torch.device
) -> dict[str, Any]:
    primitive.eval()
    by_length: dict[int, dict[str, Any]] = {
        n: {"n": 0, "exact": 0, "valid_token_count": 0, "token_ce_sum": 0.0}
        for n in REC004I_TARGET_LENGTHS
    }
    sequence_correct_all = 0
    overall_valid_token_count = 0
    overall_token_ce_sum = 0.0

    with torch.no_grad():
        for start in range(0, len(examples), 128):
            chunk = examples[start : start + 128]
            content_lengths = [len(ex.input_tokens) for ex in chunk]
            output_lengths = [
                get_operation(REC004I_TARGET_OPERATION).output_length(n) for n in content_lengths
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

            for row, ex in enumerate(chunk):
                n = output_lengths[row]
                pred = tuple(preds[row, :n].tolist())
                target = ex.target_tokens
                length = len(ex.input_tokens)
                is_exact = bool(len(pred) == len(target) and pred == target)
                sequence_correct_all += int(is_exact)
                vtok = int(valid_mask[row, :n].sum().item())
                ce_sum = float(token_ce[row, :n][valid_mask[row, :n]].sum().item())
                overall_valid_token_count += vtok
                overall_token_ce_sum += ce_sum
                bucket = by_length.get(length)
                if bucket is not None:
                    bucket["n"] += 1
                    bucket["exact"] += int(is_exact)
                    bucket["valid_token_count"] += vtok
                    bucket["token_ce_sum"] += ce_sum

    n_all = len(examples)
    by_length_out: dict[str, dict[str, Any]] = {}
    for length, bucket in sorted(by_length.items()):
        em = bucket["exact"] / bucket["n"] if bucket["n"] else None
        mean_ce = (
            bucket["token_ce_sum"] / bucket["valid_token_count"]
            if bucket["valid_token_count"] > 0 else None
        )
        by_length_out[str(length)] = {
            "n": bucket["n"], "exact": bucket["exact"],
            "exact_match": em, "mean_valid_token_CE": mean_ce,
        }

    len10 = by_length_out.get(str(REC004I_TARGET_LENGTH_10))
    return {
        "n_examples": n_all,
        "sequence_correct_all": sequence_correct_all,
        "overall_exact_match": sequence_correct_all / n_all if n_all else None,
        "overall_mean_valid_token_CE": (
            overall_token_ce_sum / overall_valid_token_count
            if overall_valid_token_count > 0 else None
        ),
        "by_length": by_length_out,
        "length10_sequence_correct": len10["exact"] if len10 else None,
    }


def run_checkpoint_trajectory_point(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    step: int,
    source_rows: dict[str, list[dict[str, Any]]],
    clean_v2_examples: list[Any],
) -> dict[str, Any]:
    """Forward-only: loads one already-saved checkpoint (never mutated),
    hash-verifies it against its own source's recorded
    `checkpoint_state_hash`, and evaluates it against
    `clean_selection_validation_v2`. Reads (but never trains) -- safe inside
    `frozen_evaluation()`."""
    path = _checkpoint_path(init_id, step)
    source = _source_for_step(step)
    if not path.is_file():
        return {
            "status": "SOURCE_ARTIFACT_UNAVAILABLE", "init_id": init_id, "step": step,
            "source_task": source, "path": str(path),
        }
    old_row = _old_dev_metric_row(init_id, step, source_rows)
    if old_row is None:
        return {
            "status": "SOURCE_ARTIFACT_UNAVAILABLE", "init_id": init_id, "step": step,
            "source_task": source, "reason": "MISSING_LEARNING_CURVE_ROW",
        }

    state = torch.load(path, map_location="cpu", weights_only=False)
    computed_hash = mb.canonical_state_hash(state)
    hash_matches = computed_hash == old_row["checkpoint_state_hash"]

    device = core.device
    pid = REC004I_TARGET_PHYSICAL_ID
    primitive = mpbr._new_arm_primitive(core, REC004I_ARM)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    primitive.eval()
    original_slot = eval_bank.replace_primitive(pid, primitive)
    try:
        clean_v2_forward = _run_full_checkpoint_forward_all_lengths(
            core, primitive, clean_v2_examples, device
        )
    finally:
        eval_bank.replace_primitive(pid, original_slot)

    return {
        "status": "VERIFIED" if hash_matches else "SOURCE_REPLAY_MISMATCH",
        "init_id": init_id, "step": step, "source_task": source,
        "raw_file_sha256": mb.raw_file_sha256(path),
        "checkpoint_state_hash": computed_hash,
        "recorded_checkpoint_state_hash": old_row["checkpoint_state_hash"],
        "clean_v2": clean_v2_forward,
        "old_development_metric": {
            "existing_validation_em": old_row["existing_validation_em"],
            "by_length": old_row["by_length"],
        },
    }


# =============================================================================
# Trajectory decomposition -- pure arithmetic on already-collected points.
# =============================================================================


def compute_trajectory_derived_stats(
    points: list[tuple[int, float | None]],
    floor: float = REC004I_FLOOR,
    eps: float = REC004I_FLOOR_EPS,
) -> dict[str, Any]:
    """`points` is `[(step, em_or_None), ...]`, any order. Reports the first
    recorded step at/above `floor`, how many consecutive recorded steps
    (starting there) stayed at/above it before the first drop, the trajectory
    peak, and the regression from that peak to the final recorded point."""
    valid = sorted(((s, e) for s, e in points if e is not None), key=lambda x: x[0])
    if not valid:
        return {"status": "EVIDENCE_INSUFFICIENT"}

    first_cross = next((s for s, e in valid if e >= floor - eps), None)
    duration = 0
    if first_cross is not None:
        for s, e in valid:
            if s < first_cross:
                continue
            if e >= floor - eps:
                duration += 1
            else:
                break

    peak_step, peak_value = max(valid, key=lambda x: x[1])
    final_step, final_value = valid[-1]
    recovered_to_peak_after_regressing = False
    if peak_step != final_step:
        recovered_to_peak_after_regressing = any(
            e >= peak_value - eps for s, e in valid if s > peak_step
        )

    return {
        "status": "OK",
        "n_points": len(valid),
        "first_step_at_or_above_floor": first_cross,
        "duration_at_or_above_floor_from_first_cross": duration,
        "ever_reaches_floor": first_cross is not None,
        "peak_step": peak_step,
        "peak_value": peak_value,
        "final_step": final_step,
        "final_value": final_value,
        "regression_from_peak_to_final": peak_value - final_value,
        "recovered_to_peak_after_regressing": recovered_to_peak_after_regressing,
    }


def classify_pattern(
    old_pass_inits: set[str], clean_pass_inits: set[str], all_inits: tuple[str, ...]
) -> dict[str, Any]:
    """The task's own three-way outcome classification (S8 of the task doc).
    `OLD_PASSES_LARGELY_INVALIDATED` is checked first because it is a
    protocol-validity concern that would undercut trusting either of the
    other two readings."""
    invalidated = sorted(old_pass_inits - clean_pass_inits)
    invalidation_fraction = (
        len(invalidated) / len(old_pass_inits) if old_pass_inits else None
    )
    never_clear = sorted(set(all_inits) - clean_pass_inits)

    if old_pass_inits and invalidation_fraction is not None and invalidation_fraction >= 0.5:
        classification = "OLD_PASSES_LARGELY_INVALIDATED"
    elif not never_clear:
        classification = "ALL_FIVE_CLEAR_CLEAN_V2"
    else:
        classification = "SOME_INITS_NEVER_CLEAR_CLEAN_V2"

    return {
        "classification": classification,
        "old_development_pass_inits": sorted(old_pass_inits),
        "clean_v2_pass_inits": sorted(clean_pass_inits),
        "invalidated_inits": invalidated,
        "invalidation_fraction_of_old_passes": invalidation_fraction,
        "never_clear_clean_v2_inits": never_clear,
    }


def _proposed_next_step_text(classification: str) -> str:
    if classification == "ALL_FIVE_CLEAR_CLEAN_V2":
        return (
            "Define a pre-registered early-stopping policy and validate it by training "
            "fresh, independent reinitializations under that SAME policy applied "
            "identically to all five -- not a best-of-five pick from checkpoints "
            "already on disk."
        )
    if classification == "SOME_INITS_NEVER_CLEAR_CLEAN_V2":
        return (
            "Scope the next task to a mechanism-level fix for the specific init(s)/"
            "length-10 failure that never clears the clean-v2 floor anywhere in the "
            "trajectory -- do not extend the training budget for every init again."
        )
    return (
        "Redesign the development-evaluation / checkpoint-selection protocol itself "
        "before any B-C005REC-005 candidate-adoption work; treat every existing "
        "old-validation pass in this lineage as unconfirmed until re-checked on a set "
        "proven disjoint from training."
    )


def build_cost_accounting(
    training_stream_examples_regenerated: int, clean_v2_forward_examples: int
) -> dict[str, Any]:
    return {
        "new_optimizer_updates": 0,
        "training_stream_examples_regenerated_for_digest_check": (
            training_stream_examples_regenerated
        ),
        "clean_v2_forward_prediction_examples": clean_v2_forward_examples,
        "max_prediction_examples_budget": REC004I_MAX_PREDICTION_EXAMPLES_BUDGET,
        "within_budget": clean_v2_forward_examples <= REC004I_MAX_PREDICTION_EXAMPLES_BUDGET,
    }


# =============================================================================
# Report writers
# =============================================================================


def _write_report_md(
    output_dir: Path, summary: dict[str, Any], pattern: dict[str, Any],
    decomposition: dict[str, Any],
) -> None:
    lines = [
        f"# {REC004I_TASK_ID} -- Contamination-Free Checkpoint Trajectory Audit",
        "",
        f"implementation_status: {summary['implementation_status']}",
        f"new_optimizer_updates: {summary['new_optimizer_updates']}",
        f"checkpoints_evaluated: {summary['checkpoints_evaluated']}",
        f"clean_selection_validation_v2 substitutions: "
        f"{summary['clean_selection_validation_v2_substitutions']}",
        "",
        f"## Pattern classification: {pattern['classification']}",
        f"old_development_pass_inits: {pattern['old_development_pass_inits']}",
        f"clean_v2_pass_inits: {pattern['clean_v2_pass_inits']}",
        f"invalidated_inits: {pattern['invalidated_inits']}",
        f"never_clear_clean_v2_inits: {pattern['never_clear_clean_v2_inits']}",
        f"proposed next step (NOT executed): {pattern['proposed_next_step']}",
        "",
        "## Per-init trajectory decomposition (clean_v2 vs old_development)",
    ]
    for init_id in REC004I_INIT_IDS:
        d = decomposition.get(init_id, {})
        lines.append(
            f"- {init_id}: clean_v2={d.get('clean_v2')} old_development={d.get('old_development')}"
        )
    lines += [
        "",
        f"selected_init={summary['selected_init']} selected_step={summary['selected_step']} "
        f"child_bundle={summary['child_bundle']} rg3_recheck={summary['rg3_recheck']}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_step_recommendation(output_dir: Path, pattern: dict[str, Any]) -> None:
    text = (
        f"# {REC004I_TASK_ID} next-step proposal\n\n"
        f"status: PROPOSED_NOT_AUTHORIZED\n"
        f"classification: {pattern['classification']}\n\n"
        f"{pattern['proposed_next_step']}\n\n"
        "This file's existence is not authorization to implement it. Per this task's "
        "own charter, B-C005REC-005, R3-011/012, B-C006, and Task Inference remain "
        "unexecuted pending an explicit next user instruction.\n"
    )
    (output_dir / "next_step_recommendation.md").write_text(text, encoding="utf-8")


# =============================================================================
# Orchestrator
# =============================================================================


def run_mirror_contamination_free_checkpoint_trajectory_audit_task(
    config: MirrorContaminationFreeCheckpointTrajectoryAuditConfig,
) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004I reads REC-004D/G/H artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; got seed={seed}"
        )

    def _blocked(reason: str) -> dict[str, Any]:
        blocked_summary = {
            "task_id": REC004I_TASK_ID,
            "implementation_status": "BLOCKED",
            "reason": reason,
            "new_optimizer_updates": 0,
            "selected_init": None, "selected_step": None, "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED", "rec005_eligible": False, "rec005_executed": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(blocked_summary, indent=2, default=str), encoding="utf-8"
        )
        return blocked_summary

    if not REC004I_CONTRACT_FILE.is_file():
        return _blocked(f"AUTHORIZATION_ARTIFACT_MISSING: {REC004I_CONTRACT_FILE} not found")
    missing_sources = [s for s, d in REC004I_SOURCE_RUN_DIRS.items() if not d.is_dir()]
    if missing_sources:
        return _blocked(
            f"SOURCE_ARTIFACT_UNAVAILABLE: missing run directories for {missing_sources}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)
    source_files_before = _source_file_hashes()

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )

    # --- Stage 1: lock clean_selection_validation_v2 BEFORE any checkpoint
    # is touched. Pure data generation -- no Core, no `_guard_not_frozen`
    # concern, so this can (and must) happen before runtime reconstruction.
    protected_digests, protected_counts = build_protected_digest_registry(seed)
    clean_v2_examples, clean_v2_detail = build_clean_selection_validation_v2(
        seed, protected_digests
    )
    clean_v2_digests = _digest_examples(clean_v2_examples)
    if clean_v2_digests & protected_digests:
        raise AssertionError(
            "CONTAMINATION_GUARANTEE_VIOLATED: clean_selection_validation_v2 collides "
            "with a protected digest despite the substitution loop"
        )
    clean_v2_manifest = {
        "task_id": REC004I_TASK_ID,
        "split": REC004I_CLEAN_V2_SPLIT,
        "seed": seed,
        "protected_digest_set_size": len(protected_digests),
        "protected_source_counts": protected_counts,
        **clean_v2_detail,
        "disjoint_from_protected_set_confirmed": True,
    }
    (output_dir / "clean_selection_validation_v2_manifest.json").write_text(
        json.dumps(clean_v2_manifest, indent=2, default=str), encoding="utf-8"
    )

    # --- Stage 2: reconstruct the frozen Core/bank (guarded; must happen
    # outside any `frozen_evaluation()` block).
    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mbe._protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_before = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004I_FIXED_CANDIDATE_OPERATIONS
    }

    source_rows = {source: _load_learning_curve_rows(source) for source in REC004I_SOURCE_RUN_DIRS}

    # --- Stage 3: forward-only checkpoint trajectory audit, wrapped in
    # frozen_evaluation() so no build/train call can silently execute here.
    trajectory: dict[str, dict[int, dict[str, Any]]] = {i: {} for i in REC004I_INIT_IDS}
    with frozen_evaluation():
        for init_id in REC004I_INIT_IDS:
            for step in REC004I_CHECKPOINT_STEPS:
                trajectory[init_id][step] = run_checkpoint_trajectory_point(
                    core, eval_bank, op_to_id, init_id, step, source_rows, clean_v2_examples
                )
    # --- end frozen_evaluation(): this task never trains, so nothing below
    # this line touches the frozen surface either.

    with (output_dir / "checkpoint_trajectory.jsonl").open("w", encoding="utf-8") as fh:
        for init_id in REC004I_INIT_IDS:
            for step in REC004I_CHECKPOINT_STEPS:
                fh.write(json.dumps(trajectory[init_id][step], default=str) + "\n")

    replay_mismatched = [
        (i, s) for i in REC004I_INIT_IDS for s in REC004I_CHECKPOINT_STEPS
        if trajectory[i][s]["status"] == "SOURCE_REPLAY_MISMATCH"
    ]
    replay_unavailable = [
        (i, s) for i in REC004I_INIT_IDS for s in REC004I_CHECKPOINT_STEPS
        if trajectory[i][s]["status"] == "SOURCE_ARTIFACT_UNAVAILABLE"
    ]

    # --- Stage 4: trajectory decomposition + old-vs-new comparison + pattern.
    decomposition: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    old_pass_inits: set[str] = set()
    clean_pass_inits: set[str] = set()
    for init_id in REC004I_INIT_IDS:
        verified_steps = [
            s for s in REC004I_CHECKPOINT_STEPS if trajectory[init_id][s]["status"] == "VERIFIED"
        ]
        clean_points = [
            (s, trajectory[init_id][s]["clean_v2"]["overall_exact_match"]) for s in verified_steps
        ]
        old_points = [
            (s, trajectory[init_id][s]["old_development_metric"]["existing_validation_em"])
            for s in verified_steps
        ]
        clean_stats = compute_trajectory_derived_stats(clean_points)
        old_stats = compute_trajectory_derived_stats(old_points)
        decomposition[init_id] = {"clean_v2": clean_stats, "old_development": old_stats}

        if any(e is not None and e >= REC004I_FLOOR - REC004I_FLOOR_EPS for _, e in old_points):
            old_pass_inits.add(init_id)
        if any(e is not None and e >= REC004I_FLOOR - REC004I_FLOOR_EPS for _, e in clean_points):
            clean_pass_inits.add(init_id)

        for s in verified_steps:
            row = trajectory[init_id][s]
            comparison_rows.append(
                {
                    "init_id": init_id, "step": s,
                    "old_development_em": row["old_development_metric"]["existing_validation_em"],
                    "clean_v2_em": row["clean_v2"]["overall_exact_match"],
                    "clean_v2_length10_correct": row["clean_v2"]["length10_sequence_correct"],
                }
            )

    (output_dir / "trajectory_decomposition.json").write_text(
        json.dumps(decomposition, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "old_vs_clean_v2_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2, default=str), encoding="utf-8"
    )

    pattern = classify_pattern(old_pass_inits, clean_pass_inits, REC004I_INIT_IDS)
    pattern["proposed_next_step"] = _proposed_next_step_text(pattern["classification"])
    pattern["status"] = "PROPOSED_NOT_AUTHORIZED"
    (output_dir / "pattern_classification.json").write_text(
        json.dumps(pattern, indent=2, default=str), encoding="utf-8"
    )

    # --- Stage 5: freeze / side-effect audit.
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_after = mbe._protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_after = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004I_FIXED_CANDIDATE_OPERATIONS
    }
    source_files_after = _source_file_hashes()
    freeze_audit = {
        "task_id": REC004I_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "fixed_candidate_hashes_before": fixed_candidate_hashes_before,
        "fixed_candidate_hashes_after": fixed_candidate_hashes_after,
        "fixed_candidates_unchanged": fixed_candidate_hashes_before == fixed_candidate_hashes_after,
        "source_files_sha256_before": source_files_before,
        "source_files_sha256_after": source_files_after,
        "source_files_unchanged": source_files_before == source_files_after,
    }
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2, default=str), encoding="utf-8"
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004I_TASK_ID,
        "forbidden_cache_hashes_before": before_hashes,
        "forbidden_cache_hashes_after": after_hashes,
        "forbidden_cache_unchanged": before_hashes == after_hashes,
    }
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    cost_accounting = build_cost_accounting(
        training_stream_examples_regenerated=(
            protected_counts["training_stream_steps_1_to_18000"]
        ),
        clean_v2_forward_examples=(
            len(REC004I_INIT_IDS) * len(REC004I_CHECKPOINT_STEPS) * len(clean_v2_examples)
        ),
    )
    (output_dir / "cost_accounting.json").write_text(
        json.dumps(cost_accounting, indent=2, default=str), encoding="utf-8"
    )

    freeze_ok = (
        freeze_audit["core_unchanged"]
        and freeze_audit["protected_operations_unchanged"]
        and freeze_audit["fixed_candidates_unchanged"]
        and freeze_audit["source_files_unchanged"]
    )
    summary = {
        "task_id": REC004I_TASK_ID,
        "source_task_ids": list(REC004I_SOURCE_TASK_IDS),
        "implementation_status": "COMPLETE",
        "source_replay_status": (
            "VERIFIED" if not replay_mismatched and not replay_unavailable else "PARTIAL"
        ),
        "checkpoints_evaluated": len(REC004I_INIT_IDS) * len(REC004I_CHECKPOINT_STEPS),
        "checkpoints_source_replay_mismatched": [f"{i}:{s}" for i, s in replay_mismatched],
        "checkpoints_source_artifact_unavailable": [f"{i}:{s}" for i, s in replay_unavailable],
        "new_optimizer_updates": 0,
        "clean_selection_validation_v2_n": len(clean_v2_examples),
        "clean_selection_validation_v2_substitutions": clean_v2_detail["substitution_count"],
        "pattern_classification": pattern["classification"],
        "selected_init": None, "selected_step": None, "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED", "rec005_eligible": False, "rec005_executed": False,
        "freeze_audit_passed": freeze_ok,
        "side_effect_audit_passed": side_effect_audit["forbidden_cache_unchanged"],
        "total_wall_clock_seconds": time.time() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    _write_report_md(output_dir, summary, pattern, decomposition)
    _write_next_step_recommendation(output_dir, pattern)

    return summary
