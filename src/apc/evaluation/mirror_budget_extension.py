"""B-C005REC-004G: MIRROR_HALVES P Cumulative Budget Extension (6000 -> 12000).

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_BUDGET_EXTENSION_12000.md`, a task the
user authorized directly (not a pre-supplied, separately-authorized contract file
like REC-004E/F): resume each of REC-004D's 5 saved `P_LENGTH_POSITION_BIAS`
(arm P) runs (I01-I05) from their saved COMPLETE step=6000 training state
(primitive weights + AdamW optimizer state + `CosineAnnealingLR` scheduler state +
CPU/CUDA RNG state) and continue training for 6000 MORE updates each (30000 new
optimizer updates total), under the exact same recipe REC-004D used -- same LR,
weight decay, grad clip, `CosineAnnealingLR(T_max=1000, eta_min=...)` mechanically
continued (never reset), same per-step training-data generator/seed continued from
step 6001 onward, same checkpoint interval (500), same existing-validation split
and 0.95 floor.

The ONLY thing this task changes is the cumulative training budget (6000 -> 12000).
Architecture, Core, other primitives, optimizer settings, schedule rule, and
training distribution are all unchanged -- verified by importing REC-004D's own
constants and helper functions rather than retyping them. Arm U is not retrained
(the comparison is each P against its OWN step=6000 state, not P against U).

Zero use of REC-004F's oracle-attention substitution anywhere in this module
(training uses the model's own real forward, exactly as REC-004D's `run_one_arm`
did). Per this task's own charter, even if all 5 inits clear the 0.95
existing-validation floor at step=12000, this task does NOT select a candidate,
build a child bundle, or run an RG3 recheck -- those remain gated behind a
separate, future, explicit user instruction. `selected_init`, `selected_intervention`,
and `child_bundle` are fixed `None`/`null` here; `rg3_recheck` is fixed
`"NOT_EXECUTED"`.
"""

from __future__ import annotations

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
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _predict_tokens,
    _snapshot_forbidden_cache_hashes,
)
from apc.evaluation.unified_oracle_causal_benchmark import _generate_parameter_free_examples
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004G_TASK_ID",
    "REC004G_SOURCE_TASK_ID",
    "REC004G_INIT_IDS",
    "REC004G_ARM",
    "REC004G_SOURCE_STEP",
    "REC004G_ADDITIONAL_UPDATES",
    "REC004G_TARGET_CUMULATIVE_STEP",
    "REC004G_CHECKPOINT_INTERVAL",
    "REC004G_EXISTING_VALIDATION_FLOOR",
    "MirrorBudgetExtensionConfig",
    "run_mirror_budget_extension_task",
]

# =============================================================================
# Constants -- reused from REC-004D wherever the recipe is unchanged, never
# retyped, so any future drift in REC-004D's own recipe constants is
# automatically inherited (or, if REC-004D's module changes incompatibly, this
# module fails to import rather than silently diverging).
# =============================================================================

REC004G_TASK_ID: Final = "B-C005REC-004G"
REC004G_SOURCE_TASK_ID: Final = "B-C005REC-004D"
REC004G_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_MIRROR_BUDGET_EXTENSION_12000.md"
)

REC004G_INIT_IDS: Final[tuple[str, ...]] = mpbr.REC004D_INIT_IDS  # ("I01", ..., "I05")
REC004G_ARM: Final = mpbr.REC004D_ARM_P  # "P_LENGTH_POSITION_BIAS" -- U is not retrained
REC004G_TARGET_OPERATION: Final = mpbr.REC004D_TARGET_OPERATION  # "MIRROR_HALVES"
REC004G_TARGET_PHYSICAL_ID: Final = mpbr.REC004D_TARGET_PHYSICAL_ID

REC004G_SOURCE_STEP: Final = 6000
REC004G_ADDITIONAL_UPDATES: Final = 6000
REC004G_TARGET_CUMULATIVE_STEP: Final = REC004G_SOURCE_STEP + REC004G_ADDITIONAL_UPDATES  # 12000
REC004G_CHECKPOINT_INTERVAL: Final = mpbr.REC004D_CHECKPOINT_INTERVAL  # 500

REC004G_OPERATOR_LR: Final = mpbr.REC004D_OPERATOR_LR
REC004G_OPERATOR_WEIGHT_DECAY: Final = mpbr.REC004D_OPERATOR_WEIGHT_DECAY
REC004G_OPERATOR_GRAD_CLIP: Final = mpbr.REC004D_OPERATOR_GRAD_CLIP
REC004G_SCHEDULER_ETA_MIN: Final = mpbr.REC004D_SCHEDULER_ETA_MIN
REC004G_T_MAX: Final = mpbr.REC004D_T_MAX  # 1000, mechanically continued (never reset)

REC004G_VOCAB_SIZE: Final = mpbr.REC004D_VOCAB_SIZE
REC004G_SEQUENCE_LENGTH_RANGE: Final = mpbr.REC004D_SEQUENCE_LENGTH_RANGE
REC004G_LEGAL_LENGTHS: Final[tuple[int, ...]] = mpbr.REC004D_LEGAL_LENGTHS

REC004G_EXISTING_VALIDATION_EXAMPLES: Final = mpbr.REC004D_EXISTING_VALIDATION_EXAMPLES
REC004G_EXISTING_VALIDATION_SPLIT: Final = mpbr.REC004D_EXISTING_VALIDATION_SPLIT
REC004G_EXISTING_VALIDATION_FLOOR: Final = mpbr.REC004D_EXISTING_VALIDATION_FLOOR  # 0.95, unchanged

REC004G_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = mpbr.REC004D_PROTECTED_OPERATIONS
REC004G_FIXED_CANDIDATE_OPERATIONS: Final[tuple[str, ...]] = mpbr.REC004D_FIXED_CANDIDATE_OPERATIONS

REC004D_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004d/run_001")
REC004D_LEARNING_CURVE_PATH: Final = REC004D_RUN_DIR / "learning_curve.jsonl"

REC004G_SOURCE_REPLAY_EM_TOL: Final = 1e-9


@dataclass(frozen=True)
class MirrorBudgetExtensionConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004g")
    seed: int = RECOVERY_PILOT_SEED
    existing_validation_examples: int = REC004G_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004G_EXISTING_VALIDATION_FLOOR
    checkpoint_interval: int = REC004G_CHECKPOINT_INTERVAL
    target_cumulative_step: int = REC004G_TARGET_CUMULATIVE_STEP


def _rec004d_training_state_path(init_id: str) -> Path:
    filename = f"step{REC004G_SOURCE_STEP}.pt"
    return REC004D_RUN_DIR / init_id / REC004G_ARM / "training_states" / filename


def _load_rec004d_learning_curve_rows() -> list[dict[str, Any]]:
    if not REC004D_LEARNING_CURVE_PATH.is_file():
        raise mb.MissingArtifactError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {REC004D_LEARNING_CURVE_PATH} not found"
        )
    rows = []
    for line in REC004D_LEARNING_CURVE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_source_training_state(init_id: str) -> dict[str, Any]:
    """A1: load REC-004D's own saved COMPLETE step=6000 training state
    (primitive weights + optimizer + scheduler + RNG). Never falls back to a
    fresh initial state or a different step."""
    path = _rec004d_training_state_path(init_id)
    if not path.is_file():
        raise mb.MissingArtifactError(f"SOURCE_ARTIFACT_UNAVAILABLE: {path} not found")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("cumulative_updates") != REC004G_SOURCE_STEP:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {path} records cumulative_updates="
            f"{state.get('cumulative_updates')}, expected {REC004G_SOURCE_STEP}"
        )
    if state.get("init_id") != init_id or state.get("arm") != REC004G_ARM:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {path} records init_id={state.get('init_id')!r}/"
            f"arm={state.get('arm')!r}, expected {init_id!r}/{REC004G_ARM!r}"
        )
    return state


def verify_source_checkpoint_hash(init_id: str, training_state: dict[str, Any]) -> dict[str, Any]:
    """A2 (part 1): the resumed `primitive_state_dict`'s canonical hash must
    match REC-004D's own `learning_curve.jsonl` row for
    (init_id, arm=P, step=6000) -- proof this is genuinely the same saved
    state, not a re-derived or corrupted one."""
    rows = _load_rec004d_learning_curve_rows()
    match = next(
        (
            r
            for r in rows
            if r["init_id"] == init_id
            and r["arm"] == REC004G_ARM
            and r["step"] == REC004G_SOURCE_STEP
        ),
        None,
    )
    if match is None:
        raise mb.MissingArtifactError(
            "SOURCE_ARTIFACT_UNAVAILABLE: no learning_curve.jsonl row for "
            f"{init_id}:{REC004G_ARM}:step{REC004G_SOURCE_STEP}"
        )
    primitive_state = {
        k: v.detach().clone().cpu() for k, v in training_state["primitive_state_dict"].items()
    }
    computed_hash = mb.canonical_state_hash(primitive_state)
    recorded_hash = match["checkpoint_state_hash"]
    return {
        "init_id": init_id,
        "recorded_checkpoint_state_hash": recorded_hash,
        "computed_checkpoint_state_hash": computed_hash,
        "hash_matches": computed_hash == recorded_hash,
        "recorded_existing_validation_em": match["existing_validation"]["correct_exact_match"],
    }


def _restore_rng_state(training_state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(training_state["cpu_rng_state"])
    cuda_state = training_state.get("cuda_rng_state")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def run_source_replay(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    config: MirrorBudgetExtensionConfig,
) -> dict[str, Any]:
    """Stage A: load + hash-verify the step=6000 state, then confirm a plain
    forward on the resumed weights reproduces REC-004D's own recorded
    existing-validation EM exactly, BEFORE any new optimizer update runs."""
    _guard_not_frozen(f"run_source_replay:{init_id}")
    training_state = load_source_training_state(init_id)
    hash_check = verify_source_checkpoint_hash(init_id, training_state)
    if not hash_check["hash_matches"]:
        return {
            "init_id": init_id,
            "status": "SOURCE_REPLAY_MISMATCH",
            "hash_check": hash_check,
        }

    pid = REC004G_TARGET_PHYSICAL_ID
    device = core.device
    primitive = mpbr._new_arm_primitive(core, REC004G_ARM)
    primitive.to(device)
    primitive.load_state_dict(
        {k: v.to(device) for k, v in training_state["primitive_state_dict"].items()}, strict=True
    )
    primitive.eval()
    original_slot = eval_bank.replace_primitive(pid, primitive)
    try:
        replay_eval = _evaluate_one_operation(
            core, eval_bank, op_to_id, REC004G_TARGET_OPERATION,
            seed=config.seed, n_examples=config.existing_validation_examples,
            split=REC004G_EXISTING_VALIDATION_SPLIT,
        )
    finally:
        eval_bank.replace_primitive(pid, original_slot)

    reproduced_em = replay_eval["correct_exact_match"]
    recorded_em = hash_check["recorded_existing_validation_em"]
    matches = abs(reproduced_em - recorded_em) < REC004G_SOURCE_REPLAY_EM_TOL
    return {
        "init_id": init_id,
        "status": "VERIFIED" if matches else "SOURCE_REPLAY_MISMATCH",
        "hash_check": hash_check,
        "reproduced_existing_validation_em": reproduced_em,
        "recorded_existing_validation_em": recorded_em,
        "matches_within_tolerance": matches,
        "training_state": training_state,
    }


# =============================================================================
# Stage B -- continued training. Reuses REC-004D's own per-step body: the
# same `ibc._generate_step_training_examples`, the same forward/loss/
# clip_grad_norm_/optimizer.step()/scheduler.step() sequence, the same
# checkpoint format -- only the starting state and the step range differ.
# =============================================================================


def run_one_extension(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    replay: dict[str, Any],
    config: MirrorBudgetExtensionConfig,
    output_dir: Path,
) -> dict[str, Any]:
    _guard_not_frozen(f"run_one_extension:{init_id}")
    pid = REC004G_TARGET_PHYSICAL_ID
    device = core.device
    seed = config.seed
    target_step = config.target_cumulative_step
    training_state = replay["training_state"]

    primitive = mpbr._new_arm_primitive(core, REC004G_ARM)
    primitive.to(device)
    primitive.load_state_dict(
        {k: v.to(device) for k, v in training_state["primitive_state_dict"].items()}, strict=True
    )
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004G_OPERATOR_LR, weight_decay=REC004G_OPERATOR_WEIGHT_DECAY
    )
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004G_T_MAX, eta_min=REC004G_SCHEDULER_ETA_MIN
    )
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    if scheduler.last_epoch != REC004G_SOURCE_STEP:
        raise ValueError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: resumed scheduler.last_epoch="
            f"{scheduler.last_epoch}, expected {REC004G_SOURCE_STEP}"
        )
    _restore_rng_state(training_state, device)

    run_dir = output_dir / init_id / REC004G_ARM
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
                "arm": REC004G_ARM,
            },
            state_dir / f"step{step}.pt",
        )

        primitive.eval()
        original_slot = eval_bank.replace_primitive(pid, primitive)
        try:
            existing_validation = _evaluate_one_operation(
                core, eval_bank, op_to_id, REC004G_TARGET_OPERATION,
                seed=seed, n_examples=config.existing_validation_examples,
                split=REC004G_EXISTING_VALIDATION_SPLIT,
            )
        finally:
            eval_bank.replace_primitive(pid, original_slot)
        train_fit = ibc._evaluate_train_fit(
            core, primitive, REC004G_TARGET_OPERATION, seed=seed,
            vocab_size=REC004G_VOCAB_SIZE, sequence_length_range=REC004G_SEQUENCE_LENGTH_RANGE,
        )
        length_strata = ibc._length_stratified_breakdown(
            core, primitive, REC004G_TARGET_OPERATION, seed=seed,
            n_examples=config.existing_validation_examples, split=REC004G_EXISTING_VALIDATION_SPLIT,
        )

        extras: dict[str, Any] | None = None
        if step == target_step:
            final_examples = _generate_parameter_free_examples(
                seed, config.existing_validation_examples, operation=REC004G_TARGET_OPERATION,
                split=REC004G_EXISTING_VALIDATION_SPLIT, vocab_size=REC004G_VOCAB_SIZE,
                sequence_length_range=REC004G_SEQUENCE_LENGTH_RANGE,
            )
            position_summary, position_confusion = mpbr._position_level_breakdown(
                core, primitive, final_examples
            )
            assert isinstance(primitive, mpbr.CrossPositionLengthBiasPrimitive)
            extras = {
                "position_summary": position_summary,
                "position_confusion": position_confusion,
                "bias_ablation": mpbr.run_bias_ablation(core, primitive, final_examples),
            }
        primitive.train()

        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        checkpoints.append(
            {
                "step": step,
                "init_id": init_id,
                "arm": REC004G_ARM,
                "cumulative_training_examples": cumulative_examples,
                "lr_used": lr_used,
                "lr_after_scheduler": lr_after,
                "existing_validation": existing_validation,
                "train_fit": train_fit,
                "length_stratified": length_strata,
                "final_step_extras": extras,
                "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
                "cumulative_wall_clock_seconds": time.time() - t_start,
                "peak_vram_bytes": peak_vram,
            }
        )

    for step in range(REC004G_SOURCE_STEP + 1, target_step + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, REC004G_TARGET_OPERATION,
            vocab_size=REC004G_VOCAB_SIZE, sequence_length_range=REC004G_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(examples)

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004G_TARGET_OPERATION).output_length(n) for n in content_lengths
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004G_OPERATOR_GRAD_CLIP)
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
    new_updates = (diverged_at - REC004G_SOURCE_STEP - 1) if diverged_at is not None else (
        target_step - REC004G_SOURCE_STEP
    )
    return {
        "init_id": init_id,
        "arm": REC004G_ARM,
        "target_cumulative_step": target_step,
        "checkpoints": checkpoints,
        "lr_trace": lr_trace,
        "diverged_at_step": diverged_at,
        "new_optimizer_updates": max(new_updates, 0),
        "final_primitive_state_dict": final_state,
        "total_wall_clock_seconds": time.time() - t_start,
    }


# =============================================================================
# Stage C -- paired self-comparison (step=6000 vs step=12000) and the
# (unchanged) 0.95 floor gate. No candidate/child-bundle/RG3 decision is made
# from this gate result -- see module docstring.
# =============================================================================


def paired_extension_comparison_for_init(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    replay: dict[str, Any],
    extension: dict[str, Any],
    config: MirrorBudgetExtensionConfig,
) -> dict[str, Any]:
    pid = REC004G_TARGET_PHYSICAL_ID
    device = core.device
    target_step = extension["target_cumulative_step"]
    examples = _generate_parameter_free_examples(
        config.seed, config.existing_validation_examples, operation=REC004G_TARGET_OPERATION,
        split=REC004G_EXISTING_VALIDATION_SPLIT, vocab_size=REC004G_VOCAB_SIZE,
        sequence_length_range=REC004G_SEQUENCE_LENGTH_RANGE,
    )
    original_slot = eval_bank.get(pid)

    state_dicts = {
        REC004G_SOURCE_STEP: replay["training_state"]["primitive_state_dict"],
        target_step: extension["final_primitive_state_dict"],
    }
    preds: dict[int, list[list[int]]] = {}
    for step, sd in state_dicts.items():
        step_primitive = mpbr._new_arm_primitive(core, REC004G_ARM)
        step_primitive.to(device)
        step_primitive.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
        step_primitive.eval()
        eval_bank.replace_primitive(pid, step_primitive)
        preds[step] = _predict_tokens(
            core, step_primitive, examples, REC004G_TARGET_OPERATION, None
        )
    eval_bank.replace_primitive(pid, original_slot)

    before_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds[REC004G_SOURCE_STEP], examples, strict=True)
    ]
    after_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds[target_step], examples, strict=True)
    ]
    pairs = list(zip(before_correct, after_correct, strict=False))
    both_correct = sum(1 for b, a in pairs if b and a)
    before_only = sum(1 for b, a in pairs if b and not a)
    after_only = sum(1 for b, a in pairs if a and not b)
    both_wrong = sum(1 for b, a in pairs if not b and not a)

    before_key, after_key = f"step{REC004G_SOURCE_STEP}_correct", f"step{target_step}_correct"
    by_length: dict[int, dict[str, int]] = {}
    for ex, b, a in zip(examples, before_correct, after_correct, strict=False):
        bucket = by_length.setdefault(len(ex.input_tokens), {"n": 0, before_key: 0, after_key: 0})
        bucket["n"] += 1
        bucket[before_key] += int(b)
        bucket[after_key] += int(a)

    em_before = sum(before_correct) / len(before_correct)
    em_after = sum(after_correct) / len(after_correct)
    return {
        "init_id": init_id,
        "n_examples": len(examples),
        "source_step": REC004G_SOURCE_STEP,
        "target_step": target_step,
        "em_source_step": em_before,
        "em_target_step": em_after,
        "delta_target_minus_source": em_after - em_before,
        "both_correct": both_correct,
        "source_step_only_correct": before_only,
        "target_step_only_correct": after_only,
        "both_wrong": both_wrong,
        "by_length": {str(k): v for k, v in sorted(by_length.items())},
    }


def build_candidate_decision(
    per_init_comparisons: dict[str, dict[str, Any]], floor: float
) -> dict[str, Any]:
    target_steps = {c["target_step"] for c in per_init_comparisons.values()}
    target_step = target_steps.pop() if len(target_steps) == 1 else None
    em_at_target = {init_id: c["em_target_step"] for init_id, c in per_init_comparisons.items()}
    clears_floor = {init_id: em >= floor for init_id, em in em_at_target.items()}
    all_clear = all(clears_floor.values())
    return {
        "task_id": REC004G_TASK_ID,
        "floor": floor,
        "target_cumulative_step": target_step,
        "existing_validation_em_at_target_step": em_at_target,
        "clears_floor": clears_floor,
        "candidate_status_at_target_step": (
            f"FIVE_INIT_VALIDATION_FLOOR_PASS_AT_STEP_{target_step}"
            if all_clear
            else f"VALIDATION_TARGET_NOT_MET_AT_STEP_{target_step}"
        ),
        "note": (
            "This is a mechanical floor check only. Per this task's own charter "
            "(a limited budget-extension experiment, not a candidate-adoption task), "
            "no candidate is selected and no child bundle is built even if all 5 "
            "clear the floor -- that decision requires a separate explicit user "
            "instruction."
        ),
        "selected_init": None,
        "selected_intervention": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
    }


# =============================================================================
# Orchestration
# =============================================================================


def _protected_scope_hashes(eval_bank: Any, op_to_id: dict[str, int]) -> dict[str, str]:
    return {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004G_PROTECTED_OPERATIONS
    }


def _rec004d_source_file_hashes() -> dict[str, str]:
    """Raw-byte hashes of the specific REC-004D `run_001` files this task
    reads (the shared learning curve plus each init's saved step=6000
    training state) -- proof this task never mutates its read-only source."""
    hashes = {"learning_curve.jsonl": mb.raw_file_sha256(REC004D_LEARNING_CURVE_PATH)}
    for init_id in REC004G_INIT_IDS:
        path = _rec004d_training_state_path(init_id)
        hashes[f"{init_id}/training_states/step{REC004G_SOURCE_STEP}.pt"] = mb.raw_file_sha256(path)
    return hashes


def _config_to_yaml_dict(config: MirrorBudgetExtensionConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "existing_validation_examples": config.existing_validation_examples,
        "existing_validation_floor": config.existing_validation_floor,
        "checkpoint_interval": config.checkpoint_interval,
    }


def build_cost_accounting(outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    wall_clock_by_init = {init_id: o["total_wall_clock_seconds"] for init_id, o in outcomes.items()}
    return {
        "task_id": REC004G_TASK_ID,
        "wall_clock_seconds_by_init": wall_clock_by_init,
        "total_wall_clock_seconds": sum(wall_clock_by_init.values()),
        "new_optimizer_updates_by_init": {
            init_id: o["new_optimizer_updates"] for init_id, o in outcomes.items()
        },
        "total_new_optimizer_updates": sum(o["new_optimizer_updates"] for o in outcomes.values()),
        "expected_total_new_optimizer_updates": REC004G_ADDITIONAL_UPDATES * len(REC004G_INIT_IDS),
    }


def run_mirror_budget_extension_task(config: MirrorBudgetExtensionConfig) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004G reads REC-004D artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; got seed={seed}"
        )

    if not REC004G_CONTRACT_FILE.is_file():
        return {
            "implementation_status": "BLOCKED",
            "reason": f"AUTHORIZATION_ARTIFACT_MISSING: {REC004G_CONTRACT_FILE} not found",
        }

    before_hashes = _snapshot_forbidden_cache_hashes(seed)
    rec004d_source_files_before = _rec004d_source_file_hashes()

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = _protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_before = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004G_FIXED_CANDIDATE_OPERATIONS
    }

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )

    pid = REC004G_TARGET_PHYSICAL_ID
    original_mirror_slice = eval_bank.get(pid)

    source_replay: dict[str, dict[str, Any]] = {}
    for init_id in REC004G_INIT_IDS:
        eval_bank.replace_primitive(pid, original_mirror_slice)
        source_replay[init_id] = run_source_replay(core, eval_bank, op_to_id, init_id, config)
    eval_bank.replace_primitive(pid, original_mirror_slice)

    (output_dir / "source_replay.json").write_text(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "training_state"}
                for k, v in source_replay.items()
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    blocked_inits = [
        init_id for init_id, r in source_replay.items() if r["status"] != "VERIFIED"
    ]
    if blocked_inits:
        return {
            "implementation_status": "BLOCKED",
            "source_replay_status": "SOURCE_REPLAY_MISMATCH",
            "blocked_inits": blocked_inits,
            "source_replay": source_replay,
        }

    outcomes: dict[str, dict[str, Any]] = {}
    for init_id in REC004G_INIT_IDS:
        eval_bank.replace_primitive(pid, original_mirror_slice)
        outcomes[init_id] = run_one_extension(
            core, eval_bank, op_to_id, init_id, source_replay[init_id], config, output_dir
        )
    eval_bank.replace_primitive(pid, original_mirror_slice)

    with (output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as fh:
        for init_id, outcome in outcomes.items():
            for row in outcome["lr_trace"]:
                fh.write(json.dumps({"init_id": init_id, "arm": REC004G_ARM, **row}) + "\n")

    with (output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for init_id, outcome in outcomes.items():
            for ckpt in outcome["checkpoints"]:
                row = {k: v for k, v in ckpt.items() if k != "final_step_extras"}
                payload = {"init_id": init_id, "arm": REC004G_ARM, **row}
                fh.write(json.dumps(payload, default=str) + "\n")

    per_length_position_metrics = {
        init_id: next(
            (
                c["final_step_extras"]
                for c in outcomes[init_id]["checkpoints"]
                if c.get("final_step_extras")
            ),
            None,
        )
        for init_id in REC004G_INIT_IDS
    }
    bias_ablation = {
        init_id: (per_length_position_metrics[init_id] or {}).get("bias_ablation")
        for init_id in REC004G_INIT_IDS
    }

    per_init_comparisons = {
        init_id: paired_extension_comparison_for_init(
            core, eval_bank, op_to_id, init_id, source_replay[init_id], outcomes[init_id], config
        )
        for init_id in REC004G_INIT_IDS
    }
    candidate_decision = build_candidate_decision(
        per_init_comparisons, config.existing_validation_floor
    )

    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_after = _protected_scope_hashes(eval_bank, op_to_id)
    fixed_candidate_hashes_after = {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004G_FIXED_CANDIDATE_OPERATIONS
    }
    rec004d_source_files_after = _rec004d_source_file_hashes()
    freeze_audit = {
        "task_id": REC004G_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "fixed_candidate_hashes_before": fixed_candidate_hashes_before,
        "fixed_candidate_hashes_after": fixed_candidate_hashes_after,
        "fixed_candidates_unchanged": fixed_candidate_hashes_before == fixed_candidate_hashes_after,
        "rec004d_source_files_sha256_before": rec004d_source_files_before,
        "rec004d_source_files_sha256_after": rec004d_source_files_after,
        "rec004d_run_001_source_files_unchanged": (
            rec004d_source_files_before == rec004d_source_files_after
        ),
    }
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "paired_extension_comparison.json").write_text(
        json.dumps(per_init_comparisons, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "candidate_decision.json").write_text(
        json.dumps(candidate_decision, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "per_length_position_metrics.json").write_text(
        json.dumps(per_length_position_metrics, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "bias_ablation.json").write_text(
        json.dumps(bias_ablation, indent=2, default=str), encoding="utf-8"
    )
    cost_accounting = build_cost_accounting(outcomes)
    (output_dir / "cost_accounting.json").write_text(
        json.dumps(cost_accounting, indent=2, default=str), encoding="utf-8"
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004G_TASK_ID,
        "forbidden_cache_hashes_before": before_hashes,
        "forbidden_cache_hashes_after": after_hashes,
        "forbidden_cache_unchanged": before_hashes == after_hashes,
    }
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    any_diverged = any(o["diverged_at_step"] is not None for o in outcomes.values())
    extension_training_status = "PARTIAL" if any_diverged else "COMPLETE"

    summary = {
        "task_id": REC004G_TASK_ID,
        "source_task_id": REC004G_SOURCE_TASK_ID,
        "implementation_status": "COMPLETE",
        "source_replay_status": "VERIFIED",
        "extension_training_status": extension_training_status,
        "diverged_inits": {
            init_id: o["diverged_at_step"]
            for init_id, o in outcomes.items()
            if o["diverged_at_step"] is not None
        },
        "target_cumulative_step": candidate_decision["target_cumulative_step"],
        "candidate_status_at_target_step": candidate_decision["candidate_status_at_target_step"],
        "existing_validation_em_at_target_step": (
            candidate_decision["existing_validation_em_at_target_step"]
        ),
        "existing_validation_em_at_source_step": {
            init_id: c["em_source_step"] for init_id, c in per_init_comparisons.items()
        },
        "delta_target_minus_source": {
            init_id: c["delta_target_minus_source"] for init_id, c in per_init_comparisons.items()
        },
        "new_optimizer_updates": cost_accounting["total_new_optimizer_updates"],
        "expected_new_optimizer_updates": cost_accounting["expected_total_new_optimizer_updates"],
        "freeze_audit_passed": (
            freeze_audit["core_unchanged"]
            and freeze_audit["protected_operations_unchanged"]
            and freeze_audit["fixed_candidates_unchanged"]
            and freeze_audit["rec004d_run_001_source_files_unchanged"]
        ),
        "side_effect_audit_passed": side_effect_audit["forbidden_cache_unchanged"],
        "selected_init": None,
        "selected_intervention": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "total_wall_clock_seconds": time.time() - start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
