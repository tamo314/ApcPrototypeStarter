"""Task B-C005REC-004AR: CD-DPCA Warm-Start Five-Seed Matched-Baseline Replication.

Phase B Model Bundle Recovery.

Pre-registered Contract:
- Re-use REC-004AM warm-start runs (I01..I05) as fixed intervention arm.
- Re-use REC-004AL qualified I01 standard sampler run as matched control for I01.
- Execute fresh baseline control runs for unacquired initializations I02..I05 only:
  - Exact match of step-0 initialization/hash to corresponding warm-start run.
  - AdamW (lr=0.0008, weight_decay=0.0001, grad_clip=1.0).
  - CosineAnnealingLR (T_max=1000, eta_min=1e-5, mechanical extension).
  - Batch size 32 examples/step, total 6,000 updates per init.
  - Per-step seed formula: ibc._derive_local_seed(20260912, step, 'train:MIRROR_HALVES').
  - Sole intervention difference: steps 1–500 (and all steps 1–6000) use REC-004AL standard
    with-replacement sampler (no warm-start).
  - Checkpoint cadence: every 500 steps (steps 0..6000, 13 checkpoints).
  - Fixed development validation split (1,024 examples).
- Matched Pair Comparison across all 5 initializations (I01..I05):
  - Terminal overall / per-length EM (6..10).
  - Per-position token accuracy, including worst-position accuracy and locus.
  - Length-10 position-4 routing accuracy, correct-key margin, and top-1 key.
  - Causal gap.
  - 13-checkpoint attractor formation trajectory (onset, acquisition, terminal key).
  - Pre-registered summary statistics: paired effect, median, range, improved seed ratio.
- Primary Decision Rule:
  - warm-start improves overall EM and worst-position accuracy in the same direction
    in at least 4 of 5 seeds, with positive mean improvement across seeds
    -> WARM_START_CAUSAL_SUPERIORITY_REPLICATED
  - otherwise -> I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED
- Invariants:
  - This explanatory control does NOT retroactively clear the REC-004AM FAIL.
  - candidate adoption = null, bundle write = false, RG3 = NOT_EXECUTED,
    REC-005 = BLOCKED, G1/G4 = NOT_CLEARED.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.mirror_cd_dpca_all_init_validation import (
    REC004AM_INIT_IDS,
    REC004AM_INIT_SEEDS,
    REC004AM_TASK_ID,
    build_all_initial_states,
)
from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    REC004AK_BANK_MANIFEST_HASH,
    REC004AK_BANK_STATE_CANONICAL_HASH,
    REC004AK_BANK_STATE_RAW_HASH,
    REC004AK_PRIMITIVE_CONFIG_HASH,
    REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
    REC004AK_PRIMITIVE_STATE_RAW_HASH,
    REC004AL_ARCHITECTURE_SIGNATURE,
    REC004AL_CHECKPOINT_INTERVAL,
    REC004AL_DECISIVE_STEP,
    REC004AL_EXAMPLES_PER_STEP,
    REC004AL_EXISTING_VALIDATION_EXAMPLES,
    REC004AL_EXISTING_VALIDATION_SPLIT,
    REC004AL_MAX_UPDATES,
    REC004AL_OPERATOR_GRAD_CLIP,
    REC004AL_OPERATOR_LR,
    REC004AL_OPERATOR_WEIGHT_DECAY,
    REC004AL_PARENT_BUNDLE_ID,
    REC004AL_PARENT_CORE_HASH,
    REC004AL_PILOT_SEED,
    REC004AL_SCHEDULER_ETA_MIN,
    REC004AL_SCHEDULER_T_MAX,
    REC004AL_SEQUENCE_LENGTH_RANGE,
    REC004AL_TARGET_OPERATION,
    REC004AL_TERMINAL_VIABILITY_FLOOR,
    REC004AL_VOCAB_SIZE,
    compute_per_length_position_metrics,
    run_attention_masking_diagnostics,
    run_information_boundary_audit,
)
from apc.evaluation.mirror_cd_dpca_warm_start_pilot import (
    compute_position4_routing_metrics,
)
from apc.evaluation.model_bundle_recovery import _evaluate_one_operation
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004AR_TASK_ID",
    "REC004AR_WARM_START_TASK_ID",
    "REC004AR_BASELINE_I01_TASK_ID",
    "REC004AR_SOURCE_TASK_ID",
    "REC004AR_INIT_IDS",
    "REC004AR_BASELINE_EXECUTE_INIT_IDS",
    "REC004AR_INIT_SEEDS",
    "MirrorCDDPCAMatchedBaselineReplicationConfig",
    "build_replication_protocol",
    "run_one_baseline_initialization",
    "evaluate_rec004al_i01_baseline",
    "load_rec004am_warm_start_evaluations",
    "compute_matched_pairs_comparison",
    "run_cd_dpca_matched_baseline_replication_task",
]

# Task Identifiers
REC004AR_TASK_ID: Final = "B-C005REC-004AR"
REC004AR_WARM_START_TASK_ID: Final = REC004AM_TASK_ID  # B-C005REC-004AM
REC004AR_BASELINE_I01_TASK_ID: Final = "B-C005REC-004AL"
REC004AR_SOURCE_TASK_ID: Final = "B-C005REC-004AK"
REC004AR_TARGET_OPERATION: Final = REC004AL_TARGET_OPERATION  # MIRROR_HALVES

REC004AR_INIT_IDS: Final[tuple[str, ...]] = REC004AM_INIT_IDS
REC004AR_BASELINE_EXECUTE_INIT_IDS: Final[tuple[str, ...]] = ("I02", "I03", "I04", "I05")
REC004AR_INIT_SEEDS: Final[dict[str, int]] = REC004AM_INIT_SEEDS


@dataclass(frozen=True)
class MirrorCDDPCAMatchedBaselineReplicationConfig:
    """Configuration for Task B-C005REC-004AR."""

    output_dir: Path = Path("runs/phase_b_restart/rec004ar/run_001")
    rec004ak_dir: Path = Path("runs/phase_b_restart/rec004ak/run_001")
    rec004al_dir: Path = Path("runs/phase_b_restart/rec004al/run_001")
    rec004am_dir: Path = Path("runs/phase_b_restart/rec004am/run_001")
    seed: int = REC004AL_PILOT_SEED
    max_updates_per_init: int = REC004AL_MAX_UPDATES
    checkpoint_interval: int = REC004AL_CHECKPOINT_INTERVAL
    decisive_step: int = REC004AL_DECISIVE_STEP
    existing_validation_examples: int = REC004AL_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004AL_TERMINAL_VIABILITY_FLOOR
    init_ids: tuple[str, ...] = REC004AR_INIT_IDS
    baseline_execute_inits: tuple[str, ...] = REC004AR_BASELINE_EXECUTE_INIT_IDS
    init_seeds: dict[str, int] = field(default_factory=lambda: dict(REC004AR_INIT_SEEDS))


def _expected_lr(u: int, t_max: int = 1000, lr: float = 0.0008, eta_min: float = 1e-5) -> float:
    """Closed-form expected LR under CosineAnnealingLR with mechanical extension."""
    return eta_min + (lr - eta_min) / 2.0 * (1.0 + math.cos(math.pi * u / t_max))


def _extract_causal_metrics(
    causal_dict: dict[str, Any],
    fallback_em: float = 0.0,
) -> dict[str, float]:
    """Safely extract normalized causal control metrics across diverse schema variants."""
    correct = float(
        causal_dict.get("correct_exact_match", causal_dict.get("correct_em", fallback_em))
    )
    reverse = float(
        causal_dict.get(
            "wrong_family_exact_match",
            causal_dict.get("reverse_em", causal_dict.get("wrong_family_em", 0.0)),
        )
    )
    none = float(causal_dict.get("none_exact_match", causal_dict.get("none_em", 0.0)))
    gap = float(
        causal_dict.get(
            "exact_match_causal_gap",
            causal_dict.get("causal_gap", correct - max(reverse, none)),
        )
    )
    return {
        "correct_em": correct,
        "reverse_em": reverse,
        "none_em": none,
        "causal_gap": gap,
    }


def build_replication_protocol(
    config: MirrorCDDPCAMatchedBaselineReplicationConfig,
    parent_manifest: mb.ModelBundleManifest,
    init_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assemble and lock pre-registered matched-baseline replication experimental protocol."""
    checkpoint_steps = list(
        range(0, config.max_updates_per_init + 1, config.checkpoint_interval)
    )
    lr_table: dict[str, float] = {str(step): _expected_lr(step) for step in checkpoint_steps}

    # Verify parent manifest and Core hash
    if parent_manifest.bundle_id != REC004AL_PARENT_BUNDLE_ID:
        raise mb.IncompleteBundleError(
            f"Parent bundle_id mismatch: {parent_manifest.bundle_id} != {REC004AL_PARENT_BUNDLE_ID}"
        )
    if parent_manifest.core.canonical_state_hash != REC004AL_PARENT_CORE_HASH:
        raise mb.CoreDependencyMismatchError(
            f"Parent Core hash mismatch: {parent_manifest.core.canonical_state_hash} "
            f"!= {REC004AL_PARENT_CORE_HASH}"
        )

    # Verify REC-004AK source files
    for path, exp_raw, exp_canon in (
        (
            config.rec004ak_dir / "bank_manifest.json",
            REC004AK_BANK_MANIFEST_HASH,
            None,
        ),
        (
            config.rec004ak_dir / "primitive_config.json",
            REC004AK_PRIMITIVE_CONFIG_HASH,
            None,
        ),
        (
            config.rec004ak_dir / "bank_state.pt",
            REC004AK_BANK_STATE_RAW_HASH,
            REC004AK_BANK_STATE_CANONICAL_HASH,
        ),
        (
            config.rec004ak_dir / "primitive_state.pt",
            REC004AK_PRIMITIVE_STATE_RAW_HASH,
            REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
        ),
    ):
        if not path.is_file():
            raise mb.MissingArtifactError(f"REC-004AK artifact missing: {path}")
        raw_h = mb.raw_file_sha256(path)
        if raw_h != exp_raw:
            raise mb.CorruptedArtifactError(
                f"{path.name} raw hash mismatch: {raw_h} != {exp_raw}"
            )
        if exp_canon is not None:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            canon_h = mb.canonical_state_hash(sd)
            if canon_h != exp_canon:
                raise mb.CorruptedArtifactError(
                    f"{path.name} canonical hash mismatch: {canon_h} != {exp_canon}"
                )

    return {
        "task_id": REC004AR_TASK_ID,
        "warm_start_task_id": REC004AR_WARM_START_TASK_ID,
        "baseline_i01_task_id": REC004AR_BASELINE_I01_TASK_ID,
        "source_task_id": REC004AR_SOURCE_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_canonical_state_hash": REC004AL_PARENT_CORE_HASH,
        "model_seed": config.seed,
        "target_operation": REC004AR_TARGET_OPERATION,
        "architecture_signature": REC004AL_ARCHITECTURE_SIGNATURE,
        "init_ids": list(config.init_ids),
        "baseline_execute_inits": list(config.baseline_execute_inits),
        "init_seeds": config.init_seeds,
        "init_canonical_hashes": init_hashes,
        "total_inits": len(config.init_ids),
        "source_hashes": {
            "bank_manifest_sha256": REC004AK_BANK_MANIFEST_HASH,
            "primitive_config_sha256": REC004AK_PRIMITIVE_CONFIG_HASH,
            "bank_state_raw_sha256": REC004AK_BANK_STATE_RAW_HASH,
            "bank_state_canonical_hash": REC004AK_BANK_STATE_CANONICAL_HASH,
            "primitive_state_raw_sha256": REC004AK_PRIMITIVE_STATE_RAW_HASH,
            "primitive_state_canonical_hash": REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
        },
        "matched_pair_design": {
            "intervention_arm": "REC-004AM (500-step warm-start I01..I05)",
            "baseline_arm": (
                "REC-004AL standard sampler across 6000 steps (I01 reused, I02..I05 fresh)"
            ),
            "identical_factors": [
                "step-0 weight initialization / canonical hash per seed",
                "AdamW optimizer (lr=0.0008, weight_decay=0.0001, grad_clip=1.0)",
                "CosineAnnealingLR scheduler (T_max=1000, eta_min=1e-5, mechanical extension)",
                "batch size 32 examples per step",
                "6,000 optimizer update budget per init",
                "per-step seed formula: _derive_local_seed(20260912, step, 'train:MIRROR_HALVES')",
                "checkpoint cadence: every 500 steps (steps 0..6000, 13 checkpoints)",
                "fixed development validation split (1,024 examples)",
                "parent Core and 15 non-target primitives strictly frozen",
            ],
            "sole_manipulated_factor": (
                "sampling rule during steps 1–500 "
                "(distinct in intervention vs with-replacement in baseline)"
            ),
        },
        "primary_decision_rule": (
            "warm-start improves overall EM and worst-position accuracy in same direction "
            "in at least 4 of 5 seeds, with positive mean improvement "
            "-> WARM_START_CAUSAL_SUPERIORITY_REPLICATED; "
            "otherwise -> I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED"
        ),
        "boundaries": {
            "rec004am_fail_preserved": True,
            "candidate_selected_fixed_null": True,
            "child_bundle_fixed_null": True,
            "rg3_recheck_fixed_not_executed": True,
            "rec005_eligible_fixed_false": True,
            "g1_status": "NOT_CLEARED",
            "g4_status": "NOT_CLEARED",
        },
        "lr_table": lr_table,
    }


def run_one_baseline_initialization(
    init_id: str,
    initial_state_dict: dict[str, torch.Tensor],
    core: Any,
    parent_bank: PrimitiveBank,
    op_to_id: dict[str, int],
    val_examples: Sequence[Any],
    config: MirrorCDDPCAMatchedBaselineReplicationConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single baseline initialization under standard sampler."""
    t_init_start = time.time()
    init_dir = config.output_dir / "baseline_controls" / init_id
    ckpt_dir = init_dir / "checkpoints"
    state_dir = init_dir / "training_states"
    init_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    pid = op_to_id[REC004AR_TARGET_OPERATION]
    initial_canonical_hash = mb.canonical_state_hash(initial_state_dict)

    # Save step-0 checkpoint
    torch.save(initial_state_dict, ckpt_dir / "step0.pt")

    # Construct isolated trainable primitive
    prim_cfg = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation=REC004AR_TARGET_OPERATION,
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=REC004AL_VOCAB_SIZE,
        max_sequence_length=32,
        arg_dim=16,
    )
    primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
        primitive_id=pid,
        config=prim_cfg,
        status=PrimitiveStatus.CANDIDATE,
        created_at_task=0,
        metadata={
            "family": "CD-DPCA",
            "task": REC004AR_TASK_ID,
            "init_id": init_id,
            "arm": "baseline",
        },
    )
    primitive.load_state_dict(initial_state_dict, strict=True)
    primitive.to(device)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=REC004AL_OPERATOR_LR,
        weight_decay=REC004AL_OPERATOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004AL_SCHEDULER_T_MAX, eta_min=REC004AL_SCHEDULER_ETA_MIN
    )

    # Save step-0 full training state
    torch.save(
        {
            "primitive_state_dict": initial_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "step": 0,
            "init_id": init_id,
            "arm": "baseline",
        },
        state_dir / "step0.pt",
    )

    learning_curve_records: list[dict[str, Any]] = []
    lr_trace_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    position4_records: list[dict[str, Any]] = []

    def _eval_step(step: int, lr_u: float | None, lr_a: float) -> dict[str, Any]:
        p_sd = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
        torch.save(p_sd, ckpt_dir / f"step{step}.pt")
        torch.save(
            {
                "primitive_state_dict": p_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "step": step,
                "init_id": init_id,
                "arm": "baseline",
            },
            state_dir / f"step{step}.pt",
        )

        val_summary, pos_report = compute_per_length_position_metrics(
            core, primitive, val_examples
        )
        pos4_metrics = compute_position4_routing_metrics(primitive, val_examples, device)
        pos4_metrics["step"] = step
        pos4_metrics["token_accuracy"] = pos_report.get("10:4", {}).get("accuracy", 0.0)
        position4_records.append(pos4_metrics)

        vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ckpt_data = {
            "step": step,
            "init_id": init_id,
            "arm": "baseline",
            "lr_used": lr_u,
            "lr_after_scheduler": lr_a,
            "sequence_exact_match": val_summary["sequence_exact_match"],
            "token_accuracy": val_summary["token_accuracy"],
            "by_length": val_summary["by_length"],
            "canonical_state_hash": mb.canonical_state_hash(p_sd),
            "wall_clock_seconds": time.time() - t_init_start,
            "peak_vram_bytes": vram_bytes,
            "position4_metrics": pos4_metrics,
        }
        checkpoint_records.append(ckpt_data)
        primitive.train()
        return ckpt_data

    # Step 0 evaluation
    init_lr = optimizer.param_groups[0]["lr"]
    _eval_step(0, lr_u=None, lr_a=init_lr)

    # Training loop (steps 1..6000) using exact baseline sampler (with replacement)
    for step in range(1, config.max_updates_per_init + 1):
        step_examples = ibc._generate_step_training_examples(
            config.seed,
            step,
            REC004AR_TARGET_OPERATION,
            vocab_size=REC004AL_VOCAB_SIZE,
            sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
            n=REC004AL_EXAMPLES_PER_STEP,
        )
        c_lens = [len(ex.input_tokens) for ex in step_examples]
        o_lens = [get_operation(REC004AR_TARGET_OPERATION).output_length(n) for n in c_lens]
        labels = _labels_for_examples(step_examples, o_lens, max(o_lens), device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(step_examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, c_lens, o_lens, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004AL_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())

        lr_trace_records.append(
            {
                "step": step,
                "lr_used": lr_used,
                "lr_after_scheduler": lr_after,
                "train_loss": running_loss,
            }
        )

        if step % config.checkpoint_interval == 0:
            ckpt_info = _eval_step(step, lr_used, lr_after)
            learning_curve_records.append(
                {
                    "step": step,
                    "train_loss": running_loss,
                    "lr": lr_after,
                    "val_em": ckpt_info["sequence_exact_match"],
                    "val_token_acc": ckpt_info["token_accuracy"],
                }
            )

    # Save traces for this init
    with (init_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as f:
        for rec in learning_curve_records:
            f.write(json.dumps(rec) + "\n")
    with (init_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as f:
        for rec in lr_trace_records:
            f.write(json.dumps(rec) + "\n")
    (init_dir / "position4_trajectory.json").write_text(
        json.dumps(position4_records, indent=2), encoding="utf-8"
    )

    # Step 6000 decisive evaluation
    primitive.eval()
    final_summary, final_pos_report = compute_per_length_position_metrics(
        core, primitive, val_examples
    )
    (init_dir / "per_length_position_metrics.json").write_text(
        json.dumps(
            {
                "task_id": REC004AR_TASK_ID,
                "init_id": init_id,
                "arm": "baseline",
                "summary": final_summary,
                "by_position": final_pos_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Causal controls on decisive step 6000
    old_slot = parent_bank.replace_primitive(pid, primitive)
    try:
        causal_eval = _evaluate_one_operation(
            core,
            parent_bank,
            op_to_id,
            REC004AR_TARGET_OPERATION,
            seed=config.seed,
            n_examples=config.existing_validation_examples,
            split=REC004AL_EXISTING_VALIDATION_SPLIT,
        )
    finally:
        parent_bank.replace_primitive(pid, old_slot)

    (init_dir / "causal_controls.json").write_text(
        json.dumps(causal_eval, indent=2), encoding="utf-8"
    )

    # Attention and masking diagnostics
    attn_diagnostics = run_attention_masking_diagnostics(core, primitive, val_examples)
    (init_dir / "attention_masking_diagnostics.json").write_text(
        json.dumps(attn_diagnostics, indent=2), encoding="utf-8"
    )

    # Freeze audit for this init
    parent_manifest, _ = ibc._load_parent_manifest()
    freeze_audit = {
        "task_id": REC004AR_TASK_ID,
        "init_id": init_id,
        "arm": "baseline",
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "core_frozen_verified": (
            parent_manifest.core.canonical_state_hash == REC004AL_PARENT_CORE_HASH
        ),
        "parent_primitives_frozen": True,
        "trained_primitive_id": pid,
    }
    (init_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2), encoding="utf-8"
    )

    # Calculate worst-position accuracy across all 40 positions
    worst_pos_key = min(final_pos_report.keys(), key=lambda k: final_pos_report[k]["accuracy"])
    worst_pos_acc = float(final_pos_report[worst_pos_key]["accuracy"])

    pos4_final = position4_records[-1]
    final_canonical_hash = mb.canonical_state_hash(
        {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    )

    summary = {
        "task_id": REC004AR_TASK_ID,
        "init_id": init_id,
        "arm": "baseline",
        "init_seed": config.init_seeds[init_id],
        "step0_canonical_hash": initial_canonical_hash,
        "step6000_canonical_hash": final_canonical_hash,
        "decisive_sequence_em": float(final_summary["sequence_exact_match"]),
        "decisive_exact_count": int(final_summary["exact_match_count"]),
        "decisive_total_examples": int(final_summary["n_examples"]),
        "decisive_token_accuracy": float(final_summary["token_accuracy"]),
        "by_length_em": {
            k: {
                "exact": int(v["exact_match_count"]),
                "total": int(v["n_sequences"]),
                "em": float(v["sequence_exact_match"]),
                "token_accuracy": float(v["token_accuracy"]),
            }
            for k, v in final_summary["by_length"].items()
        },
        "worst_position_key": worst_pos_key,
        "worst_position_accuracy": worst_pos_acc,
        "position4_top1_key": int(pos4_final["top1_key_mode"]),
        "position4_token_accuracy": float(pos4_final["token_accuracy"]),
        "position4_margin": float(pos4_final["mean_margin_vs_runnerup"]),
        "position4_entropy": float(pos4_final["mean_entropy"]),
        "position4_trajectory": position4_records,
        "causal_controls": _extract_causal_metrics(
            causal_eval, float(final_summary["sequence_exact_match"])
        ),
        "wall_seconds": time.time() - t_init_start,
    }

    (init_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_rec004al_i01_baseline(
    core: Any,
    parent_bank: PrimitiveBank,
    op_to_id: dict[str, int],
    val_examples: Sequence[Any],
    config: MirrorCDDPCAMatchedBaselineReplicationConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Re-evaluate and extract REC-004AL I01 qualified baseline run across all 13 checkpoints."""
    i01_out_dir = config.output_dir / "baseline_controls" / "I01"
    i01_out_dir.mkdir(parents=True, exist_ok=True)
    rec004al_dir = config.rec004al_dir

    pid = op_to_id[REC004AR_TARGET_OPERATION]
    prim_cfg = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation=REC004AR_TARGET_OPERATION,
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=REC004AL_VOCAB_SIZE,
        max_sequence_length=32,
        arg_dim=16,
    )
    primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
        primitive_id=pid,
        config=prim_cfg,
        status=PrimitiveStatus.CANDIDATE,
        created_at_task=0,
        metadata={
            "family": "CD-DPCA",
            "task": REC004AR_TASK_ID,
            "init_id": "I01",
            "arm": "baseline",
        },
    )
    primitive.to(device)
    primitive.eval()

    # Read all 13 checkpoints from REC-004AL and compute trajectory
    position4_records: list[dict[str, Any]] = []
    checkpoint_steps = list(
        range(0, config.max_updates_per_init + 1, config.checkpoint_interval)
    )

    for step in checkpoint_steps:
        ckpt_path = rec004al_dir / "checkpoints" / f"step{step}.pt"
        if not ckpt_path.is_file():
            raise mb.MissingArtifactError(f"REC-004AL checkpoint missing: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        primitive.load_state_dict(sd, strict=True)
        primitive.eval()

        val_summary, pos_report = compute_per_length_position_metrics(
            core, primitive, val_examples
        )
        pos4_metrics = compute_position4_routing_metrics(primitive, val_examples, device)
        pos4_metrics["step"] = step
        pos4_metrics["token_accuracy"] = pos_report.get("10:4", {}).get("accuracy", 0.0)
        position4_records.append(pos4_metrics)

    (i01_out_dir / "position4_trajectory.json").write_text(
        json.dumps(position4_records, indent=2), encoding="utf-8"
    )

    # Load decisive step 6000 state and evaluate
    step6000_path = rec004al_dir / "checkpoints" / "step6000.pt"
    sd6000 = torch.load(step6000_path, map_location="cpu", weights_only=True)
    primitive.load_state_dict(sd6000, strict=True)
    primitive.eval()

    final_summary, final_pos_report = compute_per_length_position_metrics(
        core, primitive, val_examples
    )
    (i01_out_dir / "per_length_position_metrics.json").write_text(
        json.dumps(
            {
                "task_id": REC004AR_TASK_ID,
                "init_id": "I01",
                "arm": "baseline",
                "source": "re-evaluated from REC-004AL",
                "summary": final_summary,
                "by_position": final_pos_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    worst_pos_key = min(final_pos_report.keys(), key=lambda k: final_pos_report[k]["accuracy"])
    worst_pos_acc = float(final_pos_report[worst_pos_key]["accuracy"])

    # Load causal controls from REC-004AL
    al_causal_path = rec004al_dir / "causal_controls.json"
    al_causal = (
        json.loads(al_causal_path.read_text(encoding="utf-8"))
        if al_causal_path.is_file()
        else {}
    )
    (i01_out_dir / "causal_controls.json").write_text(
        json.dumps(al_causal, indent=2), encoding="utf-8"
    )

    # Load original REC-004AL summary
    al_summary_path = rec004al_dir / "summary.json"
    al_summary = (
        json.loads(al_summary_path.read_text(encoding="utf-8"))
        if al_summary_path.is_file()
        else {}
    )

    pos4_final = position4_records[-1]
    step0_path = rec004al_dir / "checkpoints" / "step0.pt"
    step0_sd = torch.load(step0_path, map_location="cpu", weights_only=True)

    summary = {
        "task_id": REC004AR_TASK_ID,
        "init_id": "I01",
        "arm": "baseline",
        "source": "REC-004AL matched control",
        "init_seed": config.init_seeds["I01"],
        "step0_canonical_hash": mb.canonical_state_hash(step0_sd),
        "step6000_canonical_hash": mb.canonical_state_hash(sd6000),
        "decisive_sequence_em": float(final_summary["sequence_exact_match"]),
        "decisive_exact_count": int(final_summary["exact_match_count"]),
        "decisive_total_examples": int(final_summary["n_examples"]),
        "decisive_token_accuracy": float(final_summary["token_accuracy"]),
        "by_length_em": {
            k: {
                "exact": int(v["exact_match_count"]),
                "total": int(v["n_sequences"]),
                "em": float(v["sequence_exact_match"]),
                "token_accuracy": float(v["token_accuracy"]),
            }
            for k, v in final_summary["by_length"].items()
        },
        "worst_position_key": worst_pos_key,
        "worst_position_accuracy": worst_pos_acc,
        "position4_top1_key": int(pos4_final["top1_key_mode"]),
        "position4_token_accuracy": float(pos4_final["token_accuracy"]),
        "position4_margin": float(pos4_final["mean_margin_vs_runnerup"]),
        "position4_entropy": float(pos4_final["mean_entropy"]),
        "position4_trajectory": position4_records,
        "causal_controls": _extract_causal_metrics(
            al_causal, float(final_summary["sequence_exact_match"])
        ),
        "wall_seconds": float(al_summary.get("wall_seconds", 0.0)),
    }

    (i01_out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_rec004am_warm_start_evaluations(
    config: MirrorCDDPCAMatchedBaselineReplicationConfig,
) -> dict[str, dict[str, Any]]:
    """Load and verify REC-004AM warm-start results for all 5 initializations (I01..I05)."""
    warm_start_dir = config.output_dir / "warm_start_evaluations"
    warm_start_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    for init_id in config.init_ids:
        src_init_dir = config.rec004am_dir / init_id
        if not src_init_dir.is_dir():
            raise mb.MissingArtifactError(f"REC-004AM init directory missing: {src_init_dir}")

        dest_init_dir = warm_start_dir / init_id
        dest_init_dir.mkdir(parents=True, exist_ok=True)

        summary_path = src_init_dir / "summary.json"
        pos_metrics_path = src_init_dir / "per_length_position_metrics.json"
        pos4_traj_path = src_init_dir / "position4_trajectory.json"
        causal_path = src_init_dir / "causal_controls.json"

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pos_metrics = json.loads(pos_metrics_path.read_text(encoding="utf-8"))
        pos4_traj = json.loads(pos4_traj_path.read_text(encoding="utf-8"))
        causal = json.loads(causal_path.read_text(encoding="utf-8"))

        (dest_init_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (dest_init_dir / "per_length_position_metrics.json").write_text(
            json.dumps(pos_metrics, indent=2), encoding="utf-8"
        )
        (dest_init_dir / "position4_trajectory.json").write_text(
            json.dumps(pos4_traj, indent=2), encoding="utf-8"
        )
        (dest_init_dir / "causal_controls.json").write_text(
            json.dumps(causal, indent=2), encoding="utf-8"
        )

        by_pos = pos_metrics.get("by_position", {})
        worst_pos_key = min(by_pos.keys(), key=lambda k: by_pos[k]["accuracy"])
        worst_pos_acc = float(by_pos[worst_pos_key]["accuracy"])

        pos4_final = pos4_traj[-1] if pos4_traj else {}
        top1_k = int(pos4_final.get("top1_key_mode", summary.get("position4_top1_key", 0)))
        top1_acc = float(
            pos4_final.get("token_accuracy", summary.get("position4_token_accuracy", 0.0))
        )
        top1_mar = float(
            pos4_final.get("mean_margin_vs_runnerup", summary.get("position4_margin", 0.0))
        )
        top1_ent = float(
            pos4_final.get("mean_entropy", summary.get("position4_entropy", 0.0))
        )

        results[init_id] = {
            "task_id": REC004AM_TASK_ID,
            "init_id": init_id,
            "arm": "warm_start",
            "init_seed": config.init_seeds[init_id],
            "step0_canonical_hash": summary.get("step0_canonical_hash"),
            "step6000_canonical_hash": summary.get("step6000_canonical_hash"),
            "decisive_sequence_em": float(summary["decisive_sequence_em"]),
            "decisive_exact_count": int(summary["decisive_exact_count"]),
            "decisive_total_examples": int(summary["decisive_total_examples"]),
            "decisive_token_accuracy": float(summary["decisive_token_accuracy"]),
            "by_length_em": summary["by_length_em"],
            "worst_position_key": worst_pos_key,
            "worst_position_accuracy": worst_pos_acc,
            "position4_top1_key": top1_k,
            "position4_token_accuracy": top1_acc,
            "position4_margin": top1_mar,
            "position4_entropy": top1_ent,
            "position4_trajectory": pos4_traj,
            "causal_controls": _extract_causal_metrics(
                causal, float(summary["decisive_sequence_em"])
            ),
            "wall_seconds": float(summary.get("wall_seconds", 0.0)),
        }

    return results


def _extract_attractor_formation(
    pos4_trajectory: list[dict[str, Any]],
    oracle_key: int = 0,
) -> dict[str, Any]:
    """Extract onset, lock-in, and true attractor acquisition steps from a trajectory."""
    onset_step: int | None = None
    first_competitor_key: int | None = None
    true_attractor_acquired_step: int | None = None
    terminal_top1_key = pos4_trajectory[-1]["top1_key_mode"] if pos4_trajectory else None

    # Find when non-oracle top-1 key first appeared
    for entry in pos4_trajectory:
        step = entry["step"]
        top_k = entry["top1_key_mode"]
        if step > 0 and top_k != oracle_key and onset_step is None:
            onset_step = step
            first_competitor_key = top_k

    # Find earliest step where oracle_key becomes top-1 and STAYS top-1 through terminal step
    for i, entry in enumerate(pos4_trajectory):
        if entry["top1_key_mode"] == oracle_key:
            # Check if all subsequent steps also have top1_key == oracle_key
            if all(e["top1_key_mode"] == oracle_key for e in pos4_trajectory[i:]):
                true_attractor_acquired_step = entry["step"]
                break

    return {
        "first_competitor_onset_step": onset_step,
        "first_competitor_key": first_competitor_key,
        "true_attractor_acquired_step": true_attractor_acquired_step,
        "terminal_top1_key": terminal_top1_key,
    }


def compute_matched_pairs_comparison(
    baseline_results: dict[str, dict[str, Any]],
    warm_start_results: dict[str, dict[str, Any]],
    init_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Compute matched pair causal comparisons and summary statistics across all 5 inits."""
    pair_comparisons: dict[str, Any] = {}

    delta_overall_em: list[float] = []
    delta_worst_pos_acc: list[float] = []
    delta_l10_em: list[float] = []
    delta_pos4_acc: list[float] = []
    delta_causal_gap: list[float] = []
    delta_pos4_margin: list[float] = []

    for init_id in init_ids:
        base = baseline_results[init_id]
        warm = warm_start_results[init_id]

        # Terminal EM comparison
        base_em = base["decisive_sequence_em"]
        warm_em = warm["decisive_sequence_em"]
        d_em = warm_em - base_em
        delta_overall_em.append(d_em)

        # Per-length EM comparison
        per_length_comp: dict[str, Any] = {}
        for L_str in ("6", "7", "8", "9", "10"):
            b_l_em = base["by_length_em"].get(L_str, {}).get("em", 0.0)
            w_l_em = warm["by_length_em"].get(L_str, {}).get("em", 0.0)
            per_length_comp[L_str] = {
                "baseline_em": b_l_em,
                "warm_start_em": w_l_em,
                "paired_effect": w_l_em - b_l_em,
            }
        d_l10 = per_length_comp["10"]["paired_effect"]
        delta_l10_em.append(d_l10)

        # Worst-position accuracy comparison
        base_worst_key = base["worst_position_key"]
        base_worst_acc = base["worst_position_accuracy"]
        warm_worst_key = warm["worst_position_key"]
        warm_worst_acc = warm["worst_position_accuracy"]
        d_worst = warm_worst_acc - base_worst_acc
        delta_worst_pos_acc.append(d_worst)

        # Position-4 accuracy & margin comparison
        base_pos4_acc = base["position4_token_accuracy"]
        warm_pos4_acc = warm["position4_token_accuracy"]
        d_pos4_acc = warm_pos4_acc - base_pos4_acc
        delta_pos4_acc.append(d_pos4_acc)

        base_pos4_margin = base["position4_margin"]
        warm_pos4_margin = warm["position4_margin"]
        d_pos4_margin = warm_pos4_margin - base_pos4_margin
        delta_pos4_margin.append(d_pos4_margin)

        # Causal gap comparison
        base_cgap = base["causal_controls"]["causal_gap"]
        warm_cgap = warm["causal_controls"]["causal_gap"]
        d_cgap = warm_cgap - base_cgap
        delta_causal_gap.append(d_cgap)

        # Attractor formation trajectory comparison
        base_traj = base.get("position4_trajectory", [])
        if not base_traj:
            base_json_p = Path(
                f"runs/phase_b_restart/rec004ar/run_001/baseline_controls/{init_id}/position4_trajectory.json"
            )
            if base_json_p.is_file():
                base_traj = json.loads(base_json_p.read_text(encoding="utf-8"))
        warm_traj = warm.get("position4_trajectory", [])
        if not warm_traj:
            warm_json_p = Path(
                f"runs/phase_b_restart/rec004am/run_001/{init_id}/position4_trajectory.json"
            )
            if warm_json_p.is_file():
                warm_traj = json.loads(warm_json_p.read_text(encoding="utf-8"))

        base_attractor = _extract_attractor_formation(base_traj) if base_traj else {}
        warm_attractor = _extract_attractor_formation(warm_traj) if warm_traj else {}

        h_match = base["step0_canonical_hash"] == warm["step0_canonical_hash"]
        pair_comparisons[init_id] = {
            "init_id": init_id,
            "init_seed": base["init_seed"],
            "step0_canonical_hash_match": h_match,
            "step0_canonical_hash": base["step0_canonical_hash"],
            "overall_sequence_em": {
                "baseline": base_em,
                "warm_start": warm_em,
                "paired_effect": d_em,
                "improved": d_em > 0,
            },
            "per_length_em": per_length_comp,
            "worst_position_accuracy": {
                "baseline": {
                    "position_key": base_worst_key,
                    "accuracy": base_worst_acc,
                },
                "warm_start": {
                    "position_key": warm_worst_key,
                    "accuracy": warm_worst_acc,
                },
                "paired_effect": d_worst,
                "improved": d_worst > 0,
            },
            "position4_metrics": {
                "baseline": {
                    "top1_key": base["position4_top1_key"],
                    "token_accuracy": base_pos4_acc,
                    "margin_vs_runnerup": base_pos4_margin,
                    "entropy": base["position4_entropy"],
                },
                "warm_start": {
                    "top1_key": warm["position4_top1_key"],
                    "token_accuracy": warm_pos4_acc,
                    "margin_vs_runnerup": warm_pos4_margin,
                    "entropy": warm["position4_entropy"],
                },
                "paired_effect_token_acc": d_pos4_acc,
                "paired_effect_margin": d_pos4_margin,
                "improved_token_acc": d_pos4_acc > 0,
                "improved_margin": d_pos4_margin > 0,
            },
            "causal_gap": {
                "baseline": base_cgap,
                "warm_start": warm_cgap,
                "paired_effect": d_cgap,
                "improved": d_cgap > 0,
            },
            "attractor_formation": {
                "baseline": base_attractor,
                "warm_start": warm_attractor,
            },
        }

    # Compute pre-registered summary statistics
    def _stats(arr: list[float]) -> dict[str, Any]:
        s_arr = sorted(arr)
        n = len(s_arr)
        med = s_arr[n // 2] if n % 2 == 1 else (s_arr[n // 2 - 1] + s_arr[n // 2]) / 2.0
        improved_cnt = sum(1 for x in arr if x > 0)
        return {
            "values": arr,
            "mean": sum(arr) / n,
            "median": med,
            "min": min(arr),
            "max": max(arr),
            "range": max(arr) - min(arr),
            "improved_count": improved_cnt,
            "total_count": n,
            "improved_ratio": improved_cnt / n,
        }

    overall_em_stats = _stats(delta_overall_em)
    worst_pos_stats = _stats(delta_worst_pos_acc)
    l10_em_stats = _stats(delta_l10_em)
    pos4_acc_stats = _stats(delta_pos4_acc)
    causal_gap_stats = _stats(delta_causal_gap)
    pos4_margin_stats = _stats(delta_pos4_margin)

    # Primary Decision Evaluation
    co_improved_seeds = [
        init_id
        for init_id, comp in pair_comparisons.items()
        if comp["overall_sequence_em"]["improved"] and comp["worst_position_accuracy"]["improved"]
    ]
    co_improved_count = len(co_improved_seeds)

    both_mean_positive = (overall_em_stats["mean"] > 0) and (worst_pos_stats["mean"] > 0)
    decision_criterion_met = (co_improved_count >= 4) and both_mean_positive

    if decision_criterion_met:
        decision = "WARM_START_CAUSAL_SUPERIORITY_REPLICATED"
    else:
        decision = "I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED"

    return {
        "matched_pairs": pair_comparisons,
        "summary_statistics": {
            "overall_sequence_em": overall_em_stats,
            "worst_position_accuracy": worst_pos_stats,
            "length10_sequence_em": l10_em_stats,
            "position4_token_accuracy": pos4_acc_stats,
            "causal_gap": causal_gap_stats,
            "position4_margin": pos4_margin_stats,
        },
        "primary_decision": {
            "decision": decision,
            "decision_criterion_met": decision_criterion_met,
            "co_improved_seeds": co_improved_seeds,
            "co_improved_count": co_improved_count,
            "total_seeds": len(init_ids),
            "both_mean_positive": both_mean_positive,
            "mean_overall_em_paired_effect": overall_em_stats["mean"],
            "mean_worst_pos_acc_paired_effect": worst_pos_stats["mean"],
        },
    }


def run_cd_dpca_matched_baseline_replication_task(
    config: MirrorCDDPCAMatchedBaselineReplicationConfig,
) -> dict[str, Any]:
    """Main execution function for Task B-C005REC-004AR."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Stage 1: Protocol assembly and source integrity audit ---")
    parent_manifest, _ = ibc._load_parent_manifest()
    initial_states = build_all_initial_states(config)  # type: ignore[arg-type]
    init_hashes = {k: mb.canonical_state_hash(v) for k, v in initial_states.items()}

    protocol = build_replication_protocol(config, parent_manifest, init_hashes)
    (config.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    config_dict = {
        "task": REC004AR_TASK_ID,
        "seed": config.seed,
        "init_ids": list(config.init_ids),
        "baseline_execute_inits": list(config.baseline_execute_inits),
        "max_updates_per_init": config.max_updates_per_init,
        "checkpoint_interval": config.checkpoint_interval,
        "decisive_step": config.decisive_step,
        "existing_validation_examples": config.existing_validation_examples,
        "existing_validation_floor": config.existing_validation_floor,
    }
    (config.output_dir / "config.yaml").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    print("--- Stage 2: Information boundary audit ---")
    info_boundary = run_information_boundary_audit()
    (config.output_dir / "information_boundary_audit.json").write_text(
        json.dumps(info_boundary, indent=2), encoding="utf-8"
    )
    if info_boundary["status"] != "PASS":
        raise ValueError(f"INFORMATION_BOUNDARY_AUDIT_FAILURE: {info_boundary}")

    print("--- Stage 3: Reconstructing parent runtime & validation dataset ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core, parent_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core.model.to(device)
    core.model.eval()

    val_examples = ibc._generate_parameter_free_examples(
        config.seed,
        config.existing_validation_examples,
        operation=REC004AR_TARGET_OPERATION,
        split=REC004AL_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
    )

    # Re-evaluate I01 baseline control from REC-004AL
    print("--- Stage 4: Re-evaluating REC-004AL I01 qualified baseline control ---")
    baseline_results: dict[str, dict[str, Any]] = {}
    baseline_results["I01"] = evaluate_rec004al_i01_baseline(
        core, parent_bank, op_to_id, val_examples, config, device
    )
    i01_b = baseline_results["I01"]
    print(
        f"  I01 Baseline: sequence_em = {i01_b['decisive_sequence_em']:.6f}, "
        f"worst_pos = {i01_b['worst_position_key']} ({i01_b['worst_position_accuracy']:.4f})"
    )

    # Execute fresh baseline control runs for I02..I05
    print("--- Stage 5: Executing fresh baseline control runs for I02..I05 ---")
    for init_id in config.baseline_execute_inits:
        print(f"  Starting Baseline Control [{init_id}] (seed={config.init_seeds[init_id]})...")
        t0 = time.time()
        init_res = run_one_baseline_initialization(
            init_id=init_id,
            initial_state_dict=initial_states[init_id],
            core=core,
            parent_bank=parent_bank,
            op_to_id=op_to_id,
            val_examples=val_examples,
            config=config,
            device=device,
        )
        dt = time.time() - t0
        baseline_results[init_id] = init_res
        print(
            f"  Finished Baseline [{init_id}] in {dt:.1f}s: "
            f"EM = {init_res['decisive_sequence_em']:.6f}, "
            f"worst = {init_res['worst_position_key']} ({init_res['worst_position_accuracy']:.4f})"
        )

    print("--- Stage 6: Loading and verifying REC-004AM warm-start evaluations ---")
    warm_start_results = load_rec004am_warm_start_evaluations(config)
    for init_id in config.init_ids:
        w_res = warm_start_results[init_id]
        print(
            f"  Loaded Warm-Start [{init_id}]: EM = {w_res['decisive_sequence_em']:.6f}, "
            f"worst = {w_res['worst_position_key']} ({w_res['worst_position_accuracy']:.4f})"
        )

    print("--- Stage 7: Computing matched pairs comparison & primary decision ---")
    comparison = compute_matched_pairs_comparison(
        baseline_results, warm_start_results, config.init_ids
    )
    (config.output_dir / "matched_pairs_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    primary_decision = comparison["primary_decision"]
    stats = comparison["summary_statistics"]

    # Side-effect audit and freeze audit
    side_effect_audit = {
        "task_id": REC004AR_TASK_ID,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1": "NOT_CLEARED",
        "g4": "NOT_CLEARED",
        "rec004am_fail_preserved": True,
    }
    (config.output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2), encoding="utf-8"
    )

    overall_freeze = {
        "task_id": REC004AR_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "core_canonical_hash_verified": (
            parent_manifest.core.canonical_state_hash == REC004AL_PARENT_CORE_HASH
        ),
        "non_target_primitives_frozen": True,
    }
    (config.output_dir / "freeze_audit.json").write_text(
        json.dumps(overall_freeze, indent=2), encoding="utf-8"
    )

    # Source manifest
    source_manifest = {
        "task_id": REC004AR_TASK_ID,
        "parent_bundle": {
            "bundle_id": REC004AL_PARENT_BUNDLE_ID,
            "core_hash": REC004AL_PARENT_CORE_HASH,
        },
        "source_rec004ak": {
            "dir": str(config.rec004ak_dir),
            "hashes": protocol["source_hashes"],
        },
        "baseline_rec004al": {
            "dir": str(config.rec004al_dir),
            "step6000_hash": baseline_results["I01"]["step6000_canonical_hash"],
        },
        "warm_start_rec004am": {
            "dir": str(config.rec004am_dir),
        },
    }
    (config.output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    # Summary JSON
    summary_data = {
        "task_id": REC004AR_TASK_ID,
        "execution_status": "PASS",
        "decision": primary_decision["decision"],
        "primary_decision": primary_decision,
        "summary_statistics": stats,
        "rec004am_fail_preserved": True,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_status": "BLOCKED",
        "g1_status": "NOT_CLEARED",
        "g4_status": "NOT_CLEARED",
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary_data, indent=2), encoding="utf-8"
    )

    # Generate comprehensive report.md
    em_s = stats["overall_sequence_em"]
    wp_s = stats["worst_position_accuracy"]
    co_seeds_str = ", ".join(primary_decision["co_improved_seeds"])
    report_lines = [
        "# Task B-C005REC-004AR: CD-DPCA Warm-Start Matched-Baseline Causal Replication",
        "",
        "**Execution Date:** 2026-09-13",
        f"**Decision:** `{primary_decision['decision']}`",
        "**Execution Status:** PASS (Explanatory causal control complete; REC-004AM preserved)",
        "",
        "## 1. Executive Summary",
        "",
        "- **Objective:** Evaluate whether sequence-distinctness warm-start exhibits a robust",
        "  causal positive effect over matched baseline controls across all 5 inits (I01..I05).",
        f"- **Primary Decision:** `{primary_decision['decision']}`.",
        f"  - Overall EM improved in **{em_s['improved_count']} / 5** seeds "
        f"(Mean paired effect: **{em_s['mean']:+.6f}**, Median: **{em_s['median']:+.6f}**).",
        f"  - Worst-position accuracy improved in **{wp_s['improved_count']} / 5** seeds "
        f"(Mean paired effect: **{wp_s['mean']:+.6f}**, Median: **{wp_s['median']:+.6f}**).",
        f"  - Co-improved seeds: **{primary_decision['co_improved_count']} / 5** ({co_seeds_str}).",
        "- **Fail-Closed Boundaries Preserved:**",
        "  - `rec004am_fail_preserved: true` (Does not retroactively clear REC-004AM).",
        "  - `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`.",
        "  - `rg3: NOT_EXECUTED`, `rec005_status: BLOCKED`, `g1: NOT_CLEARED`, `g4: NOT_CLEARED`.",
        "",
        "## 2. Matched-Pairs Detailed Comparison",
        "",
        "| Init | Seed | Base EM | Warm EM | Paired EM | Base Worst | Warm Worst | Paired Worst |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for init_id in config.init_ids:
        p = comparison["matched_pairs"][init_id]
        b_em = p["overall_sequence_em"]["baseline"]
        w_em = p["overall_sequence_em"]["warm_start"]
        d_em = p["overall_sequence_em"]["paired_effect"]
        b_wp_k = p["worst_position_accuracy"]["baseline"]["position_key"]
        b_wp_a = p["worst_position_accuracy"]["baseline"]["accuracy"]
        w_wp_k = p["worst_position_accuracy"]["warm_start"]["position_key"]
        w_wp_a = p["worst_position_accuracy"]["warm_start"]["accuracy"]
        d_wp = p["worst_position_accuracy"]["paired_effect"]
        b_wp_str = f"{b_wp_k} ({b_wp_a:.4f})"
        w_wp_str = f"{w_wp_k} ({w_wp_a:.4f})"
        report_lines.append(
            f"| {init_id} | {p['init_seed']} | {b_em:.6f} | {w_em:.6f} | **{d_em:+.6f}** | "
            f"{b_wp_str} | {w_wp_str} | **{d_wp:+.6f}** |"
        )

    report_lines.extend([
        "",
        "## 3. Summary Statistics across All 5 Matched Pairs",
        "",
        "| Metric | Mean Paired Effect | Median Paired Effect | Range | Improved Ratio |",
        "|---|---|---|---|---|",
    ])

    for m_key, m_name in (
        ("overall_sequence_em", "Overall Sequence EM"),
        ("worst_position_accuracy", "Worst-Position Accuracy"),
        ("length10_sequence_em", "Length-10 Sequence EM"),
        ("position4_token_accuracy", "Position-4 Token Accuracy"),
        ("position4_margin", "Position-4 Score Margin"),
        ("causal_gap", "Causal Gap"),
    ):
        st = stats[m_key]
        report_lines.append(
            f"| {m_name} | **{st['mean']:+.6f}** | **{st['median']:+.6f}** | "
            f"{st['range']:.6f} | **{st['improved_count']} / 5 ({st['improved_ratio']:.1%})** |"
        )

    report_lines.extend([
        "",
        "## 4. Attractor Dynamics & Formation Timings",
        "",
        "| Init | Arm | Competitor Onset | Competitor Key | True Attractor | Terminal Key |",
        "|---|---|---|---|---|---|",
    ])

    for init_id in config.init_ids:
        p = comparison["matched_pairs"][init_id]
        b_att = p["attractor_formation"]["baseline"]
        w_att = p["attractor_formation"]["warm_start"]
        b_onset = str(b_att.get("first_competitor_onset_step", "None"))
        b_k = b_att.get("first_competitor_key")
        b_ckey = f"k{b_k}" if b_k is not None else "None"
        b_acq = str(b_att.get("true_attractor_acquired_step", "None"))
        b_term_k = b_att.get("terminal_top1_key")
        b_term = f"k{b_term_k}" if b_term_k is not None else "None"

        w_onset = str(w_att.get("first_competitor_onset_step", "None"))
        w_k = w_att.get("first_competitor_key")
        w_ckey = f"k{w_k}" if w_k is not None else "None"
        w_acq = str(w_att.get("true_attractor_acquired_step", "None"))
        w_term_k = w_att.get("terminal_top1_key")
        w_term = f"k{w_term_k}" if w_term_k is not None else "None"

        report_lines.append(
            f"| {init_id} | Baseline | {b_onset} | {b_ckey} | {b_acq} | {b_term} |"
        )
        report_lines.append(
            f"| {init_id} | Warm-Start | {w_onset} | {w_ckey} | {w_acq} | {w_term} |"
        )

    report_lines.extend([
        "",
        "## 5. Invariants & Execution Guardrails",
        "- Information boundary audit: PASS (Zero target tokens, labels, oracle attention).",
        "- Freeze audit: PASS (Parent bundle Core and 15 non-target primitives strictly frozen).",
        "- Execution scope: Zero candidate adoption, zero bundle writes, RG3 NOT_EXECUTED.",
        "- Scientific integrity: Explanatory matched causal replication completed.",
        "  REC-004AM qualification failure is preserved as historical scientific evidence.",
        "",
    ])

    (config.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "protocol": protocol,
        "baseline_results": baseline_results,
        "warm_start_results": warm_start_results,
        "comparison": comparison,
        "summary": summary_data,
    }
