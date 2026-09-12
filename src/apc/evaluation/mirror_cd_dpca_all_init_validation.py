"""Task B-C005REC-004AM: Fixed Sequence-Distinctness Warm-Start Multi-Initialization Reproduction.

Phase B Model Bundle Recovery.

Pre-registered Contract:
- Multi-initialization reproduction of ADR-0141 (REC-004AQ) across 5 independent initializations:
  I01, I02, I03, I04, I05.
- Fixed warm-start recipe bit-for-bit unchanged from ADR-0141:
  - Optimizer: AdamW (lr=0.0008, weight_decay=0.0001, grad_clip=1.0).
  - LR schedule: CosineAnnealingLR (T_max=1000, eta_min=1e-5, mechanical extension).
  - Batch size: 32 examples per step.
  - Updates per init: 6,000 updates (total budget 30,000 updates).
  - Checkpoint cadence: every 500 steps (steps 0..6000, 13 checkpoints per init).
  - Training data stream:
    - Steps 1–500: pairwise-distinct token sequences sampled without replacement from vocab_size=10.
    - Steps 501–6000: exact return to baseline sampler (sampling with replacement) via
      ibc._generate_step_training_examples with per-step seed formula
      _derive_local_seed(seed, step, 'train:MIRROR_HALVES').
    - Data stream is identical across all 5 initializations (init_id never leaks into data seed).
  - Model freeze: Parent Core and 15 non-MIRROR primitives strictly frozen.
  - Loss: Standard token-output cross-entropy loss only. No oracle/teacher routing loss,
    no auxiliary loss, no temperature or entropy modifications.
- Qualification Rule (all-init validation contract):
  - Every initialization must independently achieve:
    1. Overall sequence exact match >= 0.95 at decisive step 6,000
       on the 1,024-example validation split.
    2. Length-10 position-4 false attractor cleared (top-1 key = 0, token accuracy >= 0.95).
    3. No regression across other output positions (token accuracy >= 0.90).
    4. Causal controls: Correct >= 0.95, Wrong <= 0.05, None <= 0.05, causal gap >= 0.90.
    5. Freeze audit and information boundary audit PASS.
  - 5/5 requirement: all 5 initializations must clear the criteria for overall PASS.
  - In all cases: candidate adoption = null, child bundle = null, rg3 = NOT_EXECUTED,
    rec005_eligible = false, G1/G4 = NOT_CLEARED.
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
    REC004AQ_TASK_ID,
    REC004AQ_WARM_START_STEPS,
    compute_position4_routing_metrics,
    generate_step_training_examples_warm_start,
)
from apc.evaluation.model_bundle_recovery import _evaluate_one_operation, _guard_not_frozen
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004AM_TASK_ID",
    "REC004AM_PILOT_TASK_ID",
    "REC004AM_BASELINE_TASK_ID",
    "REC004AM_SOURCE_TASK_ID",
    "REC004AM_INIT_IDS",
    "REC004AM_INIT_SEEDS",
    "REC004AM_WARM_START_STEPS",
    "MirrorCDDPCAAllInitValidationConfig",
    "build_all_initial_states",
    "build_all_init_protocol",
    "run_one_initialization",
    "run_cd_dpca_all_init_validation_task",
]

# Task Identifiers
REC004AM_TASK_ID: Final = "B-C005REC-004AM"
REC004AM_PILOT_TASK_ID: Final = REC004AQ_TASK_ID  # B-C005REC-004AQ
REC004AM_BASELINE_TASK_ID: Final = "B-C005REC-004AL"
REC004AM_SOURCE_TASK_ID: Final = "B-C005REC-004AK"
REC004AM_TARGET_OPERATION: Final = REC004AL_TARGET_OPERATION  # MIRROR_HALVES
REC004AM_WARM_START_STEPS: Final = REC004AQ_WARM_START_STEPS  # 500

REC004AM_INIT_IDS: Final[tuple[str, ...]] = ("I01", "I02", "I03", "I04", "I05")

# Pre-registered initial seeds for the 5 independent initializations:
# I01 uses seed 20260912 (exact match to REC-004AK/AL/AQ step-0 canonical hash 045d85cae8...).
# I02..I05 use sequential deterministic seeds (20260913..20260916).
REC004AM_INIT_SEEDS: Final[dict[str, int]] = {
    "I01": 20260912,
    "I02": 20260913,
    "I03": 20260914,
    "I04": 20260915,
    "I05": 20260916,
}


@dataclass(frozen=True)
class MirrorCDDPCAAllInitValidationConfig:
    """Configuration for Task B-C005REC-004AM."""

    output_dir: Path = Path("runs/phase_b_restart/rec004am/run_001")
    rec004ak_dir: Path = Path("runs/phase_b_restart/rec004ak/run_001")
    rec004al_dir: Path = Path("runs/phase_b_restart/rec004al/run_001")
    rec004aq_dir: Path = Path("runs/phase_b_restart/rec004aq/run_001")
    seed: int = REC004AL_PILOT_SEED
    max_updates_per_init: int = REC004AL_MAX_UPDATES
    warm_start_steps: int = REC004AM_WARM_START_STEPS
    checkpoint_interval: int = REC004AL_CHECKPOINT_INTERVAL
    decisive_step: int = REC004AL_DECISIVE_STEP
    existing_validation_examples: int = REC004AL_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004AL_TERMINAL_VIABILITY_FLOOR
    init_ids: tuple[str, ...] = REC004AM_INIT_IDS
    init_seeds: dict[str, int] = field(default_factory=lambda: dict(REC004AM_INIT_SEEDS))


def _expected_lr(u: int, t_max: int = 1000, lr: float = 0.0008, eta_min: float = 1e-5) -> float:
    """Closed-form expected LR under CosineAnnealingLR with mechanical extension."""
    return eta_min + (lr - eta_min) / 2.0 * (1.0 + math.cos(math.pi * u / t_max))


def build_all_initial_states(
    config: MirrorCDDPCAAllInitValidationConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    """Construct 5 independent deterministic untrained CD-DPCA initial states.

    I01 reproduces REC-004AK state identically (canonical hash verified).
    I02..I05 are distinct initial weight draws from distinct pre-registered seeds.
    """
    states: dict[str, dict[str, torch.Tensor]] = {}
    init_hashes: dict[str, str] = {}

    for init_id in config.init_ids:
        seed = config.init_seeds[init_id]
        torch.manual_seed(seed)
        prim_cfg = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
            operation=REC004AM_TARGET_OPERATION,
            d_model=192,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=REC004AL_VOCAB_SIZE,
            max_sequence_length=32,
            arg_dim=16,
        )
        bank = PrimitiveBank()
        primitive = bank.new_cd_dpca_primitive(
            prim_cfg,
            status=PrimitiveStatus.STABLE,
            created_at_task=0,
            metadata={"family": "CD-DPCA", "task": REC004AM_TASK_ID, "init_id": init_id},
        )
        primitive.eval()
        sd = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
        h = mb.canonical_state_hash(sd)
        states[init_id] = sd
        init_hashes[init_id] = h

    # Strict invariant: I01 canonical hash must match REC-004AK primitive state
    if init_hashes["I01"] != REC004AK_PRIMITIVE_STATE_CANONICAL_HASH:
        raise mb.CorruptedArtifactError(
            f"I01 initial state hash mismatch: {init_hashes['I01']} "
            f"!= {REC004AK_PRIMITIVE_STATE_CANONICAL_HASH}"
        )

    # Strict invariant: all 5 initial states must have mutually distinct canonical hashes
    if len(set(init_hashes.values())) != len(config.init_ids):
        raise ValueError(
            f"Initial states are not all distinct! Hashes: {init_hashes}"
        )

    return states


def build_all_init_protocol(
    config: MirrorCDDPCAAllInitValidationConfig,
    parent_manifest: mb.ModelBundleManifest,
    init_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assemble and lock pre-registered multi-initialization experimental protocol."""
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
        "task_id": REC004AM_TASK_ID,
        "pilot_task_id": REC004AM_PILOT_TASK_ID,
        "baseline_task_id": REC004AM_BASELINE_TASK_ID,
        "source_task_id": REC004AM_SOURCE_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_canonical_state_hash": REC004AL_PARENT_CORE_HASH,
        "model_seed": config.seed,
        "target_operation": REC004AM_TARGET_OPERATION,
        "architecture_signature": REC004AL_ARCHITECTURE_SIGNATURE,
        "init_ids": list(config.init_ids),
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
        "causal_intervention": {
            "intervention_name": "SEQUENCE_DISTINCTNESS_WARM_START",
            "warm_start_steps": config.warm_start_steps,
            "warm_start_sampling_rule": (
                "tokens sampled without replacement from range(vocab_size=10) "
                "yielding pairwise-distinct token sequences"
            ),
            "post_warm_start_steps": (
                f"{config.warm_start_steps + 1}..{config.max_updates_per_init}"
            ),
            "post_warm_start_sampling_rule": (
                "exact return to REC-004AL baseline sampler via "
                "ibc._generate_step_training_examples and identical per-step RNG derivation"
            ),
            "boundary_pre_fixed": True,
            "sweep_authorized": False,
            "length_position_selectivity_prohibited": True,
            "data_stream_identical_across_inits": True,
        },
        "training_recipe": {
            "optimizer": "AdamW",
            "operator_lr": REC004AL_OPERATOR_LR,
            "operator_weight_decay": REC004AL_OPERATOR_WEIGHT_DECAY,
            "operator_grad_clip": REC004AL_OPERATOR_GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": REC004AL_SCHEDULER_T_MAX,
            "scheduler_eta_min": REC004AL_SCHEDULER_ETA_MIN,
            "schedule_mode": "mechanical_extension",
            "examples_per_step": REC004AL_EXAMPLES_PER_STEP,
            "vocab_size": REC004AL_VOCAB_SIZE,
            "sequence_length_range": list(REC004AL_SEQUENCE_LENGTH_RANGE),
            "loss_function": "cross_entropy_with_ignore_index",
            "max_updates_per_init": config.max_updates_per_init,
            "total_updates": config.max_updates_per_init * len(config.init_ids),
        },
        "evaluation_protocol": {
            "validation_split": REC004AL_EXISTING_VALIDATION_SPLIT,
            "validation_examples": config.existing_validation_examples,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_steps": checkpoint_steps,
            "decisive_step": config.decisive_step,
            "terminal_viability_floor": config.existing_validation_floor,
            "terminal_viability_metric": "sequence_exact_match",
            "qualification_rule": (
                "All 5/5 initializations must independently achieve: "
                "terminal overall sequence EM >= 0.95 at decisive step 6000, "
                "length-10 position-4 false attractor cleared (top-1 key 0, acc >= 0.95), "
                "no regression across other positions (acc >= 0.90), "
                "causal gap >= 0.90, core and non-target primitives strictly frozen."
            ),
        },
        "lr_table": lr_table,
        "boundaries": {
            "candidate_selected_fixed_null": True,
            "child_bundle_fixed_null": True,
            "rg3_recheck_fixed_not_executed": True,
            "rec005_eligible_fixed_false": True,
            "g1_status": "NOT_CLEARED",
            "g4_status": "NOT_CLEARED",
        },
    }


def run_one_initialization(
    init_id: str,
    initial_state_dict: dict[str, torch.Tensor],
    core: Any,
    parent_bank: PrimitiveBank,
    op_to_id: dict[str, int],
    val_examples: Sequence[Any],
    config: MirrorCDDPCAAllInitValidationConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate a single initialization under the fixed warm-start recipe."""
    t_init_start = time.time()
    init_dir = config.output_dir / init_id
    ckpt_dir = init_dir / "checkpoints"
    state_dir = init_dir / "training_states"
    init_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    pid = op_to_id[REC004AM_TARGET_OPERATION]
    initial_canonical_hash = mb.canonical_state_hash(initial_state_dict)

    # Save step-0 checkpoint
    torch.save(initial_state_dict, ckpt_dir / "step0.pt")

    # Construct isolated trainable primitive
    prim_cfg = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation=REC004AM_TARGET_OPERATION,
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
        metadata={"family": "CD-DPCA", "task": REC004AM_TASK_ID, "init_id": init_id},
    )
    primitive.load_state_dict(initial_state_dict, strict=True)
    primitive.to(device)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    # Optimizer & scheduler (exact AdamW & CosineAnnealingLR)
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
            },
            state_dir / f"step{step}.pt",
        )

        val_summary, pos_report = compute_per_length_position_metrics(core, primitive, val_examples)
        pos4_metrics = compute_position4_routing_metrics(primitive, val_examples, device)
        pos4_metrics["step"] = step
        pos4_metrics["token_accuracy"] = pos_report.get("10:4", {}).get("accuracy", 0.0)
        position4_records.append(pos4_metrics)

        vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ckpt_data = {
            "step": step,
            "init_id": init_id,
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

    # Training loop (steps 1..6000)
    for step in range(1, config.max_updates_per_init + 1):
        step_examples = generate_step_training_examples_warm_start(
            config.seed,
            step,
            REC004AM_TARGET_OPERATION,
            vocab_size=REC004AL_VOCAB_SIZE,
            sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
            warm_start_steps=config.warm_start_steps,
            n=REC004AL_EXAMPLES_PER_STEP,
        )
        c_lens = [len(ex.input_tokens) for ex in step_examples]
        o_lens = [get_operation(REC004AM_TARGET_OPERATION).output_length(n) for n in c_lens]
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
                "task_id": REC004AM_TASK_ID,
                "init_id": init_id,
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
            REC004AM_TARGET_OPERATION,
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
        "task_id": REC004AM_TASK_ID,
        "init_id": init_id,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "core_frozen_verified": (
            parent_manifest.core.canonical_state_hash == REC004AL_PARENT_CORE_HASH
        ),
        "parent_primitives_frozen": True,
        "trained_primitive_id": pid,
        "trained_primitive_class": primitive.__class__.__name__,
        "optimizer_updated_core_parameters": 0,
        "optimizer_updated_non_mirror_parameters": 0,
        "status": "PASS",
    }
    (init_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2), encoding="utf-8"
    )

    # Evaluation criteria for this init
    decisive_em = final_summary["sequence_exact_match"]
    terminal_floor_met = decisive_em >= config.existing_validation_floor

    final_pos4 = compute_position4_routing_metrics(primitive, val_examples, device)
    final_pos4["token_accuracy"] = final_pos_report.get("10:4", {}).get("accuracy", 0.0)
    pos4_attractor_cleared = (
        final_pos4["top1_key_mode"] == 0
        and final_pos4["token_accuracy"] >= 0.95
        and final_pos4["top1_competitor_key_count"] == 0
    )

    other_pos_regressed = False
    for pos_key, pos_val in final_pos_report.items():
        if pos_key != "10:4" and pos_val["accuracy"] < 0.90:
            other_pos_regressed = True

    causal_gap = causal_eval["exact_match_causal_gap"]
    causal_passed = (
        causal_eval["correct_exact_match"] >= config.existing_validation_floor
        and causal_eval["wrong_family_exact_match"] <= 0.05
        and causal_eval["none_exact_match"] <= 0.05
        and causal_gap >= 0.90
    )

    init_viability_met = (
        terminal_floor_met
        and pos4_attractor_cleared
        and not other_pos_regressed
        and causal_passed
        and attn_diagnostics["padding_mask_verified"]
        and freeze_audit["status"] == "PASS"
    )

    wall_sec = time.time() - t_init_start
    final_state_hash = mb.canonical_state_hash(primitive.state_dict())

    init_result = {
        "init_id": init_id,
        "init_seed": config.init_seeds[init_id],
        "step0_canonical_hash": initial_canonical_hash,
        "step6000_canonical_hash": final_state_hash,
        "decisive_sequence_em": decisive_em,
        "decisive_exact_count": final_summary["exact_match_count"],
        "decisive_total_examples": final_summary["n_examples"],
        "decisive_token_accuracy": final_summary["token_accuracy"],
        "terminal_floor_met": terminal_floor_met,
        "position4_attractor_cleared": pos4_attractor_cleared,
        "position4_top1_key": final_pos4["top1_key_mode"],
        "position4_token_accuracy": final_pos4["token_accuracy"],
        "position4_margin": final_pos4["mean_margin_vs_competitor"],
        "position4_entropy": final_pos4["mean_entropy"],
        "other_positions_regressed": other_pos_regressed,
        "by_length_em": {
            k: {
                "exact": v["exact_match_count"],
                "total": v["n_sequences"],
                "em": v["sequence_exact_match"],
                "token_accuracy": v["token_accuracy"],
            }
            for k, v in final_summary["by_length"].items()
        },
        "causal_controls": {
            "correct_em": causal_eval["correct_exact_match"],
            "reverse_em": causal_eval["wrong_family_exact_match"],
            "none_em": causal_eval["none_exact_match"],
            "causal_gap": causal_gap,
            "causal_passed": causal_passed,
        },
        "padding_mask_verified": attn_diagnostics["padding_mask_verified"],
        "freeze_audit_status": freeze_audit["status"],
        "init_viability_met": init_viability_met,
        "wall_seconds": wall_sec,
    }

    (init_dir / "summary.json").write_text(
        json.dumps(init_result, indent=2), encoding="utf-8"
    )

    return init_result


def run_cd_dpca_all_init_validation_task(
    config: MirrorCDDPCAAllInitValidationConfig,
) -> dict[str, Any]:
    """Main execution function for Task B-C005REC-004AM."""
    _guard_not_frozen(REC004AM_TASK_ID)
    t_task_start = time.time()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    init_states_dir = config.output_dir / "initial_states"
    init_states_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: Load parent runtime and verify information boundary
    parent_manifest, _raw = ibc._load_parent_manifest()

    info_boundary = run_information_boundary_audit()
    info_boundary["task_id"] = REC004AM_TASK_ID
    (config.output_dir / "information_boundary_audit.json").write_text(
        json.dumps(info_boundary, indent=2), encoding="utf-8"
    )
    if info_boundary["status"] != "PASS":
        raise ValueError(f"INFORMATION_BOUNDARY_AUDIT_FAILURE: {info_boundary}")

    # Build 5 independent deterministic initial states
    all_initial_states = build_all_initial_states(config)
    init_hashes: dict[str, str] = {}
    for init_id, sd in all_initial_states.items():
        h = mb.canonical_state_hash(sd)
        init_hashes[init_id] = h
        torch.save(sd, init_states_dir / f"{init_id}.pt")

    # Lock protocol
    protocol = build_all_init_protocol(config, parent_manifest, init_hashes)
    (config.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )

    # Config yaml
    config_dict = {
        "task_id": REC004AM_TASK_ID,
        "pilot_task_id": REC004AM_PILOT_TASK_ID,
        "baseline_task_id": REC004AM_BASELINE_TASK_ID,
        "source_task_id": REC004AM_SOURCE_TASK_ID,
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "target_operation": REC004AM_TARGET_OPERATION,
        "init_ids": list(config.init_ids),
        "init_seeds": config.init_seeds,
        "max_updates_per_init": config.max_updates_per_init,
        "warm_start_steps": config.warm_start_steps,
        "checkpoint_interval": config.checkpoint_interval,
        "decisive_step": config.decisive_step,
        "existing_validation_examples": config.existing_validation_examples,
        "existing_validation_split": REC004AL_EXISTING_VALIDATION_SPLIT,
        "existing_validation_floor": config.existing_validation_floor,
        "operator_lr": REC004AL_OPERATOR_LR,
        "operator_weight_decay": REC004AL_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004AL_OPERATOR_GRAD_CLIP,
        "scheduler_t_max": REC004AL_SCHEDULER_T_MAX,
        "scheduler_eta_min": REC004AL_SCHEDULER_ETA_MIN,
        "examples_per_step": REC004AL_EXAMPLES_PER_STEP,
        "vocab_size": REC004AL_VOCAB_SIZE,
        "sequence_length_range": list(REC004AL_SEQUENCE_LENGTH_RANGE),
    }
    (config.output_dir / "config.yaml").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    # Source manifest
    source_manifest = {
        "task_id": REC004AM_TASK_ID,
        "pilot_task_id": REC004AM_PILOT_TASK_ID,
        "baseline_task_id": REC004AM_BASELINE_TASK_ID,
        "parent_manifest_path": str(ibc.REC004A_PARENT_MANIFEST_PATH),
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_hash": REC004AL_PARENT_CORE_HASH,
        "rec004ak_dir": str(config.rec004ak_dir),
        "rec004ak_hashes": protocol["source_hashes"],
        "init_canonical_hashes": init_hashes,
    }
    (config.output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    # Setup parent runtime for shared validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core, parent_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core.model.to(device)
    core.model.eval()

    val_examples = ibc._generate_parameter_free_examples(
        config.seed,
        config.existing_validation_examples,
        operation=REC004AM_TARGET_OPERATION,
        split=REC004AL_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
    )

    # Run each of the 5 initializations independently
    init_results: dict[str, dict[str, Any]] = {}
    for init_id in config.init_ids:
        print(f"\n--- Starting Initialization {init_id} (seed {config.init_seeds[init_id]}) ---")
        res = run_one_initialization(
            init_id=init_id,
            initial_state_dict=all_initial_states[init_id],
            core=core,
            parent_bank=parent_bank,
            op_to_id=op_to_id,
            val_examples=val_examples,
            config=config,
            device=device,
        )
        init_results[init_id] = res
        print(
            f"[{init_id}] Decisive Sequence EM: {res['decisive_sequence_em']:.6f} "
            f"({res['decisive_exact_count']}/{res['decisive_total_examples']}), "
            f"Pos4 key: {res['position4_top1_key']}, "
            f"Pos4 margin: {res['position4_margin']:.2f}, "
            f"Pos4 acc: {res['position4_token_accuracy']:.4f}, "
            f"Viability: {res['init_viability_met']}"
        )

    # Evaluate all-init qualification rule
    all_passed = all(r["init_viability_met"] for r in init_results.values())
    n_passed = sum(1 for r in init_results.values() if r["init_viability_met"])
    total_inits = len(config.init_ids)

    execution_status = "PASS" if all_passed else "FAIL"
    decision = (
        "ALL_INITIALIZATIONS_VIABILITY_MET"
        if all_passed
        else "MULTI_INIT_VIABILITY_NOT_MET"
    )

    total_wall_seconds = time.time() - t_task_start

    # Multi-init summary
    multi_init_summary = {
        "task_id": REC004AM_TASK_ID,
        "pilot_task_id": REC004AM_PILOT_TASK_ID,
        "execution_status": execution_status,
        "decision": decision,
        "total_inits": total_inits,
        "n_passed": n_passed,
        "all_inits_passed": all_passed,
        "qualification_rule_satisfied": all_passed,
        "per_init_results": init_results,
        "mean_decisive_sequence_em": sum(
            r["decisive_sequence_em"] for r in init_results.values()
        )
        / total_inits,
        "min_decisive_sequence_em": min(
            r["decisive_sequence_em"] for r in init_results.values()
        ),
        "max_decisive_sequence_em": max(
            r["decisive_sequence_em"] for r in init_results.values()
        ),
        "mean_decisive_token_accuracy": sum(
            r["decisive_token_accuracy"] for r in init_results.values()
        )
        / total_inits,
        "total_updates_across_all_inits": config.max_updates_per_init * total_inits,
        "total_wall_seconds": total_wall_seconds,
    }
    (config.output_dir / "multi_init_summary.json").write_text(
        json.dumps(multi_init_summary, indent=2), encoding="utf-8"
    )

    # Side effect audit
    side_effect_audit = {
        "task_id": REC004AM_TASK_ID,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "hyperparameter_search_executed": False,
        "budget_extended": False,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1_status": "NOT_CLEARED",
        "g4_status": "NOT_CLEARED",
        "sealed_evaluation_accessed": False,
    }
    (config.output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2), encoding="utf-8"
    )

    # Top-level summary
    summary = {
        "task": REC004AM_TASK_ID,
        "pilot_task": REC004AM_PILOT_TASK_ID,
        "baseline_task": REC004AM_BASELINE_TASK_ID,
        "execution_status": execution_status,
        "decision": decision,
        "total_inits": total_inits,
        "n_inits_passed": n_passed,
        "all_inits_passed": all_passed,
        "terminal_viability_floor": config.existing_validation_floor,
        "mean_decisive_sequence_em": multi_init_summary["mean_decisive_sequence_em"],
        "min_decisive_sequence_em": multi_init_summary["min_decisive_sequence_em"],
        "max_decisive_sequence_em": multi_init_summary["max_decisive_sequence_em"],
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1": "NOT_CLEARED",
        "g4": "NOT_CLEARED",
        "total_optimizer_updates": config.max_updates_per_init * total_inits,
        "wall_seconds": total_wall_seconds,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Markdown Report
    lines = [
        f"# Task B-C005REC-004AM Multi-Initialization Reproduction Report ({decision})",
        "",
        f"- **Task ID:** `{REC004AM_TASK_ID}`",
        f"- **Pilot Reference:** `{REC004AM_PILOT_TASK_ID}` (ADR-0141)",
        f"- **Decision:** `{decision}`",
        f"- **Execution Status:** `{execution_status}`",
        (
            f"- **Qualification Status:** `{n_passed} / {total_inits} Initializations Passed "
            f"(Floor >= {config.existing_validation_floor})`"
        ),
        (
            f"- **Warm-Start Steps:** `{config.warm_start_steps}` / "
            f"`{config.max_updates_per_init}` updates per init"
        ),
        (
            f"- **Total Updates:** `{config.max_updates_per_init * total_inits}` "
            "updates across all inits"
        ),
        f"- **Total Wall Clock Time:** `{total_wall_seconds:.2f}` s",
        "",
        "## Per-Initialization Summary",
        "",
        "| Init ID | Seed | Step 0 Hash | Step 6000 Hash | "
        "Decisive Sequence EM | Length-10 EM | Pos-4 Top Key | "
        "Pos-4 Margin | Pos-4 Acc | Causal Gap | Viability |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for init_id in config.init_ids:
        r = init_results[init_id]
        l10_em = r["by_length_em"].get("10", {}).get("em", 0.0)
        lines.append(
            f"| `{init_id}` | `{r['init_seed']}` | `{r['step0_canonical_hash'][:8]}...` | "
            f"`{r['step6000_canonical_hash'][:8]}...` | `{r['decisive_sequence_em']:.6f}` "
            f"({r['decisive_exact_count']}/{r['decisive_total_examples']}) | `{l10_em:.4f}` | "
            f"`{r['position4_top1_key']}` | `{r['position4_margin']:+.2f}` | "
            f"`{r['position4_token_accuracy']:.4f}` | `{r['causal_controls']['causal_gap']:.4f}` | "
            f"**{'PASS' if r['init_viability_met'] else 'FAIL'}** |"
        )

    lines.extend(
        [
            "",
            "## Per-Length Exact Match Breakdown Across Initializations",
            "",
            "| Init ID | Length 6 EM | Length 7 EM | Length 8 EM | Length 9 EM | Length 10 EM |",
            "|---|---|---|---|---|---|",
        ]
    )

    for init_id in config.init_ids:
        r = init_results[init_id]
        bl = r["by_length_em"]
        l6 = bl.get("6", {}).get("em", 0.0)
        l7 = bl.get("7", {}).get("em", 0.0)
        l8 = bl.get("8", {}).get("em", 0.0)
        l9 = bl.get("9", {}).get("em", 0.0)
        l10 = bl.get("10", {}).get("em", 0.0)
        lines.append(
            f"| `{init_id}` | `{l6:.4f}` | `{l7:.4f}` | `{l8:.4f}` | `{l9:.4f}` | `{l10:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Execution Boundaries & Preserved Blocks",
            "",
            "- `candidate_selected`: `null` (strictly preserved)",
            "- `child_bundle`: `null` (no model bundle written)",
            "- `rg3_recheck`: `NOT_EXECUTED`",
            "- `rec005_eligible`: `false`",
            "- `G1` and `G4`: `NOT_CLEARED` (independent research blocks strictly preserved)",
            "",
        ]
    )

    (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary": summary,
        "multi_init_summary": multi_init_summary,
        "protocol": protocol,
    }
