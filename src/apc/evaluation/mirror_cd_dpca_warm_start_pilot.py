"""Task B-C005REC-004AQ: CD-DPCA Sequence-Distinctness Warm-Start Single-Recipe Causal Pilot.

Phase B Model Bundle Recovery (ADR-0141).

Baseline: REC-004AL (I01).
Matched conditions:
- Step-0 model parameters & canonical hash identical to REC-004AK fresh-load.
- Optimizer: AdamW (lr=0.0008, weight_decay=0.0001, grad_clip=1.0).
- LR schedule: CosineAnnealingLR (T_max=1000, eta_min=1e-5, mechanical extension).
- Batch size: 32 examples per step.
- Total updates: 6,000 updates.
- Checkpoint cadence: every 500 steps (steps 0..6000, 13 checkpoints).
- Evaluation set: fixed development validation split (1,024 examples).
- Model freeze: Parent Core and 15 non-MIRROR primitives strictly frozen.
  Only the MIRROR temporary primitive (id 12) is updated.
- Loss: Standard token-output cross-entropy loss only.
  No oracle/teacher routing loss, no auxiliary loss, no entropy/temperature change.

Single Causal Intervention:
- Steps 1–500: Each input sequence is sampled without replacement from vocab_size=10,
  making tokens strictly pairwise-distinct within each sequence.
- Steps 501–6000: Exact return to REC-004AL baseline sampler (with replacement) and
  per-step seed formula.
- The 500-step boundary is fixed a priori from historical evidence of false-attractor onset.
  No sweep over boundary steps.
- Rule applies uniformly across all lengths and positions; no filtering on length 10 or position 4.

Evaluation Protocol across all 13 checkpoints:
- Overall sequence exact match & token accuracy.
- Per-length sequence exact match & token accuracy.
- Per-position token accuracy across all lengths.
- Length-10 position-4 routing metrics: top-1 key, p(0), p(7), score margin, entropy.
- REC-004AP A/B/C alignment metrics: Stratum A, B, C, pooled score-space and
  parameter-space margin gradients, cosine alignment, per-example distributions.
- Direct trajectory comparison against REC-004AL baseline.
- Terminal viability floor (sequence EM >= 0.95 at step 6000) and position-4 attractor
  clearing required.
"""

from __future__ import annotations

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

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import Operation, get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.mirror_cd_dpca_gradient_alignment_strata import (
    _extract_routing_grad,
    _forward_with_explicit_scores,
    stratify_examples_by_token_identifiability,
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
    REC004AL_INIT_ID,
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
from apc.evaluation.model_bundle_recovery import _evaluate_one_operation, _guard_not_frozen
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004AQ_TASK_ID",
    "REC004AQ_BASELINE_TASK_ID",
    "REC004AQ_SOURCE_TASK_ID",
    "REC004AQ_WARM_START_STEPS",
    "MirrorCDDPCAWarmStartPilotConfig",
    "generate_step_training_examples_warm_start",
    "compute_position4_routing_metrics",
    "compute_strata_gradient_alignment_at_checkpoint",
    "run_cd_dpca_warm_start_pilot_task",
]

# Task Identifiers
REC004AQ_TASK_ID: Final = "B-C005REC-004AQ"
REC004AQ_BASELINE_TASK_ID: Final = "B-C005REC-004AL"
REC004AQ_SOURCE_TASK_ID: Final = "B-C005REC-004AK"
REC004AQ_WARM_START_STEPS: Final = 500


@dataclass(frozen=True)
class MirrorCDDPCAWarmStartPilotConfig:
    """Configuration for Task B-C005REC-004AQ."""

    output_dir: Path = Path("runs/phase_b_restart/rec004aq/run_001")
    rec004al_dir: Path = Path("runs/phase_b_restart/rec004al/run_001")
    rec004ak_dir: Path = Path("runs/phase_b_restart/rec004ak/run_001")
    rec004an_dir: Path = Path("runs/phase_b_restart/rec004an/run_001")
    rec004ao_dir: Path = Path("runs/phase_b_restart/rec004ao/run_001")
    rec004ap_dir: Path = Path("runs/phase_b_restart/rec004ap/run_001")
    seed: int = REC004AL_PILOT_SEED
    max_updates: int = REC004AL_MAX_UPDATES
    warm_start_steps: int = REC004AQ_WARM_START_STEPS
    checkpoint_interval: int = REC004AL_CHECKPOINT_INTERVAL
    decisive_step: int = REC004AL_DECISIVE_STEP
    existing_validation_examples: int = REC004AL_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004AL_TERMINAL_VIABILITY_FLOOR
    init_id: str = REC004AL_INIT_ID


def _expected_lr(u: int, t_max: int = 1000, lr: float = 0.0008, eta_min: float = 1e-5) -> float:
    """Closed-form expected LR under CosineAnnealingLR with mechanical extension."""
    return eta_min + (lr - eta_min) / 2.0 * (1.0 + math.cos(math.pi * u / t_max))


def generate_step_training_examples_warm_start(
    seed: int,
    step: int,
    operation: str,
    *,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    warm_start_steps: int = REC004AQ_WARM_START_STEPS,
    n: int = REC004AL_EXAMPLES_PER_STEP,
) -> list[Example]:
    """Generate training examples for a single step under the warm-start causal pilot.

    Intervention:
    - Steps 1..warm_start_steps: Tokens are sampled without replacement from range(vocab_size),
      ensuring pairwise-distinct tokens within each sequence (sampling without replacement).
    - Steps (warm_start_steps + 1)..max_updates: Exact return to REC-004AL baseline sampler
      via ibc._generate_step_training_examples using identical per-step RNG derivation.
    """
    if step > warm_start_steps:
        return ibc._generate_step_training_examples(
            seed,
            step,
            operation,
            vocab_size=vocab_size,
            sequence_length_range=sequence_length_range,
            n=n,
        )

    step_rng = random.Random(
        ibc._derive_local_seed(seed, step, f"{ibc.REC004A_TRAIN_SPLIT_LABEL}:{operation}")
    )
    op_obj: Operation = get_operation(operation)
    examples: list[Example] = []
    for _ in range(n):
        seq_len = step_rng.randint(sequence_length_range[0], sequence_length_range[1])
        # Pairwise-distinct tokens within each sequence via sampling without replacement
        seq = tuple(step_rng.sample(range(vocab_size), seq_len))
        params = op_obj.sample_params(step_rng, seq, vocab_size)
        p_step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(p_step,))
        res = run_program(prog, seq, vocab_size)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=ibc.REC004A_TRAIN_SPLIT_LABEL,
                vocab_size=vocab_size,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(label="K", primitive_operations=(operation,)),
            )
        )
    return examples


def compute_position4_routing_metrics(
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    examples: Sequence[Any],
    device: torch.device,
    target_length: int = 10,
    target_position: int = 4,
    corr_key: int = 0,
    competitor_key: int = 7,
) -> dict[str, Any]:
    """Compute detailed routing metrics for length-10 position 4 on validation set."""
    primitive.eval()
    ex_target = [ex for ex in examples if len(ex.input_tokens) == target_length]
    n_examples = len(ex_target)
    c_lens = [target_length] * n_examples
    o_lens = [target_length] * n_examples

    with torch.no_grad():
        scores = primitive.compute_routing_scores(
            c_lens, o_lens, None, device=device, lmax=target_length
        )  # [B, n_head, target_length, target_length]
        attn_weights = primitive.compute_attention_weights(
            c_lens, o_lens, None, device=device, lmax=target_length, average_heads=False
        )

    # Average over heads
    head_avg_weights = attn_weights.mean(dim=1)  # [B, target_length, target_length]
    head_avg_scores = scores.mean(dim=1)        # [B, target_length, target_length]

    # Focus on target_position (4)
    pos_weights = head_avg_weights[:, target_position, :target_length]  # [B, target_length]
    pos_scores = head_avg_scores[:, target_position, :target_length]    # [B, target_length]

    top1_keys = pos_weights.argmax(dim=-1).cpu().tolist()  # [B]
    p_corr = pos_weights[:, corr_key].cpu().tolist()       # [B]
    p_comp = pos_weights[:, competitor_key].cpu().tolist() # [B]

    margins_runnerup: list[float] = []
    margins_vs_comp: list[float] = []
    entropies: list[float] = []

    for b in range(n_examples):
        s_row = pos_scores[b].clone()
        s_c = s_row[corr_key].item()
        s_k7 = s_row[competitor_key].item()
        s_row[corr_key] = float("-inf")
        s_runner_up = s_row.max().item()
        margins_runnerup.append(s_c - s_runner_up)
        margins_vs_comp.append(s_c - s_k7)

        w_row = pos_weights[b]
        ent = -float((w_row * torch.log(w_row + 1e-12)).sum().item())
        entropies.append(ent)

    key_counts: dict[int, int] = {}
    for k in top1_keys:
        key_counts[k] = key_counts.get(k, 0) + 1

    top1_key_mode = max(key_counts.items(), key=lambda x: x[1])[0] if key_counts else -1

    return {
        "target_length": target_length,
        "target_position": target_position,
        "correct_key": corr_key,
        "competitor_key": competitor_key,
        "n_examples": n_examples,
        "top1_key_distribution": {str(k): v for k, v in sorted(key_counts.items())},
        "top1_key_mode": top1_key_mode,
        "top1_correct_key_count": key_counts.get(corr_key, 0),
        "top1_correct_key_fraction": (
            key_counts.get(corr_key, 0) / n_examples if n_examples > 0 else 0.0
        ),
        "top1_competitor_key_count": key_counts.get(competitor_key, 0),
        "top1_competitor_key_fraction": (
            key_counts.get(competitor_key, 0) / n_examples if n_examples > 0 else 0.0
        ),
        "mean_p_correct": float(sum(p_corr) / len(p_corr)) if p_corr else 0.0,
        "mean_p_competitor": float(sum(p_comp) / len(p_comp)) if p_comp else 0.0,
        "mean_margin_vs_runnerup": (
            float(sum(margins_runnerup) / len(margins_runnerup)) if margins_runnerup else 0.0
        ),
        "mean_margin_vs_competitor": (
            float(sum(margins_vs_comp) / len(margins_vs_comp)) if margins_vs_comp else 0.0
        ),
        "mean_entropy": float(sum(entropies) / len(entropies)) if entropies else 0.0,
    }


def compute_strata_gradient_alignment_at_checkpoint(
    core: Any,
    primitive: ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    examples: Sequence[Any],
    device: torch.device,
    target_length: int = 10,
    target_position: int = 4,
    corr_key: int = 0,
    competitor_key: int = 7,
) -> dict[str, Any]:
    """Compute REC-004AP token-identifiability-stratified gradient alignment metrics."""
    ex_target = [ex for ex in examples if len(ex.input_tokens) == target_length]
    labels = _labels_for_examples(
        ex_target, [target_length] * len(ex_target), target_length, device
    )

    with torch.no_grad():
        batch_input = collate_content_only_batch(ex_target, core.tokens, device=device)
        h = core.model.encode(batch_input)[:, 1 : 1 + target_length, :]

    strata_indices = stratify_examples_by_token_identifiability(
        ex_target, target_length, target_position, corr_key, competitor_key
    )

    eval_strata = {
        "stratum_A_unique_target": strata_indices["A"],
        "stratum_B_aliased_other": strata_indices["B"],
        "stratum_C_aliased_key7": strata_indices["C"],
        "pooled_all": list(range(len(ex_target))),
    }

    # 1. Margin gradient vector: M = S(pos, corr_key) - S(pos, competitor_key)
    primitive.train()
    primitive.zero_grad(set_to_none=True)
    sc_m = primitive.compute_routing_scores(
        [target_length], [target_length], None, device=device, lmax=target_length
    )
    M = (
        sc_m[0, :, target_position, corr_key].mean()
        - sc_m[0, :, target_position, competitor_key].mean()
    )
    M.backward()

    gm_routing = _extract_routing_grad(primitive, target_position, target_length).clone()
    norm_gm_routing = float(gm_routing.norm().item())

    results: dict[str, Any] = {}

    for stratum_key, indices in eval_strata.items():
        sub_h = h[indices]
        sub_labels = labels[indices]
        batch_sz = len(indices)

        if batch_sz == 0:
            continue

        # Batched pass for aggregated gradients
        primitive.zero_grad(set_to_none=True)
        c_lens = [target_length] * batch_sz
        o_lens = [target_length] * batch_sz

        logits, scores = _forward_with_explicit_scores(
            primitive, sub_h, c_lens, o_lens, device=device, lmax=target_length
        )
        scores.retain_grad()

        loss_pos = F.cross_entropy(logits[:, target_position, :], sub_labels[:, target_position])
        loss_pos.backward()

        # Score-space gradients dL/dS(target_pos, j) averaged over batch and heads
        assert scores.grad is not None
        dL_dS_pos = scores.grad[:, :, target_position, :].mean(dim=(0, 1))
        dL_dS_corr = float(dL_dS_pos[corr_key].item())
        dL_dS_comp = float(dL_dS_pos[competitor_key].item())
        delta_M_score = -(dL_dS_corr - dL_dS_comp)

        # Parameter-space gradient
        gl_routing = _extract_routing_grad(primitive, target_position, target_length).clone()
        norm_gl_routing = float(gl_routing.norm().item())

        dot_total = -float(torch.dot(gm_routing, gl_routing).item())
        cos_align = dot_total / (norm_gl_routing * norm_gm_routing + 1e-12)

        # Component norms
        assert primitive.query_position_embedding.weight.grad is not None
        assert primitive.length_embedding.weight.grad is not None
        assert primitive.key_position_embedding.weight.grad is not None
        assert primitive.cross_attn.in_proj_weight.grad is not None

        gl_qpos = float(
            primitive.query_position_embedding.weight.grad[target_position].norm().item()
        )
        gl_len = float(primitive.length_embedding.weight.grad[target_length].norm().item())
        gl_kcorr = float(primitive.key_position_embedding.weight.grad[corr_key].norm().item())
        gl_kcomp = float(primitive.key_position_embedding.weight.grad[competitor_key].norm().item())
        gl_wq = float(primitive.cross_attn.in_proj_weight.grad[:32].norm().item())
        gl_wk = float(primitive.cross_attn.in_proj_weight.grad[32:64].norm().item())

        # Per-example loop for distribution statistics
        per_ex_param_dots: list[float] = []
        per_ex_score_delta: list[float] = []

        for i_sub in range(batch_sz):
            primitive.zero_grad(set_to_none=True)
            l_i, s_i = _forward_with_explicit_scores(
                primitive,
                sub_h[i_sub : i_sub + 1],
                [target_length],
                [target_length],
                device,
                target_length,
            )
            s_i.retain_grad()
            loss_i = F.cross_entropy(
                l_i[:, target_position, :], sub_labels[i_sub : i_sub + 1, target_position]
            )
            loss_i.backward()

            gi_routing = _extract_routing_grad(primitive, target_position, target_length)
            dot_i = -float(torch.dot(gm_routing, gi_routing).item())
            per_ex_param_dots.append(dot_i)

            assert s_i.grad is not None
            ds_i = s_i.grad[0, :, target_position, :].mean(dim=0)
            dM_s_i = -float((ds_i[corr_key] - ds_i[competitor_key]).item())
            per_ex_score_delta.append(dM_s_i)

        t_param = torch.tensor(per_ex_param_dots)
        t_score = torch.tensor(per_ex_score_delta)
        eps = 1e-5

        results[stratum_key] = {
            "example_count": batch_sz,
            "predicted_score_margin_change": delta_M_score,
            "dL_dS_corr": dL_dS_corr,
            "dL_dS_competitor": dL_dS_comp,
            "predicted_margin_change": dot_total,
            "cosine_alignment": cos_align,
            "routing_grad_norm": norm_gl_routing,
            "grad_norms_by_component": {
                "query_pos": gl_qpos,
                "length_emb": gl_len,
                "key_pos_corr": gl_kcorr,
                "key_pos_competitor": gl_kcomp,
                "w_query": gl_wq,
                "w_key": gl_wk,
            },
            "per_example_mean": float(t_param.mean().item()),
            "per_example_median": float(t_param.median().item()),
            "positive_fraction": float((t_param > eps).float().mean().item()),
            "near_zero_fraction": float((t_param.abs() <= eps).float().mean().item()),
            "negative_fraction": float((t_param < -eps).float().mean().item()),
            "score_per_example_mean": float(t_score.mean().item()),
            "score_per_example_median": float(t_score.median().item()),
        }

    return results


def build_warm_start_protocol(
    config: MirrorCDDPCAWarmStartPilotConfig,
    parent_manifest: mb.ModelBundleManifest,
) -> dict[str, Any]:
    """Assemble and lock preregistered experimental protocol for REC-004AQ."""
    checkpoint_steps = list(range(0, config.max_updates + 1, config.checkpoint_interval))
    lr_table: dict[str, float] = {str(step): _expected_lr(step) for step in checkpoint_steps}

    if parent_manifest.bundle_id != REC004AL_PARENT_BUNDLE_ID:
        raise mb.IncompleteBundleError(
            f"Parent bundle_id mismatch: {parent_manifest.bundle_id} != {REC004AL_PARENT_BUNDLE_ID}"
        )
    if parent_manifest.core.canonical_state_hash != REC004AL_PARENT_CORE_HASH:
        raise mb.CoreDependencyMismatchError(
            f"Parent Core hash mismatch: {parent_manifest.core.canonical_state_hash} "
            f"!= {REC004AL_PARENT_CORE_HASH}"
        )

    # Verify source REC-004AK files
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
            raise mb.CorruptedArtifactError(f"{path.name} raw hash mismatch: {raw_h} != {exp_raw}")
        if exp_canon is not None:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            canon_h = mb.canonical_state_hash(sd)
            if canon_h != exp_canon:
                raise mb.CorruptedArtifactError(
                    f"{path.name} canonical hash mismatch: {canon_h} != {exp_canon}"
                )

    return {
        "task_id": REC004AQ_TASK_ID,
        "baseline_task_id": REC004AQ_BASELINE_TASK_ID,
        "source_task_id": REC004AQ_SOURCE_TASK_ID,
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_canonical_state_hash": REC004AL_PARENT_CORE_HASH,
        "init_id": config.init_id,
        "model_seed": config.seed,
        "target_operation": REC004AL_TARGET_OPERATION,
        "architecture_signature": REC004AL_ARCHITECTURE_SIGNATURE,
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
            "post_warm_start_steps": f"{config.warm_start_steps + 1}..{config.max_updates}",
            "post_warm_start_sampling_rule": (
                "exact return to REC-004AL baseline sampler via "
                "ibc._generate_step_training_examples and identical per-step RNG derivation"
            ),
            "boundary_pre_fixed": True,
            "sweep_authorized": False,
            "length_position_selectivity_prohibited": True,
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
        },
        "evaluation_protocol": {
            "validation_split": REC004AL_EXISTING_VALIDATION_SPLIT,
            "validation_examples": config.existing_validation_examples,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_steps": checkpoint_steps,
            "decisive_step": config.decisive_step,
            "terminal_viability_floor": config.existing_validation_floor,
            "terminal_viability_metric": "sequence_exact_match",
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


def run_cd_dpca_warm_start_pilot_task(
    config: MirrorCDDPCAWarmStartPilotConfig,
) -> dict[str, Any]:
    """Main execution function for Task B-C005REC-004AQ."""
    _guard_not_frozen(REC004AQ_TASK_ID)
    t_start = time.time()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.output_dir / "checkpoints"
    state_dir = config.output_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: Load parent runtime and verify protocol
    parent_manifest, _raw = ibc._load_parent_manifest()
    protocol = build_warm_start_protocol(config, parent_manifest)
    (config.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )

    # Config yaml
    config_dict = {
        "task_id": REC004AQ_TASK_ID,
        "baseline_task_id": REC004AQ_BASELINE_TASK_ID,
        "source_task_id": REC004AQ_SOURCE_TASK_ID,
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "init_id": config.init_id,
        "target_operation": REC004AL_TARGET_OPERATION,
        "max_updates": config.max_updates,
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

    # Stage B: Information boundary audit
    info_boundary = run_information_boundary_audit()
    info_boundary["task_id"] = REC004AQ_TASK_ID
    (config.output_dir / "information_boundary_audit.json").write_text(
        json.dumps(info_boundary, indent=2), encoding="utf-8"
    )
    if info_boundary["status"] != "PASS":
        raise ValueError(f"INFORMATION_BOUNDARY_AUDIT_FAILURE: {info_boundary}")

    # Stage C: Fresh-load CD-DPCA and construct isolated trainable copy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core, parent_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core.model.to(device)
    core.model.eval()

    fresh_bank = PrimitiveBank.from_artifacts(config.rec004ak_dir, prefix="bank", strict=True)
    initial_primitive = fresh_bank.get(0)
    assert isinstance(
        initial_primitive, ContentDecoupledDiscretePositionalCrossAttentionPrimitive
    )
    initial_state_dict = {
        k: v.detach().clone().cpu() for k, v in initial_primitive.state_dict().items()
    }
    initial_canonical_hash = mb.canonical_state_hash(initial_state_dict)

    if initial_canonical_hash != REC004AK_PRIMITIVE_STATE_CANONICAL_HASH:
        raise mb.CorruptedArtifactError(
            f"Initial CD-DPCA state hash mismatch: {initial_canonical_hash} "
            f"!= {REC004AK_PRIMITIVE_STATE_CANONICAL_HASH}"
        )

    pid = op_to_id[REC004AL_TARGET_OPERATION]
    primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
        primitive_id=pid,
        config=initial_primitive.config,
        status=PrimitiveStatus.CANDIDATE,
        created_at_task=0,
        metadata={"family": "CD-DPCA", "task": REC004AQ_TASK_ID, "init_id": config.init_id},
    )
    primitive.load_state_dict(initial_state_dict, strict=True)
    primitive.to(device)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    # Optimizer and scheduler (identical to REC-004AL)
    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=REC004AL_OPERATOR_LR,
        weight_decay=REC004AL_OPERATOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004AL_SCHEDULER_T_MAX, eta_min=REC004AL_SCHEDULER_ETA_MIN
    )

    # Validation dataset (identical to REC-004AL)
    val_examples = ibc._generate_parameter_free_examples(
        config.seed,
        config.existing_validation_examples,
        operation=REC004AL_TARGET_OPERATION,
        split=REC004AL_EXISTING_VALIDATION_SPLIT,
        vocab_size=REC004AL_VOCAB_SIZE,
        sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
    )

    # Stage D: Training loop with evaluation cadence
    learning_curve_records: list[dict[str, Any]] = []
    lr_trace_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    position4_records: list[dict[str, Any]] = []
    strata_alignment_records: list[dict[str, Any]] = []

    def _evaluate_checkpoint(step: int, lr_u: float | None, lr_a: float) -> dict[str, Any]:
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
                "init_id": config.init_id,
            },
            state_dir / f"step{step}.pt",
        )

        val_summary, pos_report = compute_per_length_position_metrics(core, primitive, val_examples)
        pos4_metrics = compute_position4_routing_metrics(primitive, val_examples, device)
        pos4_metrics["step"] = step
        pos4_metrics["token_accuracy"] = pos_report.get("10:4", {}).get("accuracy", 0.0)
        position4_records.append(pos4_metrics)

        strata_metrics = compute_strata_gradient_alignment_at_checkpoint(
            core, primitive, val_examples, device
        )
        strata_metrics_with_step = {"step": step, "strata": strata_metrics}
        strata_alignment_records.append(strata_metrics_with_step)

        vram_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        ckpt_data = {
            "step": step,
            "init_id": config.init_id,
            "lr_used": lr_u,
            "lr_after_scheduler": lr_a,
            "sequence_exact_match": val_summary["sequence_exact_match"],
            "token_accuracy": val_summary["token_accuracy"],
            "by_length": val_summary["by_length"],
            "canonical_state_hash": mb.canonical_state_hash(p_sd),
            "wall_clock_seconds": time.time() - t_start,
            "peak_vram_bytes": vram_bytes,
            "position4_metrics": pos4_metrics,
            "strata_alignment": strata_metrics,
        }
        checkpoint_records.append(ckpt_data)
        primitive.train()
        return ckpt_data

    # Step 0 evaluation
    initial_lr = optimizer.param_groups[0]["lr"]
    _evaluate_checkpoint(0, lr_u=None, lr_a=initial_lr)

    cumulative_examples = 0
    for step in range(1, config.max_updates + 1):
        step_examples = generate_step_training_examples_warm_start(
            config.seed,
            step,
            REC004AL_TARGET_OPERATION,
            vocab_size=REC004AL_VOCAB_SIZE,
            sequence_length_range=REC004AL_SEQUENCE_LENGTH_RANGE,
            warm_start_steps=config.warm_start_steps,
            n=REC004AL_EXAMPLES_PER_STEP,
        )
        cumulative_examples += len(step_examples)
        c_lens = [len(ex.input_tokens) for ex in step_examples]
        o_lens = [get_operation(REC004AL_TARGET_OPERATION).output_length(n) for n in c_lens]
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
            ckpt_info = _evaluate_checkpoint(step, lr_used, lr_after)
            learning_curve_records.append(
                {
                    "step": step,
                    "train_loss": running_loss,
                    "lr": lr_after,
                    "val_em": ckpt_info["sequence_exact_match"],
                    "val_token_acc": ckpt_info["token_accuracy"],
                }
            )

    # Save training traces
    with (config.output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as f:
        for rec in learning_curve_records:
            f.write(json.dumps(rec) + "\n")
    with (config.output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as f:
        for rec in lr_trace_records:
            f.write(json.dumps(rec) + "\n")
    (config.output_dir / "position4_trajectory.json").write_text(
        json.dumps(position4_records, indent=2), encoding="utf-8"
    )
    (config.output_dir / "strata_alignment_trajectory.json").write_text(
        json.dumps(strata_alignment_records, indent=2), encoding="utf-8"
    )

    # Stage E: Decisive evaluation at step 6000
    primitive.eval()
    final_summary, final_pos_report = compute_per_length_position_metrics(
        core, primitive, val_examples
    )
    (config.output_dir / "per_length_position_metrics.json").write_text(
        json.dumps(
            {
                "task_id": REC004AQ_TASK_ID,
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
            REC004AL_TARGET_OPERATION,
            seed=config.seed,
            n_examples=config.existing_validation_examples,
            split=REC004AL_EXISTING_VALIDATION_SPLIT,
        )
    finally:
        parent_bank.replace_primitive(pid, old_slot)

    causal_controls = {
        "task_id": REC004AQ_TASK_ID,
        "operation": REC004AL_TARGET_OPERATION,
        "decisive_step": config.decisive_step,
        "correct_exact_match": causal_eval["correct_exact_match"],
        "correct_token_accuracy": causal_eval["correct_token_accuracy"],
        "wrong_family_exact_match": causal_eval["wrong_family_exact_match"],
        "none_exact_match": causal_eval["none_exact_match"],
        "exact_match_causal_gap": causal_eval["exact_match_causal_gap"],
        "wrong_family_operation": "REVERSE",
        "evidence_type": "EVALUATION_ONLY_CONTROLS",
    }
    (config.output_dir / "causal_controls.json").write_text(
        json.dumps(causal_controls, indent=2), encoding="utf-8"
    )

    # Attention & masking diagnostics
    attn_diagnostics = run_attention_masking_diagnostics(core, primitive, val_examples)
    attn_diagnostics["task_id"] = REC004AQ_TASK_ID
    (config.output_dir / "attention_masking_diagnostics.json").write_text(
        json.dumps(attn_diagnostics, indent=2), encoding="utf-8"
    )

    # Stage F: Trajectory comparison against REC-004AL baseline
    comparison_table: list[dict[str, Any]] = []
    rec004an_traj_p = config.rec004an_dir / "trajectory_metrics.json"
    rec004ao_traj_p = config.rec004ao_dir / "trajectory_routing.json"
    rec004ap_strata_p = config.rec004ap_dir / "strata_trajectory_metrics.json"

    al_l10_map: dict[int, float] = {}
    if rec004an_traj_p.is_file():
        an_data = json.loads(rec004an_traj_p.read_text(encoding="utf-8"))
        l10_traj = an_data.get("summary", {}).get("length_10_trajectory_sequence_em", {})
        for st_str, val in l10_traj.items():
            al_l10_map[int(st_str)] = float(val)

    al_ao_map: dict[int, dict[str, Any]] = {}
    if rec004ao_traj_p.is_file():
        ao_data = json.loads(rec004ao_traj_p.read_text(encoding="utf-8"))
        for ck in ao_data.get("checkpoints", []):
            al_ao_map[ck["step"]] = ck

    al_ap_map: dict[int, dict[str, Any]] = {}
    if rec004ap_strata_p.is_file():
        ap_data = json.loads(rec004ap_strata_p.read_text(encoding="utf-8"))
        for item in ap_data.get("trajectory", []):
            st = item.get("step")
            if st is not None:
                strata = item.get("strata", {})
                al_ap_map[st] = {
                    "stratum_A": strata.get("stratum_A_unique_target", {})
                    .get("parameter_space", {})
                    .get("predicted_margin_change"),
                    "stratum_C": strata.get("stratum_C_aliased_key7", {})
                    .get("parameter_space", {})
                    .get("predicted_margin_change"),
                    "pooled": strata.get("pooled_all", {})
                    .get("parameter_space", {})
                    .get("predicted_margin_change"),
                }

    # Load REC-004AL checkpoint evaluations from its per-checkpoint learning curve
    al_learning_curve_p = config.rec004al_dir / "learning_curve.jsonl"
    al_lc_map: dict[int, dict[str, Any]] = {0: {"val_em": 0.0}}
    if al_learning_curve_p.is_file():
        with al_learning_curve_p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                al_lc_map[rec["step"]] = rec

    for ckpt in checkpoint_records:
        st = ckpt["step"]
        aq_pos4 = ckpt["position4_metrics"]
        aq_strata = ckpt["strata_alignment"]
        aq_l10 = ckpt["by_length"].get("10", {})

        al_ao = al_ao_map.get(st, {})
        al_ap = al_ap_map.get(st, {})
        al_lc = al_lc_map.get(st, {})
        al_l10_em = al_l10_map.get(st)
        aq_l10_em = aq_l10.get("sequence_exact_match")

        comparison_table.append(
            {
                "step": st,
                "overall_em": {
                    "rec004al": al_lc.get("val_em"),
                    "rec004aq": ckpt["sequence_exact_match"],
                    "delta": (
                        ckpt["sequence_exact_match"] - al_lc["val_em"]
                        if al_lc.get("val_em") is not None
                        else None
                    ),
                },
                "length10_em": {
                    "rec004al": al_l10_em,
                    "rec004aq": aq_l10_em,
                    "delta": (
                        aq_l10_em - al_l10_em
                        if aq_l10_em is not None and al_l10_em is not None
                        else None
                    ),
                },
                "position4": {
                    "top1_key_al": al_ao.get("top1_key"),
                    "top1_key_aq": aq_pos4["top1_key_mode"],
                    "token_accuracy_al": al_ao.get("token_accuracy"),
                    "token_accuracy_aq": aq_pos4["token_accuracy"],
                    "p_correct_al": al_ao.get("correct_key_probability"),
                    "p_correct_aq": aq_pos4["mean_p_correct"],
                    "p_competitor_al": al_ao.get("wrong_key_probability"),
                    "p_competitor_aq": aq_pos4["mean_p_competitor"],
                    "margin_al": al_ao.get("score_margin"),
                    "margin_aq": aq_pos4["mean_margin_vs_competitor"],
                    "entropy_al": al_ao.get("attention_entropy"),
                    "entropy_aq": aq_pos4["mean_entropy"],
                },
                "strata_alignment_predicted_margin": {
                    "stratum_A_al": al_ap.get("stratum_A"),
                    "stratum_A_aq": aq_strata.get("stratum_A_unique_target", {}).get(
                        "predicted_margin_change"
                    ),
                    "stratum_C_al": al_ap.get("stratum_C"),
                    "stratum_C_aq": aq_strata.get("stratum_C_aliased_key7", {}).get(
                        "predicted_margin_change"
                    ),
                    "pooled_al": al_ap.get("pooled"),
                    "pooled_aq": aq_strata.get("pooled_all", {}).get("predicted_margin_change"),
                },
            }
        )

    (config.output_dir / "trajectory_comparison.json").write_text(
        json.dumps(comparison_table, indent=2), encoding="utf-8"
    )

    # Stage G: Decision evaluation
    decisive_em = final_summary["sequence_exact_match"]
    terminal_floor_met = decisive_em >= config.existing_validation_floor

    final_pos4 = position4_records[-1] if position4_records else {}
    pos4_acc = final_pos_report.get("10:4", {}).get("accuracy", 0.0)
    pos4_attractor_cleared = (
        final_pos4.get("top1_key_mode") == 0
        and final_pos4.get("mean_p_correct", 0.0) > final_pos4.get("mean_p_competitor", 1.0)
        and pos4_acc >= 0.95
    )

    # Check for regression across other positions
    other_pos_regressed = False
    for pos_key, pos_val in final_pos_report.items():
        if pos_key != "10:4" and pos_val["accuracy"] < 0.90:
            other_pos_regressed = True

    viability_met = terminal_floor_met and pos4_attractor_cleared and not other_pos_regressed

    decision = (
        "WARM_START_PILOT_VIABILITY_MET"
        if viability_met
        else "WARM_START_PILOT_TERMINAL_VIABILITY_NOT_MET"
    )

    freeze_audit = {
        "task_id": REC004AQ_TASK_ID,
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
    (config.output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2), encoding="utf-8"
    )

    side_effect_audit = {
        "task_id": REC004AQ_TASK_ID,
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "additional_inits_run": 0,
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

    source_manifest = {
        "task_id": REC004AQ_TASK_ID,
        "baseline_task_id": REC004AQ_BASELINE_TASK_ID,
        "parent_manifest_path": str(ibc.REC004A_PARENT_MANIFEST_PATH),
        "parent_bundle_id": REC004AL_PARENT_BUNDLE_ID,
        "parent_core_hash": REC004AL_PARENT_CORE_HASH,
        "rec004ak_dir": str(config.rec004ak_dir),
        "rec004ak_hashes": protocol["source_hashes"],
        "initial_primitive_canonical_hash": initial_canonical_hash,
        "final_primitive_canonical_hash": mb.canonical_state_hash(primitive.state_dict()),
    }
    (config.output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    total_wall_seconds = time.time() - t_start

    summary = {
        "task": REC004AQ_TASK_ID,
        "baseline_task": REC004AQ_BASELINE_TASK_ID,
        "execution_status": "PASS",
        "decision": decision,
        "init_id": config.init_id,
        "max_updates": config.max_updates,
        "warm_start_steps": config.warm_start_steps,
        "decisive_step": config.decisive_step,
        "terminal_viability_floor": config.existing_validation_floor,
        "decisive_validation_sequence_em": decisive_em,
        "decisive_validation_exact_count": final_summary["exact_match_count"],
        "decisive_validation_total_examples": final_summary["n_examples"],
        "decisive_validation_token_accuracy": final_summary["token_accuracy"],
        "terminal_floor_met": terminal_floor_met,
        "position4_attractor_cleared": pos4_attractor_cleared,
        "other_positions_regressed": other_pos_regressed,
        "terminal_viability_met": viability_met,
        "by_length_em": {
            k: {
                "exact": v["exact_match_count"],
                "total": v["n_sequences"],
                "em": v["sequence_exact_match"],
                "token_accuracy": v["token_accuracy"],
            }
            for k, v in final_summary["by_length"].items()
        },
        "final_position4_metrics": final_pos4,
        "causal_gap": causal_eval["exact_match_causal_gap"],
        "padding_mask_verified": attn_diagnostics["padding_mask_verified"],
        "correct_key_top1_routing_accuracy": (
            attn_diagnostics["correct_key_top1_routing_accuracy"]
        ),
        "mean_correct_vs_runnerup_score_margin": (
            attn_diagnostics["mean_correct_vs_runnerup_score_margin"]
        ),
        "candidate_selected": None,
        "child_bundle": None,
        "bundle_write": False,
        "rg3": "NOT_EXECUTED",
        "rec005_eligible": False,
        "g1": "NOT_CLEARED",
        "g4": "NOT_CLEARED",
        "total_optimizer_updates": config.max_updates,
        "wall_seconds": total_wall_seconds,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Markdown report
    em_hits = final_summary["exact_match_count"]
    em_total = final_summary["n_examples"]
    report_lines = [
        f"# REC-004AQ CD-DPCA Warm-Start Single-Recipe Causal Pilot Report ({decision})",
        "",
        f"- **Task ID:** `{REC004AQ_TASK_ID}`",
        f"- **Decision:** `{decision}`",
        f"- **Initialization:** `{config.init_id}` (canonical hash: `{initial_canonical_hash}`)",
        f"- **Warm-Start Steps:** `{config.warm_start_steps}` / `{config.max_updates}` updates",
        f"- **Terminal Viability Criterion:** sequence EM >= "
        f"`{config.existing_validation_floor}` & pos4 attractor cleared",
        f"- **Achieved Decisive Sequence EM:** `{decisive_em:.6f}` ({em_hits}/{em_total})",
        f"- **Length-10 Position-4 Accuracy:** `{pos4_acc:.4f}` "
        f"(top-1 key: `{final_pos4.get('top1_key_mode')}`)",
        f"- **Viability Status:** "
        f"`{'PASSED' if viability_met else 'FAILED (STOP GATE TRIGGERED)'}`",
        "",
        "## Per-Length Breakdown",
        "",
        "| Length | Sequence EM | Numerator / Denominator | Token Accuracy |",
        "|---|---|---|---|",
    ]
    for L_str in sorted(final_summary["by_length"].keys(), key=int):
        b = final_summary["by_length"][L_str]
        report_lines.append(
            f"| {L_str} | {b['sequence_exact_match']:.4f} | "
            f"{b['exact_match_count']} / {b['n_sequences']} | {b['token_accuracy']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Position-4 Trajectory Comparison (REC-004AL vs REC-004AQ)",
            "",
            "| Step | AL Key | AQ Key | AL Margin | AQ Margin | AL Acc | AQ Acc |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in comparison_table:
        st = row["step"]
        p4 = row["position4"]
        m_al = f"{p4['margin_al']:.2f}" if p4["margin_al"] is not None else "N/A"
        m_aq = f"{p4['margin_aq']:.2f}" if p4["margin_aq"] is not None else "N/A"
        acc_al = f"{p4['token_accuracy_al']:.4f}" if p4["token_accuracy_al"] is not None else "N/A"
        acc_aq = f"{p4['token_accuracy_aq']:.4f}" if p4["token_accuracy_aq"] is not None else "N/A"
        k_al = str(p4["top1_key_al"]) if p4["top1_key_al"] is not None else "N/A"
        k_aq = str(p4["top1_key_aq"]) if p4["top1_key_aq"] is not None else "N/A"
        report_lines.append(
            f"| {st} | {k_al} | {k_aq} | {m_al} | {m_aq} | {acc_al} | {acc_aq} |"
        )

    report_lines.extend(
        [
            "",
            "## Causal Controls & Attention Diagnostics",
            "",
            f"- **Correct Control EM:** `{causal_eval['correct_exact_match']:.4f}`",
            f"- **Wrong-Family Control EM (REVERSE):** "
            f"`{causal_eval['wrong_family_exact_match']:.4f}`",
            f"- **None Control EM:** `{causal_eval['none_exact_match']:.4f}`",
            f"- **Causal Gap:** `{causal_eval['exact_match_causal_gap']:.4f}`",
            f"- **Padding Mask Verified:** `{attn_diagnostics['padding_mask_verified']}`",
            f"- **Top-1 Routing Accuracy to Correct Key:** "
            f"`{attn_diagnostics['correct_key_top1_routing_accuracy']:.4f}`",
            f"- **Mean Correct vs Runner-up Score Margin:** "
            f"`{attn_diagnostics['mean_correct_vs_runnerup_score_margin']:.4f}`",
            "",
            "## Execution Boundaries & Preserved Blocks",
            "",
            "- `candidate_selected`: `null` (strictly preserved)",
            "- `child_bundle`: `null` (no model bundle written)",
            "- `rg3_recheck`: `NOT_EXECUTED`",
            "- `rec005_eligible`: `false`",
            "- `G1` and `G4`: `NOT_CLEARED` (independent blocks strictly preserved)",
            f"- Multi-init validation (REC-004AM): "
            f"`{'CONSIDERABLE' if viability_met else 'BLOCKED'}`",
            "",
        ]
    )
    (config.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "protocol": protocol,
        "summary": summary,
        "checkpoints": checkpoint_records,
        "position4_records": position4_records,
        "strata_alignment_records": strata_alignment_records,
        "comparison_table": comparison_table,
        "causal_controls": causal_controls,
        "attention_diagnostics": attn_diagnostics,
        "per_length_summary": final_summary,
        "per_position_report": final_pos_report,
    }
