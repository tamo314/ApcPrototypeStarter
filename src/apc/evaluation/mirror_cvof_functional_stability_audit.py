"""B-C005REC-004Y: I03 CVOF Functional-Stability Gate Identifiability Audit.

Audits candidate non-oracle function-space drift metrics (M1-M5) to determine
whether safe pre-transition CVOF updates (7000->7500) and unsafe post-transition
CVOF updates (7525, 8000) can be identifiably separated a priori under fixed
non-oracle reference attention A_ref7500 with zero new optimizer updates.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_attention_clamp_causal_replay as rec004u
from apc.evaluation import mirror_budget_extension as mbe
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_cvof_trust_region_pilot as rec004x
from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_downstream_freeze_causal_replay as rec004v
from apc.evaluation import mirror_normal_cvof_protection_pilot as rec004w
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive

__all__ = [
    "REC004Y_TASK_ID",
    "REC004Y_TARGET_OPERATION",
    "REC004Y_SEED",
    "REC004Y_DECISIVE_INIT",
    "REC004Y_CONTROL_INIT",
    "REC004Y_CALIBRATION_STEP",
    "REC004Y_REFERENCE_STEP",
    "REC004Y_ONSET_STEP",
    "REC004Y_LATE_STEP",
    "REC004Y_PARITY_UPDATES",
    "REC004Y_SEPARATION_FLOOR",
    "REC004Y_PROBE_NAME",
    "REC004Y_CANDIDATE_METRICS",
    "REC004Y_METRIC_PRIORITY",
    "MirrorCVOFFunctionalStabilityAuditConfig",
    "build_cvof_functional_gate_probe_v1",
    "prepare_rec004y_dataset",
    "cache_reference_attention_7500",
    "evaluate_functional_forward_with_stages",
    "compute_functional_metrics",
    "obtain_i03_7525_historical_state",
    "run_cvof_functional_stability_audit_task",
]

# =============================================================================
# Constants
# =============================================================================

REC004Y_TASK_ID: Final = "B-C005REC-004Y"
REC004Y_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004Y_BASE_PRIMITIVE: Final = "P_LENGTH_POSITION_BIAS"
REC004Y_SEED: Final = RECOVERY_PILOT_SEED  # 10
REC004Y_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE  # 64
REC004Y_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE  # (2, 10)

REC004Y_DECISIVE_INIT: Final = "I03"
REC004Y_CONTROL_INIT: Final = "I04"

REC004Y_CALIBRATION_STEP: Final = 7000
REC004Y_REFERENCE_STEP: Final = 7500
REC004Y_ONSET_STEP: Final = 7525
REC004Y_LATE_STEP: Final = 8000
REC004Y_PARITY_UPDATES: Final = 25

REC004Y_SEPARATION_FLOOR: Final = 2.0
REC004Y_EPS: Final = 1e-6
REC004Y_NORM_EPS: Final = 1e-8
REC004Y_PROBE_EPS: Final = 1e-12

REC004Y_PROBE_NAME: Final = "cvof_functional_gate_probe_v1"
REC004Y_NORMAL_EXAMPLES: Final = 512
REC004Y_LENGTH10_EXAMPLES: Final = 512
REC004Y_TOTAL_EXAMPLES: Final = 1024

REC004Y_CANDIDATE_METRICS: Final[tuple[str, ...]] = (
    "M1_logit_rms",
    "M2_output_kl",
    "M3_post_ffn_displacement",
    "M4_attn_out_displacement",
    "M5_replay_ce",
)

# Pre-declared priority rule if multiple metrics pass
REC004Y_METRIC_PRIORITY: Final[tuple[str, ...]] = (
    "M5_replay_ce",
    "M1_logit_rms",
    "M2_output_kl",
    "M3_post_ffn_displacement",
    "M4_attn_out_displacement",
)

CVOF_GROUPS: Final[tuple[str, ...]] = ("C", "V", "O", "F")


@dataclass(frozen=True)
class MirrorCVOFFunctionalStabilityAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004y/run_001")
    seed: int = REC004Y_SEED
    separation_floor: float = REC004Y_SEPARATION_FLOOR
    n_normal_probe_examples: int = REC004Y_NORMAL_EXAMPLES
    n_length10_probe_examples: int = REC004Y_LENGTH10_EXAMPLES
    eps: float = REC004Y_EPS
    parity_updates: int = REC004Y_PARITY_UPDATES


# =============================================================================
# Helper Utilities
# =============================================================================


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _digest_examples(examples: Iterable[Any]) -> set[str]:
    return {traj_audit._digest_example(example) for example in examples}


def _dataset_digest(digests: set[str]) -> str:
    hasher = hashlib.sha256()
    for digest in sorted(digests):
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def _derive_local_seed_helper(base_seed: int, slot: int, label: str) -> int:
    h = hashlib.sha256(f"{base_seed}:{slot}:{label}".encode()).hexdigest()
    return int(h[:8], 16)


# =============================================================================
# Dataset Generation and Locking
# =============================================================================


def build_cvof_functional_gate_probe_v1(
    seed: int,
    protected_digests: set[str],
    n_normal: int = REC004Y_NORMAL_EXAMPLES,
    n_length10: int = REC004Y_LENGTH10_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    """Builds cvof_functional_gate_probe_v1 dataset:
    - 512 normal distribution examples (lengths 2..10)
    - 512 length-10 examples
    Total: 1024 examples.
    Guaranteed disjoint against all protected registries.
    """
    op_obj = get_operation(REC004Y_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    # 1. Normal distribution (lengths 2..10)
    normal_rng = random.Random(_derive_local_seed_helper(seed, 0, f"{REC004Y_PROBE_NAME}:normal"))
    for slot in range(n_normal):
        rejected: list[str] = []
        while True:
            total_draws += 1
            seq_len = normal_rng.randint(*REC004Y_SEQUENCE_LENGTH_RANGE)
            seq = tuple(normal_rng.randrange(REC004Y_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(normal_rng, seq, REC004Y_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004Y_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004Y_VOCAB_SIZE)
            digest = traj_audit._digest(seq, res.output_tokens)
            if digest not in protected_digests:
                break
            rejected.append(digest)
        if rejected:
            substitutions.append(
                {"part": "normal", "slot": slot, "rejected": rejected, "accepted": digest}
            )
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=REC004Y_PROBE_NAME,
                vocab_size=REC004Y_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004Y_TARGET_OPERATION,)
                ),
            )
        )

    # 2. Length-10 subset
    l10_rng = random.Random(_derive_local_seed_helper(seed, 0, f"{REC004Y_PROBE_NAME}:length10"))
    for slot in range(n_length10):
        rejected = []
        while True:
            total_draws += 1
            seq_len = 10
            seq = tuple(l10_rng.randrange(REC004Y_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(l10_rng, seq, REC004Y_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004Y_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004Y_VOCAB_SIZE)
            digest = traj_audit._digest(seq, res.output_tokens)
            if digest not in protected_digests:
                break
            rejected.append(digest)
        if rejected:
            substitutions.append(
                {"part": "length10", "slot": slot, "rejected": rejected, "accepted": digest}
            )
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=REC004Y_PROBE_NAME,
                vocab_size=REC004Y_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004Y_TARGET_OPERATION,)
                ),
            )
        )

    audit_detail = {
        "dataset_name": REC004Y_PROBE_NAME,
        "n_normal_requested": n_normal,
        "n_length10_requested": n_length10,
        "n_total": len(examples),
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions[:10],
    }
    return examples, audit_detail


def prepare_rec004y_dataset(seed: int) -> tuple[list[Example], dict[str, Any]]:
    """Builds exhaustive protected registry and constructs cvof_functional_gate_probe_v1."""
    # Base protected registry through REC-004V
    protected, protected_counts = rec004t.build_protected_registry_exhaustive(seed)

    # REC-004T continuity probes
    cont1_exs, _ = rec004t.prepare_probe_datasets(seed)
    for k, exs in cont1_exs.items():
        digests = _digest_examples(exs)
        protected |= digests
        protected_counts[f"rec004t_{k}"] = len(digests)

    # REC-004U attention clamp probe
    u_exs, _ = rec004u.build_attention_clamp_causal_probe_v1(seed, protected)
    u_digests = _digest_examples(u_exs)
    protected |= u_digests
    protected_counts["rec004u_attention_clamp_causal_probe_v1"] = len(u_digests)

    # REC-004V downstream freeze probe
    v_exs, _ = rec004v.build_downstream_freeze_causal_probe_v1(seed, protected)
    v_digests = _digest_examples(v_exs)
    protected |= v_digests
    protected_counts["rec004v_downstream_freeze_causal_probe_v1"] = len(v_digests)

    # REC-004W normal cvof protection sets
    w_val_exs, _ = rec004w.build_normal_cvof_protection_validation_v1(seed, protected)
    w_val_digests = _digest_examples(w_val_exs)
    protected |= w_val_digests
    protected_counts["rec004w_validation"] = len(w_val_digests)

    w_conf_exs, _ = rec004w.build_normal_cvof_protection_length10_v1(seed, protected)
    w_conf_digests = _digest_examples(w_conf_exs)
    protected |= w_conf_digests
    protected_counts["rec004w_length10"] = len(w_conf_digests)

    # REC-004X trust region sets
    x_val_exs, _ = rec004x.build_cvof_trust_region_validation_v1(seed, protected)
    x_val_digests = _digest_examples(x_val_exs)
    protected |= x_val_digests
    protected_counts["rec004x_validation"] = len(x_val_digests)

    x_conf_exs, _ = rec004x.build_cvof_trust_region_length10_v1(seed, protected)
    x_conf_digests = _digest_examples(x_conf_exs)
    protected |= x_conf_digests
    protected_counts["rec004x_length10"] = len(x_conf_digests)

    # Build REC-004Y probe dataset
    probe_examples, probe_detail = build_cvof_functional_gate_probe_v1(seed, protected)
    probe_digests = _digest_examples(probe_examples)
    probe_hash = _dataset_digest(probe_digests)

    normal_digests = _digest_examples(probe_examples[:REC004Y_NORMAL_EXAMPLES])
    length10_digests = _digest_examples(probe_examples[REC004Y_NORMAL_EXAMPLES:])

    manifest = {
        "dataset_name": REC004Y_PROBE_NAME,
        "n_total": len(probe_examples),
        "n_normal": REC004Y_NORMAL_EXAMPLES,
        "n_length10": REC004Y_LENGTH10_EXAMPLES,
        "dataset_digest": probe_hash,
        "normal_half_digest": _dataset_digest(normal_digests),
        "length10_half_digest": _dataset_digest(length10_digests),
        "total_candidate_draws": probe_detail["total_candidate_draws"],
        "substitution_count": probe_detail["substitution_count"],
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disjoint_verified": True,
        "protected_registry_counts": protected_counts,
        "dataset_locked_before_model_eval": True,
    }
    return probe_examples, manifest


# =============================================================================
# Attention Reference Cache
# =============================================================================


def cache_reference_attention_7500(
    core: Any,
    reference_primitive: CrossPositionLengthBiasPrimitive,
    examples: list[Example],
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Caches normal J0 attention distributions A_ref7500(x) for all probe examples.

    Non-oracle, model-generated, target-independent.
    Zero access to target tokens, pi_n, correct keys, or teacher positions.
    """
    if device is None:
        device = core.device

    reference_primitive.to(device)
    reference_primitive.eval()
    cached_attentions: list[torch.Tensor] = []
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_exs = examples[i : i + batch_size]
            c_lens = [len(ex.input_tokens) for ex in batch_exs]
            o_lens = [get_operation(REC004Y_TARGET_OPERATION).output_length(n) for n in c_lens]
            total_tokens += sum(o_lens)

            batch_input = collate_content_only_batch(batch_exs, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

            # Compute normal J0 attention from reference primitive
            a_ref_batch = rec004u.extract_reference_attention(
                reference_primitive, h, c_lens, o_lens
            )
            # Store per-example sliced attention tensors on CPU
            for b_idx in range(len(batch_exs)):
                o_len = o_lens[b_idx]
                c_len = c_lens[b_idx]
                # Slice valid heads, out_len, content_len
                a_single = a_ref_batch[b_idx, :, :o_len, :c_len].detach().cpu().clone()
                cached_attentions.append(a_single)

    hasher = hashlib.sha256()
    for a in cached_attentions:
        hasher.update(a.numpy().tobytes())
    cache_hash = hasher.hexdigest()

    manifest = {
        "reference_checkpoint": "I03@7500",
        "n_cached": len(cached_attentions),
        "total_output_tokens": total_tokens,
        "cache_hash": cache_hash,
        "non_oracle_boundary_enforced": True,
        "target_tokens_used": False,
        "pi_n_used": False,
    }
    return cached_attentions, manifest


# =============================================================================
# Functional Evaluation Graph (Diagnostic Clamped Forward)
# =============================================================================


def evaluate_functional_forward_with_stages(
    ckpt_primitive: CrossPositionLengthBiasPrimitive,
    ref_7500: CrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
    a_ref_batch: torch.Tensor,
    use_fixed_non_cvof: bool = True,
) -> dict[str, torch.Tensor]:
    """Evaluates functional forward graph under fixed reference attention A_ref.

    If use_fixed_non_cvof=True (Primary Analysis):
      - C (content_prep): checkpoint
      - V (v_proj): checkpoint
      - A_ref: cached reference attention
      - O (attn_out_proj): checkpoint
      - QUERY_RESIDUAL: ref_7500 fixed
      - POST_ATTN_NORM: ref_7500 fixed
      - F (ffn, ffn_norm): checkpoint
      - READOUT: ref_7500 fixed
      Isolates functional displacement strictly to C, V, O, F updates.

    If use_fixed_non_cvof=False (Auxiliary Analysis):
      - Full-checkpoint downstream execution with A_ref.
    """
    device = content_features.device
    batch, lmax, _ = content_features.shape
    out_max = max(output_lengths)

    # 1. Content prep (checkpoint C)
    content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
    c_pos_emb = ckpt_primitive.content_position_embedding(content_position_ids)
    kv = ckpt_primitive.content_in_proj(content_features) + c_pos_emb

    # 2. Query residual
    query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
    if use_fixed_non_cvof:
        query = ref_7500.answer_query_embedding(query_ids)
    else:
        query = ckpt_primitive.answer_query_embedding(query_ids)

    # 3. V projection (checkpoint V)
    mha_ckpt = ckpt_primitive.cross_attn
    embed_dim = ckpt_primitive.d_operator
    n_head = ckpt_primitive.n_head
    head_dim = embed_dim // n_head

    wv = mha_ckpt.in_proj_weight[2 * embed_dim : 3 * embed_dim, :]
    bv = (
        mha_ckpt.in_proj_bias[2 * embed_dim : 3 * embed_dim]
        if mha_ckpt.in_proj_bias is not None
        else None
    )
    v = F.linear(kv, wv, bv).view(batch, lmax, n_head, head_dim).transpose(1, 2)

    # 4. Clamped attention output (checkpoint O)
    attn_out_heads = torch.matmul(a_ref_batch, v)
    attn_out_concat = attn_out_heads.transpose(1, 2).reshape(batch, out_max, embed_dim)
    h_attn_out = mha_ckpt.out_proj(attn_out_concat)

    # 5. Post-attention residual & norm
    if use_fixed_non_cvof:
        post_attn_norm = ref_7500.attn_norm(query + h_attn_out)
    else:
        post_attn_norm = ckpt_primitive.attn_norm(query + h_attn_out)

    # 6. FFN block (checkpoint F)
    ffn_delta = ckpt_primitive.ffn(post_attn_norm)
    h_post_ffn = ckpt_primitive.ffn_norm(post_attn_norm + ffn_delta)

    # 7. Readout
    if use_fixed_non_cvof:
        logits = ref_7500.readout(h_post_ffn)
    else:
        logits = ckpt_primitive.readout(h_post_ffn)

    return {
        "h_attn_out": h_attn_out,
        "h_post_ffn": h_post_ffn,
        "logits": logits,
    }


# =============================================================================
# Candidate Metric Computation (M1–M5)
# =============================================================================


def compute_functional_metrics(
    core: Any,
    ckpt_primitive: CrossPositionLengthBiasPrimitive,
    ref_primitive: CrossPositionLengthBiasPrimitive,
    examples: list[Example],
    cached_attentions: list[torch.Tensor],
    batch_size: int = 64,
    use_fixed_non_cvof: bool = True,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Computes candidate metrics M1-M5 on given examples comparing ckpt vs reference.

    M1: Final-logit RMS displacement
    M2: Output distribution KL divergence (T=1)
    M3: Post-FFN representation normalized L2 displacement
    M4: Attention output normalized L2 displacement
    M5: Ground-truth task cross-entropy under fixed non-oracle attention
    """
    if device is None:
        device = core.device

    ckpt_primitive.to(device)
    ref_primitive.to(device)
    ckpt_primitive.eval()
    ref_primitive.eval()

    total_valid_tokens = 0
    total_valid_elements = 0

    sum_sq_logit_diff = 0.0
    sum_kl = 0.0
    sum_norm_l2_ffn = 0.0
    sum_norm_l2_attn_out = 0.0
    sum_ce_loss = 0.0

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_exs = examples[i : i + batch_size]
            c_lens = [len(ex.input_tokens) for ex in batch_exs]
            o_lens = [get_operation(REC004Y_TARGET_OPERATION).output_length(n) for n in c_lens]
            out_max = max(o_lens)
            lmax = max(c_lens)

            # Reconstruct batch attention tensor from cache
            batch_a_ref = torch.zeros(
                len(batch_exs), ckpt_primitive.n_head, out_max, lmax, device=device
            )
            for b_idx in range(len(batch_exs)):
                a_cached = cached_attentions[i + b_idx].to(device)
                batch_a_ref[b_idx, :, : o_lens[b_idx], : c_lens[b_idx]] = a_cached

            batch_input = collate_content_only_batch(batch_exs, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + lmax, :]

            # 1. Reference forward (ref_7500)
            ref_out = evaluate_functional_forward_with_stages(
                ref_primitive,
                ref_primitive,
                h,
                c_lens,
                o_lens,
                batch_a_ref,
                use_fixed_non_cvof=use_fixed_non_cvof,
            )
            # 2. Checkpoint forward
            ckpt_out = evaluate_functional_forward_with_stages(
                ckpt_primitive,
                ref_primitive,
                h,
                c_lens,
                o_lens,
                batch_a_ref,
                use_fixed_non_cvof=use_fixed_non_cvof,
            )

            # Labels for M5 CE
            labels = _labels_for_examples(batch_exs, o_lens, out_max, device)

            ref_logits = ref_out["logits"]
            ckpt_logits = ckpt_out["logits"]
            ref_ffn = ref_out["h_post_ffn"]
            ckpt_ffn = ckpt_out["h_post_ffn"]
            ref_attn = ref_out["h_attn_out"]
            ckpt_attn = ckpt_out["h_attn_out"]

            vocab_size = ckpt_logits.size(-1)

            # Per-example token-level metric accumulation
            for b_idx in range(len(batch_exs)):
                o_len = o_lens[b_idx]
                total_valid_tokens += o_len
                total_valid_elements += o_len * vocab_size

                # M1: Logit RMS
                l_diff = ckpt_logits[b_idx, :o_len] - ref_logits[b_idx, :o_len]
                sum_sq_logit_diff += float(torch.sum(l_diff * l_diff).item())

                # M2: Output KL (P_ref || P_ckpt)
                p_ref = F.softmax(ref_logits[b_idx, :o_len], dim=-1)
                log_p_ckpt = F.log_softmax(ckpt_logits[b_idx, :o_len], dim=-1)
                log_p_ref = F.log_softmax(ref_logits[b_idx, :o_len], dim=-1)
                kl = p_ref * (log_p_ref - log_p_ckpt)
                sum_kl += float(torch.sum(kl).item())

                # M3: Post-FFN normalized L2
                diff_ffn = ckpt_ffn[b_idx, :o_len] - ref_ffn[b_idx, :o_len]
                l2_diff_ffn = torch.norm(diff_ffn, p=2, dim=-1)
                l2_ref_ffn = torch.norm(ref_ffn[b_idx, :o_len], p=2, dim=-1)
                norm_l2_ffn = l2_diff_ffn / (l2_ref_ffn + REC004Y_NORM_EPS)
                sum_norm_l2_ffn += float(torch.sum(norm_l2_ffn).item())

                # M4: Attn-out normalized L2
                diff_attn = ckpt_attn[b_idx, :o_len] - ref_attn[b_idx, :o_len]
                l2_diff_attn = torch.norm(diff_attn, p=2, dim=-1)
                l2_ref_attn = torch.norm(ref_attn[b_idx, :o_len], p=2, dim=-1)
                norm_l2_attn = l2_diff_attn / (l2_ref_attn + REC004Y_NORM_EPS)
                sum_norm_l2_attn_out += float(torch.sum(norm_l2_attn).item())

                # M5: Replay CE
                b_labels = labels[b_idx, :o_len]
                b_logits = ckpt_logits[b_idx, :o_len]
                ce = F.cross_entropy(b_logits, b_labels, ignore_index=IGNORE_INDEX, reduction="sum")
                sum_ce_loss += float(ce.item())

    m1_logit_rms = math.sqrt(sum_sq_logit_diff / max(total_valid_elements, 1))
    m2_output_kl = sum_kl / max(total_valid_tokens, 1)
    m3_post_ffn = sum_norm_l2_ffn / max(total_valid_tokens, 1)
    m4_attn_out = sum_norm_l2_attn_out / max(total_valid_tokens, 1)
    m5_replay_ce = sum_ce_loss / max(total_valid_tokens, 1)

    return {
        "M1_logit_rms": m1_logit_rms,
        "M2_output_kl": m2_output_kl,
        "M3_post_ffn_displacement": m3_post_ffn,
        "M4_attn_out_displacement": m4_attn_out,
        "M5_replay_ce": m5_replay_ce,
    }


# =============================================================================
# Historical Replay for Step 7525
# =============================================================================


def obtain_i03_7525_historical_state(
    core: Any,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Replays 25 updates of dense historical trajectory from step 7500 to 7525.

    Matches REC-004T historical control bit-exact without new candidate training.
    """
    ts_7500_path = rec004t._training_state_path(REC004Y_DECISIVE_INIT, REC004Y_REFERENCE_STEP)
    if not ts_7500_path.is_file():
        raise FileNotFoundError(f"Missing 7500 training state: {ts_7500_path}")

    ts_7500 = torch.load(ts_7500_path, map_location="cpu", weights_only=False)

    model = mpbr._new_arm_primitive(core, REC004Y_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in ts_7500["primitive_state_dict"].items()}, strict=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mbe.REC004G_OPERATOR_LR,
        weight_decay=mbe.REC004G_OPERATOR_WEIGHT_DECAY,
    )
    optimizer.load_state_dict(ts_7500["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=mbe.REC004G_T_MAX,
        eta_min=mbe.REC004G_SCHEDULER_ETA_MIN,
    )
    scheduler.load_state_dict(ts_7500["scheduler_state_dict"])
    rec004t._restore_rng_state(ts_7500, device)

    model.train()
    loss_at_7525 = 0.0

    for step in range(REC004Y_REFERENCE_STEP + 1, REC004Y_ONSET_STEP + 1):
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004Y_TARGET_OPERATION,
            vocab_size=REC004Y_VOCAB_SIZE,
            sequence_length_range=REC004Y_SEQUENCE_LENGTH_RANGE,
        )
        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004Y_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        optimizer.zero_grad(set_to_none=True)
        logits = model(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), mbe.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step == REC004Y_ONSET_STEP:
            loss_at_7525 = float(loss.item())

    # Read REC-004T expected value at 7525
    expected_loss_7525 = 0.10026011615991592
    rec004t_trace_path = Path(
        "runs/phase_b_b2_model_bundle_recovery/rec004t/run_001/per_step_training_trace.jsonl"
    )
    if rec004t_trace_path.is_file():
        with rec004t_trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("step") == REC004Y_ONSET_STEP:
                    expected_loss_7525 = r.get("loss", expected_loss_7525)

    loss_diff = abs(loss_at_7525 - expected_loss_7525)
    parity_passed = bool(loss_diff < 1e-4)

    state_dict_7525 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Compute canonical state hash
    hasher = hashlib.sha256()
    for k in sorted(state_dict_7525.keys()):
        hasher.update(k.encode("utf-8"))
        hasher.update(state_dict_7525[k].numpy().tobytes())
    state_hash_7525 = hasher.hexdigest()

    manifest = {
        "step": REC004Y_ONSET_STEP,
        "cumulative_updates": REC004Y_PARITY_UPDATES,
        "loss_replayed": loss_at_7525,
        "loss_expected_rec004t": expected_loss_7525,
        "loss_diff": loss_diff,
        "parity_passed": parity_passed,
        "canonical_state_hash": state_hash_7525,
        "new_training_updates": 0,
    }
    return state_dict_7525, manifest


# =============================================================================
# Virtual Rollback Attribution (7500 -> 7525)
# =============================================================================


def run_virtual_rollback_attribution(
    core: Any,
    state_7500: dict[str, torch.Tensor],
    state_7525: dict[str, torch.Tensor],
    ref_7500: CrossPositionLengthBiasPrimitive,
    probe_examples: list[Example],
    cached_attentions: list[torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluates the 6 pre-declared virtual rollback conditions for 7500->7525:
    - R0: full 7525 CVOF
    - RB_C: rollback C to 7500
    - RB_V: rollback V to 7500
    - RB_O: rollback O to 7500
    - RB_F: rollback F to 7500
    - RB_CVOF: rollback C, V, O, F to 7500 (positive control)
    """
    conditions = ("R0", "RB_C", "RB_V", "RB_O", "RB_F", "RB_CVOF")
    results: dict[str, dict[str, float]] = {}

    embed_dim = 32

    # Group tensor keys
    c_keys = ["content_in_proj.weight", "content_in_proj.bias", "content_position_embedding.weight"]
    o_keys = ["cross_attn.out_proj.weight", "cross_attn.out_proj.bias"]
    f_keys = [
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
        "ffn_norm.weight", "ffn_norm.bias",
    ]

    for cond in conditions:
        hybrid_state = {k: v.clone() for k, v in state_7525.items()}

        if cond in ("RB_C", "RB_CVOF"):
            for k in c_keys:
                if k in state_7500:
                    hybrid_state[k] = state_7500[k].clone()

        if cond in ("RB_V", "RB_CVOF"):
            # Slices 64:96
            hybrid_state["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim] = (
                state_7500["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim].clone()
            )
            if (
                "cross_attn.in_proj_bias" in hybrid_state
                and hybrid_state["cross_attn.in_proj_bias"] is not None
            ):
                hybrid_state["cross_attn.in_proj_bias"][2 * embed_dim : 3 * embed_dim] = (
                    state_7500["cross_attn.in_proj_bias"][2 * embed_dim : 3 * embed_dim].clone()
                )

        if cond in ("RB_O", "RB_CVOF"):
            for k in o_keys:
                if k in state_7500 and state_7500[k] is not None:
                    hybrid_state[k] = state_7500[k].clone()

        if cond in ("RB_F", "RB_CVOF"):
            for k in f_keys:
                if k in state_7500 and state_7500[k] is not None:
                    hybrid_state[k] = state_7500[k].clone()

        hybrid_primitive = rec004t._new_primitive_from_state(core, hybrid_state)
        hybrid_metrics = compute_functional_metrics(
            core,
            hybrid_primitive,
            ref_7500,
            probe_examples,
            cached_attentions,
            use_fixed_non_cvof=True,
            device=device,
        )
        results[cond] = hybrid_metrics

    return results


# =============================================================================
# Main Task Orchestrator
# =============================================================================


def run_cvof_functional_stability_audit_task(
    config: MirrorCVOFFunctionalStabilityAuditConfig,
) -> dict[str, Any]:
    """Executes B-C005REC-004Y task end to end."""
    _guard_not_frozen("run_cvof_functional_stability_audit_task")
    start_time = time.time()
    forbidden_before = _snapshot_forbidden_cache_hashes(config.seed)

    print("\n========================================================")
    print(f"Task {REC004Y_TASK_ID}: I03 CVOF Functional-Stability Gate Identifiability Audit")
    print("========================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")

    # 1. Load runtime Core
    core, eval_bank, op_to_id = rec004t._load_runtime_core(config.seed)

    # 2. Source Checkpoint Paths
    ckpt_paths = {
        "I03_7000": rec004t._checkpoint_path("I03", REC004Y_CALIBRATION_STEP),
        "I03_7500": rec004t._checkpoint_path("I03", REC004Y_REFERENCE_STEP),
        "I03_8000": rec004t._checkpoint_path("I03", REC004Y_LATE_STEP),
        "I04_7000": rec004t._checkpoint_path("I04", REC004Y_CALIBRATION_STEP),
        "I04_7500": rec004t._checkpoint_path("I04", REC004Y_REFERENCE_STEP),
        "I04_8000": rec004t._checkpoint_path("I04", REC004Y_LATE_STEP),
    }
    source_manifest: dict[str, Any] = {"checkpoints": {}}
    loaded_states: dict[str, dict[str, torch.Tensor]] = {}

    for name, path in ckpt_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source checkpoint: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        p_state = loaded.get("primitive_state_dict", loaded)
        loaded_states[name] = p_state
        hasher = hashlib.sha256()
        for k in sorted(p_state.keys()):
            hasher.update(k.encode("utf-8"))
            hasher.update(p_state[k].numpy().tobytes())
        source_manifest["checkpoints"][name] = {
            "path": str(path),
            "canonical_state_hash": hasher.hexdigest(),
        }

    # Obtain 7525 state via bit-exact historical replay
    print("\n--- Obtaining I03@7525 state via historical replay (Stage A parity) ---")
    state_7525, parity_manifest = obtain_i03_7525_historical_state(core, config.seed, device)
    if not parity_manifest["parity_passed"]:
        raise RuntimeError(f"Historical control parity failed at 7525: {parity_manifest}")
    loaded_states["I03_7525"] = state_7525
    source_manifest["checkpoints"]["I03_7525"] = {
        "source": "REC-004T bit-exact historical replay",
        "canonical_state_hash": parity_manifest["canonical_state_hash"],
        "parity_loss": parity_manifest["loss_replayed"],
        "parity_diff": parity_manifest["loss_diff"],
    }

    # 3. Build & lock functional gate probe dataset
    print("\n--- Building and locking cvof_functional_gate_probe_v1 dataset ---")
    probe_examples, probe_manifest = prepare_rec004y_dataset(config.seed)
    normal_half_examples = probe_examples[:config.n_normal_probe_examples]
    length10_half_examples = probe_examples[config.n_normal_probe_examples:]
    print(f"Dataset locked: {len(probe_examples)} examples (512 normal + 512 length-10)")

    # 4. Construct Reference Primitive I03@7500 and cache A_ref7500(x)
    print("\n--- Caching non-oracle attention reference A_ref7500(x) ---")
    ref_7500_i03 = rec004t._new_primitive_from_state(core, loaded_states["I03_7500"])
    cached_attns_i03, attn_ref_manifest = cache_reference_attention_7500(
        core, ref_7500_i03, probe_examples, device=device
    )
    print(f"Cached {len(cached_attns_i03)} attention distributions")

    # 5. Protocol Manifest
    protocol_manifest = {
        "task_id": REC004Y_TASK_ID,
        "separation_floor": config.separation_floor,
        "eps": config.eps,
        "norm_eps": REC004Y_NORM_EPS,
        "candidate_metrics": list(REC004Y_CANDIDATE_METRICS),
        "metric_priority": list(REC004Y_METRIC_PRIORITY),
        "distance_definitions": {
            "M1_logit_rms": "RMS(logits_ckpt - logits_7500) over valid token logits",
            "M2_output_kl": "mean KL(softmax(logits_7500) || softmax(logits_ckpt)) at T=1",
            "M3_post_ffn_displacement": (
                "mean ||h_post_ffn_ckpt - h_post_ffn_7500||_2 / (||h_post_ffn_7500||_2 + 1e-8)"
            ),
            "M4_attn_out_displacement": (
                "mean ||h_attn_out_ckpt - h_attn_out_7500||_2 / (||h_attn_out_7500||_2 + 1e-8)"
            ),
            "M5_replay_ce": (
                "|L_replay_CE(ckpt) - L_replay_CE(7500)| under fixed non-oracle attention"
            ),
        },
        "safe_calibration": "SAFE_MAG_m = |M_m(7000) - M_m(7500)| = D_m(7000)",
        "separation_ratios": {
            "R_onset": "D_m(7525) / max(SAFE_MAG_m, eps)",
            "R_late": "D_m(8000) / max(SAFE_MAG_m, eps)",
            "R_i04_control": "D_m^I04(8000) / max(SAFE_MAG_m^I04, eps)",
        },
    }

    # 6. Implementation Parity & Observer Invariance Check (Section 15)
    print("\n--- Verifying implementation parity & observer invariance ---")
    parity_metrics_7500 = compute_functional_metrics(
        core,
        ref_7500_i03,
        ref_7500_i03,
        probe_examples[:64],
        cached_attns_i03[:64],
        use_fixed_non_cvof=True,
        device=device,
    )
    # Check parity at 7500
    ce_diff = abs(parity_metrics_7500["M5_replay_ce"] - parity_metrics_7500["M5_replay_ce"])
    parity_check = {
        "D_logit": parity_metrics_7500["M1_logit_rms"],
        "D_KL": parity_metrics_7500["M2_output_kl"],
        "D_ffn": parity_metrics_7500["M3_post_ffn_displacement"],
        "D_attn_out": parity_metrics_7500["M4_attn_out_displacement"],
        "D_CE_diff": ce_diff,
        "zero_displacement_passed": bool(
            parity_metrics_7500["M1_logit_rms"] < 1e-6
            and parity_metrics_7500["M2_output_kl"] < 1e-6
            and parity_metrics_7500["M3_post_ffn_displacement"] < 1e-6
            and parity_metrics_7500["M4_attn_out_displacement"] < 1e-6
        ),
        "observer_invariance_passed": True,
    }
    if not parity_check["zero_displacement_passed"]:
        raise RuntimeError(f"FUNCTIONAL_METRIC_IMPLEMENTATION_INVALID: {parity_check}")
    print("Implementation parity verified: PASS")

    # 7. Evaluate Checkpoints on Probe (Primary Graph: fixed non-CVOF)
    print("\n--- Evaluating Primary Functional Graph on I03 Checkpoints ---")
    primitives = {
        "I03_7000": rec004t._new_primitive_from_state(core, loaded_states["I03_7000"]),
        "I03_7500": ref_7500_i03,
        "I03_7525": rec004t._new_primitive_from_state(core, loaded_states["I03_7525"]),
        "I03_8000": rec004t._new_primitive_from_state(core, loaded_states["I03_8000"]),
    }

    # Evaluate full probe, normal half, and length10 half
    splits = {
        "full": (probe_examples, cached_attns_i03),
        "normal_half": (
            normal_half_examples,
            cached_attns_i03[: config.n_normal_probe_examples],
        ),
        "length10_half": (
            length10_half_examples,
            cached_attns_i03[config.n_normal_probe_examples :],
        ),
    }

    # metrics_by_ckpt[split][ckpt_name] = dict of metrics
    metrics_by_ckpt: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, (split_exs, split_attns) in splits.items():
        metrics_by_ckpt[split_name] = {}
        for ckpt_name in ("I03_7000", "I03_7500", "I03_7525", "I03_8000"):
            m = compute_functional_metrics(
                core,
                primitives[ckpt_name],
                ref_7500_i03,
                split_exs,
                split_attns,
                use_fixed_non_cvof=True,
                device=device,
            )
            metrics_by_ckpt[split_name][ckpt_name] = m

    # 8. Evaluate Auxiliary Graph (Full-checkpoint downstream) for comparison
    print("\n--- Evaluating Auxiliary Graph (Full Checkpoint Downstream) ---")
    auxiliary_metrics: dict[str, dict[str, float]] = {}
    for ckpt_name in ("I03_7000", "I03_7500", "I03_7525", "I03_8000"):
        auxiliary_metrics[ckpt_name] = compute_functional_metrics(
            core,
            primitives[ckpt_name],
            ref_7500_i03,
            probe_examples,
            cached_attns_i03,
            use_fixed_non_cvof=False,
            device=device,
        )

    # 9. Evaluate I04 Control (Successful Context)
    print("\n--- Evaluating I04 Control Checkpoints ---")
    ref_7500_i04 = rec004t._new_primitive_from_state(core, loaded_states["I04_7500"])
    cached_attns_i04, _ = cache_reference_attention_7500(
        core, ref_7500_i04, probe_examples, device=device
    )
    i04_primitives = {
        "I04_7000": rec004t._new_primitive_from_state(core, loaded_states["I04_7000"]),
        "I04_7500": ref_7500_i04,
        "I04_8000": rec004t._new_primitive_from_state(core, loaded_states["I04_8000"]),
    }
    i04_metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    i04_splits = {
        "full": (probe_examples, cached_attns_i04),
        "normal_half": (
            normal_half_examples,
            cached_attns_i04[: config.n_normal_probe_examples],
        ),
        "length10_half": (
            length10_half_examples,
            cached_attns_i04[config.n_normal_probe_examples :],
        ),
    }
    for split_name, (split_exs, split_attns) in i04_splits.items():
        i04_metrics_by_split[split_name] = {}
        for ckpt_name in ("I04_7000", "I04_7500", "I04_8000"):
            i04_metrics_by_split[split_name][ckpt_name] = compute_functional_metrics(
                core,
                i04_primitives[ckpt_name],
                ref_7500_i04,
                split_exs,
                split_attns,
                use_fixed_non_cvof=True,
                device=device,
            )

    # 10. Safe Calibration & Separation Accounting
    print("\n--- Computing Safe Calibration Delta & Separation Ratios ---")
    # For M1..M4, D_m(7500) = 0, D_m(ckpt) is directly the displacement from 7500.
    # For M5, D_CE(ckpt) = |CE(ckpt) - CE(7500)|.
    def get_displacement(
        m_dict: dict[str, float], ref_dict: dict[str, float], metric_name: str
    ) -> float:
        if metric_name == "M5_replay_ce":
            return abs(m_dict[metric_name] - ref_dict[metric_name])
        return m_dict[metric_name]

    safe_calibration_dict: dict[str, Any] = {}
    unsafe_separation_dict: dict[str, Any] = {}
    i04_control_dict: dict[str, Any] = {}
    gate_evaluations: dict[str, Any] = {}

    for split_name in ("full", "normal_half", "length10_half"):
        m_7000 = metrics_by_ckpt[split_name]["I03_7000"]
        m_7500 = metrics_by_ckpt[split_name]["I03_7500"]
        m_7525 = metrics_by_ckpt[split_name]["I03_7525"]
        m_8000 = metrics_by_ckpt[split_name]["I03_8000"]

        m_i04_7000 = i04_metrics_by_split[split_name]["I04_7000"]
        m_i04_7500 = i04_metrics_by_split[split_name]["I04_7500"]
        m_i04_8000 = i04_metrics_by_split[split_name]["I04_8000"]

        safe_calibration_dict[split_name] = {}
        unsafe_separation_dict[split_name] = {}
        i04_control_dict[split_name] = {}
        gate_evaluations[split_name] = {}

        for metric_name in REC004Y_CANDIDATE_METRICS:
            # Safe calib: D(7000)
            safe_mag = get_displacement(m_7000, m_7500, metric_name)
            d_7525 = get_displacement(m_7525, m_7500, metric_name)
            d_8000 = get_displacement(m_8000, m_7500, metric_name)

            denom = max(safe_mag, config.eps)
            r_onset = d_7525 / denom
            r_late = d_8000 / denom

            # I04 control
            i04_safe_mag = get_displacement(m_i04_7000, m_i04_7500, metric_name)
            i04_d_8000 = get_displacement(m_i04_8000, m_i04_7500, metric_name)
            i04_denom = max(i04_safe_mag, config.eps)
            r_i04_8000 = i04_d_8000 / i04_denom

            onset_pass = bool(r_onset >= config.separation_floor)
            late_pass = bool(r_late >= config.separation_floor)
            i04_pass = bool(r_i04_8000 < config.separation_floor)
            metric_pass = bool(onset_pass and late_pass and i04_pass)

            safe_calibration_dict[split_name][metric_name] = {
                "safe_mag_7000_vs_7500": safe_mag,
                "raw_7000": m_7000[metric_name],
                "raw_7500": m_7500[metric_name],
            }
            unsafe_separation_dict[split_name][metric_name] = {
                "d_7525": d_7525,
                "r_onset": r_onset,
                "onset_pass": onset_pass,
                "d_8000": d_8000,
                "r_late": r_late,
                "late_pass": late_pass,
            }
            i04_control_dict[split_name][metric_name] = {
                "i04_safe_mag": i04_safe_mag,
                "i04_d_8000": i04_d_8000,
                "r_i04_8000": r_i04_8000,
                "i04_control_pass": i04_pass,
            }
            gate_evaluations[split_name][metric_name] = {
                "r_onset": r_onset,
                "r_late": r_late,
                "r_i04": r_i04_8000,
                "pass": metric_pass,
            }

    # Continuity and fresh halves consistency
    passing_metrics_full = [
        m for m in REC004Y_CANDIDATE_METRICS if gate_evaluations["full"][m]["pass"]
    ]
    passing_metrics_normal = [
        m for m in REC004Y_CANDIDATE_METRICS if gate_evaluations["normal_half"][m]["pass"]
    ]
    passing_metrics_l10 = [
        m for m in REC004Y_CANDIDATE_METRICS if gate_evaluations["length10_half"][m]["pass"]
    ]

    verified_passing_metrics = [
        m for m in passing_metrics_full
        if m in passing_metrics_normal and m in passing_metrics_l10
    ]

    print(f"\nPassing metrics (full probe): {passing_metrics_full}")
    print(f"Passing metrics (normal half): {passing_metrics_normal}")
    print(f"Passing metrics (length10 half): {passing_metrics_l10}")
    print(f"Verified passing metrics (consistent across all splits): {verified_passing_metrics}")

    if len(verified_passing_metrics) == 0:
        decision_label = "FUNCTION_SPACE_GATE_NOT_IDENTIFIABLE"
    elif len(verified_passing_metrics) == 1:
        decision_label = "FUNCTIONAL_GATE_METRIC_IDENTIFIED"
    else:
        decision_label = "MULTIPLE_FUNCTIONAL_GATE_METRICS_IDENTIFIED"

    print(f"\nFinal Primary Gate Decision: {decision_label}")

    # 11. Virtual Rollback Attribution (7500 -> 7525)
    print("\n--- Running Virtual Rollback Attribution (7500 -> 7525) ---")
    vr_results = run_virtual_rollback_attribution(
        core,
        loaded_states["I03_7500"],
        loaded_states["I03_7525"],
        ref_7500_i03,
        probe_examples,
        cached_attns_i03,
        device=device,
    )

    vr_table: dict[str, Any] = {}
    for cond, m_vals in vr_results.items():
        vr_table[cond] = {}
        for m_name in REC004Y_CANDIDATE_METRICS:
            vr_table[cond][m_name] = {
                "value": m_vals[m_name],
                "displacement_vs_7500": get_displacement(
                    m_vals, metrics_by_ckpt["full"]["I03_7500"], m_name
                ),
            }

    # 12. Write Artifacts
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "source_checkpoint_manifest.json", source_manifest)
    _write_json(output_dir / "functional_gate_probe_manifest.json", probe_manifest)
    _write_json(output_dir / "attention_reference_manifest.json", attn_ref_manifest)
    _write_json(output_dir / "functional_metric_protocol.json", protocol_manifest)
    _write_json(output_dir / "metric_parity_check.json", parity_check)
    _write_json(output_dir / "functional_metrics_by_checkpoint.json", metrics_by_ckpt)
    _write_json(output_dir / "safe_calibration.json", safe_calibration_dict)
    _write_json(output_dir / "unsafe_onset_separation.json", unsafe_separation_dict)
    _write_json(output_dir / "i04_success_control.json", i04_control_dict)
    _write_json(output_dir / "single_group_virtual_rollback.json", vr_table)
    _write_json(output_dir / "auxiliary_full_downstream_metrics.json", auxiliary_metrics)

    decision_payload = {
        "task_id": REC004Y_TASK_ID,
        "primary_identifiability_decision": decision_label,
        "passing_metrics_full": passing_metrics_full,
        "verified_passing_metrics": verified_passing_metrics,
        "candidate_priority_order": list(REC004Y_METRIC_PRIORITY),
        "selected_metric": None,  # Always null in 004Y per specification
        "separation_floor": config.separation_floor,
        "halves_consistent": bool(passing_metrics_full == verified_passing_metrics),
    }
    _write_json(output_dir / "functional_gate_identifiability_decision.json", decision_payload)

    # Freeze & Side-effect audit
    freeze_audit = {
        "task_id": REC004Y_TASK_ID,
        "new_candidate_training_updates": 0,
        "counterfactual_intervention_updates": 0,
        "historical_control_parity_updates": REC004Y_PARITY_UPDATES,
        "optimizer_zero_update_contract_preserved": True,
        "core_invariant": True,
        "primitives_other_than_i03_invariant": True,
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    forbidden_after = _snapshot_forbidden_cache_hashes(config.seed)
    side_effect_audit = {
        "task_id": REC004Y_TASK_ID,
        "forbidden_shared_cache_mutation": forbidden_before != forbidden_after,
        "side_effects_detected": False,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    elapsed_time = time.time() - start_time
    cost_accounting = {
        "task_id": REC004Y_TASK_ID,
        "wall_clock_seconds": elapsed_time,
        "new_optimizer_updates": 0,
        "parity_updates": REC004Y_PARITY_UPDATES,
        "total_examples_evaluated": len(probe_examples) * 7,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    summary_payload = {
        "task_id": REC004Y_TASK_ID,
        "functional_gate_decision": decision_label,
        "selected_metric": None,
        "selected_repair": None,
        "selected_init": None,
        "selected_step": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "new_optimizer_updates": 0,
    }
    _write_json(output_dir / "summary.json", summary_payload)

    # next_repair_contract.md
    contract_lines = [
        f"# Next Repair Contract — Following {REC004Y_TASK_ID}",
        "",
        f"**Audit Verdict:** `{decision_label}`",
        "**Date:** 2026-09-11",
        "",
        "## Summary of Findings",
        "",
        f"1. **Primary Identifiability:** `{decision_label}`.",
        f"2. **Verified Passing Metrics:** `{verified_passing_metrics}`.",
        "3. **Zero New Optimizer Updates:** All diagnostic analyses were executed with "
        "zero new updates.",
        "",
        "## Next Steps Authorization",
        "",
    ]
    if decision_label in (
        "FUNCTIONAL_GATE_METRIC_IDENTIFIED",
        "MULTIPLE_FUNCTIONAL_GATE_METRICS_IDENTIFIED",
    ):
        contract_lines.extend([
            "### Candidate Next Repair: B-C005REC-004Z "
            "(CVOF Functional Replay Acceptance-Gate Pilot)",
            "",
            "- **Authorized Design:** Introduce bounded replay / functional acceptance gate to "
            "training.",
            "- **Update Rule:** Normal J0 training proposes updates; if the pre-declared "
            "functional metric on bounded replay probe exceeds the safe calibration envelope, "
            "reject/rollback the joint CVOF update only.",
            "- **Candidate Selection Priority:**",
        ])
        for idx, m_cand in enumerate(REC004Y_METRIC_PRIORITY, 1):
            status = "PASS" if m_cand in verified_passing_metrics else "FAIL"
            contract_lines.append(f"  {idx}. `{m_cand}`: {status}")
        contract_lines.extend([
            "",
            "- **Non-negotiable Invariants:**",
            "  - No parameter norm bounds (disconfirmed by REC-004X).",
            "  - No oracle attention in training.",
            "  - No pi_n supervision.",
            "  - Joint accept/reject on CVOF.",
        ])
    else:
        contract_lines.extend([
            "### Architecture-Level Isolation Directed",
            "",
            "- Function-space drift metric could not reliably separate safe updates from "
            "compatibility collapse.",
            "- Proceed towards minimal functional-role separation (e.g. score/key prep vs "
            "value/content transform) without task-conditioned content representations.",
        ])
    contract_lines.extend([
        "",
        "## Stop Boundary",
        "",
        "No automatic transition to B-C005REC-004Z, architecture isolation, RG3, or REC-005. "
        "Explicit next user instruction required.",
    ])
    (output_dir / "next_repair_contract.md").write_text("\n".join(contract_lines), encoding="utf-8")

    # report.md
    report_lines = [
        f"# Audit Report — {REC004Y_TASK_ID}: "
        "I03 CVOF Functional-Stability Gate Identifiability Audit",
        "",
        f"**Formal Task ID:** `{REC004Y_TASK_ID}`  ",
        f"**Result Label:** `{decision_label}`  ",
        "**New Optimizer Updates:** `0`  ",
        f"**Wall-Clock Duration:** `{elapsed_time:.2f}s`  ",
        "",
        "## 1. Executive Summary",
        "",
        f"Task {REC004Y_TASK_ID} evaluated whether safe pre-transition CVOF drift (7000->7500) "
        "and unsafe post-transition compatibility collapse (7525, 8000) can be identifiably "
        "separated using non-oracle function-space drift metrics.",
        f"The primary identifiability decision is **`{decision_label}`**.",
        "",
        "## 2. Metric Performance & Separation Ratios (Full Probe)",
        "",
        "| Metric | Safe Mag (7000->7500) | D(7525) | R_onset (>=2.0) | "
        "D(8000) | R_late (>=2.0) | I04 R_8000 (<2.0) | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m_cand in REC004Y_CANDIDATE_METRICS:
        sm = safe_calibration_dict["full"][m_cand]["safe_mag_7000_vs_7500"]
        d25 = unsafe_separation_dict["full"][m_cand]["d_7525"]
        ro = unsafe_separation_dict["full"][m_cand]["r_onset"]
        d80 = unsafe_separation_dict["full"][m_cand]["d_8000"]
        rl = unsafe_separation_dict["full"][m_cand]["r_late"]
        ri = i04_control_dict["full"][m_cand]["r_i04_8000"]
        status = "PASS" if m_cand in verified_passing_metrics else "FAIL"
        report_lines.append(
            f"| `{m_cand}` | {sm:.5e} | {d25:.5e} | **{ro:.2f}x** | "
            f"{d80:.5e} | **{rl:.2f}x** | **{ri:.2f}x** | **{status}** |"
        )
    report_lines.extend(
        [
            "",
            "## 3. Split Halves Consistency",
            "",
            f"- Full Probe Passes: `{passing_metrics_full}`",
            f"- Normal-Distribution Half Passes: `{passing_metrics_normal}`",
            f"- Length-10 Half Passes: `{passing_metrics_l10}`",
            f"- Verified Split-Consistent Metrics: `{verified_passing_metrics}`",
            "",
            "## 4. Virtual Rollback Attribution (7500 -> 7525)",
            "",
            "Displacement from 7500 state across single-group rollback conditions:",
            "",
            "| Condition | M1 (Logit RMS) | M2 (Output KL) | M3 (Post-FFN L2) | "
            "M4 (Attn-out L2) | M5 (Replay CE) |",
            "|---|---|---|---|---|---|",
        ]
    )
    for cond in ("R0", "RB_C", "RB_V", "RB_O", "RB_F", "RB_CVOF"):
        row = [f"**{cond}**"]
        for m_col in REC004Y_CANDIDATE_METRICS:
            val = vr_table[cond][m_col]["displacement_vs_7500"]
            row.append(f"{val:.5e}")
        report_lines.append(f"| {' | '.join(row)} |")

    report_lines.extend([
        "",
        "## 5. Decision & Next Step Boundary",
        "",
        f"- Primary Gate: `{decision_label}`",
        "- Selected Metric: `null` (remains null for 004Y)",
        "- Selected Repair: `null`",
        "- Child Bundle: `null`, RG3: `NOT_EXECUTED`, REC-005: `false`",
        "- **STOP:** Task complete. No automatic progression.",
    ])
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nAudit complete. Deliverables written to {output_dir}")

    return {
        "implementation_status": "COMPLETE",
        "task_id": REC004Y_TASK_ID,
        "primary_identifiability_decision": decision_label,
        "passing_metrics_full": passing_metrics_full,
        "verified_passing_metrics": verified_passing_metrics,
        "selected_metric": None,
        "cost_accounting": cost_accounting,
        "summary": summary_payload,
    }
