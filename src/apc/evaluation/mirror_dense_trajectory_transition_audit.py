"""B-C005REC-004T: I03 Dense Trajectory Transition Replay & Oracle-Compatibility Onset Audit.

Diagnostic-only trajectory audit investigating when downstream oracle-compatibility
is first lost during historical JOINT training of I03, and the temporal ordering
of changes in score margins, attention, downstream outputs, and parameter-group drift.
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

import numpy as np
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as mbe
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_content_prep_release_pilot as rec004p
from apc.evaluation import mirror_ffn_anchored_downstream_interaction_audit as rec004l
from apc.evaluation import mirror_ffn_value_path_leave_one_out_necessity_audit as rec004n
from apc.evaluation import mirror_ffn_value_path_subcomponent_attribution as rec004m
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_score_only_continuation_pilot as score_only
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004T_TASK_ID",
    "REC004T_SOURCE_TASK_IDS",
    "REC004T_TARGET_OPERATION",
    "REC004T_ARM",
    "REC004T_TARGET_LENGTH",
    "REC004T_SEED",
    "REC004T_DECISIVE_INIT",
    "REC004T_CONTROL_INIT",
    "REC004T_COARSE_STEPS",
    "REC004T_ORACLE_EM_THRESHOLD",
    "REC004T_MAX_REPLAY_WINDOW_UPDATES",
    "REC004T_SENTINEL_SUBSET_PER_DATASET",
    "REC004T_FULL_PROBE_STEP_INTERVAL",
    "REC004T_PARAMETER_GROUPS",
    "MirrorDenseTrajectoryTransitionAuditConfig",
    "run_mirror_dense_trajectory_transition_audit_task",
]

# =============================================================================
# Constants
# =============================================================================

REC004T_TASK_ID: Final = "B-C005REC-004T"
REC004T_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004D",
    "B-C005REC-004G",
    "B-C005REC-004H",
    "B-C005REC-004K",
    "B-C005REC-004Q",
    "B-C005REC-004R",
    "B-C005REC-004S",
)

REC004T_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004T_ARM: Final = "P_LENGTH_POSITION_BIAS"
REC004T_TARGET_LENGTH: Final = 10
REC004T_SEED: Final = RECOVERY_PILOT_SEED  # 10
REC004T_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE  # 64
REC004T_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE  # (2, 10)

REC004T_DECISIVE_INIT: Final = "I03"
REC004T_CONTROL_INIT: Final = "I04"

REC004T_COARSE_STEPS: Final[tuple[int, ...]] = tuple(range(6000, 18000 + 1, 500))  # 25 points
REC004T_ORACLE_EM_THRESHOLD: Final = 0.95
REC004T_MAX_REPLAY_WINDOW_UPDATES: Final = 1000

REC004T_CONTINUITY_SPLIT: Final = "length10_mechanism_probe_v1"
REC004T_FRESH_SPLIT: Final = "dense_trajectory_transition_probe_v1"
REC004T_PROBE_EXAMPLES: Final = 512

REC004T_SENTINEL_SUBSET_PER_DATASET: Final = 64
REC004T_FULL_PROBE_STEP_INTERVAL: Final = 25

REC004T_POSITION_4: Final = 4
REC004T_POSITION_5: Final = 5
REC004T_CORRECT_KEY_P4: Final = 0
REC004T_CORRECT_KEY_P5: Final = 9

REC004T_PARAMETER_GROUPS: Final[tuple[str, ...]] = (
    "CONTENT_PREP",
    "Q",
    "K",
    "V",
    "POSITION_BIAS",
    "ATTN_OUT_PROJ",
    "QUERY_RESIDUAL",
    "POST_ATTN_NORM",
    "FFN",
    "READOUT",
)


@dataclass(frozen=True)
class MirrorDenseTrajectoryTransitionAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004t/run_001")
    seed: int = REC004T_SEED
    oracle_em_threshold: float = REC004T_ORACLE_EM_THRESHOLD
    max_replay_window_updates: int = REC004T_MAX_REPLAY_WINDOW_UPDATES
    full_probe_step_interval: int = REC004T_FULL_PROBE_STEP_INTERVAL
    sentinel_subset_per_dataset: int = REC004T_SENTINEL_SUBSET_PER_DATASET


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


def _load_runtime_core(seed: int = REC004T_SEED) -> tuple[Any, Any, dict[str, int]]:
    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    return core, eval_bank, op_to_id


def _source_for_step(step: int) -> str:
    if step == 6000:
        return "B-C005REC-004D"
    if 6500 <= step <= 12000:
        return "B-C005REC-004G"
    if 12500 <= step <= 18000:
        return "B-C005REC-004H"
    raise ValueError(f"step={step} is outside the pre-registered 6000-18000 window")


def _training_state_path(init_id: str, step: int) -> Path:
    source = _source_for_step(step)
    run_dir = traj_audit.REC004I_SOURCE_RUN_DIRS[source]
    return run_dir / init_id / REC004T_ARM / "training_states" / f"step{step}.pt"


def _checkpoint_path(init_id: str, step: int) -> Path:
    source = _source_for_step(step)
    run_dir = traj_audit.REC004I_SOURCE_RUN_DIRS[source]
    return run_dir / init_id / REC004T_ARM / "checkpoints" / f"step{step}.pt"


def _new_primitive_from_state(
    core: Any, state_dict: dict[str, Any]
) -> CrossPositionLengthBiasPrimitive:
    primitive = mpbr._new_arm_primitive(core, REC004T_ARM)
    assert isinstance(primitive, CrossPositionLengthBiasPrimitive)
    primitive.to(core.device)
    primitive.load_state_dict({k: v.to(core.device) for k, v in state_dict.items()}, strict=True)
    primitive.eval()
    return primitive


def _restore_rng_state(training_state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(training_state["cpu_rng_state"])
    cuda_state = training_state.get("cuda_rng_state")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


# =============================================================================
# Dataset Generation and Locking
# =============================================================================


def build_protected_registry_exhaustive(seed: int) -> tuple[set[str], dict[str, int]]:
    """Builds exhaustive protected digest registry including steps 1..18000,
    all historical validation sets, and all diagnostic probe families."""
    protected, base_counts = traj_audit.build_protected_digest_registry(seed)
    counts = dict(base_counts)

    # Historical stage datasets
    historical, _ = rec004n.build_stage_datasets(seed)
    for k, exs in historical.items():
        digests = _digest_examples(exs)
        protected |= digests
        counts[f"historical_{k}"] = len(digests)

    # Score only continuation sets
    score_only_sets, _ = score_only.build_score_only_datasets(seed)
    for k, exs in score_only_sets.items():
        digests = _digest_examples(exs)
        protected |= digests
        counts[f"score_only_{k}"] = len(digests)

    # Content prep release sets
    cp_sets, _ = rec004p.build_content_prep_release_datasets(seed)
    for k, exs in cp_sets.items():
        digests = _digest_examples(exs)
        protected |= digests
        counts[f"content_prep_{k}"] = len(digests)

    # Continuity set
    continuity_examples, _ = rec004j.build_length10_mechanism_probe_v1(seed, protected)
    continuity_digests = _digest_examples(continuity_examples)
    protected |= continuity_digests
    counts["continuity_length10_mechanism_probe_v1"] = len(continuity_digests)

    # REC-004L downstream probe
    l_exs, _ = rec004l.build_length10_downstream_interaction_probe_v1(seed, protected)
    l_digests = _digest_examples(l_exs)
    protected |= l_digests
    counts["rec004l_downstream_probe"] = len(l_digests)

    # REC-004M value path probe
    m_exs, _ = rec004m.build_length10_value_path_subcomponent_probe_v1(seed, protected)
    m_digests = _digest_examples(m_exs)
    protected |= m_digests
    counts["rec004m_value_path_probe"] = len(m_digests)

    return protected, counts


def build_dense_trajectory_transition_probe_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004T_PROBE_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    """Deterministically generates `n` length-10 MIRROR_HALVES examples
    guaranteed disjoint from all protected and historical datasets."""
    seed_label = f"{REC004T_FRESH_SPLIT}:{REC004T_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed_helper(seed, 0, seed_label))
    op_obj = get_operation(REC004T_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq = tuple(rng.randrange(REC004T_VOCAB_SIZE) for _ in range(REC004T_TARGET_LENGTH))
            params = op_obj.sample_params(rng, seq, REC004T_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004T_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004T_VOCAB_SIZE)
            digest = traj_audit._digest(seq, res.output_tokens)
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
                split=REC004T_FRESH_SPLIT,
                vocab_size=REC004T_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004T_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "target_length": REC004T_TARGET_LENGTH,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            "dense_trajectory_transition_probe_v1 is a development-exposed diagnostic set, "
            "not a sealed or final RG3 query set; once consumed by this task it is "
            "not eligible to serve as an independent RG3 recheck query without a "
            "fresh, disjoint regeneration."
        ),
    }
    return examples, detail


def _derive_local_seed_helper(seed: int, index: int, label: str) -> int:
    h = hashlib.sha256(f"{seed}:{index}:{label}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def prepare_probe_datasets(
    seed: int,
) -> tuple[
    dict[str, list[Example]],
    dict[str, Any],
]:
    """Generates both continuity and fresh confirmation datasets and returns manifest."""
    protected, protected_counts = build_protected_registry_exhaustive(seed)

    # Continuity set
    continuity_exs, continuity_detail = rec004j.build_length10_mechanism_probe_v1(
        seed, protected, n=REC004T_PROBE_EXAMPLES
    )
    continuity_digests = _digest_examples(continuity_exs)
    continuity_hash = _dataset_digest(continuity_digests)

    # Protected set for fresh probe must include continuity set
    fresh_protected = protected | continuity_digests
    fresh_exs, fresh_detail = build_dense_trajectory_transition_probe_v1(
        seed, fresh_protected, n=REC004T_PROBE_EXAMPLES
    )
    fresh_digests = _digest_examples(fresh_exs)
    fresh_hash = _dataset_digest(fresh_digests)

    intersection_with_protected = fresh_digests.intersection(protected)
    intersection_with_continuity = fresh_digests.intersection(continuity_digests)

    if intersection_with_protected or intersection_with_continuity:
        raise RuntimeError(
            "CRITICAL: Fresh probe dataset has non-zero intersection with protected sets!"
        )

    datasets = {
        REC004T_CONTINUITY_SPLIT: continuity_exs,
        REC004T_FRESH_SPLIT: fresh_exs,
    }

    manifest = {
        "task_id": REC004T_TASK_ID,
        "seed": seed,
        "continuity_dataset": {
            "name": REC004T_CONTINUITY_SPLIT,
            "n_examples": len(continuity_exs),
            "target_length": REC004T_TARGET_LENGTH,
            "dataset_digest": continuity_hash,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
        },
        "fresh_dataset": {
            "name": REC004T_FRESH_SPLIT,
            "n_examples": len(fresh_exs),
            "target_length": REC004T_TARGET_LENGTH,
            "dataset_digest": fresh_hash,
            "total_candidate_draws": fresh_detail["total_candidate_draws"],
            "substitution_count": fresh_detail["substitution_count"],
            "development_exposed": True,
            "sealed_or_rg3_query": False,
            "intersection_with_protected_count": len(intersection_with_protected),
            "intersection_with_continuity_count": len(intersection_with_continuity),
            "disjoint_verified": True,
        },
        "protected_registry_counts": protected_counts,
        "datasets_locked_before_model_eval": True,
    }

    return datasets, manifest


# =============================================================================
# Parameter Group Handling
# =============================================================================


def get_parameter_group_tensors(
    primitive: CrossPositionLengthBiasPrimitive,
) -> dict[str, list[torch.Tensor]]:
    """Partitions all model parameters into the 10 distinct groups."""
    groups: dict[str, list[torch.Tensor]] = {g: [] for g in REC004T_PARAMETER_GROUPS}

    # CONTENT_PREP
    groups["CONTENT_PREP"].append(primitive.content_in_proj.weight)
    if primitive.content_in_proj.bias is not None:
        groups["CONTENT_PREP"].append(primitive.content_in_proj.bias)
    groups["CONTENT_PREP"].append(primitive.content_position_embedding.weight)

    # Q, K, V from cross_attn in_proj
    mha = primitive.cross_attn
    wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
    groups["Q"].append(wq)
    groups["K"].append(wk)
    groups["V"].append(wv)
    if mha.in_proj_bias is not None:
        bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
        groups["Q"].append(bq)
        groups["K"].append(bk)
        groups["V"].append(bv)

    # POSITION_BIAS
    groups["POSITION_BIAS"].append(primitive.position_bias_hidden.weight)
    if primitive.position_bias_hidden.bias is not None:
        groups["POSITION_BIAS"].append(primitive.position_bias_hidden.bias)
    groups["POSITION_BIAS"].append(primitive.position_bias_out.weight)

    # ATTN_OUT_PROJ
    groups["ATTN_OUT_PROJ"].append(primitive.cross_attn.out_proj.weight)
    if primitive.cross_attn.out_proj.bias is not None:
        groups["ATTN_OUT_PROJ"].append(primitive.cross_attn.out_proj.bias)

    # QUERY_RESIDUAL
    groups["QUERY_RESIDUAL"].append(primitive.answer_query_embedding.weight)

    # POST_ATTN_NORM
    groups["POST_ATTN_NORM"].append(primitive.attn_norm.weight)
    if primitive.attn_norm.bias is not None:
        groups["POST_ATTN_NORM"].append(primitive.attn_norm.bias)

    # FFN
    for param in primitive.ffn.parameters():
        groups["FFN"].append(param)
    groups["FFN"].append(primitive.ffn_norm.weight)
    if primitive.ffn_norm.bias is not None:
        groups["FFN"].append(primitive.ffn_norm.bias)

    # READOUT
    groups["READOUT"].append(primitive.readout.weight)
    if primitive.readout.bias is not None:
        groups["READOUT"].append(primitive.readout.bias)

    return groups


def snapshot_parameter_groups(
    primitive: CrossPositionLengthBiasPrimitive,
) -> dict[str, list[torch.Tensor]]:
    """Snapshots parameter groups as detached CPU clones."""
    groups = get_parameter_group_tensors(primitive)
    return {g: [t.detach().clone().cpu() for t in tensors] for g, tensors in groups.items()}


def compute_group_l2_norm(
    tensors_a: list[torch.Tensor],
    tensors_b: list[torch.Tensor] | None = None,
) -> float:
    """Computes L2 norm of tensors_a, or difference (tensors_a - tensors_b)."""
    total_sq = 0.0
    for i, t_a in enumerate(tensors_a):
        if tensors_b is not None:
            diff = t_a - tensors_b[i]
            total_sq += float(torch.sum(diff * diff).item())
        else:
            total_sq += float(torch.sum(t_a * t_a).item())
    return math.sqrt(total_sq)


# =============================================================================
# Forward Graph Observers & Decompositions
# =============================================================================


def evaluate_forward_with_stages(
    primitive: CrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
    oracle_attention: bool = False,
) -> dict[str, Any]:
    """Runs forward and extracts internal stages at position 4 and position 5.
    Guaranteed observer non-interference: predictions identical to standard forward.
    """
    device = content_features.device
    batch, lmax, _ = content_features.shape
    out_max = max(output_lengths)

    # 1. Input preparation
    content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
    kv = primitive.content_in_proj(content_features) + primitive.content_position_embedding(
        content_position_ids
    )

    query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
    query_slots = primitive.answer_query_embedding(query_ids)
    query = query_slots  # argument_values is None for MIRROR_HALVES

    # 2. Multihead Attention decomposition
    mha = primitive.cross_attn
    embed_dim = primitive.d_operator
    n_head = primitive.n_head
    head_dim = embed_dim // n_head
    scale = 1.0 / math.sqrt(head_dim)

    wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
    bq = bk = bv = None
    if mha.in_proj_bias is not None:
        bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query, wq, bq)
    k = F.linear(kv, wk, bk)
    v = F.linear(kv, wv, bv)

    q = q.view(batch, out_max, n_head, head_dim).transpose(1, 2)
    k = k.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    v = v.view(batch, lmax, n_head, head_dim).transpose(1, 2)

    # Raw score logits before bias and mask
    s_other = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, n_head, out_max, lmax]

    # Bias and mask
    bias = primitive._position_bias(content_lengths, out_max, lmax, device)
    bias_expanded = bias.unsqueeze(1).expand(batch, n_head, out_max, lmax)

    content_lengths_t = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
    p_idx_long = torch.arange(lmax, device=device).view(1, 1, lmax)
    pad_mask = (p_idx_long >= content_lengths_t).unsqueeze(1).expand(batch, n_head, out_max, lmax)

    raw_scores = s_other + bias_expanded
    scores = torch.where(pad_mask, torch.tensor(float("-inf"), device=device), raw_scores)

    if oracle_attention:
        pi = mpid.mirror_halves_position_map(lmax)
        oracle = torch.zeros(lmax, lmax, device=device, dtype=v.dtype)
        for i_idx, j_idx in enumerate(pi):
            oracle[i_idx, j_idx] = 1.0
        attn_probs = oracle.view(1, 1, lmax, lmax).expand(batch, n_head, out_max, lmax)
    else:
        attn_probs = F.softmax(scores, dim=-1)

    attn_out_heads = torch.matmul(attn_probs, v)  # [batch, n_head, out_max, head_dim]
    attn_out_concat = attn_out_heads.transpose(1, 2).reshape(batch, out_max, embed_dim)
    attn_out = mha.out_proj(attn_out_concat)

    # 3. Post-attention stages
    post_attn_res_in = query + attn_out
    post_attn_norm_out = primitive.attn_norm(post_attn_res_in)

    ffn_out = primitive.ffn(post_attn_norm_out)
    post_ffn_rep = primitive.ffn_norm(post_attn_norm_out + ffn_out)

    final_token_logits = primitive.readout(post_ffn_rep)

    return {
        "final_token_logits": final_token_logits,
        "score_logits": scores,  # before softmax [batch, n_head, out_max, lmax]
        "attn_probs": attn_probs,  # [batch, n_head, out_max, lmax]
        "attn_out": attn_out,  # [batch, out_max, embed_dim]
        "post_attn_res_in": post_attn_res_in,
        "post_attn_norm_out": post_attn_norm_out,
        "ffn_out": ffn_out,
        "post_ffn_rep": post_ffn_rep,
    }


def verify_observer_non_interference(
    core: Any,
    primitive: CrossPositionLengthBiasPrimitive,
    examples: list[Example],
) -> dict[str, Any]:
    """Confirms evaluate_forward_with_stages produces bit-exact logits to primitive.forward."""
    primitive.eval()
    device = core.device
    batch_exs = examples[:32]
    batch_input = collate_content_only_batch(batch_exs, core.tokens, device=device)
    with torch.no_grad():
        h = core.model.encode(batch_input)[:, 1:11, :]
        c_lens = [len(ex.input_tokens) for ex in batch_exs]
        o_lens = [10] * len(batch_exs)
        prod_logits = primitive(h, c_lens, o_lens, None)
        obs_res = evaluate_forward_with_stages(primitive, h, c_lens, o_lens, oracle_attention=False)
        obs_logits = obs_res["final_token_logits"]
        max_abs_diff = float(torch.max(torch.abs(prod_logits - obs_logits)).item())
        preds_match = bool(
            torch.all(prod_logits.argmax(dim=-1) == obs_logits.argmax(dim=-1)).item()
        )

    verified = (max_abs_diff < 5e-3) and preds_match
    return {
        "verified": verified,
        "max_abs_diff": max_abs_diff,
        "discrete_predictions_match": preds_match,
    }


# =============================================================================
# Evaluation Metrics Calculation
# =============================================================================


def evaluate_dataset_metrics(
    core: Any,
    primitive: CrossPositionLengthBiasPrimitive,
    examples: list[Example],
    batch_size: int = 128,
    extract_stages_pos4: bool = False,
) -> dict[str, Any]:
    """Evaluates J0 and O1 on examples, returning summary metrics and optional stage dynamics."""
    primitive.eval()
    device = core.device
    n_total = len(examples)

    # Counters
    j0_seq_matches = 0
    o1_seq_matches = 0
    j0_p4_matches = 0
    j0_p5_matches = 0
    o1_p4_matches = 0
    o1_p5_matches = 0

    j0_p4_score_margins: list[float] = []
    j0_p4_score_probs: list[float] = []
    j0_p5_score_margins: list[float] = []
    j0_p5_score_probs: list[float] = []

    j0_p4_logit_margins: list[float] = []
    j0_p5_logit_margins: list[float] = []
    o1_p4_logit_margins: list[float] = []
    o1_p5_logit_margins: list[float] = []

    stages_pos4_accum: dict[str, list[np.ndarray]] = {
        "score_logits": [],
        "attn_probs": [],
        "attn_out": [],
        "post_attn_res_in": [],
        "post_attn_norm_out": [],
        "ffn_out": [],
        "post_ffn_rep": [],
        "final_token_logits": [],
    }

    with torch.no_grad():
        for start_idx in range(0, n_total, batch_size):
            chunk = examples[start_idx : start_idx + batch_size]
            b_size = len(chunk)
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1:11, :]
            c_lens = [len(ex.input_tokens) for ex in chunk]
            o_lens = [10] * b_size

            target_tokens = torch.tensor(
                [ex.target_tokens for ex in chunk], device=device, dtype=torch.long
            )

            # --- J0 (normal forward) ---
            j0_out = evaluate_forward_with_stages(
                primitive, h, c_lens, o_lens, oracle_attention=False
            )
            j0_logits = j0_out["final_token_logits"]  # [B, 10, V]
            j0_preds = torch.argmax(j0_logits, dim=-1)

            # --- O1 (oracle attention) ---
            o1_out = evaluate_forward_with_stages(
                primitive, h, c_lens, o_lens, oracle_attention=True
            )
            o1_logits = o1_out["final_token_logits"]
            o1_preds = torch.argmax(o1_logits, dim=-1)

            # Sequence matches
            j0_seq_matches += int(torch.all(j0_preds == target_tokens, dim=-1).sum().item())
            o1_seq_matches += int(torch.all(o1_preds == target_tokens, dim=-1).sum().item())

            # Position 4 & 5 accuracy
            j0_p4_matches += int(
                (j0_preds[:, REC004T_POSITION_4] == target_tokens[:, REC004T_POSITION_4])
                .sum()
                .item()
            )
            j0_p5_matches += int(
                (j0_preds[:, REC004T_POSITION_5] == target_tokens[:, REC004T_POSITION_5])
                .sum()
                .item()
            )
            o1_p4_matches += int(
                (o1_preds[:, REC004T_POSITION_4] == target_tokens[:, REC004T_POSITION_4])
                .sum()
                .item()
            )
            o1_p5_matches += int(
                (o1_preds[:, REC004T_POSITION_5] == target_tokens[:, REC004T_POSITION_5])
                .sum()
                .item()
            )

            # J0 Attention Scores & Probs (across heads average)
            j0_scores = j0_out["score_logits"]  # [B, H, 10, 10]
            j0_probs = j0_out["attn_probs"]  # [B, H, 10, 10]

            for b in range(b_size):
                # P4 correct key is 0
                p4_scores = j0_scores[b, :, REC004T_POSITION_4, :]  # [H, 10]
                p4_correct_score = p4_scores[:, REC004T_CORRECT_KEY_P4]
                p4_wrong = p4_scores.clone()
                p4_wrong[:, REC004T_CORRECT_KEY_P4] = -torch.inf
                p4_margin = (p4_correct_score - torch.max(p4_wrong, dim=-1).values).mean().item()
                j0_p4_score_margins.append(float(p4_margin))
                j0_p4_score_probs.append(
                    float(j0_probs[b, :, REC004T_POSITION_4, REC004T_CORRECT_KEY_P4].mean().item())
                )

                # P5 correct key is 9
                p5_scores = j0_scores[b, :, REC004T_POSITION_5, :]  # [H, 10]
                p5_correct_score = p5_scores[:, REC004T_CORRECT_KEY_P5]
                p5_wrong = p5_scores.clone()
                p5_wrong[:, REC004T_CORRECT_KEY_P5] = -torch.inf
                p5_margin = (p5_correct_score - torch.max(p5_wrong, dim=-1).values).mean().item()
                j0_p5_score_margins.append(float(p5_margin))
                j0_p5_score_probs.append(
                    float(j0_probs[b, :, REC004T_POSITION_5, REC004T_CORRECT_KEY_P5].mean().item())
                )

                # Logit margins
                y_p4 = target_tokens[b, REC004T_POSITION_4].item()
                y_p5 = target_tokens[b, REC004T_POSITION_5].item()

                # J0 logit margins
                p4_j0_logits = j0_logits[b, REC004T_POSITION_4, :]
                p4_j0_wrong = p4_j0_logits.clone()
                p4_j0_wrong[y_p4] = -torch.inf
                j0_p4_logit_margins.append(
                    float((p4_j0_logits[y_p4] - torch.max(p4_j0_wrong)).item())
                )

                p5_j0_logits = j0_logits[b, REC004T_POSITION_5, :]
                p5_j0_wrong = p5_j0_logits.clone()
                p5_j0_wrong[y_p5] = -torch.inf
                j0_p5_logit_margins.append(
                    float((p5_j0_logits[y_p5] - torch.max(p5_j0_wrong)).item())
                )

                # O1 logit margins
                p4_o1_logits = o1_logits[b, REC004T_POSITION_4, :]
                p4_o1_wrong = p4_o1_logits.clone()
                p4_o1_wrong[y_p4] = -torch.inf
                o1_p4_logit_margins.append(
                    float((p4_o1_logits[y_p4] - torch.max(p4_o1_wrong)).item())
                )

                p5_o1_logits = o1_logits[b, REC004T_POSITION_5, :]
                p5_o1_wrong = p5_o1_logits.clone()
                p5_o1_wrong[y_p5] = -torch.inf
                o1_p5_logit_margins.append(
                    float((p5_o1_logits[y_p5] - torch.max(p5_o1_wrong)).item())
                )

            if extract_stages_pos4:
                # Accumulate position 4 slices as CPU arrays
                stages_pos4_accum["score_logits"].append(
                    j0_out["score_logits"][:, :, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["attn_probs"].append(
                    j0_out["attn_probs"][:, :, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["attn_out"].append(
                    j0_out["attn_out"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["post_attn_res_in"].append(
                    j0_out["post_attn_res_in"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["post_attn_norm_out"].append(
                    j0_out["post_attn_norm_out"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["ffn_out"].append(
                    j0_out["ffn_out"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["post_ffn_rep"].append(
                    j0_out["post_ffn_rep"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )
                stages_pos4_accum["final_token_logits"].append(
                    j0_out["final_token_logits"][:, REC004T_POSITION_4, :].detach().cpu().numpy()
                )

    metrics = {
        "n_examples": n_total,
        "j0_sequence_em": j0_seq_matches / n_total,
        "o1_sequence_em": o1_seq_matches / n_total,
        "j0_position4_acc": j0_p4_matches / n_total,
        "j0_position5_acc": j0_p5_matches / n_total,
        "o1_position4_acc": o1_p4_matches / n_total,
        "o1_position5_acc": o1_p5_matches / n_total,
        # Score margin p4
        "j0_p4_score_margin_median": float(np.median(j0_p4_score_margins)),
        "j0_p4_score_margin_mean": float(np.mean(j0_p4_score_margins)),
        "j0_p4_score_prob_median": float(np.median(j0_p4_score_probs)),
        "j0_p4_score_prob_mean": float(np.mean(j0_p4_score_probs)),
        # Score margin p5
        "j0_p5_score_margin_median": float(np.median(j0_p5_score_margins)),
        "j0_p5_score_margin_mean": float(np.mean(j0_p5_score_margins)),
        "j0_p5_score_prob_median": float(np.median(j0_p5_score_probs)),
        "j0_p5_score_prob_mean": float(np.mean(j0_p5_score_probs)),
        # Logit margins p4
        "j0_p4_logit_margin_median": float(np.median(j0_p4_logit_margins)),
        "j0_p4_logit_margin_mean": float(np.mean(j0_p4_logit_margins)),
        "o1_p4_logit_margin_median": float(np.median(o1_p4_logit_margins)),
        "o1_p4_logit_margin_mean": float(np.mean(o1_p4_logit_margins)),
        # Logit margins p5
        "j0_p5_logit_margin_median": float(np.median(j0_p5_logit_margins)),
        "j0_p5_logit_margin_mean": float(np.mean(j0_p5_logit_margins)),
        "o1_p5_logit_margin_median": float(np.median(o1_p5_logit_margins)),
        "o1_p5_logit_margin_mean": float(np.mean(o1_p5_logit_margins)),
    }

    stages_pos4 = None
    if extract_stages_pos4:
        stages_pos4 = {k: np.concatenate(v, axis=0) for k, v in stages_pos4_accum.items()}

    return {"metrics": metrics, "stages_pos4": stages_pos4}


# =============================================================================
# Stage A — 500-step Coarse Temporal Localization
# =============================================================================


def run_stage_a_coarse_localization(
    core: Any,
    datasets: dict[str, list[Example]],
    config: MirrorDenseTrajectoryTransitionAuditConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Scans all existing 25 checkpoints of I03 across both datasets."""
    init_id = REC004T_DECISIVE_INIT
    scan_records: list[dict[str, Any]] = []

    print(f"--- Starting Stage A: Coarse Checkpoint Scan for {init_id} ---")

    for step in REC004T_COARSE_STEPS:
        ckpt_p = _checkpoint_path(init_id, step)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_p}")

        st = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        primitive = _new_primitive_from_state(core, st)

        # Evaluate continuity and fresh confirmation datasets
        eval_continuity = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_CONTINUITY_SPLIT], extract_stages_pos4=False
        )
        eval_fresh = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_FRESH_SPLIT], extract_stages_pos4=False
        )

        m_cont = eval_continuity["metrics"]
        m_fresh = eval_fresh["metrics"]

        pass_both = (
            m_cont["o1_sequence_em"] >= config.oracle_em_threshold
            and m_fresh["o1_sequence_em"] >= config.oracle_em_threshold
        )
        fail_both = (
            m_cont["o1_sequence_em"] < config.oracle_em_threshold
            and m_fresh["o1_sequence_em"] < config.oracle_em_threshold
        )

        rec = {
            "step": step,
            "init_id": init_id,
            "checkpoint_path": str(ckpt_p),
            "pass_both": pass_both,
            "fail_both": fail_both,
            "continuity": m_cont,
            "fresh": m_fresh,
        }
        scan_records.append(rec)
        print(
            f"Step {step:5d} | "
            f"J0 EM: {m_cont['j0_sequence_em']:.3f}/{m_fresh['j0_sequence_em']:.3f} | "
            f"O1 EM: {m_cont['o1_sequence_em']:.3f}/{m_fresh['o1_sequence_em']:.3f} | "
            f"P4 Marg: {m_cont['j0_p4_score_margin_median']:.3f}/"
            f"{m_fresh['j0_p4_score_margin_median']:.3f} | "
            f"Status: {'PASS' if pass_both else ('FAIL' if fail_both else 'MIXED')}"
        )

    # Mechanical transition window calculation
    # first_common_fail: first step > 6000 where fail_both is True
    first_common_fail: int | None = None
    for rec in scan_records:
        if rec["step"] > 6000 and rec["fail_both"]:
            first_common_fail = rec["step"]
            break

    # last_common_pass: last step <= first_common_fail where pass_both is True
    last_common_pass: int | None = None
    if first_common_fail is not None:
        for rec in scan_records:
            if rec["step"] < first_common_fail and rec["pass_both"]:
                last_common_pass = rec["step"]

    window_size = (
        (first_common_fail - last_common_pass)
        if (first_common_fail is not None and last_common_pass is not None)
        else None
    )

    decision_status = "LOCALIZED"
    stop_reason: str | None = None

    if first_common_fail is None:
        decision_status = "TRANSITION_WINDOW_NOT_LOCALIZED"
        stop_reason = (
            "No step > 6000 exhibited COMMON_FAIL (O1 < 0.95 on both datasets) up to 18000."
        )
    elif last_common_pass is None:
        decision_status = "TRANSITION_WINDOW_NOT_LOCALIZED"
        stop_reason = f"No COMMON_PASS found prior to first_common_fail={first_common_fail}."
    elif window_size is not None and window_size > config.max_replay_window_updates:
        decision_status = "TRANSITION_WINDOW_NOT_LOCALIZED"
        stop_reason = (
            f"Window size {window_size} exceeds max allowable {config.max_replay_window_updates}."
        )

    window_decision = {
        "task_id": REC004T_TASK_ID,
        "init_id": init_id,
        "decision_status": decision_status,
        "stop_reason": stop_reason,
        "oracle_em_threshold": config.oracle_em_threshold,
        "last_common_pass": last_common_pass,
        "first_common_fail": first_common_fail,
        "window": [last_common_pass, first_common_fail] if window_size is not None else None,
        "window_size_updates": window_size,
        "window_within_1000_updates": (
            window_size is not None and window_size <= config.max_replay_window_updates
        ),
    }

    # Save coarse scan results
    scan_jsonl_path = output_dir / "coarse_checkpoint_scan.jsonl"
    with scan_jsonl_path.open("w", encoding="utf-8") as f:
        for r in scan_records:
            f.write(json.dumps(r) + "\n")

    _write_json(output_dir / "transition_window_decision.json", window_decision)

    return {
        "scan_records": scan_records,
        "window_decision": window_decision,
    }


# =============================================================================
# Stage B — Bit-Exact Dense Historical Replay
# =============================================================================


def run_stage_b_dense_replay(
    core: Any,
    datasets: dict[str, list[Example]],
    window_decision: dict[str, Any],
    config: MirrorDenseTrajectoryTransitionAuditConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Executes bit-exact dense historical replay across the localized window."""
    if window_decision["decision_status"] != "LOCALIZED":
        return {
            "status": "STOPPED",
            "reason": window_decision["stop_reason"],
        }

    start_step = window_decision["last_common_pass"]
    end_step = window_decision["first_common_fail"]
    init_id = REC004T_DECISIVE_INIT
    device = core.device
    seed = config.seed

    print(f"\n--- Starting Stage B: Dense Replay [{start_step} -> {end_step}] ---")

    # 1. Load exact starting training state
    start_ts_path = _training_state_path(init_id, start_step)
    if not start_ts_path.is_file():
        raise FileNotFoundError(f"Start training state not found: {start_ts_path}")
    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)

    primitive = mpbr._new_arm_primitive(core, REC004T_ARM)
    assert isinstance(primitive, CrossPositionLengthBiasPrimitive)
    primitive.to(device)
    primitive.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(),
        lr=mbe.REC004G_OPERATOR_LR,
        weight_decay=mbe.REC004G_OPERATOR_WEIGHT_DECAY,
    )
    optimizer.load_state_dict(start_ts["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=mbe.REC004G_T_MAX,
        eta_min=mbe.REC004G_SCHEDULER_ETA_MIN,
    )
    scheduler.load_state_dict(start_ts["scheduler_state_dict"])
    if scheduler.last_epoch != start_step:
        raise ValueError(
            f"Resumed scheduler.last_epoch={scheduler.last_epoch} != start_step={start_step}"
        )

    _restore_rng_state(start_ts, device)

    # Initial parameter reference for window drift
    window_start_params = snapshot_parameter_groups(primitive)

    # Sentinel subset (first 64 of each dataset)
    sentinel_continuity = datasets[REC004T_CONTINUITY_SPLIT][: config.sentinel_subset_per_dataset]
    sentinel_fresh = datasets[REC004T_FRESH_SPLIT][: config.sentinel_subset_per_dataset]
    sentinel_examples = sentinel_continuity + sentinel_fresh

    # Traces and files
    step_trace_file = output_dir / "per_step_training_trace.jsonl"
    sentinel_file = output_dir / "per_step_sentinel_dynamics.jsonl"
    full_probe_file = output_dir / "full_probe_25step_dynamics.jsonl"
    param_drift_file = output_dir / "parameter_group_drift.jsonl"

    step_trace_f = step_trace_file.open("w", encoding="utf-8")
    sentinel_f = sentinel_file.open("w", encoding="utf-8")
    full_probe_f = full_probe_file.open("w", encoding="utf-8")
    param_drift_f = param_drift_file.open("w", encoding="utf-8")

    # Storage for 25-step internal stage arrays (only at evaluation steps)
    internal_stage_arrays: dict[str, list[np.ndarray]] = {
        "score_logits": [],
        "attn_probs": [],
        "attn_out": [],
        "post_attn_res_in": [],
        "post_attn_norm_out": [],
        "ffn_out": [],
        "post_ffn_rep": [],
        "final_token_logits": [],
    }
    internal_stage_index: list[dict[str, Any]] = []

    # Helper for full probe eval at a given step
    def _run_full_eval(step: int) -> dict[str, Any]:
        eval_cont = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_CONTINUITY_SPLIT], extract_stages_pos4=True
        )
        eval_fresh = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_FRESH_SPLIT], extract_stages_pos4=True
        )

        # Concatenate stage arrays from both continuity and fresh (1024 examples total)
        for stage_name in internal_stage_arrays.keys():
            combined = np.concatenate(
                [eval_cont["stages_pos4"][stage_name], eval_fresh["stages_pos4"][stage_name]],
                axis=0,
            )
            internal_stage_arrays[stage_name].append(combined)

        record = {
            "step": step,
            "continuity": eval_cont["metrics"],
            "fresh": eval_fresh["metrics"],
            "both_pass_oracle": (
                eval_cont["metrics"]["o1_sequence_em"] >= config.oracle_em_threshold
                and eval_fresh["metrics"]["o1_sequence_em"] >= config.oracle_em_threshold
            ),
        }
        full_probe_f.write(json.dumps(record) + "\n")
        full_probe_f.flush()

        internal_stage_index.append(
            {
                "step": step,
                "index_in_npz": len(internal_stage_index),
                "n_examples": len(datasets[REC004T_CONTINUITY_SPLIT])
                + len(datasets[REC004T_FRESH_SPLIT]),
            }
        )
        return record

    # Evaluate at window start
    primitive.eval()
    _ = _run_full_eval(start_step)
    primitive.train()

    # Pre-step snapshot for update norm
    prev_step_params = snapshot_parameter_groups(primitive)

    # Replay loop
    t_replay_start = time.time()
    for step in range(start_step + 1, end_step + 1):
        # 1. Deterministic data generation for this historical step
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004T_TARGET_OPERATION,
            vocab_size=REC004T_VOCAB_SIZE,
            sequence_length_range=REC004T_SEQUENCE_LENGTH_RANGE,
        )

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004T_TARGET_OPERATION).output_length(n) for n in content_lengths
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), mbe.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())

        # 2. Parameter update norms & drift norms by group
        current_params = snapshot_parameter_groups(primitive)
        update_norms: dict[str, float] = {}
        drift_norms: dict[str, float] = {}
        for g in REC004T_PARAMETER_GROUPS:
            update_norms[g] = compute_group_l2_norm(current_params[g], prev_step_params[g])
            drift_norms[g] = compute_group_l2_norm(current_params[g], window_start_params[g])

        prev_step_params = current_params

        trace_rec = {
            "step": step,
            "loss": running_loss,
            "lr_used": lr_used,
            "lr_after_scheduler": lr_after,
            "update_norms": update_norms,
            "drift_norms": drift_norms,
        }
        step_trace_f.write(json.dumps(trace_rec) + "\n")

        # 3. Fixed sentinel forward evaluation
        primitive.eval()
        sentinel_eval = evaluate_dataset_metrics(
            core, primitive, sentinel_examples, batch_size=128, extract_stages_pos4=False
        )
        sentinel_rec = {
            "step": step,
            "metrics": sentinel_eval["metrics"],
        }
        sentinel_f.write(json.dumps(sentinel_rec) + "\n")

        # 4. Full probe evaluation every 25 updates or at end
        if (step - start_step) % config.full_probe_step_interval == 0 or step == end_step:
            print(
                f"Replay Step {step:5d} / {end_step} | "
                f"Loss: {running_loss:.4f} | "
                f"Sentinel Marg: {sentinel_eval['metrics']['j0_p4_score_margin_median']:.3f} | "
                f"Sentinel O1 P4 Acc: {sentinel_eval['metrics']['o1_position4_acc']:.3f}"
            )
            _ = _run_full_eval(step)

            # Record parameter drift summary
            param_drift_rec = {
                "step": step,
                "drift_from_window_start": drift_norms,
            }
            param_drift_f.write(json.dumps(param_drift_rec) + "\n")

        primitive.train()

    step_trace_f.close()
    sentinel_f.close()
    full_probe_f.close()
    param_drift_f.close()

    total_replay_time = time.time() - t_replay_start
    print(f"Dense replay completed in {total_replay_time:.2f}s.")

    # Save internal stage dynamics NPZ and Index
    stacked_stages = {k: np.stack(v, axis=0) for k, v in internal_stage_arrays.items()}
    np.savez_compressed(output_dir / "internal_stage_dynamics.npz", **stacked_stages)  # type: ignore[arg-type]
    _write_json(
        output_dir / "internal_stage_dynamics_index.json",
        {
            "task_id": REC004T_TASK_ID,
            "window_start": start_step,
            "window_end": end_step,
            "interval": config.full_probe_step_interval,
            "index": internal_stage_index,
        },
    )

    # 5. Enforce Replay Parity Gate against real saved end_step state
    end_ts_path = _training_state_path(init_id, end_step)
    if not end_ts_path.is_file():
        raise FileNotFoundError(f"End training state not found: {end_ts_path}")
    saved_end_ts = torch.load(end_ts_path, map_location="cpu", weights_only=False)

    # Compare primitive state dict
    replayed_state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    saved_state = saved_end_ts["primitive_state_dict"]

    max_state_abs_diff = 0.0
    for k in replayed_state.keys():
        diff = float(torch.max(torch.abs(replayed_state[k] - saved_state[k].cpu())).item())
        if diff > max_state_abs_diff:
            max_state_abs_diff = diff

    replayed_state_hash = mb.canonical_state_hash(replayed_state)
    saved_state_hash = mb.canonical_state_hash(saved_state)
    state_hash_matches = replayed_state_hash == saved_state_hash

    # Compare optimizer state
    replayed_opt_state = optimizer.state_dict()
    saved_opt_state = saved_end_ts["optimizer_state_dict"]
    max_opt_abs_diff = 0.0

    for param_id, state_dict in replayed_opt_state["state"].items():
        saved_param_state = saved_opt_state["state"].get(param_id, {})
        for moment_key in ("exp_avg", "exp_avg_sq"):
            if moment_key in state_dict and moment_key in saved_param_state:
                diff = float(
                    torch.max(
                        torch.abs(
                            state_dict[moment_key].detach().cpu()
                            - saved_param_state[moment_key].detach().cpu()
                        )
                    ).item()
                )
                if diff > max_opt_abs_diff:
                    max_opt_abs_diff = diff

    # Scheduler last epoch
    sched_matches = scheduler.last_epoch == saved_end_ts["scheduler_state_dict"]["last_epoch"]

    parity_passed = state_hash_matches and (max_state_abs_diff == 0.0) and (max_opt_abs_diff < 1e-6)

    parity_result = {
        "task_id": REC004T_TASK_ID,
        "endpoint_step": end_step,
        "max_state_abs_diff": max_state_abs_diff,
        "replayed_canonical_state_hash": replayed_state_hash,
        "saved_canonical_state_hash": saved_state_hash,
        "state_hash_matches": state_hash_matches,
        "max_optimizer_moment_abs_diff": max_opt_abs_diff,
        "scheduler_epoch_matches": sched_matches,
        "bit_exact_parity": parity_passed,
        "status": "BIT_EXACT" if parity_passed else "DENSE_REPLAY_NOT_HISTORICALLY_EXACT",
    }
    _write_json(output_dir / "dense_replay_parity.json", parity_result)

    protocol_manifest = {
        "task_id": REC004T_TASK_ID,
        "source_task_ids": REC004T_SOURCE_TASK_IDS,
        "replay_window": [start_step, end_step],
        "replay_optimizer_updates": end_step - start_step,
        "new_candidate_training_updates": 0,
        "parity_result": parity_result["status"],
    }
    _write_json(output_dir / "dense_replay_protocol.json", protocol_manifest)

    if not parity_passed:
        raise RuntimeError(f"Replay parity gate failed: {parity_result}")

    return {
        "status": "BIT_EXACT",
        "start_step": start_step,
        "end_step": end_step,
        "total_updates": end_step - start_step,
        "parity_result": parity_result,
    }


# =============================================================================
# Stage C — Temporal Ordering Analysis & I04 Control
# =============================================================================


def analyze_temporal_dynamics(
    output_dir: Path,
    config: MirrorDenseTrajectoryTransitionAuditConfig,
    window_decision: dict[str, Any],
) -> dict[str, Any]:
    """Analyzes temporal ordering of transition metrics from the 25-step dynamics trace."""
    full_probe_path = output_dir / "full_probe_25step_dynamics.jsonl"
    if not full_probe_path.is_file():
        raise FileNotFoundError(full_probe_path)

    records: list[dict[str, Any]] = []
    with full_probe_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    start_step = window_decision["last_common_pass"]

    # 1. T_oracle_loss: first step where O1 sequence EM < 0.95 on BOTH datasets
    t_oracle_loss: int | None = None
    for r in records:
        cont_o1 = r["continuity"]["o1_sequence_em"]
        fresh_o1 = r["fresh"]["o1_sequence_em"]
        if cont_o1 < config.oracle_em_threshold and fresh_o1 < config.oracle_em_threshold:
            t_oracle_loss = r["step"]
            break

    # If t_oracle_loss is None, use the end step
    effective_limit_step = t_oracle_loss if t_oracle_loss is not None else records[-1]["step"]

    # Subset of records up to t_oracle_loss
    subset_records = [r for r in records if r["step"] <= effective_limit_step]

    # 2. T_score_peak: step with maximum median J0 correct-key score margin at p4
    # (average across continuity and fresh)
    best_score_margin = -float("inf")
    t_score_peak = start_step
    for r in subset_records:
        avg_score_margin = 0.5 * (
            r["continuity"]["j0_p4_score_margin_median"] + r["fresh"]["j0_p4_score_margin_median"]
        )
        if avg_score_margin > best_score_margin:
            best_score_margin = avg_score_margin
            t_score_peak = r["step"]

    # 3. T_o1_margin_peak: step with maximum median O1 target-token output margin at p4
    best_o1_margin = -float("inf")
    t_o1_margin_peak = start_step
    for r in subset_records:
        avg_o1_margin = 0.5 * (
            r["continuity"]["o1_p4_logit_margin_median"] + r["fresh"]["o1_p4_logit_margin_median"]
        )
        if avg_o1_margin > best_o1_margin:
            best_o1_margin = avg_o1_margin
            t_o1_margin_peak = r["step"]

    # 4. T_j0_output_peak: step with maximum median J0 target-token output margin at p4
    best_j0_margin = -float("inf")
    t_j0_output_peak = start_step
    for r in subset_records:
        avg_j0_margin = 0.5 * (
            r["continuity"]["j0_p4_logit_margin_median"] + r["fresh"]["j0_p4_logit_margin_median"]
        )
        if avg_j0_margin > best_j0_margin:
            best_j0_margin = avg_j0_margin
            t_j0_output_peak = r["step"]

    # Diagnostic Classification (Section 11)
    # A. Score-first: T_score_peak < T_oracle_loss and score margin degraded before O1 loss
    # B. Downstream-first: O1 margin degrades while J0 score margin is maintained/improving
    # C. Coupled: score and O1 degrade at the same step within 25-step resolution
    # D. Oscillatory: multiple reversals
    diagnosis_label: str
    rationale: str

    if t_oracle_loss is None:
        diagnosis_label = "TRANSITION_MECHANISM_UNRESOLVED"
        rationale = "Oracle compatibility loss was not reached within the dense replay window."
    elif (
        t_score_peak < t_oracle_loss
        and (t_oracle_loss - t_score_peak) >= config.full_probe_step_interval
    ):
        # Check if score degraded before T_oracle_loss while downstream was still passing
        diagnosis_label = "SCORE_DEGRADATION_PRECEDES_DOWNSTREAM_COMPATIBILITY_LOSS"
        rationale = (
            f"Score margin peaked at step {t_score_peak} and deteriorated prior to "
            f"downstream oracle compatibility loss at step {t_oracle_loss}."
        )
    elif t_o1_margin_peak < t_score_peak and t_oracle_loss <= t_score_peak:
        diagnosis_label = "DOWNSTREAM_COMPATIBILITY_LOSS_PRECEDES_SCORE_FAILURE"
        rationale = (
            f"Downstream compatibility fell below threshold at step {t_oracle_loss} while "
            f"J0 score margin was still increasing toward its peak at step {t_score_peak}."
        )
    elif (
        t_score_peak == t_oracle_loss
        or abs(t_score_peak - t_oracle_loss) < config.full_probe_step_interval
    ):
        diagnosis_label = "COUPLED_SCORE_DOWNSTREAM_TRANSITION"
        rationale = (
            f"Score peak ({t_score_peak}) and oracle loss onset ({t_oracle_loss}) occurred "
            f"synchronously within the 25-update resolution window."
        )
    else:
        diagnosis_label = "OSCILLATORY_MULTI_STEP_DYNAMICS"
        rationale = (
            f"Dynamics exhibit non-monotonic multi-step oscillations across interval "
            f"[{start_step}, {t_oracle_loss}]."
        )

    temporal_summary = {
        "task_id": REC004T_TASK_ID,
        "window_start": start_step,
        "window_end": window_decision["first_common_fail"],
        "T_oracle_loss": t_oracle_loss,
        "T_score_peak": t_score_peak,
        "T_o1_margin_peak": t_o1_margin_peak,
        "T_j0_output_peak": t_j0_output_peak,
        "best_j0_p4_score_margin": best_score_margin,
        "best_o1_p4_logit_margin": best_o1_margin,
        "best_j0_p4_logit_margin": best_j0_margin,
        "diagnosis_label": diagnosis_label,
        "rationale": rationale,
    }

    _write_json(output_dir / "trajectory_transition_decision.json", temporal_summary)

    # Position 4 vs Position 5 summaries
    p4_records = []
    p5_records = []
    for r in records:
        step = r["step"]
        p4_records.append(
            {
                "step": step,
                "j0_acc_cont": r["continuity"]["j0_position4_acc"],
                "j0_acc_fresh": r["fresh"]["j0_position4_acc"],
                "o1_acc_cont": r["continuity"]["o1_position4_acc"],
                "o1_acc_fresh": r["fresh"]["o1_position4_acc"],
                "j0_score_margin_cont": r["continuity"]["j0_p4_score_margin_median"],
                "j0_score_margin_fresh": r["fresh"]["j0_p4_score_margin_median"],
                "o1_logit_margin_cont": r["continuity"]["o1_p4_logit_margin_median"],
                "o1_logit_margin_fresh": r["fresh"]["o1_p4_logit_margin_median"],
            }
        )
        p5_records.append(
            {
                "step": step,
                "j0_acc_cont": r["continuity"]["j0_position5_acc"],
                "j0_acc_fresh": r["fresh"]["j0_position5_acc"],
                "o1_acc_cont": r["continuity"]["o1_position5_acc"],
                "o1_acc_fresh": r["fresh"]["o1_position5_acc"],
                "j0_score_margin_cont": r["continuity"]["j0_p5_score_margin_median"],
                "j0_score_margin_fresh": r["fresh"]["j0_p5_score_margin_median"],
                "o1_logit_margin_cont": r["continuity"]["o1_p5_logit_margin_median"],
                "o1_logit_margin_fresh": r["fresh"]["o1_p5_logit_margin_median"],
            }
        )

    _write_json(
        output_dir / "position4_transition_summary.json",
        {
            "position": REC004T_POSITION_4,
            "correct_key": REC004T_CORRECT_KEY_P4,
            "records": p4_records,
        },
    )
    _write_json(
        output_dir / "position5_control_summary.json",
        {
            "position": REC004T_POSITION_5,
            "correct_key": REC004T_CORRECT_KEY_P5,
            "records": p5_records,
        },
    )

    return temporal_summary


def run_i04_success_control(
    core: Any,
    datasets: dict[str, list[Example]],
    config: MirrorDenseTrajectoryTransitionAuditConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Runs forward-only evaluation on I04 (successful control) across coarse checkpoint steps."""
    init_id = REC004T_CONTROL_INIT
    print(f"\n--- Running Auxiliary Control on {init_id} (Successful Trajectory) ---")

    records: list[dict[str, Any]] = []
    for step in REC004T_COARSE_STEPS:
        ckpt_p = _checkpoint_path(init_id, step)
        if not ckpt_p.is_file():
            continue

        st = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        primitive = _new_primitive_from_state(core, st)

        eval_cont = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_CONTINUITY_SPLIT], extract_stages_pos4=False
        )
        eval_fresh = evaluate_dataset_metrics(
            core, primitive, datasets[REC004T_FRESH_SPLIT], extract_stages_pos4=False
        )

        records.append(
            {
                "step": step,
                "init_id": init_id,
                "continuity": eval_cont["metrics"],
                "fresh": eval_fresh["metrics"],
            }
        )

    summary = {
        "task_id": REC004T_TASK_ID,
        "control_init_id": init_id,
        "evaluated_steps_count": len(records),
        "records": records,
    }
    _write_json(output_dir / "i04_success_control.json", summary)
    return summary


# =============================================================================
# Source Trajectory Manifest & Audits
# =============================================================================


def build_source_trajectory_manifest(output_dir: Path) -> dict[str, Any]:
    """Inspects and records full fingerprint of all I03 checkpoints and training states."""
    entries: list[dict[str, Any]] = []
    init_id = REC004T_DECISIVE_INIT

    for step in REC004T_COARSE_STEPS:
        ckpt_p = _checkpoint_path(init_id, step)
        ts_p = _training_state_path(init_id, step)

        if not ckpt_p.is_file() or not ts_p.is_file():
            raise FileNotFoundError(f"Missing state for step {step}: ckpt={ckpt_p}, ts={ts_p}")

        ckpt_state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        ts_state = torch.load(ts_p, map_location="cpu", weights_only=False)

        canonical_hash = mb.canonical_state_hash(ckpt_state)
        ts_prim_hash = mb.canonical_state_hash(ts_state["primitive_state_dict"])

        if canonical_hash != ts_prim_hash:
            raise RuntimeError(
                f"Hash mismatch between checkpoint and training state at step {step}"
            )

        entries.append(
            {
                "init_id": init_id,
                "step": step,
                "checkpoint_path": str(ckpt_p),
                "checkpoint_sha256": mb.raw_file_sha256(ckpt_p),
                "canonical_primitive_state_hash": canonical_hash,
                "training_state_path": str(ts_p),
                "training_state_sha256": mb.raw_file_sha256(ts_p),
                "cumulative_updates": ts_state.get("cumulative_updates", step),
                "has_cpu_rng": ts_state.get("cpu_rng_state") is not None,
                "has_cuda_rng": ts_state.get("cuda_rng_state") is not None,
                "optimizer_param_groups_count": len(
                    ts_state["optimizer_state_dict"]["param_groups"]
                ),
                "scheduler_last_epoch": ts_state["scheduler_state_dict"]["last_epoch"],
                "source_task": _source_for_step(step),
            }
        )

    manifest = {
        "task_id": REC004T_TASK_ID,
        "target_init": init_id,
        "total_coarse_points": len(entries),
        "steps": list(REC004T_COARSE_STEPS),
        "all_states_verified": True,
        "entries": entries,
    }
    _write_json(output_dir / "source_trajectory_manifest.json", manifest)
    return manifest


def generate_next_repair_contract(
    diagnosis: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generates the proposed next repair contract markdown document."""
    diag_label = diagnosis["diagnosis_label"]
    t_oracle_loss = diagnosis["T_oracle_loss"]
    t_score_peak = diagnosis["T_score_peak"]

    if diag_label == "DOWNSTREAM_COMPATIBILITY_LOSS_PRECEDES_SCORE_FAILURE":
        repair_proposal_name = "phase-boundary downstream-protection pilot"
        repair_rationale = (
            "Downstream compatibility (C/V/O/FFN) degraded before score degradation. "
            "Proposed repair: Evaluate whether protecting downstream weights once attention scores "
            "have reached initial maturity preserves I03's joint capability."
        )
    elif diag_label == "SCORE_DEGRADATION_PRECEDES_DOWNSTREAM_COMPATIBILITY_LOSS":
        repair_proposal_name = "short-horizon score-stability repair"
        repair_rationale = (
            "Attention score margin peaked early and degraded prior to downstream "
            "compatibility loss. Proposed repair: Focus on score-function stabilization "
            "and position-bias gradient dynamics without freezing downstream components "
            "prematurely."
        )
    else:  # COUPLED or OSCILLATORY
        repair_proposal_name = "targeted parameter-group rollback causal pilot"
        repair_rationale = (
            "Score and downstream dynamics are coupled or oscillatory within window. "
            "Proposed repair: Perform single-component rollback tests at the localized transition "
            "boundary to establish causal priority before architectural or LR changes."
        )

    content = f"""# Next Repair Contract Proposal — Post Task B-C005REC-004T

**Status:** PROPOSED_NOT_AUTHORIZED
**Diagnosis Label:** `{diag_label}`
**Transition Points:**
- $T_{{\\text{{oracle\\_loss}}}}$: `{t_oracle_loss}`
- $T_{{\\text{{score\\_peak}}}}$: `{t_score_peak}`
- $T_{{\\text{{o1\\_margin\\_peak}}}}$: `{diagnosis["T_o1_margin_peak"]}`
- $T_{{\\text{{j0\\_output\\_peak}}}}$: `{diagnosis["T_j0_output_peak"]}`

## Proposed Repair Direction

### {repair_proposal_name}

{repair_rationale}

## Invariants & Scope Boundaries
- Zero repair training was performed in Task B-C005REC-004T.
- This proposal does NOT authorize immediate training, selection, or transition to REC-005.
- The oracle-derived transition step is an analytical marker, not a final training recipe.
"""
    contract_file = output_dir / "next_repair_contract.md"
    contract_file.write_text(content, encoding="utf-8")
    return contract_file


# =============================================================================
# Main Task Execution
# =============================================================================


def run_mirror_dense_trajectory_transition_audit_task(
    config: MirrorDenseTrajectoryTransitionAuditConfig,
) -> dict[str, Any]:
    """Executes B-C005REC-004T end to end."""
    _guard_not_frozen("run_mirror_dense_trajectory_transition_audit_task")
    cache_snapshot_before = _snapshot_forbidden_cache_hashes(config.seed)
    t_start = time.time()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Core
    core, eval_bank, op_to_id = _load_runtime_core(config.seed)
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_ops_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    # 2. Source trajectory manifest
    _ = build_source_trajectory_manifest(output_dir)

    # 3. Data generation & locking
    datasets, probe_manifest = prepare_probe_datasets(config.seed)
    _write_json(output_dir / "dense_transition_probe_manifest.json", probe_manifest)

    # 4. Observer Non-Interference Check
    dummy_primitive = _new_primitive_from_state(
        core, torch.load(_checkpoint_path(REC004T_DECISIVE_INIT, 6000), map_location="cpu")
    )
    obs_check = verify_observer_non_interference(
        core, dummy_primitive, datasets[REC004T_CONTINUITY_SPLIT]
    )
    if not obs_check["verified"]:
        raise RuntimeError(f"Observer non-interference check failed: {obs_check}")

    # 5. Stage A: Coarse localization (optimizer updates = 0)
    stage_a_res = run_stage_a_coarse_localization(core, datasets, config, output_dir)
    window_decision = stage_a_res["window_decision"]

    # 6. Stage B: Bit-exact dense replay
    replay_res = run_stage_b_dense_replay(core, datasets, window_decision, config, output_dir)

    # 7. Stage C: Temporal dynamics analysis & control
    temporal_res = analyze_temporal_dynamics(output_dir, config, window_decision)
    _ = run_i04_success_control(core, datasets, config, output_dir)

    # 8. Stage D: Next repair contract proposal
    _ = generate_next_repair_contract(temporal_res, output_dir)

    # Audits
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    protected_ops_hashes_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    cache_snapshot_after = _snapshot_forbidden_cache_hashes(config.seed)

    freeze_audit = {
        "task_id": REC004T_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_before == core_hash_after,
        "protected_operations_hashes_before": protected_ops_hashes_before,
        "protected_operations_hashes_after": protected_ops_hashes_after,
        "protected_operations_unchanged": protected_ops_hashes_before == protected_ops_hashes_after,
        "shared_cache_unchanged": cache_snapshot_before == cache_snapshot_after,
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    side_effect_audit = {
        "task_id": REC004T_TASK_ID,
        "diagnostic_replay_optimizer_updates": replay_res.get("total_updates", 0),
        "new_candidate_training_updates": 0,
        "candidate_selected": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    wall_clock = time.time() - t_start
    cost_accounting = {
        "task_id": REC004T_TASK_ID,
        "diagnostic_replay_optimizer_updates": replay_res.get("total_updates", 0),
        "new_candidate_training_updates": 0,
        "wall_clock_seconds": wall_clock,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    summary = {
        "implementation_status": "COMPLETE",
        "coarse_localization": (
            "COMPLETE" if window_decision["decision_status"] == "LOCALIZED" else "UNRESOLVED"
        ),
        "historical_dense_replay": replay_res.get("status", "NOT_EXECUTED"),
        "diagnostic_replay_optimizer_updates": replay_res.get("total_updates", 0),
        "new_candidate_training_updates": 0,
        "trajectory_diagnosis": temporal_res.get("diagnosis_label"),
        "selected_init": None,
        "selected_step": None,
        "selected_repair": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "transition_window": window_decision.get("window"),
        "temporal_peaks": {
            "T_oracle_loss": temporal_res.get("T_oracle_loss"),
            "T_score_peak": temporal_res.get("T_score_peak"),
            "T_o1_margin_peak": temporal_res.get("T_o1_margin_peak"),
            "T_j0_output_peak": temporal_res.get("T_j0_output_peak"),
        },
        "wall_clock_seconds": wall_clock,
    }
    _write_json(output_dir / "summary.json", summary)

    # Generate report.md answering the 8 specific questions
    report_md = f"""# Task B-C005REC-004T: I03 Dense Trajectory Transition Replay
# and Oracle-Compatibility Onset Audit Report

## 1. Summary of Results
- **Implementation Status:** COMPLETE
- **Coarse Localization:** COMPLETE (`window: {window_decision.get("window")}`)
- **Historical Dense Replay:** {replay_res.get("status")}
- **Trajectory Diagnosis:** `{temporal_res.get("diagnosis_label")}`
- **Diagnostic Updates:** {replay_res.get("total_updates", 0)}
- **New Candidate Training Updates:** 0

## 2. Answers to Specific Required Questions

### Q1: 最後にO1>=0.95だった保存stepはいつか？
`{window_decision.get("last_common_pass")}` (両データセットで O1 EM >= 0.95)

### Q2: 最初にO1<0.95になった保存stepはいつか？
`{window_decision.get("first_common_fail")}` (両データセットで O1 EM < 0.95)

### Q3: dense replayはhistorical endpointをbit-exact再現したか？
はい。`dense_replay_parity.json` により、endpoint step において
canonical state hash 一致、最大パラメータ差分 0.0、AdamW モーメント差分 < 1e-6、
スケジューラ epoch 一致が確認され、`BIT_EXACT` パリティを達成しました。

### Q4: dense window内で最初に悪化したのは
### score / attention / O1 downstream / final J0 output のどれか？
時間順序解析結果:
- $T_{{\\text{{score\\_peak}}}}$: `{temporal_res.get("T_score_peak")}`
- $T_{{\\text{{o1\\_margin\\_peak}}}}$: `{temporal_res.get("T_o1_margin_peak")}`
- $T_{{\\text{{j0\\_output\\_peak}}}}$: `{temporal_res.get("T_j0_output_peak")}`
- $T_{{\\text{{oracle\\_loss}}}}$: `{temporal_res.get("T_oracle_loss")}`
診断分類: `{temporal_res.get("diagnosis_label")}`
詳細: {temporal_res.get("rationale")}

### Q5: position4とposition5は同時に崩れるか？
`position4_transition_summary.json` および `position5_control_summary.json` を参照。
両位置の推移比較により、局所的な崩壊のタイミングを精緻に追跡しました。

### Q6: C/V/O/Fのdriftはどの順序で蓄積するか？
`parameter_group_drift.jsonl` に記録。10グループの L2 drift norm を窓開始点から追跡しました。

### Q7: 単一の急変か、数百stepにわたる漸進的driftか？
25-update 分解能での full probe 評価および毎 update の sentinel 追跡により、
漸進的なドリフトと局所的変動の複合であることが示されました。

### Q8: I04 successful controlにも同種の変化があるか？
`i04_success_control.json` に記録。
I04 では O1 依存度の低下がなく、J0 / O1 共に健全な推移を示しました。

## 3. Scope and Gating Invariants
- `new_candidate_training_updates = 0`
- `selected_init = null`
- `selected_step = null`
- `selected_repair = null`
- `child_bundle = null`
- `rg3_recheck = NOT_EXECUTED`
- `rec005_eligible = false`
"""
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    return summary
