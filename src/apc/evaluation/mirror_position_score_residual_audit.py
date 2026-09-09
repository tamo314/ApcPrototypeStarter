"""B-C005REC-004E: MIRROR_HALVES Position-Score Residual Audit & Next-Repair
Contract.

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_POSITION_SCORE_RESIDUAL_AUDIT.md`.
Inserted between `B-C005REC-004D`'s result (ADR-0099, `candidate_status:
POSITION_BIAS_VALIDATION_NOT_MET`, saved) and any future `B-C005REC-005`:
audits REC-004D's saved P/I01-I05 (`CrossPositionLengthBiasPrimitive`)
checkpoints -- position-bias enumeration, score decomposition, read-only
forward-only counterfactual interventions, and the late-phase training
curve -- then writes at most one proposed (not authorized) next repair
contract. **Zero new optimizer updates.** No weights, Core, parent bundle,
router, ArgumentScorer, verifier, fixed 3 candidates, or shared cache are
ever modified; no new query set is generated or consumed; no RG3 recheck
runs.

Stage lettering (A-D) matches the task doc:
  A. lock the exact REC-004D source artifacts/hashes this audit reads,
     replay P/I01-I05's step=6000 predictions against REC-004D's own saved
     `per_length_position_metrics.json` (STOP on any unexplained mismatch),
     and verify a read-only "score observer" (a parallel `need_weights=True`
     hook through the SAME `cross_attn` submodule/weights, plus a manual
     Q/K/V reconstruction used only where the hook API cannot isolate the
     existing score term) reproduces the real forward's discrete
     predictions before trusting any score decomposition built from it.
  B. enumerate the real `_position_bias` value at every legal (i, j, n)
     pair (n=6..10, 330 pairs) for every saved P checkpoint (13 steps x 5
     inits = 65 grids, 21450 scalars), plus the hidden-unit activations
     that produce it, and the row-centered bias.
  C. decompose the real attention score into the existing term and the new
     bias term on a small observation subset, then run the task doc's
     fixed, read-only forward-only intervention matrix (J0-J6) on the full
     diagnostic suite -- never adopted as a production change.
  D. state the evidence-backed residual diagnosis and write
     `next_repair_contract.md` -- a PROPOSED_NOT_AUTHORIZED (or
     EVIDENCE_INSUFFICIENT) spec for at most one next repair, never
     implemented or trained here.

`selected_init`, `selected_intervention`, and `child_bundle` are fixed at
`null` and `rg3_recheck` at `"NOT_EXECUTED"` regardless of any measured
result -- this task never selects, publishes, or rechecks anything.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_position_bias_repair as bias_repair
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.evaluation.unified_oracle_causal_benchmark import _generate_parameter_free_examples
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004E_TASK_ID",
    "REC004E_SOURCE_TASK_ID",
    "REC004E_SOURCE_RUN_DIR",
    "REC004E_INIT_IDS",
    "REC004E_LEGAL_LENGTHS",
    "REC004E_CHECKPOINT_STEPS",
    "REC004E_DECISIVE_STEP",
    "REC004E_FIXED_POSITIONS",
    "REC004E_J_IDS",
    "MirrorPositionScoreResidualAuditConfig",
    "run_mirror_position_score_residual_audit_task",
]

# =============================================================================
# Constants -- inherited from REC-004D's own pre-registered values wherever
# possible (never re-derived/guessed), plus this task's own new-designed
# observation/intervention sizes (task doc section 1.4/3-5), pre-registered
# here rather than chosen after seeing results.
# =============================================================================

REC004E_TASK_ID: Final = "B-C005REC-004E"
REC004E_SOURCE_TASK_ID: Final = "B-C005REC-004D"
REC004E_SOURCE_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004d/run_001")

REC004E_TARGET_OPERATION: Final = bias_repair.REC004D_TARGET_OPERATION
REC004E_TARGET_PHYSICAL_ID: Final = bias_repair.REC004D_TARGET_PHYSICAL_ID
REC004E_INIT_IDS: Final[tuple[str, ...]] = bias_repair.REC004D_INIT_IDS
REC004E_ARM_U: Final = bias_repair.REC004D_ARM_U
REC004E_ARM_P: Final = bias_repair.REC004D_ARM_P

REC004E_LEGAL_LENGTHS: Final[tuple[int, ...]] = bias_repair.REC004D_LEGAL_LENGTHS  # (6,7,8,9,10)
REC004E_CHECKPOINT_INTERVAL: Final = bias_repair.REC004D_CHECKPOINT_INTERVAL  # 500
REC004E_MAX_STEP: Final = bias_repair.REC004D_MAX_UPDATES_PER_RUN  # 6000
REC004E_CHECKPOINT_STEPS: Final[tuple[int, ...]] = tuple(
    range(0, REC004E_MAX_STEP + 1, REC004E_CHECKPOINT_INTERVAL)
)  # 13 steps: 0,500,...,6000
REC004E_DECISIVE_STEP: Final = bias_repair.REC004D_DECISIVE_STEP  # 6000

REC004E_EXISTING_VALIDATION_SPLIT: Final = bias_repair.REC004D_EXISTING_VALIDATION_SPLIT
REC004E_EXISTING_VALIDATION_EXAMPLES: Final = bias_repair.REC004D_EXISTING_VALIDATION_EXAMPLES
REC004E_VOCAB_SIZE: Final = bias_repair.REC004D_VOCAB_SIZE
REC004E_SEQUENCE_LENGTH_RANGE: Final = bias_repair.REC004D_SEQUENCE_LENGTH_RANGE
REC004E_LENGTH_REF: Final = bias_repair.REC004D_LENGTH_REF  # 32
REC004E_BIAS_HIDDEN_DIM: Final = bias_repair.REC004D_BIAS_HIDDEN_DIM  # 32

REC004E_LENGTH_BALANCED_SPLIT: Final = mpid.REC004C_LENGTH_BALANCED_SPLIT
REC004E_LENGTH_BALANCED_PER_LENGTH: Final = mpid.REC004C_LENGTH_BALANCED_PER_LENGTH  # 256

REC004E_SCORE_OBS_PER_LENGTH: Final = 32
REC004E_INTERVENTION_PER_LENGTH: Final = REC004E_LENGTH_BALANCED_PER_LENGTH  # 256

# pi_n fixed positions per legal length -- a contract test target, recomputed
# from the live `mpid.mirror_halves_position_map` in Stage B3, never assumed.
REC004E_FIXED_POSITIONS: Final[dict[int, tuple[int, ...]]] = {
    6: (1, 4),
    7: (1,),
    8: (),
    9: (6,),
    10: (2, 7),
}

REC004E_J_IDS: Final[tuple[str, ...]] = ("J0", "J1", "J2", "J3", "J4", "J5", "J6")
# bias_scale (multiplies the real position-bias term) and s_other_scale
# (multiplies the existing pre-bias attention score) per J-condition.
# J4 zeroes S_other while KEEPING bias at scale 1 and the real padding mask.
REC004E_J_BIAS_SCALE: Final[dict[str, float]] = {
    "J0": 1.0, "J1": 0.0, "J2": 0.5, "J3": 2.0, "J4": 1.0, "J5": 1.0, "J6": 1.0,
}
REC004E_J_S_OTHER_SCALE: Final[dict[str, float]] = {
    "J0": 1.0, "J1": 1.0, "J2": 1.0, "J3": 1.0, "J4": 0.0, "J5": 1.0, "J6": 1.0,
}
REC004E_J_ALL_LENGTH_IDS: Final[tuple[str, ...]] = ("J0", "J1", "J2", "J3", "J4")
REC004E_J5_TARGET_LENGTH: Final = 10
REC004E_J6_TARGET_LENGTH: Final = 9
REC004E_J5_OVERRIDE_VALUE: Final = 9.0 / REC004E_LENGTH_REF
REC004E_J6_OVERRIDE_VALUE: Final = 10.0 / REC004E_LENGTH_REF

# Pre-registered comparison tolerances -- fixed here, before any observation
# is made, and never widened after seeing a failure (task doc section 3/A3).
REC004E_SOURCE_REPLAY_EM_TOL: Final = 1e-9
REC004E_SCORE_LOGIT_ABS_TOL: Final = 5e-3
REC004E_SCORE_ATTN_ABS_TOL: Final = 5e-3
REC004E_MANUAL_RECONSTRUCTION_ABS_TOL: Final = 5e-3

_REQUIRED_SOURCE_FILES: Final[tuple[str, ...]] = (
    "source_audit.json",
    "architecture_spec.json",
    "position_bias_protocol.json",
    "data_manifest.json",
    "shared_bias_initial_state.pt",
    "per_length_position_metrics.json",
    "paired_comparison.json",
    "initialization_stability.json",
    "bias_ablation.json",
    "learning_curve.jsonl",
    "lr_trace.jsonl",
    "candidate_decision.json",
    "summary.json",
)


def _assert_eval_only() -> None:
    """Dynamic guard (task doc section 7.2 item 10): every REC-004E forward
    path must run with gradients disabled. Raises rather than silently
    proceeding if a caller forgets to wrap in `torch.no_grad()`."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "REC004E_TRAINING_GUARD: gradient computation must stay disabled for every "
            "REC-004E diagnostic forward pass (new_optimizer_updates must remain 0)"
        )


@dataclass(frozen=True)
class MirrorPositionScoreResidualAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004e")
    seed: int = RECOVERY_PILOT_SEED


# =============================================================================
# Stage A1/A2 -- lock the exact source artifacts this audit reads, replay
# P/I01-I05's step=6000 predictions against REC-004D's own saved metrics.
# =============================================================================


def lock_source_manifest() -> dict[str, Any]:
    """A1: confirms the exact REC-004D run directory this task reads exists
    and hashes every file this audit depends on -- no `latest` search, no
    fallback to a different run/seed. Returns `status: SOURCE_ARTIFACT_
    UNAVAILABLE` (never fabricates a substitute) if anything required is
    missing."""
    if not REC004E_SOURCE_RUN_DIR.is_dir():
        return {
            "task_id": REC004E_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "missing": str(REC004E_SOURCE_RUN_DIR),
        }
    missing_files = [
        f for f in _REQUIRED_SOURCE_FILES if not (REC004E_SOURCE_RUN_DIR / f).is_file()
    ]
    if missing_files:
        return {
            "task_id": REC004E_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "missing": missing_files,
        }
    file_hashes = {
        f: mb.raw_file_sha256(REC004E_SOURCE_RUN_DIR / f) for f in _REQUIRED_SOURCE_FILES
    }

    checkpoint_availability: dict[str, dict[str, bool]] = {}
    for init_id in REC004E_INIT_IDS:
        for arm in (REC004E_ARM_U, REC004E_ARM_P):
            per_step: dict[str, bool] = {}
            for step in REC004E_CHECKPOINT_STEPS:
                ckpt = REC004E_SOURCE_RUN_DIR / init_id / arm / "checkpoints" / f"step{step}.pt"
                per_step[str(step)] = ckpt.is_file()
            checkpoint_availability[f"{init_id}:{arm}"] = per_step

    missing_terminal_p = [
        key
        for key, avail in checkpoint_availability.items()
        if key.endswith(f":{REC004E_ARM_P}") and not avail[str(REC004E_DECISIVE_STEP)]
    ]
    if missing_terminal_p:
        return {
            "task_id": REC004E_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "missing_terminal_p_checkpoints": missing_terminal_p,
        }

    architecture_spec = json.loads(
        (REC004E_SOURCE_RUN_DIR / "architecture_spec.json").read_text(encoding="utf-8")
    )
    live_length_ref = REC004E_LENGTH_REF
    live_hidden_dim = REC004E_BIAS_HIDDEN_DIM
    length_ref_matches = architecture_spec.get("length_ref") == live_length_ref
    if not length_ref_matches:
        return {
            "task_id": REC004E_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "reason": (
                f"architecture_spec.json length_ref={architecture_spec.get('length_ref')} "
                f"does not match the live CrossPositionLengthBiasPrimitiveConfig schema "
                f"default {live_length_ref} -- refusing to assume a stale/rebuilt schema"
            ),
        }

    return {
        "task_id": REC004E_TASK_ID,
        "status": "AVAILABLE",
        "source_run_dir": str(REC004E_SOURCE_RUN_DIR.resolve()),
        "file_hashes": file_hashes,
        "checkpoint_availability": checkpoint_availability,
        "checkpoint_steps": list(REC004E_CHECKPOINT_STEPS),
        "architecture_confirmation": {
            "length_ref_real_value": architecture_spec.get("length_ref"),
            "length_ref_schema_source": architecture_spec.get("length_ref_source"),
            "length_ref_matches_live_schema": length_ref_matches,
            "bias_hidden_dim_real_value": live_hidden_dim,
            "new_param_count": architecture_spec.get("new_param_count"),
            "architecture_signature_p": architecture_spec.get("architecture_signature_p"),
        },
    }


def _load_p_primitive(
    core: Any, init_id: str, step: int
) -> CrossPositionLengthBiasPrimitive | None:
    ckpt = REC004E_SOURCE_RUN_DIR / init_id / REC004E_ARM_P / "checkpoints" / f"step{step}.pt"
    if not ckpt.is_file():
        return None
    sd = mb.load_state_dict(ckpt)
    primitive = bias_repair._new_arm_primitive(core, REC004E_ARM_P)
    assert isinstance(primitive, CrossPositionLengthBiasPrimitive)
    primitive.to(core.device)
    primitive.load_state_dict({k: v.to(core.device) for k, v in sd.items()}, strict=True)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def run_source_replay(core: Any, eval_bank: Any, op_to_id: dict[str, int]) -> dict[str, Any]:
    """A2: replays each P/I0N's step=6000 predictions against REC-004D's own
    `per_length_position_metrics.json` sequence-EM (the strongest available
    ground truth: it was computed by REC-004D's own training run, from the
    same checkpoint this task loads). STOP (`SOURCE_REPLAY_MISMATCH`) on any
    unexplained difference rather than silently proceeding."""
    _assert_eval_only()
    recorded = json.loads(
        (REC004E_SOURCE_RUN_DIR / "per_length_position_metrics.json").read_text(encoding="utf-8")
    )
    pid = REC004E_TARGET_PHYSICAL_ID
    original_slot = eval_bank.get(pid)
    per_init: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for init_id in REC004E_INIT_IDS:
        primitive = _load_p_primitive(core, init_id, REC004E_DECISIVE_STEP)
        if primitive is None:
            per_init[init_id] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
            continue
        eval_bank.replace_primitive(pid, primitive)
        try:
            row = _evaluate_one_operation(
                core, eval_bank, op_to_id, REC004E_TARGET_OPERATION,
                seed=RECOVERY_PILOT_SEED, n_examples=REC004E_EXISTING_VALIDATION_EXAMPLES,
                split=REC004E_EXISTING_VALIDATION_SPLIT,
            )
        finally:
            eval_bank.replace_primitive(pid, original_slot)
        recorded_row = recorded.get(f"{init_id}:{REC004E_ARM_P}")
        recorded_em = (
            recorded_row["position_summary"]["sequence_exact_match"] if recorded_row else None
        )
        reproduced_em = row["correct_exact_match"]
        matches = (
            recorded_em is not None
            and abs(reproduced_em - recorded_em) < REC004E_SOURCE_REPLAY_EM_TOL
        )
        if not matches:
            mismatches.append(
                {
                    "init_id": init_id,
                    "reproduced_correct_exact_match": reproduced_em,
                    "recorded_correct_exact_match": recorded_em,
                }
            )
        per_init[init_id] = {
            "reproduced_correct_exact_match": reproduced_em,
            "recorded_correct_exact_match": recorded_em,
            "matches_within_tolerance": matches,
        }
    status = "VERIFIED" if not mismatches else "SOURCE_REPLAY_MISMATCH"
    return {
        "task_id": REC004E_TASK_ID,
        "split": REC004E_EXISTING_VALIDATION_SPLIT,
        "n_examples": REC004E_EXISTING_VALIDATION_EXAMPLES,
        "abs_tolerance": REC004E_SOURCE_REPLAY_EM_TOL,
        "per_init": per_init,
        "mismatches": mismatches,
        "status": status,
    }


# =============================================================================
# Stage A3 -- score observer: real forward reconstruction via (a) a parallel
# `need_weights=True` hook through the SAME cross_attn submodule/weights
# (used whenever only the *combined* post-softmax distribution is needed),
# and (b) a manual Q/K/V reconstruction (used only for J4, which needs the
# existing pre-softmax score term S_other zeroed independently of the bias
# term B -- something the public `attn_mask` API cannot express).
# =============================================================================


def _prepare_query_kv(
    primitive: CrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    device = content_features.device
    batch, lmax, _ = content_features.shape
    out_max = max(output_lengths)
    content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
    kv = primitive.content_in_proj(content_features) + primitive.content_position_embedding(
        content_position_ids
    )
    query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
    query = primitive.answer_query_embedding(query_ids)
    # MIRROR_HALVES is parameter-free (arg_encoder/arg_proj are None), so the
    # real forward's arg_token is always zero; asserted, never assumed.
    assert primitive.arg_encoder is None and primitive.arg_proj is None
    return query, kv, out_max, lmax, batch


def _pad_mask(content_lengths: list[int], lmax: int, device: torch.device) -> torch.Tensor:
    lengths_t = torch.tensor(content_lengths, device=device).view(-1, 1)
    p_idx = torch.arange(lmax, device=device).view(1, -1)
    return p_idx >= lengths_t  # [batch, lmax], True = padded/invalid key


def _position_bias_raw(
    primitive: CrossPositionLengthBiasPrimitive,
    content_lengths: list[int],
    out_max: int,
    lmax: int,
    device: torch.device,
    *,
    phi4_override: float | None = None,
    phi4_apply_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Exactly reproduces `CrossPositionLengthBiasPrimitive._position_bias`'s
    real feature construction (verified for bit-identity in
    `tests/test_mirror_position_score_residual_audit.py` when no override is
    given), returning every intermediate (`phi`, pre-activation `z`,
    post-ReLU `h`, and `b`) instead of only the final scalar -- needed for
    Stage B's grid/activation enumeration and for J5/J6's counterfactual
    length-feature substitution (task doc section 5/C2)."""
    batch = len(content_lengths)
    dtype = primitive.position_bias_hidden.weight.dtype
    c_lens = torch.tensor(content_lengths, device=device, dtype=dtype).view(batch, 1, 1)
    s_idx = torch.arange(out_max, device=device, dtype=dtype).view(1, out_max, 1)
    p_idx = torch.arange(lmax, device=device, dtype=dtype).view(1, 1, lmax)
    d_denom = torch.clamp(c_lens - 1.0, min=1.0)
    shape = (batch, out_max, lmax)
    feat0 = s_idx.expand(shape) / d_denom
    feat1 = p_idx.expand(shape) / d_denom
    feat2 = (p_idx - s_idx).expand(shape) / d_denom
    feat3_real = (c_lens / float(primitive.length_ref)).expand(shape)
    if phi4_override is None:
        feat3 = feat3_real
    else:
        assert phi4_apply_mask is not None
        apply = phi4_apply_mask.view(batch, 1, 1).expand(shape)
        override_t = torch.full(shape, float(phi4_override), device=device, dtype=dtype)
        feat3 = torch.where(apply, override_t, feat3_real)
    phi = torch.stack([feat0, feat1, feat2, feat3], dim=-1)
    z = primitive.position_bias_hidden(phi)
    h = F.relu(z)
    b = primitive.position_bias_out(h).squeeze(-1)
    return {"phi": phi, "z": z, "h": h, "b": b}


def _manual_attention(
    primitive: CrossPositionLengthBiasPrimitive,
    query: torch.Tensor,
    kv: torch.Tensor,
    additive_term: torch.Tensor,
    *,
    s_other_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual Q/K/V reconstruction of `primitive.cross_attn`'s real
    computation (standard scaled-dot-product-attention formula, no fused
    kernel), used ONLY where the existing score term S_other must be scaled
    independently of the additive bias/mask term (J4). Returns
    `(attn_output, attn_probs, s_other)`; parity against the real module is
    verified in `run_score_observer_parity` before any J4 result is
    trusted."""
    _assert_eval_only()
    mha = primitive.cross_attn
    embed_dim = primitive.d_operator
    n_head = primitive.n_head
    head_dim = embed_dim // n_head
    wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
    if mha.in_proj_bias is not None:
        bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
    else:
        bq = bk = bv = None
    q = F.linear(query, wq, bq)
    k = F.linear(kv, wk, bk)
    v = F.linear(kv, wv, bv)
    batch, out_max, _ = q.shape
    lmax = k.shape[1]
    q = q.view(batch, out_max, n_head, head_dim).transpose(1, 2)
    k = k.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    v = v.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)
    s_other = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, n_head, out_max, lmax]
    scores = s_other_scale * s_other + additive_term.unsqueeze(1)
    attn_probs = F.softmax(scores, dim=-1)
    attn_out_heads = torch.matmul(attn_probs, v)  # [batch, n_head, out_max, head_dim]
    attn_out = attn_out_heads.transpose(1, 2).reshape(batch, out_max, embed_dim)
    attn_out = mha.out_proj(attn_out)
    return attn_out, attn_probs, s_other


def _post_attention(
    primitive: CrossPositionLengthBiasPrimitive, query: torch.Tensor, attn_out: torch.Tensor
) -> torch.Tensor:
    hidden = primitive.attn_norm(query + attn_out)
    hidden = primitive.ffn_norm(hidden + primitive.ffn(hidden))
    return primitive.readout(hidden)


def run_intervention_forward(
    primitive: CrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
    j_id: str,
) -> dict[str, Any]:
    """Applies exactly one of the task doc's fixed J0-J6 conditions
    (section 5/C2) and returns the resulting logits plus the intermediate
    bias/score tensors used for decomposition. J0-J3/J5/J6 route through the
    REAL `primitive.cross_attn` module (a `need_weights=True` hook, same
    weights/scale/mask semantics as production, only the injected additive
    mask differs) -- J4 alone needs the manual reconstruction (`
    _manual_attention`) because zeroing S_other independently of the bias/
    padding term cannot be expressed via the public `attn_mask` argument."""
    _assert_eval_only()
    assert j_id in REC004E_J_IDS
    query, kv, out_max, lmax, batch = _prepare_query_kv(
        primitive, content_features, content_lengths, output_lengths
    )
    device = content_features.device
    pad_mask = _pad_mask(content_lengths, lmax, device)  # [batch, lmax]

    if j_id == "J5":
        apply_mask = torch.tensor(
            [cl == REC004E_J5_TARGET_LENGTH for cl in content_lengths], device=device
        )
        bias_pack = _position_bias_raw(
            primitive, content_lengths, out_max, lmax, device,
            phi4_override=REC004E_J5_OVERRIDE_VALUE, phi4_apply_mask=apply_mask,
        )
    elif j_id == "J6":
        apply_mask = torch.tensor(
            [cl == REC004E_J6_TARGET_LENGTH for cl in content_lengths], device=device
        )
        bias_pack = _position_bias_raw(
            primitive, content_lengths, out_max, lmax, device,
            phi4_override=REC004E_J6_OVERRIDE_VALUE, phi4_apply_mask=apply_mask,
        )
    else:
        bias_pack = _position_bias_raw(primitive, content_lengths, out_max, lmax, device)

    bias_scale = REC004E_J_BIAS_SCALE[j_id]
    b_scaled = bias_pack["b"] * bias_scale
    neg_inf = torch.finfo(b_scaled.dtype).min
    combined = torch.where(pad_mask.unsqueeze(1).expand(batch, out_max, lmax), neg_inf, b_scaled)

    s_other_scale = REC004E_J_S_OTHER_SCALE[j_id]
    if j_id == "J4":
        attn_out, attn_probs, s_other = _manual_attention(
            primitive, query, kv, combined, s_other_scale=s_other_scale
        )
    else:
        n_head = primitive.n_head
        attn_mask = (
            combined.unsqueeze(1)
            .expand(batch, n_head, out_max, lmax)
            .reshape(batch * n_head, out_max, lmax)
        )
        attn_out, attn_probs = primitive.cross_attn(
            query, kv, kv, attn_mask=attn_mask, need_weights=True, average_attn_weights=False
        )
        s_other = None

    logits = _post_attention(primitive, query, attn_out)
    return {
        "logits": logits,
        "attn_probs": attn_probs,
        "s_other": s_other,
        "b_scaled": b_scaled,
        "bias_pack": bias_pack,
        "pad_mask": pad_mask,
    }


def _predict_from_logits(logits: torch.Tensor, output_lengths: list[int]) -> list[list[int]]:
    preds = logits.argmax(dim=-1)
    return [preds[row, :n].tolist() for row, n in enumerate(output_lengths)]


def run_score_observer_parity(core: Any) -> dict[str, Any]:
    """A3: confirms the score-observer reconstruction reproduces the real
    forward's discrete predictions BEFORE any score decomposition or
    intervention result is trusted, on three independent checks: (1) a CPU
    tiny fixture with a random nonzero bias, (2) the real, trained P/I01
    step=6000 checkpoint on the production device against the score-
    observation subset, and (3) J1's EM against REC-004D's own saved
    `bias_ablation.json` `em_zero_bias` (an independent ground truth, since
    that number was produced by the real `zeroed_position_bias()` context
    manager through a real `need_weights=False` forward, not by this
    task's own reconstruction)."""
    _assert_eval_only()
    results: dict[str, Any] = {}

    # (1) CPU tiny fixture.
    torch.manual_seed(12345)
    kwargs = bias_repair._base_primitive_kwargs(core)
    cpu_primitive = CrossPositionLengthBiasPrimitive(
        REC004E_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004E_BIAS_HIDDEN_DIM, length_ref=REC004E_LENGTH_REF
        ),
    )
    with torch.no_grad():
        cpu_primitive.position_bias_out.weight.normal_(mean=0.0, std=0.2)
    cpu_primitive.eval()
    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    torch.manual_seed(999)
    content = torch.randn(len(content_lengths), lmax, kwargs["d_model"])
    with torch.no_grad():
        real_logits = cpu_primitive(content, content_lengths, output_lengths, None)
        recon = run_intervention_forward(
            cpu_primitive, content, content_lengths, output_lengths, "J0"
        )
    cpu_logit_diff = (real_logits - recon["logits"]).abs().max().item()
    cpu_pred_match = _predict_from_logits(real_logits, output_lengths) == _predict_from_logits(
        recon["logits"], output_lengths
    )
    results["cpu_fixture"] = {
        "max_abs_logit_diff": cpu_logit_diff,
        "predictions_match": cpu_pred_match,
        "tolerance": REC004E_SCORE_LOGIT_ABS_TOL,
        "passed": cpu_logit_diff <= REC004E_SCORE_LOGIT_ABS_TOL and cpu_pred_match,
    }

    # (1b) manual reconstruction (used only for J4) validated at s_other_scale=1
    # against the same hook-path J0 output, on the same CPU fixture.
    with torch.no_grad():
        query, kv, out_max, lmax2, batch = _prepare_query_kv(
            cpu_primitive, content, content_lengths, output_lengths
        )
        pad = _pad_mask(content_lengths, lmax2, content.device)
        bias_pack = _position_bias_raw(
            cpu_primitive, content_lengths, out_max, lmax2, content.device
        )
        combined = torch.where(
            pad.unsqueeze(1).expand(batch, out_max, lmax2),
            torch.finfo(bias_pack["b"].dtype).min,
            bias_pack["b"],
        )
        manual_out, manual_probs, _ = _manual_attention(
            cpu_primitive, query, kv, combined, s_other_scale=1.0
        )
        manual_logits = _post_attention(cpu_primitive, query, manual_out)
    manual_diff = (manual_logits - real_logits).abs().max().item()
    manual_pred_match = _predict_from_logits(manual_logits, output_lengths) == _predict_from_logits(
        real_logits, output_lengths
    )
    results["manual_reconstruction_fixture"] = {
        "max_abs_logit_diff": manual_diff,
        "predictions_match": manual_pred_match,
        "tolerance": REC004E_MANUAL_RECONSTRUCTION_ABS_TOL,
        "passed": manual_diff <= REC004E_MANUAL_RECONSTRUCTION_ABS_TOL and manual_pred_match,
    }

    # (2) production device, real trained checkpoint, score-observation subset.
    primitive = _load_p_primitive(core, "I01", REC004E_DECISIVE_STEP)
    if primitive is None:
        results["production_device"] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
    else:
        length_balanced = mpid._generate_length_balanced_diagnostic()
        by_length: dict[int, list[Any]] = {}
        for ex in length_balanced:
            by_length.setdefault(len(ex.input_tokens), []).append(ex)
        per_length_results = {}
        max_diff_overall = 0.0
        all_match = True
        for n in REC004E_LEGAL_LENGTHS:
            examples = by_length[n][:REC004E_SCORE_OBS_PER_LENGTH]
            content_lengths_n = [len(e.input_tokens) for e in examples]
            op = get_operation(REC004E_TARGET_OPERATION)
            output_lengths_n = [op.output_length(cl) for cl in content_lengths_n]
            batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
            with torch.no_grad():
                h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths_n), :]
                real_logits_n = primitive(h, content_lengths_n, output_lengths_n, None)
                recon_n = run_intervention_forward(
                    primitive, h, content_lengths_n, output_lengths_n, "J0"
                )
            diff = (real_logits_n - recon_n["logits"]).abs().max().item()
            match = _predict_from_logits(real_logits_n, output_lengths_n) == _predict_from_logits(
                recon_n["logits"], output_lengths_n
            )
            per_length_results[str(n)] = {
                "n_examples": len(examples), "max_abs_logit_diff": diff, "predictions_match": match,
            }
            max_diff_overall = max(max_diff_overall, diff)
            all_match = all_match and match
        results["production_device"] = {
            "device": str(core.device),
            "per_length": per_length_results,
            "max_abs_logit_diff": max_diff_overall,
            "all_predictions_match": all_match,
            "tolerance": REC004E_SCORE_LOGIT_ABS_TOL,
            "passed": max_diff_overall <= REC004E_SCORE_LOGIT_ABS_TOL and all_match,
        }

    # (3) J1 EM vs REC-004D's own saved bias_ablation.json em_zero_bias.
    bias_ablation_recorded = json.loads(
        (REC004E_SOURCE_RUN_DIR / "bias_ablation.json").read_text(encoding="utf-8")
    )
    j1_results: dict[str, Any] = {}
    j1_all_match = True
    for init_id in REC004E_INIT_IDS:
        primitive_i = _load_p_primitive(core, init_id, REC004E_DECISIVE_STEP)
        if primitive_i is None:
            j1_results[init_id] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
            j1_all_match = False
            continue
        examples = _generate_parameter_free_examples(
            RECOVERY_PILOT_SEED, REC004E_EXISTING_VALIDATION_EXAMPLES,
            operation=REC004E_TARGET_OPERATION, split=REC004E_EXISTING_VALIDATION_SPLIT,
            vocab_size=REC004E_VOCAB_SIZE, sequence_length_range=REC004E_SEQUENCE_LENGTH_RANGE,
        )
        content_lengths_all = [len(e.input_tokens) for e in examples]
        output_lengths_all = [
            get_operation(REC004E_TARGET_OPERATION).output_length(n) for n in content_lengths_all
        ]
        batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
        with torch.no_grad():
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths_all), :]
            recon = run_intervention_forward(
                primitive_i, h, content_lengths_all, output_lengths_all, "J1"
            )
        preds = _predict_from_logits(recon["logits"], output_lengths_all)
        correct = [
            tuple(p) == tuple(e.target_tokens[:n])
            for p, e, n in zip(preds, examples, output_lengths_all, strict=True)
        ]
        em = sum(correct) / len(correct)
        recorded_em = bias_ablation_recorded.get(init_id, {}).get("em_zero_bias")
        match = recorded_em is not None and abs(em - recorded_em) < 1e-9
        j1_all_match = j1_all_match and match
        j1_results[init_id] = {
            "reconstructed_j1_em": em,
            "recorded_em_zero_bias": recorded_em,
            "matches_exactly": match,
        }
    results["j1_vs_recorded_bias_ablation"] = {
        "per_init": j1_results, "all_match": j1_all_match,
        "note": (
            "J1 (bias_scale=0) is architecturally identical to zeroing "
            "position_bias_out.weight (both force b=0 everywhere); recorded values used "
            "primitive.forward() with need_weights=False, this task's J1 uses "
            "need_weights=True through the same module -- an exact match confirms "
            "need_weights=True does not change the discrete outcome for this model."
        ),
    }

    all_checks_passed = (
        results["cpu_fixture"]["passed"]
        and results["manual_reconstruction_fixture"]["passed"]
        and results.get("production_device", {}).get("passed", False)
        and j1_all_match
    )
    status = "VERIFIED" if all_checks_passed else "SCORE_OBSERVATION_UNAVAILABLE"
    return {
        "task_id": REC004E_TASK_ID,
        "checks": results,
        "status": status,
        "scope_note": (
            "VERIFIED licenses S_other/B decomposition and the J0-J6 intervention matrix "
            "in Stage C; SCORE_OBSERVATION_UNAVAILABLE restricts this run to the grid "
            "(Stage B), raw output-level intervention results, and curve diagnostics only."
        ),
    }


# =============================================================================
# Stage B -- enumerate the real position bias at every legal (i, j, n).
# =============================================================================


def _fixed_position_contract() -> dict[str, Any]:
    """B3: recomputes fixed/moved from the live `mpid.mirror_halves_
    position_map`, asserted against `REC004E_FIXED_POSITIONS` -- a contract
    test, not a new discovery (REC-004D already established this table)."""
    per_length: dict[str, Any] = {}
    all_match = True
    for n in REC004E_LEGAL_LENGTHS:
        pi = mpid.mirror_halves_position_map(n)
        fixed = tuple(i for i in range(n) if pi[i] == i)
        moved = tuple(i for i in range(n) if pi[i] != i)
        expected = REC004E_FIXED_POSITIONS[n]
        matches = fixed == expected
        all_match = all_match and matches
        per_length[str(n)] = {
            "pi_n": list(pi), "fixed_positions": list(fixed), "moved_positions": list(moved),
            "expected_fixed_positions": list(expected), "matches_contract": matches,
        }
    return {"per_length": per_length, "all_match": all_match}


def run_position_grid_enumeration(core: Any) -> dict[str, Any]:
    """B1/B2: for every P/I0N x checkpoint-step combination, computes the
    full (i, j) grid of `phi`, pre-activation `z`, post-ReLU `h`, and the
    real scalar bias `b` for every legal length n=6..10 (330 pairs; up to
    5 x 13 = 65 grids, 21450 scalars), plus the row-centered bias and a
    hidden-unit activation summary. Arrays are the source of truth
    (`position_grid.npz`); the returned dict holds only the index/summary
    that accompanies them."""
    _assert_eval_only()
    n_init = len(REC004E_INIT_IDS)
    n_steps = len(REC004E_CHECKPOINT_STEPS)
    hidden_dim = REC004E_BIAS_HIDDEN_DIM

    arrays: dict[str, np.ndarray] = {}
    for n in REC004E_LEGAL_LENGTHS:
        arrays[f"b_n{n}"] = np.full((n_init, n_steps, n, n), np.nan, dtype=np.float32)
        arrays[f"z_n{n}"] = np.full((n_init, n_steps, n, n, hidden_dim), np.nan, dtype=np.float32)
        arrays[f"h_n{n}"] = np.full((n_init, n_steps, n, n, hidden_dim), np.nan, dtype=np.float32)
        arrays[f"finite_n{n}"] = np.zeros((n_init, n_steps, n, n), dtype=bool)
        arrays[f"row_centered_n{n}"] = np.full((n_init, n_steps, n, n), np.nan, dtype=np.float32)
    w2_array = np.full((n_init, n_steps, hidden_dim), np.nan, dtype=np.float32)
    available = np.zeros((n_init, n_steps), dtype=bool)

    activation_rows: list[dict[str, Any]] = []
    row_centered_rows: list[dict[str, Any]] = []
    row_affine_hits = 0
    row_affine_total = 0

    for init_idx, init_id in enumerate(REC004E_INIT_IDS):
        for step_idx, step in enumerate(REC004E_CHECKPOINT_STEPS):
            primitive = _load_p_primitive(core, init_id, step)
            if primitive is None:
                continue
            available[init_idx, step_idx] = True
            device = core.device
            with torch.no_grad():
                w2_array[init_idx, step_idx] = (
                    primitive.position_bias_out.weight.detach().cpu().numpy().reshape(-1)
                )
                for n in REC004E_LEGAL_LENGTHS:
                    pack = _position_bias_raw(primitive, [n], n, n, device)
                    b = pack["b"][0].detach().cpu().numpy()
                    z = pack["z"][0].detach().cpu().numpy()
                    h = pack["h"][0].detach().cpu().numpy()
                    finite = np.isfinite(b)
                    arrays[f"b_n{n}"][init_idx, step_idx] = b
                    arrays[f"z_n{n}"][init_idx, step_idx] = z
                    arrays[f"h_n{n}"][init_idx, step_idx] = h
                    arrays[f"finite_n{n}"][init_idx, step_idx] = finite
                    row_mean = b.mean(axis=1, keepdims=True)
                    row_centered = b - row_mean
                    arrays[f"row_centered_n{n}"][init_idx, step_idx] = row_centered

                    active = h > 0.0
                    activation_rate = float(active.mean())
                    never_active = int(np.sum(~active.reshape(-1, hidden_dim).any(axis=0)))
                    always_active = int(np.sum(active.reshape(-1, hidden_dim).all(axis=0)))
                    activation_rows.append(
                        {
                            "init_id": init_id, "step": step, "n": n,
                            "activation_rate": activation_rate,
                            "never_active_units": never_active,
                            "always_active_units": always_active,
                            "partial_units": hidden_dim - never_active - always_active,
                        }
                    )
                    row_centered_rows.append(
                        {
                            "init_id": init_id, "step": step, "n": n,
                            "raw_b_range": float(b.max() - b.min()),
                            "row_centered_std_mean": float(row_centered.std(axis=1).mean()),
                            "row_centered_range_mean": float(
                                (row_centered.max(axis=1) - row_centered.min(axis=1)).mean()
                            ),
                        }
                    )
                    for i in range(n):
                        row_affine_total += 1
                        row_pattern = active[i]
                        if n > 1 and np.all(row_pattern == row_pattern[0]):
                            row_affine_hits += 1

    index = {
        "task_id": REC004E_TASK_ID,
        "init_ids": list(REC004E_INIT_IDS),
        "checkpoint_steps": list(REC004E_CHECKPOINT_STEPS),
        "legal_lengths": list(REC004E_LEGAL_LENGTHS),
        "bias_hidden_dim": hidden_dim,
        "array_layout": (
            "dim0=init index into init_ids, dim1=checkpoint index into checkpoint_steps, "
            "remaining dims=[query_position_i, key_position_j(, hidden_unit)] using 0-based "
            "real content positions (no special-token offset -- see source notes); "
            "b_n{N}/z_n{N}/h_n{N}/row_centered_n{N} shaped [5,13,N,N(,32)], "
            "finite_n{N} bool [5,13,N,N], w2 shaped [5,13,32], available bool [5,13]"
        ),
        "n_grids_available": int(available.sum()),
        "n_grids_expected": n_init * n_steps,
        "n_position_pairs_per_grid": {str(n): n * n for n in REC004E_LEGAL_LENGTHS},
        "total_position_pairs_per_grid": sum(n * n for n in REC004E_LEGAL_LENGTHS),
        "total_scalars": int(available.sum()) * sum(n * n for n in REC004E_LEGAL_LENGTHS),
        "fixed_position_contract": _fixed_position_contract(),
        "row_affine_segment": {
            "rows_with_single_activation_pattern": row_affine_hits,
            "rows_checked": row_affine_total,
            "fraction": (row_affine_hits / row_affine_total) if row_affine_total else None,
            "caveat": (
                "ROW_AFFINE_SEGMENT_OBSERVED describes the bias term alone; the existing "
                "score term is added before softmax, so this does not by itself show the "
                "combined operator cannot discriminate positions in that row."
            ),
        },
    }
    return {
        "index": index, "arrays": arrays,
        "activation_rows": activation_rows, "row_centered_rows": row_centered_rows,
        "available": available,
    }


def _summarize_activation_rows(activation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    at_decisive = [r for r in activation_rows if r["step"] == REC004E_DECISIVE_STEP]
    by_init: dict[str, Any] = {}
    for init_id in REC004E_INIT_IDS:
        rows = [r for r in at_decisive if r["init_id"] == init_id]
        if not rows:
            by_init[init_id] = {"status": "UNAVAILABLE"}
            continue
        by_init[init_id] = {
            "per_length": {
                str(r["n"]): {
                    "activation_rate": r["activation_rate"],
                    "never_active_units": r["never_active_units"],
                    "always_active_units": r["always_active_units"],
                    "partial_units": r["partial_units"],
                }
                for r in rows
            }
        }
    return {
        "task_id": REC004E_TASK_ID,
        "decisive_step": REC004E_DECISIVE_STEP,
        "per_init_at_decisive_step": by_init,
        "all_steps_raw_rows": activation_rows,
        "caveat": (
            "A unit inactive at every checked (i,j,n) is descriptive only -- it is not, by "
            "itself, evidence of overall network capacity loss (task doc section 4/B2)."
        ),
    }


def _summarize_row_centered_rows(row_centered_rows: list[dict[str, Any]]) -> dict[str, Any]:
    at_decisive = [r for r in row_centered_rows if r["step"] == REC004E_DECISIVE_STEP]
    by_init: dict[str, Any] = {}
    for init_id in REC004E_INIT_IDS:
        rows = [r for r in at_decisive if r["init_id"] == init_id]
        if not rows:
            by_init[init_id] = {"status": "UNAVAILABLE"}
            continue
        by_init[init_id] = {
            "per_length": {
                str(r["n"]): {
                    "raw_b_range": r["raw_b_range"],
                    "row_centered_std_mean": r["row_centered_std_mean"],
                    "row_centered_range_mean": r["row_centered_range_mean"],
                }
                for r in rows
            }
        }
    return {
        "task_id": REC004E_TASK_ID,
        "decisive_step": REC004E_DECISIVE_STEP,
        "per_init_at_decisive_step": by_init,
        "all_steps_raw_rows": row_centered_rows,
        "note": (
            "Row-centering is descriptive (softmax is invariant to a constant added across "
            "an entire row of VALID keys); it is never fed back into any forward pass."
        ),
    }


# =============================================================================
# Stage C1 -- score decomposition on the observation subset (32 examples per
# length, 160 total, per model).
# =============================================================================


def _length_balanced_by_length() -> dict[int, list[Any]]:
    examples = mpid._generate_length_balanced_diagnostic()
    by_length: dict[int, list[Any]] = {}
    for ex in examples:
        by_length.setdefault(len(ex.input_tokens), []).append(ex)
    return by_length


REC004E_N_HEAD: Final = 4  # fixed by _base_primitive_kwargs; asserted at runtime, never assumed


def run_score_decomposition(core: Any) -> dict[str, Any]:
    """C1: for each P/I0N's step=6000 checkpoint, decomposes the real J0
    attention score into S_other (existing, pre-bias) and B (new bias),
    on the pre-registered `score_observation_subset` (first 32 examples per
    length from the length-balanced suite, fixed before any output is
    inspected). Records per-head and head-averaged row-centered std/range
    for each component, their row-wise correlation, and the correct-
    position rank/margin (a descriptive metric only -- `pi_n` is used here
    strictly for scoring, never as a forward input). Raw per-example `b`/
    `s_other` arrays are returned alongside for `score_decomposition.npz`
    (the primary source of truth); the returned `per_init` dict is this
    file's descriptive companion (`score_decomposition_index.json`)."""
    _assert_eval_only()
    by_length = _length_balanced_by_length()
    per_init: dict[str, Any] = {}
    n_obs = REC004E_SCORE_OBS_PER_LENGTH
    arrays: dict[str, np.ndarray] = {}
    for n in REC004E_LEGAL_LENGTHS:
        arrays[f"b_n{n}"] = np.full((len(REC004E_INIT_IDS), n_obs, n, n), np.nan, dtype=np.float32)
        arrays[f"s_other_n{n}"] = np.full(
            (len(REC004E_INIT_IDS), REC004E_N_HEAD, n_obs, n, n), np.nan, dtype=np.float32
        )

    for init_idx, init_id in enumerate(REC004E_INIT_IDS):
        primitive = _load_p_primitive(core, init_id, REC004E_DECISIVE_STEP)
        if primitive is None:
            per_init[init_id] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
            continue
        assert primitive.n_head == REC004E_N_HEAD
        per_length: dict[str, Any] = {}
        op = get_operation(REC004E_TARGET_OPERATION)
        for n in REC004E_LEGAL_LENGTHS:
            examples = by_length[n][:REC004E_SCORE_OBS_PER_LENGTH]
            content_lengths = [len(e.input_tokens) for e in examples]
            output_lengths = [op.output_length(cl) for cl in content_lengths]
            batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
            with torch.no_grad():
                h_content = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
                query, kv, out_max, lmax, batch = _prepare_query_kv(
                    primitive, h_content, content_lengths, output_lengths
                )
                pad = _pad_mask(content_lengths, lmax, core.device)
                bias_pack = _position_bias_raw(
                    primitive, content_lengths, out_max, lmax, core.device
                )
                combined = torch.where(
                    pad.unsqueeze(1).expand(batch, out_max, lmax),
                    torch.finfo(bias_pack["b"].dtype).min,
                    bias_pack["b"],
                )
                _, attn_probs_manual, s_other = _manual_attention(
                    primitive, query, kv, combined, s_other_scale=1.0
                )
                recon = run_intervention_forward(
                    primitive, h_content, content_lengths, output_lengths, "J0"
                )
            b = bias_pack["b"].cpu().numpy()  # [batch, out_max, lmax]
            s_other_np = s_other.cpu().numpy()  # [batch, n_head, out_max, lmax]
            valid = (~pad.cpu().numpy()).astype(bool)  # [batch, lmax]
            n_examples_here = min(len(examples), n_obs)
            arrays[f"b_n{n}"][init_idx, :n_examples_here] = b[:n_examples_here]
            arrays[f"s_other_n{n}"][init_idx, :, :n_examples_here] = np.transpose(
                s_other_np[:n_examples_here], (1, 0, 2, 3)
            )

            def _row_centered_stats(
                arr: Any, valid_mask: Any, out_lens: list[int] = output_lengths
            ) -> dict[str, float | None]:
                stds, ranges = [], []
                for row_i in range(arr.shape[0]):
                    for i in range(out_lens[row_i]):
                        vals = arr[row_i, i][valid_mask[row_i]]
                        if vals.size < 2:
                            continue
                        centered = vals - vals.mean()
                        stds.append(float(centered.std()))
                        ranges.append(float(centered.max() - centered.min()))
                return {
                    "std_mean": float(np.mean(stds)) if stds else None,
                    "range_mean": float(np.mean(ranges)) if ranges else None,
                }

            b_stats = _row_centered_stats(b, valid)
            s_other_mean_heads = s_other_np.mean(axis=1)
            s_other_stats_avg = _row_centered_stats(s_other_mean_heads, valid)
            s_other_stats_per_head = [
                _row_centered_stats(s_other_np[:, head_idx], valid)
                for head_idx in range(s_other_np.shape[1])
            ]

            correlations = []
            for row_i in range(b.shape[0]):
                for i in range(output_lengths[row_i]):
                    v = valid[row_i]
                    b_row = b[row_i, i][v]
                    s_row = s_other_mean_heads[row_i, i][v]
                    if b_row.size < 2 or float(b_row.std()) == 0.0 or float(s_row.std()) == 0.0:
                        continue
                    correlations.append(float(np.corrcoef(b_row, s_row)[0, 1]))

            ranks, margins = [], []
            pi = mpid.mirror_halves_position_map(n)
            # head-averaged, [batch, out_max, lmax]
            probs = attn_probs_manual.mean(dim=1).cpu().numpy()
            for row_i in range(len(examples)):
                for i in range(n):
                    target_j = pi[i]
                    row_probs = probs[row_i, i, :n]
                    order = np.argsort(-row_probs)
                    rank = int(np.where(order == target_j)[0][0]) + 1
                    ranks.append(rank)
                    best_other = max(
                        (row_probs[j] for j in range(n) if j != target_j), default=float("nan")
                    )
                    margins.append(float(row_probs[target_j] - best_other))

            preds = _predict_from_logits(recon["logits"], output_lengths)
            correct = [
                tuple(p) == tuple(e.target_tokens[:no])
                for p, e, no in zip(preds, examples, output_lengths, strict=True)
            ]
            per_length[str(n)] = {
                "n_examples": len(examples),
                "b_row_centered": b_stats,
                "s_other_row_centered_head_averaged": s_other_stats_avg,
                "s_other_row_centered_per_head": s_other_stats_per_head,
                "b_vs_s_other_row_correlation_mean": (
                    float(np.mean(correlations)) if correlations else None
                ),
                "correct_position_rank_mean": float(np.mean(ranks)) if ranks else None,
                "correct_position_margin_mean": float(np.mean(margins)) if margins else None,
                "sequence_exact_match": (sum(correct) / len(correct)) if correct else None,
            }
        per_init[init_id] = {"per_length": per_length}

    index = {
        "task_id": REC004E_TASK_ID,
        "n_examples_per_length": REC004E_SCORE_OBS_PER_LENGTH,
        "n_lengths": len(REC004E_LEGAL_LENGTHS),
        "total_examples_per_model": REC004E_SCORE_OBS_PER_LENGTH * len(REC004E_LEGAL_LENGTHS),
        "init_ids": list(REC004E_INIT_IDS),
        "n_head": REC004E_N_HEAD,
        "array_layout": (
            "b_n{N} shaped [5 inits, 32 examples, N, N]; s_other_n{N} shaped "
            "[5 inits, n_head, 32 examples, N, N] -- both indexed by 0-based real content "
            "position (query i, key j), no padding within a length group (every example in "
            "a group shares the same real length N)."
        ),
        "per_init": per_init,
        "caveat": (
            "S_other is the existing (pre-bias) scaled attention score, reconstructed via a "
            "manual Q/K/V pass validated in run_score_observer_parity; it still reflects the "
            "existing position embeddings, so it is not a 'pure content' score."
        ),
    }
    return {"index": index, "per_init": per_init, "arrays": arrays}


# =============================================================================
# Stage C2/C3 -- fixed intervention matrix J0-J6 on the intervention suite.
# =============================================================================


def run_interventions(core: Any) -> dict[str, Any]:
    """C2/C3: applies each J-condition to each P/I0N's step=6000 checkpoint
    on the full intervention suite (the length-balanced diagnostic's 256
    examples per length; J0-J4 over all 5 lengths, J5 restricted to n=10,
    J6 restricted to n=9, per the task doc's own table). Never selects,
    adopts, or trains on any result -- purely descriptive, paired against
    J0 on the SAME examples."""
    _assert_eval_only()
    by_length = _length_balanced_by_length()
    rows: list[dict[str, Any]] = []
    per_init_summary: dict[str, Any] = {}

    for init_id in REC004E_INIT_IDS:
        primitive = _load_p_primitive(core, init_id, REC004E_DECISIVE_STEP)
        if primitive is None:
            per_init_summary[init_id] = {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
            continue
        j0_correct_by_length: dict[int, list[bool]] = {}
        summary_by_j: dict[str, Any] = {}
        for j_id in REC004E_J_IDS:
            lengths_for_j = (
                (REC004E_J5_TARGET_LENGTH,) if j_id == "J5"
                else (REC004E_J6_TARGET_LENGTH,) if j_id == "J6"
                else REC004E_LEGAL_LENGTHS
            )
            per_length_stats: dict[str, Any] = {}
            j_correct_total, j_n_total = 0, 0
            j0_only_total, j_only_total, both_wrong_total = 0, 0, 0
            op = get_operation(REC004E_TARGET_OPERATION)
            for n in lengths_for_j:
                examples = by_length[n]
                content_lengths = [len(e.input_tokens) for e in examples]
                output_lengths = [op.output_length(cl) for cl in content_lengths]
                batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
                with torch.no_grad():
                    h_content = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
                    out = run_intervention_forward(
                        primitive, h_content, content_lengths, output_lengths, j_id
                    )
                    labels = _labels_for_examples(
                        examples, output_lengths, max(output_lengths), core.device
                    )
                    loss = F.cross_entropy(
                        out["logits"].reshape(-1, out["logits"].size(-1)),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    ).item()
                preds = _predict_from_logits(out["logits"], output_lengths)
                correct = [
                    tuple(p) == tuple(e.target_tokens[:no])
                    for p, e, no in zip(preds, examples, output_lengths, strict=True)
                ]
                token_correct = sum(
                    int(p[k] == e.target_tokens[k])
                    for p, e, no in zip(preds, examples, output_lengths, strict=True)
                    for k in range(no)
                )
                token_total = sum(output_lengths)
                if j_id == "J0":
                    j0_correct_by_length[n] = correct
                    paired_delta = 0.0
                    j0_only = j_only = both_wrong = 0
                else:
                    j0_correct = j0_correct_by_length.get(n)
                    if j0_correct is None:
                        j0_correct = [False] * len(correct)
                    pairs = list(zip(j0_correct, correct, strict=True))
                    paired_delta = (sum(correct) - sum(j0_correct)) / len(correct)
                    j0_only = sum(1 for a, b in pairs if a and not b)
                    j_only = sum(1 for a, b in pairs if b and not a)
                    both_wrong = sum(1 for a, b in pairs if not a and not b)
                em = sum(correct) / len(correct)
                per_length_stats[str(n)] = {
                    "n": len(correct),
                    "sequence_exact_match": em,
                    "token_accuracy": token_correct / token_total if token_total else None,
                    "loss": loss,
                    "paired_delta_vs_j0": paired_delta,
                    "j0_only_correct": j0_only,
                    "j_only_correct": j_only,
                    "both_wrong": both_wrong,
                }
                j_correct_total += sum(correct)
                j_n_total += len(correct)
                j0_only_total += j0_only
                j_only_total += j_only
                both_wrong_total += both_wrong
                rows.append(
                    {
                        "init_id": init_id, "j_id": j_id, "length": n,
                        **per_length_stats[str(n)],
                    }
                )
            summary_by_j[j_id] = {
                "per_length": per_length_stats,
                "overall_sequence_exact_match": j_correct_total / j_n_total if j_n_total else None,
                "overall_n": j_n_total,
                "j0_only_correct_total": j0_only_total,
                "j_only_correct_total": j_only_total,
                "both_wrong_total": both_wrong_total,
            }
        per_init_summary[init_id] = summary_by_j

    return {"rows": rows, "per_init_summary": per_init_summary}


def run_intervention_nonmutation_audit(core: Any) -> dict[str, Any]:
    """C2's own non-mutation guard: confirms that running the FULL J0-J6
    intervention matrix against one loaded P checkpoint leaves its
    parameters/buffers byte-identical and that a normal (unmodified)
    forward afterward reproduces the pre-intervention prediction exactly."""
    _assert_eval_only()
    primitive = _load_p_primitive(core, "I01", REC004E_DECISIVE_STEP)
    if primitive is None:
        return {"status": "SOURCE_ARTIFACT_UNAVAILABLE"}
    before_hash = mb.canonical_state_hash(primitive.state_dict())

    content_lengths = [6, 7, 8, 9, 10]
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    d_model = primitive.content_in_proj.in_features
    torch.manual_seed(2026)
    content = torch.randn(len(content_lengths), lmax, d_model, device=core.device)
    with torch.no_grad():
        pre_logits = primitive(content, content_lengths, output_lengths, None)
        pre_preds = _predict_from_logits(pre_logits, output_lengths)
        forward_order_logits = {}
        for j_id in REC004E_J_IDS:
            out = run_intervention_forward(
                primitive, content, content_lengths, output_lengths, j_id
            )
            forward_order_logits[j_id] = out["logits"].detach().clone()
        post_logits = primitive(content, content_lengths, output_lengths, None)
        post_preds = _predict_from_logits(post_logits, output_lengths)
    after_hash = mb.canonical_state_hash(primitive.state_dict())

    reverse_order_logits = {}
    with torch.no_grad():
        for j_id in reversed(REC004E_J_IDS):
            out = run_intervention_forward(
                primitive, content, content_lengths, output_lengths, j_id
            )
            reverse_order_logits[j_id] = out["logits"].detach().clone()
    order_invariant = all(
        torch.equal(forward_order_logits[j_id], reverse_order_logits[j_id])
        for j_id in REC004E_J_IDS
    )
    return {
        "task_id": REC004E_TASK_ID,
        "weights_unchanged": before_hash == after_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "normal_forward_restored_after_all_interventions": pre_preds == post_preds,
        "intervention_order_invariant": order_invariant,
        "logit_diff_before_after": (pre_logits - post_logits).abs().max().item(),
    }


# =============================================================================
# Stage C4 -- late-phase learning curve, same-LR-phase comparison.
# =============================================================================


def run_late_phase_curve_audit() -> dict[str, Any]:
    """C4: reads REC-004D's own saved `learning_curve.jsonl` (no new
    forward passes) and reports the P arm's existing_validation EM/loss at
    steps 4000/4500/5000/5500/6000, plus same-LR-phase comparisons
    identified by matching `lr_after_scheduler` (not by step number alone,
    since `CosineAnnealingLR(T_max=1000)` repeats every 1000 steps)."""
    path = REC004E_SOURCE_RUN_DIR / "learning_curve.jsonl"
    if not path.is_file():
        return {"status": "UNAVAILABLE", "reason": f"{path} not found"}
    rows_by_init: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("arm") != REC004E_ARM_P:
                continue
            rows_by_init.setdefault(row["init_id"], []).append(row)

    late_steps = (4000, 4500, 5000, 5500, 6000)
    per_init: dict[str, Any] = {}
    for init_id in REC004E_INIT_IDS:
        rows = sorted(rows_by_init.get(init_id, []), key=lambda r: r["step"])
        by_step = {r["step"]: r for r in rows}
        late_curve: dict[str, Any] = {}
        for step in late_steps:
            r = by_step.get(step)
            if r is None:
                late_curve[str(step)] = "UNAVAILABLE"
                continue
            ev = r.get("existing_validation", {})
            train_fit = r.get("train_fit", {})
            late_curve[str(step)] = {
                "lr_after_scheduler": r.get("lr_after_scheduler"),
                "correct_exact_match": ev.get("correct_exact_match"),
                "correct_token_accuracy": ev.get("correct_token_accuracy"),
                "train_fit_sequence_exact_match": train_fit.get("sequence_exact_match"),
            }

        same_phase_groups: dict[str, list[int]] = {}
        for step, r in by_step.items():
            lr_val = r.get("lr_after_scheduler")
            lr_key = f"{lr_val:.6f}" if lr_val is not None else "NA"
            same_phase_groups.setdefault(lr_key, []).append(step)
        same_phase_comparisons = []
        for lr_key, steps in sorted(same_phase_groups.items()):
            if len(steps) < 2:
                continue
            steps_sorted = sorted(steps)
            ems = [
                by_step[s].get("existing_validation", {}).get("correct_exact_match")
                for s in steps_sorted
            ]
            same_phase_comparisons.append(
                {"lr_after_scheduler": lr_key, "steps": steps_sorted, "existing_validation_em": ems}
            )

        per_init[init_id] = {
            "late_curve": late_curve,
            "same_lr_phase_comparisons": same_phase_comparisons,
        }

    return {
        "task_id": REC004E_TASK_ID,
        "status": "COMPLETE",
        "per_init": per_init,
        "caveat": (
            "Same-LR-phase groups are identified from the recorded lr_after_scheduler value "
            "at each checkpoint, not assumed from step-number arithmetic. No extrapolation "
            "beyond step=6000 and no interpolation between saved checkpoints is performed."
        ),
    }


# =============================================================================
# Stage D -- residual diagnosis (evidence-backed labels, never forced) and
# the (at most one, PROPOSED_NOT_AUTHORIZED) next repair contract.
# =============================================================================

# Pre-registered thresholds for the automated labels below -- fixed before
# the real run's numbers were seen, so a label is a mechanical readout of the
# raw evidence (also reported in full alongside every label), not a
# post-hoc narrative choice.
_SCORE_BALANCE_DIVERGENCE_THRESHOLD: Final = 0.05
_LENGTH_FEATURE_SENSITIVITY_THRESHOLD: Final = 0.02
_ROW_AFFINE_LEAD_FRACTION_THRESHOLD: Final = 0.5
_OPTIMIZATION_PROGRESS_MIN_DELTA: Final = 0.01


def build_residual_diagnosis(
    source_replay: dict[str, Any],
    score_observer_parity: dict[str, Any],
    position_grid_index: dict[str, Any],
    activation_summary: dict[str, Any],
    score_decomposition: dict[str, Any],
    interventions: dict[str, Any],
    late_phase_curve: dict[str, Any],
) -> dict[str, Any]:
    """D1: mechanically derives evidence-backed labels from the pre-
    registered thresholds above -- multiple labels may co-occur, and
    `UNRESOLVED` is added whenever evidence is mixed or a prerequisite
    check did not pass. Never selects a single "cause"."""
    labels: list[str] = []
    unresolved: list[str] = []
    evidence: dict[str, Any] = {}

    contract_ok = (
        source_replay.get("status") == "VERIFIED"
        and position_grid_index.get("fixed_position_contract", {}).get("all_match") is True
    )
    if not contract_ok:
        labels.append("EXECUTION_CONTRACT_FAILURE")
        evidence["execution_contract_failure"] = {
            "source_replay_status": source_replay.get("status"),
            "fixed_position_contract_all_match": position_grid_index.get(
                "fixed_position_contract", {}
            ).get("all_match"),
        }

    score_available = score_observer_parity.get("status") == "VERIFIED"
    if not score_available:
        unresolved.append(
            "SCORE_OBSERVATION_UNAVAILABLE: score_observer_parity did not verify on every "
            "check -- S_other/B decomposition and/or the J0-J6 intervention matrix are "
            "restricted to whichever checks did pass (see score_observer_parity.json)."
        )

    per_init_summary = interventions.get("per_init_summary", {})

    def _collect_deltas(j_ids: tuple[str, ...], lengths: tuple[int, ...]) -> list[float]:
        out = []
        for summary in per_init_summary.values():
            if not isinstance(summary, dict):
                continue
            for j_id in j_ids:
                per_length = summary.get(j_id, {}).get("per_length", {})
                for n in lengths:
                    row = per_length.get(str(n))
                    if row is not None and row.get("paired_delta_vs_j0") is not None:
                        out.append(row["paired_delta_vs_j0"])
        return out

    len10_deltas = _collect_deltas(("J2", "J3"), (10,))
    other_len_deltas = _collect_deltas(("J2", "J3"), (6, 7, 8, 9))
    len10_mean = statistics.mean(len10_deltas) if len10_deltas else None
    other_mean = statistics.mean(other_len_deltas) if other_len_deltas else None
    score_balance_evidence: dict[str, Any] = {
        "j2_j3_length10_paired_delta_mean": len10_mean,
        "j2_j3_length6to9_paired_delta_mean": other_mean,
        "threshold": _SCORE_BALANCE_DIVERGENCE_THRESHOLD,
    }
    if len10_mean is not None and other_mean is not None:
        divergence = abs(len10_mean - other_mean)
        score_balance_evidence["observed_divergence"] = divergence
        if divergence >= _SCORE_BALANCE_DIVERGENCE_THRESHOLD:
            labels.append("SCORE_BALANCE_SENSITIVITY_OBSERVED")
    evidence["score_balance_sensitivity"] = score_balance_evidence

    j5_deltas = _collect_deltas(("J5",), (10,))
    j6_deltas = _collect_deltas(("J6",), (9,))
    length_feature_evidence = {
        "j5_length10_paired_delta_mean": statistics.mean(j5_deltas) if j5_deltas else None,
        "j6_length9_paired_delta_mean": statistics.mean(j6_deltas) if j6_deltas else None,
        "threshold": _LENGTH_FEATURE_SENSITIVITY_THRESHOLD,
    }
    max_abs_delta = max((abs(d) for d in (j5_deltas + j6_deltas)), default=None)
    length_feature_evidence["max_abs_paired_delta"] = max_abs_delta
    if max_abs_delta is not None and max_abs_delta >= _LENGTH_FEATURE_SENSITIVITY_THRESHOLD:
        labels.append("LENGTH_FEATURE_SENSITIVITY_OBSERVED")
    evidence["length_feature_sensitivity"] = length_feature_evidence

    row_affine_fraction = position_grid_index.get("row_affine_segment", {}).get("fraction")
    if (
        row_affine_fraction is not None
        and row_affine_fraction >= _ROW_AFFINE_LEAD_FRACTION_THRESHOLD
    ):
        labels.append("POSITION_DISCRIMINATION_LEAD")
    evidence["position_discrimination_lead"] = {
        "row_affine_fraction": row_affine_fraction,
        "threshold": _ROW_AFFINE_LEAD_FRACTION_THRESHOLD,
    }

    margin_vs_em: dict[str, Any] = {}
    len10_margins, len10_ems = [], []
    for _init_id, per_init in score_decomposition.get("per_init", {}).items():
        row = per_init.get("per_length", {}).get("10") if isinstance(per_init, dict) else None
        if row and row.get("correct_position_margin_mean") is not None:
            len10_margins.append(row["correct_position_margin_mean"])
            len10_ems.append(row.get("sequence_exact_match"))
    margin_vs_em["length10_correct_position_margin_mean"] = (
        statistics.mean(len10_margins) if len10_margins else None
    )
    margin_vs_em["length10_sequence_exact_match_mean"] = (
        statistics.mean([e for e in len10_ems if e is not None]) if len10_ems else None
    )
    interaction_lead = (
        margin_vs_em["length10_correct_position_margin_mean"] is not None
        and margin_vs_em["length10_sequence_exact_match_mean"] is not None
        and margin_vs_em["length10_correct_position_margin_mean"] > 0.05
        and margin_vs_em["length10_sequence_exact_match_mean"] < 0.5
    )
    if interaction_lead:
        labels.append("SCORE_COMPONENT_INTERACTION_LEAD")
    evidence["score_component_interaction"] = margin_vs_em

    em_mean = margin_vs_em.get("length10_sequence_exact_match_mean")
    if em_mean is not None and not interaction_lead:
        labels.append("POSITION_SCORE_SUFFICIENCY_NOT_ESTABLISHED")

    progress_deltas = []
    for _init_id, curve in late_phase_curve.get("per_init", {}).items():
        for group in curve.get("same_lr_phase_comparisons", []):
            ems = [e for e in group["existing_validation_em"] if e is not None]
            if len(ems) >= 2:
                progress_deltas.append(ems[-1] - ems[0])
    evidence["optimization_progress"] = {
        "same_phase_deltas": progress_deltas,
        "threshold": _OPTIMIZATION_PROGRESS_MIN_DELTA,
    }
    if progress_deltas and statistics.mean(progress_deltas) >= _OPTIMIZATION_PROGRESS_MIN_DELTA:
        labels.append("OPTIMIZATION_PROGRESS_OBSERVED")

    if not labels or unresolved:
        labels.append("UNRESOLVED")

    return {
        "task_id": REC004E_TASK_ID,
        "labels": sorted(set(labels)),
        "unresolved": unresolved,
        "evidence": evidence,
        "note": (
            "Labels are a mechanical readout of the pre-registered thresholds above against "
            "this run's own measured evidence (all included here); they are descriptive "
            "observation labels, not a selected root cause, per the task doc's own D1 table."
        ),
    }


def build_next_repair_contract(diagnosis: dict[str, Any]) -> str:
    """D2: writes at most one proposed (never authorized, never implemented)
    next repair contract, or `EVIDENCE_INSUFFICIENT` if the evidence does
    not support one. Selecting AND implementing a repair in the same task
    is exactly what this function must not do."""
    labels = diagnosis["labels"]
    evidence = diagnosis["evidence"]

    if "EXECUTION_CONTRACT_FAILURE" in labels:
        status = "EVIDENCE_INSUFFICIENT"
        body = (
            "An execution-contract failure was detected during this audit (source replay "
            "and/or the fixed-position contract did not verify). No repair is proposed "
            "until the contract failure itself is investigated and resolved; proposing a "
            "position/score repair on top of an unverified measurement pipeline would not "
            "be evidence-backed."
        )
    elif "UNRESOLVED" in labels and len(labels) == 1:
        status = "EVIDENCE_INSUFFICIENT"
        body = (
            "No single observation label cleared its pre-registered threshold with the "
            "measured evidence, or score observation was unavailable for part of the run. "
            "See residual_diagnosis.json's `evidence` block for the exact numbers and "
            "`unresolved` block for what is missing. The smallest next step is repeating "
            "Stage C's score decomposition with score-observer parity re-verified (or an "
            "alternative, hook-based-only reconstruction) before proposing a repair."
        )
    elif "SCORE_COMPONENT_INTERACTION_LEAD" in labels:
        status = "PROPOSED_NOT_AUTHORIZED"
        sci = evidence.get("score_component_interaction", {})
        margin_ref = sci.get("length10_correct_position_margin_mean")
        em_ref = sci.get("length10_sequence_exact_match_mean")
        body = f"""対象mechanism: MIRROR_HALVES's `CrossPositionLengthBiasPrimitive` -- specifically
how the existing (position-embedding-derived) attention score S_other combines with the
new length-conditioned bias term B before softmax.

Raw artifact reference: `score_decomposition_index.json`'s length-10
`correct_position_margin_mean`={margin_ref} against `sequence_exact_match`={em_ref}
(from this task's own `score_decomposition.npz`/`_index.json`), and
`intervention_results.jsonl`'s J2/J3 length-10-vs-6-9 divergence
(`residual_diagnosis.json`'s `score_balance_sensitivity` block).

観測: at length 10, the head-averaged correct-position margin is descriptively positive
(the bias-augmented score ranks the teacher-aligned key favorably) while sequence EM
remains low -- combined-score sufficiency for correct output is NOT established at this
length even though position discrimination itself looks reasonable by this proxy metric.
推論 (not proven): the residual may sit downstream of the attention distribution itself
(value projection, `attn_norm`/`ffn` residual path, or `readout`), not in the position
score alone.
提案 (未実装, 次task用の提案値): before touching the bias feature again, add a strictly
diagnostic (still read-only, still zero training) probe that swaps in the ORACLE softmax
attention distribution (built from the true `pi_n` mapping, one-hot over valid keys) in
place of the model's own combined attention on a copy of the SAME step=6000 checkpoints,
and measures whether length-10 EM recovers. This isolates whether the bottleneck is the
attention distribution itself or the value/residual/readout path downstream of it, without
committing to any specific score-mechanism redesign.

修正前後で不変にするもの: Core, all 16 primitive originals, router, ArgumentScorer,
verifier, the 3 fixed candidates, shared cache, and every REC-004D checkpoint file.

追加parameter数・計算量の変化: 0 (the proposed next step is a forward-only oracle-
attention substitution probe, not a new trainable module).

対照・交絡: use the SAME 5 saved P/I01-I05 step=6000 checkpoints and the SAME
length-balanced diagnostic suite as this task, so the only difference from this task's
own J0 is the substituted attention distribution -- no new init, no new data.

既存5初期化の扱い: report all 5 individually; never average weights or select a "best"
result; the pre-registered 5/5-at-0.95 adoption rule and RG3's non-SHIFT-15 floor are
UNCHANGED by this or any future task unless a new explicit instruction says otherwise.

学習を要する場合の有限上限: none -- this specific next step is forward-only.

採用条件: not defined here (this is a diagnostic probe proposal, not a candidate); any
future repair built on its result would need its own pre-registered adoption rule
evaluated across lengths 6-10, per REC-004D's own unchanged floor.

独立query／STOP: no new query set; existing_validation and the length-balanced suite
only; STOP after the probe's result is reported, same as this task.

現APIでの変更対象: none in `src/apc/primitives/primitive.py` (probe reads the same
saved checkpoints via a new, isolated evaluation function, analogous in spirit to this
task's own `run_intervention_forward`/`_manual_attention`); would add one new function to
a similar `src/apc/evaluation/*.py` diagnostic module and its own CPU fixture tests.
"""
    elif "LENGTH_FEATURE_SENSITIVITY_OBSERVED" in labels:
        status = "PROPOSED_NOT_AUTHORIZED"
        body = f"""対象mechanism: the length-normalization feature `n/length_ref` (phi's 4th
component) inside `CrossPositionLengthBiasPrimitive._position_bias`.

Raw artifact reference: `intervention_results.jsonl` J5 (n=10, phi4 set to 9/32) and J6
(n=9, phi4 set to 10/32) paired deltas vs J0
(`residual_diagnosis.json`'s `length_feature_sensitivity` block: J5 mean delta=
{evidence.get('length_feature_sensitivity', {}).get('j5_length10_paired_delta_mean')},
J6 mean delta={evidence.get('length_feature_sensitivity', {}).get('j6_length9_paired_delta_mean')}).

観測: substituting a neighboring length's normalized-length value measurably moves the
model's output at the substituted length, evidence the learned MLP is sensitive to this
specific feature's scale in the trained region.
推論 (not proven): this does NOT establish that `n/length_ref` (vs. e.g. an unnormalized
`n`, or a per-length one-hot/embedding) is itself the root cause of the length-10
residual -- it shows sensitivity, not which alternative coordinate would help.
提案 (未実装, 次task用の提案値): a future task could pre-register a SINGLE alternative
coordinate for phi's 4th component (e.g. a learned per-length scalar embedding shared
across the 5 trained lengths, replacing `n/length_ref` only) and repeat REC-004D's own
paired 5-init U/P-style comparison (same Core, same data stream, same recipe, same
0.95-on-all-5 adoption rule) -- never multiple candidate coordinates explored within one
task, per AGENTS.md's "smallest falsifying change" principle.

修正前後で不変にするもの: phi's first 3 components (i/d, j/d, (j-i)/d), hidden width 32,
head sharing, output-zero initialization, Core, all other 15 primitives, router,
ArgumentScorer, verifier, fixed 3 candidates, shared cache.

追加parameter数・計算量の変化: proposal-dependent (a learned per-length embedding over 5
legal lengths would add on the order of 5-10 new parameters, well under the existing 5%
budget ceiling) -- to be finalized by the task that implements it, not here.

対照・交絡: same parent, same Core, same REC-004C initial states (I01-I05), same
per-step training stream generator as REC-004D, so init/data are not confounded with the
feature change.

既存5初期化の扱い: same as REC-004D's own rule -- report all 5, never average or
best-of-5 select.

学習を要する場合の有限上限: same 6000-updates-per-run / 60000-total budget REC-004D used
(this task doc gives no basis to propose a different budget).

採用条件: same 0.95-on-all-5-at-step=6000 floor, evaluated across lengths 6-10
individually (never averaged into one number) -- unchanged from REC-004D/this task's own
mandate.

独立query／STOP: no independent query touched by this proposal; any RG3 recheck remains a
separate, explicitly authorized future task.

現APIでの変更対象: `src/apc/primitives/primitive.py` (`CrossPositionLengthBiasPrimitiveConfig`/
`CrossPositionLengthBiasPrimitive`, a new feature-version constant), plus tests analogous
to the existing zero-bias-parity/gradient-path/mask-layout/serialization suite.
"""
    else:
        status = "EVIDENCE_INSUFFICIENT"
        body = (
            "The measured evidence supports at least one descriptive label (see "
            "residual_diagnosis.json), but none of this task's own decision rules maps it "
            "to a single, well-scoped next mechanism change with enough specificity to "
            "propose here without guessing. Recommended smallest next step: re-run this "
            "same audit's Stage C against a wider score_observation_subset (still read-only, "
            "still forward-only) before committing to a specific repair direction."
        )

    lines = [
        "# Next Repair Contract (Proposed) -- B-C005REC-004E",
        "",
        f"status: {status}",
        "",
        f"residual_diagnosis_labels: {', '.join(labels)}",
        "",
        "**This file's existence is not authorization to implement or train it. Implementation "
        "and training each require their own explicit next user instruction.**",
        "",
        "## Contract",
        "",
        body.strip(),
        "",
        "## Fixed, unchanged rules (this or any future task)",
        "",
        "- REC-004D's own adoption rule (all 5 P runs' terminal existing_validation EM >= "
        "0.95, P/I01 fixed regardless of which init scores highest) is NOT relaxed by this "
        "diagnosis's results.",
        "- RG3's non-SHIFT-15-operations-each->=0.95 functional floor is unchanged.",
        "- No candidate is selected, no child bundle is built, and no RG3 recheck runs from "
        "this file alone.",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Orchestration.
# =============================================================================


def _config_to_yaml_dict(config: MirrorPositionScoreResidualAuditConfig) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_mirror_position_score_residual_audit_task(
    config: MirrorPositionScoreResidualAuditConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_position_score_residual_audit_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004E reads REC-004D artifacts pre-registered under seed "
            f"{RECOVERY_PILOT_SEED}; got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    source_manifest = lock_source_manifest()
    _write_json(output_dir / "source_manifest.json", source_manifest)
    if source_manifest["status"] != "AVAILABLE":
        result: dict[str, Any] = {
            "implementation_status": "PARTIAL",
            "source_replay_status": "BLOCKED",
            "position_grid_status": "NOT_EXECUTED",
            "score_observation_status": "NOT_EXECUTED" if True else "INVALID",
            "intervention_status": "NOT_EXECUTED",
            "late_curve_status": "UNAVAILABLE",
            "residual_diagnosis": "DIAGNOSTIC_BLOCKED",
            "next_repair_contract": "EVIDENCE_INSUFFICIENT",
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_intervention": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
            "source_manifest": source_manifest,
        }
        _write_json(output_dir / "summary.json", result)
        return result

    parent_manifest, _raw = ibc._load_parent_manifest()
    loaded = mb.load_bundle(parent_manifest, mode="diagnostic", expected_primitive_count=16)
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )

    protected_hashes_before = bias_repair._protected_scope_hashes(eval_bank, op_to_id)

    with torch.no_grad():
        source_replay = run_source_replay(core, eval_bank, op_to_id)
    _write_json(output_dir / "source_replay.json", source_replay)

    data_manifest = {
        "task_id": REC004E_TASK_ID,
        "source_replay": {
            "split": REC004E_EXISTING_VALIDATION_SPLIT,
            "n_examples": REC004E_EXISTING_VALIDATION_EXAMPLES,
        },
        "grid": {
            "n_values": list(REC004E_LEGAL_LENGTHS),
            "n_position_pairs": sum(n * n for n in REC004E_LEGAL_LENGTHS),
            "note": "purely functional over (i, j, n); no tokens/content required",
        },
        "score_observation_subset": {
            "split": REC004E_LENGTH_BALANCED_SPLIT,
            "n_per_length": REC004E_SCORE_OBS_PER_LENGTH,
            "selection": (
                "first N examples per length in generation order, fixed before any output "
                "is seen"
            ),
        },
        "intervention_suite": {
            "split": REC004E_LENGTH_BALANCED_SPLIT,
            "n_per_length": REC004E_INTERVENTION_PER_LENGTH,
        },
        "forbidden": ["reference", "final_query", "sealed", "rec004d_recheck_query"],
    }
    _write_json(output_dir / "data_manifest.json", data_manifest)

    if source_replay["status"] != "VERIFIED":
        after_hashes = _snapshot_forbidden_cache_hashes(seed)
        result = {
            "implementation_status": "PARTIAL",
            "source_replay_status": "BLOCKED",
            "position_grid_status": "NOT_EXECUTED",
            "score_observation_status": "NOT_EXECUTED",
            "intervention_status": "NOT_EXECUTED",
            "late_curve_status": "UNAVAILABLE",
            "residual_diagnosis": "DIAGNOSTIC_BLOCKED",
            "next_repair_contract": "EVIDENCE_INSUFFICIENT",
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_intervention": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
            "source_replay": source_replay,
            "side_effect_audit": {
                "before": before_hashes, "after": after_hashes,
                "shared_cache_unchanged": before_hashes == after_hashes,
            },
        }
        _write_json(output_dir / "summary.json", result)
        return result

    with torch.no_grad():
        score_observer_parity = run_score_observer_parity(core)
    _write_json(output_dir / "score_observer_parity.json", score_observer_parity)

    with torch.no_grad():
        grid = run_position_grid_enumeration(core)
    np.savez_compressed(output_dir / "position_grid.npz", **grid["arrays"])
    _write_json(output_dir / "position_grid_index.json", grid["index"])
    activation_summary = _summarize_activation_rows(grid["activation_rows"])
    _write_json(output_dir / "activation_summary.json", activation_summary)
    row_centered_bias_summary = _summarize_row_centered_rows(grid["row_centered_rows"])
    _write_json(output_dir / "row_centered_bias_summary.json", row_centered_bias_summary)

    score_observation_status = "UNAVAILABLE"
    intervention_status = "NOT_EXECUTED"
    score_decomposition: dict[str, Any] = {}
    interventions: dict[str, Any] = {"rows": [], "per_init_summary": {}}
    nonmutation_audit: dict[str, Any] = {}

    if score_observer_parity["status"] == "VERIFIED":
        with torch.no_grad():
            score_decomp_result = run_score_decomposition(core)
        score_decomposition = score_decomp_result["index"]
        np.savez_compressed(output_dir / "score_decomposition.npz", **score_decomp_result["arrays"])
        _write_json(output_dir / "score_decomposition_index.json", score_decomposition)
        score_observation_status = "VERIFIED"

        with torch.no_grad():
            interventions = run_interventions(core)
        with (output_dir / "intervention_results.jsonl").open("w", encoding="utf-8") as fh:
            for row in interventions["rows"]:
                fh.write(json.dumps(row, default=str) + "\n")
        _write_json(
            output_dir / "paired_intervention_summary.json", interventions["per_init_summary"]
        )
        with torch.no_grad():
            nonmutation_audit = run_intervention_nonmutation_audit(core)
        _write_json(output_dir / "intervention_nonmutation_audit.json", nonmutation_audit)
        intervention_status = "COMPLETE"
    else:
        _write_json(
            output_dir / "score_decomposition_index.json",
            {"status": "SCORE_OBSERVATION_UNAVAILABLE", "reason": score_observer_parity["status"]},
        )
        _write_json(
            output_dir / "intervention_nonmutation_audit.json",
            {"status": "NOT_EXECUTED", "reason": "score_observer_parity did not verify"},
        )

    late_phase_curve = run_late_phase_curve_audit()
    _write_json(output_dir / "late_phase_learning_audit.json", late_phase_curve)

    diagnosis = build_residual_diagnosis(
        source_replay, score_observer_parity, grid["index"], activation_summary,
        score_decomposition, interventions, late_phase_curve,
    )
    _write_json(output_dir / "residual_diagnosis.json", diagnosis)
    next_repair_contract_text = build_next_repair_contract(diagnosis)
    (output_dir / "next_repair_contract.md").write_text(next_repair_contract_text, encoding="utf-8")

    protected_hashes_after = bias_repair._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004E_TASK_ID,
        "core_canonical_state_hash_before": parent_manifest.core.canonical_state_hash,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == parent_manifest.core.canonical_state_hash,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "loaded_bundle_checks_performed": list(loaded.checks_performed),
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004E_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    cost_accounting = {
        "task_id": REC004E_TASK_ID,
        "new_optimizer_updates": 0,
        "n_grid_scalars": grid["index"]["total_scalars"],
        "n_score_observation_examples": (
            score_decomposition.get("total_examples_per_model", 0) * len(REC004E_INIT_IDS)
            if score_decomposition else 0
        ),
        "n_intervention_predictions": sum(
            row_count["overall_n"]
            for summary in interventions["per_init_summary"].values()
            if isinstance(summary, dict)
            for row_count in summary.values()
            if isinstance(row_count, dict) and "overall_n" in row_count
        ),
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=seed)
    _write_json(output_dir / "system.json", system_info)

    protocol = {
        "task_id": REC004E_TASK_ID,
        "source_task_id": REC004E_SOURCE_TASK_ID,
        "source_run_dir": str(REC004E_SOURCE_RUN_DIR.resolve()),
        "source_manifest": source_manifest,
        "checkpoint_steps": list(REC004E_CHECKPOINT_STEPS),
        "legal_lengths": list(REC004E_LEGAL_LENGTHS),
        "score_observation_per_length": REC004E_SCORE_OBS_PER_LENGTH,
        "intervention_per_length": REC004E_INTERVENTION_PER_LENGTH,
        "j_conditions": list(REC004E_J_IDS),
        "tolerances": {
            "source_replay_em_abs_tol": REC004E_SOURCE_REPLAY_EM_TOL,
            "score_logit_abs_tol": REC004E_SCORE_LOGIT_ABS_TOL,
            "manual_reconstruction_abs_tol": REC004E_MANUAL_RECONSTRUCTION_ABS_TOL,
        },
        "forbidden": [
            "new optimizer updates", "production feature/L_ref/temperature changes",
            "candidate adoption", "child assembly", "RG3 recheck",
            "oracle position lookup fed into a forward input",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "residual_audit_protocol.json", protocol)

    grid_complete = grid["index"]["n_grids_available"] == grid["index"]["n_grids_expected"]
    result = {
        "implementation_status": "COMPLETE",
        "source_replay_status": source_replay["status"],
        "position_grid_status": "COMPLETE" if grid_complete else "PARTIAL",
        "score_observation_status": score_observation_status,
        "intervention_status": intervention_status,
        "late_curve_status": late_phase_curve.get("status", "UNAVAILABLE"),
        "residual_diagnosis": diagnosis["labels"],
        "next_repair_contract": next_repair_contract_text.splitlines()[2].split(": ", 1)[1],
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_intervention": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "source_manifest": source_manifest,
        "source_replay": source_replay,
        "score_observer_parity": score_observer_parity,
        "position_grid_index": grid["index"],
        "activation_summary_available": True,
        "residual_diagnosis_full": diagnosis,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": cost_accounting,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "summary.json", result)
    return result
