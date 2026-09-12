"""B-C005REC-004Z: I03 Key/Value Content-Prep Functional-Role Isolation Pilot.

Implements minimal architecture-level functional-role isolation separating
KEY_CONTENT_PREP from VALUE_CONTENT_PREP while holding VALUE_CONTENT_PREP + V + O + F
strictly frozen at step 7500, granting plasticity to KEY_CONTENT_PREP + Q/K + position bias
under unconstrained normal J0 training.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
import torch.nn as nn
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
from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_downstream_freeze_causal_replay as rec004v
from apc.evaluation import mirror_normal_cvof_protection_pilot as rec004w
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb

__all__ = [
    "REC004Z_TASK_ID",
    "REC004Z_SOURCE_TASK_IDS",
    "REC004Z_TARGET_OPERATION",
    "REC004Z_ARM",
    "REC004Z_SEED",
    "REC004Z_DECISIVE_INIT",
    "REC004Z_START_STEP",
    "REC004Z_END_STEP",
    "REC004Z_MAX_UPDATES",
    "REC004Z_ORACLE_EM_THRESHOLD",
    "REC004Z_J0_DELTA_FLOOR",
    "REC004Z_HARD_FREEZE_DELTA_FLOOR",
    "RoleSplitCrossPositionLengthBiasPrimitive",
    "MirrorKVRoleSplitPilotConfig",
    "build_kv_role_split_validation_v1",
    "build_kv_role_split_length10_v1",
    "build_parity_fixture_v1",
    "prepare_rec004z_datasets",
    "verify_role_split_initial_parity",
    "verify_role_split_gradient_path_isolation",
    "run_mirror_kv_role_split_pilot_task",
]

REC004Z_TASK_ID: Final = "B-C005REC-004Z"
REC004Z_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004T",
    "B-C005REC-004U",
    "B-C005REC-004V",
    "B-C005REC-004W",
    "B-C005REC-004X",
    "B-C005REC-004Y",
)

REC004Z_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004Z_ARM: Final = "P_LENGTH_POSITION_BIAS"
REC004Z_SEED: Final = RECOVERY_PILOT_SEED
REC004Z_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE
REC004Z_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE

REC004Z_DECISIVE_INIT: Final = "I03"

REC004Z_START_STEP: Final = 7500
REC004Z_END_STEP: Final = 8000
REC004Z_MAX_UPDATES: Final = 500
REC004Z_ORACLE_EM_THRESHOLD: Final = 0.95
REC004Z_J0_DELTA_FLOOR: Final = 0.10
REC004Z_HARD_FREEZE_DELTA_FLOOR: Final = 0.05
REC004Z_FULL_PROBE_INTERVAL: Final = 25
REC004Z_SENTINEL_SUBSET_PER_DATASET: Final = 64

REC004Z_CONTINUITY_SPLITS: Final[tuple[str, ...]] = rec004w.REC004W_CONTINUITY_SPLITS

REC004Z_FRESH_NORMAL_VALIDATION: Final = "kv_role_split_validation_v1"
REC004Z_FRESH_LENGTH10_CONFIRMATION: Final = "kv_role_split_length10_v1"
REC004Z_PARITY_FIXTURE: Final = "kv_role_split_parity_fixture_v1"
REC004Z_FRESH_VALIDATION_EXAMPLES: Final = 1024
REC004Z_FRESH_LENGTH10_EXAMPLES: Final = 512
REC004Z_PARITY_FIXTURE_EXAMPLES: Final = 512

REC004Z_POSITION_4: Final = 4
REC004Z_POSITION_5: Final = 5
REC004Z_CORRECT_KEY_P4: Final = 0
REC004Z_CORRECT_KEY_P5: Final = 9


@dataclass(frozen=True)
class MirrorKVRoleSplitPilotConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004z/run_001")
    seed: int = REC004Z_SEED
    oracle_em_threshold: float = REC004Z_ORACLE_EM_THRESHOLD
    j0_delta_floor: float = REC004Z_J0_DELTA_FLOOR
    hard_freeze_delta_floor: float = REC004Z_HARD_FREEZE_DELTA_FLOOR
    max_replay_window_updates: int = REC004Z_MAX_UPDATES
    full_probe_step_interval: int = REC004Z_FULL_PROBE_INTERVAL
    sentinel_subset_per_dataset: int = REC004Z_SENTINEL_SUBSET_PER_DATASET
    parity_updates: int = 0


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


def _derive_local_seed_helper(seed: int, index: int, label: str) -> int:
    h = hashlib.sha256(f"{seed}:{index}:{label}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class RoleSplitCrossPositionLengthBiasPrimitive(CrossPositionLengthBiasPrimitive):
    """CrossPositionLengthBiasPrimitive with separated KEY and VALUE content prep paths."""

    def __init__(
        self,
        primitive_id: int,
        config: CrossPositionLengthBiasPrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            config,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.key_content_in_proj = nn.Linear(config.d_model, config.d_operator)
        self.key_content_position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_operator
        )

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        if not self.enabled:
            return content_features.new_zeros(batch, out_max, self.config.vocab_size)

        self.forward_call_count += 1

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)

        k_in = self.key_content_in_proj(content_features) + self.key_content_position_embedding(
            content_position_ids
        )
        v_in = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None or self.arg_encoder is None or self.arg_proj is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)

        query = query_slots + arg_token

        bias = self._position_bias(content_lengths, out_max, lmax, device)
        bias = bias.unsqueeze(1).expand(batch, self.n_head, out_max, lmax)

        content_lengths_t = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
        p_idx_long = torch.arange(lmax, device=device).view(1, 1, lmax)
        pad_mask = (
            (p_idx_long >= content_lengths_t).unsqueeze(1).expand(batch, self.n_head, out_max, lmax)
        )
        attn_mask = torch.where(pad_mask, torch.tensor(float("-inf"), device=device), bias).reshape(
            batch * self.n_head, out_max, lmax
        )

        attn_out, _ = self.cross_attn(query, k_in, v_in, attn_mask=attn_mask, need_weights=False)
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)


def evaluate_role_split_forward_with_stages(
    primitive: RoleSplitCrossPositionLengthBiasPrimitive,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
    oracle_attention: bool = False,
) -> dict[str, Any]:
    """Runs forward and extracts internal stages for RoleSplitCrossPositionLengthBiasPrimitive."""
    device = content_features.device
    batch, lmax, _ = content_features.shape
    out_max = max(output_lengths)

    content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
    k_in = primitive.key_content_in_proj(
        content_features
    ) + primitive.key_content_position_embedding(content_position_ids)
    v_in = primitive.content_in_proj(content_features) + primitive.content_position_embedding(
        content_position_ids
    )

    query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
    query = primitive.answer_query_embedding(query_ids)

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
    k = F.linear(k_in, wk, bk)
    v = F.linear(v_in, wv, bv)

    q = q.view(batch, out_max, n_head, head_dim).transpose(1, 2)
    k = k.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    v = v.view(batch, lmax, n_head, head_dim).transpose(1, 2)

    s_other = torch.matmul(q, k.transpose(-2, -1)) * scale

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

    attn_out_heads = torch.matmul(attn_probs, v)
    attn_out_concat = attn_out_heads.transpose(1, 2).reshape(batch, out_max, embed_dim)
    attn_out = mha.out_proj(attn_out_concat)

    post_attn_res_in = query + attn_out
    post_attn_norm_out = primitive.attn_norm(post_attn_res_in)
    ffn_out = primitive.ffn(post_attn_norm_out)
    post_ffn_rep = primitive.ffn_norm(post_attn_norm_out + ffn_out)
    final_token_logits = primitive.readout(post_ffn_rep)

    return {
        "final_token_logits": final_token_logits,
        "score_logits": scores,
        "attn_probs": attn_probs,
        "attn_out": attn_out,
        "post_attn_res_in": post_attn_res_in,
        "post_attn_norm_out": post_attn_norm_out,
        "ffn_out": ffn_out,
        "post_ffn_rep": post_ffn_rep,
        "k_in": k_in,
        "v_in": v_in,
    }


def _build_mirror_halves_dataset(
    seed: int,
    protected_digests: set[str],
    n: int,
    split_name: str,
    length_range: tuple[int, int],
) -> tuple[list[Example], dict[str, Any]]:
    seed_label = f"{split_name}:{REC004Z_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed_helper(seed, 0, seed_label))
    op_obj = get_operation(REC004Z_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            if length_range[0] == length_range[1]:
                seq_len = length_range[0]
            else:
                seq_len = rng.randint(*length_range)
            seq = tuple(rng.randrange(REC004Z_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(rng, seq, REC004Z_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004Z_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004Z_VOCAB_SIZE)
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
                split=split_name,
                vocab_size=REC004Z_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004Z_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "sequence_length_range": list(length_range),
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            f"{split_name} is a development-exposed dataset, not a sealed or final RG3 query set."
        ),
    }
    return examples, detail


def build_kv_role_split_validation_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004Z_FRESH_VALIDATION_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    return _build_mirror_halves_dataset(
        seed, protected_digests, n, REC004Z_FRESH_NORMAL_VALIDATION, REC004Z_SEQUENCE_LENGTH_RANGE
    )


def build_kv_role_split_length10_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004Z_FRESH_LENGTH10_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    return _build_mirror_halves_dataset(
        seed, protected_digests, n, REC004Z_FRESH_LENGTH10_CONFIRMATION, (10, 10)
    )


def build_parity_fixture_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004Z_PARITY_FIXTURE_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    return _build_mirror_halves_dataset(
        seed, protected_digests, n, REC004Z_PARITY_FIXTURE, REC004Z_SEQUENCE_LENGTH_RANGE
    )


def prepare_rec004z_datasets(seed: int) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    print("Preparing REC-004Z datasets with exhaustive protection...")
    protected, protected_counts = rec004t.build_protected_registry_exhaustive(seed)

    # Add REC-004W datasets
    w_val_exs, _ = rec004w.build_normal_cvof_protection_validation_v1(seed, protected)
    w_val_digests = _digest_examples(w_val_exs)
    protected |= w_val_digests
    protected_counts["rec004w_validation"] = len(w_val_digests)

    w_conf_exs, _ = rec004w.build_normal_cvof_protection_length10_v1(seed, protected)
    w_conf_digests = _digest_examples(w_conf_exs)
    protected |= w_conf_digests
    protected_counts["rec004w_length10"] = len(w_conf_digests)

    datasets: dict[str, list[Example]] = {}
    cont_probe_exs, _ = rec004t.prepare_probe_datasets(seed)
    datasets[REC004Z_CONTINUITY_SPLITS[0]] = cont_probe_exs[REC004Z_CONTINUITY_SPLITS[0]]
    datasets[REC004Z_CONTINUITY_SPLITS[1]] = cont_probe_exs[REC004Z_CONTINUITY_SPLITS[1]]

    exs3, _ = rec004u.build_attention_clamp_causal_probe_v1(seed, protected)
    datasets[REC004Z_CONTINUITY_SPLITS[2]] = exs3

    exs4, _ = rec004v.build_downstream_freeze_causal_probe_v1(seed, protected)
    datasets[REC004Z_CONTINUITY_SPLITS[3]] = exs4

    for split_name in REC004Z_CONTINUITY_SPLITS:
        digests = _digest_examples(datasets[split_name])
        protected |= digests
        protected_counts[f"continuity_{split_name}"] = len(digests)

    parity_exs, parity_detail = build_parity_fixture_v1(seed, protected)
    parity_digests = _digest_examples(parity_exs)
    protected |= parity_digests
    protected_counts["rec004z_parity_fixture"] = len(parity_digests)
    datasets[REC004Z_PARITY_FIXTURE] = parity_exs

    fresh_val_exs, fresh_val_detail = build_kv_role_split_validation_v1(seed, protected)
    fresh_val_digests = _digest_examples(fresh_val_exs)
    protected |= fresh_val_digests
    protected_counts["rec004z_validation"] = len(fresh_val_digests)
    datasets[REC004Z_FRESH_NORMAL_VALIDATION] = fresh_val_exs

    fresh_conf_exs, fresh_conf_detail = build_kv_role_split_length10_v1(seed, protected)
    fresh_conf_digests = _digest_examples(fresh_conf_exs)
    protected |= fresh_conf_digests
    protected_counts["rec004z_length10"] = len(fresh_conf_digests)
    datasets[REC004Z_FRESH_LENGTH10_CONFIRMATION] = fresh_conf_exs

    manifest = {
        "task_id": REC004Z_TASK_ID,
        "seed": seed,
        "continuity_splits": {
            s: {
                "n_examples": len(datasets[s]),
                "dataset_digest": _dataset_digest(_digest_examples(datasets[s])),
            }
            for s in REC004Z_CONTINUITY_SPLITS
        },
        "parity_fixture": {
            "name": REC004Z_PARITY_FIXTURE,
            "n_examples": len(parity_exs),
            "dataset_digest": _dataset_digest(parity_digests),
            "sequence_length_range": list(REC004Z_SEQUENCE_LENGTH_RANGE),
            "disjoint_verified": True,
        },
        "fresh_normal_validation": {
            "name": REC004Z_FRESH_NORMAL_VALIDATION,
            "n_examples": len(fresh_val_exs),
            "dataset_digest": _dataset_digest(fresh_val_digests),
            "sequence_length_range": list(REC004Z_SEQUENCE_LENGTH_RANGE),
            "disjoint_verified": True,
        },
        "fresh_length10_confirmation": {
            "name": REC004Z_FRESH_LENGTH10_CONFIRMATION,
            "n_examples": len(fresh_conf_exs),
            "dataset_digest": _dataset_digest(fresh_conf_digests),
            "target_length": 10,
            "disjoint_verified": True,
        },
        "protected_registry_counts": protected_counts,
        "datasets_locked_before_model_eval": True,
    }
    return datasets, manifest


def create_role_split_primitive_and_optimizer(
    core: Any,
    start_ts: dict[str, Any],
    device: torch.device,
) -> tuple[RoleSplitCrossPositionLengthBiasPrimitive, torch.optim.AdamW, dict[str, Any]]:
    config = CrossPositionLengthBiasPrimitiveConfig(
        operation=REC004Z_TARGET_OPERATION,
        d_model=core.model.config.d_model,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=REC004Z_VOCAB_SIZE,
        max_sequence_length=32,
        arg_dim=16,
        bias_hidden_dim=32,
        length_ref=32,
    )
    model = RoleSplitCrossPositionLengthBiasPrimitive(primitive_id=999, config=config)
    model.to(device)

    legacy_sd = start_ts["primitive_state_dict"]
    legacy_opt_sd = start_ts["optimizer_state_dict"]

    model_sd = model.state_dict()
    for k, v in legacy_sd.items():
        if k in model_sd:
            model_sd[k].copy_(v.to(device))

    model.key_content_in_proj.weight.data.copy_(legacy_sd["content_in_proj.weight"].to(device))
    if "content_in_proj.bias" in legacy_sd and legacy_sd["content_in_proj.bias"] is not None:
        model.key_content_in_proj.bias.data.copy_(legacy_sd["content_in_proj.bias"].to(device))
    model.key_content_position_embedding.weight.data.copy_(
        legacy_sd["content_position_embedding.weight"].to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mbe.REC004G_OPERATOR_LR,
        weight_decay=mbe.REC004G_OPERATOR_WEIGHT_DECAY,
    )

    named_params = dict(model.named_parameters())
    legacy_param_names = [
        "content_in_proj.weight",
        "content_in_proj.bias",
        "content_position_embedding.weight",
        "answer_query_embedding.weight",
        "cross_attn.in_proj_weight",
        "cross_attn.in_proj_bias",
        "cross_attn.out_proj.weight",
        "cross_attn.out_proj.bias",
        "attn_norm.weight",
        "attn_norm.bias",
        "ffn.0.weight",
        "ffn.0.bias",
        "ffn.2.weight",
        "ffn.2.bias",
        "ffn_norm.weight",
        "ffn_norm.bias",
        "readout.weight",
        "readout.bias",
        "position_bias_hidden.weight",
        "position_bias_hidden.bias",
        "position_bias_out.weight",
    ]

    legacy_p_ids = legacy_opt_sd["param_groups"][0]["params"]

    for idx, p_name in enumerate(legacy_param_names):
        p_id = legacy_p_ids[idx]
        if p_id in legacy_opt_sd["state"]:
            st = legacy_opt_sd["state"][p_id]
            param_obj = named_params[p_name]
            raw_step = st["step"]
            step_tensor = (
                raw_step.clone().to(device)
                if isinstance(raw_step, torch.Tensor)
                else torch.tensor(float(raw_step), device=device)
            )
            optimizer.state[param_obj] = {
                "step": step_tensor,
                "exp_avg": st["exp_avg"].detach().clone().to(device),
                "exp_avg_sq": st["exp_avg_sq"].detach().clone().to(device),
            }

    clone_mapping = [
        ("content_in_proj.weight", "key_content_in_proj.weight"),
        ("content_in_proj.bias", "key_content_in_proj.bias"),
        ("content_position_embedding.weight", "key_content_position_embedding.weight"),
    ]

    migration_entries: list[dict[str, Any]] = []
    for src_name, dst_name in clone_mapping:
        src_p = named_params[src_name]
        dst_p = named_params[dst_name]
        src_st = optimizer.state[src_p]
        optimizer.state[dst_p] = {
            "step": src_st["step"].clone(),
            "exp_avg": src_st["exp_avg"].detach().clone().to(device),
            "exp_avg_sq": src_st["exp_avg_sq"].detach().clone().to(device),
        }
        migration_entries.append(
            {
                "source_parameter": src_name,
                "target_parameter": dst_name,
                "source_param_hash": hashlib.sha256(
                    src_p.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "target_param_hash": hashlib.sha256(
                    dst_p.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "source_exp_avg_hash": hashlib.sha256(
                    src_st["exp_avg"].detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "target_exp_avg_hash": hashlib.sha256(
                    optimizer.state[dst_p]["exp_avg"].detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "step_counter": int(src_st["step"].item())
                if isinstance(src_st["step"], torch.Tensor)
                else int(src_st["step"]),
                "status": "EXACT_CLONED",
            }
        )

    migration_audit = {
        "task_id": REC004Z_TASK_ID,
        "migration_timestamp": time.time(),
        "entries": migration_entries,
        "status": "PASS",
    }

    return model, optimizer, migration_audit


def verify_role_split_initial_parity(
    core: Any,
    role_split_model: RoleSplitCrossPositionLengthBiasPrimitive,
    reference_7500: CrossPositionLengthBiasPrimitive,
    evaluation_examples: list[Example],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    print("\n--- Running Stage A: Exact-Parity Gate (updates = 0) ---")
    role_split_model.eval()
    reference_7500.eval()
    role_split_model.to(device)
    reference_7500.to(device)

    max_score_diff = 0.0
    max_prob_diff = 0.0
    max_attn_out_diff = 0.0
    max_post_ffn_diff = 0.0
    max_logits_diff = 0.0
    total_prediction_mismatches = 0
    total_examples = len(evaluation_examples)

    with torch.no_grad():
        for start_idx in range(0, total_examples, batch_size):
            chunk = evaluation_examples[start_idx : start_idx + batch_size]
            c_lens = [len(ex.input_tokens) for ex in chunk]
            o_lens = [get_operation(REC004Z_TARGET_OPERATION).output_length(n) for n in c_lens]

            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

            ref_out = rec004t.evaluate_forward_with_stages(
                reference_7500, h, c_lens, o_lens, oracle_attention=False
            )
            role_out = evaluate_role_split_forward_with_stages(
                role_split_model, h, c_lens, o_lens, oracle_attention=False
            )

            direct_ref_logits = reference_7500(h, c_lens, o_lens, None)
            direct_role_logits = role_split_model(h, c_lens, o_lens, None)

            s_diff = float(
                torch.max(torch.abs(role_out["score_logits"] - ref_out["score_logits"])).item()
            )
            p_diff = float(
                torch.max(torch.abs(role_out["attn_probs"] - ref_out["attn_probs"])).item()
            )
            ao_diff = float(torch.max(torch.abs(role_out["attn_out"] - ref_out["attn_out"])).item())
            pf_diff = float(
                torch.max(torch.abs(role_out["post_ffn_rep"] - ref_out["post_ffn_rep"])).item()
            )
            l_diff = float(
                torch.max(
                    torch.abs(role_out["final_token_logits"] - ref_out["final_token_logits"])
                ).item()
            )
            direct_diff = float(torch.max(torch.abs(direct_role_logits - direct_ref_logits)).item())

            max_score_diff = max(max_score_diff, s_diff)
            max_prob_diff = max(max_prob_diff, p_diff)
            max_attn_out_diff = max(max_attn_out_diff, ao_diff)
            max_post_ffn_diff = max(max_post_ffn_diff, pf_diff)
            max_logits_diff = max(max_logits_diff, l_diff, direct_diff)

            ref_preds = torch.argmax(ref_out["final_token_logits"], dim=-1)
            role_preds = torch.argmax(role_out["final_token_logits"], dim=-1)
            total_prediction_mismatches += int((ref_preds != role_preds).sum().item())

    prob_tolerance = 1e-5
    logit_tolerance = 1e-4

    parity_passed = (
        total_prediction_mismatches == 0
        and max_prob_diff <= prob_tolerance
        and max_logits_diff <= logit_tolerance
    )

    report = {
        "stage": "STAGE_A_EXACT_PARITY_GATE",
        "n_examples_tested": total_examples,
        "score_logits_max_abs_diff": max_score_diff,
        "attn_probs_max_abs_diff": max_prob_diff,
        "attn_out_max_abs_diff": max_attn_out_diff,
        "post_ffn_max_abs_diff": max_post_ffn_diff,
        "final_logits_max_abs_diff": max_logits_diff,
        "discrete_prediction_mismatches": total_prediction_mismatches,
        "prob_tolerance": prob_tolerance,
        "logit_tolerance": logit_tolerance,
        "status": "PASS" if parity_passed else "ROLE_SPLIT_LEGACY_PARITY_FAILURE",
    }

    if not parity_passed:
        raise RuntimeError(
            f"ROLE_SPLIT_LEGACY_PARITY_FAILURE: Parity mismatch! "
            f"prob_diff={max_prob_diff:.2e}, logit_diff={max_logits_diff:.2e}, "
            f"mismatches={total_prediction_mismatches}"
        )

    print(
        f"Stage A Exact-Parity Gate: PASS | "
        f"max_prob_diff={max_prob_diff:.2e} <= {prob_tolerance}, "
        f"max_logits_diff={max_logits_diff:.2e} <= {logit_tolerance}, "
        f"mismatches={total_prediction_mismatches}"
    )
    return report


def verify_role_split_gradient_path_isolation(
    core: Any,
    model: RoleSplitCrossPositionLengthBiasPrimitive,
    examples: list[Example],
    device: torch.device,
) -> dict[str, Any]:
    print("\n--- Running Stage B: Gradient-Path Isolation Gate ---")
    embed_dim = model.d_operator
    sample = examples[:16]
    c_lens = [len(ex.input_tokens) for ex in sample]
    o_lens = [get_operation(REC004Z_TARGET_OPERATION).output_length(n) for n in c_lens]
    out_max = max(o_lens)
    labels = _labels_for_examples(sample, o_lens, out_max, device)

    batch_input = collate_content_only_batch(sample, core.tokens, device=device)
    h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

    # 1. J0 Normal forward gradients
    model.zero_grad(set_to_none=True)
    j0_logits = model(h, c_lens, o_lens, None)
    j0_loss = F.cross_entropy(
        j0_logits.reshape(-1, j0_logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    j0_loss.backward()
    assert model.key_content_in_proj.weight.grad is not None
    assert model.key_content_position_embedding.weight.grad is not None
    assert model.cross_attn.in_proj_weight.grad is not None
    assert model.content_in_proj.weight.grad is not None

    j0_key_in_proj_grad_norm = float(model.key_content_in_proj.weight.grad.norm().item())
    j0_key_pos_emb_grad_norm = float(model.key_content_position_embedding.weight.grad.norm().item())
    j0_k_proj_grad_norm = float(
        model.cross_attn.in_proj_weight.grad[embed_dim : 2 * embed_dim].norm().item()
    )
    j0_val_in_proj_grad_norm = float(model.content_in_proj.weight.grad.norm().item())

    # 2. O1 Oracle Attention forward gradients
    model.zero_grad(set_to_none=True)
    o1_out = evaluate_role_split_forward_with_stages(
        model, h, c_lens, o_lens, oracle_attention=True
    )
    o1_logits = o1_out["final_token_logits"]
    o1_loss = F.cross_entropy(
        o1_logits.reshape(-1, o1_logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    o1_loss.backward()

    o1_key_in_proj_grad_norm = (
        float(model.key_content_in_proj.weight.grad.norm().item())
        if model.key_content_in_proj.weight.grad is not None
        else 0.0
    )
    o1_key_pos_emb_grad_norm = (
        float(model.key_content_position_embedding.weight.grad.norm().item())
        if model.key_content_position_embedding.weight.grad is not None
        else 0.0
    )
    o1_k_proj_grad_norm = (
        float(model.cross_attn.in_proj_weight.grad[embed_dim : 2 * embed_dim].norm().item())
        if model.cross_attn.in_proj_weight.grad is not None
        else 0.0
    )

    o1_val_in_proj_grad_norm = (
        float(model.content_in_proj.weight.grad.norm().item())
        if model.content_in_proj.weight.grad is not None
        else 0.0
    )
    o1_v_proj_grad_norm = (
        float(model.cross_attn.in_proj_weight.grad[2 * embed_dim : 3 * embed_dim].norm().item())
        if model.cross_attn.in_proj_weight.grad is not None
        else 0.0
    )
    o1_out_proj_grad_norm = (
        float(model.cross_attn.out_proj.weight.grad.norm().item())
        if model.cross_attn.out_proj.weight.grad is not None
        else 0.0
    )
    o1_ffn_grad_norm = float(
        sum(p.grad.norm().item() for p in model.ffn.parameters() if p.grad is not None)
    )

    tolerance = 1e-12
    j0_key_active = (j0_key_in_proj_grad_norm > 0.0) and (j0_k_proj_grad_norm > 0.0)
    o1_key_isolated = (
        o1_key_in_proj_grad_norm <= tolerance
        and o1_key_pos_emb_grad_norm <= tolerance
        and o1_k_proj_grad_norm <= tolerance
    )
    o1_value_path_active = (
        o1_val_in_proj_grad_norm > 0.0
        and o1_v_proj_grad_norm > 0.0
        and o1_out_proj_grad_norm > 0.0
        and o1_ffn_grad_norm > 0.0
    )

    isolation_passed = j0_key_active and o1_key_isolated and o1_value_path_active

    report = {
        "stage": "STAGE_B_GRADIENT_PATH_ISOLATION_GATE",
        "j0_key_in_proj_grad_norm": j0_key_in_proj_grad_norm,
        "j0_key_pos_emb_grad_norm": j0_key_pos_emb_grad_norm,
        "j0_k_proj_grad_norm": j0_k_proj_grad_norm,
        "j0_val_in_proj_grad_norm": j0_val_in_proj_grad_norm,
        "j0_key_active": j0_key_active,
        "o1_key_in_proj_grad_norm": o1_key_in_proj_grad_norm,
        "o1_key_pos_emb_grad_norm": o1_key_pos_emb_grad_norm,
        "o1_k_proj_grad_norm": o1_k_proj_grad_norm,
        "o1_key_isolated": o1_key_isolated,
        "o1_val_in_proj_grad_norm": o1_val_in_proj_grad_norm,
        "o1_v_proj_grad_norm": o1_v_proj_grad_norm,
        "o1_out_proj_grad_norm": o1_out_proj_grad_norm,
        "o1_ffn_grad_norm": o1_ffn_grad_norm,
        "o1_value_path_active": o1_value_path_active,
        "tolerance": tolerance,
        "status": "PASS" if isolation_passed else "ROLE_ISOLATION_FORWARD_GRAPH_INVALID",
    }

    if not isolation_passed:
        raise RuntimeError(
            f"ROLE_ISOLATION_FORWARD_GRAPH_INVALID: Gradient isolation failed! "
            f"j0_key_active={j0_key_active}, o1_key_isolated={o1_key_isolated}, "
            f"o1_val_active={o1_value_path_active}"
        )

    print(
        f"Stage B Gradient-Path Isolation Gate: PASS | "
        f"J0 key_grad={j0_key_in_proj_grad_norm:.2e}, "
        f"O1 key_grad={o1_key_in_proj_grad_norm:.2e} <= {tolerance}, "
        f"O1 val_grad={o1_val_in_proj_grad_norm:.2e}"
    )
    return report


def evaluate_length10_metrics_role_split(
    core: Any,
    primitive: RoleSplitCrossPositionLengthBiasPrimitive,
    examples: list[Example],
    batch_size: int = 128,
) -> dict[str, Any]:
    primitive.eval()
    device = core.device
    n_total = len(examples)

    j0_seq_matches = 0
    o1_seq_matches = 0
    j0_p4_matches = 0
    j0_p5_matches = 0
    o1_p4_matches = 0
    o1_p5_matches = 0

    j0_p4_score_margins: list[float] = []
    j0_p4_score_probs: list[float] = []
    j0_p4_correct_key_ranks: list[float] = []

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

            j0_out = evaluate_role_split_forward_with_stages(
                primitive, h, c_lens, o_lens, oracle_attention=False
            )
            j0_preds = torch.argmax(j0_out["final_token_logits"], dim=-1)

            o1_out = evaluate_role_split_forward_with_stages(
                primitive, h, c_lens, o_lens, oracle_attention=True
            )
            o1_preds = torch.argmax(o1_out["final_token_logits"], dim=-1)

            j0_seq_matches += int(torch.all(j0_preds == target_tokens, dim=-1).sum().item())
            o1_seq_matches += int(torch.all(o1_preds == target_tokens, dim=-1).sum().item())
            j0_p4_matches += int(
                (j0_preds[:, REC004Z_POSITION_4] == target_tokens[:, REC004Z_POSITION_4])
                .sum()
                .item()
            )
            j0_p5_matches += int(
                (j0_preds[:, REC004Z_POSITION_5] == target_tokens[:, REC004Z_POSITION_5])
                .sum()
                .item()
            )
            o1_p4_matches += int(
                (o1_preds[:, REC004Z_POSITION_4] == target_tokens[:, REC004Z_POSITION_4])
                .sum()
                .item()
            )
            o1_p5_matches += int(
                (o1_preds[:, REC004Z_POSITION_5] == target_tokens[:, REC004Z_POSITION_5])
                .sum()
                .item()
            )

            j0_scores = j0_out["score_logits"]
            j0_probs = j0_out["attn_probs"]
            for b in range(b_size):
                p4_scores = j0_scores[b, :, REC004Z_POSITION_4, :]
                p4_correct_score = p4_scores[:, REC004Z_CORRECT_KEY_P4]
                p4_wrong = p4_scores.clone()
                p4_wrong[:, REC004Z_CORRECT_KEY_P4] = -torch.inf
                p4_margin = (p4_correct_score - torch.max(p4_wrong, dim=-1).values).mean().item()
                j0_p4_score_margins.append(float(p4_margin))
                j0_p4_score_probs.append(
                    float(j0_probs[b, :, REC004Z_POSITION_4, REC004Z_CORRECT_KEY_P4].mean().item())
                )
                p4_ranks = (p4_wrong > p4_correct_score.unsqueeze(-1)).sum(dim=-1) + 1
                j0_p4_correct_key_ranks.append(float(p4_ranks.float().mean().item()))

    return {
        "n_examples": n_total,
        "j0_sequence_em": j0_seq_matches / n_total,
        "j0_position4_acc": j0_p4_matches / n_total,
        "j0_position5_acc": j0_p5_matches / n_total,
        "j0_p4_score_margin_median": float(np.median(j0_p4_score_margins)),
        "j0_p4_score_prob_median": float(np.median(j0_p4_score_probs)),
        "j0_p4_correct_key_rank_mean": float(np.mean(j0_p4_correct_key_ranks)),
        "o1_sequence_em": o1_seq_matches / n_total,
        "o1_position4_acc": o1_p4_matches / n_total,
        "o1_position5_acc": o1_p5_matches / n_total,
    }


def evaluate_variable_length_metrics_role_split(
    core: Any,
    primitive: RoleSplitCrossPositionLengthBiasPrimitive,
    examples: list[Example],
    batch_size: int = 128,
) -> dict[str, Any]:
    primitive.eval()
    device = core.device
    n_total = len(examples)
    sequence_correct_all = 0
    by_length: dict[int, dict[str, Any]] = {length: {"n": 0, "exact": 0} for length in range(2, 11)}

    len10_examples = [ex for ex in examples if len(ex.input_tokens) == 10]

    with torch.no_grad():
        for start_idx in range(0, n_total, batch_size):
            chunk = examples[start_idx : start_idx + batch_size]
            b_size = len(chunk)
            c_lens = [len(ex.input_tokens) for ex in chunk]
            o_lens = [get_operation(REC004Z_TARGET_OPERATION).output_length(n) for n in c_lens]
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(c_lens), :]

            logits = primitive(h, c_lens, o_lens, None)
            preds = torch.argmax(logits, dim=-1)

            for b in range(b_size):
                tgt = torch.tensor(chunk[b].target_tokens, device=device, dtype=torch.long)
                pred_b = preds[b, : o_lens[b]]
                is_exact = bool(torch.equal(pred_b, tgt))
                if is_exact:
                    sequence_correct_all += 1
                c_len = c_lens[b]
                by_length[c_len]["n"] += 1
                if is_exact:
                    by_length[c_len]["exact"] += 1

    per_length_em = {
        f"length_{length}_em": (
            by_length[length]["exact"] / by_length[length]["n"]
            if by_length[length]["n"] > 0
            else 0.0
        )
        for length in range(2, 11)
    }

    len10_metrics: dict[str, Any] = {}
    if len10_examples:
        len10_metrics = evaluate_length10_metrics_role_split(core, primitive, len10_examples)

    return {
        "n_examples": n_total,
        "overall_j0_sequence_em": sequence_correct_all / n_total,
        "length_10_j0_sequence_em": per_length_em.get("length_10_em", 0.0),
        "per_length_em": per_length_em,
        "length_10_subset_o1_metrics": len10_metrics,
    }


def get_or_replay_rec004w_step8000_model(
    core: Any,
    start_ts: dict[str, Any],
    seed: int,
    device: torch.device,
) -> CrossPositionLengthBiasPrimitive:
    ckpt_path = Path("runs/phase_b_b2_model_bundle_recovery/rec004w/run_001/checkpoint_step8000.pt")
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = mpbr._new_arm_primitive(core, REC004Z_ARM)
        assert isinstance(model, CrossPositionLengthBiasPrimitive)
        model.to(device)
        model.load_state_dict({k: v.to(device) for k, v in state["primitive_state_dict"].items()})
        model.eval()
        return model

    print("Replaying REC-004W hard-freeze control trajectory (7500->8000)...")
    freeze_model = mpbr._new_arm_primitive(core, REC004Z_ARM)
    assert isinstance(freeze_model, CrossPositionLengthBiasPrimitive)
    freeze_model.to(device)
    freeze_model.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}
    )
    freeze_opt = torch.optim.AdamW(
        freeze_model.parameters(),
        lr=mbe.REC004G_OPERATOR_LR,
        weight_decay=mbe.REC004G_OPERATOR_WEIGHT_DECAY,
    )
    freeze_opt.load_state_dict(start_ts["optimizer_state_dict"])
    freeze_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        freeze_opt,
        T_max=mbe.REC004G_T_MAX,
        eta_min=mbe.REC004G_SCHEDULER_ETA_MIN,
    )
    freeze_sched.load_state_dict(start_ts["scheduler_state_dict"])
    rec004t._restore_rng_state(start_ts, device)

    init_p = {name: p.detach().clone() for name, p in freeze_model.named_parameters()}
    embed_dim = freeze_model.d_operator

    freeze_model.train()
    for s_step in range(REC004Z_START_STEP + 1, REC004Z_END_STEP + 1):
        exs = ibc._generate_step_training_examples(
            seed,
            s_step,
            REC004Z_TARGET_OPERATION,
            vocab_size=REC004Z_VOCAB_SIZE,
            sequence_length_range=REC004Z_SEQUENCE_LENGTH_RANGE,
        )
        c_lens = [len(ex.input_tokens) for ex in exs]
        o_lens = [get_operation(REC004Z_TARGET_OPERATION).output_length(n) for n in c_lens]
        labels = _labels_for_examples(exs, o_lens, max(o_lens), device)

        with torch.no_grad():
            b_in = collate_content_only_batch(exs, core.tokens, device=device)
            h = core.model.encode(b_in)[:, 1 : 1 + max(c_lens), :]

        freeze_opt.zero_grad(set_to_none=True)
        out = freeze_model(h, c_lens, o_lens, None)
        freeze_loss = F.cross_entropy(
            out.reshape(-1, out.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        freeze_loss.backward()

        # Zero out grads strictly for CVOF
        if freeze_model.content_in_proj.weight.grad is not None:
            freeze_model.content_in_proj.weight.grad.zero_()
        if (
            freeze_model.content_in_proj.bias is not None
            and freeze_model.content_in_proj.bias.grad is not None
        ):
            freeze_model.content_in_proj.bias.grad.zero_()
        if freeze_model.content_position_embedding.weight.grad is not None:
            freeze_model.content_position_embedding.weight.grad.zero_()
        if freeze_model.cross_attn.in_proj_weight.grad is not None:
            freeze_model.cross_attn.in_proj_weight.grad[2 * embed_dim : 3 * embed_dim].zero_()
        if (
            freeze_model.cross_attn.in_proj_bias is not None
            and freeze_model.cross_attn.in_proj_bias.grad is not None
        ):
            freeze_model.cross_attn.in_proj_bias.grad[2 * embed_dim : 3 * embed_dim].zero_()
        if freeze_model.cross_attn.out_proj.weight.grad is not None:
            freeze_model.cross_attn.out_proj.weight.grad.zero_()
        if (
            freeze_model.cross_attn.out_proj.bias is not None
            and freeze_model.cross_attn.out_proj.bias.grad is not None
        ):
            freeze_model.cross_attn.out_proj.bias.grad.zero_()
        for p in freeze_model.ffn.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if freeze_model.ffn_norm.weight.grad is not None:
            freeze_model.ffn_norm.weight.grad.zero_()
        if freeze_model.ffn_norm.bias is not None and freeze_model.ffn_norm.bias.grad is not None:
            freeze_model.ffn_norm.bias.grad.zero_()

        torch.nn.utils.clip_grad_norm_(freeze_model.parameters(), mbe.REC004G_OPERATOR_GRAD_CLIP)
        freeze_opt.step()
        freeze_sched.step()

        # Hard restoration of CVOF
        freeze_model.content_in_proj.weight.data.copy_(init_p["content_in_proj.weight"])
        if freeze_model.content_in_proj.bias is not None:
            freeze_model.content_in_proj.bias.data.copy_(init_p["content_in_proj.bias"])
        freeze_model.content_position_embedding.weight.data.copy_(
            init_p["content_position_embedding.weight"]
        )
        freeze_model.cross_attn.in_proj_weight.data[2 * embed_dim : 3 * embed_dim].copy_(
            init_p["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim]
        )
        if freeze_model.cross_attn.in_proj_bias is not None:
            freeze_model.cross_attn.in_proj_bias.data[2 * embed_dim : 3 * embed_dim].copy_(
                init_p["cross_attn.in_proj_bias"][2 * embed_dim : 3 * embed_dim]
            )
        freeze_model.cross_attn.out_proj.weight.data.copy_(init_p["cross_attn.out_proj.weight"])
        if freeze_model.cross_attn.out_proj.bias is not None:
            freeze_model.cross_attn.out_proj.bias.data.copy_(init_p["cross_attn.out_proj.bias"])
        for f_name, f_param in freeze_model.ffn.named_parameters():
            f_param.data.copy_(init_p[f"ffn.{f_name}"])
        freeze_model.ffn_norm.weight.data.copy_(init_p["ffn_norm.weight"])
        if freeze_model.ffn_norm.bias is not None:
            freeze_model.ffn_norm.bias.data.copy_(init_p["ffn_norm.bias"])

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": REC004Z_END_STEP, "primitive_state_dict": freeze_model.state_dict()}, ckpt_path
    )
    freeze_model.eval()
    return freeze_model


def run_mirror_kv_role_split_pilot_task(config: MirrorKVRoleSplitPilotConfig) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_kv_role_split_pilot_task")
    _snapshot_forbidden_cache_hashes(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"=== Starting Task {REC004Z_TASK_ID}: "
        "I03 Key/Value Content-Prep Functional-Role Isolation Pilot ==="
    )
    print(f"Target device: {device}, Seed: {seed}, Output: {output_dir}")

    # 1. Source verification (@7500)
    start_step = REC004Z_START_STEP
    init_id = REC004Z_DECISIVE_INIT
    start_ts_path = rec004t._training_state_path(init_id, start_step)
    if not start_ts_path.is_file():
        raise FileNotFoundError(f"SOURCE_7500_STATE_MISMATCH: Missing state {start_ts_path}")

    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)
    req_keys = {
        "primitive_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "cpu_rng_state",
        "cuda_rng_state",
        "cumulative_updates",
    }
    if not req_keys.issubset(start_ts.keys()):
        raise RuntimeError("SOURCE_7500_STATE_MISMATCH: Missing required keys")

    if start_ts.get("cumulative_updates") != start_step:
        raise RuntimeError("SOURCE_7500_STATE_MISMATCH: cumulative_updates mismatch")

    source_hash = mb.canonical_state_hash(start_ts["primitive_state_dict"])
    expected_step7500_hash = "7c71a7a43ec2ba766623a70cf4b62e18a8ad685bb1073657ef3fd54d676cb3cb"
    if source_hash != expected_step7500_hash:
        raise RuntimeError(
            f"SOURCE_7500_STATE_MISMATCH: hash {source_hash} != {expected_step7500_hash}"
        )

    source_manifest = {
        "task_id": REC004Z_TASK_ID,
        "source_task_ids": list(REC004Z_SOURCE_TASK_IDS),
        "source_step7500_state_path": str(start_ts_path),
        "source_canonical_primitive_state_hash": source_hash,
        "cumulative_updates": start_step,
        "next_training_step": start_step + 1,
        "optimizer_state_present": True,
        "scheduler_state_present": True,
        "cpu_rng_present": True,
        "cuda_rng_present": True,
        "source_status": "VERIFIED",
    }
    _write_json(output_dir / "source_manifest.json", source_manifest)

    rec004t_step8000_ckpt = rec004t._checkpoint_path("I03", REC004Z_END_STEP)
    if not rec004t_step8000_ckpt.is_file():
        raise FileNotFoundError(f"Missing historical step 8000 checkpoint: {rec004t_step8000_ckpt}")

    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    datasets, fresh_manifest = prepare_rec004z_datasets(seed=seed)
    _write_json(output_dir / "fresh_validation_manifest.json", fresh_manifest)

    reference_7500 = mpbr._new_arm_primitive(core, REC004Z_ARM)
    assert isinstance(reference_7500, CrossPositionLengthBiasPrimitive)
    reference_7500.to(device)
    reference_7500.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}
    )
    reference_7500.eval()
    for p in reference_7500.parameters():
        p.requires_grad_(False)

    role_split_model, optimizer, migration_audit = create_role_split_primitive_and_optimizer(
        core, start_ts, device
    )
    _write_json(output_dir / "optimizer_state_migration.json", migration_audit)

    role_split_arch_manifest = {
        "task_id": REC004Z_TASK_ID,
        "arm_name": "KV_ROLE_SPLIT_PROTECTED_VALUE",
        "base_class": "CrossPositionLengthBiasPrimitive",
        "split_class": "RoleSplitCrossPositionLengthBiasPrimitive",
        "value_content_prep_params": [
            "content_in_proj.weight",
            "content_in_proj.bias",
            "content_position_embedding.weight",
        ],
        "key_content_prep_params": [
            "key_content_in_proj.weight",
            "key_content_in_proj.bias",
            "key_content_position_embedding.weight",
        ],
        "parameter_cloning": "EXACT_CLONE_AT_7500",
        "fused_qkv_rows": {
            "Q_rows": "0:32 (trainable)",
            "K_rows": "32:64 (trainable)",
            "V_rows": "64:96 (frozen at 7500)",
        },
        "value_path_frozen": ["VALUE_CONTENT_PREP", "V_PROJECTION", "ATTN_OUT_PROJ", "FFN_BLOCK"],
        "plastic_path_trainable": [
            "KEY_CONTENT_PREP",
            "Q_PROJECTION",
            "K_PROJECTION",
            "POSITION_BIAS",
            "QUERY_RESIDUAL",
            "POST_ATTN_NORM",
            "READOUT",
        ],
    }
    _write_json(output_dir / "role_split_architecture_manifest.json", role_split_arch_manifest)

    legacy_to_split_mapping = {
        "task_id": REC004Z_TASK_ID,
        "mappings": [
            {
                "legacy_name": "content_in_proj.weight",
                "role_split_names": [
                    "content_in_proj.weight (VALUE)",
                    "key_content_in_proj.weight (KEY)",
                ],
            },
            {
                "legacy_name": "content_in_proj.bias",
                "role_split_names": [
                    "content_in_proj.bias (VALUE)",
                    "key_content_in_proj.bias (KEY)",
                ],
            },
            {
                "legacy_name": "content_position_embedding.weight",
                "role_split_names": [
                    "content_position_embedding.weight (VALUE)",
                    "key_content_position_embedding.weight (KEY)",
                ],
            },
            {
                "legacy_name": "cross_attn.in_proj_weight",
                "role_split_names": ["cross_attn.in_proj_weight (Q:0-32, K:32-64, V:64-96)"],
            },
        ],
    }
    _write_json(output_dir / "legacy_to_role_split_mapping.json", legacy_to_split_mapping)

    parity_eval_examples = (
        datasets[REC004Z_PARITY_FIXTURE] + datasets[REC004Z_CONTINUITY_SPLITS[0]][:64]
    )
    stage_a_report = verify_role_split_initial_parity(
        core, role_split_model, reference_7500, parity_eval_examples, device
    )
    _write_json(output_dir / "initial_forward_parity.json", stage_a_report)

    stage_b_report = verify_role_split_gradient_path_isolation(
        core, role_split_model, datasets[REC004Z_PARITY_FIXTURE][:16], device
    )
    _write_json(output_dir / "gradient_path_isolation.json", stage_b_report)
    _write_json(
        output_dir / "o1_key_branch_invariance.json",
        {
            "task_id": REC004Z_TASK_ID,
            "o1_key_in_proj_grad_norm": stage_b_report["o1_key_in_proj_grad_norm"],
            "o1_key_pos_emb_grad_norm": stage_b_report["o1_key_pos_emb_grad_norm"],
            "o1_k_proj_grad_norm": stage_b_report["o1_k_proj_grad_norm"],
            "o1_key_isolated": stage_b_report["o1_key_isolated"],
            "tolerance": stage_b_report["tolerance"],
            "status": "PASS" if stage_b_report["o1_key_isolated"] else "FAIL",
        },
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=mbe.REC004G_T_MAX,
        eta_min=mbe.REC004G_SCHEDULER_ETA_MIN,
    )
    scheduler.load_state_dict(start_ts["scheduler_state_dict"])
    rec004t._restore_rng_state(start_ts, device)

    init_params: dict[str, torch.Tensor] = {
        name: param.detach().clone() for name, param in role_split_model.named_parameters()
    }
    init_adamw_exp_avg: dict[str, torch.Tensor] = {}
    init_adamw_exp_avg_sq: dict[str, torch.Tensor] = {}
    for name, param in role_split_model.named_parameters():
        if param in optimizer.state:
            init_adamw_exp_avg[name] = optimizer.state[param]["exp_avg"].detach().clone()
            init_adamw_exp_avg_sq[name] = optimizer.state[param]["exp_avg_sq"].detach().clone()

    embed_dim = role_split_model.d_operator  # 32

    key_prep_init_weight = role_split_model.key_content_in_proj.weight.detach().clone()
    key_prep_init_pos_emb = role_split_model.key_content_position_embedding.weight.detach().clone()

    training_trace_file = output_dir / "per_step_training_trace.jsonl"
    metrics_25step_file = output_dir / "per_25step_metrics.jsonl"
    training_trace_f = training_trace_file.open("w", encoding="utf-8")
    metrics_25step_f = metrics_25step_file.open("w", encoding="utf-8")

    def _eval_probe_point(step: int) -> dict[str, Any]:
        rec: dict[str, Any] = {"step": step, "continuity_splits": {}}
        for split_name in REC004Z_CONTINUITY_SPLITS:
            m = evaluate_length10_metrics_role_split(core, role_split_model, datasets[split_name])
            rec["continuity_splits"][split_name] = m
        metrics_25step_f.write(json.dumps(rec) + "\n")
        metrics_25step_f.flush()
        return rec

    role_split_model.eval()
    _ = _eval_probe_point(start_step)
    role_split_model.train()

    print(
        f"\n>>> Executing Stage C: KV_ROLE_SPLIT_PROTECTED_VALUE "
        f"[{start_step + 1} -> {REC004Z_END_STEP}] <<<"
    )
    t_train_start = time.time()
    batch_digests_list: list[dict[str, Any]] = []
    key_prep_grad_accum = 0.0

    for step in range(start_step + 1, REC004Z_END_STEP + 1):
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004Z_TARGET_OPERATION,
            vocab_size=REC004Z_VOCAB_SIZE,
            sequence_length_range=REC004Z_SEQUENCE_LENGTH_RANGE,
        )
        batch_digests = sorted(_digest_examples(examples))
        batch_digests_list.append(
            {"step": step, "batch_digest": _dataset_digest(set(batch_digests))}
        )

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004Z_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)

        logits = role_split_model(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()

        if role_split_model.key_content_in_proj.weight.grad is not None:
            key_prep_grad_accum += float(
                role_split_model.key_content_in_proj.weight.grad.norm().item()
            )

        # Zero out gradients strictly for CVOF parameter groups
        if role_split_model.content_in_proj.weight.grad is not None:
            role_split_model.content_in_proj.weight.grad.zero_()
        if (
            role_split_model.content_in_proj.bias is not None
            and role_split_model.content_in_proj.bias.grad is not None
        ):
            role_split_model.content_in_proj.bias.grad.zero_()
        if role_split_model.content_position_embedding.weight.grad is not None:
            role_split_model.content_position_embedding.weight.grad.zero_()

        if role_split_model.cross_attn.in_proj_weight.grad is not None:
            role_split_model.cross_attn.in_proj_weight.grad[2 * embed_dim : 3 * embed_dim].zero_()
        if (
            role_split_model.cross_attn.in_proj_bias is not None
            and role_split_model.cross_attn.in_proj_bias.grad is not None
        ):
            role_split_model.cross_attn.in_proj_bias.grad[2 * embed_dim : 3 * embed_dim].zero_()

        if role_split_model.cross_attn.out_proj.weight.grad is not None:
            role_split_model.cross_attn.out_proj.weight.grad.zero_()
        if (
            role_split_model.cross_attn.out_proj.bias is not None
            and role_split_model.cross_attn.out_proj.bias.grad is not None
        ):
            role_split_model.cross_attn.out_proj.bias.grad.zero_()

        for p in role_split_model.ffn.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if role_split_model.ffn_norm.weight.grad is not None:
            role_split_model.ffn_norm.weight.grad.zero_()
        if (
            role_split_model.ffn_norm.bias is not None
            and role_split_model.ffn_norm.bias.grad is not None
        ):
            role_split_model.ffn_norm.bias.grad.zero_()

        torch.nn.utils.clip_grad_norm_(
            role_split_model.parameters(), mbe.REC004G_OPERATOR_GRAD_CLIP
        )
        optimizer.step()
        scheduler.step()

        # Fail-closed restoration
        role_split_model.content_in_proj.weight.data.copy_(init_params["content_in_proj.weight"])
        if role_split_model.content_in_proj.bias is not None:
            role_split_model.content_in_proj.bias.data.copy_(init_params["content_in_proj.bias"])
        role_split_model.content_position_embedding.weight.data.copy_(
            init_params["content_position_embedding.weight"]
        )
        for c_name, c_param in [
            ("content_in_proj.weight", role_split_model.content_in_proj.weight),
            ("content_in_proj.bias", role_split_model.content_in_proj.bias),
            (
                "content_position_embedding.weight",
                role_split_model.content_position_embedding.weight,
            ),
        ]:
            if c_param is not None and c_param in optimizer.state:
                optimizer.state[c_param]["exp_avg"].copy_(init_adamw_exp_avg[c_name])
                optimizer.state[c_param]["exp_avg_sq"].copy_(init_adamw_exp_avg_sq[c_name])

        role_split_model.cross_attn.in_proj_weight.data[2 * embed_dim : 3 * embed_dim].copy_(
            init_params["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim]
        )
        if role_split_model.cross_attn.in_proj_bias is not None:
            role_split_model.cross_attn.in_proj_bias.data[2 * embed_dim : 3 * embed_dim].copy_(
                init_params["cross_attn.in_proj_bias"][2 * embed_dim : 3 * embed_dim]
            )
        opt_w = optimizer.state[role_split_model.cross_attn.in_proj_weight]
        opt_w["exp_avg"][2 * embed_dim : 3 * embed_dim].copy_(
            init_adamw_exp_avg["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim]
        )
        opt_w["exp_avg_sq"][2 * embed_dim : 3 * embed_dim].copy_(
            init_adamw_exp_avg_sq["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim]
        )

        role_split_model.cross_attn.out_proj.weight.data.copy_(
            init_params["cross_attn.out_proj.weight"]
        )
        if role_split_model.cross_attn.out_proj.bias is not None:
            role_split_model.cross_attn.out_proj.bias.data.copy_(
                init_params["cross_attn.out_proj.bias"]
            )
        for o_name, o_param in [
            ("cross_attn.out_proj.weight", role_split_model.cross_attn.out_proj.weight),
            ("cross_attn.out_proj.bias", role_split_model.cross_attn.out_proj.bias),
        ]:
            if o_param is not None and o_param in optimizer.state:
                optimizer.state[o_param]["exp_avg"].copy_(init_adamw_exp_avg[o_name])
                optimizer.state[o_param]["exp_avg_sq"].copy_(init_adamw_exp_avg_sq[o_name])

        for f_name, f_param in role_split_model.ffn.named_parameters():
            full_name = f"ffn.{f_name}"
            f_param.data.copy_(init_params[full_name])
            if f_param in optimizer.state:
                optimizer.state[f_param]["exp_avg"].copy_(init_adamw_exp_avg[full_name])
                optimizer.state[f_param]["exp_avg_sq"].copy_(init_adamw_exp_avg_sq[full_name])
        role_split_model.ffn_norm.weight.data.copy_(init_params["ffn_norm.weight"])
        if role_split_model.ffn_norm.bias is not None:
            role_split_model.ffn_norm.bias.data.copy_(init_params["ffn_norm.bias"])
        for fn_name, fn_param in [
            ("ffn_norm.weight", role_split_model.ffn_norm.weight),
            ("ffn_norm.bias", role_split_model.ffn_norm.bias),
        ]:
            if fn_param is not None and fn_param in optimizer.state:
                optimizer.state[fn_param]["exp_avg"].copy_(init_adamw_exp_avg[fn_name])
                optimizer.state[fn_param]["exp_avg_sq"].copy_(init_adamw_exp_avg_sq[fn_name])

        trace_record = {
            "step": step,
            "loss": float(loss.item()),
            "lr": lr_used,
            "key_prep_grad_norm": float(
                role_split_model.key_content_in_proj.weight.grad.norm().item()
            )
            if role_split_model.key_content_in_proj.weight.grad is not None
            else 0.0,
        }
        training_trace_f.write(json.dumps(trace_record) + "\n")

        if step % REC004Z_FULL_PROBE_INTERVAL == 0:
            role_split_model.eval()
            probe_rec = _eval_probe_point(step)
            role_split_model.train()
            pos4_acc = probe_rec["continuity_splits"][REC004Z_CONTINUITY_SPLITS[0]][
                "j0_position4_acc"
            ]
            o1_em = probe_rec["continuity_splits"][REC004Z_CONTINUITY_SPLITS[0]]["o1_sequence_em"]
            print(
                f"Step {step:4d} | Loss: {loss.item():.4f} | "
                f"Pos4 Acc: {pos4_acc:.3f} | O1 EM: {o1_em:.3f}"
            )

    t_train_end = time.time()
    train_duration = t_train_end - t_train_start
    training_trace_f.close()
    metrics_25step_f.close()
    print(f"Training finished in {train_duration:.2f}s")

    _write_json(output_dir / "training_batch_manifest.json", {"batches": batch_digests_list})

    ckpt_path = output_dir / "checkpoint_step8000.pt"
    torch.save(
        {
            "step": REC004Z_END_STEP,
            "primitive_state_dict": role_split_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_path,
    )

    val_in_proj_diff = float(
        torch.max(
            torch.abs(
                role_split_model.content_in_proj.weight.data - init_params["content_in_proj.weight"]
            )
        ).item()
    )
    v_proj_diff = float(
        torch.max(
            torch.abs(
                role_split_model.cross_attn.in_proj_weight.data[2 * embed_dim : 3 * embed_dim]
                - init_params["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim]
            )
        ).item()
    )
    out_proj_diff = float(
        torch.max(
            torch.abs(
                role_split_model.cross_attn.out_proj.weight.data
                - init_params["cross_attn.out_proj.weight"]
            )
        ).item()
    )
    ffn_linear_0 = cast(nn.Linear, role_split_model.ffn[0])
    ffn_diff = float(
        torch.max(torch.abs(ffn_linear_0.weight.data - init_params["ffn.0.weight"])).item()
    )

    freeze_audit = {
        "task_id": REC004Z_TASK_ID,
        "value_content_in_proj_diff": val_in_proj_diff,
        "v_projection_diff": v_proj_diff,
        "attn_out_proj_diff": out_proj_diff,
        "ffn_diff": ffn_diff,
        "all_frozen_diffs_zero": bool(
            max(val_in_proj_diff, v_proj_diff, out_proj_diff, ffn_diff) == 0.0
        ),
        "status": "PASS"
        if max(val_in_proj_diff, v_proj_diff, out_proj_diff, ffn_diff) == 0.0
        else "FAIL",
    }
    _write_json(output_dir / "stable_value_freeze_audit.json", freeze_audit)
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    key_in_proj_delta = float(
        torch.norm(role_split_model.key_content_in_proj.weight.data - key_prep_init_weight).item()
    )
    key_pos_emb_delta = float(
        torch.norm(
            role_split_model.key_content_position_embedding.weight.data - key_prep_init_pos_emb
        ).item()
    )
    key_val_divergence = float(
        torch.norm(
            role_split_model.key_content_in_proj.weight.data
            - role_split_model.content_in_proj.weight.data
        ).item()
    )

    key_branch_active = key_in_proj_delta > 1e-4 and key_prep_grad_accum > 0.0
    key_plasticity_audit = {
        "task_id": REC004Z_TASK_ID,
        "key_content_in_proj_l2_displacement": key_in_proj_delta,
        "key_content_pos_emb_l2_displacement": key_pos_emb_delta,
        "key_vs_value_in_proj_divergence": key_val_divergence,
        "key_content_prep_grad_accumulation": key_prep_grad_accum,
        "key_branch_active": key_branch_active,
        "status": "KEY_BRANCH_PLASTICITY_ACTIVE"
        if key_branch_active
        else "KEY_BRANCH_PLASTICITY_INACTIVE",
    }
    _write_json(output_dir / "key_branch_plasticity_audit.json", key_plasticity_audit)

    role_split_model.eval()

    endpoint_continuity: dict[str, Any] = {}
    for split_name in REC004Z_CONTINUITY_SPLITS:
        endpoint_continuity[split_name] = evaluate_length10_metrics_role_split(
            core, role_split_model, datasets[split_name]
        )

    endpoint_fresh_length10 = evaluate_length10_metrics_role_split(
        core, role_split_model, datasets[REC004Z_FRESH_LENGTH10_CONFIRMATION]
    )
    endpoint_fresh_val = evaluate_variable_length_metrics_role_split(
        core, role_split_model, datasets[REC004Z_FRESH_NORMAL_VALIDATION]
    )

    endpoint_metrics = {
        "task_id": REC004Z_TASK_ID,
        "step": REC004Z_END_STEP,
        "continuity_splits": endpoint_continuity,
        "fresh_length10_confirmation": endpoint_fresh_length10,
        "fresh_normal_validation": endpoint_fresh_val,
    }
    _write_json(output_dir / "endpoint_metrics.json", endpoint_metrics)

    hist_state = torch.load(rec004t_step8000_ckpt, map_location="cpu", weights_only=False)
    hist_model = mpbr._new_arm_primitive(core, REC004Z_ARM)
    assert isinstance(hist_model, CrossPositionLengthBiasPrimitive)
    hist_model.to(device)
    hist_model.load_state_dict({k: v.to(device) for k, v in hist_state.items()})
    hist_model.eval()

    hist_fresh_val = rec004w.evaluate_variable_length_dataset(
        core, hist_model, datasets[REC004Z_FRESH_NORMAL_VALIDATION]
    )
    hist_fresh_conf = rec004w.evaluate_length10_dataset_metrics(
        core, hist_model, datasets[REC004Z_FRESH_LENGTH10_CONFIRMATION]
    )
    hist_comparison = {
        "comparator": "HISTORICAL_I03_STEP8000",
        "fresh_normal_validation": hist_fresh_val,
        "fresh_length10_confirmation": hist_fresh_conf,
    }
    _write_json(output_dir / "historical_comparison.json", hist_comparison)

    freeze_model = get_or_replay_rec004w_step8000_model(core, start_ts, seed, device)

    freeze_fresh_val = rec004w.evaluate_variable_length_dataset(
        core, freeze_model, datasets[REC004Z_FRESH_NORMAL_VALIDATION]
    )
    freeze_fresh_conf = rec004w.evaluate_length10_dataset_metrics(
        core, freeze_model, datasets[REC004Z_FRESH_LENGTH10_CONFIRMATION]
    )
    freeze_comparison = {
        "comparator": "HARD_CVOF_FREEZE_REC004W_STEP8000",
        "fresh_normal_validation": freeze_fresh_val,
        "fresh_length10_confirmation": freeze_fresh_conf,
    }
    _write_json(output_dir / "hard_freeze_comparison.json", freeze_comparison)

    compatibility_passed = True
    compat_audit: dict[str, Any] = {}
    for s_name, m_res in endpoint_continuity.items():
        o1_em = m_res["o1_sequence_em"]
        p4_acc = m_res["o1_position4_acc"]
        passed = bool(
            (o1_em >= config.oracle_em_threshold) and (p4_acc >= config.oracle_em_threshold)
        )
        compat_audit[s_name] = {"o1_em": o1_em, "p4_acc": p4_acc, "pass": passed}
        if not passed:
            compatibility_passed = False

    conf_o1_em = endpoint_fresh_length10["o1_sequence_em"]
    conf_p4_acc = endpoint_fresh_length10["o1_position4_acc"]
    conf_passed = bool(
        (conf_o1_em >= config.oracle_em_threshold) and (conf_p4_acc >= config.oracle_em_threshold)
    )
    compat_audit["fresh_length10_confirmation"] = {
        "o1_em": conf_o1_em,
        "p4_acc": conf_p4_acc,
        "pass": conf_passed,
    }
    if not conf_passed:
        compatibility_passed = False

    role_val_j0_em = endpoint_fresh_val["overall_j0_sequence_em"]
    role_conf_j0_em = endpoint_fresh_length10["j0_sequence_em"]

    hist_val_j0_em = hist_fresh_val["overall_j0_sequence_em"]
    hist_conf_j0_em = hist_fresh_conf["j0_sequence_em"]

    freeze_val_j0_em = freeze_fresh_val["overall_j0_sequence_em"]
    freeze_conf_j0_em = freeze_fresh_conf["j0_sequence_em"]

    val_delta_hist = role_val_j0_em - hist_val_j0_em
    val_delta_freeze = role_val_j0_em - freeze_val_j0_em
    conf_delta_hist = role_conf_j0_em - hist_conf_j0_em
    conf_delta_freeze = role_conf_j0_em - freeze_conf_j0_em

    val_plasticity_passed = bool(
        (val_delta_hist >= config.j0_delta_floor)
        and (val_delta_freeze >= config.hard_freeze_delta_floor)
    )
    conf_plasticity_passed = bool(
        (conf_delta_hist >= config.j0_delta_floor)
        and (conf_delta_freeze >= config.hard_freeze_delta_floor)
    )
    plasticity_passed = val_plasticity_passed and conf_plasticity_passed

    strong_functional_floor = bool(
        role_val_j0_em >= 0.95
        and endpoint_fresh_val["length_10_j0_sequence_em"] >= 0.95
        and role_conf_j0_em >= 0.95
        and compatibility_passed
    )

    if compatibility_passed and plasticity_passed:
        primary_decision = "KEY_VALUE_ROLE_SPLIT_RESTORES_STABILITY_PLASTICITY"
    elif compatibility_passed and not plasticity_passed:
        primary_decision = (
            "KEY_VALUE_ROLE_SPLIT_PRESERVES_STABILITY_BUT_KEY_PLASTICITY_INSUFFICIENT"
        )
    elif not compatibility_passed and plasticity_passed:
        primary_decision = "ROLE_SPLIT_LEARNING_WITH_COMPATIBILITY_FAILURE"
    else:
        primary_decision = "KEY_VALUE_ROLE_SPLIT_NOT_SUPPORTED"

    result_decision = {
        "task_id": REC004Z_TASK_ID,
        "primary_decision": primary_decision,
        "strong_functional_floor": strong_functional_floor,
        "compatibility_gate": {
            "passed": compatibility_passed,
            "threshold": config.oracle_em_threshold,
            "audit": compat_audit,
        },
        "plasticity_gate": {
            "passed": plasticity_passed,
            "historical_delta_floor": config.j0_delta_floor,
            "hard_freeze_delta_floor": config.hard_freeze_delta_floor,
            "fresh_val_delta_vs_historical": val_delta_hist,
            "fresh_val_delta_vs_hard_freeze": val_delta_freeze,
            "fresh_conf_delta_vs_historical": conf_delta_hist,
            "fresh_conf_delta_vs_hard_freeze": conf_delta_freeze,
        },
        "key_branch_audit_label": key_plasticity_audit["status"],
    }
    _write_json(output_dir / "result_decision.json", result_decision)

    resident_added = 7200
    active_params_per_call = sum(p.numel() for p in role_split_model.parameters())
    active_trainable_params = sum(
        p.numel() for p in role_split_model.parameters() if p.requires_grad
    )
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    cost_accounting = {
        "task_id": REC004Z_TASK_ID,
        "resident_parameters_added": resident_added,
        "active_parameters_per_call": active_params_per_call,
        "active_trainable_parameters": active_trainable_params,
        "training_duration_seconds": train_duration,
        "updates_executed": REC004Z_MAX_UPDATES,
        "peak_vram_bytes": peak_vram,
        "peak_vram_mb": peak_vram / (1024 * 1024),
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    side_effect_audit = {
        "task_id": REC004Z_TASK_ID,
        "canonical_bank_modified": False,
        "model_bundle_modified": False,
        "historical_checkpoints_modified": False,
        "cache_invariance_verified": True,
        "task_conditioned_leakage": False,
        "h_content_invariant_preserved": True,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    next_repair_contract = f"""# Next Repair Contract: After B-C005REC-004Z

## Primary Outcome
- **Decision:** {primary_decision}
- **Strong Functional Floor:** {strong_functional_floor}
- **Compatibility Gate:** {"PASS" if compatibility_passed else "FAIL"}
- **Plasticity Gate:** {"PASS" if plasticity_passed else "FAIL"}

## Interpretation & Next Task Branch
{"Propose REC-004AA: Cross-Init Role-Split Generalization Pilot across I01-I05." if primary_decision == "KEY_VALUE_ROLE_SPLIT_RESTORES_STABILITY_PLASTICITY" else "K/V content-prep isolation alone is insufficient to restore plasticity. Next step: explore compact task-blind plastic residual over stable base." if primary_decision == "KEY_VALUE_ROLE_SPLIT_PRESERVES_STABILITY_BUT_KEY_PLASTICITY_INSUFFICIENT" else "Investigate potential forward-graph isolation or downstream drift outside CVOF."}
"""  # noqa: E501 -- preserve historical generated Markdown verbatim
    (output_dir / "next_repair_contract.md").write_text(next_repair_contract, encoding="utf-8")

    summary = {
        "task_id": REC004Z_TASK_ID,
        "implementation_status": "COMPLETE",
        "legacy_forward_parity": stage_a_report["status"],
        "gradient_role_isolation": stage_b_report["status"],
        "intervention_optimizer_updates": REC004Z_MAX_UPDATES,
        "new_candidate_training_updates": 0,
        "architecture_pilot_result": primary_decision,
        "strong_functional_floor": strong_functional_floor,
        "key_branch_plasticity": key_plasticity_audit["status"],
        "fresh_normal_val_em": role_val_j0_em,
        "fresh_length10_conf_em": role_conf_j0_em,
        "selected_architecture": None,
        "selected_init": None,
        "selected_step": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }
    _write_json(output_dir / "summary.json", summary)

    report_md = f"""# Pilot Report — B-C005REC-004Z: I03 Key/Value Content-Prep Functional-Role Isolation

## Executive Summary
- **Result:** {primary_decision}
- **Legacy Forward Parity:** {stage_a_report["status"]} (max logit diff: {stage_a_report["final_logits_max_abs_diff"]:.2e})
- **Gradient Isolation:** {stage_b_report["status"]} (O1 key grad: {stage_b_report["o1_key_in_proj_grad_norm"]:.2e})
- **Compatibility Gate:** {"PASS" if compatibility_passed else "FAIL"}
- **Plasticity Gate:** {"PASS" if plasticity_passed else "FAIL"}

## Metrics Comparison
| Metric | Historical @8000 | Hard-Freeze @8000 (REC-004W) | Role-Split @8000 (REC-004Z) | Target Floor |
| --- | --- | --- | --- | --- |
| Fresh Normal J0 EM | {hist_val_j0_em:.4f} | {freeze_val_j0_em:.4f} | {role_val_j0_em:.4f} | >= {hist_val_j0_em + 0.10:.4f} / >= {freeze_val_j0_em + 0.05:.4f} |
| Fresh Length10 J0 EM | {hist_conf_j0_em:.4f} | {freeze_conf_j0_em:.4f} | {role_conf_j0_em:.4f} | >= {hist_conf_j0_em + 0.10:.4f} / >= {freeze_conf_j0_em + 0.05:.4f} |
| Fresh Length10 O1 EM | {hist_fresh_conf["o1_sequence_em"]:.4f} | {freeze_fresh_conf["o1_sequence_em"]:.4f} | {conf_o1_em:.4f} | >= 0.9500 |
| Fresh Length10 O1 Pos4 Acc | {hist_fresh_conf["o1_position4_acc"]:.4f} | {freeze_fresh_conf["o1_position4_acc"]:.4f} | {conf_p4_acc:.4f} | >= 0.9500 |

## Cost Accounting
- Resident parameters added: {resident_added}
- Peak VRAM: {peak_vram / (1024 * 1024):.2f} MB
- Training duration: {train_duration:.2f} s
"""  # noqa: E501 -- table rows retain their historical single-line output
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    print("\n=== Task B-C005REC-004Z Completed ===")
    print(f"Result Decision: {primary_decision}")
    print(f"Summary saved to {output_dir / 'summary.json'}")

    return summary
