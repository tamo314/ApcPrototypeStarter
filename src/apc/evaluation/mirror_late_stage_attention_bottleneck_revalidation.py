"""B-C005REC-004J: Late-Stage Attention Bottleneck Revalidation (I03) &
Collapse Contrast (I05).

Follows
`docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LATE_STAGE_ATTENTION_BOTTLENECK_REVALIDATION.md`.

`B-C005REC-004F` (ADR-0101) found that at step=6000, substituting the oracle
`pi_n` one-hot attention distribution for I03's own combined attention
recovered length-10 EM from 0.086 to 1.000. `B-C005REC-004I` (ADR-0104) later
found I03 never clears the 0.95 length-10 floor on a fully disjoint set at
ANY checkpoint from step=6000 through step=18000, with its own trajectory
peak at step=17500 (44/230). REC-004F's oracle-attention finding was never
re-checked at this much later step, 11500 updates on -- extrapolating a
step=6000 mechanism finding onto step=17500 is not licensed by anything
already measured. This task asks exactly one question: **at step=17500, is
I03's length-10 failure still explained by the attention distribution, or
has the mechanism shifted downstream since step=6000?**

**Zero new optimizer updates.** Forward-only: loads five already-saved
checkpoints (I03@17500, I03@18000, I04@18000, I05@17500, I05@18000, all from
REC-004H's own `run_001` tree), hash-verifies each against REC-004H's own
`learning_curve.jsonl`, then runs exactly two conditions -- J0 (normal
forward, via REC-004E's manual-reconstruction path) and O1 (REC-004F's
`_oracle_attention`/`run_oracle_forward`, imported unmodified) -- against two
paired datasets: REC-004I's own `clean_selection_validation_v2` length-10
subset (regenerated as a pure function, byte-identical to REC-004I's own
1024-example set), and a NEW disjoint `length10_mechanism_probe_v1` (512
examples, development-exposed, not a sealed/RG3 set). `selected_init`,
`selected_step`, and `child_bundle` stay `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`, unconditionally.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.evaluation.unified_oracle_causal_benchmark import _derive_local_seed
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004J_TASK_ID",
    "REC004J_SOURCE_TASK_IDS",
    "REC004J_TARGETS",
    "REC004J_TARGET_LENGTH",
    "REC004J_PROBE_SPLIT",
    "REC004J_PROBE_EXAMPLES",
    "REC004J_DECISIVE_INIT",
    "REC004J_DECISIVE_STEP",
    "REC004J_ORACLE_EM_THRESHOLD",
    "REC004J_LARGE_DELTA_THRESHOLD",
    "MirrorLateStageAttentionBottleneckRevalidationConfig",
    "run_mirror_late_stage_attention_bottleneck_revalidation_task",
]

# =============================================================================
# Constants -- reused from REC-004F/REC-004I wherever the recipe is
# unchanged, never retyped. Target checkpoints, thresholds, and dataset sizes
# are all fixed here, before any diagnostic point is computed.
# =============================================================================

REC004J_TASK_ID: Final = "B-C005REC-004J"
REC004J_SOURCE_TASK_IDS: Final[tuple[str, ...]] = ("B-C005REC-004F", "B-C005REC-004I")
REC004J_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LATE_STAGE_ATTENTION_BOTTLENECK_REVALIDATION.md"
)

REC004J_TARGET_OPERATION: Final = traj_audit.REC004I_TARGET_OPERATION  # "MIRROR_HALVES"
REC004J_ARM: Final = traj_audit.REC004I_ARM  # "P_LENGTH_POSITION_BIAS"
REC004J_VOCAB_SIZE: Final = traj_audit.REC004I_VOCAB_SIZE
REC004J_SEED: Final = RECOVERY_PILOT_SEED

# Fixed minimum checkpoint set (Section 3 of the task doc). Every one of
# these resolves to B-C005REC-004H's own run_001 tree via
# `traj_audit._source_for_step` (12500 <= step <= 18000).
REC004J_TARGETS: Final[tuple[tuple[str, int], ...]] = (
    ("I03", 17500),
    ("I03", 18000),
    ("I04", 18000),
    ("I05", 17500),
    ("I05", 18000),
)
REC004J_DECISIVE_INIT: Final = "I03"
REC004J_DECISIVE_STEP: Final = 17500
REC004J_TARGET_LENGTH: Final = 10

REC004J_CLEAN_V2_DATASET: Final = "clean_v2_length10"
REC004J_PROBE_SPLIT: Final = "length10_mechanism_probe_v1"
REC004J_PROBE_DATASET: Final = "length10_mechanism_probe_v1"
REC004J_PROBE_EXAMPLES: Final = 512

# Pre-registered decision thresholds (Section 6 of the task doc) -- fixed
# before any point below is computed, never widened after seeing a result.
REC004J_ORACLE_EM_THRESHOLD: Final = 0.95
REC004J_LARGE_DELTA_THRESHOLD: Final = 0.30

# Descriptive-only I05 contrast threshold (Section 7) -- never gates the
# I03 decision in Section 6.
REC004J_COLLAPSE_DROP_THRESHOLD: Final = 0.15

REC004J_CHUNK_SIZE: Final = 128


def _config_to_yaml_dict(
    config: MirrorLateStageAttentionBottleneckRevalidationConfig,
) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


@dataclass(frozen=True)
class MirrorLateStageAttentionBottleneckRevalidationConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004j/run_001")
    seed: int = RECOVERY_PILOT_SEED


# =============================================================================
# Data -- 4a: regenerate REC-004I's own clean_selection_validation_v2
# (a pure function of seed + protected set) and take its length-10 subset.
# 4b: a NEW length10_mechanism_probe_v1, disjoint from REC-004I's own
# protected registry PLUS clean_selection_validation_v2 itself.
# =============================================================================


def regenerate_clean_selection_validation_v2(
    seed: int,
) -> tuple[list[Example], list[Example], dict[str, Any], dict[str, int]]:
    """Reproduces REC-004I's `clean_selection_validation_v2` exactly (same
    seed, same protected-registry inputs, same collision-substitution
    procedure -- a pure function of its inputs). Returns
    `(all_1024_examples, length_10_subset, generation_detail,
    protected_source_counts)`."""
    protected, counts = traj_audit.build_protected_digest_registry(seed)
    examples, detail = traj_audit.build_clean_selection_validation_v2(seed, protected)
    length_10_subset = [ex for ex in examples if len(ex.input_tokens) == REC004J_TARGET_LENGTH]
    return examples, length_10_subset, detail, counts


def build_length10_mechanism_probe_v1(
    seed: int, protected_digests: set[str], n: int = REC004J_PROBE_EXAMPLES
) -> tuple[list[Example], dict[str, Any]]:
    """Deterministically generates `n` length-exactly-10 MIRROR_HALVES
    examples disjoint from `protected_digests`, using the SAME
    collision-substitution rule as `traj_audit.build_clean_selection_
    validation_v2`: on a collision, keep drawing the NEXT candidate from the
    SAME continuing RNG stream (never a fresh reseed, never conditioned on
    any checkpoint prediction). A pure function of `(seed,
    protected_digests)`."""
    seed_label = f"{REC004J_PROBE_SPLIT}:{REC004J_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed(seed, 0, seed_label))
    op_obj = get_operation(REC004J_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq = tuple(rng.randrange(REC004J_VOCAB_SIZE) for _ in range(REC004J_TARGET_LENGTH))
            params = op_obj.sample_params(rng, seq, REC004J_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004J_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004J_VOCAB_SIZE)
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
                split=REC004J_PROBE_SPLIT,
                vocab_size=REC004J_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004J_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "target_length": REC004J_TARGET_LENGTH,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            "length10_mechanism_probe_v1 is a development-exposed diagnostic set, "
            "not a sealed or final RG3 query set; once consumed by this task it is "
            "not eligible to serve as an independent RG3 recheck query without a "
            "fresh, disjoint regeneration."
        ),
    }
    return examples, detail


# =============================================================================
# Checkpoint loading -- all 5 targets resolve to REC-004H's tree; hash-
# verified against REC-004H's own learning_curve.jsonl before use.
# =============================================================================


def _load_and_verify_primitive(
    core: Any, init_id: str, step: int
) -> tuple[Any, dict[str, Any]]:
    path = traj_audit._checkpoint_path(init_id, step)
    source = traj_audit._source_for_step(step)
    if not path.is_file():
        return None, {
            "status": "SOURCE_ARTIFACT_UNAVAILABLE", "init_id": init_id, "step": step,
            "source_task": source, "path": str(path),
        }
    state = torch.load(path, map_location="cpu", weights_only=False)
    computed_hash = mb.canonical_state_hash(state)
    curve_rows = traj_audit._load_learning_curve_rows(source)
    old_row = next(
        (
            r for r in curve_rows
            if r["init_id"] == init_id and r["arm"] == REC004J_ARM and r["step"] == step
        ),
        None,
    )
    recorded_hash = old_row.get("checkpoint_state_hash") if old_row else None
    hash_matches = recorded_hash is not None and computed_hash == recorded_hash

    device = core.device
    primitive = mpbr._new_arm_primitive(core, REC004J_ARM)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)

    return primitive, {
        "status": "VERIFIED" if hash_matches else "SOURCE_REPLAY_MISMATCH",
        "init_id": init_id, "step": step, "source_task": source, "path": str(path),
        "raw_file_sha256": mb.raw_file_sha256(path),
        "checkpoint_state_hash": computed_hash,
        "recorded_checkpoint_state_hash": recorded_hash,
    }


# =============================================================================
# J0 (manual-reconstruction forward, verified in REC-004E to reproduce the
# real forward within 5e-3 logit tolerance / exact discrete predictions) and
# O1 (REC-004F's oracle pi_n substitution, imported unmodified).
# =============================================================================


def _run_j0_decomposition(
    primitive: Any,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, torch.Tensor]:
    resid_audit._assert_eval_only()
    query, kv, out_max, lmax, batch = resid_audit._prepare_query_kv(
        primitive, content_features, content_lengths, output_lengths
    )
    device = content_features.device
    pad = resid_audit._pad_mask(content_lengths, lmax, device)
    bias_pack = resid_audit._position_bias_raw(primitive, content_lengths, out_max, lmax, device)
    combined = torch.where(
        pad.unsqueeze(1).expand(batch, out_max, lmax),
        torch.finfo(bias_pack["b"].dtype).min,
        bias_pack["b"],
    )
    attn_out, attn_probs, s_other = resid_audit._manual_attention(
        primitive, query, kv, combined, s_other_scale=1.0
    )
    logits = resid_audit._post_attention(primitive, query, attn_out)
    return {"logits": logits, "attn_probs": attn_probs, "s_other": s_other, "b": bias_pack["b"]}


def _run_o1(
    primitive: Any,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, torch.Tensor]:
    return oracle_probe.run_oracle_forward(
        primitive, content_features, content_lengths, output_lengths
    )


# =============================================================================
# Per (checkpoint, dataset) diagnostic point.
# =============================================================================


def compute_diagnostic_point(core: Any, primitive: Any, examples: list[Example]) -> dict[str, Any]:
    """Runs J0 and O1 on every example (all length exactly
    `REC004J_TARGET_LENGTH`), chunked, and computes every metric listed in
    Section 6 of the task doc. Rank/margin/entropy/head-agreement are
    DESCRIPTIVE ONLY -- they never feed the decision rule (built separately
    in `build_i03_decision`)."""
    resid_audit._assert_eval_only()
    n = REC004J_TARGET_LENGTH
    device = core.device
    op = get_operation(REC004J_TARGET_OPERATION)
    pi = mpid.mirror_halves_position_map(n)

    j0_correct: list[bool] = []
    o1_correct: list[bool] = []
    j0_token_correct = 0
    o1_token_correct = 0
    token_total = 0
    j0_loss_sum = 0.0
    o1_loss_sum = 0.0
    pos_j0_correct = [0] * n
    pos_o1_correct = [0] * n
    ranks: list[int] = []
    margins: list[float] = []
    entropy_head_avg: list[float] = []
    entropy_per_head: list[list[float]] = []
    agree_count = 0
    agree_total = 0
    head_div_values: list[float] = []
    b_std_values: list[float] = []
    s_other_std_values: list[float] = []
    correlations: list[float] = []
    residual_wrong_example_count = 0
    residual_wrong_position_counts = [0] * n

    n_head: int | None = None

    for start in range(0, len(examples), REC004J_CHUNK_SIZE):
        chunk = examples[start : start + REC004J_CHUNK_SIZE]
        content_lengths = [n] * len(chunk)
        output_lengths = [op.output_length(n)] * len(chunk)
        assert all(ol == n for ol in output_lengths)
        batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
        with torch.no_grad():
            h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
            j0 = _run_j0_decomposition(primitive, h_content, content_lengths, output_lengths)
            o1 = _run_o1(primitive, h_content, content_lengths, output_lengths)
            labels = _labels_for_examples(chunk, output_lengths, n, device)
            j0_loss = F.cross_entropy(
                j0["logits"].reshape(-1, j0["logits"].size(-1)), labels.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            ).item()
            o1_loss = F.cross_entropy(
                o1["logits"].reshape(-1, o1["logits"].size(-1)), labels.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            ).item()
        j0_loss_sum += j0_loss
        o1_loss_sum += o1_loss

        j0_preds = resid_audit._predict_from_logits(j0["logits"], output_lengths)
        o1_preds = resid_audit._predict_from_logits(o1["logits"], output_lengths)

        for row, ex in enumerate(chunk):
            target = tuple(ex.target_tokens[:n])
            jp = tuple(j0_preds[row])
            op_ = tuple(o1_preds[row])
            j0_ok = jp == target
            o1_ok = op_ == target
            j0_correct.append(j0_ok)
            o1_correct.append(o1_ok)
            for k in range(n):
                if jp[k] == target[k]:
                    pos_j0_correct[k] += 1
                if op_[k] == target[k]:
                    pos_o1_correct[k] += 1
            j0_token_correct += sum(1 for k in range(n) if jp[k] == target[k])
            o1_token_correct += sum(1 for k in range(n) if op_[k] == target[k])
            if not o1_ok:
                residual_wrong_example_count += 1
                for k in range(n):
                    if op_[k] != target[k]:
                        residual_wrong_position_counts[k] += 1
        token_total += n * len(chunk)

        attn_probs_np = j0["attn_probs"].detach().cpu().numpy()  # [batch, n_head, n, n]
        if n_head is None:
            n_head = attn_probs_np.shape[1]
            entropy_per_head = [[] for _ in range(n_head)]
        head_avg = attn_probs_np.mean(axis=1)  # [batch, n, n]
        eps = 1e-12

        ent_avg = -(head_avg * np.log(head_avg + eps)).sum(axis=-1)  # [batch, n]
        entropy_head_avg.extend(ent_avg.reshape(-1).tolist())
        ent_ph = -(attn_probs_np * np.log(attn_probs_np + eps)).sum(axis=-1)  # [batch, n_head, n]
        for hh in range(n_head):
            entropy_per_head[hh].extend(ent_ph[:, hh, :].reshape(-1).tolist())

        argmax_heads = attn_probs_np.argmax(axis=-1)  # [batch, n_head, n]
        for b_i in range(argmax_heads.shape[0]):
            for pos in range(n):
                vals = argmax_heads[b_i, :, pos]
                agree_total += 1
                if np.all(vals == vals[0]):
                    agree_count += 1

        diffs = np.abs(attn_probs_np - head_avg[:, None, :, :]).sum(axis=-1)  # [batch, n_head, n]
        head_div_values.extend(diffs.mean(axis=1).reshape(-1).tolist())

        for b_i in range(head_avg.shape[0]):
            for pos in range(n):
                target_j = pi[pos]
                row_probs = head_avg[b_i, pos]
                order = np.argsort(-row_probs)
                rank = int(np.where(order == target_j)[0][0]) + 1
                ranks.append(rank)
                best_other = max(
                    (row_probs[j] for j in range(n) if j != target_j), default=float("nan")
                )
                margins.append(float(row_probs[target_j] - best_other))

        b_np = j0["b"].detach().cpu().numpy()  # [batch, n, n]
        s_other_np = j0["s_other"].detach().cpu().numpy()  # [batch, n_head, n, n]
        s_other_head_avg = s_other_np.mean(axis=1)
        for b_i in range(b_np.shape[0]):
            for pos in range(n):
                b_row = b_np[b_i, pos]
                s_row = s_other_head_avg[b_i, pos]
                b_std_values.append(float(b_row.std()))
                s_other_std_values.append(float(s_row.std()))
                if float(b_row.std()) > 0.0 and float(s_row.std()) > 0.0:
                    correlations.append(float(np.corrcoef(b_row, s_row)[0, 1]))

    n_total = len(examples)
    j0_em = sum(j0_correct) / n_total if n_total else None
    o1_em = sum(o1_correct) / n_total if n_total else None
    both_correct = sum(1 for a, bb in zip(j0_correct, o1_correct, strict=True) if a and bb)
    both_wrong = sum(1 for a, bb in zip(j0_correct, o1_correct, strict=True) if not a and not bb)
    j0_only = sum(1 for a, bb in zip(j0_correct, o1_correct, strict=True) if a and not bb)
    o1_only = sum(1 for a, bb in zip(j0_correct, o1_correct, strict=True) if bb and not a)

    return {
        "n": n_total,
        "j0_sequence_exact_match": j0_em,
        "oracle_sequence_exact_match": o1_em,
        "paired_delta_oracle_minus_j0": (
            (o1_em - j0_em) if (j0_em is not None and o1_em is not None) else None
        ),
        "j0_token_accuracy": j0_token_correct / token_total if token_total else None,
        "oracle_token_accuracy": o1_token_correct / token_total if token_total else None,
        "j0_mean_loss": j0_loss_sum / token_total if token_total else None,
        "oracle_mean_loss": o1_loss_sum / token_total if token_total else None,
        "j0_only_correct": j0_only,
        "oracle_only_correct": o1_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "per_output_position": {
            str(k): {
                "j0_accuracy": pos_j0_correct[k] / n_total if n_total else None,
                "oracle_accuracy": pos_o1_correct[k] / n_total if n_total else None,
            }
            for k in range(n)
        },
        "correct_key_rank_margin": {
            "note": "descriptive only -- never gates the decision rule",
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "mean_margin": float(np.mean(margins)) if margins else None,
        },
        "attention_entropy": {
            "j0_head_averaged_mean": (
                float(np.mean(entropy_head_avg)) if entropy_head_avg else None
            ),
            "j0_head_averaged_std": (
                float(np.std(entropy_head_avg)) if entropy_head_avg else None
            ),
            "j0_per_head_mean": [
                float(np.mean(vals)) if vals else None for vals in entropy_per_head
            ],
            "oracle_entropy": 0.0,
            "oracle_entropy_note": (
                "exactly 0 by construction of the one-hot substitution, not a finding"
            ),
        },
        "head_agreement": {
            "argmax_agreement_rate": agree_count / agree_total if agree_total else None,
            "mean_l1_divergence_from_head_mean": (
                float(np.mean(head_div_values)) if head_div_values else None
            ),
        },
        "s_other_and_bias": {
            "b_row_std_mean": float(np.mean(b_std_values)) if b_std_values else None,
            "s_other_row_std_mean": (
                float(np.mean(s_other_std_values)) if s_other_std_values else None
            ),
            "b_s_other_row_correlation_mean": (
                float(np.mean(correlations)) if correlations else None
            ),
        },
        "residual_after_oracle": {
            "wrong_example_count": residual_wrong_example_count,
            "wrong_fraction": (
                residual_wrong_example_count / n_total if n_total else None
            ),
            "by_position_wrong_count": {
                str(k): residual_wrong_position_counts[k] for k in range(n)
            },
        },
    }


# =============================================================================
# Orchestration across all 5 checkpoints x 2 datasets.
# =============================================================================


def run_all_diagnostic_points(
    core: Any,
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, int]
]:
    all_1024, clean_v2_len10, clean_v2_detail, protected_counts = (
        regenerate_clean_selection_validation_v2(REC004J_SEED)
    )
    protected_v1, _ = traj_audit.build_protected_digest_registry(REC004J_SEED)
    protected_v2 = protected_v1 | traj_audit._digest_examples(all_1024)
    probe_examples, probe_detail = build_length10_mechanism_probe_v1(REC004J_SEED, protected_v2)

    datasets: dict[str, list[Example]] = {
        REC004J_CLEAN_V2_DATASET: clean_v2_len10,
        REC004J_PROBE_DATASET: probe_examples,
    }

    points: dict[str, Any] = {}
    load_infos: dict[str, dict[str, Any]] = {}
    for init_id, step in REC004J_TARGETS:
        primitive, load_info = _load_and_verify_primitive(core, init_id, step)
        load_infos[f"{init_id}@{step}"] = load_info
        for dataset_name, examples in datasets.items():
            key = f"{init_id}@{step}:{dataset_name}"
            if primitive is None:
                points[key] = {
                    "status": "SOURCE_ARTIFACT_UNAVAILABLE", "init_id": init_id, "step": step,
                    "dataset": dataset_name,
                }
                continue
            with torch.no_grad():
                point = compute_diagnostic_point(core, primitive, examples)
            point.update(
                {
                    "status": load_info["status"],
                    "init_id": init_id, "step": step, "dataset": dataset_name,
                }
            )
            points[key] = point

    return points, load_infos, clean_v2_detail, probe_detail, protected_counts


def build_i03_decision(points: dict[str, Any]) -> dict[str, Any]:
    key_v2 = f"{REC004J_DECISIVE_INIT}@{REC004J_DECISIVE_STEP}:{REC004J_CLEAN_V2_DATASET}"
    key_probe = f"{REC004J_DECISIVE_INIT}@{REC004J_DECISIVE_STEP}:{REC004J_PROBE_DATASET}"
    row_v2 = points.get(key_v2, {})
    row_probe = points.get(key_probe, {})

    def _passes(row: dict[str, Any]) -> bool:
        em = row.get("oracle_sequence_exact_match")
        delta = row.get("paired_delta_oracle_minus_j0")
        return (
            em is not None
            and delta is not None
            and em >= REC004J_ORACLE_EM_THRESHOLD
            and delta >= REC004J_LARGE_DELTA_THRESHOLD
        )

    clean_v2_passes = _passes(row_v2)
    probe_passes = _passes(row_probe)
    both_pass = clean_v2_passes and probe_passes
    label = (
        "PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED"
        if both_pass
        else "LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM"
    )
    return {
        "task_id": REC004J_TASK_ID,
        "decisive_init": REC004J_DECISIVE_INIT,
        "decisive_step": REC004J_DECISIVE_STEP,
        "oracle_em_threshold": REC004J_ORACLE_EM_THRESHOLD,
        "large_delta_threshold": REC004J_LARGE_DELTA_THRESHOLD,
        "clean_v2_length10": row_v2,
        "length10_mechanism_probe_v1": row_probe,
        "clean_v2_passes": clean_v2_passes,
        "probe_passes": probe_passes,
        "label": label,
        "consequence_note": (
            "attention-score learning stabilization is the narrower next candidate "
            "(not authorized by this task)"
            if label == "PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED"
            else "value/residual/readout-side diagnosis should precede any "
            "attention-score change (not authorized by this task)"
        ),
    }


def build_i05_collapse_contrast(points: dict[str, Any]) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    for step in (17500, 18000):
        rows[step] = {
            ds: points.get(f"I05@{step}:{ds}", {})
            for ds in (REC004J_CLEAN_V2_DATASET, REC004J_PROBE_DATASET)
        }

    def _delta(ds: str, metric: str) -> float | None:
        a = rows[17500][ds].get(metric)
        b = rows[18000][ds].get(metric)
        return (b - a) if (a is not None and b is not None) else None

    deltas = {
        ds: {
            "j0_em_delta": _delta(ds, "j0_sequence_exact_match"),
            "oracle_em_delta": _delta(ds, "oracle_sequence_exact_match"),
            "j0_head_averaged_entropy_delta": None,
            "head_argmax_agreement_delta": None,
        }
        for ds in (REC004J_CLEAN_V2_DATASET, REC004J_PROBE_DATASET)
    }
    for ds in (REC004J_CLEAN_V2_DATASET, REC004J_PROBE_DATASET):
        a_ent = rows[17500][ds].get("attention_entropy", {}).get("j0_head_averaged_mean")
        b_ent = rows[18000][ds].get("attention_entropy", {}).get("j0_head_averaged_mean")
        deltas[ds]["j0_head_averaged_entropy_delta"] = (
            (b_ent - a_ent) if (a_ent is not None and b_ent is not None) else None
        )
        a_agree = rows[17500][ds].get("head_agreement", {}).get("argmax_agreement_rate")
        b_agree = rows[18000][ds].get("head_agreement", {}).get("argmax_agreement_rate")
        deltas[ds]["head_argmax_agreement_delta"] = (
            (b_agree - a_agree) if (a_agree is not None and b_agree is not None) else None
        )

    def _collapses(delta: float | None) -> bool:
        return delta is not None and delta <= -REC004J_COLLAPSE_DROP_THRESHOLD

    j0_collapses = any(_collapses(deltas[ds]["j0_em_delta"]) for ds in deltas)
    oracle_collapses = any(_collapses(deltas[ds]["oracle_em_delta"]) for ds in deltas)

    if not j0_collapses:
        classification = "NO_COLLAPSE_OBSERVED_ON_THIS_SET"
    elif oracle_collapses:
        classification = "COLLAPSE_PERSISTS_UNDER_ORACLE_ATTENTION_DOWNSTREAM_IMPLICATED"
    else:
        classification = "ORACLE_ATTENTION_RESCUES_COLLAPSE_ATTENTION_DISTRIBUTION_IMPLICATED"

    return {
        "task_id": REC004J_TASK_ID,
        "purpose_note": (
            "a contrast, not this task's primary purpose (Section 0/7 of the task doc) -- "
            "exists to check whether a future I03 fix would also perturb I05, not to "
            "explain I05's collapse"
        ),
        "collapse_drop_threshold": REC004J_COLLAPSE_DROP_THRESHOLD,
        "rows": rows,
        "deltas_17500_to_18000": deltas,
        "j0_collapses": j0_collapses,
        "oracle_collapses": oracle_collapses,
        "classification": classification,
    }


def build_report_markdown(
    points: dict[str, Any],
    i03_decision: dict[str, Any],
    i05_contrast: dict[str, Any],
    clean_v2_source_replay: dict[str, Any],
    probe_detail: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# {REC004J_TASK_ID}: Late-Stage Attention Bottleneck Revalidation (I03) & "
        "Collapse Contrast (I05)",
        "",
        "Zero new optimizer updates. Forward-only (J0, O1) on 5 already-saved "
        "checkpoints x 2 length-10-only datasets.",
        "",
        "## Data",
        (
            "- `clean_selection_validation_v2` length-10 subset: "
            f"n={clean_v2_source_replay.get('length_10_subset_size')}, "
            f"reproduces REC-004I's manifest: "
            f"{clean_v2_source_replay.get('matches_rec004i_manifest')}"
        ),
        (
            f"- `length10_mechanism_probe_v1`: n={probe_detail.get('n')}, "
            f"substitutions={probe_detail.get('substitution_count')}, "
            f"development_exposed={probe_detail.get('development_exposed')}"
        ),
        "",
        "## Per-(checkpoint, dataset) J0 vs O1",
        "",
        "| init@step | dataset | n | J0 EM | O1 EM | delta | entropy(J0) | head agree |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for init_id, step in REC004J_TARGETS:
        for dataset_name in (REC004J_CLEAN_V2_DATASET, REC004J_PROBE_DATASET):
            row = points.get(f"{init_id}@{step}:{dataset_name}", {})
            if row.get("status") != "VERIFIED":
                status = row.get("status")
                lines.append(f"| {init_id}@{step} | {dataset_name} | - | {status} | | | | |")
                continue
            j0_em = row["j0_sequence_exact_match"]
            o1_em = row["oracle_sequence_exact_match"]
            delta = row["paired_delta_oracle_minus_j0"]
            ent = row["attention_entropy"]["j0_head_averaged_mean"]
            agree = row["head_agreement"]["argmax_agreement_rate"]
            lines.append(
                f"| {init_id}@{step} | {dataset_name} | {row['n']} | "
                f"{j0_em:.4f} | {o1_em:.4f} | {delta:+.4f} | {ent:.4f} | {agree:.4f} |"
            )
    lines += [
        "",
        "## I03@17500 decision",
        f"- label: **{i03_decision['label']}**",
        (
            f"- clean_v2 passes ({REC004J_ORACLE_EM_THRESHOLD} EM / "
            f"{REC004J_LARGE_DELTA_THRESHOLD} delta gate): {i03_decision['clean_v2_passes']}"
        ),
        f"- probe passes: {i03_decision['probe_passes']}",
        f"- consequence note: {i03_decision['consequence_note']}",
        "",
        "## I05 collapse contrast (17500 -> 18000)",
        f"- classification: **{i05_contrast['classification']}**",
        (
            f"- j0_collapses: {i05_contrast['j0_collapses']}, "
            f"oracle_collapses: {i05_contrast['oracle_collapses']}"
        ),
        f"- purpose note: {i05_contrast['purpose_note']}",
        "",
        "## Fixed non-adoption fields",
        "selected_init=null, selected_step=null, child_bundle=null, "
        "rg3_recheck=NOT_EXECUTED, new_optimizer_updates=0",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Full task orchestration.
# =============================================================================


def run_mirror_late_stage_attention_bottleneck_revalidation_task(
    config: MirrorLateStageAttentionBottleneckRevalidationConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_late_stage_attention_bottleneck_revalidation_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != REC004J_SEED:
        raise ValueError(
            f"B-C005REC-004J reads REC-004H artifacts pre-registered under seed "
            f"{REC004J_SEED}; got seed={seed}"
        )

    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    with torch.no_grad():
        points, load_infos, clean_v2_detail, probe_detail, protected_counts = (
            run_all_diagnostic_points(core)
        )

    clean_v2_source_replay = {
        "task_id": REC004J_TASK_ID,
        "reproduced_n": clean_v2_detail["n"],
        "reproduced_substitution_count": clean_v2_detail["substitution_count"],
        "reproduced_total_candidate_draws": clean_v2_detail["total_candidate_draws"],
        "recorded_n": 1024,
        "recorded_substitution_count": 32,
        "recorded_total_candidate_draws": 1056,
        "matches_rec004i_manifest": (
            clean_v2_detail["n"] == 1024
            and clean_v2_detail["substitution_count"] == 32
            and clean_v2_detail["total_candidate_draws"] == 1056
        ),
        "length_10_subset_size": next(
            (
                points[k]["n"] for k in points
                if k.endswith(f":{REC004J_CLEAN_V2_DATASET}") and "n" in points[k]
            ),
            None,
        ),
    }
    _write_json(output_dir / "clean_v2_source_replay.json", clean_v2_source_replay)
    _write_json(output_dir / "length10_mechanism_probe_v1_manifest.json", probe_detail)
    _write_json(output_dir / "checkpoint_load_info.json", load_infos)

    with (output_dir / "diagnostic_points.jsonl").open("w", encoding="utf-8") as fh:
        for key, row in points.items():
            fh.write(json.dumps({"key": key, **row}, default=str) + "\n")
    _write_json(output_dir / "diagnostic_points_summary.json", points)

    i03_decision = build_i03_decision(points)
    _write_json(output_dir / "i03_17500_decision.json", i03_decision)

    i05_contrast = build_i05_collapse_contrast(points)
    _write_json(output_dir / "i05_collapse_contrast.json", i05_contrast)

    protected_hashes_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004J_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == core_hash_before,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "checkpoint_load_status": {k: v["status"] for k, v in load_infos.items()},
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004J_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    n_predictions = sum(row.get("n", 0) for row in points.values() if isinstance(row, dict)) * 2
    cost_accounting = {
        "task_id": REC004J_TASK_ID,
        "new_optimizer_updates": 0,
        "n_forward_predictions": n_predictions,
        "protected_digest_source_counts": protected_counts,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=seed)
    _write_json(output_dir / "system.json", system_info)

    protocol = {
        "task_id": REC004J_TASK_ID,
        "source_task_ids": list(REC004J_SOURCE_TASK_IDS),
        "contract_file": str(REC004J_CONTRACT_FILE),
        "targets": [{"init_id": i, "step": s} for i, s in REC004J_TARGETS],
        "decisive_init": REC004J_DECISIVE_INIT,
        "decisive_step": REC004J_DECISIVE_STEP,
        "oracle_em_threshold": REC004J_ORACLE_EM_THRESHOLD,
        "large_delta_threshold": REC004J_LARGE_DELTA_THRESHOLD,
        "forbidden": [
            "new optimizer updates", "any J-condition other than J0/O1",
            "candidate adoption", "child assembly", "RG3 recheck",
            "any further repair auto-started from this probe's result",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)

    all_hash_verified = all(
        v.get("status") == "VERIFIED" for v in load_infos.values()
    )
    all_points_computed = all(
        row.get("status") == "VERIFIED" for row in points.values() if isinstance(row, dict)
    )

    result = {
        "implementation_status": "COMPLETE" if all_points_computed else "PARTIAL",
        "checkpoint_source_replay_status": (
            "VERIFIED" if all_hash_verified else "SOURCE_REPLAY_MISMATCH"
        ),
        "clean_v2_source_replay_status": (
            "VERIFIED" if clean_v2_source_replay["matches_rec004i_manifest"] else "MISMATCH"
        ),
        "i03_17500_decision": i03_decision["label"],
        "i05_collapse_classification": i05_contrast["classification"],
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": cost_accounting,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "summary.json", result)
    report_md = build_report_markdown(
        points, i03_decision, i05_contrast, clean_v2_source_replay, probe_detail
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    return result
