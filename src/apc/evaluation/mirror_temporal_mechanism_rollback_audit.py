"""B-C005REC-004K: I03 Matched-Data Temporal Mechanism Recheck & Same-Init
Downstream Rollback Audit.

Follows
`docs/CODEX_TASKS_PHASE_B_B2_MIRROR_TEMPORAL_MECHANISM_ROLLBACK_AUDIT.md`.

`B-C005REC-004F` (ADR-0101) found oracle `pi_n` attention substitution fully
recovers I03's length-10 EM at step=6000 (0.086 -> 1.000), but on the OLD
length-balanced diagnostic suite. `B-C005REC-004J` (ADR-0105) found the SAME
substitution recovers only ~0.75 EM for I03 at step=17500, on two newer,
digest-verified-disjoint datasets. Those two findings differ in BOTH
checkpoint step AND dataset, so REC-004J's finding alone cannot distinguish
"the mechanism shifted downstream between step=6000 and step=17500" from
"the two measurements simply used different data, and step=6000 would
already have scored ~0.75 on the newer datasets."

**Stage A (this module's `build_stage_a_temporal_decision`)** re-runs I03 at
step=6000 on REC-004J's own two datasets (byte-identically regenerated via
`rec004j.regenerate_clean_selection_validation_v2`/`build_length10_
mechanism_probe_v1`, imported unmodified). Only if step=6000 clears the 0.95
oracle-EM floor on BOTH datasets (`TEMPORAL_ORACLE_SUFFICIENCY_LOSS_
CONFIRMED`) does **Stage B/C/D** run: a forward-graph-derived, empirically-
verified 5-way parameter grouping of `CrossPositionLengthBiasPrimitive`, and
a same-init (I03-only) rollback of ONE group at a time from step=17500's
real, hash-verified checkpoint to step=6000's, evaluated under O1
(oracle-attention) with output position 4 (REC-004J's localized residual)
as the decisive metric. `R_ALL` (every group rolled back) is a positive
control: since it is the only rollback that must reproduce I03@6000's own
real O1 forward, an exact tolerance-checked parity match validates the
5-group decomposition itself before any single-group result is trusted.

**Zero new optimizer updates.** Loads three already-saved I03 checkpoints
(step=6000 from `B-C005REC-004D`'s tree, step=17500/18000 from `B-C005REC-
004H`'s), hash-verifies each, and builds every rollback variant as a fresh,
in-memory, evaluation-only `CrossPositionLengthBiasPrimitive` merged from
two real state dicts -- never a new checkpoint file, never a mutation of
either loaded primitive in place. `selected_init`, `selected_step`,
`selected_component`, and `child_bundle` stay `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`; `rec005_eligible` stays `false`, unconditionally.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import yaml

from apc.core.data import collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004K_TASK_ID",
    "REC004K_SOURCE_TASK_IDS",
    "REC004K_DECISIVE_INIT",
    "REC004K_EARLY_STEP",
    "REC004K_LATE_STEP",
    "REC004K_TERMINAL_STEP",
    "REC004K_TEMPORAL_STEPS",
    "REC004K_TARGET_LENGTH",
    "REC004K_ORACLE_EM_THRESHOLD",
    "REC004K_COMPONENT_IDS",
    "REC004K_ROLLBACK_IDS",
    "REC004K_POSITION_MAIN_RESIDUAL",
    "REC004K_POSITION_SYMMETRIC_CONTROL",
    "MirrorTemporalMechanismRollbackAuditConfig",
    "run_mirror_temporal_mechanism_rollback_audit_task",
]

# =============================================================================
# Constants. Checkpoint identity, dataset identity, and the 0.95 oracle-EM
# floor are all inherited from REC-004J (never retyped); the rollback
# component grouping and its thresholds are new here, pre-registered before
# any point below is computed.
# =============================================================================

REC004K_TASK_ID: Final = "B-C005REC-004K"
REC004K_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004E", "B-C005REC-004F", "B-C005REC-004I", "B-C005REC-004J",
)
REC004K_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_MIRROR_TEMPORAL_MECHANISM_ROLLBACK_AUDIT.md"
)

REC004K_TARGET_OPERATION: Final = rec004j.REC004J_TARGET_OPERATION  # "MIRROR_HALVES"
REC004K_ARM: Final = rec004j.REC004J_ARM  # "P_LENGTH_POSITION_BIAS"
REC004K_TARGET_LENGTH: Final = rec004j.REC004J_TARGET_LENGTH  # 10
REC004K_SEED: Final = RECOVERY_PILOT_SEED

REC004K_DECISIVE_INIT: Final = "I03"
REC004K_EARLY_STEP: Final = 6000
REC004K_LATE_STEP: Final = 17500
REC004K_TERMINAL_STEP: Final = 18000
REC004K_TEMPORAL_STEPS: Final[tuple[int, ...]] = (
    REC004K_EARLY_STEP, REC004K_LATE_STEP, REC004K_TERMINAL_STEP,
)

REC004K_CLEAN_V2_DATASET: Final = rec004j.REC004J_CLEAN_V2_DATASET
REC004K_PROBE_DATASET: Final = rec004j.REC004J_PROBE_DATASET
REC004K_DATASETS: Final[tuple[str, ...]] = (REC004K_CLEAN_V2_DATASET, REC004K_PROBE_DATASET)

# Stage A gate: reused verbatim from REC-004J -- not re-derived.
REC004K_ORACLE_EM_THRESHOLD: Final = rec004j.REC004J_ORACLE_EM_THRESHOLD  # 0.95

# Stage D decisive positions (REC-004J/ADR-0105's own localization): position
# 4's source under `mirror_halves_position_map(10)` is 0 (leftmost content
# token); position 5's source is 9 (rightmost) -- the two positions
# straddling MIRROR_HALVES's length-10 pivot.
REC004K_POSITION_MAIN_RESIDUAL: Final = 4
REC004K_POSITION_SYMMETRIC_CONTROL: Final = 5

# Stage C/D thresholds -- fixed before any rollback point is computed.
REC004K_ROLLBACK_EM_THRESHOLD: Final = 0.95
REC004K_ROLLBACK_POSITION_ACC_THRESHOLD: Final = 0.95
REC004K_PARITY_LOGIT_ABS_TOL: Final = 5e-3  # same tolerance convention as REC-004E

REC004K_CHUNK_SIZE: Final = 128
REC004K_INVARIANCE_CHECK_EXAMPLES: Final = 64
REC004K_PERTURBATION_SCALE: Final = 5.0

REC004K_REC004J_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004j/run_001")
REC004K_CROSS_CHECK_TOL: Final = 1e-9

# Stage B: five parameter groups, assigned by real `state_dict()` key,
# derived by reading `_oracle_attention`/`run_oracle_forward`
# (REC-004F) -- see the task doc Section 4 for the forward-graph trace this
# grouping is read from.
REC004K_COMPONENT_IDS: Final[tuple[str, ...]] = (
    "VALUE_OUTPROJ", "QUERY_RESIDUAL_PATH", "POST_ATTN_NORM", "FFN_BLOCK", "READOUT",
)
REC004K_COMPONENT_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "VALUE_OUTPROJ": (
        "content_in_proj.", "content_position_embedding.",
        "cross_attn.in_proj_weight", "cross_attn.in_proj_bias", "cross_attn.out_proj.",
    ),
    "QUERY_RESIDUAL_PATH": ("answer_query_embedding.",),
    "POST_ATTN_NORM": ("attn_norm.",),
    "FFN_BLOCK": ("ffn.", "ffn_norm."),
    "READOUT": ("readout.",),
}
# Feeds only the attention SCORE (S_other, b), which O1 discards entirely in
# favor of the oracle one-hot map -- provably inert for O1's output, verified
# empirically by `verify_o1_invariance_to_excluded_params` before Stage C
# trusts this partition.
REC004K_EXCLUDED_FROM_O1_PREFIXES: Final[tuple[str, ...]] = (
    "position_bias_hidden.", "position_bias_out.",
)

REC004K_ROLLBACK_IDS: Final[tuple[str, ...]] = ("R0", "R1", "R2", "R3", "R4", "R5", "R_ALL")
REC004K_ROLLBACK_TO_COMPONENT: Final[dict[str, tuple[str, ...]]] = {
    "R0": (),
    "R1": ("VALUE_OUTPROJ",),
    "R2": ("QUERY_RESIDUAL_PATH",),
    "R3": ("POST_ATTN_NORM",),
    "R4": ("FFN_BLOCK",),
    "R5": ("READOUT",),
    "R_ALL": REC004K_COMPONENT_IDS,
}


def _config_to_yaml_dict(config: MirrorTemporalMechanismRollbackAuditConfig) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


@dataclass(frozen=True)
class MirrorTemporalMechanismRollbackAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004k/run_001")
    seed: int = RECOVERY_PILOT_SEED


# =============================================================================
# Stage A -- matched-data temporal recheck. Reuses REC-004J's own dataset
# generation, checkpoint loading, and metrics computation UNMODIFIED.
# =============================================================================


def build_stage_a_datasets(
    seed: int,
) -> tuple[dict[str, list[Any]], dict[str, Any], dict[str, Any]]:
    """Byte-identical reuse of REC-004J's own two datasets -- this task does
    not define, resample, or reseed a third dataset."""
    all_1024, clean_v2_len10, clean_v2_detail, _counts = (
        rec004j.regenerate_clean_selection_validation_v2(seed)
    )
    protected_v1, _ = traj_audit.build_protected_digest_registry(seed)
    protected_v2 = protected_v1 | traj_audit._digest_examples(all_1024)
    probe_examples, probe_detail = rec004j.build_length10_mechanism_probe_v1(seed, protected_v2)
    datasets = {
        REC004K_CLEAN_V2_DATASET: clean_v2_len10,
        REC004K_PROBE_DATASET: probe_examples,
    }
    return datasets, clean_v2_detail, probe_detail


def run_stage_a_temporal_recheck(
    core: Any, datasets: dict[str, list[Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[int, Any]]:
    """Runs I03 @ {6000, 17500, 18000} x both datasets, via REC-004J's own
    `_load_and_verify_primitive`/`compute_diagnostic_point`. Returns
    `(points, load_infos, primitives_by_step)` -- the loaded step=6000/17500
    primitives are handed to Stage C so they are loaded exactly once."""
    points: dict[str, Any] = {}
    load_infos: dict[str, Any] = {}
    primitives_by_step: dict[int, Any] = {}
    for step in REC004K_TEMPORAL_STEPS:
        primitive, load_info = rec004j._load_and_verify_primitive(
            core, REC004K_DECISIVE_INIT, step
        )
        load_infos[str(step)] = load_info
        primitives_by_step[step] = primitive
        for dataset_name, examples in datasets.items():
            key = f"{step}:{dataset_name}"
            if primitive is None:
                points[key] = {
                    "status": "SOURCE_ARTIFACT_UNAVAILABLE",
                    "step": step, "dataset": dataset_name,
                }
                continue
            with torch.no_grad():
                point = rec004j.compute_diagnostic_point(core, primitive, examples)
            point.update({"status": load_info["status"], "step": step, "dataset": dataset_name})
            points[key] = point
    return points, load_infos, primitives_by_step


def cross_check_step17500_against_rec004j(stage_a_points: dict[str, Any]) -> dict[str, Any]:
    """Both this task and REC-004J run the identical function
    (`compute_diagnostic_point`) on the identical checkpoint and identical
    datasets at step=17500 -- their oracle EM must match exactly. Informational
    only (never gates Stage A's decision); reports `REC004J_ARTIFACT_
    UNAVAILABLE` rather than failing if REC-004J's run directory is absent."""
    path = REC004K_REC004J_RUN_DIR / "diagnostic_points_summary.json"
    if not path.is_file():
        return {"status": "REC004J_ARTIFACT_UNAVAILABLE"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    for dataset_name in REC004K_DATASETS:
        rec_key = f"{REC004K_DECISIVE_INIT}@{REC004K_LATE_STEP}:{dataset_name}"
        recorded_row = recorded.get(rec_key)
        recomputed_row = stage_a_points.get(f"{REC004K_LATE_STEP}:{dataset_name}", {})
        recorded_em = recorded_row.get("oracle_sequence_exact_match") if recorded_row else None
        recomputed_em = recomputed_row.get("oracle_sequence_exact_match")
        matches = (
            recorded_em is not None
            and recomputed_em is not None
            and abs(recorded_em - recomputed_em) < REC004K_CROSS_CHECK_TOL
        )
        if not matches:
            mismatches.append(
                {"dataset": dataset_name, "recorded": recorded_em, "recomputed": recomputed_em}
            )
    return {"status": "VERIFIED" if not mismatches else "MISMATCH", "mismatches": mismatches}


def build_stage_a_temporal_decision(stage_a_points: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered gate (task doc Section 3): Stage B/C/D run ONLY if
    step=6000 clears the oracle-EM floor on BOTH datasets while step=17500
    does not on at least one (already known from REC-004J to be true on
    both)."""

    def _oracle_em(step: int, dataset: str) -> float | None:
        row = stage_a_points.get(f"{step}:{dataset}", {})
        if row.get("status") != "VERIFIED":
            return None
        return row.get("oracle_sequence_exact_match")

    def _passes_both(step: int) -> bool:
        ems = [_oracle_em(step, ds) for ds in REC004K_DATASETS]
        return all(em is not None and em >= REC004K_ORACLE_EM_THRESHOLD for em in ems)

    early_passes = _passes_both(REC004K_EARLY_STEP)
    late_passes = _passes_both(REC004K_LATE_STEP)
    label = (
        "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED"
        if (early_passes and not late_passes)
        else "TEMPORAL_SHIFT_NOT_ESTABLISHED"
    )
    return {
        "task_id": REC004K_TASK_ID,
        "early_step": REC004K_EARLY_STEP,
        "late_step": REC004K_LATE_STEP,
        "oracle_em_threshold": REC004K_ORACLE_EM_THRESHOLD,
        "early_step_oracle_em": {ds: _oracle_em(REC004K_EARLY_STEP, ds) for ds in REC004K_DATASETS},
        "late_step_oracle_em": {ds: _oracle_em(REC004K_LATE_STEP, ds) for ds in REC004K_DATASETS},
        "early_step_passes_both_datasets": early_passes,
        "late_step_passes_both_datasets": late_passes,
        "label": label,
        "authorizes_component_rollback": label == "TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED",
    }


# =============================================================================
# Stage B -- forward-graph parameter grouping, assigned by real state_dict
# key, self-verified exhaustive and non-overlapping over the ACTUAL keys of
# the loaded checkpoint (never assumed from a static list alone).
# =============================================================================


def _key_matches(key: str, prefix: str) -> bool:
    return key.startswith(prefix) if prefix.endswith(".") else key == prefix


def partition_state_dict_keys(keys: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {cid: [] for cid in REC004K_COMPONENT_IDS}
    excluded: list[str] = []
    for key in keys:
        component_matches = [
            cid
            for cid, prefixes in REC004K_COMPONENT_PREFIXES.items()
            if any(_key_matches(key, p) for p in prefixes)
        ]
        excluded_match = any(_key_matches(key, p) for p in REC004K_EXCLUDED_FROM_O1_PREFIXES)
        total_matches = len(component_matches) + (1 if excluded_match else 0)
        if total_matches != 1:
            raise AssertionError(
                f"state_dict key {key!r} matched {total_matches} groups "
                f"(component_matches={component_matches}, excluded_match={excluded_match}); "
                "REC004K_COMPONENT_PREFIXES/REC004K_EXCLUDED_FROM_O1_PREFIXES must "
                "partition every real key exactly once"
            )
        if excluded_match:
            excluded.append(key)
        else:
            groups[component_matches[0]].append(key)
    return {**groups, "EXCLUDED_FROM_O1": excluded}


def verify_o1_invariance_to_excluded_params(
    primitive: Any,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, Any]:
    """Empirical proof (task doc Section 5), not a static-code assumption:
    perturbs `position_bias_hidden`/`position_bias_out` and the Q/K slices of
    `cross_attn.in_proj_weight`/`in_proj_bias` in place, confirms O1's output
    is EXACTLY unchanged, then restores the original values exactly. Mutates
    `primitive` only transiently -- byte-identical before and after."""
    resid_audit._assert_eval_only()
    with torch.no_grad():
        baseline = oracle_probe.run_oracle_forward(
            primitive, content_features, content_lengths, output_lengths
        )
        baseline_logits = baseline["logits"].clone()

        original_hidden_w = primitive.position_bias_hidden.weight.detach().clone()
        original_hidden_b = primitive.position_bias_hidden.bias.detach().clone()
        original_out_w = primitive.position_bias_out.weight.detach().clone()
        primitive.position_bias_hidden.weight.add_(
            torch.randn_like(primitive.position_bias_hidden.weight) * REC004K_PERTURBATION_SCALE
        )
        primitive.position_bias_hidden.bias.add_(
            torch.randn_like(primitive.position_bias_hidden.bias) * REC004K_PERTURBATION_SCALE
        )
        primitive.position_bias_out.weight.add_(
            torch.randn_like(primitive.position_bias_out.weight) * REC004K_PERTURBATION_SCALE
        )

        wq, wk, _wv = primitive.cross_attn.in_proj_weight.chunk(3, dim=0)
        original_wq = wq.detach().clone()
        original_wk = wk.detach().clone()
        wq.add_(torch.randn_like(wq) * REC004K_PERTURBATION_SCALE)
        wk.add_(torch.randn_like(wk) * REC004K_PERTURBATION_SCALE)
        has_bias = primitive.cross_attn.in_proj_bias is not None
        if has_bias:
            bq, bk, _bv = primitive.cross_attn.in_proj_bias.chunk(3, dim=0)
            original_bq = bq.detach().clone()
            original_bk = bk.detach().clone()
            bq.add_(torch.randn_like(bq) * REC004K_PERTURBATION_SCALE)
            bk.add_(torch.randn_like(bk) * REC004K_PERTURBATION_SCALE)

        perturbed = oracle_probe.run_oracle_forward(
            primitive, content_features, content_lengths, output_lengths
        )
        perturbed_logits = perturbed["logits"].clone()

        primitive.position_bias_hidden.weight.copy_(original_hidden_w)
        primitive.position_bias_hidden.bias.copy_(original_hidden_b)
        primitive.position_bias_out.weight.copy_(original_out_w)
        wq.copy_(original_wq)
        wk.copy_(original_wk)
        if has_bias:
            bq.copy_(original_bq)
            bk.copy_(original_bk)

        restored = oracle_probe.run_oracle_forward(
            primitive, content_features, content_lengths, output_lengths
        )
        restored_logits = restored["logits"].clone()

    max_abs_diff = (perturbed_logits - baseline_logits).abs().max().item()
    restore_diff = (restored_logits - baseline_logits).abs().max().item()
    perturbed_params = [
        "position_bias_hidden.weight", "position_bias_hidden.bias", "position_bias_out.weight",
        "cross_attn.in_proj_weight[Q]", "cross_attn.in_proj_weight[K]",
    ]
    if has_bias:
        perturbed_params += ["cross_attn.in_proj_bias[Q]", "cross_attn.in_proj_bias[K]"]
    return {
        "task_id": REC004K_TASK_ID,
        "perturbed_params": perturbed_params,
        "perturbation_scale": REC004K_PERTURBATION_SCALE,
        "max_abs_logit_diff_after_perturbation": max_abs_diff,
        "invariant": max_abs_diff == 0.0,
        "restore_check_max_abs_diff": restore_diff,
        "weights_restored_exactly": restore_diff == 0.0,
    }


# =============================================================================
# Stage C -- same-init component rollback. Every variant is a fresh,
# in-memory `CrossPositionLengthBiasPrimitive`, merged from two real,
# hash-verified state dicts; the originally loaded primitives are never
# mutated in place, and no `.pt` file is ever written.
# =============================================================================


def _build_rollback_primitive(
    core: Any,
    late_state_dict: dict[str, torch.Tensor],
    early_state_dict: dict[str, torch.Tensor],
    component_ids: tuple[str, ...],
    key_groups: dict[str, list[str]],
) -> Any:
    merged = {k: v.clone() for k, v in late_state_dict.items()}
    for cid in component_ids:
        for key in key_groups[cid]:
            merged[key] = early_state_dict[key].clone()
    primitive = mpbr._new_arm_primitive(core, REC004K_ARM)
    primitive.to(core.device)
    primitive.load_state_dict({k: v.to(core.device) for k, v in merged.items()}, strict=True)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def _position_margins_batch(
    logits: torch.Tensor, target_tokens: list[list[int]], n: int
) -> list[list[float]]:
    """Per-position mean target-token logit margin ingredients: for each
    example and output position, the readout logit for the correct token
    minus the best competing (incorrect) logit -- the quantity that changes
    right before a discrete prediction flips (task doc Section 7)."""
    logits_np = logits.detach().cpu().numpy()  # [batch, out_max, vocab]
    margins: list[list[float]] = [[] for _ in range(n)]
    for row in range(logits_np.shape[0]):
        for k in range(n):
            target = target_tokens[row][k]
            row_logits = logits_np[row, k].copy()
            target_logit = float(row_logits[target])
            row_logits[target] = -np.inf
            best_other = float(row_logits.max())
            margins[k].append(target_logit - best_other)
    return margins


def compute_rollback_diagnostic_point(
    core: Any, primitive: Any, examples: list[Any]
) -> dict[str, Any]:
    """Stage D's metric set: sequence EM, token accuracy, per-output-position
    (0-9) accuracy AND mean target-token logit margin, under both J0 and O1.
    Leaner than REC-004J's `compute_diagnostic_point` -- attention-
    distribution descriptive stats (entropy/head-agreement/S_other/bias) are
    not needed here: O1's attention is fixed (oracle) by construction across
    every rollback variant, and J0's attention pattern is unaffected by every
    rollback group except `VALUE_OUTPROJ` (whose fused Q/K/V tensor also
    shifts J0's own S_other -- see the task doc Section 7 note)."""
    resid_audit._assert_eval_only()
    n = REC004K_TARGET_LENGTH
    device = core.device
    op = get_operation(REC004K_TARGET_OPERATION)

    j0_correct: list[bool] = []
    o1_correct: list[bool] = []
    pos_j0_correct = [0] * n
    pos_o1_correct = [0] * n
    j0_token_correct = 0
    o1_token_correct = 0
    token_total = 0
    j0_margins: list[list[float]] = [[] for _ in range(n)]
    o1_margins: list[list[float]] = [[] for _ in range(n)]

    for start in range(0, len(examples), REC004K_CHUNK_SIZE):
        chunk = examples[start : start + REC004K_CHUNK_SIZE]
        content_lengths = [n] * len(chunk)
        output_lengths = [op.output_length(n)] * len(chunk)
        assert all(ol == n for ol in output_lengths)
        batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
        with torch.no_grad():
            h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
            j0 = rec004j._run_j0_decomposition(
                primitive, h_content, content_lengths, output_lengths
            )
            o1 = rec004j._run_o1(primitive, h_content, content_lengths, output_lengths)

        j0_preds = resid_audit._predict_from_logits(j0["logits"], output_lengths)
        o1_preds = resid_audit._predict_from_logits(o1["logits"], output_lengths)
        target_tokens = [list(ex.target_tokens[:n]) for ex in chunk]
        chunk_j0_margins = _position_margins_batch(j0["logits"], target_tokens, n)
        chunk_o1_margins = _position_margins_batch(o1["logits"], target_tokens, n)
        for k in range(n):
            j0_margins[k].extend(chunk_j0_margins[k])
            o1_margins[k].extend(chunk_o1_margins[k])

        for row, ex in enumerate(chunk):
            target = tuple(ex.target_tokens[:n])
            jp = tuple(j0_preds[row])
            op_ = tuple(o1_preds[row])
            j0_correct.append(jp == target)
            o1_correct.append(op_ == target)
            for k in range(n):
                if jp[k] == target[k]:
                    pos_j0_correct[k] += 1
                if op_[k] == target[k]:
                    pos_o1_correct[k] += 1
            j0_token_correct += sum(1 for k in range(n) if jp[k] == target[k])
            o1_token_correct += sum(1 for k in range(n) if op_[k] == target[k])
        token_total += n * len(chunk)

    n_total = len(examples)
    j0_em = sum(j0_correct) / n_total if n_total else None
    o1_em = sum(o1_correct) / n_total if n_total else None
    return {
        "n": n_total,
        "j0_sequence_exact_match": j0_em,
        "oracle_sequence_exact_match": o1_em,
        "paired_delta_oracle_minus_j0": (
            (o1_em - j0_em) if (j0_em is not None and o1_em is not None) else None
        ),
        "j0_token_accuracy": j0_token_correct / token_total if token_total else None,
        "oracle_token_accuracy": o1_token_correct / token_total if token_total else None,
        "per_output_position": {
            str(k): {
                "j0_accuracy": pos_j0_correct[k] / n_total if n_total else None,
                "oracle_accuracy": pos_o1_correct[k] / n_total if n_total else None,
                "j0_mean_target_logit_margin": (
                    sum(j0_margins[k]) / len(j0_margins[k]) if j0_margins[k] else None
                ),
                "oracle_mean_target_logit_margin": (
                    sum(o1_margins[k]) / len(o1_margins[k]) if o1_margins[k] else None
                ),
            }
            for k in range(n)
        },
    }


def run_stage_c_component_rollback(
    core: Any,
    late_state_dict: dict[str, torch.Tensor],
    early_state_dict: dict[str, torch.Tensor],
    key_groups: dict[str, list[str]],
    datasets: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs the fixed `{R0, R1, ..., R5, R_ALL}` set (task doc Section 6) --
    no combination search. Returns `(points, primitives)` so the caller can
    reuse `primitives["R_ALL"]` for the parity check without rebuilding it."""
    primitives: dict[str, Any] = {
        rollback_id: _build_rollback_primitive(
            core, late_state_dict, early_state_dict,
            REC004K_ROLLBACK_TO_COMPONENT[rollback_id], key_groups,
        )
        for rollback_id in REC004K_ROLLBACK_IDS
    }

    points: dict[str, Any] = {}
    for rollback_id, primitive in primitives.items():
        for dataset_name, examples in datasets.items():
            with torch.no_grad():
                point = compute_rollback_diagnostic_point(core, primitive, examples)
            point.update(
                {
                    "rollback_id": rollback_id,
                    "components_rolled_back": list(REC004K_ROLLBACK_TO_COMPONENT[rollback_id]),
                    "dataset": dataset_name,
                }
            )
            points[f"{rollback_id}:{dataset_name}"] = point
    return points, primitives


def run_component_decomposition_parity_check(
    core: Any, r_all_primitive: Any, early_primitive: Any, datasets: dict[str, list[Any]]
) -> dict[str, Any]:
    """Positive-control gate (task doc Section 6): `R_ALL` replaces every
    O1-relevant parameter with I03@6000's own values, so its O1 forward MUST
    reproduce I03@6000's own real O1 forward exactly (up to float tolerance)
    -- both use literally the same weights for every parameter that feeds
    O1, differing only in `position_bias_hidden`/`position_bias_out`, proven
    (`verify_o1_invariance_to_excluded_params`) not to affect O1's output at
    all. A mismatch means the 5-group decomposition is incomplete/incorrect."""
    resid_audit._assert_eval_only()
    n = REC004K_TARGET_LENGTH
    op = get_operation(REC004K_TARGET_OPERATION)
    per_dataset: dict[str, Any] = {}
    for dataset_name, examples in datasets.items():
        max_diff = 0.0
        all_match = True
        for start in range(0, len(examples), REC004K_CHUNK_SIZE):
            chunk = examples[start : start + REC004K_CHUNK_SIZE]
            content_lengths = [n] * len(chunk)
            output_lengths = [op.output_length(n)] * len(chunk)
            batch_input = collate_content_only_batch(chunk, core.tokens, device=core.device)
            with torch.no_grad():
                h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
                r_all_out = oracle_probe.run_oracle_forward(
                    r_all_primitive, h_content, content_lengths, output_lengths
                )
                early_out = oracle_probe.run_oracle_forward(
                    early_primitive, h_content, content_lengths, output_lengths
                )
            diff = (r_all_out["logits"] - early_out["logits"]).abs().max().item()
            match = resid_audit._predict_from_logits(
                r_all_out["logits"], output_lengths
            ) == resid_audit._predict_from_logits(early_out["logits"], output_lengths)
            max_diff = max(max_diff, diff)
            all_match = all_match and match
        per_dataset[dataset_name] = {
            "max_abs_logit_diff": max_diff,
            "predictions_match": all_match,
            "tolerance": REC004K_PARITY_LOGIT_ABS_TOL,
            "passed": max_diff <= REC004K_PARITY_LOGIT_ABS_TOL and all_match,
        }
    status = (
        "VERIFIED"
        if all(v["passed"] for v in per_dataset.values())
        else "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP"
    )
    return {"task_id": REC004K_TASK_ID, "per_dataset": per_dataset, "status": status}


# =============================================================================
# Stage D -- decision rule, centered on output position 4.
# =============================================================================


def build_i03_component_decision(
    rollback_points: dict[str, Any], parity_check: dict[str, Any]
) -> dict[str, Any]:
    if parity_check["status"] != "VERIFIED":
        return {
            "task_id": REC004K_TASK_ID,
            "label": "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP",
            "sufficient_single_components": [],
            "r_all_passes": False,
            "note": (
                "R_ALL did not reproduce I03@6000's own real O1 forward within "
                "tolerance on at least one dataset -- the 5-group decomposition "
                "(Stage B) is incomplete or incorrect; R1-R5/R_ALL results are "
                "recorded but NOT interpreted as a causal finding"
            ),
        }

    pos_key = str(REC004K_POSITION_MAIN_RESIDUAL)

    def _passes(rollback_id: str) -> bool:
        for ds in REC004K_DATASETS:
            row = rollback_points.get(f"{rollback_id}:{ds}", {})
            em = row.get("oracle_sequence_exact_match")
            pos4 = row.get("per_output_position", {}).get(pos_key, {}).get("oracle_accuracy")
            if em is None or pos4 is None:
                return False
            if em < REC004K_ROLLBACK_EM_THRESHOLD or pos4 < REC004K_ROLLBACK_POSITION_ACC_THRESHOLD:
                return False
        return True

    single_ids = ("R1", "R2", "R3", "R4", "R5")
    sufficient = [rid for rid in single_ids if _passes(rid)]
    r_all_passes = _passes("R_ALL")

    if len(sufficient) == 1:
        component = REC004K_ROLLBACK_TO_COMPONENT[sufficient[0]][0]
        label = f"{component}_ROLLBACK_SUFFICIENT"
    elif len(sufficient) > 1:
        label = "MULTIPLE_COMPONENTS_ROLLBACK_SUFFICIENT"
    elif r_all_passes:
        label = "DISTRIBUTED_DOWNSTREAM_COADAPTATION"
    else:
        label = "DOWNSTREAM_DECOMPOSITION_INCOMPLETE"

    return {
        "task_id": REC004K_TASK_ID,
        "label": label,
        "sufficient_single_components": [
            REC004K_ROLLBACK_TO_COMPONENT[rid][0] for rid in sufficient
        ],
        "r_all_passes": r_all_passes,
        "oracle_em_threshold": REC004K_ROLLBACK_EM_THRESHOLD,
        "position4_accuracy_threshold": REC004K_ROLLBACK_POSITION_ACC_THRESHOLD,
    }


# =============================================================================
# Report.
# =============================================================================


def build_report_markdown(
    stage_a_points: dict[str, Any],
    stage_a_decision: dict[str, Any],
    rec004j_cross_check: dict[str, Any],
    component_key_groups: dict[str, list[str]] | None,
    forward_graph_check: dict[str, Any] | None,
    rollback_points: dict[str, Any] | None,
    parity_check: dict[str, Any] | None,
    i03_component_decision: dict[str, Any] | None,
) -> str:
    lines: list[str] = [
        f"# {REC004K_TASK_ID}: I03 Matched-Data Temporal Mechanism Recheck & "
        "Same-Init Downstream Rollback Audit",
        "",
        "Zero new optimizer updates. Forward-only; every rollback is an "
        "in-memory, evaluation-only merged state dict.",
        "",
        "## Stage A -- temporal recheck (I03, both REC-004J datasets)",
        "",
        "| step | dataset | n | J0 EM | O1 EM | delta |",
        "|---|---|---|---|---|---|",
    ]
    for step in REC004K_TEMPORAL_STEPS:
        for dataset_name in REC004K_DATASETS:
            row = stage_a_points.get(f"{step}:{dataset_name}", {})
            if row.get("status") != "VERIFIED":
                lines.append(
                    f"| {step} | {dataset_name} | - | {row.get('status')} | | |"
                )
                continue
            lines.append(
                f"| {step} | {dataset_name} | {row['n']} | "
                f"{row['j0_sequence_exact_match']:.4f} | "
                f"{row['oracle_sequence_exact_match']:.4f} | "
                f"{row['paired_delta_oracle_minus_j0']:+.4f} |"
            )
    lines += [
        "",
        f"- Stage A decision: **{stage_a_decision['label']}**",
        f"  - step={REC004K_EARLY_STEP} passes both datasets: "
        f"{stage_a_decision['early_step_passes_both_datasets']}",
        f"  - step={REC004K_LATE_STEP} passes both datasets: "
        f"{stage_a_decision['late_step_passes_both_datasets']}",
        f"- cross-check vs REC-004J's own step={REC004K_LATE_STEP} numbers: "
        f"{rec004j_cross_check.get('status')}",
    ]

    if not stage_a_decision["authorizes_component_rollback"]:
        lines += [
            "",
            "Stage B/C/D NOT executed: Stage A did not confirm "
            "`TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED`.",
        ]
        return "\n".join(lines) + "\n"

    lines += [
        "",
        "## Stage B -- forward-graph parameter groups",
        "",
    ]
    if component_key_groups is not None:
        for cid in (*REC004K_COMPONENT_IDS, "EXCLUDED_FROM_O1"):
            lines.append(f"- `{cid}`: {component_key_groups.get(cid, [])}")
    if forward_graph_check is not None:
        lines += [
            "",
            "## Stage B -- forward-graph invariance check",
            f"- invariant (excluded params do not affect O1): "
            f"{forward_graph_check['invariant']}",
            f"- weights restored exactly: {forward_graph_check['weights_restored_exactly']}",
        ]
    if not (forward_graph_check and forward_graph_check["invariant"]
            and forward_graph_check["weights_restored_exactly"]):
        lines += [
            "",
            "Stage C/D NOT executed: forward-graph invariance check failed.",
        ]
        return "\n".join(lines) + "\n"

    lines += [
        "",
        "## Stage C -- component rollback (I03@17500 base, O1)",
        "",
        "| rollback | dataset | n | J0 EM | O1 EM | delta | pos4 acc | pos5 acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if rollback_points is not None:
        for rollback_id in REC004K_ROLLBACK_IDS:
            for dataset_name in REC004K_DATASETS:
                row = rollback_points.get(f"{rollback_id}:{dataset_name}", {})
                pos4 = row.get("per_output_position", {}).get(
                    str(REC004K_POSITION_MAIN_RESIDUAL), {}
                ).get("oracle_accuracy")
                pos5 = row.get("per_output_position", {}).get(
                    str(REC004K_POSITION_SYMMETRIC_CONTROL), {}
                ).get("oracle_accuracy")
                lines.append(
                    f"| {rollback_id} | {dataset_name} | {row.get('n')} | "
                    f"{row.get('j0_sequence_exact_match'):.4f} | "
                    f"{row.get('oracle_sequence_exact_match'):.4f} | "
                    f"{row.get('paired_delta_oracle_minus_j0'):+.4f} | "
                    f"{pos4:.4f} | {pos5:.4f} |"
                )

    lines += ["", "## Component decomposition parity check (R_ALL vs I03@6000 real O1)"]
    if parity_check is not None:
        lines.append(f"- status: **{parity_check['status']}**")
        for dataset_name, row in parity_check["per_dataset"].items():
            lines.append(
                f"  - {dataset_name}: max_abs_logit_diff={row['max_abs_logit_diff']:.6f}, "
                f"predictions_match={row['predictions_match']}, passed={row['passed']}"
            )

    lines += ["", "## I03 component decision"]
    if i03_component_decision is not None:
        lines.append(f"- label: **{i03_component_decision['label']}**")
        lines.append(
            f"- sufficient single components: "
            f"{i03_component_decision['sufficient_single_components']}"
        )
        lines.append(f"- R_ALL passes: {i03_component_decision['r_all_passes']}")

    lines += [
        "",
        "## Fixed non-adoption fields",
        "new_optimizer_updates=0, selected_init=null, selected_step=null, "
        "selected_component=null, child_bundle=null, rg3_recheck=NOT_EXECUTED, "
        "rec005_eligible=false",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Full task orchestration.
# =============================================================================


def run_mirror_temporal_mechanism_rollback_audit_task(
    config: MirrorTemporalMechanismRollbackAuditConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_temporal_mechanism_rollback_audit_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != REC004K_SEED:
        raise ValueError(
            f"B-C005REC-004K reads REC-004D/H/J artifacts pre-registered under seed "
            f"{REC004K_SEED}; got seed={seed}"
        )

    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_jsonl(path: Path, points: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for key, row in points.items():
                fh.write(json.dumps({"key": key, **row}, default=str) + "\n")

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    datasets, clean_v2_detail, probe_detail = build_stage_a_datasets(seed)
    _write_json(output_dir / "clean_v2_detail.json", clean_v2_detail)
    _write_json(output_dir / "length10_mechanism_probe_v1_manifest.json", probe_detail)

    with torch.no_grad():
        stage_a_points, stage_a_load_infos, primitives_by_step = run_stage_a_temporal_recheck(
            core, datasets
        )
    _write_jsonl(output_dir / "stage_a_temporal_points.jsonl", stage_a_points)
    _write_json(output_dir / "stage_a_temporal_points_summary.json", stage_a_points)
    _write_json(output_dir / "stage_a_checkpoint_load_info.json", stage_a_load_infos)

    rec004j_cross_check = cross_check_step17500_against_rec004j(stage_a_points)
    _write_json(output_dir / "rec004j_cross_check.json", rec004j_cross_check)

    stage_a_decision = build_stage_a_temporal_decision(stage_a_points)
    _write_json(output_dir / "stage_a_temporal_decision.json", stage_a_decision)

    component_key_groups: dict[str, list[str]] | None = None
    forward_graph_check: dict[str, Any] | None = None
    rollback_points: dict[str, Any] | None = None
    parity_check: dict[str, Any] | None = None
    i03_component_decision: dict[str, Any] | None = None
    stage_c_status = "NOT_EXECUTED_TEMPORAL_SHIFT_NOT_ESTABLISHED"

    if stage_a_decision["authorizes_component_rollback"]:
        late_primitive = primitives_by_step[REC004K_LATE_STEP]
        early_primitive = primitives_by_step[REC004K_EARLY_STEP]

        all_keys = list(late_primitive.state_dict().keys())
        component_key_groups = partition_state_dict_keys(all_keys)
        _write_json(output_dir / "component_key_groups.json", component_key_groups)

        probe_examples_for_audit = datasets[REC004K_PROBE_DATASET][
            :REC004K_INVARIANCE_CHECK_EXAMPLES
        ]
        op = get_operation(REC004K_TARGET_OPERATION)
        n = REC004K_TARGET_LENGTH
        content_lengths = [n] * len(probe_examples_for_audit)
        output_lengths = [op.output_length(n)] * len(probe_examples_for_audit)
        batch_input = collate_content_only_batch(
            probe_examples_for_audit, core.tokens, device=core.device
        )
        with torch.no_grad():
            h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
            forward_graph_check = verify_o1_invariance_to_excluded_params(
                late_primitive, h_content, content_lengths, output_lengths
            )
        _write_json(output_dir / "forward_graph_invariance_check.json", forward_graph_check)

        if forward_graph_check["invariant"] and forward_graph_check["weights_restored_exactly"]:
            late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
            early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}

            with torch.no_grad():
                rollback_points, rollback_primitives = run_stage_c_component_rollback(
                    core, late_sd, early_sd, component_key_groups, datasets
                )
            _write_jsonl(output_dir / "stage_c_rollback_points.jsonl", rollback_points)
            _write_json(output_dir / "stage_c_rollback_points_summary.json", rollback_points)

            with torch.no_grad():
                parity_check = run_component_decomposition_parity_check(
                    core, rollback_primitives["R_ALL"], early_primitive, datasets
                )
            _write_json(output_dir / "component_decomposition_parity_check.json", parity_check)

            i03_component_decision = build_i03_component_decision(rollback_points, parity_check)
            _write_json(output_dir / "i03_component_decision.json", i03_component_decision)
            stage_c_status = "EXECUTED"
        else:
            stage_c_status = "NOT_EXECUTED_FORWARD_GRAPH_INVARIANCE_CHECK_FAILED"

    protected_hashes_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004K_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == core_hash_before,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "checkpoint_load_status": {k: v["status"] for k, v in stage_a_load_infos.items()},
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004K_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    n_predictions = sum(
        row.get("n", 0) for row in stage_a_points.values() if isinstance(row, dict)
    ) * 2
    if rollback_points is not None:
        n_predictions += sum(
            row.get("n", 0) for row in rollback_points.values() if isinstance(row, dict)
        ) * 2
    cost_accounting = {
        "task_id": REC004K_TASK_ID,
        "new_optimizer_updates": 0,
        "n_forward_predictions": n_predictions,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=seed)
    _write_json(output_dir / "system.json", system_info)

    pi_10 = mpid.mirror_halves_position_map(REC004K_TARGET_LENGTH)
    protocol = {
        "task_id": REC004K_TASK_ID,
        "source_task_ids": list(REC004K_SOURCE_TASK_IDS),
        "contract_file": str(REC004K_CONTRACT_FILE),
        "decisive_init": REC004K_DECISIVE_INIT,
        "early_step": REC004K_EARLY_STEP,
        "late_step": REC004K_LATE_STEP,
        "position_main_residual": REC004K_POSITION_MAIN_RESIDUAL,
        "position_main_residual_source": pi_10[REC004K_POSITION_MAIN_RESIDUAL],
        "position_symmetric_control": REC004K_POSITION_SYMMETRIC_CONTROL,
        "position_symmetric_control_source": pi_10[REC004K_POSITION_SYMMETRIC_CONTROL],
        "rollback_ids": list(REC004K_ROLLBACK_IDS),
        "forbidden": [
            "new optimizer updates", "any I03 checkpoint file mutation",
            "combination search beyond the fixed R0-R5/R_ALL set",
            "candidate adoption", "child assembly", "RG3 recheck",
            "any further repair auto-started from this audit's result",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)

    all_stage_a_verified = all(
        v.get("status") == "VERIFIED" for v in stage_a_load_infos.values()
    )
    all_stage_a_points_computed = all(
        row.get("status") == "VERIFIED" for row in stage_a_points.values() if isinstance(row, dict)
    )

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE" if all_stage_a_points_computed else "PARTIAL",
        "checkpoint_source_replay_status": (
            "VERIFIED" if all_stage_a_verified else "SOURCE_REPLAY_MISMATCH"
        ),
        "rec004j_cross_check_status": rec004j_cross_check.get("status"),
        "stage_a_temporal_decision": stage_a_decision["label"],
        "stage_c_status": stage_c_status,
        "component_decomposition_parity_status": (
            parity_check["status"] if parity_check is not None else None
        ),
        "i03_component_decision": (
            i03_component_decision["label"] if i03_component_decision is not None else None
        ),
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "selected_component": None,
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
        stage_a_points, stage_a_decision, rec004j_cross_check,
        component_key_groups, forward_graph_check, rollback_points,
        parity_check, i03_component_decision,
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    return result
