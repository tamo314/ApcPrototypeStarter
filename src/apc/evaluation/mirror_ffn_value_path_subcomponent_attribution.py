"""B-C005REC-004M: I03 FFN-Value-Path Subcomponent Attribution & Freeze-Repair
Contract.

Follows
`docs/CODEX_TASKS_PHASE_B_B2_FFN_VALUE_PATH_SUBCOMPONENT_ATTRIBUTION.md`.

`B-C005REC-004L` (ADR-0108) found `FFN_VALUE_OUTPROJ_INTERACTION_SUFFICIENT`:
jointly rolling back `FFN_BLOCK` + REC-004K's own `VALUE_OUTPROJ` group
(`content_in_proj`, `content_position_embedding`, the fused
`cross_attn.in_proj_weight`/`in_proj_bias`, and `cross_attn.out_proj`) from
I03@17500 to I03@6000 perfectly recovers O1 sequence EM and position-4/5
accuracy. `VALUE_OUTPROJ` bundles four functionally distinct roles, and
REC-004K's own note already flagged rolling back the WHOLE fused Q/K/V
tensor as a disclosed confound (it also destroys J0's own attention
pattern, since Q/K feed the real, learnable attention score). This task
asks exactly one question: does the interaction localize to ONE of four
finer subcomponents -- `CONTENT_PREP`, `V_PROJECTION` (the V row-slice of
the fused tensor only), `ATTN_OUT_PROJ`, or `SCORE_PROJECTION_CONTROL` (the
Q/K row-slices, a negative control since O1's oracle substitution never
reads Q/K) -- and if so, can a two-stage training repair be designed that
freezes only that subcomponent (plus FFN) while leaving Q/K and position
bias free to keep learning the attention score?

**Zero new optimizer updates.** Reads the SAME already-saved I03 checkpoints
REC-004K/L used (step=6000 from `B-C005REC-004D`'s tree, step=17500 from
`B-C005REC-004H`'s), plus -- Stage E only, and only if Stage D localizes to
one subcomponent -- I04@18000's and I05@17500's own step=6000/late
checkpoints. Every condition is a fresh, in-memory, evaluation-only
`CrossPositionLengthBiasPrimitive` merged from two real state dicts; the
fused `in_proj_weight`/`in_proj_bias` tensors are split by ROW-SLICE
(`chunk(3, dim=0)` order Q, K, V) rather than swapped whole -- never a new
checkpoint file, never a mutation of either loaded primitive in place.
`selected_init`, `selected_step`, `selected_value_subcomponent`, and
`child_bundle` stay `null`; `rg3_recheck` stays `"NOT_EXECUTED"`;
`rec005_eligible` stays `false`, unconditionally.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml

from apc.core.data import collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_ffn_anchored_downstream_interaction_audit as rec004l
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation import mirror_temporal_mechanism_rollback_audit as rec004k
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.evaluation.unified_oracle_causal_benchmark import _derive_local_seed
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004M_TASK_ID",
    "REC004M_SOURCE_TASK_IDS",
    "REC004M_DECISIVE_INIT",
    "REC004M_EARLY_STEP",
    "REC004M_LATE_STEP",
    "REC004M_TARGET_LENGTH",
    "REC004M_SUBCOMPONENT_IDS",
    "REC004M_CONDITION_IDS",
    "REC004M_PAIR_CONDITION_IDS",
    "REC004M_CONDITION_TO_SUBCOMPONENTS",
    "REC004M_DATASETS",
    "REC004M_NEW_PROBE_SPLIT",
    "REC004M_NEW_PROBE_EXAMPLES",
    "REC004M_ROLLBACK_EM_THRESHOLD",
    "REC004M_ROLLBACK_POSITION_ACC_THRESHOLD",
    "REC004M_SAFETY_TARGETS",
    "REC004M_SAFETY_EM_DEGRADATION_THRESHOLD",
    "MirrorFfnValuePathSubcomponentAttributionConfig",
    "run_mirror_ffn_value_path_subcomponent_attribution_task",
]

# =============================================================================
# Constants. Checkpoint identity, dataset identity for the three pre-existing
# datasets, and the 0.95 EM/position-4 floor are all inherited from REC-004K/
# REC-004L (never retyped). Only the subcomponent grouping, the new probe
# dataset, the FFN+subcomponent condition set, and the position-4-centered
# decision labels are new here -- pre-registered before any point below is
# computed.
# =============================================================================

REC004M_TASK_ID: Final = "B-C005REC-004M"
REC004M_SOURCE_TASK_IDS: Final[tuple[str, ...]] = ("B-C005REC-004K", "B-C005REC-004L")
REC004M_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_FFN_VALUE_PATH_SUBCOMPONENT_ATTRIBUTION.md"
)

REC004M_TARGET_OPERATION: Final = rec004k.REC004K_TARGET_OPERATION  # "MIRROR_HALVES"
REC004M_ARM: Final = rec004k.REC004K_ARM  # "P_LENGTH_POSITION_BIAS"
REC004M_TARGET_LENGTH: Final = rec004k.REC004K_TARGET_LENGTH  # 10
REC004M_VOCAB_SIZE: Final = rec004j.REC004J_VOCAB_SIZE
REC004M_SEED: Final = RECOVERY_PILOT_SEED

REC004M_DECISIVE_INIT: Final = rec004k.REC004K_DECISIVE_INIT  # "I03"
REC004M_EARLY_STEP: Final = rec004k.REC004K_EARLY_STEP  # 6000
REC004M_LATE_STEP: Final = rec004k.REC004K_LATE_STEP  # 17500

REC004M_CLEAN_V2_DATASET: Final = rec004k.REC004K_CLEAN_V2_DATASET
REC004M_PROBE_DATASET: Final = rec004k.REC004K_PROBE_DATASET
REC004M_DOWNSTREAM_PROBE_DATASET: Final = rec004l.REC004L_NEW_PROBE_SPLIT
REC004M_NEW_PROBE_SPLIT: Final = "length10_value_path_subcomponent_probe_v1"
REC004M_NEW_PROBE_EXAMPLES: Final = 512
REC004M_DATASETS: Final[tuple[str, ...]] = (
    REC004M_CLEAN_V2_DATASET,
    REC004M_PROBE_DATASET,
    REC004M_DOWNSTREAM_PROBE_DATASET,
    REC004M_NEW_PROBE_SPLIT,
)

REC004M_POSITION_MAIN_RESIDUAL: Final = rec004k.REC004K_POSITION_MAIN_RESIDUAL  # 4
REC004M_POSITION_SYMMETRIC_CONTROL: Final = rec004k.REC004K_POSITION_SYMMETRIC_CONTROL  # 5
REC004M_REGRESSION_POSITIONS: Final[tuple[int, ...]] = (0, 1, 2, 3, 6, 7, 8, 9)

REC004M_ROLLBACK_EM_THRESHOLD: Final = rec004k.REC004K_ROLLBACK_EM_THRESHOLD  # 0.95
REC004M_ROLLBACK_POSITION_ACC_THRESHOLD: Final = (
    rec004k.REC004K_ROLLBACK_POSITION_ACC_THRESHOLD
)  # 0.95
REC004M_PARITY_LOGIT_ABS_TOL: Final = rec004k.REC004K_PARITY_LOGIT_ABS_TOL  # 5e-3

REC004M_CHUNK_SIZE: Final = rec004k.REC004K_CHUNK_SIZE  # 128
REC004M_INVARIANCE_CHECK_EXAMPLES: Final = rec004k.REC004K_INVARIANCE_CHECK_EXAMPLES

# Stage A -- the four O1-relevant subcomponents REC-004K's own VALUE_OUTPROJ
# group is split into (task doc Section 3). Real keys are resolved from the
# live checkpoint's own state_dict(), never assumed from this list alone --
# see `build_value_path_subcomponent_key_groups`.
REC004M_SUBCOMPONENT_IDS: Final[tuple[str, ...]] = (
    "CONTENT_PREP", "V_PROJECTION", "ATTN_OUT_PROJ", "SCORE_PROJECTION_CONTROL",
)
REC004M_WHOLE_KEY_SUBCOMPONENT_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "CONTENT_PREP": ("content_in_proj.", "content_position_embedding."),
    "ATTN_OUT_PROJ": ("cross_attn.out_proj.",),
}
# The two fused Q/K/V tensors: V_PROJECTION and SCORE_PROJECTION_CONTROL are
# both carved out of these same keys, by row-slice, not by key membership.
REC004M_FUSED_IN_PROJ_KEYS: Final[tuple[str, ...]] = (
    "cross_attn.in_proj_weight", "cross_attn.in_proj_bias",
)
REC004M_SUBCOMPONENT_TO_MERGE_KIND: Final[dict[str, str]] = {
    "CONTENT_PREP": "WHOLE",
    "ATTN_OUT_PROJ": "WHOLE",
    "V_PROJECTION": "SLICE_V",
    "SCORE_PROJECTION_CONTROL": "SLICE_QK",
}

# Stage B -- pre-registered FFN-anchored condition set (task doc Section 4).
# Exactly R0 / F / four FFN+subcomponent pairs / F_ALLV / EARLY -- no
# combination beyond this set, no condition added after seeing a result.
REC004M_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_C", "F_V", "F_O", "F_QK", "F_ALLV", "EARLY",
)
REC004M_CONDITION_TO_SUBCOMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    "R0": (),
    "F": (),
    "F_C": ("CONTENT_PREP",),
    "F_V": ("V_PROJECTION",),
    "F_O": ("ATTN_OUT_PROJ",),
    "F_QK": ("SCORE_PROJECTION_CONTROL",),
    "F_ALLV": REC004M_SUBCOMPONENT_IDS,
}
REC004M_PAIR_CONDITION_IDS: Final[tuple[str, ...]] = ("F_C", "F_V", "F_O", "F_QK")
REC004M_MERGE_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_C", "F_V", "F_O", "F_QK", "F_ALLV",
)
# FFN_BLOCK is rolled back for every merge condition except R0.
REC004M_FFN_ROLLED_BACK: Final[dict[str, bool]] = {
    "R0": False, "F": True, "F_C": True, "F_V": True, "F_O": True,
    "F_QK": True, "F_ALLV": True,
}
REC004M_SUBCOMPONENT_TO_LABEL_PARTNER: Final[dict[str, str]] = {
    "F_C": "CONTENT_PREP",
    "F_V": "V_PROJECTION",
    "F_O": "ATTN_OUT_PROJ",
    "F_QK": "SCORE_PROJECTION_CONTROL",
}

# Stage E -- success-model safety check targets (task doc Section 7). Each
# uses its OWN step=6000 state; no cross-init weight transplant.
REC004M_SAFETY_TARGETS: Final[tuple[tuple[str, int], ...]] = (("I04", 18000), ("I05", 17500))
REC004M_SAFETY_EM_DEGRADATION_THRESHOLD: Final = 0.05


def _config_to_yaml_dict(
    config: MirrorFfnValuePathSubcomponentAttributionConfig,
) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


@dataclass(frozen=True)
class MirrorFfnValuePathSubcomponentAttributionConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004m/run_001")
    seed: int = RECOVERY_PILOT_SEED


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, points: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for key, row in points.items():
            fh.write(json.dumps({"key": key, **row}, default=str) + "\n")


# =============================================================================
# Stage A -- split REC-004K's own VALUE_OUTPROJ group into 4 real-key-
# verified subcomponents. Every other REC-004K group (QUERY_RESIDUAL_PATH,
# POST_ATTN_NORM, FFN_BLOCK, READOUT, EXCLUDED_FROM_O1) passes through
# unchanged.
# =============================================================================


def build_value_path_subcomponent_key_groups(keys: list[str]) -> dict[str, list[str]]:
    """Further decomposes REC-004K's own `VALUE_OUTPROJ` group (see
    `rec004k.partition_state_dict_keys`) into `CONTENT_PREP`/`ATTN_OUT_PROJ`
    (whole real keys) plus the shared `IN_PROJ_FUSED_KEYS` basis that
    `V_PROJECTION`/`SCORE_PROJECTION_CONTROL` each carve a row-slice from.
    Self-verifies every real `VALUE_OUTPROJ` key is accounted for exactly
    once -- raises rather than silently dropping or double-counting a key
    this checkpoint's own architecture happens to have."""
    base_groups = rec004k.partition_state_dict_keys(keys)
    value_outproj_keys = set(base_groups["VALUE_OUTPROJ"])

    content_prep = sorted(
        k for k in value_outproj_keys
        if any(k.startswith(p) for p in REC004M_WHOLE_KEY_SUBCOMPONENT_PREFIXES["CONTENT_PREP"])
    )
    attn_out_proj = sorted(
        k for k in value_outproj_keys
        if any(k.startswith(p) for p in REC004M_WHOLE_KEY_SUBCOMPONENT_PREFIXES["ATTN_OUT_PROJ"])
    )
    fused_keys = sorted(k for k in value_outproj_keys if k in REC004M_FUSED_IN_PROJ_KEYS)

    accounted = set(content_prep) | set(attn_out_proj) | set(fused_keys)
    if accounted != value_outproj_keys:
        raise AssertionError(
            "VALUE_OUTPROJ's real keys are not fully accounted for by the "
            f"CONTENT_PREP/ATTN_OUT_PROJ/fused-in_proj sub-partition: "
            f"missing={sorted(value_outproj_keys - accounted)}, "
            f"unexpected={sorted(accounted - value_outproj_keys)}"
        )
    if "cross_attn.in_proj_weight" not in fused_keys:
        raise AssertionError(
            "expected cross_attn.in_proj_weight in VALUE_OUTPROJ's real keys -- "
            "the V/QK row-slice decomposition has nothing to operate on"
        )

    groups: dict[str, list[str]] = {k: list(v) for k, v in base_groups.items()}
    del groups["VALUE_OUTPROJ"]
    groups["CONTENT_PREP"] = content_prep
    groups["ATTN_OUT_PROJ"] = attn_out_proj
    groups["IN_PROJ_FUSED_KEYS"] = fused_keys
    return groups


def _rollback_in_proj_slice(
    merged: dict[str, torch.Tensor],
    late_sd: dict[str, torch.Tensor],
    early_sd: dict[str, torch.Tensor],
    key: str,
    which: str,
) -> None:
    """Replaces exactly the Q/K or V row-slice of a fused `in_proj_weight`/
    `in_proj_bias` tensor (`chunk(3, dim=0)` order Q, K, V -- the same
    `nn.MultiheadAttention` convention `_oracle_attention` and REC-004K's own
    invariance check rely on) with `early_sd`'s values, building on top of
    `merged`'s CURRENT value for `key` -- so requesting both the V and QK
    slices for the same condition (e.g. `F_ALLV`) composes to the full
    tensor instead of one call clobbering the other."""
    base = (merged[key] if key in merged else late_sd[key]).clone()
    early_t = early_sd[key]
    total = base.shape[0]
    if total % 3 != 0:
        raise AssertionError(f"{key} shape[0]={total} is not divisible by 3 (Q/K/V chunks)")
    d = total // 3
    if which == "V":
        base[2 * d : 3 * d] = early_t[2 * d : 3 * d].clone()
    elif which == "QK":
        base[0 : 2 * d] = early_t[0 : 2 * d].clone()
    else:
        raise ValueError(f"which must be 'V' or 'QK', got {which!r}")
    merged[key] = base


def _build_subcomponent_rollback_primitive(
    core: Any,
    late_sd: dict[str, torch.Tensor],
    early_sd: dict[str, torch.Tensor],
    subcomponent_ids: tuple[str, ...],
    key_groups: dict[str, list[str]],
    ffn_rolled_back: bool,
) -> Any:
    merged = {k: v.clone() for k, v in late_sd.items()}
    if ffn_rolled_back:
        for key in key_groups["FFN_BLOCK"]:
            merged[key] = early_sd[key].clone()
    for cid in subcomponent_ids:
        kind = REC004M_SUBCOMPONENT_TO_MERGE_KIND[cid]
        if kind == "WHOLE":
            for key in key_groups[cid]:
                merged[key] = early_sd[key].clone()
        elif kind == "SLICE_V":
            for key in key_groups["IN_PROJ_FUSED_KEYS"]:
                _rollback_in_proj_slice(merged, late_sd, early_sd, key, "V")
        elif kind == "SLICE_QK":
            for key in key_groups["IN_PROJ_FUSED_KEYS"]:
                _rollback_in_proj_slice(merged, late_sd, early_sd, key, "QK")
        else:
            raise ValueError(f"unknown merge kind {kind!r} for subcomponent {cid!r}")
    primitive = mpbr._new_arm_primitive(core, REC004M_ARM)
    primitive.to(core.device)
    primitive.load_state_dict({k: v.to(core.device) for k, v in merged.items()}, strict=True)
    primitive.eval()
    for p in primitive.parameters():
        p.requires_grad_(False)
    return primitive


def verify_qk_rollback_is_o1_invariant(
    core: Any,
    late_sd: dict[str, torch.Tensor],
    early_sd: dict[str, torch.Tensor],
    key_groups: dict[str, list[str]],
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, Any]:
    """Empirical, rollback-based proof (task doc Section 3/6) that
    `SCORE_PROJECTION_CONTROL` (Q/K row-slices) is inert under O1: `F_QK`'s
    real O1 forward, built from the ACTUAL I03@6000/@17500 weights (not a
    random perturbation), must be byte-identical to `F`'s. Complements
    (does not replace) `rec004k.verify_o1_invariance_to_excluded_params`,
    which proves the general claim via perturbation on a checkpoint-
    independent basis."""
    resid_audit._assert_eval_only()
    f_primitive = _build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, (), key_groups, ffn_rolled_back=True
    )
    f_qk_primitive = _build_subcomponent_rollback_primitive(
        core, late_sd, early_sd, ("SCORE_PROJECTION_CONTROL",), key_groups, ffn_rolled_back=True
    )
    with torch.no_grad():
        f_out = oracle_probe.run_oracle_forward(
            f_primitive, content_features, content_lengths, output_lengths
        )
        f_qk_out = oracle_probe.run_oracle_forward(
            f_qk_primitive, content_features, content_lengths, output_lengths
        )
    max_abs_diff = (f_qk_out["logits"] - f_out["logits"]).abs().max().item()
    predictions_match = resid_audit._predict_from_logits(
        f_qk_out["logits"], output_lengths
    ) == resid_audit._predict_from_logits(f_out["logits"], output_lengths)
    return {
        "task_id": REC004M_TASK_ID,
        "comparison": "F_QK (real I03@6000/@17500 weights) vs F, under O1",
        "max_abs_logit_diff": max_abs_diff,
        "predictions_match": predictions_match,
        "invariant": max_abs_diff == 0.0 and predictions_match,
    }


# =============================================================================
# Stage C -- one new, fully disjoint dataset. Follows
# `rec004l.build_length10_downstream_interaction_probe_v1`'s exact
# collision-substitution recipe verbatim; only the split label (hence RNG
# stream, via `_derive_local_seed`) and the (larger) protected set differ.
# A pure function of `(seed, protected_digests)`.
# =============================================================================


def build_length10_value_path_subcomponent_probe_v1(
    seed: int, protected_digests: set[str], n: int = REC004M_NEW_PROBE_EXAMPLES
) -> tuple[list[Example], dict[str, Any]]:
    seed_label = f"{REC004M_NEW_PROBE_SPLIT}:{REC004M_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed(seed, 0, seed_label))
    op_obj = get_operation(REC004M_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq = tuple(
                rng.randrange(REC004M_VOCAB_SIZE) for _ in range(REC004M_TARGET_LENGTH)
            )
            params = op_obj.sample_params(rng, seq, REC004M_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004M_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004M_VOCAB_SIZE)
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
                split=REC004M_NEW_PROBE_SPLIT,
                vocab_size=REC004M_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004M_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "target_length": REC004M_TARGET_LENGTH,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            f"{REC004M_NEW_PROBE_SPLIT} is a development-exposed diagnostic set for "
            "B-C005REC-004M, not a sealed or final RG3 query set; once consumed by "
            "this task it is not eligible to serve as an independent RG3 recheck "
            "query without a fresh, disjoint regeneration."
        ),
    }
    return examples, detail


def build_stage_c_datasets(seed: int) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Byte-identical reuse of REC-004K's own two datasets and REC-004L's
    own new dataset (paired continuity across REC-004K/L/M) plus ONE new
    dataset disjoint from all of: the training stream (steps 1-18000), old
    validation, every REC-004A-D reference/sealed split, `clean_selection_
    validation_v2`'s full 1024-example set, `length10_mechanism_probe_v1`,
    and `length10_downstream_interaction_probe_v1` itself."""
    datasets_kl, clean_v2_detail, probe_detail = rec004k.build_stage_a_datasets(seed)
    all_1024, _dup_len10, _dup_detail, _dup_counts = (
        rec004j.regenerate_clean_selection_validation_v2(seed)
    )
    protected_v1, _counts_v1 = traj_audit.build_protected_digest_registry(seed)
    protected_v2 = protected_v1 | traj_audit._digest_examples(all_1024)
    protected_v3 = protected_v2 | traj_audit._digest_examples(datasets_kl[REC004M_PROBE_DATASET])

    downstream_examples, downstream_detail = rec004l.build_length10_downstream_interaction_probe_v1(
        seed, protected_v3
    )
    protected_v4 = protected_v3 | traj_audit._digest_examples(downstream_examples)

    new_probe_examples, new_probe_detail = build_length10_value_path_subcomponent_probe_v1(
        seed, protected_v4
    )

    new_probe_digests = traj_audit._digest_examples(new_probe_examples)
    overlap: dict[str, Any] = {
        "overlap_with_training_stream_and_reference_sealed_sets": sorted(
            new_probe_digests & protected_v1
        ),
        "overlap_with_clean_selection_validation_v2_full_1024": sorted(
            new_probe_digests & traj_audit._digest_examples(all_1024)
        ),
        "overlap_with_length10_mechanism_probe_v1": sorted(
            new_probe_digests & traj_audit._digest_examples(datasets_kl[REC004M_PROBE_DATASET])
        ),
        "overlap_with_length10_downstream_interaction_probe_v1": sorted(
            new_probe_digests & traj_audit._digest_examples(downstream_examples)
        ),
    }
    overlap["disjoint"] = not any(v for v in overlap.values())
    new_probe_detail["disjointness_check"] = overlap
    new_probe_detail["frozen_dataset_digest_sha256"] = rec004l._dataset_digest(new_probe_examples)

    downstream_detail = dict(downstream_detail)
    downstream_detail["regenerated_byte_identically_from"] = "B-C005REC-004L"

    datasets = {
        REC004M_CLEAN_V2_DATASET: datasets_kl[REC004M_CLEAN_V2_DATASET],
        REC004M_PROBE_DATASET: datasets_kl[REC004M_PROBE_DATASET],
        REC004M_DOWNSTREAM_PROBE_DATASET: downstream_examples,
        REC004M_NEW_PROBE_SPLIT: new_probe_examples,
    }
    dataset_details = {
        REC004M_CLEAN_V2_DATASET: clean_v2_detail,
        REC004M_PROBE_DATASET: probe_detail,
        REC004M_DOWNSTREAM_PROBE_DATASET: downstream_detail,
        REC004M_NEW_PROBE_SPLIT: new_probe_detail,
    }
    return datasets, dataset_details


# =============================================================================
# Stage B/D -- build the 8 fixed conditions and run them, reusing REC-004L's
# own per-point metric function (O1 EM, per-position accuracy, hidden/logit
# diff vs EARLY) unmodified.
# =============================================================================


def build_condition_primitives(
    core: Any,
    late_state_dict: dict[str, torch.Tensor],
    early_state_dict: dict[str, torch.Tensor],
    key_groups: dict[str, list[str]],
    early_primitive: Any,
) -> dict[str, Any]:
    primitives: dict[str, Any] = {
        condition_id: _build_subcomponent_rollback_primitive(
            core, late_state_dict, early_state_dict,
            REC004M_CONDITION_TO_SUBCOMPONENTS[condition_id], key_groups,
            REC004M_FFN_ROLLED_BACK[condition_id],
        )
        for condition_id in REC004M_MERGE_CONDITION_IDS
    }
    primitives["EARLY"] = early_primitive
    return primitives


def run_stage_conditions(
    core: Any,
    condition_primitives: dict[str, Any],
    datasets: dict[str, list[Example]],
) -> dict[str, Any]:
    """Runs the fixed 8-condition set (task doc Section 4) across all four
    datasets -- no combination search, no additional condition added after
    seeing a result. Reuses REC-004L's own `compute_interaction_diagnostic_
    point` unmodified (O1 EM, per-position accuracy/margin, hidden-state
    diff and final-logit diff vs `EARLY`)."""
    reference_primitive = condition_primitives["EARLY"]
    points: dict[str, Any] = {}
    for condition_id in REC004M_CONDITION_IDS:
        primitive = condition_primitives[condition_id]
        for dataset_name in REC004M_DATASETS:
            examples = datasets[dataset_name]
            with torch.no_grad():
                point = rec004l.compute_interaction_diagnostic_point(
                    core, primitive, reference_primitive, examples
                )
            point.update(
                {
                    "condition_id": condition_id,
                    "subcomponents_rolled_back": list(
                        REC004M_CONDITION_TO_SUBCOMPONENTS.get(condition_id, ())
                    ),
                    "ffn_rolled_back": REC004M_FFN_ROLLED_BACK.get(condition_id, False),
                    "dataset": dataset_name,
                }
            )
            points[f"{condition_id}:{dataset_name}"] = point
    return points


def run_decomposition_parity_check(
    core: Any,
    f_allv_primitive: Any,
    late_sd: dict[str, torch.Tensor],
    early_sd: dict[str, torch.Tensor],
    datasets: dict[str, list[Example]],
) -> dict[str, Any]:
    """Positive-control gate (task doc Section 4, new for this task):
    `F_ALLV`'s row-slice composition (CONTENT_PREP + V_PROJECTION +
    ATTN_OUT_PROJ + SCORE_PROJECTION_CONTROL, all rolled back) must
    reproduce, exactly, an INDEPENDENTLY built reference that rolls back
    REC-004K's whole (un-sliced) VALUE_OUTPROJ group jointly with FFN_BLOCK
    -- literally the same call REC-004L made for its own `F_V`. A mismatch
    means the row-slice decomposition is incomplete or incorrect."""
    resid_audit._assert_eval_only()
    old_key_groups = rec004k.partition_state_dict_keys(list(late_sd.keys()))
    reference_primitive = rec004k._build_rollback_primitive(
        core, late_sd, early_sd, ("FFN_BLOCK", "VALUE_OUTPROJ"), old_key_groups
    )
    return rec004k.run_component_decomposition_parity_check(
        core, f_allv_primitive, reference_primitive, datasets
    )


def cross_check_f_allv_against_rec004l_f_v(condition_points: dict[str, Any]) -> dict[str, Any]:
    """`F_ALLV` and REC-004L's own `F_V` roll back the same real parameters
    (FFN_BLOCK + all of VALUE_OUTPROJ) from the same checkpoints on the same
    three shared datasets -- their oracle EM must match exactly.
    Informational only; never gates this task's own decision."""
    rec004l_run_dir = Path("runs/phase_b_b2_model_bundle_recovery/rec004l/run_001")
    path = rec004l_run_dir / "condition_points_summary.json"
    if not path.is_file():
        return {"status": "REC004L_ARTIFACT_UNAVAILABLE"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    shared_datasets = (
        REC004M_CLEAN_V2_DATASET, REC004M_PROBE_DATASET, REC004M_DOWNSTREAM_PROBE_DATASET,
    )
    for dataset_name in shared_datasets:
        recorded_em = recorded.get(f"F_V:{dataset_name}", {}).get("oracle_sequence_exact_match")
        recomputed_em = condition_points.get(f"F_ALLV:{dataset_name}", {}).get(
            "oracle_sequence_exact_match"
        )
        if (
            recorded_em is None
            or recomputed_em is None
            or abs(recorded_em - recomputed_em) >= 1e-9
        ):
            mismatches.append(
                {
                    "dataset": dataset_name,
                    "rec004l_f_v_oracle_em": recorded_em,
                    "rec004m_f_allv_oracle_em": recomputed_em,
                }
            )
    return {"status": "VERIFIED" if not mismatches else "MISMATCH", "mismatches": mismatches}


def build_regression_check(condition_points: dict[str, Any]) -> dict[str, Any]:
    """Descriptive-only check: confirms no FFN-anchored subcomponent
    condition breaks a position R0 already got right, at positions other
    than the two decisive ones (4, 5). Never gates the decision below."""
    report: dict[str, Any] = {}
    for condition_id in REC004M_CONDITION_IDS:
        if condition_id == "R0":
            continue
        for dataset_name in REC004M_DATASETS:
            r0_row = condition_points.get(f"R0:{dataset_name}", {})
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            regressed: list[dict[str, Any]] = []
            for pos in REC004M_REGRESSION_POSITIONS:
                pos_key = str(pos)
                r0_acc = r0_row.get("per_output_position", {}).get(pos_key, {}).get(
                    "oracle_accuracy"
                )
                acc = row.get("per_output_position", {}).get(pos_key, {}).get("oracle_accuracy")
                if r0_acc is not None and acc is not None and acc < r0_acc:
                    regressed.append(
                        {"position": pos, "r0_accuracy": r0_acc, "condition_accuracy": acc}
                    )
            report[f"{condition_id}:{dataset_name}"] = {
                "regressed_positions": regressed,
                "any_regression": bool(regressed),
            }
    return report


# =============================================================================
# Decision, centered on output position 4 (task doc Section 6).
# =============================================================================


def build_value_path_subcomponent_decision(
    condition_points: dict[str, Any], parity_check: dict[str, Any]
) -> dict[str, Any]:
    if parity_check["status"] != "VERIFIED":
        return {
            "task_id": REC004M_TASK_ID,
            "label": "VALUE_PATH_DECOMPOSITION_PARITY_FAILED_STOP",
            "sufficient_subcomponents": [],
            "f_allv_passes": False,
            "note": (
                "F_ALLV did not reproduce the independently-built FFN_BLOCK+whole-"
                "VALUE_OUTPROJ reference within tolerance on at least one dataset -- "
                "the row-slice decomposition of VALUE_OUTPROJ is incomplete or "
                "incorrect; F_C/F_V/F_O/F_QK/F_ALLV results are recorded but NOT "
                "interpreted as a causal finding"
            ),
        }

    pos_key = str(REC004M_POSITION_MAIN_RESIDUAL)

    def _passes(condition_id: str) -> bool:
        for dataset_name in REC004M_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            em = row.get("oracle_sequence_exact_match")
            pos4 = row.get("per_output_position", {}).get(pos_key, {}).get("oracle_accuracy")
            if em is None or pos4 is None:
                return False
            if em < REC004M_ROLLBACK_EM_THRESHOLD or pos4 < REC004M_ROLLBACK_POSITION_ACC_THRESHOLD:
                return False
        return True

    sufficient = [cid for cid in REC004M_PAIR_CONDITION_IDS if _passes(cid)]
    f_allv_passes = _passes("F_ALLV")

    if len(sufficient) == 1:
        partner = REC004M_SUBCOMPONENT_TO_LABEL_PARTNER[sufficient[0]]
        label = f"FFN_{partner}_INTERACTION_SUFFICIENT"
    elif len(sufficient) > 1:
        label = "MULTIPLE_VALUE_SUBPATHS_SUFFICIENT"
    elif f_allv_passes:
        label = "DISTRIBUTED_WITHIN_VALUE_PATH"
    else:
        label = "VALUE_PATH_DECOMPOSITION_INCOMPLETE"

    return {
        "task_id": REC004M_TASK_ID,
        "label": label,
        "sufficient_subcomponents": [
            REC004M_SUBCOMPONENT_TO_LABEL_PARTNER[cid] for cid in sufficient
        ],
        "sufficient_condition_ids": sufficient,
        "f_allv_passes": f_allv_passes,
        "oracle_em_threshold": REC004M_ROLLBACK_EM_THRESHOLD,
        "position4_accuracy_threshold": REC004M_ROLLBACK_POSITION_ACC_THRESHOLD,
        "content_prep_shared_with_score_path_caveat": (
            "CONTENT_PREP feeds the same kv tensor both V_PROJECTION and "
            "SCORE_PROJECTION_CONTROL (the real, learnable K path) read from -- "
            "freezing CONTENT_PREP in a future repair would also constrain what "
            "the real attention SCORE depends on, not just O1's oracle-substituted "
            "output. Recorded regardless of which label above is reached; this "
            "task does not propose a full VALUE-path freeze even under "
            "DISTRIBUTED_WITHIN_VALUE_PATH."
        ),
    }


# =============================================================================
# Stage E -- conditional success-model safety check. Reuses REC-004K's own
# `compute_rollback_diagnostic_point` (J0 AND O1) unmodified.
# =============================================================================


def run_safety_check(
    core: Any, decision: dict[str, Any], datasets: dict[str, list[Example]]
) -> dict[str, Any]:
    sufficient_ids = decision.get("sufficient_condition_ids", [])
    if decision["label"].startswith("FFN_") and len(sufficient_ids) == 1:
        localized_condition_id = sufficient_ids[0]
    else:
        return {
            "task_id": REC004M_TASK_ID,
            "status": "NOT_EXECUTED_NOT_LOCALIZED_TO_ONE_SUBCOMPONENT",
            "decision_label": decision["label"],
            "per_init": {},
        }

    subcomponents = REC004M_CONDITION_TO_SUBCOMPONENTS[localized_condition_id]
    per_init: dict[str, Any] = {}
    for init_id, late_step in REC004M_SAFETY_TARGETS:
        late_primitive, late_info = rec004j._load_and_verify_primitive(core, init_id, late_step)
        early_primitive, early_info = rec004j._load_and_verify_primitive(
            core, init_id, REC004M_EARLY_STEP
        )
        if late_primitive is None or early_primitive is None:
            per_init[init_id] = {
                "status": "SOURCE_ARTIFACT_UNAVAILABLE",
                "late_load_info": late_info, "early_load_info": early_info,
            }
            continue

        keys = list(late_primitive.state_dict().keys())
        key_groups = build_value_path_subcomponent_key_groups(keys)
        late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
        early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}
        rolled_back_primitive = _build_subcomponent_rollback_primitive(
            core, late_sd, early_sd, subcomponents, key_groups, ffn_rolled_back=True
        )

        per_dataset: dict[str, Any] = {}
        max_j0_drop = 0.0
        for dataset_name in REC004M_DATASETS:
            examples = datasets[dataset_name]
            with torch.no_grad():
                baseline_point = rec004k.compute_rollback_diagnostic_point(
                    core, late_primitive, examples
                )
                rollback_point = rec004k.compute_rollback_diagnostic_point(
                    core, rolled_back_primitive, examples
                )
            j0_delta = (
                rollback_point["j0_sequence_exact_match"]
                - baseline_point["j0_sequence_exact_match"]
            )
            o1_delta = (
                rollback_point["oracle_sequence_exact_match"]
                - baseline_point["oracle_sequence_exact_match"]
            )
            per_dataset[dataset_name] = {
                "baseline_j0_em": baseline_point["j0_sequence_exact_match"],
                "rollback_j0_em": rollback_point["j0_sequence_exact_match"],
                "j0_em_delta": j0_delta,
                "baseline_oracle_em": baseline_point["oracle_sequence_exact_match"],
                "rollback_oracle_em": rollback_point["oracle_sequence_exact_match"],
                "oracle_em_delta": o1_delta,
            }
            max_j0_drop = min(max_j0_drop, j0_delta)

        degraded = -max_j0_drop >= REC004M_SAFETY_EM_DEGRADATION_THRESHOLD
        per_init[init_id] = {
            "status": "EXECUTED",
            "late_step": late_step,
            "late_load_info_status": late_info["status"],
            "early_load_info_status": early_info["status"],
            "localized_condition_id": localized_condition_id,
            "per_dataset": per_dataset,
            "max_j0_em_drop": -max_j0_drop,
            "degradation_threshold": REC004M_SAFETY_EM_DEGRADATION_THRESHOLD,
            "classification": "DEGRADATION_OBSERVED" if degraded else "NO_DEGRADATION_OBSERVED",
        }

    any_degraded = any(
        row.get("classification") == "DEGRADATION_OBSERVED" for row in per_init.values()
    )
    return {
        "task_id": REC004M_TASK_ID,
        "status": "EXECUTED",
        "localized_condition_id": localized_condition_id,
        "localized_subcomponents": list(subcomponents),
        "per_init": per_init,
        "any_degradation_observed": any_degraded,
    }


# =============================================================================
# next_training_repair_contract.md
# =============================================================================


def build_next_training_repair_contract(
    decision: dict[str, Any], safety_check: dict[str, Any]
) -> str:
    lines: list[str] = [
        "# B-C005REC-004M: Next Training Repair Contract",
        "",
        "This file is a PROPOSAL record, not an authorization. No training is "
        "started by this task or by this file's existence.",
        "",
        f"## Stage D result: `{decision['label']}`",
        "",
    ]
    localized = decision["label"].startswith("FFN_") and (
        len(decision.get("sufficient_condition_ids", [])) == 1
    )
    if not localized:
        lines += [
            f"The FFN-value-path interaction did NOT localize to exactly one "
            f"subcomponent (label: `{decision['label']}`). Per the task's own "
            "charter, no two-stage training repair is proposed here.",
            "",
        ]
        if decision["label"] == "DISTRIBUTED_WITHIN_VALUE_PATH":
            lines += [
                "`F_ALLV` (all four subcomponents jointly) is sufficient, but no "
                "single one is. This task does NOT propose freezing the entire "
                "VALUE path as the next repair: `CONTENT_PREP` feeds the same "
                "`kv` tensor both `V_PROJECTION` and `SCORE_PROJECTION_CONTROL` "
                "(the real, learnable K path) read from, so freezing it would "
                "also constrain what the real attention SCORE depends on -- "
                "directly undermining the attention-score learning a repair is "
                "meant to preserve. A further, narrower diagnostic (not "
                "authorized here) would be needed before any repair proposal.",
                "",
            ]
        lines += ["`status: NOT_PROPOSED`", ""]
        return "\n".join(lines) + "\n"

    subcomponent = decision["sufficient_subcomponents"][0]
    condition_id = decision["sufficient_condition_ids"][0]
    safety_status = safety_check.get("status")
    any_degraded = safety_check.get("any_degradation_observed", False)

    lines += [
        f"The FFN-value-path interaction localizes to exactly one subcomponent: "
        f"**`{subcomponent}`** (condition `{condition_id}`).",
        "",
        "## Stage E safety check",
        f"- status: {safety_status}",
    ]
    if safety_status == "EXECUTED":
        for init_id, row in safety_check["per_init"].items():
            lines.append(
                f"  - {init_id}@{row.get('late_step')}: "
                f"{row.get('classification')} (max J0 EM drop="
                f"{row.get('max_j0_em_drop'):.4f}, threshold="
                f"{row.get('degradation_threshold')})"
            )
        lines.append(f"- any_degradation_observed: {any_degraded}")
    lines.append("")

    if any_degraded:
        lines += [
            "**Caution**: Stage E observed a J0 EM drop at or above the "
            f"pre-registered {REC004M_SAFETY_EM_DEGRADATION_THRESHOLD} threshold "
            "for at least one already-succeeding init when the same rollback was "
            "applied to its own step=6000 state. This does not block the proposal "
            "below (per the task's own charter, no adoption decision is made "
            "here), but it is a disclosed risk factor any actual 004N task must "
            "address -- e.g. by also measuring I04/I05 under the real Phase 2 "
            "recipe before committing to it, not just this in-memory diagnostic.",
            "",
        ]

    lines += [
        "## Proposed two-stage training design (status: PROPOSED_NOT_AUTHORIZED)",
        "",
        "```",
        "Phase 1:",
        "  step 0 -> 6000",
        "  current REC-004D/G recipe, unchanged.",
        "",
        "Phase 2:",
        "  step 6001 -> a fixed, finite target step",
        f"  FFN_BLOCK + {subcomponent} held at their step=6000 values.",
        "  Q/K (SCORE_PROJECTION_CONTROL), position bias, QUERY_RESIDUAL_PATH, "
        "POST_ATTN_NORM, and READOUT continue training normally.",
        "```",
        "",
        "step=6000 is treated as a mechanistic transition boundary, established "
        "by independent diagnostic evidence (I03@6000 was oracle-sufficient per "
        "REC-004K Stage A; oracle sufficiency was lost by step=17500) -- NOT as a "
        "result-selected best checkpoint chosen after seeing this task's own "
        "numbers.",
        "",
        "Running any such training is a SEPARATE task (tentatively `B-C005REC-"
        "004N`) requiring its own explicit user instruction and its own fixed, "
        "pre-registered protocol. This file's existence is not authorization to "
        "implement or train it.",
        "",
        "`status: PROPOSED_NOT_AUTHORIZED`",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Report.
# =============================================================================


def build_report_markdown(
    condition_points: dict[str, Any],
    forward_graph_check: dict[str, Any],
    qk_rollback_check: dict[str, Any],
    parity_check: dict[str, Any],
    decision: dict[str, Any],
    cross_check: dict[str, Any],
    dataset_details: dict[str, Any],
    safety_check: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# {REC004M_TASK_ID}: I03 FFN-Value-Path Subcomponent Attribution & "
        "Freeze-Repair Contract",
        "",
        "Zero new optimizer updates. Forward-only, O1 (oracle-attention) "
        "condition; every rollback is an in-memory, evaluation-only merged "
        "state dict, with the fused in_proj Q/K/V tensor split by row-slice "
        "(not swapped whole) for V_PROJECTION/SCORE_PROJECTION_CONTROL.",
        "",
        "## Datasets",
        "",
        f"- `{REC004M_CLEAN_V2_DATASET}` (n=230) and `{REC004M_PROBE_DATASET}` "
        f"(n=512), `{REC004M_DOWNSTREAM_PROBE_DATASET}` (n="
        f"{dataset_details[REC004M_DOWNSTREAM_PROBE_DATASET].get('n')}): "
        "byte-identical reuse from REC-004K/REC-004L.",
        f"- `{REC004M_NEW_PROBE_SPLIT}` (n="
        f"{dataset_details[REC004M_NEW_PROBE_SPLIT]['n']}): new, disjoint "
        f"(disjointness_check.disjoint="
        f"{dataset_details[REC004M_NEW_PROBE_SPLIT]['disjointness_check']['disjoint']}), "
        "development-exposed diagnostic set, frozen digest="
        f"{dataset_details[REC004M_NEW_PROBE_SPLIT]['frozen_dataset_digest_sha256']}",
        "",
        f"- Forward-graph invariance check (position bias + Q/K slices, "
        f"perturbation-based): invariant={forward_graph_check['invariant']}, "
        f"weights_restored_exactly={forward_graph_check['weights_restored_exactly']}",
        f"- QK rollback invariance check (F_QK vs F, real weights): "
        f"invariant={qk_rollback_check['invariant']}, "
        f"max_abs_logit_diff={qk_rollback_check['max_abs_logit_diff']}",
        f"- Cross-check vs REC-004L's own F_V numbers: {cross_check.get('status')}",
        "",
        "## Stage B/D -- FFN-anchored subcomponent conditions (I03@17500 base, O1)",
        "",
        "| condition | dataset | n | O1 EM | pos4 acc | pos5 acc | hidden diff | logit diff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition_id in REC004M_CONDITION_IDS:
        for dataset_name in REC004M_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            pos4 = row.get("per_output_position", {}).get(
                str(REC004M_POSITION_MAIN_RESIDUAL), {}
            ).get("oracle_accuracy")
            pos5 = row.get("per_output_position", {}).get(
                str(REC004M_POSITION_SYMMETRIC_CONTROL), {}
            ).get("oracle_accuracy")
            lines.append(
                f"| {condition_id} | {dataset_name} | {row.get('n')} | "
                f"{row.get('oracle_sequence_exact_match'):.4f} | "
                f"{pos4:.4f} | {pos5:.4f} | "
                f"{row.get('max_abs_hidden_state_diff_vs_early'):.6f} | "
                f"{row.get('max_abs_logit_diff_vs_early'):.6f} |"
            )

    lines += [
        "",
        "## Decomposition parity check (F_ALLV vs FFN_BLOCK+whole-VALUE_OUTPROJ)",
        f"- status: **{parity_check['status']}**",
    ]
    for dataset_name, row in parity_check["per_dataset"].items():
        lines.append(
            f"  - {dataset_name}: max_abs_logit_diff={row['max_abs_logit_diff']:.6f}, "
            f"predictions_match={row['predictions_match']}, passed={row['passed']}"
        )

    lines += [
        "",
        "## Value-path subcomponent decision",
        f"- label: **{decision['label']}**",
        f"- sufficient subcomponents: {decision.get('sufficient_subcomponents', [])}",
        f"- F_ALLV passes: {decision.get('f_allv_passes')}",
        "",
        "## Stage E safety check",
        f"- status: {safety_check.get('status')}",
    ]
    if safety_check.get("status") == "EXECUTED":
        for init_id, row in safety_check["per_init"].items():
            lines.append(f"  - {init_id}@{row.get('late_step')}: {row.get('classification')}")

    lines += [
        "",
        "## Fixed non-adoption fields",
        "new_optimizer_updates=0, selected_init=null, selected_step=null, "
        "selected_value_subcomponent=null, child_bundle=null, "
        "rg3_recheck=NOT_EXECUTED, rec005_eligible=false",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Full task orchestration.
# =============================================================================


def run_mirror_ffn_value_path_subcomponent_attribution_task(
    config: MirrorFfnValuePathSubcomponentAttributionConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_ffn_value_path_subcomponent_attribution_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != REC004M_SEED:
        raise ValueError(
            f"B-C005REC-004M reads REC-004D/H/K/L artifacts pre-registered under "
            f"seed {REC004M_SEED}; got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    # Stage C: datasets are generated and their digests frozen to disk BEFORE
    # any condition/rollback point below is computed (pre-registration).
    datasets, dataset_details = build_stage_c_datasets(seed)
    for dataset_name, detail in dataset_details.items():
        _write_json(output_dir / f"{dataset_name}_manifest.json", detail)

    late_primitive, late_load_info = rec004j._load_and_verify_primitive(
        core, REC004M_DECISIVE_INIT, REC004M_LATE_STEP
    )
    early_primitive, early_load_info = rec004j._load_and_verify_primitive(
        core, REC004M_DECISIVE_INIT, REC004M_EARLY_STEP
    )
    checkpoint_load_info = {
        str(REC004M_LATE_STEP): late_load_info,
        str(REC004M_EARLY_STEP): early_load_info,
    }
    _write_json(output_dir / "checkpoint_load_info.json", checkpoint_load_info)

    all_keys = list(late_primitive.state_dict().keys())
    key_groups = build_value_path_subcomponent_key_groups(all_keys)
    _write_json(output_dir / "value_path_subcomponent_key_groups.json", key_groups)

    probe_examples_for_audit = datasets[REC004M_PROBE_DATASET][:REC004M_INVARIANCE_CHECK_EXAMPLES]
    op = get_operation(REC004M_TARGET_OPERATION)
    n = REC004M_TARGET_LENGTH
    content_lengths = [n] * len(probe_examples_for_audit)
    output_lengths = [op.output_length(n)] * len(probe_examples_for_audit)
    batch_input = collate_content_only_batch(
        probe_examples_for_audit, core.tokens, device=core.device
    )
    with torch.no_grad():
        h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
        forward_graph_check = rec004k.verify_o1_invariance_to_excluded_params(
            late_primitive, h_content, content_lengths, output_lengths
        )
    _write_json(output_dir / "forward_graph_invariance_check.json", forward_graph_check)

    condition_points: dict[str, Any] | None = None
    parity_check: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    cross_check: dict[str, Any] | None = None
    regression_check: dict[str, Any] | None = None
    qk_rollback_check: dict[str, Any] | None = None
    safety_check: dict[str, Any] = {
        "task_id": REC004M_TASK_ID,
        "status": "NOT_EXECUTED_STAGE_D_NOT_RUN",
        "per_init": {},
    }
    stage_status = "NOT_EXECUTED_FORWARD_GRAPH_INVARIANCE_CHECK_FAILED"

    if forward_graph_check["invariant"] and forward_graph_check["weights_restored_exactly"]:
        late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
        early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}

        with torch.no_grad():
            qk_rollback_check = verify_qk_rollback_is_o1_invariant(
                core, late_sd, early_sd, key_groups, h_content, content_lengths, output_lengths
            )
        _write_json(output_dir / "qk_rollback_invariance_check.json", qk_rollback_check)

        condition_primitives = build_condition_primitives(
            core, late_sd, early_sd, key_groups, early_primitive
        )

        with torch.no_grad():
            condition_points = run_stage_conditions(core, condition_primitives, datasets)
        _write_jsonl(output_dir / "condition_points.jsonl", condition_points)
        _write_json(output_dir / "condition_points_summary.json", condition_points)

        cross_check = cross_check_f_allv_against_rec004l_f_v(condition_points)
        _write_json(output_dir / "cross_check_against_rec004l.json", cross_check)

        with torch.no_grad():
            parity_check = run_decomposition_parity_check(
                core, condition_primitives["F_ALLV"], late_sd, early_sd, datasets
            )
        _write_json(output_dir / "decomposition_parity_check.json", parity_check)

        decision = build_value_path_subcomponent_decision(condition_points, parity_check)
        _write_json(output_dir / "value_path_subcomponent_decision.json", decision)

        regression_check = build_regression_check(condition_points)
        _write_json(output_dir / "regression_check.json", regression_check)

        with torch.no_grad():
            safety_check = run_safety_check(core, decision, datasets)
        _write_json(output_dir / "safety_check.json", safety_check)

        stage_status = "EXECUTED"
    else:
        _write_json(output_dir / "safety_check.json", safety_check)

    protected_hashes_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004M_TASK_ID,
        "core_canonical_state_hash_before": core_hash_before,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == core_hash_before,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
        "checkpoint_load_status": {k: v["status"] for k, v in checkpoint_load_info.items()},
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004M_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    n_predictions = 0
    if condition_points is not None:
        n_predictions = sum(
            row.get("n", 0) for row in condition_points.values() if isinstance(row, dict)
        ) * 2
    if safety_check.get("status") == "EXECUTED":
        # baseline (J0+O1) + rollback (J0+O1) forward passes, per (init, dataset).
        n_predictions += sum(
            len(datasets[ds])
            for row in safety_check["per_init"].values()
            for ds in row.get("per_dataset", {})
        ) * 4
    cost_accounting = {
        "task_id": REC004M_TASK_ID,
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

    protocol = {
        "task_id": REC004M_TASK_ID,
        "source_task_ids": list(REC004M_SOURCE_TASK_IDS),
        "contract_file": str(REC004M_CONTRACT_FILE),
        "decisive_init": REC004M_DECISIVE_INIT,
        "early_step": REC004M_EARLY_STEP,
        "late_step": REC004M_LATE_STEP,
        "position_main_residual": REC004M_POSITION_MAIN_RESIDUAL,
        "position_symmetric_control": REC004M_POSITION_SYMMETRIC_CONTROL,
        "condition_ids": list(REC004M_CONDITION_IDS),
        "pair_condition_ids": list(REC004M_PAIR_CONDITION_IDS),
        "datasets": list(REC004M_DATASETS),
        "safety_targets": [{"init_id": i, "step": s} for i, s in REC004M_SAFETY_TARGETS],
        "forbidden": [
            "new optimizer updates", "any I03/I04/I05 checkpoint file mutation",
            "any 2- or 3-subcomponent combination beyond the fixed set",
            "FFN retraining", "an actual VALUE-path freeze training run",
            "feeding pi_n into a model input", "extending position bias", "Core change",
            "candidate adoption", "child assembly", "RG3 recheck",
            "starting B-C005REC-005",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE" if stage_status == "EXECUTED" else "PARTIAL",
        "forward_graph_invariance_status": (
            "VERIFIED"
            if forward_graph_check["invariant"] and forward_graph_check["weights_restored_exactly"]
            else "FAILED"
        ),
        "qk_rollback_invariance_status": (
            "VERIFIED" if qk_rollback_check and qk_rollback_check["invariant"] else None
        ),
        "cross_check_status": cross_check.get("status") if cross_check else None,
        "stage_status": stage_status,
        "decomposition_parity_status": parity_check["status"] if parity_check is not None else None,
        "value_path_subcomponent_decision": decision["label"] if decision is not None else None,
        "safety_check_status": safety_check.get("status"),
        "safety_check_any_degradation_observed": safety_check.get("any_degradation_observed"),
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "selected_value_subcomponent": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": cost_accounting,
        "wall_clock_seconds": time.time() - start,
    }
    _write_json(output_dir / "summary.json", result)

    if condition_points is not None and decision is not None and parity_check is not None:
        report_md = build_report_markdown(
            condition_points, forward_graph_check, qk_rollback_check or {}, parity_check,
            decision, cross_check or {"status": "NOT_CHECKED"}, dataset_details, safety_check,
        )
        contract_md = build_next_training_repair_contract(decision, safety_check)
    else:
        report_md = (
            f"# {REC004M_TASK_ID}: I03 FFN-Value-Path Subcomponent Attribution & "
            f"Freeze-Repair Contract\n\nStage B/D NOT executed: {stage_status}.\n"
        )
        contract_md = (
            "# B-C005REC-004M: Next Training Repair Contract\n\n"
            f"Not generated: Stage B/D did not execute ({stage_status}).\n\n"
            "`status: NOT_PROPOSED`\n"
        )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "next_training_repair_contract.md").write_text(contract_md, encoding="utf-8")
    return result
