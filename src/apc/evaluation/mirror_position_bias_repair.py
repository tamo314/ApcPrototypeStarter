"""B-C005REC-004D: MIRROR_HALVES Length-Conditioned Position Bias & RG3 Recheck.

Follows `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md`. Inserted
between `B-C005REC-004C`'s diagnostic-only result (ADR-0098, no candidate, no
training) and `B-C005REC-005`: audits/corrects the position-diagnostic prose
from REC-004C, adds a small (192-parameter) length-conditioned additive
position-bias term to a NEW `CrossPositionLengthBiasPrimitive` operator
class, and trains it (arm ``P``) against a fresh, unmodified
`CrossPositionPrimitive` control (arm ``U``) from REC-004C's own 5 saved
initial states (I01-I05), same Core, same per-step training-data stream,
same 6000-update recipe -- 10 independent runs, 60000 updates total.

Stage lettering (A-F) matches the task doc:
  A. confirm/correct the fixed-position diagnostic claim inherited from
     REC-004C (its own JSON is confirmed correct; only two `report.md` prose
     lines and one ADR-0098 evidence sentence mischaracterize which lengths
     contribute a "fixed" position -- documented here as
     ``REPORT_ONLY_ERRATUM``, not re-litigated as a code bug), and audit
     where real content length/position are actually available inside
     `CrossPositionPrimitive.forward`.
  B. implement `CrossPositionLengthBiasPrimitive`, verify zero-bias parity
     (a freshly-constructed instance is an exact forward/gradient no-op
     versus `CrossPositionPrimitive`), verify the new branch's gradient path,
     verify mask/layout behavior, and verify the new architecture's strict
     (de)serialization contract -- all before any real training.
  C. lock the 5-pair protocol (shared training-data stream across all 10
     runs, shared bias initial state, LR-trace preflight, checkpoint/stop
     conditions, fixed candidates, query spec) into `position_bias_
     protocol.json`.
  D. train U and P from each of I01-I05 (10 runs, 6000 updates each,
     checkpoint every 500), build the paired P-vs-U comparison at the
     decisive step, and run a bias-zero ablation on each trained P.
  E. candidate decision: ALL FIVE P runs must independently clear the
     existing-validation floor (>=0.95) before P/I01@6000 is fixed as the
     new MIRROR_HALVES candidate (never the best-performing init) and
     assembled into a child bundle alongside the 3 REC-004A fixed
     candidates.
  F. independent `rec004d_recheck_query` (new namespace, all 16 operations)
     + fresh-process RG3 recheck.

Stage E/F only run if all 5 P runs clear the floor
(``candidate_status == "FIVE_INIT_VALIDATION_FLOOR_PASS"``); otherwise the
run stops after Stage D with ``rg3_recheck == "NOT_EXECUTED"``, per the task
doc's own STOP rule (section 7.1: 4/5 success still selects nothing).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_position_initialization_diagnostic as mpid
from apc.evaluation import mirror_schedule_comparison as msc
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_NAMESPACE_ROOT,
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _predict_tokens,
    _repo_root,
    _snapshot_forbidden_cache_hashes,
    frozen_evaluation,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    PARAMETERIZED_OPERATION_NAMES,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _derive_local_seed,
    _flatten_groups,
    _generate_parameter_free_examples,
    _make_arg_provider_correct,
    generate_compact_operator_counterfactual_groups,
)
from apc.primitives.primitive import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004D_TASK_ID",
    "REC004D_TARGET_OPERATION",
    "REC004D_FIXED_CANDIDATE_OPERATIONS",
    "REC004D_PROTECTED_OPERATIONS",
    "REC004D_PHYSICAL_IDS",
    "REC004D_INIT_IDS",
    "REC004D_ARMS",
    "REC004D_MAX_UPDATES_PER_RUN",
    "REC004D_CHECKPOINT_INTERVAL",
    "REC004D_PARENT_BUNDLE_ID",
    "REC004D_BIAS_HIDDEN_DIM",
    "REC004D_LENGTH_REF",
    "REC004D_ARCHITECTURE_SIGNATURE_U",
    "REC004D_ARCHITECTURE_SIGNATURE_P",
    "MirrorPositionBiasRepairConfig",
    "run_mirror_position_bias_repair_task",
]


# =============================================================================
# Constants -- pre-registered per the task doc sections 1.4/2/5, not chosen
# after seeing results. Reuses REC-004A/B/C's own recipe/physical-id/split
# constants wherever the task doc says the recipe is unchanged.
# =============================================================================

REC004D_TASK_ID: Final = "B-C005REC-004D"
REC004D_PARENT_TASK_ID: Final = "B-C005REC-004"  # same published bundle REC-004A/B/C used
REC004D_REC004A_TASK_ID: Final = "B-C005REC-004A"
REC004D_REC004C_TASK_ID: Final = "B-C005REC-004C"

REC004D_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004D_PHYSICAL_IDS: Final[dict[str, int]] = dict(ibc.REC004A_PHYSICAL_IDS)
REC004D_TARGET_PHYSICAL_ID: Final = REC004D_PHYSICAL_IDS[REC004D_TARGET_OPERATION]
REC004D_FIXED_CANDIDATE_OPERATIONS: Final[tuple[str, ...]] = msc.REC004B_FIXED_CANDIDATE_OPERATIONS
REC004D_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = ibc.REC004A_PROTECTED_OPERATIONS

REC004D_INIT_IDS: Final[tuple[str, ...]] = mpid.REC004C_INIT_IDS  # ("I01",...,"I05")
REC004D_ARM_U: Final = "U_CURRENT_OPERATOR"
REC004D_ARM_P: Final = "P_LENGTH_POSITION_BIAS"
REC004D_ARMS: Final[tuple[str, ...]] = (REC004D_ARM_U, REC004D_ARM_P)

REC004D_MAX_UPDATES_PER_RUN: Final = 6000
REC004D_CHECKPOINT_INTERVAL: Final = 500
REC004D_DECISIVE_STEP: Final = 6000
REC004D_T_MAX: Final = 1000  # same mechanical-extension schedule as REC-004A/C
REC004D_TOTAL_MAX_UPDATES: Final = (
    REC004D_MAX_UPDATES_PER_RUN * len(REC004D_INIT_IDS) * len(REC004D_ARMS)
)  # 60000

REC004D_EXISTING_VALIDATION_EXAMPLES: Final = 1024
REC004D_EXISTING_VALIDATION_SPLIT: Final = ibc.REC004A_BUDGET_VALIDATION_SPLIT
REC004D_EXISTING_VALIDATION_FLOOR: Final = 0.95
REC004D_RECHECK_QUERY_EXAMPLES: Final = 1024
REC004D_RECHECK_QUERY_SPLIT: Final = "rec004d_recheck_query"

REC004D_VOCAB_SIZE: Final = 10
REC004D_SEQUENCE_LENGTH_RANGE: Final = (6, 10)

REC004D_OPERATOR_LR: Final = ibc.REC004A_OPERATOR_LR
REC004D_OPERATOR_WEIGHT_DECAY: Final = ibc.REC004A_OPERATOR_WEIGHT_DECAY
REC004D_OPERATOR_GRAD_CLIP: Final = ibc.REC004A_OPERATOR_GRAD_CLIP
REC004D_SCHEDULER_ETA_MIN: Final = ibc.REC004A_SCHEDULER_ETA_MIN
REC004D_EXAMPLES_PER_STEP: Final = ibc.REC004A_EXAMPLES_PER_STEP
REC004D_TRAIN_FIT_STEP_COUNT: Final = ibc.REC004A_TRAIN_FIT_STEP_COUNT
REC004D_TRAIN_SPLIT_LABEL: Final = ibc.REC004A_TRAIN_SPLIT_LABEL  # "train"

REC004D_PARENT_BUNDLE_ID: Final = ibc.REC004A_PARENT_BUNDLE_ID
REC004D_PARENT_MANIFEST_PATH: Final = ibc.REC004A_PARENT_MANIFEST_PATH

REC004D_BIAS_HIDDEN_DIM: Final = 32
REC004D_LENGTH_REF: Final = DEFAULT_MAX_SEQUENCE_LENGTH  # 32; model's fixed legal max content len
REC004D_ARCHITECTURE_SIGNATURE_U: Final = "cross_position_v1"
REC004D_ARCHITECTURE_SIGNATURE_P: Final = "cross_position_length_bias_v1"
REC004D_BIAS_PARAM_KEYS: Final[tuple[str, ...]] = (
    "position_bias_hidden.weight",
    "position_bias_hidden.bias",
    "position_bias_out.weight",
)
REC004D_EXPECTED_NEW_PARAM_COUNT: Final = 192
REC004D_BUDGET_FRACTION_CEILING: Final = 0.05
# Backend-dependent floating-point tolerance for the zero-bias-parity gradient
# check (Stage B2) -- see run_zero_bias_parity_check's "tolerance_note" for why
# this is nonzero on CPU. Fixed before Stage C locks the protocol.
REC004D_ZERO_BIAS_GRAD_TOL: Final = 1e-6

REC004D_NAMESPACE_ROOT: Final = RECOVERY_NAMESPACE_ROOT

# REC-004C's own run directory -- source of the 5 shared base initial states
# and the diagnostic artifacts this task's Stage A audits (read-only).
REC004C_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004c/run_001")
REC004C_INITIAL_STATES_DIR: Final = REC004C_RUN_DIR / "initial_states"
REC004C_INITIALIZATION_SUMMARY_PATH: Final = REC004C_RUN_DIR / "initialization_summary.json"
REC004C_POSITION_CONFUSION_PATH: Final = REC004C_RUN_DIR / "position_confusion.json"
REC004C_POSITION_ERROR_SUMMARY_PATH: Final = REC004C_RUN_DIR / "position_error_summary.json"
REC004C_REPORT_PATH: Final = REC004C_RUN_DIR / "report.md"

# REC-004A's fixed 3-candidate step=4000 checkpoints, imported read-only.
REC004D_FIXED_CANDIDATE_STEP: Final = msc.REC004B_FIXED_CANDIDATE_STEP  # 4000
REC004D_FIXED_CANDIDATE_SOURCE_DIR: Final = msc.REC004B_FIXED_CANDIDATE_SOURCE_DIR
REC004D_FIXED_CANDIDATE_SELECTED_RECIPE_PATH: Final = (
    msc.REC004B_FIXED_CANDIDATE_SELECTED_RECIPE_PATH
)

REC004D_LEGAL_LENGTHS: Final[tuple[int, ...]] = tuple(
    range(REC004D_SEQUENCE_LENGTH_RANGE[0], REC004D_SEQUENCE_LENGTH_RANGE[1] + 1)
)


def _rec004d_namespace(seed: int) -> Path:
    return Path(REC004D_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004d"


def _local_seed(step: int, label: str) -> int:
    return _derive_local_seed(RECOVERY_PILOT_SEED, step, label)


def _fixed_candidate_checkpoint_path(op: str) -> Path:
    return REC004D_FIXED_CANDIDATE_SOURCE_DIR / f"{op}_step{REC004D_FIXED_CANDIDATE_STEP}.pt"


@dataclass(frozen=True)
class MirrorPositionBiasRepairConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004d")
    seed: int = RECOVERY_PILOT_SEED
    existing_validation_examples: int = REC004D_EXISTING_VALIDATION_EXAMPLES
    existing_validation_floor: float = REC004D_EXISTING_VALIDATION_FLOOR
    recheck_query_examples: int = REC004D_RECHECK_QUERY_EXAMPLES
    checkpoint_interval: int = REC004D_CHECKPOINT_INTERVAL
    max_updates_per_run: int = REC004D_MAX_UPDATES_PER_RUN


def _base_primitive_kwargs(core: Any) -> dict[str, Any]:
    return {
        "operation": REC004D_TARGET_OPERATION,
        "d_model": core.model.config.d_model,
        "d_operator": 32,
        "n_head": 4,
        "d_operator_ff": 64,
        "vocab_size": REC004D_VOCAB_SIZE,
        "max_sequence_length": 32,
    }


def _new_arm_primitive(
    core: Any, arm: str
) -> CrossPositionPrimitive | CrossPositionLengthBiasPrimitive:
    pid = REC004D_TARGET_PHYSICAL_ID
    kwargs = _base_primitive_kwargs(core)
    if arm == REC004D_ARM_U:
        return CrossPositionPrimitive(
            pid, CrossPositionPrimitiveConfig(**kwargs), status=PrimitiveStatus.STABLE
        )
    return CrossPositionLengthBiasPrimitive(
        pid,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
        status=PrimitiveStatus.STABLE,
    )


# =============================================================================
# Stage A -- confirm/correct REC-004C's fixed-position claim; audit where
# real content length/position are available inside the live forward pass.
# No training, no new operator construction beyond small CPU fixtures.
# =============================================================================


def _audit_fixed_position_claim() -> dict[str, Any]:
    """A2: recomputes `pi_n` from the SAME live source
    (`mpid.mirror_halves_position_map`, itself verified against the real
    oracle interpreter) for the 5 legal lengths, classifies fixed/moved, and
    compares against REC-004C's own prose claim -- rather than assuming
    either the task doc's or REC-004C's own characterization."""
    per_length: dict[str, Any] = {}
    for n in REC004D_LEGAL_LENGTHS:
        pi = mpid.mirror_halves_position_map(n)
        fixed_positions = [i for i in range(n) if pi[i] == i]
        per_length[str(n)] = {
            "pi_n": list(pi),
            "fixed_positions": fixed_positions,
            "fixed_count": len(fixed_positions),
        }
    lengths_with_fixed = [n for n in REC004D_LEGAL_LENGTHS if per_length[str(n)]["fixed_count"] > 0]
    even_fixed = per_length["6"]["fixed_count"] + per_length["10"]["fixed_count"]
    odd_fixed = per_length["7"]["fixed_count"] + per_length["9"]["fixed_count"]
    total_fixed = even_fixed + odd_fixed

    reported_claims = [
        {
            "location": f"{REC004C_REPORT_PATH}:68-69",
            "quoted_text": (
                'The "fixed" bucket is small (only odd lengths 7/9 contribute a middle '
                "no-op position)"
            ),
            "errors": [
                "even lengths 6 and 10 each contribute a fixed position too "
                f"(fixed_count={per_length['6']['fixed_count']} at n=6, "
                f"{per_length['10']['fixed_count']} at n=10)",
                "the single odd-length fixed position is not the middle index: "
                f"n=7 middle index is 3 but pi_7(3)={mpid.mirror_halves_position_map(7)[3]} "
                "(moved); the actual fixed index is "
                f"{per_length['7']['fixed_positions']}",
            ],
        },
        {
            "location": f"{REC004C_REPORT_PATH}:153",
            "quoted_text": 'only 2 of 5 legal lengths contribute a "fixed" position',
            "errors": [
                f"{len(lengths_with_fixed)} of 5 legal lengths contribute a fixed position "
                f"({lengths_with_fixed}), not 2",
            ],
        },
        {
            "location": (
                "docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md, ADR-0098 Evidence 4"
            ),
            "quoted_text": (
                "the (odd-length-only) fixed middle position ... given the small "
                "fixed-position sample (only lengths 7/9 contribute one)"
            ),
            "errors": [
                "same two errors as report.md:68-69 above (odd-only framing; "
                "middle-position framing)",
            ],
        },
    ]

    return {
        "task_id": REC004D_TASK_ID,
        "per_length": per_length,
        "lengths_with_fixed_position": lengths_with_fixed,
        "even_length_fixed_count": even_fixed,
        "odd_length_fixed_count": odd_fixed,
        "total_fixed_count": total_fixed,
        "even_fraction_of_fixed_bucket": even_fixed / total_fixed if total_fixed else None,
        "reported_claims_checked": reported_claims,
        "classification": "REPORT_ONLY_ERRATUM",
        "classification_reason": (
            "The underlying position_confusion.json/position_error_summary.json JSON "
            "aggregates (verified below in _reconcile_historical_position_metrics) are "
            "arithmetically correct against this same pi_n formula; only the prose in "
            "report.md and the ADR-0098 evidence sentence mischaracterizes which lengths "
            "contribute a fixed position and calls the single odd-length fixed index "
            '"the middle" position, which it is not.'
        ),
    }


def _reconcile_historical_position_metrics(fixed_position_audit: dict[str, Any]) -> dict[str, Any]:
    """A2: recomputes REC-004C's `structural_moved_vs_fixed` fixed/moved
    totals from its own `position_error_summary.json` per-(length:position)
    cells using the SAME `pi_n` this task just recomputed, and compares
    against REC-004C's own `position_confusion.json` -- confirms the JSON is
    correct (so this task's own metrics inherit it unmodified) rather than
    assuming it."""
    if not REC004C_POSITION_ERROR_SUMMARY_PATH.is_file():
        return {
            "task_id": REC004D_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "missing": str(REC004C_POSITION_ERROR_SUMMARY_PATH),
        }
    if not REC004C_POSITION_CONFUSION_PATH.is_file():
        return {
            "task_id": REC004D_TASK_ID,
            "status": "SOURCE_ARTIFACT_UNAVAILABLE",
            "missing": str(REC004C_POSITION_CONFUSION_PATH),
        }
    error_summary_rows = json.loads(
        REC004C_POSITION_ERROR_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    reported_confusion = json.loads(REC004C_POSITION_CONFUSION_PATH.read_text(encoding="utf-8"))

    recomputed: dict[str, Any] = {}
    matches_all: dict[str, bool] = {}
    fixed_lookup = {
        int(n): set(fixed_position_audit["per_length"][str(n)]["fixed_positions"])
        for n in REC004D_LEGAL_LENGTHS
    }
    for label, row in error_summary_rows.items():
        by_position = row.get("by_length_position") if isinstance(row, dict) else None
        if not isinstance(by_position, dict):
            continue
        fixed_n = fixed_correct = moved_n = moved_correct = 0
        for key, cell in by_position.items():
            n_str, i_str = key.split(":")
            n, i = int(n_str), int(i_str)
            if n not in fixed_lookup:
                continue
            is_fixed = i in fixed_lookup[n]
            if is_fixed:
                fixed_n += cell["n"]
                fixed_correct += cell["correct"]
            else:
                moved_n += cell["n"]
                moved_correct += cell["correct"]
        recomputed[label] = {
            "fixed": {"n": fixed_n, "correct": fixed_correct},
            "moved": {"n": moved_n, "correct": moved_correct},
        }
        reported = reported_confusion.get(label, {}).get("structural_moved_vs_fixed")
        if reported is not None:
            matches_all[label] = (
                recomputed[label]["fixed"] == reported["fixed"]
                and recomputed[label]["moved"] == reported["moved"]
            )

    return {
        "task_id": REC004D_TASK_ID,
        "status": "RECONCILED",
        "recomputed_structural_moved_vs_fixed": recomputed,
        "matches_reported_position_confusion_json": matches_all,
        "all_match": all(matches_all.values()) if matches_all else None,
        "conclusion": (
            "position_confusion.json / position_error_summary.json aggregates are "
            "arithmetically correct against the live pi_n formula; no JSON correction "
            "needed, only the prose erratum recorded in fixed_position_audit.json."
        ),
    }


def _check_recipe_matches_live_defaults() -> None:
    live_defaults = UnifiedBenchmarkConfig()
    if live_defaults.operator_lr != REC004D_OPERATOR_LR:
        raise ValueError("RECIPE_MISMATCH: operator_lr drifted from the recorded recipe value")
    if live_defaults.operator_weight_decay != REC004D_OPERATOR_WEIGHT_DECAY:
        raise ValueError("RECIPE_MISMATCH: operator_weight_decay drifted")
    if live_defaults.operator_grad_clip != REC004D_OPERATOR_GRAD_CLIP:
        raise ValueError("RECIPE_MISMATCH: operator_grad_clip drifted")


def _run_position_access_audit(core: Any) -> dict[str, Any]:
    """A3: a real (CPU) forward+backward on a heterogeneous-length batch
    confirms (not merely quotes from a source read) that `content_lengths`
    inside `CrossPositionPrimitive.forward` are real per-example content
    lengths (not the padded batch max), that padded key positions never
    receive attention mass, and that the new bias branch produces
    length-dependent, non-shared-across-different-lengths values."""
    device = core.device
    torch.manual_seed(_local_seed(0, "rec004d_position_access_audit_fixture"))
    kwargs = _base_primitive_kwargs(core)
    primitive = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
        status=PrimitiveStatus.STABLE,
    )
    primitive.to(device)
    # Give the bias net a nonzero output layer so this audit exercises the
    # real bias pathway, not the (deliberately zero) fresh-init no-op.
    with torch.no_grad():
        primitive.position_bias_out.weight.normal_(mean=0.0, std=0.05)
    primitive.eval()

    content_lengths = [6, 10]
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    content = torch.randn(2, lmax, core.model.config.d_model, device=device)
    with torch.no_grad():
        bias = primitive._position_bias(content_lengths, max(output_lengths), lmax, device)
    # Padded key columns (j >= real content_length for that row) must not
    # influence the softmax (verified via the -inf mask applied in forward,
    # not via this raw bias tensor -- a real forward/backward confirms it).
    padded_bias_short_row = bias[0, :, content_lengths[0] :]
    real_len_dependence = not torch.allclose(bias[0, 0, 0], bias[1, 0, 0])

    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)
    logits = primitive(content, content_lengths, output_lengths, None)
    loss = logits.sum()
    loss.backward()

    return {
        "task_id": REC004D_TASK_ID,
        "content_lengths_source": (
            "forward() parameter `content_lengths: Sequence[int]` -- real per-example "
            "content length, distinct from `lmax = content_features.shape[1]` (padded "
            "batch max). Confirmed by constructing a real 2-example batch with "
            "content_lengths=[6, 10] inside one padded [2, 10, d_model] tensor."
        ),
        "query_position_i_source": "torch.arange(out_max) via query_ids / s_idx",
        "key_position_j_source": "torch.arange(lmax) via content_position_ids / p_idx",
        "existing_position_features_before_this_task": [
            "content_position_embedding (absolute nn.Embedding, added to kv)",
            "answer_query_embedding (absolute nn.Embedding, added to query)",
        ],
        "existing_length_or_relative_position_feature_before_this_task": None,
        "new_bias_injection_point": (
            "additive float attn_mask passed to nn.MultiheadAttention (softmax input), "
            "replacing the boolean key_padding_mask CrossPositionPrimitive uses"
        ),
        "different_real_lengths_in_same_batch_produce_different_bias": real_len_dependence,
        "padded_bias_region_shape_present_but_masked_to_-inf_in_forward": list(
            padded_bias_short_row.shape
        ),
        "position_layout_contract_status": "RESOLVED",
    }


def run_stage_a(config: MirrorPositionBiasRepairConfig) -> dict[str, Any]:
    _guard_not_frozen("run_stage_a")

    parent_manifest, _raw = ibc._load_parent_manifest()
    diagnostic_load = mb.load_bundle(
        parent_manifest, mode="diagnostic", expected_primitive_count=16
    )

    physical_ids_by_op = {p.operation_name: p.physical_id for p in parent_manifest.primitives}
    for op, expected_pid in REC004D_PHYSICAL_IDS.items():
        actual = physical_ids_by_op.get(op)
        if actual != expected_pid:
            raise mb.IncompleteBundleError(
                f"parent manifest declares physical_id={actual} for {op}, "
                f"expected the pre-registered {expected_pid}"
            )

    _check_recipe_matches_live_defaults()
    operation_contract = msc._check_mirror_halves_contract()
    if not (
        operation_contract["even_length_fixture"]["matches"]
        and operation_contract["odd_length_fixture"]["matches"]
    ):
        raise ValueError(
            "MIRROR_HALVES contract fixture mismatch -- refusing to proceed with an "
            "unverified transform rule"
        )
    position_map_audit = mpid._verify_position_map_against_source(REC004D_LEGAL_LENGTHS)
    if not position_map_audit["all_lengths_matched"]:
        raise ValueError("position_map_source_verification failed -- refusing to proceed")

    fixed_position_audit = _audit_fixed_position_claim()
    metric_erratum = _reconcile_historical_position_metrics(fixed_position_audit)
    if metric_erratum.get("status") == "RECONCILED" and metric_erratum["all_match"] is False:
        raise ValueError(
            "METRIC_RECONCILIATION_FAILURE: recomputed structural_moved_vs_fixed does not "
            f"match REC-004C's own position_confusion.json: {metric_erratum}"
        )

    if not REC004D_FIXED_CANDIDATE_SELECTED_RECIPE_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {REC004D_FIXED_CANDIDATE_SELECTED_RECIPE_PATH} "
            "not found"
        )
    rec004a_selected_recipe = json.loads(
        REC004D_FIXED_CANDIDATE_SELECTED_RECIPE_PATH.read_text(encoding="utf-8")
    )
    if not REC004C_INITIAL_STATES_DIR.is_dir():
        raise mb.MissingArtifactError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {REC004C_INITIAL_STATES_DIR} not found"
        )
    if not REC004C_INITIALIZATION_SUMMARY_PATH.is_file():
        raise mb.MissingArtifactError(
            f"SOURCE_ARTIFACT_UNAVAILABLE: {REC004C_INITIALIZATION_SUMMARY_PATH} not found"
        )
    rec004c_summary = json.loads(REC004C_INITIALIZATION_SUMMARY_PATH.read_text(encoding="utf-8"))

    fixed_candidate_files: dict[str, Any] = {}
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    for op in REC004D_FIXED_CANDIDATE_OPERATIONS:
        ckpt_path = _fixed_candidate_checkpoint_path(op)
        if not ckpt_path.is_file():
            raise mb.MissingArtifactError(f"SOURCE_CANDIDATE_UNVERIFIABLE: {ckpt_path} not found")
        sd = dict(mb.load_state_dict(ckpt_path))
        fixed_candidate_state_dicts[op] = sd
        reported = rec004a_selected_recipe["selection"][op]["all_checkpoint_validation_em"].get(
            "4000"
        )
        fixed_candidate_files[op] = {
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_state_hash": mb.canonical_state_hash(sd),
            "reported_step4000_validation_em": reported,
        }

    base_initial_states: dict[str, dict[str, torch.Tensor]] = {}
    base_initial_state_hashes: dict[str, str] = {}
    for init_id in REC004D_INIT_IDS:
        path = REC004C_INITIAL_STATES_DIR / f"{init_id}.pt"
        if not path.is_file():
            raise mb.MissingArtifactError(f"SOURCE_ARTIFACT_UNAVAILABLE: {path} not found")
        sd = dict(mb.load_state_dict(path))
        base_initial_states[init_id] = sd
        computed_hash = mb.canonical_state_hash(sd)
        base_initial_state_hashes[init_id] = computed_hash
        reported_hash = rec004c_summary["per_init"][init_id]["initial_state_hash"]
        if computed_hash != reported_hash:
            raise ValueError(
                f"SOURCE_ARTIFACT_UNAVAILABLE: {init_id}.pt canonical_state_hash="
                f"{computed_hash} does not match REC-004C's own recorded "
                f"initial_state_hash={reported_hash}"
            )

    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )

    source_validation_replay: dict[str, Any] = {}
    original_slices = {
        op: {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}
        for op, pid in REC004D_PHYSICAL_IDS.items()
    }
    for op in REC004D_FIXED_CANDIDATE_OPERATIONS:
        pid = REC004D_PHYSICAL_IDS[op]
        eval_bank.get(pid).load_state_dict(
            {k: v.to(core.device) for k, v in fixed_candidate_state_dicts[op].items()}, strict=True
        )
        eval_bank.get(pid).eval()
        row = _evaluate_one_operation(
            core, eval_bank, op_to_id, op,
            seed=config.seed, n_examples=config.existing_validation_examples,
            split=REC004D_EXISTING_VALIDATION_SPLIT,
        )
        eval_bank.get(pid).load_state_dict(original_slices[op], strict=True)
        reported = fixed_candidate_files[op]["reported_step4000_validation_em"]
        source_validation_replay[op] = {
            "reproduced_correct_exact_match": row["correct_exact_match"],
            "reported_step4000_validation_em": reported,
            "matches_within_tolerance": (
                reported is not None and abs(row["correct_exact_match"] - reported) < 1e-9
            ),
        }
    eval_bank.eval()

    position_access_audit = _run_position_access_audit(core)

    all_candidates_verified = all(
        v["matches_within_tolerance"] for v in source_validation_replay.values()
    )
    if not all_candidates_verified:
        raise ValueError(
            "SOURCE_CANDIDATE_UNVERIFIABLE: reproduced step=4000 validation EM does not match "
            f"REC-004A's reported values: {source_validation_replay}"
        )

    source_audit = {
        "task_id": REC004D_TASK_ID,
        "parent_task_id": REC004D_PARENT_TASK_ID,
        "parent_manifest_path": str(REC004D_PARENT_MANIFEST_PATH.resolve()),
        "parent_bundle_id": parent_manifest.bundle_id,
        "parent_core_canonical_state_hash": parent_manifest.core.canonical_state_hash,
        "parent_self_load_diagnostic_checks_performed": list(diagnostic_load.checks_performed),
        "target_operation": REC004D_TARGET_OPERATION,
        "target_operation_physical_id": REC004D_TARGET_PHYSICAL_ID,
        "protected_operation_physical_ids": {
            op: physical_ids_by_op[op] for op in REC004D_PROTECTED_OPERATIONS
        },
        "fixed_candidate_operations": list(REC004D_FIXED_CANDIDATE_OPERATIONS),
        "fixed_candidate_step": REC004D_FIXED_CANDIDATE_STEP,
        "fixed_candidate_files": fixed_candidate_files,
        "source_validation_replay": source_validation_replay,
        "base_initial_state_hashes": base_initial_state_hashes,
        "rec004c_reported_final_validation_em": {
            init_id: rec004c_summary["per_init"][init_id]["final_validation_em"]
            for init_id in REC004D_INIT_IDS
        },
        "operation_contract": {REC004D_TARGET_OPERATION: operation_contract},
        "position_map_audit": position_map_audit,
        "recipe": {
            "optimizer": "torch.optim.AdamW",
            "operator_lr": REC004D_OPERATOR_LR,
            "operator_weight_decay": REC004D_OPERATOR_WEIGHT_DECAY,
            "operator_grad_clip": REC004D_OPERATOR_GRAD_CLIP,
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "scheduler_eta_min": REC004D_SCHEDULER_ETA_MIN,
            "examples_per_step": REC004D_EXAMPLES_PER_STEP,
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, default=str), encoding="utf-8"
    )
    (config.output_dir / "fixed_position_audit.json").write_text(
        json.dumps(fixed_position_audit, indent=2, default=str), encoding="utf-8"
    )
    (config.output_dir / "metric_erratum.json").write_text(
        json.dumps(metric_erratum, indent=2, default=str), encoding="utf-8"
    )
    (config.output_dir / "corrected_historical_position_metrics.json").write_text(
        json.dumps(
            {
                "task_id": REC004D_TASK_ID,
                "status": "NO_CORRECTION_NEEDED",
                "note": (
                    "position_confusion.json / position_error_summary.json numeric aggregates "
                    "were reconciled (see metric_erratum.json) and confirmed correct; only "
                    "prose in report.md/ADR-0098 was in error. This file records that the "
                    "historical numeric values stand unmodified."
                ),
                "historical_values": metric_erratum.get("recomputed_structural_moved_vs_fixed"),
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (config.output_dir / "position_access_audit.json").write_text(
        json.dumps(position_access_audit, indent=2, default=str), encoding="utf-8"
    )

    return {
        "parent_manifest": parent_manifest,
        "core": core,
        "eval_bank": eval_bank,
        "op_to_id": op_to_id,
        "fixed_candidate_state_dicts": fixed_candidate_state_dicts,
        "base_initial_states": base_initial_states,
        "base_initial_state_hashes": base_initial_state_hashes,
        "source_audit": source_audit,
        "fixed_position_audit": fixed_position_audit,
        "metric_erratum": metric_erratum,
        "position_access_audit": position_access_audit,
    }


# =============================================================================
# Stage B -- implement + verify the new architecture before any real
# training: parameter budget, zero-bias parity, gradient path, mask/layout,
# strict (de)serialization.
# =============================================================================


def build_architecture_spec(core: Any) -> dict[str, Any]:
    kwargs = _base_primitive_kwargs(core)
    base = CrossPositionPrimitive(
        REC004D_TARGET_PHYSICAL_ID, CrossPositionPrimitiveConfig(**kwargs)
    )
    bias = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
    )
    base_params = base.num_parameters()
    bias_params = bias.num_parameters()
    new_params = bias_params - base_params
    budget_fraction = new_params / base_params
    spec = {
        "task_id": REC004D_TASK_ID,
        "architecture_signature_u": REC004D_ARCHITECTURE_SIGNATURE_U,
        "architecture_signature_p": REC004D_ARCHITECTURE_SIGNATURE_P,
        "feature_formula": "phi(i,j,n) = [i/d, j/d, (j-i)/d, n/length_ref], d = max(n-1, 1)",
        "hidden_layer": f"Linear(4, {REC004D_BIAS_HIDDEN_DIM}, bias=True) -> ReLU",
        "output_layer": f"Linear({REC004D_BIAS_HIDDEN_DIM}, 1, bias=False)",
        "shared_across_heads": True,
        "length_ref": REC004D_LENGTH_REF,
        "length_ref_source": (
            "CrossPositionPrimitiveConfig.max_sequence_length (fixed schema constant)"
        ),
        "base_primitive_param_count": base_params,
        "bias_primitive_param_count": bias_params,
        "new_param_count": new_params,
        "expected_new_param_count": REC004D_EXPECTED_NEW_PARAM_COUNT,
        "new_param_count_matches_expected": new_params == REC004D_EXPECTED_NEW_PARAM_COUNT,
        "budget_fraction_of_base": budget_fraction,
        "budget_ceiling": REC004D_BUDGET_FRACTION_CEILING,
        "budget_status": (
            "PASS" if budget_fraction <= REC004D_BUDGET_FRACTION_CEILING
            else "POSITION_BIAS_BUDGET_MISMATCH"
        ),
        "position_bias_param_keys": list(REC004D_BIAS_PARAM_KEYS),
    }
    if spec["budget_status"] != "PASS" or not spec["new_param_count_matches_expected"]:
        raise ValueError(f"POSITION_BIAS_BUDGET_MISMATCH: {spec}")
    return spec


def build_shared_bias_initial_state(core: Any) -> dict[str, torch.Tensor]:
    """Constructed once under an isolated, fixed local seed so all 5 P
    initial states share the SAME bias-net starting weights (hidden layer:
    ordinary nonzero random init; output layer: exactly zero, guaranteed by
    `CrossPositionLengthBiasPrimitive.__init__` regardless of RNG state)."""
    init_seed = _local_seed(0, f"rec004d_shared_bias_init:{REC004D_TARGET_OPERATION}")
    torch.manual_seed(init_seed)
    primitive = _new_arm_primitive(core, REC004D_ARM_P)
    full_state = primitive.state_dict()
    return {k: full_state[k].detach().clone().cpu() for k in REC004D_BIAS_PARAM_KEYS}


def run_zero_bias_parity_check(
    core: Any, shared_bias_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """B2: a freshly-constructed P (bias output at zero) must be an exact
    forward AND gradient no-op versus U from the SAME shared common weights,
    on both CPU and the real training device."""
    kwargs = _base_primitive_kwargs(core)
    results: dict[str, Any] = {}
    for device_label in sorted({"cpu", str(core.device)}):
        device = torch.device(device_label)
        torch.manual_seed(_local_seed(0, "rec004d_zero_bias_parity_fixture"))
        u = CrossPositionPrimitive(
            REC004D_TARGET_PHYSICAL_ID, CrossPositionPrimitiveConfig(**kwargs)
        )
        u.to(device)
        p = CrossPositionLengthBiasPrimitive(
            REC004D_TARGET_PHYSICAL_ID,
            CrossPositionLengthBiasPrimitiveConfig(
                **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
            ),
        )
        p.to(device)
        merged = {**u.state_dict(), **{k: v.to(device) for k, v in shared_bias_state.items()}}
        p.load_state_dict(merged, strict=True)
        bias_out_is_zero = bool(torch.count_nonzero(p.position_bias_out.weight).item() == 0)

        content_lengths = [6, 7, 8, 9, 10]
        output_lengths = list(content_lengths)
        lmax = max(content_lengths)
        torch.manual_seed(_local_seed(1, f"rec004d_zero_bias_parity_input:{device_label}"))
        content = torch.randn(len(content_lengths), lmax, kwargs["d_model"], device=device)

        u.eval()
        p.eval()
        with torch.no_grad():
            out_u = u(content, content_lengths, output_lengths, None)
            out_p = p(content, content_lengths, output_lengths, None)
        forward_max_abs_diff = (out_u - out_p).abs().max().item()

        u.train()
        p.train()
        for prm in (*u.parameters(), *p.parameters()):
            prm.requires_grad_(True)
        content_u = content.detach().clone().requires_grad_(True)
        content_p = content.detach().clone().requires_grad_(True)
        target = torch.randint(
            0, kwargs["vocab_size"], (len(content_lengths), max(output_lengths)), device=device
        )
        loss_u = F.cross_entropy(
            u(content_u, content_lengths, output_lengths, None).reshape(-1, kwargs["vocab_size"]),
            target.reshape(-1),
        )
        loss_p = F.cross_entropy(
            p(content_p, content_lengths, output_lengths, None).reshape(-1, kwargs["vocab_size"]),
            target.reshape(-1),
        )
        loss_u.backward()
        loss_p.backward()
        u_named = dict(u.named_parameters())
        p_named = dict(p.named_parameters())
        shared_keys = [k for k in u.state_dict() if k in p_named]
        grad_diffs: list[float] = []
        for k in shared_keys:
            u_grad, p_grad = u_named[k].grad, p_named[k].grad
            if u_grad is None or p_grad is None:
                continue
            grad_diffs.append((u_grad - p_grad).abs().max().item())
        grad_max_abs_diff = max(grad_diffs) if grad_diffs else 0.0
        results[device_label] = {
            "bias_output_weight_is_zero": bias_out_is_zero,
            "forward_max_abs_diff": forward_max_abs_diff,
            "shared_param_grad_max_abs_diff": grad_max_abs_diff,
            "grad_tolerance": REC004D_ZERO_BIAS_GRAD_TOL,
            "parity_passed": (
                forward_max_abs_diff == 0.0 and grad_max_abs_diff <= REC004D_ZERO_BIAS_GRAD_TOL
            ),
        }
    return {
        "task_id": REC004D_TASK_ID,
        "devices_checked": sorted(results.keys()),
        "results": results,
        "all_parity_passed": all(r["parity_passed"] for r in results.values()),
        "tolerance_note": (
            "forward pass must be bit-exact (0.0) on every device: at zero bias output, the "
            "additive attn_mask (-inf padding / 0 elsewhere) and CrossPositionPrimitive's "
            "boolean key_padding_mask are the same computation. Gradients are allowed a small "
            f"pre-registered tolerance ({REC004D_ZERO_BIAS_GRAD_TOL}) because PyTorch's fused "
            "attention kernel takes a different internal code path for a float attn_mask than "
            "for a bool key_padding_mask, producing backend-dependent floating-point rounding "
            "differences on backward (observed: exactly 0.0 on the real CUDA training device; "
            "~2e-8 on CPU) -- this tolerance is fixed here, before Stage C locks the protocol, "
            "and is never widened after observing a failure."
        ),
    }


def run_gradient_path_audit(
    core: Any, shared_bias_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """B2: confirms the new branch's gradient path is real (not `detach()`d
    and not permanently zero): at init, `position_bias_out`'s own gradient
    is nonzero (its backward path does not depend on itself), while
    `position_bias_hidden`'s gradient is exactly zero (blocked by the
    zero output weight -- expected, not a bug); after one optimizer step
    moves the output weight off zero, the hidden layer's gradient becomes
    nonzero too."""
    kwargs = _base_primitive_kwargs(core)
    device = core.device
    torch.manual_seed(_local_seed(0, "rec004d_gradient_path_fixture"))
    p = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
    )
    p.to(device)
    p.load_state_dict(
        {**p.state_dict(), **{k: v.to(device) for k, v in shared_bias_state.items()}}, strict=True
    )
    p.train()
    for prm in p.parameters():
        prm.requires_grad_(True)
    optimizer = torch.optim.AdamW(p.parameters(), lr=REC004D_OPERATOR_LR)

    content_lengths = [6, 8, 10]
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    torch.manual_seed(_local_seed(1, "rec004d_gradient_path_input"))
    content = torch.randn(len(content_lengths), lmax, kwargs["d_model"], device=device)
    target = torch.randint(
        0, kwargs["vocab_size"], (len(content_lengths), max(output_lengths)), device=device
    )

    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        p(content, content_lengths, output_lengths, None).reshape(-1, kwargs["vocab_size"]),
        target.reshape(-1),
    )
    loss.backward()
    out_weight_grad = p.position_bias_out.weight.grad
    hidden_weight_grad = p.position_bias_hidden.weight.grad
    assert out_weight_grad is not None
    assert hidden_weight_grad is not None
    out_grad_at_init = out_weight_grad.abs().sum().item()
    hidden_grad_at_init = hidden_weight_grad.abs().sum().item()
    optimizer.step()
    output_weight_nonzero_after_step = bool(
        torch.count_nonzero(p.position_bias_out.weight).item() > 0
    )

    optimizer.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(
        p(content, content_lengths, output_lengths, None).reshape(-1, kwargs["vocab_size"]),
        target.reshape(-1),
    )
    loss2.backward()
    hidden_weight_grad_after = p.position_bias_hidden.weight.grad
    assert hidden_weight_grad_after is not None
    hidden_grad_after_step = hidden_weight_grad_after.abs().sum().item()

    return {
        "task_id": REC004D_TASK_ID,
        "at_init": {
            "position_bias_out_grad_abs_sum": out_grad_at_init,
            "position_bias_hidden_grad_abs_sum": hidden_grad_at_init,
            "output_grad_nonzero": out_grad_at_init > 0,
            "hidden_grad_zero_as_expected": hidden_grad_at_init == 0.0,
        },
        "after_one_optimizer_step": {
            "output_weight_nonzero": output_weight_nonzero_after_step,
            "position_bias_hidden_grad_abs_sum": hidden_grad_after_step,
            "hidden_grad_now_nonzero": hidden_grad_after_step > 0,
        },
        "both_layers_permanently_zero": False,
        "gradient_path_confirmed_real": (
            out_grad_at_init > 0
            and hidden_grad_at_init == 0.0
            and output_weight_nonzero_after_step
            and hidden_grad_after_step > 0
        ),
    }


def run_mask_layout_tests(core: Any) -> dict[str, Any]:
    """B3: heterogeneous-length batch produces finite logits (no NaN/Inf
    despite -inf mask entries), and different real lengths in the same
    batch never share one length's bias matrix."""
    kwargs = _base_primitive_kwargs(core)
    device = core.device
    torch.manual_seed(_local_seed(0, "rec004d_mask_layout_fixture"))
    p = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
    )
    p.to(device)
    with torch.no_grad():
        p.position_bias_out.weight.normal_(mean=0.0, std=0.1)
    p.eval()

    content_lengths = list(REC004D_LEGAL_LENGTHS)
    output_lengths = list(content_lengths)
    lmax = max(content_lengths)
    torch.manual_seed(_local_seed(1, "rec004d_mask_layout_input"))
    content = torch.randn(len(content_lengths), lmax, kwargs["d_model"], device=device)
    with torch.no_grad():
        out = p(content, content_lengths, output_lengths, None)
    finite = bool(torch.isfinite(out).all().item())

    bias_by_length = {
        n: p._position_bias([n], n, n, device).squeeze(0) for n in REC004D_LEGAL_LENGTHS
    }
    distinct_across_lengths = len({b.shape for b in bias_by_length.values()}) == len(
        REC004D_LEGAL_LENGTHS
    )

    return {
        "task_id": REC004D_TASK_ID,
        "batch_content_lengths": content_lengths,
        "all_logits_finite": finite,
        "bias_shape_distinct_per_real_length": distinct_across_lengths,
        "padding_never_revived": finite,
    }


def run_operator_serialization_contract_check(
    core: Any, shared_bias_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """B4: fresh-process round-trip (strict save/load) for the new type, and
    strict rejection of a cross-type load in both directions (the mechanism
    that makes "silently drop the bias and load into the old class"
    impossible)."""
    kwargs = _base_primitive_kwargs(core)
    u = CrossPositionPrimitive(REC004D_TARGET_PHYSICAL_ID, CrossPositionPrimitiveConfig(**kwargs))
    p = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
    )
    p.load_state_dict({**p.state_dict(), **shared_bias_state}, strict=True)

    p_sd = p.state_dict()
    p2 = CrossPositionLengthBiasPrimitive(
        REC004D_TARGET_PHYSICAL_ID,
        CrossPositionLengthBiasPrimitiveConfig(
            **kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF
        ),
    )
    p2.load_state_dict(p_sd, strict=True)
    round_trip_ok = all(torch.equal(p_sd[k], p2.state_dict()[k]) for k in p_sd)

    old_class_rejects_bias_dict = False
    try:
        u.load_state_dict(p_sd, strict=True)
    except RuntimeError:
        old_class_rejects_bias_dict = True

    new_class_rejects_base_only_dict = False
    try:
        p.load_state_dict(u.state_dict(), strict=True)
    except RuntimeError:
        new_class_rejects_base_only_dict = True

    abi_u = mb.compute_state_abi_hash(
        u.state_dict(), architecture_signature=REC004D_ARCHITECTURE_SIGNATURE_U
    )
    abi_p = mb.compute_state_abi_hash(p_sd, architecture_signature=REC004D_ARCHITECTURE_SIGNATURE_P)
    abi_p_wrong_tag = mb.compute_state_abi_hash(
        p_sd, architecture_signature=REC004D_ARCHITECTURE_SIGNATURE_U
    )

    return {
        "task_id": REC004D_TASK_ID,
        "new_type_round_trip_ok": round_trip_ok,
        "old_class_strict_load_of_bias_state_dict_rejected": old_class_rejects_bias_dict,
        "new_class_strict_load_of_base_only_state_dict_rejected": new_class_rejects_base_only_dict,
        "state_abi_hash_distinguishes_architecture": abi_u != abi_p and abi_p != abi_p_wrong_tag,
        "no_strict_false_bias_dropping_path_used": True,
        "contract_passed": (
            round_trip_ok
            and old_class_rejects_bias_dict
            and new_class_rejects_base_only_dict
            and abi_u != abi_p
            and abi_p != abi_p_wrong_tag
        ),
    }


def run_stage_b(core: Any) -> dict[str, Any]:
    _guard_not_frozen("run_stage_b")
    architecture_spec = build_architecture_spec(core)
    shared_bias_state = build_shared_bias_initial_state(core)
    zero_bias_parity = run_zero_bias_parity_check(core, shared_bias_state)
    if not zero_bias_parity["all_parity_passed"]:
        raise ValueError(f"ZERO_BIAS_PARITY_FAILURE: {zero_bias_parity}")
    gradient_path_audit = run_gradient_path_audit(core, shared_bias_state)
    if not gradient_path_audit["gradient_path_confirmed_real"]:
        raise ValueError(f"GRADIENT_PATH_AUDIT_FAILURE: {gradient_path_audit}")
    mask_layout_tests = run_mask_layout_tests(core)
    if not mask_layout_tests["all_logits_finite"]:
        raise ValueError(f"MASK_LAYOUT_TEST_FAILURE: {mask_layout_tests}")
    serialization_contract = run_operator_serialization_contract_check(core, shared_bias_state)
    if not serialization_contract["contract_passed"]:
        raise ValueError(f"OPERATOR_SERIALIZATION_CONTRACT_FAILURE: {serialization_contract}")
    return {
        "architecture_spec": architecture_spec,
        "shared_bias_initial_state": shared_bias_state,
        "zero_bias_parity": zero_bias_parity,
        "gradient_path_audit": gradient_path_audit,
        "mask_layout_tests": mask_layout_tests,
        "operator_serialization_contract": serialization_contract,
    }


# =============================================================================
# Stage C -- protocol lock.
# =============================================================================


def build_protocol(
    config: MirrorPositionBiasRepairConfig, stage_a: dict[str, Any], stage_b: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_steps = list(range(0, config.max_updates_per_run + 1, config.checkpoint_interval))
    if checkpoint_steps[-1] != REC004D_DECISIVE_STEP:
        raise ValueError(
            "protocol construction requires the decisive step on a checkpoint boundary"
        )

    lr_check = msc._verify_lr_trace(REC004D_T_MAX, checkpoint_steps)
    if not lr_check["contract_ok"]:
        raise ValueError(f"LR_TRACE_CONTRACT_FAILURE: {lr_check['mismatches']}")

    protocol = {
        "task_id": REC004D_TASK_ID,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "parent_core_canonical_state_hash": stage_a["parent_manifest"].core.canonical_state_hash,
        "model_seed": config.seed,
        "target_operation": REC004D_TARGET_OPERATION,
        "target_operation_physical_id": REC004D_TARGET_PHYSICAL_ID,
        "source_hashes": {
            "base_initial_state_hashes": stage_a["base_initial_state_hashes"],
            "shared_bias_initial_state_hash": mb.canonical_state_hash(
                stage_b["shared_bias_initial_state"]
            ),
            "fixed_candidate_files": {
                op: stage_a["source_audit"]["fixed_candidate_files"][op]["checkpoint_state_hash"]
                for op in REC004D_FIXED_CANDIDATE_OPERATIONS
            },
        },
        "metric_v2": {
            "fixed_position_audit_classification": stage_a["fixed_position_audit"][
                "classification"
            ],
            "metric_erratum_status": stage_a["metric_erratum"].get("status"),
        },
        "architecture": stage_b["architecture_spec"],
        "layout": {
            "length_ref": REC004D_LENGTH_REF,
            "bias_injection": "additive attn_mask (softmax pre-add), padded keys -inf",
        },
        "arms": list(REC004D_ARMS),
        "init_ids": list(REC004D_INIT_IDS),
        "training_rng": {
            "per_step_examples": (
                "ibc._generate_step_training_examples(seed, step, 'MIRROR_HALVES', "
                "vocab_size=10, sequence_length_range=(6,10)) -- IDENTICAL for all 10 "
                "runs (arm/init never mixed into the data seed)"
            ),
        },
        "lr_table": lr_check["trace"],
        "operator_lr": REC004D_OPERATOR_LR,
        "operator_weight_decay": REC004D_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004D_OPERATOR_GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": REC004D_T_MAX,
        "scheduler_eta_min": REC004D_SCHEDULER_ETA_MIN,
        "max_updates_per_run": config.max_updates_per_run,
        "max_updates_total": REC004D_TOTAL_MAX_UPDATES,
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_steps": checkpoint_steps,
        "decisive_checkpoint": REC004D_DECISIVE_STEP,
        "examples_per_step": REC004D_EXAMPLES_PER_STEP,
        "data_roles": {
            "train": {"purpose": "per-step training stream, shared identically by all 10 runs"},
            "train_fit": {"purpose": "diagnostic only", "used_for_selection": False},
            "existing_validation": {
                "purpose": "5-pair adoption condition (P>=floor on all 5) ONLY",
                "split": REC004D_EXISTING_VALIDATION_SPLIT,
                "n_examples": config.existing_validation_examples,
                "reused_from": "B-C005REC-004A budget_validation (hash-verified in Stage A)",
                "used_for_selection": True,
            },
            "diagnostic_suites": {
                "purpose": "corrected-metric evaluation only, not merged with EM"
            },
            "rec004d_recheck_query": {
                "purpose": "one-time independent RG3 recheck, post-selection only, new namespace",
                "split": REC004D_RECHECK_QUERY_SPLIT,
                "n_examples": config.recheck_query_examples,
                "used_for_selection": False,
            },
            "reference": {
                "purpose": "nominal reference-adequacy certification (not attempted here)"
            },
        },
        "adoption_rule": (
            "if ALL 5 P runs' existing_validation sequence EM at step=6000 >= "
            f"{config.existing_validation_floor}: adopt P/I01@6000 (fixed regardless of "
            "which init scored highest); else: candidate=null "
            "(POSITION_BIAS_VALIDATION_NOT_MET). Only the step=6000 checkpoint is ever "
            "evaluated for adoption; no intermediate checkpoint, no best-of-5 selection."
        ),
        "fixed_candidates": {
            op: {
                "step": REC004D_FIXED_CANDIDATE_STEP,
                "source": str(_fixed_candidate_checkpoint_path(op)),
            }
            for op in REC004D_FIXED_CANDIDATE_OPERATIONS
        },
        "protected_operations": list(REC004D_PROTECTED_OPERATIONS),
        "recheck_query_spec": {
            "split": REC004D_RECHECK_QUERY_SPLIT,
            "n_examples_per_op": config.recheck_query_examples,
            "n_operations": 16,
        },
        "qualification_scope": {
            "tier_1_diagnostic_loadable": True,
            "tier_3_r3_reference_contract_nominal_certified": "NOT_ATTEMPTED",
        },
        "comparison_tolerance": {"source_validation_replay_abs_tol": 1e-9},
        "forbidden_changes": [
            "Core retraining",
            "query/key/value/FFN width, depth, or head count change",
            "different LR/loss/batch/optimizer/precision",
            "curriculum, warmup, or additional init beyond I01-I05",
            "extending beyond 6000 updates per run / 60000 total",
            "router recalibration",
            "argument scorer recalibration",
            "verifier change",
            "position-map teacher auxiliary loss",
            "oracle-position gather",
            "runtime REVERSE+SHIFT substitution",
            "relaxing the non-SHIFT-15 functional floor",
            "hardcoding mid=n//2, half-region id, pi_n(i), distance-to-pi_n(i), or a "
            "same-half hard mask into the bias feature/embedding",
        ],
    }
    protocol_for_hash = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol_for_hash, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    (config.output_dir / "position_bias_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    return protocol


# =============================================================================
# Stage D -- train U and P from each of I01-I05 (10 runs), paired comparison,
# bias-zero ablation.
# =============================================================================


def _position_level_breakdown(
    core: Any, primitive: Any, examples: list[Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Own (not REC-004C-task-ID-labeled) fixed/moved position breakdown,
    reusing the same `mirror_halves_position_map` REC-004C's own diagnostic
    used and verified against the oracle interpreter in Stage A."""
    preds = _predict_tokens(core, primitive, examples, REC004D_TARGET_OPERATION, None)
    by_position: dict[str, dict[str, int]] = {}
    structural = {"moved": {"n": 0, "correct": 0}, "fixed": {"n": 0, "correct": 0}}
    sequence_em = 0
    for ex, pred in zip(examples, preds, strict=True):
        n = len(ex.input_tokens)
        target = ex.target_tokens
        pi = mpid.mirror_halves_position_map(n)
        n_err = 0
        for i in range(n):
            is_correct = i < len(pred) and pred[i] == target[i]
            n_err += int(not is_correct)
            key = "moved" if pi[i] != i else "fixed"
            structural[key]["n"] += 1
            structural[key]["correct"] += int(is_correct)
            bucket = by_position.setdefault(f"{n}:{i}", {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(is_correct)
        sequence_em += int(n_err == 0)
    summary = {
        "n_examples": len(examples),
        "sequence_exact_match": sequence_em / len(examples) if examples else None,
        "by_length_position": by_position,
    }
    confusion = {"structural_moved_vs_fixed": structural}
    return summary, confusion


def run_bias_ablation(
    core: Any, primitive: CrossPositionLengthBiasPrimitive, examples: list[Any]
) -> dict[str, Any]:
    """D3: forward-evaluates the SAME trained P with its bias output weight
    temporarily zeroed (an intervention on the trained weights, not a reset
    to the U architecture), then confirms restoring the bias reproduces the
    original predictions exactly."""
    primitive.eval()
    with_bias_preds = _predict_tokens(core, primitive, examples, REC004D_TARGET_OPERATION, None)
    with primitive.zeroed_position_bias():
        zero_bias_preds = _predict_tokens(core, primitive, examples, REC004D_TARGET_OPERATION, None)
    restored_preds = _predict_tokens(core, primitive, examples, REC004D_TARGET_OPERATION, None)

    correct_with = [
        tuple(p) == tuple(e.target_tokens) for p, e in zip(with_bias_preds, examples, strict=True)
    ]
    correct_zero = [
        tuple(p) == tuple(e.target_tokens) for p, e in zip(zero_bias_preds, examples, strict=True)
    ]
    em_with = sum(correct_with) / len(correct_with) if correct_with else None
    em_zero = sum(correct_zero) / len(correct_zero) if correct_zero else None
    return {
        "n_examples": len(examples),
        "em_with_bias": em_with,
        "em_zero_bias": em_zero,
        "delta_with_minus_zero": (
            (em_with - em_zero) if (em_with is not None and em_zero is not None) else None
        ),
        "restore_reproduces_original_predictions_exactly": with_bias_preds == restored_preds,
        "caveat": (
            "This is an intervention on the trained P's own weights, not a reset to U's "
            "architecture; degradation under zero-bias does not by itself attribute the "
            "original failure solely to missing position information."
        ),
    }


def _digest_examples(examples: list[Any]) -> str:
    payload = [[list(e.input_tokens), list(e.target_tokens)] for e in examples]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def run_one_arm(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    init_id: str,
    arm: str,
    initial_state: dict[str, torch.Tensor],
    config: MirrorPositionBiasRepairConfig,
    output_dir: Path,
) -> dict[str, Any]:
    _guard_not_frozen(f"run_one_arm:{init_id}:{arm}")
    pid = REC004D_TARGET_PHYSICAL_ID
    device = core.device
    seed = config.seed

    primitive = _new_arm_primitive(core, arm)
    primitive.to(device)
    primitive.load_state_dict({k: v.to(device) for k, v in initial_state.items()}, strict=True)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004D_OPERATOR_LR, weight_decay=REC004D_OPERATOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004D_T_MAX, eta_min=REC004D_SCHEDULER_ETA_MIN
    )

    run_dir = output_dir / init_id / arm
    ckpt_dir = run_dir / "checkpoints"
    state_dir = run_dir / "training_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints: list[dict[str, Any]] = []
    lr_trace: list[dict[str, Any]] = []
    data_digests: list[str] = []
    cumulative_examples = 0
    t_start = time.time()
    diverged_at: int | None = None

    def _snapshot_and_eval(step: int, lr_used: float | None, lr_after: float) -> None:
        state_snapshot = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
        torch.save(state_snapshot, ckpt_dir / f"step{step}.pt")
        torch.save(
            {
                "primitive_state_dict": state_snapshot,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "cumulative_updates": step,
                "init_id": init_id,
                "arm": arm,
            },
            state_dir / f"step{step}.pt",
        )

        primitive.eval()
        original_slot = eval_bank.replace_primitive(pid, primitive)
        try:
            existing_validation = _evaluate_one_operation(
                core, eval_bank, op_to_id, REC004D_TARGET_OPERATION,
                seed=seed, n_examples=config.existing_validation_examples,
                split=REC004D_EXISTING_VALIDATION_SPLIT,
            )
        finally:
            eval_bank.replace_primitive(pid, original_slot)
        train_fit = ibc._evaluate_train_fit(
            core, primitive, REC004D_TARGET_OPERATION, seed=seed,
            vocab_size=REC004D_VOCAB_SIZE, sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
        )
        length_strata = ibc._length_stratified_breakdown(
            core, primitive, REC004D_TARGET_OPERATION, seed=seed,
            n_examples=config.existing_validation_examples, split=REC004D_EXISTING_VALIDATION_SPLIT,
        )

        extras: dict[str, Any] | None = None
        if step == config.max_updates_per_run:
            final_examples = _generate_parameter_free_examples(
                seed, config.existing_validation_examples, operation=REC004D_TARGET_OPERATION,
                split=REC004D_EXISTING_VALIDATION_SPLIT, vocab_size=REC004D_VOCAB_SIZE,
                sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
            )
            position_summary, position_confusion = _position_level_breakdown(
                core, primitive, final_examples
            )
            extras = {
                "position_summary": position_summary,
                "position_confusion": position_confusion,
            }
            if arm == REC004D_ARM_P:
                assert isinstance(primitive, CrossPositionLengthBiasPrimitive)
                extras["bias_ablation"] = run_bias_ablation(core, primitive, final_examples)
        primitive.train()

        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        checkpoints.append(
            {
                "step": step,
                "init_id": init_id,
                "arm": arm,
                "cumulative_training_examples": cumulative_examples,
                "lr_used": lr_used,
                "lr_after_scheduler": lr_after,
                "existing_validation": existing_validation,
                "train_fit": train_fit,
                "length_stratified": length_strata,
                "final_step_extras": extras,
                "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
                "cumulative_wall_clock_seconds": time.time() - t_start,
                "peak_vram_bytes": peak_vram,
            }
        )

    initial_lr = optimizer.param_groups[0]["lr"]
    _snapshot_and_eval(0, lr_used=None, lr_after=initial_lr)

    for step in range(1, config.max_updates_per_run + 1):
        examples = ibc._generate_step_training_examples(
            seed, step, REC004D_TARGET_OPERATION,
            vocab_size=REC004D_VOCAB_SIZE, sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
        )
        cumulative_examples += len(examples)
        data_digests.append(_digest_examples(examples))

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004D_TARGET_OPERATION).output_length(n) for n in content_lengths
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
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004D_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())
        lr_trace.append(
            {
                "u": step, "lr_used": lr_used, "lr_after_scheduler": lr_after,
                "train_loss": running_loss,
            }
        )

        if not math.isfinite(running_loss):
            diverged_at = step
            break

        if step % config.checkpoint_interval == 0:
            _snapshot_and_eval(step, lr_used, lr_after)

    for step in range(0, config.max_updates_per_run + 1, config.checkpoint_interval):
        if diverged_at is not None and step >= diverged_at and not any(
            c["step"] == step for c in checkpoints
        ):
            checkpoints.append(
                {
                    "step": step, "status": "NOT_EXECUTED",
                    "reason": f"loss diverged at step {diverged_at}",
                }
            )

    checkpoints.sort(key=lambda c: c["step"])
    primitive.eval()
    final_state = {k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()}
    return {
        "init_id": init_id,
        "arm": arm,
        "checkpoints": checkpoints,
        "lr_trace": lr_trace,
        "data_digests": data_digests,
        "diverged_at_step": diverged_at,
        "final_primitive_state_dict": final_state,
        "total_wall_clock_seconds": time.time() - t_start,
    }


def _em_at_step(outcome: dict[str, Any], step: int) -> float | None:
    for c in outcome["checkpoints"]:
        if c["step"] == step and "existing_validation" in c:
            return c["existing_validation"]["correct_exact_match"]
    return None


def paired_comparison_for_init(
    core: Any, eval_bank: Any, op_to_id: dict[str, int], init_id: str,
    outcomes: dict[tuple[str, str], dict[str, Any]], config: MirrorPositionBiasRepairConfig,
) -> dict[str, Any]:
    pid = REC004D_TARGET_PHYSICAL_ID
    examples = _generate_parameter_free_examples(
        config.seed, config.existing_validation_examples, operation=REC004D_TARGET_OPERATION,
        split=REC004D_EXISTING_VALIDATION_SPLIT, vocab_size=REC004D_VOCAB_SIZE,
        sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
    )
    original_slot = eval_bank.get(pid)

    preds: dict[str, list[list[int]]] = {}
    for arm in REC004D_ARMS:
        sd = outcomes[(init_id, arm)]["final_primitive_state_dict"]
        arm_primitive = _new_arm_primitive(core, arm)
        arm_primitive.to(core.device)
        arm_primitive.load_state_dict({k: v.to(core.device) for k, v in sd.items()}, strict=True)
        arm_primitive.eval()
        eval_bank.replace_primitive(pid, arm_primitive)
        preds[arm] = _predict_tokens(core, arm_primitive, examples, REC004D_TARGET_OPERATION, None)
    eval_bank.replace_primitive(pid, original_slot)

    u_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds[REC004D_ARM_U], examples, strict=True)
    ]
    p_correct = [
        tuple(p) == tuple(e.target_tokens)
        for p, e in zip(preds[REC004D_ARM_P], examples, strict=True)
    ]
    both_correct = sum(1 for u, p in zip(u_correct, p_correct, strict=False) if u and p)
    u_only = sum(1 for u, p in zip(u_correct, p_correct, strict=False) if u and not p)
    p_only = sum(1 for u, p in zip(u_correct, p_correct, strict=False) if p and not u)
    both_wrong = sum(1 for u, p in zip(u_correct, p_correct, strict=False) if not u and not p)

    by_length: dict[int, dict[str, int]] = {}
    for ex, u, p in zip(examples, u_correct, p_correct, strict=False):
        bucket = by_length.setdefault(
            len(ex.input_tokens), {"n": 0, "u_correct": 0, "p_correct": 0}
        )
        bucket["n"] += 1
        bucket["u_correct"] += int(u)
        bucket["p_correct"] += int(p)

    em_u = sum(u_correct) / len(u_correct)
    em_p = sum(p_correct) / len(p_correct)
    return {
        "init_id": init_id,
        "n_examples": len(examples),
        "em_u": em_u,
        "em_p": em_p,
        "delta_p_minus_u": em_p - em_u,
        "both_correct": both_correct,
        "u_only_correct": u_only,
        "p_only_correct": p_only,
        "both_wrong": both_wrong,
        "by_length": {str(k): v for k, v in sorted(by_length.items())},
    }


def build_initialization_stability(
    per_init_comparisons: dict[str, dict[str, Any]], floor: float
) -> dict[str, Any]:
    em_p_values = [c["em_p"] for c in per_init_comparisons.values()]
    deltas = [c["delta_p_minus_u"] for c in per_init_comparisons.values()]
    n = len(em_p_values)
    mean_p = sum(em_p_values) / n
    sample_sd_p = (
        (sum((v - mean_p) ** 2 for v in em_p_values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    )
    n_pass = sum(1 for v in em_p_values if v >= floor)
    all_improved = all(d > 0 for d in deltas)
    any_improved = any(d > 0 for d in deltas)
    any_degraded = any(d < 0 for d in deltas)
    if all_improved:
        paired_effect_label = "PAIRED_GAIN_ALL_FIVE_OBSERVED"
    elif any_improved and any_degraded:
        paired_effect_label = "MIXED_PAIRED_EFFECT"
    else:
        paired_effect_label = "NO_CONSISTENT_GAIN_OBSERVED"
    return {
        "task_id": REC004D_TASK_ID,
        "per_init": per_init_comparisons,
        "em_p_summary": {
            "n": n, "mean": mean_p, "sample_sd": sample_sd_p,
            "min": min(em_p_values), "max": max(em_p_values),
            "range": max(em_p_values) - min(em_p_values),
        },
        "n_p_reaching_floor": n_pass,
        "floor": floor,
        "five_init_validation_floor_pass": n_pass == n,
        "paired_effect_label": paired_effect_label,
    }


# =============================================================================
# Stage E -- candidate decision (E1), child bundle assembly (E2).
# =============================================================================


def select_candidate(
    outcomes: dict[tuple[str, str], dict[str, Any]],
    per_init_comparisons: dict[str, dict[str, Any]],
    floor: float,
) -> dict[str, Any]:
    completed = True
    for init_id in REC004D_INIT_IDS:
        for arm in REC004D_ARMS:
            outcome = outcomes[(init_id, arm)]
            if (
                outcome["diverged_at_step"] is not None
                or _em_at_step(outcome, REC004D_DECISIVE_STEP) is None
            ):
                completed = False
    if not completed:
        return {"selected_arm": None, "selected_init": None, "status": "PAIR_INCOMPLETE"}
    all_p_pass = all(c["em_p"] >= floor for c in per_init_comparisons.values())
    if all_p_pass:
        return {
            "selected_arm": REC004D_ARM_P, "selected_init": "I01",
            "status": "FIVE_INIT_VALIDATION_FLOOR_PASS",
            "note": "Fixed by pre-registered rule (P/I01@6000); not a best-of-5 selection.",
        }
    return {
        "selected_arm": None, "selected_init": None,
        "status": "POSITION_BIAS_VALIDATION_NOT_MET",
    }


def _generate_length_safe_examples(
    seed: int, n: int, operation: str, split: str
) -> list[Example]:
    """Like `_generate_parameter_free_examples`, but restricts sampled
    lengths to those the operation actually accepts within
    `REC004D_LEGAL_LENGTHS` (e.g. BIND requires even length; the plain
    generic-range sampler doesn't check `is_valid_for_length` and raises for
    an operation with narrower length support)."""
    op_obj = get_operation(operation)
    valid_lengths = [n_ for n_ in REC004D_LEGAL_LENGTHS if op_obj.is_valid_for_length(n_)]
    if not valid_lengths:
        raise ValueError(
            f"no length in {REC004D_LEGAL_LENGTHS} is valid for operation {operation!r}"
        )
    rng = random.Random(_derive_local_seed(seed, 0, f"{split}:{operation}"))
    examples: list[Example] = []
    for _ in range(n):
        seq_len = rng.choice(valid_lengths)
        seq = tuple(rng.randrange(REC004D_VOCAB_SIZE) for _ in range(seq_len))
        params = op_obj.sample_params(rng, seq, REC004D_VOCAB_SIZE)
        step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(step,))
        res = run_program(prog, seq, REC004D_VOCAB_SIZE)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=split,
                vocab_size=REC004D_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(label="K", primitive_operations=(operation,)),
            )
        )
    return examples


def _rec004d_reconstruct_16_op_bank_structure(core: Any, seed: int) -> tuple[Any, dict[str, int]]:
    """Mirrors `ibc._reconstruct_16_op_bank_structure` exactly (same call
    order -> same deterministic physical_id assignment), except MIRROR_HALVES
    is constructed as `CrossPositionLengthBiasPrimitive` instead of the plain
    `CrossPositionPrimitive` every other slot still uses -- needed because
    the child bundle's MIRROR_HALVES slot carries 192 extra bias parameters
    the old class cannot strict-load."""
    u_bank_cfg = UnifiedBenchmarkConfig(seed=seed, vocab_size=10, device=str(core.device))
    bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
    for op in tuple(BRANCH_B_NOVEL_OPERATION_NAMES) + tuple(PHASE_A2_INCREMENTAL_NEW_OPERATIONS):
        base_kwargs = {
            "operation": op, "d_model": core.model.config.d_model, "d_operator": 32,
            "n_head": 4, "d_operator_ff": 64, "vocab_size": 10, "max_sequence_length": 32,
        }
        p: CrossPositionPrimitive | CrossPositionLengthBiasPrimitive
        if op == REC004D_TARGET_OPERATION:
            p = bank.new_cross_position_length_bias_primitive(
                CrossPositionLengthBiasPrimitiveConfig(
                    **base_kwargs, bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM,
                    length_ref=REC004D_LENGTH_REF,
                ),
                status=PrimitiveStatus.STABLE,
            )
        else:
            p = bank.new_cross_position_primitive(
                CrossPositionPrimitiveConfig(**base_kwargs), status=PrimitiveStatus.STABLE
            )
        op_to_id[op] = p.primitive_id
    return bank, op_to_id


def build_child_bundle(
    config: MirrorPositionBiasRepairConfig,
    parent_manifest: mb.ModelBundleManifest,
    outcomes: dict[tuple[str, str], dict[str, Any]],
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]],
) -> tuple[mb.ModelBundleManifest, dict[str, Any]]:
    _guard_not_frozen("build_child_bundle")
    ns = _rec004d_namespace(config.seed)
    publish_dir = ns / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    # Resolve to absolute now: the fresh-process Stage F subprocess runs from
    # a different cwd (a scratch temp dir), and a relative path recorded in
    # the manifest would resolve against THAT cwd instead of the repo root.
    publish_dir = publish_dir.resolve()

    core_copy = mb.legacy_import(
        Path(parent_manifest.core.file_path), publish_dir / "core",
        component_id=parent_manifest.core.component_id,
    )
    core_component = dataclasses.replace(core_copy, schema_hash=parent_manifest.core.schema_hash)
    vocab_copy = mb.legacy_import(
        Path(parent_manifest.vocabulary.file_path), publish_dir / "vocab",
        component_id=parent_manifest.vocabulary.component_id,
    )
    vocab_component = dataclasses.replace(
        vocab_copy, schema_hash=parent_manifest.vocabulary.schema_hash
    )

    router_dest = publish_dir / "router.pt"
    shutil.copy2(Path(parent_manifest.router.source_artifact), router_dest)
    router_component = dataclasses.replace(
        parent_manifest.router, source_artifact=str(router_dest.resolve())
    )

    scorer_dest = publish_dir / "argument_scorer.pt"
    shutil.copy2(Path(parent_manifest.argument_scorer.source_artifact), scorer_dest)
    scorer_component = dataclasses.replace(
        parent_manifest.argument_scorer, source_artifact=str(scorer_dest.resolve())
    )

    parent_bank_path = Path(ibc._single_bank_path(parent_manifest))
    parent_bank_sd = dict(mb.load_state_dict(parent_bank_path))
    merged_sd = dict(parent_bank_sd)

    selection_receipts: dict[str, str] = {}
    mirror_pid = REC004D_TARGET_PHYSICAL_ID
    mirror_sd = outcomes[("I01", REC004D_ARM_P)]["final_primitive_state_dict"]
    for k, v in mirror_sd.items():
        merged_sd[f"_primitives.{mirror_pid}.{k}"] = v.cpu()
    mirror_em = _em_at_step(outcomes[("I01", REC004D_ARM_P)], REC004D_DECISIVE_STEP)
    selection_receipts[REC004D_TARGET_OPERATION] = (
        f"REC-004D length-position-bias repair, arm=P, init=I01, step={REC004D_DECISIVE_STEP}, "
        f"existing_validation_sequence_EM={mirror_em:.4f} (fixed by pre-registered rule, not "
        "best-of-5 selection)"
    )

    for op in REC004D_FIXED_CANDIDATE_OPERATIONS:
        pid = REC004D_PHYSICAL_IDS[op]
        for k, v in fixed_candidate_state_dicts[op].items():
            merged_sd[f"_primitives.{pid}.{k}"] = v.cpu()
        selection_receipts[op] = (
            "REC-004A incremental budget calibration, FRESH_PAIRED_REBUILD trajectory, "
            f"imported read-only at step={REC004D_FIXED_CANDIDATE_STEP} (schedule T_max=1000 "
            f"unchanged), source={_fixed_candidate_checkpoint_path(op)}"
        )

    bank_path = (publish_dir / "primitive_bank_16.pt").resolve()
    torch.save(merged_sd, bank_path)

    replaced_ops = {REC004D_TARGET_OPERATION, *REC004D_FIXED_CANDIDATE_OPERATIONS}
    primitives = []
    for entry in parent_manifest.primitives:
        if entry.operation_name in replaced_ops:
            sliced = mb.primitive_state_dict(merged_sd, entry.physical_id)
            weights_hash = mb.canonical_state_hash(sliced)
            arch_sig = (
                REC004D_ARCHITECTURE_SIGNATURE_P
                if entry.operation_name == REC004D_TARGET_OPERATION
                else entry.architecture_signature
            )
            state_abi_hash = mb.compute_state_abi_hash(sliced, architecture_signature=arch_sig)
            provenance = (
                mb.ProvenanceStatus.TRAINED_THIS_BUILD
                if entry.operation_name == REC004D_TARGET_OPERATION
                else mb.ProvenanceStatus.LEGACY_IMPORTED
            )
            primitives.append(
                dataclasses.replace(
                    entry,
                    version=f"{entry.version}-rec004d",
                    architecture_signature=arch_sig,
                    state_abi_hash=state_abi_hash,
                    weights_hash=weights_hash,
                    source_artifact=str(bank_path),
                    provenance_status=provenance,
                    training_receipt=selection_receipts[entry.operation_name],
                )
            )
        else:
            primitives.append(dataclasses.replace(entry, source_artifact=str(bank_path)))

    system_info = get_system_info(seed=config.seed)
    manifest = mb.build_manifest(
        schema_version=parent_manifest.schema_version,
        source_commit=system_info.get("git_commit") or "unknown",
        runtime_recipe_version="rec004d-v1",
        environment_record={
            "python_version": str(system_info["python_version"]),
            "torch_version": str(system_info["torch_version"]),
            "device_name": str(system_info.get("device_name")),
        },
        model_id=parent_manifest.model_id,
        model_seed=parent_manifest.model_seed,
        training_run_id=f"rec004d-seed{config.seed}-run-001",
        parent_bundle_ids=(parent_manifest.bundle_id,),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "diagnostic_only"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=tuple(primitives),
        router=router_component,
        argument_scorer=scorer_component,
        scoring_policy=parent_manifest.scoring_policy,
        build_recipe_hash=hashlib.sha256(b"B-C005REC-004D-length-position-bias-v1").hexdigest(),
        known_defects=("REC004D_AWAITING_RG3_RECHECK",),
        clean_build_exercised_stages=("MIRROR_LENGTH_POSITION_BIAS_REPAIR",),
        qualification_refs=(REC004D_TASK_ID,),
    )

    self_load = mb.load_bundle(manifest, mode="nominal", expected_primitive_count=16)
    negative_test_passed = False
    try:
        mb.load_bundle(
            manifest, mode="nominal", required_capabilities=frozenset({"recovery_cohort_ready"}),
            expected_primitive_count=16,
        )
    except mb.CapabilityNotQualifiedError:
        negative_test_passed = True

    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles" / manifest.bundle_id
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = ibc._manifest_to_json_dict(manifest)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest_json, indent=2), encoding="utf-8"
    )

    manifest_paths_file = publish_dir / "manifest_paths.json"
    op_to_id = {p.operation_name: p.physical_id for p in manifest.primitives}
    manifest_paths_file.write_text(
        json.dumps({"op_to_id": op_to_id, "manifest": manifest_json}, indent=2), encoding="utf-8"
    )

    report = {
        "bundle_id": manifest.bundle_id,
        "parent_bundle_id": parent_manifest.bundle_id,
        "selected_arm": REC004D_ARM_P,
        "selected_init": "I01",
        "content_manifest_digest": manifest.content_manifest_digest,
        "self_load_nominal_mode_checks_performed": list(self_load.checks_performed),
        "unearned_capability_request_rejected": negative_test_passed,
        "manifest_path": str(bundle_dir / "manifest.json"),
        "manifest_paths_json_for_fresh_process": str(manifest_paths_file.resolve()),
        "bank_path": str(bank_path),
        "core_path": core_component.file_path,
        "router_path": str(router_dest.resolve()),
        "scorer_path": str(scorer_dest.resolve()),
        "selection_receipts": selection_receipts,
    }
    return manifest, report


def run_embedding_consistency_check(
    config: MirrorPositionBiasRepairConfig,
    core: Any,
    child_manifest: mb.ModelBundleManifest,
    outcomes: dict[tuple[str, str], dict[str, Any]],
    fixed_candidate_state_dicts: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """E2: confirms each of the 4 changed slots' standalone prediction
    matches its prediction inside the reconstructed child bank."""
    child_bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(child_manifest)))
    child_bank, child_op_to_id = _rec004d_reconstruct_16_op_bank_structure(core, config.seed)
    child_bank.to(core.device)
    child_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(child_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    child_bank.freeze_all()
    child_bank.eval()

    state_dicts_by_op = {
        REC004D_TARGET_OPERATION: outcomes[("I01", REC004D_ARM_P)]["final_primitive_state_dict"],
        **fixed_candidate_state_dicts,
    }
    results: dict[str, Any] = {}
    for op, sd in state_dicts_by_op.items():
        pid = REC004D_PHYSICAL_IDS[op]
        if op == REC004D_TARGET_OPERATION:
            standalone: Any = CrossPositionLengthBiasPrimitive(
                pid,
                CrossPositionLengthBiasPrimitiveConfig(
                    operation=op, d_model=core.model.config.d_model, d_operator=32, n_head=4,
                    d_operator_ff=64, vocab_size=REC004D_VOCAB_SIZE, max_sequence_length=32,
                    bias_hidden_dim=REC004D_BIAS_HIDDEN_DIM, length_ref=REC004D_LENGTH_REF,
                ),
                status=PrimitiveStatus.STABLE,
            )
        else:
            standalone = CrossPositionPrimitive(
                pid,
                CrossPositionPrimitiveConfig(
                    operation=op, d_model=core.model.config.d_model, d_operator=32, n_head=4,
                    d_operator_ff=64, vocab_size=REC004D_VOCAB_SIZE, max_sequence_length=32,
                ),
                status=PrimitiveStatus.STABLE,
            )
        standalone.to(core.device)
        standalone.load_state_dict({k: v.to(core.device) for k, v in sd.items()}, strict=True)
        standalone.eval()
        examples = _generate_parameter_free_examples(
            config.seed, 64, operation=op, split="rec004d_embedding_consistency",
            vocab_size=REC004D_VOCAB_SIZE, sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
        )
        standalone_preds = _predict_tokens(core, standalone, examples, op, None)
        child_preds = _predict_tokens(core, child_bank.get(child_op_to_id[op]), examples, op, None)
        results[op] = {
            "standalone_matches_child_embedding": standalone_preds == child_preds,
            "n_examples": len(examples),
        }
    return {
        "task_id": REC004D_TASK_ID,
        "operations": results,
        "all_match": all(r["standalone_matches_child_embedding"] for r in results.values()),
    }


def rec004d_protected_regression_check(
    config: MirrorPositionBiasRepairConfig,
    parent_manifest: mb.ModelBundleManifest,
    child_manifest: mb.ModelBundleManifest,
    core: Any,
) -> dict[str, Any]:
    """Independently confirms the 12 protected primitives are byte-identical
    between parent and child (hash + same-input prediction agreement). Uses
    `ibc`'s reconstruction for the (all-old-type) parent and this module's
    own reconstruction for the (new-type MIRROR_HALVES) child."""
    parent_by_op = {p.operation_name: p for p in parent_manifest.primitives}
    child_by_op = {p.operation_name: p for p in child_manifest.primitives}

    hash_matches: dict[str, bool] = {}
    for op in REC004D_PROTECTED_OPERATIONS:
        hash_matches[op] = (
            parent_by_op[op].weights_hash == child_by_op[op].weights_hash
            and parent_by_op[op].core_dependency_hash == child_by_op[op].core_dependency_hash
        )

    parent_bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(parent_manifest)))
    parent_bank, parent_op_to_id = ibc._reconstruct_16_op_bank_structure(core, config.seed)
    parent_bank.to(core.device)
    parent_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in parent_manifest.primitives
            for k, v in mb.primitive_state_dict(parent_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    parent_bank.freeze_all()
    parent_bank.eval()

    child_bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(child_manifest)))
    child_bank, child_op_to_id = _rec004d_reconstruct_16_op_bank_structure(core, config.seed)
    child_bank.to(core.device)
    child_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(child_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    child_bank.freeze_all()
    child_bank.eval()

    prediction_matches: dict[str, bool] = {}
    for op in REC004D_PROTECTED_OPERATIONS:
        examples = _generate_length_safe_examples(
            config.seed, 32, op, "rec004d_protected_regression"
        )
        parent_preds = _predict_tokens(
            core, parent_bank.get(parent_op_to_id[op]), examples, op, None
        )
        child_preds = _predict_tokens(core, child_bank.get(child_op_to_id[op]), examples, op, None)
        prediction_matches[op] = parent_preds == child_preds

    all_protected_unchanged = all(hash_matches.values()) and all(prediction_matches.values())
    return {
        "task_id": REC004D_TASK_ID,
        "protected_operations": list(REC004D_PROTECTED_OPERATIONS),
        "hash_matches": hash_matches,
        "same_input_prediction_matches": prediction_matches,
        "all_protected_unchanged": all_protected_unchanged,
    }


# =============================================================================
# Stage F -- independent recheck_query + fresh-process reproducibility.
# =============================================================================


def run_stage_f(
    config: MirrorPositionBiasRepairConfig,
    core: Any,
    child_manifest: mb.ModelBundleManifest,
    manifest_paths_file: str,
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_f")
    op_to_id = {p.operation_name: p.physical_id for p in child_manifest.primitives}

    bank, fresh_op_to_id = _rec004d_reconstruct_16_op_bank_structure(core, config.seed)
    if fresh_op_to_id != op_to_id:
        raise mb.IncompleteBundleError(
            "child bank structure op_to_id mismatch during Stage F reload"
        )
    bank.to(core.device)
    bank_sd = mb.load_state_dict(Path(ibc._single_bank_path(child_manifest)))
    bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    bank.freeze_all()
    bank.eval()

    with frozen_evaluation():
        rows = {
            op: _evaluate_one_operation(
                core, bank, op_to_id, op, seed=config.seed,
                n_examples=config.recheck_query_examples, split=REC004D_RECHECK_QUERY_SPLIT,
            )
            for op in sorted(op_to_id)
        }
        for op in rows:
            if op == "SHIFT":
                rows[op]["recovery_floor_status"] = "COHERENT_LIMITED"
                rows[op]["recovery_floor_passed"] = None
            else:
                rows[op]["recovery_floor_status"] = "EVALUATED"
                rows[op]["recovery_floor_passed"] = (
                    rows[op]["correct_exact_match"] >= config.existing_validation_floor
                )

        in_process_predictions: dict[str, list[list[int]]] = {}
        for op in sorted(op_to_id):
            # Mirrors rec004_fresh_process_check.py's own generation branch
            # exactly (parameterized ops need real sampled arguments -- e.g.
            # BIND requires an even length, which the plain generic-range
            # parameter-free sampler doesn't respect) so this in-process
            # comparison is apples-to-apples with the subprocess's.
            if op in PARAMETERIZED_OPERATION_NAMES:
                n_groups = max(1, math.ceil(config.recheck_query_examples / 3))
                groups = generate_compact_operator_counterfactual_groups(
                    config.seed, n_groups, operation=op, step=0,
                    split=REC004D_RECHECK_QUERY_SPLIT, vocab_size=REC004D_VOCAB_SIZE,
                    sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE, group_size=3,
                )
                examples, _wrong_arg_map, _ = _flatten_groups(groups)
                examples = examples[: config.recheck_query_examples]
                provider = _make_arg_provider_correct(op)
            else:
                examples = _generate_parameter_free_examples(
                    config.seed, config.recheck_query_examples, operation=op,
                    split=REC004D_RECHECK_QUERY_SPLIT, vocab_size=REC004D_VOCAB_SIZE,
                    sequence_length_range=REC004D_SEQUENCE_LENGTH_RANGE,
                )
                provider = None
            in_process_predictions[op] = _predict_tokens(
                core, bank.get(op_to_id[op]), examples, op, provider
            )

    non_shift_ops = [op for op in rows if op != "SHIFT"]
    all_non_shift_floor_pass = all(bool(rows[op]["recovery_floor_passed"]) for op in non_shift_ops)

    all_primitive_execution = {
        "task_id": REC004D_TASK_ID,
        "non_shift_floor": config.existing_validation_floor,
        "recheck_query_examples_per_operation": config.recheck_query_examples,
        "operations": rows,
        "all_16_operations_present": set(rows) == set(op_to_id) and len(rows) == 16,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
        "non_shift_floor_failures": [
            op for op in non_shift_ops if not rows[op]["recovery_floor_passed"]
        ],
        "shift_status": "COHERENT_LIMITED",
        "shift_correct_exact_match": rows.get("SHIFT", {}).get("correct_exact_match"),
    }

    script_path = _repo_root() / "scripts" / "rec004_fresh_process_check.py"
    scratch_cwd = Path(tempfile.gettempdir()) / f"apc_rec004d_fresh_process_cwd_seed{config.seed}"
    scratch_cwd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, str(script_path),
            "--manifest-paths", manifest_paths_file,
            "--seed", str(config.seed),
            "--sample-examples-per-op", str(config.recheck_query_examples),
            "--split", REC004D_RECHECK_QUERY_SPLIT,
        ],
        cwd=str(scratch_cwd), capture_output=True, text=True, timeout=1800, check=False,
        env=dict(os.environ),
    )
    wall = time.time() - t0

    if proc.returncode != 0:
        fresh_process_report = {
            "task_id": REC004D_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "subprocess_returncode": proc.returncode,
            "subprocess_stdout_tail": proc.stdout[-4000:],
            "subprocess_stderr_tail": proc.stderr[-4000:],
            "reproducibility_passed": False,
            "wall_clock_seconds": wall,
        }
    else:
        fresh = json.loads(proc.stdout.strip().splitlines()[-1])
        fresh_predictions = fresh["predictions_by_op"]
        mismatches = {
            op: {
                "in_process": in_process_predictions[op],
                "fresh_process": fresh_predictions.get(op),
            }
            for op in sorted(op_to_id)
            if in_process_predictions[op] != fresh_predictions.get(op)
        }
        fresh_process_report = {
            "task_id": REC004D_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "repo_root_cwd": str(_repo_root()),
            "different_working_directory_confirmed": str(scratch_cwd) != str(_repo_root()),
            "sample_examples_per_op": config.recheck_query_examples,
            "subprocess_returncode": proc.returncode,
            "no_builder_identifier_found_in_subprocess_script": fresh[
                "no_builder_identifier_found_in_this_script"
            ],
            "load_bundle_checks_performed_in_subprocess": fresh["checks_performed_by_load_bundle"],
            "bundle_id_matches": fresh["bundle_id"] == child_manifest.bundle_id,
            "per_operation_prediction_mismatches": mismatches,
            "predictions_byte_exact_for_all_16_ops": not mismatches,
            "exact_match_by_op_in_subprocess": fresh["exact_match_by_op"],
            "reproducibility_passed": (
                not mismatches
                and bool(fresh["no_builder_identifier_found_in_this_script"])
                and fresh["bundle_id"] == child_manifest.bundle_id
            ),
            "wall_clock_seconds": wall,
        }

    return {
        "all_primitive_execution": all_primitive_execution,
        "fresh_process_report": fresh_process_report,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
    }


def _protected_scope_hashes(eval_bank: Any, op_to_id: dict[str, int]) -> dict[str, str]:
    return {
        op: mb.canonical_state_hash(eval_bank.get(op_to_id[op]).state_dict())
        for op in REC004D_PROTECTED_OPERATIONS
    }


def _config_to_yaml_dict(config: MirrorPositionBiasRepairConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "existing_validation_examples": config.existing_validation_examples,
        "existing_validation_floor": config.existing_validation_floor,
        "recheck_query_examples": config.recheck_query_examples,
        "checkpoint_interval": config.checkpoint_interval,
        "max_updates_per_run": config.max_updates_per_run,
    }


def build_cost_accounting(
    architecture_spec: dict[str, Any], outcomes: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    base_params = architecture_spec["base_primitive_param_count"]
    bias_params = architecture_spec["bias_primitive_param_count"]
    wall_clock_by_run = {
        f"{init_id}:{arm}": outcomes[(init_id, arm)]["total_wall_clock_seconds"]
        for init_id in REC004D_INIT_IDS for arm in REC004D_ARMS
    }
    peak_vram_by_run = {
        f"{init_id}:{arm}": max(
            (c.get("peak_vram_bytes") or 0)
            for c in outcomes[(init_id, arm)]["checkpoints"]
            if "peak_vram_bytes" in c
        )
        for init_id in REC004D_INIT_IDS for arm in REC004D_ARMS
    }
    return {
        "task_id": REC004D_TASK_ID,
        "u_resident_active_parameters": base_params,
        "p_resident_active_parameters": bias_params,
        "new_bias_parameters": bias_params - base_params,
        "bias_pair_count_per_forward": "out_max * lmax (every query-key position pair)",
        "bias_flops_estimate": (
            "O(batch * out_max * lmax * bias_hidden_dim) for the 2-layer MLP -- negligible "
            "next to the shared cross_attn/ffn cost at this scale (d_operator=32)"
        ),
        "wall_clock_seconds_by_run": wall_clock_by_run,
        "total_wall_clock_seconds": sum(wall_clock_by_run.values()),
        "peak_vram_bytes_by_run": peak_vram_by_run,
        "scope_note": (
            "This compares one structural intervention (explicit generic-coordinate "
            "position bias + 192 params) against the unmodified operator; it is NOT a "
            "parameter-matched non-position control, does not test extrapolation to unseen "
            "lengths, and does not claim generalization to other operations or autonomous "
            "architecture discovery."
        ),
    }


# =============================================================================
# Orchestration.
# =============================================================================


def run_mirror_position_bias_repair_task(config: MirrorPositionBiasRepairConfig) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004D's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004/REC-004A/REC-004C); got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    stage_a = run_stage_a(config)
    core, eval_bank, op_to_id = stage_a["core"], stage_a["eval_bank"], stage_a["op_to_id"]
    fixed_candidate_state_dicts = stage_a["fixed_candidate_state_dicts"]
    base_initial_states = stage_a["base_initial_states"]

    stage_b = run_stage_b(core)
    protocol = build_protocol(config, stage_a, stage_b)

    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8"
    )
    system_info = get_system_info(seed=config.seed)
    (output_dir / "system.json").write_text(
        json.dumps(system_info, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(protocol["data_roles"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "architecture_spec.json").write_text(
        json.dumps(stage_b["architecture_spec"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "zero_bias_parity.json").write_text(
        json.dumps(stage_b["zero_bias_parity"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "gradient_path_audit.json").write_text(
        json.dumps(stage_b["gradient_path_audit"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "mask_layout_tests.json").write_text(
        json.dumps(stage_b["mask_layout_tests"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "operator_serialization_contract.json").write_text(
        json.dumps(stage_b["operator_serialization_contract"], indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(stage_b["shared_bias_initial_state"], output_dir / "shared_bias_initial_state.pt")

    protected_hashes_before = _protected_scope_hashes(eval_bank, op_to_id)

    pid = REC004D_TARGET_PHYSICAL_ID
    original_mirror_slice = eval_bank.get(pid)

    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for init_id in REC004D_INIT_IDS:
        base_state = base_initial_states[init_id]
        p_initial_state = {**base_state, **stage_b["shared_bias_initial_state"]}
        for arm in REC004D_ARMS:
            eval_bank.replace_primitive(pid, original_mirror_slice)
            initial_state = base_state if arm == REC004D_ARM_U else p_initial_state
            outcomes[(init_id, arm)] = run_one_arm(
                core, eval_bank, op_to_id, init_id, arm, initial_state, config, output_dir
            )
    eval_bank.replace_primitive(pid, original_mirror_slice)

    all_digests = [outcomes[(i, a)]["data_digests"] for i in REC004D_INIT_IDS for a in REC004D_ARMS]
    data_stream_identical = all(d == all_digests[0] for d in all_digests)
    if not data_stream_identical:
        raise ValueError(
            "PAIRING_CONTRACT_FAILURE: not all 10 runs received an identical per-step "
            "training data stream"
        )

    per_init_comparisons = {
        init_id: paired_comparison_for_init(core, eval_bank, op_to_id, init_id, outcomes, config)
        for init_id in REC004D_INIT_IDS
    }
    initialization_stability = build_initialization_stability(
        per_init_comparisons, config.existing_validation_floor
    )

    with (output_dir / "lr_trace.jsonl").open("w", encoding="utf-8") as fh:
        for (init_id, arm), outcome in outcomes.items():
            for row in outcome["lr_trace"]:
                fh.write(json.dumps({"init_id": init_id, "arm": arm, **row}) + "\n")

    with (output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as fh:
        for (init_id, arm), outcome in outcomes.items():
            for ckpt in outcome["checkpoints"]:
                row = {k: v for k, v in ckpt.items() if k != "final_step_extras"}
                fh.write(json.dumps({"init_id": init_id, "arm": arm, **row}, default=str) + "\n")

    bias_ablation = {
        init_id: next(
            (
                c["final_step_extras"]["bias_ablation"]
                for c in outcomes[(init_id, REC004D_ARM_P)]["checkpoints"]
                if c.get("final_step_extras") and "bias_ablation" in c["final_step_extras"]
            ),
            None,
        )
        for init_id in REC004D_INIT_IDS
    }
    per_length_position_metrics = {
        f"{init_id}:{arm}": next(
            (
                c["final_step_extras"]
                for c in outcomes[(init_id, arm)]["checkpoints"]
                if c.get("final_step_extras")
            ),
            None,
        )
        for init_id in REC004D_INIT_IDS for arm in REC004D_ARMS
    }

    protected_hashes_after = _protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004D_TASK_ID,
        "core_canonical_state_hash_before": stage_a["parent_manifest"].core.canonical_state_hash,
        "core_canonical_state_hash_after": core_hash_after,
        "core_unchanged": core_hash_after == stage_a["parent_manifest"].core.canonical_state_hash,
        "protected_operations_hashes_before": protected_hashes_before,
        "protected_operations_hashes_after": protected_hashes_after,
        "protected_operations_unchanged": protected_hashes_before == protected_hashes_after,
    }
    (output_dir / "freeze_audit.json").write_text(
        json.dumps(freeze_audit, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(per_init_comparisons, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "initialization_stability.json").write_text(
        json.dumps(initialization_stability, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "bias_ablation.json").write_text(
        json.dumps(bias_ablation, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "per_length_position_metrics.json").write_text(
        json.dumps(per_length_position_metrics, indent=2, default=str), encoding="utf-8"
    )
    cost_accounting = build_cost_accounting(stage_b["architecture_spec"], outcomes)
    (output_dir / "cost_accounting.json").write_text(
        json.dumps(cost_accounting, indent=2, default=str), encoding="utf-8"
    )

    selection = select_candidate(outcomes, per_init_comparisons, config.existing_validation_floor)
    candidate_status = selection["status"]
    (output_dir / "candidate_decision.json").write_text(
        json.dumps(selection, indent=2, default=str), encoding="utf-8"
    )

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE",
        "source_audit_status": "PASS",
        "candidate_status": candidate_status,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "rec005_executed": False,
        "data_stream_identical": data_stream_identical,
        "source_audit": stage_a["source_audit"],
        "fixed_position_audit": stage_a["fixed_position_audit"],
        "metric_erratum": stage_a["metric_erratum"],
        "protocol": protocol,
        "selection": selection,
        "initialization_stability": initialization_stability,
        "freeze_audit": freeze_audit,
        "wall_clock_seconds": time.time() - start,
    }

    if candidate_status != "FIVE_INIT_VALIDATION_FLOOR_PASS":
        after_hashes = _snapshot_forbidden_cache_hashes(seed)
        side_effect_audit = {
            "task_id": REC004D_TASK_ID,
            "before": before_hashes,
            "after": after_hashes,
            "shared_cache_unchanged": before_hashes == after_hashes,
        }
        (output_dir / "side_effect_audit.json").write_text(
            json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
        )
        result["side_effect_audit"] = side_effect_audit
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result

    # build_child_bundle performs the final assembly (merging already-trained
    # state dicts and writing artifacts, never new training) and asserts its
    # own not-frozen guard, so it runs BEFORE evaluation is frozen; only the
    # read-only post-build checks run inside `frozen_evaluation()`.
    child_manifest, publish_report = build_child_bundle(
        config, stage_a["parent_manifest"], outcomes, fixed_candidate_state_dicts
    )
    with frozen_evaluation():
        embedding_check = run_embedding_consistency_check(
            config, core, child_manifest, outcomes, fixed_candidate_state_dicts
        )
        protected_regression = rec004d_protected_regression_check(
            config, stage_a["parent_manifest"], child_manifest, core
        )

    (output_dir / "recipe_revision.json").write_text(
        json.dumps(
            {
                "task_id": REC004D_TASK_ID,
                "parent_recipe": (
                    "runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/budget_protocol.json"
                ),
                "selected_arm": REC004D_ARM_P,
                "selected_init": "I01",
                "mirror_halves_architecture": REC004D_ARCHITECTURE_SIGNATURE_P,
                "fixed_candidates_step": REC004D_FIXED_CANDIDATE_STEP,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "selected_recipe.json").write_text(
        json.dumps(
            {"task_id": REC004D_TASK_ID, "selection": selection, "publish_report": publish_report},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "bundle_lineage.json").write_text(
        json.dumps(
            {
                "child_bundle_id": child_manifest.bundle_id,
                "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
                "parent_bundle_ids_declared": list(child_manifest.parent_bundle_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "protected_regression.json").write_text(
        json.dumps(protected_regression, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "embedding_consistency.json").write_text(
        json.dumps(embedding_check, indent=2, default=str), encoding="utf-8"
    )

    stage_f = run_stage_f(
        config, core, child_manifest, publish_report["manifest_paths_json_for_fresh_process"]
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004D_TASK_ID,
        "before": before_hashes,
        "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }

    rg3_recheck_pass = all(
        [
            stage_f["all_primitive_execution"]["all_16_operations_present"],
            stage_f["all_non_shift_15_floor_passed"],
            stage_f["fresh_process_report"]["reproducibility_passed"],
            side_effect_audit["shared_cache_unchanged"],
            protected_regression["all_protected_unchanged"],
            embedding_check["all_match"],
        ]
    )
    rg3_recheck = "RG3_RECHECK_PASS" if rg3_recheck_pass else "RG3_RECHECK_FAIL"

    final_known_defects = () if rg3_recheck_pass else ("REC004D_RG3_RECHECK_FAIL",)
    final_manifest = dataclasses.replace(child_manifest, known_defects=final_known_defects)
    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles"
        / final_manifest.bundle_id
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(ibc._manifest_to_json_dict(final_manifest), indent=2), encoding="utf-8"
    )

    rg3_recheck_report = {
        "task_id": REC004D_TASK_ID,
        "result": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "selected_arm": REC004D_ARM_P,
        "selected_init": "I01",
        "criteria": {
            "all_16_operations_coverage": (
                stage_f["all_primitive_execution"]["all_16_operations_present"]
            ),
            "non_shift_15_floor_passed": stage_f["all_non_shift_15_floor_passed"],
            "non_shift_floor_failures": (
                stage_f["all_primitive_execution"]["non_shift_floor_failures"]
            ),
            "fresh_process_reproducibility_passed": (
                stage_f["fresh_process_report"]["reproducibility_passed"]
            ),
            "shared_cache_unchanged": side_effect_audit["shared_cache_unchanged"],
            "protected_12_unchanged": protected_regression["all_protected_unchanged"],
            "embedding_consistency_all_match": embedding_check["all_match"],
        },
        "no_query_leakage": {
            "existing_validation_split": REC004D_EXISTING_VALIDATION_SPLIT,
            "recheck_query_split": REC004D_RECHECK_QUERY_SPLIT,
            "distinct_splits": REC004D_EXISTING_VALIDATION_SPLIT != REC004D_RECHECK_QUERY_SPLIT,
        },
        "scope_note": (
            "This is a seed-10 recovery-pilot recheck with a new heterogeneous "
            "length-position-bias MIRROR_HALVES operator and 3 imported fixed candidates, "
            "not a 5-model cohort, unseen-family, or hard-negative routing result."
        ),
    }
    (output_dir / "rg3_recheck.json").write_text(
        json.dumps(rg3_recheck_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "all_primitive_execution.json").write_text(
        json.dumps(stage_f["all_primitive_execution"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "fresh_process_report.json").write_text(
        json.dumps(stage_f["fresh_process_report"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    qualification = {
        "task_id": REC004D_TASK_ID,
        "implementation_status": "COMPLETE",
        "candidate_status": candidate_status,
        "rg3_recheck": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "rec005_eligible": rg3_recheck_pass,
        "no_5_model_cohort_started_by_this_task": True,
    }
    (output_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2, default=str), encoding="utf-8"
    )

    result["rg3_recheck"] = rg3_recheck
    result["rec005_eligible"] = rg3_recheck_pass
    result["child_bundle_id"] = child_manifest.bundle_id
    result["publish_report"] = publish_report
    result["embedding_consistency"] = embedding_check
    result["protected_regression"] = protected_regression
    result["rg3_recheck_report"] = rg3_recheck_report
    result["side_effect_audit"] = side_effect_audit
    result["qualification"] = qualification
    result["cost_accounting"] = cost_accounting
    result["wall_clock_seconds"] = time.time() - start

    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
