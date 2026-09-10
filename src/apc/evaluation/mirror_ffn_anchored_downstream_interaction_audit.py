"""B-C005REC-004L: FFN-Anchored Downstream Interaction Audit.

Follows the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004K`/ADR-0106 and the environment-repair sub-task
`B-C005REC-004L-ENV1`/ADR-0107) as `B-C005REC-004L`.

`B-C005REC-004K` (ADR-0106) found I03@17500's position-4 oracle-attention
residual is `DISTRIBUTED_DOWNSTREAM_COADAPTATION`: no single component group
(`VALUE_OUTPROJ`, `QUERY_RESIDUAL_PATH`, `POST_ATTN_NORM`, `FFN_BLOCK`,
`READOUT`) rolled back alone from step=17500 to step=6000 clears the 0.95
oracle-EM / position-4-accuracy floor -- though `FFN_BLOCK` alone was by far
the strongest single lever (oracle EM 0.75 -> 0.89-0.90), leaving a residual
~5-6 point shortfall. This task asks exactly one question: which downstream
component's INTERACTION with `FFN_BLOCK` explains that remaining shortfall?
`FFN_BLOCK` is fixed as the base and tested jointly with exactly ONE partner
component at a time -- four pre-registered pairs -- never a 3-component
combination, never a full subset/powerset search, and no combination is
added after seeing a result.

**Zero new optimizer updates.** Loads the SAME already-saved I03 checkpoints
REC-004K used (step=6000 from `B-C005REC-004D`'s tree, step=17500 from
`B-C005REC-004H`'s), hash-verifies each, and reuses REC-004K's own
forward-graph-derived 5-group parameter partition and rollback-merge
mechanism unmodified. Every condition below is a fresh, in-memory,
evaluation-only `CrossPositionLengthBiasPrimitive` merged from two real
state dicts -- never a new checkpoint file, never a mutation of either
loaded primitive in place. `selected_init`, `selected_step`,
`selected_component`, and `child_bundle` stay `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`; `rec005_eligible` stays `false`, unconditionally.
"""

from __future__ import annotations

import hashlib
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
    "REC004L_TASK_ID",
    "REC004L_SOURCE_TASK_IDS",
    "REC004L_DECISIVE_INIT",
    "REC004L_EARLY_STEP",
    "REC004L_LATE_STEP",
    "REC004L_TARGET_LENGTH",
    "REC004L_CONDITION_IDS",
    "REC004L_PAIR_CONDITION_IDS",
    "REC004L_CONDITION_TO_COMPONENTS",
    "REC004L_DATASETS",
    "REC004L_NEW_PROBE_SPLIT",
    "REC004L_NEW_PROBE_EXAMPLES",
    "REC004L_ROLLBACK_EM_THRESHOLD",
    "REC004L_ROLLBACK_POSITION_ACC_THRESHOLD",
    "MirrorFfnAnchoredDownstreamInteractionAuditConfig",
    "run_mirror_ffn_anchored_downstream_interaction_audit_task",
]

# =============================================================================
# Constants. Checkpoint identity, dataset identity for the two pre-existing
# datasets, the component grouping, and the 0.95 EM/position-4 floor are all
# inherited from REC-004K (never retyped). Only the FFN-anchored condition
# set, the new probe dataset, and the position-4-centered decision labels are
# new -- pre-registered here, before any point below is computed.
# =============================================================================

REC004L_TASK_ID: Final = "B-C005REC-004L"
REC004L_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004I", "B-C005REC-004J", "B-C005REC-004K", "B-C005REC-004L-ENV1",
)
REC004L_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_FFN_ANCHORED_DOWNSTREAM_INTERACTION_AUDIT.md"
)

REC004L_TARGET_OPERATION: Final = rec004k.REC004K_TARGET_OPERATION  # "MIRROR_HALVES"
REC004L_ARM: Final = rec004k.REC004K_ARM  # "P_LENGTH_POSITION_BIAS"
REC004L_TARGET_LENGTH: Final = rec004k.REC004K_TARGET_LENGTH  # 10
REC004L_SEED: Final = RECOVERY_PILOT_SEED

REC004L_DECISIVE_INIT: Final = rec004k.REC004K_DECISIVE_INIT  # "I03"
REC004L_EARLY_STEP: Final = rec004k.REC004K_EARLY_STEP  # 6000
REC004L_LATE_STEP: Final = rec004k.REC004K_LATE_STEP  # 17500

REC004L_CLEAN_V2_DATASET: Final = rec004k.REC004K_CLEAN_V2_DATASET
REC004L_PROBE_DATASET: Final = rec004k.REC004K_PROBE_DATASET
REC004L_NEW_PROBE_SPLIT: Final = "length10_downstream_interaction_probe_v1"
REC004L_NEW_PROBE_EXAMPLES: Final = 512
REC004L_DATASETS: Final[tuple[str, ...]] = (
    REC004L_CLEAN_V2_DATASET, REC004L_PROBE_DATASET, REC004L_NEW_PROBE_SPLIT,
)

REC004L_POSITION_MAIN_RESIDUAL: Final = rec004k.REC004K_POSITION_MAIN_RESIDUAL  # 4
REC004L_POSITION_SYMMETRIC_CONTROL: Final = rec004k.REC004K_POSITION_SYMMETRIC_CONTROL  # 5
REC004L_REGRESSION_POSITIONS: Final[tuple[int, ...]] = (0, 1, 2, 3, 6, 7, 8, 9)

REC004L_ROLLBACK_EM_THRESHOLD: Final = rec004k.REC004K_ROLLBACK_EM_THRESHOLD  # 0.95
REC004L_ROLLBACK_POSITION_ACC_THRESHOLD: Final = (
    rec004k.REC004K_ROLLBACK_POSITION_ACC_THRESHOLD
)  # 0.95
REC004L_PARITY_LOGIT_ABS_TOL: Final = rec004k.REC004K_PARITY_LOGIT_ABS_TOL  # 5e-3

REC004L_CHUNK_SIZE: Final = rec004k.REC004K_CHUNK_SIZE  # 128
REC004L_INVARIANCE_CHECK_EXAMPLES: Final = rec004k.REC004K_INVARIANCE_CHECK_EXAMPLES

# Stage A -- pre-registered FFN-anchored condition set (task doc Section 3).
# Exactly R0 / F / four FFN+partner pairs / R_ALL / EARLY -- no 3-component
# combination, no full subset/powerset search, no combination added after
# seeing a result.
REC004L_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_V", "F_Q", "F_N", "F_R", "R_ALL", "EARLY",
)
REC004L_CONDITION_TO_COMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    "R0": (),
    "F": ("FFN_BLOCK",),
    "F_V": ("FFN_BLOCK", "VALUE_OUTPROJ"),
    "F_Q": ("FFN_BLOCK", "QUERY_RESIDUAL_PATH"),
    "F_N": ("FFN_BLOCK", "POST_ATTN_NORM"),
    "F_R": ("FFN_BLOCK", "READOUT"),
    "R_ALL": rec004k.REC004K_COMPONENT_IDS,
}
REC004L_PAIR_CONDITION_IDS: Final[tuple[str, ...]] = ("F_V", "F_Q", "F_N", "F_R")
# Every condition except "EARLY" is built as an in-memory merge (possibly an
# empty merge, for R0) of I03@17500's and I03@6000's real state dicts.
REC004L_MERGE_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_V", "F_Q", "F_N", "F_R", "R_ALL",
)


def _config_to_yaml_dict(
    config: MirrorFfnAnchoredDownstreamInteractionAuditConfig,
) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


@dataclass(frozen=True)
class MirrorFfnAnchoredDownstreamInteractionAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004l/run_001")
    seed: int = RECOVERY_PILOT_SEED


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, points: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for key, row in points.items():
            fh.write(json.dumps({"key": key, **row}, default=str) + "\n")


# =============================================================================
# Stage B -- one new disjoint dataset for paired continuity with
# `clean_v2_length10` / `length10_mechanism_probe_v1` (both reused
# byte-identically, unmodified, from REC-004K). Follows
# `rec004j.build_length10_mechanism_probe_v1`'s exact collision-substitution
# recipe verbatim; only the split label (hence the RNG stream, via
# `_derive_local_seed`) and the (larger) protected set differ. A pure
# function of `(seed, protected_digests)` -- never conditioned on any
# checkpoint prediction or rollback result, so its digest can be frozen
# before any Stage A/C point is computed.
# =============================================================================


def build_length10_downstream_interaction_probe_v1(
    seed: int, protected_digests: set[str], n: int = REC004L_NEW_PROBE_EXAMPLES
) -> tuple[list[Example], dict[str, Any]]:
    seed_label = f"{REC004L_NEW_PROBE_SPLIT}:{REC004L_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed(seed, 0, seed_label))
    op_obj = get_operation(REC004L_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq = tuple(
                rng.randrange(rec004j.REC004J_VOCAB_SIZE) for _ in range(REC004L_TARGET_LENGTH)
            )
            params = op_obj.sample_params(rng, seq, rec004j.REC004J_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004L_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, rec004j.REC004J_VOCAB_SIZE)
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
                split=REC004L_NEW_PROBE_SPLIT,
                vocab_size=rec004j.REC004J_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004L_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "target_length": REC004L_TARGET_LENGTH,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            f"{REC004L_NEW_PROBE_SPLIT} is a development-exposed diagnostic set for "
            "B-C005REC-004L, not a sealed or final RG3 query set; once consumed by "
            "this task it is not eligible to serve as an independent RG3 recheck "
            "query without a fresh, disjoint regeneration."
        ),
    }
    return examples, detail


def _dataset_digest(examples: list[Example]) -> str:
    """A single frozen fingerprint for the whole set -- sorted per-example
    digests, hashed together -- recorded before any rollback condition is
    evaluated on it (pre-registration discipline)."""
    per_example = sorted(traj_audit._digest_examples(examples))
    return hashlib.sha256(json.dumps(per_example).encode("utf-8")).hexdigest()


def build_stage_b_datasets(seed: int) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Byte-identical reuse of REC-004K's own two datasets (paired
    continuity) plus ONE new dataset disjoint from: the training stream
    (steps 1-18000), old validation sets, every REC-004A-D reference/sealed
    split (all via `traj_audit.build_protected_digest_registry`),
    `clean_selection_validation_v2`'s full 1024-example set (which
    `clean_v2_length10` is a length-10 subset of), and
    `length10_mechanism_probe_v1` itself (REC-004J/004K's own probe)."""
    datasets_ab, clean_v2_detail, probe_detail = rec004k.build_stage_a_datasets(seed)
    all_1024, _clean_v2_len10_dup, _clean_v2_detail_dup, _counts_dup = (
        rec004j.regenerate_clean_selection_validation_v2(seed)
    )
    protected_v1, _counts_v1 = traj_audit.build_protected_digest_registry(seed)
    protected_v2 = protected_v1 | traj_audit._digest_examples(all_1024)
    protected_v3 = protected_v2 | traj_audit._digest_examples(datasets_ab[REC004L_PROBE_DATASET])

    new_probe_examples, new_probe_detail = build_length10_downstream_interaction_probe_v1(
        seed, protected_v3
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
            new_probe_digests & traj_audit._digest_examples(datasets_ab[REC004L_PROBE_DATASET])
        ),
    }
    overlap["disjoint"] = not any(v for v in overlap.values())
    new_probe_detail["disjointness_check"] = overlap
    new_probe_detail["frozen_dataset_digest_sha256"] = _dataset_digest(new_probe_examples)

    datasets = {**datasets_ab, REC004L_NEW_PROBE_SPLIT: new_probe_examples}
    dataset_details = {
        REC004L_CLEAN_V2_DATASET: clean_v2_detail,
        REC004L_PROBE_DATASET: probe_detail,
        REC004L_NEW_PROBE_SPLIT: new_probe_detail,
    }
    return datasets, dataset_details


# =============================================================================
# Stage A -- build the 8 fixed conditions from I03@17500's and I03@6000's
# real, hash-verified state dicts, reusing REC-004K's forward-graph parameter
# partition and merge mechanism unmodified.
# =============================================================================


def build_condition_primitives(
    core: Any,
    late_state_dict: dict[str, torch.Tensor],
    early_state_dict: dict[str, torch.Tensor],
    key_groups: dict[str, list[str]],
    early_primitive: Any,
) -> dict[str, Any]:
    primitives: dict[str, Any] = {
        condition_id: rec004k._build_rollback_primitive(
            core, late_state_dict, early_state_dict,
            REC004L_CONDITION_TO_COMPONENTS[condition_id], key_groups,
        )
        for condition_id in REC004L_MERGE_CONDITION_IDS
    }
    primitives["EARLY"] = early_primitive
    return primitives


def _run_o1_with_hidden(
    primitive: Any,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, torch.Tensor]:
    """O1 (oracle-attention) forward that ALSO returns the hidden state
    immediately after the attention output (before `attn_norm`/`ffn`/
    `readout`). `oracle_probe.run_oracle_forward` only returns final
    logits; this composes the same real helpers it is itself built from
    (`resid_audit._prepare_query_kv`, `oracle_probe._oracle_attention`,
    `resid_audit._post_attention`), unmodified, to also expose that
    intermediate tensor."""
    resid_audit._assert_eval_only()
    lengths = set(content_lengths)
    if len(lengths) != 1:
        raise AssertionError(
            f"oracle substitution requires a single-length batch, got {sorted(lengths)}"
        )
    n = lengths.pop()
    query, kv, _out_max, _lmax, _batch = resid_audit._prepare_query_kv(
        primitive, content_features, content_lengths, output_lengths
    )
    attn_out, attn_probs = oracle_probe._oracle_attention(primitive, query, kv, n)
    logits = resid_audit._post_attention(primitive, query, attn_out)
    return {"attn_out": attn_out, "logits": logits, "attn_probs": attn_probs}


def compute_interaction_diagnostic_point(
    core: Any,
    primitive: Any,
    reference_primitive: Any,
    examples: list[Example],
) -> dict[str, Any]:
    """O1-only diagnostic point (task doc Section 7 metric set): sequence
    EM, per-output-position (0-9) accuracy and mean target-token logit
    margin, PLUS -- against `reference_primitive` (I03's own real
    EARLY=step=6000 checkpoint, run on the SAME examples in the SAME loop)
    -- the max-abs hidden-state difference immediately after the O1
    attention output and the max-abs final-logit difference. When
    `primitive is reference_primitive` (the EARLY condition itself) both
    diffs are trivially `0.0` -- reported, not skipped, as an identity
    check."""
    resid_audit._assert_eval_only()
    n = REC004L_TARGET_LENGTH
    device = core.device
    op = get_operation(REC004L_TARGET_OPERATION)

    o1_correct: list[bool] = []
    pos_o1_correct = [0] * n
    o1_margins: list[list[float]] = [[] for _ in range(n)]
    max_hidden_diff = 0.0
    max_logit_diff = 0.0

    for start in range(0, len(examples), REC004L_CHUNK_SIZE):
        chunk = examples[start : start + REC004L_CHUNK_SIZE]
        content_lengths = [n] * len(chunk)
        output_lengths = [op.output_length(n)] * len(chunk)
        assert all(ol == n for ol in output_lengths)
        batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
        with torch.no_grad():
            h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
            o1 = _run_o1_with_hidden(primitive, h_content, content_lengths, output_lengths)
            ref = _run_o1_with_hidden(
                reference_primitive, h_content, content_lengths, output_lengths
            )

        hidden_diff = (o1["attn_out"] - ref["attn_out"]).abs().max().item()
        logit_diff = (o1["logits"] - ref["logits"]).abs().max().item()
        max_hidden_diff = max(max_hidden_diff, hidden_diff)
        max_logit_diff = max(max_logit_diff, logit_diff)

        o1_preds = resid_audit._predict_from_logits(o1["logits"], output_lengths)
        target_tokens = [list(ex.target_tokens[:n]) for ex in chunk]
        chunk_margins = rec004k._position_margins_batch(o1["logits"], target_tokens, n)
        for k in range(n):
            o1_margins[k].extend(chunk_margins[k])

        for row, ex in enumerate(chunk):
            target = tuple(ex.target_tokens[:n])
            pred = tuple(o1_preds[row])
            o1_correct.append(pred == target)
            for k in range(n):
                if pred[k] == target[k]:
                    pos_o1_correct[k] += 1

    n_total = len(examples)
    o1_em = sum(o1_correct) / n_total if n_total else None
    return {
        "n": n_total,
        "oracle_sequence_exact_match": o1_em,
        "max_abs_hidden_state_diff_vs_early": max_hidden_diff,
        "max_abs_logit_diff_vs_early": max_logit_diff,
        "per_output_position": {
            str(k): {
                "oracle_accuracy": pos_o1_correct[k] / n_total if n_total else None,
                "oracle_mean_target_logit_margin": (
                    sum(o1_margins[k]) / len(o1_margins[k]) if o1_margins[k] else None
                ),
            }
            for k in range(n)
        },
    }


def run_stage_a_ffn_anchored_conditions(
    core: Any,
    condition_primitives: dict[str, Any],
    datasets: dict[str, list[Example]],
) -> dict[str, Any]:
    """Runs the fixed 8-condition set (task doc Section 3) across all three
    datasets -- no combination search, no additional condition added after
    seeing a result."""
    reference_primitive = condition_primitives["EARLY"]
    points: dict[str, Any] = {}
    for condition_id in REC004L_CONDITION_IDS:
        primitive = condition_primitives[condition_id]
        for dataset_name in REC004L_DATASETS:
            examples = datasets[dataset_name]
            with torch.no_grad():
                point = compute_interaction_diagnostic_point(
                    core, primitive, reference_primitive, examples
                )
            point.update(
                {
                    "condition_id": condition_id,
                    "components_rolled_back": list(
                        REC004L_CONDITION_TO_COMPONENTS.get(condition_id, ())
                    ),
                    "dataset": dataset_name,
                }
            )
            points[f"{condition_id}:{dataset_name}"] = point
    return points


def cross_check_against_rec004k(condition_points: dict[str, Any]) -> dict[str, Any]:
    """R0 (=I03@17500 as-is) and EARLY (=I03@6000's own real forward) on the
    two shared datasets must reproduce REC-004K's own recorded R0/step=6000
    numbers exactly -- both tasks run the identical checkpoints through the
    identical O1 forward on the identical (byte-identically regenerated)
    datasets. Informational only; never gates this task's own decision."""
    rec004k_run_dir = Path("runs/phase_b_b2_model_bundle_recovery/rec004k/run_001")
    rollback_path = rec004k_run_dir / "stage_c_rollback_points_summary.json"
    stage_a_path = rec004k_run_dir / "stage_a_temporal_points_summary.json"
    if not rollback_path.is_file() or not stage_a_path.is_file():
        return {"status": "REC004K_ARTIFACT_UNAVAILABLE"}
    rollback_recorded = json.loads(rollback_path.read_text(encoding="utf-8"))
    stage_a_recorded = json.loads(stage_a_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    shared_datasets = (REC004L_CLEAN_V2_DATASET, REC004L_PROBE_DATASET)
    for dataset_name in shared_datasets:
        r0_recorded = rollback_recorded.get(f"R0:{dataset_name}", {}).get(
            "oracle_sequence_exact_match"
        )
        r0_recomputed = condition_points.get(f"R0:{dataset_name}", {}).get(
            "oracle_sequence_exact_match"
        )
        if r0_recorded is None or r0_recomputed is None or abs(r0_recorded - r0_recomputed) >= 1e-9:
            mismatches.append(
                {
                    "condition": "R0", "dataset": dataset_name,
                    "recorded": r0_recorded, "recomputed": r0_recomputed,
                }
            )
        early_recorded = stage_a_recorded.get(f"{REC004L_EARLY_STEP}:{dataset_name}", {}).get(
            "oracle_sequence_exact_match"
        )
        early_recomputed = condition_points.get(f"EARLY:{dataset_name}", {}).get(
            "oracle_sequence_exact_match"
        )
        if (
            early_recorded is None
            or early_recomputed is None
            or abs(early_recorded - early_recomputed) >= 1e-9
        ):
            mismatches.append(
                {
                    "condition": "EARLY", "dataset": dataset_name,
                    "recorded": early_recorded, "recomputed": early_recomputed,
                }
            )
    return {"status": "VERIFIED" if not mismatches else "MISMATCH", "mismatches": mismatches}


def build_regression_check(condition_points: dict[str, Any]) -> dict[str, Any]:
    """Descriptive-only check (task doc Section 7): confirms no FFN-anchored
    condition breaks a position R0 already got right, at positions other
    than the two decisive ones (4, 5). Never gates the decision below."""
    report: dict[str, Any] = {}
    for condition_id in REC004L_CONDITION_IDS:
        if condition_id == "R0":
            continue
        for dataset_name in REC004L_DATASETS:
            r0_row = condition_points.get(f"R0:{dataset_name}", {})
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            regressed: list[dict[str, Any]] = []
            for pos in REC004L_REGRESSION_POSITIONS:
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
# Decision, centered on output position 4 (task doc Section 7).
# =============================================================================


def build_ffn_interaction_decision(
    condition_points: dict[str, Any], parity_check: dict[str, Any]
) -> dict[str, Any]:
    if parity_check["status"] != "VERIFIED":
        return {
            "task_id": REC004L_TASK_ID,
            "label": "COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP",
            "sufficient_pairs": [],
            "r_all_passes": False,
            "note": (
                "R_ALL did not reproduce I03@6000's own real O1 forward within "
                "tolerance on at least one dataset -- REC-004K's 5-group "
                "decomposition (reused unmodified here) is incomplete or "
                "incorrect; F/F+X/R_ALL results are recorded but NOT interpreted "
                "as a causal finding"
            ),
        }

    pos_key = str(REC004L_POSITION_MAIN_RESIDUAL)

    def _passes(condition_id: str) -> bool:
        for dataset_name in REC004L_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            em = row.get("oracle_sequence_exact_match")
            pos4 = row.get("per_output_position", {}).get(pos_key, {}).get("oracle_accuracy")
            if em is None or pos4 is None:
                return False
            if em < REC004L_ROLLBACK_EM_THRESHOLD or pos4 < REC004L_ROLLBACK_POSITION_ACC_THRESHOLD:
                return False
        return True

    sufficient = [cid for cid in REC004L_PAIR_CONDITION_IDS if _passes(cid)]
    r_all_passes = _passes("R_ALL")

    if len(sufficient) == 1:
        partner = next(
            c for c in REC004L_CONDITION_TO_COMPONENTS[sufficient[0]] if c != "FFN_BLOCK"
        )
        label = f"FFN_{partner}_INTERACTION_SUFFICIENT"
    elif len(sufficient) > 1:
        label = "MULTIPLE_FFN_INTERACTION_PATHS_SUFFICIENT"
    elif r_all_passes:
        label = "HIGHER_ORDER_OR_MULTI_COMPONENT_COADAPTATION"
    else:
        label = "DOWNSTREAM_DECOMPOSITION_INCOMPLETE"

    return {
        "task_id": REC004L_TASK_ID,
        "label": label,
        "sufficient_pairs": sufficient,
        "r_all_passes": r_all_passes,
        "oracle_em_threshold": REC004L_ROLLBACK_EM_THRESHOLD,
        "position4_accuracy_threshold": REC004L_ROLLBACK_POSITION_ACC_THRESHOLD,
    }


# =============================================================================
# Report.
# =============================================================================


def build_report_markdown(
    condition_points: dict[str, Any],
    forward_graph_check: dict[str, Any],
    parity_check: dict[str, Any],
    decision: dict[str, Any],
    cross_check: dict[str, Any],
    dataset_details: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# {REC004L_TASK_ID}: FFN-Anchored Downstream Interaction Audit",
        "",
        "Zero new optimizer updates. Forward-only, O1 (oracle-attention) "
        "condition; every rollback is an in-memory, evaluation-only merged "
        "state dict.",
        "",
        "## Datasets",
        "",
        f"- `{REC004L_CLEAN_V2_DATASET}` (n=230) and `{REC004L_PROBE_DATASET}` "
        "(n=512): byte-identical reuse from REC-004K.",
        f"- `{REC004L_NEW_PROBE_SPLIT}` (n="
        f"{dataset_details[REC004L_NEW_PROBE_SPLIT]['n']}): new, disjoint "
        f"(disjointness_check.disjoint="
        f"{dataset_details[REC004L_NEW_PROBE_SPLIT]['disjointness_check']['disjoint']}), "
        "development-exposed diagnostic set, frozen digest="
        f"{dataset_details[REC004L_NEW_PROBE_SPLIT]['frozen_dataset_digest_sha256']}",
        "",
        f"- Forward-graph invariance check: invariant="
        f"{forward_graph_check['invariant']}, "
        f"weights_restored_exactly={forward_graph_check['weights_restored_exactly']}",
        f"- Cross-check vs REC-004K's own R0/step={REC004L_EARLY_STEP} numbers: "
        f"{cross_check.get('status')}",
        "",
        "## Stage A/C -- FFN-anchored conditions (I03@17500 base, O1)",
        "",
        "| condition | dataset | n | O1 EM | pos4 acc | pos5 acc | hidden diff | logit diff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition_id in REC004L_CONDITION_IDS:
        for dataset_name in REC004L_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            pos4 = row.get("per_output_position", {}).get(
                str(REC004L_POSITION_MAIN_RESIDUAL), {}
            ).get("oracle_accuracy")
            pos5 = row.get("per_output_position", {}).get(
                str(REC004L_POSITION_SYMMETRIC_CONTROL), {}
            ).get("oracle_accuracy")
            lines.append(
                f"| {condition_id} | {dataset_name} | {row.get('n')} | "
                f"{row.get('oracle_sequence_exact_match'):.4f} | "
                f"{pos4:.4f} | {pos5:.4f} | "
                f"{row.get('max_abs_hidden_state_diff_vs_early'):.6f} | "
                f"{row.get('max_abs_logit_diff_vs_early'):.6f} |"
            )

    lines += ["", "## Component decomposition parity check (R_ALL vs I03@6000 real O1)"]
    lines.append(f"- status: **{parity_check['status']}**")
    for dataset_name, row in parity_check["per_dataset"].items():
        lines.append(
            f"  - {dataset_name}: max_abs_logit_diff={row['max_abs_logit_diff']:.6f}, "
            f"predictions_match={row['predictions_match']}, passed={row['passed']}"
        )

    lines += [
        "",
        "## FFN-anchored interaction decision",
        f"- label: **{decision['label']}**",
        f"- sufficient pairs: {decision.get('sufficient_pairs', [])}",
        f"- R_ALL passes: {decision.get('r_all_passes')}",
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


def run_mirror_ffn_anchored_downstream_interaction_audit_task(
    config: MirrorFfnAnchoredDownstreamInteractionAuditConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_ffn_anchored_downstream_interaction_audit_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != REC004L_SEED:
        raise ValueError(
            f"B-C005REC-004L reads REC-004D/H/K artifacts pre-registered under seed "
            f"{REC004L_SEED}; got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    # Stage B: datasets are generated and their digests frozen to disk BEFORE
    # any condition/rollback point below is computed (pre-registration).
    datasets, dataset_details = build_stage_b_datasets(seed)
    for dataset_name, detail in dataset_details.items():
        _write_json(output_dir / f"{dataset_name}_manifest.json", detail)

    late_primitive, late_load_info = rec004j._load_and_verify_primitive(
        core, REC004L_DECISIVE_INIT, REC004L_LATE_STEP
    )
    early_primitive, early_load_info = rec004j._load_and_verify_primitive(
        core, REC004L_DECISIVE_INIT, REC004L_EARLY_STEP
    )
    checkpoint_load_info = {
        str(REC004L_LATE_STEP): late_load_info,
        str(REC004L_EARLY_STEP): early_load_info,
    }
    _write_json(output_dir / "checkpoint_load_info.json", checkpoint_load_info)

    all_keys = list(late_primitive.state_dict().keys())
    component_key_groups = rec004k.partition_state_dict_keys(all_keys)
    _write_json(output_dir / "component_key_groups.json", component_key_groups)

    probe_examples_for_audit = datasets[REC004L_PROBE_DATASET][:REC004L_INVARIANCE_CHECK_EXAMPLES]
    op = get_operation(REC004L_TARGET_OPERATION)
    n = REC004L_TARGET_LENGTH
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
    stage_status = "NOT_EXECUTED_FORWARD_GRAPH_INVARIANCE_CHECK_FAILED"

    if forward_graph_check["invariant"] and forward_graph_check["weights_restored_exactly"]:
        late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
        early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}

        condition_primitives = build_condition_primitives(
            core, late_sd, early_sd, component_key_groups, early_primitive
        )

        with torch.no_grad():
            condition_points = run_stage_a_ffn_anchored_conditions(
                core, condition_primitives, datasets
            )
        _write_jsonl(output_dir / "condition_points.jsonl", condition_points)
        _write_json(output_dir / "condition_points_summary.json", condition_points)

        cross_check = cross_check_against_rec004k(condition_points)
        _write_json(output_dir / "cross_check_against_rec004k.json", cross_check)

        with torch.no_grad():
            parity_check = rec004k.run_component_decomposition_parity_check(
                core, condition_primitives["R_ALL"], early_primitive, datasets
            )
        _write_json(output_dir / "component_decomposition_parity_check.json", parity_check)

        decision = build_ffn_interaction_decision(condition_points, parity_check)
        _write_json(output_dir / "ffn_interaction_decision.json", decision)

        regression_check = build_regression_check(condition_points)
        _write_json(output_dir / "regression_check.json", regression_check)

        stage_status = "EXECUTED"

    protected_hashes_after = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    core_hash_after = mb.canonical_state_hash(core.model.state_dict())
    freeze_audit = {
        "task_id": REC004L_TASK_ID,
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
        "task_id": REC004L_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    n_predictions = 0
    if condition_points is not None:
        n_predictions = sum(
            row.get("n", 0) for row in condition_points.values() if isinstance(row, dict)
        ) * 2  # primitive forward + reference forward, per point
    cost_accounting = {
        "task_id": REC004L_TASK_ID,
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
        "task_id": REC004L_TASK_ID,
        "source_task_ids": list(REC004L_SOURCE_TASK_IDS),
        "contract_file": str(REC004L_CONTRACT_FILE),
        "decisive_init": REC004L_DECISIVE_INIT,
        "early_step": REC004L_EARLY_STEP,
        "late_step": REC004L_LATE_STEP,
        "position_main_residual": REC004L_POSITION_MAIN_RESIDUAL,
        "position_symmetric_control": REC004L_POSITION_SYMMETRIC_CONTROL,
        "condition_ids": list(REC004L_CONDITION_IDS),
        "pair_condition_ids": list(REC004L_PAIR_CONDITION_IDS),
        "datasets": list(REC004L_DATASETS),
        "forbidden": [
            "new optimizer updates", "any I03 checkpoint file mutation",
            "any 3-component combination", "full subset/powerset search",
            "combination added after seeing a result",
            "candidate adoption", "child assembly", "RG3 recheck",
            "any further repair auto-started from this audit's result",
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
        "cross_check_status": cross_check.get("status") if cross_check else None,
        "stage_status": stage_status,
        "component_decomposition_parity_status": (
            parity_check["status"] if parity_check is not None else None
        ),
        "ffn_interaction_decision": decision["label"] if decision is not None else None,
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

    if condition_points is not None and decision is not None and parity_check is not None:
        report_md = build_report_markdown(
            condition_points, forward_graph_check, parity_check, decision,
            cross_check or {"status": "NOT_CHECKED"}, dataset_details,
        )
    else:
        report_md = (
            f"# {REC004L_TASK_ID}: FFN-Anchored Downstream Interaction Audit\n\n"
            f"Stage A/C NOT executed: {stage_status}.\n"
        )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    return result
