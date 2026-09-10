"""B-C005REC-004N: I03 FFN-Value-Path Leave-One-Out Necessity Audit &
Freezeability Gate.

Follows
`docs/CODEX_TASKS_PHASE_B_B2_FFN_VALUE_PATH_LEAVE_ONE_OUT_NECESSITY_AUDIT.md`.

`B-C005REC-004M` (ADR-0109) found `value_path_subcomponent_decision:
DISTRIBUTED_WITHIN_VALUE_PATH`: `F_ALLV` (FFN_BLOCK + CONTENT_PREP +
V_PROJECTION + ATTN_OUT_PROJ + SCORE_PROJECTION_CONTROL) reproduces
REC-004L's own perfect recovery, but no single VALUE_OUTPROJ subcomponent
suffices alone; `CONTENT_PREP`/`ATTN_OUT_PROJ` rolled back alone each
actively hurt, and `SCORE_PROJECTION_CONTROL` (Q/K) is exactly O1-invariant.
Since Q/K never enters O1's forward graph, `F_ALLV`'s O1 result already IS
`FFN_BLOCK + CONTENT_PREP + V_PROJECTION + ATTN_OUT_PROJ` (this task's own
`F_CVO`) under O1.

This task searches the OTHER direction: starting from the already-
established sufficient set `F_CVO`, remove exactly one of the three
VALUE_OUTPROJ subcomponents at a time (`F_CV`, `F_CO`, `F_VO`) and ask which
pairwise subset stays sufficient -- in particular, whether `F_VO`
(`FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ`, never touching `CONTENT_PREP`)
is sufficient. `CONTENT_PREP` feeds the SAME `kv` tensor the real, learnable
K path (`SCORE_PROJECTION_CONTROL`) also reads from, so a freeze candidate
that never rolls it back is the first one that could plausibly leave real
attention-score learning untouched. Stage C empirically verifies this
architectural claim: since `V_PROJECTION`/`ATTN_OUT_PROJ`/`FFN_BLOCK` feed
only the post-attention-score path, `F_VO`'s J0 (normal, non-oracle)
attention distribution is expected to be mathematically identical to
`R0`'s (I03@17500's own).

**Zero new optimizer updates.** Reads the SAME already-saved I03 checkpoints
REC-004K/L/M used (step=6000 from `B-C005REC-004D`'s tree, step=17500 from
`B-C005REC-004H`'s), plus -- success-model safety check only, and only if
`F_VO` passes -- I04@18000's and I05@17500's own step=6000/late checkpoints
(never I03's, never any cross-init weight transplant). Every condition is a
fresh, in-memory, evaluation-only `CrossPositionLengthBiasPrimitive` merged
via REC-004M's own row-slice-aware merge function, reused UNMODIFIED --
never a new checkpoint file, never a mutation of either loaded primitive in
place. `selected_init`, `selected_step`, `selected_value_subcomponent`, and
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
from apc.evaluation import mirror_ffn_value_path_subcomponent_attribution as rec004m
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
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
    "REC004N_TASK_ID",
    "REC004N_SOURCE_TASK_IDS",
    "REC004N_DECISIVE_INIT",
    "REC004N_EARLY_STEP",
    "REC004N_LATE_STEP",
    "REC004N_TARGET_LENGTH",
    "REC004N_CONDITION_IDS",
    "REC004N_LEAVE_ONE_OUT_CONDITION_IDS",
    "REC004N_CONDITION_TO_SUBCOMPONENTS",
    "REC004N_LEAVE_ONE_OUT_EXCLUDES",
    "REC004N_DATASETS",
    "REC004N_NEW_PROBE_SPLIT",
    "REC004N_NEW_PROBE_EXAMPLES",
    "REC004N_ROLLBACK_EM_THRESHOLD",
    "REC004N_ROLLBACK_POSITION_ACC_THRESHOLD",
    "REC004N_J0_ATTENTION_ABS_TOL",
    "REC004N_SAFETY_TARGETS",
    "REC004N_SAFETY_EM_DEGRADATION_THRESHOLD",
    "MirrorFfnValuePathLeaveOneOutNecessityAuditConfig",
    "run_mirror_ffn_value_path_leave_one_out_necessity_audit_task",
]

# =============================================================================
# Constants. Checkpoint identity, thresholds, and the pre-existing 4 datasets
# are all inherited from REC-004K/L/M (never retyped). Only the leave-one-out
# condition set, the new 5th probe dataset, the source-replay check, the J0
# attention-equivalence check, and the necessity decision labels are new
# here -- pre-registered before any point below is computed.
# =============================================================================

REC004N_TASK_ID: Final = "B-C005REC-004N"
REC004N_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004K", "B-C005REC-004L", "B-C005REC-004M",
)
REC004N_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_FFN_VALUE_PATH_LEAVE_ONE_OUT_NECESSITY_AUDIT.md"
)

REC004N_TARGET_OPERATION: Final = rec004m.REC004M_TARGET_OPERATION  # "MIRROR_HALVES"
REC004N_ARM: Final = rec004m.REC004M_ARM  # "P_LENGTH_POSITION_BIAS"
REC004N_TARGET_LENGTH: Final = rec004m.REC004M_TARGET_LENGTH  # 10
REC004N_VOCAB_SIZE: Final = rec004m.REC004M_VOCAB_SIZE
REC004N_SEED: Final = RECOVERY_PILOT_SEED

REC004N_DECISIVE_INIT: Final = rec004m.REC004M_DECISIVE_INIT  # "I03"
REC004N_EARLY_STEP: Final = rec004m.REC004M_EARLY_STEP  # 6000
REC004N_LATE_STEP: Final = rec004m.REC004M_LATE_STEP  # 17500

REC004N_CLEAN_V2_DATASET: Final = rec004m.REC004M_CLEAN_V2_DATASET
REC004N_MECHANISM_PROBE_DATASET: Final = rec004m.REC004M_PROBE_DATASET
REC004N_DOWNSTREAM_PROBE_DATASET: Final = rec004m.REC004M_DOWNSTREAM_PROBE_DATASET
REC004N_SUBCOMPONENT_PROBE_DATASET: Final = rec004m.REC004M_NEW_PROBE_SPLIT
REC004N_EXISTING_DATASETS: Final[tuple[str, ...]] = (
    REC004N_CLEAN_V2_DATASET, REC004N_MECHANISM_PROBE_DATASET,
    REC004N_DOWNSTREAM_PROBE_DATASET, REC004N_SUBCOMPONENT_PROBE_DATASET,
)
REC004N_NEW_PROBE_SPLIT: Final = "length10_value_path_necessity_probe_v1"
REC004N_NEW_PROBE_EXAMPLES: Final = 512
REC004N_DATASETS: Final[tuple[str, ...]] = (*REC004N_EXISTING_DATASETS, REC004N_NEW_PROBE_SPLIT)

REC004N_POSITION_MAIN_RESIDUAL: Final = rec004m.REC004M_POSITION_MAIN_RESIDUAL  # 4
REC004N_POSITION_SYMMETRIC_CONTROL: Final = rec004m.REC004M_POSITION_SYMMETRIC_CONTROL  # 5
REC004N_REGRESSION_POSITIONS: Final[tuple[int, ...]] = rec004m.REC004M_REGRESSION_POSITIONS

REC004N_ROLLBACK_EM_THRESHOLD: Final = rec004m.REC004M_ROLLBACK_EM_THRESHOLD  # 0.95
REC004N_ROLLBACK_POSITION_ACC_THRESHOLD: Final = (
    rec004m.REC004M_ROLLBACK_POSITION_ACC_THRESHOLD
)  # 0.95

REC004N_CHUNK_SIZE: Final = rec004m.REC004M_CHUNK_SIZE
REC004N_INVARIANCE_CHECK_EXAMPLES: Final = rec004m.REC004M_INVARIANCE_CHECK_EXAMPLES

# Stage A -- source/parity replay targets (task doc Section 3).
REC004N_SOURCE_REPLAY_EM_TOL: Final = 1e-9
REC004N_SOURCE_REPLAY_F_EM_RANGE: Final = (0.85, 0.92)
REC004N_SOURCE_REPLAY_PERFECT_EM: Final = 1.0
REC004M_RUN_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004m/run_001")

# Stage B -- pre-registered fixed condition set (task doc Section 4). Exactly
# R0 / F / three leave-one-out pairs / F_CVO (the already-established
# sufficient triple) / EARLY -- no combination beyond this set.
REC004N_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_CV", "F_CO", "F_VO", "F_CVO", "EARLY",
)
REC004N_LEAVE_ONE_OUT_CONDITION_IDS: Final[tuple[str, ...]] = ("F_CV", "F_CO", "F_VO")
REC004N_CONDITION_TO_SUBCOMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    "R0": (),
    "F": (),
    "F_CV": ("CONTENT_PREP", "V_PROJECTION"),
    "F_CO": ("CONTENT_PREP", "ATTN_OUT_PROJ"),
    "F_VO": ("V_PROJECTION", "ATTN_OUT_PROJ"),
    "F_CVO": ("CONTENT_PREP", "V_PROJECTION", "ATTN_OUT_PROJ"),
}
REC004N_MERGE_CONDITION_IDS: Final[tuple[str, ...]] = (
    "R0", "F", "F_CV", "F_CO", "F_VO", "F_CVO",
)
# FFN_BLOCK is rolled back for every merge condition except R0.
REC004N_FFN_ROLLED_BACK: Final[dict[str, bool]] = {
    "R0": False, "F": True, "F_CV": True, "F_CO": True, "F_VO": True, "F_CVO": True,
}
# Which VALUE_OUTPROJ subcomponent each leave-one-out condition EXCLUDES
# (i.e. leaves at I03@17500's real, current value).
REC004N_LEAVE_ONE_OUT_EXCLUDES: Final[dict[str, str]] = {
    "F_CV": "ATTN_OUT_PROJ",
    "F_CO": "V_PROJECTION",
    "F_VO": "CONTENT_PREP",
}
REC004N_LEAVE_ONE_OUT_TO_LABEL: Final[dict[str, str]] = {
    "F_CV": "FFN_CV_SUBPATH",
    "F_CO": "FFN_CO_SUBPATH",
    "F_VO": "FFN_V_OUTPROJ_SUBPATH",
}

# Stage C -- J0 attention equivalence for F_VO vs R0 (task doc Section 5).
# A single pre-fixed tolerance; literal 0.0 is the EXPECTED result given
# identical Q/K/CONTENT_PREP/position-bias weights, not merely tolerated.
REC004N_J0_ATTENTION_ABS_TOL: Final = 1e-6

# Success-model safety check (task doc Section 8). Each uses its OWN
# step=6000 state; no cross-init weight transplant.
REC004N_SAFETY_TARGETS: Final[tuple[tuple[str, int], ...]] = (("I04", 18000), ("I05", 17500))
REC004N_SAFETY_EM_DEGRADATION_THRESHOLD: Final = 0.05


def _config_to_yaml_dict(
    config: MirrorFfnValuePathLeaveOneOutNecessityAuditConfig,
) -> dict[str, Any]:
    return {"seed": config.seed, "output_dir": str(config.output_dir)}


@dataclass(frozen=True)
class MirrorFfnValuePathLeaveOneOutNecessityAuditConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004n/run_001")
    seed: int = RECOVERY_PILOT_SEED


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, points: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for key, row in points.items():
            fh.write(json.dumps({"key": key, **row}, default=str) + "\n")


# =============================================================================
# Stage D -- one new, fully disjoint dataset. Follows
# `rec004m.build_length10_value_path_subcomponent_probe_v1`'s exact
# collision-substitution recipe verbatim; only the split label (hence RNG
# stream, via `_derive_local_seed`) and the (larger) protected set differ. A
# pure function of `(seed, protected_digests)`. Named "Stage D" per the task
# doc even though it is implemented before Stage B/C below, so the frozen
# digest predates every point computed on it.
# =============================================================================


def build_length10_value_path_necessity_probe_v1(
    seed: int, protected_digests: set[str], n: int = REC004N_NEW_PROBE_EXAMPLES
) -> tuple[list[Example], dict[str, Any]]:
    seed_label = f"{REC004N_NEW_PROBE_SPLIT}:{REC004N_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed(seed, 0, seed_label))
    op_obj = get_operation(REC004N_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq = tuple(
                rng.randrange(REC004N_VOCAB_SIZE) for _ in range(REC004N_TARGET_LENGTH)
            )
            params = op_obj.sample_params(rng, seq, REC004N_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004N_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004N_VOCAB_SIZE)
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
                split=REC004N_NEW_PROBE_SPLIT,
                vocab_size=REC004N_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004N_TARGET_OPERATION,)
                ),
            )
        )

    detail = {
        "n": n,
        "target_length": REC004N_TARGET_LENGTH,
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "disclosure": (
            f"{REC004N_NEW_PROBE_SPLIT} is a development-exposed diagnostic set for "
            "B-C005REC-004N, not a sealed or final RG3 query set; once consumed by "
            "this task it is not eligible to serve as an independent RG3 recheck "
            "query without a fresh, disjoint regeneration."
        ),
    }
    return examples, detail


def build_stage_datasets(seed: int) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Byte-identical reuse of REC-004M's own four datasets
    (`build_stage_c_datasets`, which itself chains REC-004K's/REC-004L's own
    two datasets), plus ONE new dataset disjoint from all of: the training
    stream (steps 1-18000), old validation, every REC-004A-D reference/sealed
    split, `clean_selection_validation_v2`'s full 1024-example set,
    `length10_mechanism_probe_v1`, `length10_downstream_interaction_probe_v1`,
    and `length10_value_path_subcomponent_probe_v1`."""
    datasets_4, dataset_details_4 = rec004m.build_stage_c_datasets(seed)

    protected_v1, _counts_v1 = traj_audit.build_protected_digest_registry(seed)
    all_1024, *_rest = rec004j.regenerate_clean_selection_validation_v2(seed)
    protected_v2 = protected_v1 | traj_audit._digest_examples(all_1024)
    protected_v3 = protected_v2 | traj_audit._digest_examples(
        datasets_4[REC004N_MECHANISM_PROBE_DATASET]
    )
    protected_v4 = protected_v3 | traj_audit._digest_examples(
        datasets_4[REC004N_DOWNSTREAM_PROBE_DATASET]
    )
    protected_v5 = protected_v4 | traj_audit._digest_examples(
        datasets_4[REC004N_SUBCOMPONENT_PROBE_DATASET]
    )

    new_probe_examples, new_probe_detail = build_length10_value_path_necessity_probe_v1(
        seed, protected_v5
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
            new_probe_digests & traj_audit._digest_examples(
                datasets_4[REC004N_MECHANISM_PROBE_DATASET]
            )
        ),
        "overlap_with_length10_downstream_interaction_probe_v1": sorted(
            new_probe_digests & traj_audit._digest_examples(
                datasets_4[REC004N_DOWNSTREAM_PROBE_DATASET]
            )
        ),
        "overlap_with_length10_value_path_subcomponent_probe_v1": sorted(
            new_probe_digests & traj_audit._digest_examples(
                datasets_4[REC004N_SUBCOMPONENT_PROBE_DATASET]
            )
        ),
    }
    overlap["disjoint"] = not any(v for v in overlap.values())
    new_probe_detail["disjointness_check"] = overlap
    new_probe_detail["frozen_dataset_digest_sha256"] = rec004l._dataset_digest(new_probe_examples)

    datasets: dict[str, list[Example]] = dict(datasets_4)
    datasets[REC004N_NEW_PROBE_SPLIT] = new_probe_examples
    dataset_details: dict[str, Any] = dict(dataset_details_4)
    dataset_details[REC004N_NEW_PROBE_SPLIT] = new_probe_detail
    return datasets, dataset_details


# =============================================================================
# Stage B -- build the 6 mergeable conditions (R0/F/F_CV/F_CO/F_VO/F_CVO) plus
# EARLY, reusing REC-004M's own row-slice-aware merge function unmodified. It
# already accepts an arbitrary subcomponent tuple, so no new merge logic is
# needed for the leave-one-out combinations.
# =============================================================================


def build_condition_primitives(
    core: Any,
    late_state_dict: dict[str, torch.Tensor],
    early_state_dict: dict[str, torch.Tensor],
    key_groups: dict[str, list[str]],
    early_primitive: Any,
) -> dict[str, Any]:
    primitives: dict[str, Any] = {
        condition_id: rec004m._build_subcomponent_rollback_primitive(
            core, late_state_dict, early_state_dict,
            REC004N_CONDITION_TO_SUBCOMPONENTS[condition_id], key_groups,
            REC004N_FFN_ROLLED_BACK[condition_id],
        )
        for condition_id in REC004N_MERGE_CONDITION_IDS
    }
    primitives["EARLY"] = early_primitive
    return primitives


def run_conditions_on_datasets(
    core: Any,
    condition_primitives: dict[str, Any],
    datasets: dict[str, list[Example]],
    condition_ids: tuple[str, ...],
    dataset_names: tuple[str, ...],
) -> dict[str, Any]:
    """Reuses REC-004L's own `compute_interaction_diagnostic_point`
    unmodified (O1 EM, per-position accuracy/margin, hidden-state diff and
    final-logit diff vs `EARLY`) for exactly the requested
    `(condition_ids, dataset_names)` cross product -- never a combination
    search."""
    reference_primitive = condition_primitives["EARLY"]
    points: dict[str, Any] = {}
    for condition_id in condition_ids:
        primitive = condition_primitives[condition_id]
        for dataset_name in dataset_names:
            examples = datasets[dataset_name]
            with torch.no_grad():
                point = rec004l.compute_interaction_diagnostic_point(
                    core, primitive, reference_primitive, examples
                )
            point.update(
                {
                    "condition_id": condition_id,
                    "subcomponents_rolled_back": list(
                        REC004N_CONDITION_TO_SUBCOMPONENTS.get(condition_id, ())
                    ),
                    "ffn_rolled_back": REC004N_FFN_ROLLED_BACK.get(condition_id, False),
                    "dataset": dataset_name,
                }
            )
            points[f"{condition_id}:{dataset_name}"] = point
    return points


# =============================================================================
# Stage A -- source/parity reconfirmation (task doc Section 3). Recomputes
# R0/F/F_CVO/EARLY on the four pre-existing datasets and cross-checks against
# REC-004M's own saved numbers, plus REC-004M's own Q/K rollback invariance
# check, BEFORE any leave-one-out condition is computed or trusted.
# =============================================================================


def cross_check_against_rec004m(replay_points: dict[str, Any]) -> dict[str, Any]:
    """`F`/`F_CVO`/`EARLY`/`R0` map onto REC-004M's own `F`/`F_ALLV`/
    `EARLY`/`R0` conditions exactly (Q/K is inert under O1, so `F_CVO` and
    `F_ALLV` are identical there) -- their oracle EM must match exactly on
    every dataset REC-004M also used. Informational cross-check; feeds
    `run_source_replay`'s own pass/fail gate."""
    path = REC004M_RUN_DIR / "condition_points_summary.json"
    if not path.is_file():
        return {"status": "REC004M_ARTIFACT_UNAVAILABLE", "mismatches": []}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    rec004n_to_rec004m_condition = {"F": "F", "F_CVO": "F_ALLV", "EARLY": "EARLY", "R0": "R0"}
    mismatches: list[dict[str, Any]] = []
    for condition_id, rec004m_condition_id in rec004n_to_rec004m_condition.items():
        for dataset_name in REC004N_EXISTING_DATASETS:
            recorded_em = recorded.get(f"{rec004m_condition_id}:{dataset_name}", {}).get(
                "oracle_sequence_exact_match"
            )
            recomputed_em = replay_points.get(f"{condition_id}:{dataset_name}", {}).get(
                "oracle_sequence_exact_match"
            )
            if (
                recorded_em is None
                or recomputed_em is None
                or abs(recorded_em - recomputed_em) >= REC004N_SOURCE_REPLAY_EM_TOL
            ):
                mismatches.append(
                    {
                        "condition_id": condition_id,
                        "rec004m_condition_id": rec004m_condition_id,
                        "dataset": dataset_name,
                        "rec004m_recorded_em": recorded_em,
                        "rec004n_recomputed_em": recomputed_em,
                    }
                )
    return {"status": "VERIFIED" if not mismatches else "MISMATCH", "mismatches": mismatches}


def run_source_replay(
    replay_points: dict[str, Any], qk_rollback_check: dict[str, Any]
) -> dict[str, Any]:
    cross_check = cross_check_against_rec004m(replay_points)

    def _all_em(condition_id: str) -> list[float | None]:
        return [
            replay_points.get(f"{condition_id}:{ds}", {}).get("oracle_sequence_exact_match")
            for ds in REC004N_EXISTING_DATASETS
        ]

    f_ems = _all_em("F")
    f_cvo_ems = _all_em("F_CVO")
    early_ems = _all_em("EARLY")

    f_lo, f_hi = REC004N_SOURCE_REPLAY_F_EM_RANGE
    f_range_ok = all(em is not None and f_lo <= em <= f_hi for em in f_ems)
    f_cvo_perfect = all(em == REC004N_SOURCE_REPLAY_PERFECT_EM for em in f_cvo_ems)
    early_perfect = all(em == REC004N_SOURCE_REPLAY_PERFECT_EM for em in early_ems)
    qk_invariant = bool(qk_rollback_check.get("invariant", False))

    status = (
        "VERIFIED"
        if (
            cross_check["status"] == "VERIFIED"
            and f_range_ok and f_cvo_perfect and early_perfect and qk_invariant
        )
        else "SOURCE_REPLAY_MISMATCH"
    )
    return {
        "task_id": REC004N_TASK_ID,
        "cross_check_against_rec004m": cross_check,
        "f_oracle_em_per_dataset": dict(zip(REC004N_EXISTING_DATASETS, f_ems, strict=True)),
        "f_cvo_oracle_em_per_dataset": dict(zip(REC004N_EXISTING_DATASETS, f_cvo_ems, strict=True)),
        "early_oracle_em_per_dataset": dict(zip(REC004N_EXISTING_DATASETS, early_ems, strict=True)),
        "f_em_expected_range": list(REC004N_SOURCE_REPLAY_F_EM_RANGE),
        "f_em_in_expected_range": f_range_ok,
        "f_cvo_em_is_perfect": f_cvo_perfect,
        "early_em_is_perfect": early_perfect,
        "qk_rollback_invariant": qk_invariant,
        "qk_rollback_max_abs_logit_diff": qk_rollback_check.get("max_abs_logit_diff"),
        "status": status,
    }


# =============================================================================
# Stage C -- J0 attention equivalence for F_VO vs R0 (task doc Section 5).
# Compares two DIFFERENT primitives' real, learned attention (never oracle
# substitution) at every layer of the score computation.
# =============================================================================


def compute_j0_attention_equivalence(
    primitive_a: Any,
    primitive_b: Any,
    content_features: torch.Tensor,
    content_lengths: list[int],
    output_lengths: list[int],
) -> dict[str, Any]:
    """`primitive_a`/`primitive_b` are expected to share identical
    `CONTENT_PREP`/`SCORE_PROJECTION_CONTROL` (Q/K)/position-bias/
    `QUERY_RESIDUAL_PATH` parameters (e.g. `F_VO` vs `R0`) -- under that
    condition, `query`/`kv`/the position-bias term/the raw pre-softmax Q/K
    score/the post-softmax `attn_probs` are all expected to be
    mathematically identical, checked here empirically via the REAL
    `primitive.cross_attn` module (`run_intervention_forward(..., "J0")`)
    for `attn_probs`, and `_manual_attention`'s `s_other` return (pure
    Q/K^T score, computed before any bias/mask term) for the raw score."""
    resid_audit._assert_eval_only()
    out_a = resid_audit.run_intervention_forward(
        primitive_a, content_features, content_lengths, output_lengths, "J0"
    )
    out_b = resid_audit.run_intervention_forward(
        primitive_b, content_features, content_lengths, output_lengths, "J0"
    )
    query_a, kv_a, _out_max_a, _lmax_a, _batch_a = resid_audit._prepare_query_kv(
        primitive_a, content_features, content_lengths, output_lengths
    )
    query_b, kv_b, _out_max_b, _lmax_b, _batch_b = resid_audit._prepare_query_kv(
        primitive_b, content_features, content_lengths, output_lengths
    )
    with torch.no_grad():
        _, _, s_other_a = resid_audit._manual_attention(
            primitive_a, query_a, kv_a, out_a["b_scaled"], s_other_scale=1.0
        )
        _, _, s_other_b = resid_audit._manual_attention(
            primitive_b, query_b, kv_b, out_b["b_scaled"], s_other_scale=1.0
        )

    diffs = {
        "max_abs_diff_query": (query_a - query_b).abs().max().item(),
        "max_abs_diff_kv": (kv_a - kv_b).abs().max().item(),
        "max_abs_diff_position_bias": (out_a["b_scaled"] - out_b["b_scaled"]).abs().max().item(),
        "max_abs_diff_raw_qk_score": (s_other_a - s_other_b).abs().max().item(),
        "max_abs_diff_attn_probs": (out_a["attn_probs"] - out_b["attn_probs"]).abs().max().item(),
    }
    max_diff = max(diffs.values())
    return {
        "task_id": REC004N_TASK_ID,
        **diffs,
        "abs_tolerance": REC004N_J0_ATTENTION_ABS_TOL,
        "max_diff_overall": max_diff,
        "attention_score_path_invariant": max_diff <= REC004N_J0_ATTENTION_ABS_TOL,
        "bitwise_exact": max_diff == 0.0,
    }


# =============================================================================
# Decision rule (task doc Section 7).
# =============================================================================


def build_value_path_necessity_decision(condition_points: dict[str, Any]) -> dict[str, Any]:
    pos_key = str(REC004N_POSITION_MAIN_RESIDUAL)

    def _passes(condition_id: str) -> bool:
        for dataset_name in REC004N_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            em = row.get("oracle_sequence_exact_match")
            pos4 = row.get("per_output_position", {}).get(pos_key, {}).get("oracle_accuracy")
            if em is None or pos4 is None:
                return False
            if em < REC004N_ROLLBACK_EM_THRESHOLD or pos4 < REC004N_ROLLBACK_POSITION_ACC_THRESHOLD:
                return False
        return True

    per_condition_passes = {cid: _passes(cid) for cid in REC004N_LEAVE_ONE_OUT_CONDITION_IDS}
    f_cvo_passes = _passes("F_CVO")

    # Priority order fixed by the task doc (Section 7): F_VO first, then any
    # of F_CV/F_CO, then F_CVO-only, then incomplete. Never "pick the
    # highest EM" among multiple passing conditions.
    if per_condition_passes["F_VO"]:
        label = "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"
        tags = ["CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY"]
    elif per_condition_passes["F_CV"] or per_condition_passes["F_CO"]:
        label = "SHARED_CONTENT_PREP_REQUIRED"
        tags = ["FREEZE_REPAIR_NOT_SCORE_SAFE"]
    elif f_cvo_passes:
        label = "ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN"
        tags = []
    else:
        label = "VALUE_PATH_NECESSITY_INCOMPLETE"
        tags = []

    return {
        "task_id": REC004N_TASK_ID,
        "label": label,
        "tags": tags,
        "per_condition_passes": per_condition_passes,
        "f_cvo_passes": f_cvo_passes,
        "oracle_em_threshold": REC004N_ROLLBACK_EM_THRESHOLD,
        "position4_accuracy_threshold": REC004N_ROLLBACK_POSITION_ACC_THRESHOLD,
        "excluded_subcomponent_per_condition": dict(REC004N_LEAVE_ONE_OUT_EXCLUDES),
        "content_prep_shared_with_score_path_caveat": (
            "CONTENT_PREP feeds the same kv tensor SCORE_PROJECTION_CONTROL (the "
            "real, learnable K path) also reads from -- this is why F_VO (which "
            "never rolls CONTENT_PREP back) is the score-safe candidate this task "
            "specifically searches for, and why SHARED_CONTENT_PREP_REQUIRED does "
            "not propose a freeze-based repair even if F_CV or F_CO passes."
        ),
    }


def apply_score_path_preserved_tag(
    decision: dict[str, Any], j0_attention_check: dict[str, Any]
) -> dict[str, Any]:
    if (
        decision["label"] == "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"
        and j0_attention_check.get("attention_score_path_invariant")
    ):
        decision = dict(decision)
        decision["tags"] = [*decision["tags"], "SCORE_PATH_PRESERVED"]
    return decision


# =============================================================================
# Stage 8 -- conditional success-model safety check. Reuses REC-004K's own
# `compute_rollback_diagnostic_point` (J0 AND O1) unmodified, plus this
# task's own Stage C J0 attention-equivalence check applied to each init's
# own trajectory.
# =============================================================================


def run_safety_check(
    core: Any, decision: dict[str, Any], datasets: dict[str, list[Example]]
) -> dict[str, Any]:
    if decision["label"] != "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT":
        return {
            "task_id": REC004N_TASK_ID,
            "status": "NOT_EXECUTED_F_VO_DID_NOT_PASS",
            "decision_label": decision["label"],
            "per_init": {},
        }

    subcomponents = REC004N_CONDITION_TO_SUBCOMPONENTS["F_VO"]
    op = get_operation(REC004N_TARGET_OPERATION)
    n = REC004N_TARGET_LENGTH
    probe_examples = datasets[REC004N_MECHANISM_PROBE_DATASET][:REC004N_INVARIANCE_CHECK_EXAMPLES]
    probe_content_lengths = [n] * len(probe_examples)
    probe_output_lengths = [op.output_length(n)] * len(probe_examples)

    per_init: dict[str, Any] = {}
    for init_id, late_step in REC004N_SAFETY_TARGETS:
        late_primitive, late_info = rec004j._load_and_verify_primitive(core, init_id, late_step)
        early_primitive, early_info = rec004j._load_and_verify_primitive(
            core, init_id, REC004N_EARLY_STEP
        )
        if late_primitive is None or early_primitive is None:
            per_init[init_id] = {
                "status": "SOURCE_ARTIFACT_UNAVAILABLE",
                "late_load_info": late_info, "early_load_info": early_info,
            }
            continue

        keys = list(late_primitive.state_dict().keys())
        key_groups = rec004m.build_value_path_subcomponent_key_groups(keys)
        late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
        early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}
        rolled_back_primitive = rec004m._build_subcomponent_rollback_primitive(
            core, late_sd, early_sd, subcomponents, key_groups, ffn_rolled_back=True
        )

        batch_input = collate_content_only_batch(probe_examples, core.tokens, device=core.device)
        with torch.no_grad():
            h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]
            j0_attention_check = compute_j0_attention_equivalence(
                rolled_back_primitive, late_primitive,
                h_content, probe_content_lengths, probe_output_lengths,
            )

        per_dataset: dict[str, Any] = {}
        max_j0_drop = 0.0
        for dataset_name in REC004N_DATASETS:
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

        degraded = -max_j0_drop >= REC004N_SAFETY_EM_DEGRADATION_THRESHOLD
        per_init[init_id] = {
            "status": "EXECUTED",
            "late_step": late_step,
            "late_load_info_status": late_info["status"],
            "early_load_info_status": early_info["status"],
            "j0_attention_equivalence_check": j0_attention_check,
            "per_dataset": per_dataset,
            "max_j0_em_drop": -max_j0_drop,
            "degradation_threshold": REC004N_SAFETY_EM_DEGRADATION_THRESHOLD,
            "classification": "DEGRADATION_OBSERVED" if degraded else "NO_DEGRADATION_OBSERVED",
        }

    any_degraded = any(
        row.get("classification") == "DEGRADATION_OBSERVED" for row in per_init.values()
    )
    return {
        "task_id": REC004N_TASK_ID,
        "status": "EXECUTED",
        "localized_subcomponents": list(subcomponents),
        "per_init": per_init,
        "any_degradation_observed": any_degraded,
    }


# =============================================================================
# Regression check (descriptive only, never gates the decision).
# =============================================================================


def build_regression_check(condition_points: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for condition_id in REC004N_CONDITION_IDS:
        if condition_id == "R0":
            continue
        for dataset_name in REC004N_DATASETS:
            r0_row = condition_points.get(f"R0:{dataset_name}", {})
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            regressed: list[dict[str, Any]] = []
            for pos in REC004N_REGRESSION_POSITIONS:
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
# next_step_repair_contract.md
# =============================================================================


def build_next_step_repair_contract(
    decision: dict[str, Any], j0_attention_check: dict[str, Any], safety_check: dict[str, Any]
) -> str:
    lines: list[str] = [
        "# B-C005REC-004N: Next Step Repair Contract",
        "",
        "This file is a PROPOSAL record, not an authorization. No training is "
        "started by this task or by this file's existence.",
        "",
        f"## Decision: `{decision['label']}`",
        f"tags: {decision.get('tags', [])}",
        "",
    ]

    proposable = (
        decision["label"] == "FFN_V_OUTPROJ_SUBPATH_SUFFICIENT"
        and j0_attention_check.get("attention_score_path_invariant")
    )
    if not proposable:
        lines += [
            f"`F_VO` did not both pass the Stage 7 floor AND clear Stage C's "
            f"score-safety check (label: `{decision['label']}`, "
            f"attention_score_path_invariant: "
            f"{j0_attention_check.get('attention_score_path_invariant')}). Per the "
            "task's own charter, no two-stage training repair is proposed here.",
            "",
        ]
        if decision["label"] == "SHARED_CONTENT_PREP_REQUIRED":
            lines += [
                "`CONTENT_PREP` is required in the minimal sufficient set. Since it "
                "feeds the same `kv` tensor the real, learnable K path "
                "(`SCORE_PROJECTION_CONTROL`) also reads from, freezing it would "
                "constrain real attention-score learning -- this task does not "
                "propose a freeze-based repair. The next open lead is an "
                "architecture-level question (separating K's and V's input "
                "representations), not designed or authorized here.",
                "",
            ]
        elif decision["label"] == "ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN":
            lines += [
                "All three VALUE_OUTPROJ subcomponents are jointly necessary "
                "alongside FFN_BLOCK; no pairwise subset suffices. `CONTENT_PREP` "
                "is one of the three and is shared with the real K path, so a "
                "whole-value-path freeze is not proposed. No further subset search "
                "is proposed either.",
                "",
            ]
        lines += ["`status: NOT_PROPOSED`", ""]
        return "\n".join(lines) + "\n"

    any_degraded = safety_check.get("any_degradation_observed", False)
    lines += [
        "`F_VO` (`FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ`, never touching "
        "`CONTENT_PREP`) is sufficient for O1 recovery on all five datasets, AND "
        "Stage C confirmed its J0 attention distribution is (within tolerance, "
        f"max diff {j0_attention_check.get('max_diff_overall')}, bitwise_exact="
        f"{j0_attention_check.get('bitwise_exact')}) identical to R0's own.",
        "",
        "## Stage 8 safety check",
        f"- status: {safety_check.get('status')}",
    ]
    if safety_check.get("status") == "EXECUTED":
        for init_id, row in safety_check["per_init"].items():
            j0_check = row.get("j0_attention_equivalence_check", {})
            lines.append(
                f"  - {init_id}@{row.get('late_step')}: {row.get('classification')} "
                f"(max J0 EM drop={row.get('max_j0_em_drop'):.4f}, threshold="
                f"{row.get('degradation_threshold')}; own J0 attention invariant="
                f"{j0_check.get('attention_score_path_invariant')})"
            )
        lines.append(f"- any_degradation_observed: {any_degraded}")
    lines.append("")

    if any_degraded:
        lines += [
            "**Caution**: Stage 8 observed a J0 EM drop at or above the "
            f"pre-registered {REC004N_SAFETY_EM_DEGRADATION_THRESHOLD} threshold "
            "for at least one already-succeeding init when the same rollback was "
            "applied to its own step=6000 state. This does not block the proposal "
            "below (no adoption decision is made here), but it is a disclosed risk "
            "factor any actual B-C005REC-004O task must address.",
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
        "  FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ held at their step=6000 values.",
        "  CONTENT_PREP, SCORE_PROJECTION_CONTROL (Q/K), position bias, "
        "QUERY_RESIDUAL_PATH, POST_ATTN_NORM, and READOUT continue training "
        "normally.",
        "```",
        "",
        "step=6000 is treated as a mechanistic transition boundary, established by "
        "independent diagnostic evidence (I03@6000 was oracle-sufficient per "
        "REC-004K Stage A; oracle sufficiency was lost by step=17500) -- NOT as a "
        "result-selected best checkpoint chosen after seeing this task's own "
        "numbers.",
        "",
        "Running any such training is a SEPARATE task (tentatively "
        "`B-C005REC-004O`) requiring its own explicit user instruction and its "
        "own fixed, pre-registered protocol. This file's existence is not "
        "authorization to implement or train it.",
        "",
        "`status: PROPOSED_NOT_AUTHORIZED`",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Report.
# =============================================================================


def build_report_markdown(
    condition_points: dict[str, Any],
    source_replay: dict[str, Any],
    qk_rollback_check: dict[str, Any],
    j0_attention_check: dict[str, Any],
    decision: dict[str, Any],
    dataset_details: dict[str, Any],
    safety_check: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# {REC004N_TASK_ID}: I03 FFN-Value-Path Leave-One-Out Necessity Audit "
        "& Freezeability Gate",
        "",
        "Zero new optimizer updates. Stage B/decision conditions are forward-only "
        "under O1 (oracle-attention); Stage C's attention-equivalence check is "
        "forward-only under J0 (real, learned attention, never oracle "
        "substitution). Every rollback is an in-memory, evaluation-only merged "
        "state dict, reusing REC-004M's own row-slice-aware merge function.",
        "",
        "## Stage A -- source/parity replay",
        f"- status: **{source_replay['status']}**",
        f"- cross-check vs REC-004M: {source_replay['cross_check_against_rec004m']['status']}",
        f"- F oracle EM in expected range {source_replay['f_em_expected_range']}: "
        f"{source_replay['f_em_in_expected_range']} ({source_replay['f_oracle_em_per_dataset']})",
        f"- F_CVO oracle EM perfect: {source_replay['f_cvo_em_is_perfect']} "
        f"({source_replay['f_cvo_oracle_em_per_dataset']})",
        f"- EARLY oracle EM perfect: {source_replay['early_em_is_perfect']} "
        f"({source_replay['early_oracle_em_per_dataset']})",
        f"- Q/K rollback O1-invariant: {source_replay['qk_rollback_invariant']} "
        f"(max_abs_logit_diff={source_replay['qk_rollback_max_abs_logit_diff']})",
        "",
        "## Datasets",
        "",
    ]
    for dataset_name in REC004N_DATASETS:
        detail = dataset_details.get(dataset_name, {})
        lines.append(f"- `{dataset_name}` (n={detail.get('n')})")
    if REC004N_NEW_PROBE_SPLIT in dataset_details:
        new_detail = dataset_details[REC004N_NEW_PROBE_SPLIT]
        lines += [
            f"- `{REC004N_NEW_PROBE_SPLIT}` disjoint="
            f"{new_detail.get('disjointness_check', {}).get('disjoint')}, "
            f"frozen digest={new_detail.get('frozen_dataset_digest_sha256')}",
        ]

    lines += [
        "",
        "## Stage B -- leave-one-out conditions (I03@17500 base, O1)",
        "",
        "| condition | dataset | n | O1 EM | pos4 acc | pos5 acc |",
        "|---|---|---|---|---|---|",
    ]
    for condition_id in REC004N_CONDITION_IDS:
        for dataset_name in REC004N_DATASETS:
            row = condition_points.get(f"{condition_id}:{dataset_name}", {})
            if not row:
                continue
            pos4 = row.get("per_output_position", {}).get(
                str(REC004N_POSITION_MAIN_RESIDUAL), {}
            ).get("oracle_accuracy")
            pos5 = row.get("per_output_position", {}).get(
                str(REC004N_POSITION_SYMMETRIC_CONTROL), {}
            ).get("oracle_accuracy")
            em = row.get("oracle_sequence_exact_match")
            lines.append(
                f"| {condition_id} | {dataset_name} | {row.get('n')} | "
                f"{em:.4f} | {pos4:.4f} | {pos5:.4f} |"
            )

    lines += [
        "",
        "## Stage C -- J0 attention equivalence (F_VO vs R0)",
        f"- query diff: {j0_attention_check.get('max_abs_diff_query')}",
        f"- kv diff: {j0_attention_check.get('max_abs_diff_kv')}",
        f"- position bias diff: {j0_attention_check.get('max_abs_diff_position_bias')}",
        f"- raw QK score diff: {j0_attention_check.get('max_abs_diff_raw_qk_score')}",
        f"- attn_probs diff: {j0_attention_check.get('max_abs_diff_attn_probs')}",
        f"- tolerance: {j0_attention_check.get('abs_tolerance')}",
        "- attention_score_path_invariant: **"
        f"{j0_attention_check.get('attention_score_path_invariant')}**",
        f"- bitwise_exact: {j0_attention_check.get('bitwise_exact')}",
        "",
        "## Necessity decision",
        f"- label: **{decision['label']}**",
        f"- tags: {decision.get('tags', [])}",
        f"- per-condition passes: {decision.get('per_condition_passes')}",
        f"- F_CVO passes: {decision.get('f_cvo_passes')}",
        "",
        "## Stage 8 safety check",
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


def run_mirror_ffn_value_path_leave_one_out_necessity_audit_task(
    config: MirrorFfnValuePathLeaveOneOutNecessityAuditConfig,
) -> dict[str, Any]:
    _guard_not_frozen("run_mirror_ffn_value_path_leave_one_out_necessity_audit_task")
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != REC004N_SEED:
        raise ValueError(
            f"B-C005REC-004N reads REC-004D/H/K/L/M artifacts pre-registered under "
            f"seed {REC004N_SEED}; got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_hashes_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)

    # Stage D: dataset digests are frozen to disk BEFORE any condition/
    # rollback point below is computed (pre-registration).
    datasets, dataset_details = build_stage_datasets(seed)
    for dataset_name, detail in dataset_details.items():
        _write_json(output_dir / f"{dataset_name}_manifest.json", detail)

    late_primitive, late_load_info = rec004j._load_and_verify_primitive(
        core, REC004N_DECISIVE_INIT, REC004N_LATE_STEP
    )
    early_primitive, early_load_info = rec004j._load_and_verify_primitive(
        core, REC004N_DECISIVE_INIT, REC004N_EARLY_STEP
    )
    checkpoint_load_info = {
        str(REC004N_LATE_STEP): late_load_info,
        str(REC004N_EARLY_STEP): early_load_info,
    }
    _write_json(output_dir / "checkpoint_load_info.json", checkpoint_load_info)

    all_keys = list(late_primitive.state_dict().keys())
    key_groups = rec004m.build_value_path_subcomponent_key_groups(all_keys)
    _write_json(output_dir / "value_path_subcomponent_key_groups.json", key_groups)

    probe_examples_for_audit = datasets[REC004N_MECHANISM_PROBE_DATASET][
        :REC004N_INVARIANCE_CHECK_EXAMPLES
    ]
    op = get_operation(REC004N_TARGET_OPERATION)
    n = REC004N_TARGET_LENGTH
    content_lengths = [n] * len(probe_examples_for_audit)
    output_lengths = [op.output_length(n)] * len(probe_examples_for_audit)
    batch_input = collate_content_only_batch(
        probe_examples_for_audit, core.tokens, device=core.device
    )
    with torch.no_grad():
        h_content = core.model.encode(batch_input)[:, 1 : 1 + n, :]

    late_sd = {k: v.clone() for k, v in late_primitive.state_dict().items()}
    early_sd = {k: v.clone() for k, v in early_primitive.state_dict().items()}

    with torch.no_grad():
        qk_rollback_check = rec004m.verify_qk_rollback_is_o1_invariant(
            core, late_sd, early_sd, key_groups, h_content, content_lengths, output_lengths
        )
    _write_json(output_dir / "qk_rollback_invariance_check.json", qk_rollback_check)

    condition_primitives = build_condition_primitives(
        core, late_sd, early_sd, key_groups, early_primitive
    )

    # Stage A: replay F/F_CVO/EARLY/R0 on the four pre-existing datasets and
    # cross-check against REC-004M's own saved numbers BEFORE trusting any
    # leave-one-out condition.
    with torch.no_grad():
        replay_points = run_conditions_on_datasets(
            core, condition_primitives, datasets,
            ("R0", "F", "F_CVO", "EARLY"), REC004N_EXISTING_DATASETS,
        )
    source_replay = run_source_replay(replay_points, qk_rollback_check)
    _write_json(output_dir / "source_replay_check.json", source_replay)

    condition_points: dict[str, Any] = dict(replay_points)
    j0_attention_check: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    regression_check: dict[str, Any] | None = None
    safety_check: dict[str, Any] = {
        "task_id": REC004N_TASK_ID,
        "status": "NOT_EXECUTED_STAGE_NOT_RUN",
        "per_init": {},
    }
    stage_status = "NOT_EXECUTED_SOURCE_REPLAY_MISMATCH"

    if source_replay["status"] == "VERIFIED":
        # Stage A already computed R0/F/F_CVO/EARLY on the 4 pre-existing
        # datasets; extend those 4 to the new 5th dataset, then compute the
        # 3 leave-one-out conditions on all 5.
        with torch.no_grad():
            condition_points.update(
                run_conditions_on_datasets(
                    core, condition_primitives, datasets,
                    ("R0", "F", "F_CVO", "EARLY"), (REC004N_NEW_PROBE_SPLIT,),
                )
            )
            condition_points.update(
                run_conditions_on_datasets(
                    core, condition_primitives, datasets,
                    REC004N_LEAVE_ONE_OUT_CONDITION_IDS, REC004N_DATASETS,
                )
            )
        _write_jsonl(output_dir / "condition_points.jsonl", condition_points)
        _write_json(output_dir / "condition_points_summary.json", condition_points)

        with torch.no_grad():
            j0_attention_check = compute_j0_attention_equivalence(
                condition_primitives["F_VO"], condition_primitives["R0"],
                h_content, content_lengths, output_lengths,
            )
        _write_json(output_dir / "j0_attention_equivalence_check.json", j0_attention_check)

        decision = build_value_path_necessity_decision(condition_points)
        decision = apply_score_path_preserved_tag(decision, j0_attention_check)
        _write_json(output_dir / "value_path_necessity_decision.json", decision)

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
        "task_id": REC004N_TASK_ID,
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
        "task_id": REC004N_TASK_ID,
        "before": before_hashes, "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    n_predictions = sum(
        row.get("n", 0) for row in condition_points.values() if isinstance(row, dict)
    ) * 2
    if safety_check.get("status") == "EXECUTED":
        n_predictions += sum(
            len(datasets[ds])
            for row in safety_check["per_init"].values()
            for ds in row.get("per_dataset", {})
        ) * 4
    cost_accounting = {
        "task_id": REC004N_TASK_ID,
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
        "task_id": REC004N_TASK_ID,
        "source_task_ids": list(REC004N_SOURCE_TASK_IDS),
        "contract_file": str(REC004N_CONTRACT_FILE),
        "decisive_init": REC004N_DECISIVE_INIT,
        "early_step": REC004N_EARLY_STEP,
        "late_step": REC004N_LATE_STEP,
        "position_main_residual": REC004N_POSITION_MAIN_RESIDUAL,
        "position_symmetric_control": REC004N_POSITION_SYMMETRIC_CONTROL,
        "condition_ids": list(REC004N_CONDITION_IDS),
        "leave_one_out_condition_ids": list(REC004N_LEAVE_ONE_OUT_CONDITION_IDS),
        "datasets": list(REC004N_DATASETS),
        "safety_targets": [{"init_id": i, "step": s} for i, s in REC004N_SAFETY_TARGETS],
        "forbidden": [
            "new optimizer updates", "any I03/I04/I05 checkpoint file mutation",
            "any condition beyond the fixed R0/F/F_CV/F_CO/F_VO/F_CVO/EARLY set",
            "recomputing REC-004M's own F_C/F_V/F_O/F_QK as new conditions",
            "FFN retraining", "an actual VALUE-path freeze training run",
            "feeding pi_n into a model input", "extending position bias", "Core change",
            "candidate adoption", "child assembly", "RG3 recheck",
            "starting B-C005REC-005", "any cross-init weight transplant",
        ],
    }
    _write_json(output_dir / "protocol.json", protocol)

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE" if stage_status == "EXECUTED" else "PARTIAL",
        "source_replay_status": source_replay["status"],
        "qk_rollback_invariance_status": (
            "VERIFIED" if qk_rollback_check.get("invariant") else "FAILED"
        ),
        "stage_status": stage_status,
        "j0_attention_score_path_invariant": (
            j0_attention_check.get("attention_score_path_invariant")
            if j0_attention_check is not None else None
        ),
        "value_path_necessity_decision": decision["label"] if decision is not None else None,
        "value_path_necessity_tags": decision.get("tags") if decision is not None else None,
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

    if decision is not None and j0_attention_check is not None:
        report_md = build_report_markdown(
            condition_points, source_replay, qk_rollback_check, j0_attention_check,
            decision, dataset_details, safety_check,
        )
        contract_md = build_next_step_repair_contract(decision, j0_attention_check, safety_check)
    else:
        report_md = (
            f"# {REC004N_TASK_ID}: I03 FFN-Value-Path Leave-One-Out Necessity Audit "
            f"& Freezeability Gate\n\nSTOPPED: {source_replay['status']}.\n\n"
            "source_replay_check.json:\n```json\n"
            f"{json.dumps(source_replay, indent=2, default=str)}\n```\n"
        )
        contract_md = (
            "# B-C005REC-004N: Next Step Repair Contract\n\n"
            f"Not generated: source replay did not verify "
            f"({source_replay['status']}) -- no leave-one-out condition was "
            "computed or interpreted.\n\n"
            "`status: NOT_PROPOSED`\n"
        )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "next_step_repair_contract.md").write_text(contract_md, encoding="utf-8")
    return result
