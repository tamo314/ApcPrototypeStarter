"""B-C005REC-004X: I03 CVOF Pre-Transition Trust-Region Plasticity Pilot.

Tests whether bounding cumulative CVOF (CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ,
FFN_BLOCK) displacement from step 7500 within the empirical drift radius observed during
the pre-transition oracle-compatible interval (steps 7000->7500) simultaneously restores
normal representation plasticity (J0 score learning) while fully maintaining downstream
O1 oracle compatibility under unconstrained production J0 forward.
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
from apc.utils import model_bundle as mb

__all__ = [
    "REC004X_TASK_ID",
    "REC004X_SOURCE_TASK_IDS",
    "REC004X_TARGET_OPERATION",
    "REC004X_ARM",
    "REC004X_SEED",
    "REC004X_DECISIVE_INIT",
    "REC004X_CONTROL_INITS",
    "REC004X_START_STEP",
    "REC004X_END_STEP",
    "REC004X_MAX_UPDATES",
    "REC004X_STAGE_A_PARITY_UPDATES",
    "REC004X_ORACLE_EM_THRESHOLD",
    "REC004X_J0_DELTA_FLOOR",
    "REC004X_J0_HARD_FREEZE_DELTA_FLOOR",
    "REC004X_FULL_PROBE_INTERVAL",
    "MirrorCVOFTrustRegionPilotConfig",
    "build_cvof_trust_region_validation_v1",
    "build_cvof_trust_region_length10_v1",
    "prepare_rec004x_datasets",
    "calibrate_pretransition_radii",
    "project_cvof_trust_region",
    "run_stage_a_historical_control_parity",
    "run_cvof_trust_region_pilot_task",
]

# =============================================================================
# Constants
# =============================================================================

REC004X_TASK_ID: Final = "B-C005REC-004X"
REC004X_SOURCE_TASK_IDS: Final[tuple[str, ...]] = (
    "B-C005REC-004T",
    "B-C005REC-004U",
    "B-C005REC-004V",
    "B-C005REC-004W",
)

REC004X_TARGET_OPERATION: Final = "MIRROR_HALVES"
REC004X_ARM: Final = "CVOF_TRUST_REGION_R1"
REC004X_BASE_PRIMITIVE: Final = "P_LENGTH_POSITION_BIAS"
REC004X_SEED: Final = RECOVERY_PILOT_SEED  # 10
REC004X_VOCAB_SIZE: Final = mbe.REC004G_VOCAB_SIZE  # 64
REC004X_SEQUENCE_LENGTH_RANGE: Final = mbe.REC004G_SEQUENCE_LENGTH_RANGE  # (2, 10)

REC004X_DECISIVE_INIT: Final = "I03"
REC004X_CONTROL_INITS: Final[tuple[str, ...]] = ("I04", "I05")

REC004X_CALIBRATION_START_STEP: Final = 7000
REC004X_START_STEP: Final = 7500
REC004X_END_STEP: Final = 8000
REC004X_MAX_UPDATES: Final = 500
REC004X_STAGE_A_PARITY_STEP: Final = 7525
REC004X_STAGE_A_PARITY_UPDATES: Final = 25

REC004X_ORACLE_EM_THRESHOLD: Final = 0.95
REC004X_J0_DELTA_FLOOR: Final = 0.10
REC004X_J0_HARD_FREEZE_DELTA_FLOOR: Final = 0.05
REC004X_FULL_PROBE_INTERVAL: Final = 25
REC004X_SENTINEL_SUBSET_PER_DATASET: Final = 64

# Continuity datasets (4 sets from REC-004T, U, V)
REC004X_CONTINUITY_SPLIT_1: Final = "length10_mechanism_probe_v1"
REC004X_CONTINUITY_SPLIT_2: Final = "dense_trajectory_transition_probe_v1"
REC004X_CONTINUITY_SPLIT_3: Final = "attention_clamp_causal_probe_v1"
REC004X_CONTINUITY_SPLIT_4: Final = "downstream_freeze_causal_probe_v1"
REC004X_CONTINUITY_SPLITS: Final[tuple[str, ...]] = (
    REC004X_CONTINUITY_SPLIT_1,
    REC004X_CONTINUITY_SPLIT_2,
    REC004X_CONTINUITY_SPLIT_3,
    REC004X_CONTINUITY_SPLIT_4,
)

# Fresh validation and confirmation datasets
REC004X_FRESH_NORMAL_VALIDATION: Final = "cvof_trust_region_validation_v1"
REC004X_FRESH_LENGTH10_CONFIRMATION: Final = "cvof_trust_region_length10_v1"
REC004X_FRESH_VALIDATION_EXAMPLES: Final = 1024
REC004X_FRESH_LENGTH10_EXAMPLES: Final = 512

REC004X_POSITION_4: Final = 4
REC004X_POSITION_5: Final = 5
REC004X_CORRECT_KEY_P4: Final = 0
REC004X_CORRECT_KEY_P5: Final = 9

CVOF_GROUPS: Final[tuple[str, ...]] = ("C", "V", "O", "F")


@dataclass(frozen=True)
class MirrorCVOFTrustRegionPilotConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004x/run_001")
    seed: int = REC004X_SEED
    oracle_em_threshold: float = REC004X_ORACLE_EM_THRESHOLD
    j0_delta_floor: float = REC004X_J0_DELTA_FLOOR
    j0_hard_freeze_delta_floor: float = REC004X_J0_HARD_FREEZE_DELTA_FLOOR
    max_replay_window_updates: int = REC004X_MAX_UPDATES
    full_probe_step_interval: int = REC004X_FULL_PROBE_INTERVAL
    sentinel_subset_per_dataset: int = REC004X_SENTINEL_SUBSET_PER_DATASET
    parity_updates: int = REC004X_STAGE_A_PARITY_UPDATES


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


def _derive_local_seed_helper(seed: int, index: int, label: str) -> int:
    h = hashlib.sha256(f"{seed}:{index}:{label}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# =============================================================================
# Fresh Dataset Generation & Locking
# =============================================================================


def build_cvof_trust_region_validation_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004X_FRESH_VALIDATION_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    """Deterministically generates `n` MIRROR_HALVES examples across standard
    length distribution (2..10), guaranteed disjoint from protected digests."""
    seed_label = f"{REC004X_FRESH_NORMAL_VALIDATION}:{REC004X_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed_helper(seed, 0, seed_label))
    op_obj = get_operation(REC004X_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq_len = rng.randint(*REC004X_SEQUENCE_LENGTH_RANGE)
            seq = tuple(rng.randrange(REC004X_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(rng, seq, REC004X_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004X_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004X_VOCAB_SIZE)
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
                split=REC004X_FRESH_NORMAL_VALIDATION,
                vocab_size=REC004X_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004X_TARGET_OPERATION,)
                ),
            )
        )

    audit_detail = {
        "dataset_name": REC004X_FRESH_NORMAL_VALIDATION,
        "n_requested": n,
        "n_generated": len(examples),
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions[:10],
    }
    return examples, audit_detail


def build_cvof_trust_region_length10_v1(
    seed: int,
    protected_digests: set[str],
    n: int = REC004X_FRESH_LENGTH10_EXAMPLES,
) -> tuple[list[Example], dict[str, Any]]:
    """Deterministically generates `n` MIRROR_HALVES examples of fixed length 10,
    guaranteed disjoint from protected digests."""
    seed_label = f"{REC004X_FRESH_LENGTH10_CONFIRMATION}:{REC004X_TARGET_OPERATION}"
    rng = random.Random(_derive_local_seed_helper(seed, 0, seed_label))
    op_obj = get_operation(REC004X_TARGET_OPERATION)
    examples: list[Example] = []
    substitutions: list[dict[str, Any]] = []
    total_draws = 0

    for slot in range(n):
        rejected_digests: list[str] = []
        while True:
            total_draws += 1
            seq_len = 10
            seq = tuple(rng.randrange(REC004X_VOCAB_SIZE) for _ in range(seq_len))
            params = op_obj.sample_params(rng, seq, REC004X_VOCAB_SIZE)
            p_step = ProgramStep(operation=REC004X_TARGET_OPERATION, params=params)
            prog = Program(steps=(p_step,))
            res = run_program(prog, seq, REC004X_VOCAB_SIZE)
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
                split=REC004X_FRESH_LENGTH10_CONFIRMATION,
                vocab_size=REC004X_VOCAB_SIZE,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(
                    label="K", primitive_operations=(REC004X_TARGET_OPERATION,)
                ),
            )
        )

    audit_detail = {
        "dataset_name": REC004X_FRESH_LENGTH10_CONFIRMATION,
        "n_requested": n,
        "n_generated": len(examples),
        "total_candidate_draws": total_draws,
        "substitution_count": len(substitutions),
        "substitutions": substitutions[:10],
    }
    return examples, audit_detail


def prepare_rec004x_datasets(
    seed: int,
) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Builds and locks the complete evaluation datasets:
    - 4 Continuity datasets (REC-004T, U, V)
    - 2 Fresh evaluation datasets (B-C005REC-004X fresh validation and length-10 confirmation)
    Enforces strict disjointness against steps 1..18000, REC-004O..W datasets, and internal splits.
    """
    # 1. Base protected registry (training steps 1..18000 and standard probe datasets)
    protected, protected_counts = rec004t.build_protected_registry_exhaustive(seed)

    # 2. Continuity 1 & 2 from REC-004T
    cont1_exs, _ = rec004t.prepare_probe_datasets(seed)
    continuity_1_exs = cont1_exs[REC004X_CONTINUITY_SPLIT_1]
    cont1_digests = _digest_examples(continuity_1_exs)
    cont1_hash = _dataset_digest(cont1_digests)

    continuity_2_exs = cont1_exs[REC004X_CONTINUITY_SPLIT_2]
    cont2_digests = _digest_examples(continuity_2_exs)
    cont2_hash = _dataset_digest(cont2_digests)

    # 3. Continuity 3 from REC-004U
    continuity_3_exs, cont3_detail = rec004u.build_attention_clamp_causal_probe_v1(
        seed, protected | cont1_digests | cont2_digests
    )
    cont3_digests = _digest_examples(continuity_3_exs)
    cont3_hash = _dataset_digest(cont3_digests)

    # 4. Continuity 4 from REC-004V
    continuity_4_exs, cont4_detail = rec004v.build_downstream_freeze_causal_probe_v1(
        seed, protected | cont1_digests | cont2_digests | cont3_digests
    )
    cont4_digests = _digest_examples(continuity_4_exs)
    cont4_hash = _dataset_digest(cont4_digests)

    # 5. REC-004W datasets (protect against development exposure)
    rec004w_val_exs, _ = rec004w.build_normal_cvof_protection_validation_v1(
        seed, protected | cont1_digests | cont2_digests | cont3_digests | cont4_digests
    )
    rec004w_val_digests = _digest_examples(rec004w_val_exs)

    rec004w_conf_exs, _ = rec004w.build_normal_cvof_protection_length10_v1(
        seed,
        protected
        | cont1_digests
        | cont2_digests
        | cont3_digests
        | cont4_digests
        | rec004w_val_digests,
    )
    rec004w_conf_digests = _digest_examples(rec004w_conf_exs)

    all_prior_protected = (
        protected
        | cont1_digests
        | cont2_digests
        | cont3_digests
        | cont4_digests
        | rec004w_val_digests
        | rec004w_conf_digests
    )
    protected_counts["continuity_1_length10_mechanism_probe_v1"] = len(cont1_digests)
    protected_counts["continuity_2_dense_trajectory_transition_probe_v1"] = len(cont2_digests)
    protected_counts["continuity_3_attention_clamp_causal_probe_v1"] = len(cont3_digests)
    protected_counts["continuity_4_downstream_freeze_causal_probe_v1"] = len(cont4_digests)
    protected_counts["rec004w_normal_cvof_protection_validation_v1"] = len(rec004w_val_digests)
    protected_counts["rec004w_normal_cvof_protection_length10_v1"] = len(rec004w_conf_digests)

    # 6. Fresh normal-distribution validation set (1024 examples, lengths 2..10)
    fresh_val_exs, fresh_val_detail = build_cvof_trust_region_validation_v1(
        seed, all_prior_protected
    )
    fresh_val_digests = _digest_examples(fresh_val_exs)
    fresh_val_hash = _dataset_digest(fresh_val_digests)

    # 7. Fresh length-10 confirmation set (512 examples, length 10)
    fresh_conf_exs, fresh_conf_detail = build_cvof_trust_region_length10_v1(
        seed, all_prior_protected | fresh_val_digests
    )
    fresh_conf_digests = _digest_examples(fresh_conf_exs)
    fresh_conf_hash = _dataset_digest(fresh_conf_digests)

    # Verify disjointness
    assert len(fresh_val_digests.intersection(all_prior_protected)) == 0
    assert len(fresh_conf_digests.intersection(all_prior_protected)) == 0
    assert len(fresh_conf_digests.intersection(fresh_val_digests)) == 0

    datasets = {
        REC004X_CONTINUITY_SPLIT_1: continuity_1_exs,
        REC004X_CONTINUITY_SPLIT_2: continuity_2_exs,
        REC004X_CONTINUITY_SPLIT_3: continuity_3_exs,
        REC004X_CONTINUITY_SPLIT_4: continuity_4_exs,
        REC004X_FRESH_NORMAL_VALIDATION: fresh_val_exs,
        REC004X_FRESH_LENGTH10_CONFIRMATION: fresh_conf_exs,
    }

    manifest = {
        "task_id": REC004X_TASK_ID,
        "seed": seed,
        "continuity_dataset_1": {
            "name": REC004X_CONTINUITY_SPLIT_1,
            "n_examples": len(continuity_1_exs),
            "target_length": 10,
            "dataset_digest": cont1_hash,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
        },
        "continuity_dataset_2": {
            "name": REC004X_CONTINUITY_SPLIT_2,
            "n_examples": len(continuity_2_exs),
            "target_length": 10,
            "dataset_digest": cont2_hash,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
        },
        "continuity_dataset_3": {
            "name": REC004X_CONTINUITY_SPLIT_3,
            "n_examples": len(continuity_3_exs),
            "target_length": 10,
            "dataset_digest": cont3_hash,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
        },
        "continuity_dataset_4": {
            "name": REC004X_CONTINUITY_SPLIT_4,
            "n_examples": len(continuity_4_exs),
            "target_length": 10,
            "dataset_digest": cont4_hash,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
        },
        "fresh_normal_validation": {
            "name": REC004X_FRESH_NORMAL_VALIDATION,
            "n_examples": len(fresh_val_exs),
            "sequence_length_range": list(REC004X_SEQUENCE_LENGTH_RANGE),
            "dataset_digest": fresh_val_hash,
            "total_candidate_draws": fresh_val_detail["total_candidate_draws"],
            "substitution_count": fresh_val_detail["substitution_count"],
            "development_exposed": True,
            "sealed_or_rg3_query": False,
            "disjoint_verified": True,
        },
        "fresh_length10_confirmation": {
            "name": REC004X_FRESH_LENGTH10_CONFIRMATION,
            "n_examples": len(fresh_conf_exs),
            "target_length": 10,
            "dataset_digest": fresh_conf_hash,
            "total_candidate_draws": fresh_conf_detail["total_candidate_draws"],
            "substitution_count": fresh_conf_detail["substitution_count"],
            "development_exposed": True,
            "sealed_or_rg3_query": False,
            "disjoint_verified": True,
        },
        "protected_registry_counts": protected_counts,
        "datasets_locked_before_model_eval": True,
    }

    return datasets, manifest


# =============================================================================
# CVOF Group Definitions & Trust Region Projection
# =============================================================================


def get_cvof_group_tensors(
    model: CrossPositionLengthBiasPrimitive,
) -> dict[str, list[torch.Tensor]]:
    """Returns mapping from group ('C', 'V', 'O', 'F') to list of parameter tensors (or slices).

    Note: For 'V', returns slices of cross_attn.in_proj_weight and cross_attn.in_proj_bias.
    """
    embed_dim = model.d_operator  # 32
    # C = CONTENT_PREP
    c_tensors = [
        model.content_in_proj.weight,
        model.content_in_proj.bias,
        model.content_position_embedding.weight,
    ]
    # V = V_PROJECTION (rows 2*embed_dim : 3*embed_dim)
    v_tensors = [
        model.cross_attn.in_proj_weight[2 * embed_dim : 3 * embed_dim],
    ]
    if model.cross_attn.in_proj_bias is not None:
        v_tensors.append(model.cross_attn.in_proj_bias[2 * embed_dim : 3 * embed_dim])

    # O = ATTN_OUT_PROJ
    o_tensors = [model.cross_attn.out_proj.weight]
    if model.cross_attn.out_proj.bias is not None:
        o_tensors.append(model.cross_attn.out_proj.bias)

    # F = FFN_BLOCK
    f_tensors = list(model.ffn.parameters()) + [model.ffn_norm.weight]
    if model.ffn_norm.bias is not None:
        f_tensors.append(model.ffn_norm.bias)

    return {
        "C": [t for t in c_tensors if t is not None],
        "V": [t for t in v_tensors if t is not None],
        "O": [t for t in o_tensors if t is not None],
        "F": [t for t in f_tensors if t is not None],
    }


def extract_cvof_group_state(
    state_dict: dict[str, torch.Tensor],
    embed_dim: int = 32,
) -> dict[str, list[torch.Tensor]]:
    """Extracts CVOF group tensors from a state_dict for distance/radius calculation."""
    c_tensors = [
        state_dict["content_in_proj.weight"],
        state_dict["content_in_proj.bias"],
        state_dict["content_position_embedding.weight"],
    ]
    v_tensors = [
        state_dict["cross_attn.in_proj_weight"][2 * embed_dim : 3 * embed_dim],
    ]
    if (
        "cross_attn.in_proj_bias" in state_dict
        and state_dict["cross_attn.in_proj_bias"] is not None
    ):
        v_tensors.append(state_dict["cross_attn.in_proj_bias"][2 * embed_dim : 3 * embed_dim])

    o_tensors = [state_dict["cross_attn.out_proj.weight"]]
    if (
        "cross_attn.out_proj.bias" in state_dict
        and state_dict["cross_attn.out_proj.bias"] is not None
    ):
        o_tensors.append(state_dict["cross_attn.out_proj.bias"])

    f_tensors = [
        state_dict["ffn.0.weight"],
        state_dict["ffn.0.bias"],
        state_dict["ffn.2.weight"],
        state_dict["ffn.2.bias"],
        state_dict["ffn_norm.weight"],
    ]
    if "ffn_norm.bias" in state_dict and state_dict["ffn_norm.bias"] is not None:
        f_tensors.append(state_dict["ffn_norm.bias"])

    return {
        "C": [t.detach().cpu() for t in c_tensors],
        "V": [t.detach().cpu() for t in v_tensors],
        "O": [t.detach().cpu() for t in o_tensors],
        "F": [t.detach().cpu() for t in f_tensors],
    }


def compute_group_l2_distance(
    tensors_a: list[torch.Tensor],
    tensors_b: list[torch.Tensor],
) -> float:
    """Computes joint L2 distance across a list of corresponding tensors."""
    assert len(tensors_a) == len(tensors_b)
    sum_sq = 0.0
    for a, b in zip(tensors_a, tensors_b, strict=True):
        diff = (a - b).float()
        sum_sq += float(torch.sum(diff * diff).item())
    return math.sqrt(sum_sq)


def calibrate_pretransition_radii(
    core: Any,
    continuity_datasets: dict[str, list[Example]],
    seed: int,
    oracle_em_threshold: float = REC004X_ORACLE_EM_THRESHOLD,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluates I03 @7000 and @7500 on continuity datasets to verify oracle compatibility
    (O1 EM >= 0.95), then calculates pre-transition displacement radii
    R_g = ||theta_g(7500) - theta_g(7000)||_2.
    """
    ckpt7000_path = rec004t._checkpoint_path(REC004X_DECISIVE_INIT, REC004X_CALIBRATION_START_STEP)
    ckpt7500_path = rec004t._checkpoint_path(REC004X_DECISIVE_INIT, REC004X_START_STEP)

    if not ckpt7000_path.is_file() or not ckpt7500_path.is_file():
        raise FileNotFoundError(
            f"PRETRANSITION_RADIUS_CALIBRATION_INVALID: Missing checkpoint(s) "
            f"7000: {ckpt7000_path.exists()}, 7500: {ckpt7500_path.exists()}"
        )

    s7000 = torch.load(ckpt7000_path, map_location="cpu", weights_only=False)
    s7500 = torch.load(ckpt7500_path, map_location="cpu", weights_only=False)
    p7000 = s7000.get("primitive_state_dict", s7000)
    p7500 = s7500.get("primitive_state_dict", s7500)

    # 1. Verify oracle compatibility on continuity datasets
    calib_audits: dict[str, Any] = {}

    for step_val, state_dict in [(7000, p7000), (7500, p7500)]:
        m = rec004t._new_primitive_from_state(core, state_dict)
        step_metrics: dict[str, Any] = {}
        all_splits_pass = True
        for split_name in REC004X_CONTINUITY_SPLITS:
            exs = continuity_datasets[split_name]
            eval_res = rec004w.evaluate_length10_dataset_metrics(core, m, exs)
            o1_em = eval_res["o1_sequence_em"]
            pos4_acc = eval_res["o1_position4_acc"]
            split_pass = bool(o1_em >= oracle_em_threshold and pos4_acc >= oracle_em_threshold)
            if not split_pass:
                all_splits_pass = False
            step_metrics[split_name] = {
                "o1_sequence_em": o1_em,
                "o1_position4_acc": pos4_acc,
                "pass": split_pass,
            }
        calib_audits[f"step_{step_val}"] = {
            "splits": step_metrics,
            "overall_pass": all_splits_pass,
        }
        if not all_splits_pass:
            raise RuntimeError(
                f"PRETRANSITION_RADIUS_CALIBRATION_INVALID: Checkpoint at step {step_val} "
                f"failed oracle compatibility (O1 EM < {oracle_em_threshold})"
            )

    # 2. Compute radii per group
    groups_7000 = extract_cvof_group_state(p7000)
    groups_7500 = extract_cvof_group_state(p7500)

    radii: dict[str, float] = {}
    for g in CVOF_GROUPS:
        r_val = compute_group_l2_distance(groups_7500[g], groups_7000[g])
        radii[g] = r_val

    calibration_manifest = {
        "task_id": REC004X_TASK_ID,
        "calibration_start_step": REC004X_CALIBRATION_START_STEP,
        "calibration_anchor_step": REC004X_START_STEP,
        "decisive_init": REC004X_DECISIVE_INIT,
        "oracle_em_threshold": oracle_em_threshold,
        "calibration_checks": calib_audits,
        "radii": radii,
        "multiplier": 1.0,
        "calibration_status": "VERIFIED",
    }
    return radii, calibration_manifest


def project_cvof_trust_region(
    model: CrossPositionLengthBiasPrimitive,
    anchor_groups: dict[str, list[torch.Tensor]],
    radii: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Performs radial projection on each CVOF group if its displacement from anchor_7500
    exceeds the calibrated radius R_g.

    Groups:
      C: CONTENT_PREP (content_in_proj.*, content_position_embedding.*)
      V: V_PROJECTION (cross_attn.in_proj_weight[64:96], cross_attn.in_proj_bias[64:96])
      O: ATTN_OUT_PROJ (cross_attn.out_proj.*)
      F: FFN_BLOCK (ffn.*, ffn_norm.*)

    Crucially:
      - Q rows (0:32) and K rows (32:64) are NEVER modified.
      - Optimizer moments are NEVER modified.
    """
    group_tensors = get_cvof_group_tensors(model)
    group_dynamics: dict[str, dict[str, Any]] = {}

    for g in CVOF_GROUPS:
        current_tensors = group_tensors[g]
        anchor_tensors = anchor_groups[g]
        r_g = radii[g]

        # Compute displacement vectors and total displacement norm
        diffs: list[torch.Tensor] = []
        sum_sq = 0.0
        for cur, anch in zip(current_tensors, anchor_tensors, strict=True):
            d = cur.data - anch
            diffs.append(d)
            sum_sq += float(torch.sum(d * d).item())

        raw_disp_norm = math.sqrt(sum_sq)

        if raw_disp_norm > r_g and r_g > 0:
            projection_applied = True
            scale = r_g / raw_disp_norm
            proj_correction_norm = (1.0 - scale) * raw_disp_norm
            # Project back to sphere surface
            for cur, anch, d in zip(current_tensors, anchor_tensors, diffs, strict=True):
                cur.data.copy_(anch + scale * d)
            effective_disp_norm = r_g
        elif r_g == 0:
            # Special case: radius == 0 means strict freeze
            projection_applied = raw_disp_norm > 0
            proj_correction_norm = raw_disp_norm
            for cur, anch in zip(current_tensors, anchor_tensors, strict=True):
                cur.data.copy_(anch)
            effective_disp_norm = 0.0
        else:
            projection_applied = False
            proj_correction_norm = 0.0
            effective_disp_norm = raw_disp_norm

        group_dynamics[g] = {
            "radius": r_g,
            "raw_proposed_displacement": raw_disp_norm,
            "current_displacement_norm": effective_disp_norm,
            "displacement_over_radius": (effective_disp_norm / r_g) if r_g > 0 else 0.0,
            "projection_applied": projection_applied,
            "projection_correction_norm": proj_correction_norm,
        }

    return group_dynamics


# =============================================================================
# Initial Parity & Stage A Historical Control Parity
# =============================================================================


def verify_initial_parity(
    core: Any,
    pilot_model: CrossPositionLengthBiasPrimitive,
    reference_7500: CrossPositionLengthBiasPrimitive,
    sample_examples: list[Example],
    device: torch.device,
) -> dict[str, Any]:
    """Verifies that at step 7500, pilot model and historical reference forward
    are identical across score logits, attention probabilities, final logits,
    and discrete predictions."""
    pilot_model.eval()
    pilot_model.to(device)
    reference_7500.to(device)

    c_lens = [len(ex.input_tokens) for ex in sample_examples]
    o_lens = [10] * len(sample_examples)

    with torch.no_grad():
        batch_input = collate_content_only_batch(sample_examples, core.tokens, device=device)
        h = core.model.encode(batch_input)[:, 1:11, :]

        ref_out = rec004t.evaluate_forward_with_stages(
            reference_7500, h, c_lens, o_lens, oracle_attention=False
        )
        pilot_out = rec004t.evaluate_forward_with_stages(
            pilot_model, h, c_lens, o_lens, oracle_attention=False
        )

        score_diff = float(
            torch.max(torch.abs(pilot_out["score_logits"] - ref_out["score_logits"])).item()
        )
        prob_diff = float(
            torch.max(torch.abs(pilot_out["attn_probs"] - ref_out["attn_probs"])).item()
        )
        final_logits_diff = float(
            torch.max(
                torch.abs(pilot_out["final_token_logits"] - ref_out["final_token_logits"])
            ).item()
        )

        ref_preds = torch.argmax(ref_out["final_token_logits"], dim=-1)
        pilot_preds = torch.argmax(pilot_out["final_token_logits"], dim=-1)
        pred_diff_count = int((ref_preds != pilot_preds).sum().item())

    tolerance = 1e-4
    parity_passed = (
        score_diff < tolerance
        and prob_diff < tolerance
        and final_logits_diff < tolerance
        and pred_diff_count == 0
    )

    parity_audit = {
        "step": REC004X_START_STEP,
        "n_examples_tested": len(sample_examples),
        "score_logits_max_abs_diff": score_diff,
        "attn_probs_max_abs_diff": prob_diff,
        "final_logits_max_abs_diff": final_logits_diff,
        "discrete_prediction_mismatch_count": pred_diff_count,
        "tolerance": tolerance,
        "status": "PASS" if parity_passed else "TRUST_REGION_INITIAL_PARITY_FAILURE",
    }

    if not parity_passed:
        raise RuntimeError(
            f"TRUST_REGION_INITIAL_PARITY_FAILURE: Parity mismatch at step 7500! "
            f"score_diff={score_diff:.2e}, prob_diff={prob_diff:.2e}, "
            f"final_logits_diff={final_logits_diff:.2e}, pred_mismatches={pred_diff_count}"
        )

    return parity_audit


def run_stage_a_historical_control_parity(
    core: Any,
    start_ts: dict[str, Any],
    sample_dataset: list[Example],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Replays 25 updates of standard JOINT training from step 7500
    to 7525 to verify bit-exact parity with REC-004T historical control."""
    print("\n--- Starting Stage A: Historical Control Parity Replay (25 updates) ---")
    start_step = REC004X_START_STEP
    parity_step = REC004X_STAGE_A_PARITY_STEP

    model = mpbr._new_arm_primitive(core, REC004X_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    rec004t._restore_rng_state(start_ts, device)

    model.train()
    loss_at_7525 = 0.0

    for step in range(start_step + 1, parity_step + 1):
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004X_TARGET_OPERATION,
            vocab_size=REC004X_VOCAB_SIZE,
            sequence_length_range=REC004X_SEQUENCE_LENGTH_RANGE,
        )
        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004X_TARGET_OPERATION).output_length(n) for n in content_lengths
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

        if step == parity_step:
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
                if r.get("step") == parity_step:
                    expected_loss_7525 = r.get("loss", expected_loss_7525)
                    break

    loss_diff = abs(loss_at_7525 - expected_loss_7525)
    tolerance = 1e-4
    parity_passed = loss_diff < tolerance

    parity_audit = {
        "task_id": REC004X_TASK_ID,
        "stage": "STAGE_A_HISTORICAL_CONTROL_PARITY",
        "parity_step": parity_step,
        "updates_executed": REC004X_STAGE_A_PARITY_UPDATES,
        "loss_replayed": loss_at_7525,
        "loss_expected_rec004t": expected_loss_7525,
        "loss_abs_diff": loss_diff,
        "tolerance": tolerance,
        "status": "PASS" if parity_passed else "FAIL",
    }

    if not parity_passed:
        raise RuntimeError(
            f"HISTORICAL_CONTROL_REPLAY_MISMATCH: Control parity failed at step {parity_step}! "
            f"loss_diff={loss_diff:.2e}"
        )

    print(f"Stage A Historical Control Parity: PASS | Loss Diff: {loss_diff:.2e}")
    return parity_audit


# =============================================================================
# Main Intervention Runner: CVOF_TRUST_REGION_R1
# =============================================================================


def run_cvof_trust_region_pilot_task(
    config: MirrorCVOFTrustRegionPilotConfig,
) -> dict[str, Any]:
    """Executes Task B-C005REC-004X end to end."""
    _guard_not_frozen("run_cvof_trust_region_pilot_task")
    cache_snapshot_before = _snapshot_forbidden_cache_hashes(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"=== Starting Task {REC004X_TASK_ID}: "
        "I03 CVOF Pre-Transition Trust-Region Plasticity Pilot ==="
    )
    print(f"Target device: {device}, Seed: {seed}, Output: {output_dir}")

    # 1. Verify Source State (@7500)
    start_step = REC004X_START_STEP
    init_id = REC004X_DECISIVE_INIT

    start_ts_path = rec004t._training_state_path(init_id, start_step)
    if not start_ts_path.is_file():
        raise FileNotFoundError(f"Step {start_step} training state not found: {start_ts_path}")

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
        missing = req_keys - set(start_ts.keys())
        raise RuntimeError(f"SOURCE_7500_STATE_MISMATCH: missing keys {missing}")

    if start_ts.get("cumulative_updates") != start_step:
        raise RuntimeError(
            f"SOURCE_7500_STATE_MISMATCH: cumulative_updates is "
            f"{start_ts.get('cumulative_updates')} != {start_step}"
        )

    source_primitive_hash = mb.canonical_state_hash(start_ts["primitive_state_dict"])
    expected_step7500_hash = "7c71a7a43ec2ba766623a70cf4b62e18a8ad685bb1073657ef3fd54d676cb3cb"
    if source_primitive_hash != expected_step7500_hash:
        raise RuntimeError(
            f"SOURCE_7500_STATE_MISMATCH: step 7500 state hash {source_primitive_hash} "
            f"!= expected {expected_step7500_hash}"
        )

    source_manifest = {
        "task_id": REC004X_TASK_ID,
        "source_task_ids": list(REC004X_SOURCE_TASK_IDS),
        "source_step7500_state_path": str(start_ts_path),
        "source_canonical_primitive_state_hash": source_primitive_hash,
        "cumulative_updates": start_ts.get("cumulative_updates"),
        "data_stream_next_step": start_step + 1,
        "optimizer_state_present": True,
        "scheduler_state_present": True,
        "cpu_rng_present": True,
        "cuda_rng_present": True,
        "source_status": "VERIFIED",
    }
    _write_json(output_dir / "source_manifest.json", source_manifest)

    # 2. Load Core and Datasets
    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    datasets, fresh_manifest = prepare_rec004x_datasets(seed=seed)
    _write_json(output_dir / "fresh_validation_manifest.json", fresh_manifest)

    # 3. Trust-Region Radius Calibration
    print("\n>>> Calibrating Pre-Transition Trust-Region Radii (7000 -> 7500) <<<")
    radii, calib_manifest = calibrate_pretransition_radii(
        core,
        datasets,
        seed=seed,
        oracle_em_threshold=config.oracle_em_threshold,
    )
    _write_json(output_dir / "pretransition_radius_calibration.json", calib_manifest)
    print(f"Calibrated Radii: {radii}")

    # 4. Protocol Manifest
    protocol_manifest = {
        "task_id": REC004X_TASK_ID,
        "arm_name": REC004X_ARM,
        "start_step": start_step,
        "end_step": REC004X_END_STEP,
        "optimizer_updates": REC004X_MAX_UPDATES,
        "attention_clamp": False,
        "attention_mechanism": "NORMAL_PRODUCTION_J0_ATTENTION",
        "forbidden_attention_mechanisms": [
            "A_ref",
            "pi_n",
            "oracle_attention",
            "teacher_attention",
            "target_derived_attention",
        ],
        "trust_region_groups": {
            "C_CONTENT_PREP": {
                "keys": [
                    "content_in_proj.weight",
                    "content_in_proj.bias",
                    "content_position_embedding.weight",
                ],
                "radius": radii["C"],
            },
            "V_V_PROJECTION": {
                "keys": [
                    "cross_attn.in_proj_weight[64:96]",
                    "cross_attn.in_proj_bias[64:96]",
                ],
                "radius": radii["V"],
            },
            "O_ATTN_OUT_PROJ": {
                "keys": [
                    "cross_attn.out_proj.weight",
                    "cross_attn.out_proj.bias",
                ],
                "radius": radii["O"],
            },
            "F_FFN_BLOCK": {
                "keys": [
                    "ffn.*",
                    "ffn_norm.weight",
                    "ffn_norm.bias",
                ],
                "radius": radii["F"],
            },
        },
        "unconstrained_trainable_groups": {
            "Q_PROJECTION": "cross_attn.in_proj_weight[0:32], cross_attn.in_proj_bias[0:32]",
            "K_PROJECTION": "cross_attn.in_proj_weight[32:64], cross_attn.in_proj_bias[32:64]",
            "POSITION_BIAS": "position_bias_hidden.*, position_bias_out.*",
            "QUERY_RESIDUAL": "answer_query_embedding.weight",
            "POST_ATTN_NORM": "attn_norm.*",
            "READOUT": "readout.*",
        },
        "recipe": {
            "optimizer": "Projected AdamW",
            "lr": mbe.REC004G_OPERATOR_LR,
            "weight_decay": mbe.REC004G_OPERATOR_WEIGHT_DECAY,
            "grad_clip": mbe.REC004G_OPERATOR_GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "t_max": mbe.REC004G_T_MAX,
            "eta_min": mbe.REC004G_SCHEDULER_ETA_MIN,
            "batch_size": 64,
        },
        "new_candidate_training_updates": 0,
    }
    _write_json(output_dir / "trust_region_protocol.json", protocol_manifest)

    # 5. Lock Step 7500 Models & Verify Initial Parity
    reference_7500 = mpbr._new_arm_primitive(core, REC004X_BASE_PRIMITIVE)
    assert isinstance(reference_7500, CrossPositionLengthBiasPrimitive)
    reference_7500.to(device)
    reference_7500.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )
    reference_7500.eval()
    for param in reference_7500.parameters():
        param.requires_grad_(False)

    model = mpbr._new_arm_primitive(core, REC004X_BASE_PRIMITIVE)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )

    initial_parity_report = verify_initial_parity(
        core,
        model,
        reference_7500,
        datasets[REC004X_CONTINUITY_SPLIT_1][:64],
        device,
    )
    _write_json(output_dir / "initial_parity.json", initial_parity_report)
    print(f"Initial Parity at step 7500: {initial_parity_report['status']}")

    # 6. Optional Stage A Historical Control Parity (<= 25 updates)
    if config.parity_updates > 0:
        stage_a_report = run_stage_a_historical_control_parity(
            core,
            start_ts,
            datasets[REC004X_CONTINUITY_SPLIT_1],
            seed,
            device,
        )
    else:
        stage_a_report = {"status": "SKIPPED"}

    # 7. Setup Model, Optimizer, Scheduler for CVOF_TRUST_REGION_R1
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    rec004t._restore_rng_state(start_ts, device)

    # Snapshot anchor tensors for CVOF trust regions (on device)
    anchor_groups_device: dict[str, list[torch.Tensor]] = {}
    current_cvof = get_cvof_group_tensors(model)
    for g, t_list in current_cvof.items():
        anchor_groups_device[g] = [t.detach().clone() for t in t_list]

    # Parameter snapshot helpers
    window_start_params = rec004t.snapshot_parameter_groups(model)
    prev_step_params = window_start_params

    # Log files
    training_trace_file = output_dir / "per_step_training_trace.jsonl"
    trust_region_trace_file = output_dir / "per_step_trust_region_trace.jsonl"
    functional_metrics_file = output_dir / "per_25step_functional_metrics.jsonl"
    training_trace_f = training_trace_file.open("w", encoding="utf-8")
    trust_region_trace_f = trust_region_trace_file.open("w", encoding="utf-8")
    functional_metrics_f = functional_metrics_file.open("w", encoding="utf-8")

    # Initial 7500 evaluation on continuity datasets
    def _eval_25step_point(step: int) -> dict[str, Any]:
        rec: dict[str, Any] = {"step": step, "continuity_splits": {}}
        for split_name in REC004X_CONTINUITY_SPLITS:
            m = rec004w.evaluate_length10_dataset_metrics(core, model, datasets[split_name])
            rec["continuity_splits"][split_name] = m

        functional_metrics_f.write(json.dumps(rec) + "\n")
        functional_metrics_f.flush()
        return rec

    model.eval()
    _ = _eval_25step_point(start_step)
    model.train()

    print(f"\n>>> Executing {REC004X_ARM} [{start_step + 1} -> {REC004X_END_STEP}] <<<")
    t_train_start = time.time()
    batch_digests_list: list[dict[str, Any]] = []

    # Boundary hit trackers
    group_boundary_hits: dict[str, int] = {g: 0 for g in CVOF_GROUPS}
    group_consecutive_hits: dict[str, int] = {g: 0 for g in CVOF_GROUPS}
    group_consecutive_hits_max: dict[str, int] = {g: 0 for g in CVOF_GROUPS}
    group_total_corrections: dict[str, float] = {g: 0.0 for g in CVOF_GROUPS}

    for step in range(start_step + 1, REC004X_END_STEP + 1):
        # 1. Deterministic data generation
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            REC004X_TARGET_OPERATION,
            vocab_size=REC004X_VOCAB_SIZE,
            sequence_length_range=REC004X_SEQUENCE_LENGTH_RANGE,
        )
        batch_digests = sorted(_digest_examples(examples))
        batch_digests_list.append(
            {"step": step, "batch_digest": _dataset_digest(set(batch_digests))}
        )

        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [
            get_operation(REC004X_TARGET_OPERATION).output_length(n) for n in content_lengths
        ]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)

        # NORMAL PRODUCTION J0 FORWARD (No attention clamping!)
        logits = model(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()

        # Unconstrained AdamW update across all trainable parameters (including CVOF)
        torch.nn.utils.clip_grad_norm_(model.parameters(), mbe.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        lr_after = float(scheduler.get_last_lr()[0])
        running_loss = float(loss.item())

        # Apply Trust-Region Radial Projection on CVOF groups
        dynamics = project_cvof_trust_region(model, anchor_groups_device, radii)

        # Track boundary statistics
        for g in CVOF_GROUPS:
            hit = dynamics[g]["projection_applied"]
            corr = dynamics[g]["projection_correction_norm"]
            group_total_corrections[g] += corr
            if hit:
                group_boundary_hits[g] += 1
                group_consecutive_hits[g] += 1
                if group_consecutive_hits[g] > group_consecutive_hits_max[g]:
                    group_consecutive_hits_max[g] = group_consecutive_hits[g]
            else:
                group_consecutive_hits[g] = 0

        # Compute update norms (raw vs effective post-projection)
        current_params = rec004t.snapshot_parameter_groups(model)
        update_norms: dict[str, float] = {}
        for g_name in rec004t.REC004T_PARAMETER_GROUPS:
            update_norms[g_name] = rec004t.compute_group_l2_norm(
                current_params[g_name], prev_step_params[g_name]
            )
        prev_step_params = current_params

        # Collated norms for trace schema
        q_k_norm = math.sqrt(update_norms["Q"] ** 2 + update_norms["K"] ** 2)
        pos_bias_norm = update_norms["POSITION_BIAS"]
        query_res_norm = update_norms["QUERY_RESIDUAL"]
        norm_readout_norm = math.sqrt(
            update_norms["POST_ATTN_NORM"] ** 2 + update_norms["READOUT"] ** 2
        )

        # Save traces
        trace_rec = {
            "step": step,
            "training_loss": running_loss,
            "lr": lr_used,
            "lr_after": lr_after,
            "update_norm_q_k": q_k_norm,
            "update_norm_position_bias": pos_bias_norm,
            "update_norm_query_residual": query_res_norm,
            "update_norm_norm_readout": norm_readout_norm,
            "projection_active": any(dynamics[g]["projection_applied"] for g in CVOF_GROUPS),
        }
        training_trace_f.write(json.dumps(trace_rec) + "\n")

        tr_rec = {
            "step": step,
            "groups": dynamics,
        }
        trust_region_trace_f.write(json.dumps(tr_rec) + "\n")

        # 25-step evaluation
        if step % config.full_probe_step_interval == 0:
            model.eval()
            _ = _eval_25step_point(step)
            model.train()
            active_groups = [g for g in CVOF_GROUPS if dynamics[g]["projection_applied"]]
            print(
                f"  Step {step}/{REC004X_END_STEP} | Loss: {running_loss:.4f} | "
                f"Q/K norm: {q_k_norm:.4e} | Active Projections: {active_groups}"
            )

    training_trace_f.close()
    trust_region_trace_f.close()
    functional_metrics_f.close()
    t_train_total = time.time() - t_train_start
    print(f"Training completed in {t_train_total:.1f}s.")

    # 8. Training Batch Manifest & Trust Region Boundary Summary
    _write_json(
        output_dir / "training_batch_manifest.json",
        {
            "task_id": REC004X_TASK_ID,
            "total_batches": len(batch_digests_list),
            "step_range": [start_step + 1, REC004X_END_STEP],
            "batch_digests": batch_digests_list,
        },
    )

    total_steps = REC004X_MAX_UPDATES
    group_summaries: dict[str, dict[str, Any]] = {
        g: {
            "radius": radii[g],
            "boundary_hit_count": group_boundary_hits[g],
            "boundary_hit_fraction": group_boundary_hits[g] / total_steps,
            "consecutive_boundary_steps_max": group_consecutive_hits_max[g],
            "total_projection_correction": group_total_corrections[g],
            "mean_projection_correction": group_total_corrections[g] / total_steps,
        }
        for g in CVOF_GROUPS
    }
    cvof_boundary_summary = {
        "task_id": REC004X_TASK_ID,
        "total_updates": total_steps,
        "group_summaries": group_summaries,
        "overall_trust_region_active": any(group_boundary_hits[g] > 0 for g in CVOF_GROUPS),
        "all_groups_effectively_hard": all(
            (group_boundary_hits[g] / total_steps) >= 0.95 for g in CVOF_GROUPS
        ),
    }
    _write_json(output_dir / "cvof_boundary_summary.json", cvof_boundary_summary)

    # 9. Step 8000 Endpoint Evaluation
    print("\n>>> Running Step 8000 Endpoint Evaluation <<<")
    model.eval()

    endpoint_continuity: dict[str, Any] = {}
    for split_name in REC004X_CONTINUITY_SPLITS:
        endpoint_continuity[split_name] = rec004w.evaluate_length10_dataset_metrics(
            core, model, datasets[split_name]
        )

    # Fresh length 10 confirmation
    endpoint_fresh_length10 = rec004w.evaluate_length10_dataset_metrics(
        core, model, datasets[REC004X_FRESH_LENGTH10_CONFIRMATION]
    )

    # Fresh normal validation (lengths 2..10)
    endpoint_fresh_val = rec004w.evaluate_variable_length_dataset(
        core, model, datasets[REC004X_FRESH_NORMAL_VALIDATION]
    )

    endpoint_metrics = {
        "task_id": REC004X_TASK_ID,
        "step": REC004X_END_STEP,
        "arm": REC004X_ARM,
        "continuity_splits": endpoint_continuity,
        "fresh_length10_confirmation": endpoint_fresh_length10,
        "fresh_normal_validation": endpoint_fresh_val,
    }
    _write_json(output_dir / "endpoint_metrics.json", endpoint_metrics)

    # 10. Controls Read-Only Evaluation: HISTORICAL_JOINT @8000 and HARD_CVOF_FREEZE @8000
    print("\n>>> Evaluating Comparators on Fresh Datasets <<<")
    # Comparator 1: HISTORICAL_JOINT @8000
    hist_step8000_ckpt = rec004t._checkpoint_path("I03", REC004X_END_STEP)
    hist_state = torch.load(hist_step8000_ckpt, map_location="cpu", weights_only=False)
    hist_model = rec004t._new_primitive_from_state(
        core, hist_state.get("primitive_state_dict", hist_state)
    )
    hist_fresh_val = rec004w.evaluate_variable_length_dataset(
        core, hist_model, datasets[REC004X_FRESH_NORMAL_VALIDATION]
    )
    hist_fresh_conf = rec004w.evaluate_length10_dataset_metrics(
        core, hist_model, datasets[REC004X_FRESH_LENGTH10_CONFIRMATION]
    )

    # Comparator 2: HARD_CVOF_FREEZE @8000
    # Deterministic replay of REC-004W from step 7500
    print("Replaying HARD_CVOF_FREEZE (REC-004W) to obtain step 8000 comparator...")
    freeze_model = mpbr._new_arm_primitive(core, REC004X_BASE_PRIMITIVE)
    assert isinstance(freeze_model, CrossPositionLengthBiasPrimitive)
    freeze_model.to(device)
    freeze_model.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
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
    for s_step in range(start_step + 1, REC004X_END_STEP + 1):
        exs = ibc._generate_step_training_examples(
            seed,
            s_step,
            REC004X_TARGET_OPERATION,
            vocab_size=REC004X_VOCAB_SIZE,
            sequence_length_range=REC004X_SEQUENCE_LENGTH_RANGE,
        )
        c_lens = [len(ex.input_tokens) for ex in exs]
        o_lens = [get_operation(REC004X_TARGET_OPERATION).output_length(n) for n in c_lens]
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

    freeze_model.eval()
    freeze_fresh_val = rec004w.evaluate_variable_length_dataset(
        core, freeze_model, datasets[REC004X_FRESH_NORMAL_VALIDATION]
    )
    freeze_fresh_conf = rec004w.evaluate_length10_dataset_metrics(
        core, freeze_model, datasets[REC004X_FRESH_LENGTH10_CONFIRMATION]
    )

    # 11. Decision Audits & Gates (Section 14, 15, 16, 17, 18)
    # Gate 1: Compatibility Gate
    # (O1 sequence EM >= 0.95 and pos4 acc >= 0.95 across all 5 length-10 sets)
    compatibility_splits = {
        **endpoint_continuity,
        "fresh_length10_confirmation": endpoint_fresh_length10,
    }
    comp_gate_details: dict[str, Any] = {}
    comp_gate_pass = True
    for s_name, s_m in compatibility_splits.items():
        o1_em = s_m["o1_sequence_em"]
        pos4_acc = s_m["o1_position4_acc"]
        s_pass = bool(
            o1_em >= config.oracle_em_threshold and pos4_acc >= config.oracle_em_threshold
        )
        if not s_pass:
            comp_gate_pass = False
        comp_gate_details[s_name] = {
            "o1_sequence_em": o1_em,
            "o1_position4_acc": pos4_acc,
            "threshold": config.oracle_em_threshold,
            "pass": s_pass,
        }

    o1_compatibility_audit = {
        "task_id": REC004X_TASK_ID,
        "compatibility_gate_pass": comp_gate_pass,
        "threshold": config.oracle_em_threshold,
        "splits": comp_gate_details,
    }
    _write_json(output_dir / "o1_compatibility_audit.json", o1_compatibility_audit)

    # Gate 2: Plasticity Gate
    # fresh normal val: trust-region J0 EM >= hist + 0.10 AND trust-region J0 EM >= freeze + 0.05
    tr_val_j0 = endpoint_fresh_val["overall_j0_sequence_em"]
    hist_val_j0 = hist_fresh_val["overall_j0_sequence_em"]
    freeze_val_j0 = freeze_fresh_val["overall_j0_sequence_em"]
    val_delta_hist = tr_val_j0 - hist_val_j0
    val_delta_freeze = tr_val_j0 - freeze_val_j0
    val_plasticity_pass = bool(
        val_delta_hist >= config.j0_delta_floor
        and val_delta_freeze >= config.j0_hard_freeze_delta_floor
    )

    # fresh length10 conf: trust-region J0 EM >= hist + 0.10 AND trust-region J0 EM >= freeze + 0.05
    tr_conf_j0 = endpoint_fresh_length10["j0_sequence_em"]
    hist_conf_j0 = hist_fresh_conf["j0_sequence_em"]
    freeze_conf_j0 = freeze_fresh_conf["j0_sequence_em"]
    conf_delta_hist = tr_conf_j0 - hist_conf_j0
    conf_delta_freeze = tr_conf_j0 - freeze_conf_j0
    conf_plasticity_pass = bool(
        conf_delta_hist >= config.j0_delta_floor
        and conf_delta_freeze >= config.j0_hard_freeze_delta_floor
    )

    plasticity_gate_pass = bool(val_plasticity_pass and conf_plasticity_pass)
    plasticity_audit = {
        "task_id": REC004X_TASK_ID,
        "plasticity_gate_pass": plasticity_gate_pass,
        "fresh_normal_validation": {
            "trust_region_j0_em": tr_val_j0,
            "historical_j0_em": hist_val_j0,
            "hard_freeze_j0_em": freeze_val_j0,
            "delta_vs_historical": val_delta_hist,
            "delta_vs_hard_freeze": val_delta_freeze,
            "pass": val_plasticity_pass,
        },
        "fresh_length10_confirmation": {
            "trust_region_j0_em": tr_conf_j0,
            "historical_j0_em": hist_conf_j0,
            "hard_freeze_j0_em": freeze_conf_j0,
            "delta_vs_historical": conf_delta_hist,
            "delta_vs_hard_freeze": conf_delta_freeze,
            "pass": conf_plasticity_pass,
        },
        "floors": {
            "historical_delta_floor": config.j0_delta_floor,
            "hard_freeze_delta_floor": config.j0_hard_freeze_delta_floor,
        },
    }
    _write_json(output_dir / "plasticity_audit.json", plasticity_audit)

    # Main Decision Label
    if comp_gate_pass and plasticity_gate_pass:
        result_label = "CVOF_TRUST_REGION_RESTORES_STABILITY_PLASTICITY"
    elif comp_gate_pass and not plasticity_gate_pass:
        result_label = "CVOF_TRUST_REGION_PRESERVES_STABILITY_BUT_PLASTICITY_INSUFFICIENT"
    elif not comp_gate_pass and plasticity_gate_pass:
        result_label = "CVOF_TRUST_REGION_ALLOWS_LEARNING_BUT_NOT_STABILITY"
    else:
        result_label = "PRETRANSITION_RADIUS_TRUST_REGION_NOT_SUPPORTED"

    # Strong functional floor check
    tr_len10_val_j0 = endpoint_fresh_val["length_10_j0_sequence_em"]
    cleared_functional_floor = bool(
        tr_val_j0 >= 0.95 and tr_len10_val_j0 >= 0.95 and tr_conf_j0 >= 0.95 and comp_gate_pass
    )

    # Ablation tags
    ablation_tags: list[str] = []
    if not cvof_boundary_summary["overall_trust_region_active"]:
        ablation_tags.append("TRUST_REGION_INACTIVE")
    if cvof_boundary_summary["all_groups_effectively_hard"]:
        ablation_tags.append("TRUST_REGION_EFFECTIVELY_HARD")

    result_decision = {
        "task_id": REC004X_TASK_ID,
        "compatibility_gate_pass": comp_gate_pass,
        "plasticity_gate_pass": plasticity_gate_pass,
        "functional_floor_cleared": cleared_functional_floor,
        "result_label": result_label,
        "ablation_tags": ablation_tags,
        "interpretation": (
            f"Compatibility: {comp_gate_pass} (all O1 >= 0.95). "
            f"Plasticity: val_delta_hist={val_delta_hist:+.4f}, "
            f"conf_delta_hist={conf_delta_hist:+.4f}, "
            f"val_delta_freeze={val_delta_freeze:+.4f}, "
            f"conf_delta_freeze={conf_delta_freeze:+.4f}."
        ),
    }
    _write_json(output_dir / "result_decision.json", result_decision)

    # 12. Comparisons
    historical_comparison = {
        "task_id": REC004X_TASK_ID,
        "comparator": "HISTORICAL_JOINT",
        "checkpoint": str(hist_step8000_ckpt),
        "fresh_normal_validation": {
            "trust_region_overall_j0": tr_val_j0,
            "historical_overall_j0": hist_val_j0,
            "delta": val_delta_hist,
            "trust_region_length10_j0": tr_len10_val_j0,
            "historical_length10_j0": hist_fresh_val["length_10_j0_sequence_em"],
            "delta_length10": tr_len10_val_j0 - hist_fresh_val["length_10_j0_sequence_em"],
        },
        "fresh_length10_confirmation": {
            "trust_region_j0": tr_conf_j0,
            "historical_j0": hist_conf_j0,
            "delta": conf_delta_hist,
            "trust_region_o1": endpoint_fresh_length10["o1_sequence_em"],
            "historical_o1": hist_fresh_conf["o1_sequence_em"],
            "trust_region_pos4_acc": endpoint_fresh_length10["j0_position4_acc"],
            "historical_pos4_acc": hist_fresh_conf["j0_position4_acc"],
        },
    }
    _write_json(output_dir / "historical_comparison.json", historical_comparison)

    hard_freeze_comparison = {
        "task_id": REC004X_TASK_ID,
        "comparator": "HARD_CVOF_FREEZE",
        "fresh_normal_validation": {
            "trust_region_overall_j0": tr_val_j0,
            "hard_freeze_overall_j0": freeze_val_j0,
            "delta": val_delta_freeze,
            "trust_region_length10_j0": tr_len10_val_j0,
            "hard_freeze_length10_j0": freeze_fresh_val["length_10_j0_sequence_em"],
            "delta_length10": tr_len10_val_j0 - freeze_fresh_val["length_10_j0_sequence_em"],
        },
        "fresh_length10_confirmation": {
            "trust_region_j0": tr_conf_j0,
            "hard_freeze_j0": freeze_conf_j0,
            "delta": conf_delta_freeze,
            "trust_region_o1": endpoint_fresh_length10["o1_sequence_em"],
            "hard_freeze_o1": freeze_fresh_conf["o1_sequence_em"],
            "trust_region_pos4_acc": endpoint_fresh_length10["j0_position4_acc"],
            "hard_freeze_pos4_acc": freeze_fresh_conf["j0_position4_acc"],
        },
    }
    _write_json(output_dir / "hard_freeze_comparison.json", hard_freeze_comparison)

    # 13. Audit & Accounting Manifests
    freeze_audit = {
        "task_id": REC004X_TASK_ID,
        "mechanism": "CVOF_TRUST_REGION_RADIAL_PROJECTION",
        "hard_freeze": False,
        "radii": radii,
        "boundary_summaries": cvof_boundary_summary["group_summaries"],
        "unconstrained_parameters_verified": [
            "cross_attn.in_proj_weight[0:64] (Q and K)",
            "cross_attn.in_proj_bias[0:64] (Q and K)",
            "position_bias_hidden.*",
            "position_bias_out.*",
            "answer_query_embedding.*",
            "attn_norm.*",
            "readout.*",
        ],
    }
    _write_json(output_dir / "freeze_audit.json", freeze_audit)

    cache_snapshot_after = _snapshot_forbidden_cache_hashes(config.seed)
    assert cache_snapshot_before == cache_snapshot_after

    side_effect_audit = {
        "task_id": REC004X_TASK_ID,
        "core_modified": False,
        "router_modified": False,
        "unrelated_primitives_modified": False,
        "forbidden_cache_modified": False,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    cost_accounting = {
        "task_id": REC004X_TASK_ID,
        "counterfactual_intervention_updates": REC004X_MAX_UPDATES,
        "parity_updates": config.parity_updates,
        "total_optimizer_updates_executed": REC004X_MAX_UPDATES + config.parity_updates,
        "new_candidate_training_updates": 0,
        "elapsed_training_seconds": t_train_total,
    }
    _write_json(output_dir / "cost_accounting.json", cost_accounting)

    # 14. Next Repair Contract
    if result_label == "CVOF_TRUST_REGION_RESTORES_STABILITY_PLASTICITY":
        next_step_proposal = (
            "Propose B-C005REC-004Y: Functional Trigger / Anchor-Free CVOF Consolidation Pilot."
        )
    elif result_label == "CVOF_TRUST_REGION_PRESERVES_STABILITY_BUT_PLASTICITY_INSUFFICIENT":
        next_step_proposal = (
            "Analyze group boundary hits to determine plasticity bottleneck among C, V, O, F. "
            "Do not perform arbitrary radius sweep."
        )
    else:
        next_step_proposal = "Do not automatically continue CVOF trust-region approach."

    next_repair_contract = f"""# Next Repair Contract: After {REC004X_TASK_ID}

## Status
- **Result Label**: `{result_label}`
- **Ablation Tags**: `{ablation_tags}`
- **Compatibility Gate**: `{"PASS" if comp_gate_pass else "FAIL"}`
- **Plasticity Gate**: `{"PASS" if plasticity_gate_pass else "FAIL"}`

## Next Step Direction
{next_step_proposal}
"""
    (output_dir / "next_repair_contract.md").write_text(next_repair_contract, encoding="utf-8")

    # 15. Summary & Report
    summary = {
        "task_id": REC004X_TASK_ID,
        "implementation_status": "COMPLETE",
        "radius_calibration": calib_manifest["calibration_status"],
        "stage_a_parity": stage_a_report.get("status"),
        "trust_region_active": cvof_boundary_summary["overall_trust_region_active"],
        "intervention_optimizer_updates": REC004X_MAX_UPDATES,
        "new_candidate_training_updates": 0,
        "result_label": result_label,
        "ablation_tags": ablation_tags,
        "selected_init": None,
        "selected_step": None,
        "selected_radius": None,
        "selected_repair": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
        "metrics": {
            "val_overall_j0": tr_val_j0,
            "conf_length10_j0": tr_conf_j0,
            "val_delta_vs_historical": val_delta_hist,
            "conf_delta_vs_historical": conf_delta_hist,
            "val_delta_vs_hard_freeze": val_delta_freeze,
            "conf_delta_vs_hard_freeze": conf_delta_freeze,
            "all_length10_o1_pass": comp_gate_pass,
        },
    }
    _write_json(output_dir / "summary.json", summary)

    # Generate Markdown Report
    c_sum = group_summaries
    hist_len10 = hist_fresh_val["length_10_j0_sequence_em"]
    freeze_len10 = freeze_fresh_val["length_10_j0_sequence_em"]
    v_d_len10_hist = tr_len10_val_j0 - hist_len10
    v_d_len10_frz = tr_len10_val_j0 - freeze_len10

    c1_em = endpoint_continuity[REC004X_CONTINUITY_SPLIT_1]["o1_sequence_em"]
    c1_p4 = endpoint_continuity[REC004X_CONTINUITY_SPLIT_1]["o1_position4_acc"]
    c2_em = endpoint_continuity[REC004X_CONTINUITY_SPLIT_2]["o1_sequence_em"]
    c2_p4 = endpoint_continuity[REC004X_CONTINUITY_SPLIT_2]["o1_position4_acc"]
    c3_em = endpoint_continuity[REC004X_CONTINUITY_SPLIT_3]["o1_sequence_em"]
    c3_p4 = endpoint_continuity[REC004X_CONTINUITY_SPLIT_3]["o1_position4_acc"]
    c4_em = endpoint_continuity[REC004X_CONTINUITY_SPLIT_4]["o1_sequence_em"]
    c4_p4 = endpoint_continuity[REC004X_CONTINUITY_SPLIT_4]["o1_position4_acc"]
    f_em = endpoint_fresh_length10["o1_sequence_em"]
    f_p4 = endpoint_fresh_length10["o1_position4_acc"]

    def _sp(em: float, p4: float) -> str:
        return (
            "PASS"
            if em >= config.oracle_em_threshold and p4 >= config.oracle_em_threshold
            else "FAIL"
        )

    report_lines = [
        f"# Report: Task {REC004X_TASK_ID} -- I03 CVOF Trust-Region Pilot",
        "",
        "## 1. Summary",
        f"- **Task ID**: `{REC004X_TASK_ID}`",
        f"- **Result Decision**: `{result_label}`",
        f"- **Ablation Tags**: `{ablation_tags}`",
        f"- **Compatibility Gate**: `{'PASS' if comp_gate_pass else 'FAIL'}`",
        f"- **Plasticity Gate**: `{'PASS' if plasticity_gate_pass else 'FAIL'}`",
        "",
        "## 2. Three-Way Comparator Table (Fresh Evaluation Datasets)",
        "",
        "| Metric | Hist @8000 | Freeze @8000 | Trust-R1 | Delta Hist | Delta Frz |",
        "|---|---|---|---|---|---|",
        f"| **Val Overall J0 EM** | {hist_val_j0:.4f} | {freeze_val_j0:.4f} | "
        f"**{tr_val_j0:.4f}** | {val_delta_hist:+.4f} | {val_delta_freeze:+.4f} |",
        f"| **Val Length-10 J0** | {hist_len10:.4f} | {freeze_len10:.4f} | "
        f"**{tr_len10_val_j0:.4f}** | {v_d_len10_hist:+.4f} | {v_d_len10_frz:+.4f} |",
        f"| **Conf Length-10 J0** | {hist_conf_j0:.4f} | {freeze_conf_j0:.4f} | "
        f"**{tr_conf_j0:.4f}** | {conf_delta_hist:+.4f} | {conf_delta_freeze:+.4f} |",
        f"| **Conf Length-10 O1** | {hist_fresh_conf['o1_sequence_em']:.4f} | "
        f"{freeze_fresh_conf['o1_sequence_em']:.4f} | **{f_em:.4f}** | -- | -- |",
        f"| **Conf Pos-4 Acc** | {hist_fresh_conf['j0_position4_acc']:.4f} | "
        f"{freeze_fresh_conf['j0_position4_acc']:.4f} | **{f_p4:.4f}** | -- | -- |",
        "",
        "## 3. Trust-Region Boundary Hit Dynamics",
        "",
        "| Group | Radius $R_g$ | Hit Fraction | Max Consec | Mean Correction |",
        "|---|---|---|---|---|",
        f"| **C** | {radii['C']:.4f} | {c_sum['C']['boundary_hit_fraction']:.2%} | "
        f"{c_sum['C']['consecutive_boundary_steps_max']} | "
        f"{c_sum['C']['mean_projection_correction']:.4e} |",
        f"| **V** | {radii['V']:.4f} | {c_sum['V']['boundary_hit_fraction']:.2%} | "
        f"{c_sum['V']['consecutive_boundary_steps_max']} | "
        f"{c_sum['V']['mean_projection_correction']:.4e} |",
        f"| **O** | {radii['O']:.4f} | {c_sum['O']['boundary_hit_fraction']:.2%} | "
        f"{c_sum['O']['consecutive_boundary_steps_max']} | "
        f"{c_sum['O']['mean_projection_correction']:.4e} |",
        f"| **F** | {radii['F']:.4f} | {c_sum['F']['boundary_hit_fraction']:.2%} | "
        f"{c_sum['F']['consecutive_boundary_steps_max']} | "
        f"{c_sum['F']['mean_projection_correction']:.4e} |",
        "",
        "## 4. Downstream O1 Compatibility Across All Length-10 Splits",
        "",
        "| Split Name | O1 EM | O1 Pos-4 Acc | Pass (>= 0.95)? |",
        "|---|---|---|---|",
        f"| Continuity 1 (`{REC004X_CONTINUITY_SPLIT_1}`) | "
        f"{c1_em:.4f} | {c1_p4:.4f} | {_sp(c1_em, c1_p4)} |",
        f"| Continuity 2 (`{REC004X_CONTINUITY_SPLIT_2}`) | "
        f"{c2_em:.4f} | {c2_p4:.4f} | {_sp(c2_em, c2_p4)} |",
        f"| Continuity 3 (`{REC004X_CONTINUITY_SPLIT_3}`) | "
        f"{c3_em:.4f} | {c3_p4:.4f} | {_sp(c3_em, c3_p4)} |",
        f"| Continuity 4 (`{REC004X_CONTINUITY_SPLIT_4}`) | "
        f"{c4_em:.4f} | {c4_p4:.4f} | {_sp(c4_em, c4_p4)} |",
        f"| Fresh L10 (`{REC004X_FRESH_LENGTH10_CONFIRMATION}`) | "
        f"{f_em:.4f} | {f_p4:.4f} | {_sp(f_em, f_p4)} |",
        "",
    ]
    report_md = "\n".join(report_lines)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"\nTask {REC004X_TASK_ID} complete. Decision: {result_label}")
    return summary
