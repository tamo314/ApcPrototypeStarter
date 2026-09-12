"""Focused regression and contract tests for Task B-C005REC-004AR:
CD-DPCA Sequence-Distinctness Warm-Start Five-Seed Matched-Baseline Causal Replication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.mirror_cd_dpca_all_init_validation import (
    build_all_initial_states,
)
from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    REC004AK_BANK_MANIFEST_HASH,
    REC004AK_BANK_STATE_CANONICAL_HASH,
    REC004AK_BANK_STATE_RAW_HASH,
    REC004AK_PRIMITIVE_CONFIG_HASH,
    REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
    REC004AK_PRIMITIVE_STATE_RAW_HASH,
    REC004AL_ARCHITECTURE_SIGNATURE,
    REC004AL_PARENT_BUNDLE_ID,
    REC004AL_PARENT_CORE_HASH,
    REC004AL_TARGET_OPERATION,
    run_information_boundary_audit,
)
from apc.evaluation.mirror_cd_dpca_matched_baseline_replication import (
    REC004AR_BASELINE_EXECUTE_INIT_IDS,
    REC004AR_BASELINE_I01_TASK_ID,
    REC004AR_INIT_IDS,
    REC004AR_SOURCE_TASK_ID,
    REC004AR_TASK_ID,
    REC004AR_WARM_START_TASK_ID,
    MirrorCDDPCAMatchedBaselineReplicationConfig,
    _expected_lr,
    _extract_attractor_formation,
    build_replication_protocol,
    compute_matched_pairs_comparison,
)
from apc.utils import model_bundle as mb

ROOT = Path(__file__).resolve().parent.parent
REC004AK_DIR = ROOT / "runs/phase_b_restart/rec004ak/run_001"
REC004AL_DIR = ROOT / "runs/phase_b_restart/rec004al/run_001"
REC004AM_DIR = ROOT / "runs/phase_b_restart/rec004am/run_001"


def test_rec004ar_protocol_and_source_hashes() -> None:
    """Verifies matched-baseline replication protocol enforces parent bundle and hashes."""
    config = MirrorCDDPCAMatchedBaselineReplicationConfig(
        rec004ak_dir=REC004AK_DIR,
        rec004al_dir=REC004AL_DIR,
        rec004am_dir=REC004AM_DIR,
    )
    parent_manifest, _ = ibc._load_parent_manifest()
    initial_states = build_all_initial_states(config)  # type: ignore[arg-type]
    init_hashes = {k: mb.canonical_state_hash(v) for k, v in initial_states.items()}
    protocol = build_replication_protocol(config, parent_manifest, init_hashes)

    assert protocol["task_id"] == REC004AR_TASK_ID
    assert protocol["warm_start_task_id"] == REC004AR_WARM_START_TASK_ID
    assert protocol["baseline_i01_task_id"] == REC004AR_BASELINE_I01_TASK_ID
    assert protocol["source_task_id"] == REC004AR_SOURCE_TASK_ID
    assert protocol["parent_bundle_id"] == REC004AL_PARENT_BUNDLE_ID
    assert protocol["parent_core_canonical_state_hash"] == REC004AL_PARENT_CORE_HASH
    assert protocol["target_operation"] == REC004AL_TARGET_OPERATION
    assert protocol["architecture_signature"] == REC004AL_ARCHITECTURE_SIGNATURE
    assert protocol["init_ids"] == list(REC004AR_INIT_IDS)
    assert protocol["baseline_execute_inits"] == list(REC004AR_BASELINE_EXECUTE_INIT_IDS)
    assert protocol["total_inits"] == 5

    design = protocol["matched_pair_design"]
    assert "REC-004AM" in design["intervention_arm"]
    assert "REC-004AL" in design["baseline_arm"]
    assert len(design["identical_factors"]) >= 8

    hashes = protocol["source_hashes"]
    assert hashes["bank_manifest_sha256"] == REC004AK_BANK_MANIFEST_HASH
    assert hashes["primitive_config_sha256"] == REC004AK_PRIMITIVE_CONFIG_HASH
    assert hashes["bank_state_raw_sha256"] == REC004AK_BANK_STATE_RAW_HASH
    assert hashes["bank_state_canonical_hash"] == REC004AK_BANK_STATE_CANONICAL_HASH
    assert hashes["primitive_state_raw_sha256"] == REC004AK_PRIMITIVE_STATE_RAW_HASH
    assert hashes["primitive_state_canonical_hash"] == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH

    bounds = protocol["boundaries"]
    assert bounds["rec004am_fail_preserved"] is True
    assert bounds["candidate_selected_fixed_null"] is True
    assert bounds["child_bundle_fixed_null"] is True
    assert bounds["rg3_recheck_fixed_not_executed"] is True
    assert bounds["rec005_eligible_fixed_false"] is True


def test_rec004ar_lr_schedule() -> None:
    """Verifies CosineAnnealingLR formula matches preregistered trace at all checkpoint steps."""
    assert abs(_expected_lr(0) - 0.0008) < 1e-9
    assert abs(_expected_lr(500) - 0.000405) < 1e-9
    assert abs(_expected_lr(1000) - 1e-5) < 1e-9
    assert abs(_expected_lr(1500) - 0.000405) < 1e-9
    assert abs(_expected_lr(2000) - 0.0008) < 1e-9
    assert abs(_expected_lr(6000) - 0.0008) < 1e-9


def test_rec004ar_attractor_extraction_helper() -> None:
    """Verifies extraction of competitor onset, true attractor acquisition, and terminal top-1."""
    # Case A: Never learned, locked into key 7 from step 500
    traj_a = [
        {"step": 0, "top1_key_mode": 3},
        {"step": 500, "top1_key_mode": 7},
        {"step": 1000, "top1_key_mode": 7},
        {"step": 6000, "top1_key_mode": 7},
    ]
    att_a = _extract_attractor_formation(traj_a, oracle_key=0)
    assert att_a["first_competitor_onset_step"] == 500
    assert att_a["first_competitor_key"] == 7
    assert att_a["true_attractor_acquired_step"] is None
    assert att_a["terminal_top1_key"] == 7

    # Case B: Acquired true attractor at step 500 and maintained through step 6000
    traj_b = [
        {"step": 0, "top1_key_mode": 3},
        {"step": 500, "top1_key_mode": 0},
        {"step": 1000, "top1_key_mode": 0},
        {"step": 6000, "top1_key_mode": 0},
    ]
    att_b = _extract_attractor_formation(traj_b, oracle_key=0)
    assert att_b["first_competitor_onset_step"] is None
    assert att_b["true_attractor_acquired_step"] == 500
    assert att_b["terminal_top1_key"] == 0

    # Case C: Fluctuated then acquired true attractor at step 1500
    traj_c = [
        {"step": 0, "top1_key_mode": 3},
        {"step": 500, "top1_key_mode": 5},
        {"step": 1000, "top1_key_mode": 0},
        {"step": 1500, "top1_key_mode": 0},
        {"step": 6000, "top1_key_mode": 0},
    ]
    att_c = _extract_attractor_formation(traj_c, oracle_key=0)
    assert att_c["first_competitor_onset_step"] == 500
    assert att_c["first_competitor_key"] == 5
    assert att_c["true_attractor_acquired_step"] == 1000
    assert att_c["terminal_top1_key"] == 0


def test_rec004ar_comparison_and_decision_logic() -> None:
    """Verifies matched pairs comparison statistics and primary decision logic."""
    init_ids = ("I01", "I02", "I03", "I04", "I05")

    # Mock baseline results
    base_results: dict[str, dict[str, Any]] = {}
    warm_results: dict[str, dict[str, Any]] = {}

    for i, i_id in enumerate(init_ids):
        # 4 out of 5 show positive improvement on both EM and worst-pos acc
        b_em = 0.80
        w_em = 0.90 if i < 4 else 0.75  # I01..I04 improved, I05 slightly lower

        b_wp_acc = 0.40
        w_wp_acc = 0.70 if i < 4 else 0.35

        base_results[i_id] = {
            "init_id": i_id,
            "init_seed": 20260912 + i,
            "step0_canonical_hash": f"hash_{i_id}",
            "decisive_sequence_em": b_em,
            "by_length_em": {str(L): {"em": b_em} for L in (6, 7, 8, 9, 10)},
            "worst_position_key": "10:4",
            "worst_position_accuracy": b_wp_acc,
            "position4_top1_key": 7,
            "position4_token_accuracy": b_wp_acc,
            "position4_margin": -5.0,
            "position4_entropy": 1.5,
            "causal_controls": {"causal_gap": b_em},
            "position4_trajectory": [],
        }

        warm_results[i_id] = {
            "init_id": i_id,
            "init_seed": 20260912 + i,
            "step0_canonical_hash": f"hash_{i_id}",
            "decisive_sequence_em": w_em,
            "by_length_em": {str(L): {"em": w_em} for L in (6, 7, 8, 9, 10)},
            "worst_position_key": "10:4",
            "worst_position_accuracy": w_wp_acc,
            "position4_top1_key": 0 if i < 4 else 7,
            "position4_token_accuracy": w_wp_acc,
            "position4_margin": 3.0 if i < 4 else -6.0,
            "position4_entropy": 1.0,
            "causal_controls": {"causal_gap": w_em},
            "position4_trajectory": [],
        }

    comp = compute_matched_pairs_comparison(base_results, warm_results, init_ids)
    primary = comp["primary_decision"]
    assert primary["co_improved_count"] == 4
    assert primary["decision_criterion_met"] is True
    assert primary["decision"] == "WARM_START_CAUSAL_SUPERIORITY_REPLICATED"
    assert comp["summary_statistics"]["overall_sequence_em"]["improved_count"] == 4
    assert comp["summary_statistics"]["worst_position_accuracy"]["improved_count"] == 4

    # Now simulate 2/5 improved -> should trigger I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED
    for i, i_id in enumerate(init_ids):
        if i >= 2:
            warm_results[i_id]["decisive_sequence_em"] = 0.70
            warm_results[i_id]["worst_position_accuracy"] = 0.30

    comp_mixed = compute_matched_pairs_comparison(base_results, warm_results, init_ids)
    assert comp_mixed["primary_decision"]["decision_criterion_met"] is False
    assert comp_mixed["primary_decision"]["decision"] == "I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED"


def test_rec004ar_information_boundary_audit() -> None:
    """Verifies AST information boundary audit passes."""
    audit = run_information_boundary_audit()
    assert audit["status"] == "PASS"
    assert not audit["has_forbidden_tokens"]
    assert audit["forward_parameters_valid"]
    assert not audit["routing_uses_content_features"]
    assert not audit["routing_uses_target_tokens"]
    assert not audit["routing_uses_oracle_attention"]
    assert not audit["routing_uses_permutation_table"]
