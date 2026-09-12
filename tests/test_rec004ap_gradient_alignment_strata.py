"""Unit and regression tests for Task B-C005REC-004AP: Stratified Gradient Alignment Diagnostic."""

import json
from pathlib import Path

import pytest

from apc.evaluation.mirror_cd_dpca_gradient_alignment_strata import (
    REC004AP_SOURCE_TASK_ID,
    REC004AP_TASK_ID,
    CDDPCAGradientAlignmentStrataConfig,
    run_gradient_alignment_strata_task,
)


@pytest.fixture(scope="module")
def strata_run():
    """Load or run REC-004AP stratified gradient alignment run artifacts."""
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs/phase_b_restart/rec004ap/run_001"
    summary_p = run_dir / "summary.json"

    if summary_p.is_file():
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
    else:
        config = CDDPCAGradientAlignmentStrataConfig(
            output_dir=run_dir,
            rec004al_dir=repo_root / "runs/phase_b_restart/rec004al/run_001",
            rec004ak_dir=repo_root / "runs/phase_b_restart/rec004ak/run_001",
        )
        summary = run_gradient_alignment_strata_task(config)

    return run_dir, summary


def test_rec004ap_decision_and_classification(strata_run):
    run_dir, summary = strata_run
    assert summary["task_id"] == REC004AP_TASK_ID
    assert summary["source_task_id"] == REC004AP_SOURCE_TASK_ID
    assert summary["status"] == "PASS"
    assert summary["decision"] == "TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED"
    assert summary["next_learning_pilot_authorized"] is False
    assert summary["candidate_adoption_authorized"] is False
    assert summary["bundle_promotion_authorized"] is False


def test_rec004ap_strata_distribution_and_partition(strata_run):
    run_dir, summary = strata_run
    dist = summary["strata_distribution"]
    assert dist["target_length"] == 10
    assert dist["target_position"] == 4
    assert dist["correct_key"] == 0
    assert dist["competitor_key"] == 7
    assert dist["total_examples"] == 206

    counts = dist["counts"]
    assert counts["A"] + counts["B"] + counts["C"] == 206
    assert counts["A"] == 77
    assert counts["B"] == 102
    assert counts["C"] == 27

    fractions = dist["fractions"]
    assert pytest.approx(fractions["A"], 0.001) == 77 / 206
    assert pytest.approx(fractions["B"], 0.001) == 102 / 206
    assert pytest.approx(fractions["C"], 0.001) == 27 / 206


def test_rec004ap_strata_trajectory_mechanisms(strata_run):
    run_dir, summary = strata_run
    mech = summary["mechanism_analysis"]

    # Stratum C must be 100% negative across all 13 checkpoints
    assert mech["stratum_C_negative_checkpoint_fraction"] == 1.0
    assert mech["mid_training_stratum_C_destructive_fraction"] == 1.0

    # Stratum A must be majority positive
    assert mech["stratum_A_positive_checkpoint_fraction"] > 0.50
    assert mech["mid_training_stratum_A_positive_fraction"] >= 0.60

    # Step 500 onset: Stratum A & B are positive, Stratum C is massively negative, inverting pooled
    assert mech["step500_onset_inversion"] is True

    # Dominance ratio: Stratum C destructive magnitude is > 5x Stratum A
    assert mech["stratum_C_to_A_magnitude_dominance_ratio"] > 5.0
    assert mech["token_aliasing_credit_dilution_supported"] is True

    # Terminal step 6000: Stratum A becomes negative due to late saturation
    assert mech["late_training_step6000_stratum_A_negative"] is True
    assert mech["late_saturation_geometry_supported"] is True


def test_rec004ap_detailed_trajectory_checkpoints(strata_run):
    run_dir, summary = strata_run
    traj_data = json.loads((run_dir / "strata_trajectory_metrics.json").read_text(encoding="utf-8"))
    checkpoints = traj_data["trajectory"]
    assert len(checkpoints) == 13

    # Step 500 check
    step500 = next(c for c in checkpoints if c["step"] == 500)
    strata500 = step500["strata"]
    assert strata500["stratum_A_unique_target"]["parameter_space"]["predicted_margin_change"] > 5.0
    assert strata500["stratum_B_aliased_other"]["parameter_space"]["predicted_margin_change"] > 5.0
    assert strata500["stratum_C_aliased_key7"]["parameter_space"]["predicted_margin_change"] < -20.0

    # Step 6000 check
    step6000 = next(c for c in checkpoints if c["step"] == 6000)
    strata6000 = step6000["strata"]
    assert strata6000["stratum_A_unique_target"]["parameter_space"]["predicted_margin_change"] < 0.0
    assert strata6000["stratum_B_aliased_other"]["parameter_space"]["predicted_margin_change"] > 0.0
    assert strata6000["stratum_C_aliased_key7"]["parameter_space"]["predicted_margin_change"] < -5.0


def test_rec004ap_score_space_and_parameter_space_artifacts(strata_run):
    run_dir, summary = strata_run
    score_p = run_dir / "score_space_gradients.json"
    param_p = run_dir / "parameter_gradient_alignment.json"
    assert score_p.is_file()
    assert param_p.is_file()

    score_data = json.loads(score_p.read_text(encoding="utf-8"))
    param_data = json.loads(param_p.read_text(encoding="utf-8"))
    assert len(score_data) == 13
    assert len(param_data) == 13

    # Check score-space dL/dS vector has length 10
    step0_s = score_data[0]["strata"]["stratum_A_unique_target"]
    assert len(step0_s["dL_dS_vector"]) == 10


def test_rec004ap_controls_strata_comparison(strata_run):
    run_dir, summary = strata_run
    ctrl_p = run_dir / "controls_strata_comparison.json"
    assert ctrl_p.is_file()

    ctrl_data = json.loads(ctrl_p.read_text(encoding="utf-8"))
    assert "L10_pos3_control" in ctrl_data
    assert "L10_pos0_control" in ctrl_data
    assert "L10_pos5_control" in ctrl_data
    assert "L8_pos4_control" in ctrl_data
    assert "L9_pos4_control" in ctrl_data

    # Controls must have stratified counts
    for _cname, cinfo in ctrl_data.items():
        assert "strata_counts" in cinfo
        assert len(cinfo["steps"]) == 13


def test_rec004ap_side_effect_audit(strata_run):
    run_dir, summary = strata_run
    audit = json.loads((run_dir / "side_effect_audit.json").read_text(encoding="utf-8"))

    assert audit["optimizer_updates_performed"] == 0
    assert audit["model_parameters_modified"] is False
    assert audit["source_checkpoints_unmodified"] is True
    assert audit["parent_core_unmodified"] is True
    assert audit["parent_bank_unmodified"] is True
    assert audit["candidate_selected"] is None
    assert audit["child_bundle"] is None
    assert audit["bundle_write"] is False
    assert audit["rg3_recheck"] == "NOT_EXECUTED"
    assert audit["rec005_status"] == "BLOCKED"
    assert audit["g1_status"] == "NOT_CLEARED"
    assert audit["g4_status"] == "NOT_CLEARED"
    assert audit["sealed_evaluation_accessed"] == 0
    assert audit["status"] == "PASS"
