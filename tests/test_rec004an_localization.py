"""Unit and regression tests for Task B-C005REC-004AN: CD-DPCA Failure Localization."""

import json
from pathlib import Path

import pytest

from apc.evaluation.mirror_cd_dpca_failure_localization import (
    REC004AN_SOURCE_TASK_ID,
    REC004AN_TASK_ID,
    CDDPCALocalizationConfig,
    run_failure_localization_task,
)


@pytest.fixture(scope="module")
def localization_run():
    """Load or run REC-004AN localization run artifacts."""
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs/phase_b_restart/rec004an/run_001"
    summary_p = run_dir / "summary.json"

    if summary_p.is_file():
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
    else:
        config = CDDPCALocalizationConfig(
            output_dir=run_dir,
            rec004al_dir=repo_root / "runs/phase_b_restart/rec004al/run_001",
            rec004ak_dir=repo_root / "runs/phase_b_restart/rec004ak/run_001",
        )
        summary = run_failure_localization_task(config)

    return run_dir, summary


def test_rec004an_decision_and_classification(localization_run):
    run_dir, summary = localization_run
    assert summary["task_id"] == REC004AN_TASK_ID
    assert summary["source_task_id"] == REC004AN_SOURCE_TASK_ID
    assert summary["status"] == "PASS"
    assert summary["decision"] == "LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED"
    assert summary["trajectory_pattern"] == "NEVER_LEARNED_PATTERN"
    assert summary["next_learning_pilot_authorized"] is False


def test_rec004an_position_localization(localization_run):
    run_dir, summary = localization_run
    pos_data = json.loads((run_dir / "position_localization.json").read_text(encoding="utf-8"))

    # Position 4 must be classified as consistently failing
    assert 4 in pos_data["classification"]["consistently_failing_positions"]
    # Position 4 must account for >80% of all terminal token errors
    assert pos_data["position_4_error_fraction"] > 0.80
    assert pos_data["positions_3_and_4_error_fraction"] > 0.95


def test_rec004an_error_stability(localization_run):
    run_dir, summary = localization_run
    err_data = json.loads((run_dir / "error_stability.json").read_text(encoding="utf-8"))

    # Persistent hard examples must exist and dominate
    assert err_data["persistent_hard_example_count"] > 0
    assert err_data["step6000_error_count"] > 0
    assert err_data["persistent_hard_fraction_of_step6000_errors"] > 0.50


def test_rec004an_transplantation_interpretation(localization_run):
    run_dir, summary = localization_run
    trans_data = json.loads((run_dir / "transplant_diagnostics.json").read_text(encoding="utf-8"))

    # None of the transplants from prior checkpoints should recover L10 performance above 0.50
    assert trans_data["transplant_decision"] == "NO_TRAJECTORY_LOCALIZATION"
    assert trans_data["max_em_intervention_a"] < 0.50
    assert trans_data["max_em_intervention_b"] < 0.50
    assert trans_data["max_em_intervention_c"] < 0.50


def test_rec004an_gradient_conflict_interpretation(localization_run):
    run_dir, summary = localization_run
    grad_data = json.loads((run_dir / "gradient_conflict.json").read_text(encoding="utf-8"))

    # Cosine similarity at step 6000 is approximately orthogonal, not antiparallel
    assert grad_data["severe_gradient_interference_supported"] is False
    assert abs(grad_data["step_6000_cosine_sim_aggregate"]) < 0.30


def test_rec004an_exposure_audit(localization_run):
    run_dir, summary = localization_run
    exp_data = json.loads((run_dir / "training_exposure.json").read_text(encoding="utf-8"))

    assert exp_data["data_scarcity_detected"] is False
    # All lengths should be roughly 20%
    for L in (6, 7, 8, 9, 10):
        rate = exp_data["by_length"][str(L)]["effective_sampling_frequency"]
        assert 0.18 <= rate <= 0.22


def test_rec004an_side_effect_audit(localization_run):
    run_dir, summary = localization_run
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
