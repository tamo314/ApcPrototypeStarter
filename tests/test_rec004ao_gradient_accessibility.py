"""Unit and regression tests for Task B-C005REC-004AO: CD-DPCA Gradient Accessibility Diagnostic."""

import json
from pathlib import Path

import pytest

from apc.evaluation.mirror_cd_dpca_gradient_accessibility import (
    REC004AO_SOURCE_TASK_ID,
    REC004AO_TASK_ID,
    CDDPCAGradientAccessibilityConfig,
    run_gradient_accessibility_diagnostic_task,
)


@pytest.fixture(scope="module")
def accessibility_run():
    """Load or run REC-004AO gradient accessibility run artifacts."""
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs/phase_b_restart/rec004ao/run_001"
    summary_p = run_dir / "summary.json"

    if summary_p.is_file():
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
    else:
        config = CDDPCAGradientAccessibilityConfig(
            output_dir=run_dir,
            rec004al_dir=repo_root / "runs/phase_b_restart/rec004al/run_001",
            rec004ak_dir=repo_root / "runs/phase_b_restart/rec004ak/run_001",
        )
        summary = run_gradient_accessibility_diagnostic_task(config)

    return run_dir, summary


def test_rec004ao_decision_and_classification(accessibility_run):
    run_dir, summary = accessibility_run
    assert summary["task_id"] == REC004AO_TASK_ID
    assert summary["source_task_id"] == REC004AO_SOURCE_TASK_ID
    assert summary["status"] == "PASS"
    assert summary["decision"] == "LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED"
    assert summary["next_learning_pilot_authorized"] is False


def test_rec004ao_trajectory_metrics(accessibility_run):
    run_dir, summary = accessibility_run
    traj_data = json.loads((run_dir / "trajectory_routing.json").read_text(encoding="utf-8"))

    summary_t = traj_data["summary"]
    # Position 4 must enter wrong attractor at step 500
    assert summary_t["first_wrong_attractor_step"] == 500
    assert summary_t["final_step_top1_key"] == 7
    assert summary_t["final_step_correct_key_prob"] < 0.001
    assert summary_t["final_step_margin"] < -10.0
    assert summary_t["final_step_entropy"] < 1.5

    # Check all checkpoints 500..6000 have top-1 key 7
    ckpts = traj_data["checkpoints"]
    for c in ckpts:
        if c["step"] >= 500:
            assert c["top1_key"] == 7
            assert c["score_margin"] < 0.0


def test_rec004ao_gradient_accessibility(accessibility_run):
    run_dir, summary = accessibility_run
    grad_data = json.loads((run_dir / "gradient_accessibility.json").read_text(encoding="utf-8"))

    ckpts = grad_data["checkpoints"]
    step6000 = ckpts[-1]
    assert step6000["step"] == 6000

    # Gradient magnitude on routing parameters must be substantial
    assert step6000["grad_norms_loss"]["total_routing"] > 0.05
    # First order predicted margin change must be negative at step 6000
    assert step6000["predicted_margin_change_by_group"]["total_routing"] < 0.0
    assert step6000["cosine_alignment_routing"] < 0.0

    # Key 0 gradient must be heavily starved relative to key 7
    k0_norm = step6000["grad_norms_loss"]["key_position_embedding_corr"]
    k7_norm = step6000["grad_norms_loss"]["key_position_embedding_wrong"]
    assert k0_norm < 0.005
    assert k7_norm > 0.02
    assert k7_norm / k0_norm > 10.0


def test_rec004ao_per_example_consistency(accessibility_run):
    run_dir, summary = accessibility_run
    per_ex_data = json.loads((run_dir / "per_example_consistency.json").read_text(encoding="utf-8"))

    step6000_ex = per_ex_data["6000"]
    # Majority of all examples should show non-positive margin change
    assert step6000_ex["all_examples"]["negative_fraction"] > 0.50
    assert step6000_ex["all_examples"]["mean"] < 0.0
    assert step6000_ex["all_examples"]["median"] < 0.0

    # Starvation ratio k0/k7 must be very small
    assert step6000_ex["key_grad_norm_starvation"]["starvation_ratio_k0_over_k7"] < 0.05


def test_rec004ao_controls_comparison(accessibility_run):
    run_dir, summary = accessibility_run
    ctrl_data = json.loads((run_dir / "controls_comparison.json").read_text(encoding="utf-8"))

    # Target position 4 must have failed
    l10_pos4 = ctrl_data["L10_pos4_target"]["steps"][-1]
    assert l10_pos4["token_accuracy"] < 0.50
    assert l10_pos4["top1_key"] != l10_pos4["correct_key"]
    assert l10_pos4["margin"] < -10.0
    assert l10_pos4["predicted_margin_change"] < 0.0

    # Controls must be accurate
    l10_pos0 = ctrl_data["L10_pos0_control"]["steps"][-1]
    assert l10_pos0["token_accuracy"] == 1.0
    assert l10_pos0["top1_key"] == l10_pos0["correct_key"]

    l8_pos4 = ctrl_data["L8_pos4_control"]["steps"][-1]
    assert l8_pos4["token_accuracy"] == 1.0
    assert l8_pos4["top1_key"] == l8_pos4["correct_key"]
    assert l8_pos4["predicted_margin_change"] > 0.0

    l9_pos4 = ctrl_data["L9_pos4_control"]["steps"][-1]
    assert l9_pos4["token_accuracy"] == 1.0
    assert l9_pos4["top1_key"] == l9_pos4["correct_key"]


def test_rec004ao_side_effect_audit(accessibility_run):
    run_dir, summary = accessibility_run
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
