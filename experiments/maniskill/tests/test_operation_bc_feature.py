"""Real scripted operation -> supervised training -> learned closed loop."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from apc_maniskill.bc import sha256
from apc_maniskill.operation_bc import OperationBCPolicy, make_model

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_scripted_operation_training_and_learned_rollout(tmp_path):
    from apc_maniskill.arm_ik import ArmIKPolicy
    from apc_maniskill.runner import RunConfig, collect

    demo, initial, relabeled, training, data_analysis, rollout, rollout_analysis = [
        tmp_path / name for name in (
            "demo", "initial", "relabeled", "training", "data-analysis", "learned",
            "rollout-analysis")]
    collect(RunConfig(env_id="APC-FetchPickCube-v1", robot_uids="fetch", policy="external",
                      episodes=2, max_steps=180, seed=30, env_max_steps=180,
                      post_success_steps=5, task_label="scripted_fetch_arm_pick_place"), demo,
            policy_factory=lambda env, output: ArmIKPolicy(
                env, output, protocol="pick_place", pitch_deg=15, torso_ik=True,
                grasp_height=0.012, table_clearance=True))
    commands = [
        ["train-operation-bc", "--demo-run", str(demo), "--out", str(initial),
         "--updates", "40"],
        ["collect-operation-dagger", "--checkpoint", str(initial / "policy.pt"),
         "--out", str(relabeled), "--episodes", "2", "--seed", "40", "--max-steps", "100"],
        ["train-operation-bc", "--demo-run", str(demo), "--extra-demo-run", str(relabeled),
         "--successful-only", "--exclude-grasp-open-conflicts", "--exclude-invalid-ik-labels",
         "--out", str(training), "--updates", "40"],
        ["analyze-operation-data", "--demo-run", str(demo), "--relabel-run", str(relabeled),
         "--checkpoint", str(training / "policy.pt"), "--neighbors", "1",
         "--all-teacher-episodes", "--out", str(data_analysis)],
        ["rollout", "--config", str(WORKSPACE / "configs/fetch_pick_bc.json"),
         "--out", str(rollout), "--checkpoint", str(training / "policy.pt"),
         "--episodes", "2", "--seed", "50", "--max-steps", "12"],
        ["analyze-operation-rollouts", "--run-dir", str(rollout),
         "--out", str(rollout_analysis)],
    ]
    for args in commands:
        subprocess.run([sys.executable, "-m", "apc_maniskill", *args], check=True, timeout=180)

    report = json.loads((training / "training.json").read_text())
    analysis = json.loads((data_analysis / "analysis.json").read_text())
    rollout_report = json.loads((rollout_analysis / "analysis.json").read_text())
    manifest = json.loads((rollout / "manifest.json").read_text())
    relabel_manifest = json.loads((relabeled / "manifest.json").read_text())
    teacher_rows = [json.loads(line) for line in (demo / "episodes.jsonl").read_text().splitlines()]
    relabeled_rows = [json.loads(line) for line in (relabeled / "episodes.jsonl").read_text().splitlines()]
    assert report["status"] == manifest["status"] == "completed"
    assert report["completed_updates"] == 40
    relabeled_samples = sum(row["steps"] for row in relabeled_rows)
    raw_samples = sum(row["steps"] for row in teacher_rows) + relabeled_samples
    selected_samples = (sum(row["steps"] for row in teacher_rows if row["success_ever"])
                        + relabeled_samples)
    assert report["source_environment_steps"] == raw_samples
    assert report["selected_source_samples_before_filter"] == selected_samples
    assert report["samples"] == sum(source["retained_samples"] for source in report["sources"])
    assert report["samples"] + report["excluded_samples_total"] == selected_samples
    assert report["source"]["seeds"] == [row["seed"] for row in teacher_rows if row["success_ever"]]
    assert report["source"]["available_seeds"] == [row["seed"] for row in teacher_rows]
    assert report["source"]["excluded_seeds"] == [
        row["seed"] for row in teacher_rows if not row["success_ever"]]
    assert [source["data_kind"] for source in report["sources"]] == [
        "scripted_demonstration", "dagger_relabel"]
    assert relabel_manifest["policy_details"]["relabel"] == "dagger"
    assert relabel_manifest["policy_details"]["label_version"] == "physical_state_resync_v1"
    assert report["exclude_grasp_open_conflicts"] and report["exclude_invalid_ik_labels"]
    assert all(source["episodes_sha256"] == sha256(Path(source["run"]) / "episodes.jsonl")
               for source in report["sources"])
    assert report["learned_action_slice"] == manifest["policy_details"]["learned_action_slice"] == [0, 11]
    assert report["fixed_base_action"] == manifest["policy_details"]["fixed_base_action"] == [0.0, 0.0]
    assert np.isfinite([report["initial_train_mse"], report["final_train_mse"]]).all()
    assert report["checkpoint_sha256"] == sha256(rollout / "policy.pt")
    assert manifest["policy_details"]["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert len(manifest["policy_details"]["training_sources"]) == 2

    relabeled_steps = [json.loads(line) for line in (relabeled / "steps.jsonl").read_text().splitlines()]
    distinct_label = False
    invalid_ik_labels = 0
    grasp_open_conflicts = 0
    for row in relabeled_rows:
        steps = [step for step in relabeled_steps if step["episode"] == row["episode"]]
        with np.load(relabeled / row["trajectory"], allow_pickle=False) as data:
            for index, step in enumerate(steps):
                diagnostic = step["info"]["diagnostic"]
                teacher = np.asarray(diagnostic["teacher_action"])
                behavior = np.asarray(diagnostic["behavior_action"])
                assert teacher.shape == behavior.shape == (13,)
                assert np.isfinite(teacher).all() and np.isfinite(behavior).all()
                assert diagnostic["teacher_resynchronized"]
                np.testing.assert_array_equal(teacher[11:13], 0)
                np.testing.assert_array_equal(behavior, data["actions"][index])
                np.testing.assert_allclose(diagnostic["pre_action_qpos"],
                                           data["observations"][index, 0, :15], atol=1e-7)
                invalid_ik_labels += not (diagnostic["ik_success"]
                                          and diagnostic["ik_within_limits"]
                                          and diagnostic["ik_table_clear"])
                grasp_open_conflicts += (data["observations"][index, 0, 30] > 0.5
                                         and teacher[7] > 0)
                distinct_label |= not np.allclose(teacher, behavior)
    assert distinct_label
    assert report["excluded_invalid_ik_labels"] == invalid_ik_labels > 0
    assert report["excluded_grasp_open_conflicts"] == grasp_open_conflicts == 0
    assert analysis["status"] == "completed" and analysis["samples"] == raw_samples
    assert len(analysis["checkpoint_errors"]) == 1
    assert analysis["sources"][0]["episodes_sha256"] == sha256(demo / "episodes.jsonl")
    assert rollout_report["status"] == "completed"
    assert rollout_report["analysis"] == "operation_rollout_comparison_v2"
    rollout_summary = rollout_report["rollouts"][0]
    assert rollout_summary["episodes"] == 2
    assert rollout_summary["episodes_sha256"] == sha256(rollout / "episodes.jsonl")
    assert rollout_summary["protocol"] == {
        "max_steps": 12, "env_max_steps": 350, "post_success_steps": 20}
    assert rollout_summary["checkpoint_sha256"] == sha256(rollout / "policy.pt")
    assert all(item["observed_steps_after_first"] == 0
               and item["post_success_steps"] == 0
               and item["hold_complete"] is False
               and item["success_continuous_after_first"] is None
               for item in rollout_summary["episode_metrics"])

    policy = OperationBCPolicy(rollout / "policy.pt", manifest["task"], manifest["control_freq"])
    torch.manual_seed(0)
    initial = make_model()
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.model.parameters()))
    rows = [json.loads(line) for line in (rollout / "episodes.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [50, 51]
    for row in rows:
        with np.load(rollout / row["trajectory"], allow_pickle=False) as data:
            assert data["observations"].shape == (row["steps"] + 1, 1, 54)
            assert data["actions"].shape == (row["steps"], 13)
            assert all(np.isfinite(data[key]).all() for key in data.files)
            np.testing.assert_allclose(policy.predict(data["observations"][:-1, 0]),
                                       data["actions"][:, 0:11], atol=1e-6)
            np.testing.assert_array_equal(data["actions"][:, 11:13], 0)
    # No loss-improvement or success threshold: this verifies the learning pipeline, not performance.
