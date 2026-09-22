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

    demo, initial, relabeled, training, rollout = [
        tmp_path / name for name in ("demo", "initial", "relabeled", "training", "learned")]
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
         "--out", str(relabeled), "--episodes", "2", "--seed", "40", "--max-steps", "6"],
        ["train-operation-bc", "--demo-run", str(demo), "--extra-demo-run", str(relabeled),
         "--out", str(training), "--updates", "40"],
        ["rollout", "--config", str(WORKSPACE / "configs/fetch_pick_bc.json"),
         "--out", str(rollout), "--checkpoint", str(training / "policy.pt"),
         "--episodes", "2", "--seed", "50", "--max-steps", "12"],
    ]
    for args in commands:
        subprocess.run([sys.executable, "-m", "apc_maniskill", *args], check=True, timeout=180)

    report = json.loads((training / "training.json").read_text())
    manifest = json.loads((rollout / "manifest.json").read_text())
    relabel_manifest = json.loads((relabeled / "manifest.json").read_text())
    teacher_rows = [json.loads(line) for line in (demo / "episodes.jsonl").read_text().splitlines()]
    assert report["status"] == manifest["status"] == "completed"
    assert report["completed_updates"] == 40
    assert report["samples"] == sum(row["steps"] for row in teacher_rows) + 12
    assert report["source"]["seeds"] == [30, 31]
    assert [source["data_kind"] for source in report["sources"]] == [
        "scripted_demonstration", "dagger_relabel"]
    assert relabel_manifest["policy_details"]["relabel"] == "dagger"
    assert report["learned_action_slice"] == manifest["policy_details"]["learned_action_slice"] == [0, 11]
    assert report["fixed_base_action"] == manifest["policy_details"]["fixed_base_action"] == [0.0, 0.0]
    assert np.isfinite([report["initial_train_mse"], report["final_train_mse"]]).all()
    assert report["checkpoint_sha256"] == sha256(rollout / "policy.pt")
    assert manifest["policy_details"]["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert len(manifest["policy_details"]["training_sources"]) == 2

    relabeled_steps = [json.loads(line) for line in (relabeled / "steps.jsonl").read_text().splitlines()]
    relabeled_rows = [json.loads(line) for line in (relabeled / "episodes.jsonl").read_text().splitlines()]
    distinct_label = False
    for row in relabeled_rows:
        steps = [step for step in relabeled_steps if step["episode"] == row["episode"]]
        with np.load(relabeled / row["trajectory"], allow_pickle=False) as data:
            for index, step in enumerate(steps):
                diagnostic = step["info"]["diagnostic"]
                teacher = np.asarray(diagnostic["teacher_action"])
                behavior = np.asarray(diagnostic["behavior_action"])
                assert teacher.shape == behavior.shape == (13,)
                assert np.isfinite(teacher).all() and np.isfinite(behavior).all()
                np.testing.assert_array_equal(teacher[11:13], 0)
                np.testing.assert_array_equal(behavior, data["actions"][index])
                np.testing.assert_allclose(diagnostic["pre_action_qpos"],
                                           data["observations"][index, 0, :15], atol=1e-7)
                distinct_label |= not np.allclose(teacher, behavior)
    assert distinct_label

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
