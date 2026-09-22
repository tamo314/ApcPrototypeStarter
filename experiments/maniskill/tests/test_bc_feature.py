"""Real teacher -> supervised training -> checkpoint -> learned closed loop."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from apc_maniskill.bc import BCPolicy, features, make_model, sha256

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_teacher_training_and_learned_rollout(tmp_path):
    demo, training, rollout = [tmp_path / name for name in ("demo", "training", "learned")]
    commands = [
        ["rollout", "--config", str(WORKSPACE / "configs/fetch_reach_goal.json"),
         "--out", str(demo), "--episodes", "2", "--seed", "30", "--max-steps", "60"],
        ["train-bc", "--demo-run", str(demo), "--out", str(training), "--updates", "40"],
        ["rollout", "--config", str(WORKSPACE / "configs/fetch_reach_bc.json"),
         "--out", str(rollout), "--checkpoint", str(training / "policy.pt"),
         "--episodes", "2", "--seed", "50", "--max-steps", "12"],
    ]
    for args in commands:
        subprocess.run([sys.executable, "-m", "apc_maniskill", *args], check=True, timeout=180)
    report = json.loads((training / "training.json").read_text())
    manifest = json.loads((rollout / "manifest.json").read_text())
    teacher_rows = [json.loads(x) for x in (demo / "episodes.jsonl").read_text().splitlines()]
    assert report["status"] == manifest["status"] == "completed"
    assert report["completed_updates"] == 40
    assert report["samples"] == sum(r["steps"] for r in teacher_rows)
    assert report["source"]["seeds"] == [30, 31]
    assert np.isfinite([report["initial_train_mse"], report["final_train_mse"]]).all()
    assert report["checkpoint_sha256"] == sha256(rollout / "policy.pt")
    assert manifest["policy_details"]["checkpoint_sha256"] == report["checkpoint_sha256"]
    policy = BCPolicy(rollout / "policy.pt", manifest["task"], manifest["control_freq"])
    torch.manual_seed(0)
    initial = make_model()
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.model.parameters()))
    rows = [json.loads(x) for x in (rollout / "episodes.jsonl").read_text().splitlines()]
    steps = [json.loads(x) for x in (rollout / "steps.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in rows] == [50, 51]
    for row in rows:
        with np.load(rollout / row["trajectory"], allow_pickle=False) as data:
            assert data["observations"].shape == (row["steps"] + 1, 1, 40)
            assert data["actions"].shape == (row["steps"], 13)
            assert all(np.isfinite(data[key]).all() for key in data.files)
            # Reloaded network outputs equal the actual base commands at each PRE-action state.
            np.testing.assert_allclose(policy.predict(data["observations"][:-1, 0]),
                                       data["actions"][:, 11:13], atol=1e-6)
            infos = [row["reset_info"]] + [s["info"] for s in steps if s["episode"] == row["episode"]]
            expected = []
            for info in infos:
                pose = np.array(info["base_pose"])[0]
                rel = np.array(info["goal_xy"])[0] - pose[:2]
                velocity = np.array(info["base_velocity"])[0]
                c, s = np.cos(pose[2]), np.sin(pose[2])
                rotation = np.array([[c, s], [-s, c]])
                expected.append(np.r_[rotation @ rel, rotation @ velocity[:2], velocity[2]])
            np.testing.assert_allclose(features(data["observations"][:, 0]), expected, atol=1e-6)
    result = subprocess.run([sys.executable, "-m", "apc_maniskill", "summarize", str(rollout)],
                            check=True, capture_output=True, text=True, timeout=30)
    summary = json.loads(result.stdout)
    assert summary["episodes"] == 2
    assert summary["steps"] == sum(row["steps"] for row in rows)
    # No loss-improvement or success threshold: this verifies the learning pipeline, not performance.
