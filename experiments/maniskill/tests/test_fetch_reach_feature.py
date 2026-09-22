"""One real-simulator feature test: reaching task -> closed loop -> evidence.

Checks geometry, reward and episode accounting, never a success-rate floor.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_fetch_reach_to_saved_evidence(tmp_path):
    out = tmp_path / "fetch-reach"
    subprocess.run([
        sys.executable, "-m", "apc_maniskill", "rollout",
        "--config", str(WORKSPACE / "configs/fetch_reach_goal.json"),
        "--out", str(out), "--episodes", "2", "--max-steps", "60",
    ], check=True, timeout=180)
    result = subprocess.run([
        sys.executable, "-m", "apc_maniskill", "summarize", str(out),
    ], check=True, capture_output=True, text=True, timeout=30)
    summary = json.loads(result.stdout)
    manifest = json.loads((out / "manifest.json").read_text())
    episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert summary["status"] == "completed"
    assert summary["episodes"] == manifest["completed_episodes"] == len(episodes) == 2
    assert summary["steps"] == len(steps)
    assert [e["seed"] for e in episodes] == [0, 1]
    task = manifest["task"]
    for episode in episodes:
        rows = [row for row in steps if row["episode"] == episode["episode"]]
        initial = episode["reset_info"]
        previous_distance = initial["distance"][0]
        with np.load(out / episode["trajectory"], allow_pickle=False) as data:
            assert len(rows) == episode["steps"] == len(data["actions"])
            assert len(data["observations"]) == len(rows) + 1
            assert data["actions"].shape[1:] == tuple(manifest["action_shape"]) == (13,)
            for key in data.files:
                assert np.isfinite(data[key]).all()
            assert (data["actions"] >= manifest["action_low"]).all()
            assert (data["actions"] <= manifest["action_high"]).all()
            assert not data["terminated"][:-1].any()
            assert not data["truncated"][:-1].any()
            for i, row in enumerate(rows):
                info = row["info"]
                np.testing.assert_allclose(info["goal_xy"], initial["goal_xy"])
                distance = np.linalg.norm(np.array(info["goal_xy"])[0] - np.array(info["base_pose"])[0, :2])
                speed = np.linalg.norm(np.array(info["base_velocity"])[0, :2])
                yaw_rate = abs(info["base_velocity"][0][2])
                np.testing.assert_allclose(info["distance"][0], distance, atol=1e-6)
                np.testing.assert_allclose(info["speed"][0], speed, atol=1e-6)
                np.testing.assert_allclose(row["reward"], previous_distance - distance, atol=1e-6)
                np.testing.assert_allclose(data["rewards"][i], row["reward"], atol=1e-6)
                success = (distance <= task["goal_tolerance"] and speed <= task["speed_tolerance"]
                           and yaw_rate <= task["yaw_rate_tolerance"])
                assert bool(data["success"][i]) == row["success"] == bool(success)
                assert bool(data["terminated"][i]) == row["terminated"] == bool(success)
                previous_distance = distance
            np.testing.assert_allclose(episode["return"], initial["distance"][0] - previous_distance, atol=1e-6)
            assert episode["runner_truncated"] == (not (data["terminated"][-1] or data["truncated"][-1]))
            assert episode["final_info"] == rows[-1]["info"]
