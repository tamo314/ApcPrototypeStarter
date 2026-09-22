"""Real two-goal sequence -> continuous state, goal switch and saved evidence."""
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
def test_continuous_two_goal_sequence(tmp_path):
    out = tmp_path / "sequence"
    subprocess.run([
        sys.executable, "-m", "apc_maniskill", "rollout", "--config",
        str(WORKSPACE / "configs/fetch_reach_sequence.json"), "--policy", "fetch_goal",
        "--episodes", "2", "--seed", "30", "--out", str(out),
    ], check=True, timeout=180)
    manifest = json.loads((out / "manifest.json").read_text())
    episodes = [json.loads(s) for s in (out / "episodes.jsonl").read_text().splitlines()]
    steps = [json.loads(s) for s in (out / "steps.jsonl").read_text().splitlines()]
    assert manifest["status"] == "completed"
    assert len(episodes) == 2
    for episode in episodes:
        seq = [s for s in steps if s["episode"] == episode["episode"]]
        stage = 0
        first_success = None
        with np.load(out / episode["trajectory"], allow_pickle=False) as d:
            assert len(d["observations"]) == len(seq) + 1 == episode["steps"] + 1
            assert all(np.isfinite(d[key]).all() for key in d.files)
            assert not d["terminated"][:-1].any() and not d["truncated"][:-1].any()
            for i, row in enumerate(seq):
                info = row["info"]
                before, after = d["observations"][i, 0], d["observations"][i + 1, 0]
                reward_info = info["reward_info"]
                # Reward uses the old goal even when the next observation changes goal.
                np.testing.assert_allclose(reward_info["goal_xy"][0], before[30:32])
                distance_before = np.linalg.norm(before[30:32] - before[34:36])
                distance_after = np.linalg.norm(before[30:32] - after[34:36])
                np.testing.assert_allclose(row["reward"], distance_before - distance_after, atol=1e-6)
                np.testing.assert_allclose(d["rewards"][i], row["reward"], atol=1e-7)
                assert info["elapsed_steps"] == [i + 1]
                expected_switch = stage == 0 and reward_info["success"][0] and not row["truncated"]
                assert info["goal_switched"] == expected_switch
                if expected_switch:
                    stage = 1
                    old_obs = np.array(info["switch_observation_before"])[0]
                    # Full qpos/qvel are identical across the goal-only change.
                    np.testing.assert_array_equal(old_obs[:30], after[:30])
                    np.testing.assert_array_equal(old_obs[34:], after[34:])
                    np.testing.assert_allclose(after[30:32] - before[30:32], [0.7, 0.2], atol=1e-6)
                else:
                    np.testing.assert_array_equal(before[30:32], after[30:32])
                assert info["goal_stage"] == stage
                np.testing.assert_allclose(after[30:32], info["goal_xy"][0])
                success = stage == 1 and info["local_success"][0] and not expected_switch
                assert row["success"] == bool(d["success"][i]) == success
                if success and first_success is None:
                    first_success = i + 1
                complete = first_success is not None and i + 1 - first_success >= 20
                assert row["terminated"] == info["hold_complete"] == complete
            assert episode["completed_goals"] == seq[-1]["info"]["completed_goals"]
            assert episode["success_final"] == seq[-1]["success"]
    # No success floor: inspect actual completion separately in experiment runs.
