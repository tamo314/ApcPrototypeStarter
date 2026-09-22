"""One real-simulator test: CPU IK -> joint control -> physics -> saved evidence."""
import json
import os

import numpy as np
import pytest


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_arm_ik_to_saved_evidence(tmp_path):
    from apc_maniskill.arm_ik import ArmIKPolicy, physical_pose
    from apc_maniskill.runner import RunConfig, array, collect, summarize

    class CheckedPolicy(ArmIKPolicy):
        def after_step(self):
            # Cross-check upstream kinematics against the actual simulated link.
            self.model.compute_forward_kinematics(array(self.robot.get_qpos())[0])
            fk = physical_pose(self.robot.pose) * self.model.get_link_pose(self.link_index)
            actual = physical_pose(self.links[self.link_index].pose)
            np.testing.assert_allclose(fk.p, actual.p, atol=2e-6)
            assert abs(np.dot(fk.q, actual.q)) > 1 - 1e-5
            return super().after_step()

    for protocol in ("track", "pick"):
        out = tmp_path / protocol
        collect(RunConfig(robot_uids="fetch", policy="external", episodes=2,
                          max_steps=80, env_max_steps=80), out,
                policy_factory=lambda env, output: CheckedPolicy(
                    env, output, protocol=protocol, pitch_deg=0, torso_ik=protocol == "pick"))
        manifest = json.loads((out / "manifest.json").read_text())
        episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
        summary = summarize(out)
        assert summary["status"] == "completed"
        assert summary["episodes"] == len(episodes) == 2
        assert summary["steps"] == len(steps)
        arm_indices = manifest["policy_details"]["arm_indices"]
        start, end = manifest["policy_details"]["action_mapping"]["arm"]
        for episode in episodes:
            rows = [row for row in steps if row["episode"] == episode["episode"]]
            with np.load(out / episode["trajectory"], allow_pickle=False) as data:
                assert data["actions"].shape == (len(rows), 13)
                assert len(data["observations"]) == len(rows) + 1
                assert all(np.isfinite(data[key]).all() for key in data.files)
                assert (data["actions"] >= manifest["action_low"]).all()
                assert (data["actions"] <= manifest["action_high"]).all()
                assert not data["terminated"][:-1].any() and not data["truncated"][:-1].any()
                for i, row in enumerate(rows):
                    info = row["info"]["diagnostic"]
                    error = np.linalg.norm(np.array(info["target_position"]) - info["ee_position"])
                    np.testing.assert_allclose(info["position_error_m"], error, atol=1e-7)
                    expected = np.zeros(7)
                    if info["ik_success"] and info["ik_within_limits"]:
                        expected = (np.array(info["solution_qpos"])[arm_indices]
                                    - np.array(info["pre_action_qpos"])[arm_indices]) / 0.1
                    np.testing.assert_allclose(data["actions"][i, start:end], np.clip(expected, -1, 1), atol=1e-7)
                    np.testing.assert_allclose(data["rewards"][i], row["reward"], atol=1e-6)
                    assert bool(data["success"][i]) == row["success"]
                    if protocol == "pick":
                        assert info["grasped"] == row["info"]["is_grasped"][0]
                        np.testing.assert_allclose(info["cube_lift_m"],
                            info["cube_position"][2] - episode["reset_info"]["diagnostic"]["cube_position"][2], atol=1e-7)
                assert episode["final_info"] == rows[-1]["info"]
                np.testing.assert_allclose(episode["return"], sum(data["rewards"]), atol=1e-5)
