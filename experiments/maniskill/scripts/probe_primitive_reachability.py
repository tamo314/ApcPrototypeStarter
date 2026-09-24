"""Replay first arm rejection and compare IK targets without stepping physics."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import sapien

from apc_maniskill.arm_ik import physical_pose
from apc_maniskill.primitive_policy import BaseReadyPickSelector, PrimitivePolicy
from apc_maniskill.runner import RunConfig, array, collect, json_write, summarize


class ReachabilityProbe(PrimitivePolicy):
    def __init__(self, env, output, *, pitch_deg=15, base_switch_x_m=.195):
        super().__init__(env, output, allow_rotation=True, allow_base=True,
                         selector=BaseReadyPickSelector(desired_pitch_deg=pitch_deg,
                                                        base_switch_x_m=base_switch_x_m))
        self.output = output
        self.episode = -1
        self.metadata.update(reachability_probe=True, probe_changes_physics=False)

    def reset(self):
        self.episode += 1
        self.probed = False
        return super().reset()

    def action(self):
        command = super().action()
        if self.pre_action["override_reason_code"] == 2 and not self.probed:
            self.probed = True
            self.probe()
        return command

    def probe(self):
        qpos = array(self.robot.get_qpos())[0]
        root = physical_pose(self.robot.pose)
        actual = physical_pose(self.links[self.link_index].pose)
        target = sapien.Pose(self.pre_action["attempted_target_position"],
                             self.pre_action["attempted_target_quaternion"])
        limits = array(self.robot.get_qlimits())[0]
        report = dict(episode=self.episode, step=self.step, qpos=qpos.tolist(),
                      joint_names=[joint.name for joint in self.robot.get_active_joints()],
                      limits=[[float(v) if np.isfinite(v) else None for v in pair] for pair in limits],
                      root_p=root.p.tolist(), root_q=root.q.tolist(),
                      original_diagnostic=self.last_solver, pre_action=self.pre_action,
                      candidates=[], kinematic_only=True)
        targets = [("original", target)]
        for fraction in (0.5, 0.0):
            targets.append((f"translation_{fraction}", sapien.Pose(
                actual.p + fraction * (target.p - actual.p), target.q)))
        for pitch in (-15, 0, 5, 10, 15, 30, 45):
            angle = np.deg2rad(pitch) / 2
            targets.append((f"pitch_{pitch}", sapien.Pose(target.p, [np.cos(angle), 0, np.sin(angle), 0])))
        cube = array(self.env.cube.pose.p)[0]
        for height in (0.12, 0.06, 0.03, 0.012):
            for pitch in (0, 15, 30):
                angle = np.deg2rad(pitch) / 2
                targets.append((f"cube_height_{height}_pitch_{pitch}", sapien.Pose(
                    cube + [0, 0, height], [np.cos(angle), 0, np.sin(angle), 0])))
        for name, pose in targets:
            for start in ("current", "reset_arm"):
                initial = qpos.copy()
                if start == "reset_arm":
                    initial[self.arm_indices] = self.initial_qpos[self.arm_indices]
                for height in (None, 0.1, 0.2, 0.3, 0.386):
                    seed = initial.copy()
                    mask = self.mask.copy()
                    if height is not None:
                        seed[self.torso_index] = height
                        mask[self.torso_index] = 0
                    solution, valid, residual = self.model.compute_inverse_kinematics(
                        self.link_index, root.inv() * pose, initial_qpos=seed,
                        active_qmask=mask, eps=1e-4, max_iterations=500)
                    if not np.isfinite(solution).all() or not np.isfinite(residual):
                        raise ValueError("Nonfinite probe result")
                    bad = ((solution[self.ik_indices] < limits[self.ik_indices, 0] - 1e-5)
                           | (solution[self.ik_indices] > limits[self.ik_indices, 1] + 1e-5))
                    self.model.compute_forward_kinematics(solution)
                    achieved = root * self.model.get_link_pose(self.link_index)
                    clearance = self.path_clearance(qpos, solution, root) if valid and not bad.any() else None
                    report["candidates"].append(dict(
                        target=name, start=start, torso_height=height, success=bool(valid),
                        within_limits=not bool(bad.any()), violating_indices=self.ik_indices[bad].tolist(),
                        residual=float(residual), position_error_m=float(np.linalg.norm(achieved.p - pose.p)),
                        path_clearance_m=clearance, feasible=bool(clearance is not None and clearance >= 0.002),
                        solution=solution.tolist()))
        json_write(self.output / f"reachability_{self.episode:03d}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--pitch-deg", type=float, default=15)
    parser.add_argument("--base-switch-x-m", type=float, default=.195)
    args = parser.parse_args()
    config = RunConfig(env_id="APC-FetchPickCubeFar-v1", robot_uids="fetch", policy="external",
                       episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
                       env_max_steps=args.max_steps, task_label="primitive_reachability_probe")
    def factory(env, output):
        shutil.copy2(__file__, output / Path(__file__).name)
        return ReachabilityProbe(env, output, pitch_deg=args.pitch_deg,
                                  base_switch_x_m=args.base_switch_x_m)
    collect(config, args.out, policy_factory=factory)
    json_write(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()
