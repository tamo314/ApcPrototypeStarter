"""Hand-designed Fetch arm diagnostic using upstream CPU IK and joint control."""
from __future__ import annotations

import numpy as np
import sapien

from .runner import array


def physical_pose(pose):
    return sapien.Pose(array(pose.p)[0], array(pose.q)[0])


class ArmIKPolicy:
    def __init__(self, env, output, *, offset=(0, 0, 0.02), protocol="track", pitch_deg=90,
                 torso_ik=False):
        self.env = env.unwrapped
        self.space = env.action_space
        self.offset = np.asarray(offset, dtype=float)
        if protocol not in ("track", "pick"):
            raise ValueError("Unknown arm protocol")
        self.protocol = protocol
        if not np.isfinite(pitch_deg):
            raise ValueError("pitch must be finite")
        self.pitch_deg = pitch_deg
        self.torso_ik = torso_ik
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError("offset must be three finite metres")
        if (self.env.agent.uid != "fetch" or self.env.control_mode != "pd_joint_delta_pos"
                or self.space.shape != (13,) or self.env.num_envs != 1):
            raise ValueError("Arm diagnostic requires single Fetch / 13-D pd_joint_delta_pos")
        self.metadata = dict(source="hand-designed upstream IK joint tracking",
                             target_offset_world_m=self.offset.tolist(),
                             position_tolerance_m=0.005, orientation_tolerance_rad=0.05,
                             ik_eps=1e-4, ik_max_iterations=500,
                             learned=False, protocol=protocol, grasp_pitch_deg=pitch_deg, torso_ik=torso_ik,
                             stages=["approach", "descend", "close", "lift", "hold"] if protocol == "pick" else ["track"])
        self.metadata["torso_fallback_anchors_m"] = [0.1, 0.2, 0.3, 0.386] if torso_ik else []
        self.metadata["action_mapping"] = self.env.agent.controller.action_mapping
        self.metadata["arm_indices"] = array(self.env.agent.controller.controllers["arm"].active_joint_indices).tolist()
        self.metadata["body_indices"] = array(self.env.agent.controller.controllers["body"].active_joint_indices).tolist()

    def reset(self):
        self.agent = self.env.agent
        self.robot = self.agent.robot
        self.model = self.robot.create_pinocchio_model()
        self.links = self.robot.get_links()
        self.link_index = next(i for i, link in enumerate(self.links)
                               if link.name == self.agent.ee_link_name)
        self.arm_indices = array(self.agent.controller.controllers["arm"].active_joint_indices)
        self.body_indices = array(self.agent.controller.controllers["body"].active_joint_indices)
        self.initial_qpos = array(self.robot.get_qpos())[0]
        self.mask = np.zeros(len(self.initial_qpos), dtype=int)
        self.mask[self.arm_indices] = 1
        if self.torso_ik:
            self.torso_index = int(array(self.robot.joints_map["torso_lift_joint"].active_index).item())
            self.mask[self.torso_index] = 1
        self.ik_indices = np.flatnonzero(self.mask)
        initial = physical_pose(self.links[self.link_index].pose)
        self.target = sapien.Pose(initial.p + self.offset, initial.q)
        self.stage = 0
        self.stage_steps = 0
        self.gripper_target = 0.015
        if self.protocol == "pick":
            self.cube_initial = array(self.env.cube.pose.p)[0]
            self.gripper_target = 0.05
            half_pitch = np.deg2rad(self.pitch_deg) / 2
            self.target = sapien.Pose(self.cube_initial + [0, 0, 0.12],
                                      [np.cos(half_pitch), 0, np.sin(half_pitch), 0])
        self.initial_base = array(self.robot.get_qpos())[0, :3]
        self.last_solver = {}
        return self.observe()

    def action(self):
        qpos = array(self.robot.get_qpos())[0]
        # Pinocchio poses are in the articulation root frame, not world coordinates.
        root = physical_pose(self.robot.pose)
        local_target = root.inv() * self.target
        solution, success, error = self.model.compute_inverse_kinematics(
            self.link_index, local_target, initial_qpos=qpos,
            active_qmask=self.mask, eps=1e-4, max_iterations=500)
        if not np.isfinite(solution).all() or not np.isfinite(error):
            raise ValueError("Nonfinite IK result")
        limits = array(self.robot.get_qlimits())[0]
        within_limits = bool(((solution[self.ik_indices] >= limits[self.ik_indices, 0] - 1e-5)
                              & (solution[self.ik_indices] <= limits[self.ik_indices, 1] + 1e-5)).all())
        attempts = 1
        if self.torso_ik and not (success and within_limits):
            arm_mask = self.mask.copy()
            arm_mask[self.torso_index] = 0
            candidates = []
            for height in self.metadata["torso_fallback_anchors_m"]:
                initial = qpos.copy()
                initial[self.torso_index] = height
                candidate, valid, residual = self.model.compute_inverse_kinematics(
                    self.link_index, local_target, initial_qpos=initial,
                    active_qmask=arm_mask, eps=1e-4, max_iterations=500)
                attempts += 1
                if not np.isfinite(candidate).all() or not np.isfinite(residual):
                    raise ValueError("Nonfinite fallback IK result")
                if valid and ((candidate[self.ik_indices] >= limits[self.ik_indices, 0])
                              & (candidate[self.ik_indices] <= limits[self.ik_indices, 1])).all():
                    candidates.append((np.linalg.norm(candidate - qpos), candidate, residual))
            if candidates:
                _, solution, error = min(candidates, key=lambda item: item[0])
                success = within_limits = True
        self.model.compute_forward_kinematics(solution)
        achieved = root * self.model.get_link_pose(self.link_index)
        self.last_solver = dict(ik_success=bool(success), ik_within_limits=within_limits,
                               ik_attempts=attempts, ik_se3_error=float(error),
                               ik_position_error_m=float(np.linalg.norm(achieved.p - self.target.p)),
                               solution_qpos=solution.tolist(), pre_action_qpos=qpos.tolist())
        action = np.zeros(self.space.shape, dtype=self.space.dtype)
        mapping = self.agent.controller.action_mapping
        start, end = mapping["arm"]
        # Failed finite IK is a diagnostic outcome: hold the arm and record it.
        if success and within_limits:
            action[start:end] = (solution[self.arm_indices] - qpos[self.arm_indices]) / 0.1
        start, end = mapping["body"]
        body_target = self.initial_qpos.copy()
        if success and within_limits and self.torso_ik:
            body_target[self.ik_indices] = solution[self.ik_indices]
        elif self.torso_ik:
            body_target[self.ik_indices] = qpos[self.ik_indices]
        action[start:end] = (body_target[self.body_indices] - qpos[self.body_indices]) / 0.1
        start, end = mapping["gripper"]
        action[start:end] = 2 * (self.gripper_target + 0.01) / 0.06 - 1
        return np.clip(action, self.space.low, self.space.high)

    def observe(self):
        actual = physical_pose(self.links[self.link_index].pose)
        qpos = array(self.robot.get_qpos())[0]
        distance = float(np.linalg.norm(actual.p - self.target.p))
        angle = float(2 * np.arccos(np.clip(abs(np.dot(actual.q, self.target.q)), 0, 1)))
        report = dict(self.last_solver, stage=self.stage, target_position=self.target.p.tolist(),
                    target_quaternion=self.target.q.tolist(), ee_position=actual.p.tolist(),
                    ee_quaternion=actual.q.tolist(), qpos=qpos.tolist(),
                    position_error_m=distance, orientation_error_rad=angle,
                    reached=distance <= 0.005 and angle <= 0.05,
                    body_max_error=float(np.max(abs(qpos[self.body_indices] - self.initial_qpos[self.body_indices]))),
                    base_max_drift=float(np.max(abs(qpos[:3] - self.initial_base))))
        if self.protocol == "pick":
            cube = array(self.env.cube.pose.p)[0]
            report.update(cube_position=cube.tolist(), cube_lift_m=float(cube[2] - self.cube_initial[2]),
                          grasped=bool(array(self.agent.is_grasping(self.env.cube))[0]),
                          finger_forces=[array(self.env.scene.get_pairwise_contact_forces(link, self.env.cube))[0].tolist()
                                         for link in (self.agent.finger1_link, self.agent.finger2_link)])
            contacts = {link.name: array(self.env.scene.get_pairwise_contact_forces(
                link, self.env.table_scene.table))[0].tolist() for link in self.links}
            report["table_contact_forces_n"] = contacts
            report["table_contact_force_norm_sum_n"] = float(sum(np.linalg.norm(force) for force in contacts.values()))
        return report

    def after_step(self):
        report = self.observe()
        self.stage_steps += 1
        if self.protocol == "pick":
            if self.stage in (0, 1, 3) and report["reached"]:
                self.stage += 1
                self.stage_steps = 0
                if self.stage == 1:
                    self.target = sapien.Pose(self.cube_initial, self.target.q)
                elif self.stage == 2:
                    self.gripper_target = -0.01
            elif self.stage == 2 and self.stage_steps >= 15:
                self.stage = 3
                self.stage_steps = 0
                self.target = sapien.Pose(self.cube_initial + [0, 0, 0.15], self.target.q)
        # This is the target/action just executed; transitions affect the next action.
        return report
