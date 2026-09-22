"""Persistent target executor and a bounded manual selector for the first ten IDs."""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import sapien

from .arm_ik import ArmIKPolicy, physical_pose
from .primitives import CONTINUE, HOLD, NAMES, NAMES20, PrimitiveConfig
from .runner import array


class ManualSelector:
    """Exercise update, continued tracking, gripper retention and hold in one scene."""

    schedule = {0: 4, 15: 0, 30: 2, 45: 7, 60: 6, 75: HOLD}

    def select(self, step, observation):
        return self.schedule.get(step, CONTINUE)


class BaseDemoSelector:
    """Exercise hand/base interruptions and retained finger width in one scene."""

    schedule = {0: 4, 15: 7, 30: 16, 65: 18, 100: 17, 135: 19,
                160: 6, 170: 0, 200: HOLD}

    def select(self, step, observation):
        return self.schedule.get(step, CONTINUE)


class PickPlaceSelector:
    """Physical-state teacher using only the same ten executor operations."""

    def __init__(self, grasp_height_m=0.012, use_rotation=False):
        self.grasp_height_m = grasp_height_m
        self.use_rotation = use_rotation

    def reset(self):
        pass

    def select(self, step, observation):
        hand = np.asarray(observation["measured_hand_position"])
        if self.use_rotation:
            q = np.asarray(observation["measured_hand_quaternion"])
            desired_pitch = np.deg2rad(15)
            pitch = 2 * np.arctan2(q[2], q[0])
            pending_q = np.asarray(observation["hand_target_quaternion"])
            pending_angle = 2 * np.arccos(np.clip(abs(np.dot(q, pending_q)), 0, 1))
            if abs(pitch - desired_pitch) > np.deg2rad(2):
                if pending_angle > 0.02 and observation["target_age_steps"] < 12:
                    return CONTINUE
                return 12 if pitch < desired_pitch else 13
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        if observation["grasped"]:
            if observation["gripper_target_m"] >= 0:
                return 7
            if cube[2] < observation["cube_initial_z"] + 0.10:
                desired = hand + [0, 0, 0.15]
            else:
                desired = hand + goal - cube
        else:
            grasp = cube + [0, 0, self.grasp_height_m]
            horizontal = np.linalg.norm(hand[:2] - grasp[:2])
            if horizontal < 0.008 and np.linalg.norm(hand - grasp) < 0.008:
                if observation["gripper_target_m"] >= 0:
                    return 7
                return CONTINUE
            desired = grasp if horizontal < 0.008 else cube + [0, 0, 0.12]
        delta_world = desired - hand
        if np.linalg.norm(delta_world) < 0.007:
            return HOLD if observation["grasped"] else CONTINUE
        # One measured-position update at a time; the chosen target remains in
        # force while its error exceeds tolerance.
        pending = np.linalg.norm(np.asarray(observation["hand_target_position"]) - hand)
        if pending > observation["position_tolerance_m"] and observation["target_age_steps"] < 12:
            return CONTINUE
        root_rotation = np.asarray(observation["root_rotation"])
        delta_root = root_rotation.T @ delta_world
        axis = int(np.argmax(np.abs(delta_root)))
        return 2 * axis + int(delta_root[axis] < 0)


class PrimitivePolicy(ArmIKPolicy):
    def __init__(self, env, output, *, config: PrimitiveConfig | None = None, selector=None,
                 allow_rotation=False, allow_base=False, query_teacher=False):
        self.primitive_config = config or PrimitiveConfig()
        self.primitive_config.validate()
        self.selector = selector or ManualSelector()
        self.allow_rotation = allow_rotation
        self.allow_base = allow_base
        if allow_base and not allow_rotation:
            raise ValueError("Base candidates extend the stable 16-ID bank")
        self.teacher = PickPlaceSelector(use_rotation=allow_rotation) if query_teacher else None
        super().__init__(env, output, protocol="track", torso_ik=True, table_clearance=True,
                         offset=(0, 0, 0))
        if tuple(self.env.agent.controller.action_mapping["base"]) != (11, 13):
            raise ValueError("Fetch base channels must be action[11:13]")
        self.metadata.update(source="primitive selector and persistent IK target executor",
                             primitive_version=("fetch20_v1" if allow_base else
                                                "fetch16_v1" if allow_rotation else "fetch10_v1"),
                             primitive_names=list(NAMES20 if allow_base else
                                                  NAMES if allow_rotation else NAMES[:10]),
                             primitive_config=asdict(self.primitive_config),
                             selector=("physical_state_pick_place_v1" if isinstance(self.selector, PickPlaceSelector)
                                       else "base_demo_v1" if isinstance(self.selector, BaseDemoSelector)
                                       else "manual_schedule_v1"), learned=False,
                             target_frame="world_from_measured_ee_and_root_axis",
                             quaternion_order="wxyz", decision_period_steps=1,
                             override_reason_codes={"0": "none", "1": "target_timeout",
                                                    "2": "ik_or_joint_or_table_rejected",
                                                    "3": "base_path_rejected"},
                             interruption_reason_codes={"0": "none", "1": "hand_to_base",
                                                        "2": "base_to_hand"},
                             base_action_units=["forward_m_per_s", "yaw_rad_per_s_divided_by_3.14"],
                             teacher_query_ik_feasibility_checked=False)
        if hasattr(self.selector, "metadata"):
            self.metadata.update(self.selector.metadata)

    def reset(self):
        super().reset()
        self.cube_initial_z = float(array(self.env.cube.pose.p)[0, 2])
        if hasattr(self.selector, "reset"):
            self.selector.reset()
        if self.teacher is not None:
            self.teacher.reset()
        self.gripper_target = self.primitive_config.gripper_open_m
        qpos = array(self.robot.get_qpos())[0]
        self.mode = 0  # 0: hand target, 1: base target and held arm/body joints.
        self.base_target = qpos[:3].copy()
        self.base_target_started = 0
        self.arm_hold_qpos = qpos.copy()
        self.step = 0
        self.target_started = 0
        self.previous_id = HOLD
        self.pre_action = {}
        return self._state()

    def _state(self):
        actual = physical_pose(self.links[self.link_index].pose)
        root = physical_pose(self.robot.pose)
        cube = array(self.env.cube.pose.p)[0]
        qpos = array(self.robot.get_qpos())[0]
        qvel = array(self.robot.get_qvel())[0]
        return dict(primitive_step=self.step, selected_id=self.previous_id,
                    hand_target_position=self.target.p.tolist(),
                    hand_target_quaternion=self.target.q.tolist(),
                    gripper_target_m=self.gripper_target,
                    measured_hand_position=actual.p.tolist(),
                    measured_hand_quaternion=actual.q.tolist(),
                    position_tolerance_m=self.primitive_config.position_tolerance_m,
                    target_age_steps=self.step - self.target_started,
                    root_rotation=root.to_transformation_matrix()[:3, :3].tolist(),
                    cube_position=cube.tolist(), cube_initial_z=float(self.cube_initial_z),
                    goal_position=array(self.env.goal_site.pose.p)[0].tolist(),
                    grasped=bool(array(self.env.evaluate()["is_grasped"])[0]),
                    mode_code=self.mode, hand_target_valid=self.mode == 0,
                    base_pose=qpos[:3].tolist(), base_velocity=qvel[:3].tolist(),
                    base_target=self.base_target.tolist(),
                    base_target_age_steps=self.step - self.base_target_started)

    def action(self):
        start = time.perf_counter()
        before = self._state()
        teacher_id = self.teacher.select(self.step, before) if self.teacher is not None else None
        selected = int(self.selector.select(self.step, before))
        selection_seconds = time.perf_counter() - start
        if not 0 <= selected < (len(NAMES20) if self.allow_base else
                               len(NAMES) if self.allow_rotation else 10):
            raise ValueError("Selector returned an invalid primitive ID")
        actual = physical_pose(self.links[self.link_index].pose)
        qpos = array(self.robot.get_qpos())[0]
        base_before = self.base_target.copy()
        mode_before = self.mode
        age = self.step - (self.base_target_started if self.mode else self.target_started)
        timeout = age >= self.primitive_config.target_timeout_steps
        executed = HOLD if timeout and selected == CONTINUE else selected
        reason = 1 if timeout and selected == CONTINUE else 0
        interruption = 0
        base_path_clearance = None
        update_start = time.perf_counter()
        if (executed < 6 or 10 <= executed < 16 or executed == HOLD) and self.mode == 1:
            self.mode = 0
            self.base_target = qpos[:3].copy()
            self.target = sapien.Pose(actual.p, actual.q)
            self.target_started = self.step
            interruption = 2
        if executed < 6:
            delta = np.zeros(3)
            delta[executed // 2] = self.primitive_config.translation_m * (1 if executed % 2 == 0 else -1)
            root = physical_pose(self.robot.pose)
            world_delta = root.to_transformation_matrix()[:3, :3] @ delta
            self.target = sapien.Pose(actual.p + world_delta, self.target.q)
            self.target_started = self.step
        elif 10 <= executed < 16:
            delta = np.zeros(3)
            delta[(executed - 10) // 2] = self.primitive_config.rotation_rad * (1 if executed % 2 == 0 else -1)
            root = physical_pose(self.robot.pose)
            world_delta = root.to_transformation_matrix()[:3, :3] @ delta
            angle = float(np.linalg.norm(world_delta))
            axis = world_delta / angle
            delta_q = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
            w1, x1, y1, z1 = delta_q
            w2, x2, y2, z2 = actual.q
            quaternion = np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                                   w1*x2+x1*w2+y1*z2-z1*y2,
                                   w1*y2-x1*z2+y1*w2+z1*x2,
                                   w1*z2+x1*y2-y1*x2+z1*w2])
            self.target = sapien.Pose(self.target.p, quaternion / np.linalg.norm(quaternion))
            self.target_started = self.step
        elif 16 <= executed < 20:
            if self.mode == 0:
                interruption = 1
            self.mode = 1
            self.arm_hold_qpos = qpos.copy()
            self.base_target = qpos[:3].copy()
            if executed < 18:
                signed_step = self.primitive_config.base_step_m * (1 if executed == 16 else -1)
                self.base_target[:2] += signed_step * np.array([np.cos(qpos[2]), np.sin(qpos[2])])
            else:
                signed_turn = self.primitive_config.base_turn_rad * (1 if executed == 18 else -1)
                self.base_target[2] = np.arctan2(np.sin(qpos[2] + signed_turn),
                                                  np.cos(qpos[2] + signed_turn))
            self.base_target_started = self.step
            candidate = qpos.copy()
            candidate[:3] = self.base_target
            base_path_clearance = self.path_clearance(qpos, candidate, physical_pose(self.robot.pose))
            if base_path_clearance < 0.002:
                self.mode = 0
                self.base_target = qpos[:3].copy()
                self.target = sapien.Pose(actual.p, actual.q)
                self.target_started = self.step
                executed = HOLD
                reason = 3
        elif executed == 6:
            self.gripper_target = self.primitive_config.gripper_open_m
        elif executed == 7:
            self.gripper_target = self.primitive_config.gripper_closed_m
        elif executed == HOLD:
            self.target = sapien.Pose(actual.p, actual.q)
            self.target_started = self.step
        update_seconds = time.perf_counter() - update_start
        attempted_target = self.target.p.tolist()
        attempted_quaternion = self.target.q.tolist()
        control_start = time.perf_counter()
        command = self._base_action(qpos) if self.mode == 1 else super().action()
        control_seconds = time.perf_counter() - control_start
        if self.mode == 1 and not self.last_solver["base_step_clear"]:
            reason = 3
            self.mode = 0
            self.base_target = qpos[:3].copy()
            self.target = sapien.Pose(actual.p, actual.q)
            self.target_started = self.step
        elif self.mode == 0 and not (self.last_solver["ik_success"] and self.last_solver["ik_within_limits"]
                and self.last_solver["ik_table_clear"]):
            reason = 2
            self.target = sapien.Pose(actual.p, actual.q)
            self.target_started = self.step
            # The existing IK controller already sends zero arm/body action on failure.
        self.previous_id = executed
        self.pre_action = dict(pre_action_state=before, proposed_id=selected,
                               executed_id=executed, override_reason_code=reason,
                               interruption_reason_code=interruption,
                               mode_before_code=mode_before, mode_after_code=self.mode,
                               base_target_before=base_before.tolist(),
                               base_target_after=self.base_target.tolist(),
                               attempted_target_position=attempted_target,
                               attempted_target_quaternion=attempted_quaternion,
                               target_after_update=self.target.p.tolist(),
                               gripper_target_after_update_m=self.gripper_target,
                               selector_seconds=selection_seconds,
                               target_update_seconds=update_seconds,
                               ik_and_command_seconds=control_seconds,
                               submitted_action=array(command).tolist())
        if base_path_clearance is not None:
            self.pre_action["base_path_clearance_m"] = base_path_clearance
        if hasattr(self.selector, "last_scores"):
            self.pre_action["selector_logits"] = self.selector.last_scores
        if teacher_id is not None:
            self.pre_action.update(teacher_id=teacher_id,
                                   teacher_label_valid=0 <= teacher_id < len(NAMES),
                                   teacher_ik_feasibility_checked=False)
        return command

    def _base_action(self, qpos):
        qvel = array(self.robot.get_qvel())[0]
        mapping = self.agent.controller.action_mapping
        command = np.zeros(self.space.shape, dtype=self.space.dtype)
        for name, indices in (("arm", self.arm_indices), ("body", self.body_indices)):
            start, end = mapping[name]
            command[start:end] = np.clip((self.arm_hold_qpos[indices] - qpos[indices]) / 0.1,
                                         -0.3, 0.3)
        start, end = mapping["gripper"]
        command[start:end] = 2 * (self.gripper_target + 0.01) / 0.06 - 1
        delta = self.base_target[:2] - qpos[:2]
        forward = np.array([np.cos(qpos[2]), np.sin(qpos[2])])
        left = np.array([-forward[1], forward[0]])
        forward_error = float(np.dot(forward, delta))
        lateral_error = float(np.dot(left, delta))
        yaw_error = float(np.arctan2(np.sin(self.base_target[2] - qpos[2]),
                                     np.cos(self.base_target[2] - qpos[2])))
        forward_speed = float(np.clip(4 * forward_error - 0.5 * np.dot(forward, qvel[:2]),
                                      -0.2, 0.2))
        yaw_speed = float(np.clip(3 * yaw_error + 2 * lateral_error - 0.5 * qvel[2],
                                  -0.6, 0.6))
        predicted = qpos.copy()
        predicted[:2] += 0.05 * forward_speed * forward
        predicted[2] += 0.05 * yaw_speed
        clearance = self.path_clearance(qpos, predicted, physical_pose(self.robot.pose))
        clear = clearance >= 0.002
        start, end = mapping["base"]
        if clear:
            command[start:end] = [forward_speed, yaw_speed / 3.14]
        self.last_solver = dict(ik_skipped_base_mode=True, base_step_clear=bool(clear),
                                base_step_clearance_m=clearance,
                                base_forward_error_m=forward_error,
                                base_lateral_error_m=lateral_error,
                                base_yaw_error_rad=yaw_error,
                                pre_action_qpos=qpos.tolist())
        return np.clip(command, self.space.low, self.space.high)

    def after_step(self):
        report = super().after_step()
        if self.allow_base:
            forces = [array(self.env.scene.get_pairwise_contact_forces(
                link, self.env.table_scene.table))[0] for link in self.links]
            report["table_contact_force_norm_sum_n"] = float(
                sum(np.linalg.norm(force) for force in forces))
        report.update(self.pre_action)
        report.update(post_action_state=self._state())
        self.step += 1
        return report
