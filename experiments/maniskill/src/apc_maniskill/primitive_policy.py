"""Persistent target executor and diagnostic selectors for Fetch primitives."""
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


class PitchGuardSelector:
    """Diagnostic hybrid: retain MLP decisions except the teacher's pitch correction."""

    def __init__(self, base_selector):
        self.base_selector = base_selector
        self.metadata = dict(base_selector.metadata,
                             selector=("mlp20_with_pitch_guard_v1" if base_selector.primitive_count == 20
                                       else "mlp_with_pitch_guard_v1"),
                             learned=False, hybrid=True, pitch_guard_desired_deg=15,
                             pitch_guard_tolerance_deg=2)
        if base_selector.primitive_count == 20:
            self.metadata["guard_start_base_x_m"] = 0.195
        self.last_scores = []
        self.last_raw_id = HOLD
        self.last_guard_triggered = False

    def reset(self):
        self.base_selector.reset()
        self.last_scores = []
        self.last_raw_id = HOLD
        self.last_guard_triggered = False

    def select(self, step, observation):
        self.last_raw_id = self.base_selector.select(step, observation)
        self.last_scores = self.base_selector.last_scores
        if self.base_selector.primitive_count == 20 and observation["base_pose"][0] < 0.195:
            self.last_guard_triggered = False
            return self.last_raw_id
        q = np.asarray(observation["measured_hand_quaternion"])
        pitch = 2 * np.arctan2(q[2], q[0])
        desired = np.deg2rad(15)
        self.last_guard_triggered = abs(pitch - desired) > np.deg2rad(2)
        if not self.last_guard_triggered:
            return self.last_raw_id
        pending_q = np.asarray(observation["hand_target_quaternion"])
        pending_angle = 2 * np.arccos(np.clip(abs(np.dot(q, pending_q)), 0, 1))
        if pending_angle > 0.02 and observation["target_age_steps"] < 12:
            return CONTINUE
        return 12 if pitch < desired else 13


class TransitGuardSelector:
    """Diagnostic hybrid: redirect spurious upward drift during transit toward horizontal goal approach."""

    def __init__(self, base_selector):
        self.base_selector = base_selector
        self.metadata = dict(base_selector.metadata, hybrid=True, transit_guard=True)
        self.last_scores = []
        self.last_guard_triggered = False

    @property
    def primitive_count(self):
        return self.base_selector.primitive_count

    def reset(self):
        self.base_selector.reset()
        self.last_scores = []
        self.last_guard_triggered = False

    def select(self, step, observation):
        raw_id = self.base_selector.select(step, observation)
        self.last_scores = getattr(self.base_selector, "last_scores", [])
        grasped = observation.get("grasped", False)
        if not grasped:
            self.last_guard_triggered = False
            return raw_id
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        cube_init_z = observation.get("cube_initial_z", 0.02)
        # If cube is already lifted above initial height + 8cm and above goal + 2cm:
        # spurious hand_z_plus (4) is redirected toward horizontal goal approach
        if raw_id == 4 and cube[2] > cube_init_z + 0.08 and cube[2] > goal[2] + 0.02:
            self.last_guard_triggered = True
            root = np.asarray(observation["root_rotation"])
            delta_world = goal - cube
            delta_root = root.T @ delta_world
            axis = int(np.argmax(np.abs(delta_root[:2])))
            return 2 * axis + int(delta_root[axis] < 0)
        self.last_guard_triggered = False
        return raw_id


class RotationProbeSelector:
    """Explicit exploration: perturb rotation while retaining the learned selector elsewhere."""

    def __init__(self, base_selector, schedule=None):
        self.base_selector = base_selector
        self.schedule = schedule if schedule is not None else {450: 12, 520: 13, 590: 12, 660: 13, 730: 12}
        self.metadata = dict(base_selector.metadata, selector="cart_rotation_probe_v1",
                             learned=False, exploration=True, probe_schedule=self.schedule)
        self.last_scores = []
        self.probe_applied = False
        self.probe_raw_id = HOLD

    def reset(self):
        self.base_selector.reset()
        self.last_scores = []
        self.probe_applied = False

    def select(self, step, observation):
        self.probe_raw_id = self.base_selector.select(step, observation)
        self.last_scores = self.base_selector.last_scores
        self.probe_applied = step in self.schedule
        return self.schedule.get(step, self.probe_raw_id)


class PickPlaceSelector:
    """Physical-state teacher using only the same ten executor operations."""

    def __init__(self, grasp_height_m=0.012, use_rotation=False, desired_pitch_deg=15,
                 recover_grasp=False, pre_rotate=False, true_place=False):
        self.grasp_height_m = grasp_height_m
        self.use_rotation = use_rotation
        self.desired_pitch_deg = desired_pitch_deg
        self.recover_grasp = recover_grasp
        self.pre_rotate = pre_rotate
        self.true_place = true_place
        self.recovery_active = False
        self.closing_steps = 0
        self.lifted = False
        self.released = False
        self.released_steps = 0

    def reset(self):
        self.recovery_active = False
        self.closing_steps = 0
        self.lifted = False
        self.released = False
        self.released_steps = 0

    def select(self, step, observation):
        if self.true_place and self.released:
            if self.released_steps < 15:
                self.released_steps += 1
                return 6
            return HOLD
        hand = np.asarray(observation["measured_hand_position"])
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        pending_pos = np.linalg.norm(np.asarray(observation["hand_target_position"]) - hand)
        if self.use_rotation:
            q = np.asarray(observation["measured_hand_quaternion"])
            desired_pitch = np.deg2rad(self.desired_pitch_deg)
            pitch = 2 * np.arctan2(q[2], q[0])
            pending_q = np.asarray(observation["hand_target_quaternion"])
            pending_angle = 2 * np.arccos(np.clip(abs(np.dot(q, pending_q)), 0, 1))
            if abs(pitch - desired_pitch) > np.deg2rad(2):
                if self.pre_rotate and pending_pos > observation["position_tolerance_m"] and observation["target_age_steps"] < 12:
                    return CONTINUE
                if pending_angle > 0.02 and observation["target_age_steps"] < 12:
                    return CONTINUE
                return 12 if pitch < desired_pitch else 13
        if observation["grasped"]:
            self.recovery_active = False
            self.closing_steps = 0
            if observation["gripper_target_m"] >= 0 and not self.true_place:
                return 7
            if not self.lifted and cube[2] < observation["cube_initial_z"] + 0.10:
                desired = hand + [0, 0, 0.15]
            else:
                self.lifted = True
                desired = hand + goal - cube
                if self.true_place and np.linalg.norm(cube - goal) < 0.035:
                    self.released = True
                    return 6  # Open gripper to release onto table
        else:
            self.lifted = False
            if observation["gripper_target_m"] < 0:
                self.closing_steps = getattr(self, "closing_steps", 0) + 1
            else:
                self.closing_steps = 0
            if self.recover_grasp and observation["gripper_target_m"] < 0 and self.closing_steps > 20:
                self.recovery_active = True
                return 6
            if self.recovery_active:
                if observation["gripper_target_m"] < 0.02:
                    return 6
                if hand[2] < cube[2] + 0.08:
                    desired = cube + [0, 0, 0.12]
                else:
                    self.recovery_active = False
                    desired = cube + [0, 0, 0.12]
            else:
                grasp = cube + [0, 0, self.grasp_height_m]
                horizontal = np.linalg.norm(hand[:2] - grasp[:2])
                if horizontal < 0.008 and np.linalg.norm(hand - grasp) < 0.008:
                    if observation["gripper_target_m"] >= 0:
                        return 7
                    return CONTINUE
                if self.pre_rotate and self.use_rotation:
                    q = np.asarray(observation["measured_hand_quaternion"])
                    desired_pitch = np.deg2rad(self.desired_pitch_deg)
                    pitch = 2 * np.arctan2(q[2], q[0])
                    if abs(pitch - desired_pitch) > np.deg2rad(2):
                        desired = cube + [0, 0, 0.12]
                    else:
                        desired = grasp if horizontal < 0.008 else cube + [0, 0, 0.12]
                else:
                    desired = grasp if horizontal < 0.008 else cube + [0, 0, 0.12]
        delta_world = desired - hand
        if np.linalg.norm(delta_world) < 0.007:
            return HOLD if observation["grasped"] else CONTINUE
        # One measured-position update at a time; the chosen target remains in
        # force while its error exceeds tolerance.
        if pending_pos > observation["position_tolerance_m"] and observation["target_age_steps"] < 12:
            return CONTINUE
        root_rotation = np.asarray(observation["root_rotation"])
        delta_root = root_rotation.T @ delta_world
        axis = int(np.argmax(np.abs(delta_root)))
        return 2 * axis + int(delta_root[axis] < 0)


class StagedPitchPickSelector(PickPlaceSelector):
    """Hand-designed pose schedule, latched from measured descent geometry."""

    def __init__(self, approach_pitch_deg, descend_pitch_deg, grasp_height_m=.012,
                 recover_grasp=False, pre_rotate=False, true_place=False):
        super().__init__(use_rotation=True, desired_pitch_deg=approach_pitch_deg,
                         grasp_height_m=grasp_height_m, recover_grasp=recover_grasp,
                         pre_rotate=pre_rotate, true_place=true_place)
        if not np.isfinite([approach_pitch_deg, descend_pitch_deg]).all():
            raise ValueError("Pitch schedule must be finite")
        self.approach_pitch_deg = approach_pitch_deg
        self.descend_pitch_deg = descend_pitch_deg

    def reset(self):
        super().reset()
        self.descending = False
        self.desired_pitch_deg = self.approach_pitch_deg

    def select(self, step, observation):
        delta = np.asarray(observation["measured_hand_position"]) - observation["cube_position"]
        if self.pre_rotate:
            self.descending = self.descending or observation["grasped"] or (
                np.linalg.norm(delta[:2]) < .015)
        else:
            self.descending = self.descending or observation["grasped"] or (
                np.linalg.norm(delta[:2]) < .012 and delta[2] <= .08)
        self.desired_pitch_deg = self.descend_pitch_deg if self.descending else self.approach_pitch_deg
        return super().select(step, observation)


class BaseThenPickSelector:
    """Manual same-scene chain: one safe base advance, then physical pick teacher."""

    schedule = {0: 16}

    def __init__(self):
        self.pick = PickPlaceSelector(use_rotation=True)

    def reset(self):
        self.pick.reset()

    def select(self, step, observation):
        if step < 35:
            return self.schedule.get(step, CONTINUE)
        return self.pick.select(step, observation)


class WaitThenPickSelector:
    """Matched 35-step delay control for the manual base-to-pick chain."""

    def __init__(self):
        self.pick = PickPlaceSelector(use_rotation=True)

    def reset(self):
        self.pick.reset()

    def select(self, step, observation):
        return CONTINUE if step < 35 else self.pick.select(step, observation)


class BaseRecoverPickSelector:
    """Manual recovery: undo the base advance after repeated arm path rejection."""

    def __init__(self):
        self.pick = PickPlaceSelector(use_rotation=True)
        self.ik_rejection_streak = 0
        self.retreat_started = None

    def reset(self):
        self.pick.reset()
        self.ik_rejection_streak = 0
        self.retreat_started = None

    def select(self, step, observation):
        if step == 0:
            return 16
        if step < 35:
            return CONTINUE
        if self.retreat_started is not None and step < self.retreat_started + 35:
            return CONTINUE
        if self.retreat_started is None:
            self.ik_rejection_streak = (self.ik_rejection_streak + 1
                                        if observation["last_override_reason_code"] == 2 else 0)
            if self.ik_rejection_streak >= 5:
                self.retreat_started = step
                return 17
        return self.pick.select(step, observation)


class BaseApproachPickSelector:
    """Move from the far start toward the standard base pose, then pick in one scene."""

    move_count = 10
    move_interval_steps = 30
    metadata = dict(base_move_count=move_count, base_move_interval_steps=move_interval_steps)

    def __init__(self):
        self.pick = PickPlaceSelector(use_rotation=True)

    def reset(self):
        self.pick.reset()

    def select(self, step, observation):
        if step < self.move_count * self.move_interval_steps:
            return 16 if step % self.move_interval_steps == 0 else CONTINUE
        return self.pick.select(step, observation)


class BaseReadyPickSelector:
    """Advance only after the previous short base target has settled."""

    metadata = dict(base_switch_measured_x_m=0.195, base_ready_error_m=0.001,
                    base_ready_min_age_steps=15)

    def __init__(self, desired_pitch_deg=15, base_switch_x_m=0.195, descend_pitch_deg=None,
                 grasp_height_m=.012, recover_grasp=False, pre_rotate=False, true_place=False):
        if not np.isfinite(base_switch_x_m) or base_switch_x_m < 0:
            raise ValueError("base switch position must be finite and nonnegative")
        if not np.isfinite(grasp_height_m) or grasp_height_m < 0:
            raise ValueError("Grasp height must be finite and nonnegative")
        self.pick = (PickPlaceSelector(use_rotation=True, desired_pitch_deg=desired_pitch_deg,
                                      grasp_height_m=grasp_height_m, recover_grasp=recover_grasp,
                                      pre_rotate=pre_rotate, true_place=true_place)
                     if descend_pitch_deg is None else StagedPitchPickSelector(
                         desired_pitch_deg, descend_pitch_deg, grasp_height_m,
                         recover_grasp=recover_grasp, pre_rotate=pre_rotate, true_place=true_place))
        self.metadata = dict(type(self).metadata, desired_pitch_deg=desired_pitch_deg,
                             base_switch_measured_x_m=base_switch_x_m,
                             descend_pitch_deg=descend_pitch_deg,
                             grasp_height_m=grasp_height_m,
                             recover_grasp=recover_grasp,
                             pre_rotate=pre_rotate,
                             true_place=true_place,
                             pitch_switch_geometry={"horizontal_m": .015 if pre_rotate else .012,
                                                    "height_above_cube_m": None if pre_rotate else .08,
                                                    "latched": True} if descend_pitch_deg is not None else None)

    def reset(self):
        self.pick.reset()

    def select(self, step, observation):
        error = np.linalg.norm(np.asarray(observation["base_target"]) - observation["base_pose"])
        if observation["base_pose"][0] >= self.metadata["base_switch_measured_x_m"]:
            return self.pick.select(step, observation)
        if observation["mode_code"] == 0:
            return 16
        if error < 0.001 and observation["base_target_age_steps"] >= 15:
            return 16
        return CONTINUE


class AxisRetryPickSelector(PickPlaceSelector):
    """Diagnostic axis-order change after a rejected translation target."""

    def select(self, step, observation):
        chosen = super().select(step, observation)
        if (chosen >= 6 or observation["last_override_reason_code"] != 2
                or observation["selected_id"] != chosen):
            return chosen
        hand = np.asarray(observation["measured_hand_position"])
        cube = np.asarray(observation["cube_position"])
        if observation["grasped"]:
            delta = (np.array([0., 0., .15]) if cube[2] < observation["cube_initial_z"] + .10
                     else np.asarray(observation["goal_position"]) - cube)
        else:
            height = self.grasp_height_m if np.linalg.norm((cube - hand)[:2]) < .008 else .12
            delta = cube + [0, 0, height] - hand
        delta = np.asarray(observation["root_rotation"]).T @ delta
        magnitude = np.abs(delta)
        magnitude[chosen // 2] = 0
        axis = int(np.argmax(magnitude))
        return 2 * axis + int(delta[axis] < 0) if magnitude[axis] > .003 else chosen


class BaseReadyAxisRetryPickSelector(BaseReadyPickSelector):
    metadata = dict(BaseReadyPickSelector.metadata, selector="base_ready_axis_retry_pick_v1",
                    learned=False, retry_alternate_axis=True)

    def __init__(self):
        self.pick = AxisRetryPickSelector(use_rotation=True)


class BaseReadyRecoverPickSelector(BaseReadyPickSelector):
    """Diagnostic far-start teacher: retreat once after repeated arm rejection."""

    metadata = dict(BaseReadyPickSelector.metadata, selector="base_ready_recover_pick_v1",
                    recovery_rejections=5, recovery_wait_steps=35, recovery_max_count=1,
                    learned=False)

    def reset(self):
        super().reset()
        self.approached = False
        self.rejection_streak = 0
        self.retreat_started = None

    def select(self, step, observation):
        self.approached = self.approached or observation["base_pose"][0] >= 0.195
        if not self.approached:
            return super().select(step, observation)
        if self.retreat_started is not None and step < self.retreat_started + 35:
            return CONTINUE
        if self.retreat_started is None:
            self.rejection_streak = (self.rejection_streak + 1
                                     if observation["last_override_reason_code"] == 2 else 0)
            if self.rejection_streak >= 5:
                self.retreat_started = step
                return 17
        return self.pick.select(step, observation)


class BaseReadySettledPickSelector(BaseReadyPickSelector):
    """Diagnostic variant: wait for the terminal base target to settle."""

    metadata = dict(base_switch_target_x_m=0.19, base_ready_error_m=0.001,
                    base_ready_min_age_steps=15)

    def select(self, step, observation):
        error = np.linalg.norm(np.asarray(observation["base_target"]) - observation["base_pose"])
        if observation["base_target"][0] >= 0.19 and error < 0.001:
            return self.pick.select(step, observation)
        if observation["mode_code"] == 0:
            return 16
        if error < 0.001 and observation["base_target_age_steps"] >= 15:
            return 16
        return CONTINUE


class PrimitivePolicy(ArmIKPolicy):
    def __init__(self, env, output, *, config: PrimitiveConfig | None = None, selector=None,
                 allow_rotation=False, allow_base=False, query_teacher=False, ik_reset_seed=False,
                 translation_backoff=False):
        self.primitive_config = config or PrimitiveConfig()
        self.primitive_config.validate()
        self.selector = selector or ManualSelector()
        self.allow_rotation = allow_rotation
        self.allow_base = allow_base
        self.translation_backoff = translation_backoff
        if translation_backoff and self.primitive_config.translation_m / 2 <= self.primitive_config.position_tolerance_m:
            raise ValueError("Half translation must exceed tracking tolerance")
        if allow_base and not allow_rotation:
            raise ValueError("Base candidates extend the stable 16-ID bank")
        if allow_base and query_teacher and "start_back_m" not in env.unwrapped.experiment_metadata():
            raise ValueError("The 20-ID base teacher requires the far-start task")
        self.teacher = (BaseReadyPickSelector() if allow_base else
                        PickPlaceSelector(use_rotation=allow_rotation)) if query_teacher else None
        super().__init__(env, output, protocol="track", torso_ik=True, table_clearance=True,
                         offset=(0, 0, 0), ik_reset_seed=ik_reset_seed)
        if tuple(self.env.agent.controller.action_mapping["base"]) != (11, 13):
            raise ValueError("Fetch base channels must be action[11:13]")
        self.metadata.update(source="primitive selector and persistent IK target executor",
                             translation_failure_retry_fraction=0.5 if translation_backoff else None,
                             primitive_version=("fetch20_v1" if allow_base else
                                                "fetch16_v1" if allow_rotation else "fetch10_v1"),
                             primitive_names=list(NAMES20 if allow_base else
                                                  NAMES if allow_rotation else NAMES[:10]),
                             primitive_config=asdict(self.primitive_config),
                             selector=("physical_state_pick_place_v1" if isinstance(self.selector, PickPlaceSelector)
                                       else "base_then_pick_v2" if isinstance(self.selector, BaseThenPickSelector)
                                       else "base_recover_pick_v1" if isinstance(self.selector, BaseRecoverPickSelector)
                                       else "base_approach_pick_v1" if isinstance(self.selector, BaseApproachPickSelector)
                                       else "base_ready_settled_pick_v2" if isinstance(self.selector, BaseReadySettledPickSelector)
                                       else "base_ready_pick_v1" if isinstance(self.selector, BaseReadyPickSelector)
                                       else "wait_then_pick_v1" if isinstance(self.selector, WaitThenPickSelector)
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
                             teacher_query_source=("base_ready_pick_v1" if allow_base else
                                                   "physical_state_pick_place_v1") if query_teacher else None,
                             teacher_query_ik_feasibility_checked=False)
        if hasattr(self.selector, "metadata"):
            self.metadata.update(self.selector.metadata)

    def reset(self):
        super().reset()
        if self.allow_base:
            from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

            for index, link in enumerate(self.links):
                if link.name == "base_link":
                    self.base_collision_index = index
                    self.base_collision_vertices = get_component_mesh(
                        link._objs[0], to_world_frame=False).vertices
                    break
            else:
                raise ValueError("Fetch base_link missing from table path check")
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
                    base_target_age_steps=self.step - self.base_target_started,
                    last_override_reason_code=self.pre_action.get("override_reason_code", 0))

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
            base_path_clearance = self._base_path_clearance(qpos, candidate, physical_pose(self.robot.pose))
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
        backoff_initial_solver = None
        backoff_used = False
        if (self.translation_backoff and executed < 6 and self.mode == 0
                and not (self.last_solver["ik_success"] and self.last_solver["ik_within_limits"]
                         and self.last_solver["ik_table_clear"])):
            backoff_initial_solver = dict(self.last_solver)
            self.target = sapien.Pose(actual.p + .5 * (self.target.p - actual.p), self.target.q)
            command = super().action()
            backoff_used = bool(self.last_solver["ik_success"] and self.last_solver["ik_within_limits"]
                                and self.last_solver["ik_table_clear"])
        retried_target = self.target.p.tolist() if backoff_initial_solver is not None else None
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
                               translation_backoff_initial_solver=backoff_initial_solver,
                               translation_backoff_target=retried_target,
                               translation_backoff_used=backoff_used,
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
        if isinstance(getattr(self.selector, "pick", None), StagedPitchPickSelector):
            self.pre_action.update(pitch_schedule_descending=self.selector.pick.descending,
                                   pitch_schedule_desired_deg=self.selector.pick.desired_pitch_deg)
        pick_selector = getattr(self.selector, "pick", self.selector)
        if hasattr(pick_selector, "recovery_active"):
            self.pre_action["recovery_active"] = bool(pick_selector.recovery_active)
        if hasattr(self.selector, "probe_applied"):
            self.pre_action.update(probe_applied=self.selector.probe_applied,
                                   probe_raw_id=self.selector.probe_raw_id)
        if hasattr(self.selector, "patch_applied"):
            self.pre_action.update(patch_applied=self.selector.patch_applied)
        if hasattr(self.selector, "last_raw_id"):
            self.pre_action.update(raw_model_id=self.selector.last_raw_id,
                                   pitch_guard_triggered=self.selector.last_guard_triggered,
                                   pitch_guard_changed_id=selected != self.selector.last_raw_id)
        if teacher_id is not None:
            self.pre_action.update(teacher_id=teacher_id,
                                   teacher_label_valid=0 <= teacher_id < (len(NAMES20) if self.allow_base else len(NAMES)),
                                   teacher_ik_feasibility_checked=False)
        return command

    def _base_path_clearance(self, start, end, root):
        arm_vertices = self.collision_vertices
        self.collision_vertices = arm_vertices.copy()
        self.collision_vertices[self.base_collision_index] = self.base_collision_vertices
        try:
            return self.path_clearance(start, end, root)
        finally:
            self.collision_vertices = arm_vertices

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
        clearance = self._base_path_clearance(qpos, predicted, physical_pose(self.robot.pose))
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
            forces = {link.name: float(np.linalg.norm(array(self.env.scene.get_pairwise_contact_forces(
                link, self.env.table_scene.table))[0])) for link in self.links}
            report["table_contact_force_norm_sum_n"] = float(sum(forces.values()))
            report["table_contact_force_by_link_n"] = {
                name: force for name, force in forces.items() if force > 1e-3}
        report.update(self.pre_action)
        report.update(post_action_state=self._state())
        self.step += 1
        return report
