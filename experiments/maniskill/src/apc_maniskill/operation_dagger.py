"""Collect learner states with separate scripted operation labels."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from .arm_ik import ArmIKPolicy
from .operation_bc import OperationBCPolicy
from .runner import array


class OperationDaggerPolicy:
    def __init__(self, env, output: Path, checkpoint: Path):
        self.env = env.unwrapped
        checkpoint_copy = output / "behavior_policy.pt"
        shutil.copy2(checkpoint, checkpoint_copy)
        self.learner = OperationBCPolicy(
            checkpoint_copy, self.env.experiment_metadata(), self.env.sim_config.control_freq)
        self.teacher = ArmIKPolicy(self.env, output, protocol="pick_place", pitch_deg=15,
                                   torso_ik=True, grasp_height=0.012, table_clearance=True)
        self.metadata = dict(source="learned rollout with separate scripted teacher labels",
                             learned=True, protocol="pick_place", relabel="dagger",
                             label_source="hand-designed upstream IK joint tracking",
                             label_version="physical_state_resync_v1",
                             label_resynchronization=dict(
                                 grasped="lift below 0.10 m, then transport, then hold at goal",
                                 ungrasped="approach, aligned descent, or close from current cube pose",
                                 approach_height_m=0.12, grasp_position_tolerance_m=0.025,
                                 alignment_xy_tolerance_m=0.03,
                                 orientation_tolerance_rad=0.15),
                             behavior_policy=self.learner.metadata,
                             teacher_policy=self.teacher.metadata,
                             action_mapping=self.teacher.metadata["action_mapping"])
        self.pending_teacher = None
        self.pending_behavior = None

    def reset(self):
        self.pending_teacher = None
        self.pending_behavior = None
        return self.teacher.reset()

    def action(self):
        self.teacher.synchronize_pick_place()
        self.pending_teacher = array(self.teacher.action())
        observation = array(self.env.get_obs())
        action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        action[0:11] = self.learner.predict(observation)[0]
        self.pending_behavior = action.copy()
        return action

    def after_step(self):
        report = self.teacher.after_step()
        report["teacher_action"] = self.pending_teacher.tolist()
        report["behavior_action"] = self.pending_behavior.tolist()
        return report
