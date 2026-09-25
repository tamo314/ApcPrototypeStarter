import numpy as np
import pytest
from apc_maniskill.primitive_policy import GraspRecoveryGuardSelector
from apc_maniskill.primitives import CONTINUE

class DummyBaseSelector:
    primitive_count = 20
    metadata = {"selector": "dummy_base"}

    def __init__(self, default_id=CONTINUE):
        self.default_id = default_id
        self.last_scores = [0.0] * 20

    def reset(self):
        pass

    def select(self, step, observation):
        return self.default_id

def test_grasp_recovery_missed_grasp():
    base = DummyBaseSelector(default_id=8)
    guard = GraspRecoveryGuardSelector(base, grasp_height_m=0.012)

    # Hand precisely at cube grasp height, gripper open
    obs = {
        "grasped": False,
        "gripper_target_m": 0.05,
        "cube_position": [0.0, 0.0, 0.02],
        "measured_hand_position": [0.002, 0.001, 0.032], # d_xy ~ 0.0022, d_z ~ 0
        "goal_position": [0.15, 0.20, 0.02],
        "cube_initial_z": 0.02,
    }
    action = guard.select(0, obs)
    assert action == 7  # close_gripper
    assert guard.last_guard_triggered is True
    assert guard.active_module == "grasp_recovery"

def test_grasp_recovery_normal_grasped_no_intervention():
    base = DummyBaseSelector(default_id=8)
    guard = GraspRecoveryGuardSelector(base, grasp_height_m=0.012)

    obs = {
        "grasped": True,
        "gripper_target_m": -0.01,
        "cube_position": [0.0, 0.0, 0.10],
        "measured_hand_position": [0.0, 0.0, 0.10],
        "goal_position": [0.15, 0.20, 0.02],
        "cube_initial_z": 0.02,
    }
    action = guard.select(0, obs)
    assert action == 8  # Base's default CONTINUE
    assert guard.last_guard_triggered is False
    assert guard.active_module == "base"

def test_grasp_recovery_lost_grasp_cycle():
    base = DummyBaseSelector(default_id=8)
    guard = GraspRecoveryGuardSelector(base, grasp_height_m=0.012)

    # Cube fell on table, gripper closed, not grasped
    obs = {
        "grasped": False,
        "gripper_target_m": -0.01,
        "cube_position": [0.0, 0.0, 0.02],
        "measured_hand_position": [0.0, 0.0, 0.05],
        "goal_position": [0.15, 0.20, 0.02],
        "cube_initial_z": 0.02,
    }

    # First 15 steps should not trigger recovery
    for s in range(15):
        act = guard.select(s, obs)
        assert act == 8
        assert guard.last_guard_triggered is False

    # Step 16 (closing_steps > 15): trigger opening
    act = guard.select(16, obs)
    assert act == 6  # open_gripper
    assert guard.last_guard_triggered is True

    # When gripper is opening (target becomes open > 0.02), hand is below cube+0.08 and d_xy > 0.010
    obs["gripper_target_m"] = 0.05
    obs["measured_hand_position"] = [0.03, 0.0, 0.05]
    obs["root_rotation"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    act = guard.select(17, obs)
    assert act == 4  # hand_z_plus to clear
    assert guard.last_guard_triggered is True

    # Once hand has cleared cube (> cube_z + 0.08), align XY
    obs["measured_hand_position"] = [0.03, 0.0, 0.11]  # d_xy = 0.03 > 0.008, delta_x = -0.03 -> act = 1 (-X)
    act = guard.select(18, obs)
    assert act == 1  # -X to align with cube
    assert guard.recovery_mode == "aligning"

    # Once aligned in XY, descend toward grasp height
    obs["measured_hand_position"] = [0.002, 0.0, 0.06] # grasp_z is 0.032, hand_z > 0.038
    act = guard.select(19, obs)
    assert act == 5  # hand_z_minus to descend
    assert guard.recovery_mode == "descending"

    # At grasp height, close gripper
    obs["measured_hand_position"] = [0.002, 0.0, 0.035]
    act = guard.select(20, obs)
    assert act == 7  # close_gripper
    assert guard.recovery_mode is None
