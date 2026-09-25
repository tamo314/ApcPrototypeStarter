"""Feature test for HoldContinuousSuccess protocol (T00)."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from apc_maniskill.protocols import HoldContinuousSuccess


class MockEnv(gym.Env):
    def __init__(self, success_pattern):
        super().__init__()
        self.success_pattern = success_pattern
        self.step_idx = 0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, **kwargs):
        self.step_idx = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        idx = min(self.step_idx, len(self.success_pattern) - 1)
        succ = self.success_pattern[idx]
        self.step_idx += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = 1.0 if succ else 0.0
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])
        info = {
            "success": torch.tensor([succ]),
            "is_obj_placed": torch.tensor([succ]),
            "is_robot_static": torch.tensor([True]),
        }
        return obs, reward, terminated, truncated, info


def test_hold_continuous_success_streak_and_reset():
    # Pattern: 2 fails, 3 succs, 1 fail (bounce), 5 succs
    # Target consecutive = 5
    pattern = [False, False, True, True, True, False, True, True, True, True, True, True]
    env = MockEnv(pattern)
    wrapper = HoldContinuousSuccess(env, target_consecutive=5)

    wrapper.reset()
    completed = False
    step_count = 0
    while not completed and step_count < 20:
        step_count += 1
        obs, reward, terminated, truncated, info = wrapper.step(np.zeros(4))
        if bool(info.get("hold_complete")):
            completed = True
            assert info["consecutive_success_achieved"] is True
            assert info["current_consecutive_success"] == 5
            assert info["max_consecutive_success"] == 5
            # First success happened at step 3 (0-indexed 2)
            assert info["first_success_step"] == 3
            break

    assert completed is True
    # 2 fails + 3 succs + 1 fail + 5 succs = 11 steps
    assert step_count == 11


def test_hold_continuous_success_max_observation_timeout():
    # Pattern: 1 succ, then alternating True/False forever (never achieves 5 consecutive)
    # max_observation_steps = 10
    pattern = [True, False, True, False, True, False, True, False, True, False, True, False]
    env = MockEnv(pattern)
    wrapper = HoldContinuousSuccess(env, target_consecutive=5, max_observation_steps=6)

    wrapper.reset()
    completed = False
    step_count = 0
    while not completed and step_count < 20:
        step_count += 1
        obs, reward, terminated, truncated, info = wrapper.step(np.zeros(4))
        if bool(info.get("hold_complete")):
            completed = True
            assert info["consecutive_success_achieved"] is False
            assert info["post_success_steps"] >= 6
            break

    assert completed is True
    # First success at step 1. After 6 steps post-first-success (step 7), timeout occurs
    assert step_count == 7
