"""Autonomous Capability Deficit Detector (W5: T14).

Distinguishes genuine capability deficits from transient failures, normal approach, and infeasible requests.
Monitors execution dynamics across:
- Pre-grasp approach & alignment
- Lifted transit & directionality
- Terminal placement & settling
Does not access privileged ground-truth task labels, stages, or reward indicators.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class DeficitDiagnosis:
    status: str  # "normal", "transient_recoverable", "genuine_deficit", "infeasible"
    streak_steps: int
    reason: str
    trigger_adaptation: bool


class AutonomousDeficitDetector:
    def __init__(self,
                 max_pre_grasp_steps: int = 700,
                 max_transit_stagnation_steps: int = 120,
                 rejection_streak_threshold: int = 25):
        self.max_pre_grasp_steps = max_pre_grasp_steps
        self.max_transit_stagnation_steps = max_transit_stagnation_steps
        self.rejection_streak_threshold = rejection_streak_threshold
        self.reset()

    def reset(self):
        self.step = 0
        self.has_grasped = False
        self.min_dist_cube_goal = np.inf
        self.transit_stagnation_counter = 0
        self.rejection_streak = 0
        self.upward_action_streak = 0
        self.pre_grasp_alignment_streak = 0

    def update(self, observation: Dict[str, Any], executed_id: int, override_reason_code: int = 0) -> DeficitDiagnosis:
        self.step += 1
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        hand = np.asarray(observation["measured_hand_position"])
        grasped = observation.get("grasped", False)
        mode_code = observation.get("mode_code", 0)  # 0: hand, 1: base

        if grasped:
            self.has_grasped = True

        # Track kinodynamic / collision rejections
        if override_reason_code in (2, 3):
            self.rejection_streak += 1
        else:
            self.rejection_streak = 0

        # Physical reachability / feasibility check
        dist_robot_to_goal = np.linalg.norm(goal[:2])
        if dist_robot_to_goal > 1.2 or goal[2] < 0.0 or goal[2] > 0.8:
            return DeficitDiagnosis(
                status="infeasible",
                streak_steps=self.step,
                reason="Target position outside workspace or physical kinematic limits",
                trigger_adaptation=False
            )

        # -------------------------------------------------------------------------
        # Phase 1: Ungrasped (Approach & Grasping)
        # -------------------------------------------------------------------------
        if not grasped:
            self.transit_stagnation_counter = 0
            self.upward_action_streak = 0

            # If already grasped in the past, this is the post-placement release phase
            if self.has_grasped:
                return DeficitDiagnosis(
                    status="normal",
                    streak_steps=0,
                    reason="Post-placement release phase",
                    trigger_adaptation=False
                )

            d_xy = np.linalg.norm(hand[:2] - cube[:2])
            d_z = abs(hand[2] - (cube[2] + 0.012))
            gripper_target = observation.get("gripper_target_m", 0.05)

            # Check if hand is aligned right at grasp zone
            if d_xy < 0.015 and d_z < 0.015:
                if gripper_target >= 0:
                    self.pre_grasp_alignment_streak += 1
                    # Pre-grasp stall (e.g. seed 3009): aligned above cube with open gripper
                    if self.pre_grasp_alignment_streak >= 15:
                        return DeficitDiagnosis(
                            status="transient_recoverable",
                            streak_steps=self.pre_grasp_alignment_streak,
                            reason="Hand aligned above cube with open gripper; recoverable via close_gripper",
                            trigger_adaptation=False
                        )
                else:
                    self.pre_grasp_alignment_streak = 0
            else:
                self.pre_grasp_alignment_streak = 0

            # If arm mode is active but fails to grasp after prolonged search
            if mode_code == 0 and self.step >= self.max_pre_grasp_steps:
                return DeficitDiagnosis(
                    status="genuine_deficit",
                    streak_steps=self.step,
                    reason="Pre-grasp approach time budget exceeded without achieving grasp",
                    trigger_adaptation=True
                )

        # -------------------------------------------------------------------------
        # Phase 2: Grasped (Transit & Placement)
        # -------------------------------------------------------------------------
        else:
            self.pre_grasp_alignment_streak = 0
            dist_cube_goal = np.linalg.norm(cube[:2] - goal[:2])

            # Deficit Indicator 1: Spurious upward drift while carrying cube
            # (Executing hand_z_plus while already elevated above goal)
            if executed_id == 4 and (cube[2] - goal[2]) > 0.08:
                self.upward_action_streak += 1
                if self.upward_action_streak >= 2:
                    return DeficitDiagnosis(
                        status="genuine_deficit",
                        streak_steps=self.upward_action_streak,
                        reason="Spurious upward transit drift away from goal while holding object",
                        trigger_adaptation=True
                    )
            else:
                self.upward_action_streak = 0

            # Deficit Indicator 2: Transit progress stagnation
            # When near goal (placement phase), small movements are normal
            if dist_cube_goal < 0.035:
                self.transit_stagnation_counter = 0
            elif dist_cube_goal < self.min_dist_cube_goal - 0.003:
                self.min_dist_cube_goal = dist_cube_goal
                self.transit_stagnation_counter = 0
            else:
                self.transit_stagnation_counter += 1

            if self.transit_stagnation_counter >= self.max_transit_stagnation_steps:
                return DeficitDiagnosis(
                    status="genuine_deficit",
                    streak_steps=self.transit_stagnation_counter,
                    reason="Carrying cube but failing to make horizontal progress toward goal",
                    trigger_adaptation=True
                )

        # -------------------------------------------------------------------------
        # Kinodynamic Rejection Check
        # -------------------------------------------------------------------------
        if self.rejection_streak >= self.rejection_streak_threshold:
            if hand[2] < 0.05:
                return DeficitDiagnosis(
                    status="transient_recoverable",
                    streak_steps=self.rejection_streak,
                    reason="Table proximity rejection; recoverable via IK backoff",
                    trigger_adaptation=False
                )
            else:
                return DeficitDiagnosis(
                    status="genuine_deficit",
                    streak_steps=self.rejection_streak,
                    reason="Free-space kinodynamic rejection streak",
                    trigger_adaptation=True
                )

        return DeficitDiagnosis(
            status="normal",
            streak_steps=0,
            reason="Progressing normally",
            trigger_adaptation=False
        )
