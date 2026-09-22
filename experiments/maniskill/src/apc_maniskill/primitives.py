"""Versioned first bank of one-step Fetch control-target operations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Keep these IDs stable when rotation and base operations are introduced.
NAMES = ("hand_x_plus", "hand_x_minus", "hand_y_plus", "hand_y_minus",
         "hand_z_plus", "hand_z_minus", "gripper_open", "gripper_close",
         "continue", "hold", "rotate_x_plus", "rotate_x_minus",
         "rotate_y_plus", "rotate_y_minus", "rotate_z_plus", "rotate_z_minus")
CONTINUE = 8
HOLD = 9


@dataclass(frozen=True)
class PrimitiveConfig:
    translation_m: float = 0.01
    rotation_rad: float = float(np.deg2rad(3))
    target_timeout_steps: int = 40
    position_tolerance_m: float = 0.003
    gripper_open_m: float = 0.05
    gripper_closed_m: float = -0.01

    def validate(self):
        if (not np.isfinite(self.translation_m) or self.translation_m <= 0
                or not np.isfinite(self.rotation_rad) or self.rotation_rad <= 0
                or self.position_tolerance_m <= 0
                or self.position_tolerance_m >= self.translation_m
                or not np.isfinite(self.gripper_open_m)
                or not np.isfinite(self.gripper_closed_m)
                or self.gripper_open_m <= self.gripper_closed_m
                or self.target_timeout_steps < 1):
            raise ValueError("Invalid primitive translation, tolerance, or timeout")
