"""Upstream Fetch/PickCube with absolute joint-speed termination semantics."""
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


@register_env("APC-FetchPickCube-v1", max_episode_steps=50)
class FetchPickCube(PickCubeEnv):
    SUPPORTED_ROBOTS = ["fetch"]

    def __init__(self, *args, robot_uids="fetch", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def experiment_metadata(self):
        return dict(upstream_task="PickCube-v1", robot_uid="fetch", goal_tolerance_m=self.goal_thresh,
                    static_rule="absolute joint velocity", body_velocity_threshold=0.2,
                    base_velocity_threshold=0.05, gripper_velocity_excluded=True)

    def evaluate(self):
        info = super().evaluate()
        info["upstream_success"] = info["success"].clone()
        info["upstream_is_robot_static"] = info["is_robot_static"].clone()
        info["upstream_is_grasped"] = info["is_grasped"].clone()
        # PickCube resets the cube away from Fetch's open fingers. CPU contact
        # queries can still contain the previous episode until the first step.
        info["is_grasped"] = info["is_grasped"] & (self.elapsed_steps > 0)
        velocity = self.agent.robot.get_qvel()
        info["body_abs_qvel_max"] = torch.abs(velocity[..., 3:-2]).amax(dim=1)
        info["base_abs_qvel_max"] = torch.abs(velocity[..., :3]).amax(dim=1)
        info["is_robot_static"] = (info["body_abs_qvel_max"] <= 0.2) & (info["base_abs_qvel_max"] <= 0.05)
        info["success"] = info["is_obj_placed"] & info["is_robot_static"]
        return info
