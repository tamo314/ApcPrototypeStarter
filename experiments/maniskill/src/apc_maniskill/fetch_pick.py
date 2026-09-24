"""Upstream Fetch/PickCube with absolute joint-speed termination semantics."""
import torch
import sapien
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


@register_env("APC-FetchPickCubeFar-v1", max_episode_steps=50)
class FetchPickCubeFar(FetchPickCube):
    """Same table and cube distribution, with Fetch starting 20 cm farther back."""

    start_back_m = 0.20

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        from .runner import array

        original = self.agent.robot.pose
        position = array(original.p)[0]
        position[0] -= self.start_back_m
        self.agent.robot.set_pose(sapien.Pose(position, array(original.q)[0]))

    def experiment_metadata(self):
        return dict(super().experiment_metadata(), start_back_m=self.start_back_m,
                    placement_distribution="same as APC-FetchPickCube-v1")


@register_env("APC-FetchPlaceCubeFar-v1", max_episode_steps=50)
class FetchPlaceCubeFar(FetchPickCubeFar):
    """Same far start, with goal placed at an offset location on the table."""

    goal_offset_y_m = 0.15

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        from .runner import array

        original = self.goal_site.pose
        position = array(original.p)[0]
        position[1] += self.goal_offset_y_m
        self.goal_site.set_pose(sapien.Pose(position, array(original.q)[0]))

    def experiment_metadata(self):
        return dict(super().experiment_metadata(), goal_offset_y_m=self.goal_offset_y_m,
                    task_kind="place_cube_far")


@register_env("APC-FetchTruePlaceFar-v1", max_episode_steps=50)
class FetchTruePlaceFar(FetchPickCubeFar):
    """True place: deliver cube to table goal, open gripper to release, and verify static resting."""

    goal_offset_y_m = 0.15

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        from .runner import array

        original = self.goal_site.pose
        position = array(original.p)[0]
        position[1] += self.goal_offset_y_m
        position[2] = 0.02
        self.goal_site.set_pose(sapien.Pose(position, array(original.q)[0]))

    def evaluate(self):
        info = super().evaluate()
        # Measure cube velocity
        cube_vel = self.cube.linear_velocity
        cube_angvel = self.cube.angular_velocity
        cube_speed = torch.linalg.norm(cube_vel, dim=-1)
        cube_angspeed = torch.linalg.norm(cube_angvel, dim=-1)

        # Exact height and horizontal position on table surface (table height ~ 0.02m)
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        cube_height = cube_pos[:, 2]
        dist_xy = torch.linalg.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=-1)

        is_on_surface = (cube_height >= 0.012) & (cube_height <= 0.035)
        is_placed = (dist_xy <= 0.04) & is_on_surface
        is_released = ~info["is_grasped"]
        is_obj_static = (cube_speed <= 0.05) & (cube_angspeed <= 0.5)

        info["cube_height_m"] = cube_height
        info["cube_goal_dist_xy_m"] = dist_xy
        info["is_on_surface"] = is_on_surface
        info["is_obj_placed_surface"] = is_placed
        info["is_released"] = is_released
        info["is_obj_static"] = is_obj_static
        info["success"] = is_placed & is_released & is_obj_static & info["is_robot_static"]
        return info

    def experiment_metadata(self):
        return dict(super().experiment_metadata(), goal_offset_y_m=self.goal_offset_y_m,
                    task_kind="true_place_far", release_required=True,
                    object_static_threshold=0.05)


