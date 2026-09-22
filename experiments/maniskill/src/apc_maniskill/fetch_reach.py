"""Small, hand-designed Fetch navigation diagnostic using ManiSkill physics.

The full upstream 13-D action is retained. Diagnostic policies only choose the
two base velocities; arm/body deltas servo back to Fetch's rest keyframe.
"""
import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch import Fetch, FETCH_WHEELS_COLLISION_BIT
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env


@register_env("APC-FetchReachGoal-v1", max_episode_steps=200)
class FetchReachGoal(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch"]
    SUPPORTED_REWARD_MODES = ["dense", "none"]
    agent: Fetch
    goal_tolerance = 0.08
    speed_tolerance = 0.05
    yaw_rate_tolerance = 0.10

    def __init__(self, *args, robot_uids="fetch", control_mode="pd_joint_delta_pos", **kwargs):
        if robot_uids != "fetch" or control_mode != "pd_joint_delta_pos":
            raise ValueError("Reach v1 fixes Fetch and pd_joint_delta_pos")
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("Reach v1 supports one environment")
        super().__init__(*args, robot_uids=robot_uids, control_mode=control_mode, **kwargs)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options):
        self.ground = build_ground(self.scene, floor_width=6)
        # Same collision filtering as ManiSkill's Fetch TableSceneBuilder.
        self.ground.set_collision_group_bit(group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1)
        self.goal_xy = torch.zeros((1, 2), device=self.device)

    def _initialize_episode(self, env_idx, options):
        qpos = np.array(self.agent.keyframes["rest"].qpos, dtype=np.float32, copy=True)[None, :]
        qpos[:, :2] = self._episode_rng.uniform(-0.2, 0.2, (1, 2))
        qpos[:, 2] = self._episode_rng.uniform(-0.4, 0.4, 1)
        offset = np.array([self._episode_rng.uniform(0.5, 1.0),
                           self._episode_rng.uniform(-0.4, 0.4)])
        self.goal_xy[env_idx] = torch.as_tensor(
            qpos[:, :2] + offset, dtype=torch.float32, device=self.device
        )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose())
        self._distance_before = torch.linalg.vector_norm(
            self.goal_xy - self.agent.robot.get_qpos()[:, :2], dim=1
        ).clone()

    def _before_control_step(self):
        self._distance_before = torch.linalg.vector_norm(
            self.goal_xy - self.agent.robot.get_qpos()[:, :2], dim=1
        ).clone()

    def evaluate(self):
        qpos, qvel = self.agent.robot.get_qpos(), self.agent.robot.get_qvel()
        relative = self.goal_xy - qpos[:, :2]
        distance = torch.linalg.vector_norm(relative, dim=1)
        speed = torch.linalg.vector_norm(qvel[:, :2], dim=1)
        yaw_rate = qvel[:, 2].abs()
        in_goal = distance <= self.goal_tolerance
        stopped = (speed <= self.speed_tolerance) & (yaw_rate <= self.yaw_rate_tolerance)
        rest = torch.as_tensor(self.agent.keyframes["rest"].qpos, device=self.device)
        return dict(success=in_goal & stopped, in_goal=in_goal, stopped=stopped,
                    distance=distance, progress=self._distance_before - distance,
                    speed=speed, yaw_rate=yaw_rate, goal_xy=self.goal_xy.clone(),
                    base_pose=qpos[:, :3].clone(), base_velocity=qvel[:, :3].clone(),
                    posture_max_error=(qpos[:, 3:] - rest[3:]).abs().amax(dim=1))

    def _get_obs_extra(self, info):
        qpos = self.agent.robot.get_qpos()
        relative = self.goal_xy - qpos[:, :2]
        c, s = torch.cos(qpos[:, 2]), torch.sin(qpos[:, 2])
        relative_body = torch.stack((c * relative[:, 0] + s * relative[:, 1],
                                     -s * relative[:, 0] + c * relative[:, 1]), dim=1)
        return dict(goal_xy=self.goal_xy.clone(), relative_goal=relative_body,
                    base_pose=qpos[:, :3].clone(),
                    base_velocity=self.agent.robot.get_qvel()[:, :3].clone())

    def compute_dense_reward(self, obs, action, info):
        # Metres of progress per control step; reaching fast is not success.
        return info["progress"]

    def experiment_metadata(self):
        return dict(
            task_version=1, goal_tolerance=self.goal_tolerance,
            speed_tolerance=self.speed_tolerance, yaw_rate_tolerance=self.yaw_rate_tolerance,
            reward="distance_before - distance_after (metres)",
            start_xy_range=[-0.2, 0.2], start_yaw_range=[-0.4, 0.4],
            goal_offset_x_range=[0.5, 1.0], goal_offset_y_range=[-0.4, 0.4],
            action_mapping=self.agent.controller.action_mapping,
            base_action_units=["forward m/s = action[11]", "yaw rad/s = 3.14 * action[12]"],
            posture="Fetch rest keyframe; arm/body delta correction; gripper 0.015 m",
            policy_source="hand-designed diagnostic, no learning or teacher skill sequence",
            observation_extra="goal_xy world; relative_goal body; base_pose world x/y/yaw; base_velocity world vx/vy/yaw_rate",
        )

    def diagnostic_action(self, policy, space):
        """Full upstream action, including the actual posture corrections saved by runner."""
        controller = self.agent.controller
        qpos = self.agent.robot.get_qpos()[0].detach().cpu().numpy()
        rest = self.agent.keyframes["rest"].qpos
        action = np.zeros(space.shape, dtype=space.dtype)
        for name in ("arm", "body"):
            component = controller.controllers[name]
            indices = component.active_joint_indices.cpu().numpy()
            start, end = controller.action_mapping[name]
            action[start:end] = (rest[indices] - qpos[indices]) / 0.1
        start, end = controller.action_mapping["gripper"]
        action[start:end] = 2 * (0.015 - (-0.01)) / 0.06 - 1
        start, end = controller.action_mapping["base"]
        if policy == "fetch_random":
            action[start:end] = space.sample()[start:end]
        elif policy == "fetch_goal":
            relative = self.goal_xy[0].detach().cpu().numpy() - qpos[:2]
            distance = np.linalg.norm(relative)
            heading = np.arctan2(relative[1], relative[0]) - qpos[2]
            heading = np.arctan2(np.sin(heading), np.cos(heading))
            if distance > self.goal_tolerance * 0.75:
                action[start] = min(0.5, 1.5 * distance) * max(0.0, np.cos(heading))
                action[start + 1] = np.clip(2.0 * heading, -1.5, 1.5) / 3.14
        elif policy != "fetch_zero":
            raise ValueError(f"Unknown Fetch diagnostic: {policy}")
        return np.clip(action, space.low, space.high)
