"""Explicit rollout protocols for the local Fetch task (no physics resets)."""
import gymnasium as gym

from .runner import scalar


class HoldAfterSuccess(gym.Wrapper):
    """Delay episode termination until N further steps after first task success.

    The geometric success signal is unchanged and may become false again. The
    diagnostic ends after its observation window, regardless of final success.
    Environment time limits still take precedence; there is no auto-reset.
    """

    def __init__(self, env, steps):
        super().__init__(env)
        self.hold_steps = steps
        self.elapsed = 0
        self.first_success = None

    def reset(self, **kwargs):
        self.elapsed = 0
        self.first_success = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed += 1
        if self.first_success is None and bool(scalar(info["success"])):
            self.first_success = self.elapsed
        after = 0 if self.first_success is None else self.elapsed - self.first_success
        complete = self.first_success is not None and after >= self.hold_steps
        info = dict(info, task_terminated=info.get("task_terminated", terminated.clone()),
                    hold_input_terminated=terminated.clone(),
                    first_success_step=-1 if self.first_success is None else self.first_success,
                    post_success_steps=after, hold_complete=complete)
        # FetchReach has no failure termination. Preserve unknown future failures.
        terminated = (terminated & ~info["success"]) | terminated.new_full(terminated.shape, complete)
        return obs, reward, terminated, truncated, info


class TwoGoalSequence(gym.Wrapper):
    """Manually ordered goals, with one unchanged physical scene and policy.

    The second goal is a fixed world-frame offset from the sampled first goal.
    The returned observation always contains the goal for the NEXT action.
    reward_info retains the preceding transition's goal/geometry at a switch.
    """

    def __init__(self, env, offset):
        super().__init__(env)
        self.offset = offset
        self.stage = 0
        self.completed_goals = 0

    def reset(self, **kwargs):
        self.stage = self.completed_goals = 0
        obs, info = self.env.reset(**kwargs)
        self.second_goal = self.unwrapped.goal_xy.clone()
        self.second_goal += self.second_goal.new_tensor(self.offset)
        return obs, dict(info, goal_stage=0, completed_goals=0,
                         second_goal_xy=self.second_goal.clone())

    def step(self, action):
        obs, reward, terminated, truncated, task_info = self.env.step(action)
        info = dict(task_info)
        reward_info = {key: task_info[key] for key in
                       ("goal_xy", "distance", "progress", "base_pose", "success")}
        switch = (self.stage == 0 and bool(scalar(task_info["success"]))
                  and not bool(scalar(truncated)))
        before = None
        if switch:
            before = obs.clone()
            self.stage = self.completed_goals = 1
            self.unwrapped.goal_xy[:] = self.second_goal
            # Changing the goal does not touch robot state, controller or clock.
            self.unwrapped._before_control_step()
            info = self.unwrapped.get_info()
            obs = self.unwrapped.get_obs(info)
        elif self.stage == 1 and bool(scalar(task_info["success"])):
            self.completed_goals = 2
        info = dict(info, local_success=info["success"].clone(),
                    success=info["success"] & (self.stage == 1) & (not switch),
                    goal_stage=self.stage, completed_goals=self.completed_goals,
                    goal_switched=switch, second_goal_xy=self.second_goal.clone(),
                    reward_info=reward_info, task_terminated=terminated.clone())
        if before is not None:
            info["switch_observation_before"] = before
        terminated = (terminated & ~task_info["success"]) | info["success"]
        return obs, reward, terminated, truncated, info
