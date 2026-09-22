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
        info = dict(info, task_terminated=terminated.clone(),
                    first_success_step=-1 if self.first_success is None else self.first_success,
                    post_success_steps=after, hold_complete=complete)
        # FetchReach has no failure termination. Preserve unknown future failures.
        terminated = (terminated & ~info["success"]) | terminated.new_full(terminated.shape, complete)
        return obs, reward, terminated, truncated, info
