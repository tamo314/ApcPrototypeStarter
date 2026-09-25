from apc_maniskill.headless import install_headless_compat
install_headless_compat()

import gymnasium as gym
import mani_skill.envs
from apc_maniskill import fetch_pick  # noqa: F401

def main():
    env = gym.make(
        "APC-FetchTruePlaceFar-v1",
        num_envs=1,
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_backend="none",
        reward_mode="normalized_dense",
        obs_mode="state",
        render_mode=None,
    )
    obs, _ = env.reset(seed=3011)
    print("has get_state:", hasattr(env, "get_state"))
    print("has set_state:", hasattr(env, "set_state"))
    if hasattr(env, "get_state"):
        s = env.get_state()
        print("state type:", type(s))
    env.close()

if __name__ == "__main__":
    main()
