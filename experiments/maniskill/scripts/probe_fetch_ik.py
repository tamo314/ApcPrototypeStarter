"""Check Fetch's upstream CPU FK/IK with the existing 13-D joint controller.

Creates a new runner output directory, retaining dependency/solver failures.
The IK solution is checked kinematically; the short zero-action rollout does
not execute a grasp or establish contact-task success.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np

from apc_maniskill.runner import RunConfig, array, collect, json_write, make_env, summarize


def probe_factory(config, output):
    shutil.copy2(__file__, output / "probe.py")
    import importlib.util
    import sapien

    env = make_env(config, output)
    report = dict(status="running", action_shape=list(env.action_space.shape),
                  pinocchio_import_available=importlib.util.find_spec("pinocchio") is not None,
                  target_offset_m=[0.0, 0.0, 0.02])
    json_write(output / "ik_probe.json", report)
    try:
        env.reset(seed=config.seed)
        agent = env.unwrapped.agent
        model = agent.robot.create_pinocchio_model()
        qpos = array(agent.robot.get_qpos())[0]
        link_index = next(i for i, link in enumerate(agent.robot.get_links())
                          if link.name == agent.ee_link_name)
        model.compute_forward_kinematics(qpos)
        initial = model.get_link_pose(link_index)
        target = sapien.Pose(initial.p + np.array(report["target_offset_m"]), initial.q)
        indices = array(agent.controller.controllers["arm"].active_joint_indices)
        mask = np.zeros(len(qpos), dtype=int)
        mask[indices] = 1
        solution, success, error = model.compute_inverse_kinematics(
            link_index, target, initial_qpos=qpos, active_qmask=mask,
            eps=1e-4, max_iterations=500,
        )
        if not np.isfinite(solution).all() or not np.isfinite(error):
            raise ValueError("Nonfinite IK result")
        model.compute_forward_kinematics(solution)
        actual = model.get_link_pose(link_index)
        report.update(status="completed", ik_success=bool(success), se3_error=float(error),
                      position_error_m=float(np.linalg.norm(actual.p - target.p)),
                      inactive_joint_max_delta=float(np.max(np.abs(solution[mask == 0] - qpos[mask == 0]))),
                      arm_indices=indices.tolist(), initial_qpos=qpos.tolist(),
                      solution_qpos=solution.tolist(), target_position=target.p.tolist(),
                      actual_position=actual.p.tolist())
        json_write(output / "ik_probe.json", report)
        return env
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        json_write(output / "ik_probe.json", report)
        env.close()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = RunConfig(env_id="PickCube-v1", robot_uids="fetch",
                       control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
                       episodes=1, max_steps=5, seed=0, policy="zero",
                       task_label="fetch_arm_fk_ik_dependency_probe")
    collect(config, args.out, env_factory=probe_factory)
    json_write(args.out / "summary.json", summarize(args.out))
