"""Single-environment rollouts with explicit episode boundaries and local evidence.

Random/zero policies are instrumentation baselines, not learned APC primitives.
The simulator is imported lazily so diagnostics and summaries work without it.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    """Capture diagnostics without treating unavailable tools as research failures."""
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_source(output: Path) -> None:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    # Copy simulator-backed memory before the next step can mutate it.
    return np.array(value, copy=True)


def scalar(value: Any) -> Any:
    data = array(value)
    if data.size != 1:
        raise ValueError(f"Expected a single-env scalar, got shape {data.shape}")
    return data.item()


def json_info(info: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            result[key] = json_info(value)
        else:
            try:
                data = array(value)
                if data.dtype.kind in "biuf" and np.isfinite(data).all():
                    result[key] = data.tolist()
            except (TypeError, ValueError):
                pass  # Non-numeric metadata is not part of the step log.
    return result


@dataclass(frozen=True)
class RunConfig:
    env_id: str = "PickCube-v1"
    robot_uids: str = "panda"
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cpu"
    reward_mode: str = "normalized_dense"
    episodes: int = 3
    max_steps: int = 64
    seed: int = 0
    policy: str = "random"
    video: bool = False
    split: str = "explore"
    task_label: str = "instrumentation"
    checkpoint: str | None = None
    post_success_steps: int = 0
    next_goal_offset: list[float] | None = None
    env_max_steps: int | None = None

    def validate(self) -> None:
        if self.env_max_steps is not None and (type(self.env_max_steps) is not int or self.env_max_steps < 1):
            raise ValueError("env_max_steps must be a positive integer or None")
        if self.next_goal_offset is not None:
            offset = np.asarray(self.next_goal_offset)
            if (offset.shape != (2,) or offset.dtype.kind not in "fi"
                    or not np.isfinite(offset).all() or np.linalg.norm(offset) < 0.2):
                raise ValueError("next_goal_offset must contain two finite metres with norm >= 0.2")
            if self.env_id != "APC-FetchReachGoal-v1":
                raise ValueError("Goal sequences require APC-FetchReachGoal-v1")
        if type(self.post_success_steps) is not int or self.post_success_steps < 0:
            raise ValueError("post_success_steps must be a nonnegative integer")
        if self.post_success_steps and self.env_id not in ("APC-FetchReachGoal-v1", "APC-FetchPickCube-v1"):
            raise ValueError("Post-success diagnostics require a local Fetch task")
        for name in ("episodes", "max_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - self.episodes:
            raise ValueError("seed must permit all episode seeds in the uint32 range")
        if self.policy not in ("random", "zero", "fetch_goal", "fetch_random", "fetch_zero",
                               "fetch_bc", "fetch_pick_bc", "external"):
            raise ValueError("Unknown instrumentation/Fetch diagnostic policy")
        if self.policy in ("fetch_bc", "fetch_pick_bc"):
            if not isinstance(self.checkpoint, str) or not self.checkpoint:
                raise ValueError(f"{self.policy} requires a checkpoint path")
        elif self.checkpoint is not None:
            raise ValueError("checkpoint is only used with a learned Fetch policy")
        if self.policy in ("fetch_goal", "fetch_random", "fetch_zero", "fetch_bc") \
                and self.env_id != "APC-FetchReachGoal-v1":
            raise ValueError("Fetch diagnostics require APC-FetchReachGoal-v1")
        if self.policy == "fetch_pick_bc" and (self.env_id != "APC-FetchPickCube-v1"
                or self.robot_uids != "fetch" or self.control_mode != "pd_joint_delta_pos"):
            raise ValueError("fetch_pick_bc requires Fetch/PickCube with pd_joint_delta_pos")
        if self.sim_backend not in ("physx_cpu", "physx_cuda"):
            raise ValueError("sim_backend must be physx_cpu or physx_cuda")
        if type(self.video) is not bool:
            raise ValueError("video must be a JSON boolean")
        for name in ("env_id", "robot_uids", "control_mode", "reward_mode", "split", "task_label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")


def make_env(config: RunConfig, output: Path) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 -- registers tasks
    if config.env_id == "APC-FetchReachGoal-v1":
        from . import fetch_reach  # noqa: F401 -- registers local task
    if config.env_id == "APC-FetchPickCube-v1":
        from . import fetch_pick  # noqa: F401 -- registers corrected task

    if not config.video:
        from .headless import install_headless_compat
        install_headless_compat()

    env = gym.make(
        config.env_id,
        num_envs=1,
        robot_uids=config.robot_uids,
        obs_mode="state",
        control_mode=config.control_mode,
        reward_mode=config.reward_mode,
        sim_backend=config.sim_backend,
        render_backend="gpu" if config.video else "none",
        render_mode="rgb_array" if config.video else None,
        **({"max_episode_steps": config.env_max_steps} if config.env_max_steps is not None
           else {"max_episode_steps": 400} if config.next_goal_offset is not None else {}),
    )
    if config.next_goal_offset is not None:
        from .protocols import TwoGoalSequence
        env = TwoGoalSequence(env, config.next_goal_offset)
    if config.post_success_steps:
        from .protocols import HoldAfterSuccess
        env = HoldAfterSuccess(env, config.post_success_steps)
    if config.video:
        try:
            from mani_skill.utils.wrappers.record import RecordEpisode
            env = RecordEpisode(
                env, output_dir=str(output / "videos"), save_trajectory=False, save_video=True
            )
        except BaseException:
            env.close()
            raise
    return env


def provenance() -> dict[str, Any]:
    versions = {}
    for name in ("apc-maniskill", "mani_skill", "sapien", "torch", "gymnasium", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": command(["git", "rev-parse", "HEAD"], WORKSPACE),
        "git_status": command(["git", "status", "--porcelain"], WORKSPACE),
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }


def collect(config: RunConfig, output: Path, *, env_factory: Callable = make_env,
            policy_factory: Callable | None = None) -> Path:
    """Run a bounded experiment; never overwrite a previous run directory.

    An external policy factory receives (env, output) and returns an object with
    metadata, reset(), action(), and after_step(). Reset/step diagnostics are
    numeric dictionaries saved alongside the unchanged task success signal.
    """
    config.validate()
    if (config.policy == "external") != (policy_factory is not None):
        raise ValueError("external policy requires exactly one policy_factory")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_at": utc_now(),
        "config": asdict(config), "num_envs": 1, "obs_mode": "state",
        "provenance": provenance(), "completed_episodes": 0,
        "episode_protocol": ("hold_after_first_success" if config.post_success_steps
                             else "terminate_on_task_done"),
        "goal_protocol": ("manually_ordered_two_goals_world_offset"
                          if config.next_goal_offset is not None else "single_sampled_goal"),
    }
    json_write(output / "manifest.json", manifest)
    freeze = command([sys.executable, "-m", "pip", "freeze"])
    json_write(output / "dependencies.json", freeze)
    if freeze.get("returncode") == 0:
        (output / "requirements.freeze.txt").write_text(freeze["stdout"] + "\n", encoding="utf-8")
    env = None
    start = time.monotonic()
    try:
        snapshot_source(output)
        env = env_factory(config, output)
        space = env.action_space
        if not hasattr(space, "low") or not hasattr(space, "high"):
            raise ValueError("This first runner supports Box actions only")
        manifest["action_shape"] = list(space.shape)
        manifest["action_low"] = array(space.low).tolist()
        manifest["action_high"] = array(space.high).tolist()
        if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
            raise ValueError("Action bounds must be finite")
        manifest["control_freq"] = float(env.unwrapped.sim_config.control_freq)
        manifest["sim_freq"] = float(env.unwrapped.sim_config.sim_freq)
        if hasattr(env.unwrapped, "experiment_metadata"):
            manifest["task"] = env.unwrapped.experiment_metadata()
        learned_policy = None
        external_policy = policy_factory(env, output) if policy_factory is not None else None
        if external_policy is not None:
            manifest["policy_details"] = external_policy.metadata
        elif config.policy in ("fetch_bc", "fetch_pick_bc"):
            checkpoint_copy = output / "policy.pt"
            shutil.copy2(config.checkpoint, checkpoint_copy)
            if config.policy == "fetch_bc":
                from .bc import BCPolicy
                learned_policy = BCPolicy(checkpoint_copy, manifest["task"], manifest["control_freq"])
            else:
                from .operation_bc import OperationBCPolicy
                learned_policy = OperationBCPolicy(
                    checkpoint_copy, manifest["task"], manifest["control_freq"])
            manifest["policy_details"] = learned_policy.metadata
            manifest["policy_details"]["original_checkpoint"] = str(Path(config.checkpoint).resolve())
        else:
            manifest["policy_details"] = {"source": "hand-designed diagnostic"}
        json_write(output / "manifest.json", manifest)
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes_file, \
             (output / "steps.jsonl").open("x", encoding="utf-8") as steps_file:
            for episode in range(config.episodes):
                episode_start = time.monotonic()
                seed = config.seed + episode
                space.seed(seed)
                obs, reset_info = env.reset(seed=seed)
                if external_policy is not None:
                    reset_info = dict(reset_info, diagnostic=external_policy.reset())
                saved_reset_info = json_info(reset_info)
                observations = [array(obs)]
                if not np.isfinite(observations[0]).all():
                    raise ValueError("Nonfinite reset observation")
                actions, rewards, terminations, truncations, successes = [], [], [], [], []
                for step in range(config.max_steps):
                    action = space.sample() if config.policy == "random" else np.clip(
                        np.zeros(space.shape, dtype=space.dtype), space.low, space.high
                    )
                    if external_policy is not None:
                        action = external_policy.action()
                    elif learned_policy is not None:
                        if config.policy == "fetch_bc":
                            # Only posture is shared; no scripted base fallback or stopping rule.
                            action = env.unwrapped.diagnostic_action("fetch_zero", space)
                            start_idx, end_idx = env.unwrapped.agent.controller.action_mapping["base"]
                            action[start_idx:end_idx] = learned_policy.predict(array(obs))[0]
                        else:
                            action = np.zeros(space.shape, dtype=space.dtype)
                            action[0:11] = learned_policy.predict(array(obs))[0]
                    elif config.policy.startswith("fetch_"):
                        action = env.unwrapped.diagnostic_action(config.policy, space)
                    if not np.isfinite(action).all():
                        raise ValueError("Nonfinite action")
                    saved_action = array(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if external_policy is not None:
                        info = dict(info, diagnostic=external_policy.after_step())
                    observation, r = array(obs), float(scalar(reward))
                    if not np.isfinite(observation).all() or not math.isfinite(r):
                        raise ValueError("Nonfinite observation/reward; preserve run and inspect physics")
                    term, trunc = bool(scalar(terminated)), bool(scalar(truncated))
                    success = bool(scalar(info["success"])) if "success" in info else None
                    observations.append(observation)
                    actions.append(saved_action)
                    rewards.append(r)
                    terminations.append(term)
                    truncations.append(trunc)
                    successes.append(-1 if success is None else int(success))
                    steps_file.write(json.dumps({
                        "episode": episode, "step": step, "reward": r,
                        "terminated": term, "truncated": trunc, "success": success,
                        "info": json_info(info),
                    }, allow_nan=False) + "\n")
                    steps_file.flush()
                    if term or trunc:
                        break  # No auto-reset and no steps after an episode boundary.
                trajectory = f"episode_{episode:04d}.npz"
                np.savez_compressed(
                    output / trajectory, observations=np.stack(observations),
                    actions=np.stack(actions), rewards=np.asarray(rewards, dtype=np.float32),
                    terminated=np.asarray(terminations, dtype=bool),
                    truncated=np.asarray(truncations, dtype=bool),
                    success=np.asarray(successes, dtype=np.int8),
                )
                known = [s for s in successes if s != -1]
                row = {
                    "episode": episode, "seed": seed, "steps": len(actions),
                    "return": sum(rewards),
                    "success_ever": bool(max(known)) if known else None,
                    "success_final": bool(successes[-1]) if successes[-1] != -1 else None,
                    "terminated": terminations[-1], "truncated": truncations[-1],
                    "runner_truncated": not (terminations[-1] or truncations[-1]),
                    "wall_seconds": time.monotonic() - episode_start,
                    "trajectory": trajectory,
                    "reset_info": saved_reset_info, "final_info": json_info(info),
                }
                if config.post_success_steps:
                    row.update(first_success_step=info["first_success_step"],
                               post_success_steps=info["post_success_steps"],
                               hold_complete=info["hold_complete"])
                if config.next_goal_offset is not None:
                    row["completed_goals"] = info["completed_goals"]
                episodes_file.write(json.dumps(row, allow_nan=False) + "\n")
                episodes_file.flush()
                manifest["completed_episodes"] += 1
                json_write(output / "manifest.json", manifest)
        env.close()  # Flush optional videos before reporting completion.
        env = None
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                manifest["close_error"] = repr(exc)
        manifest["finished_at"] = utc_now()
        manifest["wall_seconds"] = time.monotonic() - start
        json_write(output / "manifest.json", manifest)
    return output


def summarize(output: Path) -> dict[str, Any]:
    """Summarize recorded episodes, retaining unknown-success and failure status."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    path = output / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    known = [e["success_ever"] for e in episodes if e["success_ever"] is not None]
    return {
        "status": manifest["status"], "env_id": manifest["config"]["env_id"],
        "policy": manifest["config"]["policy"], "episodes": len(episodes),
        "requested_episodes": manifest["config"]["episodes"],
        "success_observed_episodes": len(known),
        "success_unknown_episodes": len(episodes) - len(known),
        "success_rate_over_observed": sum(known) / len(known) if known else None,
        "success_final_episodes": sum(e["success_final"] is True for e in episodes),
        "hold_complete_episodes": sum(e.get("hold_complete", False) for e in episodes),
        "completed_goals": sum(e.get("completed_goals", 0) for e in episodes),
        "mean_return": sum(e["return"] for e in episodes) / len(episodes) if episodes else None,
        "steps": sum(e["steps"] for e in episodes),
        "runner_truncated_episodes": sum(e["runner_truncated"] for e in episodes),
        "wall_seconds": manifest.get("wall_seconds"),
        "note": ("Supervised policy rollout; not evidence of APC skill-bank learning or reuse."
                  if manifest["config"]["policy"] in ("fetch_bc", "fetch_pick_bc") else
                  "Instrumentation result; not evidence of APC learning or skill reuse."),
    }
