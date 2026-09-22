"""Explicitly supervised skill acquisition and fail-closed shadow consolidation.

A fixed PD expert supplies demonstration rollouts on TRAIN tasks only. This is
behavioral cloning into a compact linear controller, NOT autonomous discovery.
The expert is never a hidden fallback during installed-policy execution/evaluation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apc.embodied.control import (
    EmbodiedPrimitiveCall,
    EpisodeResult,
    LinearMotionPrimitive,
    MotionBank,
    MotionCompositionLibrary,
    MotionRecipe,
    policy_features,
    run_episode,
)
from apc.embodied.world import EmbodiedEnv, NavigationTask, WorldConfig, make_task


@dataclass(frozen=True)
class FitResult:
    primitive: LinearMotionPrimitive
    report: dict[str, Any]


def expert_policy(config: WorldConfig, *, spatial: bool = True,
                  name: str = "expert") -> LinearMotionPrimitive:
    """Declared deterministic PD control. Used only by the learner/control gate."""
    weights = np.zeros((3, 7))
    for axis in range(3 if spatial else 2):
        weights[axis, axis] = 2.0/config.max_force
        weights[axis, axis+3] = -2.8/config.max_force
    return LinearMotionPrimitive(name, weights)


def fit_from_demonstrations(config: WorldConfig, seeds: Sequence[int], *,
                           spatial: bool, name: str, ridge: float = 1e-6) -> FitResult:
    """Fit new, separate weights from simulated state/action demonstrations.

    Targets are pre-saturation expert forces; the actuator applies its fixed norm
    bound at runtime. Labels are not final task answers or cached trajectories.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    expert = expert_policy(config, spatial=spatial)
    features, targets = [], []
    episode_steps: list[int] = []
    for seed in seeds:
        task = make_task(seed, "spatial" if spatial else "planar")
        env = EmbodiedEnv(config)
        obs, _ = env.reset(task, seed=seed)
        call = EmbodiedPrimitiveCall(expert.name, {"target": task.goals[0]})
        steps = 0
        while True:
            action = expert.act(obs, call)
            features.append(policy_features(obs, call.arguments))
            targets.append(action[:3])
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                if not info["success"]:
                    raise RuntimeError(f"STOP: deterministic training control failed: {info}")
                episode_steps.append(steps)
                break
    x, y = np.asarray(features), np.asarray(targets)
    rank = int(np.linalg.matrix_rank(x))
    if rank < (7 if spatial else 5):
        raise RuntimeError("STOP: demonstrations do not sufficiently excite the controlled axes")
    coefficients = np.linalg.solve(x.T @ x + ridge*np.eye(7), x.T @ y)
    candidate = LinearMotionPrimitive(name, coefficients.T)
    mse = float(np.mean((x @ coefficients-y)**2))
    return FitResult(candidate, {"method": "ridge_behavior_cloning_of_PD_rollouts",
                                 "train_seeds": list(seeds), "samples": len(x),
                                 "demonstration_arrays_bytes": int(x.nbytes+y.nbytes),
                                 "environment_control_steps": sum(episode_steps),
                                 "expert_successes": len(episode_steps),
                                 "train_mse": mse, "ridge": ridge, "feature_rank": rank,
                                 "temporary_candidate_parameters": 21,
                                 "persistent_parameters_touched": 0})


def validate_policy(config: WorldConfig, policy: LinearMotionPrimitive,
                    tasks: Sequence[NavigationTask], *, backend: str) -> list[EpisodeResult]:
    if not tasks:
        raise ValueError("Shadow validation cannot be empty")
    shadow = MotionBank(backend)
    shadow.add(policy)
    recipe = MotionRecipe("shadow", (policy.name,))
    return [run_episode(config, task, shadow, recipe.bind(task)) for task in tasks]


def consolidate_after_failure(config: WorldConfig, failure: EpisodeResult,
                              candidate: LinearMotionPrimitive, bank: MotionBank,
                              library: MotionCompositionLibrary,
                              shadow_tasks: Sequence[NavigationTask]) -> dict[str, Any]:
    """A candidate becomes persistent only after ALL preregistered shadow tasks pass.

    The existing bank is not modified by fitting or shadow evaluation. A rejected
    candidate remains available to the caller as evidence; no retry is hidden here.
    """
    if failure.success:
        raise ValueError("Adaptation requires an observed failure, not a solved task")
    if candidate.name in bank.names():
        raise ValueError("Consolidation must create a separate candidate")
    recipe = MotionRecipe(f"repeat_{candidate.name}", (candidate.name,))
    if any(r.name == recipe.name for r in library.recipes()):
        raise ValueError("Candidate recipe already exists")
    before = bank.fingerprint()
    controls = validate_policy(config, expert_policy(config), shadow_tasks, backend=bank.backend)
    if not all(result.success for result in controls):
        return {"status": "STOP_CONTROL_FAILED", "installed": False,
                "control_results": [r.to_dict(include_trace=False) for r in controls]}
    results = validate_policy(config, candidate, shadow_tasks, backend=bank.backend)
    accepted = all(result.success for result in results)
    if bank.fingerprint() != before:
        raise RuntimeError("Stable weights changed during shadow validation")
    if accepted:
        bank.add(candidate)
        library.add(recipe)
    return {"status": "ACCEPTED" if accepted else "REJECTED", "installed": accepted,
            "criterion": "all fixed development shadow tasks succeed, zero contacts",
            "shadow_results": [r.to_dict(include_trace=False) for r in results],
            "deterministic_control_successes": sum(r.success for r in controls),
            "stable_weights_unchanged": all(bank.fingerprint()[k] == v for k, v in before.items()),
            "temporary_release_authorized": accepted,
            "candidate_parameters": 21}
