"""Goal-conditioned learned options, sparse execution and parameter-free recipes.

Only a selected primitive may combine the task-blind observation with its target
argument. The actuator decoder clips force/torque in the environment; it has no
access to a goal. No teacher is consulted by an installed policy at inference.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from types import MappingProxyType
from typing import Any

import numpy as np

from apc.embodied.world import (
    Array,
    EmbodiedEnv,
    NavigationTask,
    Observation,
    WorldConfig,
    rotation,
    vector,
)


@dataclass(frozen=True)
class EmbodiedPrimitiveCall:
    """Continuous-domain equivalent of PrimitiveCall, without changing token IDs."""
    operation: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a nonempty family identifier")
        if set(self.arguments) != {"target"}:
            raise ValueError("An embodied call requires exactly the target argument")
        target = tuple(float(x) for x in vector(self.arguments["target"]))
        object.__setattr__(self, "arguments", MappingProxyType({"target": target}))

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "arguments": dict(self.arguments)}


def policy_features(observation: Observation, arguments: Mapping[str, Any]) -> Array:
    """Argument binding happens INSIDE the selected policy, not the content core."""
    target = vector(arguments["target"])
    body_from_world = rotation(observation.orientation).T
    error = body_from_world @ (target - np.array(observation.position))
    velocity = body_from_world @ np.array(observation.velocity)
    return np.concatenate((error, velocity, np.ones(1)))


class LinearMotionPrimitive:
    """21 fitted parameters implementing a reusable body-frame feedback policy.

    This intentionally small linear policy is a control/learning baseline, not a
    claim that unrestricted neural skill discovery or obstacle avoidance is solved.
    """
    def __init__(self, name: str, weights: Any) -> None:
        EmbodiedPrimitiveCall(name, {"target": (0, 0, 0)})
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.shape != (3, 7) or not np.isfinite(matrix).all():
            raise ValueError("Policy weights must be finite with shape (3, 7)")
        self._name = name
        self._weights = matrix.copy()
        self._weights.flags.writeable = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameter_count(self) -> int:
        return 21

    def weights(self) -> Array:
        return self._weights.copy()

    def act(self, observation: Observation, call: EmbodiedPrimitiveCall) -> Array:
        if call.operation != self.name:
            raise ValueError("Call family does not match selected primitive")
        action = self._weights @ policy_features(observation, call.arguments)
        return np.concatenate((action, np.zeros(3)))

    def fingerprint(self) -> str:
        return hashlib.sha256(self._weights.astype("<f8").tobytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": self._weights.tolist(),
                "sha256": self.fingerprint()}


class MotionBank:
    """Domain adapter with either the real APC PrimitiveBank or a NumPy reference.

    The default uses APC's PrimitiveBase/PrimitiveBank for registration, frozen
    status, sparse forward counters, usage and persistent parameter accounting.
    The reference backend is for lightweight simulator tests, not a replacement
    for the repository's neural bank. Both execute identical learned weights.
    """
    def __init__(self, backend: str = "apc") -> None:
        if backend not in {"apc", "numpy"}:
            raise ValueError("backend must be apc or numpy")
        self.backend = backend
        self._policies: dict[str, LinearMotionPrimitive | None] = {}
        self.calls: dict[str, int] = {}
        self._native: Any = None
        if backend == "apc":
            from apc.embodied.apc_bridge import NativeMotionBank
            self._native = NativeMotionBank()

    def add(self, policy: LinearMotionPrimitive) -> None:
        if policy.name in self._policies:
            raise ValueError(f"Duplicate primitive: {policy.name}")
        snapshot = LinearMotionPrimitive(policy.name, policy.weights())
        if self._native is not None:
            self._native.add(snapshot)
        self._policies[policy.name] = None if self._native is not None else snapshot
        self.calls[policy.name] = 0

    def names(self) -> tuple[str, ...]:
        return tuple(self._policies)

    def get(self, name: str) -> LinearMotionPrimitive:
        if self._native is not None:
            return self._native.policy(name)
        policy = self._policies[name]
        assert policy is not None
        return policy

    def act(self, observation: Observation, call: EmbodiedPrimitiveCall) -> Array:
        if call.operation not in self._policies:
            raise KeyError(f"Unknown primitive: {call.operation}")
        self.calls[call.operation] += 1
        if self._native is not None:
            return self._native.act(observation, call)
        return self.get(call.operation).act(observation, call)

    @property
    def resident_parameters(self) -> int:
        if self._native is not None:
            return int(self._native.resident_parameters)
        return sum(self.get(name).parameter_count for name in self._policies)

    def fingerprint(self) -> dict[str, str]:
        return {name: self.get(name).fingerprint() for name in self._policies}

    def records(self) -> list[dict[str, Any]]:
        return [self.get(name).to_dict() for name in self._policies]


@dataclass(frozen=True)
class MotionRecipe:
    """A recipe stores operation IDs, never new weights or memorized coordinates.

    One operation repeats over any number of task waypoints. Longer recipes bind
    one operation to each corresponding waypoint and require an exact length.
    """
    name: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.operations or any(not x for x in self.operations):
            raise ValueError("A recipe needs a name and at least one operation")
        object.__setattr__(self, "operations", tuple(self.operations))

    def bind(self, task: NavigationTask) -> tuple[EmbodiedPrimitiveCall, ...]:
        operations = self.operations
        if len(operations) == 1:
            operations = operations * len(task.goals)
        if len(operations) != len(task.goals):
            raise ValueError("Recipe length does not match task waypoints")
        return tuple(EmbodiedPrimitiveCall(op, {"target": target})
                     for op, target in zip(operations, task.goals, strict=True))


class MotionCompositionLibrary:
    """Persistent weight-free recipes, kept separate from MotionBank."""
    def __init__(self) -> None:
        self._recipes: dict[str, MotionRecipe] = {}

    def add(self, recipe: MotionRecipe) -> None:
        if recipe.name in self._recipes:
            raise ValueError(f"Duplicate recipe: {recipe.name}")
        self._recipes[recipe.name] = recipe

    def recipes(self) -> tuple[MotionRecipe, ...]:
        # Most recently validated recipe first. This is a declared deterministic
        # selector, not a learned novelty or routing mechanism.
        return tuple(reversed(tuple(self._recipes.values())))

    def records(self) -> list[dict[str, Any]]:
        return [{"name": r.name, "operations": list(r.operations)}
                for r in self._recipes.values()]


@dataclass
class EpisodeResult:
    success: bool
    reason: str
    control_steps: int
    simulated_seconds: float
    collisions: int
    goals_reached: int
    primitive_calls: dict[str, int]
    max_active_parameters: int
    path_length: float
    effort: float
    task: dict[str, Any]
    trace: list[dict[str, Any]]

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        result = dict(vars(self))
        if not include_trace:
            del result["trace"]
        return result


def run_episode(config: WorldConfig, task: NavigationTask, bank: MotionBank,
                calls: Sequence[EmbodiedPrimitiveCall], *, control: str = "correct",
                record_trace: bool = False) -> EpisodeResult:
    """Temporally compose options; only the selected module executes per tick.

    Negative controls cannot change the evaluator's original goals. Wrong arguments
    target the starting point; None emits zero actuators without executing a policy.
    """
    if not calls:
        raise ValueError("At least one call is required")
    if control not in {"correct", "none", "wrong_arguments"}:
        raise ValueError("Unknown causal control")
    for call in calls:
        if call.operation not in bank.names():
            raise KeyError(call.operation)
    env = EmbodiedEnv(config)
    obs, info = env.reset(task)
    before = dict(bank.calls)
    trace: list[dict[str, Any]] = []
    if record_trace:
        trace.append({"t": obs.time, "position": obs.position,
                      "orientation": obs.orientation, "option": None, "action": [0.0]*6})
    index = option_ticks = dwell = steps = 0
    path = effort = 0.0
    max_active = 0
    reason = "recipe_exhausted"
    while index < len(calls):
        call = calls[index]
        if control == "wrong_arguments":
            call = EmbodiedPrimitiveCall(call.operation, {"target": task.start})
        if control == "none":
            action = np.zeros(6)
        else:
            action = bank.act(obs, call)
            max_active = max(max_active, bank.get(call.operation).parameter_count)
        previous = np.array(obs.position)
        previous_time = obs.time
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        option_ticks += 1
        path += float(np.linalg.norm(np.array(obs.position)-previous))
        # Actual normalized actuator effort, not unbounded pre-clipping commands.
        applied = action.copy()
        for part in (applied[:3], applied[3:]):
            part /= max(1.0, float(np.linalg.norm(part)))
        effort += float(applied @ applied)*(obs.time-previous_time)
        if record_trace:
            trace.append({"t": obs.time, "position": obs.position,
                          "orientation": obs.orientation, "option": call.operation,
                          "action": applied.tolist()})
        if terminated or truncated:
            reason = str(info["reason"])
            break
        distance = np.linalg.norm(np.array(obs.position)-call.arguments["target"])
        reached = distance <= config.tolerance
        slow = np.linalg.norm(obs.velocity) <= config.speed_tolerance
        dwell = dwell+1 if reached and slow else 0
        if dwell >= ceil(config.dwell_seconds/config.control_dt - 1e-10):
            index += 1
            option_ticks = dwell = 0
        elif option_ticks >= round(config.option_seconds/config.control_dt):
            reason = "option_timeout"
            break
    return EpisodeResult(bool(info["success"]), reason, steps, obs.time,
                         int(info["collisions"]), int(info["goals_reached"]),
                         {name: bank.calls[name]-before[name] for name in bank.names()},
                         max_active, path, effort, task.to_dict(), trace)


def try_library(config: WorldConfig, task: NavigationTask, bank: MotionBank,
                library: MotionCompositionLibrary, *, record_trace: bool = False
                ) -> list[EpisodeResult]:
    """Try installed recipes before adaptation. Every failed/reset attempt is logged."""
    attempts: list[EpisodeResult] = []
    for recipe in library.recipes():
        if len(recipe.operations) not in (1, len(task.goals)):
            continue
        result = run_episode(config, task, bank, recipe.bind(task), record_trace=record_trace)
        attempts.append(result)
        if result.success:
            break
    return attempts


def task_fingerprint(task: NavigationTask) -> str:
    spec = task.to_dict()
    spec.pop("name")
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
