"""Small, deterministic 6-DoF sphere simulator for the embodied APC control harness.

SI units; body-frame force/torque; semi-implicit Euler; frictionless conservative
swept sphere/AABB contacts. This is a reference simulator, not a validated robotics
engine. Buoyancy cancels gravity: the body can navigate all three spatial axes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Vec3 = tuple[float, ...]


def vector(value: Any, size: int = 3) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    return result.copy()


def rotation(quaternion: Any) -> Array:
    q = vector(quaternion, 4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion cannot be zero")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


@dataclass(frozen=True)
class Box:
    lower: Vec3
    upper: Vec3

    def __post_init__(self) -> None:
        lo, hi = vector(self.lower), vector(self.upper)
        if np.any(lo >= hi):
            raise ValueError("Box lower bounds must be below upper bounds")
        object.__setattr__(self, "lower", tuple(float(x) for x in lo))
        object.__setattr__(self, "upper", tuple(float(x) for x in hi))

    def contains(self, position: Any, margin: float = 0.0) -> bool:
        p = vector(position)
        return bool(np.all(p >= np.array(self.lower) - margin)
                    and np.all(p <= np.array(self.upper) + margin))


@dataclass(frozen=True)
class WorldConfig:
    physics_dt: float = 0.01
    control_dt: float = 0.05
    episode_seconds: float = 40.0
    option_seconds: float = 15.0
    mass: float = 1.0
    radius: float = 0.18
    max_force: float = 4.0
    max_torque: float = 0.1
    linear_drag: float = 0.3
    angular_drag: float = 0.1
    tolerance: float = 0.16
    speed_tolerance: float = 0.2
    dwell_seconds: float = 0.15
    lower: Vec3 = (-6.0, -6.0, 0.0)
    upper: Vec3 = (6.0, 6.0, 6.0)

    def __post_init__(self) -> None:
        positive = (self.physics_dt, self.control_dt, self.episode_seconds,
                    self.option_seconds, self.mass, self.radius, self.max_force,
                    self.max_torque, self.tolerance, self.speed_tolerance, self.dwell_seconds)
        if any(not isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Timesteps, budgets and physical scales must be finite and positive")
        if any(not isfinite(x) or x < 0 for x in (self.linear_drag, self.angular_drag)):
            raise ValueError("Drag must be finite and nonnegative")
        ratio = self.control_dt / self.physics_dt
        if abs(ratio - round(ratio)) > 1e-8 or ratio < 1:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        for duration in (self.episode_seconds, self.option_seconds):
            ratio = duration / self.control_dt
            if abs(ratio - round(ratio)) > 1e-8:
                raise ValueError("Time budgets must be integer multiples of control_dt")
        bounds = Box(self.lower, self.upper)
        object.__setattr__(self, "lower", bounds.lower)
        object.__setattr__(self, "upper", bounds.upper)
        if np.any(np.array(bounds.upper) - np.array(bounds.lower) <= 2*self.radius):
            raise ValueError("World is smaller than the body")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationTask:
    """Explicit, observable goal specification, separate from physical observation."""
    start: Vec3
    goals: tuple[Vec3, ...]
    obstacles: tuple[Box, ...] = ()
    name: str = "navigation"
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        start = tuple(float(x) for x in vector(self.start))
        goals = tuple(tuple(float(x) for x in vector(g)) for g in self.goals)
        if not goals:
            raise ValueError("At least one goal is required")
        q = vector(self.orientation, 4)
        if float(np.linalg.norm(q)) < 1e-12:
            raise ValueError("Quaternion cannot be zero")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "orientation", tuple(float(x) for x in q/np.linalg.norm(q)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    """Task-blind, privileged physical state (not an image or a task identity)."""
    position: Vec3
    velocity: Vec3
    orientation: tuple[float, ...]
    angular_velocity: Vec3
    time: float
    obstacles: tuple[Box, ...]

    def content(self) -> Array:
        """Stable content path: no target, operation, seed or task identifier."""
        return np.array((*self.position, *self.velocity, *self.orientation,
                         *self.angular_velocity), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_hit(start: Array, end: Array, lower: Array, upper: Array
                ) -> tuple[float, Array] | None:
    """Slab intersection, returning first contact fraction and outward normal."""
    direction = end - start
    near, far = 0.0, 1.0
    normal = np.zeros(3)
    for axis in range(3):
        if abs(direction[axis]) < 1e-14:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        a = float((lower[axis] - start[axis]) / direction[axis])
        b = float((upper[axis] - start[axis]) / direction[axis])
        sign = -1.0
        if a > b:
            a, b = b, a
            sign = 1.0
        if a >= near:
            near = a
            normal = np.zeros(3)
            normal[axis] = sign
        far = min(far, b)
        if near > far:
            return None
    if far < 0 or near > 1 or not np.any(normal):
        return None
    return max(0.0, near), normal


class EmbodiedEnv:
    """reset/step interface with terminated and truncated separated.

    A collision terminates an episode, preventing contact exploits. A timeout is
    a truncation, not a success. step after either condition raises until reset.
    """
    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        self.task: NavigationTask | None = None
        self._done = True
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._q = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.zeros(3)
        self._steps = 0
        self._physics_steps = 0
        self._goal = 0
        self._dwell = 0
        self._collisions = 0

    def reset(self, task: NavigationTask, *, seed: int = 0
              ) -> tuple[Observation, dict[str, Any]]:
        # Dynamics have no random state. seed is an explicit compatibility input;
        # task generation owns its local RNG and determines all randomization.
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        c = self.config
        lo, hi = np.array(c.lower) + c.radius, np.array(c.upper) - c.radius
        for point in (task.start, *task.goals):
            p = vector(point)
            if np.any(p <= lo) or np.any(p >= hi):
                raise ValueError("Start and goals must be strictly inside body-clear bounds")
            if any(box.contains(p, c.radius) for box in task.obstacles):
                raise ValueError("Start or goal intersects an inflated obstacle")
        for box in task.obstacles:
            if np.any(np.array(box.lower) < c.lower) or np.any(np.array(box.upper) > c.upper):
                raise ValueError("Obstacles must be inside world bounds")
        self.task = task
        self._position = vector(task.start)
        self._velocity = np.zeros(3)
        self._q = vector(task.orientation, 4)
        self._omega = np.zeros(3)
        self._steps = self._physics_steps = self._goal = self._dwell = self._collisions = 0
        self._done = False
        return self.observe(), self._info("running")

    def observe(self) -> Observation:
        if self.task is None:
            raise RuntimeError("reset must be called first")
        return Observation(tuple(float(x) for x in self._position),
                           tuple(float(x) for x in self._velocity),
                           tuple(float(x) for x in self._q),
                           tuple(float(x) for x in self._omega),
                           self._physics_steps*self.config.physics_dt, self.task.obstacles)

    def _info(self, reason: str) -> dict[str, Any]:
        return {"success": reason == "success", "reason": reason,
                "goals_reached": self._goal, "collisions": self._collisions,
                "physics_steps": self._physics_steps}

    def _advance(self, action: Array) -> bool:
        c = self.config
        dt = c.physics_dt
        force = rotation(self._q) @ action[:3] * c.max_force
        self._velocity += dt * (force/c.mass - c.linear_drag*self._velocity/c.mass)
        inertia = 0.4*c.mass*c.radius**2
        self._omega += dt*(action[3:]*c.max_torque - c.angular_drag*self._omega)/inertia
        w, x, y, z = self._q
        wx, wy, wz = self._omega
        self._q += 0.5*dt*np.array([-x*wx-y*wy-z*wz, w*wx+y*wz-z*wy,
                                    w*wy+z*wx-x*wz, w*wz+x*wy-y*wx])
        self._q /= np.linalg.norm(self._q)
        proposed = self._position + dt*self._velocity
        first = 1.0
        normal: Array | None = None
        assert self.task is not None
        for box in self.task.obstacles:
            hit = segment_hit(self._position, proposed, np.array(box.lower)-c.radius,
                              np.array(box.upper)+c.radius)
            if hit is not None and hit[0] <= first:
                first, normal = hit
        lo, hi = np.array(c.lower)+c.radius, np.array(c.upper)-c.radius
        for axis in range(3):
            for bound, sign in ((lo[axis], 1.0), (hi[axis], -1.0)):
                if (sign > 0 and proposed[axis] < bound) or (sign < 0 and proposed[axis] > bound):
                    fraction = float((bound-self._position[axis]) /
                                     (proposed[axis]-self._position[axis]))
                    if fraction <= first:
                        first = max(0.0, fraction)
                        normal = np.zeros(3)
                        normal[axis] = sign
        self._position += max(0.0, first-1e-9 if normal is not None else first) * (
            proposed-self._position)
        if normal is not None:
            self._velocity -= min(0.0, float(self._velocity @ normal))*normal
            return True
        return False

    def step(self, action: Any) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._done or self.task is None:
            raise RuntimeError("Episode ended or not initialized; call reset")
        u = vector(action, 6)
        # Norm bounds preserve rotation-invariant force limits, unlike per-axis clipping.
        for part in (u[:3], u[3:]):
            part /= max(1.0, float(np.linalg.norm(part)))
        previous = float(np.linalg.norm(np.array(self.task.goals[self._goal])-self._position))
        contact = False
        for _ in range(round(self.config.control_dt/self.config.physics_dt)):
            contact = self._advance(u) or contact
            self._physics_steps += 1
            if contact:
                break
        self._steps += 1
        if contact:
            self._collisions += 1
        distance = float(np.linalg.norm(np.array(self.task.goals[self._goal])-self._position))
        slow = float(np.linalg.norm(self._velocity)) <= self.config.speed_tolerance
        self._dwell = self._dwell + 1 if distance <= self.config.tolerance and slow else 0
        if self._dwell >= ceil(self.config.dwell_seconds/self.config.control_dt - 1e-10):
            self._goal += 1
            self._dwell = 0
        success = self._goal == len(self.task.goals) and not contact
        terminated = contact or success
        truncated = not terminated and self._steps >= round(
            self.config.episode_seconds/self.config.control_dt)
        self._done = terminated or truncated
        reason = ("collision" if contact else "success" if success
                  else "timeout" if truncated else "running")
        bonus = 1.0 if success else -1.0 if contact else 0.0
        reward = previous-distance - 0.001*float(u @ u) + bonus
        return self.observe(), reward, terminated, truncated, self._info(reason)


@dataclass(frozen=True)
class SeedSplit:
    train: tuple[int, ...] = field(default_factory=lambda: tuple(range(10, 18)))
    validation: tuple[int, ...] = field(default_factory=lambda: tuple(range(1000, 1004)))
    test: tuple[int, ...] = field(default_factory=lambda: tuple(range(2000, 2008)))

    def __post_init__(self) -> None:
        groups = [self.train, self.validation, self.test]
        for group in groups:
            if not group or len(set(group)) != len(group):
                raise ValueError("Splits must be nonempty and internally unique")
            if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in group):
                raise ValueError("Seeds must be nonnegative integers")
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError("Train, validation and test seeds must be disjoint")


def make_task(seed: int, family: str = "spatial") -> NavigationTask:
    """Four geometric families; all randomness is local and deterministic.

    The detour family specifies intermediate waypoints explicitly: route selection
    is a structured-task control, NOT a learned planner or autonomous path search.
    """
    if family not in {"planar", "spatial", "route", "detour"}:
        raise ValueError(f"Unknown task family: {family}")
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-np.pi, np.pi))
    q = (float(np.cos(yaw/2)), 0.0, 0.0, float(np.sin(yaw/2)))
    start: Vec3 = (-3.0, float(rng.uniform(-1, 1)), 1.2)
    end: Vec3 = (float(rng.uniform(1.5, 3.5)), float(rng.uniform(-2, 2)),
                 1.2 if family == "planar" else float(rng.uniform(3.2, 4.6)))
    goals = (end,)
    obstacles: tuple[Box, ...] = ()
    if family == "route":
        goals = ((-1.5, -2.0, 3.5), (1.0, 1.5, 4.5), end)
    if family == "detour":
        obstacles = (Box((-0.6, -3.0, 0.2), (0.6, 3.0, 2.8)),)
        goals = ((-2.0, start[1], 4.0), (2.0, end[1], 4.0), end)
    return NavigationTask(start, goals, obstacles, f"{family}:{seed}", q)
