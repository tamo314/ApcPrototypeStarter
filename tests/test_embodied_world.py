"""Physical dynamics, termination semantics and observation separation."""
from dataclasses import replace

import numpy as np
import pytest

from apc.embodied.control import EmbodiedPrimitiveCall, policy_features
from apc.embodied.world import (
    Box,
    EmbodiedEnv,
    NavigationTask,
    SeedSplit,
    WorldConfig,
    make_task,
    rotation,
    segment_hit,
)


@pytest.mark.parametrize("kwargs", [
    {"physics_dt": 0}, {"mass": 0}, {"radius": -1}, {"control_dt": 0.053},
    {"episode_seconds": 0.01}, {"max_force": float("nan")}, {"linear_drag": -1},
    {"angular_drag": float("inf")}, {"dwell_seconds": 0}, {"upper": (-5.9, -5.9, 0.1)},
])
def test_invalid_world_config(kwargs):
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)


@pytest.mark.parametrize("action", [[0]*3, [0]*7, [float("nan")]*6, [float("inf")]*6])
def test_actions_fail_closed(action):
    env = EmbodiedEnv()
    env.reset(make_task(1))
    with pytest.raises(ValueError):
        env.step(action)


def test_determinism_and_rng_isolation():
    np.random.seed(12)
    expected = np.random.random()
    np.random.seed(12)
    task = make_task(100, "spatial")
    assert np.random.random() == expected
    assert task == make_task(100, "spatial")
    a, b = EmbodiedEnv(), EmbodiedEnv()
    assert a.reset(task)[0] == b.reset(task)[0]
    for _ in range(20):
        assert a.step([0.1, 0.1, 0.1, 0, 0, 0]) == b.step([0.1, 0.1, 0.1, 0, 0, 0])


def test_body_frame_force_and_real_height_motion():
    q = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
    task = NavigationTask((0, 0, 2), ((3, 0, 4),), orientation=q)
    env = EmbodiedEnv()
    env.reset(task)
    obs, *_ = env.step([0, 0, 1, 0, 0, 0])
    assert obs.position[0] > 0 and abs(obs.position[2]-2) < 1e-10
    env.reset(replace(task, orientation=(1, 0, 0, 0)))
    obs, *_ = env.step([0, 0, 1, 0, 0, 0])
    assert obs.position[2] > 2


def test_inertia_mass_and_actuator_norm_bound():
    task = NavigationTask((0, 0, 2), ((3, 0, 4),))
    a, b = EmbodiedEnv(), EmbodiedEnv(WorldConfig(mass=2))
    a.reset(task)
    b.reset(task)
    x, *_ = a.step([100, 100, 100, 0, 0, 0])
    y, *_ = b.step([100, 100, 100, 0, 0, 0])
    assert np.linalg.norm(x.velocity) <= 4*0.05 + 1e-12
    assert np.linalg.norm(y.velocity) < np.linalg.norm(x.velocity)
    after, *_ = a.step([0]*6)
    assert np.linalg.norm(after.velocity) > 0
    assert np.linalg.norm(after.velocity) < np.linalg.norm(x.velocity)


def test_torque_changes_orientation_without_translating_body():
    env = EmbodiedEnv()
    task = NavigationTask((0, 0, 2), ((3, 0, 4),))
    env.reset(task)
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0.2, 0.1, 0.3])
    assert np.isclose(np.linalg.norm(obs.orientation), 1)
    assert np.linalg.norm(obs.angular_velocity) > 0
    assert np.allclose(obs.position, task.start)
    assert not np.allclose(rotation(obs.orientation), np.eye(3))


def test_swept_collision_cannot_tunnel_through_thin_wall():
    config = WorldConfig(max_force=1e6)
    obstacle = Box((0, -1, 1), (0.01, 1, 3))
    env = EmbodiedEnv(config)
    env.reset(NavigationTask((-2, 0, 2), ((2, 0, 2),), (obstacle,)))
    obs, _, terminated, truncated, info = env.step([1, 0, 0, 0, 0, 0])
    assert terminated and not truncated and not info["success"]
    assert info["reason"] == "collision"
    assert obs.position[0] <= -config.radius + 1e-8
    assert info["physics_steps"] == 1
    assert obs.time == config.physics_dt
    with pytest.raises(RuntimeError):
        env.step([0]*6)


def test_boundary_collision():
    env = EmbodiedEnv(WorldConfig(max_force=1e6))
    env.reset(NavigationTask((0, 0, 2), ((2, 0, 2),)))
    obs, _, terminated, _, info = env.step([1, 0, 0, 0, 0, 0])
    assert terminated and info["reason"] == "collision"
    assert obs.position[0] <= 6-0.18


def test_success_requires_dwell_and_timeout_is_not_success():
    config = WorldConfig(episode_seconds=0.2)
    env = EmbodiedEnv(config)
    env.reset(NavigationTask((0, 0, 2), ((0, 0, 2),)))
    assert not env.step([0]*6)[2]
    assert not env.step([0]*6)[2]
    _, _, terminated, truncated, info = env.step([0]*6)
    assert terminated and not truncated and info["success"]
    env.reset(NavigationTask((0, 0, 2), ((2, 0, 2),)))
    for _ in range(4):
        _, _, terminated, truncated, info = env.step([0]*6)
    assert not terminated and truncated and not info["success"]


def test_task_validation():
    with pytest.raises(ValueError):
        NavigationTask((0, 0, 2), ())
    with pytest.raises(ValueError):
        NavigationTask((0, 0, 2), ((2, 0, 2),), orientation=(0, 0, 0, 0))
    with pytest.raises(ValueError):
        EmbodiedEnv().reset(NavigationTask((0, 0, 0), ((2, 0, 2),)))
    with pytest.raises(ValueError):
        EmbodiedEnv().reset(NavigationTask((0, 0, 2), ((2, 0, 2),),
                                          (Box((-1, -1, 1), (1, 1, 3)),)))


def test_content_path_has_no_goal_or_task_identity():
    a, b = EmbodiedEnv(), EmbodiedEnv()
    start = (0, 0, 2)
    oa, _ = a.reset(NavigationTask(start, ((2, 0, 2),), name="secret-a"))
    ob, _ = b.reset(NavigationTask(start, ((0, 2, 4),), name="secret-b"))
    assert oa == ob
    assert np.array_equal(oa.content(), ob.content())
    assert "goal" not in oa.to_dict() and "name" not in oa.to_dict()
    ca = EmbodiedPrimitiveCall("move", {"target": (2, 0, 2)})
    cb = EmbodiedPrimitiveCall("move", {"target": (0, 2, 4)})
    assert not np.array_equal(policy_features(oa, ca.arguments), policy_features(ob, cb.arguments))


def test_timestep_refinement_is_close():
    task = NavigationTask((0, 0, 2), ((2, 0, 4),))
    a, b = EmbodiedEnv(), EmbodiedEnv(WorldConfig(physics_dt=0.005))
    a.reset(task)
    b.reset(task)
    for _ in range(20):
        oa, *_ = a.step([0.1, 0.1, 0.1, 0, 0, 0])
        ob, *_ = b.step([0.1, 0.1, 0.1, 0, 0, 0])
    assert np.linalg.norm(np.array(oa.position)-ob.position) < 0.003
    assert oa.time == ob.time


def test_split_overlap_and_invalid_seeds_rejected():
    with pytest.raises(ValueError):
        SeedSplit((1, 2), (2, 3), (4,))
    with pytest.raises(ValueError):
        SeedSplit((1, 1), (2,), (3,))
    with pytest.raises(ValueError):
        SeedSplit((-1,), (2,), (3,))


def test_segment_miss_parallel_and_normal():
    lo, hi = np.zeros(3), np.ones(3)
    assert segment_hit(np.array([-2., 2, 0.5]), np.array([2., 2, 0.5]), lo, hi) is None
    hit = segment_hit(np.array([-2., 0.5, 0.5]), np.array([2., 0.5, 0.5]), lo, hi)
    assert hit is not None and hit[0] == 0.5
    assert np.array_equal(hit[1], [-1, 0, 0])
