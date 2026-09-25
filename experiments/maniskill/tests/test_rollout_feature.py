"""One complete feature test, run explicitly on a ManiSkill-capable host.

This tests collection, not task-solving performance. No small helper unit tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_rollout_to_saved_evidence(tmp_path):
    import sapien
    from apc_maniskill.runner import RunConfig, array, collect, json_write, make_env

    out = tmp_path / "feature-run"

    def physical_headless_env(config, output):
        env = make_env(config, output)
        try:
            # Reconfigure as well as initial creation: the compatibility fix must
            # survive scene reconstruction, with real collisions and gravity.
            env.reset(seed=0, options={"reconfigure": True})
            base = env.unwrapped
            assert not base.scene.can_render()
            for scene in base.scene.sub_scenes:
                assert not any(entity.find_component_by_type(sapien.render.RenderBodyComponent)
                               for entity in scene.entities)
            cube = base.cube._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            assert len(cube.collision_shapes) == 1
            table = base.table_scene.table._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            assert table.collision_shapes
            assert any(link.collision_shapes for link in base.agent.robot._objs[0].links)
            half_size = base.cube_half_size
            np.testing.assert_allclose(cube.mass, 1000 * (2 * half_size) ** 3, rtol=1e-5)
            base.cube.set_pose(sapien.Pose(p=[0.3, 0.3, 0.15]))
            for _ in range(20):
                env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
            z = float(array(base.cube.pose.p)[0, 2])
            force = array(base.scene.get_pairwise_contact_forces(base.cube, base.table_scene.table))
            assert abs(z - half_size) < 0.003  # falls and rests on the table
            assert np.linalg.norm(force) > 0
            json_write(output / "headless_physics.json", dict(
                cube_mass_kg=cube.mass, drop_start_z_m=0.15, final_z_m=z,
                table_contact_force=force.tolist(), diagnostic_steps=20,
            ))
            return env  # collect resets before recording each episode
        except BaseException:
            env.close()
            raise

    values = json.loads((WORKSPACE / "configs/pickcube_cpu.json").read_text())
    values.update(episodes=2, max_steps=3)
    collect(RunConfig(**values), out, env_factory=physical_headless_env)
    result = subprocess.run([
        sys.executable, "-m", "apc_maniskill", "summarize", str(out),
    ], check=True, capture_output=True, text=True, timeout=30)
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["episodes"] == 2
    assert 2 <= summary["steps"] <= 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["env_id"] == "PickCube-v1"
    assert manifest["provenance"]["versions"]["mani_skill"] is not None
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [0, 1]
    for row in rows:
        with np.load(out / row["trajectory"], allow_pickle=False) as trajectory:
            assert len(trajectory["observations"]) == len(trajectory["actions"]) + 1
            assert len(trajectory["actions"]) == row["steps"]
            assert np.isfinite(trajectory["observations"]).all()
            assert np.isfinite(trajectory["rewards"]).all()
            assert set(np.unique(trajectory["success"])) <= {-1, 0, 1}
    # No success-rate floor: even zero successes is a valid collection result.
