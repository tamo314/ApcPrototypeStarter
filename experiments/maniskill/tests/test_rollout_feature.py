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
    out = tmp_path / "feature-run"
    subprocess.run([
        sys.executable, "-m", "apc_maniskill", "rollout",
        "--config", str(WORKSPACE / "configs/pickcube_cpu.json"),
        "--out", str(out), "--episodes", "2", "--max-steps", "3",
    ], check=True, timeout=180)
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
