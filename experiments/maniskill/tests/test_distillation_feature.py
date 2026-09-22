"""Real observations -> frozen teacher labels -> separate compact checkpoint -> rollout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from apc_maniskill.bc import BCPolicy, load_demonstrations, sha256

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_distill_and_run_without_teacher(tmp_path):
    demo, teacher, compact, rollout = [tmp_path / n for n in ("demo", "teacher", "compact", "rollout")]
    def cli(*args):
        subprocess.run([sys.executable, "-m", "apc_maniskill", *map(str, args)], check=True, timeout=180)

    cli("rollout", "--config", WORKSPACE / "configs/fetch_reach_goal.json",
        "--out", demo, "--episodes", 2, "--seed", 30, "--max-steps", 70, "--post-success-steps", 10)
    cli("train-bc", "--demo-run", demo, "--out", teacher, "--updates", 40)
    teacher_hash = sha256(teacher / "policy.pt")
    cli("distill", "--teacher-checkpoint", teacher / "policy.pt", "--state-run", demo,
        "--out", compact, "--updates", 40, "--hidden-width", 8)
    assert sha256(teacher / "policy.pt") == sha256(compact / "teacher_policy.pt") == teacher_hash
    report = json.loads((compact / "training.json").read_text())
    assert report["status"] == "completed" and report["algorithm"] == "distillation"
    assert report["completed_updates"] == 40
    assert report["parameter_count"] == 138 < report["teacher_details"]["parameter_count"] == 1314
    assert report["parameter_bytes"] == 138 * 4
    x, _, source = load_demonstrations(demo)
    teacher_policy = BCPolicy(teacher / "policy.pt", source["task"], source["control_freq"])
    candidate = BCPolicy(compact / "policy.pt", source["task"], source["control_freq"])
    with torch.inference_mode():
        t = teacher_policy.model((torch.from_numpy(x) - teacher_policy.mean) / teacher_policy.scale)
        c = candidate.model((torch.from_numpy(x) - candidate.mean) / candidate.scale)
        np.testing.assert_allclose((c - t).square().mean().item(), report["final_train_mse"], atol=1e-7)
    # Inference in a NEW process must need neither teacher file nor teacher model.
    (teacher / "policy.pt").rename(teacher / "policy.offline.pt")
    (compact / "teacher_policy.pt").rename(compact / "teacher.offline.pt")
    cli("rollout", "--config", WORKSPACE / "configs/fetch_reach_bc.json", "--out", rollout,
        "--checkpoint", compact / "policy.pt", "--episodes", 2, "--seed", 50, "--max-steps", 12)
    manifest = json.loads((rollout / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["policy_details"]["parameter_count"] == 138
    assert sha256(rollout / "policy.pt") == report["checkpoint_sha256"]
    for path in rollout.glob("episode_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
            assert len(data["observations"]) == len(data["actions"]) + 1
            np.testing.assert_allclose(candidate.predict(data["observations"][:-1, 0]),
                                       data["actions"][:, 11:13], atol=1e-6)
