from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apc.core.train import DataConfig, OptimConfig, SmokeTrainConfig, run_smoke_training


def _tiny_config(run_dir_name: str, **optim_overrides: object) -> SmokeTrainConfig:
    optim_kwargs: dict[str, object] = {
        "steps": 150,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "log_every": 25,
        "device": "cpu",
        "amp": False,
    }
    optim_kwargs.update(optim_overrides)
    return SmokeTrainConfig(
        run_name=run_dir_name,
        seed=0,
        output_dir="runs",
        data=DataConfig(
            vocab_size=10,
            sequence_length_range=(4, 6),
            max_depth=1,
            novel_composition_fraction=0.3,
            num_examples=4,
        ),
        model={
            "max_seq_len": 32,
            "d_model": 32,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 64,
            "dropout": 0.0,
        },
        optim=OptimConfig(**optim_kwargs),  # type: ignore[arg-type]
    )


def test_smoke_training_overfits_tiny_dataset(tmp_path: Path) -> None:
    """Task 003 acceptance: smoke training overfits a tiny deterministic dataset."""
    config = _tiny_config("smoke")
    run_dir = tmp_path / "smoke_run"

    summary = run_smoke_training(config, run_dir)

    assert summary["train_exact_match"] == 1.0
    assert summary["final_loss"] < 0.01
    assert summary["num_examples"] == 4


def test_smoke_training_reload_reproduces_outputs(tmp_path: Path) -> None:
    """Task 003 acceptance: reload reproduces outputs."""
    config = _tiny_config("smoke_reload")
    run_dir = tmp_path / "smoke_reload_run"

    summary = run_smoke_training(config, run_dir)

    assert summary["reload_matches_outputs"] is True
    assert summary["reload_exact_match"] == summary["train_exact_match"]


def test_smoke_training_writes_run_artifacts(tmp_path: Path) -> None:
    config = _tiny_config("artifacts")
    run_dir = tmp_path / "artifacts_run"

    run_smoke_training(config, run_dir)

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "system.json").is_file()
    assert (run_dir / "checkpoint" / "model.pt").is_file()

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.is_file()
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    last_entry = json.loads(lines[-1])
    assert last_entry["step"] == config.optim.steps


def test_smoke_training_same_seed_is_reproducible(tmp_path: Path) -> None:
    config_a = _tiny_config("repro_a")
    config_b = _tiny_config("repro_b")

    summary_a = run_smoke_training(config_a, tmp_path / "run_a")
    summary_b = run_smoke_training(config_b, tmp_path / "run_b")

    assert summary_a["final_loss"] == pytest.approx(summary_b["final_loss"])
    assert summary_a["train_exact_match"] == summary_b["train_exact_match"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_smoke_training_runs_with_cuda_amp(tmp_path: Path) -> None:
    """Mixed precision must be usable but stays confined to the training loop
    (see `apc.core.train.run_smoke_training`), never the model definition."""
    config = _tiny_config("cuda_amp", steps=5, device="cuda", amp=True)
    run_dir = tmp_path / "cuda_amp_run"

    summary = run_smoke_training(config, run_dir)

    assert summary["steps"] == 5
