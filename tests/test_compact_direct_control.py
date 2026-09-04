"""Tests for Task A1-B007X-004 Compact Direct-Learning Control."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.compact_direct_control import (
    CompactDirectControlConfig,
    compact_direct_control_config_from_dict,
    run_compact_direct_control,
)
from apc.evaluation.discovery_capacity_harness import build_tier_primitive


def test_config_defaults_and_serialization() -> None:
    cfg = CompactDirectControlConfig()
    assert cfg.candidate_tier == "T0_compact"
    assert cfg.max_candidate_params == 25000
    assert len(cfg.seeds) == 5
    assert len(cfg.novel_operations) == 3

    d = cfg.to_dict()
    restored = compact_direct_control_config_from_dict(d)
    assert restored.candidate_tier == cfg.candidate_tier
    assert restored.max_candidate_params == cfg.max_candidate_params
    assert restored.seeds == cfg.seeds


def test_candidate_parameter_bound() -> None:
    """Validate that candidate operator satisfies <= 25,000 parameter constraint."""
    prim = build_tier_primitive("T0_compact", "SWAP_PAIRS", vocab_size=10, d_model=192)
    params = prim.num_parameters()
    assert params == 17098
    assert params <= 25000


def test_compact_direct_control_smoke_run(tmp_path: Path) -> None:
    """Verify that compact direct control runs end-to-end on synthetic CPU setting."""
    cfg = CompactDirectControlConfig(
        seeds=(42,),
        novel_operations=("SWAP_PAIRS",),
        candidate_tier="T0_compact",
        train_steps=10,
        eval_interval=5,
        batch_size=8,
        num_train_examples=20,
        num_eval_examples=10,
        core_train_steps=10,
        bank_train_steps=10,
        device="cpu",
        save_checkpoints=True,
    )

    summary = run_compact_direct_control(cfg, output_dir=tmp_path)

    assert len(summary.all_runs) == 1
    run = summary.all_runs[0]
    assert run.operation == "SWAP_PAIRS"
    assert run.seed == 42
    assert run.parameters == 17098
    assert summary.all_compliant is True

    # Validate output artifacts
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "control_baseline.json").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "checkpoints" / "swap_pairs_seed42.pt").exists()

    summary_data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_data["all_compliant"] is True
    assert "SWAP_PAIRS" in summary_data["aggregated"]
    agg = summary_data["aggregated"]["SWAP_PAIRS"]
    assert agg["parameters"] == 17098
    assert agg["params_compliant"] is True

    baseline_data = json.loads(
        (tmp_path / "control_baseline.json").read_text(encoding="utf-8")
    )
    assert "SWAP_PAIRS" in baseline_data["baselines"]
    assert baseline_data["baselines"]["SWAP_PAIRS"]["parameters"] == 17098
