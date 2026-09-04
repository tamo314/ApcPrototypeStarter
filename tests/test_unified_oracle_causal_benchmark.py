"""Tests for Task A1-B002 Unified Oracle Causal Benchmark."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    NATURAL_BASELINES,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    run_unified_oracle_causal_benchmark,
    unified_benchmark_config_from_dict,
)


def _tiny_config(**overrides: object) -> UnifiedBenchmarkConfig:
    base = UnifiedBenchmarkConfig(
        seed=0,
        vocab_size=10,
        sequence_length_range=(6, 8),
        group_size=3,
        model={
            "d_model": 32,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        parameterized_train_steps=2,
        parameter_free_train_steps=2,
        core_train_steps=4,
        num_unseen_eval_groups=2,
        num_unseen_eval_examples=8,
        min_unseen_eval_examples=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def test_unified_benchmark_config_validation() -> None:
    """Verify config validation bounds."""
    cfg = _tiny_config()
    assert cfg.vocab_size == 10
    assert cfg.d_operator % cfg.n_operator_head == 0

    with pytest.raises(ValueError, match="d_operator"):
        _tiny_config(d_operator=15, n_operator_head=2)

    with pytest.raises(ValueError, match="group_size"):
        _tiny_config(group_size=1)


def test_unified_benchmark_config_round_trip() -> None:
    """Verify config serialization round-trip."""
    cfg = _tiny_config()
    d = cfg.to_dict()
    restored = unified_benchmark_config_from_dict(d)
    assert restored.seed == cfg.seed
    assert restored.d_operator == cfg.d_operator
    assert restored.model == cfg.model


def test_heterogeneous_bank_all_8_operations() -> None:
    """Verify PrimitiveBank registers all 8 canonical operations."""
    cfg = _tiny_config()
    bank, op_to_id = _build_heterogeneous_bank(cfg)

    assert len(bank) == 8
    assert set(op_to_id.keys()) == set(ALL_CANONICAL_OPERATIONS)
    assert len(op_to_id) == 8

    for op in ALL_CANONICAL_OPERATIONS:
        pid = op_to_id[op]
        prim = bank.get(pid)
        assert prim.primitive_id == pid
        assert prim.num_parameters() < 30000


def test_natural_baselines_defined() -> None:
    """Verify natural baselines are defined for all 8 operations."""
    for op in ALL_CANONICAL_OPERATIONS:
        assert op in NATURAL_BASELINES
        assert 0.0 <= NATURAL_BASELINES[op] <= 0.50


def test_sparse_execution_in_bank() -> None:
    """Verify bank enforces zero forward calls on unselected primitives."""
    cfg = _tiny_config()
    bank, op_to_id = _build_heterogeneous_bank(cfg)
    bank.reset_all_forward_call_counts()

    # Call only COPY (pid = op_to_id["COPY"])
    copy_prim = bank.get(op_to_id["COPY"])
    content = torch.randn(2, 6, 32)
    _ = copy_prim(content, [6, 6], [6, 6])

    counts = bank.forward_call_counts()
    assert counts[op_to_id["COPY"]] == 1
    for op, pid in op_to_id.items():
        if op != "COPY":
            assert counts[pid] == 0, f"Unselected primitive {op} must have zero calls"


def test_end_to_end_tiny_run(tmp_path: Path) -> None:
    """Verify end-to-end benchmark executes and produces report."""
    cfg = _tiny_config()
    report = run_unified_oracle_causal_benchmark(cfg, seed_dir=tmp_path)

    assert report.seed == 0
    assert len(report.results_by_operation) == 8
    assert report.sparse_execution_passed is True
    assert report.task_blind_passed is True
    assert isinstance(report.wall_clock_seconds, float)

    d = report.to_dict()
    assert "results_by_operation" in d
    assert "mean_correct_exact_match" in d
