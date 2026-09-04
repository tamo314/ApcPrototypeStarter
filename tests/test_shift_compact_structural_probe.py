"""Unit tests for Phase A.1 diagnostic Task A1-R005E-S006 SHIFT compact
structural probe."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apc.evaluation.shift_compact_structural_probe import (
    ShiftRelativeCrossPositionOperator,
    ShiftStructuralProbeConfig,
    run_shift_compact_structural_probe,
    run_shift_compact_structural_probe_multi_seed,
    shift_structural_probe_config_from_dict,
)


def test_operator_parameter_count_primitive_scale() -> None:
    operator = ShiftRelativeCrossPositionOperator(
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
        arg_dim=16,
    )
    param_count = sum(p.numel() for p in operator.parameters())
    # Exactly 18,282 parameters: 18,154 (standard compact) + 128 (rel_pos_bias 32x4)
    assert param_count == 18282
    assert param_count < 25000


def test_operator_forward_shape_and_arms() -> None:
    operator = ShiftRelativeCrossPositionOperator(
        d_model=64,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
        arg_dim=8,
    )
    batch_size = 4
    lmax = 8
    features = torch.randn(batch_size, lmax, 64)
    c_lens = [6, 7, 8, 5]
    o_lens = [6, 7, 8, 5]

    # Correct arm with argument values
    args = [1, 2, 0, 3]
    out = operator(features, c_lens, o_lens, args)
    assert out.shape == (batch_size, 8, 10)
    assert not torch.isnan(out).any()

    # None arm
    out_none = operator(features, c_lens, o_lens, None)
    assert out_none.shape == (batch_size, 8, 10)
    assert not torch.isnan(out_none).any()


def test_config_validation_and_roundtrip() -> None:
    config = ShiftStructuralProbeConfig(seed=42, core_train_steps=100, operator_train_steps=50)
    d = config.to_dict()
    assert d["seed"] == 42
    assert d["core_train_steps"] == 100
    assert d["operator_train_steps"] == 50

    reconstructed = shift_structural_probe_config_from_dict(d)
    assert reconstructed.seed == 42
    assert reconstructed.core_train_steps == 100
    assert reconstructed.operator_train_steps == 50

    with pytest.raises(ValueError, match="divisible"):
        ShiftStructuralProbeConfig(d_operator=31, n_operator_head=4)


def test_tiny_single_seed_run(tmp_path: Path) -> None:
    tiny_cfg = ShiftStructuralProbeConfig(
        seed=0,
        vocab_size=10,
        sequence_length_range=(4, 6),
        model={
            "d_model": 32,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        core_train_steps=4,
        operator_train_steps=4,
        num_unseen_eval_groups=3,
        min_unseen_eval_examples=4,
    )
    report = run_shift_compact_structural_probe(tiny_cfg, seed_dir=tmp_path / "seed_0")
    assert report.steps_trained == 4
    assert report.task_blind_invariant_passed is True
    assert 0.0 <= report.correct_exact_match <= 1.0
    assert 0.0 <= report.correct_token_accuracy <= 1.0
    assert report.operator_param_count > 0

    # Saved shared encoder checkpoint verification
    assert (tmp_path / "seed_0" / "shared_encoder.pt").exists()


def test_tiny_multi_seed_run(tmp_path: Path) -> None:
    tiny_cfg = ShiftStructuralProbeConfig(
        seed=0,
        vocab_size=10,
        sequence_length_range=(4, 6),
        model={
            "d_model": 32,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        core_train_steps=4,
        operator_train_steps=4,
        num_unseen_eval_groups=3,
        min_unseen_eval_examples=4,
    )
    multi_report = run_shift_compact_structural_probe_multi_seed(
        tiny_cfg, seeds=(0, 1), run_dir=tmp_path
    )
    assert len(multi_report.per_seed) == 2
    assert multi_report.seeds == (0, 1)
    assert multi_report.meets_seed_policy is False  # 2 < 5
    assert multi_report.s002_shift_correct == 0.5193
    assert (tmp_path / "seed_0" / "report.json").exists()
    assert (tmp_path / "seed_1" / "report.json").exists()
