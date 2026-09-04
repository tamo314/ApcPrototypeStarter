"""Tests for Heterogeneous Compact Primitive Bank (Task A1-B001)."""

from __future__ import annotations

import torch

from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PointwisePrimitive,
    Primitive,
    PrimitiveConfig,
    PrimitiveStatus,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
)


def test_pointwise_primitive_compatibility() -> None:
    """Verify PointwisePrimitive is backward-compatible with legacy Primitive."""
    cfg = PrimitiveConfig(d_model=64, rank=8)
    p = PointwisePrimitive(0, cfg)
    assert isinstance(p, Primitive)
    assert p.num_parameters() == 64 * 8 * 2

    x = torch.randn(2, 5, 64)
    out = p(x)
    assert out.shape == (2, 5, 64)
    assert p.forward_call_count == 1

    p.reset_forward_call_count()
    assert p.forward_call_count == 0


def test_cross_position_primitive_parameter_scale() -> None:
    """Verify parameter counts for CrossPositionPrimitive match diagnostic targets."""
    for op in ("SELECT", "COUNT", "BIND"):
        cfg = CrossPositionPrimitiveConfig(operation=op, d_model=192, d_operator=32, n_head=4)
        p = CrossPositionPrimitive(0, cfg)
        num_p = p.num_parameters()
        # Diagnostic targets: 17,802 to 20,890 params
        assert 17000 <= num_p <= 22000, f"{op} param count {num_p} out of range"


def test_shift_relative_primitive_parameter_scale() -> None:
    """Verify parameter count for ShiftRelativePrimitive matches exact 18,282 target."""
    cfg = ShiftRelativePrimitiveConfig(d_model=192, d_operator=32, n_head=4)
    p = ShiftRelativePrimitive(0, cfg)
    # Diagnostic exact match from ADR-0045: 18,282 params
    assert p.num_parameters() == 18282


def test_cross_position_primitive_forward() -> None:
    """Verify forward pass, mask handling, and None-argument behavior."""
    cfg = CrossPositionPrimitiveConfig(operation="COUNT", d_model=192, d_operator=32, vocab_size=64)
    p = CrossPositionPrimitive(0, cfg)

    batch_size = 3
    lmax = 10
    content_features = torch.randn(batch_size, lmax, 192)
    content_lengths = [10, 8, 6]
    output_lengths = [1, 1, 1]
    arguments = [5, 12, 0]

    # Correct/argument arm
    logits = p(content_features, content_lengths, output_lengths, arguments)
    assert logits.shape == (batch_size, 1, 64)
    assert p.forward_call_count == 1

    # None arm
    p.reset_forward_call_count()
    logits_none = p(content_features, content_lengths, output_lengths, None)
    assert logits_none.shape == (batch_size, 1, 64)
    assert p.forward_call_count == 1


def test_shift_relative_primitive_forward() -> None:
    """Verify forward pass with modular relative position bias."""
    cfg = ShiftRelativePrimitiveConfig(d_model=192, d_operator=32, vocab_size=64)
    p = ShiftRelativePrimitive(0, cfg)

    batch_size = 2
    lmax = 8
    content_features = torch.randn(batch_size, lmax, 192)
    content_lengths = [8, 6]
    output_lengths = [8, 6]
    arguments = [2, 3]

    logits = p(content_features, content_lengths, output_lengths, arguments)
    assert logits.shape == (batch_size, 8, 64)
    assert p.forward_call_count == 1


def test_primitive_bank_heterogeneous_registration_and_sparse_execution() -> None:
    """Verify bank can register heterogeneous primitives and enforce strict sparse calls."""
    bank = PrimitiveBank()

    # Register heterogeneous primitives
    p0 = bank.new_pointwise_primitive(PrimitiveConfig(d_model=64, rank=4))
    p1 = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(operation="SELECT", d_model=64, d_operator=16, n_head=2)
    )
    p2 = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(d_model=64, d_operator=16, n_head=2)
    )

    assert len(bank) == 3
    assert bank.ids() == [0, 1, 2]
    assert bank.get(0) is p0
    assert bank.get(1) is p1
    assert bank.get(2) is p2

    bank.reset_all_forward_call_counts()
    assert bank.forward_call_counts() == {0: 0, 1: 0, 2: 0}

    # Execute ONLY primitive 1 (sparse call)
    content = torch.randn(2, 6, 64)
    _ = p1(content, [6, 6], [3, 3], [[0, 1, 2], [1, 2, 3]])

    # Verify strict zero-call for unselected primitives (0 and 2)
    counts = bank.forward_call_counts()
    assert counts[0] == 0, "Unselected primitive 0 must receive zero forward calls"
    assert counts[1] == 1, "Selected primitive 1 must receive exactly 1 forward call"
    assert counts[2] == 0, "Unselected primitive 2 must receive zero forward calls"


def test_primitive_bank_freeze_and_parameter_accounting() -> None:
    """Verify freeze methods and parameter accounting."""
    bank = PrimitiveBank()
    p0 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=64, rank=8), status=PrimitiveStatus.STABLE
    )
    p1 = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(d_model=64, d_operator=16, n_head=2),
        status=PrimitiveStatus.CANDIDATE,
    )

    tot_params = p0.num_parameters() + p1.num_parameters()
    assert bank.total_parameter_count() == tot_params
    assert bank.persistent_parameter_count() == p0.num_parameters()
    assert bank.active_parameter_count([0]) == p0.num_parameters()
    assert bank.active_parameter_count([1]) == p1.num_parameters()
    assert bank.active_parameter_count([0, 1]) == tot_params

    # Freeze stable
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    assert p0.is_frozen()
    assert not p1.is_frozen()
    assert bank.total_parameter_count(trainable_only=True) == p1.num_parameters()

    # Freeze all
    bank.freeze_all()
    assert p0.is_frozen()
    assert p1.is_frozen()
    assert bank.total_parameter_count(trainable_only=True) == 0
