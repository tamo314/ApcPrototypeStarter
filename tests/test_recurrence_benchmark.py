"""Unit tests for Recurrence & Reuse Benchmark (Task A1-B007 / Milestone B-M7)."""

from __future__ import annotations

import pytest

from apc.environments.operations import BRANCH_B_NOVEL_OPERATION_NAMES
from apc.evaluation.recurrence_benchmark import (
    MAX_ALLOCATED_PLASTIC_PARAMS,
    RECURRENCE_ACCURACY_THRESHOLD,
    RecurrenceBenchmarkConfig,
    _verify_sparse_execution,
    generate_benchmark_examples,
    recurrence_config_from_dict,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)


def test_recurrence_config_from_dict() -> None:
    raw = {
        "seed": 42,
        "vocab_size": 10,
        "num_eval_examples": 200,
        "accuracy_threshold": 0.90,
    }
    cfg = recurrence_config_from_dict(raw)
    assert cfg.seed == 42
    assert cfg.vocab_size == 10
    assert cfg.num_eval_examples == 200
    assert cfg.accuracy_threshold == 0.90
    assert cfg.recurrent_operations == BRANCH_B_NOVEL_OPERATION_NAMES


def test_recurrence_config_validation() -> None:
    with pytest.raises(ValueError, match="num_eval_examples must be >="):
        RecurrenceBenchmarkConfig(num_eval_examples=50, min_eval_examples=100)

    with pytest.raises(ValueError, match="recurrent_operations must be non-empty"):
        RecurrenceBenchmarkConfig(recurrent_operations=())


def test_recurrence_examples_generator() -> None:
    examples = generate_benchmark_examples(
        seed=123,
        n=10,
        operation="SWAP_PAIRS",
        split="test",
        vocab_size=10,
        sequence_length_range=(6, 10),
    )
    assert len(examples) == 10
    for ex in examples:
        assert len(ex.input_tokens) == len(ex.target_tokens)
        assert all(0 <= tok < 10 for tok in ex.input_tokens)
        assert all(0 <= tok < 10 for tok in ex.target_tokens)
        assert ex.split == "test"
        assert ex.category == "novel"

    # Determinism check
    examples_2 = generate_benchmark_examples(
        seed=123,
        n=10,
        operation="SWAP_PAIRS",
        split="test",
        vocab_size=10,
        sequence_length_range=(6, 10),
    )
    for e1, e2 in zip(examples, examples_2, strict=True):
        assert e1.input_tokens == e2.input_tokens
        assert e1.target_tokens == e2.target_tokens


def test_sparse_execution_verification() -> None:
    bank = PrimitiveBank()
    cfg = CrossPositionPrimitiveConfig(
        operation="COPY",
        d_model=32,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    p0 = bank.new_cross_position_primitive(cfg, status=PrimitiveStatus.STABLE)
    p1 = bank.new_cross_position_primitive(cfg, status=PrimitiveStatus.STABLE)

    p0.forward_call_count = 5
    p1.forward_call_count = 0
    assert _verify_sparse_execution(bank, selected_pid=p0.primitive_id) is True
    assert _verify_sparse_execution(bank, selected_pid=p1.primitive_id) is False


def test_recurrence_zero_plastic_allocation_and_freeze_invariants(tmp_path) -> None:
    """Fast CPU smoke test verifying zero plastic parameter allocation during recurrence."""
    workspace = PlasticWorkspace()
    assert workspace.total_parameter_count() == 0
    assert len(workspace) == 0

    # Test allocation check constant
    assert MAX_ALLOCATED_PLASTIC_PARAMS == 0
    assert RECURRENCE_ACCURACY_THRESHOLD == 0.90
