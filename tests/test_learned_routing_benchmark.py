"""Unit tests for Learned Routing & Full Closed Loop Benchmark (Task A1-B008 / Milestone B-M8)."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_shared_core_tokens
from apc.environments.task_spec import (
    default_argument_value_span,
    num_registered_operations,
)
from apc.evaluation.learned_routing_benchmark import (
    ALL_BENCHMARK_OPERATIONS,
    CLOSED_LOOP_EM_THRESHOLD,
    MAX_ALLOCATED_PLASTIC_PARAMS,
    ROUTING_ACCURACY_THRESHOLD,
    LearnedRoutingBenchmarkConfig,
    _verify_sparse_routing_execution,
    extract_task_representations,
    routing_config_from_dict,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus
from apc.primitives.router import Router, RouterConfig


def test_routing_config_from_dict() -> None:
    raw = {
        "seed": 42,
        "vocab_size": 10,
        "num_eval_examples": 200,
        "routing_threshold": 0.95,
        "accuracy_threshold": 0.90,
    }
    cfg = routing_config_from_dict(raw)
    assert cfg.seed == 42
    assert cfg.vocab_size == 10
    assert cfg.num_eval_examples == 200
    assert cfg.routing_threshold == 0.95
    assert cfg.accuracy_threshold == 0.90
    assert cfg.operations == ALL_BENCHMARK_OPERATIONS


def test_routing_config_validation() -> None:
    with pytest.raises(ValueError, match="num_eval_examples must be >="):
        LearnedRoutingBenchmarkConfig(num_eval_examples=50, min_eval_examples=100)

    with pytest.raises(ValueError, match="operations must be non-empty"):
        LearnedRoutingBenchmarkConfig(operations=())

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        LearnedRoutingBenchmarkConfig(top_k=0)


def test_extract_task_representations_shape_and_orthogonality() -> None:
    """Verify z_task extraction shape [N, d_model] and content-independence."""
    vocab_size = 10
    d_model = 32
    tokens = build_shared_core_tokens(
        vocab_size,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(vocab_size, (6, 10)),
    )
    model_config = TransformerConfig(
        vocab_size=tokens.model_vocab_size,
        d_model=d_model,
        n_layer=2,
        n_head=2,
        d_ff=64,
        max_seq_len=48,
    )
    model = DecoderOnlyTransformer(model_config)
    model.eval()

    core = MagicMock()
    core.tokens = tokens
    core.device = torch.device("cpu")
    core.model = model

    from apc.evaluation.recurrence_benchmark import generate_benchmark_examples

    examples = generate_benchmark_examples(
        seed=0,
        n=1,
        operation="COPY",
        split="test",
        vocab_size=vocab_size,
    )
    ex1 = examples[0]
    ex2 = dataclasses.replace(ex1, input_tokens=(9, 8, 7, 0, 1, 2))

    z_tasks = extract_task_representations(core, [ex1, ex2])
    assert z_tasks.shape == (2, d_model)

    # By construction, task-only encoding must produce exactly identical z_task for ex1 and ex2
    diff = (z_tasks[0] - z_tasks[1]).abs().max().item()
    assert diff == 0.0, f"Task representations must be completely content-blind; diff={diff}"


def test_router_calibration_convergence() -> None:
    """Verify that calibrating the router on synthetic task specs converges to high accuracy."""
    d_model = 32
    router = Router(RouterConfig(d_model=d_model, top_k=1, score_fn="dot"))
    for pid in [0, 1, 2]:
        router.add_primitive_key(pid)

    # Create dummy z_task vectors with distinct direction per pid
    g_0 = torch.randn(d_model)
    g_1 = torch.randn(d_model) + 5.0
    g_2 = torch.randn(d_model) - 5.0

    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        z = torch.stack([g_0, g_1, g_2], dim=0)
        out = router(z, [0, 1, 2])
        loss = torch.nn.functional.cross_entropy(
            out.probs, torch.tensor([0, 1, 2], dtype=torch.long)
        )
        loss.backward()
        optimizer.step()

    router.eval()
    with torch.no_grad():
        out = router(torch.stack([g_0, g_1, g_2], dim=0), [0, 1, 2])
        preds = out.selected_ids[:, 0].tolist()
        assert preds == [0, 1, 2], f"Router failed to separate distinct vectors: {preds}"


def test_sparse_routing_execution_verification() -> None:
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

    p0.forward_call_count = 3
    p1.forward_call_count = 0

    assert _verify_sparse_routing_execution(bank, selected_pids={p0.primitive_id}) is True
    assert _verify_sparse_routing_execution(bank, selected_pids={p1.primitive_id}) is False


def test_routing_threshold_constants() -> None:
    assert ROUTING_ACCURACY_THRESHOLD == 0.95
    assert CLOSED_LOOP_EM_THRESHOLD == 0.90
    assert MAX_ALLOCATED_PLASTIC_PARAMS == 0
    assert len(ALL_BENCHMARK_OPERATIONS) == 10
