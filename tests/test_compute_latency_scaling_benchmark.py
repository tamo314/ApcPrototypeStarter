"""Focused CPU invariants for A2-C010's true end-to-end timing path."""

from __future__ import annotations

import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_shared_core_tokens
from apc.evaluation.compute_latency_scaling_benchmark import (
    _prepare_inference_batch,
    evaluate_end_to_end_at_size,
)
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus
from apc.primitives.router import Router, RouterConfig


class TinyCore:
    """Minimal task/content encoder with the public shared-core shape."""

    def __init__(self) -> None:
        self.tokens = build_shared_core_tokens(env_vocab_size=10, num_operations=16, arg_span=10)
        self.model = DecoderOnlyTransformer(
            TransformerConfig(
                vocab_size=self.tokens.model_vocab_size,
                max_seq_len=16,
                d_model=16,
                n_layer=1,
                n_head=2,
                d_ff=32,
                dropout=0.0,
            )
        )
        self.model.eval()
        self.device = torch.device("cpu")


def test_end_to_end_scaling_executes_full_sparse_and_dense_paths() -> None:
    """Sparse executes only the routed primitive; dense executes all stable modules."""
    torch.manual_seed(7)
    core = TinyCore()
    bank = PrimitiveBank()
    p0 = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SWAP_ENDS",
            d_model=16,
            d_operator=8,
            n_head=2,
            d_operator_ff=16,
            vocab_size=10,
            max_sequence_length=12,
        ),
        status=PrimitiveStatus.STABLE,
    )
    p1 = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SWAP_ENDS",
            d_model=16,
            d_operator=8,
            n_head=2,
            d_operator_ff=16,
            vocab_size=10,
            max_sequence_length=12,
        ),
        status=PrimitiveStatus.STABLE,
    )
    examples = generate_benchmark_examples(7, 3, operation="SWAP_ENDS", split="test")
    z_task = extract_task_representations(core, [examples[0]])[0]
    router = Router(RouterConfig(d_model=16, score_dim=16, top_k=1))
    router.add_primitive_key(p0.primitive_id)
    router.add_primitive_key(p1.primitive_id)
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(16))
        router.query_proj.bias.zero_()
        router.key_parameter(p0.primitive_id).copy_(z_task * 10.0)
        router.key_parameter(p1.primitive_id).copy_(-z_task * 10.0)
    router.eval()

    result = evaluate_end_to_end_at_size(
        seed=7,
        core=core,
        bank=bank,
        router=router,
        candidate_ids=[p0.primitive_id, p1.primitive_id],
        semantic_ids=[p0.primitive_id],
        distractor_ids=[p1.primitive_id],
        id_to_operation={p0.primitive_id: "SWAP_ENDS", p1.primitive_id: "SWAP_ENDS"},
        routing_examples_by_pid={p0.primitive_id: examples[:1]},
        profile_batch=_prepare_inference_batch(core, examples[:1]),
        warmup_steps=1,
        profile_steps=2,
    )

    assert result.routing_accuracy == 1.0
    assert result.sparse_selected_forward_calls == 1
    assert result.sparse_unselected_forward_calls == 0
    assert result.dense_forward_calls == 2
    assert result.dense_all_stable_primitives_executed is True
    assert result.hard_acceptance_passed is True
    assert result.flops_breakdown.total_sparse_flops < result.flops_breakdown.dense_baseline_flops
    assert result.router_overhead_flops_ratio > 0.0
    assert result.sparse_latency.p95_ms >= result.sparse_latency.median_ms
    assert result.dense_latency.throughput_examples_per_sec > 0.0
