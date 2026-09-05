"""Unit tests for Phase A.2 Compute Accounting & Execution Profiling (Task A2-C002).

Validates that tiny deterministic tests can rederive all accounting without learned models.
"""

from __future__ import annotations

import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.evaluation.compute_accounting import (
    compute_flops_breakdown,
    count_system_parameters,
    estimate_attention_flops,
    estimate_linear_flops,
    estimate_pointwise_primitive_flops,
    estimate_router_flops,
    execute_dense_primitive_baseline,
    profile_execution,
    verify_sparse_execution,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus
from apc.primitives.router import Router, RouterConfig


def test_estimate_linear_flops_rederivation() -> None:
    """Deterministic rederivation of linear layer FLOPs: 2 * tokens * in * out (+ bias)."""
    # in=4, out=8, tokens=3, bias=True
    # MACs = 3 * 4 * 8 = 96
    # FLOPs = 2 * 96 + (3 * 8) = 192 + 24 = 216
    assert estimate_linear_flops(in_features=4, out_features=8, num_tokens=3, bias=True) == 216

    # in=4, out=8, tokens=3, bias=False
    # FLOPs = 2 * 96 = 192
    assert estimate_linear_flops(in_features=4, out_features=8, num_tokens=3, bias=False) == 192


def test_estimate_attention_flops_rederivation() -> None:
    """Deterministic rederivation of self/cross-attention FLOPs from first principles."""
    # seq_q=4, seq_kv=4, d_model=8, n_head=2, batch_size=1
    # Q proj: 2 * 1 * 4 * 64 = 512
    # K proj: 2 * 1 * 4 * 64 = 512
    # V proj: 2 * 1 * 4 * 64 = 512
    # Q @ K^T score: 2 * 1 * 4 * 4 * 8 = 256
    # Softmax: 3 * 1 * 2 * 4 * 4 = 96
    # Attn @ V context: 2 * 1 * 4 * 4 * 8 = 256
    # Out proj: 2 * 1 * 4 * 64 = 512
    # Total = 512 + 512 + 512 + 256 + 96 + 256 + 512 = 2656
    expected = 2656
    assert (
        estimate_attention_flops(
            seq_len_q=4,
            seq_len_kv=4,
            d_model=8,
            n_head=2,
            causal=True,
            batch_size=1,
        )
        == expected
    )


def test_estimate_router_flops_rederivation() -> None:
    """Deterministic rederivation of router scoring FLOPs as candidate bank scales."""
    # d_model=16, score_dim=8, C=10, batch=2
    # Query proj: 2 * 2 * 16 * 8 = 512
    # Dot product scores: 2 * 2 * 10 * 8 = 320
    # Softmax: 3 * 2 * 10 = 60
    # Top-k: 2 * 10 = 20
    # Total = 512 + 320 + 60 + 20 = 912
    expected = 912
    assert (
        estimate_router_flops(
            d_model=16,
            score_dim=8,
            num_candidates=10,
            batch_size=2,
        )
        == expected
    )


def test_estimate_pointwise_primitive_flops_rederivation() -> None:
    """Deterministic rederivation of PointwisePrimitive low-rank transform FLOPs."""
    # seq_len=6, d_model=16, rank=4, batch=2
    # A proj: 2 * 2 * 6 * 16 * 4 = 1536
    # B proj: 2 * 2 * 6 * 4 * 16 = 1536
    # Residual gated add: 2 * 2 * 6 * 16 = 384
    # Total = 1536 + 1536 + 384 = 3456
    expected = 3456
    assert (
        estimate_pointwise_primitive_flops(
            seq_len=6,
            d_model=16,
            rank=4,
            batch_size=2,
        )
        == expected
    )


def test_count_system_parameters_deterministic() -> None:
    """Deterministic test verifying exact parameter breakdown across all subsystems."""
    d_model = 16
    vocab_size = 8

    # 1. Stable Core (1 layer, minimal dimensions)
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=16,
        d_model=d_model,
        n_layer=1,
        n_head=2,
        d_ff=32,
    )
    core = DecoderOnlyTransformer(cfg)
    core_expected = sum(p.numel() for p in core.parameters())

    # 2. Router with 3 candidate primitive keys
    router_cfg = RouterConfig(d_model=d_model, score_dim=8, top_k=1)
    router = Router(router_cfg)
    for pid in [0, 1, 2]:
        router.add_primitive_key(pid)
    # query_proj: 16*8 + 8 = 136; keys: 3 * 8 = 24. Total = 160.
    router_expected = sum(p.numel() for p in router.parameters())
    assert router_expected == 160

    # 3. Bank with 3 PointwisePrimitives (rank=4)
    # Each PointwisePrimitive: A: 16*4 = 64; B: 4*16 = 64. Total per prim = 128.
    bank = PrimitiveBank()
    p0 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    assert p0.num_parameters() == 128
    bank_expected = 3 * 128  # 384

    # 4. Workspace with 1 PointwisePrimitive
    workspace = PlasticWorkspace()
    workspace.allocate(
        "test", num_transforms=1, config=PrimitiveConfig(d_model=d_model, rank=4)
    )
    workspace_expected = 128

    # Compute breakdown selecting only primitive 0 (active = 1 primitive)
    breakdown = count_system_parameters(
        core=core,
        router=router,
        bank=bank,
        selected_ids=[0],
        workspace=workspace,
    )

    assert breakdown.stable_core_params == core_expected
    assert breakdown.router_params == router_expected
    assert breakdown.resident_primitive_params == bank_expected
    assert breakdown.active_primitive_params == 128
    assert breakdown.temporary_params == workspace_expected

    expected_resident_total = core_expected + router_expected + bank_expected
    expected_active_total = core_expected + router_expected + 128 + workspace_expected
    assert breakdown.resident_total_params == expected_resident_total
    assert breakdown.active_total_params == expected_active_total

    # Primitive savings: 1 - 128 / 384 = 2/3 ≈ 0.6667
    assert abs(breakdown.primitive_parameter_savings_ratio - (1.0 - 128 / 384)) < 1e-6
    # Total savings: 1 - active_total / resident_total
    expected_total_savings = 1.0 - (expected_active_total / expected_resident_total)
    assert abs(breakdown.total_parameter_savings_ratio - expected_total_savings) < 1e-6


def test_compute_flops_breakdown_sparse_vs_dense() -> None:
    """Verify that FLOPs breakdown correctly estimates sparse vs dense execution."""
    d_model = 16
    vocab_size = 8
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=16,
        d_model=d_model,
        n_layer=1,
        n_head=2,
        d_ff=32,
    )
    core = type("DummyCore", (), {"model": DecoderOnlyTransformer(cfg)})()

    router = Router(RouterConfig(d_model=d_model, score_dim=8, top_k=1))
    bank = PrimitiveBank()
    p0 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=d_model, rank=4), status=PrimitiveStatus.STABLE
    )
    for pid in bank.ids():
        router.add_primitive_key(pid)

    flops = compute_flops_breakdown(
        core=core,
        router=router,
        bank=bank,
        selected_ids=[p0.primitive_id],
        seq_len_task=6,
        seq_len_content=8,
        seq_len_out=8,
        batch_size=1,
    )

    # Sparse selects 1 primitive; Dense executes all 3 primitives
    assert flops.selected_primitive_flops > 0
    assert flops.dense_primitive_flops == 3 * flops.selected_primitive_flops
    assert flops.total_sparse_flops < flops.dense_baseline_flops
    assert flops.flops_savings_ratio > 0.0
    expected_savings = 1.0 - (flops.total_sparse_flops / flops.dense_baseline_flops)
    assert flops.flops_savings_ratio == expected_savings


def test_dense_primitive_baseline_execution() -> None:
    """Verify execute_dense_primitive_baseline actually runs all resident modules."""
    bank = PrimitiveBank()
    p0 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=16, rank=4), status=PrimitiveStatus.STABLE
    )
    p1 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=16, rank=4), status=PrimitiveStatus.STABLE
    )
    p2 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=16, rank=4), status=PrimitiveStatus.CANDIDATE
    )

    h = torch.randn(2, 6, 16)
    lengths = [6, 6]
    out_lengths = [6, 6]

    outputs = execute_dense_primitive_baseline(bank, h, lengths, out_lengths)

    # Only STABLE primitives p0 and p1 should execute
    assert set(outputs.keys()) == {p0.primitive_id, p1.primitive_id}
    assert outputs[p0.primitive_id].shape == (2, 6, 16)
    assert outputs[p1.primitive_id].shape == (2, 6, 16)
    assert p0.forward_call_count == 1
    assert p1.forward_call_count == 1
    assert p2.forward_call_count == 0


def test_verify_sparse_execution() -> None:
    """Verify sparse invariant checker detects both clean sparse execution and leaks."""
    bank = PrimitiveBank()
    p0 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=16, rank=4), status=PrimitiveStatus.STABLE
    )
    p1 = bank.new_pointwise_primitive(
        PrimitiveConfig(d_model=16, rank=4), status=PrimitiveStatus.STABLE
    )

    # Clean case: p0 selected and called 5 times; p1 unselected and called 0 times
    p0.forward_call_count = 5
    p1.forward_call_count = 0

    passed, details = verify_sparse_execution(
        bank, selected_pids=[p0.primitive_id], expected_selected_calls=5
    )
    assert passed is True
    assert len(details["violations"]) == 0

    # Leak case: p1 unselected but received 2 calls
    p1.forward_call_count = 2
    passed_leak, details_leak = verify_sparse_execution(
        bank, selected_pids=[p0.primitive_id], expected_selected_calls=5
    )
    assert passed_leak is False
    assert len(details_leak["violations"]) == 1
    assert details_leak["violations"][0]["type"] == "unselected_called"


def test_profile_execution_cpu() -> None:
    """Verify runtime latency profiler measures timing and percentiles on CPU."""
    call_count = 0

    def dummy_fn() -> int:
        nonlocal call_count
        call_count += 1
        x = torch.zeros(10, 10)
        return int(x.sum().item())

    res, metrics = profile_execution(
        dummy_fn,
        warmup_steps=3,
        active_steps=10,
        batch_size=4,
        device="cpu",
    )

    assert res == 0
    assert call_count == 13  # 3 warmup + 10 active
    assert metrics.total_trials == 10
    assert metrics.min_ms <= metrics.median_ms <= metrics.p95_ms <= metrics.p99_ms <= metrics.max_ms
    assert metrics.mean_ms > 0.0
    assert metrics.throughput_examples_per_sec > 0.0
    assert metrics.peak_gpu_memory_bytes == 0  # CPU has 0 VRAM
