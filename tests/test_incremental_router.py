"""Unit tests for incremental router update policies (Task A2-C003).

Tests:
1. Replay buffer bounds and exemplar retention.
2. Embedding alignment preserving argument-value and canonical tokens.
3. R0, R1, and R2 update dynamics on synthetic task representations.
4. Usage count reset and strict zero unselected calls.
"""

from __future__ import annotations

import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_shared_core_tokens
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    RouterReplayBuffer,
    align_shared_core_embeddings,
    evaluate_router_accuracy,
    update_router_incrementally,
)
from apc.primitives.router import Router, RouterConfig


def test_router_replay_buffer_bounds() -> None:
    """Verify per-class and total exemplar bounds."""
    buf = RouterReplayBuffer(max_per_class=4, max_total=10)

    # Add 6 exemplars for class 0 -> should clip to 4
    ex0 = [(torch.randn(8), 0) for _ in range(6)]
    buf.add_exemplars(0, ex0)
    assert buf.count_for_class(0) == 4
    assert len(buf) == 4

    # Add 4 exemplars for class 1 -> total is 8 <= 10
    ex1 = [(torch.randn(8), 1) for _ in range(4)]
    buf.add_exemplars(1, ex1)
    assert buf.count_for_class(1) == 4
    assert len(buf) == 8

    # Add 4 exemplars for class 2 -> total would be 12 > 10, total bound enforced
    ex2 = [(torch.randn(8), 2) for _ in range(4)]
    buf.add_exemplars(2, ex2)
    assert len(buf) <= 10
    assert set(buf.classes()) == {0, 1, 2}


def test_align_shared_core_embeddings() -> None:
    """Verify that expanding operations correctly preserves old argument token embeddings."""
    vocab_size = 10
    old_tokens = build_shared_core_tokens(vocab_size, num_operations=13, arg_span=10)
    new_tokens = build_shared_core_tokens(vocab_size, num_operations=18, arg_span=10)

    # Old model
    old_cfg = TransformerConfig(
        vocab_size=old_tokens.model_vocab_size,
        d_model=16,
        n_layer=1,
        n_head=2,
        d_ff=32,
        max_seq_len=24,
    )
    old_model = DecoderOnlyTransformer(old_cfg)
    old_sd = old_model.state_dict()

    # New model with expanded vocab
    new_cfg = TransformerConfig(
        vocab_size=new_tokens.model_vocab_size,
        d_model=16,
        n_layer=1,
        n_head=2,
        d_ff=32,
        max_seq_len=24,
    )
    new_model = DecoderOnlyTransformer(new_cfg)

    aligned_sd = align_shared_core_embeddings(new_model, old_sd, old_tokens, new_tokens)

    # 1. Base tokens & old operation tokens must match exactly
    old_prefix = old_tokens.op_base + 13
    assert torch.equal(
        aligned_sd["token_emb.weight"][:old_prefix],
        old_sd["token_emb.weight"][:old_prefix],
    )

    # 2. Argument token embeddings must match despite arg_base shift
    for v in range(10):
        old_arg_idx = old_tokens.arg_base + v
        new_arg_idx = new_tokens.arg_base + v
        assert torch.equal(
            aligned_sd["token_emb.weight"][new_arg_idx],
            old_sd["token_emb.weight"][old_arg_idx],
        ), f"Argument token {v} embedding mismatch after remapping"


def test_incremental_update_r0_r1_r2_synthetic() -> None:
    """Validate that R1 causes catastrophic forgetting while R2 and R0 preserve old classes."""
    torch.manual_seed(42)
    d_model = 32

    # Create 5 distinct clusters for 5 classes
    cluster_centers = torch.randn(5, d_model) * 10.0

    def make_data(pid: int, n: int) -> list[tuple[torch.Tensor, int]]:
        return [(cluster_centers[pid] + torch.randn(d_model) * 0.5, pid) for _ in range(n)]

    # Old classes: 0, 1, 2; New classes: 3, 4
    train_data = {pid: make_data(pid, 32) for pid in range(5)}
    eval_data = {pid: make_data(pid, 50) for pid in range(5)}

    # Initial router on classes 0, 1, 2
    r_cfg = RouterConfig(d_model=d_model, top_k=1, score_fn="dot")
    base_router = Router(r_cfg)

    replay_buffer = RouterReplayBuffer(max_per_class=16)
    # Populate replay buffer for initial classes
    for pid in [0, 1, 2]:
        replay_buffer.add_exemplars(pid, train_data[pid])

    # Initial calibration on 0, 1, 2
    inc_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
        router_lr=0.01,
        router_steps=100,
        seed=42,
    )
    update_router_incrementally(
        base_router,
        candidate_ids=[0, 1, 2],
        new_primitive_ids=[0, 1, 2],
        new_data_by_pid={pid: train_data[pid] for pid in [0, 1, 2]},
        replay_buffer=replay_buffer,
        config=inc_cfg,
        all_historical_data_by_pid=train_data,
    )

    base_metrics = evaluate_router_accuracy(
        base_router, [0, 1, 2], {pid: eval_data[pid] for pid in [0, 1, 2]}
    )
    for pid in [0, 1, 2]:
        assert base_metrics[pid]["top1"] >= 0.95, f"Initial router failed on class {pid}"

    # --- Condition R1: Naive update on classes 3, 4 ---
    import copy

    r1_router = copy.deepcopy(base_router)
    r1_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R1_NAIVE_NEW,
        router_lr=0.01,
        router_steps=150,
        seed=42,
    )
    r1_replay = copy.deepcopy(replay_buffer)
    update_router_incrementally(
        r1_router,
        candidate_ids=[0, 1, 2, 3, 4],
        new_primitive_ids=[3, 4],
        new_data_by_pid={3: train_data[3], 4: train_data[4]},
        replay_buffer=r1_replay,
        config=r1_cfg,
    )

    r1_metrics = evaluate_router_accuracy(r1_router, [0, 1, 2, 3, 4], eval_data)
    old_accs_r1 = [r1_metrics[pid]["top1"] for pid in [0, 1, 2]]
    # In R1 without replay, old class accuracy drops significantly due to unregularized key drift
    r1_old_drop = sum(base_metrics[pid]["top1"] - r1_metrics[pid]["top1"] for pid in [0, 1, 2]) / 3
    # Either old accuracy drops or new class suppresses old classes
    assert (
        min(old_accs_r1) < 0.90 or r1_old_drop > 0.05
    ), f"R1 should exhibit forgetting, got old_accs={old_accs_r1}, drop={r1_old_drop}"

    # --- Condition R2: Bounded replay update on classes 3, 4 ---
    r2_router = copy.deepcopy(base_router)
    r2_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
        router_lr=0.01,
        router_steps=150,
        seed=42,
    )
    r2_replay = copy.deepcopy(replay_buffer)
    update_router_incrementally(
        r2_router,
        candidate_ids=[0, 1, 2, 3, 4],
        new_primitive_ids=[3, 4],
        new_data_by_pid={3: train_data[3], 4: train_data[4]},
        replay_buffer=r2_replay,
        config=r2_cfg,
    )

    r2_metrics = evaluate_router_accuracy(r2_router, [0, 1, 2, 3, 4], eval_data)
    for pid in [3, 4]:
        assert r2_metrics[pid]["top1"] >= 0.95, f"R2 failed on new class {pid}: {r2_metrics[pid]}"
    for pid in [0, 1, 2]:
        drop = base_metrics[pid]["top1"] - r2_metrics[pid]["top1"]
        assert drop <= 0.05, f"R2 dropped class {pid} too much: drop={drop}"


def test_sparse_usage_reset() -> None:
    """Verify router usage count is reset to 0 after incremental update."""
    d_model = 16
    router = Router(RouterConfig(d_model=d_model, top_k=1))
    buf = RouterReplayBuffer()
    data = {0: [(torch.randn(d_model), 0) for _ in range(8)]}

    cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R1_NAIVE_NEW,
        router_steps=10,
    )
    update_router_incrementally(
        router,
        candidate_ids=[0],
        new_primitive_ids=[0],
        new_data_by_pid=data,
        replay_buffer=buf,
        config=cfg,
    )
    assert router.usage_counts().get(0, 0) == 0
