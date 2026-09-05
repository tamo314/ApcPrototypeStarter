"""Unit tests for Bank Competition and Routing Scaling Benchmark (Task A2-C004).

Tests:
1. Bank and router scaling construction across target sizes N in {10, 16, 32}.
2. Matched scale of distractor primitives (parameters match compact real primitives).
3. Compute accounting invariants: resident primitive params scale O(N) while
   active params remain O(1).
4. Strict sparse execution invariant: unselected calls == 0.
5. Accurate distractor false-selection rate calculation.
"""

from __future__ import annotations

import torch

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_shared_core_tokens
from apc.evaluation.bank_scaling_benchmark import (
    build_scaled_bank_and_router,
    evaluate_bank_scaling_at_size,
)
from apc.evaluation.compute_accounting import count_system_parameters
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.primitives.router import Router, RouterConfig


class DummyCore:
    """Lightweight dummy core for CPU testing."""

    def __init__(self, d_model: int = 32, vocab_size: int = 10) -> None:
        self.tokens = build_shared_core_tokens(vocab_size, num_operations=16, arg_span=10)
        cfg = TransformerConfig(
            vocab_size=self.tokens.model_vocab_size,
            d_model=d_model,
            n_layer=1,
            n_head=2,
            d_ff=32,
            max_seq_len=24,
        )
        self.model = DecoderOnlyTransformer(cfg)
        self.device = torch.device("cpu")


def _build_dummy_16_system(
    d_model: int = 32,
) -> tuple[DummyCore, PrimitiveBank, Router, dict[str, int]]:
    """Construct a minimal 16-primitive bank and calibrated router for testing."""
    core = DummyCore(d_model=d_model)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}

    from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS
    from apc.evaluation.incremental_router_benchmark import INITIAL_10_OPERATIONS

    all_ops = list(INITIAL_10_OPERATIONS) + list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)

    for _i, op in enumerate(all_ops):
        p_cfg = CrossPositionPrimitiveConfig(
            operation=op,
            d_model=d_model,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=24,
        )
        p = bank.new_cross_position_primitive(p_cfg, status=PrimitiveStatus.STABLE)
        op_to_id[op] = p.primitive_id

    router = Router(RouterConfig(d_model=d_model, top_k=1, score_fn="dot"))
    for pid in bank.ids():
        router.add_primitive_key(pid)
        with torch.no_grad():
            # Distinct key directions
            key = torch.randn(d_model)
            key = key / key.norm(p=2) * 5.0
            router.key_parameter(pid).copy_(key)

    return core, bank, router, op_to_id


def test_build_scaled_bank_and_router_sizes() -> None:
    """Verify bank and router construction for N=10, N=16, and N=32."""
    core, bank_16, router_16, op_to_id = _build_dummy_16_system(d_model=32)

    # Size 10
    b10, r10, all10, sem10, dist10 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=10, seed=42
    )
    assert len(b10) == 10
    assert len(r10) == 10
    assert len(sem10) == 10
    assert len(dist10) == 0
    assert len(all10) == 10

    # Size 16
    b16, r16, all16, sem16, dist16 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=16, seed=42
    )
    assert len(b16) == 16
    assert len(r16) == 16
    assert len(sem16) == 16
    assert len(dist16) == 0
    assert len(all16) == 16

    # Size 32 (16 semantic + 16 distractors)
    b32, r32, all32, sem32, dist32 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=32, seed=42
    )
    assert len(b32) == 32
    assert len(r32) == 32
    assert len(sem32) == 16
    assert len(dist32) == 16
    assert len(all32) == 32

    # Check distractor metadata and status
    for pid in dist32:
        prim = b32.get(pid)
        assert prim.status == PrimitiveStatus.STABLE
        assert prim.metadata.get("is_distractor") is True
        assert "DISTRACTOR_" in prim.metadata.get("label", "")


def test_distractor_scale_and_compute_accounting() -> None:
    """Verify matched-scale parameters and O(N) vs O(1) scaling."""
    core, bank_16, router_16, op_to_id = _build_dummy_16_system(d_model=32)

    # Check single primitive param count against a matched non-parameterized semantic primitive
    sample_sem_prim = bank_16.get(op_to_id["SWAP_ENDS"])
    sem_params = sum(p.numel() for p in sample_sem_prim.parameters())

    b32, r32, all32, sem32, dist32 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=32, seed=42
    )
    sample_dist_prim = b32.get(dist32[0])
    dist_params = sum(p.numel() for p in sample_dist_prim.parameters())

    # Distractor must be matched-scale
    assert sem_params == dist_params, (
        f"Distractor params ({dist_params}) != Semantic params ({sem_params})"
    )

    # Parameter accounting at N=16 vs N=32
    pb16 = count_system_parameters(core, router_16, bank_16, selected_ids=[op_to_id["SWAP_ENDS"]])
    pb32 = count_system_parameters(core, r32, b32, selected_ids=[op_to_id["SWAP_ENDS"]])

    # Resident primitive params scale with added distractor entries
    assert pb32.resident_primitive_params == pb16.resident_primitive_params + 16 * dist_params
    # Active primitive params remain O(1): exactly 1 primitive executed
    assert pb32.active_primitive_params == pb16.active_primitive_params
    # Primitive parameter savings increases with bank size
    assert pb32.primitive_parameter_savings_ratio > pb16.primitive_parameter_savings_ratio

    # Further scaling to N=64 adds another 32 distractors
    b64, r64, all64, sem64, dist64 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=64, seed=42
    )
    pb64 = count_system_parameters(core, r64, b64, selected_ids=[op_to_id["SWAP_ENDS"]])
    assert pb64.resident_primitive_params == pb32.resident_primitive_params + 32 * dist_params
    assert pb64.active_primitive_params == pb32.active_primitive_params


def test_evaluate_bank_scaling_at_size_synthetic() -> None:
    """Verify evaluation logic, distractor false-selection, and zero unselected calls."""
    d_model = 32
    core, bank_16, router_16, op_to_id = _build_dummy_16_system(d_model=d_model)

    # Set query_proj to identity so query = z
    with torch.no_grad():
        router_16.query_proj.weight.copy_(torch.eye(d_model))

    b32, r32, all32, sem32, dist32 = build_scaled_bank_and_router(
        core, bank_16, router_16, op_to_id, target_size=32, seed=42
    )

    # Generate synthetic z_task perfectly aligned with semantic router keys
    eval_z: dict[int, list[tuple[torch.Tensor, int]]] = {}
    rec_z: dict[int, list[tuple[torch.Tensor, int]]] = {}
    baseline_top1: dict[int, float] = {}

    with torch.no_grad():
        for pid in sem32:
            key = r32.key_parameter(pid).clone()
            # z matches key direction
            z = key.unsqueeze(0).repeat(10, 1) + torch.randn(10, d_model) * 0.01
            eval_z[pid] = [(z[i], pid) for i in range(10)]
            rec_z[pid] = [(z[i], pid) for i in range(5)]
            baseline_top1[pid] = 1.0

    res = evaluate_bank_scaling_at_size(
        core=core,
        scaled_bank=b32,
        scaled_router=r32,
        all_pids=all32,
        semantic_pids=sem32,
        distractor_pids=dist32,
        op_to_id=op_to_id,
        eval_z_by_pid=eval_z,
        recurrence_z_by_pid=rec_z,
        baseline_semantic_top1=baseline_top1,
        warmup_steps=2,
        active_profile_steps=5,
    )

    assert res.bank_size == 32
    assert res.num_semantic_ops == 16
    assert res.num_distractors == 16
    assert res.known_task_top1 >= 0.95
    assert res.distractor_false_selection_rate == 0.0
    assert res.unselected_forward_calls == 0
    assert res.passed is True
    assert res.latency_ratio > 0.0
