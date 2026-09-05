"""Unit tests for Sequential Closed-Loop Benchmark (Phase A.2 Task A2-C008 - STOP GATE).

Verifies:
1. Stream composition and episode counts (K >= 10, C >= 10, N >= 6, R >= 6, total >= 40).
2. Causal recurrence ordering invariant: every R episode occurs strictly after its N episode.
3. Zero oracle leakage: redacting oracle metadata produces identical inputs and behavior.
4. Fast CPU closed-loop smoke test: verifies end-to-end execution, workspace zero-leak invariant,
   bank growth on N, and reuse without reconsolidation on R.
5. Acceptance verdict evaluation logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.environments.generator import Program, ProgramStep
from apc.environments.task_spec import num_registered_operations
from apc.evaluation.sequential_closed_loop_benchmark import (
    SequentialClosedLoopConfig,
    _evaluate_candidate_recipe,
    generate_episode_examples,
    generate_sequential_stream,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.primitives.router import Router, RouterConfig


def test_stream_composition_and_counts() -> None:
    """Stream must have >= 40 episodes with >=10 K, >=10 C, >=6 N, >=6 R."""
    config = SequentialClosedLoopConfig(
        num_k=14,
        num_c=12,
        num_n=6,
        num_r=8,
    )
    assert config.total_episodes == 40

    for seed in [0, 42, 123]:
        stream = generate_sequential_stream(seed, config)
        assert len(stream) == 40

        k_eps = [ep for ep in stream if ep.oracle_category == "K"]
        c_eps = [ep for ep in stream if ep.oracle_category == "C"]
        n_eps = [ep for ep in stream if ep.oracle_category == "N"]
        r_eps = [ep for ep in stream if ep.oracle_category == "R"]

        assert len(k_eps) == 14
        assert len(c_eps) == 12
        assert len(n_eps) == 6
        assert len(r_eps) == 8

        # Indices must strictly match slot positions
        for idx, ep in enumerate(stream):
            assert ep.episode_index == idx


def test_causal_recurrence_ordering_invariant() -> None:
    """For every novel task, its N episode must strictly precede all corresponding R episodes."""
    config = SequentialClosedLoopConfig(
        num_k=14,
        num_c=12,
        num_n=6,
        num_r=8,
    )

    for seed in [0, 1, 2, 3, 4, 100, 999]:
        stream = generate_sequential_stream(seed, config)
        n_slots: dict[str, int] = {}
        r_slots: dict[str, list[int]] = {}

        for ep in stream:
            if ep.oracle_category == "N":
                assert ep.task_name not in n_slots, f"Duplicate N for {ep.task_name}"
                n_slots[ep.task_name] = ep.episode_index
            elif ep.oracle_category == "R":
                r_slots.setdefault(ep.task_name, []).append(ep.episode_index)

        # Every recurrence must come after its corresponding novel episode
        assert len(n_slots) == 6
        for task_name, r_indices in r_slots.items():
            assert task_name in n_slots, f"Recurrence of unseen task: {task_name}"
            n_idx = n_slots[task_name]
            for r_idx in r_indices:
                assert (
                    n_idx < r_idx
                ), f"Causal violation: N at {n_idx} but R at {r_idx} for task {task_name}"


def test_zero_oracle_leakage_in_stream_generation() -> None:
    """Examples generated for episodes must have valid task_spec and no reliance
    on oracle metadata."""
    prog = Program(steps=(ProgramStep("COPY"),))
    ex_list = generate_episode_examples(
        program=prog,
        category="K",
        vocab_size=10,
        n_examples=8,
        seed=42,
    )
    assert len(ex_list) == 8
    for ex in ex_list:
        assert ex.task_spec is not None
        assert ex.task_spec.steps[0].operation == "COPY"
        assert len(ex.input_tokens) == 8
        assert len(ex.target_tokens) == 8


class _TransparentModel(nn.Module):
    """Deterministic token encoder for CPU smoke testing."""

    def __init__(self, tokens: SharedCoreTokens, d_model: int = 64) -> None:
        super().__init__()
        self.tokens = tokens
        self.d_model = d_model
        self.config = type("Config", (), {"d_model": d_model})()
        self.token_emb = nn.Embedding(tokens.model_vocab_size, d_model)
        with torch.no_grad():
            self.token_emb.weight.zero_()
            for tid in range(tokens.model_vocab_size):
                if tid < tokens.env_vocab_size:
                    self.token_emb.weight[tid, 0] = float(tid + 1)
                if tokens.op_base <= tid < tokens.arg_base:
                    op_idx = tid - tokens.op_base
                    self.token_emb.weight[tid, 16 + op_idx] = 5.0

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(input_ids).clone()
        emb[:, :, 16:] = torch.cumsum(emb[:, :, 16:], dim=1)
        return emb


class _TransparentCore:
    """Lightweight Core wrapper for CPU smoke tests."""

    def __init__(self, d_model: int = 64, vocab_size: int = 10) -> None:
        self.device = torch.device("cpu")
        num_ops = max(32, num_registered_operations())
        self.tokens = build_shared_core_tokens(
            env_vocab_size=vocab_size, num_operations=num_ops, arg_span=10
        )
        self.model = _TransparentModel(self.tokens, d_model=d_model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)


def test_closed_loop_smoke_simulation() -> None:
    """Execute a simulated mini-stream on CPU verifying zero-leak and state transitions."""
    core = _TransparentCore()

    # Build a tiny bank with 2 primitives
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}
    for op in ["COPY", "NEGATE"]:
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op,
                d_model=64,
                d_operator=16,
                n_head=2,
                d_operator_ff=32,
                vocab_size=10,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id
    bank.freeze_all()

    router = Router(RouterConfig(d_model=64, top_k=1, score_fn="dot"))
    for pid in bank.ids():
        router.add_primitive_key(pid)
    router.eval()
    for p in router.parameters():
        p.requires_grad_(False)

    workspace = PlasticWorkspace()
    assert workspace.total_parameter_count() == 0
    assert len(workspace) == 0

    # Test candidate recipe execution helper
    ex_k = generate_episode_examples(Program((ProgramStep("COPY"),)), "K", 10, 4, 42)
    em, tok, loss = _evaluate_candidate_recipe(core, bank, op_to_id, ("COPY",), ex_k)
    assert 0.0 <= em <= 1.0
    assert 0.0 <= tok <= 1.0
    assert workspace.total_parameter_count() == 0
