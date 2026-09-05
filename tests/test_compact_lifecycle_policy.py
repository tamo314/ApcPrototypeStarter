"""Unit tests for Compact-First Plastic Lifecycle Policy (Task A2-C007).

Validates the mechanical lifecycle guarantees:
1. No plastic before direct/composition inadequacy.
2. Exactly one promotion per successful novel (N) task.
3. Temporary workspace returns to zero parameters across all paths.
4. Zero promotions for Known (K), Composition (C), and Recurrence (R) tasks.
5. Fallback invocation and controls tracking.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.environments.generator import Example, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.task_spec import num_registered_operations
from apc.meta.adequacy import AdequacyEvidence
from apc.meta.episode_log import ControllerAction
from apc.plastic.lifecycle import (
    CompactLifecycleConfig,
    CompactPlasticLifecyclePolicy,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)


class _TransparentModel(nn.Module):
    """Simple deterministic encoder for CPU unit testing."""

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
    """Lightweight deterministic Core wrapper."""

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


class _ExactFunctionalPrimitive(CrossPositionPrimitive):
    """Test primitive that applies reference operation logic to recovered tokens."""

    def __init__(
        self,
        primitive_id: int,
        operation: str,
        config: CrossPositionPrimitiveConfig,
        vocab_size: int = 10,
    ) -> None:
        super().__init__(primitive_id=primitive_id, config=config, status=PrimitiveStatus.STABLE)
        self.operation = operation
        self.vocab_size = vocab_size

    def forward(
        self,
        h_content: torch.Tensor,
        content_lengths: list[int] | None = None,
        output_lengths: list[int] | None = None,
        argument_values: list[Any] | None = None,
    ) -> torch.Tensor:
        batch_size = h_content.shape[0]
        max_out_len = max(output_lengths) if output_lengths is not None else h_content.shape[1]
        logits = torch.zeros(
            (batch_size, max_out_len, self.vocab_size),
            device=h_content.device,
            dtype=h_content.dtype,
        )
        op_def = get_operation(self.operation)

        for b in range(batch_size):
            in_len = content_lengths[b] if content_lengths is not None else h_content.shape[1]
            recovered_tokens = tuple(
                int(round(h_content[b, pos, 0].item() - 1.0)) for pos in range(in_len)
            )
            out_tokens = op_def.apply(recovered_tokens, self.vocab_size, params={})
            for pos, tok in enumerate(out_tokens):
                if 0 <= tok < self.vocab_size and pos < max_out_len:
                    logits[b, pos, tok] = 20.0

        return logits


def _make_examples(op_name: str, n: int) -> list[Example]:
    prog = Program((ProgramStep(op_name),))
    examples: list[Example] = []
    for i in range(n):
        seq = tuple(((i * 3 + j) % 9) + 1 for j in range(6))
        res = run_program(prog, seq, 10)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split="train",
                vocab_size=10,
            )
        )
    return examples


@pytest.fixture
def test_env():
    core = _TransparentCore(d_model=64, vocab_size=10)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}

    for op in ["COPY", "REVERSE", "NEGATE"]:
        cfg = CrossPositionPrimitiveConfig(
            operation=op,
            d_model=64,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=32,
        )
        pid = len(bank)
        p = _ExactFunctionalPrimitive(pid, op, cfg, 10)
        bank.add_primitive(p)
        op_to_id[op] = pid

    bank.freeze_all()
    hist_examples = {op: _make_examples(op, 8) for op in ["COPY", "REVERSE", "NEGATE"]}
    return core, bank, op_to_id, hist_examples


def _make_evidence(
    *, direct_em: float = 0.0, composition_em: float = 0.0, direct_loss: float = 0.01
) -> AdequacyEvidence:
    """Helper to create valid AdequacyEvidence records."""
    return AdequacyEvidence(
        direct_em=direct_em,
        direct_loss=direct_loss,
        direct_token_acc=direct_em,
        direct_primitive_id=0 if direct_em > 0 else None,
        composition_em=composition_em,
    )


def test_k_task_direct_reuse_no_plastic(test_env):
    """K task with DIRECT_REUSE must not trigger plastic and must not promote."""
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(compact_budget_steps=2, fallback_budget_steps=2)
    policy = CompactPlasticLifecyclePolicy(config)

    examples = hist_examples["COPY"]
    report = policy.execute_episode(
        episode_id="test_ep_k",
        task_name="COPY",
        action=ControllerAction.DIRECT_REUSE,
        evidence=_make_evidence(direct_em=1.0),
        train_examples=examples[:4],
        eval_examples=examples[4:],
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )

    assert not report.plastic_triggered
    assert not report.compact_attempted
    assert not report.fallback_invoked
    assert report.promotions_count == 0
    assert report.promoted_primitive_id is None
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size


def test_c_task_compose_no_plastic(test_env):
    """C task with COMPOSE must not trigger plastic and must not promote."""
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(compact_budget_steps=2, fallback_budget_steps=2)
    policy = CompactPlasticLifecyclePolicy(config)

    examples = hist_examples["REVERSE"]
    report = policy.execute_episode(
        episode_id="test_ep_c",
        task_name="REVERSE_NEGATE",
        action=ControllerAction.COMPOSE,
        evidence=_make_evidence(direct_em=0.0, composition_em=1.0),
        train_examples=examples[:4],
        eval_examples=examples[4:],
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )

    assert not report.plastic_triggered
    assert not report.compact_attempted
    assert not report.fallback_invoked
    assert report.promotions_count == 0
    assert report.promoted_primitive_id is None
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size


def test_r_task_recurrence_no_plastic(test_env):
    """R task with DIRECT_REUSE must not trigger plastic and must not promote."""
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(compact_budget_steps=2, fallback_budget_steps=2)
    policy = CompactPlasticLifecyclePolicy(config)

    examples = hist_examples["NEGATE"]
    report = policy.execute_episode(
        episode_id="test_ep_r",
        task_name="NEGATE",
        action=ControllerAction.DIRECT_REUSE,
        evidence=_make_evidence(direct_em=1.0),
        train_examples=examples[:4],
        eval_examples=examples[4:],
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )

    assert not report.plastic_triggered
    assert not report.compact_attempted
    assert not report.fallback_invoked
    assert report.promotions_count == 0
    assert report.promoted_primitive_id is None
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size


def test_premature_plastic_guard_raises_error(test_env):
    """If direct or composition is adequate, triggering PLASTIC_SEARCH must raise RuntimeError."""
    core, bank, op_to_id, hist_examples = test_env
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(compact_budget_steps=2, fallback_budget_steps=2)
    policy = CompactPlasticLifecyclePolicy(config)

    examples = hist_examples["COPY"]
    with pytest.raises(RuntimeError, match="Premature plastic trigger"):
        policy.execute_episode(
            episode_id="test_premature",
            task_name="COPY",
            action=ControllerAction.PLASTIC_SEARCH,
            evidence=_make_evidence(direct_em=1.0),
            train_examples=examples[:4],
            eval_examples=examples[4:],
            historical_eval_examples=hist_examples,
            core=core,
            bank=bank,
            op_to_id=op_to_id,
            workspace=workspace,
        )


def test_n_task_compact_success_one_promotion(test_env):
    """Novel task with compact success promotes exactly 1 primitive; workspace returns to 0."""
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    # Budget sufficient to learn SWAP_PAIRS in compact format
    config = CompactLifecycleConfig(
        compact_budget_steps=120,
        compact_lr=2e-3,
        compact_success_threshold_em=0.60,
        shadow_retention_threshold=0.60,
        eval_batch_size=16,
    )
    policy = CompactPlasticLifecyclePolicy(config)

    train_ex = _make_examples("SWAP_PAIRS", 40)
    eval_ex = _make_examples("SWAP_PAIRS", 20)

    report = policy.execute_episode(
        episode_id="test_ep_n_compact",
        task_name="SWAP_PAIRS",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=train_ex,
        eval_examples=eval_ex,
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=False,
    )

    assert report.plastic_triggered
    assert report.compact_attempted
    assert report.compact_success
    assert not report.compact_failure
    assert not report.fallback_invoked
    assert report.shadow_validation_passed
    assert report.promotions_count == 1
    assert report.promoted_primitive_id is not None
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size + 1
    assert "SWAP_PAIRS" in op_to_id


def test_n_task_fallback_on_compact_failure(test_env):
    """When compact search fails, fallback is invoked; on success, distills and promotes 1."""
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(
        compact_budget_steps=2,
        fallback_budget_steps=120,
        fallback_lr=2e-3,
        distillation_steps=100,
        distillation_lr=2e-3,
        fallback_success_threshold_em=0.60,
        shadow_retention_threshold=0.60,
        eval_batch_size=16,
    )
    policy = CompactPlasticLifecyclePolicy(config)

    train_ex = _make_examples("SWAP_PAIRS", 40)
    eval_ex = _make_examples("SWAP_PAIRS", 20)

    report = policy.execute_episode(
        episode_id="test_ep_n_fallback",
        task_name="SWAP_PAIRS_FALLBACK",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=train_ex,
        eval_examples=eval_ex,
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=True,
    )

    assert report.plastic_triggered
    assert report.compact_attempted
    assert not report.compact_success
    assert report.compact_failure
    assert report.fallback_invoked
    assert report.fallback_success
    assert report.distillation_attempted
    assert report.promotions_count == 1
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size + 1


def test_n_task_double_failure_zero_promotions(test_env):
    """When compact search and fallback both fail (or disabled),
    verify 0 promotions and 0 parameter leak.
    """
    core, bank, op_to_id, hist_examples = test_env
    init_bank_size = len(bank)
    workspace = PlasticWorkspace()

    config = CompactLifecycleConfig(
        compact_budget_steps=2,
        enable_overcomplete_fallback=False,
    )
    policy = CompactPlasticLifecyclePolicy(config)

    train_ex = _make_examples("SWAP_PAIRS", 10)
    eval_ex = _make_examples("SWAP_PAIRS", 10)

    report = policy.execute_episode(
        episode_id="test_ep_n_double_fail",
        task_name="FAIL_TASK",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=train_ex,
        eval_examples=eval_ex,
        historical_eval_examples=hist_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=True,
    )

    assert report.plastic_triggered
    assert report.compact_failure
    assert not report.fallback_invoked
    assert not report.shadow_validation_passed
    assert report.promotions_count == 0
    assert report.promoted_primitive_id is None
    assert report.final_workspace_param_count == 0
    assert len(bank) == init_bank_size
