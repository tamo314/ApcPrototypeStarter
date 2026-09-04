"""Unit tests for Branch B functional consolidation and shadow validation (Task A1-B006)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from apc.consolidation.compact_consolidation import (
    CompactDistillationConfig,
    CompactShadowValidationConfig,
    distill_compact_candidate,
    run_shadow_validation,
)
from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.operations import SwapPairsOp
from apc.environments.task_spec import TaskSpec
from apc.evaluation.consolidation_benchmark import (
    ConsolidationBenchmarkConfig,
    generate_benchmark_examples,
    run_consolidation_benchmark,
)
from apc.plastic.residual import verify_frozen_invariants
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)


def _make_dummy_example(
    input_tokens: tuple[int, ...], target_tokens: tuple[int, ...], op_name: str = "SWAP_PAIRS"
) -> Example:
    step = ProgramStep(operation=op_name, params={})
    prog = Program(steps=(step,))
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=prog,
        operation_graph={},
        category="novel",
        split="train",
        vocab_size=10,
        task_spec=TaskSpec.from_program(prog),
        oracle_metadata=OracleMetadata(label="SP", primitive_operations=(op_name,)),
    )


class _DummyCore:
    def __init__(self, d_model: int = 32, vocab_size: int = 10) -> None:
        self.device = torch.device("cpu")
        self.tokens = build_special_tokens(vocab_size)

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(vocab_size + 4, d_model)

                class _Cfg:
                    pass

                self.config = _Cfg()
                self.config.d_model = d_model

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.emb(x)

        self.model = _Model()
        for p in self.model.parameters():
            p.requires_grad_(False)


def test_distillation_config_validation() -> None:
    cfg = CompactDistillationConfig(steps=100, lr=0.01)
    assert cfg.steps == 100
    assert cfg.lr == 0.01

    with pytest.raises(ValueError, match="steps must be >= 1"):
        CompactDistillationConfig(steps=0)
    with pytest.raises(ValueError, match="lr must be > 0"):
        CompactDistillationConfig(lr=-0.1)
    with pytest.raises(ValueError, match="distillation_alpha must be in"):
        CompactDistillationConfig(distillation_alpha=1.5)


def test_shadow_config_validation() -> None:
    cfg = CompactShadowValidationConfig(retention_threshold=0.95, max_forgetting_threshold=0.02)
    assert cfg.retention_threshold == 0.95
    assert cfg.max_forgetting_threshold == 0.02

    with pytest.raises(ValueError, match="retention_threshold must be in"):
        CompactShadowValidationConfig(retention_threshold=1.5)
    with pytest.raises(ValueError, match="max_forgetting_threshold must be >= 0"):
        CompactShadowValidationConfig(max_forgetting_threshold=-0.1)


def test_distill_compact_candidate_runs_and_decreases_loss() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}
    workspace = PlasticWorkspace()

    prim_cfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = workspace.allocate_compact_operator(prim_cfg)

    examples = [
        _make_dummy_example((1, 2, 3, 4), (2, 1, 4, 3)),
        _make_dummy_example((5, 6, 7, 8), (6, 5, 8, 7)),
        _make_dummy_example((0, 3, 1, 2), (3, 0, 2, 1)),
        _make_dummy_example((4, 4, 2, 2), (4, 4, 2, 2)),
    ]

    distill_cfg = CompactDistillationConfig(
        steps=20,
        lr=0.01,
        batch_size=4,
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        max_sequence_length=16,
    )
    candidate, report = distill_compact_candidate(
        core,
        bank,
        op_to_id,
        workspace,
        pid,
        examples,
        "SWAP_PAIRS",
        distill_cfg,
    )

    assert isinstance(candidate, CrossPositionPrimitive)
    assert candidate.status == PrimitiveStatus.CANDIDATE
    assert candidate.is_frozen()
    assert report.steps == 20
    assert report.candidate_parameter_count == candidate.num_parameters()
    # Core must remain frozen
    assert verify_frozen_invariants(core, bank)


def test_shadow_validation_failing_preserves_workspace() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}
    workspace = PlasticWorkspace()

    prim_cfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = workspace.allocate_compact_operator(prim_cfg)
    initial_workspace_params = workspace.total_parameter_count()
    assert initial_workspace_params > 0

    candidate = CrossPositionPrimitive(10, prim_cfg, status=PrimitiveStatus.CANDIDATE)
    candidate.freeze()

    eval_examples = [_make_dummy_example((1, 2, 3, 4), (2, 1, 4, 3))]
    hist_examples: dict[str, list[Example]] = {}

    # Set retention threshold to impossible 1.05 so validation must fail
    shadow_cfg = CompactShadowValidationConfig(retention_threshold=1.0)

    installed_id, report = run_shadow_validation(
        core,
        bank,
        op_to_id,
        workspace,
        pid,
        candidate,
        "SWAP_PAIRS",
        eval_examples,
        hist_examples,
        shadow_cfg,
    )

    # When failing:
    # 1. installed_id is None
    # 2. workspace is NOT released
    # 3. bank is NOT modified
    assert installed_id is None
    assert workspace.total_parameter_count() == initial_workspace_params
    assert len(workspace) == 1
    assert len(bank) == 0


def test_shadow_validation_passing_promotes_and_releases_workspace() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    bank = PrimitiveBank()
    op_to_id: dict[str, int] = {}
    workspace = PlasticWorkspace()

    prim_cfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = workspace.allocate_compact_operator(prim_cfg)
    assert workspace.total_parameter_count() > 0

    candidate = CrossPositionPrimitive(0, prim_cfg, status=PrimitiveStatus.CANDIDATE)
    candidate.freeze()

    eval_examples = [_make_dummy_example((1, 2, 3, 4), (2, 1, 4, 3))]
    hist_examples: dict[str, list[Example]] = {}

    # Set retention threshold to 0.0 so validation passes unconditionally
    shadow_cfg = CompactShadowValidationConfig(retention_threshold=0.0)

    installed_id, report = run_shadow_validation(
        core,
        bank,
        op_to_id,
        workspace,
        pid,
        candidate,
        "SWAP_PAIRS",
        eval_examples,
        hist_examples,
        shadow_cfg,
    )

    # When passing:
    # 1. installed_id is returned
    # 2. candidate is promoted to STABLE and installed into bank
    # 3. workspace is 100% released
    assert installed_id == 0
    assert len(bank) == 1
    assert bank.get(0).status == PrimitiveStatus.STABLE
    assert bank.get(0).is_frozen()
    assert op_to_id["SWAP_PAIRS"] == 0
    assert workspace.total_parameter_count() == 0
    assert len(workspace) == 0


def test_generate_benchmark_examples_consistency() -> None:
    ex1 = generate_benchmark_examples(42, 5, operation="SWAP_PAIRS", split="test")
    ex2 = generate_benchmark_examples(42, 5, operation="SWAP_PAIRS", split="test")
    assert len(ex1) == 5
    for a, b in zip(ex1, ex2, strict=True):
        assert a.input_tokens == b.input_tokens
        assert a.target_tokens == b.target_tokens

    # Verify SwapPairsOp logic
    op = SwapPairsOp()
    assert op.apply((1, 2, 3, 4), 10, {}) == (2, 1, 4, 3)
    assert op.apply((1, 2, 3), 10, {}) == (2, 1, 3)


def test_consolidation_benchmark_tiny_end_to_end(tmp_path: Any) -> None:
    tiny_cfg = ConsolidationBenchmarkConfig(
        seed=0,
        vocab_size=10,
        sequence_length_range=(4, 6),
        group_size=2,
        model={
            "d_model": 32,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        max_sequence_length=16,
        novel_operations=("SWAP_PAIRS",),
        num_adaptation_examples=8,
        num_plastic_train_examples=16,
        num_distill_train_examples=16,
        num_eval_examples=10,
        min_eval_examples=10,
        num_historical_eval_examples=5,
        plastic_train_steps=5,
        distill_train_steps=5,
        core_train_steps=4,
        bank_train_steps=2,
        batch_size=4,
        retention_threshold=0.0,  # Ensure pass for synthetic tiny test
    )
    report = run_consolidation_benchmark(tiny_cfg, seed_dir=tmp_path)
    assert report.seed == 0
    assert "SWAP_PAIRS" in report.results_by_operation
    res = report.results_by_operation["SWAP_PAIRS"]
    assert res.workspace_released_completely is True
    assert report.final_bank_size == report.initial_bank_size + 1
    assert (tmp_path / "primitive_bank.pt").is_file()
