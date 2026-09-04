"""Unit and integration tests for Plastic Workspace Residual Learning (Task A1-B005)."""

from __future__ import annotations

import pytest

from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    InvertHalfOp,
    SwapPairsOp,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation.plastic_workspace_benchmark import (
    PlasticWorkspaceBenchmarkConfig,
    generate_novel_examples,
    plastic_workspace_config_from_dict,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedContentEncoder,
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.plastic.residual import (
    execute_plastic_residual,
    get_parameter_breakdown,
    verify_frozen_invariants,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveConfig,
)


def _tiny_core(d_model: int = 32) -> SharedContentEncoder:
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=42,
        vocab_size=10,
        sequence_length_range=(6, 8),
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        arg_dim=8,
        max_sequence_length=16,
        model={
            "d_model": d_model,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
    )
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)
    return core


def _tiny_bank(d_model: int = 16) -> tuple[PrimitiveBank, dict[str, int]]:
    bank = PrimitiveBank()
    p1 = bank.new_pointwise_primitive(PrimitiveConfig(d_model=d_model, rank=2))
    p1.freeze()
    op_to_id = {"COPY": p1.primitive_id}
    return bank, op_to_id


def test_novel_operations_semantics() -> None:
    """Verify SWAP_PAIRS and INVERT_HALF adhere to expected mathematical transforms."""
    swap_op = get_operation("SWAP_PAIRS")
    assert isinstance(swap_op, SwapPairsOp)
    assert swap_op.output_length(6) == 6
    assert swap_op.apply((1, 2, 3, 4, 5, 6), 10, {}) == (2, 1, 4, 3, 6, 5)
    assert swap_op.apply((1, 2, 3, 4, 5), 10, {}) == (2, 1, 4, 3, 5)

    inv_op = get_operation("INVERT_HALF")
    assert isinstance(inv_op, InvertHalfOp)
    assert inv_op.output_length(6) == 6
    # For length 6, half is 3. First 3 tokens negated: 9 - x, remaining 3 kept.
    assert inv_op.apply((1, 2, 3, 4, 5, 6), 10, {}) == (8, 7, 6, 4, 5, 6)

    assert "SWAP_PAIRS" in BRANCH_B_NOVEL_OPERATION_NAMES
    assert "INVERT_HALF" in BRANCH_B_NOVEL_OPERATION_NAMES


def test_plastic_workspace_allocation_and_release() -> None:
    """Test allocating a compact cross-position operator and releasing it."""
    ws = PlasticWorkspace()
    assert not ws.is_allocated
    assert ws.total_parameter_count() == 0

    cfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=16,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = ws.allocate_compact_operator(cfg, label="test_swap")
    assert ws.is_allocated
    assert ws.ids() == [pid]
    param_count = ws.total_parameter_count()
    assert param_count > 0
    assert param_count <= 100_000

    # Double allocation must raise RuntimeError
    with pytest.raises(RuntimeError, match="already has allocated capacity"):
        ws.allocate_compact_operator(cfg)

    # Release drops capacity completely
    released = ws.release()
    assert pid in released
    assert not ws.is_allocated
    assert ws.total_parameter_count() == 0
    assert ws.ids() == []


def test_parameter_breakdown_and_freeze_invariants() -> None:
    """Verify frozen invariants checks and parameter accounting."""
    core = _tiny_core(d_model=32)
    bank, _ = _tiny_bank(d_model=32)
    ws = PlasticWorkspace()

    # Initial check: core and bank frozen, ws empty
    assert verify_frozen_invariants(core, bank)
    breakdown = get_parameter_breakdown(core, bank, ws)
    assert breakdown.core_parameters > 0
    assert breakdown.bank_parameters > 0
    assert breakdown.temporary_parameters == 0
    assert breakdown.trainable_parameters == 0

    # Allocate plastic capacity
    cfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = ws.allocate_compact_operator(cfg)
    breakdown2 = get_parameter_breakdown(core, bank, ws, active_workspace_ids=[pid])
    assert breakdown2.temporary_parameters > 0
    assert breakdown2.trainable_parameters == breakdown2.temporary_parameters
    assert breakdown2.active_parameters == breakdown2.temporary_parameters

    # If any core parameter is unfrozen, verify_frozen_invariants must raise AssertionError
    for p in core.model.parameters():
        p.requires_grad_(True)
        break
    with pytest.raises(AssertionError, match="not frozen"):
        verify_frozen_invariants(core, bank)


def test_residual_execution_gradient_isolation() -> None:
    """Verify that gradients flow exclusively to plastic capacity, leaving core and bank untouched.
    """
    d_model = 32
    core = _tiny_core(d_model=d_model)
    bank, op_to_id = _tiny_bank(d_model=d_model)
    ws = PlasticWorkspace()

    prim_cfg = CrossPositionPrimitiveConfig(
        operation="INVERT_HALF",
        d_model=d_model,
        d_operator=16,
        n_head=2,
        d_operator_ff=32,
        vocab_size=10,
        max_sequence_length=16,
    )
    pid = ws.allocate_compact_operator(prim_cfg)
    plastic_op = ws.get(pid)

    ex = Example(
        input_tokens=(1, 2, 3, 4, 5, 6),
        target_tokens=(8, 7, 6, 4, 5, 6),
        program=Program(steps=(ProgramStep("INVERT_HALF", {}),)),
        operation_graph=None,
        category="novel",
        split="train",
        vocab_size=10,
        task_spec=TaskSpec.from_program(Program(steps=(ProgramStep("INVERT_HALF", {}),))),
        oracle_metadata=OracleMetadata("N", ("INVERT_HALF",)),
    )

    logits = execute_plastic_residual(
        core,
        bank,
        op_to_id,
        ws,
        pid,
        [ex],
        base_candidate_operations=None,
        target_vocab_size=10,
    )
    loss = logits.sum()
    loss.backward()

    # Verify core received ZERO gradients
    for p in core.model.parameters():
        assert p.grad is None

    # Verify bank received ZERO gradients
    for bpid in bank.ids():
        for p in bank.get(bpid).parameters():
            assert p.grad is None

    # Verify plastic operator parameters DID receive gradients
    trainable_grads = [p.grad for p in plastic_op.parameters() if p.requires_grad]
    assert len(trainable_grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in trainable_grads)


def test_generate_novel_examples_determinism() -> None:
    """Verify novel example generator is deterministic given seed."""
    exs1 = generate_novel_examples(42, 10, operation="SWAP_PAIRS", split="train")
    exs2 = generate_novel_examples(42, 10, operation="SWAP_PAIRS", split="train")
    assert len(exs1) == 10
    for e1, e2 in zip(exs1, exs2, strict=True):
        assert e1.input_tokens == e2.input_tokens
        assert e1.target_tokens == e2.target_tokens


def test_config_roundtrip() -> None:
    """Verify config dict round-trip and validation."""
    cfg = PlasticWorkspaceBenchmarkConfig(
        seed=1,
        plastic_train_steps=100,
        novel_operations=("SWAP_PAIRS",),
    )
    d = cfg.to_dict()
    reconstructed = plastic_workspace_config_from_dict(d)
    assert reconstructed.seed == 1
    assert reconstructed.plastic_train_steps == 100
    assert reconstructed.novel_operations == ("SWAP_PAIRS",)
