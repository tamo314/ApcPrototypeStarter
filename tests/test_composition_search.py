"""Tests for Composition Search Baseline (Task A1-B004, Milestone B-M4)."""

from __future__ import annotations

from typing import Any

from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.task_spec import TaskSpec, TaskStepSpec
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import resolve_candidate_calls
from apc.primitives.composition_search import (
    is_candidate_prefix_valid,
    is_candidate_structurally_valid,
    search_composition_recipe,
)
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
    ReverseRelativePrimitiveConfig,
    ShiftRelativePrimitiveConfig,
)


def _build_test_setup() -> tuple[Any, PrimitiveBank, dict[str, int], list[Example]]:
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
            "d_model": 32,
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

    bank = PrimitiveBank()
    p_shift = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
            arg_dim=8,
        ),
        status=PrimitiveStatus.STABLE,
    )
    p_select = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(
            operation="SELECT",
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
            arg_dim=8,
        ),
        status=PrimitiveStatus.STABLE,
    )
    p_reverse = bank.new_reverse_relative_primitive(
        ReverseRelativePrimitiveConfig(
            d_model=32,
            d_operator=16,
            n_head=2,
            d_operator_ff=32,
            vocab_size=10,
            max_sequence_length=16,
        ),
        status=PrimitiveStatus.STABLE,
    )

    op_to_id = {
        "SHIFT": p_shift.primitive_id,
        "SELECT": p_select.primitive_id,
        "REVERSE": p_reverse.primitive_id,
    }

    # Generate examples for SHIFT(1) -> SELECT((0, 2, 4))
    step1 = ProgramStep("SHIFT", {"amount": 1})
    step2 = ProgramStep("SELECT", {"indices": (0, 2, 4)})
    prog = Program(steps=(step1, step2))

    examples: list[Example] = []
    for seq in [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1), (2, 4, 6, 8, 0, 1)]:
        res = run_program(prog, seq, 10)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="novel_composition",
            split="test",
            vocab_size=10,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(label="C", primitive_operations=("SHIFT", "SELECT")),
        )
        examples.append(ex)

    return core, bank, op_to_id, examples


def test_structural_validity_and_pruning() -> None:
    _, _, _, examples = _build_test_setup()

    # Valid candidate: SHIFT -> SELECT matches target length (3) and argument requirements
    assert is_candidate_structurally_valid(
        ("SHIFT", "SELECT"), examples, must_match_target_length=True
    )

    # Invalid: SHIFT alone does not match target length (output len 6 != 3)
    assert not is_candidate_structurally_valid(
        ("SHIFT",), examples, must_match_target_length=True
    )

    # Valid if must_match_target_length=False (it's a valid intermediate sequence)
    assert is_candidate_structurally_valid(
        ("SHIFT",), examples, must_match_target_length=False
    )

    # Invalid: REVERSE -> SHIFT produces length 6, not 3
    assert not is_candidate_structurally_valid(
        ("REVERSE", "SHIFT"), examples, must_match_target_length=True
    )

    # Invalid: COUNT requires 'target' argument, which is missing from examples
    assert not is_candidate_structurally_valid(
        ("REVERSE", "COUNT"), examples, must_match_target_length=False
    )


def test_is_candidate_prefix_valid() -> None:
    _, _, _, examples = _build_test_setup()

    # SHIFT preserves length 6 >= 2 -> Valid prefix
    assert is_candidate_prefix_valid(("SHIFT",), examples)

    # REVERSE preserves length 6 >= 2 -> Valid prefix
    assert is_candidate_prefix_valid(("REVERSE",), examples)

    # SELECT contracts length 6 -> 2 >= 2 -> Valid prefix
    assert is_candidate_prefix_valid(("SELECT",), examples)


def test_resolve_candidate_calls_no_oracle_leakage() -> None:
    _, _, _, examples = _build_test_setup()

    # Mask the operations in task_spec to verify zero oracle operation identity leakage
    masked_examples: list[Example] = []
    for ex in examples:
        masked_steps = tuple(
            TaskStepSpec(operation="UNKNOWN", arguments=dict(s.arguments))
            for s in ex.task_spec.steps
        )
        masked_ex = Example(
            input_tokens=ex.input_tokens,
            target_tokens=ex.target_tokens,
            program=ex.program,
            operation_graph=ex.operation_graph,
            category=ex.category,
            split=ex.split,
            vocab_size=ex.vocab_size,
            task_spec=TaskSpec(steps=masked_steps),
            oracle_metadata=ex.oracle_metadata,
        )
        masked_examples.append(masked_ex)

    # Resolve candidate calls without any operation labels
    calls = resolve_candidate_calls(("SHIFT", "SELECT"), masked_examples)
    assert len(calls) == len(masked_examples)
    for ex_calls in calls:
        assert len(ex_calls) == 2
        assert ex_calls[0].operation == "SHIFT"
        assert ex_calls[0].arguments == {"amount": 1}
        assert ex_calls[1].operation == "SELECT"
        assert ex_calls[1].arguments == {"indices": (0, 2, 4)}


def test_composition_search_execution_and_zero_expansion() -> None:
    core, bank, op_to_id, examples = _build_test_setup()

    # Record initial bank state to audit zero expansion
    initial_bank_len = len(bank)
    initial_param_count = sum(p.numel() for p in bank.parameters())

    # Run composition search over available operations
    result = search_composition_recipe(
        core,
        bank,
        op_to_id,
        examples,
        available_operations=("SHIFT", "SELECT", "REVERSE"),
        max_depth=2,
        beam_width=8,
    )

    valid_struct_candidates = (
        ("SHIFT", "SELECT"),
        ("REVERSE", "SELECT"),
        ("SELECT",),
        ("SELECT", "SHIFT"),
        ("SELECT", "REVERSE"),
    )
    assert result.candidate_operations in valid_struct_candidates
    assert result.candidates_evaluated > 0
    assert result.candidates_pruned > 0
    assert result.search_time_seconds > 0.0

    # Invariant: Zero bank expansion
    assert len(bank) == initial_bank_len
    assert sum(p.numel() for p in bank.parameters()) == initial_param_count
