"""Reference interpreter tests: pure execution, graph metadata, error paths."""

from __future__ import annotations

import pytest

from apc.environments.interpreter import run_program
from apc.environments.program import Program, ProgramStep
from apc.environments.vocab import relation_tokens

VOCAB_SIZE = 10


def test_single_step_program_matches_direct_op_apply() -> None:
    program = Program(steps=(ProgramStep(operation="SHIFT", params={"amount": 1}),))
    result = run_program(program, (0, 1, 2, 3), VOCAB_SIZE)
    assert result.output_tokens == (1, 2, 3, 0)


def test_multi_step_program_chains_outputs_to_inputs() -> None:
    # NEGATE then SHIFT(1): negate(0,1,9,5) -> (9,8,0,4); shift left by 1 -> (8,0,4,9)
    program = Program(
        steps=(
            ProgramStep(operation="NEGATE", params={}),
            ProgramStep(operation="SHIFT", params={"amount": 1}),
        )
    )
    result = run_program(program, (0, 1, 9, 5), VOCAB_SIZE)
    assert result.output_tokens == (8, 0, 4, 9)
    assert result.trace == ((0, 1, 9, 5), (9, 8, 0, 4), (8, 0, 4, 9))


def test_result_is_deterministic_given_same_inputs() -> None:
    program = Program(
        steps=(
            ProgramStep(operation="COMPARE", params={}),
            ProgramStep(operation="COUNT", params={"target": relation_tokens(VOCAB_SIZE)[1]}),
        )
    )
    a = run_program(program, (1, 1, 2, 2, 2), VOCAB_SIZE)
    b = run_program(program, (1, 1, 2, 2, 2), VOCAB_SIZE)
    assert a.to_dict() == b.to_dict()


def test_graph_metadata_has_one_node_per_step_with_correct_lengths() -> None:
    program = Program(
        steps=(
            ProgramStep(operation="COPY", params={}),
            ProgramStep(operation="COMPARE", params={}),
        )
    )
    result = run_program(program, (1, 2, 3, 4), VOCAB_SIZE)
    graph = result.graph

    assert len(graph.nodes) == 2
    assert graph.operation_sequence == ("COPY", "COMPARE")

    assert graph.nodes[0].input_length == 4
    assert graph.nodes[0].output_length == 4
    assert graph.nodes[1].input_length == 4
    assert graph.nodes[1].output_length == 3

    assert graph.edges == ((-1, 0), (0, 1))


def test_empty_program_returns_input_unchanged() -> None:
    program = Program(steps=())
    result = run_program(program, (1, 2, 3), VOCAB_SIZE)
    assert result.output_tokens == (1, 2, 3)
    assert result.graph.nodes == ()
    assert result.graph.edges == ()


def test_token_outside_vocab_raises() -> None:
    program = Program(steps=(ProgramStep(operation="COPY", params={}),))
    with pytest.raises(ValueError):
        run_program(program, (0, 1, VOCAB_SIZE), VOCAB_SIZE)


def test_operation_invalid_for_length_raises() -> None:
    # COMPARE requires at least 2 tokens.
    program = Program(steps=(ProgramStep(operation="COMPARE", params={}),))
    with pytest.raises(ValueError):
        run_program(program, (5,), VOCAB_SIZE)


def test_bind_odd_length_raises() -> None:
    program = Program(steps=(ProgramStep(operation="BIND", params={"query_key": 1}),))
    with pytest.raises(ValueError):
        run_program(program, (1, 2, 3), VOCAB_SIZE)
