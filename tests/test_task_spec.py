"""Tests for the explicit task specification schema (Phase A.1 Correction
Task A1-C001).

Covers: operation identity encoding, argument round-tripping against
`ProgramStep`/`Program`, that identical content with a different task
specification correctly yields a different target (via the unmodified
reference interpreter), and that `TaskSpec` never carries `OracleMetadata`'s
oracle-only fields.
"""

from __future__ import annotations

import dataclasses

import pytest

from apc.environments.generator import OracleMetadata
from apc.environments.interpreter import run_program
from apc.environments.operations import KNOWN_OPERATION_NAMES, registered_operation_names
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec, TaskStepSpec, operation_id

VOCAB_SIZE = 10


# --- operation identity -----------------------------------------------------


def test_operation_id_matches_registration_order() -> None:
    for index, name in enumerate(registered_operation_names()):
        assert operation_id(name) == index


def test_operation_id_is_unique_per_operation() -> None:
    ids = [operation_id(name) for name in registered_operation_names()]
    assert len(ids) == len(set(ids))


def test_operation_id_rejects_unknown_operation() -> None:
    with pytest.raises(KeyError):
        operation_id("NOT_A_REAL_OPERATION")


def test_task_step_spec_operation_id_matches_module_level_helper() -> None:
    step = TaskStepSpec(operation="SHIFT", arguments={"amount": 2})
    assert step.operation_id == operation_id("SHIFT")


# --- TaskStepSpec / TaskSpec <-> ProgramStep / Program round trip ----------


@pytest.mark.parametrize(
    ("operation", "params"),
    [
        ("COPY", {}),
        ("NEGATE", {}),
        ("COMPARE", {}),
        ("ACCUMULATE", {}),
        ("SHIFT", {"amount": 3}),
        ("SELECT", {"indices": [0, 2, 4]}),
        ("COUNT", {"target": 7}),
        ("BIND", {"query_key": 5}),
    ],
)
def test_task_step_spec_round_trips_every_known_operation(
    operation: str, params: dict
) -> None:
    step = ProgramStep(operation=operation, params=params)
    spec = TaskStepSpec.from_program_step(step)
    assert spec.operation == operation
    assert spec.arguments == params
    assert spec.to_program_step() == step


def test_task_spec_round_trips_a_multi_step_program() -> None:
    program = Program(
        steps=(
            ProgramStep(operation="NEGATE", params={}),
            ProgramStep(operation="SHIFT", params={"amount": 2}),
        )
    )
    spec = TaskSpec.from_program(program)
    assert spec.operation_sequence == ("NEGATE", "SHIFT")
    assert spec.to_program() == program


def test_task_spec_from_empty_program_round_trips() -> None:
    program = Program(steps=())
    spec = TaskSpec.from_program(program)
    assert spec.steps == ()
    assert spec.to_program() == program


def test_task_spec_to_dict_is_json_serializable_shape() -> None:
    program = Program(steps=(ProgramStep(operation="COUNT", params={"target": 4}),))
    payload = TaskSpec.from_program(program).to_dict()
    assert payload == {
        "steps": [
            {
                "operation": "COUNT",
                "operation_id": operation_id("COUNT"),
                "arguments": {"target": 4},
            }
        ]
    }


# --- acceptance: task specification fully determines all previously hidden
# --- operation parameters ---------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "params"),
    [
        ("SHIFT", {"amount": 3}),
        ("SELECT", {"indices": [1, 3, 5]}),
        ("COUNT", {"target": 4}),
        ("BIND", {"query_key": 5}),
    ],
)
def test_task_spec_reconstructed_program_reproduces_original_output(
    operation: str, params: dict
) -> None:
    """`TaskSpec` must carry every parameter `Operation.sample_params` would
    otherwise hide (ADR-0017): replaying the reconstructed program on the
    same content must reproduce exactly what generated the example, not an
    approximation."""
    content = (5, 1, 6, 2, 7, 3)
    program = Program(steps=(ProgramStep(operation=operation, params=params),))
    original = run_program(program, content, VOCAB_SIZE)

    spec = TaskSpec.from_program(program)
    replay = run_program(spec.to_program(), content, VOCAB_SIZE)

    assert replay.output_tokens == original.output_tokens


# --- acceptance: two examples with identical content but different task
# --- specification can correctly have different targets --------------------


def test_identical_content_different_shift_amount_yields_different_targets() -> None:
    content = (1, 2, 3, 4, 5)
    spec_a = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 1}),))
    spec_b = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 3}),))

    result_a = run_program(spec_a.to_program(), content, VOCAB_SIZE)
    result_b = run_program(spec_b.to_program(), content, VOCAB_SIZE)

    assert result_a.output_tokens != result_b.output_tokens
    assert result_a.output_tokens == (2, 3, 4, 5, 1)
    assert result_b.output_tokens == (4, 5, 1, 2, 3)


def test_identical_content_different_count_target_yields_different_targets() -> None:
    content = (3, 1, 3, 3, 2)
    spec_a = TaskSpec(steps=(TaskStepSpec(operation="COUNT", arguments={"target": 3}),))
    spec_b = TaskSpec(steps=(TaskStepSpec(operation="COUNT", arguments={"target": 1}),))

    result_a = run_program(spec_a.to_program(), content, VOCAB_SIZE)
    result_b = run_program(spec_b.to_program(), content, VOCAB_SIZE)

    assert result_a.output_tokens != result_b.output_tokens
    assert result_a.output_tokens == (3,)
    assert result_b.output_tokens == (1,)


def test_identical_content_different_bind_query_key_yields_different_targets() -> None:
    content = (1, 8, 2, 9, 3, 7)  # keys 1,2,3 -> values 8,9,7
    spec_a = TaskSpec(steps=(TaskStepSpec(operation="BIND", arguments={"query_key": 1}),))
    spec_b = TaskSpec(steps=(TaskStepSpec(operation="BIND", arguments={"query_key": 3}),))

    result_a = run_program(spec_a.to_program(), content, VOCAB_SIZE)
    result_b = run_program(spec_b.to_program(), content, VOCAB_SIZE)

    assert result_a.output_tokens != result_b.output_tokens
    assert result_a.output_tokens == (8,)
    assert result_b.output_tokens == (7,)


def test_identical_content_different_select_indices_yields_different_targets() -> None:
    content = (0, 1, 2, 3)
    spec_a = TaskSpec(steps=(TaskStepSpec(operation="SELECT", arguments={"indices": [0, 1]}),))
    spec_b = TaskSpec(steps=(TaskStepSpec(operation="SELECT", arguments={"indices": [2, 3]}),))

    result_a = run_program(spec_a.to_program(), content, VOCAB_SIZE)
    result_b = run_program(spec_b.to_program(), content, VOCAB_SIZE)

    assert result_a.output_tokens != result_b.output_tokens
    assert result_a.output_tokens == (0, 1)
    assert result_b.output_tokens == (2, 3)


# --- acceptance: no model-facing path reads oracle-only fields -------------


def test_task_step_spec_fields_are_disjoint_from_oracle_metadata_fields() -> None:
    """Structural guard: `TaskSpec` must stay strictly separated from
    `OracleMetadata` -- it may determine the target (operation + arguments)
    but must never carry the K/C/N/R evaluation label or recurrence-operation
    bookkeeping, which are oracle-only (see this module's and `apc.
    environments.generator`'s docstrings)."""
    oracle_fields = {f.name for f in dataclasses.fields(OracleMetadata)}
    task_step_fields = {f.name for f in dataclasses.fields(TaskStepSpec)}
    assert oracle_fields.isdisjoint(task_step_fields)


def test_known_operation_names_are_all_registered_and_id_encodable() -> None:
    for name in KNOWN_OPERATION_NAMES:
        assert isinstance(operation_id(name), int)
