"""Tests for the parameterized `PrimitiveCall` abstraction (Phase A.1
Correction Task A1-C006).

Covers: `Operation.required_argument_names` as the validation source of
truth, `PrimitiveCall`'s eager missing/extra-argument validation, lossless
round-trips to/from `TaskStepSpec`/`ProgramStep`, `execute()` reproducing
the reference interpreter for at least 3 distinct argument values per
parameterized operation, parameter-free calls for the deterministic
operations, and that persistent primitive/family count does not grow with
the number of distinct argument values exercised.
"""

from __future__ import annotations

import pytest

from apc.environments.operations import (
    DETERMINISTIC_OPERATION_NAMES,
    KNOWN_OPERATION_NAMES,
    get_operation,
    registered_operation_names,
)
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.program import ProgramStep
from apc.environments.task_spec import TaskStepSpec, operation_id

VOCAB_SIZE = 10

PARAMETERIZED_OPERATIONS: dict[str, str] = {
    "SHIFT": "amount",
    "SELECT": "indices",
    "COUNT": "target",
    "BIND": "query_key",
}


# --- Operation.required_argument_names --------------------------------------


def test_deterministic_operations_require_no_arguments() -> None:
    for name in DETERMINISTIC_OPERATION_NAMES:
        assert get_operation(name).required_argument_names == frozenset()


@pytest.mark.parametrize(("name", "key"), PARAMETERIZED_OPERATIONS.items())
def test_parameterized_operations_require_their_own_key(name: str, key: str) -> None:
    assert get_operation(name).required_argument_names == frozenset({key})


@pytest.mark.parametrize("name", KNOWN_OPERATION_NAMES)
def test_required_argument_names_matches_a_sampled_params_keys(name: str) -> None:
    """Regression guard mirroring `test_operations.py`'s determinism guards:
    `required_argument_names` must actually match what `sample_params`
    produces, or `PrimitiveCall` validation would silently drift from the
    real operation contract."""
    import random

    op = get_operation(name)
    seq = (1, 8, 2, 9, 3, 7)
    params = op.sample_params(random.Random(0), seq, VOCAB_SIZE)
    assert set(params) == op.required_argument_names


# --- PrimitiveCall construction / validation ---------------------------------


def test_parameter_free_call_accepts_empty_arguments() -> None:
    call = PrimitiveCall(operation="COPY")
    assert call.arguments == {}


@pytest.mark.parametrize("name", DETERMINISTIC_OPERATION_NAMES)
def test_every_deterministic_operation_accepts_empty_arguments(name: str) -> None:
    call = PrimitiveCall(operation=name)
    assert call.arguments == {}


def test_unknown_operation_raises_key_error() -> None:
    with pytest.raises(KeyError):
        PrimitiveCall(operation="NOT_A_REAL_OPERATION")


def test_missing_required_argument_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required"):
        PrimitiveCall(operation="SHIFT", arguments={})


def test_unexpected_argument_on_parameter_free_operation_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        PrimitiveCall(operation="COPY", arguments={"amount": 1})


def test_unexpected_extra_argument_alongside_required_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        PrimitiveCall(operation="SHIFT", arguments={"amount": 1, "bogus": 2})


def test_wrong_argument_name_for_parameterized_operation_raises_value_error() -> None:
    """SELECT requires `indices`, not `index` -- the design doc's simplified
    illustration is deliberately not what this call accepts (see module
    docstring / ADR-0023)."""
    with pytest.raises(ValueError):
        PrimitiveCall(operation="SELECT", arguments={"index": 3})


# --- primitive_id -------------------------------------------------------------


def test_primitive_id_matches_operation_id() -> None:
    for name in KNOWN_OPERATION_NAMES:
        call = PrimitiveCall(operation=name, arguments=_sample_valid_arguments(name))
        assert call.primitive_id == operation_id(name)


def test_primitive_id_is_argument_invariant() -> None:
    """Different `arguments` for the same operation must select the same
    family id -- the whole point of separating selection from arguments."""
    ids = {
        PrimitiveCall(operation="SHIFT", arguments={"amount": amount}).primitive_id
        for amount in (0, 1, 2, 3, 4)
    }
    assert ids == {operation_id("SHIFT")}


# --- round trip to/from TaskStepSpec / ProgramStep ---------------------------


@pytest.mark.parametrize(
    ("operation", "arguments"),
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
def test_round_trips_every_known_operation_through_task_step_spec(
    operation: str, arguments: dict
) -> None:
    call = PrimitiveCall(operation=operation, arguments=arguments)
    step = call.to_task_step()
    assert step == TaskStepSpec(operation=operation, arguments=arguments)
    assert PrimitiveCall.from_task_step(step) == call


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("COPY", {}),
        ("SHIFT", {"amount": 3}),
        ("SELECT", {"indices": [0, 2, 4]}),
        ("COUNT", {"target": 7}),
        ("BIND", {"query_key": 5}),
    ],
)
def test_round_trips_through_program_step(operation: str, arguments: dict) -> None:
    call = PrimitiveCall(operation=operation, arguments=arguments)
    step = call.to_program_step()
    assert step == ProgramStep(operation=operation, params=arguments)
    assert PrimitiveCall.from_program_step(step) == call


def test_to_dict_is_json_serializable_shape() -> None:
    call = PrimitiveCall(operation="COUNT", arguments={"target": 4})
    assert call.to_dict() == {
        "operation": "COUNT",
        "primitive_id": operation_id("COUNT"),
        "arguments": {"target": 4},
    }


# --- execute(): correctness across >=3 distinct argument values -------------


def test_shift_executes_correctly_across_distinct_amounts() -> None:
    seq = (0, 1, 2, 3, 4)
    expected = {
        0: (0, 1, 2, 3, 4),
        1: (1, 2, 3, 4, 0),
        2: (2, 3, 4, 0, 1),
        4: (4, 0, 1, 2, 3),
    }
    for amount, want in expected.items():
        call = PrimitiveCall(operation="SHIFT", arguments={"amount": amount})
        assert call.execute(seq, VOCAB_SIZE) == want


def test_select_executes_correctly_across_distinct_index_subsets() -> None:
    seq = (10, 11, 12, 13)
    expected = {
        (0, 1): (10, 11),
        (2, 3): (12, 13),
        (0, 2): (10, 12),
        (1, 3): (11, 13),
    }
    for indices, want in expected.items():
        call = PrimitiveCall(operation="SELECT", arguments={"indices": list(indices)})
        assert call.execute(seq, VOCAB_SIZE) == want


def test_count_executes_correctly_across_distinct_targets() -> None:
    seq = (3, 1, 3, 3, 2)
    expected = {3: (3,), 1: (1,), 2: (1,), 9: (0,)}
    for target, want in expected.items():
        call = PrimitiveCall(operation="COUNT", arguments={"target": target})
        assert call.execute(seq, VOCAB_SIZE) == want


def test_bind_executes_correctly_across_distinct_keys() -> None:
    seq = (1, 8, 2, 9, 3, 7)  # keys 1,2,3 -> values 8,9,7
    expected = {1: (8,), 2: (9,), 3: (7,)}
    for query_key, want in expected.items():
        call = PrimitiveCall(operation="BIND", arguments={"query_key": query_key})
        assert call.execute(seq, VOCAB_SIZE) == want


@pytest.mark.parametrize("name", DETERMINISTIC_OPERATION_NAMES)
def test_parameter_free_execute_matches_direct_apply(name: str) -> None:
    seq = (1, 8, 2, 9, 3, 7)
    call = PrimitiveCall(operation=name)
    assert call.execute(seq, VOCAB_SIZE) == get_operation(name).apply(seq, VOCAB_SIZE, {})


def test_execute_on_invalid_length_raises_value_error() -> None:
    call = PrimitiveCall(operation="BIND", arguments={"query_key": 1})
    with pytest.raises(ValueError):
        call.execute((1, 8, 2), VOCAB_SIZE)  # odd length, BIND requires even


# --- persistent capacity: family/registry count is argument-invariant -------


def test_registered_operation_count_is_unaffected_by_many_distinct_calls() -> None:
    before = registered_operation_names()
    calls = [
        PrimitiveCall(operation="SHIFT", arguments={"amount": amount})
        for amount in range(50)
    ] + [
        PrimitiveCall(operation="COUNT", arguments={"target": target})
        for target in range(50)
    ]
    assert len(calls) == 100
    assert registered_operation_names() == before


def test_same_operation_instance_backs_every_argument_value() -> None:
    """No new `Operation` object is allocated per argument value -- every
    `PrimitiveCall` for one operation resolves to the same registered
    singleton (see `apc.environments.operations.get_operation`)."""
    shift_op = get_operation("SHIFT")
    for amount in range(5):
        PrimitiveCall(operation="SHIFT", arguments={"amount": amount})
        assert get_operation("SHIFT") is shift_op


def _sample_valid_arguments(operation: str) -> dict:
    defaults = {"amount": 1, "indices": [0], "target": 0, "query_key": 0}
    required = get_operation(operation).required_argument_names
    return {key: defaults[key] for key in required}
