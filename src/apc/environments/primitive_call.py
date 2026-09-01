"""Parameterized primitive-call abstraction (Phase A.1 Correction Task
A1-C006).

`docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md` motivates this module:
several known operations (`SHIFT`, `SELECT`, `COUNT`, `BIND`) are really one
reusable computation family plus an argument that configures a particular
instance, not one primitive per argument value (`SHIFT_1`, `SHIFT_2`, ... is
exactly what `docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`'s "Parameterized
primitive rule" forbids). `PrimitiveCall` is the executable, model-facing-
adjacent representation of that split: `operation` (equivalently
`primitive_id`) selects the family, `arguments` configures the instance.

This is deliberately *not* the same type as `apc.environments.task_spec.
TaskStepSpec`, even though the two currently carry the same information for
every known operation. `TaskStepSpec` is the model-*input* projection of a
`Program` step (rendered into tokens by `apc.core.data.encode_task_spec`,
Task A1-C003); `PrimitiveCall` is the *execution-time* call a router/oracle
hands to whatever actually runs a primitive (`execute` below today; a future
oracle/learned routing path per `docs/exec-plans/active/
PHASE_A1_CORRECTION.md` A1-CM6 and Task A1-C007's "oracle PrimitiveCall
routing adapter"). Keeping them distinct types means a caller reaching for
"what should I execute" cannot accidentally grab the model-input-encoding
type instead, and vice versa. `from_task_step`/`to_task_step` convert
between them losslessly for every currently supported operation.

Naming/shape note (flagged by `apc.environments.task_spec`'s module
docstring as open when Task A1-C001 landed): the design doc's illustrative
`PrimitiveCall(SELECT, {"index": 3})` uses a single scalar index. The actual
`apc.environments.operations.SelectOp` samples `indices`, an
`output_length(len(sequence))`-sized *subset* of positions -- never a single
index -- because that is the real hidden parameter that determines its
output (a single index cannot, by itself, determine a variable-length
subset). Resolution: `PrimitiveCall` keeps `indices` (matching
`ProgramStep.params`/`TaskStepSpec.arguments` exactly, key for key) rather
than adopting the design doc's simplified single-scalar illustration, so
`execute()` below actually reproduces `SelectOp.apply`'s real output instead
of an underspecified approximation. See `docs/DECISIONS.md` ADR-0023.

Argument validation: `Operation.required_argument_names` (added by this
task, defaulting to empty/parameter-free) is the single source of truth for
which argument keys a `PrimitiveCall` must supply for a given operation.
`PrimitiveCall.__post_init__` checks this eagerly, at construction, the same
way `apc.primitives.primitive.PrimitiveConfig`/`apc.primitives.router.
RouterConfig` validate eagerly -- so a malformed call fails fast rather than
producing a confusing error deep inside `Operation.apply`.

Persistent-capacity acceptance (Task A1-C006): `primitive_id` is
`apc.environments.task_spec.operation_id(operation)` -- a function of
`operation` alone, so every `PrimitiveCall` for the same operation shares
one family id regardless of `arguments`. Minting `PrimitiveCall`s with many
distinct argument values never registers a new `Operation`/id
(`apc.environments.operations.registered_operation_names()` is fixed at
import time); see `tests/test_primitive_call.py` for the corresponding
regression test.

Composition (multiple chained `PrimitiveCall`s, e.g. the design doc's
`recipe = [PrimitiveCall(...), PrimitiveCall(...)]`) is explicitly out of
scope here -- that is Task A1-008's Composition Library. This module only
represents and executes one call at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apc.environments.operations import get_operation
from apc.environments.program import ProgramStep
from apc.environments.task_spec import TaskStepSpec, operation_id


@dataclass(frozen=True)
class PrimitiveCall:
    """A reusable primitive family (`operation`) plus the arguments that
    configure one execution instance.

    `arguments` must supply exactly `get_operation(operation).
    required_argument_names` -- no missing keys, no unexpected extras.
    Validated eagerly in `__post_init__`; constructing an invalid call
    raises immediately rather than failing later inside `execute`.

    Raises:
        KeyError: `operation` is not a registered
            `apc.environments.operations.Operation`.
        ValueError: `arguments` is missing a required key or supplies an
            argument the operation does not accept.
    """

    operation: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = get_operation(self.operation).required_argument_names
        provided = frozenset(self.arguments)
        missing = required - provided
        if missing:
            raise ValueError(
                f"PrimitiveCall for '{self.operation}' is missing required "
                f"argument(s): {sorted(missing)}"
            )
        extra = provided - required
        if extra:
            raise ValueError(
                f"PrimitiveCall for '{self.operation}' received unexpected "
                f"argument(s): {sorted(extra)} (accepts: {sorted(required)})"
            )

    @property
    def primitive_id(self) -> int:
        """Stable id of the primitive family this call selects --
        `apc.environments.task_spec.operation_id(self.operation)`. A
        function of `operation` alone, so it is invariant across every
        `arguments` value a family can take (see module docstring)."""
        return operation_id(self.operation)

    @classmethod
    def from_task_step(cls, step: TaskStepSpec) -> PrimitiveCall:
        return cls(operation=step.operation, arguments=dict(step.arguments))

    @classmethod
    def from_program_step(cls, step: ProgramStep) -> PrimitiveCall:
        return cls(operation=step.operation, arguments=dict(step.params))

    def to_task_step(self) -> TaskStepSpec:
        return TaskStepSpec(operation=self.operation, arguments=dict(self.arguments))

    def to_program_step(self) -> ProgramStep:
        return ProgramStep(operation=self.operation, params=dict(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "primitive_id": self.primitive_id,
            "arguments": dict(self.arguments),
        }

    def execute(self, sequence: tuple[int, ...], vocab_size: int) -> tuple[int, ...]:
        """Apply this call's operation with its arguments to `sequence`.

        A single-call, non-chaining sibling of
        `apc.environments.interpreter.run_program`: it does not build a
        `GraphNode`/trace, it just runs one `Operation.apply` with this
        call's own `arguments` -- the executable meaning of "primitive
        selection plus arguments configure the computation instance"
        (`docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md` section 2).

        Raises:
            ValueError: `sequence` is not a valid length for this
                operation (e.g. `BIND` on an odd-length sequence).
        """
        operation = get_operation(self.operation)
        if not operation.is_valid_for_length(len(sequence)):
            raise ValueError(
                f"Operation '{self.operation}' is not valid for input length "
                f"{len(sequence)}"
            )
        return operation.apply(sequence, vocab_size, self.arguments)
