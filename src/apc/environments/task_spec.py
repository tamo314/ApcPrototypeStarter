"""Explicit, model-visible task specification (Phase A.1 Correction Task A1-C001).

`docs/DECISIONS.md` ADR-0017 established that four of the eight known
operations (`SELECT`, `COUNT`, `SHIFT`, `BIND`) sample a hidden per-instance
parameter (`indices`, `target`, `amount`, `query_key`) that determines the
target together with the input content, yet that parameter -- and even
*which* operation was chosen at all (ADR-0020) -- was never included in
`Example.input_tokens`: it lived only in `Example.program`/`operation_graph`,
metadata this codebase treats as latent/evaluation-only (see
`apc.environments.generator`'s module docstring). No learner, however
capable, can recover a hidden, input-independent value from the input alone,
so Task A1-006 could only gate a single operation at a time, trained as a
separate model per operation (H1a).

`TaskSpec` is the fix's data representation: an explicit, model-visible
projection of a `Program` that carries operation identity plus every
argument `apc.environments.operations.Operation.sample_params` would
otherwise hide. `apc.environments.generator.TaskGenerator` attaches one to
every generated `Example` (`Example.task_spec`), built purely from that
example's own `Program` -- so, together with the presented content,
`TaskSpec` fully determines the target, by construction (`to_program()` plus
`apc.environments.interpreter.run_program` reproduces exactly what generated
the example; see `tests/test_task_spec.py`).

`TaskSpec` is *not yet* threaded into `apc.core.data.encode_example` -- that
is Task A1-C002 (mixed-operation online generator) / A1-C003 (shared-core
input encoding)'s job, not this one. This module only defines the
representation and wires it onto generated examples.

Deliberately excluded from `TaskSpec`: everything `apc.environments.
generator.OracleMetadata` carries (the K/C/N/R evaluation label and
recurrence-operation bookkeeping). Those describe how an example is used
for *evaluation*, not what computation produced its target, so they must
stay oracle-only and must never appear here (`docs/
AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`, "No hidden control variables" /
"Oracle-only metadata may remain hidden only in experiments explicitly
labeled oracle").

Naming note: `docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md` illustrates
`PrimitiveCall(SELECT, {"index": 3})` -- a single scalar index. The actual
`apc.environments.operations.SelectOp` instead samples `indices`, an
`output_length(len(sequence))`-sized *subset* of positions (never a single
index), because that is the real hidden parameter that determines its
output. `TaskStepSpec.arguments` mirrors `ProgramStep.params` exactly (key
for key) rather than the design doc's simplified illustration, so it
actually satisfies "task specification fully determines all previously
hidden operation parameters." Task A1-C006 (parameterized `PrimitiveCall`
abstraction) should resolve this naming/shape gap when it decides how
`SELECT` is exposed as a reusable primitive family; it is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apc.environments.operations import registered_operation_names
from apc.environments.program import Program, ProgramStep

_OPERATION_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(registered_operation_names())
}


def operation_id(name: str) -> int:
    """Stable integer identity for a registered operation name.

    Stable because `apc.environments.operations` registers every operation
    at import time, in a fixed order, and never deregisters one -- the same
    process therefore always maps a given name to the same id. Not
    guaranteed stable *across code versions* (adding a new operation shifts
    nothing already registered, since `_OPERATION_INDEX` is built once from
    registration order and existing entries keep their index, but removing
    or reordering registration would).
    """
    try:
        return _OPERATION_INDEX[name]
    except KeyError:
        raise KeyError(f"'{name}' is not a registered operation") from None


@dataclass(frozen=True)
class TaskStepSpec:
    """Model-visible specification of one program step: operation + arguments.

    `arguments` is exactly `ProgramStep.params` for the same step -- every
    key is one of that operation's `Operation.sample_params` outputs -- but
    `TaskStepSpec` is a distinct type so callers reaching for a model-visible
    field cannot accidentally grab `Example.program`/`operation_graph`
    (oracle/eval-only by convention; see module docstring above) instead.
    """

    operation: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def operation_id(self) -> int:
        return operation_id(self.operation)

    @classmethod
    def from_program_step(cls, step: ProgramStep) -> TaskStepSpec:
        return cls(operation=step.operation, arguments=dict(step.params))

    def to_program_step(self) -> ProgramStep:
        return ProgramStep(operation=self.operation, params=dict(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "operation_id": self.operation_id,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class TaskSpec:
    """Model-visible task specification for one generated example.

    An ordered chain of `TaskStepSpec`, one per `Program` step -- the
    model-visible counterpart of `Example.program`. `TaskSpec.from_program`
    and `TaskStepSpec.to_program_step`/`to_program` are exact inverses of
    each other's information content (round-trips through `dict(...)`
    preserve every key/value `sample_params` produced), so a `TaskSpec`
    reconstructed program, replayed through `apc.environments.interpreter.
    run_program` on the same input, reproduces the original target exactly.
    """

    steps: tuple[TaskStepSpec, ...] = ()

    @property
    def operation_sequence(self) -> tuple[str, ...]:
        return tuple(step.operation for step in self.steps)

    @classmethod
    def from_program(cls, program: Program) -> TaskSpec:
        return cls(steps=tuple(TaskStepSpec.from_program_step(step) for step in program.steps))

    def to_program(self) -> Program:
        return Program(steps=tuple(step.to_program_step() for step in self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}
