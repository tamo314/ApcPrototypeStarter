"""Reference interpreter: pure, deterministic execution of a `Program`.

Given a fully-specified `Program` (operation names + concrete parameters)
and an input token sequence, `run_program` is a pure function: no
randomness and no neural-network code. It is the ground-truth executable
spec for every symbolic task in Phase A, and it also builds the latent
operation graph metadata that evaluation uses to distinguish known,
novel-composition, and (later) novel-operation examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apc.environments.operations import get_operation
from apc.environments.program import Program


@dataclass(frozen=True)
class GraphNode:
    """One node of the latent operation graph: an executed program step."""

    index: int
    operation: str
    params: dict[str, Any]
    input_length: int
    output_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation": self.operation,
            "params": dict(self.params),
            "input_length": self.input_length,
            "output_length": self.output_length,
        }


@dataclass(frozen=True)
class OperationGraph:
    """The latent operation graph for one example.

    Phase A programs are linear chains, so the graph is a path: node `-1`
    is the raw input, and each node feeds the next. Represented explicitly
    as nodes + edges (rather than an implicit list) so a future non-linear
    novel operation does not require a metadata-format change.
    """

    nodes: tuple[GraphNode, ...]

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        indices = [-1] + [node.index for node in self.nodes]
        return tuple(zip(indices[:-1], indices[1:], strict=True))

    @property
    def operation_sequence(self) -> tuple[str, ...]:
        return tuple(node.operation for node in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class InterpreterResult:
    """Full execution record: input, output, intermediate states, and graph."""

    input_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    trace: tuple[tuple[int, ...], ...]
    graph: OperationGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": list(self.input_tokens),
            "output_tokens": list(self.output_tokens),
            "trace": [list(state) for state in self.trace],
            "graph": self.graph.to_dict(),
        }


def run_program(
    program: Program, input_tokens: tuple[int, ...], vocab_size: int
) -> InterpreterResult:
    """Execute `program` on `input_tokens` and return the full trace + graph.

    Raises:
        ValueError: If a token falls outside `[0, vocab_size)`, or a step's
            operation is invalid for the length it receives.
    """
    if any(not (0 <= token < vocab_size) for token in input_tokens):
        raise ValueError(f"input_tokens contains a value outside [0, {vocab_size})")

    trace: list[tuple[int, ...]] = [tuple(input_tokens)]
    nodes: list[GraphNode] = []
    current = tuple(input_tokens)

    for index, step in enumerate(program.steps):
        operation = get_operation(step.operation)
        if not operation.is_valid_for_length(len(current)):
            raise ValueError(
                f"Operation '{operation.name}' is not valid for input length "
                f"{len(current)} at step {index}"
            )
        next_state = operation.apply(current, vocab_size, step.params)
        expected_length = operation.output_length(len(current))
        if len(next_state) != expected_length:
            raise ValueError(
                f"Operation '{operation.name}' produced length {len(next_state)}, "
                f"expected {expected_length}"
            )
        nodes.append(
            GraphNode(
                index=index,
                operation=operation.name,
                params=dict(step.params),
                input_length=len(current),
                output_length=len(next_state),
            )
        )
        current = next_state
        trace.append(current)

    return InterpreterResult(
        input_tokens=tuple(input_tokens),
        output_tokens=current,
        trace=tuple(trace),
        graph=OperationGraph(nodes=tuple(nodes)),
    )
