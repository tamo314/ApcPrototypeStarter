"""Reference operations for the Phase A symbolic task environment.

Each operation is a small, pure, deterministic transform over a tuple of
integer tokens drawn from the shared vocabulary (see `vocab.py`). Operations
never depend on any neural-network code so the environment can be validated
without a model.

Composition chains operations: the output tokens of one operation become
the input tokens of the next (see `program.py` and `interpreter.py`).

Extension interface: `Operation` + `register_operation` is how a genuinely
novel operation is added without changing the interpreter. Task 009 uses it
for `SortOp`/`SORT` (see `NOVEL_OPERATION_NAMES` below); the interpreter
needed no changes, and `apc.environments.generator.TaskGenerator` reads
`NOVEL_OPERATION_NAMES` separately from `KNOWN_OPERATION_NAMES` to build a
dedicated, oracle-labeled `novel_operation` split.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from apc.environments.vocab import relation_tokens


class Operation(ABC):
    """A deterministic, parameterized transform over a token sequence."""

    name: str
    min_input_length: int

    def is_valid_for_length(self, input_length: int) -> bool:
        """Whether this operation can execute on a sequence of this length.

        Overridden by operations with structural constraints beyond a
        simple minimum (e.g. BIND requires an even-length key/value input).
        """
        return input_length >= self.min_input_length

    @abstractmethod
    def output_length(self, input_length: int) -> int:
        """Length of the output sequence for a given input length.

        Must be a pure function of `input_length` alone (not of content or
        randomness): the generator and `CompositionSpace` rely on this to
        decide, without running anything, whether a chain of operations can
        execute end to end.
        """

    @abstractmethod
    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        """Sample valid parameters for this operation given the current state.

        Receives the actual current sequence (not just its length) so
        content-dependent operations (e.g. BIND's lookup key) can sample a
        parameter that is guaranteed well-defined.
        """

    @abstractmethod
    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        """Execute the operation, returning the output token sequence.

        Pure and deterministic given `sequence`, `vocab_size`, and `params`.
        """


class CopyOp(Operation):
    name = "COPY"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        return tuple(sequence)


class NegateOp(Operation):
    name = "NEGATE"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        return tuple(vocab_size - 1 - token for token in sequence)


class ShiftOp(Operation):
    name = "SHIFT"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {"amount": rng.randrange(len(sequence))}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        amount = params["amount"] % len(sequence)
        return sequence[amount:] + sequence[:amount]


class SelectOp(Operation):
    """Select a deterministic-count subset of positions, preserving order.

    The number of positions selected is a fixed function of the input
    length (`output_length`), not an independent random draw, so a
    `CompositionSpace` chain-length check stays valid for the params
    actually sampled at generation time.
    """

    name = "SELECT"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return max(1, input_length // 2)

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        count = self.output_length(len(sequence))
        indices = sorted(rng.sample(range(len(sequence)), count))
        return {"indices": indices}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        return tuple(sequence[i] for i in params["indices"])


class CompareOp(Operation):
    """Emit a relation token (LT/EQ/GT) for each adjacent pair of tokens."""

    name = "COMPARE"
    min_input_length = 2

    def output_length(self, input_length: int) -> int:
        return input_length - 1

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        lt, eq, gt = relation_tokens(vocab_size)
        result = []
        for a, b in zip(sequence, sequence[1:], strict=False):
            if a < b:
                result.append(lt)
            elif a == b:
                result.append(eq)
            else:
                result.append(gt)
        return tuple(result)


class CountOp(Operation):
    """Count occurrences of a sampled target token, clipped to the vocabulary."""

    name = "COUNT"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return 1

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {"target": rng.randrange(vocab_size)}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        target = params["target"]
        count = sum(1 for token in sequence if token == target)
        return (min(count, vocab_size - 1),)


class BindOp(Operation):
    """Associative lookup over the sequence read as (key, value) pairs.

    The query key is always sampled from a key actually present in the
    sequence, so the lookup is guaranteed well-defined. If a key repeats,
    the last matching pair wins (mirrors `dict.update` semantics).
    """

    name = "BIND"
    min_input_length = 2

    def is_valid_for_length(self, input_length: int) -> bool:
        return input_length >= self.min_input_length and input_length % 2 == 0

    def output_length(self, input_length: int) -> int:
        return 1

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        keys = sequence[0::2]
        return {"query_key": rng.choice(keys)}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        query_key = params["query_key"]
        value = 0
        for i in range(0, len(sequence) - 1, 2):
            if sequence[i] == query_key:
                value = sequence[i + 1]
        return (value,)


class AccumulateOp(Operation):
    """Running sum modulo the vocabulary size (prefix-sum style)."""

    name = "ACCUMULATE"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        total = 0
        result = []
        for token in sequence:
            total = (total + token) % vocab_size
            result.append(total)
        return tuple(result)


class SortOp(Operation):
    """Ascending sort of the whole sequence.

    Deliberately outside the Phase A known curriculum (Task 009): none of
    COPY/SELECT/COMPARE/COUNT/SHIFT/BIND/NEGATE/ACCUMULATE can reorder
    tokens by value (SHIFT only rotates, SELECT only subsets without
    reordering), so a global ascending reordering cannot be assembled from
    a bounded-depth chain of them. See `NOVEL_OPERATION_NAMES` below.
    """

    name = "SORT"
    min_input_length = 1

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        return tuple(sorted(sequence))


_REGISTRY: dict[str, Operation] = {}


def register_operation(operation: Operation, *, overwrite: bool = False) -> None:
    """Register an operation by name so the generator/interpreter can find it.

    This is the extension point for later, genuinely novel operations
    (Phase A Task 009): implement `Operation` and call this function.
    """
    if not overwrite and operation.name in _REGISTRY:
        raise ValueError(f"Operation '{operation.name}' is already registered")
    _REGISTRY[operation.name] = operation


def get_operation(name: str) -> Operation:
    """Look up a registered operation by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown operation: {name!r}") from None


def registered_operation_names() -> tuple[str, ...]:
    """All currently registered operation names, in registration order."""
    return tuple(_REGISTRY.keys())


for _op in (
    CopyOp(),
    SelectOp(),
    CompareOp(),
    CountOp(),
    ShiftOp(),
    BindOp(),
    NegateOp(),
    AccumulateOp(),
):
    register_operation(_op)

# The Phase A initial curriculum (AGENTS.md, PHASE_A.md Milestone A1). Novel
# operations added later (Task 009) are registered separately and must not
# be appended to this tuple.
KNOWN_OPERATION_NAMES: tuple[str, ...] = tuple(_REGISTRY.keys())

# Registered strictly after KNOWN_OPERATION_NAMES is captured, so it can
# never be silently absorbed into the known-composition budget (Task 009,
# PHASE_A.md Milestone A6/ARCHITECTURE.md section 13 "Novel operation
# tasks"). `apc.environments.generator.TaskGenerator` uses this tuple to
# build the dedicated `novel_operation` split, oracle-labeled `N`.
register_operation(SortOp())
NOVEL_OPERATION_NAMES: tuple[str, ...] = ("SORT",)
