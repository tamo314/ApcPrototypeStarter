"""Deterministic-by-seed synthetic task generator for Phase A.

`TaskGenerator` samples compositions of known operations, executes them
through the reference interpreter, and returns `Example`s carrying the full
latent operation graph as ground-truth metadata. This metadata is for
evaluation only: the meta-controller (Milestone A6+) must never see it.

Splits:
- `train` / `val` / `test`: known-operation tasks (depth 1) and known
  compositions (depth > 1) that were assigned to the known pool.
- `novel_composition`: depth > 1 chains of *known* operations that were
  held out of the known pool entirely, for composition-generalization
  evaluation.
- `novel_operation`: depth-1 tasks built from `novel_operation_names`
  (Task 009, e.g. `apc.environments.operations.NOVEL_OPERATION_NAMES`) --
  operations that are not part of `operation_names` at all, so they cannot
  be produced by any known-operation composition regardless of depth. Only
  available when `TaskGenerator` is constructed with a non-empty
  `novel_operation_names`.

Symbol permutation (Task A1-004, `permute_symbols=True`): every `generate`/
`generate_online` call draws one fresh `apc.environments.permutation.
SymbolPermutation` shared by every example in that call (i.e. per batch/
episode -- a `generate_online` call is exactly one training step's batch)
and uses it to relabel canonical tokens before they are stored, so a stable
token id (e.g. a relation token) cannot become a memorizable proxy for a
fixed semantic role across the *whole run*. It is deliberately not one
permutation per example: that would leave presented token ids with no
value/order relationship a learner could exploit *across* examples at all,
making arithmetic/order-dependent operations (e.g. NEGATE, COMPARE,
ACCUMULATE) unlearnable regardless of training -- see `docs/DECISIONS.md`
ADR-0018. See `Example.symbol_permutation`.

Explicit task specification (Phase A.1 Correction Task A1-C001,
`Example.task_spec`): every example also carries an
`apc.environments.task_spec.TaskSpec`, a model-visible projection of
`program` that exposes operation identity and every previously hidden
operation parameter (`SELECT`'s `indices`, `COUNT`'s `target`, `SHIFT`'s
`amount`, `BIND`'s `query_key` -- see `docs/DECISIONS.md` ADR-0017). Unlike
`program`/`operation_graph`/`oracle_metadata`, which this module documents
as latent/oracle-only, `task_spec` is intended for a future model-facing
input encoding (Task A1-C003); `apc.core.data.encode_example` does not yet
consume it.

Mixed-operation online generator (Phase A.1 Correction Task A1-C002,
`build_mixed_operation_generator`): a named, tested construction of "one
identifiable mixed-operation training stream" -- a single `TaskGenerator`
whose known pool spans every operation in `operation_names` (all eight of
`KNOWN_OPERATION_NAMES` by default, including the four ADR-0017
hidden-parameter operations) at a pinned `max_depth=1`, so every generated
example is a single known operation applied to fresh content, with its
operation identity and arguments fully recoverable from `task_spec`. Before
A1-C001, pooling multiple operations in one `TaskGenerator` produced a
mixed stream that was *not* identifiable this way -- `docs/DECISIONS.md`
ADR-0020 measured a ~0.25-0.35 exact-match ceiling from exactly this
non-identifiability, which is why `apc.evaluation.stable_core_generalization`
still trains one operation at a time. `task_spec` (once wired into a
model-facing encoding, Task A1-C003) is the fix; `build_mixed_operation_generator`
is the corresponding generator-side building block for Task A1-C004's
shared-core gate.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Final, Literal

from apc.environments.interpreter import OperationGraph, run_program
from apc.environments.operations import KNOWN_OPERATION_NAMES, get_operation
from apc.environments.permutation import SymbolPermutation, sample_permutation
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE

KNOWN_SPLITS: tuple[str, ...] = ("train", "val", "test")
NOVEL_COMPOSITION_SPLIT = "novel_composition"
NOVEL_OPERATION_SPLIT = "novel_operation"

OracleLabel = Literal["K", "C", "N", "R"]
ORACLE_LABEL_KNOWN: Final[OracleLabel] = "K"
ORACLE_LABEL_NOVEL_COMPOSITION: Final[OracleLabel] = "C"
ORACLE_LABEL_NOVEL_OPERATION: Final[OracleLabel] = "N"
ORACLE_LABEL_RECURRENCE: Final[OracleLabel] = "R"


@dataclass(frozen=True)
class OracleMetadata:
    """Latent task information reserved for evaluation and oracle paths.

    This record deliberately contains no model-ready token encoding.  In
    particular, :func:`apc.core.data.encode_example` consumes only
    ``input_tokens`` and ``target_tokens``.  Future learned routing and
    novelty code must derive their inputs from the task itself; only explicit
    oracle experiments may consume this metadata.
    """

    label: OracleLabel
    primitive_operations: tuple[str, ...]
    recurrence_operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable oracle metadata for evaluation logs."""
        return {
            "label": self.label,
            "primitive_operations": list(self.primitive_operations),
            "recurrence_operation": self.recurrence_operation,
        }


def _derive_seed(seed: int, label: str) -> int:
    """Deterministically derive a sub-seed, independent of PYTHONHASHSEED.

    `random.Random` can be seeded from an arbitrary hashable, but Python's
    built-in `hash()` of strings is salted per-process unless
    `PYTHONHASHSEED` is fixed. Using `hashlib` instead keeps generation
    reproducible across processes for a given (seed, label) pair.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Example:
    """One generated task instance with its ground-truth operation graph.

    `program` and `operation_graph` always describe the operation semantics
    in the canonical vocabulary that `apc.environments.interpreter.run_program`
    actually executed. When `symbol_permutation` is set, `input_tokens` and
    `target_tokens` are the *presented* (permuted) ids the model sees;
    `symbol_permutation.invert(...)` recovers the canonical tokens that
    `program`/`operation_graph` refer to (Task A1-004).

    `task_spec` is the model-visible counterpart of `program` (Task
    A1-C001): the same operation identity and arguments, but carried in a
    type distinct from the latent `program`/`operation_graph`/
    `oracle_metadata` fields so a future model-facing input encoding has an
    unambiguous field to read from.
    """

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    program: Program
    operation_graph: OperationGraph
    category: str  # "known" | "novel_composition" | "novel_operation"
    split: str
    vocab_size: int
    task_spec: TaskSpec | None = None
    oracle_metadata: OracleMetadata | None = None
    symbol_permutation: SymbolPermutation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": list(self.input_tokens),
            "target_tokens": list(self.target_tokens),
            "program": self.program.to_dict(),
            "operation_graph": self.operation_graph.to_dict(),
            "category": self.category,
            "split": self.split,
            "vocab_size": self.vocab_size,
            "task_spec": (None if self.task_spec is None else self.task_spec.to_dict()),
            "oracle_metadata": (
                None if self.oracle_metadata is None else self.oracle_metadata.to_dict()
            ),
            "symbol_permutation": (
                None if self.symbol_permutation is None else self.symbol_permutation.to_dict()
            ),
        }


def _valid_lengths_for_chain(
    operation_names: tuple[str, ...], length_range: tuple[int, int]
) -> tuple[int, ...]:
    """Input lengths in `length_range` for which the whole chain executes."""
    min_len, max_len = length_range
    valid = []
    for length in range(min_len, max_len + 1):
        current = length
        ok = True
        for name in operation_names:
            operation = get_operation(name)
            if not operation.is_valid_for_length(current):
                ok = False
                break
            current = operation.output_length(current)
        if ok:
            valid.append(length)
    return tuple(valid)


def enumerate_compositions(
    operation_names: tuple[str, ...],
    max_depth: int,
    length_range: tuple[int, int],
) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Enumerate every operation chain up to `max_depth`.

    Returns a mapping from operation-name chain to the (nonempty) subset of
    `length_range` for which that chain can execute end to end. Chains with
    no valid length anywhere in the range are omitted.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    entries: dict[tuple[str, ...], tuple[int, ...]] = {}
    for depth in range(1, max_depth + 1):
        for combo in product(operation_names, repeat=depth):
            lengths = _valid_lengths_for_chain(combo, length_range)
            if lengths:
                entries[combo] = lengths
    return entries


class CompositionSpace:
    """Enumerates valid operation chains and splits multi-op chains into a
    known-composition pool (train/val/test) and a held-out novel pool.

    Single-operation chains (depth 1) are always "known": they are exactly
    the known-operation tasks, never a composition.
    """

    def __init__(
        self,
        operation_names: tuple[str, ...],
        max_depth: int,
        length_range: tuple[int, int],
        seed: int,
        novel_composition_fraction: float = 0.3,
    ) -> None:
        if not 0.0 < novel_composition_fraction < 1.0:
            raise ValueError("novel_composition_fraction must be in (0, 1)")

        entries = enumerate_compositions(operation_names, max_depth, length_range)
        single_op = sorted(c for c in entries if len(c) == 1)
        multi_op = sorted(c for c in entries if len(c) > 1)
        if not single_op:
            raise ValueError(
                "No valid single-operation compositions for the given length_range; "
                "widen sequence_length_range"
            )

        rng = random.Random(_derive_seed(seed, "composition_split"))
        rng.shuffle(multi_op)
        n_novel = max(1, round(len(multi_op) * novel_composition_fraction)) if multi_op else 0
        novel = tuple(sorted(multi_op[:n_novel]))
        known_multi = tuple(sorted(multi_op[n_novel:]))

        self.known_compositions: tuple[tuple[str, ...], ...] = tuple(single_op) + known_multi
        self.novel_compositions: tuple[tuple[str, ...], ...] = novel
        self.valid_lengths: dict[tuple[str, ...], tuple[int, ...]] = entries


class TaskGenerator:
    """Deterministic-by-seed generator of Phase A symbolic task examples.

    Calling `generate(n, split)` twice with the same `seed`, config, and
    `split` reproduces byte-identical examples, independent of call order
    or of any other split having been generated first.  The historical
    method serves fixed-dataset diagnostics.  Phase A.1 scientific paths use
    :meth:`generate_online`, which samples a fresh batch deterministically
    from ``(seed, step, split)`` without retaining a finite train set.

    When `permute_symbols` is `True` (Task A1-004), every example additionally
    gets its own fresh, deterministic-by-seed `SymbolPermutation`: operation
    semantics are still computed by the interpreter on canonical tokens, but
    `input_tokens`/`target_tokens` are relabeled through that permutation
    before being stored, so no token id is a stable proxy for a fixed role
    (e.g. a relation token) across examples. Off by default so `generate`'s
    existing fixed-dataset consumers see unchanged output.
    """

    def __init__(
        self,
        seed: int,
        operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        sequence_length_range: tuple[int, int] = (6, 10),
        max_depth: int = 2,
        novel_composition_fraction: float = 0.3,
        novel_operation_names: tuple[str, ...] = (),
        permute_symbols: bool = False,
    ) -> None:
        self.seed = seed
        self.operation_names = operation_names
        self.vocab_size = vocab_size
        self.sequence_length_range = sequence_length_range
        self.novel_operation_names = novel_operation_names
        self.permute_symbols = permute_symbols
        self.composition_space = CompositionSpace(
            operation_names=operation_names,
            max_depth=max_depth,
            length_range=sequence_length_range,
            seed=seed,
            novel_composition_fraction=novel_composition_fraction,
        )
        # Task 009: novel-operation tasks are always depth-1 (a bare
        # application of one genuinely novel operation, never composed with
        # known operations), so they cannot be confused with a depth>1
        # `novel_composition` chain and stay unambiguously outside the
        # known-composition budget regardless of `max_depth` above.
        self._novel_operation_lengths = (
            enumerate_compositions(novel_operation_names, 1, sequence_length_range)
            if novel_operation_names
            else {}
        )
        self._novel_operation_pool: tuple[tuple[str, ...], ...] = tuple(
            sorted(self._novel_operation_lengths)
        )

    def _pool_lengths_and_category(
        self, split: str
    ) -> tuple[tuple[tuple[str, ...], ...], dict[tuple[str, ...], tuple[int, ...]], str]:
        if split in KNOWN_SPLITS:
            return (
                self.composition_space.known_compositions,
                self.composition_space.valid_lengths,
                "known",
            )
        if split == NOVEL_COMPOSITION_SPLIT:
            return (
                self.composition_space.novel_compositions,
                self.composition_space.valid_lengths,
                "novel_composition",
            )
        if split == NOVEL_OPERATION_SPLIT:
            if not self._novel_operation_pool:
                raise ValueError(
                    f"novel_operation_names must be non-empty to generate {NOVEL_OPERATION_SPLIT!r}"
                )
            return self._novel_operation_pool, self._novel_operation_lengths, "novel_operation"
        raise ValueError(f"Unknown split: {split!r}")

    def _resolve_oracle_label(
        self, category: str, oracle_label: OracleLabel | None
    ) -> OracleLabel:
        inferred: dict[str, OracleLabel] = {
            "known": ORACLE_LABEL_KNOWN,
            "novel_composition": ORACLE_LABEL_NOVEL_COMPOSITION,
            "novel_operation": ORACLE_LABEL_NOVEL_OPERATION,
        }
        expected = inferred[category]
        if oracle_label is None:
            return expected
        if oracle_label == expected:
            return oracle_label
        if category == "novel_operation" and oracle_label == ORACLE_LABEL_RECURRENCE:
            return oracle_label
        raise ValueError(
            f"oracle_label {oracle_label!r} is incompatible with category {category!r}"
        )

    def _generate(
        self,
        n: int,
        split: str,
        *,
        rng_label: str,
        oracle_label: OracleLabel | None = None,
    ) -> list[Example]:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        pool, lengths_by_chain, category = self._pool_lengths_and_category(split)
        resolved_label = self._resolve_oracle_label(category, oracle_label)
        rng = random.Random(_derive_seed(self.seed, rng_label))

        # One shared permutation for this whole call (i.e. per batch/episode
        # -- a `generate_online` call is exactly one training step's batch),
        # not one per example. Drawn from its own rng stream, independent of
        # `rng` above, so enabling/disabling permutation never perturbs which
        # canonical operation/length/content gets generated. See
        # docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md section 3
        # ("per-batch or per-episode symbol permutation") and
        # docs/DECISIONS.md ADR-0018: a *fresh* permutation on every single
        # example would destroy any value/order relationship a learner could
        # exploit *across* examples, making arithmetic/order-dependent
        # operations (e.g. NEGATE, COMPARE, ACCUMULATE) unlearnable as a
        # function of presented input -- only position-based ("value-blind")
        # operations commute with an independent per-example relabeling.
        permutation: SymbolPermutation | None = None
        if self.permute_symbols:
            permutation_rng = random.Random(_derive_seed(self.seed, f"{rng_label}:permutation"))
            permutation = sample_permutation(permutation_rng, self.vocab_size)

        examples: list[Example] = []
        for _ in range(n):
            operation_sequence = pool[rng.randrange(len(pool))]
            lengths = lengths_by_chain[operation_sequence]
            length = lengths[rng.randrange(len(lengths))]
            input_tokens = tuple(rng.randrange(self.vocab_size) for _ in range(length))

            current = input_tokens
            steps: list[ProgramStep] = []
            for name in operation_sequence:
                operation = get_operation(name)
                params = operation.sample_params(rng, current, self.vocab_size)
                steps.append(ProgramStep(operation=name, params=params))
                current = operation.apply(current, self.vocab_size, params)

            program = Program(steps=tuple(steps))
            result = run_program(program, input_tokens, self.vocab_size)

            presented_input = input_tokens
            presented_target = result.output_tokens
            if permutation is not None:
                presented_input = permutation.apply(input_tokens)
                presented_target = permutation.apply(result.output_tokens)

            examples.append(
                Example(
                    input_tokens=presented_input,
                    target_tokens=presented_target,
                    program=program,
                    operation_graph=result.graph,
                    category=category,
                    split=split,
                    vocab_size=self.vocab_size,
                    task_spec=TaskSpec.from_program(program),
                    oracle_metadata=OracleMetadata(
                        label=resolved_label,
                        primitive_operations=program.operation_sequence,
                        recurrence_operation=(
                            program.operation_sequence[0]
                            if resolved_label == ORACLE_LABEL_RECURRENCE
                            else None
                        ),
                    ),
                    symbol_permutation=permutation,
                )
            )
        return examples

    def generate(self, n: int, split: str) -> list[Example]:
        """Generate a fixed deterministic collection for Phase A diagnostics."""
        return self._generate(n, split, rng_label=f"generate:{split}")

    def generate_online(
        self,
        n: int,
        *,
        step: int,
        split: str,
        oracle_label: OracleLabel | None = None,
    ) -> list[Example]:
        """Generate one fresh procedural batch for a numbered online step.

        The result is a pure function of the generator configuration and
        ``(seed, step, split)``.  It stores no growing dataset, so callers can
        request arbitrarily many training batches while replaying any exact
        batch later by passing the same arguments.  ``R`` is allowed only for
        a ``novel_operation`` split, where it denotes fresh-content recurrence
        of that operation.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        return self._generate(
            n,
            split,
            rng_label=f"online:{step}:{split}",
            oracle_label=oracle_label,
        )


MIXED_OPERATION_MAX_DEPTH: Final[int] = 1
"""Composition depth pinned by `build_mixed_operation_generator` (Task A1-C002).

Not exposed as a parameter: mixing depth>1 chains into the mixed-operation
stream would confound operation-identifiability (what A1-C002/A1-C004 test)
with composition generalization, which is A1-008's separate concern. Mirrors
`apc.evaluation.stable_core_generalization`'s own "Composition depth is
fixed at 1, not exposed as a config field" choice for the same reason.
"""


def build_mixed_operation_generator(
    seed: int,
    *,
    operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    permute_symbols: bool = False,
) -> TaskGenerator:
    """Build "one identifiable mixed-operation training stream" (Task A1-C002).

    A single `TaskGenerator` whose known pool is exactly one depth-1
    application of each entry in `operation_names` -- every generated
    example applies exactly one known operation to fresh content, and that
    operation's identity plus every argument is fully recoverable from
    `Example.task_spec` (Task A1-C001), so the same content correctly pairs
    with different operations (and different arguments of the same
    operation) without any hidden control variable.

    `operation_names` defaults to all of `KNOWN_OPERATION_NAMES` -- all
    eight known operations, including `SELECT`/`COUNT`/`SHIFT`/`BIND` (the
    four `docs/DECISIONS.md` ADR-0017 operations `apc.evaluation.
    stable_core_generalization.DETERMINISTIC_OPERATION_NAMES` excludes).
    Those four are safe to pool here specifically because `task_spec`
    already carries their previously-hidden parameter; `DETERMINISTIC_OPERATION_NAMES`
    predates `task_spec` and is scoped for a model that never sees it.

    `permute_symbols` defaults to `False`, matching Task A1-C002's primary
    run; see `docs/DECISIONS.md` ADR-0019 before enabling it for anything
    beyond a value-blind operation subset.
    """
    return TaskGenerator(
        seed=seed,
        operation_names=operation_names,
        vocab_size=vocab_size,
        sequence_length_range=sequence_length_range,
        max_depth=MIXED_OPERATION_MAX_DEPTH,
        novel_operation_names=(),
        permute_symbols=permute_symbols,
    )
