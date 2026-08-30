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
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from typing import Any

from apc.environments.interpreter import OperationGraph, run_program
from apc.environments.operations import KNOWN_OPERATION_NAMES, get_operation
from apc.environments.program import Program, ProgramStep
from apc.environments.vocab import DEFAULT_VOCAB_SIZE

KNOWN_SPLITS: tuple[str, ...] = ("train", "val", "test")
NOVEL_COMPOSITION_SPLIT = "novel_composition"
NOVEL_OPERATION_SPLIT = "novel_operation"


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
    """One generated task instance with its ground-truth operation graph."""

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    program: Program
    operation_graph: OperationGraph
    category: str  # "known" | "novel_composition" | "novel_operation"
    split: str
    vocab_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": list(self.input_tokens),
            "target_tokens": list(self.target_tokens),
            "program": self.program.to_dict(),
            "operation_graph": self.operation_graph.to_dict(),
            "category": self.category,
            "split": self.split,
            "vocab_size": self.vocab_size,
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
    or of any other split having been generated first.
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
    ) -> None:
        self.seed = seed
        self.operation_names = operation_names
        self.vocab_size = vocab_size
        self.sequence_length_range = sequence_length_range
        self.novel_operation_names = novel_operation_names
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

    def generate(self, n: int, split: str) -> list[Example]:
        """Generate `n` examples for `split`, deterministic given `seed`."""
        pool, lengths_by_chain, category = self._pool_lengths_and_category(split)
        rng = random.Random(_derive_seed(self.seed, f"generate:{split}"))

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
            examples.append(
                Example(
                    input_tokens=input_tokens,
                    target_tokens=result.output_tokens,
                    program=program,
                    operation_graph=result.graph,
                    category=category,
                    split=split,
                    vocab_size=self.vocab_size,
                )
            )
        return examples
