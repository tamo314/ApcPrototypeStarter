"""Composition-generalization benchmark (Phase A Milestone A4 / Task 006).

Splits a known-operation task universe into train/val/test pools and a
held-out `novel_composition` pool are already produced by
`apc.environments.generator.CompositionSpace` (Task 002). This module adds
the evaluation-side piece: an oracle labeler that tags generated examples
`K` (known) or `C` (novel composition) for reporting, and a benchmark
runner that evaluates a model on both pools and reports the composition
generalization gap.

The K/C label is read from `Example.category`, which is metadata produced
purely by the environment generator. It is used here only to group
examples for the report; `apc.core.generation.evaluate_exact_match` (via
`apc.core.data.encode_example`) reads only `input_tokens`/`target_tokens`
from each example, so the label is never part of what the model (or, in
later tasks, the meta-controller) sees.

`label_example` also maps the `novel_operation` category (Task 009,
`apc.environments.generator.NOVEL_OPERATION_SPLIT`) to the `N` label from
`docs/EXPERIMENT_PLAN.md`'s K/C/N/R event taxonomy; `evaluate_category` is
exposed (not private) so `apc.evaluation.novel_operation_benchmark` can
reuse the same oracle-label-checked per-group evaluation instead of
duplicating it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from apc.core.generation import evaluate_exact_match
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SpecialTokens
from apc.environments.generator import (
    KNOWN_SPLITS,
    NOVEL_COMPOSITION_SPLIT,
    Example,
    TaskGenerator,
)
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE

LABEL_KNOWN = "K"
LABEL_NOVEL_COMPOSITION = "C"
LABEL_NOVEL_OPERATION = "N"

_CATEGORY_TO_LABEL = {
    "known": LABEL_KNOWN,
    "novel_composition": LABEL_NOVEL_COMPOSITION,
    "novel_operation": LABEL_NOVEL_OPERATION,
}


def label_example(example: Example) -> str:
    """Map an example's oracle `category` to its K/C benchmark label.

    This is the only function in the benchmark that reads the oracle
    label; callers must use it for report grouping only, never as model
    or controller input.
    """
    try:
        return _CATEGORY_TO_LABEL[example.category]
    except KeyError:
        raise ValueError(f"Unlabelable example category: {example.category!r}") from None


@dataclass(frozen=True)
class CompositionBenchmarkConfig:
    """Explicit, serializable config for one composition-benchmark run."""

    seed: int
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    max_depth: int = 2
    novel_composition_fraction: float = 0.3
    operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES
    known_split: str = "test"
    num_known: int = 64
    num_novel: int = 64

    def __post_init__(self) -> None:
        if self.known_split not in KNOWN_SPLITS:
            raise ValueError(
                f"known_split must be one of {KNOWN_SPLITS}, got {self.known_split!r}"
            )
        if self.num_known < 1:
            raise ValueError(f"num_known must be >= 1, got {self.num_known}")
        if self.num_novel < 1:
            raise ValueError(f"num_novel must be >= 1, got {self.num_novel}")

    def to_dict(self) -> dict[str, Any]:
        # Round-trip through JSON to turn tuples into lists, matching
        # `apc.core.train._resolved_config_dict`.
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class CategoryResult:
    """Exact-match performance for one K/C group of examples."""

    label: str
    category: str
    split: str
    num_examples: int
    exact_match: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "split": self.split,
            "num_examples": self.num_examples,
            "exact_match": self.exact_match,
        }


@dataclass(frozen=True)
class CompositionBenchmarkReport:
    """Composition-generalization report: known vs. held-out composition."""

    config: CompositionBenchmarkConfig
    known: CategoryResult
    novel_composition: CategoryResult
    generalization_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "known": self.known.to_dict(),
            "novel_composition": self.novel_composition.to_dict(),
            "generalization_gap": self.generalization_gap,
        }


def evaluate_category(
    model: DecoderOnlyTransformer,
    examples: list[Example],
    expected_label: str,
    specials: SpecialTokens,
    device: torch.device | str,
) -> CategoryResult:
    """Evaluate one oracle-labeled group of examples (shared by every
    Phase A benchmark that reports per-K/C/N/R-label exact match)."""
    if not examples:
        raise ValueError("examples must be non-empty")
    labels = {label_example(example) for example in examples}
    if labels != {expected_label}:
        raise AssertionError(
            f"Expected every example to carry oracle label {expected_label!r}, "
            f"got {sorted(labels)!r}"
        )
    exact_match, _ = evaluate_exact_match(model, examples, specials, device)
    return CategoryResult(
        label=expected_label,
        category=examples[0].category,
        split=examples[0].split,
        num_examples=len(examples),
        exact_match=exact_match,
    )


def run_composition_benchmark(
    model: DecoderOnlyTransformer,
    config: CompositionBenchmarkConfig,
    specials: SpecialTokens,
    device: torch.device | str = "cpu",
) -> CompositionBenchmarkReport:
    """Evaluate `model` on known (K) and held-out novel-composition (C)
    examples and report the composition-generalization gap.

    `config.known_split` selects which known pool (`train`/`val`/`test`)
    to draw K examples from; the C pool is always `NOVEL_COMPOSITION_SPLIT`
    — operation chains held out of the known pool entirely (see
    `apc.environments.generator.CompositionSpace`).
    """
    generator = TaskGenerator(
        seed=config.seed,
        operation_names=config.operation_names,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        max_depth=config.max_depth,
        novel_composition_fraction=config.novel_composition_fraction,
    )
    known_examples = generator.generate(config.num_known, config.known_split)
    novel_examples = generator.generate(config.num_novel, NOVEL_COMPOSITION_SPLIT)

    known_result = evaluate_category(model, known_examples, LABEL_KNOWN, specials, device)
    novel_result = evaluate_category(
        model, novel_examples, LABEL_NOVEL_COMPOSITION, specials, device
    )
    return CompositionBenchmarkReport(
        config=config,
        known=known_result,
        novel_composition=novel_result,
        generalization_gap=known_result.exact_match - novel_result.exact_match,
    )
