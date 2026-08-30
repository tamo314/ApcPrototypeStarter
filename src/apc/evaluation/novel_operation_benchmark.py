"""Novel-operation benchmark (Phase A Task 009).

Task 006's composition benchmark (`apc.evaluation.composition_benchmark`)
compares a model on known-operation tasks (`K`) against held-out
*compositions* of known operations (`C`). This module adds the third leg
required to test H1 in `docs/EXPERIMENT_PLAN.md` ("the system allocates new
trainable capacity substantially more often for genuinely novel operations
than for held-out compositions of known operations"): it compares `C`
against genuinely novel *operations* (`N`, `apc.environments.generator.
NOVEL_OPERATION_SPLIT`) -- operations outside `operation_names` entirely
(e.g. `SORT`, see `apc.environments.operations.NOVEL_OPERATION_NAMES`), not
just an unseen chain of familiar ones.

Task 009's acceptance criterion is that a static composition baseline (a
model/controller limited to composing known operations) fails on `N`
materially more often than on `C`. `novelty_gap` below is exactly that
comparison (`C` exact match minus `N` exact match); it is reported, not
asserted, because whether it is "material" is an experimental question
(see `docs/EXPERIMENT_PLAN.md` sections 4/10), not a code invariant.

As with Task 006, the oracle `N` label is for this report only.
`evaluate_category` (imported from `composition_benchmark`) evaluates
examples through the exact same `encode_example`-based path used for
K/C, so the label never reaches the model, and nothing here is wired into
`apc.meta.controller.Controller` or `apc.meta.novelty.NoveltyEstimator`,
which only ever see error/entropy scalars.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SpecialTokens
from apc.environments.generator import (
    NOVEL_COMPOSITION_SPLIT,
    NOVEL_OPERATION_SPLIT,
    TaskGenerator,
)
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.composition_benchmark import (
    LABEL_NOVEL_COMPOSITION,
    LABEL_NOVEL_OPERATION,
    CategoryResult,
    evaluate_category,
)


@dataclass(frozen=True)
class NovelOperationBenchmarkConfig:
    """Explicit, serializable config for one novel-operation-benchmark run."""

    seed: int
    novel_operation_names: tuple[str, ...]
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    max_depth: int = 2
    novel_composition_fraction: float = 0.3
    operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES
    num_novel_composition: int = 64
    num_novel_operation: int = 64

    def __post_init__(self) -> None:
        if not self.novel_operation_names:
            raise ValueError("novel_operation_names must be non-empty")
        if self.num_novel_composition < 1:
            raise ValueError(
                f"num_novel_composition must be >= 1, got {self.num_novel_composition}"
            )
        if self.num_novel_operation < 1:
            raise ValueError(f"num_novel_operation must be >= 1, got {self.num_novel_operation}")

    def to_dict(self) -> dict[str, Any]:
        # Round-trip through JSON to turn tuples into lists, matching
        # `CompositionBenchmarkConfig.to_dict`.
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class NovelOperationBenchmarkReport:
    """Held-out composition (C) vs. genuinely novel operation (N) report."""

    config: NovelOperationBenchmarkConfig
    novel_composition: CategoryResult
    novel_operation: CategoryResult
    novelty_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "novel_composition": self.novel_composition.to_dict(),
            "novel_operation": self.novel_operation.to_dict(),
            "novelty_gap": self.novelty_gap,
        }


def run_novel_operation_benchmark(
    model: DecoderOnlyTransformer,
    config: NovelOperationBenchmarkConfig,
    specials: SpecialTokens,
    device: torch.device | str = "cpu",
) -> NovelOperationBenchmarkReport:
    """Evaluate `model` on held-out novel-composition (C) and genuinely
    novel-operation (N) examples and report the `C - N` exact-match gap.

    `config.novel_operation_names` must name operations registered via
    `apc.environments.operations.register_operation` but absent from
    `config.operation_names` (see `NOVEL_OPERATION_NAMES`); otherwise
    `TaskGenerator` raises when asked to generate `NOVEL_OPERATION_SPLIT`.
    """
    generator = TaskGenerator(
        seed=config.seed,
        operation_names=config.operation_names,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        max_depth=config.max_depth,
        novel_composition_fraction=config.novel_composition_fraction,
        novel_operation_names=config.novel_operation_names,
    )
    novel_composition_examples = generator.generate(
        config.num_novel_composition, NOVEL_COMPOSITION_SPLIT
    )
    novel_operation_examples = generator.generate(config.num_novel_operation, NOVEL_OPERATION_SPLIT)

    novel_composition_result = evaluate_category(
        model, novel_composition_examples, LABEL_NOVEL_COMPOSITION, specials, device
    )
    novel_operation_result = evaluate_category(
        model, novel_operation_examples, LABEL_NOVEL_OPERATION, specials, device
    )
    return NovelOperationBenchmarkReport(
        config=config,
        novel_composition=novel_composition_result,
        novel_operation=novel_operation_result,
        novelty_gap=novel_composition_result.exact_match - novel_operation_result.exact_match,
    )
