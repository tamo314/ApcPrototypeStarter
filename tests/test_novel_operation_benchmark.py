"""Novel-operation benchmark tests (Task 009)."""

from __future__ import annotations

import json

import pytest

from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.environments.operations import NOVEL_OPERATION_NAMES
from apc.evaluation.composition_benchmark import LABEL_NOVEL_COMPOSITION, LABEL_NOVEL_OPERATION
from apc.evaluation.novel_operation_benchmark import (
    NovelOperationBenchmarkConfig,
    run_novel_operation_benchmark,
)
from apc.utils.seed import set_seed

LENGTH_RANGE = (4, 6)


def _tiny_model(vocab_size: int) -> DecoderOnlyTransformer:
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=16,
        n_layer=1,
        n_head=2,
        d_ff=32,
        dropout=0.0,
    )
    return DecoderOnlyTransformer(config)


def test_config_rejects_empty_novel_operation_names() -> None:
    with pytest.raises(ValueError):
        NovelOperationBenchmarkConfig(seed=0, novel_operation_names=())


def test_config_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError):
        NovelOperationBenchmarkConfig(
            seed=0, novel_operation_names=NOVEL_OPERATION_NAMES, num_novel_composition=0
        )
    with pytest.raises(ValueError):
        NovelOperationBenchmarkConfig(
            seed=0, novel_operation_names=NOVEL_OPERATION_NAMES, num_novel_operation=0
        )


def test_run_novel_operation_benchmark_reports_structure_and_ranges() -> None:
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    model = _tiny_model(specials.model_vocab_size)
    config = NovelOperationBenchmarkConfig(
        seed=5,
        novel_operation_names=NOVEL_OPERATION_NAMES,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        num_novel_composition=8,
        num_novel_operation=8,
    )

    report = run_novel_operation_benchmark(model, config, specials, device="cpu")

    assert report.novel_composition.label == LABEL_NOVEL_COMPOSITION
    assert report.novel_composition.category == "novel_composition"
    assert report.novel_composition.num_examples == 8
    assert 0.0 <= report.novel_composition.exact_match <= 1.0

    assert report.novel_operation.label == LABEL_NOVEL_OPERATION
    assert report.novel_operation.category == "novel_operation"
    assert report.novel_operation.num_examples == 8
    assert 0.0 <= report.novel_operation.exact_match <= 1.0

    assert report.novelty_gap == pytest.approx(
        report.novel_composition.exact_match - report.novel_operation.exact_match
    )

    # Must be plain, JSON-serializable data (for the run-artifact report).
    payload = json.dumps(report.to_dict())
    assert json.loads(payload)["config"]["seed"] == 5
    assert json.loads(payload)["config"]["novel_operation_names"] == list(NOVEL_OPERATION_NAMES)


def test_run_novel_operation_benchmark_is_deterministic_given_same_seed() -> None:
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    model = _tiny_model(specials.model_vocab_size)
    config = NovelOperationBenchmarkConfig(
        seed=9,
        novel_operation_names=NOVEL_OPERATION_NAMES,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        num_novel_composition=6,
        num_novel_operation=6,
    )

    first = run_novel_operation_benchmark(model, config, specials, device="cpu")
    second = run_novel_operation_benchmark(model, config, specials, device="cpu")

    assert first.to_dict() == second.to_dict()


def test_a_model_never_trained_on_sort_fails_it_outright() -> None:
    """A model that has never seen SORT cannot solve it by chance: exact
    match requires reproducing every one of the target's tokens (including
    length and EOS placement) via greedy decoding, so an untrained/
    randomly-initialized model's success rate on N is realistically zero.
    This is the sharpest form of Task 009's acceptance criterion (a static
    composition baseline fails on N) without requiring a full training run.
    """
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    set_seed(0)
    model = _tiny_model(specials.model_vocab_size)
    config = NovelOperationBenchmarkConfig(
        seed=1,
        novel_operation_names=NOVEL_OPERATION_NAMES,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        num_novel_composition=16,
        num_novel_operation=16,
    )

    report = run_novel_operation_benchmark(model, config, specials, device="cpu")

    assert report.novel_operation.exact_match == 0.0
