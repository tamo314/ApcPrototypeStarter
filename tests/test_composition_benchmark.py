"""Composition-generalization benchmark tests (Milestone A4 / Task 006)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from apc.core.data import encode_example
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.environments.generator import (
    NOVEL_COMPOSITION_SPLIT,
    NOVEL_OPERATION_SPLIT,
    TaskGenerator,
)
from apc.environments.operations import NOVEL_OPERATION_NAMES
from apc.evaluation.composition_benchmark import (
    LABEL_KNOWN,
    LABEL_NOVEL_COMPOSITION,
    LABEL_NOVEL_OPERATION,
    CompositionBenchmarkConfig,
    label_example,
    run_composition_benchmark,
)

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


def test_label_example_maps_known_and_novel_composition() -> None:
    generator = TaskGenerator(seed=3, sequence_length_range=LENGTH_RANGE, max_depth=2)
    known = generator.generate(5, "test")
    novel = generator.generate(5, NOVEL_COMPOSITION_SPLIT)

    assert all(label_example(e) == LABEL_KNOWN for e in known)
    assert all(label_example(e) == LABEL_NOVEL_COMPOSITION for e in novel)


def test_label_example_maps_novel_operation_to_n() -> None:
    generator = TaskGenerator(
        seed=3,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    novel_operation = generator.generate(5, NOVEL_OPERATION_SPLIT)
    assert all(label_example(e) == LABEL_NOVEL_OPERATION for e in novel_operation)


def test_label_example_rejects_unknown_category() -> None:
    [example] = TaskGenerator(seed=0, sequence_length_range=LENGTH_RANGE).generate(1, "train")
    bogus = dataclasses.replace(example, category="not_a_real_category")
    with pytest.raises(ValueError):
        label_example(bogus)


def test_composition_benchmark_config_rejects_invalid_known_split() -> None:
    with pytest.raises(ValueError):
        CompositionBenchmarkConfig(seed=0, known_split="not_a_split")


def test_composition_benchmark_config_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError):
        CompositionBenchmarkConfig(seed=0, num_known=0)
    with pytest.raises(ValueError):
        CompositionBenchmarkConfig(seed=0, num_novel=0)


def test_oracle_label_is_not_encoded_for_the_model() -> None:
    """Task 006 acceptance: the model/controller must not see the K/C label.

    `encode_example` is the only place `Example`s become model input
    (`apc.core.data`); it must depend only on `input_tokens`/`target_tokens`,
    never on `category`/`split`.
    """
    [example] = TaskGenerator(seed=1, sequence_length_range=LENGTH_RANGE).generate(1, "train")
    specials = build_special_tokens(example.vocab_size)

    relabeled = dataclasses.replace(
        example, category="novel_composition", split=NOVEL_COMPOSITION_SPLIT
    )

    assert label_example(example) != label_example(relabeled)
    assert encode_example(example, specials) == encode_example(relabeled, specials)


def test_oracle_n_label_is_not_encoded_for_the_model() -> None:
    """Task 009 acceptance: oracle metadata marks N but stays hidden from
    the model/controller, exactly like the K/C labels above."""
    [example] = TaskGenerator(seed=1, sequence_length_range=LENGTH_RANGE).generate(1, "train")
    specials = build_special_tokens(example.vocab_size)

    relabeled = dataclasses.replace(
        example, category="novel_operation", split=NOVEL_OPERATION_SPLIT
    )

    assert label_example(relabeled) == LABEL_NOVEL_OPERATION
    assert label_example(example) != label_example(relabeled)
    assert encode_example(example, specials) == encode_example(relabeled, specials)


def test_run_composition_benchmark_reports_structure_and_ranges() -> None:
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    model = _tiny_model(specials.model_vocab_size)
    config = CompositionBenchmarkConfig(
        seed=5,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        known_split="test",
        num_known=8,
        num_novel=8,
    )

    report = run_composition_benchmark(model, config, specials, device="cpu")

    assert report.known.label == LABEL_KNOWN
    assert report.known.category == "known"
    assert report.known.split == "test"
    assert report.known.num_examples == 8
    assert 0.0 <= report.known.exact_match <= 1.0

    assert report.novel_composition.label == LABEL_NOVEL_COMPOSITION
    assert report.novel_composition.category == "novel_composition"
    assert report.novel_composition.num_examples == 8
    assert 0.0 <= report.novel_composition.exact_match <= 1.0

    assert report.generalization_gap == pytest.approx(
        report.known.exact_match - report.novel_composition.exact_match
    )

    # Must be plain, JSON-serializable data (for the run-artifact report).
    payload = json.dumps(report.to_dict())
    assert json.loads(payload)["config"]["seed"] == 5


def test_run_composition_benchmark_is_deterministic_given_same_seed() -> None:
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    model = _tiny_model(specials.model_vocab_size)
    config = CompositionBenchmarkConfig(
        seed=9,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        num_known=6,
        num_novel=6,
    )

    first = run_composition_benchmark(model, config, specials, device="cpu")
    second = run_composition_benchmark(model, config, specials, device="cpu")

    assert first.to_dict() == second.to_dict()


def test_run_composition_benchmark_known_split_selects_pool() -> None:
    vocab_size = 10
    specials = build_special_tokens(vocab_size)
    model = _tiny_model(specials.model_vocab_size)
    base_kwargs = dict(
        seed=2,
        vocab_size=vocab_size,
        sequence_length_range=LENGTH_RANGE,
        max_depth=2,
        num_known=6,
        num_novel=6,
    )

    val_report = run_composition_benchmark(
        model, CompositionBenchmarkConfig(known_split="val", **base_kwargs), specials, device="cpu"
    )
    test_report = run_composition_benchmark(
        model, CompositionBenchmarkConfig(known_split="test", **base_kwargs), specials, device="cpu"
    )

    assert val_report.known.split == "val"
    assert test_report.known.split == "test"
