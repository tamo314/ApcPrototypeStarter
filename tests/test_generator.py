"""Deterministic generator and composition-split tests (Milestone A1)."""

from __future__ import annotations

import pytest

from apc.environments.generator import (
    KNOWN_SPLITS,
    NOVEL_COMPOSITION_SPLIT,
    NOVEL_OPERATION_SPLIT,
    ORACLE_LABEL_KNOWN,
    ORACLE_LABEL_NOVEL_COMPOSITION,
    ORACLE_LABEL_NOVEL_OPERATION,
    ORACLE_LABEL_RECURRENCE,
    CompositionSpace,
    TaskGenerator,
    enumerate_compositions,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import KNOWN_OPERATION_NAMES, NOVEL_OPERATION_NAMES

LENGTH_RANGE = (6, 10)


def test_generation_is_deterministic_given_same_seed() -> None:
    first = TaskGenerator(seed=42).generate(25, "train")
    second = TaskGenerator(seed=42).generate(25, "train")
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


def test_different_seeds_diverge() -> None:
    a = TaskGenerator(seed=1).generate(25, "train")
    b = TaskGenerator(seed=2).generate(25, "train")
    assert [e.to_dict() for e in a] != [e.to_dict() for e in b]


def test_splits_are_independently_reproducible_regardless_of_call_order() -> None:
    gen_a = TaskGenerator(seed=7)
    train_then_val = gen_a.generate(10, "train")
    val_1 = gen_a.generate(10, "val")

    gen_b = TaskGenerator(seed=7)
    val_2 = gen_b.generate(10, "val")  # generated first this time, no "train" call before it

    assert [e.to_dict() for e in val_1] == [e.to_dict() for e in val_2]
    assert train_then_val  # sanity: train split isn't empty


@pytest.mark.parametrize("split", [*KNOWN_SPLITS, NOVEL_COMPOSITION_SPLIT])
def test_examples_include_latent_operation_graph_metadata(split: str) -> None:
    examples = TaskGenerator(seed=3).generate(5, split)
    assert len(examples) == 5
    for example in examples:
        assert len(example.operation_graph.nodes) == len(example.program.steps)
        assert example.operation_graph.operation_sequence == example.program.operation_sequence
        # metadata must be plain, JSON-serializable data
        payload = example.to_dict()
        assert isinstance(payload["operation_graph"]["nodes"], list)
        assert isinstance(payload["program"]["steps"], list)


def test_known_splits_are_labeled_known_category() -> None:
    for split in KNOWN_SPLITS:
        examples = TaskGenerator(seed=5).generate(5, split)
        assert all(e.category == "known" for e in examples)
        assert all(e.split == split for e in examples)


def test_novel_composition_split_is_labeled_and_multi_step() -> None:
    examples = TaskGenerator(seed=5).generate(10, NOVEL_COMPOSITION_SPLIT)
    assert all(e.category == "novel_composition" for e in examples)
    assert all(len(e.program.steps) > 1 for e in examples)


def test_unknown_split_raises() -> None:
    with pytest.raises(ValueError):
        TaskGenerator(seed=0).generate(1, "not_a_real_split")


# --- novel_operation split (Task 009) ---------------------------------------


def test_novel_operation_split_without_configured_names_raises() -> None:
    with pytest.raises(ValueError):
        TaskGenerator(seed=0, sequence_length_range=LENGTH_RANGE).generate(1, NOVEL_OPERATION_SPLIT)


def test_novel_operation_split_is_labeled_and_single_step() -> None:
    generator = TaskGenerator(
        seed=5,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    examples = generator.generate(10, NOVEL_OPERATION_SPLIT)
    assert all(e.category == "novel_operation" for e in examples)
    assert all(e.split == NOVEL_OPERATION_SPLIT for e in examples)
    assert all(len(e.program.steps) == 1 for e in examples)
    assert all(e.program.operation_sequence[0] in NOVEL_OPERATION_NAMES for e in examples)


def test_novel_operation_split_is_deterministic_given_same_seed() -> None:
    kwargs = dict(
        seed=8, sequence_length_range=LENGTH_RANGE, novel_operation_names=NOVEL_OPERATION_NAMES
    )
    first = TaskGenerator(**kwargs).generate(10, NOVEL_OPERATION_SPLIT)
    second = TaskGenerator(**kwargs).generate(10, NOVEL_OPERATION_SPLIT)
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


def test_novel_operation_examples_match_independent_interpreter_replay() -> None:
    generator = TaskGenerator(
        seed=12,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    for example in generator.generate(10, NOVEL_OPERATION_SPLIT):
        replay = run_program(example.program, example.input_tokens, example.vocab_size)
        assert replay.output_tokens == example.target_tokens


def test_known_and_novel_composition_splits_unaffected_by_novel_operation_names() -> None:
    """Configuring `novel_operation_names` must not perturb the existing
    `known`/`novel_composition` pools (Task 009 must not silently widen the
    known-composition budget)."""
    plain = TaskGenerator(seed=4, sequence_length_range=LENGTH_RANGE)
    with_novel_op = TaskGenerator(
        seed=4,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    for split in (*KNOWN_SPLITS, NOVEL_COMPOSITION_SPLIT):
        plain_examples = [e.to_dict() for e in plain.generate(10, split)]
        with_novel_op_examples = [e.to_dict() for e in with_novel_op.generate(10, split)]
        assert plain_examples == with_novel_op_examples


def test_generated_examples_match_independent_interpreter_replay() -> None:
    examples = TaskGenerator(seed=11).generate(15, "train")
    for example in examples:
        replay = run_program(example.program, example.input_tokens, example.vocab_size)
        assert replay.output_tokens == example.target_tokens
        assert replay.graph.to_dict() == example.operation_graph.to_dict()


# --- online procedural generation (Phase A.1 Task A1-003) -----------------


def test_online_generation_is_reproducible_by_seed_step_and_split() -> None:
    generator = TaskGenerator(seed=23, sequence_length_range=LENGTH_RANGE)
    first = generator.generate_online(12, step=4, split="train")
    second = generator.generate_online(12, step=4, split="train")
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]


def test_online_generation_produces_fresh_content_at_different_steps() -> None:
    generator = TaskGenerator(seed=23, sequence_length_range=LENGTH_RANGE)
    first = generator.generate_online(12, step=4, split="train")
    second = generator.generate_online(12, step=5, split="train")
    assert [example.to_dict() for example in first] != [example.to_dict() for example in second]


@pytest.mark.parametrize(
    ("split", "expected_label"),
    [
        ("train", ORACLE_LABEL_KNOWN),
        (NOVEL_COMPOSITION_SPLIT, ORACLE_LABEL_NOVEL_COMPOSITION),
        (NOVEL_OPERATION_SPLIT, ORACLE_LABEL_NOVEL_OPERATION),
    ],
)
def test_online_generation_attaches_oracle_labels_and_decomposition(
    split: str, expected_label: str
) -> None:
    generator = TaskGenerator(
        seed=5,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    for example in generator.generate_online(5, step=1, split=split):
        assert example.oracle_metadata is not None
        assert example.oracle_metadata.label == expected_label
        assert example.oracle_metadata.primitive_operations == example.program.operation_sequence


def test_online_recurrence_is_fresh_novel_operation_content_with_r_label() -> None:
    generator = TaskGenerator(
        seed=5,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=("SORT",),
    )
    examples = generator.generate_online(
        5,
        step=2,
        split=NOVEL_OPERATION_SPLIT,
        oracle_label=ORACLE_LABEL_RECURRENCE,
    )
    assert all(example.oracle_metadata is not None for example in examples)
    assert all(example.oracle_metadata.label == ORACLE_LABEL_RECURRENCE for example in examples)
    assert all(example.oracle_metadata.recurrence_operation == "SORT" for example in examples)


def test_online_examples_preserve_interpreter_truth() -> None:
    generator = TaskGenerator(
        seed=29,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    for split in ("train", NOVEL_COMPOSITION_SPLIT, NOVEL_OPERATION_SPLIT):
        for example in generator.generate_online(8, step=7, split=split):
            replay = run_program(example.program, example.input_tokens, example.vocab_size)
            assert replay.output_tokens == example.target_tokens
            assert replay.graph == example.operation_graph


def test_online_generation_rejects_invalid_step_and_label_category_pair() -> None:
    generator = TaskGenerator(seed=0, sequence_length_range=LENGTH_RANGE)
    with pytest.raises(ValueError, match="step"):
        generator.generate_online(1, step=-1, split="train")
    with pytest.raises(ValueError, match="incompatible"):
        generator.generate_online(
            1,
            step=0,
            split="train",
            oracle_label=ORACLE_LABEL_RECURRENCE,
        )


def test_composition_space_known_and_novel_are_disjoint() -> None:
    space = CompositionSpace(
        operation_names=KNOWN_OPERATION_NAMES,
        max_depth=2,
        length_range=LENGTH_RANGE,
        seed=0,
    )
    known = set(space.known_compositions)
    novel = set(space.novel_compositions)
    assert known.isdisjoint(novel)
    assert novel  # held-out pool must be nonempty for the split to be meaningful


def test_composition_space_single_ops_are_always_known() -> None:
    space = CompositionSpace(
        operation_names=KNOWN_OPERATION_NAMES,
        max_depth=2,
        length_range=LENGTH_RANGE,
        seed=0,
    )
    single_op_known = {c for c in space.known_compositions if len(c) == 1}
    assert single_op_known == {(name,) for name in KNOWN_OPERATION_NAMES}
    assert all(len(c) > 1 for c in space.novel_compositions)


def test_composition_space_split_is_deterministic_by_seed() -> None:
    a = CompositionSpace(KNOWN_OPERATION_NAMES, 2, LENGTH_RANGE, seed=99)
    b = CompositionSpace(KNOWN_OPERATION_NAMES, 2, LENGTH_RANGE, seed=99)
    assert a.known_compositions == b.known_compositions
    assert a.novel_compositions == b.novel_compositions


def test_composition_space_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        CompositionSpace(
            KNOWN_OPERATION_NAMES, 2, LENGTH_RANGE, seed=0, novel_composition_fraction=0
        )
    with pytest.raises(ValueError):
        CompositionSpace(
            KNOWN_OPERATION_NAMES, 2, LENGTH_RANGE, seed=0, novel_composition_fraction=1
        )


def test_enumerate_compositions_rejects_non_positive_depth() -> None:
    with pytest.raises(ValueError):
        enumerate_compositions(KNOWN_OPERATION_NAMES, max_depth=0, length_range=LENGTH_RANGE)


def test_enumerate_compositions_includes_every_known_operation_alone() -> None:
    entries = enumerate_compositions(KNOWN_OPERATION_NAMES, max_depth=1, length_range=LENGTH_RANGE)
    assert set(entries.keys()) == {(name,) for name in KNOWN_OPERATION_NAMES}
