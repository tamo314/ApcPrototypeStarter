"""Deterministic generator and composition-split tests (Milestone A1)."""

from __future__ import annotations

import random

import pytest

from apc.environments.generator import (
    KNOWN_SPLITS,
    MIXED_OPERATION_MAX_DEPTH,
    NOVEL_COMPOSITION_SPLIT,
    NOVEL_OPERATION_SPLIT,
    ORACLE_LABEL_KNOWN,
    ORACLE_LABEL_NOVEL_COMPOSITION,
    ORACLE_LABEL_NOVEL_OPERATION,
    ORACLE_LABEL_RECURRENCE,
    CompositionSpace,
    TaskGenerator,
    build_mixed_operation_generator,
    enumerate_compositions,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    KNOWN_OPERATION_NAMES,
    NOVEL_OPERATION_NAMES,
    get_operation,
)
from apc.environments.permutation import SymbolPermutation
from apc.environments.task_spec import TaskSpec, TaskStepSpec

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


# --- explicit task specification (Phase A.1 Correction Task A1-C001) -------


@pytest.mark.parametrize("split", [*KNOWN_SPLITS, NOVEL_COMPOSITION_SPLIT])
def test_examples_carry_task_spec_matching_program(split: str) -> None:
    examples = TaskGenerator(seed=3, sequence_length_range=LENGTH_RANGE).generate(5, split)
    for example in examples:
        assert isinstance(example.task_spec, TaskSpec)
        assert example.task_spec.operation_sequence == example.program.operation_sequence
        assert example.task_spec.to_program() == example.program


def test_task_spec_reconstructed_program_reproduces_target_for_generated_examples() -> None:
    """Acceptance: task specification fully determines all previously hidden
    operation parameters -- replaying only the `TaskSpec`-reconstructed
    program (not the original `Program` object) on the presented input must
    reproduce the presented target exactly, for every known operation
    including the four ADR-0017 previously-hidden-parameter ones."""
    generator = TaskGenerator(seed=13, sequence_length_range=LENGTH_RANGE, max_depth=1)
    for example in generator.generate(60, "train"):
        assert example.task_spec is not None
        program = example.task_spec.to_program()
        replay = run_program(program, example.input_tokens, example.vocab_size)
        assert replay.output_tokens == example.target_tokens


def test_online_examples_carry_task_spec_matching_program() -> None:
    generator = TaskGenerator(
        seed=29,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
    )
    for split in ("train", NOVEL_COMPOSITION_SPLIT, NOVEL_OPERATION_SPLIT):
        for example in generator.generate_online(8, step=7, split=split):
            assert isinstance(example.task_spec, TaskSpec)
            assert example.task_spec.to_program() == example.program


def test_task_spec_survives_symbol_permutation_and_still_determines_canonical_target() -> None:
    """`task_spec` always describes the canonical `Program`, unaffected by
    `permute_symbols`; decoding the presented tokens back to canonical ids
    (as evaluation code must) and replaying the `task_spec`-reconstructed
    program must still reproduce the canonical target."""
    generator = TaskGenerator(
        seed=41, sequence_length_range=LENGTH_RANGE, permute_symbols=True
    )
    for example in generator.generate_online(15, step=2, split="train"):
        permutation = example.symbol_permutation
        assert permutation is not None
        assert example.task_spec is not None
        canonical_input = permutation.invert(example.input_tokens)
        canonical_target = permutation.invert(example.target_tokens)
        replay = run_program(example.task_spec.to_program(), canonical_input, example.vocab_size)
        assert replay.output_tokens == canonical_target


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


# --- symbol permutation anti-shortcut control (Phase A.1 Task A1-004) ------


def test_permute_symbols_off_by_default() -> None:
    generator = TaskGenerator(seed=6, sequence_length_range=LENGTH_RANGE)
    for example in generator.generate(10, "train"):
        assert example.symbol_permutation is None
    for example in generator.generate_online(10, step=0, split="train"):
        assert example.symbol_permutation is None


def test_permute_symbols_attaches_a_bijective_mapping_per_example() -> None:
    generator = TaskGenerator(
        seed=6, sequence_length_range=LENGTH_RANGE, permute_symbols=True
    )
    for example in generator.generate_online(15, step=0, split="train"):
        assert isinstance(example.symbol_permutation, SymbolPermutation)
        assert example.symbol_permutation.vocab_size == example.vocab_size
        assert sorted(example.symbol_permutation.forward) == list(range(example.vocab_size))


def test_permute_symbols_changes_at_least_some_presented_token_mappings() -> None:
    """A fixed seed should not, by construction, always land on the identity
    mapping -- otherwise permutation would be a no-op in practice."""
    generator = TaskGenerator(
        seed=6, sequence_length_range=LENGTH_RANGE, permute_symbols=True
    )
    examples = generator.generate_online(20, step=0, split="train")
    identity = tuple(range(generator.vocab_size))
    assert any(example.symbol_permutation.forward != identity for example in examples)


def test_permute_symbols_is_reproducible_by_seed_step_and_split() -> None:
    kwargs = dict(seed=17, sequence_length_range=LENGTH_RANGE, permute_symbols=True)
    first = TaskGenerator(**kwargs).generate_online(12, step=3, split="train")
    second = TaskGenerator(**kwargs).generate_online(12, step=3, split="train")
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]


def test_permute_symbols_to_dict_logs_permutation_identity() -> None:
    generator = TaskGenerator(
        seed=6, sequence_length_range=LENGTH_RANGE, permute_symbols=True
    )
    for example in generator.generate_online(5, step=0, split="train"):
        payload = example.to_dict()["symbol_permutation"]
        assert payload == {"forward": list(example.symbol_permutation.forward)}


def test_permute_symbols_preserves_interpreter_truth_after_decoding() -> None:
    """Decoding the presented (permuted) tokens back through the attached
    mapping must reproduce exactly what the unmodified interpreter computes
    on the canonical input -- operation semantics are untouched by
    permutation, only the presented token identities change."""
    generator = TaskGenerator(
        seed=41,
        sequence_length_range=LENGTH_RANGE,
        novel_operation_names=NOVEL_OPERATION_NAMES,
        permute_symbols=True,
    )
    for split in ("train", NOVEL_COMPOSITION_SPLIT, NOVEL_OPERATION_SPLIT):
        for example in generator.generate_online(8, step=2, split=split):
            permutation = example.symbol_permutation
            assert permutation is not None
            canonical_input = permutation.invert(example.input_tokens)
            canonical_target = permutation.invert(example.target_tokens)
            replay = run_program(example.program, canonical_input, example.vocab_size)
            assert replay.output_tokens == canonical_target
            assert replay.graph == example.operation_graph


def test_permute_symbols_is_shared_across_every_example_in_one_call() -> None:
    """Task A1-006 finding (docs/DECISIONS.md ADR-0018): permutation must be
    shared per batch/episode, not drawn fresh per example -- an independent
    per-example relabeling leaves no value/order relationship a learner
    could exploit across examples, making arithmetic/order-dependent
    operations unlearnable regardless of training."""
    generator = TaskGenerator(seed=6, sequence_length_range=LENGTH_RANGE, permute_symbols=True)
    examples = generator.generate_online(25, step=0, split="train")
    first = examples[0].symbol_permutation
    assert first is not None
    assert all(example.symbol_permutation == first for example in examples)


def test_permute_symbols_changes_across_different_generate_calls() -> None:
    """A shared-per-call permutation must still change across calls (steps),
    otherwise the anti-shortcut property A1-004 exists for would be lost."""
    generator = TaskGenerator(seed=6, sequence_length_range=LENGTH_RANGE, permute_symbols=True)
    step_0 = generator.generate_online(5, step=0, split="train")[0].symbol_permutation
    step_1 = generator.generate_online(5, step=1, split="train")[0].symbol_permutation
    assert step_0 != step_1


def test_permute_symbols_does_not_perturb_unpermuted_generation() -> None:
    """Enabling the flag must not change any non-permutation-related draw
    (operation choice, length, params): with permutation inverted back out,
    output must match the unpermuted generator byte for byte. Together with
    `test_permute_symbols_preserves_interpreter_truth_after_decoding`, this
    shows the same abstract task (same program, same canonical content)
    reaches the model under a different token mapping without its
    interpreter-truth semantics changing."""
    plain = TaskGenerator(seed=9, sequence_length_range=LENGTH_RANGE).generate_online(
        10, step=1, split="train"
    )
    permuted = TaskGenerator(
        seed=9, sequence_length_range=LENGTH_RANGE, permute_symbols=True
    ).generate_online(10, step=1, split="train")

    for plain_example, permuted_example in zip(plain, permuted, strict=True):
        permutation = permuted_example.symbol_permutation
        assert permutation is not None
        assert permutation.invert(permuted_example.input_tokens) == plain_example.input_tokens
        assert permutation.invert(permuted_example.target_tokens) == plain_example.target_tokens
        assert permuted_example.program == plain_example.program


# --- mixed-operation online generator (Phase A.1 Correction Task A1-C002) --


def test_build_mixed_operation_generator_defaults_to_all_known_operations_at_depth_one() -> None:
    generator = build_mixed_operation_generator(seed=1, sequence_length_range=LENGTH_RANGE)
    assert generator.operation_names == KNOWN_OPERATION_NAMES
    assert MIXED_OPERATION_MAX_DEPTH == 1
    known = generator.composition_space.known_compositions
    assert set(known) == {(name,) for name in KNOWN_OPERATION_NAMES}
    assert generator.composition_space.novel_compositions == ()


def test_build_mixed_operation_generator_respects_a_custom_operation_subset() -> None:
    subset = ("SHIFT", "BIND", "COUNT")
    generator = build_mixed_operation_generator(
        seed=1, operation_names=subset, sequence_length_range=LENGTH_RANGE
    )
    assert set(generator.composition_space.known_compositions) == {(name,) for name in subset}


def test_build_mixed_operation_generator_defaults_permute_symbols_to_false() -> None:
    generator = build_mixed_operation_generator(seed=1, sequence_length_range=LENGTH_RANGE)
    for example in generator.generate_online(10, step=0, split="train"):
        assert example.symbol_permutation is None


def test_mixed_operation_stream_is_deterministic_by_seed_step_and_split() -> None:
    kwargs = dict(seed=17, sequence_length_range=LENGTH_RANGE)
    first = build_mixed_operation_generator(**kwargs).generate_online(32, step=3, split="train")
    second = build_mixed_operation_generator(**kwargs).generate_online(32, step=3, split="train")
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]


def test_mixed_operation_stream_produces_fresh_content_at_different_steps() -> None:
    generator = build_mixed_operation_generator(seed=17, sequence_length_range=LENGTH_RANGE)
    first = generator.generate_online(32, step=3, split="train")
    second = generator.generate_online(32, step=4, split="train")
    assert [example.to_dict() for example in first] != [example.to_dict() for example in second]


def test_mixed_operation_stream_surfaces_every_known_operation() -> None:
    """Acceptance: all included operations appear in one stream."""
    generator = build_mixed_operation_generator(seed=2, sequence_length_range=LENGTH_RANGE)
    seen_operations: set[str] = set()
    for step in range(40):
        for example in generator.generate_online(64, step=step, split="train"):
            assert len(example.task_spec.steps) == 1  # depth pinned to 1
            seen_operations.add(example.task_spec.operation_sequence[0])
        if seen_operations == set(KNOWN_OPERATION_NAMES):
            break
    assert seen_operations == set(KNOWN_OPERATION_NAMES)


def test_mixed_operation_stream_examples_carry_task_spec_and_replay_to_presented_target() -> None:
    """Acceptance: outputs remain deterministic from visible task spec + content."""
    generator = build_mixed_operation_generator(seed=3, sequence_length_range=LENGTH_RANGE)
    for split in ("train", "val", "test"):
        for example in generator.generate_online(48, step=5, split=split):
            assert isinstance(example.task_spec, TaskSpec)
            replay = run_program(
                example.task_spec.to_program(), example.input_tokens, example.vocab_size
            )
            assert replay.output_tokens == example.target_tokens


def test_same_content_pairs_with_every_known_operation_via_mixed_stream_task_spec() -> None:
    """Acceptance: same content can be paired with multiple operations.

    Directly demonstrates the property `build_mixed_operation_generator`
    relies on: the generator's content sampling is independent of which
    operation is chosen, so any one content sequence can correctly be
    routed through any operation in the mixed pool -- with the operation
    identity (not the content) determining the output.
    """
    generator = build_mixed_operation_generator(seed=4, sequence_length_range=LENGTH_RANGE)
    content = generator.generate_online(1, step=0, split="train")[0].input_tokens
    vocab_size = generator.vocab_size

    outputs = {}
    for name in KNOWN_OPERATION_NAMES:
        operation = get_operation(name)
        if not operation.is_valid_for_length(len(content)):
            continue
        rng = random.Random(0)
        params = operation.sample_params(rng, content, vocab_size)
        task_spec = TaskSpec(steps=(TaskStepSpec(operation=name, arguments=params),))
        replay = run_program(task_spec.to_program(), content, vocab_size)
        outputs[name] = replay.output_tokens

    assert len(outputs) >= 6  # most of the 8 known operations accept this content's length
    assert len(set(outputs.values())) == len(outputs)  # every operation yields a distinct output
