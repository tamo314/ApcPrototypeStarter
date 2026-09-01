from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from apc.core.data import (
    IGNORE_INDEX,
    build_prompt_tokens,
    collate_batch,
    encode_example,
    encode_task_spec,
)
from apc.core.tokens import build_shared_core_tokens, build_special_tokens
from apc.environments.generator import ORACLE_LABEL_KNOWN, Example, OracleMetadata
from apc.environments.interpreter import OperationGraph
from apc.environments.permutation import SymbolPermutation
from apc.environments.program import Program
from apc.environments.task_spec import (
    TaskSpec,
    TaskStepSpec,
    num_registered_operations,
    operation_id,
)


def _example(
    input_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
    *,
    task_spec: TaskSpec | None = None,
) -> Example:
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=Program(steps=()),
        operation_graph=OperationGraph(nodes=()),
        category="known",
        split="train",
        vocab_size=6,
        task_spec=task_spec,
    )


def test_encode_example_structure() -> None:
    specials = build_special_tokens(6)
    example = _example((1, 2, 3), (4, 5))
    encoded = encode_example(example, specials)
    assert encoded == (specials.bos, 1, 2, 3, specials.sep, 4, 5, specials.eos)


def test_encode_example_excludes_oracle_metadata_from_model_input() -> None:
    """Oracle K/C/N/R labels and decompositions cannot become an input shortcut."""
    specials = build_special_tokens(6)
    plain = _example((1, 2, 3), (4, 5))
    oracle_annotated = replace(
        plain,
        oracle_metadata=OracleMetadata(
            label=ORACLE_LABEL_KNOWN,
            primitive_operations=("COPY",),
        ),
    )
    assert encode_example(oracle_annotated, specials) == encode_example(plain, specials)


def test_encode_example_excludes_symbol_permutation_from_model_input() -> None:
    """The permutation mapping/inverse cannot become an input shortcut: only
    the already-permuted `input_tokens`/`target_tokens` may reach the model
    (Phase A.1 Task A1-004)."""
    specials = build_special_tokens(6)
    plain = _example((1, 2, 3), (4, 5))
    permuted_annotated = replace(
        plain,
        symbol_permutation=SymbolPermutation(forward=(2, 0, 1, 4, 5, 3)),
    )
    assert encode_example(permuted_annotated, specials) == encode_example(plain, specials)


def test_collate_batch_rejects_empty_input() -> None:
    specials = build_special_tokens(6)
    with pytest.raises(ValueError, match="non-empty"):
        collate_batch([], specials)


def test_collate_batch_labels_mask_prompt_and_pad() -> None:
    specials = build_special_tokens(6)
    short = _example((1, 2, 3), (4, 5))  # seq len 8
    long = _example((1, 1, 2, 2), (3, 3, 4))  # seq len 10

    batch = collate_batch([short, long], specials)

    # input_ids width is max_seq_len - 1 = 9.
    assert batch.input_ids.shape == (2, 9)
    assert batch.labels.shape == (2, 9)
    assert batch.prompt_lengths == (5, 6)
    assert batch.sequence_lengths == (8, 10)

    # Row 0 (short): BOS 1 2 3 SEP [4 5 EOS] then PAD padding.
    expected_input_0 = torch.tensor(
        [
            specials.bos,
            1,
            2,
            3,
            specials.sep,
            4,
            5,
            specials.eos,
            specials.pad,
        ]
    )
    assert torch.equal(batch.input_ids[0], expected_input_0)
    expected_labels_0 = torch.tensor(
        [
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            4,
            5,
            specials.eos,
            IGNORE_INDEX,
            IGNORE_INDEX,
        ]
    )
    assert torch.equal(batch.labels[0], expected_labels_0)

    # Row 1 (long): BOS 1 1 2 2 SEP [3 3 4 EOS], no padding needed.
    expected_input_1 = torch.tensor(
        [specials.bos, 1, 1, 2, 2, specials.sep, 3, 3, 4]
    )
    assert torch.equal(batch.input_ids[1], expected_input_1)
    expected_labels_1 = torch.tensor(
        [
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            3,
            3,
            4,
            specials.eos,
        ]
    )
    assert torch.equal(batch.labels[1], expected_labels_1)


def test_collate_batch_padding_never_attends_forward() -> None:
    """Padded positions come after every real position, so causal attention
    over `input_ids` cannot let a real prediction see a PAD token."""
    specials = build_special_tokens(6)
    short = _example((1,), (2,))
    long = _example((1, 1, 2, 2), (3, 3, 4))
    batch = collate_batch([short, long], specials)
    pad_positions = (batch.input_ids[0] == specials.pad).nonzero(as_tuple=True)[0]
    real_positions = (batch.input_ids[0] != specials.pad).nonzero(as_tuple=True)[0]
    assert pad_positions.numel() > 0
    assert int(pad_positions.min()) > int(real_positions.max())


# --- Shared-core task-specification segment (Phase A.1 Correction Task
# --- A1-C003) ----------------------------------------------------------------

VOCAB_SIZE = 6
NUM_OPS = num_registered_operations()


def _shared_tokens(arg_span: int = 6):
    return build_shared_core_tokens(VOCAB_SIZE, num_operations=NUM_OPS, arg_span=arg_span)


def test_encode_task_spec_parameter_free_operation_has_no_argument_tokens() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="COPY", arguments={}),))
    encoded = encode_task_spec(spec, tokens)
    assert encoded == (
        tokens.task_start,
        tokens.operation_token(operation_id("COPY")),
        tokens.task_end,
    )


def test_encode_task_spec_scalar_argument_operation() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 2}),))
    encoded = encode_task_spec(spec, tokens)
    assert encoded == (
        tokens.task_start,
        tokens.operation_token(operation_id("SHIFT")),
        tokens.argument_value_token(2),
        tokens.task_end,
    )


def test_encode_task_spec_list_argument_operation_preserves_order() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="SELECT", arguments={"indices": [0, 2, 4]}),))
    encoded = encode_task_spec(spec, tokens)
    assert encoded == (
        tokens.task_start,
        tokens.operation_token(operation_id("SELECT")),
        tokens.argument_value_token(0),
        tokens.argument_value_token(2),
        tokens.argument_value_token(4),
        tokens.task_end,
    )


def test_encode_task_spec_empty_program_is_just_start_end_tokens() -> None:
    tokens = _shared_tokens()
    encoded = encode_task_spec(TaskSpec(steps=()), tokens)
    assert encoded == (tokens.task_start, tokens.task_end)


def test_encode_task_spec_multi_step_program_concatenates_steps() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(
        steps=(
            TaskStepSpec(operation="NEGATE", arguments={}),
            TaskStepSpec(operation="SHIFT", arguments={"amount": 1}),
        )
    )
    encoded = encode_task_spec(spec, tokens)
    assert encoded == (
        tokens.task_start,
        tokens.operation_token(operation_id("NEGATE")),
        tokens.operation_token(operation_id("SHIFT")),
        tokens.argument_value_token(1),
        tokens.task_end,
    )


def test_encode_example_includes_task_segment_between_bos_and_input() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 2}),))
    example = _example((1, 2, 3), (4, 5), task_spec=spec)
    encoded = encode_example(example, tokens, include_task_spec=True)
    assert encoded == (
        (tokens.bos,) + encode_task_spec(spec, tokens) + (1, 2, 3, tokens.sep, 4, 5, tokens.eos)
    )


def test_encode_example_include_task_spec_requires_shared_core_tokens() -> None:
    specials = build_special_tokens(VOCAB_SIZE)
    spec = TaskSpec(steps=(TaskStepSpec(operation="COPY", arguments={}),))
    example = _example((1, 2, 3), (4, 5), task_spec=spec)
    with pytest.raises(TypeError, match="SharedCoreTokens"):
        encode_example(example, specials, include_task_spec=True)


def test_encode_example_include_task_spec_requires_task_spec_present() -> None:
    tokens = _shared_tokens()
    example = _example((1, 2, 3), (4, 5))  # task_spec=None
    with pytest.raises(ValueError, match="task_spec"):
        encode_example(example, tokens, include_task_spec=True)


def test_encode_example_default_excludes_task_spec_even_when_present() -> None:
    """`include_task_spec` defaults to `False`, so pre-existing call sites
    (in particular Task A1-006's per-operation gate) keep producing the same
    token sequence regardless of whether `task_spec` happens to be set."""
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 2}),))
    with_spec = _example((1, 2, 3), (4, 5), task_spec=spec)
    without_spec = _example((1, 2, 3), (4, 5))
    assert encode_example(with_spec, tokens) == encode_example(without_spec, tokens)


def test_encode_example_identical_content_different_task_spec_yields_different_sequences() -> None:
    """Mirrors `docs/DECISIONS.md` ADR-0017's fix: once the task segment is
    included, two examples with identical content but a different sampled
    argument are no longer encoded identically."""
    tokens = _shared_tokens()
    spec_a = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 1}),))
    spec_b = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 3}),))
    example_a = _example((1, 2, 3), (2, 3, 1), task_spec=spec_a)
    example_b = _example((1, 2, 3), (2, 3, 1), task_spec=spec_b)  # same content/target shape
    encoded_a = encode_example(example_a, tokens, include_task_spec=True)
    encoded_b = encode_example(example_b, tokens, include_task_spec=True)
    assert encoded_a != encoded_b


def test_build_prompt_tokens_is_encode_example_prefix() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="COUNT", arguments={"target": 3}),))
    example = _example((1, 2, 3), (4,), task_spec=spec)
    prompt = build_prompt_tokens(example, tokens, include_task_spec=True)
    full = encode_example(example, tokens, include_task_spec=True)
    assert full[: len(prompt)] == prompt
    assert full[len(prompt) :] == example.target_tokens + (tokens.eos,)


def test_collate_batch_include_task_spec_accounts_for_task_segment_in_prompt_length() -> None:
    tokens = _shared_tokens()
    spec = TaskSpec(steps=(TaskStepSpec(operation="SHIFT", arguments={"amount": 1}),))
    example = _example((1, 2, 3), (4, 5), task_spec=spec)
    batch = collate_batch([example], tokens, include_task_spec=True)
    expected_prompt_len = len(build_prompt_tokens(example, tokens, include_task_spec=True))
    assert batch.prompt_lengths == (expected_prompt_len,)
    # Answer span (target + EOS) must still be exactly what's unmasked.
    unmasked = batch.labels[0][batch.labels[0] != IGNORE_INDEX]
    assert tuple(unmasked.tolist()) == example.target_tokens + (tokens.eos,)
