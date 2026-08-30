from __future__ import annotations

import pytest
import torch

from apc.core.data import IGNORE_INDEX, collate_batch, encode_example
from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example
from apc.environments.interpreter import OperationGraph
from apc.environments.program import Program


def _example(input_tokens: tuple[int, ...], target_tokens: tuple[int, ...]) -> Example:
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=Program(steps=()),
        operation_graph=OperationGraph(nodes=()),
        category="known",
        split="train",
        vocab_size=6,
    )


def test_encode_example_structure() -> None:
    specials = build_special_tokens(6)
    example = _example((1, 2, 3), (4, 5))
    encoded = encode_example(example, specials)
    assert encoded == (specials.bos, 1, 2, 3, specials.sep, 4, 5, specials.eos)


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
