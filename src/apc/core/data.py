"""Batching of `apc.environments.generator.Example`s for the baseline model.

Each example is encoded as a single sequence `[BOS] input... [SEP] target...
[EOS]` and trained autoregressively with the loss masked to only the answer
span (`target... [EOS]`), so the model is never rewarded for "predicting"
the prompt it was given.

Sequences are right-padded with PAD to the batch's max length. Because
padding is on the right and attention is causal, a real token's prediction
never attends to a later PAD position, so no separate key-padding mask is
needed for correctness (see `apc.core.model.DecoderOnlyTransformer`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from apc.core.tokens import SpecialTokens
from apc.environments.generator import Example

IGNORE_INDEX = -100  # torch.nn.functional.cross_entropy default ignore_index


def encode_example(example: Example, specials: SpecialTokens) -> tuple[int, ...]:
    """Build the full `[BOS] input... [SEP] target... [EOS]` token sequence."""
    return (
        (specials.bos,)
        + example.input_tokens
        + (specials.sep,)
        + example.target_tokens
        + (specials.eos,)
    )


@dataclass(frozen=True)
class Batch:
    """A right-padded batch ready for `DecoderOnlyTransformer`.

    `input_ids` and `labels` both have shape `[batch, seq_len - 1]`: standard
    next-token teacher forcing over the padded sequence, with `labels`
    masked to `IGNORE_INDEX` outside each example's answer span.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    prompt_lengths: tuple[int, ...]
    sequence_lengths: tuple[int, ...]


def collate_batch(
    examples: list[Example], specials: SpecialTokens, device: torch.device | str = "cpu"
) -> Batch:
    """Encode and right-pad a list of examples into one training `Batch`."""
    if not examples:
        raise ValueError("examples must be non-empty")

    sequences = [encode_example(example, specials) for example in examples]
    max_len = max(len(seq) for seq in sequences)

    padded = torch.full((len(sequences), max_len), specials.pad, dtype=torch.long)
    labels = torch.full((len(sequences), max_len - 1), IGNORE_INDEX, dtype=torch.long)
    prompt_lengths = []
    sequence_lengths = []
    for row, (example, seq) in enumerate(zip(examples, sequences, strict=True)):
        padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        prompt_len = 1 + len(example.input_tokens) + 1  # BOS + input + SEP
        # Positions t in [prompt_len - 1, len(seq) - 2] predict the answer
        # span (target tokens, then EOS) via labels[t] = seq[t + 1].
        answer_start = prompt_len - 1
        answer_end = len(seq) - 1  # exclusive
        labels[row, answer_start:answer_end] = torch.tensor(
            seq[prompt_len : len(seq)], dtype=torch.long
        )
        prompt_lengths.append(prompt_len)
        sequence_lengths.append(len(seq))

    input_ids = padded[:, :-1]
    return Batch(
        input_ids=input_ids.to(device),
        labels=labels.to(device),
        prompt_lengths=tuple(prompt_lengths),
        sequence_lengths=tuple(sequence_lengths),
    )
