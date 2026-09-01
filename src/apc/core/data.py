"""Batching of `apc.environments.generator.Example`s for the baseline model.

Each example is encoded as a single sequence `[BOS] input... [SEP] target...
[EOS]` and trained autoregressively with the loss masked to only the answer
span (`target... [EOS]`), so the model is never rewarded for "predicting"
the prompt it was given.

Sequences are right-padded with PAD to the batch's max length. Because
padding is on the right and attention is causal, a real token's prediction
never attends to a later PAD position, so no separate key-padding mask is
needed for correctness (see `apc.core.model.DecoderOnlyTransformer`).

Shared-core task-specification segment (Phase A.1 Correction Task A1-C003,
`include_task_spec=True`): `encode_example`/`collate_batch` can optionally
render `Example.task_spec` into a model-visible token segment between `BOS`
and the content input -- `[BOS] [TASK] op arg... [/TASK] input... [SEP]
target... [EOS]` -- via `encode_task_spec`. This requires a
`apc.core.tokens.SharedCoreTokens` in place of a plain `SpecialTokens` and
is off by default, so every existing call site (in particular Task A1-006's
per-operation gate, `apc.evaluation.stable_core_generalization`) keeps
producing the exact same token sequence it always has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from apc.core.tokens import SharedCoreTokens, SpecialTokens
from apc.environments.generator import Example
from apc.environments.task_spec import TaskSpec

IGNORE_INDEX = -100  # torch.nn.functional.cross_entropy default ignore_index


def _argument_values(arguments: dict[str, Any]) -> tuple[int, ...]:
    """Flatten one `TaskStepSpec.arguments` dict into raw integer values, in
    a deterministic (sorted-by-key) order.

    Every current operation argument is either a plain `int` (`SHIFT.amount`,
    `COUNT.target`, `BIND.query_key`) or a `list[int]` of positions
    (`SELECT.indices`, already sorted by `SelectOp.sample_params`) -- see
    `apc.environments.operations`. Sorting by key (rather than relying on
    dict insertion order) keeps this a pure function of the argument
    mapping's content, not of how `ProgramStep.params`/`sample_params`
    happened to construct it; it is a no-op for every current operation
    since none has more than one argument key.
    """
    values: list[int] = []
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, int) or isinstance(item, bool):
                    raise TypeError(
                        f"Unsupported task-spec argument element for {key!r}: {item!r}"
                    )
                values.append(item)
        else:
            raise TypeError(f"Unsupported task-spec argument value for {key!r}: {value!r}")
    return tuple(values)


def encode_task_spec(task_spec: TaskSpec, tokens: SharedCoreTokens) -> tuple[int, ...]:
    """Render a `TaskSpec` into a `[TASK_START] op arg... op arg... [TASK_END]`
    token segment.

    One step is `operation_token(step.operation_id)` followed by that step's
    argument values (empty for the four parameter-free operations). Step
    boundaries need no explicit separator: operation tokens and argument-value
    tokens occupy disjoint `SharedCoreTokens` ranges, so an operation token is
    always unambiguously the start of the next step.
    """
    step_tokens: list[int] = []
    for step in task_spec.steps:
        step_tokens.append(tokens.operation_token(step.operation_id))
        step_tokens.extend(tokens.argument_value_token(v) for v in _argument_values(step.arguments))
    return (tokens.task_start, *step_tokens, tokens.task_end)


def build_prompt_tokens(
    example: Example, specials: SpecialTokens, *, include_task_spec: bool = False
) -> tuple[int, ...]:
    """Build the `[BOS] [task segment] input... [SEP]` prompt shared by
    `encode_example` (prompt + target + EOS, for training) and
    `apc.core.generation.evaluate_exact_match` (prompt alone, as the greedy
    decoding seed) -- kept as one function so both always render the task
    segment identically.
    """
    task_segment: tuple[int, ...] = ()
    if include_task_spec:
        if not isinstance(specials, SharedCoreTokens):
            raise TypeError("include_task_spec=True requires SharedCoreTokens, not SpecialTokens")
        if example.task_spec is None:
            raise ValueError("include_task_spec=True requires example.task_spec to be set")
        task_segment = encode_task_spec(example.task_spec, specials)
    return (specials.bos,) + task_segment + example.input_tokens + (specials.sep,)


def encode_example(
    example: Example, specials: SpecialTokens, *, include_task_spec: bool = False
) -> tuple[int, ...]:
    """Build the full `[BOS] [task segment] input... [SEP] target... [EOS]`
    token sequence. The task segment is included only when
    `include_task_spec=True` (see module docstring)."""
    return (
        build_prompt_tokens(example, specials, include_task_spec=include_task_spec)
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


def build_task_only_tokens(task_spec: TaskSpec, tokens: SharedCoreTokens) -> tuple[int, ...]:
    """`[BOS] [TASK_START] op arg... [TASK_END]` -- the task specification
    alone, with no content token anywhere in the sequence (Phase A.1
    Post-Correction Task A1-R001's task-only encoding path; see
    `apc.core.model.DecoderOnlyTransformer.encode_task_content_split`).

    Unlike `build_prompt_tokens(..., include_task_spec=True)`, which renders
    the task segment as a *prefix* to the content input within one combined
    sequence, this never includes `example.input_tokens` at all: there is no
    content for a later causal position to attend back to, so a `z_task`
    read from this sequence has no computational path to content, by
    construction rather than by measurement (`docs/design-docs/
    CAUSAL_PRIMITIVE_EXECUTION.md` section 3).
    """
    return (tokens.bos,) + encode_task_spec(task_spec, tokens)


def build_content_only_tokens(example: Example, specials: SpecialTokens) -> tuple[int, ...]:
    """`[BOS] input... [SEP]` -- the content alone, with no task-segment
    token anywhere in the sequence (Task A1-R001's content-only encoding
    path).

    Exactly `build_prompt_tokens(example, specials, include_task_spec=False)`
    -- given its own name here because Task A1-R001 callers care
    specifically that the result is task-blind, not merely that it happens
    to share a shape with the pre-A1-C003 prompt. Its signature (no
    `TaskSpec`/task-spec parameter at all) makes "content-only encoding
    cannot depend on which task was requested" a property of the type
    system rather than something a test has to catch after the fact.
    """
    return build_prompt_tokens(example, specials, include_task_spec=False)


def pad_token_sequences(
    sequences: Sequence[tuple[int, ...]], pad_id: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Right-pad a list of token-id tuples to their shared max length.

    Plain encoding-only padding (no labels/prompt-length bookkeeping, unlike
    `collate_batch`'s `Batch`) for callers that only need a batched
    `input_ids`-shaped tensor to run through `DecoderOnlyTransformer.encode`
    -- see `collate_task_only_batch`/`collate_content_only_batch` below.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")
    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded.to(device)


def collate_task_only_batch(
    task_specs: Sequence[TaskSpec], tokens: SharedCoreTokens, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Batch `build_task_only_tokens` over `task_specs`, right-padded to the
    batch's max length (Task A1-R001)."""
    sequences = [build_task_only_tokens(spec, tokens) for spec in task_specs]
    return pad_token_sequences(sequences, tokens.pad, device)


def collate_content_only_batch(
    examples: Sequence[Example], specials: SpecialTokens, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Batch `build_content_only_tokens` over `examples`, right-padded to the
    batch's max length (Task A1-R001)."""
    sequences = [build_content_only_tokens(example, specials) for example in examples]
    return pad_token_sequences(sequences, specials.pad, device)


def collate_batch(
    examples: list[Example],
    specials: SpecialTokens,
    device: torch.device | str = "cpu",
    *,
    include_task_spec: bool = False,
) -> Batch:
    """Encode and right-pad a list of examples into one training `Batch`."""
    if not examples:
        raise ValueError("examples must be non-empty")

    sequences = [
        encode_example(example, specials, include_task_spec=include_task_spec)
        for example in examples
    ]
    max_len = max(len(seq) for seq in sequences)

    padded = torch.full((len(sequences), max_len), specials.pad, dtype=torch.long)
    labels = torch.full((len(sequences), max_len - 1), IGNORE_INDEX, dtype=torch.long)
    prompt_lengths = []
    sequence_lengths = []
    for row, (example, seq) in enumerate(zip(examples, sequences, strict=True)):
        padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        # prompt = BOS + [task segment] + input + SEP, i.e. everything before
        # the answer span (target tokens, then EOS) -- computed from the
        # answer span's own length rather than re-deriving the prompt's
        # composition, so it stays correct whether or not a task segment is
        # present.
        prompt_len = len(seq) - len(example.target_tokens) - 1
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
