"""Shared-core input encoding tests (Phase A.1 Correction Task A1-C003).

Covers this task's acceptance criteria directly:

- "one model forward handles all known operations" -- a single
  `DecoderOnlyTransformer` forward pass over one batch that mixes every
  `KNOWN_OPERATION_NAMES` operation, encoded with `include_task_spec=True`
  (`test_single_forward_pass_handles_a_batch_mixing_every_known_operation`).
- "task/content states are available separately" -- `encode_split` (Task
  A1-005) already returns `EncodedState`; `test_core_model.py` covers its
  probe hook. Not re-tested here.
- "existing per-operation A1-006 path remains reproducible" -- this module
  never imports or modifies `apc.evaluation.stable_core_generalization`,
  and every new `apc.core.data`/`apc.core.generation` parameter defaults to
  `include_task_spec=False` (see `tests/test_core_data.py`'s
  `test_encode_example_default_excludes_task_spec_even_when_present`), so
  that module's own token sequences are byte-for-byte unaffected.

`test_shared_model_fits_a_mixed_operation_batch_including_a_parameterized_op`
is a small, fast overfitting smoke test (not a generalization gate -- that
is Task A1-C004's job) demonstrating the well-posedness fix ADR-0017 asked
for: with the task segment included, `SHIFT` (previously unlearnable from
input alone, ADR-0017) is now solvable by the same shared model alongside
parameter-free operations, in one training loop, with no per-operation
branching.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX, collate_batch
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.environments.generator import Example, build_mixed_operation_generator
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.environments.task_spec import default_argument_value_span, num_registered_operations
from apc.utils.seed import set_seed

VOCAB_SIZE = 10
LENGTH_RANGE = (4, 6)


def _shared_tokens() -> SharedCoreTokens:
    return build_shared_core_tokens(
        VOCAB_SIZE,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(VOCAB_SIZE, LENGTH_RANGE),
    )


def _tiny_model_config(tokens: SharedCoreTokens) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=tokens.model_vocab_size,
        max_seq_len=48,
        d_model=32,
        n_layer=2,
        n_head=2,
        d_ff=64,
        dropout=0.0,
    )


def _operation_sequence(example: Example) -> tuple[str, ...]:
    assert example.task_spec is not None
    return example.task_spec.operation_sequence


def test_every_mixed_operation_generator_example_carries_a_task_spec() -> None:
    """Precondition for `include_task_spec=True`: `build_mixed_operation_generator`
    (Task A1-C002) must attach `task_spec` to every example, not just some."""
    generator = build_mixed_operation_generator(
        seed=0, operation_names=KNOWN_OPERATION_NAMES, sequence_length_range=LENGTH_RANGE
    )
    examples = generator.generate_online(64, step=0, split="train")
    assert all(example.task_spec is not None for example in examples)


def test_single_forward_pass_handles_a_batch_mixing_every_known_operation() -> None:
    """The core A1-C003 claim: one `DecoderOnlyTransformer`, one forward call,
    one shared vocabulary -- no per-operation model, no per-operation branch
    -- over a batch that actually mixes multiple distinct operations."""
    set_seed(0)
    tokens = _shared_tokens()
    generator = build_mixed_operation_generator(
        seed=1, operation_names=KNOWN_OPERATION_NAMES, sequence_length_range=LENGTH_RANGE
    )
    examples = generator.generate_online(64, step=0, split="train")
    operations_present = {_operation_sequence(example) for example in examples}
    assert len(operations_present) > 1, "batch should mix multiple distinct operations"

    batch = collate_batch(examples, tokens, include_task_spec=True)

    model = DecoderOnlyTransformer(_tiny_model_config(tokens))
    logits = model(batch.input_ids)  # the one shared forward pass

    assert logits.shape == (
        len(examples),
        batch.input_ids.shape[1],
        tokens.model_vocab_size,
    )
    assert torch.isfinite(logits).all()


def test_shared_model_fits_a_mixed_operation_batch_including_a_parameterized_op() -> None:
    """Overfitting smoke test, not a generalization gate (that is Task
    A1-C004). `SHIFT` is one of ADR-0017's hidden-parameter operations --
    unlearnable at all without the task segment -- included here alongside
    parameter-free operations in one shared model/training loop."""
    set_seed(0)
    tokens = _shared_tokens()
    generator = build_mixed_operation_generator(
        seed=2,
        operation_names=("COPY", "NEGATE", "SHIFT"),
        sequence_length_range=LENGTH_RANGE,
    )
    examples = generator.generate(24, split="train")
    assert {"COPY", "NEGATE", "SHIFT"} <= {
        _operation_sequence(example)[0] for example in examples
    }

    model_config = _tiny_model_config(tokens)
    model = DecoderOnlyTransformer(model_config)
    set_seed(0)  # re-anchor after model construction, matching ADR-0016's pattern
    batch = collate_batch(examples, tokens, include_task_spec=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    initial_loss = None
    final_loss = float("nan")
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if initial_loss is None:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

    assert initial_loss is not None
    assert final_loss < initial_loss * 0.05
    assert final_loss < 0.1
