"""Shared token vocabulary for the Phase A symbolic task environment.

All operations read and write a single closed integer vocabulary
`0 .. vocab_size - 1`. Relation-valued operations (e.g. COMPARE) reuse the
same vocabulary rather than a separate token space, so the output of any
operation can always be fed into any other operation without a domain
conversion step.
"""

from __future__ import annotations

DEFAULT_VOCAB_SIZE = 10
MIN_VOCAB_SIZE = 6  # 3 reserved relation tokens + at least 3 value tokens


def relation_tokens(vocab_size: int) -> tuple[int, int, int]:
    """Return the (LT, EQ, GT) tokens reserved at the top of the vocabulary."""
    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be >= {MIN_VOCAB_SIZE}, got {vocab_size}")
    return vocab_size - 3, vocab_size - 2, vocab_size - 1
