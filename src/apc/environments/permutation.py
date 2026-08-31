"""Symbol permutation: an anti-shortcut control (Phase A.1 Task A1-004).

`apc.environments.generator.TaskGenerator` always computes operation
semantics -- order comparisons, modular arithmetic, relation tokens -- in a
fixed canonical vocabulary `0 .. vocab_size - 1`, via the unmodified
`apc.environments.interpreter.run_program`. Without any further step, that
canonical id is also exactly what the model sees, so a stable token id (for
example the fixed relation-token ids from `apc.environments.vocab`) can
encode part of the answer on its own, independent of the input content.

`SymbolPermutation` relabels the already-computed canonical input/target
tokens with a fresh random bijection before they are presented to the
model. The interpreter and all operation logic never see the permuted
domain, so operation semantics are untouched; only the concrete token ids a
learner could try to memorize change from example to example.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolPermutation:
    """A bijection over the shared token vocabulary `0 .. vocab_size - 1`.

    `forward[v]` is the presented id shown to the model for canonical value
    `v`. This is latent decoding/evaluation metadata, analogous to
    `apc.environments.generator.OracleMetadata`: it must never be fed into
    `apc.core.data.encode_example` or otherwise reach the model as an
    additional input, only used by evaluation/oracle code to decode a
    prediction back to canonical ids or to log which mapping was used.
    """

    forward: tuple[int, ...]

    def __post_init__(self) -> None:
        if sorted(self.forward) != list(range(len(self.forward))):
            raise ValueError("forward must be a permutation of range(len(forward))")

    @property
    def vocab_size(self) -> int:
        return len(self.forward)

    @property
    def inverse(self) -> tuple[int, ...]:
        """The inverse mapping: `inverse[presented] == canonical`."""
        inverse = [0] * len(self.forward)
        for canonical, presented in enumerate(self.forward):
            inverse[presented] = canonical
        return tuple(inverse)

    def apply(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        """Relabel canonical tokens to the ids presented to the model."""
        return tuple(self.forward[token] for token in tokens)

    def invert(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        """Recover canonical tokens from presented ids (decoding, eval-only)."""
        inverse = self.inverse
        return tuple(inverse[token] for token in tokens)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable permutation identity, for run logs (eval-only)."""
        return {"forward": list(self.forward)}


def sample_permutation(rng: random.Random, vocab_size: int) -> SymbolPermutation:
    """Draw a uniformly random symbol permutation over `0 .. vocab_size - 1`."""
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    forward = list(range(vocab_size))
    rng.shuffle(forward)
    return SymbolPermutation(forward=tuple(forward))
