"""Special tokens for the decoder-only baseline, layered on the environment vocab.

The symbolic environment (`apc.environments`) uses a closed vocabulary
`0 .. vocab_size - 1` with no reserved id for model-only control tokens.
`SpecialTokens` appends PAD/BOS/SEP/EOS after the environment vocabulary so
model input ids never collide with environment token ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    """Control token ids appended after the environment vocabulary."""

    env_vocab_size: int
    pad: int
    bos: int
    sep: int
    eos: int

    @property
    def model_vocab_size(self) -> int:
        return self.env_vocab_size + 4


def build_special_tokens(env_vocab_size: int) -> SpecialTokens:
    """Derive PAD/BOS/SEP/EOS ids immediately after the environment vocabulary."""
    if env_vocab_size < 1:
        raise ValueError(f"env_vocab_size must be >= 1, got {env_vocab_size}")
    return SpecialTokens(
        env_vocab_size=env_vocab_size,
        pad=env_vocab_size,
        bos=env_vocab_size + 1,
        sep=env_vocab_size + 2,
        eos=env_vocab_size + 3,
    )
