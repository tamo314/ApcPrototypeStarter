from __future__ import annotations

import pytest

from apc.core.tokens import build_special_tokens


def test_special_token_ids_are_contiguous_after_env_vocab() -> None:
    specials = build_special_tokens(10)
    assert specials.pad == 10
    assert specials.bos == 11
    assert specials.sep == 12
    assert specials.eos == 13
    assert specials.model_vocab_size == 14


def test_special_tokens_never_overlap_env_vocab() -> None:
    specials = build_special_tokens(6)
    special_ids = {specials.pad, specials.bos, specials.sep, specials.eos}
    assert len(special_ids) == 4  # all distinct
    assert all(token_id >= specials.env_vocab_size for token_id in special_ids)


def test_invalid_vocab_size_raises() -> None:
    with pytest.raises(ValueError, match="env_vocab_size"):
        build_special_tokens(0)
