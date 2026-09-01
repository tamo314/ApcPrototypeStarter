from __future__ import annotations

import pytest

from apc.core.tokens import (
    SharedCoreTokens,
    SpecialTokens,
    build_shared_core_tokens,
    build_special_tokens,
)


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


# --- SharedCoreTokens (Phase A.1 Correction Task A1-C003) -------------------


def test_shared_core_tokens_reuses_plain_special_token_ids() -> None:
    """PAD/BOS/SEP/EOS must be identical between the two schemes for the same
    `env_vocab_size` -- `SharedCoreTokens` is a strict superset, not a
    parallel numbering."""
    plain = build_special_tokens(10)
    shared = build_shared_core_tokens(10, num_operations=8, arg_span=10)
    assert (shared.pad, shared.bos, shared.sep, shared.eos) == (
        plain.pad,
        plain.bos,
        plain.sep,
        plain.eos,
    )


def test_shared_core_tokens_ranges_are_contiguous_and_disjoint() -> None:
    shared = build_shared_core_tokens(10, num_operations=8, arg_span=10)
    assert shared.task_start == shared.eos + 1
    assert shared.task_end == shared.task_start + 1
    assert shared.op_base == shared.task_end + 1
    assert shared.arg_base == shared.op_base + 8
    assert shared.model_vocab_size == shared.arg_base + 10

    all_ids = (
        [shared.pad, shared.bos, shared.sep, shared.eos, shared.task_start, shared.task_end]
        + [shared.operation_token(i) for i in range(8)]
        + [shared.argument_value_token(v) for v in range(10)]
    )
    assert len(all_ids) == len(set(all_ids))  # every id distinct
    assert max(all_ids) == shared.model_vocab_size - 1


def test_operation_token_rejects_out_of_range_id() -> None:
    shared = build_shared_core_tokens(10, num_operations=8, arg_span=10)
    with pytest.raises(ValueError, match="op_id"):
        shared.operation_token(8)
    with pytest.raises(ValueError, match="op_id"):
        shared.operation_token(-1)


def test_argument_value_token_rejects_out_of_range_value() -> None:
    shared = build_shared_core_tokens(10, num_operations=8, arg_span=10)
    with pytest.raises(ValueError, match="argument value"):
        shared.argument_value_token(10)
    with pytest.raises(ValueError, match="argument value"):
        shared.argument_value_token(-1)


def test_shared_core_tokens_rejects_non_positive_spans() -> None:
    with pytest.raises(ValueError, match="num_operations"):
        build_shared_core_tokens(10, num_operations=0, arg_span=10)
    with pytest.raises(ValueError, match="arg_span"):
        build_shared_core_tokens(10, num_operations=8, arg_span=0)


def test_shared_core_tokens_is_a_special_tokens() -> None:
    shared = build_shared_core_tokens(10, num_operations=8, arg_span=10)
    assert isinstance(shared, SpecialTokens)
    assert isinstance(shared, SharedCoreTokens)
