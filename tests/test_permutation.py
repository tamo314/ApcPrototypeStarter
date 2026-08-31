"""Symbol permutation unit tests (Phase A.1 Task A1-004)."""

from __future__ import annotations

import random

import pytest

from apc.environments.permutation import SymbolPermutation, sample_permutation

VOCAB_SIZE = 10


def test_sample_permutation_is_a_bijection() -> None:
    permutation = sample_permutation(random.Random(0), VOCAB_SIZE)
    assert sorted(permutation.forward) == list(range(VOCAB_SIZE))


def test_sample_permutation_is_deterministic_given_same_rng_state() -> None:
    first = sample_permutation(random.Random(7), VOCAB_SIZE)
    second = sample_permutation(random.Random(7), VOCAB_SIZE)
    assert first == second


def test_invert_recovers_canonical_tokens() -> None:
    permutation = sample_permutation(random.Random(3), VOCAB_SIZE)
    canonical = tuple(range(VOCAB_SIZE))
    presented = permutation.apply(canonical)
    assert permutation.invert(presented) == canonical


def test_apply_then_invert_is_identity_for_arbitrary_tokens() -> None:
    permutation = sample_permutation(random.Random(11), VOCAB_SIZE)
    tokens = (0, 0, 3, 9, 5, 5, 1)
    assert permutation.invert(permutation.apply(tokens)) == tokens


def test_different_draws_usually_produce_different_mappings() -> None:
    first = sample_permutation(random.Random(1), VOCAB_SIZE)
    second = sample_permutation(random.Random(2), VOCAB_SIZE)
    assert first.forward != second.forward


def test_rejects_non_permutation_forward_mapping() -> None:
    with pytest.raises(ValueError):
        SymbolPermutation(forward=(0, 0, 2))  # not a bijection: 1 missing, 0 repeated
    with pytest.raises(ValueError):
        SymbolPermutation(forward=(0, 1, 3))  # 2 missing, 3 out of range


def test_to_dict_is_json_serializable_permutation_identity() -> None:
    permutation = sample_permutation(random.Random(4), VOCAB_SIZE)
    payload = permutation.to_dict()
    assert payload == {"forward": list(permutation.forward)}
    assert isinstance(payload["forward"], list)


def test_sample_permutation_rejects_non_positive_vocab_size() -> None:
    with pytest.raises(ValueError):
        sample_permutation(random.Random(0), 0)


# --- semantic invariance across mappings (A1-004 acceptance criterion) -----


def test_same_abstract_result_decodes_identically_under_different_mappings() -> None:
    """The same canonical output, presented under two independent random
    mappings, must decode back to the same canonical tokens even though the
    raw presented tokens differ -- the abstract task is mapping-invariant."""
    canonical_output = (0, VOCAB_SIZE - 1, 4, VOCAB_SIZE - 2, 2)  # includes relation-token ids
    mapping_a = sample_permutation(random.Random(101), VOCAB_SIZE)
    mapping_b = sample_permutation(random.Random(202), VOCAB_SIZE)

    presented_a = mapping_a.apply(canonical_output)
    presented_b = mapping_b.apply(canonical_output)

    assert mapping_a.invert(presented_a) == canonical_output
    assert mapping_b.invert(presented_b) == canonical_output
    # Different mappings normally disagree on how the *same* abstract result
    # is spelled out in raw token ids -- otherwise a fixed token id could
    # still act as a shortcut for a fixed semantic role.
    assert presented_a != presented_b
