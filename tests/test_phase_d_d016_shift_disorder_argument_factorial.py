"""Unit tests for Task D-016's pure-logic helpers (no GPU/model access).

These exercise the controlled-inversion permutation constructor, the paired
disorder-level example generator, and the pre-fixed disorder x argument
factorial decision rule in isolation, plus one orchestration-level regression
(with all model-loading and gate functions mocked, writing only to a tmp_path
namespace) that guards the amount/content confound fix specifically at the
``run`` call-graph level. Loading the actual D-013 model bundles and running
the full per-cell diagnostic against real D-014/D-015 artifacts requires CUDA
and those artifacts; that is exercised by the real one-time diagnostic run,
not by this CPU-only regression suite.
"""

from __future__ import annotations

import importlib
import inspect
import random
from pathlib import Path

import pytest

from apc.evaluation.phase_d_d016_shift_disorder_argument_factorial import (
    ADEQUACY_FLOOR,
    DISORDER_LEVELS,
    EFFECT_MARGIN,
    D016DiagnosticError,
    DisorderArgumentCell,
    _count_inversions,
    _disorder_targets,
    _paired_disorder_examples,
    _permutation_with_inversions,
    classify_seed_length,
)

# =============================================================================
# Controlled-inversion permutation construction
# =============================================================================


def test_disorder_targets_length_6() -> None:
    targets = _disorder_targets(6)
    assert targets == {
        "ASCENDING": 0,
        "LOW_INVERSION": 4,
        "MEDIUM_INVERSION": 8,
        "HIGH_INVERSION": 11,
        "DESCENDING": 15,
    }


def test_disorder_targets_length_10() -> None:
    targets = _disorder_targets(10)
    assert targets == {
        "ASCENDING": 0,
        "LOW_INVERSION": 11,
        "MEDIUM_INVERSION": 22,
        "HIGH_INVERSION": 34,
        "DESCENDING": 45,
    }


def test_permutation_with_zero_inversions_is_ascending() -> None:
    rng = random.Random(1)
    elements = [2, 5, 7, 9]
    perm = _permutation_with_inversions(rng, elements, 0)
    assert perm == tuple(elements)
    assert _count_inversions(perm) == 0


def test_permutation_with_max_inversions_is_descending() -> None:
    rng = random.Random(1)
    elements = [2, 5, 7, 9]
    max_inversions = len(elements) * (len(elements) - 1) // 2
    perm = _permutation_with_inversions(rng, elements, max_inversions)
    assert perm == tuple(reversed(elements))
    assert _count_inversions(perm) == max_inversions


@pytest.mark.parametrize("n", [3, 4, 6, 10])
def test_permutation_hits_exact_inversion_count_for_every_valid_target(n: int) -> None:
    rng = random.Random(42)
    elements = list(range(n))
    max_inversions = n * (n - 1) // 2
    for target in range(max_inversions + 1):
        perm = _permutation_with_inversions(rng, elements, target)
        assert sorted(perm) == elements
        assert _count_inversions(perm) == target


def test_permutation_rejects_out_of_range_target() -> None:
    rng = random.Random(1)
    with pytest.raises(D016DiagnosticError):
        _permutation_with_inversions(rng, [1, 2, 3], 100)
    with pytest.raises(D016DiagnosticError):
        _permutation_with_inversions(rng, [1, 2, 3], -1)


# =============================================================================
# Paired disorder-level example generation
# =============================================================================


def test_paired_disorder_examples_share_multiset_across_levels() -> None:
    contents = _paired_disorder_examples(length=6, eval_seed=301, n=25)
    assert set(contents) == set(DISORDER_LEVELS)
    for i in range(25):
        multisets = {tuple(sorted(contents[level][i])) for level in DISORDER_LEVELS}
        assert len(multisets) == 1, "all 5 disorder levels must share the same multiset per item"


def test_paired_disorder_examples_hit_exact_disorder_targets() -> None:
    contents = _paired_disorder_examples(length=6, eval_seed=301, n=25)
    targets = _disorder_targets(6)
    for level in DISORDER_LEVELS:
        for content in contents[level]:
            assert _count_inversions(content) == targets[level]


def test_paired_disorder_examples_deterministic_given_same_inputs() -> None:
    first = _paired_disorder_examples(6, 301, 10)
    second = _paired_disorder_examples(6, 301, 10)
    assert first == second


def test_paired_disorder_examples_length_10_uses_full_vocab() -> None:
    contents = _paired_disorder_examples(length=10, eval_seed=301, n=5)
    for level in DISORDER_LEVELS:
        for content in contents[level]:
            assert sorted(content) == list(range(10))


# =============================================================================
# Amount/content confound removal (invariant coverage)
# =============================================================================


def test_paired_disorder_examples_signature_has_no_amount_parameter() -> None:
    """Content generation must not accept (or key on) a SHIFT amount at all --
    this is what makes reuse across every amount value structurally safe,
    rather than merely accidental. A future regression that reintroduces an
    ``amount`` parameter must fail this test."""
    params = set(inspect.signature(_paired_disorder_examples).parameters)
    assert "amount" not in params
    assert params == {"length", "eval_seed", "n"}


def test_paired_disorder_examples_content_is_amount_invariant_by_construction() -> None:
    """The same call, standing in for content reused across every valid
    amount for a given length, is byte-for-byte identical regardless of how
    many times or in what order it is invoked -- i.e. no caller can
    reintroduce the amount/content confound by calling this per-amount
    inside a loop, since amount is not part of the seed derivation."""
    calls = [_paired_disorder_examples(6, 301, 25) for _ in range(6)]
    assert all(call == calls[0] for call in calls[1:])


def test_run_reuses_one_paired_content_draw_across_every_amount(monkeypatch, tmp_path) -> None:
    """Orchestration-level regression for the exact confound the task
    targets: ``run`` must call ``_paired_disorder_examples`` exactly once
    per (length, eval_seed) -- never once per (length, eval_seed, amount) --
    and every amount's evaluation must observe the identical disorder
    content dict returned by that single call. All model-loading and gate
    machinery is mocked; this test never touches CUDA or real D-013/D-014/
    D-015 artifacts, and writes only into ``tmp_path``, never the real
    (still-unused) D-016 output namespace."""
    import types

    d016 = importlib.import_module(
        "apc.evaluation.phase_d_d016_shift_disorder_argument_factorial"
    )

    monkeypatch.setattr(d016, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(d016, "OUTPUT_ROOT", Path("out"))
    monkeypatch.setattr(d016, "MODEL_SEEDS", (40,))
    monkeypatch.setattr(d016, "EVAL_SEEDS", (301, 302))
    monkeypatch.setattr(d016, "LENGTHS", (6,))

    monkeypatch.setattr(d016, "verify_d015_parity", lambda seed: {"seed": seed, "match": True})
    monkeypatch.setattr(
        d016,
        "load_bundle_for_condition",
        lambda seed, condition, device: (None, None, None, None),
    )
    monkeypatch.setattr(
        d016,
        "verify_d014_parity",
        lambda core, bank, op_to_id, seed, condition: {
            "seed": seed,
            "condition": condition,
            "match": True,
            "step0": {},
        },
    )

    content_calls: list[tuple[int, int, int]] = []
    contents_by_call: list[dict[str, list[tuple[int, ...]]]] = []
    original_generator = d016._paired_disorder_examples

    def spy_generator(length: int, eval_seed: int, n: int) -> dict[str, list[tuple[int, ...]]]:
        content_calls.append((length, eval_seed, n))
        result = original_generator(length, eval_seed, n)
        contents_by_call.append(result)
        return result

    monkeypatch.setattr(d016, "_paired_disorder_examples", spy_generator)

    observed_content_ids: list[int] = []

    def fake_standalone_arms(core, bank, op_to_id, op_name, contents, arg_dicts, vocab_size):
        observed_content_ids.append(id(contents))
        n = len(contents)
        arm = types.SimpleNamespace(
            n=n, correct_em=1.0, wrong_family_em=0.0, none_em=0.0, wrong_argument_em=0.0
        )
        return arm, [True] * n

    monkeypatch.setattr(d016, "standalone_arms", fake_standalone_arms)

    report = d016.run(device_str="cpu")

    assert report["result"] == "COMPLETED"
    amounts = list(range(6))  # length=6 -> amounts 0..5
    # Exactly one content draw per (length, eval_seed): 1 length x 2 eval seeds = 2,
    # never multiplied by len(amounts) as it would be under the pre-fix confound.
    assert content_calls == [(6, 301, 1000), (6, 302, 1000)]
    # Every amount x disorder_level evaluation for a given eval_seed observed the
    # exact same content list object drawn once for that eval_seed -- proving reuse,
    # not just equal-by-value regeneration.
    calls_per_eval_seed = len(amounts) * len(d016.DISORDER_LEVELS)
    assert len(observed_content_ids) == len(d016.EVAL_SEEDS) * calls_per_eval_seed
    for i, eval_call in enumerate(contents_by_call):
        ids_for_this_eval_seed = observed_content_ids[
            i * calls_per_eval_seed : (i + 1) * calls_per_eval_seed
        ]
        expected_ids = {id(eval_call[level]) for level in d016.DISORDER_LEVELS}
        assert set(ids_for_this_eval_seed) == expected_ids


# =============================================================================
# Fixed disorder x argument factorial decision rule
# =============================================================================


def _cell(
    seed: int, length: int, amount: int, disorder: str, em: float, n: int = 1000
) -> DisorderArgumentCell:
    return DisorderArgumentCell(
        seed=seed,
        length=length,
        amount=amount,
        disorder=disorder,
        eval_seed=301,
        n=n,
        correct_successes=round(em * n),
        correct_em=em,
        wrong_family_em=0.0,
        none_em=0.0,
        wrong_amount_em=0.0,
    )


def _grid(em_by_disorder_amount: dict[tuple[str, int], float], amounts: list[int]) -> dict:
    return {
        (level, amount): [_cell(40, 6, amount, level, em_by_disorder_amount[(level, amount)])]
        for level in DISORDER_LEVELS
        for amount in amounts
    }


def test_classify_no_defect_when_all_cells_at_ceiling() -> None:
    amounts = [0, 1]
    grid = _grid({(level, a): 0.99 for level in DISORDER_LEVELS for a in amounts}, amounts)
    result = classify_seed_length(grid, amounts)
    assert result.disorder_main_effect == "ABSENT"
    assert result.argument_main_effect == "ABSENT"
    assert result.interaction_effect == "ABSENT"
    assert result.overall == "NO_DEFECT_DETECTED"


def test_classify_disorder_main_effect() -> None:
    # DESCENDING is uniformly bad regardless of amount; every other level is fine.
    amounts = [0, 1, 2]
    em = {}
    for level in DISORDER_LEVELS:
        for a in amounts:
            em[(level, a)] = 0.05 if level == "DESCENDING" else 0.98
    grid = _grid(em, amounts)
    result = classify_seed_length(grid, amounts)
    assert result.disorder_main_effect == "PRESENT"
    assert result.argument_main_effect == "ABSENT"
    assert result.worse_disorder == "DESCENDING"
    assert "DISORDER" in result.overall


def test_classify_argument_main_effect() -> None:
    # amount=0 is uniformly bad regardless of disorder; every other amount is fine.
    amounts = [0, 1, 2]
    em = {}
    for level in DISORDER_LEVELS:
        for a in amounts:
            em[(level, a)] = 0.05 if a == 0 else 0.98
    grid = _grid(em, amounts)
    result = classify_seed_length(grid, amounts)
    assert result.argument_main_effect == "PRESENT"
    assert result.disorder_main_effect == "ABSENT"
    assert result.worse_amount == 0
    assert "ARGUMENT" in result.overall


def test_classify_interaction_only_at_one_cell() -> None:
    amounts = [0, 1]
    em = {(level, a): 0.98 for level in DISORDER_LEVELS for a in amounts}
    em[("DESCENDING", 0)] = 0.05  # only one (disorder, amount) cell fails
    grid = _grid(em, amounts)
    result = classify_seed_length(grid, amounts)
    assert result.interaction_effect == "PRESENT"
    assert "INTERACTION" in result.overall


def test_classify_undifferentiated_deficit() -> None:
    # Every cell fails by the same amount: no main effect or interaction
    # clears the fixed margin, but the floor is not met either.
    amounts = [0, 1]
    grid = _grid({(level, a): 0.80 for level in DISORDER_LEVELS for a in amounts}, amounts)
    result = classify_seed_length(grid, amounts)
    assert result.disorder_main_effect == "ABSENT"
    assert result.argument_main_effect == "ABSENT"
    assert result.interaction_effect == "ABSENT"
    assert result.overall == "UNDIFFERENTIATED_DEFICIT"


def test_effect_margin_and_adequacy_floor_are_the_documented_constants() -> None:
    assert ADEQUACY_FLOOR == 0.95
    assert EFFECT_MARGIN == 0.10
