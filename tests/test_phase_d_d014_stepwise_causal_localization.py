"""Unit tests for Task D-014's pure-logic helpers (no GPU/model access).

These exercise the oracle re-derivation, the reset/attribution bookkeeping,
and the pre-fixed classification rule in isolation, using the same
deterministic example generator D-013 itself uses
(``apc.evaluation.phase_d_executor._examples_for_recipe``). Loading the
actual D-013 model bundles and running the full evaluate_cell pipeline
requires CUDA and the D-013 run artifacts; that is exercised by the real
one-time diagnostic run, not by this CPU-only regression suite.
"""

from __future__ import annotations

from apc.evaluation.phase_d_d014_stepwise_causal_localization import (
    ATTRIBUTION_DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT,
    ATTRIBUTION_INTERFACE_FAILURE,
    ATTRIBUTION_PASSED,
    ATTRIBUTION_RESIDUAL_SORT_DEFECT,
    ATTRIBUTION_SELECT_OWN_DEFECT,
    ATTRIBUTION_UPSTREAM_ERROR_ACCUMULATION,
    D014DiagnosticError,
    call_arguments,
    classify_attribution,
    earliest_failing_step,
    oracle_chain,
)
from apc.evaluation.phase_d_executor import TARGET_CLASSES, _examples_for_recipe


def test_oracle_chain_reconstructs_target_for_every_target_class() -> None:
    for klass in TARGET_CLASSES:
        recipe = tuple(klass.split("->"))
        examples = _examples_for_recipe(999, recipe, 25, (6, 10))
        for example in examples:
            chain = oracle_chain(example, recipe)
            assert chain[0] == tuple(example.input_tokens)
            assert chain[-1] == tuple(example.target_tokens)
            assert len(chain) == len(recipe) + 1


def test_oracle_chain_rejects_mismatched_recipe() -> None:
    recipe = tuple(TARGET_CLASSES[0].split("->"))
    example = _examples_for_recipe(1, recipe, 1, (6, 10))[0]
    try:
        oracle_chain(example, ("SORT",))
    except D014DiagnosticError:
        pass
    else:
        raise AssertionError("expected a D014DiagnosticError for a mismatched recipe")


def test_call_arguments_length_matches_recipe_and_sort_is_argument_free() -> None:
    recipe = tuple(TARGET_CLASSES[1].split("->"))  # SELECT->SORT->BIND
    example = _examples_for_recipe(2, recipe, 1, (6, 10))[0]
    args = call_arguments(example, recipe)
    assert len(args) == len(recipe)
    assert args[1] == {}  # SORT is parameter-free


def test_earliest_failing_step_none_when_all_above_floor() -> None:
    assert earliest_failing_step([0.99, 0.97, 0.96]) is None


def test_earliest_failing_step_nonfinal_floor_is_085() -> None:
    assert earliest_failing_step([0.80, 1.0, 1.0]) == 0
    assert earliest_failing_step([0.90, 1.0, 1.0]) is None


def test_earliest_failing_step_final_floor_is_095() -> None:
    assert earliest_failing_step([1.0, 1.0, 0.90]) == 2
    assert earliest_failing_step([1.0, 1.0, 0.96]) is None


def test_classify_attribution_passed() -> None:
    result = classify_attribution(
        fail_step=None,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[1.0, 1.0, 1.0],
        upstream_clean_fraction=[1.0, 1.0, 1.0],
    )
    assert result == ATTRIBUTION_PASSED


def test_classify_attribution_residual_sort_defect() -> None:
    result = classify_attribution(
        fail_step=1,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[1.0, 0.10, 1.0],
        upstream_clean_fraction=[1.0, 1.0, 1.0],
    )
    assert result == ATTRIBUTION_RESIDUAL_SORT_DEFECT


def test_classify_attribution_downstream_short_sequence_primitive_defect() -> None:
    result = classify_attribution(
        fail_step=2,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[1.0, 1.0, 0.10],
        upstream_clean_fraction=[1.0, 1.0, 1.0],
    )
    assert result == ATTRIBUTION_DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT


def test_classify_attribution_select_own_defect() -> None:
    result = classify_attribution(
        fail_step=0,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[0.10, 1.0, 1.0],
        upstream_clean_fraction=[1.0, 1.0, 1.0],
    )
    assert result == ATTRIBUTION_SELECT_OWN_DEFECT


def test_classify_attribution_upstream_error_accumulation() -> None:
    result = classify_attribution(
        fail_step=1,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[1.0, 0.99, 1.0],
        upstream_clean_fraction=[1.0, 0.10, 1.0],
    )
    assert result == ATTRIBUTION_UPSTREAM_ERROR_ACCUMULATION


def test_classify_attribution_interface_failure() -> None:
    result = classify_attribution(
        fail_step=1,
        idx_select=0,
        idx_sort=1,
        last_idx=2,
        standalone_correct_em=[1.0, 0.99, 1.0],
        upstream_clean_fraction=[1.0, 0.99, 1.0],
    )
    assert result == ATTRIBUTION_INTERFACE_FAILURE
