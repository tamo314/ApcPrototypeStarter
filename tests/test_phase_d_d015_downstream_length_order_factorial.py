"""Unit tests for Task D-015's pure-logic helpers (no GPU/model access).

These exercise the non-SORT identity gate, the paired example generator, the
pooling helper, and the pre-fixed length x order factorial decision rule in
isolation. Loading the actual D-013 model bundles and running the full
per-cell diagnostic requires CUDA and the D-013 run artifacts; that is
exercised by the real one-time diagnostic run, not by this CPU-only
regression suite.
"""

from __future__ import annotations

import pytest

from apc.evaluation.phase_d_d015_downstream_length_order_factorial import (
    ADEQUACY_FLOOR,
    EFFECT_MARGIN,
    D015DiagnosticError,
    FactorialCell,
    _length_group,
    _paired_examples,
    _pool_em,
    check_non_sort_identity,
    classify_reproducibility,
    classify_seed_primitive,
)


def _cell(length_group: str, order: str, em: float, n: int = 1000) -> FactorialCell:
    return FactorialCell(length_group=length_group, order=order, n_cells=1, n_total=n, em=em)


def _four_cells(
    short_u: float, short_s: float, long_u: float, long_s: float
) -> dict[str, FactorialCell]:
    return {
        "SHORT:UNSORTED": _cell("SHORT", "UNSORTED", short_u),
        "SHORT:SORTED": _cell("SHORT", "SORTED", short_s),
        "LONG:UNSORTED": _cell("LONG", "UNSORTED", long_u),
        "LONG:SORTED": _cell("LONG", "SORTED", long_s),
    }


# =============================================================================
# Non-SORT / Core identity gate
# =============================================================================


def test_check_non_sort_identity_passes_when_only_sort_differs() -> None:
    parent = {"SORT": ("h1", "a1"), "NEGATE": ("h2", "a2"), "BIND": ("h3", "a3")}
    candidate = {"SORT": ("h1-new", "a1-new"), "NEGATE": ("h2", "a2"), "BIND": ("h3", "a3")}
    core = {"canonical_state_hash": "core-hash"}
    result = check_non_sort_identity(parent, candidate, core, core)
    assert result["gate_pass"] is True
    assert result["non_sort_ops_checked"] == ["BIND", "NEGATE"]


def test_check_non_sort_identity_rejects_non_sort_drift() -> None:
    parent = {"SORT": ("h1", "a1"), "NEGATE": ("h2", "a2")}
    candidate = {"SORT": ("h1-new", "a1-new"), "NEGATE": ("h2-DRIFTED", "a2")}
    core = {"canonical_state_hash": "core-hash"}
    with pytest.raises(D015DiagnosticError):
        check_non_sort_identity(parent, candidate, core, core)


def test_check_non_sort_identity_rejects_core_drift() -> None:
    parent = {"SORT": ("h1", "a1")}
    candidate = {"SORT": ("h1-new", "a1-new")}
    with pytest.raises(D015DiagnosticError):
        check_non_sort_identity(parent, candidate, {"h": "a"}, {"h": "b"})


def test_check_non_sort_identity_rejects_mismatched_primitive_sets() -> None:
    parent = {"SORT": ("h1", "a1"), "NEGATE": ("h2", "a2")}
    candidate = {"SORT": ("h1", "a1")}
    core = {"canonical_state_hash": "core-hash"}
    with pytest.raises(D015DiagnosticError):
        check_non_sort_identity(parent, candidate, core, core)


# =============================================================================
# Paired example generation
# =============================================================================


def test_paired_examples_share_multiset_and_argument() -> None:
    unsorted, sorted_variant, args = _paired_examples("SHIFT", 6, eval_seed=301, n=25)
    assert len(unsorted) == len(sorted_variant) == len(args) == 25
    for u, s in zip(unsorted, sorted_variant, strict=True):
        assert sorted(u) == list(s)
        assert tuple(sorted(u)) == s


def test_paired_examples_deterministic_given_same_inputs() -> None:
    first = _paired_examples("SELECT", 6, eval_seed=301, n=10)
    second = _paired_examples("SELECT", 6, eval_seed=301, n=10)
    assert first == second


def test_paired_examples_rejects_bind_at_odd_length() -> None:
    with pytest.raises(D015DiagnosticError):
        _paired_examples("BIND", 3, eval_seed=301, n=5)


def test_paired_examples_accepts_bind_at_even_length() -> None:
    unsorted, sorted_variant, args = _paired_examples("BIND", 4, eval_seed=301, n=5)
    assert len(unsorted) == 5
    for params in args:
        assert "query_key" in params


def test_paired_examples_unsorted_side_is_never_already_ascending() -> None:
    # Regression: a naive random draw has non-trivial odds (~1/6 at length 3)
    # of already being in ascending order, which would silently collapse the
    # UNSORTED arm into a SORTED-equivalent example and destroy the order
    # contrast the diagnostic depends on.
    for op_name, length in (("SHIFT", 3), ("NEGATE", 3), ("REVERSE", 4), ("SELECT", 4)):
        unsorted, sorted_variant, _ = _paired_examples(op_name, length, eval_seed=301, n=200)
        for u, s in zip(unsorted, sorted_variant, strict=True):
            assert u != s, f"UNSORTED draw {u} is already in ascending order for {op_name}"


def test_paired_examples_bind_query_key_is_valid_in_both_order_arms() -> None:
    # Regression: BindOp.sample_params draws query_key from the UNSORTED
    # content's key positions (sequence[0::2]). After sorting ascending, that
    # value is not guaranteed to still land on a key position -- BindOp.apply
    # silently returns 0 for an absent key, which would make the SORTED arm
    # score against a degenerate target instead of the same well-defined
    # lookup as the UNSORTED arm.
    unsorted, sorted_variant, args = _paired_examples("BIND", 6, eval_seed=301, n=200)
    for u, s, params in zip(unsorted, sorted_variant, args, strict=True):
        query_key = params["query_key"]
        assert query_key in u[0::2]
        assert query_key in s[0::2], (
            f"query_key {query_key} is not a valid key in the SORTED arm {s}"
        )


# =============================================================================
# Length grouping and pooling
# =============================================================================


def test_length_group_short_and_long() -> None:
    assert _length_group(3) == "SHORT"
    assert _length_group(5) == "SHORT"
    assert _length_group(6) == "LONG"
    assert _length_group(10) == "LONG"


def test_length_group_rejects_out_of_range() -> None:
    with pytest.raises(D015DiagnosticError):
        _length_group(11)


def test_pool_em_weights_by_example_count() -> None:
    from apc.evaluation.phase_d_d015_downstream_length_order_factorial import OrderCellResult

    cells = [
        OrderCellResult(
            length=3,
            length_group="SHORT",
            order="UNSORTED",
            n=100,
            correct_successes=50,
            correct_em=0.5,
            wrong_family_em=0.0,
            none_em=0.0,
            wrong_argument_em=None,
        ),
        OrderCellResult(
            length=4,
            length_group="SHORT",
            order="UNSORTED",
            n=100,
            correct_successes=100,
            correct_em=1.0,
            wrong_family_em=0.0,
            none_em=0.0,
            wrong_argument_em=None,
        ),
    ]
    pooled = _pool_em(cells)
    assert pooled.n_total == 200
    assert pooled.em == pytest.approx(0.75)


# =============================================================================
# Fixed factorial decision rule
# =============================================================================


def test_classify_no_defect_when_all_cells_at_ceiling() -> None:
    cells = _four_cells(0.99, 0.98, 0.99, 0.99)
    result = classify_seed_primitive(cells)
    assert result.length_main_effect == "ABSENT"
    assert result.order_main_effect == "ABSENT"
    assert result.interaction_effect == "ABSENT"
    assert result.overall == "NO_DEFECT_DETECTED"


def test_classify_length_main_effect() -> None:
    # Short is uniformly bad regardless of order; long is uniformly good.
    cells = _four_cells(short_u=0.10, short_s=0.10, long_u=0.98, long_s=0.98)
    result = classify_seed_primitive(cells)
    assert result.length_main_effect == "PRESENT"
    assert result.order_main_effect == "ABSENT"
    assert result.interaction_effect == "ABSENT"
    assert result.overall == "EXPLAINED_BY_LENGTH"


def test_classify_order_main_effect() -> None:
    # Sorted input is uniformly bad regardless of length; unsorted is fine.
    cells = _four_cells(short_u=0.97, short_s=0.05, long_u=0.98, long_s=0.10)
    result = classify_seed_primitive(cells)
    assert result.order_main_effect == "PRESENT"
    assert result.length_main_effect == "ABSENT"
    assert result.worse_order == "SORTED"
    assert result.overall == "EXPLAINED_BY_ORDER"


def test_classify_interaction_only_at_short_sorted() -> None:
    # Only the SHORT+SORTED cell fails; every other cell is at ceiling.
    cells = _four_cells(short_u=0.97, short_s=0.05, long_u=0.98, long_s=0.97)
    result = classify_seed_primitive(cells)
    assert result.interaction_effect == "PRESENT"
    assert "INTERACTION" in result.overall


def test_classify_undifferentiated_deficit() -> None:
    # All four cells fail by the same amount: no main effect or interaction
    # clears the fixed margin, but the floor is not met either.
    cells = _four_cells(0.80, 0.80, 0.80, 0.80)
    result = classify_seed_primitive(cells)
    assert result.length_main_effect == "ABSENT"
    assert result.order_main_effect == "ABSENT"
    assert result.interaction_effect == "ABSENT"
    assert result.overall == "UNDIFFERENTIATED_DEFICIT"


def test_effect_margin_and_adequacy_floor_are_the_documented_constants() -> None:
    assert ADEQUACY_FLOOR == 0.95
    assert EFFECT_MARGIN == 0.10


# =============================================================================
# Cross-seed reproducibility rule
# =============================================================================


def test_reproducible_present() -> None:
    assert classify_reproducibility(["PRESENT"] * 5) == "REPRODUCIBLE_PRESENT"


def test_reproducible_absent() -> None:
    assert classify_reproducibility(["ABSENT"] * 5) == "REPRODUCIBLE_ABSENT"


def test_not_reproducible_reports_split() -> None:
    result = classify_reproducibility(["PRESENT", "PRESENT", "ABSENT", "PRESENT", "ABSENT"])
    assert result.startswith("NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS")
    assert "3/5" in result
