"""Unit and regression tests for Task NRQ-007 Stepwise Causal Attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.evaluation.nrq006_argument_closed_depth3_audit import (
    _load_reconstructed_bundle,
    generate_benchmark_split_deterministic,
)
from apc.evaluation.nrq007_stepwise_causal_attribution import (
    NRQ007AttributionReport,
    evaluate_cell_stepwise_causal_attribution,
    get_wrong_argument,
    precompute_standalone_controls,
    reaggregate_nrq006_cells,
    run_nrq007_causal_attribution,
)

SUMMARY_PATH = Path("runs/nrq006_depth3_audit/summary_process_1.json")
BUNDLE_BASE = Path("runs/nrq004_reconstructed_bundles")


def test_reaggregate_nrq006_cells() -> None:
    """Verify 1200-cell re-aggregation and reconciliation."""
    if not SUMMARY_PATH.is_file():
        pytest.skip("NRQ-006 summary artifact not present")

    summary, class_recs, cell_recs = reaggregate_nrq006_cells(SUMMARY_PATH)

    assert summary.total_classes == 60
    assert summary.total_cells == 1200
    assert len(class_recs) == 60
    assert len(cell_recs) == 1200

    # Class partition
    assert summary.length_adequate_classes_count == 29
    assert summary.length_contracted_classes_count == 31
    assert summary.length_adequate_classes_count + summary.length_contracted_classes_count == 60

    # Mean threshold failures
    assert summary.mean_failure_classes_count == 35
    assert summary.mean_passing_classes_count == 25
    assert summary.length_contracted_failures_count == 27
    assert summary.length_adequate_failures_count == 8

    # All-cell gate
    assert summary.all_cell_passing_classes_count == 19
    assert summary.all_cell_failing_classes_count == 41
    assert summary.total_cells_passing_both == 558


def test_get_wrong_argument() -> None:
    """Verify deterministic generation of wrong arguments for parameterized ops."""
    # SHIFT
    wa_shift = get_wrong_argument("SHIFT", 2, in_len=8)
    assert wa_shift != 2
    assert 0 <= wa_shift < 8

    # COUNT
    wa_count = get_wrong_argument("COUNT", 3, in_len=8, vocab_size=10)
    assert wa_count != 3
    assert 0 <= wa_count < 10

    # BIND
    wa_bind = get_wrong_argument("BIND", 5, in_len=8, vocab_size=10)
    assert wa_bind != 5
    assert 0 <= wa_bind < 10

    # SELECT
    orig_indices = (0, 1, 2)
    wa_sel = get_wrong_argument("SELECT", orig_indices, in_len=6)
    assert wa_sel != orig_indices
    assert len(wa_sel) == len(orig_indices)
    assert all(0 <= i < 6 for i in wa_sel)


def test_fast_stepwise_causal_attribution_single_cell() -> None:
    """Verify stepwise evaluation on a single cell with intact bundle."""
    if not BUNDLE_BASE.is_dir():
        pytest.skip("Reconstructed bundles not present")

    core, bank, op_to_id = _load_reconstructed_bundle(1, bundle_base=BUNDLE_BASE)
    recipe = ("SELECT", "SORT", "REVERSE")
    _, test_exs = generate_benchmark_split_deterministic(
        recipe, n_support=8, n_eval=10, data_seed=101
    )

    standalone_controls = precompute_standalone_controls(
        bundle_seeds=(1,), bundle_base=BUNDLE_BASE, n_examples=10
    )

    res = evaluate_cell_stepwise_causal_attribution(
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        recipe=recipe,
        bundle_seed=1,
        data_seed=101,
        test_examples=test_exs,
        standalone_controls=standalone_controls,
    )

    assert res.canonical_class == "SELECT->SORT->REVERSE"
    assert len(res.steps) == 3
    assert res.earliest_failing_step >= 0

    valid_categories = {
        "SHORT_SEQUENCE_CAPACITY_DEFICIT",
        "HIDDEN_STATE_INTERFACE",
        "UPSTREAM_ERROR_ACCUMULATION",
        "ARGUMENT_HANDLING",
        "BUNDLE_SPECIFIC_COMPONENT_FAILURE",
        "PASSED_CELL",
    }
    assert res.cell_attribution in valid_categories

    for step in res.steps:
        assert 0.0 <= step.continuous_em <= 1.0
        assert 0.0 <= step.continuous_token_acc <= 1.0
        assert 0.0 <= step.diagnostic_reset_em <= 1.0
        assert 0.0 <= step.diagnostic_reset_token_acc <= 1.0
        assert 0.0 <= step.standalone_control_em <= 1.0


def test_fast_run_nrq007_audit(tmp_path: Path) -> None:
    """Execute fast audit on a subset of failure classes and verify artifacts."""
    if not SUMMARY_PATH.is_file() or not BUNDLE_BASE.is_dir():
        pytest.skip("Prerequisites not present")

    report, reagg = run_nrq007_causal_attribution(
        summary_path=SUMMARY_PATH,
        bundle_base=BUNDLE_BASE,
        bundle_seeds=(1,),
        data_seeds=(101,),
        support_n=8,
        eval_n=10,
        output_dir=tmp_path,
    )

    assert isinstance(report, NRQ007AttributionReport)
    assert report.task_id == "NRQ-007"
    assert report.all_cell_gate_status == "QUALIFIED_UNDER_ALL_CELL_GATE"
    assert (tmp_path / "reaggregation_summary.json").is_file()
    assert (tmp_path / "nrq007_attribution_report.json").is_file()
