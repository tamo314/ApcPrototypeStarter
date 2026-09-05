"""Pure, lightweight tests for Task A2-C009 reporting and gate logic."""

from __future__ import annotations

import dataclasses

import pytest

from apc.evaluation.repeated_bank_growth_benchmark import (
    GROWTH_STAGES,
    SeedGrowthResult,
    StageGrowthSnapshot,
    aggregate_growth_results,
    evaluate_growth_acceptance,
)

_INSERTED_OPERATIONS = (
    "ROTATE_TRIPLETS",
    "SWAP_ENDS",
    "MIRROR_HALVES",
    "ALTERNATING_NEGATE",
    "CYCLE_FOUR",
    "INCREMENT_MOD",
)


def _snapshot(
    insertion_index: int,
    *,
    seed: int = 0,
    overall_routing: float = 0.97,
    old_routing_mean_drop: float = 0.01,
    worst_old_routing_drop: float = 0.03,
    canonical_performance_drop: float = 0.01,
    consolidated_performance_drop: float = 0.01,
    recurrence_reuse: float = 0.92,
) -> StageGrowthSnapshot:
    bank_size_before = 10 + insertion_index - 1
    return StageGrowthSnapshot(
        seed=seed,
        insertion_index=insertion_index,
        bank_size_before=bank_size_before,
        bank_size=bank_size_before + 1,
        new_operation=_INSERTED_OPERATIONS[insertion_index - 1],
        controller_action="PLASTIC_SEARCH",
        promotion_count=1,
        router_update_condition="R2",
        overall_routing=overall_routing,
        old_routing_top1=1.0 - old_routing_mean_drop,
        new_routing_top1=0.96,
        old_routing_mean_drop=old_routing_mean_drop,
        worst_old_routing_drop=worst_old_routing_drop,
        canonical_performance=1.0 - canonical_performance_drop,
        canonical_performance_drop=canonical_performance_drop,
        consolidated_performance=1.0 - consolidated_performance_drop,
        consolidated_performance_drop=consolidated_performance_drop,
        recurrence_router_top1=0.96,
        recurrence_reuse=recurrence_reuse,
        recurrence_em=0.96,
        unselected_forward_calls=0,
        workspace_param_count=0,
        temporary_peak_params=17_098,
        stage_passed=True,
    )


def _passing_snapshots(*, seed: int = 0) -> list[StageGrowthSnapshot]:
    return [_snapshot(index, seed=seed) for index in range(1, 7)]


def _seed_result(seed: int, snapshots: list[StageGrowthSnapshot]) -> SeedGrowthResult:
    acceptance = evaluate_growth_acceptance(snapshots)
    return SeedGrowthResult(
        seed=seed,
        initial_bank_size=10,
        final_bank_size=snapshots[-1].bank_size,
        snapshots=snapshots,
        acceptance=acceptance,
        all_passed=all(acceptance.values()),
    )


def test_six_insertions_cover_each_bank_size_and_report_milestones() -> None:
    snapshots = _passing_snapshots()

    assert GROWTH_STAGES == (10, 12, 14, 16)
    assert [snapshot.insertion_index for snapshot in snapshots] == list(range(1, 7))
    assert [snapshot.bank_size_before for snapshot in snapshots] == list(range(10, 16))
    assert [snapshot.bank_size for snapshot in snapshots] == list(range(11, 17))
    assert [snapshots[index - 1].bank_size for index in (2, 4, 6)] == list(
        GROWTH_STAGES[1:]
    )
    assert [snapshot.new_operation for snapshot in snapshots] == list(_INSERTED_OPERATIONS)
    assert evaluate_growth_acceptance(snapshots)["six_successful_promotions"]


def test_acceptance_rejects_incomplete_or_unsuccessful_growth() -> None:
    incomplete = _passing_snapshots()[:-1]
    unsuccessful = _passing_snapshots()
    unsuccessful[3] = dataclasses.replace(unsuccessful[3], promotion_count=0)

    assert not evaluate_growth_acceptance(incomplete)["final_semantic_bank_16"]
    assert not evaluate_growth_acceptance(incomplete)["six_successful_promotions"]
    assert not evaluate_growth_acceptance(unsuccessful)["six_successful_promotions"]


def test_acceptance_checks_drop_at_every_insertion_not_only_final() -> None:
    snapshots = _passing_snapshots()
    snapshots[1] = dataclasses.replace(snapshots[1], canonical_performance_drop=0.0201)
    assert snapshots[-1].canonical_performance_drop < 0.02

    acceptance = evaluate_growth_acceptance(snapshots)

    assert acceptance["overall_routing"]
    assert acceptance["old_routing_mean_drop"]
    assert not acceptance["canonical_performance_drop"]
    assert acceptance["consolidated_task_drop"]
    assert acceptance["recurrence_reuse"]


def test_acceptance_thresholds_are_inclusive() -> None:
    snapshots = [
        dataclasses.replace(
            _snapshot(index),
            overall_routing=0.95,
            new_routing_top1=0.95,
            old_routing_mean_drop=0.02,
            worst_old_routing_drop=0.05,
            canonical_performance_drop=0.02,
            consolidated_performance_drop=0.02,
            recurrence_reuse=0.90,
        )
        for index in range(1, 7)
    ]

    acceptance = evaluate_growth_acceptance(snapshots)

    assert all(acceptance.values())


def test_multi_seed_aggregation_preserves_means_worst_drops_and_gate_conjunction() -> None:
    final_routing = (0.95, 0.96, 0.97, 0.98, 0.99)
    recurrence_reuse = (0.90, 0.91, 0.92, 0.93, 0.94)
    results: list[SeedGrowthResult] = []
    for seed in range(5):
        snapshots = _passing_snapshots(seed=seed)
        snapshots[-1] = dataclasses.replace(
            snapshots[-1],
            overall_routing=final_routing[seed],
            recurrence_reuse=recurrence_reuse[seed],
        )
        results.append(_seed_result(seed, snapshots))

    aggregate = aggregate_growth_results(results)

    assert aggregate["num_seeds"] == 5
    assert aggregate["total_insertions"] == 30
    assert aggregate["mean_final_overall_routing"] == pytest.approx(0.97)
    assert aggregate["max_old_routing_mean_drop"] == pytest.approx(0.01)
    assert aggregate["max_canonical_performance_drop"] == pytest.approx(0.01)
    assert aggregate["max_consolidated_task_drop"] == pytest.approx(0.01)
    assert aggregate["mean_recurrence_reuse"] == pytest.approx(0.92)
    assert aggregate["overall_passed"] is True


def test_multi_seed_aggregation_fails_when_any_seed_or_insertion_fails() -> None:
    results = [_seed_result(seed, _passing_snapshots(seed=seed)) for seed in range(5)]
    failed_snapshots = list(results[-1].snapshots)
    failed_snapshots[2] = dataclasses.replace(
        failed_snapshots[2], old_routing_mean_drop=0.0201
    )
    results[-1] = _seed_result(4, failed_snapshots)

    aggregate = aggregate_growth_results(results)

    assert not aggregate["acceptance"]["old_routing_mean_drop"]
    assert aggregate["overall_passed"] is False

