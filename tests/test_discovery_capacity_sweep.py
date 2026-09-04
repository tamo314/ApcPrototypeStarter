"""Unit and smoke tests for Task A1-B007X-003 Temporary Discovery Capacity Sweep."""

from __future__ import annotations

from apc.evaluation.discovery_capacity_sweep import (
    AggregatedTierResult,
    DiscoveryCapacitySweepConfig,
    SingleRunResult,
    aggregate_tier_runs,
    compute_normalized_auc,
    evaluate_operation_verdict,
    run_discovery_capacity_sweep,
)


def test_compute_normalized_auc() -> None:
    """Test normalized trapezoidal AUC calculation."""
    # Step-EM curve: (50, 0.5), (100, 1.0)
    # Area: (0,0) to (50, 0.5) -> 0.5 * 0.5 * 50 = 12.5
    # (50, 0.5) to (100, 1.0) -> 0.5 * (0.5 + 1.0) * 50 = 37.5
    # Total area = 50.0. Normalized by 100 -> 0.50
    history = [(50, 0.5), (100, 1.0)]
    auc = compute_normalized_auc(history, total_steps=100)
    assert abs(auc - 0.50) < 1e-5

    # Constant 1.0 curve from step 20 onwards
    history_fast = [(20, 1.0), (100, 1.0)]
    # (0,0) to (20, 1.0) -> 0.5 * 1.0 * 20 = 10.0
    # (20, 1.0) to (100, 1.0) -> 1.0 * 80 = 80.0
    # Total = 90.0. Normalized = 0.90
    auc_fast = compute_normalized_auc(history_fast, total_steps=100)
    assert abs(auc_fast - 0.90) < 1e-5


def test_aggregate_tier_runs() -> None:
    """Test metrics aggregation across seeds."""
    runs = [
        SingleRunResult(
            seed=0,
            operation="SWAP_PAIRS",
            tier="T0_compact",
            parameters=17098,
            final_em=0.96,
            final_loss=0.05,
            success=True,
            step_to_90=100,
            step_to_95=150,
            examples_to_90=3200,
            examples_to_95=4800,
            auc=0.75,
            wall_clock_seconds=2.0,
            peak_memory_bytes=1000,
            em_history=[(150, 0.96)],
        ),
        SingleRunResult(
            seed=1,
            operation="SWAP_PAIRS",
            tier="T0_compact",
            parameters=17098,
            final_em=0.94,
            final_loss=0.06,
            success=False,
            step_to_90=125,
            step_to_95=None,
            examples_to_90=4000,
            examples_to_95=None,
            auc=0.70,
            wall_clock_seconds=2.2,
            peak_memory_bytes=1000,
            em_history=[(150, 0.94)],
        ),
    ]

    agg = aggregate_tier_runs("SWAP_PAIRS", "T0_compact", runs)
    assert agg.num_seeds == 2
    assert agg.parameters == 17098
    assert abs(agg.mean_em - 0.95) < 1e-5
    assert agg.success_rate == 0.5
    assert abs(agg.mean_auc - 0.725) < 1e-5


def test_evaluate_operation_verdict() -> None:
    """Test decision logic for reliability and efficiency gaps."""
    # Scenario 1: Efficiency Gap passed (T2 reaches 0.95 in <= 50% steps of T0)
    compact_eff = AggregatedTierResult(
        operation="TEST_OP",
        tier="T0_compact",
        num_seeds=5,
        parameters=17098,
        mean_em=0.96,
        std_em=0.01,
        success_rate=1.0,
        median_step_to_95=200.0,
        mean_step_to_95=200.0,
        median_examples_to_95=6400.0,
        mean_auc=0.70,
        mean_wall_clock=5.0,
        peak_memory_bytes=1000,
    )
    large_eff = AggregatedTierResult(
        operation="TEST_OP",
        tier="T2_overcomplete",
        num_seeds=5,
        parameters=137482,
        mean_em=0.98,
        std_em=0.01,
        success_rate=1.0,
        median_step_to_95=100.0,  # 100 <= 50% of 200
        mean_step_to_95=100.0,
        median_examples_to_95=3200.0,
        mean_auc=0.85,
        mean_wall_clock=8.0,
        peak_memory_bytes=2000,
    )
    v1 = evaluate_operation_verdict("TEST_OP", compact_eff, large_eff)
    assert v1.advantage_found
    assert v1.efficiency_gap_passed
    assert not v1.reliability_gap_passed

    # Scenario 2: No advantage (T2 reaches in 150 steps > 50% of 200 steps, both high EM)
    large_no_adv = AggregatedTierResult(
        operation="TEST_OP",
        tier="T2_overcomplete",
        num_seeds=5,
        parameters=137482,
        mean_em=0.97,
        std_em=0.01,
        success_rate=1.0,
        median_step_to_95=150.0,
        mean_step_to_95=150.0,
        median_examples_to_95=4800.0,
        mean_auc=0.75,
        mean_wall_clock=8.0,
        peak_memory_bytes=2000,
    )
    v2 = evaluate_operation_verdict("TEST_OP", compact_eff, large_no_adv)
    assert not v2.advantage_found
    assert not v2.efficiency_gap_passed
    assert not v2.reliability_gap_passed


def test_smoke_sweep_execution(tmp_path) -> None:
    """Fast smoke test of discovery capacity sweep on CPU."""
    config = DiscoveryCapacitySweepConfig(
        seeds=(0,),
        novel_operations=("SWAP_PAIRS",),
        tiers=("T0_compact", "T2_overcomplete"),
        train_steps=6,
        eval_interval=3,
        batch_size=4,
        num_eval_examples=10,
        num_train_examples=20,
        device="cpu",
        core_train_steps=10,
        bank_train_steps=10,
    )
    summary = run_discovery_capacity_sweep(config, output_dir=tmp_path)

    assert len(summary.all_runs) == 2
    assert "SWAP_PAIRS" in summary.verdicts
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.json").exists()
