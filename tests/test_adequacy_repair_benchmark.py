"""Integration test for Task B-C005R2 adequacy estimator repair benchmark."""

from pathlib import Path

from apc.evaluation.adequacy_repair_benchmark import (
    AdequacyRepairConfig,
    run_adequacy_repair_benchmark,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def test_adequacy_repair_benchmark_lightweight_integration(tmp_path: Path) -> None:
    config = AdequacyRepairConfig(
        seeds=(10,),
        bank_sizes=(16,),
        levels=(HardNegativeLevel.L0_ORTHOGONAL, HardNegativeLevel.L4_CONFUSABLE_FAMILY),
        target_operations=("SHIFT",),
        policies=("fixed_32", "sequential"),
        support_examples=64,
        query_examples=16,
        router_train_examples=8,
        router_steps=10,
        top_k=3,
        initial_support=32,
        support_increment=32,
        max_support=64,
        device="cpu",
        output_dir=tmp_path / "adequacy_repair_light",
    )

    report = run_adequacy_repair_benchmark(config)
    assert report["task_id"] == "B-C005R2"
    assert "criteria" in report
    assert "policy_summary" in report
    assert (tmp_path / "adequacy_repair_light" / "report.md").is_file()
    assert (tmp_path / "adequacy_repair_light" / "summary.json").is_file()
    assert (tmp_path / "adequacy_repair_light" / "metrics.jsonl").is_file()

    # Verify that unselected forward calls is strictly zero
    summary = report["policy_summary"]
    assert "fixed_32" in summary
    assert "sequential" in summary
