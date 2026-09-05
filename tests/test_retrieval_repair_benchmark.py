"""Unit tests for Phase B Task B-C005R1 Retrieval Ranking Repair Benchmark."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from apc.evaluation.retrieval_repair_benchmark import (
    RetrievalRepairConfig,
    run_retrieval_repair_benchmark,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def test_sealed_evaluation_seeds_strictly_rejected() -> None:
    """Configuring any sealed evaluation seed (0-4) must be rejected
    to protect partition integrity."""
    for sealed_seed in (0, 1, 2, 3, 4):
        config = RetrievalRepairConfig(seeds=(sealed_seed,))
        with pytest.raises(ValueError, match="Sealed evaluation seeds"):
            run_retrieval_repair_benchmark(config)


def test_retrieval_repair_benchmark_smoke_cpu() -> None:
    """Smoke test of the R0/R1/R2 comparison pipeline on CPU with minimal compute."""
    with TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "repair_out"
        config = RetrievalRepairConfig(
            seeds=(10,),
            bank_sizes=(16,),
            levels=(HardNegativeLevel.L0_ORTHOGONAL, HardNegativeLevel.L4_CONFUSABLE_FAMILY),
            target_operations=("SHIFT",),
            conditions=("R0", "R2"),
            support_examples=4,
            query_examples=4,
            router_train_examples=4,
            router_steps=5,
            top_k=2,
            ranking_margin=2.0,
            ranking_beta=1.0,
            arg_lambda=2.0,
            adequacy_exact_match_threshold=0.90,
            deterministic_algorithms=True,
            device="cpu",
            bank_checkpoint_dir=out_dir / "bank_ckpt",
            output_dir=out_dir,
        )

        report = run_retrieval_repair_benchmark(config)

        assert report["task_id"] == "B-C005R1"
        assert "summary_by_condition" in report
        assert "R0" in report["summary_by_condition"]
        assert "R2" in report["summary_by_condition"]
        assert report["cells_analyzed"] == 4  # 1 seed * 1 bank * 2 levels * 1 op * 2 conditions

        # Verify artifacts
        assert (out_dir / "config.yaml").is_file()
        assert (out_dir / "summary.json").is_file()
        assert (out_dir / "system.json").is_file()
        assert (out_dir / "metrics.jsonl").is_file()
        assert (out_dir / "report.md").is_file()
