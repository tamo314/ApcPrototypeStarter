"""Unit tests for Phase B B-C005D Failure Isolation Diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

from apc.evaluation.hard_negative_failure_isolation import (
    HardNegativeFailureIsolationConfig,
    L4FailureCategory,
    compute_binomial_reference,
    run_hard_negative_failure_isolation,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def test_binomial_reference_calculation() -> None:
    """Verify mathematical properties of binomial reference curve calculation."""
    curves = compute_binomial_reference(
        support_sizes=(16, 32, 64, 128),
        p_values=(0.95, 0.98),
        threshold=0.95,
    )
    assert "p_0.95" in curves
    assert "p_0.98" in curves

    k32_p98 = curves["p_0.98"]["32"]
    assert k32_p98["required_correct"] == 31  # ceil(0.95 * 32) = 31
    assert 0.0 < k32_p98["prob_accept"] < 1.0
    assert 0.0 < k32_p98["prob_false_reject"] < 1.0
    assert math.isclose(k32_p98["prob_accept"] + k32_p98["prob_false_reject"], 1.0)

    # For higher p, accept probability must be higher
    k32_p95 = curves["p_0.95"]["32"]
    assert k32_p98["prob_accept"] > k32_p95["prob_accept"]


def test_l4_failure_taxonomy_enum() -> None:
    """Verify L4 taxonomy values match required specification."""
    expected = {"NEITHER", "FAMILY_RANKING_FAILURE", "ARGUMENT_RESOLUTION_FAILURE", "BOTH"}
    assert {m.value for m in L4FailureCategory} == expected


def test_hard_negative_failure_isolation_smoke() -> None:
    """Run lightweight smoke test of the failure isolation pipeline."""
    with TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "diag_out"
        config = HardNegativeFailureIsolationConfig(
            seeds=(0,),
            bank_sizes=(16,),
            levels=(HardNegativeLevel.L0_ORTHOGONAL, HardNegativeLevel.L4_CONFUSABLE_FAMILY),
            target_operations=("SHIFT",),
            support_examples=8,
            query_examples=8,
            router_train_examples=8,
            router_steps=10,
            top_k=2,
            adequacy_exact_match_threshold=0.90,
            support_variance_sizes=(8, 16),
            binomial_p_values=(0.95, 0.98),
            device="cpu",
            bank_checkpoint_dir=out_dir / "bank_ckpt",
            output_dir=out_dir,
        )

        summary = run_hard_negative_failure_isolation(config)
        assert summary["task_id"] == "B-C005D"
        assert summary["cells_analyzed"] == 2  # 1 seed * 1 bank_size * 2 levels * 1 op
        assert summary["queries_analyzed"] == 16  # 2 cells * 8 queries

        # Verify output files
        assert (out_dir / "summary.json").is_file()
        assert (out_dir / "system.json").is_file()
        assert (out_dir / "metrics.jsonl").is_file()
        assert (out_dir / "failure_breakdown.json").is_file()
        assert (out_dir / "margin_summary.json").is_file()
        assert (out_dir / "support_variance.json").is_file()

        # Verify plot files
        plots_dir = out_dir / "plots"
        assert (plots_dir / "top1_vs_level.png").is_file()
        assert (plots_dir / "topk_vs_level.png").is_file()
        assert (plots_dir / "margin_distribution_by_level.png").is_file()
        assert (plots_dir / "rank_histogram_by_level.png").is_file()
        assert (plots_dir / "l4_family_vs_argument_failures.png").is_file()
        assert (plots_dir / "false_plastic_vs_support_size.png").is_file()
        assert (plots_dir / "support_count_histogram_false_plastic.png").is_file()

        # Core answers
        answers = summary["core_question_answers"]
        assert len(answers) == 7
