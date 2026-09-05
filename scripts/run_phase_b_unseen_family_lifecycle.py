"""Runner script for Task B-C003 (STOP GATE B1: unseen-family lifecycle benchmark).

Executes Gate B1 evaluation across 5 seeds (0, 1, 2, 3, 4) and saves full artifacts to
runs/phase_b_unseen_family_lifecycle_gate/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from apc.evaluation.unseen_family_lifecycle_benchmark import (
    DEFAULT_GATE_SEEDS,
    DEFAULT_SEALED_OPERATIONS,
    UnseenFamilyLifecycleConfig,
    run_unseen_family_lifecycle_benchmark,
)


def main() -> None:
    output_dir = Path("runs/phase_b_unseen_family_lifecycle_gate")
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Task B-C003 STOP GATE B1 evaluation on {device_str.upper()}...")
    print(f"Seeds: {DEFAULT_GATE_SEEDS}")
    print(f"Sealed novel operations: {DEFAULT_SEALED_OPERATIONS}")

    config = UnseenFamilyLifecycleConfig(
        seeds=DEFAULT_GATE_SEEDS,
        sealed_operations=DEFAULT_SEALED_OPERATIONS,
        support_size=16,
        eval_size=64,
        plastic_train_size=1000,
        compact_budget_steps=1200,
        compact_batch_size=64,
        compact_lr=2e-3,
        fallback_budget_steps=600,
        fallback_lr=2e-3,
        distillation_steps=400,
        shadow_retention_threshold=0.90,
        device_str=device_str,
        output_dir=output_dir,
    )

    t0 = time.perf_counter()
    aggregate, seed_results = run_unseen_family_lifecycle_benchmark(config)
    elapsed = time.perf_counter() - t0

    print(f"\nBenchmark completed in {elapsed:.2f} seconds.")
    print("=" * 70)
    print("STOP GATE B1 SCIENTIFIC VERIFICATION SUMMARY")
    print("=" * 70)
    p_trig = "PASS" if aggregate.passed_plastic_trigger else "FAIL"
    p_mean = "PASS" if aggregate.passed_mean_novel_em else "FAIL"
    p_min = "PASS" if aggregate.passed_min_seed_novel_em else "FAIL"
    p_promo = "PASS" if aggregate.passed_one_to_one_promotion else "FAIL"
    p_leaks = "PASS" if aggregate.passed_workspace_leaks else "FAIL"
    p_rec = "PASS" if aggregate.passed_recurrence else "FAIL"
    trig_s = f"{aggregate.plastic_trigger_rate * 100:.2f}% (>= 95.0%) -> {p_trig}"
    mean_s = f"{aggregate.mean_final_novel_em * 100:.2f}% (>= 95.0%) -> {p_mean}"
    min_s = f"{aggregate.min_seed_novel_em * 100:.2f}% (>= 90.0%) -> {p_min}"
    promo_s = f"{aggregate.total_promotions}/{aggregate.expected_promotions} -> {p_promo}"
    leak_s = f"{aggregate.workspace_leak_count} (== 0) -> {p_leaks}"
    rec_s = f"{aggregate.mean_recurrence_em * 100:.2f}% (>= 95.0%) -> {p_rec}"
    p_em_drop = "PASS" if aggregate.max_old_task_em_drop <= 0.01 else "FAIL"
    p_rt_drop = "PASS" if aggregate.max_old_routing_drop <= 0.01 else "FAIL"
    p_fp = "PASS" if aggregate.legacy_false_plastic_rate <= 0.01 else "FAIL"

    print(f"Plastic Trigger Rate:        {trig_s}")
    print(f"Mean Final Novel EM:         {mean_s}")
    print(f"Min Seed Novel EM:           {min_s}")
    print(f"1:1 Bank Promotion:          {promo_s}")
    print(f"Workspace Leaks:             {leak_s}")
    print(f"Fresh Recurrence EM:         {rec_s}")
    print(f"Recurrence Adapt Steps:      {aggregate.max_recurrence_adaptation_steps} (== 0)")
    print(f"Recurrence Temp Params:      {aggregate.max_recurrence_temporary_params} (== 0)")
    print(f"Recurrence Bank Growth:      {aggregate.total_recurrence_bank_growth} (== 0)")
    em_drop_s = f"{aggregate.max_old_task_em_drop * 100:.2f} pp (<= 1.0 pp)"
    rt_drop_s = f"{aggregate.max_old_routing_drop * 100:.2f} pp (<= 1.0 pp)"
    fp_s = f"{aggregate.legacy_false_plastic_rate * 100:.2f}% (<= 1.0%)"
    print(f"Max Old-Task EM Drop:        {em_drop_s} -> {p_em_drop}")
    print(f"Max Old-Routing Top-1 Drop:  {rt_drop_s} -> {p_rt_drop}")
    print(f"Legacy False Plastic Rate:   {fp_s} -> {p_fp}")
    print("=" * 70)

    if aggregate.gate_b1_passed:
        print("\n>>> STOP GATE B1: PASSED <<<")
        print(f"Artifacts saved to {output_dir.resolve()}")
    else:
        print("\n>>> STOP GATE B1: FAILED <<<")
        print("Saving artifacts for failure analysis as required by STOP GATE discipline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
