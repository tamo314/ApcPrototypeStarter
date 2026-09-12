"""Runner script for Task B-C005REC-004AR:
CD-DPCA Warm-Start Five-Seed Matched-Baseline Causal Replication.

Executes:
1. Protocol and source hash verification (parent Core, REC-004AK, REC-004AL, REC-004AM).
2. AST/signature information boundary audit.
3. Re-evaluation of qualified baseline control (REC-004AL I01) across 13 checkpoints.
4. Execution of fresh baseline control runs for unacquired initializations I02..I05:
   - Same step-0 weights / hash as corresponding warm-start run
   - Standard with-replacement sampler for all 6000 steps
   - AdamW, CosineAnnealingLR, batch size 32, 6,000 updates, cadence 500
5. Re-use of REC-004AM warm-start runs I01..I05 as fixed intervention arm.
6. Matched-pairs causal comparison across all 5 seeds on terminal EM, worst-position acc,
   position-4 routing / margins, and 13-checkpoint attractor trajectories.
7. Pre-registered summary statistics (paired effect, median, range, improved ratio)
   and primary decision evaluation.
8. Preserving execution boundaries: candidate adoption = null, bundle write = false,
   rg3 = NOT_EXECUTED, rec005_eligible = false, G1/G4 = NOT_CLEARED.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_matched_baseline_replication import (
    MirrorCDDPCAMatchedBaselineReplicationConfig,
    run_cd_dpca_matched_baseline_replication_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AR CD-DPCA Warm-Start Matched-Baseline Replication"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ar/run_001",
        help="Run artifact output directory",
    )
    parser.add_argument(
        "--rec004ak-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ak/run_001",
        help="Source REC-004AK directory",
    )
    parser.add_argument(
        "--rec004al-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004al/run_001",
        help="Baseline REC-004AL directory",
    )
    parser.add_argument(
        "--rec004am-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004am/run_001",
        help="Warm-start REC-004AM directory",
    )
    parser.add_argument(
        "--max-updates-per-init",
        type=int,
        default=6000,
        help="Maximum optimizer updates per initialization",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Checkpoint evaluation interval",
    )
    args = parser.parse_args()

    config = MirrorCDDPCAMatchedBaselineReplicationConfig(
        output_dir=args.output_dir,
        rec004ak_dir=args.rec004ak_dir,
        rec004al_dir=args.rec004al_dir,
        rec004am_dir=args.rec004am_dir,
        max_updates_per_init=args.max_updates_per_init,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("=== Starting Task B-C005REC-004AR: CD-DPCA Five-Seed Matched-Baseline Replication ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Source REC-004AK: {config.rec004ak_dir}")
    print(f"Baseline REC-004AL: {config.rec004al_dir}")
    print(f"Warm-start REC-004AM: {config.rec004am_dir}")
    print(f"All matched inits: {config.init_ids}")
    print(f"Fresh baseline executions: {config.baseline_execute_inits}")
    print(
        f"Updates per init: {config.max_updates_per_init} "
        f"(cadence: every {config.checkpoint_interval} steps)"
    )

    t0 = time.time()
    results = run_cd_dpca_matched_baseline_replication_task(config)
    dt = time.time() - t0

    summary = results["summary"]
    primary = summary["primary_decision"]
    stats = summary["summary_statistics"]

    print(f"\n=== Completed in {dt:.2f}s ===")
    print(f"Execution Status: {summary['execution_status']}")
    print(f"Primary Decision: {primary['decision']}")
    print(
        f"Co-improved seeds: {primary['co_improved_count']} / {primary['total_seeds']} "
        f"({primary['co_improved_seeds']})"
    )
    em_stat = stats["overall_sequence_em"]
    print(
        f"Mean Overall EM Paired Effect: {em_stat['mean']:+.6f} "
        f"(Median: {em_stat['median']:+.6f}, Improved: {em_stat['improved_count']}/5)"
    )
    wp_stat = stats["worst_position_accuracy"]
    print(
        f"Mean Worst-Pos Acc Paired Effect: {wp_stat['mean']:+.6f} "
        f"(Median: {wp_stat['median']:+.6f}, Improved: {wp_stat['improved_count']}/5)"
    )
    l10_stat = stats["length10_sequence_em"]
    print(
        f"Mean Length-10 EM Paired Effect: {l10_stat['mean']:+.6f} "
        f"(Median: {l10_stat['median']:+.6f}, Improved: {l10_stat['improved_count']}/5)"
    )
    print(f"Candidate Selected: {summary['candidate_selected']} (fixed null)")
    print(f"Child Bundle: {summary['child_bundle']} (fixed null)")
    print(f"RG3 Status: {summary['rg3']}")
    print(f"REC-005 Eligible: {summary['rec005_status']}")


if __name__ == "__main__":
    main()
