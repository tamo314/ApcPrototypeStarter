"""Runner script for Task B-C005REC-004AM: CD-DPCA Warm-Start Multi-Initialization Reproduction.

Executes:
1. Protocol and source hash verification (parent Core, REC-004AK CD-DPCA artifacts).
2. AST/signature information boundary audit.
3. Pre-registered step-0 initial states creation for 5 independent initializations (I01..I05).
4. Sequential training of all 5 initializations under the fixed ADR-0141 warm-start recipe:
   - 500 warm-start updates with pairwise-distinct sequence sampling
   - 5,500 updates with return to standard baseline data stream
   - CosineAnnealingLR, AdamW, batch size 32, 6,000 updates per init
   - Decisive evaluation, causal controls, and freeze audit at step 6000
5. Evaluation of all-init qualification rule:
   all 5/5 inits >= 0.95 sequence EM & attractor cleared.
6. Preserving execution boundaries: candidate adoption = null, bundle write = false,
   rg3 = NOT_EXECUTED, rec005_eligible = false, G1/G4 = NOT_CLEARED.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_all_init_validation import (
    MirrorCDDPCAAllInitValidationConfig,
    run_cd_dpca_all_init_validation_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AM CD-DPCA Warm-Start Multi-Init Reproduction"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004am/run_001",
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
        "--rec004aq-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004aq/run_001",
        help="Pilot REC-004AQ directory",
    )
    parser.add_argument(
        "--max-updates-per-init",
        type=int,
        default=6000,
        help="Maximum optimizer updates per initialization",
    )
    parser.add_argument(
        "--warm-start-steps",
        type=int,
        default=500,
        help="Warm-start steps with pairwise-distinct sampling",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Checkpoint evaluation interval",
    )
    args = parser.parse_args()

    config = MirrorCDDPCAAllInitValidationConfig(
        output_dir=args.output_dir,
        rec004ak_dir=args.rec004ak_dir,
        rec004al_dir=args.rec004al_dir,
        rec004aq_dir=args.rec004aq_dir,
        max_updates_per_init=args.max_updates_per_init,
        warm_start_steps=args.warm_start_steps,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("=== Starting Task B-C005REC-004AM: CD-DPCA Multi-Init Reproduction ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Source REC-004AK: {config.rec004ak_dir}")
    print(f"Pilot REC-004AQ: {config.rec004aq_dir}")
    print(f"Inits: {config.init_ids} ({len(config.init_ids)} total)")
    print(
        f"Updates per init: {config.max_updates_per_init} "
        f"(warm-start: {config.warm_start_steps})"
    )
    print(f"Total update budget: {config.max_updates_per_init * len(config.init_ids)}")

    t0 = time.time()
    results = run_cd_dpca_all_init_validation_task(config)
    dt = time.time() - t0

    summary = results["summary"]
    print(f"\n=== Completed in {dt:.2f}s ===")
    print(f"Execution Status: {summary['execution_status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Passed Inits: {summary['n_inits_passed']} / {summary['total_inits']}")
    print(f"All Inits Passed: {summary['all_inits_passed']}")
    print(f"Mean Decisive Sequence EM: {summary['mean_decisive_sequence_em']:.6f}")
    print(f"Min Decisive Sequence EM: {summary['min_decisive_sequence_em']:.6f}")
    print(f"Max Decisive Sequence EM: {summary['max_decisive_sequence_em']:.6f}")
    print(f"Candidate Selected: {summary['candidate_selected']} (fixed null)")
    print(f"Child Bundle: {summary['child_bundle']} (fixed null)")
    print(f"RG3 Status: {summary['rg3']}")
    print(f"REC-005 Eligible: {summary['rec005_eligible']}")


if __name__ == "__main__":
    main()
