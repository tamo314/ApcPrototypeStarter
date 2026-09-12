"""Runner script for Task B-C005REC-004AL: CD-DPCA Single-Init Learning Pilot.

Executes:
1. Protocol and source hash verification (parent Core, REC-004AK CD-DPCA artifacts).
2. AST/signature information boundary audit (no target/oracle leakage into routing).
3. Trainable isolated CD-DPCA copy construction from strictly fresh-loaded state.
4. 6,000 updates of single-init (I01) training with Core/router/15 other primitives frozen.
5. Checkpointing and existing validation evaluation every 500 updates.
6. Decisive evaluation, causal controls, attention & masking diagnostics at step 6000.
7. Terminal viability criterion evaluation (sequence EM >= 0.95) and fail-closed recording.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    MirrorCDDPCALearningPilotConfig,
    run_cd_dpca_learning_pilot_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AL CD-DPCA Single-Init Learning Pilot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004al/run_001",
        help="Run artifact output directory",
    )
    parser.add_argument(
        "--rec004ak-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ak/run_001",
        help="Source REC-004AK directory",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=6000,
        help="Maximum optimizer updates",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Checkpoint evaluation interval",
    )
    args = parser.parse_args()

    config = MirrorCDDPCALearningPilotConfig(
        output_dir=args.output_dir,
        rec004ak_dir=args.rec004ak_dir,
        max_updates=args.max_updates,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("=== Starting Task B-C005REC-004AL: CD-DPCA Single-Init Learning Pilot ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Source REC-004AK: {config.rec004ak_dir}")
    print(f"Max updates: {config.max_updates} (cadence: every {config.checkpoint_interval} steps)")

    t0 = time.time()
    results = run_cd_dpca_learning_pilot_task(config)
    dt = time.time() - t0

    summary = results["summary"]
    print(f"\n=== Completed in {dt:.2f}s ===")
    print(f"Execution Status: {summary['execution_status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Decisive Step: {summary['decisive_step']}")
    em = summary['decisive_validation_sequence_em']
    hits = summary['decisive_validation_exact_count']
    total = summary['decisive_validation_total_examples']
    print(f"Validation Sequence EM: {em:.6f} ({hits}/{total})")
    print(f"Terminal Viability Met: {summary['terminal_viability_met']}")
    print(f"Causal Gap: {summary['causal_gap']:.4f}")
    print(f"Padding Mask Verified: {summary['padding_mask_verified']}")
    print(f"Top-1 Correct Key Routing Accuracy: {summary['correct_key_top1_routing_accuracy']:.4f}")
    print(f"Candidate Selected: {summary['candidate_selected']} (fixed null)")
    print(f"Child Bundle: {summary['child_bundle']} (fixed null)")
    print(f"RG3 Status: {summary['rg3']}")
    print(f"REC-005 Eligible: {summary['rec005_eligible']}")


if __name__ == "__main__":
    main()
