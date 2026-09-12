"""Runner script for Task B-C005REC-004AQ: CD-DPCA Warm-Start Single-Recipe Causal Pilot.

Executes:
1. Protocol and source hash verification (parent Core, REC-004AK CD-DPCA artifacts).
2. AST/signature information boundary audit.
3. Trainable isolated CD-DPCA copy construction from fresh-loaded state (I01).
4. 6,000 updates of single-init training with sequence-distinctness warm-start for steps 1..500.
5. Exact return to REC-004AL baseline sampler for steps 501..6000.
6. Evaluation cadence every 500 steps across all 13 checkpoints:
   - Overall and per-length exact match and token accuracy
   - Per-position token accuracy
   - Length-10 position-4 routing metrics (top-1 key, p0, p7, margin, entropy)
   - REC-004AP A/B/C alignment metrics
7. Step 6000 decisive evaluation, causal controls, attention diagnostics.
8. Trajectory comparison against REC-004AL baseline.
9. Evaluation of terminal viability floor and position-4 attractor resolution.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_warm_start_pilot import (
    MirrorCDDPCAWarmStartPilotConfig,
    run_cd_dpca_warm_start_pilot_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AQ CD-DPCA Warm-Start Single-Recipe Causal Pilot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004aq/run_001",
        help="Run artifact output directory",
    )
    parser.add_argument(
        "--rec004al-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004al/run_001",
        help="Baseline REC-004AL directory",
    )
    parser.add_argument(
        "--rec004ak-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ak/run_001",
        help="Source REC-004AK directory",
    )
    parser.add_argument(
        "--rec004an-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004an/run_001",
        help="Source REC-004AN directory",
    )
    parser.add_argument(
        "--rec004ao-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ao/run_001",
        help="Source REC-004AO directory",
    )
    parser.add_argument(
        "--rec004ap-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ap/run_001",
        help="Source REC-004AP directory",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=6000,
        help="Maximum optimizer updates",
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

    config = MirrorCDDPCAWarmStartPilotConfig(
        output_dir=args.output_dir,
        rec004al_dir=args.rec004al_dir,
        rec004ak_dir=args.rec004ak_dir,
        rec004an_dir=args.rec004an_dir,
        rec004ao_dir=args.rec004ao_dir,
        rec004ap_dir=args.rec004ap_dir,
        max_updates=args.max_updates,
        warm_start_steps=args.warm_start_steps,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("=== Starting Task B-C005REC-004AQ: CD-DPCA Warm-Start Single-Recipe Causal Pilot ===")
    print(f"Output directory: {config.output_dir}")
    print(f"Baseline REC-004AL: {config.rec004al_dir}")
    print(f"Source REC-004AK: {config.rec004ak_dir}")
    print(f"Warm-start updates: {config.warm_start_steps} (pairwise-distinct token sampling)")
    print(
        f"Total updates: {config.max_updates} (cadence: every {config.checkpoint_interval} steps)"
    )

    t0 = time.time()
    results = run_cd_dpca_warm_start_pilot_task(config)
    dt = time.time() - t0

    summary = results["summary"]
    print(f"\n=== Completed in {dt:.2f}s ===")
    print(f"Execution Status: {summary['execution_status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Decisive Step: {summary['decisive_step']}")
    em = summary["decisive_validation_sequence_em"]
    hits = summary["decisive_validation_exact_count"]
    total = summary["decisive_validation_total_examples"]
    print(f"Validation Sequence EM: {em:.6f} ({hits}/{total})")
    print(f"Terminal Floor Met: {summary['terminal_floor_met']}")
    print(f"Position-4 Attractor Cleared: {summary['position4_attractor_cleared']}")
    print(f"Other Positions Regressed: {summary['other_positions_regressed']}")
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
