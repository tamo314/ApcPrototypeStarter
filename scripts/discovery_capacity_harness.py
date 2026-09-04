#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-002 Novel-Task and Capacity-Ladder Harness.

Verifies novel operations, verifies bank & composition failure (EM < 0.20),
audits capacity ladder parameters (T0-T3), verifies absence of oracle leakage,
and validates the unified discovery data protocol.

Usage:
    python scripts/discovery_capacity_harness.py \
        --config configs/phase_a1_discovery_capacity_harness.yaml \
        --run-dir runs/phase_a1_discovery_capacity_harness
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.discovery_capacity_harness import (
    discovery_harness_config_from_dict,
    run_discovery_capacity_harness,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_discovery_capacity_harness.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for this harness run's artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = discovery_harness_config_from_dict(raw_config)
    if args.seed is not None:
        base_config = dataclasses.replace(base_config, seed=args.seed)

    save_config(base_config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    summary = run_discovery_capacity_harness(base_config, output_dir=run_dir)

    print(json.dumps(summary.to_dict(), indent=2))

    verdict = "PASS" if summary.overall_passed else "FAIL"
    print(f"\nTask A1-B007X-002 Novel-Task & Capacity-Ladder Harness: {verdict}")
    print(f"Summary written to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
