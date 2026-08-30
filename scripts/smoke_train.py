#!/usr/bin/env python
"""CLI entry point for the Phase A fixed-dense-baseline smoke run (Task 003).

Usage:
    python scripts/smoke_train.py --config configs/phase_a_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.core.train import run_smoke_training, smoke_config_from_dict
from apc.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a smoke-train YAML config")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output run directory (default: <output_dir>/<run_name> from the config)",
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    config = smoke_config_from_dict(raw_config)
    run_dir = Path(args.run_dir) if args.run_dir else Path(config.output_dir) / config.run_name

    summary = run_smoke_training(config, run_dir)
    print(json.dumps(summary, indent=2))
    print(f"Run artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
