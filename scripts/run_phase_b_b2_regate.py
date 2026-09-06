"""Run Phase B Task B-C005G: New sealed hard-negative B2 re-gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.hard_negative_repair_gate import (
    HardNegativeRegateConfig,
    run_hard_negative_regate,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B B-C005G sealed hard-negative re-gate")
    parser.add_argument("--config", type=Path, default=Path("configs/phase_b_b2_regate.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    raw: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if "levels" in raw:
        raw["levels"] = tuple(HardNegativeLevel.from_str(value) for value in raw["levels"])
    for field in ("sealed_seeds", "development_seeds", "bank_sizes", "target_operations"):
        if field in raw:
            raw[field] = tuple(raw[field])
    if "bank_checkpoint_dir" in raw:
        raw["bank_checkpoint_dir"] = Path(raw["bank_checkpoint_dir"])
    if args.device is not None:
        raw["device"] = args.device
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
    elif "output_dir" in raw:
        raw["output_dir"] = Path(raw["output_dir"])
    report = run_hard_negative_regate(HardNegativeRegateConfig(**raw))
    print(json.dumps(report["criteria"], indent=2))
    print(f"\nProtocol hash: {report['protocol_hash']}")
    print(f"Overall status: {'PASS' if report['all_criteria_passed'] else 'FAIL'}")
    return 0 if report["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
