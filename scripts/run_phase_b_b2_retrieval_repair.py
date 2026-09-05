"""Run Phase B Task B-C005R1: Retrieval Ranking Repair Benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.retrieval_repair_benchmark import (
    RetrievalRepairConfig,
    run_retrieval_repair_benchmark,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B Task B-C005R1 Retrieval Ranking Repair")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_retrieval_repair.yaml")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_dict: dict[str, Any] = {}
    if args.config.is_file():
        config_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    if args.seeds is not None:
        config_dict["seeds"] = tuple(args.seeds)
    elif "seeds" in config_dict:
        config_dict["seeds"] = tuple(config_dict["seeds"])

    if args.bank_sizes is not None:
        config_dict["bank_sizes"] = tuple(args.bank_sizes)
    elif "bank_sizes" in config_dict:
        config_dict["bank_sizes"] = tuple(config_dict["bank_sizes"])

    if args.conditions is not None:
        config_dict["conditions"] = tuple(args.conditions)
    elif "conditions" in config_dict:
        config_dict["conditions"] = tuple(config_dict["conditions"])

    if "levels" in config_dict:
        config_dict["levels"] = tuple(
            HardNegativeLevel.from_str(lvl) for lvl in config_dict["levels"]
        )
    if "target_operations" in config_dict:
        config_dict["target_operations"] = tuple(config_dict["target_operations"])

    if args.device is not None:
        config_dict["device"] = args.device
    if args.output_dir is not None:
        config_dict["output_dir"] = args.output_dir
    elif "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    if "bank_checkpoint_dir" in config_dict:
        config_dict["bank_checkpoint_dir"] = Path(config_dict["bank_checkpoint_dir"])

    config = RetrievalRepairConfig(**config_dict)
    report = run_retrieval_repair_benchmark(config)

    print(json.dumps(report["criteria"], indent=2))
    print(f"\nElapsed time: {report['elapsed_seconds']:.2f}s")
    print(f"Overall status: {'PASS' if report['all_criteria_passed'] else 'FAIL'}")

    return 0 if report["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
