#!/usr/bin/env python
"""CLI entry point for the Phase A composition benchmark (Task 006).

Loads a checkpoint written by `scripts/smoke_train.py` (or any run
directory in the same `config.yaml` + `checkpoint/model.pt` layout),
evaluates it on known-operation and held-out novel-composition examples,
and writes `composition_benchmark.json` into the run directory.

Usage:
    python scripts/smoke_train.py --config configs/phase_a_smoke.yaml
    python scripts/composition_benchmark.py --run-dir runs/phase_a_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.core.checkpoint import load_checkpoint
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.core.train import resolve_device
from apc.environments.generator import KNOWN_SPLITS
from apc.evaluation.composition_benchmark import (
    CompositionBenchmarkConfig,
    run_composition_benchmark,
)
from apc.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing config.yaml and checkpoint/model.pt",
    )
    parser.add_argument("--num-known", type=int, default=64, help="K examples to evaluate")
    parser.add_argument("--num-novel", type=int, default=64, help="C examples to evaluate")
    parser.add_argument(
        "--known-split",
        default="test",
        choices=list(KNOWN_SPLITS),
        help="Known pool to draw K from",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_config = load_config(run_dir / "config.yaml")
    data_cfg = run_config["data"]
    model_cfg = run_config["model"]
    device = resolve_device(args.device)

    specials = build_special_tokens(data_cfg["vocab_size"])
    model = DecoderOnlyTransformer(TransformerConfig(**model_cfg)).to(device)
    load_checkpoint(run_dir / "checkpoint" / "model.pt", model, map_location=str(device))

    bench_config = CompositionBenchmarkConfig(
        seed=run_config["seed"],
        vocab_size=data_cfg["vocab_size"],
        sequence_length_range=tuple(data_cfg["sequence_length_range"]),
        max_depth=data_cfg["max_depth"],
        novel_composition_fraction=data_cfg["novel_composition_fraction"],
        known_split=args.known_split,
        num_known=args.num_known,
        num_novel=args.num_novel,
    )
    report = run_composition_benchmark(model, bench_config, specials, device)

    out_path = run_dir / "composition_benchmark.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
