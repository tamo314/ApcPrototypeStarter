#!/usr/bin/env python
"""CLI entry point for the Phase A novel-operation benchmark (Task 009).

Loads a checkpoint written by `scripts/smoke_train.py` (or any run
directory in the same `config.yaml` + `checkpoint/model.pt` layout),
evaluates it on held-out novel-composition (C) and genuinely novel-operation
(N) examples, and writes `novel_operation_benchmark.json` into the run
directory. The checkpoint's training data never included the novel
operation(s), so this measures whether a static composition baseline fails
on N materially more than it fails on C (Task 009 acceptance).

Usage:
    python scripts/smoke_train.py --config configs/phase_a_smoke.yaml
    python scripts/novel_operation_benchmark.py --run-dir runs/phase_a_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.core.checkpoint import load_checkpoint
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.core.train import resolve_device
from apc.environments.operations import NOVEL_OPERATION_NAMES
from apc.evaluation.novel_operation_benchmark import (
    NovelOperationBenchmarkConfig,
    run_novel_operation_benchmark,
)
from apc.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing config.yaml and checkpoint/model.pt",
    )
    parser.add_argument(
        "--novel-operation",
        action="append",
        dest="novel_operations",
        default=None,
        help=(
            "Novel operation name to evaluate (repeatable). "
            f"Defaults to all of {list(NOVEL_OPERATION_NAMES)}."
        ),
    )
    parser.add_argument(
        "--num-novel-composition", type=int, default=64, help="C examples to evaluate"
    )
    parser.add_argument(
        "--num-novel-operation", type=int, default=64, help="N examples to evaluate"
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

    novel_operation_names = tuple(args.novel_operations or NOVEL_OPERATION_NAMES)
    bench_config = NovelOperationBenchmarkConfig(
        seed=run_config["seed"],
        novel_operation_names=novel_operation_names,
        vocab_size=data_cfg["vocab_size"],
        sequence_length_range=tuple(data_cfg["sequence_length_range"]),
        max_depth=data_cfg["max_depth"],
        novel_composition_fraction=data_cfg["novel_composition_fraction"],
        num_novel_composition=args.num_novel_composition,
        num_novel_operation=args.num_novel_operation,
    )
    report = run_novel_operation_benchmark(model, bench_config, specials, device)

    out_path = run_dir / "novel_operation_benchmark.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
