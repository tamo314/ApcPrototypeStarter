"""Explicit one-task dispatcher for the Phase B B2 post-D2 repair series.

Per `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md` Section 0: exactly one
named task runs per invocation, and there is no `--all` or implicit
next-task execution. Only `B-C005R3-001` through `B-C005R3-003` are
implemented so far; every other task ID is rejected until it is explicitly
implemented and wired in here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.functional_metrics_v2 import (
    FunctionalMetricsV2Config,
    run_functional_metrics_v2_protocol,
)
from apc.evaluation.post_d2_repair_benchmark import (
    PostD2ReproducibilityConfig,
    run_post_d2_reproducibility_task,
)
from apc.evaluation.relation_split_protocol import (
    RelationSplitProtocolConfig,
    run_relation_split_protocol,
)

_IMPLEMENTED_TASKS = ("B-C005R3-001", "B-C005R3-002", "B-C005R3-003")


def _load_r3_001_config(config_path: Path) -> PostD2ReproducibilityConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = PostD2ReproducibilityConfig()
    variants_raw = raw.get(
        "pythonhashseed_variants",
        ["<unset>" if v is None else v for v in defaults.pythonhashseed_variants],
    )
    variants = tuple(None if v in (None, "<unset>") else str(v) for v in variants_raw)
    return PostD2ReproducibilityConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        operations=tuple(raw.get("operations", defaults.operations)),
        splits=tuple(raw.get("splits", defaults.splits)),
        n_examples=raw.get("n_examples", defaults.n_examples),
        pythonhashseed_variants=variants,
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_002_config(config_path: Path) -> RelationSplitProtocolConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = RelationSplitProtocolConfig()
    return RelationSplitProtocolConfig(
        development_representativeness_bank_size=raw.get(
            "development_representativeness_bank_size",
            defaults.development_representativeness_bank_size,
        ),
        development_representativeness_query_examples=raw.get(
            "development_representativeness_query_examples",
            defaults.development_representativeness_query_examples,
        ),
        run_empirical_exposure_probe=raw.get(
            "run_empirical_exposure_probe", defaults.run_empirical_exposure_probe
        ),
        empirical_probe_seed=raw.get("empirical_probe_seed", defaults.empirical_probe_seed),
        device=raw.get("device", defaults.device),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def _load_r3_003_config(config_path: Path) -> FunctionalMetricsV2Config:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = FunctionalMetricsV2Config()
    return FunctionalMetricsV2Config(
        tau=raw.get("tau", defaults.tau),
        max_candidates=raw.get("max_candidates", defaults.max_candidates),
        looks=tuple(raw.get("looks", defaults.looks)),
        alpha_accept_episode=raw.get("alpha_accept_episode", defaults.alpha_accept_episode),
        alpha_reject_episode=raw.get("alpha_reject_episode", defaults.alpha_reject_episode),
        alpha_ref_episode=raw.get("alpha_ref_episode", defaults.alpha_ref_episode),
        output_dir=Path(raw.get("output_dir", defaults.output_dir)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B2 post-D2 repair -- explicit single-task dispatch"
    )
    parser.add_argument(
        "--task",
        required=True,
        help=f"Exact task ID to run. Implemented: {', '.join(_IMPLEMENTED_TASKS)}. "
        "No --all and no implicit next-task execution.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to this task's own configs/phase_b_b2_post_d2_<task>.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.task not in _IMPLEMENTED_TASKS:
        print(
            f"ERROR: task {args.task!r} is not implemented by this dispatcher. "
            f"Only {_IMPLEMENTED_TASKS} run today; every later task in "
            "docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md requires its own "
            "explicit user instruction and its own implementation before it "
            "can be dispatched here.",
            file=sys.stderr,
        )
        return 2

    if args.task == "B-C005R3-001":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_reproducibility.yaml")
        config = _load_r3_001_config(config_path)
        if args.output_dir is not None:
            config = PostD2ReproducibilityConfig(
                seeds=config.seeds,
                operations=config.operations,
                splits=config.splits,
                n_examples=config.n_examples,
                pythonhashseed_variants=config.pythonhashseed_variants,
                output_dir=args.output_dir,
            )
        report = run_post_d2_reproducibility_task(config)
        next_blocked = "B-C005R3-002 onward"
    elif args.task == "B-C005R3-002":
        config_path = args.config or Path("configs/phase_b_b2_post_d2_relation_split.yaml")
        r3_002_config = _load_r3_002_config(config_path)
        if args.output_dir is not None:
            r3_002_config = RelationSplitProtocolConfig(
                development_representativeness_bank_size=r3_002_config.development_representativeness_bank_size,
                development_representativeness_query_examples=r3_002_config.development_representativeness_query_examples,
                run_empirical_exposure_probe=r3_002_config.run_empirical_exposure_probe,
                empirical_probe_seed=r3_002_config.empirical_probe_seed,
                device=r3_002_config.device,
                output_dir=args.output_dir,
            )
        report = run_relation_split_protocol(r3_002_config)
        next_blocked = "B-C005R3-003 onward"
    else:  # B-C005R3-003
        config_path = args.config or Path("configs/phase_b_b2_post_d2_functional_metrics_v2.yaml")
        r3_003_config = _load_r3_003_config(config_path)
        if args.output_dir is not None:
            r3_003_config = FunctionalMetricsV2Config(
                tau=r3_003_config.tau,
                max_candidates=r3_003_config.max_candidates,
                looks=r3_003_config.looks,
                alpha_accept_episode=r3_003_config.alpha_accept_episode,
                alpha_reject_episode=r3_003_config.alpha_reject_episode,
                alpha_ref_episode=r3_003_config.alpha_ref_episode,
                output_dir=args.output_dir,
            )
        report = run_functional_metrics_v2_protocol(r3_003_config)
        next_blocked = "B-C005R3-004 onward"

    print(json.dumps(report["protocol"], indent=2))
    print(
        f"STOP: only {args.task} was executed. {next_blocked} and "
        "B-C006/Task Inference remain blocked pending an explicit next "
        "user instruction."
    )
    return 0 if report["protocol"]["result"] == "INFRASTRUCTURE_OR_PROTOCOL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
