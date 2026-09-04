"""Compact Direct-Learning Control (Task A1-B007X-004 / Milestone B007X Gate).

Evaluates whether the final compact candidate architecture (<=25k parameters) can learn
novel operations directly from supervised labels under matched discovery budget.

This provides the mandatory direct-learning baseline required prior to functional distillation
in Task A1-B007X-005.

Acceptance Criteria:
1. Candidate parameters <= 25,000 (T0 Compact: 17,098).
2. Matched discovery budget:
   - Identical data streams and splits (800 train, 200 eval).
   - Frozen shared task-blind Core content encoder.
   - Identical optimizer, batch size, steps (400), eval interval (25).
   - 5 decision seeds: [0, 1, 2, 3, 4].
   - 3 novel operations: SWAP_PAIRS, INVERT_HALF, ROTATE_TRIPLETS.
3. Report matched:
   - params
   - final EM (mean, std, min, max)
   - steps / examples to 0.90 and 0.95 (median, mean)
   - seed success rate (EM >= 0.95)
"""

from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from apc.core.data import IGNORE_INDEX
from apc.environments.generator import Example
from apc.environments.operations import DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.discovery_capacity_harness import (
    CapacityTier,
    _encode_content_features,
    _eval_primitive_exact_match,
    build_tier_primitive,
    generate_novel_examples,
)
from apc.evaluation.discovery_capacity_sweep import (
    compute_normalized_auc,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _get_or_train_frozen_shared_core,
)
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
MAX_CANDIDATE_PARAMS: Final[int] = 25000
DEFAULT_CANDIDATE_TIER: Final[str] = CapacityTier.T0_COMPACT.value


@dataclass(frozen=True)
class CompactDirectControlConfig:
    """Configuration for Task A1-B007X-004 compact direct-learning control."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    novel_operations: tuple[str, ...] = DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    candidate_tier: str = DEFAULT_CANDIDATE_TIER
    max_candidate_params: int = MAX_CANDIDATE_PARAMS

    train_steps: int = 400
    eval_interval: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4

    vocab_size: int = 10
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    num_eval_examples: int = 200
    num_train_examples: int = 800

    device: str = "auto"
    shared_encoder_checkpoint: str | None = None
    core_train_steps: int = 6000
    bank_train_steps: int = 6000
    save_checkpoints: bool = True
    model: dict[str, Any] = dataclasses.field(default_factory=_default_model_config)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def compact_direct_control_config_from_dict(raw: dict[str, Any]) -> CompactDirectControlConfig:
    """Create configuration from raw dictionary."""
    defaults = CompactDirectControlConfig()
    return CompactDirectControlConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        novel_operations=tuple(raw.get("novel_operations", defaults.novel_operations)),
        candidate_tier=raw.get("candidate_tier", defaults.candidate_tier),
        max_candidate_params=raw.get("max_candidate_params", defaults.max_candidate_params),
        train_steps=raw.get("train_steps", defaults.train_steps),
        eval_interval=raw.get("eval_interval", defaults.eval_interval),
        batch_size=raw.get("batch_size", defaults.batch_size),
        lr=raw.get("lr", defaults.lr),
        weight_decay=raw.get("weight_decay", defaults.weight_decay),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        num_train_examples=raw.get("num_train_examples", defaults.num_train_examples),
        device=raw.get("device", defaults.device),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        save_checkpoints=raw.get("save_checkpoints", defaults.save_checkpoints),
        model=dict(raw.get("model", defaults.model)),
    )


@dataclass(frozen=True)
class DirectControlSeedRun:
    """Detailed metrics from a single direct-learning control run."""

    seed: int
    operation: str
    tier: str
    parameters: int
    final_em: float
    final_loss: float
    success: bool
    step_to_90: int | None
    step_to_95: int | None
    examples_to_90: int | None
    examples_to_95: int | None
    auc: float
    wall_clock_seconds: float
    peak_memory_bytes: int
    checkpoint_path: str | None
    em_history: list[tuple[int, float]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AggregatedDirectControlResult:
    """Aggregated direct-learning control metrics across seeds for one operation."""

    operation: str
    candidate_tier: str
    num_seeds: int
    parameters: int
    max_candidate_params: int
    params_compliant: bool
    mean_em: float
    std_em: float
    min_em: float
    max_em: float
    success_rate: float
    median_step_to_90: float | None
    median_step_to_95: float | None
    mean_step_to_95: float | None
    median_examples_to_90: float | None
    median_examples_to_95: float | None
    mean_auc: float
    mean_wall_clock: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CompactDirectControlSummary:
    """Overall summary of Task A1-B007X-004 Compact Direct-Learning Control."""

    seeds: list[int]
    novel_operations: list[str]
    candidate_tier: str
    all_runs: list[DirectControlSeedRun]
    aggregated: dict[str, AggregatedDirectControlResult]
    all_compliant: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "novel_operations": self.novel_operations,
            "candidate_tier": self.candidate_tier,
            "all_compliant": self.all_compliant,
            "elapsed_seconds": self.elapsed_seconds,
            "aggregated": {op: res.to_dict() for op, res in self.aggregated.items()},
            "all_runs": [run.to_dict() for run in self.all_runs],
        }


def train_single_direct_control_run(
    core: Any,
    operation: str,
    tier: str,
    seed: int,
    train_examples: list[Example],
    eval_examples: list[Example],
    *,
    train_steps: int,
    eval_interval: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    save_checkpoint_dir: Path | None = None,
) -> DirectControlSeedRun:
    """Train a single compact candidate primitive directly from supervised labels."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    d_model = core.model.config.d_model
    prim = build_tier_primitive(
        tier, operation, vocab_size=DEFAULT_VOCAB_SIZE, d_model=d_model
    ).to(device)
    prim.train()

    optimizer = torch.optim.AdamW(prim.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    step_to_90: int | None = None
    step_to_95: int | None = None
    final_loss = 0.0
    em_history: list[tuple[int, float]] = []

    # Deterministic batch sampling order identical to discovery sweep
    batch_rng = random.Random(seed * 10007 + 42)

    start_time = time.perf_counter()

    for step in range(1, train_steps + 1):
        batch = [batch_rng.choice(train_examples) for _ in range(batch_size)]
        targets = [torch.tensor(ex.target_tokens, dtype=torch.long, device=device) for ex in batch]
        output_lens = [len(ex.target_tokens) for ex in batch]

        content_features, content_lengths = _encode_content_features(core, batch, device)

        optimizer.zero_grad()
        logits = prim(
            content_features=content_features,
            content_lengths=content_lengths,
            output_lengths=output_lens,
        )

        max_out = max(output_lens)
        target_tensor = torch.full(
            (len(batch), max_out), IGNORE_INDEX, dtype=torch.long, device=device
        )
        for b_i, t in enumerate(targets):
            target_tensor[b_i, : len(t)] = t

        loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if step % eval_interval == 0 or step == train_steps:
            em = _eval_primitive_exact_match(core, prim, eval_examples, device)
            em_history.append((step, em))
            if em >= 0.90 and step_to_90 is None:
                step_to_90 = step
            if em >= 0.95 and step_to_95 is None:
                step_to_95 = step

    elapsed = time.perf_counter() - start_time
    final_em = em_history[-1][1] if em_history else 0.0
    auc = compute_normalized_auc(em_history, train_steps)

    peak_mem = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    examples_to_90 = step_to_90 * batch_size if step_to_90 is not None else None
    examples_to_95 = step_to_95 * batch_size if step_to_95 is not None else None

    checkpoint_path_str: str | None = None
    if save_checkpoint_dir is not None:
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = save_checkpoint_dir / f"{operation.lower()}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": prim.state_dict(),
                "operation": operation,
                "tier": tier,
                "seed": seed,
                "final_em": final_em,
                "parameters": prim.num_parameters(),
            },
            ckpt_file,
        )
        checkpoint_path_str = str(ckpt_file)

    return DirectControlSeedRun(
        seed=seed,
        operation=operation,
        tier=tier,
        parameters=prim.num_parameters(),
        final_em=final_em,
        final_loss=final_loss,
        success=final_em >= 0.95,
        step_to_90=step_to_90,
        step_to_95=step_to_95,
        examples_to_90=examples_to_90,
        examples_to_95=examples_to_95,
        auc=auc,
        wall_clock_seconds=elapsed,
        peak_memory_bytes=peak_mem,
        checkpoint_path=checkpoint_path_str,
        em_history=em_history,
    )


def aggregate_direct_control_runs(
    operation: str,
    tier: str,
    runs: Sequence[DirectControlSeedRun],
    max_candidate_params: int = MAX_CANDIDATE_PARAMS,
) -> AggregatedDirectControlResult:
    """Aggregate direct-learning control metrics across seeds for one operation."""
    if not runs:
        raise ValueError(f"No runs to aggregate for op '{operation}'")

    n = len(runs)
    params = runs[0].parameters
    compliant = params <= max_candidate_params
    ems = [r.final_em for r in runs]
    mean_em = statistics.mean(ems)
    std_em = statistics.stdev(ems) if n > 1 else 0.0
    min_em = min(ems)
    max_em = max(ems)
    success_rate = sum(1 for r in runs if r.success) / n

    # Step to 90 calculations
    completed_steps_90 = [r.step_to_90 for r in runs if r.step_to_90 is not None]
    if len(completed_steps_90) >= (n + 1) // 2:
        median_step_90 = float(statistics.median(completed_steps_90))
        sample_run_90 = [r for r in runs if r.step_to_90 is not None][0]
        ex_per_step = (
            float(sample_run_90.examples_to_90 / sample_run_90.step_to_90)
            if (sample_run_90.step_to_90 is not None and sample_run_90.examples_to_90 is not None)
            else None
        )
        median_ex_90 = median_step_90 * ex_per_step if ex_per_step is not None else None
    else:
        median_step_90 = None
        median_ex_90 = None

    # Step to 95 calculations
    completed_steps_95 = [r.step_to_95 for r in runs if r.step_to_95 is not None]
    if len(completed_steps_95) >= (n + 1) // 2:
        median_step_95 = float(statistics.median(completed_steps_95))
        mean_step_95 = float(statistics.mean(completed_steps_95))
        sample_run_95 = [r for r in runs if r.step_to_95 is not None][0]
        ex_per_step = (
            float(sample_run_95.examples_to_95 / sample_run_95.step_to_95)
            if (sample_run_95.step_to_95 is not None and sample_run_95.examples_to_95 is not None)
            else None
        )
        median_ex_95 = median_step_95 * ex_per_step if ex_per_step is not None else None
    else:
        median_step_95 = None
        mean_step_95 = None
        median_ex_95 = None

    aucs = [r.auc for r in runs]
    mean_auc = statistics.mean(aucs)

    clocks = [r.wall_clock_seconds for r in runs]
    mean_clock = statistics.mean(clocks)

    peak_mem = max(r.peak_memory_bytes for r in runs)

    return AggregatedDirectControlResult(
        operation=operation,
        candidate_tier=tier,
        num_seeds=n,
        parameters=params,
        max_candidate_params=max_candidate_params,
        params_compliant=compliant,
        mean_em=mean_em,
        std_em=std_em,
        min_em=min_em,
        max_em=max_em,
        success_rate=success_rate,
        median_step_to_90=median_step_90,
        median_step_to_95=median_step_95,
        mean_step_to_95=mean_step_95,
        median_examples_to_90=median_ex_90,
        median_examples_to_95=median_ex_95,
        mean_auc=mean_auc,
        mean_wall_clock=mean_clock,
        peak_memory_bytes=peak_mem,
    )


def run_compact_direct_control(
    config: CompactDirectControlConfig,
    output_dir: Path | None = None,
) -> CompactDirectControlSummary:
    """Run full compact direct-learning control experiment across seeds and novel operations."""
    start_total = time.perf_counter()

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # 1. Obtain frozen shared Core
    u_config = UnifiedBenchmarkConfig(
        seed=config.seeds[0],
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=str(device),
        core_train_steps=config.core_train_steps,
        parameterized_train_steps=config.bank_train_steps,
        parameter_free_train_steps=config.bank_train_steps,
        min_unseen_eval_examples=10,
        shared_encoder_checkpoint=config.shared_encoder_checkpoint,
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=output_dir)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    ckpt_dir = (output_dir / "checkpoints") if (output_dir and config.save_checkpoints) else None

    all_runs: list[DirectControlSeedRun] = []
    aggregated: dict[str, AggregatedDirectControlResult] = {}

    for op in config.novel_operations:
        op_runs: list[DirectControlSeedRun] = []
        for seed in config.seeds:
            set_seed(seed)

            # Generate identical datasets per (op, seed)
            train_exs = generate_novel_examples(
                seed,
                config.num_train_examples,
                operation=op,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )
            eval_exs = generate_novel_examples(
                seed,
                config.num_eval_examples,
                operation=op,
                split="test",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            run_res = train_single_direct_control_run(
                core=core,
                operation=op,
                tier=config.candidate_tier,
                seed=seed,
                train_examples=train_exs,
                eval_examples=eval_exs,
                train_steps=config.train_steps,
                eval_interval=config.eval_interval,
                batch_size=config.batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                device=device,
                save_checkpoint_dir=ckpt_dir,
            )
            op_runs.append(run_res)
            all_runs.append(run_res)

        agg = aggregate_direct_control_runs(
            op,
            config.candidate_tier,
            op_runs,
            max_candidate_params=config.max_candidate_params,
        )
        aggregated[op] = agg

    elapsed = time.perf_counter() - start_total
    all_compliant = all(agg.params_compliant for agg in aggregated.values())

    summary = CompactDirectControlSummary(
        seeds=list(config.seeds),
        novel_operations=list(config.novel_operations),
        candidate_tier=config.candidate_tier,
        all_runs=all_runs,
        aggregated=aggregated,
        all_compliant=all_compliant,
        elapsed_seconds=elapsed,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )

        # Build baseline lookup for Task A1-B007X-005 distillation comparison
        control_baseline = {
            "metadata": {
                "candidate_tier": config.candidate_tier,
                "max_candidate_params": config.max_candidate_params,
                "num_seeds": len(config.seeds),
                "train_steps": config.train_steps,
                "batch_size": config.batch_size,
            },
            "baselines": {
                op: {
                    "parameters": agg.parameters,
                    "params_compliant": agg.params_compliant,
                    "mean_em": agg.mean_em,
                    "std_em": agg.std_em,
                    "min_em": agg.min_em,
                    "max_em": agg.max_em,
                    "success_rate": agg.success_rate,
                    "median_step_to_90": agg.median_step_to_90,
                    "median_step_to_95": agg.median_step_to_95,
                    "median_examples_to_95": agg.median_examples_to_95,
                    "mean_auc": agg.mean_auc,
                }
                for op, agg in aggregated.items()
            },
        }
        (output_dir / "control_baseline.json").write_text(
            json.dumps(control_baseline, indent=2), encoding="utf-8"
        )

        report_md = _generate_control_markdown_report(summary)
        (output_dir / "report.json").write_text(
            json.dumps({"markdown": report_md, "summary": summary.to_dict()}, indent=2),
            encoding="utf-8",
        )

    return summary


def _generate_control_markdown_report(summary: CompactDirectControlSummary) -> str:
    """Generate markdown report for Task A1-B007X-004 Compact Direct Control."""
    lines: list[str] = []
    lines.append("# Task A1-B007X-004: Compact Direct-Learning Control Baseline")
    lines.append("")
    lines.append(f"- **Seeds:** `{summary.seeds}`")
    lines.append(f"- **Candidate Tier:** `{summary.candidate_tier}`")
    lines.append(f"- **All Params Compliant (<= 25k):** `{summary.all_compliant}`")
    lines.append(f"- **Elapsed Wall-Clock:** `{summary.elapsed_seconds:.2f}s`")
    lines.append("")
    lines.append("## Aggregated Control Metrics")
    lines.append("")
    header = (
        "| Operation | Params | Compliant | Mean EM (std) | Min..Max EM | "
        "Success (>=0.95) | Med Step 90 | Med Step 95 | Med Ex 95 | Mean AUC | Mean Time |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for op, agg in summary.aggregated.items():
        s90_str = f"{agg.median_step_to_90:.1f}" if agg.median_step_to_90 is not None else "N/A"
        s95_str = f"{agg.median_step_to_95:.1f}" if agg.median_step_to_95 is not None else "N/A"
        ex95 = agg.median_examples_to_95
        ex95_str = f"{ex95:.0f}" if ex95 is not None else "N/A"
        lines.append(
            f"| **{op}** | {agg.parameters:,} | {agg.params_compliant} | "
            f"{agg.mean_em:.4f} (±{agg.std_em:.4f}) | {agg.min_em:.4f}..{agg.max_em:.4f} | "
            f"{agg.success_rate:.1%} | {s90_str} | {s95_str} | {ex95_str} | "
            f"{agg.mean_auc:.4f} | {agg.mean_wall_clock:.2f}s |"
        )
    lines.append("")
    lines.append("## Per-Seed Breakdown")
    lines.append("")
    seed_header = (
        "| Operation | Seed | Parameters | Final EM | Success | Step 90 | "
        "Step 95 | AUC | Time (s) | Checkpoint |"
    )
    lines.append(seed_header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.all_runs:
        s90 = str(r.step_to_90) if r.step_to_90 is not None else "N/A"
        s95 = str(r.step_to_95) if r.step_to_95 is not None else "N/A"
        ckpt = Path(r.checkpoint_path).name if r.checkpoint_path else "None"
        row = (
            f"| {r.operation} | {r.seed} | {r.parameters:,} | {r.final_em:.4f} | "
            f"{r.success} | {s90} | {s95} | {r.auc:.4f} | {r.wall_clock_seconds:.2f}s | `{ckpt}` |"
        )
        lines.append(row)
    return "\n".join(lines)
