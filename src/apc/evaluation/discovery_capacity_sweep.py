"""Temporary Discovery Capacity Sweep (Task A1-B007X-003 / Milestone B007X Gate).

Evaluates whether overcomplete temporary capacity improves discovery over compact capacity:
C_discover > C_represent.

Across >=5 seeds and >=2 novel operations:
1. Runs matched capacity ladder:
   - T0 (Compact Control): ~17k parameters (17,098)
   - T1 (Medium): ~67k parameters (67,082)
   - T2 (Overcomplete): ~137k parameters (137,482, 8.04x >= 4x)
2. Uses identical data streams, batch sampling order, optimizer family, and stopping rules.
3. Reports per task/tier:
   - final EM
   - success rate (EM >= 0.95)
   - steps/examples to 0.90
   - steps/examples to 0.95
   - learning-curve AUC
   - wall-clock seconds
   - peak memory (bytes)
   - parameter counts
4. Evaluates discovery-advantage criterion:
   - Reliability gap: Large mean EM >= 0.95 while Compact mean EM <= 0.80 across >=5 seeds.
   - Efficiency gap: Both succeed, but Large reaches 0.95 with <= 50% of Compact median steps
     and no worse seed reliability.
   - Otherwise: No discovery-capacity advantage demonstrated (preserve negative results).
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
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _get_or_train_frozen_shared_core,
)
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
DEFAULT_TIERS: Final[tuple[str, ...]] = (
    CapacityTier.T0_COMPACT.value,
    CapacityTier.T1_MEDIUM.value,
    CapacityTier.T2_OVERCOMPLETE.value,
)


@dataclass(frozen=True)
class DiscoveryCapacitySweepConfig:
    """Configuration for Task A1-B007X-003 discovery capacity sweep."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    novel_operations: tuple[str, ...] = DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    tiers: tuple[str, ...] = DEFAULT_TIERS

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
    model: dict[str, Any] = dataclasses.field(default_factory=_default_model_config)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def discovery_capacity_sweep_config_from_dict(raw: dict[str, Any]) -> DiscoveryCapacitySweepConfig:
    """Create sweep configuration from raw dictionary."""
    defaults = DiscoveryCapacitySweepConfig()
    return DiscoveryCapacitySweepConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        novel_operations=tuple(raw.get("novel_operations", defaults.novel_operations)),
        tiers=tuple(raw.get("tiers", defaults.tiers)),
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
        model=dict(raw.get("model", defaults.model)),
    )


@dataclass(frozen=True)
class SingleRunResult:
    """Metrics recorded from a single training run (one seed, operation, tier)."""

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
    em_history: list[tuple[int, float]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AggregatedTierResult:
    """Aggregated metrics across seeds for one operation and tier."""

    operation: str
    tier: str
    num_seeds: int
    parameters: int
    mean_em: float
    std_em: float
    success_rate: float
    median_step_to_95: float | None
    mean_step_to_95: float | None
    median_examples_to_95: float | None
    mean_auc: float
    mean_wall_clock: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class OperationVerdict:
    """Verdict on whether discovery capacity advantage is demonstrated for an operation."""

    operation: str
    compact_tier: str
    large_tier: str
    reliability_gap_passed: bool
    efficiency_gap_passed: bool
    advantage_found: bool
    summary_reason: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SweepRunSummary:
    """Overall summary of Task A1-B007X-003 capacity sweep."""

    seeds: list[int]
    novel_operations: list[str]
    tiers: list[str]
    all_runs: list[SingleRunResult]
    aggregated: dict[str, dict[str, AggregatedTierResult]]
    verdicts: dict[str, OperationVerdict]
    overall_advantage_found: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": self.seeds,
            "novel_operations": self.novel_operations,
            "tiers": self.tiers,
            "overall_advantage_found": self.overall_advantage_found,
            "elapsed_seconds": self.elapsed_seconds,
            "verdicts": {op: v.to_dict() for op, v in self.verdicts.items()},
            "aggregated": {
                op: {t: res.to_dict() for t, res in t_dict.items()}
                for op, t_dict in self.aggregated.items()
            },
            "all_runs": [run.to_dict() for run in self.all_runs],
        }


def compute_normalized_auc(history: Sequence[tuple[int, float]], total_steps: int) -> float:
    """Compute normalized trapezoidal AUC from step-EM history (0.0 to 1.0)."""
    if not history or total_steps <= 0:
        return 0.0

    points = [(0, 0.0)] + list(history)
    area = 0.0
    for i in range(len(points) - 1):
        s0, e0 = points[i]
        s1, e1 = points[i + 1]
        area += 0.5 * (e0 + e1) * (s1 - s0)

    return area / total_steps


def train_single_run(
    core: Any,
    tier: str,
    operation: str,
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
) -> SingleRunResult:
    """Train a single primitive tier under matched conditions."""
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

    # Deterministic batch sampling order for this seed + operation
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

    return SingleRunResult(
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
        em_history=em_history,
    )


def aggregate_tier_runs(
    operation: str,
    tier: str,
    runs: Sequence[SingleRunResult],
) -> AggregatedTierResult:
    """Aggregate run metrics across seeds for one operation and tier."""
    if not runs:
        raise ValueError(f"No runs to aggregate for op '{operation}', tier '{tier}'")

    n = len(runs)
    params = runs[0].parameters
    ems = [r.final_em for r in runs]
    mean_em = statistics.mean(ems)
    std_em = statistics.stdev(ems) if n > 1 else 0.0
    success_rate = sum(1 for r in runs if r.success) / n

    # Step to 95 calculations
    completed_steps_95 = [r.step_to_95 for r in runs if r.step_to_95 is not None]
    if len(completed_steps_95) == n:
        median_step_95 = float(statistics.median(completed_steps_95))
        mean_step_95 = float(statistics.mean(completed_steps_95))
        ex_per_step = (
            float(runs[0].examples_to_95 / runs[0].step_to_95)
            if (runs[0].step_to_95 is not None and runs[0].examples_to_95 is not None)
            else None
        )
        median_ex_95 = median_step_95 * ex_per_step if ex_per_step is not None else None
    elif len(completed_steps_95) > n / 2:
        median_step_95 = float(statistics.median(completed_steps_95))
        mean_step_95 = float(statistics.mean(completed_steps_95))
        sample_run = [r for r in runs if r.step_to_95 is not None][0]
        ex_per_step = (
            float(sample_run.examples_to_95 / sample_run.step_to_95)
            if (sample_run.step_to_95 is not None and sample_run.examples_to_95 is not None)
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

    return AggregatedTierResult(
        operation=operation,
        tier=tier,
        num_seeds=n,
        parameters=params,
        mean_em=mean_em,
        std_em=std_em,
        success_rate=success_rate,
        median_step_to_95=median_step_95,
        mean_step_to_95=mean_step_95,
        median_examples_to_95=median_ex_95,
        mean_auc=mean_auc,
        mean_wall_clock=mean_clock,
        peak_memory_bytes=peak_mem,
    )


def evaluate_operation_verdict(
    operation: str,
    compact_agg: AggregatedTierResult,
    large_agg: AggregatedTierResult,
) -> OperationVerdict:
    """Evaluate whether temporary discovery capacity advantage is demonstrated."""
    # 1. Reliability Gap: large mean EM >= 0.95 and compact mean EM <= 0.80
    rel_gap = (large_agg.mean_em >= 0.95) and (compact_agg.mean_em <= 0.80)

    # 2. Efficiency Gap: both succeed, large reaches 0.95 with <= 50% of compact median steps,
    # and large reliability >= compact reliability
    eff_gap = False
    if (
        large_agg.success_rate >= compact_agg.success_rate
        and large_agg.mean_em >= 0.95
        and compact_agg.mean_em >= 0.90
        and compact_agg.median_step_to_95 is not None
        and large_agg.median_step_to_95 is not None
    ):
        step_ratio = large_agg.median_step_to_95 / compact_agg.median_step_to_95
        if step_ratio <= 0.50:
            eff_gap = True

    advantage_found = rel_gap or eff_gap

    reasons: list[str] = []
    if rel_gap:
        reasons.append(
            f"Reliability gap satisfied: Large EM {large_agg.mean_em:.4f} >= 0.95 vs "
            f"Compact EM {compact_agg.mean_em:.4f} <= 0.80"
        )
    if (
        eff_gap
        and large_agg.median_step_to_95 is not None
        and compact_agg.median_step_to_95 is not None
    ):
        ratio = large_agg.median_step_to_95 / compact_agg.median_step_to_95
        reasons.append(
            f"Efficiency gap satisfied: Large median step {large_agg.median_step_to_95} <= 50% of "
            f"Compact ({compact_agg.median_step_to_95}, ratio {ratio:.2%})"
        )
    if not advantage_found:
        reasons.append(
            f"No advantage: Compact mean EM={compact_agg.mean_em:.4f} "
            f"(med step {compact_agg.median_step_to_95}), "
            f"Large mean EM={large_agg.mean_em:.4f} (med step {large_agg.median_step_to_95})"
        )

    summary_reason = "; ".join(reasons)

    return OperationVerdict(
        operation=operation,
        compact_tier=compact_agg.tier,
        large_tier=large_agg.tier,
        reliability_gap_passed=rel_gap,
        efficiency_gap_passed=eff_gap,
        advantage_found=advantage_found,
        summary_reason=summary_reason,
    )


def run_discovery_capacity_sweep(
    config: DiscoveryCapacitySweepConfig,
    *,
    output_dir: Path | None = None,
) -> SweepRunSummary:
    """Execute complete Task A1-B007X-003 discovery capacity sweep."""
    start_time = time.perf_counter()

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # 1. Load or prepare shared frozen core (task-blind content encoder)
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

    all_runs: list[SingleRunResult] = []
    runs_by_op_tier: dict[str, dict[str, list[SingleRunResult]]] = {
        op: {tier: [] for tier in config.tiers} for op in config.novel_operations
    }

    # 2. Iterate across operations and seeds
    for op_name in config.novel_operations:
        for seed in config.seeds:
            # Generate matched data stream for this seed and operation
            train_exs = generate_novel_examples(
                seed,
                config.num_train_examples,
                operation=op_name,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )
            eval_exs = generate_novel_examples(
                seed,
                config.num_eval_examples,
                operation=op_name,
                split="test",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            # Train all tiers on the identical data stream
            for tier in config.tiers:
                set_seed(seed)
                run_res = train_single_run(
                    core,
                    tier,
                    op_name,
                    seed,
                    train_exs,
                    eval_exs,
                    train_steps=config.train_steps,
                    eval_interval=config.eval_interval,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    device=device,
                )
                all_runs.append(run_res)
                runs_by_op_tier[op_name][tier].append(run_res)

    # 3. Aggregate results per operation and tier
    aggregated: dict[str, dict[str, AggregatedTierResult]] = {}
    verdicts: dict[str, OperationVerdict] = {}

    compact_tier_name = CapacityTier.T0_COMPACT.value
    large_tier_name = (
        CapacityTier.T2_OVERCOMPLETE.value
        if CapacityTier.T2_OVERCOMPLETE.value in config.tiers
        else config.tiers[-1]
    )

    for op_name in config.novel_operations:
        aggregated[op_name] = {}
        for tier in config.tiers:
            agg_res = aggregate_tier_runs(op_name, tier, runs_by_op_tier[op_name][tier])
            aggregated[op_name][tier] = agg_res

        # Evaluate verdict between compact control (T0) and large overcomplete (T2)
        v = evaluate_operation_verdict(
            op_name,
            aggregated[op_name][compact_tier_name],
            aggregated[op_name][large_tier_name],
        )
        verdicts[op_name] = v

    overall_advantage = any(v.advantage_found for v in verdicts.values())
    elapsed = time.perf_counter() - start_time

    summary = SweepRunSummary(
        seeds=list(config.seeds),
        novel_operations=list(config.novel_operations),
        tiers=list(config.tiers),
        all_runs=all_runs,
        aggregated=aggregated,
        verdicts=verdicts,
        overall_advantage_found=overall_advantage,
        elapsed_seconds=elapsed,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "summary": summary.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )

    return summary
