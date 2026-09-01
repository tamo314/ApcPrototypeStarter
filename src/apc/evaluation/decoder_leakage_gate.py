"""Decoder leakage control gate (Phase A.1 Post-Correction Task A1-R002,
conditional STOP GATE).

`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R002: before any primitive
is credited with solving an operation (A1-R003 onward), this gate must show
that a decoder with **no primitive execution at all** -- and, per the audit
in `apc.core.execution` (see that module's "Decoder-input audit and
no-primitive/identity mode" docstring section), no task-specification token
anywhere in its input either -- cannot already solve the eight known
operations at high accuracy. If it could, a later "Correct primitive -> high
accuracy" result would be uninterpretable: the accuracy could be coming from
the Stable Core/decoder alone, not from the primitive
(`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 1's "scientifically
ambiguous" architecture, `AGENTS.md`'s "If Correct, Wrong, and None are all
high, suspect Stable Core or decoder leakage").

## Relationship to Task A1-C004's negative control (ADR-0021)

This is not a new phenomenon to discover: `apc.evaluation.
shared_core_generalization`'s `include_task_spec=False` variant is, by
construction, already exactly "train a shared core with no task segment ever
visible, evaluate decoder-only exact match" -- and `docs/DECISIONS.md`
ADR-0021 already measured it at 0.139 mean overall exact match (5 seeds),
far below both the 0.95 explicit-task threshold and any reasonable
"materially below" floor. This module deliberately **reuses** that same
training procedure (`train_shared_core`) rather than re-implementing it, for
two reasons: first, duplicating a training loop this codebase already has
one canonical implementation of would violate `AGENTS.md`'s "avoid unrelated
refactors"/DRY discipline for no scientific benefit; second, keeping the
model/data/generator identical to A1-C004's own negative control makes this
gate's own "future Correct target" (0.95) a fair, apples-to-apples number
from the *same* experimental family, not an arbitrary import from a
differently-configured run. What this module adds beyond re-running that
experiment: it evaluates through `apc.core.execution.
evaluate_exact_match_no_primitive` -- the audited, structurally
primitive-free causal-mode entry point A1-R002 itself adds -- rather than
`apc.core.generation.evaluate_exact_match` directly, and cross-checks the two
agree exactly (`DecoderLeakageGateReport.causal_mode_matches_plain_generation`),
so this gate's own recorded result is traceable to the specific execution
path future A1-R003+ gates will reuse for their "None" arm, not merely to
`apc.core.generation`'s older, pre-A1-C007 entry point.

## Why the acceptance threshold is 0.30, not left as prose

`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R002's acceptance text
("no-primitive path materially below future Correct target") does not
spell out a number, and `docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md`
has no dedicated "decoder leakage" hypothesis between H2a and H2b. Rather
than leave "materially below" undeclared, this module reuses H2b/H2c's own
predeclared "None <= 0.30" floor (`docs/EXPERIMENT_PLAN_PHASE_A1_POST_
CORRECTION.md` section 3) as `NO_PRIMITIVE_MATERIAL_CEILING`: this gate is,
in substance, pre-registering the "None" arm of that same causal ablation
matrix ahead of A1-R003 wiring a "Correct"/"Wrong" arm alongside it, so
reusing its own numeric floor keeps the two gates' verdicts on the same
scale rather than inventing a second, unrelated threshold. `passed` is
computed only against the *overall* pooled mean, matching every other
Phase A.1 gate's own convention (`SharedCoreGateMultiSeedReport.passed`);
per-operation numbers are still reported (and expected to vary --
`COPY`'s value-blind identity mapping is solvable without any task or
primitive information at all, per `docs/DECISIONS.md` ADR-0020/ADR-0021,
which is an expected, documented, non-gating property of this measurement,
not a defect in it).

## Scope discipline

Same restriction as every earlier Phase A.1 Correction/Post-Correction gate:
`apc.primitives`, `apc.plastic`, `apc.consolidation`, and `apc.meta` are
never imported here. This module trains and evaluates a plain
`apc.core.model.DecoderOnlyTransformer` only -- routing, primitive execution,
and the "Correct primitive" arm of the ablation matrix are A1-R003 onward,
not this task's.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apc.core.execution import evaluate_exact_match_no_primitive
from apc.core.generation import evaluate_exact_match
from apc.environments.generator import Example
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    shared_core_gate_config_from_dict,
    train_shared_core,
)

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "FUTURE_CORRECT_TARGET",
    "NO_PRIMITIVE_MATERIAL_CEILING",
    "DecoderLeakageGateConfig",
    "SharedCoreGateTrainConfig",
    "decoder_leakage_gate_config_from_dict",
    "DecoderLeakageGateReport",
    "run_decoder_leakage_gate",
    "DecoderLeakageGateMultiSeedReport",
    "run_decoder_leakage_gate_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md: ">=5 seeds" for a gate claim.
MIN_GATE_SEEDS = 5

# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md section 3 (H2b): the
# primitive-causality gate's own "Correct >= 0.95" target -- what A1-R002's
# "materially below future Correct target" acceptance text refers to.
FUTURE_CORRECT_TARGET = 0.95

# H2b/H2c's own predeclared "None <= 0.30" floor, reused here -- see module
# docstring "Why the acceptance threshold is 0.30" for the reasoning.
NO_PRIMITIVE_MATERIAL_CEILING = 0.30

# Alias kept for config-parsing convenience: this gate's config is exactly a
# SharedCoreGateConfig with include_task_spec forced False (see
# decoder_leakage_gate_config_from_dict).
DecoderLeakageGateConfig = SharedCoreGateConfig


def decoder_leakage_gate_config_from_dict(raw: dict[str, Any]) -> SharedCoreGateConfig:
    """Parse a `configs/phase_a1_decoder_leakage_gate.yaml`-shaped dict.

    `include_task_spec` is always forced to `False` regardless of what the
    file sets -- this gate's entire point is measuring the no-task-spec
    condition, so leaving it configurable would let a config file silently
    turn this into a different (and wrongly-labeled) measurement.
    """
    config = shared_core_gate_config_from_dict(raw)
    return dataclasses.replace(config, include_task_spec=False)


def _operation_of(example: Example) -> str:
    assert example.task_spec is not None
    (operation,) = example.task_spec.operation_sequence
    return operation


def _per_operation_exact_match(
    examples: Sequence[Example],
    predictions: Sequence[tuple[int, ...]],
    operation_names: Sequence[str],
    min_examples_per_operation: int,
) -> tuple[dict[str, float], dict[str, int]]:
    matches: dict[str, int] = dict.fromkeys(operation_names, 0)
    totals: dict[str, int] = dict.fromkeys(operation_names, 0)
    for example, prediction in zip(examples, predictions, strict=True):
        operation = _operation_of(example)
        totals[operation] += 1
        if prediction == example.target_tokens:
            matches[operation] += 1

    missing = [name for name in operation_names if totals[name] < min_examples_per_operation]
    if missing:
        raise ValueError(
            f"unseen eval batch has fewer than {min_examples_per_operation} examples for "
            f"operation(s) {missing}; increase num_unseen_eval_examples"
        )

    per_operation_exact_match = {name: matches[name] / totals[name] for name in operation_names}
    return per_operation_exact_match, totals


@dataclass(frozen=True)
class DecoderLeakageGateReport:
    """Everything observed while training and evaluating one seed's
    no-primitive decoder-only condition."""

    config: SharedCoreGateConfig
    steps_trained: int
    examples_seen: int
    final_train_loss: float
    overall_exact_match: float
    per_operation_exact_match: dict[str, float]
    per_operation_eval_counts: dict[str, int]
    num_unseen_eval_examples: int
    param_count: int
    trainable_param_count: int
    causal_mode_matches_plain_generation: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_decoder_leakage_gate(
    config: SharedCoreGateConfig, metrics_path: str | Path | None = None
) -> DecoderLeakageGateReport:
    """Train one fresh `DecoderOnlyTransformer` with no task segment ever
    visible (`train_shared_core`, `include_task_spec=False`), then evaluate
    overall and per-operation exact match through `apc.core.execution.
    evaluate_exact_match_no_primitive` -- the causal ablation matrix's
    "None" arm -- on a large, independently-drawn unseen batch.

    Cross-checks that the no-primitive causal-mode path and plain
    `apc.core.generation.evaluate_exact_match` agree exactly (both are the
    same computation -- content-only prompt, no bank/router/workspace --
    reached through two different entry points); this is a regression
    guard, not part of the gate's own pass/fail verdict.
    """
    if config.include_task_spec:
        raise ValueError(
            "DecoderLeakageGate always measures the no-task-spec condition; "
            "config.include_task_spec must be False (use "
            "decoder_leakage_gate_config_from_dict, which forces this)"
        )

    start = time.perf_counter()
    trained = train_shared_core(config, metrics_path=metrics_path)
    model, tokens, generator, device = (
        trained.model,
        trained.tokens,
        trained.generator,
        trained.device,
    )

    unseen_examples = generator.generate_online(
        config.num_unseen_eval_examples, step=0, split="test"
    )
    overall_exact_match, predictions = evaluate_exact_match_no_primitive(
        model, unseen_examples, tokens, device=device
    )
    plain_exact_match, plain_predictions = evaluate_exact_match(
        model, unseen_examples, tokens, device, include_task_spec=False
    )
    causal_mode_matches_plain_generation = (
        overall_exact_match == plain_exact_match and predictions == plain_predictions
    )

    per_operation_exact_match, per_operation_counts = _per_operation_exact_match(
        unseen_examples, predictions, config.operation_names, config.min_examples_per_operation
    )

    return DecoderLeakageGateReport(
        config=config,
        steps_trained=config.train.steps,
        examples_seen=config.train.steps * config.train.batch_size,
        final_train_loss=trained.final_train_loss,
        overall_exact_match=overall_exact_match,
        per_operation_exact_match=per_operation_exact_match,
        per_operation_eval_counts=per_operation_counts,
        num_unseen_eval_examples=len(unseen_examples),
        param_count=model.num_parameters(),
        trainable_param_count=model.num_parameters(trainable_only=True),
        causal_mode_matches_plain_generation=causal_mode_matches_plain_generation,
        wall_clock_seconds=time.perf_counter() - start,
        device=str(device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class DecoderLeakageGateMultiSeedReport:
    """A1-R002's verdict aggregated across seeds."""

    seeds: tuple[int, ...]
    per_seed: tuple[DecoderLeakageGateReport, ...]
    operation_names: tuple[str, ...]
    mean_overall_exact_match: float
    stdev_overall_exact_match: float
    min_overall_exact_match: float
    max_overall_exact_match: float
    per_operation_mean_exact_match: dict[str, float]
    material_ceiling: float
    future_correct_target: float
    materially_below_future_correct_target: bool
    causal_mode_matches_plain_generation: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "operation_names": list(self.operation_names),
            "mean_overall_exact_match": self.mean_overall_exact_match,
            "stdev_overall_exact_match": self.stdev_overall_exact_match,
            "min_overall_exact_match": self.min_overall_exact_match,
            "max_overall_exact_match": self.max_overall_exact_match,
            "per_operation_mean_exact_match": self.per_operation_mean_exact_match,
            "material_ceiling": self.material_ceiling,
            "future_correct_target": self.future_correct_target,
            "materially_below_future_correct_target": (
                self.materially_below_future_correct_target
            ),
            "causal_mode_matches_plain_generation": self.causal_mode_matches_plain_generation,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_decoder_leakage_gate_multi_seed(
    base_config: SharedCoreGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    material_ceiling: float = NO_PRIMITIVE_MATERIAL_CEILING,
    future_correct_target: float = FUTURE_CORRECT_TARGET,
) -> DecoderLeakageGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed, `include_task_spec` always forced `False`) and aggregate.
    Does not raise if `len(seeds) < MIN_GATE_SEEDS`; `meets_seed_policy`
    reports it honestly instead, matching every other Phase A.1 gate.
    """
    base_config = dataclasses.replace(base_config, include_task_spec=False)
    per_seed: list[DecoderLeakageGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        metrics_path = run_dir_path / f"seed_{seed}" / "metrics.jsonl" if run_dir_path else None
        report = run_decoder_leakage_gate(config, metrics_path=metrics_path)
        if run_dir_path is not None:
            seed_dir = run_dir_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    overall_values = [report.overall_exact_match for report in per_seed]
    mean_value, stdev_value, min_value, max_value = _summarize(overall_values)

    per_operation_mean: dict[str, float] = {}
    for name in base_config.operation_names:
        op_values = [report.per_operation_exact_match[name] for report in per_seed]
        per_operation_mean[name] = statistics.fmean(op_values)

    materially_below = mean_value <= material_ceiling
    causal_mode_matches_plain_generation = all(
        report.causal_mode_matches_plain_generation for report in per_seed
    )
    passed = materially_below

    return DecoderLeakageGateMultiSeedReport(
        seeds=tuple(report.config.seed for report in per_seed),
        per_seed=tuple(per_seed),
        operation_names=tuple(base_config.operation_names),
        mean_overall_exact_match=mean_value,
        stdev_overall_exact_match=stdev_value,
        min_overall_exact_match=min_value,
        max_overall_exact_match=max_value,
        per_operation_mean_exact_match=per_operation_mean,
        material_ceiling=material_ceiling,
        future_correct_target=future_correct_target,
        materially_below_future_correct_target=materially_below,
        causal_mode_matches_plain_generation=causal_mode_matches_plain_generation,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
