"""Shared-core systematic-generalization gate (Phase A.1 Correction Task
A1-C004, STOP GATE, H1b).

`docs/exec-plans/active/PHASE_A1_CORRECTION.md` A1-CM3 / `docs/
AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`'s "Shared-core rule": before learned
primitive routing is tested, the repository must demonstrate that **one
shared Stable Core** can execute multiple operations when the requested
operation and its arguments are explicitly observable. `apc.evaluation.
stable_core_generalization` (Task A1-006) only ever proved this per
operation, one model each (retained as H1a, per `docs/DECISIONS.md`
ADR-0017/ADR-0020: pooling operations without an explicit task signal is not
even a well-posed learning problem -- `SELECT`/`COUNT`/`SHIFT`/`BIND` sample
a hidden per-instance parameter never shown to the model, and even among the
four parameter-free operations, *which* operation applies to a given example
is itself never revealed, so `COPY`/`NEGATE`/`ACCUMULATE` are pairwise
unidentifiable from content alone). This module is H1b: the same
well-posedness fix (`apc.environments.task_spec.TaskSpec`, threaded into the
model input by `apc.core.data.encode_task_spec`/`include_task_spec=True`,
Tasks A1-C001/A1-C003) applied to `apc.environments.generator.
build_mixed_operation_generator`'s single identifiable mixed-operation
stream (Task A1-C002), trained as *one* shared model, evaluated for both
overall and per-operation unseen-content exact match.

Deliberately minimal, mirroring `apc.evaluation.stable_core_generalization`'s
own scope discipline:

- **K only, no plastic/consolidation/routing.** This module trains a plain
  `apc.core.model.DecoderOnlyTransformer` directly with cross-entropy -- no
  `apc.primitives`, `apc.plastic`, `apc.consolidation`, or `apc.meta` import
  anywhere in this file. Composition (`C`), novel operations (`N`), and
  recurrence (`R`) stay out of scope (`build_mixed_operation_generator` pins
  `max_depth=1`, so every example is a single known operation applied once).
- **One shared model over every included operation, never one model per
  operation.** This is the entire point of the correction (the addendum's
  "Do not satisfy [the shared-core rule] by training separate models"):
  `run_shared_core_gate` constructs exactly one `DecoderOnlyTransformer` per
  `(config, seed)` and trains it on `build_mixed_operation_generator`'s
  mixed stream, which samples uniformly across `config.operation_names`
  every step.
- **All eight `KNOWN_OPERATION_NAMES` by default, not just the four
  ADR-0017-safe ones.** Unlike Task A1-006's `DETERMINISTIC_OPERATION_NAMES`
  restriction, `SELECT`/`COUNT`/`SHIFT`/`BIND`'s previously hidden
  parameters are now part of the model-visible task segment
  (`TaskSpec.arguments`), so excluding them here would understate what the
  correction is meant to prove. `SharedCoreGateConfig.operation_names`
  defaults to `KNOWN_OPERATION_NAMES` (mirroring `build_mixed_operation_
  generator`'s own default) for exactly this reason.
- **`include_task_spec` toggles the one variable this task's negative
  control isolates.** `SharedCoreGateConfig.include_task_spec=True` (the
  primary, "explicit-task" variant) renders `[BOS] [TASK] op arg... [/TASK]
  input... [SEP] target... [EOS]`; `include_task_spec=False` (the negative
  control) renders the exact same `[BOS] input... [SEP] target... [EOS]`
  shape A1-006 always used, over the identical mixed-operation stream and
  identical `apc.core.tokens.SharedCoreTokens` vocabulary/model
  architecture -- so a measured gap between the two variants is attributable
  to whether the task segment is present, not to any other confound
  (different tokens, different vocab size / parameter count, different
  operation pool, different seed). `run_shared_core_gate_h1b` runs both
  variants across the same seeds and reports the gap directly; per
  `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C004's acceptance, the
  negative control is expected to reproduce ADR-0017/ADR-0020's
  non-identifiability ceiling (materially below 0.95), demonstrating the
  explicit task segment -- not merely "one shared model, mixed data,
  online generation" -- is what fixes it.
- **Online generation, symbol permutation left off by default.** Same
  reasoning as Task A1-006 (see that module's docstring and `docs/
  DECISIONS.md` ADR-0018/ADR-0019): `build_mixed_operation_generator`
  already draws fresh content every step, and permutation makes
  value/order-dependent operations (`NEGATE`, `COMPARE`, `ACCUMULATE`)
  provably unrecoverable on held-out content regardless of training
  duration. `permute_symbols` stays available (`SharedCoreGateConfig.
  permute_symbols`) for a future secondary variant, not exercised by
  default here.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX, collate_batch
from apc.core.generation import evaluate_exact_match
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.core.train import resolve_device
from apc.environments.generator import Example, build_mixed_operation_generator
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.environments.task_spec import default_argument_value_span, num_registered_operations
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "OVERALL_EXACT_MATCH_THRESHOLD",
    "PER_OPERATION_EXACT_MATCH_THRESHOLD",
    "MATERIAL_UNDERPERFORMANCE_MARGIN",
    "MIN_ANY_SEED_OPERATION_THRESHOLD",
    "MIN_GATE_SEEDS",
    "SharedCoreGateTrainConfig",
    "SharedCoreGateConfig",
    "shared_core_gate_config_from_dict",
    "SharedCoreGateReport",
    "SharedCoreGateMultiSeedReport",
    "SharedCoreGateH1bReport",
    "run_shared_core_gate",
    "run_shared_core_gate_multi_seed",
    "run_shared_core_gate_h1b",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
OVERALL_EXACT_MATCH_THRESHOLD = 0.95
PER_OPERATION_EXACT_MATCH_THRESHOLD = 0.90
MIN_GATE_SEEDS = 5  # docs/EXPERIMENT_PLAN_PHASE_A1.md section 7: ">=5 seeds" for a gate claim.

# ADR-0020 measured a ~0.25-0.35 unseen exact-match ceiling for a 4-operation
# pool with no explicit task signal (mutual non-identifiability among
# same-output-length operations); ADR-0017 established that SELECT/COUNT/
# SHIFT/BIND's hidden parameter makes those four structurally unrecoverable
# without it. A negative control over all eight KNOWN_OPERATION_NAMES with
# include_task_spec=False is expected to fall well below both that ceiling
# and the 0.95 gate. 0.20 is a predeclared, generous floor for "materially" --
# large enough that ordinary seed-to-seed run variance cannot produce it by
# accident, small enough that it is satisfied even if the negative control
# does better than ADR-0020's pooled ceiling once BIND/COUNT/SELECT/SHIFT's
# stronger unlearnability drags its mean down further.
MATERIAL_UNDERPERFORMANCE_MARGIN = 0.20

# docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md section 2's H1b gate: "no
# operation below 0.85 in any seed without explicit investigation" -- a
# per-(seed, operation) outlier floor, distinct from (and stricter per-run
# than) the cross-seed PER_OPERATION_EXACT_MATCH_THRESHOLD mean. Violations
# are surfaced in SharedCoreGateMultiSeedReport.low_outlier_seed_operations
# rather than folded into `passed`: the "without explicit investigation"
# wording makes this a flag that must be investigated and documented, not an
# automatic gate failure by itself.
MIN_ANY_SEED_OPERATION_THRESHOLD = 0.85


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 48,
        "dropout": 0.0,
    }


@dataclass(frozen=True)
class SharedCoreGateTrainConfig:
    """Explicit, serializable optimization budget for one seed/variant run."""

    steps: int = 20000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 1000
    progress_eval_examples: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}")
        if self.progress_eval_examples < 1:
            raise ValueError(
                f"progress_eval_examples must be >= 1, got {self.progress_eval_examples}"
            )


@dataclass(frozen=True)
class SharedCoreGateConfig:
    """Explicit, serializable configuration for one seed/variant gate run.

    `max_depth` is deliberately not a field, same as `apc.evaluation.
    stable_core_generalization.StableCoreGateConfig` -- the runner always
    uses `build_mixed_operation_generator`'s pinned `max_depth=1`.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = KNOWN_OPERATION_NAMES
    permute_symbols: bool = False
    include_task_spec: bool = True
    model: dict[str, Any] = field(default_factory=_default_model_config)
    train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    num_unseen_eval_examples: int = 2048
    min_examples_per_operation: int = 32

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.num_unseen_eval_examples < 1:
            raise ValueError(
                f"num_unseen_eval_examples must be >= 1, got {self.num_unseen_eval_examples}"
            )
        if self.min_examples_per_operation < 1:
            raise ValueError(
                f"min_examples_per_operation must be >= 1, got {self.min_examples_per_operation}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def shared_core_gate_config_from_dict(raw: dict[str, Any]) -> SharedCoreGateConfig:
    """Parse a `configs/phase_a1_shared_core_gate.yaml`-shaped dict, matching
    `apc.evaluation.stable_core_generalization.stable_core_gate_config_from_dict`'s
    convention of filling in defaults for whatever the file omits."""
    defaults = SharedCoreGateConfig()
    train_raw = raw.get("train")
    train = dataclasses.replace(defaults.train, **train_raw) if train_raw else defaults.train
    return SharedCoreGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        permute_symbols=raw.get("permute_symbols", defaults.permute_symbols),
        include_task_spec=raw.get("include_task_spec", defaults.include_task_spec),
        model=dict(raw.get("model", defaults.model)),
        train=train,
        num_unseen_eval_examples=raw.get(
            "num_unseen_eval_examples", defaults.num_unseen_eval_examples
        ),
        min_examples_per_operation=raw.get(
            "min_examples_per_operation", defaults.min_examples_per_operation
        ),
    )


def _build_tokens(config: SharedCoreGateConfig) -> SharedCoreTokens:
    """`SharedCoreTokens` sized from the full operation registry (not merely
    `config.operation_names`), so the model vocabulary/parameter count is
    identical between the explicit-task and negative-control variants of the
    same base config -- `include_task_spec=False` simply never emits the
    task-segment tokens this reserves, rather than using a smaller/different
    vocabulary (see module docstring)."""
    return build_shared_core_tokens(
        config.vocab_size,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(config.vocab_size, config.sequence_length_range),
    )


def _operation_of(example: Example) -> str:
    """The single operation name behind a depth-1 mixed-operation-generator
    example. `task_spec` is attached by `TaskGenerator` unconditionally (Task
    A1-C001), independent of whether that generator's stream is fed to the
    model (`include_task_spec`), so this is always available for grouping
    eval results by operation -- including for the negative-control variant,
    which never shows this to the model but still needs it for reporting."""
    assert example.task_spec is not None
    (operation,) = example.task_spec.operation_sequence
    return operation


def _per_operation_exact_match(
    examples: Sequence[Example],
    predictions: Sequence[tuple[int, ...]],
    operation_names: Sequence[str],
    min_examples_per_operation: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Group one `evaluate_exact_match` call's `(examples, predictions)` by
    each example's operation and compute per-operation exact match.

    Raises if any `operation_names` entry has fewer than
    `min_examples_per_operation` examples in `examples` -- a silent
    near-zero-sample per-operation rate would be statistically meaningless
    rather than a genuine failure, so this surfaces the config problem
    (`num_unseen_eval_examples` too small for the operation pool size)
    instead of reporting it as one.
    """
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

    per_operation_exact_match = {
        name: matches[name] / totals[name] for name in operation_names
    }
    return per_operation_exact_match, totals


@dataclass(frozen=True)
class SharedCoreGateReport:
    """Everything observed while training and evaluating one seed/variant."""

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
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_shared_core_gate(
    config: SharedCoreGateConfig, metrics_path: str | Path | None = None
) -> SharedCoreGateReport:
    """Train one fresh `DecoderOnlyTransformer` from scratch on
    `build_mixed_operation_generator(operation_names=config.operation_names,
    ...)`'s online-generated mixed stream, then evaluate overall and
    per-operation exact match on a large, independently-drawn `"test"`-split
    batch. No primitive bank, router, plastic workspace, or consolidation is
    constructed anywhere in this function.

    `config.include_task_spec` selects the explicit-task (`True`) or
    negative-control (`False`) variant; both otherwise share every other
    field, including the `SharedCoreTokens` vocabulary/model architecture
    (see `_build_tokens`), so the two variants differ only in whether the
    task segment is part of the model input.

    If `metrics_path` is given, periodic `{"step", "loss",
    "progress_overall_exact_match"}` lines are appended to it as training
    proceeds, matching `apc.evaluation.stable_core_generalization.
    run_stable_core_gate`'s convention.
    """
    start = time.perf_counter()
    set_seed(config.seed)
    device = resolve_device(config.train.device)

    tokens = _build_tokens(config)
    generator = build_mixed_operation_generator(
        seed=config.seed,
        operation_names=config.operation_names,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        permute_symbols=config.permute_symbols,
    )

    model_config = TransformerConfig(vocab_size=tokens.model_vocab_size, **config.model)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    # Re-anchor the shared global RNG stream after model construction, per
    # ADR-0016 -- see apc.evaluation.stable_core_generalization.
    # run_stable_core_gate for the same pattern.
    set_seed(config.seed)

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    try:
        model.train()
        for step in range(config.train.steps):
            examples = generator.generate_online(config.train.batch_size, step=step, split="train")
            batch = collate_batch(
                examples, tokens, device=device, include_task_spec=config.include_task_spec
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            final_loss = float(loss.item())

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.train.eval_every == 0 or step_number == config.train.steps
            ):
                progress_examples = generator.generate_online(
                    config.train.progress_eval_examples, step=step, split="val"
                )
                progress_exact_match, _ = evaluate_exact_match(
                    model,
                    progress_examples,
                    tokens,
                    device,
                    include_task_spec=config.include_task_spec,
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "loss": final_loss,
                            "progress_overall_exact_match": progress_exact_match,
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
                model.train()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    unseen_examples = generator.generate_online(
        config.num_unseen_eval_examples, step=0, split="test"
    )
    overall_exact_match, predictions = evaluate_exact_match(
        model, unseen_examples, tokens, device, include_task_spec=config.include_task_spec
    )
    per_operation_exact_match, per_operation_counts = _per_operation_exact_match(
        unseen_examples, predictions, config.operation_names, config.min_examples_per_operation
    )

    return SharedCoreGateReport(
        config=config,
        steps_trained=config.train.steps,
        examples_seen=config.train.steps * config.train.batch_size,
        final_train_loss=final_loss,
        overall_exact_match=overall_exact_match,
        per_operation_exact_match=per_operation_exact_match,
        per_operation_eval_counts=per_operation_counts,
        num_unseen_eval_examples=len(unseen_examples),
        param_count=model.num_parameters(),
        trainable_param_count=model.num_parameters(trainable_only=True),
        wall_clock_seconds=time.perf_counter() - start,
        device=str(device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    """`(mean, stdev, min, max)` over a non-empty sequence. `stdev` is the
    sample standard deviation (0.0 for a single value)."""
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class SharedCoreGateMultiSeedReport:
    """One variant's (explicit-task or negative-control) verdict aggregated
    across seeds: overall unseen exact match, plus a per-operation breakdown,
    each against its own predeclared threshold (`docs/
    CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C004: "overall mean >=0.95,
    every operation mean >=0.90")."""

    seeds: tuple[int, ...]
    per_seed: tuple[SharedCoreGateReport, ...]
    operation_names: tuple[str, ...]
    mean_overall_exact_match: float
    stdev_overall_exact_match: float
    min_overall_exact_match: float
    max_overall_exact_match: float
    overall_threshold: float
    per_operation_mean_exact_match: dict[str, float]
    per_operation_stdev_exact_match: dict[str, float]
    per_operation_min_exact_match: dict[str, float]
    per_operation_max_exact_match: dict[str, float]
    per_operation_threshold: float
    low_outlier_threshold: float
    low_outlier_seed_operations: tuple[dict[str, Any], ...]
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
            "overall_threshold": self.overall_threshold,
            "per_operation_mean_exact_match": self.per_operation_mean_exact_match,
            "per_operation_stdev_exact_match": self.per_operation_stdev_exact_match,
            "per_operation_min_exact_match": self.per_operation_min_exact_match,
            "per_operation_max_exact_match": self.per_operation_max_exact_match,
            "per_operation_threshold": self.per_operation_threshold,
            "low_outlier_threshold": self.low_outlier_threshold,
            "low_outlier_seed_operations": list(self.low_outlier_seed_operations),
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def _aggregate_multi_seed_report(
    per_seed: Sequence[SharedCoreGateReport],
    *,
    operation_names: Sequence[str],
    overall_threshold: float,
    per_operation_threshold: float,
    low_outlier_threshold: float = MIN_ANY_SEED_OPERATION_THRESHOLD,
) -> SharedCoreGateMultiSeedReport:
    if not per_seed:
        raise ValueError("per_seed must be non-empty")
    overall_values = [report.overall_exact_match for report in per_seed]
    mean_value, stdev_value, min_value, max_value = _summarize(overall_values)

    per_operation_mean: dict[str, float] = {}
    per_operation_stdev: dict[str, float] = {}
    per_operation_min: dict[str, float] = {}
    per_operation_max: dict[str, float] = {}
    for name in operation_names:
        op_values = [report.per_operation_exact_match[name] for report in per_seed]
        op_mean, op_stdev, op_min, op_max = _summarize(op_values)
        per_operation_mean[name] = op_mean
        per_operation_stdev[name] = op_stdev
        per_operation_min[name] = op_min
        per_operation_max[name] = op_max

    # docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md section 2: "no operation
    # below 0.85 in any seed without explicit investigation" -- a per-run
    # outlier floor, checked directly against each report's own per-seed
    # per-operation value (not the cross-seed mean above).
    low_outliers = tuple(
        {"seed": report.config.seed, "operation": name, "exact_match": value}
        for report in per_seed
        for name, value in report.per_operation_exact_match.items()
        if value < low_outlier_threshold
    )

    passed = mean_value >= overall_threshold and all(
        per_operation_mean[name] >= per_operation_threshold for name in operation_names
    )
    seeds = tuple(report.config.seed for report in per_seed)
    return SharedCoreGateMultiSeedReport(
        seeds=seeds,
        per_seed=tuple(per_seed),
        operation_names=tuple(operation_names),
        mean_overall_exact_match=mean_value,
        stdev_overall_exact_match=stdev_value,
        min_overall_exact_match=min_value,
        max_overall_exact_match=max_value,
        overall_threshold=overall_threshold,
        per_operation_mean_exact_match=per_operation_mean,
        per_operation_stdev_exact_match=per_operation_stdev,
        per_operation_min_exact_match=per_operation_min,
        per_operation_max_exact_match=per_operation_max,
        per_operation_threshold=per_operation_threshold,
        low_outlier_threshold=low_outlier_threshold,
        low_outlier_seed_operations=low_outliers,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )


def run_shared_core_gate_multi_seed(
    base_config: SharedCoreGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    overall_threshold: float = OVERALL_EXACT_MATCH_THRESHOLD,
    per_operation_threshold: float = PER_OPERATION_EXACT_MATCH_THRESHOLD,
) -> SharedCoreGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed, every other field -- including `include_task_spec` -- shared)
    and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching `apc.
    evaluation.stable_core_generalization.run_stable_core_gate_multi_seed`.
    """
    per_seed = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        metrics_path = run_dir_path / f"seed_{seed}" / "metrics.jsonl" if run_dir_path else None
        per_seed.append(run_shared_core_gate(config, metrics_path=metrics_path))
    return _aggregate_multi_seed_report(
        per_seed,
        operation_names=base_config.operation_names,
        overall_threshold=overall_threshold,
        per_operation_threshold=per_operation_threshold,
    )


@dataclass(frozen=True)
class SharedCoreGateH1bReport:
    """The H1b gate verdict: the explicit-task variant's own pass/fail
    (overall + per-operation thresholds) combined with whether the
    negative-control variant materially underperforms it -- `docs/
    CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C004's full acceptance in one
    record."""

    explicit: SharedCoreGateMultiSeedReport
    negative_control: SharedCoreGateMultiSeedReport
    overall_exact_match_gap: float
    material_underperformance_margin: float
    negative_control_materially_underperforms: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "explicit": self.explicit.to_dict(),
            "negative_control": self.negative_control.to_dict(),
            "overall_exact_match_gap": self.overall_exact_match_gap,
            "material_underperformance_margin": self.material_underperformance_margin,
            "negative_control_materially_underperforms": (
                self.negative_control_materially_underperforms
            ),
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_shared_core_gate_h1b(
    base_config: SharedCoreGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    overall_threshold: float = OVERALL_EXACT_MATCH_THRESHOLD,
    per_operation_threshold: float = PER_OPERATION_EXACT_MATCH_THRESHOLD,
    material_underperformance_margin: float = MATERIAL_UNDERPERFORMANCE_MARGIN,
) -> SharedCoreGateH1bReport:
    """Run both the explicit-task (`include_task_spec=True`) and
    negative-control (`include_task_spec=False`) variants of `base_config`
    across the same `seeds`, and report the full H1b verdict.

    `base_config.include_task_spec` is overridden to `True`/`False` for the
    two variants respectively -- whatever it was set to in `base_config` is
    ignored, since both variants are always run together.
    """
    run_dir_path = Path(run_dir) if run_dir is not None else None
    explicit_config = dataclasses.replace(base_config, include_task_spec=True)
    control_config = dataclasses.replace(base_config, include_task_spec=False)

    explicit = run_shared_core_gate_multi_seed(
        explicit_config,
        seeds=seeds,
        run_dir=(run_dir_path / "explicit" if run_dir_path else None),
        overall_threshold=overall_threshold,
        per_operation_threshold=per_operation_threshold,
    )
    negative_control = run_shared_core_gate_multi_seed(
        control_config,
        seeds=seeds,
        run_dir=(run_dir_path / "negative_control" if run_dir_path else None),
        overall_threshold=overall_threshold,
        per_operation_threshold=per_operation_threshold,
    )

    gap = explicit.mean_overall_exact_match - negative_control.mean_overall_exact_match
    materially_underperforms = gap >= material_underperformance_margin
    # `passed` is a pure threshold/margin comparison, deliberately not gated
    # on `meets_seed_policy` -- same convention as `apc.evaluation.
    # stable_core_generalization.StableCoreGateMultiSeedReport`: a quick
    # fewer-than-5-seed dev check stays usable without an exception or a
    # forced failure blocking iteration, and `meets_seed_policy` reports the
    # honest caveat as its own field instead.
    passed = explicit.passed and materially_underperforms

    return SharedCoreGateH1bReport(
        explicit=explicit,
        negative_control=negative_control,
        overall_exact_match_gap=gap,
        material_underperformance_margin=material_underperformance_margin,
        negative_control_materially_underperforms=materially_underperforms,
        passed=passed,
        meets_seed_policy=explicit.meets_seed_policy and negative_control.meets_seed_policy,
    )
