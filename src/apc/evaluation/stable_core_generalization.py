"""Stable Core systematic-generalization gate (Phase A.1 Task A1-006, STOP GATE).

`docs/EXPERIMENT_PLAN_PHASE_A1.md` H1: "A Stable Core trained on procedurally
generated known operations executes those operations on unseen content,"
gated at mean unseen-content exact match >= 0.95 across >= 5 seeds. This is
the first Phase A.1 milestone that actually trains anything; every earlier
Phase A.1 task (A1-001..A1-005) built the online generator, top-k execution,
symbol permutation, and task/content split without yet testing whether they
fix Phase A's negative finding (`docs/DECISIONS.md` ADR-0006/0009/0013/0014:
the fixed dense core never generalized past ~0.02 exact match to unseen
token content, across every model size and fixed-dataset size tried).

Deliberately minimal, per `docs/AGENTS_PHASE_A1_ADDENDUM.md`'s "one
mechanism per task" and oracle-ladder rules:

- **K only, no plastic/consolidation/routing.** This module trains a plain
  `apc.core.model.DecoderOnlyTransformer` directly with cross-entropy --
  no `apc.primitives`, `apc.plastic`, `apc.consolidation`, or `apc.meta`
  import anywhere in this file. Composition (`C`), novel operations (`N`),
  and recurrence (`R`) are later oracle-ladder rungs (A1-007+); mixing them
  in here would violate "one mechanism per task" and confound a Stable Core
  representation/training failure with a routing or consolidation failure.
- **Only the four deterministic known operations, one at a time.**
  `KNOWN_OPERATION_NAMES` also includes `SELECT`/`COUNT`/`SHIFT`/`BIND`,
  whose target depends on a hidden per-instance parameter that is never
  part of the presented input (see `apc.environments.operations.
  DETERMINISTIC_OPERATION_NAMES` and `docs/DECISIONS.md` ADR-0017): no
  learner can recover those parameters from input alone, so mixing them in
  would make >=0.95 exact match structurally unreachable regardless of
  representation/training quality. Even among the four *parameter-free*
  operations, pooling several into one `TaskGenerator` (one shared model
  trained on a mix of operations) reintroduces the same problem one level
  up: which operation applies to a given example is itself chosen
  uniformly at random and never revealed to the model, so operations that
  happen to share an output-length signature (`COPY`/`NEGATE`/`ACCUMULATE`
  all preserve length; only `COMPARE` differs) are mutually
  non-identifiable from input alone -- measured empirically at ~0.25-0.35
  exact match ceiling for a pooled 4-operation run regardless of training
  budget (`docs/DECISIONS.md` ADR-0020). `run_stable_core_gate_grid`
  therefore trains and evaluates one *single-operation* `TaskGenerator` per
  operation (never a mixed pool), and aggregates across an
  operation x seed grid.
- **Composition depth is fixed at 1, not exposed as a config field.** K is
  "known primitive, unseen content" (`docs/EXPERIMENT_PLAN_PHASE_A1.md`
  section 3) -- a single operation applied once. `TaskGenerator` is always
  constructed here with `max_depth=1` so its `known` pool cannot silently
  include multi-operation chains; composition generalization is A1-008's
  job, not this gate's.
- **Online generation prevents finite-set memorization; symbol permutation
  is opt-in, not default, because it makes value-dependent operations
  provably unrecoverable on held-out content.** Training always draws a
  fresh batch every step via `TaskGenerator.generate_online` (never a
  fixed, reusable dataset -- A1-003), which alone already removes the
  finite-dataset shortcut symbol permutation (A1-004) was designed to
  close. `permute_symbols` therefore defaults to `False` here. Enabling it
  (`permute_symbols=True`) is still supported and meaningful for
  *value-blind* operations (`COPY`; see `apc.environments.generator`'s
  module docstring for the shared-per-batch permutation A1-004 now uses),
  but for value/order-dependent operations (`NEGATE`, `COMPARE`,
  `ACCUMULATE`) it makes >=0.95 exact match structurally unreachable on
  genuinely unseen content regardless of training duration: a held-out
  example's specific token relabeling is never revealed to the model and
  cannot be recovered from that example's content alone (measured: 30000
  steps of single-operation `NEGATE` training under permutation reaches
  loss 0.77, still falling, but 0.0 unseen exact match; the identical
  config with `permute_symbols=False` reaches 1.0 in 6000 steps). See
  `docs/DECISIONS.md` ADR-0019. `configs/phase_a1_stable_core_gate.yaml`
  (primary, all four operations) and `configs/
  phase_a1_stable_core_gate_permuted_copy_only.yaml` (secondary, `COPY`
  only, `permute_symbols=True`) report both variants. Final evaluation
  always reads a large batch from the `"test"` split at its own
  independent `(seed, step, split)` label, which draws from a
  `random.Random` stream wholly disjoint from every training-step draw
  (`apc.environments.generator.TaskGenerator._derive_seed`), so evaluation
  content is never one of the batches gradients were computed on.
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
from apc.core.tokens import build_special_tokens
from apc.core.train import resolve_device
from apc.environments.generator import TaskGenerator
from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "UNSEEN_EXACT_MATCH_THRESHOLD",
    "MIN_GATE_SEEDS",
    "StableCoreGateTrainConfig",
    "StableCoreGateConfig",
    "stable_core_gate_config_from_dict",
    "StableCoreGateReport",
    "StableCoreGateMultiSeedReport",
    "run_stable_core_gate",
    "run_stable_core_gate_multi_seed",
    "StableCoreGateOperationReport",
    "StableCoreGateGridReport",
    "run_stable_core_gate_grid",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
UNSEEN_EXACT_MATCH_THRESHOLD = 0.95
MIN_GATE_SEEDS = 5  # docs/EXPERIMENT_PLAN_PHASE_A1.md section 7: ">=5 seeds" for a gate claim.


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 32,
        "dropout": 0.0,
    }


@dataclass(frozen=True)
class StableCoreGateTrainConfig:
    """Explicit, serializable optimization budget for one seed's run."""

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
class StableCoreGateConfig:
    """Explicit, serializable configuration for one seed's gate run.

    `max_depth` is deliberately not a field -- see module docstring; the
    runner always constructs its `TaskGenerator` with `max_depth=1`.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = DETERMINISTIC_OPERATION_NAMES
    permute_symbols: bool = False
    model: dict[str, Any] = field(default_factory=_default_model_config)
    train: StableCoreGateTrainConfig = field(default_factory=StableCoreGateTrainConfig)
    num_unseen_eval_examples: int = 512

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.num_unseen_eval_examples < 1:
            raise ValueError(
                f"num_unseen_eval_examples must be >= 1, got {self.num_unseen_eval_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def stable_core_gate_config_from_dict(raw: dict[str, Any]) -> StableCoreGateConfig:
    """Parse a `configs/phase_a1_stable_core_gate.yaml`-shaped dict, matching
    `apc.evaluation.sequential_benchmark.sequential_config_from_dict`'s
    convention of filling in defaults for whatever the file omits."""
    defaults = StableCoreGateConfig()
    train_raw = raw.get("train")
    train = dataclasses.replace(defaults.train, **train_raw) if train_raw else defaults.train
    return StableCoreGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        permute_symbols=raw.get("permute_symbols", defaults.permute_symbols),
        model=dict(raw.get("model", defaults.model)),
        train=train,
        num_unseen_eval_examples=raw.get(
            "num_unseen_eval_examples", defaults.num_unseen_eval_examples
        ),
    )


@dataclass(frozen=True)
class StableCoreGateReport:
    """Everything observed while training and evaluating one seed."""

    config: StableCoreGateConfig
    steps_trained: int
    examples_seen: int
    final_train_loss: float
    unseen_exact_match: float
    num_unseen_eval_examples: int
    param_count: int
    trainable_param_count: int
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


@dataclass(frozen=True)
class StableCoreGateMultiSeedReport:
    """The H1 gate verdict: mean unseen-content exact match across seeds."""

    seeds: tuple[int, ...]
    per_seed: tuple[StableCoreGateReport, ...]
    mean_unseen_exact_match: float
    stdev_unseen_exact_match: float
    min_unseen_exact_match: float
    max_unseen_exact_match: float
    threshold: float
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "mean_unseen_exact_match": self.mean_unseen_exact_match,
            "stdev_unseen_exact_match": self.stdev_unseen_exact_match,
            "min_unseen_exact_match": self.min_unseen_exact_match,
            "max_unseen_exact_match": self.max_unseen_exact_match,
            "threshold": self.threshold,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_stable_core_gate(
    config: StableCoreGateConfig, metrics_path: str | Path | None = None
) -> StableCoreGateReport:
    """Train a fresh `DecoderOnlyTransformer` from scratch on
    `config.operation_names` with online-generated, symbol-permuted data,
    then evaluate exact match on a large, independently-drawn `"test"`-split
    batch. No primitive bank, router, plastic workspace, or consolidation is
    constructed anywhere in this function.

    If `metrics_path` is given, periodic `{"step", "loss", "progress_exact_match"}`
    lines are appended to it as training proceeds (same convention as
    `apc.core.train.run_smoke_training`'s `metrics.jsonl`), for diagnosing a
    failed gate without rerunning it (`docs/AGENTS_PHASE_A1_ADDENDUM.md`:
    "investigate only that mechanism").
    """
    start = time.perf_counter()
    set_seed(config.seed)
    device = resolve_device(config.train.device)

    specials = build_special_tokens(config.vocab_size)
    generator = TaskGenerator(
        seed=config.seed,
        operation_names=config.operation_names,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        max_depth=1,
        permute_symbols=config.permute_symbols,
    )

    model_config = TransformerConfig(vocab_size=specials.model_vocab_size, **config.model)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    # Re-anchor the shared global RNG stream now that model construction
    # (the only setup-phase piece here that consumes it) has finished, per
    # the ADR-0016 pattern -- so the training loop's data/dropout draws
    # never depend on incidental model parameter counts.
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
            batch = collate_batch(examples, specials, device=device)

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
                    model, progress_examples, specials, device
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "loss": final_loss,
                            "progress_exact_match": progress_exact_match,
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
    unseen_exact_match, _ = evaluate_exact_match(model, unseen_examples, specials, device)

    return StableCoreGateReport(
        config=config,
        steps_trained=config.train.steps,
        examples_seen=config.train.steps * config.train.batch_size,
        final_train_loss=final_loss,
        unseen_exact_match=unseen_exact_match,
        num_unseen_eval_examples=len(unseen_examples),
        param_count=model.num_parameters(),
        trainable_param_count=model.num_parameters(trainable_only=True),
        wall_clock_seconds=time.perf_counter() - start,
        device=str(device),
    )


def _summarize_exact_match(values: Sequence[float]) -> tuple[float, float, float, float]:
    """`(mean, stdev, min, max)` over a non-empty sequence of exact-match
    values. `stdev` is the sample standard deviation (0.0 for a single
    value), shared by both the per-operation and grid-level aggregations
    below."""
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


def _aggregate_multi_seed_report(
    per_seed: Sequence[StableCoreGateReport], *, threshold: float
) -> StableCoreGateMultiSeedReport:
    if not per_seed:
        raise ValueError("per_seed must be non-empty")
    values = [report.unseen_exact_match for report in per_seed]
    mean_value, stdev_value, min_value, max_value = _summarize_exact_match(values)
    seeds = tuple(report.config.seed for report in per_seed)
    return StableCoreGateMultiSeedReport(
        seeds=seeds,
        per_seed=tuple(per_seed),
        mean_unseen_exact_match=mean_value,
        stdev_unseen_exact_match=stdev_value,
        min_unseen_exact_match=min_value,
        max_unseen_exact_match=max_value,
        threshold=threshold,
        passed=mean_value >= threshold,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )


def run_stable_core_gate_multi_seed(
    base_config: StableCoreGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    threshold: float = UNSEEN_EXACT_MATCH_THRESHOLD,
) -> StableCoreGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` is
    overridden per seed) and report the H1 verdict: mean unseen-content
    exact match across seeds against `threshold`.

    Does not raise if `len(seeds) < MIN_GATE_SEEDS` -- `meets_seed_policy` on
    the returned report says so honestly instead, so a quick fewer-seed dev
    check stays usable without an exception blocking iteration; the CLI
    entry point defaults to `DEFAULT_SEEDS` (5), matching
    `docs/EXPERIMENT_PLAN_PHASE_A1.md` section 7's gate policy.
    """
    per_seed = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        metrics_path = run_dir_path / f"seed_{seed}" / "metrics.jsonl" if run_dir_path else None
        per_seed.append(run_stable_core_gate(config, metrics_path=metrics_path))
    return _aggregate_multi_seed_report(per_seed, threshold=threshold)


@dataclass(frozen=True)
class StableCoreGateOperationReport:
    """One operation's multi-seed gate result within a grid run."""

    operation_name: str
    multi_seed: StableCoreGateMultiSeedReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "multi_seed": self.multi_seed.to_dict(),
        }


@dataclass(frozen=True)
class StableCoreGateGridReport:
    """The H1 gate verdict aggregated across an operation x seed grid.

    `mean_unseen_exact_match` (and stdev/min/max) are computed over every
    individual `(operation, seed)` run's `unseen_exact_match` -- i.e. one
    data point per run in the grid, not one point per operation -- matching
    `docs/CODEX_TASKS_PHASE_A1.md` A1-006's "mean unseen-content exact
    match >= 0.95 with all seeds reported" read across the whole grid.
    `per_operation` carries each operation's own breakdown for diagnosis.
    """

    operation_names: tuple[str, ...]
    seeds: tuple[int, ...]
    per_operation: tuple[StableCoreGateOperationReport, ...]
    mean_unseen_exact_match: float
    stdev_unseen_exact_match: float
    min_unseen_exact_match: float
    max_unseen_exact_match: float
    threshold: float
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_names": list(self.operation_names),
            "seeds": list(self.seeds),
            "per_operation": [report.to_dict() for report in self.per_operation],
            "mean_unseen_exact_match": self.mean_unseen_exact_match,
            "stdev_unseen_exact_match": self.stdev_unseen_exact_match,
            "min_unseen_exact_match": self.min_unseen_exact_match,
            "max_unseen_exact_match": self.max_unseen_exact_match,
            "threshold": self.threshold,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_stable_core_gate_grid(
    base_config: StableCoreGateConfig,
    *,
    operation_names: Sequence[str] = DETERMINISTIC_OPERATION_NAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    run_dir: str | Path | None = None,
    threshold: float = UNSEEN_EXACT_MATCH_THRESHOLD,
) -> StableCoreGateGridReport:
    """Run the gate once per operation in `operation_names` (each its own
    single-operation `TaskGenerator`, never a mixed pool -- see module
    docstring's "Only the four deterministic known operations, one at a
    time" and `docs/DECISIONS.md` ADR-0020) across `seeds`, and report the
    H1 verdict over the full operation x seed grid.

    `base_config.operation_names` is overridden per operation (one-tuple);
    only `base_config`'s other fields (vocab size, sequence length range,
    permute_symbols, model, train budget, ...) are shared across the grid.
    """
    if not operation_names:
        raise ValueError("operation_names must be non-empty")
    run_dir_path = Path(run_dir) if run_dir is not None else None
    per_operation = []
    for operation_name in operation_names:
        op_config = dataclasses.replace(base_config, operation_names=(operation_name,))
        op_run_dir = run_dir_path / operation_name if run_dir_path else None
        multi_seed = run_stable_core_gate_multi_seed(
            op_config, seeds=seeds, run_dir=op_run_dir, threshold=threshold
        )
        per_operation.append(
            StableCoreGateOperationReport(operation_name=operation_name, multi_seed=multi_seed)
        )

    all_values = [
        report.unseen_exact_match
        for operation_report in per_operation
        for report in operation_report.multi_seed.per_seed
    ]
    mean_value, stdev_value, min_value, max_value = _summarize_exact_match(all_values)
    return StableCoreGateGridReport(
        operation_names=tuple(operation_names),
        seeds=tuple(seeds),
        per_operation=tuple(per_operation),
        mean_unseen_exact_match=mean_value,
        stdev_unseen_exact_match=stdev_value,
        min_unseen_exact_match=min_value,
        max_unseen_exact_match=max_value,
        threshold=threshold,
        passed=mean_value >= threshold,
        meets_seed_policy=all(
            operation_report.multi_seed.meets_seed_policy for operation_report in per_operation
        ),
    )
