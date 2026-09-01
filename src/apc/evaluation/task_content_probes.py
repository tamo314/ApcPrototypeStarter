"""Task/content representation probes (Phase A.1 Correction Task A1-C005,
STOP GATE).

`docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C005 / `docs/
EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md` section 3 (H1c): Task A1-C004
(`apc.evaluation.shared_core_generalization`, ADR-0021) proved that *one
shared* Stable Core can condition on an explicit task specification and
reach >=0.95 unseen-content exact match across all eight known operations.
That is a claim about the model's end-to-end input/output behavior. It is
not yet a claim about *where inside the model* task information lives --
`docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`'s "Task/content probe rule"
is explicit that `apc.core.model.DecoderOnlyTransformer.encode_split`
returning two differently-named tensors does not by itself establish that
`z_task` (`EncodedState.task_state`) is a useful summary of task identity/
arguments, or that `h_content` (`EncodedState.content_state`) is a useful
summary of content. This module is the diagnostic that checks that
directly, by freezing a shared-core model trained the same way A1-C004
gates it and fitting small linear probes on its frozen `encode_split`
output.

## Why probe *these* two sequence positions

`ADR-0015` records that `task_state = task_head(content_state)`
(`apc.core.model.DecoderOnlyTransformer.encode_split`) is a parameter-free,
per-position `LayerNorm`: at any single sequence position, `task_state` and
`content_state` carry the *same* information, merely renormalized. The
task/content split this module actually probes is therefore not "which
tensor field", it is "which sequence position" -- exploiting the model's
own causal attention mask:

- the prompt is rendered as `[BOS] [TASK_START] op arg... [TASK_END]
  input... [SEP]` (`apc.core.data.build_prompt_tokens`,
  `include_task_spec=True`); at the `[TASK_END]` position, causal
  attention has only ever seen the task segment -- structurally, *no*
  amount of training can leak upcoming content into that position's hidden
  state, because it has not been generated yet when `[TASK_END]` is
  encoded. Reading `task_state` there (Probes T1/T2 below) asks exactly
  "does the model summarize the task segment into something linearly
  decodable by the time it reaches the position where execution begins."
- at the `[SEP]` position, causal attention has seen the task segment *and*
  the full content input. Reading `content_state` there is not meaningful
  for content probing (SEP has no per-position content identity of its
  own); instead Probe C1 reads `content_state` at each individual content
  token's own position (which has itself seen the task segment and every
  token up to and including itself -- exactly what a primitive execution
  step reading `content_state` would see).

This also predicts the two *optional* leakage probes below structurally:
"operation from `content_state` at `[TASK_END]`" is expected to succeed
(same underlying vector as `task_state` there, modulo the parameter-free
norm) and "content from `task_state`" at a content position is expected to
succeed for the same reason -- neither is evidence against a useful
factorization; per `docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`, "the
goal is useful factorization, not perfect disentanglement," and neither
optional probe gates `passed` here.

## Scope discipline

Mirrors `apc.evaluation.shared_core_generalization`'s own scope discipline:
no `apc.primitives`, `apc.plastic`, `apc.consolidation`, or `apc.meta`
import anywhere in this file. `train_shared_core` (Task A1-C004's own
training loop, refactored out for reuse here rather than duplicated) trains
the frozen core this module probes; probing itself never touches the
core's weights (`torch.no_grad()`, and the core is never put back into
`.train()` mode once probing starts).
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

from apc.core.data import collate_batch
from apc.core.generation import evaluate_exact_match
from apc.environments.generator import Example
from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES, KNOWN_OPERATION_NAMES
from apc.environments.task_spec import operation_id
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    TrainedSharedCore,
    shared_core_gate_config_from_dict,
    train_shared_core,
)
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "OPERATION_ID_PROBE_THRESHOLD",
    "ARGUMENT_PROBE_THRESHOLD",
    "CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD",
    "PARAMETERIZED_OPERATION_NAMES",
    "SCALAR_ARGUMENT_OPERATIONS",
    "SET_ARGUMENT_OPERATIONS",
    "ProbeTrainConfig",
    "TaskContentProbeConfig",
    "task_content_probe_config_from_dict",
    "ArgumentProbeResult",
    "TaskContentProbeReport",
    "TaskContentProbeMultiSeedReport",
    "run_task_content_probes",
    "run_task_content_probes_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# docs/CODEX_TASKS_PHASE_A1_CORRECTION.md A1-C005 acceptance.
OPERATION_ID_PROBE_THRESHOLD = 0.95
ARGUMENT_PROBE_THRESHOLD = 0.90

# EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md section 3, Probe C1: "high enough to
# support task execution; predeclare metric per content format." The content
# format here is per-position token identity over a `vocab_size=10` closed
# vocabulary read from a `d_model=192` residual stream with no compression
# bottleneck -- every one of A1-C004's eight operations needs to recover
# exact token identity per position to execute correctly (COPY/NEGATE/
# ACCUMULATE act pointwise; SELECT/COMPARE/COUNT/BIND/SHIFT all need exact
# values at specific positions), so this is predeclared high: a
# content_state that does not near-perfectly retain per-position token
# identity could not support any of them, and A1-C004 already measured the
# same model class solving all eight at >=0.95 overall.
CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD = 0.98

# ADR-0017: SELECT/COUNT/SHIFT/BIND are exactly the operations whose
# `sample_params` returns a non-empty dict, i.e. exactly the complement of
# DETERMINISTIC_OPERATION_NAMES within KNOWN_OPERATION_NAMES.
PARAMETERIZED_OPERATION_NAMES: tuple[str, ...] = tuple(
    name for name in KNOWN_OPERATION_NAMES if name not in DETERMINISTIC_OPERATION_NAMES
)
# SHIFT.amount, COUNT.target, BIND.query_key: each a single non-negative
# integer (apc.core.data._argument_values' "plain int" case). SELECT.indices
# is the one list-valued argument (apc.environments.operations.SelectOp) and
# is probed separately as a set-membership target -- see SET_ARGUMENT_OPERATIONS.
SCALAR_ARGUMENT_OPERATIONS: tuple[str, ...] = tuple(
    name for name in PARAMETERIZED_OPERATION_NAMES if name != "SELECT"
)
SET_ARGUMENT_OPERATIONS: tuple[str, ...] = ("SELECT",)


@dataclass(frozen=True)
class ProbeTrainConfig:
    """Explicit, serializable optimization budget for one linear probe.

    Every probe in this module is a single `nn.Linear` fit by full-batch
    AdamW directly on precomputed frozen features (no re-running the shared
    core once features are extracted) -- "lightweight" per the task's own
    wording, and convex-ish enough (linear model, cross-entropy/BCE loss)
    that a few hundred full-batch steps is expected to reach a stable
    optimum for a `d_model=192` input.
    """

    steps: int = 800
    lr: float = 0.03
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


@dataclass(frozen=True)
class TaskContentProbeConfig:
    """Explicit, serializable configuration for one seed's A1-C005 probe run."""

    seed: int = 0
    shared_core: SharedCoreGateConfig = field(default_factory=SharedCoreGateConfig)
    probe_train: ProbeTrainConfig = field(default_factory=ProbeTrainConfig)
    num_probe_train_examples: int = 4096
    num_probe_eval_examples: int = 2048
    min_examples_per_operation: int = 32
    operation_id_threshold: float = OPERATION_ID_PROBE_THRESHOLD
    argument_threshold: float = ARGUMENT_PROBE_THRESHOLD
    content_threshold: float = CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD

    def __post_init__(self) -> None:
        if not self.shared_core.include_task_spec:
            raise ValueError(
                "TaskContentProbeConfig.shared_core.include_task_spec must be True -- "
                "there is no [TASK_START]..[TASK_END] segment (and therefore no z_task "
                "position to probe) without it"
            )
        if self.num_probe_train_examples < 1:
            raise ValueError(
                f"num_probe_train_examples must be >= 1, got {self.num_probe_train_examples}"
            )
        if self.num_probe_eval_examples < 1:
            raise ValueError(
                f"num_probe_eval_examples must be >= 1, got {self.num_probe_eval_examples}"
            )
        if self.min_examples_per_operation < 1:
            raise ValueError(
                f"min_examples_per_operation must be >= 1, got {self.min_examples_per_operation}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def task_content_probe_config_from_dict(raw: dict[str, Any]) -> TaskContentProbeConfig:
    """Parse a `configs/phase_a1_task_content_probes.yaml`-shaped dict,
    matching `apc.evaluation.shared_core_generalization.
    shared_core_gate_config_from_dict`'s convention of filling in defaults
    for whatever the file omits."""
    defaults = TaskContentProbeConfig()
    shared_core_raw = raw.get("shared_core")
    shared_core = (
        shared_core_gate_config_from_dict(shared_core_raw)
        if shared_core_raw
        else defaults.shared_core
    )
    if not shared_core.include_task_spec:
        shared_core = dataclasses.replace(shared_core, include_task_spec=True)
    probe_train_raw = raw.get("probe_train")
    probe_train = (
        dataclasses.replace(defaults.probe_train, **probe_train_raw)
        if probe_train_raw
        else defaults.probe_train
    )
    return TaskContentProbeConfig(
        seed=raw.get("seed", defaults.seed),
        shared_core=shared_core,
        probe_train=probe_train,
        num_probe_train_examples=raw.get(
            "num_probe_train_examples", defaults.num_probe_train_examples
        ),
        num_probe_eval_examples=raw.get(
            "num_probe_eval_examples", defaults.num_probe_eval_examples
        ),
        min_examples_per_operation=raw.get(
            "min_examples_per_operation", defaults.min_examples_per_operation
        ),
        operation_id_threshold=raw.get("operation_id_threshold", defaults.operation_id_threshold),
        argument_threshold=raw.get("argument_threshold", defaults.argument_threshold),
        content_threshold=raw.get("content_threshold", defaults.content_threshold),
    )


@dataclass
class _ExtractedFeatures:
    """Frozen `encode_split` features pulled from one batch of examples, at
    the sequence positions described in the module docstring. Plain
    (non-frozen) dataclass -- purely an internal transport object between
    `_extract_features` and the probe-fitting functions below."""

    task_vec: torch.Tensor  # [N, d_model] -- z_task at [TASK_END]. Probes T1/T2.
    content_vec_at_task_pos: torch.Tensor  # [N, d_model] -- optional leakage probe.
    operation_ids: torch.Tensor  # [N] long -- one per example.
    content_features: torch.Tensor  # [sum(content_len), d_model] -- h_content per content position.
    task_features_at_content_pos: torch.Tensor  # same positions, z_task field -- optional leakage.
    content_labels: torch.Tensor  # [sum(content_len)] long -- presented token id per position.
    scalar_features: dict[str, torch.Tensor]  # operation -> [n_op, d_model], task_vec rows.
    scalar_targets: dict[str, torch.Tensor]  # operation -> [n_op] long argument values.
    select_features: torch.Tensor | None  # [n_select, d_model] or None if SELECT unsampled.
    select_targets: torch.Tensor | None  # [n_select, max_content_length] multi-hot.
    select_masks: torch.Tensor | None  # [n_select, max_content_length] valid-slot mask.


def _extract_features(
    trained: TrainedSharedCore, examples: list[Example], *, max_content_length: int
) -> _ExtractedFeatures:
    """One frozen forward pass (`encode_split`, `torch.no_grad()`) over
    `examples`, gathered into the fixed-size tensors every probe below
    trains/evaluates on. `trained.model` is never put into `.train()` mode
    and never receives a gradient here."""
    model, tokens, device = trained.model, trained.tokens, trained.device
    batch = collate_batch(examples, tokens, device=device, include_task_spec=True)
    model.eval()
    with torch.no_grad():
        encoded = model.encode_split(batch.input_ids)

    task_end_idx = (batch.input_ids == tokens.task_end).to(torch.float32).argmax(dim=1)
    sep_idx = torch.tensor(batch.prompt_lengths, device=device, dtype=torch.long) - 1
    row_idx = torch.arange(len(examples), device=device)

    task_vec = encoded.task_state[row_idx, task_end_idx]
    content_vec_at_task_pos = encoded.content_state[row_idx, task_end_idx]
    operation_ids = torch.tensor(
        [operation_id(example.task_spec.operation_sequence[0]) for example in examples],  # type: ignore[union-attr]
        device=device,
        dtype=torch.long,
    )

    content_feature_rows: list[torch.Tensor] = []
    task_feature_rows: list[torch.Tensor] = []
    content_label_rows: list[torch.Tensor] = []
    scalar_features: dict[str, list[torch.Tensor]] = {
        name: [] for name in SCALAR_ARGUMENT_OPERATIONS
    }
    scalar_targets: dict[str, list[int]] = {name: [] for name in SCALAR_ARGUMENT_OPERATIONS}
    select_features: list[torch.Tensor] = []
    select_targets: list[torch.Tensor] = []
    select_masks: list[torch.Tensor] = []

    for row, example in enumerate(examples):
        assert example.task_spec is not None
        start = int(task_end_idx[row].item()) + 1
        end = int(sep_idx[row].item())
        content_feature_rows.append(encoded.content_state[row, start:end, :])
        task_feature_rows.append(encoded.task_state[row, start:end, :])
        content_label_rows.append(batch.input_ids[row, start:end])

        (step,) = example.task_spec.steps
        operation = step.operation
        if operation in SCALAR_ARGUMENT_OPERATIONS:
            (value,) = step.arguments.values()
            scalar_features[operation].append(task_vec[row])
            scalar_targets[operation].append(int(value))
        elif operation in SET_ARGUMENT_OPERATIONS:
            content_length = end - start
            target = torch.zeros(max_content_length, device=device)
            mask = torch.zeros(max_content_length, device=device)
            mask[:content_length] = 1.0
            for position in step.arguments["indices"]:
                target[position] = 1.0
            select_features.append(task_vec[row])
            select_targets.append(target)
            select_masks.append(mask)

    return _ExtractedFeatures(
        task_vec=task_vec,
        content_vec_at_task_pos=content_vec_at_task_pos,
        operation_ids=operation_ids,
        content_features=torch.cat(content_feature_rows, dim=0),
        task_features_at_content_pos=torch.cat(task_feature_rows, dim=0),
        content_labels=torch.cat(content_label_rows, dim=0),
        scalar_features={op: torch.stack(v) for op, v in scalar_features.items() if v},
        scalar_targets={
            op: torch.tensor(v, device=device, dtype=torch.long)
            for op, v in scalar_targets.items()
            if v
        },
        select_features=torch.stack(select_features) if select_features else None,
        select_targets=torch.stack(select_targets) if select_targets else None,
        select_masks=torch.stack(select_masks) if select_masks else None,
    )


def _fit_classification_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    num_classes: int,
    train_config: ProbeTrainConfig,
    device: torch.device,
) -> float:
    """Fit one `nn.Linear(d_model, num_classes)` by full-batch AdamW
    cross-entropy on `(train_features, train_labels)`; return top-1 accuracy
    on the disjoint `(eval_features, eval_labels)`."""
    d_model = train_features.shape[-1]
    probe = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    probe.train()
    for _ in range(train_config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        predictions = probe(eval_features).argmax(dim=-1)
        return (predictions == eval_labels).to(torch.float32).mean().item()


def _fit_multilabel_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    eval_mask: torch.Tensor,
    train_config: ProbeTrainConfig,
    device: torch.device,
) -> float:
    """Fit one `nn.Linear(d_model, num_slots)` by full-batch AdamW binary
    cross-entropy (masked to valid slots per row) on `(train_features,
    train_targets, train_mask)`; return masked per-slot binary accuracy on
    the disjoint eval set. Used for SELECT.indices (Task A1-C006's
    `PrimitiveCall` naming note applies: `indices` is a subset of content
    positions, not a single scalar -- see `apc.environments.task_spec`
    module docstring)."""
    d_model = train_features.shape[-1]
    num_slots = train_targets.shape[-1]
    probe = nn.Linear(d_model, num_slots).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    probe.train()
    for _ in range(train_config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_features)
        per_slot_loss = F.binary_cross_entropy_with_logits(logits, train_targets, reduction="none")
        loss = (per_slot_loss * train_mask).sum() / train_mask.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        eval_logits = probe(eval_features)
        predictions = (eval_logits > 0).to(torch.float32)
        correct = ((predictions == eval_targets).to(torch.float32) * eval_mask).sum()
        total = eval_mask.sum().clamp_min(1.0)
        return (correct / total).item()


@dataclass(frozen=True)
class ArgumentProbeResult:
    """One parameterized operation's argument-decoding probe result."""

    operation: str
    metric_name: str  # "top1_accuracy" (SHIFT/COUNT/BIND) or "masked_slot_accuracy" (SELECT)
    accuracy: float
    num_train_examples: int
    num_eval_examples: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _count_for_operation(extracted: _ExtractedFeatures, name: str) -> int:
    if name in SCALAR_ARGUMENT_OPERATIONS:
        targets = extracted.scalar_targets.get(name)
        return 0 if targets is None else targets.shape[0]
    return 0 if extracted.select_targets is None else extracted.select_targets.shape[0]


def _check_min_examples_per_operation(
    extracted: _ExtractedFeatures, min_examples_per_operation: int, *, label: str
) -> None:
    missing = [
        name
        for name in PARAMETERIZED_OPERATION_NAMES
        if _count_for_operation(extracted, name) < min_examples_per_operation
    ]
    if missing:
        raise ValueError(
            f"{label} batch has fewer than {min_examples_per_operation} examples for "
            f"operation(s) {missing}; increase the corresponding num_probe_*_examples"
        )


@dataclass(frozen=True)
class TaskContentProbeReport:
    """Everything observed while freezing one shared Stable Core and fitting
    its A1-C005 probes."""

    config: TaskContentProbeConfig
    shared_core_overall_exact_match: float
    shared_core_per_operation_exact_match: dict[str, float]
    operation_id_accuracy: float
    operation_id_passed: bool
    argument_probes: dict[str, ArgumentProbeResult]
    argument_probes_passed: bool
    content_token_accuracy: float
    content_probe_passed: bool
    operation_from_content_accuracy: float
    content_from_task_accuracy: float
    num_probe_train_examples: int
    num_probe_eval_examples: int
    passed: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "shared_core_overall_exact_match": self.shared_core_overall_exact_match,
            "shared_core_per_operation_exact_match": self.shared_core_per_operation_exact_match,
            "operation_id_accuracy": self.operation_id_accuracy,
            "operation_id_passed": self.operation_id_passed,
            "argument_probes": {
                name: result.to_dict() for name, result in self.argument_probes.items()
            },
            "argument_probes_passed": self.argument_probes_passed,
            "content_token_accuracy": self.content_token_accuracy,
            "content_probe_passed": self.content_probe_passed,
            "operation_from_content_accuracy": self.operation_from_content_accuracy,
            "content_from_task_accuracy": self.content_from_task_accuracy,
            "num_probe_train_examples": self.num_probe_train_examples,
            "num_probe_eval_examples": self.num_probe_eval_examples,
            "passed": self.passed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_task_content_probes(
    config: TaskContentProbeConfig, metrics_path: str | Path | None = None
) -> TaskContentProbeReport:
    """Train one shared Stable Core (`train_shared_core`, Task A1-C004's own
    training loop), freeze it, and fit the A1-C005 representation probes on
    fresh procedurally-generated probe-train/probe-eval batches drawn from
    `"test"`-split steps the core's own training loop never touches (steps
    far outside `[0, config.shared_core.train.steps)`, so this is
    independent of, but reproducible alongside, the core's own A1-C004-style
    unseen-content evaluation at `step=0`).
    """
    start = time.perf_counter()
    trained = train_shared_core(config.shared_core, metrics_path=metrics_path)
    model, tokens, generator, device = (
        trained.model,
        trained.tokens,
        trained.generator,
        trained.device,
    )

    # Sanity-check the frozen core is actually a passing H1b instance at this
    # seed before trusting any probe fit on top of it (a probe on a core that
    # never learned to condition on the task segment would be meaningless).
    sanity_examples = generator.generate_online(
        config.shared_core.num_unseen_eval_examples, step=0, split="test"
    )
    shared_core_overall_exact_match, sanity_predictions = evaluate_exact_match(
        model, sanity_examples, tokens, device, include_task_spec=True
    )
    shared_core_per_operation_exact_match: dict[str, float] = {}
    for name in config.shared_core.operation_names:
        matches = total = 0
        for example, prediction in zip(sanity_examples, sanity_predictions, strict=True):
            assert example.task_spec is not None
            if example.task_spec.operation_sequence[0] != name:
                continue
            total += 1
            matches += int(prediction == example.target_tokens)
        shared_core_per_operation_exact_match[name] = matches / total if total else float("nan")

    max_content_length = config.shared_core.sequence_length_range[1]
    # Distinct, far-apart online steps from the core's own train("train")/
    # progress-eval("val")/sanity-check(step=0, "test") calls above, so
    # probe-train and probe-eval are independent fresh procedural batches of
    # each other and of everything the core itself ever saw.
    probe_train_examples = generator.generate_online(
        config.num_probe_train_examples, step=10_000, split="test"
    )
    probe_eval_examples = generator.generate_online(
        config.num_probe_eval_examples, step=20_000, split="test"
    )

    train_features = _extract_features(
        trained, probe_train_examples, max_content_length=max_content_length
    )
    eval_features = _extract_features(
        trained, probe_eval_examples, max_content_length=max_content_length
    )
    _check_min_examples_per_operation(
        train_features, config.min_examples_per_operation, label="probe-train"
    )
    _check_min_examples_per_operation(
        eval_features, config.min_examples_per_operation, label="probe-eval"
    )

    # Re-anchor the shared global RNG stream before fitting probes (ADR-0016
    # pattern): probe weight init should not depend on how many draws the
    # frozen core's own training/eval calls happened to consume.
    set_seed(config.seed)

    num_operations = len(KNOWN_OPERATION_NAMES)

    # Probe T1: operation identity from z_task.
    operation_id_accuracy = _fit_classification_probe(
        train_features.task_vec,
        train_features.operation_ids,
        eval_features.task_vec,
        eval_features.operation_ids,
        num_operations,
        config.probe_train,
        device,
    )
    operation_id_passed = operation_id_accuracy >= config.operation_id_threshold

    # Probe T2: operation argument from z_task, where applicable.
    argument_probes: dict[str, ArgumentProbeResult] = {}
    arg_span = tokens.arg_span
    for name in SCALAR_ARGUMENT_OPERATIONS:
        accuracy = _fit_classification_probe(
            train_features.scalar_features[name],
            train_features.scalar_targets[name],
            eval_features.scalar_features[name],
            eval_features.scalar_targets[name],
            arg_span,
            config.probe_train,
            device,
        )
        argument_probes[name] = ArgumentProbeResult(
            operation=name,
            metric_name="top1_accuracy",
            accuracy=accuracy,
            num_train_examples=train_features.scalar_targets[name].shape[0],
            num_eval_examples=eval_features.scalar_targets[name].shape[0],
            passed=accuracy >= config.argument_threshold,
        )
    for name in SET_ARGUMENT_OPERATIONS:
        assert train_features.select_features is not None
        assert eval_features.select_features is not None
        accuracy = _fit_multilabel_probe(
            train_features.select_features,
            train_features.select_targets,  # type: ignore[arg-type]
            train_features.select_masks,  # type: ignore[arg-type]
            eval_features.select_features,
            eval_features.select_targets,  # type: ignore[arg-type]
            eval_features.select_masks,  # type: ignore[arg-type]
            config.probe_train,
            device,
        )
        argument_probes[name] = ArgumentProbeResult(
            operation=name,
            metric_name="masked_slot_accuracy",
            accuracy=accuracy,
            num_train_examples=train_features.select_features.shape[0],
            num_eval_examples=eval_features.select_features.shape[0],
            passed=accuracy >= config.argument_threshold,
        )
    argument_probes_passed = all(result.passed for result in argument_probes.values())

    # Probe C1: content feature (per-position token identity) from h_content.
    content_token_accuracy = _fit_classification_probe(
        train_features.content_features,
        train_features.content_labels,
        eval_features.content_features,
        eval_features.content_labels,
        config.shared_core.vocab_size,
        config.probe_train,
        device,
    )
    content_probe_passed = content_token_accuracy >= config.content_threshold

    # Optional leakage probes -- informational only, never gate `passed`.
    operation_from_content_accuracy = _fit_classification_probe(
        train_features.content_vec_at_task_pos,
        train_features.operation_ids,
        eval_features.content_vec_at_task_pos,
        eval_features.operation_ids,
        num_operations,
        config.probe_train,
        device,
    )
    content_from_task_accuracy = _fit_classification_probe(
        train_features.task_features_at_content_pos,
        train_features.content_labels,
        eval_features.task_features_at_content_pos,
        eval_features.content_labels,
        config.shared_core.vocab_size,
        config.probe_train,
        device,
    )

    passed = operation_id_passed and argument_probes_passed and content_probe_passed

    return TaskContentProbeReport(
        config=config,
        shared_core_overall_exact_match=shared_core_overall_exact_match,
        shared_core_per_operation_exact_match=shared_core_per_operation_exact_match,
        operation_id_accuracy=operation_id_accuracy,
        operation_id_passed=operation_id_passed,
        argument_probes=argument_probes,
        argument_probes_passed=argument_probes_passed,
        content_token_accuracy=content_token_accuracy,
        content_probe_passed=content_probe_passed,
        operation_from_content_accuracy=operation_from_content_accuracy,
        content_from_task_accuracy=content_from_task_accuracy,
        num_probe_train_examples=len(probe_train_examples),
        num_probe_eval_examples=len(probe_eval_examples),
        passed=passed,
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
class TaskContentProbeMultiSeedReport:
    """A1-C005's verdict aggregated across seeds. Unlike A1-C004/A1-006,
    `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C005 states no minimum
    seed count -- `per_seed`/`seeds` are reported regardless of length, and
    `passed` is a per-seed-then-ANDed verdict (every seed's own probes must
    pass its own thresholds), not a mean-based threshold like the
    generalization gates."""

    seeds: tuple[int, ...]
    per_seed: tuple[TaskContentProbeReport, ...]
    mean_operation_id_accuracy: float
    stdev_operation_id_accuracy: float
    min_operation_id_accuracy: float
    max_operation_id_accuracy: float
    mean_content_token_accuracy: float
    stdev_content_token_accuracy: float
    min_content_token_accuracy: float
    max_content_token_accuracy: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "mean_operation_id_accuracy": self.mean_operation_id_accuracy,
            "stdev_operation_id_accuracy": self.stdev_operation_id_accuracy,
            "min_operation_id_accuracy": self.min_operation_id_accuracy,
            "max_operation_id_accuracy": self.max_operation_id_accuracy,
            "mean_content_token_accuracy": self.mean_content_token_accuracy,
            "stdev_content_token_accuracy": self.stdev_content_token_accuracy,
            "min_content_token_accuracy": self.min_content_token_accuracy,
            "max_content_token_accuracy": self.max_content_token_accuracy,
            "passed": self.passed,
        }


def run_task_content_probes_multi_seed(
    base_config: TaskContentProbeConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> TaskContentProbeMultiSeedReport:
    """Run one A1-C005 probe seed per entry in `seeds` (`base_config.seed`
    and `base_config.shared_core.seed` both overridden per seed -- a fresh
    shared core is trained from scratch for every seed, matching
    `apc.evaluation.shared_core_generalization.run_shared_core_gate_multi_seed`'s
    convention) and aggregate."""
    per_seed = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(
            base_config,
            seed=seed,
            shared_core=dataclasses.replace(base_config.shared_core, seed=seed),
        )
        metrics_path = run_dir_path / f"seed_{seed}" / "metrics.jsonl" if run_dir_path else None
        per_seed.append(run_task_content_probes(config, metrics_path=metrics_path))

    op_id_values = [report.operation_id_accuracy for report in per_seed]
    content_values = [report.content_token_accuracy for report in per_seed]
    op_id_mean, op_id_stdev, op_id_min, op_id_max = _summarize(op_id_values)
    content_mean, content_stdev, content_min, content_max = _summarize(content_values)

    return TaskContentProbeMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        mean_operation_id_accuracy=op_id_mean,
        stdev_operation_id_accuracy=op_id_stdev,
        min_operation_id_accuracy=op_id_min,
        max_operation_id_accuracy=op_id_max,
        mean_content_token_accuracy=content_mean,
        stdev_content_token_accuracy=content_stdev,
        min_content_token_accuracy=content_min,
        max_content_token_accuracy=content_max,
        passed=all(report.passed for report in per_seed),
    )
