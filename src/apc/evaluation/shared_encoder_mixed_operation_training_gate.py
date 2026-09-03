"""Balanced mixed-operation training gate (Phase A.1 diagnostic Task
A1-R005E-S002, `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

## What this tests

A1-R005E-S001 (`apc.evaluation.shared_encoder_architecture_gate`) built the
architecture -- one shared, unfrozen, freshly-initialized content encoder
plus one dedicated `CompactCrossPositionOperator` per operation -- but never
trained it (its own Acceptance: "No milestone benchmark yet"). This task
actually trains that architecture: **one** optimizer, **one** shared encoder,
receiving gradients from all four operations' own downstream losses in a
single interleaved training loop, so that A1-R005E-S003 can later ask how
much of A1-R005E-006A's per-operation gain survives when the encoder is
shared instead of duplicated per operation.

```text
                         +-> Compact SHIFT(amount)
                         +-> Compact SELECT(indices)
content -> Shared Encoder+-> Compact COUNT(target)
                         +-> Compact BIND(key)
```

## Reused vs. new

Per the task's own "Work" list ("Balanced mixed-operation sampling",
"Preserve counterfactual groups", "Jointly train: shared encoder, selected
compact operator, selected readout"), this module builds directly on top of
`apc.evaluation.shared_encoder_architecture_gate.build_shared_encoder_
architecture`/`SharedEncoderArchitecture`/`run_shared_operator` (S001's own
architecture, unchanged -- not rebuilt here) and
`apc.evaluation.compact_cross_position_operator_probe`'s counterfactual-group
protocol (`generate_compact_operator_counterfactual_groups`, `_flatten_
groups`, `_correct_argument_value`, `_labels_for_examples`, the same
Correct/effectful-Wrong/None threshold constants A1-R005E-004/005/006A all
use), matching every earlier module in this diagnostic chain's convention of
importing the operator/group machinery unchanged rather than re-deriving it.

Genuinely new here: `_build_operation_schedule` (the balanced interleaving
policy), `_train_shared` (one optimizer over the encoder plus every
operator's own parameters, stepped once per training step against whichever
operation that step's schedule entry names), `_evaluate_arm_shared` (E-006A's
own `_evaluate_arm` shape, but dispatching through `run_shared_operator`
instead of a private `(core, operator)` pair), and the per-operation
step/example/gradient-norm accounting this task's own "Required metrics"
section names.

## Balanced mixed-operation sampling

`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` section 3 sets the
default proportions at SHIFT/SELECT/COUNT/BIND 25% each.
`_build_operation_schedule` realizes this **exactly**, not merely in
expectation: it builds one "cycle" list containing each operation repeated
`operation_sample_weights[operation]` times (1 each by default -- one full
lap of all four operations), deterministically reshuffles a fresh copy of
that cycle once per lap (`_derive_local_seed(seed, cycle_index,
"operation_schedule")`, the same per-module local-seed derivation every
A1-R005D/A1-R005E counterfactual gate uses for its own content generation --
duplicated here rather than imported, per this codebase's own "each gate
module independently reviewable end to end" convention), and concatenates
laps until `joint_train.steps` is reached. With the default weights and a
`joint_train.steps` that is a multiple of `len(operation_names)`, every
operation's own realized step count is therefore *exactly* equal, not merely
close -- stronger than IID sampling would guarantee over a finite budget, and
directly satisfies `docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`'s
"report the realized proportions."

## Why `joint_train.steps` defaults to 152000

A1-R005E-006A trained one dedicated encoder per operation, each for 38000
steps (its own module docstring's "Why...steps defaults to 38000"). For
A1-R005E-S003's later "how much of E-006A survives sharing" comparison to be
a fair like-for-like on training budget, each operation's own compact
operator should receive the *same* number of its own gradient updates here
as it did in A1-R005E-006A -- not a smaller slice of a fixed total. Since
only the selected operation's operator receives a gradient on any given step
(see "One optimizer, naturally selective updates" below), giving each of the
four operations an equal 25% share of `4 * 38000 = 152000` total steps means
every operator ends up with ~38000 of its own update steps, matching
A1-R005E-006A's own per-operator budget exactly (exactly, not
approximately, under the default balanced weights -- see above). The shared
encoder itself receives one gradient update on *every* step regardless of
which operation is selected, so it ends up far better-trained in raw update
count (152000 updates) than any single A1-R005E-006A encoder (38000 updates
each) -- an intentional asymmetry: A1-R005E-006A's own total *step* budget
(760000 gradient steps across 5 seeds x 4 operations) is preserved bit-for-
bit here too (5 seeds x 152000 steps = 760000), so this task's own wall-clock
cost is expected to closely match A1-R005E-006A's own measured ~3h29m on the
same RTX 5060 Ti (`docs/DECISIONS_A1_R005E_DIAGNOSTIC.md` ADR-0043), not
exceed it.

## One optimizer, naturally selective updates

`_train_shared` builds exactly one `torch.optim.AdamW` over the shared
encoder's parameters plus *every* operator's own parameters (all four,
concatenated once at construction). On any given step only the selected
operation's own forward/backward path touches its operator's parameters, so
every other operator's parameters keep `grad is None` for that step;
`optimizer.zero_grad(set_to_none=True)` before each step combined with
PyTorch's own per-parameter skip-if-`grad is None` behavior in
`AdamW.step()` means non-selected operators are never touched by that step's
`optimizer.step()` call -- no manual per-operator optimizer or parameter
group is needed to realize "selected compact operator, selected readout"
(task "Work" item 3) with a single shared optimizer instance.

## Preserve counterfactual groups

Every training step still draws a `CompactOperatorGroup` batch via
`generate_compact_operator_counterfactual_groups` for whichever operation the
schedule names at that step (task "Work" item 2) -- the only change from
A1-R005E-006A's own `_train_joint` is *which* operation's groups are drawn on
a given step, not how they are drawn or trained on. `generate_compact_
operator_counterfactual_groups` is deterministic by `(seed, step, split,
operation)` via its own local `random.Random` stream (not the shared global
RNG), so interleaving operations across steps does not perturb any single
operation's own content stream relative to what a dedicated per-operation
loop would have drawn at that same `step` value.

## Task-blind invariance

Same regression check as A1-R005E-006A's own `_task_blind_invariance_max_
abs_diff`, run once per operation after training completes, against the one
shared (and now jointly-trained-across-all-four-operations) encoder --
verifying the hard invariant
(`docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`: "The encoder sees content
only") still holds after training, not merely by the structural guarantee of
`encode_content`'s own operation-blind signature.

## No branch claim here

Per the task's own Acceptance ("Complete all measurements; no branch claim
yet"), this module reports per-operation `passed` flags against the same
absolute thresholds every earlier module in this diagnostic chain reports
against (`CORRECT_EXACT_MATCH_THRESHOLD` etc., imported unchanged from
`apc.evaluation.compact_cross_position_operator_probe`) -- these are routine
per-module measurements, not a Branch-B recommendation. No `R_shared`/
`R_access` comparison against A1-R005E-006A is computed here; that
retention analysis is A1-R005E-S003's own job, and no `Router`,
`PrimitiveBank`, `Primitive`, `PlasticWorkspace`, or `apc.consolidation`/
`apc.meta` module is imported here, same as every earlier module in this
diagnostic chain.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX
from apc.environments.generator import Example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    CORRECT_EXACT_MATCH_THRESHOLD,
    DEFAULT_GROUP_SIZE,
    EFFECTFUL_WRONG_ARGUMENT_CEILING,
    MIN_CAUSAL_GAP,
    MIN_GROUP_SIZE,
    NONE_CEILING,
    TOKEN_ACCURACY_GATED_OPERATIONS,
    TOKEN_ACCURACY_THRESHOLD,
    CompactOperatorGroup,
    _correct_argument_value,
    _default_model_config,
    _flatten_groups,
    _labels_for_examples,
    generate_compact_operator_counterfactual_groups,
)
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedContentEncoder,
    SharedEncoderArchitecture,
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
    encode_content,
    run_shared_operator,
)
from apc.primitives.conditioning import DEFAULT_ARG_DIM, DEFAULT_MAX_SEQUENCE_LENGTH
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "TASK_BLIND_ATOL",
    "DEFAULT_TOTAL_STEPS",
    "SharedMixedTrainingConfig",
    "shared_mixed_training_config_from_dict",
    "OperationSharedTrainingReport",
    "SharedMixedTrainingReport",
    "run_shared_mixed_training_gate",
    "OperationSharedTrainingSummary",
    "SharedMixedTrainingMultiSeedReport",
    "run_shared_mixed_training_gate_multi_seed",
]

# Same >=5-seed decision-evidence bar every A1-R005D/A1-R005E counterfactual
# gate has used since A1-R005E-004.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5

# Matches every earlier module's own task-blind regression-check tolerance.
TASK_BLIND_ATOL = 1e-5

# 4 operations x A1-R005E-006A's own 38000-step per-operator budget -- see
# module docstring "Why joint_train.steps defaults to 152000".
DEFAULT_TOTAL_STEPS = len(PARAMETERIZED_OPERATION_NAMES) * 38000

_EVAL_BATCH_SIZE = 256


def _derive_local_seed(seed: int, step: int, label: str) -> int:
    """Deterministic sub-seed for one `(seed, step, label)` triple,
    independent of `PYTHONHASHSEED` -- same construction every A1-R005D/
    A1-R005E counterfactual gate uses, kept local per this codebase's own
    "each gate module is independently reviewable end to end" convention."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SharedMixedTrainingConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-S002
    balanced mixed-operation training run.

    `model`/`d_operator`/`n_operator_head`/`d_operator_ff`/`arg_dim`/
    `max_sequence_length` default to A1-R005E-S001's own values unchanged
    (this task trains that architecture, it does not redesign it).
    `operation_sample_weights` defaults to 1 per operation -- the balanced
    25/25/25/25 split `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`
    section 3 names.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    device: str = "auto"
    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    operation_sample_weights: dict[str, int] = field(
        default_factory=lambda: {name: 1 for name in PARAMETERIZED_OPERATION_NAMES}
    )
    joint_train: PrimitiveTrainConfig = field(
        default_factory=lambda: PrimitiveTrainConfig(steps=DEFAULT_TOTAL_STEPS, eval_every=8000)
    )
    num_unseen_eval_groups: int = 1400
    min_unseen_eval_examples: int = 1024

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.group_size < MIN_GROUP_SIZE:
            raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {self.group_size}")
        if self.d_operator < 1:
            raise ValueError(f"d_operator must be >= 1, got {self.d_operator}")
        if self.n_operator_head < 1:
            raise ValueError(f"n_operator_head must be >= 1, got {self.n_operator_head}")
        if self.d_operator % self.n_operator_head != 0:
            raise ValueError(
                f"d_operator ({self.d_operator}) must be divisible by n_operator_head "
                f"({self.n_operator_head})"
            )
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= the upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]})"
            )
        if set(self.operation_sample_weights) != set(self.operation_names):
            raise ValueError(
                "operation_sample_weights must have exactly one entry per operation_names entry; "
                f"got weights for {sorted(self.operation_sample_weights)}, operations "
                f"{sorted(self.operation_names)}"
            )
        if any(weight < 1 for weight in self.operation_sample_weights.values()):
            raise ValueError(
                f"operation_sample_weights entries must all be >= 1, got "
                f"{self.operation_sample_weights}"
            )
        cycle_length = sum(self.operation_sample_weights.values())
        if self.joint_train.steps < cycle_length:
            raise ValueError(
                f"joint_train.steps ({self.joint_train.steps}) must be >= the operation "
                f"sampling cycle length ({cycle_length}) so every operation is trained at "
                "least once"
            )
        if self.num_unseen_eval_groups < 1:
            raise ValueError(
                f"num_unseen_eval_groups must be >= 1, got {self.num_unseen_eval_groups}"
            )
        if self.min_unseen_eval_examples < 1:
            raise ValueError(
                f"min_unseen_eval_examples must be >= 1, got {self.min_unseen_eval_examples}"
            )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def to_architecture_config(self) -> SharedEncoderArchitectureConfig:
        """The `apc.evaluation.shared_encoder_architecture_gate` config this
        run's architecture is built from -- every field this task shares with
        A1-R005E-S001 unchanged."""
        return SharedEncoderArchitectureConfig(
            seed=self.seed,
            vocab_size=self.vocab_size,
            sequence_length_range=self.sequence_length_range,
            operation_names=self.operation_names,
            group_size=self.group_size,
            model=self.model,
            device=self.device,
            d_operator=self.d_operator,
            n_operator_head=self.n_operator_head,
            d_operator_ff=self.d_operator_ff,
            arg_dim=self.arg_dim,
            max_sequence_length=self.max_sequence_length,
        )


def shared_mixed_training_config_from_dict(raw: dict[str, Any]) -> SharedMixedTrainingConfig:
    """Parse a `configs/phase_a1_shared_encoder_mixed_operation_training_gate.
    yaml`-shaped dict, matching every other Phase A.1 gate's convention of
    filling in defaults for whatever the file omits."""
    defaults = SharedMixedTrainingConfig()
    joint_train_raw = raw.get("joint_train")
    joint_train = (
        dataclasses.replace(defaults.joint_train, **joint_train_raw)
        if joint_train_raw
        else defaults.joint_train
    )
    operation_names = tuple(raw.get("operation_names", defaults.operation_names))
    if "operation_sample_weights" in raw:
        operation_sample_weights = dict(raw["operation_sample_weights"])
    elif set(operation_names) == set(defaults.operation_names):
        operation_sample_weights = dict(defaults.operation_sample_weights)
    else:
        operation_sample_weights = {name: 1 for name in operation_names}
    return SharedMixedTrainingConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=operation_names,
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        device=raw.get("device", defaults.device),
        d_operator=raw.get("d_operator", defaults.d_operator),
        n_operator_head=raw.get("n_operator_head", defaults.n_operator_head),
        d_operator_ff=raw.get("d_operator_ff", defaults.d_operator_ff),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        operation_sample_weights=operation_sample_weights,
        joint_train=joint_train,
        num_unseen_eval_groups=raw.get("num_unseen_eval_groups", defaults.num_unseen_eval_groups),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


def _build_operation_schedule(
    operation_names: Sequence[str],
    operation_sample_weights: dict[str, int],
    total_steps: int,
    seed: int,
) -> tuple[str, ...]:
    """Deterministic balanced interleaving of `operation_names` across
    `total_steps` training steps (module docstring, "Balanced mixed-operation
    sampling"). One "cycle" is `operation_names[i]` repeated `operation_sample_
    weights[operation_names[i]]` times; each successive cycle is freshly (and
    deterministically) reshuffled and appended until `total_steps` is
    reached, then truncated to exactly `total_steps`."""
    cycle = [
        name for name in operation_names for _ in range(operation_sample_weights[name])
    ]
    schedule: list[str] = []
    cycle_index = 0
    while len(schedule) < total_steps:
        rng = random.Random(_derive_local_seed(seed, cycle_index, "operation_schedule"))
        shuffled = list(cycle)
        rng.shuffle(shuffled)
        schedule.extend(shuffled)
        cycle_index += 1
    return tuple(schedule[:total_steps])


def _param_group_grad_norm(parameters: Sequence[nn.Parameter], grad_clip: float) -> float:
    """Total gradient norm for one parameter group, clipping it in place when
    `grad_clip > 0` (matching A1-R005E-006A's own `_param_group_grad_norm`)."""
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    return float(nn.utils.clip_grad_norm_(parameters, max_norm))


def _evaluate_arm_shared(
    architecture: SharedEncoderArchitecture,
    operation: str,
    examples: Sequence[Example],
    *,
    argument_provider: Callable[[Example], Any] | None,
) -> tuple[float, float]:
    """Exact match / token accuracy for one causal-ablation arm, dispatching
    through `run_shared_operator` -- the shared-architecture analogue of
    A1-R005E-006A's own `_evaluate_arm`."""
    core = architecture.core
    operator = architecture.operators[operation]
    core.model.eval()
    operator.eval()
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), _EVAL_BATCH_SIZE):
            chunk = examples[start : start + _EVAL_BATCH_SIZE]
            content_lengths = [len(example.input_tokens) for example in chunk]
            output_lengths = [
                get_operation(operation).output_length(length) for length in content_lengths
            ]
            argument_values = (
                None
                if argument_provider is None
                else [argument_provider(example) for example in chunk]
            )
            logits = run_shared_operator(architecture, operation, chunk, argument_values)
            predictions = logits.argmax(dim=-1)
            for row, (example, n) in enumerate(zip(chunk, output_lengths, strict=True)):
                target = example.target_tokens
                prediction = tuple(predictions[row, :n].tolist())
                total_tokens += len(target)
                correct_tokens += sum(
                    1 for p, t in zip(prediction, target, strict=True) if p == t
                )
                if prediction == target:
                    exact_matches += 1
    operator.train()
    core.model.train()
    return exact_matches / len(examples), correct_tokens / total_tokens


def _task_blind_invariance_max_abs_diff(
    core: SharedContentEncoder, group: CompactOperatorGroup
) -> float:
    """Max absolute pairwise difference in `h_content` across `group`'s own
    members -- same content, different (task-blind-invisible) argument
    values/targets -- on the trained shared encoder (module docstring,
    "Task-blind invariance")."""
    core.model.eval()
    content_features, _ = encode_content(core, list(group.examples))
    core.model.train()
    reference = content_features[0:1]
    return float((content_features - reference).abs().max().item())


@dataclass
class _TrainingAccumulator:
    step_counts: dict[str, int]
    examples_seen: dict[str, int]
    sum_loss: dict[str, float]
    sum_encoder_grad_norm: dict[str, float]
    sum_operator_grad_norm: dict[str, float]
    final_loss: dict[str, float]
    final_encoder_grad_norm: dict[str, float]
    final_operator_grad_norm: dict[str, float]

    @classmethod
    def empty(cls, operation_names: Sequence[str]) -> _TrainingAccumulator:
        zeros_int = {name: 0 for name in operation_names}
        zeros_float = {name: 0.0 for name in operation_names}
        nans = {name: float("nan") for name in operation_names}
        return cls(
            step_counts=dict(zeros_int),
            examples_seen=dict(zeros_int),
            sum_loss=dict(zeros_float),
            sum_encoder_grad_norm=dict(zeros_float),
            sum_operator_grad_norm=dict(zeros_float),
            final_loss=dict(nans),
            final_encoder_grad_norm=dict(nans),
            final_operator_grad_norm=dict(nans),
        )

    def mean_loss(self, operation: str) -> float:
        count = self.step_counts[operation]
        return self.sum_loss[operation] / count if count else float("nan")

    def mean_encoder_grad_norm(self, operation: str) -> float:
        count = self.step_counts[operation]
        return self.sum_encoder_grad_norm[operation] / count if count else float("nan")

    def mean_operator_grad_norm(self, operation: str) -> float:
        count = self.step_counts[operation]
        return self.sum_operator_grad_norm[operation] / count if count else float("nan")


def _train_shared(
    architecture: SharedEncoderArchitecture,
    config: SharedMixedTrainingConfig,
    metrics_path: str | Path | None = None,
) -> _TrainingAccumulator:
    """Jointly train the shared encoder and every operator on interleaved
    counterfactual groups, one operation per step per `_build_operation_
    schedule` (module docstring). Returns per-operation step/example/loss/
    grad-norm accounting; `architecture` is mutated in place (its encoder and
    operators are the trained result)."""
    device = architecture.core.device
    encoder_params = list(architecture.core.model.parameters())
    operator_params_by_op = {
        operation: list(operator.parameters())
        for operation, operator in architecture.operators.items()
    }
    all_operator_params = [
        parameter for params in operator_params_by_op.values() for parameter in params
    ]
    optimizer = torch.optim.AdamW(
        encoder_params + all_operator_params,
        lr=config.joint_train.lr,
        weight_decay=config.joint_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after optimizer construction,
    # per ADR-0016's pattern.
    set_seed(config.seed)

    schedule = _build_operation_schedule(
        config.operation_names,
        config.operation_sample_weights,
        config.joint_train.steps,
        config.seed,
    )
    groups_per_step = max(1, -(-config.joint_train.batch_size // config.group_size))
    accumulator = _TrainingAccumulator.empty(config.operation_names)

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    architecture.core.model.train()
    for operator in architecture.operators.values():
        operator.train()

    try:
        for step, operation in enumerate(schedule):
            groups = generate_compact_operator_counterfactual_groups(
                config.seed,
                groups_per_step,
                operation=operation,
                step=step,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, _, _ = _flatten_groups(groups)
            argument_values = [_correct_argument_value(operation, example) for example in examples]
            content_lengths = [len(example.input_tokens) for example in examples]
            output_lengths = [
                get_operation(operation).output_length(length) for length in content_lengths
            ]
            out_max = max(output_lengths)
            labels = _labels_for_examples(examples, output_lengths, out_max, device)

            optimizer.zero_grad(set_to_none=True)
            logits = run_shared_operator(architecture, operation, examples, argument_values)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
            loss.backward()
            encoder_grad_norm = _param_group_grad_norm(encoder_params, config.joint_train.grad_clip)
            operator_grad_norm = _param_group_grad_norm(
                operator_params_by_op[operation], config.joint_train.grad_clip
            )
            optimizer.step()

            accumulator.step_counts[operation] += 1
            accumulator.examples_seen[operation] += len(examples)
            accumulator.sum_loss[operation] += float(loss.item())
            accumulator.sum_encoder_grad_norm[operation] += encoder_grad_norm
            accumulator.sum_operator_grad_norm[operation] += operator_grad_norm
            accumulator.final_loss[operation] = float(loss.item())
            accumulator.final_encoder_grad_norm[operation] = encoder_grad_norm
            accumulator.final_operator_grad_norm[operation] = operator_grad_norm

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.joint_train.eval_every == 0
                or step_number == config.joint_train.steps
            ):
                progress_correct: dict[str, float] = {}
                for probe_operation in config.operation_names:
                    progress_groups = generate_compact_operator_counterfactual_groups(
                        config.seed,
                        max(
                            1,
                            -(-config.joint_train.progress_eval_examples // config.group_size),
                        ),
                        operation=probe_operation,
                        step=step,
                        split="val",
                        vocab_size=config.vocab_size,
                        sequence_length_range=config.sequence_length_range,
                        group_size=config.group_size,
                    )
                    progress_examples, _, _ = _flatten_groups(progress_groups)
                    progress_exact_match, _ = _evaluate_arm_shared(
                        architecture,
                        probe_operation,
                        progress_examples,
                        argument_provider=functools.partial(
                            _correct_argument_value, probe_operation
                        ),
                    )
                    progress_correct[probe_operation] = progress_exact_match
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "current_operation": operation,
                            "operation_step_counts": dict(accumulator.step_counts),
                            "operation_mean_loss_so_far": {
                                name: accumulator.mean_loss(name)
                                for name in config.operation_names
                            },
                            "operation_mean_encoder_grad_norm_so_far": {
                                name: accumulator.mean_encoder_grad_norm(name)
                                for name in config.operation_names
                            },
                            "operation_mean_operator_grad_norm_so_far": {
                                name: accumulator.mean_operator_grad_norm(name)
                                for name in config.operation_names
                            },
                            "operation_progress_correct_exact_match": progress_correct,
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
                # _evaluate_arm_shared restores operator.train()/core.model.
                # train() on its own; nothing further to do here.
    finally:
        if metrics_file is not None:
            metrics_file.close()

    return accumulator


@dataclass(frozen=True)
class OperationSharedTrainingReport:
    """Everything observed for one operation after the single shared-encoder
    mixed-operation training run completes -- the training accounting this
    task's own "Required metrics" section names, plus the same three-arm
    causal ablation (Correct / effectful Wrong argument / None) every earlier
    module in this diagnostic chain reports."""

    operation: str
    steps_trained: int
    examples_seen: int
    final_train_loss: float
    mean_train_loss: float
    final_encoder_grad_norm: float
    mean_encoder_grad_norm: float
    final_operator_grad_norm: float
    mean_operator_grad_norm: float
    operator_param_count: int
    correct_exact_match: float
    correct_token_accuracy: float
    effectful_wrong_argument_exact_match: float
    effectful_wrong_argument_token_accuracy: float
    none_exact_match: float
    none_token_accuracy: float
    exact_match_causal_gap: float
    token_accuracy_causal_gap: float
    argument_effect_rate: float
    task_blind_max_abs_diff: float
    task_blind_invariant_passed: bool
    num_unseen_eval_groups: int
    num_unseen_eval_examples: int
    correct_exact_match_passed: bool
    token_accuracy_passed: bool | None
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def _evaluate_operation_final(
    architecture: SharedEncoderArchitecture,
    config: SharedMixedTrainingConfig,
    operation: str,
    accumulator: _TrainingAccumulator,
) -> OperationSharedTrainingReport:
    """Post-training three-arm causal ablation for `operation`, on a large
    unseen batch of counterfactual groups, plus the training accounting
    already gathered in `accumulator` -- the shared-architecture analogue of
    A1-R005E-006A's own per-operation evaluation block."""
    unseen_groups = generate_compact_operator_counterfactual_groups(
        config.seed,
        config.num_unseen_eval_groups,
        operation=operation,
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )
    unseen_examples, wrong_argument_by_id, wrong_target_tokens_by_id = _flatten_groups(
        unseen_groups
    )
    if len(unseen_examples) < config.min_unseen_eval_examples:
        raise ValueError(
            f"unseen eval batch for {operation!r} realized only {len(unseen_examples)} examples "
            f"from {config.num_unseen_eval_groups} groups, below min_unseen_eval_examples="
            f"{config.min_unseen_eval_examples}; increase num_unseen_eval_groups"
        )

    correct_exact_match, correct_token_accuracy = _evaluate_arm_shared(
        architecture,
        operation,
        unseen_examples,
        argument_provider=lambda example: _correct_argument_value(operation, example),
    )
    wrong_exact_match, wrong_token_accuracy = _evaluate_arm_shared(
        architecture,
        operation,
        unseen_examples,
        argument_provider=lambda example: wrong_argument_by_id[id(example)],
    )
    none_exact_match, none_token_accuracy = _evaluate_arm_shared(
        architecture, operation, unseen_examples, argument_provider=None
    )

    argument_effect_rate = statistics.fmean(
        float(example.target_tokens != wrong_target_tokens_by_id[id(example)])
        for example in unseen_examples
    )
    exact_match_causal_gap = correct_exact_match - max(wrong_exact_match, none_exact_match)
    token_accuracy_causal_gap = correct_token_accuracy - max(
        wrong_token_accuracy, none_token_accuracy
    )

    task_blind_max_abs_diff = _task_blind_invariance_max_abs_diff(
        architecture.core, unseen_groups[0]
    )
    task_blind_invariant_passed = task_blind_max_abs_diff <= TASK_BLIND_ATOL

    correct_exact_match_passed = correct_exact_match >= CORRECT_EXACT_MATCH_THRESHOLD
    token_accuracy_passed = (
        correct_token_accuracy >= TOKEN_ACCURACY_THRESHOLD
        if operation in TOKEN_ACCURACY_GATED_OPERATIONS
        else None
    )
    effectful_wrong_argument_passed = wrong_exact_match <= EFFECTFUL_WRONG_ARGUMENT_CEILING
    none_passed = none_exact_match <= NONE_CEILING
    causal_gap_passed = exact_match_causal_gap >= MIN_CAUSAL_GAP
    passed = (
        correct_exact_match_passed
        and (token_accuracy_passed if token_accuracy_passed is not None else True)
        and effectful_wrong_argument_passed
        and none_passed
        and causal_gap_passed
    )

    return OperationSharedTrainingReport(
        operation=operation,
        steps_trained=accumulator.step_counts[operation],
        examples_seen=accumulator.examples_seen[operation],
        final_train_loss=accumulator.final_loss[operation],
        mean_train_loss=accumulator.mean_loss(operation),
        final_encoder_grad_norm=accumulator.final_encoder_grad_norm[operation],
        mean_encoder_grad_norm=accumulator.mean_encoder_grad_norm(operation),
        final_operator_grad_norm=accumulator.final_operator_grad_norm[operation],
        mean_operator_grad_norm=accumulator.mean_operator_grad_norm(operation),
        operator_param_count=sum(p.numel() for p in architecture.operators[operation].parameters()),
        correct_exact_match=correct_exact_match,
        correct_token_accuracy=correct_token_accuracy,
        effectful_wrong_argument_exact_match=wrong_exact_match,
        effectful_wrong_argument_token_accuracy=wrong_token_accuracy,
        none_exact_match=none_exact_match,
        none_token_accuracy=none_token_accuracy,
        exact_match_causal_gap=exact_match_causal_gap,
        token_accuracy_causal_gap=token_accuracy_causal_gap,
        argument_effect_rate=argument_effect_rate,
        task_blind_max_abs_diff=task_blind_max_abs_diff,
        task_blind_invariant_passed=task_blind_invariant_passed,
        num_unseen_eval_groups=len(unseen_groups),
        num_unseen_eval_examples=len(unseen_examples),
        correct_exact_match_passed=correct_exact_match_passed,
        token_accuracy_passed=token_accuracy_passed,
        effectful_wrong_argument_passed=effectful_wrong_argument_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        passed=passed,
    )


@dataclass(frozen=True)
class SharedMixedTrainingReport:
    """Everything observed for one seed of the A1-R005E-S002 balanced
    mixed-operation training gate."""

    config: SharedMixedTrainingConfig
    per_operation: dict[str, OperationSharedTrainingReport]
    operation_step_counts: dict[str, int]
    operation_examples_seen: dict[str, int]
    operation_realized_sampling_proportions: dict[str, float]
    total_steps: int
    encoder_update_count: int
    core_param_count: int
    core_trainable_param_count: int
    task_blind_invariant_passed: bool
    passed: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "per_operation": {
                name: report.to_dict() for name, report in self.per_operation.items()
            },
            "operation_step_counts": self.operation_step_counts,
            "operation_examples_seen": self.operation_examples_seen,
            "operation_realized_sampling_proportions": self.operation_realized_sampling_proportions,
            "total_steps": self.total_steps,
            "encoder_update_count": self.encoder_update_count,
            "core_param_count": self.core_param_count,
            "core_trainable_param_count": self.core_trainable_param_count,
            "task_blind_invariant_passed": self.task_blind_invariant_passed,
            "passed": self.passed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_shared_mixed_training_gate(
    config: SharedMixedTrainingConfig,
    *,
    metrics_path: str | Path | None = None,
) -> SharedMixedTrainingReport:
    """Run one seed of the A1-R005E-S002 benchmark: build the A1-R005E-S001
    shared-encoder architecture, jointly train it on balanced interleaved
    counterfactual groups from every operation in `config.operation_names`,
    then evaluate the three-arm causal ablation for each operation."""
    start = time.perf_counter()
    architecture = build_shared_encoder_architecture(config.to_architecture_config())
    accumulator = _train_shared(architecture, config, metrics_path=metrics_path)

    per_operation = {
        operation: _evaluate_operation_final(architecture, config, operation, accumulator)
        for operation in config.operation_names
    }

    total_steps = config.joint_train.steps
    operation_realized_sampling_proportions = {
        operation: accumulator.step_counts[operation] / total_steps
        for operation in config.operation_names
    }
    task_blind_invariant_passed = all(
        report.task_blind_invariant_passed for report in per_operation.values()
    )
    passed = all(report.passed for report in per_operation.values())

    return SharedMixedTrainingReport(
        config=config,
        per_operation=per_operation,
        operation_step_counts=dict(accumulator.step_counts),
        operation_examples_seen=dict(accumulator.examples_seen),
        operation_realized_sampling_proportions=operation_realized_sampling_proportions,
        total_steps=total_steps,
        encoder_update_count=total_steps,
        core_param_count=architecture.core.model.num_parameters(),
        core_trainable_param_count=architecture.core.model.num_parameters(trainable_only=True),
        task_blind_invariant_passed=task_blind_invariant_passed,
        passed=passed,
        wall_clock_seconds=time.perf_counter() - start,
        device=str(architecture.core.device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class OperationSharedTrainingSummary:
    """One operation's three-arm means aggregated across seeds, with
    pass/fail verdicts computed on the cross-seed mean (matching every
    earlier module's own convention). No `R_shared`/`R_access` comparison
    against A1-R005E-006A is computed here -- that retention analysis is
    A1-R005E-S003's own job (module docstring, "No branch claim here")."""

    operation: str
    mean_correct_exact_match: float
    stdev_correct_exact_match: float
    min_correct_exact_match: float
    max_correct_exact_match: float
    mean_correct_token_accuracy: float
    mean_effectful_wrong_argument_exact_match: float
    mean_effectful_wrong_argument_token_accuracy: float
    mean_none_exact_match: float
    mean_none_token_accuracy: float
    mean_exact_match_causal_gap: float
    mean_token_accuracy_causal_gap: float
    mean_argument_effect_rate: float
    mean_steps_trained: float
    mean_examples_seen: float
    mean_train_loss: float
    mean_encoder_grad_norm: float
    mean_operator_grad_norm: float
    max_task_blind_max_abs_diff: float
    task_blind_invariant_passed: bool
    operator_param_count: int
    correct_exact_match_passed: bool
    token_accuracy_passed: bool | None
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SharedMixedTrainingMultiSeedReport:
    """A1-R005E-S002's full result across seeds. Interpreting these numbers
    against A1-R005E-006A -- including any `R_shared`/branch decision -- is
    left to A1-R005E-S003/S005 (module docstring, "No branch claim here")."""

    seeds: tuple[int, ...]
    per_seed: tuple[SharedMixedTrainingReport, ...]
    per_operation_summary: dict[str, OperationSharedTrainingSummary]
    core_param_count: int
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "per_operation_summary": {
                name: summary.to_dict() for name, summary in self.per_operation_summary.items()
            },
            "core_param_count": self.core_param_count,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_shared_mixed_training_gate_multi_seed(
    base_config: SharedMixedTrainingConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> SharedMixedTrainingMultiSeedReport:
    """Run one benchmark seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation. Does not raise if
    `len(seeds) < MIN_GATE_SEEDS`; `meets_seed_policy` reports it honestly
    instead, matching every other Phase A.1 gate."""
    per_seed: list[SharedMixedTrainingReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        metrics_path = seed_dir / "mixed_training_metrics.jsonl" if seed_dir else None
        report = run_shared_mixed_training_gate(config, metrics_path=metrics_path)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    per_operation_summary: dict[str, OperationSharedTrainingSummary] = {}
    for operation in base_config.operation_names:
        op_reports = [report.per_operation[operation] for report in per_seed]
        mean_correct, stdev_correct, min_correct, max_correct = _summarize(
            [r.correct_exact_match for r in op_reports]
        )
        mean_correct_token_accuracy = statistics.fmean(r.correct_token_accuracy for r in op_reports)
        mean_wrong = statistics.fmean(r.effectful_wrong_argument_exact_match for r in op_reports)
        mean_wrong_token_accuracy = statistics.fmean(
            r.effectful_wrong_argument_token_accuracy for r in op_reports
        )
        mean_none = statistics.fmean(r.none_exact_match for r in op_reports)
        mean_none_token_accuracy = statistics.fmean(r.none_token_accuracy for r in op_reports)
        mean_exact_match_causal_gap = mean_correct - max(mean_wrong, mean_none)
        mean_token_accuracy_causal_gap = mean_correct_token_accuracy - max(
            mean_wrong_token_accuracy, mean_none_token_accuracy
        )
        mean_argument_effect_rate = statistics.fmean(r.argument_effect_rate for r in op_reports)
        mean_steps_trained = statistics.fmean(r.steps_trained for r in op_reports)
        mean_examples_seen = statistics.fmean(r.examples_seen for r in op_reports)
        mean_train_loss = statistics.fmean(r.final_train_loss for r in op_reports)
        mean_encoder_grad_norm = statistics.fmean(r.mean_encoder_grad_norm for r in op_reports)
        mean_operator_grad_norm = statistics.fmean(r.mean_operator_grad_norm for r in op_reports)
        max_task_blind_max_abs_diff = max(r.task_blind_max_abs_diff for r in op_reports)
        task_blind_invariant_passed = all(r.task_blind_invariant_passed for r in op_reports)
        operator_param_count = op_reports[0].operator_param_count

        correct_exact_match_passed = mean_correct >= CORRECT_EXACT_MATCH_THRESHOLD
        token_accuracy_passed = (
            mean_correct_token_accuracy >= TOKEN_ACCURACY_THRESHOLD
            if operation in TOKEN_ACCURACY_GATED_OPERATIONS
            else None
        )
        effectful_wrong_argument_passed = mean_wrong <= EFFECTFUL_WRONG_ARGUMENT_CEILING
        none_passed = mean_none <= NONE_CEILING
        causal_gap_passed = mean_exact_match_causal_gap >= MIN_CAUSAL_GAP
        passed = (
            correct_exact_match_passed
            and (token_accuracy_passed if token_accuracy_passed is not None else True)
            and effectful_wrong_argument_passed
            and none_passed
            and causal_gap_passed
        )

        per_operation_summary[operation] = OperationSharedTrainingSummary(
            operation=operation,
            mean_correct_exact_match=mean_correct,
            stdev_correct_exact_match=stdev_correct,
            min_correct_exact_match=min_correct,
            max_correct_exact_match=max_correct,
            mean_correct_token_accuracy=mean_correct_token_accuracy,
            mean_effectful_wrong_argument_exact_match=mean_wrong,
            mean_effectful_wrong_argument_token_accuracy=mean_wrong_token_accuracy,
            mean_none_exact_match=mean_none,
            mean_none_token_accuracy=mean_none_token_accuracy,
            mean_exact_match_causal_gap=mean_exact_match_causal_gap,
            mean_token_accuracy_causal_gap=mean_token_accuracy_causal_gap,
            mean_argument_effect_rate=mean_argument_effect_rate,
            mean_steps_trained=mean_steps_trained,
            mean_examples_seen=mean_examples_seen,
            mean_train_loss=mean_train_loss,
            mean_encoder_grad_norm=mean_encoder_grad_norm,
            mean_operator_grad_norm=mean_operator_grad_norm,
            max_task_blind_max_abs_diff=max_task_blind_max_abs_diff,
            task_blind_invariant_passed=task_blind_invariant_passed,
            operator_param_count=operator_param_count,
            correct_exact_match_passed=correct_exact_match_passed,
            token_accuracy_passed=token_accuracy_passed,
            effectful_wrong_argument_passed=effectful_wrong_argument_passed,
            none_passed=none_passed,
            causal_gap_passed=causal_gap_passed,
            passed=passed,
        )

    overall_passed = all(summary.passed for summary in per_operation_summary.values())

    return SharedMixedTrainingMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
        core_param_count=per_seed[0].core_param_count,
        passed=overall_passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
