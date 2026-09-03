"""Joint task-blind representation + compact operator probe (Phase A.1
diagnostic Task A1-R005E-006A, `docs/CODEX_TASKS_A1_R005E_E006_PLUS.md`).

## What this tests

`docs/design-docs/REPRESENTATION_OPERATOR_FACTORIAL.md` section 1's C10 cell:
does A1-R005E-005's own compact operator (`apc.evaluation.
compact_cross_position_operator_probe.CompactCrossPositionOperator`, reused
here unchanged -- same class, same default dimensions) become effective once
the task-blind content encoder it reads from is trained *jointly* with it,
instead of pretrained on a separate objective and then frozen?

`docs/AGENTS_A1_R005E_E006_PLUS_ADDENDUM.md`'s "Primary causal control":

```text
Frozen task-blind encoder -> Trainable task-blind encoder
```

with everything else held fixed: the same compact operator architecture and
size, the same counterfactual-group protocol, the same Correct/effectful
Wrong argument/None arms, the same fresh-readout convention. Comparing this
task's result (`C10`) against A1-R005E-005's own measured `C00` (frozen
representation, same compact operator) and A1-R005E-004's own measured `C01`
(frozen representation, high-capacity operator) is the `R_access`
representation-accessibility-recovery diagnostic (design doc section 3).

## Why there is no separate core-pretraining phase

A1-R005E-002/003/004/005 all pretrain a dedicated `DecoderOnlyTransformer`
on a *separate* task-blind objective (`apc.evaluation.
shared_core_generalization.train_shared_core`, `include_task_spec=False`,
`steps=30000`) and then freeze it before ever training an operator on top --
"frozen" is exactly that two-phase shape (pretrain-then-freeze). The
factorial design's own definition of "joint" (design doc section 1: "task-
blind content encoder trained jointly with operator") is not "continue
fine-tuning an already-pretrained encoder" but a single training loop: the
content encoder starts from a fresh random initialization (same architecture
and size as every other module in this diagnostic chain -- `d_model=192,
n_layer=4, n_head=4, d_ff=768`, matching `apc.evaluation.
compact_cross_position_operator_probe`'s own `core_train`/model config) and
receives its *only* training signal from the compact operator's own
downstream cross-entropy loss, backpropagated through `DecoderOnlyTransformer
.encode` on every joint-training step. This is the literal reading of "C00
-> C10 changes representation trainability while compact compute stays
fixed" (design doc section 2): the encoder's *architecture* and the
operator's *architecture/budget* are both held fixed; only whether the
encoder's own parameters ever receive a gradient is changed. Consequently
this module never imports `apc.evaluation.shared_core_generalization`.

`joint_train.steps` therefore defaults to `38000` -- the same *total* number
of gradient steps A1-R005E-005 spent across its two phases combined
(`core_train.steps=30000 + operator_train.steps=8000`), so this task's own
"training budget may change because the encoder is now trainable, but
operator capacity may not" (`docs/EXPERIMENT_PLAN_A1_R005E_E006_PLUS.md`
section 3) is satisfied by giving the freshly-initialized encoder the same
total optimization budget A1-R005E-005's encoder implicitly had (its own
30000-step pretraining phase), not a smaller one -- while leaving the
compact operator's own architecture and per-step training exactly as
A1-R005E-005 defined it.

## Reused vs. new

Per the task's own "Critical control" ("Reuse the same compact operator
class and dimensions as E-005. Do not make the operator stronger."), this
module imports `CompactCrossPositionOperator` and the entire counterfactual-
group generation protocol (`generate_compact_operator_counterfactual_groups`,
`CompactOperatorGroup`, `_flatten_groups`, `_correct_argument_value`, and the
pure per-operation helpers) directly from `apc.evaluation.
compact_cross_position_operator_probe` rather than re-implementing a second
copy, unlike that module's own precedent of duplicating `apc.evaluation.
frozen_high_capacity_operator_benchmark`'s private helpers. That precedent
existed to keep two *different* operator implementations independently
reviewable; here the operator is required to be the *identical* class, so
importing it directly is a stronger, zero-drift guarantee of "do not make
the operator stronger" than a second hand-copied implementation could be.

Genuinely new in this module: `JointStableCore` (an unfrozen, from-scratch
content encoder), `_batch_features` (a grad-enabled variant), `_evaluate_arm`
(toggles `core.model.eval()`/`train()` in addition to the operator's own,
since the encoder is no longer permanently frozen in eval mode), `_train_joint`
(one optimizer over encoder + operator parameters, with the two parameter
groups' own gradient norms tracked and clipped independently so they are
separately observable as this task's own "encoder/operator gradient
summaries" metric), the task-blind invariance regression check (design doc
section 4), and the `R_access` computation against A1-R005E-004/005's own
saved `summary.json` files (task "Work" step 7: "Load saved E-004/E-005
summaries and compute C00/C01/C10 comparisons").

## Task-blind invariance

`collate_content_only_batch` (`apc.core.data`) renders `[BOS] input... [SEP]`
with no task/argument token anywhere in the sequence -- this is a structural
property of the function's own signature (it takes no `TaskSpec`), true
regardless of what the encoder learns, exactly like every other task-blind
encoding path in this codebase (`apc.evaluation.task_blind_content_gate`,
Task A1-R001). `_task_blind_invariance_max_abs_diff` below regression-tests
this empirically on the *trained* joint encoder (design doc section 4:
"Regression-test this invariant"), rather than only relying on the
structural argument: for one held-out counterfactual group (same content,
several different argument values/targets), the encoder's own `h_content`
must be numerically identical across every group member within
`TASK_BLIND_ATOL` (`1e-5`, matching `apc.evaluation.task_blind_content_gate.
TaskBlindContentGateConfig`'s own default `atol`).

## Diagnostic-only rule / branch decision is not made here

Same as every earlier module in this diagnostic chain: no `Router`,
`PrimitiveBank`, `Primitive`, `PlasticWorkspace`, or `apc.consolidation`/
`apc.meta` module is imported here, and this module computes measurements
and threshold pass/fail flags only -- it does not select the next research
phase. `R_access` values are reported as raw numbers; interpreting them
against the design doc's qualitative bands (`>=0.70` strong, `[0.30, 0.70)`
mixed, `<0.30` operator-bottleneck-favored) is left to `docs/DECISIONS.md`'s
ADR for this task.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.core.train import resolve_device
from apc.environments.generator import Example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.task_spec import default_argument_value_span, num_registered_operations
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
    CompactCrossPositionOperator,
    CompactOperatorGroup,
    _correct_argument_value,
    _default_model_config,
    _flatten_groups,
    _labels_for_examples,
    generate_compact_operator_counterfactual_groups,
)
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.primitives.conditioning import DEFAULT_ARG_DIM, DEFAULT_MAX_SEQUENCE_LENGTH
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "TASK_BLIND_ATOL",
    "R_ACCESS_EPS",
    "DEFAULT_E004_SUMMARY_PATH",
    "DEFAULT_E005_SUMMARY_PATH",
    "JointStableCore",
    "JointOperatorConfig",
    "joint_operator_config_from_dict",
    "representation_accessibility_recovery",
    "OperationJointOperatorReport",
    "JointOperatorReport",
    "run_joint_operator_probe",
    "OperationJointOperatorSummary",
    "JointOperatorMultiSeedReport",
    "run_joint_operator_probe_multi_seed",
]

# Same >=5-seed decision-evidence bar every A1-R005D/A1-R005E counterfactual
# gate has used since A1-R005E-004 (`docs/EXPERIMENT_PLAN_A1_R005E_E006_PLUS.md`
# section 3: "Decision evidence requires >=5 seeds").
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5

# Matches apc.evaluation.task_blind_content_gate.TaskBlindContentGateConfig's
# own default `atol` for the same kind of structural task-blind invariance
# check.
TASK_BLIND_ATOL = 1e-5

# design doc section 3's `R_access = (m_C10 - m_C00) / max(eps, m_C01 - m_C00)`.
R_ACCESS_EPS = 1e-6

_EVAL_BATCH_SIZE = 256

# Standard run-dir locations for A1-R005E-004/005's own saved summaries
# (`docs/CODEX_TASKS_A1_R005E_E006_PLUS.md` A1-R005E-006A "Work" step 7:
# "Load saved E-004/E-005 summaries"). Callers may override with any other
# path (or `None` to skip R_access entirely -- see
# `run_joint_operator_probe_multi_seed`).
DEFAULT_E004_SUMMARY_PATH = Path(
    "runs/phase_a1_frozen_high_capacity_operator_benchmark/summary.json"
)
DEFAULT_E005_SUMMARY_PATH = Path(
    "runs/phase_a1_compact_cross_position_operator_probe/summary.json"
)


@dataclass(frozen=True)
class JointStableCore:
    """A task-blind content encoder that stays trainable throughout, unlike
    `apc.evaluation.compact_cross_position_operator_probe.FrozenStableCore`.
    No parameter is ever frozen and there is no separate pretraining phase
    (module docstring) -- the only supervision `model` ever receives is the
    compact operator's own downstream loss, backpropagated through `encode`.
    """

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    device: torch.device


@dataclass(frozen=True)
class JointOperatorConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-006A
    benchmark run, covering every operation in `operation_names`.

    `model` defaults to the exact same architecture A1-R005E-002/003/004/005
    used for their own frozen core (`d_model=192, n_layer=4, n_head=4,
    d_ff=768`) -- unchanged size, only trainability differs (module
    docstring). `d_operator`/`n_operator_head`/`d_operator_ff`/`arg_dim`/
    `max_sequence_length` default to A1-R005E-005's own compact-operator
    values unchanged (the task's own "Critical control": "Do not make the
    operator stronger").
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
    # Sum of A1-R005E-005's own two-phase budget (core_train.steps=30000 +
    # operator_train.steps=8000) -- see module docstring "Why...steps
    # defaults to 38000".
    joint_train: PrimitiveTrainConfig = field(
        default_factory=lambda: PrimitiveTrainConfig(steps=38000)
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


def joint_operator_config_from_dict(raw: dict[str, Any]) -> JointOperatorConfig:
    """Parse a `configs/phase_a1_joint_representation_compact_operator_probe.
    yaml`-shaped dict, matching every other Phase A.1 gate's convention of
    filling in defaults for whatever the file omits."""
    defaults = JointOperatorConfig()
    joint_train_raw = raw.get("joint_train")
    joint_train = (
        dataclasses.replace(defaults.joint_train, **joint_train_raw)
        if joint_train_raw
        else defaults.joint_train
    )
    return JointOperatorConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        device=raw.get("device", defaults.device),
        d_operator=raw.get("d_operator", defaults.d_operator),
        n_operator_head=raw.get("n_operator_head", defaults.n_operator_head),
        d_operator_ff=raw.get("d_operator_ff", defaults.d_operator_ff),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        joint_train=joint_train,
        num_unseen_eval_groups=raw.get("num_unseen_eval_groups", defaults.num_unseen_eval_groups),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


def representation_accessibility_recovery(
    m_joint: float, m_frozen_compact: float, m_frozen_high_cap: float, *, eps: float = R_ACCESS_EPS
) -> float:
    """`R_access = (m_C10 - m_C00) / max(eps, m_C01 - m_C00)`
    (`docs/design-docs/REPRESENTATION_OPERATOR_FACTORIAL.md` section 3):
    `m_joint` is this task's own measured value (`C10`, joint representation
    + compact operator), `m_frozen_compact` is A1-R005E-005's (`C00`, frozen
    representation + compact operator), `m_frozen_high_cap` is
    A1-R005E-004's (`C01`, frozen representation + high-capacity operator).
    Meaningful only when `m_frozen_high_cap` materially exceeds
    `m_frozen_compact` -- interpretation, not gated by this function, is left
    to the caller (module docstring, "Branch decision is not made here")."""
    return (m_joint - m_frozen_compact) / max(eps, m_frozen_high_cap - m_frozen_compact)


def _load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _operation_metric(summary: dict[str, Any], operation: str, key: str) -> float:
    per_operation = summary.get("per_operation_summary", {})
    if operation not in per_operation:
        raise KeyError(f"summary has no per_operation_summary entry for {operation!r}")
    return float(per_operation[operation][key])


def _build_tokens(config: JointOperatorConfig) -> SharedCoreTokens:
    """Same construction as `apc.evaluation.shared_core_generalization.
    _build_tokens` (sized from the full operation registry, not merely
    `config.operation_names`), so `model_vocab_size` -- and therefore
    `core_param_count` -- is directly comparable to every earlier module in
    this diagnostic chain at the same `vocab_size`/`sequence_length_range`."""
    return build_shared_core_tokens(
        config.vocab_size,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(config.vocab_size, config.sequence_length_range),
    )


def _build_joint_core(config: JointOperatorConfig) -> JointStableCore:
    """Construct a fresh, unfrozen `DecoderOnlyTransformer` -- no pretraining
    call, no `requires_grad_(False)` (module docstring: "Why there is no
    separate core-pretraining phase")."""
    device = resolve_device(config.device)
    tokens = _build_tokens(config)
    model_config = TransformerConfig(vocab_size=tokens.model_vocab_size, **config.model)
    model = DecoderOnlyTransformer(model_config).to(device)
    return JointStableCore(model=model, tokens=tokens, device=device)


def _build_operator(
    core: JointStableCore, config: JointOperatorConfig, operation: str
) -> CompactCrossPositionOperator:
    operator = CompactCrossPositionOperator(
        operation,
        d_model=core.model.config.d_model,
        d_operator=config.d_operator,
        n_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    )
    return operator.to(core.device)


def _batch_features(
    core: JointStableCore, examples: Sequence[Example], operation: str, *, no_grad: bool
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Per-position content features for `examples`, exactly like `apc.
    evaluation.compact_cross_position_operator_probe._batch_features`, except
    `no_grad` is a caller-controlled switch rather than always-on: joint
    training needs gradients to reach `core.model`, evaluation does not."""
    content_lengths = [len(example.input_tokens) for example in examples]
    lmax = max(content_lengths)
    output_lengths = [get_operation(operation).output_length(length) for length in content_lengths]
    content_ids = collate_content_only_batch(examples, core.tokens, device=core.device)
    if no_grad:
        with torch.no_grad():
            content_state = core.model.encode(content_ids)
    else:
        content_state = core.model.encode(content_ids)
    content_features = content_state[:, 1 : 1 + lmax, :]
    return content_features, content_lengths, output_lengths


def _evaluate_arm(
    core: JointStableCore,
    operator: CompactCrossPositionOperator,
    examples: Sequence[Example],
    operation: str,
    *,
    argument_provider: Callable[[Example], Any] | None,
) -> tuple[float, float]:
    """Exact match / token accuracy for one causal-ablation arm. Toggles both
    `core.model` and `operator` between eval/train, unlike A1-R005E-005's own
    `_evaluate_arm` (which never trains `core.model` at all)."""
    core.model.eval()
    operator.eval()
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), _EVAL_BATCH_SIZE):
            chunk = examples[start : start + _EVAL_BATCH_SIZE]
            content_features, content_lengths, output_lengths = _batch_features(
                core, chunk, operation, no_grad=True
            )
            argument_values = (
                None
                if argument_provider is None
                else [argument_provider(example) for example in chunk]
            )
            logits = operator(content_features, content_lengths, output_lengths, argument_values)
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


def _param_group_grad_norm(parameters: Sequence[nn.Parameter], grad_clip: float) -> float:
    """Total gradient norm for one parameter group, clipping it in place
    when `grad_clip > 0` (else just measuring, via `max_norm=inf`, which
    `clip_grad_norm_`'s own `min(clip_coef, 1.0)` clamp never scales down).
    Encoder and operator parameters are clipped as two *independent* groups
    (two separate calls), not one combined vector -- deliberately, so their
    own norms are separately observable as this task's own "encoder/operator
    gradient summaries" metric (module docstring)."""
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    return float(nn.utils.clip_grad_norm_(parameters, max_norm))


def _train_joint(
    core: JointStableCore,
    operator: CompactCrossPositionOperator,
    config: JointOperatorConfig,
    operation: str,
    metrics_path: str | Path | None = None,
) -> tuple[float, int, float, float]:
    """Jointly train `core.model` and `operator` end to end on counterfactual
    groups' Correct-argument targets -- the operator's own downstream loss is
    `core.model`'s only training signal (module docstring). Returns
    `(final_loss, total_examples_seen, final_encoder_grad_norm,
    final_operator_grad_norm)`."""
    device = core.device
    encoder_params = list(core.model.parameters())
    operator_params = list(operator.parameters())
    optimizer = torch.optim.AdamW(
        encoder_params + operator_params,
        lr=config.joint_train.lr,
        weight_decay=config.joint_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after optimizer construction,
    # per ADR-0016's pattern.
    set_seed(config.seed)

    groups_per_step = max(1, -(-config.joint_train.batch_size // config.group_size))

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    core.model.train()
    operator.train()
    final_loss = float("nan")
    final_encoder_grad_norm = float("nan")
    final_operator_grad_norm = float("nan")
    total_examples_seen = 0
    try:
        for step in range(config.joint_train.steps):
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
            total_examples_seen += len(examples)

            content_features, content_lengths, output_lengths = _batch_features(
                core, examples, operation, no_grad=False
            )
            argument_values = [_correct_argument_value(operation, example) for example in examples]
            out_max = max(output_lengths)
            labels = _labels_for_examples(examples, output_lengths, out_max, device)

            optimizer.zero_grad(set_to_none=True)
            logits = operator(content_features, content_lengths, output_lengths, argument_values)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
            loss.backward()
            encoder_grad_norm = _param_group_grad_norm(encoder_params, config.joint_train.grad_clip)
            operator_grad_norm = _param_group_grad_norm(
                operator_params, config.joint_train.grad_clip
            )
            optimizer.step()
            final_loss = float(loss.item())
            final_encoder_grad_norm = encoder_grad_norm
            final_operator_grad_norm = operator_grad_norm

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.joint_train.eval_every == 0
                or step_number == config.joint_train.steps
            ):
                progress_groups = generate_compact_operator_counterfactual_groups(
                    config.seed,
                    max(
                        1,
                        -(-config.joint_train.progress_eval_examples // config.group_size),
                    ),
                    operation=operation,
                    step=step,
                    split="val",
                    vocab_size=config.vocab_size,
                    sequence_length_range=config.sequence_length_range,
                    group_size=config.group_size,
                )
                progress_examples, _, _ = _flatten_groups(progress_groups)
                progress_exact_match, _ = _evaluate_arm(
                    core,
                    operator,
                    progress_examples,
                    operation,
                    argument_provider=lambda example: _correct_argument_value(operation, example),
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "loss": final_loss,
                            "encoder_grad_norm": encoder_grad_norm,
                            "operator_grad_norm": operator_grad_norm,
                            "progress_correct_exact_match": progress_exact_match,
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
                # _evaluate_arm restores operator.train()/core.model.train()
                # on its own; nothing further to do here.
    finally:
        if metrics_file is not None:
            metrics_file.close()

    return final_loss, total_examples_seen, final_encoder_grad_norm, final_operator_grad_norm


def _task_blind_invariance_max_abs_diff(
    core: JointStableCore, operation: str, group: CompactOperatorGroup
) -> float:
    """Max absolute pairwise difference in `h_content` across `group`'s own
    members -- same content, different (task-blind-invisible) argument
    values/targets -- empirically regression-testing design doc section 4's
    `E_joint(x, t1) == E_joint(x, t2)` invariant on the *trained* joint
    encoder (module docstring "Task-blind invariance"), on top of the
    structural guarantee `collate_content_only_batch` already provides."""
    core.model.eval()
    content_features, _, _ = _batch_features(core, list(group.examples), operation, no_grad=True)
    core.model.train()
    reference = content_features[0:1]
    return float((content_features - reference).abs().max().item())


@dataclass(frozen=True)
class OperationJointOperatorReport:
    """Everything observed while jointly training the content encoder and
    the one `CompactCrossPositionOperator` on top of it, and evaluating one
    seed's causal ablation (Correct / effectful Wrong argument / None) for
    one operation."""

    operation: str
    core_param_count: int
    core_trainable_param_count: int
    operator_param_count: int
    joint_steps_trained: int
    joint_examples_seen: int
    final_train_loss: float
    final_encoder_grad_norm: float
    final_operator_grad_norm: float
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
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def _run_operation_benchmark(
    config: JointOperatorConfig,
    operation: str,
    *,
    joint_metrics_path: str | Path | None = None,
) -> OperationJointOperatorReport:
    """Run the full A1-R005E-006A benchmark for one `(config.seed,
    operation)` pair: construct a fresh unfrozen content encoder, jointly
    train it with one `CompactCrossPositionOperator` on counterfactual
    groups, then evaluate Correct/effectful Wrong argument/None on a large,
    independently-drawn unseen batch of counterfactual groups."""
    start = time.perf_counter()
    core = _build_joint_core(config)
    operator = _build_operator(core, config, operation)
    (
        final_train_loss,
        examples_seen,
        final_encoder_grad_norm,
        final_operator_grad_norm,
    ) = _train_joint(core, operator, config, operation, metrics_path=joint_metrics_path)

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
            f"unseen eval batch realized only {len(unseen_examples)} examples from "
            f"{config.num_unseen_eval_groups} groups, below min_unseen_eval_examples="
            f"{config.min_unseen_eval_examples}; increase num_unseen_eval_groups"
        )

    correct_exact_match, correct_token_accuracy = _evaluate_arm(
        core,
        operator,
        unseen_examples,
        operation,
        argument_provider=lambda example: _correct_argument_value(operation, example),
    )
    wrong_exact_match, wrong_token_accuracy = _evaluate_arm(
        core,
        operator,
        unseen_examples,
        operation,
        argument_provider=lambda example: wrong_argument_by_id[id(example)],
    )
    none_exact_match, none_token_accuracy = _evaluate_arm(
        core, operator, unseen_examples, operation, argument_provider=None
    )

    argument_effect_rate = statistics.fmean(
        float(example.target_tokens != wrong_target_tokens_by_id[id(example)])
        for example in unseen_examples
    )
    exact_match_causal_gap = correct_exact_match - max(wrong_exact_match, none_exact_match)
    token_accuracy_causal_gap = correct_token_accuracy - max(
        wrong_token_accuracy, none_token_accuracy
    )

    task_blind_max_abs_diff = _task_blind_invariance_max_abs_diff(core, operation, unseen_groups[0])
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

    return OperationJointOperatorReport(
        operation=operation,
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        operator_param_count=sum(p.numel() for p in operator.parameters()),
        joint_steps_trained=config.joint_train.steps,
        joint_examples_seen=examples_seen,
        final_train_loss=final_train_loss,
        final_encoder_grad_norm=final_encoder_grad_norm,
        final_operator_grad_norm=final_operator_grad_norm,
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
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


@dataclass(frozen=True)
class JointOperatorReport:
    """Everything observed for one seed, across every operation in
    `config.operation_names`."""

    config: JointOperatorConfig
    per_operation: dict[str, OperationJointOperatorReport]
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "per_operation": {
                name: report.to_dict() for name, report in self.per_operation.items()
            },
            "wall_clock_seconds": self.wall_clock_seconds,
            "device": self.device,
        }


def run_joint_operator_probe(
    config: JointOperatorConfig,
    *,
    metrics_dir: str | Path | None = None,
) -> JointOperatorReport:
    """Run one seed of the A1-R005E-006A benchmark: for every operation in
    `config.operation_names`, construct a fresh unfrozen content encoder,
    jointly train it with its own dedicated `CompactCrossPositionOperator`,
    and evaluate the three-arm causal ablation."""
    start = time.perf_counter()
    per_operation: dict[str, OperationJointOperatorReport] = {}
    device_str = "cpu"
    metrics_dir_path = Path(metrics_dir) if metrics_dir is not None else None
    for operation in config.operation_names:
        joint_metrics_path = (
            metrics_dir_path / f"{operation}_joint_metrics.jsonl" if metrics_dir_path else None
        )
        report = _run_operation_benchmark(
            config, operation, joint_metrics_path=joint_metrics_path
        )
        per_operation[operation] = report
        device_str = report.device
    return JointOperatorReport(
        config=config,
        per_operation=per_operation,
        wall_clock_seconds=time.perf_counter() - start,
        device=device_str,
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class OperationJointOperatorSummary:
    """One operation's three-arm means aggregated across seeds, with
    pass/fail verdicts computed on the cross-seed mean (`docs/DECISIONS.md`
    ADR-0033's precedent), plus the `R_access` representation-accessibility-
    recovery diagnostic against A1-R005E-004/005's own saved summaries
    (`None` when those were not supplied -- see
    `run_joint_operator_probe_multi_seed`)."""

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
    max_task_blind_max_abs_diff: float
    task_blind_invariant_passed: bool
    operator_param_count: int
    core_param_count: int
    c00_frozen_compact_correct_exact_match: float | None
    c01_frozen_high_cap_correct_exact_match: float | None
    c00_frozen_compact_causal_gap: float | None
    c01_frozen_high_cap_causal_gap: float | None
    r_access_correct_exact_match: float | None
    r_access_causal_gap: float | None
    correct_exact_match_passed: bool
    token_accuracy_passed: bool | None
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class JointOperatorMultiSeedReport:
    """A1-R005E-006A's full result across seeds, with an overall `passed`
    verdict (every operation in `seeds`' shared `operation_names` passes its
    own targets). Interpreting these numbers -- including `R_access` -- and
    selecting a branch is left to `docs/DECISIONS.md`'s ADR for this task
    (module docstring, "Branch decision is not made here")."""

    seeds: tuple[int, ...]
    per_seed: tuple[JointOperatorReport, ...]
    per_operation_summary: dict[str, OperationJointOperatorSummary]
    e004_summary_path: str | None
    e005_summary_path: str | None
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "per_operation_summary": {
                name: summary.to_dict() for name, summary in self.per_operation_summary.items()
            },
            "e004_summary_path": self.e004_summary_path,
            "e005_summary_path": self.e005_summary_path,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_joint_operator_probe_multi_seed(
    base_config: JointOperatorConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    e004_summary_path: str | Path | None = DEFAULT_E004_SUMMARY_PATH,
    e005_summary_path: str | Path | None = DEFAULT_E005_SUMMARY_PATH,
) -> JointOperatorMultiSeedReport:
    """Run one benchmark seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation. Does not raise if
    `len(seeds) < MIN_GATE_SEEDS`; `meets_seed_policy` reports it honestly
    instead, matching every other Phase A.1 gate.

    `e004_summary_path`/`e005_summary_path` locate A1-R005E-004's/005's own
    saved `summary.json` (task "Work" step 7). Pass `None` for either to
    skip `R_access` entirely (every `OperationJointOperatorSummary.r_access_*`
    field is then `None`) -- the default paths point at this repository's own
    standard run-dir locations and raise `FileNotFoundError` if missing,
    since a real A1-R005E-006A milestone run requires them."""
    per_seed: list[JointOperatorReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        report = run_joint_operator_probe(config, metrics_dir=seed_dir)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    e004_summary = _load_summary(e004_summary_path) if e004_summary_path is not None else None
    e005_summary = _load_summary(e005_summary_path) if e005_summary_path is not None else None

    per_operation_summary: dict[str, OperationJointOperatorSummary] = {}
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
        max_task_blind_max_abs_diff = max(r.task_blind_max_abs_diff for r in op_reports)
        task_blind_invariant_passed = all(r.task_blind_invariant_passed for r in op_reports)
        operator_param_count = op_reports[0].operator_param_count
        core_param_count = op_reports[0].core_param_count

        c00_correct = c01_correct = c00_gap = c01_gap = None
        r_access_correct = r_access_causal_gap = None
        if e005_summary is not None and e004_summary is not None:
            c00_correct = _operation_metric(e005_summary, operation, "mean_correct_exact_match")
            c01_correct = _operation_metric(e004_summary, operation, "mean_correct_exact_match")
            c00_gap = _operation_metric(e005_summary, operation, "mean_exact_match_causal_gap")
            c01_gap = _operation_metric(e004_summary, operation, "mean_exact_match_causal_gap")
            r_access_correct = representation_accessibility_recovery(
                mean_correct, c00_correct, c01_correct
            )
            r_access_causal_gap = representation_accessibility_recovery(
                mean_exact_match_causal_gap, c00_gap, c01_gap
            )

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

        per_operation_summary[operation] = OperationJointOperatorSummary(
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
            max_task_blind_max_abs_diff=max_task_blind_max_abs_diff,
            task_blind_invariant_passed=task_blind_invariant_passed,
            operator_param_count=operator_param_count,
            core_param_count=core_param_count,
            c00_frozen_compact_correct_exact_match=c00_correct,
            c01_frozen_high_cap_correct_exact_match=c01_correct,
            c00_frozen_compact_causal_gap=c00_gap,
            c01_frozen_high_cap_causal_gap=c01_gap,
            r_access_correct_exact_match=r_access_correct,
            r_access_causal_gap=r_access_causal_gap,
            correct_exact_match_passed=correct_exact_match_passed,
            token_accuracy_passed=token_accuracy_passed,
            effectful_wrong_argument_passed=effectful_wrong_argument_passed,
            none_passed=none_passed,
            causal_gap_passed=causal_gap_passed,
            passed=passed,
        )

    overall_passed = all(summary.passed for summary in per_operation_summary.values())

    return JointOperatorMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
        e004_summary_path=str(e004_summary_path) if e004_summary_path is not None else None,
        e005_summary_path=str(e005_summary_path) if e005_summary_path is not None else None,
        passed=overall_passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
