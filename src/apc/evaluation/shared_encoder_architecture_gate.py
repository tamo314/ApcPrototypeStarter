"""Shared queryable representation architecture wiring (Phase A.1 diagnostic
Task A1-R005E-S001, `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

## What this tests

`docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` section 1: A1-R005E-006A
(`apc.evaluation.joint_representation_compact_operator_probe`) showed that
jointly training a task-blind content encoder with the unchanged compact
operator dramatically improves all four parameterized operations -- but that
task trained a *separate* encoder per operation (one fresh `JointStableCore`
per `(seed, operation)` pair in its own per-operation loop). This task builds
the architecture the next milestone (A1-R005E-S002, balanced mixed-operation
training) needs to test whether that gain survives when SHIFT/SELECT/COUNT/
BIND all read from **one shared** encoder instead:

```text
                         +-> Compact SHIFT(amount)
                         +-> Compact SELECT(indices)
content -> Shared Encoder+-> Compact COUNT(target)
                         +-> Compact BIND(key)
```

Per the task's own "Work" list and `docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`'s
"Hard invariant" ("There must be exactly one shared content encoder"), this
module:

1. reuses `apc.evaluation.compact_cross_position_operator_probe.
   CompactCrossPositionOperator` unchanged (imported directly, not
   reimplemented -- same rationale as A1-R005E-006A's own "the operator here
   must be the *identical* class, not merely an analogous one");
2. constructs exactly **one** `SharedContentEncoder` per architecture
   (`build_shared_encoder_architecture` calls the encoder constructor once,
   not once per operation);
3. constructs one dedicated `CompactCrossPositionOperator` per operation in
   `config.operation_names`, all reading from that one encoder;
4. gives every operator its own argument encoder/readout (`CompactCrossPosition
   Operator.__init__` already builds `arg_encoder`/`readout` per instance --
   this module adds a regression test that they are in fact *distinct*
   objects per operation, not accidentally shared);
5. dispatches by **oracle** operation selection: `run_shared_operator` takes
   `operation` as a plain caller-supplied string and does a dict lookup into
   `architecture.operators` -- there is no learned router;
6. never lets task/argument information enter the encoder: `encode_content`'s
   own signature has no `operation`/`argument_values` parameter at all (it
   calls `apc.core.data.collate_content_only_batch`, matching every earlier
   task-blind encoding path in this codebase -- `apc.evaluation.
   task_blind_content_gate`, Task A1-R001).

## No training here

The task's own "Acceptance" is explicit: "Architecture invariants pass. No
milestone benchmark yet." `SharedContentEncoder` is built unfrozen and
freshly initialized (same shape as A1-R005E-006A's own `JointStableCore` --
own copy, not imported, matching this codebase's "each gate module
independently reviewable" convention, e.g. `apc.evaluation.
compact_cross_position_operator_probe`'s own `FrozenStableCore` is not
shared with `frozen_high_capacity_operator_benchmark`'s identically-shaped
class either) so that A1-R005E-S002's balanced mixed-operation training can
backprop into it from every operation -- but this module itself trains
nothing. What it measures instead is whether the wiring satisfies the four
invariants the task's own "Tests" section names:

- one encoder object / parameter set,
- all operations call the same encoder,
- identical content produces identical encoder tokens/states across
  operations,
- no operation-specific modules exist inside the encoder.

Gradient connectivity ("all operations call the same encoder") is checked by
actually backpropagating one tiny loss per operation through `architecture.
core.model` and confirming a non-zero, finite gradient reaches it -- a
functional check, not merely an object-identity one.

## Probe content

`_sample_probe_examples` generates its content via `BindOp`'s own
counterfactual-group protocol (`generate_compact_operator_counterfactual_
groups`, imported unchanged, `operation="BIND"`) purely because `BindOp.
is_valid_for_length` (even length >= 2) is the strictest of the four
operations' own length constraints -- any length it admits is valid for
SHIFT/SELECT/COUNT too. Nothing in this module ever reads the resulting
examples' `.target_tokens`/`.program` (only `.input_tokens`, via `apc.core.
data.collate_content_only_batch`), so reusing BIND's own generated content to
probe all four operators alike is exactly "identical content, every
operation" (Work item 6's own test).

## Diagnostic-only rule

Same as every earlier module in this diagnostic chain: no `Router`,
`PrimitiveBank`, `Primitive`, `PlasticWorkspace`, or `apc.consolidation`/
`apc.meta` module is imported here.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from apc.core.data import collate_content_only_batch
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import SharedCoreTokens, build_shared_core_tokens
from apc.core.train import resolve_device
from apc.environments.generator import Example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.task_spec import default_argument_value_span, num_registered_operations
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    MIN_GROUP_SIZE,
    CompactCrossPositionOperator,
    _default_model_config,
    _flatten_groups,
    generate_compact_operator_counterfactual_groups,
)
from apc.primitives.conditioning import DEFAULT_ARG_DIM, DEFAULT_MAX_SEQUENCE_LENGTH
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "SharedContentEncoder",
    "SharedEncoderArchitectureConfig",
    "shared_encoder_architecture_config_from_dict",
    "SharedEncoderArchitecture",
    "build_shared_encoder_architecture",
    "encode_content",
    "run_shared_operator",
    "encoder_is_free_of_operation_specific_modules",
    "operator_holds_no_encoder_submodule",
    "content_state_max_abs_diff_across_operations",
    "SharedEncoderArchitectureGateReport",
    "run_shared_encoder_architecture_gate",
    "SharedEncoderArchitectureGateMultiSeedReport",
    "run_shared_encoder_architecture_gate_multi_seed",
]

# Same >=5-seed decision-evidence bar every A1-R005D/A1-R005E gate has used
# since A1-R005E-004 (`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`
# section 4: ">=5 seeds for decision evidence"), even though this gate has no
# statistical threshold of its own -- multiple fresh-init seeds still guard
# against a seed-dependent fluke in a purely structural check, matching
# `apc.evaluation.task_blind_content_gate`'s own precedent for an
# architecture-only gate.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5


@dataclass(frozen=True)
class SharedContentEncoder:
    """The ONE task-blind content encoder shared by every operation in this
    diagnostic (`docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`'s "Hard
    invariant"). Unfrozen and freshly initialized -- no separate pretraining
    phase, matching A1-R005E-006A's own `JointStableCore` -- so a future
    mixed-operation training loop (A1-R005E-S002) can backprop into it from
    every operation."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    device: torch.device


@dataclass(frozen=True)
class SharedEncoderArchitectureConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-S001
    architecture-wiring gate run.

    `model` defaults to the same architecture every module in this
    diagnostic chain has used for its content encoder (`d_model=192,
    n_layer=4, n_head=4, d_ff=768`). `d_operator`/`n_operator_head`/
    `d_operator_ff`/`arg_dim`/`max_sequence_length` default to A1-R005E-005's
    own compact-operator values unchanged (Work item 1: "Reuse E-006A
    compact operator architecture unchanged").
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
    # Small: this gate tests architecture invariants, not statistical
    # performance (task Acceptance: "No milestone benchmark yet").
    num_probe_groups: int = 32
    min_probe_examples: int = 48
    atol: float = 1e-5

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
        if self.num_probe_groups < 1:
            raise ValueError(f"num_probe_groups must be >= 1, got {self.num_probe_groups}")
        if self.min_probe_examples < 1:
            raise ValueError(f"min_probe_examples must be >= 1, got {self.min_probe_examples}")
        if self.atol < 0:
            raise ValueError(f"atol must be >= 0, got {self.atol}")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def shared_encoder_architecture_config_from_dict(
    raw: dict[str, Any],
) -> SharedEncoderArchitectureConfig:
    """Parse a `configs/phase_a1_shared_encoder_architecture_gate.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = SharedEncoderArchitectureConfig()
    return SharedEncoderArchitectureConfig(
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
        num_probe_groups=raw.get("num_probe_groups", defaults.num_probe_groups),
        min_probe_examples=raw.get("min_probe_examples", defaults.min_probe_examples),
        atol=raw.get("atol", defaults.atol),
    )


def _build_tokens(config: SharedEncoderArchitectureConfig) -> SharedCoreTokens:
    """Same construction as every other module in this diagnostic chain
    (sized from the full operation registry, not merely
    `config.operation_names`), so `model_vocab_size` stays directly
    comparable across modules at the same `vocab_size`/`sequence_length_range`."""
    return build_shared_core_tokens(
        config.vocab_size,
        num_operations=num_registered_operations(),
        arg_span=default_argument_value_span(config.vocab_size, config.sequence_length_range),
    )


def _build_shared_encoder(config: SharedEncoderArchitectureConfig) -> SharedContentEncoder:
    """Construct the ONE fresh, unfrozen content encoder -- called exactly
    once by `build_shared_encoder_architecture`, never once per operation
    (Work item 2)."""
    device = resolve_device(config.device)
    tokens = _build_tokens(config)
    model_config = TransformerConfig(vocab_size=tokens.model_vocab_size, **config.model)
    model = DecoderOnlyTransformer(model_config).to(device)
    return SharedContentEncoder(model=model, tokens=tokens, device=device)


def _build_operator(
    core: SharedContentEncoder, config: SharedEncoderArchitectureConfig, operation: str
) -> CompactCrossPositionOperator:
    """One operation-specific `CompactCrossPositionOperator` reading from
    `core`'s own `d_model` -- unchanged class/dimensions (Work item 1)."""
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


@dataclass(frozen=True)
class SharedEncoderArchitecture:
    """One shared `SharedContentEncoder` plus one operation-specific compact
    operator per `config.operation_names` -- every operator reads from the
    same `core` object (Work items 2-4)."""

    core: SharedContentEncoder
    operators: dict[str, CompactCrossPositionOperator]
    config: SharedEncoderArchitectureConfig


def build_shared_encoder_architecture(
    config: SharedEncoderArchitectureConfig,
) -> SharedEncoderArchitecture:
    """Build exactly one shared content encoder and one compact operator per
    operation in `config.operation_names`, all wired to that single
    encoder."""
    set_seed(config.seed)
    core = _build_shared_encoder(config)
    # Re-anchor the shared global RNG stream after core construction (ADR-0016
    # pattern), so operator initialization is deterministic regardless of the
    # core's own parameter count.
    set_seed(config.seed)
    operators = {
        operation: _build_operator(core, config, operation) for operation in config.operation_names
    }
    return SharedEncoderArchitecture(core=core, operators=operators, config=config)


def encode_content(
    core: SharedContentEncoder, examples: Sequence[Example]
) -> tuple[torch.Tensor, list[int]]:
    """Task-blind content encoding through the ONE shared encoder.

    This function's own signature has no `operation`/`argument_values`
    parameter -- exactly Work item 6 ("Ensure task/argument never enters the
    encoder"), matching `apc.core.data.collate_content_only_batch`'s own
    task-blind signature. Returns `(content_features[batch, Lmax, d_model],
    content_lengths)`.
    """
    content_lengths = [len(example.input_tokens) for example in examples]
    lmax = max(content_lengths)
    content_ids = collate_content_only_batch(examples, core.tokens, device=core.device)
    content_state = core.model.encode(content_ids)
    content_features = content_state[:, 1 : 1 + lmax, :]
    return content_features, content_lengths


def run_shared_operator(
    architecture: SharedEncoderArchitecture,
    operation: str,
    examples: Sequence[Example],
    argument_values: Sequence[Any] | None,
) -> torch.Tensor:
    """Dispatch through the ONE shared encoder to `operation`'s own compact
    operator.

    `operation` is oracle-selected (Work item 5): it is a plain caller-
    supplied string used for a dict lookup into `architecture.operators` --
    there is no learned router anywhere in this path.

    Raises:
        KeyError: `operation` is not one of `architecture.operators`.
    """
    if operation not in architecture.operators:
        raise KeyError(
            f"operation {operation!r} has no compact operator in this architecture "
            f"(configured operations: {sorted(architecture.operators)})"
        )
    content_features, content_lengths = encode_content(architecture.core, examples)
    output_lengths = [get_operation(operation).output_length(length) for length in content_lengths]
    operator = architecture.operators[operation]
    return operator(content_features, content_lengths, output_lengths, argument_values)


def encoder_is_free_of_operation_specific_modules(
    core: SharedContentEncoder, operation_names: Sequence[str]
) -> tuple[str, ...]:
    """Names of any module or parameter inside the shared encoder that
    references one of `operation_names` (Work item 6 / task "Tests": "no
    operation-specific modules exist inside the encoder"). Empty when the
    invariant holds -- expected always, since `core.model` is a plain
    `apc.core.model.DecoderOnlyTransformer` with no operation awareness; this
    is a regression guard against a future accidental per-operation adapter,
    not a check expected to ever fail today."""
    needles = tuple(name.lower() for name in operation_names)
    hits: set[str] = set()
    for module_name, _ in core.model.named_modules():
        lowered = module_name.lower()
        if module_name and any(needle in lowered for needle in needles):
            hits.add(module_name)
    for param_name, _ in core.model.named_parameters():
        lowered = param_name.lower()
        if any(needle in lowered for needle in needles):
            hits.add(param_name)
    return tuple(sorted(hits))


def operator_holds_no_encoder_submodule(operator: CompactCrossPositionOperator) -> bool:
    """`operator` must never own a `DecoderOnlyTransformer` submodule -- it
    is purely a function of the `content_features` tensor it is called with
    (Work items 2-3: exactly one encoder object, never cloned into an
    operator)."""
    return not any(isinstance(module, DecoderOnlyTransformer) for module in operator.modules())


def content_state_max_abs_diff_across_operations(
    architecture: SharedEncoderArchitecture,
    examples: Sequence[Example],
    operation_names: Sequence[str],
) -> float:
    """Encode the SAME `examples` once per operation in `operation_names` (as
    if invoked from that operation's own forward pass) and return the max
    pairwise absolute difference in the resulting content features.

    Regression-tests the task's own "Tests" bullet ("identical content
    produces identical encoder tokens/states across operations") empirically,
    on top of the structural guarantee `encode_content`'s own operation-blind
    signature already provides -- matching `apc.evaluation.
    task_blind_content_gate`'s own "regression-test this invariant" philosophy
    (module docstring there).
    """
    was_training = architecture.core.model.training
    architecture.core.model.eval()
    with torch.no_grad():
        states = [encode_content(architecture.core, examples)[0] for _ in operation_names]
    if was_training:
        architecture.core.model.train()
    if len(states) < 2:
        return 0.0
    reference = states[0]
    return max(float((state - reference).abs().max().item()) for state in states[1:])


def _dummy_argument_values(operation: str, examples: Sequence[Example]) -> list[Any]:
    """Well-typed (but not semantically "correct") argument values for
    `operation`, one per `examples[i]`'s own content length -- enough for
    `operation`'s own `arg_encoder`/`CompactCrossPositionOperator.forward` to
    run without raising. This gate only tests architecture wiring, not
    prediction correctness (no milestone benchmark yet, task Acceptance), so
    the *values* chosen do not matter as long as they are legal for each
    example's own content length."""
    if operation == "SHIFT":
        return [0 for _ in examples]
    if operation == "SELECT":
        return [
            list(range(get_operation("SELECT").output_length(len(example.input_tokens))))
            for example in examples
        ]
    if operation == "COUNT":
        return [0 for _ in examples]
    if operation == "BIND":
        return [example.input_tokens[0] for example in examples]
    raise ValueError(f"no dummy argument constructor for operation {operation!r}")


def _sample_probe_examples(config: SharedEncoderArchitectureConfig) -> list[Example]:
    """One batch of real `Example`s whose content length is valid for every
    one of SHIFT/SELECT/COUNT/BIND simultaneously (module docstring, "Probe
    content"). Only `.input_tokens` is ever read by anything in this module."""
    groups = generate_compact_operator_counterfactual_groups(
        config.seed,
        config.num_probe_groups,
        operation="BIND",
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )
    examples, _, _ = _flatten_groups(groups)
    return examples


def _encoder_grad_connected_for_operation(
    architecture: SharedEncoderArchitecture,
    operation: str,
    examples: Sequence[Example],
    argument_values: Sequence[Any],
) -> bool:
    """Backprop one tiny loss for `operation` through `architecture` and
    check that a non-zero, finite gradient reaches the ONE shared encoder --
    the functional version of "all operations call the same encoder" (task
    "Tests"), not merely an object-identity check."""
    for parameter in architecture.core.model.parameters():
        parameter.grad = None
    logits = run_shared_operator(architecture, operation, examples, argument_values)
    logits.sum().backward()
    grad = architecture.core.model.token_emb.weight.grad
    connected = grad is not None and bool(torch.isfinite(grad).all()) and bool((grad != 0).any())
    for parameter in architecture.core.model.parameters():
        parameter.grad = None
    for parameter in architecture.operators[operation].parameters():
        parameter.grad = None
    return connected


def _oracle_dispatch_rejects_unknown_operation(
    architecture: SharedEncoderArchitecture, examples: Sequence[Example]
) -> bool:
    """`run_shared_operator` must raise on an operation with no configured
    compact operator (Work item 5: oracle selection is a plain lookup, never
    a silent fallback)."""
    try:
        run_shared_operator(architecture, "NOT_A_CONFIGURED_OPERATION", examples, None)
    except KeyError:
        return True
    return False


@dataclass(frozen=True)
class SharedEncoderArchitectureGateReport:
    """Everything observed while building one seed's shared-encoder
    architecture and checking its structural invariants. No milestone
    benchmark is run (task Acceptance: "No milestone benchmark yet")."""

    config: SharedEncoderArchitectureConfig
    operation_names: tuple[str, ...]
    core_param_count: int
    core_trainable_param_count: int
    operator_param_counts: dict[str, int]
    operators_hold_no_encoder_submodule: bool
    operation_specific_encoder_module_names: tuple[str, ...]
    no_operation_specific_encoder_modules: bool
    per_operation_argument_encoder_distinct: bool
    per_operation_readout_distinct: bool
    num_probe_examples: int
    content_state_max_abs_diff_across_operations: float
    content_state_invariant_across_operations: bool
    per_operation_encoder_grad_connected: dict[str, bool]
    all_operations_encoder_grad_connected: bool
    per_operation_output_shape_valid: dict[str, bool]
    all_output_shapes_valid: bool
    oracle_dispatch_rejects_unknown_operation: bool
    passed: bool
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def run_shared_encoder_architecture_gate(
    config: SharedEncoderArchitectureConfig,
) -> SharedEncoderArchitectureGateReport:
    """Run one seed of the A1-R005E-S001 architecture-wiring gate: build the
    shared-encoder architecture and check every invariant the task's own
    "Tests" section names."""
    start = time.perf_counter()
    architecture = build_shared_encoder_architecture(config)
    core = architecture.core

    operator_param_counts = {
        operation: sum(p.numel() for p in operator.parameters())
        for operation, operator in architecture.operators.items()
    }
    operators_hold_no_encoder_submodule = all(
        operator_holds_no_encoder_submodule(operator)
        for operator in architecture.operators.values()
    )

    operation_specific_names = encoder_is_free_of_operation_specific_modules(
        core, config.operation_names
    )
    no_operation_specific_encoder_modules = len(operation_specific_names) == 0

    arg_encoder_ids = {id(operator.arg_encoder) for operator in architecture.operators.values()}
    readout_ids = {id(operator.readout) for operator in architecture.operators.values()}
    per_operation_argument_encoder_distinct = len(arg_encoder_ids) == len(architecture.operators)
    per_operation_readout_distinct = len(readout_ids) == len(architecture.operators)

    examples = _sample_probe_examples(config)
    if len(examples) < config.min_probe_examples:
        raise ValueError(
            f"probe batch realized only {len(examples)} examples from {config.num_probe_groups} "
            f"groups, below min_probe_examples={config.min_probe_examples}; increase "
            "num_probe_groups"
        )

    content_diff = content_state_max_abs_diff_across_operations(
        architecture, examples, config.operation_names
    )
    content_state_invariant = content_diff <= config.atol

    per_operation_output_shape_valid: dict[str, bool] = {}
    per_operation_encoder_grad_connected: dict[str, bool] = {}
    for operation in config.operation_names:
        argument_values = _dummy_argument_values(operation, examples)
        logits = run_shared_operator(architecture, operation, examples, argument_values)
        content_lengths = [len(example.input_tokens) for example in examples]
        expected_out_max = max(
            get_operation(operation).output_length(length) for length in content_lengths
        )
        per_operation_output_shape_valid[operation] = tuple(logits.shape) == (
            len(examples),
            expected_out_max,
            config.vocab_size,
        )
        per_operation_encoder_grad_connected[operation] = _encoder_grad_connected_for_operation(
            architecture, operation, examples, argument_values
        )

    all_output_shapes_valid = all(per_operation_output_shape_valid.values())
    all_operations_encoder_grad_connected = all(per_operation_encoder_grad_connected.values())
    oracle_dispatch_rejects_unknown_operation = _oracle_dispatch_rejects_unknown_operation(
        architecture, examples
    )

    passed = (
        operators_hold_no_encoder_submodule
        and no_operation_specific_encoder_modules
        and per_operation_argument_encoder_distinct
        and per_operation_readout_distinct
        and content_state_invariant
        and all_operations_encoder_grad_connected
        and all_output_shapes_valid
        and oracle_dispatch_rejects_unknown_operation
    )

    return SharedEncoderArchitectureGateReport(
        config=config,
        operation_names=config.operation_names,
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        operator_param_counts=operator_param_counts,
        operators_hold_no_encoder_submodule=operators_hold_no_encoder_submodule,
        operation_specific_encoder_module_names=operation_specific_names,
        no_operation_specific_encoder_modules=no_operation_specific_encoder_modules,
        per_operation_argument_encoder_distinct=per_operation_argument_encoder_distinct,
        per_operation_readout_distinct=per_operation_readout_distinct,
        num_probe_examples=len(examples),
        content_state_max_abs_diff_across_operations=content_diff,
        content_state_invariant_across_operations=content_state_invariant,
        per_operation_encoder_grad_connected=per_operation_encoder_grad_connected,
        all_operations_encoder_grad_connected=all_operations_encoder_grad_connected,
        per_operation_output_shape_valid=per_operation_output_shape_valid,
        all_output_shapes_valid=all_output_shapes_valid,
        oracle_dispatch_rejects_unknown_operation=oracle_dispatch_rejects_unknown_operation,
        passed=passed,
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


@dataclass(frozen=True)
class SharedEncoderArchitectureGateMultiSeedReport:
    """A1-R005E-S001's verdict aggregated across seeds. `passed` is a
    per-seed-then-ANDed verdict (every seed's own report must pass), matching
    `apc.evaluation.task_blind_content_gate`'s convention for a gate with no
    mean-based threshold."""

    seeds: tuple[int, ...]
    per_seed: tuple[SharedEncoderArchitectureGateReport, ...]
    max_content_state_absolute_difference: float
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "max_content_state_absolute_difference": self.max_content_state_absolute_difference,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_shared_encoder_architecture_gate_multi_seed(
    base_config: SharedEncoderArchitectureConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> SharedEncoderArchitectureGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 multi-seed gate."""
    per_seed: list[SharedEncoderArchitectureGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        report = run_shared_encoder_architecture_gate(config)
        if run_dir_path is not None:
            seed_dir = run_dir_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    max_diff = max(report.content_state_max_abs_diff_across_operations for report in per_seed)

    return SharedEncoderArchitectureGateMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        max_content_state_absolute_difference=max_diff,
        passed=all(report.passed for report in per_seed),
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
