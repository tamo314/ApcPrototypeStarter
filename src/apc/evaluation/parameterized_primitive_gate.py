"""Parameterized oracle primitive benchmark (Phase A.1 Post-Correction Task
A1-R005, STOP GATE, H2c).

`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R005: for
`apc.environments.operations.PARAMETERIZED_OPERATION_NAMES` (`SHIFT`,
`SELECT`, `COUNT`, `BIND` -- the four operations with a hidden per-instance
argument, `docs/DECISIONS.md` ADR-0017), register one argument-conditioned
neural primitive family per operation (`apc.primitives.conditioning.
ConditionedPrimitive`, Task A1-R004), freeze a pretrained task-blind Stable
Core, train the primitives via oracle-forced routing that actually threads
each example's own argument through (Task A1-R005's addition to
`apc.core.execution._route_and_apply_oracle_calls`), and measure the causal
ablation matrix's *four* arms (`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`
section 8): **Correct** (correct family, correct argument), **Wrong
argument** (correct family, a deliberately different argument), **Wrong
family** (an oracle-forced incorrect family), and **None** (no primitive at
all). A parameterized primitive is causally supported only if Correct is
high while every control is materially lower (`AGENTS.md`'s causal primitive
evidence rule, extended by section 6's "For parameterized primitives also
test: Wrong argument").

This module mirrors `apc.evaluation.parameter_free_primitive_gate` (A1-R003)
structurally -- same pretrain-then-freeze Stable Core strategy, same
oracle-forced training loop, same multi-seed aggregation shape -- because
A1-R005 is that same experiment design applied to the argument-conditioned
half of the operation pool, not a different one. It duplicates rather than
imports that module's private helpers (`_pretrain_frozen_stable_core`,
`_default_wrong_operation_map`, ...): there is no existing precedent in this
codebase for one gate module reaching into another's private functions
(every earlier Phase A.1 gate is self-contained), and each STOP GATE module
is meant to be independently reviewable end to end. The one exception is
`PrimitiveTrainConfig`, imported directly: it is already `parameter_free_
primitive_gate.__all__`'s public, gate-agnostic serializable config for
"the optimization budget for training a primitive bank on top of one seed's
frozen Stable Core" -- exactly what this gate also needs, with nothing
A1-R003-specific about its shape.

## How the Stable Core gets frozen

Identical strategy to A1-R003 (see that module's own docstring for the full
argument): `_pretrain_frozen_stable_core` reuses `apc.evaluation.
shared_core_generalization.train_shared_core` with `include_task_spec=False`
and `operation_names` restricted to `config.operation_names` (here,
`PARAMETERIZED_OPERATION_NAMES` by default), then freezes every parameter.
Task-blind pretraining over a *mixed* SHIFT/SELECT/COUNT/BIND stream is, if
anything, a harder non-identifiability setting than A1-R003's four
parameter-free operations: not only is the operation identity hidden from
the core, so is each hidden-parameter operation's own argument (ADR-0017) --
the core cannot resolve *which* operation applies, let alone *which*
argument, and is not asked to. It only needs to develop generically useful
token/position/content representation and cross-position aggregation
(BIND's associative lookup in particular needs the frozen core's causal
self-attention to have learned some transferable key/value-matching
capacity) for the argument-conditioned primitives to build on.

## Training primitives through the frozen core

`apc.core.execution.forward_logits_with_oracle_calls_trainable` (A1-R003's
addition, now argument-aware per A1-R005's `_route_and_apply_oracle_calls`
dispatch) is used exactly as A1-R003 uses it: run the frozen `encode`, force
each example's own correct `PrimitiveCall` (operation *and* arguments, from
`apc.environments.generator.oracle_call_for_example`) via
`apply_bank_with_oracle_calls_trainable`, then `decode` -- with a gradient
path back into the selected `ConditionedPrimitive`'s `a_proj`/`b_proj`/
`c_proj`/`argument_encoder` weights (never the frozen core, never an
unselected family). The optimizer step is scoped to `bank.parameters()`
only, identical to A1-R003.

## Correct / Wrong argument / Wrong family / None

- **Correct**: `apc.core.execution.evaluate_exact_match_with_oracle_calls`
  with the default `oracle_call_for_example` provider -- the example's own
  correct operation *and* argument.
- **Wrong argument**: the same function with a provider that keeps the
  correct operation but substitutes a deterministically different, materially
  different argument (`_wrong_argument_call_provider`; see that function's
  docstring for the per-operation "+1 modulo the argument's own valid
  domain" rule, mirroring `apc.primitives.conditioning.
  IntBucketArgumentEncoder`'s own modulo-wraparound convention).
- **Wrong family**: the same function with a provider that forces a
  *different* operation's family (`_wrong_family_call_provider`, a fixed
  derangement over `config.operation_names`, matching A1-R003's own
  `_default_wrong_operation_map` construction) with a fixed, valid default
  argument for that wrong family -- the argument value does not matter for
  this arm's own claim (forcing the wrong computation family), only that the
  call is well-formed.
- **None**: `apc.core.execution.evaluate_exact_match_no_primitive` -- A1-R002's
  own "None" arm entry point, unmodified.

## Filling in H2c's undeclared numeric thresholds

`docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md` H2c and
`docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` A1-R005 both state "materially
lower"/"materially exceeds controls" for the three control arms without a
number, unlike H2b's explicit "<= 0.30" / ">= 0.50". Per `AGENTS.md`'s
decision log ("if ... acceptance criteria [are] underspecified ... add a
concise ADR"), this gate reuses H2b's own predeclared numbers -- `<= 0.30`
per control arm, causal gap `>= 0.50` -- rather than inventing new ones,
following ADR-0026's precedent of treating 0.30/0.50 as this experiment
plan's standing "materially" convention. Only `CORRECT_THRESHOLD` (0.90) is
new here, taken directly from A1-R005's own explicit acceptance text. See
`docs/DECISIONS.md` for the ADR recording this choice.

## Family-count acceptance

"Family count stays constant across argument values" holds structurally
here, not merely by measurement: `_build_primitive_bank` registers exactly
one `ConditionedPrimitive` per `config.operation_names` entry, keyed at
`operation_id(name)` (ADR-0024's convention), and no code path in this
module ever calls `PrimitiveBank.add_primitive`/`new_primitive` again after
that -- `ParameterizedPrimitiveGateReport.bank_size` records `len(bank)` at
the end of the run so this is measured and reported, not merely asserted by
construction.

## Scope discipline

Same restriction as A1-R003: `apc.plastic`, `apc.consolidation`, and
`apc.meta` are never imported here; no `Router` is imported or reachable
(oracle routing only). Composition, promotion to a persistent/`STABLE` bank,
and the eight-operation unified benchmark are A1-R006 onward, out of scope
here -- this gate constructs a fresh `PrimitiveBank` of `CANDIDATE`-status
`ConditionedPrimitive`s per run and discards it after measuring the four arms.
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

from apc.core.data import IGNORE_INDEX, collate_batch
from apc.core.execution import (
    evaluate_exact_match_no_primitive,
    evaluate_exact_match_with_oracle_calls,
    forward_logits_with_oracle_calls_trainable,
)
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SharedCoreTokens
from apc.environments.generator import Example, TaskGenerator, oracle_call_for_example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import operation_id
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.conditioning import (
    DEFAULT_ARG_DIM,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    build_conditioned_primitive,
)
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

__all__ = [
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "CORRECT_THRESHOLD",
    "WRONG_ARGUMENT_CEILING",
    "WRONG_FAMILY_CEILING",
    "NONE_CEILING",
    "MIN_CAUSAL_GAP",
    "ParameterizedPrimitiveGateConfig",
    "parameterized_primitive_gate_config_from_dict",
    "FrozenStableCore",
    "ParameterizedPrimitiveGateReport",
    "run_parameterized_primitive_gate",
    "ParameterizedPrimitiveGateMultiSeedReport",
    "run_parameterized_primitive_gate_multi_seed",
]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md: ">=5 seeds" for a gate claim.
MIN_GATE_SEEDS = 5

# docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md A1-R005's own explicit number.
CORRECT_THRESHOLD = 0.90
# H2c gives no explicit control-arm ceiling/gap; reused from H2b's own
# predeclared numbers (module docstring, "Filling in H2c's undeclared
# numeric thresholds").
WRONG_ARGUMENT_CEILING = 0.30
WRONG_FAMILY_CEILING = 0.30
NONE_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50

# A fixed, valid default argument per operation, used only by the "Wrong
# family" arm (module docstring): that arm's claim is about forcing the
# wrong computation family, not about which argument value accompanies it,
# so any well-formed argument for the forced family suffices.
_DEFAULT_WRONG_FAMILY_ARGUMENTS: dict[str, dict[str, Any]] = {
    "SHIFT": {"amount": 0},
    "SELECT": {"indices": [0]},
    "COUNT": {"target": 0},
    "BIND": {"query_key": 0},
}


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
class ParameterizedPrimitiveGateConfig:
    """Explicit, serializable configuration for one seed's gate run.

    `operation_names` defaults to `PARAMETERIZED_OPERATION_NAMES` (A1-R005's
    own scope); every entry must require exactly one argument (the same
    constraint `apc.primitives.conditioning.build_conditioned_primitive`
    itself enforces) so a `ConditionedPrimitive` can be built for it and a
    "Wrong argument" control is always constructible.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    primitive_rank: int = 8
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    primitive_train: PrimitiveTrainConfig = field(default_factory=PrimitiveTrainConfig)
    num_unseen_eval_examples: int = 2048
    min_examples_per_operation: int = 32

    def __post_init__(self) -> None:
        if len(self.operation_names) < 2:
            raise ValueError(
                "operation_names must have >= 2 entries: a 'Wrong family' control "
                "requires at least one other operation to force instead of the correct one"
            )
        non_parameterized = [
            name
            for name in self.operation_names
            if len(get_operation(name).required_argument_names) != 1
        ]
        if non_parameterized:
            raise ValueError(
                "ParameterizedPrimitiveGateConfig.operation_names must all require exactly "
                f"one argument (Operation.required_argument_names); got operation(s) with a "
                f"different arity: {non_parameterized}. Task A1-R005 is scoped to "
                "PARAMETERIZED_OPERATION_NAMES; parameter-free primitives are Task A1-R003."
            )
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= the upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]}) so every SHIFT/SELECT "
                "argument value this config can actually generate stays inside the argument "
                "encoder's declared bucket/position range"
            )
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


def parameterized_primitive_gate_config_from_dict(
    raw: dict[str, Any],
) -> ParameterizedPrimitiveGateConfig:
    """Parse a `configs/phase_a1_parameterized_primitive_gate.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = ParameterizedPrimitiveGateConfig()
    core_train_raw = raw.get("core_train")
    core_train = (
        dataclasses.replace(defaults.core_train, **core_train_raw)
        if core_train_raw
        else defaults.core_train
    )
    primitive_train_raw = raw.get("primitive_train")
    primitive_train = (
        dataclasses.replace(defaults.primitive_train, **primitive_train_raw)
        if primitive_train_raw
        else defaults.primitive_train
    )
    return ParameterizedPrimitiveGateConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        primitive_rank=raw.get("primitive_rank", defaults.primitive_rank),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        primitive_train=primitive_train,
        num_unseen_eval_examples=raw.get(
            "num_unseen_eval_examples", defaults.num_unseen_eval_examples
        ),
        min_examples_per_operation=raw.get(
            "min_examples_per_operation", defaults.min_examples_per_operation
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind over `config.operation_names`
    (here, the parameterized pool) and then frozen: every parameter has
    `requires_grad=False`. Carries its own `TaskGenerator` so primitive
    training draws from the identical stream the core itself was pretrained
    on -- see `apc.evaluation.parameter_free_primitive_gate.FrozenStableCore`,
    which this mirrors."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    generator: TaskGenerator
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: ParameterizedPrimitiveGateConfig, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    """Pretrain one `DecoderOnlyTransformer` via `train_shared_core` with no
    task segment ever visible (`include_task_spec=False`), restricted to
    `config.operation_names`, then freeze every parameter -- see module
    docstring's "How the Stable Core gets frozen"."""
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=config.operation_names,
        permute_symbols=False,
        include_task_spec=False,
        model=config.model,
        train=config.core_train,
        num_unseen_eval_examples=config.num_unseen_eval_examples,
        min_examples_per_operation=config.min_examples_per_operation,
    )
    trained = train_shared_core(core_config, metrics_path=metrics_path)
    model = trained.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenStableCore(
        model=model,
        tokens=trained.tokens,
        generator=trained.generator,
        final_core_train_loss=trained.final_train_loss,
        device=trained.device,
    )


def _build_primitive_bank(
    operation_names: Sequence[str],
    d_model: int,
    rank: int,
    *,
    vocab_size: int,
    max_sequence_length: int,
    arg_dim: int,
) -> PrimitiveBank:
    """One `ConditionedPrimitive`, keyed at `operation_id(name)` (ADR-0024's
    convention), per entry in `operation_names`. `CANDIDATE` status and
    fully trainable, matching `apc.evaluation.parameter_free_primitive_gate.
    _build_primitive_bank`'s own convention for the parameter-free gate."""
    bank = PrimitiveBank()
    for name in operation_names:
        primitive = build_conditioned_primitive(
            operation_id(name),
            name,
            PrimitiveConfig(d_model=d_model, rank=rank),
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
            status=PrimitiveStatus.CANDIDATE,
            metadata={"operation": name},
        )
        bank.add_primitive(primitive)
    return bank


def _default_wrong_operation_map(operation_names: Sequence[str]) -> dict[str, str]:
    """A fixed derangement over `operation_names` (a cyclic shift): every
    operation maps to a different one, deterministic and reproducible across
    runs/seeds -- identical construction to `apc.evaluation.
    parameter_free_primitive_gate._default_wrong_operation_map`, used here
    for the "Wrong family" arm (module docstring)."""
    names = list(operation_names)
    n = len(names)
    return {names[i]: names[(i + 1) % n] for i in range(n)}


def _wrong_argument_value(call: PrimitiveCall, example: Example) -> dict[str, Any]:
    """A deterministic, materially different argument for `call.operation`
    given `example`'s own content shape -- the H2c "correct family + wrong
    argument" control.

    "+1 modulo the argument's own valid domain" for every currently
    registered parameterized operation, mirroring `apc.primitives.
    conditioning.IntBucketArgumentEncoder`'s own modulo-wraparound
    convention and A1-R003's fixed-derangement "Wrong family" map:
    deterministic, reproducible, and guaranteed different from the correct
    value whenever that domain has more than one member (always true for
    `sequence_length_range`/`vocab_size` >= 2, which every current gate
    config satisfies).
    """
    length = len(example.input_tokens)
    vocab_size = example.vocab_size
    if call.operation == "SHIFT":
        amount = call.arguments["amount"]
        wrong_amount = (amount + 1) % length if length > 1 else amount
        return {"amount": wrong_amount}
    if call.operation == "SELECT":
        indices = call.arguments["indices"]
        wrong_indices = (
            sorted({(i + 1) % length for i in indices}) if length > 1 else list(indices)
        )
        return {"indices": wrong_indices}
    if call.operation == "COUNT":
        target = call.arguments["target"]
        return {"target": (target + 1) % vocab_size}
    if call.operation == "BIND":
        query_key = call.arguments["query_key"]
        return {"query_key": (query_key + 1) % vocab_size}
    raise KeyError(f"_wrong_argument_value has no rule registered for operation {call.operation!r}")


def _wrong_argument_call_provider() -> Callable[[Example], PrimitiveCall]:
    def provider(example: Example) -> PrimitiveCall:
        correct = oracle_call_for_example(example)
        return PrimitiveCall(
            operation=correct.operation, arguments=_wrong_argument_value(correct, example)
        )

    return provider


def _wrong_family_call_provider(
    wrong_operation_map: dict[str, str],
) -> Callable[[Example], PrimitiveCall]:
    def provider(example: Example) -> PrimitiveCall:
        correct = oracle_call_for_example(example)
        wrong_operation = wrong_operation_map[correct.operation]
        return PrimitiveCall(
            operation=wrong_operation, arguments=_DEFAULT_WRONG_FAMILY_ARGUMENTS[wrong_operation]
        )

    return provider


def _train_primitives(
    core: FrozenStableCore,
    bank: PrimitiveBank,
    config: ParameterizedPrimitiveGateConfig,
    metrics_path: str | Path | None = None,
) -> float:
    """Train `bank`'s primitives (only) via oracle-forced routing through
    `core`'s frozen weights -- identical loop shape to `apc.evaluation.
    parameter_free_primitive_gate._train_primitives`; the only functional
    difference is that `bank` holds `ConditionedPrimitive`s, so `calls`
    carrying real arguments (`oracle_call_for_example`) actually train the
    argument-conditioning path via `apc.core.execution`'s A1-R005 dispatch."""
    model, tokens, generator, device = core.model, core.tokens, core.generator, core.device
    bank.to(device)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.primitive_train.lr,
        weight_decay=config.primitive_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after bank/optimizer
    # construction, per ADR-0016's pattern.
    set_seed(config.seed)

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    try:
        for step in range(config.primitive_train.steps):
            examples = generator.generate_online(
                config.primitive_train.batch_size, step=step, split="train"
            )
            calls = [oracle_call_for_example(example) for example in examples]
            batch = collate_batch(examples, tokens, device=device, include_task_spec=False)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_logits_with_oracle_calls_trainable(
                model, batch.input_ids, bank, calls
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if config.primitive_train.grad_clip > 0:
                nn.utils.clip_grad_norm_(bank.parameters(), config.primitive_train.grad_clip)
            optimizer.step()
            final_loss = float(loss.item())

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.primitive_train.eval_every == 0
                or step_number == config.primitive_train.steps
            ):
                progress_examples = generator.generate_online(
                    config.primitive_train.progress_eval_examples, step=step, split="val"
                )
                progress_exact_match, _, _ = evaluate_exact_match_with_oracle_calls(
                    model, progress_examples, tokens, bank, device=device
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "step": step_number,
                            "loss": final_loss,
                            "progress_correct_exact_match": progress_exact_match,
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    return final_loss


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
class ParameterizedPrimitiveGateReport:
    """Everything observed while pretraining, freezing, training primitives
    on top of, and evaluating the four-arm causal ablation matrix for one
    seed."""

    config: ParameterizedPrimitiveGateConfig
    core_steps_trained: int
    core_examples_seen: int
    final_core_train_loss: float
    primitive_steps_trained: int
    primitive_examples_seen: int
    final_primitive_train_loss: float
    correct_exact_match: float
    wrong_argument_exact_match: float
    wrong_family_exact_match: float
    none_exact_match: float
    causal_gap: float
    per_operation_correct_exact_match: dict[str, float]
    per_operation_wrong_argument_exact_match: dict[str, float]
    per_operation_wrong_family_exact_match: dict[str, float]
    per_operation_none_exact_match: dict[str, float]
    per_operation_eval_counts: dict[str, int]
    num_unseen_eval_examples: int
    wrong_operation_map: dict[str, str]
    bank_size: int
    core_param_count: int
    core_trainable_param_count: int
    primitive_param_count: int
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_parameterized_primitive_gate(
    config: ParameterizedPrimitiveGateConfig,
    *,
    core_metrics_path: str | Path | None = None,
    primitive_metrics_path: str | Path | None = None,
    wrong_operation_map: dict[str, str] | None = None,
) -> ParameterizedPrimitiveGateReport:
    """Run one seed of the A1-R005 gate end to end: pretrain and freeze a
    Stable Core, train one `ConditionedPrimitive` per `config.operation_names`,
    then evaluate Correct/Wrong argument/Wrong family/None on a large,
    independently-drawn unseen batch."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, metrics_path=core_metrics_path)
    bank = _build_primitive_bank(
        config.operation_names,
        core.model.config.d_model,
        config.primitive_rank,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    )
    final_primitive_loss = _train_primitives(
        core, bank, config, metrics_path=primitive_metrics_path
    )

    resolved_wrong_map = wrong_operation_map or _default_wrong_operation_map(
        config.operation_names
    )

    unseen_examples = core.generator.generate_online(
        config.num_unseen_eval_examples, step=0, split="test"
    )

    correct_exact_match, correct_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model, unseen_examples, core.tokens, bank, device=core.device
    )
    wrong_argument_exact_match, wrong_argument_predictions, _ = (
        evaluate_exact_match_with_oracle_calls(
            core.model,
            unseen_examples,
            core.tokens,
            bank,
            _wrong_argument_call_provider(),
            device=core.device,
        )
    )
    wrong_family_exact_match, wrong_family_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model,
        unseen_examples,
        core.tokens,
        bank,
        _wrong_family_call_provider(resolved_wrong_map),
        device=core.device,
    )
    none_exact_match, none_predictions = evaluate_exact_match_no_primitive(
        core.model, unseen_examples, core.tokens, device=core.device
    )

    min_per_op = config.min_examples_per_operation
    per_operation_correct, per_operation_counts = _per_operation_exact_match(
        unseen_examples, correct_predictions, config.operation_names, min_per_op
    )
    per_operation_wrong_argument, _ = _per_operation_exact_match(
        unseen_examples, wrong_argument_predictions, config.operation_names, min_per_op
    )
    per_operation_wrong_family, _ = _per_operation_exact_match(
        unseen_examples, wrong_family_predictions, config.operation_names, min_per_op
    )
    per_operation_none, _ = _per_operation_exact_match(
        unseen_examples, none_predictions, config.operation_names, min_per_op
    )

    causal_gap = correct_exact_match - max(
        wrong_argument_exact_match, wrong_family_exact_match, none_exact_match
    )

    return ParameterizedPrimitiveGateReport(
        config=config,
        core_steps_trained=config.core_train.steps,
        core_examples_seen=config.core_train.steps * config.core_train.batch_size,
        final_core_train_loss=core.final_core_train_loss,
        primitive_steps_trained=config.primitive_train.steps,
        primitive_examples_seen=config.primitive_train.steps * config.primitive_train.batch_size,
        final_primitive_train_loss=final_primitive_loss,
        correct_exact_match=correct_exact_match,
        wrong_argument_exact_match=wrong_argument_exact_match,
        wrong_family_exact_match=wrong_family_exact_match,
        none_exact_match=none_exact_match,
        causal_gap=causal_gap,
        per_operation_correct_exact_match=per_operation_correct,
        per_operation_wrong_argument_exact_match=per_operation_wrong_argument,
        per_operation_wrong_family_exact_match=per_operation_wrong_family,
        per_operation_none_exact_match=per_operation_none,
        per_operation_eval_counts=per_operation_counts,
        num_unseen_eval_examples=len(unseen_examples),
        wrong_operation_map=resolved_wrong_map,
        bank_size=len(bank),
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        primitive_param_count=bank.total_parameter_count(),
        wall_clock_seconds=time.perf_counter() - start,
        device=str(core.device),
    )


def _summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean_value = statistics.fmean(values)
    stdev_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, stdev_value, min(values), max(values)


@dataclass(frozen=True)
class ParameterizedPrimitiveGateMultiSeedReport:
    """A1-R005's verdict aggregated across seeds. `passed` is computed
    against the cross-seed *mean* of each arm, matching every other Phase
    A.1 gate's own convention."""

    seeds: tuple[int, ...]
    per_seed: tuple[ParameterizedPrimitiveGateReport, ...]
    operation_names: tuple[str, ...]
    mean_correct_exact_match: float
    stdev_correct_exact_match: float
    min_correct_exact_match: float
    max_correct_exact_match: float
    mean_wrong_argument_exact_match: float
    mean_wrong_family_exact_match: float
    mean_none_exact_match: float
    mean_causal_gap: float
    per_operation_mean_correct_exact_match: dict[str, float]
    correct_threshold: float
    wrong_argument_ceiling: float
    wrong_family_ceiling: float
    none_ceiling: float
    min_causal_gap: float
    correct_passed: bool
    wrong_argument_passed: bool
    wrong_family_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    family_count_passed: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "operation_names": list(self.operation_names),
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "stdev_correct_exact_match": self.stdev_correct_exact_match,
            "min_correct_exact_match": self.min_correct_exact_match,
            "max_correct_exact_match": self.max_correct_exact_match,
            "mean_wrong_argument_exact_match": self.mean_wrong_argument_exact_match,
            "mean_wrong_family_exact_match": self.mean_wrong_family_exact_match,
            "mean_none_exact_match": self.mean_none_exact_match,
            "mean_causal_gap": self.mean_causal_gap,
            "per_operation_mean_correct_exact_match": self.per_operation_mean_correct_exact_match,
            "correct_threshold": self.correct_threshold,
            "wrong_argument_ceiling": self.wrong_argument_ceiling,
            "wrong_family_ceiling": self.wrong_family_ceiling,
            "none_ceiling": self.none_ceiling,
            "min_causal_gap": self.min_causal_gap,
            "correct_passed": self.correct_passed,
            "wrong_argument_passed": self.wrong_argument_passed,
            "wrong_family_passed": self.wrong_family_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "family_count_passed": self.family_count_passed,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_parameterized_primitive_gate_multi_seed(
    base_config: ParameterizedPrimitiveGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    correct_threshold: float = CORRECT_THRESHOLD,
    wrong_argument_ceiling: float = WRONG_ARGUMENT_CEILING,
    wrong_family_ceiling: float = WRONG_FAMILY_CEILING,
    none_ceiling: float = NONE_CEILING,
    min_causal_gap: float = MIN_CAUSAL_GAP,
) -> ParameterizedPrimitiveGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 gate.
    """
    per_seed: list[ParameterizedPrimitiveGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        core_metrics_path = (
            run_dir_path / f"seed_{seed}" / "core_metrics.jsonl" if run_dir_path else None
        )
        primitive_metrics_path = (
            run_dir_path / f"seed_{seed}" / "primitive_metrics.jsonl" if run_dir_path else None
        )
        report = run_parameterized_primitive_gate(
            config,
            core_metrics_path=core_metrics_path,
            primitive_metrics_path=primitive_metrics_path,
        )
        if run_dir_path is not None:
            seed_dir = run_dir_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    correct_values = [report.correct_exact_match for report in per_seed]
    mean_correct, stdev_correct, min_correct, max_correct = _summarize(correct_values)
    mean_wrong_argument = statistics.fmean(
        report.wrong_argument_exact_match for report in per_seed
    )
    mean_wrong_family = statistics.fmean(report.wrong_family_exact_match for report in per_seed)
    mean_none = statistics.fmean(report.none_exact_match for report in per_seed)
    mean_causal_gap = mean_correct - max(mean_wrong_argument, mean_wrong_family, mean_none)

    per_operation_mean_correct: dict[str, float] = {}
    for name in base_config.operation_names:
        op_values = [report.per_operation_correct_exact_match[name] for report in per_seed]
        per_operation_mean_correct[name] = statistics.fmean(op_values)

    correct_passed = mean_correct >= correct_threshold
    wrong_argument_passed = mean_wrong_argument <= wrong_argument_ceiling
    wrong_family_passed = mean_wrong_family <= wrong_family_ceiling
    none_passed = mean_none <= none_ceiling
    causal_gap_passed = mean_causal_gap >= min_causal_gap
    family_count_passed = all(
        report.bank_size == len(base_config.operation_names) for report in per_seed
    )
    passed = (
        correct_passed
        and wrong_argument_passed
        and wrong_family_passed
        and none_passed
        and causal_gap_passed
        and family_count_passed
    )

    return ParameterizedPrimitiveGateMultiSeedReport(
        seeds=tuple(report.config.seed for report in per_seed),
        per_seed=tuple(per_seed),
        operation_names=tuple(base_config.operation_names),
        mean_correct_exact_match=mean_correct,
        stdev_correct_exact_match=stdev_correct,
        min_correct_exact_match=min_correct,
        max_correct_exact_match=max_correct,
        mean_wrong_argument_exact_match=mean_wrong_argument,
        mean_wrong_family_exact_match=mean_wrong_family,
        mean_none_exact_match=mean_none,
        mean_causal_gap=mean_causal_gap,
        per_operation_mean_correct_exact_match=per_operation_mean_correct,
        correct_threshold=correct_threshold,
        wrong_argument_ceiling=wrong_argument_ceiling,
        wrong_family_ceiling=wrong_family_ceiling,
        none_ceiling=none_ceiling,
        min_causal_gap=min_causal_gap,
        correct_passed=correct_passed,
        wrong_argument_passed=wrong_argument_passed,
        wrong_family_passed=wrong_family_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        family_count_passed=family_count_passed,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
