"""Sequence counterfactual gate: `SHIFT` and `SELECT` (Phase A.1
Post-Correction Task A1-R005D-008, `docs/CODEX_TASKS_A1_R005_RETRY.md`).

`docs/CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-008 ("SHIFT and SELECT sequence
gates") names four metrics per operation -- exact match, token accuracy,
length-stratified metrics, effectful Wrong argument -- and one shared
acceptance bar (token accuracy >= 0.95, exact match >= 0.85, effectful
Wrong-argument performance "materially lower"). `SHIFT`/`SELECT` differ from
`COUNT`/`BIND` (ADR-0033/ADR-0036) in the shape of their output: both produce
a *multi-token* sequence (`ShiftOp`/`SelectOp.output_length` return the full
or roughly-halved input length, never `1`), so "exact match" is a much
stricter bar than `COUNT`/`BIND`'s single-token case (ADR-0029's own
"multi-token exact-match compounding" concern) -- "token accuracy" (per-
position match rate) is the finer-grained metric this task's own acceptance
list adds alongside it for exactly that reason.

This one task names both operations together (unlike A1-R005D-004/A1-R005D-007,
each its own task for one operation), so this module -- unlike `apc.
evaluation.count_counterfactual_gate`/`bind_counterfactual_gate` -- is
parameterized by `operation` (`SHIFT` or `SELECT`, `SEQUENCE_OPERATIONS`)
rather than hardcoding one. A caller runs the gate once per operation
(`SequenceCounterfactualGateConfig(operation=...)`); there is still no
"Wrong family" arm, since each run registers exactly one primitive family,
matching every earlier Phase A.1 primitive gate's own convention.

## Counterfactual groups, generalized across argument shape

`generate_sequence_counterfactual_groups` follows the same discipline
A1-R005D-004/A1-R005D-007 established (`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s
"Counterfactual training rule"): for one randomly drawn content sequence,
search for `group_size` argument values whose outputs are pairwise distinct,
so every "Wrong argument" pairing this module evaluates is effectful *by
construction* (`argument_effect_rate` is measured, not assumed, same as
every earlier counterfactual gate). The task text does not restate this
requirement for A1-R005D-008 specifically -- the standing addendum rule only
requires "at least one" retry experiment to use it, already satisfied by
A1-R005D-004/A1-R005D-007 -- but this module reuses it anyway for
methodological consistency with the rest of this retry sequence and because
it is what makes the "effectful Wrong argument" arm this task's own
acceptance list names actually guaranteed-effectful rather than incidentally
so.

`SHIFT`'s candidate argument values (`amount`) are simply `range(length)`
(`ShiftOp.sample_params`'s own domain) -- small and finite, searched by
shuffling and taking the first `group_size` with pairwise-distinct outputs,
exactly like `count_counterfactual_gate`'s target search. `SELECT`'s
candidate argument values (`indices`) are combinatorial: every `k`-subset of
`range(length)` where `k = SelectOp.output_length(length)`
(`itertools.combinations`) -- at these lengths (`sequence_length_range`
bounded well under 20), `math.comb(length, k)` stays in the low hundreds at
most, so full enumeration-then-shuffle is simple and exact, unlike
`BIND`'s candidate pool (the content's own present keys) which is
enumerated directly without any combinatorial step.

## "Use the selected low-rank conditioned architecture first"

Neither A1-R005D-005 (ADR-0034, "no stage passed") nor A1-R005D-006
(ADR-0035, "no variant passed") actually *selected* an alternative
architecture or capacity setting for `COUNT` -- both concluded with no
configuration clearing the acceptance bar, so nothing was promoted over the
baseline either investigation started from. "The selected low-rank
conditioned architecture" therefore resolves to that same baseline: V0/
additive `apc.primitives.conditioning.ConditionedPrimitive`, `primitive_rank
=8`, `arg_dim=16` -- the one architecture every gate in this retry sequence
(A1-R005D-004 through A1-R005D-007) has actually run, since no investigated
alternative ever beat it. `SELECT`'s own "order-preserving argument encoder
mandatory" requirement is already the production default
(`apc.primitives.conditioning.default_argument_encoder("SELECT", ...)`
returns `OrderPreservingIndexSetArgumentEncoder`, Task A1-R005D-003/
ADR-0032) -- `build_conditioned_primitive` for `SELECT` uses it
automatically, with nothing new required here.

## The cross-attention fallback is not built in this task

A1-R005D-008's own text: "If it fails despite strong COUNT/BIND results,
implement a minimal tiny cross-attention primitive as an explicit
alternative class." This module implements only the gate itself, not that
fallback primitive -- see `docs/DECISIONS.md`'s ADR for this task for why:
in short, the fallback's own stated precondition ("strong COUNT/BIND
results") does not hold (`COUNT` failed its STOP GATE at every architecture
tried, ADR-0033/0034/0035; `BIND` failed even more severely, with a
statistically zero causal gap, ADR-0036), so a `SHIFT`/`SELECT` failure here
would not license the inference this task's own conditional draws --
building a new primitive class in response would risk exactly the "hide a
negative result by escalating architecture" pattern `AGENTS.md`'s STOP GATE
discipline rules out.

## Filling in acceptance numbers the task text does not restate

A1-R005D-008's "Acceptance" list gives two hard numbers (token accuracy >=
0.95, exact match >= 0.85) and one qualitative criterion ("effectful
Wrong-argument performance materially lower"), without a number for the
latter or a `None` ceiling. Per `AGENTS.md`'s decision-log guidance for
underspecified acceptance criteria (already invoked by A1-R005D-004/
A1-R005D-007's own `None`-ceiling fill-ins), this gate reuses this whole
retry sequence's own two standing numeric conventions rather than inventing
new ones: `min_exact_match_causal_gap = 0.50` (the same "Correct -
max(Wrong, None) >= 0.50" formula every earlier gate in this sequence
computes, applied here to exact match specifically as the "materially
lower" criterion's numeric form) and `none_ceiling = 0.30` (`NONE_CEILING`,
A1-R005D-004's own fill-in, ADR-0033). Token-accuracy's own causal gap is
computed and reported (`token_accuracy_causal_gap`) but not separately
gated, since the task's own acceptance list already gates token accuracy
directly (>= 0.95) and gates the Wrong-argument comparison once, via exact
match.

## Length-stratified metrics

`_length_stratified_metrics` buckets the Correct arm's exact match and token
accuracy by `len(example.input_tokens)` -- diagnostic only (the task's own
"Work" item lists it as a metric to report, not a numbered acceptance
threshold, matching A1-R005D-001's own precedent of computing diagnostics
that are not separately gated).

## Scope discipline

Same restriction as every earlier Phase A.1 primitive gate: `apc.plastic`,
`apc.consolidation`, and `apc.meta` are never imported here; no `Router` is
imported or reachable (oracle routing only). This gate constructs a fresh
single-primitive `PrimitiveBank` per run and discards it after measuring the
arms.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
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
from apc.core.execution import (
    evaluate_exact_match_no_primitive,
    evaluate_exact_match_with_oracle_calls,
    forward_logits_with_oracle_calls_trainable,
)
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SharedCoreTokens
from apc.environments.generator import Example, OracleMetadata, oracle_call_for_example
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec, operation_id
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
    "SHIFT_OPERATION",
    "SELECT_OPERATION",
    "SEQUENCE_OPERATIONS",
    "MIN_GROUP_SIZE",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "EXACT_MATCH_THRESHOLD",
    "TOKEN_ACCURACY_THRESHOLD",
    "MIN_EXACT_MATCH_CAUSAL_GAP",
    "NONE_CEILING",
    "SequenceCounterfactualGroup",
    "generate_sequence_counterfactual_groups",
    "SequenceCounterfactualGateConfig",
    "sequence_counterfactual_gate_config_from_dict",
    "FrozenStableCore",
    "SequenceCounterfactualGateReport",
    "run_sequence_counterfactual_gate",
    "SequenceCounterfactualGateMultiSeedReport",
    "run_sequence_counterfactual_gate_multi_seed",
]

SHIFT_OPERATION = "SHIFT"
SELECT_OPERATION = "SELECT"
SEQUENCE_OPERATIONS: tuple[str, ...] = (SHIFT_OPERATION, SELECT_OPERATION)
_ARGUMENT_NAME: dict[str, str] = {SHIFT_OPERATION: "amount", SELECT_OPERATION: "indices"}

# A counterfactual group needs at least one "wrong argument" partner besides
# the correct one.
MIN_GROUP_SIZE = 2
# Matches every earlier counterfactual gate's own DEFAULT_GROUP_SIZE
# convention (docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md section 4).
DEFAULT_GROUP_SIZE = 3
# Same order-of-magnitude safety bound as count_counterfactual_gate/
# bind_counterfactual_gate's own _MAX_CONTENT_RESAMPLE_ATTEMPTS.
_MAX_CONTENT_RESAMPLE_ATTEMPTS = 500

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5

# docs/CODEX_TASKS_A1_R005_RETRY.md A1-R005D-008's own explicit numbers.
EXACT_MATCH_THRESHOLD = 0.85
TOKEN_ACCURACY_THRESHOLD = 0.95
# Not restated by A1-R005D-008's own acceptance list; filled in from this
# retry sequence's own standing conventions (module docstring, "Filling in
# acceptance numbers the task text does not restate").
MIN_EXACT_MATCH_CAUSAL_GAP = 0.50
NONE_CEILING = 0.30


def _default_model_config() -> dict[str, Any]:
    return {
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 768,
        "max_seq_len": 48,
        "dropout": 0.0,
    }


def _derive_local_seed(seed: int, step: int, label: str) -> int:
    """Deterministic sub-seed for one `(seed, step, label)` triple,
    independent of `PYTHONHASHSEED` -- same construction as `apc.
    evaluation.count_counterfactual_gate._derive_local_seed`/`apc.evaluation.
    bind_counterfactual_gate._derive_local_seed`, kept local here since each
    is private to its own module."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sequence_output(
    operation: str, content: Sequence[int], argument_value: Any
) -> tuple[int, ...]:
    """`ShiftOp.apply`/`SelectOp.apply`'s transform, reimplemented directly
    against a raw content sequence and a raw argument value (not a full
    `params` dict) so candidate argument values can be pre-filtered for
    pairwise-distinct outputs before committing to building a real
    `Example` -- same role as `count_counterfactual_gate._count_output`/
    `bind_counterfactual_gate._bind_output`."""
    if operation == SHIFT_OPERATION:
        amount = argument_value % len(content)
        return tuple(content[amount:]) + tuple(content[:amount])
    if operation == SELECT_OPERATION:
        return tuple(content[i] for i in argument_value)
    raise ValueError(
        f"unsupported operation: {operation!r} (expected one of {SEQUENCE_OPERATIONS})"
    )


def _argument_candidates(operation: str, rng: random.Random, content: Sequence[int]) -> list[Any]:
    """Every legal argument value for `operation` given `content`'s length,
    shuffled -- `SHIFT`'s domain is `range(length)` (`ShiftOp.sample_params`);
    `SELECT`'s is every `k`-subset of `range(length)`
    (`itertools.combinations`, `k = SelectOp.output_length(length)`), matching
    `SelectOp.sample_params`'s own domain (module docstring, "Counterfactual
    groups, generalized across argument shape")."""
    length = len(content)
    if operation == SHIFT_OPERATION:
        candidates: list[Any] = list(range(length))
    elif operation == SELECT_OPERATION:
        k = get_operation(SELECT_OPERATION).output_length(length)
        candidates = [list(combo) for combo in itertools.combinations(range(length), k)]
    else:
        raise ValueError(
            f"unsupported operation: {operation!r} (expected one of {SEQUENCE_OPERATIONS})"
        )
    rng.shuffle(candidates)
    return candidates


def _build_sequence_example(
    input_tokens: tuple[int, ...], operation: str, argument_value: Any, vocab_size: int, split: str
) -> Example:
    """One real `Example` for `operation(argument_value)` applied to
    `input_tokens`, built the same way `count_counterfactual_gate._build_
    count_example`/`bind_counterfactual_gate._build_bind_example` build a
    depth-1 example -- `apc.environments.interpreter.run_program` is the
    same ground-truth executor, just driven by an explicit argument value
    instead of `Operation.sample_params`'s random draw."""
    argument_name = _ARGUMENT_NAME[operation]
    program = Program(
        steps=(ProgramStep(operation=operation, params={argument_name: argument_value}),)
    )
    result = run_program(program, input_tokens, vocab_size)
    return Example(
        input_tokens=input_tokens,
        target_tokens=result.output_tokens,
        program=program,
        operation_graph=result.graph,
        category="known",
        split=split,
        vocab_size=vocab_size,
        task_spec=TaskSpec.from_program(program),
        oracle_metadata=OracleMetadata(label="K", primitive_operations=(operation,)),
        symbol_permutation=None,
    )


@dataclass(frozen=True)
class SequenceCounterfactualGroup:
    """One content sequence paired with >=`MIN_GROUP_SIZE` argument values
    for `operation` whose outputs are pairwise distinct. `examples[i]` is
    the real `Example` for `argument_values[i]`; every `examples[i].
    target_tokens` differs from every other member's, checked eagerly here
    rather than merely assumed from the construction that produced them."""

    operation: str
    input_tokens: tuple[int, ...]
    examples: tuple[Example, ...]
    argument_values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.operation not in SEQUENCE_OPERATIONS:
            raise ValueError(
                f"operation must be one of {SEQUENCE_OPERATIONS}, got {self.operation!r}"
            )
        if len(self.examples) != len(self.argument_values):
            raise ValueError(
                "SequenceCounterfactualGroup requires len(examples) == len(argument_values), "
                f"got {len(self.examples)} and {len(self.argument_values)}"
            )
        if len(self.examples) < MIN_GROUP_SIZE:
            raise ValueError(
                f"SequenceCounterfactualGroup requires >= {MIN_GROUP_SIZE} members, got "
                f"{len(self.examples)}"
            )
        outputs = [example.target_tokens for example in self.examples]
        if len(set(outputs)) != len(outputs):
            raise ValueError(
                "SequenceCounterfactualGroup requires pairwise distinct outputs across its "
                f"members; got outputs {outputs} for argument_values {self.argument_values}"
            )

    def wrong_argument_index(self, member_index: int) -> int:
        """The index of a group member other than `member_index`, guaranteed
        to have a different output (`__post_init__`'s own invariant)."""
        return (member_index + 1) % len(self.examples)


def _generate_sequence_counterfactual_group(
    rng: random.Random,
    operation: str,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    group_size: int,
    split: str,
) -> SequenceCounterfactualGroup:
    min_len, max_len = sequence_length_range
    for _ in range(_MAX_CONTENT_RESAMPLE_ATTEMPTS):
        length = rng.randint(min_len, max_len)
        input_tokens = tuple(rng.randrange(vocab_size) for _ in range(length))

        candidates = _argument_candidates(operation, rng, input_tokens)
        chosen_values: list[Any] = []
        seen_outputs: set[tuple[int, ...]] = set()
        for value in candidates:
            output = _sequence_output(operation, input_tokens, value)
            if output in seen_outputs:
                continue
            seen_outputs.add(output)
            chosen_values.append(value)
            if len(chosen_values) == group_size:
                break

        if len(chosen_values) >= MIN_GROUP_SIZE:
            examples = tuple(
                _build_sequence_example(input_tokens, operation, value, vocab_size, split)
                for value in chosen_values
            )
            return SequenceCounterfactualGroup(
                operation=operation,
                input_tokens=input_tokens,
                examples=examples,
                argument_values=tuple(chosen_values),
            )

    raise RuntimeError(
        f"failed to construct a {operation} counterfactual group with >= {MIN_GROUP_SIZE} "
        f"pairwise distinct outputs after {_MAX_CONTENT_RESAMPLE_ATTEMPTS} resamples "
        f"(vocab_size={vocab_size}, sequence_length_range={sequence_length_range}) -- widen "
        "sequence_length_range or vocab_size if this recurs"
    )


def generate_sequence_counterfactual_groups(
    seed: int,
    n_groups: int,
    *,
    operation: str,
    step: int,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[SequenceCounterfactualGroup, ...]:
    """Deterministic-by-`(seed, step, split, operation)` counterfactual
    group batch, matching every earlier counterfactual gate's own
    determinism convention (one shared `random.Random` stream for the whole
    call). `operation` is included in the derived seed (unlike `COUNT`/
    `BIND`'s own single-operation modules) so `SHIFT` and `SELECT` groups
    for the same `(seed, step, split)` are independent draws, not
    incidentally correlated.
    """
    if operation not in SEQUENCE_OPERATIONS:
        raise ValueError(f"operation must be one of {SEQUENCE_OPERATIONS}, got {operation!r}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if group_size < MIN_GROUP_SIZE:
        raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {group_size}")
    rng = random.Random(_derive_local_seed(seed, step, f"{split}:{operation}"))
    return tuple(
        _generate_sequence_counterfactual_group(
            rng, operation, vocab_size, sequence_length_range, group_size, split
        )
        for _ in range(n_groups)
    )


def _flatten_groups(
    groups: Sequence[SequenceCounterfactualGroup],
) -> tuple[list[Example], dict[int, PrimitiveCall], dict[int, tuple[int, ...]]]:
    """Flatten `groups` into one example list plus two `id(example)`-keyed
    side tables: `wrong_call_by_id` (the "Wrong argument" `PrimitiveCall` to
    force for that example -- a different group member's argument value,
    per `SequenceCounterfactualGroup.wrong_argument_index`) and
    `wrong_target_tokens_by_id` (that partner's own ground-truth output,
    used only to report `argument_effect_rate`)."""
    examples: list[Example] = []
    wrong_call_by_id: dict[int, PrimitiveCall] = {}
    wrong_target_tokens_by_id: dict[int, tuple[int, ...]] = {}
    for group in groups:
        argument_name = _ARGUMENT_NAME[group.operation]
        for index, example in enumerate(group.examples):
            partner_index = group.wrong_argument_index(index)
            wrong_call_by_id[id(example)] = PrimitiveCall(
                operation=group.operation,
                arguments={argument_name: group.argument_values[partner_index]},
            )
            wrong_target_tokens_by_id[id(example)] = group.examples[partner_index].target_tokens
            examples.append(example)
    return examples, wrong_call_by_id, wrong_target_tokens_by_id


@dataclass(frozen=True)
class SequenceCounterfactualGateConfig:
    """Explicit, serializable configuration for one seed's A1-R005D-008 gate
    run (one operation, `SHIFT` or `SELECT`). `primitive_rank`/`arg_dim`/
    `core_train`/`primitive_train` default to the same baseline numbers this
    whole retry sequence has used since A1-R005D-004, per this task's own
    "Use the selected low-rank conditioned architecture first" (module
    docstring)."""

    operation: str = SHIFT_OPERATION
    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(default_factory=SharedCoreGateTrainConfig)
    primitive_rank: int = 8
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    primitive_train: PrimitiveTrainConfig = field(default_factory=PrimitiveTrainConfig)
    num_unseen_eval_groups: int = 1400
    min_unseen_eval_examples: int = 1024

    def __post_init__(self) -> None:
        if self.operation not in SEQUENCE_OPERATIONS:
            raise ValueError(
                f"operation must be one of {SEQUENCE_OPERATIONS}, got {self.operation!r}"
            )
        if self.group_size < MIN_GROUP_SIZE:
            raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {self.group_size}")
        if self.max_sequence_length < self.sequence_length_range[1]:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be >= the upper end of "
                f"sequence_length_range ({self.sequence_length_range[1]}) so this config stays "
                "internally consistent (matching every earlier counterfactual gate's convention)"
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


def sequence_counterfactual_gate_config_from_dict(
    raw: dict[str, Any],
) -> SequenceCounterfactualGateConfig:
    """Parse a `configs/phase_a1_{shift,select}_counterfactual_gate.yaml`-shaped
    dict, matching every other Phase A.1 gate's convention of filling in
    defaults for whatever the file omits."""
    defaults = SequenceCounterfactualGateConfig()
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
    return SequenceCounterfactualGateConfig(
        operation=raw.get("operation", defaults.operation),
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        primitive_rank=raw.get("primitive_rank", defaults.primitive_rank),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        primitive_train=primitive_train,
        num_unseen_eval_groups=raw.get("num_unseen_eval_groups", defaults.num_unseen_eval_groups),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind on ordinary i.i.d. examples for
    one operation, then frozen: every parameter has `requires_grad=False`.
    Not shared with any other gate module's own `FrozenStableCore` -- each
    operation-specific gate module defines its own (`count_counterfactual_
    gate`/`bind_counterfactual_gate`'s own precedent)."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: SequenceCounterfactualGateConfig, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=(config.operation,),
        permute_symbols=False,
        include_task_spec=False,
        model=config.model,
        train=config.core_train,
    )
    trained = train_shared_core(core_config, metrics_path=metrics_path)
    model = trained.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenStableCore(
        model=model,
        tokens=trained.tokens,
        final_core_train_loss=trained.final_train_loss,
        device=trained.device,
    )


def _build_primitive_bank(
    d_model: int,
    rank: int,
    *,
    operation: str,
    vocab_size: int,
    max_sequence_length: int,
    arg_dim: int,
) -> PrimitiveBank:
    """Exactly one argument-conditioned primitive (V0/additive,
    `ConditionedPrimitive`) for `operation`, keyed at `operation_id
    (operation)` (ADR-0024's convention) -- one family, no "Wrong family"
    arm (module docstring)."""
    bank = PrimitiveBank()
    primitive = build_conditioned_primitive(
        operation_id(operation),
        operation,
        PrimitiveConfig(d_model=d_model, rank=rank),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        arg_dim=arg_dim,
        status=PrimitiveStatus.CANDIDATE,
        metadata={"operation": operation},
    )
    bank.add_primitive(primitive)
    return bank


def _train_primitives(
    core: FrozenStableCore,
    bank: PrimitiveBank,
    config: SequenceCounterfactualGateConfig,
    metrics_path: str | Path | None = None,
) -> tuple[float, int]:
    """Train `bank`'s one `ConditionedPrimitive` via oracle-forced routing
    through `core`'s frozen weights, on counterfactual groups
    (`generate_sequence_counterfactual_groups`) -- the same content appears
    multiple times per step, each paired with a different (correct)
    argument value, so content alone can never explain the target within a
    step.

    Returns `(final_loss, total_examples_seen)`.
    """
    model, tokens, device = core.model, core.tokens, core.device
    bank.to(device)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.primitive_train.lr,
        weight_decay=config.primitive_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after bank/optimizer
    # construction, per ADR-0016's pattern.
    set_seed(config.seed)

    groups_per_step = max(1, -(-config.primitive_train.batch_size // config.group_size))

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    total_examples_seen = 0
    try:
        for step in range(config.primitive_train.steps):
            groups = generate_sequence_counterfactual_groups(
                config.seed,
                groups_per_step,
                operation=config.operation,
                step=step,
                split="train",
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
                group_size=config.group_size,
            )
            examples, _, _ = _flatten_groups(groups)
            calls = [oracle_call_for_example(example) for example in examples]
            batch = collate_batch(examples, tokens, device=device, include_task_spec=False)
            total_examples_seen += len(examples)

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
                progress_groups = generate_sequence_counterfactual_groups(
                    config.seed,
                    max(1, -(-config.primitive_train.progress_eval_examples // config.group_size)),
                    operation=config.operation,
                    step=step,
                    split="val",
                    vocab_size=config.vocab_size,
                    sequence_length_range=config.sequence_length_range,
                    group_size=config.group_size,
                )
                progress_examples, _, _ = _flatten_groups(progress_groups)
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

    return final_loss, total_examples_seen


def _token_accuracy(target: tuple[int, ...], prediction: tuple[int, ...]) -> float:
    """Fraction of `target`'s positions `prediction` matches, aligned by
    position. A `prediction` shorter than `target` scores 0 on every
    position beyond its own length (falls out of `zip`'s pairing, contributing
    nothing to `matches`); a `prediction` longer than `target` has its extra
    tail ignored -- `len(target)` is always the denominator."""
    if len(target) == 0:
        raise ValueError("target must be non-empty")
    matches = sum(1 for t, p in zip(target, prediction, strict=False) if t == p)
    return matches / len(target)


def _mean_token_accuracy(
    examples: Sequence[Example], predictions: Sequence[tuple[int, ...]]
) -> float:
    return statistics.fmean(
        _token_accuracy(example.target_tokens, prediction)
        for example, prediction in zip(examples, predictions, strict=True)
    )


def _length_stratified_metrics(
    examples: Sequence[Example], predictions: Sequence[tuple[int, ...]]
) -> dict[str, dict[str, Any]]:
    """Correct-arm exact match/token accuracy, bucketed by
    `len(example.input_tokens)` -- diagnostic only (module docstring,
    "Length-stratified metrics"). Keyed by `str(length)` for JSON-safety."""
    buckets: dict[int, list[tuple[Example, tuple[int, ...]]]] = {}
    for example, prediction in zip(examples, predictions, strict=True):
        buckets.setdefault(len(example.input_tokens), []).append((example, prediction))
    result: dict[str, dict[str, Any]] = {}
    for length, pairs in sorted(buckets.items()):
        exact_match = statistics.fmean(
            float(example.target_tokens == prediction) for example, prediction in pairs
        )
        token_accuracy = statistics.fmean(
            _token_accuracy(example.target_tokens, prediction) for example, prediction in pairs
        )
        result[str(length)] = {
            "count": len(pairs),
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
        }
    return result


@dataclass(frozen=True)
class SequenceCounterfactualGateReport:
    """Everything observed while pretraining, freezing, training the one
    `SHIFT`/`SELECT` primitive on top of, and evaluating one seed's causal
    ablation (Correct / effectful Wrong argument / None), each measured by
    exact match and token accuracy."""

    operation: str
    config: SequenceCounterfactualGateConfig
    core_steps_trained: int
    core_examples_seen: int
    final_core_train_loss: float
    primitive_steps_trained: int
    primitive_examples_seen: int
    final_primitive_train_loss: float
    correct_exact_match: float
    correct_token_accuracy: float
    effectful_wrong_argument_exact_match: float
    effectful_wrong_argument_token_accuracy: float
    none_exact_match: float
    none_token_accuracy: float
    exact_match_causal_gap: float
    token_accuracy_causal_gap: float
    length_stratified_correct: dict[str, dict[str, Any]]
    argument_effect_rate: float
    num_unseen_eval_groups: int
    num_unseen_eval_examples: int
    mean_eval_group_size: float
    bank_size: int
    core_param_count: int
    core_trainable_param_count: int
    primitive_param_count: int
    wall_clock_seconds: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        return json.loads(json.dumps(raw, default=str))


def run_sequence_counterfactual_gate(
    config: SequenceCounterfactualGateConfig,
    *,
    core_metrics_path: str | Path | None = None,
    primitive_metrics_path: str | Path | None = None,
) -> SequenceCounterfactualGateReport:
    """Run one seed of the A1-R005D-008 gate end to end for `config.operation`:
    pretrain and freeze a task-blind, single-operation Stable Core, train one
    argument-conditioned `ConditionedPrimitive` (V0/additive) on
    counterfactual groups, then evaluate Correct/effectful Wrong
    argument/None (exact match + token accuracy) on a large, independently-
    drawn unseen batch of counterfactual groups."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, metrics_path=core_metrics_path)
    bank = _build_primitive_bank(
        core.model.config.d_model,
        config.primitive_rank,
        operation=config.operation,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    )
    final_primitive_loss, primitive_examples_seen = _train_primitives(
        core, bank, config, metrics_path=primitive_metrics_path
    )

    unseen_groups = generate_sequence_counterfactual_groups(
        config.seed,
        config.num_unseen_eval_groups,
        operation=config.operation,
        step=0,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
    )
    unseen_examples, wrong_call_by_id, wrong_target_tokens_by_id = _flatten_groups(unseen_groups)
    if len(unseen_examples) < config.min_unseen_eval_examples:
        raise ValueError(
            f"unseen eval batch realized only {len(unseen_examples)} examples from "
            f"{config.num_unseen_eval_groups} groups, below min_unseen_eval_examples="
            f"{config.min_unseen_eval_examples}; increase num_unseen_eval_groups"
        )

    def _wrong_argument_call_provider(example: Example) -> PrimitiveCall:
        return wrong_call_by_id[id(example)]

    correct_exact_match, correct_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model, unseen_examples, core.tokens, bank, device=core.device
    )
    wrong_exact_match, wrong_predictions, _ = evaluate_exact_match_with_oracle_calls(
        core.model,
        unseen_examples,
        core.tokens,
        bank,
        _wrong_argument_call_provider,
        device=core.device,
    )
    none_exact_match, none_predictions = evaluate_exact_match_no_primitive(
        core.model, unseen_examples, core.tokens, device=core.device
    )

    correct_token_accuracy = _mean_token_accuracy(unseen_examples, correct_predictions)
    wrong_token_accuracy = _mean_token_accuracy(unseen_examples, wrong_predictions)
    none_token_accuracy = _mean_token_accuracy(unseen_examples, none_predictions)
    length_stratified_correct = _length_stratified_metrics(unseen_examples, correct_predictions)

    argument_effect_rate = statistics.fmean(
        float(example.target_tokens != wrong_target_tokens_by_id[id(example)])
        for example in unseen_examples
    )
    exact_match_causal_gap = correct_exact_match - max(wrong_exact_match, none_exact_match)
    token_accuracy_causal_gap = correct_token_accuracy - max(
        wrong_token_accuracy, none_token_accuracy
    )

    return SequenceCounterfactualGateReport(
        operation=config.operation,
        config=config,
        core_steps_trained=config.core_train.steps,
        core_examples_seen=config.core_train.steps * config.core_train.batch_size,
        final_core_train_loss=core.final_core_train_loss,
        primitive_steps_trained=config.primitive_train.steps,
        primitive_examples_seen=primitive_examples_seen,
        final_primitive_train_loss=final_primitive_loss,
        correct_exact_match=correct_exact_match,
        correct_token_accuracy=correct_token_accuracy,
        effectful_wrong_argument_exact_match=wrong_exact_match,
        effectful_wrong_argument_token_accuracy=wrong_token_accuracy,
        none_exact_match=none_exact_match,
        none_token_accuracy=none_token_accuracy,
        exact_match_causal_gap=exact_match_causal_gap,
        token_accuracy_causal_gap=token_accuracy_causal_gap,
        length_stratified_correct=length_stratified_correct,
        argument_effect_rate=argument_effect_rate,
        num_unseen_eval_groups=len(unseen_groups),
        num_unseen_eval_examples=len(unseen_examples),
        mean_eval_group_size=len(unseen_examples) / len(unseen_groups),
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
class SequenceCounterfactualGateMultiSeedReport:
    """A1-R005D-008's verdict aggregated across seeds, for one operation.
    `passed` is computed against the cross-seed *mean* of each arm, matching
    every other Phase A.1 gate's own convention."""

    operation: str
    seeds: tuple[int, ...]
    per_seed: tuple[SequenceCounterfactualGateReport, ...]
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
    exact_match_threshold: float
    token_accuracy_threshold: float
    none_ceiling: float
    min_exact_match_causal_gap: float
    exact_match_passed: bool
    token_accuracy_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    family_count_passed: bool
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "mean_correct_exact_match": self.mean_correct_exact_match,
            "stdev_correct_exact_match": self.stdev_correct_exact_match,
            "min_correct_exact_match": self.min_correct_exact_match,
            "max_correct_exact_match": self.max_correct_exact_match,
            "mean_correct_token_accuracy": self.mean_correct_token_accuracy,
            "mean_effectful_wrong_argument_exact_match": (
                self.mean_effectful_wrong_argument_exact_match
            ),
            "mean_effectful_wrong_argument_token_accuracy": (
                self.mean_effectful_wrong_argument_token_accuracy
            ),
            "mean_none_exact_match": self.mean_none_exact_match,
            "mean_none_token_accuracy": self.mean_none_token_accuracy,
            "mean_exact_match_causal_gap": self.mean_exact_match_causal_gap,
            "mean_token_accuracy_causal_gap": self.mean_token_accuracy_causal_gap,
            "mean_argument_effect_rate": self.mean_argument_effect_rate,
            "exact_match_threshold": self.exact_match_threshold,
            "token_accuracy_threshold": self.token_accuracy_threshold,
            "none_ceiling": self.none_ceiling,
            "min_exact_match_causal_gap": self.min_exact_match_causal_gap,
            "exact_match_passed": self.exact_match_passed,
            "token_accuracy_passed": self.token_accuracy_passed,
            "none_passed": self.none_passed,
            "causal_gap_passed": self.causal_gap_passed,
            "family_count_passed": self.family_count_passed,
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_sequence_counterfactual_gate_multi_seed(
    base_config: SequenceCounterfactualGateConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
    exact_match_threshold: float = EXACT_MATCH_THRESHOLD,
    token_accuracy_threshold: float = TOKEN_ACCURACY_THRESHOLD,
    none_ceiling: float = NONE_CEILING,
    min_exact_match_causal_gap: float = MIN_EXACT_MATCH_CAUSAL_GAP,
) -> SequenceCounterfactualGateMultiSeedReport:
    """Run one gate seed per entry in `seeds` (`base_config.seed` overridden
    per seed) and aggregate. Does not raise if `len(seeds) < MIN_GATE_SEEDS`;
    `meets_seed_policy` reports it honestly instead, matching every other
    Phase A.1 gate.
    """
    per_seed: list[SequenceCounterfactualGateReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        core_metrics_path = (
            run_dir_path / f"seed_{seed}" / "core_metrics.jsonl" if run_dir_path else None
        )
        primitive_metrics_path = (
            run_dir_path / f"seed_{seed}" / "primitive_metrics.jsonl" if run_dir_path else None
        )
        report = run_sequence_counterfactual_gate(
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
    mean_correct_token_accuracy = statistics.fmean(
        report.correct_token_accuracy for report in per_seed
    )
    mean_wrong = statistics.fmean(
        report.effectful_wrong_argument_exact_match for report in per_seed
    )
    mean_wrong_token_accuracy = statistics.fmean(
        report.effectful_wrong_argument_token_accuracy for report in per_seed
    )
    mean_none = statistics.fmean(report.none_exact_match for report in per_seed)
    mean_none_token_accuracy = statistics.fmean(report.none_token_accuracy for report in per_seed)
    mean_exact_match_causal_gap = mean_correct - max(mean_wrong, mean_none)
    mean_token_accuracy_causal_gap = mean_correct_token_accuracy - max(
        mean_wrong_token_accuracy, mean_none_token_accuracy
    )
    mean_argument_effect_rate = statistics.fmean(report.argument_effect_rate for report in per_seed)

    exact_match_passed = mean_correct >= exact_match_threshold
    token_accuracy_passed = mean_correct_token_accuracy >= token_accuracy_threshold
    none_passed = mean_none <= none_ceiling
    causal_gap_passed = mean_exact_match_causal_gap >= min_exact_match_causal_gap
    family_count_passed = all(report.bank_size == 1 for report in per_seed)
    passed = (
        exact_match_passed
        and token_accuracy_passed
        and none_passed
        and causal_gap_passed
        and family_count_passed
    )

    return SequenceCounterfactualGateMultiSeedReport(
        operation=base_config.operation,
        seeds=tuple(report.config.seed for report in per_seed),
        per_seed=tuple(per_seed),
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
        exact_match_threshold=exact_match_threshold,
        token_accuracy_threshold=token_accuracy_threshold,
        none_ceiling=none_ceiling,
        min_exact_match_causal_gap=min_exact_match_causal_gap,
        exact_match_passed=exact_match_passed,
        token_accuracy_passed=token_accuracy_passed,
        none_passed=none_passed,
        causal_gap_passed=causal_gap_passed,
        family_count_passed=family_count_passed,
        passed=passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
