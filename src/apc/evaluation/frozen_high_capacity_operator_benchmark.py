"""Frozen high-capacity operator upper bound (Phase A.1 diagnostic Task
A1-R005E-004, `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`).

`docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md` section 5 ("High-
capacity upper-bound operator") and `docs/EXPERIMENT_PLAN_A1_R005E_
DIAGNOSTIC.md` D-E3 motivate this module: A1-R005E-002's representation
audit (ADR-0039) found that `SHIFT`/`SELECT` retain strong per-position
token/position information in frozen `h_content`, and A1-R005E-003's oracle
latent operator benchmark (ADR-0040) found that perfect *oracle* addressing
on that same frozen `h_content`, decoded through the core's own pretrained
(but content-region-untrained) `decode` head, still fails for `SHIFT`/
`SELECT`/`BIND` -- a decoder/interface artifact, not evidence the
information itself is inaccessible. Neither result asks the question this
task asks: "does an expressive *learned* operator -- with its own readout,
trained on `h_content + argument` -- actually solve these tasks?" This is
`R3` in `REPRESENTATION_OPERATOR_ISOLATION.md` section 3: "the most
important test."

## Why this is not a `ConditionedPrimitive`/`PrimitiveBank` extension

`apc.core.execution._route_and_apply_oracle_calls` (the machinery every
A1-R005/A1-R005D counterfactual gate reuses) flattens `hidden` to `[batch *
positions, d_model]` and calls `primitive.forward_from_calls(selected_hidden,
selected_calls)` on a *row-wise* gather -- every currently registered
`apc.primitives.conditioning.ArgumentConditionedPrimitive` (`Conditioned
Primitive`/`FiLMConditionedPrimitive`/`BasisModulatedConditionedPrimitive`)
therefore transforms each position from only that position's own
`h_content[j]` and the argument, structurally unable to attend to any other
position (`docs/DECISIONS.md` ADR-0039's diagnosis, "every currently
registered `ConditionedPrimitive` ... is pointwise"). Bolting a genuinely
cross-position operator onto that same flattened-row interface would mean
redesigning `apc.core.execution`'s routing/dispatch shape -- exactly what
`docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`'s "No premature operator
rollout" forbids before this diagnostic answers whether frozen `h_content`
even supports it. This module is therefore a **standalone diagnostic**,
matching `apc.evaluation.oracle_latent_operator_benchmark`'s own precedent:
no `Router`, `PrimitiveBank`, `Primitive`, `PlasticWorkspace`, or `apc.
consolidation`/`apc.meta` module is imported anywhere in this file. It reads
`model.encode(...)` directly, runs its own `HighCapacityOperator` module (a
plain `nn.Module`, not an `apc.primitives.primitive.Primitive`), and is
explicitly "upper-bound probes, not candidate production architecture"
(`AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`'s "Diagnostic-only rule").

## Architecture: one shared answer-query design for all four operations

`ShiftOp`/`SelectOp`/`BindOp`/`CountOp` differ in output shape
(`Operation.output_length`): `SHIFT` copies the whole (rotated) input
(`output_length == input_length`), `SELECT` gathers a *strict subset*
(`output_length == max(1, input_length // 2)`, per `SelectOp.output_length`
-- shorter than the input, unlike `SHIFT`), and `BIND`/`COUNT` always emit
exactly one token. Rather than special-casing "read the transformed content
position directly" (valid only for `SHIFT`, where output slot `t` really is
content position `t`) against "pool to one vector" (`COUNT`'s A1-R005E-003
precedent), `HighCapacityOperator` uses one uniform mechanism for all four:
a fixed number of *answer-query* tokens (`answer_query_embedding[0]`,
`[1]`, ... one per output slot, `output_length = Operation.output_length
(len(input_tokens))` computed the same way the real interpreter would) is
concatenated after the content tokens and one argument token, and the whole
sequence self-attends through `n_layers` `nn.TransformerEncoderLayer` blocks
(bidirectional, `2-4` per the task's own "Suggested" -- default `3`). Every
output slot's final state feeds one shared, freshly trained `nn.Linear
(d_operator, vocab_size)` readout. This treats "gather output token `t` from
wherever in `content + argument` it actually lives" as the one capability
under test, uniformly, rather than assuming a positional alignment that
`SELECT` structurally does not have.

**Arguments enter the operator only** (`AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.
md`'s "Task blindness remains invariant"): `default_argument_encoder`
(`apc.primitives.conditioning`, the same encoder class every production
`ArgumentConditionedPrimitive` already uses) embeds the raw argument value,
`arg_proj` lifts it into the operator's own width, and the result is
prepended as one extra token -- the frozen content encoder never sees it,
and `Stable Core` parameters are frozen (`requires_grad_(False)`) before any
operator training begins.

## Fresh readout, not the frozen core's `decode` (ADR-0040's own lesson)

`docs/DECISIONS.md` ADR-0040 found that A1-R005E-003's re-use of the frozen
core's own pretrained `decode` head on content-region hidden states was
close to uninformative (`SHIFT`/`SELECT` token accuracy *below* the
`1/vocab_size` chance floor) because ordinary autoregressive pretraining
never rewards that head for predicting *content* positions, only the answer
span (`apc.core.data` module docstring). This module does not repeat that
mistake: `HighCapacityOperator.readout` is a brand-new `nn.Linear` trained
jointly with the rest of the operator, end to end, by ordinary cross-entropy
against `target_tokens` -- exactly the fix A1-R005E-003's own `COUNT`
readout already validated (fit-fresh, not reuse-stale), generalized here
from a single pooled vector to `output_length` query slots.

## Counterfactual training and evaluation (anti-shortcut, three arms)

Same protocol every A1-R005D retry counterfactual gate (`sequence_
counterfactual_gate`/`bind_counterfactual_gate`/`count_counterfactual_gate`)
established and this task's own "use counterfactual argument groups"
requires: one content sequence is paired with >= `MIN_GROUP_SIZE` distinct
argument values whose *outputs* are pairwise distinct
(`HighCapacityOperatorGroup.__post_init__` checks this eagerly), so content
alone can never explain the target within a training step or an eval
example -- the operator must actually read the argument. Evaluation measures
three arms per unseen example:

1. **Correct** -- the example's own true argument.
2. **Effectful Wrong argument** -- a different group member's argument
   value (guaranteed, by the group invariant, to imply a different true
   output), scored against *this* example's own true target. A high score
   here would mean the operator ignores the argument.
3. **None** -- the argument token zeroed instead of encoded (no task-
   specific signal reaches the operator at all), scored against the same
   true target.

`exact_match_causal_gap = correct_exact_match - max(wrong_exact_match,
none_exact_match)` (`token_accuracy_causal_gap` analogously) -- the same
formula every earlier counterfactual gate uses, reapplied to cross-seed
means at the multi-seed level rather than averaging per-seed gaps.

## Filling in the `None` ceiling (same precedent as every earlier gate)

A1-R005E-004's own "Targets" list states `Correct exact >= 0.90`, `SHIFT/
SELECT token >= 0.98`, `effectful Wrong argument <= 0.30`, `causal gap >=
0.50` -- no `None` ceiling. Every earlier counterfactual gate (A1-R005D-004/
007/008) filled in the same unstated number from its own "materially low"
convention and folded it into `passed` (`docs/DECISIONS.md` ADR-0033 first
sets `NONE_CEILING = 0.30`); this module does the same for consistency
(`NONE_CEILING = 0.30`, `none_passed` included in the per-operation `passed`
verdict below).

## Seed policy

`docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` A1-R005E-004's own text: "run
>= 5 seeds for branch evidence" -- unlike A1-R005E-002/003's 3-seed
diagnostic-only default, this module's `DEFAULT_SEEDS`/`MIN_GATE_SEEDS`
match every A1-R005D counterfactual gate's own `(0, 1, 2, 3, 4)`/`5`.

## Frozen-core budget matches A1-R005E-002/003, not the generic default

`core_train` defaults to `SharedCoreGateTrainConfig(steps=30000)` (the
retry's own budget, reused by A1-R005E-002/003 -- ADR-0039/ADR-0040's "Four
checkpoints, not one" precedent), not `SharedCoreGateTrainConfig`'s own
generic `steps=20000` default, so the frozen `h_content` this module tests
is the same representation already audited/oracle-tested earlier in this
diagnostic chain, keeping the eventual A1-R005E-008 cross-task comparison
meaningful.

## Branch decision is not made here

This module measures; it does not decide. `docs/CODEX_TASKS_A1_R005E_
DIAGNOSTIC.md`'s own branch text ("substantial PASS -> E-005, material FAIL
-> skip E-005 and run E-006") is deliberately qualitative ("substantial"/
"material"), consistent with `docs/design-docs/NEXT_PHASE_DECISION_MATRIX.
md`'s Branch D ("Mixed result ... do not force all operations into one
primitive class"). This module computes a strict per-operation `passed`
(AND of every stated target) and an overall `passed` (AND across
operations); interpreting a mixed result as "substantial" vs "material" is
left to the ADR/decision report (A1-R005E-008), not hard-coded here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
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

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.core.model import DecoderOnlyTransformer
from apc.core.tokens import SharedCoreTokens
from apc.environments.generator import Example, OracleMetadata
from apc.environments.interpreter import run_program
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES, get_operation
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.primitives.conditioning import (
    DEFAULT_ARG_DIM,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    default_argument_encoder,
)
from apc.utils.seed import set_seed

__all__ = [
    "MIN_GROUP_SIZE",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_SEEDS",
    "MIN_GATE_SEEDS",
    "TOKEN_ACCURACY_GATED_OPERATIONS",
    "CORRECT_EXACT_MATCH_THRESHOLD",
    "TOKEN_ACCURACY_THRESHOLD",
    "EFFECTFUL_WRONG_ARGUMENT_CEILING",
    "NONE_CEILING",
    "MIN_CAUSAL_GAP",
    "HighCapacityOperatorGroup",
    "generate_high_capacity_operator_counterfactual_groups",
    "HighCapacityOperator",
    "FrozenHighCapacityOperatorConfig",
    "frozen_high_capacity_operator_config_from_dict",
    "FrozenStableCore",
    "OperationFrozenHighCapacityOperatorReport",
    "FrozenHighCapacityOperatorReport",
    "run_frozen_high_capacity_operator_benchmark",
    "OperationFrozenHighCapacityOperatorSummary",
    "FrozenHighCapacityOperatorMultiSeedReport",
    "run_frozen_high_capacity_operator_benchmark_multi_seed",
]

# A counterfactual group needs at least one "wrong argument" partner besides
# the correct one.
MIN_GROUP_SIZE = 2
# Matches every A1-R005D counterfactual gate's own DEFAULT_GROUP_SIZE
# convention (docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md section 4).
DEFAULT_GROUP_SIZE = 3
_MAX_CONTENT_RESAMPLE_ATTEMPTS = 500

# docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md A1-R005E-004's own "run >= 5 seeds
# for branch evidence" -- matches every A1-R005D counterfactual gate, not
# A1-R005E-002/003's own 3-seed diagnostic-only default.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5

# A1-R005E-004's own "Targets": SHIFT/SELECT token accuracy is separately
# gated; BIND/COUNT's output_length is always 1, so token accuracy and exact
# match coincide by construction for them (same convention as A1-R005E-003).
TOKEN_ACCURACY_GATED_OPERATIONS: tuple[str, ...] = ("SHIFT", "SELECT")

CORRECT_EXACT_MATCH_THRESHOLD = 0.90
TOKEN_ACCURACY_THRESHOLD = 0.98
EFFECTFUL_WRONG_ARGUMENT_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50
# Not restated by A1-R005E-004's own "Targets" list; filled in from every
# earlier counterfactual gate's own precedent (module docstring, "Filling in
# the None ceiling").
NONE_CEILING = 0.30

_EVAL_BATCH_SIZE = 256

# The argument key name PrimitiveCall/ProgramStep.params uses for each
# parameterized operation (apc.environments.operations.Operation.
# required_argument_names).
_ARGUMENT_NAME: dict[str, str] = {
    "SHIFT": "amount",
    "SELECT": "indices",
    "COUNT": "target",
    "BIND": "query_key",
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


def _derive_local_seed(seed: int, step: int, label: str) -> int:
    """Deterministic sub-seed for one `(seed, step, label)` triple,
    independent of `PYTHONHASHSEED` -- same construction as every A1-R005D
    counterfactual gate's own `_derive_local_seed`, kept local here per this
    codebase's "each gate module is independently reviewable end to end"
    convention."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _valid_content_lengths(
    operation: str, sequence_length_range: tuple[int, int]
) -> tuple[int, ...]:
    """The content lengths `operation` can legally execute on, within
    `sequence_length_range` -- every length in range except for `BIND`,
    which (`apc.environments.operations.BindOp.is_valid_for_length`) requires
    an even length >= 2 (module docstring precedent: `bind_counterfactual_
    gate._valid_bind_lengths`)."""
    min_len, max_len = sequence_length_range
    if operation == "BIND":
        return tuple(
            length for length in range(min_len, max_len + 1) if length >= 2 and length % 2 == 0
        )
    return tuple(range(min_len, max_len + 1))


def _operation_output(
    operation: str, content: tuple[int, ...], argument_value: Any, vocab_size: int
) -> tuple[int, ...]:
    """Each operation's own `apply`, reimplemented directly against a raw
    content sequence and a raw argument value (not a full `params` dict) so
    candidate argument values can be pre-filtered for pairwise-distinct
    outputs before committing to building a real `Example` -- same role as
    every A1-R005D counterfactual gate's own `_{sequence,bind,count}_output`."""
    if operation == "SHIFT":
        amount = argument_value % len(content)
        return tuple(content[amount:]) + tuple(content[:amount])
    if operation == "SELECT":
        return tuple(content[i] for i in argument_value)
    if operation == "COUNT":
        return (min(sum(1 for token in content if token == argument_value), vocab_size - 1),)
    if operation == "BIND":
        value = 0
        for i in range(0, len(content) - 1, 2):
            if content[i] == argument_value:
                value = content[i + 1]
        return (value,)
    raise ValueError(
        f"unsupported operation: {operation!r} (expected one of {tuple(_ARGUMENT_NAME)})"
    )


def _argument_candidates(
    operation: str, rng: random.Random, content: tuple[int, ...], vocab_size: int
) -> list[Any]:
    """Every legal argument value for `operation` given `content`, shuffled.
    `SHIFT`'s domain is `range(length)`; `SELECT`'s is every `k`-subset of
    `range(length)` (`k = SelectOp.output_length(length)`, matching `SelectOp.
    sample_params`'s own domain); `COUNT`'s is the whole vocabulary; `BIND`'s
    is the content's own distinct present keys (`BindOp.sample_params`'s own
    domain -- querying an absent key is never something the real operation
    produces)."""
    length = len(content)
    if operation == "SHIFT":
        candidates: list[Any] = list(range(length))
    elif operation == "SELECT":
        k = get_operation("SELECT").output_length(length)
        candidates = [list(combo) for combo in itertools.combinations(range(length), k)]
    elif operation == "COUNT":
        candidates = list(range(vocab_size))
    elif operation == "BIND":
        candidates = sorted(set(content[0::2]))
    else:
        raise ValueError(
            f"unsupported operation: {operation!r} (expected one of {tuple(_ARGUMENT_NAME)})"
        )
    rng.shuffle(candidates)
    return candidates


def _build_example(
    operation: str, input_tokens: tuple[int, ...], argument_value: Any, vocab_size: int, split: str
) -> Example:
    """One real `Example` for `operation(argument_value)` applied to
    `input_tokens`, built the same way every A1-R005D counterfactual gate's
    own `_build_*_example` builds a depth-1 example -- `apc.environments.
    interpreter.run_program` is the same ground-truth executor, just driven
    by an explicit argument value instead of `Operation.sample_params`'s
    random draw."""
    program = Program(
        steps=(
            ProgramStep(operation=operation, params={_ARGUMENT_NAME[operation]: argument_value}),
        )
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


def _correct_argument_value(operation: str, example: Example) -> Any:
    """The argument value that produced `example`'s own target -- read back
    from `example.program` (set by `_build_example` above), rather than
    tracked in a separate side table."""
    assert example.program is not None
    return example.program.steps[0].params[_ARGUMENT_NAME[operation]]


@dataclass(frozen=True)
class HighCapacityOperatorGroup:
    """One content sequence paired with >= `MIN_GROUP_SIZE` argument values
    for `operation` whose outputs are pairwise distinct. `examples[i]` is the
    real `Example` for `argument_values[i]`; every `examples[i].target_tokens`
    differs from every other member's, checked eagerly here rather than
    merely assumed from the construction that produced them."""

    operation: str
    input_tokens: tuple[int, ...]
    examples: tuple[Example, ...]
    argument_values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.operation not in _ARGUMENT_NAME:
            raise ValueError(
                f"operation must be one of {tuple(_ARGUMENT_NAME)}, got {self.operation!r}"
            )
        if len(self.examples) != len(self.argument_values):
            raise ValueError(
                "HighCapacityOperatorGroup requires len(examples) == len(argument_values), "
                f"got {len(self.examples)} and {len(self.argument_values)}"
            )
        if len(self.examples) < MIN_GROUP_SIZE:
            raise ValueError(
                f"HighCapacityOperatorGroup requires >= {MIN_GROUP_SIZE} members, got "
                f"{len(self.examples)}"
            )
        outputs = [example.target_tokens for example in self.examples]
        if len(set(outputs)) != len(outputs):
            raise ValueError(
                "HighCapacityOperatorGroup requires pairwise distinct outputs across its "
                f"members; got outputs {outputs} for argument_values {self.argument_values}"
            )

    def wrong_argument_index(self, member_index: int) -> int:
        """The index of a group member other than `member_index`, guaranteed
        to have a different output (`__post_init__`'s own invariant)."""
        return (member_index + 1) % len(self.examples)


def _generate_group(
    rng: random.Random,
    operation: str,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    group_size: int,
    split: str,
) -> HighCapacityOperatorGroup:
    valid_lengths = _valid_content_lengths(operation, sequence_length_range)
    for _ in range(_MAX_CONTENT_RESAMPLE_ATTEMPTS):
        length = rng.choice(valid_lengths)
        input_tokens = tuple(rng.randrange(vocab_size) for _ in range(length))

        candidates = _argument_candidates(operation, rng, input_tokens, vocab_size)
        chosen_values: list[Any] = []
        seen_outputs: set[tuple[int, ...]] = set()
        for value in candidates:
            output = _operation_output(operation, input_tokens, value, vocab_size)
            if output in seen_outputs:
                continue
            seen_outputs.add(output)
            chosen_values.append(value)
            if len(chosen_values) == group_size:
                break

        if len(chosen_values) >= MIN_GROUP_SIZE:
            examples = tuple(
                _build_example(operation, input_tokens, value, vocab_size, split)
                for value in chosen_values
            )
            return HighCapacityOperatorGroup(
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


def generate_high_capacity_operator_counterfactual_groups(
    seed: int,
    n_groups: int,
    *,
    operation: str,
    step: int,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[HighCapacityOperatorGroup, ...]:
    """Deterministic-by-`(seed, step, split, operation)` counterfactual group
    batch, matching every A1-R005D counterfactual gate's own determinism
    convention (one shared `random.Random` stream for the whole call)."""
    if operation not in _ARGUMENT_NAME:
        raise ValueError(f"operation must be one of {tuple(_ARGUMENT_NAME)}, got {operation!r}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if group_size < MIN_GROUP_SIZE:
        raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {group_size}")
    rng = random.Random(_derive_local_seed(seed, step, f"{split}:{operation}"))
    return tuple(
        _generate_group(rng, operation, vocab_size, sequence_length_range, group_size, split)
        for _ in range(n_groups)
    )


def _flatten_groups(
    groups: Sequence[HighCapacityOperatorGroup],
) -> tuple[list[Example], dict[int, Any], dict[int, tuple[int, ...]]]:
    """Flatten `groups` into one example list plus two `id(example)`-keyed
    side tables: `wrong_argument_by_id` (a different group member's own
    argument value -- the "Wrong argument" value to force for that example,
    per `HighCapacityOperatorGroup.wrong_argument_index`) and `wrong_target_
    tokens_by_id` (that partner's own ground-truth output, used only to
    report `argument_effect_rate`)."""
    examples: list[Example] = []
    wrong_argument_by_id: dict[int, Any] = {}
    wrong_target_tokens_by_id: dict[int, tuple[int, ...]] = {}
    for group in groups:
        for index, example in enumerate(group.examples):
            partner_index = group.wrong_argument_index(index)
            wrong_argument_by_id[id(example)] = group.argument_values[partner_index]
            wrong_target_tokens_by_id[id(example)] = group.examples[partner_index].target_tokens
            examples.append(example)
    return examples, wrong_argument_by_id, wrong_target_tokens_by_id


class HighCapacityOperator(nn.Module):
    """An intentionally expressive, argument-conditioned, cross-position
    transform over frozen `h_content` (Task A1-R005E-004, design doc section
    5). Unlike every currently registered `apc.primitives.conditioning.
    ArgumentConditionedPrimitive` (module docstring: pointwise, one position
    at a time), this reads every content position and the argument together
    through `n_layers` bidirectional self-attention blocks, so it can
    genuinely gather information across positions.

    Input to the transformer stack: one argument token (`arg_proj(arg_
    encoder(argument_values))`, or an all-zero token for the "None" causal-
    ablation arm), `L` content tokens (`content_features[:, i, :]`, the
    "own position" convention `apc.evaluation.representation_audit`/`apc.
    evaluation.oracle_latent_operator_benchmark` already use), and `output_
    length` learned answer-query tokens -- one per output slot (module
    docstring's unified design across `SHIFT`/`SELECT`/`BIND`/`COUNT`).
    A freshly trained `nn.Linear(d_operator, vocab_size)` reads each
    answer-query slot's final state.
    """

    def __init__(
        self,
        operation: str,
        *,
        d_model: int,
        d_operator: int,
        n_layers: int,
        n_head: int,
        d_ff: int,
        vocab_size: int,
        max_sequence_length: int,
        arg_dim: int,
    ) -> None:
        super().__init__()
        if operation not in _ARGUMENT_NAME:
            raise ValueError(f"operation must be one of {tuple(_ARGUMENT_NAME)}, got {operation!r}")
        if d_operator % n_head != 0:
            raise ValueError(f"d_operator ({d_operator}) must be divisible by n_head ({n_head})")
        self.operation = operation
        self.d_operator = d_operator
        self.max_sequence_length = max_sequence_length

        self.content_in_proj = nn.Linear(d_model, d_operator)
        self.content_position_embedding = nn.Embedding(max_sequence_length, d_operator)
        self.answer_query_embedding = nn.Embedding(max_sequence_length, d_operator)
        self.arg_encoder = default_argument_encoder(
            operation,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
        )
        self.arg_proj = nn.Linear(arg_dim, d_operator)
        # dropout=0.0: deterministic forward pass in both train and eval
        # mode, matching every other diagnostic module's own conditioning-
        # architecture convention (e.g. OrderPreservingIndexSetArgumentEncoder).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_operator,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.readout = nn.Linear(d_operator, vocab_size)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None,
    ) -> torch.Tensor:
        """`content_features`: `[batch, Lmax, d_model]`, frozen per-position
        content states (see `_batch_features`). `argument_values=None` is
        the "None" causal-ablation arm: the argument token is zeroed instead
        of encoded, so no task-specific signal reaches the operator at all.
        Returns `[batch, max(output_lengths), vocab_size]`; callers mask
        rows beyond each example's own `output_lengths[row]`."""
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        content_tokens = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )
        content_lengths_t = torch.tensor(content_lengths, device=device).unsqueeze(1)
        content_pad_mask = content_position_ids >= content_lengths_t

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_tokens = self.answer_query_embedding(query_ids)
        output_lengths_t = torch.tensor(output_lengths, device=device).unsqueeze(1)
        query_pad_mask = query_ids >= output_lengths_t

        if argument_values is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)
        arg_pad_mask = torch.zeros(batch, 1, dtype=torch.bool, device=device)

        tokens = torch.cat([arg_token, content_tokens, query_tokens], dim=1)
        key_padding_mask = torch.cat([arg_pad_mask, content_pad_mask, query_pad_mask], dim=1)

        encoded = self.blocks(tokens, src_key_padding_mask=key_padding_mask)
        query_encoded = encoded[:, 1 + lmax :, :]
        return self.readout(query_encoded)


@dataclass(frozen=True)
class FrozenHighCapacityOperatorConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-004
    benchmark run, covering every operation in `operation_names`.

    `core_train` defaults to `steps=30000` (module docstring, "Frozen-core
    budget matches A1-R005E-002/003"), not `SharedCoreGateTrainConfig`'s own
    generic default.
    """

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    operation_names: tuple[str, ...] = PARAMETERIZED_OPERATION_NAMES
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    core_train: SharedCoreGateTrainConfig = field(
        default_factory=lambda: SharedCoreGateTrainConfig(steps=30000)
    )
    d_operator: int = 256
    n_operator_layers: int = 3
    n_operator_head: int = 4
    d_operator_ff: int = 1024
    arg_dim: int = DEFAULT_ARG_DIM
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    operator_train: PrimitiveTrainConfig = field(default_factory=PrimitiveTrainConfig)
    num_unseen_eval_groups: int = 1400
    min_unseen_eval_examples: int = 1024

    def __post_init__(self) -> None:
        if not self.operation_names:
            raise ValueError("operation_names must be non-empty")
        if self.group_size < MIN_GROUP_SIZE:
            raise ValueError(f"group_size must be >= {MIN_GROUP_SIZE}, got {self.group_size}")
        if self.d_operator < 1:
            raise ValueError(f"d_operator must be >= 1, got {self.d_operator}")
        if self.n_operator_layers < 1:
            raise ValueError(f"n_operator_layers must be >= 1, got {self.n_operator_layers}")
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


def frozen_high_capacity_operator_config_from_dict(
    raw: dict[str, Any],
) -> FrozenHighCapacityOperatorConfig:
    """Parse a `configs/phase_a1_frozen_high_capacity_operator_benchmark.yaml`-
    shaped dict, matching every other Phase A.1 gate's convention of filling
    in defaults for whatever the file omits."""
    defaults = FrozenHighCapacityOperatorConfig()
    core_train_raw = raw.get("core_train")
    core_train = (
        dataclasses.replace(defaults.core_train, **core_train_raw)
        if core_train_raw
        else defaults.core_train
    )
    operator_train_raw = raw.get("operator_train")
    operator_train = (
        dataclasses.replace(defaults.operator_train, **operator_train_raw)
        if operator_train_raw
        else defaults.operator_train
    )
    return FrozenHighCapacityOperatorConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        operation_names=tuple(raw.get("operation_names", defaults.operation_names)),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        core_train=core_train,
        d_operator=raw.get("d_operator", defaults.d_operator),
        n_operator_layers=raw.get("n_operator_layers", defaults.n_operator_layers),
        n_operator_head=raw.get("n_operator_head", defaults.n_operator_head),
        d_operator_ff=raw.get("d_operator_ff", defaults.d_operator_ff),
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        operator_train=operator_train,
        num_unseen_eval_groups=raw.get("num_unseen_eval_groups", defaults.num_unseen_eval_groups),
        min_unseen_eval_examples=raw.get(
            "min_unseen_eval_examples", defaults.min_unseen_eval_examples
        ),
    )


@dataclass(frozen=True)
class FrozenStableCore:
    """A Stable Core pretrained task-blind on one operation's ordinary i.i.d.
    stream, then frozen: every parameter has `requires_grad=False`. Own copy
    per operation-specific gate module's own convention (not shared with
    `apc.evaluation.oracle_latent_operator_benchmark`'s identically-named
    class)."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: FrozenHighCapacityOperatorConfig, operation: str, metrics_path: str | Path | None = None
) -> FrozenStableCore:
    core_config = SharedCoreGateConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=(operation,),
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


def _build_operator(
    core: FrozenStableCore, config: FrozenHighCapacityOperatorConfig, operation: str
) -> HighCapacityOperator:
    operator = HighCapacityOperator(
        operation,
        d_model=core.model.config.d_model,
        d_operator=config.d_operator,
        n_layers=config.n_operator_layers,
        n_head=config.n_operator_head,
        d_ff=config.d_operator_ff,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        arg_dim=config.arg_dim,
    )
    return operator.to(core.device)


def _batch_features(
    core: FrozenStableCore, examples: Sequence[Example], operation: str
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Frozen per-position content features for `examples`
    (`content_state[:, 1 + i, :]` is content position `i`'s own state,
    matching `apc.evaluation.representation_audit`/`apc.evaluation.
    oracle_latent_operator_benchmark`'s "state at the token itself"
    convention). No gradient reaches the frozen core. Returns `(content_
    features[batch, Lmax, d_model], content_lengths, output_lengths)`."""
    content_lengths = [len(example.input_tokens) for example in examples]
    lmax = max(content_lengths)
    output_lengths = [
        get_operation(operation).output_length(length) for length in content_lengths
    ]
    with torch.no_grad():
        content_state = core.model.encode(
            collate_content_only_batch(examples, core.tokens, device=core.device)
        )
    content_features = content_state[:, 1 : 1 + lmax, :]
    return content_features, content_lengths, output_lengths


def _labels_for_examples(
    examples: Sequence[Example], output_lengths: Sequence[int], out_max: int, device: torch.device
) -> torch.Tensor:
    labels = torch.full((len(examples), out_max), IGNORE_INDEX, dtype=torch.long, device=device)
    for row, (example, n) in enumerate(zip(examples, output_lengths, strict=True)):
        labels[row, :n] = torch.tensor(example.target_tokens[:n], dtype=torch.long, device=device)
    return labels


def _evaluate_arm(
    core: FrozenStableCore,
    operator: HighCapacityOperator,
    examples: Sequence[Example],
    operation: str,
    *,
    argument_provider: Callable[[Example], Any] | None,
) -> tuple[float, float]:
    """Exact match / token accuracy for `examples` under one causal-ablation
    arm: `argument_provider(example)` supplies the argument value to force
    for that example (`None` overall means the "None" arm -- the argument
    channel is zeroed, see `HighCapacityOperator.forward`). Chunked
    (`_EVAL_BATCH_SIZE`) to bound peak memory over a large unseen eval set.
    """
    operator.eval()
    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), _EVAL_BATCH_SIZE):
            chunk = examples[start : start + _EVAL_BATCH_SIZE]
            content_features, content_lengths, output_lengths = _batch_features(
                core, chunk, operation
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
    return exact_matches / len(examples), correct_tokens / total_tokens


def _train_operator(
    core: FrozenStableCore,
    operator: HighCapacityOperator,
    config: FrozenHighCapacityOperatorConfig,
    operation: str,
    metrics_path: str | Path | None = None,
) -> tuple[float, int]:
    """Train `operator` on counterfactual groups (module docstring,
    "Counterfactual training and evaluation") -- the same content appears
    multiple times per step, each paired with a different (correct) argument
    value, so content alone can never explain the target within a step.
    Returns `(final_loss, total_examples_seen)`."""
    device = core.device
    optimizer = torch.optim.AdamW(
        operator.parameters(),
        lr=config.operator_train.lr,
        weight_decay=config.operator_train.weight_decay,
    )
    # Re-anchor the shared global RNG stream after optimizer construction,
    # per ADR-0016's pattern.
    set_seed(config.seed)

    groups_per_step = max(1, -(-config.operator_train.batch_size // config.group_size))

    metrics_file = None
    if metrics_path is not None:
        metrics_file_path = Path(metrics_path)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_file_path.open("w", encoding="utf-8")

    final_loss = float("nan")
    total_examples_seen = 0
    try:
        for step in range(config.operator_train.steps):
            groups = generate_high_capacity_operator_counterfactual_groups(
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
                core, examples, operation
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
            if config.operator_train.grad_clip > 0:
                nn.utils.clip_grad_norm_(operator.parameters(), config.operator_train.grad_clip)
            optimizer.step()
            final_loss = float(loss.item())

            step_number = step + 1
            if metrics_file is not None and (
                step_number % config.operator_train.eval_every == 0
                or step_number == config.operator_train.steps
            ):
                progress_groups = generate_high_capacity_operator_counterfactual_groups(
                    config.seed,
                    max(
                        1,
                        -(-config.operator_train.progress_eval_examples // config.group_size),
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


@dataclass(frozen=True)
class OperationFrozenHighCapacityOperatorReport:
    """Everything observed while pretraining, freezing, training the one
    `HighCapacityOperator` on top of, and evaluating one seed's causal
    ablation (Correct / effectful Wrong argument / None) for one operation."""

    operation: str
    core_final_train_loss: float
    core_param_count: int
    core_trainable_param_count: int
    operator_param_count: int
    operator_steps_trained: int
    operator_examples_seen: int
    final_operator_train_loss: float
    correct_exact_match: float
    correct_token_accuracy: float
    effectful_wrong_argument_exact_match: float
    effectful_wrong_argument_token_accuracy: float
    none_exact_match: float
    none_token_accuracy: float
    exact_match_causal_gap: float
    token_accuracy_causal_gap: float
    argument_effect_rate: float
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
    config: FrozenHighCapacityOperatorConfig,
    operation: str,
    *,
    core_metrics_path: str | Path | None = None,
    operator_metrics_path: str | Path | None = None,
) -> OperationFrozenHighCapacityOperatorReport:
    """Run the full A1-R005E-004 benchmark for one `(config.seed, operation)`
    pair: pretrain and freeze that operation's dedicated Stable Core, train
    one `HighCapacityOperator` on counterfactual groups, then evaluate
    Correct/effectful Wrong argument/None on a large, independently-drawn
    unseen batch of counterfactual groups."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, operation, metrics_path=core_metrics_path)
    operator = _build_operator(core, config, operation)
    final_operator_loss, operator_examples_seen = _train_operator(
        core, operator, config, operation, metrics_path=operator_metrics_path
    )

    unseen_groups = generate_high_capacity_operator_counterfactual_groups(
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

    return OperationFrozenHighCapacityOperatorReport(
        operation=operation,
        core_final_train_loss=core.final_core_train_loss,
        core_param_count=core.model.num_parameters(),
        core_trainable_param_count=core.model.num_parameters(trainable_only=True),
        operator_param_count=sum(p.numel() for p in operator.parameters()),
        operator_steps_trained=config.operator_train.steps,
        operator_examples_seen=operator_examples_seen,
        final_operator_train_loss=final_operator_loss,
        correct_exact_match=correct_exact_match,
        correct_token_accuracy=correct_token_accuracy,
        effectful_wrong_argument_exact_match=wrong_exact_match,
        effectful_wrong_argument_token_accuracy=wrong_token_accuracy,
        none_exact_match=none_exact_match,
        none_token_accuracy=none_token_accuracy,
        exact_match_causal_gap=exact_match_causal_gap,
        token_accuracy_causal_gap=token_accuracy_causal_gap,
        argument_effect_rate=argument_effect_rate,
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
class FrozenHighCapacityOperatorReport:
    """Everything observed for one seed, across every operation in
    `config.operation_names`."""

    config: FrozenHighCapacityOperatorConfig
    per_operation: dict[str, OperationFrozenHighCapacityOperatorReport]
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


def run_frozen_high_capacity_operator_benchmark(
    config: FrozenHighCapacityOperatorConfig,
    *,
    metrics_dir: str | Path | None = None,
) -> FrozenHighCapacityOperatorReport:
    """Run one seed of the A1-R005E-004 benchmark: for every operation in
    `config.operation_names`, pretrain and freeze its own dedicated Stable
    Core, train its own dedicated `HighCapacityOperator`, and evaluate the
    three-arm causal ablation."""
    start = time.perf_counter()
    per_operation: dict[str, OperationFrozenHighCapacityOperatorReport] = {}
    device_str = "cpu"
    metrics_dir_path = Path(metrics_dir) if metrics_dir is not None else None
    for operation in config.operation_names:
        core_metrics_path = (
            metrics_dir_path / f"{operation}_core_metrics.jsonl" if metrics_dir_path else None
        )
        operator_metrics_path = (
            metrics_dir_path / f"{operation}_operator_metrics.jsonl"
            if metrics_dir_path
            else None
        )
        report = _run_operation_benchmark(
            config,
            operation,
            core_metrics_path=core_metrics_path,
            operator_metrics_path=operator_metrics_path,
        )
        per_operation[operation] = report
        device_str = report.device
    return FrozenHighCapacityOperatorReport(
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
class OperationFrozenHighCapacityOperatorSummary:
    """One operation's three-arm means aggregated across seeds, with
    pass/fail verdicts computed on the cross-seed mean (`docs/DECISIONS.md`
    ADR-0033's precedent)."""

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
    correct_exact_match_passed: bool
    token_accuracy_passed: bool | None
    effectful_wrong_argument_passed: bool
    none_passed: bool
    causal_gap_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FrozenHighCapacityOperatorMultiSeedReport:
    """A1-R005E-004's full result across seeds, with an overall `passed`
    verdict (every operation in `seeds`' shared `operation_names` passes its
    own targets). Interpreting a mixed per-operation result as "substantial"
    vs. "material" (module docstring, "Branch decision is not made here") is
    left to A1-R005E-008."""

    seeds: tuple[int, ...]
    per_seed: tuple[FrozenHighCapacityOperatorReport, ...]
    per_operation_summary: dict[str, OperationFrozenHighCapacityOperatorSummary]
    passed: bool
    meets_seed_policy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": [report.to_dict() for report in self.per_seed],
            "per_operation_summary": {
                name: summary.to_dict() for name, summary in self.per_operation_summary.items()
            },
            "passed": self.passed,
            "meets_seed_policy": self.meets_seed_policy,
        }


def run_frozen_high_capacity_operator_benchmark_multi_seed(
    base_config: FrozenHighCapacityOperatorConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> FrozenHighCapacityOperatorMultiSeedReport:
    """Run one benchmark seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation. Does not raise if
    `len(seeds) < MIN_GATE_SEEDS`; `meets_seed_policy` reports it honestly
    instead, matching every other Phase A.1 gate."""
    per_seed: list[FrozenHighCapacityOperatorReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        report = run_frozen_high_capacity_operator_benchmark(config, metrics_dir=seed_dir)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    per_operation_summary: dict[str, OperationFrozenHighCapacityOperatorSummary] = {}
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

        per_operation_summary[operation] = OperationFrozenHighCapacityOperatorSummary(
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
            correct_exact_match_passed=correct_exact_match_passed,
            token_accuracy_passed=token_accuracy_passed,
            effectful_wrong_argument_passed=effectful_wrong_argument_passed,
            none_passed=none_passed,
            causal_gap_passed=causal_gap_passed,
            passed=passed,
        )

    overall_passed = all(summary.passed for summary in per_operation_summary.values())

    return FrozenHighCapacityOperatorMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
        passed=overall_passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
