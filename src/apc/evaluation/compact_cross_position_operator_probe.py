"""Compact cross-position operator probe (Phase A.1 diagnostic Task
A1-R005E-005, `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`).

## Prerequisite and how it was read

A1-R005E-005's own task text conditions this task on "E-004 substantially
passes." `docs/results/A1_R005E_004_HIGH_CAPACITY_OPERATOR_SUMMARY.md`
(ADR-0041) recorded A1-R005E-004's overall verdict as strict-FAIL (no
operation cleared every acceptance target simultaneously) but with a mixed,
partly-substantial per-operation pattern: `BIND`'s causal gap moved from
statistically zero (`-0.005`) to `0.586`, and `SELECT` cleared `Correct exact
match >= 0.90` outright (`0.919`) with a `0.895` causal gap, missing only
`SHIFT`/`SELECT` token accuracy by `0.002`; `SHIFT`'s low mean was diagnosed
as a bimodal optimization artifact (3 of 5 seeds near-ceiling), not a
representational ceiling; `COUNT` remained the weakest operation. Per
`docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`'s "Branching discipline," this
task itself does not select the next phase -- the user reviewed that summary
and explicitly requested A1-R005E-005 next, which this module treats as the
qualitative "substantial" reading of a mixed result (`docs/design-docs/
NEXT_PHASE_DECISION_MATRIX.md` Branch D), not as a claim that E-004 met its
own strict `passed=true` bar. Whether a compact operator closes enough of the
remaining gap is exactly this task's own question.

## What this tests, and how it differs from A1-R005E-004

`docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md` section 7 ("Compact
operator probe") and `docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md` D-E4
specify a *small* cross-position operator, structurally different from
A1-R005E-004's `HighCapacityOperator` (`apc.evaluation.
frozen_high_capacity_operator_benchmark`, ADR-0041): where that module lets
one argument token, every content-position token, and every answer-query
token mutually self-attend (bidirectional, `n_layers=3` `nn.
TransformerEncoderLayer` blocks, `d_operator=256`), `CompactCrossPosition
Operator` below implements the design doc's own preferred shape literally:

```text
argument -> query/control
h_content -> keys/values
query-conditioned attention/routing
-> small output transform
```

One `nn.MultiheadAttention` cross-attention step (not stacked, not
self-attention): a query token per output slot (`answer_query_embedding
[slot] + arg_proj(e_a)` -- the argument additively steers every slot's own
query, "argument-derived query/control"), attending only to the frozen
content positions ("`h_content` as keys/values"); content tokens never
attend to each other or to the query, and query tokens never attend to each
other. `d_operator=32` (vs. `256`), `n_operator_head=4`, `d_operator_ff=64`
(vs. `1024`) -- "small projections," per the task's own "Suggested." A fresh
`nn.Linear(d_operator, vocab_size)` readout (same ADR-0040 lesson
A1-R005E-004 already applied: never reuse the frozen core's own pretrained
`decode` head on content-region states).

Everything else is held fixed relative to A1-R005E-004 so the operator
architecture is the only controlled variable: identical frozen-core budget
(`core_train.steps=30000`, same as A1-R005E-002/003/004, "the same frozen
`h_content` this diagnostic chain has already audited/oracle-tested"),
identical counterfactual group construction/protocol (Correct / effectful
Wrong argument / None arms, same `causal_gap` formula), identical operator
training budget (`operator_train.steps=8000`, `lr=3e-4`) so a slower/faster
optimization budget cannot explain a size difference, and the identical
>=5-seed policy. `Stable Core` is frozen and re-pretrained per-seed the same
way (this module does not reuse A1-R005E-004's own checkpoints -- no
diagnostic task in this chain has shared checkpoints across modules;
"same budget" is what keeps the comparison meaningful, not literal weight
reuse).

## Why this module is self-contained, not an import from A1-R005E-004

Matching every other Phase A.1 counterfactual gate's own convention ("each
gate module is independently reviewable end to end" --
`apc.evaluation.frozen_high_capacity_operator_benchmark`'s own docstring for
`_derive_local_seed`; `count_counterfactual_gate`/`bind_counterfactual_gate`/
`sequence_counterfactual_gate` each duplicate the same group-construction
shape rather than importing a shared helper module), this module re-
implements its own copy of the operation/group-generation plumbing
(`_operation_output`, `_argument_candidates`, `_build_example`,
`CompactOperatorGroup`, `generate_compact_operator_counterfactual_groups`,
`_flatten_groups`) rather than importing `apc.evaluation.
frozen_high_capacity_operator_benchmark`'s private helpers. The two modules'
group-generation *algorithms* are intentionally identical (same operation
semantics, same distinctness search) so the frozen-core/data budget is
comparable across E-004 and E-005 by construction, even though the literal
per-seed content sequences differ (same as every earlier diagnostic
task-to-task comparison in this chain: comparable protocol and budget, not
shared example arrays).

## Comparison baselines (not computed by this module)

This module measures Correct / effectful Wrong argument / None and the
derived causal gap, plus `operator_param_count`, for the compact operator
alone. It does not itself recompute A1-R005E-004's high-capacity upper bound
or the historical V0/FiLM low-rank `ConditionedPrimitive`/
`FiLMConditionedPrimitive` numbers (ADR-0033/ADR-0035/ADR-0036/ADR-0037) --
those are already measured and recorded; the task's own "Compare" instruction
is satisfied by citing them alongside this module's own results in
`docs/DECISIONS.md`'s ADR for this task, not by re-running them here.

## Diagnostic-only rule

Per `docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`, this remains a diagnostic
prototype ("This is a diagnostic prototype, not a full Primitive Bank
redesign" -- design doc section 7), not a `Primitive`/`PrimitiveBank`
integration: no `Router`, `PrimitiveBank`, `Primitive`, `PlasticWorkspace`,
or `apc.consolidation`/`apc.meta` module is imported here, matching
A1-R005E-004's own precedent.

## Branch decision is not made here

Same as A1-R005E-004 (module docstring, "Branch decision is not made here"):
this module computes a strict per-operation `passed` and an overall `passed`
(AND across operations) using A1-R005E-004's own thresholds (this task names
no separate numeric targets of its own -- its "Positive evidence" list is
qualitative: "materially higher Correct than historical R005, materially
larger causal gap, substantial fraction of upper bound, primitive-scale
size"). Interpreting these against that qualitative bar, and deciding
whether heterogeneous operator primitives deserve a next phase, is done in
`docs/DECISIONS.md`'s ADR for this task, informed by but not computed inside
this module.
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
    "CompactOperatorGroup",
    "generate_compact_operator_counterfactual_groups",
    "CompactCrossPositionOperator",
    "CompactOperatorConfig",
    "compact_operator_config_from_dict",
    "FrozenStableCore",
    "OperationCompactOperatorReport",
    "CompactOperatorReport",
    "run_compact_operator_probe",
    "OperationCompactOperatorSummary",
    "CompactOperatorMultiSeedReport",
    "run_compact_operator_probe_multi_seed",
]

# A counterfactual group needs at least one "wrong argument" partner besides
# the correct one.
MIN_GROUP_SIZE = 2
# Matches every A1-R005D/A1-R005E counterfactual gate's own DEFAULT_GROUP_SIZE
# convention (docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md section 4).
DEFAULT_GROUP_SIZE = 3
_MAX_CONTENT_RESAMPLE_ATTEMPTS = 500

# docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md section 8: ">=5 seeds for D-E3
# and D-E5 claims" -- D-E4 (this task) inherits the same branch-evidence bar
# A1-R005E-004 used, not A1-R005E-002/003's own 3-seed diagnostic-only
# default.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5

# Same acceptance shape as A1-R005E-004 (module docstring, "Comparison
# baselines"): this task states no separate numeric targets of its own, so
# reusing A1-R005E-004's own thresholds keeps "substantial fraction of upper
# bound" measurable on a like-for-like scale.
TOKEN_ACCURACY_GATED_OPERATIONS: tuple[str, ...] = ("SHIFT", "SELECT")

CORRECT_EXACT_MATCH_THRESHOLD = 0.90
TOKEN_ACCURACY_THRESHOLD = 0.98
EFFECTFUL_WRONG_ARGUMENT_CEILING = 0.30
MIN_CAUSAL_GAP = 0.50
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
    independent of `PYTHONHASHSEED` -- same construction every A1-R005D/
    A1-R005E counterfactual gate uses, kept local per this codebase's own
    "each gate module is independently reviewable end to end" convention."""
    digest = hashlib.sha256(f"{seed}:{step}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _valid_content_lengths(
    operation: str, sequence_length_range: tuple[int, int]
) -> tuple[int, ...]:
    """The content lengths `operation` can legally execute on, within
    `sequence_length_range` -- every length in range except for `BIND`,
    which (`apc.environments.operations.BindOp.is_valid_for_length`) requires
    an even length >= 2."""
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
    content sequence and a raw argument value so candidate argument values
    can be pre-filtered for pairwise-distinct outputs before committing to
    building a real `Example`."""
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
    `range(length)` (`k = SelectOp.output_length(length)`); `COUNT`'s is the
    whole vocabulary; `BIND`'s is the content's own distinct present keys
    (`BindOp.sample_params`'s own domain)."""
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
    `input_tokens`, driven by an explicit argument value rather than
    `Operation.sample_params`'s random draw."""
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
    from `example.program` (set by `_build_example` above)."""
    assert example.program is not None
    return example.program.steps[0].params[_ARGUMENT_NAME[operation]]


@dataclass(frozen=True)
class CompactOperatorGroup:
    """One content sequence paired with >= `MIN_GROUP_SIZE` argument values
    for `operation` whose outputs are pairwise distinct. `examples[i]` is the
    real `Example` for `argument_values[i]`; every `examples[i].target_tokens`
    differs from every other member's, checked eagerly here."""

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
                "CompactOperatorGroup requires len(examples) == len(argument_values), "
                f"got {len(self.examples)} and {len(self.argument_values)}"
            )
        if len(self.examples) < MIN_GROUP_SIZE:
            raise ValueError(
                f"CompactOperatorGroup requires >= {MIN_GROUP_SIZE} members, got "
                f"{len(self.examples)}"
            )
        outputs = [example.target_tokens for example in self.examples]
        if len(set(outputs)) != len(outputs):
            raise ValueError(
                "CompactOperatorGroup requires pairwise distinct outputs across its "
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
) -> CompactOperatorGroup:
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
            return CompactOperatorGroup(
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


def generate_compact_operator_counterfactual_groups(
    seed: int,
    n_groups: int,
    *,
    operation: str,
    step: int,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[CompactOperatorGroup, ...]:
    """Deterministic-by-`(seed, step, split, operation)` counterfactual group
    batch, matching every A1-R005D/A1-R005E counterfactual gate's own
    determinism convention (one shared `random.Random` stream per call)."""
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
    groups: Sequence[CompactOperatorGroup],
) -> tuple[list[Example], dict[int, Any], dict[int, tuple[int, ...]]]:
    """Flatten `groups` into one example list plus two `id(example)`-keyed
    side tables: `wrong_argument_by_id` (a different group member's own
    argument value -- the "Wrong argument" value to force for that example)
    and `wrong_target_tokens_by_id` (that partner's own ground-truth output,
    used only to report `argument_effect_rate`)."""
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


class CompactCrossPositionOperator(nn.Module):
    """A deliberately small, argument-conditioned, cross-position transform
    over frozen `h_content` (Task A1-R005E-005, design doc section 7). Unlike
    `apc.evaluation.frozen_high_capacity_operator_benchmark.
    HighCapacityOperator` (bidirectional self-attention across content +
    argument + query tokens, `n_layers` stacked blocks), this implements the
    design doc's preferred shape directly and literally: one query-conditioned
    cross-attention step, content tokens as keys/values only, one small
    output transform.

    Per output slot (`output_length = Operation.output_length(len(input_
    tokens))`, computed the same way the real interpreter would, one shared
    mechanism across `SHIFT`/`SELECT`/`BIND`/`COUNT` as in A1-R005E-004): a
    query token is built as `answer_query_embedding[slot] + arg_proj(e_a)`
    ("argument-derived query/control" -- the argument additively steers every
    slot's own query, the query itself carries no content information). This
    query cross-attends (`nn.MultiheadAttention`, single step, not stacked)
    over the frozen content positions only (`content_in_proj(h_content) +
    content_position_embedding`, "`h_content` as keys/values") -- content
    tokens never attend to each other or to the query, and query slots never
    attend to each other (both deliberately absent, unlike `HighCapacity
    Operator`'s full bidirectional self-attention). One small residual
    feed-forward sublayer ("-> small output transform") and a freshly
    initialized `nn.Linear(d_operator, vocab_size)` readout (ADR-0040's
    lesson: never reuse the frozen core's own pretrained `decode` head on
    content-region states) complete the block.
    """

    def __init__(
        self,
        operation: str,
        *,
        d_model: int,
        d_operator: int,
        n_head: int,
        d_operator_ff: int,
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
        # One single cross-attention step -- not stacked, no self-attention
        # among content or query tokens (module docstring). batch_first=True
        # matches every other attention module's own convention in this
        # codebase (HighCapacityOperator, OrderPreservingIndexSetArgumentEncoder).
        self.cross_attn = nn.MultiheadAttention(d_operator, n_head, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_operator)
        # "small output transform": one small residual feed-forward sublayer,
        # not a stack -- deliberately narrow (d_operator_ff, not the usual
        # 4x-width transformer convention) to keep this operator primitive-
        # scale (module docstring's own size accounting).
        self.ffn = nn.Sequential(
            nn.Linear(d_operator, d_operator_ff),
            nn.GELU(),
            nn.Linear(d_operator_ff, d_operator),
        )
        self.ffn_norm = nn.LayerNorm(d_operator)
        self.readout = nn.Linear(d_operator, vocab_size)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None,
    ) -> torch.Tensor:
        """`content_features`: `[batch, Lmax, d_model]`, frozen per-position
        content states. `argument_values=None` is the "None" causal-ablation
        arm: the argument token is zeroed instead of encoded, so no
        task-specific signal reaches the operator at all -- the query then
        carries only its own slot identity. Returns `[batch,
        max(output_lengths), vocab_size]`; callers mask rows beyond each
        example's own `output_lengths[row]` (rows beyond an example's own
        `output_length` are computed but never read, same convention
        `HighCapacityOperator.forward` uses)."""
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        kv = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )
        content_lengths_t = torch.tensor(content_lengths, device=device).unsqueeze(1)
        # True marks a padded (beyond this example's own content length)
        # key/value position -- nn.MultiheadAttention's own key_padding_mask
        # convention ("True = do not attend").
        content_pad_mask = content_position_ids >= content_lengths_t

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)
        # Broadcast-add the single argument token to every output slot's own
        # query -- "argument-derived query/control": the argument steers
        # what every slot asks for, the slot embedding says which output
        # position is being asked about.
        query = query_slots + arg_token

        attn_out, _ = self.cross_attn(
            query, kv, kv, key_padding_mask=content_pad_mask, need_weights=False
        )
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)


@dataclass(frozen=True)
class CompactOperatorConfig:
    """Explicit, serializable configuration for one seed's A1-R005E-005
    benchmark run, covering every operation in `operation_names`.

    `core_train` defaults to `steps=30000` (module docstring, "the same
    frozen `h_content` this diagnostic chain has already audited/oracle-
    tested"), matching A1-R005E-002/003/004, not `SharedCoreGateTrainConfig`'s
    own generic `steps=20000` default. `operator_train` defaults match
    A1-R005E-004's own operator budget (`steps=8000`) so training budget is
    not a confound in the operator-size comparison.
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
    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
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


def compact_operator_config_from_dict(raw: dict[str, Any]) -> CompactOperatorConfig:
    """Parse a `configs/phase_a1_compact_cross_position_operator_probe.yaml`-
    shaped dict, matching every other Phase A.1 gate's convention of filling
    in defaults for whatever the file omits."""
    defaults = CompactOperatorConfig()
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
    return CompactOperatorConfig(
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
    `apc.evaluation.frozen_high_capacity_operator_benchmark`'s identically-
    named class)."""

    model: DecoderOnlyTransformer
    tokens: SharedCoreTokens
    final_core_train_loss: float
    device: torch.device


def _pretrain_frozen_stable_core(
    config: CompactOperatorConfig, operation: str, metrics_path: str | Path | None = None
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
    core: FrozenStableCore, config: CompactOperatorConfig, operation: str
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
    core: FrozenStableCore, examples: Sequence[Example], operation: str
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Frozen per-position content features for `examples`
    (`content_state[:, 1 + i, :]` is content position `i`'s own state,
    matching `apc.evaluation.representation_audit`/`apc.evaluation.
    oracle_latent_operator_benchmark`/`apc.evaluation.
    frozen_high_capacity_operator_benchmark`'s "state at the token itself"
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
    operator: CompactCrossPositionOperator,
    examples: Sequence[Example],
    operation: str,
    *,
    argument_provider: Callable[[Example], Any] | None,
) -> tuple[float, float]:
    """Exact match / token accuracy for `examples` under one causal-ablation
    arm: `argument_provider(example)` supplies the argument value to force
    for that example (`None` overall means the "None" arm). Chunked
    (`_EVAL_BATCH_SIZE`) to bound peak memory over a large unseen eval set."""
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
    operator: CompactCrossPositionOperator,
    config: CompactOperatorConfig,
    operation: str,
    metrics_path: str | Path | None = None,
) -> tuple[float, int]:
    """Train `operator` on counterfactual groups -- the same content appears
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
                progress_groups = generate_compact_operator_counterfactual_groups(
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
class OperationCompactOperatorReport:
    """Everything observed while pretraining, freezing, training the one
    `CompactCrossPositionOperator` on top of, and evaluating one seed's
    causal ablation (Correct / effectful Wrong argument / None) for one
    operation."""

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
    config: CompactOperatorConfig,
    operation: str,
    *,
    core_metrics_path: str | Path | None = None,
    operator_metrics_path: str | Path | None = None,
) -> OperationCompactOperatorReport:
    """Run the full A1-R005E-005 benchmark for one `(config.seed, operation)`
    pair: pretrain and freeze that operation's dedicated Stable Core, train
    one `CompactCrossPositionOperator` on counterfactual groups, then
    evaluate Correct/effectful Wrong argument/None on a large,
    independently-drawn unseen batch of counterfactual groups."""
    start = time.perf_counter()
    core = _pretrain_frozen_stable_core(config, operation, metrics_path=core_metrics_path)
    operator = _build_operator(core, config, operation)
    final_operator_loss, operator_examples_seen = _train_operator(
        core, operator, config, operation, metrics_path=operator_metrics_path
    )

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

    return OperationCompactOperatorReport(
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
class CompactOperatorReport:
    """Everything observed for one seed, across every operation in
    `config.operation_names`."""

    config: CompactOperatorConfig
    per_operation: dict[str, OperationCompactOperatorReport]
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


def run_compact_operator_probe(
    config: CompactOperatorConfig,
    *,
    metrics_dir: str | Path | None = None,
) -> CompactOperatorReport:
    """Run one seed of the A1-R005E-005 benchmark: for every operation in
    `config.operation_names`, pretrain and freeze its own dedicated Stable
    Core, train its own dedicated `CompactCrossPositionOperator`, and
    evaluate the three-arm causal ablation."""
    start = time.perf_counter()
    per_operation: dict[str, OperationCompactOperatorReport] = {}
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
    return CompactOperatorReport(
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
class OperationCompactOperatorSummary:
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
class CompactOperatorMultiSeedReport:
    """A1-R005E-005's full result across seeds, with an overall `passed`
    verdict (every operation in `seeds`' shared `operation_names` passes its
    own targets). Interpreting these numbers against the task's own
    qualitative "positive evidence" bar (materially higher Correct than
    historical R005, materially larger causal gap, substantial fraction of
    upper bound, primitive-scale size) is left to `docs/DECISIONS.md`'s ADR
    for this task, per `frozen_high_capacity_operator_benchmark`'s own
    precedent ("Branch decision is not made here")."""

    seeds: tuple[int, ...]
    per_seed: tuple[CompactOperatorReport, ...]
    per_operation_summary: dict[str, OperationCompactOperatorSummary]
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


def run_compact_operator_probe_multi_seed(
    base_config: CompactOperatorConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    run_dir: str | Path | None = None,
) -> CompactOperatorMultiSeedReport:
    """Run one benchmark seed per entry in `seeds` (`base_config.seed`
    overridden per seed) and aggregate per operation. Does not raise if
    `len(seeds) < MIN_GATE_SEEDS`; `meets_seed_policy` reports it honestly
    instead, matching every other Phase A.1 gate."""
    per_seed: list[CompactOperatorReport] = []
    run_dir_path = Path(run_dir) if run_dir is not None else None
    for seed in seeds:
        config = dataclasses.replace(base_config, seed=seed)
        seed_dir = run_dir_path / f"seed_{seed}" if run_dir_path else None
        report = run_compact_operator_probe(config, metrics_dir=seed_dir)
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        per_seed.append(report)

    per_operation_summary: dict[str, OperationCompactOperatorSummary] = {}
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

        per_operation_summary[operation] = OperationCompactOperatorSummary(
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
            operator_param_count=operator_param_count,
            correct_exact_match_passed=correct_exact_match_passed,
            token_accuracy_passed=token_accuracy_passed,
            effectful_wrong_argument_passed=effectful_wrong_argument_passed,
            none_passed=none_passed,
            causal_gap_passed=causal_gap_passed,
            passed=passed,
        )

    overall_passed = all(summary.passed for summary in per_operation_summary.values())

    return CompactOperatorMultiSeedReport(
        seeds=tuple(seeds),
        per_seed=tuple(per_seed),
        per_operation_summary=per_operation_summary,
        passed=overall_passed,
        meets_seed_policy=len(seeds) >= MIN_GATE_SEEDS,
    )
