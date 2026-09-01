"""Argument conditioning module (Phase A.1 Post-Correction Task A1-R004,
milestone A1-RM4, H2c).

`docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md` and `AGENTS.md`'s
"Parameterized primitives" section establish that `SHIFT`/`SELECT`/`COUNT`/
`BIND` are one reusable primitive *family* each, configured by an argument
(`amount`/`indices`/`target`/`query_key`, `apc.environments.operations.
Operation.required_argument_names`), not one persistent primitive per
argument value ("bad: SHIFT_1, SHIFT_2, ... / preferred: SHIFT(amount)").
`apc.primitives.primitive.Primitive` (Task 004/A1-R003) has no path for an
argument at all -- `apc.core.execution._route_and_apply_oracle_calls` calls
every bank primitive as `primitive(selected_hidden)`, discarding `PrimitiveCall
.arguments` entirely. This module is the "condition transform on argument
embedding" A1-R004 asks for: a typed encoder per hidden-argument shape plus a
primitive subclass that actually consumes the encoded argument in its
forward pass, so `AGENTS.md`'s "arguments that are stored or logged but
ignored by the neural transform do not count as parameterized execution"
cannot be satisfied by accident.

## Conditioning formula

`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 6's "minimal
argument conditioning" (explicitly preferred over a hypernetwork):

```text
argument -> embedding e_a
u = A_i h
c = C_i e_a
delta = B_i phi(u + c)
out = h + delta
```

`ConditionedPrimitive` below implements this directly on top of
`Primitive`'s existing `a_proj`/`b_proj` (`A_i`/`B_i`): it adds one more
linear map `c_proj` (`C_i`, argument-embedding-dim -> rank) and a GELU
nonlinearity `phi` (matching `apc.core.model.DecoderOnlyTransformer`'s own
choice of activation), so a family already trained by A1-R003's parameter-
free path gains argument-conditioning capacity rather than being replaced.

## Typed argument encoders

Every argument the four parameterized operations require is drawn from one
of two shapes (`apc.environments.operations`):

- a single integer -- `SHIFT.amount` (a rotation distance, not a content
  symbol), `COUNT.target`/`BIND.query_key` (a content-vocabulary token,
  `apc.environments.vocab`);
- a variable-length, order-preserving *subset* of sequence positions --
  `SELECT.indices` (`docs/DECISIONS.md` ADR-0023: never a single scalar
  index, see `apc.environments.operations.SelectOp`).

`IntBucketArgumentEncoder` covers the first shape (one shared embedding-
table lookup per raw integer, `value % num_buckets` so an out-of-declared-
range value -- negative or larger than the family's own expected range --
still maps to a well-defined bucket rather than raising: this is the
"invalid argument" half of A1-R004's acceptance criterion, handled by
graceful wraparound rather than a validation error, since a wrong-but-well-
typed argument value is exactly what A1-R005's "correct family + wrong
argument" causal control needs to be able to construct and run, not reject).
`IndexSetArgumentEncoder` covers the second shape: the mean of learned
per-position embeddings over the index subset, each position also taken
modulo the encoder's own declared capacity for the same reason. Two encoder
*classes*, not four, because `SHIFT.amount`/`COUNT.target`/`BIND.query_key`
share the same underlying shape (one bounded integer); `default_argument_
encoder` below is what actually gives each of the four operations its own
named, appropriately-sized instance (own embedding table, own bucket count),
so the module's public surface still reads as "one typed encoder per
argument" the way A1-R004's task text lists them.

## Missing arguments

A *missing* argument (as opposed to an out-of-range one) is refused loudly,
not defaulted: `ConditionedPrimitive.forward`'s `argument_values` parameter
is keyword-only with no default, so a call that forgets it raises `TypeError`
immediately at the call site (Python's own argument-binding check) rather
than silently reusing whatever the module's Python object happens to hold.
This is a deliberate asymmetry from `Primitive.forward(h, gate=1.0)`'s
existing calling convention, where `apc.core.execution.apply_bank`/
`_route_and_apply_oracle_calls` invoke every routed primitive as
`primitive(selected_hidden)` (a single positional argument, gate applied
separately). Were `argument_values` an ordinary optional positional
parameter instead, that exact call pattern would silently bind `None` (or
whatever default) rather than fail -- precisely the kind of "argument
present in the call but ignored by the transform" AGENTS.md rules out.
Making it keyword-only closes that hole structurally: `ConditionedPrimitive`
cannot be routed through the existing argument-blind `apply_bank`/
`_route_and_apply_oracle_calls` call sites without raising, so a bank that
mixes parameter-free `Primitive`s and argument-conditioned families cannot
silently drop the latter's arguments merely by reusing the former's call
path. `forward_from_calls` is the safe, argument-aware entry point a caller
should use instead; it also raises (`KeyError`) if a given `PrimitiveCall`
does not carry this primitive's own `argument_name` -- e.g. a mismatched
call whose `operation` does not match this primitive's family at all.

## Persistent-capacity acceptance

One `ConditionedPrimitive` instance holds exactly one shared `a_proj`/
`b_proj`/`c_proj`/`argument_encoder` regardless of how many distinct argument
values it is ever called with -- there is no per-value parameter allocation
anywhere in this module, and `build_conditioned_primitive` mints exactly one
primitive per operation family, keyed at `apc.environments.task_spec.
operation_id(operation)` (ADR-0024's convention, reused unchanged so a
future oracle-routed benchmark gets bank lookup "for free" the same way
A1-R003's parameter-free primitives did). `tests/test_primitives_
conditioning.py` exercises this directly: a `PrimitiveBank`'s `len(...)`
stays fixed while the same primitive is driven through >=3 distinct argument
values.

## Explicitly out of scope here (A1-R005's job)

- Wiring `PrimitiveCall.arguments` through `apc.core.execution`'s oracle-
  routing entry points (`apply_bank_with_oracle_calls*`,
  `forward_logits_with_oracle_calls*`) so a full model forward pass actually
  reaches a `ConditionedPrimitive` -- those functions still call every bank
  primitive as `primitive(selected_hidden)`, which raises `TypeError` for a
  `ConditionedPrimitive` (see above) rather than silently mishandling it.
- Training any `ConditionedPrimitive` end to end, or measuring Correct /
  Wrong family / Wrong argument / None (`docs/design-docs/
  CAUSAL_PRIMITIVE_EXECUTION.md` section 8) -- that is A1-R005's own STOP
  GATE benchmark (H2c).
- Promotion to a persistent `STABLE` bank, consolidation, or composition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus

__all__ = [
    "ArgumentEncoder",
    "IntBucketArgumentEncoder",
    "IndexSetArgumentEncoder",
    "default_argument_encoder",
    "ConditionedPrimitive",
    "build_conditioned_primitive",
]

DEFAULT_ARG_DIM = 16
# Covers the default sequence_length_range (6, 10) used throughout Phase A.1
# (apc.environments.generator.TaskGenerator's own default) with headroom;
# see IntBucketArgumentEncoder's modulo-bucket handling for values beyond it.
DEFAULT_MAX_SEQUENCE_LENGTH = 32


class ArgumentEncoder(nn.Module, ABC):
    """Maps a batch of raw `PrimitiveCall` argument values (one per batch
    item) to a fixed-size embedding `e_a`, `[batch, arg_dim]` -- the design
    doc's "argument -> embedding e_a" step, factored out so
    `ConditionedPrimitive` stays agnostic to the argument's raw Python
    shape (a bare int vs. a variable-length list of ints)."""

    arg_dim: int

    @abstractmethod
    def forward(self, values: Sequence[Any]) -> torch.Tensor:
        """`values[i]` is the raw argument for batch item `i`. Returns
        `[len(values), self.arg_dim]`."""


class IntBucketArgumentEncoder(ArgumentEncoder):
    """One bounded integer argument -> a learned embedding lookup.

    Covers `SHIFT.amount` and `COUNT.target`/`BIND.query_key` alike (see
    module docstring): all three are, at the representation level, "one
    integer picked from a bounded range." `value % num_buckets` maps any
    integer -- including negative or larger-than-declared-range, i.e. an
    "invalid"/out-of-declared-range argument -- to a well-defined bucket
    instead of raising, so this encoder never rejects a syntactically valid
    `PrimitiveCall` argument.
    """

    def __init__(self, num_buckets: int, arg_dim: int = DEFAULT_ARG_DIM) -> None:
        super().__init__()
        if num_buckets < 1:
            raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
        if arg_dim < 1:
            raise ValueError(f"arg_dim must be >= 1, got {arg_dim}")
        self.num_buckets = num_buckets
        self.arg_dim = arg_dim
        self.embedding = nn.Embedding(num_buckets, arg_dim)

    def forward(self, values: Sequence[int]) -> torch.Tensor:
        if len(values) == 0:
            raise ValueError("IntBucketArgumentEncoder requires at least one batch item")
        device = self.embedding.weight.device
        index = torch.tensor([int(v) for v in values], dtype=torch.long, device=device)
        bucket = torch.remainder(index, self.num_buckets)
        return self.embedding(bucket)


class IndexSetArgumentEncoder(ArgumentEncoder):
    """`SELECT.indices` -> a permutation-invariant embedding: the mean of
    learned per-position embeddings over the (variable-size, per-example)
    index subset. Each position is also taken modulo `max_positions`, so an
    index beyond the encoder's declared capacity still maps to a well-
    defined bucket rather than raising (matching `IntBucketArgumentEncoder`'s
    "invalid argument" handling).
    """

    def __init__(
        self, max_positions: int = DEFAULT_MAX_SEQUENCE_LENGTH, arg_dim: int = DEFAULT_ARG_DIM
    ) -> None:
        super().__init__()
        if max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {max_positions}")
        if arg_dim < 1:
            raise ValueError(f"arg_dim must be >= 1, got {arg_dim}")
        self.max_positions = max_positions
        self.arg_dim = arg_dim
        self.embedding = nn.Embedding(max_positions, arg_dim)

    def forward(self, values: Sequence[Sequence[int]]) -> torch.Tensor:
        if len(values) == 0:
            raise ValueError("IndexSetArgumentEncoder requires at least one batch item")
        device = self.embedding.weight.device
        rows = []
        for indices in values:
            if len(indices) == 0:
                raise ValueError(
                    "IndexSetArgumentEncoder requires a non-empty index subset for every "
                    "batch item (SELECT always samples >=1 index; an empty subset means a "
                    "malformed PrimitiveCall)"
                )
            index = torch.tensor([int(i) for i in indices], dtype=torch.long, device=device)
            bucket = torch.remainder(index, self.max_positions)
            rows.append(self.embedding(bucket).mean(dim=0))
        return torch.stack(rows, dim=0)


def default_argument_encoder(
    operation: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    arg_dim: int = DEFAULT_ARG_DIM,
) -> ArgumentEncoder:
    """The typed encoder A1-R004 registers for one of the four parameterized
    operations' hidden argument, keyed by operation name so each family gets
    its own appropriately-sized instance (own embedding table): `SHIFT` gets
    a rotation-distance bucket table sized to `max_sequence_length`;
    `SELECT` gets a position-set encoder of the same capacity; `COUNT`/`BIND`
    each get their own content-vocabulary bucket table sized to `vocab_size`
    (separate instances -- a count target and a bind key are not the same
    learned concept even though both range over the same vocabulary).

    Raises:
        KeyError: `operation` is not one of `SHIFT`/`SELECT`/`COUNT`/`BIND`.
    """
    if operation == "SHIFT":
        return IntBucketArgumentEncoder(num_buckets=max_sequence_length, arg_dim=arg_dim)
    if operation == "SELECT":
        return IndexSetArgumentEncoder(max_positions=max_sequence_length, arg_dim=arg_dim)
    if operation == "COUNT":
        return IntBucketArgumentEncoder(num_buckets=vocab_size, arg_dim=arg_dim)
    if operation == "BIND":
        return IntBucketArgumentEncoder(num_buckets=vocab_size, arg_dim=arg_dim)
    raise KeyError(
        f"default_argument_encoder has no typed encoder registered for operation {operation!r} "
        "(expected one of SHIFT/SELECT/COUNT/BIND)"
    )


class ConditionedPrimitive(Primitive):
    """Argument-conditioned residual primitive (A1-R004):

        e_a = argument_encoder(argument_values)
        u = A_i(h)
        c = C_i(e_a)
        delta = B_i(phi(u + c))
        out = h + gate * delta

    -- `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 6, on top of
    `Primitive`'s existing `a_proj`/`b_proj` (`A_i`/`B_i`). `a_proj`/`b_proj`
    are shared across every argument value the family is ever called with
    (only the encoder's embedding lookup and the resulting `e_a` vary) --
    the "shared family weights across values" acceptance criterion.

    `argument_values` is keyword-only and has no default: see module
    docstring's "Missing arguments" for why. `gate` keeps `Primitive.
    forward`'s existing default (`1.0`, unconditional application) since
    A1-R004 does not add routing -- oracle or learned selection among
    candidates (including this primitive) remains `apc.core.execution`'s
    job, unchanged by this module.
    """

    def __init__(
        self,
        primitive_id: int,
        config: PrimitiveConfig,
        argument_encoder: ArgumentEncoder,
        argument_name: str,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            config,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.argument_name = argument_name
        self.argument_encoder = argument_encoder
        self.c_proj = nn.Linear(argument_encoder.arg_dim, config.rank, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        self.activation = nn.GELU()

    def forward(  # type: ignore[override]
        self,
        h: torch.Tensor,
        *,
        argument_values: Sequence[Any],
        gate: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """`h`: `[batch, ..., d_model]`. `argument_values[i]` is the raw
        argument for batch item `i` (`h.shape[0]`); it is broadcast across
        every non-batch dimension of `h` (e.g. sequence positions), matching
        `PrimitiveCall.arguments`' one-call-per-example semantics
        (`apc.core.execution._route_and_apply_oracle_calls`).

        Raises:
            ValueError: `len(argument_values) != h.shape[0]`.
        """
        if not self.enabled:
            return h
        if len(argument_values) != h.shape[0]:
            raise ValueError(
                f"argument_values has {len(argument_values)} entries but h's batch "
                f"dimension is {h.shape[0]} -- one argument value per batch item is required"
            )
        self.forward_call_count += 1
        e_a = self.argument_encoder(argument_values)
        c = self.c_proj(e_a)
        u = self.a_proj(h)
        extra_dims = u.dim() - c.dim()
        for _ in range(extra_dims):
            c = c.unsqueeze(1)
        delta = self.b_proj(self.activation(u + c))
        return h + gate * delta

    def forward_from_calls(
        self,
        h: torch.Tensor,
        calls: Sequence[PrimitiveCall],
        *,
        gate: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper: extract this primitive's own argument
        (`self.argument_name`) from each of `calls` and forward. The direct,
        testable link to AGENTS.md's "neural primitive execution must
        actually consume `PrimitiveCall.arguments`" -- unlike `forward`,
        which takes an already-encoder-ready raw value, this takes the
        actual execution-time `PrimitiveCall` objects a router/oracle would
        hand a caller.

        Raises:
            KeyError: some `calls[i]` does not carry `self.argument_name`
                (e.g. `calls[i].operation` is a different family entirely --
                a mismatched/invalid oracle call for this primitive).
        """
        values = []
        for call in calls:
            if self.argument_name not in call.arguments:
                raise KeyError(
                    f"PrimitiveCall for operation {call.operation!r} has no "
                    f"{self.argument_name!r} argument (arguments: {sorted(call.arguments)}) -- "
                    f"cannot condition primitive {self.primitive_id} ({self.argument_name}) on it"
                )
            values.append(call.arguments[self.argument_name])
        return self.forward(h, argument_values=values, gate=gate)


def build_conditioned_primitive(
    primitive_id: int,
    operation: str,
    config: PrimitiveConfig,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    arg_dim: int = DEFAULT_ARG_DIM,
    status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
    created_at_task: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ConditionedPrimitive:
    """Construct one `ConditionedPrimitive` for `operation`, using
    `default_argument_encoder` and `operation`'s single declared required
    argument name (`apc.environments.operations.Operation.
    required_argument_names`). `metadata` defaults to `{"operation":
    operation}`, matching `apc.evaluation.parameter_free_primitive_gate.
    _build_primitive_bank`'s own convention for the parameter-free families.

    Raises:
        KeyError: `operation` is not registered, or has no typed encoder
            (`default_argument_encoder`).
        ValueError: `operation` does not require exactly one argument (every
            currently registered parameterized operation --
            `apc.environments.operations.PARAMETERIZED_OPERATION_NAMES` --
            does; this guards against a future operation with more than one
            hidden argument, which this module does not yet support).
    """
    required = get_operation(operation).required_argument_names
    if len(required) != 1:
        raise ValueError(
            f"build_conditioned_primitive only supports operations with exactly one "
            f"required argument; {operation!r} requires {sorted(required)}"
        )
    (argument_name,) = tuple(required)
    encoder = default_argument_encoder(
        operation, vocab_size=vocab_size, max_sequence_length=max_sequence_length, arg_dim=arg_dim
    )
    return ConditionedPrimitive(
        primitive_id,
        config,
        encoder,
        argument_name,
        status=status,
        created_at_task=created_at_task,
        metadata=metadata if metadata is not None else {"operation": operation},
    )
