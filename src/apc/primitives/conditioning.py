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
modulo the encoder's own declared capacity for the same reason. `default_
argument_encoder` below is what actually gives each of the four operations
its own named, appropriately-sized instance (own embedding table, own
bucket count), so the module's public surface still reads as "one typed
encoder per argument" the way A1-R004's task text lists them.

### Update (Task A1-R005D-003): `SELECT` no longer defaults to mean pooling

`docs/DECISIONS.md` ADR-0031 (Task A1-R005D-002) found `IndexSetArgumentEncoder`'s
mean pooling to be a *structural*, weight-independent collision: any
permutation of the same index multiset (e.g. `[1, 3]` vs `[3, 1]`) encodes
identically, for any embedding weights, because mean is commutative and
order/position information is never fed into the encoder at all. Adding a
per-slot position embedding before a *linear* reduction does not fix this --
summing the same `(index, position)` pairs in a different pairing is still
just `(sum of index embeddings) + (sum of position embeddings)`, independent
of which index sat at which position. `OrderPreservingIndexSetArgumentEncoder`
below fixes this by inserting one genuinely nonlinear mixing step (a single
masked self-attention layer, whose query/key dot products are quadratic in
the per-slot `index_embedding + position_embedding` vectors, so which index
occupies which slot changes the attention weights themselves) *before* the
final masked-mean reduction. `default_argument_encoder("SELECT", ...)` now
returns this encoder; `IndexSetArgumentEncoder` itself is left exactly as
A1-R004 built it (so `tests/test_primitives_conditioning.py`'s existing
direct-construction tests for it keep demonstrating the same structural
fact), just no longer reachable through the default factory.

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

### Update (Task A1-R005D-006): conditioning architecture comparison

`docs/DECISIONS.md` ADR-0034 (Task A1-R005D-005) found that capacity/
training increases alone do not close `COUNT`'s causal gap, and pointed at
F3 (`docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md`: "additive
conditioning is too weak to make the residual actually depend on the
argument") as the leading remaining candidate: `ConditionedPrimitive`'s
formula sums the content and argument signals (`u + c`) *before* the shared
nonlinearity/projection, which structurally lets the network fit most of the
training loss from the larger content term while treating the argument term
as a small perturbation. `docs/CODEX_TASKS_A1_R005_RETRY.md` A1-R005D-006
("Conditioning architecture comparison") asks to compare that additive
formula (**V0**, `ConditionedPrimitive`, unchanged) against two alternative
formulas that change *how* the argument affects the residual rather than
merely how much capacity/training it gets:

- **V1**, `FiLMConditionedPrimitive`: gated multiplicative (FiLM-style)
  conditioning -- the argument produces a per-channel scale (`gamma`) and
  shift (`beta`) applied to the content projection `u = A_i(h)` itself
  (`(1 + gamma) * u + beta`), rather than only ever being summed alongside
  it. This directly tests whether the argument needs to *gate* the content
  signal to become causally load-bearing.
- **V2**, `BasisModulatedConditionedPrimitive`: keeps V0's additive-sum
  structure (`delta = B(phi(u + c))`) but restructures how `c` is produced --
  instead of a free linear map `C_i(e_a): R^arg_dim -> R^rank` (any point in
  `R^rank` reachable for some `e_a`), `c` is constrained to a soft mixture
  (`softmax`-weighted) over a small, shared learned basis (`num_basis`
  vectors). This tests the complementary hypothesis that an *unconstrained*
  conditioning map is too easy for training to route around, rather than too
  weak.

`ArgumentConditionedPrimitive` (new abstract base below, shared by all three)
is what lets `apc.core.execution._route_and_apply_oracle_calls` dispatch any
of them through `forward_from_calls` with no further change to that module:
it now checks `isinstance(primitive, ArgumentConditionedPrimitive)` instead
of `isinstance(primitive, ConditionedPrimitive)` specifically (a strict
generalization -- every existing `ConditionedPrimitive` call site is
unaffected, since `ConditionedPrimitive` is itself one such subclass).
`build_argument_conditioned_primitive` is `build_conditioned_primitive`'s own
generalization across `ConditioningVariant`, so a caller (`apc.evaluation.
count_counterfactual_gate`, generalized in the same task) can hold the
operation/encoder-resolution logic fixed and vary only which subclass gets
constructed -- the controlled comparison A1-R005D-006's own "Keep Stable
Core and data fixed" asks for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
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
    "OrderPreservingIndexSetArgumentEncoder",
    "default_argument_encoder",
    "ArgumentConditionedPrimitive",
    "ConditionedPrimitive",
    "FiLMConditionedPrimitive",
    "BasisModulatedConditionedPrimitive",
    "ConditioningVariant",
    "build_conditioned_primitive",
    "build_argument_conditioned_primitive",
]

DEFAULT_ARG_DIM = 16
# Covers the default sequence_length_range (6, 10) used throughout Phase A.1
# (apc.environments.generator.TaskGenerator's own default) with headroom;
# see IntBucketArgumentEncoder's modulo-bucket handling for values beyond it.
DEFAULT_MAX_SEQUENCE_LENGTH = 32
# BasisModulatedConditionedPrimitive's (V2) default basis-dictionary size --
# small enough that mix_proj + basis together stay smaller than c_proj's own
# arg_dim * rank at this experiment's arg_dim=16, rank=8 baseline (module
# docstring's A1-R005D-006 update, "do not introduce ... unnecessarily
# general function signatures").
DEFAULT_NUM_BASIS_VECTORS = 8


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


class OrderPreservingIndexSetArgumentEncoder(ArgumentEncoder):
    """`SELECT.indices` -> an order- and multiplicity-preserving embedding
    (Task A1-R005D-003, `docs/DECISIONS.md` ADR-0031/ADR-0032).

    Per selected position `i` (0-indexed by its position *within the
    argument list*, not the sequence position it names): `token_i =
    index_embedding[value_i mod max_positions] + position_embedding[i mod
    max_positions]`. The `token` sequence is then passed through one
    `nn.TransformerEncoderLayer` (`dropout=0.0`, so the forward pass is
    deterministic in both train and eval mode) with a padding mask for
    batches of unequal-length index lists, and reduced to a fixed-size
    `[batch, arg_dim]` vector by masked mean pooling over the valid
    (non-pad) positions.

    Unlike `IndexSetArgumentEncoder`, adding `position_embedding` alone does
    not break the mean-pooling collision: summing the same set of `(index,
    position)` pairs in a different pairing is still just `(sum of index
    embeddings) + (sum of position embeddings)`, independent of which index
    sat at which slot. The self-attention layer is the part that actually
    distinguishes an ordering: its query/key dot products are quadratic in
    the per-slot vectors, so the attention weights themselves (and hence the
    pre-pooling encoded sequence) depend on which index occupies which slot,
    not merely on the multiset of indices present.

    `max_positions` bounds both tables (an index value beyond it, and an
    argument-list position beyond it, both wrap via modulo -- matching
    `IntBucketArgumentEncoder`/`IndexSetArgumentEncoder`'s existing
    "invalid argument" handling): a single parameter is enough because
    `apc.environments.operations.SelectOp.output_length` never selects more
    positions than the input sequence itself has, so the same declared
    sequence-length capacity already bounds both the index values and the
    number of selected positions.
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
        self.index_embedding = nn.Embedding(max_positions, arg_dim)
        self.position_embedding = nn.Embedding(max_positions, arg_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=arg_dim,
            nhead=1,
            dim_feedforward=2 * arg_dim,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, values: Sequence[Sequence[int]]) -> torch.Tensor:
        if len(values) == 0:
            raise ValueError(
                "OrderPreservingIndexSetArgumentEncoder requires at least one batch item"
            )
        for indices in values:
            if len(indices) == 0:
                raise ValueError(
                    "OrderPreservingIndexSetArgumentEncoder requires a non-empty index "
                    "subset for every batch item (SELECT always samples >=1 index; an "
                    "empty subset means a malformed PrimitiveCall)"
                )
        device = self.index_embedding.weight.device
        batch_size = len(values)
        max_len = max(len(indices) for indices in values)

        index_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        # True marks a padded (invalid) slot -- nn.TransformerEncoderLayer's
        # own `src_key_padding_mask` convention.
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for row, indices in enumerate(values):
            n = len(indices)
            raw = torch.tensor([int(i) for i in indices], dtype=torch.long, device=device)
            index_ids[row, :n] = torch.remainder(raw, self.max_positions)
            position_ids[row, :n] = torch.remainder(
                torch.arange(n, device=device), self.max_positions
            )
            padding_mask[row, :n] = False

        tokens = self.index_embedding(index_ids) + self.position_embedding(position_ids)
        encoded = self.encoder_layer(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return pooled


def default_argument_encoder(
    operation: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    arg_dim: int = DEFAULT_ARG_DIM,
) -> ArgumentEncoder:
    """The typed encoder registered for one of the four parameterized
    operations' hidden argument, keyed by operation name so each family gets
    its own appropriately-sized instance (own embedding table): `SHIFT` gets
    a rotation-distance bucket table sized to `max_sequence_length`;
    `SELECT` gets an order-preserving position-set encoder
    (`OrderPreservingIndexSetArgumentEncoder`, Task A1-R005D-003) of the same
    capacity; `COUNT`/`BIND` each get their own content-vocabulary bucket
    table sized to `vocab_size` (separate instances -- a count target and a
    bind key are not the same learned concept even though both range over
    the same vocabulary).

    Raises:
        KeyError: `operation` is not one of `SHIFT`/`SELECT`/`COUNT`/`BIND`.
    """
    if operation == "SHIFT":
        return IntBucketArgumentEncoder(num_buckets=max_sequence_length, arg_dim=arg_dim)
    if operation == "SELECT":
        return OrderPreservingIndexSetArgumentEncoder(
            max_positions=max_sequence_length, arg_dim=arg_dim
        )
    if operation == "COUNT":
        return IntBucketArgumentEncoder(num_buckets=vocab_size, arg_dim=arg_dim)
    if operation == "BIND":
        return IntBucketArgumentEncoder(num_buckets=vocab_size, arg_dim=arg_dim)
    raise KeyError(
        f"default_argument_encoder has no typed encoder registered for operation {operation!r} "
        "(expected one of SHIFT/SELECT/COUNT/BIND)"
    )


class ArgumentConditionedPrimitive(Primitive, ABC):
    """Common base for every primitive family whose residual transform
    consumes `PrimitiveCall.arguments` (Task A1-R005D-006 -- see module
    docstring's "Update" section).

    `apc.core.execution._route_and_apply_oracle_calls` dispatches on this
    base (`isinstance(primitive, ArgumentConditionedPrimitive)`) rather than
    on `ConditionedPrimitive` specifically, so a different argument-
    conditioning *formula* -- `ConditionedPrimitive` (V0, additive),
    `FiLMConditionedPrimitive` (V1, gated multiplicative), or
    `BasisModulatedConditionedPrimitive` (V2, basis modulation) -- can be
    oracle-routed through the existing execution path with no further change
    to `apc.core.execution` for a future variant.

    Every subclass sets `self.argument_name` (the single `PrimitiveCall.
    arguments` key it reads, matching `Operation.required_argument_names`)
    and `self.argument_encoder` (an `ArgumentEncoder`) during `__init__`, and
    implements `forward(h, *, argument_values, gate=1.0)` with the same
    keyword-only-`argument_values` convention `ConditionedPrimitive`
    originally established (module docstring, "Missing arguments") --
    `forward_from_calls` below is shared, unmodified, across every subclass,
    since it only ever touches those two common attributes plus
    `self.forward`.
    """

    argument_name: str
    argument_encoder: ArgumentEncoder

    @abstractmethod
    def forward(  # type: ignore[override]
        self,
        h: torch.Tensor,
        *,
        argument_values: Sequence[Any],
        gate: torch.Tensor | float = 1.0,
    ) -> torch.Tensor: ...

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


class ConditionedPrimitive(ArgumentConditionedPrimitive):
    """Argument-conditioned residual primitive, V0/additive (A1-R004):

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
    job, unchanged by this module. `forward_from_calls` is inherited
    unmodified from `ArgumentConditionedPrimitive` (Task A1-R005D-006).
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


class FiLMConditionedPrimitive(ArgumentConditionedPrimitive):
    """Argument-conditioned residual primitive, V1/gated multiplicative
    (Task A1-R005D-006 -- module docstring's "Update" section):

        e_a = argument_encoder(argument_values)
        u = A_i(h)
        gamma, beta = split(FiLM_i(e_a), rank)
        delta = B_i(phi((1 + gamma) * u + beta))
        out = h + gate * delta

    Follows Perez et al.'s FiLM ("Feature-wise Linear Modulation"): the
    argument produces a per-channel affine transform of the *content*
    projection `u` itself, rather than only ever being summed alongside it
    (`ConditionedPrimitive`'s V0). `docs/design-docs/
    PARAMETERIZED_PRIMITIVE_RETRY.md` F3 names V0's additive-sum structure
    (`B(phi(A(h) + C(e_a)))`) as a candidate bottleneck: summing before the
    shared nonlinearity/projection lets the network fit most of the training
    loss from the (larger, more informative) content term while treating the
    argument term as a small perturbation (`docs/DECISIONS.md` ADR-0034).
    FiLM makes the argument *gate* every one of `u`'s `rank` channels
    (`gamma`) in addition to shifting them (`beta`), so the argument cannot
    be trained away as a small additive nudge -- it multiplicatively
    modulates the content signal the residual is built from.

    `gamma` is parameterized as `1 + FiLM_i(e_a)[..., :rank]` (not the raw
    projection), so a freshly constructed primitive starts at
    `gamma ~= 1, beta ~= 0` -- an approximately-identity affine transform --
    matching `c_proj`'s own small-std init intent in `ConditionedPrimitive.
    __init__`; combined with `Primitive.__init__`'s zero-initialized
    `b_proj`, the whole residual is exactly zero at init regardless of
    `gamma`/`beta` (`b_proj` gates everything to zero until trained), so the
    `1 +` offset only starts to matter once `b_proj` moves away from zero
    during training.

    One shared `film_proj` (`FiLM_i`, `Linear(arg_dim, 2 * rank)`) replaces
    `ConditionedPrimitive.c_proj` -- roughly double `c_proj`'s parameter
    count for the same `arg_dim`/`rank`, reported via `num_parameters()`
    alongside V0/V2 for A1-R005D-006's "report parameter counts" acceptance
    criterion. `a_proj`/`b_proj`/`argument_encoder` are otherwise identical
    in shape and initialization to `ConditionedPrimitive`'s, so the
    conditioning formula is the only variable a controlled V0-vs-V1
    comparison needs to change (`apc.evaluation.count_counterfactual_gate`'s
    A1-R005D-006 update keeps the Stable Core, data, `primitive_rank`, and
    `arg_dim` fixed across variants, per A1-R005D-006's own "Keep Stable
    Core and data fixed").
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
        self.film_proj = nn.Linear(argument_encoder.arg_dim, 2 * config.rank, bias=True)
        nn.init.normal_(self.film_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.film_proj.bias)
        self.activation = nn.GELU()

    def forward(  # type: ignore[override]
        self,
        h: torch.Tensor,
        *,
        argument_values: Sequence[Any],
        gate: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """See class docstring for the FiLM conditioning formula. Same
        `argument_values`/broadcast/error-handling conventions as
        `ConditionedPrimitive.forward`.

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
        gamma, beta = self.film_proj(e_a).chunk(2, dim=-1)
        u = self.a_proj(h)
        extra_dims = u.dim() - gamma.dim()
        for _ in range(extra_dims):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        delta = self.b_proj(self.activation((1.0 + gamma) * u + beta))
        return h + gate * delta


class BasisModulatedConditionedPrimitive(ArgumentConditionedPrimitive):
    """Argument-conditioned residual primitive, V2/basis modulation, optional
    (Task A1-R005D-006 -- module docstring's "Update" section):

        e_a = argument_encoder(argument_values)
        u = A_i(h)
        mix = softmax(mix_proj(e_a))        # [batch, num_basis]
        c = mix @ basis                     # [batch, rank]
        delta = B_i(phi(u + c))
        out = h + gate * delta

    Where `FiLMConditionedPrimitive` (V1) replaces the additive-sum step
    itself with a multiplicative gate, V2 keeps V0's additive-sum structure
    (`delta = B(phi(u + c))`) unchanged and instead restructures *how `c` is
    produced*: rather than `c_proj`'s free linear map `C_i(e_a): R^arg_dim ->
    R^rank` (any point in `R^rank` reachable for some `e_a`), `c` here is
    constrained to the convex hull of `num_basis` learned rank-dimensional
    vectors (`basis`, a plain `nn.Parameter[num_basis, rank]`, shared across
    every argument value) -- the argument only ever selects a soft mixture
    (`mix = softmax(mix_proj(e_a))`) over this small, shared dictionary. If
    F3's bottleneck is that a free linear conditioning map has *too much*
    capacity to be dominated by the content term during training (rather
    than too little -- D-005's already-tested and rejected capacity
    hypothesis, `docs/DECISIONS.md` ADR-0034), constraining the argument's
    reachable effect to a small, shared basis is a structurally different
    hypothesis worth testing alongside V1's multiplicative-gating one.

    `num_basis` defaults to `DEFAULT_NUM_BASIS_VECTORS` (8) -- small enough
    that `mix_proj` (`Linear(arg_dim, num_basis)`) plus `basis`
    (`num_basis * rank` parameters) together stay smaller than `c_proj`'s own
    `arg_dim * rank` at this experiment's `arg_dim=16, rank=8` baseline,
    consistent with `AGENTS.md`'s "do not introduce ... unnecessarily general
    function signatures."
    """

    def __init__(
        self,
        primitive_id: int,
        config: PrimitiveConfig,
        argument_encoder: ArgumentEncoder,
        argument_name: str,
        *,
        num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
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
        if num_basis < 1:
            raise ValueError(f"num_basis must be >= 1, got {num_basis}")
        self.argument_name = argument_name
        self.argument_encoder = argument_encoder
        self.num_basis = num_basis
        self.mix_proj = nn.Linear(argument_encoder.arg_dim, num_basis, bias=True)
        nn.init.normal_(self.mix_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mix_proj.bias)
        self.basis = nn.Parameter(torch.randn(num_basis, config.rank) * 0.02)
        self.activation = nn.GELU()

    def forward(  # type: ignore[override]
        self,
        h: torch.Tensor,
        *,
        argument_values: Sequence[Any],
        gate: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """See class docstring for the basis-modulation formula. Same
        `argument_values`/broadcast/error-handling conventions as
        `ConditionedPrimitive.forward`.

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
        mix = torch.softmax(self.mix_proj(e_a), dim=-1)
        c = mix @ self.basis
        u = self.a_proj(h)
        extra_dims = u.dim() - c.dim()
        for _ in range(extra_dims):
            c = c.unsqueeze(1)
        delta = self.b_proj(self.activation(u + c))
        return h + gate * delta


class ConditioningVariant(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """The conditioning-formula variants A1-R005D-006 compares, sharing
    everything else (`a_proj`/`b_proj`, `argument_encoder`, Stable Core,
    training/eval data) so the formula itself is the only controlled
    variable (module docstring's "Update" section)."""

    ADDITIVE = "additive"
    FILM = "film"
    BASIS = "basis"


def _single_required_argument_name(operation: str, *, caller: str) -> str:
    """Shared by `build_conditioned_primitive` and `build_argument_
    conditioned_primitive`: `operation`'s one declared hidden argument name,
    or a `ValueError` naming `caller` if `operation` does not require
    exactly one (every currently registered parameterized operation does;
    this guards against a future operation with more than one hidden
    argument, which this module does not yet support)."""
    required = get_operation(operation).required_argument_names
    if len(required) != 1:
        raise ValueError(
            f"{caller} only supports operations with exactly one required argument; "
            f"{operation!r} requires {sorted(required)}"
        )
    (argument_name,) = tuple(required)
    return argument_name


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
    argument_name = _single_required_argument_name(operation, caller="build_conditioned_primitive")
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


def build_argument_conditioned_primitive(
    primitive_id: int,
    operation: str,
    config: PrimitiveConfig,
    variant: ConditioningVariant = ConditioningVariant.ADDITIVE,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    arg_dim: int = DEFAULT_ARG_DIM,
    num_basis: int = DEFAULT_NUM_BASIS_VECTORS,
    status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
    created_at_task: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ArgumentConditionedPrimitive:
    """`build_conditioned_primitive`'s generalization across conditioning
    formulas (Task A1-R005D-006, module docstring's "Update" section):
    dispatches on `variant` but otherwise resolves the operation's own
    argument name/encoder identically for all three, so V0/V1/V2 differ only
    in which `ArgumentConditionedPrimitive` subclass gets constructed.
    `variant=ConditioningVariant.ADDITIVE` (the default) delegates straight
    to `build_conditioned_primitive`, so every pre-A1-R005D-006 caller's
    behavior is reproduced exactly.

    Raises: same as `build_conditioned_primitive` (`KeyError` for an
        unregistered operation or one with no typed encoder; `ValueError`
        for an operation that does not require exactly one argument, or an
        unrecognized `variant`).
    """
    if variant == ConditioningVariant.ADDITIVE:
        return build_conditioned_primitive(
            primitive_id,
            operation,
            config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
            status=status,
            created_at_task=created_at_task,
            metadata=metadata,
        )
    argument_name = _single_required_argument_name(
        operation, caller="build_argument_conditioned_primitive"
    )
    encoder = default_argument_encoder(
        operation, vocab_size=vocab_size, max_sequence_length=max_sequence_length, arg_dim=arg_dim
    )
    resolved_metadata = metadata if metadata is not None else {"operation": operation}
    if variant == ConditioningVariant.FILM:
        return FiLMConditionedPrimitive(
            primitive_id,
            config,
            encoder,
            argument_name,
            status=status,
            created_at_task=created_at_task,
            metadata=resolved_metadata,
        )
    if variant == ConditioningVariant.BASIS:
        return BasisModulatedConditionedPrimitive(
            primitive_id,
            config,
            encoder,
            argument_name,
            num_basis=num_basis,
            status=status,
            created_at_task=created_at_task,
            metadata=resolved_metadata,
        )
    raise ValueError(f"unknown ConditioningVariant: {variant!r}")
