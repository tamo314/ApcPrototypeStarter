"""Argument encoder integrity audit (Phase A.1 Post-Correction Task
A1-R005D-002, `docs/CODEX_TASKS_A1_R005_RETRY.md`).

`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`'s "SELECT rule" states that
`SELECT.indices` must preserve order and multiplicity, and that "a
commutative pooling representation such as mean/sum is invalid for an
ordered index sequence unless an explicit experiment proves otherwise."
`apc.primitives.conditioning.IndexSetArgumentEncoder` -- the only `SELECT`
argument encoder `apc.primitives.conditioning.default_argument_encoder`
currently registers -- is exactly that: `.mean(dim=0)` over the selected
positions' embeddings. Mean is commutative, so this is not merely an
empirical risk to test for; it is a *structural* (weight-independent)
guarantee that any permutation of the same index multiset -- e.g. the task's
own `[1, 3]` vs `[3, 1]` example -- encodes identically, for any embedding
table, trained or not. `tests/test_primitives_conditioning.py::
test_index_set_encoder_is_order_invariant` already demonstrates this at the
unit level (written as expected behavior when A1-R004 landed, before the
A1-R005 retry's stricter order-preservation rule existed).

This module turns that structural fact into an auditable, reusable finding
(`docs/DECISIONS.md` ADR-0031) plus an enforceable guard
(`require_order_preserving_select_encoder`) that a future SELECT-training
entry point (A1-R005D-003 onward) can call to refuse to proceed with a
known-non-injective encoder -- the task's own STOP GATE ("do not retrain
SELECT with known non-injective encoding") implemented as code, not only as
prose. The guard is deliberately *not* wired into
`apc.primitives.conditioning.build_conditioned_primitive`/
`default_argument_encoder` themselves: those remain exactly as A1-R004 left
them (ADR-0028), still constructible and already covered by
`tests/test_primitives_conditioning.py`'s accepted acceptance tests, since
this task's scope is audit-and-gate-future-training, not replacing the
encoder (`apc.primitives.conditioning.IndexSetArgumentEncoder` staying
unchanged is A1-R005D-003's own "skip only if D-002 proves current
representation sufficient" decision point, which this audit resolves as
"not sufficient").

### Update (Task A1-R005D-003)

`default_argument_encoder("SELECT", ...)` no longer returns
`IndexSetArgumentEncoder`; it returns `apc.primitives.conditioning.
OrderPreservingIndexSetArgumentEncoder` (`docs/DECISIONS.md` ADR-0032),
which passes `require_order_preserving_select_encoder`. The narrative above
describes the state this module found and gated at the time D-002 landed;
`audit_phase_a1_default_argument_encoders` itself now audits the *current*
default encoder (order-preserving), so re-running it reports no SELECT
collision -- `IndexSetArgumentEncoder` remains available, unchanged, as a
standalone class for direct construction/comparison, just not reachable
through the default factory any more.

The second half of this module audits the two `IntBucketArgumentEncoder`
instances `default_argument_encoder` mints for `SHIFT`/`COUNT`/`BIND`
against each operation's real legal argument domain (`apc.environments.
operations.ShiftOp`/`CountOp`/`BindOp.sample_params`, bounded by
`apc.environments.generator.TaskGenerator`'s own default
`sequence_length_range=(6, 10)` and `apc.environments.vocab.
DEFAULT_VOCAB_SIZE=10` -- the same domain `apc.environments.task_spec.
default_argument_value_span` already treats as authoritative for a related
purpose, Task A1-C003). `IntBucketArgumentEncoder.forward` deliberately
wraps any out-of-declared-range integer via `value % num_buckets`
(`docs/DECISIONS.md` ADR-0028's "invalid vs. missing arguments" reasoning:
A1-R005's own "correct family + wrong argument" causal control needs to be
able to construct such values, not have them rejected) -- that wraparound
is intentional and this audit does not flag it as a bug. What this audit
does check is narrower and more specific: whether two *distinct legal*
values (both inside the operation's own real domain) are ever forced into
the same bucket, which would be an accidental information loss, not an
intentional one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.primitives.conditioning import (
    DEFAULT_ARG_DIM,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    ArgumentEncoder,
    IntBucketArgumentEncoder,
    OrderPreservingIndexSetArgumentEncoder,
    default_argument_encoder,
)

__all__ = [
    "NonInjectiveSelectEncoderError",
    "CollisionPair",
    "IntBucketDomainAudit",
    "SelectOrderAudit",
    "ArgumentEncoderAuditReport",
    "audit_int_bucket_encoder",
    "audit_select_order_sensitivity",
    "require_order_preserving_select_encoder",
    "audit_phase_a1_default_argument_encoders",
]

# `apc.environments.generator.TaskGenerator`'s own default; duplicated here
# (rather than importing the generator) so this module stays a pure
# encoder/domain audit with no dependency on the generator/model stack --
# only `apc.environments.vocab`/`apc.primitives.conditioning`, matching
# A1-R005D-001's own "no retraining, minimal surface" precedent.
PHASE_A1_DEFAULT_SEQUENCE_LENGTH_RANGE: tuple[int, int] = (6, 10)


class NonInjectiveSelectEncoderError(RuntimeError):
    """Raised by `require_order_preserving_select_encoder` for an encoder
    that maps `SELECT.indices` `[1, 3]` and `[3, 1]` to (numerically)
    identical embeddings -- the A1-R005D-002 STOP GATE condition."""


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@dataclass(frozen=True)
class CollisionPair:
    """Two distinct raw argument values whose encoder output was
    (numerically, within `atol`) indistinguishable."""

    value_a: Any
    value_b: Any
    max_absolute_difference: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_a": self.value_a,
            "value_b": self.value_b,
            "max_absolute_difference": self.max_absolute_difference,
        }


@dataclass(frozen=True)
class IntBucketDomainAudit:
    """Injectivity audit of one `IntBucketArgumentEncoder` over one
    operation's real legal argument domain, plus one deliberately
    out-of-domain probe demonstrating the intentional wraparound."""

    operation: str
    argument_name: str
    num_buckets: int
    legal_domain: tuple[int, ...]
    colliding_pairs: tuple[CollisionPair, ...]
    out_of_domain_probe: CollisionPair | None

    @property
    def injective_on_legal_domain(self) -> bool:
        return len(self.colliding_pairs) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "argument_name": self.argument_name,
            "num_buckets": self.num_buckets,
            "legal_domain": list(self.legal_domain),
            "injective_on_legal_domain": self.injective_on_legal_domain,
            "colliding_pairs": [pair.to_dict() for pair in self.colliding_pairs],
            "out_of_domain_probe": (
                self.out_of_domain_probe.to_dict() if self.out_of_domain_probe is not None else None
            ),
        }


@dataclass(frozen=True)
class SelectOrderAudit:
    """Injectivity audit of one `IndexSetArgumentEncoder` over ordered
    index-set pairs (permutations of the same multiset -- including
    repeated indices -- and distinct same-length index sets)."""

    permutation_pairs_checked: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    permutation_collisions: tuple[CollisionPair, ...]
    distinct_pairs_checked: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    distinct_pair_collisions: tuple[CollisionPair, ...]

    @property
    def order_sensitive_collision_present(self) -> bool:
        return len(self.permutation_collisions) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation_pairs_checked": [
                [list(a), list(b)] for a, b in self.permutation_pairs_checked
            ],
            "permutation_collisions": [pair.to_dict() for pair in self.permutation_collisions],
            "order_sensitive_collision_present": self.order_sensitive_collision_present,
            "distinct_pairs_checked": [[list(a), list(b)] for a, b in self.distinct_pairs_checked],
            "distinct_pair_collisions": [pair.to_dict() for pair in self.distinct_pair_collisions],
        }


@dataclass(frozen=True)
class ArgumentEncoderAuditReport:
    """Consolidated audit over all four `default_argument_encoder`
    instances (`SHIFT`/`SELECT`/`COUNT`/`BIND`)."""

    select_audits: tuple[SelectOrderAudit, ...]
    select_seeds_checked: tuple[int, ...]
    int_bucket_audits: dict[str, IntBucketDomainAudit]

    @property
    def select_collision_present_in_every_seed(self) -> bool:
        return len(self.select_audits) > 0 and all(
            audit.order_sensitive_collision_present for audit in self.select_audits
        )

    @property
    def structural_collisions_found(self) -> bool:
        return self.select_collision_present_in_every_seed or any(
            not audit.injective_on_legal_domain for audit in self.int_bucket_audits.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "select_seeds_checked": list(self.select_seeds_checked),
            "select_audits": [audit.to_dict() for audit in self.select_audits],
            "select_collision_present_in_every_seed": self.select_collision_present_in_every_seed,
            "int_bucket_audits": {
                operation: audit.to_dict() for operation, audit in self.int_bucket_audits.items()
            },
            "structural_collisions_found": self.structural_collisions_found,
        }


def audit_int_bucket_encoder(
    encoder: IntBucketArgumentEncoder,
    legal_domain: Sequence[int],
    *,
    operation: str,
    argument_name: str,
    atol: float = 1e-6,
) -> IntBucketDomainAudit:
    """Check whether `encoder` maps any two *distinct* values in
    `legal_domain` to (numerically) identical embeddings, plus probe one
    deliberately out-of-domain value (`max(legal_domain) + encoder.
    num_buckets`, which wraps back to `max(legal_domain)` exactly when the
    domain fits inside `num_buckets`) to show the intentional wraparound
    path is reachable.

    Raises:
        ValueError: `legal_domain` has fewer than two distinct values.
    """
    values = sorted({int(v) for v in legal_domain})
    if len(values) < 2:
        raise ValueError("legal_domain must contain at least two distinct values to audit")

    with torch.no_grad():
        encoded = encoder(values)

    colliding = []
    for i, j in combinations(range(len(values)), 2):
        diff = _max_abs_diff(encoded[i], encoded[j])
        if diff <= atol:
            colliding.append(CollisionPair(values[i], values[j], diff))

    probe_value = values[-1] + encoder.num_buckets
    probe_bucket = probe_value % encoder.num_buckets
    out_of_domain_probe = None
    if probe_bucket in values:
        with torch.no_grad():
            probe_encoded = encoder([probe_value, probe_bucket])
        out_of_domain_probe = CollisionPair(
            probe_value, probe_bucket, _max_abs_diff(probe_encoded[0], probe_encoded[1])
        )

    return IntBucketDomainAudit(
        operation=operation,
        argument_name=argument_name,
        num_buckets=encoder.num_buckets,
        legal_domain=tuple(values),
        colliding_pairs=tuple(colliding),
        out_of_domain_probe=out_of_domain_probe,
    )


# Permutations of the same index multiset -- structurally guaranteed to
# collide under mean pooling regardless of embedding weights. Includes the
# task-mandated `[1, 3]` vs `[3, 1]` example, a longer order reversal, and
# two repeated-index permutations ("SELECT: ... repeated indices if legal"
# -- `apc.environments.primitive_call.PrimitiveCall.__post_init__` places
# no uniqueness/order constraint on `indices`, and `apc.environments.
# operations.SelectOp.apply` executes a repeated index without error, so a
# repeated-index `PrimitiveCall` is legal even though `SelectOp.
# sample_params` itself never produces one).
_DEFAULT_PERMUTATION_PAIRS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((1, 3), (3, 1)),
    ((1, 3, 5), (5, 3, 1)),
    ((0, 2, 4, 6), (6, 4, 2, 0)),
    ((1, 1, 3), (1, 3, 1)),
    ((1, 1, 3), (3, 1, 1)),
)

# Distinct index sets of matching length (not permutations of one another)
# -- not structurally guaranteed to collide, checked empirically.
_DEFAULT_DISTINCT_PAIRS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((0, 1, 2), (3, 4, 5)),
    ((1, 3, 5), (0, 2, 4)),
    ((0, 1, 2, 3), (4, 5, 6, 7)),
)


def audit_select_order_sensitivity(
    encoder: ArgumentEncoder,
    *,
    permutation_pairs: Sequence[tuple[Sequence[int], Sequence[int]]] = _DEFAULT_PERMUTATION_PAIRS,
    distinct_pairs: Sequence[tuple[Sequence[int], Sequence[int]]] = _DEFAULT_DISTINCT_PAIRS,
    atol: float = 1e-6,
) -> SelectOrderAudit:
    """Run `encoder` (any `SELECT.indices`-shaped `ArgumentEncoder` --
    `IndexSetArgumentEncoder` or `apc.primitives.conditioning.
    OrderPreservingIndexSetArgumentEncoder`, Task A1-R005D-003) over
    `permutation_pairs` (each pair a reordering of the same index multiset)
    and `distinct_pairs` (each pair genuinely distinct index sets of the
    same length), reporting which pairs collided."""

    def _collisions(
        pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> tuple[CollisionPair, ...]:
        if not pairs:
            return ()
        left = [list(a) for a, _ in pairs]
        right = [list(b) for _, b in pairs]
        with torch.no_grad():
            left_encoded = encoder(left)
            right_encoded = encoder(right)
        found = []
        for idx, (a, b) in enumerate(pairs):
            diff = _max_abs_diff(left_encoded[idx], right_encoded[idx])
            if diff <= atol:
                found.append(CollisionPair(tuple(a), tuple(b), diff))
        return tuple(found)

    return SelectOrderAudit(
        permutation_pairs_checked=tuple((tuple(a), tuple(b)) for a, b in permutation_pairs),
        permutation_collisions=_collisions(permutation_pairs),
        distinct_pairs_checked=tuple((tuple(a), tuple(b)) for a, b in distinct_pairs),
        distinct_pair_collisions=_collisions(distinct_pairs),
    )


def require_order_preserving_select_encoder(
    encoder: ArgumentEncoder, *, atol: float = 1e-6
) -> None:
    """STOP GATE guard (A1-R005D-002): raise unless `encoder` distinguishes
    `SELECT.indices` `[1, 3]` from `[3, 1]`.

    A future SELECT-training entry point (A1-R005D-003 onward) should call
    this before training so a known-non-injective encoder -- today, any
    `apc.primitives.conditioning.IndexSetArgumentEncoder` -- cannot be
    silently reused. Behavior-based (actually running the encoder) rather
    than an `isinstance` check, so it also catches a future replacement
    encoder that happens to still be order-insensitive, and does not need
    updating if the replacement is given a new class name.

    Raises:
        NonInjectiveSelectEncoderError: `encoder` maps `[1, 3]` and `[3, 1]`
            to (numerically) identical embeddings.
    """
    with torch.no_grad():
        encoded = encoder([[1, 3], [3, 1]])
    if _max_abs_diff(encoded[0], encoded[1]) <= atol:
        raise NonInjectiveSelectEncoderError(
            f"{type(encoder).__name__} maps SELECT.indices [1, 3] and [3, 1] to "
            "(numerically) identical embeddings -- a structural order-sensitive "
            "collision (docs/DECISIONS.md ADR-0031, A1-R005D-002). Refusing to "
            "certify this encoder for training; A1-R005D-003 must supply an "
            "order-preserving replacement first."
        )


def audit_phase_a1_default_argument_encoders(
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = PHASE_A1_DEFAULT_SEQUENCE_LENGTH_RANGE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    arg_dim: int = DEFAULT_ARG_DIM,
    select_seeds: Sequence[int] = (0, 1, 2),
) -> ArgumentEncoderAuditReport:
    """Audit the four `apc.primitives.conditioning.default_argument_encoder`
    instances against Phase A.1's actual legal argument domains.

    Position domain (`SHIFT.amount`, `SELECT.indices` entries) is
    `range(sequence_length_range[1])`: `apc.environments.operations.
    ShiftOp.sample_params`/`SelectOp.sample_params` both draw from
    `range(len(sequence))`, and `apc.environments.generator.
    TaskGenerator`'s `_valid_lengths_for_chain` treats
    `sequence_length_range[1]` as an inclusive upper bound on `len(sequence)`
    -- the same upper bound `apc.environments.task_spec.
    default_argument_value_span` already uses for this domain. Vocabulary
    domain (`COUNT.target`, `BIND.query_key`) is `range(vocab_size)`
    (`CountOp`/`BindOp.sample_params`).

    `SELECT` is audited across `select_seeds` independently constructed
    encoders (`torch.manual_seed` before each), since `audit_select_order_
    sensitivity`'s collision checks are now (Task A1-R005D-003: `default_
    argument_encoder("SELECT", ...)` returns `apc.primitives.conditioning.
    OrderPreservingIndexSetArgumentEncoder`, not the mean-pooled
    `IndexSetArgumentEncoder` this module originally audited under ADR-0031)
    an empirical property of the encoder's random self-attention weights
    rather than a structural guarantee either way -- multiple seeds sample
    that variation instead of asserting from a single draw.
    """
    position_domain = tuple(range(sequence_length_range[1]))
    vocab_domain = tuple(range(vocab_size))

    int_bucket_specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "SHIFT": ("amount", position_domain),
        "COUNT": ("target", vocab_domain),
        "BIND": ("query_key", vocab_domain),
    }
    int_bucket_audits: dict[str, IntBucketDomainAudit] = {}
    for operation, (argument_name, domain) in int_bucket_specs.items():
        encoder = default_argument_encoder(
            operation,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
        )
        assert isinstance(encoder, IntBucketArgumentEncoder)
        int_bucket_audits[operation] = audit_int_bucket_encoder(
            encoder, domain, operation=operation, argument_name=argument_name
        )

    select_audits = []
    for seed in select_seeds:
        torch.manual_seed(seed)
        encoder = default_argument_encoder(
            "SELECT",
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            arg_dim=arg_dim,
        )
        assert isinstance(encoder, OrderPreservingIndexSetArgumentEncoder)
        select_audits.append(audit_select_order_sensitivity(encoder))

    return ArgumentEncoderAuditReport(
        select_audits=tuple(select_audits),
        select_seeds_checked=tuple(select_seeds),
        int_bucket_audits=int_bucket_audits,
    )
