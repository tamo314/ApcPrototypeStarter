"""Tests for the argument encoder integrity audit (Phase A.1 Post-Correction
Task A1-R005D-002).

Covers the task's own acceptance criteria: structural collisions are
detected and documented (the `[1, 3]` vs `[3, 1]` SELECT example plus
repeated-index permutations), distinct same-length SELECT index sets are not
forced to collide, `SHIFT`/`COUNT`/`BIND` are confirmed injective on their
real legal domains under Phase A.1's defaults, out-of-domain wraparound is
confirmed reachable (intentional aliasing, `docs/DECISIONS.md` ADR-0028), a
deliberately undersized bucket table *is* caught by the audit (regression
safety net), and the STOP GATE guard raises for the (still directly
constructible, but no longer default) mean-pooled `IndexSetArgumentEncoder`.

Update (Task A1-R005D-003): `default_argument_encoder("SELECT", ...)` now
returns `apc.primitives.conditioning.OrderPreservingIndexSetArgumentEncoder`
(`docs/DECISIONS.md` ADR-0032), which passes the STOP GATE guard -- covered
by the tests at the bottom of this file re-auditing/re-gating the current
default rather than the superseded mean-pooled one.
"""

from __future__ import annotations

import pytest
import torch

from apc.primitives.argument_encoder_audit import (
    NonInjectiveSelectEncoderError,
    audit_int_bucket_encoder,
    audit_phase_a1_default_argument_encoders,
    audit_select_order_sensitivity,
    require_order_preserving_select_encoder,
)
from apc.primitives.conditioning import (
    IndexSetArgumentEncoder,
    IntBucketArgumentEncoder,
    OrderPreservingIndexSetArgumentEncoder,
    default_argument_encoder,
)

# --- audit_int_bucket_encoder --------------------------------------------------


def test_int_bucket_audit_rejects_domain_with_fewer_than_two_values() -> None:
    encoder = IntBucketArgumentEncoder(num_buckets=16, arg_dim=8)
    with pytest.raises(ValueError, match="at least two distinct values"):
        audit_int_bucket_encoder(encoder, [3], operation="SHIFT", argument_name="amount")


def test_int_bucket_audit_finds_no_collision_when_domain_fits_buckets() -> None:
    torch.manual_seed(0)
    encoder = IntBucketArgumentEncoder(num_buckets=32, arg_dim=8)
    audit = audit_int_bucket_encoder(
        encoder, range(10), operation="SHIFT", argument_name="amount"
    )
    assert audit.injective_on_legal_domain
    assert audit.colliding_pairs == ()


def test_int_bucket_audit_out_of_domain_probe_demonstrates_intentional_wraparound() -> None:
    torch.manual_seed(0)
    encoder = IntBucketArgumentEncoder(num_buckets=32, arg_dim=8)
    audit = audit_int_bucket_encoder(
        encoder, range(10), operation="SHIFT", argument_name="amount"
    )
    assert audit.out_of_domain_probe is not None
    assert audit.out_of_domain_probe.value_a == 9 + 32  # first out-of-domain probe value
    assert audit.out_of_domain_probe.value_b == 9  # wraps back to the largest legal value
    assert audit.out_of_domain_probe.max_absolute_difference == 0.0


def test_int_bucket_audit_catches_a_deliberately_undersized_bucket_table() -> None:
    """Regression safety net: if `num_buckets` were ever misconfigured below
    the legal domain size, the audit must catch it, not silently pass."""
    torch.manual_seed(0)
    encoder = IntBucketArgumentEncoder(num_buckets=5, arg_dim=8)
    audit = audit_int_bucket_encoder(
        encoder, range(10), operation="SHIFT", argument_name="amount"
    )
    assert not audit.injective_on_legal_domain
    colliding_values = {(pair.value_a, pair.value_b) for pair in audit.colliding_pairs}
    assert (0, 5) in colliding_values  # 0 % 5 == 5 % 5


# --- audit_select_order_sensitivity --------------------------------------------


def test_select_order_audit_reversed_pair_collides() -> None:
    torch.manual_seed(0)
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    audit = audit_select_order_sensitivity(encoder, distinct_pairs=())
    assert audit.order_sensitive_collision_present
    collided = {(pair.value_a, pair.value_b) for pair in audit.permutation_collisions}
    assert ((1, 3), (3, 1)) in collided


def test_select_order_audit_repeated_index_permutations_collide() -> None:
    torch.manual_seed(0)
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    audit = audit_select_order_sensitivity(encoder, distinct_pairs=())
    collided = {(pair.value_a, pair.value_b) for pair in audit.permutation_collisions}
    assert ((1, 1, 3), (1, 3, 1)) in collided
    assert ((1, 1, 3), (3, 1, 1)) in collided


def test_select_order_audit_every_default_permutation_pair_collides() -> None:
    """Weight-independence: every declared permutation pair collides
    regardless of the encoder's random initialization -- checked across
    several independent seeds."""
    for seed in (0, 1, 2, 3, 4):
        torch.manual_seed(seed)
        encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
        audit = audit_select_order_sensitivity(encoder, distinct_pairs=())
        assert len(audit.permutation_collisions) == len(audit.permutation_pairs_checked), (
            f"seed {seed}: expected every permutation pair to collide"
        )


def test_select_order_audit_distinct_index_sets_do_not_collide_at_default_init() -> None:
    torch.manual_seed(0)
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    audit = audit_select_order_sensitivity(encoder, permutation_pairs=())
    assert audit.distinct_pair_collisions == ()


def test_select_order_audit_empty_pair_sequences_are_handled() -> None:
    torch.manual_seed(0)
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    audit = audit_select_order_sensitivity(encoder, permutation_pairs=(), distinct_pairs=())
    assert audit.permutation_collisions == ()
    assert audit.distinct_pair_collisions == ()
    assert not audit.order_sensitive_collision_present


# --- require_order_preserving_select_encoder (STOP GATE guard) ----------------


def test_guard_raises_for_the_current_default_select_encoder() -> None:
    torch.manual_seed(0)
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    with pytest.raises(NonInjectiveSelectEncoderError, match=r"\[1, 3\].*\[3, 1\]"):
        require_order_preserving_select_encoder(encoder)


def test_guard_passes_for_a_hypothetical_order_preserving_encoder() -> None:
    """Behavior-based, not `isinstance`-based: an encoder that actually
    distinguishes `[1, 3]` from `[3, 1]` must not raise, regardless of its
    class."""

    class _OrderPreservingStub(torch.nn.Module):
        arg_dim = 4

        def forward(self, values):  # noqa: ANN001 - test stub matches ArgumentEncoder shape
            # A trivial order-sensitive encoding: sum of index * (position + 1).
            rows = []
            for indices in values:
                weighted = sum(
                    (position + 1) * index for position, index in enumerate(indices)
                )
                rows.append(torch.tensor([float(weighted)] * self.arg_dim))
            return torch.stack(rows, dim=0)

    require_order_preserving_select_encoder(_OrderPreservingStub())  # must not raise


# --- audit_phase_a1_default_argument_encoders (consolidated report) -----------


def test_phase_a1_audit_shift_count_bind_are_injective_on_their_real_domains() -> None:
    report = audit_phase_a1_default_argument_encoders(select_seeds=(0,))
    for operation in ("SHIFT", "COUNT", "BIND"):
        audit = report.int_bucket_audits[operation]
        assert audit.injective_on_legal_domain, f"{operation}: unexpected legal-domain collision"
        assert audit.out_of_domain_probe is not None
        assert audit.out_of_domain_probe.max_absolute_difference == 0.0


def test_phase_a1_audit_select_no_longer_collides_after_d003() -> None:
    """Task A1-R005D-003 replaced `default_argument_encoder("SELECT", ...)`'s
    mean-pooled `IndexSetArgumentEncoder` (ADR-0031's finding, which this
    test originally asserted as `select_collision_present_in_every_seed`)
    with the order-preserving encoder audited here as no longer colliding --
    the intended, documented consequence of D-003 landing, not a regression."""
    report = audit_phase_a1_default_argument_encoders(select_seeds=(0, 1, 2))
    assert not report.select_collision_present_in_every_seed
    for audit in report.select_audits:
        assert audit.permutation_collisions == ()
    assert not report.structural_collisions_found


def test_phase_a1_default_sequence_length_range_matches_the_generators_own_default() -> None:
    """Cross-check the audit module's hardcoded domain assumption against
    `apc.environments.generator.TaskGenerator`'s actual default, so the two
    cannot silently drift apart."""
    import inspect

    from apc.environments.generator import TaskGenerator
    from apc.primitives.argument_encoder_audit import PHASE_A1_DEFAULT_SEQUENCE_LENGTH_RANGE

    generator_default = inspect.signature(TaskGenerator.__init__).parameters[
        "sequence_length_range"
    ].default
    assert PHASE_A1_DEFAULT_SEQUENCE_LENGTH_RANGE == generator_default


def test_phase_a1_audit_domains_match_operations_own_sample_params_ranges() -> None:
    from apc.environments.vocab import DEFAULT_VOCAB_SIZE

    report = audit_phase_a1_default_argument_encoders(select_seeds=(0,))
    shift_domain = report.int_bucket_audits["SHIFT"].legal_domain
    assert shift_domain == tuple(range(10))
    count_domain = report.int_bucket_audits["COUNT"].legal_domain
    assert count_domain == tuple(range(DEFAULT_VOCAB_SIZE))


def test_default_argument_encoder_select_now_produces_an_order_preserving_encoder() -> None:
    """A1-R005D-003 landed: `default_argument_encoder("SELECT", ...)` now
    returns `OrderPreservingIndexSetArgumentEncoder`, not the mean-pooled
    `IndexSetArgumentEncoder` this audit found non-injective (ADR-0031).
    `IndexSetArgumentEncoder` itself is unchanged and still directly
    constructible (see the tests above using it), just no longer reachable
    through the default factory."""
    encoder = default_argument_encoder("SELECT")
    assert isinstance(encoder, OrderPreservingIndexSetArgumentEncoder)
    assert not isinstance(encoder, IndexSetArgumentEncoder)


def test_default_argument_encoder_select_now_passes_the_stop_gate_guard() -> None:
    """The STOP GATE `require_order_preserving_select_encoder` introduced by
    A1-R005D-002 to refuse a known-non-injective SELECT encoder must not
    raise for the production default any more."""
    require_order_preserving_select_encoder(default_argument_encoder("SELECT"))
