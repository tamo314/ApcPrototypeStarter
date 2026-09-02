"""Tests for the argument conditioning module (Phase A.1 Post-Correction Task
A1-R004, milestone A1-RM4).

Covers the task's own acceptance criteria: changing an argument changes a
`ConditionedPrimitive`'s output, one family instance handles >=3 distinct
argument values, persistent family/bank count does not grow with the number
of distinct values exercised, and invalid (out-of-declared-range) vs. missing
arguments are both exercised -- the former handled gracefully (modulo
wraparound), the latter refused loudly (`TypeError`/`KeyError`).
"""

from __future__ import annotations

import pytest
import torch

from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import operation_id
from apc.primitives.bank import PrimitiveBank
from apc.primitives.conditioning import (
    ArgumentConditionedPrimitive,
    BasisModulatedConditionedPrimitive,
    ConditionedPrimitive,
    ConditioningVariant,
    FiLMConditionedPrimitive,
    IndexSetArgumentEncoder,
    IntBucketArgumentEncoder,
    OrderPreservingIndexSetArgumentEncoder,
    build_argument_conditioned_primitive,
    build_conditioned_primitive,
    default_argument_encoder,
)
from apc.primitives.primitive import PrimitiveConfig

PARAMETERIZED_OPERATIONS: dict[str, str] = {
    "SHIFT": "amount",
    "SELECT": "indices",
    "COUNT": "target",
    "BIND": "query_key",
}


def _perturbed(primitive: ConditionedPrimitive) -> ConditionedPrimitive:
    """`b_proj`/`c_proj` start at/near zero (see `Primitive.__init__`'s
    zero-init and `ConditionedPrimitive.__init__`'s small-std `c_proj`
    init), so a freshly constructed primitive's residual is near-identity
    regardless of argument. Perturb both to non-trivial values before
    asserting argument-dependent behavior, mirroring `tests/
    test_primitives_primitive.py`'s own established pattern for the base
    `Primitive`."""
    with torch.no_grad():
        primitive.b_proj.weight.add_(1.0)
        primitive.c_proj.weight.add_(1.0)
    return primitive


# --- IntBucketArgumentEncoder -------------------------------------------------


def test_int_bucket_encoder_different_values_give_different_embeddings() -> None:
    encoder = IntBucketArgumentEncoder(num_buckets=16, arg_dim=8)
    out = encoder([1, 2, 3])
    assert out.shape == (3, 8)
    assert not torch.equal(out[0], out[1])
    assert not torch.equal(out[1], out[2])


def test_int_bucket_encoder_wraps_out_of_range_values() -> None:
    encoder = IntBucketArgumentEncoder(num_buckets=5, arg_dim=8)
    out = encoder([2, 7, -3])  # 7 % 5 == 2, -3 % 5 == 2 (torch.remainder convention)
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[0], out[2])


def test_int_bucket_encoder_rejects_empty_batch() -> None:
    encoder = IntBucketArgumentEncoder(num_buckets=5, arg_dim=8)
    with pytest.raises(ValueError, match="at least one batch item"):
        encoder([])


@pytest.mark.parametrize("num_buckets", [0, -1])
def test_int_bucket_encoder_rejects_non_positive_num_buckets(num_buckets: int) -> None:
    with pytest.raises(ValueError, match="num_buckets"):
        IntBucketArgumentEncoder(num_buckets=num_buckets, arg_dim=8)


# --- IndexSetArgumentEncoder --------------------------------------------------


def test_index_set_encoder_different_index_sets_give_different_embeddings() -> None:
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[0, 1], [5, 6, 7]])
    assert out.shape == (2, 8)
    assert not torch.equal(out[0], out[1])


def test_index_set_encoder_is_order_invariant() -> None:
    encoder = IndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[1, 3, 5], [5, 3, 1]])
    # A mean over the same set of rows in a different summation order can
    # differ by float rounding, so compare approximately rather than
    # bit-exactly.
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_index_set_encoder_wraps_out_of_range_positions() -> None:
    encoder = IndexSetArgumentEncoder(max_positions=5, arg_dim=8)
    out = encoder([[2], [7]])  # 7 % 5 == 2
    assert torch.equal(out[0], out[1])


def test_index_set_encoder_rejects_empty_index_subset() -> None:
    encoder = IndexSetArgumentEncoder(max_positions=5, arg_dim=8)
    with pytest.raises(ValueError, match="non-empty index subset"):
        encoder([[]])


# --- OrderPreservingIndexSetArgumentEncoder (A1-R005D-003) --------------------


def test_order_preserving_encoder_distinguishes_the_task_example_permutation() -> None:
    torch.manual_seed(0)
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[1, 3], [3, 1]])
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_order_preserving_encoder_distinguishes_permutations_across_several_seeds() -> None:
    for seed in (0, 1, 2, 3, 4):
        torch.manual_seed(seed)
        encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
        out = encoder([[1, 3], [3, 1]])
        assert not torch.allclose(out[0], out[1], atol=1e-6), f"seed {seed}: [1,3]==[3,1]"


def test_order_preserving_encoder_distinguishes_repeated_index_permutations() -> None:
    torch.manual_seed(0)
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[1, 1, 3], [1, 3, 1], [3, 1, 1]])
    assert not torch.allclose(out[0], out[1], atol=1e-6)
    assert not torch.allclose(out[0], out[2], atol=1e-6)
    assert not torch.allclose(out[1], out[2], atol=1e-6)


def test_order_preserving_encoder_supports_variable_length_in_one_batch() -> None:
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[0, 1], [5, 6, 7], [2]])
    assert out.shape == (3, 8)


def test_order_preserving_encoder_padding_does_not_leak_into_shorter_rows() -> None:
    """A short row's output must match whether or not it is batched
    alongside a longer row -- the padding mask must actually exclude padded
    slots from attention/pooling, not merely zero them post hoc."""
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    alone = encoder([[0, 1]])
    with_padding = encoder([[0, 1], [5, 6, 7]])
    assert torch.allclose(alone[0], with_padding[0], atol=1e-6)


def test_order_preserving_encoder_is_deterministic_across_repeated_calls() -> None:
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    values = [[1, 3], [3, 1], [0, 2, 4]]
    first = encoder(values)
    second = encoder(values)
    assert torch.equal(first, second)


def test_order_preserving_encoder_is_deterministic_in_train_mode_too() -> None:
    """`dropout=0.0` should make the encoder layer deterministic regardless
    of `self.training`, unlike a default `nn.TransformerEncoderLayer`."""
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    encoder.train()
    values = [[1, 3, 5], [2, 4]]
    first = encoder(values)
    second = encoder(values)
    assert torch.equal(first, second)


def test_order_preserving_encoder_different_index_sets_give_different_embeddings() -> None:
    torch.manual_seed(0)
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=16, arg_dim=8)
    out = encoder([[0, 1], [5, 6, 7]])
    assert out.shape == (2, 8)
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_order_preserving_encoder_wraps_out_of_range_positions() -> None:
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=5, arg_dim=8)
    out = encoder([[2], [7]])  # 7 % 5 == 2
    assert torch.equal(out[0], out[1])


def test_order_preserving_encoder_rejects_empty_batch() -> None:
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=5, arg_dim=8)
    with pytest.raises(ValueError, match="at least one batch item"):
        encoder([])


def test_order_preserving_encoder_rejects_empty_index_subset() -> None:
    encoder = OrderPreservingIndexSetArgumentEncoder(max_positions=5, arg_dim=8)
    with pytest.raises(ValueError, match="non-empty index subset"):
        encoder([[0, 1], []])


@pytest.mark.parametrize("max_positions", [0, -1])
def test_order_preserving_encoder_rejects_non_positive_max_positions(max_positions: int) -> None:
    with pytest.raises(ValueError, match="max_positions"):
        OrderPreservingIndexSetArgumentEncoder(max_positions=max_positions, arg_dim=8)


def test_order_preserving_encoder_rejects_non_positive_arg_dim() -> None:
    with pytest.raises(ValueError, match="arg_dim"):
        OrderPreservingIndexSetArgumentEncoder(max_positions=8, arg_dim=0)


# --- default_argument_encoder -------------------------------------------------


def test_default_argument_encoder_shift_is_int_bucket_sized_to_sequence_length() -> None:
    encoder = default_argument_encoder("SHIFT", max_sequence_length=20, arg_dim=4)
    assert isinstance(encoder, IntBucketArgumentEncoder)
    assert encoder.num_buckets == 20
    assert encoder.arg_dim == 4


def test_default_argument_encoder_select_is_order_preserving_sized_to_sequence_length() -> None:
    encoder = default_argument_encoder("SELECT", max_sequence_length=20, arg_dim=4)
    assert isinstance(encoder, OrderPreservingIndexSetArgumentEncoder)
    assert encoder.max_positions == 20


@pytest.mark.parametrize("operation", ["COUNT", "BIND"])
def test_default_argument_encoder_count_and_bind_are_int_bucket_sized_to_vocab(
    operation: str,
) -> None:
    encoder = default_argument_encoder(operation, vocab_size=12, arg_dim=4)
    assert isinstance(encoder, IntBucketArgumentEncoder)
    assert encoder.num_buckets == 12


def test_default_argument_encoder_count_and_bind_are_independent_instances() -> None:
    count_encoder = default_argument_encoder("COUNT")
    bind_encoder = default_argument_encoder("BIND")
    assert count_encoder is not bind_encoder
    assert count_encoder.embedding is not bind_encoder.embedding  # type: ignore[attr-defined]


def test_default_argument_encoder_rejects_unknown_operation() -> None:
    with pytest.raises(KeyError):
        default_argument_encoder("COPY")


# --- ConditionedPrimitive: core acceptance criteria ---------------------------


def _make_conditioned(operation: str = "SHIFT", **overrides: object) -> ConditionedPrimitive:
    defaults: dict[str, object] = {
        "primitive_id": operation_id(operation),
        "operation": operation,
        "config": PrimitiveConfig(d_model=8, rank=4),
        "max_sequence_length": 16,
        "vocab_size": 10,
        "arg_dim": 6,
    }
    defaults.update(overrides)
    return build_conditioned_primitive(**defaults)  # type: ignore[arg-type]


def test_forward_is_near_identity_at_init_regardless_of_argument() -> None:
    primitive = _make_conditioned("SHIFT")
    h = torch.randn(3, 8)
    out_a = primitive(h, argument_values=[0, 0, 0])
    out_b = primitive(h, argument_values=[5, 5, 5])
    assert torch.equal(out_a, h)
    assert torch.equal(out_b, h)


def test_changing_argument_changes_output() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(2, 8)
    out_amount_1 = primitive(h, argument_values=[1, 1])
    out_amount_2 = primitive(h, argument_values=[2, 2])
    assert not torch.equal(out_amount_1, h)
    assert not torch.equal(out_amount_1, out_amount_2)


@pytest.mark.parametrize("operation", list(PARAMETERIZED_OPERATIONS))
def test_same_family_handles_at_least_three_distinct_values(operation: str) -> None:
    primitive = _perturbed(_make_conditioned(operation))
    h = torch.randn(1, 8)
    argument_pool: dict[str, list[object]] = {
        "SHIFT": [0, 3, 7],
        "SELECT": [[0, 1], [2, 3, 4], [5]],
        "COUNT": [0, 4, 9],
        "BIND": [1, 2, 8],
    }
    outputs = [
        primitive(h, argument_values=[value]) for value in argument_pool[operation]
    ]
    n_params_before = primitive.num_parameters()
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.equal(outputs[i], outputs[j])
    # Driving the same family through several values allocates no new
    # parameters -- "persistent family count unchanged" at the primitive
    # level (the bank-level version is tested separately below).
    assert primitive.num_parameters() == n_params_before


def test_forward_requires_argument_values_keyword() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(2, 8)
    with pytest.raises(TypeError):
        primitive(h)  # type: ignore[call-arg]


def test_forward_rejects_mismatched_batch_size() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(3, 8)
    with pytest.raises(ValueError, match="batch"):
        primitive(h, argument_values=[1, 2])


def test_disabled_primitive_is_identity_even_with_argument() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    primitive.enabled = False
    h = torch.randn(2, 8)
    out = primitive(h, argument_values=[3, 3])
    assert torch.equal(out, h)


def test_gate_scales_the_residual() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(2, 8)
    out_full = primitive(h, argument_values=[3, 3], gate=1.0)
    out_zero = primitive(h, argument_values=[3, 3], gate=0.0)
    assert torch.equal(out_zero, h)
    assert not torch.equal(out_full, h)


def test_forward_broadcasts_argument_across_sequence_positions() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(2, 5, 8)
    out = primitive(h, argument_values=[1, 2])
    assert out.shape == h.shape
    assert not torch.equal(out, h)


# --- forward_from_calls: the PrimitiveCall.arguments link ---------------------


def test_forward_from_calls_matches_forward_with_the_same_raw_value() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(2, 8)
    calls = [PrimitiveCall(operation="SHIFT", arguments={"amount": 4})] * 2
    via_calls = primitive.forward_from_calls(h, calls)
    via_raw = primitive(h, argument_values=[4, 4])
    assert torch.equal(via_calls, via_raw)


def test_forward_from_calls_rejects_mismatched_operation() -> None:
    primitive = _perturbed(_make_conditioned("SHIFT"))
    h = torch.randn(1, 8)
    calls = [PrimitiveCall(operation="COUNT", arguments={"target": 3})]
    with pytest.raises(KeyError):
        primitive.forward_from_calls(h, calls)


# --- build_conditioned_primitive ----------------------------------------------


@pytest.mark.parametrize(("operation", "argument_name"), PARAMETERIZED_OPERATIONS.items())
def test_build_conditioned_primitive_uses_the_operations_own_argument_name(
    operation: str, argument_name: str
) -> None:
    primitive = _make_conditioned(operation)
    assert primitive.argument_name == argument_name
    assert primitive.primitive_id == operation_id(operation)


def test_build_conditioned_primitive_rejects_zero_argument_operations() -> None:
    with pytest.raises(ValueError, match="exactly one required argument"):
        build_conditioned_primitive(0, "COPY", PrimitiveConfig(d_model=8, rank=4))


def test_build_conditioned_primitive_rejects_unregistered_operation() -> None:
    with pytest.raises(KeyError):
        build_conditioned_primitive(0, "NOT_AN_OPERATION", PrimitiveConfig(d_model=8, rank=4))


# --- Persistent family/bank-count acceptance -----------------------------------


def test_bank_size_does_not_grow_with_distinct_argument_values() -> None:
    bank = PrimitiveBank()
    primitive = _perturbed(_make_conditioned("SHIFT"))
    bank.add_primitive(primitive)
    assert len(bank) == 1

    h = torch.randn(1, 8)
    for amount in range(10):
        bank.get(operation_id("SHIFT"))(h, argument_values=[amount])
        assert len(bank) == 1  # no per-value registration ever happens

    assert bank.get(operation_id("SHIFT")) is primitive


def test_bank_holds_exactly_one_primitive_per_parameterized_family() -> None:
    bank = PrimitiveBank()
    for operation in PARAMETERIZED_OPERATIONS:
        bank.add_primitive(_make_conditioned(operation))
    assert len(bank) == len(PARAMETERIZED_OPERATIONS)
    for operation in PARAMETERIZED_OPERATIONS:
        retrieved = bank.get(operation_id(operation))
        assert isinstance(retrieved, ConditionedPrimitive)
        assert retrieved.metadata["operation"] == operation


def test_conditioned_primitive_num_parameters_includes_encoder_and_c_proj() -> None:
    d_model, rank, arg_dim, num_buckets = 8, 4, 6, 16
    primitive = build_conditioned_primitive(
        0,
        "SHIFT",
        PrimitiveConfig(d_model=d_model, rank=rank),
        max_sequence_length=num_buckets,
        arg_dim=arg_dim,
    )
    expected = (
        d_model * rank  # a_proj
        + rank * d_model  # b_proj
        + arg_dim * rank  # c_proj
        + num_buckets * arg_dim  # argument_encoder embedding table
    )
    assert primitive.num_parameters() == expected


def test_freeze_covers_argument_encoder_and_c_proj() -> None:
    primitive = _make_conditioned("SHIFT")
    primitive.freeze()
    assert primitive.is_frozen()
    assert all(not p.requires_grad for p in primitive.argument_encoder.parameters())
    assert all(not p.requires_grad for p in primitive.c_proj.parameters())


# --- sanity: required_argument_names stays the single source of truth --------


@pytest.mark.parametrize("operation", list(PARAMETERIZED_OPERATIONS))
def test_operation_still_declares_exactly_the_expected_argument(operation: str) -> None:
    assert get_operation(operation).required_argument_names == {
        PARAMETERIZED_OPERATIONS[operation]
    }


# --- ArgumentConditionedPrimitive / build_argument_conditioned_primitive -----
# (Task A1-R005D-006: conditioning architecture comparison -- V0 additive
# ConditionedPrimitive, V1 FiLMConditionedPrimitive, V2
# BasisModulatedConditionedPrimitive)


def _perturbed_film(primitive: FiLMConditionedPrimitive) -> FiLMConditionedPrimitive:
    with torch.no_grad():
        primitive.b_proj.weight.add_(1.0)
        primitive.film_proj.weight.add_(1.0)
    return primitive


def _perturbed_basis(
    primitive: BasisModulatedConditionedPrimitive,
) -> BasisModulatedConditionedPrimitive:
    with torch.no_grad():
        primitive.b_proj.weight.add_(1.0)
        primitive.basis.add_(1.0)
    return primitive


def _make_variant(
    variant: ConditioningVariant, operation: str = "SHIFT", **overrides: object
) -> ArgumentConditionedPrimitive:
    defaults: dict[str, object] = {
        "primitive_id": operation_id(operation),
        "operation": operation,
        "config": PrimitiveConfig(d_model=8, rank=4),
        "variant": variant,
        "max_sequence_length": 16,
        "vocab_size": 10,
        "arg_dim": 6,
    }
    defaults.update(overrides)
    return build_argument_conditioned_primitive(**defaults)  # type: ignore[arg-type]


def test_argument_conditioned_primitive_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ArgumentConditionedPrimitive(0, PrimitiveConfig(d_model=8, rank=4))  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("variant", "expected_type"),
    [
        (ConditioningVariant.ADDITIVE, ConditionedPrimitive),
        (ConditioningVariant.FILM, FiLMConditionedPrimitive),
        (ConditioningVariant.BASIS, BasisModulatedConditionedPrimitive),
    ],
)
def test_build_argument_conditioned_primitive_dispatches_on_variant(
    variant: ConditioningVariant, expected_type: type
) -> None:
    primitive = _make_variant(variant)
    assert isinstance(primitive, expected_type)
    assert isinstance(primitive, ArgumentConditionedPrimitive)
    assert primitive.argument_name == "amount"


def test_build_argument_conditioned_primitive_additive_matches_build_conditioned_primitive() -> (
    None
):
    """The default variant must reproduce build_conditioned_primitive's own
    ConditionedPrimitive exactly -- every pre-A1-R005D-006 caller's
    behavior."""
    via_variant = _make_variant(ConditioningVariant.ADDITIVE)
    via_direct = _make_conditioned("SHIFT")
    assert type(via_variant) is type(via_direct)
    assert via_variant.num_parameters() == via_direct.num_parameters()


def test_build_argument_conditioned_primitive_rejects_zero_argument_operations() -> None:
    with pytest.raises(ValueError, match="exactly one required argument"):
        build_argument_conditioned_primitive(
            0, "COPY", PrimitiveConfig(d_model=8, rank=4), ConditioningVariant.FILM
        )


def test_build_argument_conditioned_primitive_rejects_unregistered_operation() -> None:
    with pytest.raises(KeyError):
        build_argument_conditioned_primitive(
            0,
            "NOT_AN_OPERATION",
            PrimitiveConfig(d_model=8, rank=4),
            ConditioningVariant.FILM,
        )


# --- FiLMConditionedPrimitive (V1) --------------------------------------------


def test_film_forward_is_near_identity_at_init_regardless_of_argument() -> None:
    primitive = _make_variant(ConditioningVariant.FILM)
    h = torch.randn(3, 8)
    out_a = primitive(h, argument_values=[0, 0, 0])
    out_b = primitive(h, argument_values=[5, 5, 5])
    assert torch.equal(out_a, h)
    assert torch.equal(out_b, h)


def test_film_changing_argument_changes_output() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(2, 8)
    out_amount_1 = primitive(h, argument_values=[1, 1])
    out_amount_2 = primitive(h, argument_values=[2, 2])
    assert not torch.equal(out_amount_1, h)
    assert not torch.equal(out_amount_1, out_amount_2)


def test_film_same_family_handles_at_least_three_distinct_values() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(1, 8)
    outputs = [primitive(h, argument_values=[value]) for value in (0, 3, 7)]
    n_params_before = primitive.num_parameters()
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.equal(outputs[i], outputs[j])
    assert primitive.num_parameters() == n_params_before


def test_film_forward_requires_argument_values_keyword() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(2, 8)
    with pytest.raises(TypeError):
        primitive(h)  # type: ignore[call-arg]


def test_film_forward_rejects_mismatched_batch_size() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(3, 8)
    with pytest.raises(ValueError, match="batch"):
        primitive(h, argument_values=[1, 2])


def test_film_disabled_primitive_is_identity_even_with_argument() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    primitive.enabled = False
    h = torch.randn(2, 8)
    out = primitive(h, argument_values=[3, 3])
    assert torch.equal(out, h)


def test_film_forward_broadcasts_argument_across_sequence_positions() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(2, 5, 8)
    out = primitive(h, argument_values=[1, 2])
    assert out.shape == h.shape
    assert not torch.equal(out, h)


def test_film_forward_from_calls_matches_forward_with_the_same_raw_value() -> None:
    primitive = _perturbed_film(_make_variant(ConditioningVariant.FILM))
    h = torch.randn(2, 8)
    calls = [PrimitiveCall(operation="SHIFT", arguments={"amount": 4})] * 2
    via_calls = primitive.forward_from_calls(h, calls)
    via_raw = primitive(h, argument_values=[4, 4])
    assert torch.equal(via_calls, via_raw)


def test_film_num_parameters_includes_encoder_and_film_proj() -> None:
    d_model, rank, arg_dim, num_buckets = 8, 4, 6, 16
    primitive = _make_variant(
        ConditioningVariant.FILM, max_sequence_length=num_buckets, arg_dim=arg_dim
    )
    assert isinstance(primitive, FiLMConditionedPrimitive)
    expected = (
        d_model * rank  # a_proj
        + rank * d_model  # b_proj
        + arg_dim * (2 * rank)
        + (2 * rank)  # film_proj weight + bias
        + num_buckets * arg_dim  # argument_encoder embedding table
    )
    assert primitive.num_parameters() == expected


# --- BasisModulatedConditionedPrimitive (V2) ----------------------------------


def test_basis_forward_is_near_identity_at_init_regardless_of_argument() -> None:
    primitive = _make_variant(ConditioningVariant.BASIS)
    h = torch.randn(3, 8)
    out_a = primitive(h, argument_values=[0, 0, 0])
    out_b = primitive(h, argument_values=[5, 5, 5])
    assert torch.equal(out_a, h)
    assert torch.equal(out_b, h)


def test_basis_changing_argument_changes_output() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(2, 8)
    out_amount_1 = primitive(h, argument_values=[1, 1])
    out_amount_2 = primitive(h, argument_values=[2, 2])
    assert not torch.equal(out_amount_1, h)
    assert not torch.equal(out_amount_1, out_amount_2)


def test_basis_same_family_handles_at_least_three_distinct_values() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(1, 8)
    outputs = [primitive(h, argument_values=[value]) for value in (0, 3, 7)]
    n_params_before = primitive.num_parameters()
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.equal(outputs[i], outputs[j])
    assert primitive.num_parameters() == n_params_before


def test_basis_forward_requires_argument_values_keyword() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(2, 8)
    with pytest.raises(TypeError):
        primitive(h)  # type: ignore[call-arg]


def test_basis_forward_rejects_mismatched_batch_size() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(3, 8)
    with pytest.raises(ValueError, match="batch"):
        primitive(h, argument_values=[1, 2])


def test_basis_disabled_primitive_is_identity_even_with_argument() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    primitive.enabled = False
    h = torch.randn(2, 8)
    out = primitive(h, argument_values=[3, 3])
    assert torch.equal(out, h)


def test_basis_forward_broadcasts_argument_across_sequence_positions() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(2, 5, 8)
    out = primitive(h, argument_values=[1, 2])
    assert out.shape == h.shape
    assert not torch.equal(out, h)


def test_basis_forward_from_calls_matches_forward_with_the_same_raw_value() -> None:
    primitive = _perturbed_basis(_make_variant(ConditioningVariant.BASIS))
    h = torch.randn(2, 8)
    calls = [PrimitiveCall(operation="SHIFT", arguments={"amount": 4})] * 2
    via_calls = primitive.forward_from_calls(h, calls)
    via_raw = primitive(h, argument_values=[4, 4])
    assert torch.equal(via_calls, via_raw)


def test_basis_num_parameters_includes_encoder_mix_proj_and_basis() -> None:
    d_model, rank, arg_dim, num_buckets, num_basis = 8, 4, 6, 16, 3
    primitive = _make_variant(
        ConditioningVariant.BASIS,
        max_sequence_length=num_buckets,
        arg_dim=arg_dim,
        num_basis=num_basis,
    )
    assert isinstance(primitive, BasisModulatedConditionedPrimitive)
    expected = (
        d_model * rank  # a_proj
        + rank * d_model  # b_proj
        + arg_dim * num_basis
        + num_basis  # mix_proj weight + bias
        + num_basis * rank  # basis
        + num_buckets * arg_dim  # argument_encoder embedding table
    )
    assert primitive.num_parameters() == expected


def test_basis_rejects_non_positive_num_basis() -> None:
    with pytest.raises(ValueError, match="num_basis"):
        _make_variant(ConditioningVariant.BASIS, num_basis=0)


# --- Bank-level acceptance, generalized across variants -----------------------


@pytest.mark.parametrize(
    "variant", [ConditioningVariant.ADDITIVE, ConditioningVariant.FILM, ConditioningVariant.BASIS]
)
def test_bank_holds_exactly_one_primitive_per_variant(variant: ConditioningVariant) -> None:
    bank = PrimitiveBank()
    primitive = _make_variant(variant, operation="SHIFT")
    bank.add_primitive(primitive)
    assert len(bank) == 1
    assert bank.get(operation_id("SHIFT")) is primitive
