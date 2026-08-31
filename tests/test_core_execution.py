from __future__ import annotations

import torch

from apc.core.execution import (
    NULL_PRIMITIVE_ID,
    apply_bank,
    apply_workspace,
    ensure_null_key,
    forward_logits,
    generate_greedy_with_capacity,
    stable_candidate_ids,
    sum_primitive_deltas,
)
from apc.core.generation import evaluate_exact_match, generate_greedy
from apc.core.model import DecoderOnlyTransformer, TransformerConfig
from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example
from apc.environments.interpreter import OperationGraph
from apc.environments.program import Program
from apc.plastic.allocator import Allocator, AllocatorPreset
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

D_MODEL = 8


def _model() -> DecoderOnlyTransformer:
    set_seed(0)
    return DecoderOnlyTransformer(
        TransformerConfig(
            vocab_size=14, max_seq_len=32, d_model=D_MODEL, n_layer=2, n_head=2, d_ff=16
        )
    )


def _router(top_k: int = 2) -> Router:
    return Router(RouterConfig(d_model=D_MODEL, top_k=top_k))


def _bank_with_stable_primitive(rank: int = 4, *, perturb: bool = True) -> PrimitiveBank:
    bank = PrimitiveBank()
    primitive = bank.new_primitive(
        PrimitiveConfig(d_model=D_MODEL, rank=rank), status=PrimitiveStatus.STABLE
    )
    if perturb:
        with torch.no_grad():
            primitive.b_proj.weight.add_(1.0)
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    return bank


def _bank_with_stable_primitives(count: int, rank: int = 4) -> PrimitiveBank:
    bank = PrimitiveBank()
    for _ in range(count):
        primitive = bank.new_primitive(
            PrimitiveConfig(d_model=D_MODEL, rank=rank), status=PrimitiveStatus.STABLE
        )
        with torch.no_grad():
            primitive.b_proj.weight.add_(1.0)
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    return bank


def _configure_identity_router(router: Router, primitive_ids: list[int]) -> None:
    ensure_null_key(router)
    for primitive_id in primitive_ids:
        router.add_primitive_key(primitive_id)
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(D_MODEL))
        assert router.query_proj.bias is not None
        router.query_proj.bias.zero_()
        router.key_parameter(NULL_PRIMITIVE_ID).zero_()
        for primitive_id in primitive_ids:
            key = torch.zeros(D_MODEL)
            key[primitive_id] = 1.0
            router.key_parameter(primitive_id).copy_(key)


def _dense_top_k_reference(
    hidden: torch.Tensor, bank: PrimitiveBank, router: Router, primitive_ids: list[int]
) -> torch.Tensor:
    """The pre-A1-002 dense implementation, used only as a numerical oracle."""
    router_out = router(hidden, [NULL_PRIMITIVE_ID, *primitive_ids])
    combined_delta = torch.zeros_like(hidden)
    for primitive_id in primitive_ids:
        selected = (router_out.selected_ids == primitive_id).to(hidden.dtype)
        weight = (selected * router_out.weights).sum(dim=-1, keepdim=True)
        primitive = bank.get(primitive_id)
        combined_delta = combined_delta + weight * primitive.b_proj(primitive.a_proj(hidden))
    return hidden + combined_delta


# --- apply_bank --------------------------------------------------------


def test_apply_bank_with_no_stable_primitives_returns_hidden_unchanged() -> None:
    bank = PrimitiveBank()
    router = _router()
    h = torch.randn(2, 3, D_MODEL)
    out, router_out = apply_bank(h, bank, router, stable_ids=[])
    assert torch.equal(out, h)
    assert router_out.candidate_ids == [NULL_PRIMITIVE_ID]


def test_apply_bank_registers_null_key_lazily() -> None:
    router = _router()
    assert not router.has_primitive(NULL_PRIMITIVE_ID)
    ensure_null_key(router)
    assert router.has_primitive(NULL_PRIMITIVE_ID)
    ensure_null_key(router)  # idempotent, does not raise on re-registration
    assert router.has_primitive(NULL_PRIMITIVE_ID)


def test_apply_bank_output_never_requires_grad() -> None:
    bank = _bank_with_stable_primitive()
    for p in bank.parameters():
        p.requires_grad_(True)  # even if the caller forgets to freeze
    router = _router()
    router.add_primitive_key(bank.ids()[0])
    h = torch.randn(2, 3, D_MODEL, requires_grad=True)
    out, _ = apply_bank(h, bank, router, stable_ids=stable_candidate_ids(bank))
    assert not out.requires_grad


def test_apply_bank_skips_disabled_primitive() -> None:
    bank = _bank_with_stable_primitive()
    pid = bank.ids()[0]
    router = _router()
    router.add_primitive_key(pid)
    bank.disable(pid)
    h = torch.randn(2, 3, D_MODEL)

    # stable_candidate_ids already excludes disabled primitives...
    assert stable_candidate_ids(bank) == []

    # ...and even if a caller explicitly passes a disabled id, apply_bank
    # contributes no delta for it.
    out, _ = apply_bank(h, bank, router, stable_ids=[pid])
    assert torch.equal(out, h)


def test_apply_bank_null_candidate_is_never_looked_up_in_bank() -> None:
    # Passing an empty bank but a router that already knows about the null
    # id must not raise, even though NULL_PRIMITIVE_ID has no bank entry.
    bank = PrimitiveBank()
    router = _router()
    ensure_null_key(router)
    h = torch.randn(2, 3, D_MODEL)
    out, router_out = apply_bank(h, bank, router, stable_ids=[])
    assert torch.equal(out, h)
    assert NULL_PRIMITIVE_ID in router_out.candidate_ids


def test_apply_bank_executes_only_selected_primitives_for_a_batched_input() -> None:
    bank = _bank_with_stable_primitives(4)
    router = _router(top_k=1)
    primitive_ids = bank.ids()
    _configure_identity_router(router, primitive_ids)
    # Different batch/sequence positions select primitives 0 and 1.  The
    # other two bank entries must not run at all.
    h = torch.stack((torch.eye(D_MODEL)[0], torch.eye(D_MODEL)[1])).reshape(1, 2, D_MODEL)
    expected = _dense_top_k_reference(h, bank, router, primitive_ids)
    for primitive_id in primitive_ids:
        bank.get(primitive_id).reset_forward_call_count()

    actual, router_out = apply_bank(h, bank, router, stable_ids=primitive_ids)

    torch.testing.assert_close(actual, expected)
    assert router_out.executed_primitive_ids == (0, 1)
    assert [bank.get(pid).forward_call_count for pid in primitive_ids] == [1, 1, 0, 0]
    assert bank.active_parameter_count(router_out.executed_primitive_ids) == (
        2 * bank.get(0).num_parameters()
    )
    assert bank.persistent_parameter_count() == 4 * bank.get(0).num_parameters()


def test_apply_bank_matches_dense_execution_when_top_k_selects_every_candidate() -> None:
    bank = _bank_with_stable_primitives(3)
    router = _router(top_k=4)  # three real primitives plus the null candidate
    primitive_ids = bank.ids()
    _configure_identity_router(router, primitive_ids)
    h = torch.randn(2, 3, D_MODEL)
    expected = _dense_top_k_reference(h, bank, router, primitive_ids)
    for primitive_id in primitive_ids:
        bank.get(primitive_id).reset_forward_call_count()

    actual, router_out = apply_bank(h, bank, router, stable_ids=primitive_ids)

    torch.testing.assert_close(actual, expected)
    assert router_out.executed_primitive_ids == tuple(primitive_ids)
    assert [bank.get(pid).forward_call_count for pid in primitive_ids] == [1, 1, 1]


# --- apply_workspace / sum_primitive_deltas --------------------------------


def test_apply_workspace_unallocated_returns_hidden_unchanged() -> None:
    workspace = PlasticWorkspace()
    h = torch.randn(2, 3, D_MODEL)
    assert torch.equal(apply_workspace(h, workspace), h)


def test_apply_workspace_fresh_transforms_are_identity() -> None:
    workspace = PlasticWorkspace()
    Allocator(d_model=D_MODEL).allocate(workspace, AllocatorPreset.SMALL)
    h = torch.randn(2, 3, D_MODEL)
    # B is zero-initialized, so a fresh batch of temporary transforms is a no-op.
    assert torch.equal(apply_workspace(h, workspace), h)


def test_apply_workspace_gradient_flows_into_temporary_transforms() -> None:
    workspace = PlasticWorkspace()
    ids = Allocator(d_model=D_MODEL).allocate(workspace, AllocatorPreset.SMALL)
    h = torch.randn(2, 3, D_MODEL)
    out = apply_workspace(h, workspace, ids)
    out.pow(2).sum().backward()
    for tid in ids:
        transform = workspace.get(tid)
        assert transform.a_proj.weight.grad is not None
        assert transform.b_proj.weight.grad is not None


def test_sum_primitive_deltas_skips_disabled_transforms() -> None:
    workspace = PlasticWorkspace()
    ids = Allocator(d_model=D_MODEL).allocate(workspace, AllocatorPreset.SMALL)
    for tid in ids:
        with torch.no_grad():
            workspace.get(tid).b_proj.weight.add_(1.0)
    workspace.get(ids[0]).enabled = False
    h = torch.randn(2, D_MODEL)
    enabled_only = sum_primitive_deltas([workspace.get(ids[0])], h)
    assert torch.equal(enabled_only, torch.zeros_like(h))


# --- forward_logits / generate_greedy_with_capacity ------------------------


def test_forward_logits_matches_plain_model_when_bank_and_workspace_are_empty() -> None:
    model = _model()
    bank = PrimitiveBank()
    router = _router()
    input_ids = torch.randint(0, 14, (2, 5))
    with torch.no_grad():
        expected = model(input_ids)
        actual, _ = forward_logits(model, input_ids, bank, router)
    assert torch.equal(actual, expected)


def test_forward_logits_shape_with_bank_and_workspace_applied() -> None:
    model = _model()
    bank = _bank_with_stable_primitive()
    router = _router()
    router.add_primitive_key(bank.ids()[0])
    workspace = PlasticWorkspace()
    ids = Allocator(d_model=D_MODEL).allocate(workspace, AllocatorPreset.SMALL)
    input_ids = torch.randint(0, 14, (2, 5))
    logits, router_out = forward_logits(
        model, input_ids, bank, router, workspace=workspace, workspace_ids=ids
    )
    assert logits.shape == (2, 5, 14)
    assert router_out.entropy.shape == (2, 5)


def test_generate_greedy_with_capacity_matches_plain_generation_when_empty() -> None:
    model = _model()
    bank = PrimitiveBank()
    router = _router()
    specials = build_special_tokens(10)
    prompt = torch.tensor([[specials.bos, 1, 2, specials.sep]])

    plain = generate_greedy(model, prompt, specials.eos, max_new_tokens=6)
    capacity = generate_greedy_with_capacity(
        model, bank, router, prompt, specials.eos, max_new_tokens=6
    )
    assert plain == capacity


def test_evaluate_exact_match_with_capacity_matches_plain_when_empty() -> None:
    from apc.core.execution import evaluate_exact_match_with_capacity

    model = _model()
    bank = PrimitiveBank()
    router = _router()
    specials = build_special_tokens(10)
    example = Example(
        input_tokens=(1, 2),
        target_tokens=(3, 4),
        program=Program(steps=()),
        operation_graph=OperationGraph(nodes=()),
        category="known",
        split="train",
        vocab_size=10,
    )
    plain_score, plain_preds = evaluate_exact_match(model, [example], specials)
    cap_score, cap_preds = evaluate_exact_match_with_capacity(
        model, [example], specials, bank, router
    )
    assert plain_score == cap_score
    assert plain_preds == cap_preds
