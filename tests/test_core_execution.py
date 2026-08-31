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
