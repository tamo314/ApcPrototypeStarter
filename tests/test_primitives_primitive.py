from __future__ import annotations

import torch

from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus


def _make(**overrides: object) -> Primitive:
    defaults: dict[str, object] = {"primitive_id": 0, "config": PrimitiveConfig(d_model=8, rank=2)}
    defaults.update(overrides)
    return Primitive(**defaults)  # type: ignore[arg-type]


def test_forward_is_identity_at_init() -> None:
    # B is zero-initialized, so a fresh primitive is a no-op regardless of A.
    primitive = _make()
    h = torch.randn(3, 5, 8)
    assert torch.equal(primitive(h), h)


def test_forward_disabled_is_identity_even_after_training() -> None:
    primitive = _make(enabled=True)
    nn_params = list(primitive.parameters())
    with torch.no_grad():
        nn_params[1].add_(1.0)  # perturb B so forward is no longer a no-op
    h = torch.randn(2, 4, 8)
    assert not torch.equal(primitive(h), h)
    primitive.enabled = False
    assert torch.equal(primitive(h), h)


def test_gate_scales_the_residual() -> None:
    primitive = _make()
    with torch.no_grad():
        primitive.b_proj.weight.add_(1.0)
    h = torch.randn(2, 8)
    out_full = primitive(h, gate=1.0)
    out_zero = primitive(h, gate=0.0)
    assert torch.equal(out_zero, h)
    assert not torch.equal(out_full, h)


def test_num_parameters_matches_low_rank_shapes() -> None:
    primitive = _make(config=PrimitiveConfig(d_model=16, rank=4))
    assert primitive.num_parameters() == 16 * 4 * 2


def test_freeze_and_unfreeze() -> None:
    primitive = _make()
    assert not primitive.is_frozen()
    primitive.freeze()
    assert primitive.is_frozen()
    assert all(not p.requires_grad for p in primitive.parameters())
    primitive.unfreeze()
    assert not primitive.is_frozen()
    assert all(p.requires_grad for p in primitive.parameters())


def test_frozen_primitive_is_unchanged_by_an_optimizer_step() -> None:
    primitive = _make()
    with torch.no_grad():
        primitive.b_proj.weight.add_(1.0)  # non-zero so gradients are non-trivial
    primitive.freeze()
    before = [p.detach().clone() for p in primitive.parameters()]

    optimizer = torch.optim.SGD(
        [p for p in primitive.parameters() if p.requires_grad] or [torch.zeros(1)], lr=1.0
    )
    h = torch.randn(4, 8, requires_grad=True)
    out = primitive(h)
    loss = out.pow(2).sum()
    loss.backward()
    optimizer.step()

    after = list(primitive.parameters())
    for p_before, p_after in zip(before, after, strict=True):
        assert torch.equal(p_before, p_after)


def test_usage_and_utility_tracking() -> None:
    primitive = _make()
    assert primitive.usage_count == 0
    primitive.record_usage()
    primitive.record_usage()
    assert primitive.usage_count == 2

    assert primitive.utility_ema == 0.0
    primitive.update_utility(1.0, decay=0.5)
    assert primitive.utility_ema == 0.5
    primitive.update_utility(1.0, decay=0.5)
    assert primitive.utility_ema == 0.75


def test_num_parameters_trainable_only_reflects_freeze() -> None:
    primitive = _make()
    total = primitive.num_parameters()
    assert primitive.num_parameters(trainable_only=True) == total
    primitive.freeze()
    assert primitive.num_parameters(trainable_only=True) == 0
    assert primitive.num_parameters() == total


def test_default_status_is_candidate() -> None:
    primitive = _make()
    assert primitive.status == PrimitiveStatus.CANDIDATE


def test_metadata_is_copied_not_aliased() -> None:
    meta = {"label": "copy-op"}
    primitive = _make(metadata=meta)
    meta["label"] = "mutated"
    assert primitive.metadata["label"] == "copy-op"
