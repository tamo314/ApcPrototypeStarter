from __future__ import annotations

import math

import pytest
import torch

from apc.primitives.router import Router, RouterConfig


def _make_router(
    n_candidates: int, *, top_k: int = 2, d_model: int = 8, score_fn: str = "dot"
) -> tuple[Router, list[int]]:
    router = Router(RouterConfig(d_model=d_model, top_k=top_k, score_fn=score_fn))
    ids = list(range(n_candidates))
    for primitive_id in ids:
        router.add_primitive_key(primitive_id)
    return router, ids


def test_add_primitive_key_rejects_duplicate_id() -> None:
    router, _ = _make_router(1)
    with pytest.raises(ValueError, match="already has a key"):
        router.add_primitive_key(0)


def test_remove_primitive_key_unknown_id_raises() -> None:
    router, _ = _make_router(1)
    with pytest.raises(KeyError):
        router.remove_primitive_key(99)


def test_forward_unknown_candidate_id_raises() -> None:
    router, ids = _make_router(2)
    h = torch.randn(4, router.config.d_model)
    with pytest.raises(KeyError, match="99"):
        router(h, [*ids, 99])


def test_forward_empty_candidates_raises() -> None:
    router, _ = _make_router(2)
    h = torch.randn(4, router.config.d_model)
    with pytest.raises(ValueError, match="non-empty"):
        router(h, [])


def test_selects_at_most_top_k() -> None:
    router, ids = _make_router(5, top_k=3)
    h = torch.randn(6, router.config.d_model)
    out = router(h, ids)
    assert out.selected_ids.shape == (6, 3)
    assert out.weights.shape == (6, 3)
    for row in out.selected_ids.tolist():
        assert len(row) == len(set(row))  # no duplicate primitive within one selection


def test_top_k_clamped_when_fewer_candidates_than_k() -> None:
    router, ids = _make_router(2, top_k=5)
    h = torch.randn(3, router.config.d_model)
    out = router(h, ids)
    assert out.selected_ids.shape == (3, 2)
    assert out.weights.shape == (3, 2)


def test_selected_ids_are_a_subset_of_candidate_ids() -> None:
    router, ids = _make_router(6, top_k=2)
    h = torch.randn(10, router.config.d_model)
    out = router(h, ids)
    selected = set(out.selected_ids.reshape(-1).tolist())
    assert selected <= set(ids)


def test_weights_sum_to_one_over_selected() -> None:
    router, ids = _make_router(5, top_k=3)
    h = torch.randn(4, router.config.d_model)
    out = router(h, ids)
    sums = out.weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_probs_sum_to_one_over_all_candidates() -> None:
    router, ids = _make_router(5, top_k=2)
    h = torch.randn(4, router.config.d_model)
    out = router(h, ids)
    assert out.probs.shape == (4, 5)
    sums = out.probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_entropy_matches_manual_computation() -> None:
    router, ids = _make_router(4, top_k=2)
    h = torch.randn(3, router.config.d_model)
    out = router(h, ids)
    manual = -(out.probs * out.probs.clamp_min(1e-12).log()).sum(dim=-1)
    assert torch.allclose(out.entropy, manual, atol=1e-5)


def test_uniform_scores_give_max_entropy() -> None:
    # Zero every key and the query projection so every candidate scores 0
    # and the distribution is exactly uniform.
    router, ids = _make_router(4, top_k=2)
    with torch.no_grad():
        router.query_proj.weight.zero_()
        router.query_proj.bias.zero_()
        for primitive_id in ids:
            router._keys[str(primitive_id)].zero_()
    h = torch.randn(2, router.config.d_model)
    out = router(h, ids)
    expected = math.log(len(ids))
    assert torch.allclose(out.entropy, torch.full_like(out.entropy, expected), atol=1e-5)


def test_forward_is_deterministic_given_seed() -> None:
    torch.manual_seed(0)
    router_a, ids_a = _make_router(5, top_k=2)
    torch.manual_seed(0)
    router_b, ids_b = _make_router(5, top_k=2)
    assert ids_a == ids_b

    h = torch.randn(4, router_a.config.d_model)
    out_a = router_a(h, ids_a)
    out_b = router_b(h, ids_b)

    assert torch.equal(out_a.selected_ids, out_b.selected_ids)
    assert torch.allclose(out_a.weights, out_b.weights)
    assert torch.allclose(out_a.probs, out_b.probs)


def test_repeated_forward_on_same_input_is_stable() -> None:
    router, ids = _make_router(4, top_k=2)
    router.eval()
    h = torch.randn(3, router.config.d_model)
    out_1 = router(h, ids)
    out_2 = router(h, ids)
    assert torch.equal(out_1.selected_ids, out_2.selected_ids)
    assert torch.allclose(out_1.weights, out_2.weights)


def test_gradient_flows_only_to_selected_candidate_keys() -> None:
    # top_k=2 out of 3 candidates: id0/id1 are selected, id2 is not.
    # `weights.sum()` is always exactly 1 (softmax invariant) and would
    # give a zero gradient everywhere, so the loss instead reads out a
    # single selected weight -- that still only depends on the *selected*
    # candidates' scores, which is exactly the path under test.
    router, ids = _make_router(3, top_k=2, d_model=4)
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(4))
        router.query_proj.bias.zero_()
        router._keys[str(ids[0])].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        router._keys[str(ids[1])].copy_(torch.tensor([0.5, 0.0, 0.0, 0.0]))
        router._keys[str(ids[2])].copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = router(h, ids)
    assert out.selected_ids.tolist() == [[ids[0], ids[1]]]

    loss = out.weights[0, 0]
    loss.backward()

    selected_key_0 = router._keys[str(ids[0])]
    selected_key_1 = router._keys[str(ids[1])]
    unselected_key = router._keys[str(ids[2])]

    assert selected_key_0.grad is not None
    assert torch.any(selected_key_0.grad != 0)
    assert selected_key_1.grad is not None
    assert torch.any(selected_key_1.grad != 0)
    assert unselected_key.grad is None or torch.all(unselected_key.grad == 0)

    assert router.query_proj.weight.grad is not None
    assert torch.any(router.query_proj.weight.grad != 0)


def test_usage_logging_counts_selected_primitives() -> None:
    router, ids = _make_router(4, top_k=2)
    assert router.usage_counts() == {pid: 0 for pid in ids}

    h = torch.randn(5, router.config.d_model)
    out = router(h, ids)

    counted = sum(router.usage_counts().values())
    assert counted == out.selected_ids.numel()
    for row in out.selected_ids.tolist():
        for pid in row:
            assert router.usage_counts()[pid] >= 1


def test_usage_counts_accumulate_across_forward_calls() -> None:
    router, ids = _make_router(3, top_k=1, d_model=4)
    with torch.no_grad():
        router.query_proj.weight.copy_(torch.eye(4))
        router.query_proj.bias.zero_()
        router._keys[str(ids[0])].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        router._keys[str(ids[1])].copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        router._keys[str(ids[2])].copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    router(h, ids)
    router(h, ids)
    router(h, ids)

    assert router.usage_counts()[ids[0]] == 3
    assert router.usage_counts()[ids[1]] == 0
    assert router.usage_counts()[ids[2]] == 0


def test_remove_primitive_key_drops_usage_entry() -> None:
    router, ids = _make_router(2)
    router.remove_primitive_key(ids[0])
    assert ids[0] not in router.usage_counts()
    assert router.ids() == [ids[1]]


def test_forward_accepts_unbatched_and_sequence_shapes() -> None:
    router, ids = _make_router(4, top_k=2)

    h_vec = torch.randn(router.config.d_model)
    out_vec = router(h_vec, ids)
    assert out_vec.selected_ids.shape == (2,)
    assert out_vec.weights.shape == (2,)
    assert out_vec.probs.shape == (4,)
    assert out_vec.entropy.shape == ()

    h_seq = torch.randn(2, 6, router.config.d_model)
    out_seq = router(h_seq, ids)
    assert out_seq.selected_ids.shape == (2, 6, 2)
    assert out_seq.weights.shape == (2, 6, 2)
    assert out_seq.probs.shape == (2, 6, 4)
    assert out_seq.entropy.shape == (2, 6)


def test_cosine_score_fn_runs_and_normalizes_direction_only() -> None:
    router, ids = _make_router(3, top_k=2, score_fn="cosine")
    h = torch.randn(4, router.config.d_model)
    out = router(h, ids)
    assert out.probs.shape == (4, 3)
    sums = out.weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_router_config_validates_fields() -> None:
    with pytest.raises(ValueError, match="d_model"):
        RouterConfig(d_model=0)
    with pytest.raises(ValueError, match="top_k"):
        RouterConfig(d_model=4, top_k=0)
    with pytest.raises(ValueError, match="score_fn"):
        RouterConfig(d_model=4, score_fn="bogus")
