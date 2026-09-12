"""Unit tests for Task B-C005REC-004AC: Parallel Low-Rank Score-Residual Routing Pilot."""

import torch
import torch.nn.functional as F

from apc.evaluation.mirror_kv_role_split_pilot import (
    RoleSplitCrossPositionLengthBiasPrimitive,
    evaluate_role_split_forward_with_stages,
)
from apc.evaluation.mirror_parallel_score_residual_pilot import (
    ParallelScoreResidual,
    RoleSplitParallelScoreResidualCrossPositionLengthBiasPrimitive,
    evaluate_parallel_score_residual_forward_with_stages,
)
from apc.primitives.primitive import CrossPositionLengthBiasPrimitiveConfig


def test_parallel_score_residual_zero_initialization() -> None:
    """Tests that ParallelScoreResidual has exact zero output upon initialization."""
    module = ParallelScoreResidual(d_in=32, rank=4, seed=10)
    assert module.rank == 4
    assert module.w_qr.weight.shape == (4, 32)
    assert module.w_kr.weight.shape == (4, 32)
    assert module.w_qr.bias is None
    assert module.w_kr.bias is None

    # Verify w_kr is exact zero
    assert torch.all(module.w_kr.weight == 0.0)
    # Verify w_qr is non-zero random
    assert torch.any(module.w_qr.weight != 0.0)

    # Test forward with arbitrary input
    h_query = torch.randn(4, 8, 32)
    h_key = torch.randn(4, 10, 32)
    delta_s = module(h_query, h_key)
    assert delta_s.shape == (4, 8, 10)
    assert torch.all(delta_s == 0.0)


def test_role_split_parallel_score_residual_primitive_forward_parity() -> None:
    """Tests forward parity between role-split primitive and
    role-split + zero score residual.
    """
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=64,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
        arg_dim=16,
        bias_hidden_dim=32,
        length_ref=32,
    )
    rs_model = RoleSplitCrossPositionLengthBiasPrimitive(primitive_id=1, config=cfg)
    res_model = RoleSplitParallelScoreResidualCrossPositionLengthBiasPrimitive(
        primitive_id=1, config=cfg, residual_rank=4, residual_seed=10
    )

    # Copy all common parameters from rs_model to res_model
    rs_sd = rs_model.state_dict()
    res_sd = res_model.state_dict()
    for k, v in rs_sd.items():
        if k in res_sd:
            res_sd[k].copy_(v)

    rs_model.eval()
    res_model.eval()

    content_features = torch.randn(4, 8, 64)
    content_lengths = [8, 6, 8, 4]
    output_lengths = [8, 6, 8, 4]

    with torch.no_grad():
        rs_out = rs_model(content_features, content_lengths, output_lengths)
        res_out = res_model(content_features, content_lengths, output_lengths)

        rs_stages = evaluate_role_split_forward_with_stages(
            rs_model, content_features, content_lengths, output_lengths
        )
        res_stages = evaluate_parallel_score_residual_forward_with_stages(
            res_model, content_features, content_lengths, output_lengths
        )

    # Output diff must be zero
    assert torch.allclose(rs_out, res_out, atol=1e-6)
    assert torch.allclose(
        rs_stages["final_token_logits"], res_stages["final_token_logits"], atol=1e-6
    )
    assert torch.allclose(rs_stages["attn_probs"], res_stages["attn_probs"], atol=1e-6)
    assert torch.allclose(rs_stages["score_logits"], res_stages["score_logits"], atol=1e-6)
    assert torch.allclose(rs_stages["attn_out"], res_stages["attn_out"], atol=1e-6)
    assert torch.all(res_stages["delta_s"] == 0.0)


def test_parallel_score_residual_gradient_flow() -> None:
    """Tests that normal backward pass propagates gradients to W_kr in score residual."""
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=64,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
        arg_dim=16,
        bias_hidden_dim=32,
        length_ref=32,
    )
    res_model = RoleSplitParallelScoreResidualCrossPositionLengthBiasPrimitive(
        primitive_id=1, config=cfg, residual_rank=4, residual_seed=10
    )
    res_model.train()

    content_features = torch.randn(2, 6, 64)
    content_lengths = [6, 6]
    output_lengths = [6, 6]
    target = torch.randint(0, 10, (2, 6))

    logits = res_model(content_features, content_lengths, output_lengths)
    loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
    loss.backward()

    # W_kr must have non-zero gradient
    assert res_model.score_residual.w_kr.weight.grad is not None
    assert res_model.score_residual.w_kr.weight.grad.norm().item() > 0.0

    # Key branch must have non-zero gradient
    assert res_model.key_content_in_proj.weight.grad is not None
    assert res_model.key_content_in_proj.weight.grad.norm().item() > 0.0


def test_parallel_score_residual_o1_structural_isolation() -> None:
    """Tests that score residual parameters are structurally isolated from O1 oracle output."""
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=64,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
        arg_dim=16,
        bias_hidden_dim=32,
        length_ref=32,
    )
    res_model = RoleSplitParallelScoreResidualCrossPositionLengthBiasPrimitive(
        primitive_id=1, config=cfg, residual_rank=4, residual_seed=10
    )
    res_model.eval()

    content_features = torch.randn(2, 6, 64)
    content_lengths = [6, 6]
    output_lengths = [6, 6]

    res_model.zero_grad(set_to_none=True)
    stages = evaluate_parallel_score_residual_forward_with_stages(
        res_model, content_features, content_lengths, output_lengths, oracle_attention=True
    )
    o1_logits = stages["final_token_logits"]
    loss = o1_logits.sum()
    loss.backward(retain_graph=True)

    w_qr_grad = res_model.score_residual.w_qr.weight.grad
    w_kr_grad = res_model.score_residual.w_kr.weight.grad
    assert w_qr_grad is None or float(w_qr_grad.norm().item()) == 0.0
    assert w_kr_grad is None or float(w_kr_grad.norm().item()) == 0.0

    # Perturbation check
    orig_w_qr = res_model.score_residual.w_qr.weight.detach().clone()
    orig_w_kr = res_model.score_residual.w_kr.weight.detach().clone()

    with torch.no_grad():
        res_model.score_residual.w_qr.weight.add_(torch.randn_like(orig_w_qr) * 1.0)
        res_model.score_residual.w_kr.weight.add_(torch.randn_like(orig_w_kr) * 1.0)

        stages_pert = evaluate_parallel_score_residual_forward_with_stages(
            res_model, content_features, content_lengths, output_lengths, oracle_attention=True
        )
        o1_logits_pert = stages_pert["final_token_logits"]

        # Restore
        res_model.score_residual.w_qr.weight.copy_(orig_w_qr)
        res_model.score_residual.w_kr.weight.copy_(orig_w_kr)

        assert torch.allclose(o1_logits, o1_logits_pert, atol=1e-7)
