"""Unit tests for Task B-C005REC-004AB: Post-Attention Compact Residual Bypass Pilot."""

import torch
import torch.nn.functional as F

from apc.evaluation.mirror_kv_role_split_pilot import (
    RoleSplitCrossPositionLengthBiasPrimitive,
    evaluate_role_split_forward_with_stages,
)
from apc.evaluation.mirror_post_attn_residual_pilot import (
    CompactPostAttnResidual,
    RoleSplitPostAttnResidualCrossPositionLengthBiasPrimitive,
    evaluate_post_attn_residual_forward_with_stages,
)
from apc.primitives.primitive import CrossPositionLengthBiasPrimitiveConfig


def test_compact_post_attn_residual_zero_initialization() -> None:
    """Tests that CompactPostAttnResidual has exact zero output upon initialization."""
    module = CompactPostAttnResidual(d_in=32, rank=4, seed=10)
    assert module.rank == 4
    assert module.w_down.weight.shape == (4, 32)
    assert module.w_up.weight.shape == (32, 4)
    assert module.w_up.bias is None
    assert module.w_down.bias is None

    # Verify w_up is exact zero
    assert torch.all(module.w_up.weight == 0.0)
    # Verify w_down is non-zero random
    assert torch.any(module.w_down.weight != 0.0)

    # Test forward with arbitrary input
    x = torch.randn(8, 10, 32)
    out = module(x)
    assert out.shape == (8, 10, 32)
    assert torch.all(out == 0.0)


def test_role_split_post_attn_residual_primitive_forward_parity() -> None:
    """Tests forward parity between role-split primitive and
    role-split + zero post-attn residual.
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
    res_model = RoleSplitPostAttnResidualCrossPositionLengthBiasPrimitive(
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
        res_stages = evaluate_post_attn_residual_forward_with_stages(
            res_model, content_features, content_lengths, output_lengths
        )

    # Output diff must be zero
    assert torch.allclose(rs_out, res_out, atol=1e-6)
    assert torch.allclose(
        rs_stages["final_token_logits"], res_stages["final_token_logits"], atol=1e-6
    )
    assert torch.allclose(rs_stages["attn_probs"], res_stages["attn_probs"], atol=1e-6)
    assert torch.allclose(rs_stages["score_logits"], res_stages["score_logits"], atol=1e-6)
    assert torch.allclose(rs_stages["attn_out"], res_stages["h_attn_base"], atol=1e-6)
    assert torch.allclose(
        res_stages["r_post_attn"], torch.zeros_like(res_stages["r_post_attn"])
    )


def test_post_attn_residual_gradient_flow() -> None:
    """Tests that backward pass propagates gradients to W_up in post-attn residual."""
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
    res_model = RoleSplitPostAttnResidualCrossPositionLengthBiasPrimitive(
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

    # W_up must have non-zero gradient
    assert res_model.post_attn_residual.w_up.weight.grad is not None
    assert res_model.post_attn_residual.w_up.weight.grad.norm().item() > 0.0

    # Key branch must have non-zero gradient
    assert res_model.key_content_in_proj.weight.grad is not None
    assert res_model.key_content_in_proj.weight.grad.norm().item() > 0.0


def test_post_attn_residual_score_path_isolation() -> None:
    """Tests that POST_ATTN_RESIDUAL parameters do not leak into attention score logits."""
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
    res_model = RoleSplitPostAttnResidualCrossPositionLengthBiasPrimitive(
        primitive_id=1, config=cfg, residual_rank=4, residual_seed=10
    )
    res_model.eval()

    content_features = torch.randn(2, 6, 64)
    content_lengths = [6, 6]
    output_lengths = [6, 6]

    res_model.zero_grad(set_to_none=True)
    stages = evaluate_post_attn_residual_forward_with_stages(
        res_model, content_features, content_lengths, output_lengths
    )
    scores = stages["score_logits"]
    finite_mask = torch.isfinite(scores)
    score_loss = scores[finite_mask].sum()
    score_loss.backward(retain_graph=True)

    w_down_grad = res_model.post_attn_residual.w_down.weight.grad
    w_up_grad = res_model.post_attn_residual.w_up.weight.grad
    assert w_down_grad is None or float(w_down_grad.norm().item()) == 0.0
    assert w_up_grad is None or float(w_up_grad.norm().item()) == 0.0

    # Perturbation check
    orig_w_down = res_model.post_attn_residual.w_down.weight.detach().clone()
    orig_w_up = res_model.post_attn_residual.w_up.weight.detach().clone()

    with torch.no_grad():
        res_model.post_attn_residual.w_down.weight.add_(torch.randn_like(orig_w_down) * 1.0)
        res_model.post_attn_residual.w_up.weight.add_(torch.randn_like(orig_w_up) * 1.0)

        stages_pert = evaluate_post_attn_residual_forward_with_stages(
            res_model, content_features, content_lengths, output_lengths
        )
        scores_pert = stages_pert["score_logits"]
        probs_pert = stages_pert["attn_probs"]

        # Restore
        res_model.post_attn_residual.w_down.weight.copy_(orig_w_down)
        res_model.post_attn_residual.w_up.weight.copy_(orig_w_up)

        assert torch.allclose(scores[finite_mask], scores_pert[finite_mask], atol=1e-7)
        assert torch.allclose(stages["attn_probs"], probs_pert, atol=1e-7)
