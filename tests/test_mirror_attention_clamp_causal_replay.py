"""Unit tests for Task B-C005REC-004U: I03 Pre-Transition Attention-Clamp Causal Replay."""

from __future__ import annotations

import inspect

import torch

from apc.core.data import collate_content_only_batch
from apc.evaluation import mirror_attention_clamp_causal_replay as rec004u
from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive


def test_oracle_information_boundary_signature() -> None:
    """Confirms that clamped forward and reference extraction APIs have zero oracle params."""
    # 1. extract_reference_attention signature
    sig_ref = inspect.signature(rec004u.extract_reference_attention)
    ref_params = list(sig_ref.parameters.keys())
    assert ref_params == [
        "reference_primitive",
        "content_features",
        "content_lengths",
        "output_lengths",
    ]
    assert "pi_n" not in ref_params
    assert "oracle" not in ref_params
    assert "target_tokens" not in ref_params
    assert "targets" not in ref_params

    # 2. clamped_forward_with_stages signature
    sig_clamp = inspect.signature(rec004u.clamped_forward_with_stages)
    clamp_params = list(sig_clamp.parameters.keys())
    assert clamp_params == [
        "primitive",
        "content_features",
        "content_lengths",
        "output_lengths",
        "attn_probs",
    ]
    assert "pi_n" not in clamp_params
    assert "oracle" not in clamp_params
    assert "target_tokens" not in clamp_params
    assert "targets" not in clamp_params


def test_fresh_dataset_generation_disjointness() -> None:
    """Verifies fresh dataset generation and strict disjointness against protected digests."""
    seed = rec004u.REC004U_SEED
    # Create dummy protected digests to test substitution and collision avoidance
    dummy_protected: set[str] = set()
    examples, detail = rec004u.build_attention_clamp_causal_probe_v1(seed, dummy_protected, n=16)

    assert len(examples) == 16
    assert detail["target_length"] == 10
    assert detail["development_exposed"] is True
    assert detail["sealed_or_rg3_query"] is False

    for ex in examples:
        assert len(ex.input_tokens) == 10
        assert len(ex.target_tokens) == 10

    digests = rec004u._digest_examples(examples)
    assert len(digests) == 16

    # Test disjointness enforcement: pass the first 8 digests as protected
    protected_half = set(list(digests)[:8])
    fresh_exs, fresh_detail = rec004u.build_attention_clamp_causal_probe_v1(
        seed, protected_half, n=16
    )
    fresh_digests = rec004u._digest_examples(fresh_exs)
    assert len(fresh_digests.intersection(protected_half)) == 0


def test_initial_parity_gate_and_fused_qkv_freeze() -> None:
    """Tests Initial Parity Gate and fail-closed Q/K freeze on step 7500 state."""
    seed = rec004u.REC004U_SEED
    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    device = core.device
    start_step = rec004u.REC004U_START_STEP
    init_id = rec004u.REC004U_DECISIVE_INIT

    start_ts_path = rec004t._training_state_path(init_id, start_step)
    if not start_ts_path.is_file():
        return

    start_ts = torch.load(start_ts_path, map_location="cpu", weights_only=False)
    reference = mpbr._new_arm_primitive(core, rec004u.REC004U_ARM)
    current = mpbr._new_arm_primitive(core, rec004u.REC004U_ARM)

    assert isinstance(reference, CrossPositionLengthBiasPrimitive)
    assert isinstance(current, CrossPositionLengthBiasPrimitive)

    reference.to(device)
    current.to(device)

    reference.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )
    current.load_state_dict(
        {k: v.to(device) for k, v in start_ts["primitive_state_dict"].items()}, strict=True
    )
    reference.eval()
    current.eval()

    # Generate 16 sample examples
    sample_exs, _ = rec004u.build_attention_clamp_causal_probe_v1(seed, set(), n=16)

    # Verify Parity
    parity = rec004u.verify_attention_clamp_initial_parity(core, current, reference, sample_exs)
    assert parity["status"] == "PASS"
    assert parity["attn_passed"] is True
    assert parity["logits_passed"] is True
    assert parity["preds_exact_match"] is True

    # Test Fused Q/K freeze mechanics over 1 simulated step
    current.train()
    current.position_bias_hidden.weight.requires_grad_(False)
    if current.position_bias_hidden.bias is not None:
        current.position_bias_hidden.bias.requires_grad_(False)
    current.position_bias_out.weight.requires_grad_(False)

    embed_dim = current.d_operator
    frozen_qkv_w = current.cross_attn.in_proj_weight.data[: 2 * embed_dim].clone()
    frozen_qkv_b = (
        current.cross_attn.in_proj_bias.data[: 2 * embed_dim].clone()
        if current.cross_attn.in_proj_bias is not None
        else None
    )

    optimizer = torch.optim.AdamW(current.parameters(), lr=1e-3, weight_decay=1e-2)
    optimizer.load_state_dict(start_ts["optimizer_state_dict"])

    frozen_opt_w = optimizer.state[current.cross_attn.in_proj_weight]["exp_avg"][
        : 2 * embed_dim
    ].clone()
    frozen_opt_sq_w = optimizer.state[current.cross_attn.in_proj_weight]["exp_avg_sq"][
        : 2 * embed_dim
    ].clone()

    # Forward
    batch_input = collate_content_only_batch(sample_exs, core.tokens, device=device)
    with torch.no_grad():
        h = core.model.encode(batch_input)[:, 1:11, :]
        c_lens = [len(ex.input_tokens) for ex in sample_exs]
        o_lens = [10] * len(sample_exs)
        a_ref = rec004u.extract_reference_attention(reference, h, c_lens, o_lens)

    optimizer.zero_grad(set_to_none=True)
    res = rec004u.clamped_forward_with_stages(current, h, c_lens, o_lens, a_ref)
    loss = res["final_token_logits"].sum()
    loss.backward()

    # In proj weight grad on Q/K rows should be zero
    in_proj_grad = current.cross_attn.in_proj_weight.grad
    assert in_proj_grad is not None
    assert torch.all(in_proj_grad[: 2 * embed_dim] == 0.0)

    optimizer.step()

    # Restore
    current.cross_attn.in_proj_weight.data[: 2 * embed_dim].copy_(frozen_qkv_w)
    if current.cross_attn.in_proj_bias is not None and frozen_qkv_b is not None:
        current.cross_attn.in_proj_bias.data[: 2 * embed_dim].copy_(frozen_qkv_b)

    optimizer.state[current.cross_attn.in_proj_weight]["exp_avg"][: 2 * embed_dim].copy_(
        frozen_opt_w
    )
    optimizer.state[current.cross_attn.in_proj_weight]["exp_avg_sq"][: 2 * embed_dim].copy_(
        frozen_opt_sq_w
    )

    # Invariance verification
    assert torch.equal(current.cross_attn.in_proj_weight.data[: 2 * embed_dim], frozen_qkv_w)
    assert torch.equal(
        optimizer.state[current.cross_attn.in_proj_weight]["exp_avg"][: 2 * embed_dim],
        frozen_opt_w,
    )
    assert torch.equal(
        optimizer.state[current.cross_attn.in_proj_weight]["exp_avg_sq"][: 2 * embed_dim],
        frozen_opt_sq_w,
    )
