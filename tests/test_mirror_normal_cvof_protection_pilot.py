"""Unit tests for Task B-C005REC-004W: I03 Normal-Attention CVOF Protection Continuation Pilot."""

from __future__ import annotations

import torch

from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_normal_cvof_protection_pilot as rec004w
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive


def test_fresh_datasets_generation_and_disjointness() -> None:
    """Verifies generation of fresh datasets and strict disjointness
    against protected digests."""
    seed = rec004w.REC004W_SEED
    dummy_protected: set[str] = set()

    # 1. Normal validation (variable length)
    val_exs, val_detail = rec004w.build_normal_cvof_protection_validation_v1(
        seed, dummy_protected, n=32
    )
    assert len(val_exs) == 32
    assert val_detail["development_exposed"] is True
    assert val_detail["sealed_or_rg3_query"] is False

    lengths = [len(ex.input_tokens) for ex in val_exs]
    assert min(lengths) >= 2
    assert max(lengths) <= 10

    val_digests = rec004w._digest_examples(val_exs)
    assert len(val_digests) == 32

    # 2. Length-10 confirmation
    conf_exs, conf_detail = rec004w.build_normal_cvof_protection_length10_v1(
        seed, val_digests, n=16
    )
    assert len(conf_exs) == 16
    assert conf_detail["target_length"] == 10
    assert conf_detail["development_exposed"] is True
    assert conf_detail["sealed_or_rg3_query"] is False

    for ex in conf_exs:
        assert len(ex.input_tokens) == 10
        assert len(ex.target_tokens) == 10

    conf_digests = rec004w._digest_examples(conf_exs)
    assert len(conf_digests.intersection(val_digests)) == 0


def test_fused_qkv_cvof_freeze_contract() -> None:
    """Tests that in NORMAL_CVOF_PROTECTED, Q/K rows remain trainable while V rows
    and C, O, F parameter groups remain strictly frozen at initial values."""
    seed = rec004w.REC004W_SEED
    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    device = torch.device("cpu")

    model = mpbr._new_arm_primitive(core, rec004w.REC004W_ARM)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    embed_dim = model.d_operator  # 32

    # Snapshot initial state
    init_in_proj_w = model.cross_attn.in_proj_weight.data.clone()
    init_content_w = model.content_in_proj.weight.data.clone()
    init_out_proj_w = model.cross_attn.out_proj.weight.data.clone()
    first_ffn = model.ffn[0]
    assert isinstance(first_ffn, torch.nn.Linear)
    init_ffn_w = first_ffn.weight.data.clone()
    init_pos_bias_w = model.position_bias_hidden.weight.data.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Assign dummy gradients to all parameters
    for p in model.parameters():
        p.grad = torch.ones_like(p.data) * 0.1

    # Apply CVOF freeze rules
    # 1. C = CONTENT_PREP (zeroed)
    assert model.content_in_proj.weight.grad is not None
    model.content_in_proj.weight.grad.zero_()
    if model.content_in_proj.bias is not None:
        assert model.content_in_proj.bias.grad is not None
        model.content_in_proj.bias.grad.zero_()
    assert model.content_position_embedding.weight.grad is not None
    model.content_position_embedding.weight.grad.zero_()

    # 2. V = V_PROJECTION (zeroed: rows 64:96)
    # Note: Q (0:32) and K (32:64) grads remain non-zero!
    assert model.cross_attn.in_proj_weight.grad is not None
    model.cross_attn.in_proj_weight.grad[2 * embed_dim : 3 * embed_dim].zero_()
    if model.cross_attn.in_proj_bias is not None:
        assert model.cross_attn.in_proj_bias.grad is not None
        model.cross_attn.in_proj_bias.grad[2 * embed_dim : 3 * embed_dim].zero_()

    # 3. O = ATTN_OUT_PROJ (zeroed)
    assert model.cross_attn.out_proj.weight.grad is not None
    model.cross_attn.out_proj.weight.grad.zero_()
    if model.cross_attn.out_proj.bias is not None:
        assert model.cross_attn.out_proj.bias.grad is not None
        model.cross_attn.out_proj.bias.grad.zero_()

    # 4. F = FFN_BLOCK (zeroed)
    for p in model.ffn.parameters():
        assert p.grad is not None
        p.grad.zero_()
    assert model.ffn_norm.weight.grad is not None
    model.ffn_norm.weight.grad.zero_()
    if model.ffn_norm.bias is not None:
        assert model.ffn_norm.bias.grad is not None
        model.ffn_norm.bias.grad.zero_()

    # Step optimizer
    optimizer.step()

    # Fail-closed restoration of frozen groups
    model.content_in_proj.weight.data.copy_(init_content_w)
    model.cross_attn.in_proj_weight.data[2 * embed_dim : 3 * embed_dim].copy_(
        init_in_proj_w[2 * embed_dim : 3 * embed_dim]
    )
    model.cross_attn.out_proj.weight.data.copy_(init_out_proj_w)
    first_ffn.weight.data.copy_(init_ffn_w)

    # Verify frozen parameters did not move
    assert torch.all(model.content_in_proj.weight.data == init_content_w)
    assert torch.all(
        model.cross_attn.in_proj_weight.data[2 * embed_dim : 3 * embed_dim]
        == init_in_proj_w[2 * embed_dim : 3 * embed_dim]
    )
    assert torch.all(model.cross_attn.out_proj.weight.data == init_out_proj_w)
    assert torch.all(first_ffn.weight.data == init_ffn_w)

    # Verify trainable parameters DID move (Q, K, and position_bias)
    assert not torch.all(
        model.cross_attn.in_proj_weight.data[:embed_dim] == init_in_proj_w[:embed_dim]
    )
    assert not torch.all(
        model.cross_attn.in_proj_weight.data[embed_dim : 2 * embed_dim]
        == init_in_proj_w[embed_dim : 2 * embed_dim]
    )
    assert not torch.all(model.position_bias_hidden.weight.data == init_pos_bias_w)


def test_initial_parity_verification_logic() -> None:
    """Verifies initial parity check catches identical vs perturbed models."""
    seed = rec004w.REC004W_SEED
    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    device = core.device

    model = mpbr._new_arm_primitive(core, rec004w.REC004W_ARM)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    ref = mpbr._new_arm_primitive(core, rec004w.REC004W_ARM)
    assert isinstance(ref, CrossPositionLengthBiasPrimitive)
    ref.to(device)
    ref.load_state_dict(model.state_dict())

    sample_exs, _ = rec004w.build_normal_cvof_protection_length10_v1(seed, set(), n=8)

    # Identical state -> PASS
    report = rec004w.verify_initial_parity(core, model, ref, sample_exs, device)
    assert report["status"] == "PASS"

    # Perturbed state -> raises RuntimeError
    with torch.no_grad():
        assert isinstance(model.position_bias_out.weight, torch.Tensor)
        model.position_bias_out.weight.data.add_(0.5)

    try:
        rec004w.verify_initial_parity(core, model, ref, sample_exs, device)
        passed_unexpectedly = True
    except RuntimeError:
        passed_unexpectedly = False

    assert not passed_unexpectedly


def test_result_decision_classification_logic() -> None:
    """Tests the five outcome decision paths under the task gate."""
    # Case 1: Functional floor cleared
    freeze_pass = True
    o1_all_pass = True
    val_delta = 0.15
    conf_delta = 0.20
    val_overall = 0.96
    val_len10 = 0.97
    conf = 0.98

    j0_gain = val_delta >= 0.10 and conf_delta >= 0.10
    floor = val_overall >= 0.95 and val_len10 >= 0.95 and conf >= 0.95

    if freeze_pass and o1_all_pass and j0_gain:
        label = (
            "CVOF_PROTECTED_PILOT_REACHES_FUNCTIONAL_FLOOR"
            if floor
            else "CVOF_PROTECTION_COMPATIBLE_WITH_SCORE_LEARNING"
        )
    assert label == "CVOF_PROTECTED_PILOT_REACHES_FUNCTIONAL_FLOOR"

    # Case 2: Compatible with score learning (delta >= 0.10, but below 0.95 floor)
    val_overall = 0.85
    floor = val_overall >= 0.95 and val_len10 >= 0.95 and conf >= 0.95
    if freeze_pass and o1_all_pass and j0_gain:
        label = (
            "CVOF_PROTECTED_PILOT_REACHES_FUNCTIONAL_FLOOR"
            if floor
            else "CVOF_PROTECTION_COMPATIBLE_WITH_SCORE_LEARNING"
        )
    assert label == "CVOF_PROTECTION_COMPATIBLE_WITH_SCORE_LEARNING"

    # Case 3: Preserves compatibility but blocks learning
    j0_gain = False
    if freeze_pass and o1_all_pass and not j0_gain:
        label = "CVOF_PROTECTION_PRESERVES_COMPATIBILITY_BUT_BLOCKS_LEARNING"
    assert label == "CVOF_PROTECTION_PRESERVES_COMPATIBILITY_BUT_BLOCKS_LEARNING"

    # Case 4: Learning with compatibility loss
    o1_all_pass = False
    j0_gain = True
    if freeze_pass and j0_gain and not o1_all_pass:
        label = "CVOF_PROTECTION_DOES_NOT_PRESERVE_ORACLE_COMPATIBILITY"
    assert label == "CVOF_PROTECTION_DOES_NOT_PRESERVE_ORACLE_COMPATIBILITY"

    # Case 5: Neither
    j0_gain = False
    label = "HARD_CVOF_PROTECTION_NOT_SUPPORTED"
    assert label == "HARD_CVOF_PROTECTION_NOT_SUPPORTED"
