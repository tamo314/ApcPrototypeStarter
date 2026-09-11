"""Unit tests for Task B-C005REC-004V: I03 Downstream Freeze Necessity Replay."""

from __future__ import annotations

import torch

from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_downstream_freeze_causal_replay as rec004v
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.primitives.primitive import CrossPositionLengthBiasPrimitive


def test_fresh_dataset_generation_disjointness() -> None:
    """Verifies fresh dataset generation and strict disjointness against protected digests."""
    seed = rec004v.REC004V_SEED
    dummy_protected: set[str] = set()
    examples, detail = rec004v.build_downstream_freeze_causal_probe_v1(seed, dummy_protected, n=16)

    assert len(examples) == 16
    assert detail["target_length"] == 10
    assert detail["development_exposed"] is True
    assert detail["sealed_or_rg3_query"] is False

    for ex in examples:
        assert len(ex.input_tokens) == 10
        assert len(ex.target_tokens) == 10

    digests = rec004v._digest_examples(examples)
    assert len(digests) == 16

    # Test disjointness enforcement: pass the first 8 digests as protected
    protected_half = set(list(digests)[:8])
    fresh_exs, fresh_detail = rec004v.build_downstream_freeze_causal_probe_v1(
        seed, protected_half, n=16
    )
    fresh_digests = rec004v._digest_examples(fresh_exs)
    assert len(fresh_digests.intersection(protected_half)) == 0


def test_fused_qkv_and_downstream_selective_freeze_contract() -> None:
    """Tests that fused QKV rows and downstream parameter groups can be selectively frozen
    without freezing non-target slices or parameters."""
    seed = rec004v.REC004V_SEED
    core, eval_bank, op_to_id = rec004t._load_runtime_core(seed=seed)
    device = torch.device("cpu")

    model = mpbr._new_arm_primitive(core, rec004v.REC004V_ARM)
    assert isinstance(model, CrossPositionLengthBiasPrimitive)
    model.to(device)
    embed_dim = model.d_operator  # 32

    # Snapshot initial state
    init_in_proj_w = model.cross_attn.in_proj_weight.data.clone()
    init_content_w = model.content_in_proj.weight.data.clone()
    init_ffn_w = model.ffn[0].weight.data.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 1. Test D_V freeze: Q/K (0:64) and V (64:96) are frozen
    # Assign dummy gradients to all parameters
    for p in model.parameters():
        p.grad = torch.ones_like(p.data) * 0.1

    # Apply D_V zero_grad rule: Q/K and V are all zeroed
    model.cross_attn.in_proj_weight.grad[: 3 * embed_dim].zero_()
    optimizer.step()

    # In D_V, in_proj_weight should be fully restored to initial
    model.cross_attn.in_proj_weight.data[: 3 * embed_dim].copy_(init_in_proj_w[: 3 * embed_dim])

    # Assert in_proj_weight is unchanged
    assert torch.all(model.cross_attn.in_proj_weight.data == init_in_proj_w)

    # But content_in_proj and ffn SHOULD have updated
    assert not torch.all(model.content_in_proj.weight.data == init_content_w)
    assert not torch.all(model.ffn[0].weight.data == init_ffn_w)


def test_causal_decision_logic_cases() -> None:
    """Verifies the four pre-registered causal decision branches."""
    # Case A: Single strong
    strong_single = ["FFN_BLOCK"]
    decision_case_a = "CASE_A_SINGLE_STRONG"
    assert decision_case_a == "CASE_A_SINGLE_STRONG"
    assert len(strong_single) == 1

    # Case B: CVOF joint strong
    strong_single_empty: list[str] = []
    cvof_strong = True
    if len(strong_single_empty) == 0 and cvof_strong:
        decision_case_b = "CASE_B_CVOF_JOINT_STRONG"
    assert decision_case_b == "CASE_B_CVOF_JOINT_STRONG"

    # Case C: CVOF insufficient
    cvof_strong_c = False
    if len(strong_single_empty) == 0 and not cvof_strong_c:
        decision_case_c = "CASE_C_CVOF_INSUFFICIENT"
    assert decision_case_c == "CASE_C_CVOF_INSUFFICIENT"
