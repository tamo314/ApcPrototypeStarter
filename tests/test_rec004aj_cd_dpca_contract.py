"""Tests for B-C005REC-004AJ: Minimal Routing Contract Implementation
& Untrained Structural Validation.

Validates the ContentDecoupledDiscretePositionalCrossAttention (CD-DPCA) primitive contract
defined in ADR-0134 and implemented in ADR-0135.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from apc.evaluation.mirror_position_initialization_diagnostic import (
    mirror_halves_position_map,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CDDPCAPrimitive,
    CDDPCAPrimitiveConfig,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
)

ROOT = Path(__file__).resolve().parent.parent
REC004AJ_DIR = ROOT / "runs/phase_b_restart/rec004aj/run_001"


def test_rec004aj_artifacts_exist_and_pass() -> None:
    """Verify that REC-004AJ artifacts exist and record PASS with all blocks preserved."""
    assert REC004AJ_DIR.is_dir()
    summary_path = REC004AJ_DIR / "summary.json"
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task"] == "B-C005REC-004AJ"
    assert summary["execution_status"] == "PASS"
    assert summary["decision"] == "MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED"
    assert summary["scorer_content_invariance"] is True
    assert summary["discrete_addressability"] is True
    assert summary["permutation_representability"] is True
    assert summary["runtime_information_boundary"] is True
    assert summary["padding_masking"] is True
    assert summary["gradient_reachability"] is True
    assert summary["downstream_compatibility"] is True
    assert summary["optimizer_updates"] == 0
    assert summary["new_parameters"] == 0
    assert summary["candidate_selected"] is None
    assert summary["bundle_write"] is False
    assert summary["rg3"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["g1"] == "NOT_CLEARED"
    assert summary["g4"] == "NOT_CLEARED"


def test_cd_dpca_scorer_content_invariance() -> None:
    """Criterion 1: Verify that routing score generation is strictly invariant to token content."""
    torch.manual_seed(123)
    config = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    primitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(0, config)
    primitive.eval()

    batch = 3
    lmax = 10
    content_lengths = [10, 8, 6]
    output_lengths = [10, 8, 6]

    # Two entirely different token content inputs
    h1 = torch.randn(batch, lmax, 192)
    h2 = torch.randn(batch, lmax, 192) * 20.0 - 50.0

    with torch.no_grad():
        scores_1 = primitive.compute_routing_scores(content_lengths, output_lengths, lmax=lmax)
        scores_2 = primitive.compute_routing_scores(content_lengths, output_lengths, lmax=lmax)

        weights_1 = primitive.compute_attention_weights(content_lengths, output_lengths, lmax=lmax)
        weights_2 = primitive.compute_attention_weights(content_lengths, output_lengths, lmax=lmax)

        logits_1, fwd_w1 = primitive.forward(
            h1, content_lengths, output_lengths, return_attention=True
        )
        logits_2, fwd_w2 = primitive.forward(
            h2, content_lengths, output_lengths, return_attention=True
        )

    # Pre-softmax scores and attention weights are bitwise/numerically identical
    assert (scores_1 == scores_2).all()
    assert torch.allclose(weights_1, weights_2, atol=1e-7)
    assert torch.allclose(fwd_w1, fwd_w2, atol=1e-7)
    assert torch.allclose(fwd_w1, weights_1, atol=1e-6)

    # Downstream logits MUST differ because content propagates via value path
    assert (logits_1 - logits_2).abs().max() > 0.1


def test_cd_dpca_discrete_addressability_and_finite_boundary() -> None:
    """Criterion 2 & Boundary: Discrete integer positions/lengths addressable; boundary enforced."""
    config = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        max_sequence_length=32,
    )
    primitive = CDDPCAPrimitive(1, config)

    # Distinct query and key position embeddings
    for i in range(16):
        for j in range(i + 1, 16):
            qi = primitive.query_position_embedding.weight[i]
            qj = primitive.query_position_embedding.weight[j]
            assert not torch.equal(qi, qj)

            ki = primitive.key_position_embedding.weight[i]
            kj = primitive.key_position_embedding.weight[j]
            assert not torch.equal(ki, kj)

    # Addressable across all Phase B legal lengths [2, 16]
    for L in range(2, 17):
        scores = primitive.compute_routing_scores([L], [L], lmax=L)
        assert scores.shape == (1, config.n_head, L, L)

    # Finite domain boundary: exceeding max_sequence_length raises ValueError
    try:
        primitive.compute_routing_scores([33], [33], lmax=33)
        msg = "Expected ValueError for length exceeding max_sequence_length"
        raise AssertionError(msg)
    except ValueError as e:
        assert "exceeds configured max_sequence_length" in str(e)


def test_cd_dpca_permutation_representability() -> None:
    """Criterion 3: Length-conditioned scoring represents full permutation matrices
    within dimension limits.
    """
    d_op = 32
    n_head = 4
    head_dim = d_op // n_head

    # Test all legal MIRROR_HALVES lengths L in [2, 16]
    for L in range(2, 17):
        pi = mirror_halves_position_map(L)
        scale = 8.0 * (head_dim**0.5)

        # Construct orthonormal embeddings in R^d_op
        q_pos = torch.zeros(L, d_op)
        k_pos = torch.zeros(L, d_op)
        for i in range(L):
            q_pos[i, pi[i]] = scale
        for j in range(L):
            k_pos[j, j] = 1.0

        scores = torch.matmul(q_pos, k_pos.T) / (head_dim**0.5)
        probs = F.softmax(scores, dim=-1)

        # For every output position i, probability on pi[i] exceeds 0.99
        for i in range(L):
            correct_prob = probs[i, pi[i]].item()
            assert correct_prob > 0.99, f"L={L}, pos={i}: prob={correct_prob}"

            # Margin over runner up > 6.0
            correct_s = scores[i, pi[i]].item()
            runner_up_s = max(scores[i, j].item() for j in range(L) if j != pi[i])
            assert (correct_s - runner_up_s) > 6.0


def test_cd_dpca_runtime_information_boundary() -> None:
    """Criterion 4: Zero target or oracle inputs reach runtime routing."""
    sig = inspect.signature(ContentDecoupledDiscretePositionalCrossAttentionPrimitive.forward)
    param_names = [p for p in sig.parameters.keys() if p != "self"]
    assert param_names == [
        "content_features",
        "content_lengths",
        "output_lengths",
        "argument_values",
        "return_attention",
    ]

    # No oracle or target in signatures
    for method in [
        ContentDecoupledDiscretePositionalCrossAttentionPrimitive.compute_routing_representations,
        ContentDecoupledDiscretePositionalCrossAttentionPrimitive.compute_routing_scores,
        ContentDecoupledDiscretePositionalCrossAttentionPrimitive.compute_attention_weights,
    ]:
        m_params = inspect.signature(method).parameters.keys()
        for p in m_params:
            assert "target" not in p.lower()
            assert "oracle" not in p.lower()
            assert "label" not in p.lower()


def test_cd_dpca_padding_masking() -> None:
    """Criterion 5: Keys at j >= L are strictly masked with -inf and 0.0 attention weight."""
    config = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        max_sequence_length=32,
    )
    primitive = CDDPCAPrimitive(2, config)
    primitive.eval()

    content_lengths = [5]
    output_lengths = [5]
    lmax = 8

    with torch.no_grad():
        scores = primitive.compute_routing_scores(content_lengths, output_lengths, lmax=lmax)
        weights = primitive.compute_attention_weights(content_lengths, output_lengths, lmax=lmax)

    # Scores at padded positions j >= 5 must be -inf
    padded_scores = scores[0, :, :, 5:]
    assert torch.isneginf(padded_scores).all()

    # Attention weights at padded positions must be exactly 0.0
    padded_weights = weights[0, :, :, 5:]
    assert (padded_weights == 0.0).all()

    # Valid positions sum to 1.0
    valid_sums = weights[0, :, :, :5].sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-6)


def test_cd_dpca_untrained_gradient_reachability() -> None:
    """Criterion 6: All routing-parameter gradients are non-null under untrained forward/loss."""
    torch.manual_seed(99)
    config = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
    )
    primitive = CDDPCAPrimitive(3, config)
    primitive.train()

    content = torch.randn(2, 8, 192, requires_grad=True)
    logits = primitive(content, [8, 6], [8, 8])
    loss = logits.sum()
    loss.backward()

    # Routing embedding gradients
    assert primitive.query_position_embedding.weight.grad is not None
    assert primitive.query_position_embedding.weight.grad.norm().item() > 0.0

    assert primitive.key_position_embedding.weight.grad is not None
    assert primitive.key_position_embedding.weight.grad.norm().item() > 0.0

    assert primitive.length_embedding.weight.grad is not None
    assert primitive.length_embedding.weight.grad.norm().item() > 0.0

    # Cross-attention projection gradients
    in_proj_grad = primitive.cross_attn.in_proj_weight.grad
    assert in_proj_grad is not None
    wq_g, wk_g, wv_g = in_proj_grad.chunk(3, dim=0)
    assert wq_g.norm().item() > 0.0
    assert wk_g.norm().item() > 0.0
    assert wv_g.norm().item() > 0.0

    assert primitive.cross_attn.out_proj.weight.grad is not None
    assert primitive.cross_attn.out_proj.weight.grad.norm().item() > 0.0


def test_cd_dpca_downstream_compatibility_and_bank_integration() -> None:
    """Criterion 7: Downstream shapes, LayerNorm, FFN, readout, PrimitiveBank, state_dict."""
    config = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )

    # Bank registration and factory
    bank = PrimitiveBank()
    p = bank.new_content_decoupled_primitive(config)
    assert len(bank) == 1
    assert bank.get(p.primitive_id) is p
    assert isinstance(p, ContentDecoupledDiscretePositionalCrossAttentionPrimitive)

    # Bank alias factory
    p2 = bank.new_cd_dpca_primitive(config)
    assert len(bank) == 2
    assert bank.get(p2.primitive_id) is p2

    # Forward call bookkeeping
    p.reset_forward_call_count()
    assert p.forward_call_count == 0
    content = torch.randn(2, 6, 192)
    logits = p(content, [6, 4], [6, 4])
    assert p.forward_call_count == 1
    assert logits.shape == (2, 6, 10)

    # Property alias answer_query_embedding works
    assert p.answer_query_embedding is p.query_position_embedding

    # state_dict does NOT contain duplicated keys from property alias
    sd = p.state_dict()
    assert "query_position_embedding.weight" in sd
    assert "answer_query_embedding.weight" not in sd
    assert "key_position_embedding.weight" in sd
    assert "length_embedding.weight" in sd
    assert "content_in_proj.weight" in sd
    assert "cross_attn.in_proj_weight" in sd
    assert "attn_norm.weight" in sd
    assert "ffn.0.weight" in sd
    assert "readout.weight" in sd


def test_cd_dpca_generic_relation_conditioning() -> None:
    """Verify generic argument conditioning for parameterized relations (e.g. SHIFT)
    and parameter-free.
    """
    # Parameter-free: MIRROR_HALVES
    cfg_mirror = CDDPCAPrimitiveConfig(operation="MIRROR_HALVES", d_model=192, d_operator=32)
    p_mirror = CDDPCAPrimitive(10, cfg_mirror)
    assert p_mirror.arg_encoder is None
    assert p_mirror.arg_proj is None

    out_mirror = p_mirror(torch.randn(2, 4, 192), [4, 4], [4, 4], argument_values=None)
    assert out_mirror.shape == (2, 4, 10)

    # Parameterized: SHIFT with argument_values
    cfg_shift = CDDPCAPrimitiveConfig(operation="SHIFT", d_model=192, d_operator=32)
    p_shift = CDDPCAPrimitive(11, cfg_shift)
    assert p_shift.arg_encoder is not None
    assert p_shift.arg_proj is not None

    out_shift = p_shift(torch.randn(2, 4, 192), [4, 4], [4, 4], argument_values=[1, 2])
    assert out_shift.shape == (2, 4, 10)
