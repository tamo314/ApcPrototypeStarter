"""Unit and integration tests for Task A1-B007X-002 Novel-Task and Capacity-Ladder Harness."""

from __future__ import annotations

import torch

from apc.environments.operations import (
    DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES,
    get_operation,
)
from apc.evaluation.discovery_capacity_harness import (
    TIER_SPECS,
    CapacityTier,
    DiscoveryCapacityHarnessConfig,
    audit_capacity_ladder,
    build_tier_primitive,
    generate_novel_examples,
    run_discovery_capacity_harness,
)


def test_novel_operations_registration() -> None:
    """Test that all discovery-compression novel operations are registered and deterministic."""
    assert len(DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES) >= 2
    assert "SWAP_PAIRS" in DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    assert "INVERT_HALF" in DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES
    assert "ROTATE_TRIPLETS" in DISCOVERY_COMPRESSION_NOVEL_OPERATION_NAMES

    # Test ROTATE_TRIPLETS specifically
    op = get_operation("ROTATE_TRIPLETS")
    assert op.min_input_length == 3
    seq = (1, 2, 3, 4, 5, 6, 7, 8)
    out = op.apply(seq, 10, {})
    # (1, 2, 3) -> (2, 3, 1), (4, 5, 6) -> (5, 6, 4), (7, 8) remains (7, 8)
    assert out == (2, 3, 1, 5, 6, 4, 7, 8)

    # Determinism
    out2 = op.apply(seq, 10, {})
    assert out == out2


def test_capacity_ladder_parameter_counts() -> None:
    """Test exact parameter counts and topology specs across T0-T3 tiers."""
    results = audit_capacity_ladder(tuple(TIER_SPECS.keys()))

    assert len(results) == 4
    # T0 Compact
    t0 = results[CapacityTier.T0_COMPACT.value]
    assert t0.actual_parameters == 17098
    assert t0.counts_match
    assert t0.actual_parameters <= 25000

    # T1 Medium
    t1 = results[CapacityTier.T1_MEDIUM.value]
    assert t1.actual_parameters == 67082
    assert t1.counts_match

    # T2 Overcomplete
    t2 = results[CapacityTier.T2_OVERCOMPLETE.value]
    assert t2.actual_parameters == 137482
    assert t2.counts_match
    # Overcomplete ratio >= 4.0x
    assert t2.ratio_to_compact >= 4.0
    assert abs(t2.ratio_to_compact - (137482 / 17098)) < 1e-4

    # T3 Extra-Large
    t3 = results[CapacityTier.T3_EXTRA_LARGE.value]
    assert t3.actual_parameters == 232458
    assert t3.counts_match
    assert t3.ratio_to_compact >= 10.0


def test_data_generation_splits_and_determinism() -> None:
    """Test deterministic generation and clean split separation."""
    train_a = generate_novel_examples(0, 20, operation="ROTATE_TRIPLETS", split="train")
    train_b = generate_novel_examples(0, 20, operation="ROTATE_TRIPLETS", split="train")
    test_a = generate_novel_examples(0, 20, operation="ROTATE_TRIPLETS", split="test")

    # Deterministic reproduction
    assert [ex.input_tokens for ex in train_a] == [ex.input_tokens for ex in train_b]
    assert [ex.target_tokens for ex in train_a] == [ex.target_tokens for ex in train_b]

    # Clean split difference
    assert [ex.input_tokens for ex in train_a] != [ex.input_tokens for ex in test_a]


def test_forward_pass_across_all_tiers() -> None:
    """Test that all tiers execute forward and backward passes correctly."""
    batch_size = 4
    seq_len = 8
    d_model = 192
    vocab_size = 10

    content_features = torch.randn(batch_size, seq_len, d_model)
    content_lengths = [seq_len] * batch_size
    output_lengths = [seq_len] * batch_size

    for tier in TIER_SPECS.keys():
        prim = build_tier_primitive(tier, operation="SWAP_PAIRS", vocab_size=vocab_size)
        prim.train()
        logits = prim(
            content_features=content_features,
            content_lengths=content_lengths,
            output_lengths=output_lengths,
        )
        assert logits.shape == (batch_size, seq_len, vocab_size)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in prim.parameters() if p.grad is not None)
        assert grad_count > 0


def test_smoke_harness_execution(tmp_path) -> None:
    """Fast end-to-end smoke test of the harness on CPU."""
    config = DiscoveryCapacityHarnessConfig(
        seed=42,
        device="cpu",
        novel_operations=("SWAP_PAIRS", "INVERT_HALF", "ROTATE_TRIPLETS"),
        tiers_to_audit=("T0_compact", "T2_overcomplete"),
        num_eval_examples=10,
        num_trial_train_examples=20,
        trial_train_steps=5,
        batch_size=4,
        core_train_steps=10,
        bank_train_steps=10,
    )

    summary = run_discovery_capacity_harness(config, output_dir=tmp_path)
    assert summary.all_novel_verified
    assert summary.all_tier_counts_verified
    assert summary.overcomplete_ratio_verified
    assert summary.no_oracle_leakage_verified
    assert summary.overall_passed
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.json").exists()
