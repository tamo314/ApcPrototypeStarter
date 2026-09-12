"""Focused regression and contract tests for Task B-C005REC-004AQ:
CD-DPCA Sequence-Distinctness Warm-Start Single-Recipe Causal Pilot (ADR-0141).
"""

from __future__ import annotations

from pathlib import Path

import torch

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    REC004AK_BANK_MANIFEST_HASH,
    REC004AK_BANK_STATE_CANONICAL_HASH,
    REC004AK_BANK_STATE_RAW_HASH,
    REC004AK_PRIMITIVE_CONFIG_HASH,
    REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
    REC004AK_PRIMITIVE_STATE_RAW_HASH,
    REC004AL_ARCHITECTURE_SIGNATURE,
    REC004AL_INIT_ID,
    REC004AL_PARENT_BUNDLE_ID,
    REC004AL_PARENT_CORE_HASH,
    REC004AL_TARGET_OPERATION,
    run_information_boundary_audit,
)
from apc.evaluation.mirror_cd_dpca_warm_start_pilot import (
    REC004AQ_BASELINE_TASK_ID,
    REC004AQ_TASK_ID,
    REC004AQ_WARM_START_STEPS,
    MirrorCDDPCAWarmStartPilotConfig,
    build_warm_start_protocol,
    compute_position4_routing_metrics,
    compute_strata_gradient_alignment_at_checkpoint,
    generate_step_training_examples_warm_start,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
)

ROOT = Path(__file__).resolve().parent.parent
REC004AK_DIR = ROOT / "runs/phase_b_restart/rec004ak/run_001"
REC004AL_DIR = ROOT / "runs/phase_b_restart/rec004al/run_001"


def test_rec004aq_protocol_and_source_hashes() -> None:
    """Verifies protocol builder enforces parent bundle, source hashes, and intervention config."""
    config = MirrorCDDPCAWarmStartPilotConfig(
        rec004ak_dir=REC004AK_DIR,
        rec004al_dir=REC004AL_DIR,
    )
    parent_manifest, _ = ibc._load_parent_manifest()
    protocol = build_warm_start_protocol(config, parent_manifest)

    assert protocol["task_id"] == REC004AQ_TASK_ID
    assert protocol["baseline_task_id"] == REC004AQ_BASELINE_TASK_ID
    assert protocol["parent_bundle_id"] == REC004AL_PARENT_BUNDLE_ID
    assert protocol["parent_core_canonical_state_hash"] == REC004AL_PARENT_CORE_HASH
    assert protocol["init_id"] == REC004AL_INIT_ID
    assert protocol["target_operation"] == REC004AL_TARGET_OPERATION
    assert protocol["architecture_signature"] == REC004AL_ARCHITECTURE_SIGNATURE

    intervention = protocol["causal_intervention"]
    assert intervention["warm_start_steps"] == REC004AQ_WARM_START_STEPS
    assert intervention["boundary_pre_fixed"] is True
    assert intervention["sweep_authorized"] is False
    assert intervention["length_position_selectivity_prohibited"] is True

    hashes = protocol["source_hashes"]
    assert hashes["bank_manifest_sha256"] == REC004AK_BANK_MANIFEST_HASH
    assert hashes["primitive_config_sha256"] == REC004AK_PRIMITIVE_CONFIG_HASH
    assert hashes["bank_state_raw_sha256"] == REC004AK_BANK_STATE_RAW_HASH
    assert hashes["bank_state_canonical_hash"] == REC004AK_BANK_STATE_CANONICAL_HASH
    assert hashes["primitive_state_raw_sha256"] == REC004AK_PRIMITIVE_STATE_RAW_HASH
    assert hashes["primitive_state_canonical_hash"] == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH


def test_rec004aq_sampling_distinctness_in_warm_start() -> None:
    """Verifies that in steps 1..500, all generated sequences are strictly pairwise-distinct."""
    seed = 10
    vocab_size = 10
    length_range = (6, 10)

    # Test several steps within warm-start window (steps 1, 50, 250, 500)
    for step in (1, 50, 250, 500):
        examples = generate_step_training_examples_warm_start(
            seed=seed,
            step=step,
            operation="MIRROR_HALVES",
            vocab_size=vocab_size,
            sequence_length_range=length_range,
            warm_start_steps=500,
            n=32,
        )
        assert len(examples) == 32
        for ex in examples:
            inp = ex.input_tokens
            assert length_range[0] <= len(inp) <= length_range[1]
            # Must be strictly pairwise-distinct: set length equals sequence length
            assert len(set(inp)) == len(inp), f"Duplicate token in warm-start step {step}: {inp}"
            # For MIRROR_HALVES on length 10: input_tokens[0] != input_tokens[7]
            if len(inp) == 10:
                assert inp[0] != inp[7], (
                    "Aliasing between key 0 and key 7 present in distinct sample"
                )


def test_rec004aq_sampling_reverts_to_baseline_post_warm_start() -> None:
    """Verifies that in steps 501..6000, sampling strictly reverts to REC-004AL baseline."""
    seed = 10
    vocab_size = 10
    length_range = (6, 10)

    for step in (501, 502, 1000, 6000):
        warm_exs = generate_step_training_examples_warm_start(
            seed=seed,
            step=step,
            operation="MIRROR_HALVES",
            vocab_size=vocab_size,
            sequence_length_range=length_range,
            warm_start_steps=500,
            n=32,
        )
        base_exs = ibc._generate_step_training_examples(
            seed=seed,
            step=step,
            operation="MIRROR_HALVES",
            vocab_size=vocab_size,
            sequence_length_range=length_range,
            n=32,
        )
        assert len(warm_exs) == len(base_exs)
        for w_ex, b_ex in zip(warm_exs, base_exs, strict=True):
            assert w_ex.input_tokens == b_ex.input_tokens
            assert w_ex.target_tokens == b_ex.target_tokens
            assert w_ex.split == b_ex.split


def test_rec004aq_information_boundary_audit() -> None:
    """Verifies that information boundary audit passes for CD-DPCA."""
    audit = run_information_boundary_audit()
    assert audit["status"] == "PASS"
    assert not audit["has_forbidden_tokens"]
    assert audit["forward_parameters_valid"]
    assert not audit["routing_uses_content_features"]
    assert not audit["routing_uses_target_tokens"]


def test_rec004aq_position4_routing_metrics_schema() -> None:
    """Verifies position-4 routing metric calculation on initial fresh-loaded primitive."""
    fresh_bank = PrimitiveBank.from_artifacts(REC004AK_DIR, prefix="bank", strict=True)
    prim = fresh_bank.get(0)
    assert isinstance(prim, ContentDecoupledDiscretePositionalCrossAttentionPrimitive)

    val_examples = ibc._generate_parameter_free_examples(
        seed=10,
        n=64,
        operation="MIRROR_HALVES",
        split="rec004a_budget_validation",
        vocab_size=10,
        sequence_length_range=(6, 10),
    )

    device = torch.device("cpu")
    prim.to(device)
    metrics = compute_position4_routing_metrics(prim, val_examples, device)

    assert metrics["target_length"] == 10
    assert metrics["target_position"] == 4
    assert metrics["correct_key"] == 0
    assert metrics["competitor_key"] == 7
    assert "top1_key_mode" in metrics
    assert "mean_p_correct" in metrics
    assert "mean_p_competitor" in metrics
    assert "mean_margin_vs_runnerup" in metrics
    assert "mean_margin_vs_competitor" in metrics
    assert "mean_entropy" in metrics


def test_rec004aq_strata_gradient_alignment_schema() -> None:
    """Verifies strata gradient alignment calculation on initial fresh-loaded primitive."""
    parent_manifest, _ = ibc._load_parent_manifest()
    core, _, _ = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=10), parent_manifest
    )
    device = torch.device("cpu")
    core.model.to(device)

    fresh_bank = PrimitiveBank.from_artifacts(REC004AK_DIR, prefix="bank", strict=True)
    prim = fresh_bank.get(0)
    assert isinstance(prim, ContentDecoupledDiscretePositionalCrossAttentionPrimitive)
    prim.to(device)

    val_examples = ibc._generate_parameter_free_examples(
        seed=10,
        n=64,
        operation="MIRROR_HALVES",
        split="rec004a_budget_validation",
        vocab_size=10,
        sequence_length_range=(6, 10),
    )

    strata_results = compute_strata_gradient_alignment_at_checkpoint(
        core, prim, val_examples, device
    )

    for stratum_name in (
        "stratum_A_unique_target",
        "stratum_B_aliased_other",
        "stratum_C_aliased_key7",
        "pooled_all",
    ):
        if stratum_name in strata_results:
            st = strata_results[stratum_name]
            assert "predicted_margin_change" in st
            assert "cosine_alignment" in st
            assert "routing_grad_norm" in st
            assert "grad_norms_by_component" in st
            assert "per_example_mean" in st
            assert "per_example_median" in st
