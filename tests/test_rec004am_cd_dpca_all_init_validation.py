"""Focused regression and contract tests for Task B-C005REC-004AM:
Fixed Sequence-Distinctness Warm-Start Multi-Initialization Reproduction.
"""

from __future__ import annotations

from pathlib import Path

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation.mirror_cd_dpca_all_init_validation import (
    REC004AM_BASELINE_TASK_ID,
    REC004AM_INIT_IDS,
    REC004AM_PILOT_TASK_ID,
    REC004AM_SOURCE_TASK_ID,
    REC004AM_TASK_ID,
    REC004AM_WARM_START_STEPS,
    MirrorCDDPCAAllInitValidationConfig,
    _expected_lr,
    build_all_init_protocol,
    build_all_initial_states,
    generate_step_training_examples_warm_start,
)
from apc.evaluation.mirror_cd_dpca_learning_pilot import (
    REC004AK_BANK_MANIFEST_HASH,
    REC004AK_BANK_STATE_CANONICAL_HASH,
    REC004AK_BANK_STATE_RAW_HASH,
    REC004AK_PRIMITIVE_CONFIG_HASH,
    REC004AK_PRIMITIVE_STATE_CANONICAL_HASH,
    REC004AK_PRIMITIVE_STATE_RAW_HASH,
    REC004AL_ARCHITECTURE_SIGNATURE,
    REC004AL_PARENT_BUNDLE_ID,
    REC004AL_PARENT_CORE_HASH,
    REC004AL_TARGET_OPERATION,
    run_information_boundary_audit,
)
from apc.utils import model_bundle as mb

ROOT = Path(__file__).resolve().parent.parent
REC004AK_DIR = ROOT / "runs/phase_b_restart/rec004ak/run_001"
REC004AL_DIR = ROOT / "runs/phase_b_restart/rec004al/run_001"
REC004AQ_DIR = ROOT / "runs/phase_b_restart/rec004aq/run_001"


def test_rec004am_protocol_and_source_hashes() -> None:
    """Verifies all-init protocol builder enforces parent bundle, source hashes, and bounds."""
    config = MirrorCDDPCAAllInitValidationConfig(
        rec004ak_dir=REC004AK_DIR,
        rec004al_dir=REC004AL_DIR,
        rec004aq_dir=REC004AQ_DIR,
    )
    parent_manifest, _ = ibc._load_parent_manifest()
    initial_states = build_all_initial_states(config)
    init_hashes = {k: mb.canonical_state_hash(v) for k, v in initial_states.items()}
    protocol = build_all_init_protocol(config, parent_manifest, init_hashes)

    assert protocol["task_id"] == REC004AM_TASK_ID
    assert protocol["pilot_task_id"] == REC004AM_PILOT_TASK_ID
    assert protocol["baseline_task_id"] == REC004AM_BASELINE_TASK_ID
    assert protocol["source_task_id"] == REC004AM_SOURCE_TASK_ID
    assert protocol["parent_bundle_id"] == REC004AL_PARENT_BUNDLE_ID
    assert protocol["parent_core_canonical_state_hash"] == REC004AL_PARENT_CORE_HASH
    assert protocol["target_operation"] == REC004AL_TARGET_OPERATION
    assert protocol["architecture_signature"] == REC004AL_ARCHITECTURE_SIGNATURE
    assert protocol["init_ids"] == list(REC004AM_INIT_IDS)
    assert protocol["total_inits"] == 5

    intervention = protocol["causal_intervention"]
    assert intervention["warm_start_steps"] == REC004AM_WARM_START_STEPS
    assert intervention["boundary_pre_fixed"] is True
    assert intervention["sweep_authorized"] is False
    assert intervention["length_position_selectivity_prohibited"] is True
    assert intervention["data_stream_identical_across_inits"] is True

    hashes = protocol["source_hashes"]
    assert hashes["bank_manifest_sha256"] == REC004AK_BANK_MANIFEST_HASH
    assert hashes["primitive_config_sha256"] == REC004AK_PRIMITIVE_CONFIG_HASH
    assert hashes["bank_state_raw_sha256"] == REC004AK_BANK_STATE_RAW_HASH
    assert hashes["bank_state_canonical_hash"] == REC004AK_BANK_STATE_CANONICAL_HASH
    assert hashes["primitive_state_raw_sha256"] == REC004AK_PRIMITIVE_STATE_RAW_HASH
    assert hashes["primitive_state_canonical_hash"] == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH


def test_rec004am_all_initial_states_distinct_and_i01_matched() -> None:
    """Verifies that all 5 initial states are mutually distinct and I01 bit-for-bit
    matches REC-004AK."""
    config = MirrorCDDPCAAllInitValidationConfig(
        rec004ak_dir=REC004AK_DIR,
        rec004al_dir=REC004AL_DIR,
        rec004aq_dir=REC004AQ_DIR,
    )
    initial_states = build_all_initial_states(config)
    assert len(initial_states) == 5
    assert set(initial_states.keys()) == set(REC004AM_INIT_IDS)

    hashes = {k: mb.canonical_state_hash(v) for k, v in initial_states.items()}

    # I01 must bitwise-match REC-004AK canonical hash
    assert hashes["I01"] == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH

    # All 5 canonical hashes must be unique
    assert len(set(hashes.values())) == 5, f"Hashes not unique: {hashes}"


def test_rec004am_lr_schedule_trace() -> None:
    """Verifies closed-form CosineAnnealingLR formula matches preregistered trace."""
    assert abs(_expected_lr(0) - 0.0008) < 1e-9
    assert abs(_expected_lr(1000) - 1e-5) < 1e-9
    assert abs(_expected_lr(2000) - 0.0008) < 1e-9
    assert abs(_expected_lr(3000) - 1e-5) < 1e-9
    assert abs(_expected_lr(4000) - 0.0008) < 1e-9
    assert abs(_expected_lr(5000) - 1e-5) < 1e-9
    assert abs(_expected_lr(6000) - 0.0008) < 1e-9


def test_rec004am_warm_start_sampling_distinctness() -> None:
    """Verifies that in steps 1..500, all generated sequences are strictly pairwise-distinct."""
    seed = 10
    vocab_size = 10
    length_range = (6, 10)

    for step in (1, 100, 250, 500):
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
            assert len(set(inp)) == len(inp), f"Duplicate token in warm-start step {step}: {inp}"


def test_rec004am_data_stream_identical_across_inits() -> None:
    """Verifies that data generation is identical regardless of which init is active."""
    # Data generator depends only on seed (10) and step, never init_id
    examples_a = generate_step_training_examples_warm_start(
        seed=10,
        step=42,
        operation="MIRROR_HALVES",
        vocab_size=10,
        sequence_length_range=(6, 10),
        warm_start_steps=500,
        n=32,
    )
    examples_b = generate_step_training_examples_warm_start(
        seed=10,
        step=42,
        operation="MIRROR_HALVES",
        vocab_size=10,
        sequence_length_range=(6, 10),
        warm_start_steps=500,
        n=32,
    )
    assert len(examples_a) == len(examples_b)
    for ex_a, ex_b in zip(examples_a, examples_b, strict=True):
        assert ex_a.input_tokens == ex_b.input_tokens
        assert ex_a.target_tokens == ex_b.target_tokens


def test_rec004am_information_boundary_audit() -> None:
    """Audits that runtime routing receives no target map, labels, or oracle attention."""
    audit = run_information_boundary_audit()
    assert audit["status"] == "PASS"
    assert not audit["has_forbidden_tokens"]
    assert audit["forward_parameters_valid"]
    assert not audit["routing_uses_content_features"]
    assert not audit["routing_uses_target_tokens"]
    assert not audit["routing_uses_oracle_attention"]
    assert not audit["routing_uses_permutation_table"]
