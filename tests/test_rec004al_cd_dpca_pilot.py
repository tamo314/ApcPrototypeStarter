"""Focused regression and contract tests for Task B-C005REC-004AL:
CD-DPCA Single-Init Learning Pilot (ADR-0137).
"""

from __future__ import annotations

import json
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
    REC004AL_INIT_ID,
    REC004AL_PARENT_BUNDLE_ID,
    REC004AL_PARENT_CORE_HASH,
    REC004AL_TARGET_OPERATION,
    MirrorCDDPCALearningPilotConfig,
    _expected_lr,
    build_pilot_protocol,
    compute_per_length_position_metrics,
    run_information_boundary_audit,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CD_DPCA_ARCHITECTURE_SIGNATURE,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
)
from apc.utils import model_bundle as mb

ROOT = Path(__file__).resolve().parent.parent
REC004AK_DIR = ROOT / "runs/phase_b_restart/rec004ak/run_001"
REC004AL_DIR = ROOT / "runs/phase_b_restart/rec004al/run_001"


def test_rec004al_protocol_and_source_hashes() -> None:
    """Verifies protocol builder enforces exact parent bundle and REC-004AK hashes."""
    config = MirrorCDDPCALearningPilotConfig(
        output_dir=REC004AL_DIR,
        rec004ak_dir=REC004AK_DIR,
    )
    parent_manifest, _ = ibc._load_parent_manifest()
    protocol = build_pilot_protocol(config, parent_manifest)

    assert protocol["task_id"] == "B-C005REC-004AL"
    assert protocol["parent_bundle_id"] == REC004AL_PARENT_BUNDLE_ID
    assert protocol["parent_core_canonical_state_hash"] == REC004AL_PARENT_CORE_HASH
    assert protocol["init_id"] == REC004AL_INIT_ID
    assert protocol["target_operation"] == REC004AL_TARGET_OPERATION
    assert protocol["architecture_signature"] == CD_DPCA_ARCHITECTURE_SIGNATURE

    hashes = protocol["source_hashes"]
    assert hashes["bank_manifest_sha256"] == REC004AK_BANK_MANIFEST_HASH
    assert hashes["primitive_config_sha256"] == REC004AK_PRIMITIVE_CONFIG_HASH
    assert hashes["bank_state_raw_sha256"] == REC004AK_BANK_STATE_RAW_HASH
    assert hashes["bank_state_canonical_hash"] == REC004AK_BANK_STATE_CANONICAL_HASH
    assert hashes["primitive_state_raw_sha256"] == REC004AK_PRIMITIVE_STATE_RAW_HASH
    assert hashes["primitive_state_canonical_hash"] == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH


def test_rec004al_lr_trace_schedule() -> None:
    """Verifies closed-form CosineAnnealingLR formula matches preregistered trace."""
    assert abs(_expected_lr(0) - 0.0008) < 1e-9
    assert abs(_expected_lr(1000) - 1e-5) < 1e-9
    assert abs(_expected_lr(2000) - 0.0008) < 1e-9
    assert abs(_expected_lr(3000) - 1e-5) < 1e-9
    assert abs(_expected_lr(4000) - 0.0008) < 1e-9
    assert abs(_expected_lr(5000) - 1e-5) < 1e-9
    assert abs(_expected_lr(6000) - 0.0008) < 1e-9


def test_rec004al_information_boundary_audit() -> None:
    """Audits that runtime routing receives no target map, labels, or oracle attention."""
    audit = run_information_boundary_audit()
    assert audit["status"] == "PASS"
    assert not audit["has_forbidden_tokens"]
    assert audit["forward_parameters_valid"]
    assert not audit["routing_uses_content_features"]
    assert not audit["routing_uses_target_tokens"]
    assert not audit["routing_uses_oracle_attention"]
    assert not audit["routing_uses_permutation_table"]


def test_rec004al_fresh_load_and_isolated_trainable_copy() -> None:
    """Verifies strict fresh load from REC-004AK and isolated trainable instance."""
    fresh_bank = PrimitiveBank.from_artifacts(REC004AK_DIR, prefix="bank", strict=True)
    initial_prim = fresh_bank.get(0)
    assert isinstance(initial_prim, ContentDecoupledDiscretePositionalCrossAttentionPrimitive)

    initial_sd = initial_prim.state_dict()
    canon_hash = mb.canonical_state_hash(initial_sd)
    assert canon_hash == REC004AK_PRIMITIVE_STATE_CANONICAL_HASH

    # Create isolated copy
    copy_prim = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(
        primitive_id=12,
        config=initial_prim.config,
    )
    copy_prim.load_state_dict(initial_sd, strict=True)

    # Verify distinct tensor references
    for k in initial_sd:
        assert copy_prim.state_dict()[k].data_ptr() != initial_sd[k].data_ptr()


def test_rec004al_padding_mask_and_attention_normalization() -> None:
    """Verifies padding positions have -inf score and 0.0 attention weight."""
    fresh_bank = PrimitiveBank.from_artifacts(REC004AK_DIR, prefix="bank", strict=True)
    primitive = fresh_bank.get(0)
    primitive.eval()

    c_lens = [6, 10]
    o_lens = [6, 10]
    scores = primitive.compute_routing_scores(c_lens, o_lens, None, lmax=10)
    attn = primitive.compute_attention_weights(c_lens, o_lens, None, lmax=10)

    # First sequence has L=6, positions 6..9 must be masked
    assert torch.all(scores[0, :, :, 6:] == float("-inf"))
    assert torch.all(attn[0, :, :, 6:] == 0.0)

    # Valid positions must sum to 1.0
    valid_sum_0 = attn[0, :, :, :6].sum(dim=-1)
    assert torch.allclose(valid_sum_0, torch.ones_like(valid_sum_0), atol=1e-5)

    valid_sum_1 = attn[1, :, :, :10].sum(dim=-1)
    assert torch.allclose(valid_sum_1, torch.ones_like(valid_sum_1), atol=1e-5)


def test_rec004al_per_length_position_metrics_structure() -> None:
    """Verifies metric computation produces exact numerators and denominators."""
    parent_manifest, _ = ibc._load_parent_manifest()
    core, _, _ = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(), parent_manifest
    )
    core.model.eval()

    fresh_bank = PrimitiveBank.from_artifacts(REC004AK_DIR, prefix="bank", strict=True)
    primitive = fresh_bank.get(0)

    val_examples = ibc._generate_parameter_free_examples(
        10, 16, operation="MIRROR_HALVES", split="rec004a_budget_validation",
        vocab_size=10, sequence_length_range=(6, 10)
    )

    summary, by_pos = compute_per_length_position_metrics(core, primitive, val_examples)
    assert summary["n_examples"] == 16
    assert 0.0 <= summary["sequence_exact_match"] <= 1.0
    assert 0.0 <= summary["token_accuracy"] <= 1.0

    # Denominators sum up correctly
    len_seq_total = sum(b["n_sequences"] for b in summary["by_length"].values())
    assert len_seq_total == 16


def test_rec004al_terminal_viability_criterion_and_boundaries() -> None:
    """Verifies fail-closed decision when terminal criterion is not met."""
    summary_path = REC004AL_DIR / "summary.json"
    assert summary_path.is_file(), "REC-004AL summary.json must exist"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task"] == "B-C005REC-004AL"
    assert summary["execution_status"] == "PASS"
    assert summary["decisive_step"] == 6000
    assert summary["terminal_viability_floor"] == 0.95

    # Viability criterion check
    achieved_em = summary["decisive_validation_sequence_em"]
    if achieved_em < 0.95:
        assert summary["decision"] == "PILOT_TERMINAL_VIABILITY_NOT_MET"
        assert summary["terminal_viability_met"] is False
    else:
        assert summary["decision"] == "PILOT_VIABILITY_MET"
        assert summary["terminal_viability_met"] is True

    # Strict boundaries
    assert summary["candidate_selected"] is None
    assert summary["child_bundle"] is None
    assert summary["bundle_write"] is False
    assert summary["rg3"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["g1"] == "NOT_CLEARED"
    assert summary["g4"] == "NOT_CLEARED"


def test_rec004al_saved_artifacts_completeness() -> None:
    """Verifies all required artifact files exist in run directory."""
    required_files = [
        "protocol.json",
        "config.yaml",
        "source_manifest.json",
        "information_boundary_audit.json",
        "freeze_audit.json",
        "side_effect_audit.json",
        "learning_curve.jsonl",
        "lr_trace.jsonl",
        "per_length_position_metrics.json",
        "causal_controls.json",
        "attention_masking_diagnostics.json",
        "summary.json",
        "report.md",
    ]
    for fname in required_files:
        p = REC004AL_DIR / fname
        assert p.is_file(), f"Missing required artifact: {fname}"

    # Checkpoint steps 0, 500, ..., 6000 exist
    for step in range(0, 6001, 500):
        ckpt_p = REC004AL_DIR / "checkpoints" / f"step{step}.pt"
        state_p = REC004AL_DIR / "training_states" / f"step{step}.pt"
        assert ckpt_p.is_file(), f"Missing checkpoint: step{step}.pt"
        assert state_p.is_file(), f"Missing training state: step{step}.pt"
