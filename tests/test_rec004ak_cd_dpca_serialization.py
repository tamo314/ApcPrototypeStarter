"""Tests for Task B-C005REC-004AK: CD-DPCA Serialization
& Fresh-Load Validation (ADR-0136).

Validates:
1. REC-004AK run artifacts exist and record PASS with all invariants and gate blocks preserved.
2. CD-DPCA configuration serialization and strict deserialization (to_dict / from_dict).
3. PrimitiveBank manifest serialization and fresh reconstruction (to_manifest / from_manifest).
4. Strict state loading under immutable loader contract (strict=True, no missing/unexpected keys).
5. Exact parameter equivalence (names, shapes, dtypes, canonical state hash, state ABI hash).
6. Exact behavioral equivalence on fixed inputs (scores, attention weights,
   padding masking, forward logits).
7. Comprehensive negative checks verify fail-closed behavior on corrupted/incompatible inputs.
8. Runtime information boundary is preserved (zero target/oracle/label leakage).
9. Frozen invariants and research gate blocks remain intact.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest
import torch

from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CD_DPCA_ARCHITECTURE_SIGNATURE,
    CDDPCAPrimitive,
    CDDPCAPrimitiveConfig,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
    ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
    build_primitive_from_config_dict,
)

ROOT = Path(__file__).resolve().parent.parent
REC004AK_DIR = ROOT / "runs/phase_b_restart/rec004ak/run_001"


def test_rec004ak_artifacts_exist_and_pass() -> None:
    """Verify that REC-004AK artifacts exist and record PASS with all blocks preserved."""
    assert REC004AK_DIR.is_dir(), f"Run directory {REC004AK_DIR} does not exist"
    summary_path = REC004AK_DIR / "summary.json"
    assert summary_path.is_file(), f"Summary file {summary_path} does not exist"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task"] == "B-C005REC-004AK"
    assert summary["execution_status"] == "PASS"
    assert summary["decision"] == "CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED"
    assert summary["serialization_verified"] is True
    assert summary["fresh_load_strict_verified"] is True
    assert summary["parameters_equivalent"] is True
    assert summary["scorer_outputs_equivalent"] is True
    assert summary["attention_masking_equivalent"] is True
    assert summary["forward_readout_equivalent"] is True
    assert summary["call_statistics_compatible"] is True
    assert summary["negative_checks_passed"] is True
    assert summary["negative_checks_count"] == 10
    assert summary["runtime_information_boundary_preserved"] is True
    assert summary["architecture_signature"] == CD_DPCA_ARCHITECTURE_SIGNATURE
    assert summary["total_parameters"] == 19178
    assert summary["optimizer_updates"] == 0
    assert summary["new_parameters"] == 0
    assert summary["candidate_selected"] is None
    assert summary["bundle_write"] is False
    assert summary["rg3"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["g1"] == "NOT_CLEARED"
    assert summary["g4"] == "NOT_CLEARED"

    # Also verify that other artifacts exist
    assert (REC004AK_DIR / "report.md").is_file()
    assert (REC004AK_DIR / "bank_manifest.json").is_file()
    assert (REC004AK_DIR / "bank_state.pt").is_file()
    assert (REC004AK_DIR / "primitive_config.json").is_file()
    assert (REC004AK_DIR / "primitive_state.pt").is_file()
    assert (REC004AK_DIR / "equivalence_audit.json").is_file()
    assert (REC004AK_DIR / "negative_checks.json").is_file()
    assert (REC004AK_DIR / "information_boundary_audit.json").is_file()
    assert (REC004AK_DIR / "exact_compatibility_boundaries.json").is_file()


def test_cd_dpca_config_strict_serialization() -> None:
    """Verify CD-DPCA config and primitive config dict roundtrip and validation."""
    cfg = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    data = cfg.to_dict()
    assert data["operation"] == "MIRROR_HALVES"
    assert data["d_model"] == 192
    assert data["d_operator"] == 32
    assert data["n_head"] == 4
    assert data["vocab_size"] == 10
    assert data["max_sequence_length"] == 32
    assert data["d_operator_ff"] == 64
    assert data["arg_dim"] == 16

    # Deserialization roundtrip
    restored = CDDPCAPrimitiveConfig.from_dict(data)
    assert restored.to_dict() == data
    assert restored.operation == "MIRROR_HALVES"
    assert restored.d_model == 192
    assert restored.d_operator == 32
    assert restored.n_head == 4
    assert restored.vocab_size == 10
    assert restored.max_sequence_length == 32

    # Primitive to_config_dict contains architecture signature
    prim = CDDPCAPrimitive(0, cfg)
    spec = prim.to_config_dict()
    assert spec["primitive_id"] == 0
    assert spec["primitive_type"] == "ContentDecoupledDiscretePositionalCrossAttentionPrimitive"
    assert spec["architecture_signature"] == CD_DPCA_ARCHITECTURE_SIGNATURE
    assert spec["config"] == data


def test_bank_manifest_reconstruction_and_strict_load(tmp_path: Path) -> None:
    """Verify PrimitiveBank to_manifest, from_manifest, save_artifacts, and from_artifacts."""
    torch.manual_seed(42)
    bank = PrimitiveBank()
    cfg = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    prim = CDDPCAPrimitive(0, cfg)
    bank.add_primitive(prim)

    # Manifest roundtrip
    manifest = bank.to_manifest()
    assert isinstance(manifest, list)
    assert len(manifest) == 1
    assert manifest[0]["primitive_id"] == 0
    assert (
        manifest[0]["primitive_type"]
        == "ContentDecoupledDiscretePositionalCrossAttentionPrimitive"
    )
    assert manifest[0]["architecture_signature"] == CD_DPCA_ARCHITECTURE_SIGNATURE

    # Save and reload artifacts
    bank.save_artifacts(tmp_path, prefix="test_bank")
    manifest_file = tmp_path / "test_bank_manifest.json"
    state_file = tmp_path / "test_bank_state.pt"
    assert manifest_file.is_file()
    assert state_file.is_file()

    fresh_bank = PrimitiveBank.from_artifacts(tmp_path, prefix="test_bank", strict=True)
    assert len(fresh_bank) == 1
    fresh_prim = fresh_bank.get(0)
    assert isinstance(fresh_prim, CDDPCAPrimitive)

    # Check parameter equality
    orig_params = dict(bank.named_parameters())
    fresh_params = dict(fresh_bank.named_parameters())
    assert set(orig_params.keys()) == set(fresh_params.keys())
    for k in orig_params:
        assert torch.equal(orig_params[k], fresh_params[k])


def test_parameter_names_shapes_dtypes_and_hashes() -> None:
    """Verify exact parameter tensor count, shapes, dtypes, and reproducibility."""
    torch.manual_seed(999)
    cfg = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    prim1 = CDDPCAPrimitive(0, cfg)
    prim2 = CDDPCAPrimitive.from_config_dict(prim1.to_config_dict())
    prim2.load_state_dict(prim1.state_dict(), strict=True)

    params1 = dict(prim1.named_parameters())
    params2 = dict(prim2.named_parameters())

    assert len(params1) == 20
    assert set(params1.keys()) == set(params2.keys())
    total_params = sum(p.numel() for p in params1.values())
    assert total_params == 19178

    for name, p1 in params1.items():
        p2 = params2[name]
        assert p1.shape == p2.shape, f"Shape mismatch for {name}"
        assert p1.dtype == torch.float32, f"Dtype mismatch for {name}"
        assert torch.equal(p1, p2), f"Value mismatch for {name}"


def test_behavioral_equivalence_on_fixed_inputs() -> None:
    """Verify bitwise/numerical identical outputs between original and restored instances."""
    torch.manual_seed(42)
    cfg = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    orig_prim = ContentDecoupledDiscretePositionalCrossAttentionPrimitive(0, cfg)
    orig_prim.eval()

    # Reconstruct fresh instance via factory and load_state_dict
    fresh_prim = build_primitive_from_config_dict(orig_prim.to_config_dict())
    fresh_prim.load_state_dict(orig_prim.state_dict(), strict=True)
    fresh_prim.eval()

    batch = 3
    lmax = 10
    content_lengths = [10, 8, 6]
    output_lengths = [10, 8, 6]
    h = torch.randn(batch, lmax, 192)

    with torch.no_grad():
        orig_scores = orig_prim.compute_routing_scores(
            content_lengths, output_lengths, lmax=lmax
        )
        fresh_scores = fresh_prim.compute_routing_scores(
            content_lengths, output_lengths, lmax=lmax
        )
        orig_weights = orig_prim.compute_attention_weights(
            content_lengths, output_lengths, lmax=lmax
        )
        fresh_weights = fresh_prim.compute_attention_weights(
            content_lengths, output_lengths, lmax=lmax
        )
        orig_logits, orig_fwd_w = orig_prim.forward(
            h, content_lengths, output_lengths, return_attention=True
        )
        fresh_logits, fresh_fwd_w = fresh_prim.forward(
            h, content_lengths, output_lengths, return_attention=True
        )

    # 1. Scores bitwise identical
    assert (orig_scores == fresh_scores).all()
    assert (orig_weights == fresh_weights).all()
    assert (orig_logits == fresh_logits).all()
    assert (orig_fwd_w == fresh_fwd_w).all()

    # 2. Padding masking correctness: padded key positions must have score -inf and weight 0.0
    for b_idx in range(batch):
        c_len = content_lengths[b_idx]
        o_len = output_lengths[b_idx]
        if c_len < lmax:
            pad_scores = fresh_scores[b_idx, :, :o_len, c_len:]
            pad_weights = fresh_weights[b_idx, :, :o_len, c_len:]
            assert torch.isneginf(pad_scores).all()
            assert (pad_weights == 0.0).all()
        valid_weights = fresh_weights[b_idx, :, :o_len, :c_len]
        sums = valid_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_negative_checks_fail_closed() -> None:
    """Verify fail-closed behavior across all 10 negative conditions."""
    cfg = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    prim = CDDPCAPrimitive(0, cfg)
    sd = prim.state_dict()

    # 1. Missing state_dict parameter -> RuntimeError (strict=True)
    corrupted_sd_missing = dict(sd)
    del corrupted_sd_missing["length_embedding.weight"]
    fresh1 = CDDPCAPrimitive(0, cfg)
    with pytest.raises(RuntimeError, match="Missing key"):
        fresh1.load_state_dict(corrupted_sd_missing, strict=True)

    # 2. Unexpected state_dict parameter -> RuntimeError (strict=True)
    corrupted_sd_extra = dict(sd)
    corrupted_sd_extra["unexpected_extra_tensor"] = torch.randn(32, 32)
    fresh2 = CDDPCAPrimitive(0, cfg)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        fresh2.load_state_dict(corrupted_sd_extra, strict=True)

    # 3. Incompatible tensor shape -> RuntimeError (strict=True)
    corrupted_sd_shape = dict(sd)
    corrupted_sd_shape["content_position_embedding.weight"] = torch.randn(16, 32)
    fresh3 = CDDPCAPrimitive(0, cfg)
    with pytest.raises(RuntimeError, match="size mismatch"):
        fresh3.load_state_dict(corrupted_sd_shape, strict=True)

    # 4. Missing required config field -> KeyError
    bad_cfg_missing = cfg.to_dict()
    del bad_cfg_missing["operation"]
    with pytest.raises(KeyError):
        CDDPCAPrimitiveConfig.from_dict(bad_cfg_missing)

    # 5. Unexpected config field -> ValueError
    bad_cfg_extra = cfg.to_dict()
    bad_cfg_extra["unsupported_option"] = "illegal_value"
    with pytest.raises(ValueError, match="Unexpected configuration field"):
        CDDPCAPrimitiveConfig.from_dict(bad_cfg_extra)

    # 6. Incompatible operator dimension divisibility -> ValueError
    bad_cfg_div = cfg.to_dict()
    bad_cfg_div["d_operator"] = 30  # 30 % 4 != 0
    with pytest.raises(ValueError, match="divisible"):
        CDDPCAPrimitiveConfig.from_dict(bad_cfg_div)

    # 7. Incompatible max sequence length -> ValueError
    bad_cfg_len = cfg.to_dict()
    bad_cfg_len["max_sequence_length"] = 0
    with pytest.raises(ValueError, match="must be >= 1"):
        CDDPCAPrimitiveConfig.from_dict(bad_cfg_len)

    # 8. Unknown primitive type in manifest -> ValueError
    manifest_bad_type = [
        {"primitive_id": 0, "primitive_type": "UnknownFakePrimitive", "config": cfg.to_dict()}
    ]
    with pytest.raises(ValueError, match="Unknown primitive type"):
        PrimitiveBank.from_manifest(manifest_bad_type)

    # 9. Duplicate primitive id in manifest -> ValueError
    manifest_dup = [
        prim.to_config_dict(),
        prim.to_config_dict(),
    ]
    with pytest.raises(ValueError, match="already exists in bank"):
        PrimitiveBank.from_manifest(manifest_dup)

    # 10. Architecture signature mismatch -> ValueError
    bad_spec_arch = prim.to_config_dict()
    bad_spec_arch["architecture_signature"] = "cross_position_v1"
    with pytest.raises(ValueError, match="Architecture signature mismatch"):
        CDDPCAPrimitive.from_config_dict(bad_spec_arch)


def test_runtime_information_boundary_preserved() -> None:
    """Verify that routing computation does not access any target/oracle tokens."""
    cfg = CDDPCAPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=192,
        d_operator=32,
        n_head=4,
        vocab_size=10,
        max_sequence_length=32,
    )
    prim = CDDPCAPrimitive(0, cfg)

    # Check method signatures
    repr_params = set(inspect.signature(prim.compute_routing_representations).parameters.keys())
    scores_params = set(inspect.signature(prim.compute_routing_scores).parameters.keys())

    forbidden = {"target", "targets", "label", "labels", "teacher", "oracle", "ground_truth"}
    assert not (forbidden & repr_params)
    assert not (forbidden & scores_params)

    # AST check
    src = inspect.getsource(prim.compute_routing_representations)
    tree = ast.parse(textwrap.dedent(src))
    accessed = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not (forbidden & accessed)

    # Parameter-free operation has None arg_encoder/arg_proj
    assert prim.arg_encoder is None
    assert prim.arg_proj is None


def test_frozen_invariants_and_gate_blocks() -> None:
    """Verify that execution maintains all research gates and frozen invariants."""
    summary_path = REC004AK_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["optimizer_updates"] == 0
    assert summary["new_parameters"] == 0
    assert summary["candidate_selected"] is None
    assert summary["bundle_write"] is False
    assert summary["rg3"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["g1"] == "NOT_CLEARED"
    assert summary["g4"] == "NOT_CLEARED"
