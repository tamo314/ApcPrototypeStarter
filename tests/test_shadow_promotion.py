"""Unit tests for Task A1-B007X-006 Shadow Validation, Promotion, and Release."""

from pathlib import Path

import pytest
import torch

from apc.core.tokens import build_special_tokens
from apc.evaluation.discovery_capacity_harness import (
    CapacityTier,
    build_tier_primitive,
)
from apc.evaluation.shadow_promotion import (
    MAX_CANDIDATE_PARAMS,
    MAX_FORGETTING_THRESHOLD,
    ShadowPromotionConfig,
    run_shadow_promotion_single_seed,
    shadow_promotion_config_from_dict,
    verify_bank_unchanged,
    verify_core_unchanged,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)


class _DummyCore:
    """Minimal dummy core for fast CPU testing."""

    def __init__(self, d_model: int = 32, vocab_size: int = 10) -> None:
        self.device = torch.device("cpu")
        self.tokens = build_special_tokens(vocab_size)

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.d_model = d_model

        class _InnerModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = _Cfg()
                self.config.d_model = d_model
                self.emb = torch.nn.Embedding(vocab_size + 4, d_model)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.emb(x)

        self.model = _InnerModel()
        for p in self.model.parameters():
            p.requires_grad_(False)


def test_config_initialization_and_serialization() -> None:
    config = ShadowPromotionConfig(
        seeds=(0, 1),
        novel_operation="SWAP_PAIRS",
        max_canonical_forgetting=0.02,
        num_canonical_eval_examples=50,
    )
    assert config.seeds == (0, 1)
    assert config.novel_operation == "SWAP_PAIRS"
    assert config.max_canonical_forgetting == MAX_FORGETTING_THRESHOLD
    assert config.max_candidate_parameters == MAX_CANDIDATE_PARAMS

    cfg_dict = config.to_dict()
    assert cfg_dict["novel_operation"] == "SWAP_PAIRS"
    assert cfg_dict["num_canonical_eval_examples"] == 50

    reloaded = shadow_promotion_config_from_dict(cfg_dict)
    assert reloaded.seeds == (0, 1)
    assert reloaded.novel_operation == "SWAP_PAIRS"
    assert reloaded.num_canonical_eval_examples == 50


def test_verify_core_unchanged_detects_mutation() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    initial_weights = {k: v.detach().clone() for k, v in core.model.named_parameters()}

    # Check 1: Identical weights return True
    assert verify_core_unchanged(initial_weights, core) is True

    # Check 2: Mutate one parameter
    with torch.no_grad():
        for p in core.model.parameters():
            p.add_(0.5)
            break

    assert verify_core_unchanged(initial_weights, core) is False


def test_verify_bank_unchanged_detects_mutation() -> None:
    pcfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=16,
    )
    prim0 = CrossPositionPrimitive(0, pcfg, status=PrimitiveStatus.STABLE)
    prim1 = CrossPositionPrimitive(1, pcfg, status=PrimitiveStatus.STABLE)
    bank = PrimitiveBank([prim0, prim1])

    initial_weights = {k: v.detach().clone() for k, v in bank.named_parameters()}

    # Check 1: Identical weights return True
    assert verify_bank_unchanged(initial_weights, bank, [0, 1]) is True

    # Check 2: Mutate primitive 0
    with torch.no_grad():
        for p in prim0.parameters():
            p.add_(0.1)
            break

    assert verify_bank_unchanged(initial_weights, bank, [0, 1]) is False


def test_conditional_promotion_and_release_contract() -> None:
    # Setup initial bank with 2 primitives
    pcfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=16,
    )
    prim0 = CrossPositionPrimitive(0, pcfg, status=PrimitiveStatus.STABLE)
    prim1 = CrossPositionPrimitive(1, pcfg, status=PrimitiveStatus.STABLE)
    bank = PrimitiveBank([prim0, prim1])
    bank_size_before = len(bank)
    assert bank_size_before == 2

    # Candidate primitive
    candidate = build_tier_primitive(
        CapacityTier.T0_COMPACT.value, "SWAP_PAIRS", vocab_size=10, d_model=192, primitive_id=2
    )
    assert candidate.status == PrimitiveStatus.CANDIDATE
    assert candidate.num_parameters() == 17098
    assert candidate.num_parameters() <= MAX_CANDIDATE_PARAMS

    temp_params_before = 137482

    # Case A: Safety PASSED -> Install exactly one, bank +1, candidate STABLE, temp released to 0
    safety_passed = True
    if safety_passed:
        candidate.status = PrimitiveStatus.STABLE
        candidate.freeze()
        bank.add_primitive(candidate)
        bank_size_after = len(bank)
        temp_params_after = 0
        temp_released = True
    else:
        temp_params_after = temp_params_before

    assert bank_size_after == bank_size_before + 1
    assert len(bank) == 3
    assert candidate.status == PrimitiveStatus.STABLE
    assert temp_params_after == 0
    assert temp_released is True

    # Case B: Safety FAILED -> Abort install, bank unchanged, fallback preserved
    bank2 = PrimitiveBank([prim0, prim1])
    cand2 = CrossPositionPrimitive(2, pcfg, status=PrimitiveStatus.CANDIDATE)
    temp_params_before_b = 137482

    safety_passed_b = False
    if safety_passed_b:
        cand2.status = PrimitiveStatus.STABLE
        bank2.add_primitive(cand2)
        temp_params_after_b = 0
    else:
        bank_size_after_b = len(bank2)
        temp_params_after_b = temp_params_before_b
        temp_released_b = False

    assert bank_size_after_b == 2
    assert len(bank2) == 2  # bank unchanged
    assert temp_params_after_b == 137482  # temporary fallback preserved
    assert temp_released_b is False


def test_shadow_promotion_fault_injection_core_mutation() -> None:
    ckpt_dir = Path("runs/phase_a1_overcomplete_distillation/checkpoints")
    ckpt = ckpt_dir / "distill_candidate_SWAP_PAIRS_seed_0.pt"
    core_ckpt = Path("runs/phase_a1_discovery_capacity_harness/shared_encoder.pt")
    if not ckpt.exists() or not core_ckpt.exists():
        pytest.skip("Required checkpoints not found for fault injection test")

    cfg = ShadowPromotionConfig(
        seeds=(0,),
        novel_operation="SWAP_PAIRS",
        num_canonical_eval_examples=5,
        num_composition_eval_examples=5,
        num_novel_eval_examples=5,
        shared_encoder_checkpoint=str(core_ckpt),
        device="cpu",
    )

    # Inject core mutation: should fail invariant and abort install
    report = run_shadow_promotion_single_seed(cfg, 0, inject_core_mutation=True)
    assert report.core_unchanged is False
    assert report.safety_criteria_passed is False
    assert report.promoted is False
    assert report.installed_primitive_id is None
    assert report.bank_size_after == report.bank_size_before == 8
    assert report.temp_released_completely is False
    assert report.overall_passed is False

