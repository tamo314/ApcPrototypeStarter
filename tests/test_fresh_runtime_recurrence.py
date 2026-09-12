"""Unit tests for Task A1-B007X-007 Fresh-Runtime Recurrence After Compression."""

from pathlib import Path

import pytest
import torch

from apc.core.tokens import build_special_tokens
from apc.evaluation import unified_oracle_causal_benchmark as unified
from apc.evaluation.fresh_runtime_recurrence import (
    EXPECTED_BANK_SIZE,
    EXPECTED_NOVEL_PRIMITIVE_ID,
    RECURRENCE_ACCURACY_THRESHOLD,
    UNCONSOLIDATED_CEILING_THRESHOLD,
    FreshRuntimeRecurrenceConfig,
    fresh_runtime_recurrence_config_from_dict,
    run_fresh_runtime_recurrence_seed,
    verify_persistent_invariants,
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
    config = FreshRuntimeRecurrenceConfig(
        seeds=(0, 1),
        novel_operation="SWAP_PAIRS",
        recurrence_accuracy_threshold=0.95,
        num_recurrence_eval_examples=50,
    )
    assert config.seeds == (0, 1)
    assert config.novel_operation == "SWAP_PAIRS"
    assert config.recurrence_accuracy_threshold == RECURRENCE_ACCURACY_THRESHOLD
    assert config.unconsolidated_threshold == UNCONSOLIDATED_CEILING_THRESHOLD

    cfg_dict = config.to_dict()
    assert cfg_dict["novel_operation"] == "SWAP_PAIRS"
    assert cfg_dict["num_recurrence_eval_examples"] == 50

    reloaded = fresh_runtime_recurrence_config_from_dict(cfg_dict)
    assert reloaded.seeds == (0, 1)
    assert reloaded.novel_operation == "SWAP_PAIRS"
    assert reloaded.num_recurrence_eval_examples == 50


def test_verify_persistent_invariants_core_mutation() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    bank = PrimitiveBank()

    core_initial = {k: v.detach().clone() for k, v in core.model.named_parameters()}
    bank_initial = {k: v.detach().clone() for k, v in bank.named_parameters()}

    # Identical weights pass
    core_ok, bank_ok = verify_persistent_invariants(core_initial, core, bank_initial, bank)
    assert core_ok is True
    assert bank_ok is True

    # Mutating core fails core invariant
    with torch.no_grad():
        for p in core.model.parameters():
            p.add_(1.0)
            break

    core_ok, bank_ok = verify_persistent_invariants(core_initial, core, bank_initial, bank)
    assert core_ok is False
    assert bank_ok is True


def test_verify_persistent_invariants_bank_mutation() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    bank = PrimitiveBank()
    pcfg = CrossPositionPrimitiveConfig(
        operation="SWAP_PAIRS",
        d_model=32,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
    )
    p = CrossPositionPrimitive(primitive_id=8, config=pcfg, status=PrimitiveStatus.STABLE)
    bank.add_primitive(p)

    core_initial = {k: v.detach().clone() for k, v in core.model.named_parameters()}
    bank_initial = {k: v.detach().clone() for k, v in bank.named_parameters()}

    core_ok, bank_ok = verify_persistent_invariants(core_initial, core, bank_initial, bank)
    assert core_ok is True
    assert bank_ok is True

    # Mutating bank fails bank invariant
    with torch.no_grad():
        for p_param in bank.parameters():
            p_param.add_(0.5)
            break

    core_ok, bank_ok = verify_persistent_invariants(core_initial, core, bank_initial, bank)
    assert core_ok is True
    assert bank_ok is False


def test_expected_constants() -> None:
    assert EXPECTED_BANK_SIZE == 9
    assert EXPECTED_NOVEL_PRIMITIVE_ID == 8
    assert RECURRENCE_ACCURACY_THRESHOLD == 0.95
    assert UNCONSOLIDATED_CEILING_THRESHOLD == 0.05


@pytest.mark.parametrize("mutation", [None, "core", "bank"])
def test_fresh_runtime_recurrence_fault_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None,
) -> None:
    """Exercise real serialization/evaluation with a tiny, coherent local fixture."""
    model = {"d_model": 32, "n_layer": 1, "n_head": 4, "d_ff": 64,
             "max_seq_len": 48, "dropout": 0.0}
    arch_config = unified.SharedEncoderArchitectureConfig(model=model, device="cpu")
    arch = unified.build_shared_encoder_architecture(arch_config)
    core_ckpt = tmp_path / "core.pt"
    torch.save(arch.core.model.state_dict(), core_ckpt)
    bank, _ = unified._build_heterogeneous_bank(
        unified.UnifiedBenchmarkConfig(model=model, device="cpu")
    )
    bank.add_primitive(CrossPositionPrimitive(
        primitive_id=8,
        config=CrossPositionPrimitiveConfig(
            operation="SWAP_PAIRS", d_model=32, d_operator=32, n_head=4,
            d_operator_ff=64, vocab_size=10,
            max_sequence_length=FreshRuntimeRecurrenceConfig().max_sequence_length,
        ),
        status=PrimitiveStatus.STABLE,
    ))
    promoted_ckpt = tmp_path / "promoted.pt"
    torch.save(bank.state_dict(), promoted_ckpt)

    def forbid_training(*args, **kwargs):
        pytest.fail("Recurrence invariant tests must load the fixture without training fallback")

    monkeypatch.setattr(unified, "_train_shared", forbid_training)
    cfg = FreshRuntimeRecurrenceConfig(
        seeds=(0,),
        novel_operation="SWAP_PAIRS",
        num_recurrence_eval_examples=5,
        num_unconsolidated_eval_examples=5,
        num_canonical_eval_examples=5,
        num_composition_eval_examples=5,
        shared_encoder_checkpoint=str(core_ckpt),
        promoted_bank_checkpoint_pattern=str(promoted_ckpt),
        op_to_id_checkpoint_pattern=str(tmp_path / "absent_op_to_id.json"),
        plastic_bank_checkpoint_pattern=str(tmp_path / "absent_plastic.pt"),
        model=model,
        device="cpu",
    )
    report = run_fresh_runtime_recurrence_seed(
        cfg, 0, inject_core_mutation=mutation == "core", inject_bank_mutation=mutation == "bank",
    )
    assert report.core_unchanged is (mutation != "core")
    assert report.bank_unchanged is (mutation != "bank")
    assert report.adaptation_steps == report.allocated_temporary_parameters == 0
    if mutation:
        assert report.overall_passed is False



