"""Unit tests for Task A1-B007X-005 Overcomplete-to-Compact Functional Distillation."""

from __future__ import annotations

import torch

from apc.core.tokens import build_special_tokens
from apc.environments.generator import Example, Program, ProgramStep
from apc.evaluation.discovery_capacity_harness import (
    TIER_SPECS,
    CapacityTier,
    build_tier_primitive,
)
from apc.evaluation.overcomplete_distillation import (
    MAX_CANDIDATE_PARAMS,
    MAX_COMPRESSION_RATIO,
    OvercompleteDistillationConfig,
    evaluate_distillation_pair,
    overcomplete_distillation_config_from_dict,
)


class _DummyCore:
    """Minimal dummy core for fast CPU unit testing."""

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


def _make_dummy_example(
    input_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
    op_name: str = "SWAP_PAIRS",
) -> Example:
    step = ProgramStep(op_name, {})
    prog = Program(steps=(step,))
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=prog,
        operation_graph={},
        category="novel",
        split="test",
        vocab_size=10,
    )


def test_config_initialization_and_serialization() -> None:
    config = OvercompleteDistillationConfig(
        seeds=(0, 1),
        teacher_tier="T2_overcomplete",
        candidate_tier="T0_compact",
        distill_train_steps=100,
    )
    assert config.seeds == (0, 1)
    assert config.distill_train_steps == 100

    cfg_dict = config.to_dict()
    assert cfg_dict["teacher_tier"] == "T2_overcomplete"
    assert cfg_dict["max_candidate_params"] == MAX_CANDIDATE_PARAMS
    assert cfg_dict["max_compression_ratio"] == MAX_COMPRESSION_RATIO

    reloaded = overcomplete_distillation_config_from_dict(cfg_dict)
    assert reloaded.seeds == (0, 1)
    assert reloaded.distill_train_steps == 100


def test_capacity_and_compression_ratio_invariants() -> None:
    t0_spec = TIER_SPECS[CapacityTier.T0_COMPACT.value]
    t2_spec = TIER_SPECS[CapacityTier.T2_OVERCOMPLETE.value]

    candidate_params = t0_spec.expected_parameters
    teacher_params = t2_spec.expected_parameters

    assert candidate_params == 17098
    assert candidate_params <= MAX_CANDIDATE_PARAMS

    ratio = candidate_params / teacher_params
    assert ratio <= MAX_COMPRESSION_RATIO
    assert abs(ratio - 0.124365) < 1e-4  # ~8.04x compression


def test_pair_evaluation_metrics_exact() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    device = torch.device("cpu")

    candidate = build_tier_primitive(
        CapacityTier.T0_COMPACT.value, "SWAP_PAIRS", vocab_size=10, d_model=32
    )
    teacher = build_tier_primitive(
        CapacityTier.T2_OVERCOMPLETE.value, "SWAP_PAIRS", vocab_size=10, d_model=32
    )

    examples = [
        _make_dummy_example((1, 2, 3, 4), (2, 1, 4, 3)),
        _make_dummy_example((5, 6, 7, 8), (6, 5, 8, 7)),
    ]

    metrics = evaluate_distillation_pair(core, candidate, teacher, examples, device, batch_size=2)
    assert 0.0 <= metrics.candidate_em <= 1.0
    assert 0.0 <= metrics.teacher_em <= 1.0
    assert 0.0 <= metrics.functional_agreement <= 1.0
    assert 0.0 <= metrics.retention
    assert metrics.num_evaluated == 2


def test_frozen_teacher_and_trainable_candidate_gradient_invariants() -> None:
    core = _DummyCore(d_model=32, vocab_size=10)
    device = torch.device("cpu")

    teacher = build_tier_primitive(
        CapacityTier.T2_OVERCOMPLETE.value, "SWAP_PAIRS", vocab_size=10, d_model=32
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    candidate = build_tier_primitive(
        CapacityTier.T0_COMPACT.value, "SWAP_PAIRS", vocab_size=10, d_model=32
    ).to(device)
    candidate.train()

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    content_features = torch.randn(1, 4, 32)
    content_lengths = [4]
    output_lengths = [4]

    with torch.no_grad():
        t_logits = teacher(
            content_features=content_features,
            content_lengths=content_lengths,
            output_lengths=output_lengths,
        )

    optimizer.zero_grad()
    c_logits = candidate(
        content_features=content_features,
        content_lengths=content_lengths,
        output_lengths=output_lengths,
    )

    loss = torch.nn.functional.mse_loss(c_logits, t_logits)
    loss.backward()

    # Verify teacher received zero gradients
    for p in teacher.parameters():
        assert p.grad is None

    # Verify core received zero gradients
    for p in core.model.parameters():
        assert p.grad is None

    # Verify candidate received gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.parameters())
    assert has_grad, "Candidate parameters must receive non-zero gradients during distillation."
