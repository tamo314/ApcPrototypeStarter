"""CPU invariants for B-C005REC-004O's fused-QKV freeze contract."""

from __future__ import annotations

import torch

from apc.evaluation import mirror_score_only_continuation_pilot as pilot
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _primitive() -> CrossPositionLengthBiasPrimitive:
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=8,
        d_operator=4,
        n_head=1,
        d_operator_ff=8,
        vocab_size=5,
        max_sequence_length=10,
        length_ref=10,
    )
    return CrossPositionLengthBiasPrimitive(99, cfg)


def _initialised_optimizer(primitive: CrossPositionLengthBiasPrimitive) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(primitive.parameters(), lr=0.01, weight_decay=0.1)
    features = torch.randn(2, 4, 8)
    logits = primitive(features, [4, 4], [4, 4], None)
    logits.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_partition_is_exhaustive_and_marks_only_score_path_trainable() -> None:
    primitive = _primitive()
    partition = pilot.build_forward_graph_partition(primitive)

    names = set(name for name, _ in primitive.named_parameters())
    trainable = set(partition["trainable_tensor_keys"])
    frozen = set(partition["frozen_whole_tensor_keys"])
    assert trainable | frozen == names
    assert "cross_attn.in_proj_weight" in trainable
    assert "position_bias_out.weight" in trainable
    assert "content_in_proj.weight" in frozen
    assert "readout.weight" in frozen


def test_freeze_guard_restores_v_rows_and_optimizer_rows_after_adamw_step() -> None:
    primitive = _primitive()
    optimizer = _initialised_optimizer(primitive)
    partition = pilot.build_forward_graph_partition(primitive)
    guard = pilot._FreezeGuard(primitive, optimizer, partition)
    named = dict(primitive.named_parameters())
    v_slice = pilot._slice_rows(named["cross_attn.in_proj_weight"])
    v_before = named["cross_attn.in_proj_weight"].detach()[v_slice].clone()
    state_before = optimizer.state[named["cross_attn.in_proj_weight"]]["exp_avg"][v_slice].clone()

    for name in partition["trainable_tensor_keys"]:
        named[name].grad = torch.ones_like(named[name])
    guard.zero_v_gradients()
    optimizer.step()
    guard.restore_and_verify(step=6001)

    assert torch.equal(named["cross_attn.in_proj_weight"].detach()[v_slice], v_before)
    assert torch.equal(
        optimizer.state[named["cross_attn.in_proj_weight"]]["exp_avg"][v_slice], state_before
    )
    assert guard.audit()["passed"] is True


def test_decision_requires_all_three_j0_floors_and_both_o1_floors() -> None:
    def metrics(j0: float, o1: float, length10: float) -> dict[str, object]:
        return {
            pilot.REC004O_VALIDATION_SPLIT: {
                "j0_sequence_exact_match": j0,
                "oracle_sequence_exact_match": o1,
                "per_length": {"10": {"j0_sequence_exact_match": length10}},
            },
            pilot.REC004O_CONFIRMATION_SPLIT: {
                "j0_sequence_exact_match": j0,
                "oracle_sequence_exact_match": o1,
                "per_length": {"10": {"j0_sequence_exact_match": j0}},
            },
        }

    result = pilot._decision(
        metrics(0.96, 0.96, 0.96),
        metrics(0.1, 0.1, 0.1),
        metrics(0.5, 0.5, 0.5),
        True,
        True,
    )
    assert result["label"] == "SCORE_ONLY_CONTINUATION_SUPPORTED_ON_I03_PILOT"
    insufficient = pilot._decision(
        metrics(0.8, 0.96, 0.8),
        metrics(0.1, 0.1, 0.1),
        metrics(0.7, 0.7, 0.7),
        True,
        True,
    )
    assert insufficient["label"] == "SCORE_ONLY_IMPROVEMENT_INSUFFICIENT"


def test_state_fingerprint_accepts_adamw_scalar_step_tensor() -> None:
    state = {"step": torch.tensor(6000.0), "moment": torch.ones(2, 3)}
    assert pilot._state_fingerprint(state) == pilot._state_fingerprint(state)
