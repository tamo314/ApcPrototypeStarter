"""CPU contracts for B-C005REC-004Q."""

from __future__ import annotations

import torch

from apc.evaluation import mirror_attention_score_credit_assignment_audit as audit
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _primitive() -> CrossPositionLengthBiasPrimitive:
    return CrossPositionLengthBiasPrimitive(
        99,
        CrossPositionLengthBiasPrimitiveConfig(
            operation="MIRROR_HALVES",
            d_model=8,
            d_operator=4,
            n_head=1,
            d_operator_ff=8,
            vocab_size=5,
            max_sequence_length=10,
            length_ref=10,
        ),
    )


def test_alignment_loss_descent_increases_the_oracle_margin() -> None:
    scores = torch.zeros(1, 1, 6, 6, requires_grad=True)
    gradient = torch.autograd.grad(audit._alignment_loss(scores, 6), scores)[0]
    correct = 3  # mirror_halves_position_map(6)[0]
    strongest_wrong = 0

    assert gradient[0, 0, 0, correct] < gradient[0, 0, 0, strongest_wrong]
    assert (gradient[0, 0, 0, strongest_wrong] - gradient[0, 0, 0, correct]).item() > 0.0


def test_score_jvp_is_zero_for_a_zero_effective_update() -> None:
    primitive = _primitive()
    content = torch.randn(2, 6, 8)
    deltas = {name: torch.zeros_like(parameter) for name, parameter in primitive.named_parameters()}

    scores = audit._score_forward(primitive, content, [6, 6])["scores"]
    tangent = audit._score_jvp(primitive, content, 6, deltas)

    assert scores.shape == (2, 1, 6, 6)
    assert torch.count_nonzero(tangent) == 0


def test_score_groups_exclude_fused_v_rows() -> None:
    primitive = _primitive()
    groups = audit._groups(primitive)
    rows = primitive.cross_attn.in_proj_weight.shape[0] // 3

    assert groups["Q_rows"][1] == slice(0, rows)
    assert groups["K_rows"][1] == slice(rows, 2 * rows)
    assert all("cross_attn.in_proj" not in name for name in groups["CONTENT_PREP"][0])


def test_score_only_active_gradient_has_no_v_or_content_rows() -> None:
    primitive = _primitive()
    rows = primitive.cross_attn.in_proj_weight.shape[0] // 3
    gradient = torch.ones_like(primitive.cross_attn.in_proj_weight)

    selected = audit._active_gradient("cross_attn.in_proj_weight", gradient, "score_only", rows)

    assert selected is not None
    assert torch.equal(selected[: 2 * rows], gradient[: 2 * rows])
    assert torch.count_nonzero(selected[2 * rows :]) == 0
    assert (
        audit._active_gradient(
            "content_in_proj.weight",
            torch.ones_like(primitive.content_in_proj.weight),
            "score_only",
            rows,
        )
        is None
    )


def test_decision_leaves_adoption_and_follow_on_work_disabled() -> None:
    def bucket(decrease: float, improve: float, norm: float) -> dict[str, float]:
        return {
            "task_descent_margin_decrease_fraction": decrease,
            "task_descent_margin_improve_fraction": improve,
            "mean_task_gradient_norm": norm,
        }

    position = {
        name: {
            "buckets": {
                "length10_position4": bucket(0.1, 0.9, 1.0),
                "length6_9_aggregate": bucket(0.1, 0.9, 1.0),
            }
        }
        for name in ("I03_SCORE_ONLY_12000", "I03_CP_SCORE_12000", "I04_P_7000")
    }
    adamw = {
        name: {
            "effective_update_margin": {
                "length10_position4": {"effective_update_margin_decrease_fraction": 0.7}
            }
        }
        for name in ("I03_SCORE_ONLY_12000", "I03_CP_SCORE_12000")
    }

    decision = audit._decision(position, adamw)

    assert decision["label"] == "ADAMW_STATE_PRECONDITIONING_CONFLICT_SUPPORTED"
    assert decision["new_optimizer_updates"] == 0
    assert decision["selected_init"] is None
    assert decision["child_bundle"] is None
    assert decision["rg3_recheck"] == "NOT_EXECUTED"
