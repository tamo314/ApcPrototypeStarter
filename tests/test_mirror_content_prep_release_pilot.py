"""CPU contracts for B-C005REC-004P."""

from __future__ import annotations

from apc.evaluation import mirror_content_prep_release_pilot as pilot
from apc.evaluation import mirror_score_only_continuation_pilot as score_only
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _primitive() -> CrossPositionLengthBiasPrimitive:
    config = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES",
        d_model=8,
        d_operator=4,
        n_head=1,
        d_operator_ff=8,
        vocab_size=5,
        max_sequence_length=10,
        length_ref=10,
    )
    return CrossPositionLengthBiasPrimitive(99, config)


def test_partition_releases_only_content_prep_relative_to_score_only() -> None:
    primitive = _primitive()
    score_partition = score_only.build_forward_graph_partition(primitive)
    release_partition = pilot.build_content_prep_partition(primitive)

    assert set(release_partition["groups"]["content_prep"]) == {
        "content_in_proj.weight",
        "content_in_proj.bias",
        "content_position_embedding.weight",
    }
    assert set(release_partition["trainable_tensor_keys"]) == (
        set(score_partition["trainable_tensor_keys"])
        | set(release_partition["groups"]["content_prep"])
    )
    assert "answer_query_embedding.weight" in release_partition["frozen_whole_tensor_keys"]


def test_decision_uses_registered_score_value_conflict_precedence() -> None:
    def metrics(j0: float, o1: float, length10: float) -> dict[str, object]:
        return {
            "j0_sequence_exact_match": j0,
            "oracle_sequence_exact_match": o1,
            "per_length": {"10": {"j0_sequence_exact_match": length10}},
        }

    cp_score = {
        pilot.REC004P_VALIDATION_SPLIT: metrics(0.20, 0.90, 0.20),
        pilot.REC004P_CONFIRMATION_SPLIT: metrics(0.20, 0.90, 0.20),
    }
    score_only_metrics = {
        pilot.REC004P_VALIDATION_SPLIT: metrics(0.00, 1.00, 0.00),
        pilot.REC004P_CONFIRMATION_SPLIT: metrics(0.00, 1.00, 0.00),
    }

    assert (
        pilot._decision(cp_score, score_only_metrics, contracts_pass=True)["label"]
        == "SHARED_CONTENT_PREP_SCORE_VALUE_CONFLICT_SUPPORTED"
    )
