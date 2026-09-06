"""Focused invariant coverage for B-C005D2-002 (no GPU checkpoints required)."""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.representation_stage_probe import (
    RepresentationStageProbeConfig,
    _centroid_accuracy,
    _classify_relation,
    _fit_binary_probe,
    _interpret,
    _probe_accuracy,
)


def _separable_clusters(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    positive_center = torch.tensor([3.0, 3.0])
    negative_center = torch.tensor([-3.0, -3.0])
    train_pos = positive_center + 0.1 * torch.randn(32, 2, generator=generator)
    train_neg = negative_center + 0.1 * torch.randn(32, 2, generator=generator)
    eval_pos = positive_center + 0.1 * torch.randn(32, 2, generator=generator)
    eval_neg = negative_center + 0.1 * torch.randn(32, 2, generator=generator)
    return train_pos, train_neg, eval_pos, eval_neg


def _overlapping_clusters(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_pos = torch.randn(32, 2, generator=generator)
    train_neg = torch.randn(32, 2, generator=generator)
    eval_pos = torch.randn(32, 2, generator=generator)
    eval_neg = torch.randn(32, 2, generator=generator)
    return train_pos, train_neg, eval_pos, eval_neg


def test_linear_probe_separates_well_clustered_classes() -> None:
    train_pos, train_neg, eval_pos, eval_neg = _separable_clusters(0)
    probe = _fit_binary_probe(train_pos, train_neg, steps=200, lr=0.1, seed=0, shuffle_labels=False)
    assert _probe_accuracy(probe, eval_pos, eval_neg) >= 0.95


def test_shuffled_label_probe_is_near_chance_on_overlapping_classes() -> None:
    train_pos, train_neg, eval_pos, eval_neg = _overlapping_clusters(1)
    probe = _fit_binary_probe(train_pos, train_neg, steps=200, lr=0.1, seed=1, shuffle_labels=True)
    accuracy = _probe_accuracy(probe, eval_pos, eval_neg)
    assert 0.3 <= accuracy <= 0.7


def test_centroid_accuracy_separates_well_clustered_classes() -> None:
    train_pos, train_neg, eval_pos, eval_neg = _separable_clusters(2)
    assert _centroid_accuracy(train_pos, train_neg, eval_pos, eval_neg) >= 0.95


def _config(**overrides: object) -> RepresentationStageProbeConfig:
    return RepresentationStageProbeConfig(**overrides)  # type: ignore[arg-type]


def test_config_rejects_development_overlap_with_sealed_partitions() -> None:
    with pytest.raises(ValueError, match="development seeds"):
        _config(development_seeds=(20,), regate_sealed_seeds=(21,))


def test_config_rejects_non_standard_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        _config(bank_size=17)


def _summary(**overrides: float | None) -> dict[str, float | None]:
    base = {
        "n": 5,
        "key_cosine_similarity": 0.5,
        "key_euclidean_distance": 1.0,
        "score_margin": 1.0,
        "nearest_key_rank": 1.0,
        "control_a_current_path_top1": 1.0,
        "control_b_z_centroid_accuracy": 1.0,
        "control_c_q_probe_accuracy": 1.0,
        "control_c_q_probe_shuffled_accuracy": 0.5,
        "control_d_oracle_lookup_top1": 1.0,
        "z_probe_accuracy": 1.0,
        "z_probe_shuffled_accuracy": 0.5,
    }
    base.update(overrides)
    return base


def test_classify_key_scoring_bottleneck_when_representation_survives_but_path_fails() -> None:
    config = _config()
    summary = _summary(control_a_current_path_top1=0.5)
    classification, evidence = _classify_relation(summary, config)
    assert classification == "KEY_SCORING_BOTTLENECK"
    assert evidence["shuffled_controls_at_chance"] is True


def test_classify_query_projection_bottleneck_when_only_q_task_fails() -> None:
    config = _config()
    summary = _summary(control_a_current_path_top1=0.5, control_c_q_probe_accuracy=0.5)
    classification, _ = _classify_relation(summary, config)
    assert classification == "QUERY_PROJECTION_BOTTLENECK"


def test_classify_task_representation_bottleneck_when_z_task_itself_fails() -> None:
    config = _config()
    summary = _summary(
        control_a_current_path_top1=0.5, control_c_q_probe_accuracy=0.5, z_probe_accuracy=0.5
    )
    classification, _ = _classify_relation(summary, config)
    assert classification == "TASK_REPRESENTATION_BOTTLENECK"


def test_classify_no_failure_when_representation_and_path_both_succeed() -> None:
    config = _config()
    classification, _ = _classify_relation(_summary(), config)
    assert classification == "NO_FAILURE"


def test_classify_unresolved_when_shuffled_control_is_not_at_chance() -> None:
    config = _config()
    summary = _summary(control_a_current_path_top1=0.5, z_probe_shuffled_accuracy=0.95)
    classification, evidence = _classify_relation(summary, config)
    assert classification == "UNRESOLVED"
    assert evidence["shuffled_controls_at_chance"] is False


def _row(target: str, competitor: str, *, control_a: float, z_acc: float) -> dict[str, object]:
    return {
        "partition": "regate_sealed",
        "router_state": "R2_frozen_post_repair",
        "target_family": target,
        "competitor_family": competitor,
        "key_cosine_similarity": 0.5,
        "key_euclidean_distance": 1.0,
        "score_margin": 0.0,
        "nearest_key_rank": 1.0,
        "control_a_current_path_top1": control_a,
        "control_b_z_centroid_accuracy": z_acc,
        "control_c_q_probe_accuracy": z_acc,
        "control_c_q_probe_shuffled_accuracy": 0.5,
        "control_d_oracle_lookup_top1": 1.0,
        "z_probe_accuracy": z_acc,
        "z_probe_shuffled_accuracy": 0.5,
    }


def test_interpret_reports_mixed_when_relations_disagree() -> None:
    config = _config(target_operations=("SHIFT", "SELECT"))
    rows = [
        _row("SHIFT", "CYCLE_FOUR", control_a=1.0, z_acc=1.0),
        _row("SELECT", "BIND", control_a=0.5, z_acc=1.0),
    ]
    result = _interpret(rows, config)
    focus = result["by_partition_state"]["regate_sealed/R2_frozen_post_repair"]
    assert focus["classification"] == "MIXED"
    assert result["verdict"] == "MIXED"
    assert set(result["per_relation_verdict"].values()) == {"NO_FAILURE", "KEY_SCORING_BOTTLENECK"}
