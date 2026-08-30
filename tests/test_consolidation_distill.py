from __future__ import annotations

import dataclasses

import pytest
import torch

from apc.consolidation.distill import (
    ConsolidationConfig,
    ConsolidationReport,
    consolidate,
    measure_activity,
)
from apc.plastic.allocator import AllocatorPreset
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.primitive import PrimitiveConfig

D_MODEL = 8


def _allocated_workspace(num_transforms: int = 2, rank: int = 4) -> PlasticWorkspace:
    workspace = PlasticWorkspace()
    workspace.allocate(
        AllocatorPreset.SMALL, num_transforms, PrimitiveConfig(d_model=D_MODEL, rank=rank)
    )
    return workspace


def _perturb(workspace: PlasticWorkspace, transform_id: int, *, scale: float = 1.0) -> None:
    """Give a transform a non-trivial (non-identity) function to distill."""
    transform = workspace.get(transform_id)
    with torch.no_grad():
        transform.b_proj.weight.add_(scale)


# --- measure_activity -------------------------------------------------------
# Architecture doc section 10 steps 1-2: "measure activity ... drop inactive".


def test_measure_activity_splits_by_usage_count_threshold() -> None:
    workspace = _allocated_workspace(3)
    ids = workspace.ids()
    workspace.record_usage([ids[0], ids[0], ids[1]])  # ids[2] stays at usage_count 0

    active, inactive = measure_activity(workspace, activity_threshold=1)
    assert active == [ids[0], ids[1]]
    assert inactive == [ids[2]]


def test_measure_activity_threshold_zero_treats_everything_active() -> None:
    workspace = _allocated_workspace(2)
    active, inactive = measure_activity(workspace, activity_threshold=0)
    assert active == workspace.ids()
    assert inactive == []


def test_measure_activity_disabled_transform_is_always_inactive() -> None:
    workspace = _allocated_workspace(2)
    ids = workspace.ids()
    workspace.record_usage([ids[0]] * 5)
    workspace.get(ids[0]).enabled = False

    active, inactive = measure_activity(workspace, activity_threshold=1)
    assert active == []  # ids[0] disabled despite usage, ids[1] never used
    assert inactive == ids


# --- consolidate: input validation ------------------------------------------


def test_consolidate_raises_if_workspace_not_allocated() -> None:
    workspace = PlasticWorkspace()
    config = ConsolidationConfig(candidate_rank=1)
    with pytest.raises(ValueError, match="no allocated"):
        consolidate(workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL), config)


def test_consolidate_raises_if_no_active_transforms() -> None:
    workspace = _allocated_workspace(2)  # usage_count 0 for every transform
    config = ConsolidationConfig(candidate_rank=1, activity_threshold=1)
    with pytest.raises(ValueError, match="activity_threshold"):
        consolidate(workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL), config)


def test_consolidate_raises_on_mismatched_hidden_state_dims() -> None:
    workspace = _allocated_workspace(1)
    workspace.record_usage(workspace.ids())
    config = ConsolidationConfig(candidate_rank=1)
    with pytest.raises(ValueError, match="d_model"):
        consolidate(workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL + 1), config)


def test_consolidate_raises_if_candidate_not_smaller_than_temporary() -> None:
    workspace = _allocated_workspace(1, rank=2)
    workspace.record_usage(workspace.ids())
    # A single active rank-2 transform has 2*8*2=32 params; rank=4 candidate has 64 params.
    config = ConsolidationConfig(candidate_rank=4, steps=1)
    with pytest.raises(ValueError, match="not smaller"):
        consolidate(workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL), config)


# --- ConsolidationConfig validation -----------------------------------------


def test_config_rejects_non_positive_candidate_rank() -> None:
    with pytest.raises(ValueError, match="candidate_rank"):
        ConsolidationConfig(candidate_rank=0)


def test_config_rejects_negative_activity_threshold() -> None:
    with pytest.raises(ValueError, match="activity_threshold"):
        ConsolidationConfig(candidate_rank=1, activity_threshold=-1)


def test_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        ConsolidationConfig(candidate_rank=1, steps=0)


def test_config_rejects_non_positive_lr() -> None:
    with pytest.raises(ValueError, match="lr"):
        ConsolidationConfig(candidate_rank=1, lr=0.0)


def test_config_rejects_both_objective_weights_zero() -> None:
    with pytest.raises(ValueError, match="current_task_weight"):
        ConsolidationConfig(candidate_rank=1, current_task_weight=0.0, replay_weight=0.0)


# --- Task 010 acceptance criteria -------------------------------------------
# "candidate is smaller; reports task imitation score and prior-task replay
#  score; temporary module is not deleted."


def test_candidate_is_smaller_than_temporary_capacity() -> None:
    workspace = _allocated_workspace(2, rank=4)
    workspace.record_usage(workspace.ids())
    config = ConsolidationConfig(candidate_rank=2, steps=5, seed=0)

    candidate, report = consolidate(
        workspace, torch.randn(6, D_MODEL), torch.randn(6, D_MODEL), config
    )

    assert report.candidate_parameter_count == candidate.num_parameters()
    assert report.candidate_parameter_count < report.temporary_parameter_count
    assert report.compression_ratio < 1.0


def test_report_includes_task_imitation_and_replay_scores() -> None:
    workspace = _allocated_workspace(1, rank=4)
    _perturb(workspace, workspace.ids()[0])
    workspace.record_usage(workspace.ids())
    config = ConsolidationConfig(candidate_rank=2, steps=5, seed=0)

    _, report = consolidate(
        workspace, torch.randn(6, D_MODEL), torch.randn(6, D_MODEL), config
    )

    assert isinstance(report.task_imitation_score, float)
    assert isinstance(report.prior_task_replay_score, float)
    assert 0.0 <= report.task_imitation_score <= 1.0
    assert 0.0 <= report.prior_task_replay_score <= 1.0


def test_temporary_module_is_not_deleted_or_mutated() -> None:
    workspace = _allocated_workspace(2, rank=4)
    _perturb(workspace, workspace.ids()[0])
    workspace.record_usage(workspace.ids())
    before_ids = workspace.ids()
    before_params = {
        tid: workspace.get(tid).b_proj.weight.detach().clone() for tid in before_ids
    }
    config = ConsolidationConfig(candidate_rank=1, steps=5, seed=0)

    consolidate(workspace, torch.randn(6, D_MODEL), torch.randn(6, D_MODEL), config)

    assert workspace.ids() == before_ids
    assert workspace.is_allocated
    for tid in before_ids:
        assert torch.equal(workspace.get(tid).b_proj.weight, before_params[tid])
        assert not workspace.get(tid).is_frozen()


# --- Distillation quality / fidelity ----------------------------------------


def test_distillation_improves_reproduction_score_over_training() -> None:
    # Two active transforms that compute the *same* function (transform 1 is
    # a copy of transform 0): their combined delta has the same effective
    # rank (<= 4) as a single transform, so a rank-4 candidate has exactly
    # enough capacity to reproduce it, while still costing fewer parameters
    # than the two transforms combined (64 vs 128) -- the redundant-primitive
    # case consolidation is meant to compress losslessly.
    torch.manual_seed(0)
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _perturb(workspace, ids[0], scale=2.0)
    with torch.no_grad():
        workspace.get(ids[1]).a_proj.weight.copy_(workspace.get(ids[0]).a_proj.weight)
        workspace.get(ids[1]).b_proj.weight.copy_(workspace.get(ids[0]).b_proj.weight)
    workspace.record_usage(ids)

    current = torch.randn(32, D_MODEL)
    replay = torch.randn(32, D_MODEL)

    zero_step_config = ConsolidationConfig(candidate_rank=4, steps=1, lr=1e-2, seed=0)
    _, zero_step_report = consolidate(workspace, current, replay, zero_step_config)

    trained_config = ConsolidationConfig(candidate_rank=4, steps=300, lr=5e-2, seed=0)
    _, trained_report = consolidate(workspace, current, replay, trained_config)

    assert trained_report.candidate_parameter_count < trained_report.temporary_parameter_count
    assert trained_report.task_imitation_score > zero_step_report.task_imitation_score
    assert trained_report.task_imitation_score > 0.9
    assert trained_report.prior_task_replay_score > 0.9


def test_inactive_transform_is_excluded_from_teacher_signal() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _perturb(workspace, ids[0], scale=2.0)  # will be active
    _perturb(workspace, ids[1], scale=5.0)  # will stay inactive (usage_count 0)
    workspace.record_usage([ids[0]])

    current = torch.randn(16, D_MODEL)
    replay = torch.randn(16, D_MODEL)
    config = ConsolidationConfig(candidate_rank=2, activity_threshold=1, steps=1)

    _, report = consolidate(workspace, current, replay, config)

    assert report.active_ids == (ids[0],)
    assert report.inactive_ids == (ids[1],)
    # Only the active transform's parameters count toward what is being compressed.
    assert report.temporary_parameter_count == workspace.get(ids[0]).num_parameters()


def test_consolidate_does_not_train_the_temporary_transforms() -> None:
    workspace = _allocated_workspace(1, rank=4)
    _perturb(workspace, workspace.ids()[0])
    workspace.record_usage(workspace.ids())
    transform = workspace.get(workspace.ids()[0])
    a_before = transform.a_proj.weight.detach().clone()
    b_before = transform.b_proj.weight.detach().clone()

    config = ConsolidationConfig(candidate_rank=2, steps=50, lr=1e-1, seed=0)
    consolidate(workspace, torch.randn(8, D_MODEL), torch.randn(8, D_MODEL), config)

    assert torch.equal(transform.a_proj.weight, a_before)
    assert torch.equal(transform.b_proj.weight, b_before)
    assert transform.a_proj.weight.grad is None
    assert transform.b_proj.weight.grad is None


def test_candidate_metadata_records_consolidated_source_ids() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    workspace.record_usage([ids[0]])  # only ids[0] is active
    config = ConsolidationConfig(candidate_rank=1, steps=1)

    candidate, report = consolidate(
        workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL), config
    )

    assert candidate.metadata["consolidated_from"] == [ids[0]]
    assert candidate.status.value == "candidate"
    assert report.active_ids == (ids[0],)


def test_consolidate_is_deterministic_given_seed() -> None:
    def _run() -> ConsolidationReport:
        # Seed before allocating: the temporary transform's own random init
        # must also be reproducible across calls, not just the candidate's.
        torch.manual_seed(123)
        workspace = _allocated_workspace(1, rank=4)
        _perturb(workspace, workspace.ids()[0])
        workspace.record_usage(workspace.ids())
        current = torch.randn(8, D_MODEL)
        replay = torch.randn(8, D_MODEL)
        config = ConsolidationConfig(candidate_rank=2, steps=20, seed=42)
        _, report = consolidate(workspace, current, replay, config)
        return report

    report_a = _run()
    report_b = _run()
    assert report_a.task_imitation_score == pytest.approx(report_b.task_imitation_score)
    assert report_a.prior_task_replay_score == pytest.approx(report_b.prior_task_replay_score)
    assert report_a.final_loss == pytest.approx(report_b.final_loss)


def test_report_to_dict_has_expected_keys() -> None:
    workspace = _allocated_workspace(1, rank=4)
    workspace.record_usage(workspace.ids())
    config = ConsolidationConfig(candidate_rank=1, steps=1)
    _, report = consolidate(
        workspace, torch.randn(4, D_MODEL), torch.randn(4, D_MODEL), config
    )

    expected_keys = {f.name for f in dataclasses.fields(ConsolidationReport)}
    assert set(report.to_dict()) == expected_keys
