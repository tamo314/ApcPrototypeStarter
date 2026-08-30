from __future__ import annotations

import dataclasses

import pytest
import torch

from apc.consolidation.distill import ConsolidationConfig, consolidate
from apc.consolidation.shadow import (
    ShadowValidationConfig,
    ShadowValidationReport,
    evaluate_shadow,
    run_shadow_validation,
)
from apc.plastic.allocator import AllocatorPreset
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus

D_MODEL = 8


def _allocated_workspace(num_transforms: int = 2, rank: int = 4) -> PlasticWorkspace:
    workspace = PlasticWorkspace()
    workspace.allocate(
        AllocatorPreset.SMALL, num_transforms, PrimitiveConfig(d_model=D_MODEL, rank=rank)
    )
    return workspace


def _perturb(primitive: Primitive, *, scale: float = 1.0) -> None:
    with torch.no_grad():
        primitive.b_proj.weight.add_(scale)


def _make_redundant_pair(workspace: PlasticWorkspace, ids: list[int]) -> None:
    """Give both transforms in `ids` the *same* non-trivial function, so
    their combined delta has the same effective rank as a single transform
    (mirrors `test_consolidation_distill.py`'s
    `test_distillation_improves_reproduction_score_over_training`): a
    same-rank candidate then has exactly enough capacity to reproduce the
    combined teacher near-losslessly while still costing half the
    parameters."""
    _perturb(workspace.get(ids[0]), scale=2.0)
    with torch.no_grad():
        workspace.get(ids[1]).a_proj.weight.copy_(workspace.get(ids[0]).a_proj.weight)
        workspace.get(ids[1]).b_proj.weight.copy_(workspace.get(ids[0]).b_proj.weight)


def _temporary_output(
    workspace: PlasticWorkspace, active_ids: list[int], h: torch.Tensor
) -> torch.Tensor:
    """`h + sum(B_i(A_i(h)))` over `active_ids` -- used as the "ground truth"
    a passing candidate must reproduce in these tests (the temporary
    solution stands in for the task's correct output, since Phase A has no
    consolidation wiring into the real environment/model yet -- see
    `apc.consolidation.shadow`'s module docstring)."""
    with torch.no_grad():
        total = h
        for transform in workspace.get_many(active_ids):
            total = total + transform.b_proj(transform.a_proj(h))
        return total


def _distilled_candidate(
    workspace: PlasticWorkspace,
    active_ids: list[int],
    *,
    candidate_id: int = 100,
    rank: int = 4,
    replay_weight: float = 1.0,
    steps: int = 400,
) -> Primitive:
    data = torch.randn(64, D_MODEL)
    config = ConsolidationConfig(
        candidate_rank=rank, steps=steps, lr=5e-2, replay_weight=replay_weight, seed=0
    )
    candidate, _ = consolidate(workspace, data, data, config, candidate_id=candidate_id)
    return candidate


def _default_config(**overrides: object) -> ShadowValidationConfig:
    return dataclasses.replace(ShadowValidationConfig(), **overrides)  # type: ignore[arg-type]


# --- ShadowValidationConfig validation --------------------------------------


def test_config_rejects_out_of_range_task_score_ratio_threshold() -> None:
    with pytest.raises(ValueError, match="task_score_ratio_threshold"):
        ShadowValidationConfig(task_score_ratio_threshold=1.5)


def test_config_rejects_negative_max_retention_degradation() -> None:
    with pytest.raises(ValueError, match="max_retention_degradation"):
        ShadowValidationConfig(max_retention_degradation=-0.1)


def test_config_rejects_non_positive_max_size_ratio() -> None:
    with pytest.raises(ValueError, match="max_size_ratio"):
        ShadowValidationConfig(max_size_ratio=0.0)


def test_config_rejects_negative_correctness_tolerance() -> None:
    with pytest.raises(ValueError, match="correctness_tolerance"):
        ShadowValidationConfig(correctness_tolerance=-0.01)


# --- evaluate_shadow: input validation ---------------------------------------


def test_evaluate_shadow_raises_on_empty_active_ids() -> None:
    workspace = _allocated_workspace(1)
    candidate = Primitive(100, PrimitiveConfig(d_model=D_MODEL, rank=1))
    data = torch.randn(4, D_MODEL)
    with pytest.raises(ValueError, match="active_ids"):
        evaluate_shadow(workspace, [], candidate, data, data, data, data, _default_config())


def test_evaluate_shadow_raises_on_mismatched_input_target_shapes() -> None:
    workspace = _allocated_workspace(1)
    ids = workspace.ids()
    candidate = Primitive(100, PrimitiveConfig(d_model=D_MODEL, rank=1))
    inputs = torch.randn(4, D_MODEL)
    bad_targets = torch.randn(5, D_MODEL)
    with pytest.raises(ValueError, match="shape"):
        evaluate_shadow(
            workspace, ids, candidate, inputs, bad_targets, inputs, inputs, _default_config()
        )


def test_evaluate_shadow_raises_on_d_model_mismatch() -> None:
    workspace = _allocated_workspace(1)
    ids = workspace.ids()
    candidate = Primitive(100, PrimitiveConfig(d_model=D_MODEL + 1, rank=1))
    data = torch.randn(4, D_MODEL)
    with pytest.raises(ValueError, match="d_model"):
        evaluate_shadow(workspace, ids, candidate, data, data, data, data, _default_config())


# --- evaluate_shadow: report shape/ranges ------------------------------------


def test_report_scores_and_ratios_are_in_expected_ranges() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids)

    current = torch.randn(16, D_MODEL)
    replay = torch.randn(16, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    target_replay = _temporary_output(workspace, ids, replay)
    report = evaluate_shadow(
        workspace, ids, candidate, current, target_current, replay, target_replay,
        _default_config(),
    )

    assert 0.0 <= report.temporary_task_score <= 1.0
    assert 0.0 <= report.candidate_task_score <= 1.0
    assert 0.0 <= report.temporary_retention_score <= 1.0
    assert 0.0 <= report.candidate_retention_score <= 1.0
    assert 0.0 <= report.disagreement_rate <= 1.0
    assert report.temporary_parameter_count == sum(
        workspace.get(tid).num_parameters() for tid in ids
    )
    assert report.candidate_parameter_count == candidate.num_parameters()


def test_report_to_dict_has_expected_keys() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids)
    data = torch.randn(8, D_MODEL)
    target = _temporary_output(workspace, ids, data)

    report = evaluate_shadow(
        workspace, ids, candidate, data, target, data, target, _default_config()
    )

    expected_keys = {f.name for f in dataclasses.fields(ShadowValidationReport)}
    assert set(report.to_dict()) == expected_keys


# --- Task 011 acceptance: passing / failing behavior -------------------------


def test_well_trained_candidate_passes_shadow_validation() -> None:
    torch.manual_seed(0)
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids, rank=4)

    current = torch.randn(32, D_MODEL)
    replay = torch.randn(32, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    target_replay = _temporary_output(workspace, ids, replay)
    report = evaluate_shadow(
        workspace, ids, candidate, current, target_current, replay, target_replay,
        _default_config(),
    )

    assert report.finite_outputs
    assert report.task_score_passed
    assert report.retention_passed
    assert report.size_passed
    assert report.passed
    assert report.failure_reasons == ()


def test_untrained_candidate_fails_task_score_check() -> None:
    workspace = _allocated_workspace(1, rank=4)
    ids = workspace.ids()
    _perturb(workspace.get(ids[0]), scale=5.0)  # a strongly non-identity teacher
    workspace.record_usage(ids)
    # A fresh candidate is the identity function (B initialized to zero, see
    # Primitive.__init__), so it reproduces the (heavily non-identity) target
    # very poorly.
    candidate = Primitive(999, PrimitiveConfig(d_model=D_MODEL, rank=2))

    current = torch.randn(16, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    report = evaluate_shadow(
        workspace, ids, candidate, current, target_current, current, target_current,
        _default_config(),
    )

    assert not report.task_score_passed
    assert not report.passed
    assert "task_score_below_threshold" in report.failure_reasons


def test_candidate_that_ignores_replay_fails_retention_check() -> None:
    # Two *independent* active transforms (not a redundant pair): their
    # combined delta spans up to rank 8, so a rank-4 candidate distilled with
    # replay_weight=0 (fit only against the current-task batch's specific
    # directions) can match current-task data well while leaving replay-batch
    # directions -- which that fit never saw -- comparatively unreproduced.
    torch.manual_seed(0)
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _perturb(workspace.get(ids[0]), scale=2.0)
    _perturb(workspace.get(ids[1]), scale=-3.0)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids, rank=1, replay_weight=0.0, steps=300)

    current = torch.randn(64, D_MODEL)
    replay = torch.randn(64, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    target_replay = _temporary_output(workspace, ids, replay)
    strict_config = _default_config(max_retention_degradation=0.0)

    report = evaluate_shadow(
        workspace, ids, candidate, current, target_current, replay, target_replay, strict_config
    )

    assert report.retention_degradation > 0.0
    assert not report.retention_passed
    assert "retention_degradation_exceeded" in report.failure_reasons
    assert not report.passed


def test_candidate_not_smaller_than_temporary_fails_size_check() -> None:
    workspace = _allocated_workspace(1, rank=2)
    ids = workspace.ids()
    workspace.record_usage(ids)
    # rank=2 candidate vs a single rank=2 temporary transform: same parameter
    # count, not smaller.
    candidate = Primitive(999, PrimitiveConfig(d_model=D_MODEL, rank=2))
    data = torch.randn(8, D_MODEL)

    report = evaluate_shadow(workspace, ids, candidate, data, data, data, data, _default_config())

    assert not report.size_passed
    assert "candidate_not_materially_smaller" in report.failure_reasons
    assert not report.passed


def test_non_finite_input_fails_and_is_reported() -> None:
    workspace = _allocated_workspace(1, rank=2)
    ids = workspace.ids()
    workspace.record_usage(ids)
    candidate = Primitive(999, PrimitiveConfig(d_model=D_MODEL, rank=1))
    bad = torch.full((4, D_MODEL), float("inf"))
    good = torch.randn(4, D_MODEL)

    report = evaluate_shadow(workspace, ids, candidate, bad, bad, good, good, _default_config())

    assert not report.finite_outputs
    assert "non_finite_output" in report.failure_reasons
    assert not report.passed
    assert report.disagreement_rate == 1.0


def test_evaluate_shadow_never_mutates_workspace_or_candidate() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids)
    candidate_b_before = candidate.b_proj.weight.detach().clone()
    temp_b_before = {tid: workspace.get(tid).b_proj.weight.detach().clone() for tid in ids}

    data = torch.randn(8, D_MODEL)
    target = _temporary_output(workspace, ids, data)
    evaluate_shadow(workspace, ids, candidate, data, target, data, target, _default_config())

    assert workspace.is_allocated
    assert workspace.ids() == ids
    for tid in ids:
        assert torch.equal(workspace.get(tid).b_proj.weight, temp_b_before[tid])
    assert torch.equal(candidate.b_proj.weight, candidate_b_before)


# --- run_shadow_validation: install-and-release action -----------------------


def test_passing_validation_installs_candidate_and_releases_workspace_atomically() -> None:
    torch.manual_seed(1)
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids, candidate_id=42, rank=4)
    bank = PrimitiveBank()

    current = torch.randn(32, D_MODEL)
    replay = torch.randn(32, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    target_replay = _temporary_output(workspace, ids, replay)
    report = run_shadow_validation(
        workspace, ids, candidate, bank, current, target_current, replay, target_replay,
        _default_config(),
    )

    assert report.passed
    assert report.released
    assert not workspace.is_allocated
    assert bank.ids() == [42]
    assert bank.get(42).status == PrimitiveStatus.STABLE
    assert bank.get(42).is_frozen()


def test_failing_validation_preserves_temporary_capacity_and_bank() -> None:
    workspace = _allocated_workspace(1, rank=4)
    ids = workspace.ids()
    _perturb(workspace.get(ids[0]), scale=5.0)
    workspace.record_usage(ids)
    candidate = Primitive(7, PrimitiveConfig(d_model=D_MODEL, rank=2))  # untrained, fails
    bank = PrimitiveBank()

    data = torch.randn(16, D_MODEL)
    target = _temporary_output(workspace, ids, data)
    report = run_shadow_validation(
        workspace, ids, candidate, bank, data, target, data, target, _default_config()
    )

    assert not report.passed
    assert not report.released
    assert workspace.is_allocated
    assert workspace.ids() == ids
    assert bank.ids() == []


def test_run_shadow_validation_raises_and_preserves_workspace_on_id_collision() -> None:
    workspace = _allocated_workspace(2, rank=4)
    ids = workspace.ids()
    _make_redundant_pair(workspace, ids)
    workspace.record_usage(ids)
    candidate = _distilled_candidate(workspace, ids, candidate_id=0, rank=4)
    bank = PrimitiveBank()
    bank.new_primitive(PrimitiveConfig(d_model=D_MODEL, rank=1))  # occupies id 0

    current = torch.randn(32, D_MODEL)
    replay = torch.randn(32, D_MODEL)
    target_current = _temporary_output(workspace, ids, current)
    target_replay = _temporary_output(workspace, ids, replay)
    with pytest.raises(ValueError, match="already exists"):
        run_shadow_validation(
            workspace, ids, candidate, bank, current, target_current, replay, target_replay,
            _default_config(),
        )

    assert workspace.is_allocated
    assert workspace.ids() == ids
