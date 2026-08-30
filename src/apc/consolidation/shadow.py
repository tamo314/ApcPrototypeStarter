"""Shadow validation and release (Phase A Milestone A8 / Task 011).

Implements `docs/design-docs/ARCHITECTURE.md` section 11: for a configured
batch of current-task and replay data, compare the temporary solution's
output, the consolidated candidate's output, and ground truth, then decide
whether the candidate may be promoted and the temporary capacity released.

Release criteria (section 11, "Release criteria should include"):

    1. candidate reaches >= `task_score_ratio_threshold` of the temporary
       solution's task score;
    2. prior-task degradation <= `max_retention_degradation`;
    3. no invariant failures (non-finite output);
    4. candidate persistent parameter cost is materially smaller than
       temporary peak capacity.

Per ADR-0005 and Task 011's acceptance criteria, this module never releases
`workspace` or registers `candidate` into `bank` unless every criterion
passes: `evaluate_shadow` is a pure comparison (no mutation, matching
`apc.consolidation.distill.consolidate`'s own no-mutation contract) and
`run_shadow_validation` performs the install-and-release action only when
`evaluate_shadow` reports `passed=True`. A failing report leaves `workspace`
and `bank` untouched, which is exactly Task 011's "failing shadow test
preserves temporary capacity".

Design choices not pinned down by the architecture doc (recorded here per
AGENTS.md workflow rather than hidden):

- "ground truth" is generic target hidden states (`current_task_targets` /
  `replay_targets`), matching the abstraction level of
  `apc.consolidation.distill` (generic `[N, d_model]` tensors, not tied to
  any particular data source or to the stable core/router, which are not
  wired to consolidation yet). "Solution output" for a set of transforms is
  `h + sum(B_i(A_i(h)))` -- the same combined-residual construction
  `distill._combined_teacher_delta` uses, but including the base `h` term
  since here the comparison is against a full ground-truth output, not
  against another transform's delta.
- "task score" and "prior-task retention score" both use the same
  `1 - relative L2 error` reproduction-score formula as
  `distill._reproduction_score`, applied to (solution output, ground truth)
  instead of (candidate delta, teacher delta). It is the natural analogue at
  this abstraction level: 1.0 for an exact match, clamped to 0.0 below that.
- shadow disagreement rate (an Experiment Plan headline metric, not one of
  section 11's four release gates) is computed per-example as "does this
  example's output fall within `correctness_tolerance` relative error of
  ground truth" for the temporary solution vs the candidate, over the
  concatenation of the current-task and replay batches; the rate is the
  fraction of examples where that per-example correctness call disagrees.
  It is reported for logging but does not gate `passed`.
- registering the candidate into `bank` uses `bank.add_primitive`, so it
  keeps whatever `primitive_id` the candidate already has (e.g. from
  `distill.consolidate`). `run_shadow_validation` does not renumber it --
  callers are responsible for giving the candidate an id that does not
  already exist in `bank` (e.g. by consolidating with a `candidate_id` drawn
  from the bank's id space). If it does collide, `bank.add_primitive` raises
  `ValueError` and, per the "no partial release" contract above,
  `workspace.release()` is never reached -- temporary capacity is preserved.
- a promoted candidate is set to `PrimitiveStatus.STABLE` and frozen
  (ADR-0004: stable primitives are frozen in Phase A).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch

from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveStatus

_EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class ShadowValidationConfig:
    """Explicit, serializable thresholds for one shadow-validation call."""

    task_score_ratio_threshold: float = 0.95
    max_retention_degradation: float = 0.02
    max_size_ratio: float = 1.0
    correctness_tolerance: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.task_score_ratio_threshold <= 1.0:
            raise ValueError(
                "task_score_ratio_threshold must be in [0, 1], got "
                f"{self.task_score_ratio_threshold}"
            )
        if self.max_retention_degradation < 0:
            raise ValueError(
                f"max_retention_degradation must be >= 0, got {self.max_retention_degradation}"
            )
        if self.max_size_ratio <= 0:
            raise ValueError(f"max_size_ratio must be > 0, got {self.max_size_ratio}")
        if self.correctness_tolerance < 0:
            raise ValueError(
                f"correctness_tolerance must be >= 0, got {self.correctness_tolerance}"
            )


@dataclasses.dataclass(frozen=True)
class ShadowValidationReport:
    """Result of one `evaluate_shadow` (and, if promoted, `run_shadow_validation`) call."""

    passed: bool
    finite_outputs: bool
    task_score_passed: bool
    retention_passed: bool
    size_passed: bool
    temporary_task_score: float
    candidate_task_score: float
    task_score_ratio: float
    temporary_retention_score: float
    candidate_retention_score: float
    retention_degradation: float
    temporary_parameter_count: int
    candidate_parameter_count: int
    size_ratio: float
    disagreement_rate: float
    failure_reasons: tuple[str, ...]
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "finite_outputs": self.finite_outputs,
            "task_score_passed": self.task_score_passed,
            "retention_passed": self.retention_passed,
            "size_passed": self.size_passed,
            "temporary_task_score": self.temporary_task_score,
            "candidate_task_score": self.candidate_task_score,
            "task_score_ratio": self.task_score_ratio,
            "temporary_retention_score": self.temporary_retention_score,
            "candidate_retention_score": self.candidate_retention_score,
            "retention_degradation": self.retention_degradation,
            "temporary_parameter_count": self.temporary_parameter_count,
            "candidate_parameter_count": self.candidate_parameter_count,
            "size_ratio": self.size_ratio,
            "disagreement_rate": self.disagreement_rate,
            "failure_reasons": list(self.failure_reasons),
            "released": self.released,
        }


def _combined_output(transforms: list[Primitive], h: torch.Tensor) -> torch.Tensor:
    """`h + sum(B_i(A_i(h)))` over `transforms` (gate=1), detached.

    Neither the temporary transforms nor the candidate are trained here --
    shadow validation only reads what consolidation already produced.
    """
    with torch.no_grad():
        total = h
        for transform in transforms:
            total = total + transform.b_proj(transform.a_proj(h))
        return total


def _score(output: torch.Tensor, target: torch.Tensor) -> float:
    """`1 - relative L2 error` between `output` and `target`, clamped to
    `[0, 1]` -- the same reproduction-score formula as
    `apc.consolidation.distill._reproduction_score`, applied to (solution
    output, ground truth) instead of (candidate delta, teacher delta)."""
    with torch.no_grad():
        diff_norm = (output - target).norm()
        target_norm = target.norm()
        if target_norm <= _EPS:
            return 1.0 if diff_norm <= _EPS else 0.0
        return float(max(0.0, 1.0 - (diff_norm / target_norm).item()))


def _row_correct(output: torch.Tensor, target: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Per-row boolean: is `output`'s relative L2 error to `target` `<= tolerance`?"""
    with torch.no_grad():
        diff_norm = (output - target).norm(dim=-1)
        target_norm = target.norm(dim=-1)
        safe_target_norm = torch.clamp(target_norm, min=_EPS)
        relative_error = diff_norm / safe_target_norm
        correct = relative_error <= tolerance
        # A near-zero target that is matched near-exactly is trivially correct
        # even though dividing by its (clamped) norm would otherwise inflate
        # the relative error.
        correct = correct | ((target_norm <= _EPS) & (diff_norm <= _EPS))
        return correct


def _validate_batch(name: str, tensor: torch.Tensor, d_model: int) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D [N, d_model], got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] != d_model:
        raise ValueError(
            f"{name} has d_model={tensor.shape[-1]}, expected {d_model}"
        )


def evaluate_shadow(
    workspace: PlasticWorkspace,
    active_ids: Sequence[int],
    candidate: Primitive,
    current_task_inputs: torch.Tensor,
    current_task_targets: torch.Tensor,
    replay_inputs: torch.Tensor,
    replay_targets: torch.Tensor,
    config: ShadowValidationConfig,
) -> ShadowValidationReport:
    """Compare the temporary solution (`workspace`'s `active_ids` transforms,
    combined) against `candidate`, on both current-task and replay batches,
    against ground truth. Never mutates `workspace`, `candidate`, or any
    primitive bank -- see module docstring.

    `active_ids` are typically `distill.measure_activity`'s (or
    `distill.ConsolidationReport`'s) active transform ids: the same
    temporary transforms `candidate` was distilled from. Raises `ValueError`
    if `active_ids` is empty, if any input/target pair's shapes disagree, or
    if `d_model` is inconsistent across `candidate`, the active transforms,
    and the four data tensors.
    """
    if not active_ids:
        raise ValueError("active_ids must be non-empty")

    d_model = candidate.config.d_model
    active_transforms = workspace.get_many(active_ids)
    for transform in active_transforms:
        if transform.config.d_model != d_model:
            raise ValueError(
                f"Temporary transform {transform.primitive_id} has d_model="
                f"{transform.config.d_model}, expected candidate's d_model={d_model}"
            )

    _validate_batch("current_task_inputs", current_task_inputs, d_model)
    _validate_batch("current_task_targets", current_task_targets, d_model)
    _validate_batch("replay_inputs", replay_inputs, d_model)
    _validate_batch("replay_targets", replay_targets, d_model)
    if current_task_inputs.shape != current_task_targets.shape:
        raise ValueError(
            "current_task_inputs and current_task_targets must share shape, got "
            f"{tuple(current_task_inputs.shape)} and {tuple(current_task_targets.shape)}"
        )
    if replay_inputs.shape != replay_targets.shape:
        raise ValueError(
            "replay_inputs and replay_targets must share shape, got "
            f"{tuple(replay_inputs.shape)} and {tuple(replay_targets.shape)}"
        )

    temporary_current = _combined_output(active_transforms, current_task_inputs)
    candidate_current = _combined_output([candidate], current_task_inputs)
    temporary_replay = _combined_output(active_transforms, replay_inputs)
    candidate_replay = _combined_output([candidate], replay_inputs)

    finite_outputs = bool(
        torch.isfinite(temporary_current).all()
        and torch.isfinite(candidate_current).all()
        and torch.isfinite(temporary_replay).all()
        and torch.isfinite(candidate_replay).all()
    )

    temporary_task_score = _score(temporary_current, current_task_targets)
    candidate_task_score = _score(candidate_current, current_task_targets)
    temporary_retention_score = _score(temporary_replay, replay_targets)
    candidate_retention_score = _score(candidate_replay, replay_targets)

    task_score_ratio = candidate_task_score / max(temporary_task_score, _EPS)
    retention_degradation = temporary_retention_score - candidate_retention_score

    temporary_parameter_count = sum(t.num_parameters() for t in active_transforms)
    candidate_parameter_count = candidate.num_parameters()
    size_ratio = candidate_parameter_count / max(temporary_parameter_count, 1)

    task_score_passed = candidate_task_score >= config.task_score_ratio_threshold * (
        temporary_task_score
    )
    retention_passed = retention_degradation <= config.max_retention_degradation
    size_passed = (
        candidate_parameter_count < temporary_parameter_count
        and candidate_parameter_count <= config.max_size_ratio * temporary_parameter_count
    )

    failure_reasons: list[str] = []
    if not finite_outputs:
        failure_reasons.append("non_finite_output")
    if not task_score_passed:
        failure_reasons.append("task_score_below_threshold")
    if not retention_passed:
        failure_reasons.append("retention_degradation_exceeded")
    if not size_passed:
        failure_reasons.append("candidate_not_materially_smaller")

    passed = finite_outputs and task_score_passed and retention_passed and size_passed

    if finite_outputs:
        current_correct_temp = _row_correct(
            temporary_current, current_task_targets, config.correctness_tolerance
        )
        current_correct_cand = _row_correct(
            candidate_current, current_task_targets, config.correctness_tolerance
        )
        replay_correct_temp = _row_correct(
            temporary_replay, replay_targets, config.correctness_tolerance
        )
        replay_correct_cand = _row_correct(
            candidate_replay, replay_targets, config.correctness_tolerance
        )
        temp_correct = torch.cat([current_correct_temp, replay_correct_temp])
        cand_correct = torch.cat([current_correct_cand, replay_correct_cand])
        disagreement_rate = float((temp_correct != cand_correct).float().mean().item())
    else:
        disagreement_rate = 1.0

    return ShadowValidationReport(
        passed=passed,
        finite_outputs=finite_outputs,
        task_score_passed=task_score_passed,
        retention_passed=retention_passed,
        size_passed=size_passed,
        temporary_task_score=temporary_task_score,
        candidate_task_score=candidate_task_score,
        task_score_ratio=task_score_ratio,
        temporary_retention_score=temporary_retention_score,
        candidate_retention_score=candidate_retention_score,
        retention_degradation=retention_degradation,
        temporary_parameter_count=temporary_parameter_count,
        candidate_parameter_count=candidate_parameter_count,
        size_ratio=size_ratio,
        disagreement_rate=disagreement_rate,
        failure_reasons=tuple(failure_reasons),
        released=False,
    )


def run_shadow_validation(
    workspace: PlasticWorkspace,
    active_ids: Sequence[int],
    candidate: Primitive,
    bank: PrimitiveBank,
    current_task_inputs: torch.Tensor,
    current_task_targets: torch.Tensor,
    replay_inputs: torch.Tensor,
    replay_targets: torch.Tensor,
    config: ShadowValidationConfig,
) -> ShadowValidationReport:
    """Run `evaluate_shadow`, then act on it: promote and release on pass,
    change nothing on failure (Task 011 acceptance criteria).

    On pass: `candidate` is added to `bank` (raising `ValueError` first if
    its id already exists there -- in which case `workspace` is left
    untouched, see module docstring), set to `PrimitiveStatus.STABLE`,
    frozen (ADR-0004), and `workspace.release()` is called to drop *all* of
    its temporary capacity (not just `active_ids` -- the inactive transforms
    consolidation dropped are released too, per architecture doc section 10
    step 2). The returned report has `released=True`.

    On failure: `workspace` and `bank` are returned unchanged; the returned
    report has `released=False`.
    """
    report = evaluate_shadow(
        workspace,
        active_ids,
        candidate,
        current_task_inputs,
        current_task_targets,
        replay_inputs,
        replay_targets,
        config,
    )
    if not report.passed:
        return report

    bank.add_primitive(candidate)
    bank.set_status(candidate.primitive_id, PrimitiveStatus.STABLE)
    bank.freeze(candidate.primitive_id)
    workspace.release()

    return dataclasses.replace(report, released=True)
