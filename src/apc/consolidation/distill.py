"""First consolidation path: distill temporary transforms (Phase A Milestone A7 / Task 010).

Implements the first few steps of the consolidation pipeline from
`docs/design-docs/ARCHITECTURE.md` section 10:

    1. measure activity of temporary transforms;
    2. drop inactive transforms;
    3. collect input/output pairs at each active temporary transform over
       replay + current-task data;
    ...
    5. fit a smaller low-rank transform to reproduce the combined residual
       delta;
    6. distill candidate transform(s) on held-out data;

Steps 4 (functional-similarity clustering for >1 remaining transform), 7
(causal-necessity ablation) and 8 (register for SHADOW) are out of scope for
this task -- per the "minimal first version" note in the architecture doc,
`consolidate` below distills every active temporary transform in a
`PlasticWorkspace` into a single candidate `Primitive`, which is the
simplest version of the pipeline that can produce a real candidate for Task
011 (shadow validation) to consume.

Design choice not pinned down by the architecture doc (recorded here per
AGENTS.md workflow rather than hidden): a `Primitive`'s residual delta
`B(A(h))` is a plain linear map of `h` (no bias, no nonlinearity, no
task-conditioning) -- see `apc.primitives.primitive.Primitive`. Consolidation
has not been wired into the stable core/router yet (that integration does
not exist anywhere in this codebase as of Task 010), so there is no
per-example gate signal available here to decide which hidden states a
temporary transform "cares about". Instead, `consolidate` distills the
teacher's combined delta (the literal sum of active transforms' `B(A(h))`,
gate=1) against *both* a batch of current-task hidden states and a batch of
prior-task replay hidden states, weighted by `current_task_weight` /
`replay_weight`, and reports a reproduction score for each split. This is a
faithful implementation of section 10's "the compression objective should
include both current-task imitation and prior-task retention" given the
ingredients available at this scope; it intentionally says nothing about
which primitive activates for a given input, only how well a smaller
transform can reproduce what the temporary capacity currently computes.

Per ADR-0005 / the Task 010 acceptance criterion, `consolidate` never
mutates `workspace`: it does not freeze, disable, or release any temporary
transform. The candidate it returns is a free-floating `Primitive`, not yet
registered in any bank or workspace -- Task 011's shadow validation decides
whether to promote it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

_EPS = 1e-9


@dataclass(frozen=True)
class ConsolidationConfig:
    """Explicit, serializable configuration for one `consolidate` call."""

    candidate_rank: int
    activity_threshold: int = 1
    steps: int = 300
    lr: float = 1e-2
    weight_decay: float = 0.0
    current_task_weight: float = 1.0
    replay_weight: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_rank < 1:
            raise ValueError(f"candidate_rank must be >= 1, got {self.candidate_rank}")
        if self.activity_threshold < 0:
            raise ValueError(
                f"activity_threshold must be >= 0, got {self.activity_threshold}"
            )
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.current_task_weight < 0:
            raise ValueError(
                f"current_task_weight must be >= 0, got {self.current_task_weight}"
            )
        if self.replay_weight < 0:
            raise ValueError(f"replay_weight must be >= 0, got {self.replay_weight}")
        if self.current_task_weight == 0 and self.replay_weight == 0:
            raise ValueError(
                "at least one of current_task_weight/replay_weight must be > 0"
            )


@dataclass(frozen=True)
class ConsolidationReport:
    """Result of one `consolidate` call.

    `compression_ratio` is `candidate_parameter_count / temporary_parameter_count`
    (a fraction < 1: how much smaller the candidate is, per Task 010's
    "candidate is smaller" acceptance criterion). `task_imitation_score` and
    `prior_task_replay_score` are the two scores Task 010 requires be
    reported: how well the candidate reproduces the teacher's combined delta
    on current-task hidden states and on replay hidden states respectively,
    each a reproduction score in `[0, 1]` (see `_reproduction_score`).
    """

    active_ids: tuple[int, ...]
    inactive_ids: tuple[int, ...]
    candidate_parameter_count: int
    temporary_parameter_count: int
    compression_ratio: float
    task_imitation_score: float
    prior_task_replay_score: float
    steps: int
    final_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_ids": list(self.active_ids),
            "inactive_ids": list(self.inactive_ids),
            "candidate_parameter_count": self.candidate_parameter_count,
            "temporary_parameter_count": self.temporary_parameter_count,
            "compression_ratio": self.compression_ratio,
            "task_imitation_score": self.task_imitation_score,
            "prior_task_replay_score": self.prior_task_replay_score,
            "steps": self.steps,
            "final_loss": self.final_loss,
        }


def measure_activity(
    workspace: PlasticWorkspace, activity_threshold: int
) -> tuple[list[int], list[int]]:
    """Split `workspace`'s temporary ids into active/inactive (architecture
    doc section 10, steps 1-2).

    A transform is active if it is enabled and its `usage_count` is
    `>= activity_threshold`; a disabled transform is always inactive
    regardless of usage count. Returns `(active_ids, inactive_ids)`, both
    sorted ascending.
    """
    active: list[int] = []
    inactive: list[int] = []
    for transform_id in workspace.ids():
        transform = workspace.get(transform_id)
        if transform.enabled and transform.usage_count >= activity_threshold:
            active.append(transform_id)
        else:
            inactive.append(transform_id)
    return active, inactive


def _combined_teacher_delta(transforms: list[Primitive], h: torch.Tensor) -> torch.Tensor:
    """Sum of every transform's residual delta `B(A(h))` (gate=1), detached.

    This is the frozen teacher signal the candidate distills -- the active
    temporary transforms are never trained during consolidation, only read.
    """
    with torch.no_grad():
        total = torch.zeros_like(h)
        for transform in transforms:
            total = total + transform.b_proj(transform.a_proj(h))
        return total


def _reproduction_score(candidate_delta: torch.Tensor, teacher_delta: torch.Tensor) -> float:
    """`1 - relative L2 error` between `candidate_delta` and `teacher_delta`,
    clamped to `[0, 1]`. A freshly-initialized candidate (zero delta, see
    `Primitive.__init__`) scores 0 against any non-trivial teacher; a
    candidate that reproduces the teacher's output exactly scores 1.
    """
    with torch.no_grad():
        diff_norm = (candidate_delta - teacher_delta).norm()
        teacher_norm = teacher_delta.norm()
        if teacher_norm <= _EPS:
            return 1.0 if diff_norm <= _EPS else 0.0
        return float(max(0.0, 1.0 - (diff_norm / teacher_norm).item()))


def consolidate(
    workspace: PlasticWorkspace,
    current_task_hidden_states: torch.Tensor,
    replay_hidden_states: torch.Tensor,
    config: ConsolidationConfig,
    *,
    candidate_id: int = 0,
    created_at_task: int = 0,
) -> tuple[Primitive, ConsolidationReport]:
    """Distill `workspace`'s active temporary transforms into one candidate
    `Primitive` of rank `config.candidate_rank`.

    `current_task_hidden_states`/`replay_hidden_states` are `[N, d_model]`/
    `[M, d_model]` batches of hidden states (`N` and `M` may differ) drawn
    from the current task and from replay of prior tasks respectively --
    generic tensors, not tied to any particular data source, matching the
    abstraction level of `apc.primitives.primitive.Primitive.forward`.

    Never mutates `workspace` (ADR-0005 / Task 010 acceptance: "temporary
    module is not deleted"). Raises `ValueError` if `workspace` has no
    allocated capacity, if no transform meets `config.activity_threshold`,
    if the hidden-state batches don't share `workspace`'s `d_model`, or if
    the resulting candidate would not be smaller than the temporary capacity
    it replaces.
    """
    if not workspace.is_allocated:
        raise ValueError("workspace has no allocated temporary transforms to consolidate")
    if current_task_hidden_states.ndim != 2:
        raise ValueError(
            "current_task_hidden_states must be 2D [N, d_model], got shape "
            f"{tuple(current_task_hidden_states.shape)}"
        )
    if replay_hidden_states.ndim != 2:
        raise ValueError(
            f"replay_hidden_states must be 2D [M, d_model], got shape "
            f"{tuple(replay_hidden_states.shape)}"
        )
    d_model = current_task_hidden_states.shape[-1]
    if replay_hidden_states.shape[-1] != d_model:
        raise ValueError(
            "current_task_hidden_states and replay_hidden_states must share d_model, got "
            f"{d_model} and {replay_hidden_states.shape[-1]}"
        )

    active_ids, inactive_ids = measure_activity(workspace, config.activity_threshold)
    if not active_ids:
        raise ValueError(
            f"No temporary transforms met activity_threshold={config.activity_threshold}; "
            "nothing to consolidate"
        )
    active_transforms = workspace.get_many(active_ids)
    for transform in active_transforms:
        if transform.config.d_model != d_model:
            raise ValueError(
                f"Temporary transform {transform.primitive_id} has d_model="
                f"{transform.config.d_model}, expected {d_model}"
            )

    temporary_parameter_count = sum(t.num_parameters() for t in active_transforms)

    if config.seed is not None:
        set_seed(config.seed)

    candidate = Primitive(
        candidate_id,
        PrimitiveConfig(d_model=d_model, rank=config.candidate_rank),
        status=PrimitiveStatus.CANDIDATE,
        created_at_task=created_at_task,
        metadata={"consolidated_from": list(active_ids)},
    )
    candidate_parameter_count = candidate.num_parameters()
    if candidate_parameter_count >= temporary_parameter_count:
        raise ValueError(
            f"Candidate parameter count ({candidate_parameter_count}) is not smaller than "
            f"the active temporary parameter count ({temporary_parameter_count}); "
            "reduce config.candidate_rank."
        )

    teacher_current = _combined_teacher_delta(active_transforms, current_task_hidden_states)
    teacher_replay = _combined_teacher_delta(active_transforms, replay_hidden_states)

    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    final_loss = float("nan")
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        candidate_current = candidate.b_proj(candidate.a_proj(current_task_hidden_states))
        candidate_replay = candidate.b_proj(candidate.a_proj(replay_hidden_states))
        loss = config.current_task_weight * F.mse_loss(
            candidate_current, teacher_current
        ) + config.replay_weight * F.mse_loss(candidate_replay, teacher_replay)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    with torch.no_grad():
        final_candidate_current = candidate.b_proj(candidate.a_proj(current_task_hidden_states))
        final_candidate_replay = candidate.b_proj(candidate.a_proj(replay_hidden_states))
    task_imitation_score = _reproduction_score(final_candidate_current, teacher_current)
    prior_task_replay_score = _reproduction_score(final_candidate_replay, teacher_replay)

    report = ConsolidationReport(
        active_ids=tuple(active_ids),
        inactive_ids=tuple(inactive_ids),
        candidate_parameter_count=candidate_parameter_count,
        temporary_parameter_count=temporary_parameter_count,
        compression_ratio=candidate_parameter_count / temporary_parameter_count,
        task_imitation_score=task_imitation_score,
        prior_task_replay_score=prior_task_replay_score,
        steps=config.steps,
        final_loss=final_loss,
    )
    return candidate, report
