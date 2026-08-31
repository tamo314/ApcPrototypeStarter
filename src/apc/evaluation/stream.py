"""Shared task-stream plumbing for the sequential benchmark and its
baselines (Phase A Milestone A9 / Tasks 012-013).

Factored out of `apc.evaluation.sequential_benchmark` (Task 012) so that
Task 013's B0-B3 baselines and the full APC loop (B4) draw from *the same*
code path rather than a parallel reimplementation that could silently
drift: identical `StreamEvent`/`default_task_stream` construction,
identical per-event example pooling (`EventExamplePool`, disjoint and
deterministic given the same seed/config and event order -- see its
docstring), and identical router-calibration/plateau-training-loop math
(`calibrate_router` / `train_until_plateau`) wherever a baseline needs
them. `apc.evaluation.sequential_benchmark` re-exports the stream-shape
names (`StreamEvent`, `LABEL_*`, `default_task_stream`) for backward
compatibility -- nothing outside this module needs to change which module
it imports them from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from apc.core.execution import NULL_PRIMITIVE_ID
from apc.environments.generator import (
    NOVEL_COMPOSITION_SPLIT,
    NOVEL_OPERATION_SPLIT,
    Example,
    TaskGenerator,
)
from apc.primitives.router import Router

LABEL_KNOWN = "K"
LABEL_NOVEL_COMPOSITION = "C"
LABEL_NOVEL_OPERATION = "N"
LABEL_RECURRENCE = "R"
_LABELS = (LABEL_KNOWN, LABEL_NOVEL_COMPOSITION, LABEL_NOVEL_OPERATION, LABEL_RECURRENCE)


@dataclass(frozen=True)
class StreamEvent:
    """One task-stream entry. `operation_name` selects which novel
    operation an `N`/`R` event drills; ignored for `K`/`C`."""

    label: str
    operation_name: str | None = None

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {_LABELS}, got {self.label!r}")
        if self.label in (LABEL_NOVEL_OPERATION, LABEL_RECURRENCE) and not self.operation_name:
            raise ValueError(f"{self.label} events require a non-empty operation_name")


def default_task_stream(novel_operation_names: tuple[str, ...]) -> tuple[StreamEvent, ...]:
    """`K C N(op0) K C N(op1) K C ... R(op0) R(op1) ...` -- a reduced form
    of Milestone A9's example stream: one learn/consolidate/release cycle
    per novel operation, each preceded by known/composition checks,
    followed by a recurrence of every novel operation to measure reuse.
    """
    if not novel_operation_names:
        raise ValueError("novel_operation_names must be non-empty")
    events = [StreamEvent(LABEL_KNOWN), StreamEvent(LABEL_NOVEL_COMPOSITION)]
    for op in novel_operation_names:
        events.append(StreamEvent(LABEL_NOVEL_OPERATION, operation_name=op))
        events.append(StreamEvent(LABEL_KNOWN))
        events.append(StreamEvent(LABEL_NOVEL_COMPOSITION))
    for op in novel_operation_names:
        events.append(StreamEvent(LABEL_RECURRENCE, operation_name=op))
    return tuple(events)


def build_task_generators(
    *,
    seed: int,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    max_depth: int,
    novel_composition_fraction: float,
    novel_operation_names: tuple[str, ...],
) -> tuple[TaskGenerator, dict[str, TaskGenerator]]:
    """The main (known + held-out-composition) generator, plus one
    per-novel-operation generator -- identical construction for every
    baseline given the same config, so `TaskGenerator`'s own by-seed
    determinism is what makes their data comparable."""
    main_generator = TaskGenerator(
        seed=seed,
        vocab_size=vocab_size,
        sequence_length_range=sequence_length_range,
        max_depth=max_depth,
        novel_composition_fraction=novel_composition_fraction,
    )
    novel_generators = {
        op: TaskGenerator(
            seed=seed,
            vocab_size=vocab_size,
            sequence_length_range=sequence_length_range,
            max_depth=max_depth,
            novel_composition_fraction=novel_composition_fraction,
            novel_operation_names=(op,),
        )
        for op in novel_operation_names
    }
    return main_generator, novel_generators


class EventExamplePool:
    """Deterministic, disjoint example slicing per `(label, operation_name)`
    key across repeated events in one stream.

    Stateful only in its offsets: each call advances the slice window for
    that key so repeated `K` (or `R(op)`) events never see the same
    examples twice, while remaining fully reproducible given the same
    generators/config and the same event-processing order -- which is
    exactly what every baseline (Task 013) and the full APC runner (Task
    012) share, so a fresh `EventExamplePool` per run, fed the same
    generators built from the same config, yields identical per-event data
    across B0-B4.
    """

    def __init__(self) -> None:
        self._offsets: dict[tuple[str, str | None], int] = {}

    def pool_for_event(
        self,
        event: StreamEvent,
        main_generator: TaskGenerator,
        novel_generators: dict[str, TaskGenerator],
        count: int,
    ) -> list[Example]:
        key = (event.label, event.operation_name)
        start = self._offsets.get(key, 0)
        self._offsets[key] = start + count
        if event.label == LABEL_KNOWN:
            generator, split = main_generator, "test"
        elif event.label == LABEL_NOVEL_COMPOSITION:
            generator, split = main_generator, NOVEL_COMPOSITION_SPLIT
        else:
            assert event.operation_name is not None  # enforced by StreamEvent.__post_init__
            generator, split = novel_generators[event.operation_name], NOVEL_OPERATION_SPLIT
        pool = generator.generate(start + count, split)
        return pool[start : start + count]


def calibrate_router(
    router: Router,
    anchors: dict[int, torch.Tensor],
    background: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> None:
    """Fit `router.query_proj` and every key in `anchors` (plus the
    permanent null candidate, `apc.core.execution.NULL_PRIMITIVE_ID`) as a
    small classifier: each `anchors[pid]` batch of hidden states should
    route to `pid`; `background` (e.g. replay hidden states, possibly a
    zero-row tensor when no prior data exists yet) should route to null.

    Extracted from `apc.evaluation.sequential_benchmark`'s Task 012
    `_calibrate_router` so Task 013's baselines that also need it
    (`apc.evaluation.baselines.B1Runner`'s one-time seed cycle) share the
    exact same math instead of a parallel reimplementation. Restores
    `router.usage_count` afterward so calibration never pollutes real
    usage statistics.
    """
    ids = sorted(anchors)
    classes = [NULL_PRIMITIVE_ID, *ids]
    saved_usage = dict(router.usage_count)
    params = list(router.query_proj.parameters()) + [router.key_parameter(c) for c in classes]
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        if background.shape[0] > 0:
            router_out = router(background, classes)
            target = torch.zeros(background.shape[0], dtype=torch.long)
            loss = loss + F.cross_entropy(router_out.probs, target)
        for class_idx, pid in enumerate(ids, start=1):
            router_out = router(anchors[pid], classes)
            target = torch.full((anchors[pid].shape[0],), class_idx, dtype=torch.long)
            loss = loss + F.cross_entropy(router_out.probs, target)
        loss.backward()
        optimizer.step()
    router.usage_count = saved_usage


def train_until_plateau(
    step_fn: Callable[[], None],
    eval_fn: Callable[[], float],
    *,
    max_steps: int,
    eval_every: int,
    improvement_margin: float,
    plateau_patience: int,
    min_accuracy: float,
) -> tuple[int, float]:
    """Run `step_fn` repeatedly, checking `eval_fn` every `eval_every`
    steps, stopping once accuracy has plateaued (no improvement greater
    than `improvement_margin` for `plateau_patience` consecutive
    evaluations) *and* has cleared `min_accuracy`, or once `max_steps` is
    reached regardless.

    Mirrors `apc.evaluation.sequential_benchmark._SequentialBenchmarkRunner.
    _train_plastic_until_promoted`'s stopping rule (itself driven by
    `apc.meta.controller.Controller._decide_plastic`) without the
    controller/bank/workspace machinery that method also handles -- the
    baselines in `apc.evaluation.baselines` (Task 013) have no controller
    state at all, only this plateau-or-cap training budget, so the two
    implementations are intentionally kept separate rather than forcing a
    shared abstraction onto genuinely different control flow.

    Returns `(steps_taken, final_accuracy)`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    best = 0.0
    plateau_steps = 0
    accuracy = 0.0
    step = 0
    while True:
        step += 1
        step_fn()
        at_cap = step >= max_steps
        if step % eval_every == 0 or at_cap:
            accuracy = eval_fn()
            improved = accuracy > best + improvement_margin
            best = max(best, accuracy)
            plateau_steps = 0 if improved else plateau_steps + 1
            if plateau_steps >= plateau_patience and accuracy >= min_accuracy:
                return step, accuracy
        if at_cap:
            return step, accuracy
