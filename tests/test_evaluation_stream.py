from __future__ import annotations

import torch

from apc.core.execution import ensure_null_key
from apc.evaluation.stream import (
    LABEL_KNOWN,
    EventExamplePool,
    StreamEvent,
    build_task_generators,
    calibrate_router,
    train_until_plateau,
)
from apc.primitives.router import Router, RouterConfig

# --- build_task_generators / EventExamplePool ----------------------------


def test_build_task_generators_has_one_generator_per_novel_operation() -> None:
    main, novel = build_task_generators(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        max_depth=2,
        novel_composition_fraction=0.3,
        novel_operation_names=("SORT", "REVERSE"),
    )
    assert set(novel) == {"SORT", "REVERSE"}
    assert main.seed == 0


def test_event_example_pool_gives_disjoint_slices_for_repeated_events() -> None:
    main, novel = build_task_generators(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        max_depth=2,
        novel_composition_fraction=0.3,
        novel_operation_names=("SORT",),
    )
    pool = EventExamplePool()
    event = StreamEvent(LABEL_KNOWN)
    first = pool.pool_for_event(event, main, novel, 5)
    second = pool.pool_for_event(event, main, novel, 5)
    assert len(first) == 5
    assert len(second) == 5
    assert first != second


def test_event_example_pool_tracks_offsets_independently_per_key() -> None:
    main, novel = build_task_generators(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        max_depth=2,
        novel_composition_fraction=0.3,
        novel_operation_names=("SORT",),
    )
    pool = EventExamplePool()
    known = pool.pool_for_event(StreamEvent(LABEL_KNOWN), main, novel, 3)
    novel_op = pool.pool_for_event(StreamEvent("N", operation_name="SORT"), main, novel, 3)
    # Different keys draw from different pools/splits, starting at offset 0
    # each -- interleaving one key's calls must not perturb the other's.
    known_again = pool.pool_for_event(StreamEvent(LABEL_KNOWN), main, novel, 3)
    assert known != known_again
    assert len(novel_op) == 3


# --- train_until_plateau --------------------------------------------------


def test_train_until_plateau_stops_early_once_plateaued_and_above_min_accuracy() -> None:
    accuracies = iter([0.5, 0.9, 0.9, 0.9])
    calls = {"steps": 0}

    def step_fn() -> None:
        calls["steps"] += 1

    def eval_fn() -> float:
        return next(accuracies)

    steps, accuracy = train_until_plateau(
        step_fn,
        eval_fn,
        max_steps=100,
        eval_every=1,
        improvement_margin=0.01,
        plateau_patience=2,
        min_accuracy=0.8,
    )
    assert steps == 4
    assert calls["steps"] == 4
    assert accuracy == 0.9


def test_train_until_plateau_stops_at_max_steps_if_never_above_min_accuracy() -> None:
    def step_fn() -> None:
        pass

    def eval_fn() -> float:
        return 0.5

    steps, accuracy = train_until_plateau(
        step_fn,
        eval_fn,
        max_steps=5,
        eval_every=1,
        improvement_margin=0.01,
        plateau_patience=2,
        min_accuracy=0.9,
    )
    assert steps == 5
    assert accuracy == 0.5


def test_train_until_plateau_evaluates_only_every_eval_every_steps() -> None:
    eval_calls = {"count": 0}

    def step_fn() -> None:
        pass

    def eval_fn() -> float:
        eval_calls["count"] += 1
        return 1.0

    steps, accuracy = train_until_plateau(
        step_fn,
        eval_fn,
        max_steps=10,
        eval_every=5,
        improvement_margin=0.01,
        plateau_patience=1,
        min_accuracy=0.5,
    )
    # First eval at step 5: accuracy=1.0 improves over best=0.0, plateau_steps
    # resets to 0, so it does not stop yet; the cap at step=10 forces a
    # second (final) evaluation and return.
    assert steps == 10
    assert eval_calls["count"] == 2
    assert accuracy == 1.0


# --- calibrate_router -------------------------------------------------


def test_calibrate_router_runs_and_restores_usage_count() -> None:
    router = Router(RouterConfig(d_model=4))
    ensure_null_key(router)
    router.add_primitive_key(0)
    router.usage_count[0] = 7

    anchors = {0: torch.randn(6, 4)}
    background = torch.randn(3, 4)
    calibrate_router(router, anchors, background, steps=5, lr=0.1)

    assert router.usage_count[0] == 7


def test_calibrate_router_handles_empty_background() -> None:
    router = Router(RouterConfig(d_model=4))
    ensure_null_key(router)
    router.add_primitive_key(0)

    anchors = {0: torch.randn(4, 4)}
    background = torch.zeros(0, 4)
    calibrate_router(router, anchors, background, steps=3, lr=0.1)  # must not raise
