from __future__ import annotations

import pytest
import torch

from apc.consolidation.distill import ConsolidationConfig
from apc.consolidation.shadow import ShadowValidationConfig
from apc.evaluation.sequential_benchmark import (
    LABEL_KNOWN,
    LABEL_NOVEL_COMPOSITION,
    LABEL_NOVEL_OPERATION,
    LABEL_RECURRENCE,
    PlasticTrainingConfig,
    PretrainConfig,
    RouterCalibrationConfig,
    SequentialBenchmarkConfig,
    SequentialBenchmarkReport,
    StreamEvent,
    _SequentialBenchmarkRunner,
    default_task_stream,
    run_sequential_benchmark,
    sequential_config_from_dict,
)
from apc.meta.controller import ControllerConfig

# --- StreamEvent -------------------------------------------------------


def test_stream_event_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="label"):
        StreamEvent("X")


def test_stream_event_requires_operation_name_for_novel_operation() -> None:
    with pytest.raises(ValueError, match="operation_name"):
        StreamEvent(LABEL_NOVEL_OPERATION)


def test_stream_event_requires_operation_name_for_recurrence() -> None:
    with pytest.raises(ValueError, match="operation_name"):
        StreamEvent(LABEL_RECURRENCE)


def test_stream_event_known_and_composition_do_not_need_operation_name() -> None:
    StreamEvent(LABEL_KNOWN)
    StreamEvent(LABEL_NOVEL_COMPOSITION)


# --- default_task_stream -------------------------------------------------


def test_default_task_stream_rejects_empty_novel_operations() -> None:
    with pytest.raises(ValueError, match="novel_operation_names"):
        default_task_stream(())


def test_default_task_stream_has_one_cycle_and_one_recurrence_per_novel_operation() -> None:
    stream = default_task_stream(("SORT", "REVERSE"))
    novel = [e for e in stream if e.label == LABEL_NOVEL_OPERATION]
    recurrence = [e for e in stream if e.label == LABEL_RECURRENCE]
    assert [e.operation_name for e in novel] == ["SORT", "REVERSE"]
    assert [e.operation_name for e in recurrence] == ["SORT", "REVERSE"]
    assert stream[0].label == LABEL_KNOWN
    assert stream[1].label == LABEL_NOVEL_COMPOSITION


# --- SequentialBenchmarkConfig validation --------------------------------


def test_config_rejects_non_positive_num_examples() -> None:
    with pytest.raises(ValueError, match="num_examples"):
        SequentialBenchmarkConfig(num_examples=0)


def test_config_rejects_non_positive_replay_buffer() -> None:
    with pytest.raises(ValueError, match="replay_buffer_max_events"):
        SequentialBenchmarkConfig(replay_buffer_max_events=0)


def test_config_rejects_non_positive_max_shadow_retries() -> None:
    with pytest.raises(ValueError, match="max_shadow_retries"):
        SequentialBenchmarkConfig(max_shadow_retries=0)


def test_config_resolved_events_defaults_to_default_task_stream() -> None:
    config = SequentialBenchmarkConfig(novel_operation_names=("SORT",))
    assert config.resolved_events() == default_task_stream(("SORT",))


def test_config_resolved_events_uses_explicit_override() -> None:
    custom = (StreamEvent(LABEL_KNOWN),)
    config = SequentialBenchmarkConfig(events=custom)
    assert config.resolved_events() == custom


def test_config_to_dict_round_trips_through_json() -> None:
    import json

    config = SequentialBenchmarkConfig()
    payload = config.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["allocator_preset"] == "small"
    assert payload["sequence_length_range"] == [6, 10]
    assert payload["novel_operation_names"] == ["SORT", "REVERSE"]


def test_sequential_config_from_dict_fills_in_defaults() -> None:
    config = sequential_config_from_dict({"seed": 7, "num_examples": 4})
    assert config.seed == 7
    assert config.num_examples == 4
    assert config.model == SequentialBenchmarkConfig().model


def test_sequential_config_from_dict_applies_nested_overrides() -> None:
    config = sequential_config_from_dict(
        {"plastic": {"lr": 0.5}, "consolidation": {"candidate_rank": 3}}
    )
    assert config.plastic.lr == 0.5
    assert config.plastic.max_steps == PlasticTrainingConfig().max_steps  # untouched default
    assert config.consolidation.candidate_rank == 3


# --- RNG isolation (ADR-0016) --------------------------------------------


def test_runner_init_reanchors_rng_regardless_of_setup_phase_model_size() -> None:
    """`_SequentialBenchmarkRunner.__init__` re-seeds after constructing its
    model/bank/router/workspace, so a random draw made immediately after
    construction must be identical regardless of how many parameters that
    setup phase happened to consume -- otherwise an unrelated change to
    model size (or anything else built during setup) would silently alter
    every subsequent run of the event loop for reasons that have nothing
    to do with the loop's own logic. See ADR-0016."""
    small = SequentialBenchmarkConfig(
        model={"d_model": 8, "n_layer": 1, "n_head": 2, "d_ff": 16, "max_seq_len": 32}
    )
    large = SequentialBenchmarkConfig(
        model={"d_model": 32, "n_layer": 4, "n_head": 4, "d_ff": 64, "max_seq_len": 32}
    )

    _SequentialBenchmarkRunner(small)
    draw_after_small = torch.randn(8)

    _SequentialBenchmarkRunner(large)
    draw_after_large = torch.randn(8)

    torch.testing.assert_close(draw_after_small, draw_after_large)


# --- sub-config validation ------------------------------------------------


def test_pretrain_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        PretrainConfig(steps=0)


def test_plastic_config_rejects_non_positive_eval_every() -> None:
    with pytest.raises(ValueError, match="eval_every"):
        PlasticTrainingConfig(eval_every=0)


def test_router_calibration_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        RouterCalibrationConfig(steps=0)


# --- Task 012 acceptance: end-to-end run ------------------------------------
# "Run at least two learn/consolidate/release cycles in one task stream."
# Sized to be as fast as reliably possible while still exercising the full
# STABLE -> SEARCH -> PLASTIC -> CONSOLIDATE -> SHADOW -> STABLE loop twice
# on real (if tiny) data -- see docs/DECISIONS.md ADR-0006/0009 for why the
# hyperparameters below (candidate_rank close to the allocated capacity,
# a relaxed retention threshold, many PLASTIC/consolidation steps) are
# needed at all: this repo's dense core does not generalize to fresh token
# content, so both PLASTIC and consolidation must fit examples largely from
# scratch every event, and low-rank compression has little slack to spare.


def _tiny_config(**overrides: object) -> SequentialBenchmarkConfig:
    defaults: dict[str, object] = dict(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        max_depth=2,
        model={
            "d_model": 32,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 16,
            "dropout": 0.0,
        },
        pretrain=PretrainConfig(steps=200, lr=1e-2, num_examples=16),
        num_examples=10,
        controller=ControllerConfig(
            enter_search_threshold=0.5,
            exit_search_threshold=0.3,
            max_search_steps=2,
            plastic_plateau_patience=3,
            plastic_min_accuracy=0.9,
        ),
        plastic=PlasticTrainingConfig(lr=1e-2, max_steps=1000, eval_every=100),
        consolidation=ConsolidationConfig(
            candidate_rank=15, steps=800, lr=5e-2, replay_weight=0.5
        ),
        shadow=ShadowValidationConfig(max_retention_degradation=0.12),
        router_calibration=RouterCalibrationConfig(steps=150, lr=5e-2),
        events=(StreamEvent(LABEL_KNOWN), StreamEvent(LABEL_NOVEL_OPERATION, "SORT")),
    )
    defaults.update(overrides)
    return SequentialBenchmarkConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def two_cycle_report() -> SequentialBenchmarkReport:
    return run_sequential_benchmark(_tiny_config())


def test_report_to_dict_round_trips_through_json(
    two_cycle_report: SequentialBenchmarkReport,
) -> None:
    import json

    payload = two_cycle_report.to_dict()
    json.dumps(payload)  # must not raise
    assert len(payload["events"]) == 2
    assert payload["config"]["seed"] == 0
    assert "persistent_parameter_count_final" not in payload
    assert payload["resident_total_parameter_count_final"] == (
        payload["stable_core_parameter_count"]
        + payload["resident_primitive_parameter_count_final"]
    )
    for event in payload["events"]:
        assert event["resident_total_param_count"] == (
            payload["stable_core_parameter_count"] + event["resident_primitive_param_count"]
        )
        assert event["active_param_count"] == (
            payload["stable_core_parameter_count"] + event["active_primitive_param_count"]
        )


def test_failing_shadow_preserves_temporary_capacity_and_retries() -> None:
    # An unreachable accuracy bar forces every PLASTIC episode to hit its
    # step cap and be force-promoted, and an unreachable task-score
    # threshold then fails shadow validation every attempt (Task 011
    # acceptance: "failing shadow test preserves temporary capacity" --
    # exercised here at the orchestration level, not just shadow.py's own
    # unit tests). The run must still finish (bounded retries) rather than
    # hang, and must record the give-up honestly instead of fabricating a
    # cycle that did not happen.
    config = _tiny_config(
        shadow=ShadowValidationConfig(task_score_ratio_threshold=0.999),
        max_shadow_retries=1,
        events=(StreamEvent(LABEL_NOVEL_OPERATION, "SORT"),),
    )
    report = run_sequential_benchmark(config)

    assert len(report.events) == 1
    event = report.events[0]
    assert event.had_cycle
    assert event.gave_up
    assert event.candidate_primitive_id is None
    assert event.shadow is not None
    assert not event.shadow.passed
    assert report.num_learn_consolidate_release_cycles == 0
    assert report.resident_primitive_parameter_count_final == 0
    assert report.resident_total_parameter_count_final == report.stable_core_parameter_count
