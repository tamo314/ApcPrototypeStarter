from __future__ import annotations

import json

import pytest

from apc.consolidation.distill import ConsolidationConfig
from apc.consolidation.shadow import ShadowValidationConfig
from apc.evaluation.baselines import (
    B0Runner,
    B1Runner,
    B2Runner,
    B3Runner,
    BaselineConfig,
    BaselineReport,
    baseline_config_from_dict,
    run_all_baselines,
    run_baseline,
)
from apc.evaluation.sequential_benchmark import (
    LABEL_KNOWN,
    LABEL_NOVEL_OPERATION,
    PlasticTrainingConfig,
    PretrainConfig,
    RouterCalibrationConfig,
    SequentialBenchmarkConfig,
    SequentialBenchmarkReport,
    StreamEvent,
)
from apc.meta.controller import ControllerConfig
from apc.plastic.allocator import Allocator

# --- BaselineConfig validation -------------------------------------------


def test_replay_weight_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="replay_weight"):
        BaselineConfig(replay_weight=-1.0)


def test_default_config_wraps_default_sequential_config() -> None:
    config = BaselineConfig()
    assert config.sequential == SequentialBenchmarkConfig()
    assert config.replay_weight == 1.0


def test_to_dict_round_trips_through_json() -> None:
    config = BaselineConfig()
    payload = config.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["replay_weight"] == 1.0
    assert payload["sequential"]["seed"] == 0


def test_baseline_config_from_dict_fills_defaults_and_applies_overrides() -> None:
    config = baseline_config_from_dict({"seed": 7, "baseline_replay_weight": 2.5})
    assert config.sequential.seed == 7
    assert config.replay_weight == 2.5


def test_baseline_config_from_dict_defaults_replay_weight() -> None:
    config = baseline_config_from_dict({"seed": 0})
    assert config.replay_weight == BaselineConfig().replay_weight


def test_run_baseline_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        run_baseline("B9", BaselineConfig())


# --- Task 013 acceptance: end-to-end runs --------------------------------
# Sized identically to test_sequential_benchmark.py's `_tiny_config` so all
# five baselines (B0-B4) run under directly comparable data/budgets -- see
# apc.evaluation.baselines's module docstring.


def _tiny_sequential_config(**overrides: object) -> SequentialBenchmarkConfig:
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
        plastic=PlasticTrainingConfig(lr=1e-2, max_steps=300, eval_every=50),
        consolidation=ConsolidationConfig(
            candidate_rank=14, steps=200, lr=5e-2, replay_weight=0.5
        ),
        shadow=ShadowValidationConfig(max_retention_degradation=0.08),
        router_calibration=RouterCalibrationConfig(steps=100, lr=5e-2),
        events=(StreamEvent(LABEL_KNOWN), StreamEvent(LABEL_NOVEL_OPERATION, "SORT")),
    )
    defaults.update(overrides)
    return SequentialBenchmarkConfig(**defaults)  # type: ignore[arg-type]


def _tiny_baseline_config() -> BaselineConfig:
    return BaselineConfig(sequential=_tiny_sequential_config())


@pytest.fixture(scope="module")
def all_baselines_report() -> dict[str, object]:
    return run_all_baselines(_tiny_baseline_config())


def test_run_all_baselines_returns_every_expected_baseline(
    all_baselines_report: dict[str, object],
) -> None:
    assert set(all_baselines_report) == {"B0", "B1", "B2", "B3", "B4"}
    assert isinstance(all_baselines_report["B4"], SequentialBenchmarkReport)
    for name in ("B0", "B1", "B2", "B3"):
        report = all_baselines_report[name]
        assert isinstance(report, BaselineReport)
        assert report.baseline == name
        assert len(report.events) == 2


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
def test_baseline_report_to_dict_round_trips_through_json(
    all_baselines_report: dict[str, object], name: str
) -> None:
    report = all_baselines_report[name]
    assert isinstance(report, BaselineReport)
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["baseline"] == name
    assert len(payload["events"]) == 2
    assert "persistent_parameter_count_final" not in payload
    assert payload["resident_total_parameter_count_final"] == (
        payload["stable_core_parameter_count"]
        + payload["resident_primitive_parameter_count_final"]
    )
    for event in payload["events"]:
        assert event["resident_total_param_count"] == (
            payload["stable_core_parameter_count"] + event["resident_primitive_param_count"]
        )


def test_b0_trains_the_dense_core_every_event_and_never_adds_capacity(
    all_baselines_report: dict[str, object],
) -> None:
    report = all_baselines_report["B0"]
    assert isinstance(report, BaselineReport)
    for event in report.events:
        assert event.train_steps > 0
    # No primitives at all: resident capacity is just the constant dense core.
    assert {e.resident_primitive_param_count for e in report.events} == {0}
    counts = {e.resident_total_param_count for e in report.events}
    assert counts == {report.resident_total_parameter_count_final}


def test_b1_bank_never_changes_after_the_one_time_seed_cycle(
    all_baselines_report: dict[str, object],
) -> None:
    report = all_baselines_report["B1"]
    assert isinstance(report, BaselineReport)
    # No expansion, ever -- fixed for the whole stream regardless of
    # whether the seed cycle happened to pass shadow validation.
    for event in report.events:
        assert event.train_steps == 0
    primitive_counts = {e.resident_primitive_param_count for e in report.events}
    assert primitive_counts == {report.resident_primitive_parameter_count_final}
    total_counts = {e.resident_total_param_count for e in report.events}
    assert total_counts == {report.resident_total_parameter_count_final}


def test_b2_grows_unconditionally_every_event(
    all_baselines_report: dict[str, object],
) -> None:
    report = all_baselines_report["B2"]
    assert isinstance(report, BaselineReport)
    config = _tiny_baseline_config().sequential
    spec = Allocator(d_model=config.model["d_model"]).spec_for(config.allocator_preset)
    per_event_params = spec.num_transforms * 2 * config.model["d_model"] * spec.rank

    for event in report.events:
        assert event.train_steps > 0
    counts = [e.resident_primitive_param_count for e in report.events]
    assert counts == [per_event_params * (i + 1) for i in range(len(counts))]
    assert report.resident_primitive_parameter_count_final == per_event_params * len(report.events)
    # Dense unconditional growth: every grown transform is always active.
    assert report.events[-1].active_param_count == report.events[-1].resident_primitive_param_count


def test_b3_grows_unconditionally_every_event_like_b2(
    all_baselines_report: dict[str, object],
) -> None:
    report = all_baselines_report["B3"]
    assert isinstance(report, BaselineReport)
    config = _tiny_baseline_config().sequential
    spec = Allocator(d_model=config.model["d_model"]).spec_for(config.allocator_preset)
    per_event_params = spec.num_transforms * 2 * config.model["d_model"] * spec.rank

    for event in report.events:
        assert event.train_steps > 0
    counts = [e.resident_primitive_param_count for e in report.events]
    assert counts == [per_event_params * (i + 1) for i in range(len(counts))]


def test_runner_classes_are_registered_under_their_baseline_name() -> None:
    assert B0Runner.name == "B0"
    assert B1Runner.name == "B1"
    assert B2Runner.name == "B2"
    assert B3Runner.name == "B3"
