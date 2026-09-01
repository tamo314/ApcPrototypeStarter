"""Shared-Core systematic-generalization gate tests (Phase A.1 Correction
Task A1-C004, H1b)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from apc.environments.generator import build_mixed_operation_generator
from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.evaluation.shared_core_generalization import (
    MATERIAL_UNDERPERFORMANCE_MARGIN,
    MIN_ANY_SEED_OPERATION_THRESHOLD,
    MIN_GATE_SEEDS,
    OVERALL_EXACT_MATCH_THRESHOLD,
    PER_OPERATION_EXACT_MATCH_THRESHOLD,
    SharedCoreGateConfig,
    SharedCoreGateReport,
    SharedCoreGateTrainConfig,
    _aggregate_multi_seed_report,
    _per_operation_exact_match,
    run_shared_core_gate,
    run_shared_core_gate_h1b,
    run_shared_core_gate_multi_seed,
    shared_core_gate_config_from_dict,
)

# --- module scope: no plastic/consolidation/primitive or meta code ---------


def test_module_imports_no_plastic_consolidation_primitive_or_meta_code() -> None:
    """A1-C004 trains a plain shared Stable Core only, same scope discipline
    as Task A1-006 -- see module docstring's 'K only, no plastic/
    consolidation/routing' design note."""
    import apc.evaluation.shared_core_generalization as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for banned in ("apc.plastic", "apc.consolidation", "apc.primitives", "apc.meta"):
        assert not any(banned in line for line in import_lines), banned


# --- SharedCoreGateTrainConfig validation -----------------------------------


def test_train_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        SharedCoreGateTrainConfig(steps=0)


def test_train_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SharedCoreGateTrainConfig(batch_size=0)


def test_train_config_rejects_non_positive_lr() -> None:
    with pytest.raises(ValueError, match="lr"):
        SharedCoreGateTrainConfig(lr=0)


def test_train_config_rejects_non_positive_eval_every() -> None:
    with pytest.raises(ValueError, match="eval_every"):
        SharedCoreGateTrainConfig(eval_every=0)


def test_train_config_rejects_non_positive_progress_eval_examples() -> None:
    with pytest.raises(ValueError, match="progress_eval_examples"):
        SharedCoreGateTrainConfig(progress_eval_examples=0)


# --- SharedCoreGateConfig validation and defaults ---------------------------


def test_config_default_operation_names_is_every_known_operation() -> None:
    """Unlike Task A1-006's DETERMINISTIC_OPERATION_NAMES restriction, the
    shared-core gate defaults to all eight KNOWN_OPERATION_NAMES: the task
    segment makes SELECT/COUNT/SHIFT/BIND's hidden parameter model-visible
    (ADR-0017), so excluding them would understate the correction."""
    assert SharedCoreGateConfig().operation_names == KNOWN_OPERATION_NAMES


def test_config_default_include_task_spec_is_true() -> None:
    assert SharedCoreGateConfig().include_task_spec is True


def test_config_default_permute_symbols_is_false() -> None:
    assert SharedCoreGateConfig().permute_symbols is False


def test_config_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        SharedCoreGateConfig(operation_names=())


def test_config_rejects_non_positive_num_unseen_eval_examples() -> None:
    with pytest.raises(ValueError, match="num_unseen_eval_examples"):
        SharedCoreGateConfig(num_unseen_eval_examples=0)


def test_config_rejects_non_positive_min_examples_per_operation() -> None:
    with pytest.raises(ValueError, match="min_examples_per_operation"):
        SharedCoreGateConfig(min_examples_per_operation=0)


def test_config_to_dict_round_trips_through_json() -> None:
    config = SharedCoreGateConfig()
    payload = config.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["seed"] == 0
    assert payload["operation_names"] == list(KNOWN_OPERATION_NAMES)
    assert payload["train"]["steps"] == SharedCoreGateTrainConfig().steps


def test_config_from_dict_fills_defaults() -> None:
    config = shared_core_gate_config_from_dict({})
    assert config == SharedCoreGateConfig()


def test_config_from_dict_applies_top_level_overrides() -> None:
    config = shared_core_gate_config_from_dict(
        {
            "seed": 7,
            "vocab_size": 12,
            "operation_names": ["COPY", "NEGATE"],
            "include_task_spec": False,
        }
    )
    assert config.seed == 7
    assert config.vocab_size == 12
    assert config.operation_names == ("COPY", "NEGATE")
    assert config.include_task_spec is False


def test_config_from_dict_applies_nested_train_overrides_and_keeps_other_defaults() -> None:
    config = shared_core_gate_config_from_dict({"train": {"steps": 5, "batch_size": 8}})
    assert config.train.steps == 5
    assert config.train.batch_size == 8
    assert config.train.lr == SharedCoreGateTrainConfig().lr  # untouched default


# --- fixtures: tiny, fast, deterministic configuration ----------------------


def _tiny_model_config() -> dict[str, object]:
    return {"d_model": 8, "n_layer": 1, "n_head": 2, "d_ff": 16, "max_seq_len": 24, "dropout": 0.0}


def _tiny_config(**overrides: object) -> SharedCoreGateConfig:
    base = SharedCoreGateConfig(
        vocab_size=6,
        sequence_length_range=(2, 3),
        operation_names=("COPY", "NEGATE"),
        model=_tiny_model_config(),
        train=SharedCoreGateTrainConfig(
            steps=2, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        num_unseen_eval_examples=120,
        min_examples_per_operation=10,
    )
    return dataclasses.replace(base, **overrides)


# --- _per_operation_exact_match ----------------------------------------------


def test_per_operation_exact_match_groups_by_operation() -> None:
    generator = build_mixed_operation_generator(
        seed=0, operation_names=("COPY", "NEGATE"), sequence_length_range=(3, 4)
    )
    examples = generator.generate_online(20, step=0, split="test")
    predictions = [example.target_tokens for example in examples]  # perfect predictions

    per_operation, counts = _per_operation_exact_match(
        examples, predictions, ("COPY", "NEGATE"), min_examples_per_operation=1
    )
    assert set(per_operation) == {"COPY", "NEGATE"}
    assert all(value == 1.0 for value in per_operation.values())
    assert sum(counts.values()) == 20


def test_per_operation_exact_match_detects_mismatches() -> None:
    generator = build_mixed_operation_generator(
        seed=0, operation_names=("COPY",), sequence_length_range=(3, 4)
    )
    examples = generator.generate_online(10, step=0, split="test")
    wrong_predictions = [() for _ in examples]  # never matches

    per_operation, _ = _per_operation_exact_match(
        examples, wrong_predictions, ("COPY",), min_examples_per_operation=1
    )
    assert per_operation["COPY"] == 0.0


def test_per_operation_exact_match_raises_when_an_operation_is_under_the_minimum() -> None:
    generator = build_mixed_operation_generator(
        seed=0, operation_names=("COPY", "NEGATE"), sequence_length_range=(3, 4)
    )
    examples = generator.generate_online(20, step=0, split="test")
    predictions = [example.target_tokens for example in examples]

    with pytest.raises(ValueError, match="fewer than"):
        _per_operation_exact_match(
            examples, predictions, ("COPY", "NEGATE"), min_examples_per_operation=1000
        )


# --- run_shared_core_gate ----------------------------------------------------


def test_run_is_deterministic_given_the_same_seed() -> None:
    config = _tiny_config(seed=3)
    first = run_shared_core_gate(config)
    second = run_shared_core_gate(config)
    assert first.final_train_loss == second.final_train_loss
    assert first.overall_exact_match == second.overall_exact_match
    assert first.per_operation_exact_match == second.per_operation_exact_match


def test_different_seeds_diverge() -> None:
    first = run_shared_core_gate(_tiny_config(seed=1))
    second = run_shared_core_gate(_tiny_config(seed=2))
    assert first.final_train_loss != second.final_train_loss


def test_report_fields() -> None:
    config = _tiny_config()
    report = run_shared_core_gate(config)
    assert isinstance(report, SharedCoreGateReport)
    assert report.steps_trained == config.train.steps
    assert report.examples_seen == config.train.steps * config.train.batch_size
    assert report.num_unseen_eval_examples == config.num_unseen_eval_examples
    assert 0.0 <= report.overall_exact_match <= 1.0
    assert set(report.per_operation_exact_match) == set(config.operation_names)
    assert all(0.0 <= v <= 1.0 for v in report.per_operation_exact_match.values())
    assert sum(report.per_operation_eval_counts.values()) == config.num_unseen_eval_examples
    assert report.param_count > 0
    assert report.trainable_param_count == report.param_count
    assert report.device == "cpu"


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_shared_core_gate(_tiny_config())
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["steps_trained"] == report.steps_trained


def test_metrics_path_writes_expected_progress_lines(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    config = _tiny_config(
        train=SharedCoreGateTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        )
    )
    run_shared_core_gate(config, metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == config.train.steps
    for step_index, line in enumerate(lines, start=1):
        payload = json.loads(line)
        assert payload["step"] == step_index
        assert isinstance(payload["loss"], float)
        assert 0.0 <= payload["progress_overall_exact_match"] <= 1.0


def test_no_metrics_path_means_no_file_side_effect(tmp_path: Path) -> None:
    run_shared_core_gate(_tiny_config())
    assert list(tmp_path.iterdir()) == []


def test_explicit_and_negative_control_share_identical_model_vocab_and_param_count() -> None:
    """The negative control is meant to isolate exactly one variable (module
    docstring): both variants must share the same SharedCoreTokens
    vocabulary/model architecture, differing only in whether the task
    segment is rendered into the input."""
    explicit_report = run_shared_core_gate(_tiny_config(include_task_spec=True))
    control_report = run_shared_core_gate(_tiny_config(include_task_spec=False))
    assert explicit_report.param_count == control_report.param_count


# --- learnability smoke test (wiring sanity, not the scientific claim) ------


def test_training_reduces_loss_with_task_spec_on_easy_operations() -> None:
    """Not a test of the 0.95/0.90 generalization claims (those are
    experimental results, reported via the real gate run/config, not
    asserted in a unit test) -- just confirms the online-generation +
    task-spec + training loop is wired correctly enough to make visible
    progress within a small, CPU-fast budget, mirroring `apc.evaluation.
    stable_core_generalization`'s own equivalent test. `SELECT` is one of
    ADR-0017's hidden-parameter operations (its `indices` argument, revealed
    here via the task segment) -- deliberately included instead of a second
    parameter-free operation like `NEGATE`/`ACCUMULATE`, which converge much
    slower even single-operation (`docs/DECISIONS.md` ADR-0019: `NEGATE`
    alone needs 6000 steps) and would make this a slow, high-budget test for
    no wiring-coverage benefit; `SELECT`'s answer is close to directly given
    once its indices are in the input, so `COPY`+`SELECT` converges fast
    while still exercising a mixed, task-spec-conditioned batch."""
    config = SharedCoreGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(3, 5),
        operation_names=("COPY", "SELECT"),
        include_task_spec=True,
        model={"d_model": 32, "n_layer": 2, "n_head": 2, "d_ff": 64, "max_seq_len": 24},
        train=SharedCoreGateTrainConfig(
            steps=2000, batch_size=32, lr=2e-3, eval_every=2000, device="cpu"
        ),
        num_unseen_eval_examples=128,
        min_examples_per_operation=16,
    )
    report = run_shared_core_gate(config)
    assert report.final_train_loss < 0.5
    assert report.overall_exact_match > 0.7


# --- multi-seed aggregation --------------------------------------------------


def _fake_report(
    seed: int, overall_exact_match: float, per_operation_exact_match: dict[str, float]
) -> SharedCoreGateReport:
    return SharedCoreGateReport(
        config=SharedCoreGateConfig(seed=seed, operation_names=tuple(per_operation_exact_match)),
        steps_trained=1,
        examples_seen=1,
        final_train_loss=0.0,
        overall_exact_match=overall_exact_match,
        per_operation_exact_match=dict(per_operation_exact_match),
        per_operation_eval_counts={name: 1 for name in per_operation_exact_match},
        num_unseen_eval_examples=1,
        param_count=1,
        trainable_param_count=1,
        wall_clock_seconds=0.0,
        device="cpu",
    )


def test_aggregate_computes_overall_mean_stdev_min_max() -> None:
    reports = [
        _fake_report(i, value, {"COPY": value, "NEGATE": value})
        for i, value in enumerate([0.9, 1.0, 0.95, 0.8, 1.0])
    ]
    result = _aggregate_multi_seed_report(
        reports,
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    assert result.seeds == (0, 1, 2, 3, 4)
    assert result.mean_overall_exact_match == pytest.approx(0.93)
    assert result.min_overall_exact_match == 0.8
    assert result.max_overall_exact_match == 1.0
    assert result.stdev_overall_exact_match > 0.0
    assert result.meets_seed_policy is True


def test_aggregate_computes_per_operation_means() -> None:
    reports = [
        _fake_report(0, 0.9, {"COPY": 1.0, "NEGATE": 0.8}),
        _fake_report(1, 0.9, {"COPY": 0.9, "NEGATE": 0.9}),
    ]
    result = _aggregate_multi_seed_report(
        reports,
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    assert result.per_operation_mean_exact_match["COPY"] == pytest.approx(0.95)
    assert result.per_operation_mean_exact_match["NEGATE"] == pytest.approx(0.85)


def test_aggregate_passed_requires_overall_and_every_operation_threshold() -> None:
    passing = _aggregate_multi_seed_report(
        [_fake_report(0, 0.96, {"COPY": 0.95, "NEGATE": 0.95})],
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    failing_overall = _aggregate_multi_seed_report(
        [_fake_report(0, 0.5, {"COPY": 0.95, "NEGATE": 0.95})],
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    failing_one_operation = _aggregate_multi_seed_report(
        [_fake_report(0, 0.96, {"COPY": 0.95, "NEGATE": 0.5})],
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    assert passing.passed is True
    assert failing_overall.passed is False
    assert failing_one_operation.passed is False


def test_aggregate_flags_low_outlier_seed_operation_pairs() -> None:
    """docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md section 2: 'no operation
    below 0.85 in any seed without explicit investigation' -- a per-(seed,
    operation) floor distinct from the cross-seed per-operation mean."""
    reports = [
        _fake_report(0, 0.9, {"COPY": 0.95, "NEGATE": 0.80}),  # NEGATE below 0.85
        _fake_report(1, 0.95, {"COPY": 0.95, "NEGATE": 0.95}),
    ]
    result = _aggregate_multi_seed_report(
        reports,
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.0,
        per_operation_threshold=0.0,
    )
    assert result.low_outlier_threshold == MIN_ANY_SEED_OPERATION_THRESHOLD
    assert result.low_outlier_seed_operations == (
        {"seed": 0, "operation": "NEGATE", "exact_match": 0.80},
    )


def test_aggregate_no_low_outliers_when_every_value_clears_the_floor() -> None:
    reports = [_fake_report(0, 0.9, {"COPY": 0.9, "NEGATE": 0.9})]
    result = _aggregate_multi_seed_report(
        reports,
        operation_names=("COPY", "NEGATE"),
        overall_threshold=0.0,
        per_operation_threshold=0.0,
    )
    assert result.low_outlier_seed_operations == ()


def test_aggregate_rejects_empty_reports() -> None:
    with pytest.raises(ValueError, match="per_seed"):
        _aggregate_multi_seed_report(
            [], operation_names=("COPY",), overall_threshold=0.95, per_operation_threshold=0.9
        )


def test_min_gate_seeds_matches_experiment_plan_policy() -> None:
    assert MIN_GATE_SEEDS == 5


def test_default_thresholds_match_correction_task_acceptance() -> None:
    assert OVERALL_EXACT_MATCH_THRESHOLD == 0.95
    assert PER_OPERATION_EXACT_MATCH_THRESHOLD == 0.90


# --- run_shared_core_gate_multi_seed (integration) --------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    base_config = _tiny_config()
    result = run_shared_core_gate_multi_seed(
        base_config,
        seeds=(10, 11),
        run_dir=tmp_path,
        overall_threshold=0.95,
        per_operation_threshold=0.9,
    )
    assert result.seeds == (10, 11)
    assert len(result.per_seed) == 2
    assert [report.config.seed for report in result.per_seed] == [10, 11]
    assert result.meets_seed_policy is False  # only 2 < MIN_GATE_SEEDS
    for seed in (10, 11):
        assert (tmp_path / f"seed_{seed}" / "metrics.jsonl").is_file()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    base_config = _tiny_config(seed=999)
    result = run_shared_core_gate_multi_seed(base_config, seeds=(1, 2, 3, 4, 5))
    assert result.meets_seed_policy is True
    assert [report.config.seed for report in result.per_seed] == [1, 2, 3, 4, 5]


def test_multi_seed_report_to_dict_round_trips_through_json() -> None:
    result = run_shared_core_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["operation_names"] == list(_tiny_config().operation_names)


# --- run_shared_core_gate_h1b (integration) ----------------------------------


def test_h1b_runs_both_variants_with_the_correct_include_task_spec(tmp_path: Path) -> None:
    base_config = _tiny_config(include_task_spec=False)  # must be overridden per variant
    result = run_shared_core_gate_h1b(
        base_config,
        seeds=(0, 1),
        run_dir=tmp_path,
        overall_threshold=0.0,
        per_operation_threshold=0.0,
        material_underperformance_margin=-1.0,
    )
    assert all(report.config.include_task_spec is True for report in result.explicit.per_seed)
    assert all(
        report.config.include_task_spec is False for report in result.negative_control.per_seed
    )
    for variant in ("explicit", "negative_control"):
        for seed in (0, 1):
            assert (tmp_path / variant / f"seed_{seed}" / "metrics.jsonl").is_file()


def test_h1b_gap_is_explicit_minus_control_mean(tmp_path: Path) -> None:
    result = run_shared_core_gate_h1b(
        _tiny_config(),
        seeds=(0, 1),
        run_dir=tmp_path,
        overall_threshold=0.0,
        per_operation_threshold=0.0,
        material_underperformance_margin=-1.0,
    )
    expected_gap = (
        result.explicit.mean_overall_exact_match - result.negative_control.mean_overall_exact_match
    )
    assert result.overall_exact_match_gap == pytest.approx(expected_gap)


def test_h1b_passed_requires_explicit_pass_and_material_underperformance(tmp_path: Path) -> None:
    trivially_passing = run_shared_core_gate_h1b(
        _tiny_config(),
        seeds=(0, 1),
        run_dir=tmp_path / "trivial",
        overall_threshold=0.0,
        per_operation_threshold=0.0,
        material_underperformance_margin=-1.0,  # any gap clears a negative margin
    )
    assert trivially_passing.passed is True

    impossible_margin = run_shared_core_gate_h1b(
        _tiny_config(),
        seeds=(0, 1),
        run_dir=tmp_path / "impossible",
        overall_threshold=0.0,
        per_operation_threshold=0.0,
        material_underperformance_margin=2.0,  # no gap can ever reach this
    )
    assert impossible_margin.negative_control_materially_underperforms is False
    assert impossible_margin.passed is False


def test_h1b_meets_seed_policy_requires_both_variants(tmp_path: Path) -> None:
    result = run_shared_core_gate_h1b(_tiny_config(), seeds=(0, 1), run_dir=tmp_path)
    assert result.meets_seed_policy is False  # 2 < MIN_GATE_SEEDS


def test_h1b_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    result = run_shared_core_gate_h1b(_tiny_config(), seeds=(0, 1), run_dir=tmp_path)
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["material_underperformance_margin"] == MATERIAL_UNDERPERFORMANCE_MARGIN
