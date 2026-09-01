"""Stable Core systematic-generalization gate tests (Phase A.1 Task A1-006)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES
from apc.evaluation.stable_core_generalization import (
    MIN_GATE_SEEDS,
    StableCoreGateConfig,
    StableCoreGateReport,
    StableCoreGateTrainConfig,
    _aggregate_multi_seed_report,
    run_stable_core_gate,
    run_stable_core_gate_grid,
    run_stable_core_gate_multi_seed,
    stable_core_gate_config_from_dict,
)

# --- module scope: no plastic/consolidation/routing path -------------------


def test_module_imports_no_plastic_consolidation_primitive_or_meta_code() -> None:
    """A1-006 trains a plain Stable Core only -- see module docstring's 'K
    only, no plastic/consolidation/routing' design note. Reading the source's
    `import`/`from` lines directly (rather than introspecting `sys.modules`,
    which every other test importing this module would have already
    populated, or substring-matching the whole file, which would also match
    this design note's own prose) verifies this file itself never imports
    those subsystems."""
    import apc.evaluation.stable_core_generalization as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for banned in ("apc.plastic", "apc.consolidation", "apc.primitives", "apc.meta"):
        assert not any(banned in line for line in import_lines), banned


# --- StableCoreGateTrainConfig validation -----------------------------------


def test_train_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        StableCoreGateTrainConfig(steps=0)


def test_train_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StableCoreGateTrainConfig(batch_size=0)


def test_train_config_rejects_non_positive_lr() -> None:
    with pytest.raises(ValueError, match="lr"):
        StableCoreGateTrainConfig(lr=0)


def test_train_config_rejects_non_positive_eval_every() -> None:
    with pytest.raises(ValueError, match="eval_every"):
        StableCoreGateTrainConfig(eval_every=0)


def test_train_config_rejects_non_positive_progress_eval_examples() -> None:
    with pytest.raises(ValueError, match="progress_eval_examples"):
        StableCoreGateTrainConfig(progress_eval_examples=0)


# --- StableCoreGateConfig validation and defaults ---------------------------


def test_config_default_operation_names_is_the_deterministic_subset() -> None:
    assert StableCoreGateConfig().operation_names == DETERMINISTIC_OPERATION_NAMES


def test_config_default_permute_symbols_is_false() -> None:
    """ADR-0019: online generation alone already prevents finite-set
    memorization, and permutation makes value/order-dependent operations
    provably unrecoverable on held-out content regardless of training --
    so it must be opt-in, not the default."""
    assert StableCoreGateConfig().permute_symbols is False


def test_config_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        StableCoreGateConfig(operation_names=())


def test_config_rejects_non_positive_num_unseen_eval_examples() -> None:
    with pytest.raises(ValueError, match="num_unseen_eval_examples"):
        StableCoreGateConfig(num_unseen_eval_examples=0)


def test_config_to_dict_round_trips_through_json() -> None:
    config = StableCoreGateConfig()
    payload = config.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["seed"] == 0
    assert payload["operation_names"] == list(DETERMINISTIC_OPERATION_NAMES)
    assert payload["train"]["steps"] == StableCoreGateTrainConfig().steps


def test_config_from_dict_fills_defaults() -> None:
    config = stable_core_gate_config_from_dict({})
    assert config == StableCoreGateConfig()


def test_config_from_dict_applies_top_level_overrides() -> None:
    config = stable_core_gate_config_from_dict(
        {"seed": 7, "vocab_size": 12, "operation_names": ["COPY", "NEGATE"]}
    )
    assert config.seed == 7
    assert config.vocab_size == 12
    assert config.operation_names == ("COPY", "NEGATE")


def test_config_from_dict_applies_nested_train_overrides_and_keeps_other_defaults() -> None:
    config = stable_core_gate_config_from_dict({"train": {"steps": 5, "batch_size": 8}})
    assert config.train.steps == 5
    assert config.train.batch_size == 8
    assert config.train.lr == StableCoreGateTrainConfig().lr  # untouched default


# --- fixtures: tiny, fast, deterministic configuration ----------------------


def _tiny_model_config() -> dict[str, object]:
    return {"d_model": 8, "n_layer": 1, "n_head": 2, "d_ff": 16, "max_seq_len": 16, "dropout": 0.0}


def _tiny_config(**overrides: object) -> StableCoreGateConfig:
    base = StableCoreGateConfig(
        vocab_size=6,
        sequence_length_range=(2, 3),
        operation_names=("COPY", "NEGATE"),
        model=_tiny_model_config(),
        train=StableCoreGateTrainConfig(
            steps=2, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        num_unseen_eval_examples=4,
    )
    return dataclasses.replace(base, **overrides)


# --- run_stable_core_gate ----------------------------------------------------


def test_run_is_deterministic_given_the_same_seed() -> None:
    config = _tiny_config(seed=3)
    first = run_stable_core_gate(config)
    second = run_stable_core_gate(config)
    assert first.final_train_loss == second.final_train_loss
    assert first.unseen_exact_match == second.unseen_exact_match


def test_different_seeds_diverge() -> None:
    first = run_stable_core_gate(_tiny_config(seed=1))
    second = run_stable_core_gate(_tiny_config(seed=2))
    assert first.final_train_loss != second.final_train_loss


def test_report_examples_seen_and_param_counts() -> None:
    config = _tiny_config()
    report = run_stable_core_gate(config)
    assert isinstance(report, StableCoreGateReport)
    assert report.steps_trained == config.train.steps
    assert report.examples_seen == config.train.steps * config.train.batch_size
    assert report.num_unseen_eval_examples == config.num_unseen_eval_examples
    assert 0.0 <= report.unseen_exact_match <= 1.0
    assert report.param_count > 0
    assert report.trainable_param_count == report.param_count
    assert report.device == "cpu"


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_stable_core_gate(_tiny_config())
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["steps_trained"] == report.steps_trained


def test_metrics_path_writes_expected_progress_lines(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    config = _tiny_config(
        train=StableCoreGateTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        )
    )
    run_stable_core_gate(config, metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == config.train.steps  # eval_every=1: one line per step
    for step_index, line in enumerate(lines, start=1):
        payload = json.loads(line)
        assert payload["step"] == step_index
        assert isinstance(payload["loss"], float)
        assert 0.0 <= payload["progress_exact_match"] <= 1.0


def test_no_metrics_path_means_no_file_side_effect(tmp_path: Path) -> None:
    run_stable_core_gate(_tiny_config())
    assert list(tmp_path.iterdir()) == []


# --- learnability smoke test (wiring sanity, not the scientific claim) ------


def test_training_reduces_loss_on_an_easy_deterministic_operation() -> None:
    """Not a test of the 0.95 generalization claim (that is an experimental
    result, reported via the real gate run/config, not asserted in a unit
    test) -- just confirms the online-generation + training loop is wired
    correctly enough to make visible progress on the easiest deterministic
    operation (COPY) within a small, CPU-fast budget."""
    config = StableCoreGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(3, 5),
        operation_names=("COPY",),
        model={"d_model": 32, "n_layer": 2, "n_head": 2, "d_ff": 64, "max_seq_len": 16},
        train=StableCoreGateTrainConfig(
            steps=200, batch_size=32, lr=1e-3, eval_every=200, device="cpu"
        ),
        num_unseen_eval_examples=64,
    )
    report = run_stable_core_gate(config)
    assert report.final_train_loss < 1.0
    assert report.unseen_exact_match > 0.5


# --- multi-seed aggregation --------------------------------------------------


def _fake_report(seed: int, unseen_exact_match: float) -> StableCoreGateReport:
    return StableCoreGateReport(
        config=StableCoreGateConfig(seed=seed),
        steps_trained=1,
        examples_seen=1,
        final_train_loss=0.0,
        unseen_exact_match=unseen_exact_match,
        num_unseen_eval_examples=1,
        param_count=1,
        trainable_param_count=1,
        wall_clock_seconds=0.0,
        device="cpu",
    )


def test_aggregate_computes_mean_stdev_min_max() -> None:
    reports = [_fake_report(i, value) for i, value in enumerate([0.9, 1.0, 0.95, 0.8, 1.0])]
    result = _aggregate_multi_seed_report(reports, threshold=0.95)
    assert result.seeds == (0, 1, 2, 3, 4)
    assert result.mean_unseen_exact_match == pytest.approx(0.93)
    assert result.min_unseen_exact_match == 0.8
    assert result.max_unseen_exact_match == 1.0
    assert result.stdev_unseen_exact_match > 0.0
    assert result.meets_seed_policy is True


def test_aggregate_single_seed_has_zero_stdev() -> None:
    result = _aggregate_multi_seed_report([_fake_report(0, 0.97)], threshold=0.95)
    assert result.stdev_unseen_exact_match == 0.0
    assert result.meets_seed_policy is False


def test_aggregate_passed_is_threshold_comparison_on_the_mean() -> None:
    passing = _aggregate_multi_seed_report(
        [_fake_report(0, 0.95), _fake_report(1, 0.96)], threshold=0.95
    )
    failing = _aggregate_multi_seed_report(
        [_fake_report(0, 0.5), _fake_report(1, 0.4)], threshold=0.95
    )
    assert passing.passed is True
    assert failing.passed is False


def test_aggregate_rejects_empty_reports() -> None:
    with pytest.raises(ValueError, match="per_seed"):
        _aggregate_multi_seed_report([], threshold=0.95)


def test_min_gate_seeds_matches_experiment_plan_policy() -> None:
    assert MIN_GATE_SEEDS == 5


# --- run_stable_core_gate_multi_seed (integration) --------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    base_config = _tiny_config()
    result = run_stable_core_gate_multi_seed(
        base_config, seeds=(10, 11), run_dir=tmp_path, threshold=0.95
    )
    assert result.seeds == (10, 11)
    assert len(result.per_seed) == 2
    assert [report.config.seed for report in result.per_seed] == [10, 11]
    assert result.meets_seed_policy is False  # only 2 < MIN_GATE_SEEDS
    for seed in (10, 11):
        assert (tmp_path / f"seed_{seed}" / "metrics.jsonl").is_file()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    base_config = _tiny_config(seed=999)
    result = run_stable_core_gate_multi_seed(base_config, seeds=(1, 2, 3, 4, 5))
    assert result.meets_seed_policy is True
    assert [report.config.seed for report in result.per_seed] == [1, 2, 3, 4, 5]


# --- run_stable_core_gate_grid (ADR-0020: one operation at a time) ----------


def test_grid_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        run_stable_core_gate_grid(_tiny_config(), operation_names=())


def test_grid_never_pools_operations_into_one_config() -> None:
    """Each per-operation run must see exactly one operation, never the
    full pool passed to `operation_names=` -- that pooling is exactly what
    ADR-0020 identified as unidentifiable."""
    base_config = _tiny_config(operation_names=("COPY", "NEGATE"))
    result = run_stable_core_gate_grid(base_config, operation_names=("COPY", "NEGATE"), seeds=(0,))
    for operation_report in result.per_operation:
        for report in operation_report.multi_seed.per_seed:
            assert report.config.operation_names == (operation_report.operation_name,)


def test_grid_reports_operation_names_and_seeds() -> None:
    result = run_stable_core_gate_grid(
        _tiny_config(), operation_names=("COPY", "NEGATE"), seeds=(0, 1)
    )
    assert result.operation_names == ("COPY", "NEGATE")
    assert result.seeds == (0, 1)
    assert [report.operation_name for report in result.per_operation] == ["COPY", "NEGATE"]
    for operation_report in result.per_operation:
        assert len(operation_report.multi_seed.per_seed) == 2


def test_grid_mean_is_over_every_operation_seed_pair_not_per_operation_means() -> None:
    """`docs/CODEX_TASKS_PHASE_A1.md` A1-006's 'mean unseen-content exact
    match' is read across the whole operation x seed grid, one data point
    per run -- not the mean of each operation's own mean."""
    result = run_stable_core_gate_grid(
        _tiny_config(), operation_names=("COPY", "NEGATE"), seeds=(0, 1, 2)
    )
    all_values = [
        report.unseen_exact_match
        for operation_report in result.per_operation
        for report in operation_report.multi_seed.per_seed
    ]
    assert len(all_values) == 6  # 2 operations x 3 seeds
    assert result.mean_unseen_exact_match == pytest.approx(sum(all_values) / len(all_values))


def test_grid_meets_seed_policy_requires_every_operation_to_meet_it() -> None:
    passing = run_stable_core_gate_grid(
        _tiny_config(), operation_names=("COPY",), seeds=(0, 1, 2, 3, 4)
    )
    failing = run_stable_core_gate_grid(_tiny_config(), operation_names=("COPY",), seeds=(0, 1))
    assert passing.meets_seed_policy is True
    assert failing.meets_seed_policy is False


def test_grid_writes_one_metrics_file_per_operation_and_seed(tmp_path: Path) -> None:
    run_stable_core_gate_grid(
        _tiny_config(), operation_names=("COPY", "NEGATE"), seeds=(0, 1), run_dir=tmp_path
    )
    for operation_name in ("COPY", "NEGATE"):
        for seed in (0, 1):
            assert (tmp_path / operation_name / f"seed_{seed}" / "metrics.jsonl").is_file()


def test_grid_report_to_dict_round_trips_through_json() -> None:
    result = run_stable_core_gate_grid(_tiny_config(), operation_names=("COPY",), seeds=(0, 1))
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["operation_names"] == ["COPY"]
    assert len(payload["per_operation"]) == 1
    assert payload["per_operation"][0]["operation_name"] == "COPY"
