"""Task/content representation probe tests (Phase A.1 Correction Task
A1-C005, STOP GATE)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES, KNOWN_OPERATION_NAMES
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
    train_shared_core,
)
from apc.evaluation.task_content_probes import (
    ARGUMENT_PROBE_THRESHOLD,
    CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD,
    OPERATION_ID_PROBE_THRESHOLD,
    PARAMETERIZED_OPERATION_NAMES,
    SCALAR_ARGUMENT_OPERATIONS,
    SET_ARGUMENT_OPERATIONS,
    ProbeTrainConfig,
    TaskContentProbeConfig,
    _extract_features,
    run_task_content_probes,
    run_task_content_probes_multi_seed,
    task_content_probe_config_from_dict,
)


def _tiny_shared_core_config(seed: int = 0, **overrides) -> SharedCoreGateConfig:
    base = SharedCoreGateConfig(
        seed=seed,
        vocab_size=6,
        sequence_length_range=(4, 6),
        model={
            "d_model": 32,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        train=SharedCoreGateTrainConfig(
            steps=150,
            batch_size=64,
            eval_every=150,
            progress_eval_examples=16,
            device="cpu",
        ),
        num_unseen_eval_examples=128,
        min_examples_per_operation=8,
    )
    return dataclasses.replace(base, **overrides)


def _tiny_probe_config(seed: int = 0, **overrides) -> TaskContentProbeConfig:
    base = TaskContentProbeConfig(
        seed=seed,
        shared_core=_tiny_shared_core_config(seed=seed),
        probe_train=ProbeTrainConfig(steps=80, lr=0.05),
        num_probe_train_examples=1024,
        num_probe_eval_examples=512,
        min_examples_per_operation=16,
    )
    return dataclasses.replace(base, **overrides)


# --- module scope: no plastic/consolidation/primitive or meta code ---------


def test_module_imports_no_plastic_consolidation_primitive_or_meta_code() -> None:
    import apc.evaluation.task_content_probes as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for banned in ("apc.plastic", "apc.consolidation", "apc.primitives", "apc.meta"):
        assert not any(banned in line for line in import_lines), banned


# --- operation-name partitions -----------------------------------------------


def test_parameterized_operation_names_is_known_minus_deterministic() -> None:
    assert set(PARAMETERIZED_OPERATION_NAMES) == set(KNOWN_OPERATION_NAMES) - set(
        DETERMINISTIC_OPERATION_NAMES
    )


def test_scalar_and_set_argument_operations_partition_parameterized_names() -> None:
    assert set(SCALAR_ARGUMENT_OPERATIONS) | set(SET_ARGUMENT_OPERATIONS) == set(
        PARAMETERIZED_OPERATION_NAMES
    )
    assert set(SCALAR_ARGUMENT_OPERATIONS).isdisjoint(SET_ARGUMENT_OPERATIONS)


def test_set_argument_operations_is_exactly_select() -> None:
    assert SET_ARGUMENT_OPERATIONS == ("SELECT",)


# --- ProbeTrainConfig validation ---------------------------------------------


def test_probe_train_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        ProbeTrainConfig(steps=0)


def test_probe_train_config_rejects_non_positive_lr() -> None:
    with pytest.raises(ValueError, match="lr"):
        ProbeTrainConfig(lr=0.0)


# --- TaskContentProbeConfig validation ---------------------------------------


def test_config_rejects_shared_core_without_task_spec() -> None:
    shared_core = SharedCoreGateConfig(include_task_spec=False)
    with pytest.raises(ValueError, match="include_task_spec"):
        TaskContentProbeConfig(shared_core=shared_core)


def test_config_rejects_non_positive_num_probe_train_examples() -> None:
    with pytest.raises(ValueError, match="num_probe_train_examples"):
        TaskContentProbeConfig(num_probe_train_examples=0)


def test_config_rejects_non_positive_num_probe_eval_examples() -> None:
    with pytest.raises(ValueError, match="num_probe_eval_examples"):
        TaskContentProbeConfig(num_probe_eval_examples=0)


def test_config_rejects_non_positive_min_examples_per_operation() -> None:
    with pytest.raises(ValueError, match="min_examples_per_operation"):
        TaskContentProbeConfig(min_examples_per_operation=0)


def test_config_to_dict_round_trips_through_json() -> None:
    config = _tiny_probe_config()
    raw = json.loads(json.dumps(config.to_dict()))
    assert raw["seed"] == 0
    assert raw["shared_core"]["include_task_spec"] is True


def test_config_default_thresholds_match_module_constants() -> None:
    config = TaskContentProbeConfig()
    assert config.operation_id_threshold == OPERATION_ID_PROBE_THRESHOLD
    assert config.argument_threshold == ARGUMENT_PROBE_THRESHOLD
    assert config.content_threshold == CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD


# --- task_content_probe_config_from_dict -------------------------------------


def test_config_from_dict_fills_defaults() -> None:
    config = task_content_probe_config_from_dict({})
    assert config == TaskContentProbeConfig()


def test_config_from_dict_applies_top_level_overrides() -> None:
    config = task_content_probe_config_from_dict(
        {"seed": 7, "num_probe_train_examples": 99, "num_probe_eval_examples": 55}
    )
    assert config.seed == 7
    assert config.num_probe_train_examples == 99
    assert config.num_probe_eval_examples == 55


def test_config_from_dict_forces_include_task_spec_true() -> None:
    config = task_content_probe_config_from_dict(
        {"shared_core": {"include_task_spec": False, "vocab_size": 6}}
    )
    assert config.shared_core.include_task_spec is True
    assert config.shared_core.vocab_size == 6


def test_config_from_dict_applies_nested_probe_train_overrides() -> None:
    config = task_content_probe_config_from_dict({"probe_train": {"steps": 42}})
    assert config.probe_train.steps == 42
    assert config.probe_train.lr == ProbeTrainConfig().lr


# --- _extract_features: position-location correctness ------------------------


def test_extract_features_content_labels_match_presented_input_tokens() -> None:
    """The content probe reads `content_state` at exactly the positions
    carrying `example.input_tokens` -- this must hold regardless of whether
    the model has learned anything, so a near-untrained (steps=1) core is
    enough to check it cheaply."""
    trained = train_shared_core(_tiny_shared_core_config(seed=0, train=SharedCoreGateTrainConfig(
        steps=1, batch_size=8, eval_every=1, progress_eval_examples=4, device="cpu"
    )))
    examples = trained.generator.generate_online(32, step=0, split="test")
    extracted = _extract_features(trained, examples, max_content_length=6)

    expected_labels = [token for example in examples for token in example.input_tokens]
    assert extracted.content_labels.tolist() == expected_labels


def test_extract_features_operation_ids_match_task_spec() -> None:
    from apc.environments.task_spec import operation_id

    trained = train_shared_core(_tiny_shared_core_config(seed=0, train=SharedCoreGateTrainConfig(
        steps=1, batch_size=8, eval_every=1, progress_eval_examples=4, device="cpu"
    )))
    examples = trained.generator.generate_online(32, step=0, split="test")
    extracted = _extract_features(trained, examples, max_content_length=6)

    expected = [operation_id(example.task_spec.operation_sequence[0]) for example in examples]
    assert extracted.operation_ids.tolist() == expected


def test_extract_features_select_targets_are_multi_hot_at_sampled_indices() -> None:
    trained = train_shared_core(_tiny_shared_core_config(seed=0, train=SharedCoreGateTrainConfig(
        steps=1, batch_size=8, eval_every=1, progress_eval_examples=4, device="cpu"
    )))
    examples = trained.generator.generate_online(256, step=0, split="test")
    extracted = _extract_features(trained, examples, max_content_length=6)
    assert extracted.select_targets is not None
    assert extracted.select_masks is not None

    select_examples = [
        example
        for example in examples
        if example.task_spec.operation_sequence[0] == "SELECT"
    ]
    assert len(select_examples) == extracted.select_targets.shape[0]
    for row, example in enumerate(select_examples):
        (step,) = example.task_spec.steps
        content_length = len(example.input_tokens)
        mask = extracted.select_masks[row]
        assert mask[:content_length].tolist() == [1.0] * content_length
        assert mask[content_length:].tolist() == [0.0] * (6 - content_length)
        target = extracted.select_targets[row]
        expected_indices = set(step.arguments["indices"])
        selected = {i for i in range(content_length) if target[i].item() == 1.0}
        assert selected == expected_indices


def test_extract_features_scalar_targets_match_sampled_arguments() -> None:
    trained = train_shared_core(_tiny_shared_core_config(seed=0, train=SharedCoreGateTrainConfig(
        steps=1, batch_size=8, eval_every=1, progress_eval_examples=4, device="cpu"
    )))
    examples = trained.generator.generate_online(256, step=0, split="test")
    extracted = _extract_features(trained, examples, max_content_length=6)

    for name in SCALAR_ARGUMENT_OPERATIONS:
        matching = [
            example for example in examples if example.task_spec.operation_sequence[0] == name
        ]
        expected = [
            next(iter(example.task_spec.steps[0].arguments.values())) for example in matching
        ]
        assert extracted.scalar_targets[name].tolist() == expected


# --- run_task_content_probes: end-to-end structure ---------------------------


def test_run_returns_argument_probe_for_every_parameterized_operation() -> None:
    report = run_task_content_probes(_tiny_probe_config())
    assert set(report.argument_probes) == set(PARAMETERIZED_OPERATION_NAMES)
    for name in SCALAR_ARGUMENT_OPERATIONS:
        assert report.argument_probes[name].metric_name == "top1_accuracy"
    for name in SET_ARGUMENT_OPERATIONS:
        assert report.argument_probes[name].metric_name == "masked_slot_accuracy"


def test_run_report_fields_are_probabilities() -> None:
    report = run_task_content_probes(_tiny_probe_config())
    assert 0.0 <= report.operation_id_accuracy <= 1.0
    assert 0.0 <= report.content_token_accuracy <= 1.0
    assert 0.0 <= report.operation_from_content_accuracy <= 1.0
    assert 0.0 <= report.content_from_task_accuracy <= 1.0
    for result in report.argument_probes.values():
        assert 0.0 <= result.accuracy <= 1.0


def test_run_passed_requires_all_three_required_probes() -> None:
    report = run_task_content_probes(_tiny_probe_config())
    expected = (
        report.operation_id_passed and report.argument_probes_passed and report.content_probe_passed
    )
    assert report.passed == expected


def test_run_report_to_dict_round_trips_through_json() -> None:
    report = run_task_content_probes(_tiny_probe_config())
    raw = json.loads(json.dumps(report.to_dict()))
    assert set(raw["argument_probes"]) == set(PARAMETERIZED_OPERATION_NAMES)
    assert raw["passed"] == report.passed


def _dict_without_wall_clock(report) -> dict:
    raw = report.to_dict()
    del raw["wall_clock_seconds"]
    return raw


def test_run_is_deterministic_given_the_same_seed() -> None:
    report_a = run_task_content_probes(_tiny_probe_config(seed=1))
    report_b = run_task_content_probes(_tiny_probe_config(seed=1))
    assert _dict_without_wall_clock(report_a) == _dict_without_wall_clock(report_b)


def test_run_different_seeds_diverge() -> None:
    report_a = run_task_content_probes(_tiny_probe_config(seed=1))
    report_b = run_task_content_probes(_tiny_probe_config(seed=2))
    assert _dict_without_wall_clock(report_a) != _dict_without_wall_clock(report_b)


def test_run_raises_when_probe_batch_is_too_small_for_an_operation() -> None:
    config = _tiny_probe_config(
        num_probe_train_examples=8, num_probe_eval_examples=8, min_examples_per_operation=16
    )
    with pytest.raises(ValueError, match="fewer than"):
        run_task_content_probes(config)


def test_metrics_path_writes_shared_core_training_progress(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    run_task_content_probes(_tiny_probe_config(), metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    first = json.loads(lines[0])
    assert {"step", "loss", "progress_overall_exact_match"} <= set(first)


# --- run_task_content_probes_multi_seed --------------------------------------


def test_multi_seed_runs_one_report_per_seed(tmp_path: Path) -> None:
    result = run_task_content_probes_multi_seed(
        _tiny_probe_config(), seeds=(1, 2), run_dir=tmp_path
    )
    assert result.seeds == (1, 2)
    assert len(result.per_seed) == 2
    assert [report.config.seed for report in result.per_seed] == [1, 2]
    assert (tmp_path / "seed_1" / "metrics.jsonl").exists()
    assert (tmp_path / "seed_2" / "metrics.jsonl").exists()


def test_multi_seed_overrides_shared_core_seed_too() -> None:
    result = run_task_content_probes_multi_seed(_tiny_probe_config(), seeds=(3, 4))
    for report in result.per_seed:
        assert report.config.shared_core.seed == report.config.seed


def test_multi_seed_passed_is_and_of_per_seed_passed() -> None:
    result = run_task_content_probes_multi_seed(_tiny_probe_config(), seeds=(1, 2))
    assert result.passed == all(report.passed for report in result.per_seed)


def test_multi_seed_report_to_dict_round_trips_through_json() -> None:
    result = run_task_content_probes_multi_seed(_tiny_probe_config(), seeds=(1,))
    raw = json.loads(json.dumps(result.to_dict()))
    assert raw["seeds"] == [1]
