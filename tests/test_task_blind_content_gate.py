from __future__ import annotations

import json
from pathlib import Path

import pytest

from apc.environments.operations import KNOWN_OPERATION_NAMES
from apc.evaluation.task_blind_content_gate import (
    TaskBlindContentGateConfig,
    TaskBlindContentGateReport,
    _build_tokens,
    _derive_local_seed,
    _task_token_id_set,
    run_task_blind_content_gate,
    run_task_blind_content_gate_multi_seed,
    task_blind_content_gate_config_from_dict,
)


def _tiny_config(**overrides: object) -> TaskBlindContentGateConfig:
    defaults: dict[str, object] = {
        "seed": 0,
        "vocab_size": 6,
        "sequence_length_range": (6, 9),
        "num_contents_per_operation": 2,
        "model": {
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        "device": "cpu",
    }
    defaults.update(overrides)
    return TaskBlindContentGateConfig(**defaults)  # type: ignore[arg-type]


# --- TaskBlindContentGateConfig ---------------------------------------------


def test_config_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=())


def test_config_rejects_non_positive_num_contents_per_operation() -> None:
    with pytest.raises(ValueError, match="num_contents_per_operation"):
        _tiny_config(num_contents_per_operation=0)


def test_config_rejects_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol"):
        _tiny_config(atol=-1.0)


def test_config_from_dict_fills_defaults() -> None:
    config = task_blind_content_gate_config_from_dict({"seed": 3})
    defaults = TaskBlindContentGateConfig()
    assert config.seed == 3
    assert config.vocab_size == defaults.vocab_size
    assert config.operation_names == defaults.operation_names


# --- _task_token_id_set ------------------------------------------------------


def test_task_token_id_set_excludes_content_and_control_token_ids() -> None:
    tokens = _build_tokens(_tiny_config())
    task_ids = _task_token_id_set(tokens)
    # BOS/SEP/EOS/PAD and every environment content id must never collide
    # with a task-segment/operation/argument id.
    assert tokens.bos not in task_ids
    assert tokens.sep not in task_ids
    assert tokens.eos not in task_ids
    assert tokens.pad not in task_ids
    for content_id in range(tokens.env_vocab_size):
        assert content_id not in task_ids


def test_task_token_id_set_includes_every_operation_and_argument_token() -> None:
    tokens = _build_tokens(_tiny_config())
    task_ids = _task_token_id_set(tokens)
    assert tokens.task_start in task_ids
    assert tokens.task_end in task_ids
    assert all(tokens.operation_token(i) in task_ids for i in range(tokens.num_operations))
    assert all(tokens.argument_value_token(v) in task_ids for v in range(tokens.arg_span))


# --- _derive_local_seed ------------------------------------------------------


def test_derive_local_seed_is_deterministic() -> None:
    assert _derive_local_seed(0, 3, "SHIFT") == _derive_local_seed(0, 3, "SHIFT")


def test_derive_local_seed_varies_with_each_input() -> None:
    base = _derive_local_seed(0, 3, "SHIFT")
    assert _derive_local_seed(1, 3, "SHIFT") != base
    assert _derive_local_seed(0, 4, "SHIFT") != base
    assert _derive_local_seed(0, 3, "COUNT") != base


# --- run_task_blind_content_gate --------------------------------------------


def test_run_task_blind_content_gate_passes_on_a_tiny_config() -> None:
    report = run_task_blind_content_gate(_tiny_config())
    assert isinstance(report, TaskBlindContentGateReport)
    assert report.passed
    assert report.token_level_invariant
    assert report.representation_invariant
    assert report.padding_invariant
    assert report.no_task_token_leak
    assert report.all_operations_covered


def test_run_task_blind_content_gate_covers_every_configured_operation() -> None:
    report = run_task_blind_content_gate(_tiny_config())
    covered = {
        name
        for name in KNOWN_OPERATION_NAMES
        if report.per_operation_content_counts[name] > 0
        or report.per_operation_alternate_counts[name] > 0
    }
    assert covered == set(KNOWN_OPERATION_NAMES)


def test_run_task_blind_content_gate_representation_difference_is_exactly_zero_on_cpu() -> None:
    """Re-running `encode_task_content_split` on the identical
    `content_ids` tensor must be bit-identical regardless of `task_ids` --
    this is a much stronger check than the configured `atol` requires, and
    catches any accidental dependency on task_ids. Padding invariance
    (a separately-shaped batched matmul) only needs to clear the
    configured `atol`, not bit-exactness."""
    report = run_task_blind_content_gate(_tiny_config())
    assert report.representation_max_absolute_difference == 0.0
    assert report.padding_max_absolute_difference <= report.config.atol


def test_run_task_blind_content_gate_single_operation_config_is_fully_covered() -> None:
    report = run_task_blind_content_gate(
        _tiny_config(operation_names=("COPY",), min_alternate_task_specs_per_content=1)
    )
    assert report.per_operation_content_counts == {"COPY": report.num_contents_tested}


def test_run_task_blind_content_gate_reports_per_content_records() -> None:
    config = _tiny_config(num_contents_per_operation=1, operation_names=("COPY", "SHIFT"))
    report = run_task_blind_content_gate(config)
    assert len(report.per_content) == report.num_contents_tested
    for record in report.per_content:
        assert record.token_level_invariant
        assert record.task_token_leak_ids == ()
        assert record.max_absolute_difference == 0.0
        assert len(record.alternate_operations) >= config.min_alternate_task_specs_per_content


# --- run_task_blind_content_gate_multi_seed ---------------------------------


def test_run_task_blind_content_gate_multi_seed_aggregates_and_passes() -> None:
    result = run_task_blind_content_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert result.seeds == (0, 1, 2)
    assert len(result.per_seed) == 3
    assert result.passed
    assert result.max_representation_absolute_difference == 0.0


def test_run_task_blind_content_gate_multi_seed_reports_seed_policy() -> None:
    result = run_task_blind_content_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert result.meets_seed_policy is False
    result_full = run_task_blind_content_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2, 3, 4))
    assert result_full.meets_seed_policy is True


def test_run_task_blind_content_gate_multi_seed_writes_per_seed_reports(tmp_path: Path) -> None:
    run_dir = tmp_path
    run_task_blind_content_gate_multi_seed(_tiny_config(), seeds=(0, 1), run_dir=run_dir)
    for seed in (0, 1):
        report_path = run_dir / f"seed_{seed}" / "report.json"
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["config"]["seed"] == seed
