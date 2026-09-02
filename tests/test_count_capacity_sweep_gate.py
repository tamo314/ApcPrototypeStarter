"""Tests for the minimal capacity/training sweep gate (Phase A.1
Post-Correction Task A1-R005D-005).

Fast, small-step tests only -- mirroring `tests/
test_count_counterfactual_gate.py`'s convention. The sweep's real scientific
claim (5 seeds, the full A1-R005D-004 training budget per stage) is run via
`scripts/count_capacity_sweep_gate.py`, not asserted here. Stage-passing
behavior is exercised via unreachable/trivial threshold overrides rather
than real training dynamics, matching `tests/test_count_counterfactual_gate.
py::test_multi_seed_passed_requires_all_four_conditions`'s own convention.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from apc.evaluation.count_capacity_sweep_gate import (
    STAGE_NAMES,
    CapacitySweepStageReport,
    CountCapacitySweepReport,
    capacity_sweep_stage_configs,
    run_count_capacity_sweep_gate,
)
from apc.evaluation.count_counterfactual_gate import (
    CountCounterfactualGateConfig,
    PrimitiveTrainConfig,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig


def _tiny_config(**overrides: object) -> CountCounterfactualGateConfig:
    base = CountCounterfactualGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        model={
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        core_train=SharedCoreGateTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        primitive_rank=4,
        arg_dim=6,
        max_sequence_length=8,
        primitive_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


_ALWAYS_PASS = {
    "correct_threshold": -1.0,
    "effectful_wrong_argument_ceiling": 2.0,
    "none_ceiling": 2.0,
    "min_causal_gap": -2.0,
}
_NEVER_PASS = {"correct_threshold": 2.0}


# --- capacity_sweep_stage_configs -----------------------------------------


def test_stage_names_match_task_text_order() -> None:
    assert STAGE_NAMES == ("steps_x2", "steps_x4", "rank_16", "arg_dim_x2")


def test_stage_configs_returned_in_documented_order() -> None:
    stages = capacity_sweep_stage_configs(_tiny_config())
    assert tuple(name for name, _ in stages) == STAGE_NAMES


def test_steps_x2_only_changes_primitive_train_steps() -> None:
    base = _tiny_config()
    stages = dict(capacity_sweep_stage_configs(base))
    config = stages["steps_x2"]
    assert config.primitive_train.steps == base.primitive_train.steps * 2
    assert config.primitive_rank == base.primitive_rank
    assert config.arg_dim == base.arg_dim
    assert dataclasses.replace(config, primitive_train=base.primitive_train) == base


def test_steps_x4_only_changes_primitive_train_steps() -> None:
    base = _tiny_config()
    stages = dict(capacity_sweep_stage_configs(base))
    config = stages["steps_x4"]
    assert config.primitive_train.steps == base.primitive_train.steps * 4
    assert config.primitive_rank == base.primitive_rank
    assert config.arg_dim == base.arg_dim


def test_rank_16_only_changes_primitive_rank() -> None:
    base = _tiny_config()
    stages = dict(capacity_sweep_stage_configs(base))
    config = stages["rank_16"]
    assert config.primitive_rank == 16
    assert config.primitive_train.steps == base.primitive_train.steps
    assert config.arg_dim == base.arg_dim


def test_arg_dim_x2_only_changes_arg_dim() -> None:
    base = _tiny_config()
    stages = dict(capacity_sweep_stage_configs(base))
    config = stages["arg_dim_x2"]
    assert config.arg_dim == base.arg_dim * 2
    assert config.primitive_rank == base.primitive_rank
    assert config.primitive_train.steps == base.primitive_train.steps


def test_stage_configs_do_not_mutate_base_config() -> None:
    base = _tiny_config()
    before = dataclasses.replace(base)
    capacity_sweep_stage_configs(base)
    assert base == before


# --- run_count_capacity_sweep_gate ----------------------------------------


def test_sweep_stops_at_first_passing_stage(tmp_path: Path) -> None:
    result = run_count_capacity_sweep_gate(
        _tiny_config(), seeds=(0,), run_dir=tmp_path, **_ALWAYS_PASS
    )
    assert isinstance(result, CountCapacitySweepReport)
    assert len(result.stages) == 1
    assert result.stages[0].name == "steps_x2"
    assert result.selected_stage == "steps_x2"
    assert result.sweep_exhausted_without_pass is False
    assert (tmp_path / "steps_x2" / "report.json").exists()
    assert (tmp_path / "steps_x2" / "seed_0" / "report.json").exists()
    for later_stage in ("steps_x4", "rank_16", "arg_dim_x2"):
        assert not (tmp_path / later_stage).exists()


def test_sweep_runs_all_stages_when_none_pass(tmp_path: Path) -> None:
    result = run_count_capacity_sweep_gate(
        _tiny_config(), seeds=(0,), run_dir=tmp_path, **_NEVER_PASS
    )
    assert len(result.stages) == 4
    assert [stage.name for stage in result.stages] == list(STAGE_NAMES)
    assert result.selected_stage is None
    assert result.sweep_exhausted_without_pass is True
    for stage_name in STAGE_NAMES:
        assert (tmp_path / stage_name / "report.json").exists()
        assert (tmp_path / stage_name / "seed_0" / "report.json").exists()


def test_sweep_reports_swept_field_and_value_per_stage() -> None:
    result = run_count_capacity_sweep_gate(_tiny_config(), seeds=(0,), **_NEVER_PASS)
    expected = {
        "steps_x2": ("primitive_train.steps", 6),
        "steps_x4": ("primitive_train.steps", 12),
        "rank_16": ("primitive_rank", 16),
        "arg_dim_x2": ("arg_dim", 12),
    }
    for stage in result.stages:
        assert isinstance(stage, CapacitySweepStageReport)
        field, value = expected[stage.name]
        assert stage.swept_field == field
        assert stage.swept_value == value


def test_sweep_stage_reports_nonzero_primitive_param_count() -> None:
    result = run_count_capacity_sweep_gate(
        _tiny_config(), seeds=(0,), **_ALWAYS_PASS
    )
    assert all(stage.primitive_param_count > 0 for stage in result.stages)


def test_sweep_report_to_dict_round_trips_through_json() -> None:
    result = run_count_capacity_sweep_gate(_tiny_config(), seeds=(0,), **_ALWAYS_PASS)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["selected_stage"] == "steps_x2"
    assert payload["sweep_exhausted_without_pass"] is False
    assert payload["stages"][0]["name"] == "steps_x2"
    assert "multi_seed" in payload["stages"][0]
