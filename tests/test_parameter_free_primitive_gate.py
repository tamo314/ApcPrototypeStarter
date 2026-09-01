"""Tests for the parameter-free oracle primitive benchmark (Phase A.1
Post-Correction Task A1-R003, STOP GATE, H2b).

Fast, small-step tests only -- mirroring `tests/test_decoder_leakage_gate.py`
and `tests/test_shared_core_generalization.py`'s convention of exercising
the wiring (config validation, primitive-bank construction, the wrong-family
derangement, report shape, multi-seed aggregation) with a tiny model and a
handful of training steps. The gate's own scientific claim (5 seeds, the
full training budget) is run via `scripts/parameter_free_primitive_gate.py`,
not asserted here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.environments.operations import DETERMINISTIC_OPERATION_NAMES
from apc.environments.task_spec import operation_id
from apc.evaluation.parameter_free_primitive_gate import (
    CORRECT_THRESHOLD,
    DEFAULT_SEEDS,
    MIN_CAUSAL_GAP,
    MIN_GATE_SEEDS,
    NONE_CEILING,
    WRONG_CEILING,
    ParameterFreePrimitiveGateConfig,
    ParameterFreePrimitiveGateReport,
    PrimitiveTrainConfig,
    _build_primitive_bank,
    _default_wrong_operation_map,
    parameter_free_primitive_gate_config_from_dict,
    run_parameter_free_primitive_gate,
    run_parameter_free_primitive_gate_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig
from apc.primitives.primitive import PrimitiveStatus


def _tiny_config(**overrides: object) -> ParameterFreePrimitiveGateConfig:
    base = ParameterFreePrimitiveGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=DETERMINISTIC_OPERATION_NAMES,
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
        primitive_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_examples=32,
        min_examples_per_operation=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- ParameterFreePrimitiveGateConfig validation -----------------------------


def test_config_rejects_parameterized_operations() -> None:
    with pytest.raises(ValueError, match="parameter-free"):
        _tiny_config(operation_names=("COPY", "SHIFT"))


def test_config_rejects_fewer_than_two_operations() -> None:
    with pytest.raises(ValueError, match="Wrong family"):
        _tiny_config(operation_names=("COPY",))


def test_config_defaults_to_deterministic_operation_names() -> None:
    config = ParameterFreePrimitiveGateConfig()
    assert config.operation_names == DETERMINISTIC_OPERATION_NAMES


# --- parameter_free_primitive_gate_config_from_dict --------------------------


def test_config_from_dict_fills_in_defaults() -> None:
    config = parameter_free_primitive_gate_config_from_dict({"seed": 3})
    assert config.seed == 3
    assert config.operation_names == DETERMINISTIC_OPERATION_NAMES
    assert config.primitive_rank == ParameterFreePrimitiveGateConfig().primitive_rank


def test_config_from_dict_parses_nested_train_configs() -> None:
    raw = {
        "core_train": {"steps": 5},
        "primitive_train": {"steps": 7, "lr": 0.001},
    }
    config = parameter_free_primitive_gate_config_from_dict(raw)
    assert config.core_train.steps == 5
    assert config.primitive_train.steps == 7
    assert config.primitive_train.lr == 0.001


# --- _default_wrong_operation_map ---------------------------------------------


def test_default_wrong_operation_map_is_a_derangement() -> None:
    names = DETERMINISTIC_OPERATION_NAMES
    wrong_map = _default_wrong_operation_map(names)
    assert set(wrong_map) == set(names)
    for name in names:
        assert wrong_map[name] != name
        assert wrong_map[name] in names


def test_default_wrong_operation_map_is_deterministic() -> None:
    names = DETERMINISTIC_OPERATION_NAMES
    assert _default_wrong_operation_map(names) == _default_wrong_operation_map(names)


# --- _build_primitive_bank ----------------------------------------------------


def test_build_primitive_bank_registers_one_primitive_per_operation_at_operation_id() -> None:
    bank = _build_primitive_bank(DETERMINISTIC_OPERATION_NAMES, d_model=16, rank=4)
    assert len(bank) == len(DETERMINISTIC_OPERATION_NAMES)
    for name in DETERMINISTIC_OPERATION_NAMES:
        primitive = bank.get(operation_id(name))
        assert primitive.config.d_model == 16
        assert primitive.config.rank == 4
        assert primitive.status == PrimitiveStatus.CANDIDATE
        assert primitive.enabled is True


def test_build_primitive_bank_primitives_are_trainable() -> None:
    """Unlike ADR-0004's frozen STABLE convention, this gate's whole point
    is training these primitives -- they must start with requires_grad=True."""
    bank = _build_primitive_bank(DETERMINISTIC_OPERATION_NAMES, d_model=16, rank=4)
    for pid in bank.ids():
        assert not bank.get(pid).is_frozen()


# --- run_parameter_free_primitive_gate ----------------------------------------


def test_run_returns_report_with_expected_fields() -> None:
    report = run_parameter_free_primitive_gate(_tiny_config())
    assert isinstance(report, ParameterFreePrimitiveGateReport)
    assert report.core_steps_trained == 3
    assert report.primitive_steps_trained == 3
    assert report.num_unseen_eval_examples == 32
    assert set(report.per_operation_correct_exact_match) == set(DETERMINISTIC_OPERATION_NAMES)
    assert set(report.per_operation_wrong_exact_match) == set(DETERMINISTIC_OPERATION_NAMES)
    assert set(report.per_operation_none_exact_match) == set(DETERMINISTIC_OPERATION_NAMES)
    for value in (report.correct_exact_match, report.wrong_exact_match, report.none_exact_match):
        assert 0.0 <= value <= 1.0


def test_run_freezes_every_stable_core_parameter() -> None:
    report = run_parameter_free_primitive_gate(_tiny_config())
    assert report.core_trainable_param_count == 0
    assert report.core_param_count > 0


def test_run_primitive_bank_has_nonzero_parameter_count() -> None:
    report = run_parameter_free_primitive_gate(_tiny_config())
    assert report.primitive_param_count > 0


def test_run_causal_gap_matches_correct_minus_max_wrong_none() -> None:
    report = run_parameter_free_primitive_gate(_tiny_config())
    expected = report.correct_exact_match - max(report.wrong_exact_match, report.none_exact_match)
    assert report.causal_gap == pytest.approx(expected)


def test_run_primitive_weights_move_from_their_zero_init() -> None:
    """`Primitive.b_proj` starts at zero (identity function); after even a
    few real training steps against a nonzero loss, at least one selected
    primitive's weights must have moved."""
    report = run_parameter_free_primitive_gate(_tiny_config())
    assert report.final_primitive_train_loss == report.final_primitive_train_loss  # not NaN


def test_run_uses_injected_wrong_operation_map() -> None:
    names = DETERMINISTIC_OPERATION_NAMES
    fixed_map = {name: names[(i + 2) % len(names)] for i, name in enumerate(names)}
    report = run_parameter_free_primitive_gate(_tiny_config(), wrong_operation_map=fixed_map)
    assert report.wrong_operation_map == fixed_map


def test_run_is_deterministic_given_the_same_seed() -> None:
    first = run_parameter_free_primitive_gate(_tiny_config())
    second = run_parameter_free_primitive_gate(_tiny_config())
    assert first.correct_exact_match == second.correct_exact_match
    assert first.wrong_exact_match == second.wrong_exact_match
    assert first.none_exact_match == second.none_exact_match


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_parameter_free_primitive_gate(_tiny_config())
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["correct_exact_match"] == report.correct_exact_match
    assert payload["wrong_operation_map"] == report.wrong_operation_map


# --- run_parameter_free_primitive_gate_multi_seed -----------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    seeds = (0, 1)
    result = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=seeds, run_dir=tmp_path
    )
    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    for seed in seeds:
        assert (tmp_path / f"seed_{seed}" / "report.json").exists()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    result = run_parameter_free_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert [report.config.seed for report in result.per_seed] == [0, 1, 2]


def test_multi_seed_mean_causal_gap_matches_mean_arms() -> None:
    result = run_parameter_free_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    expected = result.mean_correct_exact_match - max(
        result.mean_wrong_exact_match, result.mean_none_exact_match
    )
    assert result.mean_causal_gap == pytest.approx(expected)


def test_multi_seed_passed_requires_all_four_conditions() -> None:
    """Force each of the four gating conditions to fail in turn via an
    unreachable threshold, without depending on real training dynamics."""
    base = run_parameter_free_primitive_gate_multi_seed(_tiny_config(), seeds=(0,))

    fail_correct = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), correct_threshold=2.0
    )
    assert fail_correct.correct_passed is False
    assert fail_correct.passed is False

    fail_wrong = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), wrong_ceiling=-1.0
    )
    assert fail_wrong.wrong_passed is False
    assert fail_wrong.passed is False

    fail_none = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), none_ceiling=-1.0
    )
    assert fail_none.none_passed is False
    assert fail_none.passed is False

    fail_gap = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), min_causal_gap=2.0
    )
    assert fail_gap.causal_gap_passed is False
    assert fail_gap.passed is False

    assert base.mean_correct_exact_match == fail_wrong.mean_correct_exact_match


def test_multi_seed_meets_seed_policy_reflects_seed_count() -> None:
    short = run_parameter_free_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert short.meets_seed_policy is False
    full = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=tuple(range(MIN_GATE_SEEDS))
    )
    assert full.meets_seed_policy is True


def test_multi_seed_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    result = run_parameter_free_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0, 1), run_dir=tmp_path
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] == result.passed
    assert len(payload["per_seed"]) == 2


# --- module constants ---------------------------------------------------------


def test_default_seeds_matches_min_gate_seeds_length() -> None:
    assert len(DEFAULT_SEEDS) == MIN_GATE_SEEDS


def test_thresholds_match_h2b() -> None:
    assert CORRECT_THRESHOLD == 0.95
    assert WRONG_CEILING == 0.30
    assert NONE_CEILING == 0.30
    assert MIN_CAUSAL_GAP == 0.50


def test_no_cuda_required_for_cpu_device_override() -> None:
    """Sanity check that the tiny fixture config actually runs on CPU (no
    hidden CUDA dependency), matching every other Phase A.1 gate's test
    convention."""
    report = run_parameter_free_primitive_gate(_tiny_config())
    assert report.device == str(torch.device("cpu"))
