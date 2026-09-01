"""Tests for the parameterized oracle primitive benchmark (Phase A.1
Post-Correction Task A1-R005, STOP GATE, H2c).

Fast, small-step tests only -- mirroring `tests/
test_parameter_free_primitive_gate.py`'s convention of exercising the wiring
(config validation, primitive-bank construction, the wrong-argument/
wrong-family providers, report shape, multi-seed aggregation) with a tiny
model and a handful of training steps. The gate's own scientific claim (5
seeds, the full training budget) is run via `scripts/
parameterized_primitive_gate.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.environments.generator import TaskGenerator, oracle_call_for_example
from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import operation_id
from apc.evaluation.parameterized_primitive_gate import (
    CORRECT_THRESHOLD,
    DEFAULT_SEEDS,
    MIN_CAUSAL_GAP,
    MIN_GATE_SEEDS,
    NONE_CEILING,
    WRONG_ARGUMENT_CEILING,
    WRONG_FAMILY_CEILING,
    ParameterizedPrimitiveGateConfig,
    ParameterizedPrimitiveGateReport,
    PrimitiveTrainConfig,
    _build_primitive_bank,
    _default_wrong_operation_map,
    _wrong_argument_call_provider,
    _wrong_argument_value,
    _wrong_family_call_provider,
    parameterized_primitive_gate_config_from_dict,
    run_parameterized_primitive_gate,
    run_parameterized_primitive_gate_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig
from apc.primitives.conditioning import ConditionedPrimitive
from apc.primitives.primitive import PrimitiveStatus


def _tiny_config(**overrides: object) -> ParameterizedPrimitiveGateConfig:
    base = ParameterizedPrimitiveGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=PARAMETERIZED_OPERATION_NAMES,
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
        num_unseen_eval_examples=128,
        min_examples_per_operation=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- ParameterizedPrimitiveGateConfig validation ------------------------------


def test_config_rejects_parameter_free_operations() -> None:
    with pytest.raises(ValueError, match="exactly one argument"):
        _tiny_config(operation_names=("SHIFT", "COPY"))


def test_config_rejects_fewer_than_two_operations() -> None:
    with pytest.raises(ValueError, match="Wrong family"):
        _tiny_config(operation_names=("SHIFT",))


def test_config_rejects_max_sequence_length_smaller_than_length_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(sequence_length_range=(4, 20), max_sequence_length=8)


def test_config_defaults_to_parameterized_operation_names() -> None:
    config = ParameterizedPrimitiveGateConfig()
    assert config.operation_names == PARAMETERIZED_OPERATION_NAMES


# --- parameterized_primitive_gate_config_from_dict ----------------------------


def test_config_from_dict_fills_in_defaults() -> None:
    config = parameterized_primitive_gate_config_from_dict({"seed": 3})
    assert config.seed == 3
    assert config.operation_names == PARAMETERIZED_OPERATION_NAMES
    assert config.primitive_rank == ParameterizedPrimitiveGateConfig().primitive_rank


def test_config_from_dict_parses_nested_train_configs() -> None:
    raw = {
        "core_train": {"steps": 5},
        "primitive_train": {"steps": 7, "lr": 0.001},
        "arg_dim": 12,
    }
    config = parameterized_primitive_gate_config_from_dict(raw)
    assert config.core_train.steps == 5
    assert config.primitive_train.steps == 7
    assert config.primitive_train.lr == 0.001
    assert config.arg_dim == 12


# --- _default_wrong_operation_map ---------------------------------------------


def test_default_wrong_operation_map_is_a_derangement() -> None:
    names = PARAMETERIZED_OPERATION_NAMES
    wrong_map = _default_wrong_operation_map(names)
    assert set(wrong_map) == set(names)
    for name in names:
        assert wrong_map[name] != name
        assert wrong_map[name] in names


def test_default_wrong_operation_map_is_deterministic() -> None:
    names = PARAMETERIZED_OPERATION_NAMES
    assert _default_wrong_operation_map(names) == _default_wrong_operation_map(names)


# --- _wrong_argument_value / _wrong_argument_call_provider --------------------


@pytest.mark.parametrize("operation", PARAMETERIZED_OPERATION_NAMES)
def test_wrong_argument_value_differs_from_correct(operation: str) -> None:
    generator = TaskGenerator(seed=0, operation_names=(operation,), max_depth=1)
    example = generator.generate(1, "train")[0]
    correct = oracle_call_for_example(example)

    wrong_arguments = _wrong_argument_value(correct, example)

    assert wrong_arguments != correct.arguments
    # Must remain a well-formed call for the same operation.
    PrimitiveCall(operation=operation, arguments=wrong_arguments)


def test_wrong_argument_call_provider_keeps_the_correct_operation() -> None:
    generator = TaskGenerator(seed=0, operation_names=("SHIFT",), max_depth=1)
    example = generator.generate(1, "train")[0]
    provider = _wrong_argument_call_provider()

    call = provider(example)

    assert call.operation == "SHIFT"
    assert call.arguments != oracle_call_for_example(example).arguments


# --- _wrong_family_call_provider -----------------------------------------------


def test_wrong_family_call_provider_forces_a_different_well_formed_family() -> None:
    generator = TaskGenerator(seed=0, operation_names=("SHIFT",), max_depth=1)
    example = generator.generate(1, "train")[0]
    wrong_map = _default_wrong_operation_map(PARAMETERIZED_OPERATION_NAMES)
    provider = _wrong_family_call_provider(wrong_map)

    call = provider(example)

    assert call.operation == wrong_map["SHIFT"]
    assert call.operation != "SHIFT"


# --- _build_primitive_bank ------------------------------------------------------


def test_build_primitive_bank_registers_one_conditioned_primitive_per_operation() -> None:
    bank = _build_primitive_bank(
        PARAMETERIZED_OPERATION_NAMES,
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    assert len(bank) == len(PARAMETERIZED_OPERATION_NAMES)
    for name in PARAMETERIZED_OPERATION_NAMES:
        primitive = bank.get(operation_id(name))
        assert isinstance(primitive, ConditionedPrimitive)
        assert primitive.config.d_model == 16
        assert primitive.config.rank == 4
        assert primitive.status == PrimitiveStatus.CANDIDATE
        assert primitive.enabled is True


def test_build_primitive_bank_primitives_are_trainable() -> None:
    bank = _build_primitive_bank(
        PARAMETERIZED_OPERATION_NAMES,
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    for pid in bank.ids():
        assert not bank.get(pid).is_frozen()


def test_build_primitive_bank_size_does_not_grow_with_argument_values() -> None:
    bank = _build_primitive_bank(
        PARAMETERIZED_OPERATION_NAMES,
        d_model=16,
        rank=4,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    size_before = len(bank)
    h = torch.randn(1, 16)
    shift = bank.get(operation_id("SHIFT"))
    for amount in range(5):
        shift(h, argument_values=[amount])
        assert len(bank) == size_before


# --- run_parameterized_primitive_gate ------------------------------------------


def test_run_returns_report_with_expected_fields() -> None:
    report = run_parameterized_primitive_gate(_tiny_config())
    assert isinstance(report, ParameterizedPrimitiveGateReport)
    assert report.core_steps_trained == 3
    assert report.primitive_steps_trained == 3
    assert report.num_unseen_eval_examples == 128
    assert report.bank_size == len(PARAMETERIZED_OPERATION_NAMES)
    assert set(report.per_operation_correct_exact_match) == set(PARAMETERIZED_OPERATION_NAMES)
    assert set(report.per_operation_wrong_argument_exact_match) == set(
        PARAMETERIZED_OPERATION_NAMES
    )
    assert set(report.per_operation_wrong_family_exact_match) == set(
        PARAMETERIZED_OPERATION_NAMES
    )
    assert set(report.per_operation_none_exact_match) == set(PARAMETERIZED_OPERATION_NAMES)
    for value in (
        report.correct_exact_match,
        report.wrong_argument_exact_match,
        report.wrong_family_exact_match,
        report.none_exact_match,
    ):
        assert 0.0 <= value <= 1.0


def test_run_freezes_every_stable_core_parameter() -> None:
    report = run_parameterized_primitive_gate(_tiny_config())
    assert report.core_trainable_param_count == 0
    assert report.core_param_count > 0


def test_run_primitive_bank_has_nonzero_parameter_count() -> None:
    report = run_parameterized_primitive_gate(_tiny_config())
    assert report.primitive_param_count > 0


def test_run_causal_gap_matches_correct_minus_max_controls() -> None:
    report = run_parameterized_primitive_gate(_tiny_config())
    expected = report.correct_exact_match - max(
        report.wrong_argument_exact_match, report.wrong_family_exact_match, report.none_exact_match
    )
    assert report.causal_gap == pytest.approx(expected)


def test_run_is_deterministic_given_the_same_seed() -> None:
    first = run_parameterized_primitive_gate(_tiny_config())
    second = run_parameterized_primitive_gate(_tiny_config())
    assert first.correct_exact_match == second.correct_exact_match
    assert first.wrong_argument_exact_match == second.wrong_argument_exact_match
    assert first.wrong_family_exact_match == second.wrong_family_exact_match
    assert first.none_exact_match == second.none_exact_match


def test_run_uses_injected_wrong_operation_map() -> None:
    names = PARAMETERIZED_OPERATION_NAMES
    fixed_map = {name: names[(i + 2) % len(names)] for i, name in enumerate(names)}
    report = run_parameterized_primitive_gate(_tiny_config(), wrong_operation_map=fixed_map)
    assert report.wrong_operation_map == fixed_map


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_parameterized_primitive_gate(_tiny_config())
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["correct_exact_match"] == report.correct_exact_match
    assert payload["wrong_operation_map"] == report.wrong_operation_map


# --- run_parameterized_primitive_gate_multi_seed --------------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    seeds = (0, 1)
    result = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=seeds, run_dir=tmp_path
    )
    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    for seed in seeds:
        assert (tmp_path / f"seed_{seed}" / "report.json").exists()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    result = run_parameterized_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert [report.config.seed for report in result.per_seed] == [0, 1, 2]


def test_multi_seed_mean_causal_gap_matches_mean_arms() -> None:
    result = run_parameterized_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    expected = result.mean_correct_exact_match - max(
        result.mean_wrong_argument_exact_match,
        result.mean_wrong_family_exact_match,
        result.mean_none_exact_match,
    )
    assert result.mean_causal_gap == pytest.approx(expected)


def test_multi_seed_family_count_passed_reflects_bank_size() -> None:
    result = run_parameterized_primitive_gate_multi_seed(_tiny_config(), seeds=(0,))
    assert result.family_count_passed is True
    assert all(
        report.bank_size == len(PARAMETERIZED_OPERATION_NAMES) for report in result.per_seed
    )


def test_multi_seed_passed_requires_all_five_conditions() -> None:
    """Force each of the five gating conditions to fail in turn via an
    unreachable threshold, without depending on real training dynamics."""
    fail_correct = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), correct_threshold=2.0
    )
    assert fail_correct.correct_passed is False
    assert fail_correct.passed is False

    fail_wrong_argument = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), wrong_argument_ceiling=-1.0
    )
    assert fail_wrong_argument.wrong_argument_passed is False
    assert fail_wrong_argument.passed is False

    fail_wrong_family = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), wrong_family_ceiling=-1.0
    )
    assert fail_wrong_family.wrong_family_passed is False
    assert fail_wrong_family.passed is False

    fail_none = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), none_ceiling=-1.0
    )
    assert fail_none.none_passed is False
    assert fail_none.passed is False

    fail_gap = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0,), min_causal_gap=2.0
    )
    assert fail_gap.causal_gap_passed is False
    assert fail_gap.passed is False


def test_multi_seed_meets_seed_policy_reflects_seed_count() -> None:
    short = run_parameterized_primitive_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert short.meets_seed_policy is False
    full = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=tuple(range(MIN_GATE_SEEDS))
    )
    assert full.meets_seed_policy is True


def test_multi_seed_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    result = run_parameterized_primitive_gate_multi_seed(
        _tiny_config(), seeds=(0, 1), run_dir=tmp_path
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] == result.passed
    assert len(payload["per_seed"]) == 2


# --- module constants -----------------------------------------------------------


def test_default_seeds_matches_min_gate_seeds_length() -> None:
    assert len(DEFAULT_SEEDS) == MIN_GATE_SEEDS


def test_thresholds_match_h2c() -> None:
    assert CORRECT_THRESHOLD == 0.90
    assert WRONG_ARGUMENT_CEILING == 0.30
    assert WRONG_FAMILY_CEILING == 0.30
    assert NONE_CEILING == 0.30
    assert MIN_CAUSAL_GAP == 0.50


def test_no_cuda_required_for_cpu_device_override() -> None:
    """Sanity check that the tiny fixture config actually runs on CPU (no
    hidden CUDA dependency), matching every other Phase A.1 gate's test
    convention."""
    report = run_parameterized_primitive_gate(_tiny_config())
    assert report.device == str(torch.device("cpu"))
