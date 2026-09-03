"""Tests for the oracle latent operator benchmark (Phase A.1 diagnostic Task
A1-R005E-003).

Fast, small-step tests only -- mirroring `tests/test_representation_audit.py`'s
convention of exercising the wiring (config validation, per-operation frozen-
core pretraining, oracle-position derivation, report shape, multi-seed
aggregation) with a tiny model and a handful of training steps. The
benchmark's own scientific claim (the retry's full `core_train` budget, 3
seeds) is run via `scripts/oracle_latent_operator_benchmark.py`, not asserted
here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.environments.generator import oracle_call_for_example
from apc.evaluation.oracle_latent_operator_benchmark import (
    COUNT_READOUT_OPERATIONS,
    DEFAULT_SEEDS,
    EXACT_MATCH_THRESHOLD,
    TOKEN_ACCURACY_GATED_OPERATIONS,
    TOKEN_ACCURACY_THRESHOLD,
    OracleLatentOperatorConfig,
    OracleLatentOperatorReport,
    ProbeTrainConfig,
    _bind_value_position,
    _count_match_positions,
    _evaluate_oracle_decode,
    _pretrain_frozen_stable_core,
    _run_operation_benchmark,
    _select_source_positions,
    _shift_source_positions,
    _source_positions_for_example,
    oracle_latent_operator_config_from_dict,
    run_oracle_latent_operator_benchmark,
    run_oracle_latent_operator_benchmark_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig


def _tiny_config(**overrides: object) -> OracleLatentOperatorConfig:
    base = OracleLatentOperatorConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=("SHIFT", "COUNT"),
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
        num_eval_examples=16,
        count_readout_train=ProbeTrainConfig(steps=5, lr=0.03, weight_decay=0.0),
        num_count_readout_train_examples=32,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- Config validation / round-trip -----------------------------------------


def test_config_validation_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=())


def test_config_validation_rejects_non_positive_example_counts() -> None:
    with pytest.raises(ValueError, match="num_eval_examples"):
        _tiny_config(num_eval_examples=0)
    with pytest.raises(ValueError, match="num_count_readout_train_examples"):
        _tiny_config(num_count_readout_train_examples=0)


def test_config_from_dict_round_trips_defaults_and_overrides() -> None:
    raw = {
        "seed": 2,
        "vocab_size": 6,
        "sequence_length_range": [4, 6],
        "operation_names": ["SHIFT"],
        "model": {"d_model": 16, "n_layer": 2, "n_head": 2, "d_ff": 32, "max_seq_len": 32},
        "core_train": {"steps": 3, "batch_size": 4},
        "num_eval_examples": 16,
        "count_readout_train": {"steps": 5},
        "num_count_readout_train_examples": 32,
    }
    config = oracle_latent_operator_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("SHIFT",)
    assert config.model["d_model"] == 16
    assert config.core_train.steps == 3
    assert config.num_eval_examples == 16
    assert config.count_readout_train.steps == 5
    assert config.num_count_readout_train_examples == 32

    defaulted = oracle_latent_operator_config_from_dict({})
    assert defaulted == OracleLatentOperatorConfig()


def test_default_operation_names_are_the_four_parameterized_operations() -> None:
    assert set(OracleLatentOperatorConfig().operation_names) == {
        "SHIFT",
        "SELECT",
        "COUNT",
        "BIND",
    }


# --- Pure oracle-position derivation (no model needed) ----------------------


def test_shift_source_positions_matches_shiftop_rotation() -> None:
    length, amount = 5, 2
    positions = _shift_source_positions(length, amount)
    input_tokens = (10, 11, 12, 13, 14)
    # ShiftOp.apply: sequence[amount:] + sequence[:amount]
    expected = input_tokens[amount:] + input_tokens[:amount]
    assert tuple(input_tokens[p] for p in positions) == expected


def test_shift_source_positions_normalizes_amount_outside_range() -> None:
    # ShiftOp.apply normalizes params["amount"] % len(sequence); the oracle
    # mapping must match even if amount >= length.
    assert _shift_source_positions(4, 4) == _shift_source_positions(4, 0)
    assert _shift_source_positions(4, 5) == _shift_source_positions(4, 1)


def test_select_source_positions_is_identity_gather() -> None:
    indices = [0, 2, 3]
    assert _select_source_positions(indices) == indices


def test_bind_value_position_returns_key_plus_one() -> None:
    input_tokens = (1, 100, 2, 200, 3, 300)
    assert _bind_value_position(input_tokens, 2) == 3
    assert input_tokens[_bind_value_position(input_tokens, 2)] == 200


def test_bind_value_position_last_match_wins_on_repeated_key() -> None:
    # BindOp.apply iterates ascending and keeps overwriting on a match, so a
    # repeated key resolves to the *last* (highest-index) pair.
    input_tokens = (5, 111, 5, 222, 5, 333)
    assert _bind_value_position(input_tokens, 5) == 5
    assert input_tokens[_bind_value_position(input_tokens, 5)] == 333


def test_bind_value_position_raises_for_absent_key() -> None:
    with pytest.raises(ValueError, match="not found"):
        _bind_value_position((1, 100, 2, 200), 9)


def test_count_match_positions_finds_every_occurrence() -> None:
    input_tokens = (3, 1, 3, 3, 2)
    assert _count_match_positions(input_tokens, 3) == [0, 2, 3]
    assert _count_match_positions(input_tokens, 9) == []


def test_source_positions_for_example_rejects_count() -> None:
    class _FakeExample:
        input_tokens = (1, 2, 3)

    with pytest.raises(ValueError, match="COUNT"):
        _source_positions_for_example("COUNT", _FakeExample(), call=None)  # type: ignore[arg-type]


# --- Real generated examples: oracle mapping reconstructs target ------------


def test_shift_oracle_positions_reconstruct_target_from_raw_tokens() -> None:
    config = _tiny_config(operation_names=("SHIFT",))
    core = _pretrain_frozen_stable_core(config, "SHIFT")
    examples = core.generator.generate_online(24, step=10_000, split="test")
    for example in examples:
        call = oracle_call_for_example(example)
        positions = _source_positions_for_example("SHIFT", example, call)
        reconstructed = tuple(example.input_tokens[p] for p in positions)
        assert reconstructed == example.target_tokens


def test_select_oracle_positions_reconstruct_target_from_raw_tokens() -> None:
    config = _tiny_config(operation_names=("SELECT",))
    core = _pretrain_frozen_stable_core(config, "SELECT")
    examples = core.generator.generate_online(24, step=10_000, split="test")
    for example in examples:
        call = oracle_call_for_example(example)
        positions = _source_positions_for_example("SELECT", example, call)
        reconstructed = tuple(example.input_tokens[p] for p in positions)
        assert reconstructed == example.target_tokens


def test_bind_oracle_position_reconstructs_target_from_raw_tokens() -> None:
    config = _tiny_config(operation_names=("BIND",))
    core = _pretrain_frozen_stable_core(config, "BIND")
    examples = core.generator.generate_online(24, step=10_000, split="test")
    for example in examples:
        call = oracle_call_for_example(example)
        positions = _source_positions_for_example("BIND", example, call)
        reconstructed = tuple(example.input_tokens[p] for p in positions)
        assert reconstructed == example.target_tokens


def test_count_match_positions_count_matches_target_from_raw_tokens() -> None:
    config = _tiny_config(operation_names=("COUNT",))
    core = _pretrain_frozen_stable_core(config, "COUNT")
    examples = core.generator.generate_online(24, step=10_000, split="test")
    for example in examples:
        call = oracle_call_for_example(example)
        matches = _count_match_positions(example.input_tokens, call.arguments["target"])
        expected = min(len(matches), config.vocab_size - 1)
        assert expected == example.target_tokens[0]


# --- Frozen-core pretraining -------------------------------------------------


def test_pretrain_frozen_stable_core_freezes_every_parameter() -> None:
    config = _tiny_config()
    core = _pretrain_frozen_stable_core(config, "SHIFT")
    assert core.operation == "SHIFT"
    assert all(not p.requires_grad for p in core.model.parameters())
    assert core.model.training is False


def test_pretrain_frozen_stable_core_is_deterministic_given_same_seed() -> None:
    config = _tiny_config()
    core_a = _pretrain_frozen_stable_core(config, "SHIFT")
    core_b = _pretrain_frozen_stable_core(config, "SHIFT")
    for p_a, p_b in zip(core_a.model.parameters(), core_b.model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
    assert core_a.final_core_train_loss == core_b.final_core_train_loss


# --- Per-operation / multi-seed report shape --------------------------------


def test_evaluate_oracle_decode_returns_metrics_in_unit_range() -> None:
    config = _tiny_config(operation_names=("SHIFT",))
    core = _pretrain_frozen_stable_core(config, "SHIFT")
    examples = core.generator.generate_online(16, step=20_000, split="test")
    exact_match, token_accuracy = _evaluate_oracle_decode(
        core.model, core.tokens, examples, "SHIFT", device=core.device
    )
    assert 0.0 <= exact_match <= 1.0
    assert 0.0 <= token_accuracy <= 1.0


def test_run_operation_benchmark_shape_for_non_count_operation() -> None:
    config = _tiny_config(operation_names=("SHIFT",))
    report = _run_operation_benchmark(config, "SHIFT")
    assert report.operation == "SHIFT"
    assert report.readout_trained is False
    assert 0.0 <= report.exact_match <= 1.0
    assert 0.0 <= report.token_accuracy <= 1.0
    assert report.exact_match_passed == (report.exact_match >= EXACT_MATCH_THRESHOLD)
    assert report.token_accuracy_passed == (report.token_accuracy >= TOKEN_ACCURACY_THRESHOLD)
    assert report.passed == (report.exact_match_passed and report.token_accuracy_passed)
    assert report.num_eval_examples == config.num_eval_examples


def test_run_operation_benchmark_shape_for_count_uses_trained_readout() -> None:
    config = _tiny_config(operation_names=("COUNT",))
    report = _run_operation_benchmark(config, "COUNT")
    assert report.operation == "COUNT"
    assert report.readout_trained is True
    # CountOp.output_length is always 1: token accuracy == exact match.
    assert report.token_accuracy == report.exact_match
    assert report.token_accuracy_passed is None
    assert report.passed == report.exact_match_passed


def test_run_oracle_latent_operator_benchmark_report_shape() -> None:
    config = _tiny_config()
    report = run_oracle_latent_operator_benchmark(config)
    assert isinstance(report, OracleLatentOperatorReport)
    assert set(report.per_operation) == {"SHIFT", "COUNT"}
    json.dumps(report.to_dict())


def test_run_oracle_latent_operator_benchmark_multi_seed_aggregates(tmp_path: Path) -> None:
    base_config = _tiny_config()
    seeds = (0, 1)
    result = run_oracle_latent_operator_benchmark_multi_seed(
        base_config, seeds=seeds, run_dir=tmp_path
    )

    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    assert set(result.per_operation_summary) == {"SHIFT", "COUNT"}

    shift_summary = result.per_operation_summary["SHIFT"]
    expected_mean = sum(
        report.per_operation["SHIFT"].exact_match for report in result.per_seed
    ) / len(result.per_seed)
    assert shift_summary.mean_exact_match == pytest.approx(expected_mean)
    assert shift_summary.token_accuracy_passed is not None
    assert shift_summary.readout_trained is False

    count_summary = result.per_operation_summary["COUNT"]
    assert count_summary.token_accuracy_passed is None
    assert count_summary.readout_trained is True

    assert result.passed == all(s.passed for s in result.per_operation_summary.values())

    for seed in seeds:
        seed_report_path = tmp_path / f"seed_{seed}" / "report.json"
        assert seed_report_path.is_file()
        json.loads(seed_report_path.read_text(encoding="utf-8"))

    json.dumps(result.to_dict())


def test_default_seeds_and_threshold_constants() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert COUNT_READOUT_OPERATIONS == ("COUNT",)
    assert TOKEN_ACCURACY_GATED_OPERATIONS == ("SHIFT", "SELECT", "BIND")
    assert EXACT_MATCH_THRESHOLD == 0.90
    assert TOKEN_ACCURACY_THRESHOLD == 0.98
