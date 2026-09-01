"""Tests for the decoder leakage control gate (Phase A.1 Post-Correction
Task A1-R002, conditional STOP GATE).

Fast, small-step tests only -- mirroring `tests/
test_shared_core_generalization.py`'s convention of exercising the wiring
(config parsing, per-operation aggregation, report shape, multi-seed
aggregation) with a tiny model and a handful of training steps. The gate's
own scientific claim (5 seeds, the full training budget) is run via
`scripts/decoder_leakage_gate.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from apc.evaluation.decoder_leakage_gate import (
    DEFAULT_SEEDS,
    FUTURE_CORRECT_TARGET,
    MIN_GATE_SEEDS,
    NO_PRIMITIVE_MATERIAL_CEILING,
    DecoderLeakageGateReport,
    decoder_leakage_gate_config_from_dict,
    run_decoder_leakage_gate,
    run_decoder_leakage_gate_multi_seed,
)
from apc.evaluation.shared_core_generalization import (
    SharedCoreGateConfig,
    SharedCoreGateTrainConfig,
)


def _tiny_config(**overrides: object) -> SharedCoreGateConfig:
    base = SharedCoreGateConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=("COPY", "NEGATE"),
        include_task_spec=False,
        model={
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        train=SharedCoreGateTrainConfig(
            steps=2, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        num_unseen_eval_examples=16,
        min_examples_per_operation=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- decoder_leakage_gate_config_from_dict -----------------------------------


def test_config_from_dict_forces_include_task_spec_false() -> None:
    config = decoder_leakage_gate_config_from_dict({"include_task_spec": True})
    assert config.include_task_spec is False


def test_config_from_dict_forces_include_task_spec_false_even_when_absent() -> None:
    config = decoder_leakage_gate_config_from_dict({"seed": 3})
    assert config.include_task_spec is False
    assert config.seed == 3


# --- run_decoder_leakage_gate -------------------------------------------------


def test_run_rejects_include_task_spec_true() -> None:
    config = _tiny_config(include_task_spec=True)
    with pytest.raises(ValueError, match="include_task_spec"):
        run_decoder_leakage_gate(config)


def test_run_returns_report_with_expected_fields() -> None:
    report = run_decoder_leakage_gate(_tiny_config())
    assert isinstance(report, DecoderLeakageGateReport)
    assert report.steps_trained == 2
    assert report.examples_seen == 8
    assert report.num_unseen_eval_examples == 16
    assert set(report.per_operation_exact_match) == {"COPY", "NEGATE"}
    assert 0.0 <= report.overall_exact_match <= 1.0


def test_run_no_primitive_causal_path_matches_plain_generation() -> None:
    """The gate's own cross-check: the audited no-primitive causal-mode
    path and plain apc.core.generation must agree exactly, since both are
    the same computation with no bank/router/workspace involved."""
    report = run_decoder_leakage_gate(_tiny_config())
    assert report.causal_mode_matches_plain_generation is True


def test_run_is_deterministic_given_the_same_seed() -> None:
    first = run_decoder_leakage_gate(_tiny_config())
    second = run_decoder_leakage_gate(_tiny_config())
    assert first.overall_exact_match == second.overall_exact_match
    assert first.per_operation_exact_match == second.per_operation_exact_match


def test_report_to_dict_round_trips_through_json() -> None:
    import json

    report = run_decoder_leakage_gate(_tiny_config())
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["overall_exact_match"] == report.overall_exact_match


# --- run_decoder_leakage_gate_multi_seed -------------------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    seeds = (0, 1)
    result = run_decoder_leakage_gate_multi_seed(_tiny_config(), seeds=seeds, run_dir=tmp_path)
    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    for seed in seeds:
        assert (tmp_path / f"seed_{seed}" / "report.json").exists()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    result = run_decoder_leakage_gate_multi_seed(_tiny_config(), seeds=(0, 1, 2))
    assert [report.config.seed for report in result.per_seed] == [0, 1, 2]


def test_multi_seed_forces_include_task_spec_false_even_if_base_config_did_not() -> None:
    config = _tiny_config(include_task_spec=True)
    result = run_decoder_leakage_gate_multi_seed(config, seeds=(0,))
    assert result.per_seed[0].config.include_task_spec is False


def test_multi_seed_passed_is_true_when_mean_is_at_or_below_ceiling() -> None:
    result = run_decoder_leakage_gate_multi_seed(
        _tiny_config(), seeds=(0, 1), material_ceiling=1.0
    )
    assert result.materially_below_future_correct_target is True
    assert result.passed is True


def test_multi_seed_passed_is_false_when_mean_exceeds_a_zero_ceiling() -> None:
    """A ceiling of exactly 0.0 is not achievable by a real greedy-decode
    exact-match rate over a nonzero eval set unless the model is perfectly
    wrong everywhere -- forcing a fail branch here exercises `passed=False`
    without depending on real training dynamics."""
    result = run_decoder_leakage_gate_multi_seed(
        _tiny_config(), seeds=(0,), material_ceiling=-1.0
    )
    assert result.materially_below_future_correct_target is False
    assert result.passed is False


def test_multi_seed_meets_seed_policy_reflects_seed_count() -> None:
    short = run_decoder_leakage_gate_multi_seed(_tiny_config(), seeds=(0, 1))
    assert short.meets_seed_policy is False
    full = run_decoder_leakage_gate_multi_seed(_tiny_config(), seeds=tuple(range(MIN_GATE_SEEDS)))
    assert full.meets_seed_policy is True


def test_multi_seed_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    import json

    result = run_decoder_leakage_gate_multi_seed(_tiny_config(), seeds=(0, 1), run_dir=tmp_path)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] == result.passed
    assert len(payload["per_seed"]) == 2


# --- module constants ---------------------------------------------------------


def test_default_seeds_matches_min_gate_seeds_length() -> None:
    assert len(DEFAULT_SEEDS) == MIN_GATE_SEEDS


def test_future_correct_target_matches_h2b_correct_threshold() -> None:
    assert FUTURE_CORRECT_TARGET == 0.95


def test_material_ceiling_matches_h2b_none_threshold() -> None:
    assert NO_PRIMITIVE_MATERIAL_CEILING == 0.30
