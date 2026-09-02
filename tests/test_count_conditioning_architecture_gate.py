"""Tests for the conditioning architecture comparison gate (Phase A.1
Post-Correction Task A1-R005D-006, STOP GATE).

Fast, small-step tests only -- mirroring `tests/
test_count_capacity_sweep_gate.py`'s convention. The gate's real scientific
claim (5 seeds, the full A1-R005D-004 training budget per variant) is run
via `scripts/count_conditioning_architecture_gate.py`, not asserted here.
The selection rule itself (`_select_variant`) is tested directly against
hand-built inputs, independent of real training dynamics.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from apc.evaluation.count_conditioning_architecture_gate import (
    CONDITIONING_VARIANTS,
    ConditioningArchitectureComparisonReport,
    ConditioningVariantReport,
    _select_variant,
    _VariantSelectionInput,
    run_conditioning_architecture_comparison,
)
from apc.evaluation.count_counterfactual_gate import (
    CountCounterfactualGateConfig,
    PrimitiveTrainConfig,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig
from apc.primitives.conditioning import ConditioningVariant


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


# --- _select_variant (pure selection-rule logic) -------------------------------


def _input(
    variant: str, *, correct: float, wrong: float, gap: float, passed: bool
) -> _VariantSelectionInput:
    return _VariantSelectionInput(
        variant=variant,
        mean_correct_exact_match=correct,
        mean_effectful_wrong_argument_exact_match=wrong,
        mean_causal_gap=gap,
        passed=passed,
    )


def test_select_variant_picks_highest_causal_gap_among_passing() -> None:
    inputs = [
        _input("additive", correct=0.5, wrong=0.3, gap=0.2, passed=True),
        _input("film", correct=0.6, wrong=0.1, gap=0.5, passed=True),
        _input("basis", correct=0.55, wrong=0.2, gap=0.35, passed=True),
    ]
    disqualified, best, selected = _select_variant(inputs, "additive")
    assert disqualified == ()
    assert best == "film"
    assert selected == "film"


def test_select_variant_returns_none_selected_when_nothing_passes() -> None:
    inputs = [
        _input("additive", correct=0.45, wrong=0.26, gap=0.11, passed=False),
        _input("film", correct=0.55, wrong=0.20, gap=0.20, passed=False),
    ]
    disqualified, best, selected = _select_variant(inputs, "additive")
    assert disqualified == ()
    assert best == "film"  # still reported even though nothing passed
    assert selected is None


def test_select_variant_disqualifies_a_candidate_that_raises_correct_and_wrong_together() -> None:
    inputs = [
        _input("additive", correct=0.45, wrong=0.26, gap=0.11, passed=False),
        # film raises BOTH Correct and effectful-Wrong-argument above baseline
        # -- disqualified even though its causal gap looks better.
        _input("film", correct=0.70, wrong=0.40, gap=0.30, passed=False),
        _input("basis", correct=0.50, wrong=0.20, gap=0.20, passed=False),
    ]
    disqualified, best, selected = _select_variant(inputs, "additive")
    assert disqualified == ("film",)
    assert best == "basis"
    assert selected is None


def test_select_variant_never_disqualifies_the_baseline_itself() -> None:
    inputs = [
        _input("additive", correct=0.9, wrong=0.9, gap=0.0, passed=False),
    ]
    disqualified, best, selected = _select_variant(inputs, "additive")
    assert disqualified == ()
    assert best == "additive"


def test_select_variant_selects_among_disqualification_survivors_only() -> None:
    inputs = [
        _input("additive", correct=0.45, wrong=0.26, gap=0.11, passed=True),
        _input("film", correct=0.70, wrong=0.40, gap=0.30, passed=True),  # disqualified
        _input("basis", correct=0.50, wrong=0.20, gap=0.30, passed=True),
    ]
    disqualified, best, selected = _select_variant(inputs, "additive")
    assert disqualified == ("film",)
    # basis ties film's raw gap number but film is disqualified, so basis wins.
    assert best == "basis"
    assert selected == "basis"


# --- run_conditioning_architecture_comparison (integration, tiny config) ------


def test_comparison_runs_every_variant_in_declared_order() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), **_NEVER_PASS
    )
    assert isinstance(result, ConditioningArchitectureComparisonReport)
    assert [v.variant for v in result.variants] == [v.value for v in CONDITIONING_VARIANTS]
    for variant_report in result.variants:
        assert isinstance(variant_report, ConditioningVariantReport)
        assert variant_report.primitive_param_count > 0
        assert len(variant_report.multi_seed.per_seed) == 1


def test_comparison_variants_have_distinct_parameter_counts() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), num_basis=3, **_NEVER_PASS
    )
    param_counts = {v.variant: v.primitive_param_count for v in result.variants}
    assert len(set(param_counts.values())) == len(param_counts)


def test_comparison_writes_per_variant_artifacts(tmp_path: Path) -> None:
    run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), run_dir=tmp_path, **_NEVER_PASS
    )
    for variant in CONDITIONING_VARIANTS:
        assert (tmp_path / variant.value / "report.json").exists()
        assert (tmp_path / variant.value / "seed_0" / "report.json").exists()


def test_comparison_baseline_variant_is_additive() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), **_NEVER_PASS
    )
    assert result.baseline_variant == ConditioningVariant.ADDITIVE.value


def test_comparison_report_to_dict_round_trips_through_json() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), **_ALWAYS_PASS
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["baseline_variant"] == "additive"
    assert len(payload["variants"]) == 3
    assert payload["any_variant_passed"] is True


def test_comparison_any_variant_passed_true_when_thresholds_are_trivial() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), **_ALWAYS_PASS
    )
    assert result.any_variant_passed is True


def test_comparison_any_variant_passed_false_when_thresholds_are_unreachable() -> None:
    result = run_conditioning_architecture_comparison(
        _tiny_config(), seeds=(0,), **_NEVER_PASS
    )
    assert result.any_variant_passed is False
    assert result.selected_variant is None
