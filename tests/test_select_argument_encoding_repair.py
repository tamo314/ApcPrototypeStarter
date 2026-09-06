# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-007's pure logic.

Matches this repo's existing precedent (D2/R3-002/R3-004/R3-006): anything
that needs a real Core/bank/router/checkpoint (`run_select_argument_encoding_repair`
itself) is exercised only via the milestone script
(`scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-007`), not in
pytest. The schema/round-trip audits and the softmax-dilution
characterization need only the real (lightweight, CPU-only) example
generator and `PrimitiveCall`/`Operation` code -- no Core, bank, or GPU --
so they get real (not mocked) coverage here.
"""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.select_argument_encoding_repair import (
    ADR_0081_REFERENCE,
    SelectArgumentEncodingConfig,
    _legacy_softmax_combine_score,
    _LegacyPreFixArgumentScorer,
    _mean,
    _sigmoid_combine_score,
    audit_argument_scorer_train_inference_contract,
    audit_round_trip_and_canonicalization,
    audit_select_argument_schema,
    characterize_softmax_dilution,
)
from apc.primitives.argument_scoring import ArgumentScorer, ArgumentScorerConfig

# ---------------------------------------------------------------------------
# SelectArgumentEncodingConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = SelectArgumentEncodingConfig()
    assert config.bank_size == 128
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.gate_full_argument_accuracy_threshold == 0.95
    assert config.gate_full_call_top1_threshold == 0.90


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        SelectArgumentEncodingConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        SelectArgumentEncodingConfig(bank_size=100)


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="support_examples"):
        SelectArgumentEncodingConfig(support_examples=0)


def test_config_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SelectArgumentEncodingConfig(top_k=6)


def test_config_rejects_bad_gate_thresholds() -> None:
    with pytest.raises(ValueError, match="gate_full_argument_accuracy_threshold"):
        SelectArgumentEncodingConfig(gate_full_argument_accuracy_threshold=1.5)
    with pytest.raises(ValueError, match="gate_full_call_top1_threshold"):
        SelectArgumentEncodingConfig(gate_full_call_top1_threshold=0.0)


def test_config_rejects_negative_regression_budget() -> None:
    with pytest.raises(ValueError, match="gate_other_op_regression_pp_max"):
        SelectArgumentEncodingConfig(gate_other_op_regression_pp_max=-1.0)


def test_config_rejects_empty_cardinality_stress_lengths() -> None:
    with pytest.raises(ValueError, match="cardinality_stress_lengths"):
        SelectArgumentEncodingConfig(cardinality_stress_lengths=())


def test_config_rejects_bad_rare_value_percentile() -> None:
    with pytest.raises(ValueError, match="rare_value_percentile"):
        SelectArgumentEncodingConfig(rare_value_percentile=1.0)


def test_config_rejects_empty_dilution_cardinalities() -> None:
    with pytest.raises(ValueError, match="dilution_cardinalities"):
        SelectArgumentEncodingConfig(dilution_cardinalities=())


# ---------------------------------------------------------------------------
# Encoding-contract audits (real generator + real Operation/PrimitiveCall
# code; no Core/bank/GPU needed).
# ---------------------------------------------------------------------------


def test_audit_select_argument_schema_matches_known_generator_invariants() -> None:
    result = audit_select_argument_schema(seed=10)
    assert result["generator_always_emits_ascending_order"] is True
    assert result["all_indices_distinct"] is True
    assert result["all_indices_in_bounds"] is True
    assert result["all_sizes_match_output_length"] is True
    assert result["n_examples"] > 0


def test_audit_round_trip_and_canonicalization_is_exact() -> None:
    result = audit_round_trip_and_canonicalization(seed=10)
    assert result["round_trip_exact_order_equality"] is True
    assert result["wrong_argument_competitor_preserves_ascending_invariant"] is True


def test_audit_schema_is_deterministic_across_repeated_calls() -> None:
    first = audit_select_argument_schema(seed=11)
    second = audit_select_argument_schema(seed=11)
    assert first["observed_cardinalities"] == second["observed_cardinalities"]


# ---------------------------------------------------------------------------
# Softmax-dilution characterization (pure tensor math, no dependencies).
# ---------------------------------------------------------------------------


def test_legacy_and_sigmoid_combine_scores_both_confident_at_cardinality_one() -> None:
    """At cardinality 1 there is no dilution partner: both formulas read the
    single confident logit as a high compatibility score (they need not be
    numerically identical -- softmax's exact value also depends on how many
    confidently-wrong competitors share the denominator)."""
    logits = torch.tensor([[6.0, -6.0, -6.0, -6.0]])
    legacy = _legacy_softmax_combine_score(logits, [0])
    repaired = _sigmoid_combine_score(logits, [0])
    assert legacy > 0.99
    assert repaired > 0.99


def test_legacy_softmax_collapses_once_second_index_becomes_correct() -> None:
    """The dramatic, load-bearing finding: adding a second simultaneously-
    confident correct index collapses the legacy softmax score from >0.99
    to near-zero (each of the two must now share the softmax's one unit of
    probability mass), while the repaired sigmoid score is unaffected."""
    result = characterize_softmax_dilution(arg_vocab_size=32, cardinalities=(1, 2))
    by_cardinality = {row["cardinality"]: row for row in result["rows"]}
    assert by_cardinality[1]["legacy_softmax_score"] > 0.99
    assert by_cardinality[2]["legacy_softmax_score"] < 0.1
    assert by_cardinality[1]["repaired_sigmoid_score"] > 0.99
    assert by_cardinality[2]["repaired_sigmoid_score"] > 0.99


def _legacy_score_at(result: dict, cardinality: int) -> float:
    return next(row["legacy_softmax_score"] for row in result["rows"] if row["cardinality"] == cardinality)


def test_legacy_softmax_dilutes_more_than_sigmoid_as_cardinality_grows() -> None:
    result = characterize_softmax_dilution(arg_vocab_size=32, cardinalities=(1, 2, 4, 8, 16))
    assert result["dilution_confirmed_for_cardinality_gt_1"] is True
    assert result["legacy_score_monotonically_decreasing_with_cardinality"] is True
    # The repaired formula stays confidently near +1.0 regardless of cardinality.
    for row in result["rows"]:
        assert row["repaired_sigmoid_score"] > 0.99
    # The legacy formula becomes strongly NEGATIVE (worse than uninformative)
    # at higher cardinalities, despite the network being fully confident.
    assert _legacy_score_at(result, 16) < -0.8


def test_dilution_characterization_skips_cardinalities_above_vocab_size() -> None:
    result = characterize_softmax_dilution(arg_vocab_size=4, cardinalities=(1, 2, 8))
    measured = [row["cardinality"] for row in result["rows"]]
    assert measured == [1, 2]


# ---------------------------------------------------------------------------
# Train/inference contract audit (confirms the fix is actually in place).
# ---------------------------------------------------------------------------


def test_contract_audit_confirms_fix_is_in_place() -> None:
    result = audit_argument_scorer_train_inference_contract()
    assert result["select_trained_with_independent_multihot_bce"] is True
    assert result["forward_contains_select_specific_sigmoid_fix"] is True
    assert result["non_select_operations_still_softmax"] is True
    assert result["conclusion"] == "DEFECT_FOUND_PROCEED_WITH_MINIMAL_FIX"
    assert result["adr_0081_reference"] == ADR_0081_REFERENCE


# ---------------------------------------------------------------------------
# _LegacyPreFixArgumentScorer: read-only comparison adapter.
# ---------------------------------------------------------------------------


def test_legacy_adapter_shares_trained_weights_and_reproduces_old_softmax_formula() -> None:
    config = ArgumentScorerConfig(d_model=8, arg_vocab_size=16)
    trained = ArgumentScorer(config)
    adapter = _LegacyPreFixArgumentScorer(trained)

    # Same underlying weights (shared, not copied).
    assert adapter.heads["SELECT"] is trained.heads["SELECT"]

    z_task = torch.randn(2, 8)
    with torch.no_grad():
        logits = trained.heads["SELECT"](z_task)
        expected_probs = torch.softmax(logits, dim=-1)
        val_idx = torch.tensor([[3]]).expand(2, 1)
        expected_p = expected_probs.gather(dim=-1, index=val_idx).squeeze(-1)
        expected_score = 2.0 * (expected_p - 0.5)

    legacy_score = adapter(z_task, "SELECT", {"indices": [3]})
    assert torch.allclose(legacy_score, expected_score, atol=1e-6)

    # The adapter's SELECT score must differ from the real (repaired) scorer's
    # SELECT score on a multi-index argument -- otherwise this comparison
    # would be vacuous.
    repaired_score = trained(z_task, "SELECT", {"indices": [1, 3, 5]})
    legacy_multi_score = adapter(z_task, "SELECT", {"indices": [1, 3, 5]})
    assert not torch.allclose(repaired_score, legacy_multi_score)


def test_legacy_adapter_matches_repaired_scorer_on_non_select_ops() -> None:
    """For SHIFT/COUNT/BIND both formulas are softmax -- the adapter must
    reproduce the real scorer's output exactly (freeze proof by construction)."""
    config = ArgumentScorerConfig(d_model=8, arg_vocab_size=16)
    trained = ArgumentScorer(config)
    adapter = _LegacyPreFixArgumentScorer(trained)
    z_task = torch.randn(2, 8)

    for op, args in (("SHIFT", {"amount": 2}), ("COUNT", {"target": 5}), ("BIND", {"query_key": 7})):
        real = trained(z_task, op, args)
        legacy = adapter(z_task, op, args)
        assert torch.allclose(real, legacy, atol=1e-6)


# ---------------------------------------------------------------------------
# _mean helper.
# ---------------------------------------------------------------------------


def test_mean_returns_none_for_empty_list() -> None:
    assert _mean([]) is None


def test_mean_computes_average() -> None:
    assert _mean([1.0, 2.0, 3.0]) == 2.0
