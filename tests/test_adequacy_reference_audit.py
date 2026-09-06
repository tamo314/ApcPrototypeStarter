# ruff: noqa: E501
"""Focused invariant coverage for B-C005D2-005 (no GPU checkpoints required)."""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.adequacy_reference_audit import (
    AdequacyReferenceAuditConfig,
    BankSizeAudit,
    _bias_direction,
    _development_seed_for,
    _overall_classification,
    _query_seed,
    _reference_seed,
    _support_seed,
    _symmetric_rule_classification,
    classify_plastic_cell,
    verify_installed_rule_matches_spec,
)
from apc.evaluation.hard_negative_routing_benchmark import build_hard_negative_candidates
from apc.meta.adequacy_verifier import (
    AdequacyDecision,
    CandidateVerificationStep,
    CandidateVerificationTrace,
    wilson_score_interval,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def _config(**overrides: object) -> AdequacyReferenceAuditConfig:
    return AdequacyReferenceAuditConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_config_rejects_sealed_seed_outside_regate_partition() -> None:
    with pytest.raises(ValueError, match="regate_sealed_seeds partition"):
        _config(sealed_seed=99)


def test_config_rejects_regate_seed_overlap_with_original_sealed() -> None:
    with pytest.raises(ValueError, match="original B-C005 sealed partition"):
        _config(sealed_seed=0, regate_sealed_seeds=(0, 1, 2, 3, 4), development_seeds=(10, 11, 12, 13, 14))


def test_config_rejects_development_overlap_with_sealed_partitions() -> None:
    with pytest.raises(ValueError, match="development seeds"):
        _config(development_seeds=(20, 11, 12, 13, 14))


def test_config_rejects_mismatched_partition_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        _config(development_seeds=(10, 11, 12, 13))


def test_config_rejects_non_standard_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_sizes"):
        _config(bank_sizes=(17,))


def test_config_rejects_non_positive_reference_examples() -> None:
    with pytest.raises(ValueError, match="reference_examples"):
        _config(reference_examples=0)


def test_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="adequacy_exact_match_threshold"):
        _config(adequacy_exact_match_threshold=1.5)


# ---------------------------------------------------------------------------
# Seed derivation.
# ---------------------------------------------------------------------------


def test_development_seed_for_matches_index_of_sealed_seed() -> None:
    config = _config()
    assert _development_seed_for(config, 24) == 14
    assert _development_seed_for(config, 20) == 10


def test_support_query_reference_seeds_are_pairwise_distinct() -> None:
    for bank_size in (16, 32, 64, 128):
        for target_id in range(4):
            support = _support_seed(24, bank_size, target_id)
            query = _query_seed(24, bank_size, target_id)
            reference = _reference_seed(24, bank_size, target_id)
            assert len({support, query, reference}) == 3
    assert _query_seed(24, 16, 3) == _support_seed(24, 16, 3) + 1


# ---------------------------------------------------------------------------
# D2-005.4 -- installed rule matches its documented spec.
# ---------------------------------------------------------------------------


def test_installed_rule_matches_spec_for_synthetic_cases() -> None:
    result = verify_installed_rule_matches_spec(_config())
    assert result["implementation_matches_spec"] is True
    assert result["mismatches"] == []
    assert result["cases_checked"] > 0


def _single_step_trace(successes: int, trials: int, decision: AdequacyDecision) -> CandidateVerificationTrace:
    """A one-step trace representing the terminal (successes, trials) state a
    real `SequentialAdequacyVerifier` run would have stopped at -- sufficient
    for testing `_symmetric_rule_classification`, which only reads
    `trace.steps` sequentially and does not depend on earlier history."""
    ci = wilson_score_interval(successes, trials, confidence_level=0.95)
    step = CandidateVerificationStep(
        step_index=0,
        support_size=trials,
        correct_count=successes,
        empirical_accuracy=ci.point_estimate,
        confidence_interval=ci,
        decision=decision,
    )
    trace = CandidateVerificationTrace(candidate_id="primitive:0", execute_primitive_id=0, steps=[step])
    trace.final_decision = decision
    trace.final_support_consumed = trials
    trace.final_accuracy = ci.point_estimate
    return trace


def test_symmetric_rule_agrees_with_installed_rule_on_clear_reject() -> None:
    # 2/32 is clearly inadequate under both the asymmetric and symmetric rules.
    trace = _single_step_trace(2, 32, AdequacyDecision.REJECT)
    symmetric = _symmetric_rule_classification(trace, 0.95)
    assert symmetric["decision"] == AdequacyDecision.REJECT.value


def test_symmetric_rule_can_diverge_on_marginal_forced_accept() -> None:
    # 122/128 = 0.953 >= 0.95: the installed forced-decision rule (final step,
    # empirical EM >= threshold) accepts, but the Wilson lower bound at
    # n=128 is below 0.95, so the symmetric rule reports UNCERTAIN instead of
    # ACCEPT -- exactly the asymmetry D2-005.4 asks to test for.
    ci_final = wilson_score_interval(122, 128, confidence_level=0.95)
    assert ci_final.point_estimate >= 0.95
    assert ci_final.lower < 0.95
    trace = _single_step_trace(122, 128, AdequacyDecision.ACCEPT)
    symmetric = _symmetric_rule_classification(trace, 0.95)
    assert symmetric["decision"] == "UNCERTAIN"


def test_bias_direction_premature_accept_when_installed_accepts_before_lower_bound_confirms() -> None:
    # 61/64 = 0.953125 >= 0.95 (installed ACCEPT), but Wilson lower bound is
    # still well below 0.95 and n=64 < max_support=128 -- an early accept the
    # confidence interval has not actually confirmed.
    ci = wilson_score_interval(61, 64, confidence_level=0.95)
    assert ci.point_estimate >= 0.95
    assert ci.lower < 0.95
    trace = _single_step_trace(61, 64, AdequacyDecision.ACCEPT)
    symmetric = _symmetric_rule_classification(trace, 0.95)
    assert symmetric["decision"] == "UNCERTAIN"
    assert _bias_direction(trace, symmetric, max_support=128) == "PREMATURE_ACCEPT"


def test_bias_direction_forced_reject_at_budget_exhaustion_when_installed_rejects_at_max_support() -> None:
    # 119/128 = 0.9297 < 0.95 at n=128=max_support: the installed rule's
    # forced-decision fallback rejects, but the symmetric rule's undecided
    # zone (upper bound still >= 0.95) has no forced fallback, so it stays
    # UNCERTAIN -- a deliberate conservative tie-break, not rule bias.
    trace = _single_step_trace(119, 128, AdequacyDecision.REJECT)
    symmetric = _symmetric_rule_classification(trace, 0.95)
    assert symmetric["decision"] == "UNCERTAIN"
    assert _bias_direction(trace, symmetric, max_support=128) == "FORCED_REJECT_AT_BUDGET_EXHAUSTION"


def test_bias_direction_none_when_installed_and_symmetric_agree() -> None:
    trace = _single_step_trace(2, 32, AdequacyDecision.REJECT)
    symmetric = _symmetric_rule_classification(trace, 0.95)
    assert _bias_direction(trace, symmetric, max_support=128) == "NONE"


# ---------------------------------------------------------------------------
# D2-005.3 -- reclassification.
# ---------------------------------------------------------------------------


def test_classify_plastic_cell_true_false_plastic() -> None:
    assert (
        classify_plastic_cell(runtime_decision=AdequacyDecision.REJECT.value, reference_adequate=True)
        == "TRUE_FALSE_PLASTIC"
    )


def test_classify_plastic_cell_functionally_justified_plastic() -> None:
    assert (
        classify_plastic_cell(runtime_decision=AdequacyDecision.REJECT.value, reference_adequate=False)
        == "FUNCTIONALLY_JUSTIFIED_PLASTIC"
    )


def test_classify_plastic_cell_correctly_accepted() -> None:
    assert (
        classify_plastic_cell(runtime_decision=AdequacyDecision.ACCEPT.value, reference_adequate=True)
        == "CORRECTLY_ACCEPTED"
    )


def test_classify_plastic_cell_unsafe_reuse() -> None:
    assert (
        classify_plastic_cell(runtime_decision=AdequacyDecision.ACCEPT.value, reference_adequate=False)
        == "UNSAFE_REUSE"
    )


# ---------------------------------------------------------------------------
# Overall classification aggregation.
# ---------------------------------------------------------------------------


def _audit(
    *, classification: str, sequential_rule_bias_direction: str = "NONE", bank_size: int = 16
) -> BankSizeAudit:
    return BankSizeAudit(
        bank_size=bank_size,
        model_seed=4,
        development_seed=14,
        target_id=0,
        official_cells={},
        rank_diagnostics={},
        verifier_trace={},
        symmetric_rule={},
        sequential_rule_bias_direction=sequential_rule_bias_direction,
        reference_adequacy={},
        independent_query_em=0.0,
        plastic_classification=classification,
    )


def test_overall_classification_true_false_plastic_implies_finite_support_variance_and_metric_misclassification() -> None:
    audits = [_audit(classification="TRUE_FALSE_PLASTIC"), _audit(classification="CORRECTLY_ACCEPTED", bank_size=64)]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert "FINITE_SUPPORT_VARIANCE" in result["labels"]
    assert "METRIC_MISCLASSIFICATION" in result["labels"]
    assert "TRUE_PRIMITIVE_INADEQUACY" not in result["labels"]


def test_overall_classification_functionally_justified_plastic_implies_true_primitive_inadequacy() -> None:
    audits = [_audit(classification="FUNCTIONALLY_JUSTIFIED_PLASTIC")]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert result["labels"] == ["TRUE_PRIMITIVE_INADEQUACY"]


def test_overall_classification_unsafe_reuse_flagged_separately() -> None:
    audits = [_audit(classification="UNSAFE_REUSE")]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert "UNSAFE_REUSE_DETECTED" in result["labels"]
    assert result["n_unsafe_reuse"] == 1


def test_overall_classification_premature_accept_on_unsafe_reuse_triggers_bias_label() -> None:
    # This is the safety-relevant real-world case: a genuinely inadequate
    # candidate accepted early, which is not a "plastic" classification at
    # all, so the aggregate check must not be scoped to plastic cells only.
    audits = [_audit(classification="UNSAFE_REUSE", sequential_rule_bias_direction="PREMATURE_ACCEPT", bank_size=64)]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert "SEQUENTIAL_RULE_BIAS" in result["labels"]


def test_overall_classification_forced_reject_at_budget_exhaustion_is_not_bias() -> None:
    # A benign, deliberate conservative tie-break must not be mislabeled as
    # rule bias, even though it is a divergence from the symmetric rule.
    audits = [
        _audit(
            classification="FUNCTIONALLY_JUSTIFIED_PLASTIC",
            sequential_rule_bias_direction="FORCED_REJECT_AT_BUDGET_EXHAUSTION",
        )
    ]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert "SEQUENTIAL_RULE_BIAS" not in result["labels"]


def test_overall_classification_unresolved_when_all_cells_correctly_accepted() -> None:
    audits = [_audit(classification="CORRECTLY_ACCEPTED")]
    spec_check = {"implementation_matches_spec": True}
    result = _overall_classification(audits, spec_check)
    assert result["labels"] == ["UNRESOLVED"]


def test_overall_classification_implementation_bug_when_spec_check_fails() -> None:
    audits = [_audit(classification="CORRECTLY_ACCEPTED")]
    spec_check = {"implementation_matches_spec": False}
    result = _overall_classification(audits, spec_check)
    assert "IMPLEMENTATION_BUG" in result["labels"]


# ---------------------------------------------------------------------------
# Correct-candidate identity is level-invariant (the structural fact this
# module's shared verifier trace depends on).
# ---------------------------------------------------------------------------


def test_correct_candidate_identity_is_level_invariant() -> None:
    target_id, related_id = 0, 1
    keys_by_id = {
        target_id: torch.randn(8),
        related_id: torch.randn(8),
        2: torch.randn(8),
        3: torch.randn(8),
    }
    operation_by_id = {target_id: "SHIFT", related_id: "CYCLE_FOUR", 2: "COUNT", 3: "BIND"}
    identities = {}
    for level in HardNegativeLevel:
        candidates, competitor = build_hard_negative_candidates(
            level=level,
            target_id=target_id,
            target_operation="SHIFT",
            candidate_ids=list(keys_by_id),
            keys_by_id=keys_by_id,
            operation_by_id=operation_by_id,
            seed=24,
        )
        correct = next(
            c for c in candidates if c is not competitor and c.execute_primitive_id == target_id
        )
        identities[level.value] = (correct.execute_primitive_id, correct.argument_override)
    assert len(set(identities.values())) == 1
