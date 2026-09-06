# ruff: noqa: E501
"""Focused invariant coverage for B-C005R3-003 (offline, CPU-only, no GPU/model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.evaluation.functional_metrics_v2 import (
    CandidateEvaluationRecord,
    CandidateVerdict,
    ControllerAction,
    EpisodeOutcomeRecord,
    EpisodeQueryRecord,
    ExecutionStatus,
    FiniteLookVerifierContract,
    FunctionalMetricsV2Config,
    LibraryScopeStatus,
    ReferenceAdequacyState,
    ReferenceAllocation,
    SearchStatus,
    SuiteKind,
    VerifiedDecisionEnvelope,
    accepted_ref_unresolved_count,
    adequate_solution_nonreuse_rate,
    aggregate_candidate_metrics,
    aggregate_episode_metrics,
    aggregate_query_metrics,
    avoidable_plastic_rate,
    build_budget_feasibility,
    build_metric_fixture_results,
    build_metrics_schema_v2,
    build_statistical_contract,
    classify_library_scope,
    compute_rate,
    execution_coverage,
    finite_look_decision,
    full_success_lower_bound,
    inadequate_call_accept_rate,
    legacy_known_task_plastic_rate,
    min_successes_for_accept,
    one_sided_exact_bounds,
    partition_by_suite,
    reference_adequacy_state,
    reference_unresolved_rate,
    run_functional_metrics_v2_protocol,
    same_identity_unsafe_accept_rate,
    selective_query_em,
    uncertain_candidate_rate,
    unconditional_query_em,
    unsafe_reuse_episode_rate,
    verify_candidate_finite_look,
    wrong_call_accept_rate,
)

# ---------------------------------------------------------------------------
# VerifiedDecisionEnvelope: valid construction + invariant enforcement.
# ---------------------------------------------------------------------------


def test_envelope_valid_executed_direct_reuse() -> None:
    envelope = VerifiedDecisionEnvelope(
        candidate_verdict=CandidateVerdict.ACCEPT,
        search_status=SearchStatus.FOUND_SOLUTION,
        controller_action=ControllerAction.DIRECT_REUSE,
        execution_status=ExecutionStatus.EXECUTED,
    )
    assert envelope.to_dict()["execution_status"] == "EXECUTED"


def test_envelope_executed_without_accept_raises() -> None:
    with pytest.raises(ValueError, match="EXECUTED requires an ACCEPT"):
        VerifiedDecisionEnvelope(
            candidate_verdict=CandidateVerdict.REJECT,
            search_status=SearchStatus.FOUND_SOLUTION,
            controller_action=ControllerAction.DIRECT_REUSE,
            execution_status=ExecutionStatus.EXECUTED,
        )


def test_envelope_executed_with_plastic_search_raises() -> None:
    with pytest.raises(ValueError, match="DIRECT_REUSE or COMPOSE"):
        VerifiedDecisionEnvelope(
            candidate_verdict=CandidateVerdict.ACCEPT,
            search_status=SearchStatus.FOUND_SOLUTION,
            controller_action=ControllerAction.PLASTIC_SEARCH,
            execution_status=ExecutionStatus.EXECUTED,
        )


def test_envelope_needs_more_evidence_with_controller_action_raises() -> None:
    """No temporary workspace may be committed while a candidate is UNCERTAIN."""
    with pytest.raises(ValueError, match="no temporary workspace"):
        VerifiedDecisionEnvelope(
            candidate_verdict=CandidateVerdict.UNCERTAIN,
            search_status=SearchStatus.BUDGET_EXHAUSTED,
            controller_action=ControllerAction.PLASTIC_SEARCH,
            execution_status=ExecutionStatus.NEEDS_MORE_EVIDENCE,
        )


def test_envelope_needs_more_evidence_valid_defers_cleanly() -> None:
    envelope = VerifiedDecisionEnvelope(
        candidate_verdict=CandidateVerdict.UNCERTAIN,
        search_status=SearchStatus.BUDGET_EXHAUSTED,
        controller_action=None,
        execution_status=ExecutionStatus.NEEDS_MORE_EVIDENCE,
    )
    assert envelope.to_dict()["controller_action"] is None


def test_envelope_no_verified_solution_with_accept_raises() -> None:
    with pytest.raises(ValueError, match="NO_VERIFIED_SOLUTION cannot coexist"):
        VerifiedDecisionEnvelope(
            candidate_verdict=CandidateVerdict.ACCEPT,
            search_status=SearchStatus.COMPLETE_WITHIN_SCOPE,
            controller_action=None,
            execution_status=ExecutionStatus.NO_VERIFIED_SOLUTION,
        )


def test_envelope_no_verified_solution_valid() -> None:
    envelope = VerifiedDecisionEnvelope(
        candidate_verdict=CandidateVerdict.REJECT,
        search_status=SearchStatus.COMPLETE_WITHIN_SCOPE,
        controller_action=None,
        execution_status=ExecutionStatus.NO_VERIFIED_SOLUTION,
    )
    assert envelope.candidate_verdict == CandidateVerdict.REJECT


# ---------------------------------------------------------------------------
# Library scope classification: never overclaim on an unevaluated recipe.
# ---------------------------------------------------------------------------


def test_library_scope_any_accept_dominates_even_mid_search() -> None:
    result = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.ACCEPT), all_of_h_evaluated=False
    )
    assert result == LibraryScopeStatus.FOUND_ADEQUATE_SOLUTION


def test_library_scope_unevaluated_recipe_blocks_inadequate_claim() -> None:
    result = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.REJECT), all_of_h_evaluated=False
    )
    assert result == LibraryScopeStatus.UNRESOLVED_WITHIN_SCOPE


def test_library_scope_fully_evaluated_all_reject_is_inadequate_within_scope() -> None:
    result = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.REJECT), all_of_h_evaluated=True
    )
    assert result == LibraryScopeStatus.INADEQUATE_WITHIN_DECLARED_SCOPE


def test_library_scope_uncertain_present_stays_unresolved_even_if_fully_evaluated() -> None:
    result = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.UNCERTAIN), all_of_h_evaluated=True
    )
    assert result == LibraryScopeStatus.UNRESOLVED_WITHIN_SCOPE


# ---------------------------------------------------------------------------
# one_sided_exact_bounds: input validation + monotonicity.
# ---------------------------------------------------------------------------


def test_one_sided_exact_bounds_k_zero_lower_is_zero() -> None:
    lower, upper = one_sided_exact_bounds(0, 32, alpha_lower=0.0004, alpha_upper=0.0004)
    assert lower == 0.0
    assert 0.0 < upper < 1.0


def test_one_sided_exact_bounds_k_equals_n_upper_is_one() -> None:
    lower, upper = one_sided_exact_bounds(32, 32, alpha_lower=0.0004, alpha_upper=0.0004)
    assert upper == 1.0
    assert 0.0 < lower < 1.0


def test_one_sided_exact_bounds_lower_monotonically_nondecreasing_in_k() -> None:
    n = 64
    bounds = [one_sided_exact_bounds(k, n, 0.0004, 0.0004)[0] for k in range(0, n + 1)]
    assert all(b1 <= b2 + 1e-15 for b1, b2 in zip(bounds, bounds[1:], strict=False))


def test_one_sided_exact_bounds_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        one_sided_exact_bounds(5, 0, 0.01, 0.01)
    with pytest.raises(ValueError):
        one_sided_exact_bounds(-1, 10, 0.01, 0.01)
    with pytest.raises(ValueError):
        one_sided_exact_bounds(11, 10, 0.01, 0.01)
    with pytest.raises(ValueError):
        one_sided_exact_bounds(5, 10, 0.0, 0.01)


# ---------------------------------------------------------------------------
# Finite-look contract: decision rule, sequential trace never forces a result.
# ---------------------------------------------------------------------------


def test_finite_look_verifier_contract_alpha_allocation() -> None:
    contract = FiniteLookVerifierContract()
    assert contract.alpha_accept_per_candidate_look == pytest.approx(0.0004)
    assert contract.alpha_reject_per_candidate_look == pytest.approx(0.0004)


def test_finite_look_verifier_contract_rejects_bad_looks() -> None:
    with pytest.raises(ValueError):
        FiniteLookVerifierContract(looks=(64, 32))
    with pytest.raises(ValueError):
        FiniteLookVerifierContract(looks=(32, 32, 64))
    with pytest.raises(ValueError):
        FiniteLookVerifierContract(looks=())


def test_finite_look_decision_all_correct_at_512_accepts() -> None:
    contract = FiniteLookVerifierContract()
    verdict, lower, _upper = finite_look_decision(512, 512, contract)
    assert verdict == CandidateVerdict.ACCEPT
    assert lower >= contract.tau


def test_finite_look_decision_low_accuracy_rejects_early() -> None:
    contract = FiniteLookVerifierContract()
    verdict, _lower, upper = finite_look_decision(10, 32, contract)
    assert verdict == CandidateVerdict.REJECT
    assert upper < contract.tau


def test_verify_candidate_finite_look_early_accept_stops_at_256() -> None:
    contract = FiniteLookVerifierContract()

    def eval_fn(start: int, end: int) -> int:
        return end - start  # always correct

    trace = verify_candidate_finite_look(candidate_id="c1", support_eval_fn=eval_fn, contract=contract)
    assert trace.final_verdict == CandidateVerdict.ACCEPT
    assert trace.final_support_consumed == 256
    assert len(trace.steps) == 4  # looks 32, 64, 128, 256


def test_verify_candidate_finite_look_stays_uncertain_at_max_look_never_forced() -> None:
    """A candidate whose cumulative successes stay inside the UNCERTAIN band at
    every look (verified independently against the exact Beta-quantile bounds:
    look 512 gives lower=0.9185 < tau=0.95 <= upper=0.9811) must remain
    UNCERTAIN through the final look, not be forced to ACCEPT/REJECT by
    empirical point estimate (design doc S4.3)."""
    contract = FiniteLookVerifierContract()
    # cumulative successes at each look boundary: 30/32, 60/64, 120/128, 245/256, 490/512
    cumulative_successes = {32: 30, 64: 60, 128: 120, 256: 245, 512: 490}

    def eval_fn(start: int, end: int) -> int:
        previous_total = cumulative_successes[start] if start else 0
        return cumulative_successes[end] - previous_total

    trace = verify_candidate_finite_look(candidate_id="c2", support_eval_fn=eval_fn, contract=contract)
    assert trace.final_verdict == CandidateVerdict.UNCERTAIN
    assert trace.final_support_consumed == 512
    assert len(trace.steps) == 5
    assert all(step.verdict == CandidateVerdict.UNCERTAIN for step in trace.steps)


# ---------------------------------------------------------------------------
# Budget feasibility table: cross-check against design doc S4.5's numbers.
# ---------------------------------------------------------------------------


def test_full_success_lower_bound_matches_closed_form_power_law() -> None:
    contract = FiniteLookVerifierContract()
    alpha = contract.alpha_accept_per_candidate_look
    for look in contract.looks:
        closed_form = alpha ** (1.0 / look)
        assert full_success_lower_bound(look, contract) == pytest.approx(closed_form, rel=1e-9)


def test_budget_feasibility_table_matches_design_doc_s4_5() -> None:
    contract = FiniteLookVerifierContract()
    expected_full_success_lower = {
        32: 0.783095,
        64: 0.884926,
        128: 0.940705,
        256: 0.969900,
        512: 0.984835,
    }
    expected_min_successes = {32: None, 64: None, 128: None, 256: 254, 512: 502}
    table = build_budget_feasibility(contract)
    by_look = {row["look"]: row for row in table["rows"]}
    for look, expected in expected_full_success_lower.items():
        assert by_look[look]["full_success_lower_bound"] == pytest.approx(expected, abs=1e-5)
    for look, expected_k in expected_min_successes.items():
        assert by_look[look]["min_successes_for_accept"] == expected_k
        assert by_look[look]["accept_possible_at_this_look"] == (expected_k is not None)


def test_min_successes_for_accept_none_when_even_full_success_insufficient() -> None:
    contract = FiniteLookVerifierContract()
    assert min_successes_for_accept(32, contract) is None


# ---------------------------------------------------------------------------
# Reference adequacy: empty denominator -> null; mid-region -> UNRESOLVED.
# ---------------------------------------------------------------------------


def test_reference_adequacy_state_empty_denominator_is_null() -> None:
    allocation = ReferenceAllocation()
    assert reference_adequacy_state(0, 0, allocation) is None


def test_reference_adequacy_state_high_success_is_adequate() -> None:
    allocation = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=1)
    assert reference_adequacy_state(4096, 4096, allocation) == ReferenceAdequacyState.REF_ADEQUATE


def test_reference_adequacy_state_low_success_is_inadequate() -> None:
    allocation = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=1)
    assert reference_adequacy_state(10, 128, allocation) == ReferenceAdequacyState.REF_INADEQUATE


def test_reference_adequacy_state_mid_region_stays_unresolved_not_forced_binary() -> None:
    allocation = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=1)
    state = reference_adequacy_state(125, 128, allocation)
    assert state == ReferenceAdequacyState.REF_UNRESOLVED


def test_reference_adequacy_state_allocation_narrows_as_m_calls_grows() -> None:
    """More simultaneously-evaluated calls in an episode means a smaller
    per-call alpha budget, which can only widen (never narrow) the bounds for
    the same (successes, trials)."""
    small_m = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=1)
    large_m = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=10)
    lower_small, upper_small = one_sided_exact_bounds(
        120, 128, small_m.alpha_lower_per_call, small_m.alpha_upper_per_call
    )
    lower_large, upper_large = one_sided_exact_bounds(
        120, 128, large_m.alpha_lower_per_call, large_m.alpha_upper_per_call
    )
    assert lower_large <= lower_small
    assert upper_large >= upper_small


# ---------------------------------------------------------------------------
# compute_rate: null on empty denominator, never a manufactured 0%/100%.
# ---------------------------------------------------------------------------


def test_compute_rate_empty_denominator_is_null() -> None:
    result = compute_rate("x", 0, 0)
    assert result.value is None


def test_compute_rate_rejects_invalid_numerator() -> None:
    with pytest.raises(ValueError):
        compute_rate("x", 5, 3)
    with pytest.raises(ValueError):
        compute_rate("x", -1, 3)


# ---------------------------------------------------------------------------
# Candidate-level metrics.
# ---------------------------------------------------------------------------


def _candidate(
    episode_id: str,
    candidate_id: str,
    *,
    operation: str = "SHIFT",
    relation_id: str | None = None,
    model_seed: int = 0,
    identity_match: bool,
    functionally_equivalent: bool = False,
    reference_state: ReferenceAdequacyState | None,
    verdict: CandidateVerdict,
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        episode_id=episode_id,
        candidate_id=candidate_id,
        operation=operation,
        relation_id=relation_id,
        model_seed=model_seed,
        identity_match=identity_match,
        functionally_equivalent=functionally_equivalent,
        reference_state=reference_state,
        verdict=verdict,
    )


def test_wrong_call_accept_rate_excludes_functionally_equivalent_alias() -> None:
    records = (
        _candidate(
            "e1", "wrong", identity_match=False, functionally_equivalent=False,
            reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT,
        ),
        _candidate(
            "e1", "alias", identity_match=False, functionally_equivalent=True,
            reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT,
        ),
    )
    result = wrong_call_accept_rate(records)
    assert result.denominator == 1
    assert result.numerator == 0


def test_inadequate_call_accept_rate() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.ACCEPT),
        _candidate("e1", "c2", identity_match=True, reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT),
        _candidate("e1", "c3", identity_match=True, reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT),
    )
    result = inadequate_call_accept_rate(records)
    assert result.numerator == 1
    assert result.denominator == 2


def test_same_identity_unsafe_accept_rate_correct_id_insufficient_reference() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.ACCEPT),
    )
    result = same_identity_unsafe_accept_rate(records)
    assert result.value == 1.0


def test_uncertain_candidate_rate() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=None, verdict=CandidateVerdict.UNCERTAIN),
        _candidate("e1", "c2", identity_match=True, reference_state=None, verdict=CandidateVerdict.ACCEPT),
    )
    result = uncertain_candidate_rate(records)
    assert result.numerator == 1
    assert result.denominator == 2


def test_reference_unresolved_rate_excludes_not_estimable_from_denominator() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=ReferenceAdequacyState.REF_UNRESOLVED, verdict=CandidateVerdict.UNCERTAIN),
        _candidate("e1", "c2", identity_match=True, reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT),
        _candidate("e1", "c3", identity_match=True, reference_state=None, verdict=CandidateVerdict.UNCERTAIN),
    )
    result = reference_unresolved_rate(records)
    assert result.denominator == 2  # the NOT_ESTIMABLE (None) record is excluded
    assert result.numerator == 1


def test_accepted_ref_unresolved_count_is_a_raw_count() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=ReferenceAdequacyState.REF_UNRESOLVED, verdict=CandidateVerdict.ACCEPT),
        _candidate("e1", "c2", identity_match=True, reference_state=ReferenceAdequacyState.REF_UNRESOLVED, verdict=CandidateVerdict.REJECT),
    )
    assert accepted_ref_unresolved_count(records) == 1


def test_aggregate_candidate_metrics_breakdowns() -> None:
    records = (
        _candidate("e1", "c1", operation="SHIFT", relation_id="SHIFT-CYCLE_FOUR", model_seed=0, identity_match=True, reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT),
        _candidate("e2", "c2", operation="BIND", relation_id="BIND-COUNT", model_seed=1, identity_match=True, reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.ACCEPT),
    )
    aggregated = aggregate_candidate_metrics(records)
    assert aggregated["overall"]["n_records"] == 2
    assert set(aggregated["by_operation"]) == {"SHIFT", "BIND"}
    assert set(aggregated["by_model_seed"]) == {"0", "1"}
    assert set(aggregated["by_relation"]) == {"SHIFT-CYCLE_FOUR", "BIND-COUNT"}
    assert aggregated["by_operation"]["BIND"]["same_identity_unsafe_accept_rate"]["value"] == 1.0


# ---------------------------------------------------------------------------
# Episode-level metrics.
# ---------------------------------------------------------------------------


def test_legacy_known_task_plastic_rate() -> None:
    records = (
        EpisodeOutcomeRecord("e1", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=True, has_adequate_witness=True, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
        EpisodeOutcomeRecord("e2", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=False, has_adequate_witness=True, controller_action=ControllerAction.DIRECT_REUSE, final_call_reference_state=ReferenceAdequacyState.REF_ADEQUATE, final_output_is_correct_reuse_or_compose=True),
        EpisodeOutcomeRecord("e3", "SHIFT", 0, known_label=False, legacy_runtime_chose_plastic=True, has_adequate_witness=False, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
    )
    result = legacy_known_task_plastic_rate(records)
    assert result.numerator == 1
    assert result.denominator == 2  # only known_label episodes


def test_unsafe_reuse_episode_rate() -> None:
    records = (
        EpisodeOutcomeRecord("e1", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=False, has_adequate_witness=True, controller_action=ControllerAction.DIRECT_REUSE, final_call_reference_state=ReferenceAdequacyState.REF_INADEQUATE, final_output_is_correct_reuse_or_compose=False),
        EpisodeOutcomeRecord("e2", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=False, has_adequate_witness=True, controller_action=ControllerAction.DIRECT_REUSE, final_call_reference_state=ReferenceAdequacyState.REF_ADEQUATE, final_output_is_correct_reuse_or_compose=True),
        EpisodeOutcomeRecord("e3", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=True, has_adequate_witness=False, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
    )
    result = unsafe_reuse_episode_rate(records)
    assert result.denominator == 2  # only reference-determinable finals
    assert result.numerator == 1


def test_avoidable_plastic_rate_and_adequate_solution_nonreuse_rate() -> None:
    records = (
        EpisodeOutcomeRecord("e1", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=True, has_adequate_witness=True, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
        EpisodeOutcomeRecord("e2", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=False, has_adequate_witness=True, controller_action=ControllerAction.DIRECT_REUSE, final_call_reference_state=ReferenceAdequacyState.REF_ADEQUATE, final_output_is_correct_reuse_or_compose=True),
        EpisodeOutcomeRecord("e3", "SHIFT", 0, known_label=False, legacy_runtime_chose_plastic=True, has_adequate_witness=False, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
    )
    avoidable = avoidable_plastic_rate(records)
    nonreuse = adequate_solution_nonreuse_rate(records)
    assert avoidable.denominator == 2  # episodes with a witness
    assert avoidable.numerator == 1
    assert nonreuse.denominator == 2
    assert nonreuse.numerator == 1


def test_aggregate_episode_metrics_breakdowns() -> None:
    records = (
        EpisodeOutcomeRecord("e1", "SHIFT", 0, known_label=True, legacy_runtime_chose_plastic=True, has_adequate_witness=True, controller_action=ControllerAction.PLASTIC_SEARCH, final_call_reference_state=None, final_output_is_correct_reuse_or_compose=False),
        EpisodeOutcomeRecord("e2", "BIND", 1, known_label=True, legacy_runtime_chose_plastic=False, has_adequate_witness=True, controller_action=ControllerAction.DIRECT_REUSE, final_call_reference_state=ReferenceAdequacyState.REF_ADEQUATE, final_output_is_correct_reuse_or_compose=True),
    )
    aggregated = aggregate_episode_metrics(records)
    assert set(aggregated["by_operation"]) == {"SHIFT", "BIND"}
    assert set(aggregated["by_model_seed"]) == {"0", "1"}


# ---------------------------------------------------------------------------
# Query-level metrics: abstention counts against unconditional EM.
# ---------------------------------------------------------------------------


def test_unconditional_vs_selective_em_with_abstention() -> None:
    records = (
        EpisodeQueryRecord("e1", "q1", "SHIFT", 0, output_produced=True, correct=True),
        EpisodeQueryRecord("e1", "q2", "SHIFT", 0, output_produced=True, correct=False),
        EpisodeQueryRecord("e1", "q3", "SHIFT", 0, output_produced=False, correct=None),
    )
    uncond = unconditional_query_em(records)
    selective = selective_query_em(records)
    coverage = execution_coverage(records)
    assert uncond.numerator == 1
    assert uncond.denominator == 3
    assert selective.numerator == 1
    assert selective.denominator == 2  # abstained query excluded from selective denominator
    assert coverage.numerator == 2
    assert coverage.denominator == 3


def test_all_abstained_gives_zero_unconditional_em_and_null_selective_em() -> None:
    records = (EpisodeQueryRecord("e1", "q1", "SHIFT", 0, output_produced=False, correct=None),)
    assert unconditional_query_em(records).value == 0.0
    assert selective_query_em(records).value is None
    assert execution_coverage(records).value == 0.0


def test_aggregate_query_metrics_breakdowns() -> None:
    records = (
        EpisodeQueryRecord("e1", "q1", "SHIFT", 0, output_produced=True, correct=True),
        EpisodeQueryRecord("e2", "q2", "BIND", 1, output_produced=False, correct=None),
    )
    aggregated = aggregate_query_metrics(records)
    assert set(aggregated["by_operation"]) == {"SHIFT", "BIND"}
    assert set(aggregated["by_model_seed"]) == {"0", "1"}


# ---------------------------------------------------------------------------
# Nominal / stress suite partition: no result-based filtering.
# ---------------------------------------------------------------------------


def test_partition_by_suite_does_not_filter_nominal_by_outcome() -> None:
    records = (
        _candidate("e1", "c1", identity_match=True, reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT),
        _candidate("e2", "c2", identity_match=True, reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT),
    )
    suite_by_episode = {"e1": SuiteKind.NOMINAL, "e2": SuiteKind.DEGRADED_CANDIDATE_SAFETY_STRESS}
    partitioned = partition_by_suite(records, suite_by_episode)
    assert len(partitioned[SuiteKind.NOMINAL]) == 1
    assert partitioned[SuiteKind.NOMINAL][0].reference_state == ReferenceAdequacyState.REF_INADEQUATE
    assert len(partitioned[SuiteKind.DEGRADED_CANDIDATE_SAFETY_STRESS]) == 1


# ---------------------------------------------------------------------------
# Fixture results / schema / statistical contract artifacts.
# ---------------------------------------------------------------------------


def test_build_metric_fixture_results_all_pass() -> None:
    contract = FiniteLookVerifierContract()
    results = build_metric_fixture_results(contract=contract)
    assert results["all_fixtures_pass"] is True
    expected_checks = {
        "empty_denominator_to_null",
        "reference_unresolved_stays_unresolved",
        "abstention_counts_as_incorrect_unconditional",
        "same_identity_inadequate_accept_is_unsafe",
        "functionally_equivalent_wrong_id_excluded_from_wrong_call_pool",
        "unevaluated_recipe_blocks_inadequate_within_scope_claim",
        "nominal_suite_not_filtered_by_result",
    }
    assert expected_checks <= set(results["checks"])
    for check in results["checks"].values():
        assert check["pass"] is True


def test_build_metrics_schema_v2_declares_not_connected_to_runtime() -> None:
    schema = build_metrics_schema_v2()
    assert schema["runtime_connection_status"].startswith("NOT_CONNECTED")
    assert set(schema["reference_adequacy_state"]) == {"REF_ADEQUATE", "REF_INADEQUATE", "REF_UNRESOLVED"}
    assert set(schema["candidate_verdict"]) == {"ACCEPT", "REJECT", "UNCERTAIN"}


def test_build_statistical_contract_declares_frozen_and_unconnected() -> None:
    contract = FiniteLookVerifierContract()
    statistical_contract = build_statistical_contract(contract, ReferenceAllocation())
    assert statistical_contract["frozen"] is True
    assert statistical_contract["connected_to_runtime"] is False


# ---------------------------------------------------------------------------
# End-to-end orchestration.
# ---------------------------------------------------------------------------


def test_run_functional_metrics_v2_protocol_writes_artifacts_and_passes(tmp_path: Path) -> None:
    config = FunctionalMetricsV2Config(output_dir=tmp_path / "r3_003_run")
    report = run_functional_metrics_v2_protocol(config)

    assert report["protocol"]["result"] == "INFRASTRUCTURE_OR_PROTOCOL_PASS"
    assert report["protocol"]["criteria"]["runtime_verifier_connected"] is False
    assert report["protocol"]["criteria"]["model_weights_changed"] is False

    output_dir = config.output_dir
    for filename in (
        "metrics_schema_v2.json",
        "statistical_contract.json",
        "budget_feasibility.json",
        "metric_fixture_results.json",
        "config.yaml",
        "system.json",
        "protocol.json",
    ):
        assert (output_dir / filename).is_file(), f"missing artifact: {filename}"
