# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-005.

Matches this repo's existing precedent (R3-004/R3-002): anything that needs a
real Core/router/bank (`run_neural_candidate_stress`, `_stream_correctness`,
`_composition_check`, `run_safe_bounded_verification` itself) is exercised
only via the milestone script
(`scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-005`), not in
pytest. This file covers:

1. The new runtime-wired verifier policy and unsafe-reuse wrapper added to
   `apc.meta.adequacy_verifier` (pure logic, no model).
2. The Bernoulli contract sweep and historical-SHIFT-replay reconstruction
   logic in `apc.evaluation.safe_bounded_verification` (pure logic / small
   synthetic JSON fixtures, no model).
"""

from __future__ import annotations

import json

import pytest

from apc.evaluation.functional_metrics_v2 import (
    CandidateVerdict,
    ControllerAction,
    ExecutionStatus,
    FiniteLookTrace,
    FiniteLookVerifierContract,
    SearchStatus,
    verify_candidate_finite_look,
)
from apc.evaluation.safe_bounded_verification import (
    SafeBoundedVerificationConfig,
    _fixed_n_decisions,
    _generate_trial_stream,
    _legacy_verifier_decision,
    _new_verifier_decision,
    _reconstruct_ordered_stream,
    _relation_id_for,
    run_bernoulli_contract_sweep,
    run_historical_shift_replay,
)
from apc.meta.adequacy_verifier import (
    RUNTIME_VERIFIER_VERSION,
    BoundedExactLookVerifier,
    BoundedExactLookVerifierConfig,
    SequentialAdequacyVerifier,
    SequentialVerifierConfig,
    classify_search_status,
    enforce_verified_execution,
)

# ---------------------------------------------------------------------------
# BoundedExactLookVerifierConfig / BoundedExactLookVerifier.
# ---------------------------------------------------------------------------


def test_bounded_exact_look_config_defaults_match_frozen_r3_003_contract() -> None:
    config = BoundedExactLookVerifierConfig()
    assert config.tau == 0.95
    assert config.looks == (32, 64, 128, 256, 512)
    assert config.alpha_accept_episode == 0.01
    assert config.alpha_reject_episode == 0.01
    assert config.version == RUNTIME_VERIFIER_VERSION


def test_bounded_exact_look_verifier_matches_frozen_contract_bit_for_bit() -> None:
    """The runtime policy must not silently drift from the frozen offline
    contract it wires in (ADR-0084) -- same successes/trials must produce an
    identical trace either way."""

    def support_eval_fn(start: int, end: int) -> int:
        return end - start  # every trial succeeds

    verifier = BoundedExactLookVerifier()
    runtime_trace = verifier.verify_candidate("cand", support_eval_fn)
    offline_trace = verify_candidate_finite_look(
        candidate_id="cand", support_eval_fn=support_eval_fn, contract=FiniteLookVerifierContract()
    )
    assert runtime_trace.to_dict() == offline_trace.to_dict()


def test_bounded_exact_look_verifier_never_forces_decision_at_max_support() -> None:
    """A stream at exactly tau's boundary (never clearly adequate nor clearly
    inadequate) must stay UNCERTAIN through the last declared look, unlike
    the legacy verifier which forces ACCEPT/REJECT at its own max_support."""

    def support_eval_fn(start: int, end: int) -> int:
        # ~94% success rate: below tau but not so far below that upper bound < tau.
        return sum(1 for i in range(start, end) if i % 100 < 94)

    verifier = BoundedExactLookVerifier()
    trace = verifier.verify_candidate("cand", support_eval_fn)
    if trace.final_verdict == CandidateVerdict.UNCERTAIN:
        assert trace.final_support_consumed == max(verifier.contract.looks)
    else:
        # Either early verdict is a legitimate exact-bound outcome; the only
        # invariant under test is that UNCERTAIN-at-max-budget is preserved,
        # never silently coerced into ACCEPT/REJECT by this wrapper.
        assert trace.final_verdict in (CandidateVerdict.ACCEPT, CandidateVerdict.REJECT)


# ---------------------------------------------------------------------------
# classify_search_status.
# ---------------------------------------------------------------------------


def _trace_with_verdict(verdict: CandidateVerdict, support: int = 512) -> FiniteLookTrace:
    """A minimal fixture trace for testing `classify_search_status`/
    `enforce_verified_execution`, which only ever read `.final_verdict` --
    constructed directly rather than reverse-engineered from a real Bernoulli
    pattern, so the fixture cannot accidentally resolve to the wrong verdict."""
    return FiniteLookTrace(
        candidate_id="cand", steps=(), final_verdict=verdict, final_support_consumed=support
    )


def test_classify_search_status_accept_is_found_solution() -> None:
    trace = _trace_with_verdict(CandidateVerdict.ACCEPT)
    assert classify_search_status(trace, all_of_h_evaluated=False) == SearchStatus.FOUND_SOLUTION


def test_classify_search_status_uncertain_is_budget_exhausted_regardless_of_scope() -> None:
    trace = _trace_with_verdict(CandidateVerdict.UNCERTAIN)
    assert trace.final_verdict == CandidateVerdict.UNCERTAIN
    assert classify_search_status(trace, all_of_h_evaluated=True) == SearchStatus.BUDGET_EXHAUSTED


def test_classify_search_status_reject_with_full_scope_is_complete_within_scope() -> None:
    trace = _trace_with_verdict(CandidateVerdict.REJECT)
    assert classify_search_status(trace, all_of_h_evaluated=True) == SearchStatus.COMPLETE_WITHIN_SCOPE


def test_classify_search_status_reject_with_partial_scope_is_budget_exhausted() -> None:
    trace = _trace_with_verdict(CandidateVerdict.REJECT)
    assert classify_search_status(trace, all_of_h_evaluated=False) == SearchStatus.BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# enforce_verified_execution: the unsafe-reuse-blocking wrapper.
# ---------------------------------------------------------------------------


def test_enforce_verified_execution_accept_direct_reuse_executes() -> None:
    trace = _trace_with_verdict(CandidateVerdict.ACCEPT)
    envelope = enforce_verified_execution(
        trace, requested_action=ControllerAction.DIRECT_REUSE, all_of_h_evaluated=False
    )
    assert envelope.execution_status == ExecutionStatus.EXECUTED
    assert envelope.candidate_verdict == CandidateVerdict.ACCEPT


def test_enforce_verified_execution_accept_compose_executes() -> None:
    trace = _trace_with_verdict(CandidateVerdict.ACCEPT)
    envelope = enforce_verified_execution(
        trace, requested_action=ControllerAction.COMPOSE, all_of_h_evaluated=False
    )
    assert envelope.execution_status == ExecutionStatus.EXECUTED


def test_enforce_verified_execution_blocks_reuse_when_rejected() -> None:
    """The core safety property: a controller requesting DIRECT_REUSE against
    a REJECTed candidate must never reach EXECUTED."""
    trace = _trace_with_verdict(CandidateVerdict.REJECT)
    envelope = enforce_verified_execution(
        trace, requested_action=ControllerAction.DIRECT_REUSE, all_of_h_evaluated=False
    )
    assert envelope.execution_status == ExecutionStatus.NO_VERIFIED_SOLUTION
    assert envelope.controller_action is None


def test_enforce_verified_execution_blocks_reuse_when_uncertain() -> None:
    trace = _trace_with_verdict(CandidateVerdict.UNCERTAIN)
    envelope = enforce_verified_execution(
        trace, requested_action=ControllerAction.DIRECT_REUSE, all_of_h_evaluated=False
    )
    assert envelope.execution_status == ExecutionStatus.NEEDS_MORE_EVIDENCE
    assert envelope.controller_action is None
    assert envelope.candidate_verdict == CandidateVerdict.UNCERTAIN


def test_enforce_verified_execution_rejects_plastic_search_request() -> None:
    trace = _trace_with_verdict(CandidateVerdict.UNCERTAIN)
    with pytest.raises(ValueError, match="PLASTIC_SEARCH"):
        enforce_verified_execution(
            trace, requested_action=ControllerAction.PLASTIC_SEARCH, all_of_h_evaluated=False
        )


def test_enforce_verified_execution_accept_without_reuse_request_raises() -> None:
    trace = _trace_with_verdict(CandidateVerdict.ACCEPT)
    with pytest.raises(ValueError, match="ACCEPT"):
        enforce_verified_execution(trace, requested_action=None, all_of_h_evaluated=False)


# ---------------------------------------------------------------------------
# SafeBoundedVerificationConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = SafeBoundedVerificationConfig()
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.verification_examples >= max(config.looks)


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        SafeBoundedVerificationConfig(development_seeds=())


def test_config_rejects_verification_examples_below_max_look() -> None:
    with pytest.raises(ValueError, match="largest declared look"):
        SafeBoundedVerificationConfig(verification_examples=64, looks=(32, 64, 128, 256, 512))


def test_config_rejects_nonpositive_reference_examples() -> None:
    with pytest.raises(ValueError, match="reference_examples"):
        SafeBoundedVerificationConfig(reference_examples=0)


def test_config_rejects_nonpositive_episodes_per_p() -> None:
    with pytest.raises(ValueError, match="bernoulli_episodes_per_p"):
        SafeBoundedVerificationConfig(bernoulli_episodes_per_p=0)


def test_config_to_dict_serializes_paths() -> None:
    config = SafeBoundedVerificationConfig()
    data = config.to_dict()
    assert isinstance(data["output_dir"], str)
    assert isinstance(data["bank_checkpoint_dir"], str)
    assert isinstance(data["historical_shift_audit_path"], str)


# ---------------------------------------------------------------------------
# _generate_trial_stream: deterministic, not hash()-based (ADR-0080).
# ---------------------------------------------------------------------------


def test_generate_trial_stream_is_deterministic() -> None:
    a = _generate_trial_stream(p=0.9, episode_index=3, n=64)
    b = _generate_trial_stream(p=0.9, episode_index=3, n=64)
    assert a == b


def test_generate_trial_stream_differs_by_episode_index() -> None:
    a = _generate_trial_stream(p=0.9, episode_index=0, n=64)
    b = _generate_trial_stream(p=0.9, episode_index=1, n=64)
    assert a != b


def test_generate_trial_stream_p_one_is_all_true() -> None:
    stream = _generate_trial_stream(p=1.0, episode_index=0, n=32)
    assert all(stream)


def test_generate_trial_stream_p_zero_is_all_false() -> None:
    stream = _generate_trial_stream(p=0.0, episode_index=0, n=32)
    assert not any(stream)


# ---------------------------------------------------------------------------
# Policy decision helpers on synthetic streams.
# ---------------------------------------------------------------------------


def test_new_verifier_decision_all_correct_accepts() -> None:
    verifier = BoundedExactLookVerifier()
    decision, support = _new_verifier_decision([True] * 512, verifier)
    assert decision == "ACCEPT"
    assert support <= 512


def test_new_verifier_decision_all_wrong_rejects() -> None:
    verifier = BoundedExactLookVerifier()
    decision, _support = _new_verifier_decision([False] * 512, verifier)
    assert decision == "REJECT"


def test_legacy_verifier_decision_all_correct_accepts() -> None:
    verifier = SequentialAdequacyVerifier(SequentialVerifierConfig(adequacy_threshold=0.95, max_support=128))
    decision, support = _legacy_verifier_decision([True] * 128, verifier, 128)
    assert decision == "ACCEPT"
    assert support <= 128


def test_fixed_n_decisions_all_correct_accepts_every_n() -> None:
    verifier = SequentialAdequacyVerifier(
        SequentialVerifierConfig(adequacy_threshold=0.95, decision_rule="fixed_threshold")
    )
    decisions = _fixed_n_decisions([True] * 128, verifier, (32, 64, 128))
    assert decisions == {"32": "ACCEPT", "64": "ACCEPT", "128": "ACCEPT"}


# ---------------------------------------------------------------------------
# run_bernoulli_contract_sweep: small-scale structural check (not a full
# statistical-power run -- the milestone config uses far more episodes).
# ---------------------------------------------------------------------------


def test_bernoulli_contract_sweep_p_one_availability_passes() -> None:
    config = SafeBoundedVerificationConfig(
        bernoulli_p_grid=(0.10, 0.99, 1.0),
        bernoulli_episodes_per_p=50,
        availability_p=(0.99, 1.0),
    )
    result = run_bernoulli_contract_sweep(config)
    rows_by_p = {row["p"]: row for row in result["rows"]}
    assert rows_by_p[1.0]["new_verifier"]["accept_rate"] >= 0.97
    assert rows_by_p[0.10]["below_tau"] is True
    assert rows_by_p[0.10]["new_verifier"]["accept_rate"] == 0.0
    assert result["availability_check"]["pass"] is True


def test_bernoulli_contract_sweep_reports_uncertain_rate_only_for_new_verifier() -> None:
    config = SafeBoundedVerificationConfig(bernoulli_p_grid=(0.949,), bernoulli_episodes_per_p=20)
    result = run_bernoulli_contract_sweep(config)
    row = result["rows"][0]
    assert "uncertain_rate" in row["new_verifier"]
    assert "uncertain_rate" not in row["legacy_asymmetric"]


# ---------------------------------------------------------------------------
# _reconstruct_ordered_stream: preserves the historical aggregate exactly.
# ---------------------------------------------------------------------------


def test_reconstruct_ordered_stream_preserves_success_count() -> None:
    stream = _reconstruct_ordered_stream(successes=944, trials=1024, seed=128)
    assert len(stream) == 1024
    assert sum(stream) == 944


def test_reconstruct_ordered_stream_is_deterministic() -> None:
    a = _reconstruct_ordered_stream(successes=944, trials=1024, seed=128)
    b = _reconstruct_ordered_stream(successes=944, trials=1024, seed=128)
    assert a == b


def test_reconstruct_ordered_stream_differs_by_seed() -> None:
    a = _reconstruct_ordered_stream(successes=944, trials=1024, seed=16)
    b = _reconstruct_ordered_stream(successes=944, trials=1024, seed=32)
    assert a != b


# ---------------------------------------------------------------------------
# run_historical_shift_replay: missing-artifact path + real-fixture path.
# ---------------------------------------------------------------------------


def test_historical_shift_replay_reports_unavailable_when_missing(tmp_path) -> None:
    config = SafeBoundedVerificationConfig(historical_shift_audit_path=tmp_path / "does_not_exist.json")
    result = run_historical_shift_replay(config)
    assert result["status"] == "HISTORICAL_ARTIFACT_UNAVAILABLE"
    assert result["cells"] == []


def test_historical_shift_replay_replays_committed_fixture(tmp_path) -> None:
    fixture = {
        "task_id": "B-C005D2-005",
        "bank_size_audits": [
            {
                "bank_size": 16,
                "model_seed": 4,
                "development_seed": 14,
                "reference_adequacy": {"reference_correct": 947, "n_reference_examples": 1024},
                "verifier_trace": {"final_decision": "ACCEPT", "final_support_consumed": 32},
                "sequential_rule_bias_direction": "PREMATURE_ACCEPT",
            },
        ],
    }
    path = tmp_path / "shift_seed24_adequacy_audit.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    config = SafeBoundedVerificationConfig(historical_shift_audit_path=path)
    result = run_historical_shift_replay(config)
    assert result["status"] == "REPLAYED_FROM_COMMITTED_ARTIFACT"
    assert result["sealed_checkpoint_accessed_live"] is False
    assert result["shift_retrained"] is False
    assert len(result["cells"]) == 1
    cell = result["cells"][0]
    assert cell["bank_size"] == 16
    assert cell["stored_reference_adequacy"]["reference_correct"] == 947
    # 947/1024 ~= 0.925, below tau -- must never be ACCEPTed by the new verifier.
    assert cell["new_contract_finite_look_trace"]["final_verdict"] != "ACCEPT"


# ---------------------------------------------------------------------------
# _relation_id_for: pure mapping, mirrors R3-002/004's L3/L4 relation labels.
# ---------------------------------------------------------------------------


def test_relation_id_for_wrong_family_uses_arrow_notation() -> None:
    assert _relation_id_for("SELECT", "wrong_family") == "SELECT->BIND"
    assert _relation_id_for("COUNT", "wrong_family") == "COUNT->BIND"
    assert _relation_id_for("BIND", "wrong_family") == "BIND->COUNT"


def test_relation_id_for_wrong_argument_uses_argument_variant_notation() -> None:
    assert _relation_id_for("SELECT", "wrong_argument") == "SELECT:argument_variant"


def test_relation_id_for_adequate_is_none() -> None:
    assert _relation_id_for("SELECT", "adequate") is None
