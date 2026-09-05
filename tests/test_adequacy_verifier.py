"""Unit tests for bounded sequential functional adequacy verifier (Task B-C005R2)."""

from apc.meta.adequacy_verifier import (
    AdequacyDecision,
    SequentialAdequacyVerifier,
    SequentialVerifierConfig,
    clopper_pearson_interval,
    wilson_score_interval,
)


def test_wilson_interval_analytical_values() -> None:
    # Analytical checks for Wilson interval with 95% confidence
    ci_full = wilson_score_interval(32, 32, 0.95)
    assert ci_full.point_estimate == 1.0
    assert 0.89 < ci_full.lower < 0.90
    assert ci_full.upper == 1.0

    ci_near = wilson_score_interval(30, 32, 0.95)
    assert ci_near.point_estimate == 30 / 32
    assert ci_near.upper > 0.95  # Uncertain zone: upper bound reaches above 0.95
    assert ci_near.lower < 0.85

    ci_wrong = wilson_score_interval(3, 32, 0.95)
    assert ci_wrong.point_estimate == 3 / 32
    assert ci_wrong.upper < 0.30  # Clearly inadequate


def test_clopper_pearson_interval() -> None:
    ci = clopper_pearson_interval(32, 32, 0.95)
    assert ci.point_estimate == 1.0
    assert 0.88 < ci.lower < 0.91
    assert ci.upper == 1.0


def test_sequential_verifier_step_decisions() -> None:
    config = SequentialVerifierConfig(
        adequacy_threshold=0.95,
        confidence_level=0.95,
        initial_support=32,
        support_increment=32,
        max_support=128,
        decision_rule="sequential_confidence",
    )
    verifier = SequentialAdequacyVerifier(config)

    # 1. Clearly adequate (31/32 = 0.96875 >= 0.95) -> ACCEPT early
    decision, interval = verifier.evaluate_step(31, 32)
    assert decision == AdequacyDecision.ACCEPT
    assert interval.point_estimate >= 0.95

    # 2. Clearly adequate (32/32 = 1.0 >= 0.95) -> ACCEPT early
    decision, interval = verifier.evaluate_step(32, 32)
    assert decision == AdequacyDecision.ACCEPT

    # 3. Clearly inadequate (2/32 = 0.0625) -> REJECT early (upper < 0.95)
    decision, interval = verifier.evaluate_step(2, 32)
    assert decision == AdequacyDecision.REJECT
    assert interval.upper < 0.95

    # 4. Uncertain candidate (30/32 = 0.9375 < 0.95, but upper >= 0.95) -> UNCERTAIN
    decision, interval = verifier.evaluate_step(30, 32)
    assert decision == AdequacyDecision.UNCERTAIN
    assert interval.point_estimate < 0.95
    assert interval.upper >= 0.95

    # 5. At max budget (128 examples), forced decision
    decision, interval = verifier.evaluate_step(122, 128)  # 122/128 = 0.9531 >= 0.95
    assert decision == AdequacyDecision.ACCEPT

    decision, interval = verifier.evaluate_step(120, 128)  # 120/128 = 0.9375 < 0.95
    assert decision == AdequacyDecision.REJECT


def test_sequential_verification_trace_early_accept() -> None:
    config = SequentialVerifierConfig(initial_support=32, max_support=128)
    verifier = SequentialAdequacyVerifier(config)

    # Perfect candidate: 32/32 correct in first chunk
    def eval_fn(start: int, end: int) -> int:
        return end - start

    trace = verifier.verify_candidate_sequentially("cand_perfect", 1, eval_fn, 128)
    assert trace.final_decision == AdequacyDecision.ACCEPT
    assert trace.final_support_consumed == 32
    assert len(trace.steps) == 1
    assert trace.final_accuracy == 1.0


def test_sequential_verification_trace_early_reject() -> None:
    config = SequentialVerifierConfig(initial_support=32, max_support=128)
    verifier = SequentialAdequacyVerifier(config)

    # Wrong candidate: 0/32 correct
    def eval_fn(start: int, end: int) -> int:
        return 0

    trace = verifier.verify_candidate_sequentially("cand_wrong", 2, eval_fn, 128)
    assert trace.final_decision == AdequacyDecision.REJECT
    assert trace.final_support_consumed == 32
    assert len(trace.steps) == 1


def test_sequential_verification_trace_resolution_of_uncertainty() -> None:
    config = SequentialVerifierConfig(
        initial_support=32, support_increment=32, max_support=128
    )
    verifier = SequentialAdequacyVerifier(config)

    # Candidate gets 30/32 on chunk 1 (uncertain), then 32/32 on chunk 2 -> 62/64 = 0.96875 (accept)
    def eval_fn(start: int, end: int) -> int:
        if start == 0:
            return 30
        return end - start

    trace = verifier.verify_candidate_sequentially("cand_stochastic", 3, eval_fn, 128)
    assert trace.final_decision == AdequacyDecision.ACCEPT
    assert trace.final_support_consumed == 64
    assert len(trace.steps) == 2
    assert trace.steps[0].decision == AdequacyDecision.UNCERTAIN
    assert trace.steps[1].decision == AdequacyDecision.ACCEPT


def test_fixed_threshold_policy_compatibility() -> None:
    config = SequentialVerifierConfig(
        initial_support=32,
        max_support=32,
        decision_rule="fixed_threshold",
    )
    verifier = SequentialAdequacyVerifier(config)

    def eval_fn_fail(start: int, end: int) -> int:
        return 30  # 30/32 = 0.9375 < 0.95

    trace = verifier.verify_candidate_sequentially("cand_30", 4, eval_fn_fail, 32)
    assert trace.final_decision == AdequacyDecision.REJECT
    assert trace.final_support_consumed == 32
