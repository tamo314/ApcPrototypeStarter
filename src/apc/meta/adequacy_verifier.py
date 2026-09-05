"""Bounded sequential functional adequacy verifier (Phase B Task B-C005R2).

This module provides statistically justified functional verification for candidate
primitives to resolve finite-support estimator variance without lowering the
0.95 adequacy threshold or increasing wrong functional acceptance.

Key Invariants:
1. Zero query leakage:
   Candidate verification consumes strictly support examples. Query examples and
   query targets are completely inaccessible.
2. Functional execution is authoritative:
   Adequacy is evaluated exclusively by execution agreement (exact match) on
   model-visible token inputs and targets, never by router score, key distance,
   or registry status.
3. Statistically justified stopping rules:
   Clearly adequate candidates (empirical EM >= threshold) are accepted early.
   Clearly inadequate candidates (confidence interval upper bound < threshold)
   are rejected early.
   Uncertain candidates (empirical EM < threshold <= upper bound) gather additional
   evidence in declared increments up to a strict maximum support budget.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Final

_EPS: Final[float] = 1e-9


class AdequacyDecision(str, enum.Enum):  # noqa: UP042
    """Discrete adequacy verification decision."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ConfidenceInterval:
    """Binomial proportion confidence interval record."""

    lower: float
    upper: float
    point_estimate: float
    sample_size: int
    successes: int
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _normal_quantile(p: float) -> float:
    """Compute standard normal quantile Phi^{-1}(p) with high accuracy.

    Uses scipy.stats.norm.ppf if available, with an exact rational approximation
    fallback (Acklam's algorithm) ensuring zero external dependency issues.
    """
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        # Peter John Acklam's algorithm for inverse normal cumulative distribution
        if p <= 0.0 or p >= 1.0:
            raise ValueError("p must be in (0, 1)") from None

        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        p_low = 0.02425
        p_high = 1.0 - p_low

        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            return (
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
    alternative: str = "two_sided",
) -> ConfidenceInterval:
    """Calculate the Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successful matches (exact matches).
        trials: Total number of evaluated examples (trials >= 1).
        confidence_level: Confidence level in (0, 1), default 0.95.
        alternative: "two_sided" or "one_sided".

    Returns:
        ConfidenceInterval with lower and upper bounds clipped to [0, 1].
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    if alternative == "two_sided":
        alpha = 1.0 - confidence_level
        z = _normal_quantile(1.0 - alpha / 2.0)
    elif alternative == "one_sided":
        alpha = 1.0 - confidence_level
        z = _normal_quantile(1.0 - alpha)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    p_hat = successes / trials
    z2 = z * z
    n = float(trials)

    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=p_hat,
        sample_size=trials,
        successes=successes,
        confidence_level=confidence_level,
    )


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate the exact Clopper-Pearson interval for a binomial proportion."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")

    try:
        from scipy.stats import beta

        alpha = 1.0 - confidence_level
        lower = (
            0.0
            if successes == 0
            else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
        )
        upper = (
            1.0
            if successes == trials
            else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
        )
    except Exception:
        # Fallback to Wilson interval if scipy is unavailable
        return wilson_score_interval(successes, trials, confidence_level)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=successes / trials,
        sample_size=trials,
        successes=successes,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True)
class SequentialVerifierConfig:
    """Configuration for sequential bounded adequacy verification."""

    adequacy_threshold: float = 0.95
    confidence_level: float = 0.95
    initial_support: int = 32
    support_increment: int = 32
    max_support: int = 128
    interval_method: str = "wilson"
    alternative: str = "two_sided"
    decision_rule: str = "sequential_confidence"

    def __post_init__(self) -> None:
        if not 0.0 < self.adequacy_threshold <= 1.0:
            raise ValueError("adequacy_threshold must be in (0, 1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.initial_support < 1:
            raise ValueError("initial_support must be >= 1")
        if self.support_increment < 1:
            raise ValueError("support_increment must be >= 1")
        if self.max_support < self.initial_support:
            raise ValueError("max_support must be >= initial_support")
        if self.interval_method not in ("wilson", "clopper_pearson"):
            raise ValueError(f"Unsupported interval_method: {self.interval_method}")
        if self.decision_rule not in ("sequential_confidence", "fixed_threshold"):
            raise ValueError(f"Unsupported decision_rule: {self.decision_rule}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CandidateVerificationStep:
    """Record of a single verification step for a candidate."""

    step_index: int
    support_size: int
    correct_count: int
    empirical_accuracy: float
    confidence_interval: ConfidenceInterval
    decision: AdequacyDecision


@dataclass
class CandidateVerificationTrace:
    """Audit trail of sequential verification for one candidate."""

    candidate_id: str
    execute_primitive_id: int
    steps: list[CandidateVerificationStep] = field(default_factory=list)
    final_decision: AdequacyDecision = AdequacyDecision.UNCERTAIN
    final_support_consumed: int = 0
    final_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "execute_primitive_id": self.execute_primitive_id,
            "final_decision": self.final_decision.value,
            "final_support_consumed": self.final_support_consumed,
            "final_accuracy": self.final_accuracy,
            "steps": [
                {
                    "step_index": s.step_index,
                    "support_size": s.support_size,
                    "correct_count": s.correct_count,
                    "empirical_accuracy": s.empirical_accuracy,
                    "lower": s.confidence_interval.lower,
                    "upper": s.confidence_interval.upper,
                    "decision": s.decision.value,
                }
                for s in self.steps
            ],
        }


class SequentialAdequacyVerifier:
    """Bounded sequential verifier for functional adequacy.

    Evaluates candidate execution on support batches of increasing size
    (initial_support -> initial_support + support_increment -> ... -> max_support).

    Stopping logic:
    - If empirical EM >= adequacy_threshold: clearly adequate -> ACCEPT early.
    - If confidence interval upper bound < adequacy_threshold: clearly inadequate -> REJECT early.
    - If empirical EM < adequacy_threshold <= upper bound: uncertain -> request more support.
    - At max_support: forced decision based on empirical EM >= adequacy_threshold.
    """

    def __init__(self, config: SequentialVerifierConfig | None = None) -> None:
        self.config = config or SequentialVerifierConfig()

    def compute_interval(self, successes: int, trials: int) -> ConfidenceInterval:
        """Compute proportion confidence interval using configured method."""
        if self.config.interval_method == "wilson":
            return wilson_score_interval(
                successes,
                trials,
                confidence_level=self.config.confidence_level,
                alternative=self.config.alternative,
            )
        if self.config.interval_method == "clopper_pearson":
            return clopper_pearson_interval(
                successes,
                trials,
                confidence_level=self.config.confidence_level,
            )
        raise ValueError(f"Unknown interval method: {self.config.interval_method}")

    def evaluate_step(
        self,
        successes: int,
        trials: int,
    ) -> tuple[AdequacyDecision, ConfidenceInterval]:
        """Classify a candidate's status at the current sample size."""
        interval = self.compute_interval(successes, trials)

        # Fixed-threshold baseline mode
        if self.config.decision_rule == "fixed_threshold":
            if interval.point_estimate >= self.config.adequacy_threshold:
                return AdequacyDecision.ACCEPT, interval
            return AdequacyDecision.REJECT, interval

        # Sequential confidence mode
        if trials >= self.config.max_support:
            # Reached maximum support budget: final decision
            if interval.point_estimate >= self.config.adequacy_threshold:
                return AdequacyDecision.ACCEPT, interval
            return AdequacyDecision.REJECT, interval

        # Intermediate budget step
        if interval.point_estimate >= self.config.adequacy_threshold:
            # Clearly adequate: matches or exceeds the target exact match threshold
            return AdequacyDecision.ACCEPT, interval

        if interval.upper < self.config.adequacy_threshold:
            # Clearly inadequate: statistically excluded from achieving adequacy target
            return AdequacyDecision.REJECT, interval

        # Uncertain: empirical accuracy is below target,
        # but true accuracy could plausibly be >= target
        return AdequacyDecision.UNCERTAIN, interval

    def verify_candidate_sequentially(
        self,
        candidate_id: str,
        execute_primitive_id: int,
        support_eval_fn: Any,
        total_available_support: int,
    ) -> CandidateVerificationTrace:
        """Run sequential verification on a candidate using a chunk evaluation function.

        Args:
            candidate_id: Identifier of the candidate.
            execute_primitive_id: Physical primitive ID executed.
            support_eval_fn: Callable taking (start_idx, end_idx) and returning the
                number of correct predictions in that sub-slice.
            total_available_support: Total support examples available.

        Returns:
            CandidateVerificationTrace with all steps and final decision.
        """
        trace = CandidateVerificationTrace(
            candidate_id=candidate_id,
            execute_primitive_id=execute_primitive_id,
        )

        max_budget = min(self.config.max_support, total_available_support)
        current_support = 0
        total_successes = 0
        step_idx = 0

        while current_support < max_budget:
            if step_idx == 0:
                next_chunk_size = min(self.config.initial_support, max_budget)
            else:
                next_chunk_size = min(self.config.support_increment, max_budget - current_support)

            start_idx = current_support
            end_idx = current_support + next_chunk_size
            chunk_successes = support_eval_fn(start_idx, end_idx)

            total_successes += chunk_successes
            current_support = end_idx

            decision, interval = self.evaluate_step(total_successes, current_support)

            trace.steps.append(
                CandidateVerificationStep(
                    step_index=step_idx,
                    support_size=current_support,
                    correct_count=total_successes,
                    empirical_accuracy=interval.point_estimate,
                    confidence_interval=interval,
                    decision=decision,
                )
            )

            if decision != AdequacyDecision.UNCERTAIN:
                trace.final_decision = decision
                trace.final_support_consumed = current_support
                trace.final_accuracy = interval.point_estimate
                return trace

            step_idx += 1

        # Max budget exhausted without early stop
        decision, interval = self.evaluate_step(total_successes, current_support)
        trace.final_decision = decision
        trace.final_support_consumed = current_support
        trace.final_accuracy = interval.point_estimate
        return trace
