# ruff: noqa: E501
"""Task B-C005R3-003: Functional Metrics v2 & Statistical Contract (Option E, spec only).

`docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md` requires that "wrong competitor
acceptance" and "acceptance of a functionally-insufficient but correctly-identified
candidate" be measured as two distinct failure modes, and that the vocabulary for
acceptance / undetermined-evidence / insufficient-candidate / library-scope be
frozen *before* any change to the runtime verifier (that wiring is `B-C005R3-005`,
not this task). This module implements that frozen vocabulary and its supporting
offline statistics as pure, CPU-only code:

1. **Schema** (design doc S2, S6): `ReferenceAdequacyState`, `CandidateVerdict`,
   `SearchStatus`, `ControllerAction`, `ExecutionStatus`, `LibraryScopeStatus`, and
   the `VerifiedDecisionEnvelope` record with its cross-field invariants (an
   unsafe-reuse combination -- an envelope claiming `EXECUTED` without an `ACCEPT`
   verdict, or committing a `PLASTIC_SEARCH`/temporary-workspace action while still
   `UNCERTAIN` at budget exhaustion -- raises rather than silently constructing).
2. **Finite-look exact-bound statistical contract** (design doc S4): the frozen
   `tau=0.95`, `max_candidates=5`, `looks=[32,64,128,256,512]`,
   `alpha_accept_episode=alpha_reject_episode=0.01` union-bound allocation, the
   one-sided exact (Beta-quantile) bounds formula, and a sequential trace helper
   that -- unlike the legacy `SequentialAdequacyVerifier`
   (`apc.meta.adequacy_verifier`) -- never forces a decision at the final look;
   `UNCERTAIN` at max support stays `UNCERTAIN`. This is *not* wired into
   `apc.meta.adequacy_verifier` or any runtime candidate-selection path; `B-C005R3-005`
   does that wiring after this contract is frozen.
3. **Offline reference adequacy** (design doc S5): `REF_ADEQUATE` /
   `REF_INADEQUATE` / `REF_UNRESOLVED` classification from a per-call error-budget
   allocation across the episode's evaluated call set `M`, with an explicit `None`
   ("not estimable") result for an empty denominator rather than a forced binary.
4. **Metric aggregator** (design doc S7): every metric in that section's table,
   each carrying its own explicit numerator/denominator (`None` value, not `0.0`,
   when the denominator is empty), with overall / per-operation / per-relation /
   per-model-seed breakdowns.
5. **Nominal vs. degraded-candidate-safety-stress suite separation** (task doc
   item 6): partitioning is a pure lookup by a pre-declared suite tag with no
   filtering logic keyed on outcome, so a record cannot be silently dropped from
   the nominal suite because its candidate turned out to be inadequate.

Out of scope for this task (enforced by *not importing* any of the following):
`apc.meta.adequacy_verifier`'s `SequentialAdequacyVerifier`, any router/scorer/
primitive weights, and any training loop. No model is loaded, no checkpoint is
read, and no GPU is required; every function here is a pure computation over
plain Python/dataclass inputs, runnable entirely on CPU.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from apc.utils.system_info import get_system_info

_TAU_DEFAULT: Final[float] = 0.95


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Exact Beta(a, b) quantile.

    Design doc S4.3: "独自の近似式で「exact」と名乗らない" -- this function must
    use a real Beta-quantile implementation (scipy) and must raise, never
    silently substitute a normal/Wilson approximation under the "exact" name,
    if scipy is unavailable.
    """
    from scipy.stats import beta as beta_dist

    return float(beta_dist.ppf(p, a, b))


def one_sided_exact_bounds(
    successes: int, trials: int, alpha_lower: float, alpha_upper: float
) -> tuple[float, float]:
    """General one-sided exact (Clopper-Pearson-family) bounds, design doc S4.3.

    ``alpha_lower``/``alpha_upper`` are independent per-call error budgets (they
    need not be equal, unlike the classic two-sided Clopper-Pearson interval's
    ``alpha/2`` split), so this is the shared primitive behind both the
    candidate-verdict finite-look contract (S4) and offline reference adequacy
    (S5), which allocate their budgets differently.

    ``L = 0`` iff ``k == 0``; ``U = 1`` iff ``k == n``; otherwise both are exact
    Beta quantiles. Reference: [S4/S5] cited by the design doc.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    if not 0.0 < alpha_lower < 1.0:
        raise ValueError("alpha_lower must be in (0, 1)")
    if not 0.0 < alpha_upper < 1.0:
        raise ValueError("alpha_upper must be in (0, 1)")

    k, n = successes, trials
    lower = 0.0 if k == 0 else _beta_quantile(alpha_lower, k, n - k + 1)
    upper = 1.0 if k == n else _beta_quantile(1.0 - alpha_upper, k + 1, n - k)
    return lower, upper


# ---------------------------------------------------------------------------
# 1. Schema: reference state / candidate verdict / action envelope (design doc S2, S6).
# ---------------------------------------------------------------------------


class ReferenceAdequacyState(str, enum.Enum):  # noqa: UP042
    """Offline-only classification of a frozen candidate's true functional adequacy."""

    REF_ADEQUATE = "REF_ADEQUATE"
    REF_INADEQUATE = "REF_INADEQUATE"
    REF_UNRESOLVED = "REF_UNRESOLVED"


class CandidateVerdict(str, enum.Enum):  # noqa: UP042
    """Runtime (or offline-replayed) per-candidate verification verdict."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UNCERTAIN = "UNCERTAIN"


class SearchStatus(str, enum.Enum):  # noqa: UP042
    FOUND_SOLUTION = "FOUND_SOLUTION"
    COMPLETE_WITHIN_SCOPE = "COMPLETE_WITHIN_SCOPE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ControllerAction(str, enum.Enum):  # noqa: UP042
    DIRECT_REUSE = "DIRECT_REUSE"
    COMPOSE = "COMPOSE"
    PLASTIC_SEARCH = "PLASTIC_SEARCH"


class ExecutionStatus(str, enum.Enum):  # noqa: UP042
    EXECUTED = "EXECUTED"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    NO_VERIFIED_SOLUTION = "NO_VERIFIED_SOLUTION"


class LibraryScopeStatus(str, enum.Enum):  # noqa: UP042
    """Search-space/library-scope classification (design doc S2)."""

    FOUND_ADEQUATE_SOLUTION = "FOUND_ADEQUATE_SOLUTION"
    INADEQUATE_WITHIN_DECLARED_SCOPE = "INADEQUATE_WITHIN_DECLARED_SCOPE"
    UNRESOLVED_WITHIN_SCOPE = "UNRESOLVED_WITHIN_SCOPE"


class SuiteKind(str, enum.Enum):  # noqa: UP042
    """Task doc item 6: nominal performance and degraded-candidate safety stress
    are separate suites; an evaluated candidate is never moved between them
    based on its own outcome."""

    NOMINAL = "NOMINAL"
    DEGRADED_CANDIDATE_SAFETY_STRESS = "DEGRADED_CANDIDATE_SAFETY_STRESS"


def classify_library_scope(
    verdicts: Sequence[CandidateVerdict], *, all_of_h_evaluated: bool
) -> LibraryScopeStatus:
    """Design doc S2: never assert "the whole library is inadequate" (`INADEQUATE_
    WITHIN_DECLARED_SCOPE`) unless every recipe in the declared scope `H` was
    actually evaluated; a partially-evaluated scope with only REJECTs so far
    stays `UNRESOLVED_WITHIN_SCOPE`, regardless of how many REJECTs were seen.
    Any single ACCEPT dominates immediately, even mid-search."""
    if any(v == CandidateVerdict.ACCEPT for v in verdicts):
        return LibraryScopeStatus.FOUND_ADEQUATE_SOLUTION
    if all_of_h_evaluated and all(v == CandidateVerdict.REJECT for v in verdicts):
        return LibraryScopeStatus.INADEQUATE_WITHIN_DECLARED_SCOPE
    return LibraryScopeStatus.UNRESOLVED_WITHIN_SCOPE


@dataclass(frozen=True)
class VerifiedDecisionEnvelope:
    """Thin `VerifiedDecisionEnvelope`-equivalent wrapper (design doc S6).

    Cross-field invariants enforced at construction (never at the call site,
    so an invalid envelope cannot be silently built and later misread):

    - `EXECUTED` requires `candidate_verdict == ACCEPT` and `controller_action`
      in `{DIRECT_REUSE, COMPOSE}` -- "controllerがunsafe reuseを選ぼうとしても、
      ACCEPTされたcandidateがなければ実行を許可しない."
    - `NEEDS_MORE_EVIDENCE` requires `search_status == BUDGET_EXHAUSTED` and
      `controller_action is None` -- "最大予算で未確定なら...temporary workspace
      を確保せず、明示的にdeferする" (no `PLASTIC_SEARCH` commitment while still
      `UNCERTAIN`).
    - `NO_VERIFIED_SOLUTION` requires `candidate_verdict != ACCEPT`.
    """

    candidate_verdict: CandidateVerdict | None
    search_status: SearchStatus
    controller_action: ControllerAction | None
    execution_status: ExecutionStatus

    def __post_init__(self) -> None:
        if (
            self.execution_status == ExecutionStatus.EXECUTED
            and self.candidate_verdict != CandidateVerdict.ACCEPT
        ):
            raise ValueError(
                "EXECUTED requires an ACCEPT candidate_verdict (unsafe reuse without "
                "a verified ACCEPT is not a constructible envelope)"
            )
        if self.execution_status == ExecutionStatus.EXECUTED and self.controller_action not in (
            ControllerAction.DIRECT_REUSE,
            ControllerAction.COMPOSE,
        ):
            raise ValueError("EXECUTED requires controller_action DIRECT_REUSE or COMPOSE")
        if self.execution_status == ExecutionStatus.NEEDS_MORE_EVIDENCE:
            if self.search_status != SearchStatus.BUDGET_EXHAUSTED:
                raise ValueError("NEEDS_MORE_EVIDENCE requires search_status BUDGET_EXHAUSTED")
            if self.controller_action is not None:
                raise ValueError(
                    "NEEDS_MORE_EVIDENCE must not carry a controller_action "
                    "(no temporary workspace while a candidate is still UNCERTAIN)"
                )
        if (
            self.execution_status == ExecutionStatus.NO_VERIFIED_SOLUTION
            and self.candidate_verdict == CandidateVerdict.ACCEPT
        ):
            raise ValueError("NO_VERIFIED_SOLUTION cannot coexist with an ACCEPT candidate_verdict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_verdict": self.candidate_verdict.value if self.candidate_verdict else None,
            "search_status": self.search_status.value,
            "controller_action": self.controller_action.value if self.controller_action else None,
            "execution_status": self.execution_status.value,
        }


# ---------------------------------------------------------------------------
# 2. Finite-look exact-bound verifier contract (design doc S4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteLookVerifierContract:
    """Frozen candidate-verdict statistical contract (design doc S4.2).

    Not connected to any runtime path by this task; `B-C005R3-005` implements
    the runtime wiring against this frozen contract.
    """

    tau: float = _TAU_DEFAULT
    max_candidates: int = 5
    looks: tuple[int, ...] = (32, 64, 128, 256, 512)
    alpha_accept_episode: float = 0.01
    alpha_reject_episode: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if len(self.looks) == 0:
            raise ValueError("looks must be non-empty")
        if len(set(self.looks)) != len(self.looks) or list(self.looks) != sorted(self.looks):
            raise ValueError("looks must be strictly increasing with no duplicates")
        if any(look < 1 for look in self.looks):
            raise ValueError("every look must be >= 1")
        if not 0.0 < self.alpha_accept_episode < 1.0:
            raise ValueError("alpha_accept_episode must be in (0, 1)")
        if not 0.0 < self.alpha_reject_episode < 1.0:
            raise ValueError("alpha_reject_episode must be in (0, 1)")

    @property
    def alpha_accept_per_candidate_look(self) -> float:
        return self.alpha_accept_episode / (self.max_candidates * len(self.looks))

    @property
    def alpha_reject_per_candidate_look(self) -> float:
        return self.alpha_reject_episode / (self.max_candidates * len(self.looks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "max_candidates": self.max_candidates,
            "looks": list(self.looks),
            "alpha_accept_episode": self.alpha_accept_episode,
            "alpha_reject_episode": self.alpha_reject_episode,
            "alpha_accept_per_candidate_look": self.alpha_accept_per_candidate_look,
            "alpha_reject_per_candidate_look": self.alpha_reject_per_candidate_look,
        }


def finite_look_decision(
    successes: int, trials: int, contract: FiniteLookVerifierContract
) -> tuple[CandidateVerdict, float, float]:
    """One look's decision under the frozen contract (design doc S4.3)."""
    lower, upper = one_sided_exact_bounds(
        successes,
        trials,
        alpha_lower=contract.alpha_accept_per_candidate_look,
        alpha_upper=contract.alpha_reject_per_candidate_look,
    )
    if lower >= contract.tau:
        return CandidateVerdict.ACCEPT, lower, upper
    if upper < contract.tau:
        return CandidateVerdict.REJECT, lower, upper
    return CandidateVerdict.UNCERTAIN, lower, upper


@dataclass(frozen=True)
class FiniteLookStep:
    look_index: int
    support_size: int
    successes: int
    lower: float
    upper: float
    verdict: CandidateVerdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "look_index": self.look_index,
            "support_size": self.support_size,
            "successes": self.successes,
            "lower": self.lower,
            "upper": self.upper,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True)
class FiniteLookTrace:
    candidate_id: str
    steps: tuple[FiniteLookStep, ...]
    final_verdict: CandidateVerdict
    final_support_consumed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "steps": [step.to_dict() for step in self.steps],
            "final_verdict": self.final_verdict.value,
            "final_support_consumed": self.final_support_consumed,
        }


def verify_candidate_finite_look(
    *,
    candidate_id: str,
    support_eval_fn: Callable[[int, int], int],
    contract: FiniteLookVerifierContract,
) -> FiniteLookTrace:
    """Sequential trace across ``contract.looks``, fixed and ordered before any
    evidence is seen (design doc S3). Unlike the legacy
    `SequentialAdequacyVerifier.verify_candidate_sequentially`
    (`apc.meta.adequacy_verifier`), this never forces ACCEPT/REJECT at the final
    look: an `UNCERTAIN` verdict at max support stays `UNCERTAIN`
    (design doc S4.3: "最大supportまでUNCERTAINならUNCERTAINのまま返す").

    ``support_eval_fn(start_idx, end_idx)`` returns the correct-count in that
    half-open slice, matching the existing `SequentialAdequacyVerifier`
    convention so a caller can share one evaluation function across both.
    """
    steps: list[FiniteLookStep] = []
    total_successes = 0
    previous_look = 0
    for look_index, look in enumerate(contract.looks):
        total_successes += support_eval_fn(previous_look, look)
        verdict, lower, upper = finite_look_decision(total_successes, look, contract)
        steps.append(FiniteLookStep(look_index, look, total_successes, lower, upper, verdict))
        previous_look = look
        if verdict != CandidateVerdict.UNCERTAIN:
            return FiniteLookTrace(candidate_id, tuple(steps), verdict, look)
    return FiniteLookTrace(candidate_id, tuple(steps), CandidateVerdict.UNCERTAIN, previous_look)


def full_success_lower_bound(trials: int, contract: FiniteLookVerifierContract) -> float:
    """The ACCEPT lower bound when every trial in ``trials`` succeeds (design doc
    S4.5): ``BetaQuantile(alpha_accept_per_candidate_look; trials, 1)``, which
    equals the closed form ``alpha_accept_per_candidate_look ** (1 / trials)``."""
    lower, _ = one_sided_exact_bounds(
        trials,
        trials,
        alpha_lower=contract.alpha_accept_per_candidate_look,
        alpha_upper=contract.alpha_reject_per_candidate_look,
    )
    return lower


def min_successes_for_accept(trials: int, contract: FiniteLookVerifierContract) -> int | None:
    """Smallest ``k`` in ``[0, trials]`` whose exact lower bound reaches ``tau``
    at this look, or ``None`` if even ``k == trials`` cannot (design doc S4.5's
    "不可能" column). The lower bound is monotonically non-decreasing in ``k``
    for fixed ``trials``, so a binary search is exact, not a heuristic."""
    if full_success_lower_bound(trials, contract) < contract.tau:
        return None
    lo, hi = 0, trials
    while lo < hi:
        mid = (lo + hi) // 2
        lower, _ = one_sided_exact_bounds(
            mid,
            trials,
            alpha_lower=contract.alpha_accept_per_candidate_look,
            alpha_upper=contract.alpha_reject_per_candidate_look,
        )
        if lower >= contract.tau:
            hi = mid
        else:
            lo = mid + 1
    return lo


def build_budget_feasibility(contract: FiniteLookVerifierContract) -> dict[str, Any]:
    """`budget_feasibility.json`: reproduces design doc S4.5's table via the real
    Beta-quantile implementation (not the closed form alone), per look."""
    rows = []
    for look in contract.looks:
        full_lower = full_success_lower_bound(look, contract)
        min_k = min_successes_for_accept(look, contract)
        rows.append(
            {
                "look": look,
                "full_success_lower_bound": full_lower,
                "min_successes_for_accept": min_k,
                "accept_possible_at_this_look": min_k is not None,
            }
        )
    return {
        "task": "B-C005R3-003",
        "tau": contract.tau,
        "alpha_accept_per_candidate_look": contract.alpha_accept_per_candidate_look,
        "alpha_reject_per_candidate_look": contract.alpha_reject_per_candidate_look,
        "rows": rows,
        "source": (
            "computed via exact Beta quantile (scipy.stats.beta.ppf), design doc S4.5. "
            "This is a formula-level reference computation, not an APC measurement "
            "(design doc S4.5: 'これは式からの参考計算で、APC実測ではない')."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Offline reference adequacy (design doc S5).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceAllocation:
    """Per-episode reference error-budget allocation (design doc S5): a fixed
    ``alpha_ref_episode`` is split up/down and across the episode's evaluated
    call set ``m_calls`` -- never re-derived after seeing which calls turned
    out inadequate."""

    alpha_ref_episode: float = 0.01
    m_calls: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_ref_episode < 1.0:
            raise ValueError("alpha_ref_episode must be in (0, 1)")
        if self.m_calls < 1:
            raise ValueError("m_calls must be >= 1")

    @property
    def alpha_lower_per_call(self) -> float:
        return self.alpha_ref_episode / (2 * self.m_calls)

    @property
    def alpha_upper_per_call(self) -> float:
        return self.alpha_ref_episode / (2 * self.m_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_ref_episode": self.alpha_ref_episode,
            "m_calls": self.m_calls,
            "alpha_lower_per_call": self.alpha_lower_per_call,
            "alpha_upper_per_call": self.alpha_upper_per_call,
        }


def reference_adequacy_state(
    successes: int,
    trials: int,
    allocation: ReferenceAllocation,
    tau: float = _TAU_DEFAULT,
) -> ReferenceAdequacyState | None:
    """`REF_ADEQUATE` / `REF_INADEQUATE` / `REF_UNRESOLVED`, or `None` ("not
    estimable") when ``trials == 0`` (design doc S7: "空分母はnull/NOT_ESTIMABLE、
    0%としてPASSしない"). Never forces the mid-region to a binary adequate/
    inadequate call.
    """
    if trials == 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    lower, upper = one_sided_exact_bounds(
        successes,
        trials,
        alpha_lower=allocation.alpha_lower_per_call,
        alpha_upper=allocation.alpha_upper_per_call,
    )
    if lower >= tau:
        return ReferenceAdequacyState.REF_ADEQUATE
    if upper < tau:
        return ReferenceAdequacyState.REF_INADEQUATE
    return ReferenceAdequacyState.REF_UNRESOLVED


# ---------------------------------------------------------------------------
# 4. Metric records and aggregator (design doc S7).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """One metric's numerator/denominator/value. ``value`` is `None` (not
    `0.0`) when ``denominator == 0`` -- design doc S7: empty denominators are
    null, never a manufactured 0%/100%."""

    name: str
    numerator: int
    denominator: int
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


def compute_rate(name: str, numerator: int, denominator: int) -> MetricResult:
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError(
            f"invalid numerator/denominator for {name!r}: {numerator}/{denominator}"
        )
    value = None if denominator == 0 else numerator / denominator
    return MetricResult(name=name, numerator=numerator, denominator=denominator, value=value)


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    """One offline-replayed candidate evaluation within an episode.

    ``functionally_equivalent`` covers a candidate whose ``call_key`` differs
    from the task's correct call (different primitive id, alias, or an
    order-preserving-equivalent argument encoding) but which a reference
    evaluation confirms produces the correct functional output for this task
    distribution -- required test "機能同値wrong-IDの扱い": such a candidate must
    not be pooled with genuinely wrong, non-equivalent calls in
    `wrong_call_accept_rate`.
    """

    episode_id: str
    candidate_id: str
    operation: str
    relation_id: str | None
    model_seed: int
    identity_match: bool
    functionally_equivalent: bool
    reference_state: ReferenceAdequacyState | None
    verdict: CandidateVerdict

    @property
    def accepted(self) -> bool:
        return self.verdict == CandidateVerdict.ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "candidate_id": self.candidate_id,
            "operation": self.operation,
            "relation_id": self.relation_id,
            "model_seed": self.model_seed,
            "identity_match": self.identity_match,
            "functionally_equivalent": self.functionally_equivalent,
            "reference_state": self.reference_state.value if self.reference_state else None,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True)
class EpisodeQueryRecord:
    """One planned query within an episode (design doc S7 EM/coverage metrics).
    ``output_produced=False`` (abstention) always counts against
    `unconditional_query_EM`; `correct` is only meaningful when
    ``output_produced`` is True."""

    episode_id: str
    query_id: str
    operation: str
    model_seed: int
    output_produced: bool
    correct: bool | None


@dataclass(frozen=True)
class EpisodeOutcomeRecord:
    """One episode's controller-level outcome (design doc S7 plastic/reuse metrics)."""

    episode_id: str
    operation: str
    model_seed: int
    known_label: bool
    legacy_runtime_chose_plastic: bool
    has_adequate_witness: bool
    controller_action: ControllerAction | None
    final_call_reference_state: ReferenceAdequacyState | None
    final_output_is_correct_reuse_or_compose: bool


def _group_by(
    records: Sequence[Any], key_fn: Callable[[Any], Any]
) -> dict[Any, list[Any]]:
    groups: dict[Any, list[Any]] = {}
    for record in records:
        groups.setdefault(key_fn(record), []).append(record)
    return groups


# --- Candidate-level metrics -------------------------------------------------


def wrong_call_accept_rate(records: Sequence[CandidateEvaluationRecord]) -> MetricResult:
    pool = [r for r in records if not r.identity_match and not r.functionally_equivalent]
    numerator = sum(1 for r in pool if r.accepted)
    return compute_rate("wrong_call_accept_rate", numerator, len(pool))


def inadequate_call_accept_rate(records: Sequence[CandidateEvaluationRecord]) -> MetricResult:
    pool = [r for r in records if r.reference_state == ReferenceAdequacyState.REF_INADEQUATE]
    numerator = sum(1 for r in pool if r.accepted)
    return compute_rate("inadequate_call_accept_rate", numerator, len(pool))


def same_identity_unsafe_accept_rate(records: Sequence[CandidateEvaluationRecord]) -> MetricResult:
    pool = [
        r
        for r in records
        if r.identity_match and r.reference_state == ReferenceAdequacyState.REF_INADEQUATE
    ]
    numerator = sum(1 for r in pool if r.accepted)
    return compute_rate("same_identity_unsafe_accept_rate", numerator, len(pool))


def uncertain_candidate_rate(records: Sequence[CandidateEvaluationRecord]) -> MetricResult:
    numerator = sum(1 for r in records if r.verdict == CandidateVerdict.UNCERTAIN)
    return compute_rate("uncertain_candidate_rate", numerator, len(records))


def reference_unresolved_rate(records: Sequence[CandidateEvaluationRecord]) -> MetricResult:
    measured = [r for r in records if r.reference_state is not None]
    numerator = sum(1 for r in measured if r.reference_state == ReferenceAdequacyState.REF_UNRESOLVED)
    return compute_rate("reference_unresolved_rate", numerator, len(measured))


def accepted_ref_unresolved_count(records: Sequence[CandidateEvaluationRecord]) -> int:
    """Raw count, not a rate (design doc S7): accepted candidates whose
    reference adequacy was never resolved, reported separately so it is never
    hidden inside another metric's denominator."""
    return sum(
        1
        for r in records
        if r.accepted and r.reference_state == ReferenceAdequacyState.REF_UNRESOLVED
    )


_CANDIDATE_METRIC_FNS: Final[tuple[Callable[[Sequence[CandidateEvaluationRecord]], MetricResult], ...]] = (
    wrong_call_accept_rate,
    inadequate_call_accept_rate,
    same_identity_unsafe_accept_rate,
    uncertain_candidate_rate,
    reference_unresolved_rate,
)


def aggregate_candidate_metrics(records: Sequence[CandidateEvaluationRecord]) -> dict[str, Any]:
    def _block(group: Sequence[CandidateEvaluationRecord]) -> dict[str, Any]:
        block: dict[str, Any] = {fn.__name__: fn(group).to_dict() for fn in _CANDIDATE_METRIC_FNS}
        block["accepted_ref_unresolved_count"] = accepted_ref_unresolved_count(group)
        block["n_records"] = len(group)
        return block

    by_operation = {op: _block(group) for op, group in _group_by(records, lambda r: r.operation).items()}
    by_model_seed = {
        str(seed): _block(group) for seed, group in _group_by(records, lambda r: r.model_seed).items()
    }
    by_relation = {
        relation: _block(group)
        for relation, group in _group_by(records, lambda r: r.relation_id).items()
        if relation is not None
    }
    return {
        "overall": _block(records),
        "by_operation": by_operation,
        "by_model_seed": by_model_seed,
        "by_relation": by_relation,
    }


# --- Episode-level metrics ----------------------------------------------------


def legacy_known_task_plastic_rate(records: Sequence[EpisodeOutcomeRecord]) -> MetricResult:
    pool = [r for r in records if r.known_label]
    numerator = sum(1 for r in pool if r.legacy_runtime_chose_plastic)
    return compute_rate("legacy_known_task_plastic_rate", numerator, len(pool))


def unsafe_reuse_episode_rate(records: Sequence[EpisodeOutcomeRecord]) -> MetricResult:
    measured = [r for r in records if r.final_call_reference_state is not None]
    numerator = sum(
        1 for r in measured if r.final_call_reference_state == ReferenceAdequacyState.REF_INADEQUATE
    )
    return compute_rate("unsafe_reuse_episode_rate", numerator, len(measured))


def avoidable_plastic_rate(records: Sequence[EpisodeOutcomeRecord]) -> MetricResult:
    pool = [r for r in records if r.has_adequate_witness]
    numerator = sum(1 for r in pool if r.controller_action == ControllerAction.PLASTIC_SEARCH)
    return compute_rate("avoidable_plastic_rate", numerator, len(pool))


def adequate_solution_nonreuse_rate(records: Sequence[EpisodeOutcomeRecord]) -> MetricResult:
    pool = [r for r in records if r.has_adequate_witness]
    numerator = sum(1 for r in pool if not r.final_output_is_correct_reuse_or_compose)
    return compute_rate("adequate_solution_nonreuse_rate", numerator, len(pool))


_EPISODE_METRIC_FNS: Final[tuple[Callable[[Sequence[EpisodeOutcomeRecord]], MetricResult], ...]] = (
    legacy_known_task_plastic_rate,
    unsafe_reuse_episode_rate,
    avoidable_plastic_rate,
    adequate_solution_nonreuse_rate,
)


def aggregate_episode_metrics(records: Sequence[EpisodeOutcomeRecord]) -> dict[str, Any]:
    def _block(group: Sequence[EpisodeOutcomeRecord]) -> dict[str, Any]:
        block: dict[str, Any] = {fn.__name__: fn(group).to_dict() for fn in _EPISODE_METRIC_FNS}
        block["n_records"] = len(group)
        return block

    by_operation = {op: _block(group) for op, group in _group_by(records, lambda r: r.operation).items()}
    by_model_seed = {
        str(seed): _block(group) for seed, group in _group_by(records, lambda r: r.model_seed).items()
    }
    return {"overall": _block(records), "by_operation": by_operation, "by_model_seed": by_model_seed}


# --- Query-level metrics (EM / coverage) -------------------------------------


def unconditional_query_em(records: Sequence[EpisodeQueryRecord]) -> MetricResult:
    numerator = sum(1 for r in records if r.output_produced and r.correct)
    return compute_rate("unconditional_query_EM", numerator, len(records))


def selective_query_em(records: Sequence[EpisodeQueryRecord]) -> MetricResult:
    pool = [r for r in records if r.output_produced]
    numerator = sum(1 for r in pool if r.correct)
    return compute_rate("selective_query_EM", numerator, len(pool))


def execution_coverage(records: Sequence[EpisodeQueryRecord]) -> MetricResult:
    numerator = sum(1 for r in records if r.output_produced)
    return compute_rate("execution_coverage", numerator, len(records))


_QUERY_METRIC_FNS: Final[tuple[Callable[[Sequence[EpisodeQueryRecord]], MetricResult], ...]] = (
    unconditional_query_em,
    selective_query_em,
    execution_coverage,
)


def aggregate_query_metrics(records: Sequence[EpisodeQueryRecord]) -> dict[str, Any]:
    def _block(group: Sequence[EpisodeQueryRecord]) -> dict[str, Any]:
        block: dict[str, Any] = {fn.__name__: fn(group).to_dict() for fn in _QUERY_METRIC_FNS}
        block["n_records"] = len(group)
        return block

    by_operation = {op: _block(group) for op, group in _group_by(records, lambda r: r.operation).items()}
    by_model_seed = {
        str(seed): _block(group) for seed, group in _group_by(records, lambda r: r.model_seed).items()
    }
    return {"overall": _block(records), "by_operation": by_operation, "by_model_seed": by_model_seed}


# ---------------------------------------------------------------------------
# 5. Nominal / stress suite partition (task doc item 6).
# ---------------------------------------------------------------------------


def partition_by_suite(
    records: Sequence[CandidateEvaluationRecord], suite_by_episode: dict[str, SuiteKind]
) -> dict[SuiteKind, list[CandidateEvaluationRecord]]:
    """Partitions purely by the pre-declared ``suite_by_episode`` tag. Contains
    no filtering on ``reference_state``/``verdict``/outcome, by construction --
    task doc item 6: "nominalから不十分候補を結果に基づいて除外しない.\""""
    result: dict[SuiteKind, list[CandidateEvaluationRecord]] = {kind: [] for kind in SuiteKind}
    for record in records:
        result[suite_by_episode[record.episode_id]].append(record)
    return result


# ---------------------------------------------------------------------------
# 6. Fixture results and schema/contract artifacts.
# ---------------------------------------------------------------------------


def build_metrics_schema_v2() -> dict[str, Any]:
    """`metrics_schema_v2.json`."""
    return {
        "task": "B-C005R3-003",
        "reference_adequacy_state": [s.value for s in ReferenceAdequacyState],
        "candidate_verdict": [s.value for s in CandidateVerdict],
        "search_status": [s.value for s in SearchStatus],
        "controller_action": [s.value for s in ControllerAction],
        "execution_status": [s.value for s in ExecutionStatus],
        "library_scope_status": [s.value for s in LibraryScopeStatus],
        "suite_kind": [s.value for s in SuiteKind],
        "verified_decision_envelope_invariants": [
            "EXECUTED requires candidate_verdict == ACCEPT",
            "EXECUTED requires controller_action in {DIRECT_REUSE, COMPOSE}",
            "NEEDS_MORE_EVIDENCE requires search_status == BUDGET_EXHAUSTED",
            "NEEDS_MORE_EVIDENCE requires controller_action is None (no temporary workspace)",
            "NO_VERIFIED_SOLUTION cannot coexist with candidate_verdict == ACCEPT",
        ],
        "metric_definitions": {
            "legacy_known_task_plastic_rate": "known-label PLASTIC / known-label episodes",
            "wrong_call_accept_rate": "acceptance of a functionally non-equivalent wrong call / evaluated count of that candidate class",
            "inadequate_call_accept_rate": "REF_INADEQUATE acceptance / REF_INADEQUATE evaluation count",
            "unsafe_reuse_episode_rate": "final executed call is REF_INADEQUATE / reference-determinable final-reuse count",
            "same_identity_unsafe_accept_rate": "correct family/args AND REF_INADEQUATE acceptance / that evaluation count",
            "avoidable_plastic_rate": "PLASTIC chosen despite an adequate existing-solution witness / episodes with a witness",
            "adequate_solution_nonreuse_rate": "witness exists but output is not the correct reuse/compose / episodes with a witness",
            "uncertain_candidate_rate": "UNCERTAIN / all evaluated candidates",
            "reference_unresolved_rate": "REF_UNRESOLVED / reference-measured call count",
            "unconditional_query_EM": "correct queries / all planned queries (abstention counts as incorrect)",
            "selective_query_EM": "correct queries / queries actually output",
            "execution_coverage": "queries output / all planned queries",
            "accepted_ref_unresolved_count": "raw count (not a rate): accepted candidates with REF_UNRESOLVED",
        },
        "runtime_connection_status": (
            "NOT_CONNECTED -- this is an offline schema/metric contract only. "
            "B-C005R3-005 wires a runtime verifier against this frozen contract."
        ),
    }


def build_statistical_contract(
    contract: FiniteLookVerifierContract, reference_allocation_example: ReferenceAllocation
) -> dict[str, Any]:
    """`statistical_contract.json`."""
    return {
        "task": "B-C005R3-003",
        "finite_look_verifier_contract": contract.to_dict(),
        "reference_allocation_example": reference_allocation_example.to_dict(),
        "formula": {
            "one_sided_exact_bounds": (
                "L = 0 if k == 0 else BetaQuantile(alpha_lower; k, n-k+1); "
                "U = 1 if k == n else BetaQuantile(1-alpha_upper; k+1, n-k)"
            ),
            "candidate_verdict_rule": "L >= tau -> ACCEPT; U < tau -> REJECT; otherwise UNCERTAIN",
            "reference_state_rule": "L_ref >= tau -> REF_ADEQUATE; U_ref < tau -> REF_INADEQUATE; otherwise REF_UNRESOLVED",
        },
        "frozen": True,
        "connected_to_runtime": False,
        "connection_note": "B-C005R3-005 wires this contract into apc.meta.adequacy_verifier; not done by this task.",
    }


_UNRESOLVED_REFERENCE_FIXTURE: Final[tuple[int, int]] = (125, 128)  # verified: lower≈0.9169 < 0.95 <= upper≈0.9973


def build_metric_fixture_results(*, contract: FiniteLookVerifierContract) -> dict[str, Any]:
    """`metric_fixture_results.json`: runs every required fixture scenario from
    `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`'s B-C005R3-003 "必須テスト"
    list and records pass/fail, so the contract is inspectable outside pytest."""
    checks: dict[str, dict[str, Any]] = {}
    allocation = ReferenceAllocation(alpha_ref_episode=0.01, m_calls=1)

    state = reference_adequacy_state(0, 0, allocation)
    rate = compute_rate("dummy", 0, 0)
    checks["empty_denominator_to_null"] = {
        "reference_state": state,
        "metric_value": rate.value,
        "pass": state is None and rate.value is None,
    }

    successes, trials = _UNRESOLVED_REFERENCE_FIXTURE
    state2 = reference_adequacy_state(successes, trials, allocation)
    checks["reference_unresolved_stays_unresolved"] = {
        "successes": successes,
        "trials": trials,
        "reference_state": state2.value if state2 else None,
        "pass": state2 == ReferenceAdequacyState.REF_UNRESOLVED,
    }

    abstained = (
        EpisodeQueryRecord("ep1", "q1", "SHIFT", 0, output_produced=False, correct=None),
    )
    uncond = unconditional_query_em(abstained)
    coverage = execution_coverage(abstained)
    selective = selective_query_em(abstained)
    checks["abstention_counts_as_incorrect_unconditional"] = {
        "unconditional_em": uncond.value,
        "coverage": coverage.value,
        "selective_em_denominator": selective.denominator,
        "pass": uncond.value == 0.0 and coverage.value == 0.0 and selective.denominator == 0,
    }

    same_identity_inadequate = (
        CandidateEvaluationRecord(
            "ep2", "cand1", "SHIFT", None, 0,
            identity_match=True, functionally_equivalent=False,
            reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.ACCEPT,
        ),
    )
    unsafe = same_identity_unsafe_accept_rate(same_identity_inadequate)
    checks["same_identity_inadequate_accept_is_unsafe"] = {
        "rate": unsafe.value,
        "numerator": unsafe.numerator,
        "denominator": unsafe.denominator,
        "pass": unsafe.value == 1.0,
    }

    mixed_wrong_and_alias = (
        CandidateEvaluationRecord(
            "ep3", "cand_wrong", "SHIFT", None, 0,
            identity_match=False, functionally_equivalent=False,
            reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT,
        ),
        CandidateEvaluationRecord(
            "ep3", "cand_alias", "SHIFT", None, 0,
            identity_match=False, functionally_equivalent=True,
            reference_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT,
        ),
    )
    wrong_rate = wrong_call_accept_rate(mixed_wrong_and_alias)
    checks["functionally_equivalent_wrong_id_excluded_from_wrong_call_pool"] = {
        "rate": wrong_rate.value,
        "numerator": wrong_rate.numerator,
        "denominator": wrong_rate.denominator,
        "pass": wrong_rate.denominator == 1 and wrong_rate.numerator == 0,
    }

    scope_partial = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.REJECT), all_of_h_evaluated=False
    )
    scope_full = classify_library_scope(
        (CandidateVerdict.REJECT, CandidateVerdict.REJECT), all_of_h_evaluated=True
    )
    checks["unevaluated_recipe_blocks_inadequate_within_scope_claim"] = {
        "partial_evaluation_result": scope_partial.value,
        "full_evaluation_result": scope_full.value,
        "pass": (
            scope_partial == LibraryScopeStatus.UNRESOLVED_WITHIN_SCOPE
            and scope_full == LibraryScopeStatus.INADEQUATE_WITHIN_DECLARED_SCOPE
        ),
    }

    nominal_tagged_inadequate = (
        CandidateEvaluationRecord(
            "ep4", "cand_inadequate", "SHIFT", None, 0,
            identity_match=True, functionally_equivalent=False,
            reference_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT,
        ),
    )
    partitioned = partition_by_suite(nominal_tagged_inadequate, {"ep4": SuiteKind.NOMINAL})
    checks["nominal_suite_not_filtered_by_result"] = {
        "nominal_count": len(partitioned[SuiteKind.NOMINAL]),
        "pass": len(partitioned[SuiteKind.NOMINAL]) == 1,
    }

    all_pass = all(check["pass"] for check in checks.values())
    return {"task": "B-C005R3-003", "checks": checks, "all_fixtures_pass": all_pass}


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalMetricsV2Config:
    """Configuration for the B-C005R3-003 run."""

    tau: float = _TAU_DEFAULT
    max_candidates: int = 5
    looks: tuple[int, ...] = (32, 64, 128, 256, 512)
    alpha_accept_episode: float = 0.01
    alpha_reject_episode: float = 0.01
    alpha_ref_episode: float = 0.01
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_003_functional_metrics_v2")

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["output_dir"] = str(self.output_dir)
        return raw


def run_functional_metrics_v2_protocol(config: FunctionalMetricsV2Config) -> dict[str, Any]:
    """Execute B-C005R3-003 and write its four run artifacts + run metadata."""
    start = time.perf_counter()
    contract = FiniteLookVerifierContract(
        tau=config.tau,
        max_candidates=config.max_candidates,
        looks=config.looks,
        alpha_accept_episode=config.alpha_accept_episode,
        alpha_reject_episode=config.alpha_reject_episode,
    )
    reference_allocation_example = ReferenceAllocation(alpha_ref_episode=config.alpha_ref_episode, m_calls=1)

    schema = build_metrics_schema_v2()
    statistical_contract = build_statistical_contract(contract, reference_allocation_example)
    budget_feasibility = build_budget_feasibility(contract)
    fixture_results = build_metric_fixture_results(contract=contract)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_schema_v2.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output_dir / "statistical_contract.json").write_text(
        json.dumps(statistical_contract, indent=2), encoding="utf-8"
    )
    (output_dir / "budget_feasibility.json").write_text(
        json.dumps(budget_feasibility, indent=2), encoding="utf-8"
    )
    (output_dir / "metric_fixture_results.json").write_text(
        json.dumps(fixture_results, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")

    all_fixtures_pass = fixture_results["all_fixtures_pass"]
    gate_result = "INFRASTRUCTURE_OR_PROTOCOL_PASS" if all_fixtures_pass else "FAIL"

    protocol = {
        "task": "B-C005R3-003",
        "gate": "G2",
        "result": gate_result,
        "criteria": {
            "schema_and_envelope_defined": True,
            "statistical_contract_frozen": True,
            "runtime_verifier_connected": False,
            "all_required_fixtures_pass": all_fixtures_pass,
            "model_weights_changed": False,
            "loss_changed": False,
            "candidate_scorer_changed": False,
        },
        "downstream_note": (
            "G2 PASS freezes this schema/statistical contract for B-C005R3-004 "
            "(paired frozen baseline) and B-C005R3-005 (runtime verifier wiring "
            "against this exact contract). Runtime adequacy_verifier.py is "
            "unmodified by this task."
        ),
        "artifacts": [
            "metrics_schema_v2.json",
            "statistical_contract.json",
            "budget_feasibility.json",
            "metric_fixture_results.json",
            "config.yaml",
            "system.json",
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "metrics_schema_v2": schema,
        "statistical_contract": statistical_contract,
        "budget_feasibility": budget_feasibility,
        "metric_fixture_results": fixture_results,
        "protocol": protocol,
    }
