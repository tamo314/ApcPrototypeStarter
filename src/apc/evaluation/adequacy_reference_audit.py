# ruff: noqa: E501
"""Evaluation-only SHIFT seed-24 adequacy / false-plastic audit for B-C005D2-005.

`B-C005G` reported ``false plastic == 1.000`` in every cell associated with
sealed seed ``24`` and target ``SHIFT`` at bank sizes 16, 32, and 128 (all
five hard-negative levels), while bank size 64 accepted reuse
(``false_plastic == 0.0``). Because the sealed re-gate's support batch is
drawn from a seed formula that depends only on ``(sealed_seed, bank_size,
target_id)`` -- never on the hard-negative level -- the correct SHIFT
candidate's own execution and its sequential-verifier trace are identical
across all five levels for a fixed bank size; only the *competitor* added to
the candidate list differs by level (`build_hard_negative_candidates`). This
module reconstructs that frozen cell exactly (same base-system seed, same
scaled-bank seed, same R2 development-training recipe as
`hard_negative_repair_gate.run_hard_negative_regate`), then adds three
evaluation-only diagnostics ADR-0079's own summary numbers cannot separate:

1. The full step-by-step `SequentialAdequacyVerifier` trace for the correct
   candidate (support cumulative correct counts / Wilson bounds / decision at
   n=32,64,96,128), plus the diagnostic-only symmetric counterfactual rule
   (Wilson lower >= threshold -> accept, upper < threshold -> reject,
   otherwise uncertain) applied to the same evidence, to test whether the
   installed rule's early-accept/early-reject asymmetry itself drove the
   decision (D2-005.4).
2. A large (>= 1024 example), independent, never-support/query reference
   batch executed through the *same* installed SHIFT primitive to estimate
   its true functional adequacy, decoupled from any one small draw
   (D2-005.2).
3. A three-way reclassification -- true false plastic, functionally
   justified plastic, or unsafe reuse -- crossing the installed runtime
   decision against `reference_adequate` (D2-005.3), plus a spec-conformance
   check of the verifier's decision rule against its documented
   specification using synthetic (successes, trials) pairs, independent of
   any model or GPU (D2-005.4).

No sealed data enters a gradient anywhere in this module: the only training
that happens is the already-selected R2 development-partition recipe
(`_training_examples(..., regate_recipe=True)`), reused unmodified from
`hard_negative_second_diagnostic`. The support/query/reference batches are
evaluation-only reads of the frozen system.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.evaluation.adequacy_repair_benchmark import (
    AdequacyRepairConfig,
    evaluate_adequacy_policy_cell,
)
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_LEVELS,
    HardNegativeBenchmarkConfig,
    HardNegativeCandidate,
    _build_frozen_base_system,
    _execute_selected_candidates,
    build_hard_negative_candidates,
)
from apc.evaluation.hard_negative_second_diagnostic import _training_examples
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    SEALED_GATE_SEEDS,
    RetrievalRepairConfig,
    _rank_candidates_factorized,
    train_repaired_router_and_scorer,
)
from apc.meta.adequacy_verifier import (
    AdequacyDecision,
    CandidateVerificationTrace,
    SequentialAdequacyVerifier,
    SequentialVerifierConfig,
    clopper_pearson_interval,
    wilson_score_interval,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

_TARGET_OPERATION: Final[str] = "SHIFT"

# Synthetic (successes, trials) pairs spanning early-accept, early-reject,
# uncertain, and max-support-forced regions -- independent of any model.
_SPEC_CHECK_CASES: Final[tuple[tuple[int, int], ...]] = (
    (32, 32),
    (31, 32),
    (30, 32),
    (29, 32),
    (16, 32),
    (2, 32),
    (0, 32),
    (61, 64),
    (120, 128),
    (128, 128),
)


@dataclass(frozen=True)
class AdequacyReferenceAuditConfig:
    """Frozen protocol for the D2-005 SHIFT seed-24 adequacy/false-plastic audit."""

    sealed_seed: int = 24
    development_seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    regate_sealed_seeds: tuple[int, ...] = DEFAULT_REGATE_SEEDS
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operation: str = _TARGET_OPERATION
    support_examples: int = 128
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    adequacy_exact_match_threshold: float = 0.95
    confidence_level: float = 0.95
    initial_support: int = 32
    support_increment: int = 32
    max_support: int = 128
    reference_examples: int = 1024
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.sealed_seed not in self.regate_sealed_seeds:
            raise ValueError("sealed_seed must belong to the designated regate_sealed_seeds partition")
        if set(self.regate_sealed_seeds) & SEALED_GATE_SEEDS:
            raise ValueError("regate sealed seeds must stay disjoint from the original B-C005 sealed partition")
        if set(self.development_seeds) & (SEALED_GATE_SEEDS | set(self.regate_sealed_seeds)):
            raise ValueError("development seeds must be disjoint from both sealed partitions")
        if len(self.development_seeds) != len(self.regate_sealed_seeds):
            raise ValueError("development and regate sealed partitions must have equal length")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if self.support_examples < self.max_support:
            raise ValueError("support_examples must be >= max_support")
        if self.query_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")
        if self.reference_examples < 1:
            raise ValueError("reference_examples must be positive")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["levels"] = [level.value for level in self.levels]
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


# ---------------------------------------------------------------------------
# Seed derivation -- reproduces `hard_negative_repair_gate.run_hard_negative_regate`
# exactly so the frozen cell being audited is bit-for-bit the sealed B-C005G cell.
# ---------------------------------------------------------------------------


def _development_seed_for(config: AdequacyReferenceAuditConfig, sealed_seed: int) -> int:
    index = config.regate_sealed_seeds.index(sealed_seed)
    return config.development_seeds[index]


def _support_seed(sealed_seed: int, bank_size: int, target_id: int) -> int:
    return sealed_seed * 1_000_000 + bank_size * 100 + target_id


def _query_seed(sealed_seed: int, bank_size: int, target_id: int) -> int:
    return sealed_seed * 1_000_000 + bank_size * 100 + target_id + 1


def _reference_seed(sealed_seed: int, bank_size: int, target_id: int) -> int:
    """A large, independent, evaluation-only sample seed.

    Constructed with a different order of magnitude and offset than
    `_support_seed`/`_query_seed` so it cannot collide with either batch the
    sealed gate itself drew; never fed into support or query generation.
    """
    return sealed_seed * 10_000_000 + bank_size * 1_000 + target_id * 10 + 3


def _base_config(
    config: AdequacyReferenceAuditConfig, model_seed: int, bank_size: int
) -> HardNegativeBenchmarkConfig:
    return HardNegativeBenchmarkConfig(
        seeds=(model_seed,),
        bank_sizes=(bank_size,),
        levels=config.levels,
        target_operations=(config.target_operation,),
        num_eval_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        top_k=config.top_k,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _repair_config(
    config: AdequacyReferenceAuditConfig, development_seed: int, bank_size: int
) -> RetrievalRepairConfig:
    return RetrievalRepairConfig(
        seeds=(development_seed,),
        bank_sizes=(bank_size,),
        levels=config.levels,
        target_operations=(config.target_operation,),
        support_examples=config.initial_support,
        query_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        arg_lambda=config.arg_lambda,
        deterministic_algorithms=config.deterministic_algorithms,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _shared_adequacy_config(config: AdequacyReferenceAuditConfig) -> AdequacyRepairConfig:
    """Scalar knobs consumed by `evaluate_adequacy_policy_cell`; its own
    seeds/bank_sizes fields are unused by that function's per-cell entry point."""
    return AdequacyRepairConfig(
        support_examples=config.support_examples,
        query_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        arg_lambda=config.arg_lambda,
        adequacy_exact_match_threshold=config.adequacy_exact_match_threshold,
        confidence_level=config.confidence_level,
        initial_support=config.initial_support,
        support_increment=config.support_increment,
        max_support=config.max_support,
        deterministic_algorithms=config.deterministic_algorithms,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _verifier_config(config: AdequacyReferenceAuditConfig) -> SequentialVerifierConfig:
    return SequentialVerifierConfig(
        adequacy_threshold=config.adequacy_exact_match_threshold,
        confidence_level=config.confidence_level,
        initial_support=config.initial_support,
        support_increment=config.support_increment,
        max_support=config.max_support,
        interval_method="wilson",
        decision_rule="sequential_confidence",
    )


# ---------------------------------------------------------------------------
# Frozen reconstruction, mirroring `run_hard_negative_regate` per bank size.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrozenCell:
    core: Any
    bank: Any
    router: Any
    argument_scorer: Any
    operation_by_id: dict[int, str]
    candidate_ids: list[int]
    target_id: int
    model_seed: int
    development_seed: int


def _reconstruct_frozen_cell(config: AdequacyReferenceAuditConfig, *, bank_size: int) -> _FrozenCell:
    development_seed = _development_seed_for(config, config.sealed_seed)
    model_seed = config.sealed_seed % 5
    set_seed(config.sealed_seed, deterministic_algorithms=config.deterministic_algorithms)

    base_config = _base_config(config, model_seed, bank_size)
    core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)
    bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
        core, base_bank, base_router, op_to_id, bank_size, seed=config.sealed_seed
    )
    operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
    operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

    dev_train = _training_examples(
        development_seed, op_to_id, config.router_train_examples, regate_recipe=True
    )
    router, argument_scorer = train_repaired_router_and_scorer(
        core,
        router,
        candidate_ids,
        operation_by_id,
        dev_train,
        config=_repair_config(config, development_seed, bank_size),
        condition="R2",
        device=core.device,
    )
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    if argument_scorer is not None:
        argument_scorer.eval()
        for parameter in argument_scorer.parameters():
            parameter.requires_grad_(False)

    target_id = op_to_id[config.target_operation]
    return _FrozenCell(
        core=core,
        bank=bank,
        router=router,
        argument_scorer=argument_scorer,
        operation_by_id=operation_by_id,
        candidate_ids=candidate_ids,
        target_id=target_id,
        model_seed=model_seed,
        development_seed=development_seed,
    )


def _correct_candidate(router: Any, target_id: int) -> HardNegativeCandidate:
    """The installed SHIFT candidate, identical across every hard-negative
    level: `build_hard_negative_candidates` only ever changes which *other*
    candidate is added or overridden, never this one's identity or arguments
    (confirmed structurally in `test_correct_candidate_identity_is_level_invariant`)."""
    return HardNegativeCandidate(
        candidate_id=f"primitive:{target_id}",
        execute_primitive_id=target_id,
        score_key=router.key_parameter(target_id).detach().clone(),
        provenance="resident_primitive_key",
    )


# ---------------------------------------------------------------------------
# D2-005.1 -- per-episode reconstruction: verifier trace, rank diagnostics,
# official per-level reproduction.
# ---------------------------------------------------------------------------


def _run_verifier_trace(
    *,
    core: Any,
    bank: Any,
    operation_by_id: dict[int, str],
    target_id: int,
    candidate: HardNegativeCandidate,
    support_examples: Sequence[Any],
    config: AdequacyReferenceAuditConfig,
) -> CandidateVerificationTrace:
    """Run the installed `SequentialAdequacyVerifier` on exactly the correct
    SHIFT candidate over the frozen sealed support draw -- the same eval_chunk
    logic `evaluate_adequacy_policy_cell` uses, exposed here as a full
    step-by-step trace instead of only the final aggregate decision."""
    verifier = SequentialAdequacyVerifier(_verifier_config(config))

    def eval_chunk(start_idx: int, end_idx: int) -> int:
        chunk = support_examples[start_idx:end_idx]
        predictions = _execute_selected_candidates(
            core, bank, operation_by_id, chunk, [candidate] * len(chunk)
        )
        return sum(
            pred == ex.target_tokens for pred, ex in zip(predictions, chunk, strict=True)
        )

    return verifier.verify_candidate_sequentially(
        candidate_id=candidate.candidate_id,
        execute_primitive_id=target_id,
        support_eval_fn=eval_chunk,
        total_available_support=len(support_examples),
    )


def _symmetric_rule_classification(
    trace: CandidateVerificationTrace, threshold: float
) -> dict[str, Any]:
    """D2-005.4 diagnostic-only counterfactual: Wilson lower >= threshold ->
    accept, upper < threshold -> reject, otherwise uncertain -- applied at the
    same sequence of support sizes the installed rule used. Never deployed;
    reported purely to test whether the installed rule's early-accept /
    early-reject asymmetry (accept needs only the point estimate, reject
    needs the full upper-bound exclusion) drove this specific decision.
    """
    for step in trace.steps:
        ci = step.confidence_interval
        if ci.lower >= threshold:
            return {
                "decision": AdequacyDecision.ACCEPT.value,
                "support_consumed": step.support_size,
                "step_index": step.step_index,
            }
        if ci.upper < threshold:
            return {
                "decision": AdequacyDecision.REJECT.value,
                "support_consumed": step.support_size,
                "step_index": step.step_index,
            }
    last = trace.steps[-1]
    return {
        "decision": "UNCERTAIN",
        "support_consumed": last.support_size,
        "step_index": last.step_index,
    }


def _official_cell(
    *,
    core: Any,
    bank: Any,
    router: Any,
    argument_scorer: Any,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_id: int,
    level: HardNegativeLevel,
    support_examples: Sequence[Any],
    query_examples: Sequence[Any],
    config: AdequacyReferenceAuditConfig,
    adequacy_config: AdequacyRepairConfig,
) -> dict[str, Any]:
    """Reproduce the exact `evaluate_adequacy_policy_cell` result for one
    level, for a direct historical-reproducibility check against B-C005G's
    own recorded `runs/phase_b_b2_regate/metrics.jsonl`."""
    diagnostic = evaluate_adequacy_policy_cell(
        core=core,
        bank=bank,
        router=router,
        argument_scorer=argument_scorer,
        candidate_ids=candidate_ids,
        operation_by_id=operation_by_id,
        target_operation=config.target_operation,
        target_id=target_id,
        level=level,
        support_examples=support_examples,
        query_examples=query_examples,
        seed=config.sealed_seed,
        policy="sequential",
        config=adequacy_config,
    )
    return diagnostic.to_dict()


def _level_rank_diagnostics(
    *,
    core: Any,
    router: Any,
    argument_scorer: Any,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_id: int,
    level: HardNegativeLevel,
    query_examples: Sequence[Any],
    config: AdequacyReferenceAuditConfig,
) -> dict[str, Any]:
    """D2-005.1's per-episode ``candidate rank`` / ``family rank`` on the
    query set for this level, using the frozen R2 mechanism (never modified
    here)."""
    keys = {pid: router.key_parameter(pid).detach().clone() for pid in candidate_ids}
    candidates, competitor = build_hard_negative_candidates(
        level=level,
        target_id=target_id,
        target_operation=config.target_operation,
        candidate_ids=candidate_ids,
        keys_by_id=keys,
        operation_by_id=operation_by_id,
        seed=config.sealed_seed,
    )
    correct_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate is not competitor and candidate.execute_primitive_id == target_id
    )

    _family_scores, family_ordering = _rank_candidates_factorized(
        core, router, candidates, query_examples, None, operation_by_id, 0.0
    )
    combined_scores, combined_ordering = _rank_candidates_factorized(
        core, router, candidates, query_examples, argument_scorer, operation_by_id, config.arg_lambda
    )
    family_ranks = (family_ordering == correct_index).nonzero(as_tuple=False)[:, 1] + 1
    candidate_ranks = (combined_ordering == correct_index).nonzero(as_tuple=False)[:, 1] + 1
    competitor_idx = _competitor_index(candidates, competitor)

    return {
        "level": level.value,
        "n_query": len(query_examples),
        "mean_family_rank": float(family_ranks.float().mean().item()),
        "mean_candidate_rank": float(candidate_ranks.float().mean().item()),
        "family_top1": float((family_ordering[:, 0] == correct_index).to(torch.float32).mean().item()),
        "primitive_call_top1": float((combined_ordering[:, 0] == correct_index).to(torch.float32).mean().item()),
        "mean_score_margin": float(
            (combined_scores[:, correct_index] - combined_scores[:, competitor_idx]).mean().item()
        ),
    }


def _competitor_index(candidates: Sequence[HardNegativeCandidate], competitor: HardNegativeCandidate) -> int:
    return next(index for index, candidate in enumerate(candidates) if candidate is competitor)


# ---------------------------------------------------------------------------
# D2-005.2 -- large-sample reference adequacy (evaluation-only).
# ---------------------------------------------------------------------------


def _reference_adequacy(
    *,
    core: Any,
    bank: Any,
    operation_by_id: dict[int, str],
    target_id: int,
    candidate: HardNegativeCandidate,
    sealed_seed: int,
    bank_size: int,
    config: AdequacyReferenceAuditConfig,
) -> dict[str, Any]:
    examples = generate_benchmark_examples(
        _reference_seed(sealed_seed, bank_size, target_id),
        config.reference_examples,
        operation=config.target_operation,
        split="test",
    )
    predictions = _execute_selected_candidates(
        core, bank, operation_by_id, examples, [candidate] * len(examples)
    )
    correct = sum(
        pred == ex.target_tokens for pred, ex in zip(predictions, examples, strict=True)
    )
    wilson = wilson_score_interval(correct, len(examples), confidence_level=config.confidence_level)
    clopper = clopper_pearson_interval(correct, len(examples), confidence_level=config.confidence_level)
    reference_em = correct / len(examples)
    return {
        "n_reference_examples": len(examples),
        "reference_correct": correct,
        "reference_em": reference_em,
        "wilson_ci": wilson.to_dict(),
        "clopper_pearson_ci": clopper.to_dict(),
        "reference_adequate": reference_em >= config.adequacy_exact_match_threshold,
    }


def _independent_query_em(
    *,
    core: Any,
    bank: Any,
    operation_by_id: dict[int, str],
    candidate: HardNegativeCandidate,
    query_examples: Sequence[Any],
) -> float:
    """The correct candidate's own accuracy on the sealed query set, computed
    directly (bypassing routing) -- distinct from `closed_loop_exact_match`,
    which reflects whatever the pipeline actually routed to (0.0 whenever
    every query fell back to plastic)."""
    predictions = _execute_selected_candidates(
        core, bank, operation_by_id, query_examples, [candidate] * len(query_examples)
    )
    correct = sum(
        pred == ex.target_tokens for pred, ex in zip(predictions, query_examples, strict=True)
    )
    return correct / len(query_examples)


# ---------------------------------------------------------------------------
# D2-005.3 -- reclassification.
# ---------------------------------------------------------------------------


def _bias_direction(
    trace: CandidateVerificationTrace, symmetric: dict[str, Any], max_support: int
) -> str:
    """Classify how the installed asymmetric rule's decision diverges from the
    diagnostic-only symmetric counterfactual, distinguishing the
    safety-relevant direction from a benign one.

    `ACCEPT` and symmetric `REJECT` cannot co-occur on the same step (a
    Wilson upper bound below threshold implies the point estimate is too),
    so the only two divergences that occur in practice are:

    - `PREMATURE_ACCEPT`: the installed rule accepted as soon as the point
      estimate crossed the threshold, before the confidence interval itself
      confirmed adequacy (lower bound still below threshold) -- this is the
      direction with real safety consequence, since it can accept a
      genuinely inadequate candidate on a lucky draw.
    - `FORCED_REJECT_AT_BUDGET_EXHAUSTION`: the installed rule's
      forced-decision fallback rejected at `max_support` while the point
      estimate was still in the symmetric rule's undecided zone -- a
      deliberate, conservative tie-break (defaulting to plastic rather than
      reuse when uncertain), not a defect.
    """
    installed = trace.final_decision.value
    if installed == symmetric["decision"]:
        return "NONE"
    if installed == AdequacyDecision.ACCEPT.value:
        return "PREMATURE_ACCEPT"
    if installed == AdequacyDecision.REJECT.value and trace.final_support_consumed >= max_support:
        return "FORCED_REJECT_AT_BUDGET_EXHAUSTION"
    return "OTHER_DIVERGENCE"  # pragma: no cover - not reachable given the rule's structure


def classify_plastic_cell(*, runtime_decision: str, reference_adequate: bool) -> str:
    """D2-005.3's three-way retrospective classification, plus the benign
    baseline (`CORRECTLY_ACCEPTED`) for cells where the verifier accepted a
    genuinely adequate candidate."""
    if runtime_decision == AdequacyDecision.REJECT.value:
        return "TRUE_FALSE_PLASTIC" if reference_adequate else "FUNCTIONALLY_JUSTIFIED_PLASTIC"
    if runtime_decision == AdequacyDecision.ACCEPT.value:
        return "CORRECTLY_ACCEPTED" if reference_adequate else "UNSAFE_REUSE"
    return "UNCERTAIN_NO_FORCED_DECISION"


# ---------------------------------------------------------------------------
# D2-005.4 -- spec conformance, independent of any model or GPU.
# ---------------------------------------------------------------------------


def verify_installed_rule_matches_spec(config: AdequacyReferenceAuditConfig) -> dict[str, Any]:
    """Confirm `SequentialAdequacyVerifier` implements exactly the documented
    rule (early accept if EM >= threshold; early reject if Wilson upper bound
    < threshold; otherwise gather more evidence up to `max_support`, then
    force a decision by the empirical EM) for a spread of synthetic
    (successes, trials) pairs."""
    verifier = SequentialAdequacyVerifier(_verifier_config(config))
    mismatches: list[dict[str, Any]] = []
    for successes, trials in _SPEC_CHECK_CASES:
        decision, interval = verifier.evaluate_step(successes, trials)
        empirical = successes / trials
        if trials >= config.max_support:
            expected = (
                AdequacyDecision.ACCEPT
                if empirical >= config.adequacy_exact_match_threshold
                else AdequacyDecision.REJECT
            )
        elif empirical >= config.adequacy_exact_match_threshold:
            expected = AdequacyDecision.ACCEPT
        elif interval.upper < config.adequacy_exact_match_threshold:
            expected = AdequacyDecision.REJECT
        else:
            expected = AdequacyDecision.UNCERTAIN
        if decision != expected:
            mismatches.append(
                {
                    "successes": successes,
                    "trials": trials,
                    "expected": expected.value,
                    "actual": decision.value,
                }
            )
    return {
        "cases_checked": len(_SPEC_CHECK_CASES),
        "mismatches": mismatches,
        "implementation_matches_spec": not mismatches,
    }


# ---------------------------------------------------------------------------
# Aggregation and top-level run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankSizeAudit:
    """One (sealed_seed=24, bank_size, SHIFT) cell's full D2-005 audit."""

    bank_size: int
    model_seed: int
    development_seed: int
    target_id: int
    official_cells: dict[str, Any]
    rank_diagnostics: dict[str, Any]
    verifier_trace: dict[str, Any]
    symmetric_rule: dict[str, Any]
    sequential_rule_bias_direction: str
    reference_adequacy: dict[str, Any]
    independent_query_em: float
    plastic_classification: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _overall_classification(
    bank_audits: Sequence[BankSizeAudit], spec_check: dict[str, Any]
) -> dict[str, Any]:
    labels: set[str] = set()
    if any(audit.plastic_classification == "TRUE_FALSE_PLASTIC" for audit in bank_audits):
        labels.add("FINITE_SUPPORT_VARIANCE")
        labels.add("METRIC_MISCLASSIFICATION")
    if any(audit.plastic_classification == "FUNCTIONALLY_JUSTIFIED_PLASTIC" for audit in bank_audits):
        labels.add("TRUE_PRIMITIVE_INADEQUACY")
    # Only the premature-accept direction is safety-relevant (it is what lets
    # a genuinely inadequate candidate through); the benign forced-reject
    # divergence at budget exhaustion is a deliberate conservative tie-break,
    # not evidence of a biased rule, and is reported but not labeled here.
    if any(audit.sequential_rule_bias_direction == "PREMATURE_ACCEPT" for audit in bank_audits):
        labels.add("SEQUENTIAL_RULE_BIAS")
    if not spec_check["implementation_matches_spec"]:
        labels.add("IMPLEMENTATION_BUG")
    if any(audit.plastic_classification == "UNSAFE_REUSE" for audit in bank_audits):
        # Not one of the seven D2-005 goal labels; flagged separately because
        # it is a distinct and more serious safety finding than any of them.
        labels.add("UNSAFE_REUSE_DETECTED")
    if not labels:
        labels.add("UNRESOLVED")
    return {
        "labels": sorted(labels),
        "n_bank_size_cells": len(bank_audits),
        "n_true_false_plastic": sum(1 for a in bank_audits if a.plastic_classification == "TRUE_FALSE_PLASTIC"),
        "n_functionally_justified_plastic": sum(
            1 for a in bank_audits if a.plastic_classification == "FUNCTIONALLY_JUSTIFIED_PLASTIC"
        ),
        "n_unsafe_reuse": sum(1 for a in bank_audits if a.plastic_classification == "UNSAFE_REUSE"),
        "n_correctly_accepted": sum(1 for a in bank_audits if a.plastic_classification == "CORRECTLY_ACCEPTED"),
    }


def run_adequacy_reference_audit(config: AdequacyReferenceAuditConfig) -> dict[str, Any]:
    """Run the B-C005D2-005 SHIFT seed-24 adequacy / false-plastic audit."""
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    spec_check = verify_installed_rule_matches_spec(config)
    adequacy_config = _shared_adequacy_config(config)

    bank_audits: list[BankSizeAudit] = []
    for bank_size in config.bank_sizes:
        cell = _reconstruct_frozen_cell(config, bank_size=bank_size)
        support = generate_benchmark_examples(
            _support_seed(config.sealed_seed, bank_size, cell.target_id),
            config.support_examples,
            operation=config.target_operation,
            split="test",
        )
        query = generate_benchmark_examples(
            _query_seed(config.sealed_seed, bank_size, cell.target_id),
            config.query_examples,
            operation=config.target_operation,
            split="test",
        )
        candidate = _correct_candidate(cell.router, cell.target_id)

        official_cells = {
            level.value: _official_cell(
                core=cell.core,
                bank=cell.bank,
                router=cell.router,
                argument_scorer=cell.argument_scorer,
                candidate_ids=cell.candidate_ids,
                operation_by_id=cell.operation_by_id,
                target_id=cell.target_id,
                level=level,
                support_examples=support,
                query_examples=query,
                config=config,
                adequacy_config=adequacy_config,
            )
            for level in config.levels
        }
        rank_diagnostics = {
            level.value: _level_rank_diagnostics(
                core=cell.core,
                router=cell.router,
                argument_scorer=cell.argument_scorer,
                candidate_ids=cell.candidate_ids,
                operation_by_id=cell.operation_by_id,
                target_id=cell.target_id,
                level=level,
                query_examples=query,
                config=config,
            )
            for level in config.levels
        }
        trace = _run_verifier_trace(
            core=cell.core,
            bank=cell.bank,
            operation_by_id=cell.operation_by_id,
            target_id=cell.target_id,
            candidate=candidate,
            support_examples=support,
            config=config,
        )
        symmetric = _symmetric_rule_classification(trace, config.adequacy_exact_match_threshold)
        bias_direction = _bias_direction(trace, symmetric, config.max_support)

        reference = _reference_adequacy(
            core=cell.core,
            bank=cell.bank,
            operation_by_id=cell.operation_by_id,
            target_id=cell.target_id,
            candidate=candidate,
            sealed_seed=config.sealed_seed,
            bank_size=bank_size,
            config=config,
        )
        independent_em = _independent_query_em(
            core=cell.core,
            bank=cell.bank,
            operation_by_id=cell.operation_by_id,
            candidate=candidate,
            query_examples=query,
        )
        classification = classify_plastic_cell(
            runtime_decision=trace.final_decision.value,
            reference_adequate=reference["reference_adequate"],
        )

        bank_audits.append(
            BankSizeAudit(
                bank_size=bank_size,
                model_seed=cell.model_seed,
                development_seed=cell.development_seed,
                target_id=cell.target_id,
                official_cells=official_cells,
                rank_diagnostics=rank_diagnostics,
                verifier_trace=trace.to_dict(),
                symmetric_rule=symmetric,
                sequential_rule_bias_direction=bias_direction,
                reference_adequacy=reference,
                independent_query_em=independent_em,
                plastic_classification=classification,
            )
        )

    overall = _overall_classification(bank_audits, spec_check)

    report: dict[str, Any] = {
        "task_id": "B-C005D2-005",
        "config": config.to_dict(),
        "spec_verification": spec_check,
        "bank_size_audits": [audit.to_dict() for audit in bank_audits],
        "overall_classification": overall,
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-005",
            "sealed_seed": config.sealed_seed,
            "regate_sealed_seeds": list(config.regate_sealed_seeds),
            "development_seeds": list(config.development_seeds),
            "sealed_seed_used_only_for_readout": True,
            "no_sealed_gradient_updates": True,
        }
        (out / "adequacy_reference_audit_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "adequacy_reference_audit_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        (out / "adequacy_reference_audit_system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "shift_seed24_adequacy_audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (out / "reference_adequacy_summary.json").write_text(
            json.dumps(
                {
                    audit.bank_size: audit.reference_adequacy
                    for audit in bank_audits
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report
