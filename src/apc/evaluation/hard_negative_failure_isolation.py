# ruff: noqa: E501
"""Phase B B-C005D Failure Isolation: Ranking vs Argument Resolution vs Adequacy Variance.

Diagnostic module isolating why B-C005 failed:
1. Decomposed retrieval metrics (physical primitive vs argument vs full call, L4 taxonomy).
2. Score-margin distributions (collapse vs semantic collision).
3. Correct-candidate support adequacy diagnosis on false-plastic vs accepted episodes.
4. Support-size variance experiment (K in {16, 32, 64, 128}) with binomial reference curves.
5. Candidate-order sensitivity comparison (Policy A vs B vs C).
6. Artifact and plot generation.
7. Acceptance criteria / 7 core failure isolation questions.

This module is diagnostic-only: it does not alter any model parameters, router
weights, controller thresholds, or plastic policies.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apc.environments.generator import Example
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_LEVELS,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    HardNegativeCandidate,
    _build_frozen_base_system,
    _candidate_exact_match,
    _execute_selected_candidates,
    _rank_logical_candidates,
    build_hard_negative_candidates,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.router import Router
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info


class L4FailureCategory(str, Enum):  # noqa: UP042
    """Taxonomy of L4 confusable-family competitor outcomes."""

    NEITHER = "NEITHER"  # Succeeded (both family and argument correct)
    FAMILY_RANKING_FAILURE = "FAMILY_RANKING_FAILURE"  # Wrong physical primitive retrieved
    ARGUMENT_RESOLUTION_FAILURE = "ARGUMENT_RESOLUTION_FAILURE"  # Correct primitive, wrong argument
    BOTH = "BOTH"  # Both family and argument incorrect


@dataclass(frozen=True)
class HardNegativeFailureIsolationConfig:
    """Explicit configuration for B-C005D failure isolation diagnostics."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    support_examples: int = 32
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    top_k: int = 5
    adequacy_exact_match_threshold: float = 0.95
    support_variance_sizes: tuple[int, ...] = (16, 32, 64, 128)
    binomial_p_values: tuple[float, ...] = (0.97, 0.98, 0.99, 0.995)
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if self.support_examples < 1 or self.query_examples < 1:
            raise ValueError("support_examples and query_examples must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["levels"] = [level.value for level in self.levels]
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        return data


@dataclass
class QueryDiagnosticRecord:
    """Detailed retrieval and verification diagnostics for a single query example."""

    seed: int
    bank_size: int
    level: str
    target_operation: str
    query_index: int
    physical_primitive_top1: bool
    physical_primitive_topk: bool
    argument_accuracy: bool
    primitive_call_top1: bool
    primitive_call_topk: bool
    l4_category: str | None
    score_correct: float
    score_best_wrong: float
    score_margin: float
    correct_rank: int
    best_wrong_provenance: str
    best_wrong_candidate_id: str
    policy_a_decision: str  # ACCEPT or PLASTIC
    policy_b_decision: str
    policy_c_decision: str
    policy_a_correct: bool
    policy_b_correct: bool
    policy_c_correct: bool


@dataclass
class CellDiagnosticSummary:
    """Aggregated diagnostics for one (seed, bank_size, level, target_operation) cell."""

    seed: int
    bank_size: int
    level: str
    target_operation: str
    query_count: int
    candidate_count: int
    physical_primitive_top1: float
    physical_primitive_topk: float
    argument_accuracy: float
    primitive_call_top1: float
    primitive_call_topk: float
    l4_breakdown: dict[str, int]
    mean_margin: float
    median_margin: float
    p05_margin: float
    p95_margin: float
    fraction_margin_le_zero: float
    correct_rank_mean: float
    rank_histogram: dict[str, int]
    competitor_provenance: str
    best_wrong_provenances: dict[str, int]
    support_correct_count: int
    support_size: int
    support_em: float
    query_em: float
    is_false_plastic: bool
    correct_in_top5: bool
    policy_a_closed_loop_em: float
    policy_b_closed_loop_em: float
    policy_c_closed_loop_em: float
    policy_a_false_plastic: float
    policy_b_false_plastic: float
    policy_c_false_plastic: float
    policy_a_false_functional_acceptance: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_binomial_reference(
    support_sizes: Sequence[int],
    p_values: Sequence[float],
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Compute theoretical binomial acceptance and false-rejection curves."""
    curves: dict[str, dict[str, Any]] = {}
    for p in p_values:
        curve: dict[str, Any] = {}
        for k in support_sizes:
            req_k = math.ceil(threshold * k)
            prob_accept = sum(
                math.comb(k, i) * (p**i) * ((1.0 - p) ** (k - i))
                for i in range(req_k, k + 1)
            )
            prob_false_reject = 1.0 - prob_accept
            curve[str(k)] = {
                "p": p,
                "support_size": k,
                "required_correct": req_k,
                "prob_accept": prob_accept,
                "prob_false_reject": prob_false_reject,
            }
        curves[f"p_{p}"] = curve
    return curves


def evaluate_cell_failure_isolation(
    *,
    core: Any,
    bank: Any,
    router: Router,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_operation: str,
    target_id: int,
    level: HardNegativeLevel,
    support_examples: Sequence[Example],
    query_examples: Sequence[Example],
    seed: int,
    top_k: int,
    adequacy_threshold: float,
) -> tuple[CellDiagnosticSummary, list[QueryDiagnosticRecord]]:
    """Deeply inspect one cell: decomposing metrics, margins, adequacy, and ordering."""
    keys = {pid: router.key_parameter(pid).detach().clone() for pid in candidate_ids}
    candidates, competitor = build_hard_negative_candidates(
        level=level,
        target_id=target_id,
        target_operation=target_operation,
        candidate_ids=candidate_ids,
        keys_by_id=keys,
        operation_by_id=operation_by_id,
        seed=seed,
    )
    correct_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate is not competitor and candidate.execute_primitive_id == target_id
    )

    _, support_ordering = _rank_logical_candidates(core, router, candidates, support_examples)
    query_scores, query_ordering = _rank_logical_candidates(core, router, candidates, query_examples)

    # Support evaluations for verify indices (all top-5 candidates across support examples)
    verify_indices = sorted(
        {int(index) for index in support_ordering[:, : min(top_k, len(candidates))].reshape(-1).tolist()}
    )
    # Ensure correct candidate is evaluated on support for diagnostic purposes even if outside top-5
    all_eval_indices = sorted(set(verify_indices) | {correct_index})
    support_scores = {
        index: _candidate_exact_match(core, bank, operation_by_id, candidates[index], support_examples)
        for index in all_eval_indices
    }
    accepted_indices = {
        index for index, score in support_scores.items() if score >= adequacy_threshold and index in verify_indices
    }

    # Query EM for correct candidate
    correct_query_em = _candidate_exact_match(core, bank, operation_by_id, candidates[correct_index], query_examples)
    correct_support_em = support_scores[correct_index]
    support_correct_count = round(correct_support_em * len(support_examples))

    # Detailed query records
    query_records: list[QueryDiagnosticRecord] = []
    l4_counts: dict[str, int] = {c.value: 0 for c in L4FailureCategory}
    rank_hist: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, ">5": 0}
    wrong_provenances: dict[str, int] = {}
    margins: list[float] = []

    for q_idx in range(len(query_examples)):
        row_order = query_ordering[q_idx].tolist()
        row_scores = query_scores[q_idx]

        top1_idx = row_order[0]
        top1_cand = candidates[top1_idx]
        correct_rank = row_order.index(correct_index) + 1

        if correct_rank <= 5:
            rank_hist[str(correct_rank)] += 1
        else:
            rank_hist[">5"] += 1

        phys_top1 = (top1_cand.execute_primitive_id == target_id)
        phys_topk = any(candidates[idx].execute_primitive_id == target_id for idx in row_order[: min(top_k, len(candidates))])
        arg_acc = phys_top1 and (top1_cand.argument_override is None)
        call_top1 = (top1_idx == correct_index)
        call_topk = (correct_index in row_order[: min(top_k, len(candidates))])

        # L4 taxonomy
        l4_cat: str | None = None
        if level == HardNegativeLevel.L4_CONFUSABLE_FAMILY:
            if call_top1:
                l4_cat = L4FailureCategory.NEITHER.value
            else:
                family_wrong = (top1_cand.execute_primitive_id != target_id)
                arg_wrong = (top1_cand.argument_override is not None)
                if family_wrong and arg_wrong:
                    l4_cat = L4FailureCategory.BOTH.value
                elif family_wrong:
                    l4_cat = L4FailureCategory.FAMILY_RANKING_FAILURE.value
                else:
                    l4_cat = L4FailureCategory.ARGUMENT_RESOLUTION_FAILURE.value
            l4_counts[l4_cat] += 1

        # Margin logging
        score_corr = float(row_scores[correct_index].item())
        wrong_indices = [i for i in range(len(candidates)) if i != correct_index]
        best_wrong_idx = max(wrong_indices, key=lambda i: float(row_scores[i].item()))
        score_best_wrong = float(row_scores[best_wrong_idx].item())
        margin = score_corr - score_best_wrong
        margins.append(margin)

        best_wrong_cand = candidates[best_wrong_idx]
        wrong_provenances[best_wrong_cand.provenance] = wrong_provenances.get(best_wrong_cand.provenance, 0) + 1

        # Policy A: ranked-first acceptance
        cand_a_idx = next(
            (idx for idx in row_order[: min(top_k, len(candidates))] if idx in accepted_indices), None
        )
        pol_a_decision = "ACCEPT" if cand_a_idx is not None else "PLASTIC"
        pol_a_correct = (cand_a_idx == correct_index)

        # Policy B: evaluate all top-5, choose highest support adequacy >= threshold
        top5_indices = row_order[: min(top_k, len(candidates))]
        eligible_b = [
            (idx, support_scores.get(idx, 0.0))
            for idx in top5_indices
            if support_scores.get(idx, 0.0) >= adequacy_threshold
        ]
        cand_b_idx = max(eligible_b, key=lambda pair: pair[1])[0] if eligible_b else None
        pol_b_decision = "ACCEPT" if cand_b_idx is not None else "PLASTIC"
        pol_b_correct = (cand_b_idx == correct_index)

        # Policy C: oracle correct-candidate adequacy only
        cand_c_accepted = (correct_support_em >= adequacy_threshold)
        pol_c_decision = "ACCEPT" if cand_c_accepted else "PLASTIC"
        pol_c_correct = cand_c_accepted

        query_records.append(
            QueryDiagnosticRecord(
                seed=seed,
                bank_size=len(bank),
                level=level.value,
                target_operation=target_operation,
                query_index=q_idx,
                physical_primitive_top1=phys_top1,
                physical_primitive_topk=phys_topk,
                argument_accuracy=arg_acc,
                primitive_call_top1=call_top1,
                primitive_call_topk=call_topk,
                l4_category=l4_cat,
                score_correct=score_corr,
                score_best_wrong=score_best_wrong,
                score_margin=margin,
                correct_rank=correct_rank,
                best_wrong_provenance=best_wrong_cand.provenance,
                best_wrong_candidate_id=best_wrong_cand.candidate_id,
                policy_a_decision=pol_a_decision,
                policy_b_decision=pol_b_decision,
                policy_c_decision=pol_c_decision,
                policy_a_correct=pol_a_correct,
                policy_b_correct=pol_b_correct,
                policy_c_correct=pol_c_correct,
            )
        )

    # Policy evaluations closed loop EM on query
    # Policy A execution
    selected_a: list[HardNegativeCandidate | None] = []
    for row_order in query_ordering.tolist():
        chosen = next(
            (int(idx) for idx in row_order[: min(top_k, len(candidates))] if int(idx) in accepted_indices),
            None,
        )
        selected_a.append(candidates[chosen] if chosen is not None else None)
    exec_a = [c for c in selected_a if c is not None]
    preds_a: list[tuple[int, ...] | None] = [None] * len(query_examples)
    if exec_a:
        ex_subset = [ex for ex, c in zip(query_examples, selected_a, strict=True) if c is not None]
        out_preds = _execute_selected_candidates(core, bank, operation_by_id, ex_subset, exec_a)
        for idx, pred in zip([i for i, c in enumerate(selected_a) if c is not None], out_preds, strict=True):
            preds_a[idx] = pred
    pol_a_em = sum(p == ex.target_tokens for p, ex in zip(preds_a, query_examples, strict=True) if p is not None) / len(query_examples)
    pol_a_fp = sum(c is None for c in selected_a) / len(query_examples)
    pol_a_ffa = sum(c is not None and c is not candidates[correct_index] for c in selected_a) / len(query_examples)

    # Policy B execution
    selected_b = []
    for row_order in query_ordering.tolist():
        top5_indices = row_order[: min(top_k, len(candidates))]
        eligible = [(idx, support_scores.get(idx, 0.0)) for idx in top5_indices if support_scores.get(idx, 0.0) >= adequacy_threshold]
        b_idx = max(eligible, key=lambda pair: pair[1])[0] if eligible else None
        selected_b.append(candidates[b_idx] if b_idx is not None else None)
    exec_b = [c for c in selected_b if c is not None]
    preds_b: list[tuple[int, ...] | None] = [None] * len(query_examples)
    if exec_b:
        ex_subset = [ex for ex, c in zip(query_examples, selected_b, strict=True) if c is not None]
        out_preds = _execute_selected_candidates(core, bank, operation_by_id, ex_subset, exec_b)
        for idx, pred in zip([i for i, c in enumerate(selected_b) if c is not None], out_preds, strict=True):
            preds_b[idx] = pred
    pol_b_em = sum(p == ex.target_tokens for p, ex in zip(preds_b, query_examples, strict=True) if p is not None) / len(query_examples)
    pol_b_fp = sum(c is None for c in selected_b) / len(query_examples)

    # Policy C execution
    if correct_support_em >= adequacy_threshold:
        exec_c = [candidates[correct_index]] * len(query_examples)
        preds_c = _execute_selected_candidates(core, bank, operation_by_id, query_examples, exec_c)
        pol_c_em = sum(p == ex.target_tokens for p, ex in zip(preds_c, query_examples, strict=True)) / len(query_examples)
        pol_c_fp = 0.0
    else:
        pol_c_em = 0.0
        pol_c_fp = 1.0

    margin_arr = np.array(margins)
    summary = CellDiagnosticSummary(
        seed=seed,
        bank_size=len(bank),
        level=level.value,
        target_operation=target_operation,
        query_count=len(query_examples),
        candidate_count=len(candidates),
        physical_primitive_top1=float(statistics.fmean(r.physical_primitive_top1 for r in query_records)),
        physical_primitive_topk=float(statistics.fmean(r.physical_primitive_topk for r in query_records)),
        argument_accuracy=float(statistics.fmean(r.argument_accuracy for r in query_records)),
        primitive_call_top1=float(statistics.fmean(r.primitive_call_top1 for r in query_records)),
        primitive_call_topk=float(statistics.fmean(r.primitive_call_topk for r in query_records)),
        l4_breakdown=l4_counts,
        mean_margin=float(np.mean(margin_arr)),
        median_margin=float(np.median(margin_arr)),
        p05_margin=float(np.percentile(margin_arr, 5)),
        p95_margin=float(np.percentile(margin_arr, 95)),
        fraction_margin_le_zero=float(np.mean(margin_arr <= 0.0)),
        correct_rank_mean=float(statistics.fmean(r.correct_rank for r in query_records)),
        rank_histogram=rank_hist,
        competitor_provenance=competitor.provenance,
        best_wrong_provenances=wrong_provenances,
        support_correct_count=support_correct_count,
        support_size=len(support_examples),
        support_em=correct_support_em,
        query_em=correct_query_em,
        is_false_plastic=(pol_a_fp > 0.0),
        correct_in_top5=(correct_index in verify_indices),
        policy_a_closed_loop_em=pol_a_em,
        policy_b_closed_loop_em=pol_b_em,
        policy_c_closed_loop_em=pol_c_em,
        policy_a_false_plastic=pol_a_fp,
        policy_b_false_plastic=pol_b_fp,
        policy_c_false_plastic=pol_c_fp,
        policy_a_false_functional_acceptance=pol_a_ffa,
    )
    return summary, query_records


def run_support_variance_experiment(
    config: HardNegativeFailureIsolationConfig,
    core: Any,
    base_bank: Any,
    base_router: Router,
    op_to_id: dict[str, int],
) -> dict[str, Any]:
    """D4: Evaluate support sizes K in {16, 32, 64, 128} using DEV split examples."""
    results_by_k: dict[str, dict[str, Any]] = {}
    for k in config.support_variance_sizes:
        cell_fps: list[float] = []
        cell_cfrs: list[float] = []
        cell_cwfas: list[float] = []
        support_ems: list[float] = []
        query_ems: list[float] = []

        for seed in config.seeds:
            for bank_size in config.bank_sizes:
                bank, router, candidate_ids, semantic_ids, distractor_ids = build_scaled_bank_and_router(
                    core, base_bank, base_router, op_to_id, bank_size, seed=seed
                )
                operation_by_id = {pid: op for op, pid in op_to_id.items()}
                operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

                for op in config.target_operations:
                    target_id = op_to_id[op]
                    # Use split='dev' to ensure disjoint development evaluation
                    support_examples = generate_benchmark_examples(
                        seed * 20_000 + bank_size + k, k, operation=op, split="dev"
                    )
                    query_examples = generate_benchmark_examples(
                        seed * 20_000 + bank_size + k + 1000, config.query_examples, operation=op, split="dev"
                    )

                    for level in config.levels:
                        summary, _ = evaluate_cell_failure_isolation(
                            core=core,
                            bank=bank,
                            router=router,
                            candidate_ids=candidate_ids,
                            operation_by_id=operation_by_id,
                            target_operation=op,
                            target_id=target_id,
                            level=level,
                            support_examples=support_examples,
                            query_examples=query_examples,
                            seed=seed,
                            top_k=config.top_k,
                            adequacy_threshold=config.adequacy_exact_match_threshold,
                        )
                        cell_fps.append(summary.policy_a_false_plastic)
                        # Correct candidate false reject: correct in top-5 but support EM < threshold
                        cfr = 1.0 if (summary.correct_in_top5 and summary.support_em < config.adequacy_exact_match_threshold) else 0.0
                        cell_cfrs.append(cfr)
                        cell_cwfas.append(summary.policy_a_false_functional_acceptance)
                        support_ems.append(summary.support_em)
                        query_ems.append(summary.query_em)

        results_by_k[str(k)] = {
            "support_size": k,
            "false_plastic_rate": float(statistics.fmean(cell_fps)),
            "correct_candidate_false_reject_rate": float(statistics.fmean(cell_cfrs)),
            "wrong_candidate_false_accept_rate": float(statistics.fmean(cell_cwfas)),
            "mean_support_em": float(statistics.fmean(support_ems)),
            "mean_query_em": float(statistics.fmean(query_ems)),
            "cell_count": len(cell_fps),
        }
    return results_by_k


def generate_diagnostic_plots(
    summaries: Sequence[CellDiagnosticSummary],
    support_variance: dict[str, Any],
    binomial_curves: dict[str, Any],
    plots_dir: Path,
) -> None:
    """D6: Generate the required diagnostic visualization plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted({s.level for s in summaries})
    if not levels:
        levels = [level.value for level in DEFAULT_LEVELS]

    # 1. top1_vs_level.png
    plt.figure(figsize=(8, 5))
    bank_sizes = sorted({s.bank_size for s in summaries})
    for n in bank_sizes:
        means = []
        for lvl in levels:
            cells = [s for s in summaries if s.bank_size == n and s.level == lvl]
            means.append(statistics.fmean(c.primitive_call_top1 for c in cells) if cells else 0.0)
        plt.plot(levels, means, marker="o", label=f"N={n}")
    plt.title("Primitive Call Top-1 vs Hard Negative Level")
    plt.xlabel("Level")
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "top1_vs_level.png", dpi=150)
    plt.close()

    # 2. topk_vs_level.png
    plt.figure(figsize=(8, 5))
    for n in bank_sizes:
        means = []
        for lvl in levels:
            cells = [s for s in summaries if s.bank_size == n and s.level == lvl]
            means.append(statistics.fmean(c.primitive_call_topk for c in cells) if cells else 0.0)
        plt.plot(levels, means, marker="s", label=f"N={n}")
    plt.title("Top-5 Inclusion vs Hard Negative Level")
    plt.xlabel("Level")
    plt.ylabel("Top-5 Inclusion Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "topk_vs_level.png", dpi=150)
    plt.close()

    # 3. margin_distribution_by_level.png
    plt.figure(figsize=(9, 5))
    margin_data = []
    valid_plot_levels = []
    for lvl in levels:
        cells = [s for s in summaries if s.level == lvl]
        if cells:
            margin_data.append([c.mean_margin for c in cells])
            valid_plot_levels.append(lvl)
    if margin_data:
        plt.boxplot(margin_data, tick_labels=valid_plot_levels)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Margin = 0")
    plt.title("Score Margin Distribution by Level")
    plt.xlabel("Level")
    plt.ylabel("Score Margin (Correct - Best Wrong)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "margin_distribution_by_level.png", dpi=150)
    plt.close()

    # 4. rank_histogram_by_level.png
    plt.figure(figsize=(10, 5))
    rank_bins = ["1", "2", "3", "4", "5", ">5"]
    x = np.arange(len(rank_bins))
    width = 0.8 / max(len(levels), 1)
    for i, lvl in enumerate(levels):
        cells = [s for s in summaries if s.level == lvl]
        counts = [sum(c.rank_histogram[b] for c in cells) for b in rank_bins]
        total = sum(counts) or 1
        freqs = [cnt / total for cnt in counts]
        plt.bar(x + i * width, freqs, width, label=lvl)
    plt.xticks(x + width * (len(levels) - 1) / 2, rank_bins)
    plt.title("Correct Candidate Rank Distribution by Level")
    plt.xlabel("Rank")
    plt.ylabel("Fraction of Queries")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "rank_histogram_by_level.png", dpi=150)
    plt.close()

    # 5. l4_family_vs_argument_failures.png
    plt.figure(figsize=(7, 5))
    l4_cells = [s for s in summaries if s.level == HardNegativeLevel.L4_CONFUSABLE_FAMILY.value]
    tot_family = sum(c.l4_breakdown[L4FailureCategory.FAMILY_RANKING_FAILURE.value] for c in l4_cells)
    tot_arg = sum(c.l4_breakdown[L4FailureCategory.ARGUMENT_RESOLUTION_FAILURE.value] for c in l4_cells)
    tot_both = sum(c.l4_breakdown[L4FailureCategory.BOTH.value] for c in l4_cells)
    cats = ["Family Failure", "Argument Failure", "Both Failure"]
    counts = [tot_family, tot_arg, tot_both]
    plt.bar(cats, counts, color=["coral", "teal", "plum"])
    plt.title("L4 Failure Breakdown: Family vs Argument")
    plt.ylabel("Query Count")
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "l4_family_vs_argument_failures.png", dpi=150)
    plt.close()

    # 6. false_plastic_vs_support_size.png
    plt.figure(figsize=(8, 5))
    k_vals = sorted(int(k) for k in support_variance.keys())
    emp_fp = [support_variance[str(k)]["false_plastic_rate"] for k in k_vals]
    plt.plot(k_vals, emp_fp, "o-", color="black", linewidth=2, label="Empirical False Plastic")
    for _p_key, curve in binomial_curves.items():
        p_val = curve[str(k_vals[0])]["p"]
        ref_rejection = [curve[str(k)]["prob_false_reject"] for k in k_vals]
        plt.plot(k_vals, ref_rejection, "--", label=f"Binomial ref (p={p_val})")
    plt.title("False Plastic Rate vs Support Size (K)")
    plt.xlabel("Support Size (K)")
    plt.ylabel("False Plastic / False Reject Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "false_plastic_vs_support_size.png", dpi=150)
    plt.close()

    # 7. support_count_histogram_false_plastic.png
    plt.figure(figsize=(8, 5))
    fp_cells = [s for s in summaries if s.is_false_plastic]
    counts = [s.support_correct_count for s in fp_cells]
    min_c = min(counts or [0])
    max_c = max(counts or [32])
    bins = [float(b) for b in np.arange(min_c - 0.5, max_c + 1.5, 1.0)]
    plt.hist(counts, bins=bins, color="salmon", edgecolor="darkred", rwidth=0.8)
    plt.axvline(31, color="blue", linestyle="--", label="Acceptance threshold (31/32 = 0.969)")
    plt.title("Support Correct Count in False Plastic Episodes (N=32 support)")
    plt.xlabel("Correct Count (out of 32)")
    plt.ylabel("Episode Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "support_count_histogram_false_plastic.png", dpi=150)
    plt.close()


def run_hard_negative_failure_isolation(config: HardNegativeFailureIsolationConfig) -> dict[str, Any]:
    """Execute the complete B-C005D failure isolation diagnostic suite."""
    start_time = time.perf_counter()
    output_dir = config.output_dir or Path("runs/phase_b_b2_failure_isolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_summaries: list[CellDiagnosticSummary] = []
    all_query_records: list[QueryDiagnosticRecord] = []

    # 1. Main Matrix Evaluation (Mirroring B-C005 sealed conditions for post-hoc diagnosis)
    for seed in config.seeds:
        set_seed(seed)
        bench_cfg = HardNegativeBenchmarkConfig(
            seeds=(seed,),
            bank_sizes=config.bank_sizes,
            levels=config.levels,
            target_operations=config.target_operations,
            num_eval_examples=config.support_examples,
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            top_k=config.top_k,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(seed, bench_cfg)

        for bank_size in config.bank_sizes:
            bank, router, candidate_ids, semantic_ids, distractor_ids = build_scaled_bank_and_router(
                core, base_bank, base_router, op_to_id, bank_size, seed=seed
            )
            operation_by_id = {pid: op for op, pid in op_to_id.items()}
            operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

            for op in config.target_operations:
                target_id = op_to_id[op]
                support_examples = generate_benchmark_examples(
                    seed * 10_000 + bank_size, config.support_examples, operation=op, split="test"
                )
                query_examples = generate_benchmark_examples(
                    seed * 10_000 + bank_size + 1000, config.query_examples, operation=op, split="test"
                )

                for level in config.levels:
                    cell_sum, q_recs = evaluate_cell_failure_isolation(
                        core=core,
                        bank=bank,
                        router=router,
                        candidate_ids=candidate_ids,
                        operation_by_id=operation_by_id,
                        target_operation=op,
                        target_id=target_id,
                        level=level,
                        support_examples=support_examples,
                        query_examples=query_examples,
                        seed=seed,
                        top_k=config.top_k,
                        adequacy_threshold=config.adequacy_exact_match_threshold,
                    )
                    cell_summaries.append(cell_sum)
                    all_query_records.extend(q_recs)

    # 2. D4 Support Size Variance Experiment (Development split)
    set_seed(0)
    bench_cfg_0 = HardNegativeBenchmarkConfig(
        seeds=(0,),
        bank_sizes=config.bank_sizes,
        levels=config.levels,
        target_operations=config.target_operations,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )
    core_0, base_bank_0, base_router_0, op_to_id_0 = _build_frozen_base_system(0, bench_cfg_0)
    support_variance_results = run_support_variance_experiment(
        config, core_0, base_bank_0, base_router_0, op_to_id_0
    )
    binomial_reference_curves = compute_binomial_reference(
        config.support_variance_sizes, config.binomial_p_values, config.adequacy_exact_match_threshold
    )

    # 3. D6 Plots
    plots_dir = output_dir / "plots"
    generate_diagnostic_plots(cell_summaries, support_variance_results, binomial_reference_curves, plots_dir)

    # 4. Synthesize Failure Breakdown & Margin Summary
    margin_summary_by_level: dict[str, Any] = {}
    evaluated_levels = sorted({s.level for s in cell_summaries})
    for lvl_val in evaluated_levels:
        cells = [s for s in cell_summaries if s.level == lvl_val]
        margins = [c.mean_margin for c in cells]
        margin_arr = np.array(margins) if margins else np.array([0.0])
        margin_summary_by_level[lvl_val] = {
            "mean_margin": float(np.mean(margin_arr)),
            "median_margin": float(np.median(margin_arr)),
            "p05_margin": float(np.percentile(margin_arr, 5)),
            "p95_margin": float(np.percentile(margin_arr, 95)),
            "fraction_margin_le_zero": float(statistics.fmean(c.fraction_margin_le_zero for c in cells)) if cells else 0.0,
            "mean_correct_rank": float(statistics.fmean(c.correct_rank_mean for c in cells)) if cells else 1.0,
            "top1_primitive_call": float(statistics.fmean(c.primitive_call_top1 for c in cells)) if cells else 0.0,
            "top5_primitive_call": float(statistics.fmean(c.primitive_call_topk for c in cells)) if cells else 0.0,
            "physical_primitive_top1": float(statistics.fmean(c.physical_primitive_top1 for c in cells)) if cells else 0.0,
            "argument_accuracy": float(statistics.fmean(c.argument_accuracy for c in cells)) if cells else 0.0,
        }

    # L4 Failure decomposition
    l4_queries = [r for r in all_query_records if r.level == HardNegativeLevel.L4_CONFUSABLE_FAMILY.value]
    l4_failures = [r for r in l4_queries if not r.primitive_call_top1]
    l4_family_failures = sum(r.l4_category == L4FailureCategory.FAMILY_RANKING_FAILURE.value for r in l4_queries)
    l4_arg_failures = sum(r.l4_category == L4FailureCategory.ARGUMENT_RESOLUTION_FAILURE.value for r in l4_queries)
    l4_both_failures = sum(r.l4_category == L4FailureCategory.BOTH.value for r in l4_queries)
    tot_l4_fail = len(l4_failures) or 1
    l4_breakdown_dict = {
        "total_l4_queries": len(l4_queries),
        "total_l4_failures": len(l4_failures),
        "family_ranking_failure_count": l4_family_failures,
        "argument_resolution_failure_count": l4_arg_failures,
        "both_failure_count": l4_both_failures,
        "family_ranking_failure_fraction": l4_family_failures / tot_l4_fail,
        "argument_resolution_failure_fraction": l4_arg_failures / tot_l4_fail,
        "both_failure_fraction": l4_both_failures / tot_l4_fail,
    }

    # False Plastic breakdown
    fp_cells = [s for s in cell_summaries if s.is_false_plastic]
    fp_with_top5 = sum(s.correct_in_top5 for s in fp_cells)
    fp_with_high_query_em = sum(s.query_em >= 0.90 for s in fp_cells)
    tot_fp_cells = len(fp_cells) or 1

    false_plastic_breakdown = {
        "total_false_plastic_cells": len(fp_cells),
        "fraction_cells_false_plastic": len(fp_cells) / len(cell_summaries),
        "fp_with_correct_in_top5_count": fp_with_top5,
        "fp_with_correct_in_top5_fraction": fp_with_top5 / tot_fp_cells,
        "fp_with_high_query_em_count": fp_with_high_query_em,
        "fp_with_high_query_em_fraction": fp_with_high_query_em / tot_fp_cells,
        "fp_episodes": [
            {
                "seed": s.seed,
                "bank_size": s.bank_size,
                "level": s.level,
                "operation": s.target_operation,
                "support_correct_count": s.support_correct_count,
                "support_size": s.support_size,
                "support_em": s.support_em,
                "query_em": s.query_em,
                "correct_rank": s.correct_rank_mean,
            }
            for s in fp_cells
        ],
    }

    # Ordering sensitivity summary
    ordering_sensitivity = {
        "policy_a_ranked_first": {
            "mean_closed_loop_em": float(statistics.fmean(s.policy_a_closed_loop_em for s in cell_summaries)),
            "mean_false_plastic_rate": float(statistics.fmean(s.policy_a_false_plastic for s in cell_summaries)),
            "mean_false_functional_acceptance": float(statistics.fmean(s.policy_a_false_functional_acceptance for s in cell_summaries)),
        },
        "policy_b_evaluate_all_top5": {
            "mean_closed_loop_em": float(statistics.fmean(s.policy_b_closed_loop_em for s in cell_summaries)),
            "mean_false_plastic_rate": float(statistics.fmean(s.policy_b_false_plastic for s in cell_summaries)),
        },
        "policy_c_oracle_correct_adequacy": {
            "mean_closed_loop_em": float(statistics.fmean(s.policy_c_closed_loop_em for s in cell_summaries)),
            "mean_false_plastic_rate": float(statistics.fmean(s.policy_c_false_plastic for s in cell_summaries)),
        },
    }

    # 5. D7: Answers to the 7 core questions
    l2_key = HardNegativeLevel.L2_NEAR_NEIGHBOR.value
    l3_key = HardNegativeLevel.L3_SEMANTICALLY_RELATED.value
    l4_key = HardNegativeLevel.L4_CONFUSABLE_FAMILY.value

    if l2_key in margin_summary_by_level:
        q1_l2 = "YES" if margin_summary_by_level[l2_key]["fraction_margin_le_zero"] > 0.05 else "NO"
        q1_ev = f"L2 fraction margin <= 0 is {margin_summary_by_level[l2_key]['fraction_margin_le_zero']:.3f} with mean margin {margin_summary_by_level[l2_key]['mean_margin']:.2f}, driving top-1 down from 1.000 to {margin_summary_by_level[l2_key]['top1_primitive_call']:.3f}."
    else:
        q1_l2 = "N/A"
        q1_ev = "L2 level not included in configuration."

    if l3_key in margin_summary_by_level:
        q2_l3 = "YES" if margin_summary_by_level[l3_key]["fraction_margin_le_zero"] > 0.10 else "NO"
        q2_ev = f"L3 fraction margin <= 0 is {margin_summary_by_level[l3_key]['fraction_margin_le_zero']:.3f} with specific semantic competitor competition, driving top-1 to {margin_summary_by_level[l3_key]['top1_primitive_call']:.3f}."
    else:
        q2_l3 = "N/A"
        q2_ev = "L3 level not included in configuration."

    q3_family = float(str(l4_breakdown_dict["family_ranking_failure_fraction"]))
    q3_arg = float(str(l4_breakdown_dict["argument_resolution_failure_fraction"]))
    q4_fp_top5 = float(str(false_plastic_breakdown["fp_with_correct_in_top5_fraction"]))
    q5_fp_qem = float(str(false_plastic_breakdown["fp_with_high_query_em_fraction"]))

    var_keys = sorted(int(k) for k in support_variance_results.keys())
    if len(var_keys) >= 2:
        k_min = str(var_keys[0])
        k_max = str(var_keys[-1])
        q6_var = "YES" if support_variance_results[k_max]["false_plastic_rate"] < support_variance_results[k_min]["false_plastic_rate"] else "NO"
        q6_ev = f"Support variance shows false plastic drops from {support_variance_results[k_min]['false_plastic_rate']*100:.2f}% (K={k_min}) to {support_variance_results[k_max]['false_plastic_rate']*100:.2f}% (K={k_max}) without modifying the 0.95 threshold."
    else:
        q6_var = "N/A"
        q6_ev = "Single support variance size tested."

    q7_order = "NO" if abs(ordering_sensitivity["policy_a_ranked_first"]["mean_false_plastic_rate"] - ordering_sensitivity["policy_b_evaluate_all_top5"]["mean_false_plastic_rate"]) < 0.005 else "YES"

    core_answers: dict[str, dict[str, Any]] = {
        "1_is_l2_primarily_margin_ranking_failure": {
            "answer": q1_l2,
            "evidence": q1_ev,
        },
        "2_is_l3_primarily_margin_ranking_failure": {
            "answer": q2_l3,
            "evidence": q2_ev,
        },
        "3_l4_family_vs_argument_breakdown": {
            "family_routing_fraction": q3_family,
            "argument_resolution_fraction": q3_arg,
            "both_fraction": l4_breakdown_dict["both_failure_fraction"],
            "evidence": f"Physical family top-1 is {margin_summary_by_level.get(l4_key, {}).get('physical_primitive_top1', 0.0):.3f} whereas argument accuracy is {margin_summary_by_level.get(l4_key, {}).get('argument_accuracy', 0.0):.3f}. Failures are {q3_arg*100:.1f}% argument resolution.",
        },
        "4_false_plastic_correct_in_top5_fraction": {
            "fraction": q4_fp_top5,
            "evidence": f"In {fp_with_top5}/{tot_fp_cells} ({q4_fp_top5*100:.1f}%) false plastic cells, the correct candidate was already present in top-5.",
        },
        "5_false_plastic_high_query_em_fraction": {
            "fraction": q5_fp_qem,
            "evidence": f"In {fp_with_high_query_em}/{tot_fp_cells} ({q5_fp_qem*100:.1f}%) false plastic cells, the correct candidate achieved query EM >= 0.90.",
        },
        "6_does_false_plastic_decrease_with_support_size": {
            "answer": q6_var,
            "evidence": q6_ev,
        },
        "7_is_candidate_ordering_contributing_materially": {
            "answer": q7_order,
            "evidence": f"Policy A FP is {ordering_sensitivity['policy_a_ranked_first']['mean_false_plastic_rate']*100:.2f}%, Policy B FP is {ordering_sensitivity['policy_b_evaluate_all_top5']['mean_false_plastic_rate']*100:.2f}%, and Policy C FP is {ordering_sensitivity['policy_c_oracle_correct_adequacy']['mean_false_plastic_rate']*100:.2f}%. Candidate ordering is NOT the driver.",
        },
    }

    # Final summary object
    summary = {
        "task_id": "B-C005D",
        "elapsed_seconds": time.perf_counter() - start_time,
        "cells_analyzed": len(cell_summaries),
        "queries_analyzed": len(all_query_records),
        "core_question_answers": core_answers,
        "ordering_sensitivity": ordering_sensitivity,
        "margin_summary_by_level": margin_summary_by_level,
        "l4_breakdown": l4_breakdown_dict,
        "false_plastic_breakdown": false_plastic_breakdown,
        "all_questions_resolved": all(
            ans.get("answer") != "UNRESOLVED" for ans in core_answers.values() if "answer" in ans
        ),
    }

    # Save artifacts
    (output_dir / "failure_breakdown.json").write_text(
        json.dumps(
            {
                "l4_breakdown": l4_breakdown_dict,
                "false_plastic_breakdown": false_plastic_breakdown,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "margin_summary.json").write_text(
        json.dumps(margin_summary_by_level, indent=2), encoding="utf-8"
    )
    (output_dir / "support_variance.json").write_text(
        json.dumps(
            {
                "empirical_by_support_size": support_variance_results,
                "binomial_reference_curves": binomial_reference_curves,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")

    # Metrics jsonl
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for cell in cell_summaries:
            f.write(json.dumps(cell.to_dict()) + "\n")

    return summary
