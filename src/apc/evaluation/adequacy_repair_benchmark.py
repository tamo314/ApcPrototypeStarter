# ruff: noqa: E501
"""Phase B Task B-C005R2: Functional Adequacy Estimator Repair Benchmark.

Evaluates four functional adequacy verification policies under the frozen
retrieval mechanism selected by Task B-C005R1 (Condition R2: CombinedRoutingLoss +
ArgumentScorer with frozen query_proj):

Policies:
- Policy A (fixed_32): Original fixed-32 support hard threshold (K=32, EM >= 0.95)
- Policy B (fixed_64): Fixed-64 support hard threshold (K=64, EM >= 0.95)
- Policy C (fixed_128): Fixed-128 support hard threshold (K=128, EM >= 0.95)
- Policy D (sequential): Bounded sequential verifier (initial=32, step=32, max=128)
  with Wilson score confidence bounds for early accept/reject.

Acceptance Criteria (for Policy D across >= 5 dev seeds at N=128):
- false plastic <= 2.0%
- wrong functional acceptance <= 1.0%
- closed-loop EM >= 0.95
- mean support examples consumed < 64
- no query-target leakage
- unselected primitive calls == 0
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.generator import Example
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.compute_accounting import (
    compute_flops_breakdown,
    verify_sparse_execution,
)
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_LEVELS,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    HardNegativeCandidate,
    _build_frozen_base_system,
    _execute_selected_candidates,
    build_hard_negative_candidates,
)
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    RetrievalRepairConfig,
    _rank_candidates_factorized,
    train_repaired_router_and_scorer,
)
from apc.meta.adequacy_verifier import (
    AdequacyDecision,
    SequentialAdequacyVerifier,
    SequentialVerifierConfig,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.argument_scoring import ArgumentScorer
from apc.primitives.router import Router
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

DEFAULT_POLICIES: Final[tuple[str, ...]] = (
    "fixed_32",
    "fixed_64",
    "fixed_128",
    "sequential",
)


@dataclass(frozen=True)
class AdequacyRepairConfig:
    """Explicit configuration for Task B-C005R2 Adequacy Estimator Repair."""

    seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    policies: tuple[str, ...] = DEFAULT_POLICIES
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
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if self.support_examples < self.max_support:
            raise ValueError("support_examples must be >= max_support")
        if self.query_examples < 1:
            raise ValueError("query_examples must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        for policy in self.policies:
            if policy not in DEFAULT_POLICIES:
                raise ValueError(f"Unsupported policy: {policy}")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["levels"] = [level.value for level in self.levels]
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        return data


@dataclass(frozen=True)
class CellAdequacyDiagnostic:
    """Results of evaluating one adequacy policy on one (seed, bank_size, level, target_op) cell."""

    policy: str
    seed: int
    bank_size: int
    level: str
    target_operation: str
    support_count_available: int
    query_count: int
    candidate_count: int
    top1_primitive_call: float
    topk_primitive_call: float
    closed_loop_exact_match: float
    false_functional_acceptance_rate: float
    false_reuse_rate: float
    false_plastic_rate: float
    mean_support_consumed: float
    p95_support_consumed: float
    decision_latency_ms_per_query: float
    verification_flops: int
    candidate_execution_count: int
    selected_forward_calls: int
    unselected_forward_calls: int
    sparse_execution_passed: bool
    leak_audit_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_adequacy_policy_cell(
    *,
    core: Any,
    bank: Any,
    router: Router,
    argument_scorer: ArgumentScorer | None,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_operation: str,
    target_id: int,
    level: HardNegativeLevel,
    support_examples: Sequence[Example],
    query_examples: Sequence[Example],
    seed: int,
    policy: str,
    config: AdequacyRepairConfig,
) -> CellAdequacyDiagnostic:
    """Evaluate one specific adequacy verification policy on a cell under frozen R2 retrieval."""
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
    leak_audit_passed = all(
        "oracle" not in candidate.provenance.lower()
        and "family_id" not in candidate.provenance.lower()
        for candidate in candidates
    )
    correct_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate is not competitor and candidate.execute_primitive_id == target_id
    )

    # 1. Retrieval: Proposal ordering using factorized scoring
    _, support_ordering = _rank_candidates_factorized(
        core,
        router,
        candidates,
        support_examples[: config.initial_support],
        argument_scorer,
        operation_by_id,
        config.arg_lambda,
    )
    _, query_ordering = _rank_candidates_factorized(
        core,
        router,
        candidates,
        query_examples,
        argument_scorer,
        operation_by_id,
        config.arg_lambda,
    )

    top1 = float((query_ordering[:, 0] == correct_index).to(torch.float32).mean().item())
    topk = float(
        (query_ordering[:, : min(config.top_k, len(candidates))] == correct_index)
        .any(dim=-1)
        .to(torch.float32)
        .mean()
        .item()
    )

    # 2. Candidate verification set (top-k proposal, bounded)
    verify_indices = sorted(
        {
            int(index)
            for index in support_ordering[:, : min(config.top_k, len(candidates))]
            .reshape(-1)
            .tolist()
        }
    )

    # Reset forward calls for accounting
    for pid in bank.ids():
        bank.get(pid).reset_forward_call_count()

    decision_start = time.perf_counter()

    # Configure verifier according to policy
    if policy == "fixed_32":
        verifier_cfg = SequentialVerifierConfig(
            adequacy_threshold=config.adequacy_exact_match_threshold,
            initial_support=32,
            max_support=32,
            decision_rule="fixed_threshold",
        )
    elif policy == "fixed_64":
        verifier_cfg = SequentialVerifierConfig(
            adequacy_threshold=config.adequacy_exact_match_threshold,
            initial_support=64,
            max_support=64,
            decision_rule="fixed_threshold",
        )
    elif policy == "fixed_128":
        verifier_cfg = SequentialVerifierConfig(
            adequacy_threshold=config.adequacy_exact_match_threshold,
            initial_support=128,
            max_support=128,
            decision_rule="fixed_threshold",
        )
    elif policy == "sequential":
        verifier_cfg = SequentialVerifierConfig(
            adequacy_threshold=config.adequacy_exact_match_threshold,
            confidence_level=config.confidence_level,
            initial_support=config.initial_support,
            support_increment=config.support_increment,
            max_support=config.max_support,
            interval_method="wilson",
            decision_rule="sequential_confidence",
        )
    else:
        raise ValueError(f"Unknown policy: {policy}")

    verifier = SequentialAdequacyVerifier(verifier_cfg)

    # Run verification for each proposed candidate
    accepted_indices: set[int] = set()
    support_consumed_list: list[int] = []

    for c_idx in verify_indices:
        cand = candidates[c_idx]

        def eval_chunk(start_idx: int, end_idx: int, target_cand: HardNegativeCandidate = cand) -> int:
            chunk_examples = support_examples[start_idx:end_idx]
            predictions = _execute_selected_candidates(
                core,
                bank,
                operation_by_id,
                chunk_examples,
                [target_cand] * len(chunk_examples),
            )
            return sum(
                pred == ex.target_tokens
                for pred, ex in zip(predictions, chunk_examples, strict=True)
            )

        trace = verifier.verify_candidate_sequentially(
            candidate_id=cand.candidate_id,
            execute_primitive_id=cand.execute_primitive_id,
            support_eval_fn=eval_chunk,
            total_available_support=len(support_examples),
        )
        support_consumed_list.append(trace.final_support_consumed)

        if trace.final_decision == AdequacyDecision.ACCEPT:
            accepted_indices.add(c_idx)

    # Audit false functional acceptance on verified wrong candidates
    wrong_verified = [idx for idx in verify_indices if idx != correct_index]
    wrong_accepted = [idx for idx in wrong_verified if idx in accepted_indices]
    false_functional_acceptance_rate = (
        len(wrong_accepted) / len(wrong_verified) if wrong_verified else 0.0
    )

    # 3. Query routing & execution using accepted candidate set
    final_selected: list[HardNegativeCandidate | None] = []
    for row in query_ordering[:, : min(config.top_k, len(candidates))].tolist():
        accepted = next((int(index) for index in row if int(index) in accepted_indices), None)
        final_selected.append(candidates[accepted] if accepted is not None else None)

    selected_for_execution = [cand for cand in final_selected if cand is not None]
    final_predictions: list[tuple[int, ...] | None] = [None] * len(query_examples)

    if selected_for_execution:
        exec_examples = [
            ex
            for ex, cand in zip(query_examples, final_selected, strict=True)
            if cand is not None
        ]
        executed = _execute_selected_candidates(
            core, bank, operation_by_id, exec_examples, selected_for_execution
        )
        for index, pred in zip(
            [i for i, cand in enumerate(final_selected) if cand is not None],
            executed,
            strict=True,
        ):
            final_predictions[index] = pred

    decision_latency_ms = (time.perf_counter() - decision_start) * 1000.0 / len(query_examples)

    closed_loop_em = sum(
        pred == ex.target_tokens
        for pred, ex in zip(final_predictions, query_examples, strict=True)
        if pred is not None
    ) / len(query_examples)

    false_reuse = sum(
        cand is not None and cand is not candidates[correct_index]
        for cand in final_selected
    ) / len(query_examples)

    false_plastic = sum(cand is None for cand in final_selected) / len(query_examples)

    # Sparse execution and FLOP accounting
    physical_selected = {cand.execute_primitive_id for cand in selected_for_execution}
    physical_selected.update(candidates[idx].execute_primitive_id for idx in verify_indices)
    sparse_ok, _ = verify_sparse_execution(bank, physical_selected)
    selected_calls = sum(bank.get(pid).forward_call_count for pid in physical_selected)
    unselected_calls = sum(
        bank.get(pid).forward_call_count for pid in bank.ids() if pid not in physical_selected
    )

    first = query_examples[0]
    flops_breakdown = compute_flops_breakdown(
        core,
        router,
        bank,
        sorted(physical_selected),
        seq_len_task=len(first.task_spec.steps) + 4 if first.task_spec else 1,
        seq_len_content=len(first.input_tokens) + 2,
        seq_len_out=len(first.target_tokens),
        batch_size=1,
    )

    mean_consumed = statistics.fmean(support_consumed_list) if support_consumed_list else 0.0
    p95_consumed = (
        float(statistics.quantiles(support_consumed_list, n=20)[18])
        if len(support_consumed_list) >= 20
        else (float(max(support_consumed_list)) if support_consumed_list else 0.0)
    )

    return CellAdequacyDiagnostic(
        policy=policy,
        seed=seed,
        bank_size=len(bank),
        level=level.value,
        target_operation=target_operation,
        support_count_available=len(support_examples),
        query_count=len(query_examples),
        candidate_count=len(candidates),
        top1_primitive_call=top1,
        topk_primitive_call=topk,
        closed_loop_exact_match=closed_loop_em,
        false_functional_acceptance_rate=false_functional_acceptance_rate,
        false_reuse_rate=false_reuse,
        false_plastic_rate=false_plastic,
        mean_support_consumed=mean_consumed,
        p95_support_consumed=p95_consumed,
        decision_latency_ms_per_query=decision_latency_ms,
        verification_flops=flops_breakdown.total_sparse_flops,
        candidate_execution_count=len(verify_indices),
        selected_forward_calls=selected_calls,
        unselected_forward_calls=unselected_calls,
        sparse_execution_passed=sparse_ok and unselected_calls == 0,
        leak_audit_passed=leak_audit_passed,
    )


def run_adequacy_repair_benchmark(config: AdequacyRepairConfig) -> dict[str, Any]:
    """Execute Task B-C005R2 adequacy estimator repair benchmark across policies."""
    start_time = time.perf_counter()
    diagnostics: list[CellAdequacyDiagnostic] = []

    # 1. Evaluate each seed
    for seed in config.seeds:
        set_seed(seed, deterministic_algorithms=config.deterministic_algorithms)

        model_seed = seed % 5 if seed >= 5 else seed
        base_config = HardNegativeBenchmarkConfig(
            seeds=(model_seed,),
            bank_sizes=config.bank_sizes,
            levels=config.levels,
            target_operations=config.target_operations,
            num_eval_examples=config.query_examples,
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            top_k=config.top_k,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)

        # Generate disjoint development training data for Condition R2 retrieval
        dev_train_examples: dict[str, list[Example]] = {}
        for op in tuple(INITIAL_10_OPERATIONS) + tuple(config.target_operations):
            if op in op_to_id:
                pid = op_to_id[op]
                exs = generate_benchmark_examples(
                    seed * 50_000 + pid * 100 + 7,
                    config.router_train_examples,
                    operation=op,
                    split="train",
                )
                dev_train_examples[op] = list(exs)

        for bank_size in config.bank_sizes:
            bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
                core, base_bank, base_router, op_to_id, bank_size, seed=seed
            )
            operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
            operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

            # Train Condition R2 router and ArgumentScorer once per (seed, bank_size), then strictly freeze
            repair_cfg = RetrievalRepairConfig(
                seeds=(seed,),
                bank_sizes=(bank_size,),
                levels=config.levels,
                target_operations=config.target_operations,
                support_examples=config.support_examples,
                query_examples=config.query_examples,
                router_train_examples=config.router_train_examples,
                router_steps=config.router_steps,
                router_lr=config.router_lr,
                top_k=config.top_k,
                ranking_margin=config.ranking_margin,
                ranking_beta=config.ranking_beta,
                arg_lambda=config.arg_lambda,
                device=config.device,
            )
            frozen_router, frozen_scorer = train_repaired_router_and_scorer(
                core,
                router,
                candidate_ids,
                operation_by_id,
                dev_train_examples,
                config=repair_cfg,
                condition="R2",
                device=core.device,
            )
            frozen_router.eval()
            for p in frozen_router.parameters():
                p.requires_grad_(False)
            if frozen_scorer is not None:
                frozen_scorer.eval()
                for p in frozen_scorer.parameters():
                    p.requires_grad_(False)

            for target_op in config.target_operations:
                target_id = op_to_id[target_op]

                support = generate_benchmark_examples(
                    seed * 100_000 + bank_size * 100 + target_id,
                    config.support_examples,
                    operation=target_op,
                    split="train",
                )
                query = generate_benchmark_examples(
                    seed * 100_000 + bank_size * 100 + target_id + 1,
                    config.query_examples,
                    operation=target_op,
                    split="test",
                )

                for level in config.levels:
                    for policy in config.policies:
                        diag = evaluate_adequacy_policy_cell(
                            core=core,
                            bank=bank,
                            router=frozen_router,
                            argument_scorer=frozen_scorer,
                            candidate_ids=candidate_ids,
                            operation_by_id=operation_by_id,
                            target_operation=target_op,
                            target_id=target_id,
                            level=level,
                            support_examples=support,
                            query_examples=query,
                            seed=seed,
                            policy=policy,
                            config=config,
                        )
                        diagnostics.append(diag)

    elapsed = time.perf_counter() - start_time

    # Summarize results across policies
    summary_by_policy: dict[str, dict[str, Any]] = {}
    for policy in config.policies:
        pol_cells = [d for d in diagnostics if d.policy == policy]
        pol_n128 = [d for d in pol_cells if d.bank_size == 128]

        summary_by_policy[policy] = {
            "all_banks": {
                "cells": len(pol_cells),
                "mean_closed_loop_em": statistics.fmean(d.closed_loop_exact_match for d in pol_cells),
                "mean_false_plastic": statistics.fmean(d.false_plastic_rate for d in pol_cells),
                "mean_false_acceptance": statistics.fmean(d.false_functional_acceptance_rate for d in pol_cells),
                "mean_support_consumed": statistics.fmean(d.mean_support_consumed for d in pol_cells),
                "p95_support_consumed": statistics.fmean(d.p95_support_consumed for d in pol_cells),
                "mean_latency_ms": statistics.fmean(d.decision_latency_ms_per_query for d in pol_cells),
                "mean_flops": statistics.fmean(d.verification_flops for d in pol_cells),
            },
            "n128": {
                "cells": len(pol_n128),
                "mean_closed_loop_em": statistics.fmean(d.closed_loop_exact_match for d in pol_n128) if pol_n128 else 0.0,
                "mean_false_plastic": statistics.fmean(d.false_plastic_rate for d in pol_n128) if pol_n128 else 0.0,
                "mean_false_acceptance": statistics.fmean(d.false_functional_acceptance_rate for d in pol_n128) if pol_n128 else 0.0,
                "mean_support_consumed": statistics.fmean(d.mean_support_consumed for d in pol_n128) if pol_n128 else 0.0,
                "p95_support_consumed": statistics.fmean(d.p95_support_consumed for d in pol_n128) if pol_n128 else 0.0,
                "mean_latency_ms": statistics.fmean(d.decision_latency_ms_per_query for d in pol_n128) if pol_n128 else 0.0,
            },
        }

    # Evaluate Task B-C005R2 acceptance criteria for Policy D (Sequential)
    seq_n128 = summary_by_policy.get("sequential", {}).get("n128", {})
    seq_cells = [d for d in diagnostics if d.policy == "sequential"]

    seq_false_plastic = seq_n128.get("mean_false_plastic", 1.0)
    seq_false_acceptance = seq_n128.get("mean_false_acceptance", 1.0)
    seq_em = seq_n128.get("mean_closed_loop_em", 0.0)
    seq_mean_support = seq_n128.get("mean_support_consumed", 128.0)

    criteria = {
        "false_plastic_le_2_pct": {
            "target": "<= 0.02",
            "measured": seq_false_plastic,
            "passed": seq_false_plastic <= 0.02,
        },
        "wrong_functional_acceptance_le_1_pct": {
            "target": "<= 0.01",
            "measured": seq_false_acceptance,
            "passed": seq_false_acceptance <= 0.01,
        },
        "closed_loop_em_ge_95_pct": {
            "target": ">= 0.95",
            "measured": seq_em,
            "passed": seq_em >= 0.95,
        },
        "mean_support_consumed_lt_64": {
            "target": "< 64.0",
            "measured": seq_mean_support,
            "passed": seq_mean_support < 64.0,
        },
        "no_query_target_leakage": {
            "target": "True",
            "measured": all(d.leak_audit_passed for d in seq_cells),
            "passed": all(d.leak_audit_passed for d in seq_cells),
        },
        "zero_unselected_calls": {
            "target": "True",
            "measured": all(d.unselected_forward_calls == 0 for d in seq_cells),
            "passed": all(d.unselected_forward_calls == 0 for d in seq_cells),
        },
    }

    all_criteria_passed = all(c["passed"] for c in criteria.values())

    report = {
        "task_id": "B-C005R2",
        "config": config.to_dict(),
        "elapsed_seconds": elapsed,
        "criteria": criteria,
        "all_criteria_passed": all_criteria_passed,
        "policy_summary": summary_by_policy,
        "matrix_count": len(diagnostics),
    }

    # Save artifacts if output_dir provided
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "config.yaml").write_text(
            import_yaml().safe_dump(config.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        (config.output_dir / "system.json").write_text(
            json.dumps(get_system_info(), indent=2), encoding="utf-8"
        )
        (config.output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(d.to_dict()) + "\n" for d in diagnostics),
            encoding="utf-8",
        )
        (config.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        # Markdown report
        md_lines = [
            "# Task B-C005R2: Functional Adequacy Estimator Repair Benchmark",
            "",
            f"**Status:** {'PASS' if all_criteria_passed else 'FAIL'}",
            f"**Elapsed Time:** {elapsed:.2f}s",
            "",
            "## Acceptance Criteria (Policy D @ N=128 across 5 dev seeds)",
            "| Criterion | Target | Measured | Result |",
            "|---|---|---|:---:|",
        ]
        for name, res in criteria.items():
            meas_str = (
                f"{res['measured']:.4f}"
                if isinstance(res["measured"], float)
                else str(res["measured"])
            )
            md_lines.append(
                f"| {name} | {res['target']} | {meas_str} | {'PASS' if res['passed'] else 'FAIL'} |"
            )

        md_lines.extend(
            [
                "",
                "## Comparative Policy Evaluation at N=128",
                "| Policy | Closed-loop EM | False Plastic | False Acceptance | Mean Support | P95 Support | Latency (ms) |",
                "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for pol, data in summary_by_policy.items():
            n128 = data["n128"]
            md_lines.append(
                f"| {pol} | {n128['mean_closed_loop_em']:.4f} | {n128['mean_false_plastic']:.4f} | "
                f"{n128['mean_false_acceptance']:.4f} | {n128['mean_support_consumed']:.1f} | "
                f"{n128['p95_support_consumed']:.1f} | {n128['mean_latency_ms']:.2f} |"
            )

        (config.output_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report


def import_yaml() -> Any:
    import yaml

    return yaml
