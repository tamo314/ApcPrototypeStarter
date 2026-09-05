"""Compact-First Plastic Lifecycle Benchmark Harness (Task A2-C007).

Evaluates the mechanical lifecycle guarantees of CompactPlasticLifecyclePolicy:
1. No plastic before direct/composition inadequacy.
2. Exactly one promotion per successful novel operation (N).
3. Temporary workspace capacity returns to zero parameters across 100% of episodes.
4. Zero promotions for Known (K), Composition (C), and Recurrence (R) tasks.
5. Separate reporting of compact success, compact failure, fallback invoked,
   and fallback success/failure controls across >=5 decision seeds.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.generator import (
    Example,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.evaluation.discovery_capacity_harness import (
    generate_novel_examples,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
)
from apc.meta.adequacy import AdequacyEvidence
from apc.meta.episode_log import ControllerAction
from apc.plastic.lifecycle import (
    CompactLifecycleConfig,
    CompactLifecycleReport,
    CompactPlasticLifecyclePolicy,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.utils.seed import set_seed

DEFAULT_BENCHMARK_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


def _make_evidence(
    *, direct_em: float = 0.0, composition_em: float = 0.0, direct_loss: float = 0.01
) -> AdequacyEvidence:
    """Helper to create valid AdequacyEvidence records."""
    return AdequacyEvidence(
        direct_em=direct_em,
        direct_loss=direct_loss,
        direct_token_acc=direct_em,
        direct_primitive_id=0 if direct_em > 0 else None,
        composition_em=composition_em,
    )


def _make_examples_for_program(
    prog: Program, n: int, seed: int, vocab_size: int = 10
) -> list[Example]:
    """Deterministically generate input/output examples for a Program."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        length = rng.randint(6, 10)
        seq = tuple(rng.randrange(vocab_size) for _ in range(length))
        res = run_program(prog, seq, vocab_size)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split="train",
                vocab_size=vocab_size,
            )
        )
    return examples


@dataclass(frozen=True)
class BenchmarkAggregateSummary:
    """Aggregate metrics across seeds for Task A2-C007."""

    seeds_evaluated: list[int]
    total_episodes: int
    k_episodes: int
    c_episodes: int
    n_episodes: int
    r_episodes: int

    # Controls required by Task A2-C007
    compact_success_cases: int
    compact_failure_cases: int
    fallback_invoked_cases: int
    fallback_success_cases: int
    fallback_failure_cases: int

    # Lifecycle criteria
    k_promotions: int
    c_promotions: int
    r_promotions: int
    successful_n_episodes: int
    successful_n_promotions: int
    failed_n_promotions: int

    premature_plastic_count: int
    workspace_leak_count: int

    # Explicit Acceptance Verdicts
    no_plastic_before_inadequacy_passed: bool
    exactly_one_promotion_per_successful_n_passed: bool
    workspace_returns_to_zero_passed: bool
    no_k_c_r_promotion_passed: bool
    all_criteria_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_compact_lifecycle_suite_for_seed(
    seed: int,
    *,
    core: Any,
    base_bank: PrimitiveBank,
    base_op_to_id: dict[str, int],
    config: CompactLifecycleConfig,
    device: torch.device,
) -> list[CompactLifecycleReport]:
    """Run a full K/C/N/R test suite for a single seed."""
    set_seed(seed)
    policy = CompactPlasticLifecyclePolicy(config)
    workspace = PlasticWorkspace()

    # Reconstruct bank with identical primitives
    bank = PrimitiveBank(primitives=[base_bank.get(pid) for pid in base_bank.ids()])
    bank.freeze_all()
    op_to_id = dict(base_op_to_id)

    reports: list[CompactLifecycleReport] = []

    # Historical examples for shadow validation retention check
    historical_examples: dict[str, list[Example]] = {}
    for op in ALL_CANONICAL_OPERATIONS:
        historical_examples[op] = generate_benchmark_examples(
            seed=seed * 100 + 1, n=20, operation=op, split="val"
        )

    # 1. Episode K1: Known operation (COPY)
    k1_eval = generate_benchmark_examples(seed=seed * 10 + 1, n=30, operation="COPY", split="test")
    rep_k1 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_01_K1",
        task_name="COPY",
        action=ControllerAction.DIRECT_REUSE,
        evidence=_make_evidence(direct_em=1.0, direct_loss=0.01),
        train_examples=k1_eval[:10],
        eval_examples=k1_eval[10:],
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )
    reports.append(rep_k1)

    # 2. Episode K2: Known operation (REVERSE)
    k2_eval = generate_benchmark_examples(
        seed=seed * 10 + 2, n=30, operation="REVERSE", split="test"
    )
    rep_k2 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_02_K2",
        task_name="REVERSE",
        action=ControllerAction.DIRECT_REUSE,
        evidence=_make_evidence(direct_em=1.0, direct_loss=0.01),
        train_examples=k2_eval[:10],
        eval_examples=k2_eval[10:],
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )
    reports.append(rep_k2)

    # 3. Episode C1: Composition (REVERSE + NEGATE)
    c1_base = generate_benchmark_examples(
        seed=seed * 10 + 3, n=30, operation="REVERSE", split="test"
    )
    neg_op = get_operation("NEGATE")
    c1_eval: list[Example] = []
    c_prog = Program((ProgramStep("REVERSE"), ProgramStep("NEGATE")))
    for ex in c1_base:
        neg_target = neg_op.apply(ex.target_tokens, 10, {})
        c1_eval.append(
            Example(
                input_tokens=ex.input_tokens,
                target_tokens=neg_target,
                program=c_prog,
                operation_graph=ex.operation_graph,
                category="composition",
                split="test",
                vocab_size=10,
            )
        )
    rep_c1 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_03_C1",
        task_name="REVERSE_NEGATE",
        action=ControllerAction.COMPOSE,
        evidence=_make_evidence(direct_em=0.0, composition_em=1.0),
        train_examples=c1_eval[:10],
        eval_examples=c1_eval[10:],
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )
    reports.append(rep_c1)

    # 4. Episode N_compact: Novel task solved via compact search (SWAP_PAIRS)
    n_compact_train = generate_novel_examples(
        seed=seed * 100 + 41, n=80, operation="SWAP_PAIRS", split="train"
    )
    n_compact_eval = generate_novel_examples(
        seed=seed * 100 + 42, n=30, operation="SWAP_PAIRS", split="test"
    )
    rep_n1 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_04_N_compact",
        task_name="SWAP_PAIRS",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=n_compact_train,
        eval_examples=n_compact_eval,
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=False,
    )
    reports.append(rep_n1)

    # 5. Episode R1: Recurrence of SWAP_PAIRS (now installed in bank)
    n_rec_eval = generate_novel_examples(
        seed=seed * 100 + 51, n=30, operation="SWAP_PAIRS", split="test"
    )
    rep_r1 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_05_R_recurrence",
        task_name="SWAP_PAIRS",
        action=ControllerAction.DIRECT_REUSE,
        evidence=_make_evidence(direct_em=1.0, direct_loss=0.02),
        train_examples=n_rec_eval[:10],
        eval_examples=n_rec_eval[10:],
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
    )
    reports.append(rep_r1)

    # 6. Episode N_fallback: Novel task requiring fallback (forced compact fail on SWAP_PAIRS)
    n_fallback_train = generate_novel_examples(
        seed=seed * 100 + 61, n=80, operation="SWAP_PAIRS", split="train"
    )
    n_fallback_eval = generate_novel_examples(
        seed=seed * 100 + 62, n=30, operation="SWAP_PAIRS", split="test"
    )
    rep_n2 = policy.execute_episode(
        episode_id=f"seed_{seed}_ep_06_N_fallback",
        task_name="SWAP_PAIRS_FALLBACK",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=n_fallback_train,
        eval_examples=n_fallback_eval,
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=True,  # Bounded compact budget failure -> invokes fallback
    )
    reports.append(rep_n2)

    # 7. Episode N_fail_control: Novel task where compact & fallback fail or fallback disabled
    policy_no_fallback = CompactPlasticLifecyclePolicy(
        dataclasses.replace(config, enable_overcomplete_fallback=False)
    )
    rep_n3 = policy_no_fallback.execute_episode(
        episode_id=f"seed_{seed}_ep_07_N_fail_control",
        task_name="UNSOLVABLE_OP",
        action=ControllerAction.PLASTIC_SEARCH,
        evidence=_make_evidence(direct_em=0.0, composition_em=0.0),
        train_examples=n_fallback_train[:10],
        eval_examples=n_fallback_eval[:10],
        historical_eval_examples=historical_examples,
        core=core,
        bank=bank,
        op_to_id=op_to_id,
        workspace=workspace,
        forced_compact_fail=True,
    )
    reports.append(rep_n3)

    return reports


def run_compact_lifecycle_benchmark(
    seeds: Sequence[int] = DEFAULT_BENCHMARK_SEEDS,
    output_dir: str | Path = "runs/phase_a2_compact_lifecycle_benchmark",
    compact_steps: int = 250,
    fallback_steps: int = 200,
    distillation_steps: int = 150,
) -> BenchmarkAggregateSummary:
    """Run full 5-seed benchmark and produce aggregate metrics for Task A2-C007."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen core and canonical 8-primitive bank
    u_config = UnifiedBenchmarkConfig(
        seed=42,
        vocab_size=10,
        core_train_steps=2000,
        parameterized_train_steps=2000,
        parameter_free_train_steps=2000,
        device=str(device),
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=out_path)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    base_bank, base_op_to_id = _build_heterogeneous_bank(u_config)
    base_bank.freeze_all()
    base_bank.to(device)

    config = CompactLifecycleConfig(
        compact_budget_steps=compact_steps,
        compact_lr=2e-3,
        compact_success_threshold_em=0.80,
        fallback_budget_steps=fallback_steps,
        fallback_lr=2e-3,
        fallback_success_threshold_em=0.80,
        distillation_steps=distillation_steps,
        distillation_lr=2e-3,
        shadow_retention_threshold=0.80,
        eval_batch_size=32,
    )

    all_reports: list[CompactLifecycleReport] = []

    print(f"Starting Compact-First Plastic Lifecycle Benchmark across seeds {seeds}...")
    start_time = time.perf_counter()

    for s in seeds:
        seed_reports = run_compact_lifecycle_suite_for_seed(
            s,
            core=core,
            base_bank=base_bank,
            base_op_to_id=base_op_to_id,
            config=config,
            device=device,
        )
        all_reports.extend(seed_reports)
        print(f"  Seed {s} complete ({len(seed_reports)} episodes).")

    total_time = time.perf_counter() - start_time

    # Aggregate statistics
    total_episodes = len(all_reports)
    k_episodes = sum(1 for r in all_reports if "K" in r.episode_id)
    c_episodes = sum(1 for r in all_reports if "C" in r.episode_id)
    n_episodes = sum(1 for r in all_reports if "N" in r.episode_id)
    r_episodes = sum(1 for r in all_reports if "R" in r.episode_id)

    compact_success_cases = sum(1 for r in all_reports if r.compact_success)
    compact_failure_cases = sum(1 for r in all_reports if r.compact_failure)
    fallback_invoked_cases = sum(1 for r in all_reports if r.fallback_invoked)
    fallback_success_cases = sum(1 for r in all_reports if r.fallback_success)
    fallback_failure_cases = sum(1 for r in all_reports if r.fallback_failure)

    k_promotions = sum(1 for r in all_reports if "K" in r.episode_id and r.promotions_count > 0)
    c_promotions = sum(1 for r in all_reports if "C" in r.episode_id and r.promotions_count > 0)
    r_promotions = sum(1 for r in all_reports if "R" in r.episode_id and r.promotions_count > 0)

    # Successful N episodes should have exactly 1 promotion
    successful_n = [
        r for r in all_reports if "N" in r.episode_id and r.shadow_validation_passed
    ]
    successful_n_episodes = len(successful_n)
    successful_n_promotions = sum(r.promotions_count for r in successful_n)

    failed_n = [
        r for r in all_reports if "N" in r.episode_id and not r.shadow_validation_passed
    ]
    failed_n_promotions = sum(r.promotions_count for r in failed_n)

    # Checks
    premature_plastic_count = sum(
        1
        for r in all_reports
        if (r.direct_sufficient or r.composition_sufficient) and r.plastic_triggered
    )
    workspace_leak_count = sum(1 for r in all_reports if r.final_workspace_param_count != 0)

    no_plastic_before_inadequacy = premature_plastic_count == 0
    exactly_one_promotion = (
        successful_n_promotions == successful_n_episodes and failed_n_promotions == 0
    )
    workspace_returns_to_zero = workspace_leak_count == 0
    no_k_c_r_promotion = (k_promotions == 0) and (c_promotions == 0) and (r_promotions == 0)

    all_criteria_passed = (
        no_plastic_before_inadequacy
        and exactly_one_promotion
        and workspace_returns_to_zero
        and no_k_c_r_promotion
    )

    summary = BenchmarkAggregateSummary(
        seeds_evaluated=list(seeds),
        total_episodes=total_episodes,
        k_episodes=k_episodes,
        c_episodes=c_episodes,
        n_episodes=n_episodes,
        r_episodes=r_episodes,
        compact_success_cases=compact_success_cases,
        compact_failure_cases=compact_failure_cases,
        fallback_invoked_cases=fallback_invoked_cases,
        fallback_success_cases=fallback_success_cases,
        fallback_failure_cases=fallback_failure_cases,
        k_promotions=k_promotions,
        c_promotions=c_promotions,
        r_promotions=r_promotions,
        successful_n_episodes=successful_n_episodes,
        successful_n_promotions=successful_n_promotions,
        failed_n_promotions=failed_n_promotions,
        premature_plastic_count=premature_plastic_count,
        workspace_leak_count=workspace_leak_count,
        no_plastic_before_inadequacy_passed=no_plastic_before_inadequacy,
        exactly_one_promotion_per_successful_n_passed=exactly_one_promotion,
        workspace_returns_to_zero_passed=workspace_returns_to_zero,
        no_k_c_r_promotion_passed=no_k_c_r_promotion,
        all_criteria_passed=all_criteria_passed,
    )

    # Save outputs
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    with open(out_path / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in all_reports:
            f.write(json.dumps(r.to_dict()) + "\n")

    v_plastic = "PASS" if no_plastic_before_inadequacy else "FAIL"
    v_one = "PASS" if exactly_one_promotion else "FAIL"
    v_zero = "PASS" if workspace_returns_to_zero else "FAIL"
    v_k = "PASS" if k_promotions == 0 else "FAIL"
    v_c = "PASS" if c_promotions == 0 else "FAIL"
    v_r = "PASS" if r_promotions == 0 else "FAIL"

    ep_breakdown = f"K: {k_episodes}, C: {c_episodes}, N: {n_episodes}, R: {r_episodes}"
    n_promo_str = f"{successful_n_promotions} / {successful_n_episodes}"
    md_report = (
        "# Compact-First Plastic Lifecycle Benchmark Report (Task A2-C007)\n\n"
        f"- **Date:** 2026-09-05\n"
        f"- **Seeds Evaluated:** {seeds}\n"
        f"- **Total Episodes:** {total_episodes} ({ep_breakdown})\n"
        f"- **Elapsed Time:** {total_time:.2f}s\n"
        f"- **All Criteria Passed:** {all_criteria_passed}\n\n"
        "## Controls Breakdown\n"
        "| Control Category | Count |\n"
        "|---|---|\n"
        f"| Compact Success Cases | {compact_success_cases} |\n"
        f"| Compact Failure Cases | {compact_failure_cases} |\n"
        f"| Fallback Invoked Cases | {fallback_invoked_cases} |\n"
        f"| Fallback Success Cases | {fallback_success_cases} |\n"
        f"| Fallback Failure Cases | {fallback_failure_cases} |\n\n"
        "## Mechanical Lifecycle Verification\n"
        "| Criterion | Target | Measured | Verdict |\n"
        "|---|---|---|---|\n"
        f"| No plastic before direct/composition inadequacy "
        f"| 0 premature | {premature_plastic_count} | {v_plastic} |\n"
        f"| Exactly one promotion per successful N "
        f"| 1:1 match | {n_promo_str} | {v_one} |\n"
        f"| Temporary workspace returns to zero "
        f"| 0 leaks | {workspace_leak_count} | {v_zero} |\n"
        f"| Zero promotions on Known (K) | 0 | {k_promotions} | {v_k} |\n"
        f"| Zero promotions on Composition (C) | 0 | {c_promotions} | {v_c} |\n"
        f"| Zero promotions on Recurrence (R) | 0 | {r_promotions} | {v_r} |\n"
    )
    with open(out_path / "BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + md_report)
    return summary
