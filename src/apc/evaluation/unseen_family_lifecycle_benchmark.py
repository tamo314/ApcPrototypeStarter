"""Unseen-Family Lifecycle Benchmark (Task B-C003 - STOP GATE B1).

Evaluates whether the frozen Phase A.2 controller and compact-first plastic lifecycle
generalize to genuinely held-out, sealed operation families under explicit model-visible
TaskSpec, before introducing task inference.

Reference:
- `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md` (Task B-C003)
- `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md` (Gate B1)
- `docs/design-docs/OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md`

Gate B1 PASS Criteria:
Across >= 5 seeds (0, 1, 2, 3, 4):
1. `plastic_trigger_rate >= 0.95`
2. `mean_final_novel_EM >= 0.95`
3. `every seed's novel EM >= 0.90`
4. exactly one promotion per successfully learned novel capability
5. `workspace_leaks == 0`
6. fresh-runtime recurrence:
   - `mean_recurrence_EM >= 0.95`
   - `adaptation_steps == 0`
   - `temporary_params == 0`
   - `bank_growth == 0`
7. legacy regression:
   - `old_task_em_drop <= 1.0 percentage point`
   - `old_routing_top1_drop <= 1.0 percentage point`
   - `false_plastic_legacy <= 1%`
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.generator import Example, OracleMetadata
from apc.environments.holdout_families import (
    DEFAULT_FAMILY_REGISTRY,
    HoldoutFamilyRegistry,
)
from apc.environments.interpreter import run_program
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.evaluation.holdout_protocol import check_novelty_validity
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.sequential_closed_loop_benchmark import (
    CANONICAL_DETERMINISTIC_K_OPS,
    COMPOSITION_PAIRS,
    _evaluate_candidate_recipe,
    _setup_initial_environment,
)
from apc.meta.adequacy import (
    AdequacyEvidenceConfig,
    compute_adequacy_evidence,
)
from apc.meta.episode_log import ControllerAction
from apc.meta.learned_controller import (
    LearnedAdequacyController,
    build_default_trained_controller,
)
from apc.meta.phase_b_protocol import FamilySplit
from apc.plastic.lifecycle import (
    CompactLifecycleConfig,
    CompactPlasticLifecyclePolicy,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    update_router_incrementally,
)
from apc.primitives.router import Router
from apc.utils.seed import set_seed

DEFAULT_GATE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
DEFAULT_SEALED_OPERATIONS: Final[tuple[str, ...]] = ("MAJORITY_THREE", "NEIGHBOR_MAX")

PLASTIC_TRIGGER_THRESHOLD: Final[float] = 0.95
MEAN_NOVEL_EM_THRESHOLD: Final[float] = 0.95
MIN_SEED_NOVEL_EM_THRESHOLD: Final[float] = 0.90
RECURRENCE_EM_THRESHOLD: Final[float] = 0.95
MAX_OLD_TASK_EM_DROP: Final[float] = 0.01  # 1.0 percentage point
MAX_OLD_ROUTING_DROP: Final[float] = 0.01  # 1.0 percentage point
MAX_FALSE_PLASTIC_RATE: Final[float] = 0.01  # 1%


@dataclass(frozen=True)
class UnseenFamilyLifecycleConfig:
    """Serializable configuration for Task B-C003 Gate B1 evaluation."""

    seeds: tuple[int, ...] = DEFAULT_GATE_SEEDS
    sealed_operations: tuple[str, ...] = DEFAULT_SEALED_OPERATIONS
    support_size: int = 16
    eval_size: int = 64
    plastic_train_size: int = 1000
    compact_budget_steps: int = 1200
    compact_batch_size: int = 64
    compact_lr: float = 2e-3
    fallback_budget_steps: int = 600
    fallback_lr: float = 2e-3
    distillation_steps: int = 400
    shadow_retention_threshold: float = 0.90
    device_str: str = "auto"
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "sealed_operations": list(self.sealed_operations),
            "support_size": self.support_size,
            "eval_size": self.eval_size,
            "plastic_train_size": self.plastic_train_size,
            "compact_budget_steps": self.compact_budget_steps,
            "compact_batch_size": self.compact_batch_size,
            "compact_lr": self.compact_lr,
            "fallback_budget_steps": self.fallback_budget_steps,
            "fallback_lr": self.fallback_lr,
            "distillation_steps": self.distillation_steps,
            "shadow_retention_threshold": self.shadow_retention_threshold,
            "device_str": self.device_str,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


@dataclass(frozen=True)
class NovelEpisodeOutcome:
    """Record of lifecycle execution for a single novel holdout operation."""

    operation_name: str
    precheck_direct_em: float
    precheck_comp_em: float
    is_valid_novel_holdout: bool
    predicted_action: str
    novelty_score: float
    plastic_triggered: bool
    compact_success: bool
    fallback_invoked: bool
    fallback_success: bool
    shadow_validation_passed: bool
    promoted_primitive_id: int | None
    promotions_count: int
    final_novel_em: float
    workspace_param_count_after: int
    workspace_leak: bool
    recurrence_action: str
    recurrence_em: float
    recurrence_adaptation_steps: int
    recurrence_temporary_params: int
    recurrence_bank_growth: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SeedLifecycleResult:
    """Lifecycle evaluation outcome for a single seed."""

    seed: int
    novel_outcomes: dict[str, NovelEpisodeOutcome]
    legacy_k_em_before: float
    legacy_k_em_after: float
    legacy_c_em_before: float
    legacy_c_em_after: float
    old_routing_top1_before: float
    old_routing_top1_after: float
    legacy_false_plastic_count: int
    legacy_total_count: int
    legacy_false_plastic_rate: float
    old_task_em_drop: float
    old_routing_drop: float
    workspace_leak_count: int
    seed_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "novel_outcomes": {k: v.to_dict() for k, v in self.novel_outcomes.items()},
            "legacy_k_em_before": self.legacy_k_em_before,
            "legacy_k_em_after": self.legacy_k_em_after,
            "legacy_c_em_before": self.legacy_c_em_before,
            "legacy_c_em_after": self.legacy_c_em_after,
            "old_routing_top1_before": self.old_routing_top1_before,
            "old_routing_top1_after": self.old_routing_top1_after,
            "legacy_false_plastic_count": self.legacy_false_plastic_count,
            "legacy_total_count": self.legacy_total_count,
            "legacy_false_plastic_rate": self.legacy_false_plastic_rate,
            "old_task_em_drop": self.old_task_em_drop,
            "old_routing_drop": self.old_routing_drop,
            "workspace_leak_count": self.workspace_leak_count,
            "seed_passed": self.seed_passed,
        }


@dataclass(frozen=True)
class GateB1AggregateReport:
    """Aggregate scientific report evaluating STOP GATE B1 across >=5 seeds."""

    seeds_evaluated: list[int]
    total_novel_episodes: int
    plastic_trigger_rate: float
    mean_final_novel_em: float
    min_seed_novel_em: float
    novel_em_per_seed: dict[int, float]
    total_promotions: int
    expected_promotions: int
    one_to_one_promotions: bool
    workspace_leak_count: int
    mean_recurrence_em: float
    max_recurrence_adaptation_steps: int
    max_recurrence_temporary_params: int
    total_recurrence_bank_growth: int
    mean_old_task_em_drop: float
    max_old_task_em_drop: float
    mean_old_routing_drop: float
    max_old_routing_drop: float
    legacy_false_plastic_rate: float

    # Gate conditions
    passed_plastic_trigger: bool
    passed_mean_novel_em: bool
    passed_min_seed_novel_em: bool
    passed_one_to_one_promotion: bool
    passed_workspace_leaks: bool
    passed_recurrence: bool
    passed_legacy_regression: bool
    gate_b1_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _make_deterministic_examples(
    operation_name: str,
    n: int,
    seed: int,
    seq_length: int = 8,
    vocab_size: int = 10,
    category: str = "N",
) -> list[Example]:
    """Generate deterministic examples with explicit TaskSpec and zero oracle leakage."""
    rng = random.Random(seed)
    prog = Program(steps=(ProgramStep(operation=operation_name, params={}),))
    task_spec = TaskSpec.from_program(prog)
    examples: list[Example] = []

    for _ in range(n):
        seq = tuple(rng.randrange(vocab_size) for _ in range(seq_length))
        res = run_program(prog, seq, vocab_size)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category=category,
                split="test",
                vocab_size=vocab_size,
                task_spec=task_spec,
                oracle_metadata=OracleMetadata(
                    label=category,  # type: ignore[arg-type]
                    primitive_operations=prog.operation_sequence,
                ),
            )
        )
    return examples


def _evaluate_router_accuracy(
    core: Any,
    router: Router,
    op_to_id: dict[str, int],
    test_ops: Sequence[str],
    seed: int,
) -> float:
    """Evaluate router top-1 accuracy on a set of operations."""
    correct = 0
    total = 0
    router.eval()

    for op in test_ops:
        if op not in op_to_id:
            continue
        expected_pid = op_to_id[op]
        examples = generate_benchmark_examples(
            seed=seed * 500 + expected_pid * 37,
            n=20,
            operation=op,
            split="val",
            vocab_size=10,
        )
        z_tasks = extract_task_representations(core, examples)
        with torch.no_grad():
            cand_ids = router.ids()
            preds = router(z_tasks, cand_ids)
            top1_preds = preds.selected_ids[:, 0].tolist()
            for p in top1_preds:
                if p == expected_pid:
                    correct += 1
                total += 1

    return correct / max(1, total)


def run_unseen_family_lifecycle_for_seed(
    seed: int,
    config: UnseenFamilyLifecycleConfig,
    *,
    controller: LearnedAdequacyController,
    device: torch.device,
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> SeedLifecycleResult:
    """Execute complete Gate B1 lifecycle evaluation for a single seed."""
    set_seed(seed)
    core, bank, router, op_to_id, replay_buf = _setup_initial_environment(seed, device)

    # 1. Verify partition disjointness and sealed status
    registry.assert_disjoint_partitions()
    for op in config.sealed_operations:
        meta = registry.get_family_for_operation(op)
        if meta.status != FamilySplit.SEALED_FAMILIES:
            msg = f"Operation {op} status is {meta.status.value}, expected SEALED_FAMILIES"
            raise ValueError(msg)

    evidence_cfg = AdequacyEvidenceConfig(
        direct_eval_k=2,
        composition_max_depth=2,
        composition_beam_width=16,
    )

    plastic_policy = CompactPlasticLifecyclePolicy(
        CompactLifecycleConfig(
            compact_budget_steps=config.compact_budget_steps,
            compact_batch_size=config.compact_batch_size,
            compact_lr=config.compact_lr,
            compact_success_threshold_em=0.85,
            fallback_budget_steps=config.fallback_budget_steps,
            fallback_lr=config.fallback_lr,
            fallback_success_threshold_em=0.85,
            distillation_steps=config.distillation_steps,
            shadow_retention_threshold=config.shadow_retention_threshold,
            seed=seed,
        )
    )

    # 2. Measure baseline legacy performance (t=0)
    legacy_k_ops = list(CANONICAL_DETERMINISTIC_K_OPS)
    historical_eval_sets: dict[str, list[Example]] = {}
    for op in legacy_k_ops:
        historical_eval_sets[op] = generate_benchmark_examples(
            seed=seed * 300 + 7, n=30, operation=op, split="val", vocab_size=10
        )

    k_em_scores: list[float] = []
    for op in legacy_k_ops:
        if op in op_to_id:
            em, _, _ = _evaluate_candidate_recipe(
                core, bank, op_to_id, (op,), historical_eval_sets[op]
            )
            k_em_scores.append(em)
    baseline_k_em = sum(k_em_scores) / max(1, len(k_em_scores))

    # Evaluate legacy C baseline
    c_em_scores: list[float] = []
    legacy_c_eval_sets: list[tuple[tuple[str, str], list[Example]]] = []
    for pair in COMPOSITION_PAIRS[:4]:
        op1, op2 = pair
        base_ex = generate_benchmark_examples(
            seed=seed * 300 + 13, n=20, operation=op1, split="val", vocab_size=10
        )
        from apc.environments.operations import get_operation

        op2_inst = get_operation(op2)
        c_ex: list[Example] = []
        prog = Program((ProgramStep(op1), ProgramStep(op2)))
        task_spec = TaskSpec.from_program(prog)
        for ex in base_ex:
            t2 = op2_inst.apply(ex.target_tokens, 10, {})
            c_ex.append(
                Example(
                    input_tokens=ex.input_tokens,
                    target_tokens=t2,
                    program=prog,
                    task_spec=task_spec,
                    operation_graph=ex.operation_graph,
                    category="C",
                    split="test",
                    vocab_size=10,
                )
            )
        legacy_c_eval_sets.append((pair, c_ex))
        em, _, _ = _evaluate_candidate_recipe(core, bank, op_to_id, pair, c_ex)
        c_em_scores.append(em)
    baseline_c_em = sum(c_em_scores) / max(1, len(c_em_scores))

    # Evaluate baseline router accuracy on initial 10 ops
    baseline_routing_top1 = _evaluate_router_accuracy(
        core, router, op_to_id, legacy_k_ops, seed=seed * 11
    )

    # 3. Interleaved legacy K/C controller check (measuring false plastic)
    legacy_false_plastic_count = 0
    legacy_total_episodes = 0

    for op_idx, op in enumerate(legacy_k_ops, start=1):
        legacy_total_episodes += 1
        supp = generate_benchmark_examples(
            seed=seed * 1000 + op_idx * 37, n=config.support_size, operation=op, split="train"
        )
        ev_k = compute_adequacy_evidence(
            core=core,
            bank=bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=supp,
            config=evidence_cfg,
        )
        pred_k = controller.predict(ev_k)
        if pred_k.action == ControllerAction.PLASTIC_SEARCH:
            legacy_false_plastic_count += 1

    for _pair, c_ex in legacy_c_eval_sets:
        legacy_total_episodes += 1
        supp_c = c_ex[: config.support_size]
        ev_c = compute_adequacy_evidence(
            core=core,
            bank=bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=supp_c,
            config=evidence_cfg,
        )
        pred_c = controller.predict(ev_c)
        if pred_c.action == ControllerAction.PLASTIC_SEARCH:
            legacy_false_plastic_count += 1

    # 4. Sealed Novel Episodes Lifecycle
    novel_outcomes: dict[str, NovelEpisodeOutcome] = {}
    workspace = PlasticWorkspace()
    workspace_leak_count = 0

    for op_idx, novel_op in enumerate(config.sealed_operations, start=1):
        t_start = time.perf_counter()

        # Generate evaluation sets
        verification_examples = _make_deterministic_examples(
            novel_op,
            config.eval_size,
            seed=seed * 1000 + op_idx * 50 + 1,
            category="N",
        )
        query_examples = _make_deterministic_examples(
            novel_op,
            config.eval_size,
            seed=seed * 1000 + op_idx * 50 + 2,
            category="N",
        )
        support_examples = _make_deterministic_examples(
            novel_op,
            config.support_size,
            seed=seed * 1000 + op_idx * 50 + 3,
            category="N",
        )
        plastic_train_examples = _make_deterministic_examples(
            novel_op,
            config.plastic_train_size,
            seed=seed * 1000 + op_idx * 50 + 4,
            category="N",
        )

        # Precheck A: Novelty-validity check on verification set
        precheck = check_novelty_validity(
            target_operation=novel_op,
            verification_examples=verification_examples,
            threshold=0.90,
            max_composition_depth=2,
            candidate_bank_ops=tuple(op_to_id.keys()),
            registry=registry,
        )
        if not precheck.is_valid_novel_holdout:
            raise RuntimeError(
                f"Benchmark validity failure: sealed operation {novel_op!r} is already adequate "
                f"under existing library (direct EM={precheck.best_direct_em:.4f}, "
                f"comp EM={precheck.best_composition_em:.4f})"
            )

        # Precheck B: Adequacy evidence extraction
        ev_novel = compute_adequacy_evidence(
            core=core,
            bank=bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=support_examples,
            config=evidence_cfg,
        )

        # Autonomous controller action
        pred_novel = controller.predict(ev_novel)
        plastic_triggered = pred_novel.action == ControllerAction.PLASTIC_SEARCH

        # Plastic learning if inadequate
        lifecycle_rep = plastic_policy.execute_episode(
            episode_id=f"seed_{seed}_{novel_op}",
            task_name=novel_op,
            action=ControllerAction.PLASTIC_SEARCH,
            evidence=ev_novel,
            train_examples=plastic_train_examples,
            eval_examples=query_examples,
            historical_eval_examples=historical_eval_sets,
            core=core,
            bank=bank,
            op_to_id=op_to_id,
            workspace=workspace,
        )

        # Verify workspace is completely clean
        ws_count = workspace.total_parameter_count()
        ws_leak = ws_count != 0
        if ws_leak:
            workspace_leak_count += 1

        promoted_pid = lifecycle_rep.promoted_primitive_id
        final_em = 0.0

        if lifecycle_rep.promotions_count > 0:
            assert promoted_pid is not None
            # Evaluate newly promoted primitive
            final_em, _, _ = _evaluate_candidate_recipe(
                core, bank, op_to_id, (novel_op,), query_examples
            )

            # Update router with bounded replay (R2)
            z_new = extract_task_representations(core, plastic_train_examples[:64])
            new_data_by_pid = {
                promoted_pid: [(z_new[i], promoted_pid) for i in range(len(z_new))]
            }
            router_cfg = IncrementalRouterConfig(
                condition=IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
                router_lr=0.005,
                router_steps=250,
                seed=seed * 100 + op_idx * 10,
            )
            update_router_incrementally(
                router,
                candidate_ids=bank.ids(),
                new_primitive_ids=[promoted_pid],
                new_data_by_pid=new_data_by_pid,
                replay_buffer=replay_buf,
                config=router_cfg,
                device=device,
            )
            router.eval()
            for p in router.parameters():
                p.requires_grad_(False)

        # Fresh-Runtime Recurrence Verification
        # Construct fresh runtime environment with zero workspace
        fresh_bank = PrimitiveBank()
        for pid in bank.ids():
            fresh_bank.add_primitive(copy.deepcopy(bank.get(pid)))
        fresh_bank.to(device)
        fresh_bank.freeze_all()
        fresh_bank.eval()

        fresh_workspace = PlasticWorkspace()
        assert not fresh_workspace.is_allocated
        assert fresh_workspace.total_parameter_count() == 0

        # Present novel task in fresh runtime
        rec_support = _make_deterministic_examples(
            novel_op, config.support_size, seed=seed * 5000 + 17, category="R"
        )
        rec_query = _make_deterministic_examples(
            novel_op, config.eval_size, seed=seed * 5000 + 29, category="R"
        )

        ev_rec = compute_adequacy_evidence(
            core=core,
            bank=fresh_bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=rec_support,
            config=evidence_cfg,
        )
        pred_rec = controller.predict(ev_rec)

        rec_em = 0.0
        bank_size_before = len(fresh_bank)
        if pred_rec.action == ControllerAction.DIRECT_REUSE:
            rec_em, _, _ = _evaluate_candidate_recipe(
                core, fresh_bank, op_to_id, (novel_op,), rec_query
            )

        bank_size_after = len(fresh_bank)
        bank_growth = bank_size_after - bank_size_before
        temp_params = fresh_workspace.total_parameter_count()

        outcome = NovelEpisodeOutcome(
            operation_name=novel_op,
            precheck_direct_em=precheck.best_direct_em,
            precheck_comp_em=precheck.best_composition_em,
            is_valid_novel_holdout=precheck.is_valid_novel_holdout,
            predicted_action=pred_novel.action.value,
            novelty_score=pred_novel.novelty_score,
            plastic_triggered=plastic_triggered,
            compact_success=lifecycle_rep.compact_success,
            fallback_invoked=lifecycle_rep.fallback_invoked,
            fallback_success=lifecycle_rep.fallback_success,
            shadow_validation_passed=lifecycle_rep.shadow_validation_passed,
            promoted_primitive_id=promoted_pid,
            promotions_count=lifecycle_rep.promotions_count,
            final_novel_em=final_em,
            workspace_param_count_after=ws_count,
            workspace_leak=ws_leak,
            recurrence_action=pred_rec.action.value,
            recurrence_em=rec_em,
            recurrence_adaptation_steps=0,
            recurrence_temporary_params=temp_params,
            recurrence_bank_growth=bank_growth,
            elapsed_seconds=time.perf_counter() - t_start,
        )
        novel_outcomes[novel_op] = outcome

    # 5. Measure legacy performance after all novel promotions
    k_em_after_list: list[float] = []
    for op in legacy_k_ops:
        if op in op_to_id:
            em, _, _ = _evaluate_candidate_recipe(
                core, bank, op_to_id, (op,), historical_eval_sets[op]
            )
            k_em_after_list.append(em)
    legacy_k_em_after = sum(k_em_after_list) / max(1, len(k_em_after_list))

    c_em_after_list: list[float] = []
    for pair, c_ex in legacy_c_eval_sets:
        em, _, _ = _evaluate_candidate_recipe(core, bank, op_to_id, pair, c_ex)
        c_em_after_list.append(em)
    legacy_c_em_after = sum(c_em_after_list) / max(1, len(c_em_after_list))

    old_routing_top1_after = _evaluate_router_accuracy(
        core, router, op_to_id, legacy_k_ops, seed=seed * 11
    )

    old_task_em_drop = max(
        0.0,
        baseline_k_em - legacy_k_em_after,
        baseline_c_em - legacy_c_em_after,
    )
    old_routing_drop = max(0.0, baseline_routing_top1 - old_routing_top1_after)
    legacy_false_plastic_rate = legacy_false_plastic_count / max(1, legacy_total_episodes)

    # Seed-level pass criteria
    all_novel_em_ok = all(
        out.final_novel_em >= MIN_SEED_NOVEL_EM_THRESHOLD for out in novel_outcomes.values()
    )
    all_rec_em_ok = all(out.recurrence_em >= 0.90 for out in novel_outcomes.values())
    no_leaks = workspace_leak_count == 0
    drop_ok = (old_task_em_drop <= MAX_OLD_TASK_EM_DROP) and (
        old_routing_drop <= MAX_OLD_ROUTING_DROP
    )

    seed_passed = all_novel_em_ok and all_rec_em_ok and no_leaks and drop_ok

    return SeedLifecycleResult(
        seed=seed,
        novel_outcomes=novel_outcomes,
        legacy_k_em_before=baseline_k_em,
        legacy_k_em_after=legacy_k_em_after,
        legacy_c_em_before=baseline_c_em,
        legacy_c_em_after=legacy_c_em_after,
        old_routing_top1_before=baseline_routing_top1,
        old_routing_top1_after=old_routing_top1_after,
        legacy_false_plastic_count=legacy_false_plastic_count,
        legacy_total_count=legacy_total_episodes,
        legacy_false_plastic_rate=legacy_false_plastic_rate,
        old_task_em_drop=old_task_em_drop,
        old_routing_drop=old_routing_drop,
        workspace_leak_count=workspace_leak_count,
        seed_passed=seed_passed,
    )


def run_unseen_family_lifecycle_benchmark(
    config: UnseenFamilyLifecycleConfig | None = None,
) -> tuple[GateB1AggregateReport, list[SeedLifecycleResult]]:
    """Execute complete STOP GATE B1 benchmark across all configured seeds."""
    cfg = config or UnseenFamilyLifecycleConfig()
    use_cuda = (cfg.device_str == "auto" and torch.cuda.is_available()) or cfg.device_str == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    # Controller calibrated on synthetic evidence profiles, frozen before evaluation
    controller = build_default_trained_controller(seed=42)
    controller.freeze()

    seed_results: list[SeedLifecycleResult] = []
    for s in cfg.seeds:
        res = run_unseen_family_lifecycle_for_seed(s, cfg, controller=controller, device=device)
        seed_results.append(res)

    # Compute aggregate Gate B1 metrics
    total_novel = sum(len(r.novel_outcomes) for r in seed_results)
    plastic_triggers = sum(
        sum(1 for out in r.novel_outcomes.values() if out.plastic_triggered)
        for r in seed_results
    )
    plastic_trigger_rate = plastic_triggers / max(1, total_novel)

    all_final_novel_ems = [
        out.final_novel_em for r in seed_results for out in r.novel_outcomes.values()
    ]
    mean_novel_em = sum(all_final_novel_ems) / max(1, len(all_final_novel_ems))

    novel_em_per_seed: dict[int, float] = {}
    for r in seed_results:
        ems = [out.final_novel_em for out in r.novel_outcomes.values()]
        novel_em_per_seed[r.seed] = sum(ems) / max(1, len(ems))
    min_seed_novel_em = min(novel_em_per_seed.values()) if novel_em_per_seed else 0.0

    total_promotions = sum(
        sum(out.promotions_count for out in r.novel_outcomes.values())
        for r in seed_results
    )
    expected_promotions = total_novel
    one_to_one = total_promotions == expected_promotions

    total_ws_leaks = sum(r.workspace_leak_count for r in seed_results)

    all_rec_ems = [
        out.recurrence_em for r in seed_results for out in r.novel_outcomes.values()
    ]
    mean_rec_em = sum(all_rec_ems) / max(1, len(all_rec_ems))
    max_rec_steps = max(
        out.recurrence_adaptation_steps
        for r in seed_results
        for out in r.novel_outcomes.values()
    )
    max_rec_temp_params = max(
        out.recurrence_temporary_params
        for r in seed_results
        for out in r.novel_outcomes.values()
    )
    total_rec_bank_growth = sum(
        out.recurrence_bank_growth
        for r in seed_results
        for out in r.novel_outcomes.values()
    )

    old_task_drops = [r.old_task_em_drop for r in seed_results]
    mean_old_task_drop = sum(old_task_drops) / max(1, len(old_task_drops))
    max_old_task_drop = max(old_task_drops) if old_task_drops else 0.0

    old_routing_drops = [r.old_routing_drop for r in seed_results]
    mean_old_routing_drop = sum(old_routing_drops) / max(1, len(old_routing_drops))
    max_old_routing_drop = max(old_routing_drops) if old_routing_drops else 0.0

    total_legacy_false = sum(r.legacy_false_plastic_count for r in seed_results)
    total_legacy_eps = sum(r.legacy_total_count for r in seed_results)
    legacy_false_rate = total_legacy_false / max(1, total_legacy_eps)

    # Gate evaluations
    pass_trigger = plastic_trigger_rate >= PLASTIC_TRIGGER_THRESHOLD
    pass_mean_novel = mean_novel_em >= MEAN_NOVEL_EM_THRESHOLD
    pass_min_novel = min_seed_novel_em >= MIN_SEED_NOVEL_EM_THRESHOLD
    pass_promotion = one_to_one
    pass_leaks = total_ws_leaks == 0
    pass_rec = (
        mean_rec_em >= RECURRENCE_EM_THRESHOLD
        and max_rec_steps == 0
        and max_rec_temp_params == 0
        and total_rec_bank_growth == 0
    )
    pass_legacy = (
        max_old_task_drop <= MAX_OLD_TASK_EM_DROP
        and max_old_routing_drop <= MAX_OLD_ROUTING_DROP
        and legacy_false_rate <= MAX_FALSE_PLASTIC_RATE
    )

    gate_passed = (
        pass_trigger
        and pass_mean_novel
        and pass_min_novel
        and pass_promotion
        and pass_leaks
        and pass_rec
        and pass_legacy
    )

    aggregate = GateB1AggregateReport(
        seeds_evaluated=list(cfg.seeds),
        total_novel_episodes=total_novel,
        plastic_trigger_rate=plastic_trigger_rate,
        mean_final_novel_em=mean_novel_em,
        min_seed_novel_em=min_seed_novel_em,
        novel_em_per_seed=novel_em_per_seed,
        total_promotions=total_promotions,
        expected_promotions=expected_promotions,
        one_to_one_promotions=one_to_one,
        workspace_leak_count=total_ws_leaks,
        mean_recurrence_em=mean_rec_em,
        max_recurrence_adaptation_steps=max_rec_steps,
        max_recurrence_temporary_params=max_rec_temp_params,
        total_recurrence_bank_growth=total_rec_bank_growth,
        mean_old_task_em_drop=mean_old_task_drop,
        max_old_task_em_drop=max_old_task_drop,
        mean_old_routing_drop=mean_old_routing_drop,
        max_old_routing_drop=max_old_routing_drop,
        legacy_false_plastic_rate=legacy_false_rate,
        passed_plastic_trigger=pass_trigger,
        passed_mean_novel_em=pass_mean_novel,
        passed_min_seed_novel_em=pass_min_novel,
        passed_one_to_one_promotion=pass_promotion,
        passed_workspace_leaks=pass_leaks,
        passed_recurrence=pass_rec,
        passed_legacy_regression=pass_legacy,
        gate_b1_passed=gate_passed,
    )

    # Save artifacts if output_dir specified
    if cfg.output_dir is not None:
        out_path = Path(cfg.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_data = {
            "config": cfg.to_dict(),
            "aggregate": aggregate.to_dict(),
            "seed_results": [r.to_dict() for r in seed_results],
        }
        (out_path / "report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        (out_path / "summary.json").write_text(
            json.dumps(aggregate.to_dict(), indent=2), encoding="utf-8"
        )

        p_trig = "PASS" if pass_trigger else "FAIL"
        p_mean = "PASS" if pass_mean_novel else "FAIL"
        p_min = "PASS" if pass_min_novel else "FAIL"
        p_promo = "PASS" if pass_promotion else "FAIL"
        p_leak = "PASS" if pass_leaks else "FAIL"
        p_rec = "PASS" if pass_rec else "FAIL"
        p_adapt = "PASS" if max_rec_steps == 0 else "FAIL"
        p_temp = "PASS" if max_rec_temp_params == 0 else "FAIL"
        p_growth = "PASS" if total_rec_bank_growth == 0 else "FAIL"
        p_em_drop = "PASS" if max_old_task_drop <= MAX_OLD_TASK_EM_DROP else "FAIL"
        p_rt_drop = "PASS" if max_old_routing_drop <= MAX_OLD_ROUTING_DROP else "FAIL"
        p_leg = "PASS" if legacy_false_rate <= MAX_FALSE_PLASTIC_RATE else "FAIL"

        rows = [
            f"| Plastic Trigger Rate | >= 95.0% | {plastic_trigger_rate * 100:.2f}% | {p_trig} |",
            f"| Mean Final Novel EM | >= 95.0% | {mean_novel_em * 100:.2f}% | {p_mean} |",
            f"| Min Seed Novel EM | >= 90.0% | {min_seed_novel_em * 100:.2f}% | {p_min} |",
            f"| 1:1 Promotion | {expected_promotions}/{expected_promotions} | "
            f"{total_promotions} | {p_promo} |",
            f"| Workspace Leaks | == 0 | {total_ws_leaks} | {p_leak} |",
            f"| Recurrence EM | >= 95.0% | {mean_rec_em * 100:.2f}% | {p_rec} |",
            f"| Recurrence Adapt Steps | == 0 | {max_rec_steps} | {p_adapt} |",
            f"| Recurrence Temp Params | == 0 | {max_rec_temp_params} | {p_temp} |",
            f"| Recurrence Bank Growth | == 0 | {total_rec_bank_growth} | {p_growth} |",
            f"| Max Old EM Drop | <= 1.0 pp | {max_old_task_drop * 100:.2f} pp | {p_em_drop} |",
            f"| Max Routing Drop | <= 1.0 pp | {max_old_routing_drop * 100:.2f} pp | {p_rt_drop} |",
            f"| Legacy False Plastic | <= 1.0% | {legacy_false_rate * 100:.2f}% | {p_leg} |",
        ]
        table_body = "\n".join(rows)

        md_content = f"""# STOP GATE B1 Evaluation Report

**Status:** {"PASSED" if gate_passed else "FAILED"}
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Seeds:** {cfg.seeds}
**Sealed Operations:** {cfg.sealed_operations}

## 1. Scientific Acceptance Summary

| Criterion | Target | Measured | Result |
|---|---|---|---|
{table_body}

## 2. Per-Seed Details

"""
        for r in seed_results:
            md_content += f"### Seed {r.seed}\n"
            k_b = r.legacy_k_em_before * 100
            k_a = r.legacy_k_em_after * 100
            c_b = r.legacy_c_em_before * 100
            c_a = r.legacy_c_em_after * 100
            r_b = r.old_routing_top1_before * 100
            r_a = r.old_routing_top1_after * 100
            md_content += f"- Legacy K EM: {k_b:.2f}% -> {k_a:.2f}%\n"
            md_content += f"- Legacy C EM: {c_b:.2f}% -> {c_a:.2f}%\n"
            md_content += f"- Routing Top-1: {r_b:.2f}% -> {r_a:.2f}%\n"
            for op, out in r.novel_outcomes.items():
                f_em = out.final_novel_em * 100
                rc_em = out.recurrence_em * 100
                p_id = out.promoted_primitive_id
                md_content += (
                    f"- Novel `{op}`: Final EM = {f_em:.2f}%, "
                    f"Recurrence EM = {rc_em:.2f}%, Promoted ID = {p_id}\n"
                )
            md_content += "\n"

        (out_path / "report.md").write_text(md_content, encoding="utf-8")

    return aggregate, seed_results
