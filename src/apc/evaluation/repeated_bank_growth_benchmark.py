"""Repeated semantic bank-growth stress benchmark (Phase A.2 Task A2-C009).

This benchmark joins the real compact-first N -> consolidation lifecycle from
A2-C008 with the bounded-replay incremental router from A2-C003.  Starting from
the validated ten-operation bank, it installs six executable operations one at a
time and records routing, functional retention, recurrence, and sparse execution
after every insertion.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.sequential_closed_loop_benchmark import (
    _evaluate_candidate_recipe,
    _setup_initial_environment,
    generate_episode_examples,
)
from apc.meta.adequacy import AdequacyEvidenceConfig, compute_adequacy_evidence
from apc.meta.episode_log import ControllerAction
from apc.meta.learned_controller import (
    LearnedAdequacyController,
    build_default_trained_controller,
)
from apc.plastic.lifecycle import CompactLifecycleConfig, CompactPlasticLifecyclePolicy
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    evaluate_router_accuracy,
    update_router_incrementally,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    PointwisePrimitive,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
GROWTH_STAGES: Final[tuple[int, ...]] = (10, 12, 14, 16)

ROUTING_THRESHOLD: Final[float] = 0.95
OLD_ROUTING_DROP_THRESHOLD: Final[float] = 0.02
WORST_OLD_ROUTING_DROP_THRESHOLD: Final[float] = 0.05
PERFORMANCE_DROP_THRESHOLD: Final[float] = 0.02
RECURRENCE_REUSE_THRESHOLD: Final[float] = 0.90


@dataclass(frozen=True)
class RepeatedBankGrowthConfig:
    """Explicit configuration for the C009 semantic growth stress run."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    support_size: int = 16
    eval_size: int = 32
    routing_eval_size: int = 100
    plastic_train_size: int = 160
    compact_budget_steps: int = 800
    fallback_budget_steps: int = 350
    distillation_steps: int = 300
    compact_lr: float = 2e-3
    router_steps: int = 250
    router_lr: float = 0.005
    dev_seed: int = 42
    device_str: str = "auto"
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["seeds"] = list(self.seeds)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        result["growth_stages"] = list(GROWTH_STAGES)
        return result


@dataclass(frozen=True)
class StageGrowthSnapshot:
    """Measurements taken immediately after one successful semantic insertion."""

    seed: int
    insertion_index: int
    bank_size_before: int
    bank_size: int
    new_operation: str
    controller_action: str
    promotion_count: int
    router_update_condition: str
    overall_routing: float
    old_routing_top1: float
    new_routing_top1: float
    old_routing_mean_drop: float
    worst_old_routing_drop: float
    canonical_performance: float
    canonical_performance_drop: float
    consolidated_performance: float
    consolidated_performance_drop: float
    recurrence_router_top1: float
    recurrence_reuse: float
    recurrence_em: float
    unselected_forward_calls: int
    workspace_param_count: int
    temporary_peak_params: int
    stage_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SeedGrowthResult:
    """Complete insertion trajectory and gate verdict for one decision seed."""

    seed: int
    initial_bank_size: int
    final_bank_size: int
    snapshots: list[StageGrowthSnapshot]
    acceptance: dict[str, bool]
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "initial_bank_size": self.initial_bank_size,
            "final_bank_size": self.final_bank_size,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "acceptance": self.acceptance,
            "all_passed": self.all_passed,
        }


def evaluate_growth_acceptance(snapshots: list[StageGrowthSnapshot]) -> dict[str, bool]:
    """Evaluate all C009 gate criteria over a single insertion trajectory."""
    if not snapshots:
        return {
            "final_semantic_bank_16": False,
            "six_successful_promotions": False,
            "overall_routing": False,
            "old_routing_mean_drop": False,
            "canonical_performance_drop": False,
            "consolidated_task_drop": False,
            "recurrence_reuse": False,
            "new_class_routing": False,
            "worst_old_routing_drop": False,
            "unselected_forward_calls": False,
            "no_workspace_leak": False,
        }

    final = snapshots[-1]
    return {
        "final_semantic_bank_16": final.bank_size == 16,
        "six_successful_promotions": (
            len(snapshots) == 6 and all(s.promotion_count == 1 for s in snapshots)
        ),
        "overall_routing": final.overall_routing >= ROUTING_THRESHOLD,
        "old_routing_mean_drop": max(s.old_routing_mean_drop for s in snapshots)
        <= OLD_ROUTING_DROP_THRESHOLD,
        "canonical_performance_drop": max(s.canonical_performance_drop for s in snapshots)
        <= PERFORMANCE_DROP_THRESHOLD,
        "consolidated_task_drop": max(s.consolidated_performance_drop for s in snapshots)
        <= PERFORMANCE_DROP_THRESHOLD,
        "recurrence_reuse": all(
            s.recurrence_reuse >= RECURRENCE_REUSE_THRESHOLD for s in snapshots
        ),
        "new_class_routing": all(s.new_routing_top1 >= ROUTING_THRESHOLD for s in snapshots),
        "worst_old_routing_drop": max(s.worst_old_routing_drop for s in snapshots)
        <= WORST_OLD_ROUTING_DROP_THRESHOLD,
        "unselected_forward_calls": all(s.unselected_forward_calls == 0 for s in snapshots),
        "no_workspace_leak": all(s.workspace_param_count == 0 for s in snapshots),
    }


def aggregate_growth_results(results: list[SeedGrowthResult]) -> dict[str, Any]:
    """Aggregate multi-seed results without hiding a failed seed or insertion."""
    if not results:
        raise ValueError("At least one seed result is required")
    snapshots = [snapshot for result in results for snapshot in result.snapshots]
    final_snapshots = [result.snapshots[-1] for result in results if result.snapshots]
    recurrence_values = [s.recurrence_reuse for s in final_snapshots]
    acceptance_names = sorted({name for result in results for name in result.acceptance})
    acceptance = {
        name: all(result.acceptance.get(name, False) for result in results)
        for name in acceptance_names
    }
    return {
        "num_seeds": len(results),
        "total_insertions": len(snapshots),
        "mean_final_overall_routing": (
            sum(s.overall_routing for s in final_snapshots) / len(final_snapshots)
            if final_snapshots
            else 0.0
        ),
        "max_old_routing_mean_drop": max(
            (s.old_routing_mean_drop for s in snapshots), default=1.0
        ),
        "max_canonical_performance_drop": max(
            (s.canonical_performance_drop for s in snapshots), default=1.0
        ),
        "max_consolidated_task_drop": max(
            (s.consolidated_performance_drop for s in snapshots), default=1.0
        ),
        "mean_recurrence_reuse": (
            sum(recurrence_values) / len(recurrence_values) if recurrence_values else 0.0
        ),
        "min_new_class_routing": min(
            (s.new_routing_top1 for s in snapshots), default=0.0
        ),
        "max_worst_old_routing_drop": max(
            (s.worst_old_routing_drop for s in snapshots), default=1.0
        ),
        "total_unselected_forward_calls": sum(s.unselected_forward_calls for s in snapshots),
        "total_workspace_leaks": sum(s.workspace_param_count != 0 for s in snapshots),
        "acceptance": acceptance,
        "overall_passed": len(results) >= 5 and all(acceptance.values()),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _performance_for_operations(
    core: Any,
    bank: Any,
    op_to_id: dict[str, int],
    eval_sets: dict[str, list[Any]],
    operations: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for operation in operations:
        if operation in op_to_id:
            score, _, _ = _evaluate_candidate_recipe(
                core, bank, op_to_id, (operation,), eval_sets[operation]
            )
            scores[operation] = score
    return scores


def _strict_sparse_call_check(
    core: Any,
    bank: Any,
    router: Any,
    candidate_ids: list[int],
    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]],
) -> int:
    """Route a batch and verify that only selected primitive modules execute."""
    for primitive_id in candidate_ids:
        bank.get(primitive_id).forward_call_count = 0

    z_batch = torch.stack([eval_z_by_pid[pid][0][0] for pid in candidate_ids]).to(core.device)
    with torch.no_grad():
        selected_ids = router(z_batch, candidate_ids).selected_ids[:, 0].tolist()
        selected_unique = sorted(set(int(pid) for pid in selected_ids))
        dummy_h = torch.randn(1, 8, core.model.config.d_model, device=core.device)
        for primitive_id in selected_unique:
            primitive = bank.get(primitive_id)
            if isinstance(primitive, PointwisePrimitive):
                primitive(dummy_h)
            elif isinstance(
                primitive,
                (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
            ):
                primitive(dummy_h, content_lengths=[8], output_lengths=[8], argument_values=None)
            else:  # pragma: no cover - guards future primitive types
                raise TypeError(f"Unsupported primitive type: {type(primitive).__name__}")

    selected_set = set(selected_unique)
    return sum(
        bank.get(pid).forward_call_count for pid in candidate_ids if pid not in selected_set
    )


def run_repeated_bank_growth_for_seed(
    seed: int,
    *,
    config: RepeatedBankGrowthConfig,
    controller: LearnedAdequacyController,
) -> SeedGrowthResult:
    """Run six autonomous N -> consolidation -> R2 update cycles for one seed."""
    set_seed(seed)
    use_cuda = (config.device_str == "auto" and torch.cuda.is_available()) or (
        config.device_str == "cuda"
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    core, bank, router, op_to_id, replay_buffer = _setup_initial_environment(seed, device)
    workspace = PlasticWorkspace()
    lifecycle = CompactPlasticLifecyclePolicy(
        CompactLifecycleConfig(
            compact_budget_steps=config.compact_budget_steps,
            compact_lr=config.compact_lr,
            compact_success_threshold_em=0.85,
            fallback_budget_steps=config.fallback_budget_steps,
            fallback_lr=config.compact_lr,
            fallback_success_threshold_em=0.85,
            distillation_steps=config.distillation_steps,
            shadow_retention_threshold=0.85,
            eval_batch_size=32,
            seed=seed,
        )
    )
    evidence_config = AdequacyEvidenceConfig(
        direct_eval_k=2,
        composition_max_depth=2,
        composition_beam_width=16,
    )

    canonical_ops = list(INITIAL_10_OPERATIONS)
    performance_eval_sets: dict[str, list[Any]] = {
        op: generate_benchmark_examples(
            seed=seed * 5000 + index * 17 + 11,
            n=config.eval_size,
            operation=op,
            split="test",
            vocab_size=10,
        )
        for index, op in enumerate(canonical_ops)
    }
    canonical_baseline = _performance_for_operations(
        core, bank, op_to_id, performance_eval_sets, canonical_ops
    )

    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for index, operation in enumerate(canonical_ops):
        primitive_id = op_to_id[operation]
        examples = generate_benchmark_examples(
            seed=seed * 7000 + index * 19 + 23,
            n=config.routing_eval_size,
            operation=operation,
            split="test",
            vocab_size=10,
        )
        z_task = extract_task_representations(core, examples)
        eval_z_by_pid[primitive_id] = [(z_task[i], primitive_id) for i in range(len(examples))]

    active_ids = [op_to_id[op] for op in canonical_ops]
    initial_routing = evaluate_router_accuracy(router, active_ids, eval_z_by_pid, device=device)
    routing_baseline = {pid: initial_routing[pid]["top1"] for pid in active_ids}
    consolidated_ops: list[str] = []
    consolidated_baseline: dict[str, float] = {}
    snapshots: list[StageGrowthSnapshot] = []

    for insertion_index, operation in enumerate(PHASE_A2_INCREMENTAL_NEW_OPERATIONS, start=1):
        bank_size_before = len(bank)
        old_ids = list(active_ids)
        support_examples = generate_episode_examples(
            program=_single_step_program(operation),
            category="N",
            vocab_size=10,
            n_examples=config.support_size,
            seed=seed * 10000 + insertion_index * 101 + 7,
        )
        eval_examples = generate_episode_examples(
            program=_single_step_program(operation),
            category="N",
            vocab_size=10,
            n_examples=config.eval_size,
            seed=seed * 10000 + insertion_index * 101 + 19,
        )
        train_examples = generate_episode_examples(
            program=_single_step_program(operation),
            category="N",
            vocab_size=10,
            n_examples=config.plastic_train_size,
            seed=seed * 10000 + insertion_index * 101 + 43,
        )

        evidence = compute_adequacy_evidence(
            core=core,
            bank=bank,
            router=router,
            op_to_id=op_to_id,
            support_examples=support_examples,
            config=evidence_config,
        )
        prediction = controller.predict(evidence)
        if prediction.action != ControllerAction.PLASTIC_SEARCH:
            break

        historical_sets = {
            **{op: performance_eval_sets[op] for op in canonical_ops},
            **{op: performance_eval_sets[op] for op in consolidated_ops},
        }
        lifecycle_report = lifecycle.execute_episode(
            episode_id=f"c009_seed_{seed}_insert_{insertion_index}_{operation}",
            task_name=operation,
            action=prediction.action,
            evidence=evidence,
            train_examples=train_examples,
            eval_examples=eval_examples,
            historical_eval_examples=historical_sets,
            core=core,
            bank=bank,
            op_to_id=op_to_id,
            workspace=workspace,
        )
        if lifecycle_report.promotions_count != 1 or lifecycle_report.promoted_primitive_id is None:
            break

        new_pid = lifecycle_report.promoted_primitive_id
        z_train = extract_task_representations(core, train_examples)
        new_training = {new_pid: [(z_train[i], new_pid) for i in range(len(train_examples))]}
        # C003's R2 condition optimizes the complete candidate-key set while keeping
        # query_proj frozen.  The C008 setup freezes the router for inference, so
        # explicitly reopen all keys here before this bounded incremental update.
        for primitive_id in router.ids():
            router.key_parameter(primitive_id).requires_grad_(True)
        update_router_incrementally(
            router,
            candidate_ids=bank.ids(),
            new_primitive_ids=[new_pid],
            new_data_by_pid=new_training,
            replay_buffer=replay_buffer,
            config=IncrementalRouterConfig(
                condition=IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
                router_lr=config.router_lr,
                router_steps=config.router_steps,
                seed=seed + insertion_index * 10,
            ),
            device=device,
        )
        router.eval()
        for parameter in router.parameters():
            parameter.requires_grad_(False)

        active_ids.append(new_pid)
        performance_eval_sets[operation] = eval_examples
        new_z = extract_task_representations(core, eval_examples)
        eval_z_by_pid[new_pid] = [(new_z[i], new_pid) for i in range(len(eval_examples))]

        routing = evaluate_router_accuracy(router, active_ids, eval_z_by_pid, device=device)
        old_top1 = [routing[pid]["top1"] for pid in old_ids]
        old_drops = [max(0.0, routing_baseline[pid] - routing[pid]["top1"]) for pid in old_ids]
        overall_routing = _mean([routing[pid]["top1"] for pid in active_ids])
        new_routing = routing[new_pid]["top1"]

        canonical_current = _performance_for_operations(
            core, bank, op_to_id, performance_eval_sets, canonical_ops
        )
        canonical_drop = _mean(
            [max(0.0, canonical_baseline[op] - canonical_current[op]) for op in canonical_ops]
        )

        new_perf, _, _ = _evaluate_candidate_recipe(
            core, bank, op_to_id, (operation,), eval_examples
        )
        consolidated_baseline[operation] = new_perf
        consolidated_ops.append(operation)
        consolidated_current = _performance_for_operations(
            core, bank, op_to_id, performance_eval_sets, consolidated_ops
        )
        consolidated_drop = _mean(
            [
                max(0.0, consolidated_baseline[op] - consolidated_current[op])
                for op in consolidated_ops
            ]
        )

        # Re-evaluate every consolidated task after every insertion.  The final
        # snapshot therefore measures recurrence over the complete six-task set,
        # rather than only the most recently installed class.
        recurrence_reuse_values: list[float] = []
        recurrence_em_values: list[float] = []
        recurrence_routing_values: list[float] = []
        id_to_op = {pid: op for op, pid in op_to_id.items()}
        for recurrence_index, recurrence_operation in enumerate(consolidated_ops):
            recurrence_seed = (
                seed * 10000 + insertion_index * 1000 + recurrence_index * 37
            )
            recurrence_support = generate_episode_examples(
                program=_single_step_program(recurrence_operation),
                category="R",
                vocab_size=10,
                n_examples=config.support_size,
                seed=recurrence_seed + 59,
            )
            recurrence_eval = generate_episode_examples(
                program=_single_step_program(recurrence_operation),
                category="R",
                vocab_size=10,
                n_examples=config.eval_size,
                seed=recurrence_seed + 71,
            )
            recurrence_evidence = compute_adequacy_evidence(
                core=core,
                bank=bank,
                router=router,
                op_to_id=op_to_id,
                support_examples=recurrence_support,
                config=evidence_config,
            )
            recurrence_prediction = controller.predict(recurrence_evidence)
            reused = recurrence_prediction.action == ControllerAction.DIRECT_REUSE
            recurrence_reuse_values.append(float(reused))
            recurrence_em_value = 0.0
            if reused:
                direct_pid = recurrence_evidence.direct_primitive_id
                direct_op = id_to_op.get(direct_pid) if direct_pid is not None else None
                if direct_op is not None:
                    recurrence_em_value, _, _ = _evaluate_candidate_recipe(
                        core, bank, op_to_id, (direct_op,), recurrence_eval
                    )
            recurrence_em_values.append(recurrence_em_value)
            recurrence_pid = op_to_id[recurrence_operation]
            recurrence_z = extract_task_representations(core, recurrence_eval)
            recurrence_routing_values.append(
                evaluate_router_accuracy(
                    router,
                    active_ids,
                    {
                        recurrence_pid: [
                            (recurrence_z[i], recurrence_pid)
                            for i in range(len(recurrence_eval))
                        ]
                    },
                    device=device,
                )[recurrence_pid]["top1"]
            )
        recurrence_reuse = _mean(recurrence_reuse_values)
        recurrence_em = _mean(recurrence_em_values)
        recurrence_routing = _mean(recurrence_routing_values)
        unselected_calls = _strict_sparse_call_check(
            core, bank, router, active_ids, eval_z_by_pid
        )
        workspace_params = workspace.total_parameter_count()

        snapshot_values = {
            "new_routing": new_routing,
            "old_mean_drop": _mean(old_drops),
            "worst_old_drop": max(old_drops, default=0.0),
            "canonical_drop": canonical_drop,
            "consolidated_drop": consolidated_drop,
        }
        stage_passed = (
            overall_routing >= ROUTING_THRESHOLD
            and snapshot_values["new_routing"] >= ROUTING_THRESHOLD
            and snapshot_values["old_mean_drop"] <= OLD_ROUTING_DROP_THRESHOLD
            and snapshot_values["worst_old_drop"] <= WORST_OLD_ROUTING_DROP_THRESHOLD
            and snapshot_values["canonical_drop"] <= PERFORMANCE_DROP_THRESHOLD
            and snapshot_values["consolidated_drop"] <= PERFORMANCE_DROP_THRESHOLD
            and recurrence_reuse >= RECURRENCE_REUSE_THRESHOLD
            and unselected_calls == 0
            and workspace_params == 0
        )
        snapshots.append(
            StageGrowthSnapshot(
                seed=seed,
                insertion_index=insertion_index,
                bank_size_before=bank_size_before,
                bank_size=len(bank),
                new_operation=operation,
                controller_action=prediction.action.value,
                promotion_count=lifecycle_report.promotions_count,
                router_update_condition=IncrementalUpdateCondition.R2_BOUNDED_REPLAY.value,
                overall_routing=overall_routing,
                old_routing_top1=_mean(old_top1),
                new_routing_top1=new_routing,
                old_routing_mean_drop=snapshot_values["old_mean_drop"],
                worst_old_routing_drop=snapshot_values["worst_old_drop"],
                canonical_performance=_mean(list(canonical_current.values())),
                canonical_performance_drop=snapshot_values["canonical_drop"],
                consolidated_performance=_mean(list(consolidated_current.values())),
                consolidated_performance_drop=snapshot_values["consolidated_drop"],
                recurrence_router_top1=recurrence_routing,
                recurrence_reuse=recurrence_reuse,
                recurrence_em=recurrence_em,
                unselected_forward_calls=unselected_calls,
                workspace_param_count=workspace_params,
                temporary_peak_params=lifecycle_report.temporary_peak_params,
                stage_passed=stage_passed,
            )
        )
        routing_baseline.update({pid: routing[pid]["top1"] for pid in active_ids})

    acceptance = evaluate_growth_acceptance(snapshots)
    result = SeedGrowthResult(
        seed=seed,
        initial_bank_size=10,
        final_bank_size=len(bank),
        snapshots=snapshots,
        acceptance=acceptance,
        all_passed=all(acceptance.values()),
    )
    if config.output_dir:
        seed_dir = config.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bank.state_dict(), seed_dir / "primitive_bank_16.pt")
        torch.save(router.state_dict(), seed_dir / "router_16.pt")
    return result


def _single_step_program(operation: str) -> Any:
    """Construct a one-operation Program while keeping imports local and explicit."""
    from apc.environments.generator import Program, ProgramStep

    return Program(steps=(ProgramStep(operation),))


def run_repeated_bank_growth_benchmark(config: RepeatedBankGrowthConfig) -> dict[str, Any]:
    """Execute the five-seed A2-C009 STOP-gate experiment and save artifacts."""
    started = time.perf_counter()
    controller = build_default_trained_controller(seed=config.dev_seed)
    controller.freeze()
    results: list[SeedGrowthResult] = []
    for seed in config.seeds:
        result = run_repeated_bank_growth_for_seed(
            seed, config=config, controller=controller
        )
        results.append(result)
        print(
            f"Seed {seed}: bank={result.final_bank_size}, "
            f"insertions={len(result.snapshots)}, passed={result.all_passed}"
        )

    aggregate = aggregate_growth_results(results)
    report = {
        "benchmark": "Phase A.2 Task A2-C009 Repeated Bank-Growth Stress",
        "config": config.to_dict(),
        "seeds": list(config.seeds),
        "growth_stages": list(GROWTH_STAGES),
        "router_update_condition": IncrementalUpdateCondition.R2_BOUNDED_REPLAY.value,
        "aggregate_metrics": aggregate,
        "per_seed_results": [result.to_dict() for result in results],
        "elapsed_seconds": time.perf_counter() - started,
        "overall_passed": aggregate["overall_passed"],
    }
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(config.output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)
        controller.save(config.output_dir / "controller.json")
        with open(config.output_dir / "stages.jsonl", "w", encoding="utf-8") as handle:
            for result in results:
                for snapshot in result.snapshots:
                    handle.write(json.dumps(snapshot.to_dict()) + "\n")
        with open(config.output_dir / "report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        _write_markdown_report(config.output_dir / "BENCHMARK_REPORT.md", report)
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    aggregate = report["aggregate_metrics"]
    acceptance = aggregate["acceptance"]
    status = "PASSED" if report["overall_passed"] else "FAILED"
    verdict = {name: "PASS" if passed else "FAIL" for name, passed in acceptance.items()}
    lines = [
        "# Phase A.2 Task A2-C009: Repeated Bank-Growth Stress",
        "",
        f"- STOP gate: **{status}**",
        f"- Seeds: {report['seeds']}",
        f"- Growth milestones: {report['growth_stages']}",
        f"- Incremental router condition: {report['router_update_condition']} only",
        f"- Total insertions: {aggregate['total_insertions']}",
        f"- Elapsed: {report['elapsed_seconds']:.2f}s",
        "",
        "## Acceptance",
        "",
        "| Criterion | Target | Achieved | Verdict |",
        "|---|---:|---:|---|",
        (
            "| Final overall routing | >=95% | "
            f"{aggregate['mean_final_overall_routing']:.2%} | {verdict['overall_routing']} |"
        ),
        (
            "| Old-routing mean drop | <=2pp | "
            f"{aggregate['max_old_routing_mean_drop']:.2%} | "
            f"{verdict['old_routing_mean_drop']} |"
        ),
        (
            "| Canonical performance drop | <=2pp | "
            f"{aggregate['max_canonical_performance_drop']:.2%} | "
            f"{verdict['canonical_performance_drop']} |"
        ),
        (
            "| Consolidated-task drop | <=2pp | "
            f"{aggregate['max_consolidated_task_drop']:.2%} | "
            f"{verdict['consolidated_task_drop']} |"
        ),
        (
            "| Recurrence direct reuse | >=90% | "
            f"{aggregate['mean_recurrence_reuse']:.2%} | {verdict['recurrence_reuse']} |"
        ),
        (
            "| New-class routing | >=95% each | "
            f"{aggregate['min_new_class_routing']:.2%} min | "
            f"{verdict['new_class_routing']} |"
        ),
        (
            "| Worst old-class routing drop | <=5pp | "
            f"{aggregate['max_worst_old_routing_drop']:.2%} | "
            f"{verdict['worst_old_routing_drop']} |"
        ),
        (
            "| Unselected forward calls | 0 | "
            f"{aggregate['total_unselected_forward_calls']} | "
            f"{verdict['unselected_forward_calls']} |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
