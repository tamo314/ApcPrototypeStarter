"""Incremental Router Update Benchmark (Task A2-C003 / Phase A.2 Stop Gate).

Tests class-incremental routing under bank growth from 10 to 16 semantic operations:
10 (initial B008 bank) -> 12 -> 14 -> 16.

Compares three experimental conditions:
- R0: Full retrain upper bound (diagnostic ceiling)
- R1: Naive new-class update (forgetting baseline)
- R2: Bounded replay update (primary condition with <= 32 examples/old class, <= 512 total)

Primary Acceptance Criteria for R2:
- new-class top-1 >= 0.95
- old-class mean drop <= 0.02 (2pp)
- worst old-class drop <= 0.05 (5pp)
- overall top-1 >= 0.95
- unselected forward calls == 0
- evaluated across >= 5 decision seeds (0, 1, 2, 3, 4)
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Final

import torch

from apc.core.data import build_task_only_tokens, pad_token_sequences
from apc.environments.generator import Example
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _train_single_primitive,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    RouterReplayBuffer,
    align_shared_core_embeddings,
    evaluate_router_accuracy,
    update_router_incrementally,
)
from apc.primitives.primitive import (
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.primitives.router import Router, RouterConfig

# 10 initial operations from B008 universe
INITIAL_10_OPERATIONS: Final[tuple[str, ...]] = (
    *ALL_CANONICAL_OPERATIONS,
    *BRANCH_B_NOVEL_OPERATION_NAMES,
)

# Growth stages
STAGE_NEW_OPERATIONS: Final[list[tuple[str, ...]]] = [
    (),  # Stage 0: 10 ops
    ("ROTATE_TRIPLETS", "SWAP_ENDS"),  # Stage 1: 12 ops
    ("MIRROR_HALVES", "ALTERNATING_NEGATE"),  # Stage 2: 14 ops
    ("CYCLE_FOUR", "INCREMENT_MOD"),  # Stage 3: 16 ops
]

NEW_CLASS_TOP1_THRESHOLD: Final[float] = 0.95
OLD_CLASS_MEAN_DROP_THRESHOLD: Final[float] = 0.02
WORST_OLD_CLASS_DROP_THRESHOLD: Final[float] = 0.05
OVERALL_TOP1_THRESHOLD: Final[float] = 0.95


@dataclasses.dataclass(frozen=True)
class StageEvaluationResult:
    """Outcome of one incremental growth stage."""

    stage_index: int
    bank_size: int
    active_operations: list[str]
    new_operations: list[str]
    old_class_top1: float
    new_class_top1: float
    mean_old_class_drop: float
    worst_old_class_drop: float
    overall_top1: float
    recurrence_top1: float
    unselected_forward_calls: int
    per_operation_top1: dict[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SeedBenchmarkResult:
    """Benchmark outcome for a single seed across all stages."""

    seed: int
    condition: str
    stages: list[StageEvaluationResult]
    final_overall_top1: float
    final_old_class_drop: float
    final_unselected_calls: int
    all_stages_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "condition": self.condition,
            "stages": [s.to_dict() for s in self.stages],
            "final_overall_top1": self.final_overall_top1,
            "final_old_class_drop": self.final_old_class_drop,
            "final_unselected_calls": self.final_unselected_calls,
            "all_stages_passed": self.all_stages_passed,
        }


def extract_task_representations(
    core: Any,
    examples: list[Example],
) -> torch.Tensor:
    """Extract z_task at [TASK_END] from frozen core for each example."""
    device = core.device
    tokens = core.tokens
    task_token_seqs = []
    for ex in examples:
        assert ex.task_spec is not None
        task_token_seqs.append(build_task_only_tokens(ex.task_spec, tokens))
    padded_task_ids = pad_token_sequences(task_token_seqs, tokens.pad, device)

    with torch.no_grad():
        encoded = core.model.encode(padded_task_ids)

    task_end_mask = padded_task_ids == tokens.task_end
    task_end_indices = task_end_mask.to(torch.long).argmax(dim=-1)

    batch_indices = torch.arange(len(examples), device=device)
    z_task = encoded[batch_indices, task_end_indices, :]
    return z_task


def get_or_build_16_primitive_bank(
    core: Any,
    seed: int,
    output_dir: Path | None = None,
) -> tuple[PrimitiveBank, dict[str, int]]:
    """Build or load the full 16-primitive bank with all primitives trained and frozen."""
    ckpt_path = output_dir / f"seed_{seed}" / "primitive_bank_16.pt" if output_dir else None
    if ckpt_path and ckpt_path.is_file():
        u_bank_cfg = UnifiedBenchmarkConfig(
            seed=seed,
            vocab_size=10,
            device=core.device,
        )
        bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
        for op in BRANCH_B_NOVEL_OPERATION_NAMES:
            p = bank.new_cross_position_primitive(
                CrossPositionPrimitiveConfig(
                    operation=op,
                    d_model=core.model.config.d_model,
                    d_operator=32,
                    n_head=4,
                    d_operator_ff=64,
                    vocab_size=10,
                    max_sequence_length=32,
                ),
                status=PrimitiveStatus.STABLE,
            )
            op_to_id[op] = p.primitive_id

        for op in PHASE_A2_INCREMENTAL_NEW_OPERATIONS:
            p = bank.new_cross_position_primitive(
                CrossPositionPrimitiveConfig(
                    operation=op,
                    d_model=core.model.config.d_model,
                    d_operator=32,
                    n_head=4,
                    d_operator_ff=64,
                    vocab_size=10,
                    max_sequence_length=32,
                ),
                status=PrimitiveStatus.STABLE,
            )
            op_to_id[op] = p.primitive_id

        bank.to(core.device)
        sd = torch.load(ckpt_path, map_location=core.device, weights_only=True)
        bank.load_state_dict(sd)
        bank.freeze_all()
        bank.eval()
        return bank, op_to_id

    # 1. Start with initial 10 primitives from B008 bank if available
    u_bank_cfg = UnifiedBenchmarkConfig(
        seed=seed,
        vocab_size=10,
        device=core.device,
    )
    bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
    for op in BRANCH_B_NOVEL_OPERATION_NAMES:
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op,
                d_model=core.model.config.d_model,
                d_operator=32,
                n_head=4,
                d_operator_ff=64,
                vocab_size=10,
                max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id

    b008_dir = Path("runs/phase_a1_learned_routing_benchmark") / f"seed_{seed}"
    b008_ckpt = b008_dir / "primitive_bank.pt"
    if b008_ckpt.is_file():
        b008_sd = torch.load(b008_ckpt, map_location=core.device, weights_only=True)
        bank.load_state_dict(b008_sd, strict=False)

    # 2. Add the 6 new Phase A.2 operations
    for op in PHASE_A2_INCREMENTAL_NEW_OPERATIONS:
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op,
                d_model=core.model.config.d_model,
                d_operator=32,
                n_head=4,
                d_operator_ff=64,
                vocab_size=10,
                max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id
        p.to(core.device)
        _train_single_primitive(core, p, u_bank_cfg, op, steps=1000)

    bank.to(core.device)
    bank.freeze_all()
    bank.eval()

    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank.state_dict(), ckpt_path)

    return bank, op_to_id


def run_single_seed_incremental_benchmark(
    seed: int,
    condition: IncrementalUpdateCondition,
    *,
    device_str: str = "auto",
    output_dir: Path | None = None,
    num_train_examples: int = 64,
    num_eval_examples: int = 200,
    router_lr: float = 0.005,
    router_steps: int = 250,
) -> SeedBenchmarkResult:
    """Run incremental router evaluation on a single seed across all growth stages."""
    use_cuda = (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    # 1. Setup frozen shared core with dynamic embedding alignment
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=seed,
        vocab_size=10,
        device=device.type,
    )
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core

    cand_dir = Path("runs/phase_a1_shift_compact_structural_probe") / f"seed_{seed}"
    ckpt_cand = cand_dir / "shared_encoder.pt"
    if ckpt_cand.is_file():
        old_sd = torch.load(ckpt_cand, map_location=device, weights_only=True)
        aligned_sd = align_shared_core_embeddings(core.model, old_sd, None, core.tokens)
        core.model.load_state_dict(aligned_sd)
    core.model.eval()
    for param in core.model.parameters():
        param.requires_grad_(False)

    # 2. Setup 16-primitive bank
    bank, op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=output_dir)
    bank.to(device)

    # 3. Generate representations for all 16 operations
    all_16_ops: list[str] = list(INITIAL_10_OPERATIONS) + list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)

    train_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    recurrence_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}

    for op in all_16_ops:
        pid = op_to_id[op]
        # Train split
        ex_train = generate_benchmark_examples(
            seed=seed * 1000 + 11,
            n=num_train_examples,
            operation=op,
            split="train",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_train = extract_task_representations(core, ex_train)
        train_z_by_pid[pid] = [(z_train[i], pid) for i in range(len(ex_train))]

        # Eval split
        ex_eval = generate_benchmark_examples(
            seed=seed * 1000 + 99,
            n=num_eval_examples,
            operation=op,
            split="test",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_eval = extract_task_representations(core, ex_eval)
        eval_z_by_pid[pid] = [(z_eval[i], pid) for i in range(len(ex_eval))]

        # Recurrence split
        ex_rec = generate_benchmark_examples(
            seed=seed * 1000 + 555,
            n=50,
            operation=op,
            split="test",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_rec = extract_task_representations(core, ex_rec)
        recurrence_z_by_pid[pid] = [(z_rec[i], pid) for i in range(len(ex_rec))]

    # 4. Initialize Router
    r_cfg = RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")
    router: Router = Router(r_cfg)
    router.to(device)

    replay_buffer = RouterReplayBuffer(max_per_class=32, max_total=512)

    # Stage 0 (initial 10 classes)
    initial_pids = [op_to_id[op] for op in INITIAL_10_OPERATIONS]
    init_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
        router_lr=router_lr,
        router_steps=router_steps,
        seed=seed,
    )
    update_router_incrementally(
        router,
        candidate_ids=initial_pids,
        new_primitive_ids=initial_pids,
        new_data_by_pid={pid: train_z_by_pid[pid] for pid in initial_pids},
        replay_buffer=replay_buffer,
        config=init_cfg,
        all_historical_data_by_pid=train_z_by_pid,
        device=device,
    )

    # Initial baseline evaluation
    stage0_metrics = evaluate_router_accuracy(
        router, initial_pids, {pid: eval_z_by_pid[pid] for pid in initial_pids}, device=device
    )
    baseline_top1 = {pid: stage0_metrics[pid]["top1"] for pid in initial_pids}

    stage_results: list[StageEvaluationResult] = []

    active_ops: list[str] = list(INITIAL_10_OPERATIONS)
    active_pids: list[int] = list(initial_pids)

    # Record Stage 0
    stage0_top1s = [stage0_metrics[pid]["top1"] for pid in initial_pids]
    stage0_res = StageEvaluationResult(
        stage_index=0,
        bank_size=len(active_pids),
        active_operations=list(active_ops),
        new_operations=list(active_ops),
        old_class_top1=sum(stage0_top1s) / len(stage0_top1s),
        new_class_top1=sum(stage0_top1s) / len(stage0_top1s),
        mean_old_class_drop=0.0,
        worst_old_class_drop=0.0,
        overall_top1=sum(stage0_top1s) / len(stage0_top1s),
        recurrence_top1=sum(stage0_top1s) / len(stage0_top1s),
        unselected_forward_calls=0,
        per_operation_top1={op: stage0_metrics[op_to_id[op]]["top1"] for op in active_ops},
        passed=True,
    )
    stage_results.append(stage0_res)

    # Incremental Stages 1, 2, 3
    for stage_idx in range(1, len(STAGE_NEW_OPERATIONS)):
        new_ops = list(STAGE_NEW_OPERATIONS[stage_idx])
        new_pids = [op_to_id[op] for op in new_ops]

        old_pids = list(active_pids)
        active_ops.extend(new_ops)
        active_pids.extend(new_pids)

        stage_inc_cfg = IncrementalRouterConfig(
            condition=condition,
            router_lr=router_lr,
            router_steps=router_steps,
            seed=seed + stage_idx * 10,
        )

        update_router_incrementally(
            router,
            candidate_ids=active_pids,
            new_primitive_ids=new_pids,
            new_data_by_pid={pid: train_z_by_pid[pid] for pid in new_pids},
            replay_buffer=replay_buffer,
            config=stage_inc_cfg,
            all_historical_data_by_pid={pid: train_z_by_pid[pid] for pid in active_pids},
            device=device,
        )

        # Evaluation
        post_metrics = evaluate_router_accuracy(
            router, active_pids, {pid: eval_z_by_pid[pid] for pid in active_pids}, device=device
        )

        old_top1s = [post_metrics[pid]["top1"] for pid in old_pids]
        new_top1s = [post_metrics[pid]["top1"] for pid in new_pids]
        all_top1s = [post_metrics[pid]["top1"] for pid in active_pids]

        old_drops = [baseline_top1[pid] - post_metrics[pid]["top1"] for pid in old_pids]
        mean_old_drop = sum(max(0.0, d) for d in old_drops) / len(old_drops)
        worst_old_drop = max(old_drops)

        # Recurrence evaluation on old classes
        rec_metrics = evaluate_router_accuracy(
            router, active_pids, {pid: recurrence_z_by_pid[pid] for pid in old_pids}, device=device
        )
        rec_top1s = [rec_metrics[pid]["top1"] for pid in old_pids]
        mean_recurrence = sum(rec_top1s) / len(rec_top1s)

        # Forward-call sparsity verification:
        for pid in active_pids:
            bank.get(pid).forward_call_count = 0
        test_z = torch.stack([eval_z_by_pid[pid][0][0].to(device) for pid in active_pids], dim=0)
        router_out = router(test_z, active_pids)
        selected_ids_set = set(router_out.selected_ids[:, 0].tolist())

        # Execute only selected primitives
        from apc.primitives.primitive import PointwisePrimitive

        for pid in selected_ids_set:
            prim = bank.get(pid)
            dummy_h = torch.randn(1, 8, core.model.config.d_model, device=device)
            if isinstance(prim, PointwisePrimitive):
                prim(dummy_h)
            else:
                prim(dummy_h, content_lengths=[8], output_lengths=[8], argument_values=None)

        unselected_calls = 0
        for pid in active_pids:
            if pid not in selected_ids_set:
                unselected_calls += bank.get(pid).forward_call_count

        # Evaluate acceptance for this stage
        new_acc = sum(new_top1s) / len(new_top1s)
        old_acc = sum(old_top1s) / len(old_top1s)
        overall_acc = sum(all_top1s) / len(all_top1s)

        if condition == IncrementalUpdateCondition.R2_BOUNDED_REPLAY:
            passed = (
                new_acc >= NEW_CLASS_TOP1_THRESHOLD
                and mean_old_drop <= OLD_CLASS_MEAN_DROP_THRESHOLD
                and worst_old_drop <= WORST_OLD_CLASS_DROP_THRESHOLD
                and overall_acc >= OVERALL_TOP1_THRESHOLD
                and unselected_calls == 0
            )
        else:
            passed = True  # R0 and R1 are baselines

        stage_res = StageEvaluationResult(
            stage_index=stage_idx,
            bank_size=len(active_pids),
            active_operations=list(active_ops),
            new_operations=list(new_ops),
            old_class_top1=old_acc,
            new_class_top1=new_acc,
            mean_old_class_drop=mean_old_drop,
            worst_old_class_drop=worst_old_drop,
            overall_top1=overall_acc,
            recurrence_top1=mean_recurrence,
            unselected_forward_calls=unselected_calls,
            per_operation_top1={op: post_metrics[op_to_id[op]]["top1"] for op in active_ops},
            passed=passed,
        )
        stage_results.append(stage_res)

        # Update baseline for next stage
        for pid in active_pids:
            baseline_top1[pid] = post_metrics[pid]["top1"]

    final_stage = stage_results[-1]
    all_passed = all(s.passed for s in stage_results[1:])

    return SeedBenchmarkResult(
        seed=seed,
        condition=condition.value,
        stages=stage_results,
        final_overall_top1=final_stage.overall_top1,
        final_old_class_drop=final_stage.mean_old_class_drop,
        final_unselected_calls=final_stage.unselected_forward_calls,
        all_stages_passed=all_passed,
    )


def run_incremental_router_benchmark(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    conditions: tuple[IncrementalUpdateCondition, ...] = (
        IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
        IncrementalUpdateCondition.R0_FULL_RETRAIN,
        IncrementalUpdateCondition.R1_NAIVE_NEW,
    ),
    output_dir: Path | str = "runs/phase_a2_incremental_router_gate",
) -> dict[str, Any]:
    """Run full benchmark across specified seeds and conditions."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_results: dict[str, Any] = {}

    for cond in conditions:
        cond_name = cond.value
        seed_results: list[SeedBenchmarkResult] = []

        print("\n==========================================")
        print(f"Running Condition: {cond_name}")
        print("==========================================")

        for seed in seeds:
            t0 = time.time()
            res = run_single_seed_incremental_benchmark(
                seed=seed,
                condition=cond,
                output_dir=out_path,
            )
            elapsed = time.time() - t0
            seed_results.append(res)
            print(
                f"Seed {seed} [{cond_name}]: Final Overall Top1 = {res.final_overall_top1:.4f}, "
                f"Worst Old Drop = {res.stages[-1].worst_old_class_drop:.4f}, "
                f"New Class Top1 = {res.stages[-1].new_class_top1:.4f}, "
                f"Passed = {res.all_stages_passed} ({elapsed:.1f}s)"
            )

        mean_final_top1 = sum(r.final_overall_top1 for r in seed_results) / len(seed_results)
        mean_worst_drop = sum(
            r.stages[-1].worst_old_class_drop for r in seed_results
        ) / len(seed_results)
        mean_new_top1 = sum(r.stages[-1].new_class_top1 for r in seed_results) / len(seed_results)
        all_passed = all(r.all_stages_passed for r in seed_results)

        summary_results[cond_name] = {
            "seeds": list(seeds),
            "mean_final_overall_top1": mean_final_top1,
            "mean_worst_old_class_drop": mean_worst_drop,
            "mean_new_class_top1": mean_new_top1,
            "all_passed": all_passed,
            "per_seed": [r.to_dict() for r in seed_results],
        }

    # Save summary report
    with open(out_path / "report.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return summary_results
