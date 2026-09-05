"""Bank Competition and Routing Robustness Scaling Benchmark (Phase A.2 Task A2-C004).

Evaluates routing robustness, distractor false-selection rates, parameter scaling,
analytical FLOPs scaling, and wall-clock latency as the resident primitive bank scales:
N in {10, 16, 32, 64, 128}

Conditions:
- N=10: 10 initial semantic operations (Phase A.1 universe)
- N=16: 16 semantic operations (Phase A.2 class-incremental bank from A2-C003)
- N in {32, 64, 128}: 16 semantic operations + (N - 16) frozen matched-scale distractor primitives

Primary Acceptance Criteria at N=128 routing-only scale:
- known-task top-1 >= 0.95
- distractor false selection <= 0.05
- top-1 selected primitive only executes
- unselected calls == 0
- evaluated across >= 5 seeds (0, 1, 2, 3, 4)

Caveat: Explicitly labeled as routing/competition scaling with frozen distractors,
NOT as 128-semantic continual learning.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.operations import (
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
)
from apc.evaluation.compute_accounting import (
    FLOPsBreakdown,
    LatencyMetrics,
    ParameterBreakdown,
    compute_flops_breakdown,
    count_system_parameters,
    execute_dense_primitive_baseline,
    profile_execution,
    verify_sparse_execution,
)
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
    get_or_build_16_primitive_bank,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
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
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.primitives.router import Router, RouterConfig

DEFAULT_BANK_SIZES: Final[tuple[int, ...]] = (10, 16, 32, 64, 128)
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

KNOWN_TASK_TOP1_THRESHOLD: Final[float] = 0.95
DISTRACTOR_FALSE_SELECTION_THRESHOLD: Final[float] = 0.05
UNSELECTED_CALLS_THRESHOLD: Final[int] = 0


@dataclasses.dataclass(frozen=True)
class SizeEvaluationResult:
    """Outcome of bank competition evaluation at a specific resident bank size N."""

    bank_size: int
    num_semantic_ops: int
    num_distractors: int
    known_task_top1: float
    known_task_topk: float
    distractor_false_selection_rate: float
    recurrence_top1: float
    mean_known_class_drop: float
    worst_known_class_drop: float
    unselected_forward_calls: int
    selected_forward_calls: int
    parameter_breakdown: ParameterBreakdown
    flops_breakdown: FLOPsBreakdown
    sparse_latency: LatencyMetrics
    dense_latency: LatencyMetrics
    latency_ratio: float
    per_operation_top1: dict[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["parameter_breakdown"] = self.parameter_breakdown.to_dict()
        d["flops_breakdown"] = self.flops_breakdown.to_dict()
        d["sparse_latency"] = self.sparse_latency.to_dict()
        d["dense_latency"] = self.dense_latency.to_dict()
        return d


@dataclasses.dataclass(frozen=True)
class SeedScalingResult:
    """Benchmark outcome for a single seed across all evaluated bank sizes."""

    seed: int
    size_results: dict[int, SizeEvaluationResult]
    passed_n128: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "size_results": {str(k): v.to_dict() for k, v in self.size_results.items()},
            "passed_n128": self.passed_n128,
        }


@dataclasses.dataclass(frozen=True)
class BankScalingConfig:
    """Configuration for Task A2-C004 bank competition and scaling benchmark."""

    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    num_train_examples: int = 64
    num_eval_examples: int = 200
    num_recurrence_examples: int = 50
    router_lr: float = 0.005
    router_steps: int = 250
    warmup_steps: int = 5
    active_profile_steps: int = 20
    device_str: str = "auto"
    output_dir: Path | None = None


def build_scaled_bank_and_router(
    core: Any,
    base_bank_16: PrimitiveBank,
    base_router_16: Router,
    op_to_id: dict[str, int],
    target_size: int,
    *,
    seed: int,
) -> tuple[PrimitiveBank, Router, list[int], list[int], list[int]]:
    """Construct bank and router scaled to `target_size`.

    Args:
        core: Shared Core model/encoder.
        base_bank_16: Fully trained 16-primitive bank.
        base_router_16: Fully calibrated 16-operation router.
        op_to_id: Mapping from operation name to primitive ID.
        target_size: Target resident bank size (e.g. 10, 16, 32, 64, 128).
        seed: Random seed for deterministic distractor generation.

    Returns:
        tuple of (scaled_bank, scaled_router, all_pids, semantic_pids, distractor_pids)
    """
    device = next(base_router_16.parameters()).device
    all_16_ops = list(INITIAL_10_OPERATIONS) + list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)
    semantic_pids_16 = [op_to_id[op] for op in all_16_ops]

    if target_size == 10:
        # Initial 10 operations subset
        ops_10 = list(INITIAL_10_OPERATIONS)
        semantic_pids = [op_to_id[op] for op in ops_10]
        bank_10 = PrimitiveBank()
        for pid in semantic_pids:
            p = base_bank_16.get(pid)
            bank_10.add_primitive(copy.deepcopy(p))
        bank_10.to(device)
        bank_10.freeze_all()
        bank_10.eval()

        router_10 = Router(
            RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")
        )
        router_10.query_proj.load_state_dict(base_router_16.query_proj.state_dict())
        for pid in semantic_pids:
            router_10.add_primitive_key(pid)
            with torch.no_grad():
                router_10.key_parameter(pid).copy_(base_router_16.key_parameter(pid))
        router_10.to(device)
        router_10.eval()
        for param in router_10.parameters():
            param.requires_grad_(False)

        return bank_10, router_10, semantic_pids, semantic_pids, []

    # For target_size >= 16: start with all 16 semantic primitives
    scaled_bank = PrimitiveBank()
    for pid in semantic_pids_16:
        p = base_bank_16.get(pid)
        scaled_bank.add_primitive(copy.deepcopy(p))

    scaled_router = Router(
        RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")
    )
    scaled_router.query_proj.load_state_dict(base_router_16.query_proj.state_dict())
    for pid in semantic_pids_16:
        scaled_router.add_primitive_key(pid)
        with torch.no_grad():
            scaled_router.key_parameter(pid).copy_(base_router_16.key_parameter(pid))

    semantic_pids = list(semantic_pids_16)
    distractor_pids: list[int] = []

    if target_size > 16:
        # Calculate mean L2 norm of real semantic keys to match distractor key scale
        with torch.no_grad():
            real_norms = [
                scaled_router.key_parameter(pid).norm(p=2).item()
                for pid in semantic_pids
            ]
            mean_key_norm = sum(real_norms) / len(real_norms) if real_norms else 1.0

            # Compute orthogonal complement basis of semantic key subspace
            stacked_sem_keys = torch.stack(
                [scaled_router.key_parameter(pid) for pid in semantic_pids], dim=0
            )
            _, _, vh = torch.linalg.svd(stacked_sem_keys, full_matrices=True)
            v_perp = vh[len(semantic_pids):].T.to(device)

        # Dynamically derive distractor configuration from a reference stable primitive in base bank
        ref_prim = base_bank_16.get(semantic_pids_16[-1])
        if isinstance(ref_prim, CrossPositionPrimitive):
            ref_cfg = ref_prim.config
            d_operator = ref_cfg.d_operator
            n_head = ref_cfg.n_head
            d_operator_ff = ref_cfg.d_operator_ff
            vocab_size = ref_cfg.vocab_size
            max_seq_len = ref_cfg.max_sequence_length
        else:
            d_operator = 32
            n_head = 4
            d_operator_ff = 64
            vocab_size = 10
            max_seq_len = 32

        num_distractors = target_size - 16
        for i in range(num_distractors):
            # 1. Add frozen matched-scale CrossPositionPrimitive as distractor
            distractor_cfg = CrossPositionPrimitiveConfig(
                operation="SWAP_ENDS",
                d_model=core.model.config.d_model,
                d_operator=d_operator,
                n_head=n_head,
                d_operator_ff=d_operator_ff,
                vocab_size=vocab_size,
                max_sequence_length=max_seq_len,
            )
            distractor_p = scaled_bank.new_cross_position_primitive(
                distractor_cfg,
                status=PrimitiveStatus.STABLE,
                metadata={"is_distractor": True, "label": f"DISTRACTOR_{i}", "index": i},
            )
            distractor_pid = distractor_p.primitive_id
            distractor_pids.append(distractor_pid)
            distractor_p.to(device)

            # 2. Add distractor key in orthogonal complement with matched-scale norm
            scaled_router.add_primitive_key(distractor_pid)
            gen = torch.Generator().manual_seed(seed * 10000 + i * 37 + 19)
            rand_coeff = torch.randn(v_perp.shape[1], generator=gen)
            rand_dir = v_perp @ rand_coeff.to(device)
            rand_dir = rand_dir / (rand_dir.norm(p=2) + 1e-9)
            scaled_key = rand_dir * mean_key_norm

            with torch.no_grad():
                scaled_router.key_parameter(distractor_pid).copy_(scaled_key)

    scaled_bank.to(device)
    scaled_bank.freeze_all()
    scaled_bank.eval()

    scaled_router.to(device)
    scaled_router.eval()
    for param in scaled_router.parameters():
        param.requires_grad_(False)

    all_pids = semantic_pids + distractor_pids
    return scaled_bank, scaled_router, all_pids, semantic_pids, distractor_pids


def evaluate_bank_scaling_at_size(
    core: Any,
    scaled_bank: PrimitiveBank,
    scaled_router: Router,
    all_pids: list[int],
    semantic_pids: list[int],
    distractor_pids: list[int],
    op_to_id: dict[str, int],
    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]],
    recurrence_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]],
    baseline_semantic_top1: dict[int, float],
    *,
    warmup_steps: int = 5,
    active_profile_steps: int = 20,
) -> SizeEvaluationResult:
    """Evaluate router robustness, competition, parameters, FLOPs, and latency at bank size N."""
    device = next(scaled_router.parameters()).device
    scaled_bank.eval()
    scaled_router.eval()

    id_to_op = {pid: op for op, pid in op_to_id.items()}
    distractor_set = set(distractor_pids)

    # 1. Evaluate known-task top-1, top-k, and distractor false-selection rate
    total_known_eval_examples = 0
    total_known_top1_correct = 0
    total_known_topk_correct = 0
    total_distractor_selections = 0

    per_op_top1: dict[str, float] = {}
    old_drops: list[float] = []

    with torch.no_grad():
        for pid in semantic_pids:
            examples = eval_z_by_pid[pid]
            if not examples:
                continue
            z_batch = torch.stack([ex[0].to(device) for ex in examples], dim=0)
            out = scaled_router(z_batch, all_pids)

            top1_preds = out.selected_ids[:, 0]
            correct_mask = top1_preds == pid
            top1_correct = int(correct_mask.sum().item())
            total_known_top1_correct += top1_correct

            topk_mask = (out.selected_ids == pid).any(dim=-1)
            total_known_topk_correct += int(topk_mask.sum().item())

            distractor_mask = torch.tensor(
                [p.item() in distractor_set for p in top1_preds],
                dtype=torch.bool,
                device=device,
            )
            total_distractor_selections += int(distractor_mask.sum().item())

            n_ex = len(examples)
            total_known_eval_examples += n_ex
            op_acc = top1_correct / n_ex
            op_name = id_to_op.get(pid, f"OP_{pid}")
            per_op_top1[op_name] = op_acc

            if pid in baseline_semantic_top1:
                old_drops.append(baseline_semantic_top1[pid] - op_acc)

    known_task_top1 = (
        total_known_top1_correct / total_known_eval_examples
        if total_known_eval_examples > 0
        else 0.0
    )
    known_task_topk = (
        total_known_topk_correct / total_known_eval_examples
        if total_known_eval_examples > 0
        else 0.0
    )
    distractor_false_selection_rate = (
        total_distractor_selections / total_known_eval_examples
        if total_known_eval_examples > 0
        else 0.0
    )
    mean_known_drop = (
        sum(max(0.0, d) for d in old_drops) / len(old_drops) if old_drops else 0.0
    )
    worst_known_drop = max(old_drops) if old_drops else 0.0

    # 2. Recurrence evaluation
    total_rec_examples = 0
    total_rec_correct = 0
    with torch.no_grad():
        for pid in semantic_pids:
            examples = recurrence_z_by_pid.get(pid, [])
            if not examples:
                continue
            z_batch = torch.stack([ex[0].to(device) for ex in examples], dim=0)
            out = scaled_router(z_batch, all_pids)
            total_rec_correct += int((out.selected_ids[:, 0] == pid).sum().item())
            total_rec_examples += len(examples)

    recurrence_top1 = (
        total_rec_correct / total_rec_examples if total_rec_examples > 0 else 0.0
    )

    # 3. Parameter accounting
    sample_selected_id = semantic_pids[0]
    param_breakdown = count_system_parameters(
        core=core,
        router=scaled_router,
        bank=scaled_bank,
        selected_ids=[sample_selected_id],
        trainable_only=False,
    )

    # 4. Analytical FLOPs accounting
    batch_size = 16
    seq_len_task = 12
    seq_len_content = 16
    seq_len_out = 16
    flops_breakdown = compute_flops_breakdown(
        core=core,
        router=scaled_router,
        bank=scaled_bank,
        selected_ids=[sample_selected_id],
        seq_len_task=seq_len_task,
        seq_len_content=seq_len_content,
        seq_len_out=seq_len_out,
        batch_size=batch_size,
    )

    # 5. Latency profiling: Sparse path vs Dense-all-primitives baseline
    test_z = eval_z_by_pid[semantic_pids[0]][0][0].unsqueeze(0).repeat(batch_size, 1).to(device)
    dummy_h = torch.randn(
        batch_size, seq_len_content, core.model.config.d_model, device=device
    )
    lengths = [seq_len_content] * batch_size
    out_lengths = [seq_len_out] * batch_size

    from apc.primitives.primitive import ReverseRelativePrimitive, ShiftRelativePrimitive

    def _execute_single_primitive(prim: Any, h: torch.Tensor) -> torch.Tensor:
        if isinstance(
            prim,
            (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
        ):
            return prim(h, lengths[:h.shape[0]], out_lengths[:h.shape[0]])
        return prim(h)

    def run_sparse_execution() -> tuple[torch.Tensor, torch.Tensor]:
        router_out = scaled_router(test_z, all_pids)
        chosen_pid = int(router_out.selected_ids[0, 0].item())
        prim = scaled_bank.get(chosen_pid)
        out = _execute_single_primitive(prim, dummy_h)
        return router_out.selected_ids, out

    def run_dense_execution() -> dict[int, torch.Tensor]:
        _ = scaled_router(test_z, all_pids)
        return execute_dense_primitive_baseline(scaled_bank, dummy_h, lengths, out_lengths)

    _, sparse_latency = profile_execution(
        run_sparse_execution,
        warmup_steps=warmup_steps,
        active_steps=active_profile_steps,
        batch_size=batch_size,
        device=device,
    )

    _, dense_latency = profile_execution(
        run_dense_execution,
        warmup_steps=warmup_steps,
        active_steps=active_profile_steps,
        batch_size=batch_size,
        device=device,
    )

    lat_ratio = (
        sparse_latency.median_ms / dense_latency.median_ms
        if dense_latency.median_ms > 0
        else 1.0
    )

    # 6. Strict sparse call verification
    for pid in all_pids:
        scaled_bank.get(pid).forward_call_count = 0

    with torch.no_grad():
        router_out = scaled_router(test_z, all_pids)
        active_pids_in_batch = router_out.selected_ids[:, 0].tolist()
        unique_active_pids = sorted(set(active_pids_in_batch))
        for pid in unique_active_pids:
            p = scaled_bank.get(pid)
            mask = [
                idx
                for idx, selected_pid in enumerate(active_pids_in_batch)
                if selected_pid == pid
            ]
            _ = _execute_single_primitive(p, dummy_h[mask])

    is_sparse_valid, sparse_violations = verify_sparse_execution(
        scaled_bank, unique_active_pids
    )
    if not is_sparse_valid:
        raise RuntimeError(f"Sparse verification failed at N={len(all_pids)}: {sparse_violations}")

    selected_calls = sum(scaled_bank.get(pid).forward_call_count for pid in unique_active_pids)
    unselected_pids = [pid for pid in all_pids if pid not in unique_active_pids]
    unselected_calls = sum(scaled_bank.get(pid).forward_call_count for pid in unselected_pids)

    passed = (
        known_task_top1 >= KNOWN_TASK_TOP1_THRESHOLD
        and distractor_false_selection_rate <= DISTRACTOR_FALSE_SELECTION_THRESHOLD
        and unselected_calls == UNSELECTED_CALLS_THRESHOLD
    )

    return SizeEvaluationResult(
        bank_size=len(all_pids),
        num_semantic_ops=len(semantic_pids),
        num_distractors=len(distractor_pids),
        known_task_top1=known_task_top1,
        known_task_topk=known_task_topk,
        distractor_false_selection_rate=distractor_false_selection_rate,
        recurrence_top1=recurrence_top1,
        mean_known_class_drop=mean_known_drop,
        worst_known_class_drop=worst_known_drop,
        unselected_forward_calls=unselected_calls,
        selected_forward_calls=selected_calls,
        parameter_breakdown=param_breakdown,
        flops_breakdown=flops_breakdown,
        sparse_latency=sparse_latency,
        dense_latency=dense_latency,
        latency_ratio=lat_ratio,
        per_operation_top1=per_op_top1,
        passed=passed,
    )


def run_single_seed_scaling_benchmark(
    seed: int,
    *,
    config: BankScalingConfig,
    base_core: Any | None = None,
    base_bank_16: PrimitiveBank | None = None,
    op_to_id: dict[str, int] | None = None,
) -> SeedScalingResult:
    """Run bank scaling competition benchmark for a single seed across all target sizes."""
    use_cuda = (
        config.device_str == "auto" and torch.cuda.is_available()
    ) or config.device_str == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    # 1. Setup frozen shared core with dynamic embedding alignment
    if base_core is None:
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
    else:
        core = base_core

    # 2. Setup 16-primitive bank
    if base_bank_16 is None or op_to_id is None:
        bank_16, op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=config.output_dir)
    else:
        bank_16 = base_bank_16
    bank_16.to(device)

    # 3. Generate task representations for all 16 operations
    all_16_ops: list[str] = list(INITIAL_10_OPERATIONS) + list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)
    train_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    recurrence_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}

    for op in all_16_ops:
        pid = op_to_id[op]
        # Train split
        ex_train = generate_benchmark_examples(
            seed=seed * 1000 + 11,
            n=config.num_train_examples,
            operation=op,
            split="train",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_train = extract_task_representations(core, ex_train)
        train_z_by_pid[pid] = [(z_train[i], pid) for i in range(len(ex_train))]

        # Eval split
        ex_eval = generate_benchmark_examples(
            seed=seed * 1000 + 99,
            n=config.num_eval_examples,
            operation=op,
            split="test",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_eval = extract_task_representations(core, ex_eval)
        eval_z_by_pid[pid] = [(z_eval[i], pid) for i in range(len(ex_eval))]

        # Recurrence split
        ex_rec = generate_benchmark_examples(
            seed=seed * 1000 + 555,
            n=config.num_recurrence_examples,
            operation=op,
            split="test",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_rec = extract_task_representations(core, ex_rec)
        recurrence_z_by_pid[pid] = [(z_rec[i], pid) for i in range(len(ex_rec))]

    # 4. Train/calibrate 16-operation router using bounded replay (R2 protocol)
    pids_16 = [op_to_id[op] for op in all_16_ops]
    r_cfg = RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")
    base_router_16 = Router(r_cfg)
    base_router_16.to(device)

    # Use R0 or R2 calibration across the 16 operations
    router_inc_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
        router_lr=config.router_lr,
        router_steps=config.router_steps,
        seed=seed,
    )
    replay_buf = RouterReplayBuffer(max_per_class=32, max_total=512)
    update_router_incrementally(
        base_router_16,
        candidate_ids=pids_16,
        new_primitive_ids=pids_16,
        new_data_by_pid={pid: train_z_by_pid[pid] for pid in pids_16},
        replay_buffer=replay_buf,
        config=router_inc_cfg,
        all_historical_data_by_pid=train_z_by_pid,
        device=device,
    )

    # Initial baseline evaluation on 16 classes
    base_metrics = evaluate_router_accuracy(
        base_router_16, pids_16, {pid: eval_z_by_pid[pid] for pid in pids_16}, device=device
    )
    baseline_semantic_top1 = {pid: base_metrics[pid]["top1"] for pid in pids_16}

    # 5. Evaluate across all bank sizes in config.bank_sizes
    size_results: dict[int, SizeEvaluationResult] = {}

    for target_size in config.bank_sizes:
        scaled_bank, scaled_router, all_pids, semantic_pids, distractor_pids = (
            build_scaled_bank_and_router(
                core=core,
                base_bank_16=bank_16,
                base_router_16=base_router_16,
                op_to_id=op_to_id,
                target_size=target_size,
                seed=seed,
            )
        )

        res = evaluate_bank_scaling_at_size(
            core=core,
            scaled_bank=scaled_bank,
            scaled_router=scaled_router,
            all_pids=all_pids,
            semantic_pids=semantic_pids,
            distractor_pids=distractor_pids,
            op_to_id=op_to_id,
            eval_z_by_pid=eval_z_by_pid,
            recurrence_z_by_pid=recurrence_z_by_pid,
            baseline_semantic_top1=baseline_semantic_top1,
            warmup_steps=config.warmup_steps,
            active_profile_steps=config.active_profile_steps,
        )
        size_results[target_size] = res

    passed_n128 = size_results.get(128, size_results[max(config.bank_sizes)]).passed

    return SeedScalingResult(
        seed=seed,
        size_results=size_results,
        passed_n128=passed_n128,
    )


def run_bank_scaling_benchmark(
    config: BankScalingConfig,
) -> dict[str, Any]:
    """Execute full multi-seed bank competition and scaling benchmark."""
    per_seed_results: list[SeedScalingResult] = []

    for seed in config.seeds:
        seed_res = run_single_seed_scaling_benchmark(seed=seed, config=config)
        per_seed_results.append(seed_res)

    # Compute aggregate metrics per bank size across seeds
    aggregated_by_size: dict[int, dict[str, Any]] = {}
    for size in config.bank_sizes:
        size_runs = [sr.size_results[size] for sr in per_seed_results if size in sr.size_results]
        if not size_runs:
            continue

        mean_top1 = sum(r.known_task_top1 for r in size_runs) / len(size_runs)
        mean_topk = sum(r.known_task_topk for r in size_runs) / len(size_runs)
        mean_distractor_rate = (
            sum(r.distractor_false_selection_rate for r in size_runs) / len(size_runs)
        )
        mean_recurrence = sum(r.recurrence_top1 for r in size_runs) / len(size_runs)
        worst_drop = max(r.worst_known_class_drop for r in size_runs)
        mean_unselected = sum(r.unselected_forward_calls for r in size_runs) / len(size_runs)
        all_passed = all(r.passed for r in size_runs)

        # Average compute metrics
        first_run = size_runs[0]
        mean_sparse_lat = sum(r.sparse_latency.median_ms for r in size_runs) / len(size_runs)
        mean_dense_lat = sum(r.dense_latency.median_ms for r in size_runs) / len(size_runs)
        mean_lat_ratio = sum(r.latency_ratio for r in size_runs) / len(size_runs)
        peak_gpu_mem = max(r.sparse_latency.peak_gpu_memory_bytes for r in size_runs)

        aggregated_by_size[size] = {
            "bank_size": size,
            "num_semantic_ops": first_run.num_semantic_ops,
            "num_distractors": first_run.num_distractors,
            "mean_known_task_top1": mean_top1,
            "mean_known_task_topk": mean_topk,
            "mean_distractor_false_selection_rate": mean_distractor_rate,
            "mean_recurrence_top1": mean_recurrence,
            "worst_known_class_drop": worst_drop,
            "mean_unselected_calls": mean_unselected,
            "all_seeds_passed": all_passed,
            "resident_primitive_params": (
                first_run.parameter_breakdown.resident_primitive_params
            ),
            "active_primitive_params": (
                first_run.parameter_breakdown.active_primitive_params
            ),
            "primitive_param_savings_ratio": (
                first_run.parameter_breakdown.primitive_parameter_savings_ratio
            ),
            "total_param_savings_ratio": (
                first_run.parameter_breakdown.total_parameter_savings_ratio
            ),
            "router_flops": first_run.flops_breakdown.router_flops,
            "total_sparse_flops": first_run.flops_breakdown.total_sparse_flops,
            "dense_baseline_flops": first_run.flops_breakdown.dense_baseline_flops,
            "flops_savings_ratio": first_run.flops_breakdown.flops_savings_ratio,
            "mean_sparse_latency_ms": mean_sparse_lat,
            "mean_dense_latency_ms": mean_dense_lat,
            "mean_latency_ratio": mean_lat_ratio,
            "peak_gpu_memory_bytes": peak_gpu_mem,
        }

    overall_passed_n128 = all(sr.passed_n128 for sr in per_seed_results)

    final_report = {
        "bank_sizes": list(config.bank_sizes),
        "seeds": list(config.seeds),
        "overall_passed_n128": overall_passed_n128,
        "aggregated_by_size": {str(k): v for k, v in aggregated_by_size.items()},
        "per_seed": [sr.to_dict() for sr in per_seed_results],
    }

    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = config.output_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        gate_status_str = "YES (PASS)" if overall_passed_n128 else "NO (FAIL)"
        md_lines = [
            "# Phase A.2 Task A2-C004: Bank Competition and Routing Scaling Benchmark",
            "",
            f"- **Overall N=128 Gate Passed:** {gate_status_str}",
            f"- **Seeds Evaluated:** {list(config.seeds)}",
            f"- **Bank Sizes Evaluated:** {list(config.bank_sizes)}",
            "",
            "## Summary Table by Bank Size",
            "",
            "| N | SemOps | Dist | Known Top-1 | False Sel | Recurrence | "
            "Resident Params | Active Params | Sparse Lat (ms) | Dense Lat (ms) | "
            "Lat Ratio | Unselected Calls | Status |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for size in config.bank_sizes:
            s_agg = aggregated_by_size[size]
            status_str = "PASS" if s_agg["all_seeds_passed"] else "FAIL"
            row_str = (
                f"| {size} | {s_agg['num_semantic_ops']} | {s_agg['num_distractors']} | "
                f"{s_agg['mean_known_task_top1']:.4f} | "
                f"{s_agg['mean_distractor_false_selection_rate']:.4f} | "
                f"{s_agg['mean_recurrence_top1']:.4f} | "
                f"{s_agg['resident_primitive_params']:,} | "
                f"{s_agg['active_primitive_params']:,} | "
                f"{s_agg['mean_sparse_latency_ms']:.2f} | "
                f"{s_agg['mean_dense_latency_ms']:.2f} | "
                f"{s_agg['mean_latency_ratio']:.2%} | "
                f"{s_agg['mean_unselected_calls']:.0f} | **{status_str}** |"
            )
            md_lines.append(row_str)

        n128_agg = aggregated_by_size[128]
        n10_agg = aggregated_by_size[10]
        md_lines.extend([
            "",
            "## Architectural Invariants & Observations",
            "",
            "1. **Routing Robustness under Bank Competition:**",
            f"   - Known-task Top-1 accuracy at N=128: {n128_agg['mean_known_task_top1']:.4f} "
            "(threshold >= 0.95).",
            f"   - Distractor false-selection rate at N=128: "
            f"{n128_agg['mean_distractor_false_selection_rate']:.4f} (threshold <= 0.05).",
            "2. **Strict Call Sparsity:**",
            f"   - Unselected primitive forward calls: {n128_agg['mean_unselected_calls']:.0f} "
            "(invariant == 0).",
            "3. **Compute Scaling Invariant:**",
            f"   - Resident primitive parameters scale linearly O(N): "
            f"{n10_agg['resident_primitive_params']:,} (N=10) -> "
            f"{n128_agg['resident_primitive_params']:,} (N=128).",
            f"   - Active primitive parameters remain O(1): "
            f"{n128_agg['active_primitive_params']:,}.",
            f"   - Primitive parameter savings at N=128: "
            f"{n128_agg['primitive_param_savings_ratio']:.2%}.",
            "",
            "> **Scientific Caveat:**",
            "> Distractor scaling to N=128 uses frozen matched-scale distractor primitives "
            "and evaluated router competition.",
            "> Per project rules, this is NOT described as 128-semantic continual learning.",
        ])

        report_md_path = config.output_dir / "report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    return final_report
