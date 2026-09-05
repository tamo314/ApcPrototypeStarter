"""End-to-end sparse-compute and latency scaling benchmark (A2-C010).

Unlike A2-C004's routing/primitive microbenchmark, this benchmark times the
entire causal APC inference graph: TaskSpec encoding, task-blind content
encoding, learned top-1 routing, and primitive readout.  The dense comparison
uses the same encoders and router, then *actually executes* every resident
stable primitive.  Sizes above the 16 executable semantic entries use the
matched frozen distractors established by A2-C004 and are explicitly a
routing/compute scale, not semantic continual-learning, result.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.core.data import collate_content_only_batch, collate_task_only_batch
from apc.environments.generator import Example
from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS, get_operation
from apc.evaluation.bank_scaling_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_SEEDS,
    build_scaled_bank_and_router,
)
from apc.evaluation.compute_accounting import (
    FLOPsBreakdown,
    LatencyMetrics,
    ParameterBreakdown,
    compute_flops_breakdown,
    count_system_parameters,
    profile_execution,
    verify_sparse_execution,
)
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
    get_or_build_16_primitive_bank,
)
from apc.evaluation.learned_routing_benchmark import extract_operation_argument
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
    update_router_incrementally,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    PrimitiveBase,
    PrimitiveStatus,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)
from apc.primitives.router import Router, RouterConfig

ROUTING_ACCURACY_THRESHOLD: Final[float] = 0.95
PRACTICAL_N128_LATENCY_RATIO_TARGET: Final[float] = 0.30


@dataclass(frozen=True)
class ComputeLatencyScalingConfig:
    """Serializable configuration for the A2-C010 measurement protocol."""

    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    num_router_train_examples: int = 64
    num_routing_eval_examples: int = 100
    profile_batch_size: int = 1
    warmup_steps: int = 10
    profile_steps: int = 50
    device_str: str = "auto"
    bank_source_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.bank_sizes or any(size < 10 for size in self.bank_sizes):
            raise ValueError("bank_sizes must be non-empty and all >= 10")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if self.num_router_train_examples < 1 or self.num_routing_eval_examples < 1:
            raise ValueError("router train/eval example counts must be positive")
        if self.profile_batch_size < 1 or self.warmup_steps < 0 or self.profile_steps < 1:
            raise ValueError("invalid profiling batch size or step count")

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["bank_source_dir"] = str(self.bank_source_dir)
        raw["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return raw


@dataclass(frozen=True)
class PreparedInferenceBatch:
    """Pre-collated model inputs for one operation's real TaskSpec examples."""

    examples: tuple[Example, ...]
    task_ids: torch.Tensor
    content_ids: torch.Tensor
    task_end_indices: torch.Tensor
    content_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]


@dataclass(frozen=True)
class EndToEndSizeResult:
    """One seed/one resident-bank-size C010 measurement result."""

    seed: int
    bank_size: int
    num_semantic_ops: int
    num_distractors: int
    routing_accuracy: float
    parameter_breakdown: ParameterBreakdown
    flops_breakdown: FLOPsBreakdown
    router_overhead_flops_ratio: float
    sparse_latency: LatencyMetrics
    dense_latency: LatencyMetrics
    router_latency: LatencyMetrics
    median_latency_ratio: float
    p95_latency_ratio: float
    sparse_selected_forward_calls: int
    sparse_unselected_forward_calls: int
    dense_forward_calls: int
    dense_all_stable_primitives_executed: bool
    hard_acceptance_passed: bool
    practical_latency_target_passed: bool

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["parameter_breakdown"] = self.parameter_breakdown.to_dict()
        result["flops_breakdown"] = self.flops_breakdown.to_dict()
        result["sparse_latency"] = self.sparse_latency.to_dict()
        result["dense_latency"] = self.dense_latency.to_dict()
        result["router_latency"] = self.router_latency.to_dict()
        return result


def _prepare_inference_batch(core: Any, examples: Sequence[Example]) -> PreparedInferenceBatch:
    """Create real task/content token batches without timing Python collation."""
    if not examples:
        raise ValueError("examples must be non-empty")
    if any(example.task_spec is None for example in examples):
        raise ValueError("all C010 examples require a model-visible TaskSpec")
    task_specs = [
        example.task_spec for example in examples if example.task_spec is not None
    ]
    task_ids = collate_task_only_batch(task_specs, core.tokens, device=core.device)
    content_ids = collate_content_only_batch(examples, core.tokens, device=core.device)
    task_end_mask = task_ids == core.tokens.task_end
    task_end_indices = task_end_mask.to(torch.long).argmax(dim=-1)
    return PreparedInferenceBatch(
        examples=tuple(examples),
        task_ids=task_ids,
        content_ids=content_ids,
        task_end_indices=task_end_indices,
        content_lengths=tuple(len(example.input_tokens) for example in examples),
        output_lengths=tuple(len(example.target_tokens) for example in examples),
    )


def _content_and_route(
    core: Any, router: Router, candidate_ids: Sequence[int], batch: PreparedInferenceBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the common end-to-end prefix through task/content encoders and router."""
    task_state = core.model.encode(batch.task_ids)
    rows = torch.arange(len(batch.examples), device=core.device)
    z_task = task_state[rows, batch.task_end_indices, :]
    route = router(z_task, candidate_ids)
    content_state = core.model.encode(batch.content_ids)
    lmax = max(batch.content_lengths)
    return route.selected_ids[:, 0], content_state[:, 1 : 1 + lmax, :]


def _argument_values(operation: str, examples: Sequence[Example]) -> list[Any] | None:
    """Recover a primitive's legal arguments from the visible TaskSpec.

    For non-selected dense modules the TaskSpec naturally lacks that module's
    argument.  ``extract_operation_argument`` deliberately supplies a legal
    default in that case, allowing the dense baseline to execute every real
    module without peeking at oracle metadata.
    """
    if not get_operation(operation).required_argument_names:
        return None
    values: list[Any] = []
    for example in examples:
        assert example.task_spec is not None
        values.append(extract_operation_argument(operation, example.task_spec.steps[0].arguments))
    return values


def _execute_primitive(
    primitive: PrimitiveBase,
    h_content: torch.Tensor,
    content_lengths: Sequence[int],
    output_lengths: Sequence[int],
    argument_values: Sequence[Any] | None,
) -> torch.Tensor:
    """Execute one heterogeneous primitive, including its actual readout."""
    if isinstance(
        primitive, (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive)
    ):
        return primitive(h_content, content_lengths, output_lengths, argument_values)
    return primitive(h_content)


def execute_sparse_end_to_end(
    core: Any,
    bank: PrimitiveBank,
    router: Router,
    candidate_ids: Sequence[int],
    id_to_operation: dict[int, str],
    batch: PreparedInferenceBatch,
) -> torch.Tensor:
    """Execute the full top-1 APC path, calling only selected primitives."""
    selected_ids, h_content = _content_and_route(core, router, candidate_ids, batch)
    selected = selected_ids.tolist()
    outputs: list[torch.Tensor] = []
    for primitive_id in sorted(set(selected)):
        indices = [
            index for index, selected_id in enumerate(selected) if selected_id == primitive_id
        ]
        primitive = bank.get(primitive_id)
        sub_examples = [batch.examples[index] for index in indices]
        sub_h = h_content[indices]
        sub_lengths = [batch.content_lengths[index] for index in indices]
        sub_output_lengths = [batch.output_lengths[index] for index in indices]
        outputs.append(
            _execute_primitive(
                primitive,
                sub_h,
                sub_lengths,
                sub_output_lengths,
                _argument_values(id_to_operation[primitive_id], sub_examples),
            )
        )
    # Materialize the primitive readout so neither path can elide it conceptually.
    return torch.stack([output.float().mean() for output in outputs])


def execute_dense_end_to_end(
    core: Any,
    bank: PrimitiveBank,
    router: Router,
    candidate_ids: Sequence[int],
    id_to_operation: dict[int, str],
    batch: PreparedInferenceBatch,
) -> torch.Tensor:
    """Execute the same APC prefix followed by every resident stable primitive."""
    _selected_ids, h_content = _content_and_route(core, router, candidate_ids, batch)
    outputs: list[torch.Tensor] = []
    for primitive_id in bank.ids_by_status(PrimitiveStatus.STABLE):
        primitive = bank.get(primitive_id)
        if not primitive.enabled:
            continue
        outputs.append(
            _execute_primitive(
                primitive,
                h_content,
                batch.content_lengths,
                batch.output_lengths,
                _argument_values(id_to_operation[primitive_id], batch.examples),
            )
        )
    if not outputs:
        raise RuntimeError("dense baseline found no enabled stable primitives")
    return torch.stack([output.float().mean() for output in outputs])


def _routing_accuracy(
    core: Any,
    router: Router,
    candidate_ids: Sequence[int],
    examples_by_pid: dict[int, Sequence[Example]],
) -> float:
    """Measure routing accuracy through the real TaskSpec encoder path."""
    correct = 0
    total = 0
    with torch.no_grad():
        for primitive_id, examples in examples_by_pid.items():
            for example in examples:
                batch = _prepare_inference_batch(core, [example])
                task_state = core.model.encode(batch.task_ids)
                z_task = task_state[0, batch.task_end_indices[0], :].unsqueeze(0)
                prediction = int(router(z_task, candidate_ids).selected_ids[0, 0].item())
                correct += int(prediction == primitive_id)
                total += 1
    return correct / total if total else 0.0


def evaluate_end_to_end_at_size(
    *,
    seed: int,
    core: Any,
    bank: PrimitiveBank,
    router: Router,
    candidate_ids: Sequence[int],
    semantic_ids: Sequence[int],
    distractor_ids: Sequence[int],
    id_to_operation: dict[int, str],
    routing_examples_by_pid: Mapping[int, Sequence[Example]],
    profile_batch: PreparedInferenceBatch,
    warmup_steps: int,
    profile_steps: int,
) -> EndToEndSizeResult:
    """Measure C010's actual sparse and dense execution paths at one size."""
    routing_accuracy = _routing_accuracy(
        core, router, candidate_ids, {pid: routing_examples_by_pid[pid] for pid in semantic_ids}
    )
    with torch.no_grad():
        selected_ids, _ = _content_and_route(core, router, candidate_ids, profile_batch)
    selected_id = int(selected_ids[0].item())
    parameter_breakdown = count_system_parameters(core, router, bank, [selected_id])
    flops_breakdown = compute_flops_breakdown(
        core=core,
        router=router,
        bank=bank,
        selected_ids=[selected_id],
        seq_len_task=profile_batch.task_ids.shape[1],
        seq_len_content=profile_batch.content_ids.shape[1],
        seq_len_out=max(profile_batch.output_lengths),
        batch_size=len(profile_batch.examples),
    )
    router_overhead = (
        flops_breakdown.router_flops / flops_breakdown.total_sparse_flops
        if flops_breakdown.total_sparse_flops else 0.0
    )

    with torch.no_grad():
        _, sparse_latency = profile_execution(
            lambda: execute_sparse_end_to_end(
                core, bank, router, candidate_ids, id_to_operation, profile_batch
            ),
            warmup_steps=warmup_steps,
            active_steps=profile_steps,
            batch_size=len(profile_batch.examples),
            device=core.device,
        )
        _, dense_latency = profile_execution(
            lambda: execute_dense_end_to_end(
                core, bank, router, candidate_ids, id_to_operation, profile_batch
            ),
            warmup_steps=warmup_steps,
            active_steps=profile_steps,
            batch_size=len(profile_batch.examples),
            device=core.device,
        )
        task_state = core.model.encode(profile_batch.task_ids)
        z_task = task_state[
            torch.arange(len(profile_batch.examples), device=core.device),
            profile_batch.task_end_indices,
            :,
        ]
        _, router_latency = profile_execution(
            lambda: router(z_task, candidate_ids),
            warmup_steps=warmup_steps,
            active_steps=profile_steps,
            batch_size=len(profile_batch.examples),
            device=core.device,
        )

        bank.reset_all_forward_call_counts()
        _ = execute_sparse_end_to_end(
            core, bank, router, candidate_ids, id_to_operation, profile_batch
        )
        selected_ids, _ = _content_and_route(core, router, candidate_ids, profile_batch)
        selected_set = {int(value) for value in selected_ids.tolist()}
        sparse_valid, _details = verify_sparse_execution(bank, selected_set)
        sparse_selected_calls = sum(bank.get(pid).forward_call_count for pid in selected_set)
        sparse_unselected_calls = sum(
            bank.get(pid).forward_call_count for pid in bank.ids() if pid not in selected_set
        )

        bank.reset_all_forward_call_counts()
        _ = execute_dense_end_to_end(
            core, bank, router, candidate_ids, id_to_operation, profile_batch
        )
        stable_ids = bank.ids_by_status(PrimitiveStatus.STABLE)
        dense_calls = sum(bank.get(pid).forward_call_count for pid in stable_ids)
        dense_all_executed = all(bank.get(pid).forward_call_count == 1 for pid in stable_ids)

    median_ratio = sparse_latency.median_ms / dense_latency.median_ms
    p95_ratio = sparse_latency.p95_ms / dense_latency.p95_ms
    hard_pass = (
        routing_accuracy >= ROUTING_ACCURACY_THRESHOLD
        and sparse_unselected_calls == 0
        and sparse_valid
        and dense_all_executed
    )
    return EndToEndSizeResult(
        seed=seed,
        bank_size=len(candidate_ids),
        num_semantic_ops=len(semantic_ids),
        num_distractors=len(distractor_ids),
        routing_accuracy=routing_accuracy,
        parameter_breakdown=parameter_breakdown,
        flops_breakdown=flops_breakdown,
        router_overhead_flops_ratio=router_overhead,
        sparse_latency=sparse_latency,
        dense_latency=dense_latency,
        router_latency=router_latency,
        median_latency_ratio=median_ratio,
        p95_latency_ratio=p95_ratio,
        sparse_selected_forward_calls=sparse_selected_calls,
        sparse_unselected_forward_calls=sparse_unselected_calls,
        dense_forward_calls=dense_calls,
        dense_all_stable_primitives_executed=dense_all_executed,
        hard_acceptance_passed=hard_pass,
        practical_latency_target_passed=median_ratio <= PRACTICAL_N128_LATENCY_RATIO_TARGET,
    )


def _build_base_system(
    seed: int, config: ComputeLatencyScalingConfig
) -> tuple[Any, PrimitiveBank, Router, dict[str, int], dict[int, list[Example]]]:
    """Restore the validated C004 semantic bank and calibrate its 16-way router."""
    use_cuda = (config.device_str == "auto" and torch.cuda.is_available()) or (
        config.device_str == "cuda"
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=seed, vocab_size=10, device=device.type)
    )
    core = arch.core
    core.model.eval()
    for parameter in core.model.parameters():
        parameter.requires_grad_(False)

    checkpoint = (
        Path("runs/phase_a1_shift_compact_structural_probe")
        / f"seed_{seed}"
        / "shared_encoder.pt"
    )
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        aligned_state = align_shared_core_embeddings(core.model, state, None, core.tokens)
        core.model.load_state_dict(aligned_state)

    bank, op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=config.bank_source_dir)
    bank.to(device)
    operations = list(INITIAL_10_OPERATIONS) + list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)
    examples_by_pid: dict[int, list[Example]] = {}
    train_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for operation in operations:
        primitive_id = op_to_id[operation]
        train_examples = generate_benchmark_examples(
            seed=seed * 1000 + 11,
            n=config.num_router_train_examples,
            operation=operation,
            split="train",
            vocab_size=core.tokens.env_vocab_size,
        )
        train_z = extract_task_representations(core, train_examples)
        train_z_by_pid[primitive_id] = [
            (train_z[index], primitive_id) for index in range(len(train_examples))
        ]
        examples_by_pid[primitive_id] = generate_benchmark_examples(
            seed=seed * 1000 + 99,
            n=config.num_routing_eval_examples,
            operation=operation,
            split="test",
            vocab_size=core.tokens.env_vocab_size,
        )

    candidate_ids = [op_to_id[operation] for operation in operations]
    router = Router(
        RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")
    ).to(device)
    replay = RouterReplayBuffer(max_per_class=32, max_total=512)
    update_router_incrementally(
        router,
        candidate_ids=candidate_ids,
        new_primitive_ids=candidate_ids,
        new_data_by_pid=train_z_by_pid,
        replay_buffer=replay,
        config=IncrementalRouterConfig(
            condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
            router_lr=0.005,
            router_steps=250,
            seed=seed,
        ),
        all_historical_data_by_pid=train_z_by_pid,
        device=device,
    )
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    return core, bank, router, op_to_id, examples_by_pid


def run_single_seed_compute_latency_scaling(
    seed: int, config: ComputeLatencyScalingConfig
) -> dict[int, EndToEndSizeResult]:
    """Run all requested resident sizes for one independent measurement seed."""
    core, bank_16, router_16, op_to_id, examples_by_pid = _build_base_system(seed, config)
    id_to_operation = {primitive_id: operation for operation, primitive_id in op_to_id.items()}
    profile_operation = INITIAL_10_OPERATIONS[0]
    profile_examples = examples_by_pid[op_to_id[profile_operation]][: config.profile_batch_size]
    profile_batch = _prepare_inference_batch(core, profile_examples)
    results: dict[int, EndToEndSizeResult] = {}
    for size in config.bank_sizes:
        bank, router, candidate_ids, semantic_ids, distractor_ids = build_scaled_bank_and_router(
            core, bank_16, router_16, op_to_id, size, seed=seed
        )
        scaled_id_to_operation = dict(id_to_operation)
        for distractor_id in distractor_ids:
            scaled_id_to_operation[distractor_id] = "SWAP_ENDS"
        results[size] = evaluate_end_to_end_at_size(
            seed=seed,
            core=core,
            bank=bank,
            router=router,
            candidate_ids=candidate_ids,
            semantic_ids=semantic_ids,
            distractor_ids=distractor_ids,
            id_to_operation=scaled_id_to_operation,
            routing_examples_by_pid=examples_by_pid,
            profile_batch=profile_batch,
            warmup_steps=config.warmup_steps,
            profile_steps=config.profile_steps,
        )
    return results


def _mean(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0


def run_compute_latency_scaling_benchmark(config: ComputeLatencyScalingConfig) -> dict[str, Any]:
    """Execute and persist the complete five-size A2-C010 benchmark."""
    per_seed: dict[int, dict[int, EndToEndSizeResult]] = {}
    for seed in config.seeds:
        per_seed[seed] = run_single_seed_compute_latency_scaling(seed, config)

    aggregate: dict[str, dict[str, Any]] = {}
    for size in config.bank_sizes:
        results = [per_seed[seed][size] for seed in config.seeds]
        first = results[0]
        aggregate[str(size)] = {
            "bank_size": size,
            "num_semantic_ops": first.num_semantic_ops,
            "num_distractors": first.num_distractors,
            "mean_routing_accuracy": _mean([result.routing_accuracy for result in results]),
            "minimum_routing_accuracy": min(result.routing_accuracy for result in results),
            "resident_primitive_params": first.parameter_breakdown.resident_primitive_params,
            "active_primitive_params": first.parameter_breakdown.active_primitive_params,
            "primitive_active_param_savings_ratio": (
                first.parameter_breakdown.primitive_parameter_savings_ratio
            ),
            "total_sparse_flops": first.flops_breakdown.total_sparse_flops,
            "dense_total_flops": first.flops_breakdown.dense_baseline_flops,
            "flops_savings_ratio": first.flops_breakdown.flops_savings_ratio,
            "router_flops": first.flops_breakdown.router_flops,
            "router_overhead_flops_ratio": first.router_overhead_flops_ratio,
            "sparse_median_latency_ms": _mean(
                [result.sparse_latency.median_ms for result in results]
            ),
            "sparse_p95_latency_ms": _mean([result.sparse_latency.p95_ms for result in results]),
            "sparse_throughput_examples_per_sec": _mean(
                [result.sparse_latency.throughput_examples_per_sec for result in results]
            ),
            "sparse_peak_gpu_memory_bytes": max(
                result.sparse_latency.peak_gpu_memory_bytes for result in results
            ),
            "dense_median_latency_ms": _mean(
                [result.dense_latency.median_ms for result in results]
            ),
            "dense_p95_latency_ms": _mean([result.dense_latency.p95_ms for result in results]),
            "dense_throughput_examples_per_sec": _mean(
                [result.dense_latency.throughput_examples_per_sec for result in results]
            ),
            "dense_peak_gpu_memory_bytes": max(
                result.dense_latency.peak_gpu_memory_bytes for result in results
            ),
            "router_median_latency_ms": _mean(
                [result.router_latency.median_ms for result in results]
            ),
            "router_p95_latency_ms": _mean([result.router_latency.p95_ms for result in results]),
            "median_latency_ratio": _mean([result.median_latency_ratio for result in results]),
            "p95_latency_ratio": _mean([result.p95_latency_ratio for result in results]),
            "sparse_selected_forward_calls": sum(
                result.sparse_selected_forward_calls for result in results
            ),
            "sparse_unselected_forward_calls": sum(
                result.sparse_unselected_forward_calls for result in results
            ),
            "dense_forward_calls": sum(result.dense_forward_calls for result in results),
            "all_dense_primitives_executed": all(
                result.dense_all_stable_primitives_executed for result in results
            ),
            "hard_acceptance_passed": all(result.hard_acceptance_passed for result in results),
            "practical_latency_target_passed": all(
                result.practical_latency_target_passed for result in results
            ),
        }

    n128_key = "128" if "128" in aggregate else str(max(config.bank_sizes))
    report = {
        "task": "A2-C010",
        "benchmark_kind": "end_to_end_sparse_vs_executable_dense_all_primitives",
        "scope_caveat": (
            "N>16 uses frozen matched-scale distractors and is routing/compute scaling, "
            "not 128-semantic continual learning."
        ),
        "config": config.to_dict(),
        "aggregate_by_size": aggregate,
        "per_seed": {
            str(seed): {str(size): result.to_dict() for size, result in results.items()}
            for seed, results in per_seed.items()
        },
        "hard_acceptance_passed": all(
            item["hard_acceptance_passed"] for item in aggregate.values()
        ),
        "n128_practical_latency_target_passed": aggregate[n128_key][
            "practical_latency_target_passed"
        ],
        "n128_median_latency_ratio": aggregate[n128_key]["median_latency_ratio"],
    }
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        lines = [
            "# A2-C010 — End-to-end Compute and Latency Scaling",
            "",
            "The timed sparse and dense paths both run TaskSpec encoding, task-blind content "
            "encoding, the learned router, and primitive readout. Dense executes every resident "
            "stable primitive; it is not a multiplied single-primitive estimate.",
            "",
            "| N | Route Acc. | Sparse median/p95 ms | Dense median/p95 ms | Median ratio | "
            "FLOPs sparse/dense | Router FLOPs | Unselected calls | Hard |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for size in config.bank_sizes:
            value = aggregate[str(size)]
            lines.append(
                f"| {size} | {value['mean_routing_accuracy']:.4f} | "
                f"{value['sparse_median_latency_ms']:.3f}/{value['sparse_p95_latency_ms']:.3f} | "
                f"{value['dense_median_latency_ms']:.3f}/{value['dense_p95_latency_ms']:.3f} | "
                f"{value['median_latency_ratio']:.2%} | "
                f"{value['total_sparse_flops']:,}/{value['dense_total_flops']:,} | "
                f"{value['router_flops']:,} | {value['sparse_unselected_forward_calls']} | "
                f"{'PASS' if value['hard_acceptance_passed'] else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                f"N=128 practical latency target (sparse median <= 30% dense): "
                f"{'PASS' if report['n128_practical_latency_target_passed'] else 'MISSED'} "
                f"({report['n128_median_latency_ratio']:.2%}).",
                "",
                "> N>16 uses frozen matched-scale distractors and is not a 128-semantic-task "
                "continual-learning claim.",
            ]
        )
        (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
