"""Phase A.2 Compute Accounting & Execution Profiling (Task A2-C002).

Implements comprehensive parameter, FLOPs, latency, and memory accounting per:
- `docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md`
- `docs/DECISIONS_PHASE_A2.md` (ADR-0062)
- `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md` (Task A2-C002)

Provides:
- `ParameterBreakdown`: Exact parameter counts (Stable Core, router, resident primitives,
  active primitives, temporary workspace params) and parameter savings ratios.
- `count_system_parameters(...)`: Module parameter inspection utility.
- Analytical FLOPs models:
  - `estimate_linear_flops`
  - `estimate_attention_flops`
  - `estimate_transformer_block_flops`
  - `estimate_transformer_flops`
  - `estimate_router_flops`
  - `estimate_pointwise_primitive_flops`
  - `estimate_cross_position_primitive_flops`
  - `estimate_primitive_flops`
  - `compute_flops_breakdown` -> `FLOPsBreakdown`
- Latency profiling and memory tracking:
  - `LatencyMetrics`
  - `profile_execution(...)` (warmup, repeated timing, CUDA events or high-res clock, peak VRAM)
  - `ExecutionTimer` (block-level latency hooks)
- Dense baseline execution:
  - `execute_dense_primitive_baseline(...)` (true execution across all resident primitives)
- Strict sparse execution verification:
  - `verify_sparse_execution(...)` (zero unselected calls invariant)
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    PointwisePrimitive,
    PrimitiveBase,
    PrimitiveStatus,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)

# ---------------------------------------------------------------------------
# 1. Parameter Accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterBreakdown:
    """Breakdown of resident, active, and temporary parameters across the APC system."""

    stable_core_params: int
    router_params: int
    resident_primitive_params: int
    active_primitive_params: int
    temporary_params: int
    resident_total_params: int
    active_total_params: int
    primitive_parameter_savings_ratio: float
    total_parameter_savings_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_system_parameters(
    core: Any,
    router: Any | None,
    bank: PrimitiveBank | None,
    selected_ids: Iterable[int] | None = None,
    workspace: Any | None = None,
    *,
    trainable_only: bool = False,
) -> ParameterBreakdown:
    """Compute exact parameter breakdown across all APC components.

    Args:
        core: Stable Core model, SharedContentEncoder, or nn.Module.
        router: Learned top-k Router or nn.Module (or None if unrouted).
        bank: PrimitiveBank containing resident primitives.
        selected_ids: Iterable of active primitive IDs selected by router.
        workspace: PlasticWorkspace containing temporary parameters.
        trainable_only: If True, only count parameters with `requires_grad=True`.

    Returns:
        ParameterBreakdown dataclass with component counts and savings ratios.
    """
    # 1. Stable Core params
    if hasattr(core, "model") and isinstance(core.model, nn.Module):
        core_mod = core.model
    elif isinstance(core, nn.Module):
        core_mod = core
    else:
        core_mod = None

    if core_mod is not None:
        core_params = sum(
            p.numel() for p in core_mod.parameters() if not trainable_only or p.requires_grad
        )
    else:
        core_params = 0

    # 2. Router params
    if router is not None and isinstance(router, nn.Module):
        router_params = sum(
            p.numel() for p in router.parameters() if not trainable_only or p.requires_grad
        )
    else:
        router_params = 0

    # 3. Primitive Bank params
    if bank is not None:
        resident_prim = bank.persistent_parameter_count(trainable_only=trainable_only)
        if selected_ids is not None:
            active_prim = bank.active_parameter_count(
                selected_ids, trainable_only=trainable_only
            )
        else:
            active_prim = resident_prim
    else:
        resident_prim = 0
        active_prim = 0

    # 4. Plastic Workspace temporary params
    if workspace is not None and hasattr(workspace, "total_parameter_count"):
        temp_params = workspace.total_parameter_count(trainable_only=trainable_only)
    else:
        temp_params = 0

    resident_total = core_params + router_params + resident_prim
    active_total = core_params + router_params + active_prim + temp_params

    prim_savings = (
        1.0 - (active_prim / resident_prim) if resident_prim > 0 else 0.0
    )
    total_savings = (
        1.0 - (active_total / resident_total) if resident_total > 0 else 0.0
    )

    return ParameterBreakdown(
        stable_core_params=core_params,
        router_params=router_params,
        resident_primitive_params=resident_prim,
        active_primitive_params=active_prim,
        temporary_params=temp_params,
        resident_total_params=resident_total,
        active_total_params=active_total,
        primitive_parameter_savings_ratio=prim_savings,
        total_parameter_savings_ratio=total_savings,
    )


# ---------------------------------------------------------------------------
# 2. Analytical FLOPs Models
# ---------------------------------------------------------------------------


def estimate_linear_flops(
    in_features: int,
    out_features: int,
    num_tokens: int = 1,
    *,
    bias: bool = True,
) -> int:
    """Analytical FLOPs for an nn.Linear layer (2 * MACs + bias additions)."""
    macs = num_tokens * in_features * out_features
    flops = 2 * macs
    if bias:
        flops += num_tokens * out_features
    return flops


def estimate_attention_flops(
    seq_len_q: int,
    seq_len_kv: int,
    d_model: int,
    n_head: int = 1,
    *,
    causal: bool = True,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for multi-head self- or cross-attention.

    Components:
    - Q projection: 2 * B * L_q * d_model^2
    - K projection: 2 * B * L_kv * d_model^2
    - V projection: 2 * B * L_kv * d_model^2
    - Q @ K^T score matrix: 2 * B * L_q * L_kv * d_model
    - Softmax: ~3 * B * n_head * L_q * L_kv
    - Attention @ V context: 2 * B * L_q * L_kv * d_model
    - Out projection: 2 * B * L_q * d_model^2
    """
    b = batch_size
    l_q = seq_len_q
    l_kv = seq_len_kv
    d = d_model

    q_flops = 2 * b * l_q * d * d
    k_flops = 2 * b * l_kv * d * d
    v_flops = 2 * b * l_kv * d * d
    score_flops = 2 * b * l_q * l_kv * d
    softmax_flops = 3 * b * n_head * l_q * l_kv
    context_flops = 2 * b * l_q * l_kv * d
    out_proj_flops = 2 * b * l_q * d * d

    return (
        q_flops
        + k_flops
        + v_flops
        + score_flops
        + softmax_flops
        + context_flops
        + out_proj_flops
    )


def estimate_transformer_block_flops(
    seq_len: int,
    d_model: int,
    n_head: int,
    d_ff: int,
    *,
    causal: bool = True,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for one Transformer block (Causal Attention + MLP + LayerNorms)."""
    attn_flops = estimate_attention_flops(
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        d_model=d_model,
        n_head=n_head,
        causal=causal,
        batch_size=batch_size,
    )

    # MLP: FC1 (d_model -> d_ff), GELU, FC2 (d_ff -> d_model)
    fc1_flops = 2 * batch_size * seq_len * d_model * d_ff
    gelu_flops = batch_size * seq_len * d_ff
    fc2_flops = 2 * batch_size * seq_len * d_ff * d_model
    mlp_flops = fc1_flops + gelu_flops + fc2_flops

    # 2 LayerNorms (~4 FLOPs/elem) + 2 Residual additions (1 FLOP/elem)
    ln_flops = 2 * (4 * batch_size * seq_len * d_model)
    resid_flops = 2 * (batch_size * seq_len * d_model)

    return attn_flops + mlp_flops + ln_flops + resid_flops


def estimate_transformer_flops(
    seq_len: int,
    n_layer: int,
    d_model: int,
    n_head: int,
    d_ff: int,
    *,
    vocab_size: int | None = None,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for a complete DecoderOnlyTransformer (N layers + optional LM head)."""
    block_flops = estimate_transformer_block_flops(
        seq_len=seq_len,
        d_model=d_model,
        n_head=n_head,
        d_ff=d_ff,
        causal=True,
        batch_size=batch_size,
    )
    total = n_layer * block_flops

    if vocab_size is not None and vocab_size > 0:
        head_flops = estimate_linear_flops(
            d_model, vocab_size, num_tokens=batch_size * seq_len, bias=False
        )
        total += head_flops

    return total


def estimate_router_flops(
    d_model: int,
    score_dim: int,
    num_candidates: int,
    *,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for top-k Router forward pass.

    Components:
    1. Query projection: Linear(d_model, score_dim) for z_task: 2 * B * d_model * score_dim
    2. Dot product scoring against C candidate keys: 2 * B * C * score_dim
    3. Softmax over C candidates: ~3 * B * C
    4. Top-k candidate selection: ~B * C
    """
    b = batch_size
    query_flops = 2 * b * d_model * score_dim
    score_flops = 2 * b * num_candidates * score_dim
    softmax_flops = 3 * b * num_candidates
    topk_flops = b * num_candidates
    return query_flops + score_flops + softmax_flops + topk_flops


def estimate_pointwise_primitive_flops(
    seq_len: int,
    d_model: int,
    rank: int,
    *,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for PointwisePrimitive: h + gate * B(A(h))."""
    b = batch_size
    seq_l = seq_len
    a_flops = 2 * b * seq_l * d_model * rank
    b_flops = 2 * b * seq_l * rank * d_model
    residual_flops = 2 * b * seq_l * d_model  # gate scaling + addition
    return a_flops + b_flops + residual_flops


def estimate_cross_position_primitive_flops(
    seq_len_in: int,
    seq_len_out: int,
    d_model: int,
    d_operator: int,
    n_head: int,
    d_operator_ff: int,
    vocab_size: int,
    *,
    arg_dim: int = 16,
    has_args: bool = True,
    batch_size: int = 1,
) -> int:
    """Analytical FLOPs for CrossPositionPrimitive (Branch B parameterized compact operator)."""
    b = batch_size
    l_in = seq_len_in
    l_out = seq_len_out
    d_op = d_operator

    # Content input projection: Linear(d_model, d_operator)
    content_in_flops = 2 * b * l_in * d_model * d_op

    # Argument projection (if parameterized)
    arg_flops = (2 * b * arg_dim * d_op) if has_args else 0

    # Cross attention: queries L_out, keys/values L_in
    cross_attn_flops = estimate_attention_flops(
        seq_len_q=l_out,
        seq_len_kv=l_in,
        d_model=d_op,
        n_head=n_head,
        causal=False,
        batch_size=b,
    )

    # Feed-forward network
    mlp_flops = (
        2 * b * l_out * d_op * d_operator_ff  # fc1
        + b * l_out * d_operator_ff  # gelu
        + 2 * b * l_out * d_operator_ff * d_op  # fc2
    )

    # Readout head to vocab_size
    readout_flops = 2 * b * l_out * d_op * vocab_size

    return content_in_flops + arg_flops + cross_attn_flops + mlp_flops + readout_flops


def estimate_primitive_flops(
    primitive: PrimitiveBase,
    seq_len_in: int,
    seq_len_out: int,
    *,
    batch_size: int = 1,
) -> int:
    """Dispatch FLOPs calculation based on concrete primitive architecture."""
    if isinstance(
        primitive,
        (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
    ):
        cfg = primitive.config
        has_args = getattr(primitive, "arg_proj", None) is not None
        arg_dim = getattr(cfg, "arg_dim", 16)
        return estimate_cross_position_primitive_flops(
            seq_len_in=seq_len_in,
            seq_len_out=seq_len_out,
            d_model=cfg.d_model,
            d_operator=cfg.d_operator,
            n_head=cfg.n_head,
            d_operator_ff=cfg.d_operator_ff,
            vocab_size=cfg.vocab_size,
            arg_dim=arg_dim,
            has_args=has_args,
            batch_size=batch_size,
        )
    if isinstance(primitive, PointwisePrimitive):
        return estimate_pointwise_primitive_flops(
            seq_len=seq_len_in,
            d_model=primitive.config.d_model,
            rank=primitive.config.rank,
            batch_size=batch_size,
        )
    # Default fallback estimate using parameter count * sequence length
    return 2 * batch_size * seq_len_in * primitive.num_parameters()


@dataclass(frozen=True)
class FLOPsBreakdown:
    """Analytical FLOPs breakdown comparing sparse execution against dense baseline."""

    task_encoder_flops: int
    router_flops: int
    content_encoder_flops: int
    selected_primitive_flops: int
    decoder_flops: int
    total_sparse_flops: int
    dense_primitive_flops: int
    dense_baseline_flops: int
    flops_savings_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_flops_breakdown(
    core: Any,
    router: Any | None,
    bank: PrimitiveBank,
    selected_ids: Sequence[int],
    *,
    seq_len_task: int,
    seq_len_content: int,
    seq_len_out: int,
    batch_size: int = 1,
) -> FLOPsBreakdown:
    """Compute comprehensive FLOPs breakdown for sparse execution vs dense baseline.

    Args:
        core: Stable Core model (DecoderOnlyTransformer)
        router: Learned top-k Router
        bank: PrimitiveBank containing resident primitives
        selected_ids: IDs of primitives selected for top-k execution
        seq_len_task: Token sequence length for task specification
        seq_len_content: Token sequence length for input content
        seq_len_out: Target output sequence length
        batch_size: Number of examples in the batch
    """
    model_cfg = core.model.config

    # 1. Task encoding FLOPs (Stable Core encoder on task spec tokens)
    task_enc_flops = estimate_transformer_flops(
        seq_len=seq_len_task,
        n_layer=model_cfg.n_layer,
        d_model=model_cfg.d_model,
        n_head=model_cfg.n_head,
        d_ff=model_cfg.d_ff,
        vocab_size=None,
        batch_size=batch_size,
    )

    # 2. Router scoring FLOPs
    if router is not None and hasattr(router, "config"):
        score_dim = router.config.resolved_score_dim
        num_cand = len(bank)
        router_flops = estimate_router_flops(
            d_model=model_cfg.d_model,
            score_dim=score_dim,
            num_candidates=num_cand,
            batch_size=batch_size,
        )
    else:
        router_flops = 0

    # 3. Content encoding FLOPs (task-blind Stable Core encoder)
    content_enc_flops = estimate_transformer_flops(
        seq_len=seq_len_content,
        n_layer=model_cfg.n_layer,
        d_model=model_cfg.d_model,
        n_head=model_cfg.n_head,
        d_ff=model_cfg.d_ff,
        vocab_size=None,
        batch_size=batch_size,
    )

    # 4. Selected primitive FLOPs
    selected_prim_flops = 0
    for pid in set(selected_ids):
        p = bank.get(pid)
        selected_prim_flops += estimate_primitive_flops(
            p,
            seq_len_in=seq_len_content,
            seq_len_out=seq_len_out,
            batch_size=batch_size,
        )

    # 5. Decoder FLOPs (readout head if not already inside primitive)
    # CrossPositionPrimitive includes its own readout head; PointwisePrimitive uses core head
    has_own_readout = any(
        isinstance(
            bank.get(pid),
            (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
        )
        for pid in set(selected_ids)
    )
    if not has_own_readout:
        decoder_flops = estimate_linear_flops(
            model_cfg.d_model,
            model_cfg.vocab_size,
            num_tokens=batch_size * seq_len_out,
            bias=False,
        )
    else:
        decoder_flops = 0

    # 6. Dense baseline FLOPs (executing ALL enabled resident primitives in bank)
    dense_prim_flops = 0
    for pid in bank.ids():
        p = bank.get(pid)
        if p.enabled and p.status == PrimitiveStatus.STABLE:
            dense_prim_flops += estimate_primitive_flops(
                p,
                seq_len_in=seq_len_content,
                seq_len_out=seq_len_out,
                batch_size=batch_size,
            )

    total_sparse = (
        task_enc_flops
        + router_flops
        + content_enc_flops
        + selected_prim_flops
        + decoder_flops
    )
    dense_baseline = (
        task_enc_flops
        + router_flops
        + content_enc_flops
        + dense_prim_flops
        + decoder_flops
    )

    savings = 1.0 - (total_sparse / dense_baseline) if dense_baseline > 0 else 0.0

    return FLOPsBreakdown(
        task_encoder_flops=task_enc_flops,
        router_flops=router_flops,
        content_encoder_flops=content_enc_flops,
        selected_primitive_flops=selected_prim_flops,
        decoder_flops=decoder_flops,
        total_sparse_flops=total_sparse,
        dense_primitive_flops=dense_prim_flops,
        dense_baseline_flops=dense_baseline,
        flops_savings_ratio=savings,
    )


# ---------------------------------------------------------------------------
# 3. Latency Profiling & Memory Hooks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyMetrics:
    """Runtime latency and throughput statistics."""

    median_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_examples_per_sec: float
    peak_gpu_memory_bytes: int
    total_trials: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_execution(
    fn: Callable[[], Any],
    *,
    warmup_steps: int = 5,
    active_steps: int = 20,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Any, LatencyMetrics]:
    """Measure latency and peak memory of a callable with warmup and repeated trials.

    Uses CUDA events if CUDA is active on the target device, otherwise falls
    back to high-resolution monotonic wall-clock timing.

    Args:
        fn: Zero-argument callable to profile.
        warmup_steps: Iterations run to stabilize JIT/cache before measurement.
        active_steps: Measured execution iterations.
        batch_size: Number of examples processed per call.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        tuple of (last_function_output, LatencyMetrics)
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if active_steps < 1:
        raise ValueError("active_steps must be >= 1")

    use_cuda = (
        torch.cuda.is_available()
        and (device is None or str(device).startswith("cuda"))
    )

    # Warmup runs
    result = None
    for _ in range(warmup_steps):
        result = fn()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings_ms: list[float] = []

    if use_cuda:
        for _ in range(active_steps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = fn()
            end_event.record()

            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            timings_ms.append(elapsed)
        peak_vram = torch.cuda.max_memory_allocated()
    else:
        for _ in range(active_steps):
            t0 = time.perf_counter_ns()
            result = fn()
            t1 = time.perf_counter_ns()
            elapsed = (t1 - t0) / 1_000_000.0  # convert ns to ms
            timings_ms.append(elapsed)
        peak_vram = 0

    sorted_timings = sorted(timings_ms)
    n = len(sorted_timings)

    # Percentiles
    median_val = statistics.median(sorted_timings)
    p95_idx = min(n - 1, math.ceil(0.95 * n) - 1)
    p99_idx = min(n - 1, math.ceil(0.99 * n) - 1)
    p95_val = sorted_timings[p95_idx]
    p99_val = sorted_timings[p99_idx]

    mean_val = statistics.fmean(sorted_timings)
    std_val = statistics.stdev(sorted_timings) if n > 1 else 0.0
    min_val = sorted_timings[0]
    max_val = sorted_timings[-1]

    # Throughput: examples per second based on mean duration
    dur_sec = mean_val / 1000.0
    throughput = (batch_size / dur_sec) if dur_sec > 0 else 0.0

    metrics = LatencyMetrics(
        median_ms=median_val,
        p95_ms=p95_val,
        p99_ms=p99_val,
        mean_ms=mean_val,
        std_ms=std_val,
        min_ms=min_val,
        max_ms=max_val,
        throughput_examples_per_sec=throughput,
        peak_gpu_memory_bytes=peak_vram,
        total_trials=active_steps,
    )

    return result, metrics


class ExecutionTimer:
    """Lightweight context manager for measuring component-level latencies."""

    def __init__(self) -> None:
        self.durations_ms: dict[str, list[float]] = {}

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            t1 = time.perf_counter_ns()
            elapsed_ms = (t1 - t0) / 1_000_000.0
            self.durations_ms.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> dict[str, float]:
        """Return mean duration in ms per timed block."""
        return {
            name: statistics.fmean(durations)
            for name, durations in self.durations_ms.items()
        }


# ---------------------------------------------------------------------------
# 4. Dense Primitive Baseline Execution Helper
# ---------------------------------------------------------------------------


def execute_dense_primitive_baseline(
    bank: PrimitiveBank,
    h_content: torch.Tensor,
    lengths: list[int],
    out_lengths: list[int],
    *,
    arg_values: list[Any] | None = None,
) -> dict[int, torch.Tensor]:
    """Execute every enabled STABLE primitive in the bank to form an executable dense baseline.

    Satisfies Phase A.2 requirement (COMPUTE_ACCOUNTING_PHASE_A2.md section 3):
    'Dense primitive baseline must actually execute all resident primitive modules.
     Do not estimate dense latency by multiplying one primitive latency if an executable
     dense baseline is feasible.'

    Returns:
        dict mapping primitive_id -> output tensor.
    """
    outputs: dict[int, torch.Tensor] = {}
    for pid in bank.ids_by_status(PrimitiveStatus.STABLE):
        primitive = bank.get(pid)
        if not primitive.enabled:
            continue

        if isinstance(
            primitive,
            (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
        ):
            out = primitive(h_content, lengths, out_lengths, arg_values)
        else:
            out = primitive(h_content)
        outputs[pid] = out

    return outputs


# ---------------------------------------------------------------------------
# 5. Sparse Execution Verification
# ---------------------------------------------------------------------------


def verify_sparse_execution(
    bank: PrimitiveBank,
    selected_pids: Iterable[int],
    *,
    expected_selected_calls: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify strict sparse execution invariants for top-k routing.

    Per COMPUTE_ACCOUNTING_PHASE_A2.md section 4:
    - selected primitive calls == batch/task calls (if expected_selected_calls given)
    - unselected primitive calls == 0
    """
    selected_set = {int(pid) for pid in selected_pids}
    violations: list[dict[str, Any]] = []
    all_counts = bank.forward_call_counts()

    for pid, count in all_counts.items():
        if pid not in selected_set and count > 0:
            violations.append(
                {"primitive_id": pid, "type": "unselected_called", "calls": count}
            )
        elif pid in selected_set and expected_selected_calls is not None:
            if count != expected_selected_calls:
                violations.append(
                    {
                        "primitive_id": pid,
                        "type": "selected_call_mismatch",
                        "expected": expected_selected_calls,
                        "actual": count,
                    }
                )

    passed = len(violations) == 0
    details = {
        "passed": passed,
        "selected_primitive_ids": sorted(selected_set),
        "forward_call_counts": all_counts,
        "violations": violations,
    }
    return passed, details
