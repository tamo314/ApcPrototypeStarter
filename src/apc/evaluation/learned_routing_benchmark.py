"""Learned Routing & Full Closed Loop Benchmark (Task A1-B008 / Milestone B-M8).

Replaces oracle routing with a learned top-k router conditioned on z_task.
Demonstrates autonomous sparse execution, compute savings, and end-to-end continual learning.

Scientific Hypotheses & Acceptance Criteria (H-B6 / Milestone B-M8):
1. Top-k router selects oracle-required primitives with high accuracy (>= 95%).
2. Active compute significantly lower than dense execution (sparse execution restricts
   forward calls exclusively to selected primitives).
3. Full lifelong closed-loop benchmark passes without human intervention (overall mean EM >= 0.90
   across all 10 operations in the lifelong benchmark universe: 8 canonical + 2 consolidated novel).
4. Multi-seed evaluation across 5 seeds (0, 1, 2, 3, 4).
"""

from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.consolidation.compact_consolidation import collate_content_only_batch
from apc.core.data import build_task_only_tokens, pad_token_sequences
from apc.environments.generator import Example
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    get_operation,
)
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.recurrence_benchmark import (
    generate_benchmark_examples,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    PARAMETERIZED_OPERATION_NAMES,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
    _train_single_primitive,
)
from apc.plastic.residual import verify_frozen_invariants
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

MIN_GATE_SEEDS: Final[int] = 5
ROUTING_ACCURACY_THRESHOLD: Final[float] = 0.95
CLOSED_LOOP_EM_THRESHOLD: Final[float] = 0.90
MAX_ALLOCATED_PLASTIC_PARAMS: Final[int] = 0
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

ALL_BENCHMARK_OPERATIONS: Final[tuple[str, ...]] = (
    *ALL_CANONICAL_OPERATIONS,
    *BRANCH_B_NOVEL_OPERATION_NAMES,
)


@dataclass(frozen=True)
class LearnedRoutingBenchmarkConfig:
    """Explicit configuration for Task A1-B008 benchmark."""

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    device: str = "auto"

    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
    max_sequence_length: int = 32

    # Benchmark operations: all 10 canonical and consolidated operations
    operations: tuple[str, ...] = ALL_BENCHMARK_OPERATIONS

    # Router architecture and calibration
    top_k: int = 1
    router_train_examples_per_op: int = 64
    router_train_steps: int = 400
    router_lr: float = 0.005

    # Evaluation budget per operation
    num_eval_examples: int = 200
    min_eval_examples: int = 200

    # Core & bank training steps fallback
    core_train_steps: int = 6000
    bank_train_steps: int = 6000

    routing_threshold: float = ROUTING_ACCURACY_THRESHOLD
    accuracy_threshold: float = CLOSED_LOOP_EM_THRESHOLD

    consolidation_checkpoint_dir: str | None = "runs/phase_a1_consolidation_benchmark"
    shared_encoder_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.num_eval_examples < self.min_eval_examples:
            raise ValueError(f"num_eval_examples must be >= {self.min_eval_examples}")
        if not self.operations:
            raise ValueError("operations must be non-empty")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def routing_config_from_dict(raw: dict[str, Any]) -> LearnedRoutingBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = LearnedRoutingBenchmarkConfig()
    return LearnedRoutingBenchmarkConfig(
        seed=raw.get("seed", defaults.seed),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        model=dict(raw.get("model", defaults.model)),
        device=raw.get("device", defaults.device),
        d_operator=raw.get("d_operator", defaults.d_operator),
        n_operator_head=raw.get("n_operator_head", defaults.n_operator_head),
        d_operator_ff=raw.get("d_operator_ff", defaults.d_operator_ff),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        operations=tuple(raw.get("operations", defaults.operations)),
        top_k=raw.get("top_k", defaults.top_k),
        router_train_examples_per_op=raw.get(
            "router_train_examples_per_op", defaults.router_train_examples_per_op
        ),
        router_train_steps=raw.get("router_train_steps", defaults.router_train_steps),
        router_lr=raw.get("router_lr", defaults.router_lr),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        min_eval_examples=raw.get("min_eval_examples", defaults.min_eval_examples),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        routing_threshold=raw.get("routing_threshold", defaults.routing_threshold),
        accuracy_threshold=raw.get("accuracy_threshold", defaults.accuracy_threshold),
        consolidation_checkpoint_dir=raw.get(
            "consolidation_checkpoint_dir", defaults.consolidation_checkpoint_dir
        ),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
    )


def extract_task_representations(
    core: Any,
    examples: Sequence[Example],
) -> torch.Tensor:
    """Extract z_task from task specifications using frozen Stable Core.

    Strict Invariant:
    Reads only `example.task_spec`, completely orthogonal to `example.input_tokens`.
    Extracts the hidden representation at the [TASK_END] token index.
    """
    device = core.device
    tokens = core.tokens
    task_token_seqs = []
    for ex in examples:
        assert ex.task_spec is not None
        task_token_seqs.append(build_task_only_tokens(ex.task_spec, tokens))
    padded_task_ids = pad_token_sequences(task_token_seqs, tokens.pad, device)

    with torch.no_grad():
        encoded = core.model.encode(padded_task_ids)

    # Locate [TASK_END] token index for each row
    task_end_mask = padded_task_ids == tokens.task_end
    task_end_indices = task_end_mask.to(torch.long).argmax(dim=-1)

    batch_indices = torch.arange(len(examples), device=device)
    z_task = encoded[batch_indices, task_end_indices, :]
    return z_task


def calibrate_task_router(
    router: Router,
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    operations: Sequence[str],
    *,
    examples_per_op: int = 64,
    steps: int = 400,
    lr: float = 0.005,
    seed: int = 0,
) -> None:
    """Calibrate router parameters using z_task vectors across the benchmark universe.

    Strict Invariants:
    1. Only router parameters are updated (Core and Bank remain 100% frozen).
    2. Input is purely z_task; no content tokens are involved.
    3. Usage statistics on router are reset to 0 after calibration.
    """
    device = core.device
    candidate_ids = bank.ids()

    # Ensure all candidate primitives are registered in router
    for pid in candidate_ids:
        if not router.has_primitive(pid):
            router.add_primitive_key(pid)

    router.to(device)
    router.train()

    # Generate training task specifications grouped per operation
    op_training_data: dict[str, list[tuple[torch.Tensor, int]]] = {}
    for op_name in operations:
        pid = op_to_id[op_name]
        examples = generate_benchmark_examples(
            seed=seed + 999,
            n=examples_per_op,
            operation=op_name,
            split="train",
            vocab_size=core.tokens.env_vocab_size,
        )
        z_tasks = extract_task_representations(core, examples)
        op_training_data[op_name] = [(z_tasks[i], pid) for i in range(len(examples))]

    # Convert candidate IDs to class index map
    pid_to_class_idx = {pid: idx for idx, pid in enumerate(candidate_ids)}

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    rng = random.Random(seed * 4001 + 17)

    # Balanced per-operation sampling: 4 examples per operation per step
    for _ in range(steps):
        batch: list[tuple[torch.Tensor, int]] = []
        for op_name in operations:
            items = op_training_data[op_name]
            for _ in range(4):
                batch.append(rng.choice(items))

        z_batch = torch.stack([item[0] for item in batch], dim=0)
        target_classes = torch.tensor(
            [pid_to_class_idx[item[1]] for item in batch], dtype=torch.long, device=device
        )

        optimizer.zero_grad(set_to_none=True)
        # Compute scores against candidate keys
        keys = router._stacked_keys(candidate_ids)  # [C, score_dim]
        query = router.query_proj(z_batch)  # [B, score_dim]
        scores = query @ keys.transpose(0, 1)  # [B, C]

        loss = F.cross_entropy(scores, target_classes)
        loss.backward()
        optimizer.step()
        scheduler.step()

    router.eval()
    # Reset usage count so calibration never pollutes execution accounting
    for pid in candidate_ids:
        router.usage_count[pid] = 0


@dataclass(frozen=True)
class OperationBenchmarkResult:
    """Outcome of evaluating one operation under learned routing."""

    operation: str
    oracle_primitive_id: int
    selected_primitive_id_mode: int
    router_top1_accuracy: float
    router_topk_accuracy: float
    mean_routing_entropy: float
    exact_match: float
    token_accuracy: float
    active_primitive_parameters: int
    dense_primitive_parameters: int
    compute_savings_ratio: float
    sparse_execution_passed: bool
    accuracy_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LearnedRoutingBenchmarkReport:
    """Evaluation report for a single seed of Task A1-B008."""

    config: LearnedRoutingBenchmarkConfig
    seed: int
    core_param_count: int
    bank_param_count: int
    bank_size: int
    router_param_count: int
    mean_router_top1_accuracy: float
    mean_router_topk_accuracy: float
    mean_exact_match: float
    mean_token_accuracy: float
    active_primitive_parameters: int
    dense_primitive_parameters: int
    compute_savings_ratio: float
    max_allocated_plastic_parameters: int
    per_operation_results: dict[str, OperationBenchmarkResult]
    all_routing_passed: bool
    all_accuracy_passed: bool
    all_invariants_passed: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "bank_param_count": self.bank_param_count,
            "bank_size": self.bank_size,
            "router_param_count": self.router_param_count,
            "mean_router_top1_accuracy": self.mean_router_top1_accuracy,
            "mean_router_topk_accuracy": self.mean_router_topk_accuracy,
            "mean_exact_match": self.mean_exact_match,
            "mean_token_accuracy": self.mean_token_accuracy,
            "active_primitive_parameters": self.active_primitive_parameters,
            "dense_primitive_parameters": self.dense_primitive_parameters,
            "compute_savings_ratio": self.compute_savings_ratio,
            "max_allocated_plastic_parameters": self.max_allocated_plastic_parameters,
            "per_operation_results": {
                k: v.to_dict() for k, v in self.per_operation_results.items()
            },
            "all_routing_passed": self.all_routing_passed,
            "all_accuracy_passed": self.all_accuracy_passed,
            "all_invariants_passed": self.all_invariants_passed,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _verify_sparse_routing_execution(
    bank: PrimitiveBank, selected_pids: set[int]
) -> bool:
    """Verify that unselected primitives strictly received zero forward calls."""
    passed = True
    for pid in bank.ids():
        p = bank.get(pid)
        if pid not in selected_pids and p.forward_call_count > 0:
            passed = False
    return passed


def extract_operation_argument(op_name: str, arguments: dict[str, Any]) -> Any:
    """Extract argument value for parameterized operation, falling back to default if misrouted."""
    if op_name == "SHIFT":
        return arguments.get("amount", 0)
    if op_name == "SELECT":
        indices = arguments.get("indices", (0,))
        return indices if isinstance(indices, (tuple, list)) and indices else (0,)
    if op_name == "COUNT":
        return arguments.get("target", 0)
    if op_name == "BIND":
        return arguments.get("query_key", 0)
    return None


def execute_routed_batch(
    core: Any,
    bank: PrimitiveBank,
    router: Router,
    id_to_op: dict[int, str],
    examples: Sequence[Example],
    *,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
    """Execute a batch of examples through learned routing and sparse execution.

    Args:
        core: SharedContentEncoder (frozen Stable Core).
        bank: PrimitiveBank containing all candidate primitives.
        router: Calibrated learned Router.
        id_to_op: Mapping from primitive_id to operation name.
        examples: Batch of examples to evaluate.
        batch_size: Sub-batch size.

    Returns:
        tuple of:
        - all_selected_ids: [N, top_k] tensor of selected primitive IDs
        - all_probs: [N, C] tensor of router probabilities
        - all_entropies: [N] tensor of routing entropy
        - all_preds: list of predicted token sequences per example
    """
    device = core.device
    candidate_ids = bank.ids()

    all_selected_ids_list: list[torch.Tensor] = []
    all_probs_list: list[torch.Tensor] = []
    all_entropies_list: list[torch.Tensor] = []
    all_preds_list: list[tuple[int, ...]] = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        n_chunk = len(chunk)

        # 1. Extract z_task for the chunk
        z_task = extract_task_representations(core, chunk)

        # 2. Query learned router
        router_out = router(z_task, candidate_ids)
        all_selected_ids_list.append(router_out.selected_ids)
        all_probs_list.append(router_out.probs)
        all_entropies_list.append(router_out.entropy)

        # Top-1 selected primitive per example
        top1_pids = router_out.selected_ids[:, 0].tolist()

        # 3. Task-blind content encoding
        lengths = [len(ex.input_tokens) for ex in chunk]
        out_lengths = [len(ex.target_tokens) for ex in chunk]
        lmax = max(lengths)
        content_ids = collate_content_only_batch(chunk, core.tokens, device=device)

        with torch.no_grad():
            content_state = core.model.encode(content_ids)
            h_content = content_state[:, 1 : 1 + lmax, :]

        # 4. Sparse execution: group by top-1 selected primitive
        # Only the selected primitive executes; unselected primitives receive 0 calls.
        chunk_preds: list[tuple[int, ...] | None] = [None] * n_chunk
        unique_selected = set(top1_pids)

        for pid in unique_selected:
            indices = [i for i, p in enumerate(top1_pids) if p == pid]
            sub_examples = [chunk[i] for i in indices]
            sub_h = h_content[indices]
            sub_lengths = [lengths[i] for i in indices]
            sub_out_lengths = [out_lengths[i] for i in indices]

            primitive = bank.get(pid)
            op_name = id_to_op[pid]
            op_def = get_operation(op_name)

            # Resolve arguments if parameterized
            if op_def.required_argument_names:
                arg_values = []
                for ex in sub_examples:
                    assert ex.task_spec is not None
                    spec_step = ex.task_spec.steps[0]
                    arg_values.append(extract_operation_argument(op_name, spec_step.arguments))
            else:
                arg_values = None

            with torch.no_grad():
                if isinstance(
                    primitive,
                    (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive),
                ):
                    sub_logits = primitive(sub_h, sub_lengths, sub_out_lengths, arg_values)
                else:
                    sub_logits = primitive(sub_h)

                sub_preds = sub_logits.argmax(dim=-1)
                for local_idx, global_idx in enumerate(indices):
                    target_len = sub_out_lengths[local_idx]
                    pred_tokens = tuple(sub_preds[local_idx, :target_len].tolist())
                    chunk_preds[global_idx] = pred_tokens

        for p in chunk_preds:
            assert p is not None
            all_preds_list.append(p)

    all_selected_ids = torch.cat(all_selected_ids_list, dim=0)
    all_probs = torch.cat(all_probs_list, dim=0)
    all_entropies = torch.cat(all_entropies_list, dim=0)

    return all_selected_ids, all_probs, all_entropies, all_preds_list


def _ensure_learned_routing_bank_and_core(
    config: LearnedRoutingBenchmarkConfig,
    seed_dir: Path | None = None,
) -> tuple[Any, PrimitiveBank, dict[str, int]]:
    """Retrieve or construct the frozen task-blind Stable Core and 10-primitive bank."""
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    if config.device not in ("auto", None):
        device = config.device

    # 1. Obtain frozen shared encoder
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        operation_names=PARAMETERIZED_OPERATION_NAMES,
        group_size=config.group_size,
        model=config.model,
        device=device,
        d_operator=config.d_operator,
        n_operator_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        arg_dim=16,
        max_sequence_length=config.max_sequence_length,
    )
    arch = build_shared_encoder_architecture(arch_cfg)

    # Locate encoder checkpoint
    ckpt_path: Path | None = None
    if config.shared_encoder_checkpoint:
        cand = Path(config.shared_encoder_checkpoint)
        if cand.is_file():
            ckpt_path = cand
    if ckpt_path is None:
        cand = (
            Path("runs")
            / "phase_a1_shift_compact_structural_probe"
            / f"seed_{config.seed}"
            / "shared_encoder.pt"
        )
        if cand.is_file():
            ckpt_path = cand
    if ckpt_path is None and seed_dir is not None:
        cand = seed_dir / "shared_encoder.pt"
        if cand.is_file():
            ckpt_path = cand

    if ckpt_path is not None and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=arch.core.device, weights_only=True)
        with torch.no_grad():
            for key in ["token_emb.weight", "head.weight"]:
                if key in state and state[key].shape != arch.core.model.state_dict()[key].shape:
                    cur_w = arch.core.model.state_dict()[key].clone()
                    cur_w[: state[key].shape[0]] = state[key]
                    state[key] = cur_w
        arch.core.model.load_state_dict(state)
        core = arch.core
    else:
        # Fallback to standard core training
        u_cfg = UnifiedBenchmarkConfig(
            seed=config.seed,
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
            group_size=config.group_size,
            model=config.model,
            device=device,
            d_operator=config.d_operator,
            n_operator_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            max_sequence_length=config.max_sequence_length,
            core_train_steps=config.core_train_steps,
        )
        core = _get_or_train_frozen_shared_core(u_cfg, seed_dir=seed_dir)

    core.model.eval()
    for param in core.model.parameters():
        param.requires_grad_(False)

    # 2. Build 10-primitive bank (8 canonical + 2 novel operations)
    u_bank_cfg = UnifiedBenchmarkConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=device,
        d_operator=config.d_operator,
        n_operator_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        max_sequence_length=config.max_sequence_length,
    )
    bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
    for op_name in BRANCH_B_NOVEL_OPERATION_NAMES:
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op_name,
                d_model=core.model.config.d_model,
                d_operator=config.d_operator,
                n_head=config.n_operator_head,
                d_operator_ff=config.d_operator_ff,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_sequence_length,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op_name] = p.primitive_id

    bank.to(core.device)

    # Check if seed_dir has an existing 10-primitive bank
    loaded_full_bank = False
    if seed_dir is not None and (seed_dir / "primitive_bank.pt").is_file():
        try:
            full_sd = torch.load(
                seed_dir / "primitive_bank.pt", map_location=core.device, weights_only=True
            )
            pids = set(int(k.split(".")[1]) for k in full_sd if k.startswith("_primitives."))
            if len(pids) == 10:
                bank.load_state_dict(full_sd)
                loaded_full_bank = True
        except Exception:
            loaded_full_bank = False

    if not loaded_full_bank:
        # Load canonical primitives (0-7) from composition library benchmark if available
        comp_bank_ckpt = (
            Path("runs")
            / "phase_a1_composition_library_benchmark"
            / f"seed_{config.seed}"
            / "primitive_bank.pt"
        )
        if comp_bank_ckpt.is_file():
            canon_sd = torch.load(comp_bank_ckpt, map_location=core.device, weights_only=True)
            bank.load_state_dict(canon_sd, strict=False)
        else:
            for op in ALL_CANONICAL_OPERATIONS:
                prim = bank.get(op_to_id[op])
                steps = (
                    config.bank_train_steps
                    if op in PARAMETERIZED_OPERATION_NAMES
                    else config.bank_train_steps // 2
                )
                _train_single_primitive(core, prim, u_bank_cfg, op, steps=steps)

        # Train novel primitives (8 and 9) if not loaded
        for op in BRANCH_B_NOVEL_OPERATION_NAMES:
            prim = bank.get(op_to_id[op])
            _train_single_primitive(core, prim, u_bank_cfg, op, steps=3000)

        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bank.state_dict(), seed_dir / "primitive_bank.pt")

    bank.freeze_all()
    bank.eval()

    verify_frozen_invariants(core, bank)

    return core, bank, op_to_id


def run_learned_routing_benchmark(
    config: LearnedRoutingBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> LearnedRoutingBenchmarkReport:
    """Execute Task A1-B008 Learned Routing & Full Closed Loop benchmark for one seed."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    # 1. Obtain Core and 10-primitive Bank
    core, bank, op_to_id = _ensure_learned_routing_bank_and_core(
        config, seed_dir=seed_dir
    )
    id_to_op = {v: k for k, v in op_to_id.items()}

    core_params = sum(p.numel() for p in core.model.parameters())
    bank_params = bank.total_parameter_count()
    bank_size = len(bank)

    # 2. Build and calibrate Learned Router
    router_config = RouterConfig(
        d_model=core.model.config.d_model,
        top_k=config.top_k,
        score_fn="dot",
    )
    router = Router(router_config)
    for pid in bank.ids():
        router.add_primitive_key(pid)
    router.to(core.device)

    calibrate_task_router(
        router,
        core,
        bank,
        op_to_id,
        operations=config.operations,
        examples_per_op=config.router_train_examples_per_op,
        steps=config.router_train_steps,
        lr=config.router_lr,
        seed=config.seed,
    )
    router_param_count = sum(p.numel() for p in router.parameters())

    # Verify workspace is empty (zero plastic parameters allocated)
    workspace = PlasticWorkspace().to(core.device)
    allocated_plastic_params = workspace.total_parameter_count()

    # 3. Evaluate Closed Loop on held-out test sets across all operations
    per_op_results: dict[str, OperationBenchmarkResult] = {}
    top1_accuracies: list[float] = []
    topk_accuracies: list[float] = []
    exact_matches: list[float] = []
    token_accuracies: list[float] = []
    active_param_list: list[int] = []

    for op_name in config.operations:
        oracle_pid = op_to_id[op_name]
        eval_examples = generate_benchmark_examples(
            config.seed,
            config.num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )

        # Reset call counts for clean sparse call tracking
        for pid in bank.ids():
            bank.get(pid).reset_forward_call_count()

        selected_ids, probs, entropies, preds = execute_routed_batch(
            core, bank, router, id_to_op, eval_examples
        )

        top1_pids = selected_ids[:, 0].tolist()
        topk_pids = selected_ids.tolist()

        # Measure routing accuracy
        top1_correct = sum(1 for p in top1_pids if p == oracle_pid)
        topk_correct = sum(1 for k_list in topk_pids if oracle_pid in k_list)
        top1_acc = top1_correct / len(eval_examples)
        topk_acc = topk_correct / len(eval_examples)
        mean_entropy = entropies.mean().item()

        # Measure exact match and token accuracy
        correct_em = 0
        total_tokens = 0
        tok_correct = 0
        for row, ex in enumerate(eval_examples):
            pred_tokens = preds[row]
            target_tokens = ex.target_tokens
            if pred_tokens == target_tokens:
                correct_em += 1
            total_tokens += len(target_tokens)
            tok_correct += sum(
                1 for a, b in zip(pred_tokens, target_tokens, strict=True) if a == b
            )

        em = correct_em / len(eval_examples)
        tok_acc = tok_correct / max(1, total_tokens)

        # Sparse execution verification
        selected_set = set(top1_pids)
        sparse_passed = _verify_sparse_routing_execution(bank, selected_set)

        # Compute accounting
        active_params = sum(
            bank.get(p).num_parameters() for p in selected_set
        ) // max(1, len(selected_set))
        savings_ratio = 1.0 - (active_params / bank_params)

        accuracy_passed = em >= config.accuracy_threshold
        routing_passed = top1_acc >= config.routing_threshold
        op_passed = accuracy_passed and routing_passed and sparse_passed

        # Mode of selected PID
        selected_mode = statistics.mode(top1_pids)

        per_op_results[op_name] = OperationBenchmarkResult(
            operation=op_name,
            oracle_primitive_id=oracle_pid,
            selected_primitive_id_mode=selected_mode,
            router_top1_accuracy=top1_acc,
            router_topk_accuracy=topk_acc,
            mean_routing_entropy=mean_entropy,
            exact_match=em,
            token_accuracy=tok_acc,
            active_primitive_parameters=active_params,
            dense_primitive_parameters=bank_params,
            compute_savings_ratio=savings_ratio,
            sparse_execution_passed=sparse_passed,
            accuracy_passed=accuracy_passed,
            passed=op_passed,
        )

        top1_accuracies.append(top1_acc)
        topk_accuracies.append(topk_acc)
        exact_matches.append(em)
        token_accuracies.append(tok_acc)
        active_param_list.append(active_params)

    # Invariants verification
    core_frozen = verify_frozen_invariants(core, bank)
    mean_top1 = statistics.fmean(top1_accuracies)
    mean_topk = statistics.fmean(topk_accuracies)
    mean_em = statistics.fmean(exact_matches)
    mean_tok = statistics.fmean(token_accuracies)
    mean_active_params = int(statistics.fmean(active_param_list))
    overall_savings = 1.0 - (mean_active_params / bank_params)

    all_routing_passed = mean_top1 >= config.routing_threshold
    all_accuracy_passed = mean_em >= config.accuracy_threshold
    all_invariants_passed = (
        core_frozen
        and allocated_plastic_params == MAX_ALLOCATED_PLASTIC_PARAMS
        and all(r.sparse_execution_passed for r in per_op_results.values())
    )
    overall_passed = all_routing_passed and all_accuracy_passed and all_invariants_passed

    elapsed = time.perf_counter() - start_time

    return LearnedRoutingBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        bank_param_count=bank_params,
        bank_size=bank_size,
        router_param_count=router_param_count,
        mean_router_top1_accuracy=mean_top1,
        mean_router_topk_accuracy=mean_topk,
        mean_exact_match=mean_em,
        mean_token_accuracy=mean_tok,
        active_primitive_parameters=mean_active_params,
        dense_primitive_parameters=bank_params,
        compute_savings_ratio=overall_savings,
        max_allocated_plastic_parameters=allocated_plastic_params,
        per_operation_results=per_op_results,
        all_routing_passed=all_routing_passed,
        all_accuracy_passed=all_accuracy_passed,
        all_invariants_passed=all_invariants_passed,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


@dataclass(frozen=True)
class MultiSeedLearnedRoutingReport:
    """Multi-seed summary for Task A1-B008."""

    seeds: tuple[int, ...]
    per_seed: dict[int, LearnedRoutingBenchmarkReport]
    mean_router_top1_accuracy: float
    mean_router_topk_accuracy: float
    mean_exact_match: float
    mean_token_accuracy: float
    per_operation_means: dict[str, dict[str, float]]
    active_primitive_parameters: int
    dense_primitive_parameters: int
    compute_savings_ratio: float
    max_allocated_plastic_parameters: int
    all_seeds_passed: bool
    meets_seed_policy: bool
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "per_seed": {str(s): r.to_dict() for s, r in self.per_seed.items()},
            "mean_router_top1_accuracy": self.mean_router_top1_accuracy,
            "mean_router_topk_accuracy": self.mean_router_topk_accuracy,
            "mean_exact_match": self.mean_exact_match,
            "mean_token_accuracy": self.mean_token_accuracy,
            "per_operation_means": self.per_operation_means,
            "active_primitive_parameters": self.active_primitive_parameters,
            "dense_primitive_parameters": self.dense_primitive_parameters,
            "compute_savings_ratio": self.compute_savings_ratio,
            "max_allocated_plastic_parameters": self.max_allocated_plastic_parameters,
            "all_seeds_passed": self.all_seeds_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "overall_passed": self.overall_passed,
        }


def run_learned_routing_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Callable[[int], LearnedRoutingBenchmarkConfig],
    *,
    output_dir: Path | None = None,
) -> MultiSeedLearnedRoutingReport:
    """Execute Task A1-B008 across multiple seeds and aggregate."""
    per_seed: dict[int, LearnedRoutingBenchmarkReport] = {}
    for seed in seeds:
        cfg = config_factory(seed)
        seed_dir = output_dir / f"seed_{seed}" if output_dir else None
        if seed_dir:
            seed_dir.mkdir(parents=True, exist_ok=True)
        report = run_learned_routing_benchmark(cfg, seed_dir=seed_dir)
        per_seed[seed] = report
        if seed_dir:
            (seed_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )

    mean_top1 = statistics.fmean(r.mean_router_top1_accuracy for r in per_seed.values())
    mean_topk = statistics.fmean(r.mean_router_topk_accuracy for r in per_seed.values())
    mean_em = statistics.fmean(r.mean_exact_match for r in per_seed.values())
    mean_tok = statistics.fmean(r.mean_token_accuracy for r in per_seed.values())

    first_report = next(iter(per_seed.values()))
    per_op_means: dict[str, dict[str, float]] = {}
    for op_name in first_report.per_operation_results:
        op_top1 = statistics.fmean(
            r.per_operation_results[op_name].router_top1_accuracy for r in per_seed.values()
        )
        op_em = statistics.fmean(
            r.per_operation_results[op_name].exact_match for r in per_seed.values()
        )
        op_tok = statistics.fmean(
            r.per_operation_results[op_name].token_accuracy for r in per_seed.values()
        )
        per_op_means[op_name] = {
            "router_top1_accuracy": op_top1,
            "exact_match": op_em,
            "token_accuracy": op_tok,
        }

    all_passed = all(r.overall_passed for r in per_seed.values())
    meets_policy = len(seeds) >= MIN_GATE_SEEDS
    overall = all_passed and meets_policy

    multi_report = MultiSeedLearnedRoutingReport(
        seeds=tuple(seeds),
        per_seed=per_seed,
        mean_router_top1_accuracy=mean_top1,
        mean_router_topk_accuracy=mean_topk,
        mean_exact_match=mean_em,
        mean_token_accuracy=mean_tok,
        per_operation_means=per_op_means,
        active_primitive_parameters=first_report.active_primitive_parameters,
        dense_primitive_parameters=first_report.dense_primitive_parameters,
        compute_savings_ratio=first_report.compute_savings_ratio,
        max_allocated_plastic_parameters=max(
            r.max_allocated_plastic_parameters for r in per_seed.values()
        ),
        all_seeds_passed=all_passed,
        meets_seed_policy=meets_policy,
        overall_passed=overall,
    )

    if output_dir:
        (output_dir / "report.json").write_text(
            json.dumps(multi_report.to_dict(), indent=2), encoding="utf-8"
        )

    return multi_report
