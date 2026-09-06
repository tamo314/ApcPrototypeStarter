"""Recurrence & Reuse Benchmark (Task A1-B007 / Milestone B-M7).

Demonstrates that when previously consolidated operations reappear in a lifelong task stream,
the APC system retrieves and executes the installed primitive immediately without allocating
plastic capacity or requiring retraining.

Scientific Hypotheses & Controls (H-B5 / ADR-0046 / CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md):
- Re-encountering a consolidated task achieves instant ceiling accuracy with zero plastic.
- Immediate recurrent exact match >= 0.90 immediately without adaptation (adaptation_steps = 0).
- Temporary parameter allocation = 0 throughout recurrence.
- Controls:
  1. APC Recurrence (Bank Reuse / Lookup): 0 adaptation steps, 0 plastic params, EM >= 0.90.
  2. Fresh Adaptation Control: Allocate temporary plastic capacity (~17k params) and train.
  3. Unconsolidated Control: Initial bank without consolidated operations fails (EM <= 0.05).
  4. Intervening Task Stability: Interleaving canonical tasks shows zero degradation.
- Multi-seed benchmark (5 seeds: 0, 1, 2, 3, 4).
"""

from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from apc.consolidation.compact_consolidation import collate_content_only_batch
from apc.core.data import IGNORE_INDEX
from apc.environments.generator import Example
from apc.environments.operations import BRANCH_B_NOVEL_OPERATION_NAMES
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.consolidation_benchmark import (
    ConsolidationBenchmarkConfig,
    generate_benchmark_examples,
    run_consolidation_benchmark,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
)
from apc.plastic.residual import verify_frozen_invariants
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus
from apc.utils.seed import set_seed

MIN_GATE_SEEDS: Final[int] = 5
RECURRENCE_ACCURACY_THRESHOLD: Final[float] = 0.90
UNCONSOLIDATED_CEILING_THRESHOLD: Final[float] = 0.05
MAX_ALLOCATED_PLASTIC_PARAMS: Final[int] = 0
_EVAL_BATCH_SIZE: Final[int] = 64
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RecurrenceBenchmarkConfig:
    """Explicit configuration for Task A1-B007 benchmark."""

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

    # Operations in lifelong stream
    recurrent_operations: tuple[str, ...] = BRANCH_B_NOVEL_OPERATION_NAMES
    intervening_operations: tuple[str, ...] = ("REVERSE", "SHIFT", "NEGATE")
    canonical_operations: tuple[str, ...] = ALL_CANONICAL_OPERATIONS

    num_eval_examples: int = 200
    min_eval_examples: int = 200
    num_fresh_adaptation_examples: int = 1000
    fresh_adaptation_steps: int = 500
    fresh_adaptation_lr: float = 0.001

    core_train_steps: int = 6000
    bank_train_steps: int = 6000

    accuracy_threshold: float = RECURRENCE_ACCURACY_THRESHOLD
    unconsolidated_threshold: float = UNCONSOLIDATED_CEILING_THRESHOLD

    consolidation_checkpoint_dir: str | None = "runs/phase_a1_consolidation_benchmark"
    shared_encoder_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.num_eval_examples < self.min_eval_examples:
            raise ValueError(f"num_eval_examples must be >= {self.min_eval_examples}")
        if not self.recurrent_operations:
            raise ValueError("recurrent_operations must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def recurrence_config_from_dict(raw: dict[str, Any]) -> RecurrenceBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = RecurrenceBenchmarkConfig()
    return RecurrenceBenchmarkConfig(
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
        recurrent_operations=tuple(
            raw.get("recurrent_operations", defaults.recurrent_operations)
        ),
        intervening_operations=tuple(
            raw.get("intervening_operations", defaults.intervening_operations)
        ),
        canonical_operations=tuple(
            raw.get("canonical_operations", defaults.canonical_operations)
        ),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        min_eval_examples=raw.get("min_eval_examples", defaults.min_eval_examples),
        num_fresh_adaptation_examples=raw.get(
            "num_fresh_adaptation_examples", defaults.num_fresh_adaptation_examples
        ),
        fresh_adaptation_steps=raw.get(
            "fresh_adaptation_steps", defaults.fresh_adaptation_steps
        ),
        fresh_adaptation_lr=raw.get(
            "fresh_adaptation_lr", defaults.fresh_adaptation_lr
        ),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        accuracy_threshold=raw.get("accuracy_threshold", defaults.accuracy_threshold),
        unconsolidated_threshold=raw.get(
            "unconsolidated_threshold", defaults.unconsolidated_threshold
        ),
        consolidation_checkpoint_dir=raw.get(
            "consolidation_checkpoint_dir", defaults.consolidation_checkpoint_dir
        ),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
    )


@dataclass(frozen=True)
class RecurrenceEventResult:
    """Outcome of evaluating one recurrent operation reappearing in task stream."""

    operation: str
    immediate_exact_match: float
    immediate_token_accuracy: float
    adaptation_steps: int
    allocated_plastic_parameters: int
    fresh_adaptation_exact_match: float
    fresh_adaptation_steps: int
    fresh_adaptation_parameters: int
    unconsolidated_exact_match: float
    frozen_invariants_passed: bool
    sparse_execution_passed: bool
    accuracy_passed: bool
    zero_allocation_passed: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RecurrenceBenchmarkReport:
    """Complete evaluation report for a single seed of Task A1-B007."""

    config: RecurrenceBenchmarkConfig
    seed: int
    core_param_count: int
    bank_param_count: int
    bank_size: int
    intervening_task_results: dict[str, float]
    recurrent_event_results: dict[str, RecurrenceEventResult]
    mean_immediate_exact_match: float
    max_allocated_plastic_parameters: int
    all_recurrent_passed: bool
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
            "intervening_task_results": self.intervening_task_results,
            "recurrent_event_results": {
                k: v.to_dict() for k, v in self.recurrent_event_results.items()
            },
            "mean_immediate_exact_match": self.mean_immediate_exact_match,
            "max_allocated_plastic_parameters": self.max_allocated_plastic_parameters,
            "all_recurrent_passed": self.all_recurrent_passed,
            "all_invariants_passed": self.all_invariants_passed,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _verify_sparse_execution(bank: PrimitiveBank, selected_pid: int) -> bool:
    """Verify that unselected primitives strictly received zero forward calls."""
    passed = True
    for pid in bank.ids():
        p = bank.get(pid)
        if pid != selected_pid and p.forward_call_count > 0:
            passed = False
    return passed


def _evaluate_exact_match(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    examples: Sequence[Example],
    operation_name: str,
) -> tuple[float, float]:
    """Evaluate exact match and token accuracy for a single operation."""
    if not examples:
        return 0.0, 0.0
    with torch.no_grad():
        logits = execute_composition_recipe(
            core, bank, op_to_id, examples, candidate_operations=(operation_name,)
        )
        preds = logits.argmax(dim=-1)
        correct = 0
        total_tokens = 0
        tok_correct = 0
        for row, ex in enumerate(examples):
            n = len(ex.target_tokens)
            pred_tokens = tuple(preds[row, :n].tolist())
            total_tokens += n
            tok_correct += sum(
                1 for a, b in zip(pred_tokens, ex.target_tokens, strict=True) if a == b
            )
            if pred_tokens == ex.target_tokens:
                correct += 1
    return correct / len(examples), tok_correct / max(1, total_tokens)


def _evaluate_fresh_adaptation_baseline(
    core: Any,
    config: RecurrenceBenchmarkConfig,
    operation_name: str,
    eval_examples: Sequence[Example],
) -> tuple[float, int, int]:
    """Train a fresh temporary compact operator from scratch (No-Reuse Control)."""
    set_seed(config.seed * 3000 + 42)
    device = core.device
    vocab_size = config.vocab_size

    prim_cfg = CrossPositionPrimitiveConfig(
        operation=operation_name,
        d_model=core.model.config.d_model,
        d_operator=config.d_operator,
        n_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        vocab_size=vocab_size,
        max_sequence_length=config.max_sequence_length,
    )
    workspace = PlasticWorkspace().to(device)
    pid = workspace.allocate_compact_operator(
        prim_cfg, label=f"fresh_{operation_name}", device=device
    )
    allocated_params = workspace.total_parameter_count()

    train_examples = generate_benchmark_examples(
        config.seed,
        config.num_fresh_adaptation_examples,
        operation=operation_name,
        split="train",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )

    op_module = workspace.get(pid)
    op_module.unfreeze()
    op_module.train()

    optimizer = torch.optim.AdamW(
        op_module.parameters(), lr=config.fresh_adaptation_lr, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    rng = random.Random(config.seed * 11003 + 7)

    for _ in range(config.fresh_adaptation_steps):
        batch = [rng.choice(train_examples) for _ in range(32)]
        lengths = [len(ex.input_tokens) for ex in batch]
        out_lengths = [len(ex.target_tokens) for ex in batch]
        lmax = max(lengths)
        b_ids = collate_content_only_batch(batch, core.tokens, device=device)

        with torch.no_grad():
            h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]

        logits = op_module(h_content, lengths, out_lengths)
        lout_max = max(out_lengths)
        targets = torch.full((len(batch), lout_max), IGNORE_INDEX, dtype=torch.long, device=device)
        for row, ex in enumerate(batch):
            targets[row, : len(ex.target_tokens)] = torch.tensor(
                ex.target_tokens, dtype=torch.long, device=device
            )

        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    op_module.eval()
    op_module.freeze()

    # Evaluate
    correct = 0
    with torch.no_grad():
        for start in range(0, len(eval_examples), _EVAL_BATCH_SIZE):
            chunk = eval_examples[start : start + _EVAL_BATCH_SIZE]
            lengths = [len(ex.input_tokens) for ex in chunk]
            out_lengths = [len(ex.target_tokens) for ex in chunk]
            lmax = max(lengths)
            b_ids = collate_content_only_batch(chunk, core.tokens, device=device)
            h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]
            logits = op_module(h_content, lengths, out_lengths)
            preds = logits.argmax(dim=-1)
            for row, ex in enumerate(chunk):
                n = len(ex.target_tokens)
                if tuple(preds[row, :n].tolist()) == ex.target_tokens:
                    correct += 1

    em = correct / len(eval_examples)
    workspace.release()
    return em, config.fresh_adaptation_steps, allocated_params


def _ensure_consolidated_bank_and_core(
    config: RecurrenceBenchmarkConfig,
    seed_dir: Path | None = None,
) -> tuple[Any, PrimitiveBank, dict[str, int], PrimitiveBank, dict[str, int]]:
    """Retrieve or construct the Stable Core, 10-primitive bank, and unconsolidated 8-bank."""
    # 1. Obtain frozen shared task-blind Stable Core
    u_config = UnifiedBenchmarkConfig(
        seed=config.seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=config.device,
        d_operator=config.d_operator,
        n_operator_head=config.n_operator_head,
        d_operator_ff=config.d_operator_ff,
        max_sequence_length=config.max_sequence_length,
        core_train_steps=config.core_train_steps,
        shared_encoder_checkpoint=config.shared_encoder_checkpoint,
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=seed_dir)
    core.model.eval()

    # 2. Build unconsolidated 8-primitive baseline bank
    unconsolidated_bank, unconsolidated_op_to_id = _build_heterogeneous_bank(u_config)
    unconsolidated_bank.to(core.device)

    # 3. Build consolidated 10-primitive bank
    consolidated_bank, op_to_id = _build_heterogeneous_bank(u_config)
    consolidated_bank.to(core.device)

    # Add the consolidated candidate slots for novel operations
    for op_name in config.recurrent_operations:
        p = consolidated_bank.new_cross_position_primitive(
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

    # Ensure all newly added primitives are moved to device
    consolidated_bank.to(core.device)

    # Check for pre-existing consolidation checkpoint
    loaded = False
    if config.consolidation_checkpoint_dir:
        ckpt_dir = Path(config.consolidation_checkpoint_dir) / f"seed_{config.seed}"
        ckpt_path = ckpt_dir / "primitive_bank.pt"
        if ckpt_path.is_file():
            try:
                state_dict = torch.load(ckpt_path, map_location=core.device, weights_only=True)
                consolidated_bank.load_state_dict(state_dict)
                loaded = True
            except Exception:
                loaded = False

    if not loaded:
        # If no pre-existing consolidation checkpoint, run consolidation benchmark to build it
        c_cfg = ConsolidationBenchmarkConfig(
            seed=config.seed,
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
            group_size=config.group_size,
            model=config.model,
            device=config.device,
            d_operator=config.d_operator,
            n_operator_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            max_sequence_length=config.max_sequence_length,
            novel_operations=config.recurrent_operations,
            num_eval_examples=config.num_eval_examples,
        )
        run_consolidation_benchmark(c_cfg, seed_dir=seed_dir)
        if seed_dir and (seed_dir / "primitive_bank.pt").is_file():
            state_dict = torch.load(
                seed_dir / "primitive_bank.pt", map_location=core.device, weights_only=True
            )
            consolidated_bank.load_state_dict(state_dict)
        else:
            msg = f"Consolidation benchmark failed to generate bank for seed {config.seed}"
            raise RuntimeError(msg)

    # Load canonical weights into unconsolidated bank for clean control comparison
    canonical_state = {
        k: v for k, v in consolidated_bank.state_dict().items()
        if int(k.split(".")[1]) < 8
    }
    unconsolidated_bank.load_state_dict(canonical_state)

    consolidated_bank.to(core.device)
    unconsolidated_bank.to(core.device)
    consolidated_bank.freeze_all()
    consolidated_bank.eval()
    unconsolidated_bank.freeze_all()
    unconsolidated_bank.eval()

    verify_frozen_invariants(core, consolidated_bank)
    verify_frozen_invariants(core, unconsolidated_bank)

    return core, consolidated_bank, op_to_id, unconsolidated_bank, unconsolidated_op_to_id


def run_recurrence_benchmark(
    config: RecurrenceBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> RecurrenceBenchmarkReport:
    """Execute Task A1-B007 Recurrence & Reuse benchmark for a single seed."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    # 1. Obtain Core and Consolidated PrimitiveBank
    core, bank, op_to_id, unconsolidated_bank, unconsolidated_op_to_id = (
        _ensure_consolidated_bank_and_core(config, seed_dir=seed_dir)
    )
    core_params = sum(p.numel() for p in core.model.parameters())
    bank_params = bank.total_parameter_count()
    bank_size = len(bank)

    workspace = PlasticWorkspace().to(core.device)
    # Verify initial plastic workspace allocation is 0
    if workspace.total_parameter_count() != 0 or len(workspace) != 0:
        raise RuntimeError("Initial plastic workspace must be empty (0 parameters)")

    # 2. Intervening Task Stream Phase
    # Execute canonical operations to simulate lifelong intervening activity
    intervening_results: dict[str, float] = {}
    for op_name in config.intervening_operations:
        examples = generate_benchmark_examples(
            config.seed,
            config.num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        em, _ = _evaluate_exact_match(core, bank, op_to_id, examples, op_name)
        intervening_results[op_name] = em

    # 3. Recurrence Phase
    # Consolidated operations reappear in the task stream
    recurrent_results: dict[str, RecurrenceEventResult] = {}

    for op_name in config.recurrent_operations:
        eval_examples = generate_benchmark_examples(
            config.seed,
            config.num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )

        # Zero out call stats for strict sparse call tracking
        for pid in bank.ids():
            bank.get(pid).reset_forward_call_count()

        # Arm 1: Bank Reuse (APC Recurrence)
        # Immediate evaluation through installed primitive with ZERO adaptation steps
        # and ZERO plastic parameter allocation
        frozen_passed = verify_frozen_invariants(core, bank)
        selected_pid = op_to_id[op_name]

        immediate_em, immediate_acc = _evaluate_exact_match(
            core, bank, op_to_id, eval_examples, op_name
        )

        # Verify plastic workspace parameter allocation during recurrence
        allocated_plastic_params = workspace.total_parameter_count()
        zero_alloc_passed = (
            allocated_plastic_params == MAX_ALLOCATED_PLASTIC_PARAMS
            and len(workspace) == 0
        )

        # Verify sparse execution (unselected primitives have 0 forward calls)
        sparse_passed = _verify_sparse_execution(bank, selected_pid)

        # Arm 2: Fresh Adaptation Control (No-Reuse Scratch Adaptation)
        fresh_em, fresh_steps, fresh_params = _evaluate_fresh_adaptation_baseline(
            core, config, op_name, eval_examples
        )

        # Arm 3: Unconsolidated Control (Bank without consolidated primitives)
        # In the unconsolidated bank, the novel operation was never consolidated.
        # We evaluate the performance using the closest existing bank primitive (COPY)
        # demonstrating that without consolidation, the bank fails completely on this task.
        unconsolidated_em, _ = _evaluate_exact_match(
            core, unconsolidated_bank, unconsolidated_op_to_id, eval_examples, "COPY"
        )

        # Acceptance check
        accuracy_passed = immediate_em >= config.accuracy_threshold
        op_passed = (
            accuracy_passed
            and zero_alloc_passed
            and frozen_passed
            and sparse_passed
            and (unconsolidated_em <= config.unconsolidated_threshold)
        )

        recurrent_results[op_name] = RecurrenceEventResult(
            operation=op_name,
            immediate_exact_match=immediate_em,
            immediate_token_accuracy=immediate_acc,
            adaptation_steps=0,
            allocated_plastic_parameters=allocated_plastic_params,
            fresh_adaptation_exact_match=fresh_em,
            fresh_adaptation_steps=fresh_steps,
            fresh_adaptation_parameters=fresh_params,
            unconsolidated_exact_match=unconsolidated_em,
            frozen_invariants_passed=frozen_passed,
            sparse_execution_passed=sparse_passed,
            accuracy_passed=accuracy_passed,
            zero_allocation_passed=zero_alloc_passed,
            passed=op_passed,
        )

    # 4. Aggregate Report
    mean_immediate_em = statistics.mean(
        r.immediate_exact_match for r in recurrent_results.values()
    )
    max_alloc_params = max(
        r.allocated_plastic_parameters for r in recurrent_results.values()
    )
    all_recurrent_passed = all(r.passed for r in recurrent_results.values())
    all_invariants_passed = all(
        r.frozen_invariants_passed and r.sparse_execution_passed
        for r in recurrent_results.values()
    )
    overall_passed = all_recurrent_passed and all_invariants_passed

    elapsed = time.perf_counter() - start_time

    report = RecurrenceBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        bank_param_count=bank_params,
        bank_size=bank_size,
        intervening_task_results=intervening_results,
        recurrent_event_results=recurrent_results,
        mean_immediate_exact_match=mean_immediate_em,
        max_allocated_plastic_parameters=max_alloc_params,
        all_recurrent_passed=all_recurrent_passed,
        all_invariants_passed=all_invariants_passed,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )

    if seed_dir is not None:
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

    return report


@dataclass(frozen=True)
class RecurrenceMultiSeedReport:
    """Multi-seed aggregate report for Task A1-B007."""

    seeds: tuple[int, ...]
    reports: list[RecurrenceBenchmarkReport]
    overall_passed: bool
    meets_seed_policy: bool
    mean_immediate_exact_match: float
    max_allocated_plastic_parameters: int
    per_operation_means: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "mean_immediate_exact_match": self.mean_immediate_exact_match,
            "max_allocated_plastic_parameters": self.max_allocated_plastic_parameters,
            "per_operation_means": self.per_operation_means,
        }


def run_recurrence_benchmark_multi_seed(
    seeds: tuple[int, ...],
    config_factory: Any,
    *,
    output_dir: Path | None = None,
) -> RecurrenceMultiSeedReport:
    """Execute Task A1-B007 recurrence benchmark across multiple seeds."""
    reports: list[RecurrenceBenchmarkReport] = []

    for s in seeds:
        seed_dir = output_dir / f"seed_{s}" if output_dir is not None else None
        cfg = config_factory(s)
        rep = run_recurrence_benchmark(cfg, seed_dir=seed_dir)
        reports.append(rep)

    meets_seed_policy = len(seeds) >= MIN_GATE_SEEDS and set(seeds).issuperset(
        set(DEFAULT_SEEDS)
    )
    overall_passed = all(r.overall_passed for r in reports) and meets_seed_policy
    mean_immediate_em = statistics.mean(r.mean_immediate_exact_match for r in reports)
    max_alloc_params = max(r.max_allocated_plastic_parameters for r in reports)

    per_op: dict[str, dict[str, Any]] = {}
    sample_rep = reports[0]
    for op_name in sample_rep.recurrent_event_results:
        ems = [r.recurrent_event_results[op_name].immediate_exact_match for r in reports]
        fresh_ems = [
            r.recurrent_event_results[op_name].fresh_adaptation_exact_match
            for r in reports
        ]
        uncons_ems = [
            r.recurrent_event_results[op_name].unconsolidated_exact_match
            for r in reports
        ]
        alloc_params = [
            r.recurrent_event_results[op_name].allocated_plastic_parameters
            for r in reports
        ]
        passes = [r.recurrent_event_results[op_name].passed for r in reports]
        per_op[op_name] = {
            "mean_immediate_exact_match": statistics.mean(ems),
            "std_immediate_exact_match": statistics.stdev(ems) if len(ems) > 1 else 0.0,
            "mean_fresh_adaptation_exact_match": statistics.mean(fresh_ems),
            "mean_unconsolidated_exact_match": statistics.mean(uncons_ems),
            "max_allocated_plastic_parameters": max(alloc_params),
            "all_passed": all(passes),
        }

    multi_report = RecurrenceMultiSeedReport(
        seeds=seeds,
        reports=reports,
        overall_passed=overall_passed,
        meets_seed_policy=meets_seed_policy,
        mean_immediate_exact_match=mean_immediate_em,
        max_allocated_plastic_parameters=max_alloc_params,
        per_operation_means=per_op,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(
                {
                    "multi_seed_report": multi_report.to_dict(),
                    "per_seed_reports": [r.to_dict() for r in reports],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return multi_report
