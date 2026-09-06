"""Functional Consolidation Benchmark (Task A1-B006 / Milestone B-M6, STOP GATE).

Evaluates the distillation of temporary plastic solutions into compact candidate primitives
(~18k parameters), shadow validation against historical canonical tasks (<= 2% forgetting)
and temporary accuracy (>= 95% retention), and conditional 100% release of temporary
capacity into the persistent PrimitiveBank.

Acceptance Criteria (ADR-0046 / CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md):
- Multi-seed benchmark (5 seeds: 0, 1, 2, 3, 4) on held-out test data.
- Consolidated candidate achieves >= 95% of temporary plastic accuracy.
- Zero degradation on historical tasks (<= 2% forgetting).
- Temporary plastic capacity released completely (100% released) upon successful shadow validation.
- Bank size expands by consolidated primitives (8 -> 9 -> 10).
- STOP GATE.
"""

from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from apc.consolidation.compact_consolidation import (
    CompactDistillationConfig,
    CompactShadowValidationConfig,
    distill_compact_candidate,
    run_shadow_validation,
)
from apc.core.data import IGNORE_INDEX
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
    _train_single_primitive,
)
from apc.plastic.residual import execute_plastic_residual, verify_frozen_invariants
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.composition_search import search_composition_recipe
from apc.primitives.primitive import CrossPositionPrimitiveConfig
from apc.utils.seed import set_seed
from apc.utils.seed_derivation import (
    DEFAULT_GENERATOR_VERSION,
    GENERATOR_VERSION_V1_LEGACY,
    derive_seed,
    derive_seed_v1_legacy,
    legacy_split_salt,
)

MIN_GATE_SEEDS: Final[int] = 5
RETENTION_THRESHOLD: Final[float] = 0.95
MAX_FORGETTING_THRESHOLD: Final[float] = 0.02
MAX_CANDIDATE_PARAMETERS: Final[int] = 25_000
_EVAL_BATCH_SIZE: Final[int] = 64


def generate_benchmark_examples(
    seed: int,
    n: int,
    *,
    operation: str,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
    generator_version: str = DEFAULT_GENERATOR_VERSION,
    start_index: int = 0,
) -> list[Example]:
    """Deterministically generate examples for any canonical or novel operation.

    This is the single canonical implementation; `recurrence_benchmark.py`
    imports this same function rather than defining its own copy (ADR-0080
    found the two had drifted into a literal duplicate).

    `generator_version` defaults to the SHA256-indexed v2 generator (ADR-0080
    fix, Task B-C005R3-001): each example's RNG seed is derived independently
    from ``(generator_version, seed, split, operation, start_index + i)`` via
    `apc.utils.seed_derivation.derive_seed`, so results do not depend on
    Python's per-process string-hash randomization, cell iteration order,
    resumption point, or worker partitioning. `start_index` lets a caller
    generate a sub-range (for resumption or worker partitioning) whose
    examples are identical to the corresponding slice of a full `n`-example
    call.

    Pass `generator_version="v1_legacy_hash"` only to diagnostically
    reconstruct a pre-fix run under a pinned `PYTHONHASHSEED`; that path
    reuses one continuing RNG stream (the original behavior) and is *not*
    guaranteed reproducible across processes or iteration orders. It must
    not be used as a new default.
    """
    op = get_operation(operation)
    valid_lengths = [
        length
        for length in range(sequence_length_range[0], sequence_length_range[1] + 1)
        if op.is_valid_for_length(length)
    ]
    if not valid_lengths:
        raise ValueError(
            f"No valid lengths for '{operation}' in range {sequence_length_range}"
        )

    legacy_rng: random.Random | None = None
    if generator_version == GENERATOR_VERSION_V1_LEGACY:
        salt = legacy_split_salt(split)
        legacy_rng = random.Random(
            derive_seed_v1_legacy(master_seed=seed, salt=salt, task_key=operation)
        )

    examples: list[Example] = []
    for i in range(n):
        if legacy_rng is not None:
            example_rng = legacy_rng
        else:
            example_seed = derive_seed(
                master_seed=seed,
                stream_namespace=split,
                task_key=operation,
                sample_index=start_index + i,
                generator_version=generator_version,
            )
            example_rng = random.Random(example_seed)
        seq_len = example_rng.choice(valid_lengths)
        seq = tuple(example_rng.randrange(vocab_size) for _ in range(seq_len))
        params = op.sample_params(example_rng, seq, vocab_size)
        step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(step,))
        res = run_program(prog, seq, vocab_size)
        category = "known" if operation in ALL_CANONICAL_OPERATIONS else "novel"
        meta_label: Any = "K" if category == "known" else "N"
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category=category,
            split=split,
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(label=meta_label, primitive_operations=(operation,)),
        )
        examples.append(ex)
    return examples


@dataclass(frozen=True)
class ConsolidationBenchmarkConfig:
    """Explicit configuration for Task A1-B006 benchmark."""

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

    novel_operations: tuple[str, ...] = BRANCH_B_NOVEL_OPERATION_NAMES
    num_adaptation_examples: int = 32
    num_plastic_train_examples: int = 1000
    num_distill_train_examples: int = 1000
    num_eval_examples: int = 200
    min_eval_examples: int = 200
    num_historical_eval_examples: int = 100

    plastic_train_steps: int = 1500
    distill_train_steps: int = 4000
    core_train_steps: int = 6000
    bank_train_steps: int = 6000
    plastic_lr: float = 0.001
    distill_lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    distillation_alpha: float = 0.25
    distillation_temperature: float = 2.0

    retention_threshold: float = RETENTION_THRESHOLD
    max_forgetting_threshold: float = MAX_FORGETTING_THRESHOLD
    max_candidate_params: int = MAX_CANDIDATE_PARAMETERS
    shared_encoder_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.num_eval_examples < self.min_eval_examples:
            raise ValueError(f"num_eval_examples must be >= {self.min_eval_examples}")
        if self.num_adaptation_examples < 8:
            raise ValueError("num_adaptation_examples must be >= 8")
        if not self.novel_operations:
            raise ValueError("novel_operations must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def consolidation_config_from_dict(raw: dict[str, Any]) -> ConsolidationBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = ConsolidationBenchmarkConfig()
    return ConsolidationBenchmarkConfig(
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
        novel_operations=tuple(raw.get("novel_operations", defaults.novel_operations)),
        num_adaptation_examples=raw.get(
            "num_adaptation_examples", defaults.num_adaptation_examples
        ),
        num_plastic_train_examples=raw.get(
            "num_plastic_train_examples", defaults.num_plastic_train_examples
        ),
        num_distill_train_examples=raw.get(
            "num_distill_train_examples", defaults.num_distill_train_examples
        ),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        min_eval_examples=raw.get("min_eval_examples", defaults.min_eval_examples),
        num_historical_eval_examples=raw.get(
            "num_historical_eval_examples", defaults.num_historical_eval_examples
        ),
        plastic_train_steps=raw.get("plastic_train_steps", defaults.plastic_train_steps),
        distill_train_steps=raw.get("distill_train_steps", defaults.distill_train_steps),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        plastic_lr=raw.get("plastic_lr", defaults.plastic_lr),
        distill_lr=raw.get("distill_lr", defaults.distill_lr),
        weight_decay=raw.get("weight_decay", defaults.weight_decay),
        batch_size=raw.get("batch_size", defaults.batch_size),
        distillation_alpha=raw.get("distillation_alpha", defaults.distillation_alpha),
        distillation_temperature=raw.get(
            "distillation_temperature", defaults.distillation_temperature
        ),
        retention_threshold=raw.get("retention_threshold", defaults.retention_threshold),
        max_forgetting_threshold=raw.get(
            "max_forgetting_threshold", defaults.max_forgetting_threshold
        ),
        max_candidate_params=raw.get("max_candidate_params", defaults.max_candidate_params),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
    )


@dataclass(frozen=True)
class ConsolidationOperationResult:
    """Evaluation outcome of consolidating one novel operation."""

    operation: str
    base_candidate_operations: tuple[str, ...]
    frozen_bank_exact_match: float
    temporary_exact_match: float
    temporary_token_accuracy: float
    candidate_exact_match: float
    candidate_token_accuracy: float
    bank_installed_exact_match: float
    retention_ratio: float
    shadow_agreement_rate: float
    historical_baseline_mean_em: float
    historical_after_mean_em: float
    historical_forgetting: float
    candidate_parameter_count: int
    temporary_parameters_before_release: int
    temporary_parameters_after_release: int
    workspace_released_completely: bool
    frozen_invariants_passed: bool
    retention_passed: bool
    forgetting_passed: bool
    capacity_passed: bool
    passed: bool


@dataclass(frozen=True)
class ConsolidationBenchmarkReport:
    """Single-seed benchmark report for Task A1-B006."""

    config: ConsolidationBenchmarkConfig
    seed: int
    core_param_count: int
    initial_bank_param_count: int
    final_bank_param_count: int
    initial_bank_size: int
    final_bank_size: int
    results_by_operation: dict[str, ConsolidationOperationResult]
    mean_candidate_exact_match: float
    mean_temporary_exact_match: float
    mean_retention_ratio: float
    max_historical_forgetting: float
    all_retention_passed: bool
    all_forgetting_passed: bool
    all_released_passed: bool
    all_frozen_passed: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "initial_bank_param_count": self.initial_bank_param_count,
            "final_bank_param_count": self.final_bank_param_count,
            "initial_bank_size": self.initial_bank_size,
            "final_bank_size": self.final_bank_size,
            "mean_candidate_exact_match": self.mean_candidate_exact_match,
            "mean_temporary_exact_match": self.mean_temporary_exact_match,
            "mean_retention_ratio": self.mean_retention_ratio,
            "max_historical_forgetting": self.max_historical_forgetting,
            "all_retention_passed": self.all_retention_passed,
            "all_forgetting_passed": self.all_forgetting_passed,
            "all_released_passed": self.all_released_passed,
            "all_frozen_passed": self.all_frozen_passed,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "results_by_operation": {
                name: dataclasses.asdict(res)
                for name, res in self.results_by_operation.items()
            },
        }


def _train_plastic_residual_operator(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    workspace: PlasticWorkspace,
    plastic_id: int,
    train_examples: Sequence[Example],
    *,
    base_candidate_operations: Sequence[str] | None,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    vocab_size: int,
    seed: int = 42,
) -> float:
    """Train temporary plastic operator as a residual over frozen core and bank."""
    plastic_op = workspace.get(plastic_id)
    plastic_op.unfreeze()
    plastic_op.train()

    optimizer = torch.optim.AdamW(plastic_op.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    device = core.device

    final_loss = 0.0
    rng = random.Random(seed)

    for _step in range(1, steps + 1):
        batch = [rng.choice(train_examples) for _ in range(batch_size)]
        logits = execute_plastic_residual(
            core,
            bank,
            op_to_id,
            workspace,
            plastic_id,
            batch,
            base_candidate_operations=base_candidate_operations,
            target_vocab_size=vocab_size,
        )

        output_lengths = [len(ex.target_tokens) for ex in batch]
        max_out = max(output_lengths)
        targets = torch.full((batch_size, max_out), IGNORE_INDEX, dtype=torch.long, device=device)
        for i, ex in enumerate(batch):
            targets[i, : len(ex.target_tokens)] = torch.tensor(
                ex.target_tokens, dtype=torch.long, device=device
            )

        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()

    plastic_op.eval()
    plastic_op.freeze()
    return final_loss


def _evaluate_historical_canonical_tasks(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    historical_examples: Mapping[str, Sequence[Example]],
) -> dict[str, float]:
    """Evaluate exact match on historical canonical operations."""
    results: dict[str, float] = {}
    with torch.no_grad():
        for op, examples in historical_examples.items():
            if not examples:
                continue
            logits = execute_composition_recipe(
                core, bank, op_to_id, examples, candidate_operations=(op,)
            )
            preds = logits.argmax(dim=-1)
            corr = 0
            for row, ex in enumerate(examples):
                n = len(ex.target_tokens)
                if tuple(preds[row, :n].tolist()) == ex.target_tokens:
                    corr += 1
            results[op] = corr / len(examples)
    return results


def run_consolidation_benchmark(
    config: ConsolidationBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> ConsolidationBenchmarkReport:
    """Execute Task A1-B006 benchmark for a single seed."""
    start_time = time.perf_counter()
    set_seed(config.seed)

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
    core_params = sum(p.numel() for p in core.model.parameters())

    # 2. Build and freeze initial PrimitiveBank
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(core.device)

    # Load pre-trained canonical 8-primitive bank checkpoint if available
    # Note: seed_dir / "primitive_bank.pt" is the OUTPUT of this consolidation run,
    # so it must never be loaded as the initial canonical bank.
    loaded_bank = False
    candidate_ckpts = []
    plastic_bench_dir = Path("runs/phase_a1_plastic_workspace_benchmark")
    comp_bench_dir = Path("runs/phase_a1_composition_library_benchmark")
    candidate_ckpts.append(plastic_bench_dir / f"seed_{config.seed}" / "primitive_bank.pt")
    candidate_ckpts.append(comp_bench_dir / f"seed_{config.seed}" / "primitive_bank.pt")

    for bank_ckpt in candidate_ckpts:
        if bank_ckpt.is_file():
            try:
                state_dict = torch.load(bank_ckpt, map_location=core.device, weights_only=True)
                bank.load_state_dict(state_dict)
                loaded_bank = True
                break
            except Exception:
                loaded_bank = False

    if not loaded_bank:
        for op in ALL_CANONICAL_OPERATIONS:
            p = bank.get(op_to_id[op])
            _train_single_primitive(core, p, u_config, op, steps=config.bank_train_steps)

    bank.freeze_all()
    bank.eval()
    initial_bank_params = bank.total_parameter_count()
    initial_bank_size = len(bank)
    verify_frozen_invariants(core, bank)

    # 3. Pre-generate historical evaluation datasets across canonical tasks
    historical_eval_examples: dict[str, list[Example]] = {}
    for h_op in ALL_CANONICAL_OPERATIONS:
        historical_eval_examples[h_op] = generate_benchmark_examples(
            config.seed,
            config.num_historical_eval_examples,
            operation=h_op,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )

    # Compute baseline historical performance before consolidation
    hist_baseline_before = _evaluate_historical_canonical_tasks(
        core, bank, op_to_id, historical_eval_examples
    )
    mean_hist_baseline = statistics.mean(hist_baseline_before.values())

    # 4. Consolidate each novel operation sequentially
    results_by_operation: dict[str, ConsolidationOperationResult] = {}
    workspace = PlasticWorkspace().to(core.device)

    for op_idx, op_name in enumerate(config.novel_operations):
        adapt_examples = generate_benchmark_examples(
            config.seed,
            config.num_adaptation_examples,
            operation=op_name,
            split="adapt",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        plastic_train_examples = generate_benchmark_examples(
            config.seed,
            config.num_plastic_train_examples,
            operation=op_name,
            split="train",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        distill_train_examples = plastic_train_examples
        eval_examples = generate_benchmark_examples(
            config.seed,
            config.num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )

        # Step 4a: Best bank recipe search
        search_res = search_composition_recipe(
            core, bank, op_to_id, adapt_examples, max_depth=2
        )
        base_ops = search_res.candidate_operations

        # Frozen bank baseline on novel op
        with torch.no_grad():
            logits_bank = execute_composition_recipe(
                core, bank, op_to_id, eval_examples, candidate_operations=base_ops
            )
            preds_bank = logits_bank.argmax(dim=-1)
            corr_bank = 0
            for row, ex in enumerate(eval_examples):
                n = len(ex.target_tokens)
                if tuple(preds_bank[row, :n].tolist()) == ex.target_tokens:
                    corr_bank += 1
            frozen_bank_em = corr_bank / len(eval_examples)

        # Step 4b: Allocate temporary plastic capacity and train residual
        set_seed(config.seed * 1000 + op_idx * 100 + 7)
        prim_cfg = CrossPositionPrimitiveConfig(
            operation=op_name,
            d_model=core.model.config.d_model,
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
        )
        pid_residual = workspace.allocate_compact_operator(
            prim_cfg, label=f"temp_residual_{op_name}", device=core.device
        )
        temp_params_before = workspace.total_parameter_count()
        verify_frozen_invariants(core, bank)

        _train_plastic_residual_operator(
            core,
            bank,
            op_to_id,
            workspace,
            pid_residual,
            plastic_train_examples,
            base_candidate_operations=base_ops,
            steps=config.plastic_train_steps,
            lr=config.plastic_lr,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
            seed=config.seed * 1000 + op_idx * 100 + 13,
        )

        # Step 4c: Distill temporary solution into compact candidate primitive
        distill_cfg = CompactDistillationConfig(
            steps=config.distill_train_steps,
            lr=config.distill_lr,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            temperature=config.distillation_temperature,
            distillation_alpha=config.distillation_alpha,
            d_operator=config.d_operator,
            n_operator_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            max_sequence_length=config.max_sequence_length,
            seed=config.seed * 1000 + op_idx * 100 + 17,
        )
        candidate, _distill_report = distill_compact_candidate(
            core,
            bank,
            op_to_id,
            workspace,
            pid_residual,
            distill_train_examples,
            op_name,
            distill_cfg,
            base_candidate_operations=base_ops,
        )

        # Step 4d: Shadow Validation & Conditional Release
        shadow_cfg = CompactShadowValidationConfig(
            retention_threshold=config.retention_threshold,
            max_forgetting_threshold=config.max_forgetting_threshold,
            max_candidate_parameters=config.max_candidate_params,
            eval_batch_size=_EVAL_BATCH_SIZE,
        )
        installed_id, shadow_report = run_shadow_validation(
            core,
            bank,
            op_to_id,
            workspace,
            pid_residual,
            candidate,
            op_name,
            eval_examples,
            historical_eval_examples,
            shadow_cfg,
            base_candidate_operations=base_ops,
        )

        temp_params_after = workspace.total_parameter_count()
        workspace_cleared = (
            installed_id is not None and temp_params_after == 0 and len(workspace) == 0
        )

        # Step 4e: Verify direct execution through bank after installation
        bank_installed_em = 0.0
        if installed_id is not None:
            with torch.no_grad():
                logits_installed = execute_composition_recipe(
                    core, bank, op_to_id, eval_examples, candidate_operations=(op_name,)
                )
                preds_installed = logits_installed.argmax(dim=-1)
                corr_inst = 0
                for row, ex in enumerate(eval_examples):
                    n = len(ex.target_tokens)
                    if tuple(preds_installed[row, :n].tolist()) == ex.target_tokens:
                        corr_inst += 1
                bank_installed_em = corr_inst / len(eval_examples)

        # Check empirical post-consolidation historical performance
        hist_after = _evaluate_historical_canonical_tasks(
            core, bank, op_to_id, historical_eval_examples
        )
        mean_hist_after = statistics.mean(hist_after.values())
        empirical_forgetting = max(
            0.0,
            max((hist_baseline_before[op] - hist_after[op]) for op in ALL_CANONICAL_OPERATIONS),
        )

        target_em = config.retention_threshold * shadow_report.temporary_exact_match
        frozen_passed = verify_frozen_invariants(core, bank)
        retention_passed = shadow_report.retention_passed and (bank_installed_em >= target_em)
        forgetting_passed = empirical_forgetting <= config.max_forgetting_threshold
        capacity_passed = shadow_report.size_passed
        op_passed = (
            frozen_passed
            and retention_passed
            and forgetting_passed
            and capacity_passed
            and workspace_cleared
        )

        results_by_operation[op_name] = ConsolidationOperationResult(
            operation=op_name,
            base_candidate_operations=base_ops,
            frozen_bank_exact_match=frozen_bank_em,
            temporary_exact_match=shadow_report.temporary_exact_match,
            temporary_token_accuracy=shadow_report.temporary_token_accuracy,
            candidate_exact_match=shadow_report.candidate_exact_match,
            candidate_token_accuracy=shadow_report.candidate_token_accuracy,
            bank_installed_exact_match=bank_installed_em,
            retention_ratio=shadow_report.retention_ratio,
            shadow_agreement_rate=shadow_report.shadow_agreement_rate,
            historical_baseline_mean_em=mean_hist_baseline,
            historical_after_mean_em=mean_hist_after,
            historical_forgetting=empirical_forgetting,
            candidate_parameter_count=candidate.num_parameters(),
            temporary_parameters_before_release=temp_params_before,
            temporary_parameters_after_release=temp_params_after,
            workspace_released_completely=workspace_cleared,
            frozen_invariants_passed=frozen_passed,
            retention_passed=retention_passed,
            forgetting_passed=forgetting_passed,
            capacity_passed=capacity_passed,
            passed=op_passed,
        )

    final_bank_params = bank.total_parameter_count()
    final_bank_size = len(bank)

    if seed_dir is not None:
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bank.state_dict(), seed_dir / "primitive_bank.pt")

    # 5. Aggregate report metrics
    mean_cand_em = statistics.mean(
        res.bank_installed_exact_match for res in results_by_operation.values()
    )
    mean_temp_em = statistics.mean(
        res.temporary_exact_match for res in results_by_operation.values()
    )
    mean_ret = statistics.mean(res.retention_ratio for res in results_by_operation.values())
    max_forg = max(res.historical_forgetting for res in results_by_operation.values())

    all_ret = all(res.retention_passed for res in results_by_operation.values())
    all_forg = all(res.forgetting_passed for res in results_by_operation.values())
    all_rel = all(res.workspace_released_completely for res in results_by_operation.values())
    all_froz = all(res.frozen_invariants_passed for res in results_by_operation.values())
    overall_passed = all(res.passed for res in results_by_operation.values())

    elapsed = time.perf_counter() - start_time

    return ConsolidationBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        initial_bank_param_count=initial_bank_params,
        final_bank_param_count=final_bank_params,
        initial_bank_size=initial_bank_size,
        final_bank_size=final_bank_size,
        results_by_operation=results_by_operation,
        mean_candidate_exact_match=mean_cand_em,
        mean_temporary_exact_match=mean_temp_em,
        mean_retention_ratio=mean_ret,
        max_historical_forgetting=max_forg,
        all_retention_passed=all_ret,
        all_forgetting_passed=all_forg,
        all_released_passed=all_rel,
        all_frozen_passed=all_froz,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ConsolidationMultiSeedReport:
    """Multi-seed aggregate report for Task A1-B006."""

    seeds: tuple[int, ...]
    reports: list[ConsolidationBenchmarkReport]
    overall_passed: bool
    meets_seed_policy: bool
    mean_overall_candidate_exact_match: float
    mean_overall_temporary_exact_match: float
    mean_overall_retention_ratio: float
    max_overall_historical_forgetting: float
    per_operation_means: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "mean_overall_candidate_exact_match": self.mean_overall_candidate_exact_match,
            "mean_overall_temporary_exact_match": self.mean_overall_temporary_exact_match,
            "mean_overall_retention_ratio": self.mean_overall_retention_ratio,
            "max_overall_historical_forgetting": self.max_overall_historical_forgetting,
            "per_operation_means": self.per_operation_means,
            "reports": [r.to_dict() for r in self.reports],
        }


def run_consolidation_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Any,
    output_dir: Path | None = None,
) -> ConsolidationMultiSeedReport:
    """Run multi-seed functional consolidation and shadow validation benchmark."""
    seed_tuple = tuple(seeds)
    meets_policy = len(seed_tuple) >= MIN_GATE_SEEDS

    reports: list[ConsolidationBenchmarkReport] = []
    for s in seed_tuple:
        s_dir = output_dir / f"seed_{s}" if output_dir is not None else None
        cfg = config_factory(s)
        rep = run_consolidation_benchmark(cfg, seed_dir=s_dir)
        reports.append(rep)
        if s_dir is not None:
            s_dir.mkdir(parents=True, exist_ok=True)
            (s_dir / "report.json").write_text(
                json.dumps(rep.to_dict(), indent=2), encoding="utf-8"
            )
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_passed = meets_policy and all(r.overall_passed for r in reports)
    mean_cand_em = statistics.mean(r.mean_candidate_exact_match for r in reports)
    mean_temp_em = statistics.mean(r.mean_temporary_exact_match for r in reports)
    mean_ret = statistics.mean(r.mean_retention_ratio for r in reports)
    max_forg = max(r.max_historical_forgetting for r in reports)

    per_op_means: dict[str, dict[str, Any]] = {}
    sample_rep = reports[0]
    for op_name in sample_rep.results_by_operation:
        cand_ems = [r.results_by_operation[op_name].bank_installed_exact_match for r in reports]
        cand_accs = [r.results_by_operation[op_name].candidate_token_accuracy for r in reports]
        temp_ems = [r.results_by_operation[op_name].temporary_exact_match for r in reports]
        rets = [r.results_by_operation[op_name].retention_ratio for r in reports]
        agrees = [r.results_by_operation[op_name].shadow_agreement_rate for r in reports]
        forgs = [r.results_by_operation[op_name].historical_forgetting for r in reports]
        cand_params = [r.results_by_operation[op_name].candidate_parameter_count for r in reports]

        per_op_means[op_name] = {
            "mean_candidate_exact_match": statistics.mean(cand_ems),
            "stdev_candidate_exact_match": statistics.stdev(cand_ems) if len(cand_ems) > 1 else 0.0,
            "min_candidate_exact_match": min(cand_ems),
            "max_candidate_exact_match": max(cand_ems),
            "mean_candidate_token_accuracy": statistics.mean(cand_accs),
            "mean_temporary_exact_match": statistics.mean(temp_ems),
            "mean_retention_ratio": statistics.mean(rets),
            "mean_shadow_agreement_rate": statistics.mean(agrees),
            "max_historical_forgetting": max(forgs),
            "candidate_parameter_count": cand_params[0],
            "workspace_released_completely": all(
                r.results_by_operation[op_name].workspace_released_completely for r in reports
            ),
            "frozen_invariants_passed": all(
                r.results_by_operation[op_name].frozen_invariants_passed for r in reports
            ),
            "retention_passed": all(
                r.results_by_operation[op_name].retention_passed for r in reports
            ),
            "forgetting_passed": all(
                r.results_by_operation[op_name].forgetting_passed for r in reports
            ),
            "capacity_passed": all(
                r.results_by_operation[op_name].capacity_passed for r in reports
            ),
            "passed": all(r.results_by_operation[op_name].passed for r in reports),
        }

    return ConsolidationMultiSeedReport(
        seeds=seed_tuple,
        reports=reports,
        overall_passed=overall_passed,
        meets_seed_policy=meets_policy,
        mean_overall_candidate_exact_match=mean_cand_em,
        mean_overall_temporary_exact_match=mean_temp_em,
        mean_overall_retention_ratio=mean_ret,
        max_overall_historical_forgetting=max_forg,
        per_operation_means=per_op_means,
    )
