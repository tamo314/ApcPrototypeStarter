"""Plastic Workspace Residual Learning Benchmark (Task A1-B005 / Milestone B-M5, STOP GATE).

Evaluates rapid residual adaptation for novel operations (unsolvable by existing bank
primitives or compositions) over a single frozen shared task-blind Stable Core
($h_{content} = f(content)$) and a frozen compact PrimitiveBank.

Acceptance Criteria (ADR-0046 / CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md):
- Multi-seed benchmark (5 seeds: 0, 1, 2, 3, 4) on held-out test data.
- Stable Core and persistent primitives remain 100% frozen (`requires_grad == False`).
- Temporary capacity achieves novel task accuracy (exact match) >= 0.90.
- Temporary capacity <= 100k parameters.
- Memory/compute accounting strictly isolates temporary parameters.
- Controls evaluated:
  1. Frozen Bank Control: Existing bank primitives/recipes fail on novel operations (EM < 0.20).
  2. Full-Task Plastic Control: Plastic workspace trained from scratch without base recipe.
  3. Residual Plastic Learning: Plastic workspace trained as residual on top of best bank candidate.
- STOP GATE.
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
from apc.plastic.residual import (
    execute_plastic_residual,
    verify_frozen_invariants,
)
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.composition_search import search_composition_recipe
from apc.primitives.primitive import CrossPositionPrimitiveConfig
from apc.utils.seed import set_seed

MIN_GATE_SEEDS: Final[int] = 5
NOVEL_TASK_ACCURACY_THRESHOLD: Final[float] = 0.90
MAX_TEMPORARY_PARAMETERS: Final[int] = 100_000
_EVAL_BATCH_SIZE: Final[int] = 64


def generate_novel_examples(
    seed: int,
    n: int,
    *,
    operation: str,
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
) -> list[Example]:
    """Deterministically generate examples for a novel operation."""
    op = get_operation(operation)
    salt = 1000 if split == "test" else (500 if split == "adapt" else 0)
    rng = random.Random(seed * 7919 + salt)

    examples: list[Example] = []
    for _ in range(n):
        seq_len = rng.randint(sequence_length_range[0], sequence_length_range[1])
        seq = tuple(rng.randrange(vocab_size) for _ in range(seq_len))
        params = op.sample_params(rng, seq, vocab_size)
        step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(step,))
        res = run_program(prog, seq, vocab_size)
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="novel",
            split=split,
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(label="N", primitive_operations=(operation,)),
        )
        examples.append(ex)
    return examples


@dataclass(frozen=True)
class PlasticWorkspaceBenchmarkConfig:
    """Explicit configuration for Task A1-B005 benchmark."""

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
    num_train_examples: int = 1000
    num_eval_examples: int = 200
    min_eval_examples: int = 200

    plastic_train_steps: int = 1500
    plastic_lr: float = 0.001
    plastic_weight_decay: float = 0.0001
    batch_size: int = 32

    accuracy_threshold: float = NOVEL_TASK_ACCURACY_THRESHOLD
    max_temporary_params: int = MAX_TEMPORARY_PARAMETERS
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


def plastic_workspace_config_from_dict(raw: dict[str, Any]) -> PlasticWorkspaceBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = PlasticWorkspaceBenchmarkConfig()
    return PlasticWorkspaceBenchmarkConfig(
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
        num_train_examples=raw.get("num_train_examples", defaults.num_train_examples),
        num_eval_examples=raw.get("num_eval_examples", defaults.num_eval_examples),
        min_eval_examples=raw.get("min_eval_examples", defaults.min_eval_examples),
        plastic_train_steps=raw.get("plastic_train_steps", defaults.plastic_train_steps),
        plastic_lr=raw.get("plastic_lr", defaults.plastic_lr),
        plastic_weight_decay=raw.get("plastic_weight_decay", defaults.plastic_weight_decay),
        batch_size=raw.get("batch_size", defaults.batch_size),
        accuracy_threshold=raw.get("accuracy_threshold", defaults.accuracy_threshold),
        max_temporary_params=raw.get("max_temporary_params", defaults.max_temporary_params),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
    )


@dataclass(frozen=True)
class NovelOperationResult:
    """Evaluation result for one novel operation under plastic adaptation."""

    operation: str
    base_candidate_operations: tuple[str, ...]
    frozen_bank_exact_match: float
    frozen_bank_token_accuracy: float
    scratch_exact_match: float
    scratch_token_accuracy: float
    scratch_final_loss: float
    residual_exact_match: float
    residual_token_accuracy: float
    residual_final_loss: float
    temporary_parameter_count: int
    frozen_invariants_passed: bool
    accuracy_passed: bool
    capacity_passed: bool
    isolation_passed: bool
    passed: bool


@dataclass(frozen=True)
class PlasticWorkspaceBenchmarkReport:
    """Report for a single seed of Task A1-B005."""

    config: PlasticWorkspaceBenchmarkConfig
    seed: int
    core_param_count: int
    bank_param_count: int
    results_by_operation: dict[str, NovelOperationResult]
    mean_residual_exact_match: float
    mean_scratch_exact_match: float
    mean_frozen_bank_exact_match: float
    all_frozen_passed: bool
    all_accuracy_passed: bool
    all_capacity_passed: bool
    all_isolation_passed: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "bank_param_count": self.bank_param_count,
            "mean_residual_exact_match": self.mean_residual_exact_match,
            "mean_scratch_exact_match": self.mean_scratch_exact_match,
            "mean_frozen_bank_exact_match": self.mean_frozen_bank_exact_match,
            "all_frozen_passed": self.all_frozen_passed,
            "all_accuracy_passed": self.all_accuracy_passed,
            "all_capacity_passed": self.all_capacity_passed,
            "all_isolation_passed": self.all_isolation_passed,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "results_by_operation": {
                name: dataclasses.asdict(res)
                for name, res in self.results_by_operation.items()
            },
        }


def _train_plastic_operator(
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
) -> float:
    """Train the plastic operator inside workspace while keeping core and bank frozen."""
    plastic_op = workspace.get(plastic_id)
    plastic_op.unfreeze()
    plastic_op.train()

    optimizer = torch.optim.AdamW(plastic_op.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    device = core.device

    final_loss = 0.0
    rng = random.Random(42)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    plastic_op.eval()
    plastic_op.freeze()
    return final_loss


def _evaluate_plastic_model(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    workspace: PlasticWorkspace,
    plastic_id: int,
    eval_examples: Sequence[Example],
    *,
    base_candidate_operations: Sequence[str] | None,
    vocab_size: int,
) -> tuple[float, float]:
    """Evaluate plastic model on held-out evaluation examples."""
    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for start in range(0, len(eval_examples), _EVAL_BATCH_SIZE):
            chunk = eval_examples[start : start + _EVAL_BATCH_SIZE]
            logits = execute_plastic_residual(
                core,
                bank,
                op_to_id,
                workspace,
                plastic_id,
                chunk,
                base_candidate_operations=base_candidate_operations,
                target_vocab_size=vocab_size,
            )
            predictions = logits.argmax(dim=-1)

            for row, ex in enumerate(chunk):
                n = len(ex.target_tokens)
                pred = tuple(predictions[row, :n].tolist())
                target = ex.target_tokens
                total_tokens += n
                correct_tokens += sum(1 for p, t in zip(pred, target, strict=True) if p == t)
                if pred == target:
                    exact_matches += 1

    return exact_matches / len(eval_examples), correct_tokens / max(1, total_tokens)


def run_plastic_workspace_benchmark(
    config: PlasticWorkspaceBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> PlasticWorkspaceBenchmarkReport:
    """Execute Task A1-B005 benchmark for a single seed."""
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
        shared_encoder_checkpoint=config.shared_encoder_checkpoint,
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=seed_dir)
    core_params = sum(p.numel() for p in core.model.parameters())

    # 2. Build and freeze PrimitiveBank
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(core.device)

    # Load pre-trained bank checkpoint if available
    loaded_bank = False
    candidate_ckpts = []
    if seed_dir is not None:
        candidate_ckpts.append(seed_dir / "primitive_bank.pt")
    comp_bench_dir = Path("runs/phase_a1_composition_library_benchmark")
    oracle_bench_dir = Path("runs/phase_a1_unified_oracle_causal_benchmark")
    candidate_ckpts.append(comp_bench_dir / f"seed_{config.seed}" / "primitive_bank.pt")
    candidate_ckpts.append(oracle_bench_dir / f"seed_{config.seed}" / "primitive_bank.pt")

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
            _train_single_primitive(core, p, u_config, op, steps=6000)

    bank.freeze_all()
    bank.eval()
    bank_params = bank.total_parameter_count()

    if seed_dir is not None and not (seed_dir / "primitive_bank.pt").is_file():
        torch.save(bank.state_dict(), seed_dir / "primitive_bank.pt")

    # Verify initial freeze invariants
    verify_frozen_invariants(core, bank)

    # 3. Evaluate each novel operation
    results_by_operation: dict[str, NovelOperationResult] = {}
    workspace = PlasticWorkspace().to(core.device)

    for op_name in config.novel_operations:
        adapt_examples = generate_novel_examples(
            config.seed,
            config.num_adaptation_examples,
            operation=op_name,
            split="adapt",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        train_examples = generate_novel_examples(
            config.seed,
            config.num_train_examples,
            operation=op_name,
            split="train",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        eval_examples = generate_novel_examples(
            config.seed,
            config.num_eval_examples,
            operation=op_name,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )

        # --- Control 1: Frozen Bank Baseline ---
        # Search the bank for the best candidate recipe
        search_res = search_composition_recipe(
            core, bank, op_to_id, adapt_examples, max_depth=2
        )
        base_ops = search_res.candidate_operations

        # Evaluate best bank candidate on test set
        with torch.no_grad():
            logits_bank = execute_composition_recipe(
                core, bank, op_to_id, eval_examples, candidate_operations=base_ops
            )
            preds_bank = logits_bank.argmax(dim=-1)
            corr_bank = 0
            tok_corr_bank = 0
            total_tok_bank = 0
            for row, ex in enumerate(eval_examples):
                n = len(ex.target_tokens)
                pred_tokens = tuple(preds_bank[row, :n].tolist())
                total_tok_bank += n
                tok_corr_bank += sum(
                    1 for a, b in zip(pred_tokens, ex.target_tokens, strict=True) if a == b
                )
                if pred_tokens == ex.target_tokens:
                    corr_bank += 1
            frozen_bank_em = corr_bank / len(eval_examples)
            frozen_bank_acc = tok_corr_bank / max(1, total_tok_bank)

        # --- Control 2: Full-Task Plastic Learning (from scratch, F_existing = 0) ---
        prim_cfg = CrossPositionPrimitiveConfig(
            operation=op_name,
            d_model=core.model.config.d_model,
            d_operator=config.d_operator,
            n_head=config.n_operator_head,
            d_operator_ff=config.d_operator_ff,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
        )
        pid_scratch = workspace.allocate_compact_operator(
            prim_cfg, label=f"scratch_{op_name}", device=core.device
        )
        verify_frozen_invariants(core, bank)

        scratch_loss = _train_plastic_operator(
            core,
            bank,
            op_to_id,
            workspace,
            pid_scratch,
            train_examples,
            base_candidate_operations=None,
            steps=config.plastic_train_steps,
            lr=config.plastic_lr,
            weight_decay=config.plastic_weight_decay,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
        )
        scratch_em, scratch_acc = _evaluate_plastic_model(
            core,
            bank,
            op_to_id,
            workspace,
            pid_scratch,
            eval_examples,
            base_candidate_operations=None,
            vocab_size=config.vocab_size,
        )
        verify_frozen_invariants(core, bank)
        workspace.release()

        # --- Condition 3: Residual Plastic Learning (F_existing + R_plastic) ---
        pid_residual = workspace.allocate_compact_operator(
            prim_cfg, label=f"residual_{op_name}", device=core.device
        )
        temp_params = workspace.total_parameter_count()
        verify_frozen_invariants(core, bank)

        residual_loss = _train_plastic_operator(
            core,
            bank,
            op_to_id,
            workspace,
            pid_residual,
            train_examples,
            base_candidate_operations=base_ops,
            steps=config.plastic_train_steps,
            lr=config.plastic_lr,
            weight_decay=config.plastic_weight_decay,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
        )
        residual_em, residual_acc = _evaluate_plastic_model(
            core,
            bank,
            op_to_id,
            workspace,
            pid_residual,
            eval_examples,
            base_candidate_operations=base_ops,
            vocab_size=config.vocab_size,
        )

        # Invariant checks
        frozen_passed = verify_frozen_invariants(core, bank)
        accuracy_passed = residual_em >= config.accuracy_threshold
        capacity_passed = temp_params <= config.max_temporary_params

        # Release temporary capacity and verify zero leakage
        released = workspace.release()
        isolation_passed = (
            len(released) == 1
            and workspace.total_parameter_count() == 0
            and bank.total_parameter_count() == bank_params
            and len(bank) == len(ALL_CANONICAL_OPERATIONS)
        )

        op_passed = (
            frozen_passed
            and accuracy_passed
            and capacity_passed
            and isolation_passed
            and (frozen_bank_em < 0.20)
        )

        results_by_operation[op_name] = NovelOperationResult(
            operation=op_name,
            base_candidate_operations=base_ops,
            frozen_bank_exact_match=frozen_bank_em,
            frozen_bank_token_accuracy=frozen_bank_acc,
            scratch_exact_match=scratch_em,
            scratch_token_accuracy=scratch_acc,
            scratch_final_loss=scratch_loss,
            residual_exact_match=residual_em,
            residual_token_accuracy=residual_acc,
            residual_final_loss=residual_loss,
            temporary_parameter_count=temp_params,
            frozen_invariants_passed=frozen_passed,
            accuracy_passed=accuracy_passed,
            capacity_passed=capacity_passed,
            isolation_passed=isolation_passed,
            passed=op_passed,
        )

    # 4. Aggregate report
    mean_res_em = statistics.mean(res.residual_exact_match for res in results_by_operation.values())
    mean_scr_em = statistics.mean(res.scratch_exact_match for res in results_by_operation.values())
    mean_bnk_em = statistics.mean(
        res.frozen_bank_exact_match for res in results_by_operation.values()
    )

    all_frozen = all(res.frozen_invariants_passed for res in results_by_operation.values())
    all_acc = all(res.accuracy_passed for res in results_by_operation.values())
    all_cap = all(res.capacity_passed for res in results_by_operation.values())
    all_iso = all(res.isolation_passed for res in results_by_operation.values())
    overall_passed = all(res.passed for res in results_by_operation.values())

    elapsed = time.perf_counter() - start_time

    return PlasticWorkspaceBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        bank_param_count=bank_params,
        results_by_operation=results_by_operation,
        mean_residual_exact_match=mean_res_em,
        mean_scratch_exact_match=mean_scr_em,
        mean_frozen_bank_exact_match=mean_bnk_em,
        all_frozen_passed=all_frozen,
        all_accuracy_passed=all_acc,
        all_capacity_passed=all_cap,
        all_isolation_passed=all_iso,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class PlasticWorkspaceMultiSeedReport:
    """Multi-seed aggregate report for Task A1-B005."""

    seeds: tuple[int, ...]
    reports: list[PlasticWorkspaceBenchmarkReport]
    overall_passed: bool
    meets_seed_policy: bool
    mean_overall_residual_exact_match: float
    mean_overall_scratch_exact_match: float
    mean_overall_frozen_bank_exact_match: float
    per_operation_means: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "mean_overall_residual_exact_match": self.mean_overall_residual_exact_match,
            "mean_overall_scratch_exact_match": self.mean_overall_scratch_exact_match,
            "mean_overall_frozen_bank_exact_match": self.mean_overall_frozen_bank_exact_match,
            "per_operation_means": self.per_operation_means,
            "reports": [r.to_dict() for r in self.reports],
        }


def run_plastic_workspace_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Any,
    output_dir: Path | None = None,
) -> PlasticWorkspaceMultiSeedReport:
    """Run multi-seed plastic workspace residual learning benchmark."""
    seed_tuple = tuple(seeds)
    meets_policy = len(seed_tuple) >= MIN_GATE_SEEDS

    reports: list[PlasticWorkspaceBenchmarkReport] = []
    for s in seed_tuple:
        s_dir = output_dir / f"seed_{s}" if output_dir is not None else None
        cfg = config_factory(s)
        rep = run_plastic_workspace_benchmark(cfg, seed_dir=s_dir)
        reports.append(rep)
        if s_dir is not None:
            s_dir.mkdir(parents=True, exist_ok=True)
            (s_dir / "report.json").write_text(
                json.dumps(rep.to_dict(), indent=2), encoding="utf-8"
            )

    overall_passed = meets_policy and all(r.overall_passed for r in reports)
    mean_res_em = statistics.mean(r.mean_residual_exact_match for r in reports)
    mean_scr_em = statistics.mean(r.mean_scratch_exact_match for r in reports)
    mean_bnk_em = statistics.mean(r.mean_frozen_bank_exact_match for r in reports)

    per_op_means: dict[str, dict[str, Any]] = {}
    sample_rep = reports[0]
    for op_name in sample_rep.results_by_operation:
        res_ems = [r.results_by_operation[op_name].residual_exact_match for r in reports]
        res_accs = [r.results_by_operation[op_name].residual_token_accuracy for r in reports]
        scr_ems = [r.results_by_operation[op_name].scratch_exact_match for r in reports]
        bnk_ems = [r.results_by_operation[op_name].frozen_bank_exact_match for r in reports]
        temp_params = [r.results_by_operation[op_name].temporary_parameter_count for r in reports]

        per_op_means[op_name] = {
            "mean_residual_exact_match": statistics.mean(res_ems),
            "stdev_residual_exact_match": statistics.stdev(res_ems) if len(res_ems) > 1 else 0.0,
            "min_residual_exact_match": min(res_ems),
            "max_residual_exact_match": max(res_ems),
            "mean_residual_token_accuracy": statistics.mean(res_accs),
            "mean_scratch_exact_match": statistics.mean(scr_ems),
            "mean_frozen_bank_exact_match": statistics.mean(bnk_ems),
            "temporary_parameter_count": temp_params[0],
            "frozen_invariants_passed": all(
                r.results_by_operation[op_name].frozen_invariants_passed for r in reports
            ),
            "accuracy_passed": all(
                r.results_by_operation[op_name].accuracy_passed for r in reports
            ),
            "capacity_passed": all(
                r.results_by_operation[op_name].capacity_passed for r in reports
            ),
            "isolation_passed": all(
                r.results_by_operation[op_name].isolation_passed for r in reports
            ),
            "passed": all(r.results_by_operation[op_name].passed for r in reports),
        }

    return PlasticWorkspaceMultiSeedReport(
        seeds=seed_tuple,
        reports=reports,
        overall_passed=overall_passed,
        meets_seed_policy=meets_policy,
        mean_overall_residual_exact_match=mean_res_em,
        mean_overall_scratch_exact_match=mean_scr_em,
        mean_overall_frozen_bank_exact_match=mean_bnk_em,
        per_operation_means=per_op_means,
    )

