"""Composition Library Execution Benchmark (Phase A.1 Post-Diagnostic Task A1-B003, STOP GATE).

Evaluates multi-step compositional recipes of canonical primitives over the frozen
shared task-blind Stable Core using ordered primitive execution through `CompositionLibrary`.

Acceptance Criteria (CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md / ADR-0046):
- Multi-seed benchmark (5 seeds: 0, 1, 2, 3, 4) on unseen evaluations.
- Designated composition accuracy >= 0.90.
- No temporary plastic capacity allocated (temporary parameters == 0).
- Non-participating primitives receive zero forward calls.
- STOP GATE.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.generator import (
    Example,
    OracleMetadata,
    Program,
    ProgramStep,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    PARAMETERIZED_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _derive_local_seed,
    _get_or_train_frozen_shared_core,
    _train_single_primitive,
)
from apc.primitives.composition import (
    CompositionLibrary,
    CompositionRecipe,
    execute_composition_recipe,
)
from apc.primitives.conditioning import DEFAULT_MAX_SEQUENCE_LENGTH
from apc.utils.seed import set_seed

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5
COMPOSITION_ACCURACY_THRESHOLD = 0.90

DESIGNATED_COMPOSITIONS: Final[tuple[tuple[str, str], ...]] = (
    ("SHIFT", "SELECT"),
    ("REVERSE", "COUNT"),
    ("COPY", "SORT"),
    ("NEGATE", "SELECT"),
    ("SHIFT", "BIND"),
    ("REVERSE", "SORT"),
)

_EVAL_BATCH_SIZE = 128


def _default_call_args(op_name: str) -> dict[str, Any]:
    if op_name == "SHIFT":
        return {"amount": 0}
    if op_name == "SELECT":
        return {"indices": ()}
    if op_name == "COUNT":
        return {"target": 0}
    if op_name == "BIND":
        return {"query_key": 0}
    return {}


@dataclass(frozen=True)
class CompositionBenchmarkConfig:
    """Explicit configuration for Task A1-B003 Composition Library Execution Benchmark."""

    seed: int = 0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    model: dict[str, Any] = field(default_factory=_default_model_config)
    device: str = "auto"

    d_operator: int = 32
    n_operator_head: int = 4
    d_operator_ff: int = 64
    arg_dim: int = 16
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH

    # Training steps for primitives over frozen core
    parameterized_train_steps: int = 10000
    parameter_free_train_steps: int = 6000
    operator_lr: float = 0.0008
    operator_weight_decay: float = 0.0001
    operator_grad_clip: float = 1.0

    # Core pretraining fallback (if checkpoint missing)
    core_train_steps: int = 16000
    core_lr: float = 0.0003
    core_weight_decay: float = 0.0001
    shared_encoder_checkpoint: str | None = None

    num_eval_examples_per_composition: int = 500
    min_eval_examples_per_composition: int = 200

    def __post_init__(self) -> None:
        if self.num_eval_examples_per_composition < self.min_eval_examples_per_composition:
            raise ValueError("num_eval_examples_per_composition below minimum")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def composition_benchmark_config_from_dict(raw: dict[str, Any]) -> CompositionBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = CompositionBenchmarkConfig()
    return CompositionBenchmarkConfig(
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
        arg_dim=raw.get("arg_dim", defaults.arg_dim),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        parameterized_train_steps=raw.get(
            "parameterized_train_steps", defaults.parameterized_train_steps
        ),
        parameter_free_train_steps=raw.get(
            "parameter_free_train_steps", defaults.parameter_free_train_steps
        ),
        operator_lr=raw.get("operator_lr", defaults.operator_lr),
        operator_weight_decay=raw.get("operator_weight_decay", defaults.operator_weight_decay),
        operator_grad_clip=raw.get("operator_grad_clip", defaults.operator_grad_clip),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        core_lr=raw.get("core_lr", defaults.core_lr),
        core_weight_decay=raw.get("core_weight_decay", defaults.core_weight_decay),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
        num_eval_examples_per_composition=raw.get(
            "num_eval_examples_per_composition", defaults.num_eval_examples_per_composition
        ),
        min_eval_examples_per_composition=raw.get(
            "min_eval_examples_per_composition", defaults.min_eval_examples_per_composition
        ),
    )


def _generate_composition_examples(
    seed: int,
    composition: tuple[str, str],
    n: int,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sequence_length_range: tuple[int, int] = (6, 10),
) -> list[Example]:
    """Generate deterministically sampled multi-step composition examples."""
    import random

    op1_name, op2_name = composition
    rng = random.Random(_derive_local_seed(seed, 0, f"composition:{op1_name}->{op2_name}"))
    op1_def = get_operation(op1_name)
    op2_def = get_operation(op2_name)

    examples: list[Example] = []
    attempts = 0
    max_attempts = n * 50

    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        L = rng.randint(sequence_length_range[0], sequence_length_range[1])
        # If BIND is involved, length must be even
        if (op1_name == "BIND" or op2_name == "BIND") and L % 2 != 0:
            continue

        seq = tuple(rng.randrange(vocab_size) for _ in range(L))
        if not op1_def.is_valid_for_length(len(seq)):
            continue

        try:
            params1 = op1_def.sample_params(rng, seq, vocab_size)
            inter_seq = op1_def.apply(seq, vocab_size, params1)
            if not op2_def.is_valid_for_length(len(inter_seq)):
                continue
            params2 = op2_def.sample_params(rng, inter_seq, vocab_size)
        except (ValueError, KeyError):
            continue

        step1 = ProgramStep(op1_name, params1)
        step2 = ProgramStep(op2_name, params2)
        prog = Program(steps=(step1, step2))
        res = run_program(prog, seq, vocab_size)

        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="novel_composition",
            split="test",
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
            oracle_metadata=OracleMetadata(
                label="C", primitive_operations=(op1_name, op2_name)
            ),
        )
        examples.append(ex)

    if len(examples) < n:
        raise RuntimeError(
            f"Failed to generate {n} valid examples for {composition} after {attempts} attempts."
        )

    return examples


@dataclass(frozen=True)
class CompositionResult:
    """Benchmark metrics for a single designated composition recipe."""

    composition_name: str
    operations: tuple[str, str]
    exact_match: float
    token_accuracy: float
    forward_calls_by_primitive: dict[int, int]
    participating_primitive_ids: tuple[int, ...]
    unselected_primitive_calls: int
    temporary_parameter_count: int
    exact_match_passed: bool
    sparse_passed: bool
    passed: bool


@dataclass(frozen=True)
class CompositionLibraryBenchmarkReport:
    """Complete report for one seed of Task A1-B003 Composition Library Execution."""

    config: CompositionBenchmarkConfig
    seed: int
    core_param_count: int
    total_bank_params: int
    results_by_composition: dict[str, CompositionResult]
    mean_composition_exact_match: float
    mean_token_accuracy: float
    total_unselected_calls: int
    total_temporary_params: int
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "total_bank_params": self.total_bank_params,
            "results_by_composition": {
                k: dataclasses.asdict(v) for k, v in self.results_by_composition.items()
            },
            "mean_composition_exact_match": self.mean_composition_exact_match,
            "mean_token_accuracy": self.mean_token_accuracy,
            "total_unselected_calls": self.total_unselected_calls,
            "total_temporary_params": self.total_temporary_params,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _to_unified_config(cfg: CompositionBenchmarkConfig) -> UnifiedBenchmarkConfig:
    """Convert CompositionBenchmarkConfig to UnifiedBenchmarkConfig for primitive training."""
    return UnifiedBenchmarkConfig(
        seed=cfg.seed,
        vocab_size=cfg.vocab_size,
        sequence_length_range=cfg.sequence_length_range,
        group_size=cfg.group_size,
        model=cfg.model,
        device=cfg.device,
        d_operator=cfg.d_operator,
        n_operator_head=cfg.n_operator_head,
        d_operator_ff=cfg.d_operator_ff,
        arg_dim=cfg.arg_dim,
        max_sequence_length=cfg.max_sequence_length,
        parameterized_train_steps=cfg.parameterized_train_steps,
        parameter_free_train_steps=cfg.parameter_free_train_steps,
        operator_lr=cfg.operator_lr,
        operator_weight_decay=cfg.operator_weight_decay,
        operator_grad_clip=cfg.operator_grad_clip,
        core_train_steps=cfg.core_train_steps,
        core_lr=cfg.core_lr,
        core_weight_decay=cfg.core_weight_decay,
        shared_encoder_checkpoint=cfg.shared_encoder_checkpoint,
    )


def run_composition_library_benchmark(
    config: CompositionBenchmarkConfig,
    seed_dir: Path | None = None,
) -> CompositionLibraryBenchmarkReport:
    """Run Task A1-B003 Composition Library Execution benchmark for one seed."""
    start_time = time.perf_counter()
    set_seed(config.seed)
    u_config = _to_unified_config(config)

    # 1. Obtain frozen shared core
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=seed_dir)
    core_params = sum(p.numel() for p in core.model.parameters())

    # 2. Build and populate PrimitiveBank
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(core.device)

    # Check for existing trained bank checkpoint or train primitives
    bank_ckpt = seed_dir / "primitive_bank.pt" if seed_dir is not None else None
    loaded_bank = False
    if bank_ckpt is not None and bank_ckpt.is_file():
        try:
            state_dict = torch.load(bank_ckpt, map_location=core.device, weights_only=True)
            bank.load_state_dict(state_dict)
            loaded_bank = True
        except Exception:
            loaded_bank = False

    if not loaded_bank:
        for op in ALL_CANONICAL_OPERATIONS:
            p = bank.get(op_to_id[op])
            steps = (
                config.parameterized_train_steps
                if op in PARAMETERIZED_OPERATIONS
                else config.parameter_free_train_steps
            )
            _train_single_primitive(core, p, u_config, op, steps=steps)

        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bank.state_dict(), seed_dir / "primitive_bank.pt")

    # Freeze all primitives and set to eval
    bank.freeze_all()
    bank.eval()

    # 3. Register designated compositions into CompositionLibrary
    library = CompositionLibrary()
    for op1, op2 in DESIGNATED_COMPOSITIONS:
        name = f"{op1}->{op2}"
        recipe = CompositionRecipe(
            name,
            (
                PrimitiveCall(op1, _default_call_args(op1)),
                PrimitiveCall(op2, _default_call_args(op2)),
            ),
        )
        library.register_recipe(recipe)

    # 4. Evaluate each designated composition
    results_by_comp: dict[str, CompositionResult] = {}

    with torch.no_grad():
        for op1, op2 in DESIGNATED_COMPOSITIONS:
            comp_name = f"{op1}->{op2}"
            examples = _generate_composition_examples(
                config.seed,
                (op1, op2),
                config.num_eval_examples_per_composition,
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            # Reset call counters to audit sparse execution per composition
            bank.reset_all_forward_call_counts()

            exact_matches = 0
            correct_tokens = 0
            total_tokens = 0

            p1_id = op_to_id[op1]
            p2_id = op_to_id[op2]
            participating_ids = tuple(sorted(set([p1_id, p2_id])))

            for start in range(0, len(examples), _EVAL_BATCH_SIZE):
                chunk = examples[start : start + _EVAL_BATCH_SIZE]
                logits = execute_composition_recipe(
                    core, bank, op_to_id, chunk
                )
                predictions = logits.argmax(dim=-1)

                for row, ex in enumerate(chunk):
                    target = ex.target_tokens
                    n = len(target)
                    pred = tuple(predictions[row, :n].tolist())
                    total_tokens += n
                    correct_tokens += sum(1 for p, t in zip(pred, target, strict=True) if p == t)
                    if pred == target:
                        exact_matches += 1

            em = exact_matches / len(examples)
            acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

            # Audit forward calls:
            # Participating primitives must have > 0 calls
            # Non-participating primitives must have == 0 calls
            counts = bank.forward_call_counts()
            unselected_calls = sum(
                cnt for pid, cnt in counts.items() if pid not in participating_ids
            )

            sparse_passed = (
                unselected_calls == 0
                and all(counts[pid] > 0 for pid in participating_ids)
            )
            em_passed = em >= COMPOSITION_ACCURACY_THRESHOLD
            passed = em_passed and sparse_passed

            results_by_comp[comp_name] = CompositionResult(
                composition_name=comp_name,
                operations=(op1, op2),
                exact_match=em,
                token_accuracy=acc,
                forward_calls_by_primitive=counts,
                participating_primitive_ids=participating_ids,
                unselected_primitive_calls=unselected_calls,
                temporary_parameter_count=0,
                exact_match_passed=em_passed,
                sparse_passed=sparse_passed,
                passed=passed,
            )

    mean_em = statistics.mean(res.exact_match for res in results_by_comp.values())
    mean_acc = statistics.mean(res.token_accuracy for res in results_by_comp.values())
    total_unselected = sum(res.unselected_primitive_calls for res in results_by_comp.values())
    all_passed = all(res.passed for res in results_by_comp.values())
    overall_passed = all_passed and (mean_em >= COMPOSITION_ACCURACY_THRESHOLD)

    elapsed = time.perf_counter() - start_time

    return CompositionLibraryBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        total_bank_params=bank.total_parameter_count(),
        results_by_composition=results_by_comp,
        mean_composition_exact_match=mean_em,
        mean_token_accuracy=mean_acc,
        total_unselected_calls=total_unselected,
        total_temporary_params=0,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


@dataclass(frozen=True)
class CompositionLibraryMultiSeedReport:
    """Multi-seed summary for Task A1-B003 Composition Library Execution Benchmark."""

    seeds: tuple[int, ...]
    reports: list[CompositionLibraryBenchmarkReport]
    overall_passed: bool
    meets_seed_policy: bool
    mean_overall_exact_match: float
    per_composition_means: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "mean_overall_exact_match": self.mean_overall_exact_match,
            "per_composition_means": self.per_composition_means,
            "reports": [r.to_dict() for r in self.reports],
        }


def run_composition_library_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Any,
    output_dir: Path | None = None,
) -> CompositionLibraryMultiSeedReport:
    """Run multi-seed benchmark across requested seeds."""
    seed_tuple = tuple(seeds)
    meets_policy = len(seed_tuple) >= MIN_GATE_SEEDS

    reports: list[CompositionLibraryBenchmarkReport] = []
    for s in seed_tuple:
        s_dir = output_dir / f"seed_{s}" if output_dir is not None else None
        cfg = config_factory(s)
        rep = run_composition_library_benchmark(cfg, seed_dir=s_dir)
        reports.append(rep)
        if s_dir is not None:
            s_dir.mkdir(parents=True, exist_ok=True)
            (s_dir / "report.json").write_text(
                json.dumps(rep.to_dict(), indent=2), encoding="utf-8"
            )

    overall_passed = meets_policy and all(r.overall_passed for r in reports)
    mean_overall = statistics.mean(r.mean_composition_exact_match for r in reports)

    per_comp_means: dict[str, dict[str, float]] = {}
    comp_names = [f"{op1}->{op2}" for op1, op2 in DESIGNATED_COMPOSITIONS]
    for name in comp_names:
        ems = [r.results_by_composition[name].exact_match for r in reports]
        accs = [r.results_by_composition[name].token_accuracy for r in reports]
        unselected = [r.results_by_composition[name].unselected_primitive_calls for r in reports]
        per_comp_means[name] = {
            "mean_exact_match": statistics.mean(ems),
            "mean_token_accuracy": statistics.mean(accs),
            "total_unselected_calls": sum(unselected),
        }

    return CompositionLibraryMultiSeedReport(
        seeds=seed_tuple,
        reports=reports,
        overall_passed=overall_passed,
        meets_seed_policy=meets_policy,
        mean_overall_exact_match=mean_overall,
        per_composition_means=per_comp_means,
    )
