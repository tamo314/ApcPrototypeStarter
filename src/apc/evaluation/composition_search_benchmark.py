"""Composition Search Baseline Benchmark (Phase A.1 Post-Diagnostic Task A1-B004, Milestone B-M4).

Recovers multi-step composition recipes for novel composite tasks without oracle primitive
identity using heuristic/beam search over the compact primitive bank, and evaluates recovered
recipes on held-out test data.

Acceptance Criteria (CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md):
1. Recovered recipe accuracy >= 0.85 on held-out test data.
2. Recovered recipe functionally matches oracle composition
   (identical operations or >= 99% agreement).
3. Zero bank expansion (resident primitives and parameters strictly unchanged).
4. Evaluated across 5 seeds (0, 1, 2, 3, 4).
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.composition_library_benchmark import (
    DESIGNATED_COMPOSITIONS,
    _generate_composition_examples,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    PARAMETERIZED_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
    _train_single_primitive,
)
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.composition_search import search_composition_recipe
from apc.primitives.conditioning import DEFAULT_MAX_SEQUENCE_LENGTH
from apc.utils.seed import set_seed

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS = 5
COMPOSITION_SEARCH_ACCURACY_THRESHOLD = 0.85
FUNCTIONAL_AGREEMENT_THRESHOLD = 0.99
_EVAL_BATCH_SIZE = 128


@dataclass(frozen=True)
class CompositionSearchBenchmarkConfig:
    """Configuration for Task A1-B004 Composition Search Baseline Benchmark."""

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

    # Training steps for primitives over frozen core (fallback if missing)
    parameterized_train_steps: int = 10000
    parameter_free_train_steps: int = 6000
    operator_lr: float = 0.0008
    operator_weight_decay: float = 0.0001
    operator_grad_clip: float = 1.0

    core_train_steps: int = 16000
    core_lr: float = 0.0003
    core_weight_decay: float = 0.0001
    shared_encoder_checkpoint: str | None = None

    # Search & evaluation parameters
    num_adaptation_examples: int = 32
    num_eval_examples_per_composition: int = 500
    min_eval_examples_per_composition: int = 200
    max_depth: int = 2
    beam_width: int = 16
    accuracy_threshold: float = COMPOSITION_SEARCH_ACCURACY_THRESHOLD

    def __post_init__(self) -> None:
        if self.num_eval_examples_per_composition < self.min_eval_examples_per_composition:
            raise ValueError("num_eval_examples_per_composition below minimum")
        if self.num_adaptation_examples < 8:
            raise ValueError("num_adaptation_examples must be >= 8")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def composition_search_config_from_dict(raw: dict[str, Any]) -> CompositionSearchBenchmarkConfig:
    """Parse configuration dict, filling in defaults."""
    defaults = CompositionSearchBenchmarkConfig()
    return CompositionSearchBenchmarkConfig(
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
        num_adaptation_examples=raw.get(
            "num_adaptation_examples", defaults.num_adaptation_examples
        ),
        num_eval_examples_per_composition=raw.get(
            "num_eval_examples_per_composition", defaults.num_eval_examples_per_composition
        ),
        min_eval_examples_per_composition=raw.get(
            "min_eval_examples_per_composition", defaults.min_eval_examples_per_composition
        ),
        max_depth=raw.get("max_depth", defaults.max_depth),
        beam_width=raw.get("beam_width", defaults.beam_width),
        accuracy_threshold=raw.get("accuracy_threshold", defaults.accuracy_threshold),
    )


@dataclass(frozen=True)
class SearchCompositionResult:
    """Benchmark metrics for a single composition search evaluation."""

    composition_name: str
    oracle_operations: tuple[str, str]
    recovered_operations: tuple[str, ...]
    recovered_exact_match: float
    recovered_token_accuracy: float
    oracle_exact_match: float
    functional_agreement: float
    functionally_matches: bool
    accuracy_passed: bool
    zero_expansion_passed: bool
    passed: bool
    candidates_evaluated: int
    candidates_pruned: int
    search_time_seconds: float
    unselected_primitive_calls: int


@dataclass(frozen=True)
class CompositionSearchBenchmarkReport:
    """Single seed report for Task A1-B004 Composition Search Benchmark."""

    config: CompositionSearchBenchmarkConfig
    seed: int
    core_param_count: int
    initial_bank_params: int
    final_bank_params: int
    initial_bank_primitives: int
    final_bank_primitives: int
    results_by_composition: dict[str, SearchCompositionResult]
    mean_recovered_exact_match: float
    mean_functional_agreement: float
    mean_oracle_exact_match: float
    total_candidates_evaluated: int
    total_candidates_pruned: int
    total_search_time_seconds: float
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "core_param_count": self.core_param_count,
            "initial_bank_params": self.initial_bank_params,
            "final_bank_params": self.final_bank_params,
            "initial_bank_primitives": self.initial_bank_primitives,
            "final_bank_primitives": self.final_bank_primitives,
            "mean_recovered_exact_match": self.mean_recovered_exact_match,
            "mean_functional_agreement": self.mean_functional_agreement,
            "mean_oracle_exact_match": self.mean_oracle_exact_match,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "total_candidates_pruned": self.total_candidates_pruned,
            "total_search_time_seconds": self.total_search_time_seconds,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "results_by_composition": {
                name: dataclasses.asdict(res)
                for name, res in self.results_by_composition.items()
            },
        }


def run_composition_search_benchmark(
    config: CompositionSearchBenchmarkConfig,
    *,
    seed_dir: Path | None = None,
) -> CompositionSearchBenchmarkReport:
    """Run composition search baseline benchmark for a single seed."""
    start_time = time.perf_counter()
    set_seed(config.seed)

    # 1. Obtain or train frozen shared Stable Core
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
        arg_dim=config.arg_dim,
        max_sequence_length=config.max_sequence_length,
        parameterized_train_steps=config.parameterized_train_steps,
        parameter_free_train_steps=config.parameter_free_train_steps,
        operator_lr=config.operator_lr,
        operator_weight_decay=config.operator_weight_decay,
        operator_grad_clip=config.operator_grad_clip,
        core_train_steps=config.core_train_steps,
        core_lr=config.core_lr,
        core_weight_decay=config.core_weight_decay,
        shared_encoder_checkpoint=config.shared_encoder_checkpoint,
    )

    core = _get_or_train_frozen_shared_core(u_config, seed_dir=seed_dir)
    core_params = sum(p.numel() for p in core.model.parameters())

    # 2. Build or load compact heterogeneous primitive bank
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(core.device)

    # Check for existing checkpoint: seed_dir first, then fallbacks
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
            steps = (
                config.parameterized_train_steps
                if op in PARAMETERIZED_OPERATIONS
                else config.parameter_free_train_steps
            )
            _train_single_primitive(core, p, u_config, op, steps=steps)

        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bank.state_dict(), seed_dir / "primitive_bank.pt")

    # Freeze bank completely
    bank.freeze_all()
    bank.eval()

    initial_bank_params = bank.total_parameter_count()
    initial_bank_primitives = len(bank)

    # 3. Evaluate each designated composition via search
    results_by_comp: dict[str, SearchCompositionResult] = {}

    with torch.no_grad():
        for op1, op2 in DESIGNATED_COMPOSITIONS:
            comp_name = f"{op1}->{op2}"

            # Adaptation support set (N=32)
            adaptation_examples = _generate_composition_examples(
                config.seed,
                (op1, op2),
                config.num_adaptation_examples,
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            # Held-out evaluation set (N=500)
            eval_examples = _generate_composition_examples(
                config.seed + 10000,
                (op1, op2),
                config.num_eval_examples_per_composition,
                vocab_size=config.vocab_size,
                sequence_length_range=config.sequence_length_range,
            )

            # Execute composition search WITHOUT oracle primitive identity
            search_res = search_composition_recipe(
                core,
                bank,
                op_to_id,
                adaptation_examples,
                max_depth=config.max_depth,
                beam_width=config.beam_width,
            )

            # Reset call counters to audit evaluation phase
            bank.reset_all_forward_call_counts()

            # Evaluate recovered recipe on held-out evaluation set
            rec_exact_matches = 0
            rec_correct_tokens = 0
            total_tokens = 0
            recovered_preds_all: list[tuple[int, ...]] = []

            for start in range(0, len(eval_examples), _EVAL_BATCH_SIZE):
                chunk = eval_examples[start : start + _EVAL_BATCH_SIZE]
                logits = execute_composition_recipe(
                    core,
                    bank,
                    op_to_id,
                    chunk,
                    candidate_operations=search_res.candidate_operations,
                )
                predictions = logits.argmax(dim=-1)

                for row, ex in enumerate(chunk):
                    target = ex.target_tokens
                    n = len(target)
                    pred = tuple(predictions[row, :n].tolist())
                    recovered_preds_all.append(pred)
                    total_tokens += n
                    rec_correct_tokens += sum(
                        1 for p, t in zip(pred, target, strict=True) if p == t
                    )
                    if pred == target:
                        rec_exact_matches += 1

            rec_em = rec_exact_matches / len(eval_examples)
            rec_acc = rec_correct_tokens / max(1, total_tokens)

            # Evaluate oracle recipe on held-out evaluation set
            oracle_exact_matches = 0
            oracle_preds_all: list[tuple[int, ...]] = []

            for start in range(0, len(eval_examples), _EVAL_BATCH_SIZE):
                chunk = eval_examples[start : start + _EVAL_BATCH_SIZE]
                logits_oracle = execute_composition_recipe(
                    core,
                    bank,
                    op_to_id,
                    chunk,
                )
                predictions_oracle = logits_oracle.argmax(dim=-1)

                for row, ex in enumerate(chunk):
                    target = ex.target_tokens
                    n = len(target)
                    pred_oracle = tuple(predictions_oracle[row, :n].tolist())
                    oracle_preds_all.append(pred_oracle)
                    if pred_oracle == target:
                        oracle_exact_matches += 1

            oracle_em = oracle_exact_matches / len(eval_examples)

            # Compute functional agreement between recovered and oracle outputs
            agreements = sum(
                1 for p_rec, p_ora in zip(recovered_preds_all, oracle_preds_all, strict=True)
                if p_rec == p_ora
            )
            functional_agreement = agreements / len(eval_examples)
            functionally_matches = (
                functional_agreement >= FUNCTIONAL_AGREEMENT_THRESHOLD
                or search_res.candidate_operations == (op1, op2)
            )

            # Check sparse calls
            counts = bank.forward_call_counts()
            cand_pids = set(op_to_id[op] for op in search_res.candidate_operations)
            participating_ids = tuple(sorted(cand_pids))
            unselected_calls = sum(
                cnt for pid, cnt in counts.items() if pid not in participating_ids
            )

            accuracy_passed = rec_em >= config.accuracy_threshold
            zero_expansion_passed = (
                bank.total_parameter_count() == initial_bank_params
                and len(bank) == initial_bank_primitives
            )
            passed = accuracy_passed and functionally_matches and zero_expansion_passed

            results_by_comp[comp_name] = SearchCompositionResult(
                composition_name=comp_name,
                oracle_operations=(op1, op2),
                recovered_operations=search_res.candidate_operations,
                recovered_exact_match=rec_em,
                recovered_token_accuracy=rec_acc,
                oracle_exact_match=oracle_em,
                functional_agreement=functional_agreement,
                functionally_matches=functionally_matches,
                accuracy_passed=accuracy_passed,
                zero_expansion_passed=zero_expansion_passed,
                passed=passed,
                candidates_evaluated=search_res.candidates_evaluated,
                candidates_pruned=search_res.candidates_pruned,
                search_time_seconds=search_res.search_time_seconds,
                unselected_primitive_calls=unselected_calls,
            )

    mean_rec_em = statistics.mean(res.recovered_exact_match for res in results_by_comp.values())
    mean_agreement = statistics.mean(res.functional_agreement for res in results_by_comp.values())
    mean_oracle_em = statistics.mean(res.oracle_exact_match for res in results_by_comp.values())

    total_evaluated = sum(res.candidates_evaluated for res in results_by_comp.values())
    total_pruned = sum(res.candidates_pruned for res in results_by_comp.values())
    total_search_time = sum(res.search_time_seconds for res in results_by_comp.values())

    all_passed = all(res.passed for res in results_by_comp.values())
    overall_passed = all_passed and (mean_rec_em >= config.accuracy_threshold)

    elapsed = time.perf_counter() - start_time

    return CompositionSearchBenchmarkReport(
        config=config,
        seed=config.seed,
        core_param_count=core_params,
        initial_bank_params=initial_bank_params,
        final_bank_params=bank.total_parameter_count(),
        initial_bank_primitives=initial_bank_primitives,
        final_bank_primitives=len(bank),
        results_by_composition=results_by_comp,
        mean_recovered_exact_match=mean_rec_em,
        mean_functional_agreement=mean_agreement,
        mean_oracle_exact_match=mean_oracle_em,
        total_candidates_evaluated=total_evaluated,
        total_candidates_pruned=total_pruned,
        total_search_time_seconds=total_search_time,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


@dataclass(frozen=True)
class CompositionSearchMultiSeedReport:
    """Multi-seed summary for Task A1-B004 Composition Search Benchmark."""

    seeds: tuple[int, ...]
    reports: list[CompositionSearchBenchmarkReport]
    overall_passed: bool
    meets_seed_policy: bool
    mean_overall_recovered_exact_match: float
    mean_overall_functional_agreement: float
    per_composition_means: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "mean_overall_recovered_exact_match": self.mean_overall_recovered_exact_match,
            "mean_overall_functional_agreement": self.mean_overall_functional_agreement,
            "per_composition_means": self.per_composition_means,
            "reports": [r.to_dict() for r in self.reports],
        }


def run_composition_search_benchmark_multi_seed(
    seeds: Sequence[int],
    config_factory: Any,
    output_dir: Path | None = None,
) -> CompositionSearchMultiSeedReport:
    """Run multi-seed composition search benchmark."""
    seed_tuple = tuple(seeds)
    meets_policy = len(seed_tuple) >= MIN_GATE_SEEDS

    reports: list[CompositionSearchBenchmarkReport] = []
    for s in seed_tuple:
        s_dir = output_dir / f"seed_{s}" if output_dir is not None else None
        cfg = config_factory(s)
        rep = run_composition_search_benchmark(cfg, seed_dir=s_dir)
        reports.append(rep)
        if s_dir is not None:
            s_dir.mkdir(parents=True, exist_ok=True)
            (s_dir / "report.json").write_text(
                json.dumps(rep.to_dict(), indent=2), encoding="utf-8"
            )

    overall_passed = meets_policy and all(r.overall_passed for r in reports)
    mean_overall_em = statistics.mean(r.mean_recovered_exact_match for r in reports)
    mean_overall_agree = statistics.mean(r.mean_functional_agreement for r in reports)

    per_comp_means: dict[str, dict[str, Any]] = {}
    comp_names = [f"{op1}->{op2}" for op1, op2 in DESIGNATED_COMPOSITIONS]
    for name in comp_names:
        rec_ems = [r.results_by_composition[name].recovered_exact_match for r in reports]
        rec_accs = [r.results_by_composition[name].recovered_token_accuracy for r in reports]
        agrees = [r.results_by_composition[name].functional_agreement for r in reports]
        recovered_ops_list = [
            r.results_by_composition[name].recovered_operations for r in reports
        ]

        per_comp_means[name] = {
            "mean_recovered_exact_match": statistics.mean(rec_ems),
            "mean_recovered_token_accuracy": statistics.mean(rec_accs),
            "mean_functional_agreement": statistics.mean(agrees),
            "recovered_operations_sample": list(recovered_ops_list[0]),
            "accuracy_passed": all(
                r.results_by_composition[name].accuracy_passed for r in reports
            ),
            "functionally_matches": all(
                r.results_by_composition[name].functionally_matches for r in reports
            ),
            "zero_expansion_passed": all(
                r.results_by_composition[name].zero_expansion_passed for r in reports
            ),
        }

    return CompositionSearchMultiSeedReport(
        seeds=seed_tuple,
        reports=reports,
        overall_passed=overall_passed,
        meets_seed_policy=meets_policy,
        mean_overall_recovered_exact_match=mean_overall_em,
        mean_overall_functional_agreement=mean_overall_agree,
        per_composition_means=per_comp_means,
    )
