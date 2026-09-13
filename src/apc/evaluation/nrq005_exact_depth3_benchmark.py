"""NRQ-005 — Missing-Artifact-Robust Exact-Depth-3 Irreducible Composition Benchmark.

Independent replacement protocol for NRQ-003. NRQ-003 is not resumed or marked
completed. Seed 0 exclusion is pre-fixed to existing artifact loss (checkpoint
overwrite on 2026-09-13, BUNDLE_LOSS documented in NRQ-004 / ADR-0164).

Evaluates all integrity-verified seeds (1, 2, 3, 4) from reconstructed bundles
in runs/nrq004_reconstructed_bundles/. Zero new training updates, zero parameter
modifications, zero relation additions, and zero sealed-partition access.

Workflow:
1. Symbolic Probe: Before looking at model outputs, verifies that candidate
   exact-depth-3 recipes are not functionally equivalent to any depth <= 2
   recipe, and establishes reducible depth-3 recipes and depth-2 canonical recipes
   as controls.
2. Dataset Generation: 5 fixed data seeds, support N=32, disjoint held-out inputs
   (N=100) per recipe.
3. Comparative Search: On identical inputs, compares:
   - Oracle recipe execution
   - Existing heuristic beam search (beam_width=16, max_depth=3)
   - Exhaustive lawful search over all 584 depth <= 3 candidates
4. Diagnostic Accounting: Records recipe recovery, functional EM, candidate counts,
   wall time, peak memory, and beam-budget sensitivity.
5. Decision Gate:
   - If oracle floor is failed (< 0.85), halts as depth-composition execution failure.
   - If oracle healthy and APC >= 0.95 and strongest deterministic baseline < 0.95:
     rejects ADR-0162 closure.
   - If baseline is also >= 0.95 across all bundle and data seeds: empirically
     supports ADR-0162 closure restricted to depth <= 3 of the current registry.
"""

from __future__ import annotations

import itertools
import json
import random
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from apc.environments.generator import Example
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.composition_search_benchmark import (
    COMPOSITION_SEARCH_ACCURACY_THRESHOLD,
)
from apc.evaluation.nrq004_bundle_reconstruction import (
    build_a1_b004_tokens,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    PARAMETERIZED_OPERATION_NAMES,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import (
    CompositionRecipe,
    execute_composition_recipe,
    resolve_candidate_calls,
)
from apc.primitives.composition_search import (
    SearchResult,
    _evaluate_candidate_on_adaptation,
    is_candidate_structurally_valid,
    search_composition_recipe,
)

# 8 canonical/novel operations present in Phase A.1 PrimitiveBank
BANK_OPERATIONS: tuple[str, ...] = (
    "SELECT",
    "COUNT",
    "BIND",
    "SHIFT",
    "COPY",
    "REVERSE",
    "SORT",
    "NEGATE",
)

# Fixed Panel of Exact-Depth-3 Irreducible Recipes
EXACT_DEPTH3_IRREDUCIBLE_PANEL: tuple[tuple[str, str, str], ...] = (
    ("SHIFT", "REVERSE", "SELECT"),
    ("SHIFT", "NEGATE", "SELECT"),
    ("REVERSE", "NEGATE", "SELECT"),
    ("SHIFT", "REVERSE", "BIND"),
    ("NEGATE", "SHIFT", "SELECT"),
    ("REVERSE", "SHIFT", "BIND"),
)

# Controls: Reducible Depth-3 Recipes
REDUCIBLE_DEPTH3_PANEL: tuple[tuple[str, str, str], ...] = (
    ("COPY", "SHIFT", "SELECT"),       # reduces to SHIFT->SELECT (depth 2)
    ("REVERSE", "REVERSE", "COUNT"),   # reduces to COUNT (depth 1)
    ("NEGATE", "NEGATE", "SELECT"),    # reduces to SELECT (depth 1)
    ("SORT", "SORT", "SELECT"),        # reduces to SORT->SELECT (depth 2)
    ("REVERSE", "NEGATE", "SORT"),     # reduces to NEGATE->SORT (depth 2)
    ("SHIFT", "REVERSE", "SORT"),      # reduces to SORT (depth 1)
)

# Controls: Existing Depth-2 Canonical Recipes (Task A1-B004)
DEPTH2_CANONICAL_PANEL: tuple[tuple[str, str], ...] = (
    ("SHIFT", "SELECT"),
    ("REVERSE", "COUNT"),
    ("COPY", "SORT"),
    ("NEGATE", "SELECT"),
    ("SHIFT", "BIND"),
    ("REVERSE", "SORT"),
)

# Standard evaluation settings
DEFAULT_DATA_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
DEFAULT_SUPPORT_N: int = 32
DEFAULT_EVAL_N: int = 100
ORACLE_FLOOR_THRESHOLD: float = COMPOSITION_SEARCH_ACCURACY_THRESHOLD  # 0.85
CLOSURE_CONFIRMATION_THRESHOLD: float = 0.95
BEAM_SENSITIVITY_WIDTHS: tuple[int, ...] = (1, 4, 8, 16, 32, 64)


def enumerate_all_depth_le_3_candidates() -> list[tuple[str, ...]]:
    """Enumerate all 584 candidate operation sequences of depth <= 3."""
    candidates: list[tuple[str, ...]] = []
    for d in (1, 2, 3):
        for seq in itertools.product(BANK_OPERATIONS, repeat=d):
            candidates.append(seq)
    return candidates


@dataclass(frozen=True)
class SymbolicProbeResult:
    """Outcome of symbolic irreducibility probe for a recipe."""

    recipe: tuple[str, ...]
    category: str
    is_exact_depth3_irreducible: bool
    matching_depth_le_2_candidates: list[tuple[str, ...]]
    probe_examples_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_symbolic_probe(
    panel: Sequence[tuple[str, ...]],
    category: str,
    n_probe: int = 100,
    seed: int = 777,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> list[SymbolicProbeResult]:
    """Symbolically test candidate recipes against all depth <= 2 candidates."""
    all_depth_le_2: list[tuple[str, ...]] = []
    for d in (1, 2):
        for cand_seq in itertools.product(BANK_OPERATIONS, repeat=d):
            all_depth_le_2.append(cand_seq)

    probe_results: list[SymbolicProbeResult] = []

    for recipe in panel:
        rng = random.Random(seed + len(recipe) * 1000 + hash(recipe) % 10000)
        examples: list[Example] = []
        attempts = 0
        max_attempts = n_probe * 100

        while len(examples) < n_probe and attempts < max_attempts:
            attempts += 1
            length = rng.randint(6, 10)
            if "BIND" in recipe and length % 2 != 0:
                continue

            seq = tuple(rng.randrange(vocab_size) for _ in range(length))
            cur = seq
            valid = True
            steps: list[ProgramStep] = []

            for op_name in recipe:
                op_def = get_operation(op_name)
                if not op_def.is_valid_for_length(len(cur)):
                    valid = False
                    break
                try:
                    params = op_def.sample_params(rng, cur, vocab_size)
                    cur = op_def.apply(cur, vocab_size, params)
                    steps.append(ProgramStep(op_name, params))
                except (ValueError, KeyError):
                    valid = False
                    break

            if not valid:
                continue

            prog = Program(steps=tuple(steps))
            res = run_program(prog, seq, vocab_size)
            ex = Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="symbolic_probe",
                split="probe",
                vocab_size=vocab_size,
                task_spec=TaskSpec.from_program(prog),
            )
            examples.append(ex)

        if len(examples) < n_probe:
            raise RuntimeError(
                f"Failed to generate {n_probe} probe examples for recipe {recipe}."
            )

        # Probe against all depth <= 2 candidates
        matching_cands: list[tuple[str, ...]] = []
        for cand in all_depth_le_2:
            matches = 0
            cand_valid = True
            for ex in examples:
                cand_cur = ex.input_tokens
                step_failed = False
                spec_steps = ex.task_spec.steps if ex.task_spec is not None else ()

                for s_idx, op_name in enumerate(cand):
                    op_def = get_operation(op_name)
                    if not op_def.is_valid_for_length(len(cand_cur)):
                        step_failed = True
                        break

                    req_args = op_def.required_argument_names
                    resolved_args: dict[str, Any] = {}
                    if req_args:
                        if s_idx < len(spec_steps) and all(
                            k in spec_steps[s_idx].arguments for k in req_args
                        ):
                            resolved_args = {
                                k: spec_steps[s_idx].arguments[k] for k in req_args
                            }
                        else:
                            for step in spec_steps:
                                if all(k in step.arguments for k in req_args):
                                    resolved_args = {
                                        k: step.arguments[k] for k in req_args
                                    }
                                    break
                        if not resolved_args:
                            step_failed = True
                            break

                    if op_name == "SELECT" and any(
                        idx >= len(cand_cur) for idx in resolved_args.get("indices", ())
                    ):
                        step_failed = True
                        break

                    try:
                        cand_cur = op_def.apply(cand_cur, vocab_size, resolved_args)
                    except (ValueError, KeyError, IndexError):
                        step_failed = True
                        break

                if step_failed:
                    cand_valid = False
                    break
                if cand_cur == ex.target_tokens:
                    matches += 1

            if cand_valid and matches == len(examples):
                matching_cands.append(cand)

        is_irreducible = len(recipe) == 3 and len(matching_cands) == 0
        probe_results.append(
            SymbolicProbeResult(
                recipe=recipe,
                category=category,
                is_exact_depth3_irreducible=is_irreducible,
                matching_depth_le_2_candidates=matching_cands,
                probe_examples_evaluated=len(examples),
            )
        )

    return probe_results


def generate_benchmark_split(
    recipe: tuple[str, ...],
    n_support: int = DEFAULT_SUPPORT_N,
    n_eval: int = DEFAULT_EVAL_N,
    data_seed: int = 101,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[list[Example], list[Example]]:
    """Generate disjoint support and held-out evaluation examples."""
    rng = random.Random(data_seed * 10007 + hash(recipe) % 1000003)
    total_needed = n_support + n_eval
    examples: list[Example] = []
    attempts = 0
    max_attempts = total_needed * 100

    while len(examples) < total_needed and attempts < max_attempts:
        attempts += 1
        length = rng.randint(6, 10)
        if "BIND" in recipe and length % 2 != 0:
            continue

        seq = tuple(rng.randrange(vocab_size) for _ in range(length))
        cur = seq
        valid = True
        steps: list[ProgramStep] = []

        for op_name in recipe:
            op_def = get_operation(op_name)
            if not op_def.is_valid_for_length(len(cur)):
                valid = False
                break
            try:
                params = op_def.sample_params(rng, cur, vocab_size)
                cur = op_def.apply(cur, vocab_size, params)
                steps.append(ProgramStep(op_name, params))
            except (ValueError, KeyError):
                valid = False
                break

        if not valid:
            continue

        prog = Program(steps=tuple(steps))
        res = run_program(prog, seq, vocab_size)
        split = "support" if len(examples) < n_support else "held_out"
        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=prog,
            operation_graph=res.graph,
            category="nrq005_composition",
            split=split,
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
        )
        examples.append(ex)

    if len(examples) < total_needed:
        raise RuntimeError(
            f"Failed to generate {total_needed} examples for recipe {recipe}."
        )

    return examples[:n_support], examples[n_support:]


def exhaustive_lawful_search(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    adaptation_examples: Sequence[Example],
    candidate_space: Sequence[tuple[str, ...]] | None = None,
) -> SearchResult:
    """Exhaustive lawful search over all candidate operation sequences.

    Evaluates every structurally valid candidate among all 584 candidates
    without beam pruning. Selects candidate with highest support exact match
    (tie-breaking: shortest depth, lowest cross-entropy loss).
    """
    candidates = candidate_space or enumerate_all_depth_le_3_candidates()
    start_time = time.perf_counter()

    best_candidate: tuple[str, ...] | None = None
    best_score: tuple[float, int, float] = (-1.0, 0, -float("inf"))
    best_metrics: tuple[float, float] = (0.0, float("inf"))

    candidates_evaluated = 0
    candidates_pruned = 0

    with torch.no_grad():
        for cand in candidates:
            if is_candidate_structurally_valid(
                cand, adaptation_examples, must_match_target_length=True
            ):
                candidates_evaluated += 1
                em, loss = _evaluate_candidate_on_adaptation(
                    core, bank, op_to_id, cand, adaptation_examples
                )
                score = (em, -len(cand), -loss)
                if score > best_score:
                    best_score = score
                    best_candidate = cand
                    best_metrics = (em, loss)
            else:
                candidates_pruned += 1

    if best_candidate is None:
        raise RuntimeError(
            "Exhaustive search failed to discover any structurally valid candidate recipe."
        )

    elapsed = time.perf_counter() - start_time
    recipe_name = "->".join(best_candidate)
    template_calls = resolve_candidate_calls(best_candidate, adaptation_examples[:1])[0]
    recovered_recipe = CompositionRecipe(
        name=recipe_name,
        steps=template_calls,
        metadata={
            "exact_match_adapt": best_metrics[0],
            "loss_adapt": best_metrics[1],
            "depth": len(best_candidate),
        },
    )

    return SearchResult(
        recovered_recipe=recovered_recipe,
        candidate_operations=best_candidate,
        exact_match_adapt=best_metrics[0],
        loss_adapt=best_metrics[1],
        candidates_evaluated=candidates_evaluated,
        candidates_pruned=candidates_pruned,
        search_time_seconds=elapsed,
        search_metadata={
            "max_depth": 3,
            "best_score": best_score,
            "search_type": "exhaustive_lawful",
        },
    )


@dataclass(frozen=True)
class RecipeEvaluationRecord:
    """Detailed evaluation metrics for one recipe under one bundle and data seed."""

    recipe: tuple[str, ...]
    panel_type: str
    bundle_seed: int
    data_seed: int

    # Oracle Metrics
    oracle_em: float
    oracle_time_seconds: float

    # Beam Search Metrics
    beam_recovered_recipe: tuple[str, ...]
    beam_exact_recovery: bool
    beam_functional_em: float
    beam_agreement_with_oracle: float
    beam_candidates_evaluated: int
    beam_candidates_pruned: int
    beam_time_seconds: float
    beam_peak_memory_bytes: int

    # Exhaustive Search Metrics
    exhaustive_recovered_recipe: tuple[str, ...]
    exhaustive_exact_recovery: bool
    exhaustive_functional_em: float
    exhaustive_agreement_with_oracle: float
    exhaustive_candidates_evaluated: int
    exhaustive_candidates_pruned: int
    exhaustive_time_seconds: float
    exhaustive_peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeamSensitivityPoint:
    """Performance summary at a specific beam width."""

    beam_width: int
    mean_recipe_recovery: float
    mean_functional_em: float
    mean_candidates_evaluated: float
    mean_candidates_pruned: float
    mean_time_seconds: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NRQ005BenchmarkReport:
    """Comprehensive artifact report for Task NRQ-005."""

    task_id: str = "NRQ-005"
    task_name: str = "Missing-Artifact-Robust Exact-Depth-3 Irreducible Composition Benchmark"
    date: str = "2026-09-13"
    protocol_type: str = "INDEPENDENT_REPLACEMENT_PROTOCOL"
    seed0_exclusion_reason: str = "PRE_FIXED_ARTIFACT_LOSS_BUNDLE_LOSS"

    bundle_seeds_evaluated: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    data_seeds_evaluated: list[int] = field(default_factory=list)
    support_n: int = DEFAULT_SUPPORT_N
    eval_n: int = DEFAULT_EVAL_N

    # Symbolic Probe
    symbolic_probe_results: list[dict[str, Any]] = field(default_factory=list)

    # Gate Status
    oracle_floor_threshold: float = ORACLE_FLOOR_THRESHOLD
    oracle_floor_passed: bool = False

    # Aggregate Metrics (Irreducible Panel)
    mean_oracle_em_irreducible: float = 0.0
    mean_beam_em_irreducible: float = 0.0
    mean_exhaustive_em_irreducible: float = 0.0
    mean_beam_recovery_irreducible: float = 0.0
    mean_exhaustive_recovery_irreducible: float = 0.0

    # Aggregate Metrics (Controls)
    mean_oracle_em_reducible_depth3: float = 0.0
    mean_exhaustive_em_reducible_depth3: float = 0.0
    mean_oracle_em_depth2: float = 0.0
    mean_exhaustive_em_depth2: float = 0.0

    # Sensitivity & Resources
    beam_budget_sensitivity: list[dict[str, Any]] = field(default_factory=list)
    per_bundle_summary: dict[str, Any] = field(default_factory=dict)

    # Architectural Decision
    adr0162_decision: str = ""
    decision_rationale: str = ""

    # All granular records
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_bundle(
    seed: int,
    bundle_base: Path,
    device: str = "cpu",
) -> tuple[Any, PrimitiveBank, dict[str, int]]:
    """Load reconstructed Phase A.1 core and primitive bank."""
    tokens = build_a1_b004_tokens(DEFAULT_VOCAB_SIZE)
    arch_cfg = SharedEncoderArchitectureConfig(
        seed=seed,
        vocab_size=DEFAULT_VOCAB_SIZE,
        sequence_length_range=(6, 10),
        operation_names=PARAMETERIZED_OPERATION_NAMES,
        group_size=4,
        d_operator=32,
        n_operator_head=4,
        d_operator_ff=64,
        arg_dim=16,
        max_sequence_length=32,
    )
    arch = build_shared_encoder_architecture(arch_cfg, tokens=tokens)
    core_file = bundle_base / f"seed_{seed}" / "core" / "shared_encoder.pt"
    arch.core.model.load_state_dict(
        torch.load(core_file, map_location=device, weights_only=True)
    )
    core = arch.core

    u_config = UnifiedBenchmarkConfig(
        seed=seed,
        vocab_size=DEFAULT_VOCAB_SIZE,
        sequence_length_range=(6, 10),
        group_size=4,
        d_operator=32,
        n_operator_head=4,
        d_operator_ff=64,
        arg_dim=16,
        max_sequence_length=32,
    )
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank_file = bundle_base / f"seed_{seed}" / "primitives" / "primitive_bank.pt"
    bank.load_state_dict(
        torch.load(bank_file, map_location=device, weights_only=True)
    )

    return core, bank, op_to_id


def evaluate_single_recipe(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    recipe: tuple[str, ...],
    panel_type: str,
    bundle_seed: int,
    data_seed: int,
    support_examples: list[Example],
    test_examples: list[Example],
) -> RecipeEvaluationRecord:
    """Evaluate oracle, beam search, and exhaustive search on identical inputs."""
    with torch.no_grad():
        # 1. Oracle execution
        t0 = time.perf_counter()
        or_logits = execute_composition_recipe(
            core, bank, op_to_id, test_examples, candidate_operations=recipe
        )
        or_time = time.perf_counter() - t0
        or_preds = or_logits.argmax(dim=-1)
        or_matches = sum(
            1
            for i, ex in enumerate(test_examples)
            if tuple(or_preds[i, : len(ex.target_tokens)].tolist()) == ex.target_tokens
        )
        or_em = or_matches / len(test_examples)

        # 2. Existing Beam Search (beam_width=16, max_depth=3)
        tracemalloc.start()
        b_res = search_composition_recipe(
            core, bank, op_to_id, support_examples, max_depth=3, beam_width=16
        )
        _, b_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        b_logits = execute_composition_recipe(
            core, bank, op_to_id, test_examples, candidate_operations=b_res.candidate_operations
        )
        b_preds = b_logits.argmax(dim=-1)
        b_matches = sum(
            1
            for i, ex in enumerate(test_examples)
            if tuple(b_preds[i, : len(ex.target_tokens)].tolist()) == ex.target_tokens
        )
        b_em = b_matches / len(test_examples)
        b_agr = sum(
            1
            for i, ex in enumerate(test_examples)
            if tuple(b_preds[i, : len(ex.target_tokens)].tolist())
            == tuple(or_preds[i, : len(ex.target_tokens)].tolist())
        ) / len(test_examples)
        b_exact_recovery = b_res.candidate_operations == recipe

        # 3. Exhaustive Lawful Search over all 584 candidates
        tracemalloc.start()
        e_res = exhaustive_lawful_search(core, bank, op_to_id, support_examples)
        _, e_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        e_logits = execute_composition_recipe(
            core, bank, op_to_id, test_examples, candidate_operations=e_res.candidate_operations
        )
        e_preds = e_logits.argmax(dim=-1)
        e_matches = sum(
            1
            for i, ex in enumerate(test_examples)
            if tuple(e_preds[i, : len(ex.target_tokens)].tolist()) == ex.target_tokens
        )
        e_em = e_matches / len(test_examples)
        e_agr = sum(
            1
            for i, ex in enumerate(test_examples)
            if tuple(e_preds[i, : len(ex.target_tokens)].tolist())
            == tuple(or_preds[i, : len(ex.target_tokens)].tolist())
        ) / len(test_examples)
        e_exact_recovery = e_res.candidate_operations == recipe

    return RecipeEvaluationRecord(
        recipe=recipe,
        panel_type=panel_type,
        bundle_seed=bundle_seed,
        data_seed=data_seed,
        oracle_em=or_em,
        oracle_time_seconds=or_time,
        beam_recovered_recipe=b_res.candidate_operations,
        beam_exact_recovery=b_exact_recovery,
        beam_functional_em=b_em,
        beam_agreement_with_oracle=b_agr,
        beam_candidates_evaluated=b_res.candidates_evaluated,
        beam_candidates_pruned=b_res.candidates_pruned,
        beam_time_seconds=b_res.search_time_seconds,
        beam_peak_memory_bytes=b_peak_mem,
        exhaustive_recovered_recipe=e_res.candidate_operations,
        exhaustive_exact_recovery=e_exact_recovery,
        exhaustive_functional_em=e_em,
        exhaustive_agreement_with_oracle=e_agr,
        exhaustive_candidates_evaluated=e_res.candidates_evaluated,
        exhaustive_candidates_pruned=e_res.candidates_pruned,
        exhaustive_time_seconds=e_res.search_time_seconds,
        exhaustive_peak_memory_bytes=e_peak_mem,
    )


def run_beam_sensitivity_sweep(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    recipes: Sequence[tuple[str, ...]],
    widths: Sequence[int] = BEAM_SENSITIVITY_WIDTHS,
    n_support: int = DEFAULT_SUPPORT_N,
    n_eval: int = DEFAULT_EVAL_N,
    data_seed: int = 101,
) -> list[BeamSensitivityPoint]:
    """Sweep beam width on irreducible recipes and record performance sensitivity."""
    sweep_results: list[BeamSensitivityPoint] = []

    # Pre-generate datasets for each recipe
    datasets: dict[tuple[str, ...], tuple[list[Example], list[Example]]] = {}
    for r in recipes:
        datasets[r] = generate_benchmark_split(
            r, n_support=n_support, n_eval=n_eval, data_seed=data_seed
        )

    for bw in widths:
        rec_matches = 0
        em_sum = 0.0
        eval_sum = 0
        pruned_sum = 0
        time_sum = 0.0
        peak_mem = 0

        for r in recipes:
            support, test = datasets[r]
            tracemalloc.start()
            res = search_composition_recipe(
                core, bank, op_to_id, support, max_depth=3, beam_width=bw
            )
            _, mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mem = max(peak_mem, mem)
            if res.candidate_operations == r:
                rec_matches += 1

            with torch.no_grad():
                logits = execute_composition_recipe(
                    core, bank, op_to_id, test, candidate_operations=res.candidate_operations
                )
                preds = logits.argmax(dim=-1)
                matches = sum(
                    1
                    for i, ex in enumerate(test)
                    if tuple(preds[i, : len(ex.target_tokens)].tolist()) == ex.target_tokens
                )
                em_sum += matches / len(test)

            eval_sum += res.candidates_evaluated
            pruned_sum += res.candidates_pruned
            time_sum += res.search_time_seconds

        n_r = len(recipes)
        sweep_results.append(
            BeamSensitivityPoint(
                beam_width=bw,
                mean_recipe_recovery=rec_matches / n_r,
                mean_functional_em=em_sum / n_r,
                mean_candidates_evaluated=eval_sum / n_r,
                mean_candidates_pruned=pruned_sum / n_r,
                mean_time_seconds=time_sum / n_r,
                peak_memory_bytes=peak_mem,
            )
        )

    return sweep_results


def run_nrq005_benchmark(
    bundle_base: Path = Path("runs/nrq004_reconstructed_bundles"),
    bundle_seeds: Sequence[int] = (1, 2, 3, 4),
    data_seeds: Sequence[int] = DEFAULT_DATA_SEEDS,
    support_n: int = DEFAULT_SUPPORT_N,
    eval_n: int = DEFAULT_EVAL_N,
    fast_mode: bool = False,
) -> NRQ005BenchmarkReport:
    """Execute complete NRQ-005 benchmark suite."""
    print("=== Step 1: Symbolic Probe Execution ===")
    probe_irred = run_symbolic_probe(
        EXACT_DEPTH3_IRREDUCIBLE_PANEL, category="irreducible_depth3", n_probe=100
    )
    probe_red = run_symbolic_probe(
        REDUCIBLE_DEPTH3_PANEL, category="reducible_depth3", n_probe=100
    )
    probe_d2 = run_symbolic_probe(
        DEPTH2_CANONICAL_PANEL, category="canonical_depth2", n_probe=100
    )
    all_probes = probe_irred + probe_red + probe_d2

    # Verify symbolic conditions before proceeding to models
    for p in probe_irred:
        if not p.is_exact_depth3_irreducible:
            raise RuntimeError(
                f"Symbolic probe failed: recipe {p.recipe} matched depth <= 2: "
                f"{p.matching_depth_le_2_candidates}"
            )
    for p in probe_red:
        if len(p.matching_depth_le_2_candidates) == 0:
            raise RuntimeError(
                f"Symbolic probe failed: reducible recipe {p.recipe} has no depth <= 2 matches."
            )
    print("Symbolic probe passed: exact-depth-3 panel is strictly irreducible to depth <= 2.")

    eval_data_seeds = data_seeds[:2] if fast_mode else data_seeds
    records: list[RecipeEvaluationRecord] = []

    print(
        f"=== Step 2: Comparative Search Evaluation ({len(bundle_seeds)} bundles x "
        f"{len(eval_data_seeds)} data seeds) ==="
    )

    for b_seed in bundle_seeds:
        print(f"--- Loading Bundle Seed {b_seed} ---")
        core, bank, op_to_id = _load_bundle(b_seed, bundle_base=bundle_base)

        for d_seed in eval_data_seeds:
            # 1. Irreducible Exact-Depth-3 Panel
            for r_irred in EXACT_DEPTH3_IRREDUCIBLE_PANEL:
                supp, test = generate_benchmark_split(
                    r_irred, n_support=support_n, n_eval=eval_n, data_seed=d_seed
                )
                rec = evaluate_single_recipe(
                    core, bank, op_to_id, r_irred, "irreducible_depth3", b_seed, d_seed, supp, test
                )
                records.append(rec)

            # 2. Reducible Depth-3 Controls
            for r_red in REDUCIBLE_DEPTH3_PANEL:
                supp, test = generate_benchmark_split(
                    r_red, n_support=support_n, n_eval=eval_n, data_seed=d_seed
                )
                rec = evaluate_single_recipe(
                    core, bank, op_to_id, r_red, "reducible_depth3", b_seed, d_seed, supp, test
                )
                records.append(rec)

            # 3. Canonical Depth-2 Controls
            for r_d2 in DEPTH2_CANONICAL_PANEL:
                supp, test = generate_benchmark_split(
                    r_d2, n_support=support_n, n_eval=eval_n, data_seed=d_seed
                )
                rec = evaluate_single_recipe(
                    core, bank, op_to_id, r_d2, "canonical_depth2", b_seed, d_seed, supp, test
                )
                records.append(rec)

    # Step 3: Beam Sensitivity Sweep (using bundle seed 1, first data seed)
    print("=== Step 3: Beam-Budget Sensitivity Sweep ===")
    core_s1, bank_s1, op_to_id_s1 = _load_bundle(1, bundle_base=bundle_base)
    sensitivity_points = run_beam_sensitivity_sweep(
        core_s1,
        bank_s1,
        op_to_id_s1,
        EXACT_DEPTH3_IRREDUCIBLE_PANEL,
        widths=BEAM_SENSITIVITY_WIDTHS,
        n_support=support_n,
        n_eval=eval_n,
        data_seed=eval_data_seeds[0],
    )

    # Step 4: Aggregate Metrics & Gate Evaluation
    irred_records = [r for r in records if r.panel_type == "irreducible_depth3"]
    red_records = [r for r in records if r.panel_type == "reducible_depth3"]
    d2_records = [r for r in records if r.panel_type == "canonical_depth2"]

    mean_or_irred = sum(r.oracle_em for r in irred_records) / max(1, len(irred_records))
    mean_beam_irred = sum(r.beam_functional_em for r in irred_records) / max(1, len(irred_records))
    mean_exh_irred = (
        sum(r.exhaustive_functional_em for r in irred_records) / max(1, len(irred_records))
    )
    mean_beam_rec_irred = (
        sum(1.0 if r.beam_exact_recovery else 0.0 for r in irred_records)
        / max(1, len(irred_records))
    )
    mean_exh_rec_irred = (
        sum(1.0 if r.exhaustive_exact_recovery else 0.0 for r in irred_records)
        / max(1, len(irred_records))
    )

    mean_or_red = sum(r.oracle_em for r in red_records) / max(1, len(red_records))
    mean_exh_red = (
        sum(r.exhaustive_functional_em for r in red_records) / max(1, len(red_records))
    )

    mean_or_d2 = sum(r.oracle_em for r in d2_records) / max(1, len(d2_records))
    mean_exh_d2 = sum(r.exhaustive_functional_em for r in d2_records) / max(1, len(d2_records))

    oracle_floor_passed = mean_or_irred >= ORACLE_FLOOR_THRESHOLD

    # Per-bundle summary
    per_bundle: dict[str, Any] = {}
    for b_seed in bundle_seeds:
        b_irred = [r for r in irred_records if r.bundle_seed == b_seed]
        per_bundle[f"seed_{b_seed}"] = {
            "mean_oracle_em": sum(r.oracle_em for r in b_irred) / len(b_irred),
            "mean_beam_em": sum(r.beam_functional_em for r in b_irred) / len(b_irred),
            "mean_exhaustive_em": (
                sum(r.exhaustive_functional_em for r in b_irred) / len(b_irred)
            ),
            "mean_exhaustive_recovery": (
                sum(1.0 if r.exhaustive_exact_recovery else 0.0 for r in b_irred)
                / len(b_irred)
            ),
        }

    # ADR-0162 Closure Evaluation
    if not oracle_floor_passed:
        adr_decision = "DEPTH_COMPOSITION_EXECUTION_FAILURE"
        rationale = (
            f"Mean oracle EM on exact-depth-3 irreducible panel collapsed to "
            f"{mean_or_irred:.4f} < {ORACLE_FLOOR_THRESHOLD}. Depth-composition execution "
            f"failed; search results are strictly barred from interpretation."
        )
    else:
        # Check baseline performance across all bundle and data seeds
        baseline_all_ge_95 = all(
            b_info["mean_exhaustive_em"] >= CLOSURE_CONFIRMATION_THRESHOLD
            for b_info in per_bundle.values()
        )
        if mean_beam_irred >= 0.95 and mean_exh_irred < 0.95:
            adr_decision = "REJECT_ADR0162_CLOSURE"
            rationale = (
                f"APC side recovered >= 0.95 (EM = {mean_beam_irred:.4f}) while strongest "
                f"deterministic baseline failed (< 0.95, EM = {mean_exh_irred:.4f}). "
                f"Falsifies ADR-0162's closure."
            )
        elif baseline_all_ge_95:
            adr_decision = "EMPIRICALLY_SUPPORT_CLOSURE_DEPTH_LE_3"
            rationale = (
                f"Oracle floor passed (mean Oracle EM = {mean_or_irred:.4f} >= 0.85). "
                f"Strongest deterministic baseline (exhaustive lawful search over all 584 "
                f"candidates) achieves near-ceiling functional EM (mean = {mean_exh_irred:.4f} "
                f">= 0.95, and >= 0.95 across every tested bundle seed). "
                f"Confirms that exact-depth-3 compositions of the current 8-primitive registry "
                f"are within the tractable reach of lawful deterministic search, empirically "
                f"supporting ADR-0162's Bounded-Resource Corollary restricted to depth <= 3."
            )
        else:
            adr_decision = "BASELINE_SUBCEILING_DEPTH_LE_3"
            rationale = (
                f"Strongest deterministic baseline achieved mean EM = {mean_exh_irred:.4f}, "
                f"failing the >= 0.95 ceiling across all bundles."
            )

    report = NRQ005BenchmarkReport(
        bundle_seeds_evaluated=list(bundle_seeds),
        data_seeds_evaluated=list(eval_data_seeds),
        support_n=support_n,
        eval_n=eval_n,
        symbolic_probe_results=[p.to_dict() for p in all_probes],
        oracle_floor_passed=oracle_floor_passed,
        mean_oracle_em_irreducible=mean_or_irred,
        mean_beam_em_irreducible=mean_beam_irred,
        mean_exhaustive_em_irreducible=mean_exh_irred,
        mean_beam_recovery_irreducible=mean_beam_rec_irred,
        mean_exhaustive_recovery_irreducible=mean_exh_rec_irred,
        mean_oracle_em_reducible_depth3=mean_or_red,
        mean_exhaustive_em_reducible_depth3=mean_exh_red,
        mean_oracle_em_depth2=mean_or_d2,
        mean_exhaustive_em_depth2=mean_exh_d2,
        beam_budget_sensitivity=[p.to_dict() for p in sensitivity_points],
        per_bundle_summary=per_bundle,
        adr0162_decision=adr_decision,
        decision_rationale=rationale,
        records=[r.to_dict() for r in records],
    )

    return report


def save_nrq005_artifacts(
    report: NRQ005BenchmarkReport,
    repo_root: Path = Path("."),
) -> tuple[Path, Path]:
    """Save NRQ-005 JSON review record and run summary."""
    research_dir = repo_root / "docs" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    review_file = research_dir / "NRQ005_REVIEW_RECORD.json"
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    runs_dir = repo_root / "runs" / "nrq005_exact_depth3_benchmark"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = runs_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return review_file, summary_file
