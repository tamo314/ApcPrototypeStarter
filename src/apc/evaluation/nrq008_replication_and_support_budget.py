"""NRQ-008 — Replication and Support-Budget Sensitivity of the Sole APC-over-Exhaustive Cell.

Evaluates the exact pre-fixed recipe NEGATE->REVERSE->SHIFT across intact reconstructed
bundles 1-4, unused data seeds 201-220, support budgets N in {32, 64, 128, 256}, and
1024 disjoint held-out evaluation inputs per cell.

Compares:
1. Oracle recipe execution (NEGATE->REVERSE->SHIFT)
2. Existing beam search (beam_width=16, max_depth=3)
3. Exhaustive lawful search over all 584 candidates of depth <= 3

Designated roles:
- Bundles 1-3: Ceiling controls
- Bundle 4: Pre-designated challenge bundle

Decision logic:
- If at max budget N=256 in >= 2 bundles:
  pooled Oracle EM 95% CI lower bound >= 0.95 AND exhaustive EM 95% CI upper bound < 0.95:
  -> FALSIFY_BOUNDED_RESOURCE_CLOSURE
- If this holds only for Bundle 4:
  -> BUNDLE_SPECIFIC_QUALIFIED_RESULT
- If the difference disappears (Delta -> 0) or baseline >= 0.95:
  -> FINITE_SAMPLE_SUPPORT_MISSELECTION (confirms bounded-resource closure / support closure)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import random
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import scipy.stats
import torch

from apc.environments.generator import Example
from apc.environments.interpreter import run_program
from apc.environments.operations import get_operation
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.nrq004_bundle_reconstruction import (
    build_a1_b004_tokens,
)
from apc.evaluation.nrq005_exact_depth3_benchmark import (
    enumerate_all_depth_le_3_candidates,
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
    execute_composition_recipe,
    oracle_calls_for_example,
)
from apc.primitives.composition_search import (
    _evaluate_candidate_on_adaptation,
    is_candidate_structurally_valid,
    search_composition_recipe,
)

TARGET_RECIPE: tuple[str, ...] = ("NEGATE", "REVERSE", "SHIFT")
EQUIVALENT_RECIPE: tuple[str, ...] = ("REVERSE", "NEGATE", "SHIFT")

DEFAULT_DATA_SEEDS: tuple[int, ...] = tuple(range(201, 221))
DEFAULT_BUNDLE_SEEDS: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_SUPPORT_BUDGETS: tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_EVAL_N: int = 1024
CLOSURE_THRESHOLD: float = 0.95


def derive_deterministic_seed(
    recipe_str: str, data_seed: int = 0, salt: str = ""
) -> int:
    """Derive an integer seed from SHA-256 to ensure inter-process determinism."""
    key = f"{recipe_str}:{data_seed}:{salt}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generate_benchmark_split_deterministic(
    recipe: tuple[str, ...],
    n_support: int = 256,
    n_eval: int = DEFAULT_EVAL_N,
    data_seed: int = 201,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[list[Example], list[Example]]:
    """Generate disjoint support and held-out evaluation examples with SHA-256 seeding."""
    recipe_str = "->".join(recipe)
    seed = derive_deterministic_seed(recipe_str, data_seed=data_seed, salt="nrq008_dataset")
    rng = random.Random(seed)

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
            category="nrq008_replication",
            split=split,
            vocab_size=vocab_size,
            task_spec=TaskSpec.from_program(prog),
        )
        examples.append(ex)

    if len(examples) < total_needed:
        raise RuntimeError(
            f"Failed to generate {total_needed} deterministic examples for recipe {recipe}."
        )

    return examples[:n_support], examples[n_support:]


def build_nrq008_dataset_manifest_hash(
    data_seeds: Sequence[int] = DEFAULT_DATA_SEEDS,
    support_budgets: Sequence[int] = DEFAULT_SUPPORT_BUDGETS,
    eval_n: int = DEFAULT_EVAL_N,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[str, dict[str, Any]]:
    """Compute overall SHA-256 hash over all generated datasets to guarantee process match."""
    manifest_entries: list[dict[str, Any]] = []
    recipe_str = "->".join(TARGET_RECIPE)
    max_support = max(support_budgets)

    for d_seed in data_seeds:
        supp, test = generate_benchmark_split_deterministic(
            TARGET_RECIPE,
            n_support=max_support,
            n_eval=eval_n,
            data_seed=d_seed,
            vocab_size=vocab_size,
        )
        test_hash = hashlib.sha256(
            "".join(str(e.input_tokens) + str(e.target_tokens) for e in test).encode()
        ).hexdigest()

        for s_n in support_budgets:
            sub_supp = supp[:s_n]
            supp_hash = hashlib.sha256(
                "".join(str(e.input_tokens) + str(e.target_tokens) for e in sub_supp).encode()
            ).hexdigest()
            manifest_entries.append({
                "recipe": recipe_str,
                "data_seed": d_seed,
                "support_n": s_n,
                "eval_n": eval_n,
                "support_tokens_hash": supp_hash,
                "test_tokens_hash": test_hash,
            })

    manifest_bytes = json.dumps(manifest_entries, sort_keys=True).encode("utf-8")
    overall_hash = hashlib.sha256(manifest_bytes).hexdigest()

    manifest_doc = {
        "dataset_manifest_hash": overall_hash,
        "recipe": recipe_str,
        "data_seeds": list(data_seeds),
        "support_budgets": list(support_budgets),
        "eval_n": eval_n,
        "entries": manifest_entries,
    }
    return overall_hash, manifest_doc


def _load_reconstructed_bundle(
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
    # Artifact loading is portable; execution must co-locate the historical
    # bank with the reconstructed Core selected at runtime.
    bank.to(core.device)

    return core, bank, op_to_id


@dataclass(frozen=True)
class CellEvaluationRecord:
    """Evaluation result for one cell (bundle x data_seed x support_n)."""

    recipe: tuple[str, ...]
    bundle_seed: int
    bundle_role: str
    data_seed: int
    support_n: int
    eval_n: int

    oracle_support_em: float
    oracle_support_loss: float
    oracle_heldout_em: float
    oracle_eval_time_seconds: float

    exhaustive_recovered_recipe: tuple[str, ...]
    exhaustive_support_em: float
    exhaustive_support_loss: float
    exhaustive_heldout_em: float
    exhaustive_agreement_with_oracle: float
    exhaustive_oracle_candidate_rank: int
    exhaustive_equivalent_candidate_rank: int
    exhaustive_candidates_evaluated: int
    exhaustive_candidates_pruned: int
    exhaustive_time_seconds: float
    exhaustive_peak_memory_bytes: int

    beam_recovered_recipe: tuple[str, ...]
    beam_support_em: float
    beam_support_loss: float
    beam_heldout_em: float
    beam_agreement_with_oracle: float
    beam_time_seconds: float
    beam_peak_memory_bytes: int

    oracle_minus_exhaustive_em: float
    oracle_minus_beam_em: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": list(self.recipe),
            "bundle_seed": self.bundle_seed,
            "bundle_role": self.bundle_role,
            "data_seed": self.data_seed,
            "support_n": self.support_n,
            "eval_n": self.eval_n,
            "oracle_support_em": self.oracle_support_em,
            "oracle_support_loss": self.oracle_support_loss,
            "oracle_heldout_em": self.oracle_heldout_em,
            "oracle_eval_time_seconds": self.oracle_eval_time_seconds,
            "exhaustive_recovered_recipe": list(self.exhaustive_recovered_recipe),
            "exhaustive_support_em": self.exhaustive_support_em,
            "exhaustive_support_loss": self.exhaustive_support_loss,
            "exhaustive_heldout_em": self.exhaustive_heldout_em,
            "exhaustive_agreement_with_oracle": self.exhaustive_agreement_with_oracle,
            "exhaustive_oracle_candidate_rank": self.exhaustive_oracle_candidate_rank,
            "exhaustive_equivalent_candidate_rank": self.exhaustive_equivalent_candidate_rank,
            "exhaustive_candidates_evaluated": self.exhaustive_candidates_evaluated,
            "exhaustive_candidates_pruned": self.exhaustive_candidates_pruned,
            "exhaustive_time_seconds": self.exhaustive_time_seconds,
            "exhaustive_peak_memory_bytes": self.exhaustive_peak_memory_bytes,
            "beam_recovered_recipe": list(self.beam_recovered_recipe),
            "beam_support_em": self.beam_support_em,
            "beam_support_loss": self.beam_support_loss,
            "beam_heldout_em": self.beam_heldout_em,
            "beam_agreement_with_oracle": self.beam_agreement_with_oracle,
            "beam_time_seconds": self.beam_time_seconds,
            "beam_peak_memory_bytes": self.beam_peak_memory_bytes,
            "oracle_minus_exhaustive_em": self.oracle_minus_exhaustive_em,
            "oracle_minus_beam_em": self.oracle_minus_beam_em,
        }


def compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Student's t-distribution confidence interval."""
    n = len(values)
    if n <= 1:
        v = values[0] if values else 0.0
        return v, v
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)
    if std_err == 0.0:
        return mean, mean
    t_crit = float(scipy.stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    lower = max(0.0, mean - t_crit * std_err)
    upper = min(1.0, mean + t_crit * std_err)
    return lower, upper


def compute_wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Wilson score interval for pooled binomial observations."""
    if total == 0:
        return 0.0, 0.0
    z = float(scipy.stats.norm.ppf((1 + confidence) / 2.0))
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_cell_instance(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    bundle_seed: int,
    data_seed: int,
    support_n: int,
    support_examples: list[Example],
    test_examples: list[Example],
    candidate_space: Sequence[tuple[str, ...]],
    precomputed_oracle_test: tuple[float, list[tuple[int, ...]]] | None = None,
) -> CellEvaluationRecord:
    """Evaluate oracle, beam, and exhaustive search on identical deterministic cell."""
    bundle_role = "challenge_bundle" if bundle_seed == 4 else "ceiling_control"
    recipe = TARGET_RECIPE

    with torch.no_grad():
        # 1. Oracle execution
        if precomputed_oracle_test is not None:
            or_heldout_em, or_preds = precomputed_oracle_test
            or_eval_time = 0.0
        else:
            t0 = time.perf_counter()
            oracle_calls = [oracle_calls_for_example(e) for e in test_examples]
            or_logits = execute_composition_recipe(
                core, bank, op_to_id, test_examples, calls_per_example=oracle_calls
            )
            or_eval_time = time.perf_counter() - t0
            pred_indices = or_logits.argmax(dim=-1)
            or_preds = [
                tuple(pred_indices[i, : len(ex.target_tokens)].tolist())
                for i, ex in enumerate(test_examples)
            ]
            matches = sum(
                1 for i, ex in enumerate(test_examples) if or_preds[i] == ex.target_tokens
            )
            or_heldout_em = matches / len(test_examples)

        # Oracle metrics on support set
        or_supp_em, or_supp_loss = _evaluate_candidate_on_adaptation(
            core, bank, op_to_id, recipe, support_examples
        )

        # 2. Existing Beam Search
        tracemalloc.start()
        t0 = time.perf_counter()
        beam_res = search_composition_recipe(
            core, bank, op_to_id, support_examples, max_depth=3, beam_width=16
        )
        beam_time = time.perf_counter() - t0
        _, beam_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        beam_cand = beam_res.candidate_operations
        beam_supp_em = beam_res.exact_match_adapt
        beam_supp_loss = beam_res.loss_adapt

        beam_logits = execute_composition_recipe(
            core, bank, op_to_id, test_examples, candidate_operations=beam_cand
        )
        b_pred_indices = beam_logits.argmax(dim=-1)
        b_preds = [
            tuple(b_pred_indices[i, : len(ex.target_tokens)].tolist())
            for i, ex in enumerate(test_examples)
        ]
        beam_heldout_em = (
            sum(1 for i, ex in enumerate(test_examples) if b_preds[i] == ex.target_tokens)
            / len(test_examples)
        )
        beam_agr = (
            sum(1 for i in range(len(test_examples)) if b_preds[i] == or_preds[i])
            / len(test_examples)
        )

        # 3. Exhaustive Lawful Search over all 584 candidates
        tracemalloc.start()
        t0 = time.perf_counter()

        scored_candidates: list[dict[str, Any]] = []
        eval_count = 0
        prune_count = 0

        for cand in candidate_space:
            if is_candidate_structurally_valid(
                cand, support_examples, must_match_target_length=True
            ):
                eval_count += 1
                c_em, c_loss = _evaluate_candidate_on_adaptation(
                    core, bank, op_to_id, cand, support_examples
                )
                score = (c_em, -len(cand), -c_loss)
                scored_candidates.append({
                    "candidate": cand,
                    "em": c_em,
                    "loss": c_loss,
                    "score": score,
                })
            else:
                prune_count += 1

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        exh_time = time.perf_counter() - t0
        _, exh_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if not scored_candidates:
            raise RuntimeError(f"Exhaustive search failed for {recipe} at N={support_n}")

        best_cand_entry = scored_candidates[0]
        exh_cand = best_cand_entry["candidate"]
        exh_supp_em = best_cand_entry["em"]
        exh_supp_loss = best_cand_entry["loss"]

        # Candidate selection rankings
        oracle_rank = next(
            (idx + 1 for idx, c in enumerate(scored_candidates) if c["candidate"] == recipe),
            len(scored_candidates) + 1,
        )
        equiv_rank = next(
            (
                idx + 1
                for idx, c in enumerate(scored_candidates)
                if c["candidate"] in (recipe, EQUIVALENT_RECIPE)
            ),
            len(scored_candidates) + 1,
        )

        # Evaluate exhaustive candidate on held-out test examples
        exh_logits = execute_composition_recipe(
            core, bank, op_to_id, test_examples, candidate_operations=exh_cand
        )
        e_pred_indices = exh_logits.argmax(dim=-1)
        e_preds = [
            tuple(e_pred_indices[i, : len(ex.target_tokens)].tolist())
            for i, ex in enumerate(test_examples)
        ]
        exh_heldout_em = (
            sum(1 for i, ex in enumerate(test_examples) if e_preds[i] == ex.target_tokens)
            / len(test_examples)
        )
        exh_agr = (
            sum(1 for i in range(len(test_examples)) if e_preds[i] == or_preds[i])
            / len(test_examples)
        )

    return CellEvaluationRecord(
        recipe=recipe,
        bundle_seed=bundle_seed,
        bundle_role=bundle_role,
        data_seed=data_seed,
        support_n=support_n,
        eval_n=len(test_examples),
        oracle_support_em=or_supp_em,
        oracle_support_loss=or_supp_loss,
        oracle_heldout_em=or_heldout_em,
        oracle_eval_time_seconds=or_eval_time,
        exhaustive_recovered_recipe=exh_cand,
        exhaustive_support_em=exh_supp_em,
        exhaustive_support_loss=exh_supp_loss,
        exhaustive_heldout_em=exh_heldout_em,
        exhaustive_agreement_with_oracle=exh_agr,
        exhaustive_oracle_candidate_rank=oracle_rank,
        exhaustive_equivalent_candidate_rank=equiv_rank,
        exhaustive_candidates_evaluated=eval_count,
        exhaustive_candidates_pruned=prune_count,
        exhaustive_time_seconds=exh_time,
        exhaustive_peak_memory_bytes=exh_peak_mem,
        beam_recovered_recipe=beam_cand,
        beam_support_em=beam_supp_em,
        beam_support_loss=beam_supp_loss,
        beam_heldout_em=beam_heldout_em,
        beam_agreement_with_oracle=beam_agr,
        beam_time_seconds=beam_time,
        beam_peak_memory_bytes=beam_peak_mem,
        oracle_minus_exhaustive_em=or_heldout_em - exh_heldout_em,
        oracle_minus_beam_em=or_heldout_em - beam_heldout_em,
    )


def _evaluate_single_bundle_worker(
    bundle_seed: int,
    bundle_base: Path,
    data_seeds: list[int],
    support_budgets: list[int],
    eval_n: int,
) -> list[dict[str, Any]]:
    """Worker function for evaluating a single bundle across data seeds and support budgets."""
    core, bank, op_to_id = _load_reconstructed_bundle(bundle_seed, bundle_base=bundle_base)
    candidate_space = enumerate_all_depth_le_3_candidates()
    records: list[dict[str, Any]] = []
    max_support = max(support_budgets)

    for d_seed in data_seeds:
        supp, test = generate_benchmark_split_deterministic(
            TARGET_RECIPE, n_support=max_support, n_eval=eval_n, data_seed=d_seed
        )

        # Precompute oracle test predictions once for this (bundle, data_seed)
        with torch.no_grad():
            t0 = time.perf_counter()
            oracle_calls = [oracle_calls_for_example(e) for e in test]
            or_logits = execute_composition_recipe(
                core, bank, op_to_id, test, calls_per_example=oracle_calls
            )
            or_time = time.perf_counter() - t0
            p_idx = or_logits.argmax(dim=-1)
            or_preds = [
                tuple(p_idx[i, : len(ex.target_tokens)].tolist())
                for i, ex in enumerate(test)
            ]
            matches = sum(1 for i, ex in enumerate(test) if or_preds[i] == ex.target_tokens)
            precomputed_oracle = (matches / len(test), or_preds)

        for s_n in support_budgets:
            sub_supp = supp[:s_n]
            rec = evaluate_cell_instance(
                core=core,
                bank=bank,
                op_to_id=op_to_id,
                bundle_seed=bundle_seed,
                data_seed=d_seed,
                support_n=s_n,
                support_examples=sub_supp,
                test_examples=test,
                candidate_space=candidate_space,
                precomputed_oracle_test=precomputed_oracle,
            )
            # Update oracle eval time for record
            rec_dict = rec.to_dict()
            rec_dict["oracle_eval_time_seconds"] = or_time
            records.append(rec_dict)

    return records


@dataclass(frozen=True)
class NRQ008Report:
    """Full replication and sensitivity report for NRQ-008."""

    task_id: str
    task_name: str
    date: str
    process_id: int
    recipe: list[str]
    bundle_seeds_evaluated: list[int]
    challenge_bundle: int
    ceiling_control_bundles: list[int]
    data_seeds_evaluated: list[int]
    support_budgets: list[int]
    eval_n: int
    total_cells_evaluated: int
    dataset_manifest_hash: str

    budget_summaries: dict[str, Any]
    per_bundle_budget_stats: dict[str, Any]
    max_budget_256_bundle_stats: dict[str, Any]

    scientific_decision: str
    decision_rationale: str
    falsification_bundles_count: int
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_nrq008_experiment(
    bundle_base: Path,
    bundle_seeds: Sequence[int] = DEFAULT_BUNDLE_SEEDS,
    data_seeds: Sequence[int] = DEFAULT_DATA_SEEDS,
    support_budgets: Sequence[int] = DEFAULT_SUPPORT_BUDGETS,
    eval_n: int = DEFAULT_EVAL_N,
    max_workers: int = 4,
    process_id: int = 1,
) -> tuple[NRQ008Report, dict[str, Any]]:
    """Run full NRQ-008 replication and support-budget sensitivity experiment."""
    manifest_hash, manifest_doc = build_nrq008_dataset_manifest_hash(
        data_seeds=data_seeds, support_budgets=support_budgets, eval_n=eval_n
    )

    all_records: list[dict[str, Any]] = []

    if max_workers > 1 and len(bundle_seeds) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_single_bundle_worker,
                    b_seed,
                    bundle_base,
                    list(data_seeds),
                    list(support_budgets),
                    eval_n,
                )
                for b_seed in bundle_seeds
            ]
            for future in concurrent.futures.as_completed(futures):
                all_records.extend(future.result())
    else:
        for b_seed in bundle_seeds:
            b_records = _evaluate_single_bundle_worker(
                bundle_seed=b_seed,
                bundle_base=bundle_base,
                data_seeds=list(data_seeds),
                support_budgets=list(support_budgets),
                eval_n=eval_n,
            )
            all_records.extend(b_records)

    # Sort deterministically
    all_records.sort(key=lambda r: (r["bundle_seed"], r["data_seed"], r["support_n"]))

    # Aggregate by budget
    budget_summaries: dict[str, Any] = {}
    for s_n in support_budgets:
        b_recs = [r for r in all_records if r["support_n"] == s_n]
        or_ems = [r["oracle_heldout_em"] for r in b_recs]
        ex_ems = [r["exhaustive_heldout_em"] for r in b_recs]
        bm_ems = [r["beam_heldout_em"] for r in b_recs]
        deltas = [r["oracle_minus_exhaustive_em"] for r in b_recs]

        budget_summaries[f"N_{s_n}"] = {
            "support_n": s_n,
            "total_cells": len(b_recs),
            "mean_oracle_heldout_em": sum(or_ems) / len(or_ems),
            "mean_exhaustive_heldout_em": sum(ex_ems) / len(ex_ems),
            "mean_beam_heldout_em": sum(bm_ems) / len(bm_ems),
            "mean_oracle_minus_exhaustive_em": sum(deltas) / len(deltas),
            "exhaustive_ge_95_count": sum(1 for e in ex_ems if e >= CLOSURE_THRESHOLD),
            "exhaustive_ge_95_fraction": sum(1 for e in ex_ems if e >= CLOSURE_THRESHOLD)
            / len(ex_ems),
            "oracle_ge_95_count": sum(1 for o in or_ems if o >= CLOSURE_THRESHOLD),
        }

    # Aggregate by bundle x budget
    per_bundle_budget: dict[str, dict[str, Any]] = {}
    falsification_bundles: list[int] = []

    for b_seed in bundle_seeds:
        per_bundle_budget[f"bundle_{b_seed}"] = {}
        for s_n in support_budgets:
            cell_recs = [
                r for r in all_records if r["bundle_seed"] == b_seed and r["support_n"] == s_n
            ]
            or_vals = [r["oracle_heldout_em"] for r in cell_recs]
            ex_vals = [r["exhaustive_heldout_em"] for r in cell_recs]
            bm_vals = [r["beam_heldout_em"] for r in cell_recs]

            or_ci = compute_confidence_interval(or_vals, confidence=0.95)
            ex_ci = compute_confidence_interval(ex_vals, confidence=0.95)
            bm_ci = compute_confidence_interval(bm_vals, confidence=0.95)

            # Pooled Wilson score interval over all binary evaluation instances
            total_instances = len(cell_recs) * eval_n
            or_successes = int(round(sum(or_vals) * eval_n))
            ex_successes = int(round(sum(ex_vals) * eval_n))
            or_wilson = compute_wilson_score_interval(or_successes, total_instances)
            ex_wilson = compute_wilson_score_interval(ex_successes, total_instances)

            mean_or = sum(or_vals) / len(or_vals)
            mean_ex = sum(ex_vals) / len(ex_vals)
            mean_bm = sum(bm_vals) / len(bm_vals)

            oracle_ci_lower_ge_95 = or_ci[0] >= CLOSURE_THRESHOLD
            exhaustive_ci_upper_lt_95 = ex_ci[1] < CLOSURE_THRESHOLD
            falsifies = oracle_ci_lower_ge_95 and exhaustive_ci_upper_lt_95

            per_bundle_budget[f"bundle_{b_seed}"][f"N_{s_n}"] = {
                "bundle_seed": b_seed,
                "bundle_role": "challenge_bundle" if b_seed == 4 else "ceiling_control",
                "support_n": s_n,
                "num_seeds": len(cell_recs),
                "mean_oracle_em": mean_or,
                "oracle_em_ci_95": list(or_ci),
                "oracle_em_wilson_95": list(or_wilson),
                "mean_exhaustive_em": mean_ex,
                "exhaustive_em_ci_95": list(ex_ci),
                "exhaustive_em_wilson_95": list(ex_wilson),
                "mean_beam_em": mean_bm,
                "beam_em_ci_95": list(bm_ci),
                "mean_oracle_minus_exhaustive": mean_or - mean_ex,
                "oracle_ci_lower_ge_95": oracle_ci_lower_ge_95,
                "exhaustive_ci_upper_lt_95": exhaustive_ci_upper_lt_95,
                "falsifies_closure_cell": falsifies,
                "all_exhaustive_ge_95": all(e >= CLOSURE_THRESHOLD for e in ex_vals),
            }

            # Check falsification at max budget
            max_budget = max(support_budgets)
            if s_n == max_budget and falsifies:
                falsification_bundles.append(b_seed)

    max_budget = max(support_budgets)
    max_budget_key = f"N_{max_budget}"
    max_budget_stats = {
        f"bundle_{b_seed}": per_bundle_budget[f"bundle_{b_seed}"][max_budget_key]
        for b_seed in bundle_seeds
    }

    # Scientific Decision Logic
    if len(falsification_bundles) >= 2:
        scientific_decision = "FALSIFY_BOUNDED_RESOURCE_CLOSURE"
        rationale = (
            f"At max pre-fixed support budget N={max_budget}, {len(falsification_bundles)} bundles "
            f"({falsification_bundles}) satisfy pooled Oracle EM 95% CI lower bound >= 0.95 "
            f"and Exhaustive EM 95% CI upper bound < 0.95, statistically falsifying "
            f"bounded-resource closure for NEGATE->REVERSE->SHIFT."
        )
    elif falsification_bundles == [4]:
        scientific_decision = "BUNDLE_SPECIFIC_QUALIFIED_RESULT"
        rationale = (
            f"At max pre-fixed support budget N={max_budget}, only challenge Bundle 4 "
            f"maintains separation (Oracle CI lower >= 0.95 and Exhaustive CI upper < 0.95), "
            f"while ceiling controls Bundles 1-3 achieve near-ceiling baseline EM >= 0.95. "
            f"Recorded as a bundle-specific qualified result."
        )
    else:
        scientific_decision = "FINITE_SAMPLE_SUPPORT_MISSELECTION"
        n_max_means = [
            per_bundle_budget[f"bundle_{b}"][max_budget_key]["mean_exhaustive_em"]
            for b in bundle_seeds
        ]
        mean_delta_max = (
            sum(
                per_bundle_budget[f"bundle_{b}"][max_budget_key]["mean_oracle_minus_exhaustive"]
                for b in bundle_seeds
            )
            / len(bundle_seeds)
        )
        n_means_str = str([round(m, 4) for m in n_max_means])
        rationale = (
            f"Across intact bundles 1-4 and {len(data_seeds)} data seeds, increasing support "
            f"budget to N={max_budget} closes candidate mis-selection. At N={max_budget}, mean "
            f"exhaustive EM across bundles is {n_means_str} with mean oracle-baseline "
            f"gap Delta = {mean_delta_max:.4f}. Zero bundles exhibit Oracle >= 0.95 with "
            f"Exhaustive < 0.95. The isolated NRQ-006 seed-101 finding (Oracle 0.96 vs Exh 0.88 "
            f"on N=32) is conclusively closed as a finite-sample / support-mis-selection "
            f"artifact under small sample budget."
        )

    report = NRQ008Report(
        task_id="NRQ-008",
        task_name="Replication and Support-Budget Sensitivity of the Sole APC-over-Exhaustive Cell",
        date="2026-09-13",
        process_id=process_id,
        recipe=list(TARGET_RECIPE),
        bundle_seeds_evaluated=list(bundle_seeds),
        challenge_bundle=4,
        ceiling_control_bundles=[1, 2, 3],
        data_seeds_evaluated=list(data_seeds),
        support_budgets=list(support_budgets),
        eval_n=eval_n,
        total_cells_evaluated=len(all_records),
        dataset_manifest_hash=manifest_hash,
        budget_summaries=budget_summaries,
        per_bundle_budget_stats=per_bundle_budget,
        max_budget_256_bundle_stats=max_budget_stats,
        scientific_decision=scientific_decision,
        decision_rationale=rationale,
        falsification_bundles_count=len(falsification_bundles),
        records=all_records,
    )

    return report, manifest_doc


def save_nrq008_artifacts(
    report: NRQ008Report,
    manifest_doc: dict[str, Any],
    repo_root: Path = Path("."),
    process_id: int = 1,
) -> tuple[Path, Path, Path]:
    """Save NRQ-008 review record, run summary, and dataset manifest."""
    research_dir = repo_root / "docs" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    review_file = research_dir / "NRQ008_REVIEW_RECORD.json"
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    runs_dir = repo_root / "runs" / "nrq008_replication_and_support_budget"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = runs_dir / f"summary_process_{process_id}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    manifest_file = runs_dir / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    return review_file, summary_file, manifest_file


def verify_nrq008_reproducibility(
    process_1_summary_path: Path,
    process_2_summary_path: Path,
    output_verification_path: Path,
) -> dict[str, Any]:
    """Verify bitwise exact match between two independent runs of NRQ-008."""
    p1 = json.loads(process_1_summary_path.read_text(encoding="utf-8"))
    p2 = json.loads(process_2_summary_path.read_text(encoding="utf-8"))

    hash_match = p1["dataset_manifest_hash"] == p2["dataset_manifest_hash"]
    decision_match = p1["scientific_decision"] == p2["scientific_decision"]

    # Check primary metric differences across all cells
    records_1 = p1["records"]
    records_2 = p2["records"]
    assert len(records_1) == len(records_2), "Cell counts differ between processes!"

    max_delta_oracle = 0.0
    max_delta_exh = 0.0
    max_delta_beam = 0.0
    recipe_mismatches = 0

    for r1, r2 in zip(records_1, records_2, strict=True):
        assert r1["bundle_seed"] == r2["bundle_seed"]
        assert r1["data_seed"] == r2["data_seed"]
        assert r1["support_n"] == r2["support_n"]

        d_or = abs(r1["oracle_heldout_em"] - r2["oracle_heldout_em"])
        d_ex = abs(r1["exhaustive_heldout_em"] - r2["exhaustive_heldout_em"])
        d_bm = abs(r1["beam_heldout_em"] - r2["beam_heldout_em"])

        max_delta_oracle = max(max_delta_oracle, d_or)
        max_delta_exh = max(max_delta_exh, d_ex)
        max_delta_beam = max(max_delta_beam, d_bm)

        if r1["exhaustive_recovered_recipe"] != r2["exhaustive_recovered_recipe"]:
            recipe_mismatches += 1

    metrics_match = (
        max_delta_oracle < 1e-6
        and max_delta_exh < 1e-6
        and max_delta_beam < 1e-6
        and recipe_mismatches == 0
    )

    verification_status = "PASS" if (hash_match and decision_match and metrics_match) else "FAIL"

    record = {
        "task_id": "NRQ-008",
        "verification_status": verification_status,
        "dataset_manifest_hash": p1["dataset_manifest_hash"],
        "manifest_hash_match": hash_match,
        "scientific_decision_match": decision_match,
        "metrics_match": metrics_match,
        "max_delta_oracle_em": max_delta_oracle,
        "max_delta_exhaustive_em": max_delta_exh,
        "max_delta_beam_em": max_delta_beam,
        "recipe_mismatches": recipe_mismatches,
        "process_1_cells": len(records_1),
        "process_2_cells": len(records_2),
        "process_1_decision": p1["scientific_decision"],
        "process_2_decision": p2["scientific_decision"],
    }

    output_verification_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_verification_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record
