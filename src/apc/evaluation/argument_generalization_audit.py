# ruff: noqa: E501
"""Evaluation-only L4 argument-generalization decomposition for B-C005D2-004.

`B-C005G` reported physical-family routing at L4 (`L4_CONFUSABLE_FAMILY`)
pinned to 1.000 while argument accuracy fell to 0.8883 on the new sealed
partition, after the repaired factorized scorer reached ~0.97 on development.
This module explains that gap without changing `ArgumentScorer`, the router,
or any threshold: it reconstructs the frozen R0/R2 cells exactly as
`hard_negative_second_diagnostic` and `semantic_relation_audit` already do,
restricted to the L4 level and the sealed-gate bank size, and adds the
argument-specific descriptors and calibration readout the L4 rows in those
earlier modules do not capture.

Three things are worth stating up front because they shape every metric here:

1. At L4 the competitor virtual candidate shares `execute_primitive_id` with
   the correct candidate (`build_hard_negative_candidates`), so
   ``family_top1`` is 1.0 unless the router ranks some *unrelated* resident
   primitive above both -- it is not, by construction, informative about
   argument resolution. ``argument_accuracy`` and ``primitive_call_top1`` are
   computed identically for L4 in the installed pipeline
   (`retrieval_repair_benchmark.evaluate_repair_cell`); both are reproduced
   here for schema completeness, not because they carry independent evidence.
2. `ArgumentScorer.forward` scores a *candidate argument* by reading one
   entry (or the mean of several, for SELECT) out of a per-example softmax
   distribution over `arg_vocab_size` classes produced by `z_task` alone --
   the candidate's proposed value only selects which entry to read, it is
   never embedded as an input. Argument generalization, at this
   architecture, reduces exactly to "does a per-operation linear softmax
   classifier over `z_task` predict the true argument value" -- a pure
   representation/classification question, not an argument-encoding one,
   for SHIFT/COUNT/BIND.
3. SELECT is the one exception to (2): `train_on_examples` fits it with
   independent multi-hot `BCEWithLogitsLoss` targets (each index scored on
   its own), but `forward` always reads the *same* softmax distribution
   used by the single-valued operations, which forces the selected indices'
   probability mass to compete against each other and against every other
   vocab entry. This train/inference mismatch is structural (always present
   for SELECT, independent of data), so it is reported as its own
   `ARGUMENT_ENCODING_FAILURE` evidence rather than folded into the
   representation/data-coverage questions asked of the other three
   operations.

Reconstruction is bounded to one bank size (128, the sealed-gate scale) and
the four parameterized operations, following B-C005D2-002/003's own scope
precedent. No sealed seed is ever used to fit a gradient -- the one added
control (D2-004's "does more data help" question) trains an extra,
never-installed `ArgumentScorer` on a *larger development-only* sample and
reads its accuracy out on sealed content, exactly mirroring how
`hard_negative_second_diagnostic` and `semantic_relation_audit` already
restrict gradient updates to development seeds.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
    _wrong_arguments,
    build_hard_negative_candidates,
)
from apc.evaluation.hard_negative_second_diagnostic import _quantiles, _training_examples
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    SEALED_GATE_SEEDS,
    RetrievalRepairConfig,
    _rank_candidates_factorized,
    train_repaired_router_and_scorer,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.argument_scoring import (
    ArgumentScorer,
    ArgumentScorerConfig,
    extract_raw_argument_values,
)
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

_L4: Final[HardNegativeLevel] = HardNegativeLevel.L4_CONFUSABLE_FAMILY
_PRE_REPAIR: Final[str] = "R0_frozen_pre_repair"
_POST_REPAIR: Final[str] = "R2_frozen_post_repair"
_SET_VALUED_OPERATIONS: Final[tuple[str, ...]] = ("SELECT",)
_CONTENT_KEY_OPERATIONS: Final[tuple[str, ...]] = ("COUNT", "BIND")


@dataclass(frozen=True)
class ArgumentGeneralizationAuditConfig:
    """Frozen protocol for the D2-004 L4 argument-generalization decomposition."""

    original_sealed_seeds: tuple[int, ...] = tuple(sorted(SEALED_GATE_SEEDS))
    development_seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    regate_sealed_seeds: tuple[int, ...] = DEFAULT_REGATE_SEEDS
    bank_size: int = 128
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    probe_train_examples: int = 128
    probe_steps: int = 250
    no_failure_threshold: float = 0.95
    value_coverage_gap_tolerance: float = 0.10
    rare_share_threshold: float = 0.10
    generalization_gap_tolerance: float = 0.10
    data_scale_improvement_tolerance: float = 0.05
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        groups = (self.original_sealed_seeds, self.development_seeds, self.regate_sealed_seeds)
        if any(not seeds for seeds in groups):
            raise ValueError("all original, development, and re-gate partitions must be non-empty")
        if len(self.development_seeds) != len(self.regate_sealed_seeds):
            raise ValueError("development and re-gate sealed partitions must have equal lengths")
        if set(self.original_sealed_seeds) != SEALED_GATE_SEEDS:
            raise ValueError("original_sealed_seeds must remain the historical B-C005 partition")
        all_sealed_seeds = set(self.original_sealed_seeds) | set(DEFAULT_REGATE_SEEDS)
        if set(self.development_seeds) & all_sealed_seeds:
            raise ValueError("development seeds must be disjoint from both sealed partitions")
        if set(self.regate_sealed_seeds) & set(self.original_sealed_seeds):
            raise ValueError("the two sealed partitions must be disjoint")
        if self.bank_size not in DEFAULT_BANK_SIZES:
            raise ValueError(f"bank_size must be one of {DEFAULT_BANK_SIZES}")
        if set(self.target_operations) - set(DEFAULT_TARGET_OPERATIONS):
            raise ValueError("target_operations must be drawn from the parameterized operations")
        if self.query_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")
        if self.router_steps < 1 or self.probe_steps < 1:
            raise ValueError("optimization step counts must be positive")
        if self.probe_train_examples < 1:
            raise ValueError("probe_train_examples must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        for value in (
            self.no_failure_threshold,
            self.value_coverage_gap_tolerance,
            self.rare_share_threshold,
            self.generalization_gap_tolerance,
            self.data_scale_improvement_tolerance,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("classification thresholds must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


# ---------------------------------------------------------------------------
# Structural argument descriptors (D2-004.2). All derived from frozen-model /
# pre-model facts (the generator's own sampling rule, or the fixed
# `_wrong_arguments` perturbation) -- never from a query outcome label.
# ---------------------------------------------------------------------------


def _argument_distance(operation: str, correct_values: Sequence[int], wrong_values: Sequence[int], modulus: int) -> float:
    """Distance between the correct and L4-competitor argument values.

    For the three scalar operations `_wrong_arguments` always adds exactly
    one modulo the relevant domain, so this is expected to be a near-constant
    minimal circular distance; SELECT's per-index +1 perturbation can differ
    because re-sorting the shifted index set can coincide with the original
    set at wraparound.
    """
    if operation in _SET_VALUED_OPERATIONS:
        return float(len(set(correct_values).symmetric_difference(set(wrong_values))))
    correct, wrong = correct_values[0], wrong_values[0]
    raw = abs(correct - wrong)
    return float(min(raw, modulus - raw)) if modulus > 0 else float(raw)


def _content_dependent_ambiguity(operation: str, example: Any, wrong_values: Sequence[int]) -> bool | None:
    """Whether the L4 competitor's wrong argument is itself content-plausible.

    COUNT's `target` and BIND's `query_key` are vocabulary token values, so a
    wrong value that happens to also appear in this example's content is a
    harder, content-consistent confusion than one that never appears at all.
    SHIFT/SELECT arguments are positions, not content values, so this
    descriptor is not meaningful for them and is reported as `None`.
    """
    if operation == "COUNT":
        return wrong_values[0] in example.input_tokens
    if operation == "BIND":
        return wrong_values[0] in example.input_tokens[0::2]
    return None


def _training_values(training_examples_by_op: dict[str, list[Any]], operation: str) -> list[int]:
    """Flatten every raw correct argument value seen while fitting this operation's head."""
    values: list[int] = []
    for example in training_examples_by_op.get(operation, []):
        assert example.task_spec is not None
        values.extend(extract_raw_argument_values(operation, example.task_spec.steps[0].arguments))
    return values


# ---------------------------------------------------------------------------
# Frozen reconstruction + row collection, mirroring
# `hard_negative_second_diagnostic._collect_rows`, restricted to L4 and
# enriched with the argument-specific fields D2-004 needs.
# ---------------------------------------------------------------------------


def _repair_config(config: ArgumentGeneralizationAuditConfig, seed: int) -> RetrievalRepairConfig:
    return RetrievalRepairConfig(
        seeds=(seed,),
        bank_sizes=(config.bank_size,),
        levels=(_L4,),
        target_operations=config.target_operations,
        conditions=("R2",),
        support_examples=32,
        query_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        arg_lambda=config.arg_lambda,
        deterministic_algorithms=config.deterministic_algorithms,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _base_config(config: ArgumentGeneralizationAuditConfig, model_seed: int) -> HardNegativeBenchmarkConfig:
    return HardNegativeBenchmarkConfig(
        seeds=(model_seed,),
        bank_sizes=(config.bank_size,),
        levels=(_L4,),
        target_operations=config.target_operations,
        num_eval_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        top_k=config.top_k,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


def _query_seed(partition: str, evaluation_seed: int, bank_size: int, target_id: int) -> int:
    if partition == "regate_sealed":
        return evaluation_seed * 1_000_000 + bank_size * 100 + target_id + 1
    if partition == "original_sealed":
        return evaluation_seed * 100_000 + bank_size * 100 + target_id + 1
    return evaluation_seed * 50_000 + bank_size + target_id + 100


def _rows_for_operation(
    *,
    operation: str,
    examples: Sequence[Any],
    z_task: torch.Tensor,
    vocab: int,
    candidates: Sequence[Any],
    correct_index: int,
    competitor_index: int,
    family_scores: torch.Tensor,
    total_scores: torch.Tensor,
    ordering: torch.Tensor,
    argument_scorer: ArgumentScorer | None,
    top_k: int,
    training_values: list[int] | None,
    partition: str,
    state: str,
    evaluation_seed: int,
    model_seed: int,
    training_seed: int | None,
    bank_size: int,
) -> list[dict[str, Any]]:
    is_set_valued = operation in _SET_VALUED_OPERATIONS
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        assert example.task_spec is not None
        correct_arguments = dict(example.task_spec.steps[0].arguments)
        wrong_arguments = _wrong_arguments(operation, example, vocab)
        correct_values = extract_raw_argument_values(operation, correct_arguments)
        wrong_values = extract_raw_argument_values(operation, wrong_arguments)

        top1 = int(ordering[index, 0].item())
        top1_correct = top1 == correct_index
        top5_included = bool(
            (ordering[index, : min(top_k, len(candidates))] == correct_index).any().item()
        )
        family_top1 = candidates[top1].execute_primitive_id == candidates[correct_index].execute_primitive_id
        family_margin = float((family_scores[index, correct_index] - family_scores[index, competitor_index]).item())
        combined_margin = float((total_scores[index, correct_index] - total_scores[index, competitor_index]).item())

        p_correct = p_specific_wrong = p_best_wrong = argument_score_margin = None
        best_wrong_value: Any = None
        if argument_scorer is not None:
            vocab_size = argument_scorer.config.arg_vocab_size
            clamped_correct = [min(max(0, value), vocab_size - 1) for value in correct_values]
            clamped_wrong = [min(max(0, value), vocab_size - 1) for value in wrong_values]
            logits = argument_scorer.heads[operation](z_task[index : index + 1])
            probs = F.softmax(logits, dim=-1).squeeze(0)
            p_correct = float(probs[clamped_correct].mean().item())
            p_specific_wrong = float(probs[clamped_wrong].mean().item())
            argument_score_margin = 2.0 * (p_correct - p_specific_wrong)
            masked = probs.clone()
            masked[clamped_correct] = -1.0
            top_indices = torch.topk(masked, len(clamped_correct)).indices.tolist()
            p_best_wrong = float(probs[top_indices].mean().item())
            best_wrong_value = top_indices if is_set_valued else top_indices[0]

        modulus = len(example.input_tokens) if operation in ("SHIFT", "SELECT") else vocab
        distance = _argument_distance(operation, correct_values, wrong_values, modulus)
        content_ambiguous = _content_dependent_ambiguity(operation, example, wrong_values)
        seen_in_training = (
            all(value in training_values for value in correct_values) if training_values is not None else None
        )
        training_value_count = (
            min(training_values.count(value) for value in correct_values) if training_values is not None else None
        )

        rows.append(
            {
                "partition": partition,
                "router_state": state,
                "operation": operation,
                "checkpoint_label": f"base_seed_{model_seed};train_seed_{training_seed}",
                "evaluation_seed": evaluation_seed,
                "model_seed": model_seed,
                "training_seed": training_seed,
                "bank_size": bank_size,
                "sequence_length": len(example.input_tokens),
                "argument_value_type": "set_valued" if is_set_valued else "single_valued",
                "correct_argument_value": correct_values if is_set_valued else correct_values[0],
                "wrong_argument_value": wrong_values if is_set_valued else wrong_values[0],
                "argument_distance": distance,
                "content_dependent_ambiguity": content_ambiguous,
                "seen_in_training": seen_in_training,
                "training_value_count": training_value_count,
                "family_top1": bool(family_top1),
                "argument_correct": bool(top1_correct),
                "primitive_call_top1": bool(top1_correct),
                "top5_included": top5_included,
                "family_score_margin": family_margin,
                "combined_score_margin": combined_margin,
                "argument_score_margin": argument_score_margin,
                "p_correct_argument": p_correct,
                "p_specific_wrong_argument": p_specific_wrong,
                "p_best_wrong_argument": p_best_wrong,
                "best_wrong_argument_value": best_wrong_value,
            }
        )
    return rows


def _collect_l4_rows(
    *,
    config: ArgumentGeneralizationAuditConfig,
    partition: str,
    state: str,
    evaluation_seed: int,
    model_seed: int,
    training_seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]] | None]:
    """Rebuild one frozen (partition, state, seed) cell and collect its L4 rows."""
    set_seed(evaluation_seed, deterministic_algorithms=config.deterministic_algorithms)
    base_config = _base_config(config, model_seed)
    core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)
    bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
        core, base_bank, base_router, op_to_id, config.bank_size, seed=evaluation_seed
    )
    operation_by_id = {primitive_id: operation for operation, primitive_id in op_to_id.items()}
    operation_by_id.update({primitive_id: "SWAP_ENDS" for primitive_id in distractor_ids})

    argument_scorer: ArgumentScorer | None = None
    training_examples_by_op: dict[str, list[Any]] | None = None
    if state == _POST_REPAIR:
        if training_seed is None:
            raise ValueError("post-repair reconstruction requires a development training seed")
        training_examples_by_op = _training_examples(
            training_seed, op_to_id, config.router_train_examples, regate_recipe=partition == "regate_sealed"
        )
        router, argument_scorer = train_repaired_router_and_scorer(
            core,
            router,
            candidate_ids,
            operation_by_id,
            training_examples_by_op,
            config=_repair_config(config, training_seed),
            condition="R2",
            device=core.device,
        )
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    if argument_scorer is not None:
        argument_scorer.eval()
        for parameter in argument_scorer.parameters():
            parameter.requires_grad_(False)

    rows: list[dict[str, Any]] = []
    for operation in config.target_operations:
        target_id = op_to_id[operation]
        examples = generate_benchmark_examples(
            _query_seed(partition, evaluation_seed, config.bank_size, target_id),
            config.query_examples,
            operation=operation,
            split="dev" if partition == "development" else "test",
        )
        keys = {primitive_id: router.key_parameter(primitive_id).detach().clone() for primitive_id in candidate_ids}
        candidates, competitor = build_hard_negative_candidates(
            level=_L4,
            target_id=target_id,
            target_operation=operation,
            candidate_ids=candidate_ids,
            keys_by_id=keys,
            operation_by_id=operation_by_id,
            seed=evaluation_seed,
        )
        correct_index = next(
            i for i, candidate in enumerate(candidates) if candidate is not competitor and candidate.execute_primitive_id == target_id
        )
        competitor_index = next(i for i, candidate in enumerate(candidates) if candidate is competitor)

        family_scores, _ = _rank_candidates_factorized(core, router, candidates, examples, None, operation_by_id, 0.0)
        total_scores, ordering = _rank_candidates_factorized(
            core,
            router,
            candidates,
            examples,
            argument_scorer,
            operation_by_id,
            config.arg_lambda if state == _POST_REPAIR else 0.0,
        )
        z_task = extract_task_representations(core, list(examples))

        rows.extend(
            _rows_for_operation(
                operation=operation,
                examples=examples,
                z_task=z_task,
                vocab=core.tokens.env_vocab_size,
                candidates=candidates,
                correct_index=correct_index,
                competitor_index=competitor_index,
                family_scores=family_scores,
                total_scores=total_scores,
                ordering=ordering,
                argument_scorer=argument_scorer,
                top_k=config.top_k,
                training_values=_training_values(training_examples_by_op, operation) if training_examples_by_op else None,
                partition=partition,
                state=state,
                evaluation_seed=evaluation_seed,
                model_seed=model_seed,
                training_seed=training_seed,
                bank_size=config.bank_size,
            )
        )
    return rows, training_examples_by_op


# ---------------------------------------------------------------------------
# Data-scale control: does more (still development-only) R2 training data
# close the sealed gap, without touching the installed router/ArgumentScorer?
# ---------------------------------------------------------------------------


def _data_scale_control_rows(
    config: ArgumentGeneralizationAuditConfig,
    *,
    sealed_seed: int,
    development_seed: int,
    model_seed: int,
    sealed_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fit a never-installed `ArgumentScorer` on a larger development-only sample
    and read its accuracy out on the same sealed query content used for the
    real R2 cell, to separate few-shot data starvation from a deeper failure.
    """
    set_seed(sealed_seed, deterministic_algorithms=config.deterministic_algorithms)
    core, _, _, op_to_id = _build_frozen_base_system(model_seed, _base_config(config, model_seed))
    larger_examples_by_op = _training_examples(
        development_seed, op_to_id, config.probe_train_examples, regate_recipe=True
    )
    z_by_op = {
        operation: extract_task_representations(core, examples)
        for operation, examples in larger_examples_by_op.items()
        if examples
    }
    probe = ArgumentScorer(
        ArgumentScorerConfig(
            d_model=core.model.config.d_model, lambda_weight=config.arg_lambda, steps=config.probe_steps
        )
    ).to(core.device)
    probe.train_on_examples(z_by_op, larger_examples_by_op, device=core.device)

    rows: list[dict[str, Any]] = []
    for operation in config.target_operations:
        operation_rows = [row for row in sealed_rows if row["operation"] == operation]
        if not operation_rows:
            continue
        target_id = op_to_id[operation]
        examples = generate_benchmark_examples(
            _query_seed("regate_sealed", sealed_seed, config.bank_size, target_id),
            config.query_examples,
            operation=operation,
            split="test",
        )
        z_task = extract_task_representations(core, list(examples))
        correct = 0
        for index, example in enumerate(examples):
            assert example.task_spec is not None
            correct_arguments = dict(example.task_spec.steps[0].arguments)
            wrong_arguments = _wrong_arguments(operation, example, core.tokens.env_vocab_size)
            p_correct = probe(z_task[index : index + 1], operation, correct_arguments).item()
            p_wrong = probe(z_task[index : index + 1], operation, wrong_arguments).item()
            correct += int(p_correct > p_wrong)
        rows.append(
            {
                "operation": operation,
                "evaluation_seed": sealed_seed,
                "development_seed": development_seed,
                "n_query_examples": len(examples),
                "probe_train_examples": config.probe_train_examples,
                "actual_r2_argument_accuracy": statistics.fmean(
                    float(row["argument_correct"]) for row in operation_rows
                ),
                "data_scale_probe_argument_accuracy": correct / len(examples),
            }
        )
    return rows


def _data_scale_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for operation in DEFAULT_TARGET_OPERATIONS:
        operation_rows = [row for row in rows if row["operation"] == operation]
        if not operation_rows:
            continue
        result[operation] = {
            "n_seeds": len(operation_rows),
            "mean_actual_r2_argument_accuracy": statistics.fmean(
                row["actual_r2_argument_accuracy"] for row in operation_rows
            ),
            "mean_data_scale_probe_argument_accuracy": statistics.fmean(
                row["data_scale_probe_argument_accuracy"] for row in operation_rows
            ),
            "mean_improvement": statistics.fmean(
                row["data_scale_probe_argument_accuracy"] - row["actual_r2_argument_accuracy"]
                for row in operation_rows
            ),
        }
    return result


# ---------------------------------------------------------------------------
# D2-004.1 -- per-operation breakdown.
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _operation_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = sorted({(row["partition"], row["router_state"]) for row in rows})
    result: dict[str, Any] = {}
    for partition, state in cells:
        per_operation: dict[str, Any] = {}
        for operation in DEFAULT_TARGET_OPERATIONS:
            operation_rows = [
                row for row in rows if row["partition"] == partition and row["router_state"] == state and row["operation"] == operation
            ]
            if not operation_rows:
                continue
            per_operation[operation] = {
                "n": len(operation_rows),
                "family_top1": statistics.fmean(float(row["family_top1"]) for row in operation_rows),
                "argument_accuracy": statistics.fmean(float(row["argument_correct"]) for row in operation_rows),
                "primitive_call_top1": statistics.fmean(float(row["primitive_call_top1"]) for row in operation_rows),
                "top5": statistics.fmean(float(row["top5_included"]) for row in operation_rows),
                "argument_score_margin_mean": _safe_mean([row["argument_score_margin"] for row in operation_rows]),
                "family_score_margin_mean": statistics.fmean(row["family_score_margin"] for row in operation_rows),
                "combined_score_margin_mean": statistics.fmean(row["combined_score_margin"] for row in operation_rows),
            }
        result[f"{partition}/{state}"] = per_operation
    return result


# ---------------------------------------------------------------------------
# D2-004.2 -- argument-structure stratification.
# ---------------------------------------------------------------------------


def _group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "argument_accuracy": None, "primitive_call_top1": None}
    return {
        "n": len(rows),
        "argument_accuracy": statistics.fmean(float(row["argument_correct"]) for row in rows),
        "primitive_call_top1": statistics.fmean(float(row["primitive_call_top1"]) for row in rows),
    }


def _argument_structure_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    post_repair_cells = sorted(
        {(row["partition"], row["router_state"]) for row in rows if row["router_state"] == _POST_REPAIR}
    )
    seen_vs_rare: dict[str, Any] = {}
    content_ambiguity: dict[str, Any] = {}
    sequence_length: dict[str, Any] = {}
    argument_distance: dict[str, Any] = {}
    value_type: dict[str, Any] = {}

    for partition, state in post_repair_cells:
        cell_key = f"{partition}/{state}"
        cell_rows = [row for row in rows if row["partition"] == partition and row["router_state"] == state]

        seen_vs_rare[cell_key] = {}
        for operation in DEFAULT_TARGET_OPERATIONS:
            operation_rows = [
                row for row in cell_rows if row["operation"] == operation and row["seen_in_training"] is not None
            ]
            seen_vs_rare[cell_key][operation] = {
                "seen": _group_metrics([row for row in operation_rows if row["seen_in_training"]]),
                "rare": _group_metrics([row for row in operation_rows if not row["seen_in_training"]]),
            }

        content_ambiguity[cell_key] = {}
        for operation in _CONTENT_KEY_OPERATIONS:
            operation_rows = [
                row
                for row in cell_rows
                if row["operation"] == operation and row["content_dependent_ambiguity"] is not None
            ]
            content_ambiguity[cell_key][operation] = {
                "content_plausible_wrong_argument": _group_metrics(
                    [row for row in operation_rows if row["content_dependent_ambiguity"]]
                ),
                "content_absent_wrong_argument": _group_metrics(
                    [row for row in operation_rows if not row["content_dependent_ambiguity"]]
                ),
            }

        sequence_length[cell_key] = {}
        for operation in DEFAULT_TARGET_OPERATIONS:
            operation_rows = [row for row in cell_rows if row["operation"] == operation]
            by_length: dict[int, Any] = {}
            for length in sorted({row["sequence_length"] for row in operation_rows}):
                by_length[length] = _group_metrics(
                    [row for row in operation_rows if row["sequence_length"] == length]
                )
            sequence_length[cell_key][operation] = by_length

        argument_distance[cell_key] = {}
        for operation in DEFAULT_TARGET_OPERATIONS:
            distances = [row["argument_distance"] for row in cell_rows if row["operation"] == operation]
            argument_distance[cell_key][operation] = {
                "distinct_distances": sorted(set(distances)),
                "distance_stats": _quantiles(distances),
            }

        value_type[cell_key] = {
            "single_valued": _group_metrics(
                [row for row in cell_rows if row["argument_value_type"] == "single_valued"]
            ),
            "set_valued": _group_metrics([row for row in cell_rows if row["argument_value_type"] == "set_valued"]),
        }

    return {
        "seen_vs_rare": seen_vs_rare,
        "content_dependent_ambiguity": content_ambiguity,
        "sequence_length": sequence_length,
        "argument_distance": argument_distance,
        "value_type": value_type,
    }


# ---------------------------------------------------------------------------
# D2-004.3 -- calibration.
# ---------------------------------------------------------------------------


def _calibration_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    post_repair_cells = sorted(
        {(row["partition"], row["router_state"]) for row in rows if row["router_state"] == _POST_REPAIR}
    )
    result: dict[str, Any] = {}
    for partition, state in post_repair_cells:
        cell_key = f"{partition}/{state}"
        per_operation: dict[str, Any] = {}
        for operation in DEFAULT_TARGET_OPERATIONS:
            operation_rows = [
                row
                for row in rows
                if row["partition"] == partition
                and row["router_state"] == state
                and row["operation"] == operation
                and row["p_correct_argument"] is not None
            ]
            if not operation_rows:
                continue
            correct_rows = [row for row in operation_rows if row["argument_correct"]]
            incorrect_rows = [row for row in operation_rows if not row["argument_correct"]]
            per_operation[operation] = {
                "n": len(operation_rows),
                "mean_p_correct_argument": statistics.fmean(row["p_correct_argument"] for row in operation_rows),
                "mean_p_specific_wrong_argument": statistics.fmean(
                    row["p_specific_wrong_argument"] for row in operation_rows
                ),
                "mean_p_best_wrong_argument": statistics.fmean(
                    row["p_best_wrong_argument"] for row in operation_rows
                ),
                "mean_argument_score_margin": statistics.fmean(
                    row["argument_score_margin"] for row in operation_rows
                ),
                "mean_p_correct_when_argument_correct": (
                    statistics.fmean(row["p_correct_argument"] for row in correct_rows) if correct_rows else None
                ),
                "mean_p_correct_when_argument_incorrect": (
                    statistics.fmean(row["p_correct_argument"] for row in incorrect_rows)
                    if incorrect_rows
                    else None
                ),
            }
        result[cell_key] = per_operation
    return result


# ---------------------------------------------------------------------------
# D2-004.4 -- per-operation failure classification.
# ---------------------------------------------------------------------------

_SELECT_ENCODING_NOTE: Final[str] = (
    "ArgumentScorer.train_on_examples fits SELECT with independent multi-hot "
    "BCEWithLogitsLoss targets, but ArgumentScorer.forward always reads a "
    "single softmax distribution over the shared vocab head at inference, "
    "diluting probability mass across every selected index instead of "
    "scoring them independently -- a structural train/inference mismatch "
    "specific to the one set-valued operation."
)


def _classify_operations(
    breakdown: dict[str, Any],
    structure: dict[str, Any],
    data_scale: dict[str, Any],
    config: ArgumentGeneralizationAuditConfig,
) -> dict[str, Any]:
    sealed_cell = breakdown.get("regate_sealed/R2_frozen_post_repair", {})
    development_cell = breakdown.get("development/R2_frozen_post_repair", {})
    seen_rare_cell = structure.get("seen_vs_rare", {}).get("regate_sealed/R2_frozen_post_repair", {})

    result: dict[str, Any] = {}
    for operation in DEFAULT_TARGET_OPERATIONS:
        sealed_metrics = sealed_cell.get(operation)
        if sealed_metrics is None:
            result[operation] = {
                "classification": "UNRESOLVED",
                "evidence": {"reason": "no regate_sealed post-repair rows for this operation"},
            }
            continue

        sealed_accuracy = sealed_metrics["argument_accuracy"]
        development_accuracy = development_cell.get(operation, {}).get("argument_accuracy")
        seen_metrics = seen_rare_cell.get(operation, {}).get("seen", {})
        rare_metrics = seen_rare_cell.get(operation, {}).get("rare", {})
        seen_accuracy, rare_accuracy = seen_metrics.get("argument_accuracy"), rare_metrics.get("argument_accuracy")
        seen_n, rare_n = seen_metrics.get("n", 0) or 0, rare_metrics.get("n", 0) or 0
        rare_share = rare_n / (seen_n + rare_n) if (seen_n + rare_n) else None
        scale_improvement = data_scale.get(operation, {}).get("mean_improvement")

        evidence: dict[str, Any] = {
            "sealed_post_repair_argument_accuracy": sealed_accuracy,
            "development_post_repair_argument_accuracy": development_accuracy,
            "sealed_seen_value_argument_accuracy": seen_accuracy,
            "sealed_rare_value_argument_accuracy": rare_accuracy,
            "sealed_rare_value_share": rare_share,
            "data_scale_mean_improvement": scale_improvement,
        }

        if sealed_accuracy >= config.no_failure_threshold:
            classification = "NO_FAILURE"
        elif (
            seen_accuracy is not None
            and rare_accuracy is not None
            and (seen_accuracy - rare_accuracy) >= config.value_coverage_gap_tolerance
            and rare_share is not None
            and rare_share >= config.rare_share_threshold
        ):
            classification = "VALUE_COVERAGE_FAILURE"
        elif operation == "SELECT":
            classification = "ARGUMENT_ENCODING_FAILURE"
            evidence["structural_note"] = _SELECT_ENCODING_NOTE
        elif scale_improvement is not None and scale_improvement >= config.data_scale_improvement_tolerance:
            classification = "ARGUMENT_SCORER_GENERALIZATION_FAILURE"
        elif (
            development_accuracy is not None
            and (development_accuracy - sealed_accuracy) >= config.generalization_gap_tolerance
        ):
            classification = "TASK_REPRESENTATION_FAILURE"
        else:
            classification = "UNRESOLVED"

        result[operation] = {"classification": classification, "evidence": evidence}
    return result


# ---------------------------------------------------------------------------
# Top-level run.
# ---------------------------------------------------------------------------


def run_argument_generalization_audit(config: ArgumentGeneralizationAuditConfig) -> dict[str, Any]:
    """Run the B-C005D2-004 L4 argument-generalization decomposition."""
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    rows: list[dict[str, Any]] = []
    data_scale_rows: list[dict[str, Any]] = []

    for seed in config.original_sealed_seeds:
        cell_rows, _ = _collect_l4_rows(
            config=config, partition="original_sealed", state=_PRE_REPAIR, evaluation_seed=seed, model_seed=seed, training_seed=None
        )
        rows.extend(cell_rows)

    for seed in config.development_seeds:
        pre_rows, _ = _collect_l4_rows(
            config=config, partition="development", state=_PRE_REPAIR, evaluation_seed=seed, model_seed=seed, training_seed=None
        )
        rows.extend(pre_rows)
        post_rows, _ = _collect_l4_rows(
            config=config, partition="development", state=_POST_REPAIR, evaluation_seed=seed, model_seed=seed, training_seed=seed
        )
        rows.extend(post_rows)

    for sealed_seed, development_seed in zip(config.regate_sealed_seeds, config.development_seeds, strict=True):
        model_seed = sealed_seed % 5
        post_rows, _ = _collect_l4_rows(
            config=config,
            partition="regate_sealed",
            state=_POST_REPAIR,
            evaluation_seed=sealed_seed,
            model_seed=model_seed,
            training_seed=development_seed,
        )
        rows.extend(post_rows)
        data_scale_rows.extend(
            _data_scale_control_rows(
                config,
                sealed_seed=sealed_seed,
                development_seed=development_seed,
                model_seed=model_seed,
                sealed_rows=post_rows,
            )
        )

    breakdown = _operation_breakdown(rows)
    structure = _argument_structure_breakdown(rows)
    calibration = _calibration_summary(rows)
    data_scale = _data_scale_summary(data_scale_rows)
    classification = _classify_operations(breakdown, structure, data_scale, config)

    report = {
        "task_id": "B-C005D2-004",
        "config": config.to_dict(),
        "rows_analyzed": len(rows),
        "l4_operation_breakdown": breakdown,
        "argument_structure_breakdown": structure,
        "calibration_summary": calibration,
        "data_scale_control": data_scale,
        "failure_classification": classification,
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-004",
            "sealed_partitions": {
                "original": list(config.original_sealed_seeds),
                "regate": list(config.regate_sealed_seeds),
            },
            "development_partition": list(config.development_seeds),
            "sealed_training_prohibited": True,
            "data_scale_control_trains_only_on_development_seeds": True,
        }
        (out / "argument_generalization_audit_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "argument_generalization_audit_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        (out / "argument_generalization_audit_system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "argument_generalization_audit_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out / "l4_argument_breakdown.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
