"""Composition search baseline over the compact primitive bank.

Phase A.1 Post-Diagnostic: Branch B Integration (Task A1-B004, Milestone B-M4).

Recovers multi-step composition recipes for novel composite tasks without oracle
primitive identity using heuristic/beam search over the primitive bank.

Strict Invariants Enforced:
1. Zero oracle primitive identity: does not access `example.oracle_metadata`,
   `example.program`, or `step.operation`.
2. Structural heuristic pruning: rejects candidates violating length constraints
   or missing required arguments before neural execution.
3. Zero bank expansion: searches existing primitives only, adding zero new parameters.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from apc.environments.generator import Example
from apc.environments.operations import get_operation
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import (
    CompositionRecipe,
    execute_composition_recipe,
    resolve_candidate_calls,
)

__all__ = [
    "SearchResult",
    "is_candidate_prefix_valid",
    "is_candidate_structurally_valid",
    "search_composition_recipe",
]


@dataclass(frozen=True)
class SearchResult:
    """Outcome of composition recipe search for a composite task."""

    recovered_recipe: CompositionRecipe
    candidate_operations: tuple[str, ...]
    exact_match_adapt: float
    loss_adapt: float
    candidates_evaluated: int
    candidates_pruned: int
    search_time_seconds: float
    search_metadata: dict[str, Any] = field(default_factory=dict)


def _predicted_output_length(
    candidate_operations: Sequence[str],
    input_length: int,
) -> int | None:
    """Compute predicted output length, or None if structurally invalid."""
    cur_len = input_length
    for op_name in candidate_operations:
        op_def = get_operation(op_name)
        if not op_def.is_valid_for_length(cur_len):
            return None
        cur_len = op_def.output_length(cur_len)
    return cur_len


def is_candidate_structurally_valid(
    candidate_operations: Sequence[str],
    examples: Sequence[Example],
    *,
    must_match_target_length: bool = True,
) -> bool:
    """Check whether a candidate operation sequence is structurally valid for examples.

    Verifies:
    1. Intermediate sequence lengths satisfy operation constraints.
    2. Final output length matches target length (if must_match_target_length=True).
    3. Required arguments exist in model-visible task specification.
    """
    if not candidate_operations or not examples:
        return False

    for ex in examples:
        pred_len = _predicted_output_length(candidate_operations, len(ex.input_tokens))
        if pred_len is None:
            return False
        if must_match_target_length and pred_len != len(ex.target_tokens):
            return False

    try:
        resolve_candidate_calls(candidate_operations, examples)
    except (ValueError, KeyError):
        return False

    return True


def is_candidate_prefix_valid(
    candidate_operations: Sequence[str],
    examples: Sequence[Example],
) -> bool:
    """Check if candidate can serve as a valid prefix for deeper expansion.

    Because all canonical primitives either preserve length or reduce length,
    an intermediate length strictly less than target length can never reach target length.
    """
    if not candidate_operations or not examples:
        return False

    for ex in examples:
        pred_len = _predicted_output_length(candidate_operations, len(ex.input_tokens))
        if pred_len is None:
            return False
        target_len = len(ex.target_tokens)
        if pred_len < target_len:
            return False

    return True


def _evaluate_candidate_on_adaptation(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    candidate: tuple[str, ...],
    examples: Sequence[Example],
) -> tuple[float, float]:
    """Compute (exact_match, token_cross_entropy_loss) on adaptation examples."""
    logits = execute_composition_recipe(
        core,
        bank,
        op_to_id,
        examples,
        candidate_operations=candidate,
    )
    predictions = logits.argmax(dim=-1)

    exact_matches = 0
    total_loss = 0.0
    total_tokens = 0

    for i, ex in enumerate(examples):
        target = ex.target_tokens
        n = len(target)
        pred = tuple(predictions[i, :n].tolist())
        if pred == target:
            exact_matches += 1

        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        pred_logits = logits[i, :n, :]
        step_loss = F.cross_entropy(pred_logits, target_tensor, reduction="sum")
        total_loss += step_loss.item()
        total_tokens += n

    em = exact_matches / len(examples)
    avg_loss = total_loss / max(1, total_tokens)
    return em, avg_loss


def search_composition_recipe(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    adaptation_examples: Sequence[Example],
    *,
    available_operations: Sequence[str] | None = None,
    max_depth: int = 2,
    beam_width: int = 16,
    early_stop_exact_match: float = 1.0,
) -> SearchResult:
    """Recover composition recipe for a composite task using heuristic/beam search.

    Strict Invariants Enforced:
    1. Zero oracle operation identity: `task_spec.steps[*].operation` is never inspected.
    2. Structural pruning: non-matching lengths and missing arguments are pruned
       before neural execution.
    3. Zero bank expansion: only resident primitives in `bank` are queried.

    Args:
        core: SharedContentEncoder (task-blind content encoder).
        bank: PrimitiveBank containing compact primitives.
        op_to_id: Mapping from canonical operation name to primitive id.
        adaptation_examples: Small support set of examples from composite task (e.g. N=16..32).
        available_operations: Optional candidate operation names. If None, uses `op_to_id.keys()`.
        max_depth: Maximum recipe depth to search (default: 2, supports up to 3).
        beam_width: Beam width for candidate search (default: 16).
        early_stop_exact_match: Early stopping threshold (default: 1.0).

    Returns:
        `SearchResult` containing the best recovered recipe and search diagnostics.
    """
    if not adaptation_examples:
        raise ValueError("adaptation_examples must be non-empty.")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1.")

    start_time = time.perf_counter()
    ops = tuple(available_operations or sorted(op_to_id.keys()))

    candidates_evaluated = 0
    candidates_pruned = 0

    best_candidate: tuple[str, ...] | None = None
    best_score: tuple[float, float, int] = (-1.0, -float("inf"), 0)
    best_metrics: tuple[float, float] = (0.0, float("inf"))

    # Beam maintains prefixes for expansion
    beam: list[tuple[str, ...]] = [()]

    with torch.no_grad():
        for d in range(1, max_depth + 1):
            next_beam_candidates: list[tuple[tuple[float, float, int], tuple[str, ...]]] = []

            for prefix in beam:
                for op in ops:
                    candidate = prefix + (op,)

                    # Check 1: Can this candidate be evaluated as a complete solution?
                    if is_candidate_structurally_valid(
                        candidate, adaptation_examples, must_match_target_length=True
                    ):
                        candidates_evaluated += 1
                        em, loss = _evaluate_candidate_on_adaptation(
                            core, bank, op_to_id, candidate, adaptation_examples
                        )
                        score = (em, -loss, -d)
                        next_beam_candidates.append((score, candidate))

                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                            best_metrics = (em, loss)

                        # Early exit if ceiling accuracy reached
                        if em >= early_stop_exact_match:
                            break
                    else:
                        # Check 2: Can it serve as a prefix for depth d+1?
                        if d < max_depth and is_candidate_prefix_valid(
                            candidate, adaptation_examples
                        ):
                            # Retain as prefix with default neutral score
                            next_beam_candidates.append(((0.0, -float("inf"), -d), candidate))
                        else:
                            candidates_pruned += 1

                if best_metrics[0] >= early_stop_exact_match:
                    break

            if best_metrics[0] >= early_stop_exact_match:
                break

            # Select top beam_width candidates for next depth
            next_beam_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [cand for _, cand in next_beam_candidates[:beam_width]]

    if best_candidate is None:
        raise RuntimeError(
            "Composition search failed to discover any structurally valid candidate recipe."
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
            "max_depth": max_depth,
            "beam_width": beam_width,
            "best_score": best_score,
        },
    )
