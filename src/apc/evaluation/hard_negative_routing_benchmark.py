# ruff: noqa: E501
"""Phase B B-C004 hard-negative bank construction and diagnostic matrix.

This module is deliberately evaluation-only.  It builds ephemeral score-space
competitors around a frozen router/bank without modifying either object's
parameters or the function of any primitive.  In particular, the L4
competitor is a logical candidate that reuses a parameterized primitive with
an incorrect argument; it is never installed as another persistent primitive.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.consolidation.compact_consolidation import collate_content_only_batch
from apc.environments.generator import Example
from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS, get_operation
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.compute_accounting import verify_sparse_execution
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
    get_or_build_16_primitive_bank,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
    RouterReplayBuffer,
    align_shared_core_embeddings,
    update_router_incrementally,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)
from apc.primitives.router import Router, RouterConfig
from apc.utils.seed import set_seed

DEFAULT_BANK_SIZES: Final[tuple[int, ...]] = (16, 32, 64, 128)
DEFAULT_LEVELS: Final[tuple[HardNegativeLevel, ...]] = tuple(HardNegativeLevel)
# L4 is intentionally limited to parameterized operations: it must challenge
# arguments without manufacturing an argument-specific persistent primitive.
DEFAULT_TARGET_OPERATIONS: Final[tuple[str, ...]] = ("SHIFT", "SELECT", "COUNT", "BIND")

_RELATED_OPERATION: Final[dict[str, str]] = {
    "SHIFT": "CYCLE_FOUR",
    "SELECT": "BIND",
    "COUNT": "BIND",
    "BIND": "COUNT",
}


@dataclass(frozen=True)
class HardNegativeCandidate:
    """One evaluation-only candidate and its executable primitive identity."""

    candidate_id: str
    execute_primitive_id: int
    score_key: torch.Tensor
    provenance: str
    semantic_relation: str | None = None
    argument_override: dict[str, Any] | None = None


@dataclass(frozen=True)
class HardNegativeDiagnostic:
    """One cell in the required (bank size x difficulty) diagnostic matrix."""

    seed: int
    bank_size: int
    level: str
    episode_count: int
    candidate_count: int
    top1: float
    topk: float
    positive_negative_margin: float
    candidate_rank: float
    target_negative_cosine: float
    false_reuse: float
    false_plastic: float
    closed_loop_exact_match: float
    selected_forward_calls: int
    unselected_forward_calls: int
    sparse_execution_passed: bool
    primitive_functions_unchanged: bool
    router_unchanged: bool
    leak_audit_passed: bool
    bank_composition: dict[str, int]
    competitor_provenance: str
    semantic_relation: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class HardNegativeBenchmarkConfig:
    """Explicit B-C004 infrastructure configuration; no router updates occur after calibration."""

    seeds: tuple[int, ...] = (0,)
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    num_eval_examples: int = 32
    router_train_examples: int = 32
    router_steps: int = 250
    top_k: int = 5
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if self.num_eval_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["levels"] = [level.value for level in self.levels]
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        return data


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(p=2).clamp_min(1e-8)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(_unit(left), _unit(right)).item())


def _wrong_arguments(operation: str, example: Example, vocab_size: int) -> dict[str, Any]:
    """Return a valid but different argument value for a parameterized call."""
    assert example.task_spec is not None
    arguments = dict(example.task_spec.steps[0].arguments)
    if operation == "SHIFT":
        arguments["amount"] = (int(arguments["amount"]) + 1) % len(example.input_tokens)
    elif operation == "COUNT":
        arguments["target"] = (int(arguments["target"]) + 1) % vocab_size
    elif operation == "BIND":
        arguments["query_key"] = (int(arguments["query_key"]) + 1) % vocab_size
    elif operation == "SELECT":
        indices = list(arguments["indices"])
        arguments["indices"] = [(index + 1) % len(example.input_tokens) for index in indices]
        arguments["indices"].sort()
    else:
        raise ValueError(f"L4 requires a parameterized operation, got {operation}")
    return arguments


def _primitive_argument_value(operation: str, arguments: dict[str, Any]) -> Any:
    """Extract the raw single argument consumed by the established encoders."""
    if operation == "SELECT":
        return arguments.get("indices", [0])
    if operation == "SHIFT":
        return arguments.get("amount", 0)
    if operation == "COUNT":
        return arguments.get("target", 0)
    if operation == "BIND":
        return arguments.get("query_key", 0)
    raise ValueError(f"unexpected parameterized operation: {operation}")


def build_hard_negative_candidates(
    *,
    level: HardNegativeLevel,
    target_id: int,
    target_operation: str,
    candidate_ids: Sequence[int],
    keys_by_id: dict[int, torch.Tensor],
    operation_by_id: dict[int, str],
    seed: int,
) -> tuple[list[HardNegativeCandidate], HardNegativeCandidate]:
    """Build deterministic, non-mutating score-space competitors for one target.

    ``score_key`` belongs only to this diagnostic candidate list.  Neither the
    Router nor any Primitive parameter is edited by this function.
    """
    if target_id not in candidate_ids:
        raise ValueError("target_id must be a candidate")
    target_key = keys_by_id[target_id].detach().clone()
    other_ids = [pid for pid in candidate_ids if pid != target_id]
    if not other_ids:
        raise ValueError("a hard-negative benchmark needs at least two candidates")

    candidates = [
        HardNegativeCandidate(
            candidate_id=f"primitive:{pid}",
            execute_primitive_id=pid,
            score_key=keys_by_id[pid].detach().clone(),
            provenance="resident_primitive_key",
        )
        for pid in candidate_ids
    ]
    related_op = _RELATED_OPERATION[target_operation]
    related_id = next(
        (pid for pid in other_ids if operation_by_id.get(pid) == related_op), other_ids[0]
    )
    competitor_id = related_id
    gen = torch.Generator(device="cpu").manual_seed(seed * 100_003 + target_id * 97)
    random_key = torch.randn(target_key.shape, generator=gen, dtype=target_key.dtype)
    random_key = random_key.to(target_key.device)
    random_key = _unit(random_key) * target_key.norm(p=2)

    relation: str | None = None
    argument_override: dict[str, Any] | None = None
    if level == HardNegativeLevel.L0_ORTHOGONAL:
        # Projection removes the target-key component: an easy orthogonal control.
        orthogonal = random_key - torch.dot(random_key, _unit(target_key)) * _unit(target_key)
        score_key = _unit(orthogonal) * target_key.norm(p=2)
        provenance = "seeded_orthogonal_score_space"
    elif level == HardNegativeLevel.L1_RANDOM_SCORE_SPACE:
        score_key = random_key
        provenance = "seeded_random_score_space"
    elif level == HardNegativeLevel.L2_NEAR_NEIGHBOR:
        score_key = _unit(0.70 * target_key + 0.30 * random_key) * target_key.norm(p=2)
        provenance = "controlled_near_neighbor_score_space"
    elif level == HardNegativeLevel.L3_SEMANTICALLY_RELATED:
        related_key = keys_by_id[related_id].detach().clone()
        # Retain the learned semantic key as the dominant source, with a
        # deterministic target-proximity component so it is a real competitor.
        score_key = _unit(0.75 * related_key + 0.25 * target_key) * target_key.norm(p=2)
        provenance = "related_learned_primitive_key"
        relation = f"{target_operation}->{related_op}"
    elif level == HardNegativeLevel.L4_CONFUSABLE_FAMILY:
        # A virtual candidate shares the existing executable primitive but uses
        # an incorrect argument at execution time.  No persistent family is added.
        competitor_id = target_id
        score_key = _unit(1.01 * target_key + 0.01 * random_key) * target_key.norm(p=2)
        provenance = "same_parameterized_family_wrong_argument"
        relation = f"{target_operation}:argument_variant"
        argument_override = {"__wrong_argument__": True}
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"unsupported level: {level}")

    if level == HardNegativeLevel.L4_CONFUSABLE_FAMILY:
        competitor = HardNegativeCandidate(
            candidate_id=f"virtual:{target_id}:wrong_argument",
            execute_primitive_id=target_id,
            score_key=score_key,
            provenance=provenance,
            semantic_relation=relation,
            argument_override=argument_override,
        )
        candidates.append(competitor)
    else:
        competitor = HardNegativeCandidate(
            candidate_id=f"primitive:{competitor_id}",
            execute_primitive_id=competitor_id,
            score_key=score_key,
            provenance=provenance,
            semantic_relation=relation,
        )
        candidates = [
            competitor if candidate.execute_primitive_id == competitor_id else candidate
            for candidate in candidates
        ]
    return candidates, competitor


def _execute_selected_candidates(
    core: Any,
    bank: Any,
    operation_by_id: dict[int, str],
    examples: Sequence[Example],
    selected: Sequence[HardNegativeCandidate],
) -> list[tuple[int, ...]]:
    """Execute exactly the selected primitive per row and return token predictions."""
    device = core.device
    lengths = [len(example.input_tokens) for example in examples]
    out_lengths = [len(example.target_tokens) for example in examples]
    content_ids = collate_content_only_batch(examples, core.tokens, device=device)
    with torch.no_grad():
        state = core.model.encode(content_ids)
        h_content = state[:, 1 : 1 + max(lengths), :]

    result: list[tuple[int, ...] | None] = [None] * len(examples)
    grouped: dict[tuple[int, bool], list[int]] = {}
    for index, candidate in enumerate(selected):
        grouped.setdefault(
            (candidate.execute_primitive_id, candidate.argument_override is not None), []
        ).append(index)
    for (pid, use_wrong_arguments), indices in grouped.items():
        primitive = bank.get(pid)
        operation = operation_by_id[pid]
        op_def = get_operation(operation)
        args: list[Any] | None = None
        if op_def.required_argument_names:
            args = []
            for index in indices:
                example = examples[index]
                assert example.task_spec is not None
                if use_wrong_arguments:
                    wrong = _wrong_arguments(operation, example, core.tokens.env_vocab_size)
                    args.append(_primitive_argument_value(operation, wrong))
                else:
                    actual = dict(example.task_spec.steps[0].arguments)
                    args.append(_primitive_argument_value(operation, actual))
        sub_h = h_content[indices]
        sub_lengths = [lengths[index] for index in indices]
        sub_out_lengths = [out_lengths[index] for index in indices]
        with torch.no_grad():
            if isinstance(primitive, (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive)):
                logits = primitive(sub_h, sub_lengths, sub_out_lengths, args)
            else:
                logits = primitive(sub_h)
        tokens = logits.argmax(dim=-1)
        for local_index, global_index in enumerate(indices):
            result[global_index] = tuple(tokens[local_index, : sub_out_lengths[local_index]].tolist())
    assert all(prediction is not None for prediction in result)
    return [prediction for prediction in result if prediction is not None]


def evaluate_hard_negative_condition(
    *,
    core: Any,
    bank: Any,
    router: Router,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_operation: str,
    target_id: int,
    level: HardNegativeLevel,
    examples: Sequence[Example],
    seed: int,
    top_k: int,
    bank_composition: dict[str, int],
) -> HardNegativeDiagnostic:
    """Evaluate a frozen system against one deterministic hard-negative condition."""
    parameter_snapshot = {name: value.detach().clone() for name, value in router.state_dict().items()}
    primitive_snapshot = {
        pid: tuple(parameter.detach().clone() for parameter in bank.get(pid).parameters())
        for pid in bank.ids()
    }
    keys = {pid: router.key_parameter(pid).detach().clone() for pid in candidate_ids}
    candidates, competitor = build_hard_negative_candidates(
        level=level,
        target_id=target_id,
        target_operation=target_operation,
        candidate_ids=candidate_ids,
        keys_by_id=keys,
        operation_by_id=operation_by_id,
        seed=seed,
    )
    # Candidate records never enter the model.  The audit protects against
    # accidentally serializing evaluator-only oracle fields as runtime input.
    leak_audit_passed = all(
        "oracle" not in candidate.provenance.lower()
        and "family_id" not in candidate.provenance.lower()
        for candidate in candidates
    )
    z_task = extract_task_representations(core, list(examples))
    with torch.no_grad():
        query = router.query_proj(z_task)
        matrix = torch.stack([candidate.score_key.to(query.device) for candidate in candidates])
        scores = query @ matrix.transpose(0, 1)
        ordering = torch.argsort(scores, dim=-1, descending=True)
    correct_index = next(i for i, candidate in enumerate(candidates) if candidate is not competitor and candidate.execute_primitive_id == target_id)
    comp_index = next(index for index, candidate in enumerate(candidates) if candidate is competitor)
    ranks = (ordering == correct_index).nonzero(as_tuple=False)[:, 1] + 1
    top1 = float((ordering[:, 0] == correct_index).to(torch.float32).mean().item())
    topk = float((ordering[:, : min(top_k, len(candidates))] == correct_index).any(dim=-1).to(torch.float32).mean().item())
    margin = float((scores[:, correct_index] - scores[:, comp_index]).mean().item())
    selected = [candidates[int(index)] for index in ordering[:, 0].tolist()]
    for pid in bank.ids():
        bank.get(pid).reset_forward_call_count()
    predictions = _execute_selected_candidates(core, bank, operation_by_id, examples, selected)
    closed_loop_em = sum(
        prediction == example.target_tokens for prediction, example in zip(predictions, examples, strict=True)
    ) / len(examples)
    wrong_top1 = [candidate is not candidates[correct_index] for candidate in selected]
    # B-C004 does not run functional acceptance; this is the conservative
    # direct-reuse diagnostic (B-C005 will separate acceptance explicitly).
    false_reuse = sum(wrong_top1) / len(examples)
    false_plastic = 0.0
    selected_ids = {candidate.execute_primitive_id for candidate in selected}
    sparse_ok, _ = verify_sparse_execution(bank, selected_ids)
    selected_calls = sum(bank.get(pid).forward_call_count for pid in selected_ids)
    unselected_calls = sum(
        bank.get(pid).forward_call_count for pid in bank.ids() if pid not in selected_ids
    )
    router_unchanged = all(torch.equal(value, router.state_dict()[name]) for name, value in parameter_snapshot.items())
    primitive_functions_unchanged = all(
        all(torch.equal(before, after) for before, after in zip(primitive_snapshot[pid], bank.get(pid).parameters(), strict=True))
        for pid in bank.ids()
    )
    return HardNegativeDiagnostic(
        seed=seed,
        bank_size=len(bank),
        level=level.value,
        episode_count=len(examples),
        candidate_count=len(candidates),
        top1=top1,
        topk=topk,
        positive_negative_margin=margin,
        candidate_rank=float(ranks.to(torch.float32).mean().item()),
        target_negative_cosine=_cosine(candidates[correct_index].score_key, competitor.score_key),
        false_reuse=false_reuse,
        false_plastic=false_plastic,
        closed_loop_exact_match=closed_loop_em,
        selected_forward_calls=selected_calls,
        unselected_forward_calls=unselected_calls,
        sparse_execution_passed=sparse_ok and unselected_calls == 0,
        primitive_functions_unchanged=primitive_functions_unchanged,
        router_unchanged=router_unchanged,
        leak_audit_passed=leak_audit_passed,
        bank_composition=bank_composition,
        competitor_provenance=competitor.provenance,
        semantic_relation=competitor.semantic_relation,
    )


def _build_frozen_base_system(seed: int, config: HardNegativeBenchmarkConfig) -> tuple[Any, Any, Router, dict[str, int]]:
    """Load frozen A2 components, calibrating the baseline router once before diagnostics."""
    device = "cuda" if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()) else "cpu"
    architecture = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(seed=seed, vocab_size=10, device=device)
    )
    core = architecture.core
    checkpoint = Path("runs/phase_a1_shift_compact_structural_probe") / f"seed_{seed}" / "shared_encoder.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=core.device, weights_only=True)
        state = align_shared_core_embeddings(core.model, state, None, core.tokens)
        core.model.load_state_dict(state)
    core.model.eval()
    for parameter in core.model.parameters():
        parameter.requires_grad_(False)
    bank, op_to_id = get_or_build_16_primitive_bank(core, seed, output_dir=config.bank_checkpoint_dir)
    operations = tuple(INITIAL_10_OPERATIONS) + tuple(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)
    candidate_ids = [op_to_id[name] for name in operations]
    router = Router(RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot")).to(core.device)
    training: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for operation in operations:
        pid = op_to_id[operation]
        examples = generate_benchmark_examples(seed * 100 + pid, config.router_train_examples, operation=operation, split="train")
        z_task = extract_task_representations(core, examples)
        training[pid] = [(z_task[index], pid) for index in range(len(z_task))]
    update_router_incrementally(
        router,
        candidate_ids=candidate_ids,
        new_primitive_ids=candidate_ids,
        new_data_by_pid=training,
        replay_buffer=RouterReplayBuffer(max_per_class=32, max_total=512),
        config=IncrementalRouterConfig(condition=IncrementalUpdateCondition.R0_FULL_RETRAIN, router_steps=config.router_steps, seed=seed),
        all_historical_data_by_pid=training,
        device=core.device,
    )
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    return core, bank, router, op_to_id


def run_hard_negative_diagnostic(config: HardNegativeBenchmarkConfig) -> dict[str, Any]:
    """Run the complete B-C004 construction matrix and write portable artifacts."""
    start = time.perf_counter()
    diagnostics: list[HardNegativeDiagnostic] = []
    for seed in config.seeds:
        set_seed(seed)
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(seed, config)
        for bank_size in config.bank_sizes:
            bank, router, candidate_ids, semantic_ids, distractor_ids = build_scaled_bank_and_router(
                core, base_bank, base_router, op_to_id, bank_size, seed=seed
            )
            operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
            operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})
            composition = {
                "real_semantic_primitives": len(semantic_ids),
                "consolidated_primitives": 0,
                "hard_negative_learned_competitors": 0,
                "synthetic_distractors": len(distractor_ids),
            }
            for operation in config.target_operations:
                target_id = op_to_id[operation]
                examples = generate_benchmark_examples(
                    seed * 10_000 + bank_size, config.num_eval_examples, operation=operation, split="test"
                )
                for level in config.levels:
                    diagnostics.append(evaluate_hard_negative_condition(
                        core=core, bank=bank, router=router, candidate_ids=candidate_ids,
                        operation_by_id=operation_by_id, target_operation=operation,
                        target_id=target_id, level=level, examples=examples, seed=seed,
                        top_k=config.top_k, bank_composition=composition,
                    ))
    expected = len(config.seeds) * len(config.bank_sizes) * len(config.target_operations) * len(config.levels)
    matrix_complete = len(diagnostics) == expected
    level_counts = {level.value: sum(item.level == level.value for item in diagnostics) for level in config.levels}
    deterministic = all(item.router_unchanged and item.primitive_functions_unchanged for item in diagnostics)
    leak_free = all(item.leak_audit_passed for item in diagnostics)
    sparse = all(item.sparse_execution_passed for item in diagnostics)
    ordering_measurable = len({round(item.target_negative_cosine, 5) for item in diagnostics}) >= 3
    report = {
        "task_id": "B-C004",
        "config": config.to_dict(),
        "matrix": [item.to_dict() for item in diagnostics],
        "summary": {
            "expected_cells": expected,
            "actual_cells": len(diagnostics),
            "level_counts": level_counts,
            "matrix_complete": matrix_complete,
            "difficulty_ordering_measurable": ordering_measurable,
            "deterministic_non_mutating": deterministic,
            "leak_audit_passed": leak_free,
            "zero_unselected_forward_calls": sparse,
            "infrastructure_passed": matrix_complete and ordering_measurable and deterministic and leak_free and sparse,
            "elapsed_seconds": time.perf_counter() - start,
        },
    }
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (config.output_dir / "summary.json").write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
        rows = "\n".join(
            f"| {item.seed} | {item.bank_size} | {item.level} | {item.top1:.3f} | {item.topk:.3f} | {item.positive_negative_margin:.3f} | {item.candidate_rank:.2f} | {item.closed_loop_exact_match:.3f} | {item.unselected_forward_calls} |"
            for item in diagnostics
        )
        (config.output_dir / "report.md").write_text(
            "# B-C004 Hard-negative diagnostic matrix\n\n"
            "Infrastructure-only result; this is not the B2 scientific gate.\n\n"
            "| Seed | N | Level | Top-1 | Top-k | Margin | Target rank | Closed-loop EM | Unselected calls |\n"
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|\n" + rows + "\n",
            encoding="utf-8",
        )
    return report
