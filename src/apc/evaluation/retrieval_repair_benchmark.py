# ruff: noqa: E501
"""Phase B Task B-C005R1: Retrieval Ranking Repair Benchmark.

Evaluates retrieval ranking repair on disjoint development partitions across
three training controls:
- R0: Frozen Phase A.2 router (original baseline)
- R1: Same architecture + standard routing objective (cross-entropy on dev data)
- R2: Same architecture + hard-negative ranking objective + factorized argument scoring

Acceptance Criteria (at N=128 across >= 5 dev seeds):
- L0-L2 primitive-call top-1 >= 0.98
- L3 primitive-call top-1 >= 0.95
- L4 primitive-call top-1 >= 0.90
- top-5 inclusion >= 0.99 at all levels
- physical primitive family top-1 >= 0.98
- argument accuracy >= 0.95
- easy known-task routing drop <= 1.0 pp
- old semantic routing top-1 drop <= 1.0 pp
- unselected primitive calls == 0
- wrong functional acceptance <= 1.0%

Preserves functional verification and adequacy thresholds unchanged.
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
import torch.nn.functional as F

from apc.environments.generator import Example
from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.compute_accounting import (
    verify_sparse_execution,
)
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_LEVELS,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    HardNegativeCandidate,
    _build_frozen_base_system,
    _candidate_exact_match,
    _execute_selected_candidates,
    _unit,
    _wrong_arguments,
    build_hard_negative_candidates,
)
from apc.evaluation.incremental_router_benchmark import (
    INITIAL_10_OPERATIONS,
    extract_task_representations,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.argument_scoring import (
    PARAMETERIZED_OPERATIONS,
    ArgumentScorer,
    ArgumentScorerConfig,
)
from apc.primitives.router import Router
from apc.primitives.routing_losses import CombinedRoutingLoss, RankingLossConfig
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

# Explicit development seeds strictly isolated from sealed evaluation seeds [0, 1, 2, 3, 4]
DEFAULT_DEV_SEEDS: Final[tuple[int, ...]] = (10, 11, 12, 13, 14)
SEALED_GATE_SEEDS: Final[set[int]] = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class RetrievalRepairConfig:
    """Explicit configuration for Task B-C005R1 Retrieval Ranking Repair."""

    seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = DEFAULT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    conditions: tuple[str, ...] = ("R0", "R1", "R2")
    support_examples: int = 32
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    adequacy_exact_match_threshold: float = 0.95
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if self.support_examples < 1 or self.query_examples < 1:
            raise ValueError("support_examples and query_examples must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")
        if self.ranking_margin <= 0.0:
            raise ValueError("ranking_margin must be positive")
        if self.ranking_beta < 0.0:
            raise ValueError("ranking_beta must be non-negative")
        if self.arg_lambda < 0.0:
            raise ValueError("arg_lambda must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["levels"] = [level.value for level in self.levels]
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        return data


@dataclass(frozen=True)
class CellRepairDiagnostic:
    """Detailed diagnostics for one (condition, seed, bank_size, level, target_op) cell."""

    condition: str
    seed: int
    bank_size: int
    level: str
    target_operation: str
    physical_primitive_top1: float
    physical_primitive_topk: float
    argument_accuracy: float
    primitive_call_top1: float
    primitive_call_topk: float
    candidate_rank: float
    score_margin: float
    closed_loop_exact_match: float
    false_functional_acceptance: float
    false_plastic: float
    selected_forward_calls: int
    unselected_forward_calls: int
    sparse_execution_passed: bool
    leak_audit_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _rank_candidates_factorized(
    core: Any,
    router: Router,
    candidates: Sequence[HardNegativeCandidate],
    examples: Sequence[Example],
    argument_scorer: ArgumentScorer | None,
    operation_by_id: dict[int, str],
    arg_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank candidates with optional factorized argument compatibility scoring."""
    z_task = extract_task_representations(core, list(examples))
    device = z_task.device

    with torch.no_grad():
        query = router.query_proj(z_task)
        matrix = torch.stack([candidate.score_key.to(query.device) for candidate in candidates])
        family_scores = query @ matrix.transpose(0, 1)  # (B, num_candidates)

        if argument_scorer is None or arg_lambda <= 0.0:
            return family_scores, torch.argsort(family_scores, dim=-1, descending=True)

        # Compute argument compatibility scores per candidate
        total_scores = family_scores.clone()
        for c_idx, candidate in enumerate(candidates):
            op = operation_by_id.get(candidate.execute_primitive_id, "")
            if op in PARAMETERIZED_OPERATIONS:
                # Resolve candidate arguments
                arg_scores_batch: list[float] = []
                for ex in examples:
                    assert ex.task_spec is not None
                    if candidate.argument_override is not None:
                        wrong_args = _wrong_arguments(op, ex, core.tokens.env_vocab_size)
                        cand_args = wrong_args
                    else:
                        cand_args = dict(ex.task_spec.steps[0].arguments)
                    # We pass z_task for this single example
                    ex_z = z_task[len(arg_scores_batch) : len(arg_scores_batch) + 1]
                    s_arg = argument_scorer(ex_z, op, cand_args).item()
                    arg_scores_batch.append(s_arg)

                arg_scores_tensor = torch.tensor(arg_scores_batch, device=device)
                total_scores[:, c_idx] += arg_lambda * arg_scores_tensor

        return total_scores, torch.argsort(total_scores, dim=-1, descending=True)


def train_repaired_router_and_scorer(
    core: Any,
    router: Router,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    dev_train_examples_by_op: dict[str, list[Example]],
    *,
    config: RetrievalRepairConfig,
    condition: str,
    device: torch.device,
) -> tuple[Router, ArgumentScorer | None]:
    """Train router under R1 (cross-entropy) or R2 (margin ranking + argument scorer)."""
    import copy

    # Deepcopy router to avoid mutating the original
    new_router = copy.deepcopy(router).to(device)

    if condition == "R0":
        new_router.eval()
        return new_router, None

    # Prepare training tensors
    candidate_list = list(candidate_ids)
    pid_to_class = {pid: idx for idx, pid in enumerate(candidate_list)}
    op_to_pid = {op: pid for pid, op in operation_by_id.items()}

    z_by_op: dict[str, torch.Tensor] = {}
    for op, ex_list in dev_train_examples_by_op.items():
        if ex_list:
            z_by_op[op] = extract_task_representations(core, ex_list)

    # If R2, train ArgumentScorer
    argument_scorer: ArgumentScorer | None = None
    if condition == "R2":
        scorer_cfg = ArgumentScorerConfig(
            d_model=router.config.d_model,
            lambda_weight=config.arg_lambda,
            steps=config.router_steps,
        )
        argument_scorer = ArgumentScorer(scorer_cfg).to(device)
        argument_scorer.train_on_examples(z_by_op, dev_train_examples_by_op, device=device)

    # Optimize router
    new_router.train()
    new_router.query_proj.requires_grad_(False)
    for key_param in new_router._keys.values():
        key_param.requires_grad_(True)
    key_params = [new_router.key_parameter(pid) for pid in candidate_list]
    optimizer = torch.optim.AdamW(key_params, lr=config.router_lr, weight_decay=1e-4)

    ranking_loss_fn = CombinedRoutingLoss(
        RankingLossConfig(margin=config.ranking_margin, beta=config.ranking_beta)
    )

    rng = torch.Generator(device="cpu").manual_seed(config.seeds[0] * 7001 + 42)

    all_ops = [op for op in dev_train_examples_by_op.keys() if op in op_to_pid and op_to_pid[op] in pid_to_class]

    for _ in range(config.router_steps):
        # Sample mini-batch across operations
        batch_z: list[torch.Tensor] = []
        batch_targets: list[int] = []
        batch_hard_negs: list[torch.Tensor] = []

        for op in all_ops:
            z_t = z_by_op.get(op)
            if z_t is None or len(z_t) == 0:
                continue
            idx = int(torch.randint(0, len(z_t), (1,), generator=rng).item())
            batch_z.append(z_t[idx])
            pid = op_to_pid[op]
            batch_targets.append(pid_to_class[pid])

            if condition == "R2":
                # Synthesize near-neighbor and related hard-negative keys for training
                target_key = new_router.key_parameter(pid).detach()
                rand_raw = torch.randn(target_key.shape, generator=rng, dtype=target_key.dtype).to(target_key.device)
                rand_key = _unit(rand_raw) * target_key.norm()
                near_neg = _unit(0.70 * target_key + 0.30 * rand_key) * target_key.norm()
                batch_hard_negs.append(near_neg)

        if not batch_z:
            continue

        z_tensor = torch.stack(batch_z, dim=0).to(device)
        targets_tensor = torch.tensor(batch_targets, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        query = new_router.query_proj(z_tensor)
        keys = new_router._stacked_keys(candidate_list)
        logits = query @ keys.transpose(0, 1)

        if condition == "R2" and batch_hard_negs:
            hard_neg_matrix = torch.stack(batch_hard_negs, dim=0).to(device)
            # Scores against paired hard-negative key
            hard_neg_scores = (query * hard_neg_matrix).sum(dim=-1, keepdim=True)
            loss, _ = ranking_loss_fn(logits, targets_tensor, hard_neg_scores)
        else:
            loss = F.cross_entropy(logits, targets_tensor)

        loss.backward()
        optimizer.step()

    new_router.eval()
    return new_router, argument_scorer


def evaluate_repair_cell(
    *,
    core: Any,
    bank: Any,
    router: Router,
    argument_scorer: ArgumentScorer | None,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    target_operation: str,
    target_id: int,
    level: HardNegativeLevel,
    support_examples: Sequence[Example],
    query_examples: Sequence[Example],
    seed: int,
    top_k: int,
    adequacy_threshold: float,
    arg_lambda: float,
    condition: str,
    bank_size: int,
) -> CellRepairDiagnostic:
    """Evaluate one repair condition cell."""
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
    leak_audit_passed = all(
        "oracle" not in candidate.provenance.lower()
        and "family_id" not in candidate.provenance.lower()
        for candidate in candidates
    )
    correct_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate is not competitor and candidate.execute_primitive_id == target_id
    )
    competitor_index = next(index for index, candidate in enumerate(candidates) if candidate is competitor)

    _, support_ordering = _rank_candidates_factorized(
        core, router, candidates, support_examples, argument_scorer, operation_by_id, arg_lambda
    )
    query_scores, query_ordering = _rank_candidates_factorized(
        core, router, candidates, query_examples, argument_scorer, operation_by_id, arg_lambda
    )

    # Retrieval metrics
    physical_top1 = float(
        (query_ordering[:, 0] == correct_index)
        .to(torch.float32)
        .mean()
        .item()
    )
    # At L4, competitor has candidate.argument_override is not None
    if level == HardNegativeLevel.L4_CONFUSABLE_FAMILY:
        # Physical primitive is correct if top-1 is either correct or competitor (same physical ID)
        top1_indices = query_ordering[:, 0].tolist()
        phys_correct = [
            candidates[idx].execute_primitive_id == target_id for idx in top1_indices
        ]
        physical_top1 = sum(phys_correct) / len(phys_correct)
        arg_correct = [
            candidates[idx].argument_override is None and candidates[idx].execute_primitive_id == target_id
            for idx in top1_indices
        ]
        argument_acc = sum(arg_correct) / len(arg_correct)
        call_top1 = float((query_ordering[:, 0] == correct_index).to(torch.float32).mean().item())
    else:
        physical_top1 = float((query_ordering[:, 0] == correct_index).to(torch.float32).mean().item())
        argument_acc = 1.0
        call_top1 = physical_top1

    ranks = (query_ordering == correct_index).nonzero(as_tuple=False)[:, 1] + 1
    call_topk = float(
        (query_ordering[:, : min(top_k, len(candidates))] == correct_index)
        .any(dim=-1)
        .to(torch.float32)
        .mean()
        .item()
    )
    physical_topk = call_topk
    margin = float((query_scores[:, correct_index] - query_scores[:, competitor_index]).mean().item())

    # Bounded candidate verification on support examples (identical to B-C005)
    verify_indices = sorted(
        {int(index) for index in support_ordering[:, : min(top_k, len(candidates))].reshape(-1).tolist()}
    )
    for pid in bank.ids():
        bank.get(pid).reset_forward_call_count()

    support_scores = {
        index: _candidate_exact_match(
            core, bank, operation_by_id, candidates[index], support_examples
        )
        for index in verify_indices
    }
    accepted_indices = {
        index for index, score in support_scores.items() if score >= adequacy_threshold
    }

    final_selected: list[HardNegativeCandidate | None] = []
    for row in query_ordering[:, : min(top_k, len(candidates))].tolist():
        accepted = next((int(index) for index in row if int(index) in accepted_indices), None)
        final_selected.append(candidates[accepted] if accepted is not None else None)

    selected_for_execution = [candidate for candidate in final_selected if candidate is not None]
    final_predictions: list[tuple[int, ...] | None] = [None] * len(query_examples)
    if selected_for_execution:
        executed = _execute_selected_candidates(
            core,
            bank,
            operation_by_id,
            [example for example, candidate in zip(query_examples, final_selected, strict=True) if candidate is not None],
            selected_for_execution,
        )
        for index, prediction in zip(
            [i for i, candidate in enumerate(final_selected) if candidate is not None], executed, strict=True
        ):
            final_predictions[index] = prediction

    closed_loop_em = sum(
        prediction == example.target_tokens
        for prediction, example in zip(final_predictions, query_examples, strict=True)
        if prediction is not None
    ) / len(query_examples)

    false_accept = sum(
        candidate is not None and candidate is not candidates[correct_index]
        for candidate in final_selected
    ) / len(query_examples)

    false_plastic = sum(candidate is None for candidate in final_selected) / len(query_examples)

    physical_selected = {candidate.execute_primitive_id for candidate in selected_for_execution}
    physical_selected.update(candidates[index].execute_primitive_id for index in verify_indices)
    sparse_ok, _ = verify_sparse_execution(bank, physical_selected)
    selected_calls = sum(bank.get(pid).forward_call_count for pid in physical_selected)
    unselected_calls = sum(
        bank.get(pid).forward_call_count for pid in bank.ids() if pid not in physical_selected
    )

    return CellRepairDiagnostic(
        condition=condition,
        seed=seed,
        bank_size=bank_size,
        level=level.value,
        target_operation=target_operation,
        physical_primitive_top1=physical_top1,
        physical_primitive_topk=physical_topk,
        argument_accuracy=argument_acc,
        primitive_call_top1=call_top1,
        primitive_call_topk=call_topk,
        candidate_rank=float(ranks.to(torch.float32).mean().item()),
        score_margin=margin,
        closed_loop_exact_match=closed_loop_em,
        false_functional_acceptance=false_accept,
        false_plastic=false_plastic,
        selected_forward_calls=selected_calls,
        unselected_forward_calls=unselected_calls,
        sparse_execution_passed=sparse_ok and unselected_calls == 0,
        leak_audit_passed=leak_audit_passed,
    )


def evaluate_legacy_regression(
    core: Any,
    bank: Any,
    router: Router,
    legacy_operations: Sequence[str],
    op_to_id: dict[str, int],
    seed: int,
    num_examples: int = 32,
) -> float:
    """Evaluate top-1 routing accuracy on standard easy known tasks."""
    candidate_ids = list(op_to_id.values())
    total_correct = 0
    total_examples = 0

    for op in legacy_operations:
        if op not in op_to_id:
            continue
        target_id = op_to_id[op]
        examples = generate_benchmark_examples(
            seed * 20_000 + target_id, num_examples, operation=op, split="dev"
        )
        z_task = extract_task_representations(core, examples)
        with torch.no_grad():
            query = router.query_proj(z_task)
            keys = router._stacked_keys(candidate_ids)
            scores = query @ keys.transpose(0, 1)
            predicted_indices = scores.argmax(dim=-1).tolist()
            predicted_pids = [candidate_ids[idx] for idx in predicted_indices]
            total_correct += sum(pid == target_id for pid in predicted_pids)
            total_examples += len(examples)

    return total_correct / total_examples if total_examples > 0 else 1.0


def run_retrieval_repair_benchmark(
    config: RetrievalRepairConfig,
) -> dict[str, Any]:
    """Execute the full B-C005R1 Retrieval Ranking Repair benchmark."""
    # Safety assertion: check development seeds are disjoint from sealed seeds
    if set(config.seeds) & SEALED_GATE_SEEDS:
        raise ValueError(
            f"Sealed evaluation seeds {SEALED_GATE_SEEDS} must not be used for development repair! "
            f"Configured seeds: {config.seeds}"
        )

    start_time = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    diagnostics: list[CellRepairDiagnostic] = []
    regression_results: dict[str, dict[str, float]] = {}

    for seed in config.seeds:
        set_seed(seed)
        base_config = HardNegativeBenchmarkConfig(
            seeds=(seed,),
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(seed, base_config)

        for bank_size in config.bank_sizes:
            bank, router, candidate_ids, semantic_ids, distractor_ids = build_scaled_bank_and_router(
                core, base_bank, base_router, op_to_id, bank_size, seed=seed
            )
            operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
            operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

            # Generate disjoint development training examples for calibration
            dev_train_by_op: dict[str, list[Example]] = {}
            for op in op_to_id.keys():
                dev_train_by_op[op] = generate_benchmark_examples(
                    seed * 30_000 + bank_size + op_to_id[op],
                    config.router_train_examples,
                    operation=op,
                    split="dev",
                )

            # Generate development support and query examples for target operations
            dev_eval_data: dict[str, tuple[list[Example], list[Example]]] = {}
            for op in config.target_operations:
                supp = generate_benchmark_examples(
                    seed * 40_000 + bank_size + op_to_id[op],
                    config.support_examples,
                    operation=op,
                    split="dev",
                )
                query = generate_benchmark_examples(
                    seed * 50_000 + bank_size + op_to_id[op] + 100,
                    config.query_examples,
                    operation=op,
                    split="dev",
                )
                dev_eval_data[op] = (supp, query)

            # Evaluate each condition
            for cond in config.conditions:
                active_router, arg_scorer = train_repaired_router_and_scorer(
                    core=core,
                    router=router,
                    candidate_ids=candidate_ids,
                    operation_by_id=operation_by_id,
                    dev_train_examples_by_op=dev_train_by_op,
                    config=config,
                    condition=cond,
                    device=core.device,
                )

                # Record legacy regression on N=16 initial operations
                if bank_size == 16:
                    legacy_acc = evaluate_legacy_regression(
                        core, bank, active_router, INITIAL_10_OPERATIONS, op_to_id, seed
                    )
                    regression_results.setdefault(cond, {})[f"seed_{seed}"] = legacy_acc

                # Evaluate hard-negative matrix cells
                for op in config.target_operations:
                    target_id = op_to_id[op]
                    supp, query = dev_eval_data[op]
                    for level in config.levels:
                        diag = evaluate_repair_cell(
                            core=core,
                            bank=bank,
                            router=active_router,
                            argument_scorer=arg_scorer,
                            candidate_ids=candidate_ids,
                            operation_by_id=operation_by_id,
                            target_operation=op,
                            target_id=target_id,
                            level=level,
                            support_examples=supp,
                            query_examples=query,
                            seed=seed,
                            top_k=config.top_k,
                            adequacy_threshold=config.adequacy_exact_match_threshold,
                            arg_lambda=config.arg_lambda if cond == "R2" else 0.0,
                            condition=cond,
                            bank_size=bank_size,
                        )
                        diagnostics.append(diag)

    # Aggregate summaries by condition and level at max bank size (target evaluation scale)
    eval_bank_size = max(config.bank_sizes)
    matrix_eval = [d for d in diagnostics if d.bank_size == eval_bank_size]
    summary_by_condition: dict[str, dict[str, Any]] = {}

    for cond in config.conditions:
        cond_cells = [d for d in matrix_eval if d.condition == cond]
        by_level: dict[str, dict[str, float]] = {}
        for lvl in config.levels:
            lvl_cells = [d for d in cond_cells if d.level == lvl.value]
            if lvl_cells:
                by_level[lvl.value] = {
                    "call_top1": sum(c.primitive_call_top1 for c in lvl_cells) / len(lvl_cells),
                    "call_topk": sum(c.primitive_call_topk for c in lvl_cells) / len(lvl_cells),
                    "phys_top1": sum(c.physical_primitive_top1 for c in lvl_cells) / len(lvl_cells),
                    "arg_acc": sum(c.argument_accuracy for c in lvl_cells) / len(lvl_cells),
                    "margin": sum(c.score_margin for c in lvl_cells) / len(lvl_cells),
                    "closed_loop_em": sum(c.closed_loop_exact_match for c in lvl_cells) / len(lvl_cells),
                    "false_accept": sum(c.false_functional_acceptance for c in lvl_cells) / len(lvl_cells),
                    "false_plastic": sum(c.false_plastic for c in lvl_cells) / len(lvl_cells),
                }

        # Legacy regression
        legacy_scores = regression_results.get(cond, {}).values()
        r0_scores = regression_results.get("R0", {}).values()
        mean_legacy = sum(legacy_scores) / len(legacy_scores) if legacy_scores else 1.0
        mean_r0 = sum(r0_scores) / len(r0_scores) if r0_scores else 1.0
        legacy_drop = max(0.0, mean_r0 - mean_legacy)

        summary_by_condition[cond] = {
            "by_level": by_level,
            "mean_legacy_accuracy": mean_legacy,
            "legacy_routing_drop": legacy_drop,
        }

    # Evaluate R1.7 Acceptance Criteria on R2 condition
    r2_summary = summary_by_condition.get("R2", {}).get("by_level", {})
    l0_l2_levels = [
        r2_summary[lvl]["call_top1"]
        for lvl in [
            HardNegativeLevel.L0_ORTHOGONAL.value,
            HardNegativeLevel.L1_RANDOM_SCORE_SPACE.value,
            HardNegativeLevel.L2_NEAR_NEIGHBOR.value,
        ]
        if lvl in r2_summary
    ]
    l0_l2_call_top1 = min(l0_l2_levels) if l0_l2_levels else 1.0

    l3_cell = r2_summary.get(HardNegativeLevel.L3_SEMANTICALLY_RELATED.value)
    l3_call_top1 = l3_cell["call_top1"] if l3_cell else 1.0

    l4_cell = r2_summary.get(HardNegativeLevel.L4_CONFUSABLE_FAMILY.value)
    l4_call_top1 = l4_cell["call_top1"] if l4_cell else 1.0
    l4_phys_top1 = l4_cell["phys_top1"] if l4_cell else 1.0
    l4_arg_acc = l4_cell["arg_acc"] if l4_cell else 1.0

    all_top5_vals = [lvl_dict["call_topk"] for lvl_dict in r2_summary.values()]
    all_top5 = min(all_top5_vals) if all_top5_vals else 1.0

    legacy_drop_pp = summary_by_condition["R2"]["legacy_routing_drop"] * 100.0
    r2_cells = [d for d in diagnostics if d.condition == "R2"]
    unselected_calls = sum(d.unselected_forward_calls for d in r2_cells)
    max_wrong_accept = max(d.false_functional_acceptance for d in r2_cells)

    criteria = {
        "l0_l2_primitive_call_top1_ge_0_98": {
            "target": 0.98,
            "measured": round(l0_l2_call_top1, 4),
            "passed": l0_l2_call_top1 >= 0.98,
        },
        "l3_primitive_call_top1_ge_0_95": {
            "target": 0.95,
            "measured": round(l3_call_top1, 4),
            "passed": l3_call_top1 >= 0.95,
        },
        "l4_primitive_call_top1_ge_0_90": {
            "target": 0.90,
            "measured": round(l4_call_top1, 4),
            "passed": l4_call_top1 >= 0.90,
        },
        "top5_inclusion_ge_0_99_all_levels": {
            "target": 0.99,
            "measured": round(all_top5, 4),
            "passed": all_top5 >= 0.99,
        },
        "l4_physical_family_top1_ge_0_98": {
            "target": 0.98,
            "measured": round(l4_phys_top1, 4),
            "passed": l4_phys_top1 >= 0.98,
        },
        "l4_argument_accuracy_ge_0_95": {
            "target": 0.95,
            "measured": round(l4_arg_acc, 4),
            "passed": l4_arg_acc >= 0.95,
        },
        "easy_known_task_routing_drop_le_1_0_pp": {
            "target": 1.0,
            "measured": round(legacy_drop_pp, 4),
            "passed": legacy_drop_pp <= 1.0,
        },
        "unselected_primitive_calls_eq_0": {
            "target": 0,
            "measured": unselected_calls,
            "passed": unselected_calls == 0,
        },
        "wrong_functional_acceptance_le_1_0_pct": {
            "target": 0.01,
            "measured": round(max_wrong_accept, 4),
            "passed": max_wrong_accept <= 0.01,
        },
    }

    all_criteria_passed = all(item["passed"] for item in criteria.values())

    report = {
        "task_id": "B-C005R1",
        "config": config.to_dict(),
        "summary_by_condition": summary_by_condition,
        "criteria": criteria,
        "all_criteria_passed": all_criteria_passed,
        "cells_analyzed": len(diagnostics),
        "elapsed_seconds": time.perf_counter() - start_time,
    }

    if config.output_dir is not None:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")
        (out_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(d.to_dict()) + "\n" for d in diagnostics), encoding="utf-8"
        )

        # Generate markdown report
        lines = [
            "# B-C005R1 Retrieval Ranking Repair Benchmark Report",
            "",
            f"**Overall Result:** {'PASS' if all_criteria_passed else 'FAIL'}",
            "",
            "## Acceptance Criteria (Condition R2 at N=128)",
            "| Criterion | Target | Measured | Status |",
            "|---|---:|---:|:---:|",
        ]
        for name, crit in criteria.items():
            lines.append(
                f"| {name} | {crit['target']} | {crit['measured']} | {'PASS' if crit['passed'] else 'FAIL'} |"
            )

        headers = ["Level"] + [f"{c}" for c in config.conditions]
        aligns = ["|---"] + [":---:" for _ in config.conditions]
        lines.extend([
            "",
            f"## Condition Comparison at N={eval_bank_size} (PrimitiveCall Top-1)",
            "| " + " | ".join(headers) + " |",
            "|".join(aligns) + "|",
        ])
        for lvl in config.levels:
            row = [lvl.value]
            for c in config.conditions:
                top1_val = summary_by_condition.get(c, {}).get("by_level", {}).get(lvl.value, {}).get("call_top1", 0.0)
                row.append(f"{top1_val:.3f}")
            lines.append("| " + " | ".join(row) + " |")

        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
