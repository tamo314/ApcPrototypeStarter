# ruff: noqa: E501
"""Evaluation-only L3 representation-stage localization for B-C005D2-002.

B-C005G (ADR-0079) failed the L3 (semantically related) hard-negative cell on
the new sealed partition (top-1 0.6555) while the frozen `query_proj` never
changes across the R0/R1/R2 repair conditions -- only the router's per
primitive key vectors are retrained (development-only). This module localizes
*where* the L3 semantic discrimination is lost along:

    TaskSpec -> Task Encoder -> z_task -> query_proj -> q_task -> key scoring

without changing the runtime path: it only reads frozen `z_task`/`q_task`
values and fits small evaluation-only linear probes (never fed back into
the router or any primitive).

Cross-checkpoint caveat
------------------------
`development` seeds (10-14) have no pretrained checkpoint under
`runs/phase_a1_shift_compact_structural_probe/`, so `_build_frozen_base_system`
silently falls back to a fresh random initialization for those seeds -- each
development seed is therefore an independently initialized core, not
comparable in raw coordinates to any other seed's `z_task`/`q_task`, nor to
the pretrained sealed cores (seeds 0-4). `regate_sealed` seed `20 + i` reuses
the *same* pretrained core as `original_sealed` seed `i` (`model_seed = seed
% 5`), so those two partitions ARE directly comparable per matched core index.

Consequently every probe here is fit and evaluated **within one core**
(same seed) using disjoint probe-only example splits, never pooled across
seeds/cores. Only the resulting *scalar accuracy* per core is aggregated
across seeds -- comparing accuracy distributions is valid even though the
underlying feature spaces are not shared.
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
from torch import nn

from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    _RELATED_OPERATION,
    DEFAULT_BANK_SIZES,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
    _cosine,
    build_hard_negative_candidates,
)
from apc.evaluation.hard_negative_second_diagnostic import _training_examples
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
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

_PRE_REPAIR: Final[str] = "R0_frozen_pre_repair"
_POST_REPAIR: Final[str] = "R2_frozen_post_repair"
_L3: Final[HardNegativeLevel] = HardNegativeLevel.L3_SEMANTICALLY_RELATED


@dataclass(frozen=True)
class RepresentationStageProbeConfig:
    """Frozen protocol for the D2-002 L3 representation-stage localization."""

    original_sealed_seeds: tuple[int, ...] = tuple(sorted(SEALED_GATE_SEEDS))
    development_seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    regate_sealed_seeds: tuple[int, ...] = DEFAULT_REGATE_SEEDS
    bank_size: int = 128
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    query_examples: int = 64
    probe_train_examples: int = 64
    probe_eval_examples: int = 64
    probe_steps: int = 300
    probe_lr: float = 0.05
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    separability_threshold: float = 0.90
    shuffled_chance_tolerance: float = 0.15
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
        if self.query_examples < 1 or self.probe_train_examples < 1 or self.probe_eval_examples < 1:
            raise ValueError("example counts must be positive")
        if self.probe_steps < 1 or self.router_steps < 1:
            raise ValueError("optimization step counts must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if not 0.5 < self.separability_threshold <= 1.0:
            raise ValueError("separability_threshold must be in (0.5, 1.0]")
        if not 0.0 <= self.shuffled_chance_tolerance < 0.5:
            raise ValueError("shuffled_chance_tolerance must be in [0.0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


# ---------------------------------------------------------------------------
# Evaluation-only probes.  Never wired into Router/ArgumentScorer/runtime.
# ---------------------------------------------------------------------------


def _fit_binary_probe(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
    shuffle_labels: bool,
) -> nn.Module:
    """Fit a tiny evaluation-only linear probe; never touches runtime parameters."""
    dim = positive.shape[-1]
    device = positive.device
    features = torch.cat([positive, negative], dim=0).detach()
    labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))]).to(device)
    if shuffle_labels:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        labels = labels[torch.randperm(len(labels), generator=generator).to(device)]
    torch.manual_seed(seed)
    probe = nn.Linear(dim, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return probe


def _probe_accuracy(probe: nn.Module, positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Accuracy of a fitted binary probe on a disjoint (positive, negative) eval split."""
    with torch.no_grad():
        pos_logits = probe(positive).squeeze(-1)
        neg_logits = probe(negative).squeeze(-1)
    correct = float((pos_logits > 0).sum().item()) + float((neg_logits <= 0).sum().item())
    total = len(positive) + len(negative)
    return correct / total


def _centroid_accuracy(
    train_positive: torch.Tensor,
    train_negative: torch.Tensor,
    eval_positive: torch.Tensor,
    eval_negative: torch.Tensor,
) -> float:
    """Simple untrained evaluation-only readout: nearest development-fit centroid."""
    centroid_positive = train_positive.mean(dim=0)
    centroid_negative = train_negative.mean(dim=0)

    def predicts_positive(x: torch.Tensor) -> torch.Tensor:
        d_pos = (x - centroid_positive).norm(dim=-1)
        d_neg = (x - centroid_negative).norm(dim=-1)
        return d_pos < d_neg

    correct = float(predicts_positive(eval_positive).sum().item())
    correct += float((~predicts_positive(eval_negative)).sum().item())
    return correct / (len(eval_positive) + len(eval_negative))


# ---------------------------------------------------------------------------
# Per-core, per-relation collection.
# ---------------------------------------------------------------------------


def _operation_salt(operation: str) -> int:
    """Deterministic per-operation salt (``hash()`` is randomized per process)."""
    return sum(ord(character) for character in operation) * 31


def _probe_examples(seed_base: int, operation: str, count: int, tag: int, split: str) -> list[Any]:
    return list(
        generate_benchmark_examples(
            seed_base * 20_000_003 + _operation_salt(operation) + tag, count, operation=operation, split=split
        )
    )


def _collect_core_relation_rows(
    *,
    config: RepresentationStageProbeConfig,
    core: Any,
    router: Any,
    argument_scorer: Any | None,
    candidate_ids: Sequence[int],
    operation_by_id: dict[int, str],
    op_to_id: dict[str, int],
    partition: str,
    state: str,
    evaluation_seed: int,
    model_seed: int,
) -> list[dict[str, Any]]:
    """Collect one geometry+probe row per target operation for one frozen core/state."""
    rows: list[dict[str, Any]] = []
    for target_operation in config.target_operations:
        target_id = op_to_id[target_operation]
        competitor_operation = _RELATED_OPERATION[target_operation]

        keys = {pid: router.key_parameter(pid).detach().clone() for pid in candidate_ids}
        candidates, competitor = build_hard_negative_candidates(
            level=_L3,
            target_id=target_id,
            target_operation=target_operation,
            candidate_ids=candidate_ids,
            keys_by_id=keys,
            operation_by_id=operation_by_id,
            seed=evaluation_seed,
        )
        correct_index = next(
            i
            for i, candidate in enumerate(candidates)
            if candidate is not competitor and candidate.execute_primitive_id == target_id
        )
        competitor_index = next(i for i, candidate in enumerate(candidates) if candidate is competitor)

        # --- Control A: current q_task -> current keys (the deployed path). ---
        query_examples = generate_benchmark_examples(
            evaluation_seed * 50_000 + config.bank_size + target_id + 100,
            config.query_examples,
            operation=target_operation,
            split="dev" if partition == "development" else "test",
        )
        arg_lambda = config.arg_lambda if state == _POST_REPAIR else 0.0
        scores, ordering = _rank_candidates_factorized(
            core, router, candidates, query_examples, argument_scorer, operation_by_id, arg_lambda
        )
        ranks = (ordering == correct_index).nonzero(as_tuple=False)[:, 1] + 1
        control_a_top1 = float((ordering[:, 0] == correct_index).to(torch.float32).mean().item())
        score_margin = float(
            (scores[:, correct_index] - scores[:, competitor_index]).mean().item()
        )
        target_key, competitor_key = candidates[correct_index].score_key, candidates[competitor_index].score_key

        # --- Control D: oracle target family -> direct key lookup (sanity floor). ---
        control_d_top1 = 1.0  # bypasses scoring entirely by construction; not a repair candidate.

        # --- Probe-only data: disjoint seeds, never used for router/scorer training. ---
        train_target = _probe_examples(evaluation_seed, target_operation, config.probe_train_examples, 1, "dev")
        train_competitor = _probe_examples(
            evaluation_seed, competitor_operation, config.probe_train_examples, 2, "dev"
        )
        eval_target = _probe_examples(evaluation_seed, target_operation, config.probe_eval_examples, 3, "test")
        eval_competitor = _probe_examples(
            evaluation_seed, competitor_operation, config.probe_eval_examples, 4, "test"
        )

        z_train_target = extract_task_representations(core, train_target)
        z_train_competitor = extract_task_representations(core, train_competitor)
        z_eval_target = extract_task_representations(core, eval_target)
        z_eval_competitor = extract_task_representations(core, eval_competitor)
        with torch.no_grad():
            q_train_target = router.query_proj(z_train_target)
            q_train_competitor = router.query_proj(z_train_competitor)
            q_eval_target = router.query_proj(z_eval_target)
            q_eval_competitor = router.query_proj(z_eval_competitor)

        probe_seed = evaluation_seed * 7_919 + target_id
        z_probe = _fit_binary_probe(
            z_train_target, z_train_competitor,
            steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed, shuffle_labels=False,
        )
        z_probe_shuffled = _fit_binary_probe(
            z_train_target, z_train_competitor,
            steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed, shuffle_labels=True,
        )
        q_probe = _fit_binary_probe(
            q_train_target, q_train_competitor,
            steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed + 1, shuffle_labels=False,
        )
        q_probe_shuffled = _fit_binary_probe(
            q_train_target, q_train_competitor,
            steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed + 1, shuffle_labels=True,
        )

        rows.append(
            {
                "partition": partition,
                "router_state": state,
                "evaluation_seed": evaluation_seed,
                "model_seed": model_seed,
                "target_family": target_operation,
                "competitor_family": competitor_operation,
                "bank_size": config.bank_size,
                # Geometry-only diagnostics (D2-002.2, first pass).
                "key_cosine_similarity": _cosine(target_key, competitor_key),
                "key_euclidean_distance": float((target_key - competitor_key).norm(p=2).item()),
                "score_margin": score_margin,
                "nearest_key_rank": float(ranks.to(torch.float32).mean().item()),
                # Control A: current q_task -> current keys.
                "control_a_current_path_top1": control_a_top1,
                # Control B: z_task -> evaluation-only nearest-centroid readout.
                "control_b_z_centroid_accuracy": _centroid_accuracy(
                    z_train_target, z_train_competitor, z_eval_target, z_eval_competitor
                ),
                # Control C: q_task -> evaluation-only target/competitor classifier.
                "control_c_q_probe_accuracy": _probe_accuracy(q_probe, q_eval_target, q_eval_competitor),
                "control_c_q_probe_shuffled_accuracy": _probe_accuracy(
                    q_probe_shuffled, q_eval_target, q_eval_competitor
                ),
                # Control D: oracle target family -> current key lookup (sanity floor).
                "control_d_oracle_lookup_top1": control_d_top1,
                # z_task probe (separate from q_task; distinguishes representation from projection).
                "z_probe_accuracy": _probe_accuracy(z_probe, z_eval_target, z_eval_competitor),
                "z_probe_shuffled_accuracy": _probe_accuracy(z_probe_shuffled, z_eval_target, z_eval_competitor),
            }
        )
    return rows


def _repair_config(config: RepresentationStageProbeConfig, seed: int) -> RetrievalRepairConfig:
    return RetrievalRepairConfig(
        seeds=(seed,),
        bank_sizes=(config.bank_size,),
        levels=(_L3,),
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


def _build_cell(
    config: RepresentationStageProbeConfig,
    *,
    partition: str,
    state: str,
    evaluation_seed: int,
    model_seed: int,
    training_seed: int | None,
) -> list[dict[str, Any]]:
    """Reconstruct one frozen (partition, state) cell for one core and collect its rows."""
    set_seed(evaluation_seed, deterministic_algorithms=config.deterministic_algorithms)
    base_config = HardNegativeBenchmarkConfig(
        seeds=(model_seed,),
        bank_sizes=(config.bank_size,),
        levels=(_L3,),
        target_operations=config.target_operations,
        num_eval_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        top_k=config.top_k,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )
    core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)
    bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
        core, base_bank, base_router, op_to_id, config.bank_size, seed=evaluation_seed
    )
    operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
    operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

    argument_scorer = None
    if state == _POST_REPAIR:
        if training_seed is None:
            raise ValueError("post-repair reconstruction requires a development training seed")
        router, argument_scorer = train_repaired_router_and_scorer(
            core,
            router,
            candidate_ids,
            operation_by_id,
            _training_examples(
                training_seed, op_to_id, config.router_train_examples, regate_recipe=partition == "regate_sealed"
            ),
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

    return _collect_core_relation_rows(
        config=config,
        core=core,
        router=router,
        argument_scorer=argument_scorer,
        candidate_ids=candidate_ids,
        operation_by_id=operation_by_id,
        op_to_id=op_to_id,
        partition=partition,
        state=state,
        evaluation_seed=evaluation_seed,
        model_seed=model_seed,
    )


# ---------------------------------------------------------------------------
# Aggregation and interpretation (D2-002.4).
# ---------------------------------------------------------------------------

_METRIC_KEYS: Final[tuple[str, ...]] = (
    "key_cosine_similarity",
    "key_euclidean_distance",
    "score_margin",
    "nearest_key_rank",
    "control_a_current_path_top1",
    "control_b_z_centroid_accuracy",
    "control_c_q_probe_accuracy",
    "control_c_q_probe_shuffled_accuracy",
    "control_d_oracle_lookup_top1",
    "z_probe_accuracy",
    "z_probe_shuffled_accuracy",
)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else None


def _cell_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(rows), **{key: _mean(rows, key) for key in _METRIC_KEYS}}


def _is_chance(value: float | None, tolerance: float) -> bool:
    return value is not None and abs(value - 0.5) <= tolerance


def _classify_relation(summary: dict[str, Any], config: RepresentationStageProbeConfig) -> tuple[str, dict[str, Any]]:
    """Apply the D2-002.4 interpretation logic to one aggregated (partition, state) cell."""
    threshold = config.separability_threshold
    shuffled_valid = _is_chance(summary["z_probe_shuffled_accuracy"], config.shuffled_chance_tolerance) and _is_chance(
        summary["control_c_q_probe_shuffled_accuracy"], config.shuffled_chance_tolerance
    )
    evidence = {
        "z_probe_accuracy": summary["z_probe_accuracy"],
        "q_probe_accuracy": summary["control_c_q_probe_accuracy"],
        "control_a_current_path_top1": summary["control_a_current_path_top1"],
        "shuffled_controls_at_chance": shuffled_valid,
    }
    if not shuffled_valid:
        return "UNRESOLVED", {**evidence, "reason": "shuffled-label probe control was not near chance"}

    z_hi = (summary["z_probe_accuracy"] or 0.0) >= threshold
    q_hi = (summary["control_c_q_probe_accuracy"] or 0.0) >= threshold
    a_hi = (summary["control_a_current_path_top1"] or 0.0) >= threshold

    if z_hi and a_hi:
        return "NO_FAILURE", evidence
    if not z_hi:
        return "TASK_REPRESENTATION_BOTTLENECK", evidence
    if z_hi and not q_hi:
        return "QUERY_PROJECTION_BOTTLENECK", evidence
    if z_hi and q_hi and not a_hi:
        return "KEY_SCORING_BOTTLENECK", evidence
    return "UNRESOLVED", evidence


def _interpret(rows: Sequence[dict[str, Any]], config: RepresentationStageProbeConfig) -> dict[str, Any]:
    """Combine per-relation classifications into the D2-002 stage-localization verdict."""
    labels = sorted({(str(r["partition"]), str(r["router_state"])) for r in rows})
    by_cell: dict[str, Any] = {}
    for partition, state in labels:
        cell_rows = [r for r in rows if r["partition"] == partition and r["router_state"] == state]
        summary = _cell_summary(cell_rows)
        per_relation: dict[str, Any] = {}
        for target_operation in config.target_operations:
            relation_rows = [r for r in cell_rows if r["target_family"] == target_operation]
            relation_summary = _cell_summary(relation_rows)
            classification, evidence = _classify_relation(relation_summary, config)
            per_relation[f"{target_operation}->{_RELATED_OPERATION[target_operation]}"] = {
                "classification": classification,
                "summary": relation_summary,
                "evidence": evidence,
            }
        overall_classification, overall_evidence = _classify_relation(summary, config)
        relation_labels = {item["classification"] for item in per_relation.values()}
        if len(relation_labels) > 1:
            overall_classification = "MIXED"
        by_cell[f"{partition}/{state}"] = {
            "summary": summary,
            "classification": overall_classification,
            "evidence": overall_evidence,
            "per_relation": per_relation,
        }

    focus_key = "regate_sealed/R2_frozen_post_repair"
    focus = by_cell.get(focus_key, {})
    return {
        "by_partition_state": by_cell,
        "focus_cell": focus_key,
        "verdict": focus.get("classification", "UNRESOLVED"),
        "verdict_evidence": focus.get("evidence", {}),
        "per_relation_verdict": {
            relation: item["classification"] for relation, item in focus.get("per_relation", {}).items()
        },
    }


def run_representation_stage_probe(config: RepresentationStageProbeConfig) -> dict[str, Any]:
    """Run the B-C005D2-002 L3 representation-stage localization audit."""
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    rows: list[dict[str, Any]] = []
    for seed in config.original_sealed_seeds:
        rows.extend(
            _build_cell(
                config, partition="original_sealed", state=_PRE_REPAIR,
                evaluation_seed=seed, model_seed=seed, training_seed=None,
            )
        )
    for seed in config.development_seeds:
        rows.extend(
            _build_cell(
                config, partition="development", state=_PRE_REPAIR,
                evaluation_seed=seed, model_seed=seed, training_seed=None,
            )
        )
        rows.extend(
            _build_cell(
                config, partition="development", state=_POST_REPAIR,
                evaluation_seed=seed, model_seed=seed, training_seed=seed,
            )
        )
    for sealed_seed, development_seed in zip(config.regate_sealed_seeds, config.development_seeds, strict=True):
        rows.extend(
            _build_cell(
                config, partition="regate_sealed", state=_POST_REPAIR,
                evaluation_seed=sealed_seed, model_seed=sealed_seed % 5, training_seed=development_seed,
            )
        )

    interpretation = _interpret(rows, config)
    report = {
        "task_id": "B-C005D2-002",
        "config": config.to_dict(),
        "rows_analyzed": len(rows),
        "representation_stage_localization": interpretation,
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-002",
            "sealed_partitions": {
                "original": list(config.original_sealed_seeds),
                "regate": list(config.regate_sealed_seeds),
            },
            "development_partition": list(config.development_seeds),
            "sealed_training_prohibited": True,
            "probes_are_evaluation_only": True,
        }
        (out / "representation_stage_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "representation_stage_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        (out / "representation_stage_system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "representation_stage_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out / "representation_stage_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report
