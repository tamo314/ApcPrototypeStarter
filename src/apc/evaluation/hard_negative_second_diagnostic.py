"""Evaluation-only development-versus-sealed audit for B-C005D2-001.

The original B-C005 and B-C005G artifacts store cell aggregates, but not the
per-query key geometry and score descriptors needed to assess whether their
development split represented sealed difficulty.  This module reconstructs
the frozen states without updating a model and writes those descriptors.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
    _cosine,
    _primitive_argument_value,
    _wrong_arguments,
    build_hard_negative_candidates,
)
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

_AUDIT_LEVELS: Final[tuple[HardNegativeLevel, ...]] = (
    HardNegativeLevel.L3_SEMANTICALLY_RELATED,
    HardNegativeLevel.L4_CONFUSABLE_FAMILY,
)
_PRE_REPAIR = "R0_frozen_pre_repair"
_POST_REPAIR = "R2_frozen_post_repair"


@dataclass(frozen=True)
class HardNegativeSecondDiagnosticConfig:
    """Frozen reconstruction protocol for the D2-001 descriptive audit."""

    original_sealed_seeds: tuple[int, ...] = tuple(sorted(SEALED_GATE_SEEDS))
    development_seeds: tuple[int, ...] = DEFAULT_DEV_SEEDS
    regate_sealed_seeds: tuple[int, ...] = DEFAULT_REGATE_SEEDS
    bank_sizes: tuple[int, ...] = DEFAULT_BANK_SIZES
    levels: tuple[HardNegativeLevel, ...] = _AUDIT_LEVELS
    target_operations: tuple[str, ...] = DEFAULT_TARGET_OPERATIONS
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        groups = (
            self.original_sealed_seeds,
            self.development_seeds,
            self.regate_sealed_seeds,
        )
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
        if set(self.bank_sizes) - set(DEFAULT_BANK_SIZES):
            raise ValueError(f"bank_sizes must be drawn from {DEFAULT_BANK_SIZES}")
        if set(self.levels) - set(_AUDIT_LEVELS):
            raise ValueError("D2-001 is restricted to L3 and L4")
        if self.query_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")
        if self.router_steps < 1 or not 1 <= self.top_k <= 5:
            raise ValueError("router_steps must be positive and top_k must be in [1, 5]")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["levels"] = [level.value for level in self.levels]
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return stable descriptive statistics without treating an empty group as zero."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lower, upper = math.floor(index), math.ceil(index)
        return (
            ordered[lower]
            if lower == upper
            else ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
        )

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": percentile(0.50),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def _frequency(values: Iterable[str]) -> list[dict[str, Any]]:
    """Return a deterministic empirical distribution suitable for JSON artifacts."""
    counts = Counter(values)
    total = sum(counts.values())
    return [
        {"value": value, "count": count, "frequency": count / total}
        for value, count in sorted(counts.items())
    ]


def _argument_descriptor(
    operation: str, example: Any, *, wrong: bool, vocab_size: int
) -> dict[str, Any]:
    """Describe the effective primitive-call argument without exposing it to routing."""
    assert example.task_spec is not None
    arguments = dict(example.task_spec.steps[0].arguments)
    if wrong:
        arguments = _wrong_arguments(operation, example, vocab_size)
    argument_names = {
        "SHIFT": "amount",
        "SELECT": "indices",
        "COUNT": "target",
        "BIND": "query_key",
    }
    argument_name = argument_names.get(operation)
    if argument_name is None:
        return {}
    return {argument_name: _primitive_argument_value(operation, arguments)}


def _family_group(operation: str) -> str:
    """Use the real operation registry's canonical operation as the smallest taxonomy."""
    return f"operation:{operation}"


def _training_examples(
    seed: int,
    op_to_id: dict[str, int],
    count: int,
    *,
    regate_recipe: bool,
) -> dict[str, list[Any]]:
    """Generate development-only examples used by the already selected R2 recipe."""
    if seed in SEALED_GATE_SEEDS or seed in DEFAULT_REGATE_SEEDS:
        raise ValueError("sealed examples are prohibited from D2 training reconstruction")
    if regate_recipe:
        return {
            operation: list(
                generate_benchmark_examples(
                    seed * 50_000 + primitive_id * 100 + 7,
                    count,
                    operation=operation,
                    split="train",
                )
            )
            for operation, primitive_id in op_to_id.items()
        }
    return {
        operation: list(
            generate_benchmark_examples(
                seed * 30_000 + 128 + primitive_id, count, operation=operation, split="dev"
            )
        )
        for operation, primitive_id in op_to_id.items()
    }


def _repair_config(config: HardNegativeSecondDiagnosticConfig, seed: int) -> RetrievalRepairConfig:
    return RetrievalRepairConfig(
        seeds=(seed,),
        bank_sizes=config.bank_sizes,
        levels=config.levels,
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


def _collect_rows(
    *,
    config: HardNegativeSecondDiagnosticConfig,
    partition: str,
    state: str,
    evaluation_seed: int,
    model_seed: int,
    training_seed: int | None,
) -> list[dict[str, Any]]:
    """Collect pair descriptors under one labelled, immutable model reconstruction."""
    # B-C005G initializes the evaluation cell from the sealed evaluation seed
    # while loading its historical base checkpoint by ``model_seed``.
    set_seed(evaluation_seed, deterministic_algorithms=config.deterministic_algorithms)
    base_config = HardNegativeBenchmarkConfig(
        seeds=(model_seed,),
        bank_sizes=config.bank_sizes,
        levels=config.levels,
        target_operations=config.target_operations,
        num_eval_examples=config.query_examples,
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        top_k=config.top_k,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )
    core, base_bank, base_router, op_to_id = _build_frozen_base_system(model_seed, base_config)
    rows: list[dict[str, Any]] = []
    for bank_size in config.bank_sizes:
        bank, router, candidate_ids, _, distractor_ids = build_scaled_bank_and_router(
            core, base_bank, base_router, op_to_id, bank_size, seed=evaluation_seed
        )
        operation_by_id = {primitive_id: operation for operation, primitive_id in op_to_id.items()}
        operation_by_id.update({primitive_id: "SWAP_ENDS" for primitive_id in distractor_ids})
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
                    training_seed,
                    op_to_id,
                    config.router_train_examples,
                    regate_recipe=partition == "regate_sealed",
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
        for operation in config.target_operations:
            target_id = op_to_id[operation]
            if partition == "regate_sealed":
                query_seed = evaluation_seed * 1_000_000 + bank_size * 100 + target_id + 1
            elif partition == "original_sealed":
                query_seed = evaluation_seed * 100_000 + bank_size * 100 + target_id + 1
            else:
                query_seed = evaluation_seed * 50_000 + bank_size + target_id + 100
            examples = generate_benchmark_examples(
                query_seed,
                config.query_examples,
                operation=operation,
                split="dev" if partition == "development" else "test",
            )
            for level in config.levels:
                keys = {
                    primitive_id: router.key_parameter(primitive_id).detach().clone()
                    for primitive_id in candidate_ids
                }
                candidates, competitor = build_hard_negative_candidates(
                    level=level,
                    target_id=target_id,
                    target_operation=operation,
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
                competitor_index = next(
                    i for i, candidate in enumerate(candidates) if candidate is competitor
                )
                scores, ordering = _rank_candidates_factorized(
                    core,
                    router,
                    candidates,
                    examples,
                    argument_scorer,
                    operation_by_id,
                    config.arg_lambda if state == _POST_REPAIR else 0.0,
                )
                target_key, competitor_key = (
                    candidates[correct_index].score_key,
                    candidates[competitor_index].score_key,
                )
                competitor_operation = operation_by_id[competitor.execute_primitive_id]
                for example_index, example in enumerate(examples):
                    rank = (
                        int(
                            (ordering[example_index] == correct_index)
                            .nonzero(as_tuple=False)[0]
                            .item()
                        )
                        + 1
                    )
                    top1 = int(ordering[example_index, 0].item())
                    target_arguments = _argument_descriptor(
                        operation, example, wrong=False, vocab_size=core.tokens.env_vocab_size
                    )
                    competitor_arguments = _argument_descriptor(
                        competitor_operation,
                        example,
                        wrong=competitor.argument_override is not None,
                        vocab_size=core.tokens.env_vocab_size,
                    )
                    rows.append(
                        {
                            "partition": partition,
                            "router_state": state,
                            "checkpoint_label": (
                                f"base_seed_{model_seed};train_seed_{training_seed}"
                            ),
                            "evaluation_seed": evaluation_seed,
                            "model_seed": model_seed,
                            "training_seed": training_seed,
                            "bank_size": bank_size,
                            "level": level.value,
                            "target_family": operation,
                            "competitor_family": competitor_operation,
                            "target_arguments": target_arguments,
                            "competitor_arguments": competitor_arguments,
                            "relation_type": competitor.provenance,
                            "semantic_relation_id": competitor.semantic_relation
                            if level == HardNegativeLevel.L3_SEMANTICALLY_RELATED
                            else None,
                            "target_family_group": _family_group(operation)
                            if level == HardNegativeLevel.L3_SEMANTICALLY_RELATED
                            else None,
                            "competitor_family_group": _family_group(competitor_operation)
                            if level == HardNegativeLevel.L3_SEMANTICALLY_RELATED
                            else None,
                            "target_key_norm": float(target_key.norm().item()),
                            "competitor_key_norm": float(competitor_key.norm().item()),
                            "key_cosine_similarity": _cosine(target_key, competitor_key),
                            "query_target_score": float(
                                scores[example_index, correct_index].item()
                            ),
                            "query_competitor_score": float(
                                scores[example_index, competitor_index].item()
                            ),
                            "score_margin": float(
                                (
                                    scores[example_index, correct_index]
                                    - scores[example_index, competitor_index]
                                ).item()
                            ),
                            "correct_rank": rank,
                            "top1_correct": top1 == correct_index,
                            "top5_included": bool(
                                (
                                    ordering[example_index, : min(config.top_k, len(candidates))]
                                    == correct_index
                                )
                                .any()
                                .item()
                            ),
                            "argument_correct": top1 == correct_index
                            if level == HardNegativeLevel.L4_CONFUSABLE_FAMILY
                            else True,
                            "family_top1": candidates[top1].execute_primitive_id == target_id,
                        }
                    )
    return rows


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate required comparable metrics for one labelled state and level."""
    if not rows:
        return {
            "n": 0,
            "family_top1": None,
            "argument_accuracy": None,
            "primitive_call_top1": None,
            "top5": None,
            "score_margin": _quantiles([]),
            "correct_rank": _quantiles([]),
        }
    return {
        "n": len(rows),
        "family_top1": statistics.fmean(float(row["family_top1"]) for row in rows),
        "argument_accuracy": statistics.fmean(float(row["argument_correct"]) for row in rows),
        "primitive_call_top1": statistics.fmean(float(row["top1_correct"]) for row in rows),
        "top5": statistics.fmean(float(row["top5_included"]) for row in rows),
        "score_margin": _quantiles([float(row["score_margin"]) for row in rows]),
        "correct_rank": _quantiles([float(row["correct_rank"]) for row in rows]),
    }


def _comparison(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Create the D2-001 distribution comparison without outcome-based matching."""
    evaluation_bank_size = max(int(row["bank_size"]) for row in rows)
    result: dict[str, Any] = {
        "evaluation_bank_size": evaluation_bank_size,
        "conditions": {},
        "comparisons": {},
    }
    labels = sorted({(str(row["partition"]), str(row["router_state"])) for row in rows})
    for partition, state in labels:
        selected = [
            row
            for row in rows
            if row["partition"] == partition
            and row["router_state"] == state
            and row["bank_size"] == evaluation_bank_size
        ]
        result["conditions"][f"{partition}/{state}"] = {
            "checkpoint_labels": sorted(set(str(row["checkpoint_label"]) for row in selected)),
            "by_level": {
                level.value: _metric_summary(
                    [row for row in selected if row["level"] == level.value]
                )
                for level in _AUDIT_LEVELS
            },
        }
    for sealed_partition in ("original_sealed", "regate_sealed"):
        for level in _AUDIT_LEVELS:
            dev = [
                row
                for row in rows
                if row["partition"] == "development"
                and row["router_state"] == _POST_REPAIR
                and row["level"] == level.value
                and row["bank_size"] == evaluation_bank_size
            ]
            sealed = [
                row
                for row in rows
                if row["partition"] == sealed_partition
                and row["level"] == level.value
                and row["bank_size"] == evaluation_bank_size
            ]
            prefix = f"development_post_repair_vs_{sealed_partition}/{level.value}"
            result["comparisons"][prefix] = {
                "warning": (
                    "Cross-checkpoint descriptive comparison; checkpoint labels are retained "
                    "and this is not a model-selection comparison."
                ),
                "key_similarity": {
                    "development": _quantiles([r["key_cosine_similarity"] for r in dev]),
                    "sealed": _quantiles([r["key_cosine_similarity"] for r in sealed]),
                },
                "score_margin": {
                    "development": _quantiles([r["score_margin"] for r in dev]),
                    "sealed": _quantiles([r["score_margin"] for r in sealed]),
                },
                "target_rank": {
                    "development": _quantiles([float(r["correct_rank"]) for r in dev]),
                    "sealed": _quantiles([float(r["correct_rank"]) for r in sealed]),
                },
                "relation_type_frequencies": {
                    "development": _frequency(
                        str(r["semantic_relation_id"] or r["relation_type"]) for r in dev
                    ),
                    "sealed": _frequency(
                        str(r["semantic_relation_id"] or r["relation_type"]) for r in sealed
                    ),
                },
                "target_competitor_pair_frequencies": {
                    "development": _frequency(
                        f"{r['target_family']}->{r['competitor_family']}" for r in dev
                    ),
                    "sealed": _frequency(
                        f"{r['target_family']}->{r['competitor_family']}" for r in sealed
                    ),
                },
                "argument_value_frequencies": {
                    "development": _frequency(
                        json.dumps(r["target_arguments"], sort_keys=True) for r in dev
                    ),
                    "sealed": _frequency(
                        json.dumps(r["target_arguments"], sort_keys=True) for r in sealed
                    ),
                },
            }
    return result


def _verdict(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify representativeness using declared descriptive, not tuning, rules."""

    def top1(partition: str, state: str) -> float:
        evaluation_bank_size = max(int(row["bank_size"]) for row in rows)
        selected = [
            r
            for r in rows
            if r["partition"] == partition
            and r["router_state"] == state
            and r["level"] == HardNegativeLevel.L3_SEMANTICALLY_RELATED.value
            and r["bank_size"] == evaluation_bank_size
        ]
        return statistics.fmean(float(r["top1_correct"]) for r in selected) if selected else 0.0

    pre_dev, pre_original = top1("development", _PRE_REPAIR), top1("original_sealed", _PRE_REPAIR)
    post_dev, post_regate = top1("development", _POST_REPAIR), top1("regate_sealed", _POST_REPAIR)
    mismatch = pre_dev >= 0.95 and pre_original < 0.95 and post_regate < 0.95
    if mismatch:
        classification = "NON_REPRESENTATIVE"
    elif abs(post_dev - post_regate) <= 0.05:
        classification = "REPRESENTATIVE"
    else:
        classification = "PARTIALLY_REPRESENTATIVE"
    return {
        "classification": classification,
        "flags": ["DEVELOPMENT_DIFFICULTY_MISMATCH"] if mismatch else [],
        "evidence": {
            "pre_repair_development_l3_top1": pre_dev,
            "pre_repair_original_sealed_l3_top1": pre_original,
            "post_repair_development_l3_top1": post_dev,
            "post_repair_regate_sealed_l3_top1": post_regate,
        },
    }


def run_hard_negative_second_diagnostic(
    config: HardNegativeSecondDiagnosticConfig,
) -> dict[str, Any]:
    """Run the B-C005D2-001 audit with evaluation-only sealed readout."""
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    rows: list[dict[str, Any]] = []
    for seed in config.original_sealed_seeds:
        rows.extend(
            _collect_rows(
                config=config,
                partition="original_sealed",
                state=_PRE_REPAIR,
                evaluation_seed=seed,
                model_seed=seed,
                training_seed=None,
            )
        )
    for seed in config.development_seeds:
        rows.extend(
            _collect_rows(
                config=config,
                partition="development",
                state=_PRE_REPAIR,
                evaluation_seed=seed,
                model_seed=seed,
                training_seed=None,
            )
        )
        rows.extend(
            _collect_rows(
                config=config,
                partition="development",
                state=_POST_REPAIR,
                evaluation_seed=seed,
                model_seed=seed,
                training_seed=seed,
            )
        )
    for sealed_seed, development_seed in zip(
        config.regate_sealed_seeds, config.development_seeds, strict=True
    ):
        # Match the B-C005G model construction exactly: frozen base seed 0--4,
        # development-only R2 fit, then sealed evaluation.
        rows.extend(
            _collect_rows(
                config=config,
                partition="regate_sealed",
                state=_POST_REPAIR,
                evaluation_seed=sealed_seed,
                model_seed=sealed_seed % 5,
                training_seed=development_seed,
            )
        )
    comparison = _comparison(rows)
    verdict = _verdict(rows)
    report = {
        "task_id": "B-C005D2-001",
        "config": config.to_dict(),
        "rows_analyzed": len(rows),
        "development_sealed_comparison": comparison,
        "representativeness_verdict": verdict,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-001",
            "sealed_partitions": {
                "original": list(config.original_sealed_seeds),
                "regate": list(config.regate_sealed_seeds),
            },
            "development_partition": list(config.development_seeds),
            "sealed_training_prohibited": True,
        }
        (out / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        (out / "system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out / "development_sealed_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        relation_rows = [
            row for row in rows if row["level"] == HardNegativeLevel.L3_SEMANTICALLY_RELATED.value
        ]
        semantic_summary = {
            "relation_sample_counts": _frequency(
                str(row["semantic_relation_id"]) for row in relation_rows
            ),
            "taxonomy": (
                "canonical target-family -> competitor-family mapping from the real "
                "operation registry"
            ),
        }
        (out / "semantic_relation_summary.json").write_text(
            json.dumps(semantic_summary, indent=2) + "\n", encoding="utf-8"
        )
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
