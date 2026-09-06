# ruff: noqa: E501
"""Evaluation-only L3 semantic-relation generalization audit for B-C005D2-003.

B-C005D2-002 localized the sealed L3 failure to a `KEY_SCORING_BOTTLENECK` that
is concentrated in the `COUNT<->BIND` mutually-confusable pair, while
`SHIFT->CYCLE_FOUR` shows `NO_FAILURE` -- the failure is not uniform across the
four canonical L3 relations. This module asks the D2-003 question directly:
was the original seed-only development/sealed split insufficient *because*
semantic-relation distributions were uncontrolled, so that future protocols
require an explicit semantic-relation holdout rather than a seed-only one?

Three evaluation-only analyses, all reusing the frozen reconstruction already
validated by B-C005D2-001 (`hard_negative_second_diagnostic._collect_rows`)
restricted to the L3 level and the sealed-gate bank size:

1. Relation-group evaluation (D2-003.2, base check): per-relation top-1
   grouped by partition/router-state, without any outcome-based matching.
2. Relation-holdout generalization probe (D2-003.2, probe check): a tiny
   evaluation-only linear probe over frozen-model difficulty descriptors
   (`key_cosine_similarity`, `score_margin`) predicting per-example top-1
   correctness, fit on three of the four relations and evaluated on the
   held-out fourth -- never fed back into the router.
3. Difficulty-matched comparison (D2-003.3): development vs. sealed top-1
   compared within matched (relation, key-cosine-similarity-bin) cells, to
   separate "sealed is just harder on average" from "sealed fails even on
   examples matched for relation and difficulty".

Per D2-003's own constraint, matched-subset construction here uses only
pre-model/frozen-model descriptors (`semantic_relation_id`,
`key_cosine_similarity`) -- never query-outcome labels.

Reconstruction cost is bounded to one bank size (the sealed-gate scale) and
the L3 level only, following B-C005D2-002's own scope precedent.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    _RELATED_OPERATION,
    DEFAULT_BANK_SIZES,
    DEFAULT_TARGET_OPERATIONS,
)
from apc.evaluation.hard_negative_second_diagnostic import (
    HardNegativeSecondDiagnosticConfig,
    _collect_rows,
)
from apc.evaluation.representation_stage_probe import _fit_binary_probe, _probe_accuracy
from apc.evaluation.retrieval_repair_benchmark import DEFAULT_DEV_SEEDS, SEALED_GATE_SEEDS
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.utils.system_info import get_system_info

_L3: Final[HardNegativeLevel] = HardNegativeLevel.L3_SEMANTICALLY_RELATED
_PRE_REPAIR: Final[str] = "R0_frozen_pre_repair"
_POST_REPAIR: Final[str] = "R2_frozen_post_repair"
_RELATION_UNITS: Final[tuple[str, ...]] = tuple(
    f"{target}->{competitor}" for target, competitor in _RELATED_OPERATION.items()
)


@dataclass(frozen=True)
class SemanticRelationAuditConfig:
    """Frozen protocol for the D2-003 L3 semantic-relation generalization audit."""

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
    probe_steps: int = 300
    probe_lr: float = 0.05
    cosine_bin_count: int = 3
    match_gap_tolerance: float = 0.05
    relation_spread_threshold: float = 0.15
    probe_generalization_gap_tolerance: float = 0.15
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
        if self.query_examples < 1 or self.router_train_examples < 1:
            raise ValueError("example counts must be positive")
        if self.router_steps < 1 or self.probe_steps < 1:
            raise ValueError("optimization step counts must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if self.cosine_bin_count < 2:
            raise ValueError("cosine_bin_count must be at least 2")
        if not 0.0 <= self.shuffled_chance_tolerance < 0.5:
            raise ValueError("shuffled_chance_tolerance must be in [0.0, 0.5)")
        if self.match_gap_tolerance < 0.0 or self.relation_spread_threshold < 0.0:
            raise ValueError("tolerance/threshold values must be non-negative")
        if self.probe_generalization_gap_tolerance < 0.0:
            raise ValueError("probe_generalization_gap_tolerance must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        result["output_dir"] = str(self.output_dir) if self.output_dir else None
        return result


# ---------------------------------------------------------------------------
# D2-003.1 -- relation taxonomy (structural, not outcome-derived).
# ---------------------------------------------------------------------------


def _relation_taxonomy() -> dict[str, Any]:
    """The smallest taxonomy the existing registry supports: one (target, competitor)
    family pair per parameterized operation, taken from `_RELATED_OPERATION`."""
    return {
        "unit_definition": "(target family, competitor family) pair from the canonical operation registry",
        "relations": [
            {"relation_id": f"{target}->{competitor}", "target_family": target, "competitor_family": competitor}
            for target, competitor in _RELATED_OPERATION.items()
        ],
        "note": "Labels are structural (registry-defined), not derived from success/failure outcomes.",
    }


# ---------------------------------------------------------------------------
# Row reconstruction: reuse B-C005D2-001's frozen-core rebuild, L3-only.
# ---------------------------------------------------------------------------


def _reconstruction_config(config: SemanticRelationAuditConfig) -> HardNegativeSecondDiagnosticConfig:
    return HardNegativeSecondDiagnosticConfig(
        original_sealed_seeds=config.original_sealed_seeds,
        development_seeds=config.development_seeds,
        regate_sealed_seeds=config.regate_sealed_seeds,
        bank_sizes=(config.bank_size,),
        levels=(_L3,),
        target_operations=config.target_operations,
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
        output_dir=None,
    )


def _collect_all_rows(config: SemanticRelationAuditConfig) -> list[dict[str, Any]]:
    """Rebuild every frozen (partition, state, seed) cell needed for D2-003, L3 only."""
    reconstruction_config = _reconstruction_config(config)
    rows: list[dict[str, Any]] = []
    for seed in config.original_sealed_seeds:
        rows.extend(
            _collect_rows(
                config=reconstruction_config,
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
                config=reconstruction_config,
                partition="development",
                state=_PRE_REPAIR,
                evaluation_seed=seed,
                model_seed=seed,
                training_seed=None,
            )
        )
        rows.extend(
            _collect_rows(
                config=reconstruction_config,
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
        rows.extend(
            _collect_rows(
                config=reconstruction_config,
                partition="regate_sealed",
                state=_POST_REPAIR,
                evaluation_seed=sealed_seed,
                model_seed=sealed_seed % 5,
                training_seed=development_seed,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# D2-003.2 (base check) -- relation-group evaluation, no matching/probing.
# ---------------------------------------------------------------------------


def _relation_group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-relation top-1/margin/key-similarity within each (partition, state) cell."""
    labels = sorted({(str(row["partition"]), str(row["router_state"])) for row in rows})
    result: dict[str, Any] = {}
    for partition, state in labels:
        cell_rows = [row for row in rows if row["partition"] == partition and row["router_state"] == state]
        per_relation: dict[str, Any] = {}
        for relation in _RELATION_UNITS:
            relation_rows = [row for row in cell_rows if row["semantic_relation_id"] == relation]
            if not relation_rows:
                continue
            per_relation[relation] = {
                "n": len(relation_rows),
                "top1": statistics.fmean(float(row["top1_correct"]) for row in relation_rows),
                "top5": statistics.fmean(float(row["top5_included"]) for row in relation_rows),
                "score_margin_mean": statistics.fmean(float(row["score_margin"]) for row in relation_rows),
                "key_cosine_similarity_mean": statistics.fmean(
                    float(row["key_cosine_similarity"]) for row in relation_rows
                ),
            }
        top1_values = [item["top1"] for item in per_relation.values()]
        result[f"{partition}/{state}"] = {
            "per_relation": per_relation,
            "cross_relation_spread": (max(top1_values) - min(top1_values)) if len(top1_values) > 1 else None,
        }
    return result


# ---------------------------------------------------------------------------
# D2-003.2 (probe check) -- relation-holdout difficulty -> correctness probe.
# ---------------------------------------------------------------------------


def _class_split(rows: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Split rows into (positive=correct, negative=incorrect) 2D difficulty features."""
    positive = [
        [float(row["key_cosine_similarity"]), float(row["score_margin"])]
        for row in rows
        if bool(row["top1_correct"])
    ]
    negative = [
        [float(row["key_cosine_similarity"]), float(row["score_margin"])]
        for row in rows
        if not bool(row["top1_correct"])
    ]
    return torch.tensor(positive, dtype=torch.float32), torch.tensor(negative, dtype=torch.float32)


def _fit_difficulty_probe(
    rows: Sequence[dict[str, Any]], *, steps: int, lr: float, seed: int, shuffle_labels: bool
) -> nn.Module | None:
    """Fit an evaluation-only 2D (key-cosine-similarity, score-margin) -> correctness probe.

    Returns ``None`` when one class is entirely absent (near-ceiling cells provide
    no negative examples) rather than fitting a degenerate single-class probe.
    """
    positive, negative = _class_split(rows)
    if len(positive) == 0 or len(negative) == 0:
        return None
    return _fit_binary_probe(positive, negative, steps=steps, lr=lr, seed=seed, shuffle_labels=shuffle_labels)


def _difficulty_probe_accuracy(probe: nn.Module | None, rows: Sequence[dict[str, Any]]) -> float | None:
    """Accuracy of a fitted difficulty probe on a disjoint eval split; ``None`` if degenerate."""
    if probe is None:
        return None
    positive, negative = _class_split(rows)
    if len(positive) == 0 or len(negative) == 0:
        return None
    return _probe_accuracy(probe, positive, negative)


def _relation_holdout_generalization(
    rows: Sequence[dict[str, Any]], config: SemanticRelationAuditConfig
) -> dict[str, Any]:
    """Leave-one-relation-out generalization of a frozen-descriptor difficulty probe.

    Uses only pre-repair rows (development and original-sealed), which is where
    both classes (correct/incorrect) are populated for every relation; post-repair
    development is near-ceiling and would starve the negative class entirely.
    """
    pool = [
        row
        for row in rows
        if row["router_state"] == _PRE_REPAIR and row["partition"] in ("development", "original_sealed")
    ]
    per_relation: dict[str, Any] = {}
    for held_out in _RELATION_UNITS:
        train_rows = [row for row in pool if row["semantic_relation_id"] != held_out]
        held_out_rows = [row for row in pool if row["semantic_relation_id"] == held_out]
        in_relation_train = [row for row in held_out_rows if row["partition"] == "original_sealed"]
        in_relation_eval = [row for row in held_out_rows if row["partition"] == "development"]

        seed = sum(ord(character) for character in held_out)
        cross_probe = _fit_difficulty_probe(
            train_rows, steps=config.probe_steps, lr=config.probe_lr, seed=seed, shuffle_labels=False
        )
        cross_probe_shuffled = _fit_difficulty_probe(
            train_rows, steps=config.probe_steps, lr=config.probe_lr, seed=seed, shuffle_labels=True
        )
        in_relation_probe = _fit_difficulty_probe(
            in_relation_train, steps=config.probe_steps, lr=config.probe_lr, seed=seed + 1, shuffle_labels=False
        )

        per_relation[held_out] = {
            "n_train_other_relations": len(train_rows),
            "n_held_out": len(held_out_rows),
            "cross_relation_generalization_accuracy": _difficulty_probe_accuracy(cross_probe, held_out_rows),
            "cross_relation_shuffled_control_accuracy": _difficulty_probe_accuracy(
                cross_probe_shuffled, held_out_rows
            ),
            "in_relation_accuracy": _difficulty_probe_accuracy(in_relation_probe, in_relation_eval),
        }
    return per_relation


def _classify_relation_holdout(
    per_relation: dict[str, Any], config: SemanticRelationAuditConfig
) -> tuple[str, dict[str, Any]]:
    """Compare in-relation vs. cross-relation difficulty-probe generalization."""
    gaps = []
    for item in per_relation.values():
        cross = item["cross_relation_generalization_accuracy"]
        in_relation = item["in_relation_accuracy"]
        if cross is not None and in_relation is not None:
            gaps.append(in_relation - cross)
    if not gaps:
        return "UNRESOLVED", {"reason": "insufficient class variety to fit/evaluate holdout probes"}
    mean_gap = statistics.fmean(gaps)
    evidence = {"mean_in_relation_minus_cross_relation_gap": mean_gap, "n_relations_compared": len(gaps)}
    if mean_gap > config.probe_generalization_gap_tolerance:
        return "SEMANTIC_RELATION_HOLDOUT_REQUIRED", evidence
    return "SEED_SPLIT_SUFFICIENT", evidence


# ---------------------------------------------------------------------------
# D2-003.3 -- difficulty-matched comparison (relation x key-cosine-similarity bin).
# ---------------------------------------------------------------------------


def _quantile_edges(values: Sequence[float], bin_count: int) -> list[float]:
    """Return ``bin_count - 1`` interior edges splitting ``values`` into equal-mass bins."""
    if not values:
        return []
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return [percentile(i / bin_count) for i in range(1, bin_count)]


def _bin_label(value: float, edges: Sequence[float]) -> int:
    """Index of the bin ``value`` falls into given interior quantile edges."""
    index = 0
    for edge in edges:
        if value > edge:
            index += 1
    return index


def _difficulty_matched_comparison(
    development_rows: Sequence[dict[str, Any]],
    sealed_rows: Sequence[dict[str, Any]],
    config: SemanticRelationAuditConfig,
    *,
    label: str,
) -> dict[str, Any]:
    """Compare development vs. sealed top-1 within matched (relation, cosine-bin) cells.

    Uses only frozen-model descriptors (`semantic_relation_id`,
    `key_cosine_similarity`) to define the matching cells -- never query
    outcome labels, per D2-003.3.
    """
    pooled_cosine = [float(row["key_cosine_similarity"]) for row in list(development_rows) + list(sealed_rows)]
    edges = _quantile_edges(pooled_cosine, config.cosine_bin_count)

    def cell_key(row: dict[str, Any]) -> tuple[str, int]:
        return (str(row["semantic_relation_id"]), _bin_label(float(row["key_cosine_similarity"]), edges))

    development_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sealed_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in development_rows:
        development_by_cell.setdefault(cell_key(row), []).append(row)
    for row in sealed_rows:
        sealed_by_cell.setdefault(cell_key(row), []).append(row)

    shared_cells = sorted(set(development_by_cell) & set(sealed_by_cell))
    per_cell: list[dict[str, Any]] = []
    weighted_gap_numerator = 0.0
    weighted_gap_denominator = 0
    for relation, bin_index in shared_cells:
        development_cell = development_by_cell[(relation, bin_index)]
        sealed_cell = sealed_by_cell[(relation, bin_index)]
        development_top1 = statistics.fmean(float(row["top1_correct"]) for row in development_cell)
        sealed_top1 = statistics.fmean(float(row["top1_correct"]) for row in sealed_cell)
        gap = development_top1 - sealed_top1
        weighted_gap_numerator += gap * len(sealed_cell)
        weighted_gap_denominator += len(sealed_cell)
        per_cell.append(
            {
                "relation": relation,
                "cosine_bin": bin_index,
                "n_development": len(development_cell),
                "n_sealed": len(sealed_cell),
                "development_top1": development_top1,
                "sealed_top1": sealed_top1,
                "gap": gap,
            }
        )

    raw_development_top1 = (
        statistics.fmean(float(row["top1_correct"]) for row in development_rows) if development_rows else None
    )
    raw_sealed_top1 = statistics.fmean(float(row["top1_correct"]) for row in sealed_rows) if sealed_rows else None
    raw_gap = (
        raw_development_top1 - raw_sealed_top1
        if raw_development_top1 is not None and raw_sealed_top1 is not None
        else None
    )
    matched_gap = (weighted_gap_numerator / weighted_gap_denominator) if weighted_gap_denominator else None

    return {
        "label": label,
        "cosine_bin_edges": edges,
        "raw_development_top1": raw_development_top1,
        "raw_sealed_top1": raw_sealed_top1,
        "raw_gap": raw_gap,
        "matched_gap": matched_gap,
        "shared_cell_count": len(shared_cells),
        "per_cell": per_cell,
    }


def _classify_matching(matched: dict[str, Any], config: SemanticRelationAuditConfig) -> str:
    """Apply the D2-003.3 interpretation logic to one matched comparison."""
    raw_gap, matched_gap = matched["raw_gap"], matched["matched_gap"]
    if raw_gap is None or matched_gap is None or matched["shared_cell_count"] == 0:
        return "UNRESOLVED"
    if abs(raw_gap) <= config.match_gap_tolerance:
        return "SEED_SPLIT_SUFFICIENT"
    if abs(matched_gap) <= config.match_gap_tolerance:
        return "DIFFICULTY_MATCHING_REQUIRED"
    return "MODEL_GENERALIZATION_FAILURE_AFTER_MATCHING"


# ---------------------------------------------------------------------------
# D2-003.4 -- combined verdict (one or more labels, per the task doc).
# ---------------------------------------------------------------------------


def _final_verdict(
    relation_summary: dict[str, Any],
    holdout: dict[str, Any],
    matched_post_repair: dict[str, Any],
    matched_pre_repair: dict[str, Any],
    config: SemanticRelationAuditConfig,
) -> dict[str, Any]:
    focus = relation_summary.get("regate_sealed/R2_frozen_post_repair", {})
    spread = focus.get("cross_relation_spread")
    if spread is None:
        spread_classification = "UNRESOLVED"
    elif spread >= config.relation_spread_threshold:
        spread_classification = "SEMANTIC_RELATION_HOLDOUT_REQUIRED"
    else:
        spread_classification = "SEED_SPLIT_SUFFICIENT"

    holdout_classification, holdout_evidence = _classify_relation_holdout(holdout, config)
    matched_post_classification = _classify_matching(matched_post_repair, config)
    matched_pre_classification = _classify_matching(matched_pre_repair, config)

    labels = {spread_classification, holdout_classification, matched_post_classification}
    if len(labels) > 1:
        labels.discard("UNRESOLVED")
    if not labels:
        labels = {"UNRESOLVED"}

    return {
        "labels": sorted(labels),
        "semantic_relation_holdout_required": "SEMANTIC_RELATION_HOLDOUT_REQUIRED" in labels,
        "evidence": {
            "cross_relation_spread_at_focus_cell": spread,
            "spread_based_classification": spread_classification,
            "relation_holdout_probe_classification": holdout_classification,
            "relation_holdout_probe_evidence": holdout_evidence,
            "post_repair_matched_classification": matched_post_classification,
            "pre_repair_matched_classification": matched_pre_classification,
        },
    }


def run_semantic_relation_audit(config: SemanticRelationAuditConfig) -> dict[str, Any]:
    """Run the B-C005D2-003 L3 semantic-relation generalization audit."""
    start = time.perf_counter()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    rows = _collect_all_rows(config)
    taxonomy = _relation_taxonomy()
    relation_summary = _relation_group_summary(rows)
    holdout = _relation_holdout_generalization(rows, config)

    development_post = [
        row for row in rows if row["partition"] == "development" and row["router_state"] == _POST_REPAIR
    ]
    sealed_post = [
        row for row in rows if row["partition"] == "regate_sealed" and row["router_state"] == _POST_REPAIR
    ]
    development_pre = [
        row for row in rows if row["partition"] == "development" and row["router_state"] == _PRE_REPAIR
    ]
    sealed_pre = [
        row for row in rows if row["partition"] == "original_sealed" and row["router_state"] == _PRE_REPAIR
    ]

    matched_post_repair = _difficulty_matched_comparison(
        development_post, sealed_post, config, label="development_post_repair_vs_regate_sealed_post_repair"
    )
    matched_pre_repair = _difficulty_matched_comparison(
        development_pre, sealed_pre, config, label="development_pre_repair_vs_original_sealed_pre_repair"
    )

    verdict = _final_verdict(relation_summary, holdout, matched_post_repair, matched_pre_repair, config)

    report = {
        "task_id": "B-C005D2-003",
        "config": config.to_dict(),
        "rows_analyzed": len(rows),
        "relation_taxonomy": taxonomy,
        "relation_group_summary": relation_summary,
        "relation_holdout_generalization": holdout,
        "difficulty_matched_comparison": {
            "post_repair": matched_post_repair,
            "pre_repair": matched_pre_repair,
        },
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        import yaml

        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        protocol = {
            "task_id": "B-C005D2-003",
            "sealed_partitions": {
                "original": list(config.original_sealed_seeds),
                "regate": list(config.regate_sealed_seeds),
            },
            "development_partition": list(config.development_seeds),
            "sealed_training_prohibited": True,
            "probes_are_evaluation_only": True,
            "matching_uses_outcome_labels": False,
        }
        (out / "semantic_relation_audit_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (out / "semantic_relation_audit_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        (out / "semantic_relation_audit_system.json").write_text(
            json.dumps(get_system_info(), indent=2) + "\n", encoding="utf-8"
        )
        (out / "semantic_relation_audit_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (out / "semantic_relation_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report
