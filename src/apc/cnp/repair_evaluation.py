"""Cell-preserving evidence and gates for the schedule/retention diagnostics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from apc.cnp.data import CNPRecord
from apc.cnp.repair_protocol import parse_condition

Panel = Literal["new", "old"]


@dataclass(frozen=True, order=True)
class CellKey:
    """One non-aggregated shadow evaluation cell."""

    domain_id: str
    query_id: str
    length: int
    threshold: str

    @classmethod
    def from_record(cls, record: CNPRecord) -> CellKey:
        domain, query = parse_condition(record.condition_key)
        return cls(domain, query, record.state.width, f"{float(record.arguments.threshold[0]):.2f}")

    def text(self) -> str:
        return (
            f"domain={self.domain_id}|query={self.query_id}|length={self.length}|"
            f"threshold={self.threshold}"
        )


@dataclass(frozen=True)
class SetEvidence:
    """Confusion evidence for one unpadded set in a single cell."""

    input_id: str
    f1: float
    exact_match: bool
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    valid_items: int


@dataclass(frozen=True)
class CellMetrics:
    """Aggregate metrics plus the per-set evidence needed for paired bootstrap."""

    sets: tuple[SetEvidence, ...]

    def summary(self) -> dict[str, float | int | None]:
        tp = sum(item.true_positive for item in self.sets)
        tn = sum(item.true_negative for item in self.sets)
        fp = sum(item.false_positive for item in self.sets)
        fn = sum(item.false_negative for item in self.sets)
        positives = tp + fn
        negatives = tn + fp
        balanced = None
        if positives and negatives:
            balanced = 0.5 * (tp / positives + tn / negatives)
        count = len(self.sets)
        return {
            "sets": count,
            "valid_items": sum(item.valid_items for item in self.sets),
            "positive_items": positives,
            "negative_items": negatives,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "balanced_accuracy": balanced,
            "mean_set_f1": None if not count else sum(item.f1 for item in self.sets) / count,
            "exact_match": None
            if not count
            else sum(item.exact_match for item in self.sets) / count,
        }


def evidence_from_masks(
    *, input_id: str, predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> SetEvidence:
    """Compute one set's evidence while excluding padding from every metric."""

    if predicted.shape != target.shape or target.shape != valid.shape:
        raise ValueError("prediction, target, and valid tensors must have equal shapes")
    if predicted.ndim != 1:
        raise ValueError("set evidence expects one-dimensional tensors")
    predicted = predicted.bool() & valid.bool()
    target = target.bool() & valid.bool()
    tp = int((predicted & target).sum())
    tn = int((~predicted & ~target & valid).sum())
    fp = int((predicted & ~target & valid).sum())
    fn = int((~predicted & target).sum())
    denominator = 2 * tp + fp + fn
    return SetEvidence(
        input_id=input_id,
        f1=1.0 if denominator == 0 else 2 * tp / denominator,
        exact_match=bool(torch.equal(predicted, target)),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        valid_items=int(valid.sum()),
    )


def build_cell_metrics(rows: list[tuple[CellKey, SetEvidence]]) -> dict[CellKey, CellMetrics]:
    """Construct cells without collapsing their query dimension."""

    grouped: dict[CellKey, list[SetEvidence]] = defaultdict(list)
    for cell, evidence in rows:
        grouped[cell].append(evidence)
    return {
        cell: CellMetrics(tuple(sorted(evidence, key=lambda item: item.input_id)))
        for cell, evidence in grouped.items()
    }


def validate_panel(
    cells: dict[CellKey, CellMetrics], *, expected_sets: int = 128
) -> dict[str, object]:
    """Validate the full panel before a gate can be interpreted."""

    expected_cells = 8 * 5 * 3
    invalid: list[str] = []
    if len(cells) != expected_cells:
        invalid.append(f"cell_count={len(cells)}")
    for cell, metrics in sorted(cells.items()):
        summary = metrics.summary()
        if (
            summary["sets"] != expected_sets
            or not summary["positive_items"]
            or not summary["negative_items"]
        ):
            invalid.append(cell.text())
    return {"status": "PASS" if not invalid else "INCONCLUSIVE_PANEL", "invalid_cells": invalid}


def quality_pass(metrics: CellMetrics) -> bool:
    """Apply the non-negotiable BA and set-F1 floors to one complete cell."""

    summary = metrics.summary()
    balanced = summary["balanced_accuracy"]
    f1 = summary["mean_set_f1"]
    return isinstance(balanced, float) and balanced >= 0.95 and isinstance(f1, float) and f1 >= 0.90


def _mean_f1(metrics: CellMetrics) -> float:
    value = metrics.summary()["mean_set_f1"]
    if not isinstance(value, float):
        raise ValueError("cell has no finite mean set F1")
    return value


def paired_bootstrap_delta(
    parent: CellMetrics, candidate: CellMetrics, *, root: int, replicates: int = 1000
) -> dict[str, float | int | None]:
    """Return diagnostic paired-F1 bootstrap intervals using a fixed local RNG."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    parent_sets = {item.input_id: item for item in parent.sets}
    candidate_sets = {item.input_id: item for item in candidate.sets}
    if set(parent_sets) != set(candidate_sets) or not parent_sets:
        raise ValueError("paired bootstrap requires exactly matched non-empty input IDs")
    input_ids = sorted(parent_sets)
    random_source = random.Random(root)
    deltas: list[float] = []
    for _ in range(replicates):
        sample = [input_ids[random_source.randrange(len(input_ids))] for _ in input_ids]
        deltas.append(
            sum(candidate_sets[item].f1 - parent_sets[item].f1 for item in sample) / len(sample)
        )
    deltas.sort()

    def percentile(percent: float) -> float:
        position = (len(deltas) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        return deltas[lower] + (deltas[upper] - deltas[lower]) * (position - lower)

    return {
        "replicates": replicates,
        "candidate_minus_parent_f1": _mean_f1(candidate) - _mean_f1(parent),
        "lower_2_5": percentile(0.025),
        "upper_97_5": percentile(0.975),
    }


def _transition(parent: CellMetrics, candidate: CellMetrics) -> str:
    parent_pass = quality_pass(parent)
    candidate_pass = quality_pass(candidate)
    if parent_pass and candidate_pass:
        return "CONTINUED_PASS"
    if parent_pass and not candidate_pass:
        return "NEW_FAILURE"
    if not parent_pass and candidate_pass:
        return "RECOVERED"
    return "CONTINUED_FAILURE"


def gate_report(
    *,
    parent_old: dict[CellKey, CellMetrics],
    candidate_new: dict[CellKey, CellMetrics],
    candidate_old: dict[CellKey, CellMetrics],
    invariant_status: bool,
) -> dict[str, object]:
    """Keep panel validity, parent eligibility, quality, and retention separate."""

    old_parent_panel = validate_panel(parent_old)
    new_panel = validate_panel(candidate_new)
    old_candidate_panel = validate_panel(candidate_old)
    panel_valid = all(
        panel["status"] == "PASS" for panel in (old_parent_panel, new_panel, old_candidate_panel)
    )
    if set(parent_old) != set(candidate_old):
        raise ValueError("parent and candidate old panels must have the same cells")
    parent_failed = [
        cell.text() for cell, value in sorted(parent_old.items()) if not quality_pass(value)
    ]
    new_failed = [
        cell.text() for cell, value in sorted(candidate_new.items()) if not quality_pass(value)
    ]
    old_failed = [
        cell.text() for cell, value in sorted(candidate_old.items()) if not quality_pass(value)
    ]
    retention_failed = [
        cell.text()
        for cell in sorted(parent_old)
        if _mean_f1(candidate_old[cell]) < _mean_f1(parent_old[cell]) - 0.01
    ]
    transitions = {
        cell.text(): _transition(parent_old[cell], candidate_old[cell])
        for cell in sorted(parent_old)
    }
    candidate_gate = (
        panel_valid
        and not new_failed
        and not old_failed
        and not retention_failed
        and invariant_status
    )
    return {
        "panel_validity": {
            "status": "PASS" if panel_valid else "INCONCLUSIVE_PANEL",
            "panels": {
                "parent_old": old_parent_panel,
                "candidate_new": new_panel,
                "candidate_old": old_candidate_panel,
            },
        },
        "parent_old_quality": {
            "status": "PASS" if not parent_failed else "PARENT_INELIGIBLE",
            "failed_cells": parent_failed,
        },
        "new_quality": {"status": "PASS" if not new_failed else "FAIL", "failed_cells": new_failed},
        "old_quality": {"status": "PASS" if not old_failed else "FAIL", "failed_cells": old_failed},
        "old_retention": {
            "status": "PASS" if not retention_failed else "FAIL",
            "failed_cells": retention_failed,
        },
        "invariants": {"status": "PASS" if invariant_status else "INVALID_RUN_STOP"},
        "old_transitions": transitions,
        "candidate_gate": "PASS" if candidate_gate else "FAIL",
    }


def serialize_cells(cells: dict[CellKey, CellMetrics]) -> dict[str, object]:
    """Serialize all per-set and per-cell evidence for a run artifact."""

    return {
        cell.text(): {"summary": metrics.summary(), "sets": [asdict(item) for item in metrics.sets]}
        for cell, metrics in sorted(cells.items())
    }
