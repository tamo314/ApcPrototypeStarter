"""Contract-aligned CNP adaptation components for a separately scoped repair run.

These classes do not replace CNP v1.  They make the corrected adapter placement,
set-weighted objective, replay rule, and cell-level shadow gate available without
altering the code path used to produce the historical CNP-004 artifacts.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from apc.cnp.contracts import SelectArguments, SelectionResult, SetState
from apc.cnp.data import CNPRecord, phi
from apc.cnp.primitive import ConditionalSelectPrimitive, ResidualAdapter

CORRECTED_ARCHITECTURE_SIGNATURE = "cnp_conditional_select_mlp_adaptation_repair_v1"


@dataclass
class _MetricsAccumulator:
    """Small local accumulator that keeps repair checks independent of WSL-only runners."""

    true_positive: int = 0
    true_negative: int = 0
    positives: int = 0
    negatives: int = 0
    f1_sum: float = 0.0
    exact_sum: float = 0.0
    sets: int = 0
    valid_items: int = 0

    def add(self, predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        predicted = predicted & valid
        target = target & valid
        positives = target
        negatives = ~target & valid
        self.true_positive += int((predicted & positives).sum())
        self.true_negative += int((~predicted & negatives).sum())
        self.positives += int(positives.sum())
        self.negatives += int(negatives.sum())
        self.valid_items += int(valid.sum())
        intersection = (predicted & target).sum(dim=1).to(torch.float32)
        denominator = (predicted.sum(dim=1) + target.sum(dim=1)).to(torch.float32)
        f1 = torch.where(
            denominator == 0, torch.ones_like(denominator), 2 * intersection / denominator
        )
        self.f1_sum += float(f1.sum())
        self.exact_sum += float((predicted == target).all(dim=1).sum())
        self.sets += predicted.shape[0]

    def result(self) -> dict[str, float | int | None]:
        balanced = None
        if self.positives and self.negatives:
            balanced = 0.5 * (
                self.true_positive / self.positives + self.true_negative / self.negatives
            )
        return {
            "balanced_accuracy": balanced,
            "mean_set_f1": self.f1_sum / self.sets,
            "exact_match": self.exact_sum / self.sets,
            "valid_items": self.valid_items,
        }


class CorrectedConditionalSelectPrimitive(ConditionalSelectPrimitive):
    """CNP selector whose local adapter acts after the first hidden layer."""

    ARCHITECTURE_SIGNATURE = CORRECTED_ARCHITECTURE_SIGNATURE

    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        arguments.validate_batch_size(state.batch_size)
        self.forward_call_count += 1
        if not self.enabled:
            logits = torch.full(
                (state.batch_size, state.width),
                float("-inf"),
                dtype=state.values.dtype,
                device=state.values.device,
            )
            selected = torch.zeros_like(state.valid)
            return SelectionResult(
                logits=logits, selected=selected, state=state.with_valid(selected)
            )

        content = phi(state.values)
        query = phi(arguments.query).unsqueeze(1).expand(-1, state.width, -1)
        threshold = arguments.threshold.reshape(-1, 1, 1).expand(-1, state.width, 1)
        features = torch.cat(
            (content, query, torch.abs(content - query), content * query, threshold), dim=-1
        )
        hidden = torch.nn.functional.gelu(self.input_proj(features))
        if self.adapter is not None:
            hidden = self.adapter(hidden)
        hidden = torch.nn.functional.gelu(self.hidden_proj(hidden))
        logits = self.readout(hidden).squeeze(-1).to(torch.float32)
        selected = state.valid & (logits >= 0.0)
        return SelectionResult(logits=logits, selected=selected, state=state.with_valid(selected))


def clone_corrected_local_candidate(
    parent: CorrectedConditionalSelectPrimitive,
) -> CorrectedConditionalSelectPrimitive:
    """Copy a corrected parent and expose exactly one first-hidden-layer adapter."""

    candidate = copy.deepcopy(parent)
    if candidate.adapter is None:
        candidate.attach_adapter(ResidualAdapter())
    candidate.freeze_base()
    return candidate


def set_mean_masked_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Average BCE within each non-empty set, then equally across sets."""

    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("logits, target, and valid must have the same shape")
    valid_counts = valid.sum(dim=1)
    non_empty = valid_counts > 0
    if not bool(non_empty.any()):
        raise ValueError("cannot compute a training loss for an all-invalid batch")
    losses = functional.binary_cross_entropy_with_logits(
        logits, target.to(torch.float32), reduction="none"
    )
    per_set = (losses * valid).sum(dim=1) / valid_counts.clamp_min(1)
    return per_set[non_empty].mean()


def corrected_one_training_step(
    primitive: CorrectedConditionalSelectPrimitive,
    state: SetState,
    arguments: SelectArguments,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform one contract-aligned update for unit testing or an authorized run."""

    optimizer.zero_grad(set_to_none=True)
    result = primitive(state, arguments)
    loss = set_mean_masked_bce_loss(result.logits, target, state.valid)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(primitive.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def replay_stratum(record: CNPRecord) -> tuple[str, float]:
    """Group source records by observable query condition and threshold only."""

    return record.condition_key, round(float(record.arguments.threshold[0]), 2)


def select_stratified_replay_records(
    records: list[CNPRecord], *, buffer_size: int
) -> list[CNPRecord]:
    """Choose the lowest-digest records equally from each source condition cell."""

    if buffer_size <= 0:
        raise ValueError("replay buffer_size must be positive")
    strata: dict[tuple[str, float], list[CNPRecord]] = defaultdict(list)
    for record in records:
        strata[replay_stratum(record)].append(record)
    if not strata or buffer_size % len(strata):
        raise ValueError("replay buffer must divide evenly across source condition cells")
    per_stratum = buffer_size // len(strata)
    insufficient = {
        f"{condition}|threshold={threshold:.2f}": len(group)
        for (condition, threshold), group in strata.items()
        if len(group) < per_stratum
    }
    if insufficient:
        raise ValueError(f"insufficient source records for replay strata: {insufficient}")
    selected = [
        record
        for key in sorted(strata)
        for record in sorted(strata[key], key=CNPRecord.digest)[:per_stratum]
    ]
    if len(selected) != buffer_size:
        raise RuntimeError("stratified replay selected an unexpected number of records")
    return selected


def cell_key(record: CNPRecord) -> str:
    """Use a domain, length, and threshold key for every adaptation gate."""

    return (
        f"domain={record.condition_key}|length={record.state.width}|"
        f"threshold={float(record.arguments.threshold[0]):.2f}"
    )


def _collate_records(
    records: list[CNPRecord], device: torch.device
) -> tuple[SetState, SelectArguments, torch.Tensor]:
    """Pad a small evaluation batch without exposing record metadata to the model."""

    if not records:
        raise ValueError("cannot collate an empty repair evaluation batch")
    width = max(record.state.width for record in records)
    batch_size = len(records)
    values = torch.zeros((batch_size, width, 8), dtype=torch.float32, device=device)
    valid = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    item_ids = torch.zeros((batch_size, width), dtype=torch.int64, device=device)
    target = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    query = torch.empty((batch_size, 8), dtype=torch.float32, device=device)
    threshold = torch.empty((batch_size,), dtype=torch.float32, device=device)
    for index, record in enumerate(records):
        record_width = record.state.width
        values[index, :record_width] = record.state.values[0].to(device)
        valid[index, :record_width] = record.state.valid[0].to(device)
        item_ids[index, :record_width] = record.state.item_ids[0].to(device)
        target[index, :record_width] = record.target[0].to(device)
        query[index] = record.arguments.query[0].to(device)
        threshold[index] = record.arguments.threshold[0].to(device)
    return (
        SetState(values=values, valid=valid, item_ids=item_ids),
        SelectArguments(query=query, threshold=threshold),
        target,
    )


def evaluate_panel_by_cell(
    model: nn.Module, records: list[CNPRecord], *, device: torch.device
) -> dict[str, Any]:
    """Return aggregate and domain/length/threshold metrics from one fixed panel."""

    if not records:
        raise ValueError("cannot evaluate an empty adaptation panel")
    aggregate = _MetricsAccumulator()
    cells: dict[str, _MetricsAccumulator] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            record_batch = records[start : start + 32]
            state, arguments, target = _collate_records(record_batch, device)
            result = model(state, arguments)
            aggregate.add(result.selected, target, state.valid)
            for index, record in enumerate(record_batch):
                width = record.state.width
                cells.setdefault(cell_key(record), _MetricsAccumulator()).add(
                    result.selected[index : index + 1, :width],
                    target[index : index + 1, :width],
                    state.valid[index : index + 1, :width],
                )
    return {
        "aggregate": aggregate.result(),
        "cells": {key: accumulator.result() for key, accumulator in sorted(cells.items())},
    }


def quality_floor(metrics: dict[str, float | int | None]) -> bool:
    """Apply the registered balanced-accuracy and mean-set-F1 floor to one cell."""

    balanced = metrics.get("balanced_accuracy")
    f1 = metrics.get("mean_set_f1")
    return isinstance(balanced, float) and balanced >= 0.95 and isinstance(f1, float) and f1 >= 0.90


def evaluate_shadow_gate(
    candidate: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless every aligned cell meets quality and retention requirements."""

    candidate_cells = candidate.get("cells")
    parent_cells = parent.get("cells")
    if not isinstance(candidate_cells, dict) or not isinstance(parent_cells, dict):
        raise ValueError("candidate and parent shadow reports require cell metrics")
    if set(candidate_cells) != set(parent_cells) or not candidate_cells:
        raise ValueError("candidate and parent shadow cell keys must match and be non-empty")
    failed_quality = sorted(
        key for key, metrics in candidate_cells.items() if not quality_floor(metrics)
    )
    failed_retention = sorted(
        key
        for key in candidate_cells
        if (
            float(candidate_cells[key]["mean_set_f1"])
            < float(parent_cells[key]["mean_set_f1"]) - 0.01
        )
    )
    return {
        "status": "PASS" if not failed_quality and not failed_retention else "FAIL",
        "quality_failed_cells": failed_quality,
        "retention_failed_cells": failed_retention,
    }
