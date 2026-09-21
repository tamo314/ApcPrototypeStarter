"""Metric calculations and causal-control bookkeeping for future CNP runners."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from apc.cnp.contracts import SelectionResult


@dataclass(frozen=True)
class SelectionMetrics:
    balanced_accuracy: float | None
    mean_set_f1: float
    exact_match: float
    valid_items: int


def selection_metrics(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> SelectionMetrics:
    """Measure element and per-set quality without treating padding as negatives."""

    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError("predicted, target, and valid must have identical shapes")
    if predicted.dtype != torch.bool or target.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError("selection metrics require boolean masks")
    valid_items = int(valid.sum())
    if valid_items == 0:
        raise ValueError("cannot calculate selection metrics without valid items")
    positives = target & valid
    negatives = ~target & valid
    true_positive = int((predicted & positives).sum())
    true_negative = int((~predicted & negatives).sum())
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    balanced = None
    if positive_count and negative_count:
        balanced = 0.5 * (true_positive / positive_count + true_negative / negative_count)
    f1_values: list[float] = []
    exact_values: list[float] = []
    for batch_index in range(predicted.shape[0]):
        mask = valid[batch_index]
        pred = predicted[batch_index] & mask
        gold = target[batch_index] & mask
        intersection = int((pred & gold).sum())
        denominator = int(pred.sum() + gold.sum())
        f1_values.append(1.0 if denominator == 0 else 2.0 * intersection / denominator)
        exact_values.append(float(torch.equal(pred, gold)))
    return SelectionMetrics(
        balanced_accuracy=balanced,
        mean_set_f1=sum(f1_values) / len(f1_values),
        exact_match=sum(exact_values) / len(exact_values),
        valid_items=valid_items,
    )


@dataclass(frozen=True)
class CausalAccuracy:
    correct: float
    intervention: float
    effectful_items: int


def effectful_causal_accuracy(
    target: torch.Tensor,
    valid: torch.Tensor,
    correct: SelectionResult,
    intervention: SelectionResult,
    reference_intervention: SelectionResult,
) -> CausalAccuracy:
    """Measure one intervention on its own reference-defined effectful support."""

    if target.shape != valid.shape:
        raise ValueError("target and valid shapes must match")
    changed = valid & (target != reference_intervention.selected)
    count = int(changed.sum())
    if count == 0:
        raise ValueError("no effectful elements available for a causal comparison")

    def accuracy(mask: torch.Tensor) -> float:
        return float((mask[changed] == target[changed]).to(torch.float32).mean())

    return CausalAccuracy(
        correct=accuracy(correct.selected),
        intervention=accuracy(intervention.selected),
        effectful_items=count,
    )
