"""Pre-update evidence, fixed-panel traces, and parameter accounting for CNP repairs."""

from __future__ import annotations

from collections import defaultdict

import torch

from apc.cnp.data import CNPRecord, canonical_json_hash
from apc.cnp.repair import _collate_records, set_mean_masked_bce_loss
from apc.cnp.repair_evaluation import CellKey
from apc.cnp.repair_retention import retention_loss


def parameter_accounting(model: torch.nn.Module) -> dict[str, int]:
    """Report resident, frozen, and update-eligible parameter counts separately."""

    parameters = list(model.named_parameters())
    resident_parameters = sum(parameter.numel() for _, parameter in parameters)
    update_eligible_parameters = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    frozen_parameters = resident_parameters - update_eligible_parameters
    return {
        "resident_parameters": resident_parameters,
        "resident_weight_bytes": sum(
            parameter.numel() * parameter.element_size() for _, parameter in parameters
        ),
        "update_eligible_parameters": update_eligible_parameters,
        "update_eligible_weight_bytes": sum(
            parameter.numel() * parameter.element_size()
            for _, parameter in parameters
            if parameter.requires_grad
        ),
        "frozen_parameters": frozen_parameters,
        "temporary_parameters": update_eligible_parameters,
    }


def preflight_panel_evidence(
    panels: dict[str, list[CNPRecord]], *, expected_shadow_sets: int
) -> dict[str, object]:
    """Audit labels and support before constructing a model or optimizer.

    This consumes already generated teachers only.  It does not invoke a model,
    alter a split, or use labels to select examples.
    """

    panel_evidence: dict[str, object] = {}
    for name, records in sorted(panels.items()):
        cells: dict[str, dict[str, int]] = defaultdict(
            lambda: {"sets": 0, "positive_items": 0, "negative_items": 0}
        )
        for record in records:
            key = CellKey.from_record(record).text()
            valid = record.state.valid
            target = record.target.bool() & valid
            cells[key]["sets"] += 1
            cells[key]["positive_items"] += int(target.sum())
            cells[key]["negative_items"] += int((~target & valid).sum())
        invalid = [
            key
            for key, value in cells.items()
            if not value["positive_items"]
            or not value["negative_items"]
            or (
                name.endswith("shadow")
                and value["sets"] != expected_shadow_sets
            )
        ]
        if invalid:
            raise ValueError(f"preflight panel support failed for {name}: {sorted(invalid)[:3]}")
        panel_evidence[name] = {
            "records": len(records),
            "input_digest_hash": canonical_json_hash(
                sorted(record.input_digest() for record in records)
            ),
            "record_digest_hash": canonical_json_hash(
                sorted(record.digest() for record in records)
            ),
            "cells": {key: value for key, value in sorted(cells.items())},
        }
    return {"status": "PASS", "panels": panel_evidence}


def initial_output_parity(
    parent: torch.nn.Module,
    candidate: torch.nn.Module,
    *,
    records: list[CNPRecord],
    device: torch.device,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Check zero-initialized adapter parity on a fixed non-shadow probe."""

    if not records:
        raise ValueError("initial parity requires a non-empty probe")
    parent.eval()
    candidate.eval()
    max_logit_difference = 0.0
    masks_match = True
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            state, arguments, _ = _collate_records(records[start : start + 32], device)
            parent_result = parent(state, arguments)
            candidate_result = candidate(state, arguments)
            max_logit_difference = max(
                max_logit_difference,
                float((parent_result.logits - candidate_result.logits).abs().max().cpu()),
            )
            masks_match = masks_match and bool(
                torch.equal(parent_result.selected, candidate_result.selected)
            )
    status = masks_match and max_logit_difference <= atol + rtol * max(1.0, max_logit_difference)
    return {
        "status": "PASS" if status else "FAIL",
        "records": len(records),
        "max_logit_abs_difference": max_logit_difference,
        "selection_masks_match": masks_match,
        "atol": atol,
        "rtol": rtol,
    }


def fixed_panel_trace(
    candidate: torch.nn.Module,
    parent: torch.nn.Module,
    *,
    new_records: list[CNPRecord],
    replay_records: list[CNPRecord],
    device: torch.device,
    retention_weight: float,
    step: int,
) -> dict[str, float | int]:
    """Measure registered losses on fixed panels without checkpoint selection."""

    if not new_records or not replay_records or len(new_records) % 32 or len(replay_records) % 32:
        raise ValueError("fixed trace requires non-empty panels divisible by 32")
    candidate.eval()
    parent.eval()

    def aggregate(records: list[CNPRecord], *, replay: bool) -> tuple[float, float, float, float]:
        task_values: list[float] = []
        keep_values: list[float] = []
        correct_deltas: list[float] = []
        incorrect_deltas: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(records), 32):
                state, arguments, target = _collate_records(records[start : start + 32], device)
                candidate_result = candidate(state, arguments)
                task_values.append(
                    float(
                        set_mean_masked_bce_loss(
                            candidate_result.logits, target, state.valid
                        ).cpu()
                    )
                )
                if replay:
                    parent_result = parent(state, arguments)
                    keep_values.append(
                        float(
                            retention_loss(
                                candidate_result.logits, parent_result.logits, state.valid
                            ).cpu()
                        )
                    )
                    for index in range(state.batch_size):
                        valid = state.valid[index]
                        delta = (
                            candidate_result.logits[index] - parent_result.logits[index]
                        ).square()
                        mean_delta = float(delta[valid].mean().cpu())
                        parent_correct = bool(
                            torch.equal(
                                parent_result.selected[index] & valid,
                                target[index].bool() & valid,
                            )
                        )
                        (correct_deltas if parent_correct else incorrect_deltas).append(mean_delta)
        return (
            sum(task_values) / len(task_values),
            0.0 if not keep_values else sum(keep_values) / len(keep_values),
            0.0 if not correct_deltas else sum(correct_deltas) / len(correct_deltas),
            0.0 if not incorrect_deltas else sum(incorrect_deltas) / len(incorrect_deltas),
        )

    new_bce, _, _, _ = aggregate(new_records, replay=False)
    replay_bce, keep_mse, parent_correct_mse, parent_incorrect_mse = aggregate(
        replay_records, replay=True
    )
    return {
        "step": step,
        "new_bce": new_bce,
        "replay_bce": replay_bce,
        "keep_mse": keep_mse,
        "weighted_keep_mse": retention_weight * keep_mse,
        "parent_correct_replay_logit_mse": parent_correct_mse,
        "parent_incorrect_replay_logit_mse": parent_incorrect_mse,
    }
