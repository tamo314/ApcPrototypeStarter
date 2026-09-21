from __future__ import annotations

import torch
from torch import nn

from apc.cnp.contracts import SelectArguments, SelectionResult, SetState
from apc.cnp.data import CNPRecord
from apc.cnp.repair import (
    CORRECTED_ARCHITECTURE_SIGNATURE,
    CorrectedConditionalSelectPrimitive,
    cell_key,
    clone_corrected_local_candidate,
    evaluate_panel_by_cell,
    evaluate_shadow_gate,
    select_stratified_replay_records,
    set_mean_masked_bce_loss,
)


def _record(condition: str, threshold: float, item: float, target: bool) -> CNPRecord:
    state = SetState(
        values=torch.full((1, 1, 8), item, dtype=torch.float32),
        valid=torch.tensor([[True]]),
        item_ids=torch.tensor([[0]], dtype=torch.int64),
    )
    return CNPRecord(
        state=state,
        arguments=SelectArguments(
            query=torch.zeros((1, 8), dtype=torch.float32),
            threshold=torch.tensor([threshold]),
        ),
        target=torch.tensor([[target]]),
        role="adapt_train",
        condition_key=condition,
    )


def test_set_mean_loss_weights_sets_equally_despite_length() -> None:
    logits = torch.zeros((2, 16), dtype=torch.float32)
    logits[1] = 2.0
    target = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.zeros_like(target)
    valid[0, 0] = True
    valid[1] = True
    target[0, 0] = True
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0, :1], target[0, :1].to(torch.float32)
    )
    expected += torch.nn.functional.binary_cross_entropy_with_logits(
        logits[1], target[1].to(torch.float32)
    )
    assert torch.allclose(set_mean_masked_bce_loss(logits, target, valid), expected / 2)


def test_corrected_adapter_operates_before_second_hidden_layer() -> None:
    torch.manual_seed(11)
    parent = CorrectedConditionalSelectPrimitive(1)
    candidate = clone_corrected_local_candidate(parent)
    assert candidate.architecture_signature == CORRECTED_ARCHITECTURE_SIGNATURE
    assert candidate.adapter is not None
    candidate.adapter.up.weight.data.fill_(0.2)
    state = SetState(
        values=torch.rand((1, 2, 8), dtype=torch.float32),
        valid=torch.tensor([[True, True]]),
        item_ids=torch.tensor([[0, 1]], dtype=torch.int64),
    )
    arguments = SelectArguments(torch.zeros((1, 8)), torch.tensor([0.5]))
    result = candidate(state, arguments)
    content = torch.cat((state.values, torch.sin(torch.pi * state.values)), dim=-1)
    query = torch.cat((arguments.query, torch.sin(torch.pi * arguments.query)), dim=-1).unsqueeze(1)
    query = query.expand(-1, state.width, -1)
    threshold = arguments.threshold.reshape(-1, 1, 1).expand(-1, state.width, 1)
    features = torch.cat((content, query, abs(content - query), content * query, threshold), dim=-1)
    first_hidden = torch.nn.functional.gelu(candidate.input_proj(features))
    adapted = candidate.adapter(first_hidden)
    expected_hidden = torch.nn.functional.gelu(candidate.hidden_proj(adapted))
    expected = candidate.readout(expected_hidden).squeeze(-1)
    assert torch.allclose(result.logits, expected)


def test_replay_selection_is_equal_per_condition_threshold_and_hash_ordered() -> None:
    records = [
        _record(condition, threshold, item, bool(index % 2))
        for index, (condition, threshold, item) in enumerate(
            (condition, threshold, item)
            for condition in ("source_00", "source_01")
            for threshold in (0.5, 0.8)
            for item in (0.1, 0.2, 0.3)
        )
    ]
    selected = select_stratified_replay_records(records, buffer_size=8)
    groups: dict[tuple[str, float], list[CNPRecord]] = {}
    for record in selected:
        key = record.condition_key, round(float(record.arguments.threshold[0]), 2)
        groups.setdefault(key, []).append(record)
    assert {key: len(value) for key, value in groups.items()} == {
        (condition, threshold): 2
        for condition in ("source_00", "source_01")
        for threshold in (0.5, 0.8)
    }
    for key, group in groups.items():
        candidates = [
            record
            for record in records
            if (record.condition_key, round(float(record.arguments.threshold[0]), 2)) == key
        ]
        expected_digests = sorted(record.digest() for record in candidates)[:2]
        assert [record.digest() for record in group] == expected_digests


class _ThresholdModel(nn.Module):
    def forward(self, state: SetState, arguments: SelectArguments) -> SelectionResult:
        del arguments
        selected = state.valid & (state.values[..., 0] >= 0.3)
        return SelectionResult(
            logits=state.values[..., 0] - 0.3,
            selected=selected,
            state=state.with_valid(selected),
        )


def test_cell_evaluation_and_gate_reject_a_failed_cell() -> None:
    records = [
        _record("q1+q2+_00", 0.5, 0.1, True),
        _record("q1+q2+_00", 0.5, 0.2, False),
        _record("q1+q2+_00", 0.8, 0.4, True),
        _record("q1+q2+_00", 0.8, 0.1, False),
    ]
    report = evaluate_panel_by_cell(_ThresholdModel(), records, device=torch.device("cpu"))
    assert set(report["cells"]) == {cell_key(record) for record in records}
    parent = {
        "cells": {
            key: {"balanced_accuracy": 1.0, "mean_set_f1": 1.0}
            for key in report["cells"]
        }
    }
    gate = evaluate_shadow_gate(report, parent)
    assert gate["status"] == "FAIL"
    assert gate["quality_failed_cells"] == [cell_key(records[0])]
    assert gate["retention_failed_cells"] == [cell_key(records[0])]
