from __future__ import annotations

from collections import Counter

import pytest
import torch

from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.data import CNPRecord
from apc.cnp.repair_evaluation import CellKey, CellMetrics, SetEvidence, gate_report
from apc.cnp.repair_retention import (
    ParentLogitCache,
    combined_decision_repair_loss,
    combined_repair_loss,
    parent_correct_decision_loss,
    retention_loss,
)
from apc.cnp.repair_schedule import ScheduleArm, build_side_schedule


def _record(query: int, threshold: float, index: int, target: bool) -> CNPRecord:
    value = (query * 1000 + int(threshold * 100) * 10 + index) / 10000
    state = SetState(
        values=torch.full((1, 1, 8), value, dtype=torch.float32),
        valid=torch.tensor([[True]]),
        item_ids=torch.tensor([[0]], dtype=torch.int64),
    )
    return CNPRecord(
        state=state,
        arguments=SelectArguments(
            query=torch.full((1, 8), float(query), dtype=torch.float32),
            threshold=torch.tensor([threshold], dtype=torch.float32),
        ),
        target=torch.tensor([[target]]),
        role="cnp_repair_schedule_v1_new_train",
        condition_key=f"domain=q1+q2+|query={query:02d}",
        schedule_index=index,
    )


def _panel() -> list[CNPRecord]:
    return [
        _record(query, threshold, index, bool((query + index) % 2))
        for query in range(8)
        for threshold in (0.5, 0.8, 1.1)
        for index in range(64)
    ]


def test_schedule_preserves_a_quotas_and_balances_c_independent_of_input_order() -> None:
    records = _panel()
    ordered = build_side_schedule(records, arm=ScheduleArm.ORDERED, side="new", root=620020)
    matched = build_side_schedule(
        list(reversed(records)), arm=ScheduleArm.DISPERSED_MATCHED, side="new", root=620020
    )
    balanced = build_side_schedule(
        list(reversed(records)), arm=ScheduleArm.DISPERSED_BALANCED, side="new", root=620020
    )
    assert ordered.quotas() == matched.quotas()
    assert [entry.input_id for entry in matched.entries] == [
        entry.input_id
        for entry in build_side_schedule(
            records, arm=ScheduleArm.DISPERSED_MATCHED, side="new", root=620020
        ).entries
    ]
    stratum_counts = Counter(entry.stratum_id for entry in balanced.entries)
    assert set(stratum_counts.values()) == {170, 171}
    assert len(balanced.entries) == 4096
    assert len(balanced.batches) == 256
    assert max(balanced.quotas().values()) - min(balanced.quotas().values()) <= 1


def test_schedule_ignores_target_values() -> None:
    records = _panel()
    flipped = [
        CNPRecord(
            state=record.state,
            arguments=record.arguments,
            target=~record.target,
            role=record.role,
            condition_key=record.condition_key,
            schedule_index=record.schedule_index,
        )
        for record in records
    ]
    first = build_side_schedule(
        records, arm=ScheduleArm.DISPERSED_BALANCED, side="new", root=620020
    )
    second = build_side_schedule(
        flipped, arm=ScheduleArm.DISPERSED_BALANCED, side="new", root=620020
    )
    assert first.sha256() == second.sha256()


def _perfect_cell() -> CellMetrics:
    return CellMetrics(
        tuple(
            [SetEvidence(f"p{index}", 1.0, True, 1, 0, 0, 0, 1) for index in range(64)]
            + [SetEvidence(f"n{index}", 1.0, True, 0, 1, 0, 0, 1) for index in range(64)]
        )
    )


def test_new_relative_drop_does_not_replace_old_quality_or_retention_gate() -> None:
    parent = _perfect_cell()
    weak_old = CellMetrics(
        tuple(
            [SetEvidence(f"p{index}", 0.0, False, 0, 0, 0, 1, 1) for index in range(64)]
            + [SetEvidence(f"n{index}", 1.0, True, 0, 1, 0, 0, 1) for index in range(64)]
        )
    )
    # Supply a complete 120-cell panel with unique queries.
    cells = {
        CellKey("source", f"{query:02d}", length, threshold): parent
        for query in range(8)
        for length in (1, 2, 4, 8, 16)
        for threshold in ("0.50", "0.80", "1.10")
    }
    candidate_old = dict(cells)
    failed = next(iter(candidate_old))
    candidate_old[failed] = weak_old
    candidate_new = {
        CellKey("q1+q2+", cell.query_id, cell.length, cell.threshold): value
        for cell, value in cells.items()
    }
    report = gate_report(
        parent_old=cells,
        candidate_new=candidate_new,
        candidate_old=candidate_old,
        invariant_status=True,
    )
    assert report["candidate_gate"] == "FAIL"
    assert report["old_quality"]["status"] == "FAIL"  # type: ignore[index]
    assert report["old_retention"]["status"] == "FAIL"  # type: ignore[index]


def test_retention_loss_is_set_weighted_and_parent_cache_fails_closed() -> None:
    candidate = torch.tensor([[1.0, 9.0], [2.0, 4.0]], requires_grad=True)
    parent = torch.tensor([[1.0, 3.0], [1.0, 2.0]])
    valid = torch.tensor([[True, False], [True, True]])
    assert torch.allclose(retention_loss(candidate, parent, valid), torch.tensor(1.25))
    retention_loss(candidate, parent, valid).backward()
    assert candidate.grad is not None
    assert not parent.requires_grad
    cache = ParentLogitCache("parent", "signature", ("a", "b"), parent)
    cache.validate(parent_hash="parent", architecture_signature="signature", input_ids=("a", "b"))
    with pytest.raises(ValueError, match="input IDs"):
        cache.validate(
            parent_hash="parent", architecture_signature="signature", input_ids=("b", "a")
        )
    task_logits = torch.zeros((1, 1), requires_grad=True)
    target = torch.zeros((1, 1), dtype=torch.bool)
    total, values = combined_repair_loss(
        new_logits=task_logits,
        new_target=target,
        new_valid=torch.ones((1, 1), dtype=torch.bool),
        replay_logits=task_logits,
        replay_target=target,
        replay_valid=torch.ones((1, 1), dtype=torch.bool),
        parent_replay_logits=torch.zeros((1, 1)),
        retention_weight=0.0,
    )
    assert values["retention"] == 0.0
    assert torch.allclose(
        total, torch.nn.functional.binary_cross_entropy_with_logits(task_logits, target.float())
    )


def test_parent_correct_decision_loss_preserves_only_correct_parent_decisions() -> None:
    candidate = torch.tensor([[-0.3, 0.2]], requires_grad=True)
    target = torch.tensor([[False, True]])
    parent = torch.tensor([[-1.0, -1.0]])  # first correct, second wrong
    valid = torch.tensor([[True, True]])
    decision = parent_correct_decision_loss(candidate, target, parent, valid, margin=0.5)
    assert torch.allclose(decision, torch.tensor(0.04))
    decision.backward()
    assert torch.allclose(candidate.grad, torch.tensor([[0.4, 0.0]]))

    total, values = combined_decision_repair_loss(
        new_logits=candidate.detach().clone().requires_grad_(),
        new_target=target,
        new_valid=valid,
        replay_logits=candidate.detach().clone().requires_grad_(),
        replay_target=target,
        replay_valid=valid,
        parent_replay_logits=parent,
        decision_weight=0.0,
        decision_margin=0.5,
    )
    assert values["decision"] == pytest.approx(0.04)
    expected_task = torch.nn.functional.binary_cross_entropy_with_logits(
        candidate.detach(), target.float()
    )
    assert torch.allclose(total, expected_task)
