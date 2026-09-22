from __future__ import annotations

import torch

from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.data import CNPRecord
from apc.cnp.repair import (
    CorrectedConditionalSelectPrimitive,
    _collate_records,
    clone_corrected_local_candidate,
)
from apc.cnp.repair_evidence import (
    fixed_panel_trace,
    initial_output_parity,
    parameter_accounting,
    preflight_panel_evidence,
)
from apc.cnp.repair_parity import verify_checkpoint, write_probe


def _record(index: int, *, target: bool) -> CNPRecord:
    return CNPRecord(
        state=SetState(
            values=torch.full((1, 1, 8), index / 100, dtype=torch.float32),
            valid=torch.tensor([[True]]),
            item_ids=torch.tensor([[index]], dtype=torch.int64),
        ),
        arguments=SelectArguments(
            query=torch.full((1, 8), 0.25, dtype=torch.float32),
            threshold=torch.tensor([0.5], dtype=torch.float32),
        ),
        target=torch.tensor([[target]]),
        role="repair_evidence_fixture",
        condition_key="domain=fixture|query=00",
        schedule_index=index,
    )


def _records() -> list[CNPRecord]:
    return [_record(index, target=bool(index % 2)) for index in range(32)]


def test_preflight_trace_and_initial_parity_are_cpu_testable() -> None:
    records = _records()
    evidence = preflight_panel_evidence({"train": records}, expected_shadow_sets=128)
    assert evidence["status"] == "PASS"

    torch.manual_seed(4)
    parent = CorrectedConditionalSelectPrimitive(0)
    candidate = clone_corrected_local_candidate(parent)
    device = torch.device("cpu")
    parity = initial_output_parity(parent, candidate, records=records, device=device)
    assert parity["status"] == "PASS"
    assert parameter_accounting(candidate)["update_eligible_parameters"] == 1024

    trace = fixed_panel_trace(
        candidate,
        parent,
        new_records=records,
        replay_records=records,
        device=device,
        retention_weight=1.0,
        step=0,
    )
    assert trace["new_bce"] >= 0.0
    assert trace["keep_mse"] == 0.0
    assert trace["parent_correct_replay_logit_mse"] == 0.0
    assert trace["parent_incorrect_replay_logit_mse"] == 0.0


def test_fresh_checkpoint_parity_is_independent_of_the_live_candidate(tmp_path) -> None:
    records = _records()
    torch.manual_seed(5)
    parent = CorrectedConditionalSelectPrimitive(0)
    candidate = clone_corrected_local_candidate(parent)
    checkpoint = tmp_path / "candidate.pt"
    probe = tmp_path / "probe.pt"
    torch.save({"model": candidate.state_dict()}, checkpoint)
    state, arguments, _ = _collate_records(records, torch.device("cpu"))
    write_probe(
        candidate,
        batches={"new_train": (state, arguments), "replay": (state, arguments)},
        path=probe,
        device=torch.device("cpu"),
    )
    result = verify_checkpoint(checkpoint, probe, device=torch.device("cpu"))
    assert result["status"] == "PASS"
    assert result["selection_masks_match"] is True
