"""Tests for NRQ-003 prerequisite audit and model adequacy stop gate."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.nrq003_prerequisite_audit import (
    audit_checkpoint_provenance,
    run_nrq003_prerequisite_audit,
)


def test_checkpoint_provenance_audit() -> None:
    results = audit_checkpoint_provenance(seeds=(0, 1, 2, 3, 4))
    assert len(results) == 5

    # Check seed 0
    s0 = results[0]
    assert s0.seed == 0
    assert s0.core_token_emb_shape == [44, 192]
    assert not s0.provenance_intact

    # Check seeds 1..4 have shape mismatch [36, 192] vs 44
    for res in results[1:]:
        assert res.core_token_emb_shape == [36, 192]
        assert not res.core_compatible_with_current_vocab
        assert not res.provenance_intact
        assert "Vocabulary size mismatch" in res.notes


def test_nrq003_report_record_matches_expected_decision(tmp_path: Path) -> None:
    out_file = tmp_path / "NRQ003_REVIEW_RECORD.json"
    report = run_nrq003_prerequisite_audit(output_path=out_file)

    assert report.task_id == "NRQ-003"
    assert report.prerequisite_status == "FAIL"
    assert report.decision == "BLOCKED_BY_MODEL_ADEQUACY"
    assert report.interpret_depth3_search is False
    assert out_file.is_file()

    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_data["decision"] == "BLOCKED_BY_MODEL_ADEQUACY"
    assert saved_data["interpret_depth3_search"] is False
    assert saved_data["depth2_positive_controls_seed0"]["overall_passed"] is False
    assert saved_data["depth2_positive_controls_seed0"]["mean_oracle_exact_match"] < 0.20
