"""Focused contract checks for the completed artifact-only REC-004AS audit."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_initialization_geometry_audit import (
    REC004AS_DECISION,
    REC004AS_TASK_ID,
    classify_basin_at_500,
)

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/phase_b_restart/rec004as/run_001"


def test_rec004as_basin_classification_is_fixed() -> None:
    assert (
        classify_basin_at_500({"correct_key_margin": 0.1, "top1_key": 2, "correct_key": 2})
        == "TRUE_BASIN_ENTRY"
    )
    assert (
        classify_basin_at_500({"correct_key_margin": -0.1, "top1_key": 1, "correct_key": 2})
        == "FALSE_BASIN_ENTRY"
    )
    assert (
        classify_basin_at_500({"correct_key_margin": 0.0, "top1_key": 1, "correct_key": 2})
        == "UNRESOLVED_AT_500"
    )


def test_rec004as_completed_artifact_boundaries_and_schema() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    side_effects = json.loads((RUN / "side_effect_audit.json").read_text(encoding="utf-8"))
    geometry = json.loads((RUN / "step0_routing_geometry.json").read_text(encoding="utf-8"))
    gradients = json.loads((RUN / "early_gradient_interaction.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == REC004AS_TASK_ID
    assert summary["status"] == "PASS"
    assert summary["decision"] == REC004AS_DECISION
    assert summary["n_initializations"] == 5
    assert summary["n_routing_cells_per_initialization"] == 40
    assert summary["optimizer_updates"] == 0
    assert summary["source_checkpoints_unchanged"] is True
    assert side_effects["candidate_selected"] is None
    assert side_effects["child_bundle"] is None
    assert side_effects["bundle_write"] is False
    assert side_effects["sealed_data_access"] == 0
    assert len(geometry["I01"]["baseline"]["cells"]) == 40
    assert "correct_key_rank" in geometry["I02"]["warm_start"]["cells"][0]
    assert "10:4" in gradients["I02"]["baseline"]
    assert "gradient_contribution_by_parameter_family" in gradients["I02"]["baseline"]["10:4"]
