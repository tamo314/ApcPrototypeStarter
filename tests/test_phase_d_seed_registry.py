"""Regression coverage for D-005's no-model static seed registry audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apc.evaluation.phase_d_seed_registry import (
    REGISTRY_PATH,
    SeedRegistryAuditError,
    audit_phase_d_seed_registry,
)


def test_phase_d_seed_registry_matches_all_registered_sources_and_provenance() -> None:
    report = audit_phase_d_seed_registry()

    assert report["status"] == "PASS"
    assert report["candidate_model_seeds"] == [40, 41, 42, 43, 44]
    assert set(report["candidate_model_seeds"]).isdisjoint(report["forbidden_model_seeds"])
    assert set(report["candidate_model_seeds"]).isdisjoint(report["historical_bundle_model_seeds"])
    assert set(report["candidate_model_seeds"]).isdisjoint(report["historical_data_seeds"])


def test_phase_d_seed_registry_fails_closed_for_a_candidate_collision(tmp_path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["candidate_cohort"]["model_seeds"] = [30, 31, 32, 33, 34]
    bad_registry = tmp_path / "registry.json"
    bad_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(SeedRegistryAuditError, match="collides"):
        audit_phase_d_seed_registry(bad_registry)


def test_phase_d_seed_registry_cli_is_runnable_from_a_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/verify_phase_d_seed_registry.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["candidate_model_seeds"] == [40, 41, 42, 43, 44]
