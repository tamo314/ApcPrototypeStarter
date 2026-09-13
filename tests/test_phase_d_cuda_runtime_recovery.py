"""CPU-safe coverage for the Phase D CUDA precondition probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_probe_fails_closed_without_cuda(tmp_path: Path) -> None:
    """The probe records a precondition failure without starting a cohort."""
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/phase_d_cuda_runtime_recovery.py",
            "--output-dir",
            str(tmp_path / "probe"),
            "--json",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["runtime_gate"] in {"PASS", "FAIL"}
    if not report["cuda_available"]:
        assert report["runtime_gate"] == "FAIL"
        assert report["failure"] == "cuda_unavailable"
    assert report["cohort_construction"] == "NOT_PERFORMED"
    assert report["model_data_or_sealed_access"] == 0
