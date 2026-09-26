"""One real-simulator path for T34: stream item -> online event -> acquisition -> adoption -> S row.

Mechanics only (tiny search budget); task success and adoption outcome are not asserted.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_stream_event_update_and_s_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_t34_stream

    log = run_t34_stream.main([
        "--out", str(tmp_path / "stream"), "--stream", "true_place:3014", "--max-steps", "800",
        "--eval", "true_place:3014", "--replay-set", "true_place:3014", "--workers", "2",
        "--acq-args", "--options 18:1 --horizon 30 --max-decisions 1"])
    item = log["items"][0]
    assert item["steps"] == 800 and not item["success"]
    assert item["events"], "3014 stalls before step 800 and must raise an event"
    ev = item["events"][0]
    assert ev["decision"] in ("adopted", "rejected_update", "no_candidate")
    assert log["s_matrix"][0]["bank_version"] == 0
    assert len(log["s_matrix"]) == 1 + (ev["decision"] == "adopted")
    summary = json.loads((tmp_path / "stream" / "stream_summary.json").read_text())
    assert summary["adopted_updates"] == (ev["decision"] == "adopted")
