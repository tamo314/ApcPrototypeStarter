"""One real-simulator path for T32-R: executed transitions -> event -> acquisition -> bank.

Checks mechanics only (record per env step incl. terminal, event gating, duplicate
suppression, candidate/router files that reload).  Task success is not asserted.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_transitions_event_acquisition_bank(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_t32r1_transitions
    import run_t32r3_acquire
    from apc_maniskill.experience_loop import Bank, read_jsonl
    from apc_maniskill.experience_loop import OptionSelector

    run = tmp_path / "r1"
    run_t32r1_transitions.main(["--out", str(run), "--episode", "true_place:3014",
                                "--max-steps", "800"])
    rows = read_jsonl(run / "transitions.jsonl")
    summary = json.loads((run / "summary.json").read_text())
    assert len(rows) == summary["episodes"][0]["steps"] == 800
    assert rows[-1]["truncated"] and rows[-1]["control_step"] == 799
    assert summary["consistency"] == {"records_equal_env_steps": True, "all_terminal_recorded": True}
    rejected = [r for r in rows if r["override_reason_code"] == 2]
    assert rejected and all(r["target_updated"] is False or r["ik_success"] is False for r in rejected)
    assert all(len(r["router_x"]) == 12 and len(r["candidate_x"]) > 12 for r in rows)
    events = sorted((run / "events").glob("*.json"))
    assert events, "the 3014 stall should raise an online detector event before step 800"

    registry = tmp_path / "registry.json"
    empty = run_t32r3_acquire.main(["--out", str(tmp_path / "a0"), "--registry", str(registry)])
    assert empty["result"] == "no_event_no_learning" and not registry.exists()

    common = ["--event", str(events[0]), "--replay", str(run), "--registry", str(registry),
              "--options", "18:1", "--horizon", "30", "--max-decisions", "1", "--workers", "1"]
    learned = run_t32r3_acquire.main(["--out", str(tmp_path / "a1"), *common])
    entry = learned["events"][0]
    assert entry["status"] == "accepted" and entry["prefix_replay_matches_history"]
    assert learned["result"] in ("learned", "no_candidate")
    if learned["result"] == "learned":
        bank_dir = Path(learned["banks"]["bank_A_full"]["dir"])
        manifest = json.loads((bank_dir / "bank_manifest.json").read_text())
        bank = Bank.from_roles(bank_dir, manifest["roles"])
        assert manifest["parent_bank_hash"] != manifest["bank_hash"] == bank.manifest()["bank_hash"]
        OptionSelector(bank.candidate_A)
        again = run_t32r3_acquire.main(["--out", str(tmp_path / "a2"), *common])
        assert again["events"][0]["status"] == "duplicate_skipped"
        assert again["result"] == "no_event_no_learning"
