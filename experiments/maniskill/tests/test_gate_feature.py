"""One real-simulator path for T47: candidate decision -> counterfactual branches -> gate -> gated bank.

A synthetic bank whose router always proposes candidate_A (an option candidate
that only holds) keeps the episode short; only the mechanics are asserted.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.maniskill
@pytest.mark.skipif(os.getenv("APC_RUN_MANISKILL_TEST") != "1", reason="Explicit real-simulator test")
def test_counterfactual_gate_and_veto(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_bank_matrix
    import run_t47_counterfactual
    from run_t32r3_acquire import write_bank

    from apc_maniskill.experience_loop import (CART, HOLD, MODULES_EXT, ROUTER_FEATURE_NAMES, ROUTER_V2_EXTRA,
                                               save_option_candidate, save_router)

    dim = len(ROUTER_FEATURE_NAMES) + len(ROUTER_V2_EXTRA)
    router = CART.fit(np.zeros((2, dim), np.float32), np.array([4, 4]), np.ones(2), 5, max_depth=1, min_leaf=1)
    save_router(tmp_path / "router.pt", router, feature_schema="router_v2", class_names=MODULES_EXT)
    hold = CART.fit(np.zeros((2, 1), np.float32), np.array([0, 0]), np.ones(2), 1, max_depth=1, min_leaf=1)
    save_option_candidate(tmp_path / "cand.pt", hold, np.zeros(1, np.float32), np.ones(1, np.float32),
                          [(HOLD, 1)], source={})
    bundle = ROOT / "dist_autonomous_bundle_v1"
    write_bank(tmp_path / "syn", dict(base=bundle / "base_selector.pt", router=tmp_path / "router.pt",
                                      transit=bundle / "transit_candidate.pt",
                                      place=bundle / "place_candidate.pt", candidate_A=tmp_path / "cand.pt"),
               dict(bank_hash="synthetic"), dict(design="test"))
    run_bank_matrix.main(["--out", str(tmp_path / "m"), "--bank", f"syn={tmp_path / 'syn'}",
                          "--episode", "true_place:3112", "--max-steps", "60", "--workers", "1"])
    cell = json.loads((tmp_path / "m" / "summary.json").read_text())["cells"][0]
    assert cell["run_error"] is None and cell["module_counts"].get("candidate_A", 0) > 0

    log = run_t47_counterfactual.main([
        "--out", str(tmp_path / "cf"), "--run", str(tmp_path / "m"), "--bank-name", "syn",
        "--bank", str(tmp_path / "syn"), "--max-decisions", "1", "--max-steps", "60", "--workers", "2"])
    assert log["decision_points"] == 1 and log["run_errors"] == 0 and log["labels"] == 1
    assert log["replay_faithful"] == 1, "continue branch must reproduce the logged episode"
    roles = json.loads((tmp_path / "cf" / "bank_gated" / "bank_manifest.json").read_text())["roles"]
    assert "gate" in roles and "candidate_A" in roles

    run_bank_matrix.main(["--out", str(tmp_path / "m2"), "--bank", f"gated={tmp_path / 'cf' / 'bank_gated'}",
                          "--bank", f"veto={tmp_path / 'syn'}@veto_candidates",
                          "--episode", "true_place:3112", "--max-steps", "60", "--workers", "2"])
    cells = {c["bank"]: c for c in json.loads((tmp_path / "m2" / "summary.json").read_text())["cells"]}
    assert not any(k.startswith("candidate") for k in cells["veto"]["module_counts"])
    rows = [json.loads(line) for line in
            (tmp_path / "m2" / "cells" / "gated__true_place_3112" / "transitions.jsonl").read_text().splitlines()]
    assert any(r["candidate_gate"] is not None for r in rows)
