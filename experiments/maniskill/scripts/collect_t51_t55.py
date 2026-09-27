"""T55: collect T51-T54 results, paired comparisons and the cost ledger.

  python scripts/collect_t51_t55.py --out docs/T51_T55_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_t34_t44 import final_table  # noqa: E402

RUNS = Path("runs")
LEDGER = [("t52-consistency-20260927-a", "matrix"),
          ("t53-stream-v2-20260927-a-failed-budget-not-loaded", "stream"),
          ("t53-stream-v2-20260927-b", "stream"),
          ("t54-final-20260927-a", "matrix")]


def sign_test(gain: int, loss: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ledger, total = [], 0
    for run, kind in LEDGER:
        path = RUNS / run
        f = path / ("summary.json" if kind == "matrix" else "stream_summary.json")
        if not f.exists():
            ledger.append(dict(run=run, status="missing"))
            continue
        d = json.loads(f.read_text())
        s = d["env_steps"] if kind == "matrix" else d.get("env_steps_items", 0) + d.get("env_steps_s_matrix", 0)
        ledger.append(dict(run=run, env_steps=s))
        total += s
    stream = json.loads((RUNS / "t53-stream-v2-20260927-b" / "stream_summary.json").read_text())
    out = dict(note="Collected by scripts/collect_t51_t55.py; success = 20 consecutive steps.",
               ledger=ledger, env_steps_total=total,
               t51_budget=json.loads(Path("docs/T51_BUDGET.json").read_text()),
               t53_stream=dict(
                   events=[dict(t=i["t"], item=i["item"], decision=e.get("decision"), candidate=e.get("candidate"),
                                history=e.get("history"), adoption=e.get("adoption"))
                           for i in stream["items"] for e in i["events"]],
                   s_matrix=[dict(version=s["bank_version"], total=sum(s["row"].values()), row=s["row"])
                             for s in stream["s_matrix"]],
                   adopted=stream["adopted_updates"], rejected=stream["rejected_updates"],
                   missed=stream["missed_failures"], wall_seconds=stream["wall_seconds"]))
    if (RUNS / "t54-final-20260927-a" / "summary.json").exists():
        t = final_table("t54-final-20260927-a")
        for row in t.values():
            row["sign_test_p_vs_initial"] = sign_test(len(row["gained_vs_initial"]), len(row["lost_vs_initial"]))
        out["t54_final"] = t
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", total)
    for bank, row in out.get("t54_final", {}).items():
        print(bank, {k: row[k] for k in ("successes", "n", "wilson95", "gained_vs_initial", "lost_vs_initial",
                                         "candidate_invoked", "candidate_invoked_success", "run_errors",
                                         "sign_test_p_vs_initial")})


if __name__ == "__main__":
    main()
