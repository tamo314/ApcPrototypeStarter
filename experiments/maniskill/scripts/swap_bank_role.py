"""T61: new bank version with some roles replaced (e.g. another router or candidate).

  python scripts/swap_bank_role.py --bank runs/t48-deferring-20260926-a \
     --role candidate_A=runs/t61-cand-succ/candidate_A.pt --out runs/t61-oldR-succC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t32r3_acquire import write_bank  # noqa: E402

from apc_maniskill.experience_loop import Bank  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--role", action="append", required=True, help="role=path")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    bank = Bank.load(args.bank)
    roles = {k: v for k, v in bank.files().items()}
    swapped = {}
    for spec in args.role:
        role, path = spec.split("=", 1)
        if role not in roles:
            raise SystemExit(f"bank has no role {role}")
        roles[role] = Path(path)
        swapped[role] = path
    notes = dict(design="role_swap", swapped=swapped)
    if bank.candidate_budget is not None:
        notes["candidate_budget"] = bank.candidate_budget
    m = write_bank(args.out, roles, bank.manifest(), notes)
    print(m["bank_hash"])
    return m


if __name__ == "__main__":
    main()
