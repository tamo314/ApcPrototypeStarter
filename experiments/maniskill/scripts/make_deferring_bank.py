"""T48: parent-deferring extension of an acquired bank.

Copies the bank's roles and adds ``parent_router`` (the parent bank's router): the
refit router then only decides whether a candidate acts; every other routing
decision is the parent's, so without candidate proposals the bank behaves exactly
like its parent.

  python scripts/make_deferring_bank.py --bank <acquired bank> --parent dist_autonomous_bundle_v1 \
     --out runs/t48-deferring-20260926-a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bank_matrix import load_bank  # noqa: E402
from run_t32r3_acquire import write_bank  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    bank, parent = load_bank(args.bank), load_bank(args.parent)
    roles = dict(base=bank.base, router=bank.router, transit=bank.transit, place=bank.place,
                 **bank.candidate_roles(), parent_router=parent.router)
    m = write_bank(args.out, {k: v for k, v in roles.items() if v is not None}, bank.manifest(),
                   dict(design="extend_parent_deferring", parent_router_from=parent.manifest()["bank_hash"]))
    print(m["bank_hash"])
    return m


if __name__ == "__main__":
    main()
