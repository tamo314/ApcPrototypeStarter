"""Run the artifact-only REC-004AS CD-DPCA initialization/data interaction audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_initialization_geometry_audit import (
    MirrorCDDPCAInitializationGeometryAuditConfig,
    run_cd_dpca_initialization_geometry_audit,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="REC-004AS archived CD-DPCA geometry audit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004as/run_001",
    )
    args = parser.parse_args()
    summary = run_cd_dpca_initialization_geometry_audit(
        MirrorCDDPCAInitializationGeometryAuditConfig(output_dir=args.output_dir)
    )
    print(f"REC-004AS status: {summary['status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Optimizer updates: {summary['optimizer_updates']}")


if __name__ == "__main__":
    main()
