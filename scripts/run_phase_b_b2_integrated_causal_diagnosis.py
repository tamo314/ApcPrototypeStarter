"""Run B-C005D2-006: integrated causal diagnosis + next-repair decision gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.integrated_causal_diagnosis import (
    IntegratedCausalDiagnosisConfig,
    run_integrated_causal_diagnosis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B-C005D2-006 integrated causal diagnosis and decision gate"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_integrated_causal_diagnosis.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_b2_second_diagnostic")
    )
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = IntegratedCausalDiagnosisConfig(
        d2001_summary_path=Path(data["d2001_summary_path"]),
        d2002_summary_path=Path(data["d2002_summary_path"]),
        d2003_summary_path=Path(data["d2003_summary_path"]),
        d2004_summary_path=Path(data["d2004_summary_path"]),
        d2005_summary_path=Path(data["d2005_summary_path"]),
        no_failure_threshold=data["no_failure_threshold"],
        output_dir=args.output_dir,
    )
    report = run_integrated_causal_diagnosis(config)
    print(json.dumps(report["source_representativeness_verdict_d2001"], indent=2))
    for row in report["findings_table"]:
        print(f"{row['mechanism']:<28} verdict={row['verdict']:<40} confidence={row['confidence']}")
    print(json.dumps(report["recommendation"], indent=2))
    print("STOP: B-C006 and Task Inference remain blocked pending explicit user instruction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
