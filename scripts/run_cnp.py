"""CNP command-line entry point.

Static audit, planning, confirmation, and authorized CNP-004 stages are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running a tracked script directly is supported in addition to an editable
# install.  This only locates this repository's own source tree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> int:
    from apc.cnp.data import canonical_json_hash
    from apc.cnp.seed_registry import DEFAULT_REGISTRY_PATH, audit_cnp_seed_registry

    def config(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="CNP v1 static audit and future runner entry point"
    )
    parser.add_argument(
        "mode",
        choices=(
            "audit",
            "dry-run",
            "develop",
            "confirm",
            "correct",
            "confirm-v2",
            "adapt",
            "adapt-block1",
            "repair-r001",
            "repair-schedule",
            "repair-retention",
            "repair-alignment",
            "repair-decision",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--schedule-report", type=Path)
    arguments = parser.parse_args()
    if arguments.config is None:
        defaults = {
            "repair-schedule": REPOSITORY_ROOT / "configs/cnp/repair_schedule_v1.json",
            "repair-retention": REPOSITORY_ROOT / "configs/cnp/repair_retention_v1.json",
            "repair-alignment": REPOSITORY_ROOT / "configs/cnp/repair_alignment_v1.json",
            "repair-decision": REPOSITORY_ROOT / "configs/cnp/repair_decision_v1.json",
        }
        arguments.config = defaults.get(arguments.mode, DEFAULT_REGISTRY_PATH)
    loaded_config = config(arguments.config)
    if arguments.mode == "audit":
        print(json.dumps(audit_cnp_seed_registry(arguments.config), sort_keys=True, indent=2))
        return 0
    if arguments.mode == "dry-run":
        audit = audit_cnp_seed_registry(arguments.config)
        print(
            json.dumps(
                {
                    "status": "PLAN_READY_NO_MODEL_OR_DATA_ACCESS",
                    "config_hash": canonical_json_hash(loaded_config),
                    "seed_audit": audit,
                    "enabled_modes": ["audit", "dry-run", "develop", "confirm"],
                    "blocked_modes": ["adapt", "report"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if arguments.mode == "develop":
        from apc.cnp.development import run_development

        output = run_development(
            arguments.config,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v1/develop",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "confirm":
        from apc.cnp.confirmation import run_confirmation

        output = run_confirmation(
            arguments.config,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v1/confirm",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "correct":
        if arguments.source_run is None:
            parser.error("correct requires --source-run")
        from apc.cnp.confirmation import run_confirmation_correction

        output = run_confirmation_correction(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v1/correction",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "confirm-v2":
        if arguments.source_run is None:
            parser.error("confirm-v2 requires --source-run")
        from apc.cnp.confirmation import run_confirmation_v2

        output = run_confirmation_v2(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v2/confirm",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "adapt":
        if arguments.source_run is None:
            parser.error("adapt requires --source-run")
        from apc.cnp.adaptation import run_adaptation_preflight

        output = run_adaptation_preflight(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v1/adapt",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "adapt-block1":
        if arguments.source_run is None:
            parser.error("adapt-block1 requires --source-run")
        from apc.cnp.adaptation import run_adaptation_block1

        output = run_adaptation_block1(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_v1/adapt",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "repair-r001":
        if arguments.source_run is None:
            parser.error("repair-r001 requires --source-run")
        from apc.cnp.repair_diagnostic import run_r001

        output = run_r001(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_repair/r001",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "repair-schedule":
        if arguments.source_run is None:
            parser.error("repair-schedule requires --source-run")
        from apc.cnp.repair_followup import run_repair_schedule

        output = run_repair_schedule(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_repair/r001s",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "repair-retention":
        if arguments.source_run is None:
            parser.error("repair-retention requires --source-run")
        if arguments.schedule_report is None:
            parser.error("repair-retention requires --schedule-report from R-CNP-001S")
        from apc.cnp.repair_followup import run_repair_retention

        output = run_repair_retention(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_repair/r001r",
            arguments.run_id,
            arguments.schedule_report,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "repair-alignment":
        if arguments.source_run is None:
            parser.error("repair-alignment requires --source-run")
        if arguments.run_id is None:
            parser.error("repair-alignment requires an explicit --run-id")
        from apc.cnp.repair_alignment import run_repair_alignment

        output = run_repair_alignment(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_repair/r001d",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    if arguments.mode == "repair-decision":
        if arguments.source_run is None:
            parser.error("repair-decision requires --source-run")
        if arguments.run_id is None:
            parser.error("repair-decision requires an explicit --run-id")
        from apc.cnp.repair_followup import run_repair_decision

        output = run_repair_decision(
            arguments.config,
            arguments.source_run,
            arguments.output_root or REPOSITORY_ROOT / "runs/cnp_repair/r001m",
            arguments.run_id,
        )
        print(json.dumps({"status": "COMPLETE", "run_directory": str(output)}, sort_keys=True))
        return 0
    raise SystemExit(
        f"CNP mode {arguments.mode!r} is not enabled by CNP-003; "
        "run its separately authorized stage first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
