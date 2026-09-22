"""Explicit runners for the separately authorised schedule and retention diagnostics.

Importing this module has no data or CUDA side effects.  Its run functions are
only reached through named CLI modes and retain every candidate without selecting
or promoting one.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from apc.cnp.adaptation import _base_weights_hash, load_fixed_parent
from apc.cnp.data import CNPRecord, canonical_json_hash
from apc.cnp.repair import (
    CorrectedConditionalSelectPrimitive,
    _collate_records,
    clone_corrected_local_candidate,
    set_mean_masked_bce_loss,
)
from apc.cnp.repair_diagnostic import _validate_inputs, r001_corrected_replay_records
from apc.cnp.repair_evaluation import (
    CellKey,
    CellMetrics,
    build_cell_metrics,
    evidence_from_masks,
    gate_report,
    paired_bootstrap_delta,
    serialize_cells,
)
from apc.cnp.repair_evidence import (
    fixed_panel_trace,
    initial_output_parity,
    parameter_accounting,
    preflight_panel_evidence,
)
from apc.cnp.repair_parity import write_probe
from apc.cnp.repair_protocol import (
    ROOTS,
    ProtocolKind,
    audit_protocol_disjoint,
    make_protocol_panel,
    protocol_manifest,
)
from apc.cnp.repair_retention import (
    ParentLogitCache,
    combined_decision_repair_loss,
    combined_repair_loss,
)
from apc.cnp.repair_schedule import ScheduleArm, SideSchedule, build_side_schedule, schedule_audit

UPDATES = 256
NEW_BATCH = 16
REPLAY_BATCH = 16


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_config(config_path: Path, *, program: str, protocol: ProtocolKind) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("program") != program or config.get("schema_version") != 1:
        raise ValueError(f"expected registered {program} configuration")
    if config.get("roots") != {
        "data": ROOTS[protocol].data,
        "schedule": ROOTS[protocol].schedule,
        "bootstrap": ROOTS[protocol].bootstrap,
    }:
        raise ValueError("repair diagnostic roots differ from the registered values")
    if config.get("updates") != UPDATES or config.get("new_batch_sets") != NEW_BATCH:
        raise ValueError("repair diagnostic update or new batch budget differs from registration")
    if config.get("replay_batch_sets") != REPLAY_BATCH:
        raise ValueError("repair diagnostic replay batch budget differs from registration")
    if config.get("candidate_selection") != 0 or config.get("legacy_sealed_access") != 0:
        raise ValueError("repair diagnostics never select candidates or open legacy sealed data")
    seeds = config.get("parent_model_seeds")
    if seeds != [610200, 610201, 610202, 610203, 610204]:
        raise ValueError("repair diagnostic requires the fixed five CNP-003 parent seeds")
    return config


def _make_run_directory(output_root: Path, *, label: str, run_id: str | None) -> Path:
    identifier = run_id or f"{label}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    directory = output_root / identifier
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic evidence: {directory}")
    directory.mkdir(parents=True)
    return directory


def _corrected_parent(
    source_run: Path, seed: int, device: torch.device
) -> CorrectedConditionalSelectPrimitive:
    legacy = load_fixed_parent(source_run, "CONDITIONAL_MLP", seed, device=torch.device("cpu"))
    parent = CorrectedConditionalSelectPrimitive(0)
    parent.load_state_dict(legacy.state_dict(), strict=True)
    return parent.to(device).eval()


def _evaluate(
    model: torch.nn.Module, records: list[CNPRecord], *, device: torch.device
) -> dict[CellKey, CellMetrics]:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            batch_records = records[start : start + 32]
            state, arguments, target = _collate_records(batch_records, device)
            selected = model(state, arguments).selected
            for index, record in enumerate(batch_records):
                width = record.state.width
                rows.append(
                    (
                        CellKey.from_record(record),
                        evidence_from_masks(
                            input_id=record.input_digest(),
                            predicted=selected[index, :width],
                            target=target[index, :width],
                            valid=state.valid[index, :width],
                        ),
                    )
                )
    return build_cell_metrics(rows)


def _records_for_schedule(
    records: list[CNPRecord], schedule: SideSchedule
) -> list[list[CNPRecord]]:
    record_by_id = {record.input_digest(): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("training records need unique input IDs")
    return [[record_by_id[entry.input_id] for entry in batch] for batch in schedule.batches]


def _build_parent_caches(
    parent: CorrectedConditionalSelectPrimitive,
    replay_batches: list[list[CNPRecord]],
    *,
    parent_hash: str,
    device: torch.device,
) -> list[ParentLogitCache]:
    caches: list[ParentLogitCache] = []
    parent.eval()
    with torch.inference_mode():
        for records in replay_batches:
            state, arguments, _ = _collate_records(records, device)
            logits = parent(state, arguments).logits.detach().cpu().to(torch.float32)
            caches.append(
                ParentLogitCache(
                    parent_hash=parent_hash,
                    architecture_signature=parent.architecture_signature,
                    input_ids=tuple(record.input_digest() for record in records),
                    logits=logits,
                )
            )
    return caches


def _resource_status(started: float, limits: dict[str, Any]) -> dict[str, int | float]:
    torch.cuda.synchronize()
    try:
        import resource

        process_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - the diagnostic runner executes under Linux/WSL
        process_bytes = 0
    return {
        "wall_seconds": time.perf_counter() - started,
        "gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_ram_bytes": process_bytes,
        "wall_limit": int(limits["wall_seconds"]),
        "gpu_limit": int(limits["gpu_bytes"]),
        "process_ram_limit": int(limits["process_ram_bytes"]),
    }


def _enforce_resource_limits(started: float, limits: dict[str, Any]) -> None:
    status = _resource_status(started, limits)
    if (
        status["wall_seconds"] > status["wall_limit"]
        or status["gpu_bytes"] > status["gpu_limit"]
        or status["process_ram_bytes"] > status["process_ram_limit"]
    ):
        raise RuntimeError(f"RESOURCE_STOP: {status}")


def _train(
    candidate: CorrectedConditionalSelectPrimitive,
    *,
    parent: CorrectedConditionalSelectPrimitive,
    new_batches: list[list[CNPRecord]],
    replay_batches: list[list[CNPRecord]],
    fixed_new: list[CNPRecord],
    fixed_replay: list[CNPRecord],
    caches: list[ParentLogitCache] | None,
    parent_hash: str,
    retention_weight: float,
    objective: str,
    decision_margin: float,
    started: float,
    limits: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float | int]]:
    if len(new_batches) != UPDATES or len(replay_batches) != UPDATES:
        raise ValueError("registered training requires exactly 256 batches per side")
    if caches is not None and len(caches) != UPDATES:
        raise ValueError("parent cache must contain one entry for every replay batch")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in candidate.parameters() if parameter.requires_grad),
        lr=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    milestones = [
        fixed_panel_trace(
            candidate,
            parent,
            new_records=fixed_new,
            replay_records=fixed_replay,
            device=device,
            retention_weight=retention_weight,
            step=0,
            decision_weight=retention_weight if objective == "decision" else 0.0,
            decision_margin=decision_margin,
        )
    ]
    candidate.train()
    for step, (new_records, replay_records) in enumerate(
        zip(new_batches, replay_batches, strict=True)
    ):
        records = [*new_records, *replay_records]
        state, arguments, target = _collate_records(records, device)
        new_width = max(record.state.width for record in new_records)
        replay_width = max(record.state.width for record in replay_records)
        optimizer.zero_grad(set_to_none=True)
        result = candidate(state, arguments)
        new_logits = result.logits[:NEW_BATCH, :new_width]
        replay_logits = result.logits[NEW_BATCH:, :replay_width]
        new_target = target[:NEW_BATCH, :new_width]
        replay_target = target[NEW_BATCH:, :replay_width]
        new_valid = state.valid[:NEW_BATCH, :new_width]
        replay_valid = state.valid[NEW_BATCH:, :replay_width]
        if caches is None:
            loss = 0.5 * set_mean_masked_bce_loss(new_logits, new_target, new_valid)
            loss = loss + 0.5 * set_mean_masked_bce_loss(replay_logits, replay_target, replay_valid)
            values: dict[str, float] = {"task": float(loss.detach().cpu()), "retention": 0.0}
        else:
            cache = caches[step]
            input_ids = tuple(record.input_digest() for record in replay_records)
            cache.validate(
                parent_hash=parent_hash,
                architecture_signature=candidate.architecture_signature,
                input_ids=input_ids,
            )
            parent_logits = cache.logits.to(device)[:, :replay_width]
            if objective == "retention":
                loss, values = combined_repair_loss(
                    new_logits=new_logits,
                    new_target=new_target,
                    new_valid=new_valid,
                    replay_logits=replay_logits,
                    replay_target=replay_target,
                    replay_valid=replay_valid,
                    parent_replay_logits=parent_logits,
                    retention_weight=retention_weight,
                )
            elif objective == "decision":
                loss, values = combined_decision_repair_loss(
                    new_logits=new_logits,
                    new_target=new_target,
                    new_valid=new_valid,
                    replay_logits=replay_logits,
                    replay_target=replay_target,
                    replay_valid=replay_valid,
                    parent_replay_logits=parent_logits,
                    decision_weight=retention_weight,
                    decision_margin=decision_margin,
                )
            else:
                raise ValueError(f"unknown registered repair objective: {objective}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0)
        optimizer.step()
        _enforce_resource_limits(started, limits)
        if step + 1 in {16, 64, UPDATES}:
            trace = fixed_panel_trace(
                candidate,
                parent,
                new_records=fixed_new,
                replay_records=fixed_replay,
                device=device,
                retention_weight=retention_weight,
                step=step + 1,
                decision_weight=retention_weight if objective == "decision" else 0.0,
                decision_margin=decision_margin,
            )
            milestones.append({"batch_loss": float(loss.detach().cpu()), **values, **trace})
    return milestones


def _common_panels(
    protocol: ProtocolKind,
) -> tuple[
    list[CNPRecord], list[CNPRecord], list[CNPRecord], list[CNPRecord], dict[str, object]
]:
    new_train = make_protocol_panel(protocol=protocol, panel="new_train")
    new_shadow = make_protocol_panel(protocol=protocol, panel="new_shadow")
    old_shadow = make_protocol_panel(protocol=protocol, panel="old_shadow")
    replay = r001_corrected_replay_records()
    if len(new_train) != 1536 or len(replay) != 1536:
        raise RuntimeError("registered diagnostic training panels must contain 1,536 records")
    audit = audit_protocol_disjoint(
        {
            "new_train": new_train,
            "new_shadow": new_shadow,
            "old_shadow": old_shadow,
            "replay": replay,
        }
    )
    return new_train, new_shadow, old_shadow, replay, audit


def _fresh_process_parity(
    *,
    candidate: CorrectedConditionalSelectPrimitive,
    checkpoint_path: Path,
    new_records: list[CNPRecord],
    replay_records: list[CNPRecord],
    directory: Path,
    device: torch.device,
) -> dict[str, object]:
    """Verify a final candidate from a new process on fixed train/replay probes."""

    probe_path = directory / f"{checkpoint_path.stem}_parity_probe.pt"
    output_path = directory / f"{checkpoint_path.stem}_fresh_parity.json"
    write_probe(
        candidate,
        batches={
            "new_train": _collate_records(new_records[:32], device)[:2],
            "replay": _collate_records(replay_records[:32], device)[:2],
        },
        path=probe_path,
        device=device,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apc.cnp.repair_parity",
            "--checkpoint",
            str(checkpoint_path),
            "--probe",
            str(probe_path),
            "--output",
            str(output_path),
        ],
        cwd=directory.parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    if not output_path.is_file():
        return {
            "status": "FAIL",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["returncode"] = completed.returncode
    return result


def _run(
    *,
    config_path: Path,
    source_run: Path,
    output_root: Path,
    run_id: str | None,
    program: str,
    protocol: ProtocolKind,
    arms: list[tuple[str, float, ScheduleArm]],
    objective: str = "task",
    decision_margin: float = 0.0,
    status_prefix: str = "R",
    prerequisite_schedule_report: Path | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: repair diagnostics require one CUDA device")
    config = _load_config(config_path, program=program, protocol=protocol)
    source_config = config_path.parents[2] / config["source_config_path"]
    source_config_payload, source_manifest = _validate_inputs(source_config, source_run)
    if prerequisite_schedule_report is not None:
        if not prerequisite_schedule_report.is_file():
            raise FileNotFoundError("R-CNP-001R requires the recorded R-CNP-001S report")
        schedule_report = json.loads(prerequisite_schedule_report.read_text(encoding="utf-8"))
        if schedule_report.get("program") != "cnp_repair_schedule_v1":
            raise ValueError("retention prerequisite is not an R-CNP-001S report")
        if schedule_report.get("status") not in {
            "S_DIAGNOSTIC_VIABILITY_PASS",
            "S_DIAGNOSTIC_FAIL_STOP",
        }:
            raise ValueError("R-CNP-001S report does not contain a completed diagnostic status")
    directory = _make_run_directory(output_root, label=program, run_id=run_id)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    manifest: dict[str, Any] = {
        "program": program,
        "status": "RUNNING",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": canonical_json_hash(config),
        "source_config_hash": canonical_json_hash(source_config_payload),
        "source_manifest_task": source_manifest["task"],
        "source_run": str(source_run.resolve()),
        "schedule_prerequisite_report": (
            None
            if prerequisite_schedule_report is None
            else str(prerequisite_schedule_report.resolve())
        ),
        "candidate_selection": 0,
        "bundle_promotion": "NOT_AUTHORIZED",
        "legacy_sealed_access": 0,
        "platform": platform.platform(),
        "device": str(device),
    }
    _write_json(directory / "run_manifest.json", manifest)
    try:
        new_train, new_shadow, old_shadow, replay, split_audit = _common_panels(protocol)
        preflight = preflight_panel_evidence(
            {
                "new_train": new_train,
                "new_shadow": new_shadow,
                "old_shadow": old_shadow,
                "replay": replay,
            },
            expected_shadow_sets=int(config["shadow_sets_per_cell"]),
        )
        schedules = {
            arm_name: {
                "new": build_side_schedule(
                    new_train, arm=schedule_arm, side="new", root=ROOTS[protocol].schedule
                ),
                "replay": build_side_schedule(
                    replay, arm=schedule_arm, side="replay", root=ROOTS[protocol].schedule
                ),
            }
            for arm_name, _, schedule_arm in arms
        }
        _write_json(
            directory / "data_manifest.json",
            {
                "status": "PASS",
                "split_audit": split_audit,
                "preflight": preflight,
                "protocol_manifests": {
                    "new_train": protocol_manifest(new_train),
                    "new_shadow": protocol_manifest(new_shadow),
                    "old_shadow": protocol_manifest(old_shadow),
                    "replay": protocol_manifest(replay),
                },
                "source_lineage": (
                    "CNP-003 source replay, immutable digest-selected R-CNP-001 buffer"
                ),
            },
        )
        for arm_name, sides in schedules.items():
            for side, schedule in sides.items():
                (directory / f"schedule_{arm_name}_{side}.jsonl").write_text(
                    schedule.json_lines(), encoding="utf-8"
                )
        _write_json(
            directory / "schedule_audit.json",
            {
                arm: {side: schedule_audit(value) for side, value in sides.items()}
                for arm, sides in schedules.items()
            },
        )
        rows: list[dict[str, object]] = []
        for seed in config["parent_model_seeds"]:
            parent = _corrected_parent(source_run, seed, device)
            parent_hash = _base_weights_hash(parent)
            parent_new = _evaluate(parent, new_shadow, device=device)
            parent_old = _evaluate(parent, old_shadow, device=device)
            for arm_name, lambda_value, _schedule_arm in arms:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                candidate = clone_corrected_local_candidate(parent)
                candidate.to(device)
                before = _base_weights_hash(candidate)
                new_batches = _records_for_schedule(new_train, schedules[arm_name]["new"])
                replay_batches = _records_for_schedule(replay, schedules[arm_name]["replay"])
                initial_parity = {
                    "new_train": initial_output_parity(
                        parent, candidate, records=new_train, device=device
                    ),
                    "replay": initial_output_parity(
                        parent, candidate, records=replay, device=device
                    ),
                }
                accounting = parameter_accounting(candidate)
                caches = (
                    _build_parent_caches(
                        parent, replay_batches, parent_hash=parent_hash, device=device
                    )
                    if objective in {"retention", "decision"}
                    else None
                )
                training = _train(
                    candidate,
                    parent=parent,
                    new_batches=new_batches,
                    replay_batches=replay_batches,
                    fixed_new=new_train,
                    fixed_replay=replay,
                    caches=caches,
                    parent_hash=parent_hash,
                    retention_weight=lambda_value,
                    objective=objective,
                    decision_margin=decision_margin,
                    started=started,
                    limits=config["limits"],
                    device=device,
                )
                after = _base_weights_hash(candidate)
                new_cells = _evaluate(candidate, new_shadow, device=device)
                old_cells = _evaluate(candidate, old_shadow, device=device)
                report = gate_report(
                    parent_old=parent_old,
                    candidate_new=new_cells,
                    candidate_old=old_cells,
                    invariant_status=(
                        before == after
                        and all(value["status"] == "PASS" for value in initial_parity.values())
                        and accounting["update_eligible_parameters"] == 1024
                    ),
                )
                candidate_path = directory / f"candidate_{arm_name.lower()}_seed_{seed}.pt"
                torch.save({"model": candidate.state_dict(), "gate": report}, candidate_path)
                fresh_parity = _fresh_process_parity(
                    candidate=candidate,
                    checkpoint_path=candidate_path,
                    new_records=new_train,
                    replay_records=replay,
                    directory=directory,
                    device=device,
                )
                bootstrap = {
                    cell.text(): paired_bootstrap_delta(
                        parent_old[cell],
                        old_cells[cell],
                        root=ROOTS[protocol].bootstrap + seed,
                    )
                    for cell in sorted(parent_old)
                }
                previous_invariants = report["invariants"]
                previous_gate = report["candidate_gate"]
                if not isinstance(previous_invariants, dict) or not isinstance(previous_gate, str):
                    raise RuntimeError("repair gate report has an invalid invariant payload")
                updated_invariants = {
                    "status": "PASS"
                    if previous_invariants.get("status") == "PASS"
                    and fresh_parity.get("status") == "PASS"
                    else "INVALID_RUN_STOP",
                    "base_hash_match": before == after,
                    "initial_output_parity": initial_parity,
                    "trainable_parameter_count": accounting["update_eligible_parameters"],
                    "fresh_process_parity": fresh_parity,
                }
                report["invariants"] = updated_invariants
                report["candidate_gate"] = (
                    "PASS"
                    if previous_gate == "PASS" and updated_invariants["status"] == "PASS"
                    else "FAIL"
                )
                rows.append(
                    {
                        "parent_seed": seed,
                        "arm": arm_name,
                        "parent_base_hash": parent_hash,
                        "candidate_base_hash_before": before,
                        "candidate_base_hash_after": after,
                        "training": training,
                        "gate": report,
                        "parameter_accounting": accounting,
                        "old_bootstrap": bootstrap,
                        "parent_new_cells": serialize_cells(parent_new),
                        "parent_old_cells": serialize_cells(parent_old),
                        "candidate_new_cells": serialize_cells(new_cells),
                        "candidate_old_cells": serialize_cells(old_cells),
                        "parent_cache_hashes": []
                        if caches is None
                        else [cache.cache_hash for cache in caches],
                        "candidate_path": str(candidate_path),
                    }
                )
                del candidate
                torch.cuda.empty_cache()
            del parent
            torch.cuda.empty_cache()
        candidate_name = config["candidate_arm"]
        candidate_rows = [row for row in rows if row["arm"] == candidate_name]
        viability = all(row["gate"]["candidate_gate"] == "PASS" for row in candidate_rows)  # type: ignore[index]
        report = {
            "program": program,
            "status": (
                f"{status_prefix}_DIAGNOSTIC_VIABILITY_PASS"
                if viability
                else f"{status_prefix}_DIAGNOSTIC_FAIL_STOP"
            ),
            "rows": rows,
            "resource": _resource_status(started, config["limits"]),
            "candidate_selection": 0,
            "bundle_promotion": "NOT_AUTHORIZED",
        }
        _write_json(directory / "report.json", report)
        manifest["status"] = "COMPLETE"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(directory / "run_manifest.json", manifest)
        return directory
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["resource"] = _resource_status(started, config["limits"])
        _write_json(directory / "run_manifest.json", manifest)
        raise


def run_repair_schedule(
    config_path: Path, source_run: Path, output_root: Path, run_id: str | None = None
) -> Path:
    """Execute only the named R-CNP-001S A/B/C diagnostic when authorised."""

    return _run(
        config_path=config_path,
        source_run=source_run,
        output_root=output_root,
        run_id=run_id,
        program="cnp_repair_schedule_v1",
        protocol="schedule",
        arms=[
            (ScheduleArm.ORDERED.value, 0.0, ScheduleArm.ORDERED),
            (ScheduleArm.DISPERSED_MATCHED.value, 0.0, ScheduleArm.DISPERSED_MATCHED),
            (ScheduleArm.DISPERSED_BALANCED.value, 0.0, ScheduleArm.DISPERSED_BALANCED),
        ],
        status_prefix="S",
    )


def run_repair_retention(
    config_path: Path,
    source_run: Path,
    output_root: Path,
    run_id: str | None = None,
    schedule_report: Path | None = None,
) -> Path:
    """Execute only the named R-CNP-001R lambda=0/1 diagnostic when authorised."""

    if schedule_report is None:
        raise ValueError("R-CNP-001R requires an explicit completed R-CNP-001S report path")

    return _run(
        config_path=config_path,
        source_run=source_run,
        output_root=output_root,
        run_id=run_id,
        program="cnp_repair_retention_v1",
        protocol="retention",
        prerequisite_schedule_report=schedule_report,
        arms=[
            ("LAMBDA_0", 0.0, ScheduleArm.DISPERSED_BALANCED),
            ("LAMBDA_1", 1.0, ScheduleArm.DISPERSED_BALANCED),
        ],
        objective="retention",
    )


def run_repair_decision(
    config_path: Path, source_run: Path, output_root: Path, run_id: str | None = None
) -> Path:
    """Execute the registered parent-correct decision-retention diagnostic only."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("decision_margin") != 0.0:
        raise ValueError("R-CNP-001M fixes the decision margin at zero")
    return _run(
        config_path=config_path,
        source_run=source_run,
        output_root=output_root,
        run_id=run_id,
        program="cnp_repair_decision_v1",
        protocol="decision",
        arms=[
            ("TASK_ONLY", 0.0, ScheduleArm.DISPERSED_BALANCED),
            ("PARENT_CORRECT_DECISION", 1.0, ScheduleArm.DISPERSED_BALANCED),
        ],
        objective="decision",
        decision_margin=0.0,
        status_prefix="M",
    )
