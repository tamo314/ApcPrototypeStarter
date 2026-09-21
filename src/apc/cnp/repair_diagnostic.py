"""Fixed-parent R-CNP-001 diagnostic for the isolated CNP adaptation repair.

This module is intentionally separate from the historical CNP-004 runner.  It
creates a new development-only split, stores every candidate, and never selects
or promotes one.  Its results cannot establish a fully repaired H-CNP2 claim.
"""

from __future__ import annotations

import json
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from apc.cnp.adaptation import (
    ADAPT_TRAIN_EXEMPLARS,
    SETS_PER_CONDITION,
    _base_weights_hash,
    _data_manifest,
    _length_for_example,
    load_fixed_parent,
)
from apc.cnp.artifacts import source_hash
from apc.cnp.data import (
    SOURCE_THRESHOLDS,
    CNPRecord,
    canonical_json_hash,
    make_adaptation_record,
    make_record,
    make_world,
)
from apc.cnp.development import (
    EVAL_BATCH_SETS,
    _code_hashes,
    _collate,
    _git_commit,
    _source_records,
    _write_json,
)
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.repair import (
    CorrectedConditionalSelectPrimitive,
    clone_corrected_local_candidate,
    evaluate_panel_by_cell,
    evaluate_shadow_gate,
    select_stratified_replay_records,
    set_mean_masked_bce_loss,
)
from apc.cnp.seed_registry import audit_cnp_seed_registry
from apc.cnp.training import clone_local_candidate, masked_bce_loss

R001_BLOCK = "q1+q2+"
R001_STEPS = 256
R001_REPLAY_RECORDS = 1536


def r001_new_records(role: str) -> list[CNPRecord]:
    """Construct the fixed development-only new-domain panel for one R-CNP-001 role."""

    if role not in {"repair_adapt_train", "repair_new_shadow"}:
        raise ValueError(f"unsupported R-CNP-001 new-domain role: {role}")
    world = make_world()
    return [
        make_adaptation_record(
            role=role,  # type: ignore[arg-type]
            block=R001_BLOCK,
            exemplar_index=exemplar_index,
            length=_length_for_example(example_index),
            threshold=threshold,
            example_index=example_index,
            world=world,
        )
        for exemplar_index in range(ADAPT_TRAIN_EXEMPLARS)
        for threshold in SOURCE_THRESHOLDS
        for example_index in range(SETS_PER_CONDITION)
    ]


def r001_old_shadow_records() -> list[CNPRecord]:
    """Construct a role-disjoint source-domain retention panel for R-CNP-001."""

    world = make_world()
    return [
        make_record(
            role="repair_old_shadow",
            condition_key=f"repair_source_shadow_{exemplar_index:02d}",
            length=_length_for_example(example_index),
            threshold=threshold,
            example_index=example_index,
            world=world,
        )
        for exemplar_index in range(ADAPT_TRAIN_EXEMPLARS)
        for threshold in SOURCE_THRESHOLDS
        for example_index in range(SETS_PER_CONDITION)
    ]


def r001_legacy_replay_records() -> list[CNPRecord]:
    """Reconstruct v1's first-1,536-record replay rule for the paired legacy control."""

    return [record for step in range(48) for record in _source_records(step, EVAL_BATCH_SETS)]


def r001_corrected_replay_records() -> list[CNPRecord]:
    """Select the corrected replay buffer from all fixed source-training records."""

    records = [record for step in range(4000) for record in _source_records(step, EVAL_BATCH_SETS)]
    return select_stratified_replay_records(records, buffer_size=R001_REPLAY_RECORDS)


def _train_candidate(
    model: nn.Module,
    *,
    new_records: list[CNPRecord],
    replay_records: list[CNPRecord],
    device: torch.device,
    corrected_loss: bool,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
        weight_decay=0.0,
    )
    milestones: list[dict[str, float | int]] = []
    model.train()
    for step in range(R001_STEPS):
        records = [
            *[new_records[(step * 16 + offset) % len(new_records)] for offset in range(16)],
            *[replay_records[(step * 16 + offset) % len(replay_records)] for offset in range(16)],
        ]
        batch = _collate(records, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch.state, batch.arguments)
        loss = (
            set_mean_masked_bce_loss(result.logits, batch.target, batch.state.valid)
            if corrected_loss
            else masked_bce_loss(result.logits, batch.target, batch.state.valid)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        completed = step + 1
        if completed in {16, 64, R001_STEPS}:
            milestones.append({"step": completed, "loss": float(loss.detach().cpu())})
    return milestones


def _validate_inputs(config_path: Path, source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = source_run / "run_manifest.json"
    report_path = source_run / "report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("R-CNP-001 source run is incomplete")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("task") != "CNP-003" or source_manifest.get("status") != "COMPLETE":
        raise ValueError("R-CNP-001 requires the completed fixed CNP-003 source run")
    if source_manifest.get("config_hash") != canonical_json_hash(config):
        raise ValueError("R-CNP-001 source configuration hash mismatch")
    seeds = config.get("seeds", {}).get("confirmation_model")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("R-CNP-001 requires the fixed five CNP-003 parent seeds")
    return config, source_manifest


def run_r001(
    config_path: Path, source_run: Path, output_root: Path, run_id: str | None = None
) -> Path:
    """Run the pre-registered fixed-parent diagnostic once without candidate selection."""

    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: R-CNP-001 requires one CUDA device")
    source_run = source_run.resolve()
    config, source_manifest = _validate_inputs(config_path, source_run)
    identifier = run_id or f"rcnp001_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = output_root / identifier
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite R-CNP-001 directory: {run_directory}")
    run_directory.mkdir(parents=True)
    (run_directory / "candidates").mkdir()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    new_train = r001_new_records("repair_adapt_train")
    new_shadow = r001_new_records("repair_new_shadow")
    old_shadow = r001_old_shadow_records()
    legacy_replay = r001_legacy_replay_records()
    corrected_replay = r001_corrected_replay_records()
    if not all(len(panel) == R001_REPLAY_RECORDS for panel in (new_train, new_shadow, old_shadow)):
        raise RuntimeError("R-CNP-001 panels must each contain 1,536 records")
    shared_manifest = _data_manifest(
        {
            "new_train": new_train,
            "new_shadow": new_shadow,
            "old_shadow": old_shadow,
        }
    )
    legacy_manifest = _data_manifest(
        {
            "new_train": new_train,
            "new_shadow": new_shadow,
            "old_shadow": old_shadow,
            "legacy_replay": legacy_replay,
        }
    )
    corrected_manifest = _data_manifest(
        {
            "new_train": new_train,
            "new_shadow": new_shadow,
            "old_shadow": old_shadow,
            "corrected_replay": corrected_replay,
        }
    )
    replay_overlap = len({record.digest() for record in legacy_replay} & {
        record.digest() for record in corrected_replay
    })
    data_manifest = {
        "status": "PASS",
        "shared_panels": shared_manifest,
        "legacy_method_panels": legacy_manifest,
        "repaired_method_panels": corrected_manifest,
        "allowed_cross_method_replay_overlap": replay_overlap,
    }
    _write_json(run_directory / "data_manifest.json", data_manifest)
    source_manifest_path = source_run / "run_manifest.json"
    source_report_path = source_run / "report.json"
    run_manifest: dict[str, Any] = {
        "program": "cnp_adaptation_repair_v2",
        "task": "R-CNP-001",
        "status": "RUNNING",
        "run_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": canonical_json_hash(config),
        "code_hashes": _code_hashes(config_path.parents[2]),
        "git_commit": _git_commit(config_path.parents[2]),
        "source_run": str(source_run),
        "source_hashes": {
            str(source_manifest_path): source_hash(source_manifest_path),
            str(source_report_path): source_hash(source_report_path),
        },
        "parent_seed_audit": audit_cnp_seed_registry(config_path),
        "parent_model_seeds": config["seeds"]["confirmation_model"],
        "methods": ["LEGACY_LOCAL", "REPAIRED_LOCAL"],
        "block": R001_BLOCK,
        "new_records_per_panel": len(new_train),
        "replay_records_per_method": R001_REPLAY_RECORDS,
        "updates_per_candidate": R001_STEPS,
        "new_training_steps": 2 * len(config["seeds"]["confirmation_model"]) * R001_STEPS,
        "candidate_selection": 0,
        "bundle_promotion": "NOT_AUTHORIZED",
        "legacy_sealed_access": 0,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    _write_json(run_directory / "run_manifest.json", run_manifest)
    try:
        rows: list[dict[str, Any]] = []
        for model_seed in config["seeds"]["confirmation_model"]:
            torch.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)
            legacy_parent = load_fixed_parent(
                source_run, "CONDITIONAL_MLP", model_seed, device=device
            )
            if not isinstance(legacy_parent, ConditionalSelectPrimitive):
                raise RuntimeError("R-CNP-001 parent must be a conditional primitive")
            repaired_parent = CorrectedConditionalSelectPrimitive(0).to(device)
            repaired_parent.load_state_dict(legacy_parent.state_dict(), strict=True)
            parent_new = evaluate_panel_by_cell(legacy_parent, new_shadow, device=device)
            parent_old = evaluate_panel_by_cell(legacy_parent, old_shadow, device=device)
            torch.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)
            legacy_candidate = clone_local_candidate(legacy_parent)
            torch.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)
            repaired_candidate = clone_corrected_local_candidate(repaired_parent)
            methods: tuple[tuple[str, nn.Module, list[CNPRecord], bool], ...] = (
                ("LEGACY_LOCAL", legacy_candidate, legacy_replay, False),
                (
                    "REPAIRED_LOCAL",
                    repaired_candidate,
                    corrected_replay,
                    True,
                ),
            )
            for method, candidate, replay, corrected_loss in methods:
                if not isinstance(candidate, ConditionalSelectPrimitive):
                    raise RuntimeError("R-CNP-001 local candidate type mismatch")
                base_before = _base_weights_hash(candidate)
                candidate.to(device)
                training = _train_candidate(
                    candidate,
                    new_records=new_train,
                    replay_records=replay,
                    device=device,
                    corrected_loss=corrected_loss,
                )
                candidate_new = evaluate_panel_by_cell(candidate, new_shadow, device=device)
                candidate_old = evaluate_panel_by_cell(candidate, old_shadow, device=device)
                base_after = _base_weights_hash(candidate)
                if base_before != base_after:
                    raise RuntimeError("R-CNP-001 LOCAL candidate modified frozen base weights")
                path = run_directory / "candidates" / f"{method.lower()}_seed_{model_seed}.pt"
                torch.save({"model": candidate.state_dict(), "selected": False}, path)
                rows.append(
                    {
                        "model_seed": model_seed,
                        "method": method,
                        "parent_new_shadow": parent_new,
                        "parent_old_shadow": parent_old,
                        "candidate_new_shadow": candidate_new,
                        "candidate_old_shadow": candidate_old,
                        "new_shadow_gate": evaluate_shadow_gate(candidate_new, parent_new),
                        "old_shadow_gate": evaluate_shadow_gate(candidate_old, parent_old),
                        "base_weights_hash_before": base_before,
                        "base_weights_hash_after": base_after,
                        "training": training,
                        "candidate_path": str(path),
                    }
                )
                del candidate
                torch.cuda.empty_cache()
            del repaired_parent
            del legacy_parent
            torch.cuda.empty_cache()
        report = {
            "task": "R-CNP-001",
            "run_id": identifier,
            "scope": "FIXED_PARENT_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "hypothesis_status": "NOT_A_CONFIRMATION_OF_H_CNP2",
            "candidate_selection": 0,
            "bundle_promotion": "NOT_AUTHORIZED",
            "rows": rows,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "legacy_sealed_access": 0,
        }
        _write_json(run_directory / "report.json", report)
        with (run_directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        (run_directory / "report.md").write_text(
            "# R-CNP-001 fixed-parent development diagnostic\n\n"
            "This diagnostic stores paired legacy and repaired candidates without selection.\n",
            encoding="utf-8",
        )
        run_manifest["status"] = "COMPLETE"
        run_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_directory / "run_manifest.json", run_manifest)
        return run_directory
    except Exception as error:
        run_manifest["status"] = "FAILED"
        run_manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(run_directory / "run_manifest.json", run_manifest)
        raise
