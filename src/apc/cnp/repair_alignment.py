"""Zero-update diagnosis of CNP new/replay/retention gradient alignment."""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

from apc.cnp.adaptation import _base_weights_hash
from apc.cnp.data import CNPRecord, canonical_json_hash
from apc.cnp.repair import (
    CorrectedConditionalSelectPrimitive,
    _collate_records,
    clone_corrected_local_candidate,
    set_mean_masked_bce_loss,
)
from apc.cnp.repair_diagnostic import _validate_inputs, r001_corrected_replay_records
from apc.cnp.repair_evidence import (
    initial_output_parity,
    parameter_accounting,
    preflight_panel_evidence,
)
from apc.cnp.repair_followup import _corrected_parent, _enforce_resource_limits, _resource_status
from apc.cnp.repair_protocol import (
    ROOTS,
    audit_protocol_disjoint,
    make_protocol_panel,
    protocol_manifest,
)
from apc.cnp.repair_retention import retention_loss

GradientKind = Literal["new_bce", "replay_bce", "keep_mse"]


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not values:
        raise RuntimeError("alignment diagnostic found no adapter gradient")
    return torch.cat(values)


def _panel_gradient(
    candidate: CorrectedConditionalSelectPrimitive,
    parent: CorrectedConditionalSelectPrimitive,
    records: list[CNPRecord],
    *,
    kind: GradientKind,
    device: torch.device,
    batch_sets: int = 32,
) -> tuple[float, torch.Tensor]:
    """Return one full-panel loss and its adapter gradient without optimizer state."""

    if not records or len(records) % batch_sets:
        raise ValueError("alignment panels must be non-empty and batch divisible")
    candidate.train()
    parent.eval()
    candidate.zero_grad(set_to_none=True)
    values: list[float] = []
    batch_count = len(records) // batch_sets
    for start in range(0, len(records), batch_sets):
        state, arguments, target = _collate_records(records[start : start + batch_sets], device)
        candidate_logits = candidate(state, arguments).logits
        if kind == "new_bce" or kind == "replay_bce":
            loss = set_mean_masked_bce_loss(candidate_logits, target, state.valid)
        else:
            with torch.inference_mode():
                parent_logits = parent(state, arguments).logits
            loss = retention_loss(candidate_logits, parent_logits, state.valid)
        values.append(float(loss.detach().cpu()))
        (loss / batch_count).backward()
    return sum(values) / len(values), _gradient_vector(candidate).detach().cpu()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(left.norm() * right.norm())
    return None if denominator == 0.0 else float(torch.dot(left, right) / denominator)


def _virtual_first_adamw_step(
    candidate: CorrectedConditionalSelectPrimitive,
    gradient: torch.Tensor,
    *,
    learning_rate: float,
    epsilon: float,
) -> list[torch.Tensor]:
    """Apply an in-memory first AdamW step and return adapter tensors for rollback."""

    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    originals = [parameter.detach().clone() for parameter in parameters]
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            local = gradient[offset : offset + size].reshape_as(parameter).to(parameter.device)
            parameter.add_(-learning_rate * local / (local.abs() + epsilon))
            offset += size
    if offset != gradient.numel():
        raise RuntimeError("alignment gradient does not match trainable adapter parameters")
    return originals


def _restore_adapter(
    candidate: CorrectedConditionalSelectPrimitive, originals: list[torch.Tensor]
) -> None:
    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    if len(parameters) != len(originals):
        raise RuntimeError("alignment rollback parameter count changed")
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)


def gradient_alignment(
    candidate: CorrectedConditionalSelectPrimitive,
    parent: CorrectedConditionalSelectPrimitive,
    *,
    new_records: list[CNPRecord],
    replay_records: list[CNPRecord],
    device: torch.device,
    batch_sets: int = 32,
) -> dict[str, float | None]:
    """Measure all registered component gradients on paired fixed panels."""

    new_loss, new_gradient = _panel_gradient(
        candidate, parent, new_records, kind="new_bce", device=device, batch_sets=batch_sets
    )
    replay_loss, replay_gradient = _panel_gradient(
        candidate, parent, replay_records, kind="replay_bce", device=device, batch_sets=batch_sets
    )
    base_keep_loss, base_keep_gradient = _panel_gradient(
        candidate, parent, replay_records, kind="keep_mse", device=device, batch_sets=batch_sets
    )
    task_gradient = 0.5 * (new_gradient + replay_gradient)
    originals = _virtual_first_adamw_step(
        candidate, task_gradient, learning_rate=0.001, epsilon=1e-8
    )
    try:
        virtual_new_loss, virtual_new_gradient = _panel_gradient(
            candidate, parent, new_records, kind="new_bce", device=device, batch_sets=batch_sets
        )
        virtual_replay_loss, virtual_replay_gradient = _panel_gradient(
            candidate,
            parent,
            replay_records,
            kind="replay_bce",
            device=device,
            batch_sets=batch_sets,
        )
        virtual_keep_loss, virtual_keep_gradient = _panel_gradient(
            candidate, parent, replay_records, kind="keep_mse", device=device, batch_sets=batch_sets
        )
    finally:
        _restore_adapter(candidate, originals)
    virtual_task_gradient = 0.5 * (virtual_new_gradient + virtual_replay_gradient)
    return {
        "base_new_bce": new_loss,
        "base_replay_bce": replay_loss,
        "base_keep_mse": base_keep_loss,
        "base_new_gradient_norm": float(new_gradient.norm()),
        "base_replay_gradient_norm": float(replay_gradient.norm()),
        "base_keep_gradient_norm": float(base_keep_gradient.norm()),
        "base_task_gradient_norm": float(task_gradient.norm()),
        "base_task_keep_cosine": _cosine(task_gradient, base_keep_gradient),
        "virtual_new_bce": virtual_new_loss,
        "virtual_replay_bce": virtual_replay_loss,
        "virtual_keep_mse": virtual_keep_loss,
        "virtual_new_gradient_norm": float(virtual_new_gradient.norm()),
        "virtual_replay_gradient_norm": float(virtual_replay_gradient.norm()),
        "virtual_keep_gradient_norm": float(virtual_keep_gradient.norm()),
        "virtual_task_gradient_norm": float(virtual_task_gradient.norm()),
        "virtual_task_keep_cosine": _cosine(virtual_task_gradient, virtual_keep_gradient),
        "base_new_replay_cosine": _cosine(new_gradient, replay_gradient),
        "virtual_new_replay_cosine": _cosine(virtual_new_gradient, virtual_replay_gradient),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("program") != "cnp_repair_alignment_v1" or config.get("schema_version") != 1:
        raise ValueError("expected registered cnp_repair_alignment_v1 configuration")
    if config.get("roots") != ROOTS["alignment"].__dict__:
        raise ValueError("alignment roots differ from registration")
    if config.get("updates") != 0 or config.get("gradient_batch_sets") != 32:
        raise ValueError("alignment diagnostic must have zero updates and fixed 32-set gradients")
    if config.get("parent_model_seeds") != [610200, 610201, 610202, 610203, 610204]:
        raise ValueError("alignment diagnostic requires the fixed five CNP-003 parents")
    if config.get("candidate_selection") != 0 or config.get("legacy_sealed_access") != 0:
        raise ValueError("alignment diagnostic cannot select or access sealed data")
    return config


def run_repair_alignment(
    config_path: Path, source_run: Path, output_root: Path, run_id: str
) -> Path:
    """Run the separately registered, zero-update retention alignment diagnostic."""

    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: alignment diagnostic requires one CUDA device")
    config = _load_config(config_path)
    output = output_root / run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic evidence: {output}")
    source_config = config_path.parents[2] / config["source_config_path"]
    source_config_payload, source_manifest = _validate_inputs(source_config, source_run)
    output.mkdir(parents=True)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    manifest: dict[str, object] = {
        "program": config["program"],
        "status": "RUNNING",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": canonical_json_hash(config),
        "source_config_hash": canonical_json_hash(source_config_payload),
        "source_manifest_task": source_manifest["task"],
        "source_run": str(source_run.resolve()),
        "candidate_selection": 0,
        "bundle_promotion": "NOT_AUTHORIZED",
        "legacy_sealed_access": 0,
        "optimizer_updates": 0,
        "device": str(device),
        "platform": platform.platform(),
    }
    _write_json(output / "run_manifest.json", manifest)
    try:
        new_train = make_protocol_panel(protocol="alignment", panel="new_train")
        replay = r001_corrected_replay_records()
        split_audit = audit_protocol_disjoint({"new_train": new_train, "replay": replay})
        preflight = preflight_panel_evidence(
            {"new_train": new_train, "replay": replay}, expected_shadow_sets=128
        )
        _write_json(
            output / "data_manifest.json",
            {
                "status": "PASS",
                "split_audit": split_audit,
                "preflight": preflight,
                "protocol_manifests": {
                    "new_train": protocol_manifest(new_train),
                    "replay": protocol_manifest(replay),
                },
            },
        )
        rows: list[dict[str, object]] = []
        for seed in config["parent_model_seeds"]:
            if not isinstance(seed, int):
                raise RuntimeError("registered parent seed is not an integer")
            parent = _corrected_parent(source_run, seed, device)
            candidate = clone_corrected_local_candidate(parent).to(device)
            before = _base_weights_hash(candidate)
            parity = {
                "new_train": initial_output_parity(
                    parent, candidate, records=new_train, device=device
                ),
                "replay": initial_output_parity(parent, candidate, records=replay, device=device),
            }
            alignment = gradient_alignment(
                candidate,
                parent,
                new_records=new_train,
                replay_records=replay,
                device=device,
                batch_sets=int(config["gradient_batch_sets"]),
            )
            after = _base_weights_hash(candidate)
            if before != after:
                raise RuntimeError("zero-update alignment diagnostic altered frozen base weights")
            rows.append(
                {
                    "parent_seed": seed,
                    "parent_base_hash": _base_weights_hash(parent),
                    "candidate_base_hash_before": before,
                    "candidate_base_hash_after": after,
                    "initial_parity": parity,
                    "parameter_accounting": parameter_accounting(candidate),
                    "alignment": alignment,
                }
            )
            _enforce_resource_limits(started, config["limits"])
            del candidate
            del parent
            torch.cuda.empty_cache()
        threshold = float(config["cosine_conflict_threshold"])
        task_keep = [row["alignment"]["virtual_task_keep_cosine"] for row in rows]  # type: ignore[index]
        all_conflict = all(isinstance(value, float) and value <= threshold for value in task_keep)
        report = {
            "program": config["program"],
            "status": "ALIGNMENT_CONFLICT_SUPPORTED"
            if all_conflict
            else "ALIGNMENT_CONFLICT_NOT_ESTABLISHED_STOP",
            "criterion": {"task_keep_cosine_at_most": threshold, "required_seeds": 5},
            "rows": rows,
            "resource": _resource_status(started, config["limits"]),
            "candidate_selection": 0,
            "bundle_promotion": "NOT_AUTHORIZED",
            "optimizer_updates": 0,
            "legacy_sealed_access": 0,
        }
        _write_json(output / "report.json", report)
        manifest["status"] = "COMPLETE"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(output / "run_manifest.json", manifest)
        return output
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["resource"] = _resource_status(started, config["limits"])
        _write_json(output / "run_manifest.json", manifest)
        raise
