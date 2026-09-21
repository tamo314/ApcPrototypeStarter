"""Fixed CNP-004 sequential-adaptation data and model helpers.

The public helpers in this module deliberately stop short of experiment execution:
they construct the registered data and training batches so the runner can preserve
paired inputs across LOCAL and baselines without consulting evaluation panels.
"""

from __future__ import annotations

import json
import platform
import resource
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from apc.cnp.artifacts import source_hash
from apc.cnp.baselines import LearnedMetricBaseline
from apc.cnp.confirmation import _checkpoint_path
from apc.cnp.data import (
    ADAPTATION_BLOCKS,
    KNOWN_LENGTHS,
    SOURCE_THRESHOLDS,
    CNPRecord,
    DataRole,
    canonical_json_hash,
    make_adaptation_record,
    make_record,
    make_world,
)
from apc.cnp.development import (
    EVAL_BATCH_SETS,
    MetricsAccumulator,
    _code_hashes,
    _collate,
    _git_commit,
    _source_records,
    _write_json,
)
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.seed_registry import audit_cnp_seed_registry
from apc.cnp.training import masked_bce_loss

ADAPT_TRAIN_EXEMPLARS = 8
TRANSFER_EXEMPLARS = 16
SETS_PER_CONDITION = 64
SMALL_ARM_SETS_PER_CONDITION = 16


def _length_for_example(example_index: int) -> int:
    return KNOWN_LENGTHS[example_index % len(KNOWN_LENGTHS)]


def adaptation_records(
    *,
    role: DataRole,
    block: str,
    exemplar_count: int,
    sets_per_condition: int,
) -> Iterator[CNPRecord]:
    """Yield one fixed CNP-004 panel; callers never choose examples from labels."""

    if block not in ADAPTATION_BLOCKS:
        raise ValueError(f"unsupported CNP adaptation block: {block}")
    if role not in {"adapt_train", "shadow", "transfer_eval"}:
        raise ValueError(f"unsupported CNP adaptation role: {role}")
    world = make_world()
    for exemplar_index in range(exemplar_count):
        for threshold in SOURCE_THRESHOLDS:
            for example_index in range(sets_per_condition):
                yield make_adaptation_record(
                    role=role,
                    block=block,
                    exemplar_index=exemplar_index,
                    length=_length_for_example(example_index),
                    threshold=threshold,
                    example_index=example_index,
                    world=world,
                )


def paired_adaptation_training_records(block: str) -> tuple[list[CNPRecord], list[CNPRecord]]:
    """Return fixed 64- and prefix-sharing 16-teacher arms for one new condition block."""

    full = list(
        adaptation_records(
            role="adapt_train",
            block=block,
            exemplar_count=ADAPT_TRAIN_EXEMPLARS,
            sets_per_condition=SETS_PER_CONDITION,
        )
    )
    small = list(
        adaptation_records(
            role="adapt_train",
            block=block,
            exemplar_count=ADAPT_TRAIN_EXEMPLARS,
            sets_per_condition=SMALL_ARM_SETS_PER_CONDITION,
        )
    )
    full_by_digest = {record.digest(): record for record in full}
    if not all(record.digest() in full_by_digest for record in small):
        raise RuntimeError("CNP-004 small arm must be a prefix subset of the full arm")
    return full, small


def source_replay_records() -> list[CNPRecord]:
    """Select the fixed, source-training-derived CNP-004 replay buffer once."""

    records: list[CNPRecord] = []
    for step in range(48):
        records.extend(_source_records(step, EVAL_BATCH_SETS))
    if len(records) != 1536:
        raise RuntimeError("CNP-004 source replay must contain exactly 1,536 initial records")
    return sorted(records, key=CNPRecord.digest)


def source_shadow_records() -> list[CNPRecord]:
    """Generate a role-disjoint initial-domain shadow panel for retention checks."""

    world = make_world()
    records: list[CNPRecord] = []
    for exemplar_index in range(ADAPT_TRAIN_EXEMPLARS):
        for threshold in SOURCE_THRESHOLDS:
            for example_index in range(SETS_PER_CONDITION):
                records.append(
                    make_record(
                        role="shadow",
                        condition_key=f"source_shadow_{exemplar_index:02d}",
                        length=_length_for_example(example_index),
                        threshold=threshold,
                        example_index=example_index,
                        world=world,
                    )
                )
    return records


def batched(
    records: list[CNPRecord], batch_size: int = EVAL_BATCH_SETS
) -> Iterator[list[CNPRecord]]:
    """Yield fixed-order batches without shuffling or replacement."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def evaluate_panel(
    model: nn.Module, records: list[CNPRecord], *, device: torch.device
) -> dict[str, float | int | None]:
    """Evaluate a fixed CNP-004 panel without exposing labels to the model."""

    accumulator = MetricsAccumulator()
    model.eval()
    with torch.inference_mode():
        for record_batch in batched(records):
            batch = _collate(record_batch, device)
            result = model(batch.state, batch.arguments)
            accumulator.add(result.selected, batch.target, batch.state.valid)
    return asdict(accumulator.result())


def quality_floor(metrics: dict[str, float | int | None]) -> bool:
    """Apply the registered CNP-004 balanced-accuracy and F1 floor."""

    balanced = metrics.get("balanced_accuracy")
    f1 = metrics.get("mean_set_f1")
    return (
        isinstance(balanced, float)
        and balanced >= 0.95
        and isinstance(f1, float)
        and f1 >= 0.90
    )


def train_fixed_candidate(
    model: nn.Module,
    *,
    new_records: list[CNPRecord],
    replay_records: list[CNPRecord],
    device: torch.device,
    steps: int = 256,
    replay_enabled: bool = True,
) -> list[dict[str, float | int]]:
    """Apply the registered fixed CNP-004 update schedule without checkpoint selection."""

    if not new_records:
        raise ValueError("CNP-004 candidate requires new training records")
    if replay_enabled and not replay_records:
        raise ValueError("CNP-004 replay candidate requires replay records")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
        weight_decay=0.0,
    )
    milestones: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        if replay_enabled:
            new_batch = [
                new_records[(step * 16 + offset) % len(new_records)] for offset in range(16)
            ]
            replay_batch = [
                replay_records[(step * 16 + offset) % len(replay_records)] for offset in range(16)
            ]
            records = [*new_batch, *replay_batch]
        else:
            records = [new_records[(step * 32 + offset) % len(new_records)] for offset in range(32)]
        batch = _collate(records, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch.state, batch.arguments)
        loss = masked_bce_loss(result.logits, batch.target, batch.state.valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        completed = step + 1
        if completed in {16, 64, 256}:
            milestones.append({"step": completed, "loss": float(loss.detach().cpu())})
    return milestones


def load_fixed_parent(
    source_run: Path, method: str, model_seed: int, *, device: torch.device
) -> nn.Module:
    """Load only the registered final CNP-003 parent checkpoint for CNP-004."""

    if method == "CONDITIONAL_MLP":
        model: nn.Module = ConditionalSelectPrimitive(0)
    elif method == "LEARNED_METRIC":
        model = LearnedMetricBaseline()
    else:
        raise ValueError(f"unsupported CNP-004 parent method: {method}")
    path = _checkpoint_path(source_run, method, model_seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_step = 4000
    if not isinstance(checkpoint, dict) or checkpoint.get("step") != expected_step:
        raise ValueError(f"invalid fixed CNP-003 parent checkpoint: {path}")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"invalid CNP-003 parent state: {path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_adaptation_preflight(
    config_path: Path, source_run: Path, output_root: Path, run_id: str | None = None
) -> Path:
    """Measure all fixed CNP-004 parent shadows before any candidate is constructed."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: CNP-004 requires one CUDA device")
    source_run = source_run.resolve()
    source_manifest_path = source_run / "run_manifest.json"
    source_report_path = source_run / "report.json"
    if not source_manifest_path.is_file() or not source_report_path.is_file():
        raise FileNotFoundError("CNP-004 source run is incomplete")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("task") != "CNP-003" or source_manifest.get("status") != "COMPLETE":
        raise ValueError("CNP-004 requires a completed CNP-003 source run")
    config_hash = canonical_json_hash(config)
    if source_manifest.get("config_hash") != config_hash:
        raise ValueError("CNP-004 source configuration hash mismatch")
    identifier = run_id or f"cnp004_preflight_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = output_root / identifier
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite CNP-004 preflight directory: {run_directory}")
    run_directory.mkdir(parents=True)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    seeds = config["seeds"]["confirmation_model"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("CNP-004 requires the fixed five confirmation seeds")
    run_manifest: dict[str, Any] = {
        "program": "cnp_v1",
        "task": "CNP-004-PREFLIGHT",
        "status": "RUNNING",
        "run_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "code_hashes": _code_hashes(config_path.parents[2]),
        "git_commit": _git_commit(config_path.parents[2]),
        "seed_audit": audit_cnp_seed_registry(config_path),
        "source_run": str(source_run),
        "source_hashes": {
            str(source_manifest_path): source_hash(source_manifest_path),
            str(source_report_path): source_hash(source_report_path),
        },
        "legacy_sealed_access": 0,
        "new_training_steps": 0,
        "new_optimizer_steps": 0,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    _write_json(run_directory / "run_manifest.json", run_manifest)
    try:
        rows: list[dict[str, Any]] = []
        for model_seed in seeds:
            for method in ("CONDITIONAL_MLP", "LEARNED_METRIC"):
                model = load_fixed_parent(source_run, method, model_seed, device=device)
                for block in ADAPTATION_BLOCKS:
                    shadow = list(
                        adaptation_records(
                            role="shadow",
                            block=block,
                            exemplar_count=ADAPT_TRAIN_EXEMPLARS,
                            sets_per_condition=SETS_PER_CONDITION,
                        )
                    )
                    metrics = evaluate_panel(model, shadow, device=device)
                    rows.append(
                        {
                            "model_seed": model_seed,
                            "method": method,
                            "block": block,
                            "shadow_metrics": metrics,
                            "parent_reuse": "ADAPTATION_NOT_NEEDED"
                            if quality_floor(metrics)
                            else "ADAPTATION_REQUIRED",
                        }
                    )
                del model
                torch.cuda.empty_cache()
        report = {
            "task": "CNP-004-PREFLIGHT",
            "run_id": identifier,
            "legacy_sealed_access": 0,
            "new_training_steps": 0,
            "new_optimizer_steps": 0,
            "rows": rows,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
        }
        _write_json(run_directory / "report.json", report)
        (run_directory / "report.md").write_text(
            "# CNP-004 parent shadow preflight\n\n"
            "Fixed CNP-003 parents were measured on CNP-004 shadow panels before any update.\n",
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
