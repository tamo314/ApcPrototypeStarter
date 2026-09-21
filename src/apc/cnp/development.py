"""Fixed CNP-002 development execution with explicit data and evidence boundaries."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from apc.cnp.artifacts import build_manifest, save_bundle, source_hash
from apc.cnp.baselines import LearnedMetricBaseline, RawDistanceBaseline, UnconditionedBaseline
from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.data import (
    INTERPOLATION_THRESHOLDS,
    KNOWN_LENGTHS,
    SOURCE_THRESHOLDS,
    CNPRecord,
    canonical_json_hash,
    derive_seed,
    make_record,
    make_world,
)
from apc.cnp.evaluation import SelectionMetrics
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.reference import reference_controls
from apc.cnp.seed_registry import audit_cnp_seed_registry
from apc.cnp.training import masked_bce_loss

DEVELOPMENT_QUERY_COUNT = 32
DEVELOPMENT_SETS_PER_QUERY = 128
EVAL_BATCH_SETS = 32


@dataclass(frozen=True)
class Batch:
    """A padded training or evaluation batch without metadata in the model path."""

    state: SetState
    arguments: SelectArguments
    target: torch.Tensor


@dataclass
class DataAudit:
    """Content-addressed input audit for one generated source and development panel."""

    digests: dict[str, set[str]] = field(default_factory=dict)
    records: dict[str, int] = field(default_factory=dict)
    valid_items: dict[str, int] = field(default_factory=dict)
    positive_items: dict[str, int] = field(default_factory=dict)
    empty_sets: dict[str, int] = field(default_factory=dict)
    cell_items: dict[str, int] = field(default_factory=dict)
    cell_positive: dict[str, int] = field(default_factory=dict)

    def add(self, record: CNPRecord) -> None:
        role = record.role
        self.digests.setdefault(role, set()).add(record.digest())
        self.records[role] = self.records.get(role, 0) + 1
        valid = int(record.state.valid.sum())
        positive = int(record.target.sum())
        self.valid_items[role] = self.valid_items.get(role, 0) + valid
        self.positive_items[role] = self.positive_items.get(role, 0) + positive
        self.empty_sets[role] = self.empty_sets.get(role, 0) + int(not record.target.any())
        if role == "dev_eval":
            parts = record.condition_key.split("_")
            if len(parts) != 2:
                raise ValueError(f"unexpected development condition key: {record.condition_key}")
            cell = (
                f"length={record.state.width}|"
                f"threshold={float(record.arguments.threshold[0]):.2f}"
            )
            self.cell_items[cell] = self.cell_items.get(cell, 0) + valid
            self.cell_positive[cell] = self.cell_positive.get(cell, 0) + positive

    def manifest(self) -> dict[str, Any]:
        duplicate_pairs: list[dict[str, Any]] = []
        roles = sorted(self.digests)
        for index, first in enumerate(roles):
            for second in roles[index + 1 :]:
                shared = self.digests[first] & self.digests[second]
                if shared:
                    duplicate_pairs.append(
                        {"first_role": first, "second_role": second, "shared_records": len(shared)}
                    )
        if duplicate_pairs:
            raise ValueError(f"CNP split overlap detected: {duplicate_pairs}")
        roles_manifest = {
            role: {
                "records": self.records.get(role, 0),
                "unique_records": len(self.digests.get(role, set())),
                "valid_items": self.valid_items.get(role, 0),
                "positive_items": self.positive_items.get(role, 0),
                "positive_rate": (
                    None
                    if self.valid_items.get(role, 0) == 0
                    else self.positive_items.get(role, 0) / self.valid_items[role]
                ),
                "empty_rate": (
                    None
                    if self.records.get(role, 0) == 0
                    else self.empty_sets.get(role, 0) / self.records[role]
                ),
            }
            for role in roles
        }
        cell_rates = {
            cell: self.cell_positive[cell] / self.cell_items[cell]
            for cell in sorted(self.cell_items)
        }
        return {
            "status": "PASS",
            "roles": roles_manifest,
            "cross_role_overlap": duplicate_pairs,
            "development_cell_positive_rates": cell_rates,
        }


@dataclass
class MetricsAccumulator:
    """Exact aggregate selection statistics without padding or element-level bootstrap claims."""

    true_positive: int = 0
    true_negative: int = 0
    positives: int = 0
    negatives: int = 0
    f1_sum: float = 0.0
    exact_sum: float = 0.0
    sets: int = 0
    valid_items: int = 0

    def add(self, predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        predicted = predicted & valid
        target = target & valid
        positives = target
        negatives = ~target & valid
        self.true_positive += int((predicted & positives).sum())
        self.true_negative += int((~predicted & negatives).sum())
        self.positives += int(positives.sum())
        self.negatives += int(negatives.sum())
        self.valid_items += int(valid.sum())
        intersection = (predicted & target).sum(dim=1).to(torch.float32)
        denominator = (predicted.sum(dim=1) + target.sum(dim=1)).to(torch.float32)
        f1 = torch.where(
            denominator == 0, torch.ones_like(denominator), 2 * intersection / denominator
        )
        self.f1_sum += float(f1.sum())
        self.exact_sum += float((predicted == target).all(dim=1).sum())
        self.sets += predicted.shape[0]

    def result(self) -> SelectionMetrics:
        balanced = None
        if self.positives and self.negatives:
            balanced = 0.5 * (
                self.true_positive / self.positives + self.true_negative / self.negatives
            )
        return SelectionMetrics(
            balanced_accuracy=balanced,
            mean_set_f1=self.f1_sum / self.sets,
            exact_match=self.exact_sum / self.sets,
            valid_items=self.valid_items,
        )


@dataclass
class _CausalControlAccumulator:
    """One intervention measured on its own reference-defined effectful support."""

    correct_hits: int = 0
    intervention_hits: int = 0
    effectful_items: int = 0
    effectful_sets: int = 0

    def add(
        self,
        target: torch.Tensor,
        valid: torch.Tensor,
        correct: torch.Tensor,
        intervention: torch.Tensor,
        reference_intervention: torch.Tensor,
    ) -> None:
        changed = valid & (target != reference_intervention)
        count = int(changed.sum())
        self.effectful_items += count
        self.effectful_sets += int(changed.any(dim=1).sum())
        if count == 0:
            return
        self.correct_hits += int((correct[changed] == target[changed]).sum())
        self.intervention_hits += int((intervention[changed] == target[changed]).sum())

    def report(self, *, minimum_effectful_sets: int) -> dict[str, Any]:
        if self.effectful_items == 0:
            return {
                "status": "INCONCLUSIVE",
                "reason": "no_reference_effectful_items",
                "effectful_items": 0,
                "effectful_sets": 0,
            }
        correct = self.correct_hits / self.effectful_items
        intervention = self.intervention_hits / self.effectful_items
        return {
            "status": (
                "PASS" if self.effectful_sets >= minimum_effectful_sets else "INCONCLUSIVE"
            ),
            "effectful_items": self.effectful_items,
            "effectful_sets": self.effectful_sets,
            "correct": correct,
            "intervention": intervention,
            "correct_minus_intervention": correct - intervention,
        }


@dataclass
class CausalAccumulator:
    """Keep Correct-vs-control evidence separate for every causal intervention."""

    controls: dict[str, _CausalControlAccumulator] = field(
        default_factory=lambda: {
            "wrong_family": _CausalControlAccumulator(),
            "wrong_argument": _CausalControlAccumulator(),
            "none": _CausalControlAccumulator(),
        }
    )

    def add(
        self,
        target: torch.Tensor,
        valid: torch.Tensor,
        correct: torch.Tensor,
        interventions: dict[str, torch.Tensor],
        reference_interventions: dict[str, torch.Tensor],
    ) -> None:
        expected = set(self.controls)
        if set(interventions) != expected or set(reference_interventions) != expected:
            raise ValueError(
                "causal interventions must provide wrong_family, wrong_argument, and none"
            )
        for name, accumulator in self.controls.items():
            accumulator.add(
                target,
                valid,
                correct,
                interventions[name],
                reference_interventions[name],
            )

    def merge(self, other: CausalAccumulator) -> None:
        for name, accumulator in self.controls.items():
            source = other.controls[name]
            accumulator.correct_hits += source.correct_hits
            accumulator.intervention_hits += source.intervention_hits
            accumulator.effectful_items += source.effectful_items
            accumulator.effectful_sets += source.effectful_sets

    def report(self, *, minimum_effectful_sets: int = 256) -> dict[str, Any]:
        controls = {
            name: accumulator.report(minimum_effectful_sets=minimum_effectful_sets)
            for name, accumulator in self.controls.items()
        }
        status = (
            "PASS"
            if all(item["status"] == "PASS" for item in controls.values())
            else "INCONCLUSIVE"
        )
        return {"status": status, "controls": controls}


def _source_condition(draw: int) -> tuple[str, float]:
    """Select one of the fixed 64-by-3 source conditions in a balanced cycle."""

    cycle_size = 64 * len(SOURCE_THRESHOLDS)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed("source_train", "condition_cycle"))
    order = torch.randperm(cycle_size, generator=generator)
    index = int(order[draw % cycle_size])
    query_index, threshold_index = divmod(index, len(SOURCE_THRESHOLDS))
    return f"source_{query_index:02d}", SOURCE_THRESHOLDS[threshold_index]


def _source_length(draw: int) -> int:
    return KNOWN_LENGTHS[(draw * 3 + 1) % len(KNOWN_LENGTHS)]


def _collate(records: list[CNPRecord], device: torch.device) -> Batch:
    if not records:
        raise ValueError("cannot collate an empty CNP batch")
    width = max(record.state.width for record in records)
    batch_size = len(records)
    values = torch.zeros((batch_size, width, 8), dtype=torch.float32, device=device)
    valid = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    item_ids = torch.zeros((batch_size, width), dtype=torch.int64, device=device)
    target = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    queries = torch.empty((batch_size, 8), dtype=torch.float32, device=device)
    thresholds = torch.empty((batch_size,), dtype=torch.float32, device=device)
    for index, record in enumerate(records):
        record_width = record.state.width
        values[index, :record_width] = record.state.values[0].to(device)
        valid[index, :record_width] = record.state.valid[0].to(device)
        item_ids[index, :record_width] = record.state.item_ids[0].to(device)
        target[index, :record_width] = record.target[0].to(device)
        queries[index] = record.arguments.query[0].to(device)
        thresholds[index] = record.arguments.threshold[0].to(device)
    return Batch(
        state=SetState(values=values, valid=valid, item_ids=item_ids),
        arguments=SelectArguments(query=queries, threshold=thresholds),
        target=target,
    )


def _source_records(step: int, batch_sets: int, audit: DataAudit | None = None) -> list[CNPRecord]:
    world = make_world()
    records: list[CNPRecord] = []
    for offset in range(batch_sets):
        draw = step * batch_sets + offset
        condition_key, threshold = _source_condition(draw)
        record = make_record(
            role="source_train",
            condition_key=condition_key,
            length=_source_length(draw),
            threshold=threshold,
            example_index=draw,
            world=world,
        )
        records.append(record)
        if audit is not None:
            audit.add(record)
    return records


def _development_records(
    length: int, threshold: float, query_index: int, audit: DataAudit | None = None
) -> Iterator[CNPRecord]:
    world = make_world()
    condition_key = f"dev_{query_index:02d}"
    for example_index in range(DEVELOPMENT_SETS_PER_QUERY):
        record = make_record(
            role="dev_eval",
            condition_key=condition_key,
            length=length,
            threshold=threshold,
            example_index=example_index,
            world=world,
        )
        if audit is not None:
            audit.add(record)
        yield record


def _preflight_data(config: dict[str, Any]) -> dict[str, Any]:
    """Generate and audit every CNP-002 source/dev record before any model updates."""

    initial = config["initial_training"]
    audit = DataAudit()
    for step in range(int(initial["steps"])):
        _source_records(step, int(initial["batch_sets"]), audit)
    for length in KNOWN_LENGTHS:
        for threshold in (*SOURCE_THRESHOLDS, *INTERPOLATION_THRESHOLDS):
            for query_index in range(DEVELOPMENT_QUERY_COUNT):
                list(_development_records(length, threshold, query_index, audit))
    manifest = audit.manifest()
    rates = manifest["development_cell_positive_rates"]
    assert isinstance(rates, dict)
    degenerate = {key: value for key, value in rates.items() if not 0.05 <= float(value) <= 0.95}
    if degenerate:
        raise ValueError(
            "DEGENERATE_PANEL: development positive rates outside [0.05, 0.95]: "
            f"{degenerate}"
        )
    return manifest


def _checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path
    )


def _train(
    model: nn.Module,
    *,
    steps: int,
    batch_sets: int,
    device: torch.device,
    checkpoints: Path,
    metric_sink: list[dict[str, Any]],
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    model.train()
    _checkpoint(checkpoints / "step_0000.pt", model, optimizer, 0)
    started = time.perf_counter()
    last_loss = float("nan")
    for step in range(steps):
        batch = _collate(_source_records(step, batch_sets), device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch.state, batch.arguments)
        loss = masked_bce_loss(result.logits, batch.target, batch.state.valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        completed = step + 1
        if completed in {1000, 2000, 4000} or completed == steps:
            _checkpoint(checkpoints / f"step_{completed:04d}.pt", model, optimizer, completed)
            metric_sink.append({"event": "checkpoint", "step": completed, "loss": last_loss})
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"steps": steps, "last_loss": last_loss, "seconds": time.perf_counter() - started}


def _evaluate(
    model: nn.Module,
    *,
    device: torch.device,
    causal: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the fixed unused development panel; no output selects a checkpoint."""

    rows: list[dict[str, Any]] = []
    whole_causal = CausalAccumulator()
    model.eval()
    with torch.inference_mode():
        for length in KNOWN_LENGTHS:
            for threshold in (*SOURCE_THRESHOLDS, *INTERPOLATION_THRESHOLDS):
                accumulator = MetricsAccumulator()
                causal_accumulator = CausalAccumulator()
                for query_index in range(DEVELOPMENT_QUERY_COUNT):
                    records = list(_development_records(length, threshold, query_index))
                    wrong_records = list(
                        _development_records(
                            length, threshold, (query_index + 1) % DEVELOPMENT_QUERY_COUNT
                        )
                    )
                    for start in range(0, DEVELOPMENT_SETS_PER_QUERY, EVAL_BATCH_SETS):
                        batch = _collate(records[start : start + EVAL_BATCH_SETS], device)
                        result = model(batch.state, batch.arguments)
                        accumulator.add(result.selected, batch.target, batch.state.valid)
                        if causal:
                            wrong_batch = _collate(
                                wrong_records[start : start + EVAL_BATCH_SETS], device
                            )
                            wrong_result = model(batch.state, wrong_batch.arguments)
                            controls = reference_controls(
                                batch.state,
                                batch.arguments,
                                wrong_batch.arguments,
                                make_world(),
                            )
                            causal_accumulator.add(
                                batch.target,
                                batch.state.valid,
                                result.selected,
                                {
                                    "wrong_family": controls.wrong_family.selected,
                                    "wrong_argument": wrong_result.selected,
                                    "none": batch.state.valid,
                                },
                                {
                                    "wrong_family": controls.wrong_family.selected,
                                    "wrong_argument": controls.wrong_argument.selected,
                                    "none": controls.none.valid,
                                },
                            )
                metrics = asdict(accumulator.result())
                row: dict[str, Any] = {
                    "event": "evaluation_cell",
                    "length": length,
                    "threshold": threshold,
                    "metrics": metrics,
                }
                if causal:
                    causal_row = causal_accumulator.report()
                    row["causal"] = causal_row
                    whole_causal.merge(causal_accumulator)
                rows.append(row)
    return rows, whole_causal.report() if causal else {"status": "NOT_APPLICABLE"}


def _g1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        balanced = metrics["balanced_accuracy"]
        f1 = metrics["mean_set_f1"]
        if balanced is None or float(balanced) < 0.95 or float(f1) < 0.90:
            failed.append(
                {
                    "length": row["length"],
                    "threshold": row["threshold"],
                    "balanced_accuracy": balanced,
                    "mean_set_f1": f1,
                }
            )
    return {"status": "PASS" if not failed else "FAIL", "failed_cells": failed}


def _model_bytes(model: nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


def _code_hashes(repository_root: Path) -> dict[str, str]:
    paths = sorted((repository_root / "src/apc/cnp").glob("*.py"))
    paths.append(repository_root / "scripts/run_cnp.py")
    return {str(path.relative_to(repository_root)): source_hash(path) for path in paths}


def _git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _report_markdown(report: dict[str, Any]) -> str:
    g1 = report["g1"]
    assert isinstance(g1, dict)
    lines = [
        "# CNP-002 development report",
        "",
        f"- Task result: `{report['task_result']}`",
        f"- G1: `{g1['status']}`",
        f"- Run ID: `{report['run_id']}`",
        f"- Legacy sealed access: `{report['legacy_sealed_access']}`",
        "",
        "## Methods",
        "",
        "| Method | Seed | Train seconds | G1 |",
        "| --- | ---: | ---: | --- |",
    ]
    methods = report["methods"]
    assert isinstance(methods, list)
    for method in methods:
        assert isinstance(method, dict)
        g1_status = method.get("g1", {}).get("status", "N/A")
        lines.append(
            f"| {method['method']} | {method['model_seed']} | "
            f"{method['training']['seconds']:.3f} | {g1_status} |"
        )
    lines.extend(
        [
            "",
            "The full per-cell metrics, controls, data audit, and accounting are in `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_development(config_path: Path, output_root: Path, run_id: str | None = None) -> Path:
    """Run the CNP-002 fixed two-seed development experiment once into a fresh directory."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = audit_cnp_seed_registry(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: CNP-002 requires one CUDA device")
    repository_root = config_path.parents[2]
    config_hash = canonical_json_hash(config)
    identifier = run_id or f"{config_hash[:12]}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = output_root / identifier
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing CNP run directory: {run_directory}")
    run_directory.mkdir(parents=True)
    (run_directory / "bundle").mkdir()
    (run_directory / "checkpoints").mkdir()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    initial = config["initial_training"]
    run_manifest = {
        "program": "cnp_v1",
        "task": "CNP-002",
        "status": "RUNNING",
        "run_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "code_hashes": _code_hashes(repository_root),
        "git_commit": _git_commit(repository_root),
        "seed_audit": audit,
        "legacy_sealed_access": 0,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "budget": {
            "conditional_metric_unconditioned_steps": 24000,
            "raw_distance_steps": 2000,
            "wall_clock_seconds": 8 * 60 * 60,
            "vram_bytes": 12 * 1024**3,
            "process_ram_bytes": 32 * 1024**3,
        },
    }
    _write_json(run_directory / "run_manifest.json", run_manifest)
    data_manifest = _preflight_data(config)
    _write_json(run_directory / "data_manifest.json", data_manifest)

    methods: list[dict[str, Any]] = []
    metric_events: list[dict[str, Any]] = []
    development_seeds = config["seeds"]["development_model"]
    assert isinstance(development_seeds, list)
    method_factories: list[tuple[str, int, bool, Any]] = [
        ("CONDITIONAL_MLP", int(initial["steps"]), True, ConditionalSelectPrimitive),
        ("LEARNED_METRIC", int(initial["steps"]), False, LearnedMetricBaseline),
        ("UNCONDITIONED", int(initial["steps"]), False, UnconditionedBaseline),
        ("RAW_DISTANCE_FIT", 1000, False, RawDistanceBaseline),
    ]
    try:
        for model_seed in development_seeds:
            for method, steps, causal, factory in method_factories:
                torch.manual_seed(int(model_seed))
                torch.cuda.manual_seed_all(int(model_seed))
                if factory is UnconditionedBaseline:
                    model: nn.Module = UnconditionedBaseline(ConditionalSelectPrimitive(0))
                elif factory is ConditionalSelectPrimitive:
                    model = ConditionalSelectPrimitive(0)
                else:
                    model = factory()
                model.to(device)
                checkpoint_directory = (
                    run_directory / "checkpoints" / f"{method.lower()}_seed_{model_seed}"
                )
                checkpoint_directory.mkdir()
                local_events: list[dict[str, Any]] = []
                training = _train(
                    model,
                    steps=steps,
                    batch_sets=int(initial["batch_sets"]),
                    device=device,
                    checkpoints=checkpoint_directory,
                    metric_sink=local_events,
                )
                rows, causal_report = _evaluate(model, device=device, causal=causal)
                g1 = _g1(rows) if method == "CONDITIONAL_MLP" else {"status": "NOT_APPLICABLE"}
                method_report: dict[str, Any] = {
                    "method": method,
                    "model_seed": model_seed,
                    "training": training,
                    "model_bytes": _model_bytes(model),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "evaluation": rows,
                    "causal": causal_report,
                    "g1": g1,
                    "peak_cuda_bytes_so_far": int(torch.cuda.max_memory_allocated()),
                }
                methods.append(method_report)
                metric_events.extend(
                    [
                        {"method": method, "model_seed": model_seed, **event}
                        for event in [*local_events, *rows]
                    ]
                )
                if isinstance(model, ConditionalSelectPrimitive):
                    bundle_manifest = build_manifest(
                        model,
                        model_seed=int(model_seed),
                        config=config,
                        usage_conditions={"stage": "CNP-002", "method": method},
                    )
                    save_bundle(
                        run_directory / "bundle" / f"{method.lower()}_seed_{model_seed}",
                        model,
                        bundle_manifest,
                    )
                elif isinstance(model, UnconditionedBaseline):
                    if not isinstance(model.conditional, ConditionalSelectPrimitive):
                        raise RuntimeError(
                            "CNP unconditioned baseline must wrap a conditional primitive"
                        )
                    primitive = model.conditional
                    bundle_manifest = build_manifest(
                        primitive,
                        model_seed=int(model_seed),
                        config=config,
                        usage_conditions={"stage": "CNP-002", "method": method},
                    )
                    save_bundle(
                        run_directory / "bundle" / f"{method.lower()}_seed_{model_seed}",
                        primitive,
                        bundle_manifest,
                    )
                del model
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        conditional = [method for method in methods if method["method"] == "CONDITIONAL_MLP"]
        conditional_g1 = [method["g1"]["status"] for method in conditional]
        task_result = "G1_PASS" if conditional_g1 == ["PASS", "PASS"] else "G1_FAIL_STOP"
        report = {
            "task": "CNP-002",
            "task_result": task_result,
            "run_id": identifier,
            "legacy_sealed_access": 0,
            "g1": {
                "status": "PASS" if task_result == "G1_PASS" else "FAIL",
                "per_seed": conditional_g1,
            },
            "methods": methods,
            "total_wall_seconds": elapsed,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_ram_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            ),
            "wall_clock_budget_status": "PASS" if elapsed <= 8 * 60 * 60 else "RESOURCE_STOP",
        }
        _write_json(run_directory / "report.json", report)
        (run_directory / "report.md").write_text(_report_markdown(report), encoding="utf-8")
        with (run_directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for event in metric_events:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
        run_manifest["status"] = "COMPLETE"
        run_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_directory / "run_manifest.json", run_manifest)
        return run_directory
    except Exception as error:
        run_manifest["status"] = "FAILED"
        run_manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(run_directory / "run_manifest.json", run_manifest)
        raise
