"""Fixed CNP-003 confirmation execution over the sealed CNP confirmation split."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from apc.cnp.artifacts import build_manifest, save_bundle
from apc.cnp.baselines import LearnedMetricBaseline, RawDistanceBaseline, UnconditionedBaseline
from apc.cnp.contracts import CNPPrimitiveCall, CNPRecipe
from apc.cnp.data import (
    INTERPOLATION_LENGTHS,
    INTERPOLATION_THRESHOLDS,
    KNOWN_LENGTHS,
    SOURCE_THRESHOLDS,
    CNPRecord,
    canonical_json_hash,
    make_record,
    make_world,
)
from apc.cnp.development import (
    DEVELOPMENT_QUERY_COUNT,
    DEVELOPMENT_SETS_PER_QUERY,
    EVAL_BATCH_SETS,
    CausalAccumulator,
    DataAudit,
    MetricsAccumulator,
    _code_hashes,
    _collate,
    _git_commit,
    _model_bytes,
    _source_records,
    _train,
    _write_json,
)
from apc.cnp.execution import execute_recipe
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.reference import reference_controls, reference_select
from apc.cnp.seed_registry import audit_cnp_seed_registry
from apc.primitives.bank import PrimitiveBank


def _confirmation_records(
    length: int, threshold: float, query_index: int, audit: DataAudit | None = None
) -> Iterator[CNPRecord]:
    world = make_world()
    condition_key = f"confirm_{query_index:02d}"
    for example_index in range(DEVELOPMENT_SETS_PER_QUERY):
        record = make_record(
            role="confirm_eval",
            condition_key=condition_key,
            length=length,
            threshold=threshold,
            example_index=example_index,
            world=world,
        )
        if audit is not None:
            audit.add(record)
        yield record


def _preflight_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    """Audit source/confirmation disjointness before constructing a confirmation model."""

    initial = config["initial_training"]
    audit = DataAudit()
    cell_items: dict[str, int] = {}
    cell_positive: dict[str, int] = {}
    for step in range(int(initial["steps"])):
        _source_records(step, int(initial["batch_sets"]), audit)
    for length in (*KNOWN_LENGTHS, *INTERPOLATION_LENGTHS):
        for threshold in (*SOURCE_THRESHOLDS, *INTERPOLATION_THRESHOLDS):
            cell = f"length={length}|threshold={threshold:.2f}"
            for query_index in range(DEVELOPMENT_QUERY_COUNT):
                for record in _confirmation_records(length, threshold, query_index, audit):
                    cell_items[cell] = cell_items.get(cell, 0) + int(record.state.valid.sum())
                    cell_positive[cell] = cell_positive.get(cell, 0) + int(record.target.sum())
    manifest = audit.manifest()
    manifest["confirmation_cell_positive_rates"] = {
        cell: cell_positive[cell] / cell_items[cell] for cell in sorted(cell_items)
    }
    manifest["confirmation_query_count"] = DEVELOPMENT_QUERY_COUNT
    manifest["source_query_count"] = int(initial["source_query_count"])
    return manifest


def _confirmation_boundary(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless confirmation uses the fixed 64-source/32-confirm query boundary."""

    initial = config["initial_training"]
    source_count = manifest.get("source_query_count")
    confirmation_count = manifest.get("confirmation_query_count")
    expected_source = initial["source_query_count"]
    passed = source_count == expected_source == 64 and confirmation_count == 32
    result = {
        "status": "PASS" if passed else "FAIL",
        "source_query_count": source_count,
        "confirmation_query_count": confirmation_count,
        "expected_source_query_count": expected_source,
    }
    if not passed:
        raise ValueError(f"CNP confirmation query-boundary violation: {result}")
    return result


def _quantile_linear(values: list[float], quantile: float) -> float:
    """Return the fixed linear-interpolation quantile used by the G2 protocol."""

    if not values:
        raise ValueError("cannot calculate a quantile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def _causal_cell_gate(causal: dict[str, Any]) -> dict[str, Any]:
    """Apply G2 causal conditions to one length-threshold cell without mean rescue."""

    if int(causal.get("effectful_items", 0)) < 32:
        return {"status": "INCONCLUSIVE", "reason": "effectful_items_below_32"}
    correct = float(causal["correct"])
    gaps = [
        float(causal["correct_minus_wrong_family"]),
        float(causal["correct_minus_wrong_argument"]),
        float(causal["correct_minus_none"]),
    ]
    passed = correct >= 0.95 and all(gap >= 0.50 for gap in gaps)
    return {
        "status": "PASS" if passed else "FAIL",
        "correct_accuracy": correct,
        "minimum_gap": min(gaps),
    }


def _g2(rows: list[dict[str, Any]], fresh_load: dict[str, Any]) -> dict[str, Any]:
    """Apply every CNP-003 G2 condition to fixed known-length cells."""

    failed: list[dict[str, Any]] = []
    inconclusive: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        causal = row["causal"]
        assert isinstance(causal, dict)
        cell_gate = _causal_cell_gate(causal)
        if cell_gate["status"] == "INCONCLUSIVE":
            inconclusive.append(
                {"length": row["length"], "threshold": row["threshold"], **cell_gate}
            )
            continue
        valid = (
            metrics["balanced_accuracy"] is not None
            and float(metrics["balanced_accuracy"]) >= 0.95
            and float(metrics["mean_set_f1"]) >= 0.90
            and float(row["exemplar_f1_p10"]) >= 0.80
            and cell_gate["status"] == "PASS"
        )
        if not valid:
            failed.append(
                {
                    "length": row["length"],
                    "threshold": row["threshold"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "mean_set_f1": metrics["mean_set_f1"],
                    "exemplar_f1_p10": row["exemplar_f1_p10"],
                    "causal": cell_gate,
                }
            )
    status = (
        "PASS" if not failed and not inconclusive and fresh_load["status"] == "PASS" else "FAIL"
    )
    return {
        "status": status,
        "failed_cells": failed,
        "inconclusive_cells": inconclusive,
        "fresh_load": fresh_load,
    }


def _evaluate_confirmation(
    model: nn.Module, *, device: torch.device, include_interpolation: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure an unused confirmation panel and causal controls once, with no tuning path."""

    lengths = (*KNOWN_LENGTHS, *INTERPOLATION_LENGTHS) if include_interpolation else KNOWN_LENGTHS
    rows: list[dict[str, Any]] = []
    whole_causal = CausalAccumulator()
    model.eval()
    with torch.inference_mode():
        for length in lengths:
            for threshold in (*SOURCE_THRESHOLDS, *INTERPOLATION_THRESHOLDS):
                accumulator = MetricsAccumulator()
                causal_accumulator = CausalAccumulator()
                exemplar_f1: list[float] = []
                for query_index in range(DEVELOPMENT_QUERY_COUNT):
                    query_accumulator = MetricsAccumulator()
                    records = list(_confirmation_records(length, threshold, query_index))
                    wrong_records = list(
                        _confirmation_records(
                            length, threshold, (query_index + 1) % DEVELOPMENT_QUERY_COUNT
                        )
                    )
                    for start in range(0, DEVELOPMENT_SETS_PER_QUERY, EVAL_BATCH_SETS):
                        batch = _collate(records[start : start + EVAL_BATCH_SETS], device)
                        result = model(batch.state, batch.arguments)
                        accumulator.add(result.selected, batch.target, batch.state.valid)
                        query_accumulator.add(result.selected, batch.target, batch.state.valid)
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
                            controls.wrong_family.selected,
                            wrong_result.selected,
                        )
                    exemplar_f1.append(query_accumulator.result().mean_set_f1)
                metrics = asdict(accumulator.result())
                causal = causal_accumulator.report()
                row = {
                    "event": "confirmation_cell",
                    "scope": (
                        "known" if length in KNOWN_LENGTHS else "interpolation_or_extrapolation"
                    ),
                    "length": length,
                    "threshold": threshold,
                    "metrics": metrics,
                    "exemplar_f1": exemplar_f1,
                    "exemplar_f1_p10": _quantile_linear(exemplar_f1, 0.10),
                    "causal": causal,
                }
                rows.append(row)
                whole_causal.correct_hits += causal_accumulator.correct_hits
                whole_causal.wrong_family_hits += causal_accumulator.wrong_family_hits
                whole_causal.wrong_argument_hits += causal_accumulator.wrong_argument_hits
                whole_causal.none_hits += causal_accumulator.none_hits
                whole_causal.effectful_items += causal_accumulator.effectful_items
                whole_causal.effectful_sets += causal_accumulator.effectful_sets
    return rows, whole_causal.report()


def _fresh_load_check(
    directory: Path, config: dict[str, Any], model_seed: int, output_directory: Path
) -> dict[str, Any]:
    """Require a separate Python process to restore a saved CNP primitive exactly on CPU."""

    record = make_record(
        role="source_train",
        condition_key="source_00",
        length=8,
        threshold=0.8,
        example_index=0,
        world=make_world(),
    )
    expected_model = ConditionalSelectPrimitive(0).eval()
    from apc.cnp.artifacts import load_bundle

    load_bundle(directory, expected_model, expected_config=config)
    expected = expected_model(record.state, record.arguments)
    child_output = output_directory / f"fresh_load_seed_{model_seed}.pt"
    program = "\n".join(
        (
            "import json",
            "from pathlib import Path",
            "import torch",
            "from apc.cnp.artifacts import load_bundle",
            "from apc.cnp.data import make_record, make_world",
            "from apc.cnp.primitive import ConditionalSelectPrimitive",
            f"directory = Path({str(directory)!r})",
            f"config = json.loads(Path({str(output_directory / 'config.json')!r}).read_text())",
            "model = ConditionalSelectPrimitive(0).eval()",
            "load_bundle(directory, model, expected_config=config)",
            "record = make_record("
            "role='source_train', condition_key='source_00', length=8, "
            "threshold=0.8, example_index=0, world=make_world())",
            "result = model(record.state, record.arguments)",
            "torch.save({'logits': result.logits, 'selected': result.selected}, "
            f"{str(child_output)!r})",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0 or not child_output.is_file():
        return {"status": "FAIL", "stderr": completed.stderr[-1000:]}
    loaded = torch.load(child_output, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        return {"status": "FAIL", "reason": "fresh process output is invalid"}
    logits = loaded.get("logits")
    selected = loaded.get("selected")
    if not isinstance(logits, torch.Tensor) or not isinstance(selected, torch.Tensor):
        return {"status": "FAIL", "reason": "fresh process tensors are missing"}
    exact = torch.equal(expected.logits, logits) and torch.equal(expected.selected, selected)
    return {
        "status": "PASS" if exact else "FAIL",
        "logits_exact": bool(torch.equal(expected.logits, logits)),
        "mask_exact": bool(torch.equal(expected.selected, selected)),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    """Render a concise report that points reviewers to machine-readable evidence."""

    g2 = report["g2"]
    assert isinstance(g2, dict)
    return "\n".join(
        (
            "# CNP-003 confirmation report",
            "",
            f"- Task result: `{report['task_result']}`",
            f"- G2: `{g2['status']}`",
            f"- Run ID: `{report['run_id']}`",
            f"- Legacy sealed access: `{report['legacy_sealed_access']}`",
            "",
            "Per-cell metrics, causal controls, interpolation rows, composition rows,",
            "fresh-load results, and resource accounting are in `report.json`.",
            "",
        )
    )


def _composition_panel(model: ConditionalSelectPrimitive, device: torch.device) -> dict[str, Any]:
    """Measure fixed cyclic two-SELECT recipes plus deterministic terminals as a non-G2 panel."""

    bank = PrimitiveBank([model])
    family_to_id = {"CONDITIONAL_SELECT": 0}
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for length in (4, 8, 16):
            for first_threshold in SOURCE_THRESHOLDS:
                for second_threshold in SOURCE_THRESHOLDS:
                    acc = MetricsAccumulator()
                    count_exact = 0
                    sum_absolute_error = 0.0
                    total_sets = 0
                    for query_index in range(DEVELOPMENT_QUERY_COUNT):
                        records = list(_confirmation_records(length, first_threshold, query_index))
                        second_records = list(
                            _confirmation_records(
                                length,
                                second_threshold,
                                (query_index + 1) % DEVELOPMENT_QUERY_COUNT,
                            )
                        )
                        for start in range(0, DEVELOPMENT_SETS_PER_QUERY, EVAL_BATCH_SETS):
                            batch = _collate(records[start : start + EVAL_BATCH_SETS], device)
                            second = _collate(
                                second_records[start : start + EVAL_BATCH_SETS], device
                            )
                            recipe = CNPRecipe(
                                "confirm_two_select_count",
                                (
                                    CNPPrimitiveCall("CONDITIONAL_SELECT", batch.arguments),
                                    CNPPrimitiveCall("CONDITIONAL_SELECT", second.arguments),
                                    CNPPrimitiveCall("COUNT"),
                                ),
                            )
                            predicted = execute_recipe(bank, family_to_id, batch.state, recipe)
                            reference_first = reference_select(
                                batch.state, batch.arguments, make_world()
                            )
                            reference_second = reference_select(
                                reference_first.state, second.arguments, make_world()
                            )
                            acc.add(
                                predicted.state.valid,
                                reference_second.selected,
                                batch.state.valid,
                            )
                            assert predicted.terminal is not None
                            reference_count = reference_second.state.valid.sum(
                                dim=1, dtype=torch.int64
                            )
                            count_exact += int((predicted.terminal == reference_count).sum())
                            sum_absolute_error += float(
                                (predicted.state.values[..., 0]
                                * predicted.state.valid.to(predicted.state.values.dtype)).sum(dim=1)
                                .sub(
                                    (reference_second.state.values[..., 0]
                                    * reference_second.state.valid.to(
                                        reference_second.state.values.dtype
                                    )
                                ).sum(dim=1)
                                )
                                .abs()
                                .sum()
                            )
                            total_sets += batch.state.batch_size
                    rows.append(
                        {
                            "length": length,
                            "first_threshold": first_threshold,
                            "second_threshold": second_threshold,
                            "metrics": asdict(acc.result()),
                            "count_exact_match": count_exact / total_sets,
                            "sum_first_mean_absolute_error": sum_absolute_error / total_sets,
                        }
                    )
    return {"status": "REPORTED_NON_G2", "rows": rows}


def run_confirmation(config_path: Path, output_root: Path, run_id: str | None = None) -> Path:
    """Execute the authorized CNP-003 five-seed confirmation protocol once."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = audit_cnp_seed_registry(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_STOP: CNP-003 requires one CUDA device")
    repository_root = config_path.parents[2]
    config_hash = canonical_json_hash(config)
    identifier = run_id or f"{config_hash[:12]}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_directory = output_root / identifier
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing CNP run directory: {run_directory}")
    run_directory.mkdir(parents=True)
    (run_directory / "bundle").mkdir()
    (run_directory / "checkpoints").mkdir()
    (run_directory / "fresh_load").mkdir()
    _write_json(run_directory / "fresh_load" / "config.json", config)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    initial = config["initial_training"]
    run_manifest: dict[str, Any] = {
        "program": "cnp_v1",
        "task": "CNP-003",
        "status": "RUNNING",
        "run_id": identifier,
        "created_at_utc": datetime.now(UTC).isoformat(),
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
            "training_steps": 60000,
            "raw_distance_steps": 5000,
            "wall_clock_seconds": 86400,
        },
    }
    _write_json(run_directory / "run_manifest.json", run_manifest)
    data_manifest = _preflight_confirmation(config)
    _write_json(run_directory / "data_manifest.json", data_manifest)
    query_boundary = _confirmation_boundary(config, data_manifest)
    confirmation_seeds = config["seeds"]["confirmation_model"]
    assert isinstance(confirmation_seeds, list)
    methods: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    factories: list[tuple[str, int, Any]] = [
        ("CONDITIONAL_MLP", int(initial["steps"]), ConditionalSelectPrimitive),
        ("LEARNED_METRIC", int(initial["steps"]), LearnedMetricBaseline),
        ("UNCONDITIONED", int(initial["steps"]), UnconditionedBaseline),
        ("RAW_DISTANCE_FIT", 1000, RawDistanceBaseline),
    ]
    try:
        for model_seed in confirmation_seeds:
            for method, steps, factory in factories:
                torch.manual_seed(int(model_seed))
                torch.cuda.manual_seed_all(int(model_seed))
                if factory is ConditionalSelectPrimitive:
                    model: nn.Module = ConditionalSelectPrimitive(0)
                elif factory is UnconditionedBaseline:
                    model = UnconditionedBaseline(ConditionalSelectPrimitive(0))
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
                rows, causal = _evaluate_confirmation(
                    model, device=device, include_interpolation=True
                )
                known_rows = [row for row in rows if row["scope"] == "known"]
                fresh_load: dict[str, Any] = {"status": "NOT_APPLICABLE"}
                composition: dict[str, Any] = {"status": "NOT_APPLICABLE"}
                if isinstance(model, ConditionalSelectPrimitive):
                    bundle_directory = (
                        run_directory / "bundle" / f"{method.lower()}_seed_{model_seed}"
                    )
                    manifest = build_manifest(
                        model,
                        model_seed=int(model_seed),
                        config=config,
                        usage_conditions={"stage": "CNP-003", "method": method},
                    )
                    save_bundle(bundle_directory, model, manifest)
                    fresh_load = _fresh_load_check(
                        bundle_directory, config, int(model_seed), run_directory / "fresh_load"
                    )
                    composition = _composition_panel(model, device)
                    gate = _g2(known_rows, fresh_load)
                elif isinstance(model, UnconditionedBaseline):
                    if not isinstance(model.conditional, ConditionalSelectPrimitive):
                        raise RuntimeError(
                            "CNP unconditioned baseline must wrap a conditional primitive"
                        )
                    primitive = model.conditional
                    manifest = build_manifest(
                        primitive,
                        model_seed=int(model_seed),
                        config=config,
                        usage_conditions={"stage": "CNP-003", "method": method},
                    )
                    save_bundle(
                        run_directory / "bundle" / f"{method.lower()}_seed_{model_seed}",
                        primitive,
                        manifest,
                    )
                    gate = {"status": "NOT_APPLICABLE"}
                else:
                    gate = {"status": "NOT_APPLICABLE"}
                method_report = {
                    "method": method,
                    "model_seed": model_seed,
                    "training": training,
                    "model_bytes": _model_bytes(model),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "evaluation": rows,
                    "causal": causal,
                    "g2": gate,
                    "fresh_load": fresh_load,
                    "composition": composition,
                    "peak_cuda_bytes_so_far": int(torch.cuda.max_memory_allocated()),
                }
                methods.append(method_report)
                events.extend(
                    [
                        {"method": method, "model_seed": model_seed, **event}
                        for event in [*local_events, *rows]
                    ]
                )
                del model
                torch.cuda.empty_cache()
        mlp = [item for item in methods if item["method"] == "CONDITIONAL_MLP"]
        g2_by_seed = [item["g2"]["status"] for item in mlp]
        task_result = (
            "G2_PASS"
            if g2_by_seed == ["PASS"] * 5 and query_boundary["status"] == "PASS"
            else "G2_FAIL_STOP"
        )
        report = {
            "task": "CNP-003",
            "task_result": task_result,
            "run_id": identifier,
            "legacy_sealed_access": 0,
            "g2": {
                "status": "PASS" if task_result == "G2_PASS" else "FAIL",
                "per_seed": g2_by_seed,
                "query_boundary": query_boundary,
            },
            "methods": methods,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_ram_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            ),
        }
        _write_json(run_directory / "report.json", report)
        (run_directory / "report.md").write_text(_report_markdown(report), encoding="utf-8")
        with (run_directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for event in events:
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
