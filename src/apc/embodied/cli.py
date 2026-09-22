"""Build and evaluate the independent embodied APC infrastructure experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from apc.embodied.bundle import BASE_COMMIT, load_bundle, load_config, save_bundle, write_json
from apc.embodied.control import (
    MotionBank,
    MotionCompositionLibrary,
    MotionRecipe,
    run_episode,
    task_fingerprint,
    try_library,
)
from apc.embodied.learning import (
    consolidate_after_failure,
    expert_policy,
    fit_from_demonstrations,
    validate_policy,
)
from apc.embodied.viewer import write_replay
from apc.embodied.world import make_task

FAMILIES = ("planar", "spatial", "route", "detour")


def environment_record(backend: str) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform(), "backend": backend,
                            "base_commit": BASE_COMMIT, "device": "cpu"}
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode()+b"\0"+source.read_bytes())
    info["implementation_sha256"] = digest.hexdigest()
    if backend == "apc":
        import torch
        info["torch"] = torch.__version__
    return info


def build(config_path: Path, output: Path, *, backend: str = "apc") -> dict[str, Any]:
    """Failure-triggered fit, fixed shadow gate, install, reuse and fresh-load check.

    The declared research intervention is explicit: bootstrap horizontal motion,
    then expose a 3D task. The candidate's linear architecture and PD teacher are
    supplied, not autonomously discovered. Test tasks are not generated here.
    """
    config, splits = load_config(config_path)
    report: dict[str, Any] = {"scope": "embodied infrastructure smoke, not research-gate evidence",
                              "environment": environment_record(backend),
                              "status": "RUNNING", "test_evaluation": "NOT_EXECUTED"}
    bank = MotionBank(backend)
    library = MotionCompositionLibrary()
    episodes: list[dict[str, Any]] = []
    planar_shadow = [make_task(s, "planar") for s in splits.validation]
    initial = fit_from_demonstrations(config, splits.train, spatial=False, name="move_planar")
    report["bootstrap_fit"] = initial.report
    controls = validate_policy(config, expert_policy(config, spatial=False),
                               planar_shadow, backend=backend)
    initial_shadow = validate_policy(config, initial.primitive, planar_shadow, backend=backend)
    if not all(r.success for r in (*controls, *initial_shadow)):
        report["status"] = "STOP_BOOTSTRAP_GATE"
        report["bootstrap_shadow"] = [r.to_dict(include_trace=False) for r in initial_shadow]
        write_json(output / "rejected-bootstrap.json", initial.primitive.to_dict())
        return report
    bank.add(initial.primitive)
    del initial
    library.add(MotionRecipe("repeat_move_planar", ("move_planar",)))
    report["bootstrap_gate"] = {"expert_successes": len(controls),
                                 "learned_successes": len(initial_shadow)}
    stable_hashes = bank.fingerprint()
    known = try_library(config, planar_shadow[0], bank, library, record_trace=True)
    episodes.extend({"label": "known / reuse", "result": r.to_dict()} for r in known)
    novel = make_task(splits.validation[0], "spatial")
    failures = try_library(config, novel, bank, library, record_trace=True)
    episodes.extend({"label": "novel / before learning", "result": r.to_dict()} for r in failures)
    report["novel_attempts"] = [r.to_dict(include_trace=False) for r in failures]
    if not failures or failures[-1].success:
        report["status"] = "STOP_EXPECTED_NOVELTY_NOT_OBSERVED"
        return report
    candidate = fit_from_demonstrations(config, splits.train, spatial=True, name="move_3d")
    report["candidate_fit"] = candidate.report
    shadow_tasks = [make_task(s, family) for family in FAMILIES for s in splits.validation]
    gate = consolidate_after_failure(config, failures[-1], candidate.primitive,
                                     bank, library, shadow_tasks)
    report["consolidation"] = gate
    if not gate["installed"]:
        write_json(output / "rejected-candidate.json", candidate.primitive.to_dict())
        report["status"] = gate["status"]
    else:
        # Candidate workspace is released only after acceptance. Fit diagnostics
        # remain; no mutable candidate weights are retained by the runtime bank.
        del candidate
        recurrence = try_library(config, novel, bank, library, record_trace=True)
        episodes.extend({"label": "recurrence / after learning", "result": r.to_dict()}
                        for r in recurrence)
        composed = try_library(config, make_task(splits.validation[0], "detour"),
                               bank, library, record_trace=True)
        episodes.extend({"label": "composition / over obstacle", "result": r.to_dict()}
                        for r in composed)
        report["recurrence"] = [r.to_dict(include_trace=False) for r in recurrence]
        report["composition"] = [r.to_dict(include_trace=False) for r in composed]
        report["stable_weights_unchanged"] = all(bank.fingerprint()[k] == v
                                                 for k, v in stable_hashes.items())
        report["capacity"] = {"resident_parameters": bank.resident_parameters,
                               "max_active_parameters_per_tick": 21,
                               "peak_temporary_candidate_parameters": 21,
                               "temporary_candidate_parameters_at_end": 0,
                               "primitive_count": len(bank.names()),
                               "recipe_count": len(library.recipes())}
        exposures = [task_fingerprint(make_task(s, family))
                     for s in splits.train for family in ("planar", "spatial")]
        exposures += [task_fingerprint(task) for task in shadow_tasks]
        save_bundle(output / "bank.json", config, splits, bank, library, exposures)
        fresh_config, _, fresh_bank, fresh_library, _ = load_bundle(
            output / "bank.json", backend=backend)
        fresh = try_library(fresh_config, novel, fresh_bank, fresh_library, record_trace=True)
        identical = (fresh_bank.fingerprint() == bank.fingerprint()
                     and fresh[-1].trace == recurrence[-1].trace)
        report["fresh_load"] = {"success": fresh[-1].success, "identical_replay": identical,
                                 "training_calls": 0}
        passed = (recurrence[-1].success and composed[-1].success and identical
                  and report["stable_weights_unchanged"])
        report["status"] = "INFRASTRUCTURE_SMOKE_PASS" if passed else "STOP_RECURRENCE_CHECK"
    write_json(output / "episodes.json", {"world": config.to_dict(), "episodes": episodes})
    write_replay(output / "replay.html", config.to_dict(), episodes)
    return report


def evaluate(bundle_path: Path, output: Path, *, backend: str = "apc") -> dict[str, Any]:
    """Read-only held-out evaluation: no call to fit, expert or consolidation."""
    config, splits, bank, library, exposure = load_bundle(bundle_path, backend=backend)
    before = bank.fingerprint()
    records: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    for family in FAMILIES:
        rows: dict[str, list[dict[str, Any]]] = {
            key: [] for key in ("correct", "none", "wrong_arguments", "wrong_primitive")}
        for seed in splits.test:
            task = make_task(seed, family)
            if task_fingerprint(task) in exposure:
                raise ValueError("Held-out task overlaps build exposures")
            recipe = library.recipes()[0]
            for control in rows:
                calls = recipe.bind(task)
                intervention = control
                if control == "wrong_primitive":
                    other = next((name for name in bank.names()
                                  if name != recipe.operations[0]), None)
                    if other is None:
                        raise ValueError("Wrong-primitive control requires another installed family")
                    calls = MotionRecipe("wrong", (other,)).bind(task)
                    intervention = "correct"
                result = run_episode(config, task, bank, calls, control=intervention,
                                     record_trace=(seed == splits.test[0]
                                                   and control == "correct"))
                rows[control].append(result.to_dict(include_trace=False))
                if result.trace:
                    episodes.append({"label": f"held-out / {family}", "result": result.to_dict()})
        records[family] = {control: {"successes": sum(r["success"] for r in results),
                                     "episodes": len(results),
                                     "success_rate": (sum(r["success"] for r in results)
                                                      / len(results)),
                                     "results": results} for control, results in rows.items()}
    if before != bank.fingerprint():
        raise RuntimeError("Evaluation modified stable weights")
    write_json(output / "episodes.json", {"world": config.to_dict(), "episodes": episodes})
    write_replay(output / "replay.html", config.to_dict(), episodes)
    return {"status": "EVALUATION_COMPLETE", "environment": environment_record(backend),
            "training_calls": 0, "weights_unchanged": True, "test_seeds": list(splits.test),
            "families": records,
            "interpretation": "Development harness only. Wrong planar primitive is not an "
                              "effectful intervention on planar tasks. No autonomous planner, "
                              "learned router or open-ended skill discovery claim."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/embodied/reference.json"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing paths are never overwritten")
    parser.add_argument("--backend", choices=("apc", "numpy"), default="apc")
    args = parser.parse_args(argv)
    if args.mode == "evaluate" and args.bundle is None:
        parser.error("evaluate requires --bundle; it never trains a missing bundle")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Output already exists; choose a fresh run directory")
    start = time.perf_counter()
    try:
        report = (build(args.config, args.output, backend=args.backend) if args.mode == "build"
                  else evaluate(args.bundle, args.output, backend=args.backend))
        report["wall_seconds"] = time.perf_counter()-start
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["process_peak_rss_bytes"] = rss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            report["process_peak_rss_bytes"] = None
        write_json(args.output / "report.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
        return 0 if report["status"] in {"INFRASTRUCTURE_SMOKE_PASS", "EVALUATION_COMPLETE"} else 2
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, ImportError) as error:
        write_json(args.output / "error.json", {"status": "ERROR", "error": str(error),
                                               "wall_seconds": time.perf_counter()-start})
        print(f"Embodied run failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
