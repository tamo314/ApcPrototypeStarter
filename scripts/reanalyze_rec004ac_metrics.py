"""Reanalyze saved REC-004AC endpoints without training or historical writes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_dense_trajectory_transition_audit as trajectory
from apc.evaluation import mirror_normal_cvof_protection_pilot as hard_freeze
from apc.evaluation import mirror_parallel_score_residual_pilot as ac
from apc.evaluation.mirror_attention_metrics import cvof_parameter_accounting
from apc.primitives.primitive import CrossPositionLengthBiasPrimitiveConfig
from apc.utils.model_bundle import canonical_state_hash

SOURCE_COMMIT = "6f54fdfe1639d44a466f05eab7061d813034749e"
SOURCE_DIR = Path("runs/phase_b_b2_model_bundle_recovery/rec004ac/run_001")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_evaluator() -> Any:
    """Compile only the historical read-only aggregator, not its training runner."""
    text = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:src/apc/evaluation/"
         "mirror_parallel_score_residual_pilot.py"], text=True, encoding="utf-8"
    )
    function = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == "evaluate_length10_metrics")
    namespace = dict(vars(ac))
    namespace.update(REC004AC_CORRECT_KEY_P4=4, REC004AC_CORRECT_KEY_P5=5)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "historical_aggregator", "exec"),
         namespace)
    return namespace[function.name]


def new_model(core: Any, state: dict[str, Any], *, initial: bool = False) -> Any:
    cfg = CrossPositionLengthBiasPrimitiveConfig(
        operation="MIRROR_HALVES", d_model=core.model.config.d_model,
        d_operator=32, n_head=4, d_operator_ff=64, vocab_size=10,
        max_sequence_length=32, arg_dim=16, bias_hidden_dim=32, length_ref=32,
    )
    model = ac.RoleSplitParallelScoreResidualCrossPositionLengthBiasPrimitive(
        primitive_id=0, config=cfg, residual_seed=10
    )
    if initial:
        initial_state = model.state_dict()
        initial_state.update(state)
        for suffix in ("in_proj.weight", "in_proj.bias", "position_embedding.weight"):
            initial_state[f"key_content_{suffix}"] = state[f"content_{suffix}"].clone()
        state = initial_state
    model.load_state_dict(state, strict=True)
    model.to(core.device).eval()
    model.requires_grad_(False)
    return model


def initial_parity(core: Any, state: dict[str, Any], examples: Any) -> dict[str, Any]:
    """Check zero-residual migration against both unchanged forward implementations."""
    initial = new_model(core, state, initial=True)
    reference = trajectory._new_primitive_from_state(core, state)
    role = ac.rec004z.RoleSplitCrossPositionLengthBiasPrimitive(0, initial.config)
    role.load_state_dict({k: v for k, v in initial.state_dict().items()
                          if not k.startswith("score_residual.")}, strict=True)
    role.to(core.device).eval().requires_grad_(False)
    return ac.verify_score_residual_initial_parity(
        core, initial, role, reference, examples, core.device
    )


def assert_em_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Missing or changed discrete evidence must block a metric-only correction."""
    for key in ("j0_sequence_em", "o1_sequence_em", "j0_position4_acc", "j0_position5_acc",
                "o1_position4_acc", "o1_position5_acc", "overall_j0_sequence_em",
                "length_10_j0_sequence_em"):
        if key in expected and actual[key] != expected[key]:
            raise RuntimeError(f"EM_PARITY_FAILED: {key}: {actual[key]} != {expected[key]}")


def write_report(output: Path, result: dict[str, Any]) -> None:
    """Summarize the corrected evidence without replacing historical conclusions."""
    length10 = result["length10"]
    counts = read(output / "cost_accounting.json")
    lines = [
        "# REC-004AC metric_v2 reanalysis", "",
        "Measurement/reproduction: PASS. Historical plasticity FAIL is preserved.",
        "No optimizer updates, candidate selection, bundle construction or sealed evaluation.",
        "", "| Corrected observation | Value |", "|---|---|",
        f"| Normal J0 sequence EM | {result['normal_j0_em']} |",
        *[f"| {key} | {length10[key]} |" for key in (
            "j0_sequence_em", "o1_sequence_em", "j0_p4_score_margin_median",
            "j0_p4_score_prob_median", "j0_p4_correct_key_rank_mean",
            "j0_p4_top1_key_recall", "j0_p4_top3_key_recall", "j0_p4_top5_key_recall",
        )],
        "", "Correct output-position 4/5 keys at length 10 are 0/9.",
        "The residual's median correct-key margin contribution is negative; small norm alone",
        "does not establish that amplification will repair routing or that low rank is impossible.",
        "", f"Target resident/active scalars: {counts['total_resident_parameters_model']}; "
        f"frozen/restored: {counts['frozen_restored_parameters']}; "
        f"update-eligible: {counts['active_trainable_parameters']}.",
        "", "Data manifest exactly matches AC run_001. All comparator EMs match historical JSONs.",
        "See metric_erratum.json for old/new observations, initial_parity.json for migration,",
        "and side_effect_audit.json for source preservation. RG3 was not executed.",
        "Repository verification is recorded separately in the Phase B restart plan.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", {
        "task": "B-C005REC-004AC", "revision": "metric_v2",
        "plan": "docs/exec-plans/active/PHASE_B_RESTART.md#5",
        "source_commit": SOURCE_COMMIT, "model_seed": 10, "init": "I03",
        "optimizer_updates": 0, "candidate_selection": False, "sealed": False,
        "discrete_em_tolerance": 0, "descriptive_tolerance": 1e-4,
        "endpoint": 8000, "data": "AC run_001 manifest, development-exposed only",
    })
    write(output / "system.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    })
    write(output / "config.json", {"batch_size": 128, "threads": torch.get_num_threads()})
    sources = list(SOURCE_DIR.glob("*.json")) + [SOURCE_DIR / "checkpoint_step8000.pt"]
    sources.extend(trajectory._training_state_path("I03", s) for s in (7500, 8000))
    sources.append(trajectory._checkpoint_path("I03", 8000))
    for task in ("z", "aa", "ab", "w"):
        sources.append(Path(f"runs/phase_b_b2_model_bundle_recovery/rec004{task}/run_001/"
                            "checkpoint_step8000.pt"))
    parent, raw = ibc._load_parent_manifest()
    sources.extend(Path(p) for p in _artifact_paths(raw))
    sources = list(dict.fromkeys(sources))
    before = {str(p): sha(p) for p in sources}
    write(output / "source_hashes.json", before)
    diff = subprocess.check_output(["git", "diff", "--", "src", "scripts"], encoding="utf-8")
    (output / "source_diff.patch").write_text(diff, encoding="utf-8")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in (
        "scripts/reanalyze_rec004ac_metrics.py",
        "src/apc/evaluation/mirror_attention_metrics.py",
        "src/apc/evaluation/mirror_parallel_score_residual_pilot.py",
        "src/apc/evaluation/mirror_kv_role_split_pilot.py",
    ):
        source = Path(name)
        (snapshot / source.name).write_bytes(source.read_bytes())
    try:
        with ExitStack() as stack:
            for optimizer in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
                stack.enter_context(patch.object(optimizer, "__init__",
                                                 side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")))
            core, bank, _ = ibc._reconstruct_parent_runtime(
                ibc.IncrementalBudgetCalibrationConfig(seed=10), parent
            )
            frozen_before = {"core": canonical_state_hash(core.model.state_dict()),
                             "bank": canonical_state_hash(bank.state_dict())}
            print("Restoring manifest-identical development inputs", flush=True)
            datasets, manifest = ac.prepare_rec004ac_datasets(10)
            if manifest != read(SOURCE_DIR / "fresh_validation_manifest.json"):
                write(output / "actual_manifest.json", manifest)
                raise RuntimeError("DATA_MANIFEST_MISMATCH")
            write(output / "data_manifest.json", manifest)
            write(output / "examples.json", {
                name: [{"input": list(ex.input_tokens), "target": list(ex.target_tokens)}
                       for ex in exs] for name, exs in datasets.items()
            })
            old_endpoint = read(SOURCE_DIR / "endpoint_metrics.json")
            state = torch.load(SOURCE_DIR / "checkpoint_step8000.pt", map_location="cpu",
                               weights_only=True)["primitive_state_dict"]
            model = new_model(core, state)
            model_hash = canonical_state_hash(model.state_dict())
            legacy = legacy_evaluator()
            endpoints: dict[str, Any] = {}
            comparison: dict[str, Any] = {}
            forward = ac.evaluate_parallel_score_residual_forward_with_stages
            for name in (*ac.REC004AC_CONTINUITY_SPLITS, ac.REC004AC_FRESH_LENGTH10_CONFIRMATION):
                exs = datasets[name]
                traces: list[list[str]] = [[], []]

                def capture(which: int, traces: list[list[str]] = traces) -> Any:
                    def observer(*args: Any, **kwargs: Any) -> Any:
                        stages = forward(*args, **kwargs)
                        predictions = stages["final_token_logits"].argmax(-1).cpu().numpy()
                        traces[which].append(hashlib.sha256(predictions.tobytes()).hexdigest())
                        return stages
                    return observer

                # Legacy function has its own globals copied from ac.
                legacy.__globals__["evaluate_parallel_score_residual_forward_with_stages"] = (
                    capture(0)
                )
                old = legacy(core, model, exs)
                with patch.object(ac, "evaluate_parallel_score_residual_forward_with_stages",
                                  capture(1)):
                    corrected = ac.evaluate_length10_metrics(core, model, exs)
                if traces[0] != traces[1]:
                    raise RuntimeError("PREDICTION_PARITY_FAILED")
                expected = (old_endpoint["fresh_length10_confirmation"]
                            if name == ac.REC004AC_FRESH_LENGTH10_CONFIRMATION
                            else old_endpoint["continuity_splits"][name])
                assert_em_equal(corrected, expected)
                for key in ("j0_p4_score_margin_median", "j0_p4_score_prob_median",
                            "j0_p4_correct_key_rank_mean"):
                    if abs(old[key] - expected[key]) > 1e-4:
                        raise RuntimeError(f"HISTORICAL_OBSERVER_PARITY_FAILED: {name}/{key}")
                endpoints[name] = corrected
                comparison[name] = {"legacy": old, "corrected": corrected,
                                    "prediction_hashes_identical": True,
                                    "prediction_batch_hashes": traces[1]}
                write(output / "endpoint_metrics_partial.json", endpoints)
                write(output / "metric_erratum_partial.json", comparison)
                print(name, corrected["j0_sequence_em"], corrected["j0_p4_score_margin_median"],
                      flush=True)
            val = datasets[ac.REC004AC_FRESH_NORMAL_VALIDATION]
            conf = datasets[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION]
            normal = ac.evaluate_variable_length_metrics(core, model, val)
            assert_em_equal(normal, old_endpoint["fresh_validation_normal"])
            endpoints["normal"] = normal
            start = torch.load(trajectory._training_state_path("I03", 7500),
                               map_location="cpu", weights_only=True)["primitive_state_dict"]
            if canonical_state_hash(start) != read(SOURCE_DIR / "source_manifest.json")[
                "source_canonical_primitive_state_hash"
            ]:
                raise RuntimeError("SOURCE_7500_MISMATCH")
            initial = new_model(core, start, initial=True)
            reference = trajectory._new_primitive_from_state(core, start)
            initial_metrics = ac.evaluate_length10_metrics(core, initial, conf)
            reference_metrics = hard_freeze.evaluate_length10_dataset_metrics(core, reference, conf)
            assert_em_equal(initial_metrics, reference_metrics)
            endpoints["initial_7500"] = initial_metrics
            write(output / "initial_parity.json", initial_parity(core, start, conf))
            comparators = {}
            for name, loader, evaluator in (
                ("hard_freeze_rec004w", ac.get_step8000_rec004w_model, hard_freeze),
                ("role_split_rec004z", ac.get_step8000_rec004z_model, ac.rec004z),
                ("pre_v_rec004aa", ac.get_step8000_rec004aa_model, ac.rec004aa),
                ("post_attn_rec004ab", ac.get_step8000_rec004ab_model, ac.rec004ab),
            ):
                other = loader(core, core.device)
                if name == "hard_freeze_rec004w":
                    vm = evaluator.evaluate_variable_length_dataset(core, other, val)
                    cm = evaluator.evaluate_length10_dataset_metrics(core, other, conf)
                elif name == "role_split_rec004z":
                    vm = evaluator.evaluate_variable_length_metrics_role_split(core, other, val)
                    cm = evaluator.evaluate_length10_metrics_role_split(core, other, conf)
                else:
                    vm = evaluator.evaluate_variable_length_metrics(core, other, val)
                    cm = evaluator.evaluate_length10_metrics(core, other, conf)
                assert_em_equal(vm, old_endpoint["comparators"][name]["normal"])
                assert_em_equal(cm, old_endpoint["comparators"][name]["length10"])
                comparators[name] = {"normal": vm, "length10": cm}
            hist_state = torch.load(trajectory._checkpoint_path("I03", 8000),
                                    map_location="cpu", weights_only=True)
            historical = trajectory._new_primitive_from_state(core, hist_state)
            hv = hard_freeze.evaluate_variable_length_dataset(core, historical, val)
            hc = hard_freeze.evaluate_length10_dataset_metrics(core, historical, conf)
            assert_em_equal(hv, old_endpoint["comparators"]["historical"]["normal"])
            assert_em_equal(hc, old_endpoint["comparators"]["historical"]["length10"])
            comparators["historical"] = {"normal": hv, "length10": hc}
            write(output / "comparators.json", comparators)
            write(output / "endpoint_metrics.json", endpoints)
            write(output / "metric_erratum.json", comparison)
            counts = cvof_parameter_accounting(model)
            write(output / "cost_accounting.json", {
                **counts, "scope": "target primitive; no whole-system capacity claim",
                "reanalysis_wall_seconds": time.perf_counter() - started,
                "historical_training_cost": read(SOURCE_DIR / "cost_accounting.json"),
            })
            if model_hash != canonical_state_hash(model.state_dict()):
                raise RuntimeError("TARGET_MUTATED")
            frozen_after = {"core": canonical_state_hash(core.model.state_dict()),
                            "bank": canonical_state_hash(bank.state_dict())}
            if frozen_before != frozen_after:
                raise RuntimeError("PARENT_MUTATED")
            write(output / "freeze_audit.json", {"before": frozen_before, "after": frozen_after})
            result = {"status": "PASS", "task": "B-C005REC-004AC", "revision": "metric_v2",
                      "historical_em_parity": True, "prediction_parity": True,
                      "normal_j0_em": normal["overall_j0_sequence_em"],
                      "length10": endpoints[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION],
                      "rg3": "NOT_EXECUTED", "rec005_eligible": False,
                      "scientific_plasticity_gate": "FAIL_PRESERVED"}
            write(output / "summary.json", result)
            write_report(output, result)
            return result
    except Exception as exc:
        write(output / "summary.json", {"status": "FAIL", "reason": str(exc),
                                        "rg3": "NOT_EXECUTED", "rec005_eligible": False})
        raise
    finally:
        after = {str(p): sha(p) for p in sources}
        write(output / "side_effect_audit.json", {"source_hashes_unchanged": before == after,
              "optimizer_updates": 0, "candidate_selected": None,
              "child_bundle": None, "sealed_evaluation": False})
        if before != after:
            write(output / "summary.json", {"status": "FAIL",
                  "reason": "HISTORICAL_FILES_MUTATED", "rg3": "NOT_EXECUTED",
                  "rec005_eligible": False})
            raise RuntimeError("HISTORICAL_FILES_MUTATED")


def _artifact_paths(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [v for k, v in value.items() if k == "file_path" and isinstance(v, str)] + [
            p for v in value.values() for p in _artifact_paths(v)
        ]
    if isinstance(value, list):
        return [p for v in value for p in _artifact_paths(v)]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)
