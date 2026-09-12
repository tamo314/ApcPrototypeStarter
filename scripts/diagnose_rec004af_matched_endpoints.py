"""REC-004AF: fixed-endpoint QK versus position-routing causal diagnosis.

This is an evaluation-only diagnostic.  It never creates an optimizer, changes a
model tensor, or supplies oracle routing information to the J0 runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import localize_mirror_failure_modes as rec004ae  # noqa: E402
import reanalyze_rec004ac_metrics as reanalysis  # noqa: E402
from apc.evaluation import mirror_parallel_score_residual_pilot as ac  # noqa: E402
from apc.evaluation.mirror_position_initialization_diagnostic import (  # noqa: E402
    mirror_halves_position_map,
)
from apc.utils.model_bundle import canonical_state_hash  # noqa: E402

INPUT = Path("runs/phase_b_restart/rec004ac_metric_v2/run_003")
AE_INPUT = Path("runs/phase_b_restart/rec004ae/run_004")
CHECKPOINT = reanalysis.SOURCE_DIR / "checkpoint_step8000.pt"
CONTROL_COUNT = 4
PRIMARY_STRATA = ("normal_validation_length10", "length10_confirmation")
ROUTING_IMPROVEMENT_MIN = 0.05
MARGIN_IMPROVEMENT_MIN = 0.25
RECOVERY_MIN = 0.05


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint_identity(dataset: str, index: int, tokens: tuple[int, ...]) -> str:
    payload = json.dumps(
        {"dataset": dataset, "index": index, "input_tokens": list(tokens)},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_preregistered_controls(
    dataset: str, input_tokens: list[tuple[int, ...]]
) -> tuple[dict[tuple[int, int], tuple[int, ...]], list[dict[str, Any]]]:
    """Build the control table from non-oracle endpoint identity fields only.

    Targets, baseline predictions, attention scores, oracle maps, and intervention
    outcomes are deliberately absent from this function's inputs.
    """
    pools: dict[tuple[int, int, int], list[int]] = {}
    identities = {
        index: _endpoint_identity(dataset, index, tokens)
        for index, tokens in enumerate(input_tokens)
    }
    for index, tokens in enumerate(input_tokens):
        length = len(tokens)
        token_sum_mod4 = sum(tokens) % 4
        for output_position in range(length):
            pools.setdefault((length, output_position, token_sum_mod4), []).append(index)
    controls: dict[tuple[int, int], tuple[int, ...]] = {}
    table: list[dict[str, Any]] = []
    for (length, output_position, token_sum_mod4), members in sorted(pools.items()):
        ordered = sorted(members, key=lambda item: identities[item])
        if len(ordered) < CONTROL_COUNT + 1:
            raise RuntimeError(
                "MATCH_POOL_TOO_SMALL: "
                f"dataset={dataset}, length={length}, output_position={output_position}, "
                f"input_token_sum_mod4={token_sum_mod4}, pool={len(ordered)}"
            )
        for offset, index in enumerate(ordered):
            selected = tuple(
                ordered[(offset + rank + 1) % len(ordered)] for rank in range(CONTROL_COUNT)
            )
            controls[(index, output_position)] = selected
            table.append(
                {
                    "dataset": dataset,
                    "relation": "MIRROR_HALVES",
                    "example_index": index,
                    "endpoint_identity_sha256": identities[index],
                    "sequence_length": length,
                    "output_position": output_position,
                    "input_token_sum_mod4": token_sum_mod4,
                    "control_example_indices": list(selected),
                    "control_endpoint_identity_sha256": [identities[item] for item in selected],
                }
            )
    return controls, table


def _readout_from_attention(
    model: Any, stages: dict[str, Any], attention: torch.Tensor
) -> torch.Tensor:
    """Use the unchanged saved value, output, FFN, and readout path."""
    return rec004ae._readout_from_attention(model, stages, attention)


def compose_counterfactual(
    stages: dict[str, Any],
    controls: torch.Tensor,
    failed_sequences: torch.Tensor,
    component: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace one saved score component only on baseline-failing sequences."""
    qk = stages["s_qk"].clone()
    position_bias = (stages["s_base"] - stages["s_qk"]).clone()
    residual = stages["delta_s"][:, None].expand_as(qk)
    batch, _, output_length, _ = qk.shape
    selected = torch.nonzero(failed_sequences, as_tuple=False).flatten()
    if component not in {"qk", "position"}:
        raise ValueError(f"unknown component {component!r}")
    for output_position in range(output_length):
        control_rows = controls[:, output_position]
        if component == "qk":
            qk[selected, :, output_position, :] = stages["s_qk"][
                control_rows[selected], :, output_position, :
            ]
        else:
            position_bias[selected, :, output_position, :] = position_bias[
                control_rows[selected], :, output_position, :
            ]
    scores = qk + position_bias + residual
    scores = scores.masked_fill(torch.isneginf(stages["score_logits"]), -torch.inf)
    return scores, qk, position_bias


def _correct_key_tensors(length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    correct = torch.tensor(mirror_halves_position_map(length), device=device)
    wrong_mask = torch.ones(length, length, dtype=torch.bool, device=device)
    wrong_mask[torch.arange(length, device=device), correct] = False
    return correct, wrong_mask


def collect_metrics(
    logits: torch.Tensor,
    scores: torch.Tensor,
    qk: torch.Tensor,
    position_bias: torch.Tensor,
    target: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_top1: torch.Tensor,
) -> dict[str, float | int]:
    """Calculate output and evaluation-only correct-key routing statistics."""
    batch, _, output_length, _ = scores.shape
    correct_key, wrong_mask = _correct_key_tensors(output_length, scores.device)
    attention = scores.softmax(-1)
    top1 = attention.argmax(-1)
    correct = correct_key.view(1, 1, output_length).expand_as(top1)
    correct_score = scores.gather(-1, correct[..., None]).squeeze(-1)
    wrong_score = scores.masked_fill(~wrong_mask[None, None], -torch.inf).amax(-1)
    qk_correct = qk.gather(-1, correct[..., None]).squeeze(-1)
    qk_wrong = qk.gather(-1, top1[..., None]).squeeze(-1)
    position_correct = position_bias.gather(-1, correct[..., None]).squeeze(-1)
    position_wrong = position_bias.gather(-1, top1[..., None]).squeeze(-1)
    prediction = logits.argmax(-1)
    baseline_prediction = baseline_logits.argmax(-1)
    baseline_wrong = ~baseline_prediction.eq(target)
    now_correct = prediction.eq(target)
    recovered = baseline_wrong & now_correct
    worsened = ~baseline_wrong & ~now_correct
    sequence_correct = now_correct.all(-1)
    baseline_sequence_correct = baseline_prediction.eq(target).all(-1)
    changed_route = top1.ne(baseline_top1)
    direct_errors = int(baseline_wrong.sum())
    return {
        "n_examples": batch,
        "sequence_em": int(sequence_correct.sum()) / batch,
        "token_accuracy": int(now_correct.sum()) / now_correct.numel(),
        "direct_error_token_recovery_rate": float(recovered.sum() / direct_errors)
        if direct_errors
        else 0.0,
        "top1_correct_key_routing_rate": int(top1.eq(correct).sum()) / top1.numel(),
        "correct_key_probability": float(
            attention.gather(-1, correct[..., None]).squeeze(-1).mean()
        ),
        "correct_key_margin": float((correct_score - wrong_score).mean()),
        "qk_contribution_to_correct_key_margin": float((qk_correct - qk_wrong).mean()),
        "position_contribution_to_correct_key_margin": float(
            (position_correct - position_wrong).mean()
        ),
        "routing_changed_sample_count": int(changed_route.any(dim=(1, 2)).sum()),
        "routing_changed_token_head_count": int(changed_route.sum()),
        "wrong_to_correct_token_count": int(recovered.sum()),
        "correct_to_wrong_token_count": int(worsened.sum()),
        "wrong_to_correct_sequence_count": int(
            (~baseline_sequence_correct & sequence_correct).sum()
        ),
        "correct_to_wrong_sequence_count": int(
            (baseline_sequence_correct & ~sequence_correct).sum()
        ),
        "baseline_direct_error_token_count": direct_errors,
    }


def _oracle_metrics(
    model: Any, content: torch.Tensor, length: int, target: torch.Tensor, direct: torch.Tensor
) -> dict[str, float]:
    """Evaluation-only oracle control; no oracle value reaches a counterfactual."""
    oracle = ac.evaluate_parallel_score_residual_forward_with_stages(
        model, content, [length] * len(target), [length] * len(target), oracle_attention=True
    )["final_token_logits"]
    direct_ok = direct.argmax(-1).eq(target)
    oracle_ok = oracle.argmax(-1).eq(target)
    direct_errors = int((~direct_ok).sum())
    return {
        "oracle_sequence_em": float(oracle_ok.all(-1).float().mean()),
        "oracle_persistent_error_rate": float((~direct_ok & ~oracle_ok).float().mean()),
        "direct_error_oracle_recovery_rate": float((~direct_ok & oracle_ok).sum() / direct_errors)
        if direct_errors
        else 1.0,
    }


def _append_metrics(destination: dict[str, Any], label: str, value: dict[str, float | int]) -> None:
    destination[label] = value


def _selection_evidence(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the preregistered all-control, all-primary-stratum rule."""
    support: dict[str, dict[str, Any]] = {}
    for component, other in (("qk", "position"), ("position", "qk")):
        checks: list[dict[str, Any]] = []
        for stratum in PRIMARY_STRATA:
            baseline = metrics[stratum]["baseline"]
            for rank in range(CONTROL_COUNT):
                current = metrics[stratum][component][str(rank)]
                comparator = metrics[stratum][other][str(rank)]
                checks.append(
                    {
                        "stratum": stratum,
                        "control_rank": rank,
                        "routing_improvement": current["top1_correct_key_routing_rate"]
                        - baseline["top1_correct_key_routing_rate"],
                        "margin_improvement": current["correct_key_margin"]
                        - baseline["correct_key_margin"],
                        "direct_error_recovery": current["direct_error_token_recovery_rate"],
                        "routing_advantage_over_other": current["top1_correct_key_routing_rate"]
                        - comparator["top1_correct_key_routing_rate"],
                        "margin_advantage_over_other": current["correct_key_margin"]
                        - comparator["correct_key_margin"],
                        "recovery_advantage_over_other": current["direct_error_token_recovery_rate"]
                        - comparator["direct_error_token_recovery_rate"],
                    }
                )
        passed = all(
            row["routing_improvement"] >= ROUTING_IMPROVEMENT_MIN
            and row["margin_improvement"] >= MARGIN_IMPROVEMENT_MIN
            and row["direct_error_recovery"] >= RECOVERY_MIN
            and row["routing_advantage_over_other"] >= ROUTING_IMPROVEMENT_MIN
            and row["margin_advantage_over_other"] >= MARGIN_IMPROVEMENT_MIN
            and row["recovery_advantage_over_other"] >= RECOVERY_MIN
            for row in checks
        )
        support[component] = {"supported": passed, "checks": checks}
    if support["qk"]["supported"] and not support["position"]["supported"]:
        return "QK_CONTENT_TARGET_SUPPORTED", support
    if support["position"]["supported"] and not support["qk"]["supported"]:
        return "POSITION_ROUTING_TARGET_SUPPORTED", support
    return "INSUFFICIENT_EVIDENCE_STOP", support


def _report(output: Path, decision: str, metrics: dict[str, Any]) -> None:
    lines = [
        "# REC-004AF matched-endpoint causal diagnosis",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Stratum | Condition | Sequence EM | Token accuracy | Direct-error recovery | "
        "Top-1 correct-key routing | Correct-key margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for stratum, values in metrics.items():
        conditions: list[tuple[str, dict[str, Any]]] = [("baseline", values["baseline"])]
        for component in ("qk", "position"):
            for rank, value in values[component].items():
                conditions.append((f"{component}_counterfactual/control_{rank}", value))
        for label, value in conditions:
            lines.append(
                f"| {stratum} | {label} | {value['sequence_em']:.6f} | "
                f"{value['token_accuracy']:.6f} | "
                f"{value['direct_error_token_recovery_rate']:.6f} | "
                f"{value['top1_correct_key_routing_rate']:.6f} | "
                f"{value['correct_key_margin']:.6f} |"
            )
    lines.extend(
        [
            "",
            "Controls were fixed before model forward from split, relation, length,",
            "output position, and input-token-sum modulo 4 only. No target, correct-key index,",
            "oracle routing, or intervention outcome was used by matching or the J0 runtime.",
            "",
            "All components except the named saved score row were held fixed; value path, FFN, and",
            "readout are the immutable endpoint's path. No optimizer was constructed or stepped.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [
        INPUT / "examples.json",
        INPUT / "data_manifest.json",
        INPUT / "summary.json",
        AE_INPUT / "failure_modes.json",
        AE_INPUT / "summary.json",
        CHECKPOINT,
        Path(__file__),
    ]
    before = {str(path): sha(path) for path in sources}
    write(output / "source_hashes.json", before)
    write(
        output / "protocol.json",
        {
            "task": "B-C005REC-004AF",
            "plan": "docs/exec-plans/active/PHASE_B_RESTART.md#7a",
            "endpoint": "REC-004AC I03@8000 immutable checkpoint",
            "development_only": True,
            "sealed": False,
            "optimizer_updates": 0,
            "new_parameters": 0,
            "candidate_selection": False,
            "rg3": "NOT_EXECUTED",
            "score_partition": "S = S_QK + S_position_bias + S_residual",
            "counterfactuals": {
                "qk": "replace saved S_QK only; retain target position bias and residual",
                "position": "replace saved S_position_bias only; retain target QK and residual",
            },
            "matching": {
                "control_pool": "all saved examples in the same split; no success filtering",
                "strata": [
                    "relation=MIRROR_HALVES",
                    "sequence_length",
                    "output_position",
                    "input_token_sum_mod4",
                ],
                "endpoint_order": "sha256(split, example index, input token tuple)",
                "control_count": CONTROL_COUNT,
                "selection": "next four circular endpoints excluding self",
                "uses_target_or_oracle": False,
            },
            "primary_strata": list(PRIMARY_STRATA),
            "selection_thresholds": {
                "routing_improvement_min": ROUTING_IMPROVEMENT_MIN,
                "margin_improvement_min": MARGIN_IMPROVEMENT_MIN,
                "direct_error_recovery_min": RECOVERY_MIN,
                "must_hold_for_each_control_rank": True,
            },
            "oracle_is_evaluation_only": True,
        },
    )
    write(
        output / "system.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    )
    Path(output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    raw = read(INPUT / "examples.json")
    manifest = read(INPUT / "data_manifest.json")
    if manifest != read(reanalysis.SOURCE_DIR / "fresh_validation_manifest.json"):
        raise RuntimeError("DATA_MANIFEST_MISMATCH")
    datasets = {
        name: [
            SimpleNamespace(input_tokens=tuple(row["input"]), target_tokens=tuple(row["target"]))
            for row in rows
        ]
        for name, rows in raw.items()
    }
    control_maps: dict[str, dict[tuple[int, int], tuple[int, ...]]] = {}
    matching_rows: list[dict[str, Any]] = []
    for name, examples in datasets.items():
        controls, rows = build_preregistered_controls(
            name, [example.input_tokens for example in examples]
        )
        control_maps[name] = controls
        matching_rows.extend(rows)
    write(output / "preregistered_matching.json", matching_rows)
    write(
        output / "data_manifest.json",
        {
            "input_sha256": sha(INPUT / "examples.json"),
            "source_manifest_sha256": sha(INPUT / "data_manifest.json"),
            "source_manifest_matches": True,
            "counts": {name: len(examples) for name, examples in datasets.items()},
            "matching_rows": len(matching_rows),
            "sealed": False,
        },
    )
    metrics: dict[str, Any] = {
        "normal_validation": {"baseline": None, "qk": {}, "position": {}},
        "normal_validation_length10": {"baseline": None, "qk": {}, "position": {}},
        "length10_confirmation": {"baseline": None, "qk": {}, "position": {}},
    }
    oracle_audit: dict[str, Any] = {}
    try:
        with ExitStack() as stack, torch.no_grad():
            for optimizer in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
                stack.enter_context(
                    patch.object(
                        optimizer, "__init__", side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")
                    )
                )
            parent, _ = reanalysis.ibc._load_parent_manifest()
            core, bank, _ = reanalysis.ibc._reconstruct_parent_runtime(
                reanalysis.ibc.IncrementalBudgetCalibrationConfig(seed=10), parent
            )
            model = reanalysis.new_model(
                core,
                torch.load(CHECKPOINT, weights_only=True, map_location="cpu")[
                    "primitive_state_dict"
                ],
            )
            core.model.eval().requires_grad_(False)
            bank.eval().requires_grad_(False)
            model.eval().requires_grad_(False)
            frozen_before = [
                canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)
            ]
            selected_sets = {
                "normal_validation": datasets[ac.REC004AC_FRESH_NORMAL_VALIDATION],
                "length10_confirmation": datasets[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION],
            }
            for public_name, examples in selected_sets.items():
                source_name = (
                    ac.REC004AC_FRESH_NORMAL_VALIDATION
                    if public_name == "normal_validation"
                    else ac.REC004AC_FRESH_LENGTH10_CONFIRMATION
                )
                grouped: dict[int, list[tuple[int, Any]]] = {}
                for index, example in enumerate(examples):
                    grouped.setdefault(len(example.input_tokens), []).append((index, example))
                aggregate: dict[str, list[dict[str, float | int]]] = {
                    "baseline": [],
                    **{f"qk/{rank}": [] for rank in range(CONTROL_COUNT)},
                    **{f"position/{rank}": [] for rank in range(CONTROL_COUNT)},
                }
                primary_aggregate: dict[str, list[dict[str, float | int]]] = {
                    key: [] for key in aggregate
                }
                for length, indexed_examples in sorted(grouped.items()):
                    global_indices = [index for index, _ in indexed_examples]
                    subset = [example for _, example in indexed_examples]
                    local_index = {
                        global_index: row for row, global_index in enumerate(global_indices)
                    }
                    stage_chunks: list[dict[str, Any]] = []
                    direct_chunks: list[torch.Tensor] = []
                    content_chunks: list[torch.Tensor] = []
                    # REC-004AE's baseline used this exact batch partition. Preserve it
                    # before concatenating endpoint rows for cross-example replacement.
                    for start in range(0, len(subset), 128):
                        chunk = subset[start : start + 128]
                        batch = ac.collate_content_only_batch(chunk, core.tokens, core.device)
                        chunk_content = core.model.encode(batch)[:, 1 : 1 + length]
                        stage_chunks.append(
                            ac.evaluate_parallel_score_residual_forward_with_stages(
                                model,
                                chunk_content,
                                [length] * len(chunk),
                                [length] * len(chunk),
                            )
                        )
                        direct_chunks.append(
                            model(chunk_content, [length] * len(chunk), [length] * len(chunk), None)
                        )
                        content_chunks.append(chunk_content)
                    stages = {
                        key: torch.cat([stage[key] for stage in stage_chunks], dim=0)
                        for key in stage_chunks[0]
                    }
                    direct = torch.cat(direct_chunks, dim=0)
                    content = torch.cat(content_chunks, dim=0)
                    if not torch.equal(direct.argmax(-1), stages["final_token_logits"].argmax(-1)):
                        raise RuntimeError("BASELINE_FORWARD_PARITY_FAILED")
                    target = torch.tensor(
                        [item.target_tokens for item in subset], device=core.device
                    )
                    baseline_top1 = stages["score_logits"].softmax(-1).argmax(-1)
                    baseline_metric = collect_metrics(
                        direct,
                        stages["score_logits"],
                        stages["s_qk"],
                        stages["s_base"] - stages["s_qk"],
                        target,
                        direct,
                        baseline_top1,
                    )
                    aggregate["baseline"].append(baseline_metric)
                    failed_sequences = ~direct.argmax(-1).eq(target).all(-1)
                    for rank in range(CONTROL_COUNT):
                        control_rows = torch.tensor(
                            [
                                [
                                    local_index[
                                        control_maps[source_name][(global_index, output_position)][
                                            rank
                                        ]
                                    ]
                                    for output_position in range(length)
                                ]
                                for global_index in global_indices
                            ],
                            device=core.device,
                            dtype=torch.long,
                        )
                        for component in ("qk", "position"):
                            score, qk, position_bias = compose_counterfactual(
                                stages, control_rows, failed_sequences, component
                            )
                            logits = _readout_from_attention(model, stages, score.softmax(-1))
                            metric = collect_metrics(
                                logits,
                                score,
                                qk,
                                position_bias,
                                target,
                                direct,
                                baseline_top1,
                            )
                            aggregate[f"{component}/{rank}"].append(metric)
                    if length == 10:
                        for key, values in aggregate.items():
                            if values:
                                primary_aggregate[key].append(values[-1])
                        oracle_audit[public_name] = _oracle_metrics(
                            model, content, length, target, direct
                        )

                def merge(values: list[dict[str, float | int]]) -> dict[str, float | int]:
                    total_examples = sum(int(value["n_examples"]) for value in values)
                    result: dict[str, float | int] = {"n_examples": total_examples}
                    additive = {
                        "routing_changed_sample_count",
                        "routing_changed_token_head_count",
                        "wrong_to_correct_token_count",
                        "correct_to_wrong_token_count",
                        "wrong_to_correct_sequence_count",
                        "correct_to_wrong_sequence_count",
                        "baseline_direct_error_token_count",
                    }
                    weighted = (
                        set(values[0])
                        - additive
                        - {"n_examples", "direct_error_token_recovery_rate"}
                    )
                    for key in additive:
                        result[key] = sum(int(value[key]) for value in values)
                    for key in weighted:
                        result[key] = (
                            sum(float(value[key]) * int(value["n_examples"]) for value in values)
                            / total_examples
                        )
                    result["direct_error_token_recovery_rate"] = (
                        int(result["wrong_to_correct_token_count"])
                        / int(result["baseline_direct_error_token_count"])
                        if int(result["baseline_direct_error_token_count"])
                        else 0.0
                    )
                    return result

                metrics[public_name]["baseline"] = merge(aggregate["baseline"])
                for component in ("qk", "position"):
                    for rank in range(CONTROL_COUNT):
                        metrics[public_name][component][str(rank)] = merge(
                            aggregate[f"{component}/{rank}"]
                        )
                if public_name == "normal_validation":
                    metrics["normal_validation_length10"]["baseline"] = merge(
                        primary_aggregate["baseline"]
                    )
                    for component in ("qk", "position"):
                        for rank in range(CONTROL_COUNT):
                            metrics["normal_validation_length10"][component][str(rank)] = merge(
                                primary_aggregate[f"{component}/{rank}"]
                            )
            ae_modes = read(AE_INPUT / "failure_modes.json")
            expected_normal = ae_modes[ac.REC004AC_FRESH_NORMAL_VALIDATION]["sequence_em"][
                "baseline"
            ]
            expected_confirmation = ae_modes[ac.REC004AC_FRESH_LENGTH10_CONFIRMATION][
                "sequence_em"
            ]["baseline"]
            if (
                abs(metrics["normal_validation"]["baseline"]["sequence_em"] - expected_normal)
                > 1e-12
            ):
                raise RuntimeError(
                    "NORMAL_BASELINE_REPRODUCTION_FAILED: "
                    f"actual={metrics['normal_validation']['baseline']['sequence_em']}, "
                    f"expected={expected_normal}"
                )
            if (
                abs(
                    metrics["length10_confirmation"]["baseline"]["sequence_em"]
                    - expected_confirmation
                )
                > 1e-12
            ):
                raise RuntimeError("LENGTH10_BASELINE_REPRODUCTION_FAILED")
            if frozen_before != [
                canonical_state_hash(item.state_dict()) for item in (core.model, bank, model)
            ]:
                raise RuntimeError("FROZEN_STATE_MUTATED")
            decision, selection = _selection_evidence(metrics)
            write(output / "metrics.json", metrics)
            write(output / "oracle_evaluation_audit.json", oracle_audit)
            write(output / "selection_evidence.json", selection)
            write(
                output / "freeze_audit.json",
                {
                    "state_hashes_before": frozen_before,
                    "state_hashes_after": frozen_before,
                    "all_frozen": True,
                },
            )
            summary = {
                "execution_status": "PASS",
                "decision": decision,
                "baseline_reproduced": True,
                "source_hashes_reproduced": True,
                "matching_uses_target_or_oracle": False,
                "optimizer_updates": 0,
                "new_parameters": 0,
                "candidate_selected": None,
                "bundle_write": False,
                "rg3": "NOT_EXECUTED",
                "rec005_eligible": False,
                "g1": "NOT_CLEARED",
                "g4": "NOT_CLEARED",
                "wall_seconds": time.perf_counter() - started,
            }
            write(output / "summary.json", summary)
            _report(output, decision, metrics)
    except Exception as exc:
        write(
            output / "summary.json",
            {
                "execution_status": "FAIL",
                "decision": "INSUFFICIENT_EVIDENCE_STOP",
                "reason": repr(exc),
                "rg3": "NOT_EXECUTED",
                "rec005_eligible": False,
            },
        )
        raise
    finally:
        after = {str(path): sha(path) for path in sources}
        write(
            output / "side_effect_audit.json",
            {
                "sources_unchanged": before == after,
                "optimizer_updates": 0,
                "candidate_selected": None,
                "sealed_data_read": False,
            },
        )
        if before != after:
            raise RuntimeError("SOURCE_ARTIFACT_MUTATED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)
