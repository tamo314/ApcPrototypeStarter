# ruff: noqa: E501
"""B-C005REC-004R: read-only shared-score gradient aggregation audit.

No optimizer is constructed or stepped here.  Every gradient is obtained with
``torch.autograd.grad`` from an in-memory checkpoint copy.  The decomposition
uses the exact live token-CE denominator of each regenerated training batch.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_attention_score_credit_assignment_audit as rec004q
from apc.evaluation import mirror_budget_extension as rec004g
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.utils import model_bundle as mb

REC004R_TASK_ID: Final = "B-C005REC-004R"
REC004R_OUTPUT_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004r/run_001")
REC004R_STEPS: Final = tuple(6001 + (k * 6000) // 128 for k in range(128))
REC004R_ATOL: Final = 1e-6
REC004R_RTOL: Final = 1e-4
REC004R_PER_EXAMPLE_COUNT: Final = 128
REC004R_REFERENCE_PER_LENGTH: Final = 8


@dataclass(frozen=True)
class MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig:
    output_dir: Path = REC004R_OUTPUT_DIR
    seed: int = 10


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _digest_example(example: Any) -> str:
    return hashlib.sha256(
        json.dumps([example.input_tokens, example.target_tokens]).encode()
    ).hexdigest()


def _census(seed: int) -> tuple[dict[int, list[Any]], dict[str, Any]]:
    if len(set(REC004R_STEPS)) != 128:
        raise RuntimeError("training_gradient_census_v1 step collision")
    batches: dict[int, list[Any]] = {}
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for step in REC004R_STEPS:
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            rec004g.REC004G_TARGET_OPERATION,
            vocab_size=rec004g.REC004G_VOCAB_SIZE,
            sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        batches[step] = examples
        digest.update(str(step).encode())
        for example in examples:
            digest.update(json.dumps([example.input_tokens, example.target_tokens]).encode())
            counts[str(len(example.input_tokens))] = (
                counts.get(str(len(example.input_tokens)), 0) + 1
            )
    full = rec004q.rec004p._training_manifest(seed)
    return batches, {
        "namespace": "training_gradient_census_v1",
        "formula": "step_k = 6001 + floor(k * 6000 / 128), k=0..127",
        "steps": list(REC004R_STEPS),
        "unique_step_count": len(set(REC004R_STEPS)),
        "examples_by_input_length": counts,
        "sampled_input_target_digest_sha256": digest.hexdigest(),
        "full_stream_input_target_digest_sha256": full["input_target_digest_sha256"],
        "full_stream_contract": "REC-004O/P deterministic MIRROR_HALVES stream steps 6001--12000",
        "stream_regeneration_verified": full["step_range"] == [6001, 12000],
        "model_output_used_to_select_steps": False,
    }


def _score_schema(primitive: Any) -> dict[str, tuple[list[str], slice | None]]:
    return {
        key: value for key, value in rec004q._groups(primitive).items() if key != "CONTENT_PREP"
    }


def _selected_params(primitive: Any) -> tuple[dict[str, torch.nn.Parameter], list[str]]:
    named = dict(primitive.named_parameters())
    names = [
        "cross_attn.in_proj_weight",
        "cross_attn.in_proj_bias",
        "position_bias_hidden.weight",
        "position_bias_hidden.bias",
        "position_bias_out.weight",
    ]
    return named, names


def _gradient(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return {
        str(i): value.detach() if value is not None else torch.zeros_like(param)
        for i, (value, param) in enumerate(zip(values, params, strict=True))
    }


def _batched_gradient_vectors(
    losses: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    names: list[str],
    schema: dict[str, tuple[list[str], slice | None]],
) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Get each scalar-loss VJP in one autograd call, without changing its weight."""
    output = torch.stack(losses)
    gradients = torch.autograd.grad(
        output,
        params,
        grad_outputs=torch.eye(len(losses), device=output.device, dtype=output.dtype),
        is_grads_batched=True,
        retain_graph=True,
        allow_unused=True,
    )
    result: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for index in range(len(losses)):
        by_index = {
            str(param_index): (
                value[index].detach() if value is not None else torch.zeros_like(param)
            )
            for param_index, (value, param) in enumerate(zip(gradients, params, strict=True))
        }
        result.append(_vector(by_index, names, schema))
    return result


def _vector(
    grad: dict[str, torch.Tensor],
    names: list[str],
    schema: dict[str, tuple[list[str], slice | None]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    by_name = {name: grad[str(index)] for index, name in enumerate(names)}
    groups: dict[str, torch.Tensor] = {}
    for group, (group_names, rows) in schema.items():
        groups[group] = torch.cat(
            [
                (by_name[name][rows] if rows is not None else by_name[name]).reshape(-1)
                for name in group_names
            ]
        )
    return torch.cat(
        [groups[key] for key in ("Q_rows", "K_rows", "position_bias_hidden", "position_bias_out")]
    ), groups


def _metrics(target: torch.Tensor, other: torch.Tensor) -> dict[str, float | None]:
    dot = float(torch.dot(target, other).item())
    denominator = torch.linalg.vector_norm(target) * torch.linalg.vector_norm(other)
    cosine = (
        None
        if float(denominator.item()) == 0.0
        else float((torch.dot(target, other) / denominator).item())
    )
    return {"dot": dot, "cosine": cosine}


def _add(destination: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    for name, value in source.items():
        destination[name] += value.cpu()


def _empty_vectors(primitive: Any) -> dict[str, torch.Tensor]:
    schema = _score_schema(primitive)
    named, names = _selected_params(primitive)
    zero = {str(index): torch.zeros_like(named[name]) for index, name in enumerate(names)}
    all_score, groups = _vector(zero, names, schema)
    return {"ALL_SCORE": all_score.cpu(), **{name: value.cpu() for name, value in groups.items()}}


def _loss_parts(
    primitive: Any, content: torch.Tensor, lengths: list[int], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
    logits = primitive(content, lengths, lengths, None)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(labels)
    valid = labels != IGNORE_INDEX
    denominator = valid.sum()
    total = losses[valid].sum() / denominator
    parts: dict[tuple[int, int], torch.Tensor] = {}
    for length in sorted(set(lengths)):
        for position in range(length):
            row = torch.tensor([value == length for value in lengths], device=labels.device)
            mask = row[:, None] & (
                torch.arange(labels.size(1), device=labels.device)[None, :] == position
            )
            parts[(length, position)] = losses[mask].sum() / denominator
    return total, losses, parts


def _audit_census_checkpoint(
    core: Any, state: dict[str, Any], batches: dict[int, list[Any]]
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    primitive = rec004q._new_primitive(core, state)
    primitive.eval()
    named, names = _selected_params(primitive)
    params = [named[name] for name in names]
    zeros = _empty_vectors(primitive)
    strata: dict[tuple[int, int], dict[str, torch.Tensor]] = {
        (length, pos): {key: value.clone() for key, value in zeros.items()}
        for length in range(6, 11)
        for pos in range(length)
    }
    direct = {key: value.clone() for key, value in zeros.items()}
    decomposed = {key: value.clone() for key, value in zeros.items()}
    parity: list[dict[str, Any]] = []
    for step, examples in batches.items():
        lengths = [len(example.input_tokens) for example in examples]
        labels = _labels_for_examples(examples, lengths, max(lengths), core.device)
        with torch.no_grad():
            batch = collate_content_only_batch(examples, core.tokens, device=core.device)
            content = core.model.encode(batch)[:, 1 : 1 + max(lengths), :]
        total, _losses, parts = _loss_parts(primitive, content, lengths, labels)
        direct_grad = _gradient(total, params)
        direct_vector, direct_groups = _vector(direct_grad, names, _score_schema(primitive))
        part_sum = torch.zeros_like(direct_vector)
        part_values = list(parts.items())
        part_vectors = _batched_gradient_vectors(
            [loss for _key, loss in part_values],
            params,
            names,
            _score_schema(primitive),
        )
        for (key, _loss), (part_vector, part_groups) in zip(part_values, part_vectors, strict=True):
            part_sum += part_vector
            for group, vector in {"ALL_SCORE": part_vector, **part_groups}.items():
                strata[key][group] += vector.cpu() / len(batches)
        maximum = float((direct_vector - part_sum).abs().max().item())
        reference = max(float(direct_vector.abs().max().item()), 1e-30)
        passed = bool(torch.allclose(direct_vector, part_sum, atol=REC004R_ATOL, rtol=REC004R_RTOL))
        parity.append(
            {
                "step": step,
                "max_abs_error": maximum,
                "relative_to_direct_max": maximum / reference,
                "passed": passed,
            }
        )
        for group, vector in {"ALL_SCORE": direct_vector, **direct_groups}.items():
            direct[group] += vector.cpu() / len(batches)
        for group in decomposed:
            decomposed[group] = sum(
                (strata[key][group] for key in strata), torch.zeros_like(decomposed[group])
            )
    result = {
        "strata": {
            f"length{length}_position{position}": value
            for (length, position), value in strata.items()
        },
        "direct": direct,
        "decomposed": decomposed,
    }
    return (
        result,
        {
            "tolerance": {"atol": REC004R_ATOL, "rtol": REC004R_RTOL},
            "batches": parity,
            "passed": all(row["passed"] for row in parity),
        },
        primitive,
    )


def _aggregate(
    strata: dict[str, dict[str, torch.Tensor]], group: str, target_position: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = strata[f"length10_position{target_position}"][group]
    same = sum(
        (strata[f"length10_position{pos}"][group] for pos in range(10) if pos != target_position),
        torch.zeros_like(target),
    )
    cross = sum(
        (
            strata[f"length{length}_position{pos}"][group]
            for length in range(6, 10)
            for pos in range(length)
        ),
        torch.zeros_like(target),
    )
    return target, same, cross, same + cross


def _summary_for_target(
    result: dict[str, Any],
    target_position: int,
    align: torch.Tensor | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in result["direct"]:
        target, same, cross, all_other = _aggregate(result["strata"], group, target_position)
        self_drive = float(torch.dot(target, target).item())
        total_drive = float(torch.dot(target, result["direct"][group]).item())
        row: dict[str, Any] = {
            "same_length_other_positions": _metrics(target, same),
            "other_lengths": _metrics(target, cross),
            "all_other": _metrics(target, all_other),
            "self_drive": self_drive,
            "total_drive": total_drive,
            "retention_ratio": None if self_drive == 0.0 else total_drive / self_drive,
        }
        if align is not None and group == "ALL_SCORE":
            row["oracle_alignment_auxiliary"] = {
                "cosine_target_vs_align": _metrics(target, align)["cosine"],
                "cosine_total_vs_align": _metrics(result["direct"][group], align)["cosine"],
                "cosine_same_vs_align": _metrics(same, align)["cosine"],
                "cosine_cross_vs_align": _metrics(cross, align)["cosine"],
            }
        values[group] = row
    return values


def _align_gradient(core: Any, state: dict[str, Any], probe: list[Any]) -> torch.Tensor:
    primitive = rec004q._new_primitive(core, state)
    primitive.eval()
    named, names = _selected_params(primitive)
    with torch.no_grad():
        batch = collate_content_only_batch(probe, core.tokens, device=core.device)
        content = core.model.encode(batch)[:, 1:11, :]
    scores = rec004q._score_forward(primitive, content, [10] * len(probe))["scores"][:, :, 4, :10]
    targets = torch.full(
        (scores.size(0), scores.size(1)), 9, device=scores.device, dtype=torch.long
    )
    # pi_10(4) = (4 + 5) mod 10 = 9.
    gradient = _gradient(
        F.cross_entropy(scores.reshape(-1, 10), targets.reshape(-1)),
        [named[name] for name in names],
    )
    return _vector(gradient, names, _score_schema(primitive))[0].cpu()


def _fixed_examples(probe: dict[int, list[Any]]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    target = sorted(probe[10], key=_digest_example)[:REC004R_PER_EXAMPLE_COUNT]
    reference = [
        example
        for length in range(6, 10)
        for example in sorted(probe[length], key=_digest_example)[:REC004R_REFERENCE_PER_LENGTH]
    ]
    return (
        target,
        reference,
        {
            "target_examples": [_digest_example(item) for item in target],
            "cross_length_reference_examples": [_digest_example(item) for item in reference],
        },
    )


def _per_example(
    core: Any,
    state: dict[str, Any],
    target_examples: list[Any],
    reference: list[Any],
    checkpoint_id: str,
) -> list[dict[str, Any]]:
    primitive = rec004q._new_primitive(core, state)
    primitive.eval()
    named, names = _selected_params(primitive)
    params = [named[name] for name in names]
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(target_examples):
        examples = [example, *reference]
        lengths = [10, *[len(item.input_tokens) for item in reference]]
        labels = _labels_for_examples(examples, lengths, 10, core.device)
        with torch.no_grad():
            batch = collate_content_only_batch(examples, core.tokens, device=core.device)
            content = core.model.encode(batch)[:, 1:11, :]
        _total, losses, _parts = _loss_parts(primitive, content, lengths, labels)
        denominator = (labels != IGNORE_INDEX).sum()
        target_loss = losses[0, 4] / denominator
        same_loss = losses[0, [pos for pos in range(10) if pos != 4]].sum() / denominator
        cross_loss = losses[1:][labels[1:] != IGNORE_INDEX].sum() / denominator
        vectors = [
            vector.cpu()
            for vector, _groups in _batched_gradient_vectors(
                [target_loss, same_loss, cross_loss],
                params,
                names,
                _score_schema(primitive),
            )
        ]
        target_vector, same, cross = vectors
        total = target_vector + same + cross
        self_drive = float(torch.dot(target_vector, target_vector).item())
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "sample_index": index,
                "sample_id": _digest_example(example),
                "target_descent_retention_ratio": None
                if self_drive == 0.0
                else float(torch.dot(target_vector, total).item()) / self_drive,
                "target_descent_retention_sign": float(torch.dot(target_vector, total).item())
                > 0.0,
                "same_length_other_position_drive": float(torch.dot(target_vector, same).item()),
                "same_length_other_position_conflict": float(torch.dot(target_vector, same).item())
                < 0.0,
                "cross_length_drive": float(torch.dot(target_vector, cross).item()),
                "cross_length_conflict": float(torch.dot(target_vector, cross).item()) < 0.0,
            }
        )
    return rows


def _per_example_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = sorted(
        row["target_descent_retention_ratio"]
        for row in rows
        if row["target_descent_retention_ratio"] is not None
    )
    return {
        "n": len(rows),
        "same_length_negative_fraction": sum(
            row["same_length_other_position_conflict"] for row in rows
        )
        / len(rows),
        "cross_length_negative_fraction": sum(row["cross_length_conflict"] for row in rows)
        / len(rows),
        "median_retention_ratio": ratios[len(ratios) // 2] if ratios else None,
    }


def _decision(per_example: dict[str, dict[str, Any]]) -> dict[str, Any]:
    terminal = (per_example["I03_SCORE_ONLY_12000"], per_example["I03_CP_SCORE_12000"])
    control = per_example["I04_P_7000"]
    position = (
        all(row["same_length_negative_fraction"] >= 0.60 for row in terminal)
        and control["same_length_negative_fraction"] < 0.60
    )
    length = (
        all(row["cross_length_negative_fraction"] >= 0.60 for row in terminal)
        and control["cross_length_negative_fraction"] < 0.60
    )
    suppression = (
        all(
            (row["median_retention_ratio"] is not None and row["median_retention_ratio"] <= 0.25)
            for row in terminal
        )
        and control["median_retention_ratio"] is not None
        and control["median_retention_ratio"] > 0.50
    )
    if position and length:
        label, proposal = (
            "MIXED_POSITION_LENGTH_GRADIENT_INTERFERENCE",
            "EVIDENCE_INSUFFICIENT_FOR_SINGLE_REPAIR",
        )
    elif position:
        label, proposal = (
            "CROSS_POSITION_GRADIENT_INTERFERENCE_SUPPORTED",
            "Design one output-position shared-score gradient-interference mitigation; do not add oracle position supervision.",
        )
    elif length:
        label, proposal = (
            "CROSS_LENGTH_GRADIENT_INTERFERENCE_SUPPORTED",
            "Design one length-stratum shared-score gradient-interference mitigation; do not adopt length-10 oversampling.",
        )
    elif suppression:
        label, proposal = (
            "STRONG_AGGREGATE_GRADIENT_SUPPRESSION",
            "Measure finite-step score-function change against local gradient prediction before proposing a repair.",
        )
    else:
        label, proposal = (
            "SHARED_PARAMETER_GRADIENT_INTERFERENCE_NOT_SUPPORTED",
            "Measure finite-step score-function change against local gradient prediction; do not change objective, optimizer, or architecture by inference.",
        )
    return {
        "label": label,
        "criteria": {
            "cross_position": position,
            "cross_length": length,
            "strong_aggregate_suppression": suppression,
            "per_example": per_example,
        },
        "next_repair_proposal": proposal,
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "selected_loss_change": None,
        "selected_gradient_method": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "flattening": "C-order, group key order Q_rows,K_rows,position_bias_hidden,position_bias_out",
        }
        for key, value in arrays.items()
    }


def run_mirror_cross_position_cross_length_score_gradient_interference_audit_task(
    config: MirrorCrossPositionCrossLengthScoreGradientInterferenceAuditConfig,
) -> dict[str, Any]:
    """Run the explicitly bounded I03-R audit and stop at its diagnostic result."""
    if config.seed != 10:
        raise ValueError("B-C005REC-004R is fixed to recovery seed 10")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    census, census_manifest = _census(config.seed)
    _write_json(output_dir / "training_gradient_census_manifest.json", census_manifest)
    loaded, checkpoint_manifest = rec004q._source_checkpoint_manifest()
    _write_json(output_dir / "source_checkpoint_manifest.json", checkpoint_manifest)
    probe, probe_manifest = rec004q._build_probe(config.seed)
    _write_json(output_dir / "score_credit_probe_manifest.json", probe_manifest)
    parent_manifest, _raw = ibc._load_parent_manifest()
    core, bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core_before = mb.canonical_state_hash(core.model.state_dict())
    protected_before = mpbr._protected_scope_hashes(bank, op_to_id)
    sources_before = {spec.checkpoint_id: mb.raw_file_sha256(spec.path) for spec, _state in loaded}
    census_results: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    primitives: dict[str, Any] = {}
    for spec, state in loaded:
        result, check, primitive = _audit_census_checkpoint(core, state, census)
        (
            census_results[spec.checkpoint_id],
            parity[spec.checkpoint_id],
            primitives[spec.checkpoint_id],
        ) = result, check, primitive
    _write_json(output_dir / "gradient_decomposition_parity.json", parity)
    if not all(value["passed"] for value in parity.values()):
        decision = {
            "label": "LOSS_GRADIENT_DECOMPOSITION_INVALID",
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_step": None,
            "selected_loss_change": None,
            "selected_gradient_method": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
        }
        _write_json(output_dir / "gradient_interference_decision.json", decision)
        return {
            "task_id": REC004R_TASK_ID,
            "implementation_status": "STOPPED",
            "result_label": decision["label"],
        }
    position_arrays: dict[str, np.ndarray] = {}
    length_arrays: dict[str, np.ndarray] = {}
    position4: dict[str, Any] = {}
    position5: dict[str, Any] = {}
    cross_length: dict[str, Any] = {}
    for spec, state in loaded:
        alignment = _align_gradient(core, state, probe[10])
        position4[spec.checkpoint_id] = _summary_for_target(
            census_results[spec.checkpoint_id], 4, alignment
        )
        position5[spec.checkpoint_id] = _summary_for_target(census_results[spec.checkpoint_id], 5)
        cross_length[spec.checkpoint_id] = {
            group: position4[spec.checkpoint_id][group]["other_lengths"]
            for group in position4[spec.checkpoint_id]
        }
        for pos in range(10):
            for group in census_results[spec.checkpoint_id]["direct"]:
                position_arrays[f"{spec.checkpoint_id}__position{pos}__{group}"] = census_results[
                    spec.checkpoint_id
                ]["strata"][f"length10_position{pos}"][group].numpy()
        for length in range(6, 11):
            for group in census_results[spec.checkpoint_id]["direct"]:
                value = sum(
                    (
                        census_results[spec.checkpoint_id]["strata"][
                            f"length{length}_position{pos}"
                        ][group]
                        for pos in range(length)
                    ),
                    torch.zeros_like(census_results[spec.checkpoint_id]["direct"][group]),
                )
                length_arrays[f"{spec.checkpoint_id}__length{length}__{group}"] = value.numpy()
    np.savez_compressed(output_dir / "per_position_parameter_gradients.npz", **position_arrays)  # type: ignore[arg-type]
    np.savez_compressed(output_dir / "per_length_parameter_gradients.npz", **length_arrays)  # type: ignore[arg-type]
    _write_json(
        output_dir / "gradient_array_manifest.json",
        {
            "per_position": _array_manifest(position_arrays),
            "per_length": _array_manifest(length_arrays),
            "parameter_ordering": "each listed group follows primitive named tensor order; Q/K are exact fused in_proj row slices",
        },
    )
    _write_json(output_dir / "position4_interference_summary.json", position4)
    _write_json(output_dir / "position5_control_summary.json", position5)
    _write_json(output_dir / "cross_length_interference_summary.json", cross_length)
    targets, reference, fixed_manifest = _fixed_examples(probe)
    _write_json(output_dir / "per_example_selection_manifest.json", fixed_manifest)
    per_rows: list[dict[str, Any]] = []
    per_summary: dict[str, dict[str, Any]] = {}
    selected_ids = {"I03_SCORE_ONLY_12000", "I03_CP_SCORE_12000", "I04_P_7000"}
    for spec, state in loaded:
        if spec.checkpoint_id in selected_ids:
            rows = _per_example(core, state, targets, reference, spec.checkpoint_id)
            per_rows.extend(rows)
            per_summary[spec.checkpoint_id] = _per_example_summary(rows)
    with (output_dir / "per_example_interference.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    _write_json(
        output_dir / "successful_control_comparison.json",
        {
            "I04_P_6000": position4["I04_P_6000"],
            "I04_P_7000": position4["I04_P_7000"],
            "per_example_I04_P_7000": per_summary["I04_P_7000"],
            "control_checkpoint_substituted": False,
        },
    )
    decision = _decision(per_summary)
    _write_json(output_dir / "gradient_interference_decision.json", decision)
    (output_dir / "next_repair_contract.md").write_text(
        f"# Next repair contract\n\nDiagnostic label: `{decision['label']}`.\n\n{decision['next_repair_proposal']}\n\nThis is a proposal only; no mitigation training is authorized.\n",
        encoding="utf-8",
    )
    sources_after = {spec.checkpoint_id: mb.raw_file_sha256(spec.path) for spec, _state in loaded}
    freeze = {
        "core_unchanged": core_before == mb.canonical_state_hash(core.model.state_dict()),
        "protected_operations_unchanged": protected_before
        == mpbr._protected_scope_hashes(bank, op_to_id),
        "source_checkpoints_unchanged": sources_before == sources_after,
    }
    freeze["passed"] = all(freeze.values())
    _write_json(output_dir / "freeze_audit.json", freeze)
    side_effect = {
        "new_optimizer_updates": 0,
        "optimizer_step_called": False,
        "parameter_mutation": False,
        "training": "NOT_EXECUTED",
        "autograd_method": "torch.autograd.grad on in-memory checkpoint copies",
        "sealed_or_rg3_query": "NOT_EXECUTED",
        "child_bundle": None,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect)
    report = {
        "task_id": REC004R_TASK_ID,
        "implementation_status": "COMPLETED",
        "result_label": decision["label"],
        "decision": decision,
        "freeze_audit": freeze,
        "side_effect_audit": side_effect,
        "cost_accounting": {
            "wall_clock_seconds": time.time() - started,
            "new_optimizer_updates": 0,
            "census_steps": len(census),
            "checkpoint_count": len(loaded),
        },
        "selected_init": None,
        "selected_step": None,
        "selected_loss_change": None,
        "selected_gradient_method": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(
        f"# {REC004R_TASK_ID} — Score-gradient interference audit\n\nResult: `{decision['label']}`.\n\nThe diagnostic performed zero optimizer updates and did not run repair training, candidate selection, RG3, REC-005, or sealed evaluation.\n",
        encoding="utf-8",
    )
    return report
