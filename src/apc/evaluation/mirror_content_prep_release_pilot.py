"""B-C005REC-004P: I03 CONTENT_PREP-release score/value compatibility pilot."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as rec004g
from apc.evaluation import mirror_late_stage_attention_bottleneck_revalidation as rec004j
from apc.evaluation import mirror_oracle_attention_substitution_probe as oracle_probe
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_position_score_residual_audit as resid_audit
from apc.evaluation import mirror_score_only_continuation_pilot as score_only
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_PILOT_SEED,
    _guard_not_frozen,
    _snapshot_forbidden_cache_hashes,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

REC004P_TASK_ID: Final = "B-C005REC-004P"
REC004P_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_I03_SHARED_CONTENT_PREP_RELEASE_PILOT.md"
)
REC004P_VALIDATION_SPLIT: Final = "content_prep_release_validation_v1"
REC004P_CONFIRMATION_SPLIT: Final = "content_prep_release_length10_confirmation_v1"
REC004P_SCORE_ONLY_STATE: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004o/run_003/I03/"
    "SCORE_ONLY_QK_POSITION_BIAS/training_states/step12000.pt"
)
REC004P_SCORE_ONLY_SUMMARY: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004o/run_003/summary.json"
)
REC004P_CONTENT_PREP_PREFIXES: Final = (
    "content_in_proj.",
    "content_position_embedding.",
)
REC004P_CHUNK_SIZE: Final = 128


@dataclass(frozen=True)
class MirrorContentPrepReleasePilotConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004p/run_001")
    seed: int = RECOVERY_PILOT_SEED
    checkpoint_interval: int = score_only.REC004O_CHECKPOINT_INTERVAL


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_content_prep_release_datasets(seed: int) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Lock new data before any model forward, treating REC-004O data as protected."""
    old_sets, old_details = score_only.build_score_only_datasets(seed)
    protected, protected_counts = score_only.traj_audit.build_protected_digest_registry(seed)
    for examples in old_sets.values():
        protected |= score_only._digest_examples(examples)
    validation, validation_detail = score_only._build_dataset(
        seed=seed,
        split=REC004P_VALIDATION_SPLIT,
        n_examples=score_only.REC004O_VALIDATION_EXAMPLES,
        protected=protected,
        fixed_length=None,
    )
    confirmation, confirmation_detail = score_only._build_dataset(
        seed=seed,
        split=REC004P_CONFIRMATION_SPLIT,
        n_examples=score_only.REC004O_CONFIRMATION_EXAMPLES,
        protected=protected | score_only._digest_examples(validation),
        fixed_length=score_only.REC004O_CONFIRMATION_LENGTH,
    )
    validation_digests = score_only._digest_examples(validation)
    confirmation_digests = score_only._digest_examples(confirmation)
    overlap: dict[str, Any] = {
        "validation_with_protected": sorted(validation_digests & protected),
        "confirmation_with_protected": sorted(confirmation_digests & protected),
        "new_sets_mutual": sorted(validation_digests & confirmation_digests),
    }
    overlap["disjoint"] = not any(overlap.values())
    if not overlap["disjoint"]:
        raise score_only.SelectiveFreezeUnsafe("validation collision after pre-output lock")
    return (
        {REC004P_VALIDATION_SPLIT: validation, REC004P_CONFIRMATION_SPLIT: confirmation},
        {
            REC004P_VALIDATION_SPLIT: validation_detail,
            REC004P_CONFIRMATION_SPLIT: confirmation_detail,
            "protected_registry": {
                "base_source_counts": protected_counts,
                "rec004o_dataset_details": old_details,
                "training_stream_contract": "steps_1_to_18000 (superset of required 1_to_12000)",
                "protected_digest_count_before_new_sets": len(protected),
            },
            "cross_dataset_disjointness": overlap,
        },
    )


def build_content_prep_partition(primitive: Any) -> dict[str, Any]:
    """Derive the release from the real parameter graph, not checkpoint names."""
    base = score_only.build_forward_graph_partition(primitive)
    groups = base["groups"]
    content_prep = [
        name
        for name in groups["frozen_downstream"]
        if name.startswith(REC004P_CONTENT_PREP_PREFIXES)
    ]
    frozen = [name for name in groups["frozen_downstream"] if name not in content_prep]
    if not content_prep:
        raise score_only.SelectiveFreezeUnsafe("CONTENT_PREP_RELEASE_UNSAFE: CONTENT_PREP absent")
    groups = {**groups, "content_prep": content_prep, "frozen_downstream": frozen}
    trainable = content_prep + groups["fused_qk"] + groups["score_position_bias"]
    if set(trainable) | set(frozen) != set(base["parameter_names_from_loaded_runtime"]):
        raise score_only.SelectiveFreezeUnsafe("CONTENT_PREP_RELEASE_UNSAFE: incomplete partition")
    return {
        **base,
        "groups": groups,
        "trainable_tensor_keys": trainable,
        "frozen_whole_tensor_keys": frozen,
        "release_contract": "CONTENT_PREP is the only release relative to REC-004O.",
    }


def _training_manifest(seed: int) -> dict[str, Any]:
    """Hash the inherited stream without a model forward or output observation."""
    digest = hashlib.sha256()
    n_examples = 0
    for step in range(6001, 12001):
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            rec004g.REC004G_TARGET_OPERATION,
            vocab_size=rec004g.REC004G_VOCAB_SIZE,
            sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        digest.update(str(step).encode())
        for example in examples:
            digest.update(json.dumps([example.input_tokens, example.target_tokens]).encode())
        n_examples += len(examples)
    return {
        "step_range": [6001, 12000],
        "new_optimizer_updates": 6000,
        "n_examples": n_examples,
        "input_target_digest_sha256": digest.hexdigest(),
        "generator": "REC-004D/G deterministic MIRROR_HALVES stream",
        "no_length10_oversampling": True,
        "oracle_used_in_training": False,
    }


def _train(
    core: Any,
    source_state: dict[str, Any],
    partition: dict[str, Any],
    config: MirrorContentPrepReleasePilotConfig,
    output_dir: Path,
) -> dict[str, Any]:
    primitive, optimizer, scheduler = score_only._build_resumed_runtime(core, source_state)
    primitive.train()
    guard = score_only._FreezeGuard(primitive, optimizer, partition)
    rec004g._restore_rng_state(source_state, core.device)
    state_dir = output_dir / "I03" / "CP_SCORE_CONTINUATION" / "training_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    named = dict(primitive.named_parameters())
    gradients = {name: 0.0 for name in partition["trainable_tensor_keys"]}
    v_gradients = {name: 0.0 for name in guard.fused_names}
    learning_curve: list[dict[str, Any]] = []
    start = time.time()
    for step in range(6001, 12001):
        examples = ibc._generate_step_training_examples(
            config.seed,
            step,
            rec004g.REC004G_TARGET_OPERATION,
            vocab_size=rec004g.REC004G_VOCAB_SIZE,
            sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        lengths = [len(example.input_tokens) for example in examples]
        output_lengths = [
            get_operation(rec004g.REC004G_TARGET_OPERATION).output_length(length)
            for length in lengths
        ]
        labels = score_only._labels_for_examples(
            examples, output_lengths, max(output_lengths), core.device
        )
        with torch.no_grad():
            batch = collate_content_only_batch(examples, core.tokens, device=core.device)
            h_content = core.model.encode(batch)[:, 1 : 1 + max(lengths), :]
        optimizer.zero_grad(set_to_none=True)
        lr_used = float(optimizer.param_groups[0]["lr"])
        logits = primitive(h_content, lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        for name in gradients:
            if named[name].grad is not None:
                gradients[name] += float(named[name].grad.abs().sum().item())
        for name, value in guard.zero_v_gradients().items():
            v_gradients[name] += value
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), rec004g.REC004G_OPERATOR_GRAD_CLIP)
        optimizer.step()
        guard.restore_and_verify(step)
        scheduler.step()
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise score_only.SelectiveFreezeUnsafe(
                "FUSED_QKV_SELECTIVE_FREEZE_FAILURE: non-finite loss"
            )
        learning_curve.append(
            {
                "step": step,
                "lr_used": lr_used,
                "lr_after_scheduler": float(scheduler.get_last_lr()[0]),
                "loss": loss_value,
            }
        )
        if step % config.checkpoint_interval == 0:
            torch.save(
                score_only._training_state(primitive, optimizer, scheduler, core.device, step),
                state_dir / f"step{step}.pt",
            )
    primitive.eval()
    return {
        "primitive": primitive,
        "final_state": score_only._training_state(
            primitive, optimizer, scheduler, core.device, 12000
        ),
        "learning_curve": learning_curve,
        "gradient_l1": gradients,
        "v_gradient_l1_before_mask": v_gradients,
        "freeze_audit": guard.audit(),
        "wall_clock_seconds": time.time() - start,
    }


def _evaluate(core: Any, primitive: Any, datasets: dict[str, list[Any]]) -> dict[str, Any]:
    """Evaluate all required J0/O1 endpoint and descriptive diagnostics."""
    operation = get_operation(rec004g.REC004G_TARGET_OPERATION)
    results: dict[str, Any] = {}
    for dataset_name, examples in datasets.items():
        by_length: dict[int, list[Any]] = {}
        for example in examples:
            by_length.setdefault(len(example.input_tokens), []).append(example)
        total = j0_correct = o1_correct = token_total = j0_tokens = o1_tokens = 0
        j0_loss_sum = o1_loss_sum = 0.0
        per_length: dict[str, Any] = {}
        position_correct: dict[int, int] = {}
        position_total: dict[int, int] = {}
        ranks: list[float] = []
        margins: list[float] = []
        entropies: list[float] = []
        agreements: list[float] = []
        for length, length_examples in sorted(by_length.items()):
            length_j0 = length_o1 = 0
            for start in range(0, len(length_examples), REC004P_CHUNK_SIZE):
                chunk = length_examples[start : start + REC004P_CHUNK_SIZE]
                lengths = [length] * len(chunk)
                output_lengths = [operation.output_length(length)] * len(chunk)
                labels = score_only._labels_for_examples(
                    chunk, output_lengths, max(output_lengths), core.device
                )
                batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
                with torch.no_grad():
                    h_content = core.model.encode(batch)[:, 1 : 1 + length, :]
                    j0 = rec004j._run_j0_decomposition(
                        primitive, h_content, lengths, output_lengths
                    )
                    o1 = oracle_probe.run_oracle_forward(
                        primitive, h_content, lengths, output_lengths
                    )
                    j0_loss_sum += F.cross_entropy(
                        j0["logits"].reshape(-1, j0["logits"].size(-1)),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    ).item()
                    o1_loss_sum += F.cross_entropy(
                        o1["logits"].reshape(-1, o1["logits"].size(-1)),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    ).item()
                j0_predictions = resid_audit._predict_from_logits(j0["logits"], output_lengths)
                o1_predictions = resid_audit._predict_from_logits(o1["logits"], output_lengths)
                probabilities = j0["attn_probs"].detach().cpu().numpy()
                average_probabilities = probabilities.mean(axis=1)
                for row, (j0_prediction, o1_prediction, example) in enumerate(
                    zip(j0_predictions, o1_predictions, chunk, strict=True)
                ):
                    target = tuple(example.target_tokens)
                    length_j0 += int(tuple(j0_prediction) == target)
                    length_o1 += int(tuple(o1_prediction) == target)
                    j0_tokens += sum(
                        predicted == actual
                        for predicted, actual in zip(j0_prediction, target, strict=True)
                    )
                    o1_tokens += sum(
                        predicted == actual
                        for predicted, actual in zip(o1_prediction, target, strict=True)
                    )
                    for position, (predicted, actual) in enumerate(
                        zip(j0_prediction, target, strict=True)
                    ):
                        position_correct[position] = position_correct.get(position, 0) + int(
                            predicted == actual
                        )
                        position_total[position] = position_total.get(position, 0) + 1
                    for position in range(length):
                        correct_key = (
                            position + length // 2
                            if position < length // 2
                            else position - length // 2
                        )
                        probability_row = average_probabilities[row, position]
                        ordered = np.argsort(-probability_row)
                        ranks.append(float(np.where(ordered == correct_key)[0][0] + 1))
                        margins.append(
                            float(
                                probability_row[correct_key]
                                - max(
                                    probability_row[index]
                                    for index in range(length)
                                    if index != correct_key
                                )
                            )
                        )
                    entropies.extend(
                        (
                            -(
                                average_probabilities[row]
                                * np.log(average_probabilities[row] + 1e-12)
                            ).sum(axis=-1)
                        ).tolist()
                    )
                    per_head_argmax = probabilities[row].argmax(axis=-1)
                    agreements.extend(
                        [
                            float(
                                np.all(per_head_argmax[:, position] == per_head_argmax[0, position])
                            )
                            for position in range(length)
                        ]
                    )
                token_total += len(chunk) * length
            total += len(length_examples)
            j0_correct += length_j0
            o1_correct += length_o1
            per_length[str(length)] = {
                "n": len(length_examples),
                "j0_sequence_exact_match": length_j0 / len(length_examples),
                "oracle_sequence_exact_match": length_o1 / len(length_examples),
            }
        results[dataset_name] = {
            "n": total,
            "j0_sequence_exact_match": j0_correct / total,
            "oracle_sequence_exact_match": o1_correct / total,
            "j0_token_accuracy": j0_tokens / token_total,
            "oracle_token_accuracy": o1_tokens / token_total,
            "j0_mean_loss": j0_loss_sum / token_total,
            "oracle_mean_loss": o1_loss_sum / token_total,
            "per_length": per_length,
            "per_output_position": {
                str(position): {
                    "j0_accuracy": position_correct[position] / position_total[position]
                }
                for position in position_total
            },
            "position4_accuracy": position_correct.get(4, 0) / position_total.get(4, 1),
            "position5_accuracy": position_correct.get(5, 0) / position_total.get(5, 1),
            "correct_key_rank_margin": {
                "descriptive_only": True,
                "mean_rank": float(np.mean(ranks)),
                "mean_margin": float(np.mean(margins)),
            },
            "attention_entropy": {
                "descriptive_only": True,
                "j0_head_averaged_mean": float(np.mean(entropies)),
            },
            "head_agreement": {
                "descriptive_only": True,
                "argmax_agreement_rate": float(np.mean(agreements)),
            },
        }
    return results


def _decision(
    cp_score: dict[str, Any], score_only_metrics: dict[str, Any], contracts_pass: bool
) -> dict[str, Any]:
    validation = cp_score[REC004P_VALIDATION_SPLIT]
    confirmation = cp_score[REC004P_CONFIRMATION_SPLIT]
    score_validation = score_only_metrics[REC004P_VALIDATION_SPLIT]
    score_confirmation = score_only_metrics[REC004P_CONFIRMATION_SPLIT]
    validation_length10 = validation["per_length"].get("10", {}).get("j0_sequence_exact_match", 0.0)
    score_length10 = (
        score_validation["per_length"].get("10", {}).get("j0_sequence_exact_match", 0.0)
    )
    primary = all(
        (
            validation["j0_sequence_exact_match"] >= 0.95,
            validation_length10 >= 0.95,
            confirmation["j0_sequence_exact_match"] >= 0.95,
            validation["oracle_sequence_exact_match"] >= 0.95,
            confirmation["oracle_sequence_exact_match"] >= 0.95,
            contracts_pass,
        )
    )
    deltas = {
        "validation_length10": validation_length10 - score_length10,
        "confirmation_length10": (
            confirmation["j0_sequence_exact_match"] - score_confirmation["j0_sequence_exact_match"]
        ),
    }
    helpful = all(delta >= 0.10 for delta in deltas.values())
    oracle_compatible = (
        validation["oracle_sequence_exact_match"] >= 0.95
        and confirmation["oracle_sequence_exact_match"] >= 0.95
    )
    if primary:
        label = "SHARED_CONTENT_PREP_RELEASE_SUPPORTED_WITH_COMPATIBILITY"
    elif helpful and not oracle_compatible:
        label = "SHARED_CONTENT_PREP_SCORE_VALUE_CONFLICT_SUPPORTED"
    elif helpful:
        label = "CONTENT_PREP_ADAPTATION_HELPFUL_BUT_INSUFFICIENT"
    else:
        label = "CONTENT_PREP_RELEASE_NOT_SUPPORTED"
    return {
        "label": label,
        "primary_gate_passed": primary,
        "contracts_passed": contracts_pass,
        "effect_size_floor": 0.10,
        "j0_length10_deltas_vs_score_only": deltas,
        "o1_compatible": oracle_compatible,
        "selected_init": None,
        "selected_step": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }


def _load_score_only_primitive(core: Any) -> Any:
    state = torch.load(REC004P_SCORE_ONLY_STATE, map_location="cpu", weights_only=False)
    primitive = mpbr._new_arm_primitive(core, score_only.REC004O_ARM)
    primitive.to(core.device)
    primitive.load_state_dict(
        {key: value.to(core.device) for key, value in state["primitive_state_dict"].items()},
        strict=True,
    )
    primitive.eval()
    return primitive


def _render_report(report: dict[str, Any]) -> str:
    return (
        f"# {REC004P_TASK_ID} — I03 shared CONTENT_PREP release pilot\n\n"
        f"Result: `{report['result_label']}`\n\n"
        "Terminal-only pilot: no candidate was selected, no child bundle was created, "
        "and RG3/REC-005 were not executed.\n"
    )


def run_mirror_content_prep_release_pilot_task(
    config: MirrorContentPrepReleasePilotConfig,
) -> dict[str, Any]:
    """Run exactly the authorized I03 step-6000 to step-12000 release arm."""
    _guard_not_frozen("run_mirror_content_prep_release_pilot_task")
    if config.seed != RECOVERY_PILOT_SEED:
        raise ValueError(f"{REC004P_TASK_ID} is fixed to seed {RECOVERY_PILOT_SEED}")
    required = (REC004P_CONTRACT_FILE, REC004P_SCORE_ONLY_STATE, REC004P_SCORE_ONLY_SUMMARY)
    if not all(path.is_file() for path in required):
        return {"implementation_status": "STOPPED", "result_label": "SOURCE_ARTIFACT_UNAVAILABLE"}
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _write_json(
        output_dir / "config.yaml",
        {
            "seed": config.seed,
            "output_dir": str(output_dir),
            "checkpoint_interval": config.checkpoint_interval,
        },
    )
    _write_json(output_dir / "system.json", get_system_info(seed=config.seed))
    source_paths = {
        "source_6000": rec004g._rec004d_training_state_path("I03"),
        "joint_12000": score_only._rec004g_state_path(12000),
        "rec004o_run_003": REC004P_SCORE_ONLY_SUMMARY,
        "score_only_12000": REC004P_SCORE_ONLY_STATE,
    }
    if not all(path.is_file() for path in source_paths.values()):
        return {"implementation_status": "STOPPED", "result_label": "SOURCE_ARTIFACT_UNAVAILABLE"}
    source_hashes_before = {name: mb.raw_file_sha256(path) for name, path in source_paths.items()}
    cache_before = _snapshot_forbidden_cache_hashes(config.seed)

    # These writes deliberately precede runtime reconstruction and all model output.
    datasets, validation_manifest = build_content_prep_release_datasets(config.seed)
    _write_json(output_dir / "new_validation_manifest.json", validation_manifest)
    _write_json(output_dir / "training_data_manifest.json", _training_manifest(config.seed))

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core_hash_before = mb.canonical_state_hash(core.model.state_dict())
    protected_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    source_state = rec004g.load_source_training_state("I03")
    trace_primitive, _optimizer, _scheduler = score_only._build_resumed_runtime(core, source_state)
    partition = build_content_prep_partition(trace_primitive)
    partition["real_forward_backward_trace"] = score_only._forward_backward_trace(
        core, trace_primitive, datasets[REC004P_VALIDATION_SPLIT]
    )
    _write_json(output_dir / "trainable_partition.json", partition)
    control = score_only.run_joint_control_replay(core, source_state)
    source_replay = {
        "status": control["status"],
        "joint_control_replay": control,
        "official_score_only_run": json.loads(
            REC004P_SCORE_ONLY_SUMMARY.read_text(encoding="utf-8")
        ),
        "source_hashes_before": source_hashes_before,
    }
    _write_json(output_dir / "source_replay.json", source_replay)
    if control["status"] != "VERIFIED":
        report = {
            "implementation_status": "STOPPED",
            "result_label": "SOURCE_REPLAY_MISMATCH",
            "source_replay": source_replay,
        }
        _write_json(output_dir / "summary.json", report)
        (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
        return report
    try:
        outcome = _train(core, source_state, partition, config, output_dir)
    except score_only.SelectiveFreezeUnsafe as error:
        report = {
            "implementation_status": "STOPPED",
            "result_label": "FUSED_QKV_SELECTIVE_FREEZE_FAILURE",
            "reason": str(error),
            "source_replay": source_replay,
        }
        _write_json(output_dir / "summary.json", report)
        (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
        return report
    (output_dir / "learning_curve.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcome["learning_curve"]), encoding="utf-8"
    )
    _write_json(output_dir / "fused_qkv_freeze_audit.json", outcome["freeze_audit"])

    early = score_only._load_historical_primitive(core, 6000, "source")
    joint = score_only._load_historical_primitive(core, 12000, "joint")
    score_only_terminal = _load_score_only_primitive(core)
    cp_metrics = _evaluate(core, outcome["primitive"], datasets)
    early_metrics = _evaluate(core, early, datasets)
    joint_metrics = _evaluate(core, joint, datasets)
    score_metrics = _evaluate(core, score_only_terminal, datasets)
    endpoint_metrics = {
        "EARLY": early_metrics,
        "SCORE_ONLY": score_metrics,
        "JOINT": joint_metrics,
        "CP_SCORE": cp_metrics,
    }
    _write_json(output_dir / "endpoint_metrics.json", endpoint_metrics)
    comparisons = {
        "CP_SCORE_vs_SCORE_ONLY": {
            split: cp_metrics[split]["j0_sequence_exact_match"]
            - score_metrics[split]["j0_sequence_exact_match"]
            for split in datasets
        },
        "CP_SCORE_vs_JOINT": {
            split: cp_metrics[split]["j0_sequence_exact_match"]
            - joint_metrics[split]["j0_sequence_exact_match"]
            for split in datasets
        },
        "JOINT_vs_SCORE_ONLY": {
            split: joint_metrics[split]["j0_sequence_exact_match"]
            - score_metrics[split]["j0_sequence_exact_match"]
            for split in datasets
        },
    }
    _write_json(output_dir / "j0_comparison.json", comparisons)
    o1_audit = {
        comparator: {
            split: {
                "sequence_em": metric[split]["oracle_sequence_exact_match"],
                "position4_accuracy": metric[split]["position4_accuracy"],
                "position5_accuracy": metric[split]["position5_accuracy"],
            }
            for split in datasets
        }
        for comparator, metric in endpoint_metrics.items()
    }
    _write_json(output_dir / "o1_compatibility_audit.json", o1_audit)

    source_hashes_after = {name: mb.raw_file_sha256(path) for name, path in source_paths.items()}
    cache_after = _snapshot_forbidden_cache_hashes(config.seed)
    freeze_audit = {
        "selective_qkv": outcome["freeze_audit"],
        "source_artifacts_unchanged": source_hashes_before == source_hashes_after,
        "core_unchanged": core_hash_before == mb.canonical_state_hash(core.model.state_dict()),
        "protected_operations_unchanged": (
            protected_before == mpbr._protected_scope_hashes(eval_bank, op_to_id)
        ),
        "shared_cache_unchanged": cache_before == cache_after,
    }
    freeze_audit["passed"] = (
        outcome["freeze_audit"]["passed"]
        and freeze_audit["source_artifacts_unchanged"]
        and freeze_audit["core_unchanged"]
        and freeze_audit["protected_operations_unchanged"]
        and freeze_audit["shared_cache_unchanged"]
    )
    _write_json(output_dir / "freeze_audit.json", freeze_audit)
    final_state = outcome["final_state"]["primitive_state_dict"]
    content_changes = {
        name: not torch.equal(final_state[name], source_state["primitive_state_dict"][name])
        for name in partition["groups"]["content_prep"]
    }
    side_effect_audit = {
        "content_prep_updated": content_changes,
        "v_activation_may_change": True,
        "v_parameter_rows_exact": outcome["freeze_audit"]["passed"],
        "no_candidate_adoption": True,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)
    decision = _decision(cp_metrics, score_metrics, freeze_audit["passed"])
    _write_json(output_dir / "result_decision.json", decision)
    (output_dir / "next_repair_contract.md").write_text(
        "# Next repair contract\n\nstatus: `NOT_PROPOSED_IN_THIS_TASK`\n\n"
        "No architecture repair, candidate adoption, RG3, or REC-005 is authorized here.\n",
        encoding="utf-8",
    )
    final_report: dict[str, Any] = {
        "task_id": REC004P_TASK_ID,
        "implementation_status": "COMPLETED",
        "result_label": decision["label"],
        "decision": decision,
        "source_replay": source_replay,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": {
            "wall_clock_seconds": time.time() - started,
            "new_optimizer_updates": 6000,
            "content_prep_gradient_l1": {
                key: outcome["gradient_l1"][key] for key in partition["groups"]["content_prep"]
            },
            "score_gradient_l1": {
                key: outcome["gradient_l1"][key]
                for key in (
                    partition["groups"]["fused_qk"] + partition["groups"]["score_position_bias"]
                )
            },
            "v_gradient_l1_before_mask": outcome["v_gradient_l1_before_mask"],
        },
        "selected_init": None,
        "selected_step": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }
    _write_json(output_dir / "summary.json", final_report)
    (output_dir / "report.md").write_text(_render_report(final_report), encoding="utf-8")
    return final_report
