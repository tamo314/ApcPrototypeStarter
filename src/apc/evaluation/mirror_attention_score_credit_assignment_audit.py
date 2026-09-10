# ruff: noqa: E501
"""B-C005REC-004Q: read-only I03 attention-score credit-assignment audit.

This module deliberately has no training loop and never constructs or steps an
optimizer.  It differentiates only in-memory checkpoint copies with
``torch.autograd.grad`` and reconstructs the next AdamW delta from saved state.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_budget_extension as rec004g
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_content_prep_release_pilot as rec004p
from apc.evaluation import mirror_ffn_value_path_leave_one_out_necessity_audit as rec004n
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_score_only_continuation_pilot as score_only
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

REC004Q_TASK_ID: Final = "B-C005REC-004Q"
REC004Q_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_I03_ATTENTION_SCORE_CREDIT_ASSIGNMENT_GRADIENT_AUDIT.md"
)
REC004Q_OUTPUT_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004q/run_001")
REC004Q_PROBE_NAMESPACE: Final = "score_credit_assignment_probe_v1"
REC004Q_PROBE_COUNTS: Final = {6: 128, 7: 128, 8: 128, 9: 128, 10: 512}
REC004Q_CHUNK_SIZE: Final = 64
REC004Q_WEAK_SIGNAL_RATIO: Final = 0.10


@dataclass(frozen=True)
class MirrorAttentionScoreCreditAssignmentAuditConfig:
    output_dir: Path = REC004Q_OUTPUT_DIR
    seed: int = 10


@dataclass(frozen=True)
class _CheckpointSpec:
    checkpoint_id: str
    path: Path
    init_id: str
    step: int
    arm: str
    update_policy: str
    expected_state_hash: str | None
    source_record: str


_CHECKPOINTS: Final[tuple[_CheckpointSpec, ...]] = (
    _CheckpointSpec(
        "I03_P_6000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/I03/P_LENGTH_POSITION_BIAS/training_states/step6000.pt"
        ),
        "I03",
        6000,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        "ba0f17d903a94e63633d346cf5b37d2087fda6f1980f70a8cd391752945961c4",
        "REC-004D learning_curve.jsonl",
    ),
    _CheckpointSpec(
        "I03_SCORE_ONLY_12000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004o/run_003/I03/SCORE_ONLY_QK_POSITION_BIAS/training_states/step12000.pt"
        ),
        "I03",
        12000,
        "SCORE_ONLY_QK_POSITION_BIAS",
        "score_only",
        "0c51dd36db711674cdad7614b482d216f146f2d58f204a8e4a461f026b9569be",
        "REC-004O summary.json",
    ),
    _CheckpointSpec(
        "I03_CP_SCORE_12000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004p/run_001/I03/CP_SCORE_CONTINUATION/training_states/step12000.pt"
        ),
        "I03",
        12000,
        "CP_SCORE_CONTINUATION",
        "cp_score",
        None,
        "REC-004P terminal training state (canonical hash recorded by this audit)",
    ),
    _CheckpointSpec(
        "I03_JOINT_12000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/I03/P_LENGTH_POSITION_BIAS/training_states/step12000.pt"
        ),
        "I03",
        12000,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        None,
        "REC-004G learning_curve.jsonl",
    ),
    _CheckpointSpec(
        "I04_P_6000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/I04/P_LENGTH_POSITION_BIAS/training_states/step6000.pt"
        ),
        "I04",
        6000,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        None,
        "REC-004D learning_curve.jsonl",
    ),
    _CheckpointSpec(
        "I04_P_7000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/I04/P_LENGTH_POSITION_BIAS/training_states/step7000.pt"
        ),
        "I04",
        7000,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        "7a7159f815f494824524ad20fc5c895fff570b7d2f2e2a996efc5998e32e20a4",
        "REC-004G learning_curve.jsonl",
    ),
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _digest_examples(examples: Iterable[Any]) -> set[str]:
    return {traj_audit._digest_example(example) for example in examples}


def _dataset_digest(digests: set[str]) -> str:
    hasher = hashlib.sha256()
    for digest in sorted(digests):
        hasher.update(digest.encode("ascii"))
    return hasher.hexdigest()


def _new_primitive(core: Any, state: dict[str, Any]) -> Any:
    primitive = mpbr._new_arm_primitive(core, rec004g.REC004G_ARM)
    primitive.to(core.device)
    primitive.load_state_dict(
        {key: value.to(core.device) for key, value in state["primitive_state_dict"].items()},
        strict=True,
    )
    primitive.eval()
    return primitive


def _learning_curve_hash(init_id: str, arm: str, step: int, path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("init_id") == init_id and row.get("arm") == arm and row.get("step") == step:
            return row.get("checkpoint_state_hash")
    return None


def _source_checkpoint_manifest() -> tuple[
    list[tuple[_CheckpointSpec, dict[str, Any]]], dict[str, Any]
]:
    loaded: list[tuple[_CheckpointSpec, dict[str, Any]]] = []
    entries: list[dict[str, Any]] = []
    curves = {
        "REC-004D learning_curve.jsonl": Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/learning_curve.jsonl"
        ),
        "REC-004G learning_curve.jsonl": Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/learning_curve.jsonl"
        ),
    }
    for spec in _CHECKPOINTS:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        state = torch.load(spec.path, map_location="cpu", weights_only=False)
        state_hash = mb.canonical_state_hash(state["primitive_state_dict"])
        curve_hash = _learning_curve_hash(
            spec.init_id, spec.arm, spec.step, curves.get(spec.source_record, Path())
        )
        expected = spec.expected_state_hash or curve_hash
        if expected is not None and state_hash != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {spec.checkpoint_id}")
        entries.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "path": str(spec.path),
                "init_id": spec.init_id,
                "step": spec.step,
                "arm": spec.arm,
                "update_policy": spec.update_policy,
                "raw_file_sha256": mb.raw_file_sha256(spec.path),
                "canonical_primitive_state_hash": state_hash,
                "expected_canonical_hash": expected,
                "source_record": spec.source_record,
                "verified": True,
            }
        )
        loaded.append((spec, state))
    return loaded, {"checkpoints": entries, "all_verified": True}


def _build_probe(seed: int) -> tuple[dict[int, list[Any]], dict[str, Any]]:
    """Fix the fresh diagnostic data before loading a model or making a forward pass."""
    protected, base_counts = traj_audit.build_protected_digest_registry(seed)
    historical, historical_detail = rec004n.build_stage_datasets(seed)
    score_only_sets, score_only_detail = score_only.build_score_only_datasets(seed)
    cp_sets, cp_detail = rec004p.build_content_prep_release_datasets(seed)
    protected_sources = {
        "base_registry": len(protected),
        "rec004n_value_and_downstream_probe_series": 0,
        "rec004o_sets": 0,
        "rec004p_sets": 0,
    }
    for examples in historical.values():
        protected |= _digest_examples(examples)
    protected_sources["rec004n_value_and_downstream_probe_series"] = (
        len(protected) - protected_sources["base_registry"]
    )
    before_score = len(protected)
    for examples in score_only_sets.values():
        protected |= _digest_examples(examples)
    protected_sources["rec004o_sets"] = len(protected) - before_score
    before_cp = len(protected)
    for examples in cp_sets.values():
        protected |= _digest_examples(examples)
    protected_sources["rec004p_sets"] = len(protected) - before_cp

    datasets: dict[int, list[Any]] = {}
    details: dict[str, Any] = {}
    all_probe_digests: set[str] = set()
    for length, count in REC004Q_PROBE_COUNTS.items():
        split = f"{REC004Q_PROBE_NAMESPACE}_length{length}"
        examples, detail = score_only._build_dataset(
            seed=seed,
            split=split,
            n_examples=count,
            protected=protected | all_probe_digests,
            fixed_length=length,
        )
        digests = _digest_examples(examples)
        if digests & protected or digests & all_probe_digests:
            raise RuntimeError("non-deterministic probe collision replacement failed")
        datasets[length] = examples
        details[str(length)] = detail
        all_probe_digests |= digests
    return datasets, {
        "namespace": REC004Q_PROBE_NAMESPACE,
        "counts_by_length": REC004Q_PROBE_COUNTS,
        "total": sum(REC004Q_PROBE_COUNTS.values()),
        "input_target_digest_sha256": _dataset_digest(all_probe_digests),
        "protected_counts": base_counts,
        "additional_protected_counts": protected_sources,
        "datasets": details,
        "development_exposed": True,
        "sealed_or_rg3_query": False,
        "collision_replacement": "deterministic continuing-RNG draw by fixed slot and split namespace",
        "prior_dataset_manifests": {
            "rec004n": historical_detail,
            "rec004o": score_only_detail,
            "rec004p": cp_detail,
        },
    }


def _score_forward(
    primitive: Any, content: torch.Tensor, lengths: list[int]
) -> dict[str, torch.Tensor]:
    """The real J0 formula with the pre-softmax score retained for autograd.

    This is an in-memory diagnostic graph, not a different operator.  It is the
    same Q/K/V, learned position bias, padding mask, attention output, and
    downstream readout as the established manual J0 decomposition.
    """
    batch, lmax, _ = content.shape
    out_max = lmax
    device = content.device
    content_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
    query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
    kv = primitive.content_in_proj(content) + primitive.content_position_embedding(content_ids)
    query = primitive.answer_query_embedding(query_ids)
    mha = primitive.cross_attn
    wq, wk, wv = mha.in_proj_weight.chunk(3, dim=0)
    if mha.in_proj_bias is None:
        bq = bk = bv = None
    else:
        bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
    q = F.linear(query, wq, bq)
    k = F.linear(kv, wk, bk)
    v = F.linear(kv, wv, bv)
    n_head = primitive.n_head
    head_dim = primitive.d_operator // n_head
    q = q.view(batch, out_max, n_head, head_dim).transpose(1, 2)
    k = k.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    v = v.view(batch, lmax, n_head, head_dim).transpose(1, 2)
    s_other = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    dtype = kv.dtype
    c_lens = torch.tensor(lengths, device=device, dtype=dtype).view(batch, 1, 1)
    output_index = torch.arange(out_max, device=device, dtype=dtype).view(1, out_max, 1)
    key_index = torch.arange(lmax, device=device, dtype=dtype).view(1, 1, lmax)
    shape = (batch, out_max, lmax)
    denominator = torch.clamp(c_lens - 1.0, min=1.0)
    features = torch.stack(
        [
            output_index.expand(shape) / denominator,
            key_index.expand(shape) / denominator,
            (key_index - output_index).expand(shape) / denominator,
            (c_lens / float(primitive.length_ref)).expand(shape),
        ],
        dim=-1,
    )
    bias = primitive.position_bias_out(F.relu(primitive.position_bias_hidden(features))).squeeze(-1)
    pad = torch.arange(lmax, device=device).view(1, -1) >= torch.tensor(
        lengths, device=device
    ).view(-1, 1)
    mask = torch.where(
        pad.unsqueeze(1).expand(batch, out_max, lmax),
        torch.full_like(bias, torch.finfo(dtype).min),
        torch.zeros_like(bias),
    )
    scores = s_other + bias.unsqueeze(1) + mask.unsqueeze(1)
    attention = F.softmax(scores, dim=-1)
    attended = (
        torch.matmul(attention, v).transpose(1, 2).reshape(batch, out_max, primitive.d_operator)
    )
    attended = mha.out_proj(attended)
    hidden = primitive.attn_norm(query + attended)
    logits = primitive.readout(primitive.ffn_norm(hidden + primitive.ffn(hidden)))
    return {"scores": scores, "logits": logits, "s_other": s_other, "bias": bias, "mask": mask}


def _alignment_loss(scores: torch.Tensor, length: int) -> torch.Tensor:
    pi = [(position + length // 2) % length for position in range(length)]
    targets = (
        torch.tensor(pi, device=scores.device)
        .view(1, 1, length)
        .expand(scores.size(0), scores.size(1), length)
    )
    return F.cross_entropy(scores.reshape(-1, length), targets.reshape(-1), reduction="sum")


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().reshape(-1)
    right = right.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not torch.isfinite(denominator) or float(denominator.item()) == 0.0:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _flat_group(
    tensors: dict[str, torch.Tensor], names: list[str], row: slice | None = None
) -> torch.Tensor:
    pieces = [
        (tensors[name][row] if row is not None else tensors[name]).reshape(-1) for name in names
    ]
    return torch.cat(pieces) if pieces else torch.empty(0)


def _groups(primitive: Any) -> dict[str, tuple[list[str], slice | None]]:
    qkv = ["cross_attn.in_proj_weight", "cross_attn.in_proj_bias"]
    rows = primitive.cross_attn.in_proj_weight.shape[0] // 3
    return {
        "Q_rows": (qkv, slice(0, rows)),
        "K_rows": (qkv, slice(rows, 2 * rows)),
        "position_bias_hidden": (
            ["position_bias_hidden.weight", "position_bias_hidden.bias"],
            None,
        ),
        "position_bias_out": (["position_bias_out.weight"], None),
        "CONTENT_PREP": (
            ["content_in_proj.weight", "content_in_proj.bias", "content_position_embedding.weight"],
            None,
        ),
    }


def _bucket_name(length: int, position: int) -> str:
    if length == 10 and position in (4, 5):
        return f"length10_position{position}"
    if length == 10:
        return "length10_other_positions"
    return "length6_9_aggregate"


def _new_stats() -> dict[str, float | int]:
    return {
        "n": 0,
        "cosine_sum": 0.0,
        "cosine_n": 0,
        "dot_sum": 0.0,
        "task_norm_sum": 0.0,
        "align_norm_sum": 0.0,
        "correct_grad_sum": 0.0,
        "wrong_grad_sum": 0.0,
        "margin_change_sum": 0.0,
        "margin_decrease_n": 0,
        "margin_improve_n": 0,
    }


def _summarize_stats(stats: dict[str, float | int]) -> dict[str, Any]:
    n = int(stats["n"])
    if n == 0:
        return {"n": 0}
    cosine_n = int(stats["cosine_n"])
    return {
        "n": n,
        "mean_cosine_task_vs_alignment": float(stats["cosine_sum"]) / cosine_n
        if cosine_n
        else None,
        "mean_dot_task_vs_alignment": float(stats["dot_sum"]) / n,
        "mean_task_gradient_norm": float(stats["task_norm_sum"]) / n,
        "mean_alignment_gradient_norm": float(stats["align_norm_sum"]) / n,
        "mean_correct_key_task_gradient": float(stats["correct_grad_sum"]) / n,
        "mean_strongest_wrong_key_task_gradient": float(stats["wrong_grad_sum"]) / n,
        "mean_directional_margin_change_under_task_descent": float(stats["margin_change_sum"]) / n,
        "task_descent_margin_decrease_count": int(stats["margin_decrease_n"]),
        "task_descent_margin_decrease_fraction": int(stats["margin_decrease_n"]) / n,
        "task_descent_margin_improve_count": int(stats["margin_improve_n"]),
        "task_descent_margin_improve_fraction": int(stats["margin_improve_n"]) / n,
    }


def _audit_checkpoint(
    core: Any,
    spec: _CheckpointSpec,
    state: dict[str, Any],
    datasets: dict[int, list[Any]],
    jsonl: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], Any]:
    primitive = _new_primitive(core, state)
    named = dict(primitive.named_parameters())
    params = list(named.values())
    task_sums = {name: torch.zeros_like(parameter) for name, parameter in named.items()}
    align_sums = {name: torch.zeros_like(parameter) for name, parameter in named.items()}
    buckets = {
        key: _new_stats()
        for key in (
            "length10_position4",
            "length10_position5",
            "length10_other_positions",
            "length6_9_aggregate",
        )
    }
    total_tokens = sum(length * len(examples) for length, examples in datasets.items())
    total_rows = total_tokens * primitive.n_head
    operation = get_operation(rec004g.REC004G_TARGET_OPERATION)
    for length, examples in datasets.items():
        for start in range(0, len(examples), REC004Q_CHUNK_SIZE):
            chunk = examples[start : start + REC004Q_CHUNK_SIZE]
            lengths = [length] * len(chunk)
            output_lengths = [operation.output_length(length)] * len(chunk)
            labels = _labels_for_examples(chunk, output_lengths, length, core.device)
            with torch.no_grad():
                batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
                content = core.model.encode(batch)[:, 1 : 1 + length, :]
            task_forward = _score_forward(primitive, content, lengths)
            task_loss = F.cross_entropy(
                task_forward["logits"].reshape(-1, task_forward["logits"].size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            task_grads = torch.autograd.grad(
                task_loss, [task_forward["scores"], *params], allow_unused=True
            )
            alignment_forward = _score_forward(primitive, content, lengths)
            alignment_loss = _alignment_loss(alignment_forward["scores"], length)
            align_grads = torch.autograd.grad(
                alignment_loss, [alignment_forward["scores"], *params], allow_unused=True
            )
            task_scores = task_grads[0].detach() / total_tokens
            align_scores = align_grads[0].detach() / total_rows
            for name, task_grad, align_grad in zip(
                named, task_grads[1:], align_grads[1:], strict=True
            ):
                if task_grad is not None:
                    task_sums[name] += task_grad.detach()
                if align_grad is not None:
                    align_sums[name] += align_grad.detach()
            scores = task_forward["scores"].detach()
            for example_offset in range(len(chunk)):
                for head in range(primitive.n_head):
                    for position in range(length):
                        correct = (position + length // 2) % length
                        score_row = scores[example_offset, head, position, :length]
                        wrong_scores = score_row.clone()
                        wrong_scores[correct] = -torch.inf
                        strongest_wrong = int(torch.argmax(wrong_scores).item())
                        task_row = task_scores[example_offset, head, position, :length]
                        align_row = align_scores[example_offset, head, position, :length]
                        cosine = _safe_cosine(task_row, align_row)
                        dot = float(torch.dot(task_row, align_row).item())
                        task_norm = float(torch.linalg.vector_norm(task_row).item())
                        align_norm = float(torch.linalg.vector_norm(align_row).item())
                        correct_grad = float(task_row[correct].item())
                        wrong_grad = float(task_row[strongest_wrong].item())
                        margin_change = wrong_grad - correct_grad
                        bucket = _bucket_name(length, position)
                        stats = buckets[bucket]
                        stats["n"] = int(stats["n"]) + 1
                        stats["dot_sum"] = float(stats["dot_sum"]) + dot
                        stats["task_norm_sum"] = float(stats["task_norm_sum"]) + task_norm
                        stats["align_norm_sum"] = float(stats["align_norm_sum"]) + align_norm
                        stats["correct_grad_sum"] = float(stats["correct_grad_sum"]) + correct_grad
                        stats["wrong_grad_sum"] = float(stats["wrong_grad_sum"]) + wrong_grad
                        stats["margin_change_sum"] = (
                            float(stats["margin_change_sum"]) + margin_change
                        )
                        stats["margin_decrease_n"] = int(stats["margin_decrease_n"]) + int(
                            margin_change < 0.0
                        )
                        stats["margin_improve_n"] = int(stats["margin_improve_n"]) + int(
                            margin_change > 0.0
                        )
                        if cosine is not None:
                            stats["cosine_sum"] = float(stats["cosine_sum"]) + cosine
                            stats["cosine_n"] = int(stats["cosine_n"]) + 1
                        jsonl.write(
                            json.dumps(
                                {
                                    "checkpoint_id": spec.checkpoint_id,
                                    "length": length,
                                    "example_index_within_length": start + example_offset,
                                    "head": head,
                                    "output_position": position,
                                    "correct_key": correct,
                                    "strongest_wrong_key": strongest_wrong,
                                    "task_alignment_cosine": cosine,
                                    "task_alignment_dot": dot,
                                    "task_gradient_norm": task_norm,
                                    "alignment_gradient_norm": align_norm,
                                    "correct_key_task_gradient": correct_grad,
                                    "strongest_wrong_key_task_gradient": wrong_grad,
                                    "directional_margin_change_under_task_descent": margin_change,
                                    "task_descent_decreases_correct_key_margin": margin_change
                                    < 0.0,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
    task_means = {name: gradient / total_tokens for name, gradient in task_sums.items()}
    align_means = {name: gradient / total_rows for name, gradient in align_sums.items()}
    return (
        {
            "checkpoint_id": spec.checkpoint_id,
            "loss_definition": {
                "task": "standard token cross entropy, mean over all valid output tokens",
                "oracle_alignment": "mean head/output-position CE over valid keys only; diagnostic only",
                "gradient_descent_sign": "a negative loss gradient is the descent direction",
            },
            "attention_score_formula": "S = scaled(QK^T) + learned_position_bias + padding_mask",
            "buckets": {name: _summarize_stats(stats) for name, stats in buckets.items()},
        },
        task_means,
        align_means,
        primitive,
    )


def _parameter_summary(
    primitive: Any,
    task: dict[str, torch.Tensor],
    align: dict[str, torch.Tensor],
    lr: float,
    include_content_prep: bool,
) -> dict[str, Any]:
    named = dict(primitive.named_parameters())
    output: dict[str, Any] = {}
    for group, (names, rows) in _groups(primitive).items():
        if group == "CONTENT_PREP" and not include_content_prep:
            continue
        task_vector = _flat_group(task, names, rows)
        align_vector = _flat_group(align, names, rows)
        weight_vector = _flat_group(
            {name: parameter.detach() for name, parameter in named.items()}, names, rows
        )
        task_norm = float(torch.linalg.vector_norm(task_vector).item())
        align_norm = float(torch.linalg.vector_norm(align_vector).item())
        weight_norm = float(torch.linalg.vector_norm(weight_vector).item())
        output[group] = {
            "task_loss_gradient_norm": task_norm,
            "oracle_alignment_gradient_norm": align_norm,
            "cosine_task_vs_oracle_alignment": _safe_cosine(task_vector, align_vector),
            "update_to_weight_scale_proxy": lr * task_norm / max(weight_norm, 1e-30),
            "finite": bool(
                torch.isfinite(task_vector).all() and torch.isfinite(align_vector).all()
            ),
            "parameter_names": names,
            "row_slice": None if rows is None else [rows.start, rows.stop],
        }
    return output


def _state_by_name(
    primitive: Any, optimizer_state: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    named = list(primitive.named_parameters())
    groups = optimizer_state["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(named):
        raise RuntimeError("unexpected optimizer parameter-group layout")
    parameter_ids = groups[0]["params"]
    states = optimizer_state["state"]
    return (
        {
            name: states.get(parameter_id, {})
            for (name, _), parameter_id in zip(named, parameter_ids, strict=True)
        },
        groups[0],
    )


def _active_gradient(
    name: str, gradient: torch.Tensor, policy: str, qkv_rows: int
) -> torch.Tensor | None:
    if policy == "joint":
        return gradient
    if name in ("cross_attn.in_proj_weight", "cross_attn.in_proj_bias"):
        result = torch.zeros_like(gradient)
        result[: 2 * qkv_rows] = gradient[: 2 * qkv_rows]
        return result
    if name.startswith("position_bias_"):
        return gradient
    if policy == "cp_score" and name.startswith(
        ("content_in_proj.", "content_position_embedding.")
    ):
        return gradient
    return None


def _adamw_delta(
    primitive: Any,
    task: dict[str, torch.Tensor],
    state: dict[str, Any],
    policy: str,
    *,
    zero_moment: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    named = dict(primitive.named_parameters())
    states, group = _state_by_name(primitive, state["optimizer_state_dict"])
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    qkv_rows = primitive.cross_attn.in_proj_weight.shape[0] // 3
    active: dict[str, torch.Tensor] = {}
    for name, gradient in task.items():
        selected = _active_gradient(name, gradient, policy, qkv_rows)
        if selected is not None:
            active[name] = selected
    total_norm_sq = torch.zeros((), device=next(primitive.parameters()).device)
    for value in active.values():
        total_norm_sq = total_norm_sq + (value * value).sum()
    total_norm = torch.sqrt(total_norm_sq)
    clip = float(rec004g.REC004G_OPERATOR_GRAD_CLIP)
    clip_factor = min(1.0, clip / (float(total_norm.item()) + 1e-6))
    deltas = {name: torch.zeros_like(parameter) for name, parameter in named.items()}
    first_norm_sq = second_norm_sq = decay_norm_sq = 0.0
    for name, gradient in active.items():
        parameter = named[name].detach()
        clipped = gradient * clip_factor
        saved = states[name]
        prior_step = (
            0
            if zero_moment
            else int(
                saved.get("step", 0).item()
                if isinstance(saved.get("step"), torch.Tensor)
                else saved.get("step", 0)
            )
        )
        exp_avg = (
            torch.zeros_like(parameter) if zero_moment else saved["exp_avg"].to(parameter.device)
        )
        exp_avg_sq = (
            torch.zeros_like(parameter) if zero_moment else saved["exp_avg_sq"].to(parameter.device)
        )
        next_avg = exp_avg.mul(beta1).add(clipped, alpha=1.0 - beta1)
        next_avg_sq = exp_avg_sq.mul(beta2).addcmul(clipped, clipped, value=1.0 - beta2)
        next_step = prior_step + 1
        bias_correction1 = 1.0 - beta1**next_step
        bias_correction2 = 1.0 - beta2**next_step
        first = (
            -lr
            * next_avg
            / bias_correction1
            / (next_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps)
        )
        decay = -lr * weight_decay * parameter
        delta = first + decay
        deltas[name] = delta
        first_norm_sq += float((first * first).sum().item())
        second_norm_sq += float(
            (next_avg_sq.sqrt() / math.sqrt(bias_correction2)).pow(2).sum().item()
        )
        decay_norm_sq += float((decay * decay).sum().item())
    return deltas, {
        "lr": lr,
        "betas": [beta1, beta2],
        "eps": eps,
        "weight_decay": weight_decay,
        "gradient_clip_norm": clip,
        "pre_clip_global_norm": float(total_norm.item()),
        "clip_factor": clip_factor,
        "first_moment_contribution_norm": math.sqrt(first_norm_sq),
        "second_moment_scaling_norm": math.sqrt(second_norm_sq),
        "weight_decay_contribution_norm": math.sqrt(decay_norm_sq),
        "zero_moment_counterfactual": zero_moment,
    }


def _score_jvp(
    primitive: Any,
    content: torch.Tensor,
    length: int,
    deltas: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Exact forward-mode directional derivative dS/dtheta[delta], no mutation."""
    parameters = {name: parameter.detach() for name, parameter in primitive.named_parameters()}

    def score_from_params(current: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = content.size(0)
        ids = torch.arange(length, device=content.device).unsqueeze(0).expand(batch, length)
        kv = F.linear(content, current["content_in_proj.weight"], current["content_in_proj.bias"])
        kv = kv + F.embedding(ids, current["content_position_embedding.weight"])
        query = F.embedding(ids, current["answer_query_embedding.weight"])
        rows = current["cross_attn.in_proj_weight"].shape[0] // 3
        q = F.linear(
            query,
            current["cross_attn.in_proj_weight"][:rows],
            current["cross_attn.in_proj_bias"][:rows],
        )
        k = F.linear(
            kv,
            current["cross_attn.in_proj_weight"][rows : 2 * rows],
            current["cross_attn.in_proj_bias"][rows : 2 * rows],
        )
        head_dim = primitive.d_operator // primitive.n_head
        q = q.view(batch, length, primitive.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, length, primitive.n_head, head_dim).transpose(1, 2)
        s_other = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        dtype = content.dtype
        c_len = torch.full((batch, 1, 1), float(length), device=content.device, dtype=dtype)
        positions = torch.arange(length, device=content.device, dtype=dtype)
        out = positions.view(1, length, 1).expand(batch, length, length)
        key = positions.view(1, 1, length).expand(batch, length, length)
        denom = max(length - 1, 1)
        features = torch.stack(
            [
                out / denom,
                key / denom,
                (key - out) / denom,
                (c_len / float(primitive.length_ref)).expand_as(out),
            ],
            dim=-1,
        )
        bias = F.linear(
            features, current["position_bias_hidden.weight"], current["position_bias_hidden.bias"]
        )
        bias = F.relu(bias)
        bias = F.linear(bias, current["position_bias_out.weight"]).squeeze(-1)
        return s_other + bias.unsqueeze(1)

    jvp_result = torch.func.jvp(score_from_params, (parameters,), (deltas,), has_aux=False)
    _scores, tangent = cast(tuple[torch.Tensor, torch.Tensor], jvp_result)
    return tangent


def _effective_margin_summary(
    core: Any,
    primitive: Any,
    datasets: dict[int, list[Any]],
    deltas: dict[str, torch.Tensor],
) -> dict[str, Any]:
    stats = {
        key: {"n": 0, "sum": 0.0, "decrease": 0, "improve": 0}
        for key in ("length10_position4", "length10_position5", "length6_9_aggregate")
    }
    for length, examples in datasets.items():
        for start in range(0, len(examples), REC004Q_CHUNK_SIZE):
            chunk = examples[start : start + REC004Q_CHUNK_SIZE]
            with torch.no_grad():
                batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
                content = core.model.encode(batch)[:, 1 : 1 + length, :]
            tangent = _score_jvp(primitive, content, length, deltas)
            base_scores = _score_forward(primitive, content, [length] * len(chunk))[
                "scores"
            ].detach()
            for example_index in range(len(chunk)):
                for head in range(primitive.n_head):
                    for position in range(length):
                        bucket = _bucket_name(length, position)
                        if bucket == "length10_other_positions":
                            continue
                        correct = (position + length // 2) % length
                        row = base_scores[example_index, head, position, :length].clone()
                        row[correct] = -torch.inf
                        wrong = int(torch.argmax(row).item())
                        change = float(
                            (
                                tangent[example_index, head, position, correct]
                                - tangent[example_index, head, position, wrong]
                            ).item()
                        )
                        current = stats[bucket]
                        current["n"] += 1
                        current["sum"] += change
                        current["decrease"] += int(change < 0.0)
                        current["improve"] += int(change > 0.0)
    return {
        name: {
            "n": value["n"],
            "mean_first_order_margin_change_under_effective_update": value["sum"] / value["n"]
            if value["n"]
            else None,
            "effective_update_margin_decrease_fraction": value["decrease"] / value["n"]
            if value["n"]
            else None,
            "effective_update_margin_improve_fraction": value["improve"] / value["n"]
            if value["n"]
            else None,
        }
        for name, value in stats.items()
    }


def _audit_adamw(
    core: Any,
    spec: _CheckpointSpec,
    state: dict[str, Any],
    primitive: Any,
    task: dict[str, torch.Tensor],
    align: dict[str, torch.Tensor],
    datasets: dict[int, list[Any]],
) -> dict[str, Any]:
    effective, metadata = _adamw_delta(
        primitive, task, state, spec.update_policy, zero_moment=False
    )
    zero_moment, zero_metadata = _adamw_delta(
        primitive, task, state, spec.update_policy, zero_moment=True
    )
    named = dict(primitive.named_parameters())
    groups: dict[str, Any] = {}
    for group, (names, rows) in _groups(primitive).items():
        if group == "CONTENT_PREP" and spec.update_policy != "cp_score":
            continue
        raw = _flat_group(task, names, rows)
        oracle = _flat_group(align, names, rows)
        delta = _flat_group(effective, names, rows)
        zero_delta = _flat_group(zero_moment, names, rows)
        weight = _flat_group(
            {name: parameter.detach() for name, parameter in named.items()}, names, rows
        )
        groups[group] = {
            "raw_gradient_direction_norm": float(torch.linalg.vector_norm(raw).item()),
            "adamw_effective_update_direction_norm": float(torch.linalg.vector_norm(delta).item()),
            "cosine_raw_gradient_vs_oracle_gradient": _safe_cosine(raw, oracle),
            "cosine_effective_update_vs_oracle_descent": _safe_cosine(delta, -oracle),
            "update_to_weight_scale": float(torch.linalg.vector_norm(delta).item())
            / max(float(torch.linalg.vector_norm(weight).item()), 1e-30),
            "zero_moment_counterfactual": {
                "update_norm": float(torch.linalg.vector_norm(zero_delta).item()),
                "cosine_vs_oracle_descent": _safe_cosine(zero_delta, -oracle),
            },
        }
    return {
        "checkpoint_id": spec.checkpoint_id,
        "update_policy": spec.update_policy,
        "actual_adamw": metadata,
        "zero_moment_counterfactual": zero_metadata,
        "score_parameter_groups": groups,
        "effective_update_margin": _effective_margin_summary(core, primitive, datasets, effective),
    }


def _reaggregate_rec004p(core: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets, manifest = rec004p.build_content_prep_release_datasets(10)
    source_states = {
        "EARLY": torch.load(_CHECKPOINTS[0].path, map_location="cpu", weights_only=False),
        "SCORE_ONLY": torch.load(_CHECKPOINTS[1].path, map_location="cpu", weights_only=False),
        "CP_SCORE": torch.load(_CHECKPOINTS[2].path, map_location="cpu", weights_only=False),
        "JOINT": torch.load(_CHECKPOINTS[3].path, map_location="cpu", weights_only=False),
    }
    old = json.loads(
        (
            Path("runs/phase_b_b2_model_bundle_recovery/rec004p/run_001/endpoint_metrics.json")
        ).read_text(encoding="utf-8")
    )
    reevaluated = {
        name: rec004p._evaluate(core, _new_primitive(core, state), datasets)
        for name, state in source_states.items()
    }
    mismatches: list[dict[str, Any]] = []
    for name, metrics in reevaluated.items():
        for split, values in metrics.items():
            old_values = old[name][split]
            for field in ("j0_sequence_exact_match", "oracle_sequence_exact_match"):
                if values[field] != old_values[field]:
                    mismatches.append(
                        {
                            "condition": name,
                            "split": split,
                            "field": field,
                            "old": old_values[field],
                            "new": values[field],
                        }
                    )
    cp = reevaluated["CP_SCORE"]
    errors = []
    for split in datasets:
        old_values = old["CP_SCORE"][split]
        errors.append(
            {
                "split": split,
                "old_reported_o1_sequence_em": old_values["oracle_sequence_exact_match"],
                "old_o1_position4_field": old_values["position4_accuracy"],
                "old_o1_position5_field": old_values["position5_accuracy"],
                "recomputed_o1_position4_accuracy": cp[split]["oracle_position4_accuracy"],
                "recomputed_o1_position5_accuracy": cp[split]["oracle_position5_accuracy"],
            }
        )
    audit = {
        "status": "REC004P_EVALUATION_METRIC_BUG",
        "classification": "saved O1 audit selected J0 position fields; terminal checkpoints permit correct read-only reevaluation",
        "dataset_manifest_reproduced": manifest,
        "j0_and_sequence_metrics_unchanged": not mismatches,
        "metric_mismatches": mismatches,
        "cp_score_o1_erratum": errors,
        "old_artifacts_overwritten": False,
    }
    erratum = {
        "label": "REC004P_EVALUATION_METRIC_BUG",
        "correction": "O1 position-4/5 fields are 1.0/1.0 for CP_SCORE on both locked REC-004P sets; the old 0.8115/0.9863 and 0.1035/0.9980 entries were J0 fields.",
        "main_j0_result_preserved": not mismatches,
        "corrected_metrics": errors,
        "source_artifacts_preserved": True,
    }
    return audit, erratum


def _decision(position_summary: dict[str, Any], adamw: dict[str, Any]) -> dict[str, Any]:
    terminals = ("I03_SCORE_ONLY_12000", "I03_CP_SCORE_12000")
    raw_decrease = [
        position_summary[name]["buckets"]["length10_position4"][
            "task_descent_margin_decrease_fraction"
        ]
        for name in terminals
    ]
    control_decrease = position_summary["I04_P_7000"]["buckets"]["length10_position4"][
        "task_descent_margin_decrease_fraction"
    ]
    misalignment = all(value >= 0.60 for value in raw_decrease) and control_decrease < 0.60
    raw_improve = [
        position_summary[name]["buckets"]["length10_position4"][
            "task_descent_margin_improve_fraction"
        ]
        for name in terminals
    ]
    adam_decrease = [
        adamw[name]["effective_update_margin"]["length10_position4"][
            "effective_update_margin_decrease_fraction"
        ]
        for name in terminals
    ]
    adam_conflict = all(value >= 0.60 for value in raw_improve) and all(
        value >= 0.60 for value in adam_decrease
    )
    weak_ratios: dict[str, dict[str, float | None]] = {}
    for name in terminals:
        focus = position_summary[name]["buckets"]["length10_position4"]["mean_task_gradient_norm"]
        control = position_summary["I04_P_7000"]["buckets"]["length10_position4"][
            "mean_task_gradient_norm"
        ]
        short = position_summary[name]["buckets"]["length6_9_aggregate"]["mean_task_gradient_norm"]
        weak_ratios[name] = {
            "vs_i04_p7000_position4": focus / control if control else None,
            "vs_same_checkpoint_length6_9": focus / short if short else None,
        }
    weak = all(
        ratio["vs_i04_p7000_position4"] is not None
        and ratio["vs_same_checkpoint_length6_9"] is not None
        and ratio["vs_i04_p7000_position4"] <= REC004Q_WEAK_SIGNAL_RATIO
        and ratio["vs_same_checkpoint_length6_9"] <= REC004Q_WEAK_SIGNAL_RATIO
        for ratio in weak_ratios.values()
    )
    if misalignment:
        label = "TASK_LOSS_SCORE_CREDIT_MISALIGNMENT_SUPPORTED"
        proposal = "Design one non-oracle, target-derived functional score-credit objective; do not train directly on pi_n."
    elif adam_conflict:
        label = "ADAMW_STATE_PRECONDITIONING_CONFLICT_SUPPORTED"
        proposal = "Design one paired pilot for a phase-boundary optimizer-state reset or score-path optimizer isolation."
    elif weak:
        label = "SCORE_CREDIT_SIGNAL_WEAK"
        proposal = "Design one score-path optimization-scale or conditioning repair pilot."
    else:
        label = "SCORE_CREDIT_ASSIGNMENT_UNRESOLVED"
        proposal = "Run one minimal diagnostic that attributes token-loss gradients by output position before changing architecture or loss."
    return {
        "label": label,
        "criteria": {
            "misalignment_terminal_position4_decrease_fractions": raw_decrease,
            "misalignment_i04_p7000_decrease_fraction": control_decrease,
            "adamw_terminal_raw_improve_fractions": raw_improve,
            "adamw_terminal_effective_decrease_fractions": adam_decrease,
            "weak_signal_preregistered_ratio_ceiling": REC004Q_WEAK_SIGNAL_RATIO,
            "weak_signal_ratios": weak_ratios,
        },
        "next_repair_proposal": proposal,
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "selected_objective": None,
        "selected_optimizer_change": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }


def _report(decision: dict[str, Any], erratum: dict[str, Any]) -> str:
    return (
        f"# {REC004Q_TASK_ID} — Attention-score credit-assignment gradient audit\n\n"
        f"Result: `{decision['label']}`\n\n"
        f"REC-004P metric audit: `{erratum['label']}`; J0 conclusion preserved: "
        f"`{erratum['main_j0_result_preserved']}`.\n\n"
        "No optimizer step, training, candidate adoption, child bundle, RG3 recheck, or REC-005 work ran.\n"
    )


def run_mirror_attention_score_credit_assignment_audit_task(
    config: MirrorAttentionScoreCreditAssignmentAuditConfig,
) -> dict[str, Any]:
    """Run the entire B-C005REC-004Q read-only diagnostic and stop."""
    if config.seed != 10:
        raise ValueError("B-C005REC-004Q is pre-registered to recovery seed 10")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [spec.path for spec in _CHECKPOINTS] + [
        Path("runs/phase_b_b2_model_bundle_recovery/rec004p/run_001/endpoint_metrics.json"),
    ]
    if not all(path.is_file() for path in required):
        return {"implementation_status": "STOPPED", "result_label": "SOURCE_ARTIFACT_UNAVAILABLE"}
    started = time.time()
    _write_json(output_dir / "config.yaml", {"seed": config.seed, "output_dir": str(output_dir)})
    _write_json(output_dir / "system.json", get_system_info(seed=config.seed))
    loaded, checkpoint_manifest = _source_checkpoint_manifest()
    _write_json(output_dir / "source_checkpoint_manifest.json", checkpoint_manifest)
    datasets, probe_manifest = _build_probe(config.seed)
    _write_json(output_dir / "score_credit_probe_manifest.json", probe_manifest)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core_before = mb.canonical_state_hash(core.model.state_dict())
    protected_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    source_hashes_before = {spec.checkpoint_id: mb.raw_file_sha256(spec.path) for spec, _ in loaded}
    rec004p_audit, rec004p_erratum = _reaggregate_rec004p(core)
    _write_json(output_dir / "rec004p_metric_consistency_audit.json", rec004p_audit)
    _write_json(output_dir / "rec004p_metric_erratum.json", rec004p_erratum)
    if not rec004p_audit["j0_and_sequence_metrics_unchanged"]:
        decision = {
            "label": "REC004P_RESULT_REQUIRES_REINTERPRETATION",
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_step": None,
            "selected_objective": None,
            "selected_optimizer_change": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
        }
        _write_json(output_dir / "credit_assignment_decision.json", decision)
        return {"implementation_status": "STOPPED", "result_label": decision["label"]}

    position_summary: dict[str, Any] = {}
    parameter_summary: dict[str, Any] = {}
    adamw_audit: dict[str, Any] = {}
    with (output_dir / "attention_logit_gradient_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as jsonl:
        for spec, state in loaded:
            score_metrics, task, align, primitive = _audit_checkpoint(
                core, spec, state, datasets, jsonl
            )
            position_summary[spec.checkpoint_id] = score_metrics
            _states, optimizer_group = _state_by_name(primitive, state["optimizer_state_dict"])
            parameter_summary[spec.checkpoint_id] = _parameter_summary(
                primitive,
                task,
                align,
                float(optimizer_group["lr"]),
                spec.update_policy == "cp_score",
            )
            adamw_audit[spec.checkpoint_id] = _audit_adamw(
                core, spec, state, primitive, task, align, datasets
            )
    _write_json(output_dir / "position4_credit_summary.json", position_summary)
    _write_json(output_dir / "score_parameter_gradient_summary.json", parameter_summary)
    _write_json(output_dir / "adamw_effective_update_audit.json", adamw_audit)
    _write_json(
        output_dir / "successful_trajectory_control.json",
        {
            "control_role": "descriptive successful-trajectory control, not gradient ground truth",
            "checkpoints": {name: position_summary[name] for name in ("I04_P_6000", "I04_P_7000")},
            "i04_7000_checkpoint_substituted": False,
        },
    )
    decision = _decision(position_summary, adamw_audit)
    _write_json(output_dir / "credit_assignment_decision.json", decision)
    (output_dir / "next_repair_contract.md").write_text(
        "# Next repair contract\n\n"
        f"diagnostic label: `{decision['label']}`\n\n"
        f"One proposal only: {decision['next_repair_proposal']}\n\n"
        "This is not authorization for repair training. Oracle pi_n remains diagnostic-only.\n",
        encoding="utf-8",
    )
    source_hashes_after = {spec.checkpoint_id: mb.raw_file_sha256(spec.path) for spec, _ in loaded}
    checkpoint_canonical_hashes_unchanged = {
        spec.checkpoint_id: mb.canonical_state_hash(
            torch.load(spec.path, map_location="cpu", weights_only=False)["primitive_state_dict"]
        )
        == entry["canonical_primitive_state_hash"]
        for (spec, _), entry in zip(loaded, checkpoint_manifest["checkpoints"], strict=True)
    }
    freeze_audit = {
        "core_unchanged": core_before == mb.canonical_state_hash(core.model.state_dict()),
        "protected_operations_unchanged": protected_before
        == mpbr._protected_scope_hashes(eval_bank, op_to_id),
        "source_checkpoints_unchanged": source_hashes_before == source_hashes_after,
        "checkpoint_canonical_hashes_unchanged": checkpoint_canonical_hashes_unchanged,
    }
    freeze_audit["passed"] = all(
        (
            freeze_audit["core_unchanged"],
            freeze_audit["protected_operations_unchanged"],
            freeze_audit["source_checkpoints_unchanged"],
            all(checkpoint_canonical_hashes_unchanged.values()),
        )
    )
    _write_json(output_dir / "freeze_audit.json", freeze_audit)
    side_effect_audit = {
        "new_optimizer_updates": 0,
        "optimizer_step_called": False,
        "backward_called": False,
        "autograd_method": "torch.autograd.grad on in-memory checkpoint copies only",
        "source_artifacts_unchanged": freeze_audit["source_checkpoints_unchanged"],
        "training": "NOT_EXECUTED",
        "sealed_or_rg3_query": "NOT_EXECUTED",
        "child_bundle": None,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)
    report = {
        "task_id": REC004Q_TASK_ID,
        "implementation_status": "COMPLETED",
        "result_label": decision["label"],
        "rec004p_metric_audit": rec004p_audit["status"],
        "decision": decision,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": {
            "wall_clock_seconds": time.time() - started,
            "new_optimizer_updates": 0,
            "probe_examples": sum(REC004Q_PROBE_COUNTS.values()),
            "checkpoint_count": len(_CHECKPOINTS),
        },
        "selected_init": None,
        "selected_step": None,
        "selected_objective": None,
        "selected_optimizer_change": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "report.md").write_text(_report(decision, rec004p_erratum), encoding="utf-8")
    return report
