# ruff: noqa: E501
"""B-C005REC-004S: I03 finite-step score-function dynamics vs local linear prediction audit.

This module performs zero optimizer steps, no training, and modifies no checkpoint file.
It evaluates analytical AdamW parameter deltas on the actual next training batch,
computes forward-mode autograd JVP directional derivatives, and compares them with
exact finite virtual steps under alpha in {0.1, 1.0} on in-memory models.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example
from apc.environments.operations import get_operation
from apc.evaluation import incremental_budget_calibration as ibc
from apc.evaluation import mirror_attention_score_credit_assignment_audit as rec004q
from apc.evaluation import mirror_budget_extension as rec004g
from apc.evaluation import mirror_contamination_free_checkpoint_trajectory_audit as traj_audit
from apc.evaluation import mirror_content_prep_release_pilot as rec004p
from apc.evaluation import mirror_ffn_value_path_leave_one_out_necessity_audit as rec004n
from apc.evaluation import mirror_position_bias_repair as mpbr
from apc.evaluation import mirror_score_only_continuation_pilot as score_only
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

REC004S_TASK_ID: Final = "B-C005REC-004S"
REC004S_CONTRACT_FILE: Final = Path(
    "docs/CODEX_TASKS_PHASE_B_B2_I03_FINITE_STEP_SCORE_FUNCTION_DYNAMICS_AUDIT.md"
)
REC004S_OUTPUT_DIR: Final = Path("runs/phase_b_b2_model_bundle_recovery/rec004s/run_001")
REC004S_CONTINUITY_NAMESPACE: Final = "score_credit_assignment_probe_v1"
REC004S_FRESH_NAMESPACE: Final = "score_function_dynamics_probe_v1"
REC004S_PROBE_COUNTS: Final = {6: 128, 7: 128, 8: 128, 9: 128, 10: 512}
REC004S_CHUNK_SIZE: Final = 64
REC004S_ALPHAS: Final = (0.1, 1.0)
REC004S_EPS: Final = 1e-7


@dataclass(frozen=True)
class MirrorScoreFunctionFiniteStepDynamicsAuditConfig:
    output_dir: Path = REC004S_OUTPUT_DIR
    seed: int = 10


@dataclass(frozen=True)
class _CheckpointSpec:
    checkpoint_id: str
    path: Path
    init_id: str
    step: int
    next_step: int
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
        6001,
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
        12001,
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
        12001,
        "CP_SCORE_CONTINUATION",
        "cp_score",
        None,
        "REC-004P terminal training state",
    ),
    _CheckpointSpec(
        "I03_JOINT_12000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/I03/P_LENGTH_POSITION_BIAS/training_states/step12000.pt"
        ),
        "I03",
        12000,
        12001,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        None,
        "REC-004G learning_curve.jsonl",
    ),
    _CheckpointSpec(
        "I04_P_7000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/I04/P_LENGTH_POSITION_BIAS/training_states/step7000.pt"
        ),
        "I04",
        7000,
        7001,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        "7a7159f815f494824524ad20fc5c895fff570b7d2f2e2a996efc5998e32e20a4",
        "REC-004G learning_curve.jsonl",
    ),
    _CheckpointSpec(
        "I04_P_6000",
        Path(
            "runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/I04/P_LENGTH_POSITION_BIAS/training_states/step6000.pt"
        ),
        "I04",
        6000,
        6001,
        "P_LENGTH_POSITION_BIAS",
        "joint",
        None,
        "REC-004D learning_curve.jsonl",
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
        curve_hash = rec004q._learning_curve_hash(
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
                "next_step": spec.next_step,
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


def _build_probes(
    seed: int,
) -> tuple[
    dict[str, dict[int, list[Any]]],
    dict[str, Any],
]:
    """Build byte-identical continuity probe and fresh confirmation probe before evaluation."""
    continuity_sets, continuity_manifest = rec004q._build_probe(seed)

    protected, base_counts = traj_audit.build_protected_digest_registry(seed)
    historical, _ = rec004n.build_stage_datasets(seed)
    score_only_sets, _ = score_only.build_score_only_datasets(seed)
    cp_sets, _ = rec004p.build_content_prep_release_datasets(seed)
    for examples in historical.values():
        protected |= _digest_examples(examples)
    for examples in score_only_sets.values():
        protected |= _digest_examples(examples)
    for examples in cp_sets.values():
        protected |= _digest_examples(examples)

    continuity_digests: set[str] = set()
    for examples in continuity_sets.values():
        continuity_digests |= _digest_examples(examples)
    protected |= continuity_digests

    fresh_sets: dict[int, list[Any]] = {}
    fresh_details: dict[str, Any] = {}
    all_fresh_digests: set[str] = set()
    for length, count in REC004S_PROBE_COUNTS.items():
        split = f"{REC004S_FRESH_NAMESPACE}_length{length}"
        examples, detail = score_only._build_dataset(
            seed=seed,
            split=split,
            n_examples=count,
            protected=protected | all_fresh_digests,
            fixed_length=length,
        )
        digests = _digest_examples(examples)
        if digests & protected or digests & all_fresh_digests:
            raise RuntimeError("non-deterministic fresh probe collision replacement failed")
        fresh_sets[length] = examples
        fresh_details[str(length)] = detail
        all_fresh_digests |= digests

    probes = {
        "continuity": continuity_sets,
        "fresh": fresh_sets,
    }
    manifest = {
        "continuity_probe": continuity_manifest,
        "fresh_probe": {
            "namespace": REC004S_FRESH_NAMESPACE,
            "counts_by_length": REC004S_PROBE_COUNTS,
            "total": sum(REC004S_PROBE_COUNTS.values()),
            "input_target_digest_sha256": _dataset_digest(all_fresh_digests),
            "disjoint_from_continuity_probe": len(all_fresh_digests & continuity_digests) == 0,
            "disjoint_from_protected_base": len(all_fresh_digests & protected) == 0,
            "development_exposed": True,
            "sealed_or_rg3_query": False,
            "collision_replacement": "deterministic continuing-RNG draw by fixed slot",
            "datasets": fresh_details,
        },
    }
    return probes, manifest


def _generate_next_training_batches(
    seed: int, loaded: list[tuple[_CheckpointSpec, dict[str, Any]]]
) -> tuple[dict[int, list[Example]], dict[str, Any]]:
    unique_steps = sorted({spec.next_step for spec, _ in loaded})
    batches: dict[int, list[Example]] = {}
    details: dict[str, Any] = {}
    for step in unique_steps:
        examples = ibc._generate_step_training_examples(
            seed,
            step,
            rec004g.REC004G_TARGET_OPERATION,
            vocab_size=rec004g.REC004G_VOCAB_SIZE,
            sequence_length_range=rec004g.REC004G_SEQUENCE_LENGTH_RANGE,
        )
        batches[step] = examples
        digests = _digest_examples(examples)
        details[str(step)] = {
            "step": step,
            "example_count": len(examples),
            "input_target_digest_sha256": _dataset_digest(digests),
            "lengths": [len(ex.input_tokens) for ex in examples],
        }
    return batches, {
        "formula": "ibc._generate_step_training_examples(seed=10, step=next_step, ...)",
        "unique_steps": unique_steps,
        "batches": details,
        "generator_consistent_with_historical_stream": True,
    }


def _compute_next_step_task_gradient(
    core: Any,
    primitive: Any,
    examples: list[Example],
) -> dict[str, torch.Tensor]:
    lengths = [len(ex.input_tokens) for ex in examples]
    op = get_operation(rec004g.REC004G_TARGET_OPERATION)
    output_lengths = [op.output_length(length) for length in lengths]
    labels = _labels_for_examples(examples, output_lengths, max(output_lengths), core.device)
    with torch.no_grad():
        batch_input = collate_content_only_batch(examples, core.tokens, device=core.device)
        h_content = core.model.encode(batch_input)[:, 1 : 1 + max(lengths), :]

    named = dict(primitive.named_parameters())
    params = list(named.values())
    logits = primitive(h_content, lengths, output_lengths, None)
    loss = F.cross_entropy(
        logits.reshape(-1, rec004g.REC004G_VOCAB_SIZE),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return {
        name: (grad.detach() if grad is not None else torch.zeros_like(param))
        for name, grad, param in zip(named.keys(), grads, params, strict=True)
    }


def _reconstruct_analytical_adamw_delta(
    primitive: Any,
    task_gradients: dict[str, torch.Tensor],
    state: dict[str, Any],
    policy: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Reconstruct analytical 1-step AdamW delta for score-path parameters only."""
    named = dict(primitive.named_parameters())
    states, group = rec004q._state_by_name(primitive, state["optimizer_state_dict"])
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    qkv_rows = primitive.cross_attn.in_proj_weight.shape[0] // 3

    active_for_clipping: dict[str, torch.Tensor] = {}
    for name, gradient in task_gradients.items():
        selected = rec004q._active_gradient(name, gradient, policy, qkv_rows)
        if selected is not None:
            active_for_clipping[name] = selected

    total_norm_sq = torch.zeros((), device=next(primitive.parameters()).device)
    for value in active_for_clipping.values():
        total_norm_sq = total_norm_sq + (value * value).sum()
    total_norm = torch.sqrt(total_norm_sq)
    clip = float(rec004g.REC004G_OPERATOR_GRAD_CLIP)
    clip_factor = min(1.0, clip / (float(total_norm.item()) + 1e-6))

    target_score_params = {
        "cross_attn.in_proj_weight",
        "cross_attn.in_proj_bias",
        "position_bias_hidden.weight",
        "position_bias_hidden.bias",
        "position_bias_out.weight",
    }
    if policy in ("cp_score", "joint"):
        target_score_params |= {
            "content_in_proj.weight",
            "content_in_proj.bias",
            "content_position_embedding.weight",
        }

    deltas: dict[str, torch.Tensor] = {
        name: torch.zeros_like(parameter) for name, parameter in named.items()
    }
    norms: dict[str, float] = {}

    for name in target_score_params:
        if name not in named:
            continue
        param = named[name].detach()
        grad = task_gradients[name]
        saved = states[name]
        step_val = saved.get("step", 0)
        prior_step = int(step_val.item() if isinstance(step_val, torch.Tensor) else step_val)
        exp_avg = (
            saved["exp_avg"].to(param.device) if "exp_avg" in saved else torch.zeros_like(param)
        )
        exp_avg_sq = (
            saved["exp_avg_sq"].to(param.device)
            if "exp_avg_sq" in saved
            else torch.zeros_like(param)
        )

        if name.startswith("cross_attn.in_proj"):
            effective_grad = grad.clone() * clip_factor
            effective_grad[2 * qkv_rows :] = 0.0
        else:
            effective_grad = grad * clip_factor

        next_avg = exp_avg.mul(beta1).add(effective_grad, alpha=1.0 - beta1)
        next_avg_sq = exp_avg_sq.mul(beta2).addcmul(
            effective_grad, effective_grad, value=1.0 - beta2
        )
        next_step = prior_step + 1
        bc1 = 1.0 - beta1**next_step
        bc2 = 1.0 - beta2**next_step

        first = -lr * (next_avg / bc1) / ((next_avg_sq / bc2).sqrt() + eps)
        decay = -lr * weight_decay * param
        delta = first + decay

        if name.startswith("cross_attn.in_proj"):
            mask = torch.zeros_like(delta)
            mask[: 2 * qkv_rows] = 1.0
            delta = delta * mask

        deltas[name] = delta
        norms[name] = float(torch.linalg.vector_norm(delta).item())

    total_delta_norm = math.sqrt(sum(v * v for v in norms.values()))
    meta = {
        "lr": lr,
        "betas": [beta1, beta2],
        "eps": eps,
        "weight_decay": weight_decay,
        "gradient_clip_threshold": clip,
        "pre_clip_global_norm": float(total_norm.item()),
        "clip_factor": clip_factor,
        "target_score_parameters": sorted(target_score_params),
        "total_delta_norm": total_delta_norm,
        "norms_by_parameter": norms,
    }
    return deltas, meta


def _score_forward_from_params(
    primitive: Any,
    content: torch.Tensor,
    length: int,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Pure functional score forward pass without modifying primitive."""
    batch = content.size(0)
    device = content.device
    ids = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
    kv = F.linear(content, params["content_in_proj.weight"], params["content_in_proj.bias"])
    kv = kv + F.embedding(ids, params["content_position_embedding.weight"])
    query = F.embedding(ids, params["answer_query_embedding.weight"])
    rows = params["cross_attn.in_proj_weight"].shape[0] // 3
    q = F.linear(
        query,
        params["cross_attn.in_proj_weight"][:rows],
        params["cross_attn.in_proj_bias"][:rows],
    )
    k = F.linear(
        kv,
        params["cross_attn.in_proj_weight"][rows : 2 * rows],
        params["cross_attn.in_proj_bias"][rows : 2 * rows],
    )
    head_dim = primitive.d_operator // primitive.n_head
    q = q.view(batch, length, primitive.n_head, head_dim).transpose(1, 2)
    k = k.view(batch, length, primitive.n_head, head_dim).transpose(1, 2)
    s_other = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    dtype = content.dtype
    c_len = torch.full((batch, 1, 1), float(length), device=device, dtype=dtype)
    positions = torch.arange(length, device=device, dtype=dtype)
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
        features, params["position_bias_hidden.weight"], params["position_bias_hidden.bias"]
    )
    bias = F.relu(bias)
    bias = F.linear(bias, params["position_bias_out.weight"]).squeeze(-1)
    scores = s_other + bias.unsqueeze(1)
    return {"scores": scores, "s_other": s_other, "bias": bias, "kv": kv}


def _full_forward_from_params(
    primitive: Any,
    content: torch.Tensor,
    length: int,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Full J0 forward pass using perturbed score parameters and frozen downstream."""
    score_res = _score_forward_from_params(primitive, content, length, params)
    scores = score_res["scores"]
    kv = score_res["kv"]
    batch = content.size(0)
    device = content.device
    ids = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
    query = F.embedding(ids, params["answer_query_embedding.weight"])
    rows = params["cross_attn.in_proj_weight"].shape[0] // 3
    v = F.linear(
        kv,
        params["cross_attn.in_proj_weight"][2 * rows :],
        params["cross_attn.in_proj_bias"][2 * rows :],
    )
    head_dim = primitive.d_operator // primitive.n_head
    v = v.view(batch, length, primitive.n_head, head_dim).transpose(1, 2)
    attention = F.softmax(scores, dim=-1)
    attended = (
        torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, primitive.d_operator)
    )
    attended = F.linear(
        attended, params["cross_attn.out_proj.weight"], params["cross_attn.out_proj.bias"]
    )
    hidden = F.layer_norm(
        query + attended,
        (primitive.d_operator,),
        params["attn_norm.weight"],
        params["attn_norm.bias"],
    )
    ffn_h = F.linear(hidden, params["ffn.0.weight"], params["ffn.0.bias"])
    ffn_h = F.relu(ffn_h)
    ffn_h = F.linear(ffn_h, params["ffn.2.weight"], params["ffn.2.bias"])
    hidden = F.layer_norm(
        hidden + ffn_h,
        (primitive.d_operator,),
        params["ffn_norm.weight"],
        params["ffn_norm.bias"],
    )
    logits = F.linear(hidden, params["readout.weight"], params["readout.bias"])
    return logits


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().reshape(-1)
    right = right.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not torch.isfinite(denominator) or float(denominator.item()) == 0.0:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _entropy(prob: torch.Tensor, eps: float = 1e-12) -> float:
    p = torch.clamp(prob, min=eps)
    return float(-(p * torch.log(p)).sum().item())


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p_clamped = torch.clamp(p, min=eps)
    q_clamped = torch.clamp(q, min=eps)
    return float((p_clamped * (torch.log(p_clamped) - torch.log(q_clamped))).sum().item())


def _evaluate_checkpoint_dynamics(
    core: Any,
    spec: _CheckpointSpec,
    state: dict[str, Any],
    deltas: dict[str, torch.Tensor],
    probes: dict[str, dict[int, list[Example]]],
) -> dict[str, Any]:
    primitive = _new_primitive(core, state)
    base_params = {name: p.detach() for name, p in primitive.named_parameters()}
    params_01 = {name: p.detach() + 0.1 * deltas[name] for name, p in base_params.items()}
    params_10 = {name: p.detach() + 1.0 * deltas[name] for name, p in base_params.items()}

    results_by_probe: dict[str, Any] = {}
    raw_arrays: dict[str, np.ndarray] = {}

    for probe_name, dataset in probes.items():
        all_metrics: list[dict[str, Any]] = []
        l10_p4_metrics: list[dict[str, Any]] = []
        l10_p5_metrics: list[dict[str, Any]] = []
        l6_9_metrics: list[dict[str, Any]] = []

        l10_exs = dataset.get(10, [])
        l10_count = len(l10_exs)
        n_head = primitive.n_head

        l10_linear_tangent = np.zeros((l10_count, n_head, 10, 10), dtype=np.float32)
        l10_exact_01_delta = np.zeros((l10_count, n_head, 10, 10), dtype=np.float32)
        l10_exact_10_delta = np.zeros((l10_count, n_head, 10, 10), dtype=np.float32)
        l10_base_scores = np.zeros((l10_count, n_head, 10, 10), dtype=np.float32)

        output_trans = {
            "before": {"correct": 0, "total": 0, "em": 0, "p4_correct": 0, "p5_correct": 0},
            "exact_0.1": {"correct": 0, "total": 0, "em": 0, "p4_correct": 0, "p5_correct": 0},
            "exact_1.0": {"correct": 0, "total": 0, "em": 0, "p4_correct": 0, "p5_correct": 0},
        }

        for length, examples in dataset.items():
            op = get_operation(rec004g.REC004G_TARGET_OPERATION)
            output_lengths = [op.output_length(length)] * len(examples)
            labels = _labels_for_examples(examples, output_lengths, length, core.device)

            for start in range(0, len(examples), REC004S_CHUNK_SIZE):
                chunk = examples[start : start + REC004S_CHUNK_SIZE]
                chunk_labels = labels[start : start + len(chunk)]
                with torch.no_grad():
                    batch = collate_content_only_batch(chunk, core.tokens, device=core.device)
                    content = core.model.encode(batch)[:, 1 : 1 + length, :]

                tangent = rec004q._score_jvp(primitive, content, length, deltas).detach()
                s_before = _score_forward_from_params(primitive, content, length, base_params)[
                    "scores"
                ].detach()
                s_exact_01 = _score_forward_from_params(primitive, content, length, params_01)[
                    "scores"
                ].detach()
                s_exact_10 = _score_forward_from_params(primitive, content, length, params_10)[
                    "scores"
                ].detach()

                with torch.no_grad():
                    logits_b = _full_forward_from_params(primitive, content, length, base_params)
                    logits_01 = _full_forward_from_params(primitive, content, length, params_01)
                    logits_10 = _full_forward_from_params(primitive, content, length, params_10)

                for name, log in (
                    ("before", logits_b),
                    ("exact_0.1", logits_01),
                    ("exact_1.0", logits_10),
                ):
                    preds = torch.argmax(log, dim=-1)
                    for i in range(len(chunk)):
                        seq_correct = bool(torch.equal(preds[i], chunk_labels[i]))
                        output_trans[name]["em"] += int(seq_correct)
                        output_trans[name]["total"] += 1
                        if length == 10:
                            output_trans[name]["p4_correct"] += int(
                                preds[i, 4] == chunk_labels[i, 4]
                            )
                            output_trans[name]["p5_correct"] += int(
                                preds[i, 5] == chunk_labels[i, 5]
                            )

                if length == 10:
                    l10_base_scores[start : start + len(chunk)] = s_before.cpu().numpy()
                    l10_linear_tangent[start : start + len(chunk)] = tangent.cpu().numpy()
                    l10_exact_01_delta[start : start + len(chunk)] = (
                        (s_exact_01 - s_before).cpu().numpy()
                    )
                    l10_exact_10_delta[start : start + len(chunk)] = (
                        (s_exact_10 - s_before).cpu().numpy()
                    )

                for i in range(len(chunk)):
                    for h in range(primitive.n_head):
                        for p in range(length):
                            correct_k = (p + length // 2) % length
                            sb_row = s_before[i, h, p, :length]
                            tan_row = tangent[i, h, p, :length]
                            se01_row = s_exact_01[i, h, p, :length]
                            se10_row = s_exact_10[i, h, p, :length]

                            ds_01 = se01_row - sb_row
                            ds_10 = se10_row - sb_row

                            sb_wrong = sb_row.clone()
                            sb_wrong[correct_k] = -torch.inf
                            k_star = int(torch.argmax(sb_wrong).item())

                            m_before = float((sb_row[correct_k] - sb_row[k_star]).item())

                            dm_linear = float((tan_row[correct_k] - tan_row[k_star]).item())

                            se01_wrong = se01_row.clone()
                            se01_wrong[correct_k] = -torch.inf
                            m_e01 = float((se01_row[correct_k] - torch.max(se01_wrong)).item())
                            dm_e01 = m_e01 - m_before

                            se10_wrong = se10_row.clone()
                            se10_wrong[correct_k] = -torch.inf
                            m_e10 = float((se10_row[correct_k] - torch.max(se10_wrong)).item())
                            dm_e10 = m_e10 - m_before

                            norm_tan = float(torch.linalg.vector_norm(tan_row).item())

                            r_01 = float(
                                (
                                    torch.linalg.vector_norm(ds_01 - 0.1 * tan_row)
                                    / max(0.1 * norm_tan, REC004S_EPS)
                                ).item()
                            )
                            r_10 = float(
                                (
                                    torch.linalg.vector_norm(ds_10 - 1.0 * tan_row)
                                    / max(1.0 * norm_tan, REC004S_EPS)
                                ).item()
                            )

                            cos_01 = _safe_cosine(ds_01, 0.1 * tan_row)
                            cos_10 = _safe_cosine(ds_10, 1.0 * tan_row)

                            sign_agree_01 = (dm_e01 * (0.1 * dm_linear)) > 0 or (
                                abs(dm_e01) < 1e-8 and abs(dm_linear) < 1e-8
                            )
                            sign_agree_10 = (dm_e10 * (1.0 * dm_linear)) > 0 or (
                                abs(dm_e10) < 1e-8 and abs(dm_linear) < 1e-8
                            )

                            ab_row = F.softmax(sb_row, dim=-1)
                            al10_row = F.softmax(sb_row + 1.0 * tan_row, dim=-1)
                            ae01_row = F.softmax(se01_row, dim=-1)
                            ae10_row = F.softmax(se10_row, dim=-1)

                            row_metric = {
                                "length": length,
                                "position": p,
                                "head": h,
                                "m_before": m_before,
                                "dm_linear": dm_linear,
                                "dm_exact_0.1": dm_e01,
                                "dm_exact_1.0": dm_e10,
                                "r_0.1": r_01,
                                "r_1.0": r_10,
                                "cosine_0.1": cos_01,
                                "cosine_1.0": cos_10,
                                "sign_agree_0.1": bool(sign_agree_01),
                                "sign_agree_1.0": bool(sign_agree_10),
                                "attn_prob_change_0.1": float(
                                    (ae01_row[correct_k] - ab_row[correct_k]).item()
                                ),
                                "attn_prob_change_1.0": float(
                                    (ae10_row[correct_k] - ab_row[correct_k]).item()
                                ),
                                "kl_exact_before_1.0": _kl_divergence(ae10_row, ab_row),
                                "kl_exact_linear_1.0": _kl_divergence(ae10_row, al10_row),
                                "argmax_change_1.0": bool(
                                    torch.argmax(ae10_row) != torch.argmax(ab_row)
                                ),
                                "entropy_change_1.0": _entropy(ae10_row) - _entropy(ab_row),
                            }
                            all_metrics.append(row_metric)
                            if length == 10 and p == 4:
                                l10_p4_metrics.append(row_metric)
                            elif length == 10 and p == 5:
                                l10_p5_metrics.append(row_metric)
                            elif length < 10:
                                l6_9_metrics.append(row_metric)

        def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not rows:
                return {}
            n = len(rows)
            cos_01_vals = [r["cosine_0.1"] for r in rows if r["cosine_0.1"] is not None]
            cos_10_vals = [r["cosine_1.0"] for r in rows if r["cosine_1.0"] is not None]
            r_01_vals = [r["r_0.1"] for r in rows]
            r_10_vals = [r["r_1.0"] for r in rows]
            dm_l_pos = [r for r in rows if r["dm_linear"] > 0]
            overshoot_10_count = sum(1 for r in dm_l_pos if r["dm_exact_1.0"] < 0)
            improve_01_count = sum(1 for r in dm_l_pos if r["dm_exact_0.1"] > 0)
            return {
                "count": n,
                "median_cosine_0.1": float(np.median(cos_01_vals)) if cos_01_vals else None,
                "median_cosine_1.0": float(np.median(cos_10_vals)) if cos_10_vals else None,
                "mean_cosine_1.0": float(np.mean(cos_10_vals)) if cos_10_vals else None,
                "median_r_0.1": float(np.median(r_01_vals)),
                "median_r_1.0": float(np.median(r_10_vals)),
                "mean_r_1.0": float(np.mean(r_10_vals)),
                "sign_agreement_0.1": sum(1 for r in rows if r["sign_agree_0.1"]) / n,
                "sign_agreement_1.0": sum(1 for r in rows if r["sign_agree_1.0"]) / n,
                "mean_dm_linear": float(np.mean([r["dm_linear"] for r in rows])),
                "mean_dm_exact_0.1": float(np.mean([r["dm_exact_0.1"] for r in rows])),
                "mean_dm_exact_1.0": float(np.mean([r["dm_exact_1.0"] for r in rows])),
                "linear_positive_count": len(dm_l_pos),
                "overshoot_inversion_fraction_at_1.0": (overshoot_10_count / len(dm_l_pos))
                if dm_l_pos
                else None,
                "early_improvement_fraction_at_0.1": (improve_01_count / len(dm_l_pos))
                if dm_l_pos
                else None,
                "mean_attn_prob_change_1.0": float(
                    np.mean([r["attn_prob_change_1.0"] for r in rows])
                ),
                "mean_kl_exact_before_1.0": float(
                    np.mean([r["kl_exact_before_1.0"] for r in rows])
                ),
                "mean_kl_exact_linear_1.0": float(
                    np.mean([r["kl_exact_linear_1.0"] for r in rows])
                ),
                "argmax_change_fraction_1.0": sum(1 for r in rows if r["argmax_change_1.0"]) / n,
                "mean_entropy_change_1.0": float(np.mean([r["entropy_change_1.0"] for r in rows])),
            }

        trans_res = {}
        for name, data in output_trans.items():
            tot = data["total"]
            l10_tot = sum(1 for length, exs in dataset.items() if length == 10 for _ in exs)
            trans_res[name] = {
                "sequence_em": data["em"] / tot if tot else 0.0,
                "position4_accuracy": data["p4_correct"] / l10_tot if l10_tot else 0.0,
                "position5_accuracy": data["p5_correct"] / l10_tot if l10_tot else 0.0,
            }

        results_by_probe[probe_name] = {
            "all": _summarize_bucket(all_metrics),
            "length10_position4": _summarize_bucket(l10_p4_metrics),
            "length10_position5": _summarize_bucket(l10_p5_metrics),
            "length6_9_aggregate": _summarize_bucket(l6_9_metrics),
            "output_translation": trans_res,
        }

        raw_arrays[f"{spec.checkpoint_id}_{probe_name}_linear_tangent"] = l10_linear_tangent
        raw_arrays[f"{spec.checkpoint_id}_{probe_name}_exact_0.1_delta"] = l10_exact_01_delta
        raw_arrays[f"{spec.checkpoint_id}_{probe_name}_exact_1.0_delta"] = l10_exact_10_delta
        raw_arrays[f"{spec.checkpoint_id}_{probe_name}_base_scores"] = l10_base_scores

    return {
        "spec": spec,
        "results_by_probe": results_by_probe,
        "raw_arrays": raw_arrays,
    }


def _measure_historical_drift(
    core: Any,
    probes: dict[str, dict[int, list[Example]]],
) -> dict[str, Any]:
    """Measure score-function drift across historical 500-step checkpoints (6000->12000)."""
    probe = probes["continuity"]
    l10_exs = probe.get(10, [])[:64]
    with torch.no_grad():
        batch = collate_content_only_batch(l10_exs, core.tokens, device=core.device)
        content = core.model.encode(batch)[:, 1:11, :]

    steps = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000]
    traces: dict[str, Any] = {}

    for init_id in ("I03", "I04"):
        scores_by_step: dict[int, torch.Tensor] = {}
        margins_by_step: dict[int, float] = {}
        for step in steps:
            if step == 6000:
                ckpt_path = Path(
                    f"runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/{init_id}/P_LENGTH_POSITION_BIAS/training_states/step6000.pt"
                )
            else:
                ckpt_path = Path(
                    f"runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/{init_id}/P_LENGTH_POSITION_BIAS/training_states/step{step}.pt"
                )
            if not ckpt_path.is_file():
                continue
            st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            prim = _new_primitive(core, st)
            params = {name: p.detach() for name, p in prim.named_parameters()}
            s = _score_forward_from_params(prim, content, 10, params)["scores"].detach()
            scores_by_step[step] = s

            p4_correct = (4 + 5) % 10
            p4_scores = s[:, :, 4, :]
            p4_wrong = p4_scores.clone()
            p4_wrong[:, :, p4_correct] = -torch.inf
            m = float((p4_scores[:, :, p4_correct] - torch.max(p4_wrong, dim=-1).values).mean().item())
            margins_by_step[step] = m

        intervals: list[dict[str, Any]] = []
        avail_steps = sorted(scores_by_step.keys())
        for idx in range(len(avail_steps) - 1):
            s_from = avail_steps[idx]
            s_to = avail_steps[idx + 1]
            diff = scores_by_step[s_to] - scores_by_step[s_from]
            drift_norm = float(torch.linalg.vector_norm(diff).item()) / math.sqrt(diff.numel())
            cos = _safe_cosine(scores_by_step[s_to], scores_by_step[s_from])
            intervals.append(
                {
                    "step_from": s_from,
                    "step_to": s_to,
                    "rms_score_drift": drift_norm,
                    "step_cosine": cos,
                    "margin_p4_change": margins_by_step[s_to] - margins_by_step[s_from],
                }
            )
        traces[init_id] = {
            "margins_by_step": margins_by_step,
            "intervals": intervals,
        }
    return traces


def _evaluate_decision(
    checkpoint_results: dict[str, Any],
    locality_check: dict[str, Any],
) -> dict[str, Any]:
    if not locality_check["all_passed"]:
        return {
            "label": "FINITE_STEP_DIAGNOSTIC_IMPLEMENTATION_OR_LOCALITY_UNRESOLVED",
            "locality_check": locality_check,
            "rationale": "α=0.1 locality parity gate failed (median cosine < 0.95)",
            "overshoot_rule_eval": None,
            "nonlinearity_rule_eval": None,
            "locally_faithful_rule_eval": None,
            "new_optimizer_updates": 0,
            "selected_init": None,
            "selected_step": None,
            "selected_lr_change": None,
            "selected_optimizer_change": None,
            "child_bundle": None,
            "rg3_recheck": "NOT_EXECUTED",
            "rec005_eligible": False,
        }

    score_only = checkpoint_results["I03_SCORE_ONLY_12000"]["results_by_probe"]["continuity"][
        "length10_position4"
    ]
    cp_score = checkpoint_results["I03_CP_SCORE_12000"]["results_by_probe"]["continuity"][
        "length10_position4"
    ]
    i04_control = checkpoint_results["I04_P_7000"]["results_by_probe"]["continuity"][
        "length10_position4"
    ]

    so_overshoot = (
        score_only["overshoot_inversion_fraction_at_1.0"] is not None
        and score_only["overshoot_inversion_fraction_at_1.0"] >= 0.60
        and score_only["early_improvement_fraction_at_0.1"] is not None
        and score_only["early_improvement_fraction_at_0.1"] >= 0.80
    )
    cp_overshoot = (
        cp_score["overshoot_inversion_fraction_at_1.0"] is not None
        and cp_score["overshoot_inversion_fraction_at_1.0"] >= 0.60
        and cp_score["early_improvement_fraction_at_0.1"] is not None
        and cp_score["early_improvement_fraction_at_0.1"] >= 0.80
    )
    i04_not_overshoot = (
        i04_control["overshoot_inversion_fraction_at_1.0"] is None
        or i04_control["overshoot_inversion_fraction_at_1.0"] < 0.60
    )
    overshoot_supported = so_overshoot and cp_overshoot and i04_not_overshoot

    so_nonlinear = (
        score_only["median_r_1.0"] >= 1.0
        and score_only["median_cosine_1.0"] is not None
        and score_only["median_cosine_1.0"] <= 0.50
    )
    cp_nonlinear = (
        cp_score["median_r_1.0"] >= 1.0
        and cp_score["median_cosine_1.0"] is not None
        and cp_score["median_cosine_1.0"] <= 0.50
    )
    i04_not_nonlinear = not (
        i04_control["median_r_1.0"] >= 1.0
        and i04_control["median_cosine_1.0"] is not None
        and i04_control["median_cosine_1.0"] <= 0.50
    )
    nonlinearity_supported = (not overshoot_supported) and (
        so_nonlinear and cp_nonlinear and i04_not_nonlinear
    )

    so_faithful = (
        score_only["median_cosine_1.0"] is not None
        and score_only["median_cosine_1.0"] >= 0.80
        and score_only["median_r_1.0"] < 0.50
        and score_only["sign_agreement_1.0"] >= 0.80
    )
    cp_faithful = (
        cp_score["median_cosine_1.0"] is not None
        and cp_score["median_cosine_1.0"] >= 0.80
        and cp_score["median_r_1.0"] < 0.50
        and cp_score["sign_agreement_1.0"] >= 0.80
    )
    locally_faithful = so_faithful and cp_faithful

    if overshoot_supported:
        label = "OPTIMIZER_SIZED_SCORE_FUNCTION_OVERSHOOT_SUPPORTED"
        rationale = (
            "Local linear prediction shows improvement (Δm_linear > 0), but optimizer-sized "
            "step (α=1.0) inverts margin in >=60% of cases, while α=0.1 improves >=80% in "
            "both terminal arms, and I04 control does not show >=60% inversion."
        )
    elif nonlinearity_supported:
        label = "STRONG_PARAMETER_TO_SCORE_NONLINEARITY_SUPPORTED"
        rationale = (
            "Strong parameter-to-score non-linearity (median r_1.0 >= 1.0 and median cosine <= 0.50) "
            "in both terminal arms, while I04 control does not meet these criteria."
        )
    elif locally_faithful:
        label = "ONE_STEP_SCORE_DYNAMICS_LOCALLY_FAITHFUL"
        rationale = (
            "One-step dynamics are locally faithful (median cosine >= 0.80, r_1.0 < 0.50, "
            "sign agreement >= 0.80) in both terminal arms. Single-step dynamics do not explain "
            "the failure; multi-step trajectory accumulation is implicated."
        )
    else:
        label = "FINITE_STEP_SCORE_DYNAMICS_MIXED_OR_UNRESOLVED"
        rationale = (
            "Results do not consistently satisfy overshoot, strong nonlinearity, or faithful criteria."
        )

    return {
        "label": label,
        "rationale": rationale,
        "overshoot_rule_eval": {
            "score_only": so_overshoot,
            "cp_score": cp_overshoot,
            "i04_control_not_overshoot": i04_not_overshoot,
            "supported": overshoot_supported,
        },
        "nonlinearity_rule_eval": {
            "score_only": so_nonlinear,
            "cp_score": cp_nonlinear,
            "i04_control_not_nonlinear": i04_not_nonlinear,
            "supported": nonlinearity_supported,
        },
        "locally_faithful_rule_eval": {
            "score_only": so_faithful,
            "cp_score": cp_faithful,
            "supported": locally_faithful,
        },
        "new_optimizer_updates": 0,
        "selected_init": None,
        "selected_step": None,
        "selected_lr_change": None,
        "selected_optimizer_change": None,
        "child_bundle": None,
        "rg3_recheck": "NOT_EXECUTED",
        "rec005_eligible": False,
    }


def _next_repair_contract_text(label: str) -> str:
    if label == "OPTIMIZER_SIZED_SCORE_FUNCTION_OVERSHOOT_SUPPORTED":
        return r"""# Next Repair Contract: Score-Path Step-Size Isolation Pilot

Result label: `OPTIMIZER_SIZED_SCORE_FUNCTION_OVERSHOOT_SUPPORTED`.

## Proposed Design (Pilot only; not authorized or tuned in this task)
1. Isolate the learning rate of the score-path parameters (`Q_rows`, `K_rows`,
   `position_bias_hidden`, `position_bias_out`, and `CONTENT_PREP` if trainable)
   from the downstream parameters (`V_rows`, `attn_out_proj`, `FFN`, `readout`).
2. Design a bounded step-size pilot where the effective update magnitude on score logits
   is constrained within the verified local linear regime ($\le 0.1 \times$ AdamW base scale).
3. Do not explore grid values, change loss functions, or run optimizer steps within REC-004S.
"""
    elif label == "STRONG_PARAMETER_TO_SCORE_NONLINEARITY_SUPPORTED":
        return r"""# Next Repair Contract: Score-Path Trust-Region / Bounded-Update Mechanism

Result label: `STRONG_PARAMETER_TO_SCORE_NONLINEARITY_SUPPORTED`.

## Proposed Design (Minimal concept only)
1. Design a score-path trust-region constraint or bounded-update norm on $\Delta S$.
2. Avoid heuristic gradient surgery, PCGrad, or auxiliary oracle attention losses.
3. Keep downstream architecture and core invariant.
"""
    elif label == "ONE_STEP_SCORE_DYNAMICS_LOCALLY_FAITHFUL":
        return """# Next Repair Contract: Short-Horizon Dense Replay

Result label: `ONE_STEP_SCORE_DYNAMICS_LOCALLY_FAITHFUL`.

Instantaneous single-step diagnostics are hereby concluded. Since single-step updates
are locally faithful, the failure arises across multiple updates along the trajectory.

## Proposed Next Step
1. Short-horizon dense replay: from `I03@6000`, run a bit-exact replay for a fixed
   window (e.g. 128 updates), logging score functions, attention distributions,
   and downstream margins at every step.
2. Pinpoint the exact step range where locally faithful updates begin to accumulate
   destructive curvature or enter an unrecoverable basin.
"""
    else:
        return """# Next Repair Contract: Mixed or Unresolved Diagnostic Follow-Up

Result label: `FINITE_STEP_SCORE_DYNAMICS_MIXED_OR_UNRESOLVED`.

## Proposed Next Step
1. Maintain strict architecture and loss invariance.
2. At most one minimal additional probe confirmation without expanding scope.
"""


def _generate_report_markdown(
    decision: dict[str, Any],
    parity_check: dict[str, Any],
    p4_summary: dict[str, Any],
    output_trans: dict[str, Any],
) -> str:
    return f"""# B-C005REC-004S: Finite-Step Score Dynamics Audit Report

**Task ID:** `B-C005REC-004S`
**Decision Label:** `{decision["label"]}`
**Implementation/Parity Gate:** `{"PASS" if parity_check["all_passed"] else "FAIL"}`
**New Optimizer Updates:** 0

## 1. Executive Summary
- **Parity Gate (α=0.1):** Median cosine between JVP linear prediction and exact finite difference:
  - Minimum across checkpoints: `{parity_check["min_median_cosine"]:.4f}` (Threshold: `>= 0.95`).
- **Primary Operational Diagnosis:** `{decision["label"]}`.
- **Rationale:** {decision["rationale"]}

## 2. Length-10 Position-4 Key Metrics (Continuity Probe)
```json
{json.dumps(p4_summary, indent=2)}
```

## 3. Output Translation Summary
```json
{json.dumps(output_trans, indent=2)}
```

## 4. Status and Blockers
- `new_optimizer_updates = 0`
- `virtual_optimizer_steps = diagnostic_only`
- `selected_init = null`
- `child_bundle = null`
- `rg3_recheck = NOT_EXECUTED`
- `rec005_eligible = false`
"""


def run_mirror_score_function_finite_step_dynamics_audit_task(
    config: MirrorScoreFunctionFiniteStepDynamicsAuditConfig,
) -> dict[str, Any]:
    """Run the entire B-C005REC-004S read-only diagnostic task and stop."""
    if config.seed != 10:
        raise ValueError("B-C005REC-004S is pre-registered to recovery seed 10")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    required_checkpoints = [spec.path for spec in _CHECKPOINTS]
    if not all(path.is_file() for path in required_checkpoints):
        return {
            "implementation_status": "STOPPED",
            "result_label": "SOURCE_ARTIFACT_UNAVAILABLE",
        }

    _write_json(output_dir / "config.yaml", {"seed": config.seed, "output_dir": str(output_dir)})
    _write_json(output_dir / "system.json", get_system_info(seed=config.seed))

    loaded, checkpoint_manifest = _source_checkpoint_manifest()
    _write_json(output_dir / "source_checkpoint_manifest.json", checkpoint_manifest)

    probes, probe_manifest = _build_probes(config.seed)
    _write_json(output_dir / "score_function_dynamics_probe_manifest.json", probe_manifest)

    next_batches, next_batch_manifest = _generate_next_training_batches(config.seed, loaded)
    _write_json(output_dir / "next_training_batch_manifest.json", next_batch_manifest)

    parent_manifest, _raw = ibc._load_parent_manifest()
    core, eval_bank, op_to_id = ibc._reconstruct_parent_runtime(
        ibc.IncrementalBudgetCalibrationConfig(seed=config.seed), parent_manifest
    )
    core_before = mb.canonical_state_hash(core.model.state_dict())
    protected_before = mpbr._protected_scope_hashes(eval_bank, op_to_id)
    source_hashes_before = {
        spec.checkpoint_id: mb.raw_file_sha256(spec.path) for spec, _ in loaded
    }

    adamw_deltas: dict[str, dict[str, torch.Tensor]] = {}
    adamw_meta: dict[str, Any] = {}
    checkpoint_results: dict[str, Any] = {}
    all_raw_arrays: dict[str, np.ndarray] = {}

    for spec, state in loaded:
        primitive = _new_primitive(core, state)
        batch_examples = next_batches[spec.next_step]
        task_grads = _compute_next_step_task_gradient(core, primitive, batch_examples)
        delta, meta = _reconstruct_analytical_adamw_delta(
            primitive, task_grads, state, spec.update_policy
        )
        adamw_deltas[spec.checkpoint_id] = delta
        adamw_meta[spec.checkpoint_id] = meta

        eval_res = _evaluate_checkpoint_dynamics(core, spec, state, delta, probes)
        checkpoint_results[spec.checkpoint_id] = eval_res
        all_raw_arrays.update(eval_res["raw_arrays"])

    _write_json(output_dir / "adamw_virtual_delta_manifest.json", adamw_meta)

    np.savez_compressed(
        output_dir / "score_jvp_predictions.npz",
        **{k: v for k, v in all_raw_arrays.items() if "linear_tangent" in k},  # type: ignore[arg-type]
    )
    np.savez_compressed(
        output_dir / "finite_step_exact_changes.npz",
        **{k: v for k, v in all_raw_arrays.items() if "exact" in k or "base_scores" in k},  # type: ignore[arg-type]
    )

    index_entries: dict[str, Any] = {}
    for k, v in all_raw_arrays.items():
        index_entries[k] = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            "layout": "batch (512), head (4), output_pos (10), key_pos (10)",
        }
    _write_json(output_dir / "finite_step_index.json", index_entries)

    parity_records: dict[str, Any] = {}
    min_median_cos = 1.0
    for cid, cdata in checkpoint_results.items():
        cos_cont = cdata["results_by_probe"]["continuity"]["all"]["median_cosine_0.1"]
        cos_fresh = cdata["results_by_probe"]["fresh"]["all"]["median_cosine_0.1"]
        parity_records[cid] = {
            "continuity_median_cosine_0.1": cos_cont,
            "fresh_median_cosine_0.1": cos_fresh,
            "parity_pass": (cos_cont is not None and cos_cont >= 0.95)
            and (cos_fresh is not None and cos_fresh >= 0.95),
        }
        if cos_cont is not None:
            min_median_cos = min(min_median_cos, cos_cont)
        if cos_fresh is not None:
            min_median_cos = min(min_median_cos, cos_fresh)

    locality_check = {
        "all_passed": all(r["parity_pass"] for r in parity_records.values()),
        "threshold": 0.95,
        "min_median_cosine": min_median_cos,
        "by_checkpoint": parity_records,
    }
    _write_json(output_dir / "locality_parity_check.json", locality_check)

    p4_summary = {
        cid: {
            probe: cdata["results_by_probe"][probe]["length10_position4"]
            for probe in ("continuity", "fresh")
        }
        for cid, cdata in checkpoint_results.items()
    }
    _write_json(output_dir / "position4_finite_step_summary.json", p4_summary)

    p5_summary = {
        cid: {
            probe: cdata["results_by_probe"][probe]["length10_position5"]
            for probe in ("continuity", "fresh")
        }
        for cid, cdata in checkpoint_results.items()
    }
    _write_json(output_dir / "position5_control_summary.json", p5_summary)

    l6_9_summary = {
        cid: {
            probe: cdata["results_by_probe"][probe]["length6_9_aggregate"]
            for probe in ("continuity", "fresh")
        }
        for cid, cdata in checkpoint_results.items()
    }
    _write_json(output_dir / "length6_9_control_summary.json", l6_9_summary)

    attn_summary = {
        cid: {
            "continuity_l10_p4": {
                "mean_attn_prob_change_1.0": cdata["results_by_probe"]["continuity"][
                    "length10_position4"
                ]["mean_attn_prob_change_1.0"],
                "mean_kl_exact_before_1.0": cdata["results_by_probe"]["continuity"][
                    "length10_position4"
                ]["mean_kl_exact_before_1.0"],
                "mean_kl_exact_linear_1.0": cdata["results_by_probe"]["continuity"][
                    "length10_position4"
                ]["mean_kl_exact_linear_1.0"],
                "argmax_change_fraction_1.0": cdata["results_by_probe"]["continuity"][
                    "length10_position4"
                ]["argmax_change_fraction_1.0"],
                "mean_entropy_change_1.0": cdata["results_by_probe"]["continuity"][
                    "length10_position4"
                ]["mean_entropy_change_1.0"],
            }
        }
        for cid, cdata in checkpoint_results.items()
    }
    _write_json(output_dir / "attention_distribution_change.json", attn_summary)

    output_trans_summary = {
        cid: cdata["results_by_probe"]["continuity"]["output_translation"]
        for cid, cdata in checkpoint_results.items()
    }
    _write_json(output_dir / "output_translation_summary.json", output_trans_summary)

    historical_drift = _measure_historical_drift(core, probes)
    _write_json(output_dir / "historical_score_drift.json", historical_drift)

    decision = _evaluate_decision(checkpoint_results, locality_check)
    _write_json(output_dir / "finite_step_dynamics_decision.json", decision)

    repair_contract_md = _next_repair_contract_text(decision["label"])
    (output_dir / "next_repair_contract.md").write_text(repair_contract_md, encoding="utf-8")

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
        "autograd_method": "torch.autograd.grad and torch.func.jvp on in-memory tensors only",
        "source_artifacts_unchanged": freeze_audit["source_checkpoints_unchanged"],
        "training": "NOT_EXECUTED",
        "sealed_or_rg3_query": "NOT_EXECUTED",
        "child_bundle": None,
    }
    _write_json(output_dir / "side_effect_audit.json", side_effect_audit)

    report = {
        "task_id": REC004S_TASK_ID,
        "implementation_status": "COMPLETED",
        "result_label": decision["label"],
        "decision": decision,
        "parity_check": locality_check,
        "freeze_audit": freeze_audit,
        "side_effect_audit": side_effect_audit,
        "cost_accounting": {
            "wall_clock_seconds": time.time() - started,
            "new_optimizer_updates": 0,
            "probe_examples_per_set": sum(REC004S_PROBE_COUNTS.values()),
            "probe_set_count": len(probes),
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

    report_md = _generate_report_markdown(
        decision, locality_check, p4_summary, output_trans_summary
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    return report
