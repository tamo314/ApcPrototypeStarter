"""Functional adequacy evidence interface (Phase A.2 Task A2-C005).

Provides runtime evidence used by the autonomous controller to decide among:
`DIRECT_REUSE`, `COMPOSE`, and `PLASTIC_SEARCH`.

Strict Invariants Enforced (ADR-0062 / CODEX Task A2-C005):
1. Zero oracle leakage:
   The interface MUST NOT inspect or expose:
   - oracle operation class / label (`ORACLE_LABEL_*`),
   - latent `example.program` or `operation_graph`,
   - latent `example.oracle_metadata`,
   - held-out evaluation targets.
   Evidence is computed exclusively from model-visible `support_examples[i].input_tokens`,
   `support_examples[i].target_tokens`, and `support_examples[i].task_spec`.
2. Functional adequacy as primary signal:
   Computational sufficiency is established by actual exact match (EM) and token loss
   on support examples, never by registry membership or operation token identity.
3. Deterministic output:
   Given identical support sets and model parameters, evidence metrics are
   strictly deterministic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import build_task_only_tokens, pad_token_sequences
from apc.environments.generator import Example
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import (
    execute_composition_recipe,
)
from apc.primitives.composition_search import (
    is_candidate_structurally_valid,
    search_composition_recipe,
)
from apc.primitives.router import Router

_EPS: Final[float] = 1e-9
MAX_CLAMPED_LOSS: Final[float] = 10.0


@dataclass(frozen=True)
class AdequacyEvidence:
    """Runtime evidence record extracted from a small support set.

    Captures:
    - Direct primitive functional performance & margin.
    - Multi-step composition search performance & improvement.
    - Learned router confidence, margin, and entropy.
    - Recurrence & representation similarity.
    - Zero oracle metadata or held-out targets.
    """

    # Direct evidence
    direct_em: float
    direct_loss: float
    direct_token_acc: float
    direct_primitive_id: int | None
    direct_runner_up_em: float | None = None
    direct_runner_up_loss: float | None = None
    direct_margin: float = 0.0
    direct_candidates_evaluated: int = 0

    # Composition evidence
    composition_em: float = 0.0
    composition_loss: float = float("inf")
    composition_depth: int = 1
    composition_recipe: tuple[str, ...] | None = None
    composition_improvement_em: float = 0.0
    composition_improvement_loss: float = 0.0
    composition_candidates_evaluated: int = 0
    composition_candidates_pruned: int = 0

    # Router evidence
    router_confidence: float = 0.0
    router_margin: float = 0.0
    router_entropy: float = 0.0
    proposed_primitive_id: int | None = None
    runner_up_primitive_id: int | None = None

    # Recurrence & retrieval evidence
    recurrence_key_similarity: float = 0.0
    prototype_similarity: float | None = None

    # Execution & audit metadata
    support_size: int = 0
    compute_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> list[float]:
        """Convert evidence to a standardized 13-dimensional numerical vector.

        Suitable for direct consumption by the learned adequacy controller (Task A2-C006).
        Guaranteed to contain zero NaNs and zero Infs.
        """
        clamped_direct_loss = min(
            MAX_CLAMPED_LOSS,
            self.direct_loss if not math.isinf(self.direct_loss) else MAX_CLAMPED_LOSS,
        )
        clamped_comp_loss = min(
            MAX_CLAMPED_LOSS,
            self.composition_loss if not math.isinf(self.composition_loss) else MAX_CLAMPED_LOSS,
        )
        clamped_comp_loss_imp = min(
            MAX_CLAMPED_LOSS,
            max(
                0.0,
                (
                    self.composition_improvement_loss
                    if not math.isinf(self.composition_improvement_loss)
                    else 0.0
                ),
            ),
        )

        return [
            float(self.direct_em),
            float(clamped_direct_loss),
            float(self.direct_token_acc),
            float(self.direct_margin),
            float(self.composition_em),
            float(clamped_comp_loss),
            float(self.composition_depth),
            float(self.composition_improvement_em),
            float(clamped_comp_loss_imp),
            float(self.router_confidence),
            float(self.router_margin),
            float(self.router_entropy),
            float(self.recurrence_key_similarity),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence record to a JSON-serializable dictionary."""
        data = asdict(self)
        if isinstance(data.get("composition_recipe"), tuple):
            data["composition_recipe"] = list(data["composition_recipe"])
        # Replace inf with large float for clean JSON serialization
        if math.isinf(data["composition_loss"]):
            data["composition_loss"] = 1e9
        if math.isinf(data["direct_loss"]):
            data["direct_loss"] = 1e9
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdequacyEvidence:
        """Construct evidence record from dictionary."""
        d = dict(data)
        if "composition_recipe" in d and isinstance(d["composition_recipe"], list):
            d["composition_recipe"] = tuple(d["composition_recipe"])
        if d.get("composition_loss") == 1e9:
            d["composition_loss"] = float("inf")
        if d.get("direct_loss") == 1e9:
            d["direct_loss"] = float("inf")
        return cls(**d)


@dataclass(frozen=True)
class AdequacyEvidenceConfig:
    """Configuration options for adequacy evidence computation."""

    direct_eval_k: int = 2
    composition_max_depth: int = 2
    composition_beam_width: int = 16
    composition_early_stop_em: float = 1.0
    allow_composition_search: bool = True
    clamp_max_loss: float = MAX_CLAMPED_LOSS


def extract_task_representations(
    core: Any,
    examples: Sequence[Example],
) -> torch.Tensor:
    """Extract z_task at [TASK_END] from frozen core for each example.

    Strict Invariant: Reads ONLY `example.task_spec`. Never reads `example.program`
    or `example.oracle_metadata`.
    """
    device = core.device
    tokens = core.tokens
    task_token_seqs = []
    for ex in examples:
        if ex.task_spec is None:
            raise ValueError("Example must contain a model-visible task_spec.")
        task_token_seqs.append(build_task_only_tokens(ex.task_spec, tokens))
    padded_task_ids = pad_token_sequences(task_token_seqs, tokens.pad, device)

    with torch.no_grad():
        encoded = core.model.encode(padded_task_ids)

    task_end_mask = padded_task_ids == tokens.task_end
    task_end_indices = task_end_mask.to(torch.long).argmax(dim=-1)

    batch_indices = torch.arange(len(examples), device=device)
    z_task = encoded[batch_indices, task_end_indices, :]
    return z_task


def _evaluate_single_primitive_on_support(
    core: Any,
    bank: PrimitiveBank,
    op_name: str,
    primitive_id: int,
    examples: Sequence[Example],
) -> tuple[float, float, float]:
    """Compute (exact_match, token_cross_entropy_loss, token_accuracy) for a primitive.

    Args:
        core: SharedContentEncoder (task-blind content encoder).
        bank: PrimitiveBank holding the primitive.
        op_name: Canonical operation name.
        primitive_id: ID of the primitive in bank.
        examples: Small support set.

    Returns:
        tuple of (exact_match [0..1], cross_entropy_loss, token_accuracy [0..1]).
    """
    if not is_candidate_structurally_valid((op_name,), examples, must_match_target_length=True):
        return 0.0, MAX_CLAMPED_LOSS, 0.0

    logits = execute_composition_recipe(
        core,
        bank,
        {op_name: primitive_id},
        examples,
        candidate_operations=(op_name,),
    )
    predictions = logits.argmax(dim=-1)

    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0
    total_loss = 0.0

    for i, ex in enumerate(examples):
        target = ex.target_tokens
        n = len(target)
        pred = tuple(predictions[i, :n].tolist())
        if pred == target:
            exact_matches += 1

        correct_tokens += sum(p == t for p, t in zip(pred, target, strict=True))
        total_tokens += n

        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        pred_logits = logits[i, :n, :]
        step_loss = F.cross_entropy(pred_logits, target_tensor, reduction="sum")
        total_loss += step_loss.item()

    em = exact_matches / len(examples)
    avg_loss = total_loss / max(1, total_tokens)
    token_acc = correct_tokens / max(1, total_tokens)
    return em, avg_loss, token_acc


def compute_adequacy_evidence(
    core: Any,
    bank: PrimitiveBank,
    router: Router,
    op_to_id: dict[str, int],
    support_examples: Sequence[Example],
    *,
    config: AdequacyEvidenceConfig | None = None,
    candidate_ids: Sequence[int] | None = None,
    prototypes_by_pid: dict[int, torch.Tensor] | None = None,
) -> AdequacyEvidence:
    """Extract runtime adequacy evidence from a support set S = {(x, y)}.

    Strict Invariants:
    1. Zero oracle leakage: `example.oracle_metadata` and `example.program` are never accessed.
    2. Model visibility: Only `task_spec` and input/target tokens are consumed.
    3. Deterministic output: Guaranteed repeatable under identical parameters.

    Args:
        core: SharedContentEncoder (frozen task-blind core).
        bank: PrimitiveBank containing compact primitives.
        router: Learned or calibrated Router.
        op_to_id: Mapping from operation names to bank primitive IDs.
        support_examples: Small support set (e.g. N=8..32).
        config: Configuration options for search & evaluation.
        candidate_ids: Optional list of candidate primitive IDs to consider.
        prototypes_by_pid: Optional prototype embeddings for recurrence similarity.

    Returns:
        `AdequacyEvidence` dataclass.
    """
    if not support_examples:
        raise ValueError("support_examples must be non-empty.")

    start_time = time.perf_counter()
    cfg = config or AdequacyEvidenceConfig()

    active_candidate_ids = list(candidate_ids or router.ids())
    if not active_candidate_ids:
        raise ValueError("candidate_ids must be non-empty.")

    id_to_op = {pid: name for name, pid in op_to_id.items()}

    # ---------------------------------------------------------
    # 1. Router Scoring & Evidence
    # ---------------------------------------------------------
    z_tasks = extract_task_representations(core, support_examples)
    z_mean = z_tasks.mean(dim=0, keepdim=True)

    with torch.no_grad():
        router_out = router(z_mean, active_candidate_ids)
        probs = router_out.probs[0]
        entropy = router_out.entropy[0].item()

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        top1_idx = int(sorted_indices[0].item())
        top1_pid = active_candidate_ids[top1_idx]
        top1_prob = sorted_probs[0].item()

        if len(active_candidate_ids) > 1:
            top2_idx = int(sorted_indices[1].item())
            top2_pid: int | None = active_candidate_ids[top2_idx]
            top2_prob = sorted_probs[1].item()
        else:
            top2_pid = None
            top2_prob = 0.0

        router_confidence = top1_prob
        router_margin = top1_prob - top2_prob
        router_entropy = entropy

        # Recurrence key similarity: cosine similarity of query projection to keys
        query = router.query_proj(z_mean)[0]
        key_similarities: list[float] = []
        for pid in active_candidate_ids:
            if router.has_primitive(pid):
                key_param = router.key_parameter(pid)
                cos = F.cosine_similarity(query.unsqueeze(0), key_param.unsqueeze(0), dim=-1).item()
                key_similarities.append(cos)
        max_key_sim = max(key_similarities) if key_similarities else 0.0

        # Prototype similarity (if prototype store provided)
        max_proto_sim: float | None = None
        if prototypes_by_pid:
            proto_sims: list[float] = []
            for _pid, proto in prototypes_by_pid.items():
                p_vec = proto.to(z_mean.device)
                cos = F.cosine_similarity(z_mean[0].unsqueeze(0), p_vec.unsqueeze(0), dim=-1).item()
                proto_sims.append(cos)
            if proto_sims:
                max_proto_sim = max(proto_sims)

    # ---------------------------------------------------------
    # 2. Direct Primitive Evaluation
    # ---------------------------------------------------------
    # Evaluate top-k candidate primitives proposed by the router
    candidates_to_eval: list[int] = [top1_pid]
    if top2_pid is not None and cfg.direct_eval_k >= 2 and top2_pid != top1_pid:
        candidates_to_eval.append(top2_pid)

    direct_results: list[tuple[int, float, float, float]] = []
    with torch.no_grad():
        for pid in candidates_to_eval:
            op_name = id_to_op.get(pid)
            if op_name is not None and pid in bank.ids():
                try:
                    em, loss, tok_acc = _evaluate_single_primitive_on_support(
                        core, bank, op_name, pid, support_examples
                    )
                except Exception:
                    em, loss, tok_acc = 0.0, MAX_CLAMPED_LOSS, 0.0
            else:
                # Distractor or non-mapped primitive
                em, loss, tok_acc = 0.0, MAX_CLAMPED_LOSS, 0.0
            direct_results.append((pid, em, loss, tok_acc))

    # Identify best direct candidate and runner-up
    # Primary sort by EM descending, secondary by loss ascending
    direct_results.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    best_pid, best_em, best_loss, best_tok_acc = direct_results[0]

    runner_up_em: float | None = None
    runner_up_loss: float | None = None
    direct_margin = 0.0

    if len(direct_results) > 1:
        runner_up_pid, runner_up_em, runner_up_loss, _ = direct_results[1]
        direct_margin = max(0.0, best_em - runner_up_em)
    else:
        direct_margin = best_em

    # ---------------------------------------------------------
    # 3. Composition Search Evidence
    # ---------------------------------------------------------
    comp_em = 0.0
    comp_loss = float("inf")
    comp_depth = 1
    comp_recipe: tuple[str, ...] | None = None
    comp_eval_count = 0
    comp_prune_count = 0

    if cfg.allow_composition_search:
        available_ops = [id_to_op[pid] for pid in active_candidate_ids if pid in id_to_op]
        try:
            search_res = search_composition_recipe(
                core=core,
                bank=bank,
                op_to_id=op_to_id,
                adaptation_examples=support_examples,
                available_operations=available_ops,
                max_depth=cfg.composition_max_depth,
                beam_width=cfg.composition_beam_width,
                early_stop_exact_match=cfg.composition_early_stop_em,
            )
            comp_em = search_res.exact_match_adapt
            comp_loss = search_res.loss_adapt
            comp_depth = len(search_res.candidate_operations)
            comp_recipe = search_res.candidate_operations
            comp_eval_count = search_res.candidates_evaluated
            comp_prune_count = search_res.candidates_pruned
        except RuntimeError:
            # Search discovered no structurally valid recipe
            comp_em = 0.0
            comp_loss = MAX_CLAMPED_LOSS
            comp_depth = 1
            comp_recipe = None

    comp_improvement_em = max(0.0, comp_em - best_em)
    comp_improvement_loss = (
        max(0.0, best_loss - comp_loss) if not math.isinf(comp_loss) else 0.0
    )

    elapsed = time.perf_counter() - start_time

    return AdequacyEvidence(
        direct_em=best_em,
        direct_loss=best_loss,
        direct_token_acc=best_tok_acc,
        direct_primitive_id=best_pid,
        direct_runner_up_em=runner_up_em,
        direct_runner_up_loss=runner_up_loss,
        direct_margin=direct_margin,
        direct_candidates_evaluated=len(direct_results),
        composition_em=comp_em,
        composition_loss=comp_loss,
        composition_depth=comp_depth,
        composition_recipe=comp_recipe,
        composition_improvement_em=comp_improvement_em,
        composition_improvement_loss=comp_improvement_loss,
        composition_candidates_evaluated=comp_eval_count,
        composition_candidates_pruned=comp_prune_count,
        router_confidence=router_confidence,
        router_margin=router_margin,
        router_entropy=router_entropy,
        proposed_primitive_id=top1_pid,
        runner_up_primitive_id=top2_pid,
        recurrence_key_similarity=max_key_sim,
        prototype_similarity=max_proto_sim,
        support_size=len(support_examples),
        compute_time_seconds=elapsed,
        metadata={
            "direct_eval_k": cfg.direct_eval_k,
            "composition_max_depth": cfg.composition_max_depth,
            "composition_beam_width": cfg.composition_beam_width,
        },
    )
