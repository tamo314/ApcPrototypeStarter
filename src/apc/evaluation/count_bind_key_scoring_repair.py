# ruff: noqa: E501
"""Task B-C005R3-006: COUNT<->BIND Key/Scoring Repair.

ADR-0081 localized a `KEY_SCORING_BOTTLENECK` for the `COUNT->BIND` /
`BIND->COUNT` L3 relation (regate_sealed seed 24 / model_seed=4,
bank_size=128: `primitive_call_top1` 0.40 / 0.628). `B-C005R3-004` (ADR-0085)
already found this specific failure does **not** reproduce on `development`
seeds 10-14 under the existing, unmodified R1/R2 recipe (`R2` there already
measures 1.0 / 0.984), and flagged this task `NEEDS_SCOPE_REVIEW`. Before
training anything, this module performs the task doc's mandatory
"修正前の必須確認" (pre-repair diagnostics): key/physical-ID correspondence,
key normalization/norm, the exact planned trainable-parameter scope, an
empirical check for any artificial score offset, and the relation-exposure
classification -- so a decision to train (or not) is grounded in what is
actually true of the current code and checkpoints, not assumed.

Because R0 (the untouched frozen parent) already measures >=0.95 top-1 for
both directions on `development` seeds (R3-004's own numbers), there is no
COUNT<->BIND confusion to repair *on this partition*. This task still
implements and evaluates the scoped repair mechanism the task doc requires
(the original sealed-partition failure remains inaccessible under the
current sealed-access rules, so it cannot be re-diagnosed or re-repaired
here) and reports the resulting near-absence of measurable delta honestly,
per the task doc's own escape hatch ("十分なdeltaがない場合も、そのまま報告する").

Scope actually implemented (deliberately narrower than `B-C005R1`'s existing
`train_repaired_router_and_scorer`, which is not modified and remains
available as the "normal objective control"):

- Task Encoder, `router.query_proj`, `ArgumentScorer` (never instantiated),
  the verifier, and every primitive are frozen for the whole task.
- Of the router's keys, only COUNT's and BIND's have `requires_grad=True`;
  every other key (including SELECT's, despite `SELECT-BIND` sharing BIND's
  physical primitive per R3-002/ADR-0083's coupling finding) is held at
  `requires_grad=False` and its bit-for-bit invariance is empirically
  confirmed afterward (`build_freeze_audit`), not merely inferred from code.
- Two candidate scoped mechanisms are trained and compared on an internal,
  disjoint selection split (never the final Gate query split): `scoped_ce`
  (plain cross-entropy restricted to the two trainable keys) and
  `scoped_pairwise_margin` (an explicit margin loss against the REAL
  registered competitor key from `_RELATED_OPERATION`, rather than R2's
  synthetic random near-neighbor negative). One is selected per the task
  doc's "development の該当pairと独立validationで一つ選ぶ".
- Frozen baseline (R0) and the existing generic repair recipe (R2) are
  recomputed fresh via the unmodified `retrieval_repair_benchmark` functions,
  as the "frozen baseline" and "normal objective control" the task doc
  requires for comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
)
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.relation_split_protocol import (
    FULL_BANK_16_OPERATIONS,
    assert_sealed_access_permitted,
    classify_group_exposure,
)
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    RetrievalRepairConfig,
    evaluate_legacy_regression,
    evaluate_repair_cell,
    train_repaired_router_and_scorer,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.router import Router
from apc.primitives.routing_losses import CombinedRoutingLoss, RankingLossConfig
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = DEFAULT_DEV_SEEDS
TARGET_OPERATIONS: Final[tuple[str, ...]] = ("COUNT", "BIND")
DIRECTION_LABEL: Final[dict[str, str]] = {"COUNT": "COUNT->BIND", "BIND": "BIND->COUNT"}
DISCLOSURE_ONLY_OPERATION: Final[str] = "SELECT"  # SELECT->BIND: reported, never gated (task doc scope limit).
_L3: Final[HardNegativeLevel] = HardNegativeLevel.L3_SEMANTICALLY_RELATED
VARIANTS: Final[tuple[str, ...]] = ("scoped_ce", "scoped_pairwise_margin")

# Original ADR-0081 finding this task's mandate refers to, cited (never
# rewritten) for context; R3-004/ADR-0085 already found it NOT_REPRODUCED_ON_V2
# on `development` seeds -- see `runs/phase_b_b2_post_d2/r3_004_paired_baseline/
# failure_reproduction_matrix.json`.
ORIGINAL_ADR0081_REFERENCE: Final[dict[str, dict[str, Any]]] = {
    "COUNT->BIND": {
        "source": "runs/phase_b_b2_second_diagnostic/representation_stage_summary.json (regate_sealed/R2_frozen_post_repair, bank_size=128)",
        "control_a_current_path_top1": 0.4,
        "classification": "KEY_SCORING_BOTTLENECK",
    },
    "BIND->COUNT": {
        "source": "runs/phase_b_b2_second_diagnostic/representation_stage_summary.json (regate_sealed/R2_frozen_post_repair, bank_size=128)",
        "control_a_current_path_top1": 0.628125,
        "classification": "KEY_SCORING_BOTTLENECK",
    },
}
R3_004_REFERENCE: Final[dict[str, dict[str, Any]]] = {
    "COUNT->BIND": {
        "source": "runs/phase_b_b2_post_d2/r3_004_paired_baseline/failure_reproduction_matrix.json",
        "status": "NOT_REPRODUCED_ON_V2",
        "measured_v2_development_R2_primitive_call_top1_mean": 1.0,
    },
    "BIND->COUNT": {
        "source": "runs/phase_b_b2_post_d2/r3_004_paired_baseline/failure_reproduction_matrix.json",
        "status": "NOT_REPRODUCED_ON_V2",
        "measured_v2_development_R2_primitive_call_top1_mean": 0.984375,
    },
}


@dataclass(frozen=True)
class CountBindKeyScoringConfig:
    """Explicit configuration for Task B-C005R3-006."""

    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    bank_size: int = 128
    support_examples: int = 32
    query_examples: int = 64
    selection_query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    top_k: int = 5
    adequacy_exact_match_threshold: float = 0.95
    gate_top1_threshold: float = 0.95
    gate_topk_threshold: float = 0.99
    gate_legacy_regression_pp_max: float = 1.0
    variants: tuple[str, ...] = VARIANTS
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_006_count_bind_key_scoring_repair")

    def __post_init__(self) -> None:
        if not self.development_seeds:
            raise ValueError("development_seeds must be non-empty")
        if self.bank_size not in DEFAULT_BANK_SIZES:
            raise ValueError(f"bank_size must be one of {DEFAULT_BANK_SIZES}")
        if self.support_examples < 1 or self.query_examples < 1 or self.selection_query_examples < 1:
            raise ValueError("support_examples, query_examples, and selection_query_examples must be positive")
        if self.router_steps < 1:
            raise ValueError("router_steps must be positive")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be in [1, 5]")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")
        if not 0.0 < self.gate_top1_threshold <= 1.0 or not 0.0 < self.gate_topk_threshold <= 1.0:
            raise ValueError("gate_top1_threshold and gate_topk_threshold must be in (0, 1]")
        if self.gate_legacy_regression_pp_max < 0.0:
            raise ValueError("gate_legacy_regression_pp_max must be non-negative")
        if not self.variants or set(self.variants) - set(VARIANTS):
            raise ValueError(f"variants must be a non-empty subset of {VARIANTS}")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir)
        return data


# ---------------------------------------------------------------------------
# 1. Mandatory pre-repair diagnostics ("修正前の必須確認").
# ---------------------------------------------------------------------------


def audit_key_id_correspondence(router: Router, op_to_id: dict[str, int]) -> dict[str, Any]:
    """Confirm every bank-16 operation has exactly one, correctly-registered router key."""
    ids = list(op_to_id.values())
    unique_ids = set(ids)
    duplicate_ids_found = len(ids) != len(unique_ids)
    missing = [op for op, pid in op_to_id.items() if not router.has_primitive(pid)]
    return {
        "num_operations": len(op_to_id),
        "num_unique_primitive_ids": len(unique_ids),
        "duplicate_ids_found": duplicate_ids_found,
        "operations_missing_router_key": missing,
        "bijective_and_registered": not duplicate_ids_found and not missing,
    }


def audit_key_norms(router: Router, op_to_id: dict[str, int]) -> dict[str, Any]:
    """Report per-operation key L2 norms and the scoring function's normalization
    behavior. `score_fn='dot'` (confirmed below) means raw key-norm magnitude
    directly scales dot-product scores -- a norm imbalance between COUNT and
    BIND could bias routing independent of semantic content, so this is
    measured, not assumed."""
    norms = {op: float(router.key_parameter(pid).norm(p=2).item()) for op, pid in op_to_id.items()}
    values = list(norms.values())
    mean_norm = sum(values) / len(values) if values else None
    count_norm = norms.get("COUNT")
    bind_norm = norms.get("BIND")
    return {
        "per_operation_key_norm": norms,
        "population_mean_key_norm": mean_norm,
        "count_key_norm": count_norm,
        "bind_key_norm": bind_norm,
        "count_bind_norm_ratio": (count_norm / bind_norm) if count_norm is not None and bind_norm else None,
        "score_fn": router.config.score_fn,
        "normalization_applied_to_scoring": router.config.score_fn == "cosine",
    }


def audit_trainable_parameter_scope(
    candidate_ids: Sequence[int], count_id: int, bind_id: int
) -> dict[str, Any]:
    """The exact, planned trainable-parameter scope for this task, stated
    before any training runs (confirmed empirically afterward by
    `build_freeze_audit`)."""
    return {
        "planned_trainable_key_ids": [count_id, bind_id],
        "planned_trainable_operations": ["COUNT", "BIND"],
        "planned_frozen_key_ids": sorted(pid for pid in candidate_ids if pid not in (count_id, bind_id)),
        "query_proj_trainable": False,
        "argument_scorer_instantiated": False,
        "task_encoder_trainable": False,
        "primitive_bank_trainable": False,
        "verifier_touched": False,
    }


def audit_no_artificial_score_offset(
    core: Any, router: Router, candidate_ids: Sequence[int], examples: Sequence[Any]
) -> dict[str, Any]:
    """Empirically confirm the routing score is exactly `dot(query_proj(z), key)`
    with no hidden additive/multiplicative offset, by recomputing it manually
    element-by-element and comparing against the router's own batched formula."""
    z = extract_task_representations(core, list(examples))
    candidate_list = list(candidate_ids)
    with torch.no_grad():
        query = router.query_proj(z)
        keys = router._stacked_keys(candidate_list)
        formula_scores = query @ keys.transpose(0, 1)
        manual_scores = torch.stack(
            [torch.stack([torch.dot(query[i], keys[j]) for j in range(keys.shape[0])]) for i in range(query.shape[0])]
        )
    max_abs_difference = float((formula_scores - manual_scores).abs().max().item())
    matches = torch.allclose(formula_scores, manual_scores, atol=1e-5)
    return {
        "method": "recompute score(z, key) = dot(query_proj(z), key) manually per (example, candidate) pair and compare against the router's batched matmul formula",
        "n_examples": len(examples),
        "n_candidates": len(candidate_list),
        "max_abs_difference": max_abs_difference,
        "matches_within_tolerance": matches,
        "conclusion": "NO_ARTIFICIAL_OFFSET_FOUND" if matches else "UNEXPLAINED_DISCREPANCY_FOUND",
    }


def build_pre_repair_diagnostics(
    core: Any,
    router: Router,
    candidate_ids: Sequence[int],
    op_to_id: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """The task doc's mandatory pre-repair confirmation, run once per seed on
    the untouched frozen (R0) router before any scoped training."""
    count_id = op_to_id["COUNT"]
    bind_id = op_to_id["BIND"]

    id_correspondence = audit_key_id_correspondence(router, op_to_id)
    key_norms = audit_key_norms(router, op_to_id)
    trainable_scope = audit_trainable_parameter_scope(candidate_ids, count_id, bind_id)

    probe_examples = generate_benchmark_examples(seed * 910_001 + count_id, 8, operation="COUNT", split="dev")
    offset_audit = audit_no_artificial_score_offset(core, router, candidate_ids, probe_examples)

    legacy_style_exposure = classify_group_exposure(
        held_out_ops=frozenset(op for op in op_to_id if op not in ("COUNT", "BIND")),
        positive_ops_seen=["COUNT", "BIND"],
        candidate_list_ops=list(op_to_id.keys()),
    )

    return {
        "seed": seed,
        "key_id_correspondence": id_correspondence,
        "key_normalization_and_norms": key_norms,
        "planned_trainable_parameter_scope": trainable_scope,
        "artificial_score_offset_check": offset_audit,
        "relation_exposure_classification_legacy_style": legacy_style_exposure,
        "relation_exposure_note": (
            "R3-002's classify_group_exposure() (ADR-0083) would label this MINING_HOLDOUT_ONLY "
            "because every non-target op remains a column of the softmax denominator. This task's "
            "scoped training additionally sets requires_grad=False on every non-target key AND on "
            "query_proj, so a non-target op's own score for any fixed query is provably unchanged "
            "(neither its key nor query_proj moved) -- confirmed bit-for-bit by frozen_state_audit.json, "
            "not merely inferred from code. Only COUNT/BIND's own scores can move."
        ),
        "all_pre_repair_checks_clean": (
            id_correspondence["bijective_and_registered"] and offset_audit["matches_within_tolerance"]
        ),
    }


# ---------------------------------------------------------------------------
# 2. Two candidate scoped repair mechanisms (COUNT/BIND keys only).
# ---------------------------------------------------------------------------


def train_count_bind_scoped_repair(
    router: Router,
    candidate_ids: Sequence[int],
    count_id: int,
    bind_id: int,
    z_count_train: torch.Tensor,
    z_bind_train: torch.Tensor,
    *,
    variant: str,
    router_steps: int,
    router_lr: float,
    ranking_margin: float,
    ranking_beta: float,
    seed: int,
    device: torch.device,
) -> Router:
    """Train ONLY the COUNT and BIND router keys (`requires_grad=True`);
    every other key and `query_proj` stay `requires_grad=False` for the
    whole run -- unlike `train_repaired_router_and_scorer`, which puts the
    full resident `candidate_list` under one optimizer (R3-002/ADR-0083's
    `MINING_HOLDOUT_ONLY` structural-ceiling finding). Two variants:

    - ``scoped_ce``: plain cross-entropy over the full candidate softmax,
      gradient restricted to the two trainable keys by `requires_grad`.
    - ``scoped_pairwise_margin``: an explicit margin loss against the REAL
      registered competitor key from `_RELATED_OPERATION` (BIND's key when
      training on a COUNT example, COUNT's key when training on a BIND
      example) rather than R2's synthesized random near-neighbor negative.
    """
    import copy

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    new_router = copy.deepcopy(router).to(device)
    candidate_list = list(candidate_ids)
    pid_to_class = {pid: idx for idx, pid in enumerate(candidate_list)}

    new_router.query_proj.requires_grad_(False)
    for pid_str, param in new_router._keys.items():
        param.requires_grad_(int(pid_str) in (count_id, bind_id))

    key_params = [new_router.key_parameter(count_id), new_router.key_parameter(bind_id)]
    optimizer = torch.optim.AdamW(key_params, lr=router_lr, weight_decay=1e-4)
    ranking_loss_fn = CombinedRoutingLoss(RankingLossConfig(margin=ranking_margin, beta=ranking_beta))

    rng = torch.Generator(device="cpu").manual_seed(seed * 8_101 + 17)
    z_count_train = z_count_train.to(device)
    z_bind_train = z_bind_train.to(device)

    new_router.train()
    for _ in range(router_steps):
        idx_c = int(torch.randint(0, len(z_count_train), (1,), generator=rng).item())
        idx_b = int(torch.randint(0, len(z_bind_train), (1,), generator=rng).item())
        z_batch = torch.stack([z_count_train[idx_c], z_bind_train[idx_b]], dim=0)
        targets = torch.tensor([pid_to_class[count_id], pid_to_class[bind_id]], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        query = new_router.query_proj(z_batch)
        keys = new_router._stacked_keys(candidate_list)
        logits = query @ keys.transpose(0, 1)

        if variant == "scoped_ce":
            loss = F.cross_entropy(logits, targets)
        else:  # scoped_pairwise_margin
            hard_neg_keys = torch.stack(
                [new_router.key_parameter(bind_id), new_router.key_parameter(count_id)], dim=0
            )
            hard_neg_scores = (query * hard_neg_keys).sum(dim=-1, keepdim=True)
            loss, _ = ranking_loss_fn(logits, targets, hard_neg_scores)

        loss.backward()
        optimizer.step()

    new_router.eval()
    return new_router


# ---------------------------------------------------------------------------
# 3. Freeze audit and checkpoint hashing.
# ---------------------------------------------------------------------------


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_dict_hash(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def build_freeze_audit(
    core: Any,
    base_router: Router,
    scoped_router: Router,
    candidate_ids: Sequence[int],
    count_id: int,
    bind_id: int,
) -> dict[str, Any]:
    """Bit-for-bit confirmation of what actually changed: Task Encoder is
    never touched by this task (hashed once as a sanity record, not a
    before/after -- `core` is shared, not copied, across conditions);
    `query_proj` and every non-COUNT/BIND key must be byte-identical between
    the frozen parent and the scoped-repair router."""
    task_encoder_hash = _state_dict_hash(core.model)
    query_proj_unchanged = torch.equal(base_router.query_proj.weight, scoped_router.query_proj.weight) and torch.equal(
        base_router.query_proj.bias, scoped_router.query_proj.bias
    )
    non_target_key_unchanged: dict[str, bool] = {}
    for pid in candidate_ids:
        if pid in (count_id, bind_id):
            continue
        non_target_key_unchanged[str(pid)] = torch.equal(
            base_router.key_parameter(pid), scoped_router.key_parameter(pid)
        )
    all_non_target_unchanged = all(non_target_key_unchanged.values()) if non_target_key_unchanged else True
    count_key_changed = not torch.equal(base_router.key_parameter(count_id), scoped_router.key_parameter(count_id))
    bind_key_changed = not torch.equal(base_router.key_parameter(bind_id), scoped_router.key_parameter(bind_id))

    return {
        "task_encoder_state_hash": task_encoder_hash,
        "task_encoder_touched": False,
        "query_proj_unchanged": query_proj_unchanged,
        "non_target_key_unchanged_by_pid": non_target_key_unchanged,
        "all_non_target_keys_unchanged": all_non_target_unchanged,
        "count_key_changed_by_training": count_key_changed,
        "bind_key_changed_by_training": bind_key_changed,
        "freeze_audit_passed": query_proj_unchanged and all_non_target_unchanged,
    }


def build_checkpoint_hashes(
    base_router: Router, scoped_router: Router, candidate_ids: Sequence[int], count_id: int, bind_id: int
) -> dict[str, Any]:
    return {
        "parent_router_full_state_hash": _state_dict_hash(base_router),
        "scoped_repair_router_full_state_hash": _state_dict_hash(scoped_router),
        "count_key_hash": {
            "before": _tensor_hash(base_router.key_parameter(count_id)),
            "after": _tensor_hash(scoped_router.key_parameter(count_id)),
        },
        "bind_key_hash": {
            "before": _tensor_hash(base_router.key_parameter(bind_id)),
            "after": _tensor_hash(scoped_router.key_parameter(bind_id)),
        },
        "sample_non_target_key_hash": {
            str(pid): {
                "before": _tensor_hash(base_router.key_parameter(pid)),
                "after": _tensor_hash(scoped_router.key_parameter(pid)),
            }
            for pid in list(candidate_ids)[:3]
            if pid not in (count_id, bind_id)
        },
    }


def build_loss_exposure_log(
    candidate_ids: Sequence[int], operation_by_id: dict[int, str], count_id: int, bind_id: int
) -> dict[str, Any]:
    per_op: dict[str, str] = {}
    for pid in candidate_ids:
        op = operation_by_id.get(pid, f"primitive_{pid}")
        per_op[op] = (
            "POSITIVE_TRAINING_TARGET_PARAMETER_UPDATED"
            if pid in (count_id, bind_id)
            else "FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE"
        )
    return {
        "classification_scheme": {
            "POSITIVE_TRAINING_TARGET_PARAMETER_UPDATED": "op is COUNT or BIND; its router key has requires_grad=True and receives optimizer updates every step.",
            "FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE": "op's key remains a fixed column of the softmax denominator (used to compute the CE/ranking loss for COUNT/BIND examples) but requires_grad=False, so its own parameter value is provably unchanged -- confirmed per-seed by frozen_state_audit.json's key-level equality checks, not merely inferred from code.",
        },
        "per_operation_exposure": per_op,
        "comparison_to_r3_002_finding": (
            "R3-002/ADR-0083 found the EXISTING train_repaired_router_and_scorer recipe structurally "
            "caps L3 holdout at MINING_HOLDOUT_ONLY because it puts the full resident candidate_list "
            "under one optimizer; changing that function was explicitly out of scope through R3-010. "
            "This task adds a NEW, narrower function instead (train_repaired_router_and_scorer itself "
            "is untouched and remains available as the 'normal objective control') that achieves a "
            "strictly stronger, empirically-confirmed guarantee for every op other than COUNT/BIND: "
            "FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE, not just a mining-pool argument."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Orchestration: one pass per development seed.
# ---------------------------------------------------------------------------


def _final_split(seed: int, bank_size: int, target_id: int, target_op: str, support_n: int, query_n: int) -> tuple[list[Any], list[Any]]:
    """The same support/query seed-offset convention as `retrieval_repair_benchmark`
    / `paired_baseline_repair` (support: *40_000, query: *50_000+100), so
    R0/R2/scoped numbers are directly, apples-to-apples comparable."""
    support = generate_benchmark_examples(seed * 40_000 + bank_size + target_id, support_n, operation=target_op, split="dev")
    query = generate_benchmark_examples(seed * 50_000 + bank_size + target_id + 100, query_n, operation=target_op, split="dev")
    return list(support), list(query)


def _selection_split(seed: int, bank_size: int, target_id: int, target_op: str, support_n: int, query_n: int) -> tuple[list[Any], list[Any]]:
    """A disjoint example split (different seed offsets) used ONLY to select
    between the two candidate variants -- never reused for the final Gate
    numbers, so variant selection cannot leak into the reported metrics."""
    support = generate_benchmark_examples(seed * 70_000 + bank_size + target_id, support_n, operation=target_op, split="dev")
    query = generate_benchmark_examples(seed * 80_000 + bank_size + target_id + 100, query_n, operation=target_op, split="dev")
    return list(support), list(query)


def run_count_bind_key_scoring_repair(config: CountBindKeyScoringConfig) -> dict[str, Any]:
    """Executes B-C005R3-006 end to end on `development` seeds only."""
    start = time.perf_counter()
    assert_sealed_access_permitted(config.development_seeds, purpose="B-C005R3-006_count_bind_key_scoring_repair")
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    pre_repair_diagnostics: dict[str, Any] = {}
    per_variant_selection: dict[str, list[float]] = {variant: [] for variant in config.variants}
    per_seed_variant_cells: dict[str, dict[str, list[dict[str, Any]]]] = {variant: {} for variant in config.variants}
    per_seed_control_cells: dict[str, list[dict[str, Any]]] = {"R0": [], "R2": []}
    legacy_regression: dict[str, dict[str, float]] = {"R0": {}, "R2": {}, **{v: {} for v in config.variants}}
    freeze_audits: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, dict[str, Any]] = {}
    structure_snapshot: tuple[list[int], dict[int, str], int, int] | None = None

    for seed in config.development_seeds:
        set_seed(seed)
        base_hn_config = HardNegativeBenchmarkConfig(
            seeds=(seed,),
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )
        core, base_bank, base_router, op_to_id = _build_frozen_base_system(seed, base_hn_config)
        bank, router, candidate_ids, _semantic_ids, distractor_ids = build_scaled_bank_and_router(
            core, base_bank, base_router, op_to_id, config.bank_size, seed=seed
        )
        operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
        operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

        count_id = op_to_id["COUNT"]
        bind_id = op_to_id["BIND"]
        if structure_snapshot is None:
            structure_snapshot = (list(candidate_ids), dict(operation_by_id), count_id, bind_id)

        pre_repair_diagnostics[str(seed)] = build_pre_repair_diagnostics(core, router, candidate_ids, op_to_id, seed)

        dev_train_by_op = {
            op: generate_benchmark_examples(
                seed * 30_000 + config.bank_size + op_to_id[op], config.router_train_examples, operation=op, split="dev"
            )
            for op in op_to_id
        }
        z_count_train = extract_task_representations(core, dev_train_by_op["COUNT"])
        z_bind_train = extract_task_representations(core, dev_train_by_op["BIND"])

        # --- Frozen baseline (R0) and existing generic repair (R2), reused verbatim. ---
        for condition in ("R0", "R2"):
            active_router, _arg_scorer = train_repaired_router_and_scorer(
                core=core,
                router=router,
                candidate_ids=candidate_ids,
                operation_by_id=operation_by_id,
                dev_train_examples_by_op=dev_train_by_op,
                config=_shared_retrieval_repair_config(config, seed),
                condition=condition,
                device=core.device,
            )
            legacy_regression[condition][str(seed)] = evaluate_legacy_regression(
                core, bank, active_router, FULL_BANK_16_OPERATIONS, op_to_id, seed
            )
            for target_op in (*TARGET_OPERATIONS, DISCLOSURE_ONLY_OPERATION):
                target_id = op_to_id[target_op]
                support, query = _final_split(seed, config.bank_size, target_id, target_op, config.support_examples, config.query_examples)
                diag = evaluate_repair_cell(
                    core=core, bank=bank, router=active_router, argument_scorer=None,
                    candidate_ids=candidate_ids, operation_by_id=operation_by_id,
                    target_operation=target_op, target_id=target_id, level=_L3,
                    support_examples=support, query_examples=query, seed=seed,
                    top_k=config.top_k, adequacy_threshold=config.adequacy_exact_match_threshold,
                    arg_lambda=0.0, condition=condition, bank_size=config.bank_size,
                )
                per_seed_control_cells[condition].append(diag.to_dict())

        # --- Two candidate scoped repair mechanisms. ---
        for variant in config.variants:
            scoped_router = train_count_bind_scoped_repair(
                router, candidate_ids, count_id, bind_id, z_count_train, z_bind_train,
                variant=variant, router_steps=config.router_steps, router_lr=config.router_lr,
                ranking_margin=config.ranking_margin, ranking_beta=config.ranking_beta,
                seed=seed, device=core.device,
            )
            freeze_audits[f"{variant}_seed_{seed}"] = build_freeze_audit(
                core, router, scoped_router, candidate_ids, count_id, bind_id
            )
            checkpoint_hashes[f"{variant}_seed_{seed}"] = build_checkpoint_hashes(
                router, scoped_router, candidate_ids, count_id, bind_id
            )
            legacy_regression[variant][str(seed)] = evaluate_legacy_regression(
                core, bank, scoped_router, FULL_BANK_16_OPERATIONS, op_to_id, seed
            )

            selection_scores: list[float] = []
            for target_op in (*TARGET_OPERATIONS, DISCLOSURE_ONLY_OPERATION):
                target_id = op_to_id[target_op]

                sel_support, sel_query = _selection_split(
                    seed, config.bank_size, target_id, target_op, config.support_examples, config.selection_query_examples
                )
                sel_diag = evaluate_repair_cell(
                    core=core, bank=bank, router=scoped_router, argument_scorer=None,
                    candidate_ids=candidate_ids, operation_by_id=operation_by_id,
                    target_operation=target_op, target_id=target_id, level=_L3,
                    support_examples=sel_support, query_examples=sel_query, seed=seed,
                    top_k=config.top_k, adequacy_threshold=config.adequacy_exact_match_threshold,
                    arg_lambda=0.0, condition=f"{variant}_selection", bank_size=config.bank_size,
                )
                if target_op in TARGET_OPERATIONS:
                    selection_scores.append(sel_diag.primitive_call_top1)

                final_support, final_query = _final_split(
                    seed, config.bank_size, target_id, target_op, config.support_examples, config.query_examples
                )
                final_diag = evaluate_repair_cell(
                    core=core, bank=bank, router=scoped_router, argument_scorer=None,
                    candidate_ids=candidate_ids, operation_by_id=operation_by_id,
                    target_operation=target_op, target_id=target_id, level=_L3,
                    support_examples=final_support, query_examples=final_query, seed=seed,
                    top_k=config.top_k, adequacy_threshold=config.adequacy_exact_match_threshold,
                    arg_lambda=0.0, condition=variant, bank_size=config.bank_size,
                )
                per_seed_variant_cells[variant].setdefault(target_op, []).append(final_diag.to_dict())

            per_variant_selection[variant].append(sum(selection_scores) / len(selection_scores) if selection_scores else 0.0)

    assert structure_snapshot is not None
    snap_candidate_ids, snap_operation_by_id, snap_count_id, snap_bind_id = structure_snapshot
    loss_exposure_log = build_loss_exposure_log(snap_candidate_ids, snap_operation_by_id, snap_count_id, snap_bind_id)

    variant_selection = _build_variant_selection(per_variant_selection)
    chosen_variant = variant_selection["chosen_variant"]

    summary = _build_condition_relation_summary(per_seed_control_cells, per_seed_variant_cells, chosen_variant)
    pair_margin_report = _build_pair_margin_report(per_seed_control_cells, per_seed_variant_cells, chosen_variant)
    gate = _build_gate(summary, legacy_regression, chosen_variant, config)

    count_bind_scoring_repair = {
        "task": "B-C005R3-006",
        "development_seeds": list(config.development_seeds),
        "bank_size": config.bank_size,
        "pre_repair_diagnostics": {
            "by_seed": pre_repair_diagnostics,
            "all_seeds_clean": all(d["all_pre_repair_checks_clean"] for d in pre_repair_diagnostics.values()),
        },
        "original_reference": {
            "adr_0081": ORIGINAL_ADR0081_REFERENCE,
            "r3_004_adr_0085": R3_004_REFERENCE,
        },
        "variant_selection_summary": {
            "chosen_variant": chosen_variant,
            "per_variant_mean_selection_top1": variant_selection["per_variant_mean_selection_top1"],
        },
        "condition_relation_summary": summary,
        "legacy_routing_regression": _summarize_legacy_regression(legacy_regression, chosen_variant),
        "gate": gate,
    }

    frozen_state_audit = {
        "task": "B-C005R3-006",
        "by_variant_seed": freeze_audits,
        "all_freeze_audits_passed": all(a["freeze_audit_passed"] for a in freeze_audits.values()),
    }

    protocol = {
        "task": "B-C005R3-006",
        "gate": "local_repair_gate",
        "result": gate["result"],
        "chosen_variant": chosen_variant,
        "needs_scope_review_inherited_from": "B-C005R3-004 (ADR-0085): both directions NOT_REPRODUCED_ON_V2 on development seeds",
        "scientific_caveat": gate["scientific_caveat"],
        "downstream_note": (
            "This gate result applies only to the `development` partition (seeds 10-14). It does not "
            "confirm repair of the original ADR-0081 sealed-partition (regate_sealed seed 24, "
            "model_seed=4, bank_size=128) KEY_SCORING_BOTTLENECK finding, which remains inaccessible "
            "under current sealed-access rules (R3-011/R3-012 pathway required). SELECT->BIND was "
            "measured for disclosure only and is explicitly excluded from this task's Gate, per the "
            "task doc's own scope boundary."
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")
        (output_dir / "count_bind_scoring_repair.json").write_text(json.dumps(count_bind_scoring_repair, indent=2), encoding="utf-8")
        (output_dir / "pair_margin_report.json").write_text(json.dumps(pair_margin_report, indent=2), encoding="utf-8")
        (output_dir / "checkpoint_hashes.json").write_text(json.dumps(checkpoint_hashes, indent=2), encoding="utf-8")
        (output_dir / "frozen_state_audit.json").write_text(json.dumps(frozen_state_audit, indent=2), encoding="utf-8")
        (output_dir / "loss_exposure_log.json").write_text(json.dumps(loss_exposure_log, indent=2), encoding="utf-8")
        (output_dir / "variant_selection.json").write_text(json.dumps(variant_selection, indent=2), encoding="utf-8")
        (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "count_bind_scoring_repair": count_bind_scoring_repair,
        "pair_margin_report": pair_margin_report,
        "checkpoint_hashes": checkpoint_hashes,
        "frozen_state_audit": frozen_state_audit,
        "loss_exposure_log": loss_exposure_log,
        "variant_selection": variant_selection,
        "protocol": protocol,
    }


def _shared_retrieval_repair_config(config: CountBindKeyScoringConfig, seed: int) -> RetrievalRepairConfig:
    """The config passed to the unmodified `train_repaired_router_and_scorer`
    for the R0/R2 controls -- shares this task's own ranking/lr/step
    hyperparameters so R0/R2 are on equal footing with the scoped variants."""
    return RetrievalRepairConfig(
        seeds=(seed,),
        bank_sizes=(config.bank_size,),
        router_train_examples=config.router_train_examples,
        router_steps=config.router_steps,
        router_lr=config.router_lr,
        top_k=config.top_k,
        ranking_margin=config.ranking_margin,
        ranking_beta=config.ranking_beta,
        adequacy_exact_match_threshold=config.adequacy_exact_match_threshold,
        device=config.device,
        bank_checkpoint_dir=config.bank_checkpoint_dir,
    )


# ---------------------------------------------------------------------------
# 5. Aggregation helpers.
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _build_variant_selection(per_variant_selection: dict[str, list[float]]) -> dict[str, Any]:
    per_variant_mean = {variant: _mean(scores) for variant, scores in per_variant_selection.items()}
    best_score = max((score for score in per_variant_mean.values() if score is not None), default=None)
    tied = [v for v, s in per_variant_mean.items() if s is not None and best_score is not None and abs(s - best_score) < 1e-9]
    if len(tied) > 1 and "scoped_pairwise_margin" in tied:
        chosen = "scoped_pairwise_margin"
        tie_break_applied = True
    else:
        chosen = tied[0] if tied else next(iter(per_variant_mean))
        tie_break_applied = len(tied) > 1
    return {
        "task": "B-C005R3-006",
        "selection_split": "disjoint from both training and final-Gate query examples (seed*70_000/*80_000 offsets)",
        "per_variant_mean_selection_top1": per_variant_mean,
        "per_seed_selection_scores": per_variant_selection,
        "chosen_variant": chosen,
        "tie_break_applied": tie_break_applied,
        "tie_break_rule": "on a tie within 1e-9, prefer scoped_pairwise_margin (the mechanism directly informed by the real registered _RELATED_OPERATION competitor, over the generic scoped_ce)",
        "selection_leakage": "none: neither variant's final Gate numbers (count_bind_scoring_repair.json) use any example from this selection split",
    }


def _build_condition_relation_summary(
    per_seed_control_cells: dict[str, list[dict[str, Any]]],
    per_seed_variant_cells: dict[str, dict[str, list[dict[str, Any]]]],
    chosen_variant: str,
) -> dict[str, Any]:
    def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_seeds": len(cells),
            "primitive_call_top1_mean": _mean([c["primitive_call_top1"] for c in cells]),
            "primitive_call_topk_mean": _mean([c["primitive_call_topk"] for c in cells]),
            "score_margin_mean": _mean([c["score_margin"] for c in cells]),
            "unselected_forward_calls_total": sum(c["unselected_forward_calls"] for c in cells),
            "leak_audit_all_passed": all(c["leak_audit_passed"] for c in cells),
            "sparse_execution_all_passed": all(c["sparse_execution_passed"] for c in cells),
        }

    result: dict[str, Any] = {"R0": {}, "R2": {}, "R3006_chosen": {}}
    for target_op in (*TARGET_OPERATIONS, DISCLOSURE_ONLY_OPERATION):
        relation = DIRECTION_LABEL.get(target_op, f"{target_op}->BIND_disclosure_only")
        for condition in ("R0", "R2"):
            cells = [c for c in per_seed_control_cells[condition] if c["target_operation"] == target_op]
            result[condition][relation] = summarize(cells)
        chosen_cells = per_seed_variant_cells[chosen_variant].get(target_op, [])
        result["R3006_chosen"][relation] = summarize(chosen_cells)
    result["chosen_variant"] = chosen_variant
    return result


def _build_pair_margin_report(
    per_seed_control_cells: dict[str, list[dict[str, Any]]],
    per_seed_variant_cells: dict[str, dict[str, list[dict[str, Any]]]],
    chosen_variant: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for condition, cells in per_seed_control_cells.items():
        for cell in cells:
            entries.append(
                {
                    "condition": condition,
                    "relation": DIRECTION_LABEL.get(cell["target_operation"], f"{cell['target_operation']}->BIND_disclosure_only"),
                    "seed": cell["seed"],
                    "score_margin": cell["score_margin"],
                    "candidate_rank": cell["candidate_rank"],
                }
            )
    for target_op, cells in per_seed_variant_cells[chosen_variant].items():
        for cell in cells:
            entries.append(
                {
                    "condition": "R3006_chosen",
                    "relation": DIRECTION_LABEL.get(target_op, f"{target_op}->BIND_disclosure_only"),
                    "seed": cell["seed"],
                    "score_margin": cell["score_margin"],
                    "candidate_rank": cell["candidate_rank"],
                }
            )
    return {"task": "B-C005R3-006", "chosen_variant": chosen_variant, "entries": entries}


def _summarize_legacy_regression(legacy_regression: dict[str, dict[str, float]], chosen_variant: str) -> dict[str, Any]:
    r0_scores = list(legacy_regression["R0"].values())
    mean_r0 = _mean(r0_scores) or 1.0
    by_condition: dict[str, Any] = {}
    for condition, per_seed in legacy_regression.items():
        scores = list(per_seed.values())
        mean_score = _mean(scores) if scores else None
        drop_pp = max(0.0, mean_r0 - mean_score) * 100.0 if mean_score is not None else None
        by_condition[condition] = {"per_seed": per_seed, "mean": mean_score, "drop_pp_vs_r0": drop_pp}
    return {
        "operations_evaluated": list(FULL_BANK_16_OPERATIONS),
        "mean_r0": mean_r0,
        "by_condition": by_condition,
        "chosen_variant_drop_pp_vs_r0": by_condition[chosen_variant]["drop_pp_vs_r0"],
    }


def _build_gate(
    summary: dict[str, Any],
    legacy_regression: dict[str, dict[str, float]],
    chosen_variant: str,
    config: CountBindKeyScoringConfig,
) -> dict[str, Any]:
    r0_scores = list(legacy_regression["R0"].values())
    mean_r0_legacy = _mean(r0_scores) or 1.0
    chosen_scores = list(legacy_regression[chosen_variant].values())
    mean_chosen_legacy = _mean(chosen_scores)
    legacy_drop_pp = max(0.0, mean_r0_legacy - mean_chosen_legacy) * 100.0 if mean_chosen_legacy is not None else None

    per_direction: dict[str, Any] = {}
    all_pass = True
    for target_op in TARGET_OPERATIONS:
        relation = DIRECTION_LABEL[target_op]
        chosen = summary["R3006_chosen"][relation]
        r0 = summary["R0"][relation]
        top1 = chosen["primitive_call_top1_mean"] or 0.0
        topk = chosen["primitive_call_topk_mean"] or 0.0
        top1_pass = top1 >= config.gate_top1_threshold
        topk_pass = topk >= config.gate_topk_threshold
        per_direction[relation] = {
            "measured_top1": top1,
            "top1_threshold": config.gate_top1_threshold,
            "top1_pass": top1_pass,
            "measured_topk": topk,
            "topk_threshold": config.gate_topk_threshold,
            "topk_pass": topk_pass,
            "unselected_forward_calls_total": chosen["unselected_forward_calls_total"],
            "leak_audit_all_passed": chosen["leak_audit_all_passed"],
            "r0_frozen_baseline_top1": r0["primitive_call_top1_mean"],
            "r0_already_meets_threshold": (r0["primitive_call_top1_mean"] or 0.0) >= config.gate_top1_threshold,
            "delta_over_r0": top1 - (r0["primitive_call_top1_mean"] or 0.0),
        }
        if not (top1_pass and topk_pass and chosen["unselected_forward_calls_total"] == 0 and chosen["leak_audit_all_passed"]):
            all_pass = False

    legacy_pass = legacy_drop_pp is not None and legacy_drop_pp <= config.gate_legacy_regression_pp_max
    all_pass = all_pass and legacy_pass

    r0_already_passes_both = all(per_direction[rel]["r0_already_meets_threshold"] for rel in per_direction)
    scientific_caveat = (
        "R0 (the untouched frozen parent) already meets both directions' thresholds on `development` "
        "seeds -- consistent with B-C005R3-004/ADR-0085's NOT_REPRODUCED_ON_V2 finding for this exact "
        "relation. This VALIDATION_PASS therefore does NOT demonstrate that the scoped repair mechanism "
        "fixes the original ADR-0081 sealed-partition KEY_SCORING_BOTTLENECK; it demonstrates only that "
        "the mechanism (a) is implemented correctly, (b) does not regress an already-passing partition, "
        "and (c) satisfies every stated freeze/leak/exposure constraint. Repairing (or re-diagnosing) "
        "the original sealed-partition failure requires sealed access via the R3-011/R3-012 pathway."
        if r0_already_passes_both
        else "R0 does not already meet both directions' thresholds on this partition; the scoped repair's delta over R0 is load-bearing for this Gate result."
    )

    return {
        "result": "VALIDATION_PASS" if all_pass else "FAIL",
        "chosen_variant": chosen_variant,
        "per_direction": per_direction,
        "legacy_routing_regression_pp": legacy_drop_pp,
        "legacy_routing_regression_threshold_pp": config.gate_legacy_regression_pp_max,
        "legacy_routing_regression_pass": legacy_pass,
        "r0_already_meets_both_thresholds": r0_already_passes_both,
        "select_bind_excluded_from_gate": True,
        "scientific_caveat": scientific_caveat,
    }
