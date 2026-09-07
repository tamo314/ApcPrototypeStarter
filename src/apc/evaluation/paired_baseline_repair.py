# ruff: noqa: E501
"""Task B-C005R3-004: Paired Frozen Baseline & Comparison Repair.

Establishes the repair starting point on the reproducible v2 fixture (ADR-0082)
without repeating the full second-diagnostic phase: this module re-evaluates
only the specific mechanisms the next repair tasks (`B-C005R3-006`..`-009`)
need a starting reference for -- COUNT<->BIND key/scoring, SELECT/BIND
argument handling, and SHIFT reference adequacy -- on the `development`
partition registered by `B-C005R3-002` (seeds 10-14; never `validation`
15-19 or `sealed_v2` 30-34, enforced by `assert_sealed_access_permitted`).

Per the task doc this task does **not** require a performance PASS. It
requires the *comparison itself* to be valid (same candidate graph/support
construction reused across R0/R1/R2, only the router/scorer weights differ
per condition) and each individual mechanism's reproduction status on v2 to
be reported honestly, including `NOT_REPRODUCED_ON_V2` where applicable --
never erasing the original (regate_sealed, bank_size=128) D2 numbers this
compares against.

Four real, GPU-backed measurements, all on `development` seeds only:

1. **Retrieval comparison** (`_run_retrieval_comparison`): reuses the
   existing, unmodified `B-C005R1` machinery (`train_repaired_router_and_scorer`,
   `evaluate_repair_cell`, `build_hard_negative_candidates`) for conditions
   R0 (frozen parent) / R1 (cross-entropy reconstruction) / R2 (current
   ranking+argument-scorer repair recipe), restricted to the `development_units`
   scope from `relation_split.json`: target operations SELECT/COUNT/BIND at
   levels L3 (semantically-related) and L4 (confusable-family). SHIFT is
   deliberately excluded here -- its L3/L4 relation-split units are
   `validation`/`sealed_v2`, out of this task's scope.
2. **Adequacy comparison** (`_apply_adequacy_evaluators`): builds ONE real,
   frozen per-example correctness stream for the identity-match SHIFT
   candidate (up to 512 fresh v2 query examples; primitive weights are never
   retrained by any repair condition, so this stream is condition-independent
   and computed once per seed) and replays it through three offline
   evaluators unchanged from their existing implementations: the legacy
   asymmetric sequential verifier (`SequentialAdequacyVerifier`, the same
   class ADR-0080 found had `SEQUENTIAL_RULE_BIAS`), a non-sequential
   fixed-N@{32,64,128} point-estimate baseline (the same class with
   `decision_rule="fixed_threshold"`), and the new frozen finite-look
   contract from `B-C005R3-003` (`functional_metrics_v2.verify_candidate_finite_look`).
   No new runtime verifier is wired; this is a pure offline replay comparison.
3. **SELECT/BIND shuffled-control re-check** (`_select_bind_shuffled_control`):
   re-runs, scoped to only SELECT (competitor BIND) and only the R2 condition,
   the exact z/q shuffled-label probe methodology from
   `representation_stage_probe.py` (same `_fit_binary_probe`/`_probe_accuracy`
   helpers, reused unchanged) that produced D2-002's `SELECT->BIND: UNRESOLVED
   (shuffled-control artifact)` finding, plus an explicit train/eval
   content-overlap audit (the original code relied on distinct generator
   seeds for disjointness but never asserted it) -- a one-time wiring/leakage
   check per the task doc's own scope, never touching router weights based on
   its outcome.
4. **Failure reproduction matrix** (`_build_failure_reproduction_matrix`):
   compares this task's own v2/development numbers against the cited original
   D2 findings (`representation_stage_summary.json`'s regate_sealed/bank_size=128
   focus-cell values, `shift_seed24_adequacy_audit.json`'s closed_loop_exact_match),
   labeling each mechanism `REPRODUCED_ON_V2` or `NOT_REPRODUCED_ON_V2` without
   modifying or deleting the original artifacts.

Frozen throughout: Task Encoder, `query_proj`, all primitive weights. Only
router keys and (for R2) the `ArgumentScorer` are ever retrained, and only by
the existing, unmodified `train_repaired_router_and_scorer` -- this task adds
no new training code.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.evaluation.bank_scaling_benchmark import build_scaled_bank_and_router
from apc.evaluation.functional_metrics_v2 import (
    FiniteLookVerifierContract,
    verify_candidate_finite_look,
)
from apc.evaluation.hard_negative_routing_benchmark import (
    _RELATED_OPERATION,
    DEFAULT_BANK_SIZES,
    HardNegativeBenchmarkConfig,
    HardNegativeCandidate,
    _build_frozen_base_system,
    _execute_selected_candidates,
)
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.relation_split_protocol import assert_sealed_access_permitted
from apc.evaluation.representation_stage_probe import _fit_binary_probe, _probe_accuracy
from apc.evaluation.retrieval_repair_benchmark import (
    CellRepairDiagnostic,
    RetrievalRepairConfig,
    evaluate_repair_cell,
    train_repaired_router_and_scorer,
)
from apc.meta.adequacy_verifier import SequentialAdequacyVerifier, SequentialVerifierConfig
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (10, 11, 12, 13, 14)
RETRIEVAL_TARGET_OPERATIONS: Final[tuple[str, ...]] = ("SELECT", "COUNT", "BIND")
RETRIEVAL_LEVELS: Final[tuple[HardNegativeLevel, ...]] = (
    HardNegativeLevel.L3_SEMANTICALLY_RELATED,
    HardNegativeLevel.L4_CONFUSABLE_FAMILY,
)
RETRIEVAL_CONDITIONS: Final[tuple[str, ...]] = ("R0", "R1", "R2")
ADEQUACY_FIXED_NS: Final[tuple[int, ...]] = (32, 64, 128)

# Cited original findings this task compares against (never rewritten):
# representation_stage_summary.json (regate_sealed/R2_frozen_post_repair, bank_size=128
# focus cell) and shift_seed24_adequacy_audit.json (model_seed=4, i.e. regate_sealed
# seed 24, bank_size=16).
ORIGINAL_D2_REFERENCE: Final[dict[str, dict[str, Any]]] = {
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
    "SELECT:argument_variant": {
        "source": "runs/phase_b_b2_second_diagnostic/l4_argument_breakdown.json:failure_classification",
        "classification": "ARGUMENT_ENCODING_FAILURE",
    },
    "BIND:argument_variant": {
        "source": "runs/phase_b_b2_second_diagnostic/l4_argument_breakdown.json:failure_classification",
        "classification": "ARGUMENT_SCORER_GENERALIZATION_FAILURE",
    },
    "SHIFT_reference_adequacy": {
        "source": "runs/phase_b_b2_second_diagnostic/shift_seed24_adequacy_audit.json (model_seed=4 / regate_sealed seed 24, bank_size=16)",
        "closed_loop_exact_match": 0.921875,
        "classification": "TRUE_PRIMITIVE_INADEQUACY / SEQUENTIAL_RULE_BIAS (premature accept at n=32)",
    },
}


@dataclass(frozen=True)
class PairedBaselineConfig:
    """Explicit configuration for Task B-C005R3-004."""

    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    bank_size: int = 128
    retrieval_target_operations: tuple[str, ...] = RETRIEVAL_TARGET_OPERATIONS
    support_examples: int = 32
    query_examples: int = 64
    router_train_examples: int = 32
    router_steps: int = 250
    router_lr: float = 0.005
    top_k: int = 5
    ranking_margin: float = 3.0
    ranking_beta: float = 1.0
    arg_lambda: float = 2.0
    adequacy_exact_match_threshold: float = 0.95
    shift_operation: str = "SHIFT"
    shift_reference_examples: int = 512
    select_bind_target_operation: str = "SELECT"
    probe_train_examples: int = 64
    probe_eval_examples: int = 64
    probe_steps: int = 300
    probe_lr: float = 0.05
    shuffled_chance_tolerance: float = 0.15
    deterministic_algorithms: bool = True
    device: str = "auto"
    bank_checkpoint_dir: Path = Path("runs/phase_a2_bank_scaling_benchmark")
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_004_paired_baseline")

    def __post_init__(self) -> None:
        if not self.development_seeds:
            raise ValueError("development_seeds must be non-empty")
        if self.bank_size not in DEFAULT_BANK_SIZES:
            raise ValueError(f"bank_size must be one of {DEFAULT_BANK_SIZES}")
        if self.support_examples < 1 or self.query_examples < 1:
            raise ValueError("support_examples and query_examples must be positive")
        if self.shift_reference_examples < max(ADEQUACY_FIXED_NS):
            raise ValueError("shift_reference_examples must cover every fixed-N arm")
        if not 0.0 < self.adequacy_exact_match_threshold <= 1.0:
            raise ValueError("adequacy_exact_match_threshold must be in (0, 1]")
        if not 0.0 <= self.shuffled_chance_tolerance < 0.5:
            raise ValueError("shuffled_chance_tolerance must be in [0.0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["bank_checkpoint_dir"] = str(self.bank_checkpoint_dir)
        data["output_dir"] = str(self.output_dir)
        return data


# ---------------------------------------------------------------------------
# 1. SHIFT reference-adequacy stream + the four offline evaluators.
# ---------------------------------------------------------------------------


_PRETRAINED_CORE_DIR: Final[Path] = Path("runs/phase_a1_shift_compact_structural_probe")


def _pretrained_core_available(seed: int) -> bool:
    """At the time this task (`B-C005R3-004`/ADR-0085) ran, `development` seeds
    (10-14) had no entry under `_PRETRAINED_CORE_DIR` -- only the sealed seeds
    0-4 did -- so `_build_frozen_base_system` silently fell back to a fresh
    random Core init for them (see `representation_stage_probe.py`'s module
    docstring, which already disclosed this for its own probes). That made raw
    execution-correctness (closed-loop EM) meaningless on `development` seeds
    even though routing/ranking comparisons stayed valid; this task must not
    access the sealed seeds that do have a pretrained core
    (`assert_sealed_access_permitted`).

    **Superseded for seeds 10-14 by `B-C005R3-009`** (with the user's explicit,
    pre-confirmed authorization): that task pretrained a genuinely new,
    non-sealed shared-encoder checkpoint for each development seed via the
    existing, unmodified Phase A.1 recipe and cached it at this exact path, so
    `_pretrained_core_available(10..14)` now returns `True`. This function
    itself is unchanged (still a plain file-existence check reflecting
    whatever is really on disk); only the *fact* it reports for those five
    seeds changed. R3-004's own `paired_baseline.json`/ADR-0085 finding is
    left untouched as a historical record of what was true when it ran."""
    return (_PRETRAINED_CORE_DIR / f"seed_{seed}" / "shared_encoder.pt").is_file()


def _shift_reference_stream(
    config: PairedBaselineConfig,
    core: Any,
    base_bank: Any,
    op_to_id_16: dict[str, int],
    seed: int,
) -> list[bool]:
    """One real, condition-independent per-example correctness stream for the
    identity-match SHIFT candidate (primitive weights are never retrained by
    any repair condition, so this is computed once per seed, not per condition)."""
    operation_by_id_16 = {pid: op for op, pid in op_to_id_16.items()}
    pid = op_to_id_16[config.shift_operation]
    examples = generate_benchmark_examples(
        seed * 900_003 + pid, config.shift_reference_examples, operation=config.shift_operation, split="test"
    )
    candidate = HardNegativeCandidate(
        candidate_id=f"primitive:{pid}",
        execute_primitive_id=pid,
        score_key=torch.zeros(1),
        provenance="identity_reference_check_no_argument_override",
    )
    predictions = _execute_selected_candidates(
        core, base_bank, operation_by_id_16, examples, [candidate] * len(examples)
    )
    return [prediction == example.target_tokens for prediction, example in zip(predictions, examples, strict=True)]


def _apply_adequacy_evaluators(
    correct: list[bool], config: PairedBaselineConfig, *, core_pretrained: bool
) -> dict[str, Any]:
    """Replays one frozen correctness stream through legacy_asymmetric, fixed-N,
    and the new B-C005R3-003 finite-look contract. No new verifier code.

    ``core_pretrained=False`` (always true for `development` seeds 10-14) means
    the underlying Core encoder is a fresh random init, so
    ``true_closed_loop_exact_match_full_sample`` reflects encoder noise, not
    SHIFT's real functional adequacy -- the four evaluator verdicts below are
    still a valid mechanical comparison of the decision procedures themselves,
    just not a valid reproduction of D2-005's ~92% finding.
    """
    n = len(correct)

    def support_eval_fn(start: int, end: int) -> int:
        return sum(1 for value in correct[start:end] if value)

    legacy_verifier = SequentialAdequacyVerifier(
        SequentialVerifierConfig(adequacy_threshold=config.adequacy_exact_match_threshold)
    )
    legacy_trace = legacy_verifier.verify_candidate_sequentially(
        candidate_id="SHIFT_identity",
        execute_primitive_id=-1,
        support_eval_fn=support_eval_fn,
        total_available_support=min(legacy_verifier.config.max_support, n),
    )

    fixed_verifier = SequentialAdequacyVerifier(
        SequentialVerifierConfig(
            adequacy_threshold=config.adequacy_exact_match_threshold, decision_rule="fixed_threshold"
        )
    )
    fixed_results: dict[str, Any] = {}
    for look in ADEQUACY_FIXED_NS:
        successes = sum(1 for value in correct[:look] if value)
        decision, interval = fixed_verifier.evaluate_step(successes, look)
        fixed_results[str(look)] = {
            "successes": successes,
            "trials": look,
            "decision": decision.value,
            "point_estimate": interval.point_estimate,
        }

    new_contract = FiniteLookVerifierContract()
    new_trace = verify_candidate_finite_look(
        candidate_id="SHIFT_identity", support_eval_fn=support_eval_fn, contract=new_contract
    )

    return {
        "n_examples": n,
        "core_pretrained": core_pretrained,
        "true_closed_loop_exact_match_full_sample": sum(1 for value in correct if value) / n,
        "legacy_asymmetric": legacy_trace.to_dict(),
        "fixed_n": fixed_results,
        "new_contract_finite_look": new_trace.to_dict(),
    }


# ---------------------------------------------------------------------------
# 2. SELECT/BIND shuffled-control re-check (one relation, one condition).
# ---------------------------------------------------------------------------


def _operation_salt(operation: str) -> int:
    return sum(ord(character) for character in operation) * 31


def _select_bind_shuffled_control(
    config: PairedBaselineConfig, core: Any, router: Any, op_to_id: dict[str, int], seed: int
) -> dict[str, Any]:
    """Re-runs D2-002's exact z/q shuffled-label probe methodology
    (`_fit_binary_probe`/`_probe_accuracy`, reused verbatim), scoped to only
    SELECT (competitor BIND) at the R2 condition, plus an explicit train/eval
    content-overlap audit the original code never asserted directly."""
    target_op = config.select_bind_target_operation
    competitor_op = _RELATED_OPERATION[target_op]
    target_id = op_to_id[target_op]

    def probe_examples(tag: int, operation: str, count: int, split: str) -> list[Any]:
        return list(
            generate_benchmark_examples(
                seed * 20_000_003 + _operation_salt(operation) + tag, count, operation=operation, split=split
            )
        )

    train_target = probe_examples(1, target_op, config.probe_train_examples, "dev")
    train_competitor = probe_examples(2, competitor_op, config.probe_train_examples, "dev")
    eval_target = probe_examples(3, target_op, config.probe_eval_examples, "test")
    eval_competitor = probe_examples(4, competitor_op, config.probe_eval_examples, "test")

    train_contents = {tuple(ex.input_tokens) for ex in train_target} | {
        tuple(ex.input_tokens) for ex in train_competitor
    }
    eval_contents = {tuple(ex.input_tokens) for ex in eval_target} | {
        tuple(ex.input_tokens) for ex in eval_competitor
    }
    overlap = train_contents & eval_contents

    z_train_target = extract_task_representations(core, train_target)
    z_train_competitor = extract_task_representations(core, train_competitor)
    z_eval_target = extract_task_representations(core, eval_target)
    z_eval_competitor = extract_task_representations(core, eval_competitor)
    with torch.no_grad():
        q_train_target = router.query_proj(z_train_target)
        q_train_competitor = router.query_proj(z_train_competitor)
        q_eval_target = router.query_proj(z_eval_target)
        q_eval_competitor = router.query_proj(z_eval_competitor)

    probe_seed = seed * 7_919 + target_id
    z_probe = _fit_binary_probe(
        z_train_target, z_train_competitor, steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed, shuffle_labels=False
    )
    z_probe_shuffled = _fit_binary_probe(
        z_train_target, z_train_competitor, steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed, shuffle_labels=True
    )
    q_probe = _fit_binary_probe(
        q_train_target, q_train_competitor, steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed + 1, shuffle_labels=False
    )
    q_probe_shuffled = _fit_binary_probe(
        q_train_target, q_train_competitor, steps=config.probe_steps, lr=config.probe_lr, seed=probe_seed + 1, shuffle_labels=True
    )

    z_probe_accuracy = _probe_accuracy(z_probe, z_eval_target, z_eval_competitor)
    z_probe_shuffled_accuracy = _probe_accuracy(z_probe_shuffled, z_eval_target, z_eval_competitor)
    q_probe_accuracy = _probe_accuracy(q_probe, q_eval_target, q_eval_competitor)
    q_probe_shuffled_accuracy = _probe_accuracy(q_probe_shuffled, q_eval_target, q_eval_competitor)

    tol = config.shuffled_chance_tolerance
    shuffled_at_chance = abs(z_probe_shuffled_accuracy - 0.5) <= tol and abs(q_probe_shuffled_accuracy - 0.5) <= tol

    return {
        "seed": seed,
        "target_operation": target_op,
        "competitor_operation": competitor_op,
        "z_probe_accuracy": z_probe_accuracy,
        "z_probe_shuffled_accuracy": z_probe_shuffled_accuracy,
        "q_probe_accuracy": q_probe_accuracy,
        "q_probe_shuffled_accuracy": q_probe_shuffled_accuracy,
        "shuffled_chance_tolerance": tol,
        "shuffled_controls_at_chance": shuffled_at_chance,
        "train_eval_content_overlap_count": len(overlap),
        "probe_leakage_detected": len(overlap) > 0,
    }


# ---------------------------------------------------------------------------
# 3. Orchestration: one pass per development seed.
# ---------------------------------------------------------------------------


def run_paired_baseline_repair(config: PairedBaselineConfig) -> dict[str, Any]:
    """Executes B-C005R3-004: retrieval comparison (R0/R1/R2), SHIFT adequacy
    comparison (4 offline evaluators on one frozen stream), and the SELECT/BIND
    shuffled-control re-check, all on `development` seeds only."""
    start = time.perf_counter()
    assert_sealed_access_permitted(config.development_seeds, purpose="B-C005R3-004_paired_baseline")
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    retrieval_cells: list[dict[str, Any]] = []
    adequacy_by_seed: dict[str, Any] = {}
    select_bind_by_seed: dict[str, Any] = {}

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

        # SHIFT reference-adequacy stream: condition-independent (primitive
        # weights are never retrained), so computed once per seed here.
        shift_correct = _shift_reference_stream(config, core, base_bank, op_to_id, seed)
        adequacy_by_seed[str(seed)] = _apply_adequacy_evaluators(
            shift_correct, config, core_pretrained=_pretrained_core_available(seed)
        )

        bank, router, candidate_ids, _semantic_ids, distractor_ids = build_scaled_bank_and_router(
            core, base_bank, base_router, op_to_id, config.bank_size, seed=seed
        )
        operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
        operation_by_id.update({pid: "SWAP_ENDS" for pid in distractor_ids})

        dev_train_by_op = {
            op: generate_benchmark_examples(
                seed * 30_000 + config.bank_size + op_to_id[op], config.router_train_examples, operation=op, split="dev"
            )
            for op in op_to_id
        }
        repair_config = RetrievalRepairConfig(
            seeds=(seed,),
            bank_sizes=(config.bank_size,),
            target_operations=config.retrieval_target_operations,
            support_examples=config.support_examples,
            query_examples=config.query_examples,
            router_train_examples=config.router_train_examples,
            router_steps=config.router_steps,
            router_lr=config.router_lr,
            top_k=config.top_k,
            ranking_margin=config.ranking_margin,
            ranking_beta=config.ranking_beta,
            arg_lambda=config.arg_lambda,
            adequacy_exact_match_threshold=config.adequacy_exact_match_threshold,
            device=config.device,
            bank_checkpoint_dir=config.bank_checkpoint_dir,
        )

        r2_router: Any = None
        for condition in RETRIEVAL_CONDITIONS:
            active_router, arg_scorer = train_repaired_router_and_scorer(
                core=core,
                router=router,
                candidate_ids=candidate_ids,
                operation_by_id=operation_by_id,
                dev_train_examples_by_op=dev_train_by_op,
                config=repair_config,
                condition=condition,
                device=core.device,
            )
            if condition == "R2":
                r2_router = active_router

            for target_op in config.retrieval_target_operations:
                target_id = op_to_id[target_op]
                support_examples = generate_benchmark_examples(
                    seed * 40_000 + config.bank_size + target_id, config.support_examples, operation=target_op, split="dev"
                )
                query_examples = generate_benchmark_examples(
                    seed * 50_000 + config.bank_size + target_id + 100,
                    config.query_examples,
                    operation=target_op,
                    split="dev",
                )
                for level in RETRIEVAL_LEVELS:
                    diagnostic: CellRepairDiagnostic = evaluate_repair_cell(
                        core=core,
                        bank=bank,
                        router=active_router,
                        argument_scorer=arg_scorer,
                        candidate_ids=candidate_ids,
                        operation_by_id=operation_by_id,
                        target_operation=target_op,
                        target_id=target_id,
                        level=level,
                        support_examples=support_examples,
                        query_examples=query_examples,
                        seed=seed,
                        top_k=config.top_k,
                        adequacy_threshold=config.adequacy_exact_match_threshold,
                        arg_lambda=config.arg_lambda if condition == "R2" else 0.0,
                        condition=condition,
                        bank_size=config.bank_size,
                    )
                    retrieval_cells.append(diagnostic.to_dict())

        assert r2_router is not None
        select_bind_by_seed[str(seed)] = _select_bind_shuffled_control(config, core, r2_router, op_to_id, seed)

    paired_baseline = _build_paired_baseline(retrieval_cells, adequacy_by_seed, select_bind_by_seed, config)
    comparison_manifest = _build_comparison_manifest(config)
    failure_reproduction_matrix = _build_failure_reproduction_matrix(paired_baseline, select_bind_by_seed)
    select_bind_control_status = _build_select_bind_control_status(select_bind_by_seed, config)

    protocol = {
        "task": "B-C005R3-004",
        "gate": None,
        "result": "COMPARISON_ESTABLISHED",
        "requires_performance_pass": False,
        "comparison_validity": comparison_manifest["comparison_validity"],
        "needs_scope_review": failure_reproduction_matrix["any_needs_scope_review"],
        "downstream_note": (
            "This task does not gate B-C005R3-006/007/008/009; it establishes their "
            "starting reference on the v2 fixture. A NOT_REPRODUCED_ON_V2 mechanism "
            "flags its corresponding downstream task NEEDS_SCOPE_REVIEW rather than "
            "auto-authorizing training."
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")
        (output_dir / "paired_baseline.json").write_text(json.dumps(paired_baseline, indent=2), encoding="utf-8")
        (output_dir / "comparison_manifest.json").write_text(json.dumps(comparison_manifest, indent=2), encoding="utf-8")
        (output_dir / "failure_reproduction_matrix.json").write_text(
            json.dumps(failure_reproduction_matrix, indent=2), encoding="utf-8"
        )
        (output_dir / "select_bind_control_status.json").write_text(
            json.dumps(select_bind_control_status, indent=2), encoding="utf-8"
        )
        (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "paired_baseline": paired_baseline,
        "comparison_manifest": comparison_manifest,
        "failure_reproduction_matrix": failure_reproduction_matrix,
        "select_bind_control_status": select_bind_control_status,
        "protocol": protocol,
    }


# ---------------------------------------------------------------------------
# 4. Aggregation into the four required artifacts.
# ---------------------------------------------------------------------------


def _relation_label(target_operation: str, level: HardNegativeLevel) -> str:
    if level == HardNegativeLevel.L3_SEMANTICALLY_RELATED:
        return f"{target_operation}->{_RELATED_OPERATION[target_operation]}"
    return f"{target_operation}:argument_variant"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _build_paired_baseline(
    retrieval_cells: list[dict[str, Any]],
    adequacy_by_seed: dict[str, Any],
    select_bind_by_seed: dict[str, Any],
    config: PairedBaselineConfig,
) -> dict[str, Any]:
    by_relation_condition: dict[str, dict[str, Any]] = {}
    for cell in retrieval_cells:
        level = HardNegativeLevel(cell["level"])
        relation = _relation_label(cell["target_operation"], level)
        bucket = by_relation_condition.setdefault(relation, {})
        cond_cells = bucket.setdefault(cell["condition"], [])
        cond_cells.append(cell)

    relation_summary: dict[str, Any] = {}
    for relation, by_condition in by_relation_condition.items():
        relation_summary[relation] = {
            condition: {
                "n_seeds": len(cells),
                "primitive_call_top1_mean": _mean([c["primitive_call_top1"] for c in cells]),
                "primitive_call_topk_mean": _mean([c["primitive_call_topk"] for c in cells]),
                "physical_primitive_top1_mean": _mean([c["physical_primitive_top1"] for c in cells]),
                "argument_accuracy_mean": _mean([c["argument_accuracy"] for c in cells]),
                "candidate_rank_mean": _mean([c["candidate_rank"] for c in cells]),
                "score_margin_mean": _mean([c["score_margin"] for c in cells]),
                "seeds": [c["seed"] for c in cells],
            }
            for condition, cells in by_condition.items()
        }

    return {
        "task": "B-C005R3-004",
        "development_seeds": list(config.development_seeds),
        "bank_size": config.bank_size,
        "retrieval": {
            "per_relation_by_condition": relation_summary,
            "raw_cells": retrieval_cells,
        },
        "shift_reference_adequacy": {
            "by_seed": adequacy_by_seed,
        },
        "select_bind_shuffled_control": {
            "by_seed": select_bind_by_seed,
        },
    }


def _build_comparison_manifest(config: PairedBaselineConfig) -> dict[str, Any]:
    return {
        "task": "B-C005R3-004",
        "fixed_across_conditions": [
            "generator_version (v2_sha256_indexed, ADR-0082) and every example-generation seed formula "
            "(support/query/dev-train) depend only on (seed, bank_size, target_id) -- never on `condition` "
            "-- so R0/R1/R2 see byte-identical support/query/training example sets per (seed, target_operation).",
            "candidate identity graph: `build_hard_negative_candidates` is called with the same `seed` and "
            "`target_id` per cell across conditions, so which primitive plays target/competitor/distractor "
            "is fixed; only the router's key VALUES for those same primitives differ by condition.",
            "top_k, ranking_margin, ranking_beta, arg_lambda, adequacy_exact_match_threshold are shared "
            "RetrievalRepairConfig fields, identical across R0/R1/R2 within one seed.",
        ],
        "explicitly_varied_across_conditions": [
            "router key parameters (R0: frozen parent; R1: cross-entropy reconstruction; R2: margin-ranking "
            "+ ArgumentScorer, the current repair recipe) -- this is the deliberate per-condition change "
            "being measured, per `train_repaired_router_and_scorer`.",
        ],
        "scale_disclosure": {
            "support_examples": config.support_examples,
            "query_examples": config.query_examples,
            "note": (
                "Smaller than the primary experiment design's proposed 256 query examples/episode "
                "(EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md S2); this task is an explicitly bounded "
                "comparison-validity check, not the G4/G5 statistical-power run."
            ),
        },
        "scope_exclusions": [
            "SHIFT is excluded from the retrieval-comparison matrix: its L3 (SHIFT-CYCLE_FOUR) and L4 "
            "(SHIFT:argument_variant) relation-split units are `validation`/`sealed_v2`, not `development` "
            "(relation_split.json), and are out of this task's scope.",
            "validation seeds (15-19) and sealed_v2 seeds (30-34) are never accessed by this task "
            "(assert_sealed_access_permitted enforced).",
        ],
        "untrained_core_disclosure": (
            "Development seeds 10-14 have no entry under "
            "runs/phase_a1_shift_compact_structural_probe/ (only sealed seeds 0-4 do), so "
            "`_build_frozen_base_system` falls back to a fresh random Core init for every seed this "
            "task uses -- already disclosed for probe geometry by `representation_stage_probe.py`'s "
            "own module docstring. Routing/ranking/argument-scorer comparisons (primitive_call_top1, "
            "argument_accuracy, candidate_rank, score_margin) stay valid under a random-but-fixed core "
            "-- this is the same methodology B-C005R1's own acceptance gate already uses on these same "
            "seeds. Raw execution-correctness (closed_loop_exact_match) does NOT stay valid under a "
            "random core; see failure_reproduction_matrix.json's SHIFT_reference_adequacy entry, which "
            "is reported as UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE for exactly this reason "
            "rather than a misleading REPRODUCED/NOT_REPRODUCED binary."
        ),
        "exact_replay_vs_v2_reconstruction": (
            "Every number in this task is a fresh V2_RECONSTRUCTION on development seeds under the v2 "
            "generator; none is an EXACT_REPLAY of the original regate_sealed/bank_size=128 D2 run (that "
            "original input is not independently persisted, per ADR-0082's artifact inventory). Comparisons "
            "against ORIGINAL_D2_REFERENCE are qualitative reproduction checks, not exact-value replays."
        ),
        "comparison_validity": "VALID",
    }


def _build_failure_reproduction_matrix(
    paired_baseline: dict[str, Any], select_bind_by_seed: dict[str, Any]
) -> dict[str, Any]:
    relation_summary = paired_baseline["retrieval"]["per_relation_by_condition"]
    entries: dict[str, Any] = {}

    for relation, original in ORIGINAL_D2_REFERENCE.items():
        if relation == "SHIFT_reference_adequacy":
            continue
        r2 = relation_summary.get(relation, {}).get("R2")
        if r2 is None:
            entries[relation] = {
                "status": "NOT_RUN",
                "original": original,
                "downstream_task": _downstream_task_for(relation),
                "needs_scope_review": True,
                "reason": "no R2 cells were evaluated for this relation on development seeds",
            }
            continue
        measured_top1 = r2["primitive_call_top1_mean"]
        reproduced = measured_top1 is not None and measured_top1 < 0.95
        entries[relation] = {
            "status": "REPRODUCED_ON_V2" if reproduced else "NOT_REPRODUCED_ON_V2",
            "original": original,
            "measured_v2_development_R2_primitive_call_top1_mean": measured_top1,
            "downstream_task": _downstream_task_for(relation),
            "needs_scope_review": not reproduced,
        }

    shift_seed_results = list(paired_baseline["shift_reference_adequacy"]["by_seed"].values())
    shift_entries = [seed_result["true_closed_loop_exact_match_full_sample"] for seed_result in shift_seed_results]
    shift_mean = _mean(shift_entries)
    any_pretrained_core = any(seed_result["core_pretrained"] for seed_result in shift_seed_results)
    if not any_pretrained_core:
        # Development seeds (10-14) have no pretrained shared-encoder checkpoint
        # (only sealed seeds 0-4 do, per `_pretrained_core_available`), so the
        # Core is a fresh random init and `shift_mean` reflects encoder noise,
        # not SHIFT's real functional adequacy. This is a structural scope
        # limit, not a "the failure went away" finding -- do not report a
        # REPRODUCED/NOT_REPRODUCED binary here.
        entries["SHIFT_reference_adequacy"] = {
            "status": "UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE",
            "original": ORIGINAL_D2_REFERENCE["SHIFT_reference_adequacy"],
            "measured_v2_development_mean_closed_loop_exact_match": shift_mean,
            "downstream_task": "B-C005R3-009",
            "needs_scope_review": True,
            "reason": (
                "development seeds 10-14 have no entry under "
                "runs/phase_a1_shift_compact_structural_probe/, so "
                "_build_frozen_base_system falls back to a fresh random Core "
                "init; only sealed seeds 0-4 have a pretrained checkpoint, and "
                "this task is not authorized to access them "
                "(assert_sealed_access_permitted). The four-evaluator "
                "mechanical comparison (legacy_asymmetric/fixed_n/new_contract) "
                "in paired_baseline.json is still valid; the *closed_loop_exact_match* "
                "value it was applied to is not a reproduction of D2-005's ~92% finding. "
                "Validating SHIFT reference adequacy for real requires either sealed "
                "access via the R3-011/R3-012 pathway, or a separately-authorized "
                "change of validation strategy -- not something this task can resolve."
            ),
        }
    else:
        shift_reproduced = shift_mean is not None and shift_mean < 0.95
        entries["SHIFT_reference_adequacy"] = {
            "status": "REPRODUCED_ON_V2" if shift_reproduced else "NOT_REPRODUCED_ON_V2",
            "original": ORIGINAL_D2_REFERENCE["SHIFT_reference_adequacy"],
            "measured_v2_development_mean_closed_loop_exact_match": shift_mean,
            "downstream_task": "B-C005R3-009",
            "needs_scope_review": not shift_reproduced,
        }

    select_bind_at_chance = [seed_result["shuffled_controls_at_chance"] for seed_result in select_bind_by_seed.values()]
    select_bind_all_at_chance = bool(select_bind_at_chance) and all(select_bind_at_chance)
    entries["SELECT->BIND_shuffled_control"] = {
        "status": "NOT_REPRODUCED_ON_V2" if select_bind_all_at_chance else "REPRODUCED_ON_V2",
        "original": {
            "source": "runs/phase_b_b2_second_diagnostic/representation_stage_summary.json (SELECT->BIND: UNRESOLVED, shuffled-control artifact)",
            "classification": "UNRESOLVED",
        },
        "downstream_task": "N/A (measurement-bug scope only; see select_bind_control_status.json)",
        "needs_scope_review": not select_bind_all_at_chance,
        "note": "UNRESOLVED stays UNRESOLVED if the shuffled control still fails the chance check on v2; this is not a routable repair mechanism.",
    }

    any_needs_scope_review = any(entry.get("needs_scope_review") for entry in entries.values())
    return {
        "task": "B-C005R3-004",
        "entries": entries,
        "any_needs_scope_review": any_needs_scope_review,
    }


def _downstream_task_for(relation: str) -> str:
    if relation in ("COUNT->BIND", "BIND->COUNT"):
        return "B-C005R3-006"
    if relation == "SELECT:argument_variant":
        return "B-C005R3-007"
    if relation == "BIND:argument_variant":
        return "B-C005R3-008"
    return "UNKNOWN"


def _build_select_bind_control_status(select_bind_by_seed: dict[str, Any], config: PairedBaselineConfig) -> dict[str, Any]:
    per_seed = list(select_bind_by_seed.values())
    any_leakage = any(seed_result["probe_leakage_detected"] for seed_result in per_seed)
    all_at_chance = bool(per_seed) and all(seed_result["shuffled_controls_at_chance"] for seed_result in per_seed)

    if any_leakage:
        status = "MEASUREMENT_BUG_FOUND_PROBE_LEAKAGE"
    elif all_at_chance:
        status = "RESOLVED_ON_V2_SHUFFLED_CONTROLS_AT_CHANCE"
    else:
        status = "UNRESOLVED"

    return {
        "task": "B-C005R3-004",
        "checked_once": True,
        "router_weights_changed": False,
        "wiring_audit": {
            "label_split_metric": (
                "train/eval probe examples are generated from four disjoint (operation, tag) seed "
                "combinations (tag 1/2 = train target/competitor, tag 3/4 = eval target/competitor); "
                "shuffling (`shuffle_labels=True`) permutes only the training-label vector before fitting, "
                "never the evaluation labels used to score the fitted probe -- verified by reading "
                "`_fit_binary_probe` (representation_stage_probe.py), reused unchanged by this task."
            ),
            "probe_leakage_content_overlap_check": "explicit input_tokens set-intersection between train and eval probe examples, per seed",
        },
        "status": status,
        "per_seed": select_bind_by_seed,
        "shuffled_chance_tolerance": config.shuffled_chance_tolerance,
    }
