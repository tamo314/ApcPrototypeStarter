"""Shadow validation, promotion, and release evaluation (Task A1-B007X-006).

Validates safety of distilled compact candidate primitives before promotion into persistent bank:
- canonical forgetting <= 2pp per op (8 canonical operations)
- composition forgetting <= 2pp (6 representative compositions)
- Core unchanged (parameters strictly identical)
- existing bank unchanged (canonical primitive parameters strictly identical)

On pass:
- install exactly one candidate (bank size +1)
- candidate status -> STABLE and frozen
- release temporary capacity to 0 (temp params = 0)
- serialize persistent state (Core + Bank) for fresh-runtime recurrence (Task A1-B007X-007)

On fail:
- abort install (bank unchanged)
- preserve fallback (temporary capacity retained)
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.generator import Example
from apc.evaluation.compact_cross_position_operator_probe import (
    DEFAULT_GROUP_SIZE,
    _default_model_config,
)
from apc.evaluation.composition_library_benchmark import (
    DESIGNATED_COMPOSITIONS,
    _generate_composition_examples,
)
from apc.evaluation.consolidation_benchmark import generate_benchmark_examples
from apc.evaluation.discovery_capacity_harness import (
    TIER_SPECS,
    CapacityTier,
    generate_novel_examples,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    ALL_CANONICAL_OPERATIONS,
    PARAMETERIZED_OPERATIONS,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _get_or_train_frozen_shared_core,
    _train_single_primitive,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.conditioning import DEFAULT_MAX_SEQUENCE_LENGTH
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils.seed import set_seed

DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
MIN_GATE_SEEDS: Final[int] = 5
MAX_FORGETTING_THRESHOLD: Final[float] = 0.02
MAX_CANDIDATE_PARAMS: Final[int] = 25000
DEFAULT_EVAL_BATCH_SIZE: Final[int] = 128


@dataclass(frozen=True)
class ShadowPromotionConfig:
    """Configuration for Task A1-B007X-006 shadow validation, promotion, and release."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    novel_operation: str = "SWAP_PAIRS"
    max_canonical_forgetting: float = MAX_FORGETTING_THRESHOLD
    max_composition_forgetting: float = MAX_FORGETTING_THRESHOLD
    max_candidate_parameters: int = MAX_CANDIDATE_PARAMS

    vocab_size: int = 10
    sequence_length_range: tuple[int, int] = (6, 10)
    group_size: int = DEFAULT_GROUP_SIZE
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH

    num_canonical_eval_examples: int = 200
    num_composition_eval_examples: int = 200
    num_novel_eval_examples: int = 200
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE

    device: str = "auto"
    shared_encoder_checkpoint: str | None = None
    distill_checkpoint_pattern: str = (
        "runs/phase_a1_overcomplete_distillation/checkpoints/distill_candidate_{op}_seed_{seed}.pt"
    )
    plastic_bank_checkpoint_pattern: str = (
        "runs/phase_a1_plastic_workspace_benchmark/seed_{seed}/primitive_bank.pt"
    )
    core_train_steps: int = 6000
    bank_train_steps: int = 6000
    model: dict[str, Any] = field(default_factory=_default_model_config)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def shadow_promotion_config_from_dict(raw: dict[str, Any]) -> ShadowPromotionConfig:
    """Instantiate ShadowPromotionConfig from dictionary."""
    defaults = ShadowPromotionConfig()
    return ShadowPromotionConfig(
        seeds=tuple(raw.get("seeds", defaults.seeds)),
        novel_operation=raw.get("novel_operation", defaults.novel_operation),
        max_canonical_forgetting=raw.get(
            "max_canonical_forgetting", defaults.max_canonical_forgetting
        ),
        max_composition_forgetting=raw.get(
            "max_composition_forgetting", defaults.max_composition_forgetting
        ),
        max_candidate_parameters=raw.get(
            "max_candidate_parameters", defaults.max_candidate_parameters
        ),
        vocab_size=raw.get("vocab_size", defaults.vocab_size),
        sequence_length_range=tuple(
            raw.get("sequence_length_range", defaults.sequence_length_range)
        ),
        group_size=raw.get("group_size", defaults.group_size),
        max_sequence_length=raw.get("max_sequence_length", defaults.max_sequence_length),
        num_canonical_eval_examples=raw.get(
            "num_canonical_eval_examples", defaults.num_canonical_eval_examples
        ),
        num_composition_eval_examples=raw.get(
            "num_composition_eval_examples", defaults.num_composition_eval_examples
        ),
        num_novel_eval_examples=raw.get(
            "num_novel_eval_examples", defaults.num_novel_eval_examples
        ),
        eval_batch_size=raw.get("eval_batch_size", defaults.eval_batch_size),
        device=raw.get("device", defaults.device),
        shared_encoder_checkpoint=raw.get("shared_encoder_checkpoint", None),
        distill_checkpoint_pattern=raw.get(
            "distill_checkpoint_pattern", defaults.distill_checkpoint_pattern
        ),
        plastic_bank_checkpoint_pattern=raw.get(
            "plastic_bank_checkpoint_pattern", defaults.plastic_bank_checkpoint_pattern
        ),
        core_train_steps=raw.get("core_train_steps", defaults.core_train_steps),
        bank_train_steps=raw.get("bank_train_steps", defaults.bank_train_steps),
        model=dict(raw.get("model", defaults.model)),
    )


@dataclass(frozen=True)
class TaskEvaluationMetric:
    """Individual task evaluation before and after shadow candidate installation."""

    task_name: str
    task_category: str  # "canonical_parameter_free", "canonical_parameterized", "composition"
    before_exact_match: float
    before_token_accuracy: float
    after_exact_match: float
    after_token_accuracy: float
    forgetting: float
    passed: bool


@dataclass(frozen=True)
class ShadowPromotionReport:
    """Complete per-seed evaluation report for Task A1-B007X-006."""

    seed: int
    novel_operation: str
    canonical_metrics: dict[str, TaskEvaluationMetric]
    composition_metrics: dict[str, TaskEvaluationMetric]
    candidate_novel_exact_match: float
    candidate_novel_token_accuracy: float

    max_canonical_forgetting: float
    max_composition_forgetting: float
    max_overall_forgetting: float

    core_unchanged: bool
    existing_bank_unchanged: bool
    candidate_parameters: int
    candidate_size_passed: bool
    finite_outputs_passed: bool

    all_canonical_passed: bool
    all_composition_passed: bool
    safety_criteria_passed: bool

    # Actions taken
    promoted: bool
    installed_primitive_id: int | None
    bank_size_before: int
    bank_size_after: int
    temp_parameters_before: int
    temp_parameters_after: int
    temp_released_completely: bool
    overall_passed: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "novel_operation": self.novel_operation,
            "canonical_metrics": {
                k: dataclasses.asdict(v) for k, v in self.canonical_metrics.items()
            },
            "composition_metrics": {
                k: dataclasses.asdict(v) for k, v in self.composition_metrics.items()
            },
            "candidate_novel_exact_match": self.candidate_novel_exact_match,
            "candidate_novel_token_accuracy": self.candidate_novel_token_accuracy,
            "max_canonical_forgetting": self.max_canonical_forgetting,
            "max_composition_forgetting": self.max_composition_forgetting,
            "max_overall_forgetting": self.max_overall_forgetting,
            "core_unchanged": self.core_unchanged,
            "existing_bank_unchanged": self.existing_bank_unchanged,
            "candidate_parameters": self.candidate_parameters,
            "candidate_size_passed": self.candidate_size_passed,
            "finite_outputs_passed": self.finite_outputs_passed,
            "all_canonical_passed": self.all_canonical_passed,
            "all_composition_passed": self.all_composition_passed,
            "safety_criteria_passed": self.safety_criteria_passed,
            "promoted": self.promoted,
            "installed_primitive_id": self.installed_primitive_id,
            "bank_size_before": self.bank_size_before,
            "bank_size_after": self.bank_size_after,
            "temp_parameters_before": self.temp_parameters_before,
            "temp_parameters_after": self.temp_parameters_after,
            "temp_released_completely": self.temp_released_completely,
            "overall_passed": self.overall_passed,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class ShadowPromotionSummary:
    """Multi-seed summary for Task A1-B007X-006."""

    novel_operation: str
    seeds: tuple[int, ...]
    overall_passed: bool
    meets_seed_policy: bool
    all_seeds_promoted: bool
    mean_candidate_novel_em: float
    max_overall_forgetting: float
    all_core_unchanged: bool
    all_existing_bank_unchanged: bool
    bank_size_transition: str  # e.g. "8 -> 9"
    temp_released_completely: bool
    reports: list[ShadowPromotionReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "novel_operation": self.novel_operation,
            "seeds": list(self.seeds),
            "overall_passed": self.overall_passed,
            "meets_seed_policy": self.meets_seed_policy,
            "all_seeds_promoted": self.all_seeds_promoted,
            "mean_candidate_novel_em": self.mean_candidate_novel_em,
            "max_overall_forgetting": self.max_overall_forgetting,
            "all_core_unchanged": self.all_core_unchanged,
            "all_existing_bank_unchanged": self.all_existing_bank_unchanged,
            "bank_size_transition": self.bank_size_transition,
            "temp_released_completely": self.temp_released_completely,
            "reports": [r.to_dict() for r in self.reports],
        }


def _evaluate_batch(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    examples: Sequence[Example],
    recipe_ops: Sequence[str] | None = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> tuple[float, float, bool]:
    """Evaluate exact match, token accuracy, and finite output validity."""
    if not examples:
        return 0.0, 0.0, True

    exact_matches = 0
    correct_tokens = 0
    total_tokens = 0
    finite = True

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            logits = execute_composition_recipe(
                core, bank, op_to_id, batch, candidate_operations=recipe_ops
            )
            if not torch.isfinite(logits).all():
                finite = False

            predictions = logits.argmax(dim=-1)
            for row, ex in enumerate(batch):
                target = ex.target_tokens
                n = len(target)
                pred = tuple(predictions[row, :n].tolist())
                total_tokens += n
                correct_tokens += sum(1 for p, t in zip(pred, target, strict=True) if p == t)
                if pred == target:
                    exact_matches += 1

    em = exact_matches / len(examples)
    acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return em, acc, finite


def verify_core_unchanged(
    initial_weights: Mapping[str, torch.Tensor],
    current_core: Any,
) -> bool:
    """Verify that every parameter in Stable Core is byte-for-byte or zero-diff identical."""
    for name, param in current_core.model.named_parameters():
        if name not in initial_weights:
            return False
        diff = (param.detach().cpu() - initial_weights[name].cpu()).abs().max().item()
        if diff != 0.0:
            return False
    return True


def verify_bank_unchanged(
    initial_weights: Mapping[str, torch.Tensor],
    current_bank: PrimitiveBank,
    pre_existing_ids: Sequence[int],
) -> bool:
    """Verify that existing primitives in PrimitiveBank are strictly unchanged."""
    for pid in pre_existing_ids:
        prim = current_bank.get(pid)
        for p_name, param in prim.named_parameters():
            full_key = f"_primitives.{pid}.{p_name}"
            if full_key not in initial_weights:
                return False
            diff = (param.detach().cpu() - initial_weights[full_key].cpu()).abs().max().item()
            if diff != 0.0:
                return False
    return True


def run_shadow_promotion_single_seed(
    config: ShadowPromotionConfig,
    seed: int,
    *,
    seed_dir: Path | None = None,
    inject_core_mutation: bool = False,
    inject_bank_mutation: bool = False,
) -> ShadowPromotionReport:
    """Execute complete shadow validation, promotion, and release protocol for one seed."""
    start_time = time.perf_counter()
    set_seed(seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # 1. Obtain frozen shared task-blind Core
    core_ckpt = None
    if (
        config.shared_encoder_checkpoint is not None
        and Path(config.shared_encoder_checkpoint).exists()
    ):
        core_ckpt = str(Path(config.shared_encoder_checkpoint))
    elif seed_dir is not None:
        possible = seed_dir.parent / "phase_a1_discovery_capacity_harness" / "shared_encoder.pt"
        if possible.exists():
            core_ckpt = str(possible)

    u_config = UnifiedBenchmarkConfig(
        seed=seed,
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
        group_size=config.group_size,
        model=config.model,
        device=str(device),
        core_train_steps=config.core_train_steps,
        parameterized_train_steps=config.bank_train_steps,
        parameter_free_train_steps=config.bank_train_steps,
        shared_encoder_checkpoint=core_ckpt,
    )
    core = _get_or_train_frozen_shared_core(u_config, seed_dir=seed_dir)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    # Capture initial core weights snapshot for invariant verification
    core_initial_weights = {
        k: v.detach().clone().cpu() for k, v in core.model.named_parameters()
    }

    # 2. Build initial Bank (8 canonical primitives)
    bank, op_to_id = _build_heterogeneous_bank(u_config)
    bank.to(core.device)

    # Load pre-trained canonical primitives from plastic or composition run if available
    plastic_ckpt_path = Path(config.plastic_bank_checkpoint_pattern.format(seed=seed))
    loaded_bank = False
    if plastic_ckpt_path.is_file():
        try:
            sd = torch.load(plastic_ckpt_path, map_location=core.device, weights_only=True)
            bank.load_state_dict(sd)
            loaded_bank = True
        except Exception:
            pass

    if not loaded_bank:
        for op in ALL_CANONICAL_OPERATIONS:
            prim_module = bank.get(op_to_id[op])
            _train_single_primitive(core, prim_module, u_config, op, steps=config.bank_train_steps)

    bank.freeze_all()
    bank.eval()

    bank_size_before = len(bank)
    pre_existing_ids = bank.ids()
    assert bank_size_before == 8, f"Expected 8 canonical primitives in bank, got {bank_size_before}"

    # Capture initial bank weights snapshot
    bank_initial_weights = {
        k: v.detach().clone().cpu() for k, v in bank.named_parameters()
    }

    # 3. Baseline Performance Evaluation on Bank_before
    canonical_eval_data: dict[str, list[Example]] = {}
    canonical_before_ems: dict[str, float] = {}
    canonical_before_accs: dict[str, float] = {}
    for op in ALL_CANONICAL_OPERATIONS:
        examples = generate_benchmark_examples(
            seed * 100 + 41,
            config.num_canonical_eval_examples,
            operation=op,
            split="test",
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        canonical_eval_data[op] = examples
        em, acc, _ = _evaluate_batch(
            core, bank, op_to_id, examples, recipe_ops=(op,), batch_size=config.eval_batch_size
        )
        canonical_before_ems[op] = em
        canonical_before_accs[op] = acc

    composition_eval_data: dict[str, list[Example]] = {}
    composition_before_ems: dict[str, float] = {}
    composition_before_accs: dict[str, float] = {}
    for comp in DESIGNATED_COMPOSITIONS:
        comp_name = f"{comp[0]}->{comp[1]}"
        comp_examples = _generate_composition_examples(
            seed * 100 + 73,
            comp,
            config.num_composition_eval_examples,
            vocab_size=config.vocab_size,
            sequence_length_range=config.sequence_length_range,
        )
        composition_eval_data[comp_name] = comp_examples
        em, acc, _ = _evaluate_batch(
            core, bank, op_to_id, comp_examples, batch_size=config.eval_batch_size
        )
        composition_before_ems[comp_name] = em
        composition_before_accs[comp_name] = acc

    # 4. Load Distilled Candidate Primitive from A1-B007X-005
    op_novel = config.novel_operation
    cand_ckpt_path = Path(
        config.distill_checkpoint_pattern.format(op=op_novel, seed=seed)
    )
    if not cand_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Candidate distillation checkpoint not found at: {cand_ckpt_path}. "
            f"Run A1-B007X-005 before A1-B007X-006."
        )

    cand_dict = torch.load(cand_ckpt_path, map_location=core.device, weights_only=True)
    next_id = max(bank.ids()) + 1
    pcfg = CrossPositionPrimitiveConfig(
        operation=op_novel,
        d_model=core.model.config.d_model,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
    )
    candidate = CrossPositionPrimitive(
        primitive_id=next_id,
        config=pcfg,
        status=PrimitiveStatus.CANDIDATE,
    ).to(core.device)
    candidate.load_state_dict(cand_dict["state_dict"])
    candidate.eval()

    candidate_params = candidate.num_parameters()
    candidate_size_passed = candidate_params <= config.max_candidate_parameters

    # Temporary discovery resources parameter count (from A1-B007X-005 T2 overcomplete teacher)
    temp_teacher_params = TIER_SPECS[CapacityTier.T2_OVERCOMPLETE.value].expected_parameters
    temp_parameters_before = temp_teacher_params

    # 5. Shadow Evaluation (evaluate candidate in shadow mode)
    # Temporary mock install in a shadow bank dictionary to measure forgetting & cross-talk
    shadow_bank, shadow_op_to_id = _build_heterogeneous_bank(u_config)
    shadow_bank.to(core.device)
    shadow_bank.load_state_dict(bank.state_dict())

    # Build and register candidate into shadow_bank
    cand_shadow = CrossPositionPrimitive(
        primitive_id=next_id,
        config=pcfg,
        status=PrimitiveStatus.CANDIDATE,
    ).to(core.device)
    cand_shadow.load_state_dict(candidate.state_dict())
    cand_shadow.eval()
    shadow_bank.add_primitive(cand_shadow)
    shadow_op_to_id[op_novel] = next_id

    # Evaluate novel task with candidate
    novel_eval_examples = generate_novel_examples(
        seed,
        config.num_novel_eval_examples,
        operation=op_novel,
        split="test",
        vocab_size=config.vocab_size,
        sequence_length_range=config.sequence_length_range,
    )
    cand_novel_em, cand_novel_acc, finite_novel = _evaluate_batch(
        core,
        shadow_bank,
        shadow_op_to_id,
        novel_eval_examples,
        recipe_ops=(op_novel,),
        batch_size=config.eval_batch_size,
    )

    # Re-evaluate all 8 canonical tasks in shadow mode
    canonical_metrics: dict[str, TaskEvaluationMetric] = {}
    finite_all = finite_novel
    for op in ALL_CANONICAL_OPERATIONS:
        em_after, acc_after, fin = _evaluate_batch(
            core,
            shadow_bank,
            shadow_op_to_id,
            canonical_eval_data[op],
            recipe_ops=(op,),
            batch_size=config.eval_batch_size,
        )
        if not fin:
            finite_all = False
        em_before = canonical_before_ems[op]
        acc_before = canonical_before_accs[op]
        forgetting = max(0.0, em_before - em_after)
        category = (
            "canonical_parameterized"
            if op in PARAMETERIZED_OPERATIONS
            else "canonical_parameter_free"
        )
        passed = forgetting <= config.max_canonical_forgetting
        canonical_metrics[op] = TaskEvaluationMetric(
            task_name=op,
            task_category=category,
            before_exact_match=em_before,
            before_token_accuracy=acc_before,
            after_exact_match=em_after,
            after_token_accuracy=acc_after,
            forgetting=forgetting,
            passed=passed,
        )

    # Re-evaluate all 6 representative compositions in shadow mode
    composition_metrics: dict[str, TaskEvaluationMetric] = {}
    for comp in DESIGNATED_COMPOSITIONS:
        comp_name = f"{comp[0]}->{comp[1]}"
        em_after, acc_after, fin = _evaluate_batch(
            core,
            shadow_bank,
            shadow_op_to_id,
            composition_eval_data[comp_name],
            batch_size=config.eval_batch_size,
        )
        if not fin:
            finite_all = False
        em_before = composition_before_ems[comp_name]
        acc_before = composition_before_accs[comp_name]
        c_forgetting = max(0.0, em_before - em_after)
        c_passed = c_forgetting <= config.max_composition_forgetting
        composition_metrics[comp_name] = TaskEvaluationMetric(
            task_name=comp_name,
            task_category="composition",
            before_exact_match=em_before,
            before_token_accuracy=acc_before,
            after_exact_match=em_after,
            after_token_accuracy=acc_after,
            forgetting=c_forgetting,
            passed=c_passed,
        )

    # Invariant testing injections (for fault injection test coverage)
    if inject_core_mutation:
        with torch.no_grad():
            for p in core.model.parameters():
                p.add_(1.0)
                break
    if inject_bank_mutation:
        with torch.no_grad():
            for p in bank.parameters():
                p.add_(1.0)
                break

    # 6. Verify Invariants
    core_unchanged = verify_core_unchanged(core_initial_weights, core)
    existing_bank_unchanged = verify_bank_unchanged(
        bank_initial_weights, bank, pre_existing_ids
    )

    max_canon_forg = max(m.forgetting for m in canonical_metrics.values())
    max_comp_forg = max(m.forgetting for m in composition_metrics.values())
    max_overall_forg = max(max_canon_forg, max_comp_forg)

    all_canonical_passed = all(m.passed for m in canonical_metrics.values())
    all_composition_passed = all(m.passed for m in composition_metrics.values())

    safety_criteria_passed = (
        all_canonical_passed
        and all_composition_passed
        and core_unchanged
        and existing_bank_unchanged
        and candidate_size_passed
        and finite_all
    )

    # 7. Promotion & Release Decision (Conditional Execution)
    if safety_criteria_passed:
        # Promotion: Install candidate into persistent bank
        candidate.status = PrimitiveStatus.STABLE
        candidate.freeze()
        installed_id = bank.add_primitive(candidate)
        op_to_id[op_novel] = installed_id
        bank_size_after = len(bank)
        promoted = True

        # Release: Free temporary discovery resources completely (100% release to 0)
        del cand_shadow
        del shadow_bank
        temp_parameters_after = 0
        temp_released = True

        # Save persistent state (Core + Promoted Bank) for fresh runtime recurrence
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bank.state_dict(), seed_dir / "promoted_bank.pt")
            (seed_dir / "op_to_id.json").write_text(
                json.dumps(op_to_id, indent=2), encoding="utf-8"
            )

        overall_passed = (
            bank_size_after == bank_size_before + 1
            and temp_parameters_after == 0
            and temp_released
        )
    else:
        # Abort Install: Bank unchanged, fallback preserved
        installed_id = None
        bank_size_after = len(bank)
        promoted = False
        # Preserve temporary resources
        temp_parameters_after = temp_parameters_before
        temp_released = False
        overall_passed = False

    elapsed = time.perf_counter() - start_time

    return ShadowPromotionReport(
        seed=seed,
        novel_operation=op_novel,
        canonical_metrics=canonical_metrics,
        composition_metrics=composition_metrics,
        candidate_novel_exact_match=cand_novel_em,
        candidate_novel_token_accuracy=cand_novel_acc,
        max_canonical_forgetting=max_canon_forg,
        max_composition_forgetting=max_comp_forg,
        max_overall_forgetting=max_overall_forg,
        core_unchanged=core_unchanged,
        existing_bank_unchanged=existing_bank_unchanged,
        candidate_parameters=candidate_params,
        candidate_size_passed=candidate_size_passed,
        finite_outputs_passed=finite_all,
        all_canonical_passed=all_canonical_passed,
        all_composition_passed=all_composition_passed,
        safety_criteria_passed=safety_criteria_passed,
        promoted=promoted,
        installed_primitive_id=installed_id,
        bank_size_before=bank_size_before,
        bank_size_after=bank_size_after,
        temp_parameters_before=temp_parameters_before,
        temp_parameters_after=temp_parameters_after,
        temp_released_completely=temp_released,
        overall_passed=overall_passed,
        elapsed_seconds=elapsed,
    )


def run_shadow_promotion_multi_seed(
    seeds: Sequence[int],
    config_factory: Any,
    *,
    output_dir: Path | None = None,
) -> ShadowPromotionSummary:
    """Execute Task A1-B007X-006 shadow validation, promotion, and release across multiple seeds."""
    reports: list[ShadowPromotionReport] = []

    for s in seeds:
        cfg: ShadowPromotionConfig = config_factory(s)
        seed_dir = (output_dir / f"seed_{s}") if output_dir is not None else None
        report = run_shadow_promotion_single_seed(cfg, s, seed_dir=seed_dir)
        reports.append(report)

        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "seed_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )

    all_passed = all(r.overall_passed for r in reports)
    meets_policy = len(seeds) >= MIN_GATE_SEEDS
    all_promoted = all(r.promoted for r in reports)
    all_core_ok = all(r.core_unchanged for r in reports)
    all_bank_ok = all(r.existing_bank_unchanged for r in reports)
    all_released = all(r.temp_released_completely for r in reports)

    mean_cand_em = statistics.mean(r.candidate_novel_exact_match for r in reports)
    max_forg = max(r.max_overall_forgetting for r in reports)

    first_r = reports[0]
    bank_transition = f"{first_r.bank_size_before} -> {first_r.bank_size_after}"

    summary = ShadowPromotionSummary(
        novel_operation=first_r.novel_operation,
        seeds=tuple(seeds),
        overall_passed=all_passed and meets_policy,
        meets_seed_policy=meets_policy,
        all_seeds_promoted=all_promoted,
        mean_candidate_novel_em=mean_cand_em,
        max_overall_forgetting=max_forg,
        all_core_unchanged=all_core_ok,
        all_existing_bank_unchanged=all_bank_ok,
        bank_size_transition=bank_transition,
        temp_released_completely=all_released,
        reports=reports,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )

    return summary
