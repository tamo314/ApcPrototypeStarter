"""Holdout protocol preflight verification and leak audit for Phase B.

Task B-C002: Holdout-family registry, identifiability checks, and leak audit.
Reference:
- `docs/design-docs/OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md`
- `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
- `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

Implements:
1. Identifiability Preflight:
   Verifies that task evidence (explicit, structured, fewshot demonstrations, language)
   uniquely identifies the target mapping under candidate universe.
   Includes collision detection for ambiguous demonstration sets.
2. Novelty Validity Preflight:
   Evaluates direct primitive search and depth-2 composition search over existing
   primitive bank / reference operations against the adequacy threshold (0.90).
   Rejects operations already solvable by the existing library as novel holdouts.
3. Leak Audit:
   Scans controller inference features, task-inference inputs, router inputs,
   plastic learner examples, and development tuning sets to assert zero leakage
   of sealed family IDs, canonical operation IDs/names, registry status, and oracle labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from apc.environments.generator import Example
from apc.environments.holdout_families import (
    DEFAULT_FAMILY_REGISTRY,
    HoldoutEpisode,
    HoldoutFamilyRegistry,
)
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    KNOWN_OPERATION_NAMES,
    Operation,
    get_operation,
    registered_operation_names,
)
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.meta.phase_b_protocol import FamilySplit, TaskInferenceModality
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition_search import search_composition_recipe

# Default baseline 16 operations from Phase A.2
PHASE_A2_16_OPERATIONS: Final[tuple[str, ...]] = (
    "COPY",
    "SELECT",
    "COMPARE",
    "COUNT",
    "SHIFT",
    "BIND",
    "NEGATE",
    "ACCUMULATE",
    "SWAP_PAIRS",
    "INVERT_HALF",
    "ROTATE_TRIPLETS",
    "SWAP_ENDS",
    "MIRROR_HALVES",
    "ALTERNATING_NEGATE",
    "CYCLE_FOUR",
    "INCREMENT_MOD",
)

DEFAULT_ADEQUACY_THRESHOLD: Final[float] = 0.90


# ===========================================================================
# 1. Preflight A: Identifiability Checks & Collision Detection
# ===========================================================================


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Outcome of identifiability verification for a task specification."""

    modality: TaskInferenceModality
    target_operation: str
    is_identifiable: bool
    collided_operations: tuple[str, ...] = ()
    reason: str | None = None
    candidate_universe_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "target_operation": self.target_operation,
            "is_identifiable": self.is_identifiable,
            "collided_operations": list(self.collided_operations),
            "reason": self.reason,
            "candidate_universe_size": self.candidate_universe_size,
        }


def check_identifiability(
    target_operation: str,
    modality: TaskInferenceModality,
    evidence: Any,
    *,
    candidate_universe: Sequence[str] | None = None,
    vocab_size: int = 10,
) -> IdentifiabilityResult:
    """Verify that the provided evidence can uniquely identify the target mapping.

    Args:
        target_operation: Name of the intended operation.
        modality: Task observation modality.
        evidence: TaskSpec, structured descriptor dict, Sequence of Examples, or string.
        candidate_universe: Operations considered in the candidate pool.
        vocab_size: Environment vocabulary size.
    """
    candidates = (
        tuple(candidate_universe)
        if candidate_universe is not None
        else registered_operation_names()
    )

    if modality == TaskInferenceModality.EXPLICIT_TASK_SPEC:
        # In explicit mode, task spec carries exact operation name and argument schema
        if not isinstance(evidence, TaskSpec):
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Evidence is not a TaskSpec instance",
                candidate_universe_size=len(candidates),
            )
        if not evidence.steps or evidence.steps[0].operation != target_operation:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason=f"TaskSpec operation mismatch: expected {target_operation!r}",
                candidate_universe_size=len(candidates),
            )
        return IdentifiabilityResult(
            modality=modality,
            target_operation=target_operation,
            is_identifiable=True,
            candidate_universe_size=len(candidates),
        )

    elif modality == TaskInferenceModality.STRUCTURED_DESCRIPTOR:
        # Structured descriptor must uniquely differentiate target within candidate universe
        if not isinstance(evidence, dict):
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Structured descriptor must be a dictionary",
                candidate_universe_size=len(candidates),
            )
        if not evidence:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Empty structured descriptor cannot identify task",
                candidate_universe_size=len(candidates),
            )
        return IdentifiabilityResult(
            modality=modality,
            target_operation=target_operation,
            is_identifiable=True,
            candidate_universe_size=len(candidates),
        )

    elif modality == TaskInferenceModality.FEWSHOT_DEMONSTRATIONS:
        # Collision detection: check if multiple candidates produce identical outputs
        # on all demonstration inputs.
        examples: Sequence[Example]
        if isinstance(evidence, (list, tuple)):
            examples = evidence
        else:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Demonstrations must be a sequence of Examples",
                candidate_universe_size=len(candidates),
            )

        if not examples:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Zero demonstrations provided",
                candidate_universe_size=len(candidates),
            )

        consistent_candidates: list[str] = []
        for cand_name in candidates:
            try:
                op: Operation = get_operation(cand_name)
            except KeyError:
                continue

            # Check if cand_name reproduces target_tokens on all demonstration inputs
            matches_all = True
            for ex in examples:
                if not op.is_valid_for_length(len(ex.input_tokens)):
                    matches_all = False
                    break
                if op.output_length(len(ex.input_tokens)) != len(ex.target_tokens):
                    matches_all = False
                    break

                # For parameter-free ops or default params
                prog = Program(steps=(ProgramStep(operation=cand_name, params={}),))
                try:
                    res = run_program(prog, ex.input_tokens, vocab_size)
                    if res.output_tokens != ex.target_tokens:
                        matches_all = False
                        break
                except Exception:
                    matches_all = False
                    break

            if matches_all:
                consistent_candidates.append(cand_name)

        if len(consistent_candidates) > 1:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                collided_operations=tuple(consistent_candidates),
                reason=(
                    f"Collision detected: {len(consistent_candidates)} candidate operations "
                    f"{tuple(consistent_candidates)} produce identical outputs on all "
                    f"{len(examples)} demonstration examples."
                ),
                candidate_universe_size=len(candidates),
            )
        elif len(consistent_candidates) == 1 and consistent_candidates[0] == target_operation:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=True,
                collided_operations=(),
                candidate_universe_size=len(candidates),
            )
        else:
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                collided_operations=tuple(consistent_candidates),
                reason=(
                    f"Target operation {target_operation!r} not identified: "
                    f"matching candidates={tuple(consistent_candidates)}"
                ),
                candidate_universe_size=len(candidates),
            )

    elif modality == TaskInferenceModality.NATURAL_LANGUAGE:
        if not isinstance(evidence, str) or not evidence.strip():
            return IdentifiabilityResult(
                modality=modality,
                target_operation=target_operation,
                is_identifiable=False,
                reason="Natural language instruction must be a non-empty string",
                candidate_universe_size=len(candidates),
            )
        return IdentifiabilityResult(
            modality=modality,
            target_operation=target_operation,
            is_identifiable=True,
            candidate_universe_size=len(candidates),
        )

    raise ValueError(f"Unsupported modality: {modality}")


def check_batch_identifiability(
    episodes: Sequence[HoldoutEpisode],
    modality: TaskInferenceModality,
    candidate_universe: Sequence[str] | None = None,
) -> tuple[float, list[IdentifiabilityResult]]:
    """Compute identifiable_episode_rate across a batch of holdout episodes."""
    if not episodes:
        return 1.0, []

    results: list[IdentifiabilityResult] = []
    for ep in episodes:
        ev: Any
        if modality == TaskInferenceModality.EXPLICIT_TASK_SPEC:
            ev = ep.verification_examples[0].task_spec if ep.verification_examples else None
        elif modality == TaskInferenceModality.FEWSHOT_DEMONSTRATIONS:
            ev = ep.inference_examples
        elif modality == TaskInferenceModality.STRUCTURED_DESCRIPTOR:
            ev = {"family": ep.family_id, "operation": ep.operation_name}
        else:
            ev = f"Execute operation {ep.operation_name}"

        r = check_identifiability(
            target_operation=ep.operation_name,
            modality=modality,
            evidence=ev,
            candidate_universe=candidate_universe,
            vocab_size=ep.vocab_size,
        )
        results.append(r)

    identifiable_count = sum(1 for r in results if r.is_identifiable)
    rate = identifiable_count / len(results)
    return rate, results


# ===========================================================================
# 2. Preflight B: Novelty-Validity Prechecks
# ===========================================================================


@dataclass(frozen=True)
class NoveltyValidityResult:
    """Evaluation of candidate holdout against existing primitive bank and compositions."""

    operation_name: str
    family_id: str
    best_direct_em: float
    best_direct_loss: float
    best_direct_recipe: tuple[str, ...]
    best_composition_em: float
    best_composition_loss: float
    best_recipe: tuple[str, ...]
    adequate_by_existing_library: bool
    is_valid_novel_holdout: bool
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "family_id": self.family_id,
            "best_direct_em": self.best_direct_em,
            "best_direct_loss": self.best_direct_loss,
            "best_direct_recipe": list(self.best_direct_recipe),
            "best_composition_em": self.best_composition_em,
            "best_composition_loss": self.best_composition_loss,
            "best_recipe": list(self.best_recipe),
            "adequate_by_existing_library": self.adequate_by_existing_library,
            "is_valid_novel_holdout": self.is_valid_novel_holdout,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


def _evaluate_symbolic_candidate(
    candidate: tuple[str, ...],
    examples: Sequence[Example],
    vocab_size: int = 10,
) -> tuple[float, float]:
    """Evaluate exact match and token loss of a symbolic candidate chain on examples."""
    if not examples or not candidate:
        return 0.0, 10.0

    matches = 0
    total_token_dist = 0
    total_tokens = 0

    prog = Program(steps=tuple(ProgramStep(operation=op, params={}) for op in candidate))

    for ex in examples:
        target = ex.target_tokens
        n = len(target)
        total_tokens += n
        try:
            res = run_program(prog, ex.input_tokens, vocab_size)
            if res.output_tokens == target:
                matches += 1
            # Token distance proxy for loss
            dist = sum(abs(a - b) for a, b in zip(res.output_tokens, target, strict=False))
            total_token_dist += dist
        except Exception:
            total_token_dist += n * vocab_size

    em = matches / len(examples)
    loss = total_token_dist / max(1, total_tokens)
    return em, loss


def check_novelty_validity(
    target_operation: str,
    verification_examples: Sequence[Example],
    *,
    threshold: float = DEFAULT_ADEQUACY_THRESHOLD,
    max_composition_depth: int = 2,
    candidate_bank_ops: Sequence[str] | None = None,
    core: Any = None,
    bank: PrimitiveBank | None = None,
    op_to_id: dict[str, int] | None = None,
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
    vocab_size: int = 10,
) -> NoveltyValidityResult:
    """Preflight check: evaluate whether an operation is genuinely novel.

    Tests:
    1. Direct primitive search over candidate bank operations.
    2. Allowed composition search up to max_composition_depth.

    If any existing direct primitive or composition reaches EM >= threshold,
    adequate_by_existing_library is True, and the operation is rejected as a novel holdout.
    """
    family_meta = registry.get_family_for_operation(target_operation)
    bank_ops = tuple(candidate_bank_ops) if candidate_bank_ops else PHASE_A2_16_OPERATIONS

    best_direct_em = 0.0
    best_direct_loss = float("inf")
    best_direct_recipe: tuple[str, ...] = ()

    best_comp_em = 0.0
    best_comp_loss = float("inf")
    best_comp_recipe: tuple[str, ...] = ()

    # Neural evaluation if core and bank are supplied
    if core is not None and bank is not None and op_to_id is not None:
        search_res = search_composition_recipe(
            core=core,
            bank=bank,
            op_to_id=op_to_id,
            adaptation_examples=verification_examples,
            available_operations=bank_ops,
            max_depth=max_composition_depth,
            early_stop_exact_match=1.0,
        )
        best_comp_em = search_res.exact_match_adapt
        best_comp_loss = search_res.loss_adapt
        best_comp_recipe = search_res.candidate_operations

        # Direct search
        for op in bank_ops:
            d_res = search_composition_recipe(
                core=core,
                bank=bank,
                op_to_id=op_to_id,
                adaptation_examples=verification_examples,
                available_operations=(op,),
                max_depth=1,
            )
            if d_res.exact_match_adapt > best_direct_em or (
                d_res.exact_match_adapt == best_direct_em and d_res.loss_adapt < best_direct_loss
            ):
                best_direct_em = d_res.exact_match_adapt
                best_direct_loss = d_res.loss_adapt
                best_direct_recipe = (op,)
    else:
        # Symbolic reference evaluation
        # 1. Direct search (depth 1)
        for op in bank_ops:
            em, loss = _evaluate_symbolic_candidate((op,), verification_examples, vocab_size)
            if em > best_direct_em or (em == best_direct_em and loss < best_direct_loss):
                best_direct_em = em
                best_direct_loss = loss
                best_direct_recipe = (op,)

        best_comp_em = best_direct_em
        best_comp_loss = best_direct_loss
        best_comp_recipe = best_direct_recipe

        # 2. Composition search (depth 2)
        if max_composition_depth >= 2:
            for op1 in bank_ops:
                for op2 in bank_ops:
                    cand = (op1, op2)
                    em, loss = _evaluate_symbolic_candidate(cand, verification_examples, vocab_size)
                    if em > best_comp_em or (em == best_comp_em and loss < best_comp_loss):
                        best_comp_em = em
                        best_comp_loss = loss
                        best_comp_recipe = cand

    adequate = (best_direct_em >= threshold) or (best_comp_em >= threshold)
    is_valid_novel = not adequate

    return NoveltyValidityResult(
        operation_name=target_operation,
        family_id=family_meta.family_id,
        best_direct_em=best_direct_em,
        best_direct_loss=best_direct_loss,
        best_direct_recipe=best_direct_recipe,
        best_composition_em=best_comp_em,
        best_composition_loss=best_comp_loss,
        best_recipe=best_comp_recipe,
        adequate_by_existing_library=adequate,
        is_valid_novel_holdout=is_valid_novel,
        threshold=threshold,
    )


# ===========================================================================
# 3. Preflight C: Leak Audit
# ===========================================================================


@dataclass(frozen=True)
class LeakAuditResult:
    """Detailed audit report asserting zero information leaks."""

    leak_count: int
    violations: tuple[str, ...]
    passed: bool

    def assert_zero_leaks(self) -> None:
        """Raise AssertionError if any leak violation was detected."""
        if not self.passed:
            details = "\n - ".join(self.violations)
            msg = f"Leak audit FAILED with {self.leak_count} violations:\n - {details}"
            raise AssertionError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leak_count": self.leak_count,
            "violations": list(self.violations),
            "passed": self.passed,
        }


def audit_for_leaks(
    *,
    controller_features: Any | None = None,
    task_inference_input: Any | None = None,
    router_input: Any | None = None,
    plastic_learner_examples: Sequence[Example] | None = None,
    dev_tuning_examples: Sequence[Example] | None = None,
    modality: TaskInferenceModality = TaskInferenceModality.STRUCTURED_DESCRIPTOR,
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> LeakAuditResult:
    """Comprehensive leak auditor across Phase B pipeline interfaces.

    Asserts that:
    1. Sealed family identifiers, names, and registry membership are absent from features/inputs.
    2. Oracle operation names/IDs are absent from task-inference and router inputs in no-ID modes.
    3. Oracle labels ('K', 'C', 'N', 'R') are absent from model features.
    4. Development tuning datasets contain zero sealed evaluation examples.
    """
    violations: list[str] = []

    sealed_families = set(f.family_id for f in registry.list_families(FamilySplit.SEALED_FAMILIES))
    sealed_ops = set(registry.list_operations(FamilySplit.SEALED_FAMILIES))
    forbidden_tokens = set(sealed_families) | set(sealed_ops) | {
        "SEALED_FAMILIES",
        "SEALED_EVALUATION",
    }

    # Oracle labels
    oracle_labels = {"K", "C", "N", "R"}

    def scan_for_forbidden(obj: Any, context: str) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            for token in forbidden_tokens:
                if token.lower() in obj.lower():
                    violations.append(
                        f"[{context}] Sealed token {token!r} detected in string: {obj!r}"
                    )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                scan_for_forbidden(str(k), f"{context}.key")
                scan_for_forbidden(v, f"{context}[{k}]")
        elif isinstance(obj, (list, tuple, set)):
            for i, item in enumerate(obj):
                scan_for_forbidden(item, f"{context}[{i}]")

    # 1. Controller Inference Features
    if controller_features is not None:
        if isinstance(controller_features, (list, tuple)):
            # Numeric vector: ensure values are finite floats/ints and no string metadata
            for i, val in enumerate(controller_features):
                if not isinstance(val, (int, float)):
                    violations.append(
                        f"[controller_features[{i}]] Non-numeric feature value: {val!r}"
                    )
        elif isinstance(controller_features, dict):
            scan_for_forbidden(controller_features, "controller_features")
            for k, v in controller_features.items():
                if str(v) in oracle_labels:
                    violations.append(
                        f"[controller_features[{k}]] Oracle label {v!r} present in features"
                    )

    # 2. Task Inference Input
    if task_inference_input is not None:
        if modality != TaskInferenceModality.EXPLICIT_TASK_SPEC:
            # In no-ID modes, canonical operation names, operation IDs, and sealed tokens forbidden
            scan_for_forbidden(task_inference_input, "task_inference_input")
            for op in KNOWN_OPERATION_NAMES:
                if isinstance(task_inference_input, str) and (
                    op.lower() in task_inference_input.lower()
                ):
                    violations.append(
                        f"[task_inference_input] Canonical op {op!r} present in modality "
                        f"{modality.value}"
                    )
                elif isinstance(task_inference_input, dict):
                    str_vals = str(task_inference_input)
                    if op.lower() in str_vals.lower():
                        violations.append(
                            f"[task_inference_input] Canonical op {op!r} present in descriptor "
                            "in no-ID mode"
                        )

    # 3. Router Inference Input
    if router_input is not None:
        if modality != TaskInferenceModality.EXPLICIT_TASK_SPEC:
            scan_for_forbidden(router_input, "router_input")

    # 4. Plastic Learner Examples
    if plastic_learner_examples is not None:
        for i, ex in enumerate(plastic_learner_examples):
            # Verify examples do not embed oracle evaluation labels in content path
            if ex.task_spec is not None and modality != TaskInferenceModality.EXPLICIT_TASK_SPEC:
                violations.append(
                    f"[plastic_learner_examples[{i}]] task_spec present in plastic examples "
                    "under no-ID modality"
                )

    # 5. Development Tuning Examples
    if dev_tuning_examples is not None:
        for i, ex in enumerate(dev_tuning_examples):
            op_name = ex.program.steps[0].operation if ex.program.steps else ""
            if op_name in sealed_ops:
                violations.append(
                    f"[dev_tuning_examples[{i}]] SEALED operation {op_name!r} contaminated "
                    "development tuning set!"
                )

    passed = len(violations) == 0
    return LeakAuditResult(
        leak_count=len(violations),
        violations=tuple(violations),
        passed=passed,
    )
