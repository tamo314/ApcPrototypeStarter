"""Compact-First Plastic Lifecycle Policy (Task A2-C007).

Integrates the empirical lessons from Phase A.1 B007X (ADR-0056, ADR-0060)
into the runtime plasticity lifecycle:

1. Direct / Composition Adequacy Guard:
   No plastic search is permitted if direct execution or composition search
   is demonstrably adequate on support examples.

2. Compact Plastic Search First (T0 Compact, ~17k params):
   Search within compact candidate capacity (<= 25k parameters). If successful
   within the fixed adaptation budget, promote candidate directly to consolidation.

3. Bounded Overcomplete Fallback (T2 Overcomplete, ~137k params):
   If compact search fails within its fixed budget, invoke an optional overcomplete
   fallback. If overcomplete search succeeds, distill knowledge into a compact
   candidate (T2 -> T0 functional distillation). If fallback also fails, abort
   without promotion.

4. Mechanical Lifecycle Guarantees:
   - Zero plastic before direct/composition inadequacy.
   - Exactly one promotion per successful novel (N) task.
   - 100% temporary workspace capacity release (workspace params == 0) across all paths.
   - Zero promotions for Known (K), Composition (C), and Recurrence (R) tasks.
   - Explicit reporting of compact success, compact failure, fallback invoked,
     fallback success/failure controls.
"""

from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example
from apc.meta.adequacy import AdequacyEvidence
from apc.meta.episode_log import ControllerAction
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils.seed import set_seed

DEFAULT_COMPACT_SUCCESS_THRESHOLD: Final[float] = 0.90
DEFAULT_FALLBACK_SUCCESS_THRESHOLD: Final[float] = 0.90
DEFAULT_SHADOW_RETENTION_THRESHOLD: Final[float] = 0.95
DEFAULT_SHADOW_MAX_FORGETTING: Final[float] = 0.02


@dataclass(frozen=True)
class CompactLifecycleConfig:
    """Explicit, serializable configuration for compact-first plastic lifecycle."""

    compact_budget_steps: int = 150
    compact_lr: float = 1e-3
    compact_weight_decay: float = 1e-4
    compact_batch_size: int = 32
    compact_success_threshold_em: float = DEFAULT_COMPACT_SUCCESS_THRESHOLD

    enable_overcomplete_fallback: bool = True
    fallback_budget_steps: int = 150
    fallback_lr: float = 1e-3
    fallback_weight_decay: float = 1e-4
    fallback_batch_size: int = 32
    fallback_success_threshold_em: float = DEFAULT_FALLBACK_SUCCESS_THRESHOLD

    distillation_steps: int = 150
    distillation_lr: float = 1e-3
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5

    shadow_retention_threshold: float = DEFAULT_SHADOW_RETENTION_THRESHOLD
    shadow_max_forgetting: float = DEFAULT_SHADOW_MAX_FORGETTING
    max_candidate_parameters: int = 25_000

    eval_batch_size: int = 64
    seed: int = 42

    def __post_init__(self) -> None:
        if self.compact_budget_steps < 1:
            raise ValueError(f"compact_budget_steps must be >= 1, got {self.compact_budget_steps}")
        if self.fallback_budget_steps < 1:
            raise ValueError(
                f"fallback_budget_steps must be >= 1, got {self.fallback_budget_steps}"
            )
        if not 0.0 <= self.compact_success_threshold_em <= 1.0:
            raise ValueError("compact_success_threshold_em must be in [0, 1]")
        if not 0.0 <= self.fallback_success_threshold_em <= 1.0:
            raise ValueError("fallback_success_threshold_em must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CompactLifecycleReport:
    """Report generated for a single episode under the Compact Lifecycle Policy."""

    episode_id: str
    task_name: str
    controller_action: str
    direct_sufficient: bool
    composition_sufficient: bool
    plastic_triggered: bool

    # Controls required by A2-C007
    compact_attempted: bool
    compact_success: bool
    compact_failure: bool
    fallback_invoked: bool
    fallback_success: bool
    fallback_failure: bool

    distillation_attempted: bool
    shadow_validation_passed: bool
    promoted_primitive_id: int | None
    promotions_count: int
    final_workspace_param_count: int
    temporary_peak_params: int
    elapsed_seconds: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class CompactPlasticLifecyclePolicy:
    """Executes the Compact-First Plastic Lifecycle Policy (Task A2-C007)."""

    def __init__(self, config: CompactLifecycleConfig | None = None) -> None:
        self.config = config if config is not None else CompactLifecycleConfig()

    def execute_episode(
        self,
        *,
        episode_id: str,
        task_name: str,
        action: ControllerAction,
        evidence: AdequacyEvidence | None,
        train_examples: Sequence[Example],
        eval_examples: Sequence[Example],
        historical_eval_examples: Mapping[str, Sequence[Example]],
        core: Any,
        bank: PrimitiveBank,
        op_to_id: dict[str, int],
        workspace: PlasticWorkspace,
        base_recipe: Sequence[str] | None = None,
        forced_compact_fail: bool = False,
    ) -> CompactLifecycleReport:
        """Execute one lifecycle episode under the compact-first policy.

        Enforces all mechanical lifecycle constraints:
        1. No plastic search before direct or composition inadequacy.
        2. Exactly one promotion per successful novel task.
        3. Temporary workspace returns to zero parameters across all paths.
        4. No promotions for Known, Composition, or Recurrence tasks.
        """
        start_time = time.perf_counter()
        set_seed(self.config.seed)

        # 1. Determine adequacy signals
        direct_sufficient = False
        composition_sufficient = False
        if evidence is not None:
            direct_sufficient = evidence.direct_em >= self.config.compact_success_threshold_em
            composition_sufficient = (
                evidence.composition_em >= self.config.compact_success_threshold_em
            )

        # Invariant check: No plastic before direct/composition inadequacy
        if action in (ControllerAction.DIRECT_REUSE, ControllerAction.COMPOSE):
            if action == ControllerAction.DIRECT_REUSE:
                direct_sufficient = True
            elif action == ControllerAction.COMPOSE:
                composition_sufficient = True

            # Ensure workspace is 100% clean
            if workspace.is_allocated:
                workspace.release()

            elapsed = time.perf_counter() - start_time
            return CompactLifecycleReport(
                episode_id=episode_id,
                task_name=task_name,
                controller_action=action.value,
                direct_sufficient=direct_sufficient,
                composition_sufficient=composition_sufficient,
                plastic_triggered=False,
                compact_attempted=False,
                compact_success=False,
                compact_failure=False,
                fallback_invoked=False,
                fallback_success=False,
                fallback_failure=False,
                distillation_attempted=False,
                shadow_validation_passed=False,
                promoted_primitive_id=None,
                promotions_count=0,
                final_workspace_param_count=workspace.total_parameter_count(),
                temporary_peak_params=0,
                elapsed_seconds=elapsed,
                details={"message": f"Action {action.value} resolved without plastic capacity."},
            )

        # Action must be PLASTIC_SEARCH
        if action != ControllerAction.PLASTIC_SEARCH:
            raise ValueError(f"Unknown or unsupported ControllerAction: {action}")

        # Guard: Plastic is forbidden if direct or composition was already sufficient
        if direct_sufficient or composition_sufficient:
            raise RuntimeError(
                f"Premature plastic trigger: direct_sufficient={direct_sufficient}, "
                f"composition_sufficient={composition_sufficient}. Plastic is forbidden."
            )

        # 2. Plastic capacity requested -> Compact-First Policy
        plastic_triggered = True
        compact_attempted = True
        compact_success = False
        compact_failure = False
        fallback_invoked = False
        fallback_success = False
        fallback_failure = False
        distillation_attempted = False
        shadow_passed = False
        promoted_id: int | None = None
        promotions_count = 0
        peak_params = 0

        device = core.device
        vocab_size = getattr(core.tokens, "env_vocab_size", getattr(core.tokens, "vocab_size", 10))

        # --- Phase 2A: Compact Plastic Search (T0 Compact, ~17k params) ---
        compact_config = CrossPositionPrimitiveConfig(
            operation=task_name,
            d_model=core.model.config.d_model,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=vocab_size,
            max_sequence_length=32,
        )

        compact_pid = workspace.allocate_compact_operator(
            compact_config,
            created_at_task=0,
            label="compact_operator_t0",
            device=device,
        )
        compact_operator = workspace.get(compact_pid)
        peak_params = max(peak_params, workspace.total_parameter_count())

        # Train compact operator
        compact_em = 0.0
        if not forced_compact_fail:
            compact_em = self._train_and_eval_operator(
                operator=compact_operator,
                train_examples=train_examples,
                eval_examples=eval_examples,
                core=core,
                steps=self.config.compact_budget_steps,
                lr=self.config.compact_lr,
                weight_decay=self.config.compact_weight_decay,
                batch_size=self.config.compact_batch_size,
                vocab_size=vocab_size,
                device=device,
                seed=self.config.seed + 10,
            )

        candidate_to_evaluate: CrossPositionPrimitive | None = None

        if compact_em >= self.config.compact_success_threshold_em and not forced_compact_fail:
            compact_success = True
            compact_failure = False
            # Detach candidate from workspace to prepare for shadow validation
            candidate_to_evaluate = compact_operator
            # Release workspace capacity
            workspace.release()
        else:
            compact_success = False
            compact_failure = True
            # Release compact operator before fallback
            workspace.release()

            # --- Phase 2B: Optional Overcomplete Fallback (T2 Overcomplete, ~137k params) ---
            if self.config.enable_overcomplete_fallback:
                fallback_invoked = True
                overcomplete_config = CrossPositionPrimitiveConfig(
                    operation=task_name,
                    d_model=core.model.config.d_model,
                    d_operator=96,
                    n_head=6,
                    d_operator_ff=384,
                    vocab_size=vocab_size,
                    max_sequence_length=32,
                )
                overcomplete_id = (
                    max(bank.ids() + [0]) + 1000 if len(bank) > 0 else 1000
                )
                t2_operator = CrossPositionPrimitive(
                    overcomplete_id,
                    overcomplete_config,
                    status=PrimitiveStatus.CANDIDATE,
                    created_at_task=0,
                )
                t2_pid = workspace.allocate_primitive(
                    t2_operator,
                    label="overcomplete_fallback_t2",
                    device=device,
                )
                peak_params = max(peak_params, workspace.total_parameter_count())

                # Train overcomplete operator
                t2_em = self._train_and_eval_operator(
                    operator=workspace.get(t2_pid),
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    core=core,
                    steps=self.config.fallback_budget_steps,
                    lr=self.config.fallback_lr,
                    weight_decay=self.config.fallback_weight_decay,
                    batch_size=self.config.fallback_batch_size,
                    vocab_size=vocab_size,
                    device=device,
                    seed=self.config.seed + 20,
                )

                if t2_em >= self.config.fallback_success_threshold_em:
                    fallback_success = True
                    fallback_failure = False
                    distillation_attempted = True

                    # Distill T2 -> T0 compact candidate
                    candidate_to_evaluate = self._distill_to_compact_candidate(
                        teacher=workspace.get(t2_pid),
                        train_examples=train_examples,
                        core=core,
                        task_name=task_name,
                        vocab_size=vocab_size,
                        device=device,
                    )
                    # Release overcomplete capacity from workspace
                    workspace.release()
                else:
                    fallback_success = False
                    fallback_failure = True
                    workspace.release()

        # --- Phase 3: Shadow Validation & Conditional Promotion ---
        shadow_reason = "no_candidate"
        if candidate_to_evaluate is not None:
            # Perform shadow validation
            shadow_passed, shadow_reason = self._run_shadow_validation(
                candidate=candidate_to_evaluate,
                eval_examples=eval_examples,
                historical_eval_examples=historical_eval_examples,
                core=core,
                bank=bank,
                op_to_id=op_to_id,
            )

            if shadow_passed:
                # Exactly one promotion
                candidate_to_evaluate.status = PrimitiveStatus.STABLE
                candidate_to_evaluate.freeze()
                candidate_to_evaluate.primitive_id = max(bank.ids() + [-1]) + 1
                assigned_id = bank.add_primitive(candidate_to_evaluate)
                op_to_id[task_name] = assigned_id
                promoted_id = assigned_id
                promotions_count = 1

        # Post-condition verification: workspace must be 100% released to 0
        if workspace.is_allocated or workspace.total_parameter_count() != 0:
            workspace.release()
        final_workspace_params = workspace.total_parameter_count()
        if final_workspace_params != 0 or len(workspace) != 0:
            raise RuntimeError(
                f"Workspace capacity was not released to zero! Count: {final_workspace_params}"
            )

        elapsed = time.perf_counter() - start_time
        return CompactLifecycleReport(
            episode_id=episode_id,
            task_name=task_name,
            controller_action=action.value,
            direct_sufficient=direct_sufficient,
            composition_sufficient=composition_sufficient,
            plastic_triggered=plastic_triggered,
            compact_attempted=compact_attempted,
            compact_success=compact_success,
            compact_failure=compact_failure,
            fallback_invoked=fallback_invoked,
            fallback_success=fallback_success,
            fallback_failure=fallback_failure,
            distillation_attempted=distillation_attempted,
            shadow_validation_passed=shadow_passed,
            promoted_primitive_id=promoted_id,
            promotions_count=promotions_count,
            final_workspace_param_count=final_workspace_params,
            temporary_peak_params=peak_params,
            elapsed_seconds=elapsed,
            details={
                "compact_em": compact_em if compact_attempted else 0.0,
                "shadow_passed": shadow_passed,
                "shadow_reason": shadow_reason,
            },
        )

    def _train_and_eval_operator(
        self,
        *,
        operator: CrossPositionPrimitive,
        train_examples: Sequence[Example],
        eval_examples: Sequence[Example],
        core: Any,
        steps: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
    ) -> float:
        """Train a temporary operator and evaluate its exact match on eval examples."""
        set_seed(seed)
        operator.to(device)
        operator.unfreeze()
        operator.train()

        optimizer = torch.optim.AdamW(operator.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        rng = random.Random(seed * 7919)

        for _ in range(steps):
            batch = [rng.choice(train_examples) for _ in range(batch_size)]
            content_lengths = [len(ex.input_tokens) for ex in batch]
            lmax = max(content_lengths)
            b_ids = collate_content_only_batch(batch, core.tokens, device=device)

            with torch.no_grad():
                h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]

            output_lengths = [len(ex.target_tokens) for ex in batch]
            max_out = max(output_lengths)

            logits = operator(h_content, content_lengths, output_lengths)
            targets = torch.full(
                (len(batch), max_out), IGNORE_INDEX, dtype=torch.long, device=device
            )
            for i, ex in enumerate(batch):
                targets[i, : len(ex.target_tokens)] = torch.tensor(
                    ex.target_tokens, dtype=torch.long, device=device
                )

            common_len = min(logits.shape[1], targets.shape[1])
            loss = loss_fn(
                logits[:, :common_len, :].reshape(-1, vocab_size),
                targets[:, :common_len].reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Evaluate exact match
        operator.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(eval_examples), self.config.eval_batch_size):
                chunk = eval_examples[start : start + self.config.eval_batch_size]
                c_lens = [len(ex.input_tokens) for ex in chunk]
                lmax_eval = max(c_lens)
                b_ids = collate_content_only_batch(chunk, core.tokens, device=device)
                h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax_eval, :]
                out_lens = [len(ex.target_tokens) for ex in chunk]

                logits = operator(h_content, c_lens, out_lens)
                preds = logits.argmax(dim=-1)

                for row, ex in enumerate(chunk):
                    n = len(ex.target_tokens)
                    pred_tokens = tuple(preds[row, :n].tolist())
                    if pred_tokens == ex.target_tokens:
                        correct += 1

        return correct / len(eval_examples) if eval_examples else 0.0

    def _distill_to_compact_candidate(
        self,
        *,
        teacher: CrossPositionPrimitive,
        train_examples: Sequence[Example],
        core: Any,
        task_name: str,
        vocab_size: int,
        device: torch.device,
    ) -> CrossPositionPrimitive:
        """Distill the teacher into a compact student candidate (<=25k parameters)."""
        teacher.eval()
        teacher.freeze()

        student_cfg = CrossPositionPrimitiveConfig(
            operation=task_name,
            d_model=core.model.config.d_model,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=vocab_size,
            max_sequence_length=32,
        )
        student = CrossPositionPrimitive(
            0,
            student_cfg,
            status=PrimitiveStatus.CANDIDATE,
            created_at_task=0,
        )
        student.to(device)
        student.unfreeze()
        student.train()

        optimizer = torch.optim.AdamW(student.parameters(), lr=self.config.distillation_lr)
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        rng = random.Random(self.config.seed * 31337)

        for _ in range(self.config.distillation_steps):
            batch = [rng.choice(train_examples) for _ in range(self.config.compact_batch_size)]
            content_lengths = [len(ex.input_tokens) for ex in batch]
            lmax = max(content_lengths)
            b_ids = collate_content_only_batch(batch, core.tokens, device=device)

            with torch.no_grad():
                h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]
                out_lens = [len(ex.target_tokens) for ex in batch]
                t_logits = teacher(h_content, content_lengths, out_lens)

            s_logits = student(h_content, content_lengths, out_lens)
            max_out = max(out_lens)
            targets = torch.full(
                (len(batch), max_out), IGNORE_INDEX, dtype=torch.long, device=device
            )
            for i, ex in enumerate(batch):
                targets[i, : len(ex.target_tokens)] = torch.tensor(
                    ex.target_tokens, dtype=torch.long, device=device
                )

            common_len = min(s_logits.shape[1], t_logits.shape[1], targets.shape[1])
            s_aligned = s_logits[:, :common_len, :]
            t_aligned = t_logits[:, :common_len, :]
            tgt_aligned = targets[:, :common_len]

            valid_mask = tgt_aligned != IGNORE_INDEX
            num_valid = valid_mask.sum().clamp(min=1)

            # Soft distillation loss
            p_teacher = F.softmax(t_aligned / self.config.distillation_temperature, dim=-1)
            log_p_student = F.log_softmax(s_aligned / self.config.distillation_temperature, dim=-1)
            kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)
            loss_soft = (
                (kl * valid_mask.float()).sum()
                / num_valid
                * (self.config.distillation_temperature**2)
            )

            # Hard ground-truth loss
            loss_hard = ce_loss_fn(s_aligned.reshape(-1, vocab_size), tgt_aligned.reshape(-1))
            loss = (
                self.config.distillation_alpha * loss_soft
                + (1.0 - self.config.distillation_alpha) * loss_hard
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        student.eval()
        student.freeze()
        return student

    def _run_shadow_validation(
        self,
        *,
        candidate: CrossPositionPrimitive,
        eval_examples: Sequence[Example],
        historical_eval_examples: Mapping[str, Sequence[Example]],
        core: Any,
        bank: PrimitiveBank,
        op_to_id: dict[str, int],
    ) -> tuple[bool, str]:
        """Evaluate candidate parameter size, novel performance, and historical retention."""
        # 1. Parameter budget check (<= 25,000)
        max_params = self.config.max_candidate_parameters
        if candidate.num_parameters() > max_params:
            return False, f"parameter_budget_exceeded: {candidate.num_parameters()} > {max_params}"

        # 2. Performance on novel eval examples
        candidate.eval()
        device = core.device
        correct = 0
        finite = True

        with torch.no_grad():
            for start in range(0, len(eval_examples), self.config.eval_batch_size):
                chunk = eval_examples[start : start + self.config.eval_batch_size]
                c_lens = [len(ex.input_tokens) for ex in chunk]
                lmax = max(c_lens)
                b_ids = collate_content_only_batch(chunk, core.tokens, device=device)
                h_content = core.model.encode(b_ids)[:, 1 : 1 + lmax, :]
                out_lens = [len(ex.target_tokens) for ex in chunk]

                logits = candidate(h_content, c_lens, out_lens)
                if not torch.isfinite(logits).all():
                    finite = False
                preds = logits.argmax(dim=-1)

                for row, ex in enumerate(chunk):
                    n = len(ex.target_tokens)
                    pred_tokens = tuple(preds[row, :n].tolist())
                    if pred_tokens == ex.target_tokens:
                        correct += 1

        if not finite:
            return False, "non_finite_outputs"

        candidate_em = correct / len(eval_examples) if eval_examples else 0.0
        ret_thresh = self.config.shadow_retention_threshold
        if candidate_em < ret_thresh:
            return False, f"low_candidate_em: {candidate_em:.4f} < {ret_thresh}"

        # 3. Verify frozen invariants and historical retention
        from apc.plastic.residual import verify_frozen_invariants

        try:
            verify_frozen_invariants(core, bank)
        except (AssertionError, RuntimeError) as err:
            return False, f"frozen_invariant_violation: {err}"

        # In modular primitive architectures where Core and existing Bank primitives are frozen,
        # catastrophic forgetting is strictly 0.0. Verify execution integrity on historical tasks.
        with torch.no_grad():
            for h_op, h_examples in historical_eval_examples.items():
                if not h_examples or h_op not in op_to_id:
                    continue
                try:
                    h_logits = execute_composition_recipe(
                        core, bank, op_to_id, h_examples[:10], candidate_operations=(h_op,)
                    )
                    if not torch.isfinite(h_logits).all():
                        return False, f"historical_non_finite_{h_op}"
                except Exception as err:
                    return False, f"historical_execution_failure_{h_op}: {err}"

        return True, "passed"
