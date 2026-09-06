"""Argument Compatibility Scoring for Parameterized Primitives (Task B-C005R1).

Implements factorized PrimitiveCall candidate scoring per:
- `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md` section R1.4
- `docs/design-docs/HARD_NEGATIVE_ROUTING_PHASE_B.md`

Score Formulation:
    score(PrimitiveCall) = score_family(z_task, primitive_key)
                         + lambda * score_args(z_task, call.arguments)

Separates physical primitive family retrieval from argument resolution,
preventing key explosion while resolving L4 same-family confusable candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from apc.environments.generator import Example

DEFAULT_ARG_LAMBDA: Final[float] = 2.0
DEFAULT_ARG_VOCAB_SIZE: Final[int] = 32
DEFAULT_ARG_LR: Final[float] = 0.005
DEFAULT_ARG_STEPS: Final[int] = 200

PARAMETERIZED_OPERATIONS: Final[tuple[str, ...]] = ("SHIFT", "SELECT", "COUNT", "BIND")


@dataclass(frozen=True)
class ArgumentScorerConfig:
    """Configuration for ArgumentScorer."""

    d_model: int = 128
    arg_vocab_size: int = DEFAULT_ARG_VOCAB_SIZE
    lambda_weight: float = DEFAULT_ARG_LAMBDA
    lr: float = DEFAULT_ARG_LR
    steps: int = DEFAULT_ARG_STEPS

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.arg_vocab_size < 1:
            raise ValueError(f"arg_vocab_size must be positive, got {self.arg_vocab_size}")
        if self.lambda_weight < 0.0:
            raise ValueError(f"lambda_weight must be non-negative, got {self.lambda_weight}")


def extract_raw_argument_values(operation: str, arguments: dict[str, Any]) -> list[int]:
    """Extract integer argument value(s) from an arguments mapping."""
    if operation == "SHIFT":
        val = arguments.get("amount", 0)
        return [int(val)] if isinstance(val, (int, float)) else [0]
    if operation == "COUNT":
        val = arguments.get("target", 0)
        return [int(val)] if isinstance(val, (int, float)) else [0]
    if operation == "BIND":
        val = arguments.get("query_key", 0)
        return [int(val)] if isinstance(val, (int, float)) else [0]
    if operation == "SELECT":
        indices = arguments.get("indices", [0])
        if isinstance(indices, (list, tuple)):
            return [int(i) for i in indices]
        return [int(indices)]
    return []


class ArgumentScorer(nn.Module):
    """Predicts argument compatibility scores from z_task for parameterized calls."""

    def __init__(self, config: ArgumentScorerConfig) -> None:
        super().__init__()
        self.config = config
        # Dedicated linear heads per parameterized family to avoid interference
        self.heads = nn.ModuleDict({
            op: nn.Linear(config.d_model, config.arg_vocab_size)
            for op in PARAMETERIZED_OPERATIONS
        })

    def forward(
        self,
        z_task: torch.Tensor,
        operation: str,
        arguments: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Compute compatibility score for candidate arguments.

        Args:
            z_task: (B, d_model) task representations.
            operation: Canonical operation name.
            arguments: Candidate arguments mapping.

        Returns:
            scores: (B,) compatibility scores for this candidate call.
        """
        batch_size = z_task.shape[0]
        device = z_task.device

        if operation not in self.heads or arguments is None:
            # Parameter-free primitives receive zero argument score offset
            return torch.zeros(batch_size, device=device)

        raw_vals = extract_raw_argument_values(operation, arguments)
        if not raw_vals:
            return torch.zeros(batch_size, device=device)

        logits = self.heads[operation](z_task)  # (B, vocab_size)
        # SELECT is the one set-valued operation (see PARAMETERIZED_OPERATIONS
        # / `train_on_examples` below): it is fit with independent multi-hot
        # BCEWithLogitsLoss targets, so its members must be scored with
        # independent per-index sigmoids here too. Reading a shared softmax
        # (correct for the other, single-label operations) would force every
        # simultaneously-true index to compete for one unit of probability
        # mass, diluting each one's score as the set grows -- the
        # `ARGUMENT_ENCODING_FAILURE` train/inference mismatch diagnosed in
        # `runs/phase_b_b2_second_diagnostic/l4_argument_breakdown.json`
        # (ADR-0081) and repaired by Task B-C005R3-007 (ADR-0088).
        probs = torch.sigmoid(logits) if operation == "SELECT" else F.softmax(logits, dim=-1)

        scores_per_val = []
        for val in raw_vals:
            clamped_val = min(max(0, val), self.config.arg_vocab_size - 1)
            val_idx = torch.tensor([clamped_val], device=device).expand(batch_size, 1)
            # Centered normalized compatibility score in [-1.0, 1.0]:
            # High confidence match -> ~ +1.0, low confidence -> ~ -1.0
            p = probs.gather(dim=-1, index=val_idx).squeeze(-1)
            scores_per_val.append(2.0 * (p - 0.5))

        # Average compatibility score across argument elements (e.g. for SELECT indices)
        return torch.stack(scores_per_val, dim=-1).mean(dim=-1)

    def train_on_examples(
        self,
        z_task_by_op: Mapping[str, torch.Tensor],
        examples_by_op: Mapping[str, Sequence[Example]],
        *,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        """Train argument scoring heads on development task examples.

        Args:
            z_task_by_op: dict mapping operation name to (N, d_model) tensor.
            examples_by_op: dict mapping operation name to Sequence[Example].
            device: Optional torch device.

        Returns:
            final_losses: dict mapping operation to final training loss.
        """
        target_device = device or next(self.parameters()).device
        self.to(target_device)
        self.train()

        losses: dict[str, float] = {}
        for op in PARAMETERIZED_OPERATIONS:
            if op not in z_task_by_op or op not in examples_by_op:
                continue
            z_batch = z_task_by_op[op].to(target_device)
            examples = examples_by_op[op]
            if len(examples) == 0:
                continue

            head = self.heads[op]
            optimizer = torch.optim.AdamW(head.parameters(), lr=self.config.lr, weight_decay=1e-4)

            loss_fn: nn.Module
            # Build targets
            if op == "SELECT":
                # Multi-hot targets for SELECT indices
                targets = torch.zeros(
                    len(examples), self.config.arg_vocab_size, device=target_device
                )
                for i, ex in enumerate(examples):
                    assert ex.task_spec is not None
                    for idx in ex.task_spec.steps[0].arguments.get("indices", []):
                        if 0 <= idx < self.config.arg_vocab_size:
                            targets[i, idx] = 1.0
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                target_list = []
                for ex in examples:
                    assert ex.task_spec is not None
                    vals = extract_raw_argument_values(op, ex.task_spec.steps[0].arguments)
                    val = vals[0] if vals else 0
                    target_list.append(min(max(0, val), self.config.arg_vocab_size - 1))
                targets = torch.tensor(target_list, dtype=torch.long, device=target_device)
                loss_fn = nn.CrossEntropyLoss()

            final_loss = 0.0
            for _ in range(self.config.steps):
                optimizer.zero_grad(set_to_none=True)
                logits = head(z_batch)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

            losses[op] = final_loss

        self.eval()
        return losses
