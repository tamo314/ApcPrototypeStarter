"""Routing Ranking Losses (Phase B Task B-C005R1).

Implements explicit ranking and margin objectives for primitive routing
under hard-negative competition, per:
- `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md` section R1.3
- `docs/design-docs/HARD_NEGATIVE_ROUTING_PHASE_B.md`

Objectives:
1. MarginRankingLoss:
   L = max(0, margin - s_pos + s_neg)
2. MultiNegativeRankingLoss:
   L = mean_j max(0, margin - s_pos + s_neg_j)
3. CombinedRoutingLoss:
   L = L_ce + beta * L_rank
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RANKING_MARGIN: Final[float] = 3.0
DEFAULT_RANKING_BETA: Final[float] = 1.0


@dataclass(frozen=True)
class RankingLossConfig:
    """Configuration for routing ranking objectives."""

    margin: float = DEFAULT_RANKING_MARGIN
    beta: float = DEFAULT_RANKING_BETA
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {self.reduction!r}")


class MarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss between positive and negative candidate scores."""

    def __init__(self, margin: float = DEFAULT_RANKING_MARGIN, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hinge margin loss: max(0, margin - pos + neg).

        Args:
            pos_scores: Tensor of positive candidate scores, shape (...)
            neg_scores: Tensor of negative candidate scores, same shape (...)

        Returns:
            Computed ranking loss.
        """
        diff = self.margin - pos_scores + neg_scores
        loss = torch.clamp(diff, min=0.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiNegativeRankingLoss(nn.Module):
    """Multi-negative margin ranking loss against multiple hard-negative competitors."""

    def __init__(self, margin: float = DEFAULT_RANKING_MARGIN, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-negative ranking loss across a batch of candidates.

        Args:
            pos_scores: Tensor of shape (batch_size, 1) or (batch_size,)
            neg_scores: Tensor of shape (batch_size, num_negatives)

        Returns:
            Computed ranking loss averaged or summed over negatives and batch.
        """
        if pos_scores.ndim == 1:
            pos_scores = pos_scores.unsqueeze(-1)
        # pos_scores: (B, 1), neg_scores: (B, M)
        diff = self.margin - pos_scores + neg_scores
        hinge = torch.clamp(diff, min=0.0)
        per_example_loss = hinge.mean(dim=-1)
        if self.reduction == "mean":
            return per_example_loss.mean()
        if self.reduction == "sum":
            return per_example_loss.sum()
        return per_example_loss


class CombinedRoutingLoss(nn.Module):
    """Combines cross-entropy classification loss with hard-negative ranking loss."""

    def __init__(
        self,
        config: RankingLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RankingLossConfig()
        self.margin_loss = MultiNegativeRankingLoss(
            margin=self.config.margin,
            reduction=self.config.reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hard_neg_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: (B, C) classification logits over candidate keys.
            targets: (B,) ground truth class indices.
            hard_neg_scores: Optional (B, M) scores for external hard-negative candidates.

        Returns:
            total_loss: scalar tensor
            metrics: dict of individual loss values
        """
        ce_loss = F.cross_entropy(logits, targets)
        if hard_neg_scores is None or hard_neg_scores.numel() == 0 or self.config.beta <= 0.0:
            return ce_loss, {
                "loss_total": ce_loss.item(),
                "loss_ce": ce_loss.item(),
                "loss_rank": 0.0,
            }

        # Gather positive scores corresponding to targets
        pos_scores = logits.gather(dim=1, index=targets.unsqueeze(1))
        rank_loss = self.margin_loss(pos_scores, hard_neg_scores)
        total_loss = ce_loss + self.config.beta * rank_loss

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_ce": ce_loss.item(),
            "loss_rank": rank_loss.item(),
        }
