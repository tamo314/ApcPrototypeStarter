"""Unit tests for routing ranking losses (Phase B Task B-C005R1)."""

from __future__ import annotations

import pytest
import torch

from apc.primitives.routing_losses import (
    CombinedRoutingLoss,
    MarginRankingLoss,
    MultiNegativeRankingLoss,
    RankingLossConfig,
)


def test_margin_ranking_loss_mathematical_properties() -> None:
    loss_fn = MarginRankingLoss(margin=2.0)
    pos = torch.tensor([5.0])
    neg = torch.tensor([1.0])
    # diff = 2.0 - 5.0 + 1.0 = -2.0 -> clamp -> 0.0
    assert loss_fn(pos, neg).item() == 0.0

    pos_violating = torch.tensor([2.0])
    neg_violating = torch.tensor([3.0])
    # diff = 2.0 - 2.0 + 3.0 = 3.0
    assert loss_fn(pos_violating, neg_violating).item() == 3.0


def test_margin_ranking_loss_gradient_direction() -> None:
    """Positive score must receive negative loss gradient (to increase score),
    and negative score must receive positive loss gradient (to decrease score)."""
    pos = torch.tensor([2.0], requires_grad=True)
    neg = torch.tensor([3.0], requires_grad=True)
    loss_fn = MarginRankingLoss(margin=2.0)
    loss = loss_fn(pos, neg)
    loss.backward()

    assert pos.grad is not None
    assert neg.grad is not None
    assert pos.grad.item() < 0  # increasing pos decreases loss
    assert neg.grad.item() > 0  # increasing neg increases loss


def test_multi_negative_ranking_loss() -> None:
    loss_fn = MultiNegativeRankingLoss(margin=3.0)
    pos = torch.tensor([[4.0], [2.0]])  # (2, 1)
    neg = torch.tensor([[1.0, 0.0], [3.0, 4.0]])  # (2, 2)
    # Row 0: margin 3.0 - 4.0 + [1.0, 0.0] = [0.0, -1.0] -> clamp -> [0.0, 0.0] -> mean = 0.0
    # Row 1: margin 3.0 - 2.0 + [3.0, 4.0] = [4.0, 5.0] -> clamp -> [4.0, 5.0] -> mean = 4.5
    # Overall mean = (0.0 + 4.5) / 2 = 2.25
    loss = loss_fn(pos, neg)
    assert pytest.approx(loss.item(), abs=1e-5) == 2.25


def test_combined_routing_loss() -> None:
    config = RankingLossConfig(margin=2.0, beta=0.5)
    loss_fn = CombinedRoutingLoss(config)

    logits = torch.tensor([[3.0, 1.0, 0.5]], requires_grad=True)
    targets = torch.tensor([0])
    hard_neg = torch.tensor([[2.5, 2.0]])

    total_loss, metrics = loss_fn(logits, targets, hard_neg)
    assert total_loss.item() > 0.0
    assert "loss_total" in metrics
    assert "loss_ce" in metrics
    assert "loss_rank" in metrics
    assert metrics["loss_rank"] > 0.0

    total_loss.backward()
    assert logits.grad is not None
