"""Low-rank residual primitive (Phase A Milestone A3 / Task 004).

Implements the primitive representation from
`docs/design-docs/ARCHITECTURE.md` section 4:

    P_i(h) = h + s_i(h) * B_i A_i h

`A: d_model -> rank` and `B: rank -> d_model` are the only learnable
weights; the gate `s_i(h)` is supplied by the caller (the top-k router,
Task 005) rather than computed here. Each primitive also carries the
bookkeeping fields the architecture doc assigns to a "primitive record":
status, usage count, creation step, utility EMA, and stability score.

No routing, no dynamic expansion, no consolidation: this module only
defines the primitive unit and its own accounting/freeze behavior. The
bank that owns a collection of these lives in `apc.primitives.bank`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PrimitiveConfig:
    """Explicit, serializable configuration for a single `Primitive`."""

    d_model: int
    rank: int = 8

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


class PrimitiveStatus(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """Lifecycle state of a primitive record.

    CANDIDATE: newly created or produced by consolidation, not yet
        promoted to the stable bank and not yet frozen by default.
    STABLE: promoted, persistent, frozen in Phase A per ADR-0004.
    ARCHIVED: retired (e.g. superseded by a merge); excluded from
        persistent/active accounting but kept for audit.
    """

    CANDIDATE = "candidate"
    STABLE = "stable"
    ARCHIVED = "archived"


class Primitive(nn.Module):
    """A single low-rank residual transform plus its bookkeeping record."""

    def __init__(
        self,
        primitive_id: int,
        config: PrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.primitive_id = primitive_id
        self.config = config
        self.status = status
        self.created_at_task = created_at_task
        self.enabled = enabled
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

        self.usage_count = 0
        self.utility_ema = 0.0
        self.stability_score = 0.0

        self.a_proj = nn.Linear(config.d_model, config.rank, bias=False)
        self.b_proj = nn.Linear(config.rank, config.d_model, bias=False)
        nn.init.normal_(self.a_proj.weight, mean=0.0, std=0.02)
        # B starts at zero so a freshly created primitive is the identity
        # function until trained, regardless of A's random init.
        nn.init.zeros_(self.b_proj.weight)

    def forward(self, h: torch.Tensor, gate: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Apply `h + gate * B(A(h))`, or return `h` unchanged if disabled.

        `gate` is `s_i(h)` from the architecture doc: a scalar or a tensor
        broadcastable against `h` except in the last (`d_model`) axis. It
        is supplied by the caller because the router does not exist yet
        (Task 005).
        """
        if not self.enabled:
            return h
        delta = self.b_proj(self.a_proj(h))
        return h + gate * delta

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def freeze(self) -> None:
        """Set `requires_grad = False` on `A` and `B`."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Set `requires_grad = True` on `A` and `B`."""
        for p in self.parameters():
            p.requires_grad_(True)

    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.parameters())

    def record_usage(self) -> None:
        self.usage_count += 1

    def update_utility(self, value: float, *, decay: float = 0.99) -> None:
        """Exponential moving average update of `utility_ema`."""
        self.utility_ema = decay * self.utility_ema + (1.0 - decay) * value
