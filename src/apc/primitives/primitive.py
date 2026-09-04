"""Heterogeneous compact primitives and base record.

Phase A.1 Post-Diagnostic: Branch B Integration.

Provides:
- `PrimitiveBase`: Common base module providing record bookkeeping, freeze/unfreeze,
  and call instrumentation for all primitive classes.
- `PointwisePrimitive` (aliased as `Primitive`): Legacy low-rank residual transform
  $P(h) = h + s(h) * B A h$.
- `CrossPositionPrimitive`: Primitive-scale (~18k-21k parameter) single-layer
  cross-attention operator over content states, conditioned on typed arguments.
- `ShiftRelativePrimitive`: Primitive-scale (18,282 parameter) cross-attention operator
  equipped with modular relative-position attention bias for cyclic shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

DEFAULT_ARG_DIM: int = 32
DEFAULT_MAX_SEQUENCE_LENGTH: int = 32

__all__ = [
    "CrossPositionPrimitive",
    "CrossPositionPrimitiveConfig",
    "PointwisePrimitive",
    "Primitive",
    "PrimitiveBase",
    "PrimitiveConfig",
    "PrimitiveStatus",
    "ShiftRelativePrimitive",
    "ShiftRelativePrimitiveConfig",
]


class PrimitiveStatus(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """Lifecycle state of a primitive record."""

    CANDIDATE = "candidate"
    STABLE = "stable"
    ARCHIVED = "archived"


class PrimitiveBase(nn.Module):
    """Abstract base module for all APC primitives.

    Owns record metadata, freeze state, usage tracking, and execution-instrumentation
    counters required for strict sparse accounting.
    """

    def __init__(
        self,
        primitive_id: int,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.primitive_id = primitive_id
        self.status = status
        self.created_at_task = created_at_task
        self.enabled = enabled
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

        self.usage_count = 0
        self.forward_call_count = 0
        self.utility_ema = 0.0
        self.stability_score = 0.0

    def reset_forward_call_count(self) -> None:
        """Reset the execution-instrumentation counter used by sparse tests."""
        self.forward_call_count = 0

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def freeze(self) -> None:
        """Set `requires_grad = False` on all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Set `requires_grad = True` on all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.parameters())

    def record_usage(self) -> None:
        self.usage_count += 1

    def update_utility(self, value: float, *, decay: float = 0.99) -> None:
        """Exponential moving average update of `utility_ema`."""
        self.utility_ema = decay * self.utility_ema + (1.0 - decay) * value


# ---------------------------------------------------------------------------
# Pointwise Primitive (Legacy / Phase A / Parameter-Free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveConfig:
    """Explicit, serializable configuration for a `PointwisePrimitive`."""

    d_model: int
    rank: int = 8

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


class PointwisePrimitive(PrimitiveBase):
    """A single low-rank residual transform $h + gate * B(A(h))$."""

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
        super().__init__(
            primitive_id,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.config = config
        self.a_proj = nn.Linear(config.d_model, config.rank, bias=False)
        self.b_proj = nn.Linear(config.rank, config.d_model, bias=False)
        nn.init.normal_(self.a_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.b_proj.weight)

    def forward(self, h: torch.Tensor, gate: torch.Tensor | float = 1.0) -> torch.Tensor:
        if not self.enabled:
            return h
        self.forward_call_count += 1
        delta = self.b_proj(self.a_proj(h))
        return h + gate * delta


# Backward-compatibility alias
Primitive = PointwisePrimitive


# ---------------------------------------------------------------------------
# Cross-Position Primitive (Branch B / Parameterized Compact Operator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossPositionPrimitiveConfig:
    """Configuration for `CrossPositionPrimitive`."""

    operation: str
    d_model: int = 192
    d_operator: int = 32
    n_head: int = 4
    d_operator_ff: int = 64
    vocab_size: int = 10
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    arg_dim: int = 16

    def __post_init__(self) -> None:
        if self.d_operator % self.n_head != 0:
            raise ValueError(
                f"d_operator ({self.d_operator}) must be divisible by n_head ({self.n_head})"
            )


class CrossPositionPrimitive(PrimitiveBase):
    """Primitive-scale (~18k-21k params) argument-conditioned cross-attention operator.

    Reads from frozen shared content representations $h_{\\text{content}}$, querying
    content tokens via output-slot queries conditioned on the argument.
    """

    def __init__(
        self,
        primitive_id: int,
        config: CrossPositionPrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.config = config
        self.operation = config.operation
        self.d_operator = config.d_operator
        self.max_sequence_length = config.max_sequence_length

        self.content_in_proj = nn.Linear(config.d_model, config.d_operator)
        self.content_position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_operator
        )
        self.answer_query_embedding = nn.Embedding(config.max_sequence_length, config.d_operator)
        from apc.primitives.conditioning import default_argument_encoder

        self.arg_encoder = default_argument_encoder(
            config.operation,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            arg_dim=config.arg_dim,
        )
        self.arg_proj = nn.Linear(config.arg_dim, config.d_operator)

        self.cross_attn = nn.MultiheadAttention(
            config.d_operator, config.n_head, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config.d_operator)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_operator, config.d_operator_ff),
            nn.GELU(),
            nn.Linear(config.d_operator_ff, config.d_operator),
        )
        self.ffn_norm = nn.LayerNorm(config.d_operator)
        self.readout = nn.Linear(config.d_operator, config.vocab_size)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None,
    ) -> torch.Tensor:
        """Apply cross-attention operator over content features.

        Returns:
            Logits of shape `[batch, max(output_lengths), vocab_size]`.
        """
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        if not self.enabled:
            return content_features.new_zeros(batch, out_max, self.config.vocab_size)

        self.forward_call_count += 1

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        kv = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )
        content_lengths_t = torch.tensor(content_lengths, device=device).unsqueeze(1)
        content_pad_mask = content_position_ids >= content_lengths_t

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)

        query = query_slots + arg_token

        attn_out, _ = self.cross_attn(
            query, kv, kv, key_padding_mask=content_pad_mask, need_weights=False
        )
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)


# ---------------------------------------------------------------------------
# Shift Relative Primitive (Branch B / Inductive Bias for Cyclic Shifts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftRelativePrimitiveConfig:
    """Configuration for `ShiftRelativePrimitive`."""

    d_model: int = 192
    d_operator: int = 32
    n_head: int = 4
    d_operator_ff: int = 64
    vocab_size: int = 10
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    arg_dim: int = 16

    def __post_init__(self) -> None:
        if self.d_operator % self.n_head != 0:
            raise ValueError(
                f"d_operator ({self.d_operator}) must be divisible by n_head ({self.n_head})"
            )


class ShiftRelativePrimitive(PrimitiveBase):
    """Primitive-scale (18,282 params) SHIFT operator with modular relative-position bias."""

    def __init__(
        self,
        primitive_id: int,
        config: ShiftRelativePrimitiveConfig | None = None,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        cfg = config or ShiftRelativePrimitiveConfig()
        self.config = cfg
        self.operation = "SHIFT"
        self.d_operator = cfg.d_operator
        self.n_head = cfg.n_head
        self.max_sequence_length = cfg.max_sequence_length

        self.content_in_proj = nn.Linear(cfg.d_model, cfg.d_operator)
        self.content_position_embedding = nn.Embedding(cfg.max_sequence_length, cfg.d_operator)
        self.answer_query_embedding = nn.Embedding(cfg.max_sequence_length, cfg.d_operator)
        from apc.primitives.conditioning import default_argument_encoder

        self.arg_encoder = default_argument_encoder(
            "SHIFT",
            vocab_size=cfg.vocab_size,
            max_sequence_length=cfg.max_sequence_length,
            arg_dim=cfg.arg_dim,
        )
        self.arg_proj = nn.Linear(cfg.arg_dim, cfg.d_operator)

        # Learned modular relative position bias: [max_sequence_length, n_head]
        self.rel_pos_bias = nn.Embedding(cfg.max_sequence_length, cfg.n_head)
        nn.init.zeros_(self.rel_pos_bias.weight)

        self.cross_attn = nn.MultiheadAttention(cfg.d_operator, cfg.n_head, batch_first=True)
        self.attn_norm = nn.LayerNorm(cfg.d_operator)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_operator, cfg.d_operator_ff),
            nn.GELU(),
            nn.Linear(cfg.d_operator_ff, cfg.d_operator),
        )
        self.ffn_norm = nn.LayerNorm(cfg.d_operator)
        self.readout = nn.Linear(cfg.d_operator, cfg.vocab_size)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None,
    ) -> torch.Tensor:
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        if not self.enabled:
            return content_features.new_zeros(batch, out_max, self.config.vocab_size)

        self.forward_call_count += 1

        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        kv = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
            shifts = torch.zeros(batch, 1, 1, dtype=torch.long, device=device)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)
            shift_list = [int(v) for v in argument_values]
            shifts = torch.tensor(shift_list, device=device).view(batch, 1, 1)

        query = query_slots + arg_token

        # Vectorized modular relative position displacement:
        # disp[b, i, j] = (j - i - a) % L
        c_lens = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
        s_idx = torch.arange(out_max, device=device).view(1, out_max, 1)
        p_idx = torch.arange(lmax, device=device).view(1, 1, lmax)
        disp = (p_idx - s_idx - shifts) % c_lens
        attn_bias = self.rel_pos_bias(disp).permute(0, 3, 1, 2)

        pad_mask = (p_idx >= c_lens).unsqueeze(1).expand(-1, self.n_head, out_max, -1)
        attn_mask = torch.where(
            pad_mask,
            torch.tensor(float("-inf"), device=device),
            attn_bias,
        ).reshape(batch * self.n_head, out_max, lmax)

        attn_out, _ = self.cross_attn(query, kv, kv, attn_mask=attn_mask, need_weights=False)
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)
