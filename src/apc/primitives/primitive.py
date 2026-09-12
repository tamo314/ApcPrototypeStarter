"""Heterogeneous compact primitives and base record.

Phase A.1 Post-Diagnostic: Branch B Integration.

Provides:
- `PrimitiveBase`: Common base module providing record bookkeeping, freeze/unfreeze,
  and call instrumentation for all primitive classes.
- `PointwisePrimitive` (aliased as `Primitive`): Legacy low-rank residual transform
  $P(h) = h + s(h) * B A h$.
- `CrossPositionPrimitive`: Primitive-scale (~18k-21k parameter) single-layer
  cross-attention operator over content states, conditioned on typed arguments.
- `CrossPositionLengthBiasPrimitive`: `CrossPositionPrimitive` plus a small
  (192-parameter) length-conditioned additive position-bias term added to the
  attention scores before softmax (Task B-C005REC-004D).
- `ShiftRelativePrimitive`: Primitive-scale (18,282 parameter) cross-attention operator
  equipped with modular relative-position attention bias for cyclic shifts.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_ARG_DIM: int = 32
DEFAULT_MAX_SEQUENCE_LENGTH: int = 32

__all__ = [
    "CDDPCAPrimitive",
    "CDDPCAPrimitiveConfig",
    "ContentDecoupledDiscretePositionalCrossAttentionPrimitive",
    "ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig",
    "CrossPositionLengthBiasPrimitive",
    "CrossPositionLengthBiasPrimitiveConfig",
    "CrossPositionPrimitive",
    "CrossPositionPrimitiveConfig",
    "PointwisePrimitive",
    "Primitive",
    "PrimitiveBase",
    "PrimitiveConfig",
    "PrimitiveStatus",
    "ReverseRelativePrimitive",
    "ReverseRelativePrimitiveConfig",
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
        from apc.environments.operations import get_operation

        try:
            op = get_operation(config.operation)
            has_args = bool(op.required_argument_names)
        except KeyError:
            has_args = False

        if has_args:
            from apc.primitives.conditioning import default_argument_encoder

            self.arg_encoder: nn.Module | None = default_argument_encoder(
                config.operation,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_sequence_length,
                arg_dim=config.arg_dim,
            )
            self.arg_proj: nn.Linear | None = nn.Linear(config.arg_dim, config.d_operator)
        else:
            self.arg_encoder = None
            self.arg_proj = None

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
        argument_values: Sequence[Any] | None = None,
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

        if argument_values is None or self.arg_encoder is None or self.arg_proj is None:
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
# Cross-Position Length-Bias Primitive (B-C005REC-004D / MIRROR_HALVES repair)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossPositionLengthBiasPrimitiveConfig(CrossPositionPrimitiveConfig):
    """`CrossPositionPrimitiveConfig` plus the length-conditioned position-bias
    head's own two hyperparameters. `length_ref` is the model's fixed legal
    max content length (recorded once at protocol-lock time, not derived from
    a batch's own max length)."""

    bias_hidden_dim: int = 32
    length_ref: int = DEFAULT_MAX_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bias_hidden_dim < 1:
            raise ValueError(f"bias_hidden_dim must be >= 1, got {self.bias_hidden_dim}")
        if self.length_ref < 1:
            raise ValueError(f"length_ref must be >= 1, got {self.length_ref}")


class CrossPositionLengthBiasPrimitive(CrossPositionPrimitive):
    """`CrossPositionPrimitive` with a small additive position-bias term.

    Adds `score_new = score_existing + b_theta(i, j, n)` before softmax, where
    `b_theta` is a 2-layer MLP (4 -> `bias_hidden_dim` -> 1, no output bias,
    shared across attention heads) over generic coordinates only:
    `phi(i, j, n) = [i/d, j/d, (j-i)/d, n/length_ref]` with `d = max(n-1, 1)`,
    `i` the output/query content position, `j` the input/key content position,
    and `n` the real (unpadded) content length -- never the teacher position
    map, a half-index label, or any operation-specific lookup.

    192 new parameters total (`4*bias_hidden_dim + bias_hidden_dim +
    bias_hidden_dim` = 128+32+32 for the default `bias_hidden_dim=32`). The
    output layer is initialized to exactly zero so a freshly-constructed
    instance is an exact no-op versus `CrossPositionPrimitive` (the hidden
    layer keeps its ordinary nonzero random init so gradients reach it once
    the output layer's own weight moves off zero -- zeroing both layers would
    make the whole branch permanently untrainable).

    Masking is folded into a single additive float `attn_mask` (padded keys
    get `-inf`, valid keys get `b_theta`) instead of `CrossPositionPrimitive`'s
    boolean `key_padding_mask`, matching the precedent already established by
    `ShiftRelativePrimitive`/`ReverseRelativePrimitive`. With the bias output
    at zero this produces byte-identical masking semantics to the parent
    class (a `-inf`/`0` additive mask is exactly what PyTorch's own bool
    `key_padding_mask` lowers to internally).
    """

    def __init__(
        self,
        primitive_id: int,
        config: CrossPositionLengthBiasPrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            primitive_id,
            config,
            status=status,
            created_at_task=created_at_task,
            enabled=enabled,
            metadata=metadata,
        )
        self.n_head = config.n_head
        self.length_ref = config.length_ref
        self.position_bias_hidden = nn.Linear(4, config.bias_hidden_dim)
        self.position_bias_out = nn.Linear(config.bias_hidden_dim, 1, bias=False)
        nn.init.zeros_(self.position_bias_out.weight)

    def _position_bias(
        self, content_lengths: Sequence[int], out_max: int, lmax: int, device: torch.device
    ) -> torch.Tensor:
        """Returns `b_theta(i, j, n)` of shape `[batch, out_max, lmax]`."""
        batch = len(content_lengths)
        dtype = self.position_bias_hidden.weight.dtype
        c_lens = torch.tensor(content_lengths, device=device, dtype=dtype).view(batch, 1, 1)
        s_idx = torch.arange(out_max, device=device, dtype=dtype).view(1, out_max, 1)
        p_idx = torch.arange(lmax, device=device, dtype=dtype).view(1, 1, lmax)
        d_denom = torch.clamp(c_lens - 1.0, min=1.0)
        shape = (batch, out_max, lmax)
        phi = torch.stack(
            [
                s_idx.expand(shape) / d_denom,
                p_idx.expand(shape) / d_denom,
                (p_idx - s_idx).expand(shape) / d_denom,
                (c_lens / float(self.length_ref)).expand(shape),
            ],
            dim=-1,
        )
        hidden = F.relu(self.position_bias_hidden(phi))
        return self.position_bias_out(hidden).squeeze(-1)

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
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

        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_slots = self.answer_query_embedding(query_ids)

        if argument_values is None or self.arg_encoder is None or self.arg_proj is None:
            arg_token = content_features.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)

        query = query_slots + arg_token

        bias = self._position_bias(content_lengths, out_max, lmax, device)
        bias = bias.unsqueeze(1).expand(batch, self.n_head, out_max, lmax)

        content_lengths_t = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
        p_idx_long = torch.arange(lmax, device=device).view(1, 1, lmax)
        pad_mask = (p_idx_long >= content_lengths_t).unsqueeze(1).expand(
            batch, self.n_head, out_max, lmax
        )
        attn_mask = torch.where(
            pad_mask, torch.tensor(float("-inf"), device=device), bias
        ).reshape(batch * self.n_head, out_max, lmax)

        attn_out, _ = self.cross_attn(query, kv, kv, attn_mask=attn_mask, need_weights=False)
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return self.readout(hidden)

    @contextlib.contextmanager
    def zeroed_position_bias(self) -> Iterator[None]:
        """Diagnostic-only context manager: temporarily zeros the trained
        `position_bias_out` weight (an intervention on the learned bias, NOT
        a reset to `CrossPositionPrimitive`'s architecture) so a caller can
        forward-evaluate this exact trained instance with its bias term
        switched off, then restores the original weight on exit. Never used
        during training; never mutates any other parameter."""
        original = self.position_bias_out.weight.detach().clone()
        with torch.no_grad():
            self.position_bias_out.weight.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                self.position_bias_out.weight.copy_(original)


# ---------------------------------------------------------------------------
# Content-Decoupled Discrete Positional Cross-Attention (CD-DPCA)
# Task B-C005REC-004AJ (ADR-0134 / ADR-0135)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig(
    CrossPositionPrimitiveConfig
):
    """Configuration for `ContentDecoupledDiscretePositionalCrossAttentionPrimitive` (CD-DPCA).

    Inherits all standard cross-position operator hyperparameters:
      operation: str
      d_model: int = 192
      d_operator: int = 32
      n_head: int = 4
      d_operator_ff: int = 64
      vocab_size: int = 10
      max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH  # 32
      arg_dim: int = 16
    """


# Short alias
CDDPCAPrimitiveConfig = ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig


class ContentDecoupledDiscretePositionalCrossAttentionPrimitive(PrimitiveBase):
    """Content-Decoupled Discrete Positional Cross-Attention (CD-DPCA) primitive.

    Phase B Model Bundle Recovery: Task B-C005REC-004AJ (ADR-0134 / ADR-0135).

    Replaces score-path dependence on token content with discrete query-position,
    key-position, and length representations while retaining the existing content-only
    value path, output projection, LayerNorm, FFN, readout, masks, and non-SHIFT
    bundle interface.

    Key architectural invariants:
    1. Score path is strictly content-invariant: key k(j) = E_key_pos(j) and
       query q(i, L) = E_query_pos(i) + E_length(L) + arg_token contain NO token
       content features h_content.
    2. Discrete integer coordinates: integer positions i, j in {0..max_sequence_length-1}
       and integer lengths L in {0..max_sequence_length} are mapped via standard
       nn.Embedding tables, avoiding continuous coordinate normalization and grid
       aliasing.
    3. Content flows exclusively into the value path:
       v(j) = content_in_proj(h_content(j)) + content_position_embedding(j).
    4. Relation-conditioning boundary is generic: no MIRROR-specific branch, target map,
       or oracle inputs. Parameterized operations use the standard generic arg_encoder/arg_proj
       to add an arg_token to query; parameter-free operations (like MIRROR_HALVES) pass None.
    5. Masking strictly suppresses padded key positions (j >= L) with -inf logits.
    6. Downstream modules (attn_norm, ffn, ffn_norm, readout) and tensor interfaces
       remain 100% compatible with existing non-SHIFT primitives.
    """

    def __init__(
        self,
        primitive_id: int,
        config: ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig,
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
        self.n_head = config.n_head
        self.max_sequence_length = config.max_sequence_length

        # Content-only value path
        self.content_in_proj = nn.Linear(config.d_model, config.d_operator)
        self.content_position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_operator
        )

        # Content-decoupled routing path: discrete query, key, length embeddings
        self.query_position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_operator
        )
        self.key_position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_operator
        )
        self.length_embedding = nn.Embedding(
            config.max_sequence_length + 1, config.d_operator
        )

        # Generic relation-conditioning boundary (identical to CrossPositionPrimitive)
        from apc.environments.operations import get_operation

        try:
            op = get_operation(config.operation)
            has_args = bool(op.required_argument_names)
        except KeyError:
            has_args = False

        if has_args:
            from apc.primitives.conditioning import default_argument_encoder

            self.arg_encoder: nn.Module | None = default_argument_encoder(
                config.operation,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_sequence_length,
                arg_dim=config.arg_dim,
            )
            self.arg_proj: nn.Linear | None = nn.Linear(config.arg_dim, config.d_operator)
        else:
            self.arg_encoder = None
            self.arg_proj = None

        # Cross-attention (standard multihead attention with output projection)
        self.cross_attn = nn.MultiheadAttention(
            config.d_operator, config.n_head, batch_first=True
        )

        # Downstream modules (retaining existing LayerNorm, FFN, readout)
        self.attn_norm = nn.LayerNorm(config.d_operator)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_operator, config.d_operator_ff),
            nn.GELU(),
            nn.Linear(config.d_operator_ff, config.d_operator),
        )
        self.ffn_norm = nn.LayerNorm(config.d_operator)
        self.readout = nn.Linear(config.d_operator, config.vocab_size)

    @property
    def answer_query_embedding(self) -> nn.Embedding:
        """Alias property for backward compatibility with audit code expecting
        answer_query_embedding."""
        return self.query_position_embedding

    def compute_routing_representations(
        self,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
        *,
        device: torch.device | None = None,
        lmax: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discrete query and key representations and key padding mask.

        Runtime inputs strictly exclude token content, teacher maps, target tokens,
        and oracle information.

        Returns:
            Tuple of:
            - query: `[batch, out_max, d_operator]`
            - key: `[batch, lmax, d_operator]`
            - content_pad_mask: `[batch, lmax]` (boolean, True for padded positions j >= L)
        """
        if device is None:
            device = self.query_position_embedding.weight.device

        batch = len(content_lengths)
        out_max = max(output_lengths)
        if lmax is None:
            lmax = max(content_lengths)

        # Finite domain boundary validation
        if out_max > self.max_sequence_length:
            raise ValueError(
                f"output length {out_max} exceeds configured "
                f"max_sequence_length {self.max_sequence_length}"
            )
        if lmax > self.max_sequence_length:
            raise ValueError(
                f"content length {lmax} exceeds configured "
                f"max_sequence_length {self.max_sequence_length}"
            )

        # Discrete query position representation: i in {0..out_max-1}
        query_ids = torch.arange(out_max, device=device).unsqueeze(0).expand(batch, out_max)
        query_pos = self.query_position_embedding(query_ids)

        # Discrete length representation: L in {0..max_sequence_length}
        c_lens = torch.tensor(content_lengths, device=device, dtype=torch.long)
        length_token = self.length_embedding(c_lens).unsqueeze(1)

        # Generic relation-conditioning boundary (no relation-specific table or branch)
        if argument_values is None or self.arg_encoder is None or self.arg_proj is None:
            arg_token = query_pos.new_zeros(batch, 1, self.d_operator)
        else:
            arg_embedding = self.arg_encoder(argument_values)
            arg_token = self.arg_proj(arg_embedding).unsqueeze(1)

        query = query_pos + length_token + arg_token

        # Discrete key position representation: j in {0..lmax-1} (NO content features)
        key_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        key = self.key_position_embedding(key_ids)

        # Padding mask for keys: True where key position >= real content length L
        content_pad_mask = key_ids >= c_lens.unsqueeze(1)

        return query, key, content_pad_mask

    def compute_routing_scores(
        self,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
        *,
        device: torch.device | None = None,
        lmax: int | None = None,
    ) -> torch.Tensor:
        """Compute pre-softmax multihead routing scores S_h(i, j; L).

        Returns:
            Scores tensor of shape `[batch, n_head, out_max, lmax]` with -inf
            at padded key positions (j >= L).
        """
        query, key, pad_mask = self.compute_routing_representations(
            content_lengths, output_lengths, argument_values, device=device, lmax=lmax
        )
        batch, out_max, _ = query.shape
        _, l_seq, _ = key.shape
        head_dim = self.d_operator // self.n_head

        # Extract Q and K linear projection weights and biases from cross_attn
        wq, wk, _ = self.cross_attn.in_proj_weight.chunk(3, dim=0)
        if self.cross_attn.in_proj_bias is not None:
            bq, bk, _ = self.cross_attn.in_proj_bias.chunk(3, dim=0)
        else:
            bq, bk = None, None

        q_proj = F.linear(query, wq, bq).view(batch, out_max, self.n_head, head_dim).transpose(1, 2)
        k_proj = F.linear(key, wk, bk).view(batch, l_seq, self.n_head, head_dim).transpose(1, 2)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (head_dim ** 0.5)
        # Apply padding mask: -inf for padded key positions
        mask_expanded = pad_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, lmax]
        scores = scores.masked_fill(mask_expanded, float("-inf"))
        return scores

    def compute_attention_weights(
        self,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
        *,
        device: torch.device | None = None,
        lmax: int | None = None,
        average_heads: bool = False,
    ) -> torch.Tensor:
        """Compute post-softmax attention weights.

        Returns:
            If average_heads is False: `[batch, n_head, out_max, lmax]`
            If average_heads is True: `[batch, out_max, lmax]`
        """
        scores = self.compute_routing_scores(
            content_lengths, output_lengths, argument_values, device=device, lmax=lmax
        )
        weights = F.softmax(scores, dim=-1)
        if average_heads:
            return weights.mean(dim=1)
        return weights

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: Sequence[int],
        output_lengths: Sequence[int],
        argument_values: Sequence[Any] | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply CD-DPCA cross-attention operator over content features.

        Args:
            content_features: Tensor `[batch, lmax, d_model]`
            content_lengths: Sequence of integer lengths L
            output_lengths: Sequence of integer output lengths L_out
            argument_values: Optional sequence of generic argument values
            return_attention: If True, returns `(logits, attn_weights)`

        Returns:
            Logits of shape `[batch, max(output_lengths), vocab_size]`
            (or tuple `(logits, attn_weights)` if return_attention=True).
        """
        device = content_features.device
        batch, lmax, _ = content_features.shape
        out_max = max(output_lengths)

        if not self.enabled:
            zeros = content_features.new_zeros(batch, out_max, self.config.vocab_size)
            if return_attention:
                attn_zeros = content_features.new_zeros(batch, self.n_head, out_max, lmax)
                return zeros, attn_zeros
            return zeros

        self.forward_call_count += 1

        # 1. Routing representations (strictly content-free)
        query, key, content_pad_mask = self.compute_routing_representations(
            content_lengths, output_lengths, argument_values, device=device, lmax=lmax
        )

        # 2. Content-only value path (retaining existing value projection & pos emb)
        content_position_ids = torch.arange(lmax, device=device).unsqueeze(0).expand(batch, lmax)
        value = self.content_in_proj(content_features) + self.content_position_embedding(
            content_position_ids
        )

        # 3. Cross-attention execution with key padding mask
        if return_attention:
            attn_out, attn_weights = self.cross_attn(
                query,
                key,
                value,
                key_padding_mask=content_pad_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out, _ = self.cross_attn(
                query,
                key,
                value,
                key_padding_mask=content_pad_mask,
                need_weights=False,
            )
            attn_weights = None

        # 4. Downstream modules (retaining existing LayerNorm, FFN, readout)
        hidden = self.attn_norm(query + attn_out)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        logits = self.readout(hidden)

        if return_attention:
            assert attn_weights is not None
            return logits, attn_weights
        return logits


# Short alias
CDDPCAPrimitive = ContentDecoupledDiscretePositionalCrossAttentionPrimitive


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


# ---------------------------------------------------------------------------
# Reverse Relative Primitive (Branch B / Inductive Bias for Sequence Reversal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReverseRelativePrimitiveConfig:
    """Configuration for `ReverseRelativePrimitive`."""

    d_model: int = 192
    d_operator: int = 32
    n_head: int = 4
    d_operator_ff: int = 64
    vocab_size: int = 10
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.d_operator % self.n_head != 0:
            raise ValueError(
                f"d_operator ({self.d_operator}) must be divisible by n_head ({self.n_head})"
            )


class ReverseRelativePrimitive(PrimitiveBase):
    """Primitive-scale (17,290 params) REVERSE operator with modular reverse relative bias."""

    def __init__(
        self,
        primitive_id: int,
        config: ReverseRelativePrimitiveConfig | None = None,
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
        cfg = config or ReverseRelativePrimitiveConfig()
        self.config = cfg
        self.operation = "REVERSE"
        self.d_operator = cfg.d_operator
        self.n_head = cfg.n_head
        self.max_sequence_length = cfg.max_sequence_length

        self.content_in_proj = nn.Linear(cfg.d_model, cfg.d_operator)
        self.content_position_embedding = nn.Embedding(cfg.max_sequence_length, cfg.d_operator)
        self.answer_query_embedding = nn.Embedding(cfg.max_sequence_length, cfg.d_operator)

        # Learned modular reverse relative position bias: [max_sequence_length, n_head]
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
        argument_values: Sequence[Any] | None = None,
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
        query = self.answer_query_embedding(query_ids)

        # Modular reverse displacement: disp[b, s, p] = (p - (L - 1 - s)) % L
        c_lens = torch.tensor(content_lengths, device=device).view(batch, 1, 1)
        s_idx = torch.arange(out_max, device=device).view(1, out_max, 1)
        p_idx = torch.arange(lmax, device=device).view(1, 1, lmax)
        target_pos = c_lens - 1 - s_idx
        disp = (p_idx - target_pos) % c_lens
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
