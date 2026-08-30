"""Fixed dense Transformer baseline (Phase A Milestone A2 / Task 003).

A small decoder-only (GPT-style) Transformer that is the "stable core"
referenced in `docs/design-docs/ARCHITECTURE.md`. It is trained
autoregressively on `[BOS] input... [SEP] target... [EOS]` sequences
(see `apc.core.data`), which handles the variable input/output lengths
produced by the Phase A symbolic environment without a separate encoder
stack.

Deliberately minimal: LayerNorm + causal self-attention + MLP blocks,
learned absolute positional embeddings, tied input/output embeddings.
No KV-cache, no rotary embeddings, no dynamic capacity — those belong to
later Phase A milestones (primitive bank, plastic workspace).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    """Explicit, serializable configuration for `DecoderOnlyTransformer`."""

    vocab_size: int
    max_seq_len: int
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 768
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled dot-product attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_dropout)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Standard post-attention feed-forward block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: attention then MLP, each with a residual."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """The Phase A fixed dense baseline: a small autoregressive Transformer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids`: `[batch, seq_len]`. Returns logits `[batch, seq_len, vocab_size]`."""
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input length {seq_len} exceeds configured max_seq_len "
                f"{self.config.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits: torch.Tensor = self.head(x)
        return logits
