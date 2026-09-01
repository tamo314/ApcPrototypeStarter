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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle


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


@dataclass(frozen=True)
class EncodedState:
    """Factorized Stable Core encoding (Phase A.1 Task A1-005,
    `docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md` section 4).

    `task_state` (`z_task`) is what routing/novelty read (`apc.primitives.
    router.Router`, `apc.meta.novelty`); `content_state` (`h_content`) is
    the state primitives transform and `decode` reads. Both are
    `[batch, seq_len, d_model]` -- same leading shape as plain `encode`'s
    return, so either field is a drop-in replacement for it.
    """

    task_state: torch.Tensor
    content_state: torch.Tensor


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
        # Task A1-005: a separate (parameter-free -- see `encode_split`)
        # view of the trunk output that routing/novelty can read instead of
        # `content_state` directly.
        self.task_head = nn.LayerNorm(config.d_model, elementwise_affine=False)

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

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids`: `[batch, seq_len]`. Returns the final hidden state
        `[batch, seq_len, d_model]`, i.e. everything up to (not including)
        the output head.

        This is the "insertion point for primitive transforms" that
        `docs/design-docs/ARCHITECTURE.md` section 3 asks the stable core to
        expose: `apc.core.execution` (Task 012) reads this hidden state,
        adds primitive-bank/plastic-workspace residual deltas to it, and
        passes the result to `decode` -- the core itself stays unaware that
        primitives exist.
        """
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
        return x

    def encode_split(self, input_ids: torch.Tensor) -> EncodedState:
        """Factorized encoding (Task A1-005): `content_state` is exactly
        `encode(input_ids)` -- unchanged value and gradient path, so every
        Phase A/A.1 call site that still calls plain `encode` is byte-for-
        byte unaffected by this method's existence. `task_state` is a
        separate *parameter-free* view of that same trunk output
        (`task_head`, a `LayerNorm` with `elementwise_affine=False`): not
        literally the content tensor primitives transform, without adding a
        second encoder stack up front.

        Deliberately parameter-free rather than a learned projection
        (recorded per AGENTS.md workflow -- see `docs/DECISIONS.md`): this
        codebase seeds one shared global `torch` RNG stream per run
        (`apc.utils.seed.set_seed`), and every `nn.Module` with learnable
        weights consumes from it at construction. A *new* randomly-
        initialized submodule -- even one nothing yet reads, like an
        unused learned `task_head` -- shifts every later random draw in
        the same process (primitive bank/router initialization, data
        sampling, training noise), silently changing already-seeded
        integration tests' step-by-step trajectories even though its own
        output is never consumed. A parameter-free transform reads
        `content_state` but registers no `nn.Parameter`, so constructing a
        `DecoderOnlyTransformer` consumes exactly the RNG draws it did
        before this task.

        `PHASE_A1_ARCHITECTURE_DELTA.md` section 4 leaves the exact
        implementation flexible ("must not depend *solely* on a
        low-information content state" is a later milestone's
        generalization gate, not this task's acceptance criterion) -- a
        genuinely learned, separate task-stream encoder can replace
        `task_head` later, once a concrete downstream mechanism (e.g. a
        learned router, A1-015) actually needs one and can afford to
        retune every seed-sensitive test/config that assumes today's RNG
        trajectory.
        """
        content_state = self.encode(input_ids)
        task_state = self.task_head(content_state)
        return EncodedState(task_state=task_state, content_state=content_state)

    def encode_task_content_split(
        self, task_ids: torch.Tensor, content_ids: torch.Tensor
    ) -> TaskContentEncoding:
        """Task A1-R001's structurally task-blind factorization: `encode`
        run twice through the same shared weights, once over a task-only
        sequence and once over a content-only sequence -- see
        `TaskContentEncoding`. `task_ids` and `content_ids` may differ in
        `seq_len`/batch composition; each is encoded independently, so
        neither forward pass observes the other's tokens.
        """
        return TaskContentEncoding(
            task_state=self.encode(task_ids), content_state=self.encode(content_ids)
        )

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project a final hidden state `[..., d_model]` to vocabulary logits."""
        logits: torch.Tensor = self.head(hidden)
        return logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids`: `[batch, seq_len]`. Returns logits `[batch, seq_len, vocab_size]`."""
        return self.decode(self.encode(input_ids))


@dataclass(frozen=True)
class TaskContentEncoding:
    """Structurally task-blind factorized encoding (Phase A.1
    Post-Correction Task A1-R001).

    Unlike `EncodedState`/`encode_split` (Task A1-005), whose `task_state`
    and `content_state` are two readouts of *one* causal forward pass over a
    combined `[task segment][content]` sequence -- meaning a content
    position's hidden state has, by construction, already attended over
    every task token preceding it (see `docs/DECISIONS.md` ADR-0022) --
    `task_state` and `content_state` here come from two *independent*
    forward passes (`DecoderOnlyTransformer.encode_task_content_split`),
    each over its own token-only sequence (`apc.core.data.
    build_task_only_tokens`/`build_content_only_tokens`). `content_state`
    therefore cannot depend on task specification for any input at all,
    regardless of training: there is no computational path from a task
    token to `content_state`, because no task token is ever part of the
    sequence `content_state` is computed from. This is the "shared weights,
    called twice" factorization `docs/design-docs/
    CAUSAL_PRIMITIVE_EXECUTION.md` section 3 describes.

    `task_state` and `content_state` are not required to share a leading
    (batch/seq_len) shape -- they come from different-length sequences and,
    unlike `EncodedState`'s two fields, are not interchangeable drop-ins for
    one another's positions.
    """

    task_state: torch.Tensor
    content_state: torch.Tensor


def register_encode_split_probe(
    model: DecoderOnlyTransformer, callback: Callable[[EncodedState], None]
) -> RemovableHandle:
    """Attach a forward hook that calls `callback` with the `EncodedState`
    (`task_state`/`z_task`, `content_state`/`h_content`) produced by every
    subsequent `encode_split` call (Task A1-C003's "logging/probe hooks").

    Implemented as a forward hook on `model.task_head`, the one submodule
    `encode_split` routes through (`task_state = self.task_head
    (content_state)`): a hook there observes both halves of the split in one
    callback, since its `inputs[0]` is exactly `content_state` and its
    `output` is exactly `task_state`. This lets a training/evaluation loop
    (and later, Task A1-C005's frozen-model representation probes) tap
    `z_task`/`h_content` for logging without threading a new return value
    through every `encode_split` call site. Never fires for plain `encode`
    calls, which do not touch `task_head`.

    Returns the hook handle; call `.remove()` to stop observing.
    """

    def _hook(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module
        callback(EncodedState(task_state=output, content_state=inputs[0]))

    handle: RemovableHandle = model.task_head.register_forward_hook(_hook)
    return handle
