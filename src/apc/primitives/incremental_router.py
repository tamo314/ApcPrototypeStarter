"""Class-incremental router update policies and replay management (Phase A.2 Task A2-C003).

Implements the incremental router update conditions defined in:
`docs/design-docs/INCREMENTAL_ROUTER_AND_BANK_SCALING.md` section 2:
- R0: Full retrain upper bound (diagnostic unconstrained retrain across all classes)
- R1: Naive new-class-only update (expected catastrophic forgetting baseline)
- R2: Bounded incremental update (primary condition with bounded replay memory / prototypes)

And bounded replay memory:
- <= 32 examples per old class
- <= 512 total historical examples

Also provides token embedding alignment for frozen Stable Core when new operations
are registered, preserving argument-value and canonical operation token embeddings.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from apc.core.tokens import SharedCoreTokens
from apc.primitives.router import Router

MAX_REPLAY_PER_CLASS_DEFAULT: Final[int] = 32
MAX_REPLAY_TOTAL_DEFAULT: Final[int] = 512


class IncrementalUpdateCondition(str, Enum):  # noqa: UP042
    """Class-incremental update conditions per ADR / INCREMENTAL_ROUTER_AND_BANK_SCALING.md."""

    R0_FULL_RETRAIN = "R0"
    R1_NAIVE_NEW = "R1"
    R2_BOUNDED_REPLAY = "R2"


def align_shared_core_embeddings(
    target_model: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    old_tokens: SharedCoreTokens | None,
    new_tokens: SharedCoreTokens,
) -> dict[str, torch.Tensor]:
    """Safely map token embeddings from an older token layout to a newer expanded layout.

    Strict Invariant:
    1. Environment tokens, specials, and task framing are preserved identically.
    2. Existing operation tokens (up to old_tokens.num_operations) are preserved identically.
    3. New operation tokens are initialized with normal(0.0, 0.02).
    4. Argument tokens (old_tokens.arg_base + v) are strictly re-mapped to
       new_tokens.arg_base + v so argument-value embeddings are NOT shifted or scrambled.
    """
    if "token_emb.weight" in checkpoint_state and old_tokens is None:
        old_vocab = checkpoint_state["token_emb.weight"].shape[0]
        # Infer old_tokens from vocabulary size
        from apc.core.tokens import build_shared_core_tokens

        base_specials = new_tokens.op_base  # e.g. 16
        arg_span = new_tokens.arg_span  # e.g. 10
        old_num_ops = max(1, old_vocab - base_specials - arg_span)
        old_tokens = build_shared_core_tokens(
            new_tokens.env_vocab_size,
            num_operations=old_num_ops,
            arg_span=arg_span,
        )

    assert old_tokens is not None
    aligned_state = dict(checkpoint_state)
    with torch.no_grad():
        for key in ["token_emb.weight", "head.weight"]:
            if key not in checkpoint_state:
                continue
            old_w = checkpoint_state[key]
            cur_w = target_model.state_dict()[key].clone()

            # 1. Base tokens & common operations prefix
            common_ops = min(old_tokens.num_operations, new_tokens.num_operations)
            prefix_len = old_tokens.op_base + common_ops
            cur_w[:prefix_len] = old_w[:prefix_len]

            # 2. Initialize any new operation tokens freshly
            if new_tokens.num_operations > old_tokens.num_operations:
                new_op_start = new_tokens.op_base + old_tokens.num_operations
                new_op_end = new_tokens.arg_base
                nn.init.normal_(cur_w[new_op_start:new_op_end], mean=0.0, std=0.02)

            # 3. Remap argument-value tokens from old_tokens.arg_base to new_tokens.arg_base
            common_args = min(old_tokens.arg_span, new_tokens.arg_span)
            for v in range(common_args):
                old_idx = old_tokens.arg_base + v
                new_idx = new_tokens.arg_base + v
                if old_idx < old_w.shape[0] and new_idx < cur_w.shape[0]:
                    cur_w[new_idx] = old_w[old_idx]

            aligned_state[key] = cur_w

    return aligned_state


class RouterReplayBuffer:
    """Bounded exemplar replay buffer for incremental router updates.

    Enforces:
    - max_per_class: <= 32 exemplars per class by default.
    - max_total: <= 512 total exemplars by default.
    - Balanced mini-batch sampling.
    """

    def __init__(
        self,
        max_per_class: int = MAX_REPLAY_PER_CLASS_DEFAULT,
        max_total: int = MAX_REPLAY_TOTAL_DEFAULT,
    ) -> None:
        if max_per_class < 1:
            raise ValueError(f"max_per_class must be >= 1, got {max_per_class}")
        if max_total < 1:
            raise ValueError(f"max_total must be >= 1, got {max_total}")
        self.max_per_class = max_per_class
        self.max_total = max_total
        self._buffer: dict[int, list[tuple[torch.Tensor, int]]] = {}

    def __len__(self) -> int:
        return sum(len(items) for items in self._buffer.values())

    def total_count(self) -> int:
        return len(self)

    def classes(self) -> list[int]:
        return sorted(self._buffer.keys())

    def count_for_class(self, primitive_id: int) -> int:
        return len(self._buffer.get(primitive_id, []))

    def get_exemplars(self, primitive_id: int) -> list[tuple[torch.Tensor, int]]:
        return list(self._buffer.get(primitive_id, []))

    def add_exemplars(
        self,
        primitive_id: int,
        exemplars: Sequence[tuple[torch.Tensor, int]],
        rng: random.Random | None = None,
    ) -> None:
        """Add exemplars for primitive_id, respecting per-class and total bounds."""
        local_rng = rng or random.Random(42 + primitive_id)
        current = self._buffer.get(primitive_id, [])
        combined = list(current) + [(item[0].detach().cpu(), item[1]) for item in exemplars]

        # Enforce per-class limit
        if len(combined) > self.max_per_class:
            combined = local_rng.sample(combined, self.max_per_class)

        self._buffer[primitive_id] = combined
        self._enforce_total_bound(local_rng)

    def _enforce_total_bound(self, rng: random.Random) -> None:
        """Enforce max_total by proportionally trimming per-class quotas if total > max_total."""
        total = self.total_count()
        if total <= self.max_total:
            return

        num_classes = len(self._buffer)
        if num_classes == 0:
            return

        per_class_limit = max(1, self.max_total // num_classes)
        for pid in list(self._buffer.keys()):
            items = self._buffer[pid]
            if len(items) > per_class_limit:
                self._buffer[pid] = rng.sample(items, per_class_limit)


@dataclass(frozen=True)
class IncrementalRouterConfig:
    """Configuration for incremental router calibration."""

    condition: IncrementalUpdateCondition = IncrementalUpdateCondition.R2_BOUNDED_REPLAY
    replay_max_per_class: int = MAX_REPLAY_PER_CLASS_DEFAULT
    replay_max_total: int = MAX_REPLAY_TOTAL_DEFAULT
    router_lr: float = 0.005
    router_steps: int = 250
    weight_decay: float = 1e-4
    seed: int = 0


def update_router_incrementally(
    router: Router,
    candidate_ids: Sequence[int],
    new_primitive_ids: Sequence[int],
    new_data_by_pid: dict[int, list[tuple[torch.Tensor, int]]],
    replay_buffer: RouterReplayBuffer,
    *,
    config: IncrementalRouterConfig,
    all_historical_data_by_pid: dict[int, list[tuple[torch.Tensor, int]]] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute one incremental update step on the Router under R0, R1, or R2.

    Strict Invariants:
    1. Router keys for all candidate_ids are registered before training.
    2. R0: Uses all historical data for unconstrained retrain (diagnostic upper bound).
    3. R1: Uses ONLY new_data_by_pid (demonstrates catastrophic forgetting).
    4. R2: Uses new_data_by_pid + bounded replay from replay_buffer (primary condition).
    5. Usage counts are reset to 0 after training so execution accounting is never polluted.
    """
    for pid in candidate_ids:
        if not router.has_primitive(pid):
            router.add_primitive_key(pid)

    active_device = device or next(router.parameters()).device
    router.to(active_device)
    router.train()

    candidate_list = list(candidate_ids)
    pid_to_class_idx = {pid: idx for idx, pid in enumerate(candidate_list)}

    rng = random.Random(config.seed * 3001 + 73)
    losses: list[float] = []

    if config.condition == IncrementalUpdateCondition.R0_FULL_RETRAIN:
        # Full historical retrain across all classes (query_proj + keys)
        if all_historical_data_by_pid is None:
            raise ValueError("R0_FULL_RETRAIN requires all_historical_data_by_pid")

        router.query_proj.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=config.router_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.router_steps, eta_min=1e-5
        )

        for _ in range(config.router_steps):
            batch: list[tuple[torch.Tensor, int]] = []
            for pid in candidate_list:
                items = all_historical_data_by_pid.get(pid, [])
                if items:
                    for _ in range(min(4, len(items))):
                        batch.append(rng.choice(items))

            if not batch:
                continue

            z_batch = torch.stack([item[0].to(active_device) for item in batch], dim=0)
            targets = torch.tensor(
                [pid_to_class_idx[item[1]] for item in batch],
                dtype=torch.long,
                device=active_device,
            )

            optimizer.zero_grad(set_to_none=True)
            keys = router._stacked_keys(candidate_list)
            query = router.query_proj(z_batch)
            scores = query @ keys.transpose(0, 1)

            loss = F.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

    elif config.condition == IncrementalUpdateCondition.R1_NAIVE_NEW:
        # Naive update: trained exclusively on new class data without replay
        router.query_proj.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=config.router_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.router_steps, eta_min=1e-5
        )

        new_items: list[tuple[torch.Tensor, int]] = []
        for pid in new_primitive_ids:
            new_items.extend(new_data_by_pid.get(pid, []))

        if not new_items:
            raise ValueError("R1_NAIVE_NEW requires non-empty new_data_by_pid")

        for _ in range(config.router_steps):
            batch = [rng.choice(new_items) for _ in range(min(16, len(new_items)))]

            z_batch = torch.stack([item[0].to(active_device) for item in batch], dim=0)
            targets = torch.tensor(
                [pid_to_class_idx[item[1]] for item in batch],
                dtype=torch.long,
                device=active_device,
            )

            optimizer.zero_grad(set_to_none=True)
            keys = router._stacked_keys(candidate_list)
            query = router.query_proj(z_batch)
            scores = query @ keys.transpose(0, 1)

            loss = F.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

    elif config.condition == IncrementalUpdateCondition.R2_BOUNDED_REPLAY:
        # Bounded replay: freeze query_proj to preserve the established score projection geometry,
        # and optimize candidate primitive keys against balanced new + replay exemplars.
        router.query_proj.requires_grad_(False)
        key_params = [router.key_parameter(p) for p in candidate_list]
        optimizer = torch.optim.AdamW(
            key_params,
            lr=config.router_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.router_steps, eta_min=1e-5
        )

        old_classes = [pid for pid in replay_buffer.classes() if pid not in new_primitive_ids]

        for _ in range(config.router_steps):
            batch = []

            # Sample from new primitive data (4 per new class)
            for new_pid in new_primitive_ids:
                items = new_data_by_pid.get(new_pid, [])
                if items:
                    for _ in range(min(4, len(items))):
                        batch.append(rng.choice(items))

            # Sample from replay buffer for old classes (up to 2 per old class)
            for old_pid in old_classes:
                items = replay_buffer.get_exemplars(old_pid)
                if items:
                    for _ in range(min(2, len(items))):
                        batch.append(rng.choice(items))

            if not batch:
                continue

            z_batch = torch.stack([item[0].to(active_device) for item in batch], dim=0)
            targets = torch.tensor(
                [pid_to_class_idx[item[1]] for item in batch],
                dtype=torch.long,
                device=active_device,
            )

            optimizer.zero_grad(set_to_none=True)
            keys = router._stacked_keys(candidate_list)
            query = router.query_proj(z_batch)
            scores = query @ keys.transpose(0, 1)

            loss = F.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

    # Add new primitive data to replay buffer for future incremental stages
    for new_pid in new_primitive_ids:
        items = new_data_by_pid.get(new_pid, [])
        if items:
            replay_buffer.add_exemplars(new_pid, items, rng=rng)

    router.eval()
    for pid in candidate_list:
        router.usage_count[pid] = 0

    return {
        "condition": config.condition.value,
        "candidate_count": len(candidate_list),
        "replay_buffer_size": len(replay_buffer),
        "final_loss": losses[-1] if losses else 0.0,
    }


def evaluate_router_accuracy(
    router: Router,
    candidate_ids: Sequence[int],
    eval_data_by_pid: dict[int, list[tuple[torch.Tensor, int]]],
    *,
    device: torch.device | None = None,
) -> dict[int, dict[str, float]]:
    """Evaluate router top-1 and top-k accuracy per primitive ID."""
    active_device = device or next(router.parameters()).device
    router.eval()
    candidate_list = list(candidate_ids)

    per_pid_metrics: dict[int, dict[str, float]] = {}

    with torch.no_grad():
        for target_pid, examples in eval_data_by_pid.items():
            if not examples:
                continue
            z_batch = torch.stack([ex[0].to(active_device) for ex in examples], dim=0)
            out = router(z_batch, candidate_list)

            # selected_ids shape: [B, k_eff]
            top1_preds = out.selected_ids[:, 0]
            top1_correct = (top1_preds == target_pid).float().sum().item()
            top1_acc = top1_correct / len(examples)

            # top-k match: target in any selected slot
            topk_match = (out.selected_ids == target_pid).any(dim=-1).float().sum().item()
            topk_acc = topk_match / len(examples)

            mean_entropy = out.entropy.mean().item()

            per_pid_metrics[target_pid] = {
                "top1": top1_acc,
                "topk": topk_acc,
                "entropy": mean_entropy,
                "num_examples": float(len(examples)),
            }

    return per_pid_metrics
