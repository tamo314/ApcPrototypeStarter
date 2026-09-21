"""Functional replay retention constraint and immutable parent-logit cache."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from apc.cnp.repair import set_mean_masked_bce_loss


@dataclass(frozen=True)
class ParentLogitCache:
    """Parent replay logits bound to a model, architecture, and input identity."""

    parent_hash: str
    architecture_signature: str
    input_ids: tuple[str, ...]
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not self.parent_hash or not self.architecture_signature:
            raise ValueError("parent cache requires parent identity and architecture signature")
        if self.logits.dtype != torch.float32 or self.logits.ndim != 2:
            raise ValueError("parent cache logits must be float32[B,N]")
        if len(self.input_ids) != self.logits.shape[0] or len(set(self.input_ids)) != len(
            self.input_ids
        ):
            raise ValueError("parent cache IDs must be unique and match batch rows")
        if self.logits.requires_grad:
            raise ValueError("parent cache must be detached from the parent graph")

    @property
    def cache_hash(self) -> str:
        payload = hashlib.sha256()
        payload.update(self.parent_hash.encode("ascii"))
        payload.update(self.architecture_signature.encode("utf-8"))
        for input_id in self.input_ids:
            payload.update(input_id.encode("ascii"))
        payload.update(self.logits.detach().cpu().contiguous().numpy().tobytes())
        return payload.hexdigest()

    def validate(
        self, *, parent_hash: str, architecture_signature: str, input_ids: tuple[str, ...]
    ) -> None:
        """Reject stale, reordered, incomplete, or cross-architecture cache use."""

        if self.parent_hash != parent_hash:
            raise ValueError("parent cache hash does not match the immutable parent")
        if self.architecture_signature != architecture_signature:
            raise ValueError("parent cache architecture signature mismatch")
        if self.input_ids != input_ids:
            raise ValueError("parent cache input IDs do not match the replay batch")


def retention_loss(
    replay_logits: torch.Tensor, parent_logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Compute an equal-set mean squared logit constraint against detached parent output."""

    if replay_logits.shape != parent_logits.shape or replay_logits.shape != valid.shape:
        raise ValueError("replay, parent, and valid shapes must match")
    if parent_logits.requires_grad:
        raise ValueError("parent logits must be detached before retention loss")
    valid_counts = valid.sum(dim=1)
    non_empty = valid_counts > 0
    if not bool(non_empty.any()):
        raise ValueError("cannot retain an all-invalid replay batch")
    squared = (replay_logits - parent_logits.detach()).square()
    per_set = (squared * valid).sum(dim=1) / valid_counts.clamp_min(1)
    return per_set[non_empty].mean()


def combined_repair_loss(
    *,
    new_logits: torch.Tensor,
    new_target: torch.Tensor,
    new_valid: torch.Tensor,
    replay_logits: torch.Tensor,
    replay_target: torch.Tensor,
    replay_valid: torch.Tensor,
    parent_replay_logits: torch.Tensor,
    retention_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the registered task-plus-functional-retention objective."""

    if retention_weight not in {0.0, 1.0}:
        raise ValueError("retention_weight must be the registered lambda 0 or 1")
    new_task = set_mean_masked_bce_loss(new_logits, new_target, new_valid)
    replay_task = set_mean_masked_bce_loss(replay_logits, replay_target, replay_valid)
    task = 0.5 * new_task + 0.5 * replay_task
    keep = retention_loss(replay_logits, parent_replay_logits, replay_valid)
    total = task + retention_weight * keep
    return total, {
        "task": float(task.detach().cpu()),
        "retention": float(keep.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
