"""Temporary plastic workspace (Phase A Milestone A5 / Task 007).

Implements the plastic workspace from `docs/design-docs/ARCHITECTURE.md`
section 7: temporary low-rank transforms that share the persistent
`Primitive` execution interface (`h + gate * B(A(h))`) but live in a
registry kept separate from `apc.primitives.bank.PrimitiveBank`, per the
AGENTS.md rule to keep temporary plastic capacity separate from
persistent primitive capacity in code and checkpoints.

Key invariant (architecture doc section 7): temporary capacity is not
part of the persistent primitive bank until consolidation and shadow
validation succeed (ADR-0005). This module only owns the temporary
registry's lifecycle -- allocate a batch of fresh trainable transforms,
train them, and release them -- it does not implement consolidation or
shadow validation (Task 010+).

Sizing a batch of transforms (preset name, transform count, rank) is the
job of `apc.plastic.allocator.Allocator`; `PlasticWorkspace.allocate`
takes an already-resolved count/config so the workspace itself stays
agnostic of preset naming.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from torch import nn

from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus

if TYPE_CHECKING:
    from apc.plastic.allocator import AllocatorPreset


class PlasticWorkspace(nn.Module):
    """Registry of temporary `Primitive` transforms allocated during PLASTIC.

    Unlike `PrimitiveBank`, this registry is meant to be emptied: `allocate`
    fills it with a fresh batch of trainable transforms, and `release`
    drops that batch entirely once consolidation output has passed shadow
    validation (or been discarded). Only one batch may be resident at a
    time -- allocating on top of an existing batch is an error, so callers
    cannot silently accumulate temporary capacity across PLASTIC episodes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._transforms = nn.ModuleDict()
        self._next_id = 0
        self.preset: AllocatorPreset | None = None
        self.allocated_at_task: int | None = None

    def __len__(self) -> int:
        return len(self._transforms)

    @property
    def is_allocated(self) -> bool:
        return len(self._transforms) > 0

    def ids(self) -> list[int]:
        return sorted(int(key) for key in self._transforms)

    def get(self, transform_id: int) -> Primitive:
        try:
            transform = self._transforms[str(transform_id)]
        except KeyError:
            raise KeyError(f"No temporary transform with id {transform_id} in workspace") from None
        assert isinstance(transform, Primitive)
        return transform

    def get_many(self, ids: Iterable[int]) -> list[Primitive]:
        return [self.get(tid) for tid in ids]

    def allocate(
        self,
        preset: AllocatorPreset,
        num_transforms: int,
        config: PrimitiveConfig,
        *,
        created_at_task: int = 0,
    ) -> list[int]:
        """Fill the (currently empty) workspace with `num_transforms` fresh
        trainable transforms built from `config`.

        `preset` is stored only as a label for reporting/logging (e.g. the
        `AllocatorPreset` that produced this batch); the workspace itself
        does not interpret it. Raises `RuntimeError` if the workspace
        already holds a batch -- call `release()` first.
        """
        if self.is_allocated:
            raise RuntimeError(
                "PlasticWorkspace already has allocated capacity; call release() first"
            )
        if num_transforms < 1:
            raise ValueError(f"num_transforms must be >= 1, got {num_transforms}")

        new_ids: list[int] = []
        for _ in range(num_transforms):
            transform = Primitive(
                self._next_id,
                config,
                status=PrimitiveStatus.CANDIDATE,
                created_at_task=created_at_task,
            )
            self._transforms[str(transform.primitive_id)] = transform
            new_ids.append(transform.primitive_id)
            self._next_id += 1

        self.preset = preset
        self.allocated_at_task = created_at_task
        return new_ids

    def release(self) -> dict[int, Primitive]:
        """Drop every temporary transform and return them keyed by id.

        The caller (consolidation/shadow-validation code, Task 010+) is
        responsible for extracting whatever it needs before capacity is
        released; this method only clears the registry.
        """
        released = {tid: self.get(tid) for tid in self.ids()}
        self._transforms = nn.ModuleDict()
        self.preset = None
        self.allocated_at_task = None
        return released

    def freeze_all(self) -> None:
        for tid in self.ids():
            self.get(tid).freeze()

    def unfreeze_all(self) -> None:
        for tid in self.ids():
            self.get(tid).unfreeze()

    def record_usage(self, ids: Iterable[int]) -> None:
        for tid in ids:
            self.get(tid).record_usage()

    def usage_counts(self) -> dict[int, int]:
        return {tid: self.get(tid).usage_count for tid in self.ids()}

    def total_parameter_count(self, *, trainable_only: bool = False) -> int:
        """Current temporary-capacity size: 0 when unallocated, the full
        batch cost while allocated (the "temporary peak parameters" of
        architecture doc section 12, tracked across the run by the
        caller)."""
        return sum(
            self.get(tid).num_parameters(trainable_only=trainable_only) for tid in self.ids()
        )

    def active_parameter_count(
        self, selected_ids: Iterable[int], *, trainable_only: bool = False
    ) -> int:
        """Parameters of the given (e.g. router-selected) temporary ids
        that are actually enabled."""
        return sum(
            self.get(tid).num_parameters(trainable_only=trainable_only)
            for tid in selected_ids
            if self.get(tid).enabled
        )
