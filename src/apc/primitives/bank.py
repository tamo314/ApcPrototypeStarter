"""Persistent primitive bank (Phase A.1 Post-Diagnostic: Branch B Integration).

Owns a collection of heterogeneous `PrimitiveBase` records and exposes the bookkeeping
the architecture doc assigns to the bank: id assignment, status/enable transitions,
usage statistics, and strict sparse parameter/call tracking.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import nn

from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PointwisePrimitive,
    PrimitiveBase,
    PrimitiveConfig,
    PrimitiveStatus,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
)

__all__ = ["PrimitiveBank"]


class PrimitiveBank(nn.Module):
    """A registry of heterogeneous `PrimitiveBase` modules keyed by integer id."""

    def __init__(self, primitives: Iterable[PrimitiveBase] | None = None) -> None:
        super().__init__()
        self._primitives = nn.ModuleDict()
        self._next_id = 0
        for primitive in primitives or []:
            self.add_primitive(primitive)

    def __len__(self) -> int:
        return len(self._primitives)

    def ids(self) -> list[int]:
        return sorted(int(key) for key in self._primitives)

    def ids_by_status(self, status: PrimitiveStatus) -> list[int]:
        return [pid for pid in self.ids() if self.get(pid).status == status]

    def get(self, primitive_id: int) -> PrimitiveBase:
        try:
            primitive = self._primitives[str(primitive_id)]
        except KeyError:
            raise KeyError(f"No primitive with id {primitive_id} in bank") from None
        assert isinstance(primitive, PrimitiveBase)
        return primitive

    def get_many(self, ids: Iterable[int]) -> list[PrimitiveBase]:
        return [self.get(pid) for pid in ids]

    def add_primitive(self, primitive: PrimitiveBase) -> int:
        """Register a pre-constructed primitive. Raises on a duplicate id."""
        key = str(primitive.primitive_id)
        if key in self._primitives:
            raise ValueError(f"Primitive id {primitive.primitive_id} already exists in bank")
        self._primitives[key] = primitive
        self._next_id = max(self._next_id, primitive.primitive_id + 1)
        return primitive.primitive_id

    def new_pointwise_primitive(
        self,
        config: PrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PointwisePrimitive:
        """Construct a PointwisePrimitive with an auto-assigned id, add it, and return it."""
        primitive = PointwisePrimitive(
            self._next_id,
            config,
            status=status,
            created_at_task=created_at_task,
            metadata=metadata,
        )
        self.add_primitive(primitive)
        return primitive

    # Backward-compatible alias for Phase A code
    new_primitive = new_pointwise_primitive

    def new_cross_position_primitive(
        self,
        config: CrossPositionPrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> CrossPositionPrimitive:
        """Construct a CrossPositionPrimitive with an auto-assigned id, add it, and return it."""
        primitive = CrossPositionPrimitive(
            self._next_id,
            config,
            status=status,
            created_at_task=created_at_task,
            metadata=metadata,
        )
        self.add_primitive(primitive)
        return primitive

    def new_shift_relative_primitive(
        self,
        config: ShiftRelativePrimitiveConfig | None = None,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ShiftRelativePrimitive:
        """Construct a ShiftRelativePrimitive with an auto-assigned id, add it, and return it."""
        primitive = ShiftRelativePrimitive(
            self._next_id,
            config,
            status=status,
            created_at_task=created_at_task,
            metadata=metadata,
        )
        self.add_primitive(primitive)
        return primitive

    def set_status(self, primitive_id: int, status: PrimitiveStatus) -> None:
        self.get(primitive_id).status = status

    def enable(self, primitive_id: int) -> None:
        self.get(primitive_id).enabled = True

    def disable(self, primitive_id: int) -> None:
        self.get(primitive_id).enabled = False

    def archive(self, primitive_id: int) -> None:
        """Retire a primitive: mark ARCHIVED and disable it."""
        primitive = self.get(primitive_id)
        primitive.status = PrimitiveStatus.ARCHIVED
        primitive.enabled = False

    def freeze(self, primitive_id: int) -> None:
        self.get(primitive_id).freeze()

    def unfreeze(self, primitive_id: int) -> None:
        self.get(primitive_id).unfreeze()

    def freeze_by_status(self, status: PrimitiveStatus) -> None:
        """Freeze every primitive currently in `status`."""
        for pid in self.ids_by_status(status):
            self.freeze(pid)

    def freeze_all(self) -> None:
        """Freeze every primitive in the bank."""
        for pid in self.ids():
            self.freeze(pid)

    def record_usage(self, ids: Iterable[int]) -> None:
        """Increment `usage_count` for each primitive id."""
        for pid in ids:
            self.get(pid).record_usage()

    def usage_counts(self) -> dict[int, int]:
        return {pid: self.get(pid).usage_count for pid in self.ids()}

    def reset_all_forward_call_counts(self) -> None:
        """Reset execution counters across all primitives in the bank."""
        for pid in self.ids():
            self.get(pid).reset_forward_call_count()

    def forward_call_counts(self) -> dict[int, int]:
        return {pid: self.get(pid).forward_call_count for pid in self.ids()}

    def update_utility(self, primitive_id: int, value: float, *, decay: float = 0.99) -> None:
        self.get(primitive_id).update_utility(value, decay=decay)

    def total_parameter_count(self, *, trainable_only: bool = False) -> int:
        """All primitives in the bank, any status, enabled or not."""
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only) for pid in self.ids()
        )

    def persistent_parameter_count(self, *, trainable_only: bool = False) -> int:
        """Enabled STABLE primitives: the resident persistent primitive cost."""
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only)
            for pid in self.ids_by_status(PrimitiveStatus.STABLE)
            if self.get(pid).enabled
        )

    def active_parameter_count(
        self, selected_ids: Iterable[int], *, trainable_only: bool = False
    ) -> int:
        """Parameters of unique enabled router-selected ids."""
        unique_ids = dict.fromkeys(int(pid) for pid in selected_ids)
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only)
            for pid in unique_ids
            if str(pid) in self._primitives and self.get(pid).enabled
        )
