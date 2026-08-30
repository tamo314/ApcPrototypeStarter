"""Persistent primitive bank (Phase A Milestone A3 / Task 004).

Owns a collection of `Primitive` records and exposes the bookkeeping the
architecture doc (`docs/design-docs/ARCHITECTURE.md` section 5) assigns
to the bank: id assignment, status/enable transitions, usage statistics,
and the three separate parameter counts required by section 12 (total
resident, persistent, active-per-step).

Fixed capacity only: this task does not implement a controller-driven
allocator. New primitives are added explicitly via `new_primitive` /
`add_primitive`; nothing here grows the bank on its own.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus


class PrimitiveBank(nn.Module):
    """A registry of `Primitive` modules keyed by integer id."""

    def __init__(self, primitives: Iterable[Primitive] | None = None) -> None:
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

    def get(self, primitive_id: int) -> Primitive:
        try:
            primitive = self._primitives[str(primitive_id)]
        except KeyError:
            raise KeyError(f"No primitive with id {primitive_id} in bank") from None
        assert isinstance(primitive, Primitive)
        return primitive

    def get_many(self, ids: Iterable[int]) -> list[Primitive]:
        return [self.get(pid) for pid in ids]

    def add_primitive(self, primitive: Primitive) -> int:
        """Register a pre-constructed primitive. Raises on a duplicate id."""
        key = str(primitive.primitive_id)
        if key in self._primitives:
            raise ValueError(f"Primitive id {primitive.primitive_id} already exists in bank")
        self._primitives[key] = primitive
        self._next_id = max(self._next_id, primitive.primitive_id + 1)
        return primitive.primitive_id

    def new_primitive(
        self,
        config: PrimitiveConfig,
        *,
        status: PrimitiveStatus = PrimitiveStatus.CANDIDATE,
        created_at_task: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> Primitive:
        """Construct a primitive with an auto-assigned id, add it, and return it."""
        primitive = Primitive(
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
        """Retire a primitive: mark ARCHIVED and disable it.

        A hook for future consolidation/merge steps (Task 010+); this task
        only needs the status/enabled transition, not the merge itself.
        """
        primitive = self.get(primitive_id)
        primitive.status = PrimitiveStatus.ARCHIVED
        primitive.enabled = False

    def freeze(self, primitive_id: int) -> None:
        self.get(primitive_id).freeze()

    def unfreeze(self, primitive_id: int) -> None:
        self.get(primitive_id).unfreeze()

    def freeze_by_status(self, status: PrimitiveStatus) -> None:
        """Freeze every primitive currently in `status` (e.g. STABLE, per ADR-0004)."""
        for pid in self.ids_by_status(status):
            self.freeze(pid)

    def record_usage(self, ids: Iterable[int]) -> None:
        """Increment `usage_count` for each primitive id, e.g. after a
        forward pass reports which ids the router selected."""
        for pid in ids:
            self.get(pid).record_usage()

    def usage_counts(self) -> dict[int, int]:
        return {pid: self.get(pid).usage_count for pid in self.ids()}

    def update_utility(self, primitive_id: int, value: float, *, decay: float = 0.99) -> None:
        self.get(primitive_id).update_utility(value, decay=decay)

    def total_parameter_count(self, *, trainable_only: bool = False) -> int:
        """All primitives in the bank, any status, enabled or not."""
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only) for pid in self.ids()
        )

    def persistent_parameter_count(self, *, trainable_only: bool = False) -> int:
        """Enabled STABLE primitives: the resident persistent primitive cost
        (see architecture doc section 12)."""
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only)
            for pid in self.ids_by_status(PrimitiveStatus.STABLE)
            if self.get(pid).enabled
        )

    def active_parameter_count(
        self, selected_ids: Iterable[int], *, trainable_only: bool = False
    ) -> int:
        """Parameters of the given (e.g. router-selected) ids that are
        actually enabled — the "active parameters" of one forward step."""
        return sum(
            self.get(pid).num_parameters(trainable_only=trainable_only)
            for pid in selected_ids
            if self.get(pid).enabled
        )
