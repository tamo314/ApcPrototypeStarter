"""Plastic-capacity allocator with named size presets (Phase A Milestone A5 / Task 007).

Implements the allocator from `docs/design-docs/ARCHITECTURE.md` section 7:
"The allocator adds capacity in small increments" using named presets

    small:  4 transforms, rank 4
    medium: 8 transforms, rank 8
    large: 16 transforms, rank 8 or 16

The architecture doc is explicit that "exact values are experimental
configuration, not architecture constants", so `DEFAULT_PRESETS` (which
picks rank 16 for `large`) can be overridden per experiment via
`Allocator(presets=...)` without touching this module.

The allocator only decides *how much* capacity a preset means (transform
count and rank) and turns that into a `PlasticWorkspace.allocate` call;
it holds no state of its own about what is currently allocated -- that
lives on the `PlasticWorkspace` instance it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.primitive import PrimitiveConfig


class AllocatorPreset(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """Named plastic-capacity sizes, per architecture doc section 7."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class PresetSpec:
    """How many temporary transforms one preset allocates, and at what rank."""

    num_transforms: int
    rank: int

    def __post_init__(self) -> None:
        if self.num_transforms < 1:
            raise ValueError(f"num_transforms must be >= 1, got {self.num_transforms}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


DEFAULT_PRESETS: dict[AllocatorPreset, PresetSpec] = {
    AllocatorPreset.SMALL: PresetSpec(num_transforms=4, rank=4),
    AllocatorPreset.MEDIUM: PresetSpec(num_transforms=8, rank=8),
    AllocatorPreset.LARGE: PresetSpec(num_transforms=16, rank=16),
}


class Allocator:
    """Resolves a named preset to a transform count/rank and allocates it
    into a `PlasticWorkspace`."""

    def __init__(
        self,
        d_model: int,
        presets: dict[AllocatorPreset, PresetSpec] | None = None,
    ) -> None:
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        self.d_model = d_model
        self.presets = dict(presets) if presets is not None else dict(DEFAULT_PRESETS)
        missing = [p for p in AllocatorPreset if p not in self.presets]
        if missing:
            raise ValueError(f"presets is missing entries for {[p.value for p in missing]}")

    def spec_for(self, preset: AllocatorPreset) -> PresetSpec:
        return self.presets[preset]

    def allocate(
        self,
        workspace: PlasticWorkspace,
        preset: AllocatorPreset,
        *,
        created_at_task: int = 0,
    ) -> list[int]:
        """Allocate `preset`'s transform count/rank into `workspace`.

        Raises whatever `PlasticWorkspace.allocate` raises (e.g. if the
        workspace already holds a batch) -- this method only resolves the
        preset, it does not change allocation semantics.
        """
        spec = self.spec_for(preset)
        config = PrimitiveConfig(d_model=self.d_model, rank=spec.rank)
        return workspace.allocate(
            preset, spec.num_transforms, config, created_at_task=created_at_task
        )
