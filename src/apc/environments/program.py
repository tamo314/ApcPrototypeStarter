"""Program representation: a concrete, ordered chain of operation steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProgramStep:
    """One concrete operation application: a name plus sampled parameters."""

    operation: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "params": dict(self.params)}


@dataclass(frozen=True)
class Program:
    """An ordered chain of `ProgramStep`s applied left to right."""

    steps: tuple[ProgramStep, ...]

    @property
    def operation_sequence(self) -> tuple[str, ...]:
        return tuple(step.operation for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}
