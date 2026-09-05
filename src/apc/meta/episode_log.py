"""Controller instrumentation and sequential episode logging (Phase A.2 Task A2-C002).

Provides:
- `ControllerAction`: Enum specifying the three autonomous controller decisions
  (`DIRECT_REUSE`, `COMPOSE`, `PLASTIC_SEARCH`) defined in ADR-0061 and
  `docs/design-docs/AUTONOMOUS_CONTROLLER_POLICY.md`.
- `EpisodeRecord`: Dataclass capturing all 11 required per-episode runtime signals:
  controller action, proposed primitive ID, composition recipe, support direct score,
  support composition score, plastic trigger, bank size before/after, router version,
  adaptation steps, temporary parameters, and selected/unselected forward calls.
- `EpisodeLogger`: Accumulator for sequential streaming evaluation, computing summary
  diagnostics and JSON persistence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ControllerAction(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """Runtime action space for the Phase A.2 autonomous controller."""

    DIRECT_REUSE = "DIRECT_REUSE"
    COMPOSE = "COMPOSE"
    PLASTIC_SEARCH = "PLASTIC_SEARCH"


@dataclass(frozen=True)
class EpisodeRecord:
    """Complete per-episode instrumentation record for Phase A.2.

    Captures all required episode-level signals declared in
    `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md` (Task A2-C002):
    1. controller action
    2. proposed primitive ID
    3. composition recipe
    4. support-set direct score
    5. support-set composition score
    6. plastic trigger
    7. bank size before/after
    8. router version
    9. adaptation steps
    10. temporary params
    11. selected/unselected forward calls
    """

    episode_id: int
    controller_action: ControllerAction
    proposed_primitive_id: int | None = None
    composition_recipe: tuple[str, ...] | str | None = None
    support_set_direct_score: float | None = None
    support_set_composition_score: float | None = None
    plastic_trigger: bool = False
    bank_size_before: int = 0
    bank_size_after: int = 0
    router_version: int = 0
    adaptation_steps: int = 0
    temporary_params: int = 0
    selected_forward_calls: int = 0
    unselected_forward_calls: int = 0
    task_name: str | None = None
    evaluation_exact_match: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episode_id < 0:
            raise ValueError(f"episode_id must be non-negative, got {self.episode_id}")
        if self.bank_size_before < 0 or self.bank_size_after < 0:
            raise ValueError(
                f"bank sizes must be non-negative, got before={self.bank_size_before}, "
                f"after={self.bank_size_after}"
            )
        if self.bank_size_after < self.bank_size_before:
            raise ValueError(
                f"bank size cannot decrease across an episode: "
                f"{self.bank_size_after} < {self.bank_size_before}"
            )
        if self.router_version < 0:
            raise ValueError(f"router_version must be non-negative, got {self.router_version}")
        if self.adaptation_steps < 0:
            raise ValueError(f"adaptation_steps must be non-negative, got {self.adaptation_steps}")
        if self.temporary_params < 0:
            raise ValueError(f"temporary_params must be non-negative, got {self.temporary_params}")
        if self.selected_forward_calls < 0 or self.unselected_forward_calls < 0:
            raise ValueError(
                f"forward calls must be non-negative, got selected={self.selected_forward_calls}, "
                f"unselected={self.unselected_forward_calls}"
            )

    @property
    def is_sparse_valid(self) -> bool:
        """Strict sparse invariant: non-selected primitives receive 0 forward calls."""
        return self.unselected_forward_calls == 0

    @property
    def bank_expanded(self) -> bool:
        """True if the persistent primitive bank grew during this episode."""
        return self.bank_size_after > self.bank_size_before

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        data = asdict(self)
        data["controller_action"] = self.controller_action.value
        if isinstance(data.get("composition_recipe"), tuple):
            data["composition_recipe"] = list(data["composition_recipe"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        """Construct record from dictionary."""
        d = dict(data)
        d["controller_action"] = ControllerAction(d["controller_action"])
        if "composition_recipe" in d and isinstance(d["composition_recipe"], list):
            d["composition_recipe"] = tuple(d["composition_recipe"])
        return cls(**d)


class EpisodeLogger:
    """Accumulates and summarizes sequential episode records."""

    def __init__(self, records: Sequence[EpisodeRecord] | None = None) -> None:
        self.records: list[EpisodeRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self.records)

    def log(self, record: EpisodeRecord) -> None:
        """Append an episode record."""
        self.records.append(record)

    def filter_by_action(self, action: ControllerAction | str) -> list[EpisodeRecord]:
        """Return all records matching a specific controller action."""
        act = ControllerAction(action) if isinstance(action, str) else action
        return [r for r in self.records if r.controller_action == act]

    def summary(self) -> dict[str, Any]:
        """Compute cumulative diagnostics over logged episodes."""
        total = len(self.records)
        if total == 0:
            return {
                "total_episodes": 0,
                "direct_reuse_count": 0,
                "compose_count": 0,
                "plastic_search_count": 0,
                "plastic_trigger_count": 0,
                "bank_expansion_count": 0,
                "total_adaptation_steps": 0,
                "mean_adaptation_steps": 0.0,
                "sparse_violations": 0,
            }

        direct_reuse = sum(
            1 for r in self.records if r.controller_action == ControllerAction.DIRECT_REUSE
        )
        compose = sum(1 for r in self.records if r.controller_action == ControllerAction.COMPOSE)
        plastic = sum(
            1 for r in self.records if r.controller_action == ControllerAction.PLASTIC_SEARCH
        )
        plastic_triggers = sum(1 for r in self.records if r.plastic_trigger)
        bank_expansions = sum(1 for r in self.records if r.bank_expanded)
        total_steps = sum(r.adaptation_steps for r in self.records)
        sparse_violations = sum(1 for r in self.records if not r.is_sparse_valid)

        return {
            "total_episodes": total,
            "direct_reuse_count": direct_reuse,
            "compose_count": compose,
            "plastic_search_count": plastic,
            "plastic_trigger_count": plastic_triggers,
            "bank_expansion_count": bank_expansions,
            "total_adaptation_steps": total_steps,
            "mean_adaptation_steps": total_steps / total,
            "sparse_violations": sparse_violations,
        }

    def save_json(self, path: Path | str) -> None:
        """Serialize all records to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in self.records]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> EpisodeLogger:
        """Load records from a JSON file."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = [EpisodeRecord.from_dict(item) for item in payload]
        return cls(records)
