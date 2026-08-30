"""Finite-state novelty controller with hysteresis (Phase A Milestone A6 / Task 008).

Implements the controller from `docs/design-docs/ARCHITECTURE.md` section 9
and ADR-0003: a deterministic finite-state controller (no learned weights,
no RL) that walks the `STABLE -> SEARCH -> PLASTIC -> CONSOLIDATE -> SHADOW
-> STABLE` loop from section 2, using separate enter/exit thresholds on the
novelty score ("hysteresis", per section 9: "Use separate enter/exit
thresholds to avoid mode oscillation") to damp rapid flip-flopping.

Task 008 scope: the STABLE<->SEARCH<->PLASTIC boundary is driven by
`apc.meta.novelty.NoveltyEstimator` output (error + router entropy only,
per Milestone A6) and a bounded SEARCH step count. CONSOLIDATE and SHADOW
exist as state-machine plumbing for Task 010 (consolidation) and Task 011
(shadow validation) to drive via `ControllerSignals.candidate_ready` /
`shadow_passed`; this module does not implement consolidation or shadow
validation itself.

Assumption not pinned down by the architecture doc's state diagram (recorded
here rather than hidden, per AGENTS.md workflow): a failed shadow validation
returns to PLASTIC to keep learning rather than halting in SHADOW forever.
This matches Task 011's acceptance criterion that "a failing shadow test
preserves temporary capacity" (docs/TASKS.md) -- capacity is preserved
because PLASTIC keeps it allocated, not because SHADOW blocks release some
other way.

Every state change is appended to `Controller.transition_log` as a
`Transition`: step, old/new state, a human-readable trigger, the metrics
that drove the decision, and the allocated/released parameter count for
that step, per Milestone A6's "Required transition logging". The controller
never allocates or releases capacity itself -- it only records what the
caller reports happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

TriggerMetrics = dict[str, float | int | bool]


class ControllerState(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    """States of the architecture doc section 2 state machine."""

    STABLE = "stable"
    SEARCH = "search"
    PLASTIC = "plastic"
    CONSOLIDATE = "consolidate"
    SHADOW = "shadow"


@dataclass(frozen=True)
class ControllerConfig:
    """Explicit, serializable thresholds for `Controller`.

    `enter_search_threshold` / `exit_search_threshold` implement the
    hysteresis band for the STABLE<->SEARCH boundary: STABLE only leaves
    for SEARCH once novelty >= `enter_search_threshold`; SEARCH only
    resolves back to STABLE once novelty <= `exit_search_threshold`. A
    genuine band (`exit < enter`) is what damps oscillation; `exit ==
    enter` degenerates to a single-threshold controller (kept legal so
    tests can use it as the "no hysteresis" baseline), and `exit > enter`
    is rejected as nonsensical (there would be novelty values for which
    SEARCH both must and must not resolve immediately).
    """

    enter_search_threshold: float = 0.6
    exit_search_threshold: float = 0.4
    max_search_steps: int = 5
    plastic_plateau_patience: int = 3
    plastic_min_accuracy: float = 0.8

    def __post_init__(self) -> None:
        if self.exit_search_threshold > self.enter_search_threshold:
            raise ValueError(
                "exit_search_threshold must be <= enter_search_threshold, got "
                f"{self.exit_search_threshold} > {self.enter_search_threshold}"
            )
        if self.max_search_steps < 1:
            raise ValueError(f"max_search_steps must be >= 1, got {self.max_search_steps}")
        if self.plastic_plateau_patience < 1:
            raise ValueError(
                f"plastic_plateau_patience must be >= 1, got {self.plastic_plateau_patience}"
            )
        if not 0.0 <= self.plastic_min_accuracy <= 1.0:
            raise ValueError(
                f"plastic_min_accuracy must be in [0, 1], got {self.plastic_min_accuracy}"
            )


@dataclass(frozen=True)
class ControllerSignals:
    """One step's external inputs to `Controller.step`.

    Which fields are read depends on `Controller.state`; fields irrelevant
    to the current state are ignored (see the per-state `_decide_*`
    docstrings below). `allocated_params`/`released_params` are read every
    step regardless of state and are just recorded on whatever `Transition`
    (if any) occurs that step -- the controller does not allocate or
    release capacity itself, callers report what they did.
    """

    novelty: float = 0.0  # used in STABLE, SEARCH
    plastic_improved: bool = False  # used in PLASTIC
    plastic_accuracy: float = 0.0  # used in PLASTIC
    candidate_ready: bool = False  # used in CONSOLIDATE
    shadow_passed: bool | None = None  # used in SHADOW; None = validation still in progress
    allocated_params: int = 0
    released_params: int = 0


@dataclass(frozen=True)
class Transition:
    """One logged state change, per Milestone A6's required transition-log fields."""

    step: int
    old_state: ControllerState
    new_state: ControllerState
    trigger: str
    metrics: TriggerMetrics
    allocated_params: int = 0
    released_params: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "old_state": self.old_state.value,
            "new_state": self.new_state.value,
            "trigger": self.trigger,
            "metrics": dict(self.metrics),
            "allocated_params": self.allocated_params,
            "released_params": self.released_params,
        }


_Decision = tuple[ControllerState, str, TriggerMetrics]


class Controller:
    """Deterministic finite-state controller (ADR-0003): no learned weights, no RL."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config if config is not None else ControllerConfig()
        self.state = ControllerState.STABLE
        self.step_count = 0
        self.transition_log: list[Transition] = []
        self._search_steps = 0
        self._plastic_plateau_steps = 0

    def step(self, signals: ControllerSignals) -> Transition | None:
        """Advance the controller by one control step.

        Returns the `Transition` if the state changed this step, else
        `None`. `step_count` increments on every call regardless of
        whether a transition occurred.
        """
        self.step_count += 1
        decision = self._decide(signals)
        if decision is None:
            self._update_counters_without_transition(signals)
            return None

        new_state, trigger, metrics = decision
        transition = Transition(
            step=self.step_count,
            old_state=self.state,
            new_state=new_state,
            trigger=trigger,
            metrics=metrics,
            allocated_params=signals.allocated_params,
            released_params=signals.released_params,
        )
        self.transition_log.append(transition)
        self.state = new_state
        self._reset_counters()
        return transition

    def _decide(self, signals: ControllerSignals) -> _Decision | None:
        if self.state == ControllerState.STABLE:
            return self._decide_stable(signals)
        if self.state == ControllerState.SEARCH:
            return self._decide_search(signals)
        if self.state == ControllerState.PLASTIC:
            return self._decide_plastic(signals)
        if self.state == ControllerState.CONSOLIDATE:
            return self._decide_consolidate(signals)
        if self.state == ControllerState.SHADOW:
            return self._decide_shadow(signals)
        raise AssertionError(f"Unhandled controller state {self.state}")

    def _decide_stable(self, signals: ControllerSignals) -> _Decision | None:
        """STABLE -> SEARCH once novelty crosses the enter threshold."""
        if signals.novelty >= self.config.enter_search_threshold:
            return (
                ControllerState.SEARCH,
                "novelty_above_enter_search_threshold",
                {
                    "novelty": signals.novelty,
                    "enter_search_threshold": self.config.enter_search_threshold,
                },
            )
        return None

    def _decide_search(self, signals: ControllerSignals) -> _Decision | None:
        """SEARCH resolves back to STABLE once novelty drops to the exit
        threshold (architecture doc: "existing composition succeeds");
        otherwise it is bounded by `max_search_steps` attempts before
        escalating to PLASTIC ("bounded search failure")."""
        if signals.novelty <= self.config.exit_search_threshold:
            return (
                ControllerState.STABLE,
                "composition_found",
                {
                    "novelty": signals.novelty,
                    "exit_search_threshold": self.config.exit_search_threshold,
                    "search_steps": self._search_steps,
                },
            )
        if self._search_steps >= self.config.max_search_steps:
            return (
                ControllerState.PLASTIC,
                "search_bounded_failure",
                {
                    "novelty": signals.novelty,
                    "search_steps": self._search_steps,
                    "max_search_steps": self.config.max_search_steps,
                },
            )
        return None

    def _decide_plastic(self, signals: ControllerSignals) -> _Decision | None:
        """PLASTIC -> CONSOLIDATE once improvement has plateaued for
        `plastic_plateau_patience` steps and accuracy clears the minimum
        (architecture doc: "leave PLASTIC when improvement-per-compute
        plateaus and validation accuracy exceeds a minimum")."""
        plateaued = self._plastic_plateau_steps >= self.config.plastic_plateau_patience
        if plateaued and signals.plastic_accuracy >= self.config.plastic_min_accuracy:
            return (
                ControllerState.CONSOLIDATE,
                "plastic_plateaued",
                {
                    "plastic_accuracy": signals.plastic_accuracy,
                    "plastic_min_accuracy": self.config.plastic_min_accuracy,
                    "plateau_steps": self._plastic_plateau_steps,
                },
            )
        return None

    def _decide_consolidate(self, signals: ControllerSignals) -> _Decision | None:
        """CONSOLIDATE -> SHADOW once a candidate primitive has been produced."""
        if signals.candidate_ready:
            return (
                ControllerState.SHADOW,
                "candidate_ready",
                {"candidate_ready": signals.candidate_ready},
            )
        return None

    def _decide_shadow(self, signals: ControllerSignals) -> _Decision | None:
        """SHADOW -> STABLE on validation pass (architecture doc: "enter
        STABLE only after shadow validation and retention checks pass"),
        releasing temporary capacity; SHADOW -> PLASTIC on failure, keeping
        temporary capacity allocated (see module docstring). `None` means
        validation is still in progress -- no transition yet."""
        if signals.shadow_passed is None:
            return None
        if signals.shadow_passed:
            return (ControllerState.STABLE, "shadow_passed", {"shadow_passed": True})
        return (ControllerState.PLASTIC, "shadow_failed", {"shadow_passed": False})

    def _update_counters_without_transition(self, signals: ControllerSignals) -> None:
        if self.state == ControllerState.SEARCH:
            self._search_steps += 1
        elif self.state == ControllerState.PLASTIC:
            if signals.plastic_improved:
                self._plastic_plateau_steps = 0
            else:
                self._plastic_plateau_steps += 1

    def _reset_counters(self) -> None:
        self._search_steps = 0
        self._plastic_plateau_steps = 0
