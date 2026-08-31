from __future__ import annotations

import dataclasses
import inspect

import pytest

from apc.meta import controller as controller_module
from apc.meta.controller import (
    Controller,
    ControllerConfig,
    ControllerSignals,
    ControllerState,
    Transition,
)

DEFAULT_CONFIG = ControllerConfig(
    enter_search_threshold=0.6,
    exit_search_threshold=0.4,
    max_search_steps=3,
    plastic_plateau_patience=2,
    plastic_min_accuracy=0.8,
)


def _drive_search_to_plastic(controller: Controller, novelty: float = 0.5) -> Transition:
    """Feed novelty in the ambiguous band (`exit < novelty < enter`) until
    the bounded SEARCH failure escalates to PLASTIC."""
    for _ in range(controller.config.max_search_steps + 2):
        transition = controller.step(ControllerSignals(novelty=novelty))
        if transition is not None:
            return transition
    raise AssertionError("SEARCH never escalated to PLASTIC")


def _drive_plastic_to_consolidate(controller: Controller, accuracy: float = 0.95) -> Transition:
    for _ in range(controller.config.plastic_plateau_patience + 2):
        transition = controller.step(
            ControllerSignals(plastic_improved=False, plastic_accuracy=accuracy)
        )
        if transition is not None:
            return transition
    raise AssertionError("PLASTIC never promoted to CONSOLIDATE")


def test_starts_stable_with_empty_log() -> None:
    controller = Controller(DEFAULT_CONFIG)
    assert controller.state == ControllerState.STABLE
    assert controller.transition_log == []
    assert controller.step_count == 0


def test_step_count_increments_even_without_transition() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.1))
    controller.step(ControllerSignals(novelty=0.1))
    assert controller.step_count == 2


# --- STABLE ------------------------------------------------------------
# Milestone A6 acceptance: "known tasks stay STABLE".


def test_stable_stays_stable_for_low_novelty() -> None:
    controller = Controller(DEFAULT_CONFIG)
    for _ in range(5):
        transition = controller.step(ControllerSignals(novelty=0.1))
        assert transition is None
    assert controller.state == ControllerState.STABLE
    assert controller.transition_log == []


def test_stable_enters_search_when_novelty_crosses_enter_threshold() -> None:
    controller = Controller(DEFAULT_CONFIG)
    transition = controller.step(ControllerSignals(novelty=0.7))
    assert transition is not None
    assert transition.old_state == ControllerState.STABLE
    assert transition.new_state == ControllerState.SEARCH
    assert transition.trigger == "novelty_above_enter_search_threshold"
    assert transition.metrics["novelty"] == pytest.approx(0.7)
    assert transition.step == 1
    assert controller.state == ControllerState.SEARCH


# --- SEARCH --------------------------------------------------------------
# Milestone A6 acceptance: "held-out composition enters SEARCH before
# PLASTIC" / "novel operation reaches PLASTIC after bounded SEARCH failure".


def test_search_returns_to_stable_when_novelty_resolves() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    transition = controller.step(ControllerSignals(novelty=0.3))
    assert transition is not None
    assert transition.old_state == ControllerState.SEARCH
    assert transition.new_state == ControllerState.STABLE
    assert transition.trigger == "composition_found"
    assert controller.state == ControllerState.STABLE


def test_search_escalates_to_plastic_after_bounded_failure() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    transition = _drive_search_to_plastic(controller)
    assert transition.new_state == ControllerState.PLASTIC
    assert transition.trigger == "search_bounded_failure"
    assert transition.metrics["search_steps"] == DEFAULT_CONFIG.max_search_steps


def test_search_does_not_escalate_before_max_steps_reached() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    for _ in range(DEFAULT_CONFIG.max_search_steps):
        transition = controller.step(ControllerSignals(novelty=0.5))
        assert transition is None
    assert controller.state == ControllerState.SEARCH


def test_allocated_params_recorded_on_plastic_entry() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))
    for _ in range(DEFAULT_CONFIG.max_search_steps):
        controller.step(ControllerSignals(novelty=0.5))
    transition = controller.step(ControllerSignals(novelty=0.5, allocated_params=1234))
    assert transition is not None
    assert transition.new_state == ControllerState.PLASTIC
    assert transition.allocated_params == 1234
    assert transition.released_params == 0


def test_search_step_counter_resets_after_reentering_search() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    controller.step(ControllerSignals(novelty=0.5))  # ambiguous, no transition
    controller.step(ControllerSignals(novelty=0.3))  # resolved -> STABLE
    assert controller.state == ControllerState.STABLE

    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH again
    # Counter must have been reset; escalation still needs a fresh full bound.
    transition = _drive_search_to_plastic(controller)
    assert transition.metrics["search_steps"] == DEFAULT_CONFIG.max_search_steps


# --- PLASTIC / CONSOLIDATE / SHADOW --------------------------------------


def test_plastic_stays_while_improving() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.PLASTIC
    for _ in range(5):
        transition = controller.step(
            ControllerSignals(plastic_improved=True, plastic_accuracy=0.9)
        )
        assert transition is None
    assert controller.state == ControllerState.PLASTIC


def test_plastic_promotes_to_consolidate_after_plateau_and_min_accuracy() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.PLASTIC
    transition = _drive_plastic_to_consolidate(controller, accuracy=0.95)
    assert transition.new_state == ControllerState.CONSOLIDATE
    assert transition.trigger == "plastic_plateaued"
    assert transition.metrics["plateau_steps"] == DEFAULT_CONFIG.plastic_plateau_patience


def test_plastic_does_not_promote_below_min_accuracy() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.PLASTIC
    for _ in range(DEFAULT_CONFIG.plastic_plateau_patience + 5):
        transition = controller.step(
            ControllerSignals(plastic_improved=False, plastic_accuracy=0.1)
        )
        assert transition is None
    assert controller.state == ControllerState.PLASTIC


def test_consolidate_waits_for_candidate_then_moves_to_shadow() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.CONSOLIDATE
    assert controller.step(ControllerSignals(candidate_ready=False)) is None
    transition = controller.step(ControllerSignals(candidate_ready=True))
    assert transition is not None
    assert transition.new_state == ControllerState.SHADOW
    assert transition.trigger == "candidate_ready"


def test_shadow_in_progress_does_not_transition() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.SHADOW
    assert controller.step(ControllerSignals(shadow_passed=None)) is None
    assert controller.state == ControllerState.SHADOW


def test_shadow_pass_releases_to_stable_with_released_params() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.SHADOW
    transition = controller.step(ControllerSignals(shadow_passed=True, released_params=99))
    assert transition is not None
    assert transition.new_state == ControllerState.STABLE
    assert transition.trigger == "shadow_passed"
    assert transition.released_params == 99


def test_shadow_fail_returns_to_plastic_preserving_capacity() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.SHADOW
    transition = controller.step(ControllerSignals(shadow_passed=False))
    assert transition is not None
    assert transition.new_state == ControllerState.PLASTIC
    assert transition.trigger == "shadow_failed"
    assert transition.released_params == 0


# --- Full loop / transition log -------------------------------------------


def test_transition_log_covers_full_cycle_and_every_entry_has_required_fields() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # STABLE -> SEARCH
    _drive_search_to_plastic(controller)  # SEARCH -> PLASTIC
    _drive_plastic_to_consolidate(controller)  # PLASTIC -> CONSOLIDATE
    controller.step(ControllerSignals(candidate_ready=True))  # -> SHADOW
    controller.step(ControllerSignals(shadow_passed=True, released_params=42))  # -> STABLE

    states = [t.new_state for t in controller.transition_log]
    assert states == [
        ControllerState.SEARCH,
        ControllerState.PLASTIC,
        ControllerState.CONSOLIDATE,
        ControllerState.SHADOW,
        ControllerState.STABLE,
    ]
    assert controller.state == ControllerState.STABLE
    assert controller.transition_log[-1].released_params == 42

    required_keys = {
        "step",
        "old_state",
        "new_state",
        "trigger",
        "metrics",
        "allocated_params",
        "released_params",
    }
    for transition in controller.transition_log:
        as_dict = transition.to_dict()
        assert set(as_dict) == required_keys
        assert isinstance(as_dict["metrics"], dict) and as_dict["metrics"]
        assert isinstance(as_dict["old_state"], str)
        assert isinstance(as_dict["new_state"], str)

    # Steps are strictly increasing across the whole run.
    steps = [t.step for t in controller.transition_log]
    assert steps == sorted(steps)
    assert len(set(steps)) == len(steps)


def test_shadow_failure_then_retry_reaches_stable() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.state = ControllerState.SHADOW
    controller.step(ControllerSignals(shadow_passed=False))  # -> PLASTIC
    assert controller.state == ControllerState.PLASTIC

    _drive_plastic_to_consolidate(controller)
    controller.step(ControllerSignals(candidate_ready=True))
    transition = controller.step(ControllerSignals(shadow_passed=True, released_params=10))
    assert transition is not None
    assert transition.new_state == ControllerState.STABLE


# --- Config validation ------------------------------------------------------


def test_force_state_sets_state_without_logging_a_transition() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    controller.force_state(ControllerState.STABLE)
    assert controller.state == ControllerState.STABLE
    assert len(controller.transition_log) == 1  # only the earlier real transition


def test_force_state_resets_counters() -> None:
    controller = Controller(DEFAULT_CONFIG)
    controller.step(ControllerSignals(novelty=0.7))  # -> SEARCH
    controller.step(ControllerSignals(novelty=0.5))  # ambiguous, bumps _search_steps
    controller.force_state(ControllerState.SEARCH)
    # Counter must have been reset; escalation still needs a fresh full bound.
    transition = _drive_search_to_plastic(controller)
    assert transition.metrics["search_steps"] == DEFAULT_CONFIG.max_search_steps


def test_config_rejects_exit_above_enter_threshold() -> None:
    with pytest.raises(ValueError, match="exit_search_threshold"):
        ControllerConfig(enter_search_threshold=0.3, exit_search_threshold=0.5)


def test_config_allows_equal_thresholds_as_no_hysteresis_baseline() -> None:
    ControllerConfig(enter_search_threshold=0.5, exit_search_threshold=0.5)  # does not raise


def test_config_rejects_non_positive_max_search_steps() -> None:
    with pytest.raises(ValueError, match="max_search_steps"):
        ControllerConfig(max_search_steps=0)


def test_config_rejects_non_positive_plateau_patience() -> None:
    with pytest.raises(ValueError, match="plastic_plateau_patience"):
        ControllerConfig(plastic_plateau_patience=0)


def test_config_rejects_accuracy_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="plastic_min_accuracy"):
        ControllerConfig(plastic_min_accuracy=1.5)


# --- Milestone A6 / Task 008 acceptance: hysteresis prevents oscillation --


def _count_transitions_for(config: ControllerConfig, novelty_sequence: list[float]) -> int:
    controller = Controller(config)
    count = 0
    for novelty in novelty_sequence:
        transition = controller.step(ControllerSignals(novelty=novelty))
        if transition is not None:
            count += 1
    return count


def test_hysteresis_reduces_transitions_versus_single_threshold_on_noisy_signal() -> None:
    # A novelty trace that jitters back and forth around 0.5: without a
    # hysteresis band this flips STABLE<->SEARCH on nearly every step that
    # crosses 0.5. `max_search_steps` is set high so escalation to PLASTIC
    # never confounds the count -- this test is only about the STABLE/SEARCH
    # boundary.
    noisy_sequence = [0.3, 0.65, 0.5, 0.55, 0.45, 0.52, 0.48, 0.51, 0.49, 0.35]

    no_hysteresis = ControllerConfig(
        enter_search_threshold=0.5, exit_search_threshold=0.5, max_search_steps=100
    )
    with_hysteresis = ControllerConfig(
        enter_search_threshold=0.6, exit_search_threshold=0.4, max_search_steps=100
    )

    flips = _count_transitions_for(no_hysteresis, noisy_sequence)
    damped = _count_transitions_for(with_hysteresis, noisy_sequence)

    assert flips >= 4
    assert damped <= 2
    assert damped < flips


def test_hysteresis_keeps_state_in_search_through_the_ambiguous_band() -> None:
    with_hysteresis = ControllerConfig(
        enter_search_threshold=0.6, exit_search_threshold=0.4, max_search_steps=100
    )
    controller = Controller(with_hysteresis)
    controller.step(ControllerSignals(novelty=0.65))  # -> SEARCH
    for novelty in (0.5, 0.55, 0.45, 0.52, 0.48, 0.51, 0.49):
        transition = controller.step(ControllerSignals(novelty=novelty))
        assert transition is None
        assert controller.state == ControllerState.SEARCH
    transition = controller.step(ControllerSignals(novelty=0.35))
    assert transition is not None
    assert transition.new_state == ControllerState.STABLE


def test_controller_signals_carry_no_oracle_task_label() -> None:
    """Task 009 acceptance: oracle metadata (K/C/N/R, `Example.category`)
    must remain hidden from the controller. `ControllerSignals` only has
    scalar/bool novelty and bookkeeping fields -- never a category/label
    field an evaluator's oracle could leak through."""
    field_names = {f.name for f in dataclasses.fields(ControllerSignals)}
    assert field_names == {
        "novelty",
        "plastic_improved",
        "plastic_accuracy",
        "candidate_ready",
        "shadow_passed",
        "allocated_params",
        "released_params",
    }


def test_controller_module_does_not_import_the_environment_or_examples() -> None:
    """The controller must be reachable only through scalar signals, never
    through `apc.environments.generator.Example` or its oracle category."""
    source = inspect.getsource(controller_module)
    assert "environments" not in source
    assert "category" not in source
