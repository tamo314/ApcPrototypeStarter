"""Unit tests for Phase A.2 Controller Instrumentation & Episode Logging (Task A2-C002)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apc.meta.episode_log import ControllerAction, EpisodeLogger, EpisodeRecord


def test_episode_record_creation_and_fields() -> None:
    """Verify that all 11 required per-episode signals are correctly stored."""
    record = EpisodeRecord(
        episode_id=0,
        controller_action=ControllerAction.DIRECT_REUSE,
        proposed_primitive_id=3,
        composition_recipe=None,
        support_set_direct_score=0.98,
        support_set_composition_score=None,
        plastic_trigger=False,
        bank_size_before=10,
        bank_size_after=10,
        router_version=1,
        adaptation_steps=0,
        temporary_params=0,
        selected_forward_calls=64,
        unselected_forward_calls=0,
        task_name="SHIFT",
        evaluation_exact_match=0.97,
    )

    assert record.episode_id == 0
    assert record.controller_action == ControllerAction.DIRECT_REUSE
    assert record.proposed_primitive_id == 3
    assert record.composition_recipe is None
    assert record.support_set_direct_score == 0.98
    assert record.support_set_composition_score is None
    assert record.plastic_trigger is False
    assert record.bank_size_before == 10
    assert record.bank_size_after == 10
    assert record.router_version == 1
    assert record.adaptation_steps == 0
    assert record.temporary_params == 0
    assert record.selected_forward_calls == 64
    assert record.unselected_forward_calls == 0
    assert record.is_sparse_valid is True
    assert record.bank_expanded is False


def test_episode_record_sparse_validation() -> None:
    """Verify sparse invariant flag catches non-zero unselected forward calls."""
    valid_record = EpisodeRecord(
        episode_id=1,
        controller_action=ControllerAction.COMPOSE,
        proposed_primitive_id=None,
        composition_recipe=("SHIFT", "REVERSE"),
        support_set_direct_score=0.10,
        support_set_composition_score=0.99,
        plastic_trigger=False,
        bank_size_before=10,
        bank_size_after=10,
        selected_forward_calls=128,
        unselected_forward_calls=0,
    )
    assert valid_record.is_sparse_valid is True

    invalid_record = EpisodeRecord(
        episode_id=2,
        controller_action=ControllerAction.DIRECT_REUSE,
        proposed_primitive_id=0,
        bank_size_before=10,
        bank_size_after=10,
        selected_forward_calls=64,
        unselected_forward_calls=12,  # Violation
    )
    assert invalid_record.is_sparse_valid is False


def test_episode_record_bank_expansion_property() -> None:
    """Verify bank_expanded flag tracks bank growth."""
    no_expansion = EpisodeRecord(
        episode_id=3,
        controller_action=ControllerAction.DIRECT_REUSE,
        bank_size_before=10,
        bank_size_after=10,
    )
    assert no_expansion.bank_expanded is False

    expansion = EpisodeRecord(
        episode_id=4,
        controller_action=ControllerAction.PLASTIC_SEARCH,
        plastic_trigger=True,
        bank_size_before=10,
        bank_size_after=11,
        adaptation_steps=50,
        temporary_params=18282,
    )
    assert expansion.bank_expanded is True


def test_episode_record_validation_errors() -> None:
    """Verify post-init validation rejects nonsensical values."""
    with pytest.raises(ValueError, match="episode_id must be non-negative"):
        EpisodeRecord(episode_id=-1, controller_action=ControllerAction.DIRECT_REUSE)

    with pytest.raises(ValueError, match="bank sizes must be non-negative"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            bank_size_before=-1,
        )

    with pytest.raises(ValueError, match="bank size cannot decrease"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            bank_size_before=10,
            bank_size_after=9,
        )

    with pytest.raises(ValueError, match="router_version must be non-negative"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            router_version=-1,
        )

    with pytest.raises(ValueError, match="adaptation_steps must be non-negative"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            adaptation_steps=-5,
        )

    with pytest.raises(ValueError, match="temporary_params must be non-negative"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            temporary_params=-100,
        )

    with pytest.raises(ValueError, match="forward calls must be non-negative"):
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            selected_forward_calls=-1,
        )


def test_episode_record_serialization_roundtrip() -> None:
    """Verify JSON dictionary conversion matches exactly."""
    record = EpisodeRecord(
        episode_id=42,
        controller_action=ControllerAction.PLASTIC_SEARCH,
        proposed_primitive_id=None,
        composition_recipe=("REVERSE", "SHIFT"),
        support_set_direct_score=0.15,
        support_set_composition_score=0.20,
        plastic_trigger=True,
        bank_size_before=10,
        bank_size_after=11,
        router_version=2,
        adaptation_steps=120,
        temporary_params=20480,
        selected_forward_calls=64,
        unselected_forward_calls=0,
        task_name="NOVEL_OP",
        evaluation_exact_match=0.92,
        metadata={"seed": 42, "lr": 0.001},
    )

    data = record.to_dict()
    assert data["controller_action"] == "PLASTIC_SEARCH"
    assert data["composition_recipe"] == ["REVERSE", "SHIFT"]

    # Roundtrip from dict
    restored = EpisodeRecord.from_dict(data)
    assert restored == record

    # JSON roundtrip
    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    restored_json = EpisodeRecord.from_dict(deserialized)
    assert restored_json == record


def test_episode_logger_aggregation_and_summary(tmp_path: Path) -> None:
    """Verify EpisodeLogger correctly aggregates stream metrics and writes/reads JSON."""
    logger = EpisodeLogger()

    # Episode 0: Known task (K) -> DIRECT_REUSE
    logger.log(
        EpisodeRecord(
            episode_id=0,
            controller_action=ControllerAction.DIRECT_REUSE,
            proposed_primitive_id=0,
            support_set_direct_score=0.99,
            bank_size_before=10,
            bank_size_after=10,
            selected_forward_calls=64,
            unselected_forward_calls=0,
        )
    )

    # Episode 1: Composition task (C) -> COMPOSE
    logger.log(
        EpisodeRecord(
            episode_id=1,
            controller_action=ControllerAction.COMPOSE,
            composition_recipe=("COPY", "REVERSE"),
            support_set_direct_score=0.20,
            support_set_composition_score=0.98,
            bank_size_before=10,
            bank_size_after=10,
            selected_forward_calls=128,
            unselected_forward_calls=0,
        )
    )

    # Episode 2: Novel task (N) -> PLASTIC_SEARCH, bank grows 10 -> 11
    logger.log(
        EpisodeRecord(
            episode_id=2,
            controller_action=ControllerAction.PLASTIC_SEARCH,
            support_set_direct_score=0.10,
            support_set_composition_score=0.15,
            plastic_trigger=True,
            bank_size_before=10,
            bank_size_after=11,
            adaptation_steps=80,
            temporary_params=18282,
            selected_forward_calls=64,
            unselected_forward_calls=0,
        )
    )

    # Episode 3: Recurrence task (R) -> DIRECT_REUSE of new primitive 10
    logger.log(
        EpisodeRecord(
            episode_id=3,
            controller_action=ControllerAction.DIRECT_REUSE,
            proposed_primitive_id=10,
            support_set_direct_score=0.97,
            bank_size_before=11,
            bank_size_after=11,
            adaptation_steps=0,
            temporary_params=0,
            selected_forward_calls=64,
            unselected_forward_calls=0,
        )
    )

    assert len(logger) == 4

    direct_records = logger.filter_by_action(ControllerAction.DIRECT_REUSE)
    assert len(direct_records) == 2

    compose_records = logger.filter_by_action("COMPOSE")
    assert len(compose_records) == 1

    summary = logger.summary()
    assert summary["total_episodes"] == 4
    assert summary["direct_reuse_count"] == 2
    assert summary["compose_count"] == 1
    assert summary["plastic_search_count"] == 1
    assert summary["plastic_trigger_count"] == 1
    assert summary["bank_expansion_count"] == 1
    assert summary["total_adaptation_steps"] == 80
    assert summary["mean_adaptation_steps"] == 20.0
    assert summary["sparse_violations"] == 0

    # Persistence roundtrip
    out_file = tmp_path / "episodes.json"
    logger.save_json(out_file)
    assert out_file.is_file()

    loaded_logger = EpisodeLogger.load_json(out_file)
    assert len(loaded_logger) == 4
    assert loaded_logger.summary() == summary
