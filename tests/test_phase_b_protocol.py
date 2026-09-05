"""Tests for Phase B Protocol and Baseline Freeze (Task B-C001).

Verifies:
1. Protocol serialization round-trip (dataclass <-> dict <-> YAML).
2. Configuration validation rejecting impossible combinations.
3. Default frozen Phase A.2 upper-bound mode reproducing the intended A2-style task-visible path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.meta.phase_b_protocol import (
    CandidateSearchBudget,
    FamilySplit,
    HardNegativeLevel,
    PhaseBProtocol,
    TaskInferenceModality,
    create_frozen_phase_a2_upper_bound,
)
from apc.primitives.incremental_router import IncrementalUpdateCondition


def test_protocol_serialization_round_trip(tmp_path: Path) -> None:
    """Verify dataclass <-> dict <-> YAML serialization round-trip preserving all fields."""
    protocol = PhaseBProtocol(
        family_split=FamilySplit.SEALED_FAMILIES,
        explicit_task_spec_visible=False,
        operation_id_visible=False,
        task_inference_modality=TaskInferenceModality.STRUCTURED_DESCRIPTOR,
        support_count=12,
        inference_count=8,
        verification_count=24,
        query_count=48,
        hard_negative_level=HardNegativeLevel.L2_NEAR_NEIGHBOR,
        bank_size=16,
        candidate_search_budget=CandidateSearchBudget(
            max_candidates=10,
            beam_width=5,
            max_composition_depth=3,
            max_direct_evaluations=20,
            timeout_seconds=45.0,
        ),
    )

    # 1. Dict round-trip
    d = protocol.to_dict()
    reconstructed = PhaseBProtocol.from_dict(d)
    assert reconstructed == protocol

    # 2. YAML round-trip
    yaml_path = tmp_path / "test_protocol.yaml"
    protocol.to_yaml(yaml_path)
    from_file = PhaseBProtocol.from_yaml(yaml_path)
    assert from_file == protocol


def test_config_rejects_impossible_combinations() -> None:
    """Verify protocol validation strictly rejects invalid or contradictory configurations."""
    # 1. Explicit modality without TaskSpec visibility
    with pytest.raises(ValueError, match="explicit_task_spec_visible must be True"):
        PhaseBProtocol(
            task_inference_modality=TaskInferenceModality.EXPLICIT_TASK_SPEC,
            explicit_task_spec_visible=False,
            operation_id_visible=True,
        )

    # 2. Explicit modality without operation ID visibility
    with pytest.raises(ValueError, match="operation_id_visible must be True"):
        PhaseBProtocol(
            task_inference_modality=TaskInferenceModality.EXPLICIT_TASK_SPEC,
            explicit_task_spec_visible=True,
            operation_id_visible=False,
        )

    # 3. No-ID semantic modality with operation ID visible
    for modality in (
        TaskInferenceModality.STRUCTURED_DESCRIPTOR,
        TaskInferenceModality.FEWSHOT_DEMONSTRATIONS,
        TaskInferenceModality.NATURAL_LANGUAGE,
    ):
        with pytest.raises(ValueError, match="operation_id_visible must be False"):
            PhaseBProtocol(
                task_inference_modality=modality,
                explicit_task_spec_visible=False,
                operation_id_visible=True,
            )

    # 4. Invalid count fields
    with pytest.raises(ValueError, match="support_count must be >= 1"):
        PhaseBProtocol(support_count=0)

    with pytest.raises(ValueError, match="verification_count must be >= 1"):
        PhaseBProtocol(verification_count=0)

    with pytest.raises(ValueError, match="query_count must be >= 1"):
        PhaseBProtocol(query_count=0)

    with pytest.raises(ValueError, match="bank_size must be >= 1"):
        PhaseBProtocol(bank_size=0)

    # 5. Non-explicit modality requires positive inference_count
    with pytest.raises(ValueError, match="inference_count must be >= 1"):
        PhaseBProtocol(
            task_inference_modality=TaskInferenceModality.STRUCTURED_DESCRIPTOR,
            explicit_task_spec_visible=False,
            operation_id_visible=False,
            inference_count=0,
        )

    # 6. Invalid candidate search budget parameters
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        CandidateSearchBudget(max_candidates=0)

    with pytest.raises(ValueError, match="beam_width must be >= 1"):
        CandidateSearchBudget(beam_width=0)

    with pytest.raises(ValueError, match="max_composition_depth must be >= 1"):
        CandidateSearchBudget(max_composition_depth=0)

    with pytest.raises(ValueError, match="timeout_seconds must be > 0.0"):
        CandidateSearchBudget(timeout_seconds=0.0)


def test_default_frozen_upper_bound_mode() -> None:
    """Verify default frozen Phase A.2 upper bound reproduces intended task-visible path."""
    upper_bound = create_frozen_phase_a2_upper_bound(family_split=FamilySplit.DEV_FAMILIES)

    assert upper_bound.family_split == FamilySplit.DEV_FAMILIES
    assert upper_bound.explicit_task_spec_visible is True
    assert upper_bound.operation_id_visible is True
    assert upper_bound.task_inference_modality == TaskInferenceModality.EXPLICIT_TASK_SPEC
    assert upper_bound.hard_negative_level == HardNegativeLevel.L0_ORTHOGONAL
    assert upper_bound.bank_size == 10
    assert upper_bound.support_count == 16
    assert upper_bound.verification_count == 32
    assert upper_bound.query_count == 64

    # Verify controller and plastic policies are non-empty and retain A2 defaults
    assert upper_bound.adequacy_controller_config.hidden_dim == 32
    assert upper_bound.plastic_lifecycle_config.compact_budget_steps == 150
    assert upper_bound.plastic_lifecycle_config.enable_overcomplete_fallback is True
    assert (
        upper_bound.incremental_router_config.condition
        == IncrementalUpdateCondition.R2_BOUNDED_REPLAY
    )
    assert upper_bound.incremental_router_config.replay_max_per_class == 32
    assert upper_bound.incremental_router_config.replay_max_total == 512

    # Verify config file exists and matches factory output
    config_path = Path("configs/phase_b_baseline_upper_bound.yaml")
    assert config_path.is_file(), f"Config file missing at {config_path}"
    loaded_from_yaml = PhaseBProtocol.from_yaml(config_path)
    assert loaded_from_yaml == upper_bound


def test_enum_parsing_helpers() -> None:
    """Verify string parsing helpers for protocol enums handle synonyms and reject unknowns."""
    # FamilySplit
    assert FamilySplit.from_str("dev") == FamilySplit.DEV_FAMILIES
    assert FamilySplit.from_str("development") == FamilySplit.DEV_FAMILIES
    assert FamilySplit.from_str("DEV_FAMILIES") == FamilySplit.DEV_FAMILIES
    assert FamilySplit.from_str("sealed") == FamilySplit.SEALED_FAMILIES
    assert FamilySplit.from_str("retired") == FamilySplit.RETIRED_FROM_SEALED
    with pytest.raises(ValueError, match="Unknown family split"):
        FamilySplit.from_str("invalid_split")

    # TaskInferenceModality
    assert TaskInferenceModality.from_str("explicit") == TaskInferenceModality.EXPLICIT_TASK_SPEC
    assert TaskInferenceModality.from_str("task_spec") == TaskInferenceModality.EXPLICIT_TASK_SPEC
    desc_mod = TaskInferenceModality.from_str("structured")
    assert desc_mod == TaskInferenceModality.STRUCTURED_DESCRIPTOR
    demo_mod = TaskInferenceModality.from_str("demonstrations")
    assert demo_mod == TaskInferenceModality.FEWSHOT_DEMONSTRATIONS
    nl_mod = TaskInferenceModality.from_str("natural_language")
    assert nl_mod == TaskInferenceModality.NATURAL_LANGUAGE
    with pytest.raises(ValueError, match="Unknown task inference modality"):
        TaskInferenceModality.from_str("invalid_modality")

    # HardNegativeLevel
    assert HardNegativeLevel.from_str("L0") == HardNegativeLevel.L0_ORTHOGONAL
    assert HardNegativeLevel.from_str("orthogonal") == HardNegativeLevel.L0_ORTHOGONAL
    assert HardNegativeLevel.from_str("L2") == HardNegativeLevel.L2_NEAR_NEIGHBOR
    sr_level = HardNegativeLevel.from_str("semantically_related")
    assert sr_level == HardNegativeLevel.L3_SEMANTICALLY_RELATED
    assert HardNegativeLevel.from_str("confusable") == HardNegativeLevel.L4_CONFUSABLE_FAMILY
    with pytest.raises(ValueError, match="Unknown hard-negative level"):
        HardNegativeLevel.from_str("invalid_level")

