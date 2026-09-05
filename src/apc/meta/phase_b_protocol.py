"""Phase B Protocol Specification and Baseline Freeze (Task B-C001).

Defines the serialized protocol contract for Phase B:
- Semantic Task Inference & Open-World Extension
as specified in `docs/design-docs/OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md`,
`docs/design-docs/TASK_INFERENCE_PHASE_B.md`,
`docs/design-docs/HARD_NEGATIVE_ROUTING_PHASE_B.md`, and
`docs/design-docs/DECISION_SEARCH_SCALING_PHASE_B.md`.

Invariants:
1. Upper Bound Reproducibility:
   The default explicit TaskSpec configuration precisely mirrors Phase A.2's
   verified task-visible routing and lifecycle baseline.
2. Identifiability & Leak Isolation:
   Non-explicit modalities forbid canonical operation ID visibility.
3. Strict Serialization:
   Complete round-trip dataclass <-> dict <-> YAML serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from apc.meta.learned_controller import AdequacyControllerConfig
from apc.primitives.incremental_router import (
    IncrementalRouterConfig,
    IncrementalUpdateCondition,
)


def _get_default_plastic_lifecycle_config() -> Any:
    from apc.plastic.lifecycle import CompactLifecycleConfig

    return CompactLifecycleConfig()


class FamilySplit(str, Enum):  # noqa: UP042
    """Operation family research status partition (ADR-0074 / Phase B design)."""

    DEV_FAMILIES = "DEV_FAMILIES"
    SEALED_FAMILIES = "SEALED_FAMILIES"
    RETIRED_FROM_SEALED = "RETIRED_FROM_SEALED"

    @classmethod
    def from_str(cls, value: str) -> FamilySplit:
        normalized = value.strip().upper()
        if normalized in ("DEV", "DEVELOPMENT", "DEV_FAMILIES"):
            return cls.DEV_FAMILIES
        if normalized in ("SEALED", "SEALED_EVALUATION", "SEALED_FAMILIES"):
            return cls.SEALED_FAMILIES
        if normalized in ("RETIRED", "RETIRED_FROM_SEALED"):
            return cls.RETIRED_FROM_SEALED
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown family split: {value!r}. Valid: {[m.value for m in cls]}")


class TaskInferenceModality(str, Enum):  # noqa: UP042
    """Observation modality providing task-side conditioning to APC."""

    EXPLICIT_TASK_SPEC = "explicit_task_spec"
    STRUCTURED_DESCRIPTOR = "structured_descriptor"
    FEWSHOT_DEMONSTRATIONS = "fewshot_demonstrations"
    NATURAL_LANGUAGE = "natural_language"

    @classmethod
    def from_str(cls, value: str) -> TaskInferenceModality:
        normalized = value.strip().lower()
        alias_map = {
            "explicit_task_spec": cls.EXPLICIT_TASK_SPEC,
            "explicit": cls.EXPLICIT_TASK_SPEC,
            "task_spec": cls.EXPLICIT_TASK_SPEC,
            "structured_descriptor": cls.STRUCTURED_DESCRIPTOR,
            "descriptor": cls.STRUCTURED_DESCRIPTOR,
            "structured": cls.STRUCTURED_DESCRIPTOR,
            "fewshot_demonstrations": cls.FEWSHOT_DEMONSTRATIONS,
            "demonstrations": cls.FEWSHOT_DEMONSTRATIONS,
            "fewshot": cls.FEWSHOT_DEMONSTRATIONS,
            "natural_language": cls.NATURAL_LANGUAGE,
            "language": cls.NATURAL_LANGUAGE,
            "instruction": cls.NATURAL_LANGUAGE,
        }
        if normalized in alias_map:
            return alias_map[normalized]
        raise ValueError(
            f"Unknown task inference modality: {value!r}. Valid: {[m.value for m in cls]}"
        )


class HardNegativeLevel(str, Enum):  # noqa: UP042
    """Retrieval competition difficulty ladder for primitive routing."""

    L0_ORTHOGONAL = "L0_orthogonal"
    L1_RANDOM_SCORE_SPACE = "L1_random_score_space"
    L2_NEAR_NEIGHBOR = "L2_near_neighbor"
    L3_SEMANTICALLY_RELATED = "L3_semantically_related"
    L4_CONFUSABLE_FAMILY = "L4_confusable_family"

    @classmethod
    def from_str(cls, value: str) -> HardNegativeLevel:
        normalized = value.strip()
        alias_map = {
            "L0": cls.L0_ORTHOGONAL,
            "L0_orthogonal": cls.L0_ORTHOGONAL,
            "orthogonal": cls.L0_ORTHOGONAL,
            "L1": cls.L1_RANDOM_SCORE_SPACE,
            "L1_random_score_space": cls.L1_RANDOM_SCORE_SPACE,
            "random": cls.L1_RANDOM_SCORE_SPACE,
            "L2": cls.L2_NEAR_NEIGHBOR,
            "L2_near_neighbor": cls.L2_NEAR_NEIGHBOR,
            "near_neighbor": cls.L2_NEAR_NEIGHBOR,
            "L3": cls.L3_SEMANTICALLY_RELATED,
            "L3_semantically_related": cls.L3_SEMANTICALLY_RELATED,
            "semantically_related": cls.L3_SEMANTICALLY_RELATED,
            "L4": cls.L4_CONFUSABLE_FAMILY,
            "L4_confusable_family": cls.L4_CONFUSABLE_FAMILY,
            "confusable": cls.L4_CONFUSABLE_FAMILY,
        }
        if normalized in alias_map:
            return alias_map[normalized]
        for member in cls:
            if member.value.lower() == normalized.lower():
                return member
        raise ValueError(f"Unknown hard-negative level: {value!r}. Valid: {[m.value for m in cls]}")


@dataclass(frozen=True)
class CandidateSearchBudget:
    """Bounded candidate evaluation and composition search budget."""

    max_candidates: int = 8
    beam_width: int = 4
    max_composition_depth: int = 2
    max_direct_evaluations: int = 16
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.max_composition_depth < 1:
            raise ValueError(
                f"max_composition_depth must be >= 1, got {self.max_composition_depth}"
            )
        if self.max_direct_evaluations < 1:
            raise ValueError(
                f"max_direct_evaluations must be >= 1, got {self.max_direct_evaluations}"
            )
        if self.timeout_seconds <= 0.0:
            raise ValueError(f"timeout_seconds must be > 0.0, got {self.timeout_seconds}")


@dataclass(frozen=True)
class PhaseBProtocol:
    """Serializable experiment and benchmark protocol specification for Phase B."""

    family_split: FamilySplit = FamilySplit.DEV_FAMILIES
    explicit_task_spec_visible: bool = True
    operation_id_visible: bool = True
    task_inference_modality: TaskInferenceModality = TaskInferenceModality.EXPLICIT_TASK_SPEC

    # Partition counts for support / task-inference / verification / query evaluation
    support_count: int = 16
    inference_count: int = 16
    verification_count: int = 32
    query_count: int = 64

    # Routing competition difficulty and bank bounds
    hard_negative_level: HardNegativeLevel = HardNegativeLevel.L0_ORTHOGONAL
    bank_size: int = 10
    candidate_search_budget: CandidateSearchBudget = field(default_factory=CandidateSearchBudget)

    # Frozen baseline configurations preserving Phase A.2 verified mechanics
    adequacy_controller_config: AdequacyControllerConfig = field(
        default_factory=AdequacyControllerConfig
    )
    plastic_lifecycle_config: Any = field(
        default_factory=_get_default_plastic_lifecycle_config
    )
    incremental_router_config: IncrementalRouterConfig = field(
        default_factory=IncrementalRouterConfig
    )

    def __post_init__(self) -> None:
        # 1. Modality & TaskSpec visibility invariants
        if self.task_inference_modality == TaskInferenceModality.EXPLICIT_TASK_SPEC:
            if not self.explicit_task_spec_visible:
                raise ValueError(
                    "When task_inference_modality is EXPLICIT_TASK_SPEC, "
                    "explicit_task_spec_visible must be True."
                )
            if not self.operation_id_visible:
                raise ValueError(
                    "When task_inference_modality is EXPLICIT_TASK_SPEC (upper bound), "
                    "operation_id_visible must be True."
                )
        else:
            if self.operation_id_visible:
                raise ValueError(
                    f"Task inference modality {self.task_inference_modality.value!r} represents "
                    "no-canonical-ID inference; operation_id_visible must be False to prevent "
                    "oracle shortcut leakage."
                )

        # 2. Partition counts validation
        if self.support_count < 1:
            raise ValueError(f"support_count must be >= 1, got {self.support_count}")
        if self.task_inference_modality != TaskInferenceModality.EXPLICIT_TASK_SPEC:
            if self.inference_count < 1:
                raise ValueError(
                    f"inference_count must be >= 1 for modality "
                    f"{self.task_inference_modality.value}, got {self.inference_count}"
                )
        else:
            if self.inference_count < 0:
                raise ValueError(f"inference_count must be >= 0, got {self.inference_count}")
        if self.verification_count < 1:
            raise ValueError(f"verification_count must be >= 1, got {self.verification_count}")
        if self.query_count < 1:
            raise ValueError(f"query_count must be >= 1, got {self.query_count}")

        # 3. Bank size
        if self.bank_size < 1:
            raise ValueError(f"bank_size must be >= 1, got {self.bank_size}")

    def to_dict(self) -> dict[str, Any]:
        """Convert protocol configuration to serializable dictionary."""
        router_dict = asdict(self.incremental_router_config)
        if isinstance(self.incremental_router_config.condition, Enum):
            router_dict["condition"] = self.incremental_router_config.condition.value

        return {
            "family_split": self.family_split.value,
            "explicit_task_spec_visible": self.explicit_task_spec_visible,
            "operation_id_visible": self.operation_id_visible,
            "task_inference_modality": self.task_inference_modality.value,
            "support_count": self.support_count,
            "inference_count": self.inference_count,
            "verification_count": self.verification_count,
            "query_count": self.query_count,
            "hard_negative_level": self.hard_negative_level.value,
            "bank_size": self.bank_size,
            "candidate_search_budget": asdict(self.candidate_search_budget),
            "adequacy_controller_config": asdict(self.adequacy_controller_config),
            "plastic_lifecycle_config": asdict(self.plastic_lifecycle_config),
            "incremental_router_config": router_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseBProtocol:
        """Construct protocol configuration from dictionary."""
        split_val = data.get("family_split", FamilySplit.DEV_FAMILIES.value)
        family_split = FamilySplit.from_str(split_val)
        modality = TaskInferenceModality.from_str(
            data.get("task_inference_modality", TaskInferenceModality.EXPLICIT_TASK_SPEC.value)
        )
        hn_level = HardNegativeLevel.from_str(
            data.get("hard_negative_level", HardNegativeLevel.L0_ORTHOGONAL.value)
        )

        budget_data = data.get("candidate_search_budget", {})
        budget = (
            CandidateSearchBudget(**budget_data)
            if isinstance(budget_data, dict)
            else CandidateSearchBudget()
        )

        adequacy_data = data.get("adequacy_controller_config", {})
        adequacy_cfg = (
            AdequacyControllerConfig(**adequacy_data)
            if isinstance(adequacy_data, dict)
            else AdequacyControllerConfig()
        )

        plastic_data = data.get("plastic_lifecycle_config", {})
        if isinstance(plastic_data, dict):
            from apc.plastic.lifecycle import CompactLifecycleConfig

            plastic_cfg = CompactLifecycleConfig(**plastic_data)
        else:
            plastic_cfg = _get_default_plastic_lifecycle_config()

        router_data = data.get("incremental_router_config", {})
        if isinstance(router_data, dict) and router_data:
            router_kwargs = dict(router_data)
            default_cond = IncrementalUpdateCondition.R2_BOUNDED_REPLAY.value
            cond_val = router_kwargs.get("condition", default_cond)
            if isinstance(cond_val, str):
                try:
                    router_kwargs["condition"] = IncrementalUpdateCondition(cond_val)
                except ValueError:
                    router_kwargs["condition"] = IncrementalUpdateCondition.R2_BOUNDED_REPLAY
            router_cfg = IncrementalRouterConfig(**router_kwargs)
        else:
            router_cfg = IncrementalRouterConfig()


        return cls(
            family_split=family_split,
            explicit_task_spec_visible=bool(data.get("explicit_task_spec_visible", True)),
            operation_id_visible=bool(data.get("operation_id_visible", True)),
            task_inference_modality=modality,
            support_count=int(data.get("support_count", 16)),
            inference_count=int(data.get("inference_count", 16)),
            verification_count=int(data.get("verification_count", 32)),
            query_count=int(data.get("query_count", 64)),
            hard_negative_level=hn_level,
            bank_size=int(data.get("bank_size", 10)),
            candidate_search_budget=budget,
            adequacy_controller_config=adequacy_cfg,
            plastic_lifecycle_config=plastic_cfg,
            incremental_router_config=router_cfg,
        )

    def to_yaml(self, path: Path | str) -> None:
        """Write configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PhaseBProtocol:
        """Load configuration from a YAML file."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Config file not found: {source}")
        with source.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {source}")
        return cls.from_dict(data)


def create_frozen_phase_a2_upper_bound(
    family_split: FamilySplit | str = FamilySplit.DEV_FAMILIES,
    bank_size: int = 10,
) -> PhaseBProtocol:
    """Create the frozen Phase A.2 upper-bound protocol configuration for Phase B.

    Ensures exact reproducibility of Phase A.2's explicit model-visible TaskSpec
    path, validated controller policy, compact-first plasticity, and bounded replay.
    """
    if isinstance(family_split, str):
        split = FamilySplit.from_str(family_split)
    else:
        split = family_split

    return PhaseBProtocol(
        family_split=split,
        explicit_task_spec_visible=True,
        operation_id_visible=True,
        task_inference_modality=TaskInferenceModality.EXPLICIT_TASK_SPEC,
        support_count=16,
        inference_count=0,  # Not used in explicit TaskSpec mode
        verification_count=32,
        query_count=64,
        hard_negative_level=HardNegativeLevel.L0_ORTHOGONAL,
        bank_size=bank_size,
        candidate_search_budget=CandidateSearchBudget(
            max_candidates=8,
            beam_width=4,
            max_composition_depth=2,
            max_direct_evaluations=16,
            timeout_seconds=60.0,
        ),
        adequacy_controller_config=AdequacyControllerConfig(),
        plastic_lifecycle_config=_get_default_plastic_lifecycle_config(),
        incremental_router_config=IncrementalRouterConfig(),
    )
