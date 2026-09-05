"""Tests for Phase B Holdout-Family Registry, Identifiability Checks, and Leak Audit.

Task B-C002 Acceptance Criteria:
- generators are deterministic under seed;
- development/sealed partitions are disjoint;
- leak audit is zero on clean inputs and strictly detects leaks;
- at least one sealed operation passes novelty-validity precheck;
- ambiguous demonstration sets are detected rather than silently treated as model errors.
"""

from __future__ import annotations

import pytest

from apc.environments.generator import Example
from apc.environments.holdout_families import (
    DEFAULT_FAMILY_REGISTRY,
    FamilyMetadata,
    HoldoutFamilyRegistry,
    generate_holdout_episode,
    generate_recurrence_episode,
)
from apc.environments.interpreter import run_program
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.evaluation.holdout_protocol import (
    audit_for_leaks,
    check_identifiability,
    check_novelty_validity,
)
from apc.meta.phase_b_protocol import (
    FamilySplit,
    PhaseBProtocol,
    TaskInferenceModality,
)


def test_generators_deterministic_under_seed() -> None:
    """Verify that HoldoutEpisode generation is strictly deterministic given a seed."""
    protocol = PhaseBProtocol(
        support_count=8,
        inference_count=8,
        verification_count=16,
        query_count=32,
    )

    # 1. Run twice with identical seed -> exact match on all tokens and properties
    ep1 = generate_holdout_episode("MAJORITY_THREE", protocol, seed=123, seq_length=8)
    ep2 = generate_holdout_episode("MAJORITY_THREE", protocol, seed=123, seq_length=8)

    assert ep1.operation_name == ep2.operation_name == "MAJORITY_THREE"
    assert ep1.family_id == ep2.family_id == "sealed_local_neighborhood"
    assert ep1.category == ep2.category == "N"

    assert len(ep1.support_examples) == 8
    assert len(ep1.inference_examples) == 8
    assert len(ep1.verification_examples) == 16
    assert len(ep1.query_examples) == 32

    for ex1, ex2 in zip(ep1.all_examples, ep2.all_examples, strict=True):
        assert ex1.input_tokens == ex2.input_tokens
        assert ex1.target_tokens == ex2.target_tokens
        assert ex1.category == ex2.category == "N"

    # 2. Run with different seed -> produces different inputs
    ep3 = generate_holdout_episode("MAJORITY_THREE", protocol, seed=999, seq_length=8)
    assert ep3.support_examples[0].input_tokens != ep1.support_examples[0].input_tokens


def test_development_and_sealed_partitions_disjoint() -> None:
    """Verify DEV_FAMILIES, SEALED_FAMILIES, and RETIRED_FROM_SEALED partitions are disjoint."""
    registry = DEFAULT_FAMILY_REGISTRY
    registry.assert_disjoint_partitions()

    dev_ops = set(registry.list_operations(FamilySplit.DEV_FAMILIES))
    sealed_ops = set(registry.list_operations(FamilySplit.SEALED_FAMILIES))

    assert len(dev_ops) > 0
    assert len(sealed_ops) > 0
    assert dev_ops.isdisjoint(sealed_ops)

    # Verify operations match expected membership
    assert "DEV_DELTA_MOD" in dev_ops
    assert "DEV_WINDOW_SUM" in dev_ops
    assert "MAJORITY_THREE" in sealed_ops
    assert "NEIGHBOR_MAX" in sealed_ops
    assert "NEIGHBOR_CONDITIONAL" in sealed_ops

    # Verify duplicate assignment to a different family raises ValueError
    custom_reg = HoldoutFamilyRegistry()
    fam1 = FamilyMetadata(
        family_id="fam1",
        status=FamilySplit.DEV_FAMILIES,
        structural_dependency_type="test",
        output_shape_rule="same_length",
        argument_schema={},
        generator_version="1.0.0",
        description="test",
        operations=("DEV_DELTA_MOD",),
    )
    fam2 = FamilyMetadata(
        family_id="fam2",
        status=FamilySplit.SEALED_FAMILIES,
        structural_dependency_type="test",
        output_shape_rule="same_length",
        argument_schema={},
        generator_version="1.0.0",
        description="test",
        operations=("DEV_DELTA_MOD",),
    )
    custom_reg.register_family(fam1)
    with pytest.raises(ValueError, match="already assigned to family"):
        custom_reg.register_family(fam2)


def test_retire_sealed_family() -> None:
    """Verify retirement transition of a sealed family (ADR-0074)."""
    custom_reg = HoldoutFamilyRegistry()
    fam = FamilyMetadata(
        family_id="test_sealed_fam",
        status=FamilySplit.SEALED_FAMILIES,
        structural_dependency_type="test",
        output_shape_rule="same_length",
        argument_schema={},
        generator_version="1.0.0",
        description="Sealed test family",
        operations=("NEIGHBOR_MAX",),
    )
    custom_reg.register_family(fam)
    assert custom_reg.get_family("test_sealed_fam").status == FamilySplit.SEALED_FAMILIES

    retired = custom_reg.retire_sealed_family("test_sealed_fam", "Tuned threshold on test")
    assert retired.status == FamilySplit.RETIRED_FROM_SEALED
    assert "RETIRED: Tuned threshold on test" in retired.description
    assert custom_reg.get_family("test_sealed_fam").status == FamilySplit.RETIRED_FROM_SEALED


def test_sealed_operations_pass_novelty_validity() -> None:
    """Verify that all sealed evaluation operations pass the novelty-validity precheck.

    Confirms that neither direct primitive search nor depth-2 composition over
    the existing bank can achieve the adequacy threshold (0.90).
    """
    for op_name in ("MAJORITY_THREE", "NEIGHBOR_MAX", "NEIGHBOR_CONDITIONAL"):
        ep = generate_holdout_episode(op_name, seed=42, seq_length=8)
        res = check_novelty_validity(
            target_operation=op_name,
            verification_examples=ep.verification_examples,
            threshold=0.90,
            max_composition_depth=2,
        )

        assert res.is_valid_novel_holdout is True, f"{op_name} failed novelty check"
        assert res.adequate_by_existing_library is False
        assert res.best_direct_em < 0.90
        assert res.best_composition_em < 0.90


def test_novelty_validity_rejects_adequate_operations() -> None:
    """Verify that operations solvable by direct reuse or composition are flagged as adequate.

    A candidate task that already reaches the adequacy threshold is NOT a valid novel holdout.
    """
    # 1. Test direct known operation (e.g. NEGATE)
    prog_negate = Program(steps=(ProgramStep(operation="NEGATE", params={}),))
    task_spec = TaskSpec.from_program(prog_negate)
    examples: list[Example] = []
    for _i in range(16):
        inp = (1, 2, 3, 4, 5, 6, 7, 8)
        res = run_program(prog_negate, inp, vocab_size=10)
        examples.append(
            Example(
                input_tokens=inp,
                target_tokens=res.output_tokens,
                program=prog_negate,
                operation_graph=res.graph,
                category="K",
                split="test",
                vocab_size=10,
                task_spec=task_spec,
            )
        )

    # Use a dummy custom family registry with NEGATE registered as a candidate
    custom_reg = HoldoutFamilyRegistry()
    custom_reg.register_family(
        FamilyMetadata(
            family_id="test_candidate",
            status=FamilySplit.DEV_FAMILIES,
            structural_dependency_type="test",
            output_shape_rule="same_length",
            argument_schema={},
            generator_version="1.0.0",
            description="test",
            operations=("NEGATE",),
        )
    )

    res = check_novelty_validity(
        target_operation="NEGATE",
        verification_examples=examples,
        threshold=0.90,
        max_composition_depth=1,
        candidate_bank_ops=("COPY", "NEGATE", "SHIFT"),
        registry=custom_reg,
    )

    # NEGATE is in candidate_bank_ops, so direct search achieves EM = 1.0 >= 0.90
    assert res.best_direct_em == 1.0
    assert res.adequate_by_existing_library is True
    assert res.is_valid_novel_holdout is False


def test_demonstration_collision_detection() -> None:
    """Verify that ambiguous demonstration sets are detected as collisions."""
    # 1. Ambiguous demonstration set: multiple operations produce identical outputs
    # For instance, a constant sequence of all zeros (0, 0, 0, 0):
    # COPY(0,0,0,0) = (0,0,0,0)
    # MAJORITY_THREE(0,0,0,0) = (0,0,0,0)
    # NEIGHBOR_MAX(0,0,0,0) = (0,0,0,0)
    prog = Program(steps=(ProgramStep(operation="MAJORITY_THREE", params={}),))
    ambiguous_examples = [
        Example(
            input_tokens=(0, 0, 0, 0),
            target_tokens=(0, 0, 0, 0),
            program=prog,
            operation_graph=run_program(prog, (0, 0, 0, 0), 10).graph,
            category="N",
            split="test",
            vocab_size=10,
        )
    ]

    result = check_identifiability(
        target_operation="MAJORITY_THREE",
        modality=TaskInferenceModality.FEWSHOT_DEMONSTRATIONS,
        evidence=ambiguous_examples,
        candidate_universe=("COPY", "MAJORITY_THREE", "NEIGHBOR_MAX"),
    )

    # Must detect collision!
    assert result.is_identifiable is False
    assert len(result.collided_operations) > 1
    assert "MAJORITY_THREE" in result.collided_operations
    assert "Collision detected" in (result.reason or "")

    # 2. Unambiguous demonstration set: 16 diverse examples of length 8
    ep = generate_holdout_episode("MAJORITY_THREE", seed=42, seq_length=8)
    good_result = check_identifiability(
        target_operation="MAJORITY_THREE",
        modality=TaskInferenceModality.FEWSHOT_DEMONSTRATIONS,
        evidence=ep.inference_examples,
        candidate_universe=("COPY", "MAJORITY_THREE", "NEIGHBOR_MAX", "NEGATE"),
    )
    assert good_result.is_identifiable is True
    assert good_result.collided_operations == ()


def test_leak_audit_zero_on_clean_and_catches_contamination() -> None:
    """Verify that leak audit passes on clean features and catches injected leaks."""
    # 1. Clean pipeline audit
    clean_features = [0.95, 0.05, 0.98, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clean_descriptor = {"domain": "window_3", "reduction": "majority"}
    audit = audit_for_leaks(
        controller_features=clean_features,
        task_inference_input=clean_descriptor,
        modality=TaskInferenceModality.STRUCTURED_DESCRIPTOR,
    )
    assert audit.passed is True
    assert audit.leak_count == 0
    audit.assert_zero_leaks()

    # 2. Contamination: Sealed family ID in descriptor
    leaky_desc = {"family": "sealed_local_neighborhood", "window": 3}
    audit_leaky = audit_for_leaks(
        task_inference_input=leaky_desc,
        modality=TaskInferenceModality.STRUCTURED_DESCRIPTOR,
    )
    assert audit_leaky.passed is False
    assert audit_leaky.leak_count > 0
    assert any("sealed_local_neighborhood" in v for v in audit_leaky.violations)

    # 3. Contamination: Oracle label 'N' in controller features
    leaky_controller = {"em": 0.0, "loss": 2.5, "label": "N"}
    audit_ctrl = audit_for_leaks(
        controller_features=leaky_controller,
        modality=TaskInferenceModality.STRUCTURED_DESCRIPTOR,
    )
    assert audit_ctrl.passed is False
    assert any("Oracle label 'N'" in v for v in audit_ctrl.violations)

    # 4. Contamination: Sealed operation in development tuning set
    sealed_ep = generate_holdout_episode("NEIGHBOR_MAX", seed=42)
    audit_dev = audit_for_leaks(
        dev_tuning_examples=sealed_ep.support_examples,
    )
    assert audit_dev.passed is False
    assert any("contaminated development tuning set" in v for v in audit_dev.violations)


def test_recurrence_episode_generation() -> None:
    """Verify that recurrence episode generation correctly sets category 'R' and metadata."""
    ep = generate_recurrence_episode("MAJORITY_THREE", seed=42, seq_length=8)
    assert ep.category == "R"
    for ex in ep.all_examples:
        assert ex.category == "R"
        assert ex.oracle_metadata is not None
        assert ex.oracle_metadata.label == "R"
        assert ex.oracle_metadata.recurrence_operation == "MAJORITY_THREE"
