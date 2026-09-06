# ruff: noqa: E501
"""Focused invariant coverage for B-C005R3-002 (no GPU checkpoints required).

The one function that does perform a real (tiny, throwaway) reconstruction
and repair-training call, `_probe_key_exposure_empirically`, is exercised via
the milestone runner script, not here -- matching this repo's existing D2
test precedent (`test_semantic_relation_audit.py` etc.) of keeping pytest
CPU-only and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.relation_split_protocol import (
    NEW_SEALED_V2_SEEDS,
    NEW_VALIDATION_SEEDS,
    assert_sealed_access_permitted,
    assign_relation_partitions,
    build_quota_preregistration,
    build_relation_catalog,
    build_relation_split,
    build_seed_partition_registration,
    classify_group_exposure,
    historical_exposure_classification,
    l3_coupled_components,
    l3_relation_groups,
    l4_relation_groups,
)
from apc.evaluation.retrieval_repair_benchmark import DEFAULT_DEV_SEEDS, SEALED_GATE_SEEDS
from apc.primitives.argument_scoring import PARAMETERIZED_OPERATIONS

# ---------------------------------------------------------------------------
# Relation catalogue: reverse-relation / alias merging.
# ---------------------------------------------------------------------------


def test_l3_relation_groups_merge_bidirectional_pair_but_keep_others_separate() -> None:
    groups = l3_relation_groups()
    assert len(groups) == 3
    bind_count = groups["BIND-COUNT"]
    assert set(bind_count["directed_units"]) == {"COUNT->BIND", "BIND->COUNT"}
    assert groups["CYCLE_FOUR-SHIFT"]["directed_units"] == ["SHIFT->CYCLE_FOUR"]
    assert groups["BIND-SELECT"]["directed_units"] == ["SELECT->BIND"]


def test_l3_coupled_components_merge_groups_sharing_bind() -> None:
    components = l3_coupled_components()
    assert len(components) == 2
    merged = next(c for c in components.values() if len(c["l3_groups"]) == 2)
    assert set(merged["l3_groups"]) == {"BIND-COUNT", "BIND-SELECT"}
    assert set(merged["member_operations"]) == {"BIND", "COUNT", "SELECT"}
    standalone = next(c for c in components.values() if len(c["l3_groups"]) == 1)
    assert standalone["l3_groups"] == ["CYCLE_FOUR-SHIFT"]
    assert "independent" in standalone["coupling_reason"]
    assert "coupled" in merged["coupling_reason"]


def test_l4_relation_groups_one_independent_group_per_parameterized_operation() -> None:
    groups = l4_relation_groups()
    assert set(groups) == {f"{op}:argument_variant" for op in PARAMETERIZED_OPERATIONS}
    for group in groups.values():
        assert len(group["members"]) == 1


# ---------------------------------------------------------------------------
# Relation catalogue: reading real committed D2 verdicts (never fabricated).
# ---------------------------------------------------------------------------


def test_build_relation_catalog_reads_real_committed_d2_verdicts() -> None:
    catalog = build_relation_catalog()
    assert catalog["l3"]["observed_status_available"] is True
    shift = catalog["l3"]["groups"]["CYCLE_FOUR-SHIFT"]["directed_unit_status"]
    assert shift["SHIFT->CYCLE_FOUR"] == "NO_FAILURE"
    bind_count = catalog["l3"]["groups"]["BIND-COUNT"]["directed_unit_status"]
    assert bind_count["COUNT->BIND"] == "KEY_SCORING_BOTTLENECK"
    assert bind_count["BIND->COUNT"] == "KEY_SCORING_BOTTLENECK"
    bind_select = catalog["l3"]["groups"]["BIND-SELECT"]["directed_unit_status"]
    assert bind_select["SELECT->BIND"] == "UNRESOLVED"

    assert catalog["l4"]["observed_status_available"] is True
    assert catalog["l4"]["groups"]["SELECT:argument_variant"]["status"] == "ARGUMENT_ENCODING_FAILURE"
    assert catalog["l4"]["groups"]["BIND:argument_variant"]["status"] == "ARGUMENT_SCORER_GENERALIZATION_FAILURE"
    assert catalog["l4"]["groups"]["SHIFT:argument_variant"]["status"] == "NO_FAILURE"
    assert catalog["l4"]["groups"]["COUNT:argument_variant"]["status"] == "NO_FAILURE"

    non_target = catalog["bank_members_with_no_l3_relation_defined"]["operations"]
    assert "SHIFT" not in non_target and "BIND" not in non_target
    assert catalog["bank_members_with_no_l3_relation_defined"]["evaluated"] == "UNEVALUATED_NO_RELATION_DEFINED"


def test_build_relation_catalog_falls_back_gracefully_when_d2_artifacts_missing(tmp_path: Path) -> None:
    catalog = build_relation_catalog(repo_root=tmp_path)
    assert catalog["l3"]["observed_status_available"] is False
    for group in catalog["l3"]["groups"].values():
        for status in group["directed_unit_status"].values():
            assert status == "NOT_YET_EVALUATED"
    assert catalog["l4"]["observed_status_available"] is False
    for group in catalog["l4"]["groups"].values():
        assert group["status"] == "NOT_YET_EVALUATED"


# ---------------------------------------------------------------------------
# Partition assignment: disjointness, sufficiency, no alias/reverse leak.
# ---------------------------------------------------------------------------


def test_assign_relation_partitions_never_splits_a_group_across_partitions() -> None:
    catalog = build_relation_catalog()
    partitions = assign_relation_partitions(catalog)
    all_units = (
        partitions["development_units"] + partitions["validation_units"] + partitions["sealed_v2_units"]
    )
    # No unit appears in more than one partition (disjointness).
    assert len(all_units) == len(set(all_units))
    # The BIND-COUNT / BIND-SELECT coupled component (which contains the
    # reverse pair COUNT->BIND / BIND->COUNT) is assigned as ONE atomic unit,
    # never split -- a reverse-relation leak would show up as the same
    # underlying group appearing in two different partitions' unit lists.
    assert "L3:BIND-COUNT+BIND-SELECT" in partitions["development_units"]
    assert not any("BIND-COUNT" in u or "BIND-SELECT" in u for u in partitions["validation_units"] + partitions["sealed_v2_units"])


def test_assign_relation_partitions_flags_real_insufficiency() -> None:
    """Grounded in the actual, currently-committed D2 verdicts: only 3 L3
    non-alias groups exist (2 independent components after BIND-coupling)
    plus 4 L4 groups = 6 independent units total; 3 are already-failing and
    must go to development, leaving only 3 clean units to split between
    validation and sealed_v2 -- one short of the >=2-each minimum. This test
    intentionally pins today's real, disclosed shortfall rather than a
    synthetic scenario, so a future fix (e.g. a richer relation registry)
    will visibly break it instead of silently going stale.
    """
    catalog = build_relation_catalog()
    partitions = assign_relation_partitions(catalog)
    assert partitions["sufficiency"]["development"] is True
    assert partitions["all_partitions_sufficient"] is False
    assert sum(partitions["sufficiency"].values()) == 2  # exactly one partition falls short


def test_relation_split_gate_result_matches_sufficiency() -> None:
    split = build_relation_split()
    if split["relation_group_axis"]["all_partitions_sufficient"]:
        assert split["gate_result"] == "INFRASTRUCTURE_OR_PROTOCOL_PASS"
    else:
        assert split["gate_result"] == "PROTOCOL_INSUFFICIENT_RELATIONS"


# ---------------------------------------------------------------------------
# Seed axis: disjointness and no `seed % 5` aliasing.
# ---------------------------------------------------------------------------


def test_seed_partition_registration_is_disjoint_and_does_not_use_modulo_alias() -> None:
    registration = build_seed_partition_registration()
    assert registration["all_partitions_pairwise_disjoint"] is True
    assert registration["collisions"] == []
    assert registration["modulo_5_alias_used"] is False
    assert registration["development"]["model_seed_recipe"] == "model_seed = seed"
    assert registration["validation"]["model_seed_recipe"] == "model_seed = seed"
    assert registration["sealed_v2"]["model_seed_recipe"] == "model_seed = seed"
    assert registration["sealed_v2"]["status"] == "MEMBERSHIP_REGISTERED_NOT_MEASURED"


def test_new_seed_ranges_never_collide_with_any_prior_b_c005_family_seed() -> None:
    prior_seeds = set(SEALED_GATE_SEEDS) | set(DEFAULT_REGATE_SEEDS) | set(DEFAULT_DEV_SEEDS)
    assert prior_seeds.isdisjoint(NEW_VALIDATION_SEEDS)
    assert prior_seeds.isdisjoint(NEW_SEALED_V2_SEEDS)
    assert set(NEW_VALIDATION_SEEDS).isdisjoint(NEW_SEALED_V2_SEEDS)
    # A `seed % 5` mapping would silently alias every new seed onto the SAME
    # 0-4 base checkpoints already used elsewhere -- this module must not
    # rely on that aliasing for VALIDATION/SEALED_V2 seed independence.
    assert {seed % 5 for seed in NEW_VALIDATION_SEEDS} == {0, 1, 2, 3, 4}
    assert {seed % 5 for seed in NEW_SEALED_V2_SEEDS} == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Sealed access guard.
# ---------------------------------------------------------------------------


def test_assert_sealed_access_permitted_blocks_every_sealed_group_without_purpose() -> None:
    for seeds in (tuple(sorted(SEALED_GATE_SEEDS)), DEFAULT_REGATE_SEEDS, NEW_SEALED_V2_SEEDS):
        with pytest.raises(ValueError, match="sealed seeds"):
            assert_sealed_access_permitted(seeds, purpose="some_repair_task")


def test_assert_sealed_access_permitted_allows_the_declared_gate_purpose() -> None:
    assert_sealed_access_permitted(NEW_SEALED_V2_SEEDS, purpose="R3-011_seal_or_R3-012_gate")


def test_assert_sealed_access_permitted_allows_non_sealed_seeds() -> None:
    assert_sealed_access_permitted(DEFAULT_DEV_SEEDS, purpose="anything")
    assert_sealed_access_permitted(NEW_VALIDATION_SEEDS, purpose="anything")


# ---------------------------------------------------------------------------
# Training-exposure classification (held-out pair entering CE/replay).
# ---------------------------------------------------------------------------


def test_classify_group_exposure_strict_holdout() -> None:
    result = classify_group_exposure(
        held_out_ops=frozenset({"SHIFT"}), positive_ops_seen=["SELECT", "COUNT"], candidate_list_ops=["SELECT", "COUNT"]
    )
    assert result == "STRICT_HOLDOUT"


def test_classify_group_exposure_mining_holdout_only_when_present_but_not_positive() -> None:
    result = classify_group_exposure(
        held_out_ops=frozenset({"SHIFT"}), positive_ops_seen=["SELECT", "COUNT"], candidate_list_ops=["SELECT", "COUNT", "SHIFT"]
    )
    assert result == "MINING_HOLDOUT_ONLY"


def test_classify_group_exposure_no_holdout_when_sampled_as_positive() -> None:
    result = classify_group_exposure(
        held_out_ops=frozenset({"SHIFT"}), positive_ops_seen=["SHIFT", "COUNT"], candidate_list_ops=["SHIFT", "COUNT"]
    )
    assert result == "NO_HOLDOUT"


def test_historical_exposure_classification_unknown_when_provenance_undeterminable() -> None:
    assert historical_exposure_classification(None) == {"classification": "UNKNOWN", "trained_operations": None}


def test_historical_exposure_classification_confirmed_when_provenance_known() -> None:
    from apc.evaluation.relation_split_protocol import FULL_BANK_16_OPERATIONS

    result = historical_exposure_classification(list(FULL_BANK_16_OPERATIONS))
    assert result["classification"] == "CONFIRMED_ALL_GROUPS_NO_HOLDOUT"

    partial = historical_exposure_classification(["SHIFT", "SELECT"])
    assert partial["classification"] == "CONFIRMED_PARTIAL"


# ---------------------------------------------------------------------------
# Quota pre-registration: counts only, no measured sealed output.
# ---------------------------------------------------------------------------


def test_build_quota_preregistration_declares_counts_not_measured_values() -> None:
    catalog = build_relation_catalog()
    quotas = build_quota_preregistration(catalog)
    assert "never a measured bin edge" in quotas["difficulty_bin_basis"]
    for quota in quotas["per_relation_quota"].values():
        assert set(quota) == {"query_examples_per_episode", "support_episodes_per_cell", "difficulty_bin_count"}
        assert all(isinstance(v, int) for v in quota.values())
