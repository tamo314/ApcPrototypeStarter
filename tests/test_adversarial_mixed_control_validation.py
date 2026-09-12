"""Tests for C-D001W Adversarial Mixed-Control Validation
of the Five-Dimensional Oracle Criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "phase_c"
    / "artifacts"
    / "adversarial_mixed_control_validation.json"
)


@pytest.fixture
def validation_data() -> dict:
    assert ARTIFACT_PATH.is_file(), f"Artifact missing: {ARTIFACT_PATH}"
    with open(ARTIFACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_integrity_boundary_all_zeros(validation_data: dict) -> None:
    boundary = validation_data["integrity_boundary"]
    for key, value in boundary.items():
        assert value == 0, f"Integrity violation in {key}: {value} != 0"


def test_mandated_controls_present_and_scored(validation_data: dict) -> None:
    controls = {c["control_id"]: c for c in validation_data["controls_evaluated"]}
    mandated_ids = ["MC1", "MC2", "MC3", "MC4", "MC5", "MC6"]
    for mid in mandated_ids:
        assert mid in controls, f"Mandated control missing: {mid}"

    # Verify 5 dimensions scored per control
    expected_dims = [
        "provenance",
        "example_specificity",
        "relation_specificity",
        "inference_time_availability",
        "counterfactual_invariance",
    ]
    for cid, c in controls.items():
        dim_scores = c["dimension_scores"]
        for d in expected_dims:
            assert d in dim_scores, f"Dimension {d} missing in control {cid}"
            assert dim_scores[d]["score"] in (0, 1)
        # Verify sum matches numeric_score
        assert c["numeric_score"] == sum(dim_scores[d]["score"] for d in expected_dims)


def test_representation_invariance_unmasks_disguised_oracles(validation_data: dict) -> None:
    controls = {c["control_id"]: c for c in validation_data["controls_evaluated"]}
    # MC1 (universal interpreter + bytecode) and MC2 (compressed/encrypted lookup) must score 5/5
    assert controls["MC1"]["numeric_score"] == 5
    assert controls["MC2"]["numeric_score"] == 5
    # Lawful FIRST/LAST must score 0/5
    assert controls["MC6"]["numeric_score"] == 0

    rep_inv = validation_data["adversarial_validation_analyses"]["representation_invariance"]
    assert rep_inv["verdict"] == "CONFIRMED_REPRESENTATION_INVARIANT"


def test_monotonicity_across_leakage_strata(validation_data: dict) -> None:
    controls = {c["control_id"]: c for c in validation_data["controls_evaluated"]}
    zero_leakage = ["NC1", "NC2", "MC3", "MC5", "MC6"]
    partial_leakage = ["MC4"]
    full_leakage = ["PC1", "PC2", "MC1", "MC2", "MC7"]

    for zid in zero_leakage:
        assert controls[zid]["numeric_score"] == 0

    for pid in partial_leakage:
        assert 1 <= controls[pid]["numeric_score"] <= 4

    for fid in full_leakage:
        assert controls[fid]["numeric_score"] == 5

    # Strict monotonicity between strata
    max_zero = max(controls[c]["numeric_score"] for c in zero_leakage)
    min_partial = min(controls[c]["numeric_score"] for c in partial_leakage)
    max_partial = max(controls[c]["numeric_score"] for c in partial_leakage)
    min_full = min(controls[c]["numeric_score"] for c in full_leakage)

    assert max_zero < min_partial
    assert max_partial <= min_full


def test_leave_one_out_necessity(validation_data: dict) -> None:
    loo = validation_data["adversarial_validation_analyses"]["leave_one_out_analysis"]
    assert loo["all_dimensions_necessary"] is True
    assert len(loo["dimension_removals"]) == 5
    for removal in loo["dimension_removals"]:
        assert removal["strictly_necessary"] is True


def test_aggregation_threshold_false_positives_and_negatives(validation_data: dict) -> None:
    analyses = validation_data["adversarial_validation_analyses"]
    thresh_data = analyses["aggregation_threshold_analysis"]
    thresholds = {t["threshold"]: t for t in thresh_data["thresholds_evaluated"]}

    # Threshold 1 (Any-Hit): optimal with 0 FP and 0 FN
    assert thresholds[1]["false_positives"] == 0
    assert thresholds[1]["false_negatives"] == 0
    assert thresholds[1]["precision"] == 1.0
    assert thresholds[1]["recall"] == 1.0

    # Threshold 5 (Unanimous): fails on MC4 (FN > 0)
    assert thresholds[5]["false_negatives"] == 1
    assert "MC4" in thresholds[5]["fn_controls"][0]


def test_contract_v1_1_and_deterministic_baseline_specs(validation_data: dict) -> None:
    contract = validation_data["training_information_contract_v1_1"]
    assert contract["version"] == "1.1"
    assert "h_content = f(content)" in contract["core_invariants"]["content_path_isolation"]

    baseline = validation_data["descriptor_only_deterministic_baseline_requirements"]
    specs = baseline["specifications"]
    floors = specs["target_performance_floors"]
    assert specs["learned_parameters"] == 0
    assert floors["clean_instance_sequence_em"] == 1.0
    assert floors["collision_instance_sequence_em"] == 1.0


def test_deterministic_baseline_simulation() -> None:
    """Executable symbolic verification of Descriptor-Only Deterministic Baseline."""
    def deterministic_baseline(
        x: list[int],
        query_key: int,
        tie_break: str,
    ) -> tuple[int, int]:
        # Step 1: scan sequence for matching keys (at even indices in BindOp)
        matching_value_coords = [
            2 * j + 1 for j in range(len(x) // 2) if x[2 * j] == query_key
        ]
        assert matching_value_coords, "Key not found"
        # Step 2: apply tie-break policy
        if tie_break == "FIRST":
            z_star = min(matching_value_coords)
        elif tie_break == "LAST":
            z_star = max(matching_value_coords)
        else:
            raise ValueError(f"Unknown tie-break {tie_break}")
        # Step 3: emit target token
        y = x[z_star]
        return z_star, y

    # Collision test case: key K appears twice with value V
    # x = [K, V, K, V]
    K, V = 10, 42
    x = [K, V, K, V]

    z_first, y_first = deterministic_baseline(x, query_key=K, tie_break="FIRST")
    z_last, y_last = deterministic_baseline(x, query_key=K, tie_break="LAST")

    # Both emit correct target token V
    assert y_first == V
    assert y_last == V

    # Coordinates diverge as expected by World A vs World B
    assert z_first == 1
    assert z_last == 3
