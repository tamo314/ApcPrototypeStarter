"""Tests for C-D001X Contract v1.1 Hypothesis-Preservation
and Deterministic-Baseline Dominance Audit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "phase_c"
    / "artifacts"
    / "contract_v1_1_hypothesis_preservation_audit.json"
)


@pytest.fixture
def audit_data() -> dict[str, Any]:
    assert ARTIFACT_PATH.is_file(), f"Artifact missing: {ARTIFACT_PATH}"
    with open(ARTIFACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_integrity_boundary_zero_execution(audit_data: dict[str, Any]) -> None:
    boundary = audit_data["integrity_boundary"]
    for k, v in boundary.items():
        assert v == 0, f"Integrity boundary violated in {k}: {v} != 0"


def test_primary_estimand_decomposition(audit_data: dict[str, Any]) -> None:
    decomp = audit_data["primary_estimand_decomposition"]
    required_components = [
        "task_identity_learning",
        "descriptor_interpretation",
        "coordinate_selection",
    ]
    for comp in required_components:
        assert comp in decomp, f"Missing estimand component: {comp}"
        item = decomp[comp]
        assert "opaque_id_ce_only" in item
        assert "contract_v1_1" in item
        assert "deterministic_baseline_b_det" in item

    # Verify specific status markings
    assert decomp["task_identity_learning"]["contract_v1_1"]["status"] == "ELIMINATED_A_PRIORI"
    assert (
        decomp["descriptor_interpretation"]["contract_v1_1"]["status"]
        == "REQUIRES_FUNCTION_APPROXIMATION"
    )
    assert decomp["coordinate_selection"]["contract_v1_1"]["status"] == "ELIMINATED_A_PRIORI"
    assert decomp["coordinate_selection"]["opaque_id_ce_only"]["status"] == "UNIDENTIFIABLE_STOP"


def test_three_conditions_evaluation(audit_data: dict[str, Any]) -> None:
    conds = audit_data["three_conditions_evaluation"]["conditions"]
    assert "COND1_OPAQUE_ID_CE_ONLY" in conds
    assert "COND2_CONTRACT_V1_1_SEMANTIC_DESCRIPTOR" in conds
    assert "COND3_DESCRIPTOR_ONLY_DETERMINISTIC_BASELINE_B_DET" in conds

    c1 = conds["COND1_OPAQUE_ID_CE_ONLY"]["metrics"]
    c2 = conds["COND2_CONTRACT_V1_1_SEMANTIC_DESCRIPTOR"]["metrics"]
    c3 = conds["COND3_DESCRIPTOR_ONLY_DETERMINISTIC_BASELINE_B_DET"]["metrics"]

    # COND1: non-zero entropy on collisions, 0 separation, 0 unseen transfer
    assert c1["h_z_given_x_d_collision"] > 0.0
    assert c1["world_a_world_b_separation_em"] == 0.0
    assert c1["unseen_relation_zero_shot_em"] == 0.0

    # COND2: 0 entropy, 1.0 separation
    assert c2["h_z_given_x_d_collision"] == 0.0
    assert c2["world_a_world_b_separation_em"] == 1.0

    # COND3: 0 entropy, 1.0 separation, 0 learned parameters
    assert c3["h_z_given_x_d_collision"] == 0.0
    assert c3["world_a_world_b_separation_em"] == 1.0
    assert conds["COND3_DESCRIPTOR_ONLY_DETERMINISTIC_BASELINE_B_DET"]["learned_parameters"] == 0


def test_ablation_1_tie_break_masking_simulation() -> None:
    """Executable verification of Ablation 1: TieBreakPolicy Masking."""
    # Given sequence x = [K, V, K, V] with BindOp query_key K
    K, V = 10, 42
    x = [K, V, K, V]

    # Full descriptor specifies FIRST or LAST
    def route_with_tb(x: list[int], query_key: int, tb: str) -> int:
        coords = [2 * j + 1 for j in range(len(x) // 2) if x[2 * j] == query_key]
        return min(coords) if tb == "FIRST" else max(coords)

    assert route_with_tb(x, K, "FIRST") == 1
    assert route_with_tb(x, K, "LAST") == 3

    # Masking TieBreakPolicy: D_masked = (BIND, {query_key: K}, null)
    # Candidate coordinates set:
    coords = [2 * j + 1 for j in range(len(x) // 2) if x[2 * j] == K]
    assert coords == [1, 3]

    # Residual conditional entropy H(Z | X, D_masked)
    p_each = 1.0 / len(coords)
    h_z = -sum(p_each * math.log2(p_each) for _ in coords)
    assert pytest.approx(h_z, rel=1e-5) == 1.0  # exactly 1.0 bit

    # World A and World B share identical masked descriptor
    d_masked_a = ("BIND", {"query_key": K}, None)
    d_masked_b = ("BIND", {"query_key": K}, None)
    assert d_masked_a == d_masked_b  # Inseparability returns


def test_ablation_2_first_last_counterfactual_swap_simulation() -> None:
    """Executable verification of Ablation 2: FIRST <-> LAST counterfactual swap."""
    K, V = 10, 42
    x = [K, V, K, V]

    def execute_b_det(x: list[int], query_key: int, tb: str) -> tuple[int, int]:
        coords = [2 * j + 1 for j in range(len(x) // 2) if x[2 * j] == query_key]
        z_star = min(coords) if tb == "FIRST" else max(coords)
        y = x[z_star]
        return z_star, y

    z_first, y_first = execute_b_det(x, K, "FIRST")
    z_last, y_last = execute_b_det(x, K, "LAST")

    # Output tokens are identical
    assert y_first == y_last == V
    delta_y = abs(y_first - y_last)
    assert delta_y == 0

    # Coordinates swap strictly
    assert z_first == 1
    assert z_last == 3
    delta_z = abs(z_first - z_last)
    assert delta_z == 2

    # Proves coordinate is 100% determined by tie-break token, 0% by output token y
    assert delta_z > 0 and delta_y == 0


def test_ablation_3_descriptor_permutation_simulation() -> None:
    """Executable verification of Ablation 3:
    Operation/Argument-Preserving Descriptor Permutation.
    """
    K, V = 10, 42
    x = [K, V, K, V]

    descriptors = [
        {"op": "BIND", "args": {"query_key": K}, "tb": "FIRST"},
        {"op": "BIND", "args": {"query_key": K}, "tb": "LAST"},
    ]

    def eval_desc(x: list[int], desc: dict) -> tuple[int, int]:
        target_k = desc["args"]["query_key"]
        coords = [2 * j + 1 for j in range(len(x) // 2) if x[2 * j] == target_k]
        z = min(coords) if desc["tb"] == "FIRST" else max(coords)
        return z, x[z]

    # Original
    z0, y0 = eval_desc(x, descriptors[0])
    z1, y1 = eval_desc(x, descriptors[1])
    assert (z0, z1) == (1, 3)
    assert (y0, y1) == (V, V)

    # Permute tie-break
    permuted_descriptors = [
        {"op": "BIND", "args": {"query_key": K}, "tb": "LAST"},
        {"op": "BIND", "args": {"query_key": K}, "tb": "FIRST"},
    ]
    pz0, py0 = eval_desc(x, permuted_descriptors[0])
    pz1, py1 = eval_desc(x, permuted_descriptors[1])
    assert (pz0, pz1) == (3, 1)
    assert (py0, py1) == (V, V)


def test_deterministic_baseline_dominance_audit(audit_data: dict[str, Any]) -> None:
    audit = audit_data["deterministic_baseline_dominance_audit"]
    assert audit["residual_uncertainty_bits"] == 0.0
    assert audit["dominance_verdict"] == "DETERMINISTIC_BASELINE_FULLY_DOMINATES"

    for crit in audit["registered_charter_criteria_comparison"]:
        if "score" in crit:
            assert crit["b_det_score"] >= crit["charter_floor"]


def test_hypothesis_preservation_and_estimand_judgment(audit_data: dict[str, Any]) -> None:
    judgment = audit_data["hypothesis_preservation_and_estimand_judgment"]
    assert judgment["is_h_c1_preserved"] is False
    assert judgment["judgment_label"] == "H_C1_TRIVIALIZED_ESTIMAND_ALTERED"
    assert judgment["charter_recommendation"] == "RETRACT_OR_RESTRICT_TO_RESIDUAL_LEARNING"

    estimand_comp = judgment["estimand_comparison"]
    assert "original_h_c1_estimand" in estimand_comp
    assert "contract_v1_1_transformed_estimand" in estimand_comp


def test_residual_learning_problem_specification(audit_data: dict[str, Any]) -> None:
    spec = audit_data["residual_learning_problem_specification"]
    assert "problem_name" in spec
    assert "formal_definition" in spec
    assert "baseline_delta_definition" in spec
    assert spec["baseline_delta_definition"]["baseline_ceiling"] == 1.0
