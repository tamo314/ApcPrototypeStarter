"""Focused invariants for REC-004AH score-decomposition structural identifiability review."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/review_rec004ah_score_decomposition_identifiability.py")
    spec = importlib.util.spec_from_file_location("rec004ah", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scorer_computation_graph_structure() -> None:
    module = _module()
    graph = module.map_scorer_computation_graph()

    arch = graph["model_architecture"]
    assert arch["d_model"] == 32
    assert arch["n_head"] == 2
    assert arch["head_dim"] == 16
    assert arch["residual_rank"] == 4

    # Check that upstream tensors include k_in, query, v_in
    upstream = graph["upstream_tensors"]
    assert "k_in" in upstream
    assert "v_in" in upstream
    assert "query" in upstream

    # Check that score components are properly mapped
    components = graph["score_components"]
    assert "S_QK" in components
    assert "S_position_bias" in components
    assert "S_residual" in components

    # Downstream path includes losslessness proof under oracle
    assert "downstream_validity" in graph["downstream_pathway"]


def test_identifiability_matrix_evaluations() -> None:
    module = _module()
    matrix = module.build_component_identifiability_matrix()

    comps = matrix["components"]
    assert "S_QK" in comps
    assert "S_position_bias" in comps
    assert "S_residual" in comps
    assert "joint_coupling" in comps

    # S_QK: (a)-(c) are True, (d) is False
    qk_eval = comps["S_QK"]["evaluation"]
    assert qk_eval["a_architecture_native"] is True
    assert qk_eval["b_target_oracle_independent"] is True
    assert qk_eval["c_preserves_other_components_and_downstream"] is True
    assert qk_eval["d_repair_relevant_local_perturbation"] is False
    assert comps["S_QK"]["identifiable_for_repair"] is False

    # S_position_bias: both candidate interventions fail (d)
    pos_intervs = comps["S_position_bias"]["candidate_interventions"]
    for interv in pos_intervs:
        assert interv["evaluation"]["d_repair_relevant_local_perturbation"] is False
    assert comps["S_position_bias"]["identifiable_for_repair"] is False

    # S_residual: all candidate interventions fail (d)
    res_intervs = comps["S_residual"]["candidate_interventions"]
    for interv in res_intervs:
        assert interv["evaluation"]["d_repair_relevant_local_perturbation"] is False
    assert comps["S_residual"]["identifiable_for_repair"] is False


def test_decision_rule_binary_enforcement() -> None:
    module = _module()
    matrix = module.build_component_identifiability_matrix()

    decision, evidence = module.evaluate_decision_rule(matrix)
    assert decision == "SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP"
    assert evidence["passing_components_count"] == 0
    assert len(evidence["passing_components"]) == 0

    # Counterfactual synthetic test: if exactly one component passed all 4 conditions
    synth_matrix = {
        "components": {
            "S_QK": {
                "evaluation": {
                    "a_architecture_native": True,
                    "b_target_oracle_independent": True,
                    "c_preserves_other_components_and_downstream": True,
                    "d_repair_relevant_local_perturbation": True,
                }
            },
            "S_position_bias": {
                "candidate_interventions": [
                    {
                        "name": "interv1",
                        "evaluation": {
                            "a_architecture_native": True,
                            "b_target_oracle_independent": True,
                            "c_preserves_other_components_and_downstream": True,
                            "d_repair_relevant_local_perturbation": False,
                        },
                    }
                ]
            },
            "S_residual": {
                "candidate_interventions": [
                    {
                        "name": "interv1",
                        "evaluation": {
                            "a_architecture_native": True,
                            "b_target_oracle_independent": True,
                            "c_preserves_other_components_and_downstream": True,
                            "d_repair_relevant_local_perturbation": False,
                        },
                    }
                ]
            },
        }
    }
    synth_dec, synth_ev = module.evaluate_decision_rule(synth_matrix)
    assert synth_dec == "SCORE_DECOMPOSITION_IDENTIFIABLE_FOR_REPAIR"
    assert synth_ev["passing_components_count"] == 1


def test_review_run_artifacts_exist_and_consistent(tmp_path: Path) -> None:
    module = _module()
    out_dir = tmp_path / "rec004ah_test"
    module.run_review(out_dir)

    expected_files = [
        "protocol.json",
        "source_hashes.json",
        "computation_graph.json",
        "identifiability_matrix.json",
        "decision_evidence.json",
        "freeze_audit.json",
        "side_effect_audit.json",
        "system.json",
        "summary.json",
        "report.md",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing artifact: {fname}"

    summary = module.read_json(out_dir / "summary.json")
    assert summary["decision"] == "SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP"
    assert summary["execution_status"] == "PASS"
    assert summary["optimizer_updates"] == 0
    assert summary["new_parameters"] == 0
    assert summary["score_decomposition_identifiable_for_repair"] is False

    freeze = module.read_json(out_dir / "freeze_audit.json")
    assert freeze["all_frozen_hashes_match"] is True
    assert freeze["model_hash_matches_rec004ag"] is True
