# ruff: noqa: E501
"""Focused coverage for B-C005D2-006 (pure synthesis; no GPU/model required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apc.evaluation.integrated_causal_diagnosis import (
    FindingRow,
    IntegratedCausalDiagnosisConfig,
    _assert_no_forbidden_conclusions,
    _recommend_next_action,
    _row_adequacy_estimator,
    _row_false_plastic_metric,
    _row_installed_shift_adequacy,
    _row_l2_retrieval,
    _row_l3_key_scoring,
    _row_l3_query_projection,
    _row_l3_relation_split,
    _row_l3_task_representation,
    _row_l4_argument_resolution,
    _row_l4_family_routing,
    build_findings_table,
    evaluate_next_repair_options,
    run_integrated_causal_diagnosis,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures mirroring the real D2-001..005 artifact shapes, but with
# small, hand-checkable numbers instead of the real (large) run outputs.
# ---------------------------------------------------------------------------


def _d2001(*, non_representative: bool = True) -> dict:
    return {
        "representativeness_verdict": {
            "classification": "NON_REPRESENTATIVE" if non_representative else "REPRESENTATIVE",
            "flags": ["DEVELOPMENT_DIFFICULTY_MISMATCH"] if non_representative else [],
            "evidence": {
                "pre_repair_development_l3_top1": 0.98,
                "pre_repair_original_sealed_l3_top1": 0.68,
                "post_repair_development_l3_top1": 1.0,
                "post_repair_regate_sealed_l3_top1": 0.75,
            },
        }
    }


def _d2002(*, key_scoring_failure: bool = True) -> dict:
    per_relation = {
        "SHIFT->CYCLE_FOUR": "NO_FAILURE",
        "SELECT->BIND": "UNRESOLVED",
        "COUNT->BIND": "KEY_SCORING_BOTTLENECK" if key_scoring_failure else "NO_FAILURE",
        "BIND->COUNT": "KEY_SCORING_BOTTLENECK" if key_scoring_failure else "NO_FAILURE",
    }
    return {
        "representation_stage_localization": {
            "verdict_evidence": {
                "z_probe_accuracy": 1.0,
                "q_probe_accuracy": 1.0,
                "control_a_current_path_top1": 0.757,
                "shuffled_controls_at_chance": True,
            },
            "per_relation_verdict": per_relation,
        }
    }


def _d2003(*, holdout_required: bool = True) -> dict:
    return {
        "verdict": {
            "labels": ["SEMANTIC_RELATION_HOLDOUT_REQUIRED"] if holdout_required else ["SEED_SPLIT_SUFFICIENT"],
            "semantic_relation_holdout_required": holdout_required,
            "evidence": {"cross_relation_spread_at_focus_cell": 0.878 if holdout_required else 0.03},
        },
        "difficulty_matched_comparison": {
            "post_repair": {"matched_gap": 0.366, "raw_gap": 0.3695},
        },
    }


def _d2004(*, bind_select_fail: bool = True) -> dict:
    classification = {
        "SHIFT": {"classification": "NO_FAILURE"},
        "SELECT": {
            "classification": "ARGUMENT_ENCODING_FAILURE" if bind_select_fail else "NO_FAILURE"
        },
        "COUNT": {"classification": "NO_FAILURE"},
        "BIND": {
            "classification": "ARGUMENT_SCORER_GENERALIZATION_FAILURE"
            if bind_select_fail
            else "NO_FAILURE"
        },
    }
    family_top1 = {"SHIFT": 1.0, "SELECT": 1.0, "COUNT": 1.0, "BIND": 1.0}
    return {
        "l4_operation_breakdown": {
            "regate_sealed/R2_frozen_post_repair": {
                op: {"family_top1": family_top1[op]} for op in family_top1
            }
        },
        "failure_classification": classification,
    }


def _d2005(*, rule_biased: bool = True, inadequate: bool = True, unsafe_reuse: bool = True) -> dict:
    audits = []
    for bank_size in (16, 32, 64, 128):
        upper = 0.94 if inadequate else 0.97
        em = 0.92 if inadequate else 0.97
        audits.append(
            {
                "bank_size": bank_size,
                "rank_diagnostics": {
                    "L0_orthogonal": {"primitive_call_top1": 1.0},
                    "L1_random_score_space": {"primitive_call_top1": 1.0},
                    "L2_near_neighbor": {"primitive_call_top1": 1.0},
                },
                "reference_adequacy": {
                    "reference_em": em,
                    "wilson_ci": {"upper": upper},
                },
            }
        )
    labels = []
    if rule_biased:
        labels.append("SEQUENTIAL_RULE_BIAS")
    if inadequate:
        labels.append("TRUE_PRIMITIVE_INADEQUACY")
    if unsafe_reuse:
        labels.append("UNSAFE_REUSE_DETECTED")
    return {
        "spec_verification": {"implementation_matches_spec": True, "mismatches": []},
        "bank_size_audits": audits,
        "overall_classification": {
            "labels": labels,
            "n_true_false_plastic": 0,
            "n_functionally_justified_plastic": 2 if inadequate else 0,
            "n_unsafe_reuse": 2 if unsafe_reuse else 0,
            "n_correctly_accepted": 0 if inadequate else 4,
        },
    }


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="no_failure_threshold"):
        IntegratedCausalDiagnosisConfig(no_failure_threshold=1.5)


def test_config_to_dict_stringifies_paths() -> None:
    config = IntegratedCausalDiagnosisConfig(output_dir=Path("runs/x"))
    as_dict = config.to_dict()
    assert as_dict["output_dir"] == str(Path("runs/x"))
    assert isinstance(as_dict["d2001_summary_path"], str)


# ---------------------------------------------------------------------------
# Individual findings-table rows.
# ---------------------------------------------------------------------------


def test_l2_retrieval_no_failure_when_all_top1_above_threshold() -> None:
    row = _row_l2_retrieval(_d2005(), threshold=0.95)
    assert row.verdict == "NO_FAILURE"
    assert row.confidence == "HIGH"


def test_l2_retrieval_unresolved_when_below_threshold() -> None:
    d2005 = _d2005()
    d2005["bank_size_audits"][0]["rank_diagnostics"]["L2_near_neighbor"]["primitive_call_top1"] = 0.5
    row = _row_l2_retrieval(d2005, threshold=0.95)
    assert row.verdict == "UNRESOLVED"


def test_l3_task_representation_no_failure_when_z_probe_perfect() -> None:
    row = _row_l3_task_representation(_d2002())
    assert row.verdict == "NO_FAILURE"


def test_l3_task_representation_bottleneck_when_z_probe_fails() -> None:
    d2002 = _d2002()
    d2002["representation_stage_localization"]["verdict_evidence"]["z_probe_accuracy"] = 0.5
    row = _row_l3_task_representation(d2002)
    assert row.verdict == "TASK_REPRESENTATION_BOTTLENECK"


def test_l3_query_projection_no_failure_when_q_probe_perfect() -> None:
    row = _row_l3_query_projection(_d2002())
    assert row.verdict == "NO_FAILURE"


def test_l3_query_projection_bottleneck_when_q_probe_fails() -> None:
    d2002 = _d2002()
    d2002["representation_stage_localization"]["verdict_evidence"]["q_probe_accuracy"] = 0.4
    row = _row_l3_query_projection(d2002)
    assert row.verdict == "QUERY_PROJECTION_BOTTLENECK"


def test_l3_key_scoring_identifies_bottleneck_relations() -> None:
    row = _row_l3_key_scoring(_d2002(key_scoring_failure=True))
    assert row.verdict == "KEY_SCORING_BOTTLENECK"
    assert row.evidence["bottleneck_relations"] == ["COUNT->BIND", "BIND->COUNT"]
    assert row.confidence == "MEDIUM"  # SELECT->BIND stays UNRESOLVED in the fixture


def test_l3_key_scoring_mixed_when_no_bottleneck_relations() -> None:
    row = _row_l3_key_scoring(_d2002(key_scoring_failure=False))
    assert row.verdict == "MIXED"
    assert row.evidence["bottleneck_relations"] == []


def test_l3_relation_split_holdout_required() -> None:
    row = _row_l3_relation_split(_d2003(holdout_required=True))
    assert row.verdict == "SEMANTIC_RELATION_HOLDOUT_REQUIRED"
    assert "relation sets" in row.next_action


def test_l3_relation_split_seed_split_sufficient() -> None:
    row = _row_l3_relation_split(_d2003(holdout_required=False))
    assert row.verdict == "SEED_SPLIT_SUFFICIENT"
    assert row.next_action == "none"


def test_l4_family_routing_no_failure() -> None:
    row = _row_l4_family_routing(_d2004(), threshold=0.95)
    assert row.verdict == "NO_FAILURE"


def test_l4_argument_resolution_mixed_when_some_operations_fail() -> None:
    row = _row_l4_argument_resolution(_d2004(bind_select_fail=True))
    assert row.verdict == "MIXED"
    assert set(row.evidence["classification_by_operation"].keys()) == {
        "SHIFT",
        "SELECT",
        "COUNT",
        "BIND",
    }
    assert "BIND" in row.next_action and "SELECT" in row.next_action


def test_l4_argument_resolution_no_failure_when_all_operations_pass() -> None:
    row = _row_l4_argument_resolution(_d2004(bind_select_fail=False))
    assert row.verdict == "NO_FAILURE"
    assert row.next_action == "none"


def test_adequacy_estimator_flags_sequential_rule_bias() -> None:
    row = _row_adequacy_estimator(_d2005(rule_biased=True))
    assert row.verdict == "SEQUENTIAL_RULE_BIAS"


def test_adequacy_estimator_no_failure_without_bias_label() -> None:
    row = _row_adequacy_estimator(_d2005(rule_biased=False, inadequate=False, unsafe_reuse=False))
    assert row.verdict == "NO_FAILURE"


def test_installed_shift_adequacy_true_inadequacy_when_all_wilson_upper_below_threshold() -> None:
    row = _row_installed_shift_adequacy(_d2005(inadequate=True), threshold=0.95)
    assert row.verdict == "TRUE_PRIMITIVE_INADEQUACY"


def test_installed_shift_adequacy_unresolved_when_any_cell_clears_threshold() -> None:
    row = _row_installed_shift_adequacy(_d2005(inadequate=False), threshold=0.95)
    assert row.verdict == "UNRESOLVED"


def test_false_plastic_metric_misclassification_when_unsafe_reuse_without_true_false_plastic() -> None:
    row = _row_false_plastic_metric(_d2005(unsafe_reuse=True))
    assert row.verdict == "METRIC_MISCLASSIFICATION"


def test_false_plastic_metric_no_failure_when_no_unsafe_reuse() -> None:
    row = _row_false_plastic_metric(_d2005(unsafe_reuse=False, inadequate=False, rule_biased=False))
    assert row.verdict == "NO_FAILURE"


# ---------------------------------------------------------------------------
# Full table, option evaluation, and recommendation.
# ---------------------------------------------------------------------------


def test_build_findings_table_has_ten_rows_in_declared_order() -> None:
    rows = build_findings_table(
        d2001=_d2001(), d2002=_d2002(), d2003=_d2003(), d2004=_d2004(), d2005=_d2005(), threshold=0.95
    )
    assert [row.mechanism for row in rows] == [
        "L2 retrieval",
        "L3 task representation",
        "L3 query projection",
        "L3 key/scoring",
        "L3 relation split",
        "L4 family routing",
        "L4 argument resolution",
        "adequacy estimator",
        "installed SHIFT adequacy",
        "false-plastic metric",
    ]
    assert all(isinstance(row, FindingRow) for row in rows)


def test_evaluate_next_repair_options_matches_real_evidence_pattern() -> None:
    rows = build_findings_table(
        d2001=_d2001(), d2002=_d2002(), d2003=_d2003(), d2004=_d2004(), d2005=_d2005(), threshold=0.95
    )
    options = evaluate_next_repair_options(rows, _d2004())
    assert options["A_query_projection_repair"]["allowed"] is False
    assert options["B_task_representation_repair"]["allowed"] is False
    assert options["C_semantic_relation_holdout_redesign"]["allowed"] is True
    assert options["D_argument_scorer_repair"]["allowed"] is True
    assert set(options["D_argument_scorer_repair"]["operations"]) == {"SELECT", "BIND"}
    assert options["E_adequacy_metric_protocol_repair"]["allowed"] is True
    assert options["F_primitive_functional_generalization_repair"]["allowed"] is True


def test_evaluate_next_repair_options_disallows_everything_when_no_failures() -> None:
    rows = build_findings_table(
        d2001=_d2001(non_representative=False),
        d2002=_d2002(key_scoring_failure=False),
        d2003=_d2003(holdout_required=False),
        d2004=_d2004(bind_select_fail=False),
        d2005=_d2005(rule_biased=False, inadequate=False, unsafe_reuse=False),
        threshold=0.95,
    )
    options = evaluate_next_repair_options(rows, _d2004(bind_select_fail=False))
    assert all(not entry["allowed"] for entry in options.values())


def test_recommend_next_action_prefers_option_c_when_allowed() -> None:
    options = {
        "A_query_projection_repair": {"allowed": False},
        "B_task_representation_repair": {"allowed": False},
        "C_semantic_relation_holdout_redesign": {"allowed": True},
        "D_argument_scorer_repair": {"allowed": True, "operations": ["BIND"]},
        "E_adequacy_metric_protocol_repair": {"allowed": True},
        "F_primitive_functional_generalization_repair": {"allowed": True},
    }
    recommendation = _recommend_next_action(options)
    assert recommendation["primary_next_action"] == "OPTION_C_SEMANTIC_RELATION_HOLDOUT_REDESIGN"
    assert "D_argument_scorer_repair" in recommendation["evidence_supported_but_not_recommended_as_primary"]
    assert "C_semantic_relation_holdout_redesign" not in recommendation[
        "evidence_supported_but_not_recommended_as_primary"
    ]


def test_recommend_next_action_unresolved_when_nothing_allowed() -> None:
    options = {
        "A_query_projection_repair": {"allowed": False},
        "B_task_representation_repair": {"allowed": False},
        "C_semantic_relation_holdout_redesign": {"allowed": False},
        "D_argument_scorer_repair": {"allowed": False, "operations": []},
        "E_adequacy_metric_protocol_repair": {"allowed": False},
        "F_primitive_functional_generalization_repair": {"allowed": False},
    }
    recommendation = _recommend_next_action(options)
    assert recommendation["primary_next_action"] == "UNRESOLVED"
    assert recommendation["evidence_supported_but_not_recommended_as_primary"] == []


# ---------------------------------------------------------------------------
# D2-006.3 forbidden-conclusion guard.
# ---------------------------------------------------------------------------


def test_forbidden_conclusion_guard_passes_on_clean_payload() -> None:
    _assert_no_forbidden_conclusions({"recommendation": "define relation holdout"})


def test_forbidden_conclusion_guard_rejects_router_size_conclusion() -> None:
    with pytest.raises(AssertionError, match="forbidden conclusion"):
        _assert_no_forbidden_conclusions({"note": "the router needs to be larger"})


def test_forbidden_conclusion_guard_rejects_threshold_conclusion() -> None:
    with pytest.raises(AssertionError, match="forbidden conclusion"):
        _assert_no_forbidden_conclusions({"note": "the adequacy threshold should be lower"})


def test_real_diagnosis_output_never_trips_forbidden_conclusion_guard() -> None:
    """Every real finding/recommendation this module can produce, across the
    full space of the toggles above, must never itself trip the guard --
    exercising the guard against synthetic bad input (above) is necessary but
    not sufficient; this checks the module's own normal output stays clean."""
    for non_repr in (True, False):
        for key_fail in (True, False):
            for holdout in (True, False):
                for bind_select in (True, False):
                    for biased in (True, False):
                        for inadequate in (True, False):
                            report_rows = build_findings_table(
                                d2001=_d2001(non_representative=non_repr),
                                d2002=_d2002(key_scoring_failure=key_fail),
                                d2003=_d2003(holdout_required=holdout),
                                d2004=_d2004(bind_select_fail=bind_select),
                                d2005=_d2005(rule_biased=biased, inadequate=inadequate, unsafe_reuse=inadequate),
                                threshold=0.95,
                            )
                            options = evaluate_next_repair_options(
                                report_rows, _d2004(bind_select_fail=bind_select)
                            )
                            recommendation = _recommend_next_action(options)
                            payload = {
                                "findings_table": [row.to_dict() for row in report_rows],
                                "next_repair_options": options,
                                "recommendation": recommendation,
                            }
                            _assert_no_forbidden_conclusions(payload)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end run against real fixture files on disk.
# ---------------------------------------------------------------------------


def test_run_integrated_causal_diagnosis_end_to_end(tmp_path: Path) -> None:
    fixtures = {
        "summary.json": _d2001(),
        "representation_stage_summary.json": _d2002(),
        "semantic_relation_summary.json": _d2003(),
        "l4_argument_breakdown.json": _d2004(),
        "shift_seed24_adequacy_audit.json": _d2005(),
    }
    for name, content in fixtures.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")

    output_dir = tmp_path / "out"
    config = IntegratedCausalDiagnosisConfig(
        d2001_summary_path=tmp_path / "summary.json",
        d2002_summary_path=tmp_path / "representation_stage_summary.json",
        d2003_summary_path=tmp_path / "semantic_relation_summary.json",
        d2004_summary_path=tmp_path / "l4_argument_breakdown.json",
        d2005_summary_path=tmp_path / "shift_seed24_adequacy_audit.json",
        output_dir=output_dir,
    )
    report = run_integrated_causal_diagnosis(config)

    assert report["task_id"] == "B-C005D2-006"
    assert len(report["findings_table"]) == 10
    assert report["recommendation"]["primary_next_action"] == "OPTION_C_SEMANTIC_RELATION_HOLDOUT_REDESIGN"
    assert report["downstream_blocked"] is True
    assert report["b_c006_and_task_inference_remain_blocked"] is True

    assert (output_dir / "final_causal_diagnosis.json").exists()
    assert (output_dir / "integrated_causal_diagnosis_config.yaml").exists()
    assert (output_dir / "integrated_causal_diagnosis_protocol.json").exists()
    assert (output_dir / "integrated_causal_diagnosis_system.json").exists()
    on_disk = json.loads((output_dir / "final_causal_diagnosis.json").read_text(encoding="utf-8"))
    assert on_disk["task_id"] == "B-C005D2-006"


def test_run_integrated_causal_diagnosis_without_output_dir_writes_nothing(tmp_path: Path) -> None:
    fixtures = {
        "summary.json": _d2001(),
        "representation_stage_summary.json": _d2002(),
        "semantic_relation_summary.json": _d2003(),
        "l4_argument_breakdown.json": _d2004(),
        "shift_seed24_adequacy_audit.json": _d2005(),
    }
    for name, content in fixtures.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")

    config = IntegratedCausalDiagnosisConfig(
        d2001_summary_path=tmp_path / "summary.json",
        d2002_summary_path=tmp_path / "representation_stage_summary.json",
        d2003_summary_path=tmp_path / "semantic_relation_summary.json",
        d2004_summary_path=tmp_path / "l4_argument_breakdown.json",
        d2005_summary_path=tmp_path / "shift_seed24_adequacy_audit.json",
        output_dir=None,
    )
    report = run_integrated_causal_diagnosis(config)
    assert report["task_id"] == "B-C005D2-006"
    assert list(tmp_path.iterdir()) == [tmp_path / name for name in sorted(fixtures)]
