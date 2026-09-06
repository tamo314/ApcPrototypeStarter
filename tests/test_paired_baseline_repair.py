# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-004's pure logic.

Matches this repo's existing precedent (D2/R3-002): anything that needs a real
Core/router/bank (`_shift_reference_stream`, `_select_bind_shuffled_control`,
`run_paired_baseline_repair` itself) is exercised only via the milestone
script (`scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-004`), not
in pytest. This file covers the aggregation/classification logic and the
adequacy-evaluator replay (`_apply_adequacy_evaluators` takes a plain
``list[bool]`` -- no model needed) with synthetic inputs.
"""

from __future__ import annotations

import pytest

from apc.evaluation.paired_baseline_repair import (
    ADEQUACY_FIXED_NS,
    ORIGINAL_D2_REFERENCE,
    PairedBaselineConfig,
    _apply_adequacy_evaluators,
    _build_comparison_manifest,
    _build_failure_reproduction_matrix,
    _build_paired_baseline,
    _build_select_bind_control_status,
    _downstream_task_for,
    _mean,
    _pretrained_core_available,
    _relation_label,
)
from apc.meta.adequacy_verifier import AdequacyDecision
from apc.meta.phase_b_protocol import HardNegativeLevel

# ---------------------------------------------------------------------------
# PairedBaselineConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = PairedBaselineConfig()
    assert config.bank_size == 128
    assert config.development_seeds == (10, 11, 12, 13, 14)


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        PairedBaselineConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        PairedBaselineConfig(bank_size=100)


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="support_examples"):
        PairedBaselineConfig(support_examples=0)


def test_config_rejects_shift_reference_examples_below_max_fixed_n() -> None:
    with pytest.raises(ValueError, match="fixed-N"):
        PairedBaselineConfig(shift_reference_examples=max(ADEQUACY_FIXED_NS) - 1)


def test_config_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="adequacy_exact_match_threshold"):
        PairedBaselineConfig(adequacy_exact_match_threshold=1.5)


def test_config_rejects_bad_shuffled_chance_tolerance() -> None:
    with pytest.raises(ValueError, match="shuffled_chance_tolerance"):
        PairedBaselineConfig(shuffled_chance_tolerance=0.5)


def test_config_to_dict_serializes_paths() -> None:
    config = PairedBaselineConfig()
    data = config.to_dict()
    assert isinstance(data["output_dir"], str)
    assert isinstance(data["bank_checkpoint_dir"], str)


# ---------------------------------------------------------------------------
# _pretrained_core_available: sealed seeds 0-4 have a checkpoint, development
# seeds 10-14 do not (this is the real, checked-in repo state this task's
# SHIFT-reference-adequacy scoping decision depends on).
# ---------------------------------------------------------------------------


def test_pretrained_core_available_for_sealed_seed_0() -> None:
    assert _pretrained_core_available(0) is True


def test_pretrained_core_unavailable_for_development_seed_10() -> None:
    assert _pretrained_core_available(10) is False


def test_pretrained_core_unavailable_for_unknown_seed() -> None:
    assert _pretrained_core_available(999_999) is False


# ---------------------------------------------------------------------------
# _relation_label / _downstream_task_for / _mean: pure mappings.
# ---------------------------------------------------------------------------


def test_relation_label_l3_uses_arrow_notation() -> None:
    assert _relation_label("COUNT", HardNegativeLevel.L3_SEMANTICALLY_RELATED) == "COUNT->BIND"
    assert _relation_label("BIND", HardNegativeLevel.L3_SEMANTICALLY_RELATED) == "BIND->COUNT"


def test_relation_label_l4_uses_argument_variant_notation() -> None:
    assert _relation_label("SELECT", HardNegativeLevel.L4_CONFUSABLE_FAMILY) == "SELECT:argument_variant"


def test_downstream_task_for_known_relations() -> None:
    assert _downstream_task_for("COUNT->BIND") == "B-C005R3-006"
    assert _downstream_task_for("BIND->COUNT") == "B-C005R3-006"
    assert _downstream_task_for("SELECT:argument_variant") == "B-C005R3-007"
    assert _downstream_task_for("BIND:argument_variant") == "B-C005R3-008"


def test_downstream_task_for_unknown_relation() -> None:
    assert _downstream_task_for("NOT_A_RELATION") == "UNKNOWN"


def test_mean_empty_is_none() -> None:
    assert _mean([]) is None


def test_mean_nonempty() -> None:
    assert _mean([0.5, 1.0]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _apply_adequacy_evaluators: real statistical logic over a synthetic stream.
# ---------------------------------------------------------------------------


def test_apply_adequacy_evaluators_all_correct_accepts_everywhere() -> None:
    config = PairedBaselineConfig()
    correct = [True] * 512
    result = _apply_adequacy_evaluators(correct, config, core_pretrained=True)
    assert result["core_pretrained"] is True
    assert result["true_closed_loop_exact_match_full_sample"] == 1.0
    assert result["legacy_asymmetric"]["final_decision"] == AdequacyDecision.ACCEPT.value
    assert result["new_contract_finite_look"]["final_verdict"] == "ACCEPT"
    for look in ADEQUACY_FIXED_NS:
        assert result["fixed_n"][str(look)]["decision"] == AdequacyDecision.ACCEPT.value


def test_apply_adequacy_evaluators_all_wrong_rejects_everywhere() -> None:
    config = PairedBaselineConfig()
    correct = [False] * 512
    result = _apply_adequacy_evaluators(correct, config, core_pretrained=False)
    assert result["core_pretrained"] is False
    assert result["true_closed_loop_exact_match_full_sample"] == 0.0
    assert result["legacy_asymmetric"]["final_decision"] == AdequacyDecision.REJECT.value
    assert result["new_contract_finite_look"]["final_verdict"] == "REJECT"
    for look in ADEQUACY_FIXED_NS:
        assert result["fixed_n"][str(look)]["decision"] == AdequacyDecision.REJECT.value


def test_apply_adequacy_evaluators_premature_accept_divergence() -> None:
    """A stream at ~92% true accuracy (D2-005's SHIFT finding) is exactly the
    case where legacy_asymmetric accepts early on point-estimate while the new
    finite-look contract either rejects or stays UNCERTAIN -- this is the
    divergence the whole comparison exists to surface."""
    config = PairedBaselineConfig()
    correct = [(i % 100) < 92 for i in range(512)]  # 92% true accuracy, deterministic pattern
    result = _apply_adequacy_evaluators(correct, config, core_pretrained=True)
    assert result["true_closed_loop_exact_match_full_sample"] == pytest.approx(0.92, abs=0.01)
    assert result["legacy_asymmetric"]["final_decision"] == AdequacyDecision.ACCEPT.value
    assert result["new_contract_finite_look"]["final_verdict"] != "ACCEPT"


# ---------------------------------------------------------------------------
# _build_paired_baseline: relation grouping/aggregation math.
# ---------------------------------------------------------------------------


def _cell(condition: str, target_operation: str, level: HardNegativeLevel, seed: int, top1: float) -> dict:
    return {
        "condition": condition,
        "target_operation": target_operation,
        "level": level.value,
        "seed": seed,
        "primitive_call_top1": top1,
        "primitive_call_topk": 1.0,
        "physical_primitive_top1": top1,
        "argument_accuracy": 1.0,
        "candidate_rank": 1.0,
        "score_margin": 1.0,
    }


def test_build_paired_baseline_aggregates_per_relation_per_condition() -> None:
    cells = [
        _cell("R0", "COUNT", HardNegativeLevel.L3_SEMANTICALLY_RELATED, 10, 1.0),
        _cell("R0", "COUNT", HardNegativeLevel.L3_SEMANTICALLY_RELATED, 11, 0.8),
        _cell("R2", "COUNT", HardNegativeLevel.L3_SEMANTICALLY_RELATED, 10, 0.4),
        _cell("R2", "COUNT", HardNegativeLevel.L3_SEMANTICALLY_RELATED, 11, 0.4),
    ]
    config = PairedBaselineConfig()
    baseline = _build_paired_baseline(cells, {}, {}, config)
    relation = baseline["retrieval"]["per_relation_by_condition"]["COUNT->BIND"]
    assert relation["R0"]["primitive_call_top1_mean"] == pytest.approx(0.9)
    assert relation["R0"]["n_seeds"] == 2
    assert relation["R2"]["primitive_call_top1_mean"] == pytest.approx(0.4)
    assert baseline["development_seeds"] == list(config.development_seeds)


# ---------------------------------------------------------------------------
# _build_comparison_manifest: structure only (content is static prose).
# ---------------------------------------------------------------------------


def test_build_comparison_manifest_reports_valid_and_discloses_scale() -> None:
    config = PairedBaselineConfig(support_examples=32, query_examples=64)
    manifest = _build_comparison_manifest(config)
    assert manifest["comparison_validity"] == "VALID"
    assert manifest["scale_disclosure"]["support_examples"] == 32
    assert manifest["scale_disclosure"]["query_examples"] == 64
    assert "untrained_core_disclosure" in manifest


# ---------------------------------------------------------------------------
# _build_failure_reproduction_matrix: every classification branch.
# ---------------------------------------------------------------------------


def _paired_baseline_stub(relation_summary: dict, shift_by_seed: dict) -> dict:
    return {
        "retrieval": {"per_relation_by_condition": relation_summary},
        "shift_reference_adequacy": {"by_seed": shift_by_seed},
    }


def test_failure_matrix_reproduces_low_top1_relation() -> None:
    relation_summary = {"COUNT->BIND": {"R2": {"primitive_call_top1_mean": 0.4}}}
    shift_by_seed = {"10": {"true_closed_loop_exact_match_full_sample": 0.5, "core_pretrained": True}}
    baseline = _paired_baseline_stub(relation_summary, shift_by_seed)
    matrix = _build_failure_reproduction_matrix(baseline, {})
    entry = matrix["entries"]["COUNT->BIND"]
    assert entry["status"] == "REPRODUCED_ON_V2"
    assert entry["needs_scope_review"] is False
    assert entry["downstream_task"] == "B-C005R3-006"


def test_failure_matrix_not_reproduced_high_top1_relation() -> None:
    relation_summary = {"COUNT->BIND": {"R2": {"primitive_call_top1_mean": 0.99}}}
    shift_by_seed = {"10": {"true_closed_loop_exact_match_full_sample": 0.5, "core_pretrained": True}}
    baseline = _paired_baseline_stub(relation_summary, shift_by_seed)
    matrix = _build_failure_reproduction_matrix(baseline, {})
    entry = matrix["entries"]["COUNT->BIND"]
    assert entry["status"] == "NOT_REPRODUCED_ON_V2"
    assert entry["needs_scope_review"] is True


def test_failure_matrix_not_run_when_relation_missing() -> None:
    shift_by_seed = {"10": {"true_closed_loop_exact_match_full_sample": 0.5, "core_pretrained": True}}
    baseline = _paired_baseline_stub({}, shift_by_seed)
    matrix = _build_failure_reproduction_matrix(baseline, {})
    entry = matrix["entries"]["COUNT->BIND"]
    assert entry["status"] == "NOT_RUN"
    assert entry["needs_scope_review"] is True


def test_failure_matrix_shift_unverifiable_when_no_pretrained_core() -> None:
    shift_by_seed = {
        "10": {"true_closed_loop_exact_match_full_sample": 0.0, "core_pretrained": False},
        "11": {"true_closed_loop_exact_match_full_sample": 0.0, "core_pretrained": False},
    }
    baseline = _paired_baseline_stub({}, shift_by_seed)
    matrix = _build_failure_reproduction_matrix(baseline, {})
    entry = matrix["entries"]["SHIFT_reference_adequacy"]
    assert entry["status"] == "UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE"
    assert entry["needs_scope_review"] is True


def test_failure_matrix_shift_reproduced_when_pretrained_core_available() -> None:
    shift_by_seed = {"0": {"true_closed_loop_exact_match_full_sample": 0.92, "core_pretrained": True}}
    baseline = _paired_baseline_stub({}, shift_by_seed)
    matrix = _build_failure_reproduction_matrix(baseline, {})
    entry = matrix["entries"]["SHIFT_reference_adequacy"]
    assert entry["status"] == "REPRODUCED_ON_V2"
    assert entry["needs_scope_review"] is False


def test_failure_matrix_select_bind_control_reproduced_when_not_at_chance() -> None:
    shift_by_seed = {"10": {"true_closed_loop_exact_match_full_sample": 0.5, "core_pretrained": True}}
    baseline = _paired_baseline_stub({}, shift_by_seed)
    select_bind_by_seed = {"10": {"shuffled_controls_at_chance": False}}
    matrix = _build_failure_reproduction_matrix(baseline, select_bind_by_seed)
    entry = matrix["entries"]["SELECT->BIND_shuffled_control"]
    assert entry["status"] == "REPRODUCED_ON_V2"
    assert entry["needs_scope_review"] is True


def test_failure_matrix_select_bind_control_not_reproduced_when_all_at_chance() -> None:
    shift_by_seed = {"10": {"true_closed_loop_exact_match_full_sample": 0.5, "core_pretrained": True}}
    baseline = _paired_baseline_stub({}, shift_by_seed)
    select_bind_by_seed = {"10": {"shuffled_controls_at_chance": True}, "11": {"shuffled_controls_at_chance": True}}
    matrix = _build_failure_reproduction_matrix(baseline, select_bind_by_seed)
    entry = matrix["entries"]["SELECT->BIND_shuffled_control"]
    assert entry["status"] == "NOT_REPRODUCED_ON_V2"
    assert entry["needs_scope_review"] is False


def test_original_d2_reference_covers_every_matrix_entry_except_select_bind() -> None:
    """ORIGINAL_D2_REFERENCE must cite a real prior source for each mechanism
    this task compares against; a missing key would silently fall back to no
    citation."""
    expected_keys = {
        "COUNT->BIND",
        "BIND->COUNT",
        "SELECT:argument_variant",
        "BIND:argument_variant",
        "SHIFT_reference_adequacy",
    }
    assert expected_keys <= set(ORIGINAL_D2_REFERENCE)
    for entry in ORIGINAL_D2_REFERENCE.values():
        assert entry["source"].startswith("runs/phase_b_b2_second_diagnostic/")


# ---------------------------------------------------------------------------
# _build_select_bind_control_status: leakage takes priority, then chance, then UNRESOLVED.
# ---------------------------------------------------------------------------


def test_select_bind_status_flags_leakage_first() -> None:
    per_seed = {
        "10": {"probe_leakage_detected": True, "shuffled_controls_at_chance": True},
        "11": {"probe_leakage_detected": False, "shuffled_controls_at_chance": True},
    }
    status = _build_select_bind_control_status(per_seed, PairedBaselineConfig())
    assert status["status"] == "MEASUREMENT_BUG_FOUND_PROBE_LEAKAGE"
    assert status["router_weights_changed"] is False


def test_select_bind_status_resolved_when_all_at_chance_no_leakage() -> None:
    per_seed = {
        "10": {"probe_leakage_detected": False, "shuffled_controls_at_chance": True},
        "11": {"probe_leakage_detected": False, "shuffled_controls_at_chance": True},
    }
    status = _build_select_bind_control_status(per_seed, PairedBaselineConfig())
    assert status["status"] == "RESOLVED_ON_V2_SHUFFLED_CONTROLS_AT_CHANCE"


def test_select_bind_status_unresolved_when_not_at_chance_no_leakage() -> None:
    per_seed = {
        "10": {"probe_leakage_detected": False, "shuffled_controls_at_chance": False},
    }
    status = _build_select_bind_control_status(per_seed, PairedBaselineConfig())
    assert status["status"] == "UNRESOLVED"
