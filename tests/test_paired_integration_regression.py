# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-010's pure logic.

Matches this repo's existing precedent (R3-006/007/008/009): anything that
needs a real Core/bank/router/scorer/checkpoint (`run_paired_integration_regression`,
`_build_seed_conditions`, `_evaluate_nominal_cells`, `_evaluate_shift`,
`_evaluate_deterministic_ops`, `_evaluate_safety`) is exercised only via the
milestone script (`scripts/run_phase_b_b2_post_d2_repair.py --task
B-C005R3-010`), not in pytest. What pytest covers here needs no Core, no GPU,
no pretrained checkpoint: config validation, the pure gate-computation
helpers (`_nominal_gate`, `_safety_gate`, `_legacy_gate`, `_shift_gate`,
`_mean`, `_relation_id_for`), `_state_dict_hash`/`_build_freeze_audit` against
small hand-built `nn.Module` stand-ins, and `_resolve_shift_replacement`'s
non-COMMITTED branches (the COMMITTED branch needs a real Core + checkpoint,
exercised only by the milestone run).
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from apc.evaluation.functional_metrics_v2 import (
    CandidateEvaluationRecord,
    CandidateVerdict,
    ReferenceAdequacyState,
    aggregate_candidate_metrics,
)
from apc.evaluation.paired_integration_regression import (
    CONDITIONS,
    NEVER_REPAIRED_DETERMINISTIC_OPS,
    ConditionSystem,
    PairedIntegrationRegressionConfig,
    _build_freeze_audit,
    _legacy_gate,
    _mean,
    _nominal_gate,
    _relation_id_for,
    _resolve_shift_replacement,
    _safety_gate,
    _shift_gate,
    _state_dict_hash,
)
from apc.meta.phase_b_protocol import HardNegativeLevel

# ---------------------------------------------------------------------------
# PairedIntegrationRegressionConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = PairedIntegrationRegressionConfig()
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.bank_size == 128
    assert config.nominal_target_operations == ("SELECT", "COUNT", "BIND")
    assert len(config.nominal_levels) == 5


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        PairedIntegrationRegressionConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        PairedIntegrationRegressionConfig(bank_size=8)


def test_config_rejects_empty_target_operations() -> None:
    with pytest.raises(ValueError, match="nominal_target_operations"):
        PairedIntegrationRegressionConfig(nominal_target_operations=())


def test_config_rejects_empty_levels() -> None:
    with pytest.raises(ValueError, match="nominal_levels"):
        PairedIntegrationRegressionConfig(nominal_levels=())


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="support_examples"):
        PairedIntegrationRegressionConfig(support_examples=0)
    with pytest.raises(ValueError, match="shift_query_examples"):
        PairedIntegrationRegressionConfig(shift_query_examples=0)
    with pytest.raises(ValueError, match="legacy_deterministic_eval_examples"):
        PairedIntegrationRegressionConfig(legacy_deterministic_eval_examples=0)


def test_config_rejects_safety_examples_below_max_look() -> None:
    with pytest.raises(ValueError, match="safety_verification_examples"):
        PairedIntegrationRegressionConfig(safety_verification_examples=10)


def test_config_rejects_bad_tau() -> None:
    with pytest.raises(ValueError, match="tau"):
        PairedIntegrationRegressionConfig(tau=1.5)


def test_config_rejects_unknown_count_bind_variant() -> None:
    with pytest.raises(ValueError, match="count_bind_variant"):
        PairedIntegrationRegressionConfig(count_bind_variant="not_a_real_variant")


def test_config_rejects_unknown_bind_head_variant() -> None:
    with pytest.raises(ValueError, match="bind_head_variant"):
        PairedIntegrationRegressionConfig(bind_head_variant="not_a_real_variant")


def test_config_rejects_negative_legacy_budgets() -> None:
    with pytest.raises(ValueError, match="regression budgets"):
        PairedIntegrationRegressionConfig(legacy_mean_regression_pp_max=-0.01)


def test_config_to_dict_serializes_paths_and_levels() -> None:
    config = PairedIntegrationRegressionConfig()
    data = config.to_dict()
    assert isinstance(data["output_dir"], str)
    assert isinstance(data["nominal_levels"], list)
    assert data["nominal_levels"][0] == HardNegativeLevel.L0_ORTHOGONAL.value


# ---------------------------------------------------------------------------
# _mean / _relation_id_for.
# ---------------------------------------------------------------------------


def test_mean_of_empty_is_none() -> None:
    assert _mean([]) is None
    assert _mean([1.0, 3.0]) == 2.0


def test_relation_id_for_wrong_family_and_wrong_argument() -> None:
    assert _relation_id_for("COUNT", "wrong_family") == "COUNT->BIND"
    assert _relation_id_for("BIND", "wrong_argument") == "BIND:argument_variant"
    assert _relation_id_for("SELECT", "adequate") is None


# ---------------------------------------------------------------------------
# _state_dict_hash / _build_freeze_audit against small nn.Module stand-ins
# (the ladder's freeze-audit logic only needs .state_dict(); it does not care
# whether the module is a real PrimitiveBank/Router/ArgumentScorer).
# ---------------------------------------------------------------------------


def _seed_ctx_from_systems(conditions: dict[str, ConditionSystem]) -> dict:
    return {"conditions": conditions}


def _linear(seed: int, out_features: int = 4) -> nn.Linear:
    generator = torch.Generator().manual_seed(seed)
    layer = nn.Linear(4, out_features)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
        layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator))
    return layer


def test_state_dict_hash_is_stable_and_sensitive_to_weights() -> None:
    a = _linear(0)
    b = _linear(1)
    assert _state_dict_hash(a) != _state_dict_hash(b)
    assert _state_dict_hash(a) == _state_dict_hash(a)


def test_build_freeze_audit_detects_exactly_one_change_per_transition() -> None:
    bank0, router0, scorer0 = _linear(10), _linear(20), _linear(30)
    router2 = _linear(21)  # only the router differs starting at C2

    conditions = {
        "C0": ConditionSystem(bank0, router0, scorer0, "legacy"),
        "C1": ConditionSystem(bank0, router0, scorer0, "new"),
        "C2": ConditionSystem(bank0, router2, scorer0, "new"),
        "C3": ConditionSystem(bank0, router2, scorer0, "new"),
        "C4": ConditionSystem(bank0, router2, scorer0, "new"),
        "C5": ConditionSystem(bank0, router2, scorer0, "new"),
    }
    audit = _build_freeze_audit(_seed_ctx_from_systems(conditions))
    assert audit["transitions"]["C0->C1"] == {"bank_changed": False, "router_changed": False, "scorer_changed": False}
    assert audit["transitions"]["C1->C2"] == {"bank_changed": False, "router_changed": True, "scorer_changed": False}
    assert audit["transitions"]["C2->C3"] == {"bank_changed": False, "router_changed": False, "scorer_changed": False}
    assert audit["transitions"]["C4->C5"] == {"bank_changed": False, "router_changed": False, "scorer_changed": False}


# ---------------------------------------------------------------------------
# _resolve_shift_replacement: non-COMMITTED branches only (no Core needed).
# ---------------------------------------------------------------------------


def test_resolve_shift_replacement_missing_log_file(tmp_path) -> None:
    config = PairedIntegrationRegressionConfig(r3009_bank_transaction_log=tmp_path / "missing.json")
    status, primitive = _resolve_shift_replacement(core=None, seed=10, config=config)
    assert status == "R3_009_ARTIFACT_UNAVAILABLE"
    assert primitive is None


def test_resolve_shift_replacement_rolled_back_seed(tmp_path) -> None:
    log_path = tmp_path / "bank_transaction_log.json"
    log_path.write_text(json.dumps({"12": {"action": "ROLLED_BACK"}}), encoding="utf-8")
    config = PairedIntegrationRegressionConfig(r3009_bank_transaction_log=log_path)
    status, primitive = _resolve_shift_replacement(core=None, seed=12, config=config)
    assert status == "ROLLED_BACK_NO_CHANGE"
    assert primitive is None


def test_resolve_shift_replacement_seed_absent_from_log(tmp_path) -> None:
    log_path = tmp_path / "bank_transaction_log.json"
    log_path.write_text(json.dumps({"10": {"action": "COMMITTED", "committed_checkpoint": "x"}}), encoding="utf-8")
    config = PairedIntegrationRegressionConfig(r3009_bank_transaction_log=log_path)
    status, primitive = _resolve_shift_replacement(core=None, seed=99, config=config)
    assert status == "ROLLED_BACK_NO_CHANGE"
    assert primitive is None


# ---------------------------------------------------------------------------
# _nominal_gate.
# ---------------------------------------------------------------------------


def _cell(op: str, level: HardNegativeLevel, *, call_top1: float, call_top5: float, physical_top1: float, arg_acc: float) -> dict:
    return {
        "target_operation": op,
        "level": level.value,
        "primitive_call_top1": call_top1,
        "primitive_call_topk": call_top5,
        "physical_primitive_top1": physical_top1,
        "argument_accuracy": arg_acc,
    }


def _all_passing_cells(config: PairedIntegrationRegressionConfig) -> list[dict]:
    cells = []
    for op in config.nominal_target_operations:
        for level in config.nominal_levels:
            cells.append(_cell(op, level, call_top1=1.0, call_top5=1.0, physical_top1=1.0, arg_acc=1.0))
    return cells


def test_nominal_gate_passes_when_everything_is_perfect() -> None:
    config = PairedIntegrationRegressionConfig()
    gate = _nominal_gate(_all_passing_cells(config), config)
    assert gate["pass"] is True
    assert gate["reasons"] == []


def test_nominal_gate_fails_on_low_l3_top1() -> None:
    config = PairedIntegrationRegressionConfig()
    cells = _all_passing_cells(config)
    for row in cells:
        if row["level"] == HardNegativeLevel.L3_SEMANTICALLY_RELATED.value and row["target_operation"] == "COUNT":
            row["primitive_call_top1"] = 0.5
    gate = _nominal_gate(cells, config)
    assert gate["pass"] is False
    assert any("L3_semantically_related" in reason for reason in gate["reasons"])


def test_nominal_gate_fails_on_low_argument_accuracy() -> None:
    config = PairedIntegrationRegressionConfig()
    cells = _all_passing_cells(config)
    for row in cells:
        if row["level"] == HardNegativeLevel.L4_CONFUSABLE_FAMILY.value and row["target_operation"] == "SELECT":
            row["argument_accuracy"] = 0.1
    gate = _nominal_gate(cells, config)
    assert gate["pass"] is False
    assert any("SELECT: argument_accuracy" in reason for reason in gate["reasons"])
    assert gate["argument_accuracy_by_op"]["COUNT"]["pass"] is True


# ---------------------------------------------------------------------------
# _legacy_gate.
# ---------------------------------------------------------------------------


def _op_em_map(value: float) -> dict[str, float]:
    ops = ("SELECT", "COUNT", "BIND", "SHIFT", *NEVER_REPAIRED_DETERMINISTIC_OPS)
    return dict.fromkeys(ops, value)


def test_legacy_gate_passes_with_no_regression() -> None:
    config = PairedIntegrationRegressionConfig()
    per_condition = {c: _op_em_map(1.0) for c in CONDITIONS}
    gate = _legacy_gate(per_condition, config)
    assert gate["pass"] is True
    assert gate["mean_regression_pp"] == 0.0


def test_legacy_gate_fails_on_worst_operation_regression() -> None:
    config = PairedIntegrationRegressionConfig()
    per_condition = {c: _op_em_map(1.0) for c in CONDITIONS}
    per_condition["C5"] = dict(per_condition["C5"])
    per_condition["C5"]["SHIFT"] = 0.5  # 50pp drop on one op only
    gate = _legacy_gate(per_condition, config)
    assert gate["pass"] is False
    assert gate["worst_operation"] == "SHIFT"
    assert gate["worst_regression_pp"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _shift_gate.
# ---------------------------------------------------------------------------


def test_shift_gate_passes_when_all_adequate() -> None:
    config = PairedIntegrationRegressionConfig()
    per_seed = {
        str(seed): {"closed_loop_exact_match": 1.0, "reference_adequacy_state": "REF_ADEQUATE"}
        for seed in config.development_seeds
    }
    gate = _shift_gate(per_seed, config)
    assert gate["pass"] is True
    assert gate["mean_query_exact_match"] == 1.0


def test_shift_gate_fails_when_some_seeds_inadequate() -> None:
    config = PairedIntegrationRegressionConfig()
    per_seed = {
        "10": {"closed_loop_exact_match": 1.0, "reference_adequacy_state": "REF_ADEQUATE"},
        "12": {"closed_loop_exact_match": 0.9, "reference_adequacy_state": "REF_INADEQUATE"},
    }
    gate = _shift_gate(per_seed, config)
    assert gate["pass"] is False
    assert gate["all_ref_adequate"] is False


# ---------------------------------------------------------------------------
# _safety_gate (against aggregate_candidate_metrics's real output shape).
# ---------------------------------------------------------------------------


def _record(*, identity_match: bool, ref_state: ReferenceAdequacyState | None, verdict: CandidateVerdict) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        episode_id="ep", candidate_id="c", operation="SELECT", relation_id=None, model_seed=10,
        identity_match=identity_match, functionally_equivalent=False, reference_state=ref_state, verdict=verdict,
    )


def test_safety_gate_passes_when_nothing_unsafe_is_accepted() -> None:
    config = PairedIntegrationRegressionConfig()
    records = [
        _record(identity_match=True, ref_state=ReferenceAdequacyState.REF_ADEQUATE, verdict=CandidateVerdict.ACCEPT),
        _record(identity_match=False, ref_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.REJECT),
    ]
    gate = _safety_gate(aggregate_candidate_metrics(records), config)
    assert gate["pass"] is True


def test_safety_gate_fails_when_inadequate_call_is_accepted() -> None:
    config = PairedIntegrationRegressionConfig()
    records = [
        _record(identity_match=True, ref_state=ReferenceAdequacyState.REF_INADEQUATE, verdict=CandidateVerdict.ACCEPT),
    ]
    gate = _safety_gate(aggregate_candidate_metrics(records), config)
    assert gate["pass"] is False
    assert gate["reasons"]
