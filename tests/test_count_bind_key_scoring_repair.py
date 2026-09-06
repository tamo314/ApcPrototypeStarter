# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-006's pure logic and its
scoped-training function.

Matches this repo's existing precedent (D2/R3-002/R3-004): anything that
needs a real Core/bank/example generator (`build_pre_repair_diagnostics`,
`audit_no_artificial_score_offset`, `run_count_bind_key_scoring_repair`
itself) is exercised only via the milestone script
(`scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-006`), not in
pytest.

Unlike those, `train_count_bind_scoped_repair`, `build_freeze_audit`, and
`build_checkpoint_hashes` need only a `Router` and plain tensors -- no Core,
bank, or example generator -- so they get real (not just mocked) coverage
here, including training a few real optimizer steps on CPU.
"""

from __future__ import annotations

import copy

import pytest
import torch

from apc.evaluation.count_bind_key_scoring_repair import (
    VARIANTS,
    CountBindKeyScoringConfig,
    _build_variant_selection,
    _mean,
    audit_key_id_correspondence,
    audit_key_norms,
    audit_trainable_parameter_scope,
    build_checkpoint_hashes,
    build_freeze_audit,
    build_loss_exposure_log,
    train_count_bind_scoped_repair,
)
from apc.primitives.router import Router, RouterConfig


class _FakeCore:
    """Stand-in for `apc.evaluation.hard_negative_routing_benchmark`'s real
    frozen-core object: `build_freeze_audit` only needs `core.model` to be a
    hashable `nn.Module`, never anything else about a real Core."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)


def _build_test_router(num_ops: int, *, score_fn: str = "dot", d_model: int = 4) -> tuple[Router, list[int]]:
    router = Router(RouterConfig(d_model=d_model, top_k=1, score_fn=score_fn))
    ids = list(range(num_ops))
    for pid in ids:
        router.add_primitive_key(pid)
    return router, ids


# ---------------------------------------------------------------------------
# CountBindKeyScoringConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = CountBindKeyScoringConfig()
    assert config.bank_size == 128
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.variants == VARIANTS


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        CountBindKeyScoringConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        CountBindKeyScoringConfig(bank_size=100)


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="support_examples"):
        CountBindKeyScoringConfig(support_examples=0)


def test_config_rejects_nonpositive_router_steps() -> None:
    with pytest.raises(ValueError, match="router_steps"):
        CountBindKeyScoringConfig(router_steps=0)


def test_config_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        CountBindKeyScoringConfig(top_k=6)


def test_config_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="adequacy_exact_match_threshold"):
        CountBindKeyScoringConfig(adequacy_exact_match_threshold=1.5)


def test_config_rejects_bad_gate_thresholds() -> None:
    with pytest.raises(ValueError, match="gate_top1_threshold"):
        CountBindKeyScoringConfig(gate_top1_threshold=0.0)


def test_config_rejects_negative_legacy_regression_budget() -> None:
    with pytest.raises(ValueError, match="gate_legacy_regression_pp_max"):
        CountBindKeyScoringConfig(gate_legacy_regression_pp_max=-1.0)


def test_config_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variants"):
        CountBindKeyScoringConfig(variants=("not_a_real_variant",))


def test_config_rejects_empty_variants() -> None:
    with pytest.raises(ValueError, match="variants"):
        CountBindKeyScoringConfig(variants=())


def test_config_to_dict_serializes_paths() -> None:
    config = CountBindKeyScoringConfig()
    data = config.to_dict()
    assert isinstance(data["output_dir"], str)
    assert isinstance(data["bank_checkpoint_dir"], str)


# ---------------------------------------------------------------------------
# audit_key_id_correspondence / audit_key_norms / audit_trainable_parameter_scope:
# pure functions over a real (small, CPU) Router.
# ---------------------------------------------------------------------------


def test_audit_key_id_correspondence_bijective_case() -> None:
    router, ids = _build_test_router(4)
    op_to_id = {"SHIFT": ids[0], "SELECT": ids[1], "COUNT": ids[2], "BIND": ids[3]}
    result = audit_key_id_correspondence(router, op_to_id)
    assert result["bijective_and_registered"] is True
    assert result["duplicate_ids_found"] is False
    assert result["operations_missing_router_key"] == []


def test_audit_key_id_correspondence_detects_missing_key() -> None:
    router, ids = _build_test_router(3)
    op_to_id = {"SHIFT": ids[0], "SELECT": ids[1], "COUNT": 999}
    result = audit_key_id_correspondence(router, op_to_id)
    assert result["bijective_and_registered"] is False
    assert result["operations_missing_router_key"] == ["COUNT"]


def test_audit_key_id_correspondence_detects_duplicate_ids() -> None:
    router, ids = _build_test_router(2)
    op_to_id = {"COUNT": ids[0], "BIND": ids[0]}
    result = audit_key_id_correspondence(router, op_to_id)
    assert result["duplicate_ids_found"] is True
    assert result["bijective_and_registered"] is False


def test_audit_key_norms_reports_dot_score_fn_without_normalization() -> None:
    router, ids = _build_test_router(2, score_fn="dot")
    op_to_id = {"COUNT": ids[0], "BIND": ids[1]}
    with torch.no_grad():
        router.key_parameter(ids[0]).copy_(torch.tensor([3.0, 4.0, 0.0, 0.0]))
        router.key_parameter(ids[1]).copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    result = audit_key_norms(router, op_to_id)
    assert result["score_fn"] == "dot"
    assert result["normalization_applied_to_scoring"] is False
    assert result["count_key_norm"] == pytest.approx(5.0)
    assert result["bind_key_norm"] == pytest.approx(1.0)
    assert result["count_bind_norm_ratio"] == pytest.approx(5.0)


def test_audit_key_norms_reports_cosine_normalization() -> None:
    router, ids = _build_test_router(2, score_fn="cosine")
    op_to_id = {"COUNT": ids[0], "BIND": ids[1]}
    result = audit_key_norms(router, op_to_id)
    assert result["score_fn"] == "cosine"
    assert result["normalization_applied_to_scoring"] is True


def test_audit_key_norms_missing_operations_yield_none_ratio() -> None:
    router, ids = _build_test_router(1)
    result = audit_key_norms(router, {"COUNT": ids[0]})
    assert result["bind_key_norm"] is None
    assert result["count_bind_norm_ratio"] is None


def test_audit_trainable_parameter_scope_lists_only_count_bind_as_trainable() -> None:
    result = audit_trainable_parameter_scope([1, 2, 3, 4], count_id=2, bind_id=4)
    assert result["planned_trainable_key_ids"] == [2, 4]
    assert result["planned_frozen_key_ids"] == [1, 3]
    assert result["query_proj_trainable"] is False
    assert result["argument_scorer_instantiated"] is False
    assert result["task_encoder_trainable"] is False
    assert result["primitive_bank_trainable"] is False
    assert result["verifier_touched"] is False


# ---------------------------------------------------------------------------
# train_count_bind_scoped_repair: real CPU optimizer steps on a small Router.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_scoped_repair_only_changes_count_and_bind_keys(variant: str) -> None:
    router, ids = _build_test_router(6)
    count_id, bind_id = ids[2], ids[3]
    before = {pid: router.key_parameter(pid).detach().clone() for pid in ids}
    before_qp_weight = router.query_proj.weight.detach().clone()

    torch.manual_seed(0)
    z_count = torch.randn(8, 4)
    z_bind = torch.randn(8, 4)

    new_router = train_count_bind_scoped_repair(
        router, ids, count_id, bind_id, z_count, z_bind,
        variant=variant, router_steps=25, router_lr=0.1,
        ranking_margin=3.0, ranking_beta=1.0, seed=1, device=torch.device("cpu"),
    )

    assert not torch.equal(before[count_id], new_router.key_parameter(count_id))
    assert not torch.equal(before[bind_id], new_router.key_parameter(bind_id))
    for pid in ids:
        if pid not in (count_id, bind_id):
            assert torch.equal(before[pid], new_router.key_parameter(pid)), f"non-target key {pid} changed"
    assert torch.equal(before_qp_weight, new_router.query_proj.weight)

    # The original router (deepcopy'd internally) must be untouched.
    for pid in ids:
        assert torch.equal(before[pid], router.key_parameter(pid))


def test_scoped_repair_rejects_unknown_variant() -> None:
    router, ids = _build_test_router(4)
    with pytest.raises(ValueError, match="unknown variant"):
        train_count_bind_scoped_repair(
            router, ids, ids[0], ids[1], torch.randn(4, 4), torch.randn(4, 4),
            variant="not_a_variant", router_steps=5, router_lr=0.1,
            ranking_margin=3.0, ranking_beta=1.0, seed=1, device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# build_freeze_audit / build_checkpoint_hashes.
# ---------------------------------------------------------------------------


def test_freeze_audit_passes_after_real_scoped_training() -> None:
    router, ids = _build_test_router(5)
    count_id, bind_id = ids[0], ids[1]
    scoped = train_count_bind_scoped_repair(
        router, ids, count_id, bind_id, torch.randn(4, 4), torch.randn(4, 4),
        variant="scoped_ce", router_steps=10, router_lr=0.1,
        ranking_margin=3.0, ranking_beta=1.0, seed=2, device=torch.device("cpu"),
    )
    audit = build_freeze_audit(_FakeCore(), router, scoped, ids, count_id, bind_id)
    assert audit["count_key_changed_by_training"] is True
    assert audit["bind_key_changed_by_training"] is True
    assert audit["all_non_target_keys_unchanged"] is True
    assert audit["query_proj_unchanged"] is True
    assert audit["freeze_audit_passed"] is True
    assert audit["task_encoder_touched"] is False


def test_freeze_audit_flags_unexpected_non_target_key_mutation() -> None:
    router, ids = _build_test_router(4)
    count_id, bind_id, other_id = ids[0], ids[1], ids[2]
    mutated = copy.deepcopy(router)
    with torch.no_grad():
        mutated.key_parameter(other_id).add_(1.0)
    audit = build_freeze_audit(_FakeCore(), router, mutated, ids, count_id, bind_id)
    assert audit["all_non_target_keys_unchanged"] is False
    assert audit["non_target_key_unchanged_by_pid"][str(other_id)] is False
    assert audit["freeze_audit_passed"] is False


def test_freeze_audit_flags_unexpected_query_proj_mutation() -> None:
    router, ids = _build_test_router(4)
    count_id, bind_id = ids[0], ids[1]
    mutated = copy.deepcopy(router)
    with torch.no_grad():
        mutated.query_proj.weight.add_(1.0)
    audit = build_freeze_audit(_FakeCore(), router, mutated, ids, count_id, bind_id)
    assert audit["query_proj_unchanged"] is False
    assert audit["freeze_audit_passed"] is False


def test_checkpoint_hashes_reflect_key_changes_and_non_target_invariance() -> None:
    router, ids = _build_test_router(5)
    count_id, bind_id = ids[0], ids[1]
    scoped = train_count_bind_scoped_repair(
        router, ids, count_id, bind_id, torch.randn(4, 4), torch.randn(4, 4),
        variant="scoped_pairwise_margin", router_steps=10, router_lr=0.1,
        ranking_margin=3.0, ranking_beta=1.0, seed=3, device=torch.device("cpu"),
    )
    hashes = build_checkpoint_hashes(router, scoped, ids, count_id, bind_id)
    assert hashes["count_key_hash"]["before"] != hashes["count_key_hash"]["after"]
    assert hashes["bind_key_hash"]["before"] != hashes["bind_key_hash"]["after"]
    assert hashes["sample_non_target_key_hash"], "expected at least one sampled non-target key"
    for entry in hashes["sample_non_target_key_hash"].values():
        assert entry["before"] == entry["after"]


# ---------------------------------------------------------------------------
# build_loss_exposure_log: pure classification.
# ---------------------------------------------------------------------------


def test_loss_exposure_log_classifies_target_vs_frozen_ops() -> None:
    candidate_ids = [1, 2, 3, 4]
    operation_by_id = {1: "SHIFT", 2: "SELECT", 3: "COUNT", 4: "BIND"}
    log = build_loss_exposure_log(candidate_ids, operation_by_id, count_id=3, bind_id=4)
    assert log["per_operation_exposure"]["COUNT"] == "POSITIVE_TRAINING_TARGET_PARAMETER_UPDATED"
    assert log["per_operation_exposure"]["BIND"] == "POSITIVE_TRAINING_TARGET_PARAMETER_UPDATED"
    assert log["per_operation_exposure"]["SHIFT"] == "FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE"
    assert log["per_operation_exposure"]["SELECT"] == "FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE"


# ---------------------------------------------------------------------------
# _mean / _build_variant_selection: pure aggregation logic.
# ---------------------------------------------------------------------------


def test_mean_of_empty_list_is_none() -> None:
    assert _mean([]) is None


def test_mean_of_values() -> None:
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_variant_selection_picks_clear_winner() -> None:
    result = _build_variant_selection({"scoped_ce": [0.9, 0.92], "scoped_pairwise_margin": [0.99, 0.98]})
    assert result["chosen_variant"] == "scoped_pairwise_margin"
    assert result["tie_break_applied"] is False


def test_variant_selection_tie_break_prefers_pairwise_margin() -> None:
    result = _build_variant_selection({"scoped_ce": [0.95, 0.95], "scoped_pairwise_margin": [0.95, 0.95]})
    assert result["chosen_variant"] == "scoped_pairwise_margin"
    assert result["tie_break_applied"] is True


def test_variant_selection_reports_selection_split_disclosure() -> None:
    result = _build_variant_selection({"scoped_ce": [1.0], "scoped_pairwise_margin": [0.5]})
    assert "disjoint" in result["selection_split"]
    assert result["selection_leakage"].startswith("none")
