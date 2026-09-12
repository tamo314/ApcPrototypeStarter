"""Focused CPU invariants for B-C005R3-002R's fixed feasibility protocol."""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.single_family_g1_strict_holdout import (
    PREREGISTERED_FAMILY_ID,
    SingleFamilyG1Config,
    _parameter_probe,
    build_single_family_coupling_graph,
    preregistered_family,
)
from apc.primitives.router import Router, RouterConfig


def test_preregistration_allows_exactly_the_one_parent_family() -> None:
    declaration = preregistered_family()
    assert declaration["exactly_one_family"] is True
    assert declaration["family_id"] == PREREGISTERED_FAMILY_ID
    assert declaration["alternative_families_compared"] == []
    assert declaration["sealed_model_outputs_inspected"] == 0
    with pytest.raises(ValueError, match="exactly one family"):
        SingleFamilyG1Config(family_id="dev_local_difference")


def test_single_family_graph_does_not_miscount_members_as_independent_groups() -> None:
    graph = build_single_family_coupling_graph()
    assert graph["clean_non_alias_non_coupled_component_count"] == 1
    assert graph["sufficiency"] == {"validation": False, "sealed_v2": False}
    component = graph["components"]["family:sealed_local_neighborhood"]
    assert len(component["members"]) == 3
    assert "one preregistered family" in component["internal_coupling_reason"]


def _router() -> Router:
    torch.manual_seed(9)
    router = Router(RouterConfig(d_model=4, top_k=1))
    for primitive_id in (0, 1, 2, 3):
        router.add_primitive_key(primitive_id)
    return router


def test_relation_scoped_ce_has_exact_held_out_gradient_update_and_state_isolation() -> None:
    router = _router()
    representations = torch.tensor(
        [[0.2, -0.3, 0.4, 0.6], [-0.5, 0.7, -0.1, 0.3]], dtype=torch.float32
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    scoped = _parameter_probe(
        router,
        candidate_ids=[0, 1, 2, 3],
        in_scope_ids=[0, 1],
        held_out_ids=[2, 3],
        representations=representations,
        targets=targets,
        relation_scoped=True,
        learning_rate=1e-3,
    )
    assert scoped["held_out_gradient_exactly_zero"] is True
    assert scoped["held_out_key_update_exactly_zero"] is True
    assert scoped["held_out_optimizer_state_absent"] is True
    assert scoped["in_scope_gradient_nonzero"] is True
    assert scoped["in_scope_optimizer_state_present"] is True


def test_existing_full_class_ce_exposes_held_out_keys() -> None:
    router = _router()
    representations = torch.tensor(
        [[0.2, -0.3, 0.4, 0.6], [-0.5, 0.7, -0.1, 0.3]], dtype=torch.float32
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    full_class = _parameter_probe(
        router,
        candidate_ids=[0, 1, 2, 3],
        in_scope_ids=[0, 1],
        held_out_ids=[2, 3],
        representations=representations,
        targets=targets,
        relation_scoped=False,
        learning_rate=1e-3,
    )
    assert full_class["held_out_gradient_exactly_zero"] is False
    assert full_class["held_out_key_update_exactly_zero"] is False
    assert full_class["held_out_optimizer_state_absent"] is False
