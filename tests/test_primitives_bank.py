from __future__ import annotations

import pytest
import torch

from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import Primitive, PrimitiveConfig, PrimitiveStatus

CONFIG = PrimitiveConfig(d_model=8, rank=2)
PARAMS_PER_PRIMITIVE = 8 * 2 * 2  # A (d_model x rank) + B (rank x d_model)


def _filled_bank(n_stable: int = 2, n_candidate: int = 1) -> PrimitiveBank:
    bank = PrimitiveBank()
    for _ in range(n_stable):
        bank.new_primitive(CONFIG, status=PrimitiveStatus.STABLE)
    for _ in range(n_candidate):
        bank.new_primitive(CONFIG, status=PrimitiveStatus.CANDIDATE)
    return bank


def test_new_primitive_assigns_sequential_ids() -> None:
    bank = PrimitiveBank()
    p0 = bank.new_primitive(CONFIG)
    p1 = bank.new_primitive(CONFIG)
    assert (p0.primitive_id, p1.primitive_id) == (0, 1)
    assert bank.ids() == [0, 1]


def test_add_primitive_rejects_duplicate_id() -> None:
    bank = PrimitiveBank()
    bank.add_primitive(Primitive(0, CONFIG))
    with pytest.raises(ValueError, match="already exists"):
        bank.add_primitive(Primitive(0, CONFIG))


def test_add_primitive_advances_next_id() -> None:
    bank = PrimitiveBank()
    bank.add_primitive(Primitive(5, CONFIG))
    next_primitive = bank.new_primitive(CONFIG)
    assert next_primitive.primitive_id == 6


def test_get_unknown_id_raises() -> None:
    bank = PrimitiveBank()
    with pytest.raises(KeyError):
        bank.get(42)


def test_get_many_preserves_requested_order() -> None:
    bank = _filled_bank(n_stable=3, n_candidate=0)
    fetched = bank.get_many([2, 0, 1])
    assert [p.primitive_id for p in fetched] == [2, 0, 1]


def test_total_parameter_count_includes_every_status() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=1)
    bank.archive(bank.ids()[0])
    assert bank.total_parameter_count() == 3 * PARAMS_PER_PRIMITIVE


def test_persistent_parameter_count_only_counts_enabled_stable() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=1)
    assert bank.persistent_parameter_count() == 2 * PARAMS_PER_PRIMITIVE

    stable_id = bank.ids_by_status(PrimitiveStatus.STABLE)[0]
    bank.disable(stable_id)
    assert bank.persistent_parameter_count() == 1 * PARAMS_PER_PRIMITIVE


def test_active_parameter_count_respects_selection_and_enabled_flag() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=1)
    all_ids = bank.ids()

    assert bank.active_parameter_count(all_ids) == 3 * PARAMS_PER_PRIMITIVE
    assert bank.active_parameter_count(all_ids[:1]) == PARAMS_PER_PRIMITIVE
    assert bank.active_parameter_count([]) == 0

    bank.disable(all_ids[0])
    assert bank.active_parameter_count(all_ids) == 2 * PARAMS_PER_PRIMITIVE


def test_active_parameter_count_is_not_limited_to_stable_status() -> None:
    # A candidate under evaluation is still "active" for the forward step
    # that uses it; persistent accounting is a separate, stricter count.
    bank = _filled_bank(n_stable=0, n_candidate=1)
    candidate_id = bank.ids()[0]
    assert bank.active_parameter_count([candidate_id]) == PARAMS_PER_PRIMITIVE
    assert bank.persistent_parameter_count() == 0


def test_archive_disables_and_marks_status() -> None:
    bank = _filled_bank(n_stable=1, n_candidate=0)
    pid = bank.ids()[0]
    bank.archive(pid)
    primitive = bank.get(pid)
    assert primitive.status == PrimitiveStatus.ARCHIVED
    assert primitive.enabled is False
    assert bank.persistent_parameter_count() == 0


def test_freeze_by_status_only_freezes_matching_primitives() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=1)
    bank.freeze_by_status(PrimitiveStatus.STABLE)

    for pid in bank.ids_by_status(PrimitiveStatus.STABLE):
        assert bank.get(pid).is_frozen()
    for pid in bank.ids_by_status(PrimitiveStatus.CANDIDATE):
        assert not bank.get(pid).is_frozen()


def test_frozen_stable_primitives_do_not_move_under_an_optimizer_step() -> None:
    bank = _filled_bank(n_stable=1, n_candidate=1)
    for pid in bank.ids():
        with torch.no_grad():
            bank.get(pid).b_proj.weight.add_(1.0)
    bank.freeze_by_status(PrimitiveStatus.STABLE)

    stable_id = bank.ids_by_status(PrimitiveStatus.STABLE)[0]
    candidate_id = bank.ids_by_status(PrimitiveStatus.CANDIDATE)[0]
    stable_before = bank.get(stable_id).b_proj.weight.detach().clone()
    candidate_before = bank.get(candidate_id).b_proj.weight.detach().clone()

    trainable = [p for p in bank.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    h = torch.randn(4, CONFIG.d_model)
    out = h
    for pid in bank.ids():
        out = bank.get(pid)(out)
    loss = out.pow(2).sum()
    loss.backward()
    optimizer.step()

    assert torch.equal(bank.get(stable_id).b_proj.weight, stable_before)
    assert not torch.equal(bank.get(candidate_id).b_proj.weight, candidate_before)


def test_persistent_parameter_count_trainable_only_reflects_freeze() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=1)
    assert bank.persistent_parameter_count(trainable_only=True) == 2 * PARAMS_PER_PRIMITIVE

    bank.freeze_by_status(PrimitiveStatus.STABLE)
    assert bank.persistent_parameter_count(trainable_only=True) == 0
    assert bank.persistent_parameter_count() == 2 * PARAMS_PER_PRIMITIVE


def test_usage_counts_and_record_usage() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=0)
    ids = bank.ids()
    assert bank.usage_counts() == {ids[0]: 0, ids[1]: 0}

    bank.record_usage([ids[0], ids[0], ids[1]])
    assert bank.usage_counts() == {ids[0]: 2, ids[1]: 1}


def test_update_utility_delegates_to_primitive() -> None:
    bank = _filled_bank(n_stable=1, n_candidate=0)
    pid = bank.ids()[0]
    bank.update_utility(pid, 1.0, decay=0.0)
    assert bank.get(pid).utility_ema == 1.0


def test_len_reflects_all_registered_primitives() -> None:
    bank = _filled_bank(n_stable=2, n_candidate=3)
    assert len(bank) == 5
