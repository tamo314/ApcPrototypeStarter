from __future__ import annotations

import pytest
import torch

from apc.plastic.allocator import DEFAULT_PRESETS, Allocator, AllocatorPreset, PresetSpec
from apc.plastic.workspace import PlasticWorkspace
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import PrimitiveConfig, PrimitiveStatus

D_MODEL = 8


def _params(num_transforms: int, rank: int) -> int:
    return num_transforms * D_MODEL * rank * 2  # A (d_model x rank) + B (rank x d_model)


def test_allocate_assigns_sequential_ids_and_records_preset() -> None:
    workspace = PlasticWorkspace()
    ids = workspace.allocate(AllocatorPreset.SMALL, 4, PrimitiveConfig(d_model=D_MODEL, rank=4))
    assert ids == [0, 1, 2, 3]
    assert workspace.ids() == [0, 1, 2, 3]
    assert workspace.preset == AllocatorPreset.SMALL
    assert workspace.is_allocated


def test_allocate_raises_if_already_allocated() -> None:
    workspace = PlasticWorkspace()
    workspace.allocate(AllocatorPreset.SMALL, 4, PrimitiveConfig(d_model=D_MODEL, rank=4))
    with pytest.raises(RuntimeError, match="already"):
        workspace.allocate(AllocatorPreset.SMALL, 4, PrimitiveConfig(d_model=D_MODEL, rank=4))


def test_allocate_rejects_non_positive_count() -> None:
    workspace = PlasticWorkspace()
    with pytest.raises(ValueError, match="num_transforms"):
        workspace.allocate(AllocatorPreset.SMALL, 0, PrimitiveConfig(d_model=D_MODEL, rank=4))


def test_release_clears_workspace_and_returns_transforms() -> None:
    workspace = PlasticWorkspace()
    workspace.allocate(AllocatorPreset.SMALL, 4, PrimitiveConfig(d_model=D_MODEL, rank=4))
    released = workspace.release()
    assert set(released) == {0, 1, 2, 3}
    assert workspace.ids() == []
    assert not workspace.is_allocated
    assert workspace.preset is None
    assert workspace.allocated_at_task is None


def test_new_allocation_after_release_does_not_reuse_ids() -> None:
    workspace = PlasticWorkspace()
    workspace.allocate(AllocatorPreset.SMALL, 2, PrimitiveConfig(d_model=D_MODEL, rank=4))
    workspace.release()
    ids = workspace.allocate(AllocatorPreset.SMALL, 2, PrimitiveConfig(d_model=D_MODEL, rank=4))
    assert ids == [2, 3]


def test_get_unknown_id_raises() -> None:
    workspace = PlasticWorkspace()
    with pytest.raises(KeyError):
        workspace.get(0)


def test_get_many_preserves_requested_order() -> None:
    workspace = PlasticWorkspace()
    workspace.allocate(AllocatorPreset.SMALL, 3, PrimitiveConfig(d_model=D_MODEL, rank=4))
    fetched = workspace.get_many([2, 0, 1])
    assert [t.primitive_id for t in fetched] == [2, 0, 1]


def test_total_parameter_count_reflects_allocate_and_release() -> None:
    workspace = PlasticWorkspace()
    assert workspace.total_parameter_count() == 0

    workspace.allocate(AllocatorPreset.MEDIUM, 8, PrimitiveConfig(d_model=D_MODEL, rank=8))
    assert workspace.total_parameter_count() == _params(8, 8)

    workspace.release()
    assert workspace.total_parameter_count() == 0


def test_active_parameter_count_respects_selection_and_enabled_flag() -> None:
    workspace = PlasticWorkspace()
    ids = workspace.allocate(AllocatorPreset.SMALL, 4, PrimitiveConfig(d_model=D_MODEL, rank=4))

    assert workspace.active_parameter_count(ids) == _params(4, 4)
    assert workspace.active_parameter_count(ids[:1]) == _params(1, 4)
    assert workspace.active_parameter_count([]) == 0

    workspace.get(ids[0]).enabled = False
    assert workspace.active_parameter_count(ids) == _params(3, 4)


def test_freeze_all_and_unfreeze_all() -> None:
    workspace = PlasticWorkspace()
    ids = workspace.allocate(AllocatorPreset.SMALL, 2, PrimitiveConfig(d_model=D_MODEL, rank=4))

    assert all(not workspace.get(tid).is_frozen() for tid in ids)
    workspace.freeze_all()
    assert all(workspace.get(tid).is_frozen() for tid in ids)
    workspace.unfreeze_all()
    assert all(not workspace.get(tid).is_frozen() for tid in ids)


def test_usage_counts_and_record_usage() -> None:
    workspace = PlasticWorkspace()
    ids = workspace.allocate(AllocatorPreset.SMALL, 2, PrimitiveConfig(d_model=D_MODEL, rank=4))
    assert workspace.usage_counts() == {ids[0]: 0, ids[1]: 0}

    workspace.record_usage([ids[0], ids[0], ids[1]])
    assert workspace.usage_counts() == {ids[0]: 2, ids[1]: 1}


def test_forward_uses_same_execution_interface_as_a_persistent_primitive() -> None:
    workspace = PlasticWorkspace()
    ids = workspace.allocate(AllocatorPreset.SMALL, 1, PrimitiveConfig(d_model=D_MODEL, rank=4))
    transform = workspace.get(ids[0])
    h = torch.randn(3, D_MODEL)
    # B is zero-initialized, so a fresh transform is a no-op, same as `Primitive`.
    assert torch.equal(transform(h), h)


def test_allocator_default_presets_match_architecture_doc() -> None:
    allocator = Allocator(d_model=D_MODEL)
    assert allocator.spec_for(AllocatorPreset.SMALL) == PresetSpec(num_transforms=4, rank=4)
    assert allocator.spec_for(AllocatorPreset.MEDIUM) == PresetSpec(num_transforms=8, rank=8)
    assert allocator.spec_for(AllocatorPreset.LARGE) == PresetSpec(num_transforms=16, rank=16)
    assert allocator.presets == DEFAULT_PRESETS


def test_allocator_allocate_wires_resolved_preset_into_workspace() -> None:
    workspace = PlasticWorkspace()
    allocator = Allocator(d_model=D_MODEL)
    ids = allocator.allocate(workspace, AllocatorPreset.MEDIUM)
    assert len(ids) == 8
    assert workspace.total_parameter_count() == _params(8, 8)
    assert workspace.preset == AllocatorPreset.MEDIUM
    assert workspace.allocated_at_task == 0


def test_allocator_forwards_created_at_task() -> None:
    workspace = PlasticWorkspace()
    allocator = Allocator(d_model=D_MODEL)
    allocator.allocate(workspace, AllocatorPreset.SMALL, created_at_task=7)
    assert workspace.allocated_at_task == 7


def test_allocator_rejects_incomplete_custom_presets() -> None:
    with pytest.raises(ValueError, match="missing"):
        Allocator(
            d_model=D_MODEL,
            presets={AllocatorPreset.SMALL: PresetSpec(num_transforms=2, rank=2)},
        )


def test_allocator_accepts_overridden_presets() -> None:
    custom = {
        AllocatorPreset.SMALL: PresetSpec(num_transforms=1, rank=2),
        AllocatorPreset.MEDIUM: PresetSpec(num_transforms=2, rank=2),
        AllocatorPreset.LARGE: PresetSpec(num_transforms=3, rank=2),
    }
    allocator = Allocator(d_model=D_MODEL, presets=custom)
    workspace = PlasticWorkspace()
    ids = allocator.allocate(workspace, AllocatorPreset.LARGE)
    assert len(ids) == 3


def test_preset_spec_rejects_non_positive_fields() -> None:
    with pytest.raises(ValueError, match="num_transforms"):
        PresetSpec(num_transforms=0, rank=4)
    with pytest.raises(ValueError, match="rank"):
        PresetSpec(num_transforms=4, rank=0)


# --- Task 007 acceptance criteria -----------------------------------------
# "temporary parameter count changes without changing persistent count;
#  frozen persistent weights remain unchanged after a plastic training step."


def test_temporary_parameter_count_changes_without_changing_persistent_count() -> None:
    bank = PrimitiveBank()
    bank.new_primitive(PrimitiveConfig(d_model=D_MODEL, rank=2), status=PrimitiveStatus.STABLE)
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    persistent_before = bank.persistent_parameter_count()

    workspace = PlasticWorkspace()
    allocator = Allocator(d_model=D_MODEL)

    allocator.allocate(workspace, AllocatorPreset.SMALL)
    assert bank.persistent_parameter_count() == persistent_before
    assert workspace.total_parameter_count() == _params(4, 4)

    workspace.release()
    assert bank.persistent_parameter_count() == persistent_before
    assert workspace.total_parameter_count() == 0

    allocator.allocate(workspace, AllocatorPreset.LARGE)
    assert bank.persistent_parameter_count() == persistent_before
    assert workspace.total_parameter_count() == _params(16, 16)


def test_frozen_persistent_weights_unchanged_after_plastic_training_step() -> None:
    bank = PrimitiveBank()
    stable = bank.new_primitive(
        PrimitiveConfig(d_model=D_MODEL, rank=2), status=PrimitiveStatus.STABLE
    )
    with torch.no_grad():
        stable.b_proj.weight.add_(1.0)  # non-zero so an update would be visible if not frozen
    bank.freeze_by_status(PrimitiveStatus.STABLE)
    stable_before = stable.b_proj.weight.detach().clone()

    workspace = PlasticWorkspace()
    Allocator(d_model=D_MODEL).allocate(workspace, AllocatorPreset.SMALL)
    transform = workspace.get(workspace.ids()[0])
    with torch.no_grad():
        transform.b_proj.weight.add_(1.0)
    transform_before = transform.b_proj.weight.detach().clone()

    all_params = list(bank.parameters()) + list(workspace.parameters())
    trainable = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)

    h = torch.randn(4, D_MODEL)
    out = transform(stable(h))
    loss = out.pow(2).sum()
    loss.backward()
    optimizer.step()

    assert torch.equal(stable.b_proj.weight, stable_before)
    assert not torch.equal(transform.b_proj.weight, transform_before)
