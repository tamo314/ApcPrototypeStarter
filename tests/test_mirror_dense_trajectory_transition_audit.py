"""Tests for B-C005REC-004T: I03 Dense Trajectory Transition Replay Audit."""

from __future__ import annotations

from pathlib import Path

import torch

from apc.evaluation import mirror_dense_trajectory_transition_audit as rec004t
from apc.evaluation import mirror_position_bias_repair as mpbr


def test_constants_and_groups() -> None:
    assert rec004t.REC004T_TASK_ID == "B-C005REC-004T"
    assert rec004t.REC004T_TARGET_OPERATION == "MIRROR_HALVES"
    assert rec004t.REC004T_TARGET_LENGTH == 10
    assert rec004t.REC004T_DECISIVE_INIT == "I03"
    assert rec004t.REC004T_CONTROL_INIT == "I04"
    assert rec004t.REC004T_ORACLE_EM_THRESHOLD == 0.95
    assert rec004t.REC004T_MAX_REPLAY_WINDOW_UPDATES == 1000
    assert rec004t.REC004T_POSITION_4 == 4
    assert rec004t.REC004T_POSITION_5 == 5
    assert rec004t.REC004T_CORRECT_KEY_P4 == 0
    assert rec004t.REC004T_CORRECT_KEY_P5 == 9
    assert len(rec004t.REC004T_PARAMETER_GROUPS) == 10
    assert len(rec004t.REC004T_COARSE_STEPS) == 25


def test_probe_datasets_construction_and_disjointness() -> None:
    datasets, manifest = rec004t.prepare_probe_datasets(rec004t.REC004T_SEED)

    assert rec004t.REC004T_CONTINUITY_SPLIT in datasets
    assert rec004t.REC004T_FRESH_SPLIT in datasets

    cont_exs = datasets[rec004t.REC004T_CONTINUITY_SPLIT]
    fresh_exs = datasets[rec004t.REC004T_FRESH_SPLIT]

    assert len(cont_exs) == rec004t.REC004T_PROBE_EXAMPLES
    assert len(fresh_exs) == rec004t.REC004T_PROBE_EXAMPLES

    for ex in cont_exs:
        assert len(ex.input_tokens) == 10
        assert len(ex.target_tokens) == 10
    for ex in fresh_exs:
        assert len(ex.input_tokens) == 10
        assert len(ex.target_tokens) == 10

    cont_digests = rec004t._digest_examples(cont_exs)
    fresh_digests = rec004t._digest_examples(fresh_exs)

    assert len(cont_digests) == rec004t.REC004T_PROBE_EXAMPLES
    assert len(fresh_digests) == rec004t.REC004T_PROBE_EXAMPLES
    assert len(cont_digests.intersection(fresh_digests)) == 0
    assert manifest["fresh_dataset"]["disjoint_verified"] is True


def test_parameter_group_coverage() -> None:
    core, _, _ = rec004t._load_runtime_core()
    primitive = mpbr._new_arm_primitive(core, rec004t.REC004T_ARM)

    groups = rec004t.get_parameter_group_tensors(primitive)
    assert set(groups.keys()) == set(rec004t.REC004T_PARAMETER_GROUPS)

    # Count total elements across groups
    grouped_numel = sum(sum(t.numel() for t in tensor_list) for tensor_list in groups.values())
    total_primitive_numel = sum(p.numel() for p in primitive.parameters())

    assert grouped_numel == total_primitive_numel, (
        f"Grouped numel {grouped_numel} != primitive numel {total_primitive_numel}"
    )

    # Verify snapshotting and norm
    snapshots = rec004t.snapshot_parameter_groups(primitive)
    for g in rec004t.REC004T_PARAMETER_GROUPS:
        norm = rec004t.compute_group_l2_norm(snapshots[g])
        assert norm > 0.0, f"Group {g} norm should be > 0"
        diff_norm = rec004t.compute_group_l2_norm(snapshots[g], snapshots[g])
        assert diff_norm == 0.0


def test_observer_non_interference() -> None:
    core, _, _ = rec004t._load_runtime_core()
    dummy_state = torch.load(
        rec004t._checkpoint_path(rec004t.REC004T_DECISIVE_INIT, 6000), map_location="cpu"
    )
    primitive = rec004t._new_primitive_from_state(core, dummy_state)

    datasets, _ = rec004t.prepare_probe_datasets(rec004t.REC004T_SEED)
    check = rec004t.verify_observer_non_interference(
        core, primitive, datasets[rec004t.REC004T_CONTINUITY_SPLIT]
    )

    assert check["verified"] is True
    assert check["max_abs_diff"] < 5e-3


def test_source_trajectory_manifest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest = rec004t.build_source_trajectory_manifest(tmp_path)
        assert manifest["task_id"] == "B-C005REC-004T"
        assert manifest["total_coarse_points"] == 25
        assert manifest["all_states_verified"] is True
        assert (tmp_path / "source_trajectory_manifest.json").is_file()
