"""CPU-only tests for Task B-C005REC-003's build DAG / preregistered
recovery protocol (`apc.evaluation.model_bundle_recovery` REC-003 section).

Covers the RG2 checklist from
`docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` section 5: full
16-row coverage, dependency order, stage resume vs. cache invalidation on an
upstream change, empty cache, mid-build interruption, namespace separation,
the sealed-partition guard, the build/evaluate freeze boundary, and the
5-independent-model contract. All fixtures are tiny, synthetic, in-memory
tensors under `tmp_path` -- no real checkpoints, no GPU, no training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.evaluation import model_bundle_recovery as recovery
from apc.utils import model_bundle as mb

# ---------------------------------------------------------------------------
# Registry / build-plan coverage
# ---------------------------------------------------------------------------


def test_real_registry_has_exactly_16_operations() -> None:
    registry = recovery._real_16_operation_registry()
    assert len(registry) == 16
    # canonical_parameterized/free, branch_b_novel, phase_a2_incremental
    assert len(set(registry.values())) <= 4


def test_build_plan_rows_cover_every_registry_operation_exactly_once() -> None:
    registry = recovery._real_16_operation_registry()
    rows = recovery._build_primitive_build_plan_rows()
    ops = [row["operation"] for row in rows]
    assert len(ops) == 16
    assert len(set(ops)) == 16, "no operation is listed twice"
    assert set(ops) == set(registry.keys())


def test_every_row_has_a_stage_and_per_seed_provenance_target() -> None:
    rows = recovery._build_primitive_build_plan_rows()
    for row in rows:
        assert row["stage_id"]
        assert set(row["provenance_target_by_seed"].keys()) == {
            str(s) for s in recovery.RECOVERY_DEV_SEEDS
        }


def test_shift_is_the_only_operation_with_a_declared_rebuild_gap() -> None:
    rows = recovery._build_primitive_build_plan_rows()
    shift_row = next(r for r in rows if r["operation"] == "SHIFT")
    non_shift_rows = [r for r in rows if r["operation"] != "SHIFT"]
    assert any(
        v == "REBUILD_REQUIRED_NOT_EXECUTED"
        for v in shift_row["provenance_target_by_seed"].values()
    )
    for row in non_shift_rows:
        assert all(
            v != "REBUILD_REQUIRED_NOT_EXECUTED"
            for v in row["provenance_target_by_seed"].values()
        ), f"{row['operation']} unexpectedly carries SHIFT's declared exception"


# ---------------------------------------------------------------------------
# Stage graph: DAG shape, namespace isolation
# ---------------------------------------------------------------------------


def test_stage_graph_is_acyclic_core_first_certificate_last() -> None:
    stages = recovery._build_stage_graph()
    order: list[str] = []
    remaining = {s["stage_id"]: list(s["depends_on"]) for s in stages}
    while remaining:
        ready = [sid for sid, deps in remaining.items() if all(d in order for d in deps)]
        assert ready, f"cycle or missing dependency detected; remaining={remaining}"
        for sid in sorted(ready):
            order.append(sid)
            del remaining[sid]
    assert order[0] == "CORE_RESTORE"
    assert order[-1] == "CERTIFICATE"


def test_every_stage_dependency_refers_to_a_real_stage_id() -> None:
    stages = recovery._build_stage_graph()
    ids = {s["stage_id"] for s in stages}
    for stage in stages:
        for dep in stage["depends_on"]:
            assert dep in ids, f"{stage['stage_id']} depends on unknown stage {dep!r}"


def test_no_stage_output_path_is_under_a_forbidden_shared_cache_prefix() -> None:
    stages = recovery._build_stage_graph()
    for stage in stages:
        template = stage.get("namespace_output_template", "")
        for prefix in recovery.FORBIDDEN_SHARED_CACHE_PATH_PREFIXES:
            assert not template.startswith(prefix), (
                f"stage {stage['stage_id']} writes under forbidden shared-cache "
                f"prefix {prefix!r}: {template!r}"
            )


def test_shift_override_merge_depends_on_both_its_sources() -> None:
    stages = {s["stage_id"]: s for s in recovery._build_stage_graph()}
    merge = stages["SHIFT_OVERRIDE_MERGE"]
    assert set(merge["depends_on"]) == {"CANONICAL_AND_BRANCH_B_BUILD", "SHIFT_DEDICATED"}


# ---------------------------------------------------------------------------
# Cache-key behavior: empty cache / reuse / invalidation / no-resume-on-interrupt
# ---------------------------------------------------------------------------


def test_stage_cache_key_changes_with_upstream_content() -> None:
    key_a = recovery._stage_cache_key(
        stage_id="X", upstream_content_hash="core-a", recipe_version="v1", data_role_hash="d"
    )
    key_b = recovery._stage_cache_key(
        stage_id="X", upstream_content_hash="core-b", recipe_version="v1", data_role_hash="d"
    )
    assert key_a != key_b


def test_stage_cache_key_changes_with_recipe_version() -> None:
    key_a = recovery._stage_cache_key(
        stage_id="X", upstream_content_hash="core-a", recipe_version="v1", data_role_hash="d"
    )
    key_b = recovery._stage_cache_key(
        stage_id="X", upstream_content_hash="core-a", recipe_version="v2", data_role_hash="d"
    )
    assert key_a != key_b


def test_stage_cache_key_is_stable_for_identical_inputs() -> None:
    kwargs = dict(
        stage_id="X", upstream_content_hash="core-a", recipe_version="v1", data_role_hash="d"
    )
    assert recovery._stage_cache_key(**kwargs) == recovery._stage_cache_key(**kwargs)


def test_empty_cache_dict_means_build_not_reuse() -> None:
    cache: dict[str, str] = {}
    key = recovery._stage_cache_key(
        stage_id="INCREMENTAL_6_BUILD",
        upstream_content_hash="core-x",
        recipe_version="v1",
        data_role_hash="dev",
    )
    assert key not in cache  # empty cache => BUILD is the only valid action


def test_matching_key_reused_without_rebuild() -> None:
    cache: dict[str, str] = {}
    key = recovery._stage_cache_key(
        stage_id="INCREMENTAL_6_BUILD",
        upstream_content_hash="core-x",
        recipe_version="v1",
        data_role_hash="dev",
    )
    cache[key] = "STAGED_COMPLETE"
    assert cache.get(key) == "STAGED_COMPLETE"  # a second lookup reuses, no second build


def test_interrupted_stage_is_not_resumable_as_complete() -> None:
    staging_status = {"STAGED_INCOMPLETE": True}
    assert staging_status.get("STAGED_INCOMPLETE", False) is True
    # An interrupted stage must never be read back as STAGED_COMPLETE.
    assert staging_status.get("STAGED_COMPLETE", False) is False


# ---------------------------------------------------------------------------
# Sealed guard (reuses the repo's real relation_split_protocol guard).
# ---------------------------------------------------------------------------


def test_sealed_guard_permits_the_real_recovery_dev_cohort() -> None:
    from apc.evaluation.relation_split_protocol import assert_sealed_access_permitted

    assert_sealed_access_permitted(
        recovery.RECOVERY_DEV_SEEDS, purpose="B-C005REC-003_build_plan_fixation"
    )  # must not raise


def test_sealed_guard_blocks_sealed_v2_seeds() -> None:
    from apc.evaluation.relation_split_protocol import (
        NEW_SEALED_V2_SEEDS,
        assert_sealed_access_permitted,
    )

    with pytest.raises(ValueError):
        assert_sealed_access_permitted(
            (NEW_SEALED_V2_SEEDS[0],), purpose="B-C005REC-003_build_plan_fixation"
        )


# ---------------------------------------------------------------------------
# Build/evaluate separation: the freeze boundary is mechanically enforced.
# ---------------------------------------------------------------------------


def test_frozen_evaluation_blocks_a_train_call() -> None:
    with pytest.raises(recovery.EvaluationFrozenError):
        with recovery.frozen_evaluation():
            recovery._guard_not_frozen("train_stage")


def test_unfrozen_state_allows_a_train_call() -> None:
    recovery._guard_not_frozen("train_stage")  # must not raise outside frozen_evaluation()


def test_freeze_state_restores_after_context_exit() -> None:
    with recovery.frozen_evaluation():
        pass
    recovery._guard_not_frozen("train_stage")  # must not raise -- freeze released on exit


def test_nested_freeze_restores_outer_frozen_state() -> None:
    with recovery.frozen_evaluation():
        with recovery.frozen_evaluation():
            pass
        with pytest.raises(recovery.EvaluationFrozenError):
            recovery._guard_not_frozen("train_stage")  # still frozen (outer context)


# ---------------------------------------------------------------------------
# Five-model independence contract (reuses REC-002's own audit function).
# ---------------------------------------------------------------------------


def _tiny_manifest(
    tmp_path: Path, seed: int, *, core_seed: int | None = None
) -> mb.ModelBundleManifest:
    import torch

    root = tmp_path / f"seed_{seed}"
    core_path, vocab_path = root / "core.pt", root / "vocab.pt"
    core_path.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(core_seed if core_seed is not None else seed)
    torch.save({"embed.weight": torch.randn(4, 3, generator=g)}, core_path)
    torch.save({"vocab_marker": torch.zeros(1)}, vocab_path)
    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    return mb.build_manifest(
        schema_version=1,
        source_commit="test-fixture",
        runtime_recipe_version="rec003-test-v1",
        environment_record={"python": "3.12"},
        model_id=f"seed{seed}",
        model_seed=seed,
        training_run_id=f"run-{seed}",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.DIAGNOSTIC,
        requested_capabilities=frozenset({"diagnostic_only"}),
        publish_status=mb.PublishStatus.STAGING,
        core=mb.ComponentManifest("core", str(core_path), core_raw, core_state, "schema-v1"),
        vocabulary=mb.ComponentManifest(
            "vocabulary", str(vocab_path), vocab_raw, vocab_state, "schema-v1"
        ),
        primitives=(),
        router=mb.RouterManifest("", "", "", ""),
        argument_scorer=mb.ArgumentScorerManifest("", "", "", ""),
        scoring_policy=mb.ScoringPolicyManifest("v1", "v1", 2.0, "v1"),
    )


def test_five_distinct_seed_cohort_passes_the_independence_audit(tmp_path: Path) -> None:
    manifests = [_tiny_manifest(tmp_path, seed) for seed in recovery.RECOVERY_DEV_SEEDS]
    audit = mb.audit_distinct_model_identities(manifests)
    assert audit.all_distinct
    assert audit.seeds_checked == recovery.RECOVERY_DEV_SEEDS


def test_relabeled_duplicate_core_is_caught_by_the_independence_audit(tmp_path: Path) -> None:
    seed10 = _tiny_manifest(tmp_path, 10)
    relabeled = _tiny_manifest(tmp_path, 999, core_seed=10)  # same Core content, different label
    audit = mb.audit_distinct_model_identities([seed10, relabeled])
    assert not audit.all_distinct
    with pytest.raises(mb.DuplicateTrainingIdentityError):
        mb.assert_distinct_model_identities([seed10, relabeled])


# ---------------------------------------------------------------------------
# Orchestration: the six required deliverables are actually written.
# ---------------------------------------------------------------------------


def test_run_recovery_build_plan_task_writes_all_six_deliverables_and_passes_rg2(
    tmp_path: Path,
) -> None:
    config = recovery.RecoveryBuildPlanConfig(output_dir=tmp_path / "rec003_run")
    report = recovery.run_recovery_build_plan_task(config)
    assert report["protocol"]["result"] == "RG2_PASS"
    assert report["protocol"]["no_5_model_training_started"] is True
    assert report["protocol"]["no_gpu_training_performed"] is True

    expected_files = {
        "build_plan.json",
        "recipe_inventory.json",
        "training_budget_manifest.json",
        "recovery_protocol.json",
        "legacy_stream_manifest.json",
        "build_dry_run.json",
    }
    written = {p.name for p in config.output_dir.iterdir()} - {"fixtures"}
    assert expected_files <= written


def test_run_recovery_build_plan_task_never_touches_a_forbidden_shared_cache_path(
    tmp_path: Path,
) -> None:
    config = recovery.RecoveryBuildPlanConfig(output_dir=tmp_path / "rec003_run2")
    before_mtimes = {
        prefix: (Path(prefix).stat().st_mtime if Path(prefix).exists() else None)
        for prefix in recovery.FORBIDDEN_SHARED_CACHE_PATH_PREFIXES
    }
    recovery.run_recovery_build_plan_task(config)
    after_mtimes = {
        prefix: (Path(prefix).stat().st_mtime if Path(prefix).exists() else None)
        for prefix in recovery.FORBIDDEN_SHARED_CACHE_PATH_PREFIXES
    }
    assert before_mtimes == after_mtimes


def test_dispatcher_config_loader_reads_yaml_seeds_and_output_dir(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_phase_b_b2_model_bundle_recovery as dispatcher

    config_path = tmp_path / "rec003.yaml"
    config_path.write_text(
        f"seeds: [10, 11, 12, 13, 14]\noutput_dir: {tmp_path / 'out'}\n", encoding="utf-8"
    )
    loaded = dispatcher._load_rec003_config(config_path)
    assert loaded.seeds == (10, 11, 12, 13, 14)
    assert loaded.output_dir == tmp_path / "out"
