"""CPU-only contract tests for Task B-C005REC-004 (One-Seed Restore / Clean
Build & Fresh-Process Validation).

Per the task's own boundary and `docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_
ADDENDUM.md` ("重い学習を伴うREC-004... はunit testから呼ばない"): the real
seed-10 build (GPU training of 15 primitives + router + scorer calibration)
is never invoked from this test module. These tests cover the structural/
contract logic that does not require GPU training: the operation-name
extension helpers, the manifest JSON round-trip used by the fresh-process
subprocess, the shared-cache hash snapshot, the pre-registered-seed guard,
and the dispatcher's config loader. `runs/phase_b_b2_model_bundle_recovery/
rec004/run_001/`'s real deliverables are produced by the milestone command,
not by this suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apc.evaluation import model_bundle_recovery as rec
from apc.utils import model_bundle as mb

# ---------------------------------------------------------------------------
# _extra_8_ops / _extra_8_wrong_family_map
# ---------------------------------------------------------------------------


def test_extra_8_ops_is_branch_b_plus_incremental_no_overlap_with_canonical() -> None:
    from apc.evaluation.unified_oracle_causal_benchmark import ALL_CANONICAL_OPERATIONS

    extra = rec._extra_8_ops()
    assert len(extra) == 8
    assert len(set(extra)) == 8
    assert set(extra).isdisjoint(set(ALL_CANONICAL_OPERATIONS))


def test_extra_8_wrong_family_map_is_a_derangement() -> None:
    ops = rec._extra_8_ops()
    wrong_map = rec._extra_8_wrong_family_map()
    assert set(wrong_map.keys()) == set(ops)
    assert set(wrong_map.values()) == set(ops)
    for op, wrong_op in wrong_map.items():
        assert wrong_op != op


# ---------------------------------------------------------------------------
# PilotRestoreBuildConfig defaults / seed pre-registration guard
# ---------------------------------------------------------------------------


def test_pilot_config_defaults_match_recovery_protocol() -> None:
    config = rec.PilotRestoreBuildConfig()
    assert config.seed == rec.RECOVERY_PILOT_SEED == 10
    assert config.direct_query_examples_per_operation == 1024
    assert config.non_shift_floor == pytest.approx(0.95)


def test_run_pilot_restore_build_task_refuses_a_non_pilot_seed(tmp_path: Path) -> None:
    bad_config = rec.PilotRestoreBuildConfig(output_dir=tmp_path, seed=11)
    with pytest.raises(ValueError, match="pre-registered"):
        rec.run_pilot_restore_build_task(bad_config)


# ---------------------------------------------------------------------------
# _read_rec001_recorded_hash
# ---------------------------------------------------------------------------


def test_read_rec001_recorded_hash_returns_none_when_inventory_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert rec._read_rec001_recorded_hash("core", 10) is None


def test_read_rec001_recorded_hash_reads_real_seed10_core_row_if_present() -> None:
    inv_path = Path("runs/phase_b_b2_model_bundle_recovery/rec001/run_001/artifact_inventory.json")
    if not inv_path.is_file():
        pytest.skip("REC-001 artifact_inventory.json not present in this checkout")
    value = rec._read_rec001_recorded_hash("core", 10)
    data = json.loads(inv_path.read_text(encoding="utf-8"))
    expected = next(row["sha256"] for row in data["core_checkpoints"] if row["seed"] == 10)
    assert value == expected


# ---------------------------------------------------------------------------
# _snapshot_forbidden_cache_hashes
# ---------------------------------------------------------------------------


def test_snapshot_forbidden_cache_hashes_none_for_nonexistent_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    snap = rec._snapshot_forbidden_cache_hashes(seed=999999)
    assert set(snap.keys()) == {
        "primitive_bank_16",
        "shared_encoder",
        "learned_routing_bank_b008",
        "composition_library_bank",
        "shift_committed_bank_v1",
    }
    assert all(v is None for v in snap.values())


def test_snapshot_forbidden_cache_hashes_is_stable_and_content_addressed(tmp_path: Path) -> None:
    fake_root = tmp_path / "runs" / "phase_a2_bank_scaling_benchmark" / "seed_10"
    fake_root.mkdir(parents=True)
    fake_file = fake_root / "primitive_bank_16.pt"
    torch.save({"a": torch.zeros(2)}, fake_file)

    import os

    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        snap_a = rec._snapshot_forbidden_cache_hashes(seed=10)
        snap_b = rec._snapshot_forbidden_cache_hashes(seed=10)
    finally:
        os.chdir(old_cwd)
    assert snap_a["primitive_bank_16"] == snap_b["primitive_bank_16"]
    assert snap_a["primitive_bank_16"] == mb.raw_file_sha256(fake_file)
    assert snap_a["shared_encoder"] is None


# ---------------------------------------------------------------------------
# _manifest_to_json_dict round trip (the exact serialization the
# fresh-process subprocess deserializes -- see scripts/rec004_fresh_process_
# check.py's manual ModelBundleManifest(**...) reconstruction).
# ---------------------------------------------------------------------------


def _tiny_manifest(tmp_path: Path) -> mb.ModelBundleManifest:
    core_path = tmp_path / "core.pt"
    vocab_path = tmp_path / "vocab.pt"
    bank_path = tmp_path / "bank.pt"
    router_path = tmp_path / "router.pt"
    scorer_path = tmp_path / "scorer.pt"

    torch.save({"embed.weight": torch.randn(4, 3)}, core_path)
    torch.save({"marker": torch.zeros(1)}, vocab_path)
    torch.save(
        {"_primitives.0.weight": torch.randn(4, 4), "_primitives.0.bias": torch.randn(4)}, bank_path
    )
    torch.save(
        {
            "query_proj.weight": torch.randn(4, 4),
            "query_proj.bias": torch.randn(4),
            "_keys.0": torch.randn(4),
        },
        router_path,
    )
    torch.save({"heads.SELECT.weight": torch.randn(8, 4)}, scorer_path)

    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    schema_hash = "vocab10_ops8_argspan10_v1"
    core_component = mb.ComponentManifest("core", str(core_path), core_raw, core_state, schema_hash)
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    vocab_component = mb.ComponentManifest(
        "vocabulary", str(vocab_path), vocab_raw, vocab_state, schema_hash
    )

    bank_sd = mb.load_state_dict(bank_path)
    primitive = mb.PrimitiveManifestEntry(
        physical_id=0,
        operation_name="SELECT",
        version="v1",
        architecture_signature="cross_position_v1",
        state_abi_hash=mb.canonical_state_hash(mb.primitive_state_dict(bank_sd, 0)),
        core_dependency_hash=core_state,
        decoder_dependency_hash="NOT_APPLICABLE_NO_SEPARATE_DECODER_COMPONENT",
        weights_hash=mb.canonical_state_hash(mb.primitive_state_dict(bank_sd, 0)),
        source_artifact=str(bank_path),
        provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD,
        argument_schema_hash="arg-schema-v1",
        training_receipt="test-receipt",
    )
    router_sd = mb.load_state_dict(router_path)
    router = mb.RouterManifest(
        source_artifact=str(router_path),
        weights_hash=mb.canonical_state_hash(mb.router_non_key_state_dict(router_sd)),
        task_state_dependency_hash=core_state,
        key_to_primitive_mapping_hash=mb.router_key_to_primitive_mapping_hash(router_sd, [0]),
    )
    scorer_sd = mb.load_state_dict(scorer_path)
    scorer = mb.ArgumentScorerManifest(
        source_artifact=str(scorer_path),
        weights_hash=mb.canonical_state_hash(scorer_sd),
        input_dependency_hash=core_state,
        argument_schema_hash="arg-schema-v1",
    )
    policy = mb.ScoringPolicyManifest(
        family_formula_version="dot_product_v1",
        argument_formula_version="select_sigmoid_v2_adr0088",
        lambda_weight=2.0,
        application_policy_version="rec004_recovery_v1",
        calibrated_pair_id="pair-1",
    )
    return mb.build_manifest(
        schema_version=1,
        source_commit="test",
        runtime_recipe_version="rec004-v1",
        environment_record={"python_version": "3.12"},
        model_id="seed10",
        model_seed=10,
        training_run_id="run-1",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=(primitive,),
        router=router,
        argument_scorer=scorer,
        scoring_policy=policy,
    )


def test_manifest_to_json_dict_round_trips_through_load_bundle(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    as_json = rec._manifest_to_json_dict(manifest)
    # Round-trip exactly like scripts/rec004_fresh_process_check.py does.
    reconstructed = mb.ModelBundleManifest(
        schema_version=as_json["schema_version"],
        bundle_id=as_json["bundle_id"],
        content_manifest_digest=as_json["content_manifest_digest"],
        source_commit=as_json["source_commit"],
        runtime_recipe_version=as_json["runtime_recipe_version"],
        environment_record=as_json["environment_record"],
        model_id=as_json["model_id"],
        model_seed=as_json["model_seed"],
        training_run_id=as_json["training_run_id"],
        parent_bundle_ids=tuple(as_json["parent_bundle_ids"]),
        build_route=mb.BuildRoute(as_json["build_route"]),
        scope=mb.BundleScope(as_json["scope"]),
        requested_capabilities=frozenset(as_json["requested_capabilities"]),
        publish_status=mb.PublishStatus(as_json["publish_status"]),
        core=mb.ComponentManifest(**as_json["core"]),
        vocabulary=mb.ComponentManifest(**as_json["vocabulary"]),
        primitives=tuple(
            mb.PrimitiveManifestEntry(
                **{**p, "provenance_status": mb.ProvenanceStatus(p["provenance_status"])}
            )
            for p in as_json["primitives"]
        ),
        router=mb.RouterManifest(**as_json["router"]),
        argument_scorer=mb.ArgumentScorerManifest(**as_json["argument_scorer"]),
        scoring_policy=mb.ScoringPolicyManifest(**as_json["scoring_policy"]),
        build_recipe_hash=as_json["build_recipe_hash"],
        known_defects=tuple(as_json["known_defects"]),
    )
    assert reconstructed.bundle_id == manifest.bundle_id
    assert reconstructed.content_manifest_digest == manifest.content_manifest_digest
    loaded = mb.load_bundle(reconstructed, mode="nominal", expected_primitive_count=1)
    assert loaded.mode == "nominal"
    assert 0 in loaded.primitive_state_dicts


def test_manifest_to_json_dict_is_actually_json_serializable(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    as_json = rec._manifest_to_json_dict(manifest)
    # Must round-trip through a real json.dumps/loads, not just be a dict of
    # Python objects that happen to look serializable.
    reparsed = json.loads(json.dumps(as_json))
    assert reparsed["bundle_id"] == manifest.bundle_id


# ---------------------------------------------------------------------------
# _vocab_state_dict
# ---------------------------------------------------------------------------


class _FakeTokens:
    env_vocab_size = 10
    op_base = 2
    arg_base = 18
    arg_span = 10
    num_operations = 16


class _FakeCore:
    tokens = _FakeTokens()


def test_vocab_state_dict_has_expected_keys_and_values() -> None:
    sd = rec._vocab_state_dict(_FakeCore())
    assert set(sd.keys()) == {"vocab_size", "op_base", "arg_base", "arg_span", "num_operations"}
    assert int(sd["vocab_size"].item()) == 10
    assert int(sd["num_operations"].item()) == 16


# ---------------------------------------------------------------------------
# fresh-process script: static "no builder call" check runs without
# executing the script or requiring a GPU.
# ---------------------------------------------------------------------------


def test_fresh_process_script_source_contains_no_builder_identifier() -> None:
    script_path = Path("scripts/rec004_fresh_process_check.py")
    assert script_path.is_file()
    source = script_path.read_text(encoding="utf-8")
    forbidden = (
        "get_or_build",
        "_train_single_primitive",
        "update_router_incrementally",
        "_ensure_learned_routing_bank_and_core",
        "torch.optim",
        ".backward(",
    )
    found = [name for name in forbidden if name in source]
    assert found == []


def test_fresh_process_script_never_hardcodes_a_shared_cache_path() -> None:
    script_path = Path("scripts/rec004_fresh_process_check.py")
    source = script_path.read_text(encoding="utf-8")
    forbidden_prefixes = (
        "phase_a1_learned_routing_benchmark",
        "phase_a2_bank_scaling_benchmark",
        "phase_a1_shift_compact_structural_probe",
        "r3_009_shift_functional_generalization_repair",
    )
    found = [p for p in forbidden_prefixes if p in source]
    assert found == []


# ---------------------------------------------------------------------------
# Dispatcher config loader
# ---------------------------------------------------------------------------


def test_dispatcher_rec004_config_loader_reads_yaml(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, "scripts")
    import importlib

    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004.yaml"
    config_path.write_text(
        "seed: 10\noutput_dir: runs/x/rec004/run_001\n"
        "direct_query_examples_per_operation: 512\nnon_shift_floor: 0.9\n",
        encoding="utf-8",
    )
    config = dispatcher._load_rec004_config(config_path)
    assert config.seed == 10
    assert config.direct_query_examples_per_operation == 512
    assert config.non_shift_floor == pytest.approx(0.9)


def test_dispatcher_rec004_config_loader_refuses_non_pilot_seed(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, "scripts")
    import importlib

    dispatcher = importlib.import_module("run_phase_b_b2_model_bundle_recovery")

    config_path = tmp_path / "rec004_bad.yaml"
    config_path.write_text("seed: 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        dispatcher._load_rec004_config(config_path)
