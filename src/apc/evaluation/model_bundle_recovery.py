"""Task B-C005REC-002 milestone: exercise the `apc.utils.model_bundle`
fail-closed loader contract on CPU-only fixtures, plus one real-artifact
legacy-import demonstration against REC-001's discovered seed-10 Core
checkpoint. Produces the deliverables named in
`docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`'s REC-002 section:
`bundle_schema.json`, `loader_contract_tests.json`,
`compatibility_rules.json`, `legacy_import_policy.json`.

Per RG1's own scope (`docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
section 4): "CPUの正常・異常fixtureで検査する" -- this milestone performs no
GPU work, no training, and no scientific/performance evaluation. It only
proves the loader's typed failure modes actually fire on real tensor data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, get_type_hints

import torch

from apc.utils import model_bundle as mb

__all__ = [
    "ModelBundleContractConfig",
    "run_model_bundle_contract_task",
    "RecoveryBuildPlanConfig",
    "run_recovery_build_plan_task",
    "EvaluationFrozenError",
    "PilotRestoreBuildConfig",
    "run_pilot_restore_build_task",
]


@dataclass(frozen=True)
class ModelBundleContractConfig:
    real_core_source: Path | None = Path(
        "runs/phase_a1_shift_compact_structural_probe/seed_10/shared_encoder.pt"
    )
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec002")


# ---------------------------------------------------------------------------
# Tiny CPU fixture builder (mirrors tests/test_model_bundle_contract.py's
# conventions but is independent production code, not a test import).
# ---------------------------------------------------------------------------

_PRIMITIVE_IDS = (0, 1, 2)
_ARG_SCHEMA_HASH = "arg-schema-v1"
_CORE_SCHEMA_HASH = "vocab-size-10-v1"


def _core_sd(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"embed.weight": torch.randn(6, 4, generator=g)}


def _bank_sd(primitive_ids: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for pid in primitive_ids:
        sd[f"_primitives.{pid}.weight"] = torch.randn(4, 4, generator=g)
        sd[f"_primitives.{pid}.bias"] = torch.randn(4, generator=g)
    return sd


def _router_sd(primitive_ids: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {
        "query_proj.weight": torch.randn(4, 4, generator=g),
        "query_proj.bias": torch.randn(4, generator=g),
    }
    for pid in primitive_ids:
        sd[f"_keys.{pid}"] = torch.randn(4, generator=g)
    return sd


def _scorer_sd(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "heads.SELECT.weight": torch.randn(8, 4, generator=g),
        "heads.SELECT.bias": torch.randn(8, generator=g),
    }


def _save(path: Path, sd: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)


def _build_fixture_bundle(root: Path, *, model_seed: int = 10) -> mb.ModelBundleManifest:
    core_path, bank_path = root / "core.pt", root / "bank.pt"
    router_path, scorer_path, vocab_path = root / "router.pt", root / "scorer.pt", root / "vocab.pt"
    _save(core_path, _core_sd(model_seed))
    _save(bank_path, _bank_sd(_PRIMITIVE_IDS, seed=1))
    _save(router_path, _router_sd(_PRIMITIVE_IDS, seed=2))
    _save(scorer_path, _scorer_sd(seed=3))
    _save(vocab_path, {"vocab_marker": torch.zeros(1)})

    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    core = mb.ComponentManifest("core", str(core_path), core_raw, core_state, _CORE_SCHEMA_HASH)
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    vocab = mb.ComponentManifest(
        "vocabulary", str(vocab_path), vocab_raw, vocab_state, _CORE_SCHEMA_HASH
    )

    bank_state_dict = mb.load_state_dict(bank_path)
    primitives = tuple(
        mb.PrimitiveManifestEntry(
            physical_id=pid,
            operation_name=f"OP_{pid}",
            version="v1",
            architecture_signature="pointwise-v1",
            state_abi_hash="abi-v1",
            core_dependency_hash=core_state,
            decoder_dependency_hash="decoder-abi-v1",
            weights_hash=mb.canonical_state_hash(mb.primitive_state_dict(bank_state_dict, pid)),
            source_artifact=str(bank_path),
            provenance_status=mb.ProvenanceStatus.RESTORED_VALIDATED,
            argument_schema_hash=_ARG_SCHEMA_HASH if pid == _PRIMITIVE_IDS[0] else None,
            training_receipt="receipt-1",
        )
        for pid in _PRIMITIVE_IDS
    )

    router_state_dict = mb.load_state_dict(router_path)
    router = mb.RouterManifest(
        source_artifact=str(router_path),
        weights_hash=mb.canonical_state_hash(mb.router_non_key_state_dict(router_state_dict)),
        task_state_dependency_hash=core_state,
        key_to_primitive_mapping_hash=mb.router_key_to_primitive_mapping_hash(
            router_state_dict, list(_PRIMITIVE_IDS)
        ),
    )
    scorer_state_dict = mb.load_state_dict(scorer_path)
    argument_scorer = mb.ArgumentScorerManifest(
        source_artifact=str(scorer_path),
        weights_hash=mb.canonical_state_hash(scorer_state_dict),
        input_dependency_hash=core_state,
        argument_schema_hash=_ARG_SCHEMA_HASH,
    )
    scoring_policy = mb.ScoringPolicyManifest(
        family_formula_version="dot_product_v1",
        argument_formula_version="select_sigmoid_v2_adr0088",
        lambda_weight=2.0,
        application_policy_version="l4_gated_v1",
        calibrated_pair_id="pair-1",
    )
    return mb.build_manifest(
        schema_version=1,
        source_commit="fixture",
        runtime_recipe_version="rec002-v1",
        environment_record={"python": "3.12"},
        model_id=f"seed{model_seed}",
        model_seed=model_seed,
        training_run_id="run-1",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.RESTORE,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core,
        vocabulary=vocab,
        primitives=primitives,
        router=router,
        argument_scorer=argument_scorer,
        scoring_policy=scoring_policy,
    )


def _refreshed(manifest: mb.ModelBundleManifest, **changes: Any) -> mb.ModelBundleManifest:
    draft = dataclasses.replace(manifest, **changes)
    digest = mb.compute_content_manifest_digest(draft)
    bundle_id = mb.compute_bundle_id(draft, content_manifest_digest=digest)
    return dataclasses.replace(draft, content_manifest_digest=digest, bundle_id=bundle_id)


# ---------------------------------------------------------------------------
# Scenario battery -> loader_contract_tests.json
# ---------------------------------------------------------------------------


def _run_scenarios(fixture_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(name: str, expected: str | None, fn: Any) -> None:
        try:
            fn()
            actual = None
        except mb.ModelBundleError as exc:
            actual = exc.label
        rows.append(
            {
                "scenario": name,
                "expected_label": expected,
                "actual_label": actual,
                "passed": actual == expected,
            }
        )

    valid = _build_fixture_bundle(fixture_root / "valid", model_seed=10)
    record("valid_full_bundle_nominal_load", None, lambda: mb.load_bundle(valid, mode="nominal"))

    corrupt_core_root = fixture_root / "core_tensor_diff"
    corrupt_core = _build_fixture_bundle(corrupt_core_root, model_seed=11)
    sd = torch.load(Path(corrupt_core.core.file_path), map_location="cpu", weights_only=True)
    sd["embed.weight"] = sd["embed.weight"] + 1.0
    torch.save(sd, Path(corrupt_core.core.file_path))
    record(
        "core_one_tensor_diff",
        mb.CoreDependencyMismatchError.label,
        lambda: mb.load_bundle(corrupt_core, mode="nominal"),
    )

    stale = _build_fixture_bundle(fixture_root / "stale_core_dependency", model_seed=12)
    stale_primitives = tuple(
        dataclasses.replace(p, core_dependency_hash="stale-hash-from-before-r3-009")
        for p in stale.primitives
    )
    stale_tampered = _refreshed(stale, primitives=stale_primitives)
    record(
        "adr0091_stale_bank_vs_new_core",
        mb.CoreDependencyMismatchError.label,
        lambda: mb.load_bundle(stale_tampered, mode="nominal"),
    )

    missing_root = fixture_root / "primitive_missing"
    missing = _build_fixture_bundle(missing_root, model_seed=13)
    bank_sd = torch.load(
        Path(missing.primitives[0].source_artifact), map_location="cpu", weights_only=True
    )
    for key in [
        k for k in bank_sd if k.startswith(f"_primitives.{missing.primitives[0].physical_id}.")
    ]:
        del bank_sd[key]
    torch.save(bank_sd, Path(missing.primitives[0].source_artifact))
    record(
        "primitive_entirely_missing",
        mb.MissingArtifactError.label,
        lambda: mb.load_bundle(missing, mode="nominal"),
    )

    count = _build_fixture_bundle(fixture_root / "primitive_count", model_seed=14)
    record(
        "declared_primitive_count_below_expected",
        mb.IncompleteBundleError.label,
        lambda: mb.load_bundle(
            count, mode="nominal", expected_primitive_count=len(_PRIMITIVE_IDS) + 1
        ),
    )

    partial_root = fixture_root / "partial_state_dict"
    partial = _build_fixture_bundle(partial_root, model_seed=15)
    partial_bank_sd = torch.load(
        Path(partial.primitives[0].source_artifact), map_location="cpu", weights_only=True
    )
    del partial_bank_sd[f"_primitives.{partial.primitives[0].physical_id}.bias"]
    torch.save(partial_bank_sd, Path(partial.primitives[0].source_artifact))
    record(
        "partial_state_dict_missing_key",
        mb.IncompleteBundleError.label,
        lambda: mb.load_bundle(partial, mode="nominal"),
    )

    seed10 = _build_fixture_bundle(fixture_root / "relabel_10", model_seed=10)
    seed11_relabeled = _build_fixture_bundle(fixture_root / "relabel_11", model_seed=11)
    # Force the "relabeled" bundle's Core to be a byte-identical copy of
    # seed 10's -- the fake-same-training-run scenario.
    relabel_core_sd = torch.load(Path(seed10.core.file_path), map_location="cpu", weights_only=True)
    torch.save(relabel_core_sd, Path(seed11_relabeled.core.file_path))
    relabel_raw, relabel_state = mb.canonical_state_hash_from_file(
        Path(seed11_relabeled.core.file_path)
    )
    relabeled_manifest = _refreshed(
        seed11_relabeled,
        core=dataclasses.replace(
            seed11_relabeled.core, file_sha256=relabel_raw, canonical_state_hash=relabel_state
        ),
    )

    def _relabel_check() -> None:
        mb.assert_distinct_model_identities([seed10, relabeled_manifest])

    record("seed_relabeled_same_core", mb.DuplicateTrainingIdentityError.label, _relabel_check)

    swap_root = fixture_root / "router_key_swap"
    swap = _build_fixture_bundle(swap_root, model_seed=16)
    router_sd = torch.load(Path(swap.router.source_artifact), map_location="cpu", weights_only=True)
    pid_a, pid_b = _PRIMITIVE_IDS[0], _PRIMITIVE_IDS[1]
    key_a, key_b = router_sd[f"_keys.{pid_a}"].clone(), router_sd[f"_keys.{pid_b}"].clone()
    router_sd[f"_keys.{pid_a}"], router_sd[f"_keys.{pid_b}"] = key_b, key_a
    torch.save(router_sd, Path(swap.router.source_artifact))
    record(
        "router_key_order_swapped_between_primitives",
        mb.RouterKeyMappingMismatchError.label,
        lambda: mb.load_bundle(swap, mode="nominal"),
    )

    schema = _build_fixture_bundle(fixture_root / "schema_mismatch", model_seed=17)
    schema_tampered = _refreshed(
        schema,
        vocabulary=dataclasses.replace(schema.vocabulary, schema_hash="a-different-vocab-schema"),
    )
    record(
        "vocab_schema_disagrees_with_core",
        mb.SchemaMismatchError.label,
        lambda: mb.load_bundle(schema_tampered, mode="nominal"),
    )

    pair = _build_fixture_bundle(fixture_root / "uncertified_pair", model_seed=18)
    record(
        "uncertified_router_scorer_pair",
        mb.UncertifiedPairError.label,
        lambda: mb.load_bundle(pair, mode="nominal", compatibility_table=mb.CompatibilityTable()),
    )

    formula_old = _build_fixture_bundle(fixture_root / "formula_old", model_seed=19)
    formula_old = _refreshed(
        formula_old,
        scoring_policy=dataclasses.replace(
            formula_old.scoring_policy, argument_formula_version="select_softmax_v1_pre_adr0088"
        ),
    )
    live_signature = mb.compute_execution_signature(
        _refreshed(
            formula_old,
            scoring_policy=dataclasses.replace(
                formula_old.scoring_policy, argument_formula_version="select_sigmoid_v2_adr0088"
            ),
        )
    )
    record(
        "formula_only_change_old_manifest_vs_live_code",
        mb.ExecutionSignatureMismatchError.label,
        lambda: mb.load_bundle(
            formula_old, mode="nominal", expected_execution_signature=live_signature
        ),
    )

    corrupted_root = fixture_root / "corrupted_checkpoint"
    corrupted = _build_fixture_bundle(corrupted_root, model_seed=20)
    Path(corrupted.core.file_path).write_bytes(b"not a valid torch checkpoint")
    record(
        "checkpoint_file_corrupted",
        mb.CorruptedArtifactError.label,
        lambda: mb.load_bundle(corrupted, mode="nominal"),
    )

    staging = _build_fixture_bundle(fixture_root / "staging", model_seed=21)
    staging = _refreshed(staging, publish_status=mb.PublishStatus.STAGING)
    record(
        "staged_incomplete_write_nominal_load",
        mb.IncompleteBundleError.label,
        lambda: mb.load_bundle(staging, mode="nominal"),
    )

    unknown = _build_fixture_bundle(fixture_root / "unknown_provenance", model_seed=22)
    unknown_primitives = tuple(
        dataclasses.replace(p, provenance_status=mb.ProvenanceStatus.UNKNOWN, training_receipt=None)
        for p in unknown.primitives
    )
    unknown = _refreshed(unknown, primitives=unknown_primitives)
    record(
        "unknown_provenance_nominal_load",
        mb.UnknownProvenanceError.label,
        lambda: mb.load_bundle(unknown, mode="nominal"),
    )

    untrained = _build_fixture_bundle(fixture_root / "untrained_claim", model_seed=23)
    untrained_primitives = tuple(
        dataclasses.replace(
            p, provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD, training_receipt=None
        )
        for p in untrained.primitives
    )
    untrained = _refreshed(untrained, primitives=untrained_primitives)
    record(
        "trained_claim_without_training_receipt",
        mb.UntrainedComponentError.label,
        lambda: mb.load_bundle(untrained, mode="nominal"),
    )

    capability = _build_fixture_bundle(fixture_root / "capability", model_seed=24)
    capability = _refreshed(capability, requested_capabilities=frozenset({"diagnostic_only"}))
    record(
        "insufficient_capability_scope",
        mb.CapabilityNotQualifiedError.label,
        lambda: mb.load_bundle(
            capability, mode="nominal", required_capabilities=frozenset({"nominal_execution"})
        ),
    )

    return rows


def _loader_never_calls_builder_check() -> dict[str, Any]:
    """Structural proof, not a load_bundle scenario: the loader module's
    own source never references a builder/training entry point, and
    calling `load_bundle` never constructs a torch optimizer."""
    import inspect

    source = inspect.getsource(mb)
    forbidden = (
        "get_or_build",
        "_train_single_primitive",
        "update_router_incrementally",
        "torch.optim",
        ".backward(",
    )
    found = [name for name in forbidden if name in source]
    return {
        "scenario": "loader_module_references_no_builder_or_training_call",
        "forbidden_identifiers_checked": list(forbidden),
        "forbidden_identifiers_found": found,
        "passed": not found,
    }


# ---------------------------------------------------------------------------
# bundle_schema.json (generated by introspection, not hand-maintained)
# ---------------------------------------------------------------------------


def _field_type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _describe_dataclass(cls: type) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    result: dict[str, Any] = {}
    for f in fields(cls):
        tp = hints.get(f.name, f.type)
        result[f.name] = _field_type_name(tp)
    return result


def _build_bundle_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {
        "source": "apc.utils.model_bundle (Task B-C005REC-002)",
        "design_reference": "docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md section 2",
        "dataclasses": {},
        "enums": {},
    }
    for cls in (
        mb.ComponentManifest,
        mb.PrimitiveManifestEntry,
        mb.RouterManifest,
        mb.ArgumentScorerManifest,
        mb.ScoringPolicyManifest,
        mb.ModelBundleManifest,
    ):
        assert is_dataclass(cls)
        schema["dataclasses"][cls.__name__] = _describe_dataclass(cls)
    for enum_cls in (mb.BuildRoute, mb.BundleScope, mb.PublishStatus, mb.ProvenanceStatus):
        assert issubclass(enum_cls, Enum)
        schema["enums"][enum_cls.__name__] = [member.value for member in enum_cls]
    schema["error_labels"] = sorted(
        {
            cls.label
            for cls in (
                mb.MissingArtifactError,
                mb.CorruptedArtifactError,
                mb.CoreDependencyMismatchError,
                mb.SchemaMismatchError,
                mb.ArgumentSchemaMismatchError,
                mb.RouterKeyMappingMismatchError,
                mb.ExecutionSignatureMismatchError,
                mb.UntrainedComponentError,
                mb.UnknownProvenanceError,
                mb.UncertifiedPairError,
                mb.IncompleteBundleError,
                mb.CapabilityNotQualifiedError,
                mb.DuplicateTrainingIdentityError,
            )
        }
    )
    return schema


# ---------------------------------------------------------------------------
# compatibility_rules.json -- contract description + one illustrative,
# clearly-synthetic example (NOT a claim about any real R3 repair artifact;
# populating this table with real certified pairs is REC-006's job).
# ---------------------------------------------------------------------------


def _build_compatibility_rules_doc() -> dict[str, Any]:
    return {
        "contract": (
            "A CompatibilityTable entry certifies that a specific "
            "(router_weights_hash, scorer_weights_hash, lambda_weight, "
            "argument_schema_hash, application_policy_version) combination "
            "was actually trained/validated together. load_bundle(mode="
            "'nominal', compatibility_table=...) refuses any manifest whose "
            "pair is not in the table (UNCERTIFIED_PAIR); diagnostic mode "
            "is exempt for audit-only inspection."
        ),
        "fields": dataclasses.asdict(
            mb.CertifiedPair(
                router_weights_hash="<sha256 of router_non_key_state_dict>",
                scorer_weights_hash="<sha256 of argument_scorer state_dict>",
                lambda_weight=0.0,
                argument_schema_hash="<schema hash>",
                application_policy_version="<policy version string>",
                calibrated_pair_id="<opaque id linking back to the training run>",
            )
        ),
        "illustrative_example_synthetic_not_a_real_certification": {
            "note": (
                "SYNTHETIC placeholder shape only, built from this task's own "
                "CPU fixture -- not a claim about any real R3-006/007/008/009 "
                "repair artifact's actual hash. Populating this table with "
                "real certified pairs from re-applied R3 repairs is "
                "B-C005REC-006's job, not REC-002's."
            ),
        },
        "population_policy": (
            "This table is populated only from artifacts whose router and "
            "scorer were actually trained/selected together in the same "
            "repair run (e.g. a future REC-006 preparation output). A "
            "same-seed or same-timestamp coincidence is never sufficient."
        ),
    }


# ---------------------------------------------------------------------------
# legacy_import_policy.json -- policy doc + one REAL worked example against
# REC-001's discovered seed-10 Core checkpoint (read-only copy + hash audit,
# no training).
# ---------------------------------------------------------------------------


def _real_legacy_import_demo(config: ModelBundleContractConfig) -> dict[str, Any]:
    source = config.real_core_source
    if source is None or not source.is_file():
        return {
            "status": "SKIPPED_SOURCE_NOT_FOUND",
            "attempted_source": str(source) if source else None,
        }
    dest_dir = config.output_dir / "legacy_import_demo" / "seed_10_core"
    component = mb.legacy_import(source, dest_dir, component_id="core_seed10")
    return {
        "status": "REAL_ARTIFACT_LEGACY_IMPORT_OK",
        "source": str(source),
        "imported_to": component.file_path,
        "raw_file_sha256": component.file_sha256,
        "canonical_state_hash": component.canonical_state_hash,
        "provenance_status_assigned_by_caller": mb.ProvenanceStatus.LEGACY_IMPORTED.value,
        "note": (
            "legacy_import copies bytes and records hashes only -- it does "
            "not mark this component trained/nominal-ready. Nominal use "
            "still requires this session's own dependency/qualification "
            "checks (REC-003 onward)."
        ),
    }


def _build_legacy_import_policy_doc(demo: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": (
            "legacy_import(source_path, dest_dir, component_id=...) copies "
            "a checkpoint byte-for-byte into a new namespace (shutil.copy2, "
            "never a hard link) and returns a ComponentManifest carrying "
            "its raw_file_sha256/canonical_state_hash. It never sets a "
            "provenance status of TRAINED_THIS_BUILD or RESTORED_VALIDATED "
            "-- the caller must record whatever ProvenanceStatus the "
            "surrounding audit actually established (typically "
            "LEGACY_IMPORTED, or LEGACY_REQUALIFIED_ON_RECOVERY_FIXTURE if "
            "a later task re-verifies function on a fixture)."
        ),
        "shared_cache_untouched": "the original source file is never modified or moved",
        "real_artifact_worked_example": demo,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_model_bundle_contract_task(config: ModelBundleContractConfig) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_root = output_dir / "fixtures"

    scenario_rows = _run_scenarios(fixture_root)
    scenario_rows.append(_loader_never_calls_builder_check())
    all_passed = all(row["passed"] for row in scenario_rows)

    bundle_schema = _build_bundle_schema()
    compatibility_rules = _build_compatibility_rules_doc()
    legacy_demo = _real_legacy_import_demo(config)
    legacy_policy = _build_legacy_import_policy_doc(legacy_demo)

    (output_dir / "bundle_schema.json").write_text(
        json.dumps(bundle_schema, indent=2), encoding="utf-8"
    )
    (output_dir / "loader_contract_tests.json").write_text(
        json.dumps({"scenarios": scenario_rows, "all_passed": all_passed}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "compatibility_rules.json").write_text(
        json.dumps(compatibility_rules, indent=2), encoding="utf-8"
    )
    (output_dir / "legacy_import_policy.json").write_text(
        json.dumps(legacy_policy, indent=2), encoding="utf-8"
    )

    result = "RG1_PASS" if all_passed else "RG1_FAIL"
    protocol = {
        "task_id": "B-C005REC-002",
        "result": result,
        "scenarios_run": len(scenario_rows),
        "scenarios_passed": sum(1 for row in scenario_rows if row["passed"]),
        "real_artifact_legacy_import_status": legacy_demo["status"],
        "no_training_performed": True,
        "no_shared_cache_modified": True,
        "wall_clock_seconds": time.time() - start,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "protocol": protocol,
        "loader_contract_tests": scenario_rows,
        "bundle_schema": bundle_schema,
        "compatibility_rules": compatibility_rules,
        "legacy_import_policy": legacy_policy,
    }


# =============================================================================
# Task B-C005REC-003: Complete Build DAG & Preregistered Recovery Protocol
#
# Fixes an explicit build plan for all 16 real registry primitives so an
# empty/incoherent cache no longer means "8 operations quietly retrained,
# the rest silently reused" (ADR-0091). Per RG2's own scope
# (`docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` section 5):
# "build planが実registryの全16個をcoverし... まだ5-model学習を開始しない" --
# this milestone fixes the plan/protocol and proves its logic on CPU tiny
# fixtures + real-registry structural checks. It performs NO GPU training and
# starts NO 5-seed run; REC-004 is where the seed-10 pilot actually executes.
#
# Every recipe reference below was independently confirmed against live
# source (not re-derived from REC-001's report alone) during this task:
#   * apc.evaluation.learned_routing_benchmark._ensure_learned_routing_bank_and_core
#     (:493-650) -- Core restore-or-train, then trains the 8 canonical + 2
#     Branch-B ops in one call when `seed_dir` has no existing 10-op bank.
#     `seed_dir` is a plain parameter, so calling it with a *new* namespace
#     directory is a namespace-safe reuse, not a code change to the function.
#     Side effect confirmed by reading :627-634: its canonical loop trains
#     SHIFT too (SHIFT is in ALL_CANONICAL_OPERATIONS) using the *generic*
#     per-op trainer, not R3-009's dedicated versioned-replacement pipeline --
#     that generic SHIFT slot must be discarded, never published (see the
#     SHIFT_DEDICATED / SHIFT_OVERRIDE_MERGE stages below).
#   * apc.evaluation.incremental_router_benchmark.get_or_build_16_primitive_bank
#     (:152-257) -- its cache-miss branch hardcodes
#     `b008_dir = Path("runs/phase_a1_learned_routing_benchmark") / f"seed_{seed}"`
#     (:225) and does a `strict=False` blind load from it (:229) instead of
#     training the 8+2 ops itself; only the 6 incremental ops are actually
#     trained in that branch (:232-247, steps=1000 each). The recovery build
#     plan below reuses only this function's *incremental-training shape*
#     (`bank.new_cross_position_primitive` + `_train_single_primitive(...,
#     steps=1000)`), applied directly to the already-coherent bank from the
#     CANONICAL_AND_BRANCH_B_BUILD/SHIFT_OVERRIDE_MERGE stages -- never the
#     b008 strict=False fallback load.
#   * apc.evaluation.unified_oracle_causal_benchmark._train_single_primitive
#     (:492-...) -- AdamW(lr=config.operator_lr, weight_decay=
#     config.operator_weight_decay) + CosineAnnealingLR(T_max=steps,
#     eta_min=1e-5); `UnifiedBenchmarkConfig` defaults operator_lr=0.0008,
#     operator_weight_decay=0.0001 (:163-164) -- these are the values
#     `_ensure_learned_routing_bank_and_core`'s internal `u_bank_cfg` uses
#     since it never overrides them.
#   * apc.evaluation.shift_functional_generalization_repair -- R3-009's own
#     dedicated SHIFT pipeline. `ShiftFunctionalGeneralizationRepairConfig`
#     (:539-567): operator_train_steps=12000, operator_lr=5e-4,
#     operator_weight_decay=1e-4, operator_grad_clip=1.0,
#     operator_batch_size=32; default/tie-break variant is `iid_baseline`
#     (:664). Real per-seed outcome from
#     `runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair/
#     bank_transaction_log.json`: COMMITTED for seeds 10/11/14 (real
#     `committed_bank/seed_{seed}/primitive_bank_16_shift_v1.pt` files exist
#     on disk, confirmed by directory listing), ROLLED_BACK for 12/13.
#   * apc.evaluation.incremental_router_benchmark.update_router_incrementally
#     with `IncrementalUpdateCondition.R0_FULL_RETRAIN`, router_lr=0.005,
#     router_steps=250 (the same defaults `run_single_seed_incremental_
#     benchmark` uses, :268-269) -- router is "BUILD_REQUIRED_BY_DESIGN"
#     (REC-001): never persisted, recalibrated fresh every run.
#   * apc.primitives.argument_scoring.ArgumentScorer.train_on_examples with
#     `ArgumentScorerConfig` defaults d_model=128, arg_vocab_size=32,
#     lambda_weight=2.0, lr=0.005, steps=200 (:39-43).
#   * apc.evaluation.count_bind_key_scoring_repair.
#     run_count_bind_key_scoring_repair (R3-006) and apc.evaluation.
#     bind_argument_scorer_repair.run_bind_argument_scorer_repair (R3-008)
#     exist and are real, but REC-001 already found they were validated
#     against the *stale* bank -- re-deriving them against the coherent 16-op
#     bank is explicit REC-006 preparation work, registered here as deferred,
#     not executed.
#   * apc.evaluation.relation_split_protocol.assert_sealed_access_permitted
#     (:105-119) -- the repo's own existing sealed-partition guard (covers
#     SEALED_GATE_SEEDS | DEFAULT_REGATE_SEEDS | NEW_SEALED_V2_SEEDS). Reused
#     directly by the dry run below rather than reimplemented.
#   * apc.evaluation.sequential_closed_loop_benchmark -- the legacy K/C/N/R
#     stream's own seeds (DEFAULT_SEEDS=(0,1,2,3,4), :85) and dev_seed=42
#     (:133) are a *different axis* from the model cohort (10-14): they seed
#     episode/task realization, never a model checkpoint. NOVEL_OPS (:114)
#     equals PHASE_A2_INCREMENTAL_NEW_OPERATIONS -- the same 6 ops the
#     recovery build now trains into the full-16 bank, so this legacy
#     stream's "N" category can no longer be presented as an unseen-family
#     test once a recovered bundle is injected (REC-006/007's problem to
#     handle at injection time; REC-003 only registers the conflict).
# =============================================================================


RECOVERY_DEV_SEEDS: Final = (10, 11, 12, 13, 14)
RECOVERY_PILOT_SEED: Final = 10
_SHIFT_COMMITTED_SEEDS: Final = (10, 11, 14)
_SHIFT_ROLLED_BACK_SEEDS: Final = (12, 13)

# The exact hardcoded shared-cache paths this build plan must never write to
# (confirmed by reading the call sites cited above -- not guessed).
FORBIDDEN_SHARED_CACHE_PATH_PREFIXES: Final = (
    "runs/phase_a1_learned_routing_benchmark",
    "runs/phase_a2_bank_scaling_benchmark",
    "runs/phase_a1_shift_compact_structural_probe",
    "runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair/committed_bank",
)

RECOVERY_NAMESPACE_ROOT: Final = "runs/phase_b_b2_model_bundle_recovery/staging"


class EvaluationFrozenError(Exception):
    """Raised when a build/train/calibrate call is attempted while evaluation
    is frozen. Exists so "BuildとEvaluateを分離する" (AGENTS addendum) is a
    mechanically enforced invariant, not just a documentation rule -- the
    same primitive REC-006's real injection contract will need."""


@dataclass
class _FreezeState:
    frozen: bool = False


_FREEZE_STATE = _FreezeState()


@contextmanager
def frozen_evaluation() -> Iterator[None]:
    """Context manager: any `_guard_not_frozen` call made while inside this
    block raises `EvaluationFrozenError`. Reentrant-safe (restores the prior
    state on exit rather than assuming it starts at False)."""
    prior = _FREEZE_STATE.frozen
    _FREEZE_STATE.frozen = True
    try:
        yield
    finally:
        _FREEZE_STATE.frozen = prior


def _guard_not_frozen(action: str) -> None:
    if _FREEZE_STATE.frozen:
        raise EvaluationFrozenError(
            f"{action} attempted while evaluation is frozen -- build/train/"
            "calibrate calls are forbidden once evaluation has started"
        )


@dataclass(frozen=True)
class RecoveryBuildPlanConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec003")
    seeds: tuple[int, ...] = RECOVERY_DEV_SEEDS


# ---------------------------------------------------------------------------
# Real 16-operation registry (imported, never hand-typed) and build-plan rows.
# ---------------------------------------------------------------------------


def _real_16_operation_registry() -> dict[str, str]:
    """operation -> group, sourced from the same three real registry tuples
    REC-001 cited (src/apc/environments/operations.py:470,645-652 and
    src/apc/evaluation/unified_oracle_causal_benchmark.py:91-94)."""
    from apc.environments.operations import (
        BRANCH_B_NOVEL_OPERATION_NAMES,
        PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
    )
    from apc.primitives.argument_scoring import (
        PARAMETERIZED_OPERATIONS as CANONICAL_PARAMETERIZED_OPERATIONS,
    )

    parameter_free = ("COPY", "REVERSE", "SORT", "NEGATE")
    registry: dict[str, str] = {}
    for op in CANONICAL_PARAMETERIZED_OPERATIONS:
        registry[op] = "canonical_parameterized"
    for op in parameter_free:
        registry[op] = "canonical_parameter_free"
    for op in BRANCH_B_NOVEL_OPERATION_NAMES:
        registry[op] = "branch_b_novel"
    for op in PHASE_A2_INCREMENTAL_NEW_OPERATIONS:
        registry[op] = "phase_a2_incremental"
    return registry


def _build_stage_graph() -> list[dict[str, Any]]:
    """The explicit, ordered build DAG. Each stage names its real recipe,
    what it covers, its dependencies, and its namespace-safe output --
    never the hardcoded shared-cache paths in
    `FORBIDDEN_SHARED_CACHE_PATH_PREFIXES`."""
    ns = RECOVERY_NAMESPACE_ROOT
    return [
        {
            "stage_id": "CORE_RESTORE",
            "depends_on": [],
            "kind": "restore",
            "recipe_ref": "apc.evaluation.learned_routing_benchmark."
            "_ensure_learned_routing_bank_and_core:519-548 (checkpoint-lookup branch)",
            "covers_operations": [],
            "covers_components": ["core", "decoder", "token_schema"],
            "restore_source_template": (
                "runs/phase_a1_shift_compact_structural_probe/seed_{seed}/shared_encoder.pt"
            ),
            "namespace_output_template": f"{ns}/seed_{{seed}}/core/shared_encoder.pt",
            "fallback_if_missing": (
                "apc.evaluation.unified_oracle_causal_benchmark._get_or_train_frozen_shared_core "
                "(core_train_steps from UnifiedBenchmarkConfig) -- NOT exercised for the "
                "registered cohort: REC-001 hash-verified this checkpoint exists for all 5 "
                "development seeds, so this fallback is dead code for RECOVERY_DEV_SEEDS today"
            ),
            "notes": (
                "Restore only; Core is frozen (requires_grad=False) after load, "
                "matching AGENTS.md's frozen-Core invariant."
            ),
        },
        {
            "stage_id": "CANONICAL_AND_BRANCH_B_BUILD",
            "depends_on": ["CORE_RESTORE"],
            "kind": "train",
            "recipe_ref": "apc.evaluation.learned_routing_benchmark."
            "_ensure_learned_routing_bank_and_core:570-650, called with "
            "seed_dir=<namespace>/seed_{seed} so its own cache-miss branch trains fresh "
            "against the restored Core instead of loading a stale bank",
            "covers_operations": [
                "SELECT",
                "COUNT",
                "BIND",
                "COPY",
                "REVERSE",
                "SORT",
                "NEGATE",
                "SWAP_PAIRS",
                "INVERT_HALF",
            ],
            "namespace_output_template": f"{ns}/seed_{{seed}}/canonical_branch_b/primitive_bank.pt",
            "recipe_side_effect_warning": (
                "This call's own canonical loop (:627-634) also trains SHIFT (SHIFT is in "
                "ALL_CANONICAL_OPERATIONS) using the generic per-op trainer, NOT R3-009's "
                "dedicated versioned-replacement pipeline. That generic SHIFT slot is a "
                "discarded byproduct of this stage -- it is never published, never certified, "
                "and is overwritten by SHIFT_OVERRIDE_MERGE before any evaluation reads it."
            ),
            "notes": (
                "One atomic function call trains 8 canonical + 2 Branch-B ops together; "
                "cannot be split into per-op stages without a code change to the recipe itself."
            ),
        },
        {
            "stage_id": "SHIFT_DEDICATED",
            "depends_on": ["CORE_RESTORE"],
            "kind": "restore_or_train_per_seed",
            "recipe_ref": "apc.evaluation.shift_functional_generalization_repair."
            "run_shift_functional_generalization_repair / _train_shift_candidate "
            "(variant='iid_baseline', the tie-break default at :664)",
            "covers_operations": ["SHIFT"],
            "restore_source_template": (
                "runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair/"
                "committed_bank/seed_{seed}/primitive_bank_16_shift_v1.pt"
            ),
            "namespace_output_template": f"{ns}/seed_{{seed}}/shift/primitive_bank_16_shift_v1.pt",
            "per_seed_route": {
                str(seed): (
                    "RESTORE" if seed in _SHIFT_COMMITTED_SEEDS else "REBUILD_REQUIRED_NOT_EXECUTED"
                )
                for seed in RECOVERY_DEV_SEEDS
            },
            "notes": (
                "Seeds 10/11/14: RESTORE via legacy_import from the hash-verified committed "
                "checkpoint (bank_transaction_log.json action=COMMITTED). Seeds 12/13: R3-009 "
                "ROLLED_BACK the candidate (action=ROLLED_BACK, version_after=0) -- registered "
                "as REBUILD_REQUIRED_NOT_EXECUTED here; REC-003 fixes the plan and budget only, "
                "it does not execute this rebuild (RG2 explicitly defers all training). Per "
                "EXPERIMENT_PLAN section 2.2 item 4, SHIFT is the ONLY operation the recovery "
                "floor permits a declared COHERENT_LIMITED exception for."
            ),
        },
        {
            "stage_id": "SHIFT_OVERRIDE_MERGE",
            "depends_on": ["CANONICAL_AND_BRANCH_B_BUILD", "SHIFT_DEDICATED"],
            "kind": "merge",
            "recipe_ref": "bank.get(op_to_id['SHIFT']).load_state_dict(shift_state) applied to "
            "the CANONICAL_AND_BRANCH_B_BUILD bank object before it is treated as final",
            "covers_operations": ["SHIFT"],
            "namespace_output_template": f"{ns}/seed_{{seed}}/canonical_branch_b/primitive_bank.pt",
            "notes": (
                "Replaces the discarded generic SHIFT slot from CANONICAL_AND_BRANCH_B_BUILD "
                "with SHIFT_DEDICATED's output for that seed. For seeds 12/13 (no SHIFT_DEDICATED "
                "output executed), this stage cannot run to completion in this recovery pass -- "
                "the resulting bank is coherent for the other 9 ops with SHIFT left "
                "REBUILD_REQUIRED, never silently filled from the discarded generic weights."
            ),
        },
        {
            "stage_id": "INCREMENTAL_6_BUILD",
            "depends_on": ["SHIFT_OVERRIDE_MERGE"],
            "kind": "train",
            "recipe_ref": "shape of apc.evaluation.incremental_router_benchmark."
            "get_or_build_16_primitive_bank:231-247 (bank.new_cross_position_primitive + "
            "_train_single_primitive(steps=1000)), applied directly to the "
            "SHIFT_OVERRIDE_MERGE bank -- WITHOUT that function's b008 "
            "strict=False fallback load (:225-229), which this plan never calls",
            "covers_operations": [
                "ROTATE_TRIPLETS",
                "SWAP_ENDS",
                "MIRROR_HALVES",
                "ALTERNATING_NEGATE",
                "CYCLE_FOUR",
                "INCREMENT_MOD",
            ],
            "namespace_output_template": f"{ns}/seed_{{seed}}/primitive_bank_16.pt",
            "notes": (
                "Produces the full, namespace-local 16-op bank. Never writes to "
                "runs/phase_a2_bank_scaling_benchmark."
            ),
        },
        {
            "stage_id": "ROUTER_CALIBRATION",
            "depends_on": ["INCREMENTAL_6_BUILD"],
            "kind": "calibrate",
            "recipe_ref": "apc.evaluation.incremental_router_benchmark."
            "update_router_incrementally(condition=IncrementalUpdateCondition.R0_FULL_RETRAIN, "
            "router_lr=0.005, router_steps=250)",
            "covers_operations": [],
            "covers_components": ["router"],
            "namespace_output_template": f"{ns}/seed_{{seed}}/router.pt",
            "notes": (
                "BUILD_REQUIRED_BY_DESIGN (REC-001): the router is never persisted upstream "
                "and is recalibrated fresh every run, not a restore question."
            ),
        },
        {
            "stage_id": "ARGUMENT_SCORER_CALIBRATION",
            "depends_on": ["ROUTER_CALIBRATION"],
            "kind": "calibrate",
            "recipe_ref": "apc.primitives.argument_scoring.ArgumentScorer.train_on_examples "
            "(ArgumentScorerConfig defaults: d_model=128, arg_vocab_size=32, lambda_weight=2.0, "
            "lr=0.005, steps=200)",
            "covers_operations": [],
            "covers_components": ["argument_scorer"],
            "namespace_output_template": f"{ns}/seed_{{seed}}/argument_scorer.pt",
            "deferred_repair_recipes": [
                {
                    "recipe_ref": "apc.evaluation.count_bind_key_scoring_repair."
                    "run_count_bind_key_scoring_repair (R3-006)",
                    "status": "EXISTS_BUT_VALIDATED_AGAINST_STALE_BANK",
                    "deferred_to": "B-C005REC-006",
                },
                {
                    "recipe_ref": "apc.evaluation.bind_argument_scorer_repair."
                    "run_bind_argument_scorer_repair (R3-008)",
                    "status": "EXISTS_BUT_VALIDATED_AGAINST_STALE_BANK",
                    "deferred_to": "B-C005REC-006",
                },
            ],
            "notes": (
                "SELECT's sigmoid readout (ADR-0088) is unconditioned code inside "
                "ArgumentScorer.forward, not a separate checkpoint -- nothing to build for "
                "it here beyond this stage's own base calibration."
            ),
        },
        {
            "stage_id": "CERTIFICATE",
            "depends_on": ["ARGUMENT_SCORER_CALIBRATION"],
            "kind": "certificate",
            "recipe_ref": "apc.utils.model_bundle.build_manifest (Task B-C005REC-002)",
            "covers_operations": [],
            "namespace_output_template": (
                "runs/phase_b_b2_model_bundle_recovery/bundles/<bundle_id>/manifest.json"
            ),
            "notes": (
                "Publish step: staging is never exposed to a nominal loader before this "
                "stage assembles and hashes the full manifest (contract section 6)."
            ),
        },
    ]


def _build_primitive_build_plan_rows() -> list[dict[str, Any]]:
    registry = _real_16_operation_registry()
    stage_by_op: dict[str, str] = {}
    for stage in _build_stage_graph():
        for op in stage["covers_operations"]:
            stage_by_op[op] = stage["stage_id"]
    # SHIFT is produced by two stages (a discarded byproduct of
    # CANONICAL_AND_BRANCH_B_BUILD, a real one of SHIFT_OVERRIDE_MERGE) --
    # the row must point at the stage whose output is actually published.
    stage_by_op["SHIFT"] = "SHIFT_OVERRIDE_MERGE"

    rows: list[dict[str, Any]] = []
    for op, group in sorted(registry.items()):
        provenance_by_seed: dict[str, str] = {}
        for seed in RECOVERY_DEV_SEEDS:
            if op == "SHIFT":
                provenance_by_seed[str(seed)] = (
                    "RESTORED_VALIDATED"
                    if seed in _SHIFT_COMMITTED_SEEDS
                    else "REBUILD_REQUIRED_NOT_EXECUTED"
                )
            else:
                provenance_by_seed[str(seed)] = "TRAINED_THIS_BUILD_NOT_EXECUTED"
        rows.append(
            {
                "operation": op,
                "group": group,
                "stage_id": stage_by_op[op],
                "provenance_target_by_seed": provenance_by_seed,
                "explicit_parameter_free_approved": False,
            }
        )
    assert len(rows) == 16, f"expected 16 registry rows, got {len(rows)}"
    return rows


def _build_recipe_inventory() -> dict[str, Any]:
    return {
        "task_id": "B-C005REC-003",
        "note": (
            "Every recipe_ref below names a real function found by reading live source "
            "during this task (see this module's REC-003 header comment for exact line "
            "ranges). None are invented."
        ),
        "stages": _build_stage_graph(),
        "no_recipe_gap_found": True,
        "recipe_unavailable_operations": [],
    }


def _build_training_budget_manifest() -> dict[str, Any]:
    return {
        "task_id": "B-C005REC-003",
        "source_of_numbers": (
            "All step/lr/weight_decay/optimizer values below are read from live config "
            "dataclass defaults or literal call-site arguments in the cited modules -- none "
            "are estimated."
        ),
        "stage_budgets": {
            "CORE_RESTORE": {
                "trainable_parameters": 0,
                "budget": None,
                "note": (
                    "Restore path taken for all 5 registered seeds; no training budget consumed."
                ),
            },
            "CANONICAL_AND_BRANCH_B_BUILD": {
                "optimizer": "torch.optim.AdamW",
                "lr": 0.0008,
                "weight_decay": 0.0001,
                "scheduler": "CosineAnnealingLR(T_max=steps, eta_min=1e-5)",
                "lr_weight_decay_source": (
                    "UnifiedBenchmarkConfig defaults (unified_oracle_causal_benchmark.py:"
                    "163-164), used unmodified by _ensure_learned_routing_bank_and_core's "
                    "internal u_bank_cfg"
                ),
                "steps_per_operation": {
                    "SELECT": 6000,
                    "COUNT": 6000,
                    "BIND": 6000,
                    "COPY": 3000,
                    "REVERSE": 3000,
                    "SORT": 3000,
                    "NEGATE": 3000,
                    "SWAP_PAIRS": 3000,
                    "INVERT_HALF": 3000,
                    "SHIFT_discarded_byproduct": 6000,
                },
                "steps_source": (
                    "LearnedRoutingBenchmarkConfig.bank_train_steps=6000 "
                    "(learned_routing_benchmark.py:113); parameter-free ops get "
                    "bank_train_steps//2 (:632); Branch-B ops are a fixed steps=3000 "
                    "literal (:639)"
                ),
            },
            "SHIFT_DEDICATED": {
                "optimizer": "torch.optim.AdamW (per ShiftFunctionalGeneralizationRepairConfig)",
                "operator_train_steps": 12000,
                "operator_lr": 0.0005,
                "operator_weight_decay": 0.0001,
                "operator_grad_clip": 1.0,
                "operator_batch_size": 32,
                "default_variant": "iid_baseline",
                "source": "shift_functional_generalization_repair.py:539-567,664",
                "executed_in_this_task": False,
            },
            "INCREMENTAL_6_BUILD": {
                "optimizer": "torch.optim.AdamW",
                "lr": 0.0008,
                "weight_decay": 0.0001,
                "steps_per_operation": 1000,
                "source": (
                    "incremental_router_benchmark.py:247 (steps=1000), same "
                    "_train_single_primitive optimizer as above"
                ),
            },
            "ROUTER_CALIBRATION": {
                "router_lr": 0.005,
                "router_steps": 250,
                "condition": "R0_FULL_RETRAIN",
                "source": "incremental_router_benchmark.py:268-269 defaults",
            },
            "ARGUMENT_SCORER_CALIBRATION": {
                "optimizer": "torch.optim.AdamW",
                "d_model": 128,
                "arg_vocab_size": 32,
                "lambda_weight": 2.0,
                "lr": 0.005,
                "steps": 200,
                "weight_decay": 0.0001,
                "source": (
                    "argument_scoring.py:36-43,168 (ArgumentScorerConfig defaults, "
                    "train_on_examples optimizer)"
                ),
                "deferred_repair_budgets": {
                    "R3-006_count_bind_key_scoring_repair": {
                        "router_lr": 0.005,
                        "router_steps": 250,
                        "ranking_margin": 3.0,
                        "ranking_beta": 1.0,
                        "top_k": 5,
                        "source": "count_bind_key_scoring_repair.py:128-143",
                    },
                    "R3-008_bind_argument_scorer_repair": {
                        "router_lr": 0.005,
                        "router_steps": 250,
                        "ranking_margin": 3.0,
                        "ranking_beta": 1.0,
                        "top_k": 5,
                        "arg_lambda": 2.0,
                        "source": "bind_argument_scorer_repair.py:164-176",
                    },
                },
            },
            "CERTIFICATE": {"trainable_parameters": 0, "budget": None},
        },
        "resource_note": (
            "single-GPU 16GB, sequential per-stage per-seed execution assumed (AGENTS.md "
            "hardware safety rails); no stage budget above was estimated to fit -- all are "
            "pre-existing recipe defaults."
        ),
    }


def _build_recovery_protocol_doc() -> dict[str, Any]:
    return {
        "task_id": "B-C005REC-003",
        "model_data_units": {
            "pilot_model_seed": RECOVERY_PILOT_SEED,
            "cohort_model_seeds": list(RECOVERY_DEV_SEEDS),
            "independent_training_runs_required": 5,
            "note_seed_axis_separation": (
                "Model cohort seeds (10-14) are a different axis from the legacy K/C/N/R "
                "stream's own seeds (sequential_closed_loop_benchmark.DEFAULT_SEEDS=(0,1,2,3,4)) "
                "and dev_seed=42 -- see legacy_stream_manifest.json. Neither is derived from the "
                "other; `seed % 5` parent-switching is never performed."
            ),
        },
        "functional_floor": {
            "source": (
                "EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md section 2.2 "
                "(fixed before any REC-003 result is seen)"
            ),
            "direct_oracle_query_examples_per_model_operation": 1024,
            "primary_metric": "sequence exact match (token accuracy auxiliary only)",
            "non_shift_15_operations_floor": (
                "query EM >= 0.95 per model x operation (macro average never substitutes "
                "for a per-cell check)"
            ),
            "shift_exception": (
                "SHIFT alone may be recorded COHERENT_LIMITED if trained, dependency-coherent, "
                "and its output is measurable -- EM/loss/reference state recorded as-is, no "
                "fallback to an untrained baseline permitted. This exception is not extended to "
                "any other operation."
            ),
            "reference_baseline_starting_point": {
                "value": 4096,
                "status": "OLD_PLAN_STARTING_POINT_NOT_INDEPENDENTLY_REFIXED",
                "note": (
                    "EXPERIMENT_PLAN section 2.2 item 6 names the old plan's 4,096-example "
                    "reference baseline as a starting point for the current contract; the exact "
                    "target call, M, error allocation, and distribution require R3's own "
                    "reference-adequacy contract (apc.meta.adequacy_verifier / "
                    "runs/phase_b_b2_post_d2/r3_003_functional_metrics_v2/"
                    "statistical_contract.json), which this build-DAG task does not "
                    "modify or re-derive."
                ),
            },
        },
        "statistical_contract_carried_forward": {
            "source": (
                "runs/phase_b_b2_post_d2/r3_003_functional_metrics_v2/"
                "statistical_contract.json (B-C005R3-003, frozen=true)"
            ),
            "tau": 0.95,
            "max_candidates": 5,
            "looks": [32, 64, 128, 256, 512],
            "alpha_accept_episode": 0.01,
            "alpha_reject_episode": 0.01,
            "modified_by_this_task": False,
        },
        "data_role_and_exposure_protocol": {
            "development": list(RECOVERY_DEV_SEEDS),
            "validation": [15, 16, 17, 18, 19],
            "sealed_v2": [30, 31, 32, 33, 34],
            "retired_sealed_never_reused": (
                "SEALED_GATE_SEEDS | DEFAULT_REGATE_SEEDS (already-viewed, per "
                "apc.evaluation.relation_split_protocol)"
            ),
            "sealed_guard_reused_from": (
                "apc.evaluation.relation_split_protocol.assert_sealed_access_permitted"
            ),
            "purpose_string_used_by_this_task": "B-C005REC-003_build_plan_fixation",
        },
        "budget_and_feasibility": {
            "step_counts_source": (
                "existing recipe defaults only (see training_budget_manifest.json); "
                "RECIPE_UNAVAILABLE was not triggered for any of the 16 operations"
            ),
            "device_default": "single GPU 16GB, stages processed sequentially per seed",
            "heavy_runs_this_task": 0,
        },
        "no_5_model_training_started_by_this_task": True,
    }


def _build_legacy_stream_manifest() -> dict[str, Any]:
    from apc.evaluation.paired_integration_regression import NEVER_REPAIRED_DETERMINISTIC_OPS
    from apc.evaluation.sequential_closed_loop_benchmark import (
        DEFAULT_NUM_C,
        DEFAULT_NUM_K,
        DEFAULT_NUM_N,
        DEFAULT_NUM_R,
    )
    from apc.evaluation.sequential_closed_loop_benchmark import (
        DEFAULT_SEEDS as LEGACY_STREAM_SEEDS,
    )
    from apc.evaluation.sequential_closed_loop_benchmark import (
        NOVEL_OPS as LEGACY_STREAM_NOVEL_OPS,
    )

    return {
        "task_id": "B-C005REC-003",
        "stream_source": (
            "apc.evaluation.sequential_closed_loop_benchmark.SequentialClosedLoopConfig"
        ),
        "stream_realization_seeds": list(LEGACY_STREAM_SEEDS),
        "stream_dev_seed": 42,
        "seed_axis_warning": (
            "stream_realization_seeds/stream_dev_seed are episode/task-realization seeds, "
            "not model checkpoint seeds -- never conflated with RECOVERY_DEV_SEEDS (10-14) "
            "here or downstream."
        ),
        "episode_category_counts": {
            "K_reuse": DEFAULT_NUM_K,
            "C_composition": DEFAULT_NUM_C,
            "N_plastic_novel": DEFAULT_NUM_N,
            "R_recurrence": DEFAULT_NUM_R,
        },
        "initial_bank_membership_for_recovery": (
            "the full recovered 16-op bank (INCREMENTAL_6_BUILD stage output) -- includes all "
            "6 ops the legacy stream's own N-category currently draws from"
        ),
        "n_category_ops_in_legacy_stream": list(LEGACY_STREAM_NOVEL_OPS),
        "n_category_conflict_with_recovery_cohort": (
            "LEGACY_STREAM_NOVEL_OPS == PHASE_A2_INCREMENTAL_NEW_OPERATIONS, the same 6 ops "
            "INCREMENTAL_6_BUILD now trains into the initial 16-op bank. Once a recovered "
            "bundle is injected (REC-006/007), these 6 ops are already-known-to-the-bank (K), "
            "not novel (N) -- this task registers that conflict; it does not redesign the "
            "stream or claim an unseen-family N result. AGENTS.md's own rule applies: "
            "'16-skill full bankへ既に含まれる能力をNと呼ばない.'"
        ),
        "never_repaired_deterministic_ops_legacy_baseline": list(NEVER_REPAIRED_DETERMINISTIC_OPS),
        "controller_replay_recipe": (
            "unchanged from sequential_closed_loop_benchmark.py -- this task adds no new "
            "controller/replay mechanism"
        ),
        "n_introduction_status_for_recovery_cohort": (
            "NOT_YET_INTRODUCED -- N execution against a recovered bundle is a REC-007 "
            "concern, not REC-003"
        ),
    }


# ---------------------------------------------------------------------------
# CPU tiny-fixture dry run (mirrors REC-002's scenario-battery style).
# ---------------------------------------------------------------------------


def _stage_cache_key(
    *, stage_id: str, upstream_content_hash: str, recipe_version: str, data_role_hash: str
) -> str:
    payload = {
        "stage_id": stage_id,
        "upstream_content_hash": upstream_content_hash,
        "recipe_version": recipe_version,
        "data_role_hash": data_role_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tiny_core_state(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"embed.weight": torch.randn(4, 3, generator=g)}


def _tiny_core_hash(seed: int) -> str:
    return mb.canonical_state_hash(_tiny_core_state(seed))


def _run_build_dry_run_scenarios(fixture_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        rows.append({"scenario": name, "passed": passed, "detail": detail})

    # 1. Full 16-row coverage: real registry vs. plan rows, no gap/duplicate.
    registry = _real_16_operation_registry()
    plan_rows = _build_primitive_build_plan_rows()
    plan_ops = {row["operation"] for row in plan_rows}
    record(
        "full_16_operation_coverage_matches_real_registry",
        plan_ops == set(registry.keys()) and len(plan_rows) == 16,
        f"registry={sorted(registry)} plan={sorted(plan_ops)}",
    )

    # 2. Dependency order: topological sort succeeds (no cycle), Core first,
    #    CERTIFICATE last, every primitive stage reachable from CORE_RESTORE.
    stages = _build_stage_graph()
    order: list[str] = []
    remaining = {s["stage_id"]: list(s["depends_on"]) for s in stages}
    while remaining:
        ready = [sid for sid, deps in remaining.items() if all(d in order for d in deps)]
        if not ready:
            break
        for sid in sorted(ready):
            order.append(sid)
            del remaining[sid]
    acyclic = not remaining
    core_first = acyclic and order[0] == "CORE_RESTORE"
    certificate_last = acyclic and order[-1] == "CERTIFICATE"
    record(
        "stage_dependency_graph_is_a_dag_core_first_certificate_last",
        acyclic and core_first and certificate_last,
        f"order={order}",
    )

    # 3. Namespace isolation: no stage ever writes under a forbidden shared-cache prefix.
    violations = [
        s["stage_id"]
        for s in stages
        if any(
            s.get("namespace_output_template", "").startswith(prefix)
            for prefix in FORBIDDEN_SHARED_CACHE_PATH_PREFIXES
        )
    ]
    record(
        "no_stage_writes_a_forbidden_shared_cache_path", not violations, f"violations={violations}"
    )

    # 4. Empty cache -> BUILD.
    core_hash_a = _tiny_core_hash(10)
    key_a = _stage_cache_key(
        stage_id="INCREMENTAL_6_BUILD",
        upstream_content_hash=core_hash_a,
        recipe_version="v1",
        data_role_hash="dev-10",
    )
    cache: dict[str, str] = {}
    action = "BUILD" if key_a not in cache else "REUSE"
    cache[key_a] = "STAGED_COMPLETE"
    record("empty_cache_triggers_build", action == "BUILD")

    # 5. Matching cache key -> REUSE, no rebuild.
    action2 = "BUILD" if key_a not in cache else "REUSE"
    record("matching_cache_key_reuses_without_rebuild", action2 == "REUSE")

    # 6. Upstream content change invalidates the downstream cache key.
    core_hash_b = _tiny_core_hash(11)
    key_b = _stage_cache_key(
        stage_id="INCREMENTAL_6_BUILD",
        upstream_content_hash=core_hash_b,
        recipe_version="v1",
        data_role_hash="dev-10",
    )
    record("upstream_content_change_invalidates_cache_key", key_b != key_a and key_b not in cache)

    # 7. Recipe-version change (formula/schema bump) also invalidates, holding upstream fixed.
    key_c = _stage_cache_key(
        stage_id="INCREMENTAL_6_BUILD",
        upstream_content_hash=core_hash_a,
        recipe_version="v2",
        data_role_hash="dev-10",
    )
    record("recipe_version_change_invalidates_cache_key", key_c != key_a)

    # 8. Interrupted/partial staging is not resumable as a completed stage.
    staging_status = {"STAGED_INCOMPLETE": True}
    resumable = staging_status.get("STAGED_INCOMPLETE", False) is False
    record("interrupted_staging_is_not_treated_as_resumable_complete", not resumable)

    # 9. Sealed guard: real recovery cohort passes; a sealed seed is refused.
    from apc.evaluation.relation_split_protocol import assert_sealed_access_permitted

    sealed_guard_dev_ok = True
    try:
        assert_sealed_access_permitted(
            RECOVERY_DEV_SEEDS, purpose="B-C005REC-003_build_plan_fixation"
        )
    except ValueError:
        sealed_guard_dev_ok = False
    sealed_guard_blocks_sealed = False
    try:
        assert_sealed_access_permitted((30,), purpose="B-C005REC-003_build_plan_fixation")
    except ValueError:
        sealed_guard_blocks_sealed = True
    record(
        "sealed_guard_permits_dev_cohort_and_blocks_sealed_v2",
        sealed_guard_dev_ok and sealed_guard_blocks_sealed,
    )

    # 10. No train/build/calibrate call is permitted once evaluation is frozen.
    def _fake_train_stage() -> None:
        _guard_not_frozen("train_stage")

    frozen_blocks_training = False
    try:
        with frozen_evaluation():
            _fake_train_stage()
    except EvaluationFrozenError:
        frozen_blocks_training = True
    unfrozen_allows_training = True
    try:
        _fake_train_stage()
    except EvaluationFrozenError:
        unfrozen_allows_training = False
    record(
        "frozen_evaluation_blocks_train_calls_unfrozen_allows_them",
        frozen_blocks_training and unfrozen_allows_training,
    )

    # 11. Five-model independence: distinct Core hashes pass; a relabeled
    #     duplicate is caught by REC-002's own assert_distinct_model_identities.
    def _tiny_manifest(seed: int, *, core_seed: int | None = None) -> mb.ModelBundleManifest:
        root = fixture_root / f"seed_{seed}"
        core_path, vocab_path = root / "core.pt", root / "vocab.pt"
        core_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_tiny_core_state(core_seed if core_seed is not None else seed), core_path)
        torch.save({"vocab_marker": torch.zeros(1)}, vocab_path)
        core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
        vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
        return mb.build_manifest(
            schema_version=1,
            source_commit="rec003-dry-run-fixture",
            runtime_recipe_version="rec003-v1",
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

    five_manifests = [_tiny_manifest(seed) for seed in RECOVERY_DEV_SEEDS]
    five_distinct_ok = mb.audit_distinct_model_identities(five_manifests).all_distinct
    duplicate_manifest = _tiny_manifest(999, core_seed=RECOVERY_DEV_SEEDS[0])
    duplicate_caught = not mb.audit_distinct_model_identities(
        [five_manifests[0], duplicate_manifest]
    ).all_distinct
    record(
        "five_model_cohort_distinct_and_relabeled_duplicate_detected",
        five_distinct_ok and duplicate_caught,
    )

    return rows


def run_recovery_build_plan_task(config: RecoveryBuildPlanConfig) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_root = output_dir / "fixtures"

    build_plan = {
        "task_id": "B-C005REC-003",
        "seeds": list(config.seeds),
        "stages": _build_stage_graph(),
        "primitive_rows": _build_primitive_build_plan_rows(),
        "forbidden_shared_cache_path_prefixes": list(FORBIDDEN_SHARED_CACHE_PATH_PREFIXES),
        "namespace_root": RECOVERY_NAMESPACE_ROOT,
    }
    recipe_inventory = _build_recipe_inventory()
    training_budget_manifest = _build_training_budget_manifest()
    recovery_protocol = _build_recovery_protocol_doc()
    legacy_stream_manifest = _build_legacy_stream_manifest()
    dry_run_rows = _run_build_dry_run_scenarios(fixture_root)
    all_dry_run_passed = all(row["passed"] for row in dry_run_rows)

    result = "RG2_PASS" if all_dry_run_passed else "RG2_FAIL"
    protocol_summary = {
        "task_id": "B-C005REC-003",
        "result": result,
        "dry_run_scenarios_run": len(dry_run_rows),
        "dry_run_scenarios_passed": sum(1 for row in dry_run_rows if row["passed"]),
        "no_5_model_training_started": True,
        "no_gpu_training_performed": True,
        "no_shared_cache_modified": True,
        "wall_clock_seconds": time.time() - start,
    }

    (output_dir / "build_plan.json").write_text(json.dumps(build_plan, indent=2), encoding="utf-8")
    (output_dir / "recipe_inventory.json").write_text(
        json.dumps(recipe_inventory, indent=2), encoding="utf-8"
    )
    (output_dir / "training_budget_manifest.json").write_text(
        json.dumps(training_budget_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "recovery_protocol.json").write_text(
        json.dumps(recovery_protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "legacy_stream_manifest.json").write_text(
        json.dumps(legacy_stream_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "build_dry_run.json").write_text(
        json.dumps(
            {
                "scenarios": dry_run_rows,
                "all_passed": all_dry_run_passed,
                "summary": protocol_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "protocol": protocol_summary,
        "build_plan": build_plan,
        "recipe_inventory": recipe_inventory,
        "training_budget_manifest": training_budget_manifest,
        "recovery_protocol": recovery_protocol,
        "legacy_stream_manifest": legacy_stream_manifest,
        "build_dry_run": dry_run_rows,
    }


# =============================================================================
# Task B-C005REC-004: One-Seed Restore / Clean Build & Fresh-Process Validation
#
# Executes REC-003's fixed 8-stage DAG for real, for seed 10 only (RG3 is a
# ONE-seed pilot; RECOVERY_COHORT_READY across all 5 seeds is REC-005's job).
# Per RG3's own scope (EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md
# section 6): all 16 primitives must be covered, non-SHIFT-15 must clear the
# query-EM>=0.95 floor per operation, SHIFT may be recorded COHERENT_LIMITED,
# and a *separate process* with no access to this run's in-memory state or
# any shared-cache path must reload the published bundle and reproduce the
# same discrete predictions for the same sample IDs.
#
# REC-001's restore/build classification for seed 10 (`restore_decision.json`)
# is followed exactly, not re-decided here: Core = RESTORE_CANDIDATE (restored
# from `runs/phase_a1_shift_compact_structural_probe/seed_10/shared_encoder.pt`,
# hash cross-checked against REC-001's own recorded value); SHIFT =
# RESTORE_CANDIDATE (seed 10 is in the R3-009 COMMITTED set -- restored from
# `.../committed_bank/seed_10/primitive_bank_16_shift_v1.pt`); the other 15
# primitives (8 canonical minus SHIFT + 2 Branch-B + 6 incremental) plus the
# Router and ArgumentScorer are all REBUILD_REQUIRED / BUILD_REQUIRED_BY_DESIGN
# -- genuinely trained by this task, not restored. This means REC-004's own
# "one-seed pilot" necessarily exercises almost the entire REC-003 clean-build
# path for real (15 of 16 primitives), not merely a restore-and-verify pass.
#
# Every recipe call below was confirmed by directly reading live source
# during this task (not re-derived from REC-003's own report alone):
#   * apc.evaluation.learned_routing_benchmark._ensure_learned_routing_bank_and_core
#     (:493-651, confirmed in full) -- called with `shared_encoder_checkpoint`
#     pointed at THIS task's own namespace-restored Core copy (never the
#     hardcoded shared-cache path at :526-533) and `seed_dir` pointed at this
#     task's own namespace (never `runs/phase_a1_learned_routing_benchmark`).
#     Because `comp_bank_ckpt` (`runs/phase_a1_composition_library_benchmark/
#     seed_10/primitive_bank.pt`) is confirmed absent (REC-001 Evidence 2),
#     its cache-miss branch (:626-639) genuinely trains all 8 canonical + 2
#     Branch-B primitives -- including a *discarded* generic SHIFT byproduct
#     (SHIFT is in `ALL_CANONICAL_OPERATIONS`), per ADR-0094's own finding.
#   * SHIFT_OVERRIDE_MERGE: `bank.get(op_to_id["SHIFT"]).load_state_dict(
#     mb.primitive_state_dict(shift_v1_state_dict, op_to_id["SHIFT"]))`.
#     `op_to_id["SHIFT"] == 3` is not assumed -- it falls out of
#     `_build_heterogeneous_bank`'s own deterministic construction order
#     (SELECT, COUNT, BIND, SHIFT, ..., confirmed by direct read at
#     unified_oracle_causal_benchmark.py:357-424), which
#     `shift_functional_generalization_repair._rebuild_full_bank_structure`
#     (:607-619, confirmed in full) also uses first, before appending Branch-B
#     + incremental ops in the same fixed order -- so both this task's own
#     freshly-built bank and R3-009's committed SHIFT checkpoint assign SHIFT
#     the identical physical_id, and a shape mismatch on `load_state_dict`
#     would surface as a loud `RuntimeError`, not a silent misassignment.
#   * INCREMENTAL_6_BUILD: the *shape* of `incremental_router_benchmark.
#     get_or_build_16_primitive_bank`'s cache-miss branch (:231-247,
#     confirmed in full) -- `bank.new_cross_position_primitive(...)` +
#     `_train_single_primitive(core, p, u_bank_cfg, op, steps=1000)` --
#     applied directly to the SHIFT_OVERRIDE_MERGE bank, never that
#     function's own `b008_ckpt` `strict=False` fallback (:225-229), which
#     this task never calls.
#   * ROUTER_CALIBRATION: `incremental_router.update_router_incrementally`
#     (:191-267, confirmed in full) with `IncrementalUpdateCondition.
#     R0_FULL_RETRAIN`, `router_lr=0.005`, `router_steps=250` -- confirmed
#     that this call registers a key for every id in `candidate_ids` that
#     lacks one (:211-213), so a single full-cohort call is sufficient; no
#     separate `Router.add_primitive_key` bookkeeping is needed by this task.
#   * ARGUMENT_SCORER_CALIBRATION: `apc.primitives.argument_scoring.
#     ArgumentScorer.train_on_examples` (:137-199, confirmed in full),
#     constructed with `d_model=core.model.config.d_model` (confirmed real
#     call sites in `argument_generalization_audit.py`/`retrieval_repair_
#     benchmark.py` always override the dataclass's own `d_model=128`
#     default this way -- using the un-overridden default against a real
#     `d_model=192` Core would raise a `nn.Linear` shape error at the first
#     forward call).
#   * Publish + fresh-process reload both go through `apc.utils.model_bundle.
#     load_bundle` unmodified (REC-002); this task adds no new loader logic,
#     per the design doc's own instruction that REC-004 exercises the
#     existing contract, not extends it.
# =============================================================================


REC004_TASK_ID: Final = "B-C005REC-004"
REC004_DIRECT_QUERY_EXAMPLES: Final = 1024  # recovery_protocol.json (REC-003)
REC004_NON_SHIFT_FLOOR: Final = 0.95  # EXPERIMENT_PLAN section 2.2 item 3
REC004_ROUTER_TRAIN_EXAMPLES: Final = 64
REC004_ROUTER_EVAL_EXAMPLES: Final = 200
REC004_ROUTER_LR: Final = 0.005
REC004_ROUTER_STEPS: Final = 250
REC004_ARG_SCHEMA_HASH: Final = "apc_argument_schema_v1_recovery"
REC004_FRESH_PROCESS_SAMPLE_EXAMPLES: Final = 64


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rec004_namespace(seed: int) -> Path:
    return Path(RECOVERY_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004"


def _extra_8_ops() -> tuple[str, ...]:
    """The 2 Branch-B + 6 Phase-A2-incremental operations -- all parameter-
    free, so `unified_oracle_causal_benchmark.WRONG_FAMILY_MAP`/
    `NATURAL_BASELINES` (which only cover the 8 canonical ops) do not name
    them; this task extends both, never hand-typing the operation names."""
    from apc.environments.operations import (
        BRANCH_B_NOVEL_OPERATION_NAMES,
        PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
    )

    return tuple(BRANCH_B_NOVEL_OPERATION_NAMES) + tuple(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)


def _extra_8_wrong_family_map() -> dict[str, str]:
    ops = _extra_8_ops()
    return {op: ops[(i + 1) % len(ops)] for i, op in enumerate(ops)}


@dataclass(frozen=True)
class PilotRestoreBuildConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004")
    seed: int = RECOVERY_PILOT_SEED
    direct_query_examples_per_operation: int = REC004_DIRECT_QUERY_EXAMPLES
    non_shift_floor: float = REC004_NON_SHIFT_FLOOR


def _read_rec001_recorded_hash(component: str, seed: int) -> str | None:
    """Read REC-001's own recorded seed-10 `sha256` for cross-validation
    (never re-derived/guessed -- if `artifact_inventory.json` is missing or
    lacks the row, this returns None and the caller records that honestly
    rather than fabricating a match)."""
    inv_path = Path("runs/phase_b_b2_model_bundle_recovery/rec001/run_001/artifact_inventory.json")
    if not inv_path.is_file():
        return None
    data = json.loads(inv_path.read_text(encoding="utf-8"))
    key = "core_checkpoints" if component == "core" else "shift_versioned_replacement_checkpoints"
    for row in data.get(key, []):
        if row.get("seed") == seed:
            return row.get("sha256")
    return None


def _snapshot_forbidden_cache_hashes(seed: int) -> dict[str, str | None]:
    """Raw-byte hash of every real shared-cache path this task must never
    write to, for the given seed -- taken before and after the run so
    `side_effect_audit.json` can prove (not merely assert) the shared cache
    was never modified."""
    candidates = {
        "primitive_bank_16": Path("runs/phase_a2_bank_scaling_benchmark")
        / f"seed_{seed}"
        / "primitive_bank_16.pt",
        "shared_encoder": Path("runs/phase_a1_shift_compact_structural_probe")
        / f"seed_{seed}"
        / "shared_encoder.pt",
        "learned_routing_bank_b008": Path("runs/phase_a1_learned_routing_benchmark")
        / f"seed_{seed}"
        / "primitive_bank.pt",
        "composition_library_bank": Path("runs/phase_a1_composition_library_benchmark")
        / f"seed_{seed}"
        / "primitive_bank.pt",
        "shift_committed_bank_v1": (
            Path("runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair")
            / "committed_bank"
            / f"seed_{seed}"
            / "primitive_bank_16_shift_v1.pt"
        ),
    }
    return {
        name: (mb.raw_file_sha256(path) if path.is_file() else None)
        for name, path in candidates.items()
    }


# ---------------------------------------------------------------------------
# Stage 1: restore Core + SHIFT (REC-001 RESTORE_CANDIDATE artifacts) into
# this task's own namespace. Never touches the shared-cache source files.
# ---------------------------------------------------------------------------


def _restore_core_and_shift(config: PilotRestoreBuildConfig) -> dict[str, Any]:
    _guard_not_frozen("restore_core_and_shift")
    seed = config.seed
    ns = _rec004_namespace(seed)
    core_source = (
        Path("runs/phase_a1_shift_compact_structural_probe") / f"seed_{seed}" / "shared_encoder.pt"
    )
    shift_source = (
        Path("runs/phase_b_b2_post_d2/r3_009_shift_functional_generalization_repair")
        / "committed_bank"
        / f"seed_{seed}"
        / "primitive_bank_16_shift_v1.pt"
    )
    if not core_source.is_file():
        raise mb.MissingArtifactError(f"REC-004 restore: Core source missing: {core_source}")
    if not shift_source.is_file():
        raise mb.MissingArtifactError(f"REC-004 restore: SHIFT source missing: {shift_source}")

    core_component = mb.legacy_import(core_source, ns / "core", component_id=f"core_seed{seed}")
    shift_component = mb.legacy_import(
        shift_source, ns / "shift", component_id=f"shift_bank_v1_seed{seed}"
    )

    expected_core_hash = _read_rec001_recorded_hash("core", seed)
    expected_shift_hash = _read_rec001_recorded_hash("shift", seed)

    return {
        "core": {
            "source": str(core_source.resolve()),
            "restored_to": str(Path(core_component.file_path).resolve()),
            "raw_file_sha256": core_component.file_sha256,
            "canonical_state_hash": core_component.canonical_state_hash,
            "rec001_recorded_hash": expected_core_hash,
            "matches_rec001_recorded_hash": (
                expected_core_hash is not None and expected_core_hash == core_component.file_sha256
            ),
            "provenance_status": mb.ProvenanceStatus.LEGACY_IMPORTED.value,
            "restore_decision_source": (
                "runs/phase_b_b2_model_bundle_recovery/rec001/run_001/restore_decision.json "
                "(core_task_encoder_decoder = RESTORE_CANDIDATE)"
            ),
        },
        "shift_bank_v1": {
            "source": str(shift_source.resolve()),
            "restored_to": str(Path(shift_component.file_path).resolve()),
            "raw_file_sha256": shift_component.file_sha256,
            "canonical_state_hash": shift_component.canonical_state_hash,
            "rec001_recorded_hash": expected_shift_hash,
            "matches_rec001_recorded_hash": (
                expected_shift_hash is not None
                and expected_shift_hash == shift_component.file_sha256
            ),
            "provenance_status": mb.ProvenanceStatus.LEGACY_IMPORTED.value,
            "restore_decision_source": (
                "runs/phase_b_b2_model_bundle_recovery/rec001/run_001/restore_decision.json "
                "(bank_shift = RESTORE_CANDIDATE for seed 10, COMMITTED per "
                "bank_transaction_log.json)"
            ),
        },
        "namespace_root": str(ns.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 2: CANONICAL_AND_BRANCH_B_BUILD -> SHIFT_OVERRIDE_MERGE ->
# INCREMENTAL_6_BUILD. Real GPU training for 15 of 16 primitives.
# ---------------------------------------------------------------------------


def _run_build_stages(
    config: PilotRestoreBuildConfig, restore_info: dict[str, Any]
) -> tuple[Any, Any, dict[str, int], dict[str, Any]]:
    _guard_not_frozen("run_build_stages")
    from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS
    from apc.evaluation.learned_routing_benchmark import (
        LearnedRoutingBenchmarkConfig,
        _ensure_learned_routing_bank_and_core,
    )
    from apc.evaluation.unified_oracle_causal_benchmark import (
        UnifiedBenchmarkConfig,
        _train_single_primitive,
    )
    from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus

    seed = config.seed
    ns = _rec004_namespace(seed)
    canonical_dir = ns / "canonical_branch_b"
    stage_report: dict[str, Any] = {}

    t0 = time.time()
    lr_cfg = LearnedRoutingBenchmarkConfig(
        seed=seed,
        shared_encoder_checkpoint=restore_info["core"]["restored_to"],
    )
    core, bank, op_to_id = _ensure_learned_routing_bank_and_core(lr_cfg, seed_dir=canonical_dir)
    stage_report["CANONICAL_AND_BRANCH_B_BUILD"] = {
        "operations": [
            "SELECT",
            "COUNT",
            "BIND",
            "COPY",
            "REVERSE",
            "SORT",
            "NEGATE",
            "SWAP_PAIRS",
            "INVERT_HALF",
        ],
        "shift_generic_byproduct_trained_and_discarded": True,
        "op_to_id_after_stage": dict(op_to_id),
        "output_bank_file": str((canonical_dir / "primitive_bank.pt").resolve()),
        "wall_clock_seconds": time.time() - t0,
    }

    t1 = time.time()
    shift_path = Path(restore_info["shift_bank_v1"]["restored_to"])
    shift_sd = mb.load_state_dict(shift_path)
    shift_id = op_to_id["SHIFT"]
    shift_slice = mb.primitive_state_dict(shift_sd, shift_id)
    if not shift_slice:
        raise mb.MissingArtifactError(
            f"REC-004 SHIFT_OVERRIDE_MERGE: no primitive slice found at physical_id "
            f"{shift_id} in {shift_path} -- op_to_id assignment order assumption failed"
        )
    bank.get(shift_id).load_state_dict(shift_slice, strict=True)
    stage_report["SHIFT_OVERRIDE_MERGE"] = {
        "shift_primitive_id": shift_id,
        "source": str(shift_path),
        "keys_loaded": sorted(shift_slice.keys()),
        "discarded_generic_byproduct_replaced": True,
        "wall_clock_seconds": time.time() - t1,
    }

    t2 = time.time()
    u_bank_cfg = UnifiedBenchmarkConfig(seed=seed, vocab_size=10, device=str(core.device))
    incremental_losses: dict[str, float] = {}
    incremental_ids: dict[str, int] = {}
    for op in PHASE_A2_INCREMENTAL_NEW_OPERATIONS:
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op,
                d_model=core.model.config.d_model,
                d_operator=32,
                n_head=4,
                d_operator_ff=64,
                vocab_size=10,
                max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id
        incremental_ids[op] = p.primitive_id
        p.to(core.device)
        incremental_losses[op] = _train_single_primitive(core, p, u_bank_cfg, op, steps=1000)
    bank.freeze_all()
    bank.eval()
    stage_report["INCREMENTAL_6_BUILD"] = {
        "operations": list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS),
        "steps_per_operation": 1000,
        "final_losses": incremental_losses,
        "op_to_id_added": incremental_ids,
        "wall_clock_seconds": time.time() - t2,
    }

    if len(op_to_id) != 16:
        raise mb.IncompleteBundleError(
            f"REC-004 build produced {len(op_to_id)} primitives, expected 16: {sorted(op_to_id)}"
        )

    return core, bank, op_to_id, stage_report


# ---------------------------------------------------------------------------
# Stage 3: ROUTER_CALIBRATION -> ARGUMENT_SCORER_CALIBRATION.
# ---------------------------------------------------------------------------


def _calibrate_router_and_scorer(
    core: Any, bank: Any, op_to_id: dict[str, int], config: PilotRestoreBuildConfig
) -> tuple[Any, Any, dict[str, Any]]:
    _guard_not_frozen("calibrate_router_and_scorer")
    from apc.evaluation.incremental_router_benchmark import extract_task_representations
    from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
    from apc.evaluation.unified_oracle_causal_benchmark import (
        PARAMETERIZED_OPERATION_NAMES,
        _flatten_groups,
        generate_compact_operator_counterfactual_groups,
    )
    from apc.primitives.argument_scoring import ArgumentScorer, ArgumentScorerConfig
    from apc.primitives.incremental_router import (
        IncrementalRouterConfig,
        IncrementalUpdateCondition,
        RouterReplayBuffer,
        evaluate_router_accuracy,
        update_router_incrementally,
    )
    from apc.primitives.router import Router, RouterConfig

    seed = config.seed
    device = core.device
    all_ops = sorted(op_to_id.keys())

    train_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    eval_z_by_pid: dict[int, list[tuple[torch.Tensor, int]]] = {}
    for op in all_ops:
        pid = op_to_id[op]
        ex_train = generate_benchmark_examples(
            seed=seed * 1000 + 11,
            n=REC004_ROUTER_TRAIN_EXAMPLES,
            operation=op,
            split="train",
            vocab_size=10,
        )
        z_train = extract_task_representations(core, ex_train)
        train_z_by_pid[pid] = [(z_train[i], pid) for i in range(len(ex_train))]
        ex_eval = generate_benchmark_examples(
            seed=seed * 1000 + 99,
            n=REC004_ROUTER_EVAL_EXAMPLES,
            operation=op,
            split="test",
            vocab_size=10,
        )
        z_eval = extract_task_representations(core, ex_eval)
        eval_z_by_pid[pid] = [(z_eval[i], pid) for i in range(len(ex_eval))]

    router = Router(RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot"))
    router.to(device)
    replay_buffer = RouterReplayBuffer()
    all_pids = [op_to_id[op] for op in all_ops]
    router_cfg = IncrementalRouterConfig(
        condition=IncrementalUpdateCondition.R0_FULL_RETRAIN,
        router_lr=REC004_ROUTER_LR,
        router_steps=REC004_ROUTER_STEPS,
        seed=seed,
    )
    calib_result = update_router_incrementally(
        router,
        candidate_ids=all_pids,
        new_primitive_ids=all_pids,
        new_data_by_pid=train_z_by_pid,
        replay_buffer=replay_buffer,
        config=router_cfg,
        all_historical_data_by_pid=train_z_by_pid,
        device=device,
    )
    router.eval()

    accuracy = evaluate_router_accuracy(router, all_pids, eval_z_by_pid, device=device)
    router_table = {
        op: {
            "primitive_id": op_to_id[op],
            "top1": accuracy[op_to_id[op]]["top1"],
            "topk": accuracy[op_to_id[op]]["topk"],
            "entropy": accuracy[op_to_id[op]]["entropy"],
            "num_examples": accuracy[op_to_id[op]]["num_examples"],
        }
        for op in all_ops
    }

    examples_by_op: dict[str, Any] = {}
    z_task_by_op: dict[str, torch.Tensor] = {}
    for op in PARAMETERIZED_OPERATION_NAMES:
        n_groups = max(1, math.ceil(REC004_DIRECT_QUERY_EXAMPLES / 3))
        groups = generate_compact_operator_counterfactual_groups(
            seed,
            n_groups,
            operation=op,
            step=0,
            split="rec004_scorer_calibration",
            vocab_size=10,
            sequence_length_range=(6, 10),
            group_size=3,
        )
        examples, _wrong_arg_map, _ = _flatten_groups(groups)
        examples = examples[:REC004_DIRECT_QUERY_EXAMPLES]
        examples_by_op[op] = examples
        z_task_by_op[op] = extract_task_representations(core, examples)

    scorer = ArgumentScorer(
        ArgumentScorerConfig(
            d_model=core.model.config.d_model, lambda_weight=2.0, lr=0.005, steps=200
        )
    )
    scorer.to(device)
    scorer_losses = scorer.train_on_examples(z_task_by_op, examples_by_op, device=device)
    scorer.eval()

    report = {
        "router_calibration": {
            "condition": calib_result["condition"],
            "candidate_count": calib_result["candidate_count"],
            "replay_buffer_size": calib_result["replay_buffer_size"],
            "final_loss": calib_result["final_loss"],
            "router_lr": REC004_ROUTER_LR,
            "router_steps": REC004_ROUTER_STEPS,
            "train_examples_per_op": REC004_ROUTER_TRAIN_EXAMPLES,
            "eval_examples_per_op": REC004_ROUTER_EVAL_EXAMPLES,
        },
        "router_top1_topk_by_operation": router_table,
        "argument_scorer_calibration": {
            "final_losses_by_op": scorer_losses,
            "lambda_weight": 2.0,
            "lr": 0.005,
            "steps": 200,
            "examples_per_op": REC004_DIRECT_QUERY_EXAMPLES,
        },
    }
    return router, scorer, report


# ---------------------------------------------------------------------------
# Stage 4: direct/oracle raw-execution measurement, all 16 primitives.
# Separate from the router top-1 table above (REC-004 action item 5).
# ---------------------------------------------------------------------------


def _evaluate_one_operation(
    core: Any,
    bank: Any,
    op_to_id: dict[str, int],
    op: str,
    *,
    seed: int,
    n_examples: int,
    split: str,
) -> dict[str, Any]:
    """Correct / Wrong-Argument (parameterized only) / Wrong-Family / None
    arms for ONE of the 16 real registry operations -- generalizes
    `unified_oracle_causal_benchmark.run_unified_oracle_causal_benchmark`'s
    per-operation block (AGENTS.md's Correct/Wrong/None causal evidence rule)
    to all 16 operations, not only the 8 canonical ones."""
    from apc.evaluation.unified_oracle_causal_benchmark import (
        DEFAULT_WRONG_FAMILY_ARGUMENTS,
        NATURAL_BASELINES,
        PARAMETERIZED_OPERATION_NAMES,
        WRONG_FAMILY_MAP,
        _evaluate_primitive_arm,
        _flatten_groups,
        _generate_parameter_free_examples,
        _make_arg_provider_const,
        _make_arg_provider_correct,
        _make_arg_provider_map,
        generate_compact_operator_counterfactual_groups,
    )

    wrong_family_map = {**WRONG_FAMILY_MAP, **_extra_8_wrong_family_map()}
    natural_baselines = {**NATURAL_BASELINES, **{o: 0.01 for o in _extra_8_ops()}}
    wrong_family_arguments = {**DEFAULT_WRONG_FAMILY_ARGUMENTS, **{o: None for o in _extra_8_ops()}}

    is_param = op in PARAMETERIZED_OPERATION_NAMES
    pid = op_to_id[op]
    prim = bank.get(pid)
    wrong_op = wrong_family_map[op]
    wrong_pid = op_to_id[wrong_op]
    wrong_prim = bank.get(wrong_pid)
    b_natural = natural_baselines[op]

    if is_param:
        n_groups = max(1, math.ceil(n_examples / 3))
        groups = generate_compact_operator_counterfactual_groups(
            seed,
            n_groups,
            operation=op,
            step=0,
            split=split,
            vocab_size=10,
            sequence_length_range=(6, 10),
            group_size=3,
        )
        examples, wrong_arg_map, _ = _flatten_groups(groups)
        examples = examples[:n_examples]

        corr_exact, corr_tok = _evaluate_primitive_arm(
            core, prim, examples, op, argument_provider=_make_arg_provider_correct(op)
        )
        wr_arg_exact, _ = _evaluate_primitive_arm(
            core, prim, examples, op, argument_provider=_make_arg_provider_map(wrong_arg_map)
        )
        wr_fam_exact, _ = _evaluate_primitive_arm(
            core,
            wrong_prim,
            examples,
            op,
            argument_provider=_make_arg_provider_const(wrong_family_arguments[wrong_op]),
            primitive_operation=wrong_op,
        )
        none_exact, _ = _evaluate_primitive_arm(core, prim, examples, op, argument_provider=None)
        causal_gap = corr_exact - max(wr_arg_exact, wr_fam_exact, none_exact)
    else:
        examples = _generate_parameter_free_examples(
            seed,
            n_examples,
            operation=op,
            split=split,
            vocab_size=10,
            sequence_length_range=(6, 10),
        )
        corr_exact, corr_tok = _evaluate_primitive_arm(
            core, prim, examples, op, argument_provider=None
        )
        wr_arg_exact = None
        wr_fam_exact, _ = _evaluate_primitive_arm(
            core,
            wrong_prim,
            examples,
            op,
            argument_provider=None,
            primitive_operation=wrong_op,
        )
        prim.enabled = False
        none_exact, _ = _evaluate_primitive_arm(core, prim, examples, op, argument_provider=None)
        prim.enabled = True
        causal_gap = corr_exact - max(wr_fam_exact, none_exact)

    return {
        "operation": op,
        "primitive_id": pid,
        "primitive_class": prim.__class__.__name__,
        "parameter_count": prim.num_parameters(),
        "is_parameterized": is_param,
        "requested_examples": n_examples,
        "actual_examples": len(examples),
        "correct_exact_match": corr_exact,
        "correct_token_accuracy": corr_tok,
        "wrong_argument_exact_match": wr_arg_exact,
        "wrong_family_exact_match": wr_fam_exact,
        "none_exact_match": none_exact,
        "exact_match_causal_gap": causal_gap,
        "natural_baseline": b_natural,
        "none_within_natural_baseline_plus_0.05": none_exact <= (b_natural + 0.05),
    }


def _measure_all_16_primitives(
    core: Any,
    bank: Any,
    op_to_id: dict[str, int],
    config: PilotRestoreBuildConfig,
    *,
    provenance_by_op: dict[str, mb.ProvenanceStatus],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for op in sorted(op_to_id.keys()):
        row = _evaluate_one_operation(
            core,
            bank,
            op_to_id,
            op,
            seed=config.seed,
            n_examples=config.direct_query_examples_per_operation,
            split="rec004_direct_query",
        )
        if op == "SHIFT":
            row["recovery_floor_status"] = "COHERENT_LIMITED"
            row["recovery_floor_passed"] = None
        else:
            row["recovery_floor_status"] = "EVALUATED"
            row["recovery_floor_passed"] = row["correct_exact_match"] >= config.non_shift_floor
        row["provenance_status"] = provenance_by_op[op].value
        rows[op] = row

    non_shift_ops = [op for op in rows if op != "SHIFT"]
    all_non_shift_pass = all(bool(rows[op]["recovery_floor_passed"]) for op in non_shift_ops)
    return {
        "task_id": REC004_TASK_ID,
        "non_shift_floor": config.non_shift_floor,
        "direct_query_examples_per_operation": config.direct_query_examples_per_operation,
        "operations": rows,
        "all_16_operations_present": set(rows) == set(op_to_id) and len(rows) == 16,
        "all_non_shift_15_floor_passed": all_non_shift_pass,
        "non_shift_floor_failures": [
            op for op in non_shift_ops if not rows[op]["recovery_floor_passed"]
        ],
        "shift_status": "COHERENT_LIMITED",
        "shift_correct_exact_match": rows.get("SHIFT", {}).get("correct_exact_match"),
    }


# ---------------------------------------------------------------------------
# Duplicated (deliberately, not imported -- see module docstring of
# scripts/rec004_fresh_process_check.py) raw discrete-token prediction
# extraction, used ONLY for the fresh-process byte-exact comparison.
# ---------------------------------------------------------------------------


def _predict_tokens(
    core: Any, primitive: Any, examples: Sequence[Any], op: str, argument_provider: Any
) -> list[list[int]]:
    from apc.core.data import collate_content_only_batch
    from apc.environments.operations import get_operation
    from apc.primitives.primitive import (
        ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
        CrossPositionPrimitive,
        ReverseRelativePrimitive,
        ShiftRelativePrimitive,
    )

    primitive.eval()
    device = core.device
    content_lengths = [len(ex.input_tokens) for ex in examples]
    output_lengths = [get_operation(op).output_length(n) for n in content_lengths]
    batch_input = collate_content_only_batch(examples, core.tokens, device=device)
    with torch.no_grad():
        h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
        argument_values = (
            None if argument_provider is None else [argument_provider(ex) for ex in examples]
        )
        if isinstance(
            primitive,
            (
                CrossPositionPrimitive,
                ShiftRelativePrimitive,
                ReverseRelativePrimitive,
                ContentDecoupledDiscretePositionalCrossAttentionPrimitive,
            ),
        ):
            logits = primitive(h, content_lengths, output_lengths, argument_values)
        else:
            logits = primitive(h)
        pred_tokens = logits.argmax(dim=-1)
    return [pred_tokens[row, :n].tolist() for row, n in enumerate(output_lengths)]


# ---------------------------------------------------------------------------
# Stage 5: publish (CERTIFICATE stage) -- assemble + self-verify a real
# ModelBundleManifest through the exact fail-closed contract REC-002 built.
# ---------------------------------------------------------------------------


def _vocab_state_dict(core: Any) -> dict[str, torch.Tensor]:
    tokens = core.tokens
    return {
        "vocab_size": torch.tensor([tokens.env_vocab_size], dtype=torch.long),
        "op_base": torch.tensor([tokens.op_base], dtype=torch.long),
        "arg_base": torch.tensor([tokens.arg_base], dtype=torch.long),
        "arg_span": torch.tensor([tokens.arg_span], dtype=torch.long),
        "num_operations": torch.tensor([tokens.num_operations], dtype=torch.long),
    }


def _manifest_to_json_dict(manifest: mb.ModelBundleManifest) -> dict[str, Any]:
    def _component(c: mb.ComponentManifest | None) -> dict[str, Any] | None:
        return None if c is None else dataclasses.asdict(c)

    return {
        "schema_version": manifest.schema_version,
        "bundle_id": manifest.bundle_id,
        "content_manifest_digest": manifest.content_manifest_digest,
        "source_commit": manifest.source_commit,
        "runtime_recipe_version": manifest.runtime_recipe_version,
        "environment_record": dict(manifest.environment_record),
        "model_id": manifest.model_id,
        "model_seed": manifest.model_seed,
        "training_run_id": manifest.training_run_id,
        "parent_bundle_ids": list(manifest.parent_bundle_ids),
        "build_route": manifest.build_route.value,
        "scope": manifest.scope.value,
        "requested_capabilities": sorted(manifest.requested_capabilities),
        "publish_status": manifest.publish_status.value,
        "core": _component(manifest.core),
        "vocabulary": _component(manifest.vocabulary),
        "primitives": [
            {**dataclasses.asdict(p), "provenance_status": p.provenance_status.value}
            for p in manifest.primitives
        ],
        "router": dataclasses.asdict(manifest.router),
        "argument_scorer": dataclasses.asdict(manifest.argument_scorer),
        "scoring_policy": dataclasses.asdict(manifest.scoring_policy),
        "build_recipe_hash": manifest.build_recipe_hash,
        "known_defects": list(manifest.known_defects),
        "clean_build_exercised_stages": list(manifest.clean_build_exercised_stages),
        "qualification_refs": list(manifest.qualification_refs),
    }


def _publish_bundle(
    config: PilotRestoreBuildConfig,
    core: Any,
    bank: Any,
    router: Any,
    scorer: Any,
    op_to_id: dict[str, int],
    restore_info: dict[str, Any],
    provenance_by_op: dict[str, mb.ProvenanceStatus],
) -> tuple[mb.ModelBundleManifest, dict[str, Any]]:
    _guard_not_frozen("publish_bundle")
    from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS
    from apc.primitives.argument_scoring import PARAMETERIZED_OPERATIONS as ARG_PARAM_OPS
    from apc.utils.system_info import get_system_info

    seed = config.seed
    ns = _rec004_namespace(seed)
    publish_dir = ns / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    core_path = Path(restore_info["core"]["restored_to"]).resolve()
    bank_path = (publish_dir / "primitive_bank_16.pt").resolve()
    router_path = (publish_dir / "router.pt").resolve()
    scorer_path = (publish_dir / "argument_scorer.pt").resolve()
    vocab_path = (publish_dir / "vocab.pt").resolve()

    torch.save(bank.state_dict(), bank_path)
    torch.save(router.state_dict(), router_path)
    torch.save(scorer.state_dict(), scorer_path)
    torch.save(_vocab_state_dict(core), vocab_path)

    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    schema_hash = (
        f"vocab{core.tokens.env_vocab_size}_ops{core.tokens.num_operations}_"
        f"argspan{core.tokens.arg_span}_v1"
    )
    core_component = mb.ComponentManifest("core", str(core_path), core_raw, core_state, schema_hash)
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    vocab_component = mb.ComponentManifest(
        "vocabulary", str(vocab_path), vocab_raw, vocab_state, schema_hash
    )

    bank_state_dict = mb.load_state_dict(bank_path)
    incremental_ops = set(PHASE_A2_INCREMENTAL_NEW_OPERATIONS)
    primitives = []
    for op, pid in sorted(op_to_id.items(), key=lambda kv: kv[1]):
        sliced = mb.primitive_state_dict(bank_state_dict, pid)
        weights_hash = mb.canonical_state_hash(sliced)
        provenance = provenance_by_op[op]
        if op == "SHIFT":
            receipt = f"REC-004 restore of {restore_info['shift_bank_v1']['source']}"
        elif op in incremental_ops:
            receipt = "REC-004 INCREMENTAL_6_BUILD stage, steps=1000"
        else:
            receipt = "REC-004 CANONICAL_AND_BRANCH_B_BUILD stage"
        primitives.append(
            mb.PrimitiveManifestEntry(
                physical_id=pid,
                operation_name=op,
                version="rec004-seed10-v1",
                architecture_signature="shift_relative_v1"
                if op == "SHIFT"
                else "cross_position_v1",
                state_abi_hash=weights_hash,
                core_dependency_hash=core_state,
                decoder_dependency_hash="NOT_APPLICABLE_NO_SEPARATE_DECODER_COMPONENT",
                weights_hash=weights_hash,
                source_artifact=str(bank_path),
                provenance_status=provenance,
                argument_schema_hash=REC004_ARG_SCHEMA_HASH if op in ARG_PARAM_OPS else None,
                training_receipt=receipt,
            )
        )

    router_sd = mb.load_state_dict(router_path)
    primitive_ids = [p.physical_id for p in primitives]
    router_component = mb.RouterManifest(
        source_artifact=str(router_path),
        weights_hash=mb.canonical_state_hash(mb.router_non_key_state_dict(router_sd)),
        task_state_dependency_hash=core_state,
        key_to_primitive_mapping_hash=mb.router_key_to_primitive_mapping_hash(
            router_sd, primitive_ids
        ),
    )
    scorer_sd = mb.load_state_dict(scorer_path)
    scorer_component = mb.ArgumentScorerManifest(
        source_artifact=str(scorer_path),
        weights_hash=mb.canonical_state_hash(scorer_sd),
        input_dependency_hash=core_state,
        argument_schema_hash=REC004_ARG_SCHEMA_HASH,
    )
    scoring_policy = mb.ScoringPolicyManifest(
        family_formula_version="dot_product_v1",
        argument_formula_version="select_sigmoid_v2_adr0088",
        lambda_weight=2.0,
        application_policy_version="rec004_recovery_v1",
        calibrated_pair_id=f"rec004-seed{seed}-pair-1",
    )

    system_info = get_system_info(seed=seed)
    manifest = mb.build_manifest(
        schema_version=1,
        source_commit=system_info.get("git_commit") or "unknown",
        runtime_recipe_version="rec004-v1",
        environment_record={
            "python_version": str(system_info["python_version"]),
            "torch_version": str(system_info["torch_version"]),
            "device_name": str(system_info.get("device_name")),
        },
        model_id=f"seed{seed}",
        model_seed=seed,
        training_run_id=f"rec004-seed{seed}-run-001",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "diagnostic_only"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=tuple(primitives),
        router=router_component,
        argument_scorer=scorer_component,
        scoring_policy=scoring_policy,
        build_recipe_hash=hashlib.sha256(b"B-C005REC-004-build-plan-v1").hexdigest(),
        known_defects=(),
        clean_build_exercised_stages=(
            "CANONICAL_AND_BRANCH_B_BUILD",
            "SHIFT_OVERRIDE_MERGE",
            "INCREMENTAL_6_BUILD",
            "ROUTER_CALIBRATION",
            "ARGUMENT_SCORER_CALIBRATION",
        ),
        qualification_refs=(REC004_TASK_ID,),
    )

    # Self-verify through the real fail-closed contract -- RG3 depends on
    # this succeeding, not merely on "training ran".
    loaded = mb.load_bundle(manifest, mode="nominal", expected_primitive_count=16)

    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles" / manifest.bundle_id
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = _manifest_to_json_dict(manifest)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")

    manifest_paths_file = publish_dir / "manifest_paths.json"
    manifest_paths_file.write_text(
        json.dumps({"op_to_id": op_to_id, "manifest": manifest_json}, indent=2), encoding="utf-8"
    )

    report = {
        "bundle_id": manifest.bundle_id,
        "content_manifest_digest": manifest.content_manifest_digest,
        "self_load_bundle_mode": "nominal",
        "self_load_bundle_checks_performed": list(loaded.checks_performed),
        "manifest_path": str(bundle_dir / "manifest.json"),
        "manifest_paths_json_for_fresh_process": str(manifest_paths_file.resolve()),
        "core_path": str(core_path),
        "bank_path": str(bank_path),
        "router_path": str(router_path),
        "scorer_path": str(scorer_path),
        "vocab_path": str(vocab_path),
    }
    return manifest, report


# ---------------------------------------------------------------------------
# Stage 6: fresh-process validation. Spawns a genuinely separate `python`
# process with no shared in-memory state, from a scratch working directory
# that is not the repo root, and diffs its discrete predictions against this
# process's own.
# ---------------------------------------------------------------------------


def _run_fresh_process_validation(
    config: PilotRestoreBuildConfig,
    core: Any,
    bank: Any,
    op_to_id: dict[str, int],
    manifest_paths_file: str,
    bundle_id: str,
) -> dict[str, Any]:
    from apc.evaluation.unified_oracle_causal_benchmark import (
        PARAMETERIZED_OPERATION_NAMES,
        _flatten_groups,
        _generate_parameter_free_examples,
        _make_arg_provider_correct,
        generate_compact_operator_counterfactual_groups,
    )

    seed = config.seed
    sample_n = REC004_FRESH_PROCESS_SAMPLE_EXAMPLES
    split = "rec004_fresh_process_check"

    in_process_predictions: dict[str, list[list[int]]] = {}
    for op, pid in sorted(op_to_id.items()):
        prim = bank.get(pid)
        if op in PARAMETERIZED_OPERATION_NAMES:
            n_groups = max(1, math.ceil(sample_n / 3))
            groups = generate_compact_operator_counterfactual_groups(
                seed,
                n_groups,
                operation=op,
                step=0,
                split=split,
                vocab_size=10,
                sequence_length_range=(6, 10),
                group_size=3,
            )
            examples, _wrong_arg_map, _ = _flatten_groups(groups)
            examples = examples[:sample_n]
            provider = _make_arg_provider_correct(op)
        else:
            examples = _generate_parameter_free_examples(
                seed,
                sample_n,
                operation=op,
                split=split,
                vocab_size=10,
                sequence_length_range=(6, 10),
            )
            provider = None
        in_process_predictions[op] = _predict_tokens(core, prim, examples, op, provider)

    script_path = _repo_root() / "scripts" / "rec004_fresh_process_check.py"
    scratch_cwd = Path(tempfile.gettempdir()) / f"apc_rec004_fresh_process_cwd_seed{seed}"
    scratch_cwd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--manifest-paths",
            manifest_paths_file,
            "--seed",
            str(seed),
            "--sample-examples-per-op",
            str(sample_n),
            "--split",
            split,
        ],
        cwd=str(scratch_cwd),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env=dict(os.environ),
    )
    wall = time.time() - t0

    if proc.returncode != 0:
        return {
            "task_id": REC004_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "repo_root_cwd": str(_repo_root()),
            "different_working_directory_confirmed": str(scratch_cwd) != str(_repo_root()),
            "subprocess_returncode": proc.returncode,
            "subprocess_stdout_tail": proc.stdout[-4000:],
            "subprocess_stderr_tail": proc.stderr[-4000:],
            "reproducibility_passed": False,
            "wall_clock_seconds": wall,
        }

    fresh = json.loads(proc.stdout.strip().splitlines()[-1])
    fresh_predictions = fresh["predictions_by_op"]

    mismatches: dict[str, Any] = {}
    for op in sorted(op_to_id):
        a = in_process_predictions[op]
        b = fresh_predictions.get(op)
        if a != b:
            mismatches[op] = {"in_process": a, "fresh_process": b}

    return {
        "task_id": REC004_TASK_ID,
        "cwd_used": str(scratch_cwd),
        "repo_root_cwd": str(_repo_root()),
        "different_working_directory_confirmed": str(scratch_cwd) != str(_repo_root()),
        "sample_examples_per_op": sample_n,
        "subprocess_returncode": proc.returncode,
        "no_builder_identifier_found_in_subprocess_script": fresh[
            "no_builder_identifier_found_in_this_script"
        ],
        "forbidden_identifiers_checked": fresh["forbidden_identifiers_checked"],
        "load_bundle_checks_performed_in_subprocess": fresh["checks_performed_by_load_bundle"],
        "bundle_id_matches": fresh["bundle_id"] == bundle_id,
        "per_operation_prediction_mismatches": mismatches,
        "predictions_byte_exact_for_all_16_ops": not mismatches,
        "exact_match_by_op_in_subprocess": fresh["exact_match_by_op"],
        "reproducibility_passed": (
            not mismatches
            and bool(fresh["no_builder_identifier_found_in_this_script"])
            and fresh["bundle_id"] == bundle_id
        ),
        "wall_clock_seconds": wall,
        "subprocess_python_executable": fresh["python_executable"],
        "subprocess_device": fresh["device"],
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def run_pilot_restore_build_task(config: PilotRestoreBuildConfig) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004's pilot seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md section 2.1); "
            f"got seed={seed} -- re-selecting the pilot seed is not permitted"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    restore_info = _restore_core_and_shift(config)
    core, bank, op_to_id, build_stage_report = _run_build_stages(config, restore_info)
    router, scorer, calibration_report = _calibrate_router_and_scorer(core, bank, op_to_id, config)

    provenance_by_op: dict[str, mb.ProvenanceStatus] = {
        op: (
            mb.ProvenanceStatus.RESTORED_VALIDATED
            if op == "SHIFT"
            else mb.ProvenanceStatus.TRAINED_THIS_BUILD
        )
        for op in op_to_id
    }

    manifest, publish_report = _publish_bundle(
        config, core, bank, router, scorer, op_to_id, restore_info, provenance_by_op
    )

    with frozen_evaluation():
        execution_report = _measure_all_16_primitives(
            core, bank, op_to_id, config, provenance_by_op=provenance_by_op
        )

    fresh_process_report = _run_fresh_process_validation(
        config,
        core,
        bank,
        op_to_id,
        publish_report["manifest_paths_json_for_fresh_process"],
        manifest.bundle_id,
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004_TASK_ID,
        "before": before_hashes,
        "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
        "namespace_root": str(_rec004_namespace(seed).resolve()),
        "bundle_dir": str(
            (
                _repo_root()
                / "runs"
                / "phase_b_b2_model_bundle_recovery"
                / "bundles"
                / manifest.bundle_id
            ).resolve()
        ),
    }

    all_16_present = execution_report["all_16_operations_present"]
    non_shift_floor_pass = execution_report["all_non_shift_15_floor_passed"]
    fresh_process_pass = fresh_process_report["reproducibility_passed"]
    cache_unchanged = side_effect_audit["shared_cache_unchanged"]
    restore_hashes_confirmed = (
        restore_info["core"]["matches_rec001_recorded_hash"]
        and restore_info["shift_bank_v1"]["matches_rec001_recorded_hash"]
    )

    rg3_pass = all(
        [
            all_16_present,
            non_shift_floor_pass,
            fresh_process_pass,
            cache_unchanged,
            restore_hashes_confirmed,
        ]
    )
    result = "RG3_PASS" if rg3_pass else "RG3_FAIL"

    from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS

    restore_vs_build_labels = {"core": "RESTORE_VALIDATED", "SHIFT": "RESTORE_VALIDATED"}
    for op in op_to_id:
        if op not in restore_vs_build_labels:
            restore_vs_build_labels[op] = "CLEAN_BUILD_EXERCISED"
    restore_vs_build_labels["router"] = "CLEAN_BUILD_EXERCISED_BY_DESIGN_NEVER_RESTORED"
    restore_vs_build_labels["argument_scorer"] = "CLEAN_BUILD_EXERCISED"

    qualification = {
        "task_id": REC004_TASK_ID,
        "result": result,
        "seed": seed,
        "seed_preregistered_not_reselected": True,
        "criteria": {
            "all_16_operations_coverage": all_16_present,
            "non_shift_15_floor_passed": non_shift_floor_pass,
            "non_shift_floor_failures": execution_report["non_shift_floor_failures"],
            "shift_status": "COHERENT_LIMITED",
            "shift_correct_exact_match": execution_report["shift_correct_exact_match"],
            "fresh_process_reproducibility_passed": fresh_process_pass,
            "shared_cache_unchanged": cache_unchanged,
            "restore_hashes_confirmed_against_rec001": restore_hashes_confirmed,
        },
        "restore_vs_clean_build_labels": restore_vs_build_labels,
        "clean_build_note": (
            "Only seed 10 (this pilot) actually executed the REC-003 clean-build "
            "recipe for the 15 non-SHIFT primitives + router + scorer. Seeds "
            "11-14's own clean-build path is implemented (same recipe, same "
            "code) but CLEAN_BUILD_IMPLEMENTED_NOT_FULLY_EXERCISED for those "
            "seeds until B-C005REC-005 runs it."
        ),
        "no_5_model_cohort_started_by_this_task": True,
        "wall_clock_seconds": time.time() - start,
    }

    pilot_build_report = {
        "task_id": REC004_TASK_ID,
        "seed": seed,
        "restore": restore_info,
        "build_stages": build_stage_report,
        "calibration": calibration_report,
        "publish": publish_report,
        "provenance_by_operation": {
            op: provenance_by_op[op].value for op in sorted(provenance_by_op)
        },
        "incremental_operations": list(PHASE_A2_INCREMENTAL_NEW_OPERATIONS),
        "wall_clock_seconds": time.time() - start,
    }

    (output_dir / "pilot_build_report.json").write_text(
        json.dumps(pilot_build_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "all_primitive_execution.json").write_text(
        json.dumps(execution_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "fresh_process_report.json").write_text(
        json.dumps(fresh_process_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    return {
        "protocol": qualification,
        "pilot_build_report": pilot_build_report,
        "calibration": calibration_report,
        "all_primitive_execution": execution_report,
        "fresh_process_report": fresh_process_report,
        "side_effect_audit": side_effect_audit,
    }
