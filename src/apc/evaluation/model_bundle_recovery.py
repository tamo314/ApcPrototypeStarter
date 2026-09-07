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
import json
import time
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, get_type_hints

import torch

from apc.utils import model_bundle as mb

__all__ = ["ModelBundleContractConfig", "run_model_bundle_contract_task"]


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
