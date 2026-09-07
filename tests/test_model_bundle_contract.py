"""CPU-only contract tests for Task B-C005REC-002's `apc.utils.model_bundle`.

Covers every scenario `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`'s
REC-002 section and `docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
RG1 require: a normal full bundle loads; each declared corruption/mismatch/
incompleteness mode fails explicitly with the right typed error; the loader
never calls a builder/optimizer. All fixtures are tiny, synthetic, in-memory
tensors written to `tmp_path` -- no real checkpoints, no GPU, no training.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest
import torch

from apc.utils import model_bundle as mb


def _refreshed(manifest: mb.ModelBundleManifest, **changes: object) -> mb.ModelBundleManifest:
    """Apply `changes` via `dataclasses.replace` and recompute
    `content_manifest_digest`/`bundle_id` so the result passes
    `_verify_manifest_self_consistency` -- used to build a manifest whose
    *content* is deliberately wrong (e.g. a stale `core_dependency_hash`)
    without also tripping the unrelated hand-tamper-detection check."""
    draft = dataclasses.replace(manifest, **changes)
    digest = mb.compute_content_manifest_digest(draft)
    bundle_id = mb.compute_bundle_id(draft, content_manifest_digest=digest)
    return dataclasses.replace(draft, content_manifest_digest=digest, bundle_id=bundle_id)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_PRIMITIVE_IDS = (0, 1, 2)
_PARAMETERIZED_OP = "SELECT"
_ARG_SCHEMA_HASH = "arg-schema-v1"
_CORE_SCHEMA_HASH = "vocab-size-10-v1"


def _core_state_dict(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"embed.weight": torch.randn(6, 4, generator=g)}


def _bank_state_dict(primitive_ids: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for pid in primitive_ids:
        sd[f"_primitives.{pid}.weight"] = torch.randn(4, 4, generator=g)
        sd[f"_primitives.{pid}.bias"] = torch.randn(4, generator=g)
    return sd


def _router_state_dict(primitive_ids: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {
        "query_proj.weight": torch.randn(4, 4, generator=g),
        "query_proj.bias": torch.randn(4, generator=g),
    }
    for pid in primitive_ids:
        sd[f"_keys.{pid}"] = torch.randn(4, generator=g)
    return sd


def _scorer_state_dict(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        f"heads.{_PARAMETERIZED_OP}.weight": torch.randn(8, 4, generator=g),
        f"heads.{_PARAMETERIZED_OP}.bias": torch.randn(8, generator=g),
    }


def _save(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


class Bundle:
    """A built manifest plus the tmp_path root its files live under, so
    tests can mutate the on-disk files after the manifest was frozen."""

    def __init__(self, root: Path, manifest: mb.ModelBundleManifest) -> None:
        self.root = root
        self.manifest = manifest


def build_valid_bundle(
    tmp_path: Path,
    *,
    model_seed: int = 10,
    core_seed: int | None = None,
    primitive_ids: tuple[int, ...] = _PRIMITIVE_IDS,
    bank_seed: int = 1,
    router_seed: int = 2,
    scorer_seed: int = 3,
    publish_status: mb.PublishStatus = mb.PublishStatus.PUBLISHED,
    scope: mb.BundleScope = mb.BundleScope.NOMINAL,
    requested_capabilities: frozenset[str] = frozenset({"nominal_execution"}),
    provenance_status: mb.ProvenanceStatus = mb.ProvenanceStatus.RESTORED_VALIDATED,
    training_receipt: str | None = "receipt-1",
    argument_formula_version: str = "select_sigmoid_v2_adr0088",
) -> Bundle:
    root = tmp_path / f"bundle_seed{model_seed}"
    core_seed = model_seed if core_seed is None else core_seed

    core_path = root / "core" / "shared_encoder.pt"
    bank_path = root / "bank" / "primitive_bank.pt"
    router_path = root / "router" / "router.pt"
    scorer_path = root / "scorer" / "argument_scorer.pt"
    vocab_path = root / "vocab" / "vocab.pt"

    _save(core_path, _core_state_dict(core_seed))
    _save(bank_path, _bank_state_dict(primitive_ids, bank_seed))
    _save(router_path, _router_state_dict(primitive_ids, router_seed))
    _save(scorer_path, _scorer_state_dict(scorer_seed))
    _save(vocab_path, {"vocab_marker": torch.zeros(1)})

    core_raw, core_state = mb.canonical_state_hash_from_file(core_path)
    core_component = mb.ComponentManifest(
        component_id="core",
        file_path=str(core_path),
        file_sha256=core_raw,
        canonical_state_hash=core_state,
        schema_hash=_CORE_SCHEMA_HASH,
    )
    vocab_raw, vocab_state = mb.canonical_state_hash_from_file(vocab_path)
    vocab_component = mb.ComponentManifest(
        component_id="vocabulary",
        file_path=str(vocab_path),
        file_sha256=vocab_raw,
        canonical_state_hash=vocab_state,
        schema_hash=_CORE_SCHEMA_HASH,
    )

    bank_state_dict = mb._load_state_dict(bank_path)  # noqa: SLF001 -- test-internal reuse
    primitives = []
    for pid in primitive_ids:
        sliced = mb.primitive_state_dict(bank_state_dict, pid)
        weights_hash = mb.canonical_state_hash(sliced)
        primitives.append(
            mb.PrimitiveManifestEntry(
                physical_id=pid,
                operation_name=f"OP_{pid}",
                version="v1",
                architecture_signature="pointwise-v1",
                state_abi_hash="abi-v1",
                core_dependency_hash=core_state,
                decoder_dependency_hash="decoder-abi-v1",
                weights_hash=weights_hash,
                source_artifact=str(bank_path),
                provenance_status=provenance_status,
                argument_schema_hash=_ARG_SCHEMA_HASH if pid == primitive_ids[0] else None,
                training_receipt=training_receipt,
            )
        )

    router_state_dict = mb._load_state_dict(router_path)  # noqa: SLF001
    router = mb.RouterManifest(
        source_artifact=str(router_path),
        weights_hash=mb.canonical_state_hash(mb.router_non_key_state_dict(router_state_dict)),
        task_state_dependency_hash=core_state,
        key_to_primitive_mapping_hash=mb.router_key_to_primitive_mapping_hash(
            router_state_dict, list(primitive_ids)
        ),
    )

    scorer_state_dict = mb._load_state_dict(scorer_path)  # noqa: SLF001
    argument_scorer = mb.ArgumentScorerManifest(
        source_artifact=str(scorer_path),
        weights_hash=mb.canonical_state_hash(scorer_state_dict),
        input_dependency_hash=core_state,
        argument_schema_hash=_ARG_SCHEMA_HASH,
    )

    scoring_policy = mb.ScoringPolicyManifest(
        family_formula_version="dot_product_v1",
        argument_formula_version=argument_formula_version,
        lambda_weight=2.0,
        application_policy_version="l4_gated_v1",
        calibrated_pair_id="pair-1",
    )

    manifest = mb.build_manifest(
        schema_version=1,
        source_commit="deadbeef",
        runtime_recipe_version="rec002-v1",
        environment_record={"python": "3.12"},
        model_id=f"seed{model_seed}",
        model_seed=model_seed,
        training_run_id="run-1",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.RESTORE,
        scope=scope,
        requested_capabilities=requested_capabilities,
        publish_status=publish_status,
        core=core_component,
        vocabulary=vocab_component,
        primitives=tuple(primitives),
        router=router,
        argument_scorer=argument_scorer,
        scoring_policy=scoring_policy,
    )
    return Bundle(root, manifest)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_valid_bundle_loads_nominal(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    loaded = mb.load_bundle(
        bundle.manifest, mode="nominal", required_capabilities=frozenset({"nominal_execution"})
    )
    assert loaded.manifest.bundle_id == bundle.manifest.bundle_id
    assert set(loaded.primitive_state_dicts.keys()) == set(_PRIMITIVE_IDS)
    assert "core_integrity" in loaded.checks_performed
    assert "primitive_integrity_and_core_dependency" in loaded.checks_performed


def test_valid_bundle_loads_diagnostic(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    loaded = mb.load_bundle(bundle.manifest, mode="diagnostic")
    assert loaded.mode == "diagnostic"


def test_bundle_id_deterministic_for_identical_content(tmp_path: Path) -> None:
    b1 = build_valid_bundle(tmp_path / "a", model_seed=10)
    b2 = build_valid_bundle(
        tmp_path / "b", model_seed=10, core_seed=10, bank_seed=1, router_seed=2, scorer_seed=3
    )
    assert b1.manifest.content_manifest_digest == b2.manifest.content_manifest_digest
    assert b1.manifest.bundle_id == b2.manifest.bundle_id


def test_bundle_id_differs_for_different_seed_same_content_digest(tmp_path: Path) -> None:
    b1 = build_valid_bundle(tmp_path / "a", model_seed=10)
    b2 = build_valid_bundle(
        tmp_path / "b", model_seed=11, core_seed=10, bank_seed=1, router_seed=2, scorer_seed=3
    )
    assert b1.manifest.content_manifest_digest == b2.manifest.content_manifest_digest
    assert b1.manifest.bundle_id != b2.manifest.bundle_id


# ---------------------------------------------------------------------------
# 2. Core 1-tensor diff -> CORE_DEPENDENCY_MISMATCH
# ---------------------------------------------------------------------------


def test_core_tensor_mutation_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    core_path = Path(bundle.manifest.core.file_path)
    sd = torch.load(core_path, map_location="cpu", weights_only=True)
    sd["embed.weight"] = sd["embed.weight"] + 1.0
    torch.save(sd, core_path)
    with pytest.raises(mb.CoreDependencyMismatchError):
        mb.load_bundle(bundle.manifest, mode="nominal")


def test_stale_bank_trained_against_different_core_is_rejected(tmp_path: Path) -> None:
    """The exact ADR-0091 scenario: a bank whose recorded core_dependency_hash
    does not match the Core actually present in the bundle (R3-009 pretrained
    a new Core; the cached `primitive_bank_16.pt` was never retrained
    against it, but nothing checked that before combining them)."""
    bundle = build_valid_bundle(tmp_path)
    stale_primitives = tuple(
        dataclasses.replace(p, core_dependency_hash="stale-core-hash-from-before-r3-009")
        for p in bundle.manifest.primitives
    )
    tampered = _refreshed(bundle.manifest, primitives=stale_primitives)
    with pytest.raises(mb.CoreDependencyMismatchError):
        mb.load_bundle(tampered, mode="nominal")


# ---------------------------------------------------------------------------
# 3. Primitive missing (whole primitive, and 16-of-N count) -> MISSING_ARTIFACT / INCOMPLETE_BUNDLE
# ---------------------------------------------------------------------------


def test_primitive_entirely_absent_from_bank_file_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    bank_path = Path(bundle.manifest.primitives[0].source_artifact)
    sd = torch.load(bank_path, map_location="cpu", weights_only=True)
    missing_pid = bundle.manifest.primitives[0].physical_id
    for key in [k for k in sd if k.startswith(f"_primitives.{missing_pid}.")]:
        del sd[key]
    torch.save(sd, bank_path)
    with pytest.raises(mb.MissingArtifactError):
        mb.load_bundle(bundle.manifest, mode="nominal")


def test_declared_primitive_count_below_expected_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    with pytest.raises(mb.IncompleteBundleError):
        mb.load_bundle(
            bundle.manifest, mode="nominal", expected_primitive_count=len(_PRIMITIVE_IDS) + 1
        )


def test_partial_state_dict_key_removed_is_rejected(tmp_path: Path) -> None:
    """One primitive keeps its weight but loses its bias key -- a partial
    write must not be silently completed with a random-init substitute."""
    bundle = build_valid_bundle(tmp_path)
    bank_path = Path(bundle.manifest.primitives[0].source_artifact)
    sd = torch.load(bank_path, map_location="cpu", weights_only=True)
    pid = bundle.manifest.primitives[0].physical_id
    del sd[f"_primitives.{pid}.bias"]
    torch.save(sd, bank_path)
    with pytest.raises(mb.IncompleteBundleError):
        mb.load_bundle(bundle.manifest, mode="nominal")


# ---------------------------------------------------------------------------
# 4. Fake same seed -> distinct-model audit fails
# ---------------------------------------------------------------------------


def test_relabeled_seed_fails_distinct_model_audit(tmp_path: Path) -> None:
    b10 = build_valid_bundle(tmp_path / "seed10", model_seed=10)
    # Same exact core file (byte copy) relabeled as seed 11.
    b11 = build_valid_bundle(
        tmp_path / "seed11",
        model_seed=11,
        core_seed=10,  # identical core content -> identical canonical_state_hash
        bank_seed=99,
        router_seed=98,
        scorer_seed=97,
    )
    with pytest.raises(mb.DuplicateTrainingIdentityError):
        mb.assert_distinct_model_identities([b10.manifest, b11.manifest])


def test_genuinely_distinct_seeds_pass_audit(tmp_path: Path) -> None:
    manifests = [
        build_valid_bundle(tmp_path / f"seed{s}", model_seed=s).manifest
        for s in (10, 11, 12, 13, 14)
    ]
    audit = mb.assert_distinct_model_identities(manifests)
    assert audit.all_distinct
    assert audit.seeds_checked == (10, 11, 12, 13, 14)


# ---------------------------------------------------------------------------
# 5. Bank/router key order swap -> ROUTER_KEY_MAPPING_MISMATCH
# ---------------------------------------------------------------------------


def test_router_key_swap_between_primitives_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    router_path = Path(bundle.manifest.router.source_artifact)
    sd = torch.load(router_path, map_location="cpu", weights_only=True)
    pid_a, pid_b = _PRIMITIVE_IDS[0], _PRIMITIVE_IDS[1]
    key_a, key_b = sd[f"_keys.{pid_a}"].clone(), sd[f"_keys.{pid_b}"].clone()
    sd[f"_keys.{pid_a}"], sd[f"_keys.{pid_b}"] = key_b, key_a
    torch.save(sd, router_path)
    with pytest.raises(mb.RouterKeyMappingMismatchError):
        mb.load_bundle(bundle.manifest, mode="nominal")


# ---------------------------------------------------------------------------
# 6. Schema mismatch -> SCHEMA_MISMATCH
# ---------------------------------------------------------------------------


def test_vocab_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    bad_vocab = dataclasses.replace(
        bundle.manifest.vocabulary, schema_hash="a-different-vocab-schema"
    )
    # The declared file_sha256/canonical_state_hash still match the real
    # file on disk (only schema_hash, a cross-component declaration, is
    # wrong) -- isolates the schema-agreement check from file integrity.
    tampered = _refreshed(bundle.manifest, vocabulary=bad_vocab)
    with pytest.raises(mb.SchemaMismatchError):
        mb.load_bundle(tampered, mode="nominal")


# ---------------------------------------------------------------------------
# 7. Scorer pair mismatch -> UNCERTIFIED_PAIR
# ---------------------------------------------------------------------------


def test_uncertified_router_scorer_pair_is_rejected_nominal(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    empty_table = mb.CompatibilityTable(certified_pairs=())
    with pytest.raises(mb.UncertifiedPairError):
        mb.load_bundle(bundle.manifest, mode="nominal", compatibility_table=empty_table)


def test_certified_pair_allows_nominal_load(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    m = bundle.manifest
    table = mb.CompatibilityTable(
        certified_pairs=(
            mb.CertifiedPair(
                router_weights_hash=m.router.weights_hash,
                scorer_weights_hash=m.argument_scorer.weights_hash,
                lambda_weight=m.scoring_policy.lambda_weight,
                argument_schema_hash=m.argument_scorer.argument_schema_hash,
                application_policy_version=m.scoring_policy.application_policy_version,
                calibrated_pair_id=m.scoring_policy.calibrated_pair_id or "",
            ),
        )
    )
    loaded = mb.load_bundle(m, mode="nominal", compatibility_table=table)
    assert "pair_compatibility" in loaded.checks_performed


def test_diagnostic_mode_skips_pair_certification(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path, scope=mb.BundleScope.DIAGNOSTIC)
    empty_table = mb.CompatibilityTable(certified_pairs=())
    loaded = mb.load_bundle(bundle.manifest, mode="diagnostic", compatibility_table=empty_table)
    assert loaded.mode == "diagnostic"


# ---------------------------------------------------------------------------
# 8. Formula-only change -> EXECUTION_SIGNATURE_MISMATCH
# ---------------------------------------------------------------------------


def test_formula_only_change_alters_execution_signature(tmp_path: Path) -> None:
    old_bundle = build_valid_bundle(
        tmp_path / "old", argument_formula_version="select_softmax_v1_pre_adr0088"
    )
    new_bundle = build_valid_bundle(
        tmp_path / "new",
        model_seed=old_bundle.manifest.model_seed,
        core_seed=old_bundle.manifest.model_seed,
        bank_seed=1,
        router_seed=2,
        scorer_seed=3,
        argument_formula_version="select_sigmoid_v2_adr0088",
    )
    old_sig = mb.compute_execution_signature(old_bundle.manifest)
    new_sig = mb.compute_execution_signature(new_bundle.manifest)
    assert old_sig != new_sig
    # Loading the OLD manifest while the caller (live code) expects the NEW
    # formula version must fail explicitly.
    with pytest.raises(mb.ExecutionSignatureMismatchError):
        mb.load_bundle(old_bundle.manifest, mode="nominal", expected_execution_signature=new_sig)
    # Loading it with the matching expectation succeeds.
    loaded = mb.load_bundle(
        old_bundle.manifest, mode="nominal", expected_execution_signature=old_sig
    )
    assert "execution_signature" in loaded.checks_performed


# ---------------------------------------------------------------------------
# 9. Checkpoint corruption -> CORRUPTED_ARTIFACT
# ---------------------------------------------------------------------------


def test_corrupted_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    core_path = Path(bundle.manifest.core.file_path)
    core_path.write_bytes(b"not a valid torch checkpoint")
    with pytest.raises(mb.CorruptedArtifactError):
        mb.load_bundle(bundle.manifest, mode="nominal")


# ---------------------------------------------------------------------------
# 10. Staged / interrupted write -> INCOMPLETE_BUNDLE
# ---------------------------------------------------------------------------


def test_staging_bundle_blocks_nominal_load(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path, publish_status=mb.PublishStatus.STAGING)
    with pytest.raises(mb.IncompleteBundleError):
        mb.load_bundle(bundle.manifest, mode="nominal")


def test_staging_bundle_allows_diagnostic_load(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path, publish_status=mb.PublishStatus.STAGING)
    loaded = mb.load_bundle(bundle.manifest, mode="diagnostic")
    assert loaded.mode == "diagnostic"


# ---------------------------------------------------------------------------
# 11. UNKNOWN provenance -> UNKNOWN_PROVENANCE (nominal); allowed for audit
# ---------------------------------------------------------------------------


def test_unknown_provenance_blocks_nominal_load(tmp_path: Path) -> None:
    bundle = build_valid_bundle(
        tmp_path, provenance_status=mb.ProvenanceStatus.UNKNOWN, training_receipt=None
    )
    with pytest.raises(mb.UnknownProvenanceError):
        mb.load_bundle(bundle.manifest, mode="nominal")


def test_unknown_provenance_allows_diagnostic_load(tmp_path: Path) -> None:
    bundle = build_valid_bundle(
        tmp_path, provenance_status=mb.ProvenanceStatus.UNKNOWN, training_receipt=None
    )
    loaded = mb.load_bundle(bundle.manifest, mode="diagnostic")
    assert loaded.mode == "diagnostic"


def test_trained_claim_without_receipt_is_untrained_component(tmp_path: Path) -> None:
    bundle = build_valid_bundle(
        tmp_path, provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD, training_receipt=None
    )
    with pytest.raises(mb.UntrainedComponentError):
        mb.load_bundle(bundle.manifest, mode="nominal")


# ---------------------------------------------------------------------------
# 12. Capability insufficient -> CAPABILITY_NOT_QUALIFIED
# ---------------------------------------------------------------------------


def test_missing_required_capability_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path, requested_capabilities=frozenset({"diagnostic_only"}))
    with pytest.raises(mb.CapabilityNotQualifiedError):
        mb.load_bundle(
            bundle.manifest, mode="nominal", required_capabilities=frozenset({"nominal_execution"})
        )


def test_diagnostic_scope_bundle_refuses_nominal_mode(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path, scope=mb.BundleScope.DIAGNOSTIC)
    with pytest.raises(mb.CapabilityNotQualifiedError):
        mb.load_bundle(bundle.manifest, mode="nominal")


# ---------------------------------------------------------------------------
# 13. Manifest self-consistency (hand-tampered digest/bundle_id)
# ---------------------------------------------------------------------------


def test_hand_tampered_bundle_id_is_rejected(tmp_path: Path) -> None:
    bundle = build_valid_bundle(tmp_path)
    tampered = dataclasses.replace(bundle.manifest, bundle_id="not-the-real-bundle-id")
    with pytest.raises(mb.IncompleteBundleError):
        mb.load_bundle(tampered, mode="nominal")


# ---------------------------------------------------------------------------
# 14. Loader never calls a builder / trains / constructs an optimizer.
# ---------------------------------------------------------------------------

_FORBIDDEN_IDENTIFIERS = (
    "get_or_build",
    "_train_single_primitive",
    "update_router_incrementally",
    "torch.optim",
    "requires_grad_(True)",
    ".backward(",
)


def test_module_source_contains_no_builder_or_training_calls() -> None:
    source = inspect.getsource(mb)
    for identifier in _FORBIDDEN_IDENTIFIERS:
        assert identifier not in source, f"model_bundle.py must never reference {identifier!r}"


def test_load_bundle_never_constructs_an_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_valid_bundle(tmp_path)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_bundle must never construct an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "__init__", _forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", _forbidden)
    loaded = mb.load_bundle(bundle.manifest, mode="nominal")
    assert loaded.manifest.bundle_id == bundle.manifest.bundle_id


# ---------------------------------------------------------------------------
# 15. legacy_import: copy-only, hash-preserving, no auto-provenance-upgrade.
# ---------------------------------------------------------------------------


def test_legacy_import_preserves_bytes_and_hashes(tmp_path: Path) -> None:
    source_dir = tmp_path / "legacy_source"
    dest_dir = tmp_path / "new_namespace"
    source_path = source_dir / "shared_encoder.pt"
    _save(source_path, _core_state_dict(seed=42))

    component = mb.legacy_import(source_path, dest_dir, component_id="core")

    assert Path(component.file_path).parent == dest_dir
    assert component.file_sha256 == mb.raw_file_sha256(source_path)
    # Source file is untouched (copy, not a move or hard link).
    assert source_path.is_file()
    original_bytes = source_path.read_bytes()
    Path(component.file_path).write_bytes(original_bytes + b"\x00")
    assert source_path.read_bytes() == original_bytes


def test_legacy_import_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(mb.MissingArtifactError):
        mb.legacy_import(tmp_path / "does_not_exist.pt", tmp_path / "dest", component_id="core")


# ---------------------------------------------------------------------------
# 16. Hash helper properties: raw vs canonical, byte-copy vs re-serialization.
# ---------------------------------------------------------------------------


def test_byte_copy_matches_both_hashes(tmp_path: Path) -> None:
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    sd = _core_state_dict(seed=5)
    _save(path_a, sd)
    path_b.write_bytes(path_a.read_bytes())
    raw_a, state_a = mb.canonical_state_hash_from_file(path_a)
    raw_b, state_b = mb.canonical_state_hash_from_file(path_b)
    assert raw_a == raw_b
    assert state_a == state_b


def test_independently_trained_weights_match_neither_hash(tmp_path: Path) -> None:
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _save(path_a, _core_state_dict(seed=5))
    _save(path_b, _core_state_dict(seed=6))
    raw_a, state_a = mb.canonical_state_hash_from_file(path_a)
    raw_b, state_b = mb.canonical_state_hash_from_file(path_b)
    assert raw_a != raw_b
    assert state_a != state_b


def test_canonical_hash_independent_of_dict_insertion_order() -> None:
    sd = _core_state_dict(seed=7)
    sd["extra.weight"] = torch.randn(2, 2)
    reordered = dict(reversed(list(sd.items())))
    assert mb.canonical_state_hash(sd) == mb.canonical_state_hash(reordered)
