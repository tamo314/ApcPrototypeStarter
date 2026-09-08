"""Immutable ModelBundle contract: manifest, hashing, and a fail-closed loader.

Task B-C005REC-002. Implements the manifest/hash/loader contract proposed by
`docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md` in response to
ADR-0091 (R3-010): a stale `primitive_bank_16.pt` silently combined with a
new, incoherent Core because nothing checked that the bank was trained
against *this* Core before loading it. This module exists so that
incoherent combination is a mechanical, typed load failure instead of a
silent wrong answer.

Hard invariants (do not weaken these to make a caller's life easier):
  * This module never trains, calibrates, selects a variant, or constructs
    a builder/optimizer. It only reads files, hashes their content, and
    compares those hashes to what a manifest declares.
  * `load_bundle` is fail-closed: any missing artifact, hash mismatch,
    unknown provenance, uncertified pair, incomplete bundle, or missing
    capability raises a typed `ModelBundleError` subclass. There is no
    partial `state_dict` load, no missing-key random fill, no same-seed or
    latest-mtime fallback, and no silent downgrade of a nominal request to
    a diagnostic one.
  * Two independent hashes are kept for every component: `file_sha256`
    (raw bytes -- confirms "this exact file") and `canonical_state_hash`
    (tensor name/shape/dtype/content, independent of pickling metadata --
    confirms "these exact trained weights", and is what cross-component
    dependency checks like `core_dependency_hash` compare against). A byte
    copy of a checkpoint matches both; two independently trained
    checkpoints match neither; a re-serialized copy of the same weights
    matches only the canonical one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch

__all__ = [
    "BuildRoute",
    "BundleScope",
    "PublishStatus",
    "ProvenanceStatus",
    "ComponentManifest",
    "PrimitiveManifestEntry",
    "RouterManifest",
    "ArgumentScorerManifest",
    "ScoringPolicyManifest",
    "ModelBundleManifest",
    "LoadedModelBundle",
    "CertifiedPair",
    "CompatibilityTable",
    "DistinctModelAudit",
    "ModelBundleError",
    "MissingArtifactError",
    "CorruptedArtifactError",
    "CoreDependencyMismatchError",
    "SchemaMismatchError",
    "ArgumentSchemaMismatchError",
    "RouterKeyMappingMismatchError",
    "ExecutionSignatureMismatchError",
    "UntrainedComponentError",
    "UnknownProvenanceError",
    "UncertifiedPairError",
    "IncompleteBundleError",
    "CapabilityNotQualifiedError",
    "DuplicateTrainingIdentityError",
    "raw_file_sha256",
    "canonical_state_hash",
    "compute_state_abi_hash",
    "canonical_state_hash_from_file",
    "load_state_dict",
    "primitive_state_dict",
    "router_non_key_state_dict",
    "router_key_to_primitive_mapping_hash",
    "compute_content_manifest_digest",
    "compute_bundle_id",
    "compute_execution_signature",
    "build_manifest",
    "legacy_import",
    "load_bundle",
    "audit_distinct_model_identities",
    "assert_distinct_model_identities",
]


# ---------------------------------------------------------------------------
# 1. Enums
# ---------------------------------------------------------------------------


class BuildRoute(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    RESTORE = "RESTORE"
    PARTIAL_BUILD = "PARTIAL_BUILD"
    CLEAN_BUILD = "CLEAN_BUILD"


class BundleScope(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    DIAGNOSTIC = "diagnostic"
    NOMINAL = "nominal"


class PublishStatus(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    STAGING = "STAGING"
    PUBLISHED = "PUBLISHED"


class ProvenanceStatus(str, Enum):  # noqa: UP042 (StrEnum needs Python >= 3.11)
    TRAINED_THIS_BUILD = "TRAINED_THIS_BUILD"
    RESTORED_VALIDATED = "RESTORED_VALIDATED"
    EXPLICIT_PARAMETER_FREE_APPROVED = "EXPLICIT_PARAMETER_FREE_APPROVED"
    LEGACY_IMPORTED = "LEGACY_IMPORTED"
    LEGACY_REQUALIFIED_ON_RECOVERY_FIXTURE = "LEGACY_REQUALIFIED_ON_RECOVERY_FIXTURE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# 2. Typed errors -- the loader's only failure surface. `label` is the
#    machine-readable identifier recorded in loader_contract_tests.json.
# ---------------------------------------------------------------------------


class ModelBundleError(Exception):
    label: str = "MODEL_BUNDLE_ERROR"


class MissingArtifactError(ModelBundleError):
    label = "MISSING_ARTIFACT"


class CorruptedArtifactError(ModelBundleError):
    label = "CORRUPTED_ARTIFACT"


class CoreDependencyMismatchError(ModelBundleError):
    label = "CORE_DEPENDENCY_MISMATCH"


class SchemaMismatchError(ModelBundleError):
    label = "SCHEMA_MISMATCH"


class ArgumentSchemaMismatchError(ModelBundleError):
    label = "ARGUMENT_SCHEMA_MISMATCH"


class RouterKeyMappingMismatchError(ModelBundleError):
    label = "ROUTER_KEY_MAPPING_MISMATCH"


class ExecutionSignatureMismatchError(ModelBundleError):
    label = "EXECUTION_SIGNATURE_MISMATCH"


class UntrainedComponentError(ModelBundleError):
    label = "UNTRAINED_COMPONENT"


class UnknownProvenanceError(ModelBundleError):
    label = "UNKNOWN_PROVENANCE"


class UncertifiedPairError(ModelBundleError):
    label = "UNCERTIFIED_PAIR"


class IncompleteBundleError(ModelBundleError):
    label = "INCOMPLETE_BUNDLE"


class CapabilityNotQualifiedError(ModelBundleError):
    label = "CAPABILITY_NOT_QUALIFIED"


class DuplicateTrainingIdentityError(ModelBundleError):
    label = "DUPLICATE_TRAINING_IDENTITY"


# ---------------------------------------------------------------------------
# 3. Hashing. `weights_only=True` restricts `torch.load` to plain tensors --
#    the same convention every existing checkpoint writer in this repo
#    already uses (`torch.save(module.state_dict(), path)`), and it refuses
#    to unpickle arbitrary objects.
# ---------------------------------------------------------------------------


def raw_file_sha256(path: Path) -> str:
    """Hash of the file's exact bytes. Confirms "this exact file"."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_state_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Content hash of tensor name/shape/dtype/deterministic byte order.

    Independent of file/pickle metadata: two files with different
    `raw_file_sha256` but the same `canonical_state_hash` hold the same
    trained weights (e.g. re-saved with a different torch/pickle version);
    two independently trained checkpoints match neither hash.
    """
    hasher = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def compute_state_abi_hash(
    state_dict: Mapping[str, torch.Tensor], *, architecture_signature: str
) -> str:
    """Structural (name/shape/dtype) hash of a primitive's parameter ABI,
    tagged with its declared `architecture_signature` -- deliberately
    independent of tensor *content* (that is `canonical_state_hash`'s job).
    Two state dicts with the same keys/shapes/dtypes under the SAME
    `architecture_signature` hash identically regardless of trained values;
    a state dict missing/extra keys (e.g. a bias-carrying primitive's slice
    with its bias tensors silently dropped), or declared under a different
    `architecture_signature` string, never collides with either.

    Existing bundle producers set `state_abi_hash = weights_hash`, which
    carries no ABI information of its own today (`load_bundle`'s
    `structural_completeness` check only verifies it is non-empty). This
    function exists so a new architecture (B-C005REC-004D's
    `cross_position_length_bias_v1`) can record a real ABI fingerprint
    instead, without touching any of `load_bundle`'s existing checks.
    """
    hasher = hashlib.sha256()
    hasher.update(architecture_signature.encode("utf-8"))
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
    return hasher.hexdigest()


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 -- any load failure is a corrupted artifact
        raise CorruptedArtifactError(f"{path}: failed to load as a checkpoint ({exc})") from exc
    if not isinstance(obj, Mapping):
        raise CorruptedArtifactError(
            f"{path}: expected a state_dict mapping, got {type(obj).__name__}"
        )
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            raise CorruptedArtifactError(
                f"{path}: state_dict entry {key!r} is not a tensor ({type(value).__name__})"
            )
    return obj


def load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Public, validated `state_dict` load: raises `MissingArtifactError` /
    `CorruptedArtifactError` rather than letting a raw I/O or unpickle
    exception escape. Callers that need to inspect real tensor content
    while *building* a manifest (never while loading one -- `load_bundle`
    does its own internal loading) should use this instead of reaching
    into a private helper."""
    if not path.is_file():
        raise MissingArtifactError(f"{path}: file does not exist")
    return _load_state_dict(path)


def canonical_state_hash_from_file(path: Path) -> tuple[str, str]:
    """Returns `(raw_file_sha256, canonical_state_hash)` for a checkpoint
    file holding a plain `state_dict` (this repo's universal convention).
    Raises `MissingArtifactError` / `CorruptedArtifactError` on failure --
    never returns a placeholder hash for a file that could not be read."""
    if not path.is_file():
        raise MissingArtifactError(f"{path}: file does not exist")
    raw_hash = raw_file_sha256(path)
    state_dict = _load_state_dict(path)
    return raw_hash, canonical_state_hash(state_dict)


_PRIMITIVE_KEY_PREFIX = "_primitives."
_ROUTER_KEY_PREFIX = "_keys."


def primitive_state_dict(
    bank_state_dict: Mapping[str, torch.Tensor], primitive_id: int
) -> dict[str, torch.Tensor]:
    """Slice one primitive's own parameters out of a full bank `state_dict`
    (bank.py's `PrimitiveBank` stores primitives in an `nn.ModuleDict` keyed
    by `str(primitive_id)`, so its state_dict keys are prefixed
    `_primitives.<id>.<param>`)."""
    prefix = f"{_PRIMITIVE_KEY_PREFIX}{primitive_id}."
    return {
        key[len(prefix) :]: value
        for key, value in bank_state_dict.items()
        if key.startswith(prefix)
    }


def router_non_key_state_dict(
    router_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The router's shared parameters (e.g. `query_proj.*`), excluding the
    per-primitive `_keys.<id>` entries. `RouterManifest.weights_hash`
    covers only this subset -- the per-id keys are tracked separately by
    `key_to_primitive_mapping_hash` so that a corrupted/swapped key is
    distinguishable from a change to the router's shared parameters."""
    return {k: v for k, v in router_state_dict.items() if not k.startswith(_ROUTER_KEY_PREFIX)}


def router_key_to_primitive_mapping_hash(
    router_state_dict: Mapping[str, torch.Tensor], primitive_ids: Sequence[int]
) -> str:
    """Content-addressed hash of `{primitive_id: that id's key vector}`.

    Router.py stores keys in an `nn.ParameterDict` keyed by `str(id)`
    (state_dict keys `_keys.<id>`). This hash is over *which vector belongs
    to which id*, not storage order -- reordering a dict changes nothing,
    but swapping two ids' key vectors (a real routing-table corruption)
    changes the hash for both ids.
    """
    hasher = hashlib.sha256()
    for primitive_id in sorted(primitive_ids):
        key = f"{_ROUTER_KEY_PREFIX}{primitive_id}"
        if key not in router_state_dict:
            raise MissingArtifactError(
                f"router state_dict missing key for primitive id {primitive_id}"
            )
        tensor = router_state_dict[key]
        hasher.update(str(primitive_id).encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 4. Manifest dataclasses (docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md
#    section 2's recommended fields).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentManifest:
    component_id: str
    file_path: str
    file_sha256: str
    canonical_state_hash: str
    schema_hash: str = ""


@dataclass(frozen=True)
class PrimitiveManifestEntry:
    physical_id: int
    operation_name: str
    version: str
    architecture_signature: str
    state_abi_hash: str
    core_dependency_hash: str
    decoder_dependency_hash: str
    weights_hash: str
    source_artifact: str
    provenance_status: ProvenanceStatus
    argument_schema_hash: str | None = None
    training_receipt: str | None = None


@dataclass(frozen=True)
class RouterManifest:
    source_artifact: str
    weights_hash: str
    task_state_dependency_hash: str
    key_to_primitive_mapping_hash: str


@dataclass(frozen=True)
class ArgumentScorerManifest:
    source_artifact: str
    weights_hash: str
    input_dependency_hash: str
    argument_schema_hash: str


@dataclass(frozen=True)
class ScoringPolicyManifest:
    family_formula_version: str
    argument_formula_version: str
    lambda_weight: float
    application_policy_version: str
    calibrated_pair_id: str | None = None


@dataclass(frozen=True)
class ModelBundleManifest:
    schema_version: int
    bundle_id: str
    content_manifest_digest: str
    source_commit: str
    runtime_recipe_version: str
    environment_record: Mapping[str, str]
    model_id: str
    model_seed: int
    training_run_id: str
    parent_bundle_ids: tuple[str, ...]
    build_route: BuildRoute
    scope: BundleScope
    requested_capabilities: frozenset[str]
    publish_status: PublishStatus
    core: ComponentManifest
    vocabulary: ComponentManifest
    primitives: tuple[PrimitiveManifestEntry, ...]
    router: RouterManifest
    argument_scorer: ArgumentScorerManifest
    scoring_policy: ScoringPolicyManifest
    task_encoder: ComponentManifest | None = None
    query_projection: ComponentManifest | None = None
    decoder: ComponentManifest | None = None
    controller_verifier_signature: str | None = None
    build_recipe_hash: str | None = None
    dataset_role_hashes: Mapping[str, str] = field(default_factory=dict)
    generator_version: str | None = None
    known_defects: tuple[str, ...] = ()
    exposure_manifest: Mapping[str, Any] = field(default_factory=dict)
    clean_build_exercised_stages: tuple[str, ...] = ()
    qualification_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedModelBundle:
    """Result of a successful `load_bundle` call: the checked manifest plus
    the raw state dicts it was verified against. Reconstructing live
    `Core`/`PrimitiveBank`/`Router`/`ArgumentScorer` objects from these is a
    runtime-injection concern (B-C005REC-006), not this loader's job."""

    manifest: ModelBundleManifest
    core_state_dict: Mapping[str, torch.Tensor]
    primitive_state_dicts: Mapping[int, Mapping[str, torch.Tensor]]
    router_state_dict: Mapping[str, torch.Tensor]
    argument_scorer_state_dict: Mapping[str, torch.Tensor]
    mode: Literal["diagnostic", "nominal"]
    checks_performed: tuple[str, ...]


@dataclass(frozen=True)
class CertifiedPair:
    """One router/scorer/policy combination that was actually
    trained/validated together (e.g. an R3-006/008 repair recipe output)."""

    router_weights_hash: str
    scorer_weights_hash: str
    lambda_weight: float
    argument_schema_hash: str
    application_policy_version: str
    calibrated_pair_id: str


@dataclass(frozen=True)
class CompatibilityTable:
    certified_pairs: tuple[CertifiedPair, ...] = ()

    def find(self, manifest: ModelBundleManifest) -> CertifiedPair | None:
        for pair in self.certified_pairs:
            if (
                pair.router_weights_hash == manifest.router.weights_hash
                and pair.scorer_weights_hash == manifest.argument_scorer.weights_hash
                and pair.lambda_weight == manifest.scoring_policy.lambda_weight
                and pair.argument_schema_hash == manifest.argument_scorer.argument_schema_hash
                and pair.application_policy_version
                == manifest.scoring_policy.application_policy_version
            ):
                return pair
        return None


@dataclass(frozen=True)
class DistinctModelAudit:
    seeds_checked: tuple[int, ...]
    duplicate_pairs: tuple[tuple[int, int], ...]
    all_distinct: bool


# ---------------------------------------------------------------------------
# 5. bundle_id / content_manifest_digest / execution signature
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, frozenset):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _component_digest_payload(component: ComponentManifest | None) -> dict[str, Any] | None:
    """Content-identity fields only -- deliberately excludes `file_path`:
    a namespace copy of the same bytes at a different path is the same
    content, not a different bundle."""
    if component is None:
        return None
    return {
        "component_id": component.component_id,
        "file_sha256": component.file_sha256,
        "canonical_state_hash": component.canonical_state_hash,
        "schema_hash": component.schema_hash,
    }


def _primitive_digest_payload(entry: PrimitiveManifestEntry) -> dict[str, Any]:
    return {
        "physical_id": entry.physical_id,
        "operation_name": entry.operation_name,
        "version": entry.version,
        "architecture_signature": entry.architecture_signature,
        "state_abi_hash": entry.state_abi_hash,
        "core_dependency_hash": entry.core_dependency_hash,
        "decoder_dependency_hash": entry.decoder_dependency_hash,
        "weights_hash": entry.weights_hash,
        "provenance_status": entry.provenance_status,
        "argument_schema_hash": entry.argument_schema_hash,
        # `source_artifact` (a filesystem location) and `training_receipt`
        # (free-form audit text) are deliberately excluded: neither is part
        # of the trained-content identity this digest exists to capture.
    }


def _router_digest_payload(router: RouterManifest) -> dict[str, Any]:
    return {
        "weights_hash": router.weights_hash,
        "task_state_dependency_hash": router.task_state_dependency_hash,
        "key_to_primitive_mapping_hash": router.key_to_primitive_mapping_hash,
    }


def _argument_scorer_digest_payload(scorer: ArgumentScorerManifest) -> dict[str, Any]:
    return {
        "weights_hash": scorer.weights_hash,
        "input_dependency_hash": scorer.input_dependency_hash,
        "argument_schema_hash": scorer.argument_schema_hash,
    }


def compute_content_manifest_digest(manifest: ModelBundleManifest) -> str:
    """Hash over component content/dependency/execution-policy fields only
    -- excludes `bundle_id`/`content_manifest_digest` themselves (no
    self-reference), excludes filesystem-location fields (`file_path`/
    `source_artifact`: a namespace copy is the same content), and excludes
    identity/lineage fields (those belong to `compute_bundle_id`, so two
    bundles with identical content but different declared model_seed still
    get different bundle_id's, which is what makes the "relabeled seed"
    case detectable downstream)."""
    payload = {
        "schema_version": manifest.schema_version,
        "core": _component_digest_payload(manifest.core),
        "task_encoder": _component_digest_payload(manifest.task_encoder),
        "query_projection": _component_digest_payload(manifest.query_projection),
        "decoder": _component_digest_payload(manifest.decoder),
        "vocabulary": _component_digest_payload(manifest.vocabulary),
        "primitives": [
            _primitive_digest_payload(p)
            for p in sorted(manifest.primitives, key=lambda p: p.physical_id)
        ],
        "router": _router_digest_payload(manifest.router),
        "argument_scorer": _argument_scorer_digest_payload(manifest.argument_scorer),
        "scoring_policy": manifest.scoring_policy,
        "controller_verifier_signature": manifest.controller_verifier_signature,
        "build_recipe_hash": manifest.build_recipe_hash,
        "dataset_role_hashes": dict(manifest.dataset_role_hashes),
        "generator_version": manifest.generator_version,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_bundle_id(manifest: ModelBundleManifest, *, content_manifest_digest: str) -> str:
    payload = {
        "content_manifest_digest": content_manifest_digest,
        "model_id": manifest.model_id,
        "model_seed": manifest.model_seed,
        "training_run_id": manifest.training_run_id,
        "parent_bundle_ids": list(manifest.parent_bundle_ids),
        "build_route": manifest.build_route,
        "scope": manifest.scope,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_execution_signature(manifest: ModelBundleManifest) -> str:
    """Formula/schema/policy-sensitive signature, deliberately *separate*
    from `argument_scorer.weights_hash`: a formula-only change (e.g.
    ADR-0088's SELECT softmax->sigmoid readout) changes this signature even
    though the underlying trained weights (and their hash) are untouched."""
    payload = {
        "argument_scorer_weights_hash": manifest.argument_scorer.weights_hash,
        "argument_schema_hash": manifest.argument_scorer.argument_schema_hash,
        "family_formula_version": manifest.scoring_policy.family_formula_version,
        "argument_formula_version": manifest.scoring_policy.argument_formula_version,
        "application_policy_version": manifest.scoring_policy.application_policy_version,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_manifest(**kwargs: Any) -> ModelBundleManifest:
    """Factory that fills in `content_manifest_digest`/`bundle_id` from the
    rest of the fields, so callers never hand-compute (and risk
    desynchronizing) the self-referential-looking identity fields."""
    draft = ModelBundleManifest(bundle_id="", content_manifest_digest="", **kwargs)
    digest = compute_content_manifest_digest(draft)
    bundle_id = compute_bundle_id(draft, content_manifest_digest=digest)
    return dataclasses.replace(draft, content_manifest_digest=digest, bundle_id=bundle_id)


def _verify_manifest_self_consistency(manifest: ModelBundleManifest) -> None:
    expected_digest = compute_content_manifest_digest(manifest)
    if expected_digest != manifest.content_manifest_digest:
        raise IncompleteBundleError(
            "manifest.content_manifest_digest does not match its own declared "
            "fields -- manifest was hand-edited or corrupted after construction"
        )
    expected_bundle_id = compute_bundle_id(manifest, content_manifest_digest=expected_digest)
    if expected_bundle_id != manifest.bundle_id:
        raise IncompleteBundleError(
            "manifest.bundle_id does not match its own declared identity fields"
        )


# ---------------------------------------------------------------------------
# 6. Legacy import: copy-only, provenance-preserving.
# ---------------------------------------------------------------------------


def legacy_import(source_path: Path, dest_dir: Path, *, component_id: str) -> ComponentManifest:
    """Copy `source_path` byte-for-byte into `dest_dir` (never a hard link:
    mutating the legacy source must not silently change the imported
    bundle) and record its hashes. Does NOT set a provenance status of
    "trained" or "clean exposure" -- callers must record the resulting
    `ComponentManifest`'s hashes under whatever `ProvenanceStatus` the
    surrounding audit actually established (typically `LEGACY_IMPORTED`)."""
    if not source_path.is_file():
        raise MissingArtifactError(f"{source_path}: legacy source file does not exist")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    raw_hash, state_hash = canonical_state_hash_from_file(dest_path)
    source_raw_hash = raw_file_sha256(source_path)
    if source_raw_hash != raw_hash:
        raise CorruptedArtifactError(
            f"{source_path}: copy to {dest_path} did not preserve file bytes"
        )
    return ComponentManifest(
        component_id=component_id,
        file_path=str(dest_path),
        file_sha256=raw_hash,
        canonical_state_hash=state_hash,
    )


# ---------------------------------------------------------------------------
# 7. The fail-closed loader.
# ---------------------------------------------------------------------------


def _verify_component_integrity(
    component: ComponentManifest | None, *, label: str, mismatch_error: type[ModelBundleError]
) -> Mapping[str, torch.Tensor] | None:
    if component is None:
        return None
    path = Path(component.file_path)
    raw_hash, state_hash = canonical_state_hash_from_file(path)
    if raw_hash != component.file_sha256 or state_hash != component.canonical_state_hash:
        raise mismatch_error(
            f"{label} ({path}): recomputed hashes (raw={raw_hash}, canonical={state_hash}) "
            f"do not match manifest-declared hashes "
            f"(raw={component.file_sha256}, canonical={component.canonical_state_hash})"
        )
    return _load_state_dict(path)


def load_bundle(
    manifest: ModelBundleManifest,
    *,
    mode: Literal["diagnostic", "nominal"],
    required_capabilities: frozenset[str] = frozenset(),
    expected_primitive_count: int | None = None,
    expected_execution_signature: str | None = None,
    compatibility_table: CompatibilityTable | None = None,
) -> LoadedModelBundle:
    """Load-and-verify a bundle from its manifest. Read-only: never trains,
    calibrates, selects a variant, or searches for a `latest`/best-guess
    path -- every field consulted below is exactly what `manifest`
    declares, and every mismatch is a typed `ModelBundleError`, never a
    silent substitution.
    """
    checks: list[str] = []

    _verify_manifest_self_consistency(manifest)
    checks.append("manifest_self_consistency")

    # -- capability / scope gating --------------------------------------
    if mode == "nominal" and manifest.scope is BundleScope.DIAGNOSTIC:
        raise CapabilityNotQualifiedError(
            "bundle scope is 'diagnostic'; nominal execution was requested"
        )
    if not required_capabilities.issubset(manifest.requested_capabilities):
        missing = required_capabilities - manifest.requested_capabilities
        raise CapabilityNotQualifiedError(
            f"bundle does not qualify capabilities: {sorted(missing)}"
        )
    checks.append("capability_scope")

    # -- publish status ---------------------------------------------------
    if manifest.publish_status is not PublishStatus.PUBLISHED:
        if mode == "nominal":
            raise IncompleteBundleError(
                f"bundle publish_status={manifest.publish_status.value}; "
                "nominal load requires PUBLISHED"
            )
    checks.append("publish_status")

    # -- structural completeness ------------------------------------------
    if expected_primitive_count is not None and len(manifest.primitives) < expected_primitive_count:
        raise IncompleteBundleError(
            f"manifest declares {len(manifest.primitives)} primitives, "
            f"expected {expected_primitive_count}"
        )
    for entry in manifest.primitives:
        if not entry.weights_hash or not entry.core_dependency_hash or not entry.state_abi_hash:
            raise IncompleteBundleError(
                f"primitive {entry.physical_id} ({entry.operation_name}) has an empty "
                "required hash field -- partial manifest entries are not loadable"
            )
    checks.append("structural_completeness")

    # -- provenance ---------------------------------------------------------
    if mode == "nominal":
        unknown = [
            e.physical_id
            for e in manifest.primitives
            if e.provenance_status is ProvenanceStatus.UNKNOWN
        ]
        if unknown:
            raise UnknownProvenanceError(
                f"primitives with UNKNOWN provenance cannot be nominal-loaded: {unknown}"
            )
        trained_statuses = (
            ProvenanceStatus.TRAINED_THIS_BUILD,
            ProvenanceStatus.RESTORED_VALIDATED,
        )
        untrained = [
            e.physical_id
            for e in manifest.primitives
            if e.provenance_status in trained_statuses and not e.training_receipt
        ]
        if untrained:
            raise UntrainedComponentError(
                f"primitives claim {ProvenanceStatus.TRAINED_THIS_BUILD.value}/"
                f"{ProvenanceStatus.RESTORED_VALIDATED.value} provenance but carry no "
                f"training_receipt: {untrained}"
            )
    checks.append("provenance")

    # -- core integrity -----------------------------------------------------
    core_state_dict = _verify_component_integrity(
        manifest.core, label="core", mismatch_error=CoreDependencyMismatchError
    )
    assert core_state_dict is not None
    checks.append("core_integrity")

    # -- decoder / vocabulary integrity + cross-component schema agreement --
    _verify_component_integrity(
        manifest.decoder, label="decoder", mismatch_error=CoreDependencyMismatchError
    )
    vocab_state_dict = _verify_component_integrity(
        manifest.vocabulary, label="vocabulary", mismatch_error=CoreDependencyMismatchError
    )
    if vocab_state_dict is None:
        raise MissingArtifactError("manifest.vocabulary is required")
    if manifest.vocabulary.schema_hash != manifest.core.schema_hash:
        raise SchemaMismatchError(
            f"vocabulary.schema_hash={manifest.vocabulary.schema_hash} does not match "
            f"core.schema_hash={manifest.core.schema_hash} -- vocab/token schema disagrees "
            "with the Core this bundle declares"
        )
    if manifest.decoder is not None and manifest.decoder.schema_hash != manifest.core.schema_hash:
        raise SchemaMismatchError(
            f"decoder.schema_hash={manifest.decoder.schema_hash} does not match "
            f"core.schema_hash={manifest.core.schema_hash}"
        )
    checks.append("schema_integrity")

    # -- primitive integrity + core-dependency check -------------------------
    core_dep_hash = manifest.core.canonical_state_hash
    primitive_state_dicts: dict[int, Mapping[str, torch.Tensor]] = {}
    for entry in manifest.primitives:
        source_path = Path(entry.source_artifact)
        if not source_path.is_file():
            raise MissingArtifactError(
                f"primitive {entry.physical_id} ({entry.operation_name}): "
                f"source_artifact {source_path} does not exist"
            )
        bank_state_dict = _load_state_dict(source_path)
        sliced = primitive_state_dict(bank_state_dict, entry.physical_id)
        if not sliced:
            raise MissingArtifactError(
                f"primitive {entry.physical_id} ({entry.operation_name}): "
                f"no state_dict entries found under prefix "
                f"'{_PRIMITIVE_KEY_PREFIX}{entry.physical_id}.' in {source_path}"
            )
        recomputed_weights_hash = canonical_state_hash(sliced)
        if recomputed_weights_hash != entry.weights_hash:
            raise IncompleteBundleError(
                f"primitive {entry.physical_id} ({entry.operation_name}): recomputed "
                f"weights_hash does not match manifest -- partial or corrupted "
                f"state_dict slice"
            )
        # This is the check that would have caught ADR-0091's bug directly:
        # a primitive whose *recorded* training dependency does not match
        # the Core actually present in this bundle.
        if entry.core_dependency_hash != core_dep_hash:
            raise CoreDependencyMismatchError(
                f"primitive {entry.physical_id} ({entry.operation_name}): "
                f"core_dependency_hash={entry.core_dependency_hash} does not match "
                f"this bundle's core canonical_state_hash={core_dep_hash} -- these "
                "weights were trained against a different Core"
            )
        if (
            manifest.decoder is not None
            and entry.decoder_dependency_hash != manifest.decoder.canonical_state_hash
        ):
            raise CoreDependencyMismatchError(
                f"primitive {entry.physical_id} ({entry.operation_name}): "
                f"decoder_dependency_hash={entry.decoder_dependency_hash} does not match "
                f"this bundle's decoder canonical_state_hash="
                f"{manifest.decoder.canonical_state_hash}"
            )
        primitive_state_dicts[entry.physical_id] = sliced
    checks.append("primitive_integrity_and_core_dependency")

    # -- argument schema consistency (parameterized primitives only) --------
    for entry in manifest.primitives:
        if entry.argument_schema_hash is None:
            continue
        if entry.argument_schema_hash != manifest.argument_scorer.argument_schema_hash:
            raise ArgumentSchemaMismatchError(
                f"primitive {entry.physical_id} ({entry.operation_name}): "
                f"argument_schema_hash={entry.argument_schema_hash} does not match "
                f"argument_scorer.argument_schema_hash={manifest.argument_scorer.argument_schema_hash}"
            )
    checks.append("argument_schema_consistency")

    # -- router integrity + core dependency + key/primitive mapping ---------
    router_path = Path(manifest.router.source_artifact)
    if not router_path.is_file():
        raise MissingArtifactError(f"router: source_artifact {router_path} does not exist")
    router_state_dict = _load_state_dict(router_path)
    router_state_hash = canonical_state_hash(router_non_key_state_dict(router_state_dict))
    if router_state_hash != manifest.router.weights_hash:
        raise CoreDependencyMismatchError(
            f"router ({router_path}): recomputed canonical_state_hash={router_state_hash} "
            f"does not match manifest-declared weights_hash={manifest.router.weights_hash}"
        )
    if manifest.router.task_state_dependency_hash != core_dep_hash:
        raise CoreDependencyMismatchError(
            f"router: task_state_dependency_hash={manifest.router.task_state_dependency_hash} "
            f"does not match this bundle's core canonical_state_hash={core_dep_hash}"
        )
    primitive_ids = [entry.physical_id for entry in manifest.primitives]
    recomputed_mapping_hash = router_key_to_primitive_mapping_hash(router_state_dict, primitive_ids)
    if recomputed_mapping_hash != manifest.router.key_to_primitive_mapping_hash:
        raise RouterKeyMappingMismatchError(
            f"router: recomputed key_to_primitive_mapping_hash={recomputed_mapping_hash} "
            f"does not match manifest-declared "
            f"key_to_primitive_mapping_hash={manifest.router.key_to_primitive_mapping_hash} "
            "-- a primitive's routing key does not match what the manifest recorded "
            "for that id (e.g. two ids' keys were swapped)"
        )
    checks.append("router_integrity")

    # -- argument scorer integrity + core dependency -------------------------
    scorer_path = Path(manifest.argument_scorer.source_artifact)
    if not scorer_path.is_file():
        raise MissingArtifactError(f"argument_scorer: source_artifact {scorer_path} does not exist")
    argument_scorer_state_dict = _load_state_dict(scorer_path)
    scorer_state_hash = canonical_state_hash(argument_scorer_state_dict)
    if scorer_state_hash != manifest.argument_scorer.weights_hash:
        raise CoreDependencyMismatchError(
            f"argument_scorer ({scorer_path}): recomputed canonical_state_hash="
            f"{scorer_state_hash} does not match manifest-declared weights_hash="
            f"{manifest.argument_scorer.weights_hash}"
        )
    if manifest.argument_scorer.input_dependency_hash != core_dep_hash:
        raise CoreDependencyMismatchError(
            f"argument_scorer: input_dependency_hash="
            f"{manifest.argument_scorer.input_dependency_hash} does not match this "
            f"bundle's core canonical_state_hash={core_dep_hash}"
        )
    checks.append("argument_scorer_integrity")

    # -- compatibility (router/scorer/lambda/argument-schema/policy pair) ----
    if compatibility_table is not None and mode == "nominal":
        certified = compatibility_table.find(manifest)
        if certified is None:
            raise UncertifiedPairError(
                "router/scorer/lambda/argument-schema/application-policy combination "
                "is not in the compatibility table -- refusing nominal load of an "
                "uncertified pair"
            )
        if manifest.scoring_policy.calibrated_pair_id != certified.calibrated_pair_id:
            raise UncertifiedPairError(
                "manifest.scoring_policy.calibrated_pair_id does not match the "
                "compatibility table's calibrated_pair_id for this combination"
            )
    checks.append("pair_compatibility")

    # -- execution signature (formula-only changes) ---------------------------
    if expected_execution_signature is not None:
        live_signature = compute_execution_signature(manifest)
        if live_signature != expected_execution_signature:
            raise ExecutionSignatureMismatchError(
                f"manifest execution signature {live_signature} does not match "
                f"the caller's expected (live-code) signature {expected_execution_signature} "
                "-- a formula/schema/policy version differs even though weights_hash may match"
            )
    checks.append("execution_signature")

    return LoadedModelBundle(
        manifest=manifest,
        core_state_dict=core_state_dict,
        primitive_state_dicts=primitive_state_dicts,
        router_state_dict=router_state_dict,
        argument_scorer_state_dict=argument_scorer_state_dict,
        mode=mode,
        checks_performed=tuple(checks),
    )


# ---------------------------------------------------------------------------
# 8. Cross-bundle identity audit.
# ---------------------------------------------------------------------------


def audit_distinct_model_identities(manifests: Sequence[ModelBundleManifest]) -> DistinctModelAudit:
    """A bundle's declared `model_seed` is just a label -- this checks it
    against the one thing that cannot lie: the Core's own trained-weight
    content hash. Two manifests with different `model_seed` but the same
    `core.canonical_state_hash` are the same training run relabeled, not
    two independent models (`docs/design-docs/..._CONTRACT.md` section 6:
    "同じCoreのcopy5個を独立modelと数えない")."""
    duplicates: list[tuple[int, int]] = []
    for i in range(len(manifests)):
        for j in range(i + 1, len(manifests)):
            a, b = manifests[i], manifests[j]
            same_core = a.core.canonical_state_hash == b.core.canonical_state_hash
            if a.model_seed != b.model_seed and same_core:
                duplicates.append((a.model_seed, b.model_seed))
    return DistinctModelAudit(
        seeds_checked=tuple(m.model_seed for m in manifests),
        duplicate_pairs=tuple(duplicates),
        all_distinct=not duplicates,
    )


def assert_distinct_model_identities(
    manifests: Sequence[ModelBundleManifest],
) -> DistinctModelAudit:
    audit = audit_distinct_model_identities(manifests)
    if not audit.all_distinct:
        raise DuplicateTrainingIdentityError(
            f"seed pairs {audit.duplicate_pairs} share an identical core "
            "canonical_state_hash despite different declared model_seed -- "
            "not independent training identities"
        )
    return audit
