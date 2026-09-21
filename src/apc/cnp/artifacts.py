"""Fail-closed CNP v1 artifact creation and loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from apc.cnp.data import canonical_json_hash
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.utils.model_bundle import canonical_state_hash

BUNDLE_SCHEMA = "cnp_bundle_v1"


class CNPArtifactError(ValueError):
    """Raised for incomplete, overwritten, or hash-inconsistent CNP artifacts."""


@dataclass(frozen=True)
class CNPBundleManifest:
    """Self-contained continuous primitive manifest, separate from token bundles."""

    schema_version: str
    family: str
    architecture_signature: str
    model_seed: int
    config_hash: str
    base_weights_hash: str
    adapter_weights_hash: str | None
    usage_conditions: dict[str, Any]
    parent_bundle: str | None
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_for_prefix(
    primitive: ConditionalSelectPrimitive, prefix: str, include: bool
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in primitive.state_dict().items()
        if name.startswith(prefix) is include
    }


def build_manifest(
    primitive: ConditionalSelectPrimitive,
    *,
    model_seed: int,
    config: dict[str, Any],
    usage_conditions: dict[str, Any],
    parent_bundle: str | None = None,
) -> CNPBundleManifest:
    """Build hash evidence for one primitive without embedding private world state."""

    base = _state_for_prefix(primitive, "adapter.", include=False)
    adapter = _state_for_prefix(primitive, "adapter.", include=True)
    return CNPBundleManifest(
        schema_version=BUNDLE_SCHEMA,
        family="CONDITIONAL_SELECT",
        architecture_signature=primitive.architecture_signature,
        model_seed=model_seed,
        config_hash=canonical_json_hash(config),
        base_weights_hash=canonical_state_hash(base),
        adapter_weights_hash=canonical_state_hash(adapter) if adapter else None,
        usage_conditions=dict(usage_conditions),
        parent_bundle=parent_bundle,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def save_bundle(
    directory: Path,
    primitive: ConditionalSelectPrimitive,
    manifest: CNPBundleManifest,
) -> Path:
    """Create a new bundle directory once; never overwrite existing evidence."""

    if directory.exists():
        raise CNPArtifactError(f"Refusing to overwrite existing CNP bundle directory: {directory}")
    directory.mkdir(parents=True)
    weights_path = directory / "primitive.pt"
    manifest_path = directory / "manifest.json"
    torch.save(primitive.state_dict(), weights_path)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_bundle(
    directory: Path,
    primitive: ConditionalSelectPrimitive,
    *,
    expected_config: dict[str, Any],
) -> CNPBundleManifest:
    """Load exact CNP weights after schema, config, architecture, and hashes validate."""

    manifest_path = directory / "manifest.json"
    weights_path = directory / "primitive.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise CNPArtifactError(f"CNP bundle is incomplete: {directory}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = CNPBundleManifest(**raw)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise CNPArtifactError(f"Invalid CNP manifest: {error}") from error
    if manifest.schema_version != BUNDLE_SCHEMA:
        raise CNPArtifactError("CNP bundle schema mismatch")
    if manifest.architecture_signature != primitive.architecture_signature:
        raise CNPArtifactError("CNP bundle architecture mismatch")
    if manifest.config_hash != canonical_json_hash(expected_config):
        raise CNPArtifactError("CNP bundle configuration hash mismatch")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise CNPArtifactError(f"Cannot read CNP weights: {error}") from error
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise CNPArtifactError("CNP weights must be a tensor state dictionary")
    base = {name: tensor for name, tensor in state.items() if not name.startswith("adapter.")}
    adapter = {name: tensor for name, tensor in state.items() if name.startswith("adapter.")}
    if canonical_state_hash(base) != manifest.base_weights_hash:
        raise CNPArtifactError("CNP base weights hash mismatch")
    if (canonical_state_hash(adapter) if adapter else None) != manifest.adapter_weights_hash:
        raise CNPArtifactError("CNP adapter weights hash mismatch")
    primitive.load_state_dict(state, strict=True)
    return manifest


def source_hash(path: Path) -> str:
    """Hash a tracked config or source file for a run manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
