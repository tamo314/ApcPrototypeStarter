"""Fail-closed seed registry for the independent CNP v1 program."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs/cnp/v1.json"

# These are the model/data seeds already reserved by Phase B/C/D documents and
# registries.  CNP v1 must not silently consume them.
FORBIDDEN_MODEL_SEEDS = frozenset(
    (*range(0, 5), *range(10, 25), *range(30, 35), *range(40, 45), *range(50, 55))
)
FORBIDDEN_DATA_SEEDS = frozenset(
    (*range(101, 106), *range(201, 221), *range(301, 306), *range(401, 406))
)


class SeedRegistryAuditError(ValueError):
    """Raised when a proposed CNP seed is duplicated or belongs to an old registry."""


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SeedRegistryAuditError(f"Cannot read CNP registry {path}: {error}") from error
    if data.get("schema_version") != 1:
        raise SeedRegistryAuditError("Unsupported or missing CNP v1 seed registry schema_version")
    return data


def audit_cnp_seed_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, object]:
    """Statically validate the CNP v1 proposed world, data, and model seed namespaces."""

    data = _read_registry(path)
    seeds = data.get("seeds")
    if not isinstance(seeds, dict):
        raise SeedRegistryAuditError("CNP registry must define a seeds object")
    world = seeds.get("world")
    data_seed = seeds.get("data")
    development = seeds.get("development_model")
    confirmation = seeds.get("confirmation_model")
    if not isinstance(world, int) or not isinstance(data_seed, int):
        raise SeedRegistryAuditError("CNP world and data seeds must be integers")
    if not isinstance(development, list) or not isinstance(confirmation, list):
        raise SeedRegistryAuditError("CNP model seed lists must be lists")
    model_seeds = [*development, *confirmation]
    if not all(isinstance(seed, int) for seed in model_seeds):
        raise SeedRegistryAuditError("CNP model seed lists may contain integers only")
    if len(model_seeds) != len(set(model_seeds)):
        raise SeedRegistryAuditError("CNP model seed lists overlap")
    collisions = sorted(set(model_seeds) & FORBIDDEN_MODEL_SEEDS)
    if collisions:
        raise SeedRegistryAuditError(
            f"CNP model seeds collide with existing registry: {collisions}"
        )
    if data_seed in FORBIDDEN_DATA_SEEDS:
        raise SeedRegistryAuditError(f"CNP data seed collides with existing registry: {data_seed}")
    all_seeds = [world, data_seed, *model_seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise SeedRegistryAuditError("CNP world/data/model roles must use distinct seeds")
    return {
        "status": "PASS",
        "registry": str(path),
        "world_seed": world,
        "data_seed": data_seed,
        "development_model_seeds": development,
        "confirmation_model_seeds": confirmation,
        "legacy_sealed_access": 0,
        "forbidden_model_seed_count": len(FORBIDDEN_MODEL_SEEDS),
        "forbidden_data_seed_count": len(FORBIDDEN_DATA_SEEDS),
    }
