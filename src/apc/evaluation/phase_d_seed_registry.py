"""Static, fail-closed audit for Phase D's pre-result model-seed registry.

This module reads only Python source ASTs and tracked JSON provenance records.  It
does not import experiment modules, construct a model, generate examples, or read
run artifacts, so it is safe to use before any Phase D execution authorization is
acted on.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "docs/phase_d/PHASE_D_D005_SEED_REGISTRY.json"


class SeedRegistryAuditError(ValueError):
    """Raised when the frozen D-005 registry no longer matches its evidence."""


def _literal_assignment(source_path: Path, symbol: str) -> tuple[int, ...]:
    """Return a top-level literal seed assignment without importing its module."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id != symbol or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except ValueError as exc:
            raise SeedRegistryAuditError(
                f"{source_path}: {symbol} is not a literal seed registration"
            ) from exc
        if not isinstance(literal, (tuple, list, set)) or not all(
            isinstance(seed, int) for seed in literal
        ):
            raise SeedRegistryAuditError(
                f"{source_path}: {symbol} is not an integer seed collection"
            )
        return tuple(sorted(literal))
    raise SeedRegistryAuditError(f"Missing source seed registration {symbol} in {source_path}")


def _load_registry(registry_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedRegistryAuditError(f"Cannot read frozen registry {registry_path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise SeedRegistryAuditError("Unsupported or missing D-005 seed registry schema_version")
    return data


def audit_phase_d_seed_registry(registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Mechanically verify D-005's source registrations and historical provenance."""
    registry = _load_registry(registry_path)
    candidate = tuple(registry["candidate_cohort"]["model_seeds"])
    if candidate != tuple(sorted(candidate)) or len(candidate) != 5 or len(set(candidate)) != 5:
        raise SeedRegistryAuditError(
            "Candidate cohort must contain five sorted, unique model seeds"
        )

    forbidden: set[int] = set()
    checked_sources: list[str] = []
    for entry in registry["source_seed_registrations"]:
        registered = _literal_assignment(REPO_ROOT / entry["source"], entry["symbol"])
        expected = tuple(entry["seeds"])
        if registered != expected:
            raise SeedRegistryAuditError(
                f"{entry['id']}: registry has {expected}, source has {registered}"
            )
        forbidden.update(registered)
        checked_sources.append(f"{entry['source']}:{entry['symbol']}")

    collision = sorted(set(candidate) & forbidden)
    if collision:
        raise SeedRegistryAuditError(
            f"Candidate cohort collides with reserved model seeds {collision}"
        )

    checked_provenance: list[str] = []
    historical_bundle_seeds: set[int] = set()
    historical_data_seeds: set[int] = set()
    for entry in registry["historical_run_provenance"]:
        source = REPO_ROOT / entry["source"]
        try:
            record = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SeedRegistryAuditError(f"Cannot read provenance {source}: {exc}") from exc
        for field in ("bundle_seeds_evaluated", "data_seeds_evaluated"):
            observed = tuple(record.get(field, ()))
            expected = tuple(entry[field])
            if observed != expected:
                raise SeedRegistryAuditError(
                    f"{entry['id']}: {field} has {observed}, expected {expected}"
                )
        historical_bundle_seeds.update(entry["bundle_seeds_evaluated"])
        historical_data_seeds.update(entry["data_seeds_evaluated"])
        checked_provenance.append(entry["source"])

    if set(candidate) & historical_bundle_seeds:
        raise SeedRegistryAuditError("Candidate cohort overlaps historical bundle model seeds")
    if set(candidate) & historical_data_seeds:
        raise SeedRegistryAuditError("Candidate cohort overlaps historical data seed values")

    return {
        "status": "PASS",
        "candidate_cohort_id": registry["candidate_cohort"]["cohort_id"],
        "candidate_model_seeds": list(candidate),
        "forbidden_model_seeds": sorted(forbidden),
        "historical_bundle_model_seeds": sorted(historical_bundle_seeds),
        "historical_data_seeds": sorted(historical_data_seeds),
        "checked_source_registrations": checked_sources,
        "checked_historical_provenance": checked_provenance,
        "sealed_access": 0,
    }
