"""Reproducible data protocol for the post-R-CNP-001 diagnostics.

The historical CNP generators retain their v1 roots.  This module owns the
new roots and role namespace so a later authorised diagnostic cannot silently
reuse an opened panel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from apc.cnp.contracts import FEATURE_DIM, SelectArguments, SetState
from apc.cnp.data import SOURCE_THRESHOLDS, CNPRecord, CNPWorld, make_world
from apc.cnp.reference import reference_select

ProtocolKind = Literal["schedule", "retention", "alignment", "decision"]
PanelKind = Literal["new_train", "new_shadow", "old_shadow"]

NEW_DOMAIN = "q1+q2+"
OLD_DOMAIN = "source"
QUERY_COUNT = 8
SHADOW_SETS_PER_CELL = 128
TRAIN_SETS_PER_STRATUM = 64
LENGTHS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class RepairRoots:
    """The independently registered roots for one diagnostic family."""

    data: int
    schedule: int
    bootstrap: int


ROOTS: dict[ProtocolKind, RepairRoots] = {
    "schedule": RepairRoots(data=620010, schedule=620020, bootstrap=620030),
    "retention": RepairRoots(data=620110, schedule=620120, bootstrap=620130),
    "alignment": RepairRoots(data=620210, schedule=620220, bootstrap=620230),
    "decision": RepairRoots(data=620310, schedule=620320, bootstrap=620330),
}


def role_name(protocol: ProtocolKind, panel: PanelKind) -> str:
    """Return the registered role namespace for one generated panel."""

    return f"cnp_repair_{protocol}_v1_{panel}"


def _canonical(parts: tuple[object, ...]) -> bytes:
    return json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def derive_protocol_seed(
    *, root: int, role: str, query_id: int, length: int, threshold: float, example_index: int
) -> int:
    """Derive the registered uint64 record seed from a canonical JSON tuple."""

    if root < 0 or query_id < 0 or length < 0 or example_index < 0:
        raise ValueError("protocol seed components must be non-negative")
    parts = ("cnp_repair_v1", root, role, query_id, length, f"{threshold:.2f}", example_index)
    return int.from_bytes(hashlib.sha256(_canonical(parts)).digest()[:8], "big")


def derive_query_seed(*, root: int, role: str, query_id: int) -> int:
    """Derive a role-separated query seed without cell or target information."""

    parts = ("cnp_repair_v1", root, role, query_id, "query")
    return int.from_bytes(hashlib.sha256(_canonical(parts)).digest()[:8], "big")


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _condition(domain: str, query_id: int) -> str:
    return f"domain={domain}|query={query_id:02d}"


def parse_condition(condition: str) -> tuple[str, str]:
    """Decode a protocol condition identifier into explicit domain and query IDs."""

    parts = dict(item.split("=", 1) for item in condition.split("|") if "=" in item)
    domain = parts.get("domain")
    query = parts.get("query")
    if not domain or query is None:
        raise ValueError(f"not a repair protocol condition: {condition!r}")
    return domain, query


def _query(*, root: int, role: str, domain: str, query_id: int) -> torch.Tensor:
    generator = _generator(derive_query_seed(root=root, role=role, query_id=query_id))
    query = torch.rand((1, FEATURE_DIM), generator=generator, dtype=torch.float32) - 0.5
    if domain == NEW_DOMAIN:
        magnitudes = 0.5 + 0.5 * torch.rand((2,), generator=generator, dtype=torch.float32)
        query[0, :2] = magnitudes
    return query


def make_protocol_record(
    *,
    protocol: ProtocolKind,
    panel: PanelKind,
    domain: str,
    query_id: int,
    length: int,
    threshold: float,
    example_index: int,
    world: CNPWorld | None = None,
) -> CNPRecord:
    """Build one root-explicit record without target-conditioned sampling."""

    if domain not in {NEW_DOMAIN, OLD_DOMAIN}:
        raise ValueError(f"unknown repair protocol domain: {domain}")
    if length not in LENGTHS:
        raise ValueError(f"unsupported protocol length: {length}")
    if threshold not in SOURCE_THRESHOLDS:
        raise ValueError(f"unsupported protocol threshold: {threshold}")
    if not 0 <= query_id < QUERY_COUNT:
        raise ValueError("query_id is outside the registered range")
    roots = ROOTS[protocol]
    role = role_name(protocol, panel)
    generator = _generator(
        derive_protocol_seed(
            root=roots.data,
            role=role,
            query_id=query_id,
            length=length,
            threshold=threshold,
            example_index=example_index,
        )
    )
    values = 2.0 * torch.rand((1, length, FEATURE_DIM), generator=generator) - 1.0
    state = SetState(
        values=values,
        valid=torch.ones((1, length), dtype=torch.bool),
        item_ids=torch.arange(length, dtype=torch.int64).unsqueeze(0),
    )
    arguments = SelectArguments(
        query=_query(root=roots.data, role=role, domain=domain, query_id=query_id),
        threshold=torch.tensor([threshold], dtype=torch.float32),
    )
    target = reference_select(state, arguments, world or make_world()).selected
    return CNPRecord(
        state=state,
        arguments=arguments,
        target=target,
        role=role,  # type: ignore[arg-type]
        condition_key=_condition(domain, query_id),
        schedule_index=example_index,
    )


def _train_length(example_index: int) -> int:
    """Preserve the 13/13/13/13/12 train length allocation."""

    if not 0 <= example_index < TRAIN_SETS_PER_STRATUM:
        raise ValueError("train example index is outside the registered range")
    return LENGTHS[min(example_index // 13, len(LENGTHS) - 1)]


def make_protocol_panel(
    *, protocol: ProtocolKind, panel: PanelKind, world: CNPWorld | None = None
) -> list[CNPRecord]:
    """Generate a complete registered train or 120-cell shadow panel."""

    domain = NEW_DOMAIN if panel != "old_shadow" else OLD_DOMAIN
    if panel == "new_train":
        return [
            make_protocol_record(
                protocol=protocol,
                panel=panel,
                domain=domain,
                query_id=query_id,
                length=_train_length(example_index),
                threshold=threshold,
                example_index=example_index,
                world=world,
            )
            for query_id in range(QUERY_COUNT)
            for threshold in SOURCE_THRESHOLDS
            for example_index in range(TRAIN_SETS_PER_STRATUM)
        ]
    return [
        make_protocol_record(
            protocol=protocol,
            panel=panel,
            domain=domain,
            query_id=query_id,
            length=length,
            threshold=threshold,
            example_index=example_index,
            world=world,
        )
        for query_id in range(QUERY_COUNT)
        for length in LENGTHS
        for threshold in SOURCE_THRESHOLDS
        for example_index in range(SHADOW_SETS_PER_CELL)
    ]


def protocol_manifest(records: Iterable[CNPRecord]) -> dict[str, object]:
    """Produce split-audit material for a generated protocol panel."""

    materialized = list(records)
    return {
        "records": len(materialized),
        "input_ids": sorted(record.input_digest() for record in materialized),
        "record_ids": sorted(record.digest() for record in materialized),
        "query_ids": sorted(
            {
                record.arguments.query.detach().cpu().numpy().tobytes().hex()
                for record in materialized
            }
        ),
    }


def audit_protocol_disjoint(records_by_role: dict[str, Iterable[CNPRecord]]) -> dict[str, object]:
    """Reject shared model-visible inputs or queries across protocol roles."""

    inputs: dict[str, str] = {}
    queries: dict[str, str] = {}
    counts: dict[str, int] = {}
    overlaps: list[dict[str, str]] = []
    for role, records in records_by_role.items():
        counts[role] = 0
        for record in records:
            counts[role] += 1
            input_id = record.input_digest()
            query_id = record.arguments.query.detach().cpu().numpy().tobytes().hex()
            previous_input = inputs.setdefault(input_id, role)
            previous_query = queries.setdefault(query_id, role)
            if previous_input != role:
                overlaps.append(
                    {"kind": "input", "first_role": previous_input, "second_role": role}
                )
            if previous_query != role:
                overlaps.append(
                    {"kind": "query", "first_role": previous_query, "second_role": role}
                )
    if overlaps:
        raise ValueError(f"repair protocol split overlap detected: {overlaps[:3]}")
    return {"status": "PASS", "role_counts": counts, "unique_inputs": len(inputs)}
