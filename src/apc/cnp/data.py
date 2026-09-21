"""Deterministic CNP v1 data generation and split auditing.

This module produces labeled records for explicitly requested roles.  It never
loads legacy APC bundles or sealed partitions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from apc.cnp.contracts import FEATURE_DIM, SelectArguments, SetState

DATA_ROOT_SEED = 610010
WORLD_SEED = 610000
KNOWN_LENGTHS = (1, 2, 4, 8, 16)
INTERPOLATION_LENGTHS = (3, 12, 32)
SOURCE_THRESHOLDS = (0.5, 0.8, 1.1)
INTERPOLATION_THRESHOLDS = (0.65, 0.95)
DataRole = Literal[
    "source_train",
    "dev_eval",
    "confirm_eval",
    "adapt_train",
    "replay",
    "shadow",
    "transfer_eval",
    "stress_eval",
]


def derive_seed(*parts: object) -> int:
    """Derive a process-independent uint64 seed from a complete record key."""

    payload = "cnp_v1|" + "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _generator(*parts: object) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(*parts))
    return generator


@dataclass(frozen=True)
class CNPWorld:
    """The evaluation-only metric matrix used to label CNP v1 records."""

    matrix: torch.Tensor

    def __post_init__(self) -> None:
        if self.matrix.dtype != torch.float32 or self.matrix.shape != (16, 16):
            raise ValueError("CNP world matrix must be float32[16,16]")
        if not torch.allclose(self.matrix, self.matrix.T):
            raise ValueError("CNP world matrix must be symmetric")


def make_world(seed: int = WORLD_SEED) -> CNPWorld:
    """Construct the fixed private metric used only by reference generation."""

    a = torch.randn((4, 16), generator=_generator("world", seed), dtype=torch.float32)
    raw = a.T @ a + 0.05 * torch.eye(16, dtype=torch.float32)
    matrix = 16.0 * raw / torch.trace(raw)
    return CNPWorld(matrix=matrix)


def phi(values: torch.Tensor) -> torch.Tensor:
    """Public, fixed 8D-to-16D feature map shared by CNP model and metric baseline."""

    if values.dtype != torch.float32 or values.shape[-1] != FEATURE_DIM:
        raise ValueError("phi expects float32 tensors with final dimension 8")
    return torch.cat((values, torch.sin(torch.pi * values)), dim=-1)


@dataclass(frozen=True)
class CNPRecord:
    """One labeled set; labels are intentionally outside :class:`SetState`."""

    state: SetState
    arguments: SelectArguments
    target: torch.Tensor
    role: DataRole
    condition_key: str

    def __post_init__(self) -> None:
        if self.target.dtype != torch.bool or self.target.shape != self.state.valid.shape:
            raise ValueError("target must be bool[B,N] matching state validity")
        if self.target.device != self.state.values.device:
            raise ValueError("target must be on the state device")
        self.arguments.validate_batch_size(self.state.batch_size)
        if torch.any(self.target & ~self.state.valid):
            raise ValueError("target may not select invalid padding")

    def digest(self) -> str:
        """Hash every observable record field used for split-overlap auditing."""

        payload = hashlib.sha256()
        for tensor in (
            self.state.values,
            self.state.valid,
            self.state.item_ids,
            self.arguments.query,
            self.arguments.threshold,
            self.target,
        ):
            payload.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        payload.update(self.role.encode("utf-8"))
        payload.update(self.condition_key.encode("utf-8"))
        return payload.hexdigest()


def make_record(
    *,
    role: DataRole,
    condition_key: str,
    length: int,
    threshold: float,
    example_index: int,
    query: torch.Tensor | None = None,
    world: CNPWorld | None = None,
) -> CNPRecord:
    """Make one deterministic B=1 record without label-conditioned rejection sampling."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if query is None:
        query = (
            torch.rand(
                (1, FEATURE_DIM),
                generator=_generator(DATA_ROOT_SEED, role, condition_key, "query"),
                dtype=torch.float32,
            )
            - 0.5
        )
    if query.dtype != torch.float32 or query.shape != (1, FEATURE_DIM):
        raise ValueError("query must be float32[1,8]")
    values = (
        2.0
        * torch.rand(
            (1, length, FEATURE_DIM),
            generator=_generator(
                DATA_ROOT_SEED, role, condition_key, length, threshold, example_index
            ),
            dtype=torch.float32,
        )
        - 1.0
    )
    valid = torch.ones((1, length), dtype=torch.bool)
    item_ids = torch.arange(length, dtype=torch.int64).unsqueeze(0)
    state = SetState(values=values, valid=valid, item_ids=item_ids)
    arguments = SelectArguments(
        query=query,
        threshold=torch.tensor([threshold], dtype=torch.float32),
    )
    from apc.cnp.reference import reference_select

    target = reference_select(state, arguments, world or make_world()).selected
    return CNPRecord(state, arguments, target, role, condition_key)


def audit_split_disjoint(records_by_role: dict[str, Iterable[CNPRecord]]) -> dict[str, object]:
    """Fail closed if record digests cross a non-exempt split boundary."""

    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    role_counts: dict[str, int] = {}
    for role, records in records_by_role.items():
        role_counts[role] = 0
        for record in records:
            role_counts[role] += 1
            digest = record.digest()
            previous = seen.setdefault(digest, role)
            if previous != role:
                duplicates.append({"digest": digest, "first_role": previous, "second_role": role})
    if duplicates:
        raise ValueError(f"CNP split overlap detected: {duplicates[:3]}")
    return {"status": "PASS", "role_counts": role_counts, "unique_records": len(seen)}


def records_manifest(records: Iterable[CNPRecord]) -> dict[str, object]:
    """Summarize generated records without exposing the private world matrix."""

    materialized = list(records)
    valid = sum(int(record.state.valid.sum()) for record in materialized)
    positive = sum(int(record.target.sum()) for record in materialized)
    empty = sum(int(not record.target.any()) for record in materialized)
    digest = hashlib.sha256(
        "".join(record.digest() for record in materialized).encode("utf-8")
    ).hexdigest()
    return {
        "records": len(materialized),
        "valid_items": valid,
        "positive_items": positive,
        "positive_rate": None if valid == 0 else positive / valid,
        "empty_rate": None if not materialized else empty / len(materialized),
        "records_sha256": digest,
    }


def canonical_json_hash(payload: object) -> str:
    """Hash a serializable CNP config or manifest deterministically."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
