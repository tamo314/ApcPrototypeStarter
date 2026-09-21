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
ADAPTATION_BLOCKS = ("q1+q2+", "q1+q2-", "q1-q2+", "q1-q2-")
DataRole = Literal[
    "source_train",
    "dev_eval",
    "confirm_eval",
    "confirm_v2_eval",
    "adapt_train",
    "replay",
    "shadow",
    "transfer_eval",
    "stress_eval",
    "repair_adapt_train",
    "repair_new_shadow",
    "repair_old_shadow",
    "cnp_repair_schedule_v1_new_train",
    "cnp_repair_schedule_v1_new_shadow",
    "cnp_repair_schedule_v1_old_shadow",
    "cnp_repair_retention_v1_new_train",
    "cnp_repair_retention_v1_new_shadow",
    "cnp_repair_retention_v1_old_shadow",
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
    schedule_index: int | None = None

    def __post_init__(self) -> None:
        if self.target.dtype != torch.bool or self.target.shape != self.state.valid.shape:
            raise ValueError("target must be bool[B,N] matching state validity")
        if self.target.device != self.state.values.device:
            raise ValueError("target must be on the state device")
        self.arguments.validate_batch_size(self.state.batch_size)
        if torch.any(self.target & ~self.state.valid):
            raise ValueError("target may not select invalid padding")
        if self.schedule_index is not None and self.schedule_index < 0:
            raise ValueError("schedule_index must be non-negative when supplied")

    def digest(self) -> str:
        """Hash model-visible inputs and labels for split-overlap auditing.

        The role and condition identifier intentionally do not participate: an
        identical record relabelled as another split must still be rejected.
        """

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
        return payload.hexdigest()

    def input_digest(self) -> str:
        """Hash only model-visible inputs, never the reference target.

        Schedule construction may use this stable identifier.  Keeping the target
        outside the digest makes label changes incapable of changing exposure.
        """

        payload = hashlib.sha256()
        for tensor in (
            self.state.values,
            self.state.valid,
            self.state.item_ids,
            self.arguments.query,
            self.arguments.threshold,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            payload.update(str(contiguous.dtype).encode("ascii"))
            payload.update(str(tuple(contiguous.shape)).encode("ascii"))
            payload.update(contiguous.numpy().tobytes())
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
        query = fixed_query(role, condition_key)
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


def fixed_query(role: DataRole, condition_key: str) -> torch.Tensor:
    """Return one deterministic, role-separated query in the initial condition domain."""

    return (
        torch.rand(
            (1, FEATURE_DIM),
            generator=_generator(DATA_ROOT_SEED, role, condition_key, "query"),
            dtype=torch.float32,
        )
        - 0.5
    )


def fixed_adaptation_query(role: DataRole, block: str, exemplar_index: int) -> torch.Tensor:
    """Return one deterministic query in the registered CNP-004 condition block."""

    if block not in ADAPTATION_BLOCKS:
        raise ValueError(f"unsupported CNP adaptation block: {block}")
    if exemplar_index < 0:
        raise ValueError("adaptation exemplar index must be non-negative")
    generator = _generator(DATA_ROOT_SEED, role, block, exemplar_index, "query")
    query = torch.rand((1, FEATURE_DIM), generator=generator, dtype=torch.float32) - 0.5
    magnitudes = 0.5 + 0.5 * torch.rand((2,), generator=generator, dtype=torch.float32)
    signs = torch.tensor(
        (1.0 if block[2] == "+" else -1.0, 1.0 if block[5] == "+" else -1.0),
        dtype=torch.float32,
    )
    query[0, :2] = magnitudes * signs
    return query


def make_adaptation_record(
    *,
    role: DataRole,
    block: str,
    exemplar_index: int,
    length: int,
    threshold: float,
    example_index: int,
    world: CNPWorld | None = None,
) -> CNPRecord:
    """Generate one CNP-004 record with a public block query and role-separated content."""

    return make_record(
        role=role,
        condition_key=f"{block}_{exemplar_index:02d}",
        length=length,
        threshold=threshold,
        example_index=example_index,
        query=fixed_adaptation_query(role, block, exemplar_index),
        world=world,
    )


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
