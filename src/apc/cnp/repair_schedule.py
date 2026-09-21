"""Target-independent schedules for the R-CNP-001S order diagnostic."""

from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from apc.cnp.data import CNPRecord

Side = Literal["new", "replay"]


class ScheduleArm(StrEnum):
    """The pre-registered R-CNP-001S schedule arms."""

    ORDERED = "ORDERED"
    DISPERSED_MATCHED = "DISPERSED_MATCHED"
    DISPERSED_BALANCED = "DISPERSED_BALANCED"


@dataclass(frozen=True)
class ScheduledRecord:
    """A serializable scheduled input selected before any model execution."""

    side: Side
    input_id: str
    stratum_id: str
    position: int


@dataclass(frozen=True)
class SideSchedule:
    """One 4,096-record side of a diagnostic training schedule."""

    arm: ScheduleArm
    side: Side
    entries: tuple[ScheduledRecord, ...]
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or len(self.entries) % self.batch_size:
            raise ValueError("schedule entries must form complete batches")
        if any(entry.side != self.side for entry in self.entries):
            raise ValueError("side schedule contains an entry from another side")
        if [entry.position for entry in self.entries] != list(range(len(self.entries))):
            raise ValueError("schedule positions must be contiguous")

    @property
    def batches(self) -> tuple[tuple[ScheduledRecord, ...], ...]:
        return tuple(
            tuple(self.entries[index : index + self.batch_size])
            for index in range(0, len(self.entries), self.batch_size)
        )

    def quotas(self) -> Counter[str]:
        return Counter(entry.input_id for entry in self.entries)

    def sha256(self) -> str:
        payload = [
            {"side": entry.side, "input_id": entry.input_id, "stratum_id": entry.stratum_id}
            for entry in self.entries
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def json_lines(self) -> str:
        return (
            "\n".join(
                json.dumps(
                    {
                        "position": entry.position,
                        "side": entry.side,
                        "input_id": entry.input_id,
                        "stratum_id": entry.stratum_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for entry in self.entries
            )
            + "\n"
        )


def record_stratum(record: CNPRecord) -> str:
    """Return the schedule stratum from observable condition and threshold only."""

    return f"{record.condition_key}|threshold={float(record.arguments.threshold[0]):.2f}"


def _hash(parts: tuple[object, ...]) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compare_positions(left: tuple[int, int, str, str], right: tuple[int, int, str, str]) -> int:
    """Compare rational positions by cross multiplication; never use floats."""

    left_numerator, left_denominator, left_tie, left_final = left
    right_numerator, right_denominator, right_tie, right_final = right
    cross_left = left_numerator * right_denominator
    cross_right = right_numerator * left_denominator
    if cross_left != cross_right:
        return -1 if cross_left < cross_right else 1
    if left_tie != right_tie:
        return -1 if left_tie < right_tie else 1
    if left_final != right_final:
        return -1 if left_final < right_final else 1
    return 0


def _compare_occurrences(
    left: tuple[int, int, str, str, int], right: tuple[int, int, str, str, int]
) -> int:
    return _compare_positions(left[:4], right[:4])


def _compare_events(
    left: tuple[int, int, str, str, int], right: tuple[int, int, str, str, int]
) -> int:
    return _compare_positions(
        (left[0], left[1], left[3], f"{left[2]}/{left[4]}"),
        (right[0], right[1], right[3], f"{right[2]}/{right[4]}"),
    )


def _input_map(records: list[CNPRecord]) -> dict[str, CNPRecord]:
    result = {record.input_digest(): record for record in records}
    if len(result) != len(records):
        raise ValueError("schedule records must have unique target-free input IDs")
    return result


def _registered_order(records: list[CNPRecord]) -> list[CNPRecord]:
    """Recover the registered order without trusting the caller's list order."""

    return sorted(
        records,
        key=lambda record: (
            record.condition_key,
            f"{float(record.arguments.threshold[0]):.2f}",
            record.schedule_index if record.schedule_index is not None else 2**63,
            record.input_digest(),
        ),
    )


def ordered_schedule(records: list[CNPRecord], *, side: Side, steps: int = 256) -> SideSchedule:
    """Reproduce the registered sequential-index baseline exactly."""

    if not records:
        raise ValueError("cannot schedule an empty record list")
    ordered = _registered_order(records)
    _input_map(ordered)
    entries = tuple(
        ScheduledRecord(
            side=side,
            input_id=ordered[(step * 16 + offset) % len(ordered)].input_digest(),
            stratum_id=record_stratum(ordered[(step * 16 + offset) % len(ordered)]),
            position=step * 16 + offset,
        )
        for step in range(steps)
        for offset in range(16)
    )
    return SideSchedule(arm=ScheduleArm.ORDERED, side=side, entries=entries)


def balanced_quotas(
    records: list[CNPRecord], *, side: Side, root: int, total: int = 4096
) -> Counter[str]:
    """Allocate near-equal stratum and record exposure from target-free IDs."""

    record_by_id = _input_map(records)
    groups: dict[str, list[str]] = defaultdict(list)
    for input_id, record in record_by_id.items():
        groups[record_stratum(record)].append(input_id)
    if not groups or total < len(records):
        raise ValueError("balanced schedule needs at least one exposure per record")
    base, remainder = divmod(total, len(groups))
    ranked_strata = sorted(groups, key=lambda stratum: (_hash((root, side, stratum)), stratum))
    quotas: Counter[str] = Counter()
    for rank, stratum in enumerate(ranked_strata):
        stratum_total = base + (rank < remainder)
        members = sorted(groups[stratum])
        per_record, member_remainder = divmod(stratum_total, len(members))
        ranked_members = sorted(
            members,
            key=lambda input_id: (_hash((root, side, stratum, input_id)), input_id),
        )
        for member_rank, input_id in enumerate(ranked_members):
            quotas[input_id] = per_record + (member_rank < member_remainder)
    if sum(quotas.values()) != total or set(quotas) != set(record_by_id):
        raise RuntimeError("balanced quota allocation is incomplete")
    return quotas


def dispersed_schedule(
    records: list[CNPRecord],
    *,
    arm: ScheduleArm,
    side: Side,
    root: int,
    quotas: Counter[str],
) -> SideSchedule:
    """Interleave quota-fixed records with the registered rational disperser."""

    if arm not in {ScheduleArm.DISPERSED_MATCHED, ScheduleArm.DISPERSED_BALANCED}:
        raise ValueError("the disperser only constructs B or C schedules")
    record_by_id = _input_map(records)
    if set(quotas) != set(record_by_id) or any(value <= 0 for value in quotas.values()):
        raise ValueError("quotas must name every record exactly once with positive exposure")
    queues: dict[str, list[str]] = {}
    events: list[tuple[int, int, str, str, int]] = []
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for input_id, record in record_by_id.items():
        by_stratum[record_stratum(record)].append(input_id)
    for stratum, members in by_stratum.items():
        occurrences: list[tuple[int, int, str, str, int]] = []
        for input_id in members:
            count = quotas[input_id]
            for occurrence in range(count):
                occurrences.append(
                    (
                        2 * occurrence + 1,
                        2 * count,
                        _hash((root, side, stratum, input_id, occurrence)),
                        f"{input_id}/{occurrence}",
                        occurrence,
                    )
                )
        occurrences.sort(key=functools.cmp_to_key(_compare_occurrences))
        queues[stratum] = [item[3].rsplit("/", 1)[0] for item in occurrences]
        total = len(occurrences)
        events.extend(
            (
                2 * occurrence + 1,
                2 * total,
                stratum,
                _hash((root, side, stratum, occurrence)),
                occurrence,
            )
            for occurrence in range(total)
        )
    events.sort(key=functools.cmp_to_key(_compare_events))
    positions = {stratum: 0 for stratum in queues}
    entries: list[ScheduledRecord] = []
    for position, (_, _, stratum, _, _) in enumerate(events):
        index = positions[stratum]
        entries.append(
            ScheduledRecord(
                side=side, input_id=queues[stratum][index], stratum_id=stratum, position=position
            )
        )
        positions[stratum] += 1
    schedule = SideSchedule(arm=arm, side=side, entries=tuple(entries))
    if schedule.quotas() != quotas:
        raise RuntimeError("dispersed schedule did not preserve record quotas")
    return schedule


def build_side_schedule(
    records: list[CNPRecord], *, arm: ScheduleArm, side: Side, root: int
) -> SideSchedule:
    """Build one registered side schedule for arm A, B, or C."""

    ordered = ordered_schedule(records, side=side)
    if arm is ScheduleArm.ORDERED:
        return ordered
    quotas = (
        ordered.quotas()
        if arm is ScheduleArm.DISPERSED_MATCHED
        else balanced_quotas(records, side=side, root=root)
    )
    return dispersed_schedule(records, arm=arm, side=side, root=root, quotas=quotas)


def schedule_audit(schedule: SideSchedule) -> dict[str, object]:
    """Return pre-training quota and tail-distribution evidence."""

    quotas = schedule.quotas()
    strata = Counter(entry.stratum_id for entry in schedule.entries)
    tail = {
        str(size): dict(Counter(entry.stratum_id for entry in schedule.entries[-size:]))
        for size in (16, 64)
    }
    maximum_run = 0
    run = 0
    previous: str | None = None
    for entry in schedule.entries:
        run = run + 1 if entry.stratum_id == previous else 1
        maximum_run = max(maximum_run, run)
        previous = entry.stratum_id
    return {
        "arm": schedule.arm.value,
        "side": schedule.side,
        "records": len(schedule.entries),
        "batches": len(schedule.batches),
        "schedule_sha256": schedule.sha256(),
        "record_quotas": dict(sorted(quotas.items())),
        "stratum_quotas": dict(sorted(strata.items())),
        "max_consecutive_stratum": maximum_run,
        "tail_stratum_counts": tail,
    }
