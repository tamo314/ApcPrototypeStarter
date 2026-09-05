"""Holdout-family registry, operation definitions, and deterministic generators for Phase B.

Task B-C002: Holdout-family registry, identifiability checks, and leak audit.
Reference:
- `docs/design-docs/OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md`
- `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
- `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

Invariants:
1. Disjoint partitions:
   `DEV_FAMILIES`, `SEALED_FAMILIES`, and `RETIRED_FROM_SEALED` are strictly disjoint.
2. Sealed isolation:
   Sealed family identifiers, operation names, and status must never be model-visible.
3. Determinism:
   Holdout generators are strictly deterministic given a seed.
4. Same-length transforms:
   Operations preserve sequence length to isolate computational novelty from output-format changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Final

from apc.environments.generator import Example, OracleMetadata
from apc.environments.interpreter import run_program
from apc.environments.operations import Operation, get_operation, register_operation
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.meta.phase_b_protocol import FamilySplit, PhaseBProtocol

# ---------------------------------------------------------------------------
# Development Family Operations (DEV_FAMILIES)
# ---------------------------------------------------------------------------


class DevDeltaModOp(Operation):
    """Circular adjacent difference modulo vocabulary size.

    For each position i in [0, L-1]:
        output[i] = (sequence[i] - sequence[(i - 1) % L]) % vocab_size

    Development-only operation: structurally distinct from Phase A.2 operations
    by computing a same-length circular difference map.
    """

    name = "DEV_DELTA_MOD"
    min_input_length = 2

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        n = len(sequence)
        return tuple((sequence[i] - sequence[(i - 1) % n]) % vocab_size for i in range(n))


class DevWindowSumOp(Operation):
    """Circular 3-window sum modulo vocabulary size.

    For each position i in [0, L-1]:
        output[i] = (sequence[(i - 1) % L] + sequence[i] + sequence[(i + 1) % L]) % vocab_size

    Development-only operation: local neighborhood arithmetic sum.
    """

    name = "DEV_WINDOW_SUM"
    min_input_length = 3

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        n = len(sequence)
        return tuple(
            (sequence[(i - 1) % n] + sequence[i] + sequence[(i + 1) % n]) % vocab_size
            for i in range(n)
        )


# ---------------------------------------------------------------------------
# Sealed Evaluation Family Operations (SEALED_FAMILIES)
# ---------------------------------------------------------------------------


class MajorityThreeOp(Operation):
    """Circular 3-neighborhood majority vote or median transform.

    For each position i in [0, L-1], examining window w = (x_{i-1}, x_i, x_{i+1}):
    - If at least two elements are equal, outputs that repeated value.
    - If all three elements are distinct, outputs their median value.

    Primary sealed evaluation operation: cross-position conditional logic
    unsolvable by any Phase A.2 primitive or depth-2 composition.
    """

    name = "MAJORITY_THREE"
    min_input_length = 3

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        n = len(sequence)
        result: list[int] = []
        for i in range(n):
            a, b, c = sequence[(i - 1) % n], sequence[i], sequence[(i + 1) % n]
            if a == b or a == c:
                result.append(a)
            elif b == c:
                result.append(b)
            else:
                # Median of 3 distinct values
                result.append(sorted((a, b, c))[1])
        return tuple(result)


class NeighborMaxOp(Operation):
    """Circular 3-neighborhood maximum transform.

    For each position i in [0, L-1]:
        output[i] = max(sequence[(i - 1) % L], sequence[i], sequence[(i + 1) % L])

    Primary sealed evaluation operation: local sliding-window maximum.
    """

    name = "NEIGHBOR_MAX"
    min_input_length = 3

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        n = len(sequence)
        return tuple(
            max(sequence[(i - 1) % n], sequence[i], sequence[(i + 1) % n]) for i in range(n)
        )


class NeighborConditionalOp(Operation):
    """Circular neighbor-conditioned modular transform.

    For each position i in [0, L-1]:
        if sequence[(i - 1) % L] > sequence[i]:
            output[i] = (sequence[i] + sequence[(i + 1) % L]) % vocab_size
        else:
            output[i] = sequence[i]

    Primary sealed evaluation operation: relational conditional branching per token.
    """

    name = "NEIGHBOR_CONDITIONAL"
    min_input_length = 3

    def output_length(self, input_length: int) -> int:
        return input_length

    def sample_params(
        self, rng: random.Random, sequence: tuple[int, ...], vocab_size: int
    ) -> dict[str, Any]:
        return {}

    def apply(
        self, sequence: tuple[int, ...], vocab_size: int, params: dict[str, Any]
    ) -> tuple[int, ...]:
        n = len(sequence)
        result: list[int] = []
        for i in range(n):
            left = sequence[(i - 1) % n]
            mid = sequence[i]
            right = sequence[(i + 1) % n]
            if left > mid:
                result.append((mid + right) % vocab_size)
            else:
                result.append(mid)
        return tuple(result)


# Register all holdout operations into apc.environments.operations._REGISTRY
register_operation(DevDeltaModOp(), overwrite=True)
register_operation(DevWindowSumOp(), overwrite=True)
register_operation(MajorityThreeOp(), overwrite=True)
register_operation(NeighborMaxOp(), overwrite=True)
register_operation(NeighborConditionalOp(), overwrite=True)


# ---------------------------------------------------------------------------
# Family Metadata and Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyMetadata:
    """Evaluation-only metadata for an operation family (ADR-0074 / Phase B design)."""

    family_id: str
    status: FamilySplit
    structural_dependency_type: str
    output_shape_rule: str
    argument_schema: dict[str, Any]
    generator_version: str
    description: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must be non-empty")
        if not self.operations:
            raise ValueError(f"Family {self.family_id} must have at least one operation")


class HoldoutFamilyRegistry:
    """Central registry for Phase B development, sealed, and retired families."""

    def __init__(self) -> None:
        self._families: dict[str, FamilyMetadata] = {}
        self._op_to_family: dict[str, str] = {}

    def register_family(self, metadata: FamilyMetadata, *, overwrite: bool = False) -> None:
        """Register an operation family. Enforces partition disjointness."""
        fid = metadata.family_id
        if not overwrite and fid in self._families:
            raise ValueError(f"Family {fid!r} already registered")

        # Verify operation names exist in operations registry
        for op in metadata.operations:
            get_operation(op)  # Raises KeyError if not registered

            # Check disjointness across families
            existing_fam = self._op_to_family.get(op)
            if existing_fam and existing_fam != fid and not overwrite:
                raise ValueError(
                    f"Operation {op!r} is already assigned to family {existing_fam!r}; "
                    f"cannot reassign to {fid!r}"
                )

        self._families[fid] = metadata
        for op in metadata.operations:
            self._op_to_family[op] = fid

    def get_family(self, family_id: str) -> FamilyMetadata:
        """Look up family metadata by family ID."""
        try:
            return self._families[family_id]
        except KeyError:
            raise KeyError(f"Unknown family ID: {family_id!r}") from None

    def get_family_for_operation(self, operation_name: str) -> FamilyMetadata:
        """Look up family metadata by operation name."""
        try:
            fid = self._op_to_family[operation_name]
            return self._families[fid]
        except KeyError:
            raise KeyError(f"Operation {operation_name!r} not in any registered family") from None

    def list_families(self, status: FamilySplit | None = None) -> list[FamilyMetadata]:
        """List registered families, optionally filtered by research status."""
        if status is None:
            return list(self._families.values())
        return [f for f in self._families.values() if f.status == status]

    def list_operations(self, status: FamilySplit | None = None) -> tuple[str, ...]:
        """List operation names belonging to families of a given status."""
        fams = self.list_families(status)
        ops: list[str] = []
        for f in fams:
            ops.extend(f.operations)
        return tuple(ops)

    def retire_sealed_family(self, family_id: str, reason: str) -> FamilyMetadata:
        """Retire a sealed family that was used to tune architecture or thresholds (ADR-0074)."""
        current = self.get_family(family_id)
        if current.status != FamilySplit.SEALED_FAMILIES:
            raise ValueError(
                f"Cannot retire family {family_id!r}: status is {current.status.value}, "
                f"expected SEALED_FAMILIES"
            )
        retired = FamilyMetadata(
            family_id=current.family_id,
            status=FamilySplit.RETIRED_FROM_SEALED,
            structural_dependency_type=current.structural_dependency_type,
            output_shape_rule=current.output_shape_rule,
            argument_schema=current.argument_schema,
            generator_version=current.generator_version,
            description=f"{current.description} [RETIRED: {reason}]",
            operations=current.operations,
        )
        self.register_family(retired, overwrite=True)
        return retired

    def assert_disjoint_partitions(self) -> None:
        """Assert that DEV_FAMILIES, SEALED_FAMILIES, and RETIRED_FROM_SEALED are disjoint."""
        dev_ops = set(self.list_operations(FamilySplit.DEV_FAMILIES))
        sealed_ops = set(self.list_operations(FamilySplit.SEALED_FAMILIES))
        retired_ops = set(self.list_operations(FamilySplit.RETIRED_FROM_SEALED))

        dev_sealed_overlap = dev_ops & sealed_ops
        if dev_sealed_overlap:
            raise AssertionError(
                f"DEV_FAMILIES and SEALED_FAMILIES overlap: {dev_sealed_overlap}"
            )

        dev_retired_overlap = dev_ops & retired_ops
        if dev_retired_overlap:
            raise AssertionError(
                f"DEV_FAMILIES and RETIRED_FROM_SEALED overlap: {dev_retired_overlap}"
            )

        sealed_retired_overlap = sealed_ops & retired_ops
        if sealed_retired_overlap:
            raise AssertionError(
                f"SEALED_FAMILIES and RETIRED_FROM_SEALED overlap: {sealed_retired_overlap}"
            )


# Default registry pre-populated with standard Phase B partitions
DEFAULT_FAMILY_REGISTRY: Final[HoldoutFamilyRegistry] = HoldoutFamilyRegistry()

# 1. Development Family
DEV_FAMILY_METADATA: Final[FamilyMetadata] = FamilyMetadata(
    family_id="dev_local_difference",
    status=FamilySplit.DEV_FAMILIES,
    structural_dependency_type="local_neighborhood_arithmetic",
    output_shape_rule="same_length",
    argument_schema={},
    generator_version="1.0.0",
    description="Development-only local circular difference and window sum transforms.",
    operations=("DEV_DELTA_MOD", "DEV_WINDOW_SUM"),
)
DEFAULT_FAMILY_REGISTRY.register_family(DEV_FAMILY_METADATA)

# 2. Sealed Evaluation Family
SEALED_FAMILY_METADATA: Final[FamilyMetadata] = FamilyMetadata(
    family_id="sealed_local_neighborhood",
    status=FamilySplit.SEALED_FAMILIES,
    structural_dependency_type="local_neighborhood_conditional",
    output_shape_rule="same_length",
    argument_schema={},
    generator_version="1.0.0",
    description=(
        "Primary Phase B sealed evaluation family: "
        "local 3-neighborhood conditional transforms."
    ),
    operations=("MAJORITY_THREE", "NEIGHBOR_MAX", "NEIGHBOR_CONDITIONAL"),
)
DEFAULT_FAMILY_REGISTRY.register_family(SEALED_FAMILY_METADATA)


# ---------------------------------------------------------------------------
# Deterministic Holdout Episode Generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutEpisode:
    """Disjoint support, inference, verification, and query sets for one holdout episode."""

    operation_name: str
    family_id: str
    category: str  # "N" for novel, "R" for recurrence
    seed: int
    vocab_size: int
    seq_length: int
    support_examples: tuple[Example, ...]
    inference_examples: tuple[Example, ...]
    verification_examples: tuple[Example, ...]
    query_examples: tuple[Example, ...]

    @property
    def all_examples(self) -> tuple[Example, ...]:
        return (
            self.support_examples
            + self.inference_examples
            + self.verification_examples
            + self.query_examples
        )


def generate_holdout_episode(
    operation_name: str,
    protocol: PhaseBProtocol | None = None,
    *,
    seed: int = 0,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seq_length: int = 8,
    is_recurrence: bool = False,
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> HoldoutEpisode:
    """Deterministically generate an episode with disjoint example partitions under seed.

    Partitions:
    - support_examples: for plasticity adaptation
    - inference_examples: for task inference
    - verification_examples: for candidate adequacy checks
    - query_examples: for primary reported evaluation
    """
    family_meta = registry.get_family_for_operation(operation_name)
    category = "R" if is_recurrence else "N"

    # Default partition counts if protocol is not supplied
    if protocol is not None:
        n_support = protocol.support_count
        n_inference = protocol.inference_count
        n_verification = protocol.verification_count
        n_query = protocol.query_count
    else:
        n_support = 16
        n_inference = 16
        n_verification = 32
        n_query = 64

    total_count = n_support + n_inference + n_verification + n_query

    # Deterministic generation
    rng = random.Random(seed)
    program = Program(steps=(ProgramStep(operation=operation_name, params={}),))
    task_spec = TaskSpec.from_program(program)

    all_examples: list[Example] = []
    for _ in range(total_count):
        seq = tuple(rng.randrange(vocab_size) for _ in range(seq_length))
        res = run_program(program, seq, vocab_size)

        oracle_meta = OracleMetadata(
            label=category,  # type: ignore[arg-type]
            primitive_operations=program.operation_sequence,
            recurrence_operation=(operation_name if is_recurrence else None),
        )

        ex = Example(
            input_tokens=seq,
            target_tokens=res.output_tokens,
            program=program,
            operation_graph=res.graph,
            category=category,
            split="test",
            vocab_size=vocab_size,
            task_spec=task_spec,
            oracle_metadata=oracle_meta,
        )
        all_examples.append(ex)

    # Slice into strictly disjoint partitions
    idx = 0
    support = tuple(all_examples[idx : idx + n_support])
    idx += n_support
    inference = tuple(all_examples[idx : idx + n_inference])
    idx += n_inference
    verification = tuple(all_examples[idx : idx + n_verification])
    idx += n_verification
    query = tuple(all_examples[idx : idx + n_query])

    return HoldoutEpisode(
        operation_name=operation_name,
        family_id=family_meta.family_id,
        category=category,
        seed=seed,
        vocab_size=vocab_size,
        seq_length=seq_length,
        support_examples=support,
        inference_examples=inference,
        verification_examples=verification,
        query_examples=query,
    )


def generate_recurrence_episode(
    operation_name: str,
    protocol: PhaseBProtocol | None = None,
    *,
    seed: int = 0,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seq_length: int = 8,
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> HoldoutEpisode:
    """Convenience helper to generate a recurrence ('R') episode for a consolidated holdout."""
    return generate_holdout_episode(
        operation_name=operation_name,
        protocol=protocol,
        seed=seed,
        vocab_size=vocab_size,
        seq_length=seq_length,
        is_recurrence=True,
        registry=registry,
    )
