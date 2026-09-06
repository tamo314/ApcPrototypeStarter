# ruff: noqa: E501
"""Task B-C005R3-002: Semantic-Relation Split & Exposure Protocol (Option C).

ADR-0081 requires development / validation / sealed **relation** sets, not
just seed sets, before any further repair training begins. This module:

1. Enumerates the real relation catalogue from the actual hard-negative
   registry (`apc.evaluation.hard_negative_routing_benchmark._RELATED_OPERATION`
   and `apc.primitives.argument_scoring.PARAMETERIZED_OPERATIONS`) -- never a
   fabricated or hand-picked relation list.
2. Groups directed relation units into unordered-family-pair groups and, for
   L3, further merges groups that share a physical primitive into coupled
   components (repairing one perturbs the other's router key too -- see
   `build_exposure_manifest`).
3. Assigns DEV / VALIDATION / SEALED_V2 membership at both the relation-group
   axis and a new seed axis, and reports whether the real registry has enough
   independent, non-alias groups to satisfy the design contract's minimum of
   2 per partition (`docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`
   S5 step 7) -- `PROTOCOL_INSUFFICIENT_RELATIONS` when it does not, per that
   same document's own explicit escape hatch, rather than silently loosening
   the bar.
4. Audits the *existing* (unmodified) repair-training code
   (`apc.evaluation.retrieval_repair_benchmark.train_repaired_router_and_scorer`,
   `apc.primitives.argument_scoring.ArgumentScorer`) for what relation-group
   holdout is structurally achievable without touching that code -- this task
   is forbidden from changing weights, losses, or the scorer itself.
5. Profiles a development fixture on the frozen parent checkpoint to confirm
   it genuinely contains the already-diagnosed COUNT<->BIND / SELECT->BIND
   failures, without cherry-picking only the failing rows.
6. Pre-registers sealed-partition (SEALED_V2) seed membership and per-relation
   quotas -- membership and recipe only; SEALED_V2 model outputs are not
   measured by this task (that is R3-012's job, after R3-011 seals it).

Out of scope (must not be imported or touched by this module): router
weights, ArgumentScorer weights, verifier, controller, primitive weights,
adequacy thresholds. The one real training call this module makes
(`_probe_key_exposure_empirically`) trains a throwaway `deepcopy`d router on
a tiny, non-persisted reconstruction purely to empirically confirm an
exposure claim; it never writes back to any resident checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from apc.environments.operations import PHASE_A2_INCREMENTAL_NEW_OPERATIONS
from apc.evaluation.hard_negative_repair_gate import DEFAULT_REGATE_SEEDS
from apc.evaluation.hard_negative_routing_benchmark import (
    _RELATED_OPERATION,
    DEFAULT_TARGET_OPERATIONS,
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
)
from apc.evaluation.hard_negative_second_diagnostic import (
    HardNegativeSecondDiagnosticConfig,
    _collect_rows,
)
from apc.evaluation.incremental_router_benchmark import INITIAL_10_OPERATIONS
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.evaluation.retrieval_repair_benchmark import (
    DEFAULT_DEV_SEEDS,
    SEALED_GATE_SEEDS,
    RetrievalRepairConfig,
    train_repaired_router_and_scorer,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.primitives.argument_scoring import PARAMETERIZED_OPERATIONS
from apc.utils.system_info import get_system_info

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_L3: Final[HardNegativeLevel] = HardNegativeLevel.L3_SEMANTICALLY_RELATED
_L4: Final[HardNegativeLevel] = HardNegativeLevel.L4_CONFUSABLE_FAMILY
_PRE_REPAIR: Final[str] = "R0_frozen_pre_repair"

FULL_BANK_16_OPERATIONS: Final[tuple[str, ...]] = tuple(INITIAL_10_OPERATIONS) + tuple(
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS
)

# New seed partitions this task registers. Neither range has ever been used
# by any prior B-C005 family task (B-C005/D/R1/R2/G/D2 use only 0-4, 10-14,
# 20-24 -- confirmed against R3-001's artifact_inventory.json). Both original
# sealed partitions (0-4, 20-24) are RETIRED for future sealed measurement:
# both have already been viewed/measured extensively (B-C005, B-C005G,
# B-C005D2), and design-doc S8/addendum rule 4 forbid reusing an
# already-viewed sealed partition as a new sealed gate.
NEW_VALIDATION_SEEDS: Final[tuple[int, ...]] = (15, 16, 17, 18, 19)
NEW_SEALED_V2_SEEDS: Final[tuple[int, ...]] = (30, 31, 32, 33, 34)
RETIRED_SEALED_SEED_GROUPS: Final[dict[str, tuple[int, ...]]] = {
    "original_sealed_b_c005": tuple(sorted(SEALED_GATE_SEEDS)),
    "regate_sealed_b_c005g_d2": DEFAULT_REGATE_SEEDS,
}
_MIN_GROUPS_PER_PARTITION: Final[int] = 2

_D2_RUN_DIR: Final[str] = "runs/phase_b_b2_second_diagnostic"


def assert_sealed_access_permitted(seeds: tuple[int, ...], *, purpose: str) -> None:
    """Guard used by anything that must never silently touch a sealed partition.

    Mirrors the existing `SEALED_GATE_SEEDS` guard in
    `retrieval_repair_benchmark.run_retrieval_repair_benchmark`, extended to
    also cover the new SEALED_V2 partition this task registers.
    """
    sealed = set(SEALED_GATE_SEEDS) | set(DEFAULT_REGATE_SEEDS) | set(NEW_SEALED_V2_SEEDS)
    hit = sealed & set(seeds)
    if hit and purpose != "R3-011_seal_or_R3-012_gate":
        raise ValueError(
            f"Refusing to access sealed seeds {sorted(hit)} for purpose "
            f"{purpose!r}. Sealed partitions may only be accessed with "
            "purpose='R3-011_seal_or_R3-012_gate'."
        )


# ---------------------------------------------------------------------------
# 1. Relation catalogue (structural, derived from the real registry).
# ---------------------------------------------------------------------------


def l3_relation_groups() -> dict[str, dict[str, Any]]:
    """Unordered-family-pair L3 groups, derived purely from the real
    `_RELATED_OPERATION` registry. Directed units that name the same two
    families (e.g. `COUNT->BIND` and `BIND->COUNT`) merge into one group so
    they can never be split across DEV/VALIDATION/SEALED_V2."""
    groups: dict[frozenset[str], list[str]] = {}
    for target, competitor in _RELATED_OPERATION.items():
        key = frozenset({target, competitor})
        groups.setdefault(key, []).append(f"{target}->{competitor}")
    result: dict[str, dict[str, Any]] = {}
    for members, directed_units in groups.items():
        group_id = "-".join(sorted(members))
        result[group_id] = {
            "members": sorted(members),
            "directed_units": sorted(directed_units),
            "level": _L3.value,
        }
    return result


def l3_coupled_components() -> dict[str, dict[str, Any]]:
    """Merge L3 groups sharing a physical primitive into one coupled component.

    R3-006 repairs "COUNT/BINDに関係するkey/scoring" -- since BIND's router
    key is shared between the `COUNT-BIND` and `SELECT-BIND` groups, that
    repair necessarily also perturbs `SELECT-BIND`'s key geometry even though
    SELECT's own data is never touched. Two groups can therefore only be
    independently held out if they share no physical primitive.
    """
    groups = l3_relation_groups()
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for group in groups.values():
        members = group["members"]
        for member in members:
            find(member)
        for member in members[1:]:
            union(members[0], member)

    components: dict[str, list[str]] = {}
    for group_id, group in groups.items():
        root = find(group["members"][0])
        components.setdefault(root, []).append(group_id)

    result: dict[str, dict[str, Any]] = {}
    for group_ids in components.values():
        comp_id = "+".join(sorted(group_ids))
        member_ops = sorted({m for gid in group_ids for m in groups[gid]["members"]})
        result[comp_id] = {
            "l3_groups": sorted(group_ids),
            "member_operations": member_ops,
            "coupling_reason": (
                "independent: single unordered family pair, shares no physical "
                "primitive with any other L3 group"
                if len(group_ids) == 1
                else f"coupled: shares physical primitive(s) among operations {member_ops}"
            ),
        }
    return result


def l4_relation_groups() -> dict[str, dict[str, Any]]:
    """One structurally independent L4 group per parameterized operation.

    Independence is a code fact, not an assumption: `ArgumentScorer.heads`
    (`src/apc/primitives/argument_scoring.py`) is an `nn.ModuleDict` with one
    dedicated `nn.Linear` per operation, and `train_on_examples` builds a
    separate `AdamW` optimizer over only that op's `head.parameters()` --
    training one op's head can never change another op's parameters.
    """
    return {
        f"{op}:argument_variant": {
            "members": [op],
            "level": _L4.value,
            "independence_basis": (
                "ArgumentScorer.heads[op] is a dedicated nn.Linear with its own "
                "AdamW optimizer in train_on_examples; no parameter is shared "
                "across operations."
            ),
        }
        for op in PARAMETERIZED_OPERATIONS
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _l3_observed_status(repo_root: Path) -> dict[str, Any]:
    """Read D2-006's already-committed per-relation L3 verdict (never re-derived)."""
    path = repo_root / _D2_RUN_DIR / "final_causal_diagnosis.json"
    diagnosis = _read_json(path)
    if diagnosis is None:
        return {"available": False, "per_relation_verdict": {}, "source": str(path)}
    per_relation: dict[str, str] = {}
    for row in diagnosis.get("findings_table", []):
        if row.get("mechanism") == "L3 key/scoring":
            per_relation = dict(row.get("evidence", {}).get("per_relation_verdict", {}))
    return {
        "available": True,
        "per_relation_verdict": per_relation,
        "source": (
            "runs/phase_b_b2_second_diagnostic/final_causal_diagnosis.json:"
            "findings_table[mechanism='L3 key/scoring'].evidence.per_relation_verdict "
            "(B-C005D2-006, ADR-0081)"
        ),
    }


def _l4_observed_status(repo_root: Path) -> dict[str, Any]:
    """Read D2-004's already-committed per-operation L4 classification (never re-derived)."""
    path = repo_root / _D2_RUN_DIR / "l4_argument_breakdown.json"
    breakdown = _read_json(path)
    if breakdown is None:
        return {"available": False, "per_operation_classification": {}, "source": str(path)}
    classification = {
        op: entry.get("classification")
        for op, entry in breakdown.get("failure_classification", {}).items()
    }
    return {
        "available": True,
        "per_operation_classification": classification,
        "source": (
            "runs/phase_b_b2_second_diagnostic/l4_argument_breakdown.json:"
            "failure_classification (B-C005D2-004)"
        ),
    }


def build_relation_catalog(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """`relation_catalog.json`: the real relation catalogue plus observed status."""
    l3_groups = l3_relation_groups()
    l3_components = l3_coupled_components()
    l4_groups = l4_relation_groups()
    l3_status = _l3_observed_status(repo_root)
    l4_status = _l4_observed_status(repo_root)

    non_target_bank_members = sorted(set(FULL_BANK_16_OPERATIONS) - set(DEFAULT_TARGET_OPERATIONS))

    l3_catalog: dict[str, Any] = {}
    for group_id, group in l3_groups.items():
        directed_status = {
            unit: l3_status["per_relation_verdict"].get(unit, "NOT_YET_EVALUATED")
            for unit in group["directed_units"]
        }
        l3_catalog[group_id] = {**group, "directed_unit_status": directed_status}

    l4_catalog: dict[str, Any] = {}
    for group_id, group in l4_groups.items():
        op = group["members"][0]
        l4_catalog[group_id] = {
            **group,
            "status": l4_status["per_operation_classification"].get(op, "NOT_YET_EVALUATED"),
        }

    return {
        "task": "B-C005R3-002",
        "source_registry": "apc.evaluation.hard_negative_routing_benchmark._RELATED_OPERATION",
        "target_operations": list(DEFAULT_TARGET_OPERATIONS),
        "full_bank_16_operations": list(FULL_BANK_16_OPERATIONS),
        "l3": {
            "unit_definition": (
                "unordered_family_pair; directed units sharing a physical "
                "primitive (COUNT->BIND / BIND->COUNT) merge into one group"
            ),
            "groups": l3_catalog,
            "coupled_components": l3_components,
            "non_alias_group_count": len(l3_groups),
            "independent_component_count": len(l3_components),
            "observed_status_source": l3_status["source"],
            "observed_status_available": l3_status["available"],
        },
        "l4": {
            "unit_definition": "per-parameterized-operation argument-variant (same physical primitive, wrong argument)",
            "groups": l4_catalog,
            "non_alias_group_count": len(l4_groups),
            "observed_status_source": l4_status["source"],
            "observed_status_available": l4_status["available"],
        },
        "bank_members_with_no_l3_relation_defined": {
            "operations": non_target_bank_members,
            "reason": (
                "_RELATED_OPERATION only maps the 4 parameterized target "
                "operations to a competitor; the other 12 bank-16 operations "
                "never appear as an L3 target and have no defined semantic-"
                "relation competitor in the current registry. Recorded as a "
                "real gap, not filled with a fabricated relation."
            ),
            "evaluated": "UNEVALUATED_NO_RELATION_DEFINED",
        },
        "total_independent_holdout_units": len(l3_components) + len(l4_groups),
    }


# ---------------------------------------------------------------------------
# 2. Relation split (DEV / VALIDATION / SEALED_V2 membership).
# ---------------------------------------------------------------------------


def _unit_is_failing(status_values: list[str]) -> bool:
    return any(value != "NO_FAILURE" for value in status_values)


def assign_relation_partitions(catalog: dict[str, Any]) -> dict[str, Any]:
    """Assign each independent holdout unit to DEV / VALIDATION / SEALED_V2.

    Units with a diagnosed failure that an already-planned R3-0xx repair task
    will actively train on go to DEV (design-doc S5 step 5: known failures
    belong to `targeted_repair_regression`'s DEV membership, they are not
    fresh holdout material). Only units with `NO_FAILURE` status are eligible
    for VALIDATION/SEALED_V2 (`repair_relation_transfer` material) -- split as
    evenly as possible between the two.
    """
    dev: list[str] = []
    clean: list[str] = []

    for comp_id, comp in catalog["l3"]["coupled_components"].items():
        statuses = [
            catalog["l3"]["groups"][gid]["directed_unit_status"][unit]
            for gid in comp["l3_groups"]
            for unit in catalog["l3"]["groups"][gid]["directed_units"]
        ]
        (dev if _unit_is_failing(statuses) else clean).append(f"L3:{comp_id}")

    for group_id, group in catalog["l4"]["groups"].items():
        (dev if _unit_is_failing([group["status"]]) else clean).append(f"L4:{group_id}")

    validation = sorted(clean[0::2])
    sealed = sorted(clean[1::2])
    dev = sorted(dev)

    sufficiency = {
        "development": len(dev) >= _MIN_GROUPS_PER_PARTITION,
        "validation": len(validation) >= _MIN_GROUPS_PER_PARTITION,
        "sealed_v2": len(sealed) >= _MIN_GROUPS_PER_PARTITION,
    }
    return {
        "development_units": dev,
        "validation_units": validation,
        "sealed_v2_units": sealed,
        "minimum_required_per_partition": _MIN_GROUPS_PER_PARTITION,
        "sufficiency": sufficiency,
        "all_partitions_sufficient": all(sufficiency.values()),
        "assignment_rule": (
            "units with any non-NO_FAILURE directed-unit/operation status go "
            "to development (targeted_repair_regression); remaining NO_FAILURE "
            "units split as evenly as possible between validation and "
            "sealed_v2 (repair_relation_transfer material)"
        ),
    }


def build_seed_partition_registration() -> dict[str, Any]:
    """Seed-axis registration: DEV/VALIDATION/SEALED_V2, disjoint, non-aliased.

    VALIDATION_SEEDS and SEALED_V2_SEEDS use `model_seed = seed` directly
    (never `seed % 5`) so that each registered seed is its own independently
    initialized core, matching how the existing `development` partition
    (10-14) is already reconstructed in `hard_negative_second_diagnostic
    ._collect_rows` -- unlike `regate_sealed`, which deliberately reuses the
    original 0-4 checkpoints via `seed % 5` and must not be miscounted as 5
    independent models (design-doc S3).
    """
    all_seed_groups = {
        "development": tuple(DEFAULT_DEV_SEEDS),
        "validation": NEW_VALIDATION_SEEDS,
        "sealed_v2": NEW_SEALED_V2_SEEDS,
        **{f"retired_{name}": seeds for name, seeds in RETIRED_SEALED_SEED_GROUPS.items()},
    }
    seen: dict[int, str] = {}
    collisions: list[dict[str, Any]] = []
    for group_name, seeds in all_seed_groups.items():
        for seed in seeds:
            if seed in seen:
                collisions.append({"seed": seed, "groups": [seen[seed], group_name]})
            else:
                seen[seed] = group_name

    return {
        "development": {"seeds": list(DEFAULT_DEV_SEEDS), "model_seed_recipe": "model_seed = seed"},
        "validation": {"seeds": list(NEW_VALIDATION_SEEDS), "model_seed_recipe": "model_seed = seed"},
        "sealed_v2": {
            "seeds": list(NEW_SEALED_V2_SEEDS),
            "model_seed_recipe": "model_seed = seed",
            "status": "MEMBERSHIP_REGISTERED_NOT_MEASURED",
            "note": "Model outputs for these seeds are not measured by R3-002; measurement is R3-012's job, after R3-011 seals this membership.",
        },
        "retired_sealed_partitions": {
            name: {
                "seeds": list(seeds),
                "status": "RETIRED_ALREADY_VIEWED",
                "reason": "Already measured/viewed by B-C005/B-C005G/B-C005D2; not eligible to be reused as a new sealed partition.",
            }
            for name, seeds in RETIRED_SEALED_SEED_GROUPS.items()
        },
        "all_partitions_pairwise_disjoint": len(collisions) == 0,
        "collisions": collisions,
        "modulo_5_alias_used": False,
    }


_QUERY_EXAMPLES_PER_EPISODE: Final[int] = 256  # EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md S2
_SUPPORT_EPISODES_PER_CELL: Final[int] = 16  # ibid.
_DIFFICULTY_BIN_COUNT: Final[int] = 3  # matches semantic_relation_audit.py's cosine_bin_count default


def build_quota_preregistration(catalog: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered per-relation/per-difficulty-bin quotas (counts only, no
    sealed model output measured -- design-doc S8 / task doc R3-002 item 6)."""
    l3_units = list(catalog["l3"]["coupled_components"].keys())
    l4_units = list(catalog["l4"]["groups"].keys())
    quotas = {
        f"L3:{unit}": {
            "query_examples_per_episode": _QUERY_EXAMPLES_PER_EPISODE,
            "support_episodes_per_cell": _SUPPORT_EPISODES_PER_CELL,
            "difficulty_bin_count": _DIFFICULTY_BIN_COUNT,
        }
        for unit in l3_units
    }
    quotas.update(
        {
            f"L4:{unit}": {
                "query_examples_per_episode": _QUERY_EXAMPLES_PER_EPISODE,
                "support_episodes_per_cell": _SUPPORT_EPISODES_PER_CELL,
                "difficulty_bin_count": _DIFFICULTY_BIN_COUNT,
            }
            for unit in l4_units
        }
    )
    return {
        "source": "EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md S2 primary design numbers",
        "difficulty_bin_basis": (
            "difficulty bins are computed from frozen-model key_cosine_similarity "
            "once a partition's model is measured; this task pre-registers only "
            "the bin COUNT, never a measured bin edge on a sealed partition"
        ),
        "per_relation_quota": quotas,
    }


def build_relation_split(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """`relation_split.json`."""
    catalog = build_relation_catalog(repo_root)
    partitions = assign_relation_partitions(catalog)
    seeds = build_seed_partition_registration()
    quotas = build_quota_preregistration(catalog)
    gate_result = (
        "INFRASTRUCTURE_OR_PROTOCOL_PASS" if partitions["all_partitions_sufficient"] else "PROTOCOL_INSUFFICIENT_RELATIONS"
    )
    return {
        "task": "B-C005R3-002",
        "gate": "G1",
        "relation_group_axis": partitions,
        "seed_axis": seeds,
        "quota_preregistration": quotas,
        "suites": {
            "targeted_repair_regression": {
                "units": partitions["development_units"],
                "note": "Already-diagnosed failing pairs, repaired and re-evaluated on independent development examples; does not require novel/unseen relation groups.",
            },
            "repair_relation_transfer": {
                "validation_units": partitions["validation_units"],
                "sealed_v2_units": partitions["sealed_v2_units"],
                "sufficient": partitions["sufficiency"]["validation"] and partitions["sufficiency"]["sealed_v2"],
                "note": (
                    "Requires >=2 non-alias, non-coupled units per partition. "
                    "See relation_group_axis.sufficiency for the actual count."
                    if not (partitions["sufficiency"]["validation"] and partitions["sufficiency"]["sealed_v2"])
                    else "Sufficiency requirement met."
                ),
            },
        },
        "gate_result": gate_result,
        "gate_result_scope": (
            "Applies to the relation-group axis (repair_relation_transfer suite) only. "
            "targeted_repair_regression and the seed axis are independently reported "
            "and are not blocked by this gate_result."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Training-exposure audit (design-doc S6).
# ---------------------------------------------------------------------------


def classify_group_exposure(
    *,
    held_out_ops: frozenset[str],
    positive_ops_seen: list[str],
    candidate_list_ops: list[str],
) -> str:
    """Classify one held-out relation group's exposure under a described run.

    STRICT_HOLDOUT: no held-out op appears anywhere (not a positive target,
    not in the candidate/key list that forms the CE softmax denominator).
    NO_HOLDOUT: at least one held-out op was sampled as a positive target.
    MINING_HOLDOUT_ONLY: no held-out op was ever a positive target, but at
    least one sits in the candidate list (so its key/head is still inside
    the negative-class/mining pool every step).
    """
    if held_out_ops & set(positive_ops_seen):
        return "NO_HOLDOUT"
    if held_out_ops & set(candidate_list_ops):
        return "MINING_HOLDOUT_ONLY"
    return "STRICT_HOLDOUT"


def _probe_key_exposure_empirically(
    *, seed: int, held_out_op: str, device: str = "cpu"
) -> dict[str, Any]:
    """Empirically confirm (not merely infer from reading code) whether a held-
    out op's router key changes when its OWN examples are excluded from R2
    training but it remains in the full resident `candidate_list`.

    Uses a tiny bank_size=16 reconstruction and a throwaway `deepcopy`d
    router (the same deepcopy `train_repaired_router_and_scorer` itself
    performs); nothing is written back to any resident checkpoint.
    """
    base_config = HardNegativeBenchmarkConfig(
        seeds=(seed,), bank_sizes=(16,), num_eval_examples=8,
        router_train_examples=8, router_steps=20, device=device,
    )
    core, bank, router, op_to_id = _build_frozen_base_system(seed, base_config)
    operation_by_id = {pid: operation for operation, pid in op_to_id.items()}
    candidate_ids = list(op_to_id.values())

    before = router.key_parameter(op_to_id[held_out_op]).detach().clone()

    dev_examples = {
        operation: generate_benchmark_examples(seed * 999 + pid, 8, operation=operation, split="train")
        for operation, pid in op_to_id.items()
        if operation != held_out_op
    }
    repair_config = RetrievalRepairConfig(
        seeds=(seed,), bank_sizes=(16,), router_train_examples=8, router_steps=20, device=device,
    )
    new_router, _ = train_repaired_router_and_scorer(
        core, router, candidate_ids, operation_by_id, dev_examples,
        config=repair_config, condition="R2", device=torch.device(device),
    )
    after = new_router.key_parameter(op_to_id[held_out_op]).detach().clone()
    key_changed = not torch.equal(before, after)
    classification = classify_group_exposure(
        held_out_ops=frozenset({held_out_op}),
        positive_ops_seen=list(dev_examples.keys()),
        candidate_list_ops=[operation_by_id[pid] for pid in candidate_ids],
    )
    return {
        "held_out_op": held_out_op,
        "held_out_op_excluded_from_positive_training_examples": True,
        "held_out_op_present_in_candidate_list": True,
        "router_key_changed_despite_exclusion": key_changed,
        "predicted_classification": classification,
        "empirically_confirms_prediction": (
            key_changed and classification == "MINING_HOLDOUT_ONLY"
        )
        or (not key_changed and classification != "MINING_HOLDOUT_ONLY"),
    }


def historical_exposure_classification(trained_operations: list[str] | None) -> dict[str, Any]:
    """Classify a checkpoint's historical training exposure.

    Returns ``UNKNOWN`` when provenance cannot be determined (design-doc S6:
    "親checkpointの過去training exposureが不明なら historical_exposure=UNKNOWN"),
    rather than assuming either full or no exposure when the answer is not
    actually knowable.
    """
    if trained_operations is None:
        return {"classification": "UNKNOWN", "trained_operations": None}
    if set(trained_operations) >= set(FULL_BANK_16_OPERATIONS):
        return {"classification": "CONFIRMED_ALL_GROUPS_NO_HOLDOUT", "trained_operations": sorted(trained_operations)}
    return {"classification": "CONFIRMED_PARTIAL", "trained_operations": sorted(trained_operations)}


def build_exposure_manifest(*, run_empirical_probe: bool, probe_seed: int, device: str) -> dict[str, Any]:
    """`exposure_manifest.json`."""
    base_training_operations = list(FULL_BANK_16_OPERATIONS)
    shift_component = next(
        comp for comp in l3_coupled_components().values() if "SHIFT" in comp["member_operations"]
    )
    l3_router_classification = classify_group_exposure(
        held_out_ops=frozenset(shift_component["member_operations"]),
        positive_ops_seen=base_training_operations,
        candidate_list_ops=base_training_operations,
    )
    manifest: dict[str, Any] = {
        "task": "B-C005R3-002",
        "parent_checkpoint_historical_exposure": {
            "mechanism": "_build_frozen_base_system -> update_router_incrementally(condition=R0_FULL_RETRAIN)",
            "source": "src/apc/evaluation/hard_negative_routing_benchmark.py:_build_frozen_base_system",
            **historical_exposure_classification(base_training_operations),
            "reason": (
                "R0_FULL_RETRAIN trains every one of the 16 bank operations as a "
                "positive CE target every calibration; this is knowable from code, "
                "not left as historical_exposure=UNKNOWN."
            ),
        },
        "repair_time_recipe_audit": {
            "router_key_axis": {
                "mechanism": "train_repaired_router_and_scorer(condition='R2')",
                "source": "src/apc/evaluation/retrieval_repair_benchmark.py:train_repaired_router_and_scorer",
                "finding": (
                    "`key_params = [new_router.key_parameter(pid) for pid in candidate_list]` "
                    "puts every resident primitive's key (the FULL candidate_list, not just "
                    "the ops with training examples) under one AdamW optimizer with "
                    "requires_grad=True. Every step, `keys = new_router._stacked_keys(candidate_list)` "
                    "rebuilds the full key matrix and `loss = F.cross_entropy(logits, targets)` "
                    "(or the R2 ranking loss) backprops through the softmax denominator over "
                    "ALL columns of `logits = query @ keys.T` -- so every resident key receives "
                    "a nonzero gradient every step, whether or not its own operation ever "
                    "appears in `dev_train_examples_by_op` that step."
                ),
                "structural_ceiling": "MINING_HOLDOUT_ONLY",
                "structural_ceiling_reason": (
                    "STRICT_HOLDOUT for a relation group is NOT achievable under the "
                    "current, unmodified R2 recipe: excluding a held-out op's examples "
                    "from dev_train_examples_by_op only removes it as a positive target "
                    "(avoids NO_HOLDOUT); it cannot be removed from candidate_list without "
                    "changing train_repaired_router_and_scorer itself, which R3-002 may not do."
                ),
                "primary_pass_eligible": False,
                "design_doc_rule": "docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md S6: MINING_HOLDOUT_ONLY must not be used for the primary relation-holdout PASS.",
            },
            "argument_scorer_axis": {
                "mechanism": "ArgumentScorer.train_on_examples",
                "source": "src/apc/primitives/argument_scoring.py:ArgumentScorer.train_on_examples",
                "finding": (
                    "Each parameterized operation has a dedicated nn.Linear head and its "
                    "own AdamW optimizer scoped to only that head's parameters; an op "
                    "absent from examples_by_op is skipped entirely (`continue`), so its "
                    "head parameters are never touched."
                ),
                "structural_ceiling": "STRICT_HOLDOUT",
                "structural_ceiling_reason": (
                    "Achievable as long as a future repair task does not pass the held-out "
                    "op's examples into examples_by_op -- no candidate_list-style coupling "
                    "exists at L4."
                ),
                "primary_pass_eligible": True,
            },
        },
        "l3_shift_cycle_four_base_training_classification": l3_router_classification,
        "replay_and_mining_note": (
            "R2 training as currently coded does not use a bounded replay buffer "
            "(RouterReplayBuffer is only constructed for R0_FULL_RETRAIN base "
            "calibration, not inside train_repaired_router_and_scorer); its 'hard "
            "negative' per step is a synthesized near-neighbor key (random mix with "
            "the positive's own key), not literally the registered `_RELATED_OPERATION` "
            "competitor -- so R2's explicit hard-negative construction does not itself "
            "encode `_RELATED_OPERATION` exposure. The full-class CE denominator "
            "(above) is the actual exposure channel, independent of that."
        ),
        "empirical_probe": None,
    }
    if run_empirical_probe:
        manifest["empirical_probe"] = _probe_key_exposure_empirically(
            seed=probe_seed, held_out_op="SHIFT", device=device
        )
    return manifest


# ---------------------------------------------------------------------------
# 4. Development-fixture representativeness (design-doc / task item 5).
# ---------------------------------------------------------------------------


def build_development_representativeness(
    *, bank_size: int, query_examples: int, device: str
) -> dict[str, Any]:
    """`development_representativeness.json`.

    Profiles the real development partition (10-14) on the frozen parent
    checkpoint, pre-repair, restricted to L3, and reports the FULL per-
    relation outcome distribution (not just the failing rows) to confirm the
    already-diagnosed COUNT<->BIND / SELECT->BIND failures are genuinely
    present in this fixture rather than assumed.
    """
    reconstruction_config = HardNegativeSecondDiagnosticConfig(
        development_seeds=DEFAULT_DEV_SEEDS,
        bank_sizes=(bank_size,),
        levels=(_L3,),
        query_examples=query_examples,
        device=device,
        output_dir=None,
    )
    rows: list[dict[str, Any]] = []
    for seed in DEFAULT_DEV_SEEDS:
        rows.extend(
            _collect_rows(
                config=reconstruction_config,
                partition="development",
                state=_PRE_REPAIR,
                evaluation_seed=seed,
                model_seed=seed,
                training_seed=None,
            )
        )

    per_relation: dict[str, Any] = {}
    for row in rows:
        relation = str(row["semantic_relation_id"])
        bucket = per_relation.setdefault(relation, {"n": 0, "n_correct": 0, "n_incorrect": 0})
        bucket["n"] += 1
        if row["top1_correct"]:
            bucket["n_correct"] += 1
        else:
            bucket["n_incorrect"] += 1
    for bucket in per_relation.values():
        bucket["top1"] = bucket["n_correct"] / bucket["n"] if bucket["n"] else None

    contains_known_failures = all(
        per_relation.get(unit, {}).get("n_incorrect", 0) > 0
        for unit in ("COUNT->BIND", "BIND->COUNT")
    )
    return {
        "task": "B-C005R3-002",
        "fixture": {"partition": "development", "seeds": list(DEFAULT_DEV_SEEDS), "state": _PRE_REPAIR, "bank_size": bank_size},
        "rows_analyzed": len(rows),
        "per_relation_full_distribution": per_relation,
        "known_failures_present": {
            "count_bind_both_directions_have_incorrect_examples": contains_known_failures,
            "note": "Reports the FULL distribution (correct and incorrect); no row is dropped to make this fixture look more or less failure-laden than it is.",
        },
        "scale_disclaimer": (
            f"This profile pass runs at bank_size={bank_size} for speed and only "
            "confirms presence of real COUNT<->BIND/SELECT->BIND failure examples "
            "in this fixture. It is NOT the sealed-gate scale (bank_size=128) used "
            "by D2-002/D2-003's reported failure rates (e.g. COUNT->BIND top1 "
            "collapsing to 0.40 there); failure rates at this smaller scale are "
            "expected to be much milder and are not comparable to those numbers."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationSplitProtocolConfig:
    """Configuration for the B-C005R3-002 run."""

    development_representativeness_bank_size: int = 16
    development_representativeness_query_examples: int = 16
    run_empirical_exposure_probe: bool = True
    empirical_probe_seed: int = 10
    device: str = "auto"
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_002_relation_split")

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["output_dir"] = str(self.output_dir)
        return raw


def run_relation_split_protocol(config: RelationSplitProtocolConfig) -> dict[str, Any]:
    """Execute B-C005R3-002 and write its four run artifacts + run metadata."""
    start = time.perf_counter()
    device = (
        "cuda" if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    relation_catalog = build_relation_catalog()
    relation_split = build_relation_split()
    exposure_manifest = build_exposure_manifest(
        run_empirical_probe=config.run_empirical_exposure_probe,
        probe_seed=config.empirical_probe_seed,
        device=device,
    )
    development_representativeness = build_development_representativeness(
        bank_size=config.development_representativeness_bank_size,
        query_examples=config.development_representativeness_query_examples,
        device=device,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relation_catalog.json").write_text(json.dumps(relation_catalog, indent=2), encoding="utf-8")
    (output_dir / "relation_split.json").write_text(json.dumps(relation_split, indent=2), encoding="utf-8")
    (output_dir / "exposure_manifest.json").write_text(json.dumps(exposure_manifest, indent=2), encoding="utf-8")
    (output_dir / "development_representativeness.json").write_text(
        json.dumps(development_representativeness, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "system.json").write_text(json.dumps(get_system_info(), indent=2), encoding="utf-8")

    protocol = {
        "task": "B-C005R3-002",
        "gate": "G1",
        "result": relation_split["gate_result"],
        "criteria": {
            "relation_group_axis_all_partitions_sufficient": relation_split["relation_group_axis"]["all_partitions_sufficient"],
            "seed_axis_all_partitions_pairwise_disjoint": relation_split["seed_axis"]["all_partitions_pairwise_disjoint"],
            "router_key_axis_primary_pass_eligible": exposure_manifest["repair_time_recipe_audit"]["router_key_axis"]["primary_pass_eligible"],
            "argument_scorer_axis_primary_pass_eligible": exposure_manifest["repair_time_recipe_audit"]["argument_scorer_axis"]["primary_pass_eligible"],
            "known_failures_present_in_dev_fixture": development_representativeness["known_failures_present"]["count_bind_both_directions_have_incorrect_examples"],
            "model_weights_changed": False,
            "adequacy_threshold_changed": False,
        },
        "downstream_note": (
            "targeted_repair_regression (R3-006/007/008) may proceed using the "
            "development_units membership above regardless of this gate_result. "
            "repair_relation_transfer claims require gate_result == "
            "INFRASTRUCTURE_OR_PROTOCOL_PASS AND router_key_axis primary_pass_eligible "
            "for any L3 transfer claim specifically."
        ),
        "artifacts": [
            "relation_catalog.json", "relation_split.json", "exposure_manifest.json",
            "development_representativeness.json", "config.yaml", "system.json",
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "relation_catalog": relation_catalog,
        "relation_split": relation_split,
        "exposure_manifest": exposure_manifest,
        "development_representativeness": development_representativeness,
        "protocol": protocol,
    }
