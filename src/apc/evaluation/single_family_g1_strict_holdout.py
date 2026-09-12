"""B-C005R3-002R: fixed-family G1 strict-holdout feasibility experiment.

This is deliberately a feasibility protocol, not a repair recipe or an
evaluation of a sealed model.  It preregisters exactly the parent Phase-B
``sealed_local_neighborhood`` family, runs the B-C002 deterministic oracle,
identifiability, and symbolic existing-library/depth-2 checks, then audits the
family as one conservatively coupled relation component.  A family member is
not treated as an independent transfer relation merely because it has a
different operation name: no learned primitive separates members of this
unseen family.

The comparison uses the existing full-class CE denominator and a relation-scoped
denominator on deep-copied routers; all throwaway updates are separately accounted for.  No
production model, bank, REC-004 artifact, or sealed model output is changed or
read by this module.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.environments.holdout_families import (
    DEFAULT_FAMILY_REGISTRY,
    HoldoutFamilyRegistry,
    generate_holdout_episode,
)
from apc.environments.operations import get_operation
from apc.evaluation.hard_negative_routing_benchmark import (
    HardNegativeBenchmarkConfig,
    _build_frozen_base_system,
)
from apc.evaluation.holdout_protocol import (
    PHASE_A2_16_OPERATIONS,
    check_identifiability,
    check_novelty_validity,
)
from apc.evaluation.incremental_router_benchmark import extract_task_representations
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.meta.phase_b_protocol import TaskInferenceModality
from apc.primitives.router import Router
from apc.utils.system_info import get_system_info

TASK_ID: Final[str] = "B-C005R3-002R"
GATE: Final[str] = "G1"
PREREGISTERED_FAMILY_ID: Final[str] = "sealed_local_neighborhood"
_MIN_CLEAN_COMPONENTS: Final[int] = 2
_GRADIENT_PROBE_SEED: Final[int] = 10
_IN_SCOPE_OPERATIONS: Final[tuple[str, str]] = ("COPY", "NEGATE")
_HELD_OUT_OPERATIONS: Final[tuple[str, str]] = ("SHIFT", "SELECT")


@dataclass(frozen=True)
class SingleFamilyG1Config:
    """Immutable pre-registration for the one permitted family."""

    family_id: str = PREREGISTERED_FAMILY_ID
    oracle_seed: int = 20260913
    sequence_length: int = 8
    verification_examples: int = 32
    novelty_threshold: float = 0.90
    max_composition_depth: int = 2
    gradient_probe_seed: int = _GRADIENT_PROBE_SEED
    gradient_probe_examples_per_operation: int = 4
    gradient_probe_lr: float = 1e-3
    device: str = "cpu"
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_002r_single_family_g1/run_002")

    def __post_init__(self) -> None:
        if self.family_id != PREREGISTERED_FAMILY_ID:
            raise ValueError(
                "B-C005R3-002R preregisters exactly one family: "
                f"{PREREGISTERED_FAMILY_ID!r}; alternatives are prohibited."
            )
        if self.sequence_length < 3:
            raise ValueError(
                "sequence_length must satisfy the registered local-neighborhood operations"
            )
        if self.verification_examples < 1 or self.gradient_probe_examples_per_operation < 1:
            raise ValueError("example counts must be positive")
        if self.max_composition_depth != 2:
            raise ValueError("B-C002 novelty control is fixed at existing-bank depth-2 composition")
        if self.gradient_probe_seed != _GRADIENT_PROBE_SEED:
            raise ValueError("gradient probe seed is fixed by this feasibility preregistration")
        if self.gradient_probe_lr <= 0:
            raise ValueError("gradient_probe_lr must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def preregistered_family(
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> dict[str, Any]:
    """Return the fixed family declaration before any outcome calculation."""
    family = registry.get_family(PREREGISTERED_FAMILY_ID)
    if family.output_shape_rule != "same_length":
        raise AssertionError("parent family no longer satisfies same-length contract")
    if family.structural_dependency_type != "local_neighborhood_conditional":
        raise AssertionError(
            "parent family no longer satisfies local-neighborhood conditional contract"
        )
    return {
        "task": TASK_ID,
        "exactly_one_family": True,
        "family_id": family.family_id,
        "family_status": family.status.value,
        "operations": list(family.operations),
        "structural_dependency_type": family.structural_dependency_type,
        "output_shape_rule": family.output_shape_rule,
        "argument_schema": family.argument_schema,
        "generator_version": family.generator_version,
        "alternative_families_compared": [],
        "sealed_model_outputs_inspected": 0,
    }


def build_single_family_coupling_graph(
    registry: HoldoutFamilyRegistry = DEFAULT_FAMILY_REGISTRY,
) -> dict[str, Any]:
    """Conservatively add the registered family as one coupled graph component.

    The graph is intentionally not inflated into one independent relation per
    operation.  Every operation belongs to the same newly proposed family and
    there is no existing learned primitive or separately frozen parameter set
    that could establish independent repair-time transfer evidence.
    """
    family = registry.get_family(PREREGISTERED_FAMILY_ID)
    nodes = [f"operation:{operation}" for operation in family.operations]
    edges = [
        {
            "source": left,
            "target": right,
            "reason": "same_preregistered_family_no_independent_learned_boundary",
        }
        for index, left in enumerate(nodes)
        for right in nodes[index + 1 :]
    ]
    component_id = f"family:{family.family_id}"
    return {
        "task": TASK_ID,
        "family_id": family.family_id,
        "nodes": nodes,
        "edges": edges,
        "components": {
            component_id: {
                "members": nodes,
                "clean": True,
                "alias_free": True,
                "non_coupled_with_other_components": True,
                "internal_coupling_reason": (
                    "All members are definitions of the one preregistered family; "
                    "counting them as independent relation-transfer groups would "
                    "violate the no-coupling requirement."
                ),
            }
        },
        "clean_non_alias_non_coupled_component_count": 1,
        "minimum_required_per_partition": _MIN_CLEAN_COMPONENTS,
        "partition_registration": {
            "validation": [],
            "sealed_v2": [component_id],
            "assignment_rule": (
                "The one family component is registered once for future sealed_v2 "
                "membership; no alternative family or synthetic relation is added "
                "to manufacture validation support."
            ),
        },
        "sufficiency": {"validation": False, "sealed_v2": False},
    }


def run_b_c002_controls(config: SingleFamilyG1Config) -> dict[str, Any]:
    """Run deterministic-oracle, identifiability, and symbolic novelty checks."""
    family = DEFAULT_FAMILY_REGISTRY.get_family(config.family_id)
    candidate_universe = tuple(PHASE_A2_16_OPERATIONS) + family.operations
    per_operation: dict[str, Any] = {}
    for index, operation in enumerate(family.operations):
        seed = config.oracle_seed + index
        first = generate_holdout_episode(operation, seed=seed, seq_length=config.sequence_length)
        second = generate_holdout_episode(operation, seed=seed, seq_length=config.sequence_length)
        deterministic = [
            (left.input_tokens, left.target_tokens) == (right.input_tokens, right.target_tokens)
            for left, right in zip(first.all_examples, second.all_examples, strict=True)
        ]
        oracle = get_operation(operation)
        oracle_matches = [
            oracle.apply(example.input_tokens, example.vocab_size, {}) == example.target_tokens
            and len(example.target_tokens) == len(example.input_tokens)
            for example in first.verification_examples
        ]
        explicit = check_identifiability(
            target_operation=operation,
            modality=TaskInferenceModality.EXPLICIT_TASK_SPEC,
            evidence=first.verification_examples[0].task_spec,
            candidate_universe=candidate_universe,
            vocab_size=first.vocab_size,
        )
        fewshot = check_identifiability(
            target_operation=operation,
            modality=TaskInferenceModality.FEWSHOT_DEMONSTRATIONS,
            evidence=first.inference_examples,
            candidate_universe=candidate_universe,
            vocab_size=first.vocab_size,
        )
        novelty = check_novelty_validity(
            target_operation=operation,
            verification_examples=first.verification_examples[: config.verification_examples],
            threshold=config.novelty_threshold,
            max_composition_depth=config.max_composition_depth,
            candidate_bank_ops=PHASE_A2_16_OPERATIONS,
            vocab_size=first.vocab_size,
        )
        per_operation[operation] = {
            "oracle_deterministic": all(deterministic),
            "oracle_output_matches": all(oracle_matches),
            "same_length": all(
                len(example.input_tokens) == len(example.target_tokens)
                for example in first.verification_examples
            ),
            "explicit_taskspec_identifiable": explicit.to_dict(),
            "fewshot_identifiability": fewshot.to_dict(),
            "novelty_validity": novelty.to_dict(),
        }
    return {
        "task": TASK_ID,
        "control_source": (
            "B-C002 holdout_protocol deterministic oracle / identifiability / "
            "existing-bank depth-2 baseline"
        ),
        "per_operation": per_operation,
        "all_deterministic_oracle_checks_pass": all(
            row["oracle_deterministic"] and row["oracle_output_matches"] and row["same_length"]
            for row in per_operation.values()
        ),
        "all_identifiability_checks_pass": all(
            row["explicit_taskspec_identifiable"]["is_identifiable"]
            and row["fewshot_identifiability"]["is_identifiable"]
            for row in per_operation.values()
        ),
        "all_novelty_validity_checks_pass": all(
            row["novelty_validity"]["is_valid_novel_holdout"] for row in per_operation.values()
        ),
    }


def _parameter_probe(
    router: Router,
    *,
    candidate_ids: Sequence[int],
    in_scope_ids: Sequence[int],
    held_out_ids: Sequence[int],
    representations: torch.Tensor,
    targets: torch.Tensor,
    relation_scoped: bool,
    learning_rate: float,
) -> dict[str, Any]:
    """One exact-gradient/update probe on a copied router."""
    probe_router = copy.deepcopy(router).train()
    probe_router.query_proj.requires_grad_(False)
    for key in probe_router._keys.values():
        key.requires_grad_(True)

    active_ids = list(in_scope_ids) if relation_scoped else list(candidate_ids)
    class_index = {primitive_id: index for index, primitive_id in enumerate(active_ids)}
    expected_targets = [class_index[int(target)] for target in targets.tolist()]
    active_targets = torch.tensor(expected_targets, dtype=torch.long, device=representations.device)
    active_params = [probe_router.key_parameter(primitive_id) for primitive_id in active_ids]
    held_params = [probe_router.key_parameter(primitive_id) for primitive_id in held_out_ids]
    before = {
        primitive_id: probe_router.key_parameter(primitive_id).detach().clone()
        for primitive_id in candidate_ids
    }
    optimizer = torch.optim.AdamW(active_params, lr=learning_rate, weight_decay=1e-4)

    optimizer.zero_grad(set_to_none=True)
    query = probe_router.query_proj(representations)
    keys = probe_router._stacked_keys(active_ids)
    loss = F.cross_entropy(query @ keys.transpose(0, 1), active_targets)
    loss.backward()
    gradients: dict[int, float] = {}
    for primitive_id in candidate_ids:
        gradient = probe_router.key_parameter(primitive_id).grad
        gradients[primitive_id] = (
            0.0 if gradient is None else float(gradient.detach().abs().max().item())
        )
    optimizer.step()

    return {
        "denominator": "relation_scoped_in_scope_only"
        if relation_scoped
        else "existing_full_class",
        "loss": float(loss.detach().item()),
        "active_candidate_ids": active_ids,
        "held_out_key_max_abs_grad": {str(pid): gradients[pid] for pid in held_out_ids},
        "in_scope_key_max_abs_grad": {str(pid): gradients[pid] for pid in in_scope_ids},
        "held_out_gradient_exactly_zero": all(gradients[pid] == 0.0 for pid in held_out_ids),
        "in_scope_gradient_nonzero": all(gradients[pid] > 0.0 for pid in in_scope_ids),
        "held_out_key_update_exactly_zero": all(
            torch.equal(before[pid], probe_router.key_parameter(pid).detach())
            for pid in held_out_ids
        ),
        "held_out_optimizer_state_absent": all(
            parameter not in optimizer.state for parameter in held_params
        ),
        "in_scope_optimizer_state_present": all(
            parameter in optimizer.state for parameter in active_params
        ),
        "optimizer_steps": 1,
        "model_copy_only": True,
    }


def run_gradient_isolation_probe(config: SingleFamilyG1Config) -> dict[str, Any]:
    """Compare current full-class CE with fixed relation-scoped CE once."""
    # This seed is a development seed and the existing bank checkpoint must
    # already exist; `_build_frozen_base_system` otherwise rebuilds primitives.
    checkpoint = (
        Path("runs/phase_a2_bank_scaling_benchmark")
        / f"seed_{config.gradient_probe_seed}"
        / "primitive_bank_16.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Required frozen development bank is absent; refusing a probe that could "
            "rebuild/train it: "
            f"{checkpoint}"
        )
    base_config = HardNegativeBenchmarkConfig(
        seeds=(config.gradient_probe_seed,),
        bank_sizes=(16,),
        num_eval_examples=config.gradient_probe_examples_per_operation,
        router_train_examples=config.gradient_probe_examples_per_operation,
        router_steps=1,
        device=config.device,
    )
    core, _bank, router, operation_to_id = _build_frozen_base_system(
        config.gradient_probe_seed, base_config
    )
    device = core.device
    candidate_ids = list(operation_to_id.values())
    in_scope_ids = [operation_to_id[operation] for operation in _IN_SCOPE_OPERATIONS]
    held_out_ids = [operation_to_id[operation] for operation in _HELD_OUT_OPERATIONS]
    examples = [
        example
        for operation in _IN_SCOPE_OPERATIONS
        for example in generate_benchmark_examples(
            config.gradient_probe_seed * 1000 + operation_to_id[operation],
            config.gradient_probe_examples_per_operation,
            operation=operation,
            split="train",
        )
    ]
    with torch.no_grad():
        representations = extract_task_representations(core, examples).detach()
    target_ids: list[int] = []
    for example in examples:
        assert example.task_spec is not None
        target_ids.append(operation_to_id[example.task_spec.steps[0].operation])
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    full_class = _parameter_probe(
        router,
        candidate_ids=candidate_ids,
        in_scope_ids=in_scope_ids,
        held_out_ids=held_out_ids,
        representations=representations,
        targets=targets,
        relation_scoped=False,
        learning_rate=config.gradient_probe_lr,
    )
    scoped = _parameter_probe(
        router,
        candidate_ids=candidate_ids,
        in_scope_ids=in_scope_ids,
        held_out_ids=held_out_ids,
        representations=representations,
        targets=targets,
        relation_scoped=True,
        learning_rate=config.gradient_probe_lr,
    )
    return {
        "task": TASK_ID,
        "throwaway_router_seed": config.gradient_probe_seed,
        "development_seed_only": True,
        "sealed_model_outputs_inspected": 0,
        "in_scope_operations": list(_IN_SCOPE_OPERATIONS),
        "held_out_operations": list(_HELD_OUT_OPERATIONS),
        "full_class_ce": full_class,
        "relation_scoped_ce": scoped,
        "strict_gradient_isolation_pass": (
            scoped["held_out_gradient_exactly_zero"]
            and scoped["held_out_key_update_exactly_zero"]
            and scoped["held_out_optimizer_state_absent"]
            and scoped["in_scope_gradient_nonzero"]
            and scoped["in_scope_optimizer_state_present"]
        ),
    }


def run_single_family_g1_feasibility(config: SingleFamilyG1Config) -> dict[str, Any]:
    """Run exactly the preregistered feasibility protocol and write evidence."""
    started = time.perf_counter()
    preregistration = preregistered_family()
    coupling_graph = build_single_family_coupling_graph()
    controls = run_b_c002_controls(config)
    gradients = run_gradient_isolation_probe(config)
    relation_sufficient = all(coupling_graph["sufficiency"].values())
    criteria = {
        "exactly_one_family_preregistered": preregistration["exactly_one_family"],
        "b_c002_deterministic_oracle": controls["all_deterministic_oracle_checks_pass"],
        "b_c002_identifiability": controls["all_identifiability_checks_pass"],
        "b_c002_novelty_validity": controls["all_novelty_validity_checks_pass"],
        "relation_count_sufficiency": relation_sufficient,
        "strict_gradient_isolation": gradients["strict_gradient_isolation_pass"],
        "sealed_outputs_inspected_zero": (
            preregistration["sealed_model_outputs_inspected"] == 0
            and gradients["sealed_model_outputs_inspected"] == 0
        ),
    }
    result = (
        "G1_STRICT_HOLDOUT_FEASIBILITY_PASS"
        if all(criteria.values())
        else "G1_RELATION_TRANSFER_STOP"
    )
    protocol = {
        "task": TASK_ID,
        "gate": GATE,
        "result": result,
        "criteria": criteria,
        "stop_reason": (
            None
            if result.endswith("PASS")
            else (
                "Single preregistered family supplies one conservatively coupled clean "
                "component; validation and sealed_v2 each require at least two clean, "
                "non-alias, non-coupled groups."
            )
        ),
        "execution_boundary": {
            "production_models_trained": 0,
            "production_models_modified": 0,
            "REC_004_artifacts_modified": 0,
            "candidate_selected": None,
            "child_bundle": None,
            "bundle_write": False,
            "rg3": "NOT_EXECUTED",
            "rec005": "BLOCKED",
            "sealed_model_outputs_inspected": 0,
            "throwaway_router_setup_optimizer_steps": 1,
            "throwaway_router_comparison_optimizer_steps": 2,
            "throwaway_router_optimizer_steps_total": 3,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "preregistration.json": preregistration,
        "relation_coupling_graph.json": coupling_graph,
        "b_c002_controls.json": controls,
        "gradient_isolation.json": gradients,
        "protocol.json": protocol,
        "config.json": config.to_dict(),
        "system.json": get_system_info(),
    }
    for filename, payload in artifacts.items():
        (config.output_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**artifacts, "output_dir": str(config.output_dir)}
