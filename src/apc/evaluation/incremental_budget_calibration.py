"""B-C005REC-004A: Incremental Primitive Budget Calibration & RG3 Recheck.

Follows `docs/CODEX_TASKS_PHASE_B_B2_INCREMENTAL_BUDGET_CALIBRATION.md`. This
is a single inserted task between REC-004's RG3 FAIL (ADR-0095) and REC-005:
it re-trains ONLY the 4 `INCREMENTAL_6_BUILD` operations that failed REC-004's
non-SHIFT recovery floor (CYCLE_FOUR, MIRROR_HALVES, ROTATE_TRIPLETS,
SWAP_ENDS -- physical ids 14, 12, 10, 11 in the real seed-10 bundle), on the
exact same frozen Core/12-protected-primitive/Router/ArgumentScorer bundle
REC-004 published, at a ladder of cumulative optimizer-step budgets
(1000/2000/4000/6000), selects the smallest budget clearing a fixed
`budget_validation` floor (sequence EM >= 0.95) per operation, and only if
all 4 select a candidate does it build a child bundle and re-run an
independent `recheck_query` + fresh-process RG3 recheck. It does not decide
in advance that more steps will fix anything, does not touch Core/the other
12 primitives/Router/ArgumentScorer, does not extend the search past 6000
steps per operation, and does not auto-start REC-005 regardless of outcome.

Stage lettering (A-E) matches the task doc exactly:
  A. audit existing artifacts/contracts/qualification scope (no training).
  B. fix the comparison protocol (`budget_protocol.json`).
  C. train the 4 operations, one continuous trajectory each, with
     checkpoints at the step ladder.
  D. select a candidate per operation from `budget_validation` and build a
     child `ModelBundleManifest` referencing REC-004's bundle as parent.
  E. independent `recheck_query` + fresh-process reproducibility check.

Stage D/E only run if all 4 operations select a validation-passing
checkpoint (`calibration_status == "SELECTED_ALL"`); otherwise the run stops
after Stage C with `calibration_status == "VALIDATION_TARGET_NOT_MET"` and
`rg3_recheck == "NOT_EXECUTED"`, per the task doc's own STOP rule.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    BRANCH_B_NOVEL_OPERATION_NAMES,
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation.compact_cross_position_operator_probe import _labels_for_examples
from apc.evaluation.model_bundle_recovery import (
    RECOVERY_NAMESPACE_ROOT,
    RECOVERY_PILOT_SEED,
    _evaluate_one_operation,
    _guard_not_frozen,
    _predict_tokens,
    _repo_root,
    _snapshot_forbidden_cache_hashes,
    frozen_evaluation,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _derive_local_seed,
    _evaluate_primitive_arm,
    _generate_parameter_free_examples,
)
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
)
from apc.utils import model_bundle as mb
from apc.utils.system_info import get_system_info

__all__ = [
    "REC004A_TASK_ID",
    "REC004A_TARGET_OPERATIONS",
    "REC004A_PROTECTED_NON_SHIFT_OPERATIONS",
    "REC004A_PROTECTED_OPERATIONS",
    "REC004A_STEP_LADDER",
    "REC004A_PARENT_BUNDLE_ID",
    "REC004A_PARENT_MANIFEST_PATH",
    "REC004A_PHYSICAL_IDS",
    "IncrementalBudgetCalibrationConfig",
    "run_incremental_budget_calibration_task",
]


# =============================================================================
# Constants -- pre-registered per the task doc sections 1.3/3/4, not chosen
# after seeing results.
# =============================================================================

REC004A_TASK_ID: Final = "B-C005REC-004A"
REC004A_PARENT_TASK_ID: Final = "B-C005REC-004"

# Order matches the task doc's own table (section 1.2).
REC004A_TARGET_OPERATIONS: Final[tuple[str, ...]] = (
    "CYCLE_FOUR",
    "MIRROR_HALVES",
    "ROTATE_TRIPLETS",
    "SWAP_ENDS",
)
REC004A_FROZEN_CONTROL_OPERATIONS: Final[tuple[str, ...]] = ("ALTERNATING_NEGATE", "INCREMENT_MOD")
REC004A_PROTECTED_NON_SHIFT_OPERATIONS: Final[tuple[str, ...]] = (
    "SELECT",
    "COUNT",
    "BIND",
    "COPY",
    "NEGATE",
    "SWAP_PAIRS",
    "INVERT_HALF",
    "REVERSE",
    "SORT",
    "ALTERNATING_NEGATE",
    "INCREMENT_MOD",
)
REC004A_PROTECTED_OPERATIONS: Final[tuple[str, ...]] = (
    *REC004A_PROTECTED_NON_SHIFT_OPERATIONS,
    "SHIFT",
)

# Real physical ids in the seed-10 REC-004 bundle (verified against
# manifest.json during Stage A, not assumed from name/build order alone).
REC004A_PHYSICAL_IDS: Final[dict[str, int]] = {
    "ROTATE_TRIPLETS": 10,
    "SWAP_ENDS": 11,
    "MIRROR_HALVES": 12,
    "CYCLE_FOUR": 14,
}

REC004A_STEP_LADDER: Final[tuple[int, ...]] = (1000, 2000, 4000, 6000)
REC004A_BUDGET_CAP_PER_OPERATION: Final = 6000
REC004A_VALIDATION_EXAMPLES: Final = 1024
REC004A_VALIDATION_FLOOR: Final = 0.95
REC004A_RECHECK_QUERY_EXAMPLES: Final = 1024
REC004A_TRAIN_FIT_STEP_COUNT: Final = 32  # 32 steps x 32 examples/step = 1024
REC004A_ARG_SCHEMA_HASH: Final = "apc_argument_schema_v1_recovery"

REC004A_BUDGET_VALIDATION_SPLIT: Final = "rec004a_budget_validation"
REC004A_RECHECK_QUERY_SPLIT: Final = "rec004a_recheck_query"
REC004A_TRAIN_SPLIT_LABEL: Final = "train"  # matches _train_single_primitive's own label

REC004A_PARENT_BUNDLE_ID: Final = (
    "2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995"
)
REC004A_PARENT_MANIFEST_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/bundles"
) / REC004A_PARENT_BUNDLE_ID / "manifest.json"
REC004A_PARENT_QUALIFICATION_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004/run_001/qualification.json"
)
REC004A_PARENT_PILOT_BUILD_REPORT_PATH: Final = Path(
    "runs/phase_b_b2_model_bundle_recovery/rec004/run_001/pilot_build_report.json"
)

# Optimizer/schedule recipe, read from runs/.../rec003/run_001/
# training_budget_manifest.json's INCREMENTAL_6_BUILD row and cross-checked
# against the live UnifiedBenchmarkConfig() defaults at runtime (Stage A/B) --
# not independently re-derived or guessed.
REC004A_OPERATOR_LR: Final = 0.0008
REC004A_OPERATOR_WEIGHT_DECAY: Final = 0.0001
REC004A_OPERATOR_GRAD_CLIP: Final = 1.0
REC004A_SCHEDULER_T_MAX: Final = 1000  # fixed at the ORIGINAL per-op recipe's steps=1000
REC004A_SCHEDULER_ETA_MIN: Final = 1e-5
REC004A_EXAMPLES_PER_STEP: Final = 32

REC004A_NAMESPACE_ROOT: Final = RECOVERY_NAMESPACE_ROOT  # same staging root as REC-004


def _rec004a_namespace(seed: int) -> Path:
    return Path(REC004A_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004a"


@dataclass(frozen=True)
class IncrementalBudgetCalibrationConfig:
    output_dir: Path = Path("runs/phase_b_b2_model_bundle_recovery/rec004a")
    seed: int = RECOVERY_PILOT_SEED
    step_ladder: tuple[int, ...] = REC004A_STEP_LADDER
    validation_examples: int = REC004A_VALIDATION_EXAMPLES
    validation_floor: float = REC004A_VALIDATION_FLOOR
    recheck_query_examples: int = REC004A_RECHECK_QUERY_EXAMPLES


# =============================================================================
# Manifest (de)serialization -- same round-trip shape as
# model_bundle_recovery._manifest_to_json_dict /
# scripts/rec004_fresh_process_check.py's manual reconstruction.
# =============================================================================


def _manifest_to_json_dict(manifest: mb.ModelBundleManifest) -> dict[str, Any]:
    def _component(c: mb.ComponentManifest | None) -> dict[str, Any] | None:
        return None if c is None else dataclasses.asdict(c)

    return {
        "schema_version": manifest.schema_version,
        "bundle_id": manifest.bundle_id,
        "content_manifest_digest": manifest.content_manifest_digest,
        "source_commit": manifest.source_commit,
        "runtime_recipe_version": manifest.runtime_recipe_version,
        "environment_record": dict(manifest.environment_record),
        "model_id": manifest.model_id,
        "model_seed": manifest.model_seed,
        "training_run_id": manifest.training_run_id,
        "parent_bundle_ids": list(manifest.parent_bundle_ids),
        "build_route": manifest.build_route.value,
        "scope": manifest.scope.value,
        "requested_capabilities": sorted(manifest.requested_capabilities),
        "publish_status": manifest.publish_status.value,
        "core": _component(manifest.core),
        "vocabulary": _component(manifest.vocabulary),
        "primitives": [
            {**dataclasses.asdict(p), "provenance_status": p.provenance_status.value}
            for p in manifest.primitives
        ],
        "router": dataclasses.asdict(manifest.router),
        "argument_scorer": dataclasses.asdict(manifest.argument_scorer),
        "scoring_policy": dataclasses.asdict(manifest.scoring_policy),
        "build_recipe_hash": manifest.build_recipe_hash,
        "known_defects": list(manifest.known_defects),
        "clean_build_exercised_stages": list(manifest.clean_build_exercised_stages),
        "qualification_refs": list(manifest.qualification_refs),
    }


def _manifest_from_json_dict(d: dict[str, Any]) -> mb.ModelBundleManifest:
    return mb.ModelBundleManifest(
        schema_version=d["schema_version"],
        bundle_id=d["bundle_id"],
        content_manifest_digest=d["content_manifest_digest"],
        source_commit=d["source_commit"],
        runtime_recipe_version=d["runtime_recipe_version"],
        environment_record=d["environment_record"],
        model_id=d["model_id"],
        model_seed=d["model_seed"],
        training_run_id=d["training_run_id"],
        parent_bundle_ids=tuple(d["parent_bundle_ids"]),
        build_route=mb.BuildRoute(d["build_route"]),
        scope=mb.BundleScope(d["scope"]),
        requested_capabilities=frozenset(d["requested_capabilities"]),
        publish_status=mb.PublishStatus(d["publish_status"]),
        core=mb.ComponentManifest(**d["core"]),
        vocabulary=mb.ComponentManifest(**d["vocabulary"]),
        primitives=tuple(
            mb.PrimitiveManifestEntry(
                **{**p, "provenance_status": mb.ProvenanceStatus(p["provenance_status"])}
            )
            for p in d["primitives"]
        ),
        router=mb.RouterManifest(**d["router"]),
        argument_scorer=mb.ArgumentScorerManifest(**d["argument_scorer"]),
        scoring_policy=mb.ScoringPolicyManifest(**d["scoring_policy"]),
        build_recipe_hash=d.get("build_recipe_hash"),
        known_defects=tuple(d.get("known_defects", ())),
        clean_build_exercised_stages=tuple(d.get("clean_build_exercised_stages", ())),
        qualification_refs=tuple(d.get("qualification_refs", ())),
    )


def _load_parent_manifest() -> tuple[mb.ModelBundleManifest, dict[str, Any]]:
    """Load REC-004's real, pre-registered parent bundle manifest. Never
    falls back to a different bundle_id, seed, or a "latest" search."""
    if not REC004A_PARENT_MANIFEST_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {REC004A_PARENT_MANIFEST_PATH} not found"
        )
    raw = json.loads(REC004A_PARENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = _manifest_from_json_dict(raw)
    if manifest.bundle_id != REC004A_PARENT_BUNDLE_ID:
        raise mb.IncompleteBundleError(
            f"parent manifest at {REC004A_PARENT_MANIFEST_PATH} declares bundle_id="
            f"{manifest.bundle_id}, expected the pre-registered {REC004A_PARENT_BUNDLE_ID}"
        )
    return manifest, raw


# =============================================================================
# Runtime reconstruction: frozen Core + full 16-op bank from the parent
# bundle's own checked state dicts (never a shared-cache path, never a
# builder/trainer call for the 12 protected + SHIFT slots).
# =============================================================================


def _reconstruct_16_op_bank_structure(core: Any, seed: int) -> tuple[Any, dict[str, int]]:
    """Structural rebuild only -- no weights, no training. Mirrors
    `scripts/rec004_fresh_process_check.py::_rebuild_16_bank_structure`
    exactly (same call order -> same deterministic physical_id assignment)."""
    u_bank_cfg = UnifiedBenchmarkConfig(seed=seed, vocab_size=10, device=str(core.device))
    bank, op_to_id = _build_heterogeneous_bank(u_bank_cfg)
    for op in tuple(BRANCH_B_NOVEL_OPERATION_NAMES) + tuple(PHASE_A2_INCREMENTAL_NEW_OPERATIONS):
        p = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=op,
                d_model=core.model.config.d_model,
                d_operator=32,
                n_head=4,
                d_operator_ff=64,
                vocab_size=10,
                max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[op] = p.primitive_id
    return bank, op_to_id


def _single_bank_path(manifest: mb.ModelBundleManifest) -> str:
    """All of this repo's REC-004/REC-004A manifests store every primitive's
    slice in one shared bank file (`source_artifact` identical across all
    entries) -- confirmed, not assumed, before relying on any one entry."""
    artifacts = {p.source_artifact for p in manifest.primitives}
    if len(artifacts) != 1:
        raise mb.IncompleteBundleError(
            f"expected all primitives in bundle {manifest.bundle_id} to share one bank "
            f"file, found {len(artifacts)}: {sorted(artifacts)}"
        )
    return next(iter(artifacts))


def _reconstruct_parent_runtime(
    config: IncrementalBudgetCalibrationConfig, parent_manifest: mb.ModelBundleManifest
) -> tuple[Any, Any, dict[str, int]]:
    """Load the frozen Core and the full 16-primitive bank from the parent
    bundle's own checked files (loaded through `mb.load_bundle`'s fail-closed
    contract first) -- never a shared-cache path, never a builder call."""
    _guard_not_frozen("reconstruct_parent_runtime")
    loaded = mb.load_bundle(parent_manifest, mode="diagnostic", expected_primitive_count=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg = SharedEncoderArchitectureConfig(seed=config.seed, vocab_size=10, device=str(device))
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.load_state_dict(loaded.core_state_dict)
    core.model.to(device)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    bank, op_to_id = _reconstruct_16_op_bank_structure(core, config.seed)
    fresh_op_to_id_expected = {p.operation_name: p.physical_id for p in parent_manifest.primitives}
    if op_to_id != fresh_op_to_id_expected:
        raise mb.IncompleteBundleError(
            f"reconstructed bank structure op_to_id {op_to_id} does not match parent "
            f"manifest's declared physical ids {fresh_op_to_id_expected}"
        )
    bank.to(device)
    combined_bank_sd = {
        f"_primitives.{pid}.{k}": v
        for pid, sd in loaded.primitive_state_dicts.items()
        for k, v in sd.items()
    }
    bank.load_state_dict(combined_bank_sd, strict=True)
    bank.freeze_all()
    bank.eval()
    return core, bank, op_to_id


# =============================================================================
# Stage A -- audit existing artifacts/contracts/qualification scope. No
# training, no bank/core construction beyond what's needed to prove the
# parent loads.
# =============================================================================


def _check_resume_state_available(seed: int) -> dict[str, Any]:
    """Real filesystem check (not an assumption) for any optimizer/scheduler/
    RNG/data-iterator checkpoint under REC-004's own staging namespace for
    this seed -- `_train_single_primitive` never wrote one, so this is
    expected to come back empty, but it is verified, not assumed."""
    root = Path(RECOVERY_NAMESPACE_ROOT) / f"seed_{seed}" / "rec004"
    if not root.is_dir():
        return {"namespace_root": str(root), "namespace_exists": False, "candidate_files_found": []}
    keywords = ("optim", "scheduler", "rng_state", "random_state", "data_iterator", "sampler_state")
    found = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and any(k in p.name.lower() for k in keywords)
    ]
    return {
        "namespace_root": str(root.resolve()),
        "namespace_exists": True,
        "candidate_files_found": sorted(found),
    }


def run_stage_a_audit(config: IncrementalBudgetCalibrationConfig) -> dict[str, Any]:
    _guard_not_frozen("run_stage_a_audit")
    parent_manifest, parent_manifest_raw = _load_parent_manifest()

    # A1/A2: parent artifact + hash confirmation, self-load through the real
    # fail-closed contract (diagnostic succeeds; an unearned capability is
    # rejected -- both are real `load_bundle` calls, not narrative claims).
    diagnostic_load = mb.load_bundle(
        parent_manifest, mode="diagnostic", expected_primitive_count=16
    )
    capability_rejected = False
    capability_rejection_label = None
    try:
        mb.load_bundle(
            parent_manifest,
            mode="nominal",
            required_capabilities=frozenset({"non_shift_15_recovery_floor_certified"}),
            expected_primitive_count=16,
        )
    except mb.CapabilityNotQualifiedError as exc:
        capability_rejected = True
        capability_rejection_label = exc.label

    physical_ids_by_op = {p.operation_name: p.physical_id for p in parent_manifest.primitives}
    for op, expected_pid in REC004A_PHYSICAL_IDS.items():
        actual = physical_ids_by_op.get(op)
        if actual != expected_pid:
            raise mb.IncompleteBundleError(
                f"parent manifest declares physical_id={actual} for {op}, "
                f"expected the pre-registered {expected_pid}"
            )

    parent_audit = {
        "task_id": REC004A_TASK_ID,
        "parent_task_id": REC004A_PARENT_TASK_ID,
        "manifest_path": str(REC004A_PARENT_MANIFEST_PATH.resolve()),
        "manifest_exists": True,
        "bundle_id": parent_manifest.bundle_id,
        "core_canonical_state_hash": parent_manifest.core.canonical_state_hash,
        "source_commit": parent_manifest.source_commit,
        "training_run_id": parent_manifest.training_run_id,
        "model_seed": parent_manifest.model_seed,
        "known_defects": list(parent_manifest.known_defects),
        "self_load_diagnostic_mode_checks_performed": list(diagnostic_load.checks_performed),
        "self_load_diagnostic_mode_passed": True,
        "unearned_capability_request_rejected": capability_rejected,
        "unearned_capability_rejection_label": capability_rejection_label,
        "target_operation_physical_ids": dict(REC004A_PHYSICAL_IDS),
        "protected_operation_physical_ids": {
            op: physical_ids_by_op[op] for op in REC004A_PROTECTED_OPERATIONS
        },
        "all_16_physical_ids": dict(sorted(physical_ids_by_op.items(), key=lambda kv: kv[1])),
    }

    # A3: resume state -- real filesystem check, per operation (identical
    # result expected for all 4 since they share one staging namespace, but
    # checked as one fact rather than asserted per-op).
    resume_check = _check_resume_state_available(config.seed)
    resume_audit = {
        "task_id": REC004A_TASK_ID,
        "checked": resume_check,
        "per_operation": {
            op: {
                "weights_only_artifact_available": True,
                "weights_only_source": (
                    f"parent bundle {parent_manifest.bundle_id}, physical_id "
                    f"{REC004A_PHYSICAL_IDS[op]}, source_artifact="
                    + next(
                        p.source_artifact
                        for p in parent_manifest.primitives
                        if p.operation_name == op
                    )
                ),
                "optimizer_state_available": False,
                "scheduler_state_available": False,
                "rng_state_available": False,
                "data_iterator_state_available": False,
                "conclusion": "FRESH_PAIRED_REBUILD",
                "conclusion_reason": (
                    "_train_single_primitive (unified_oracle_causal_benchmark.py) "
                    "constructs a fresh AdamW+CosineAnnealingLR every call and never "
                    "persists optimizer/scheduler/RNG/data-iterator state; REC-004's "
                    "own run_001 outputs and staging namespace contain no such file "
                    "(see 'checked' above). Weight-only continuation is not called "
                    "EXACT_CONTINUATION per the task doc's own rule."
                ),
            }
            for op in REC004A_TARGET_OPERATIONS
        },
    }

    # A4: operation contract -- exact source facts (operations.py), not
    # inferred from operation names. Cross-checked live against
    # `get_operation` rather than only quoted from a prior read.
    operation_contract_audit: dict[str, Any] = {"task_id": REC004A_TASK_ID, "operations": {}}
    contract_facts: dict[str, dict[str, Any]] = {
        "ROTATE_TRIPLETS": {
            "min_input_length": 3,
            "group_size": 3,
            "transform": "chunks of 3 rotated left: (x0,x1,x2) -> (x1,x2,x0)",
            "remainder_handling": "trailing 1-2 tokens (length % 3 != 0) left unchanged",
            "source": "src/apc/environments/operations.py:473-497",
        },
        "SWAP_ENDS": {
            "min_input_length": 2,
            "group_size": None,
            "transform": "only positions 0 and n-1 swap; all other positions unchanged",
            "remainder_handling": "not applicable (only 2 fixed positions touched)",
            "source": "src/apc/environments/operations.py:508-530",
        },
        "MIRROR_HALVES": {
            "min_input_length": 2,
            "group_size": None,
            "transform": (
                "mid = len // 2 (floor); output = reversed(seq[:mid]) + reversed(seq[mid:]); "
                "for odd length the second half receives the extra middle element"
            ),
            "remainder_handling": "n/a (both halves always covered; split point is floor(n/2))",
            "source": "src/apc/environments/operations.py:533-555",
        },
        "CYCLE_FOUR": {
            "min_input_length": 4,
            "group_size": 4,
            "transform": "chunks of 4 rotated left: (x0,x1,x2,x3) -> (x1,x2,x3,x0)",
            "remainder_handling": "trailing 1-3 tokens (length % 4 != 0) left unchanged",
            "source": "src/apc/environments/operations.py:585-613",
        },
    }
    for op, facts in contract_facts.items():
        op_obj = get_operation(op)
        operation_contract_audit["operations"][op] = {
            **facts,
            "output_length_is_identity_live_check": op_obj.output_length(7) == 7
            and op_obj.output_length(9) == 9,
            "is_parameter_free_live_check": not op_obj.required_argument_names,
            "vocab_untouched": True,
            "padding_masking": (
                "sequences right-padded to batch max with PAD; loss uses IGNORE_INDEX "
                "over the full output_length (no operation-specific position exclusion)"
            ),
        }

    # A5: qualification scope. Parent's own diagnostic-load success and RG3
    # FAIL are both real facts, kept in separate fields (never conflated).
    if not REC004A_PARENT_QUALIFICATION_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {REC004A_PARENT_QUALIFICATION_PATH} not found"
        )
    parent_qualification = json.loads(REC004A_PARENT_QUALIFICATION_PATH.read_text(encoding="utf-8"))
    qualification_scope_audit = {
        "task_id": REC004A_TASK_ID,
        "tier_1_diagnostic_loadable": True,
        "tier_1_evidence": (
            "mb.load_bundle(parent_manifest, mode='diagnostic') succeeded (see parent_audit)"
        ),
        "tier_2_non_shift_15_recovery_floor_passed": parent_qualification["criteria"][
            "non_shift_15_floor_passed"
        ],
        "tier_2_evidence": str(REC004A_PARENT_QUALIFICATION_PATH),
        "tier_2_failures": parent_qualification["criteria"]["non_shift_floor_failures"],
        "tier_3_r3_reference_contract_nominal_certified": "NOT_ATTEMPTED",
        "tier_3_note": (
            "R3's own REF_ADEQUATE/reference-adequacy contract is a separate statistical "
            "certification this task does not attempt or grant; query EM alone never "
            "implies it (task doc section 1.3, section 7 E2 note)."
        ),
        "unearned_full_capability_request_rejected": capability_rejected,
        "status": "QUALIFICATION_SCOPE_RESOLVED",
    }

    # existing learning-curve audit -- only what REC-004 actually recorded.
    if not REC004A_PARENT_PILOT_BUILD_REPORT_PATH.is_file():
        raise mb.MissingArtifactError(
            f"PARENT_ARTIFACT_UNAVAILABLE: {REC004A_PARENT_PILOT_BUILD_REPORT_PATH} not found"
        )
    pilot_build_report = json.loads(
        REC004A_PARENT_PILOT_BUILD_REPORT_PATH.read_text(encoding="utf-8")
    )
    incremental_stage = pilot_build_report["build_stages"]["INCREMENTAL_6_BUILD"]
    existing_learning_curve_audit = {
        "task_id": REC004A_TASK_ID,
        "source": str(REC004A_PARENT_PILOT_BUILD_REPORT_PATH),
        "steps_per_operation_original": incremental_stage["steps_per_operation"],
        "final_loss_at_1000_steps": {
            op: incremental_stage["final_losses"][op] for op in REC004A_TARGET_OPERATIONS
        },
        "intermediate_step_curve": "NOT_RECORDED",
        "train_fit_at_1000_steps": "NOT_RECORDED",
        "note": (
            "REC-004 recorded only the final loss after exactly 1000 steps per operation; "
            "no intermediate checkpoint, curve, or train-fit measurement exists for the "
            "original run. This task does not fabricate a curve or convergence claim for "
            "that original run from its single final value."
        ),
    }

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("parent_audit", parent_audit),
        ("resume_audit", resume_audit),
        ("operation_contract_audit", operation_contract_audit),
        ("existing_learning_curve_audit", existing_learning_curve_audit),
        ("qualification_scope_audit", qualification_scope_audit),
    ):
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    return {
        "parent_manifest": parent_manifest,
        "parent_manifest_raw": parent_manifest_raw,
        "parent_audit": parent_audit,
        "resume_audit": resume_audit,
        "operation_contract_audit": operation_contract_audit,
        "existing_learning_curve_audit": existing_learning_curve_audit,
        "qualification_scope_audit": qualification_scope_audit,
    }


# =============================================================================
# Stage B -- fix the comparison protocol before any training.
# =============================================================================


def build_budget_protocol(
    config: IncrementalBudgetCalibrationConfig, stage_a: dict[str, Any]
) -> dict[str, Any]:
    live_defaults = UnifiedBenchmarkConfig()
    if live_defaults.operator_lr != REC004A_OPERATOR_LR:
        raise ValueError(
            f"SCHEDULE_EXTENSION_REQUIRED: live UnifiedBenchmarkConfig.operator_lr="
            f"{live_defaults.operator_lr} no longer matches the recorded recipe value "
            f"{REC004A_OPERATOR_LR} this protocol was fixed against"
        )
    if live_defaults.operator_weight_decay != REC004A_OPERATOR_WEIGHT_DECAY:
        raise ValueError("SCHEDULE_EXTENSION_REQUIRED: operator_weight_decay drifted from recipe")
    if live_defaults.operator_grad_clip != REC004A_OPERATOR_GRAD_CLIP:
        raise ValueError("SCHEDULE_EXTENSION_REQUIRED: operator_grad_clip drifted from recipe")

    protocol = {
        "task_id": REC004A_TASK_ID,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "parent_core_canonical_state_hash": stage_a["parent_manifest"].core.canonical_state_hash,
        "model_seed": config.seed,
        "target_operations_physical_ids": dict(REC004A_PHYSICAL_IDS),
        "protected_operations_physical_ids": (
            stage_a["parent_audit"]["protected_operation_physical_ids"]
        ),
        "per_operation_route": {
            op: "FRESH_PAIRED_REBUILD" for op in REC004A_TARGET_OPERATIONS
        },
        "initialization_policy": (
            "torch.manual_seed(_derive_local_seed(seed, 0, f'rec004a_init:{op}')) immediately "
            "before constructing each fresh CrossPositionPrimitive; the per-step *training data* "
            "stream (_derive_local_seed(seed, step, f'train:{op}')) is identical to the original "
            "REC-004 1000-step run's stream since it depends only on (seed, step, operation), "
            "not on construction order -- only the primitive's own weight-init draw differs from "
            "the original run's, which is disclosed, not treated as bit-identical."
        ),
        "optimizer": "torch.optim.AdamW",
        "operator_lr": REC004A_OPERATOR_LR,
        "operator_weight_decay": REC004A_OPERATOR_WEIGHT_DECAY,
        "operator_grad_clip": REC004A_OPERATOR_GRAD_CLIP,
        "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "scheduler_t_max": REC004A_SCHEDULER_T_MAX,
        "scheduler_eta_min": REC004A_SCHEDULER_ETA_MIN,
        "lr_schedule_extension_policy": (
            "MECHANICAL_EXTENSION_FIXED_T_MAX: T_max stays fixed at 1000 (the original "
            "per-operation recipe's own steps=1000), and the SAME scheduler object is simply "
            "stepped past its nominal length up to 6000 total steps, rather than being "
            "reconstructed with a larger T_max. This keeps the first 1000 steps' LR sequence "
            "byte-identical to a plain 1000-step run (CosineAnnealingLR's closed-form value at "
            "step t depends only on t and T_max, never on how many further steps follow), "
            "satisfying the task doc's requirement not to let the 1000-step and 6000-step "
            "conditions diverge over their shared first 1000 steps. Steps 1001-6000 follow the "
            "same cosine formula's natural periodic continuation (period 2*T_max=2000) -- this "
            "is disclosed as a mechanical, no-redesign extension of the existing absolute-step "
            "schedule, not a new LR design."
        ),
        "recipe_source": (
            "runs/phase_b_b2_model_bundle_recovery/rec003/run_001/training_budget_manifest.json"
            "#INCREMENTAL_6_BUILD, cross-checked against live UnifiedBenchmarkConfig() defaults"
        ),
        "step_ladder": list(config.step_ladder),
        "step_definition": "optimizer update count (one call to optimizer.step() per ladder unit)",
        "budget_cap_per_operation": REC004A_BUDGET_CAP_PER_OPERATION,
        "budget_cap_total": REC004A_BUDGET_CAP_PER_OPERATION * len(REC004A_TARGET_OPERATIONS),
        "examples_per_step": REC004A_EXAMPLES_PER_STEP,
        "data_roles": {
            "train": {
                "purpose": "online per-step training stream",
                "seed_formula": "_derive_local_seed(seed, step, f'train:{op}')",
                "used_for_checkpoint_selection": False,
            },
            "train_fit": {
                "purpose": "diagnostic only: fixed already-trained subset",
                "n_examples": REC004A_TRAIN_FIT_STEP_COUNT * REC004A_EXAMPLES_PER_STEP,
                "source_steps": list(range(1, REC004A_TRAIN_FIT_STEP_COUNT + 1)),
                "used_for_checkpoint_selection": False,
            },
            "budget_validation": {
                "purpose": "checkpoint selection only",
                "n_examples": config.validation_examples,
                "split": REC004A_BUDGET_VALIDATION_SPLIT,
                "seed": config.seed,
                "fixed_across_all_checkpoints": True,
                "used_for_checkpoint_selection": True,
            },
            "recheck_query": {
                "purpose": "one-time independent RG3 recheck, post-selection only",
                "n_examples": config.recheck_query_examples,
                "split": REC004A_RECHECK_QUERY_SPLIT,
                "seed": config.seed,
                "used_for_checkpoint_selection": False,
            },
        },
        "selection_rule": (
            "first step in step_ladder with budget_validation.correct_exact_match >= "
            f"{config.validation_floor}; null if none clear the floor by the budget cap"
        ),
        "metric_definitions": {
            "sequence_exact_match": (
                "prediction tuple == target tuple (post-argmax discrete tokens)"
            ),
            "token_accuracy": "position-wise correct token count / total target token count",
            "causal_gap": "correct_exact_match - max(wrong_family_exact_match, none_exact_match)",
        },
        "error_strata": {
            "CYCLE_FOUR": "by_length, by_length_mod_4 (group-boundary/remainder)",
            "ROTATE_TRIPLETS": "by_length, by_length_mod_3 (group-boundary/remainder)",
            "SWAP_ENDS": "by_length, changed_position_accuracy (positions 0 and n-1 only)",
            "MIRROR_HALVES": "by_length, train_fit vs budget_validation comparison",
        },
        "rg3_continuity": (
            "Inherits EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md section 2.2/6 "
            "(non-SHIFT-15 floor EM>=0.95 per operation, 1024-example direct query, SHIFT "
            "COHERENT_LIMITED exception) unmodified; this protocol adds only the finite "
            "budget-calibration procedure for the 4 named operations."
        ),
        "protected_operations_frozen": list(REC004A_PROTECTED_OPERATIONS),
        "forbidden_changes": [
            "Core retraining",
            "operator architecture/size change",
            "loss/LR-formula/optimizer/batch/dtype change",
            "router recalibration",
            "argument scorer recalibration",
            "excluding a failing operation from the floor",
            "extending SHIFT's exception to another operation",
        ],
    }
    protocol_for_hash = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_for_hash, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    protocol["protocol_hash"] = protocol_hash

    (config.output_dir / "budget_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    return protocol


# =============================================================================
# Training-data generation -- exact reproduction of _train_single_primitive's
# non-parameterized branch (unified_oracle_causal_benchmark.py:534-561). All
# 4 REC-004A target operations are parameter-free, so this is the only
# branch needed. Extracted (not imported) because the original is inline
# code inside that function, not a separate callable; computation, data
# ordering, and RNG derivation are unchanged.
# =============================================================================


def _generate_step_training_examples(
    seed: int,
    step: int,
    operation: str,
    *,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
    n: int = REC004A_EXAMPLES_PER_STEP,
) -> list[Example]:
    step_rng = random.Random(
        _derive_local_seed(seed, step, f"{REC004A_TRAIN_SPLIT_LABEL}:{operation}")
    )
    op_obj = get_operation(operation)
    examples: list[Example] = []
    for _ in range(n):
        seq_len = step_rng.randint(sequence_length_range[0], sequence_length_range[1])
        seq = tuple(step_rng.randrange(vocab_size) for _ in range(seq_len))
        params = op_obj.sample_params(step_rng, seq, vocab_size)
        p_step = ProgramStep(operation=operation, params=params)
        prog = Program(steps=(p_step,))
        res = run_program(prog, seq, vocab_size)
        examples.append(
            Example(
                input_tokens=seq,
                target_tokens=res.output_tokens,
                program=prog,
                operation_graph=res.graph,
                category="known",
                split=REC004A_TRAIN_SPLIT_LABEL,
                vocab_size=vocab_size,
                task_spec=TaskSpec.from_program(prog),
                oracle_metadata=OracleMetadata(label="K", primitive_operations=(operation,)),
            )
        )
    return examples


def _evaluate_train_fit(
    core: Any,
    primitive: Any,
    operation: str,
    *,
    seed: int,
    vocab_size: int,
    sequence_length_range: tuple[int, int],
) -> dict[str, Any]:
    examples: list[Example] = []
    for step in range(1, REC004A_TRAIN_FIT_STEP_COUNT + 1):
        examples.extend(
            _generate_step_training_examples(
                seed,
                step,
                operation,
                vocab_size=vocab_size,
                sequence_length_range=sequence_length_range,
            )
        )
    exact, tok = _evaluate_primitive_arm(
        core, primitive, examples, operation, argument_provider=None
    )
    return {
        "n_examples": len(examples),
        "source_steps": list(range(1, REC004A_TRAIN_FIT_STEP_COUNT + 1)),
        "exposure_basis": (
            "regenerated via the exact per-step training-data seed formula for steps "
            "already trained through by every checkpoint (min ladder step 1000 >> 32)"
        ),
        "sequence_exact_match": exact,
        "token_accuracy": tok,
    }


def _length_stratified_breakdown(
    core: Any, primitive: Any, operation: str, *, seed: int, n_examples: int, split: str
) -> dict[str, Any]:
    examples = _generate_parameter_free_examples(
        seed,
        n_examples,
        operation=operation,
        split=split,
        vocab_size=10,
        sequence_length_range=(6, 10),
    )
    primitive.eval()
    device = core.device
    by_length: dict[int, dict[str, int]] = {}
    group_size = {"CYCLE_FOUR": 4, "ROTATE_TRIPLETS": 3}.get(operation)
    by_remainder: dict[int, dict[str, int]] = {}
    changed_position_correct = 0
    changed_position_total = 0
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            chunk = examples[start : start + 128]
            content_lengths = [len(ex.input_tokens) for ex in chunk]
            output_lengths = [get_operation(operation).output_length(n) for n in content_lengths]
            batch_input = collate_content_only_batch(chunk, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
            logits = primitive(h, content_lengths, output_lengths, None)
            preds = logits.argmax(dim=-1)
            for row, ex in enumerate(chunk):
                n = output_lengths[row]
                pred = tuple(preds[row, :n].tolist())
                target = ex.target_tokens
                length = len(ex.input_tokens)
                bucket = by_length.setdefault(length, {"n": 0, "exact": 0})
                bucket["n"] += 1
                is_exact = int(len(pred) == len(target) and pred == target)
                bucket["exact"] += is_exact
                if group_size:
                    rem = length % group_size
                    rbucket = by_remainder.setdefault(rem, {"n": 0, "exact": 0})
                    rbucket["n"] += 1
                    rbucket["exact"] += is_exact
                if operation == "SWAP_ENDS":
                    inp = ex.input_tokens
                    for i in range(min(len(inp), len(target))):
                        if inp[i] != target[i]:
                            changed_position_total += 1
                            if i < len(pred) and pred[i] == target[i]:
                                changed_position_correct += 1

    result: dict[str, Any] = {"by_length": {str(k): v for k, v in sorted(by_length.items())}}
    if group_size:
        result["group_size"] = group_size
        result["by_length_mod_group_size"] = {str(k): v for k, v in sorted(by_remainder.items())}
    if operation == "SWAP_ENDS":
        result["changed_position_denominator"] = changed_position_total
        result["changed_position_accuracy"] = (
            None
            if changed_position_total == 0
            else changed_position_correct / changed_position_total
        )
    return result


# =============================================================================
# Stage C -- train each of the 4 operations on one continuous trajectory,
# capturing checkpoints at the step ladder.
# =============================================================================


def _train_operation_with_checkpoints(
    core: Any,
    eval_bank: Any,
    op_to_id: dict[str, int],
    operation: str,
    *,
    config: IncrementalBudgetCalibrationConfig,
) -> dict[str, Any]:
    _guard_not_frozen(f"train_operation_with_checkpoints:{operation}")
    physical_id = REC004A_PHYSICAL_IDS[operation]
    device = core.device
    seed = config.seed
    vocab_size = 10
    sequence_length_range = (6, 10)

    torch.manual_seed(_derive_local_seed(seed, 0, f"rec004a_init:{operation}"))
    primitive = CrossPositionPrimitive(
        physical_id,
        CrossPositionPrimitiveConfig(
            operation=operation,
            d_model=core.model.config.d_model,
            d_operator=32,
            n_head=4,
            d_operator_ff=64,
            vocab_size=vocab_size,
            max_sequence_length=32,
        ),
        status=PrimitiveStatus.STABLE,
    )
    primitive.to(device)
    primitive.train()
    for p in primitive.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        primitive.parameters(), lr=REC004A_OPERATOR_LR, weight_decay=REC004A_OPERATOR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REC004A_SCHEDULER_T_MAX, eta_min=REC004A_SCHEDULER_ETA_MIN
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    max_steps = max(config.step_ladder)
    ladder = set(config.step_ladder)
    checkpoints: list[dict[str, Any]] = []
    checkpoint_state_dicts: dict[int, dict[str, torch.Tensor]] = {}
    cumulative_examples = 0
    running_loss = float("nan")
    t_start = time.time()
    diverged_at: int | None = None

    for step in range(1, max_steps + 1):
        examples = _generate_step_training_examples(
            seed,
            step,
            operation,
            vocab_size=vocab_size,
            sequence_length_range=sequence_length_range,
        )
        cumulative_examples += len(examples)
        content_lengths = [len(ex.input_tokens) for ex in examples]
        output_lengths = [get_operation(operation).output_length(n) for n in content_lengths]
        out_max = max(output_lengths)
        labels = _labels_for_examples(examples, output_lengths, out_max, device)

        with torch.no_grad():
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]

        optimizer.zero_grad(set_to_none=True)
        logits = primitive(h, content_lengths, output_lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primitive.parameters(), REC004A_OPERATOR_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        running_loss = float(loss.item())

        if not math.isfinite(running_loss):
            diverged_at = step
            break

        if step in ladder:
            state_snapshot = {
                k: v.detach().clone().cpu() for k, v in primitive.state_dict().items()
            }
            checkpoint_state_dicts[step] = state_snapshot

            live_primitive = eval_bank.get(physical_id)
            live_primitive.load_state_dict(
                {k: v.to(device) for k, v in state_snapshot.items()}, strict=True
            )
            live_primitive.eval()

            budget_validation = _evaluate_one_operation(
                core,
                eval_bank,
                op_to_id,
                operation,
                seed=seed,
                n_examples=config.validation_examples,
                split=REC004A_BUDGET_VALIDATION_SPLIT,
            )
            train_fit = _evaluate_train_fit(
                core, primitive, operation, seed=seed, vocab_size=vocab_size,
                sequence_length_range=sequence_length_range,
            )
            length_strata = _length_stratified_breakdown(
                core, primitive, operation, seed=seed,
                n_examples=config.validation_examples, split=REC004A_BUDGET_VALIDATION_SPLIT,
            )
            primitive.train()

            peak_vram = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            checkpoints.append(
                {
                    "step": step,
                    "cumulative_training_examples": cumulative_examples,
                    "unique_training_examples": cumulative_examples,
                    "train_loss_at_step": running_loss,
                    "learning_rate_at_step": scheduler.get_last_lr()[0],
                    "budget_validation": budget_validation,
                    "train_fit": train_fit,
                    "length_stratified": length_strata,
                    "checkpoint_state_hash": mb.canonical_state_hash(state_snapshot),
                    "cumulative_wall_clock_seconds": time.time() - t_start,
                    "peak_vram_bytes": peak_vram,
                }
            )

    for step in sorted(ladder):
        if step not in checkpoint_state_dicts and diverged_at is not None and step >= diverged_at:
            checkpoints.append(
                {
                    "step": step,
                    "status": "NOT_EXECUTED",
                    "reason": f"loss diverged at step {diverged_at}",
                }
            )

    checkpoints.sort(key=lambda c: c["step"])
    primitive.eval()
    return {
        "operation": operation,
        "physical_id": physical_id,
        "checkpoints": checkpoints,
        "checkpoint_state_dicts": checkpoint_state_dicts,
        "final_primitive": primitive,
        "diverged_at_step": diverged_at,
        "total_wall_clock_seconds": time.time() - t_start,
    }


def run_stage_c(
    config: IncrementalBudgetCalibrationConfig, core: Any, eval_bank: Any, op_to_id: dict[str, int]
) -> dict[str, dict[str, Any]]:
    _guard_not_frozen("run_stage_c")
    original_slices: dict[int, dict[str, torch.Tensor]] = {
        pid: {k: v.detach().clone() for k, v in eval_bank.get(pid).state_dict().items()}
        for pid in REC004A_PHYSICAL_IDS.values()
    }
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for op in REC004A_TARGET_OPERATIONS:
        pid = REC004A_PHYSICAL_IDS[op]
        eval_bank.get(pid).load_state_dict(original_slices[pid], strict=True)
        outcome = _train_operation_with_checkpoints(core, eval_bank, op_to_id, op, config=config)
        for step, sd in outcome["checkpoint_state_dicts"].items():
            torch.save(sd, checkpoint_dir / f"{op}_step{step}.pt")
        eval_bank.get(pid).load_state_dict(original_slices[pid], strict=True)
        results[op] = outcome

    return results


# =============================================================================
# Stage D -- candidate selection and child bundle construction.
# =============================================================================


def select_candidates(
    stage_c_results: dict[str, dict[str, Any]], floor: float
) -> tuple[dict[str, dict[str, Any]], bool]:
    selection: dict[str, dict[str, Any]] = {}
    for op, outcome in stage_c_results.items():
        selected_step = None
        all_em = {}
        for ckpt in outcome["checkpoints"]:
            if "budget_validation" not in ckpt:
                continue
            em = ckpt["budget_validation"]["correct_exact_match"]
            all_em[ckpt["step"]] = em
            if selected_step is None and em >= floor:
                selected_step = ckpt["step"]
        selection[op] = {
            "selected_step": selected_step,
            "all_checkpoint_validation_em": all_em,
            "diverged_at_step": outcome["diverged_at_step"],
        }
    all_selected = all(v["selected_step"] is not None for v in selection.values())
    return selection, all_selected


def _dedup_audit(
    stage_c_results: dict[str, dict[str, Any]], config: IncrementalBudgetCalibrationConfig
) -> dict[str, Any]:
    """Confirms train/budget_validation/recheck_query example sets for each
    target operation don't collide, by content digest (not merely by seed
    label difference)."""
    report: dict[str, Any] = {}
    for op in REC004A_TARGET_OPERATIONS:
        val_examples = _generate_parameter_free_examples(
            config.seed, config.validation_examples, operation=op,
            split=REC004A_BUDGET_VALIDATION_SPLIT, vocab_size=10, sequence_length_range=(6, 10),
        )
        recheck_examples = _generate_parameter_free_examples(
            config.seed, config.recheck_query_examples, operation=op,
            split=REC004A_RECHECK_QUERY_SPLIT, vocab_size=10, sequence_length_range=(6, 10),
        )
        val_digests = {(e.input_tokens, e.target_tokens) for e in val_examples}
        recheck_digests = {(e.input_tokens, e.target_tokens) for e in recheck_examples}
        train_digests: set[tuple[Any, Any]] = set()
        for step in range(1, REC004A_TRAIN_FIT_STEP_COUNT + 1):
            for e in _generate_step_training_examples(
                config.seed, step, op, vocab_size=10, sequence_length_range=(6, 10)
            ):
                train_digests.add((e.input_tokens, e.target_tokens))
        report[op] = {
            "train_fit_vs_budget_validation_overlap": len(train_digests & val_digests),
            "train_fit_vs_recheck_query_overlap": len(train_digests & recheck_digests),
            "budget_validation_vs_recheck_query_overlap": len(val_digests & recheck_digests),
        }
    return report


def build_child_bundle(
    config: IncrementalBudgetCalibrationConfig,
    parent_manifest: mb.ModelBundleManifest,
    stage_c_results: dict[str, dict[str, Any]],
    selection: dict[str, dict[str, Any]],
) -> tuple[mb.ModelBundleManifest, dict[str, Any]]:
    _guard_not_frozen("build_child_bundle")
    ns = _rec004a_namespace(config.seed)
    publish_dir = ns / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    core_copy = mb.legacy_import(
        Path(parent_manifest.core.file_path),
        publish_dir / "core",
        component_id=parent_manifest.core.component_id,
    )
    core_component = dataclasses.replace(
        core_copy, schema_hash=parent_manifest.core.schema_hash
    )
    vocab_copy = mb.legacy_import(
        Path(parent_manifest.vocabulary.file_path),
        publish_dir / "vocab",
        component_id=parent_manifest.vocabulary.component_id,
    )
    vocab_component = dataclasses.replace(
        vocab_copy, schema_hash=parent_manifest.vocabulary.schema_hash
    )

    router_dest = publish_dir / "router.pt"
    shutil.copy2(Path(parent_manifest.router.source_artifact), router_dest)
    router_component = dataclasses.replace(
        parent_manifest.router, source_artifact=str(router_dest.resolve())
    )

    scorer_dest = publish_dir / "argument_scorer.pt"
    shutil.copy2(Path(parent_manifest.argument_scorer.source_artifact), scorer_dest)
    scorer_component = dataclasses.replace(
        parent_manifest.argument_scorer, source_artifact=str(scorer_dest.resolve())
    )

    parent_bank_path = Path(_single_bank_path(parent_manifest))
    parent_bank_sd = dict(mb.load_state_dict(parent_bank_path))
    merged_sd = dict(parent_bank_sd)
    selection_receipts: dict[str, str] = {}
    for op, sel in selection.items():
        pid = REC004A_PHYSICAL_IDS[op]
        step = sel["selected_step"]
        chosen_sd = stage_c_results[op]["checkpoint_state_dicts"][step]
        for k, v in chosen_sd.items():
            merged_sd[f"_primitives.{pid}.{k}"] = v.cpu()
        val_em = sel["all_checkpoint_validation_em"][step]
        selection_receipts[op] = (
            f"REC-004A budget calibration, FRESH_PAIRED_REBUILD trajectory, "
            f"selected_step={step} (ladder {list(config.step_ladder)}), "
            f"budget_validation_sequence_EM={val_em:.4f}"
        )

    bank_path = (publish_dir / "primitive_bank_16.pt").resolve()
    torch.save(merged_sd, bank_path)

    primitives = []
    for entry in parent_manifest.primitives:
        if entry.operation_name in selection:
            sliced = mb.primitive_state_dict(merged_sd, entry.physical_id)
            weights_hash = mb.canonical_state_hash(sliced)
            primitives.append(
                dataclasses.replace(
                    entry,
                    version=f"{entry.version}-rec004a-calibrated",
                    state_abi_hash=weights_hash,
                    weights_hash=weights_hash,
                    source_artifact=str(bank_path),
                    provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD,
                    training_receipt=selection_receipts[entry.operation_name],
                )
            )
        else:
            primitives.append(dataclasses.replace(entry, source_artifact=str(bank_path)))

    system_info = get_system_info(seed=config.seed)
    manifest = mb.build_manifest(
        schema_version=parent_manifest.schema_version,
        source_commit=system_info.get("git_commit") or "unknown",
        runtime_recipe_version="rec004a-v1",
        environment_record={
            "python_version": str(system_info["python_version"]),
            "torch_version": str(system_info["torch_version"]),
            "device_name": str(system_info.get("device_name")),
        },
        model_id=parent_manifest.model_id,
        model_seed=parent_manifest.model_seed,
        training_run_id=f"rec004a-seed{config.seed}-run-001",
        parent_bundle_ids=(parent_manifest.bundle_id,),
        build_route=mb.BuildRoute.PARTIAL_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "diagnostic_only"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=core_component,
        vocabulary=vocab_component,
        primitives=tuple(primitives),
        router=router_component,
        argument_scorer=scorer_component,
        scoring_policy=parent_manifest.scoring_policy,
        build_recipe_hash=hashlib.sha256(b"B-C005REC-004A-budget-calibration-v1").hexdigest(),
        known_defects=("REC004A_AWAITING_RG3_RECHECK",),
        clean_build_exercised_stages=("INCREMENTAL_BUDGET_CALIBRATION",),
        qualification_refs=(REC004A_TASK_ID,),
    )

    self_load = mb.load_bundle(manifest, mode="nominal", expected_primitive_count=16)
    negative_test_passed = False
    try:
        mb.load_bundle(
            manifest, mode="nominal", required_capabilities=frozenset({"recovery_cohort_ready"}),
            expected_primitive_count=16,
        )
    except mb.CapabilityNotQualifiedError:
        negative_test_passed = True

    bundle_dir = (
        _repo_root() / "runs" / "phase_b_b2_model_bundle_recovery" / "bundles" / manifest.bundle_id
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = _manifest_to_json_dict(manifest)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")

    manifest_paths_file = publish_dir / "manifest_paths.json"
    op_to_id = {p.operation_name: p.physical_id for p in manifest.primitives}
    manifest_paths_file.write_text(
        json.dumps({"op_to_id": op_to_id, "manifest": manifest_json}, indent=2), encoding="utf-8"
    )

    report = {
        "bundle_id": manifest.bundle_id,
        "parent_bundle_id": parent_manifest.bundle_id,
        "content_manifest_digest": manifest.content_manifest_digest,
        "self_load_nominal_mode_checks_performed": list(self_load.checks_performed),
        "unearned_capability_request_rejected": negative_test_passed,
        "manifest_path": str(bundle_dir / "manifest.json"),
        "manifest_paths_json_for_fresh_process": str(manifest_paths_file.resolve()),
        "bank_path": str(bank_path),
        "core_path": core_component.file_path,
        "router_path": str(router_dest.resolve()),
        "scorer_path": str(scorer_dest.resolve()),
    }
    return manifest, report


def run_protected_regression_check(
    config: IncrementalBudgetCalibrationConfig,
    parent_manifest: mb.ModelBundleManifest,
    child_manifest: mb.ModelBundleManifest,
    core: Any,
) -> dict[str, Any]:
    """Independently confirms the 12 protected primitives are byte-identical
    between parent and child (hash + same-input prediction agreement)."""
    parent_by_op = {p.operation_name: p for p in parent_manifest.primitives}
    child_by_op = {p.operation_name: p for p in child_manifest.primitives}

    hash_matches: dict[str, bool] = {}
    for op in REC004A_PROTECTED_OPERATIONS:
        hash_matches[op] = (
            parent_by_op[op].weights_hash == child_by_op[op].weights_hash
            and parent_by_op[op].core_dependency_hash == child_by_op[op].core_dependency_hash
        )

    parent_bank_sd = mb.load_state_dict(Path(_single_bank_path(parent_manifest)))
    parent_bank, parent_op_to_id = _reconstruct_16_op_bank_structure(core, config.seed)
    parent_bank.to(core.device)
    parent_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in parent_manifest.primitives
            for k, v in mb.primitive_state_dict(parent_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    parent_bank.freeze_all()
    parent_bank.eval()

    child_bank_sd = mb.load_state_dict(Path(_single_bank_path(child_manifest)))
    child_bank, child_op_to_id = _reconstruct_16_op_bank_structure(core, config.seed)
    child_bank.to(core.device)
    child_bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(child_bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    child_bank.freeze_all()
    child_bank.eval()

    prediction_matches: dict[str, bool] = {}
    for op in REC004A_PROTECTED_OPERATIONS:
        examples = _generate_parameter_free_examples(
            config.seed, 32, operation=op, split="rec004a_protected_regression",
            vocab_size=10, sequence_length_range=(6, 10),
        )
        parent_preds = _predict_tokens(
            core, parent_bank.get(parent_op_to_id[op]), examples, op, None
        )
        child_preds = _predict_tokens(core, child_bank.get(child_op_to_id[op]), examples, op, None)
        prediction_matches[op] = parent_preds == child_preds

    all_protected_unchanged = all(hash_matches.values()) and all(prediction_matches.values())
    return {
        "task_id": REC004A_TASK_ID,
        "protected_operations": list(REC004A_PROTECTED_OPERATIONS),
        "hash_matches": hash_matches,
        "same_input_prediction_matches": prediction_matches,
        "all_protected_unchanged": all_protected_unchanged,
    }


# =============================================================================
# Stage E -- independent recheck_query + fresh-process reproducibility.
# =============================================================================


def run_stage_e(
    config: IncrementalBudgetCalibrationConfig,
    core: Any,
    child_manifest: mb.ModelBundleManifest,
    manifest_paths_file: str,
) -> dict[str, Any]:
    _guard_not_frozen("run_stage_e")
    op_to_id = {p.operation_name: p.physical_id for p in child_manifest.primitives}

    bank, fresh_op_to_id = _reconstruct_16_op_bank_structure(core, config.seed)
    if fresh_op_to_id != op_to_id:
        raise mb.IncompleteBundleError(
            "child bank structure op_to_id mismatch during Stage E reload"
        )
    bank.to(core.device)
    bank_sd = mb.load_state_dict(Path(_single_bank_path(child_manifest)))
    bank.load_state_dict(
        {
            f"_primitives.{p.physical_id}.{k}": v
            for p in child_manifest.primitives
            for k, v in mb.primitive_state_dict(bank_sd, p.physical_id).items()
        },
        strict=True,
    )
    bank.freeze_all()
    bank.eval()

    with frozen_evaluation():
        rows = {
            op: _evaluate_one_operation(
                core, bank, op_to_id, op, seed=config.seed,
                n_examples=config.recheck_query_examples, split=REC004A_RECHECK_QUERY_SPLIT,
            )
            for op in sorted(op_to_id)
        }
        for op in rows:
            if op == "SHIFT":
                rows[op]["recovery_floor_status"] = "COHERENT_LIMITED"
                rows[op]["recovery_floor_passed"] = None
            else:
                rows[op]["recovery_floor_status"] = "EVALUATED"
                rows[op]["recovery_floor_passed"] = (
                    rows[op]["correct_exact_match"] >= config.validation_floor
                )

        in_process_predictions: dict[str, list[list[int]]] = {}
        for op in sorted(op_to_id):
            examples = _generate_parameter_free_examples(
                config.seed, config.recheck_query_examples, operation=op,
                split=REC004A_RECHECK_QUERY_SPLIT, vocab_size=10, sequence_length_range=(6, 10),
            )
            in_process_predictions[op] = _predict_tokens(
                core, bank.get(op_to_id[op]), examples, op, None
            )

    non_shift_ops = [op for op in rows if op != "SHIFT"]
    all_non_shift_floor_pass = all(bool(rows[op]["recovery_floor_passed"]) for op in non_shift_ops)

    all_primitive_execution = {
        "task_id": REC004A_TASK_ID,
        "non_shift_floor": config.validation_floor,
        "recheck_query_examples_per_operation": config.recheck_query_examples,
        "operations": rows,
        "all_16_operations_present": set(rows) == set(op_to_id) and len(rows) == 16,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
        "non_shift_floor_failures": [
            op for op in non_shift_ops if not rows[op]["recovery_floor_passed"]
        ],
        "shift_status": "COHERENT_LIMITED",
        "shift_correct_exact_match": rows.get("SHIFT", {}).get("correct_exact_match"),
    }

    script_path = _repo_root() / "scripts" / "rec004_fresh_process_check.py"
    scratch_cwd = Path(tempfile.gettempdir()) / f"apc_rec004a_fresh_process_cwd_seed{config.seed}"
    scratch_cwd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, str(script_path),
            "--manifest-paths", manifest_paths_file,
            "--seed", str(config.seed),
            "--sample-examples-per-op", str(config.recheck_query_examples),
            "--split", REC004A_RECHECK_QUERY_SPLIT,
        ],
        cwd=str(scratch_cwd), capture_output=True, text=True, timeout=1800, check=False,
        env=dict(os.environ),
    )
    wall = time.time() - t0

    if proc.returncode != 0:
        fresh_process_report = {
            "task_id": REC004A_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "subprocess_returncode": proc.returncode,
            "subprocess_stdout_tail": proc.stdout[-4000:],
            "subprocess_stderr_tail": proc.stderr[-4000:],
            "reproducibility_passed": False,
            "wall_clock_seconds": wall,
        }
    else:
        fresh = json.loads(proc.stdout.strip().splitlines()[-1])
        fresh_predictions = fresh["predictions_by_op"]
        mismatches = {
            op: {
                "in_process": in_process_predictions[op],
                "fresh_process": fresh_predictions.get(op),
            }
            for op in sorted(op_to_id)
            if in_process_predictions[op] != fresh_predictions.get(op)
        }
        fresh_process_report = {
            "task_id": REC004A_TASK_ID,
            "cwd_used": str(scratch_cwd),
            "repo_root_cwd": str(_repo_root()),
            "different_working_directory_confirmed": str(scratch_cwd) != str(_repo_root()),
            "sample_examples_per_op": config.recheck_query_examples,
            "subprocess_returncode": proc.returncode,
            "no_builder_identifier_found_in_subprocess_script": fresh[
                "no_builder_identifier_found_in_this_script"
            ],
            "load_bundle_checks_performed_in_subprocess": fresh["checks_performed_by_load_bundle"],
            "bundle_id_matches": fresh["bundle_id"] == child_manifest.bundle_id,
            "per_operation_prediction_mismatches": mismatches,
            "predictions_byte_exact_for_all_16_ops": not mismatches,
            "exact_match_by_op_in_subprocess": fresh["exact_match_by_op"],
            "reproducibility_passed": (
                not mismatches
                and bool(fresh["no_builder_identifier_found_in_this_script"])
                and fresh["bundle_id"] == child_manifest.bundle_id
            ),
            "wall_clock_seconds": wall,
        }

    return {
        "all_primitive_execution": all_primitive_execution,
        "fresh_process_report": fresh_process_report,
        "all_non_shift_15_floor_passed": all_non_shift_floor_pass,
    }


# =============================================================================
# Orchestration.
# =============================================================================


def run_incremental_budget_calibration_task(
    config: IncrementalBudgetCalibrationConfig,
) -> dict[str, Any]:
    start = time.time()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.seed
    if seed != RECOVERY_PILOT_SEED:
        raise ValueError(
            f"B-C005REC-004A's model seed is pre-registered as {RECOVERY_PILOT_SEED} "
            f"(same seed10 Core as REC-004); got seed={seed}"
        )

    before_hashes = _snapshot_forbidden_cache_hashes(seed)

    stage_a = run_stage_a_audit(config)
    protocol = build_budget_protocol(config, stage_a)
    core, eval_bank, op_to_id = _reconstruct_parent_runtime(config, stage_a["parent_manifest"])

    stage_c_results = run_stage_c(config, core, eval_bank, op_to_id)
    selection, all_selected = select_candidates(stage_c_results, config.validation_floor)
    dedup_audit = _dedup_audit(stage_c_results, config)

    learning_curve_path = output_dir / "learning_curve.jsonl"
    with learning_curve_path.open("w", encoding="utf-8") as fh:
        for op, outcome in stage_c_results.items():
            for ckpt in outcome["checkpoints"]:
                fh.write(json.dumps({"operation": op, **ckpt}, default=str) + "\n")

    error_breakdown = {
        op: [
            {"step": c["step"], "length_stratified": c.get("length_stratified")}
            for c in outcome["checkpoints"]
            if "length_stratified" in c
        ]
        for op, outcome in stage_c_results.items()
    }
    (output_dir / "error_breakdown.json").write_text(
        json.dumps(error_breakdown, indent=2, default=str), encoding="utf-8"
    )

    selected_recipe = {
        "task_id": REC004A_TASK_ID,
        "selection_rule": protocol["selection_rule"],
        "selection": selection,
        "dedup_audit": dedup_audit,
        "alternating_negate_increment_mod_unchanged": True,
    }
    (output_dir / "selected_recipe.json").write_text(
        json.dumps(selected_recipe, indent=2, default=str), encoding="utf-8"
    )

    calibration_status = "SELECTED_ALL" if all_selected else "VALIDATION_TARGET_NOT_MET"

    result: dict[str, Any] = {
        "implementation_status": "COMPLETE",
        "calibration_status": calibration_status,
        "rg3_recheck": "NOT_EXECUTED",
        "stage_a": {
            "parent_audit": stage_a["parent_audit"],
            "resume_audit": stage_a["resume_audit"],
            "operation_contract_audit": stage_a["operation_contract_audit"],
            "existing_learning_curve_audit": stage_a["existing_learning_curve_audit"],
            "qualification_scope_audit": stage_a["qualification_scope_audit"],
        },
        "budget_protocol": protocol,
        "selected_recipe": selected_recipe,
        "wall_clock_seconds": time.time() - start,
    }

    if not all_selected:
        after_hashes = _snapshot_forbidden_cache_hashes(seed)
        side_effect_audit = {
            "task_id": REC004A_TASK_ID,
            "before": before_hashes,
            "after": after_hashes,
            "shared_cache_unchanged": before_hashes == after_hashes,
        }
        (output_dir / "side_effect_audit.json").write_text(
            json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
        )
        result["side_effect_audit"] = side_effect_audit
        (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
        return result

    with frozen_evaluation():
        child_manifest, publish_report = build_child_bundle(
            config, stage_a["parent_manifest"], stage_c_results, selection
        )
        protected_regression = run_protected_regression_check(
            config, stage_a["parent_manifest"], child_manifest, core
        )

    (output_dir / "recipe_revision.json").write_text(
        json.dumps(
            {
                "task_id": REC004A_TASK_ID,
                "parent_recipe": (
                    "runs/phase_b_b2_model_bundle_recovery/rec003/run_001/"
                    "training_budget_manifest.json#INCREMENTAL_6_BUILD"
                ),
                "revised_operations": {op: sel["selected_step"] for op, sel in selection.items()},
                "unchanged_operations": list(REC004A_FROZEN_CONTROL_OPERATIONS),
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "bundle_lineage.json").write_text(
        json.dumps(
            {
                "child_bundle_id": child_manifest.bundle_id,
                "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
                "parent_bundle_ids_declared": list(child_manifest.parent_bundle_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "protected_regression.json").write_text(
        json.dumps(protected_regression, indent=2, default=str), encoding="utf-8"
    )

    stage_e = run_stage_e(
        config, core, child_manifest, publish_report["manifest_paths_json_for_fresh_process"]
    )

    after_hashes = _snapshot_forbidden_cache_hashes(seed)
    side_effect_audit = {
        "task_id": REC004A_TASK_ID,
        "before": before_hashes,
        "after": after_hashes,
        "shared_cache_unchanged": before_hashes == after_hashes,
    }

    rg3_recheck_pass = all(
        [
            stage_e["all_primitive_execution"]["all_16_operations_present"],
            stage_e["all_non_shift_15_floor_passed"],
            stage_e["fresh_process_report"]["reproducibility_passed"],
            side_effect_audit["shared_cache_unchanged"],
            protected_regression["all_protected_unchanged"],
        ]
    )
    rg3_recheck = "RG3_RECHECK_PASS" if rg3_recheck_pass else "RG3_RECHECK_FAIL"

    final_known_defects = () if rg3_recheck_pass else ("REC004A_RG3_RECHECK_FAIL",)
    final_manifest = dataclasses.replace(child_manifest, known_defects=final_known_defects)
    bundle_dir = (
        _repo_root()
        / "runs"
        / "phase_b_b2_model_bundle_recovery"
        / "bundles"
        / final_manifest.bundle_id
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(_manifest_to_json_dict(final_manifest), indent=2), encoding="utf-8"
    )

    rg3_recheck_report = {
        "task_id": REC004A_TASK_ID,
        "result": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "parent_bundle_id": stage_a["parent_manifest"].bundle_id,
        "criteria": {
            "all_16_operations_coverage": (
                stage_e["all_primitive_execution"]["all_16_operations_present"]
            ),
            "non_shift_15_floor_passed": stage_e["all_non_shift_15_floor_passed"],
            "non_shift_floor_failures": (
                stage_e["all_primitive_execution"]["non_shift_floor_failures"]
            ),
            "fresh_process_reproducibility_passed": (
                stage_e["fresh_process_report"]["reproducibility_passed"]
            ),
            "shared_cache_unchanged": side_effect_audit["shared_cache_unchanged"],
            "protected_12_unchanged": protected_regression["all_protected_unchanged"],
        },
        "no_query_leakage": {
            "budget_validation_split": REC004A_BUDGET_VALIDATION_SPLIT,
            "recheck_query_split": REC004A_RECHECK_QUERY_SPLIT,
            "distinct_splits": REC004A_BUDGET_VALIDATION_SPLIT != REC004A_RECHECK_QUERY_SPLIT,
        },
        "scope_note": (
            "This is a seed-10 recovery-pilot recheck with a revised operation-specific "
            "budget, not a 5-model cohort, unseen-family, or hard-negative routing result."
        ),
    }
    (output_dir / "rg3_recheck.json").write_text(
        json.dumps(rg3_recheck_report, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "all_primitive_execution.json").write_text(
        json.dumps(stage_e["all_primitive_execution"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "fresh_process_report.json").write_text(
        json.dumps(stage_e["fresh_process_report"], indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "side_effect_audit.json").write_text(
        json.dumps(side_effect_audit, indent=2, default=str), encoding="utf-8"
    )

    qualification = {
        "task_id": REC004A_TASK_ID,
        "implementation_status": "COMPLETE",
        "calibration_status": calibration_status,
        "rg3_recheck": rg3_recheck,
        "child_bundle_id": child_manifest.bundle_id,
        "rec005_eligible": rg3_recheck_pass,
        "no_5_model_cohort_started_by_this_task": True,
    }
    (output_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2, default=str), encoding="utf-8"
    )

    result["rg3_recheck"] = rg3_recheck
    result["child_bundle_id"] = child_manifest.bundle_id
    result["publish_report"] = publish_report
    result["protected_regression"] = protected_regression
    result["rg3_recheck_report"] = rg3_recheck_report
    result["side_effect_audit"] = side_effect_audit
    result["qualification"] = qualification
    result["wall_clock_seconds"] = time.time() - start

    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result
