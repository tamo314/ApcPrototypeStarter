"""Fail-closed executor for the preregistered Phase-D SORT repair pilot.

This module intentionally has one configuration only.  It is not a benchmark
framework: changing a seed, panel, recipe, budget, or output namespace raises
before model construction.  The public dry-run is read-only and is used to
review the exact immutable inputs before the one authorized cohort run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from apc.core.data import IGNORE_INDEX, collate_content_only_batch
from apc.environments.generator import Example, OracleMetadata, Program, ProgramStep
from apc.environments.interpreter import run_program
from apc.environments.operations import (
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
    get_operation,
)
from apc.environments.task_spec import TaskSpec
from apc.evaluation.learned_routing_benchmark import (
    LearnedRoutingBenchmarkConfig,
    _ensure_learned_routing_bank_and_core,
)
from apc.evaluation.phase_d_seed_registry import audit_phase_d_seed_registry
from apc.evaluation.shift_functional_generalization_repair import (
    ShiftFunctionalGeneralizationRepairConfig,
    _train_shift_candidate,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    NATURAL_BASELINES,
    WRONG_FAMILY_MAP,
    UnifiedBenchmarkConfig,
    _generate_parameter_free_examples,
    _labels_for_examples,
    _train_single_primitive,
)
from apc.primitives.composition import execute_composition_recipe
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus
from apc.utils import model_bundle as mb
from apc.utils.seed import set_seed
from apc.utils.system_info import get_system_info

REPO_ROOT = Path(__file__).resolve().parents[3]
COHORT_ID = "phase_d_five_model_cohort_v2"
MODEL_SEEDS = (40, 41, 42, 43, 44)
EVAL_SEEDS = (301, 302, 303, 304, 305)
REPAIR_STEPS = 6_000
REQUIRED_LOAD_CHECKS = frozenset(
    {
        "manifest_self_consistency",
        "core_integrity",
        "schema_integrity",
        "primitive_integrity_and_core_dependency",
        "router_integrity",
        "argument_scorer_integrity",
    }
)
CAUSAL_LENGTH_GROUPS = (("target_L3_L5", (3, 5)), ("regression_L6_L10", (6, 10)))
CAUSAL_ARMS = ("correct", "wrong_family", "none")
TARGET_CLASSES = (
    "NEGATE->SELECT->SORT",
    "SELECT->SORT->BIND",
    "SELECT->SORT->NEGATE",
    "SELECT->SORT->REVERSE",
    "SELECT->SORT->SELECT",
    "SELECT->SORT->SHIFT",
    "SHIFT->SELECT->SORT",
)
REGRESSION_CLASSES = (
    "NEGATE->SORT->BIND",
    "NEGATE->SORT->SELECT",
    "NEGATE->SORT->SHIFT",
    "SORT->NEGATE->SELECT",
    "SORT->NEGATE->SHIFT",
    "SORT->REVERSE->BIND",
    "SORT->REVERSE->SELECT",
    "SORT->REVERSE->SHIFT",
    "SORT->SELECT->BIND",
    "SORT->SHIFT->BIND",
    "SORT->SHIFT->SELECT",
)
CANARY_CLASSES = (
    "NEGATE->REVERSE->BIND",
    "NEGATE->REVERSE->SELECT",
    "NEGATE->SELECT->BIND",
    "NEGATE->SHIFT->BIND",
    "NEGATE->SHIFT->SELECT",
    "REVERSE->SELECT->BIND",
    "REVERSE->SHIFT->BIND",
    "SHIFT->SELECT->BIND",
)
PREREG_FILES = (
    "docs/DECISIONS_PHASE_D.md",
    "docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md",
    "docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json",
    "docs/phase_d/PHASE_D_D005_SEED_REGISTRY.json",
    "docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md",
)


class PhaseDStopGateError(RuntimeError):
    """A preregistered condition failed; callers must not continue to repair."""


@dataclass(frozen=True)
class PhaseDExecutorConfig:
    """The one D-005-authorized configuration; every field is fail-closed."""

    output_root: Path = Path("runs/phase_d_d008_executor")
    cohort_root: Path = Path("runs/phase_d_five_model_cohort")
    candidate_root: Path = Path("runs/phase_d_d001_sort_repair")
    model_seeds: tuple[int, ...] = MODEL_SEEDS
    evaluation_seeds: tuple[int, ...] = EVAL_SEEDS
    repair_steps: int = REPAIR_STEPS
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model_seeds != MODEL_SEEDS:
            raise PhaseDStopGateError("D-005 authorizes exactly model seeds 40-44")
        if self.evaluation_seeds != EVAL_SEEDS:
            raise PhaseDStopGateError("D-001 fixes evaluation seeds 301-305")
        if self.repair_steps != REPAIR_STEPS:
            raise PhaseDStopGateError("D-001 fixes LOCAL_SORT_REPAIR at exactly 6,000 updates")
        if self.device != "cuda":
            raise PhaseDStopGateError("Phase D execution requires the registered CUDA device")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _supported_torch_version(version: str) -> bool:
    """Keep D-006's project dependency boundary executable, not documentary."""
    release = version.split("+", 1)[0].split(".")
    try:
        major, minor = int(release[0]), int(release[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 12) and (major, minor) < (2, 14)


def _json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _manifest_to_json_dict(manifest: mb.ModelBundleManifest) -> dict[str, Any]:
    """Serialize every manifest field needed by the independent loader."""
    def component(value: mb.ComponentManifest | None) -> dict[str, Any] | None:
        return None if value is None else dataclasses.asdict(value)

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
        "core": component(manifest.core),
        "vocabulary": component(manifest.vocabulary),
        "primitives": [
            {**dataclasses.asdict(entry), "provenance_status": entry.provenance_status.value}
            for entry in manifest.primitives
        ],
        "router": dataclasses.asdict(manifest.router),
        "argument_scorer": dataclasses.asdict(manifest.argument_scorer),
        "scoring_policy": dataclasses.asdict(manifest.scoring_policy),
        "task_encoder": component(manifest.task_encoder),
        "query_projection": component(manifest.query_projection),
        "decoder": component(manifest.decoder),
        "controller_verifier_signature": manifest.controller_verifier_signature,
        "build_recipe_hash": manifest.build_recipe_hash,
        "dataset_role_hashes": dict(manifest.dataset_role_hashes),
        "generator_version": manifest.generator_version,
        "known_defects": list(manifest.known_defects),
        "exposure_manifest": dict(manifest.exposure_manifest),
        "clean_build_exercised_stages": list(manifest.clean_build_exercised_stages),
        "qualification_refs": list(manifest.qualification_refs),
        "cohort_id": manifest.cohort_id,
        "cohort_member_seeds": list(manifest.cohort_member_seeds),
        "cohort_construction_recipe_hash": manifest.cohort_construction_recipe_hash,
    }


def manifest_from_json_dict(raw: dict[str, Any]) -> mb.ModelBundleManifest:
    """Rehydrate a manifest without trusting paths outside its declaration."""
    def component(value: dict[str, Any] | None) -> mb.ComponentManifest | None:
        return None if value is None else mb.ComponentManifest(**value)

    return mb.ModelBundleManifest(
        schema_version=int(raw["schema_version"]),
        bundle_id=str(raw["bundle_id"]),
        content_manifest_digest=str(raw["content_manifest_digest"]),
        source_commit=str(raw["source_commit"]),
        runtime_recipe_version=str(raw["runtime_recipe_version"]),
        environment_record=dict(raw["environment_record"]),
        model_id=str(raw["model_id"]),
        model_seed=int(raw["model_seed"]),
        training_run_id=str(raw["training_run_id"]),
        parent_bundle_ids=tuple(raw["parent_bundle_ids"]),
        build_route=mb.BuildRoute(raw["build_route"]),
        scope=mb.BundleScope(raw["scope"]),
        requested_capabilities=frozenset(raw["requested_capabilities"]),
        publish_status=mb.PublishStatus(raw["publish_status"]),
        core=mb.ComponentManifest(**raw["core"]),
        vocabulary=mb.ComponentManifest(**raw["vocabulary"]),
        primitives=tuple(
            mb.PrimitiveManifestEntry(
                **{**entry, "provenance_status": mb.ProvenanceStatus(entry["provenance_status"])}
            )
            for entry in raw["primitives"]
        ),
        router=mb.RouterManifest(**raw["router"]),
        argument_scorer=mb.ArgumentScorerManifest(**raw["argument_scorer"]),
        scoring_policy=mb.ScoringPolicyManifest(**raw["scoring_policy"]),
        task_encoder=component(raw.get("task_encoder")),
        query_projection=component(raw.get("query_projection")),
        decoder=component(raw.get("decoder")),
        controller_verifier_signature=raw.get("controller_verifier_signature"),
        build_recipe_hash=raw.get("build_recipe_hash"),
        dataset_role_hashes=dict(raw.get("dataset_role_hashes", {})),
        generator_version=raw.get("generator_version"),
        known_defects=tuple(raw.get("known_defects", ())),
        exposure_manifest=dict(raw.get("exposure_manifest", {})),
        clean_build_exercised_stages=tuple(raw.get("clean_build_exercised_stages", ())),
        qualification_refs=tuple(raw.get("qualification_refs", ())),
        cohort_id=raw.get("cohort_id"),
        cohort_member_seeds=tuple(raw.get("cohort_member_seeds", ())),
        cohort_construction_recipe_hash=raw.get("cohort_construction_recipe_hash"),
    )


def _assert_new_namespace(path: Path) -> None:
    if path.exists():
        raise PhaseDStopGateError(f"namespace already exists and cannot be overwritten: {path}")


def _static_gate(config: PhaseDExecutorConfig) -> dict[str, Any]:
    registry = audit_phase_d_seed_registry()
    if tuple(registry["candidate_model_seeds"]) != config.model_seeds:
        raise PhaseDStopGateError("seed registry does not declare the authorized cohort")
    if not torch.cuda.is_available():
        raise PhaseDStopGateError("CUDA runtime unavailable; no model/data access permitted")
    if not _supported_torch_version(torch.__version__):
        raise PhaseDStopGateError(
            f"torch {torch.__version__} is outside the registered project range >=2.12,<2.14"
        )
    if tuple(torch.version.cuda or "") == ():  # defensive: no CUDA build is ineligible
        raise PhaseDStopGateError("Torch has no CUDA build")
    for relative in PREREG_FILES:
        if not (REPO_ROOT / relative).is_file():
            raise PhaseDStopGateError(f"missing preregistration evidence: {relative}")
    occupied_namespaces = [
        str(path)
        for path in (config.output_root, config.cohort_root, config.candidate_root)
        if (REPO_ROOT / path).exists()
    ]
    if occupied_namespaces:
        raise PhaseDStopGateError(
            "registered Phase-D namespace already exists and cannot be reused: "
            f"{occupied_namespaces}"
        )
    # Generic caches would make the helper restore rather than fresh-build.
    stale = [
        str(Path("runs/phase_a1_shift_compact_structural_probe") / f"seed_{seed}")
        for seed in config.model_seeds
        if (REPO_ROOT / "runs/phase_a1_shift_compact_structural_probe" / f"seed_{seed}").exists()
    ]
    if stale:
        raise PhaseDStopGateError(f"fresh-Core gate failed; shared cache exists: {stale}")
    return {
        "status": "PASS",
        "seed_registry": registry,
        "cuda": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "preregistration_hashes": {p: _sha256(REPO_ROOT / p) for p in PREREG_FILES},
        "sealed_access": 0,
    }


def dry_run_manifest(config: PhaseDExecutorConfig | None = None) -> dict[str, Any]:
    """Read-only contract review; does not create a namespace or initialize a model."""
    effective_config = config or PhaseDExecutorConfig()
    gate = _static_gate(effective_config)
    return {
        "task": "D-008",
        "mode": "DRY_RUN",
        "authorization": "ADR-0170 as amended by ADR-0172",
        "config": {
            **asdict(effective_config),
            "output_root": str(effective_config.output_root),
            "cohort_root": str(effective_config.cohort_root),
            "candidate_root": str(effective_config.candidate_root),
        },
        "conditions": ("FROZEN_PARENT", "LOCAL_SORT_REPAIR", "SYMBOLIC_REFERENCE"),
        "allowed_panels": {
            "target": TARGET_CLASSES,
            "regression": REGRESSION_CLASSES,
            "canary": CANARY_CLASSES,
            "causal": ("Correct", "Wrong-family", "None"),
        },
        "information_boundary": {
            "task_conditioned_core": False,
            "oracle_coordinate_input": False,
            "ground_truth_intermediate_injection": False,
            "symbolic_execution_fallback": False,
            "sealed_access": 0,
        },
        "static_gate": gate,
    }


def _vocab_state(core: Any) -> dict[str, torch.Tensor]:
    tokens = core.tokens
    return {
        "vocab_size": torch.tensor([tokens.env_vocab_size]),
        "op_base": torch.tensor([tokens.op_base]),
        "arg_base": torch.tensor([tokens.arg_base]),
        "arg_span": torch.tensor([tokens.arg_span]),
        "num_operations": torch.tensor([tokens.num_operations]),
    }


def _schema_hash(core: Any) -> str:
    t = core.tokens
    return f"vocab{t.env_vocab_size}_ops{t.num_operations}_argspan{t.arg_span}_v1"


def _publish_parent(
    seed: int,
    seed_dir: Path,
    core: Any,
    bank: Any,
    router: Any,
    scorer: Any,
    op_to_id: dict[str, int],
) -> tuple[mb.ModelBundleManifest, Path]:
    publish = seed_dir / "parent"
    publish.mkdir(parents=True)
    core_path = seed_dir / "canonical_branch_b" / "shared_encoder.pt"
    if not core_path.is_file():
        raise PhaseDStopGateError(f"fresh Core artifact missing: {core_path}")
    bank_path, router_path, scorer_path, vocab_path = (
        publish / "primitive_bank_16.pt",
        publish / "router.pt",
        publish / "argument_scorer.pt",
        publish / "vocabulary.pt",
    )
    torch.save(bank.state_dict(), bank_path)
    torch.save(router.state_dict(), router_path)
    torch.save(scorer.state_dict(), scorer_path)
    torch.save(_vocab_state(core), vocab_path)
    bank_structure_path = publish / "primitive_bank_structure.json"
    _json_write(bank_structure_path, bank.to_manifest())
    core_raw, core_hash = mb.canonical_state_hash_from_file(core_path)
    vocab_raw, vocab_hash = mb.canonical_state_hash_from_file(vocab_path)
    schema = _schema_hash(core)
    bank_sd = mb.load_state_dict(bank_path)
    primitives: list[mb.PrimitiveManifestEntry] = []
    for operation, pid in sorted(op_to_id.items(), key=lambda row: row[1]):
        state = mb.primitive_state_dict(bank_sd, pid)
        weight_hash = mb.canonical_state_hash(state)
        primitives.append(
            mb.PrimitiveManifestEntry(
                physical_id=pid,
                operation_name=operation,
                version=f"phase-d-parent-seed{seed}-v1",
                architecture_signature="shift_relative_v1"
                if operation == "SHIFT"
                else "cross_position_v1",
                state_abi_hash=mb.compute_state_abi_hash(
                    state,
                    architecture_signature=(
                        "shift_relative_v1" if operation == "SHIFT" else "cross_position_v1"
                    ),
                ),
                core_dependency_hash=core_hash,
                decoder_dependency_hash="NOT_APPLICABLE_NO_SEPARATE_DECODER_COMPONENT",
                weights_hash=weight_hash,
                source_artifact=str(bank_path.resolve()),
                provenance_status=mb.ProvenanceStatus.TRAINED_THIS_BUILD,
                argument_schema_hash="argument_scorer_schema_v1"
                if operation in {"SELECT", "COUNT", "BIND", "SHIFT"}
                else None,
                training_receipt="D-008 fresh Phase-D cohort build",
            )
        )
    router_sd, scorer_sd = mb.load_state_dict(router_path), mb.load_state_dict(scorer_path)
    manifest = mb.build_manifest(
        schema_version=1,
        source_commit=str(get_system_info().get("git_commit") or "unknown"),
        runtime_recipe_version="phase_d_d008_v1",
        environment_record={"python": sys.version, "torch": torch.__version__},
        model_id=f"phase-d-seed-{seed}",
        model_seed=seed,
        training_run_id=f"phase-d-d008-seed-{seed}",
        parent_bundle_ids=(),
        build_route=mb.BuildRoute.CLEAN_BUILD,
        scope=mb.BundleScope.NOMINAL,
        requested_capabilities=frozenset({"nominal_execution", "phase_d_sort_repair"}),
        publish_status=mb.PublishStatus.PUBLISHED,
        core=mb.ComponentManifest("core", str(core_path.resolve()), core_raw, core_hash, schema),
        vocabulary=mb.ComponentManifest(
            "vocabulary", str(vocab_path.resolve()), vocab_raw, vocab_hash, schema
        ),
        primitives=tuple(primitives),
        router=mb.RouterManifest(
            str(router_path.resolve()),
            mb.canonical_state_hash(mb.router_non_key_state_dict(router_sd)),
            core_hash,
            mb.router_key_to_primitive_mapping_hash(router_sd, [p.physical_id for p in primitives]),
        ),
        argument_scorer=mb.ArgumentScorerManifest(
            str(scorer_path.resolve()),
            mb.canonical_state_hash(scorer_sd),
            core_hash,
            "argument_scorer_schema_v1",
        ),
        scoring_policy=mb.ScoringPolicyManifest(
            "dot_product_v1",
            "select_sigmoid_v2_adr0088",
            2.0,
            "phase_d_d008_v1",
            f"phase-d-{seed}-pair",
        ),
        build_recipe_hash=hashlib.sha256(b"phase-d-d001-fixed-cohort-recipe-v2").hexdigest(),
        clean_build_exercised_stages=(
            "CORE_PRETRAIN",
            "CANONICAL_AND_BRANCH_B_BUILD",
            "SHIFT_DEDICATED",
            "INCREMENTAL_6_BUILD",
            "ROUTER_CALIBRATION",
            "ARGUMENT_SCORER_CALIBRATION",
        ),
        qualification_refs=("ADR-0170", "ADR-0172", "D-008"),
        cohort_id=COHORT_ID,
        cohort_member_seeds=MODEL_SEEDS,
        cohort_construction_recipe_hash=hashlib.sha256(
            b"phase-d-d001-fixed-cohort-recipe-v2"
        ).hexdigest(),
    )
    loaded = mb.load_bundle(manifest, mode="nominal", expected_primitive_count=16)
    if not REQUIRED_LOAD_CHECKS.issubset(loaded.checks_performed):
        raise PhaseDStopGateError("parent strict loader omitted required integrity checks")
    manifest_path = publish / "manifest.json"
    _json_write(
        manifest_path,
        {
            "manifest": _manifest_to_json_dict(manifest),
            "op_to_id": op_to_id,
            "primitive_bank_structure": str(bank_structure_path.resolve()),
            "loader_checks": list(loaded.checks_performed),
        },
    )
    return manifest, manifest_path


def _build_parent(
    seed: int, config: PhaseDExecutorConfig
) -> tuple[Any, Any, dict[str, int], Any, Any, mb.ModelBundleManifest, Path]:
    seed_dir = REPO_ROOT / config.cohort_root / f"seed_{seed}"
    _assert_new_namespace(seed_dir)
    set_seed(seed)
    lr = LearnedRoutingBenchmarkConfig(
        seed=seed,
        device="cuda",
        core_train_steps=16_000,
        bank_train_steps=6_000,
        shared_encoder_checkpoint=None,
    )
    core, bank, op_to_id = _ensure_learned_routing_bank_and_core(
        lr, seed_dir=seed_dir / "canonical_branch_b"
    )
    # The generic SHIFT byproduct is replaced by the fixed iid dedicated recipe only.
    shift_config = ShiftFunctionalGeneralizationRepairConfig(
        development_seeds=(seed,),
        variants=("iid_baseline",),
        shared_encoder_cache_dir=seed_dir / "forbidden_unused_cache",
        output_dir=seed_dir / "shift_recipe",
    )
    candidate_shift = _train_shift_candidate(
        core, op_to_id["SHIFT"], variant="iid_baseline", seed=seed, config=shift_config
    )
    bank.replace_primitive(op_to_id["SHIFT"], candidate_shift)
    ucfg = UnifiedBenchmarkConfig(seed=seed, device="cuda")
    for operation in PHASE_A2_INCREMENTAL_NEW_OPERATIONS:
        primitive = bank.new_cross_position_primitive(
            CrossPositionPrimitiveConfig(
                operation=operation,
                d_model=core.model.config.d_model,
                d_operator=32,
                n_head=4,
                d_operator_ff=64,
                vocab_size=10,
                max_sequence_length=32,
            ),
            status=PrimitiveStatus.STABLE,
        )
        op_to_id[operation] = primitive.primitive_id
        _train_single_primitive(core, primitive, ucfg, operation, steps=1_000)
    if len(op_to_id) != 16:
        raise PhaseDStopGateError(f"cohort bank must contain 16 primitives, got {len(op_to_id)}")
    bank.freeze_all()
    bank.eval()
    # The primary execution is explicit-recipe. The parent manifest nevertheless includes
    # frozen router/scorer calibration from the existing fixed recovery recipe.
    from apc.evaluation.model_bundle_recovery import _calibrate_router_and_scorer

    router, scorer, _ = _calibrate_router_and_scorer(
        core, bank, op_to_id, type("C", (), {"seed": seed})()
    )
    manifest, manifest_path = _publish_parent(seed, seed_dir, core, bank, router, scorer, op_to_id)
    return core, bank, op_to_id, router, scorer, manifest, manifest_path


def _examples_for_recipe(
    seed: int, recipe: tuple[str, ...], n: int, lengths: tuple[int, int]
) -> list[Example]:
    if len(recipe) == 1:
        return _generate_parameter_free_examples(
            seed, n, operation=recipe[0], split="phase_d_eval", sequence_length_range=lengths
        )
    # Use a local fixed generator; a process-randomized hash would invalidate fresh-load equality.
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"phase-d:{seed}:{recipe}:{n}:{lengths}".encode()).digest()[:8], "big"
        )
    )
    result: list[Example] = []
    while len(result) < n:
        sequence = tuple(rng.randrange(10) for _ in range(rng.randint(*lengths)))
        current = sequence
        steps: list[ProgramStep] = []
        try:
            for operation in recipe:
                op = get_operation(operation)
                if not op.is_valid_for_length(len(current)):
                    raise ValueError("invalid intermediate length")
                params = op.sample_params(rng, current, 10)
                current = op.apply(current, 10, params)
                steps.append(ProgramStep(operation, params))
        except (KeyError, ValueError):
            continue
        program = Program(steps=tuple(steps))
        executed = run_program(program, sequence, 10)
        result.append(
            Example(
                sequence,
                executed.output_tokens,
                program,
                executed.graph,
                "phase_d",
                "evaluation",
                10,
                TaskSpec.from_program(program),
                OracleMetadata(label="C", primitive_operations=recipe),
            )
        )
    return result


def _em(
    core: Any, bank: Any, op_to_id: dict[str, int], examples: Sequence[Example]
) -> tuple[int, int, str]:
    successes = 0
    outputs = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            chunk = examples[start : start + 128]
            logits = execute_composition_recipe(core, bank, op_to_id, chunk)
            pred = logits.argmax(dim=-1)
            for i, ex in enumerate(chunk):
                tokens = tuple(pred[i, : len(ex.target_tokens)].tolist())
                successes += tokens == ex.target_tokens
                outputs.update(json.dumps(tokens, separators=(",", ":")).encode())
                outputs.update(b"\n")
    return successes, len(examples), outputs.hexdigest()


def _primitive_em(
    core: Any, primitive: Any, examples: Sequence[Example], operation: str
) -> tuple[int, int, str]:
    """Direct primitive-arm measurement with a digest of every discrete output."""
    from apc.environments.operations import get_operation
    from apc.primitives.primitive import (
        CrossPositionPrimitive,
        ReverseRelativePrimitive,
        ShiftRelativePrimitive,
    )

    successes = 0
    outputs = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            chunk = examples[start : start + 128]
            lengths = [len(example.input_tokens) for example in chunk]
            output_lengths = [get_operation(operation).output_length(length) for length in lengths]
            inputs = collate_content_only_batch(chunk, core.tokens, device=core.device)
            hidden = core.model.encode(inputs)[:, 1 : 1 + max(lengths), :]
            if isinstance(
                primitive,
                (CrossPositionPrimitive, ReverseRelativePrimitive, ShiftRelativePrimitive),
            ):
                logits = primitive(hidden, lengths, output_lengths, None)
            else:
                logits = primitive(hidden)
            prediction = logits.argmax(dim=-1)
            for index, example in enumerate(chunk):
                tokens = tuple(prediction[index, : len(example.target_tokens)].tolist())
                successes += tokens == example.target_tokens
                outputs.update(json.dumps(tokens, separators=(",", ":")).encode())
                outputs.update(b"\n")
    return successes, len(examples), outputs.hexdigest()


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _evaluate_panels(core: Any, bank: Any, op_to_id: dict[str, int], seed: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, classes, n, lengths in (
        ("target", TARGET_CLASSES, 1_000, (6, 10)),
        ("regression", REGRESSION_CLASSES, 500, (6, 10)),
        ("canary", CANARY_CLASSES, 500, (6, 10)),
    ):
        for klass in classes:
            per_seed: list[dict[str, Any]] = []
            for eval_seed in EVAL_SEEDS:
                successes, total, outputs_hash = _em(
                    core,
                    bank,
                    op_to_id,
                    _examples_for_recipe(eval_seed, tuple(klass.split("->")), n, lengths),
                )
                per_seed.append(
                    {
                        "seed": eval_seed,
                        "successes": successes,
                        "n": total,
                        "outputs_sha256": outputs_hash,
                    }
                )
            s = sum(row["successes"] for row in per_seed)
            total = sum(row["n"] for row in per_seed)
            lo, hi = _wilson(s, total)
            records[f"{name}:{klass}"] = {
                "successes": s,
                "n": total,
                "em": s / total,
                "wilson_95": [lo, hi],
                "per_evaluation_seed": per_seed,
                "outputs_sha256": hashlib.sha256(
                    "".join(row["outputs_sha256"] for row in per_seed).encode()
                ).hexdigest(),
            }
    for length in (3, 4, 5):
        examples = _examples_for_recipe(seed, ("SORT",), 10**length, (length, length))
        s, total, outputs_hash = _em(core, bank, op_to_id, examples)
        records[f"standalone_target_L{length}"] = {
            "successes": s,
            "n": total,
            "em": s / total,
            "population": "exhaustive",
            "outputs_sha256": outputs_hash,
        }
    for length in range(6, 11):
        regression_per_seed: list[dict[str, Any]] = []
        for eval_seed in EVAL_SEEDS[:3]:
            examples = _examples_for_recipe(eval_seed, ("SORT",), 2_000, (length, length))
            successes, total, outputs_hash = _em(core, bank, op_to_id, examples)
            regression_per_seed.append(
                {
                    "seed": eval_seed,
                    "successes": successes,
                    "n": total,
                    "outputs_sha256": outputs_hash,
                }
            )
        s = sum(row["successes"] for row in regression_per_seed)
        total = sum(row["n"] for row in regression_per_seed)
        records[f"standalone_regression_L{length}"] = {
            "successes": s,
            "n": total,
            "em": s / total,
            "wilson_95": list(_wilson(s, total)),
            "per_evaluation_seed": regression_per_seed,
            "outputs_sha256": hashlib.sha256(
                "".join(row["outputs_sha256"] for row in regression_per_seed).encode()
            ).hexdigest(),
        }
    return records


def _evaluate_causal_controls(
    core: Any, bank: Any, op_to_id: dict[str, int]
) -> dict[str, Any]:
    """Measure every preregistered SORT control arm in both length groups.

    SORT is parameter-free, so no Wrong-argument arm exists.  The report is
    deliberately per evaluation seed and arm: an aggregate without a cell is
    invalid evidence, not a value that may be averaged into a pass.
    """
    sort = bank.get(op_to_id["SORT"])
    wrong_family = bank.get(op_to_id[WRONG_FAMILY_MAP["SORT"]])
    results: dict[str, Any] = {}
    for group_name, lengths in CAUSAL_LENGTH_GROUPS:
        arms: dict[str, Any] = {}
        for arm in CAUSAL_ARMS:
            per_seed: list[dict[str, Any]] = []
            for eval_seed in EVAL_SEEDS:
                examples = _generate_parameter_free_examples(
                    eval_seed,
                    1_000,
                    operation="SORT",
                    split=f"phase_d_causal_{group_name}_{arm}",
                    sequence_length_range=lengths,
                )
                primitive = sort if arm != "wrong_family" else wrong_family
                was_enabled = primitive.enabled
                if arm == "none":
                    primitive.enabled = False
                try:
                    successes, total, outputs_hash = _primitive_em(
                        core, primitive, examples, "SORT"
                    )
                finally:
                    primitive.enabled = was_enabled
                per_seed.append(
                    {
                        "seed": eval_seed,
                        "successes": successes,
                        "n": total,
                        "em": successes / total,
                        "wilson_95": list(_wilson(successes, total)),
                        "outputs_sha256": outputs_hash,
                    }
                )
            successes = sum(row["successes"] for row in per_seed)
            total = sum(row["n"] for row in per_seed)
            arms[arm] = {
                "successes": successes,
                "n": total,
                "em": successes / total,
                "wilson_95": list(_wilson(successes, total)),
                "per_evaluation_seed": per_seed,
                "outputs_sha256": hashlib.sha256(
                    "".join(row["outputs_sha256"] for row in per_seed).encode()
                ).hexdigest(),
            }
        correct = arms["correct"]["em"]
        wrong = arms["wrong_family"]["em"]
        none = arms["none"]["em"]
        results[group_name] = {
            "arms": arms,
            "wrong_argument": "NOT_APPLICABLE_PARAMETER_FREE_SORT",
            "causal_gap": correct - max(wrong, none),
            "correct_pass": correct >= 0.95,
            "causal_gap_pass": correct - max(wrong, none) >= 0.50,
            "none_pass": none <= NATURAL_BASELINES["SORT"] + 0.05,
        }
        results[group_name]["pass"] = all(
            results[group_name][key]
            for key in ("correct_pass", "causal_gap_pass", "none_pass")
        )
    _assert_causal_controls_complete(results)
    return results


def _assert_causal_controls_complete(results: dict[str, Any]) -> None:
    """Reject missing arms/seeds rather than treating absent cells as zeroes."""
    if set(results) != {name for name, _ in CAUSAL_LENGTH_GROUPS}:
        raise PhaseDStopGateError("causal controls missing a preregistered length group")
    for group_name, _lengths in CAUSAL_LENGTH_GROUPS:
        group = results[group_name]
        if set(group.get("arms", ())) != set(CAUSAL_ARMS):
            raise PhaseDStopGateError(f"causal controls missing an arm for {group_name}")
        for arm in CAUSAL_ARMS:
            records = group["arms"][arm].get("per_evaluation_seed", ())
            if tuple(row.get("seed") for row in records) != EVAL_SEEDS:
                raise PhaseDStopGateError(
                    f"causal controls missing or reordered evaluation seeds for {group_name}:{arm}"
                )
            if any(row.get("n", 0) != 1_000 for row in records):
                raise PhaseDStopGateError(
                    f"causal controls have wrong sample count for {group_name}:{arm}"
                )


def _repair(core: Any, bank: Any, op_to_id: dict[str, int], seed: int) -> dict[str, Any]:
    bank.freeze_all()
    target = bank.get(op_to_id["SORT"])
    target.train()
    for parameter in target.parameters():
        parameter.requires_grad_(True)
    before = {pid: mb.canonical_state_hash(bank.get(pid).state_dict()) for pid in bank.ids()}
    optimiser = torch.optim.AdamW(target.parameters(), lr=0.0008, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=REPAIR_STEPS, eta_min=1e-5
    )
    for step in range(1, REPAIR_STEPS + 1):
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"d001_train:{seed}:{step}:SORT".encode()).digest()[:8], "big"
            )
        )
        examples = _examples_for_recipe(rng.randrange(2**31), ("SORT",), 32, (3, 10))
        lengths = [len(x.input_tokens) for x in examples]
        labels = _labels_for_examples(examples, lengths, max(lengths), core.device)
        with torch.no_grad():
            inp = collate_content_only_batch(examples, core.tokens, device=core.device)
            hidden = core.model.encode(inp)[:, 1 : 1 + max(lengths), :]
        optimiser.zero_grad(set_to_none=True)
        logits = target(hidden, lengths, lengths, None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(target.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
    target.eval()
    bank.freeze_all()
    after = {pid: mb.canonical_state_hash(bank.get(pid).state_dict()) for pid in bank.ids()}
    if any(before[pid] != after[pid] for pid in bank.ids() if pid != op_to_id["SORT"]):
        raise PhaseDStopGateError("non-SORT parameter changed during LOCAL_SORT_REPAIR")
    if before[op_to_id["SORT"]] == after[op_to_id["SORT"]]:
        raise PhaseDStopGateError("SORT parameters did not change during registered repair")
    return {
        "steps": REPAIR_STEPS,
        "examples": REPAIR_STEPS * 32,
        "resident_parameters_touched": target.num_parameters(),
        "active_parameters": target.num_parameters(),
        "temporary_parameters": 0,
        "before_hashes": before,
        "after_hashes": after,
    }


def _save_candidate(
    parent: mb.ModelBundleManifest,
    bank: Any,
    op_to_id: dict[str, int],
    seed: int,
    config: PhaseDExecutorConfig,
) -> tuple[mb.ModelBundleManifest, Path, Path]:
    """Persist a diagnostic candidate in its own namespace, never in the parent path."""
    candidate_dir = REPO_ROOT / config.candidate_root / f"seed_{seed}" / "candidate"
    candidate_dir.mkdir(parents=True)
    bank_path = candidate_dir / "primitive_bank.pt"
    torch.save(bank.state_dict(), bank_path)
    bank_manifest_path = candidate_dir / "primitive_bank_structure.json"
    _json_write(bank_manifest_path, bank.to_manifest())
    state = mb.load_state_dict(bank_path)
    entries = []
    for entry in parent.primitives:
        if entry.operation_name != "SORT":
            entries.append(entry)
            continue
        sort_state = mb.primitive_state_dict(state, op_to_id["SORT"])
        entries.append(
            dataclasses.replace(
                entry,
                version="phase-d-local-sort-repair-v1",
                weights_hash=mb.canonical_state_hash(sort_state),
                state_abi_hash=mb.compute_state_abi_hash(
                    sort_state, architecture_signature=entry.architecture_signature
                ),
                source_artifact=str(bank_path.resolve()),
                training_receipt="D-008 LOCAL_SORT_REPAIR, exactly 6,000 updates",
            )
        )
    candidate = dataclasses.replace(
        parent,
        bundle_id="",
        content_manifest_digest="",
        parent_bundle_ids=(parent.bundle_id,),
        primitives=tuple(entries),
        scope=mb.BundleScope.DIAGNOSTIC,
        training_run_id=f"phase-d-d008-local-sort-repair-seed-{seed}",
    )
    candidate = mb.build_manifest(
        **{
            field.name: getattr(candidate, field.name)
            for field in dataclasses.fields(candidate)
            if field.name not in {"bundle_id", "content_manifest_digest"}
        }
    )
    loaded = mb.load_bundle(candidate, mode="diagnostic", expected_primitive_count=16)
    if not REQUIRED_LOAD_CHECKS.issubset(loaded.checks_performed):
        raise PhaseDStopGateError("candidate strict loader omitted required integrity checks")
    manifest_path = candidate_dir / "candidate_manifest.json"
    _json_write(
        manifest_path,
        {
            "manifest": _manifest_to_json_dict(candidate),
            "op_to_id": op_to_id,
            "primitive_bank_structure": str(bank_manifest_path.resolve()),
            "bundle_id": candidate.bundle_id,
            "parent_bundle_id": parent.bundle_id,
            "loader_checks": list(loaded.checks_performed),
        },
    )
    return candidate, manifest_path, bank_manifest_path


def _acceptance(
    baseline: dict[str, Any],
    repaired: dict[str, Any],
    causal: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, Any]:
    """Fixed cellwise floors. Missing or mismatched evidence is never a pass."""
    target_keys = [key for key in repaired if key.startswith("target:")]
    target_keys.extend(f"standalone_target_L{length}" for length in (3, 4, 5))
    preservation_keys = [
        key
        for key in repaired
        if key.startswith(("regression:", "canary:", "standalone_regression_"))
    ]
    target_pass = all(repaired[key]["em"] >= 0.95 for key in target_keys)
    preservation_pass = all(
        repaired[key]["em"] >= baseline[key]["em"] - 0.01 and repaired[key]["em"] >= 0.95
        for key in preservation_keys
    )
    causal_pass = all(causal[name]["pass"] for name, _ in CAUSAL_LENGTH_GROUPS)
    fresh_pass = bool(fresh.get("parity_pass"))
    return {
        "target_recovery_pass": target_pass,
        "existing_capability_preservation_pass": preservation_pass,
        "causal_control_pass": causal_pass,
        "fresh_load_parity_pass": fresh_pass,
        "pass": target_pass and preservation_pass and causal_pass and fresh_pass,
    }


def _run_fresh_load_parity(
    candidate_manifest: Path,
    bank_structure: Path,
    seed: int,
    expected: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Re-evaluate a candidate in a separate interpreter and require exact parity."""
    script = REPO_ROOT / "scripts" / "phase_d_fresh_load_check.py"
    if not script.is_file():
        raise PhaseDStopGateError("fresh-load checker script is missing")
    expected_path = candidate_manifest.parent / "in_process_metrics.json"
    _json_write(expected_path, expected)
    scratch = Path(tempfile.gettempdir()) / f"apc_phase_d_fresh_load_seed_{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidate-manifest",
            str(candidate_manifest.resolve()),
            "--bank-structure",
            str(bank_structure.resolve()),
            "--expected-metrics",
            str(expected_path.resolve()),
            "--mode",
            mode,
        ],
        cwd=scratch,
        capture_output=True,
        text=True,
        timeout=1_800,
        check=False,
    )
    result: dict[str, Any] = {
        "separate_process": True,
        "mode": mode,
        "different_working_directory_confirmed": scratch.resolve() != REPO_ROOT.resolve(),
        "subprocess_returncode": process.returncode,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "parity_pass": False,
    }
    if process.returncode == 0:
        try:
            fresh = json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise PhaseDStopGateError(f"fresh-load checker returned invalid JSON: {exc}") from exc
        result.update(fresh)
        result["parity_pass"] = bool(fresh.get("parity_pass"))
    if not result["parity_pass"]:
        raise PhaseDStopGateError(f"fresh-load parity failed for seed {seed}: {result}")
    return result


def run(config: PhaseDExecutorConfig | None = None) -> dict[str, Any]:
    """Execute once in new namespaces.  A stop gate writes evidence then raises."""
    effective_config = config or PhaseDExecutorConfig()
    _assert_new_namespace(REPO_ROOT / effective_config.output_root)
    _assert_new_namespace(REPO_ROOT / effective_config.cohort_root)
    _assert_new_namespace(REPO_ROOT / effective_config.candidate_root)
    root = REPO_ROOT / effective_config.output_root
    root.mkdir(parents=True)
    (REPO_ROOT / effective_config.candidate_root).mkdir(parents=True)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    report: dict[str, Any] = {
        "task": "D-008",
        "static_gate": _static_gate(effective_config),
        "models": {},
        "sealed_access": 0,
    }
    try:
        parents: list[mb.ModelBundleManifest] = []
        for seed in effective_config.model_seeds:
            core, bank, op_to_id, _router, _scorer, manifest, manifest_path = _build_parent(
                seed, effective_config
            )
            parents.append(manifest)
            baseline = _evaluate_panels(core, bank, op_to_id, seed)
            parent_causal_controls = _evaluate_causal_controls(core, bank, op_to_id)
            parent_record = json.loads(manifest_path.read_text(encoding="utf-8"))
            parent_fresh_load = _run_fresh_load_parity(
                manifest_path,
                Path(parent_record["primitive_bank_structure"]),
                seed,
                {
                    "manifest": {
                        "bundle_id": manifest.bundle_id,
                        "content_manifest_digest": manifest.content_manifest_digest,
                    },
                    "panels": baseline,
                    "causal_controls": parent_causal_controls,
                },
                "nominal",
            )
            # FROZEN_PARENT eligibility means an independently built, strict-loaded parent whose
            # known length-adequate capabilities are present; the known target deficit is measured,
            # not silently used to reject the preregistered repair.
            eligible = all(
                baseline[f"standalone_regression_L{length}"]["em"] >= 0.95
                for length in range(6, 11)
            )
            if not eligible:
                raise PhaseDStopGateError(f"FROZEN_PARENT eligibility failed for seed {seed}")
            repair = _repair(core, bank, op_to_id, seed)
            repaired = _evaluate_panels(core, bank, op_to_id, seed)
            causal_controls = _evaluate_causal_controls(core, bank, op_to_id)
            candidate, candidate_manifest, bank_structure = _save_candidate(
                manifest, bank, op_to_id, seed, effective_config
            )
            in_process_metrics = {
                "manifest": {
                    "bundle_id": candidate.bundle_id,
                    "content_manifest_digest": candidate.content_manifest_digest,
                },
                "panels": repaired,
                "causal_controls": causal_controls,
            }
            fresh_load = _run_fresh_load_parity(
                candidate_manifest, bank_structure, seed, in_process_metrics, "diagnostic"
            )
            acceptance = _acceptance(baseline, repaired, causal_controls, fresh_load)
            report["models"][str(seed)] = {
                "parent_manifest": str(manifest_path),
                "candidate_manifest": str(candidate_manifest),
                "candidate_bundle_id": candidate.bundle_id,
                "frozen_parent": baseline,
                "frozen_parent_causal_controls": parent_causal_controls,
                "frozen_parent_fresh_load": parent_fresh_load,
                "local_sort_repair": repair,
                "repaired": repaired,
                "causal_controls": causal_controls,
                "fresh_load": fresh_load,
                "symbolic_reference": "SortOp.apply; evaluator only",
                "frozen_parent_eligible": eligible,
                "acceptance": acceptance,
            }
            _json_write(root / f"seed_{seed}.json", report["models"][str(seed)])
        mb.assert_distinct_model_identities(parents)
        report["result"] = (
            "PASS"
            if all(model["acceptance"]["pass"] for model in report["models"].values())
            else "FAIL"
        )
    except PhaseDStopGateError as exc:
        report["result"] = "STOP_GATE_FAIL"
        report["failure"] = str(exc)
        _json_write(root / "stop_gate.json", report)
        raise
    finally:
        report["wall_clock_seconds"] = time.perf_counter() - start
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["bundle_promotion"] = "NOT_AUTHORIZED"
        report["sealed_access"] = 0
        _json_write(root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = dry_run_manifest() if args.dry_run else run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
