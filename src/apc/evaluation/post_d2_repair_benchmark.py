"""Task B-C005R3-001: Reproducible Benchmark Generation & Artifact Inventory.

Thin runner over `apc.utils.seed_derivation` and the consolidated
`generate_benchmark_examples` (ADR-0080 fix). Produces the three run
artifacts required by `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`'s
R3-001 section:

- `generator_audit.json` -- what changed in the seed-derivation call graph,
  what was deliberately left out of scope, and a programmatic check that no
  builtin-hash seed derivation remains reachable from the v2 default path.
- `artifact_inventory.json` -- existence / sha256 / availability ledger for
  checkpoints and data artifacts referenced by B-C005 / D / R1 / R2 / G / D2.
  This module only *reads* what is already on disk; it never regenerates a
  missing artifact and never labels a fresh regeneration as the same data
  as a historical run (see `EXACT_REPLAY_UNAVAILABLE` / `V2_RECONSTRUCTION`
  below, per `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md` S2.3).
- `subprocess_comparison.json` -- the G0 PYTHONHASHSEED-independence check,
  run as a recorded run artifact (the same property is also covered, with
  finer-grained assertions, by `tests/test_seed_derivation.py`).

Out of scope (must not be imported or touched by this module): router,
ArgumentScorer, verifier, controller, any primitive, model weights,
adequacy thresholds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from apc.utils.seed_derivation import (
    DEFAULT_GENERATOR_VERSION,
    GENERATOR_VERSION_V1_LEGACY,
    GENERATOR_VERSION_V2,
)
from apc.utils.system_info import get_system_info

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# 1. Generator call-graph audit
# ---------------------------------------------------------------------------

_FIXED_LOCATIONS: Final[tuple[str, ...]] = (
    "src/apc/evaluation/consolidation_benchmark.py:generate_benchmark_examples "
    "(canonical implementation; was one of the two literal-duplicate "
    "hash(operation) % 10000 sites named in ADR-0080 evidence item 5)",
    "src/apc/evaluation/recurrence_benchmark.py:generate_benchmark_examples "
    "(the other literal-duplicate site; now re-exports the consolidation_benchmark "
    "implementation instead of defining its own copy, so the two can no longer "
    "silently drift apart again)",
)

# ADR-0080/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md name exactly these two
# locations as R3-001's scope. This third occurrence was found while tracing
# the call graph but is deliberately NOT modified here: it belongs to Phase
# A1 (A1-B007X discovery-capacity infrastructure), is not one of R3-001's
# named "2 locations", is not in R3-001's read/target list, and touching it
# would be an unrelated broadening of scope forbidden by both AGENTS.md and
# the task doc.
_KNOWN_UNFIXED_RELATED_OCCURRENCE: Final[dict[str, str]] = {
    "location": "src/apc/evaluation/discovery_capacity_harness.py:170 "
    "(generate_novel_examples)",
    "pattern": "rng = random.Random(seed * 7919 + salt + (hash(operation) % 10000))",
    "reason_not_fixed": (
        "Out of R3-001's declared 2-location scope (Phase A1 A1-B007X "
        "discovery-capacity harness, not a Phase B B2 module, not in "
        "R3-001's read/target list). No other module imports "
        "generate_novel_examples, so it does not affect any B-C005 family "
        "result. Recorded here as a candidate for a future, separately "
        "scoped follow-up task; not authorized by this ADR."
    ),
}

_PRECEDENT_NOTE: Final[str] = (
    "src/apc/environments/generator.py:_derive_seed already uses a "
    "SHA256(f'{seed}:{label}') digest instead of builtin hash() for a "
    "different sub-seed derivation, predating this task -- this fix follows "
    "that existing repository convention rather than inventing a new one."
)


def _source_still_contains_bug_pattern(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "hash(operation) % 10000" in text


def _v2_default_reaches_builtin_hash() -> bool:
    """Runtime check: does the *default* generation path call builtin hash()?

    Patches `builtins.hash` to raise, calls `generate_benchmark_examples`
    with no `generator_version` override (i.e. whatever the real default
    is), and reports whether that raised. This is the same technique used
    in `tests/test_seed_derivation.py::test_v2_default_path_never_calls_builtin_hash`,
    duplicated here (deliberately, not by import) so the run artifact is
    self-contained and does not depend on the test suite being present.
    """
    import builtins

    from apc.evaluation.recurrence_benchmark import generate_benchmark_examples

    real_hash = builtins.hash
    called = {"value": False}

    def _tripwire(obj: object) -> int:
        called["value"] = True
        return real_hash(obj)

    builtins.hash = _tripwire
    try:
        generate_benchmark_examples(
            seed=1, n=2, operation="SWAP_PAIRS", split="test", vocab_size=10
        )
    finally:
        builtins.hash = real_hash
    return called["value"]


def build_generator_audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build `generator_audit.json`'s content."""
    consolidation_path = repo_root / "src/apc/evaluation/consolidation_benchmark.py"
    recurrence_path = repo_root / "src/apc/evaluation/recurrence_benchmark.py"

    bug_pattern_remaining = {
        "consolidation_benchmark.py": _source_still_contains_bug_pattern(consolidation_path),
        "recurrence_benchmark.py": _source_still_contains_bug_pattern(recurrence_path),
    }
    builtin_hash_reached_on_default_path = _v2_default_reaches_builtin_hash()

    from apc.evaluation.consolidation_benchmark import (
        generate_benchmark_examples as cb_fn,
    )
    from apc.evaluation.recurrence_benchmark import (
        generate_benchmark_examples as rb_fn,
    )

    return {
        "task": "B-C005R3-001",
        "adr_context": ["ADR-0080", "ADR-0081"],
        "default_generator_version": DEFAULT_GENERATOR_VERSION,
        "known_generator_versions": [GENERATOR_VERSION_V2, GENERATOR_VERSION_V1_LEGACY],
        "fixed_locations": list(_FIXED_LOCATIONS),
        "wrappers_are_same_function_object": rb_fn is cb_fn,
        "known_unfixed_related_occurrence": _KNOWN_UNFIXED_RELATED_OCCURRENCE,
        "existing_repo_precedent": _PRECEDENT_NOTE,
        "verification": {
            "bug_pattern_literal_remaining_in_source": bug_pattern_remaining,
            "bug_pattern_fully_removed_from_fixed_files": not any(bug_pattern_remaining.values()),
            "builtin_hash_reached_on_default_generation_path": builtin_hash_reached_on_default_path,
            "g0_no_builtin_hash_on_primary_path": not builtin_hash_reached_on_default_path,
        },
        "unchanged_by_this_task": [
            "router", "ArgumentScorer", "adequacy_verifier", "controller",
            "primitive weights", "adequacy thresholds",
        ],
        "call_sites_unaffected": (
            "All other callers of generate_benchmark_examples (e.g. "
            "adequacy_reference_audit.py, hard_negative_routing_benchmark.py, "
            "bank_scaling_benchmark.py, and ~20 further evaluation modules) "
            "pass no generator_version argument and therefore automatically "
            "receive the fixed v2 default; none required a code change."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Artifact inventory (checkpoints / data artifacts for B-C005 family)
# ---------------------------------------------------------------------------

_HASH_SIZE_LIMIT_BYTES: Final[int] = 200 * 1024 * 1024  # avoid hashing huge files


def _describe_path(repo_root: Path, rel_path: str) -> dict[str, Any]:
    p = repo_root / rel_path
    if not p.is_file():
        return {
            "path": rel_path,
            "exists": False,
            "availability": "UNAVAILABLE",
            "sha256": None,
            "size_bytes": None,
        }
    size = p.stat().st_size
    sha256 = None
    if size <= _HASH_SIZE_LIMIT_BYTES:
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
    return {
        "path": rel_path,
        "exists": True,
        "availability": "AVAILABLE",
        "sha256": sha256,
        "size_bytes": size,
    }


_BASE_BANK_CHECKPOINT_DIR: Final[str] = "runs/phase_a2_bank_scaling_benchmark"
# Seeds with a persisted seed_<n>/primitive_bank_16.pt on disk today.
_BASE_BANK_PERSISTED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
# Regate-sealed seeds referenced by B-C005G/D2 configs; never independently
# persisted -- every regate/D2 run reconstructs the bank for these seeds
# on the fly via build_scaled_bank_and_router.
_REGATE_SEALED_SEEDS_NEVER_PERSISTED: Final[tuple[int, ...]] = (20, 21, 22, 23, 24)
# Bank sizes other than the persisted 16; configs sweep [16, 32, 64, 128]
# but only size 16 is ever written to primitive_bank_16.pt.
_BANK_SIZES_NEVER_PERSISTED: Final[tuple[int, ...]] = (32, 64, 128)


@dataclass(frozen=True)
class TaskRunDir:
    """One B-C005-family task's declared output directory and expected files."""

    task_id: str
    adr: str
    status: str
    run_dir: str
    expected_files: tuple[str, ...]
    note: str = ""


TASK_RUN_DIRS: Final[tuple[TaskRunDir, ...]] = (
    TaskRunDir(
        "B-C005", "ADR-0075", "FAILED_STOP_GATE_B2",
        "runs/phase_b_hard_negative_safety_gate",
        (
            "config.yaml", "metrics.jsonl", "protocol.json", "report.json",
            "report.md", "summary.json", "system.json",
        ),
    ),
    TaskRunDir(
        "B-C005D", "ADR-0076", "COMPLETE_DIAGNOSTIC",
        "runs/phase_b_b2_failure_isolation",
        (
            "config.yaml", "failure_breakdown.json", "margin_summary.json",
            "metrics.jsonl", "summary.json", "support_variance.json", "system.json",
        ),
    ),
    TaskRunDir(
        "B-C005R1", "ADR-0077", "COMPLETE_REPAIR",
        "runs/phase_b_b2_retrieval_repair",
        ("config.yaml", "metrics.jsonl", "report.md", "summary.json", "system.json"),
    ),
    TaskRunDir(
        "B-C005R2", "ADR-0078", "COMPLETE_REPAIR",
        "runs/phase_b_b2_adequacy_repair",
        ("config.yaml", "metrics.jsonl", "report.md", "summary.json", "system.json"),
    ),
    TaskRunDir(
        "B-C005G", "ADR-0079", "FAILED_SEALED_REGATE",
        "runs/phase_b_b2_regate",
        (
            "config.yaml", "metrics.jsonl", "protocol.json",
            "report.md", "summary.json", "system.json",
        ),
    ),
    TaskRunDir(
        "B-C005D2", "ADR-0080/ADR-0081", "DIAGNOSTIC_ONLY_STOP",
        "runs/phase_b_b2_second_diagnostic",
        (
            "config.yaml", "summary.json", "representation_stage_summary.json",
            "semantic_relation_summary.json", "l4_argument_breakdown.json",
            "shift_seed24_adequacy_audit.json", "reference_adequacy_summary.json",
            "final_causal_diagnosis.json", "development_sealed_comparison.json",
            "system.json", "protocol.json",
        ),
        note=(
            "Directory holds artifacts from D2-001 through D2-006 "
            "(sub-task-specific *_config.yaml/*_summary.json files also "
            "present but not individually enumerated here; see the "
            "directory listing recorded under 'other_files_present')."
        ),
    ),
)


def build_artifact_inventory(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Ledger of existence/hash/availability for B-C005/D/R1/R2/G/D2 artifacts.

    Reads only; never regenerates a missing artifact and never asserts that
    a fresh regeneration equals a historical run's exact data (that
    distinction -- EXACT_REPLAY_AVAILABLE / EXACT_REPLAY_UNAVAILABLE /
    V2_RECONSTRUCTION -- belongs to whichever later task actually attempts a
    reconstruction; this task only inventories what is on disk today).
    """
    base_bank: dict[str, Any] = {"checkpoint_dir": _BASE_BANK_CHECKPOINT_DIR, "per_seed": {}}
    for seed in _BASE_BANK_PERSISTED_SEEDS:
        rel = f"{_BASE_BANK_CHECKPOINT_DIR}/seed_{seed}/primitive_bank_16.pt"
        base_bank["per_seed"][str(seed)] = _describe_path(repo_root, rel)
    for seed in _REGATE_SEALED_SEEDS_NEVER_PERSISTED:
        rel = f"{_BASE_BANK_CHECKPOINT_DIR}/seed_{seed}/primitive_bank_16.pt"
        seed_entry = _describe_path(repo_root, rel)
        seed_entry["note"] = (
            "Regate-sealed seed (B-C005G/B-C005D2 config). No independently "
            "persisted checkpoint has ever existed for this seed; every "
            "regate/diagnostic run reconstructs it at runtime via "
            "build_scaled_bank_and_router from the base training recipe."
        )
        base_bank["per_seed"][str(seed)] = seed_entry
    base_bank["never_persisted_bank_sizes"] = {
        str(size): (
            "Configs sweep bank_sizes [16, 32, 64, 128], but only size 16 is "
            "ever written to disk as primitive_bank_16.pt; sizes 32/64/128 "
            "are synthesized at runtime from the size-16 checkpoint and are "
            "not independently reproducible from a saved file."
        )
        for size in _BANK_SIZES_NEVER_PERSISTED
    }
    base_bank["report"] = _describe_path(repo_root, f"{_BASE_BANK_CHECKPOINT_DIR}/report.json")

    tasks: dict[str, Any] = {}
    for entry in TASK_RUN_DIRS:
        run_dir_path = repo_root / entry.run_dir
        files = {
            fname: _describe_path(repo_root, f"{entry.run_dir}/{fname}")
            for fname in entry.expected_files
        }
        other_files_present: list[str] = []
        if run_dir_path.is_dir():
            for child in sorted(run_dir_path.iterdir()):
                if child.is_file() and child.name not in entry.expected_files:
                    other_files_present.append(child.name)
        tasks[entry.task_id] = {
            "adr": entry.adr,
            "status": entry.status,
            "run_dir": entry.run_dir,
            "run_dir_exists": run_dir_path.is_dir(),
            "expected_files": files,
            "other_files_present": other_files_present,
            "note": entry.note,
            "trained_model_state_persisted": False,
            "trained_model_state_rationale": (
                "This task's router/candidate/verifier state is reconstructed "
                "deterministically from the shared base bank checkpoint plus "
                "its declared seeds each time it runs; only JSON "
                "summary/report/protocol artifacts are persisted, not the "
                "trained weights themselves. Per ADR-0080 evidence item 5, "
                "prior to this fix that reconstruction was not guaranteed "
                "process-reproducible (hash(operation) nondeterminism), so a "
                "fresh reconstruction of this task's model state must be "
                "labeled V2_RECONSTRUCTION, never claimed identical to the "
                "historical run that produced these JSON numbers."
            ),
        }

    all_described = [base_bank["report"], *base_bank["per_seed"].values()]
    for t in tasks.values():
        all_described.extend(t["expected_files"].values())
    n_available = sum(1 for d in all_described if d["availability"] == "AVAILABLE")
    n_unavailable = sum(1 for d in all_described if d["availability"] == "UNAVAILABLE")

    return {
        "task": "B-C005R3-001",
        "scope": "B-C005, B-C005D, B-C005R1, B-C005R2, B-C005G, B-C005D2 (D2-001..006)",
        "shared_base_bank": base_bank,
        "tasks": tasks,
        "totals": {
            "files_checked": len(all_described),
            "available": n_available,
            "unavailable": n_unavailable,
        },
        "policy": (
            "This inventory reads existing files only. No missing artifact "
            "is regenerated by this task, and no regenerated data is ever "
            "recorded here as equivalent to a historical run's exact "
            "artifact (see design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md S2.3)."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Cross-process / cross-PYTHONHASHSEED comparison (G0 run artifact)
# ---------------------------------------------------------------------------

_SUBPROCESS_TEMPLATE: Final[str] = """
import hashlib, json, sys
sys.path.insert(0, {src_path!r})
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples

examples = generate_benchmark_examples(
    seed={seed}, n={n}, operation={operation!r}, split={split!r}, vocab_size=13,
    sequence_length_range=(6, 10),
)
payload = [[list(e.input_tokens), list(e.target_tokens)] for e in examples]
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
"""


def _run_one_subprocess(
    *, repo_root: Path, seed: int, n: int, operation: str, split: str, pythonhashseed: str | None
) -> str:
    import os

    env = dict(os.environ)
    if pythonhashseed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = pythonhashseed
    src_path = str(repo_root / "src")
    script = _SUBPROCESS_TEMPLATE.format(
        src_path=src_path, seed=seed, n=n, operation=operation, split=split
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class PostD2ReproducibilityConfig:
    """Configuration for the B-C005R3-001 reproducibility run."""

    seeds: tuple[int, ...] = (0, 10, 24)
    operations: tuple[str, ...] = ("SHIFT", "SWAP_PAIRS")
    splits: tuple[str, ...] = ("train", "test")
    n_examples: int = 16
    pythonhashseed_variants: tuple[str | None, ...] = (None, "0", "1", "123")
    output_dir: Path = Path("runs/phase_b_b2_post_d2/r3_001_reproducibility")

    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["output_dir"] = str(self.output_dir)
        raw["pythonhashseed_variants"] = [
            "<unset>" if v is None else v for v in self.pythonhashseed_variants
        ]
        return raw


def run_subprocess_comparison(
    config: PostD2ReproducibilityConfig, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Run the (seed, operation, split) x PYTHONHASHSEED-variant grid."""
    cells: list[dict[str, Any]] = []
    all_cells_agree = True
    for seed in config.seeds:
        for operation in config.operations:
            for split in config.splits:
                hashes: dict[str, str] = {}
                for variant in config.pythonhashseed_variants:
                    key = "<unset>" if variant is None else variant
                    hashes[key] = _run_one_subprocess(
                        repo_root=repo_root,
                        seed=seed,
                        n=config.n_examples,
                        operation=operation,
                        split=split,
                        pythonhashseed=variant,
                    )
                distinct = set(hashes.values())
                agree = len(distinct) == 1
                all_cells_agree = all_cells_agree and agree
                cells.append(
                    {
                        "seed": seed,
                        "operation": operation,
                        "split": split,
                        "hashes_by_pythonhashseed": hashes,
                        "all_variants_agree": agree,
                    }
                )
    return {
        "task": "B-C005R3-001",
        "generator_version": DEFAULT_GENERATOR_VERSION,
        "n_examples_per_cell": config.n_examples,
        "pythonhashseed_variants": [
            "<unset>" if v is None else v for v in config.pythonhashseed_variants
        ],
        "cells": cells,
        "all_cells_agree_across_pythonhashseed": all_cells_agree,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_post_d2_reproducibility_task(
    config: PostD2ReproducibilityConfig,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute B-C005R3-001 and write its three run artifacts + run metadata."""
    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_audit = build_generator_audit(repo_root)
    artifact_inventory = build_artifact_inventory(repo_root)
    subprocess_comparison = run_subprocess_comparison(config, repo_root)

    g0_pass = (
        generator_audit["verification"]["bug_pattern_fully_removed_from_fixed_files"]
        and generator_audit["verification"]["g0_no_builtin_hash_on_primary_path"]
        and subprocess_comparison["all_cells_agree_across_pythonhashseed"]
    )

    (output_dir / "generator_audit.json").write_text(
        json.dumps(generator_audit, indent=2), encoding="utf-8"
    )
    (output_dir / "artifact_inventory.json").write_text(
        json.dumps(artifact_inventory, indent=2), encoding="utf-8"
    )
    (output_dir / "subprocess_comparison.json").write_text(
        json.dumps(subprocess_comparison, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "system.json").write_text(
        json.dumps(get_system_info(), indent=2), encoding="utf-8"
    )
    protocol = {
        "task": "B-C005R3-001",
        "gate": "G0",
        "result": "INFRASTRUCTURE_OR_PROTOCOL_PASS" if g0_pass else "FAIL",
        "criteria": {
            "bug_pattern_fully_removed_from_fixed_files": generator_audit["verification"][
                "bug_pattern_fully_removed_from_fixed_files"
            ],
            "g0_no_builtin_hash_on_primary_path": generator_audit["verification"][
                "g0_no_builtin_hash_on_primary_path"
            ],
            "all_cells_agree_across_pythonhashseed": subprocess_comparison[
                "all_cells_agree_across_pythonhashseed"
            ],
            "model_weights_changed": False,
            "adequacy_threshold_changed": False,
        },
        "artifacts": [
            "generator_audit.json", "artifact_inventory.json",
            "subprocess_comparison.json", "config.yaml", "system.json",
        ],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "generator_audit": generator_audit,
        "artifact_inventory": artifact_inventory,
        "subprocess_comparison": subprocess_comparison,
        "protocol": protocol,
    }
