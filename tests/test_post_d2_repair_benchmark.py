"""Tests for the B-C005R3-001 runner (`apc.evaluation.post_d2_repair_benchmark`).

CPU-only, no GPU/model reconstruction: this task only touches seed
derivation and reads/hashes existing files.
"""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.post_d2_repair_benchmark import (
    REPO_ROOT,
    TASK_RUN_DIRS,
    PostD2ReproducibilityConfig,
    build_artifact_inventory,
    build_generator_audit,
    run_post_d2_reproducibility_task,
    run_subprocess_comparison,
)


def test_build_generator_audit_reports_fix_verified() -> None:
    audit = build_generator_audit()
    v = audit["verification"]
    assert v["bug_pattern_fully_removed_from_fixed_files"] is True
    assert v["bug_pattern_literal_remaining_in_source"] == {
        "consolidation_benchmark.py": False,
        "recurrence_benchmark.py": False,
    }
    assert v["g0_no_builtin_hash_on_primary_path"] is True
    assert audit["wrappers_are_same_function_object"] is True
    assert audit["default_generator_version"] == "v2_sha256_indexed"
    # The out-of-scope third occurrence must be disclosed, not silently fixed.
    assert "discovery_capacity_harness.py" in audit["known_unfixed_related_occurrence"]["location"]


def test_build_artifact_inventory_marks_regate_sealed_seeds_unavailable() -> None:
    inventory = build_artifact_inventory()
    per_seed = inventory["shared_base_bank"]["per_seed"]
    # Persisted seeds actually on disk today.
    for seed in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
        assert per_seed[str(seed)]["availability"] == "AVAILABLE"
        assert per_seed[str(seed)]["sha256"] is not None
    # Regate-sealed seeds (20-24): never independently persisted.
    for seed in (20, 21, 22, 23, 24):
        assert per_seed[str(seed)]["availability"] == "UNAVAILABLE"
        assert per_seed[str(seed)]["sha256"] is None
        assert "note" in per_seed[str(seed)]

    # Bank sizes other than 16 are documented as never persisted, not silently omitted.
    assert set(inventory["shared_base_bank"]["never_persisted_bank_sizes"]) == {"32", "64", "128"}

    task_ids = {t.task_id for t in TASK_RUN_DIRS}
    assert task_ids == {"B-C005", "B-C005D", "B-C005R1", "B-C005R2", "B-C005G", "B-C005D2"}
    for task in inventory["tasks"].values():
        assert task["trained_model_state_persisted"] is False
        assert task["trained_model_state_rationale"]

    assert inventory["totals"]["files_checked"] > 0
    assert inventory["totals"]["available"] + inventory["totals"]["unavailable"] == (
        inventory["totals"]["files_checked"]
    )
    # This task never claims a missing artifact was regenerated as-if-identical.
    assert "no missing artifact is regenerated" in inventory["policy"].lower()


def test_artifact_inventory_never_writes_or_regenerates_anything() -> None:
    """Building the inventory must be read-only against the paths it inspects.

    Scoped to exactly the directories `build_artifact_inventory` reads
    (the shared base bank checkpoint dir + each task's declared run dir),
    not the whole `runs/` tree, since other tasks/tests may be writing
    elsewhere in `runs/` concurrently.
    """
    watched_dirs = [REPO_ROOT / "runs/phase_a2_bank_scaling_benchmark"]
    watched_dirs += [REPO_ROOT / t.run_dir for t in TASK_RUN_DIRS]

    def _snapshot() -> dict[Path, float]:
        snap: dict[Path, float] = {}
        for d in watched_dirs:
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file():
                        snap[p] = p.stat().st_mtime
        return snap

    before = _snapshot()
    build_artifact_inventory()
    after = _snapshot()
    assert before == after


def test_config_to_dict_serializes_pythonhashseed_variants() -> None:
    cfg = PostD2ReproducibilityConfig(
        seeds=(1,), operations=("SWAP_PAIRS",), splits=("test",),
        n_examples=4, pythonhashseed_variants=(None, "0"),
        output_dir=Path("runs/does_not_matter"),
    )
    raw = cfg.to_dict()
    assert raw["pythonhashseed_variants"] == ["<unset>", "0"]
    assert raw["output_dir"] == str(Path("runs/does_not_matter"))


def test_run_subprocess_comparison_small_grid_agrees() -> None:
    cfg = PostD2ReproducibilityConfig(
        seeds=(7,),
        operations=("SWAP_PAIRS",),
        splits=("test",),
        n_examples=4,
        pythonhashseed_variants=(None, "0", "42"),
        output_dir=Path("runs/phase_b_b2_post_d2/_unused_in_this_test"),
    )
    result = run_subprocess_comparison(cfg)
    assert len(result["cells"]) == 1
    cell = result["cells"][0]
    assert cell["all_variants_agree"] is True
    assert len(cell["hashes_by_pythonhashseed"]) == 3
    assert result["all_cells_agree_across_pythonhashseed"] is True


def test_run_post_d2_reproducibility_task_writes_all_artifacts_and_passes_g0(
    tmp_path: Path,
) -> None:
    cfg = PostD2ReproducibilityConfig(
        seeds=(1,),
        operations=("SWAP_PAIRS",),
        splits=("test",),
        n_examples=4,
        pythonhashseed_variants=(None, "0"),
        output_dir=tmp_path / "r3_001_test_run",
    )
    report = run_post_d2_reproducibility_task(cfg)
    assert report["protocol"]["result"] == "INFRASTRUCTURE_OR_PROTOCOL_PASS"

    out_dir = tmp_path / "r3_001_test_run"
    for name in (
        "generator_audit.json", "artifact_inventory.json",
        "subprocess_comparison.json", "config.yaml", "system.json", "protocol.json",
    ):
        path = out_dir / name
        assert path.is_file(), f"missing expected artifact {name}"
        json.loads(path.read_text(encoding="utf-8"))  # must be valid JSON

    protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["criteria"]["model_weights_changed"] is False
    assert protocol["criteria"]["adequacy_threshold_changed"] is False
