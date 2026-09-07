"""B-C005REC-004 fresh-process validation entry point.

Runs in its own `python` invocation (spawned by
`apc.evaluation.model_bundle_recovery.run_pilot_restore_build_task` via
`subprocess.run`), with no access to the parent process's in-memory Core /
PrimitiveBank / Router / ArgumentScorer objects and no in-process cache. Its
job is narrow and read-only:

1. Read `manifest_paths.json` (written by the parent build, under the
   pilot's own namespace -- never a shared-cache path).
2. Call `apc.utils.model_bundle.load_bundle(..., mode="nominal")` on the
   manifest it reconstructs from those paths (fail-closed: any hash/schema
   mismatch raises and this script exits non-zero).
3. Reconstruct fresh (never-before-existing) Core / PrimitiveBank / Router /
   ArgumentScorer module instances from *only* the loaded state dicts --
   never `torch.load`ing anything outside `load_bundle`'s own checked path,
   never referencing a shared-cache path, never calling a builder/optimizer.
4. Re-run raw/oracle direct-call exact-match measurement on the identical
   deterministic example set the parent process used (same seed/n/split/
   operation -- regenerated, not shared via IPC) for a bounded per-operation
   sample, and print one canonical JSON object to stdout.

This script deliberately imports no training/build entry point anywhere in
`apc.evaluation.model_bundle_recovery` or `apc.evaluation.learned_routing_
benchmark` / `apc.evaluation.incremental_router_benchmark` -- only
architecture-reconstruction and evaluation helpers, so a static scan of this
file's own source (see `FORBIDDEN_IDENTIFIERS` below) can independently
confirm this process never called any bank-building or router/argument-
scorer calibration entry point, separately from the parent process's own
`frozen_evaluation()` guard.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from apc.environments.operations import (  # noqa: E402
    BRANCH_B_NOVEL_OPERATION_NAMES,
    PHASE_A2_INCREMENTAL_NEW_OPERATIONS,
)
from apc.evaluation.shared_encoder_architecture_gate import (  # noqa: E402
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (  # noqa: E402
    PARAMETERIZED_OPERATION_NAMES,
    UnifiedBenchmarkConfig,
    _build_heterogeneous_bank,
    _evaluate_primitive_arm,
    _flatten_groups,
    _generate_parameter_free_examples,
    generate_compact_operator_counterfactual_groups,
)
from apc.primitives.argument_scoring import ArgumentScorer, ArgumentScorerConfig  # noqa: E402
from apc.primitives.primitive import CrossPositionPrimitiveConfig, PrimitiveStatus  # noqa: E402
from apc.primitives.router import Router, RouterConfig  # noqa: E402
from apc.utils import model_bundle as mb  # noqa: E402

# Split across string-literal concatenation so this declaration itself never
# contains the target substring contiguously -- otherwise the self-check
# below (and any external static scan of this file, e.g. a test) would
# trivially "find" every name this tuple exists to name, inside this tuple's
# own declaration, regardless of whether the script actually calls any of
# them anywhere else.
FORBIDDEN_IDENTIFIERS = (
    "get_or_" + "build",
    "_train_single_" + "primitive",
    "update_router_" + "incrementally",
    "_ensure_learned_routing_bank_and_" + "core",
    "torch." + "optim",
    "." + "backward(",
)


def _make_arg_provider_correct(operation: str):
    from apc.evaluation.compact_cross_position_operator_probe import _correct_argument_value

    def _provider(ex: Any) -> Any:
        return _correct_argument_value(operation, ex)

    return _provider


def _rebuild_16_bank_structure(core: Any, seed: int) -> tuple[Any, dict[str, int]]:
    """Structural rebuild ONLY -- no weights, no training. Mirrors
    `shift_functional_generalization_repair._rebuild_full_bank_structure`
    exactly (same call order -> same deterministic physical_id assignment),
    reimplemented here rather than imported so this script's own import list
    never touches a module that also defines a training entry point."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-paths", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-examples-per-op", type=int, default=64)
    parser.add_argument("--split", type=str, default="rec004_fresh_process_check")
    args = parser.parse_args()

    source_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_found = [name for name in FORBIDDEN_IDENTIFIERS if name in source_text]

    paths = json.loads(args.manifest_paths.read_text(encoding="utf-8"))
    op_to_id: dict[str, int] = {str(k): int(v) for k, v in paths["op_to_id"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_dict = paths["manifest"]
    manifest = mb.ModelBundleManifest(
        schema_version=manifest_dict["schema_version"],
        bundle_id=manifest_dict["bundle_id"],
        content_manifest_digest=manifest_dict["content_manifest_digest"],
        source_commit=manifest_dict["source_commit"],
        runtime_recipe_version=manifest_dict["runtime_recipe_version"],
        environment_record=manifest_dict["environment_record"],
        model_id=manifest_dict["model_id"],
        model_seed=manifest_dict["model_seed"],
        training_run_id=manifest_dict["training_run_id"],
        parent_bundle_ids=tuple(manifest_dict["parent_bundle_ids"]),
        build_route=mb.BuildRoute(manifest_dict["build_route"]),
        scope=mb.BundleScope(manifest_dict["scope"]),
        requested_capabilities=frozenset(manifest_dict["requested_capabilities"]),
        publish_status=mb.PublishStatus(manifest_dict["publish_status"]),
        core=mb.ComponentManifest(**manifest_dict["core"]),
        vocabulary=mb.ComponentManifest(**manifest_dict["vocabulary"]),
        primitives=tuple(
            mb.PrimitiveManifestEntry(
                **{**p, "provenance_status": mb.ProvenanceStatus(p["provenance_status"])}
            )
            for p in manifest_dict["primitives"]
        ),
        router=mb.RouterManifest(**manifest_dict["router"]),
        argument_scorer=mb.ArgumentScorerManifest(**manifest_dict["argument_scorer"]),
        scoring_policy=mb.ScoringPolicyManifest(**manifest_dict["scoring_policy"]),
        build_recipe_hash=manifest_dict.get("build_recipe_hash"),
        known_defects=tuple(manifest_dict.get("known_defects", ())),
    )

    loaded = mb.load_bundle(
        manifest,
        mode="nominal",
        expected_primitive_count=16,
    )

    # Reconstruct fresh runtime objects from ONLY the loaded state dicts.
    arch_cfg = SharedEncoderArchitectureConfig(seed=args.seed, vocab_size=10, device=str(device))
    arch = build_shared_encoder_architecture(arch_cfg)
    core = arch.core
    core.model.load_state_dict(loaded.core_state_dict)
    core.model.to(device)
    core.model.eval()
    for p in core.model.parameters():
        p.requires_grad_(False)

    bank, fresh_op_to_id = _rebuild_16_bank_structure(core, args.seed)
    assert fresh_op_to_id == op_to_id, (
        f"fresh-process bank structure produced a different op_to_id mapping "
        f"than the build process recorded: {fresh_op_to_id} != {op_to_id}"
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

    router = Router(RouterConfig(d_model=core.model.config.d_model, top_k=1, score_fn="dot"))
    # A fresh Router starts with an empty `_keys` ParameterDict -- unlike the
    # bank/core/scorer, its per-primitive key parameters are registered
    # dynamically (`add_primitive_key`), not present in a freshly-constructed
    # instance. The training-side calibration helper registers these keys as
    # part of its own setup; a structural rebuild needs the same step before
    # `load_state_dict` can see a matching key set.
    for pid in sorted(op_to_id.values()):
        router.add_primitive_key(pid)
    router.load_state_dict(loaded.router_state_dict)
    router.to(device)
    router.eval()

    scorer = ArgumentScorer(ArgumentScorerConfig(d_model=core.model.config.d_model))
    scorer.load_state_dict(loaded.argument_scorer_state_dict)
    scorer.to(device)
    scorer.eval()

    predictions_by_op: dict[str, list[list[int]]] = {}
    exact_match_by_op: dict[str, float] = {}
    for op, pid in sorted(op_to_id.items()):
        prim = bank.get(pid)
        if op in PARAMETERIZED_OPERATION_NAMES:
            n_groups = max(1, math.ceil(args.sample_examples_per_op / 3))
            groups = generate_compact_operator_counterfactual_groups(
                args.seed,
                n_groups,
                operation=op,
                step=0,
                split=args.split,
                vocab_size=10,
                sequence_length_range=(6, 10),
                group_size=3,
            )
            examples, _wrong_arg_map, _ = _flatten_groups(groups)
            examples = examples[: args.sample_examples_per_op]
            provider = _make_arg_provider_correct(op)
        else:
            examples = _generate_parameter_free_examples(
                args.seed,
                args.sample_examples_per_op,
                operation=op,
                split=args.split,
                vocab_size=10,
                sequence_length_range=(6, 10),
            )
            provider = None

        em, _tok = _evaluate_primitive_arm(core, prim, examples, op, argument_provider=provider)
        exact_match_by_op[op] = em

        # Record actual discrete token predictions (post-argmax) for a
        # sample-ID-keyed byte-exact reproducibility comparison.
        preds: list[list[int]] = []
        with torch.no_grad():
            from apc.core.data import collate_content_only_batch
            from apc.environments.operations import get_operation
            from apc.primitives.primitive import (
                CrossPositionPrimitive,
                ReverseRelativePrimitive,
                ShiftRelativePrimitive,
            )

            content_lengths = [len(ex.input_tokens) for ex in examples]
            output_lengths = [get_operation(op).output_length(n) for n in content_lengths]
            batch_input = collate_content_only_batch(examples, core.tokens, device=device)
            h = core.model.encode(batch_input)[:, 1 : 1 + max(content_lengths), :]
            argument_values = None if provider is None else [provider(ex) for ex in examples]
            if isinstance(
                prim, (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive)
            ):
                logits = prim(h, content_lengths, output_lengths, argument_values)
            else:
                logits = prim(h)
            pred_tokens = logits.argmax(dim=-1)
            for row, n in enumerate(output_lengths):
                preds.append(pred_tokens[row, :n].tolist())
        predictions_by_op[op] = preds

    result = {
        "seed": args.seed,
        "python_executable": sys.executable,
        "device": str(device),
        "no_builder_identifier_found_in_this_script": not forbidden_found,
        "forbidden_identifiers_checked": list(FORBIDDEN_IDENTIFIERS),
        "forbidden_identifiers_found": forbidden_found,
        "bundle_id": manifest.bundle_id,
        "checks_performed_by_load_bundle": list(loaded.checks_performed),
        "sample_examples_per_op": args.sample_examples_per_op,
        "exact_match_by_op": exact_match_by_op,
        "predictions_by_op": predictions_by_op,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
