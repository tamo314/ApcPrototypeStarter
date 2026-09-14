"""Task D-014 — Post-repair stepwise causal localization over D-013 artifacts.

Read-only diagnostic. Loads the five D-013 ``FROZEN_PARENT`` bundles and the
five D-013 ``LOCAL_SORT_REPAIR`` candidates exactly as saved
(``runs/phase_d_d013_five_model_cohort``, ``runs/phase_d_d013_sort_repair``),
and re-evaluates the same 7-class target panel at the same fixed evaluation
seed (301, the first of D-013's own ``EVAL_SEEDS``) with the same per-class
sample size (1,000). No optimizer is constructed, no parameter is updated, no
new seed is introduced, and no D-013 artifact is modified.

For each of the 7 target classes (``SELECT`` always immediately followed by
``SORT``), four comparison conditions are evaluated:

  (A) Continuous execution: the unmodified multi-step forward pass.
  (B) Diagnostic reset immediately after SELECT only: SELECT's own output is
      replaced by the deterministic oracle function applied to whatever
      content SELECT actually received (i.e. only SELECT's own execution
      error is corrected; anything upstream of SELECT is left exactly as
      continuous execution produced it). Every step after SELECT then runs
      continuously (the model's own forward), uncorrected.
  (C) Diagnostic reset immediately after SORT only: symmetric to (B), but the
      single corrected boundary is SORT's own output, computed by applying
      ``SortOp.apply`` to whatever content SORT actually received under
      unmodified continuous execution. SELECT is *not* corrected under (C).
  (D) Standalone, length-matched primitive execution: at every step boundary,
      the primitive that owns that step is evaluated in isolation (a fresh
      content-only encoding of exactly the boundary content continuous
      execution actually produced at that point -- no composition chaining),
      scored against Correct / Wrong-family / None targets, plus
      Wrong-argument for parameterized steps (SELECT here).

Every reset in (B)/(C) is diagnostic-only: it is never treated as a success
metric or fed back as a training signal (no optimizer exists in this module
at all), matching the existing NRQ-007 diagnostic-reset convention.

Attribution rule (pre-fixed before any run, not tuned to observed results):
for each (model, condition, class) cell, take the earliest step whose
condition-(A) EM drops below the reused NRQ-007 floors (0.85 for a
non-terminal step, 0.95 for the terminal step). If no step fails, the cell is
``PASSED``. Otherwise:

  - failing step is SORT and its own standalone (D) EM there is below 0.95:
    ``RESIDUAL_SORT_DEFECT``.
  - failing step is the final (post-SORT) step and its own standalone (D) EM
    there is below 0.95: ``DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT``.
  - failing step is SELECT and its own standalone (D) EM there is below 0.95:
    ``SELECT_OWN_DEFECT_OUT_OF_SCOPE`` (not one of the four requested
    categories, reported for completeness rather than silently dropped).
  - otherwise, if the fraction of examples whose actual realized input at the
    failing step already differs from the globally-correct oracle input is
    high (upstream cleanliness below 0.95): ``UPSTREAM_ERROR_ACCUMULATION``.
  - otherwise (standalone succeeds and upstream was clean, yet continuous
    execution still fails at that exact boundary): ``INTERFACE_FAILURE``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from apc.core.data import pad_token_sequences
from apc.environments.generator import Example, oracle_calls_for_example
from apc.environments.operations import get_operation
from apc.environments.primitive_call import PrimitiveCall
from apc.environments.vocab import DEFAULT_VOCAB_SIZE
from apc.evaluation.nrq007_stepwise_causal_attribution import get_wrong_argument
from apc.evaluation.phase_d_executor import (
    EVAL_SEEDS,
    MODEL_SEEDS,
    TARGET_CLASSES,
    _em,
    _examples_for_recipe,
    manifest_from_json_dict,
)
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    build_shared_encoder_architecture,
)
from apc.evaluation.unified_oracle_causal_benchmark import (
    DEFAULT_WRONG_FAMILY_ARGUMENTS,
    WRONG_FAMILY_MAP,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.composition import extract_argument_value
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    ReverseRelativePrimitive,
    ShiftRelativePrimitive,
)
from apc.utils import model_bundle as mb

REPO_ROOT = Path(__file__).resolve().parents[3]

TASK_ID = "D-014"
PARENT_COHORT_ROOT = Path("runs/phase_d_d013_five_model_cohort")
CANDIDATE_ROOT = Path("runs/phase_d_d013_sort_repair")
EXECUTOR_REPORT_ROOT = Path("runs/phase_d_d013_executor")
OUTPUT_ROOT = Path("runs/phase_d_d014_stepwise_causal_localization")

CONDITIONS: tuple[str, ...] = ("FROZEN_PARENT", "LOCAL_SORT_REPAIR")
EVAL_SEED = EVAL_SEEDS[0]  # 301 -- reused verbatim from D-013, not a new seed.
N_EXAMPLES = 1_000  # matches D-013's per-eval-seed target-panel sample size
INITIAL_LENGTH_RANGE = (6, 10)  # matches D-013's target-panel initial-length range
BATCH = 128

STEP_FLOOR_NONFINAL = 0.85  # reused verbatim from NRQ-007's ORACLE_FLOOR_THRESHOLD
STEP_FLOOR_FINAL = 0.95  # reused verbatim from NRQ-007's CLOSURE_CONFIRMATION_THRESHOLD
STANDALONE_DEFECT_FLOOR = 0.95  # matches D-001 pilot's own target-recovery floor
UPSTREAM_CLEAN_FLOOR = 0.95

_PRIMITIVE_CLASSES = (CrossPositionPrimitive, ShiftRelativePrimitive, ReverseRelativePrimitive)

VOCAB_SIZE = DEFAULT_VOCAB_SIZE


class D014DiagnosticError(RuntimeError):
    """A prerequisite artifact or parity check failed; do not fabricate results."""


# =============================================================================
# 1. Read-only bundle loading (parent / candidate), mirroring
#    scripts/phase_d_fresh_load_check.py's independent-process pattern.
# =============================================================================


def _bundle_paths(seed: int, condition: str) -> tuple[Path, Literal["nominal", "diagnostic"]]:
    if condition == "FROZEN_PARENT":
        return (
            REPO_ROOT / PARENT_COHORT_ROOT / f"seed_{seed}" / "parent" / "manifest.json",
            "nominal",
        )
    if condition == "LOCAL_SORT_REPAIR":
        return (
            REPO_ROOT / CANDIDATE_ROOT / f"seed_{seed}" / "candidate" / "candidate_manifest.json",
            "diagnostic",
        )
    raise ValueError(f"unknown condition: {condition}")


def load_bundle_for_condition(
    seed: int, condition: str, device: torch.device
) -> tuple[Any, PrimitiveBank, dict[str, int], mb.ModelBundleManifest]:
    """Load a D-013 bundle read-only. Raises if the artifact is missing."""
    manifest_path, mode = _bundle_paths(seed, condition)
    if not manifest_path.is_file():
        raise D014DiagnosticError(f"D-013 artifact missing, cannot proceed: {manifest_path}")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = manifest_from_json_dict(record["manifest"])
    op_to_id = {str(key): int(value) for key, value in record["op_to_id"].items()}
    loaded = mb.load_bundle(manifest, mode=mode, expected_primitive_count=16)

    bank_structure = Path(record["primitive_bank_structure"])
    if not bank_structure.is_file():
        bank_structure = manifest_path.parent / "primitive_bank_structure.json"
    if not bank_structure.is_file():
        raise D014DiagnosticError(f"D-013 bank structure artifact missing for seed {seed}")

    arch = build_shared_encoder_architecture(
        SharedEncoderArchitectureConfig(
            seed=manifest.model_seed, vocab_size=VOCAB_SIZE, device=str(device)
        )
    )
    core = arch.core
    core.model.load_state_dict(loaded.core_state_dict, strict=True)
    core.model.to(device)
    core.model.eval()
    for parameter in core.model.parameters():
        parameter.requires_grad_(False)

    bank = PrimitiveBank.load_manifest(bank_structure)
    bank.load_state_dict(
        {
            f"_primitives.{primitive_id}.{key}": value
            for primitive_id, state in loaded.primitive_state_dicts.items()
            for key, value in state.items()
        },
        strict=True,
    )
    bank.to(device)
    bank.freeze_all()
    bank.eval()
    for primitive_id in bank.ids():
        for parameter in bank.get(primitive_id).parameters():
            parameter.requires_grad_(False)
    return core, bank, op_to_id, manifest


# =============================================================================
# 2. Oracle chain reconstruction (pure, deterministic; no model access).
# =============================================================================


def oracle_chain(
    example: Example, recipe: Sequence[str], vocab_size: int = VOCAB_SIZE
) -> list[tuple[int, ...]]:
    """Reconstruct the globally-correct per-step oracle token chain for one example."""
    calls = oracle_calls_for_example(example)
    if tuple(call.operation for call in calls) != tuple(recipe):
        raise D014DiagnosticError("oracle call sequence does not match the registered recipe")
    chain: list[tuple[int, ...]] = [tuple(example.input_tokens)]
    for step_idx, op_name in enumerate(recipe):
        op_def = get_operation(op_name)
        chain.append(op_def.apply(chain[-1], vocab_size, calls[step_idx].arguments))
    if chain[-1] != tuple(example.target_tokens):
        raise D014DiagnosticError("oracle re-derivation mismatch vs. example.target_tokens")
    return chain


def call_arguments(example: Example, recipe: Sequence[str]) -> tuple[dict[str, Any], ...]:
    calls = oracle_calls_for_example(example)
    if tuple(call.operation for call in calls) != tuple(recipe):
        raise D014DiagnosticError("oracle call sequence does not match the registered recipe")
    return tuple(dict(call.arguments) for call in calls)


# =============================================================================
# 3. Primitive forward helpers.
# =============================================================================


def _forward_primitive(
    primitive: Any, h: torch.Tensor, lengths: list[int], out_lengths: list[int], arg_values: Any
) -> torch.Tensor:
    if isinstance(primitive, _PRIMITIVE_CLASSES):
        return primitive(h, lengths, out_lengths, arg_values)
    return primitive(h)


def _encode_fresh(core: Any, contents: Sequence[tuple[int, ...]]) -> torch.Tensor:
    """Fresh, task-blind, content-only encoding -- identical framing to
    ``_encode_initial_content``/``_encode_intermediate_tokens`` (both wrap
    exactly ``[BOS] content [SEP]``), used for both continuous re-encoding
    and standalone (condition D) boundary evaluation."""
    framed = [(core.tokens.bos, *tuple(content), core.tokens.sep) for content in contents]
    padded = pad_token_sequences(framed, core.tokens.pad, device=core.device)
    hidden = core.model.encode(padded)
    max_len = max(len(content) for content in contents)
    return hidden[:, 1 : 1 + max_len, :]


def _step_forward(
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    op_name: str,
    h: torch.Tensor,
    lengths: list[int],
    arg_dicts: Sequence[dict[str, Any]] | None,
) -> tuple[list[tuple[int, ...]], list[int]]:
    op_def = get_operation(op_name)
    out_lengths = [op_def.output_length(length) for length in lengths]
    is_param = bool(op_def.required_argument_names)
    arg_values = (
        [extract_argument_value(op_name, PrimitiveCall(op_name, args)) for args in arg_dicts]
        if is_param and arg_dicts is not None
        else None
    )
    primitive = bank.get(op_to_id[op_name])
    with torch.no_grad():
        logits = _forward_primitive(primitive, h, lengths, out_lengths, arg_values)
        preds = logits.argmax(dim=-1)
    token_seqs = [tuple(preds[b, : out_lengths[b]].tolist()) for b in range(len(lengths))]
    return token_seqs, out_lengths


def _run_continuous_chunk(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    recipe: Sequence[str],
    contents: Sequence[tuple[int, ...]],
    args_by_step: Sequence[Sequence[dict[str, Any]]],
) -> list[list[tuple[int, ...]]]:
    """Continuous forward for one chunk. Returns boundary_input[step] for
    step in 0..len(recipe) (index 0 is the raw content; index k is the
    model's own realized input to step k, i.e. its predecessor's output)."""
    boundary_input: list[list[tuple[int, ...]]] = [list(contents)]
    lengths = [len(content) for content in contents]
    h = _encode_fresh(core, contents)
    for step_idx, op_name in enumerate(recipe):
        preds, out_lengths = _step_forward(
            bank, op_to_id, op_name, h, lengths, args_by_step[step_idx]
        )
        boundary_input.append(preds)
        lengths = out_lengths
        if step_idx < len(recipe) - 1:
            h = _encode_fresh(core, preds)
    return boundary_input


def _run_suffix_chunk(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    recipe: Sequence[str],
    start_idx: int,
    start_contents: Sequence[tuple[int, ...]],
    args_by_step: Sequence[Sequence[dict[str, Any]]],
) -> list[tuple[int, ...]]:
    """Continuous forward for recipe[start_idx:], starting from corrected content.
    Returns the final step's predicted tokens for this chunk."""
    lengths = [len(content) for content in start_contents]
    h = _encode_fresh(core, start_contents)
    preds: list[tuple[int, ...]] = list(start_contents)
    for step_idx in range(start_idx, len(recipe)):
        op_name = recipe[step_idx]
        preds, out_lengths = _step_forward(
            bank, op_to_id, op_name, h, lengths, args_by_step[step_idx]
        )
        lengths = out_lengths
        if step_idx < len(recipe) - 1:
            h = _encode_fresh(core, preds)
    return preds


# =============================================================================
# 4. Standalone (condition D) causal-control battery on an exact boundary input.
# =============================================================================


@dataclass(frozen=True)
class StandaloneArmResult:
    n: int
    correct_em: float
    wrong_family_em: float
    none_em: float
    wrong_argument_em: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def standalone_arms(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    op_name: str,
    contents: Sequence[tuple[int, ...]],
    arg_dicts: Sequence[dict[str, Any]],
    vocab_size: int = VOCAB_SIZE,
) -> tuple[StandaloneArmResult, list[bool]]:
    """Correct / Wrong-family / None (+ Wrong-argument) on the exact given boundary content."""
    n = len(contents)
    lengths = [len(content) for content in contents]
    op_def = get_operation(op_name)
    is_param = bool(op_def.required_argument_names)
    targets = [op_def.apply(contents[i], vocab_size, arg_dicts[i]) for i in range(n)]
    h = _encode_fresh(core, contents)

    correct_args = (
        [extract_argument_value(op_name, PrimitiveCall(op_name, arg_dicts[i])) for i in range(n)]
        if is_param
        else None
    )
    pred_correct, out_lens = _step_forward(
        bank, op_to_id, op_name, h, lengths, arg_dicts if is_param else None
    )
    correct_matches = [pred_correct[i] == targets[i] for i in range(n)]

    wf_op = WRONG_FAMILY_MAP[op_name]
    wf_def = get_operation(wf_op)
    wf_is_param = bool(wf_def.required_argument_names)
    wf_arg_dicts = None
    if wf_is_param:
        default = DEFAULT_WRONG_FAMILY_ARGUMENTS[wf_op]
        wf_arg_dicts = [{next(iter(wf_def.required_argument_names)): default}] * n
    pred_wf, wf_out_lens = _step_forward(bank, op_to_id, wf_op, h, lengths, wf_arg_dicts)
    wf_matches = [pred_wf[i] == targets[i] for i in range(n)]

    primitive = bank.get(op_to_id[op_name])
    primitive.enabled = False
    try:
        pred_none, none_out_lens = _step_forward(
            bank, op_to_id, op_name, h, lengths, arg_dicts if is_param else None
        )
    finally:
        primitive.enabled = True
    none_matches = [pred_none[i] == targets[i] for i in range(n)]

    wrong_argument_em: float | None = None
    if is_param and correct_args is not None:
        wrong_args_raw = [
            get_wrong_argument(op_name, correct_args[i], lengths[i], vocab_size) for i in range(n)
        ]
        req_name = next(iter(op_def.required_argument_names))
        wrong_arg_dicts = [{req_name: value} for value in wrong_args_raw]
        pred_wa, wa_out_lens = _step_forward(bank, op_to_id, op_name, h, lengths, wrong_arg_dicts)
        wa_matches = [pred_wa[i] == targets[i] for i in range(n)]
        wrong_argument_em = sum(wa_matches) / n

    result = StandaloneArmResult(
        n=n,
        correct_em=sum(correct_matches) / n,
        wrong_family_em=sum(wf_matches) / n,
        none_em=sum(none_matches) / n,
        wrong_argument_em=wrong_argument_em,
    )
    return result, correct_matches


# =============================================================================
# 5. Attribution rule (fixed before execution; see module docstring).
# =============================================================================

ATTRIBUTION_PASSED = "PASSED"
ATTRIBUTION_RESIDUAL_SORT_DEFECT = "RESIDUAL_SORT_DEFECT"
ATTRIBUTION_DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT = (
    "DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT"
)
ATTRIBUTION_SELECT_OWN_DEFECT = "SELECT_OWN_DEFECT_OUT_OF_SCOPE"
ATTRIBUTION_UPSTREAM_ERROR_ACCUMULATION = "UPSTREAM_ERROR_ACCUMULATION"
ATTRIBUTION_INTERFACE_FAILURE = "INTERFACE_FAILURE"
ATTRIBUTION_UNCLASSIFIED = "UNCLASSIFIED_BOUNDARY"


def earliest_failing_step(continuous_em: Sequence[float]) -> int | None:
    n = len(continuous_em)
    for step_idx, em in enumerate(continuous_em):
        floor = STEP_FLOOR_FINAL if step_idx == n - 1 else STEP_FLOOR_NONFINAL
        if em < floor:
            return step_idx
    return None


def classify_attribution(
    *,
    fail_step: int | None,
    idx_select: int,
    idx_sort: int,
    last_idx: int,
    standalone_correct_em: Sequence[float],
    upstream_clean_fraction: Sequence[float],
) -> str:
    """Pre-fixed rule; see module docstring for the full decision table."""
    if fail_step is None:
        return ATTRIBUTION_PASSED
    d_here = standalone_correct_em[fail_step]
    clean_here = upstream_clean_fraction[fail_step]
    if fail_step == idx_sort:
        if d_here < STANDALONE_DEFECT_FLOOR:
            return ATTRIBUTION_RESIDUAL_SORT_DEFECT
    elif fail_step == last_idx and fail_step != idx_sort:
        if d_here < STANDALONE_DEFECT_FLOOR:
            return ATTRIBUTION_DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT
    elif fail_step == idx_select:
        if d_here < STANDALONE_DEFECT_FLOOR:
            return ATTRIBUTION_SELECT_OWN_DEFECT
    else:
        return ATTRIBUTION_UNCLASSIFIED
    if clean_here < UPSTREAM_CLEAN_FLOOR:
        return ATTRIBUTION_UPSTREAM_ERROR_ACCUMULATION
    return ATTRIBUTION_INTERFACE_FAILURE


# =============================================================================
# 6. Per-cell (model, condition, class) evaluation.
# =============================================================================


@dataclass(frozen=True)
class StepRecord:
    step_idx: int
    operation: str
    is_parameterized: bool
    continuous_em: float
    upstream_clean_fraction: float
    standalone: StandaloneArmResult
    standalone_clean_subset: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "operation": self.operation,
            "is_parameterized": self.is_parameterized,
            "continuous_em": self.continuous_em,
            "upstream_clean_fraction": self.upstream_clean_fraction,
            "standalone": self.standalone.to_dict(),
            "standalone_clean_subset": self.standalone_clean_subset,
        }


@dataclass(frozen=True)
class ResetResult:
    boundary_step_idx: int
    final_em: float
    a_failures: int
    rescued: int
    rescue_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellResult:
    seed: int
    condition: str
    canonical_class: str
    recipe: list[str]
    n: int
    idx_select: int
    idx_sort: int
    steps: list[StepRecord]
    final_em_A: float
    reset_B: ResetResult
    reset_C: ResetResult
    earliest_failing_step: int | None
    attribution: str
    parity_check: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "condition": self.condition,
            "canonical_class": self.canonical_class,
            "recipe": self.recipe,
            "n": self.n,
            "idx_select": self.idx_select,
            "idx_sort": self.idx_sort,
            "steps": [s.to_dict() for s in self.steps],
            "final_em_A": self.final_em_A,
            "reset_B": self.reset_B.to_dict(),
            "reset_C": self.reset_C.to_dict(),
            "earliest_failing_step": self.earliest_failing_step,
            "attribution": self.attribution,
            "parity_check": self.parity_check,
        }


def _load_d013_expected(seed: int, condition: str, klass: str) -> dict[str, Any] | None:
    report_path = REPO_ROOT / EXECUTOR_REPORT_ROOT / f"seed_{seed}.json"
    if not report_path.is_file():
        return None
    record = json.loads(report_path.read_text(encoding="utf-8"))
    panel_key = "frozen_parent" if condition == "FROZEN_PARENT" else "repaired"
    panel = record.get(panel_key, {}).get(f"target:{klass}")
    if not panel:
        return None
    for row in panel.get("per_evaluation_seed", ()):
        if row.get("seed") == EVAL_SEED:
            return row
    return None


def evaluate_cell(
    core: Any,
    bank: PrimitiveBank,
    op_to_id: dict[str, int],
    seed: int,
    condition: str,
    klass: str,
    vocab_size: int = VOCAB_SIZE,
) -> CellResult:
    recipe = tuple(klass.split("->"))
    idx_select = recipe.index("SELECT")
    idx_sort = idx_select + 1
    if recipe[idx_sort] != "SORT":
        raise D014DiagnosticError(f"class {klass} does not have SORT immediately after SELECT")
    last_idx = len(recipe) - 1

    examples = _examples_for_recipe(EVAL_SEED, recipe, N_EXAMPLES, INITIAL_LENGTH_RANGE)
    n = len(examples)

    # Parity precondition: reproduce D-013's own recorded target-panel digest for this
    # exact (seed, condition, class, eval_seed) cell before trusting any further computation.
    successes, total, outputs_hash = _em(core, bank, op_to_id, examples)
    expected = _load_d013_expected(seed, condition, klass)
    parity_check: dict[str, Any] = {
        "recomputed_successes": successes,
        "recomputed_n": total,
        "recomputed_outputs_sha256": outputs_hash,
        "d013_expected": expected,
        "match": expected is not None
        and expected.get("successes") == successes
        and expected.get("n") == total
        and expected.get("outputs_sha256") == outputs_hash,
    }
    if expected is not None and not parity_check["match"]:
        raise D014DiagnosticError(
            f"parity check against D-013 seed_{seed}.json FAILED for "
            f"{condition}:{klass} at eval_seed={EVAL_SEED}: {parity_check}"
        )

    gt_chains = [oracle_chain(ex, recipe, vocab_size) for ex in examples]
    step_args = [call_arguments(ex, recipe) for ex in examples]
    args_by_step: list[list[dict[str, Any]]] = [
        [step_args[i][step_idx] for i in range(n)] for step_idx in range(len(recipe))
    ]

    boundary_input: list[list[tuple[int, ...]]] = [[] for _ in range(len(recipe) + 1)]
    final_B: list[tuple[int, ...]] = []
    final_C: list[tuple[int, ...]] = []

    for start in range(0, n, BATCH):
        chunk_examples = examples[start : start + BATCH]
        chunk_args = [
            args_by_step[step_idx][start : start + BATCH] for step_idx in range(len(recipe))
        ]
        contents = [tuple(ex.input_tokens) for ex in chunk_examples]

        chunk_boundaries = _run_continuous_chunk(core, bank, op_to_id, recipe, contents, chunk_args)
        for step_idx in range(len(recipe) + 1):
            boundary_input[step_idx].extend(chunk_boundaries[step_idx])

        select_input = chunk_boundaries[idx_select]
        select_args = chunk_args[idx_select]
        corrected_select_output = [
            get_operation("SELECT").apply(select_input[i], vocab_size, select_args[i])
            for i in range(len(chunk_examples))
        ]
        final_B.extend(
            _run_suffix_chunk(
                core, bank, op_to_id, recipe, idx_sort, corrected_select_output, chunk_args
            )
        )

        sort_input = chunk_boundaries[idx_sort]
        sort_args = chunk_args[idx_sort]
        corrected_sort_output = [
            get_operation("SORT").apply(sort_input[i], vocab_size, sort_args[i])
            for i in range(len(chunk_examples))
        ]
        if idx_sort == last_idx:
            final_C.extend(corrected_sort_output)
        else:
            final_C.extend(
                _run_suffix_chunk(
                    core, bank, op_to_id, recipe, idx_sort + 1, corrected_sort_output, chunk_args
                )
            )

    gt_by_step = [[gt_chains[i][step_idx] for i in range(n)] for step_idx in range(len(recipe) + 1)]
    final_target = gt_by_step[-1]

    continuous_em = [
        sum(boundary_input[step_idx + 1][i] == gt_by_step[step_idx + 1][i] for i in range(n)) / n
        for step_idx in range(len(recipe))
    ]
    upstream_clean = [
        sum(boundary_input[step_idx][i] == gt_by_step[step_idx][i] for i in range(n)) / n
        for step_idx in range(len(recipe))
    ]

    a_correct = [boundary_input[-1][i] == final_target[i] for i in range(n)]
    b_correct = [final_B[i] == final_target[i] for i in range(n)]
    c_correct = [final_C[i] == final_target[i] for i in range(n)]
    a_failures = sum(1 for x in a_correct if not x)

    def _reset_result(
        step_idx: int, correct: list[bool], final_preds: list[tuple[int, ...]]
    ) -> ResetResult:
        rescued = sum(1 for i in range(n) if not a_correct[i] and correct[i])
        rate = (rescued / a_failures) if a_failures else None
        return ResetResult(
            boundary_step_idx=step_idx,
            final_em=sum(correct) / n,
            a_failures=a_failures,
            rescued=rescued,
            rescue_rate=rate,
        )

    reset_B = _reset_result(idx_select, b_correct, final_B)
    reset_C = _reset_result(idx_sort, c_correct, final_C)

    steps: list[StepRecord] = []
    standalone_correct_em: list[float] = []
    for step_idx, op_name in enumerate(recipe):
        contents = boundary_input[step_idx]
        args = args_by_step[step_idx]
        arm_result, correct_matches = standalone_arms(
            core, bank, op_to_id, op_name, contents, args, vocab_size
        )
        standalone_correct_em.append(arm_result.correct_em)
        clean_mask = [boundary_input[step_idx][i] == gt_by_step[step_idx][i] for i in range(n)]
        n_clean = sum(clean_mask)
        clean_subset: dict[str, Any] = {"n_clean": n_clean}
        min_clean_support = 20
        if n_clean >= min_clean_support:
            clean_subset["correct_em"] = sum(
                correct_matches[i] for i in range(n) if clean_mask[i]
            ) / n_clean
        else:
            clean_subset["correct_em"] = None
            clean_subset["insufficient_support"] = True
        op_def = get_operation(op_name)
        steps.append(
            StepRecord(
                step_idx=step_idx,
                operation=op_name,
                is_parameterized=bool(op_def.required_argument_names),
                continuous_em=continuous_em[step_idx],
                upstream_clean_fraction=upstream_clean[step_idx],
                standalone=arm_result,
                standalone_clean_subset=clean_subset,
            )
        )

    fail_step = earliest_failing_step(continuous_em)
    attribution = classify_attribution(
        fail_step=fail_step,
        idx_select=idx_select,
        idx_sort=idx_sort,
        last_idx=last_idx,
        standalone_correct_em=standalone_correct_em,
        upstream_clean_fraction=upstream_clean,
    )

    return CellResult(
        seed=seed,
        condition=condition,
        canonical_class=klass,
        recipe=list(recipe),
        n=n,
        idx_select=idx_select,
        idx_sort=idx_sort,
        steps=steps,
        final_em_A=continuous_em[-1],
        reset_B=reset_B,
        reset_C=reset_C,
        earliest_failing_step=fail_step,
        attribution=attribution,
        parity_check=parity_check,
    )


# =============================================================================
# 7. Top-level run.
# =============================================================================


def run(device_str: str | None = None) -> dict[str, Any]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = REPO_ROOT / OUTPUT_ROOT
    if output_root.exists():
        raise D014DiagnosticError(
            f"output namespace already exists and cannot be reused: {output_root}"
        )
    output_root.mkdir(parents=True)

    start = time.perf_counter()
    report: dict[str, Any] = {
        "task": TASK_ID,
        "eval_seed": EVAL_SEED,
        "n_examples": N_EXAMPLES,
        "model_seeds": list(MODEL_SEEDS),
        "conditions": list(CONDITIONS),
        "target_classes": list(TARGET_CLASSES),
        "sealed_access": 0,
        "optimizer_constructed": False,
        "parameter_updates": 0,
        "candidate_selected": None,
        "bundle_promotion": "NOT_AUTHORIZED",
        "cells": {},
    }
    try:
        for seed in MODEL_SEEDS:
            for condition in CONDITIONS:
                core, bank, op_to_id, manifest = load_bundle_for_condition(seed, condition, device)
                for klass in TARGET_CLASSES:
                    cell = evaluate_cell(core, bank, op_to_id, seed, condition, klass)
                    key = f"{seed}:{condition}:{klass}"
                    report["cells"][key] = cell.to_dict()
                    cell_name = f"cell_{seed}_{condition}_{klass.replace('->', '_')}.json"
                    cell_file = output_root / cell_name
                    cell_file.write_text(
                        json.dumps(cell.to_dict(), indent=2, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
        attribution_counts: dict[str, int] = {}
        for cell in report["cells"].values():
            attribution = cell["attribution"]
            attribution_counts[attribution] = attribution_counts.get(attribution, 0) + 1
        report["attribution_counts"] = attribution_counts
        report["result"] = "COMPLETED"
    finally:
        report["wall_clock_seconds"] = time.perf_counter() - start
        if torch.cuda.is_available():
            report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        (output_root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    result = run()
    summary = {
        "task": result["task"],
        "result": result["result"],
        "attribution_counts": result["attribution_counts"],
    }
    print(json.dumps(summary, indent=2))
