"""Task D-015 -- Downstream primitive length x input-order paired factorial diagnostic.

Read-only diagnostic over D-013's already-saved ``LOCAL_SORT_REPAIR`` candidate
bundles (seeds 40-44) and D-013's own registered evaluation seeds (301-305).
No training, no optimizer, no new seed, no sealed access, no candidate
selection, no promotion, and no modification of any D-013/D-014 artifact.

Motivation
----------
ADR-0182 (D-014) newly identified ``DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT``
for 8/35 ``LOCAL_SORT_REPAIR`` cells: once SORT's own defect is repaired, the
*next* primitive in ``SELECT->SORT->{NEGATE,REVERSE,SELECT,SHIFT}`` still
fails on SORT's short ({3,4,5}-token) output, while ``SELECT->SORT->BIND``
did not show this pattern. Two confounded explanations are possible for the
failing four operations: (a) the primitive is simply unreliable at short
lengths regardless of content order, or (b) the primitive is unreliable
specifically on already-*sorted* (ascending) input -- the one thing every
real SORT output has in common regardless of length -- or (c) an
interaction of both. This diagnostic isolates length from order with a
paired 2x2 factorial:

  - Length factor: SHORT (``{3,4,5}``, SORT's known-deficient domain per
    D-001) vs. LONG (``{6,...,10}``, D-013's own "existing qualified"
    regression-panel domain).
  - Order factor: UNSORTED (ordinary randomly-drawn content, matching every
    other standalone-primitive panel in this project) vs. SORTED (the same
    multiset, sorted ascending -- exactly what a real SORT output looks
    like), paired example-by-example: the same multiset and the same
    sampled primitive argument are used in both order arms; only token
    order differs.

``NEGATE``, ``REVERSE``, ``SELECT``, ``SHIFT`` are the four operations under
test; ``BIND`` is run identically as a negative control (D-014 did not
classify any ``SELECT->SORT->BIND`` cell as the downstream defect).

Gate: identity of every non-SORT primitive, and of the Stable Core, between
the already-saved ``FROZEN_PARENT`` and ``LOCAL_SORT_REPAIR`` bundles is
verified per seed from the saved manifests' own
``weights_hash``/``state_abi_hash``/``core`` digests before any computation.
Only ``LOCAL_SORT_REPAIR`` is then loaded and evaluated per seed: since the
gate proves every operand this diagnostic touches is bit-identical across
both conditions, evaluating the other condition as well would be a fully
redundant repeat of the same computation, not independent evidence.

Decision rules (fixed here, before any run; NOT tuned to observed results)
---------------------------------------------------------------------------
For each (model seed, primitive), pool exact-match rate across the
primitive's valid lengths in each length group and across all 5 registered
evaluation seeds (301-305), separately per order arm, giving four cell
means: ``EM(SHORT,UNSORTED)``, ``EM(SHORT,SORTED)``, ``EM(LONG,UNSORTED)``,
``EM(LONG,SORTED)``::

    length_margin      = mean(EM(LONG,*))   - mean(EM(SHORT,*))
    order_margin       = mean(EM(*,SORTED)) - mean(EM(*,UNSORTED))
    interaction_margin = (EM(SHORT,SORTED) - EM(SHORT,UNSORTED))
                        - (EM(LONG,SORTED) - EM(LONG,UNSORTED))

``ADEQUACY_FLOOR = 0.95`` is reused verbatim from this project's existing
``STANDALONE_DEFECT_FLOOR``/target-recovery convention. ``EFFECT_MARGIN =
0.10`` is a new, fixed threshold for this diagnostic: twice this project's
existing None-arm slack (0.05, ADR-0044/ADR-0046), so a classified effect
must exceed double the codebase's own established noise tolerance, while
remaining well below the 0.50 "large causal gap" threshold used elsewhere
(``_evaluate_causal_controls``) -- a deliberately intermediate bar for "a
practically meaningful main effect or interaction," not a formal
significance test::

    LENGTH_MAIN_EFFECT = PRESENT iff length_margin >= EFFECT_MARGIN
                                  and mean(EM(SHORT,*)) < ADEQUACY_FLOOR
                         else ABSENT
    ORDER_MAIN_EFFECT  = PRESENT iff abs(order_margin) >= EFFECT_MARGIN
                                  and min(EM(*,SORTED-mean), EM(*,UNSORTED-mean)) < ADEQUACY_FLOOR
                         else ABSENT
    INTERACTION_EFFECT = PRESENT iff abs(interaction_margin) >= EFFECT_MARGIN
                         else ABSENT

Per (seed, primitive): if none of the three effects is PRESENT and all four
cells clear ``ADEQUACY_FLOOR``: ``NO_DEFECT_DETECTED``. If some cell is
below floor but none of the three named effects clears ``EFFECT_MARGIN`` (a
uniform deficiency that does not cleanly localize to length, order, or
their interaction): ``UNDIFFERENTIATED_DEFICIT``, reported rather than
forced into one of the three named effects. Otherwise:
``EXPLAINED_BY_<LENGTH|ORDER|INTERACTION[+...]>``.

Non-reproducibility: for each primitive, each of the three effects is also
classified independently per seed. If all 5 model seeds agree (all PRESENT
or all ABSENT), the effect is ``REPRODUCIBLE_PRESENT``/``REPRODUCIBLE_ABSENT``;
otherwise ``NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS`` (the per-seed split is
reported alongside, never silently averaged away).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from apc.environments.operations import get_operation
from apc.evaluation.phase_d_d014_stepwise_causal_localization import (
    VOCAB_SIZE,
    _bundle_paths,
    load_bundle_for_condition,
    standalone_arms,
)
from apc.evaluation.phase_d_executor import EVAL_SEEDS, MODEL_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[3]

TASK_ID = "D-015"
OUTPUT_ROOT = Path("runs/phase_d_d015_downstream_length_order_factorial")

TARGET_PRIMITIVES: tuple[str, ...] = ("NEGATE", "REVERSE", "SELECT", "SHIFT", "BIND")
NEGATIVE_CONTROL = "BIND"
SHORT_LENGTHS: tuple[int, ...] = (3, 4, 5)
LONG_LENGTHS: tuple[int, ...] = (6, 7, 8, 9, 10)
ORDER_ARMS: tuple[str, ...] = ("UNSORTED", "SORTED")
N_EXAMPLES = 1_000  # matches D-013's own per-eval-seed target-panel sample size

ADEQUACY_FLOOR = 0.95  # reused verbatim from D-014's STANDALONE_DEFECT_FLOOR
EFFECT_MARGIN = 0.10  # fixed here; see module docstring for the rationale


class D015DiagnosticError(RuntimeError):
    """A prerequisite artifact or gate check failed; do not fabricate results."""


# =============================================================================
# 1. Non-SORT primitive / Stable Core identity gate (pure logic + I/O wrapper).
# =============================================================================


def _primitive_hash_map(record: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        str(entry["operation_name"]): (str(entry["weights_hash"]), str(entry["state_abi_hash"]))
        for entry in record["manifest"]["primitives"]
    }


def check_non_sort_identity(
    parent_hashes: dict[str, tuple[str, str]],
    candidate_hashes: dict[str, tuple[str, str]],
    parent_core: dict[str, Any],
    candidate_core: dict[str, Any],
) -> dict[str, Any]:
    """Pure gate logic: every non-SORT primitive and the Stable Core must be
    bit-identical between the saved FROZEN_PARENT and LOCAL_SORT_REPAIR
    manifests. Raises rather than silently proceeding on any mismatch."""
    if set(parent_hashes) != set(candidate_hashes):
        raise D015DiagnosticError("parent/candidate primitive sets differ")
    if parent_core != candidate_core:
        raise D015DiagnosticError("parent/candidate Stable Core manifests differ")
    mismatches = {
        op: {"parent": parent_hashes[op], "candidate": candidate_hashes[op]}
        for op in parent_hashes
        if op != "SORT" and parent_hashes[op] != candidate_hashes[op]
    }
    if mismatches:
        raise D015DiagnosticError(f"non-SORT primitive hash identity gate failed: {mismatches}")
    return {
        "non_sort_ops_checked": sorted(op for op in parent_hashes if op != "SORT"),
        "core_identical": True,
        "gate_pass": True,
    }


def verify_non_sort_hash_identity(seed: int) -> dict[str, Any]:
    """I/O wrapper: read the already-saved D-013 manifests and run the gate."""
    parent_path, _ = _bundle_paths(seed, "FROZEN_PARENT")
    candidate_path, _ = _bundle_paths(seed, "LOCAL_SORT_REPAIR")
    if not parent_path.is_file():
        raise D015DiagnosticError(f"D-013 artifact missing, cannot proceed: {parent_path}")
    if not candidate_path.is_file():
        raise D015DiagnosticError(f"D-013 artifact missing, cannot proceed: {candidate_path}")
    parent_record = json.loads(parent_path.read_text(encoding="utf-8"))
    candidate_record = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = check_non_sort_identity(
        _primitive_hash_map(parent_record),
        _primitive_hash_map(candidate_record),
        parent_record["manifest"]["core"],
        candidate_record["manifest"]["core"],
    )
    result["seed"] = seed
    return result


# =============================================================================
# 2. Paired (unsorted, sorted) example generation for one (op, length) cell.
# =============================================================================


def _length_group(length: int) -> str:
    if length in SHORT_LENGTHS:
        return "SHORT"
    if length in LONG_LENGTHS:
        return "LONG"
    raise D015DiagnosticError(f"length {length} is outside the registered SHORT/LONG groups")


MAX_PAIR_DRAW_ATTEMPTS = 10_000


def _paired_examples(
    op_name: str, length: int, eval_seed: int, n: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[dict[str, Any]]]:
    """Generate ``n`` paired (unsorted, sorted) examples sharing a multiset and
    a sampled argument per pair; only token order differs between the two.

    Each pair is accepted only by rejection sampling under two invariants,
    enforced before the pair is ever seen by a model (never by discarding
    results after the fact):

      - the UNSORTED draw must genuinely be out of ascending order. A content
        that happens to already be ascending (non-trivial probability at
        SHORT lengths, e.g. ~1/6 for three distinct tokens) contributes no
        order contrast and would silently dilute the UNSORTED arm with
        SORTED-equivalent input.
      - for BIND, the sampled ``query_key`` (drawn from the UNSORTED
        content's key positions, ``sequence[0::2]``) must also occur among
        the SORTED content's key positions. ``BindOp.apply`` silently
        returns ``0`` for a key that is not present rather than raising, so
        without this check the SORTED arm could score against a degenerate,
        order-scrambled target instead of the same well-defined lookup.
    """
    op_def = get_operation(op_name)
    if not op_def.is_valid_for_length(length):
        raise D015DiagnosticError(f"{op_name} is not valid for length {length}")
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(
                f"phase-d-d015:{op_name}:{length}:{eval_seed}:{n}".encode()
            ).digest()[:8],
            "big",
        )
    )
    unsorted_contents: list[tuple[int, ...]] = []
    sorted_contents: list[tuple[int, ...]] = []
    arg_dicts: list[dict[str, Any]] = []
    for _ in range(n):
        for _attempt in range(MAX_PAIR_DRAW_ATTEMPTS):
            content = tuple(rng.randrange(VOCAB_SIZE) for _ in range(length))
            ordered = tuple(sorted(content))
            if content == ordered:
                continue  # UNSORTED draw must genuinely differ in order
            params = op_def.sample_params(rng, content, VOCAB_SIZE)
            if op_name == "BIND" and params["query_key"] not in ordered[0::2]:
                continue  # query_key must remain a valid lookup key once sorted
            unsorted_contents.append(content)
            sorted_contents.append(ordered)
            arg_dicts.append(params)
            break
        else:
            raise D015DiagnosticError(
                f"could not draw a valid paired example for {op_name} length={length} "
                f"eval_seed={eval_seed} after {MAX_PAIR_DRAW_ATTEMPTS} attempts"
            )
    return unsorted_contents, sorted_contents, arg_dicts


# =============================================================================
# 3. Per (seed, primitive, eval_seed, length) paired evaluation.
# =============================================================================


@dataclass(frozen=True)
class OrderCellResult:
    length: int
    length_group: str
    order: str
    n: int
    correct_successes: int
    correct_em: float
    wrong_family_em: float
    none_em: float
    wrong_argument_em: float | None


@dataclass(frozen=True)
class PairedDifference:
    length: int
    eval_seed: int
    n: int
    both_correct: int
    both_wrong: int
    only_unsorted_correct: int
    only_sorted_correct: int
    em_unsorted: float
    em_sorted: float
    paired_em_diff: float


def evaluate_order_cell(
    core: Any,
    bank: Any,
    op_to_id: dict[str, int],
    op_name: str,
    length: int,
    eval_seed: int,
) -> tuple[OrderCellResult, OrderCellResult, list[bool], list[bool]]:
    """Evaluate one (primitive, length, eval_seed) pair in both order arms."""
    unsorted_contents, sorted_contents, arg_dicts = _paired_examples(
        op_name, length, eval_seed, N_EXAMPLES
    )
    group = _length_group(length)
    cells: list[OrderCellResult] = []
    matches: list[list[bool]] = []
    for order, contents in (("UNSORTED", unsorted_contents), ("SORTED", sorted_contents)):
        arm, correct_matches = standalone_arms(
            core, bank, op_to_id, op_name, contents, arg_dicts, VOCAB_SIZE
        )
        cells.append(
            OrderCellResult(
                length=length,
                length_group=group,
                order=order,
                n=arm.n,
                correct_successes=sum(correct_matches),
                correct_em=arm.correct_em,
                wrong_family_em=arm.wrong_family_em,
                none_em=arm.none_em,
                wrong_argument_em=arm.wrong_argument_em,
            )
        )
        matches.append(correct_matches)
    return cells[0], cells[1], matches[0], matches[1]


def paired_difference(
    length: int,
    eval_seed: int,
    unsorted_matches: Sequence[bool],
    sorted_matches: Sequence[bool],
    em_unsorted: float,
    em_sorted: float,
) -> PairedDifference:
    n = len(unsorted_matches)
    both_correct = sum(1 for u, s in zip(unsorted_matches, sorted_matches, strict=True) if u and s)
    both_wrong = sum(
        1 for u, s in zip(unsorted_matches, sorted_matches, strict=True) if not u and not s
    )
    only_u = sum(1 for u, s in zip(unsorted_matches, sorted_matches, strict=True) if u and not s)
    only_s = sum(1 for u, s in zip(unsorted_matches, sorted_matches, strict=True) if not u and s)
    return PairedDifference(
        length=length,
        eval_seed=eval_seed,
        n=n,
        both_correct=both_correct,
        both_wrong=both_wrong,
        only_unsorted_correct=only_u,
        only_sorted_correct=only_s,
        em_unsorted=em_unsorted,
        em_sorted=em_sorted,
        paired_em_diff=em_sorted - em_unsorted,
    )


# =============================================================================
# 4. Pooling and the pre-fixed length x order factorial decision rule.
# =============================================================================


@dataclass(frozen=True)
class FactorialCell:
    length_group: str
    order: str
    n_cells: int
    n_total: int
    em: float


def _pool_em(cells: Sequence[OrderCellResult]) -> FactorialCell:
    n_total = sum(c.n for c in cells)
    successes_total = sum(c.correct_successes for c in cells)
    return FactorialCell(
        length_group=cells[0].length_group,
        order=cells[0].order,
        n_cells=len(cells),
        n_total=n_total,
        em=successes_total / n_total,
    )


@dataclass(frozen=True)
class FactorialClassification:
    cells: dict[str, FactorialCell]
    length_margin: float
    order_margin: float
    interaction_margin: float
    length_main_effect: str
    order_main_effect: str
    interaction_effect: str
    worse_order: str
    overall: str


@dataclass(frozen=True)
class SeedPrimitiveFactorial:
    seed: int
    primitive: str
    cells: dict[str, FactorialCell]
    length_margin: float
    order_margin: float
    interaction_margin: float
    length_main_effect: str
    order_main_effect: str
    interaction_effect: str
    worse_order: str
    overall: str


def classify_seed_primitive(cells: dict[str, FactorialCell]) -> FactorialClassification:
    """Apply the pre-fixed length x order decision rule to four pooled cells.

    ``cells`` must have exactly the keys "SHORT:UNSORTED", "SHORT:SORTED",
    "LONG:UNSORTED", "LONG:SORTED".
    """
    em = {key: cells[key].em for key in cells}
    short_mean = (em["SHORT:UNSORTED"] + em["SHORT:SORTED"]) / 2
    long_mean = (em["LONG:UNSORTED"] + em["LONG:SORTED"]) / 2
    sorted_mean = (em["SHORT:SORTED"] + em["LONG:SORTED"]) / 2
    unsorted_mean = (em["SHORT:UNSORTED"] + em["LONG:UNSORTED"]) / 2
    length_margin = long_mean - short_mean
    order_margin = sorted_mean - unsorted_mean
    interaction_margin = (em["SHORT:SORTED"] - em["SHORT:UNSORTED"]) - (
        em["LONG:SORTED"] - em["LONG:UNSORTED"]
    )

    length_effect = (
        "PRESENT" if length_margin >= EFFECT_MARGIN and short_mean < ADEQUACY_FLOOR else "ABSENT"
    )
    order_effect = (
        "PRESENT"
        if abs(order_margin) >= EFFECT_MARGIN and min(sorted_mean, unsorted_mean) < ADEQUACY_FLOOR
        else "ABSENT"
    )
    interaction_effect = "PRESENT" if abs(interaction_margin) >= EFFECT_MARGIN else "ABSENT"
    worse_order = "SORTED" if order_margin < 0 else "UNSORTED"

    present = [
        name
        for name, effect in (
            ("LENGTH", length_effect),
            ("ORDER", order_effect),
            ("INTERACTION", interaction_effect),
        )
        if effect == "PRESENT"
    ]
    if not present:
        overall = (
            "NO_DEFECT_DETECTED"
            if all(v >= ADEQUACY_FLOOR for v in em.values())
            else "UNDIFFERENTIATED_DEFICIT"
        )
    else:
        overall = "EXPLAINED_BY_" + "+".join(present)

    return FactorialClassification(
        cells=cells,
        length_margin=length_margin,
        order_margin=order_margin,
        interaction_margin=interaction_margin,
        length_main_effect=length_effect,
        order_main_effect=order_effect,
        interaction_effect=interaction_effect,
        worse_order=worse_order,
        overall=overall,
    )


def classify_reproducibility(per_seed_effects: Sequence[str]) -> str:
    unique = set(per_seed_effects)
    if unique == {"PRESENT"}:
        return "REPRODUCIBLE_PRESENT"
    if unique == {"ABSENT"}:
        return "REPRODUCIBLE_ABSENT"
    n_present = sum(1 for effect in per_seed_effects if effect == "PRESENT")
    return f"NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS ({n_present}/{len(per_seed_effects)} PRESENT)"


# =============================================================================
# 5. Top-level run.
# =============================================================================


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    return value


def run(device_str: str | None = None) -> dict[str, Any]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = REPO_ROOT / OUTPUT_ROOT
    if output_root.exists():
        raise D015DiagnosticError(
            f"output namespace already exists and cannot be reused: {output_root}"
        )
    output_root.mkdir(parents=True)

    start = time.perf_counter()
    report: dict[str, Any] = {
        "task": TASK_ID,
        "model_seeds": list(MODEL_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "target_primitives": list(TARGET_PRIMITIVES),
        "negative_control": NEGATIVE_CONTROL,
        "short_lengths": list(SHORT_LENGTHS),
        "long_lengths": list(LONG_LENGTHS),
        "n_examples_per_cell": N_EXAMPLES,
        "adequacy_floor": ADEQUACY_FLOOR,
        "effect_margin": EFFECT_MARGIN,
        "sealed_access": 0,
        "optimizer_constructed": False,
        "parameter_updates": 0,
        "candidate_selected": None,
        "bundle_promotion": "NOT_AUTHORIZED",
        "non_sort_hash_gate": {},
        "order_cells": {},
        "paired_differences": {},
        "seed_primitive_factorial": {},
    }
    try:
        for seed in MODEL_SEEDS:
            gate = verify_non_sort_hash_identity(seed)
            report["non_sort_hash_gate"][str(seed)] = gate
            core, bank, op_to_id, _manifest = load_bundle_for_condition(
                seed, "LOCAL_SORT_REPAIR", device
            )
            for op_name in TARGET_PRIMITIVES:
                op_def = get_operation(op_name)
                valid_lengths = [
                    length
                    for length in (*SHORT_LENGTHS, *LONG_LENGTHS)
                    if op_def.is_valid_for_length(length)
                ]
                group_cells: dict[str, list[OrderCellResult]] = {
                    "SHORT:UNSORTED": [],
                    "SHORT:SORTED": [],
                    "LONG:UNSORTED": [],
                    "LONG:SORTED": [],
                }
                for eval_seed in EVAL_SEEDS:
                    for length in valid_lengths:
                        u_cell, s_cell, u_matches, s_matches = evaluate_order_cell(
                            core, bank, op_to_id, op_name, length, eval_seed
                        )
                        key = f"{seed}:{op_name}:{eval_seed}:{length}"
                        report["order_cells"][f"{key}:UNSORTED"] = u_cell
                        report["order_cells"][f"{key}:SORTED"] = s_cell
                        report["paired_differences"][key] = paired_difference(
                            length,
                            eval_seed,
                            u_matches,
                            s_matches,
                            u_cell.correct_em,
                            s_cell.correct_em,
                        )
                        group_cells[f"{u_cell.length_group}:UNSORTED"].append(u_cell)
                        group_cells[f"{s_cell.length_group}:SORTED"].append(s_cell)
                factorial_cells = {gkey: _pool_em(cells) for gkey, cells in group_cells.items()}
                classified = classify_seed_primitive(factorial_cells)
                result = SeedPrimitiveFactorial(
                    seed=seed,
                    primitive=op_name,
                    cells=classified.cells,
                    length_margin=classified.length_margin,
                    order_margin=classified.order_margin,
                    interaction_margin=classified.interaction_margin,
                    length_main_effect=classified.length_main_effect,
                    order_main_effect=classified.order_main_effect,
                    interaction_effect=classified.interaction_effect,
                    worse_order=classified.worse_order,
                    overall=classified.overall,
                )
                report["seed_primitive_factorial"][f"{seed}:{op_name}"] = result
                cell_file = output_root / f"cell_{seed}_{op_name}.json"
                cell_file.write_text(
                    json.dumps(_to_jsonable(result), indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
        reproducibility: dict[str, Any] = {}
        for op_name in TARGET_PRIMITIVES:
            per_seed = [
                report["seed_primitive_factorial"][f"{seed}:{op_name}"] for seed in MODEL_SEEDS
            ]
            reproducibility[op_name] = {
                "length_main_effect": classify_reproducibility(
                    [r.length_main_effect for r in per_seed]
                ),
                "order_main_effect": classify_reproducibility(
                    [r.order_main_effect for r in per_seed]
                ),
                "interaction_effect": classify_reproducibility(
                    [r.interaction_effect for r in per_seed]
                ),
                "per_seed_overall": {
                    str(seed): report["seed_primitive_factorial"][f"{seed}:{op_name}"].overall
                    for seed in MODEL_SEEDS
                },
            }
        report["reproducibility"] = reproducibility
        report["result"] = "COMPLETED"
    finally:
        report["wall_clock_seconds"] = time.perf_counter() - start
        if torch.cuda.is_available():
            report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        (output_root / "report.json").write_text(
            json.dumps(_to_jsonable(report), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return report


if __name__ == "__main__":
    final_report = run()
    summary = {
        "task": final_report["task"],
        "result": final_report["result"],
        "reproducibility": _to_jsonable(final_report["reproducibility"]),
    }
    print(json.dumps(summary, indent=2))
