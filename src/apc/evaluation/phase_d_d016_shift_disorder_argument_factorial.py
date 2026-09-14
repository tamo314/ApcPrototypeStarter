"""Task D-016 -- SHIFT disorder x argument paired factorial diagnostic (long sequences).

Read-only diagnostic over D-013's already-saved bundles (seeds 40-44) and
D-013's own registered evaluation seeds (301-305). No training, no optimizer,
no new model/data seed, no sealed access, no candidate selection, no
promotion, and no modification of any D-013/D-014/D-015 artifact.

Motivation
----------
ADR-0182 (D-014) reported one out-of-scope finding while localizing the
``DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT``: for seed 44 only, the
``SHIFT->SELECT->SORT`` class's own step-0 ``SHIFT`` (on raw, initial-length
{6..10} content) has continuous EM 0.844, a hair below the NRQ-007
non-terminal floor (0.85), classified ``UNCLASSIFIED_BOUNDARY`` because SHIFT
is neither SELECT, SORT, nor the final step. ADR-0183 (D-015) later measured
SHIFT as one of its four downstream-defect target primitives and found, as a
side effect at seed 44 specifically, ``EM(LONG,UNSORTED)=0.8368`` against
``EM(LONG,SORTED)=0.95044`` -- an order margin of +0.0839, just under D-015's
own 0.10 ``EFFECT_MARGIN`` and therefore not classified ``ORDER_MAIN_EFFECT``
there, but suggestive that SHIFT's weakness at long lengths may not be
diffuse. Two further questions were left open by both prior diagnostics:

  1. D-015's order factor was a coarse binary (UNSORTED vs. SORTED); it
     cannot show whether SHIFT's sensitivity is a threshold effect, a smooth
     gradient across degrees of disorder, or absent once disorder is
     measured on a finer scale.
  2. Neither D-014 nor D-015 varied or balanced SHIFT's own argument (the
     shift amount) as a controlled factor -- both used amounts drawn
     independently at random per example, confounding any argument-specific
     failure with the disorder/order effect already measured.

This diagnostic isolates disorder from argument with a paired factorial at
the two boundary lengths of D-013's own initial-length range ({6, 10}):

  - Disorder factor (5 ordinal levels): controlled permutations of the same
    per-item token multiset (``length`` *distinct* tokens drawn without
    replacement from the shared vocabulary), constructed to have an exact
    number of inversions relative to ascending order:
    ``ASCENDING`` (0 inversions), ``LOW_INVERSION``
    (``round(0.25 * max_inversions)``), ``MEDIUM_INVERSION``
    (``round(0.5 * max_inversions)``), ``HIGH_INVERSION``
    (``round(0.75 * max_inversions)``), ``DESCENDING`` (``max_inversions =
    length * (length - 1) / 2``, i.e. full reverse-sorted order).
  - Argument factor: every SHIFT amount valid for the length under test
    (``0 .. length - 1``, matching ``ShiftOp.sample_params``'s own
    ``rng.randrange(len(sequence))`` range) is balanced -- evaluated with
    an equal, fixed sample size per amount, rather than sampled at random
    per example as in D-014/D-015.
  - **Full pairing across both factors, removing the amount/content
    confound**: the same drawn multiset (and its 5 disorder-level
    permutations) is generated once per (length, evaluation seed) by
    ``_paired_disorder_examples`` -- which takes no ``amount`` argument at
    all -- and reused, unchanged, across every valid amount for that length.
    Content therefore cannot vary with amount by construction: any observed
    argument-amount difference is attributable to the amount itself, never
    to incidental multiset-composition drift between amounts, and likewise
    any observed disorder difference is attributable to order alone
    (mirrors and generalizes D-015's unsorted/sorted pairing, from 2 levels
    x unpaired amount to a fully paired 5-level x N-amount grid).

Gate and parity preconditions (all enforced per model seed before any new
computation is trusted)
------------------------------------------------------------------------
1. **Stable-Core/SHIFT hash identity, exact D-015 parity**: re-run D-015's
   own ``verify_non_sort_hash_identity`` (unchanged, imported directly) and
   require the recomputed gate dict to equal, key-for-key, the gate already
   recorded in D-015's saved ``report.json`` for this seed. Since that gate
   proves the Stable Core and every non-SORT primitive (including SHIFT) are
   bit-identical between the saved ``FROZEN_PARENT`` and ``LOCAL_SORT_REPAIR``
   manifests, only ``LOCAL_SORT_REPAIR`` is loaded for the new disorder x
   argument evaluation below (matching D-015's own justification: evaluating
   the other condition too would be a fully redundant repeat).
2. **Exact D-014 parity**: recompute D-014's own step-0 (``SHIFT``) cell for
   the ``SHIFT->SELECT->SORT`` class, at D-013's own eval seed 301, for both
   ``FROZEN_PARENT`` and ``LOCAL_SORT_REPAIR`` (reusing D-014's own
   ``evaluate_cell`` verbatim, which itself re-verifies exact D-013 parity as
   a precondition), and require the recomputed step-0 record to equal,
   field-for-field, D-014's already-saved cell JSON. This directly
   re-confirms the seed-44 anomaly (and the other 4 seeds' clean results)
   from source before treating it as the diagnostic's starting point.

A mismatch at either gate raises ``D016DiagnosticError`` and halts before any
disorder x argument computation, matching D-014's own fail-closed convention.

Decision rules (fixed here, before any run; NOT tuned to observed results)
---------------------------------------------------------------------------
For each (model seed, length), pool ``standalone_arms``'s Correct-arm exact
match rate across all 5 registered evaluation seeds (301-305), giving one
pooled cell mean per (disorder level, argument amount) pair::

    disorder_mean[d]  = pooled EM across all amounts and eval seeds at
                         disorder level d
    argument_mean[a]  = pooled EM across all disorder levels and eval seeds
                         at amount a
    grand_mean        = pooled EM across every (d, a) cell

    disorder_margin    = max_d(disorder_mean[d]) - min_d(disorder_mean[d])
    argument_margin    = max_a(argument_mean[a]) - min_a(argument_mean[a])
    interaction_margin = max_{d,a} | cell_em[d,a]
                                     - (disorder_mean[d] + argument_mean[a]
                                        - grand_mean) |

``ADEQUACY_FLOOR = 0.95`` and ``EFFECT_MARGIN = 0.10`` are reused verbatim
from D-014/D-015 (D-014's ``STANDALONE_DEFECT_FLOOR`` and D-015's own fixed
``EFFECT_MARGIN``, chosen there as twice the codebase's established None-arm
noise tolerance) rather than re-derived, so this diagnostic applies the same
preregistered bar as its two predecessors::

    DISORDER_MAIN_EFFECT  = PRESENT iff disorder_margin >= EFFECT_MARGIN
                                     and min_d(disorder_mean[d]) < ADEQUACY_FLOOR
                             else ABSENT
    ARGUMENT_MAIN_EFFECT  = PRESENT iff argument_margin >= EFFECT_MARGIN
                                     and min_a(argument_mean[a]) < ADEQUACY_FLOOR
                             else ABSENT
    INTERACTION_EFFECT    = PRESENT iff interaction_margin >= EFFECT_MARGIN
                             else ABSENT

Per (seed, length): if no effect is PRESENT and every (d, a) cell clears
``ADEQUACY_FLOOR``: ``NO_DEFECT_DETECTED``. If some cell is below floor but
none of the three named effects clears ``EFFECT_MARGIN`` (a uniform
deficiency not localized to disorder, argument, or their interaction --
i.e. diffuse seed-specific inadequacy): ``UNDIFFERENTIATED_DEFICIT``.
Otherwise: ``EXPLAINED_BY_<DISORDER|ARGUMENT|INTERACTION[+...]>``.

Each of the three effects is also classified for cross-seed reproducibility
at each length (D-015's own ``classify_reproducibility``, reused verbatim):
``REPRODUCIBLE_PRESENT``/``REPRODUCIBLE_ABSENT`` if all 5 model seeds agree,
else ``NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS`` with the per-seed split
reported alongside.
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

from apc.evaluation.phase_d_d014_stepwise_causal_localization import (
    CONDITIONS as D014_CONDITIONS,
)
from apc.evaluation.phase_d_d014_stepwise_causal_localization import (
    OUTPUT_ROOT as D014_OUTPUT_ROOT,
)
from apc.evaluation.phase_d_d014_stepwise_causal_localization import (
    VOCAB_SIZE,
    evaluate_cell,
    load_bundle_for_condition,
    standalone_arms,
)
from apc.evaluation.phase_d_d015_downstream_length_order_factorial import (
    OUTPUT_ROOT as D015_OUTPUT_ROOT,
)
from apc.evaluation.phase_d_d015_downstream_length_order_factorial import (
    classify_reproducibility,
    verify_non_sort_hash_identity,
)
from apc.evaluation.phase_d_executor import EVAL_SEEDS, MODEL_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[3]

TASK_ID = "D-016"
OUTPUT_ROOT = Path("runs/phase_d_d016_shift_disorder_argument_factorial")

TARGET_PRIMITIVE = "SHIFT"
D014_TARGET_CLASS = "SHIFT->SELECT->SORT"
D014_TARGET_STEP_IDX = 0  # SHIFT is step 0 in this recipe (idx_select=1, idx_sort=2)

LENGTHS: tuple[int, ...] = (6, 10)  # boundary lengths of D-013's own initial-length range
DISORDER_LEVELS: tuple[str, ...] = (
    "ASCENDING",
    "LOW_INVERSION",
    "MEDIUM_INVERSION",
    "HIGH_INVERSION",
    "DESCENDING",
)
DISORDER_FRACTIONS: dict[str, float] = {
    "ASCENDING": 0.0,
    "LOW_INVERSION": 0.25,
    "MEDIUM_INVERSION": 0.5,
    "HIGH_INVERSION": 0.75,
    "DESCENDING": 1.0,
}
N_EXAMPLES = 1_000  # matches D-013's own per-eval-seed target-panel sample size

ADEQUACY_FLOOR = 0.95  # reused verbatim from D-014's STANDALONE_DEFECT_FLOOR
EFFECT_MARGIN = 0.10  # reused verbatim from D-015's EFFECT_MARGIN


class D016DiagnosticError(RuntimeError):
    """A prerequisite artifact, gate, or parity check failed; do not fabricate results."""


# =============================================================================
# 1. Exact D-014/D-015 parity preconditions.
# =============================================================================


def _load_d015_expected_gate(seed: int) -> dict[str, Any]:
    report_path = REPO_ROOT / D015_OUTPUT_ROOT / "report.json"
    if not report_path.is_file():
        raise D016DiagnosticError(f"D-015 artifact missing, cannot proceed: {report_path}")
    record = json.loads(report_path.read_text(encoding="utf-8"))
    gate = record.get("non_sort_hash_gate", {}).get(str(seed))
    if gate is None:
        raise D016DiagnosticError(f"D-015 report.json missing non_sort_hash_gate for seed {seed}")
    return gate


def verify_d015_parity(seed: int) -> dict[str, Any]:
    """Re-run D-015's Stable-Core/SHIFT (+ all non-SORT ops) identity gate and
    require exact agreement with D-015's own saved result."""
    recomputed = verify_non_sort_hash_identity(seed)
    expected = _load_d015_expected_gate(seed)
    if recomputed != expected:
        raise D016DiagnosticError(
            f"D-015 parity check FAILED for seed {seed}: "
            f"recomputed={recomputed} expected={expected}"
        )
    return recomputed


def _load_d014_expected_step0(seed: int, condition: str) -> dict[str, Any]:
    cell_name = f"cell_{seed}_{condition}_{D014_TARGET_CLASS.replace('->', '_')}.json"
    cell_path = REPO_ROOT / D014_OUTPUT_ROOT / cell_name
    if not cell_path.is_file():
        raise D016DiagnosticError(f"D-014 artifact missing, cannot proceed: {cell_path}")
    record = json.loads(cell_path.read_text(encoding="utf-8"))
    return dict(record["steps"][D014_TARGET_STEP_IDX])


def verify_d014_parity(
    core: Any, bank: Any, op_to_id: dict[str, int], seed: int, condition: str
) -> dict[str, Any]:
    """Recompute D-014's own SHIFT->SELECT->SORT step-0 cell (which itself
    re-verifies exact D-013 parity) and require exact agreement with D-014's
    own saved cell JSON."""
    if condition not in D014_CONDITIONS:
        raise D016DiagnosticError(f"unknown condition: {condition}")
    cell = evaluate_cell(core, bank, op_to_id, seed, condition, D014_TARGET_CLASS)
    recomputed_step0 = cell.to_dict()["steps"][D014_TARGET_STEP_IDX]
    expected_step0 = _load_d014_expected_step0(seed, condition)
    if recomputed_step0 != expected_step0:
        raise D016DiagnosticError(
            f"D-014 parity check FAILED for {condition} seed {seed}: "
            f"recomputed={recomputed_step0} expected={expected_step0}"
        )
    return {"seed": seed, "condition": condition, "match": True, "step0": recomputed_step0}


# =============================================================================
# 2. Controlled-inversion permutation construction (pure logic).
# =============================================================================


def _count_inversions(seq: Sequence[int]) -> int:
    n = len(seq)
    return sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])


def _disorder_targets(length: int) -> dict[str, int]:
    max_inversions = length * (length - 1) // 2
    return {level: round(frac * max_inversions) for level, frac in DISORDER_FRACTIONS.items()}


def _permutation_with_inversions(
    rng: random.Random, elements_sorted_asc: Sequence[int], target_inversions: int
) -> tuple[int, ...]:
    """Construct a permutation of ``elements_sorted_asc`` with exactly
    ``target_inversions`` inversions relative to ascending order.

    Uses the standard inversion-table (Lehmer code) bijection: processing
    positions left to right, position ``i`` may contribute between 0 and
    ``n - 1 - i`` inversions (the number of not-yet-placed elements smaller
    than the one chosen for it); at each step a value is drawn uniformly
    from the range that still leaves the remaining target exactly
    achievable by the remaining positions' caps. ``target_inversions = 0``
    always yields ascending order; ``target_inversions = n*(n-1)/2`` (the
    maximum) always yields descending order.
    """
    n = len(elements_sorted_asc)
    max_inversions = n * (n - 1) // 2
    if not 0 <= target_inversions <= max_inversions:
        raise D016DiagnosticError(
            f"target_inversions {target_inversions} outside [0, {max_inversions}] for n={n}"
        )
    caps = [n - 1 - i for i in range(n)]
    suffix_max = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + caps[i]
    remaining = list(elements_sorted_asc)
    remaining_k = target_inversions
    perm: list[int] = []
    for i in range(n):
        achievable_after = suffix_max[i + 1]
        lo = max(0, remaining_k - achievable_after)
        hi = min(caps[i], remaining_k)
        choice = rng.randint(lo, hi)
        perm.append(remaining.pop(choice))
        remaining_k -= choice
    return tuple(perm)


def _paired_disorder_examples(
    length: int, eval_seed: int, n: int
) -> dict[str, list[tuple[int, ...]]]:
    """Generate ``n`` items, each rendered at all 5 disorder levels sharing
    the same drawn distinct-token multiset (paired across disorder levels;
    only token order differs).

    Deliberately takes no ``amount`` argument: content generation never
    depends on the SHIFT amount under test, so calling this once per
    (``length``, ``eval_seed``) and reusing its result across every valid
    amount pairs content across the full disorder x argument grid, not just
    across disorder levels. This removes the amount/content confound that a
    per-amount RNG seed would otherwise introduce -- with amount absent from
    both the signature and the seed derivation, it is structurally
    impossible for content to vary by amount regardless of caller order.
    """
    if length > VOCAB_SIZE:
        raise D016DiagnosticError(
            f"length {length} exceeds vocab size {VOCAB_SIZE} for distinct draws"
        )
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"phase-d-d016:{length}:{eval_seed}:{n}".encode()).digest()[:8],
            "big",
        )
    )
    targets = _disorder_targets(length)
    contents: dict[str, list[tuple[int, ...]]] = {level: [] for level in DISORDER_LEVELS}
    for _ in range(n):
        elements = sorted(rng.sample(range(VOCAB_SIZE), length))
        for level in DISORDER_LEVELS:
            contents[level].append(_permutation_with_inversions(rng, elements, targets[level]))
    return contents


# =============================================================================
# 3. Per (seed, length, amount, disorder, eval_seed) evaluation.
# =============================================================================


@dataclass(frozen=True)
class DisorderArgumentCell:
    seed: int
    length: int
    amount: int
    disorder: str
    eval_seed: int
    n: int
    correct_successes: int
    correct_em: float
    wrong_family_em: float
    none_em: float
    wrong_amount_em: float | None


def _weighted_em(cells: Sequence[DisorderArgumentCell]) -> tuple[int, int, float]:
    n_total = sum(c.n for c in cells)
    successes_total = sum(c.correct_successes for c in cells)
    return successes_total, n_total, successes_total / n_total


# =============================================================================
# 4. Pooling and the pre-fixed disorder x argument factorial decision rule.
# =============================================================================


@dataclass(frozen=True)
class SeedLengthFactorial:
    seed: int
    length: int
    amounts: list[int]
    cell_em: dict[str, float]
    disorder_mean: dict[str, float]
    argument_mean: dict[str, float]
    grand_mean: float
    disorder_margin: float
    argument_margin: float
    interaction_margin: float
    disorder_main_effect: str
    argument_main_effect: str
    interaction_effect: str
    worse_disorder: str
    worse_amount: int
    overall: str


def classify_seed_length(
    cells_by_disorder_amount: dict[tuple[str, int], list[DisorderArgumentCell]],
    amounts: Sequence[int],
) -> SeedLengthFactorial:
    """Apply the pre-fixed disorder x argument decision rule.

    ``cells_by_disorder_amount`` must have exactly one entry per (disorder
    level, amount) pair in ``DISORDER_LEVELS`` x ``amounts``.
    """
    first_cell = next(iter(cells_by_disorder_amount.values()))[0]
    seed, length = first_cell.seed, first_cell.length

    cell_em: dict[tuple[str, int], float] = {
        key: _weighted_em(cells)[2] for key, cells in cells_by_disorder_amount.items()
    }
    disorder_mean = {
        level: _weighted_em(
            [c for a in amounts for c in cells_by_disorder_amount[(level, a)]]
        )[2]
        for level in DISORDER_LEVELS
    }
    argument_mean = {
        a: _weighted_em(
            [c for level in DISORDER_LEVELS for c in cells_by_disorder_amount[(level, a)]]
        )[2]
        for a in amounts
    }
    grand_mean = _weighted_em([c for cells in cells_by_disorder_amount.values() for c in cells])[2]

    disorder_margin = max(disorder_mean.values()) - min(disorder_mean.values())
    argument_margin = max(argument_mean.values()) - min(argument_mean.values())
    interaction_margin = max(
        abs(cell_em[(level, a)] - (disorder_mean[level] + argument_mean[a] - grand_mean))
        for level in DISORDER_LEVELS
        for a in amounts
    )

    disorder_effect = (
        "PRESENT"
        if disorder_margin >= EFFECT_MARGIN and min(disorder_mean.values()) < ADEQUACY_FLOOR
        else "ABSENT"
    )
    argument_effect = (
        "PRESENT"
        if argument_margin >= EFFECT_MARGIN and min(argument_mean.values()) < ADEQUACY_FLOOR
        else "ABSENT"
    )
    interaction_effect = "PRESENT" if interaction_margin >= EFFECT_MARGIN else "ABSENT"

    worse_disorder = min(disorder_mean, key=lambda level: disorder_mean[level])
    worse_amount = min(argument_mean, key=lambda a: argument_mean[a])

    present = [
        name
        for name, effect in (
            ("DISORDER", disorder_effect),
            ("ARGUMENT", argument_effect),
            ("INTERACTION", interaction_effect),
        )
        if effect == "PRESENT"
    ]
    if not present:
        overall = (
            "NO_DEFECT_DETECTED"
            if all(v >= ADEQUACY_FLOOR for v in cell_em.values())
            else "UNDIFFERENTIATED_DEFICIT"
        )
    else:
        overall = "EXPLAINED_BY_" + "+".join(present)

    return SeedLengthFactorial(
        seed=seed,
        length=length,
        amounts=list(amounts),
        cell_em={f"{level}:{a}": cell_em[(level, a)] for level in DISORDER_LEVELS for a in amounts},
        disorder_mean=disorder_mean,
        argument_mean={str(a): v for a, v in argument_mean.items()},
        grand_mean=grand_mean,
        disorder_margin=disorder_margin,
        argument_margin=argument_margin,
        interaction_margin=interaction_margin,
        disorder_main_effect=disorder_effect,
        argument_main_effect=argument_effect,
        interaction_effect=interaction_effect,
        worse_disorder=worse_disorder,
        worse_amount=worse_amount,
        overall=overall,
    )


# =============================================================================
# 5. Top-level run.
# =============================================================================


def run(device_str: str | None = None) -> dict[str, Any]:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = REPO_ROOT / OUTPUT_ROOT
    if output_root.exists():
        raise D016DiagnosticError(
            f"output namespace already exists and cannot be reused: {output_root}"
        )
    output_root.mkdir(parents=True)

    start = time.perf_counter()
    report: dict[str, Any] = {
        "task": TASK_ID,
        "model_seeds": list(MODEL_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "target_primitive": TARGET_PRIMITIVE,
        "lengths": list(LENGTHS),
        "disorder_levels": list(DISORDER_LEVELS),
        "n_examples_per_cell": N_EXAMPLES,
        "adequacy_floor": ADEQUACY_FLOOR,
        "effect_margin": EFFECT_MARGIN,
        "sealed_access": 0,
        "optimizer_constructed": False,
        "parameter_updates": 0,
        "candidate_selected": None,
        "bundle_promotion": "NOT_AUTHORIZED",
        "d015_parity_gate": {},
        "d014_parity_check": {},
        "raw_cells": {},
        "seed_length_factorial": {},
    }
    try:
        for seed in MODEL_SEEDS:
            report["d015_parity_gate"][str(seed)] = verify_d015_parity(seed)

            frozen_core, frozen_bank, frozen_op_to_id, _ = load_bundle_for_condition(
                seed, "FROZEN_PARENT", device
            )
            report["d014_parity_check"][f"{seed}:FROZEN_PARENT"] = verify_d014_parity(
                frozen_core, frozen_bank, frozen_op_to_id, seed, "FROZEN_PARENT"
            )

            core, bank, op_to_id, _manifest = load_bundle_for_condition(
                seed, "LOCAL_SORT_REPAIR", device
            )
            report["d014_parity_check"][f"{seed}:LOCAL_SORT_REPAIR"] = verify_d014_parity(
                core, bank, op_to_id, seed, "LOCAL_SORT_REPAIR"
            )

            for length in LENGTHS:
                amounts = list(range(length))  # every valid SHIFT amount, balanced
                cells_by_disorder_amount: dict[tuple[str, int], list[DisorderArgumentCell]] = {
                    (level, amount): [] for level in DISORDER_LEVELS for amount in amounts
                }
                for eval_seed in EVAL_SEEDS:
                    # Generated once per (length, eval_seed) and reused across every
                    # amount below: pairs content across the full disorder x argument
                    # grid, not just across disorder levels for a fixed amount.
                    disorder_contents = _paired_disorder_examples(length, eval_seed, N_EXAMPLES)
                    for amount in amounts:
                        arg_dicts = [{"amount": amount}] * N_EXAMPLES
                        for level in DISORDER_LEVELS:
                            arm, correct_matches = standalone_arms(
                                core,
                                bank,
                                op_to_id,
                                TARGET_PRIMITIVE,
                                disorder_contents[level],
                                arg_dicts,
                                VOCAB_SIZE,
                            )
                            cell = DisorderArgumentCell(
                                seed=seed,
                                length=length,
                                amount=amount,
                                disorder=level,
                                eval_seed=eval_seed,
                                n=arm.n,
                                correct_successes=sum(correct_matches),
                                correct_em=arm.correct_em,
                                wrong_family_em=arm.wrong_family_em,
                                none_em=arm.none_em,
                                wrong_amount_em=arm.wrong_argument_em,
                            )
                            key = f"{seed}:{length}:{amount}:{level}:{eval_seed}"
                            report["raw_cells"][key] = asdict(cell)
                            cells_by_disorder_amount[(level, amount)].append(cell)
                result = classify_seed_length(cells_by_disorder_amount, amounts)
                report["seed_length_factorial"][f"{seed}:{length}"] = asdict(result)
                cell_file = output_root / f"cell_{seed}_{length}.json"
                cell_file.write_text(
                    json.dumps(asdict(result), indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
        reproducibility: dict[str, Any] = {}
        for length in LENGTHS:
            per_seed = [
                report["seed_length_factorial"][f"{seed}:{length}"] for seed in MODEL_SEEDS
            ]
            reproducibility[str(length)] = {
                "disorder_main_effect": classify_reproducibility(
                    [r["disorder_main_effect"] for r in per_seed]
                ),
                "argument_main_effect": classify_reproducibility(
                    [r["argument_main_effect"] for r in per_seed]
                ),
                "interaction_effect": classify_reproducibility(
                    [r["interaction_effect"] for r in per_seed]
                ),
                "per_seed_overall": {
                    str(seed): report["seed_length_factorial"][f"{seed}:{length}"]["overall"]
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
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    final_report = run()
    summary = {
        "task": final_report["task"],
        "result": final_report["result"],
        "reproducibility": final_report["reproducibility"],
    }
    print(json.dumps(summary, indent=2))
