# Phase D / Task D-001 — Target, Regression, and Causal-Control Panel Manifest

**Document ID:** `DOC-PHASE-D-D001-PANEL-MANIFEST`
**Date:** 2026-09-13
**Status:** Panel definitions fixed by re-aggregation of existing NRQ-006/NRQ-007/NRQ-008 artifacts. No model forward pass, no new training, no new data generation was executed to produce this document.
**Machine-readable companion:** `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json`

---

## 0. Purpose

This document fixes, before any repair execution, the exact class membership of:
1. the **target panel** (7 classes) the SORT-only repair pilot is registered against,
2. the **regression panels** (standalone-length, composition, and non-SORT canary) the pilot must not degrade,
3. the **causal-control panel** (Correct / Wrong-family / None) used to certify that any observed improvement is attributable to SORT actually executing the sort operation, not a coincidental or degenerate output.

All membership is taken verbatim from existing artifacts. No class was added, dropped, or re-labeled to make the panel look more tractable.

## 1. Cross-check of the 60-class / 1200-cell base and the 35 vs. 41 failure counts

Per the task's instruction, the following base numbers were cross-checked against existing artifacts, not re-derived from a fresh run:

| Quantity | Value | Source |
|---|---:|---|
| Total canonical equivalence classes | 60 | `docs/research/NRQ006_REVIEW_RECORD.json: selected_canonical_classes` (len 60) |
| Total cells | 1200 | 60 classes x 4 bundle seeds (1,2,3,4) x 5 data seeds (101-105); `NRQ006_REVIEW_RECORD.json: records` (len 1200) |
| Mean-threshold failure classes | 35 | `NRQ006_REVIEW_RECORD.json: failure_classes` (len 35); confirmed by `NRQ007_REVIEW_RECORD.json: failure_classes_count = 35` |
| All-cell-gate failure classes | 41 | `docs/research/STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md` section 1.1 ("41 classes fail the all-cell gate"); independently recomputed directly from the 1200 raw per-cell records (`oracle_em >= 0.85 and exhaustive_functional_em >= 0.95` per cell, all 20 cells of a class must pass) |
| All-cell-gate passing classes | 19 | 60 - 41 |

### 1.1 The 6-class difference (41 - 35)

35 classes fail the **mean**-level floor (class-average Oracle EM/Exhaustive EM). 41 classes fail the **strict all-cell** gate (any one of the 20 cells below floor). The difference is exactly **6 classes** that pass at the mean level but fail at least one of their 20 individual cells:

| Class | Mean Oracle EM | Mean Exhaustive EM | Failing cells (of 20) | Failing bundle seed(s) |
|---|---:|---:|---:|---|
| `NEGATE->REVERSE->SHIFT` | 0.973 | 0.971 | 5 | 4 |
| `NEGATE->SHIFT->SHIFT` | 0.955 | 0.953 | 7 | 2, 4 |
| `REVERSE->SHIFT->SELECT` | 0.984 | 0.984 | 3 | 2, 4 |
| `SHIFT->SHIFT->BIND` | 0.985 | 0.985 | 1 | 4 |
| `SHIFT->SHIFT->SELECT` | 0.971 | 0.971 | 3 | 4 |
| `SORT->SHIFT->SHIFT` | 0.988 | 0.988 | 2 | 4 |

**Derivation method:** direct re-aggregation of the 1200 raw per-cell records already stored in `NRQ006_REVIEW_RECORD.json["records"]`. For each of the 60 classes, `all_cell_pass = all(cell.oracle_em >= 0.85 and cell.exhaustive_functional_em >= 0.95 for cell in the 20 cells)`. The diff set is `{classes failing all_cell_pass} - {the 35 classes in NRQ006_REVIEW_RECORD.json["failure_classes"]}`. This is arithmetic over already-recorded numbers, not a new experiment.

**Formal attribution status: `UNCLASSIFIED_BY_NRQ007`.** NRQ-007's stepwise causal attribution (the 3-arm continuous / diagnostic-reset / standalone intervention, section 2.1 of `STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md`) was executed only over "all 35 failure recipes." These 6 classes pass the mean-level floor and were therefore never inputs to that procedure — they carry no `final_attribution` label anywhere in `NRQ007_REVIEW_RECORD.json`. This task does not run a new causal attribution to fill that gap; per the task's own instruction, the gap is recorded, not resolved.

**Descriptive observation (not a new causal claim):** all 6 classes are length-adequate (`is_length_adequate: true` for every cell) and every class contains at least one `SHIFT` step. In 5 of 6 classes, every failing cell is on `bundle_seed=4`; the sixth (`NEGATE->SHIFT->SHIFT`) has 2 of 7 failing cells on `bundle_seed=2` and the rest on `bundle_seed=4`. This pattern is consistent with — but not a re-derivation of — two already-recorded findings: NRQ-007 section 3.2's `BUNDLE_SPECIFIC_COMPONENT_FAILURE` "Chained Shifts" subset ("Bundle 4 exhibits slight degradation in `SHIFT` accuracy... causing 3-step chained shifts to drop below the 0.95 threshold on Bundle 4"), and NRQ-008/ADR-0168's independently measured Bundle 4 challenge-cell Oracle EM of 0.9147 on an unrelated cell (`REPLICATION_AND_SUPPORT_BUDGET_SENSITIVITY_NRQ008.md`). Both independently document a weaker Bundle-4 substrate. This is offered here only as a plausible, pre-existing explanation for why the 6-class diff exists; it is not adopted as a formal attribution, and no repair for it is authorized or proposed by this task.

**Importance for Task D-001:** none of the 7 target-panel classes (below) are among this 6-class diff. The two sets are disjoint and must not be conflated.

## 2. Target panel — `SHORT_SEQUENCE_CAPACITY_DEFICIT`, SORT immediately after SELECT (7 classes)

Membership taken verbatim from `NRQ007_REVIEW_RECORD.json: class_attributions` filtered on `final_attribution == "SHORT_SEQUENCE_CAPACITY_DEFICIT"` (also listed in `STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md` section 3.2.1). All 7 classes are **100% reproducible** — all 20/20 cells (4 bundle seeds x 5 data seeds) receive the identical attribution, the strongest reproducibility grade NRQ-007 records.

| Class | SORT step index | SORT input length set | Continuous step EM at SORT | Standalone step EM at SORT |
|---|:-:|---|---:|---:|
| `NEGATE->SELECT->SORT` | 3 | {3,4,5} | 0.078 | 0.075 |
| `SELECT->SORT->BIND` | 2 | {3,4,5} | 0.049 | 0.075 |
| `SELECT->SORT->NEGATE` | 2 | {3,4,5} | 0.082 | 0.075 |
| `SELECT->SORT->REVERSE` | 2 | {3,4,5} | 0.063 | 0.075 |
| `SELECT->SORT->SELECT` | 2 | {3,4,5} | 0.066 | 0.075 |
| `SELECT->SORT->SHIFT` | 2 | {3,4,5} | 0.086 | 0.075 |
| `SHIFT->SELECT->SORT` | 3 | {3,4,5} | 0.070 | 0.075 |

**This panel is historical development knowledge, not a new unseen relation.** These 7 classes were the object of NRQ-006's diagnostic discovery and NRQ-007's causal attribution; they are named here for repair pre-registration, not presented as a novel-relation transfer benchmark. Per the task's framing, this pilot does not evaluate unknown-relation transfer, and any later claim about generalization from single-primitive training to composition must be scoped as "no end-to-end training over composed sequences was performed."

**Derivation of the SORT input length set `{3,4,5}` (not assumed):** `SelectOp.output_length(L) = max(1, L // 2)` (`src/apc/environments/operations.py`), applied to the training-exposure length range `L in {6,7,8,9,10}` (`UnifiedBenchmarkConfig.sequence_length_range` default `(6,10)`), yields `{3,3,4,4,5} = {3,4,5}`.

## 3. Regression panels

### 3.1 Standalone length-adequate regression (`L in {6,7,8,9,10}`)
SORT executed standalone on its original training-exposure length domain, i.i.d. sampled exactly as the historical generator does. Pre-repair reference: the historical acceptance criterion `correct_exact_match >= 0.95` (`unified_oracle_causal_benchmark.py:905`) and the length-adequate composition evidence below (mean Oracle EM >= 0.988 wherever SORT sits at a length-adequate position).

### 3.2 Composition regression — SORT at a length-adequate position (11 classes)
Depth-3 classes containing SORT at a position where its input length is in `{6,...,10}` (i.e. **not** immediately after `SELECT` at step 1/2), currently passing the strict all-cell gate (20/20):

`NEGATE->SORT->BIND`, `NEGATE->SORT->SELECT`, `NEGATE->SORT->SHIFT`, `SORT->NEGATE->SELECT`, `SORT->NEGATE->SHIFT`, `SORT->REVERSE->BIND`, `SORT->REVERSE->SELECT`, `SORT->REVERSE->SHIFT`, `SORT->SELECT->BIND`, `SORT->SHIFT->BIND`, `SORT->SHIFT->SELECT`.

These verify that repairing SORT for `{3,4,5}` does not regress SORT's already-adequate behavior at length-adequate composition positions.

### 3.3 Non-SORT capability canary (8 classes)
Classes containing **no** SORT step at all, currently passing the strict all-cell gate (20/20):

`NEGATE->REVERSE->BIND`, `NEGATE->REVERSE->SELECT`, `NEGATE->SELECT->BIND`, `NEGATE->SHIFT->BIND`, `NEGATE->SHIFT->SELECT`, `REVERSE->SELECT->BIND`, `REVERSE->SHIFT->BIND`, `SHIFT->SELECT->BIND`.

These verify Core / other-primitive / router / argument-scorer invariance: since the repair recipe updates only SORT's own primitive weights (`docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` section 7), none of these 8 classes' pre-repair scores should move at all. Any movement here is a direct, behavioral signal of an unintended non-target parameter change.

## 4. Causal-control panel — Correct / Wrong-family / None

SORT is parameter-free (`required_argument_names = frozenset()`), so the applicable triple is **Correct / Wrong-family / None** — there is no Wrong-argument arm for SORT, matching the task's own wording for this primitive.

This reuses the existing parameter-free causal-control procedure at `src/apc/evaluation/unified_oracle_causal_benchmark.py:883-908` verbatim (not reimplemented):

| Arm | Definition | What it computes for SORT |
|---|---|---|
| **Correct** | SORT's own primitive, unconditioned forward | Should match `SortOp.apply` (`tuple(sorted(sequence))`) |
| **Wrong-family** | `WRONG_FAMILY_MAP["SORT"] = "NEGATE"` — NEGATE's own primitive evaluated on the same input, scored against SORT's correct target | Should match `NegateOp.apply` (`vocab_size - 1 - token`), which is a different, semantically distinguishable transform |
| **None** | SORT's own primitive with `enabled=False` → all-zero logits → argmax predicts token 0 at every position (`src/apc/primitives/primitive.py`) | Degenerate constant-zero output |

Existing acceptance thresholds (unchanged, reused as-is): `correct_exact_match >= 0.95`; `causal_gap = correct - max(wrong_family, none) >= 0.50`; `none_exact_match <= natural_baseline["SORT"] + 0.05 = 0.06`.

**Coincidence exclusion rules (registered before execution, not chosen post hoc):**
- Exclude any example where `sorted(seq) == seq` (already sorted) from None-arm causal-sensitivity scoring, since the all-zero None output can only coincide with the correct output for the all-zero sequence.
- Exclude any example where `sorted(seq) == negate(seq)` from Wrong-family-arm causal-sensitivity scoring.

**Extension beyond the existing benchmark:** the existing causal-control evaluation only exercises `L in {6,...,10}`. This pilot registers the identical procedure additionally at `L in {3,4,5}` (the target panel's length set) — an extension of scope, not a new methodology.

## 5. Primary artifacts

- This document: `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`
- Machine-readable manifest: `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json`
- Composition execution contract: `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md`
- SORT-only repair pilot preregistration: `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`
- Upstream sources: `docs/research/NRQ006_REVIEW_RECORD.json`, `docs/research/NRQ007_REVIEW_RECORD.json`, `docs/research/ARGUMENT_CLOSED_DEPTH3_CLOSURE_AUDIT_NRQ006.md`, `docs/research/STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md`, `docs/research/REPLICATION_AND_SUPPORT_BUDGET_SENSITIVITY_NRQ008.md`
