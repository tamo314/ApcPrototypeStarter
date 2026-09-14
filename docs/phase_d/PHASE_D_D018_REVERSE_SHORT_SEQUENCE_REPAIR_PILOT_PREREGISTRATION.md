# Phase D / Task D-018 — REVERSE Short-Sequence Repair Causal-Transfer Pilot Preregistration

**Document ID:** `DOC-PHASE-D-D018-REVERSE-REPAIR-PREREGISTRATION`
**Date:** 2026-09-15
**Status:** Design/preregistration. **No optimizer step, model initialization, candidate
construction, or sealed-data access was performed to produce this document.**
`training_execution` for this document's recipe is fixed by this task's own review section (§12)
and the corresponding ADR — see that section for the final authorization statement. No training or
model forward may occur before that authorization is recorded, per task instruction.

**Relationship to D-001/H-D1 (SORT):** this is a **new, independent hypothesis (H-D2)** under the
same Phase D charter (`docs/research/PHASE_D_RESEARCH_CHARTER.md`), opened because the charter's
own text reserves any primitive beyond SORT to "a separate charter task; it is not implied or
pre-authorized" by H-D1. H-D1's `REFUTED` verdict (D-013, ADR-0181) and D-014-D-017's diagnostics
are unmodified and are used here only as already-recorded evidence, not re-derived.

**Depends on:** `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` (input-domain
derivation, ownership boundary, dependency hashes — REVERSE's own row, section 2, is reused
unchanged), `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` (source of the verbatim
composition-class lists filtered in section 7 below), `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`
(cohort construction procedure, reused unchanged for a new seed set — section 3),
`docs/phase_d/PHASE_D_D018_SEED_REGISTRY.json` and
`docs/design-docs/PHASE_D_D018_SEED_REGISTRY_AUDIT_AND_COHORT_FIXATION.md` (new, unused,
non-sealed cohort fixation, mechanically audited).

---

## 1. Why REVERSE, and why this is a *causal-transfer* test, not a repeat of D-001

D-017's evidence-integrity audit (`docs/research/EVIDENCE_INTEGRITY_AUDIT_D017.md` section 4,
`D017_REVIEW_RECORD.json: next_charter_boundary_items[1]`) judged **PROCEED** specifically on the
"REVERSE/SELECT short-length (`{3,4,5}`) capacity deficit," the only finding in that entire evidence
chain reproduced across all 5 independently-built D-013 models, independent of each seed's own
SORT/BIND health. This task registers **REVERSE only** (not SELECT — a separate charter task per
the same boundary-item separation rule this charter already applies to SORT/other-primitives).

**REVERSE's own deficit is independently evidenced, not inferred from the chain's aggregate
failure.** `NRQ007_REVIEW_RECORD.json: class_attributions` (already-recorded, no new computation)
gives per-step arrays for `SELECT->SORT->REVERSE` (`[SELECT, SORT, REVERSE]`):

| Step | `mean_standalone_step_ems` | `mean_diagnostic_reset_step_ems` | `mean_continuous_step_ems` |
|---|---:|---:|---:|
| 1 SELECT | 1.000 | 0.999 | 0.999 |
| 2 SORT | 0.075 | 0.063 | 0.063 |
| 3 REVERSE | **0.050** | **0.043** | **0.010** |

REVERSE's own standalone score (0.050) and, critically, its **diagnostic-reset** score (0.043 — SORT's
*ground-truth* output fed directly to REVERSE, per NRQ-007's condition (B) / D-014's identical
mechanism, `src/apc/evaluation/phase_d_d014_stepwise_causal_localization.py`) are both far below
floor **even when SORT's own defect is removed from the measurement**. This is direct,
already-recorded evidence that REVERSE has its own short-length capacity deficit, independent of
SORT's — the same structural pattern D-001/NRQ-007 established for SORT, now shown for REVERSE by
the identical methodology. This task does not re-run that measurement; it cites it as the basis for
H-D2 and as the pre-repair reference point for criterion 2 (section 8).

**The causal-transfer question this task adds (not present in D-001):** D-001 trained SORT on the
*entire* valid-length set `{3,...,10}` uniformly and asked only "does training fix it." This task
asks the stronger question the task instruction requires: **is repairing the short-length region
specifically attributable to exposing the model to that region**, or would *any* additional
REVERSE training (even confined to the lengths it already handles) produce the same effect via
generic capacity/regularization changes? Section 5 registers a same-compute control
(`LONG_SEQUENCE_REVERSE_TRAINING_CONTROL`) built to falsify the naive "more training helps
regardless of content" explanation.

## 2. The two competing hypotheses this task fixes before any run

> **H-D2 (causal, exposure-specific):** starting from a frozen parent bundle (Stable Core and every
> primitive except REVERSE frozen), standalone training of REVERSE alone, restricted to
> `L ~ Uniform{3,4,5}`, for a fixed step budget, recovers REVERSE's standalone and
> diagnostic-reset-isolated execution at `{3,4,5}` without degrading REVERSE's existing
> length-adequate behavior or any other primitive's behavior — **and** an equal-compute control
> trained instead on `L ~ Uniform{6,...,10}` (REVERSE's pre-existing training distribution) does
> **not** produce the same short-length recovery.

> **H-D2-null (generic-capacity confound):** the equal-compute long-length control recovers
> short-length REVERSE execution as well as (or comparably to) the short-length-trained candidate,
> which would mean any apparent "repair" is not attributable to exposure matching the failure
> region and would refute the causal framing (independent of whether either candidate's raw
> accuracy clears the floor).

Both outcomes are informative and neither is rescued by seed-averaging (section 8, section 11).

## 3. Update scope and ownership (identical mechanism to D-001, different target)

| Component | Frozen during repair | Source |
|---|:-:|---|
| `SharedContentEncoder` (Core) | **Yes** | `core.model.eval()`, `param.requires_grad_(False)` (`learned_routing_benchmark.py:566-568`) |
| REVERSE's `ReverseRelativePrimitive` instance | **No — the only unfrozen module** | `PrimitiveBank.freeze(primitive_id)`/`unfreeze` (`bank.py:235,238`); disjoint `ModuleDict` slice, `model_bundle.py:294-306` |
| SELECT / COUNT / BIND / SHIFT / COPY / SORT / NEGATE | **Yes** | Same per-primitive freeze mechanism |
| Router, argument scorer | **Yes** | REVERSE is invoked directly via `op_to_id["REVERSE"]` / `CompositionRecipe`; `required_argument_names = frozenset()` (composition contract section 2) so the argument scorer is never consulted |
| Vocabulary / token schema | **Yes** | `vocabulary.schema_hash` unchanged |

Mechanical enforcement and verification (before/after `canonical_state_hash` on Core and every
non-REVERSE primitive) are identical to D-001 section 2; only the unfrozen primitive id changes
(`op_to_id["REVERSE"]` instead of `op_to_id["SORT"]`).

## 4. Architecture (unchanged, and materially different from SORT's)

REVERSE is **not** a `CrossPositionPrimitive` (SORT's class). It is
`ReverseRelativePrimitive` — "Primitive-scale (17,290 params) REVERSE operator with modular reverse
relative bias" (`src/apc/primitives/primitive.py:1385-1451`, docstring at :1386), built as:

```
ReverseRelativePrimitiveConfig(d_model=config.model["d_model"], d_operator=32, n_head=4,
                                d_operator_ff=64, vocab_size=10, max_sequence_length=32)
```
(`src/apc/evaluation/unified_oracle_causal_benchmark.py:444-456`, via
`bank.new_reverse_relative_primitive`, `src/apc/primitives/bank.py:182-189`). Unlike
`CrossPositionPrimitiveConfig`, this config has **no `arg_dim` field** — consistent with REVERSE
being parameter-free (`required_argument_names = frozenset()`, composition contract section 2).
This recipe changes none of these config fields; no new layer, bias term, or argument encoder is
added. The exact trainable parameter count is read from `num_parameters()` at execution time, not
fabricated here (documented order of magnitude: 17,290, per the class docstring).

## 5. Training recipe — three conditions, one shared mechanism, only the length-sampling bound differs

All three trained/untrained conditions below reuse `_train_single_primitive`
(`unified_oracle_causal_benchmark.py:493-599`) **verbatim** — the same function that already trains
REVERSE (and every other bank primitive) in ordinary cohort construction
(`learned_routing_benchmark.py:634-639` imports and calls it). REVERSE already goes through this
function's non-parameterized branch exactly like SORT (`isinstance(primitive, (...,
ReverseRelativePrimitive, ...))`, line 577-585). Nothing in this function is modified.

### 5.1 Optimizer, scheduler, loss, gradient clipping, batch size (inherited, fixed, identical to D-001)

| Field | Value | Source |
|---|---|---|
| Optimizer | `torch.optim.AdamW(reverse_primitive.parameters(), lr=0.0008, weight_decay=0.0001)` | `UnifiedBenchmarkConfig.operator_lr/operator_weight_decay` (`:164-165`), instantiated `:506-510` |
| Scheduler | `CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)` | `:511-513` |
| Loss | Per-token cross-entropy, `ignore_index=-100` | `:590-592`; `src/apc/core/data.py:36` |
| Gradient clipping | `clip_grad_norm_(..., 1.0)` | `operator_grad_clip=1.0` (`:166`), applied `:594-595` |
| Batch size | 32 examples/step | `:539` |
| Step budget | **6,000 steps/model/condition** (registered ceiling, section 9) | This document |

**Step-budget rationale:** REVERSE's own historical construction budget is 3,000 steps
(`bank_train_steps // 2`, the same parameter-free-primitive rule SORT uses,
`learned_routing_benchmark.py:630-639`; cohort contract section 7 lists REVERSE alongside
COPY/SORT/NEGATE at 3,000). This pilot registers 6,000 — the same ~2x-historical ceiling D-001 used
for SORT — for **both** trained conditions, making their compute identical by construction (same
step count x same batch size x same optimizer/schedule = same FLOPs), which is exactly what
"same-compute long-sequence control" (task instruction) requires. If 6,000 is insufficient for
either condition, that is a recipe **failure** for that condition, not grounds to raise the budget
mid-run (section 11).

### 5.2 Length-sampling rule — the *only* difference between conditions

At every training step, the 32 example lengths are drawn i.i.d. via
`step_rng.randint(lo, hi)` (`unified_oracle_causal_benchmark.py:540`, unmodified) with **only the
bounds `(lo, hi)` changed per condition**; tokens remain i.i.d. `Uniform{0,...,9}` in every case
(`:543`, unmodified):

| Condition | `(lo, hi)` passed to `step_rng.randint` | Meaning |
|---|:-:|---|
| `LOCAL_REVERSE_REPAIR_SHORT_ONLY` | `(3, 5)` | `L ~ Uniform{3,4,5}` — **exactly** the task-mandated short-length-only exposure; never samples `{6,...,10}` |
| `LONG_SEQUENCE_REVERSE_TRAINING_CONTROL` | `(6, 10)` | `L ~ Uniform{6,...,10}` — identical to REVERSE's pre-existing historical training distribution (section 5, composition execution contract); same steps, same compute, but **never samples the failing region** |

Neither condition uses a curriculum, mixed-length batches, or adaptive oversampling. Both are fixed
before any run and do not change based on interim accuracy. `LOCAL_REVERSE_REPAIR_SHORT_ONLY`
deliberately never re-exposes REVERSE to `{6,...,10}` during repair — existing-capability
preservation at those lengths (criterion 5, section 8) is therefore a genuine test of whether a
narrowly-targeted update leaves length-adequate behavior untouched, not an artifact of continued
exposure.

## 6. Comparison conditions (four, matching the task's explicit list)

| Condition | Definition | Update rule |
|---|---|---|
| `FROZEN_PARENT` | One of the 5 new-cohort models (section 6 of the audit doc), no update. | Zero optimizer steps. |
| `LOCAL_REVERSE_REPAIR_SHORT_ONLY` | Starts from the same parent bundle; executes section 5's short-only recipe, updating only REVERSE's own parameters (section 3). **The candidate repair.** | Section 5.1/5.2, `(lo,hi)=(3,5)`. |
| `LONG_SEQUENCE_REVERSE_TRAINING_CONTROL` | Starts from the same parent bundle; executes the identical recipe except `(lo,hi)=(6,10)`, updating only REVERSE's own parameters. **The causal-specificity control**, not a repair candidate for promotion. | Section 5.1/5.2, `(lo,hi)=(6,10)`. |
| `SYMBOLIC_REFERENCE` | The deterministic ground-truth function `ReverseOp.apply` (`tuple(reversed(sequence))`, `src/apc/environments/operations.py:330-333`). Used only to certify panel/label correctness and as the causal-control panel's ground truth — **never** a training signal, fallback executor, or learner input. | Pure function; never receives or emits a signal into either trained condition's forward/backward pass. |

Both trained conditions start from the **same** `FROZEN_PARENT` instance per model (not
independently re-initialized), so any difference between them is attributable only to the
length-sampling bound, not to a different starting point.

### 6.1 Information-boundary flags (identical structural guarantee to D-001 section 5.2)

`task_conditioned_core=false`, `oracle_coordinate_input=false` (REVERSE has no argument),
`ground_truth_intermediate_injection=false` for the **primary** success metric (the diagnostic-reset
measurement in section 8 criterion 2 is registered separately and explicitly labeled diagnostic,
never substituted for a primary metric, per composition execution contract section 8),
`symbolic_execution_fallback=false`, `non_target_parameter_updates=0` (verified by hash comparison,
section 3).

## 7. Panels — derived by re-filtering D-001's already-published, verbatim class lists

No new class-level analysis was run. Every class below is copied verbatim from
`PHASE_D_D001_TARGET_PANEL_MANIFEST.md` sections 2/3.2/3.3 (themselves taken verbatim from
NRQ-006/007), then partitioned by "contains a REVERSE step" instead of "contains a SORT step,"
since this task's target/frozen boundary is REVERSE, not SORT.

### 7.1 Target (1 class, task-mandated)

`SELECT->SORT->REVERSE` — REVERSE at step 3, input length `{3,4,5}` (composition execution contract
section 3, `SelectOp.output_length` then SORT's identity length). This is the exact class cited in
section 1's evidence table. **No other class is a target of this task.**

### 7.2 REVERSE composition regression (7 classes — REVERSE at a currently-passing, length-adequate position)

From D-001's 11-class SORT-length-adequate list, the 3 containing REVERSE:
`SORT->REVERSE->BIND`, `SORT->REVERSE->SELECT`, `SORT->REVERSE->SHIFT`.
From D-001's 8-class non-SORT canary list, the 4 containing REVERSE (already all-cell-gate passing,
hence REVERSE sits at a length-adequate position in each):
`NEGATE->REVERSE->BIND`, `NEGATE->REVERSE->SELECT`, `REVERSE->SELECT->BIND`, `REVERSE->SHIFT->BIND`.

These 7 verify that repairing REVERSE for `{3,4,5}` does not regress REVERSE's already-adequate
behavior at length-adequate composition positions.

### 7.3 Non-REVERSE canary (12 classes — zero REVERSE steps, currently all-cell-gate passing)

From D-001's 11-class list, the 8 without REVERSE: `NEGATE->SORT->BIND`, `NEGATE->SORT->SELECT`,
`NEGATE->SORT->SHIFT`, `SORT->NEGATE->SELECT`, `SORT->NEGATE->SHIFT`, `SORT->SELECT->BIND`,
`SORT->SHIFT->BIND`, `SORT->SHIFT->SELECT`.
From D-001's 8-class canary list, the 4 without REVERSE: `NEGATE->SELECT->BIND`,
`NEGATE->SHIFT->BIND`, `NEGATE->SHIFT->SELECT`, `SHIFT->SELECT->BIND`.

These verify Core/other-primitive/router/argument-scorer invariance behaviorally (the hash check,
section 3, is the primary, mechanically exact guarantee; this panel is a secondary, redundant
behavioral cross-check, identical in purpose to D-001 section 3.3).

### 7.4 Standalone regression

REVERSE executed standalone at `L in {6,...,10}`, i.i.d. sampled exactly as the historical
generator does (same regime as D-001 section 3.1, substituting REVERSE for SORT).

## 8. Acceptance criteria (numeric, fixed before execution, evaluated per model, no mean-rescue)

All criteria are evaluated **per model** for **every** registered run; a mean across the 5 models
never substitutes for a per-model or per-cell floor.

1. **Target recovery — standalone (primary, gating, `LOCAL_REVERSE_REPAIR_SHORT_ONLY` only):**
   sequence EM `>= 0.95` on standalone REVERSE at `L=3` (exhaustive, N=1,000), `L=4` (exhaustive,
   N=10,000), `L=5` (exhaustive, N=100,000).
2. **Target recovery — diagnostic-reset-isolated boundary (primary, gating,
   `LOCAL_REVERSE_REPAIR_SHORT_ONLY` only):** using the existing reset-immediately-after-SORT
   mechanism (`src/apc/evaluation/phase_d_d014_stepwise_causal_localization.py`, NRQ-007 condition
   (B), reused verbatim — ground-truth `SortOp.apply` output re-encoded and fed to REVERSE) on
   `SELECT->SORT->REVERSE`'s step-3 position at `L in {3,4,5}`: sequence EM `>= 0.95` (N=1,000/model,
   eval seeds 401-405, section 10). This isolates REVERSE's own in-context competence from frozen
   SORT's own independent, unrepaired defect (section 1's evidence table; pre-registered here
   because that confound is foreseeable and must not be discovered post hoc).
3. **Continuous full-chain (registered, reported, `LOCAL_REVERSE_REPAIR_SHORT_ONLY` and
   `LONG_SEQUENCE_REVERSE_TRAINING_CONTROL` both, non-gating):** `SELECT->SORT->REVERSE` under the
   normal continuous-execution path (condition A, frozen/unrepaired SORT's own predicted output).
   **Pre-registered expectation, stated before any run:** this is expected to remain low regardless
   of REVERSE's repair quality, because SORT is frozen throughout this task and SORT's own
   short-length defect is independently large (continuous step EM ~0.063, section 1) — a low score
   here must **not** be read as evidence against H-D2 on its own; interpretation is anchored to
   criteria 1-2. A high score here (both SORT's and REVERSE's errors happening to cancel or REVERSE
   becoming robust to SORT's specific error pattern) would be a bonus, not a required, finding.
4. **Causal-transfer contrast (primary claim, gating):** H-D2 is supported for a given model **only
   if** that model's `LOCAL_REVERSE_REPAIR_SHORT_ONLY` candidate passes criteria 1 AND 2, **and**
   that model's `LONG_SEQUENCE_REVERSE_TRAINING_CONTROL` candidate **fails** criterion 1 (standalone
   `{3,4,5}` EM `< 0.95` despite identical compute, section 5.1). If `LONG_SEQUENCE_REVERSE_TRAINING_CONTROL`
   *also* passes criterion 1, this specific model's result refutes the causal/exposure-specific
   framing of H-D2 (H-D2-null, section 2) even though generic repair may still have occurred — this
   is recorded as a distinct outcome, not folded into criterion 1's pass/fail.
5. **Existing-capability preservation (both trained conditions independently):** for every model, on
   every panel in section 7.2 (7 classes), 7.3 (12 classes), and 7.4 (standalone `{6,...,10}`):
   degradation `<= 1.0` percentage point vs. that model's own `FROZEN_PARENT`, **and** the panel's
   own pre-existing floor is still met.
6. **Causal control (Correct / Wrong-family / None):** REVERSE is parameter-free
   (`required_argument_names = frozenset()`), so — matching SORT's own panel design — the applicable
   triple is Correct/Wrong-family/None, no Wrong-argument arm. Reuses
   `unified_oracle_causal_benchmark.py:883-908` verbatim. `WRONG_FAMILY_MAP["REVERSE"] = "SORT"`
   (`:120`) — SORT's own primitive substituted, scored against REVERSE's correct target.
   `NATURAL_BASELINES["REVERSE"] = 0.01` (`:106`). Thresholds: `correct_exact_match >= 0.95`;
   `causal_gap >= 0.50`; `none_exact_match <= 0.06` (`0.01 + 0.05`). Evaluated at both length groups
   (`{3,4,5}` for `LOCAL_REVERSE_REPAIR_SHORT_ONLY`; `{6,...,10}` for both trained conditions, as a
   regression check). **Coincidence exclusions (fixed now, not post hoc):** exclude any example
   where `tuple(reversed(seq)) == seq` (palindromes) from the None-arm; exclude any example where
   `tuple(reversed(seq)) == tuple(sorted(seq))` from the Wrong-family arm.
7. **Invariance:** Core and all 7 non-REVERSE primitives' `canonical_state_hash` unchanged
   before/after, for **both** trained conditions independently (`Delta = 0`, exact match, section 3).
8. **Strict fresh-load parity:** a fresh process reload (`load_bundle(..., mode="diagnostic")`)
   reproduces every metric above to `Delta = 0`, for both trained conditions.
9. **Full reporting:** every one of the 5 models' full result set, for all 3 non-frozen-parent
   conditions counted separately, is recorded whether it passes or fails. A failing model or
   condition is reported as such, never omitted or averaged away.

Secondary (recorded, not gating): parameter count (`num_parameters()`), total training examples
consumed, wall-clock, peak VRAM/RAM, for each of the two trained conditions independently.

## 9. Resource budget (repair + evaluation; new-cohort construction budget is separate, section 6 of the audit doc)

| Item | Per model (both conditions) | 5 models |
|---|---:|---:|
| Repair optimizer steps (registered ceiling) | 6,000 x 2 conditions = 12,000 | 60,000 |
| Training examples consumed | 12,000 x 32 = 384,000 | 1,920,000 |
| Resident parameters touched (unfrozen) | REVERSE's own `ReverseRelativePrimitive` only (17,290, exact count re-read at execution time), independently for each of the two conditions | same, x5 |
| Active parameters during repair | = resident (nothing else receives a gradient) | same |
| Temporary/plastic parameters | 0 (direct update of an existing persistent bank slot in a staged namespace; not a plastic-workspace residual) | 0 |
| Wall-clock [ESTIMATE, not measured] | `<=` ~10 minutes (`<=` 50ms/step, same conservative basis as D-001 section 9, doubled for 2 conditions) | `<=` ~50 minutes |
| Peak GPU VRAM / RAM [ESTIMATE] | Well within the 16GB/64GB budgets; batch 32, `L<=10`, ~17k-parameter module | same |
| Evaluation forward passes | Standalone exhaustive (1,000+10,000+100,000) + standalone regression (2,000x5 lengths x3 eval seeds) + target class (1,000x5 eval seeds) + diagnostic-reset boundary (1,000x5 eval seeds) + regression panel (500x7x5) + canary panel (500x12x5) + causal control (1,000x3x2 length groups x5 eval seeds) ~= 189,500 per model per trained condition | ~= 1,895,000 total (both conditions) |

Wall-clock/VRAM/RAM are explicit planning-upper-bounds, not measurements; no timing run was executed
to produce this document. At execution time these are replaced with recorded measurements.

## 10. Evaluation seeds (new, not reused from D-001/D-013)

Eval seeds **401, 402, 403, 404, 405** are fixed now for this task's every sampled evaluation
(composition target/regression/canary panels, causal control, diagnostic-reset boundary; 3 of the 5
— 401-403 — for the standalone `{6,...,10}` regression regime, matching D-001's own 3-of-5
convention). These are a **new** block, deliberately not reusing D-001/D-013's `301-305`, to keep
this task's evaluation namespace independent rather than merely relying on the fact that eval seeds
are not subject to the model-seed collision audit. Manually verified (not by the mechanical
model-seed audit, which does not cover this value type) against every model-seed range in
`PHASE_D_D018_SEED_REGISTRY.json`'s audit output (0-4, 10-14, 15-19, 20-24, 30-34, 40-44, 50-54) and
every historical data-seed range (101-105, 201-220, 301-305): no collision.

## 11. Success and failure claim scope (fixed regardless of outcome)

**If the pilot passes every gating criterion (1, 2, 4-9) for a given model:** the only claim
authorized is **local, exposure-specific repair success for the registered `SELECT->SORT->REVERSE`
target and the `{3,4,5}` standalone/diagnostic-isolated domain**, on the specific 5-model cohort in
`PHASE_D_D018_SEED_REGISTRY.json`. This is explicitly **not**: a claim about the full
60-class/1200-cell registry; a claim about unknown-relation transfer (this panel is pre-existing
development knowledge); a reversal of any Phase B/C terminal status; a production/candidate-bundle
adoption decision (`bundle_promotion` remains `NOT_AUTHORIZED`); a claim that single-primitive
training generalizes to end-to-end composition training (only standalone training was performed);
or a claim about SELECT's own short-length deficit (out of scope, section 1).

**If any gating criterion fails for a model:** the failure is recorded as a failure of this exact
registered recipe on this cohort. It does not authorize additional steps beyond the 6,000/condition
ceiling, additional seeds, an alternative architecture, extending repair to SELECT or any other
primitive, or reusing the D-013 cohort as a substitute confirmatory cohort. A follow-up requires a
new, separately authorized task.

**G1/sealed status:** unchanged; `sealed_access: 0` throughout; this pilot does not touch sealed
data or evaluate unknown-relation transfer.

## 12. Authorization statement

This document is a preregistration produced together with its own approval review in the same
task, per explicit task instruction. See `docs/research/PHASE_D_RESEARCH_CHARTER.md` H-D2 section
and the corresponding ADR in `docs/DECISIONS_PHASE_D.md` for the recorded review outcome and the
authoritative `training_execution` status. No optimizer step, model initialization, candidate
construction, or sealed-data access was performed to produce this document or its review.
`bundle_promotion` remains `NOT_AUTHORIZED` and `sealed_access` remains `0` regardless of this
authorization or of the pilot's eventual outcome.
