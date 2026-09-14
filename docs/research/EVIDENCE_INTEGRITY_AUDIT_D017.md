# Phase D / Task D-017 — Saved-Evidence Integrity Audit and Next-Charter Transition Judgment

**Document ID:** `DOC-PHASE-D-D017-EVIDENCE-INTEGRITY-AUDIT`
**Date:** 2026-09-15
**Status:** Audit complete. **Read-only.** No training, no optimizer construction, no model forward
pass for any new result, no new bundle construction, no D-013 re-execution, no seed addition/
exchange/exclusion, no panel/threshold/budget change, no sealed access, no candidate selection or
promotion. Where an input population needed disclosure that the saved artifacts do not record
directly (D-016 input diversity, section 3.3), this audit used the exact registered, already-executed
generator function with the exact recorded `(length, eval_seed, n)` arguments to regenerate **inputs
only** — zero model access — per the task's explicit deterministic-regeneration allowance.
**Machine-readable companion:** `docs/research/D017_REVIEW_RECORD.json`.

Verdict legend used throughout: **CONFIRMED** (directly verified against saved artifacts or exact
code trace), **UNSUPPORTED_INTERPRETATION** (the underlying number is correct but a causal/
explanatory claim attached to it is not established by the evidence), **UNDETERMINED** (cannot be
resolved from saved artifacts without further, separately-authorized work).

This audit does not change D-013's `FAIL`/`REFUTED` verdict (ADR-0181), D-014/D-015/D-016's own
`DIAGNOSTIC_COMPLETED` status, Phase B/C's terminal states, or the G1 deficit. Where this audit
finds an interpretive claim in ADR-0182/0183/0184 unsupported, the ADRs themselves are **not**
edited or renumbered; the correction is recorded here and in new ADR-0185.

---

## 1. Evidence and provenance fixation

| Item | Value |
|---|---|
| Repository HEAD at audit time | `a9f7cebfcf6830edb2cdea791295bffc97d18afd` (D-016 commit) |
| Uncommitted working-tree change | `config.json` only (orchestrator executor/iteration settings) — unrelated to Phase D, not touched by this audit |
| D-013 commit | `8e2b86cb379a5e33033c62c6f0924794ad5b4d12` |
| D-014 commit | `865fc5bd969feaae20c6b0d59bc6a46f8f151193` |
| D-015 commit | `4617f4ff1a62a33b46474633afbe3693dffb36f4` |
| D-016 commit | `a9f7cebfcf6830edb2cdea791295bffc97d18afd` |
| Model seeds | 40, 41, 42, 43, 44 (D-005/ADR-0172 cohort; unchanged, not touched) |
| D-013 evaluation seeds (target/regression/canary/causal panels) | 301, 302, 303, 304, 305 (all 5) |
| D-014 evaluation seed | **301 only** — a single eval seed, not the full D-013 5-seed panel |
| D-015/D-016 evaluation seeds | 301, 302, 303, 304, 305 (all 5, pooled) |
| `vocab_size` | 10 (`DEFAULT_VOCAB_SIZE`, `src/apc/environments/vocab.py:12`) |

SHA-256 digests of every document, report.json, and evaluation module read for this audit are
recorded in `D017_REVIEW_RECORD.json.provenance.file_sha256`. All five seeds' `parent/manifest.json`
(`runs/phase_d_d013_five_model_cohort/seed_{40..44}/`) and `candidate/candidate_manifest.json`
(`runs/phase_d_d013_sort_repair/seed_{40..44}/`) were confirmed present on disk; none is missing.

**Terminology cross-check (D-013 vs D-014 vs D-015 vs D-016):** all four tasks share the same
`vocab_size=10`, the same 5 model seeds, and the same `SORT` input-length domain `{3,...,10}`
(`PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` §3). They do **not** share the same aggregation unit:
D-013's target-panel metric is a per-class, per-seed EM pooled over 5 eval seeds x 1,000 examples/seed
(5,000 total); D-014's per-cell metric is a single-eval-seed (301) measurement of 1,000 examples;
D-015's four factorial cells pool over 5 lengths (per length group) x 5 eval seeds; D-016's cells pool
over 5 eval seeds x a balanced amount grid at exactly 2 lengths (6, 10), not the full `{3,...,10}`
range. Section 3.2 below documents a further difference in the underlying **content distribution**
(duplicate-tolerant vs. distinct-token-only) that is not visible from the aggregate numbers alone.
No missing artifact or hash mismatch was found; the gaps found are in the **scope of what each
number represents**, not in file integrity.

**One index-hygiene gap found (not a research-evidence issue):** `docs/DECISIONS.md`'s Phase D
index lists ADRs through ADR-0183 (D-015) and its own "next unused" hint still reads `ADR-0184`,
but ADR-0184 (D-016) was already appended to `docs/DECISIONS_PHASE_D.md` in the D-016 commit and
was never back-filled into the master index. This audit adds the missing index line for ADR-0184
alongside its own new ADR-0185 entry (section 5) — this restores index completeness for an
already-committed, unmodified decision; it does not alter any decision's content or status.

## 2. D-013 acceptance-criteria correspondence table

D-001 (`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` §8) fixes 6 acceptance criteria,
evaluated **per model, never rescued by a cross-model mean**. The table below maps each criterion to
its exact saved field in `runs/phase_d_d013_executor/report.json` and its per-seed outcome.

| # | Criterion | Field | 40 | 41 | 42 | 43 | 44 | Verdict |
|---|---|---|:-:|:-:|:-:|:-:|:-:|---|
| 1 | Target recovery (EM≥0.95, all 7 classes + standalone L3-5) | `acceptance.target_recovery_pass` | F | F | F | F | F | **CONFIRMED FAIL 0/5** |
| 2 | Existing-capability preservation (≤1pp degradation AND panel floor met) | `acceptance.existing_capability_preservation_pass` | P | F | P | P | F | **CONFIRMED PASS 3/5** |
| 3 | Causal control (correct≥0.95, gap≥0.50, none≤0.06) | `acceptance.causal_control_pass` | P | F | P | P | F | **CONFIRMED PASS 3/5** |
| 4 | Invariance (non-SORT hashes unchanged) | `local_sort_repair.before_hashes`/`after_hashes` | P | P | P | P | P | **CONFIRMED PASS 5/5** |
| 5 | Fresh-load parity (Δ=0 in fresh process) | `fresh_load.parity_pass` | P | P | P | P | P | **CONFIRMED PASS 5/5** |
| 6 | Full reporting (no omission) | — | P | P | P | P | P | **CONFIRMED PASS 5/5** |

Overall `acceptance.pass` is `False` for all 5 seeds (criterion 1 alone forces this), matching
ADR-0181's `task_result: FAIL`, 0/5 models. **No summary flag count is used to infer missing
evidence** — every cell above traces to a specific saved field, re-read directly from
`report.json`, not re-derived or estimated.

### 2.1 Criterion 4 re-verification detail (a caution against a naive check)

A naive whole-dict comparison of `before_hashes`/`after_hashes` initially appears to show a mismatch
in every seed. Re-inspection shows this is **not** a contract violation: in every one of the 5 seeds,
**exactly one** of 16 primitive-id keys differs (`'6'`, consistent with SORT's own fixed `op_to_id`
slot across all 5 independently-constructed models) and all 15 other keys are bit-identical. This is
the expected signature of "only SORT's own parameters change" (D-001 §2), not a defect. Recorded here
because a careless read of the same field would produce a false invariance-violation finding.

### 2.2 Per-class target-panel granularity (model seed × class)

| Class | 40 | 41 | 42 | 43 | 44 | Pattern |
|---|---:|---:|---:|---:|---:|---|
| `NEGATE->SELECT->SORT` | 0.9998 | 0.6546 | 1.0000 | 1.0000 | 0.6242 | seed-dependent (41/44 fail) |
| `SELECT->SORT->BIND` | 1.0000 | 0.3238 | 1.0000 | 0.9998 | 0.9994 | seed-dependent (41 fails) |
| `SELECT->SORT->NEGATE` | 0.6906 | 0.1270 | 1.0000 | 0.9992 | 0.4608 | seed-dependent, but also fails for "clean" seed 40 |
| `SELECT->SORT->REVERSE` | 0.0018 | 0.0348 | 0.0038 | 0.0034 | 0.0150 | **universal failure, all 5 seeds** |
| `SELECT->SORT->SELECT` | 0.4326 | 0.4976 | 0.4178 | 0.7422 | 0.4480 | **universal failure, all 5 seeds** |
| `SELECT->SORT->SHIFT` | 0.4934 | 0.0948 | 0.9578 | 0.9996 | 0.1538 | seed-dependent, but also fails for "clean" seed 40 |
| `SHIFT->SELECT->SORT` | 0.9994 | 0.6164 | 1.0000 | 0.9998 | 0.5738 | seed-dependent (41/44 fail) |

`SELECT->SORT->REVERSE` and `SELECT->SORT->SELECT` fail the 0.95 floor **in every seed**, including
seeds 40/42/43 whose SORT causal-control passes cleanly (criterion 3) — this failure is independent
of that seed's own SORT/BIND health. `NEGATE`/`SHIFT` as the terminal step fail for 41/44 (seed-level
SORT defect) **and separately** for seed 40 (0.6906, 0.4934) despite seed 40 passing causal_control
— consistent with D-014's `DOWNSTREAM_SHORT_SEQUENCE_PRIMITIVE_DEFECT` finding, not with seed 40's
own SORT quality.

### 2.3 Why seeds 41 and 44 fail criteria 2 and 3 (not rescued by averaging, not conflated)

- **Causal control failure (both):** `causal_controls.target_L3_L5.arms.correct.em` = 0.6952 (seed
  41) and 0.6978 (seed 44), against the 0.95 floor — SORT itself, even post-repair, does not reliably
  execute the sort operation at `{3,4,5}` for these two seeds. This is the **residual SORT defect**.
- **Existing-capability-preservation failure (both):** driven by the panel's-own-floor clause, not
  the degradation clause. Delta vs. `FROZEN_PARENT` is exactly `0.0000` on every canary/regression
  cell for both seeds (SORT repair cannot move non-SORT weights; criterion 4 confirms this
  structurally). The failure is that certain canary classes (`NEGATE->SELECT->BIND`,
  `REVERSE->SELECT->BIND`, `SHIFT->SELECT->BIND` — **no SORT step in the recipe**) were *already*
  below the expected near-ceiling floor in `FROZEN_PARENT` itself: seed 41 shows 0.4976/0.4968/0.4828
  on these three classes. This is a **pre-existing capability deficit of that particular
  independently-constructed model's BIND primitive**, not a repair side-effect. Seed 44's canary
  panel does not show this pattern (0.91-1.00); seed 44's own criterion-3/2 failures trace instead to
  SORT's residual defect and to a lower non-BIND canary score (`NEGATE->SHIFT->SELECT`=0.9060).

These are recorded as **distinct** findings (residual SORT defect vs. an independent, seed-41-only
BIND defect vs. seed-44's own profile), per the task's explicit instruction not to collapse seeds 41
and 44 into a single undifferentiated "failed seeds" bucket.

## 3. D-014/D-015/D-016 interpretation audit

### 3.1 (a) BIND seed 41 — co-occurrence vs. causation — **F1, CONFIRMED**

**Claim examined:** ADR-0183 (D-015) states seed-41 BIND's `EXPLAINED_BY_LENGTH` classification
"reflects that seed's own pre-existing upstream SORT defect rather than an independent BIND-specific
short-length weakness."

**Code trace:** `phase_d_d015_downstream_length_order_factorial.py`'s `_paired_examples` draws BIND's
standalone-evaluation content from an independent seeded RNG
(`hashlib.sha256(f"phase-d-d015:{op_name}:{length}:{eval_seed}:{n}")`) — synthetic content, never
derived from any SORT forward pass or composed boundary. `standalone_arms` (imported verbatim from
D-014) then runs BIND directly on this content via a fresh, content-only encoding
(`_encode_fresh` → `core.model.encode`). **There is no code path connecting SORT's own execution to
this measurement.** The non-SORT hash-identity gate (verified 5/5 seeds pass) further confirms BIND's
weights are bit-identical between `FROZEN_PARENT` and `LOCAL_SORT_REPAIR`, i.e. untouched by the
repair.

**Corroborating evidence:** D-013's own canary panel — classes with **no SORT step at all** — shows
the identical pattern for seed 41 (`NEGATE->SELECT->BIND`=0.4976, `REVERSE->SELECT->BIND`=0.4968,
`SHIFT->SELECT->BIND`=0.4828, in `FROZEN_PARENT`, delta=0 after repair). Seed 44 (which shares seed
41's SORT causal-control failure) shows **no** BIND effect in D-015 (`NO_DEFECT_DETECTED`) and
near-ceiling BIND canary scores (0.91-1.00); seeds 40/42/43 have neither defect.

**Conclusion:** the BIND short-length weakness and the SORT residual defect are two independent,
co-occurring properties of the one independently-constructed seed-41 model. The measurement that
produced the finding cannot be affected by SORT, and the cross-seed pattern (44 has the SORT defect
without the BIND defect) contradicts a causal link. **CONFIRMED: the underlying EM numbers
(SHORT≈0.463-0.473, LONG≈0.9999-1.0) are correct; the causal attribution to "upstream SORT defect" in
ADR-0183 is UNSUPPORTED_INTERPRETATION.**

### 3.2 (b) D-015/D-016 comparability — **F2, CONFIRMED**

**Claim examined:** ADR-0184 (D-016) attributes the size gap between D-015's seed-44 SHIFT
`order_margin` (+0.0839, `ABSENT`) and D-016's seed-44 length-10 `disorder_margin` (0.5000, `PRESENT`)
to D-015's coarser design "mixing ASCENDING's 0.0/1.0 split together with near-ceiling non-ascending
orders" that "full pairing... now exposes."

| | D-015 | D-016 |
|---|---|---|
| Content generation | i.i.d. **with replacement** (duplicate tokens allowed) | i.i.d. **without replacement** (distinct tokens only) |
| Order/disorder granularity | binary: UNSORTED / SORTED(=ascending) | 5 ordinal inversion-count levels |
| Length coverage | pooled `{3,4,5}` and `{6,...,10}` | exactly 2 boundary lengths (6, 10), tested separately |
| Argument (shift amount) | drawn at random per example | balanced, every valid amount, fully paired |
| `interaction_margin` definition | signed 2×2 contrast, `abs()` thresholded | max deviation from additive model over a 5×N grid |

Re-reading D-015's own `report.json["order_cells"]` for seed 44 / SHIFT / **length 10 only** (no
cross-length pooling): `SORTED` em = {0.902, 0.906, 0.908, 0.909, 0.929} across eval seeds 301-305 —
far above D-016's ASCENDING mean of 0.5000 at the identical length. **Isolating length 10 in D-015's
own data does not reproduce D-016's severity.** This shows cross-length pooling is not a sufficient
explanation for the gap.

**Conclusion:** a substantial part of the gap traces to the content-population change (duplicate-
tolerant vs. distinct-only tokens), documented further in 3.3, not purely to resolution/pairing
refinement. **CONFIRMED: ADR-0184's "dilution by coarser pairing" narrative is only partially
supported — an unacknowledged population change is an additional contributor.** This does not
overturn D-016's seed-44 numbers; it qualifies the "exposes the same effect" framing, since the two
diagnostics do not sample the same input population.

### 3.3 (c) D-016 input diversity — **F3, CONFIRMED**

D-016's `report.json`/cell files do not store raw input tokens, only aggregate EM per cell. Per the
task's explicit allowance, inputs (not model outputs) were deterministically regenerated using the
exact, already-registered `_paired_disorder_examples` generator at the exact recorded
`(length, eval_seed, n)` arguments — **zero model forward passes**, and the function is a pure,
already-committed part of the D-016 module (SHA-256 `670712a6...`).

| Length | Disorder level | Unique inputs (pooled, 5 eval seeds × 1,000) |
|---:|---|---:|
| 6 | ASCENDING | 210 (= all of C(10,6)) |
| 6 | LOW/MEDIUM/HIGH_INVERSION | 3,028 / 4,188 / 3,088 |
| 6 | DESCENDING | 210 (= all of C(10,6)) |
| **10** | **ASCENDING** | **1** — always exactly `(0,1,2,...,9)` |
| 10 | LOW/MEDIUM/HIGH_INVERSION | 1,747 / 4,453 / 1,780 |
| **10** | **DESCENDING** | **1** — always exactly `(9,8,...,1,0)` |

**Root cause:** at `length == vocab_size == 10`, `rng.sample(range(10), 10)` (sampling without
replacement) can only ever return a permutation of the full alphabet `{0,...,9}`; after `sorted()`,
the drawn multiset is deterministically `{0,...,9}` on every call, for every RNG seed. The 0- and
max-inversion permutations of a fixed multiset are each unique, so ASCENDING and DESCENDING at
length 10 collapse to a single fixed input regardless of `eval_seed` or the nominal `n=1000`.

**Consequence for the headline seed-44/length-10 finding:** "exactly 5 of the 10 amounts
(`{0,2,3,5,7}`) give EM=0.0 while the other 5 (`{1,4,6,8,9}`) give EM=1.0" (ADR-0184) is an accurate,
CONFIRMED description of **one deterministic model's response to one fixed input** under 10 shift
amounts — the clean 0.0/1.0 split (no intermediate values) is exactly what a deterministic model
repeatedly given an identical input would produce, which is independently consistent with this
diversity finding rather than contradicting it. It is **not** a sampled measurement of "ascending
order at length 10" as a population, and `disorder_margin=0.5000`/`interaction_margin=0.4300` are
correspondingly anchored at their two extreme disorder levels by single-input case studies, with only
the three interior levels (LOW/MEDIUM/HIGH_INVERSION) reflecting genuine sample diversity at that
length. ADR-0184 does not state this scope limitation.

### 3.4 (d) Scope of applicability: ABSENT-by-threshold vs. no-effect; single-seed vs. 5-model — **F4, CONFIRMED**

| Finding | Reproducibility (5 model seeds) |
|---|---|
| `REVERSE`/`SELECT` `LENGTH_MAIN_EFFECT` | **REPRODUCIBLE_PRESENT (5/5)** |
| `NEGATE`/`SHIFT` `LENGTH_MAIN_EFFECT` (downstream defect) | `NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS` (3/5) |
| `BIND` `LENGTH_MAIN_EFFECT` | `NOT_REPRODUCIBLE_ACROSS_MODEL_SEEDS` (1/5 — seed 41 only, see 3.1) |
| D-015 `ORDER_MAIN_EFFECT`, all 5 primitives | `REPRODUCIBLE_ABSENT` (margins in [-0.0215, +0.0839] vs. 0.10 threshold — none clears the bar; this is "did not clear a pre-registered bar," not "zero order sensitivity exists") |
| D-016 `DISORDER`/`ARGUMENT`/`INTERACTION` effects | `PRESENT` in at most 1/5 seeds (seed 44) at either tested length |

**Conclusion:** the `REVERSE`/`SELECT` short-length capacity deficit is the only finding in this
entire evidence chain reproduced across all 5 independently-constructed models — it is the
strongest-supported candidate for carrying into a next charter's hypothesis. The seed-44 SHIFT and
seed-41 BIND findings are each specific to one model out of five and (per 3.1, 3.3) partially
artifacts of measurement design; they must not be presented as general `SHIFT` or `BIND` properties.
"`ABSENT`" in D-015's order effect means "did not clear the pre-registered 0.10 margin," not "provably
zero effect" — seed 44's own `+0.0839` sits close to that bar.

## 4. Boundary items for next-charter design (not authorized here)

Per task scope, this section separates issues for a **future, separately-authorized charter's
review** — no new hypothesis, primitive target, comparison condition, data boundary, budget, or
acceptance criterion is proposed or approved here.

1. **SORT's own residual defect**, surviving the registered single-primitive repair recipe, present
   with varying severity across seeds 41/44 (severe) vs. 40/42/43 (largely resolved per
   `causal_control_pass=True`). This is D-013's own headline result and is unchanged.
2. **REVERSE/SELECT short-length (`{3,4,5}`) deficits**, reproduced in all 5 models, independent of
   that seed's own SORT/BIND health (`SELECT->SORT->REVERSE`/`SELECT->SORT->SELECT` fail in every
   seed; D-015 `LENGTH_MAIN_EFFECT` `REPRODUCIBLE_PRESENT` 5/5). **The best-evidenced item.**
3. **NEGATE/SHIFT seed-dependent downstream deficits** (3/5 seeds), distinct from item 2 by
   reproducibility grade — present but not universal.
4. **Seed-44-only SHIFT anomaly at boundary lengths**, now shown (section 3.2-3.3) to be partially
   confounded by a single-input measurement artifact at length 10 and a content-population mismatch
   relative to D-015; requires its own re-scoped, population-matched diagnostic before being treated
   as confirmatory evidence, separate from item 3.
5. **Seed-41-only BIND short-length weakness**, independent of SORT (section 3.1); a fifth,
   previously-uncategorized item, not to be merged into item 1 or item 4.

**On the framing for a candidate future hypothesis** ("does saturating each primitive's own
standalone capacity over the composition-boundary input region recover continuous execution?"): items
1-3 above are consistent with, and could inform the design of, such a hypothesis; items 4-5 are
narrower single-model observations that would need independent confirmation first. **This audit does
not evaluate, endorse, or authorize that hypothesis, any repair, or any new training** — that
is explicitly reserved for a separate charter's review, and the D-013 cohort (already observed by
this audit and its predecessors) may not be reused as an unobserved confirmatory cohort for it
(per task instruction).

## 5. Completion, ADR, and transition judgment

- Every criterion in section 2 and every sub-item in section 3 traces to a specific saved field, an
  exact code trace, or a deterministic input-only regeneration; none is inferred from a summary flag
  or plausible-sounding narrative alone.
- No additional experiment was run. Where an item could not be fully resolved from saved artifacts
  without new computation, it is recorded as **UNDETERMINED** in `D017_REVIEW_RECORD.json` rather
  than filled with a plausible explanation (none of the section-3 items required this; all four
  resolved to CONFIRMED given the available code and artifacts).
- New ADR: **ADR-0185** (`docs/DECISIONS_PHASE_D.md`), recording this audit, its four findings
  (F1-F4), and the boundary-item separation in section 4. `docs/DECISIONS.md`'s Phase D index is
  updated with the missing ADR-0184 line and a new ADR-0185 line; the "next unused" hint is advanced
  to `ADR-0186`.
- **Transition judgment:** **PROCEED** to next-charter design review for the REVERSE/SELECT
  short-length line (item 2, section 4) — it has 5/5-model reproducible, cross-validated (target
  panel + D-015 factorial) saved evidence and is free of the SORT/BIND seed-specific confounds found
  elsewhere in this audit. **HOLD** the seed-44 SHIFT and seed-41 BIND findings (items 4-5) out of
  any next charter's confirmatory scope until a re-scoped, population-matched diagnostic addresses
  the gaps in sections 3.2-3.3; they remain valid, recorded, single-model observations, not general
  primitive properties. This judgment does not authorize any of the boundary-item follow-ups in
  section 4 — a next charter requires its own separate authorization, hypothesis, and acceptance
  criteria per AGENTS.md.
