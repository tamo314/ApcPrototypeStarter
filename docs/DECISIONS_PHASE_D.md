# Decisions — Phase D: Compositional Execution & Local Repair

This file holds the detailed Architectural Decision Records (ADRs) for Phase D. See
`docs/DECISIONS.md` for the master index and cross-phase context.

## ADR-0169: D-001 Phase D Charter Draft, Composition Execution Contract, and SORT-Only Repair Pilot Preregistration

**Date:** 2026-09-13
**Task:** D-001 — Compositional Execution Contract & SORT-Only Repair Pilot Preregistration
**Status:** Design/preregistration task completed. `charter_status: DRAFT_NOT_APPROVED`,
`training_execution: NOT_AUTHORIZED`, `candidate_selected: null`, `bundle_promotion: NOT_AUTHORIZED`,
`sealed_access: 0`.

**Scope and Integrity Boundary:**
- This task opens a new, independently scoped research charter (Phase D). It is explicitly **not**
  a resumption of Phase B (`CLOSED_ARCHIVED`, `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`),
  Phase C (`TERMINATED_CURRENT_CHARTER`), or NRQ-003 (`BLOCKED_BY_MODEL_ADEQUACY`). All of those
  terminal states, their FAILs, the Phase-B G1 independent-relation deficit, and the sealed-data
  partition boundary are preserved unmodified (`docs/DECISIONS_PHASE_B.md`,
  `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`, `docs/DECISIONS_PHASE_C.md`).
- Work performed was limited to: reading existing artifacts and source, re-aggregating already
  recorded per-cell numbers (independently re-derived and bit-for-bit verified against
  `docs/research/NRQ006_REVIEW_RECORD.json`'s 1200 raw per-cell records during this task's own
  verification pass), static semantic analysis of `src/apc/environments/operations.py`,
  `src/apc/primitives/{bank,composition,primitive}.py`, and `src/apc/utils/model_bundle.py`, and
  document authorship. No optimizer step, no model initialization, no candidate construction, and
  no sealed-data access occurred.
- The new research question is fixed to a single, narrow claim: under an explicitly given operation
  sequence and its arguments, with the Stable Core and every non-target primitive frozen, can
  updating only one target primitive's own parameters repair an execution failure in a sub-region of
  the composition input domain, while preserving existing capability, reproducibly across
  independently constructed Core/bank models? The first (and only, for this charter) target
  primitive is **SORT**. Unknown task-identity inference, latent routing-identity discovery, unknown
  relation transfer, and learned-search superiority are explicitly excluded from this initial
  research's claims.

**Key Findings / Deliverables:**
1. **Phase D research charter (draft):** `docs/research/PHASE_D_RESEARCH_CHARTER.md`. States H-D1,
   the excluded claims, and the final decision/status block. Also documents and resolves a
   pre-existing, unrelated name collision: `docs/HARDWARE_ENVIRONMENT.md` previously used "Phase D"
   as an informal placeholder label for hypothetical, out-of-scope 0.5B–2B pretrained-LM/LoRA
   adaptation work; that file is corrected in this task to remove the now-ambiguous label (pretrained
   LMs remain excluded by `AGENTS.md`'s non-negotiable invariants).
2. **Composition execution contract:** `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md`.
   Fixes, for the existing 8-primitive canonical registry and fixed depth-<=3 grammar, the input
   domain, argument domain/normalization, output domain, composition conditions, training-exposure
   vs. newly-registered valid-length extension, and execution-evidence separation, all derived
   directly from `Operation` subclasses' own `min_input_length` / `output_length` /
   `is_valid_for_length` and from `_train_single_primitive`'s actual sampling code — not assumed.
   Defines and keeps distinct four states — `SHAPE_COMPATIBLE`, `LENGTH_SUPPORTED`,
   `STANDALONE_QUALIFIED`, `COMPOSITION_QUALIFIED` — the latter two left `UNDETERMINED` pending a
   future execution task. SORT's valid length set is derived (not guessed) as
   `{3,4,5,6,7,8,9,10}`, where `{3,4,5}` = `SelectOp.output_length` applied to the training range
   `{6,...,10}` and `{6,...,10}` is the existing training-exposure range.
3. **Target/regression/causal-control panel manifests:**
   `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` and
   `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json`. Fix the target panel as the 7 canonical classes
   NRQ-007 attributed to `SHORT_SEQUENCE_CAPACITY_DEFICIT` with SORT immediately after SELECT
   (membership independently re-verified against `NRQ007_REVIEW_RECORD.json.class_attributions`
   during this task: exactly 7 classes match), three regression panels (standalone length-adequate
   SORT; 11 length-adequate SORT compositions; 8 non-SORT canary compositions), and the applicable
   Correct/Wrong-family/None causal control for SORT's parameter-free signature. Cross-checks and
   reconciles the 60-class/1200-cell base against the mean-threshold 35 failure classes and the
   all-cell-gate 41 failure classes; the 6-class difference (independently recomputed in this task
   directly from the 1200 raw per-cell records and found to match exactly:
   `NEGATE->REVERSE->SHIFT`, `NEGATE->SHIFT->SHIFT`, `REVERSE->SHIFT->SELECT`, `SHIFT->SHIFT->BIND`,
   `SHIFT->SHIFT->SELECT`, `SORT->SHIFT->SHIFT`) is recorded as `UNCLASSIFIED_BY_NRQ007` (never an
   input to NRQ-007's causal attribution, which ran only over the 35) rather than folded into the
   35-class attribution registry. None of the 7 target-panel classes are in this 6-class diff.
4. **Five-model cohort construction and provenance plan (contract only, cohort not built):**
   `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`. Fixes five new model seeds
   (`30,31,32,33,34`, disjoint from every existing seed namespace in the repository), reuses the
   existing `ModelBundleManifest` / `load_bundle` fail-closed provenance contract
   (`src/apc/utils/model_bundle.py`) without modification, and explicitly does **not** treat the old
   4 reconstructed bundles (seeds 1-4, NRQ-004/ADR-0164) as 4/5 of this cohort — seed 0 remains lost
   and is not backfilled by relabeling. Proposes (but does not implement) a `cohort_id` /
   `cohort_member_seeds` / `cohort_construction_recipe_hash` manifest extension. Fixes a dedicated
   parent-cohort construction budget (Core pretrain 16,000 steps/model + 42,400 bank/router steps/
   model, explicitly separate from the repair budget).
5. **SORT-only repair pilot preregistration:**
   `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`. Fixes a single repair
   hypothesis (no architecture/optimizer/LR/loss/sampling sweep), confirms and cites the
   shared-vs-SORT-owned parameter ownership boundary from source (`PrimitiveBank`'s per-primitive
   `ModuleDict` slicing, `bank.freeze_all()` + `bank.unfreeze(op_to_id["SORT"])`), the unchanged
   `CrossPositionPrimitive` architecture, the inherited optimizer/scheduler/loss
   (`AdamW(lr=0.0008, weight_decay=0.0001)`, cosine schedule, plain per-token cross-entropy, grad
   clip 1.0, batch 32 — all read from `_train_single_primitive`'s existing defaults), and a single
   fixed length-sampling rule (`L ~ Uniform{3,...,10}`). Registers three comparison conditions
   (`FROZEN_PARENT`, `LOCAL_SORT_REPAIR`, `SYMBOLIC_REFERENCE`), five required-`false`/`0`
   information-boundary flags for `LOCAL_SORT_REPAIR`, staged-candidate namespacing that never
   overwrites the parent bundle, fixed exhaustive-vs-sampled evaluation regimes and Wilson-CI sample
   sizes, six numeric acceptance criteria (per-model, per-cell, no rescue-by-averaging), and a
   registered repair budget ceiling (6,000 updates/model, 30,000/5-model cohort, exclusive of
   cohort-construction budget).
6. **Final decision recorded in the charter:** every dimension required to be numerically fixed
   before execution (source citations, input domain, cohort seeds/procedure, evaluation sample
   sizes/regime, budget) was resolved from existing source/artifacts; design completeness is
   explicitly not execution approval.

**Decision:**
Declare **`COMPOSITION_REPAIR_CONTRACT_READY_FOR_REVIEW`**.
The Phase D charter, composition execution contract, panel manifests, five-model cohort
construction contract, and SORT-only repair pilot preregistration are complete and internally
consistent. No design prerequisite gap remains open. Approval, cohort construction, and pilot
execution remain separate, not-yet-authorized steps.

```
design_status       = READY_FOR_REVIEW
charter_status       = DRAFT_NOT_APPROVED
training_execution   = NOT_AUTHORIZED
candidate_selected   = null
bundle_promotion     = NOT_AUTHORIZED
sealed_access        = 0
```

**Consequences:**
- Phase B/C terminal states, their FAILs, the G1 independent-relation deficit, and the sealed-data
  boundary are unmodified and continue to apply.
- No training, candidate construction, cohort construction, or sealed access is authorized by this
  ADR. A separately authorized execution task is required before any optimizer step runs, and that
  task must itself first build and strict-fresh-load-verify the five-model cohort under its own
  separately tracked budget before running the repair recipe.
- A future pilot's success, if it occurs, would license only a local-repair claim scoped to the
  registered 7-class target panel and the `{3,4,5}` standalone SORT domain on the specific 5-model
  cohort — not a 60-class composition-execution guarantee, not unknown-relation transfer, not a
  Phase B/C reversal, and not a candidate-bundle production adoption decision. A pilot failure would
  be recorded as a failure of the exact registered recipe only, with no automatic escalation to
  additional seeds, architectures, other-primitive repairs, or a general APC-infeasibility claim.

**Primary Artifacts:**
- Charter: `docs/research/PHASE_D_RESEARCH_CHARTER.md`
- Composition execution contract: `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md`
- Panel manifests: `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`,
  `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json`
- SORT-only repair pilot preregistration:
  `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`
- Five-model cohort construction/provenance contract:
  `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`
- Incidental correction: `docs/HARDWARE_ENVIRONMENT.md` (removed a pre-existing, unrelated,
  now-ambiguous "Phase D" placeholder label)
- Upstream sources re-aggregated (not re-executed): `docs/research/NRQ006_REVIEW_RECORD.json`,
  `docs/research/NRQ007_REVIEW_RECORD.json`, `docs/research/ARGUMENT_CLOSED_DEPTH3_CLOSURE_AUDIT_NRQ006.md`,
  `docs/research/STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md`,
  `docs/research/REPLICATION_AND_SUPPORT_BUDGET_SENSITIVITY_NRQ008.md`

## ADR-0170: D-003 — Phase D Charter Authorization Decision (Scoped Approval)

**Date:** 2026-09-13
**Task:** D-003 — Phase D Charter Authorization Decision
**Status:** Authorization decision recorded. `charter_status: APPROVED`, `training_execution: AUTHORIZED`
(scoped, see Decision below), `candidate_selected: null`, `bundle_promotion: NOT_AUTHORIZED`,
`sealed_access: 0`.

**Nature of this task:** a bulk approval review of D-001's completed deliverables, not an
implementation or execution task. No training, cohort construction, candidate construction, or
sealed-data access was performed to produce this decision.

**Review performed:**
The reviewer independently re-read, in full, the four D-001 primary artifacts and ADR-0169 itself:
- `docs/research/PHASE_D_RESEARCH_CHARTER.md` (charter, H-D1, exclusions, final decision block)
- `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` (four-state distinction, derived
  valid-length set, dependency-hash contract)
- `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` and `PHASE_D_D001_PANEL_MANIFEST.json`
  (target/regression/canary/causal-control panels, 60/1200/35/41 cross-check, 6-class
  `UNCLASSIFIED_BY_NRQ007` diff)
- `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` (seeds 30-34, construction
  stage graph, budget table, unresolved items)
- `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` (single recipe,
  ownership/freeze boundary, comparison conditions, information-boundary flags, sample sizes, six
  acceptance criteria, budget, claim-scope statement)

As an independent spot-check (not a full re-verification of every citation already re-verified
during D-001), the reviewer re-read source directly and confirmed: `SortOp`
(`src/apc/environments/operations.py:279-303`) has no `required_argument_names` override
(parameter-free), `output_length` is the identity map, and `apply` is exactly
`tuple(sorted(sequence))`; `SelectOp.output_length` (`:151-152`) is exactly
`max(1, input_length // 2)`, confirming the derived `{3,4,5}` intermediate-length set;
`PrimitiveBank.freeze` / `.unfreeze` / `.freeze_all` (`src/apc/primitives/bank.py:235-249`) exist
exactly as cited and give the per-primitive freeze granularity the repair recipe's ownership
boundary depends on. No discrepancy was found between the preregistration's claims and the current
repository state.

**Findings:**
1. All four D-001 deliverables and ADR-0169 are internally consistent with each other and with
   `AGENTS.md`'s non-negotiable invariants: Core stays frozen and content-only
   (`h_content = f(content)` preserved), only SORT's own parameter slice is unfrozen, execution
   uses the existing unmodified `execute_composition_recipe` path (no ground-truth intermediate
   injection as a training or primary-metric signal), a Correct/Wrong-family/None causal control is
   registered, candidate output is staged into a new namespace with `bundle_promotion` remaining
   `NOT_AUTHORIZED` regardless of outcome, and `sealed_access` remains `0` throughout with the
   Phase-B G1 deficit and sealed-partition boundary explicitly preserved.
2. The registered budget (repair: 6,000 updates/model, 30,000/5-model cohort; cohort construction:
   16,000 Core-pretrain + 42,400 bank/router steps/model, plus the already-bounded existing
   SHIFT-dedicated-repair recipe) fits `AGENTS.md`'s single RTX 5060 Ti (16GB)/64GB-RAM default
   hardware envelope as a one-time milestone experiment, not a routine test; wall-time/VRAM/RAM
   figures are explicitly labeled planning estimates and the preregistration already commits to
   replacing them with measurements at execution time.
3. Every dimension the task instructions require to be fixed before execution (architecture,
   optimizer/scheduler/loss, input domain, cohort seeds/procedure, evaluation sample sizes/regime,
   numeric acceptance floors, budget ceilings) is resolved to a single fixed value with no
   simultaneous sweep, and STOP conditions and non-escalation-on-failure are explicitly registered
   (pilot preregistration section 10; charter "STOP and execution budget boundary").
4. No prerequisite gap remains open that would block execution under the exact registered recipe.
   Two items are explicitly left as future/registered-only (not gaps in this review): the
   `cohort_id`/`cohort_member_seeds`/`cohort_construction_recipe_hash` manifest fields are a
   proposal the execution task must still implement or explicitly forgo before/while building the
   cohort, and `STANDALONE_QUALIFIED`/`COMPOSITION_QUALIFIED` remain `UNDETERMINED` pending the
   pilot's own result — both are correctly scoped as execution-task responsibilities, not
   authorization blockers.

**Decision:**
**Approve**, limited exactly to the scope named in this task's instruction: (a) construction and
strict fresh-load verification of the five-model cohort at seeds `30,31,32,33,34` per
`PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`, (b) execution of the single preregistered
`LOCAL_SORT_REPAIR` recipe (`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 4)
exactly as fixed, with no architecture/optimizer/LR/loss/sampling/step-budget sweep, and (c) the
registered `FROZEN_PARENT`/`SYMBOLIC_REFERENCE` comparison-condition evaluations and the registered
target/regression/canary/causal-control panels (sections 5, 7-8 of the pilot preregistration; the
panel manifest). No broader authorization is granted or implied.

```
charter_status       = APPROVED
training_execution   = AUTHORIZED   # scoped to (a)+(b)+(c) above only
candidate_selected   = null
bundle_promotion     = NOT_AUTHORIZED
sealed_access        = 0
design_status        = READY_FOR_REVIEW  (unchanged from ADR-0169)
```

**Explicitly not authorized by this decision:**
- Any primitive other than `SORT` (e.g. the `*->BIND->COUNT` `ARGUMENT_HANDLING` class) — requires
  a separate charter task.
- Any deviation from the single fixed recipe: no additional seeds beyond `30-34`, no step-budget
  increase beyond 6,000/model repair (30,000/5-model) or the registered cohort-construction budget,
  no architecture/optimizer/LR/loss/sampling sweep.
- Candidate/bundle promotion of any kind — `bundle_promotion` remains `NOT_AUTHORIZED` regardless of
  pilot outcome.
- Any sealed-data read, generation, or evaluation — `sealed_access` remains `0`.
- Any resumption of Phase B (`CLOSED_ARCHIVED`) or Phase C (`TERMINATED_CURRENT_CHARTER`), any
  relaxation of the Phase-B G1 independent-relation deficit, or any reversal of NRQ-003's
  `BLOCKED_BY_MODEL_ADEQUACY` status.
- Extension of a pilot pass into a claim beyond the registered 7-class target panel and `{3,4,5}`
  standalone domain on this specific 5-model cohort (pilot preregistration section 10), or
  extension of a pilot fail into a general APC-infeasibility claim.
- Selection or exclusion of cohort members based on results (the 5 seeds are fixed
  pre-registration; no post-hoc replacement).

**This approval-decision task itself performed no training, no cohort construction, no candidate
construction, and no sealed-data access.** It consisted only of reading the existing D-001
artifacts, this ADR's own independent source spot-check, and recording this decision. The actual
cohort build and the `LOCAL_SORT_REPAIR` execution remain a separate, subsequent execution task's
responsibility, which must itself record commit/config/seeds, budgets, hardware/time/memory,
parameter accounting, and the full result set for every registered run per `AGENTS.md`'s evidence
requirements, and which is bound by the exact recipe, panels, and acceptance criteria fixed by
D-001 — this authorization does not permit that task to re-open or re-fix any of those numbers.

**Consequences:**
- Phase B/C terminal states, their FAILs, the G1 independent-relation deficit, and the sealed-data
  boundary remain unmodified and continue to apply, unaffected by this approval.
- A future execution task may now build the seed-`30-34` cohort and run the exact registered
  `LOCAL_SORT_REPAIR` recipe and its registered evaluations without a further charter-level
  approval step, provided it does not deviate from any fixed value in the D-001 documents; any
  deviation (recipe, budget, seeds, panels, criteria) requires a new authorization.
- If that execution task's pilot fails any of the six acceptance criteria, the failure must be
  recorded as a failure of the exact registered recipe on the exact registered cohort, artifacts
  preserved, and dependent work (any `bundle_promotion`, any other-primitive repair, any
  composition-level retraining) stopped pending a new authorized task, per `AGENTS.md`'s STOP-GATE
  handling.

**Primary Artifacts (reviewed; unmodified by this decision except the status-field/pointer updates
noted):**
- Charter: `docs/research/PHASE_D_RESEARCH_CHARTER.md` (status header and final-decision block
  updated to reflect this approval)
- Composition execution contract: `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md`
  (unchanged)
- Panel manifests: `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`,
  `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json` (unchanged)
- SORT-only repair pilot preregistration:
  `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` (status/non-authorization
  statement updated to point to this ADR; no numeric criterion, recipe field, or budget number
  changed)
- Five-model cohort construction/provenance contract:
  `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` (unchanged; pointer note
  added)
- Prior ADR: [ADR-0169](#adr-0169-d-001-phase-d-charter-draft-composition-execution-contract-and-sort-only-repair-pilot-preregistration)
