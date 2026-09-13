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
