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

## ADR-0171: D-004 — SORT-Only Repair Confirmation Execution STOP GATE FAILS on the Data-Boundary Prerequisite (Seeds 30-34 Collide with the Sealed V2 Partition)

**Date:** 2026-09-13
**Task:** D-004 — Execute the D-001-preregistered, D-003-authorized single 5-model `LOCAL_SORT_REPAIR`
confirmation experiment (seed `30-34` cohort construction, then repair, then the full registered
evaluation suite).
**Status:** **STOP GATE FAIL**, at the data-boundary/sealed-partition prerequisite check, before any
cohort construction. `training_execution` remains `AUTHORIZED` per ADR-0170 in principle, but this
task performed **zero** optimizer steps, **zero** model initializations, **zero** candidate
constructions, and **zero** sealed-data accesses — execution did not begin. `candidate_selected: null`,
`bundle_promotion: NOT_AUTHORIZED`, `sealed_access: 0` (unchanged).

**What this task attempted:** per its own instruction, D-004 was to verify provenance, data boundary
(データ境界), hash integrity, strict fresh-load, and `FROZEN_PARENT` eligibility as STOP GATEs
*before* constructing the seed-`30,31,32,33,34` cohort registered by
`docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` (D-001) and approved by
ADR-0170 (D-003). Before writing any orchestration code, this task re-verified the cohort contract's
own seed-disjointness claim (§2 of that document) against the repository's *complete* existing
seed-namespace registry, not only the four namespaces that document's own table checked
(NRQ-005–008 reconstructed-bundle model seeds `1-4`; Model Bundle Recovery dev seeds `10-14`;
NRQ-006/007 data seed `101-105`; NRQ-008 data seed pool `201-220`).

**Finding — a genuine, mechanically-enforced seed collision:**

`src/apc/evaluation/relation_split_protocol.py:95` registers
`NEW_SEALED_V2_SEEDS: Final[tuple[int, ...]] = (30, 31, 32, 33, 34)` — **the exact same five integers**
the Phase D cohort contract independently chose, believing them unused. This is not a coincidental
namespace (e.g. an unrelated data-seed range): `build_seed_partition_registration`
(`relation_split_protocol.py:393-427`) fixes `"sealed_v2": {"seeds": list(NEW_SEALED_V2_SEEDS),
"model_seed_recipe": "model_seed = seed", "status": "MEMBERSHIP_REGISTERED_NOT_MEASURED", "note":
"Model outputs for these seeds are not measured by R3-002; measurement is R3-012's job, after R3-011
seals this membership."}` — i.e. it is the **same `ModelBundleManifest.model_seed` namespace** the
Phase D cohort contract's §5 proposes to populate, reserved by an earlier Phase B task
(`B-C005R3-002`, `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`) specifically as a
**held-out sealed partition** for the not-yet-executed `R3-011` (seal) / `R3-012` (sealed gate)
pathway.

`assert_sealed_access_permitted` (`relation_split_protocol.py:105-119`) mechanically enforces this
reservation: `sealed = set(SEALED_GATE_SEEDS) | set(DEFAULT_REGATE_SEEDS) | set(NEW_SEALED_V2_SEEDS)`
— i.e. `{0,1,2,3,4} | {20,21,22,23,24} | {30,31,32,33,34}` (values independently confirmed at
`src/apc/evaluation/retrieval_repair_benchmark.py:73-74` and
`src/apc/evaluation/hard_negative_repair_gate.py:53`) — and raises `ValueError` on any access to a
seed in that set unless `purpose == "R3-011_seal_or_R3-012_gate"`. Every existing caller of this guard
in the repository (e.g. `shift_functional_generalization_repair.py:673-675`, the exact SHIFT-repair
module the cohort contract's §3 depends on for its `SHIFT_DEDICATED` stage) passes its own
task-specific purpose string and is therefore correctly blocked from touching seeds `30-34`.
`docs/DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md:64` confirms the current status in prose: "no model
output for these seeds is measured until R3-011 seals this membership and R3-012 runs the sealed
gate," and `docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md:4,18` confirm `R3-011`/`R3-012` remain
paused/blocked and have never executed (independently confirmed here by `git log --all --oneline`
across the full repository history: no commit references either task). This finding was independently
re-derived from source during this task, not taken on the sub-agent report's word alone.

**Why this is a STOP GATE, not a routing-around problem:** building the Phase D cohort on seeds
`30-34` — training a fresh Core + primitive bank per seed, running `LOCAL_SORT_REPAIR`, and evaluating
the registered target/regression/canary/causal-control panels on the result — would be exactly the
"measurement of model output" for these seeds that `B-C005R3-002` reserved exclusively for the
`R3-011`/`R3-012` pathway, performed instead by an unrelated charter outside that pathway. This
directly conflicts with `AGENTS.md`'s non-negotiable invariant ("Protect sealed data and enforce the
applicable data-disjointness checks before training") and with the Phase D charter's own explicit,
repeated promise that "the sealed-partition boundary... [is] preserved unmodified" by this charter
(`docs/research/PHASE_D_RESEARCH_CHARTER.md`). It would not merely be an undesirable side effect —
`assert_sealed_access_permitted` would raise `ValueError` the moment any orchestration script reused
the existing SHIFT-repair module (as the cohort contract's own §3 stage table requires) with these
seeds, so the collision is also mechanically fail-closed, not just a documentation inconsistency.

**Root cause:** `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` §2's
seed-disjointness table cross-checked only four specific prior-usage namespaces and concluded seeds
`30-34` were disjoint from "既存repo内で使用済みの全seed namespace" (every seed namespace already used
in the existing repo). It did not check `relation_split_protocol.py`'s `NEW_SEALED_V2_SEEDS`
registration (nor `NEW_VALIDATION_SEEDS = (15,16,17,18,19)`, nor `DEFAULT_REGATE_SEEDS =
(20,21,22,23,24)`), which predates the Phase D charter and was registered by the earlier Phase B task
`B-C005R3-002`. ADR-0169 (D-001)'s own artifact authorship and ADR-0170 (D-003)'s independent
source spot-check both missed this collision — D-003's spot-check re-verified `SortOp`/`SelectOp`/
`PrimitiveBank.freeze`-family citations but did not re-derive the cohort contract's seed-disjointness
claim against `relation_split_protocol.py`.

**Decision: STOP.** D-004 does not proceed to cohort construction, `LOCAL_SORT_REPAIR` training, or
any registered evaluation. This is recorded as a STOP GATE FAIL on the data-boundary/sealed-partition
prerequisite check — **not** a pilot result. H-D1 (the SORT local-repair hypothesis) is neither
confirmed nor refuted by this task; it remains untested. No optimizer step, no model initialization,
no candidate construction, no primitive repair, and no evaluation forward pass was performed; no
artifacts were created under `runs/phase_d_five_model_cohort/` or `runs/phase_d_d001_sort_repair/`
(none exist on disk — confirmed before writing this record).

Per this task's own explicit instruction and `AGENTS.md`'s STOP-GATE handling: the failed criterion is
reported here precisely (data-boundary / sealed-partition disjointness, not a recipe/hash/regression/
causal-control failure), this ADR is the evidence handoff, and dependent work stops — no additional
updates, seed exchange, alternate recipe/primitive, sealed access, candidate selection, or bundle
promotion follows from this finding.

**What this finding does *not* authorize or imply:**
- Does **not** authorize this task to unilaterally substitute different seeds and proceed — that is an
  unauthorized recipe/seed deviation requiring a new, separately recorded authorization per both
  ADR-0169's and ADR-0170's own explicit terms.
- Does **not** authorize invoking `assert_sealed_access_permitted(..., purpose="R3-011_seal_or_R3-012_gate")`
  to bypass the guard — this task is not the `R3-011`/`R3-012` sealed-gate pathway, and mischaracterizing
  its purpose to pass the guard would itself be the sealed-data violation the guard exists to prevent.
- Does **not** reverse or weaken ADR-0169/ADR-0170's approval of the `LOCAL_SORT_REPAIR` recipe design,
  the composition execution contract, or the panel manifests — those remain sound and approved; only
  the specific seed values `30-34` collide with a pre-existing reservation.
- Does **not** touch any Phase B/C terminal state, the G1 independent-relation deficit, or the
  sealed-partition boundary. This finding's purpose is to *protect* that boundary from being
  inadvertently breached by an unrelated charter, consistent with every prior Phase B/C/D document's
  repeated commitment to preserve it unmodified.

**Recommendation (not self-authorized by this task):** a future task would need either (a) a new,
explicitly authorized charter amendment selecting a genuinely unused model-seed set — checked against
`relation_split_protocol.py`'s complete registry (`SEALED_GATE_SEEDS={0-4}`,
`DEFAULT_REGATE_SEEDS=(20-24)`, `NEW_SEALED_V2_SEEDS=(30-34)`, `NEW_VALIDATION_SEEDS=(15-19)`) and
`RECOVERY_DEV_SEEDS=(10-14)`, not only the four namespaces D-001 originally checked — or (b) an
explicit, knowing decision by the charter owner to consume the `sealed_v2` partition for this
unrelated purpose, accepting that doing so would foreclose or complicate the future `R3-011`/`R3-012`
sealed-gate pathway Phase B reserved seeds `30-34` for. Both are charter-level decisions for the user,
not something this execution task can decide for itself.

```
design_status       = READY_FOR_REVIEW    (unchanged, ADR-0169)
charter_status       = APPROVED            (unchanged, ADR-0170)
training_execution   = AUTHORIZED          (unchanged, ADR-0170; scope unaffected, but execution
                                             did not begin due to this STOP GATE)
cohort_construction  = STOP_GATE_FAIL_DATA_BOUNDARY   (seeds 30-34 collide with NEW_SEALED_V2_SEEDS)
pilot_execution      = NOT_PERFORMED       (blocked on cohort_construction)
candidate_selected   = null
bundle_promotion     = NOT_AUTHORIZED
sealed_access        = 0
```

**Consequences:**
- Phase B/C terminal states, their FAILs, the G1 independent-relation deficit, and the sealed-data
  boundary (now including the `sealed_v2` partition specifically) remain unmodified and continue to
  apply.
- No cohort exists, no repair candidate exists, and no evaluation result exists for H-D1 under any
  seed. A future execution task requires a new authorization step (charter amendment or explicit
  sealed-partition-consumption decision, per the Recommendation above) before it may proceed; it is
  not a resumption of D-004's authorization as-is, since D-004's authorized seed values are precisely
  what collided.
- The `LOCAL_SORT_REPAIR` recipe, panel manifests, and evaluation/acceptance-criteria design
  (ADR-0169/ADR-0170) remain valid and reusable once a conforming seed set is authorized; none of
  their numeric content is invalidated by this finding.

**Primary Artifacts (reviewed; none created by this task's execution attempt):**
- Blocking source citation: `src/apc/evaluation/relation_split_protocol.py:95,105-119,393-427`
- Corroborating status record: `docs/DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md:64`,
  `docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md:4,18`
- Reviewed, unmodified: `docs/research/PHASE_D_RESEARCH_CHARTER.md`,
  `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`,
  `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`,
  `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` (pointer notes added to the cohort
  construction contract and pilot preregistration status headers only; no fixed seed, recipe,
  budget, or acceptance-criterion value changed)
- Prior ADRs: [ADR-0169](#adr-0169-d-001-phase-d-charter-draft-composition-execution-contract-and-sort-only-repair-pilot-preregistration),
   [ADR-0170](#adr-0170-d-003-phase-d-charter-authorization-decision-scoped-approval)

## ADR-0172: D-005 — Phase D Cohort Seed Amendment — Complete Static Registry Audit and Replacement Authorization

**Date:** 2026-09-13
**Task:** D-005 — Phase D cohort seed amendment
**Status:** **PASS — pre-result static provenance gate.** `charter_status: APPROVED`;
`training_execution: AUTHORIZED` only for the replacement seed-`40,41,42,43,44` cohort and the
unchanged D-001/ADR-0170 recipe, panels, controls, and budgets. `candidate_selected: null`,
`bundle_promotion: NOT_AUTHORIZED`, `sealed_access: 0`.

**Nature and boundary of this task:** This was a registry/provenance amendment only. It read Python
source ASTs and tracked JSON review records; it did **not** import experiment modules, initialize a
model, build a cohort, generate/evaluate data, train, inspect sealed model outputs, or access sealed
data. D-004's `STOP_GATE_FAIL_DATA_BOUNDARY` is historical evidence of the old seed set's collision,
not a pilot result; H-D1 remains untested.

**Mechanical audit performed before fixing a replacement:**

1. Added the frozen, reviewable registry
   `docs/phase_d/PHASE_D_D005_SEED_REGISTRY.json` and the fail-closed static checker
   `apc.evaluation.phase_d_seed_registry` / `python scripts/verify_phase_d_seed_registry.py`.
   The checker parses source assignments with `ast` rather than importing their modules, then compares
   them to the registry; it also checks top-level fields in each tracked NRQ review record.
2. Source split registrations exactly matched the registry: original sealed `SEALED_GATE_SEEDS=0–4`;
   recovery development `DEFAULT_DEV_SEEDS=10–14` and `RECOVERY_DEV_SEEDS=10–14`; validation
   `NEW_VALIDATION_SEEDS=15–19`; re-gate sealed `DEFAULT_REGATE_SEEDS=20–24`; and sealed V2
   `NEW_SEALED_V2_SEEDS=30–34`. The final group is explicitly a permanently forbidden Phase D
   model-seed set, not an unused interval.
3. Historical run provenance exactly matched: NRQ-005/006/007 report reconstructed bundle model
   seeds `1–4` and data seeds `101–105`; NRQ-008 reports bundle model seeds `1–4` and data seeds
   `201–220`. These data-seed axes are documented separately from model identity and are nevertheless
   checked for numeric collision as a conservative guard.
4. Candidate seed-`40–44` is sorted, unique, five members, and has an empty intersection with every
   source registry, historical bundle model seed, and historical data seed above. The checker would
   fail closed for the old candidate `30–34`; regression coverage records that case.

**Data lineage and invariants:** The amendment changes only `ModelBundleManifest.model_seed` values.
The construction lineage remains D-001's existing Model Bundle Recovery stage graph, fresh model
initialization per new model seed, frozen-parent/repair boundaries, and new Phase D namespaces. The
repair/evaluation data lineage remains the registered deterministic role-derived scheme
`d001_train:<model_seed>:<step>:<op>` and the separately registered evaluation derivation; it neither
uses a Phase B validation/sealed split nor reuses an NRQ data split. Core freezing after cohort build,
the `h_content = f(content)` boundary, sparse primitive execution, Correct/Wrong/None controls,
fresh-load parity, and all sealed-data prohibitions are unchanged.

**Decision — replace, do not broaden, ADR-0170's seed scope:**

ADR-0170's authorization to construct/evaluate the seed-`30–34` cohort is **revoked and replaced**.
Only seed-`40,41,42,43,44` is now authorized for: (a) five-model cohort construction and strict
fresh-load verification under the amended cohort contract, (b) the one pre-registered
`LOCAL_SORT_REPAIR` recipe, and (c) its already-registered `FROZEN_PARENT` / `SYMBOLIC_REFERENCE`
conditions and target/regression/canary/causal-control evaluations. The construction procedure,
architecture, optimizer, scheduler, loss, sampling, acceptance criteria, panels, output namespaces,
and both budgets are unchanged: cohort construction remains 16,000 Core-pretrain plus 42,400
bank/router steps per model (excluding the existing bounded SHIFT recipe); repair remains at most
6,000 updates/model and 30,000/5-model cohort.

```
design_status       = READY_FOR_REVIEW
charter_status      = APPROVED
training_execution  = AUTHORIZED  # seed 40-44 only; same D-001 recipe/panels/budgets
cohort_construction = NOT_PERFORMED
pilot_execution     = NOT_PERFORMED
candidate_selected  = null
bundle_promotion    = NOT_AUTHORIZED
sealed_access       = 0
```

**Adoption/replace prohibition:** The five seeds are fixed before any Phase D result. No cohort member
may be excluded, added, or exchanged after a build, training, or evaluation observation. A missing
artifact, registry mismatch, or a newly discovered collision is a STOP condition: preserve evidence,
record the exact failed check, and do not substitute a seed, increase a budget, or run a partial
cohort. Any change requires a new ADR with a complete static re-audit and explicit charter-level
authorization. Seed-`30–34` cannot be adopted or exchanged into this charter under any result.

**Not authorized:** consuming any sealed partition; any model/data access outside the registered
development lineage; another primitive; a recipe or budget change; an additional model seed; model
selection/promotion; or any Phase B/C status change. This amendment does not reopen Phase B,
relax G1, or consume the R3-011/R3-012 sealed V2 reservation.

**Artifacts changed by this amendment:**
- `docs/phase_d/PHASE_D_D005_SEED_REGISTRY.json` (complete static registry and lineage declaration)
- `src/apc/evaluation/phase_d_seed_registry.py`, `scripts/verify_phase_d_seed_registry.py`, and
  `tests/test_phase_d_seed_registry.py` (static fail-closed verification)
- `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`,
  `docs/research/PHASE_D_RESEARCH_CHARTER.md`, and
  `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` (seed scope only)
- Prior ADRs: [ADR-0169](#adr-0169-d-001-phase-d-charter-draft-composition-execution-contract-and-sort-only-repair-pilot-preregistration),
  [ADR-0170](#adr-0170-d-003-phase-d-charter-authorization-decision-scoped-approval), and
  [ADR-0171](#adr-0171-d-004-sort-only-repair-confirmation-execution-stop-gate-fails-on-the-data-boundary-prerequisite-seeds-30-34-collide-with-the-sealed-v2-partition)

## ADR-0173: D-006 — Execution-precondition STOP — Python 3.12 environment has CPU-only Torch and no constraint-compatible CUDA wheel

**Date:** 2026-09-13
**Task:** D-006 — Execute the ADR-0172-reauthorized seed-`40-44` single five-model SORT-only confirmation
**Status:** **EXECUTION_PRECONDITION_STOP — no scientific result.**

**Checks completed before any model/data access:**

1. `python scripts/verify_phase_d_seed_registry.py` passed.  It confirmed that cohort model seeds
   `40,41,42,43,44` are unique, disjoint from every registered model/data seed namespace, and do
   not access the sealed V2 partition.  This was a static AST/JSON provenance check only.
2. The repository's required runtime (`.venv\\Scripts\\python.exe`) is Python `3.12.13`, satisfying
   `pyproject.toml`'s `requires-python = ">=3.12"`.  Its installed Torch reports
   `2.13.0+cpu`, `torch.version.cuda is None`, `torch.backends.cuda.is_built() is False`, and
   `torch.cuda.is_available() is False`.
3. The only locally available CUDA-enabled interpreter reports CUDA on the RTX 5060 Ti, but is
   Python `3.10.11`, which is outside the project dependency contract.  It was not used for the
   experiment.
4. Read-only package-index inspection of the official CUDA 12.8 wheel index found its newest
   available wheel is `torch 2.11.0+cu128`.  That version is excluded by the tracked dependency
   range `torch>=2.12,<2.14`; replacing the 3.12 environment with it, lowering the constraint, or
   using the Python-3.10 environment would silently change the registered execution environment.

**Decision: STOP before cohort construction.**  The D-006 task requires the recorded wall-time,
GPU, and RAM measurements for five independently built models.  Running the registered
`16,000`-step Core pretraining, `42,400`-step bank/router construction (plus the registered SHIFT
recipe), and up-to-`6,000` SORT-repair updates per model in a CPU-only environment would not be
the registered single-GPU execution.  Per `AGENTS.md`'s implementation rule to use Python 3.12 and
the dependency constraints in `pyproject.toml`, this environment is not eligible for the cohort
build.  This is a missing execution prerequisite, not a failed FROZEN_PARENT or H-D1 cell.

No model was initialized; no training/evaluation examples, parent bundle, candidate, or
`LOCAL_SORT_REPAIR` optimizer were constructed; and sealed data/model outputs were not accessed.
`cohort_construction=NOT_PERFORMED`, `pilot_execution=NOT_PERFORMED`,
`candidate_selected=null`, `bundle_promotion=NOT_AUTHORIZED`, `sealed_access=0`, and all
pre-registered seed, recipe, panel, and budget values remain unchanged.  In particular, this STOP
does **not** authorize a seed exchange, additional updates, another recipe, sealed access,
candidate selection, or bundle promotion.

**Required condition to resume the already-authorized task:** install or otherwise provide a
Python-3.12 environment whose CUDA-enabled Torch satisfies the tracked dependency constraint and
can see the single RTX 5060 Ti; then rerun the unchanged D-006 execution from the static registry
gate.  This requires no change to the D-005 cohort authorization, but no CPU result may be used as
a substitute for it.
