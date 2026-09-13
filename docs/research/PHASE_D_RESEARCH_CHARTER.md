# Phase D — Compositional Execution & Local Repair, Research Charter (APPROVED, D-005 seed-amended scoped execution authorized)

Date: 2026-09-13. Version: charter-v2. **Status: `APPROVED` — Task D-013 completed full five-model cohort build and LOCAL_SORT_REPAIR confirmation on CUDA without errors. Under preregistered acceptance criteria, all 5 models failed target recovery threshold (collapsed on SELECT->SORT->REVERSE), yielding task_result: FAIL, h_d1_status: REFUTED (negative result). Sealed access: 0, candidate_selected: null, bundle_promotion: NOT_AUTHORIZED.**
`training_execution: AUTHORIZED`, scoped strictly to (a) the seed-`40,41,42,43,44` five-model
cohort construction (`PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`), (b) the single
preregistered `LOCAL_SORT_REPAIR` recipe (`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`
section 4), and (c) its registered `FROZEN_PARENT`/`SYMBOLIC_REFERENCE` comparison-condition and
panel evaluations (same document, sections 5, 7-8) — no other primitive, no additional seed, no
recipe/budget deviation, no candidate/bundle promotion, and no sealed-data access is authorized.
It is an independent, narrowly-scoped new research charter opened by Task D-001, not a resumption
of Phase B (`CLOSED_ARCHIVED`, `NEGATE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`), Phase C
(`TERMINATED_CURRENT_CHARTER`), or NRQ-003 (`BLOCKED_BY_MODEL_ADEQUACY`). Those terminal states,
their FAILs, their G1 deficit, and their sealed-partition boundary are preserved unmodified by this
document (`docs/DECISIONS.md` Phase B/C sections). A diagnostic success (NRQ-007's causal
attribution) or a proposed follow-up alone does not authorize execution or satisfy a
recovery/research gate (`AGENTS.md`, "Find the applicable contract").

**Note on a pre-existing name collision:** `docs/HARDWARE_ENVIRONMENT.md` previously used the
label "Phase D" as a placeholder for hypothetical future 0.5B-2B pretrained-LM/LoRA adaptation
work. That placeholder use predates this charter, is unrelated to it, and is not activated,
claimed, or extended by this document. `docs/HARDWARE_ENVIRONMENT.md` has been corrected (Task
D-001) to remove the now-ambiguous label; this charter is the only substantive "Phase D" going
forward. Pretrained LMs remain out of scope per `AGENTS.md`'s non-negotiable invariants.

## Independent question and falsifiable hypothesis

**Research question:** Under an explicitly given operation sequence and its arguments — with the
Stable Core and every non-target primitive frozen — can updating only a single target primitive's
own parameters repair an execution failure in a sub-region of the composition input domain, while
preserving existing capability, and does the effect reproduce across independent Core/bank models?

**H-D1:** Standalone training of one target primitive (`SORT`, held to its existing architecture)
over a valid input-length set derived from the composition execution contract — not merely
guessed — recovers both (a) that primitive's standalone execution and (b) its behavior inside
pre-registered depth-<=3 compositions, on a pre-registered sub-region where it was previously
failing (`SHORT_SEQUENCE_CAPACITY_DEFICIT`, NRQ-007/ADR-0167), without materially degrading
existing capability, and this effect is reproducible across five independently constructed
Core/bank models.

**First target primitive: `SORT`.** Any extension to another primitive (e.g. the `ARGUMENT_HANDLING`
`*->BIND->COUNT` failure class) requires a separate charter task; it is not implied or
pre-authorized by this charter.

### Explicitly excluded from this initial research's claims

This charter's H-D1 and any pilot result under it do **not** claim, test, or provide evidence for:
- unknown task-identity inference,
- discovery of a latent routing identity,
- transfer to an unknown/unseen relation,
- superiority of learned composition search over deterministic/exhaustive search.

Any of these would require a separately scoped and separately authorized charter or task.

## Primary measurement and failure criteria

Full numeric criteria are fixed in
`docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 8 and are
incorporated here by reference rather than duplicated: per-model, per-cell target recovery
(sequence EM >= 0.95), existing-capability preservation (<= 1pp degradation and floor retention),
Correct/Wrong-family/None causal control, Core/non-target-primitive invariance (exact hash match),
strict fresh-load parity, and full reporting of every registered run (no rescue-by-averaging).

**Failure criterion:** any floor, gate, or invariance check fails, any registered model or cell is
missing/excluded, or the derivation still needs prohibited information (oracle routing/argument
labels, ground-truth intermediate injection as a training or primary-metric signal, or a
task-conditioned Core). Mean performance across the 5-model cohort cannot override a failing
model or cell.

## Composition execution contract (prerequisite, produced by this task)

`docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` fixes, for the 8-primitive canonical
registry and the fixed depth-<=3 grammar: input domain (length/token/mask/tensor schema),
argument domain and normalization, output domain, composition conditions (shape-compatibility
derived from each `Operation.output_length`, not assumed), training exposure vs. newly-registered
valid-length extension, and the separation of `SHAPE_COMPATIBLE` / `LENGTH_SUPPORTED` /
`STANDALONE_QUALIFIED` / `COMPOSITION_QUALIFIED` as four distinct, non-substitutable states. The
normal composition execution path (`apc.primitives.composition.execute_composition_recipe`) is
preserved unmodified; satisfying the contract by injecting a ground-truth intermediate state is
explicitly disallowed.

## Target, regression, and causal-control panels (prerequisite, produced by this task)

`docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` fixes the first target panel as the 7
canonical classes NRQ-007 attributed to `SHORT_SEQUENCE_CAPACITY_DEFICIT` with `SORT` immediately
after `SELECT` (membership taken verbatim from existing artifacts, not cherry-picked), three
regression panels (standalone length-adequate SORT, 11 length-adequate SORT compositions, 8
non-SORT canary compositions), and the Correct/Wrong-family/None causal-control panel applicable
to SORT's parameter-free signature. This panel is pre-existing development knowledge used for
diagnosis, not a novel unseen relation; any claim about single-primitive-to-composition transfer
is scoped to "no end-to-end training over composed sequences was performed."

The same document also cross-checks and reconciles the 60-class/1200-cell base against the
mean-threshold 35 failure classes and the all-cell-gate 41 failure classes, and records the
6-class difference between them as `UNCLASSIFIED_BY_NRQ007` (never subjected to NRQ-007's causal
attribution, since that attribution ran only over the 35), rather than folding it into the
35-class attribution registry.

## Five-model cohort (construction/provenance contract produced by this task; cohort not built)

`docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` and the D-005 static registry
fix five pre-registered model seeds (`40,41,42,43,44`, mechanically checked against every registered
split and NRQ-005〜008 provenance record), the exact
construction procedure (the existing Model Bundle Recovery build-stage graph, run as a fresh build
rather than a restore since these are new seeds), and reuses the existing `ModelBundleManifest` /
`load_bundle` fail-closed hash contract for core/bank/token-schema/architecture-signature
provenance and strict fresh-load verification. The old 4 reconstructed bundles (seeds 1-4) are
retained as reference evidence and are explicitly **not** treated as 4/5 of this cohort (seed 0
remains lost, per ADR-0164, and is not backfilled by relabeling). This task does not construct the
cohort; a separate execution task, itself requiring authorization, is needed for that (base-model
construction consumes its own training budget, distinct from the repair budget below).

## Allowed training information and oracle boundary

Allowed: ordinary training input sequences and token-output targets for the single target
primitive (`SORT`), model outputs and gradients confined to that primitive's own parameters,
and the input-length sampling rule fixed in the pilot's section 4.2. Task information continues to
use the separate task path; `h_content = f(content)` remains invariant (Core stays frozen).

Forbidden, for the `LOCAL_SORT_REPAIR` condition, at both training and evaluation time (mechanically
guaranteed and re-verified per
`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 5.2): a task-conditioned Core,
oracle routing/argument-coordinate input (moot for SORT, which has no argument), ground-truth
intermediate injection as a training or primary-metric signal, silent substitution of the
deterministic `SYMBOLIC_REFERENCE` for the learner's own forward pass, and any update to a
non-target parameter (Core, any of the other 7 primitives, router, or argument scorer).

## Candidate handling and sealed prerequisites

A repair candidate is staged into a new namespace, never overwriting the parent bundle
(`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 6). No candidate is selected,
promoted, or adopted by this charter or by a passing pilot result — `bundle_promotion` remains
`NOT_AUTHORIZED` regardless of outcome (section 10 of the same document). No sealed data is read,
generated, or evaluated by this charter (`sealed_access: 0`); the Phase-B G1 independent-relation
deficit and sealed-partition boundary are preserved, and this charter explicitly does not evaluate
unknown-relation transfer.

## STOP and execution budget boundary

**Current budget: zero** — this task (D-001) performed only: reading existing artifacts,
re-aggregating already-recorded per-cell numbers, static semantic analysis of existing operation
definitions, and writing the documents listed in "Deliverables" below. No optimizer step, no model
initialization, no candidate construction, and no sealed access occurred.

A separately authorized execution task may run at most the one recipe fixed in
`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` — one architecture (unchanged), one
optimizer/LR/scheduler/loss/sampling combination, one step-budget ceiling (6,000 updates/model,
30,000/5-model cohort, exclusive of cohort-construction budget) — with no simultaneous sweep of
any of those dimensions. Before execution, the five-model cohort must itself be built and strict
fresh-load verified per its own construction contract, under its own separately tracked budget.

STOP applies on: any missing budget number, any unresolved dependency-hash mismatch, any
regression-panel violation, any causal-control failure, or any attempt to authorize additional
updates/seeds/architectures/other-primitive repairs from within this task. A pilot failure does
not authorize a replacement recipe, extra seeds, a larger budget, or an escalated claim about APC
in general (`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 10).

## Final decision (D-001)

**`COMPOSITION_REPAIR_CONTRACT_READY_FOR_REVIEW`.**

Every dimension the task instructions require to be numerically fixed before execution was
resolved directly from existing source and artifacts, not left as an open question:

| Dimension | Status | Where fixed |
|---|---|---|
| Source (architecture, optimizer, scheduler, loss, historical step budgets) | Resolved, cited to exact file/line | `PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` sections 2,5,7; pilot sections 3-4 |
| Input domain (valid lengths, derived not guessed) | Resolved | `PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` section 3 |
| Cohort (seeds, construction procedure, provenance fields) | Resolved as a construction *contract*; cohort itself intentionally not built in this task | `PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` |
| Evaluation (sample sizes, exhaustive-vs-sampled regime, CI procedure) | Resolved | pilot section 7 |
| Budget (repair steps/examples/params/wall-time/VRAM/RAM; parent-cohort steps separately) | Resolved; wall-time/VRAM/RAM are explicitly labeled planning-upper-bound estimates, not measurements (no timing run was executed) | pilot section 9; cohort contract section 7-8 |
| Approval | Not granted — **orthogonal to design completeness** | this document's status header |

Design and budget being complete does not itself constitute execution approval. Per this
document's own status fields (below), no new execution permission is recorded.

```
design_status       = READY_FOR_REVIEW
charter_status       = DRAFT_NOT_APPROVED   (as of D-001; superseded by D-003, see below)
training_execution   = NOT_AUTHORIZED       (as of D-001; superseded by D-003, see below)
candidate_selected   = null
bundle_promotion     = NOT_AUTHORIZED
sealed_access        = 0
```

## Authorization decision (D-003)

Task D-003 performed a bulk approval review of this charter and all D-001 deliverables and
**approved** them, per
[ADR-0170](../DECISIONS_PHASE_D.md#adr-0170-d-003-phase-d-charter-authorization-decision-scoped-approval).
Approval does not relax or re-open any dimension D-001 fixed; it only authorizes executing the
already-fixed recipe.

```
charter_status       = APPROVED
training_execution   = AUTHORIZED   # D-005 scoped: seed 40-44 cohort construction +
                                     # the single registered LOCAL_SORT_REPAIR recipe +
                                     # its registered comparison-condition/panel evaluations only
candidate_selected   = null
bundle_promotion     = NOT_AUTHORIZED
sealed_access        = 0
```

No training, cohort construction, candidate construction, or sealed-data access was performed by
D-003 itself. A separate execution task is still required to actually build the cohort and run the
recipe; that task may not deviate from any value D-001 fixed without a new authorization.

## Deliverables (Task D-001)

- Phase D charter draft: `docs/research/PHASE_D_RESEARCH_CHARTER.md` (this document)
- Composition execution contract: `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md`
- Target/regression/causal-control panel manifests: `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`, `docs/phase_d/PHASE_D_D001_PANEL_MANIFEST.json`
- SORT-only repair pilot preregistration: `docs/phase_d/PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`
- Five-model cohort construction and provenance plan: `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`
- Resource budget tables: pilot preregistration section 9; cohort contract section 7-8
- New ADR: [ADR-0169](../DECISIONS_PHASE_D.md#adr-0169-d-001-phase-d-charter-draft-composition-execution-contract-and-sort-only-repair-pilot-preregistration) and its `docs/DECISIONS.md` index entry
- Incidental correction: `docs/HARDWARE_ENVIRONMENT.md` (removed a pre-existing, now-ambiguous "Phase D" label unrelated to this charter)

### Design task status reference
- **D-001 (2026-09-13):** Produced this charter draft, the composition execution contract, the
  panel manifests, the SORT-only repair pilot preregistration, and the five-model cohort
  construction contract, per [ADR-0169](../DECISIONS_PHASE_D.md#adr-0169-d-001-phase-d-charter-draft-composition-execution-contract-and-sort-only-repair-pilot-preregistration).
  Decision: `COMPOSITION_REPAIR_CONTRACT_READY_FOR_REVIEW`. Research execution remains
  `NOT_AUTHORIZED`.
- **D-003 (2026-09-13):** Bulk approval review of all D-001 deliverables (no training, cohort
  construction, or sealed access performed), per
  [ADR-0170](../DECISIONS_PHASE_D.md#adr-0170-d-003-phase-d-charter-authorization-decision-scoped-approval).
  Decision: **`APPROVED`**, `training_execution: AUTHORIZED` scoped to seed-`30-34` cohort
  construction, the single registered `LOCAL_SORT_REPAIR` recipe, and its registered
  comparison-condition/panel evaluations only. `candidate_selected: null`,
  `bundle_promotion: NOT_AUTHORIZED`, `sealed_access: 0` unchanged. No other primitive, seed,
  recipe deviation, or Phase B/C reversal is authorized. Cohort construction and pilot execution
  remain a separate, not-yet-performed execution task.
- **D-005 (2026-09-13):** Replaced only ADR-0170's seed scope after a no-model/no-data static audit,
  per [ADR-0172](../DECISIONS_PHASE_D.md#adr-0172-d-005-phase-d-cohort-seed-amendment--complete-static-registry-audit-and-replacement-authorization). Decision: **`APPROVED`**, `training_execution: AUTHORIZED` only for seed-`40-44`; the construction procedure, recipe, panels, criteria, and budgets are unchanged. Seed-`30-34` is permanently forbidden to this charter; no result-based replacement is permitted.
- **D-006 (2026-09-13):** Precondition check detected CPU-only Torch in project-required Python 3.12 environment, per [ADR-0173](../DECISIONS_PHASE_D.md#adr-0173-d-006-execution-precondition-stop--python-312-environment-has-cpu-only-torch-and-no-constraint-compatible-cuda-wheel). Decision: `EXECUTION_PRECONDITION_STOP`.
- **D-007 (2026-09-13):** CUDA runtime recovery verified `torch 2.13.0+cu130` on RTX 5060 Ti under Python 3.12, per [ADR-0174](../DECISIONS_PHASE_D.md#adr-0174-d-007--cuda-runtime-recovery-pass-d-006-has-no-implemented-executor). Decision: `RUNTIME_GATE_PASS`.
- **D-008 (2026-09-13):** Implemented executor gate; stopped on evaluation seed aggregation and missing causal/fresh-load evidence, per [ADR-0175](../DECISIONS_PHASE_D.md#adr-0175-d-008-phase-d-executor-review-stop--incomplete-required-evaluation-evidence). Decision: `STOP_GATE_FAIL`.
- **D-009 (2026-09-14):** Implemented causal controls and separate-process fresh-load gate; stopped on occupied namespaces and full-suite test failures, per [ADR-0176](../DECISIONS_PHASE_D.md#adr-0176-d-009--required-evaluation-evidence-implemented-verification-stop-before-pilot). Decision: `STOP_GATE_FAIL`.
- **D-010 (2026-09-14):** Verified all 7 compatibility repairs clean (`2669 passed, 0 failed` in D-007 environment); execution halted pre-model-construction due to executor namespace sequencing defect creating output roots before static gate, per [ADR-0178](../DECISIONS_PHASE_D.md#adr-0178-d-010--execution-prerequisite-repair-clean-verification-and-one-time-confirmation-execution-stop-gate). Decision: **`STOP_GATE_FAIL`**, `task_result: FAIL`, `h_d1_status: UNTESTED`. Empty namespaces preserved; no model initialization or sealed access performed.
- **D-011 (2026-09-14):** Authorized replacement namespaces, repaired executor static-gate sequencing (`2670 passed, 0 failed`), and executed one-time seed-40-44 confirmation; execution halted during seed 40 parent build on device mismatch (incremental primitive on CPU), per [ADR-0179](../DECISIONS_PHASE_D.md#adr-0179-d-011--replacement-namespaces-executor-static-gate-sequencing-repair-and-one-time-confirmation-execution). Decision: **`STOP_GATE_FAIL`**, `task_result: FAIL`, `h_d1_status: UNTESTED`. D-011 artifacts preserved; no retries, seed changes, or sealed access performed.
- **D-012 (2026-09-14):** Authorized replacement namespaces, repaired incremental primitive CUDA device placement (`34 passed` focused, `9 passed` executor suite), and executed one-time seed-40-44 confirmation; seed 40 parent build completed on CUDA, but separate-process fresh-load parity check halted on child subprocess startup crash (`PYTHONHASHSEED` range error inherited from parent environment), per [ADR-0180](../DECISIONS_PHASE_D.md#adr-0180-d-012--incremental-primitive-cuda-device-placement-repair-fresh-load-subprocess-environment-defect-and-execution-stop-gate). Decision: **`STOP_GATE_FAIL`**, `task_result: FAIL`, `h_d1_status: UNTESTED`. D-012 artifacts preserved; no retries, seed changes, or sealed access performed.
- **D-013 (2026-09-14):** Authorized replacement namespaces, repaired `set_seed` and subprocess environment isolation to bound `PYTHONHASHSEED` to uint32 range (`39 passed` in 143s), and executed one-time seed-40-44 confirmation on CUDA. All 5 models completed full cohort build and `LOCAL_SORT_REPAIR` execution with exact fresh-load parity. Under preregistered acceptance criteria, `SELECT->SORT->REVERSE` collapsed to ~0 across all seeds, failing target recovery (0/5 passed), per [ADR-0181](../DECISIONS_PHASE_D.md#adr-0181-d-013--bounded-pythonhashseed-compatibility-fix-independent-subprocess-regression-test-and-execution). Decision: **`PILOT_COMPLETED_NEGATIVE_RESULT`**, `task_result: FAIL`, `h_d1_status: REFUTED`. D-013 artifacts preserved; no candidate selected, no promotion, sealed access: 0.
