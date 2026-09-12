# Phase C — Routing identifiability research charter

Date: 2026-09-13. Version: charter-v1, `READY_FOR_REVIEW_NOT_APPROVED`.
Created by PHASE-B-CLOSEOUT / ADR-0149. Architecture: `NOT_SELECTED`.
First experiment: `NOT_DEFINED`. Research execution: `NOT_AUTHORIZED`.
Status reference: Audited by C-D001 / ADR-0150 and C-D001R / ADR-0151 (`ROUTING_IDENTIFIABILITY_STOP`). Pre-execution stop; research execution remains unapproved.


## Independent question and falsifiable hypothesis

**Research question:** Can a bounded training-information contract make the
semantically correct routing identity identifiable under ambiguous token outputs,
without oracle routing supervision, and learn it consistently across independent
initializations and clean held-out relation components?

**H-C1:** One prospectively fixed, oracle-free training contract can supply
information that distinguishes semantic routing coordinates despite duplicate
target tokens, and a single design derived from that contract can satisfy the
coordinate-accuracy, execution, multi-init and relation-transfer criteria below
within a fixed finite budget. Token-output accuracy alone is insufficient evidence.

This is an independent hypothesis prompted by the
[Phase-B evidence ledger](../results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md), not
REC-004AU, recovery continuation, or a claim to repair the Phase-B terminal state.
H-C1 is untested. A finite failure rejects its registered instantiation; it does
not prove no conceivable oracle-free learning contract exists. A proof that the
permitted observations remain indistinguishable is already a valid pre-execution
STOP, without architecture implementation.

## Primary measurement and failure criteria

The primary metric is **worst-initialization, worst-relation-component semantic
routing-coordinate accuracy on collision-bearing examples**. A coordinate is
the source position required by the frozen task semantics, not merely a position
holding the same token. Compute accuracy for each preregistered valid routing
cell (task/length/output-position), macro-average those cells per relation
component, then take the minimum over components and independent initializations.
Ties or an unresolved routing identity count as incorrect. A latent representation
needs a fixed, auditable mapping to semantic coordinates before outcome inspection;
post-hoc oracle alignment or a learned oracle-label probe cannot establish H-C1.

Charter acceptance floors: primary >=0.95; every registered collision and
non-collision cell accuracy >=0.95; sequence exact match >=0.95 for every
initialization and component. Empty strata, missing valid cells, insufficient
independent components or unauditable coordinates fail the measurement
prerequisite. Report counts and uncertainty alongside the fixed point-estimate
floors; the later protocol must freeze sample sizes and confidence procedure
before data generation, and may not weaken these floors after seeing outcomes.

**Failure criterion:** Any floor or required gate fails, any registered init is
missing/excluded, or the derivation still needs prohibited information. Mean
performance cannot override a failing init/cell. A causal-control failure also
fails the claim even if output EM is high. No numerical measurement is made in
this charter task.

## Initialization and relation-transfer contract

- At least five independently drawn and preregistered primitive initializations,
  the same fixed recipe/budget and terminal selection time, with **5/5 passing**
  all criteria. Report every run, worst/mean and cell regressions. A later
  full-system claim additionally needs at least five independent Core/bank models;
  five primitives sharing one Core do not satisfy that model-cohort condition.
- Before any training, the design must be able to allocate **at least two
  independent clean relation components in validation and at least two in the
  sealed partition**, with a separate development partition. Audit the actual
  alias/coupling graph and parameter/optimizer exposure. Nominally different
  names, inverse pairs and more data seeds do not increase the component count.
- Keep targeted development regression distinct from relation-transfer evidence.
  Relations used to diagnose Phase B are exposed development knowledge, never
  fresh unseen relations. State whether the eventual claim is repair-time
  holdout or genuinely unseen-family transfer; base-training exposure limits it.
- No relation is registered, dataset generated, initialization seed selected or
  executed in this task. Feasibility of the required relation inventory remains
  an explicit prerequisite to later work, not an assumed PASS.

## Allowed training information and oracle boundary

Allowed in principle: ordinary training input sequences, token-output targets,
model outputs and gradients, input-derived position/length/duplicate statistics,
and declared model-visible task-side observations. Task information must use the
separate task path: `h_content = f(content)` remains invariant. Its exact form
and every additional non-oracle signal must be explicitly justified and fixed in
the separately approved design contract; this list grants no training permission.

Forbidden in training, initialization, sampling, weighting, optimizer design and
candidate selection: correct source-coordinate maps; oracle attention/routing
labels or gradients derived from them; teacher route trajectories; latent
operation graphs or hidden identity metadata not declared model-visible;
sealed inputs, labels or outputs; and rules tailored to known failed cells using
their oracle coordinates. A task-side identifier cannot become a hard-coded
correct routing map or an operation-specific Core/decoder solver.

Evaluator-only oracle/deterministic controls establish semantics and primitive
causality. Correct-coordinate labels may score a frozen output and diagnose its
margin direction after the forward/backward computation, but cannot feed back
into the learner, pick a seed, fit the latent-to-coordinate map, or tune a repair.
Development pass/fail gates may reject the fixed design; diagnostic labels may
not be used to adapt it within the same contract. The new metric's evaluator
privilege is not new training information.

## Required change in the information premise

Before a first experiment can even be defined, a separate design review must
derive one identifiable contract and explain which observable breaks Phase B's
collision ambiguity. Three categories remain undecided:

| Category to reason about | Obligation; no adoption here |
|---|---|
| Separate lawful training signal | Prove its provenance and that it conveys disambiguating information without encoding the correct route from oracle metadata. |
| Task/data constraints | Show how observable evidence across examples identifies the route; removing all collisions from evaluation changes H-C1 and needs a new charter. |
| Explicit latent routing treatment | Specify identifiability under allowed observations; adding a latent variable alone cannot resolve indistinguishable observations. |

These are conceptual alternatives, not experiment arms. No architecture,
optimizer, sampler, loss formula, coefficient, dataset family, seed list or pilot
schedule is selected. The failed Phase-B CE-only premise and uniform warm-start
cannot be assumed to work as a default.

## Candidate adoption and sealed prerequisites

A candidate may be considered only after the one fixed design clears every
preregistered initialization and development gate. Freeze the designated artifact
identity before outcomes; no successful-seed choice, best checkpoint search,
post-hoc promotion or splicing components from separately successful runs.
Persist a coherent candidate separately, validate in shadow, and release temporary
capacity only after validation. No Phase-B checkpoint is adopted by this charter.

Sealed evaluation requires all of the following, independently recorded:

1. Approved and committed charter, followed by a separately approved finite design
   and experiment contract; no inherited Phase-B continuation authority.
2. Identifiability and oracle/deterministic controls, causal Correct/Wrong/None
   and, when parameterized, Wrong-argument controls; no task-conditioned Core or
   decoder bypass and only selected primitives executed.
3. Coherent candidate and required independent model cohort, all-init qualification,
   immutable dependencies, strict fresh-load parity, and no hidden build fallback.
4. **RG3 equivalent:** every declared capability has qualified execution, normal
   non-SHIFT floors at least 0.95; a legacy exception cannot hide a new failure.
5. **G1 equivalent:** the independent-component counts above and proven no held-out
   gradient/update/optimizer exposure, plus disjoint data/lineage manifests.
6. **G4 equivalent:** development integration, functional safety, coverage and
   legacy regression. Preserve the parent numerical safety floors from
   [Post-D2 acceptance section 8](../EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md#8-g4--development統合回帰)
   (including nominal EM >=0.95, coverage >=0.97, unsafe acceptance <=0.01 and
   legacy degradation <=1 pp; where SHIFT is in scope, mean EM >=0.99 and every
   model REF_ADEQUATE), with operation/routing criteria retained where
   applicable. If those criteria cannot map to the future scope, resolve that
   explicitly before implementation; do not silently waive a gate.
7. Separately frozen candidate hashes, relation membership, data budgets, controls,
   analysis and one-shot sealed protocol analogous to R3-011; explicit permission
   for the sealed evaluation itself. Any opened partition is retired from tuning.

The overall question is not answered by a collision metric alone. Independent
relation transfer and system safety remain distinct gates; none is marked PASS
here. Phase-B G1/G4/G5 and its RG3 recheck remain stopped/blocked/unexecuted.

## STOP and architecture/search budget boundary

**Current budget: zero** architecture/optimizer implementation, training updates,
candidate construction, dataset generation, relation registration, seed execution,
pilot, GPU experiment time and sealed access. This task produces only the charter
and closeout evidence. It does not define the first experiment.

After charter approval, a separate task may propose at most **one** architecture
and training formulation from the information argument, with one fixed recipe
and no architecture, seed, coefficient or optimizer sweep. Before execution is
authorized it must freeze numerical limits for parameters (resident/active/
temporary), data, steps, seeds, wall time and memory on the single 16-GB GPU /
64-GB RAM workstation. Missing budgets mean STOP, not discretion to explore.

STOP on non-identifiability, inadequate relation inventory, exposure/invariant
violation, missing/corrupt provenance, any primary/gate failure, or budget
exhaustion. Preserve evidence and append an ADR. A failure does not authorize a
replacement candidate, extra seeds, larger budget, revised metric or another
pilot. Any new hypothesis requires a separate review and explicit authorization;
it is not a retry loop in this charter. Pretrained LMs, RL, distributed training,
custom kernels and open-ended architecture search are outside this scope.

## Review readiness

Research question, falsifiable hypothesis, metrics/failure criteria, all-init rule,
information boundary, relation requirement, adoption rule, sealed prerequisites,
STOP and finite search boundary are defined. Design feasibility is **unproved**;
the charter is ready for approval, while implementation and the first experiment
remain undefined and unstarted. Acceptance of this document as a deliverable
does not itself approve the future research phase.

### Design task status reference
- **C-D001 Audit (2026-09-13):** [Identifiability and feasibility review](../phase_c/PHASE_C_C_D001_IDENTIFIABILITY_AND_FEASIBILITY_REVIEW.md) and [ADR-0150](../DECISIONS_PHASE_C.md#adr-0150-c-d001-oracle-free-routing-identifiability-contract-derivation--relation-inventory-feasibility-audit-stops-on-relation-count-sufficiency-relation_inventory_feasibility_stop) derived `Phase C Training Information Contract v1` but recorded `RELATION_INVENTORY_FEASIBILITY_STOP` due to validation (1/2) and sealed (1/2) clean component deficits in existing non-sealed metadata. Research execution remains `NOT_AUTHORIZED`.
- **C-D001R Adversarial Falsification Audit (2026-09-13):** [Adversarial identifiability audit](../phase_c/PHASE_C_C_D001R_ADVERSARIAL_IDENTIFIABILITY_FALSIFICATION_AUDIT.md) and [ADR-0151](../DECISIONS_PHASE_C.md#adr-0151-c-d001r-adversarial-cross-examplecross-relation-identifiability-falsification-audit-corrects-stop-to-routing_identifiability_stop) constructed identical-training-history adversarial worlds (World A vs World B) and audited five contract dimensions, mathematically falsifying Contract v1; formally corrected the pre-execution stoppage rationale to `ROUTING_IDENTIFIABILITY_STOP`. Research execution remains `NOT_AUTHORIZED`.
- **C-D001S Quantifier-Complete Boundary Audit (2026-09-13):** [Quantifier-complete boundary audit](../phase_c/PHASE_C_C_D001S_QUANTIFIER_COMPLETE_IDENTIFIABILITY_AUDIT.md) and [ADR-0152](../DECISIONS_PHASE_C.md#adr-0152-c-d001s-quantifier-complete-task-sidesupport-identifiability-boundary-audit-confirms-routing_identifiability_stop-across-permitted-observable-space) evaluated all permitted task-side observables (opaque ID, compositional descriptor, finite output support set); proved counterexamples hold across all classifications; showed that bounded separating support sets cannot resolve intensional routing under identical token outputs without hard-coding oracle maps; confirmed `ROUTING_IDENTIFIABILITY_STOP` as a quantifier-complete impossibility theorem for H-C1. Research execution remains `NOT_AUTHORIZED`.


