# Task Queue

## Current queue — 2026-09-13

Phase B: `CLOSED_ARCHIVED`, `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`
(ADR-0148/0149). **Active Phase-B experimental tasks: none.**
The [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state)
is authoritative. RG3, REC-005--008, R3-011/012 and B-C006 onward are archived
non-executions due to upstream STOP; do not register them as unfinished work.

PHASE-B-CLOSEOUT is complete: `PHASE_B_CLOSED_NEXT_RESEARCH_CHARTER_READY`.
The [Phase C charter](research/PHASE_C_RESEARCH_CHARTER.md) is ready for user
review/approval.
Task C-D001 (Identifiability Contract Derivation & Relation-Inventory Feasibility Audit) is complete: `RELATION_INVENTORY_FEASIBILITY_STOP` ([ADR-0150](DECISIONS_PHASE_C.md#adr-0150-c-d001-oracle-free-routing-identifiability-contract-derivation--relation-inventory-feasibility-audit-stops-on-relation-count-sufficiency-relation_inventory_feasibility_stop)).
Task C-D001R (Adversarial Cross-Example/Cross-Relation Identifiability Falsification Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP` ([ADR-0151](DECISIONS_PHASE_C.md#adr-0151-c-d001r-adversarial-cross-examplecross-relation-identifiability-falsification-audit-corrects-stop-to-routing_identifiability_stop)). Mathematical construction of identical-training-history adversarial worlds (World A vs World B) and audit of five core contract dimensions falsified Contract v1; stoppage rationale formally corrected from relation inventory deficit to fundamental identifiability failure. Research execution remains `NOT_AUTHORIZED`.
Task C-D001S (Quantifier-Complete Task-Side/Support Identifiability Boundary Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP (CONFIRMED_QUANTIFIER_COMPLETE)` ([ADR-0152](DECISIONS_PHASE_C.md#adr-0152-c-d001s-quantifier-complete-task-sidesupport-identifiability-boundary-audit-confirms-routing_identifiability_stop-across-permitted-observable-space)). Evaluated all permitted task-side observables (opaque ID, compositional descriptor, finite output support set) and proved counterexamples hold across all classifications; pre-fixed separating support sets cannot resolve intensional routing coordinates on duplicate tokens without hard-coding oracle maps; confirms `ROUTING_IDENTIFIABILITY_STOP` as a quantifier-complete impossibility theorem for H-C1. Research execution remains `NOT_AUTHORIZED`.
Task C-D001T (Semantic-Descriptor Boundary & Impossibility-Proof Repair Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0153](DECISIONS_PHASE_C.md#adr-0153-c-d001t-semantic-descriptor-boundary--impossibility-proof-repair-audit-qualifies-stop-to-routing_identifiability_qualified_stop-adr-0152-quantifier-retracted--restricted)). Specified formal language of permitted compositional descriptors and oracle leakage rules; repaired ADR-0151/ADR-0152 circular definitions using actual APC operations (BindOp, NeighborMaxOp); proved lawful descriptors equipped with general tie-break semantics separate adversarial worlds (Theorem 3); retracted and restricted ADR-0152's quantifier-complete impossibility claim; reclassified stop to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`. Research execution remains `NOT_AUTHORIZED`.
Task C-D001U (Semantic Descriptor Oracle-Equivalence & Minimality Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP (RECONFIRMED)` ([ADR-0154](DECISIONS_PHASE_C.md#adr-0154-c-d001u-semantic-descriptor-oracle-equivalence--minimality-audit-reconfirms-routing_identifiability_stop)). Formalized deterministic $z$-computable reduction mapping $(x, D) \mapsto z^*$; determined that candidate-selection tie-break procedures are oracle-equivalent supervision under Charter H-C1; constructed minimality counterexamples proving lawful non-oracle descriptors cannot separate World A/B on duplicate tokens; rejected Training Information Contract v1.1; definitively reconfirmed `ROUTING_IDENTIFIABILITY_STOP` prior to architecture derivation. Research execution remains `NOT_AUTHORIZED`.
Task C-D001V (Non-Circular Oracle-Equivalence Falsification Experiment) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0155](DECISIONS_PHASE_C.md#adr-0155-c-d001v-non-circular-oracle-equivalence-falsification-experiment-retracts-adr-0154-universal-stop-to-routing_identifiability_qualified_stop)). Established 5-dimensional non-circular oracle-supervision criteria calibrated against negative/positive controls; proved `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, and procedural composition are strictly non-oracle (0/5 oracle score); proved they separate World A/B on collision tokens; mathematically falsified and retracted ADR-0154's universal STOP; reclassified stoppage to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`. Research execution remains `NOT_AUTHORIZED`.
Task C-D001W (Adversarial Mixed-Control Validation of the Five-Dimensional Oracle Criterion) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0156](DECISIONS_PHASE_C.md#adr-0156-c-d001w-adversarial-mixed-control-validation-confirms-adr-0155-05-non-oracle-determination-and-derives-training-information-contract-v11--deterministic-baseline)). Evaluated adversarial mixed controls (bytecode VM, encrypted lookups, metadata, static routing programs, support selectors) under ADR-0155's 5-dimensional criterion; confirmed strict representation invariance, monotonicity, leave-one-out necessity, and fail-closed optimality of the disjunctive rule (theta=1); upheld the 0/5 non-oracle classification of FIRST/LAST; derived candidate Training Information Contract v1.1 and descriptor-only deterministic baseline requirements. Research execution remains `NOT_AUTHORIZED`.
Task C-D001X (Contract-v1.1 Hypothesis-Preservation & Deterministic-Baseline Dominance Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0157](DECISIONS_PHASE_C.md#adr-0157-c-d001x-contract-v11-hypothesis-preservation--deterministic-baseline-dominance-audit-finds-h-c1-estimand-trivialized-by-unlearned-deterministic-reduction)). Deconstructed H-C1's estimand; proved descriptor-only deterministic baseline $B_{\text{det}}$ satisfies 100% of routing and execution floors with 0 learning and $H(Z \mid X, D) = 0.0$; executed tie-break masking, swap, and permutation ablations; determined Contract v1.1 trivializes H-C1 and alters the estimand to neural emulation; recommends Charter retraction or restriction to residual continuous grounding (`H_C1_TRIVIALIZED_ESTIMAND_ALTERED_BY_CONTRACT_V1_1`). Research execution remains `NOT_AUTHORIZED`.
Task C-D001Y (Embedding-Aware Deterministic Baseline Closure Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0158](DECISIONS_PHASE_C.md#adr-0158-c-d001y-embedding-aware-deterministic-baseline-closure-audit-finds-continuous-neural-grounding-dominated-by-b_det_emb-under-invertible-representations)). Defined Embedding-Aware Deterministic Baseline ($B_{\text{det\_emb}}$); proved $B_{\text{det\_emb}}$ achieves 100% routing and execution ceiling with 0 learning under invertible representations (Controls 1, 2, 5); proved performance drops under lossy/colliding representations (Controls 3, 4) are an identifiability limit rather than a learnability limit; determined continuous neural grounding is not a non-trivial residual estimand; recommends further Charter retraction or restriction to blind manifold discovery (`CONTINUOUS_GROUNDING_DOMINATED_BY_B_DET_EMB_UNDER_INVERTIBLE_REPRESENTATIONS`). Research execution remains `NOT_AUTHORIZED`.








## Historical bootstrap queue

The original issue prompts below are historical reference, not current execution authority.

## Task 001 — Bootstrap the repository

**Goal:** Create the Python package skeleton and development tooling described in `README.md` and `AGENTS.md`.

**Required files:**
- `pyproject.toml`;
- `src/apc/...` package directories;
- `tests/`;
- `configs/`;
- `scripts/`;
- `.gitignore`.

**Constraints:**
- Python 3.12;
- PyTorch only as required ML framework;
- pytest, ruff, mypy for dev tooling;
- no model implementation yet beyond an import smoke test.

**Acceptance:** documented verification commands pass.

---

## Task 002 — Implement deterministic symbolic task engine

**Goal:** Implement the reference interpreter and data generator for the Phase A known operations.

Read `docs/exec-plans/active/PHASE_A.md`, Milestone A1.

**Acceptance:**
- deterministic by seed;
- generated sample includes latent operation graph metadata;
- interpreter tests cover every primitive operation;
- no neural-model dependency.

---

## Task 003 — Implement fixed dense baseline

**Goal:** Add the smallest Transformer baseline that can learn initial synthetic tasks.

**Constraints:**
- configurable model size;
- CPU smoke configuration;
- CUDA mixed precision only inside training code;
- checkpoint + metrics output.

**Acceptance:** smoke training overfits a tiny deterministic dataset; reload reproduces outputs.

---

## Task 004 — Implement low-rank primitive module and bank

**Goal:** Add `Primitive`, `PrimitiveBank`, accounting, status and usage metrics.

Do not add dynamic expansion yet.

**Acceptance:** tests prove total/persistent/active counts and freeze behavior.

---

## Task 005 — Implement top-k primitive router

**Goal:** Route a hidden state to at most k primitives and expose scores/entropy/IDs.

**Acceptance:** deterministic tests for top-k; gradients flow only through intended selected path/weights according to implementation choice; usage logging works.

---

## Task 006 — Build the composition benchmark

**Goal:** Define train and held-out composition splits of known operations.

**Acceptance:** evaluator can label examples K/C; model/controller does not see these oracle labels; benchmark report includes composition generalization.

---

## Task 007 — Implement plastic workspace

**Goal:** Add temporary low-rank capacity and an allocator with small/medium/large presets.

**Acceptance:** temporary parameter count changes without changing persistent count; frozen persistent weights remain unchanged after a plastic training step.

---

## Task 008 — Implement novelty signals and state controller

**Goal:** Implement error + router entropy first, finite-state transitions and hysteresis.

**Acceptance:** transition log includes trigger metrics and capacity changes; unit tests cover no-oscillation behavior.

Do not implement RL.

---

## Task 009 — Add novel-operation task family

**Goal:** Add at least one operation not expressible under the configured known-composition budget.

**Acceptance:** static composition baseline fails materially more often than on held-out compositions; oracle metadata marks N but remains hidden from controller.

---

## Task 010 — Implement first consolidation path

**Goal:** Distill active temporary transforms into lower-rank candidate primitive(s).

**Acceptance:** candidate is smaller; reports task imitation score and prior-task replay score; temporary module is not deleted.

---

## Task 011 — Implement shadow validation and release

**Goal:** Compare candidate vs temporary solution and release temporary capacity only after configured thresholds pass.

**Acceptance:** failing shadow test preserves temporary capacity; passing test installs candidate and releases temporary capacity atomically.

---

## Task 012 — Sequential Phase A benchmark

**Goal:** Run at least two learn/consolidate/release cycles in one task stream.

**Acceptance:** produce the metrics and plots required by `docs/EXPERIMENT_PLAN.md`; no unsupported research claims in the report.

---

## Task 013 — Add baselines

Implement B0–B4 from `docs/EXPERIMENT_PLAN.md` using shared data, evaluation and budgets where possible.

---

## Task 014 — Phase A review

Do not add features. Audit:
- reproducibility;
- capacity accounting;
- result validity;
- baseline fairness;
- failure cases;
- documentation drift.

Write `docs/exec-plans/completed/PHASE_A_RESULT.md` only after measured results exist.
