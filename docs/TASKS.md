# Task Queue

## Current queue — 2026-09-13

Phase B: `CLOSED_ARCHIVED`, `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`
(ADR-0148/0149). **Active Phase-B experimental tasks: none.**
The [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state)
is authoritative. RG3, REC-005--008, R3-011/012 and B-C006 onward are archived
non-executions due to upstream STOP; do not register them as unfinished work.

PHASE-B-CLOSEOUT is complete: `PHASE_B_CLOSED_NEXT_RESEARCH_CHARTER_READY`.
The [Phase C charter](research/PHASE_C_RESEARCH_CHARTER.md) is `TERMINATED_CURRENT_CHARTER`
(never approved, never executed experimentally, falsified/retracted entirely during
pre-execution mathematical review). **Active Phase-C tasks: none.**
Task C-D001 (Identifiability Contract Derivation & Relation-Inventory Feasibility Audit) is complete: `RELATION_INVENTORY_FEASIBILITY_STOP` ([ADR-0150](DECISIONS_PHASE_C.md#adr-0150-c-d001-oracle-free-routing-identifiability-contract-derivation--relation-inventory-feasibility-audit-stops-on-relation-count-sufficiency-relation_inventory_feasibility_stop)).
Task C-D001R (Adversarial Cross-Example/Cross-Relation Identifiability Falsification Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP` ([ADR-0151](DECISIONS_PHASE_C.md#adr-0151-c-d001r-adversarial-cross-examplecross-relation-identifiability-falsification-audit-corrects-stop-to-routing_identifiability_stop)). Mathematical construction of identical-training-history adversarial worlds (World A vs World B) and audit of five core contract dimensions falsified Contract v1; stoppage rationale formally corrected from relation inventory deficit to fundamental identifiability failure. Research execution remains `NOT_AUTHORIZED`.
Task C-D001S (Quantifier-Complete Task-Side/Support Identifiability Boundary Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP (CONFIRMED_QUANTIFIER_COMPLETE)` ([ADR-0152](DECISIONS_PHASE_C.md#adr-0152-c-d001s-quantifier-complete-task-sidesupport-identifiability-boundary-audit-confirms-routing_identifiability_stop-across-permitted-observable-space)). Evaluated all permitted task-side observables (opaque ID, compositional descriptor, finite output support set) and proved counterexamples hold across all classifications; pre-fixed separating support sets cannot resolve intensional routing coordinates on duplicate tokens without hard-coding oracle maps; confirms `ROUTING_IDENTIFIABILITY_STOP` as a quantifier-complete impossibility theorem for H-C1. Research execution remains `NOT_AUTHORIZED`.
Task C-D001T (Semantic-Descriptor Boundary & Impossibility-Proof Repair Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0153](DECISIONS_PHASE_C.md#adr-0153-c-d001t-semantic-descriptor-boundary--impossibility-proof-repair-audit-qualifies-stop-to-routing_identifiability_qualified_stop-adr-0152-quantifier-retracted--restricted)). Specified formal language of permitted compositional descriptors and oracle leakage rules; repaired ADR-0151/ADR-0152 circular definitions using actual APC operations (BindOp, NeighborMaxOp); proved lawful descriptors equipped with general tie-break semantics separate adversarial worlds (Theorem 3); retracted and restricted ADR-0152's quantifier-complete impossibility claim; reclassified stop to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`. Research execution remains `NOT_AUTHORIZED`.
Task C-D001U (Semantic Descriptor Oracle-Equivalence & Minimality Audit) is complete: `ROUTING_IDENTIFIABILITY_STOP (RECONFIRMED)` ([ADR-0154](DECISIONS_PHASE_C.md#adr-0154-c-d001u-semantic-descriptor-oracle-equivalence--minimality-audit-reconfirms-routing_identifiability_stop)). Formalized deterministic $z$-computable reduction mapping $(x, D) \mapsto z^*$; determined that candidate-selection tie-break procedures are oracle-equivalent supervision under Charter H-C1; constructed minimality counterexamples proving lawful non-oracle descriptors cannot separate World A/B on duplicate tokens; rejected Training Information Contract v1.1; definitively reconfirmed `ROUTING_IDENTIFIABILITY_STOP` prior to architecture derivation. Research execution remains `NOT_AUTHORIZED`.
Task C-D001V (Non-Circular Oracle-Equivalence Falsification Experiment) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0155](DECISIONS_PHASE_C.md#adr-0155-c-d001v-non-circular-oracle-equivalence-falsification-experiment-retracts-adr-0154-universal-stop-to-routing_identifiability_qualified_stop)). Established 5-dimensional non-circular oracle-supervision criteria calibrated against negative/positive controls; proved `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, and procedural composition are strictly non-oracle (0/5 oracle score); proved they separate World A/B on collision tokens; mathematically falsified and retracted ADR-0154's universal STOP; reclassified stoppage to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`. Research execution remains `NOT_AUTHORIZED`.
Task C-D001W (Adversarial Mixed-Control Validation of the Five-Dimensional Oracle Criterion) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0156](DECISIONS_PHASE_C.md#adr-0156-c-d001w-adversarial-mixed-control-validation-confirms-adr-0155-05-non-oracle-determination-and-derives-training-information-contract-v11--deterministic-baseline)). Evaluated adversarial mixed controls (bytecode VM, encrypted lookups, metadata, static routing programs, support selectors) under ADR-0155's 5-dimensional criterion; confirmed strict representation invariance, monotonicity, leave-one-out necessity, and fail-closed optimality of the disjunctive rule (theta=1); upheld the 0/5 non-oracle classification of FIRST/LAST; derived candidate Training Information Contract v1.1 and descriptor-only deterministic baseline requirements. Research execution remains `NOT_AUTHORIZED`.
Task C-D001X (Contract-v1.1 Hypothesis-Preservation & Deterministic-Baseline Dominance Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0157](DECISIONS_PHASE_C.md#adr-0157-c-d001x-contract-v11-hypothesis-preservation--deterministic-baseline-dominance-audit-finds-h-c1-estimand-trivialized-by-unlearned-deterministic-reduction)). Deconstructed H-C1's estimand; proved descriptor-only deterministic baseline $B_{\text{det}}$ satisfies 100% of routing and execution floors with 0 learning and $H(Z \mid X, D) = 0.0$; executed tie-break masking, swap, and permutation ablations; determined Contract v1.1 trivializes H-C1 and alters the estimand to neural emulation; recommends Charter retraction or restriction to residual continuous grounding (`H_C1_TRIVIALIZED_ESTIMAND_ALTERED_BY_CONTRACT_V1_1`). Research execution remains `NOT_AUTHORIZED`.
Task C-D001Y (Embedding-Aware Deterministic Baseline Closure Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0158](DECISIONS_PHASE_C.md#adr-0158-c-d001y-embedding-aware-deterministic-baseline-closure-audit-finds-continuous-neural-grounding-dominated-by-b_det_emb-under-invertible-representations)). Defined Embedding-Aware Deterministic Baseline ($B_{\text{det\_emb}}$); proved $B_{\text{det\_emb}}$ achieves 100% routing and execution ceiling with 0 learning under invertible representations (Controls 1, 2, 5); proved performance drops under lossy/colliding representations (Controls 3, 4) are an identifiability limit rather than a learnability limit; determined continuous neural grounding is not a non-trivial residual estimand; recommends further Charter retraction or restriction to blind manifold discovery (`CONTINUOUS_GROUNDING_DOMINATED_BY_B_DET_EMB_UNDER_INVERTIBLE_REPRESENTATIONS`). Research execution remains `NOT_AUTHORIZED`.
Task C-D001Z (Blind Codebook Identifiability & Symmetry-Breaking Audit) is complete: `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0159](DECISIONS_PHASE_C.md#adr-0159-c-d001z-blind-codebook-identifiability--symmetry-breaking-audit-proves-blind-grounding-impossible-and-retracts-residual-charter-candidate)). Formalized permutation ($S_V$) and orthogonal ($O(d)$) group actions on unknown codebooks; proved blind grounding is an identifiability impossibility ($H(Z \mid \text{obs}) > 0$) via indistinguishable symmetric worlds ($H_A=H_B, D_A=D_B, y_A=y_B$) with diverging routing coordinates ($z^*_A \ne z^*_B$); evaluated four pre-registered controls proving unanchored configurations fail floors while complete anchors trigger deterministic baseline dominance ($B_{\text{det\_emb}} = 1.000$); proved via the Dilemma Theorem that the non-trivial residual learning space is empty; formally retracted residual Charter candidate `H-C1-Residual` (`BLIND_GROUNDING_IDENTIFIABILITY_IMPOSSIBILITY_PROVEN`). Research execution remains `NOT_AUTHORIZED`.
Task C-D001AA (Phase C Falsification Sufficiency & Charter Termination Audit) is complete: `PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT` ([ADR-0160](DECISIONS_PHASE_C.md#adr-0160-c-d001aa-phase-c-falsification-sufficiency--charter-termination-audit-closes-the-current-charter-phase_c_current_charter_falsification_sufficient)). Audited ADR-0150-0159 for falsification sufficiency without designing any new repair, architecture, or residual hypothesis; confirmed the fourteen-field current state; classified each ADR CURRENT/SUPERSEDED/RETRACTED (ADR-0152 and ADR-0154 retracted by their successors); clarified that descriptor-based identifiability replaced rather than validated H-C1's estimand; scoped the impossibility claim strictly to the examined representation/group-symmetry classes (explicitly not APC-general impossibility); confirmed the non-trivial residual set is empty; confirmed the relation-inventory deficit as an independent, unremedied FAIL; found no in-charter conclusion-changing task remains; and terminated the charter. **Phase C charter status: `TERMINATED_CURRENT_CHARTER`.** See [termination evidence ledger](results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md). **No Phase C task is queued.** Any future research question requires a new, independently authorized charter (see charter §"Phase C — Routing identifiability research charter").

Task NRQ-001 (Next-Research-Question Review: Non-Trivial Identifiable Estimand Existence Test) is complete: `NO_NONTRIVIAL_ESTIMAND_IDENTIFIED` ([ADR-0161](DECISIONS_PHASE_C.md#adr-0161-nrq-001-next-research-question-review-finds-no-non-trivial-identifiable-estimand-no_nontrivial_estimand_identified)). Not a Phase C task. Generalized ADR-0159's Impossibility-Dominance Dilemma to a representation-agnostic dichotomy covering nonlinear and interactive reformulations; surveyed six candidate estimands (stochastic ground truth, interactive query, sample-efficiency-of-induction, failure-containment, compute-scaling, plus the generalized-dichotomy class) and found none admissible against the five required criteria (non-trivial, identifiable, oracle-free, relation-transfer compatible, actually tests core separation); re-confirmed the relation-inventory deficit (1/2 validation, 1/2 sealed) via two independent audits (ADR-0147, ADR-0150); confirmed `PROGRAM_LINE_CLOSURE_CONFIRMED` for the oracle-free task/relation-inference research line (the Phase B -> Phase C throughline), explicitly not a claim of APC's general impossibility and not a re-opening of Phase A/A.1/A.2's positive core-separation evidence. No minimal experiment is pre-registered (none was admissible). See the [review document](research/NEXT_RESEARCH_QUESTION_REVIEW_NRQ001.md). **No task is queued.** A future research question requires a new, differently-scoped, explicitly user-authorized premise.

Task NRQ-002 (Constructive Falsification Experiment for the Lawful-Disambiguation Dichotomy) is complete: `NO_COUNTEREXAMPLE_CONSTRUCTED` ([ADR-0162](DECISIONS_PHASE_C.md#adr-0162-nrq-002-constructive-falsification-experiment-finds-no-counterexample-to-the-lawful-disambiguation-dichotomy-no_counterexample_constructed)). Not a Phase C task. Actively attempted to construct a counterexample to NRQ-001's dichotomy, targeting the "one honest loophole" (tractability of the disambiguating function under finite composition) NRQ-001 left open for arbitrary compositions; tried four constructions (recipe-space combinatorics at tested scale, recipe-space growth past any fixed budget, argument-space combinatorics, a deliberately hard/cryptographic-style composition target) and closed each one -- Attempt A empirically, via Task A1-B004's already-measured 99.62%-EM oracle-free beam-search baseline on this exact construction; Attempts B/C via this project's own measured bank/argument-domain scale (N<=128, argument domains <=~100 joint combinations); Attempt D on the same scope/inventory grounds as NRQ-001's candidate N2 -- and derived a Bounded-Resource Corollary generalizing the closure to any finite composition of the current registry at in-scope depth/domain size. Reinforces (does not reopen) `PROGRAM_LINE_CLOSURE_CONFIRMED`; the relation-inventory deficit (1/2 validation, 1/2 sealed) is unchanged. See the [review document](research/CONSTRUCTIVE_FALSIFICATION_NRQ002.md). **No task is queued.**

Task NRQ-003 (Exact-Depth-3 Irreducible Composition Search Benchmark: Prerequisite Audit & Model Adequacy Gate) is complete: `BLOCKED_BY_MODEL_ADEQUACY` ([ADR-0163](DECISIONS_PHASE_C.md#adr-0163-nrq-003-exact-depth-3-composition-search-prerequisite-check-blocked-by-bundle-provenance-and-model-inadequacy-blocked_by_model_adequacy)). Audited frozen core and primitive-bank bundle provenance and evaluated Task A1-B004 depth-2 positive controls across 6 canonical recipes. Found broken bundle provenance: seeds 1-4 possess token_emb shape [36, 192], incompatible with current vocab_size=44, raising `RuntimeError: size mismatch` upon loading; seed 0 possesses shape [44, 192] but exhibits latent space de-synchronization with primitive_bank.pt, causing depth-2 positive controls to collapse to 11.8% mean oracle exact match, 18.3% recovered exact match, and 40.5% functional agreement (0/6 passed). Following the mandatory prerequisite contract ("require the depth-2 positive controls to reproduce; if that prerequisite fails, record NRQ-003 as invalid/blocked by model adequacy without interpreting depth-3 search"), execution and interpretation of depth-3 search are strictly barred. See the [review document](research/EXACT_DEPTH3_COMPOSITION_SEARCH_AUDIT_NRQ003.md).

Task NRQ-004 (Frozen Bundle Compatibility Reconstruction & Depth-2 Control Reproduction) is complete: `STOP_NRQ003_BLOCKED` ([ADR-0164](DECISIONS_PHASE_C.md#adr-0164-nrq-004-frozen-bundle-compatibility-reconstruction--depth-2-control-reproduction-fails-on-bundle-loss-stop_nrq003_blocked)). Audited git history at commit `c3f291f`, run manifests, and checkpoint hashes. Reconstructed native Phase A.1 36-token vocabulary schema (`SharedCoreTokens(num_operations=10, arg_span=10)`) into a non-destructive bundle namespace (`runs/nrq004_reconstructed_bundles/`). Resolved schema drift for seeds 1–4, achieving near-ceiling control reproduction (mean recovered EM = 99.50%, functional agreement = 99.96%, passing 6/6), decisively refuting intrinsic model inadequacy. Confirmed Seed 0 failure is exclusively caused by bundle loss (checkpoint overwrite on 2026-09-13). Enforced fail-closed stop gate (`STOP_NRQ003_BLOCKED`): because all 5 seeds are required for NRQ-003 resumption and unauthorized retraining is prohibited, execution halts without proceeding to training or depth-3 search. See the [review document](research/FROZEN_BUNDLE_COMPATIBILITY_RECONSTRUCTION_NRQ004.md). **No task is queued.**











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
