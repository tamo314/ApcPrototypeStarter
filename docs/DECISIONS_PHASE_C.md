# Phase C Decision Ledger

This document records architectural, contractual, and scientific decisions for Phase C: Routing Identifiability Research.
Historical Phase A and Phase B decisions remain preserved in their respective files (`DECISIONS_PHASE_A.md`, `DECISIONS_PHASE_B_*.md`).

---

## ADR-0150: C-D001 Oracle-Free Routing Identifiability Contract Derivation & Relation-Inventory Feasibility Audit Stops on Relation-Count Sufficiency (`RELATION_INVENTORY_FEASIBILITY_STOP`)

**Date:** 2026-09-13  
**Task:** C-D001 — Oracle-Free Routing Identifiability Contract Derivation & Relation-Inventory Feasibility Audit  
**Status:** Completed non-experimental design & audit; `execution_status: PASS`, `decision: RELATION_INVENTORY_FEASIBILITY_STOP`.  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Read only the prospective Phase C Research Charter, historical ADRs (ADR-0074 through ADR-0149), and non-sealed relation manifests (`runs/phase_b_b2_post_d2/r3_002_relation_split/relation_split.json`, `runs/phase_b_b2_post_d2/r3_002r_single_family_g1/run_002/relation_coupling_graph.json`, and codebase registry definitions).
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Charter Approval State:**  
   The Phase C Research Charter was published under ADR-0149 as `READY_FOR_REVIEW_NOT_APPROVED` with `Research execution: NOT_AUTHORIZED`. No subsequent approval record exists in git history. In accordance with Section 0 of the task contract, charter integrity was reviewed and an approval proposal recorded, but research execution authority was not inferred.
2. **Duplicate-Token Collision Formalization:**  
   Generalized Phase B's token-alias failure into an observable identifiability framework $(x, y_k, z_k^*, t, \mathcal{O})$. Proved that in an isolated collision cell where $x_i = x_j = y_k$, token cross-entropy loss provides symmetric gradients to $i$ and $j$, making single-example token observations information-theoretically unidentifiable ($\mathcal{O}(x, y_k, t, z_1) = \mathcal{O}(x, y_k, t, z_2)$).
3. **Information Contract Derivation (`Phase C Training Information Contract v1`):**  
   Deductively converged on a single oracle-free training information contract without candidate sweeps. Identifiability is achieved via:
   - Content-decoupled positional routing geometry $g_\theta(j, k, L, t)$, preserving $h_{\text{content}} = f(\text{content})$.
   - Cross-example structural task consistency: learning routing rules from clean, collision-free instances ($x_m$ all distinct).
   - Duplicate-aware gradient masking: setting routing loss gradients to zero on duplicate-bearing cells during training to eliminate destructive alias gradients.
4. **Oracle Leakage Audit:**  
   Confirmed zero leakage: ground-truth coordinates, teacher trajectories, and historical cell overrides (e.g. key 0 / key 7 patches) are strictly excluded. The task identifier $t$ provides relation identity via the separate task path without encoding routing maps.
5. **Relation Inventory Feasibility Audit:**  
   Phase C Charter requires at least 2 independent clean relation components in validation and at least 2 in sealed before training starts. Audit of non-sealed registries and coupling graphs showed:
   - Development has 1 component (`L3:BIND-COUNT+BIND-SELECT`, contaminated by Phase B development exposure).
   - Validation has 1 component (`L3:CYCLE_FOUR-SHIFT`).
   - Sealed has 1 component (`family:sealed_local_neighborhood`, coupled internally).
   - Validation (1/2) and Sealed (1/2) both fail the sufficiency floor (total available clean components = 2; required = 4).

**Decision:**  
Declare **`RELATION_INVENTORY_FEASIBILITY_STOP`**.

**Consequences:**  
- While the mathematical information contract is proven and derived, research execution cannot proceed to task `C-D002 — Single Architecture & Training Formulation Derivation` because the necessary independent relation inventory does not exist.
- Per Section 10, creating synthetic relations or renaming aliases within this task is prohibited. A new relation family expansion requires an independent research decision and charter amendment.
- Architecture selection, optimizer creation, and training execution remain strictly blocked (`NOT_AUTHORIZED`).

**Primary Artifacts:**  
- Review Document: `docs/phase_c/PHASE_C_C_D001_IDENTIFIABILITY_AND_FEASIBILITY_REVIEW.md`
- Observable Matrix: `docs/phase_c/artifacts/observable_provenance_matrix.json`
- Feasibility Audit: `docs/phase_c/artifacts/relation_inventory_feasibility_audit.json`

---

## ADR-0151: C-D001R Adversarial Cross-Example/Cross-Relation Identifiability Falsification Audit Corrects Stop to `ROUTING_IDENTIFIABILITY_STOP`

**Date:** 2026-09-13  
**Task:** C-D001R — Adversarial Cross-Example/Cross-Relation Identifiability Falsification Audit  
**Status:** Completed non-experimental adversarial audit; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_STOP` (Corrected from `RELATION_INVENTORY_FEASIBILITY_STOP`).  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated Contract v1 (ADR-0150), Phase C Research Charter, and adversarial counterexample constructions across collision cells and held-out relations.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Intra-Relation Collision-Cell Counterexample (World A vs World B):**  
   Constructed two distinct ground-truth semantic environments under a single task $t$: World A (`first_occurrence: z^*_A = min { i | x_i == y_k }`) and World B (`last_occurrence: z^*_B = max { i | x_i == y_k }`). Under Contract v1's duplicate-aware gradient masking, all duplicate cells receive zero routing gradients, while clean instances produce bitwise identical inputs, targets, losses, and gradients ($\mathcal{H}_{\text{train}}(\text{World A}) \equiv \mathcal{H}_{\text{train}}(\text{World B})$). On collision-bearing test inputs (e.g. $x=[A, B, C, D, A]$ with target $A$), World A requires $z^*_A = 0$ while World B requires $z^*_B = 4$. Because learner states are identical, distinguishing these coordinates is information-theoretically impossible. Contract v1 is **mathematically falsified**.
2. **Cross-Relation Holdout Counterexample:**  
   Constructed World A (identity positional map $z^*(k, L) = k$) and World B (reversal positional map $z^*(k, L) = L - 1 - k$) for an unseen relation $t_{\text{unseen}}$. Because $t_{\text{unseen}}$ receives zero optimizer updates during training, learner training histories are identical. Furthermore, symmetric few-shot support sequences (e.g. palindromes) yield identical observed targets across both worlds, proving that support-derived context cannot break symmetry on asymmetric query sequences without oracle disambiguation supervision.
3. **Five Key Contract Dimensions Audit:**  
   - *Clean-Support Coverage:* When vocabulary size $|\mathcal{V}| < L$, the pigeonhole principle guarantees clean instances cannot exist. In empirical distributions, specific $(t, L, k)$ triplets lack clean coverage, leaving parameters unoptimized under gradient masking. (**FAIL**)  
   - *Content-Independent Semantics:* Reducing routing to a pure positional map $g_\theta(j, k, L, t)$ eviscerates APC's core semantic routing requirements (e.g. `SELECT`, `BIND`, `COUNT`, `NEIGHBOR_*`), which inherently require content-dependent addressing. (**FAIL**)  
   - *Task ID / Context Transfer:* Opaque task IDs have zero transferability without parameter updates; support-derived context encounters duplicate-token ambiguity in the support set itself. (**FAIL**)  
   - *Optimizer Non-Exposure:* Contract v1 relies on gradient descent updates to converge routing geometry, which directly contradicts the strict zero-exposure requirement for held-out relations. (**FAIL**)  
   - *Collision Predicate:* Contract v1's predicate `exists j != z: x_j == y_k == x_z` explicitly conditions on the oracle coordinate $z^*$. Observable predicates without $z^*$ are uncomputable at test time and fail on indirect value addressing. (**FAIL**)  
4. **Contract v1 Falsification & Impossibility of Lawful Contract v1.1:**  
   Because the counterexamples definitively hold, Contract v1 is falsified. Deriving a Contract v1.1 would require imposing artificial, non-general axioms (e.g. restricting all tasks to pure positional permutations, forcing $|\mathcal{V}| > L$, or injecting oracle coordinate supervision), violating the Phase C charter.

**Decision:**  
Formally correct the stoppage rationale from `RELATION_INVENTORY_FEASIBILITY_STOP` to **`ROUTING_IDENTIFIABILITY_STOP`**.

**Consequences:**  
- Identifiability of semantic routing coordinates under ambiguous token outputs without oracle routing supervision is mathematically refuted under the permitted observable contract.
- Phase C research execution remains strictly blocked (`NOT_AUTHORIZED`), and architecture task `C-D002` is permanently barred under this premise.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001R_ADVERSARIAL_IDENTIFIABILITY_FALSIFICATION_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/adversarial_identifiability_audit.json`  

---

## ADR-0152: C-D001S Quantifier-Complete Task-Side/Support Identifiability Boundary Audit Confirms `ROUTING_IDENTIFIABILITY_STOP` across Permitted Observable Space

**Date:** 2026-09-13  
**Task:** C-D001S — Quantifier-Complete Task-Side/Support Identifiability Boundary Audit  
**Status:** Completed non-experimental mathematical boundary audit; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_STOP (CONFIRMED_QUANTIFIER_COMPLETE)`.  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated all permitted task-side observables partitioned into Opaque ID ($t$), Compositional Semantic Descriptor ($D$), and Finite Output-Labeled Support Set ($\mathcal{S}$).
- Evaluated bounded separating support set pre-fixability over finite relation hypothesis classes.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Permitted Observable Taxonomy:**  
   Exhaustively partitioned permitted oracle-free task-side signals under `h_content = f(content)` into: (1) Opaque ID, (2) Pre-declared Compositional Semantic Descriptor, and (3) Finite Output-Labeled Support Set.
2. **Counterexamples across All Classifications:**  
   - *Opaque ID:* Constructed World A (first-occurrence) and World B (last-occurrence) sharing identical ID $t_0$. On clean training instances, $y_A = y_B$, losses, and gradients are bitwise identical; on collision test inputs ($x=[A, B, C, D, A]$, $y_0=A$), true coordinates diverge ($z^*_A = 0 \neq 4 = z^*_B$). (**FAIL**)  
   - *Compositional Descriptor:* Structured AST descriptor $D$ declares operation type and argument slots. Under the charter's non-leakage invariant, $D$ cannot be a hard-coded coordinate lookup table. Worlds A and B both truthfully satisfy $D$, yielding identical training histories and identical descriptors, but divergent test coordinates. (**FAIL**)  
   - *Finite Output-Labeled Support Set:* Proved Theorem 1 (Universal Token Output Identity): for any sequence $x \in \mathcal{V}^L$, $x_{\min \{i \mid x_i = y_k\}} = y_k = x_{\max \{i \mid x_i = y_k\}}$. Thus $f_A(x) \equiv f_B(x)$ identically on all inputs. For any finite support set $\mathcal{S}$, observed output tokens $y^{(m)}$ are bitwise identical across World A and World B. Support sets cannot break coordinate symmetry. (**FAIL**)  
3. **Audit of Bounded Separating Support Set Hypothesis:**  
   Audited whether a bounded separating support set pre-fixed without oracle coordinates over a finite relation hypothesis class $\mathcal{H}_{\text{rel}}$ can resolve routing ambiguity:
   - *Extensional vs Intensional Gap:* A bounded separating support set can identify which extensional function $f_m$ is active if hypotheses differ in output behavior on clean inputs. However, it cannot separate intensionally distinct routing coordinate rules ($f_A \equiv f_B$, yet $z^*_A \neq z^*_B$).
   - *Charter Invariant Violation:* Restricting $\mathcal{H}_{\text{rel}}$ to exclude alternative routing rules requires hard-coding canonical routing maps into hypothesis definitions, violating the non-negotiable charter prohibition against hard-coded routing maps.
   - *Open-World Contradiction:* G1 relation transfer requires transfer to unseen open-world relations, where the hypothesis class is unbounded, making a pre-fixed bounded separating support set mathematically impossible.
4. **Resolution of ADR-0151 Scope:**  
   Because counterexamples hold across all three classifications, ADR-0151's stoppage rationale is not an artifact of Contract v1 or adversarial palindrome support sets. It is strengthened into a Quantifier-Complete Impossibility Theorem across all permitted oracle-free observables for hypothesis H-C1. ADR-0151 is not narrowed to Contract-v1.

**Decision:**  
Definitively confirm **`ROUTING_IDENTIFIABILITY_STOP`** as a quantifier-complete impossibility proof over all permitted observables for H-C1.

**Consequences:**  
- Phase C research execution remains strictly `NOT_AUTHORIZED`.
- Architecture derivation (`C-D002`) and training formulation are permanently barred under the oracle-free premise of H-C1.
- All candidate, bundle, and sealed evaluation prerequisites remain permanently stopped.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001S_QUANTIFIER_COMPLETE_IDENTIFIABILITY_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/quantifier_complete_identifiability_audit.json`  

---

## ADR-0153: C-D001T Semantic-Descriptor Boundary & Impossibility-Proof Repair Audit Qualifies Stop to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` (ADR-0152 Quantifier Retracted / Restricted)

**Date:** 2026-09-13  
**Task:** C-D001T — Semantic-Descriptor Boundary & Impossibility-Proof Repair Audit  
**Status:** Completed non-experimental boundary audit; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`.  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Read only the prospective Phase C Research Charter, ADR-0150 through ADR-0152, operation registries in `src/apc/environments/operations.py`, and holdout families in `src/apc/environments/holdout_families.py`.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Formal Language $\mathcal{L}_{\text{desc}}$ and Oracle-Leakage Rules:**  
   Charter invariants permit high-level symbolic operations, typed arguments, and universal tie-break policies (e.g. FIRST, LAST, LEFTMOST, RIGHTMOST, CENTER) via the separate task path while keeping content encoding task-blind ($h_{\text{content}} = f(\text{content})$). Four decision rules enforce: (1) content-invariance ($D$ cannot vary with runtime sequence $x$), (2) no coordinate map lookup tables ($k \mapsto z^*$), (3) no decoder bypass ($y$ not encoded in $D$), and (4) admissibility of universal tie-break policies (abstract ordering relations independent of content and target).
2. **Non-Circular Counterexample Repair & Permitted Task Classes:**  
   Identified fatal circularity in ADR-0151/ADR-0152 ($z^* = \min \{ i \mid x_i == y_k \}$ where $y_k = x_{z^*}$). Repaired definitions using actual registered APC operations:
   - *`BindOp` (`src/apc/environments/operations.py`):* Matching key set $I_K(x) = \{ 2j \mid x_{2j} = K \}$ for task parameter `query_key` $K$. World A (first-match: $z^*_A = \min I_K(x) + 1$, $f_A(x) = x_{z^*_A}$) vs World B (last-match / canonical APC `BindOp`: $z^*_B = \max I_K(x) + 1$, $f_B(x) = x_{z^*_B}$). On clean keys, $z^*_A = z^*_B$ and $f_A = f_B$; on duplicate keys with identical value ($x=[K, V, K, V]$), $f_A(x) = V = f_B(x)$, yet $z^*_A = 1 \neq 3 = z^*_B$.
   - *`NeighborMaxOp` (`src/apc/environments/holdout_families.py`):* Local window max $f_A(x)_i = f_B(x)_i = \max W_i(x)$ everywhere. Leftmost-max ($z^*_A$) vs Rightmost-max ($z^*_B$) diverge on duplicate maxima in window ($[5, 5, 2] \implies z^*_A = 0 \neq 1 = z^*_B$).  
   Both constructions define $f$ completely without referencing $y$ and belong to actual registered APC task classes.
3. **Separation Capacity of Lawful Descriptors (Theorem 3):**  
   Under $\mathcal{L}_{\text{desc}}$, World A has descriptor $D_A = (\dots, \text{tie\_break}=\text{FIRST})$ and World B has $D_B = (\dots, \text{tie\_break}=\text{LAST})$. Because $D_A \neq D_B$, task-side observations are strictly distinct ($O_{\text{task}}(A) \neq O_{\text{task}}(B)$). Proved Theorem 3 (Descriptor Separation Theorem): any deterministic descriptor in $\mathcal{L}_{\text{desc}}$ uniquely maps $(D, x, k) \mapsto z^*(x, k)$, so divergent coordinates imply distinct descriptors ($z^*_A \neq z^*_B \implies D_A \neq D_B$). World A and World B cannot share the same descriptor under the lawful language.
4. **Retraction and Scope Restriction of ADR-0152:**  
   Because lawful descriptors separate the worlds, ADR-0152's claim of a "quantifier-complete impossibility theorem across all permitted task-side observables" is mathematically refuted and retracted. Non-identifiability is strictly restricted to: (1) Opaque Task Identifiers ($t$), (2) Finite Output-Labeled Support Sets ($\mathcal{S}$) under token supervision alone, and (3) Underspecified Descriptors lacking tie-break semantics.

**Decision:**  
Reclassify the stoppage rationale from `ROUTING_IDENTIFIABILITY_STOP (CONFIRMED_QUANTIFIER_COMPLETE)` to **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`**.

**Consequences:**  
- Phase C routing identifiability is mathematically achievable under fully specified compositional descriptors ($\mathcal{L}_{\text{desc}}$).
- Research execution remains strictly `NOT_AUTHORIZED` due to the unresolved relation inventory deficit from ADR-0150 (`RELATION_INVENTORY_FEASIBILITY_STOP`: 1/2 clean components in validation and sealed) and unapproved charter status.
- Incorporating $\mathcal{L}_{\text{desc}}$ as the official training information contract requires a formal charter amendment and user approval before any architecture derivation or experiment can be proposed.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001T_SEMANTIC_DESCRIPTOR_BOUNDARY_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/semantic_descriptor_boundary_audit.json`  

---

## ADR-0154: C-D001U Semantic Descriptor Oracle-Equivalence & Minimality Audit Reconfirms `ROUTING_IDENTIFIABILITY_STOP`

**Date:** 2026-09-13  
**Task:** C-D001U — Semantic Descriptor Oracle-Equivalence & Minimality Audit  
**Status:** Completed non-experimental mathematical reduction & minimality audit; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_STOP (RECONFIRMED)`.  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated C-D001T descriptor primitives (`operation_family`, `arguments`, `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, `procedural composition`) against Phase C Charter H-C1 and oracle prohibitions.
- Constructed component-wise minimality counterexamples by eliminating individual information elements.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Universal $z$-Computability of Fully Specified Descriptors:**  
   Proved that for any descriptor $D \in \mathcal{L}_{\text{desc}}$ equipped with operational predicates, arguments, and total tie-break rules, there exists a deterministic reduction $R(x, D, k)$ that computes the exact target routing coordinate $z^*(x, k)$ in $O(L)$ time without target tokens or external labels. All separating descriptors derived in C-D001T are strictly $z$-computable.
2. **Oracle-Equivalence of Candidate Selection Procedures:**  
   Audited candidate selection procedures under Charter Hypothesis H-C1 ("*without oracle routing supervision*") and non-negotiable prohibitions ("*A task-side identifier cannot become a hard-coded correct routing map*"; "*Forbidden: correct source-coordinate maps*"). Found that an intensional deterministic reduction algorithm $R(x, D) \mapsto z^*$ is mathematically isomorphic to an extensional coordinate map $(x, k) \mapsto z^*$. Specifying index-selection rules (e.g. `FIRST/LAST`, `LEFTMOST/RIGHTMOST`) provides direct coordinate guidance rather than learning routing from token supervision, constituting oracle-equivalent supervision.
3. **Minimality Counterexamples and Failure of Non-Oracle Separation:**  
   Constructed counterexamples by component-wise elimination:
   - Eliminating the oracle-equivalent `TieBreakPolicy` yields a lawful, non-oracle descriptor $D_{\text{no\_tb}} = (\text{OpFamily}, \text{Arguments})$.
   - However, World A (first-match) and World B (last-match) share identical lawful descriptors ($D_{\text{no\_tb}}^A \equiv D_{\text{no\_tb}}^B$).
   - On collision-bearing sequences ($x=[K, V, K, V]$ for `BindOp`), output tokens are identical ($y=V$), but required coordinates diverge ($z^*_A = 1 \neq 3 = z^*_B$).
   - Thus, lawful non-oracle descriptors **cannot separate World A and World B**.
   - Eliminating arguments or operation families renders the extensional function $f$ undefined even on clean inputs.
4. **Rejection of Training Information Contract v1.1:**  
   Because all descriptors capable of separating World A and World B are $z$-computable and oracle-equivalent, no lawful, non-oracle Contract v1.1 can be formed. Contract v1.1 convergence is **FORMALLY REJECTED**.

**Decision:**  
Definitively reconfirm **`ROUTING_IDENTIFIABILITY_STOP`** across the entire lawful, non-oracle observable space.

**Consequences:**  
- Identifiability of semantic routing coordinates under duplicate token outputs without oracle routing supervision is impossible under any lawful, non-oracle task observable.
- Architecture derivation (`C-D002`) remains strictly blocked. No progression to architecture design is permitted without a fundamental Charter amendment approved by the user.
- Phase C research execution remains strictly `NOT_AUTHORIZED`.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001U_SEMANTIC_DESCRIPTOR_ORACLE_EQUIVALENCE_AND_MINIMALITY_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/semantic_descriptor_oracle_equivalence_audit.json`

---

## ADR-0155: C-D001V Non-Circular Oracle-Equivalence Falsification Experiment Retracts ADR-0154 Universal STOP to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`

**Date:** 2026-09-13  
**Task:** C-D001V — Non-Circular Oracle-Equivalence Falsification Experiment  
**Status:** Completed non-experimental falsification experiment; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`.  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Established a 5-dimensional non-circular oracle-supervision criterion (provenance, example specificity, relation specificity, inference-time availability, counterfactual invariance) calibrated against two negative controls (pre-declared semantic rule, opaque task ID) and two positive controls (per-example coordinate label, relation lookup map).
- Applied registered criteria to `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, and `procedural composition`.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Circularity Defect of ADR-0154:**  
   Identified that ADR-0154 equated deterministic $z$-computability ($R(x, D) \to z^*$) with oracle supervision. In any formal deterministic task semantics, target behaviors and intermediate states are mathematically computable from input and task specification. Equating computability with oracle supervision renders all valid semantic specifications tautologically "oracle", producing an unfalsifiable circular criterion.
2. **Pre-Registered Multi-Dimensional Criteria & Control Calibration:**  
   Pre-registered five criteria before evaluation:
   - Positive Controls (coordinate label, relation lookup map) score 5/5 as ORACLE (empirical evaluator provenance, instance-dependent, brittle under counterfactual mutation, forbidden at inference).
   - Negative Controls (pre-declared semantic rule, opaque ID) score 0/5 as ORACLE (a priori formal grammar, $x$-invariant universal rules, domain-general, legitimate prompt inputs, counterfactually robust).
3. **Candidate Primitives Evaluated as Strictly Non-Oracle:**  
   `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, and `procedural composition` score **0/5 as ORACLE**, matching Negative Controls in 100% of dimensions and Positive Controls in 0% of dimensions. They specify the intensional operational objective (what relation to compute), leaving the router to dynamically search sequence $x$ to resolve coordinates.
4. **Falsification and Retraction of ADR-0154 Universal STOP:**  
   Because lawful, non-oracle general semantic descriptors exist and separate World A and World B ($D_A \neq D_B \implies \mathcal{O}_A \neq \mathcal{O}_B$ on duplicate-token collisions), ADR-0154's universal claim that all separating descriptors are oracle-equivalent is **falsified and retracted**.
5. **Reclassification of Stoppage Rationale:**  
   Stoppage is formally reclassified to **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`**:
   - Under opaque task IDs ($t$) and CE-only token supervision, routing coordinates remain unidentifiable on collision tokens (`ROUTING_IDENTIFIABILITY_STOP` strictly maintained).
   - Under lawful general semantic descriptors ($\mathcal{L}_{\text{desc}}$), routing identifiability is mathematically achievable without oracle supervision.

**Decision:**  
Reclassify the stoppage rationale from `ROUTING_IDENTIFIABILITY_STOP (RECONFIRMED)` to **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`**.

**Consequences:**  
- Phase C routing identifiability is mathematically achievable under lawful general semantic descriptors ($\mathcal{L}_{\text{desc}}$).
- Research execution remains strictly `NOT_AUTHORIZED` due to the unresolved relation inventory deficit from ADR-0150 (`RELATION_INVENTORY_FEASIBILITY_STOP`: 1/2 clean components in validation and sealed) and unapproved charter status.
- Advancing to architecture derivation (`C-D002`) requires a formal Charter amendment incorporating $\mathcal{L}_{\text{desc}}$ and user approval.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001V_ORACLE_EQUIVALENCE_FALSIFICATION_EXPERIMENT.md`  
- Audit Artifact: `docs/phase_c/artifacts/oracle_equivalence_falsification_experiment.json`

---

## ADR-0156: C-D001W Adversarial Mixed-Control Validation Confirms ADR-0155 0/5 Non-Oracle Determination and Derives Training Information Contract v1.1 & Deterministic Baseline

**Date:** 2026-09-13  
**Task:** C-D001W — Adversarial Mixed-Control Validation of the Five-Dimensional Oracle Criterion  
**Status:** Completed non-experimental adversarial validation; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` (ADR-0155 0/5 Determination Upheld; Candidate Contract v1.1 & Deterministic Baseline Derived).  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated ADR-0155's fixed 5-dimensional oracle criterion against adversarial mixed controls between 0/5 and 5/5: universal interpreter + bytecode relation code (MC1), compressed/encrypted lookup (MC2), relation-specific coordinate-free metadata (MC3), example-independent task-specific routing program (MC4), support-derived selector (MC5), lawful FIRST/LAST descriptor (MC6), and noisy coordinate hint (MC7).
- Verified representation invariance, positive/negative control monotonicity, leave-one-out dimension necessity, and aggregation threshold false positives / false negatives.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Representation Invariance of Oracle Nature:**  
   Meaning-preserving transformations of coordinate lookups—compiling into universal bytecode (`MC1`) or encrypting/compressing (`MC2`)—preserve full mutual information with ground truth ($I(Z^*; T(C)) = I(Z^*; C)$). The 5-dimensional criterion inspects semantic provenance and counterfactual fragility rather than surface syntax, scoring both as **5/5 ORACLE**. The criterion is strictly representation-invariant.
2. **Positive/Negative Control Monotonicity:**  
   The oracle score $S(C)$ monotonically tracks coordinate leakage: $S(\text{NC1, NC2, MC3, MC5, MC6}) = 0 < S(\text{MC4}) = 4 \le S(\text{PC1, PC2, MC1, MC2, MC7}) = 5$.
3. **Leave-One-Out Dimension Necessity:**  
   Ablating any single dimension $d_j \in \{1, 2, 3, 4, 5\}$ exposes a concrete adversarial loophole (e.g. omitting relation specificity admits `MC4`; omitting provenance admits evaluator-fitted closed forms). All five dimensions are **jointly necessary and non-redundant**.
4. **Aggregation Threshold Analysis (FP = 0, FN = 0):**  
   The Disjunctive Coordinate-Leakage Rejection Rule ($\theta = 1$, Any-Hit) achieves optimal fail-closed classification: 0.0% False Positives on lawful task specifications (which all score strictly 0/5) and 0.0% False Negatives on oracles (unanimous rule $\theta = 5$ fails by admitting `MC4` at 4/5).
5. **Reconfirmation of ADR-0155 0/5 Determination:**  
   Exhaustive search revealed zero counterexamples where re-encoding altered oracle determinations or exposed `FIRST/LAST` as an oracle. ADR-0155's 0/5 non-oracle classification is **unconditionally upheld**.
6. **Candidate Training Information Contract v1.1 & Deterministic Baseline:**  
   Formally derived `Phase C Training Information Contract v1.1` (specifying $D \in \mathcal{L}_{\text{desc}}$ with tie-break policies under strict $0/5$ compliance and $h_{\text{content}} = f(\text{content})$) and `Descriptor-Only Deterministic Baseline` requirements (software reduction achieving 100% EM on clean and collision instances as a causal ceiling control).

**Decision:**  
Confirm **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`** (ADR-0155 0/5 Determination Upheld; Candidate Contract v1.1 & Deterministic Baseline Derived).

**Consequences:**  
- The 5-dimensional oracle criterion is verified as robust, representation-invariant, and minimal.
- Contract v1.1 provides a single mathematically consistent information contract candidate that resolves duplicate-token routing without oracle supervision.
- Research execution remains strictly `NOT_AUTHORIZED` due to the unresolved relation inventory deficit from ADR-0150 (`RELATION_INVENTORY_FEASIBILITY_STOP`: 1/2 clean components in validation and sealed) and unapproved charter status.

**Primary Artifacts:**  
- Review Document: `docs/phase_c/PHASE_C_C_D001W_ADVERSARIAL_MIXED_CONTROL_VALIDATION.md`  
- Audit Artifact: `docs/phase_c/artifacts/adversarial_mixed_control_validation.json`  
- Verification Suite: `tests/test_adversarial_mixed_control_validation.py`

---

## ADR-0157: C-D001X Contract-v1.1 Hypothesis-Preservation & Deterministic-Baseline Dominance Audit Finds H-C1 Estimand Trivialized by Unlearned Deterministic Reduction

**Date:** 2026-09-13  
**Task:** C-D001X — Contract-v1.1 Hypothesis-Preservation & Deterministic-Baseline Dominance Audit  
**Status:** Completed formal audit & controlled ablations; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` (`audit_verdict: H_C1_TRIVIALIZED_ESTIMAND_ALTERED_BY_CONTRACT_V1_1`; `charter_recommendation: RETRACT_OR_RESTRICT_TO_RESIDUAL_LEARNING`).  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Deconstructed H-C1's primary estimand into three distinct sub-problems: task identity learning, descriptor interpretation, and coordinate selection.
- Evaluated three conditions (opaque-ID/CE-only, Contract v1.1 semantic descriptor, descriptor-only deterministic baseline $B_{\text{det}}$) over existing public operation semantics (`BindOp`, `NeighborMaxOp`) on conditional routing entropy $H(Z \mid X, D)$, World A/B separation, and unseen relation transfer.
- Executed three controlled ablations: TieBreakPolicy masking, FIRST $\leftrightarrow$ LAST counterfactual swap, and operation/argument-preserving descriptor permutation.
- Sealed partition data access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B terminal state was modified.

**Key Findings:**  
1. **Estimand Deconstruction and Trivialization:**  
   In Phase B (opaque ID $t$), coordinate selection under duplicate tokens was unidentifiable because token loss gradients are symmetric ($H(Z \mid X, t, y) = 1.0\text{ bit} > 0$). In Contract v1.1, task identity (`OpFamily`, `Args`) and coordinate selection (`TieBreakPolicy`) are both declared explicitly in descriptor $D \in \mathcal{L}_{\text{desc}}$. Given $(X, D)$, the routing coordinate $Z^*$ is a single-valued deterministic reduction computable before seeing any target token $y$, rendering residual uncertainty identically zero ($H(Z \mid X, D) = 0.0\text{ bits}$).
2. **Deterministic Baseline Dominance ($B_{\text{det}}$ achieves 100% without learning):**  
   The unlearned Descriptor-Only Deterministic Baseline ($B_{\text{det}}$ / `BASE-DET-DESC-V1`, 0 parameters, 0 training updates) satisfies **all registered routing and execution acceptance floors** ($\ge 0.95$) with 1.000 (100.0%) precision on clean and collision instances, passes all causal controls, and exhibits zero initialization variance.
3. **Controlled Ablation Results:**  
   - *TieBreakPolicy Masking:* Omitting the tie-break policy causes $H(Z \mid X, D)$ to jump from 0.0 to 1.0 bit, collapsing World A/B separation from 100% to 0.0%. This confirms TieBreakPolicy is the sole causal component resolving ambiguity.
   - *FIRST $\leftrightarrow$ LAST Counterfactual Swap:* Swapping the tie-break policy shifts target coordinates ($\Delta z^* = 2$) with zero change in target tokens ($\Delta y = 0$), proving coordinate selection is 100% causally commanded by descriptor syntax and 0% driven by token output supervision.
   - *Descriptor Permutation:* Coordinate selection tracks descriptor permutations with 100% fidelity while sequence token content remains invariant.
4. **Estimand Transformation and Charter Verdict:**  
   Because $B_{\text{det}}$ solves 100% of routing and execution without learning, Contract v1.1 **trivializes H-C1** and fundamentally alters the primary estimand from "learning latent routing identity from ambiguous token supervision" to "neural compilation / function approximation of an already-known 0-parameter deterministic reduction algorithm". The Phase C Charter candidate must be **retracted or restricted to a non-trivial residual learning objective**.
5. **Specification of Single Residual Learning Problem & Baseline Delta:**  
   Identified that $B_{\text{det}}$ cannot process continuous distributed embeddings and cannot execute unhandled relation families without code additions. Articulated the single residual learning problem: "Continuous Neural Grounding & Zero-Shot Generalization of Structured Descriptors under Token Supervision alone", with baseline delta $\Delta_{\text{baseline}}(M) = \text{Metric}(M) - \text{Metric}(B_{\text{det}})$, where $B_{\text{det}} = 1.000$ serves as the theoretical ceiling control.

**Decision:**  
Confirm **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`** (`H_C1_TRIVIALIZED_ESTIMAND_ALTERED_BY_CONTRACT_V1_1`; `charter_recommendation: RETRACT_OR_RESTRICT_TO_RESIDUAL_LEARNING`).

**Consequences:**  
- Advancing H-C1 as originally formulated under Contract v1.1 is rejected as scientifically trivialized by deterministic baseline dominance.
- Advancing Phase C requires revising the Charter to explicitly target continuous neural grounding / generalization against the $B_{\text{det}}$ ceiling, or retracting the Charter candidate.
- Research execution remains strictly `NOT_AUTHORIZED` due to unresolved relation inventory deficit (ADR-0150: 1/2 clean components), unapproved charter status, and the estimand trivialization stop.
- Architecture derivation (`C-D002`) remains strictly barred.

**Primary Artifacts:**  
- Review Document: `docs/phase_c/PHASE_C_C_D001X_CONTRACT_V1_1_HYPOTHESIS_PRESERVATION_AND_DETERMINISTIC_BASELINE_DOMINANCE_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/contract_v1_1_hypothesis_preservation_audit.json`  
- Verification Suite: `tests/test_contract_v1_1_hypothesis_preservation_audit.py`

---

## ADR-0158: C-D001Y Embedding-Aware Deterministic Baseline Closure Audit Finds Continuous Neural Grounding Dominated by B_det_emb Under Invertible Representations

**Date:** 2026-09-13  
**Task:** C-D001Y — Embedding-Aware Deterministic Baseline Closure Audit  
**Status:** Completed formal audit & controlled evaluations; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` (`audit_verdict: CONTINUOUS_GROUNDING_DOMINATED_BY_B_DET_EMB_UNDER_INVERTIBLE_REPRESENTATIONS`; `charter_recommendation: RETRACT_OR_RESTRICT_TO_BLIND_MANIFOLD_DISCOVERY`).  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated C-D001X's residual "continuous neural grounding" claim against a single unlearned embedding-aware deterministic baseline ($B_{\text{det\_emb}}$ / `BASE-DET-EMB-V1`, 0 learned parameters, 0 training updates) combining fixed codebook decoding, nearest-neighbor projection, and known invertible linear map inversion with discrete $B_{\text{det}}$.
- Evaluated five pre-registered controls: Control 1 (discrete symbols), Control 2 (invertible distributed representations), Control 3 (equidistant Voronoi boundary collisions), Control 4 (information-lossy rank-deficient mappings), and Control 5 (bounded perturbations within registered safety margin $\epsilon < \frac{1}{2} d_{\min}$).
- Formally evaluated decodability EM, clean and collision routing accuracy, execution EM, conditional routing entropy $H(Z \mid H(X), D)$, and descriptor swap covariance ($\text{FIRST} \leftrightarrow \text{LAST}$, $\text{LEFTMOST} \leftrightarrow \text{RIGHTMOST}$).
- Sealed partition data access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B/C terminal state was modified.

**Key Findings:**  
1. **Ceiling Dominance Under Invertible Representations ($B_{\text{det\_emb}}$ achieves 100%):**  
   Under Controls 1, 2, and 5 (discrete symbols, dense invertible linear transforms $W \in GL(d)$, and bounded perturbations within the Voronoi safety margin $\epsilon < \frac{1}{2} d_{\min}$), $B_{\text{det\_emb}}$ achieves **1.000 (100.0%) decodability EM, 1.000 routing accuracy, 1.000 execution EM, $H(Z \mid H(X), D) = 0.0\text{ bits}$, and 1.000 descriptor swap covariance** without any learned parameters or training updates.
2. **Falsification of Continuous Grounding as a Non-Trivial Estimand:**  
   Continuous representation per se (representing tokens as continuous vectors in $\mathbb{R}^d$ rather than discrete integers) does not pose an inductive learning barrier. An unlearned, 0-parameter deterministic baseline with nearest-neighbor projection and linear inversion solves continuous grounding completely. "Continuous neural grounding" is therefore **not a non-trivial residual estimand**; Charter candidates proposing it as an empirical justification for neural learning must be **further retracted and restricted**.
3. **Identifiability Limit vs. Learnability Limit:**  
   Performance drops exclusively under Controls 3 and 4 (equidistant Voronoi midpoint collisions and rank-deficient lossy projections). In both cases, information is mathematically destroyed ($H(X \mid H(X)) > 0$ bits). This degradation is an insurmountable **information-theoretic identifiability limit** (識別可能性限界), not a learnability limit (学習可能性の限界). No neural network can resolve unidentifiable inputs without oracle supervision.
4. **Specification of Single Residual Learning Hypothesis & Baseline Delta:**  
   An unlearned baseline fails only when the codebook $\mathcal{C}$ and transformation $W$ are completely unlabelled and unknown. Articulated the single remaining valid residual hypothesis: `H-C1-Residual: Blind Continuous Manifold Grounding & Unsupervised Codebook Discovery` ("Can a continuous neural attention mechanism discover discrete token equivalence classes and execute relational routing under token loss supervision alone without an explicit codebook or known inverse map?"), with baseline delta $\Delta_{\text{baseline}}(M) = \text{Metric}(M) - \text{Metric}(B_{\text{det\_emb}})$, where $B_{\text{det\_emb}} = 1.000$ serves as the theoretical ceiling control.

**Decision:**  
Confirm **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`** (`CONTINUOUS_GROUNDING_DOMINATED_BY_B_DET_EMB_UNDER_INVERTIBLE_REPRESENTATIONS`; `charter_recommendation: RETRACT_OR_RESTRICT_TO_BLIND_MANIFOLD_DISCOVERY`).

**Consequences:**  
- Advancing H-C1 under the guise of "continuous representation grounding" is formally rejected as dominated by an unlearned deterministic baseline.
- Phase C Charter candidate must be further retracted or restricted strictly to blind manifold discovery / unsupervised codebook learning.
- Research execution remains strictly `NOT_AUTHORIZED` due to unresolved relation inventory deficit (ADR-0150: 1/2 clean components), unapproved charter status, and the continuous baseline dominance stop.
- Architecture derivation (`C-D002`) remains strictly barred.

**Primary Artifacts:**  
- Review Document: `docs/phase_c/PHASE_C_C_D001Y_EMBEDDING_AWARE_DETERMINISTIC_BASELINE_CLOSURE_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/embedding_aware_deterministic_baseline_audit.json`  
- Baseline Implementation: `src/apc/evaluation/embedding_aware_baseline.py`  
- Verification Suite: `tests/test_embedding_aware_deterministic_baseline_audit.py`




