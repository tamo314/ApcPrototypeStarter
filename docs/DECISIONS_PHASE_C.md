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

---

## ADR-0159: C-D001Z Blind Codebook Identifiability & Symmetry-Breaking Audit Proves Blind Grounding Impossible and Retracts Residual Charter Candidate

**Date:** 2026-09-13  
**Task:** C-D001Z — Blind Codebook Identifiability & Symmetry-Breaking Audit  
**Status:** Completed formal audit & controlled evaluations; `execution_status: PASS`, `decision: ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` (`audit_verdict: BLIND_GROUNDING_IDENTIFIABILITY_IMPOSSIBILITY_PROVEN`; `charter_candidate_decision: RETRACT_CHARTER_CANDIDATE`).  
Phase C charter status remains `READY_FOR_REVIEW_NOT_APPROVED`; research execution remains `NOT_AUTHORIZED`.  

**Scope and Integrity Boundary:**  
- Evaluated C-D001Y's residual hypothesis `H-C1-Residual: Blind Continuous Manifold Grounding & Unsupervised Codebook Discovery` against formal permutation ($S_V$) and orthogonal ($O(d)$) group actions on unknown codebooks $\mathcal{C}$ and unknown invertible bases $W$.
- Evaluated four pre-registered controls: Control 1 (no-anchor, pure blind), Control 2 (1-anchor), Control 3 (partial-anchor), and Control 4 (full-codebook) over public `BindOp` semantics and descriptors $D \in \mathcal{L}_{\text{desc}}$.
- Constructed mathematically indistinguishable symmetric worlds (World A vs World B) preserving sequence embeddings $H(X)$, semantic descriptors $D$, and execution target feedback $y$ ($H_A = H_B, D_A = D_B, y_A = y_B$) with divergent true semantic routing coordinates ($z^*_A \ne z^*_B$).
- Evaluated minimal lawful anchor sets under ADR-0155's 5-dimensional oracle criteria (0/5 non-oracle) and analyzed interaction with $B_{\text{det\_emb}}$.
- Sealed partition data access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, and zero GPU execution time.
- No historical measurement, threshold, or Phase-B/C terminal state was modified.

**Key Findings:**  
1. **Identifiability Impossibility Under Blind Representation ($H(Z \mid \text{obs}) > 0$):**  
   Under no-anchor or incomplete anchors, the permutation group $S_V$ and orthogonal group $O(d)$ act transitively on unanchored coordinates. Observables $(H, D, y)$ are 100% statistically identical across symmetric worlds while required coordinates diverge ($z^*_A \ne z^*_B$), proving that blind grounding is an insurmountable **information-theoretic identifiability impossibility** (識別不能性定理), not an inductive learnability problem.
2. **Floor Failures Across Incomplete Anchor Controls:**  
   - *Control 1 (No-Anchor):* Orbit size $4! = 24$, $H(Z \mid \text{obs}) = 2.000\text{ bits}$, baseline upper bound bounded by chance $0.250 \ll 0.95$, swap covariance $0.000$.
   - *Control 2 (1-Anchor):* Unanchored orbit $3! = 6$, $H(Z \mid \text{obs}) = 1.189\text{ bits}$, mean accuracy $0.500 < 0.95$.
   - *Control 3 (Partial-Anchor):* Unanchored orbit $2! = 2$, $H(Z \mid \text{obs}) = 0.500\text{ bits}$, mean accuracy $0.750 < 0.95$.
3. **Deterministic Baseline Dominance Under Complete Anchors:**  
   Breaking permutation symmetry requires pre-declaring at least $K - 1$ anchors ($4.585\text{ bits}$ of information). While static anchors are strictly non-oracle (0/5 on ADR-0155 criteria), providing them enables the unlearned deterministic baseline $B_{\text{det\_emb}}$ to achieve **1.000 (100.0%) routing accuracy and 1.000 execution EM with 0 learned parameters**, eliminating any learning margin ($\Delta_{\text{baseline}}(M) \le 0.000$).
4. **The Impossibility-Dominance Dilemma and Residual Retraction:**  
   Formulated the Dilemma Theorem: between identifiability impossibility ($H(Z \mid \text{obs}) > 0$ when $I < I_{\text{crit}}$) and deterministic baseline dominance ($B_{\text{det\_emb}} = 1.000$ when $I \ge I_{\text{crit}}$), the admissible residual learning set $\mathcal{H}_{\text{residual}} = \{ \mathcal{R} \mid H(Z \mid \text{obs}) = 0 \text{ and } \operatorname{Acc}(B_{\text{det\_emb}}) < 0.95 \}$ is **identically EMPTY** ($\emptyset$). The Phase C residual Charter candidate `H-C1-Residual` is formally **retracted**.

**Decision:**  
Confirm **`ROUTING_IDENTIFIABILITY_QUALIFIED_STOP`** (`BLIND_GROUNDING_IDENTIFIABILITY_IMPOSSIBILITY_PROVEN`; `charter_candidate_decision: RETRACT_CHARTER_CANDIDATE`).

**Consequences:**  
- Advancing H-C1 under blind continuous manifold grounding is formally rejected as mathematically unidentifiable.
- Phase C Charter candidate `H-C1-Residual` is formally retracted; no viable residual formulation remains under linear/isometric representations.
- Research execution remains strictly `NOT_AUTHORIZED` due to unresolved relation inventory deficit (ADR-0150: 1/2 clean components), retracted charter candidate, and complete identifiability/dominance closure.
- Architecture derivation (`C-D002`) remains strictly barred.

**Primary Artifacts:**  
- Review Document: `docs/phase_c/PHASE_C_C_D001Z_BLIND_CODEBOOK_IDENTIFIABILITY_AUDIT.md`  
- Audit Artifact: `docs/phase_c/artifacts/blind_codebook_identifiability_audit.json`  
- Audit Implementation: `src/apc/evaluation/blind_codebook_audit.py`  
- Verification Suite: `tests/test_blind_codebook_identifiability_audit.py`

---

## ADR-0160: C-D001AA Phase C Falsification Sufficiency & Charter Termination Audit Closes the Current Charter (`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`)

**Date:** 2026-09-13  
**Task:** C-D001AA — Phase C Falsification Sufficiency & Charter Termination Audit  
**Status:** Completed documentation/artifact-consistency audit; `execution_status: PASS`, `decision: PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`.  
Phase C charter status transitions from `READY_FOR_REVIEW_NOT_APPROVED` to **`TERMINATED_CURRENT_CHARTER`**. Research execution remains, and has always been, `NOT_AUTHORIZED`.

**Scope and Integrity Boundary:**  
- Read only the Phase C Research Charter, ADR-0150 through ADR-0159 and their ten review documents and eleven JSON artifacts, `docs/TASKS.md`, `README.md`, `docs/DECISIONS.md`, and `docs/exec-plans/active/PHASE_B_RESTART.md`.
- Sealed partition data (inputs, labels, model outputs) access count: **0**.
- Zero training updates, zero optimizer construction, zero model initialization, zero dataset generation, zero relation registration, zero candidate construction, zero architecture implementation, and zero GPU execution time.
- No historical ADR text (ADR-0150–ADR-0159 or earlier phases) is rewritten, renumbered, or deleted. No charter threshold, floor, oracle-boundary criterion, initialization count, or relation requirement is relaxed, added to, or removed.

**Key Findings:**  
1. **Source-of-truth state confirmed, not corrected:** all fourteen candidate state fields (charter status, research-execution authorization, C-D002 bar, Contract v1/v1.1 status, continuous-grounding and blind-grounding residual status, relation inventory 1/2·1/2, sealed access 0, candidate none) were checked directly against ADR-0150–0159 text and found accurate without modification.
2. **ADR validity map:** ADR-0152 and ADR-0154 are `RETRACTED` by their explicit successors (ADR-0153 and ADR-0155 respectively); ADR-0150's stop rationale is `SUPERSEDED` by ADR-0151 while its relation-inventory measurement remains an independently current, never-revisited fact; ADR-0151, ADR-0153, ADR-0155, ADR-0156 (validation methodology), ADR-0157, ADR-0158, and ADR-0159 are `CURRENT`. ADR-0154's "z-computable = oracle-equivalent" criterion is confirmed retracted by ADR-0155/0156, whose 5-dimensional oracle criterion is confirmed as the current, unmodified source of truth through ADR-0159.
3. **Scientific estimand clarified:** descriptor-based identifiability (ADR-0153/0155/0156) and H-C1 satisfaction are not the same accomplishment. The same descriptor that resolves identifiability makes $H(Z\mid X,D)=0$ exactly, so identifiability was purchased by supplying enough information to make the coordinate a 0-parameter deterministic function — meaning H-C1 ("learn routing identity under ambiguous supervision") was never actually tested; its estimand was replaced (ADR-0157).
4. **Impossibility-claim scope fixed:** the record supports "unidentifiable OR deterministic-baseline-dominated" only across the enumerated representation classes and the permutation/orthogonal group-symmetry assumptions of ADR-0159 (§3.3 of the audit). It explicitly does **not** support any claim of APC's general impossibility, universal task-information-contract futility, or impossibility of nonlinear/stochastic/interactive/future formulations or of routing architectures in general — none of these was constructed or tested.
5. **Impossibility–Dominance Dilemma audited:** the Case A/B dichotomy is exhaustive over the charter-permitted space *as modeled* (linear/group-symmetric representations, anchor-count-parameterized information), verified in part by ADR-0159's own partial-anchor control landing inside Case A rather than a third regime. Exhaustiveness is scoped to that representation model, not asserted for arbitrary non-group-structured representations.
6. **Non-trivial residual set audited as empty:** $H_{\text{nontrivial}}$ cannot be shown non-empty from existing evidence; every charter-permitted, oracle-free case constructed across ADR-0150–0159 (opaque ID, descriptor without/with tie-break, deterministic reduction, invertible/lossy continuous embeddings, blind codebook under four anchor regimes) resolves to Case A or Case B. No new residual hypothesis is proposed, per task instruction.
7. **Conclusion-changing experiment audit:** no independently executable, in-charter task remains that could change Phase C's conclusion. C-D002, architecture implementation, optimizer trial, extra seed, new relation family, new dataset, coefficient search, descriptor variants, anchor-count variants, and sealed evaluation are all classified `OUTSIDE_CURRENT_CHARTER` or excluded per task instruction.
8. **Relation-inventory blocker re-positioned:** ADR-0150's validation 1/2, sealed 1/2 deficit stands as an independent necessary-condition FAIL, explicitly **not** framed as "adding relations reopens Phase C" — the estimand/identifiability chain (ADR-0151–0159) is a second, independent blocker that relation-count repair alone cannot cure.
9. **Charter-amendment boundary respected:** no threshold, floor, oracle-boundary criterion, initialization count, relation requirement, or H-C1 wording was changed by this audit.
10. **Termination checklist:** all ten conditions in the task contract (H-C1 formulation fails; Contract v1 falsified; Contract v1.1 trivializes estimand; continuous-grounding residual dominated; blind-grounding residual closed by the dilemma; no lawful residual remains; relation inventory independently insufficient; research execution never authorized; sealed access 0; no in-charter conclusion-changing task remains) are satisfied.

**Decision:**  
Declare **`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`**. Transition Phase C charter status to **`TERMINATED_CURRENT_CHARTER`**.

**Consequences:**  
- The Phase C charter (`docs/research/PHASE_C_RESEARCH_CHARTER.md`) is preserved as a historical artifact: original hypothesis intact, never approved, never executed experimentally, falsified/retracted entirely during pre-execution mathematical review (ADR-0150–0159), now formally closed by this audit.
- `docs/results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md` is the authoritative claim classification for ADR-0150–ADR-0159, analogous to the Phase-B closeout ledger.
- No Phase D hypothesis, architecture, supervision scheme, relation family, or learning formulation is proposed by this ADR. A future research question requires a separate, independently authorized `NEXT-RESEARCH-QUESTION REVIEW`.
- `candidate=null`, `C-D002=NOT_AUTHORIZED`, sealed access remains `0`, relation inventory deficit remains unresolved and independently blocking. ADR-0150 through ADR-0159 are preserved verbatim; this ADR adds no new retraction of their text, only a consolidated reading of retractions they already record.
- Phase B's separate terminal state (ADR-0148/0149, `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`) is unaffected and not reused as evidence for this closure.

**Primary Artifacts:**  
- Audit Document: `docs/phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md`  
- Evidence Ledger: `docs/results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md`  
- Verification Record: `docs/results/PHASE_C_TERMINATION_AUDIT.json`

---

## ADR-0161: NRQ-001 Next-Research-Question Review Finds No Non-Trivial Identifiable Estimand (`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`)

**Date:** 2026-09-13  
**Task:** NRQ-001 — Next-Research-Question Review: Non-Trivial Identifiable Estimand Existence Test.
**Not a Phase C task.** Phase C remains `TERMINATED_CURRENT_CHARTER` (ADR-0160), unmodified. This
is the independent, cross-phase review that ADR-0160 §11 and the
[Phase C termination evidence ledger](../results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md) both name
as the required next step before any future research question may be proposed. Filed in this
ledger per `docs/DECISIONS.md`'s "append to the latest record file" instruction, mirroring how
ADR-0149 (a Phase B/C boundary event) was filed in the then-latest active decision file.  
**Status:** Completed non-experimental mathematical/design review; `decision: NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`.
Research execution remains, and has always been, `NOT_AUTHORIZED`.

**Scope and Integrity Boundary:**  
- Read-only review of `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_C.md` (ADR-0150–0160), the
  Phase C charter and its ten `PHASE_C_C_D001*.md` review documents, the Phase C and Phase B
  termination evidence ledgers, `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`,
  `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`, `AGENTS.md`, and the Phase A/A.1/A.2 decision
  ledgers via `docs/DECISIONS.md`'s index.
- Zero training, optimizer construction, model initialization, dataset generation, relation
  registration, candidate construction, architecture implementation, GPU execution time, and
  sealed-partition access (input/label/output).
- No ADR text (ADR-0001–ADR-0160) is rewritten, renumbered, or deleted. No charter threshold,
  floor, oracle-boundary criterion, initialization count, or relation requirement is relaxed,
  added to, or removed.

**Task instruction (five admission criteria):** an admissible estimand must (1) not be solvable
by a deterministic lawful baseline, (2) be information-theoretically identifiable, (3) be
oracle-free, (4) be compatible with the standing relation-transfer requirement (>=2 independent
clean relation components in validation, >=2 in sealed), and (5) actually test at least one of
APC's non-negotiable core-separation invariants in a way not already settled by existing evidence.

**Key Findings:**  
1. **Already-settled evidence excluded from the search.** Core separation under explicit,
   oracle-provided task specification is extensively `VALIDATED` (ADR-0025–0028, A1-B002–B008,
   A2-C002–C012) and not re-opened. Phase B's architecture-level negative conclusion (ADR-0148)
   and Phase C's estimand-level closure (ADR-0150–0159) are reconfirmed, not re-litigated.
2. **Generalized Lawful-Disambiguation Dichotomy (review §3):** for any discrete latent quantity
   that is a deterministic function of frozen task semantics (true of every APC primitive
   registered to date) and any lawful oracle-free observable set — regardless of whether the
   representation map is linear, nonlinear, invertible, lossy, or delivered interactively —
   exactly one of two cases holds: (A) $H(Z\mid\mathcal O)>0$, informationally unidentifiable by
   any learner of any function class; or (B) $H(Z\mid\mathcal O)=0$, hence $Z$ is a lawfully
   computable deterministic function of $\mathcal O$ and is dominated by a zero-learned-parameter
   baseline by definition. This strictly generalizes ADR-0159's Impossibility–Dominance Dilemma
   beyond its stated linear/group-symmetric scope, closing nonlinear-representation and
   interactive-query reformulations without a fresh proof for each.
3. **The one honest loophole is closed under current scope.** Case B assumes the disambiguating
   function is tractable to state as a baseline; an intractable-but-true function could in
   principle escape into a non-trivial learning regime. This does not apply to any currently
   registered APC relation, because AGENTS.md's own invariant (oracle/deterministic controls
   must be established before crediting learned mechanisms) requires every relation's ground
   truth to already be exactly and tractably computable — that is how ground truth is generated
   and scored throughout Phase A–C. An intentionally intractable relation would be a new-primitive
   scope change, not a minimal experiment on the existing registry.
4. **Six-candidate survey (review §4):** nonlinear/interactive reformulations of routing-identity
   recovery (N1, N3) are absorbed by finding 2; a genuinely stochastic ground-truth relation (N2)
   is the one theoretically open case but requires an unauthorized new-primitive scope change and
   independently fails the relation-transfer gate; sample-efficiency-of-induction (N4),
   failure-containment (N5), and compute-scaling (N6) framings either fail the relation-transfer
   gate, are not conclusion-changing (already substantially covered by existing Correct/Wrong/None
   causal-control and Phase A2 ablation evidence), or fall outside the identifiability genre the
   task specifies.
5. **Relation-transfer gate independently re-confirmed insufficient.** The non-sealed relation
   inventory provides exactly 1 clean independent component usable for validation and 1 for
   sealed (2 total against 4 required), confirmed by two independent audits using different
   methods on the same day (ADR-0147's coupling-graph audit and ADR-0150's catalog/metadata
   audit), unremedied as of this review.
6. **Decision:** $H_{\text{admissible}}=\emptyset$. No estimand survives all five criteria.
   Per the task's own branching rule, no minimal conclusion-changing experiment is pre-registered
   and no execution-authorization question arises.
7. **Program-termination re-audit (review §7):** the oracle-free, open-world semantic/relation
   task-inference research line — the throughline connecting Phase B's architecture-level closure
   and Phase C's estimand-level closure, now extended by this review's representation-agnostic
   argument and six-candidate survey — has no remaining conclusion-changing, executable, in-scope
   research question, and is confirmed closed at the **program level** for this line. This is
   explicitly **not** a claim of APC's general impossibility, not a re-opening or invalidation of
   Phase A/A.1/A.2's positive core-separation evidence, and not a claim that a differently-scoped
   future premise (e.g., a stochastic-ground-truth relation family, pursued under separate,
   explicit scope authorization) is impossible. The relation-inventory deficit is separately
   classified as a resource/engineering gap, not a scientific impossibility, and remains an
   open, unauthorized, separately schedulable task.

**Decision:**  
Declare **`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`**. Confirm `PROGRAM_LINE_CLOSURE_CONFIRMED` for the
oracle-free task/relation-inference research line (Phase B -> Phase C throughline). Do not extend
this closure to Phase A/A.1/A.2's core-separation evidence, and do not assert APC's general
impossibility.

**Consequences:**  
- No Phase D (or any other) hypothesis, architecture, supervision scheme, relation family, or
  learning formulation is proposed or authorized by this ADR.
- `candidate=null`, `research_execution=NOT_AUTHORIZED`, sealed access remains `0`, relation
  inventory deficit remains unresolved and independently blocking.
- ADR-0001 through ADR-0160 are preserved verbatim; this ADR adds no retraction of any prior
  entry, only a new, independent finding building on their existing (unmodified) conclusions.
- A future research question that adopts a genuinely different premise (e.g., a scope change
  admitting stochastic task semantics, or a dedicated relation-inventory-expansion project) would
  require its own separate, explicit user authorization and its own charter; this ADR does not
  grant either.

**Primary Artifacts:**  
- Review Document: `docs/research/NEXT_RESEARCH_QUESTION_REVIEW_NRQ001.md`  
- Verification Record: `docs/research/NRQ001_REVIEW_RECORD.json`

## ADR-0162: NRQ-002 Constructive Falsification Experiment Finds No Counterexample to the Lawful-Disambiguation Dichotomy (`NO_COUNTEREXAMPLE_CONSTRUCTED`)

**Date:** 2026-09-13  
**Task:** NRQ-002 — Constructive Falsification Experiment for the Lawful-Disambiguation Dichotomy.
**Not a Phase C task.** Phase C remains `TERMINATED_CURRENT_CHARTER` (ADR-0160), unmodified. A
direct, targeted follow-on to NRQ-001 (ADR-0161): rather than surveying candidate estimands against
the dichotomy NRQ-001 derived, this task actively attempts to *construct* a counterexample to it,
specifically targeting the "one honest loophole" NRQ-001 §3 flagged as closed only for a single
relation's ground truth, not for arbitrary finite compositions. Filed in this ledger per
`docs/DECISIONS.md`'s "append to the latest record file" instruction, as ADR-0161 was.  
**Status:** Completed non-experimental constructive falsification attempt;
`decision: NO_COUNTEREXAMPLE_CONSTRUCTED` (sub-finding:
`BOUNDED_RESOURCE_LOOPHOLE_CLOSED_FOR_CURRENT_REGISTRY`). Research execution remains, and has
always been, `NOT_AUTHORIZED`.

**Scope and Integrity Boundary:**  
- Read-only construction/reasoning task over NRQ-001's review and JSON record, ADR-0161,
  `src/apc/primitives/composition_search.py` (the actual A1-B004 beam-search implementation), the
  A1-B004/A1-B005/A2-C004 ADR entries, and `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`.
- Zero training, optimizer construction, model initialization, dataset generation, relation
  registration, candidate construction, architecture/primitive implementation, GPU execution time,
  and sealed-partition access. Zero instances of a constructed baseline receiving task identity or
  the ground-truth `apply` that generated a target quantity.
- No ADR text (ADR-0001–ADR-0161) is rewritten, renumbered, or deleted. No charter threshold,
  floor, oracle-boundary criterion, initialization count, or relation requirement is relaxed,
  added to, or removed.

**Task instruction (four admission criteria for a counterexample):** a constructed estimand $E$
falsifies the dichotomy only if it simultaneously satisfies (1) $H(Z\mid\mathcal O)=0$
(identifiable), (2) oracle-free (no task identity or ground-truth `apply` given to the baseline),
(3) relation-transfer compatible, and (4) the strongest lawful deterministic baseline, under a
pre-fixed finite compute/description-length budget, scores below 0.95.

**Key Findings:**  
1. **Four construction attempts, each closed by a different mechanism.** (A) Composition-recipe
   identification via recipe-space combinatorics — closed **empirically**: this exact construction
   already exists in this repository as Task A1-B004 (ADR-0049), where a zero-oracle heuristic beam
   search (bank size 8, `max_depth<=3`, raw bound 584 candidates, structurally pruned far below
   that) achieves 99.62% mean exact match and 99.93% functional agreement from a 32-example
   adaptation set with zero bank expansion — the strongest lawful deterministic baseline already
   exceeds 0.95, falling directly on criterion 4. (B) Pushing recipe-space size past any pre-fixed
   budget — closed on **scope grounds**: the largest bank scale ever exercised is $N=128$
   (ADR-0065), where depth-3 enumeration is $\approx 2.1\times10^6$, trivially tractable; reaching
   $m^k>10^{12}$ requires depth $\ge 6$, never used, and would itself violate `AGENTS.md`'s
   single-workstation/no-architecture-search invariant. (C) Argument-space combinatorial blow-up —
   closed by **measured domain size**: ADR-0031 confirms registered argument domains are small
   (range(10), $\le 32$ buckets, injective), so joint combinations stay near $100$, already
   exhaustively audited as routine verification. (D) A deliberately hard (cryptographic/NP-hard-
   style) composition target — closed on the **same scope/inventory grounds as NRQ-001's candidate
   N2**: no registered primitive has cryptographic hardness properties, and manufacturing one is an
   unauthorized new-primitive scope change that is itself an unregistered relation, independently
   failing criterion 3 via the unchanged 2-of-4 relation-inventory deficit.
2. **Bounded-Resource Corollary (review §4).** Generalizing beyond the four instances: for any
   finite composition of the currently registered, non-cryptographic, structurally-pruneable APC
   primitives, at any scale this repository could exercise within its own stated invariants,
   structural pruning defeats naive combinatorial blow-up, any exploitable statistical structure
   available to a learner is equally available to a structured lawful deterministic search (no
   known statistical-query-style separation exists for these primitives), and reaching genuine
   intractability requires either an out-of-scope scale escalation or a new, unauthorized primitive
   family. This extends NRQ-001's closure of its own flagged loophole from "a single relation's
   ground truth" to "any finite composition of registered relations at in-scope depth/domain size."
3. **Relation-transfer criterion re-confirmed, not re-derived.** The 2-of-4 independent-clean-
   component deficit (ADR-0147, ADR-0150) is unchanged as of this task (same day as ADR-0161, no
   relation-registration work performed in between).
4. **Decision.** No constructed estimand survives all four criteria. Per the task's own branching
   rule, §7 of the review document records the search space and reasons for impossibility and
   re-audits the termination decision.
5. **Termination re-audit.** `PROGRAM_LINE_CLOSURE_CONFIRMED` (ADR-0161) is **reinforced, not
   reopened**: this task closes a specific gap ADR-0161 itself left open (tractability under finite
   composition) rather than reversing any of its conclusions. No component of the program-status
   table changes; the relation-inventory deficit remains a separate, unremedied, non-terminated
   engineering gap; no claim of APC's general impossibility is made or implied.

**Decision:**  
Declare **`NO_COUNTEREXAMPLE_CONSTRUCTED`**. Reaffirm `PROGRAM_LINE_CLOSURE_CONFIRMED` for the
oracle-free task/relation-inference research line, now with the composition/tractability gap
explicitly closed rather than merely unexamined. Do not extend this closure to Phase A/A.1/A.2's
core-separation evidence, and do not assert APC's general impossibility.

**Consequences:**  
- No Phase D (or any other) hypothesis, architecture, supervision scheme, relation family, or
  learning formulation is proposed or authorized by this ADR.
- `candidate=null`, `research_execution=NOT_AUTHORIZED`, sealed access remains `0`, relation
  inventory deficit remains unresolved and independently blocking.
- ADR-0001 through ADR-0161 are preserved verbatim; this ADR adds no retraction of any prior entry,
  only a new, independent construction attempt closing a gap ADR-0161 itself flagged as open.
- A future research question that adopts a genuinely different premise (e.g., a scope change
  admitting a new, deliberately hard primitive family, or a dedicated relation-inventory-expansion
  project) would require its own separate, explicit user authorization and its own charter; this
  ADR does not grant either.

**Primary Artifacts:**  
- Review Document: `docs/research/CONSTRUCTIVE_FALSIFICATION_NRQ002.md`  
- Verification Record: `docs/research/NRQ002_REVIEW_RECORD.json`

---

## ADR-0163: NRQ-003 Exact-Depth-3 Composition Search Prerequisite Check Blocked by Bundle Provenance and Model Inadequacy (`BLOCKED_BY_MODEL_ADEQUACY`)

**Date:** 2026-09-13  
**Task:** NRQ-003 — Exact-Depth-3 Irreducible Composition Search Benchmark  
**Status:** Prerequisite audit completed; `prerequisite_status: FAIL`, `decision: BLOCKED_BY_MODEL_ADEQUACY`. Depth-3 search interpretation barred.  

**Scope and Integrity Boundary:**  
- Evaluated frozen core checkpoints (`runs/phase_a1_shift_compact_structural_probe/seed_{0..4}/shared_encoder.pt`) and primitive bank checkpoints (`runs/phase_a1_composition_library_benchmark/seed_{0..4}/primitive_bank.pt`) under the pre-registered requirement that existing frozen bundles be used without unauthorized retraining, architecture changes, or sealed data access.
- Evaluated Task A1-B004 depth-2 positive controls across 6 canonical recipes (`SHIFT->SELECT`, `REVERSE->COUNT`, `COPY->SORT`, `NEGATE->SELECT`, `SHIFT->BIND`, `REVERSE->SORT`) on seed 0 without retraining.
- Zero new training updates, zero relation registrations, zero architecture modifications, zero sealed-partition access.
- No historical ADR or benchmark threshold modified.

**Key Findings:**  
1. **Broken Checkpoint Bundle Provenance:**  
   - Core checkpoints for seeds 1–4 have `token_emb.weight` shape `[36, 192]`, incompatible with current repository vocabulary size 44 (`[44, 192]`), raising `RuntimeError: size mismatch` upon loading.
   - Seed 0 core checkpoint (`[44, 192]`) was overwritten on 2026-09-13, desynchronizing its latent representation from the frozen primitive bank saved on 2026-09-04.
2. **Depth-2 Positive Control Reproduction Collapse (Model Inadequacy):**  
   - Running the depth-2 positive controls benchmark on seed 0 yielded mean oracle exact match = 0.1180, mean recovered exact match = 0.1833, and mean functional agreement = 0.4050 (0/6 passed, far below the >=0.85 / >=0.99 thresholds).
   - Because oracle recipe execution itself collapses to 11.8% EM, the failure reflects substrate degradation in the core/primitive bank coupling rather than combinatorial search limits.
3. **Enforcement of Stop Gate:**  
   - Following the mandatory prerequisite contract ("require the depth-2 positive controls to reproduce; if that prerequisite fails, record NRQ-003 as invalid/blocked by model adequacy without interpreting depth-3 search"), execution of depth-3 search is strictly blocked from interpretation.

**Decision:**  
Declare **`BLOCKED_BY_MODEL_ADEQUACY`**. Record NRQ-003 as invalid/blocked by prerequisite model adequacy failure without interpreting depth-3 search results.

**Consequences:**  
- Depth-3 composition search is not interpreted as evidence for or against ADR-0162's Bounded-Resource Corollary or program-line closure.
- The research line remains terminated under ADR-0160 (`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`) and ADR-0161 (`PROGRAM_LINE_CLOSURE_CONFIRMED`).
- Any future composition search benchmark requires an explicitly authorized bundle alignment and retraining cycle under a new charter.

**Primary Artifacts:**  
- Review Document: `docs/research/EXACT_DEPTH3_COMPOSITION_SEARCH_AUDIT_NRQ003.md`  
- Verification Record: `docs/research/NRQ003_REVIEW_RECORD.json`  
- Implementation / Runner: `src/apc/evaluation/nrq003_prerequisite_audit.py`  
- Verification Suite: `tests/test_nrq003_prerequisite_audit.py`

---

## ADR-0164: NRQ-004 Frozen Bundle Compatibility Reconstruction & Depth-2 Control Reproduction Fails on Bundle Loss (`STOP_NRQ003_BLOCKED`)

**Date:** 2026-09-13  
**Task:** NRQ-004 — Frozen Bundle Compatibility Reconstruction & Depth-2 Control Reproduction  
**Status:** Audit and reproduction completed; `decision: STOP_NRQ003_BLOCKED`. Resumption of NRQ-003 barred.  

**Scope and Integrity Boundary:**  
- Audited git history at commit `c3f291f` (Task A1-B004), run manifests, and checkpoint hashes. Reconstructed native Phase A.1 36-token vocabulary schema into a non-destructive bundle namespace (`runs/nrq004_reconstructed_bundles/`).
- Evaluated Task A1-B004 depth-2 positive controls across all 5 fixed seeds (0, 1, 2, 3, 4) on all 6 designated compositions with heuristic beam search.
- Zero new training updates, zero parameter modifications, zero sealed-partition access.
- No historical ADR or benchmark threshold modified.

**Key Findings:**  
1. **Vocabulary Schema Drift Resolved:**  
   The `RuntimeError: size mismatch` on seeds 1–4 was caused by post-A1-B004 novel operation registrations that expanded the active token table from 36 to 44 tokens. Reconstructing the 10-operation schema (`SharedCoreTokens(vocab_size=10, num_operations=10, arg_span=10)`) eliminates the mismatch without weight modification.
2. **Ceiling Reproduction on Intact Seeds (1–4) — Intrinsic Inadequacy Rejected:**  
   Seeds 1, 2, 3, and 4 achieve near-perfect performance across all 6 compositions:
   - Seed 1: Oracle EM = 1.0000, Recovered EM = 1.0000, Agreement = 1.0000 (PASS 6/6)
   - Seed 2: Oracle EM = 0.9942, Recovered EM = 0.9942, Agreement = 1.0000 (PASS 6/6)
   - Seed 3: Oracle EM = 0.9975, Recovered EM = 0.9983, Agreement = 0.9992 (PASS 6/6)
   - Seed 4: Oracle EM = 0.9883, Recovered EM = 0.9875, Agreement = 0.9992 (PASS 6/6)
   - Mean (Seeds 1–4): Recovered EM = **0.9950**, Agreement = **0.9996** (both $\gg$ 0.85 / 0.99 thresholds).  
   This definitively refutes ADR-0163's hypothesis that the APC neural substrate is intrinsically inadequate for depth-2 compositions.
3. **Causal Attribution of Seed 0 Failure to Bundle Loss:**  
   Seed 0's original 36-token core checkpoint was destructively overwritten on 2026-09-13 13:44 by an uncoordinated run with vocab 44. The original weights are unrecoverable from disk or git. Evaluated against the available 44-token core, Seed 0 exhibits latent de-synchronization: Oracle EM = 0.1192, Recovered EM = 0.1958, Agreement = 0.4125 (FAIL 0/6).
4. **Enforcement of Fail-Closed Stop Gate:**  
   Because resumption of NRQ-003 strictly requires all 5 seeds to reproduce ceiling controls, and because Seed 0 cannot be reconstituted without unauthorized retraining, the 5-seed completion criterion is not met.

**Decision:**  
Declare **`STOP_NRQ003_BLOCKED`**. Record failure cause as **`BUNDLE_LOSS`** (not intrinsic model inadequacy). Bar resumption of NRQ-003, and stop without proceeding to training or depth-3 composition search.

**Consequences:**  
- Depth-3 composition search remains barred from execution and interpretation.
- The research line remains terminated under ADR-0160 (`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`) and ADR-0161 (`PROGRAM_LINE_CLOSURE_CONFIRMED`).
- Reconstructed coherent bundles for seeds 1–4 are preserved in `runs/nrq004_reconstructed_bundles/` as reference evidence.

**Primary Artifacts:**  
- Review Document: `docs/research/FROZEN_BUNDLE_COMPATIBILITY_RECONSTRUCTION_NRQ004.md`  
- Verification Record: `docs/research/NRQ004_REVIEW_RECORD.json`  
- Reconstructed Bundles: `runs/nrq004_reconstructed_bundles/`  
- Implementation / Runner: `src/apc/evaluation/nrq004_bundle_reconstruction.py`, `scripts/nrq004_bundle_reconstruction.py`  
- Verification Suite: `tests/test_nrq004_bundle_reconstruction.py`
