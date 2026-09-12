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
