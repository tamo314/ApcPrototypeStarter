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

