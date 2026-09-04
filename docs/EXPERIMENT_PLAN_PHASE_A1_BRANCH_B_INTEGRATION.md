# Experiment Plan — Phase A.1 Post-Diagnostic: Branch B Integration

## 1. Scientific Objective

Validate that the full APC continual learning lifecycle:
$$\text{Core} \to \text{Bank} \to \text{Composition} \to \text{Plasticity} \to \text{Consolidation} \to \text{Recurrence} \to \text{Learned Routing}$$
operates robustly over a single shared task-blind Stable Core using heterogeneous compact primitives.

---

## 2. Hypotheses & Controls

### H-B1: Primitive Modularity
Heterogeneous compact primitives (~18k-30k parameters) achieve ceiling performance on their designated operations when fed by a frozen shared task-blind Stable Core, while non-selected primitives remain inactive.
- **Controls**: Correct arm, Wrong argument arm, None arm.
- **Thresholds**: Correct $\ge 0.90$, Causal gap $\ge 0.50$, None $\le B_{\text{natural}} + 0.05$.

### H-B2: Compositional Scaling
Sequential application of compact primitives computes composite multi-step functions without intermediate task-conditioned core reprocessing.
- **Controls**: Oracle composition vs monolithic baseline vs search recovery.
- **Threshold**: Composition exact match $\ge 0.90$.

### H-B3: Residual Plastic Adaptation
Novel operations can be learned rapidly via temporary plastic capacity operating as a residual on top of frozen core activations.
- **Controls**: Frozen core + frozen bank vs plastic workspace.
- **Threshold**: Adaptation exact match $\ge 0.90$, temporary capacity $\le 100\text{k}$ parameters.

### H-B4: Compact Consolidation
Temporary plastic capacity can be distilled into a persistent compact primitive ($\approx$ 18k params) without forgetting previous tasks.
- **Controls**: Temporary solution vs shadow candidate vs historical task evaluation.
- **Thresholds**: Retention $\ge 0.95$, historical forgetting $\le 0.02$, temporary resource release $= 100\%$.

### H-B5: Recurrence Without Adaptation
Re-encountering a consolidated task achieves instant ceiling accuracy with zero plastic resource allocation.
- **Controls**: Fresh adaptation vs bank lookup.
- **Threshold**: Zero plastic allocation, exact match $\ge 0.90$.

### H-B6: Autonomous Sparse Execution & Learned Routing [PASSED - ADR-0061]
Replacing oracle selection with a learned top-k router conditioned on $z_{\text{task}}$ achieves autonomous end-to-end closed-loop execution, significant compute reduction, and high exact match across the entire benchmark universe without human intervention.
- **Controls**: Learned top-k router vs dense execution vs unselected primitive leakage.
- **Thresholds**: Router Top-1 selection $\ge 0.95$, closed-loop exact match $\ge 0.90$, compute savings $\ge 0.80$, unselected calls $== 0$.
- **Outcome**: Passed (Router Top-1: 99.99%, Closed-Loop EM: 99.79%, Compute Savings: 90.11%, Unselected Calls: 0, Temporary Params: 0 across 5 seeds).

---

## 3. Resource & Hardware Budget

- Workstation: GeForce RTX 5060 Ti (16 GB VRAM), AMD Ryzen 7 9800X3D.
- Max VRAM per run $\le 8\text{ GB}$.
- CPU fallback supported for all modules and unit tests.
- 5 seeds required for milestone gates.

