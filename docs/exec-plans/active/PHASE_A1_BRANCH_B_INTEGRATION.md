# Phase A.1 Post-Diagnostic: Branch B Integration Execution Plan

**Status:** Active after ADR-0046 (Diagnostic closure and Branch B adoption).

## 1. Objective

Integrate the scientifically validated **Branch B (Shared Queryable Representation + Heterogeneous Compact Primitives)** architecture into the canonical APC prototype pipeline:

1. One shared, task-blind Stable Core content encoder ($h_{\text{content}} = f(\text{content})$) generating queryable latent states.
2. A persistent bank of sparse, compact (~18k-30k parameter) cross-attention primitives carrying minimal, operation-appropriate structural inductive biases.
3. Strict sparse execution (unselected primitives receive zero forward calls).
4. Sequential progression through composition, temporary plastic adaptation, functional consolidation, recurrence reuse, and learned routing.

---

## 2. Sequence of Milestones & Tasks

```text
A1-B001: Heterogeneous Compact Primitive Bank & Production Wiring (B-M1)
   ↓
A1-B002: Unified Oracle Causal Benchmark (B-M2)
   ↓
A1-B003: Composition Library Execution (B-M3)
   ↓
A1-B004: Composition Search Baseline (B-M4)
   ↓
A1-B005: Plastic Workspace Residual Learning (B-M5)
   ↓
A1-B006: Functional Consolidation & Shadow Validation (B-M6)
   ↓
A1-B007: Recurrence & Reuse Benchmark (B-M7)
   ↓
A1-B008: Learned Routing & Full Closed Loop (B-M8)
```

---

## 3. Detailed Milestone Definitions

### B-M1: Heterogeneous Compact Primitive Bank (`A1-B001`)
Refactor the diagnostic operators (`CompactCrossPositionOperator`, `ShiftRelativeCrossPositionOperator`) into formal production primitive classes (`CrossPositionPrimitive`, `ShiftRelativePrimitive`) within `src/apc/primitives/`. Update `PrimitiveBank` to hold and orchestrate heterogeneous primitive classes with strict sparse execution guarantees.

### B-M2: Unified Oracle Causal Benchmark (`A1-B002`)
Benchmark all 8 canonical operations (4 parameter-free: COPY, REVERSE, UNIQUE, SORT; 4 parameterized: SELECT, COUNT, BIND, SHIFT) in a single integrated pipeline over the frozen shared task-blind Stable Core using oracle routing. Confirm strict causal selectivity (Correct vs Wrong vs None) across all operations.

### B-M3: Composition Library Execution (`A1-B003`)
Execute multi-step compositional recipes (e.g., `SHIFT(2) -> SELECT([0, 2])`, `REVERSE -> COUNT(k)`) by sequentially invoking primitives on transformed states without re-running task-conditioned core layers between steps.

### B-M4: Composition Search Baseline (`A1-B004`)
Implement search-based recipe recovery (beam/exhaustive search) over the primitive bank for multi-step tasks without oracle task labels, establishing the composition baseline.

### B-M5: Plastic Workspace Residual Learning (`A1-B005`)
Introduce temporary plastic capacity for novel operations (unexplained by existing primitives). The Stable Core and persistent primitives remain frozen while temporary residual capacity rapidly adapts to the new computation.

### B-M6: Functional Consolidation & Shadow Validation (`A1-B006`)
Distill the temporary plastic solution into a compact candidate primitive. Evaluate candidate in shadow mode on held-out and historical data; release temporary resources only upon passing validation.

### B-M7: Recurrence & Reuse Benchmark (`A1-B007`)
Re-introduce previously consolidated operations in lifelong sequence; verify that the system reuses installed primitives without re-triggering plastic adaptation.

### B-M8: Learned Routing & Full Closed Loop (`A1-B008`)
Replace oracle selection with a learned top-k router consuming $z_{\text{task}}$. Verify end-to-end autonomous execution, sparse compute savings, and retention across the entire benchmark universe.

---

## 4. Invariants & Rules

1. **Task-Blind Invariance**: Content representation must strictly satisfy $h_{\text{content}} = f(\text{content})$ with `max_abs_diff <= 1e-5` across different task specifications.
2. **Sparse Execution**: In top-k or oracle mode, unselected primitives must strictly receive zero forward calls (`forward_call_count == 0`).
3. **Primitive Scale**: Persistent primitives must remain compact ($\approx$ 18k-30k parameters), avoiding high-capacity dense sub-networks.
4. **Baseline-Relative Evaluation**: None-arm evaluation must follow the ADR-0044/0046 rule: $\text{None} \le B_{\text{natural}} + 0.05$.
5. **No Lookahead**: Complete and verify each task independently before moving to the next.
