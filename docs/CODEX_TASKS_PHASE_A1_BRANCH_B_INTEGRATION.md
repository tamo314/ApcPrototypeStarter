# Codex Tasks — Phase A.1 Post-Diagnostic: Branch B Integration

This document defines the task sequence `A1-B001` through `A1-B008` superseding the historical `A1-R006` through `A1-R022` sequence, in accordance with ADR-0046.

---

## A1-B001 — Heterogeneous Compact Primitive Bank & Production Wiring

### Goal
Integrate the empirically validated compact operators into the formal production primitives package (`src/apc/primitives/`) and establish bank orchestration with sparse execution guarantees.

### Work
1. Define a shared base class `PrimitiveBase` providing common bookkeeping (`usage_count`, `forward_call_count`, parameter accounting, freeze/unfreeze).
2. Retain `PointwisePrimitive` (formerly `Primitive`) for legacy pointwise operations.
3. Implement `CrossPositionPrimitive` encapsulating `CompactCrossPositionOperator` with argument encoder integration.
4. Implement `ShiftRelativePrimitive` encapsulating `ShiftRelativeCrossPositionOperator` with modular relative position bias.
5. Upgrade `PrimitiveBank` to manage heterogeneous primitive instances seamlessly, registering families and arguments.
6. Provide execution hooks ensuring strict non-selected zero-call accounting.

### Acceptance Criteria
- `test_heterogeneous_primitive_bank.py` passes all unit tests.
- Parameter count regression verified:
  - `CrossPositionPrimitive`: ~18k-21k parameters.
  - `ShiftRelativePrimitive`: 18,282 parameters.
- Unselected primitives strictly record `forward_call_count == 0` during forward execution.
- Frozen state enforcement: freezing the bank freezes all constituent primitives.
- Code passes `ruff` and `mypy` without errors.

---

## A1-B002 — Unified Oracle Causal Benchmark

### Goal
Evaluate all 8 canonical operations (4 parameter-free: COPY, REVERSE, UNIQUE, SORT; 4 parameterized: SELECT, COUNT, BIND, SHIFT) using oracle selection over a single frozen shared task-blind Stable Core.

### Acceptance Criteria
- Multi-seed benchmark (5 seeds) on unseen evaluations.
- Overall Correct exact match $\ge 0.90$.
- Parameter-free operations Correct exact match $\ge 0.95$.
- Parameterized operations:
  - SELECT, COUNT, BIND, SHIFT Correct exact match $\ge 0.90$.
  - Causal gap $\ge 0.50$.
  - None arm $\le B_{\text{natural}} + 0.05$.
- Task-blind maximum absolute difference $\le 10^{-5}$.
- STOP GATE.

---

## A1-B003 — Composition Library Execution

### Goal
Execute sequential recipes of existing primitive calls (e.g. `SHIFT -> SELECT`, `REVERSE -> COUNT`) by piping latent activations through ordered primitive executions without re-running task-conditioned core layers.

### Acceptance Criteria
- Designated composition accuracy $\ge 0.90$.
- No temporary plastic capacity allocated.
- Non-participating primitives receive zero forward calls.
- STOP GATE.

---

## A1-B004 — Composition Search Baseline

### Goal
Recover composition recipes for novel composite tasks without oracle primitive identity using heuristic/beam search over the primitive bank.

### Acceptance Criteria
- Recovered recipe accuracy $\ge 0.85$.
- Recovered recipe functionally matches oracle composition.
- Zero bank expansion.

---

## A1-B005 — Plastic Workspace Residual Learning

### Goal
Verify that novel operations (unsolvable by existing bank primitives or compositions) trigger temporary plastic capacity, which adapts rapidly as a residual without updating the frozen Stable Core or persistent primitives.

### Acceptance Criteria
- Stable Core and persistent primitives remain 100% frozen (`requires_grad == False`).
- Temporary capacity achieves novel task accuracy $\ge 0.90$.
- Memory/compute accounting strictly isolates temporary parameters.
- STOP GATE.

---

## A1-B006 — Functional Consolidation & Shadow Validation

### Goal
Distill temporary plastic solutions into compact candidate primitives (~18k params). Evaluate candidates in shadow mode; release temporary resources only upon passing validation.

### Acceptance Criteria
- Consolidated candidate achieves $\ge 95\%$ of temporary plastic accuracy.
- Zero degradation on historical tasks ($\le 2\%$ forgetting).
- Temporary plastic capacity released completely upon successful shadow validation.
- STOP GATE.

---

## A1-B007 — Recurrence & Reuse Benchmark [PASSED - ADR-0052]

### Goal
Demonstrate that when previously learned operations reappear in a lifelong sequence, the system retrieves and executes the installed primitive without allocating plastic capacity.

### Acceptance Criteria
- Recurrent task accuracy $\ge 0.90$ immediately without adaptation. **[PASSED: 99.15% overall mean exact match across 5 seeds: SWAP_PAIRS 100.0%, INVERT_HALF 98.3%]**
- Temporary parameter allocation $= 0$. **[PASSED: exactly 0 temporary parameters allocated throughout recurrence]**
- Baseline and Control verification. **[PASSED: unconsolidated bank EM <= 0.05%, fresh adaptation requires 500 steps and 17,098 parameters achieving only 90.30%]**

---

## A1-B007X — Discovery-to-Compact Consolidation Gate [PASSED - ADR-0060]

See: `docs/CODEX_TASKS_A1_B007X_DISCOVERY_COMPRESSION.md`, `docs/exec-plans/active/A1_B007X_DISCOVERY_COMPRESSION.md`, and `docs/results/A1_B007X_DISCOVERY_COMPRESSION_RESULT.md`.
Directly tests large temporary discovery workspace against compact persistent primitives ($P_{\text{temp}} \gg P_{\text{persistent}}$).
Tasks: `A1-B007X-001` through `A1-B007X-008` successfully resolved and audited under **Compressibility only** verdict (8.04x compression, 97.5% retention, 0.0% forgetting, 97.1% fresh-runtime recurrence, 0 adaptation steps, 0 temp params; direct compact learning equally effective). Authorizes unblocking of Task A1-B008.

---


## A1-B008 — Learned Routing & Full Closed Loop

### Goal
Replace oracle routing with a learned top-k router conditioned on $z_{\text{task}}$. Demonstrate autonomous sparse execution, compute savings, and end-to-end continual learning.

### Acceptance Criteria
- Top-k router selects oracle-required primitives with high accuracy ($\ge 95\%$).
- Active compute significantly lower than dense execution.
- Full lifelong benchmark passes without human intervention.
