# NRQ-005 — Missing-Artifact-Robust Exact-Depth-3 Irreducible Composition Benchmark

**Document ID:** `DOC-NRQ-005-EXACT-DEPTH3-BENCHMARK`  
**Date:** 2026-09-13  
**Status:** Completed independent benchmark; `decision: EMPIRICALLY_SUPPORT_CLOSURE_DEPTH_LE_3`  
**Task Type:** Empirical composition search benchmark and constructive boundary evaluation  
**Predecessor Tasks:** NRQ-004 (`STOP_NRQ003_BLOCKED`, ADR-0164), NRQ-003 (`BLOCKED_BY_MODEL_ADEQUACY`, ADR-0163), NRQ-002 (`NO_COUNTEREXAMPLE_CONSTRUCTED`, ADR-0162), A1-B004 (ADR-0049)  

---

## 0. Executive Summary & Core Results

Task NRQ-005 executes an independent replacement benchmark for exact-depth-3 irreducible composition search, resolving the blocker encountered in NRQ-003 without treating NRQ-003 as resumed or completed. In accordance with NRQ-004's causal findings ([ADR-0164](docs/DECISIONS_PHASE_C.md#adr-0164-nrq-004-frozen-bundle-compatibility-reconstruction--depth-2-control-reproduction-fails-on-bundle-loss-stop_nrq003_blocked)), Seed 0 exclusion was pre-fixed to historical artifact loss (checkpoint overwrite on 2026-09-13). Evaluation proceeded across all four integrity-verified coherent seeds (1, 2, 3, 4) in `runs/nrq004_reconstructed_bundles/` under strict Phase A.1 36-token schema constraints with zero new training updates, zero parameter modifications, zero relation additions, and zero sealed data access.

### Key Empirical Findings

1. **Symbolic Probe Guarantees Exact-Depth-3 Irreducibility Prior to Model Evaluation:**  
   Before inspecting any model output, a symbolic interpreter probe over diverse inputs established a panel of six exact-depth-3 recipes that exhibit **zero** functional equivalents across all 72 possible depth $\le 2$ candidates from the 8-primitive registry:
   - `SHIFT -> REVERSE -> SELECT` (Irreducible, 0 matches in depth $\le 2$)
   - `SHIFT -> NEGATE -> SELECT` (Irreducible, 0 matches in depth $\le 2$)
   - `REVERSE -> NEGATE -> SELECT` (Irreducible, 0 matches in depth $\le 2$)
   - `SHIFT -> REVERSE -> BIND` (Irreducible, 0 matches in depth $\le 2$)
   - `NEGATE -> SHIFT -> SELECT` (Irreducible, 0 matches in depth $\le 2$)
   - `REVERSE -> SHIFT -> BIND` (Irreducible, 0 matches in depth $\le 2$)  
   Six reducible depth-3 recipes and the canonical six Task A1-B004 depth-2 recipes were evaluated in parallel as positive and negative controls.

2. **Oracle Floor Passed with Flying Colors:**  
   The neural substrates of all four coherent seeds reliably execute depth-3 compositions under oracle calls:
   - **Irreducible Depth-3 Mean Oracle EM:** **0.9905** (Floor threshold $\ge 0.85$ passed across all seeds: Seed 1 = 1.0000, Seed 2 = 0.9907, Seed 3 = 0.9997, Seed 4 = 0.9717).
   - **Controls Mean Oracle EM:** Reducible depth-3 = **0.9933**; Canonical depth-2 = **0.9961**.

3. **Strongest Deterministic Baseline Reaches Near-Ceiling Performance ($\ge 0.95$ Across All Bundles):**  
   The strongest lawful deterministic baseline — exhaustive lawful search over the entire 584-candidate depth $\le 3$ space — achieves near-perfect test functional exact match on disjoint held-out inputs ($N=100$) from a small support set ($N=32$):
   - **Exhaustive Search Mean Test EM:** **0.9904** ($\ge 0.95$ across every tested bundle: Seed 1 = 1.0000, Seed 2 = 0.9903, Seed 3 = 0.9997, Seed 4 = 0.9717).
   - Structural pruning reduces the raw 584 candidates to an average of **86 evaluated candidates** (498 pruned by length and argument constraints before neural execution).
   - Mean search wall time is **1.28 seconds**; peak memory is **1.8 MB**.

4. **Beam Search Truncation vs Beam-Budget Sensitivity:**  
   The existing heuristic beam search algorithm configured at `beam_width=16` achieves **0.3660** mean EM on the irreducible panel because intermediate prefixes that do not match the final target length receive neutral priority and are truncated before depth-3 expansion. When the beam budget is expanded, functional EM scales monotonically:
   - Width 1: EM = 8.83% (10.7 candidates evaluated)
   - Width 4: EM = 11.33% (27.8 candidates evaluated)
   - Width 8: EM = 41.33% (49.0 candidates evaluated)
   - Width 16: EM = 69.67% (58.3 candidates evaluated)
   - Width 32: EM = 83.50% (62.3 candidates evaluated)
   - Width 64: EM = **100.00%** (64.7 candidates evaluated)

5. **Operational Verdict:**  
   **`EMPIRICALLY_SUPPORT_CLOSURE_DEPTH_LE_3`**  
   Because the strongest deterministic baseline achieves $\ge 0.95$ across all intact bundle seeds and data seeds on the exact-depth-3 irreducible composition benchmark, depth-3 compositions of the current 8-primitive registry remain completely within the tractable reach of lawful deterministic search. ADR-0162's closure is **not rejected**; rather, its Bounded-Resource Corollary is empirically supported and confirmed for the current registry restricted to depth $\le 3$.

---

## 1. Protocol Specification & Governance Boundary

### 1.1 Replacement Protocol Framing
- NRQ-003 was terminated under ADR-0163 (`BLOCKED_BY_MODEL_ADEQUACY`). Resumption of NRQ-003 was barred under ADR-0164 (`STOP_NRQ003_BLOCKED`).
- NRQ-005 operates as a standalone replacement protocol with a pre-fixed exclusion of Seed 0 attributed to historical artifact loss (checkpoint overwrite on 2026-09-13). It does not resume, reopen, or retroactively alter NRQ-003.

### 1.2 Integrity Invariants Enforced
- **Zero Training:** No optimizer steps, backpropagation, or parameter updates.
- **Zero Relation Additions:** Restricted strictly to the 8 canonical primitives established at Task A1-B004 (`SELECT`, `COUNT`, `BIND`, `SHIFT`, `COPY`, `REVERSE`, `SORT`, `NEGATE`).
- **Zero Sealed Data Access:** All evaluations run strictly on synthetic programmatic data.
- **Lawful Search Protocol:** Zero access to oracle metadata (`example.oracle_metadata`, `example.program`, or `step.operation`). Primitive arguments are resolved purely from model-visible `task_spec` dictionaries (`step.arguments`).

---

## 2. Symbolic Probe & Panel Design

Before executing neural models, a symbolic interpreter probe evaluated all candidate operation sequences against all 72 possible depth $\le 2$ candidates ($8$ singletons + $64$ pairs).

### 2.1 Irreducible Exact-Depth-3 Panel
A recipe is exact-depth-3 irreducible if and only if no depth $\le 2$ sequence achieves identical outputs on all probe inputs:

| Recipe | Operation Types | Length Transformation | Matching Depth $\le 2$ Candidates | Irreducibility Status |
|:---|:---|:---:|:---:|:---:|
| `SHIFT -> REVERSE -> SELECT` | Cyclic rotation + Inversion + Subsetting | $L \to L \to L \to \lfloor L/2 \rfloor$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |
| `SHIFT -> NEGATE -> SELECT` | Cyclic rotation + Value negation + Subsetting | $L \to L \to L \to \lfloor L/2 \rfloor$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |
| `REVERSE -> NEGATE -> SELECT` | Inversion + Value negation + Subsetting | $L \to L \to L \to \lfloor L/2 \rfloor$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |
| `SHIFT -> REVERSE -> BIND` | Cyclic rotation + Inversion + Key lookup | $L \to L \to L \to 1$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |
| `NEGATE -> SHIFT -> SELECT` | Value negation + Cyclic rotation + Subsetting | $L \to L \to L \to \lfloor L/2 \rfloor$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |
| `REVERSE -> SHIFT -> BIND` | Inversion + Cyclic rotation + Key lookup | $L \to L \to L \to 1$ | None (0 / 72) | **EXACT_DEPTH3_IRREDUCIBLE** |

### 2.2 Control Panels
- **Reducible Depth-3 Controls:**
  - `COPY -> SHIFT -> SELECT` $\equiv$ `SHIFT -> SELECT` (depth 2)
  - `REVERSE -> REVERSE -> COUNT` $\equiv$ `COUNT` (depth 1)
  - `NEGATE -> NEGATE -> SELECT` $\equiv$ `SELECT` (depth 1)
  - `SORT -> SORT -> SELECT` $\equiv$ `SORT -> SELECT` (depth 2)
  - `REVERSE -> NEGATE -> SORT` $\equiv$ `NEGATE -> SORT` (depth 2)
  - `SHIFT -> REVERSE -> SORT` $\equiv$ `SORT` (depth 1)
- **Canonical Depth-2 Controls (Task A1-B004):**
  - `SHIFT -> SELECT`, `REVERSE -> COUNT`, `COPY -> SORT`, `NEGATE -> SELECT`, `SHIFT -> BIND`, `REVERSE -> SORT`.

---

## 3. Comparative Benchmark Results

Evaluated across 4 bundle seeds (1, 2, 3, 4) and 5 fixed data seeds (101, 102, 103, 104, 105), with support set $N=32$ and disjoint held-out test set $N=100$.

### 3.1 Aggregate Performance by Panel

| Panel Type | Oracle Mean EM | Beam Search Mean EM | Exhaustive Search Mean EM | Exhaustive Recipe Recovery |
|:---|:---:|:---:|:---:|:---:|
| **Exact-Depth-3 Irreducible** | **0.9905** | 0.3660 | **0.9904** | 78.33% |
| **Reducible Depth-3 Controls** | **0.9933** | 0.8123 | **0.9974** | 93.33% |
| **Canonical Depth-2 Controls** | **0.9961** | 0.9950 | **0.9964** | 100.00% |

### 3.2 Performance per Bundle Seed on Irreducible Depth-3 Panel

| Bundle Seed | Bundle Provenance | Oracle EM | Beam Search EM | Exhaustive Search EM | Baseline $\ge 0.95$? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 1** | Coherent Reconstructed (`runs/nrq004_...`) | **1.0000** | 0.6920 | **1.0000** | **YES** |
| **Seed 2** | Coherent Reconstructed (`runs/nrq004_...`) | **0.9907** | 0.2423 | **0.9903** | **YES** |
| **Seed 3** | Coherent Reconstructed (`runs/nrq004_...`) | **0.9997** | 0.2570 | **0.9997** | **YES** |
| **Seed 4** | Coherent Reconstructed (`runs/nrq004_...`) | **0.9717** | 0.2727 | **0.9717** | **YES** |
| **All Seeds Mean** | — | **0.9905** | **0.3660** | **0.9904** | **YES (4 / 4 passed)** |

---

## 4. Search Complexity, Memory, and Budget Sensitivity

### 4.1 Candidate Space Accounting
- Total candidates in $\bigcup_{d=1}^3 \mathcal{O}^d$: $8^1 + 8^2 + 8^3 = 8 + 64 + 512 = \mathbf{584}$.
- **Structural Length and Argument Pruning:** Of the 584 candidates, an average of **498** are pruned without neural execution.
- **Candidates Evaluated:** Exactly **86 candidates** undergo forward passes on the support set.
- **Compute Budget:** Exhaustive evaluation completes in **1.28 seconds** per task on a single CPU core, allocating **1.8 MB** peak memory.

### 4.2 Beam-Budget Sensitivity Analysis
Evaluated on Bundle Seed 1 across beam widths $W \in \{1, 4, 8, 16, 32, 64\}$:

| Beam Width ($W$) | Mean Recipe Recovery | Mean Test Functional EM | Mean Evaluated Candidates | Mean Pruned Candidates | Mean Search Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.00% | 0.0883 | 10.7 | 6.8 | 0.170 |
| **4** | 0.00% | 0.1133 | 27.8 | 20.7 | 0.595 |
| **8** | 0.00% | 0.4133 | 49.0 | 36.0 | 0.909 |
| **16** | 0.00% | 0.6967 | 58.3 | 62.7 | 1.233 |
| **32** | 16.67% | 0.8350 | 62.3 | 117.0 | 1.191 |
| **64** | **50.00%** | **1.0000** | 64.7 | 147.0 | 1.378 |

*Observation:* At $W=64$, heuristic beam search recovers **100.0%** test functional EM, matching exhaustive search while evaluating only 64.7 candidates on average.

---

## 5. Architectural Decision & Relation to ADR-0162

1. **Oracle Floor Confirmed:**  
   The neural representations and primitive composition mechanics of APC are fully capable of executing exact-depth-3 compositions (mean Oracle EM = 0.9905 $\ge 0.85$). Stoppage under `DEPTH_COMPOSITION_EXECUTION_FAILURE` does not apply.

2. **Falsification Criterion Evaluated:**  
   ADR-0162's closure of the Lawful-Disambiguation Dichotomy would be rejected if and only if APC side achieved $\ge 0.95$ while the strongest deterministic baseline failed ($< 0.95$).  
   - Strongest Deterministic Baseline: Exhaustive Lawful Search.  
   - Result: Strongest deterministic baseline achieves **0.9904** mean EM (and $\ge 0.95$ across all 4 bundle seeds and all 5 data seeds).  
   - Falsification condition fails: the deterministic baseline easily solves the depth-3 composition space.

3. **Conclusion:**  
   **`EMPIRICALLY_SUPPORT_CLOSURE_DEPTH_LE_3`**  
   The exact-depth-3 composition benchmark provides direct empirical evidence confirming ADR-0162's Bounded-Resource Corollary for the current 8-primitive registry restricted to depth $\le 3$. Because the hypothesis space is small ($584$ candidates) and heavily constrained by lawful structural properties ($86$ viable candidates), an oracle-free deterministic baseline solves the relation identification problem at ceiling accuracy without neural task-inference capacity.

---

## 6. Primary Artifacts

- Review Document: `docs/research/EXACT_DEPTH3_COMPOSITION_BENCHMARK_NRQ005.md`
- Verification Record: `docs/research/NRQ005_REVIEW_RECORD.json`
- Run Summary: `runs/nrq005_exact_depth3_benchmark/summary.json`
- Implementation: `src/apc/evaluation/nrq005_exact_depth3_benchmark.py`
- CLI Runner: `scripts/nrq005_exact_depth3_benchmark.py`
- Test Suite: `tests/test_nrq005_exact_depth3_benchmark.py`
- Architectural Decision: `ADR-0165` (in `docs/DECISIONS_PHASE_C.md`)
