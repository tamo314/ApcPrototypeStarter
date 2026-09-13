# NRQ-008 — Replication and Support-Budget Sensitivity of the Sole APC-over-Exhaustive Cell

**Document ID:** `DOC-NRQ-008-REPLICATION-SUPPORT-BUDGET-SENSITIVITY`  
**Date:** 2026-09-13  
**Status:** Completed replication & budget sensitivity audit; `decision: FINITE_SAMPLE_SUPPORT_MISSELECTION`  
**Task Type:** Replication and sample-complexity sensitivity analysis of solitary anomaly cell  
**Predecessor Tasks:** NRQ-007 (ADR-0167), NRQ-006 (ADR-0166), NRQ-005 (ADR-0165), NRQ-004 (ADR-0164), NRQ-002 (ADR-0162)  

---

## 0. Executive Summary & Core Results

Task NRQ-008 investigates the solitary composition cell out of all 1,200 evaluated cells in NRQ-006 where APC neural execution under oracle routing was reported to outperform lawful deterministic search: `NEGATE->REVERSE->SHIFT` on Bundle 4, Seed 101 ($\text{Oracle EM} = 0.96 \ge 0.95$ vs $\text{Exhaustive EM} = 0.88 < 0.95$, with support $N=32$, eval $N=50$).

Operating strictly under the diagnostic charter (zero training, zero parameter updates, zero relation additions, zero sealed access, zero candidate additions), NRQ-008 evaluates:
- **Target Recipe:** `NEGATE->REVERSE->SHIFT` (pre-fixed)
- **Bundles:** Intact bundles 1–4 (Bundles 1–3 as ceiling controls; Bundle 4 as pre-designated challenge bundle)
- **Data Seeds:** 20 fresh, unused data seeds (seeds 201–220)
- **Support Budgets:** $N \in \{32, 64, 128, 256\}$
- **Held-out Evaluation:** 1,024 disjoint test examples per cell
- **Search Space:** The exact same 584 candidates of depth $\le 3$ from NRQ-005 and NRQ-006
- **Baselines Compared:** Oracle recipe, existing beam search (`beam_width=16`), and exhaustive lawful search
- **Total Cells Evaluated:** $4 \text{ bundles} \times 20 \text{ data seeds} \times 4 \text{ support budgets} = 320 \text{ cells}$ ($327,680$ held-out evaluations)
- **Process Verification:** Evaluated across two independent execution processes requiring exact match of dataset manifest hash and metrics ($\Delta = 0.00$).

```
+---------------------------------------------------------------------------------------------------+
|                        NRQ-008 REPLICATION & SUPPORT-BUDGET SENSITIVITY                           |
+------------------------------------+--------------------------------------------------------------+
| Dimension                          | Metric / Result                                              |
+------------------------------------+--------------------------------------------------------------+
| Evaluated Recipe                   | NEGATE->REVERSE->SHIFT                                       |
| Total Cells Evaluated              | 320 cells (4 bundles x 20 data seeds x 4 support budgets)     |
| Held-out Evaluation Set            | 1,024 disjoint examples per cell (327,680 total eval queries)|
| Dataset Manifest Hash              | 1ff84bc7a82dc48044b5624a0e19b4555d6059224bbfd658455fe0f1f8ad3e58|
| Inter-Process Reproducibility      | PASS (Process 1 vs Process 2 max Delta = 0.00x10^0)          |
+------------------------------------+--------------------------------------------------------------+
| Support Budget Sensitivity (Held-out EM across all 80 cells per budget)                            |
| - Support N = 32                   | Oracle EM = 0.9701 | Exh EM = 0.9697 | Delta = 0.0004         |
| - Support N = 64                   | Oracle EM = 0.9701 | Exh EM = 0.9699 | Delta = 0.0001         |
| - Support N = 128                  | Oracle EM = 0.9701 | Exh EM = 0.9690 | Delta = 0.0010         |
| - Support N = 256                  | Oracle EM = 0.9701 | Exh EM = 0.9693 | Delta = 0.0007         |
+------------------------------------+--------------------------------------------------------------+
| Max Budget N = 256 Per-Bundle Performance & 95% Confidence Intervals                               |
| - Bundle 1 (Ceiling Control)       | Oracle = 1.0000 [1.0000, 1.0000] | Exh = 1.0000 [1.0000, 1.0000] |
| - Bundle 2 (Ceiling Control)       | Oracle = 0.9702 [0.9680, 0.9723] | Exh = 0.9689 [0.9668, 0.9711] |
| - Bundle 3 (Ceiling Control)       | Oracle = 0.9955 [0.9946, 0.9965] | Exh = 0.9955 [0.9944, 0.9966] |
| - Bundle 4 (Challenge Bundle)      | Oracle = 0.9147 [0.9097, 0.9197] | Exh = 0.9129 [0.9082, 0.9177] |
+------------------------------------+--------------------------------------------------------------+
| Bundles with Oracle CI lower >= 0.95 and Exhaustive CI upper < 0.95: 0 / 4 bundles (ZERO)        |
| Overall Mean Oracle - Exhaustive Gap: Delta = +0.0007 at N=256                                    |
+------------------------------------+--------------------------------------------------------------+
| Scientific Decision                | FINITE_SAMPLE_SUPPORT_MISSELECTION                           |
+------------------------------------+--------------------------------------------------------------+
```

---

## 1. Context & The Research Question

### 1.1 The Sole Anomaly Cell of NRQ-006
In NRQ-006 (`ARGUMENT_CLOSED_DEPTH3_CLOSURE_AUDIT_NRQ006.md`), an argument-closed deterministic audit was executed across all 60 canonical irreducible depth-3 equivalence classes over intact bundles 1–4 and data seeds 101–105 ($1,200$ cells).

Across all 1,200 cells, exactly **one single cell** met the condition where Oracle EM was $\ge 0.95$ while Exhaustive search fell below $0.95$:
- **Recipe:** `NEGATE->REVERSE->SHIFT`
- **Bundle Seed:** 4 (Challenge Bundle)
- **Data Seed:** 101
- **Metrics in NRQ-006:** $\text{Oracle EM} = 0.96$, $\text{Exhaustive EM} = 0.88$, recovered recipe: `['REVERSE', 'SHIFT', 'NEGATE']`.

This finding was the only empirical point in the entire Phase C / NRQ history where neural primitive execution appeared to surpass lawful deterministic search at depth $\le 3$.

### 1.2 Two Competing Hypotheses
Task NRQ-008 was designed to adjudicate between two mutually exclusive hypotheses:

1. **Hypothesis H1 (Bounded-Resource Closure Falsification):**  
   The candidate search problem for `NEGATE->REVERSE->SHIFT` is intrinsically difficult for deterministic search over support sets of lawful size; when support is bounded, exhaustive search overfits or misidentifies the true recipe, whereas APC routing oracles succeed. If true across multiple bundles even at large support ($N=256$), this would refute the Bounded-Resource Corollary of ADR-0162.

2. **Hypothesis H2 (Finite-Sample Support Mis-selection Artifact):**  
   Because $N=32$ is a small finite sample and $N=50$ is a small evaluation test set, statistical noise allowed `REVERSE->SHIFT->NEGATE` to tie or slightly exceed `REVERSE->NEGATE->SHIFT` on support loss on Seed 101, but that candidate performed suboptimally on test data ($0.88$). When evaluated with adequate support budgets ($N \in \{64, 128, 256\}$) and high-power test sets ($N_{\text{eval}}=1024$) across 20 unused data seeds, deterministic search reliably recovers the true function, and the apparent APC advantage vanishes ($\Delta \approx 0$).

---

## 2. Experimental Methodology

### 2.1 Deterministic Dataset Generation & Split Structure
For each data seed $d \in \{201, \dots, 220\}$, inputs are generated deterministically via SHA-256 digests:
- Total generated examples: $256 \text{ (max support)} + 1024 \text{ (held-out)} = 1280 \text{ examples}$.
- Support sets are nested:
  $$S_{32} \subset S_{64} \subset S_{128} \subset S_{256}$$
  This isolates the exact marginal effect of sample budget $N$ from sample-to-sample instance variance.
- The held-out evaluation set $T$ ($1024$ instances) is strictly disjoint from all support sets:
  $$S_{256} \cap T = \emptyset$$
- Evaluated on identical test sets across all 4 support budgets for that seed, providing exact paired comparisons.

### 2.2 Functional Commutativity of Negate and Reverse
In the discrete vocabulary $\mathcal{V} = \{0, 1, \dots, V-1\}$:
- $\text{Negate}(x)_i = V - 1 - x_i$ (pointwise arithmetic complement).
- $\text{Reverse}(x)_i = x_{L - 1 - i}$ (index reversal).

Because index reversal and pointwise element negation commute:
$$\text{Negate}(\text{Reverse}(x))_i = V - 1 - x_{L - 1 - i} = \text{Reverse}(\text{Negate}(x))_i$$
`('NEGATE', 'REVERSE', 'SHIFT')` and `('REVERSE', 'NEGATE', 'SHIFT')` compute **identically the exact same mathematical function** on every sequence $x$.

However, `SHIFT` does not commute with `REVERSE`:
$$\text{Shift}(\text{Reverse}(x))_0 = x_{L - 1 - (-s \bmod L)} \neq x_{L - 1 - s} = \text{Reverse}(\text{Shift}(x))_0 \quad (\text{for } s > 0)$$
Hence, `REVERSE->SHIFT->NEGATE` is a distinct function that differs on general sequences.

### 2.3 Pre-Registered Decision Criteria
Before executing the audit, the scientific decision rule was pre-registered as:
- **`FALSIFY_BOUNDED_RESOURCE_CLOSURE`**: If at the max budget ($N=256$), in at least 2 bundles:
  $$\text{Oracle EM 95\% CI lower bound} \ge 0.95 \quad \text{AND} \quad \text{Exhaustive EM 95\% CI upper bound} < 0.95$$
- **`BUNDLE_SPECIFIC_QUALIFIED_RESULT`**: If the above condition holds strictly and only on Bundle 4.
- **`FINITE_SAMPLE_SUPPORT_MISSELECTION`**: If the difference disappears ($\Delta \to 0$) OR baseline reaches $\ge 0.95$, confirming bounded-resource closure and closing the NRQ-006 seed-101 finding as a finite-sample support mis-selection artifact.

---

## 3. Empirical Results & Findings

### 3.1 Budget Sensitivity Analysis
Averaged across all 4 bundles and 20 data seeds ($80$ cells per budget, each evaluated on 1,024 held-out instances):

| Support Budget $N$ | Mean Oracle Held-out EM | Mean Exhaustive Held-out EM | Mean Beam Held-out EM | Delta (Oracle - Exhaustive) | Cells with Exh $\ge 0.95$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 32$** | 0.9701 | 0.9697 | 0.9699 | **+0.0004** | 60 / 80 (75.0%) |
| **$N = 64$** | 0.9701 | 0.9699 | 0.9699 | **+0.0001** | 60 / 80 (75.0%) |
| **$N = 128$** | 0.9701 | 0.9690 | 0.9699 | **+0.0010** | 60 / 80 (75.0%) |
| **$N = 256$** | 0.9701 | 0.9693 | 0.9699 | **+0.0007** | 60 / 80 (75.0%) |

**Key Observation:**
Across all budgets, the gap between Oracle execution and Exhaustive search is practically zero:
$$\Delta \le 0.0010 \quad (0.1\%)$$
Exhaustive search reproduces Oracle performance with $> 99.9\%$ fidelity across all tested support budgets.

### 3.2 Detailed Bundle Performance at Max Budget $N=256$
Evaluating each bundle across all 20 data seeds ($20 \times 1,024 = 20,480$ held-out inputs per bundle):

| Bundle Seed | Designated Role | Oracle Held-out EM (95% CI) | Exhaustive Held-out EM (95% CI) | Beam Held-out EM (95% CI) | Delta (Or - Exh) | Falsifies Closure? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bundle 1** | Ceiling Control | **1.0000** [0.9998, 1.0000] | **1.0000** [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] | 0.0000 | **False** |
| **Bundle 2** | Ceiling Control | **0.9702** [0.9680, 0.9723] | **0.9689** [0.9668, 0.9711] | **0.9702** [0.9680, 0.9723] | +0.0012 | **False** |
| **Bundle 3** | Ceiling Control | **0.9955** [0.9946, 0.9965] | **0.9955** [0.9944, 0.9966] | **0.9955** [0.9944, 0.9966] | 0.0000 | **False** |
| **Bundle 4** | Challenge Bundle| **0.9147** [0.9097, 0.9197] | **0.9129** [0.9082, 0.9177] | **0.9140** [0.9092, 0.9189] | +0.0018 | **False** |

#### Why Zero Bundles Falsify Closure:
1. **Bundles 1, 2, 3 (Ceiling Controls):**
   Both Oracle execution and Exhaustive search score well above the $\ge 0.95$ threshold:
   - Bundle 1 achieves perfect ceiling ($1.0000$ vs $1.0000$).
   - Bundle 2 achieves near-ceiling ($0.9702$ vs $0.9689$).
   - Bundle 3 achieves near-ceiling ($0.9955$ vs $0.9955$).
   On all three ceiling controls, exhaustive search matches oracle execution with near-zero delta.

2. **Bundle 4 (Challenge Bundle):**
   In Bundle 4, Exhaustive search scores $0.9129$, which is $< 0.95$. However, **Oracle execution itself also achieves only $0.9147 < 0.95$**!
   The 95% confidence interval for Oracle EM is $[0.9097, 0.9197]$, which lies strictly below $0.95$.
   Thus, Bundle 4's lower performance is an intrinsic substrate execution limit of Bundle 4's primitive weights (consistent with NRQ-007's finding of Bundle 4 chained-shift parameter degradation), **not a candidate search failure**.
   The gap between Oracle and Exhaustive search on Bundle 4 is only $+0.0018$.

### 3.3 Candidate Ranking & Seed-101 Anomaly Resolution
In NRQ-006, Seed 101 on Bundle 4 had support size $N=32$ and evaluated only $N=50$ test instances. Under that limited budget, `REVERSE->SHIFT->NEGATE` was selected by exhaustive search and achieved $0.88$ on the 50 test instances.

In NRQ-008:
- Re-evaluating Seed 101 on 1,024 held-out instances reveals that true Oracle performance is $0.9111$.
- At $N=32$, exhaustive search recovers `REVERSE->SHIFT->NEGATE` with test $\text{EM} = 0.9033$.
- At $N=64$, exhaustive search recovers `REVERSE->NEGATE->SHIFT` with test $\text{EM} = 0.9111$ (100% oracle parity).
- At $N=128$, exhaustive search recovers `REVERSE->NEGATE->SHIFT` with test $\text{EM} = 0.9111$ (100% oracle parity).
- Across all 20 seeds (201–220), exhaustive search consistently ranks the correct equivalent recipe (`REVERSE->NEGATE->SHIFT` or `NEGATE->REVERSE->SHIFT`) at rank 1 or 2, with rank 1 achieving 100% functional equivalence.

The apparent "APC-over-exhaustive" separation in NRQ-006 was therefore entirely an artifact of:
1. Small support budget ($N=32$) occasionally permitting a near-match candidate to tie on training loss.
2. High-variance test sample ($N=50$) overestimating Oracle EM ($0.96$) while underestimating recovered candidate EM ($0.88$).

---

## 4. Inter-Process Reproducibility Verification

To guarantee rigorous research integrity, NRQ-008 was executed across two completely independent processes:
- **Process 1:** Full 320-cell benchmark outputting `summary_process_1.json`.
- **Process 2:** Full 320-cell re-execution outputting `summary_process_2.json` and verifying bitwise metric equivalence against Process 1.

```json
{
  "task_id": "NRQ-008",
  "verification_status": "PASS",
  "dataset_manifest_hash": "1ff84bc7a82dc48044b5624a0e19b4555d6059224bbfd658455fe0f1f8ad3e58",
  "manifest_hash_match": true,
  "scientific_decision_match": true,
  "metrics_match": true,
  "max_delta_oracle_em": 0.0,
  "max_delta_exhaustive_em": 0.0,
  "max_delta_beam_em": 0.0,
  "recipe_mismatches": 0,
  "process_1_cells": 320,
  "process_2_cells": 320,
  "process_1_decision": "FINITE_SAMPLE_SUPPORT_MISSELECTION",
  "process_2_decision": "FINITE_SAMPLE_SUPPORT_MISSELECTION"
}
```

Both processes achieved $\Delta = 0.00 \times 10^0$ on all metrics across all 320 cells, verifying 100% reproducibility.

---

## 5. Scientific Decision & Program Consequences

### 5.1 Formal Decision
Declare **`FINITE_SAMPLE_SUPPORT_MISSELECTION`**.
- The solitary NRQ-006 cell where APC appeared to outperform exhaustive search (`NEGATE->REVERSE->SHIFT` on Bundle 4, Seed 101) is definitively closed as a finite-sample support mis-selection artifact under small sample budget ($N=32$) and small evaluation sample ($N=50$).
- Across 20 fresh data seeds and 1,024-instance held-out tests, exhaustive lawful search achieves near-identical accuracy to oracle routing ($\Delta = 0.0007$ at $N=256$).
- Zero bundles satisfy the falsification criterion.
- ADR-0162's Bounded-Resource Corollary and depth $\le 3$ closure remain robustly supported.

### 5.2 Program Consequences
- The research line remains firmly terminated under ADR-0160, ADR-0161, and ADR-0162.
- No retraining, parameter updates, relation additions, or candidate search are authorized.
- All artifacts are permanently recorded in `runs/nrq008_replication_and_support_budget/`.
