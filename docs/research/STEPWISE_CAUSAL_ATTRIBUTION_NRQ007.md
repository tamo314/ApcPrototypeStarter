# NRQ-007 — Stepwise Causal Attribution of NRQ-006 Composition Failures

**Document ID:** `DOC-NRQ-007-STEPWISE-CAUSAL-ATTRIBUTION`  
**Date:** 2026-09-13  
**Status:** Completed stepwise causal attribution; `decision: ADR0166_QUALIFIED_BY_STEPWISE_CAUSAL_ATTRIBUTION` (`QUALIFIED`)  
**Task Type:** Stepwise causal attribution of composition failures and multi-bundle reconciliation  
**Predecessor Tasks:** NRQ-006 (`ADR0165_QUALIFIED_BY_FULL_REGISTRY_AUDIT`, ADR-0166), NRQ-005 (ADR-0165), NRQ-004 (ADR-0164), NRQ-002 (ADR-0162), A1-B004 (ADR-0049)  

---

## 0. Executive Summary & Core Results

Task NRQ-007 conducts the definitive stepwise causal attribution of exact-depth-3 composition failures discovered during Task NRQ-006. Operating strictly under the post-STOP-GATE diagnostic charter, NRQ-007 performs **zero new training, zero parameter updates, zero relation additions, zero sealed access, and zero candidate selection**.

Using the exact same 60 canonical equivalence classes, 4 intact reconstructed bundles (seeds 1–4), 5 data seeds (101–105), and deterministic SHA-256 splits from NRQ-006, NRQ-007:
1. **Re-aggregates all 1200 cells of NRQ-006**, reconciling the class partitions, failure counts, and documentation discrepancies.
2. **Executes a 3-arm stepwise causal intervention** across all 35 failure recipes ($35 \times 4 \times 5 = 700$ cells) comparing:
   - **(A) Continuous Hidden-State Execution:** Sequential propagation across ordered primitives through the frozen task-blind Stable Core encoder.
   - **(B) Diagnostic Reset:** Re-encoding ground truth symbolic intermediate sequences $s_k$ with the frozen core encoder to isolate upstream error accumulation from downstream execution failure.
   - **(C) Standalone Length-Matched Control:** Direct single-step execution of the same primitive on clean inputs of matching length regime ($L \in [3, 5]$ vs $L \in [6, 10]$) to isolate intrinsic capacity deficits from composition interface breakdowns.
   - **Parameterized Causal Controls:** Correct, Wrong-argument, None, and Wrong-family controls for all parameterized primitives (`SHIFT`, `COUNT`, `BIND`, `SELECT`).
3. **Applies pre-fixed mutually exclusive attribution rules** across all 4 bundles $\times$ 5 data seeds, requiring reproducibility across all cells.
4. **Applies the mandatory all-cell gate** (rather than averages) to formally amend ADR-0166.

```
+---------------------------------------------------------------------------------------------------+
|                               NRQ-007 STEPWISE CAUSAL ATTRIBUTION                                 |
+------------------------------------+--------------------------------------------------------------+
| Dimension                          | Metric / Result                                              |
+------------------------------------+--------------------------------------------------------------+
| Total Re-aggregated Cells          | 1200 cells (60 classes x 4 bundles x 5 data seeds)          |
| Canonical Class Partition          | 29 Length-Adequate (L >= 6), 31 Length-Contracted (L < 6)    |
| Mean-Threshold Failure Classes     | 35 classes (27 Length-Contracted, 8 Length-Adequate)        |
| All-Cell Gate Pass Rate            | 19 / 60 classes pass all 20 cells (41 classes fail gate)     |
| All-Cell Pass Count                | 558 / 1200 cells (46.50% pass both Oracle and Closure floor) |
+------------------------------------+--------------------------------------------------------------+
| Primary Attribution Breakdown      |                                                              |
| - Bundle-Specific Component Failure| 24 classes (cross-bundle parameter divergence)               |
| - Short-Sequence Capacity Deficit  | 7 classes (SORT immediately following SELECT)                |
| - Argument Handling Failure        | 4 classes (*->BIND->COUNT on length-1 intermediate)          |
+------------------------------------+--------------------------------------------------------------+
| Cell-Level Distribution (700 cells)| 330 Short-Sequence, 250 Argument, 26 Upstream, 15 Interface, |
|                                    | 79 Passed Cells                                              |
+------------------------------------+--------------------------------------------------------------+
| Decision on ADR-0166               | QUALIFIED (ADR0166_QUALIFIED_BY_STEPWISE_CAUSAL_ATTRIBUTION) |
+------------------------------------+--------------------------------------------------------------+
```

---

## 1. Re-aggregation of NRQ-006 1200 Cells & Documentation Reconciliation

### 1.1 Class Partition & True Cell Counts
Prior documentation in NRQ-006 and ADR-0166 cited "41 length-adequate classes" and "16 failure classes where Oracle EM collapses". Comprehensive programmatic re-aggregation of all 1200 cells (`summary_process_1.json`) reconciles this historical reporting:

1. **Structural Class Partition (60 Total):**
   - **Length-Adequate (29 classes):** Compositions where `SELECT` does not appear at step 1 or step 2. Intermediate sequence lengths remain within the Phase A.1 curriculum training bounds ($L \in [6, 10]$).
   - **Length-Contracted (31 classes):** Compositions where `SELECT` appears at step 1 or step 2, halving intermediate sequence length to $L \in \{3, 4, 5\}$.
   *(Note: The figure 41 in ADR-0166 arose from an earlier draft counting convention that misclassified 12 classes; the exact count is 29 length-adequate and 31 length-contracted).*

2. **Mean-Threshold Failure Classes (35 Total):**
   Evaluating classes against prior mean thresholds (Oracle floor $\ge 0.85$, Closure confirmation $\ge 0.95$):
   - **35 Failure Classes:** Exactly 35 classes fail at mean level (27 length-contracted classes + 8 length-adequate classes).
   - **25 Passing Classes:** Exactly 25 classes pass at mean level (21 length-adequate classes + 4 length-contracted classes: `NEGATE->SELECT->BIND`, `REVERSE->SELECT->BIND`, `SHIFT->SELECT->BIND`, `SORT->SELECT->BIND`).
   *(Note: The figure 16 in ADR-0166 only counted a subset of length-contracted classes with Oracle EM < 0.85, omitting 3 length-contracted classes with Oracle EM $\ge 0.85$ and omitting all 8 length-adequate failure classes).*

3. **All-Cell Gate (All 20 Cells Pass):**
   When the strict non-averaging all-cell gate is applied:
   - **Passing Classes:** Exactly **19 classes** pass all 20 cells (all bundles and data seeds achieve Oracle EM $\ge 0.85$ and Exhaustive EM $\ge 0.95$).
   - **Failing Classes:** **41 classes** fail the all-cell gate.
   - **Overall Cell Pass Rate:** 558 / 1200 cells (46.50%) pass both criteria; 674 / 1200 cells (56.17%) pass Oracle EM $\ge 0.85$; 560 / 1200 cells (46.67%) pass Exhaustive EM $\ge 0.95$.

---

## 2. Stepwise Causal Attribution Methodology

### 2.1 Intervention Arms per Prefix
For each of the 35 failure recipes across all 4 intact bundles and 5 data seeds ($N = 50$ test examples per cell), each prefix $k \in \{1, 2, 3\}$ is evaluated under three contrasting conditions:

1. **Condition (A) — Continuous Execution:**
   Latent representations are sequentially piped through the frozen task-blind Stable Core encoder using predicted tokens from previous steps:
   $$\hat{s}_k = \text{argmax}(\text{primitive}_k(\text{core.model.encode}(\hat{s}_{k-1})))$$
2. **Condition (B) — Diagnostic Reset:**
   The ground truth symbolic intermediate $s_{k-1}$ is re-encoded by the frozen core encoder and fed to $\text{primitive}_k$:
   $$\tilde{s}_k = \text{argmax}(\text{primitive}_k(\text{core.model.encode}(s_{k-1})))$$
   If Condition (A) fails but Condition (B) succeeds, failure is unequivocally isolated to upstream token corruption rather than primitive inability.
3. **Condition (C) — Standalone Length-Matched Control:**
   $\text{primitive}_k$ is evaluated as a standalone single-step operation on clean i.i.d. synthetic sequences matching the input length regime ($L \in [3, 5]$ for contracted, $L \in [6, 10]$ for standard). If Condition (C) fails on short lengths, the primitive possesses an intrinsic sub-curriculum capacity deficit.
4. **Parameterized Causal Controls:**
   For parameterized primitives (`SHIFT`, `COUNT`, `BIND`, `SELECT`), Condition (B) is augmented with:
   - Correct argument: $a$
   - Wrong argument: $a' \ne a$ (valid deterministic alternative)
   - None argument: unconditioned execution ($\text{arg} = \text{None}$)
   - Wrong family: alternate primitive execution ($\text{WRONG\_FAMILY\_MAP}$)
   - Causal gap: $\Delta_{\text{causal}} = \text{EM}_{\text{corr}} - \max(\text{EM}_{\text{wr\_arg}}, \text{EM}_{\text{none}}, \text{EM}_{\text{wr\_fam}})$

### 2.2 Pre-Fixed Mutually Exclusive Attribution Rules
For each cell, the earliest failing step $k$ in Continuous Execution (EM $< 0.85$ or step 3 $< 0.95$) is classified in strict hierarchical order:

1. **`ARGUMENT_HANDLING` (引数処理):**
   Step is parameterized, and either $\Delta_{\text{causal}} < 0.20$ or $\text{EM}_{\text{wr\_arg}} \ge \text{EM}_{\text{corr}}$.
2. **`SHORT_SEQUENCE_CAPACITY_DEFICIT` (短系列長能力不足):**
   Input length $L < 6$, and standalone control $\text{EM}_C < 0.85$.
3. **`UPSTREAM_ERROR_ACCUMULATION` (上流誤差蓄積):**
   Diagnostic reset succeeds ($\text{EM}_B \ge 0.85$) while continuous fails ($\text{EM}_A < 0.85$), and an upstream step $u < k$ had errors ($\text{EM}_A < 1.0$).
4. **`HIDDEN_STATE_INTERFACE` (hidden-state interface):**
   Standalone control succeeds ($\text{EM}_C \ge 0.85$), but diagnostic reset fails ($\text{EM}_B < 0.85$); or continuous fails despite 100% upstream EM.

### 2.3 Multi-Bundle Reproducibility Requirement
To establish a class-level attribution, the attribution must be **100% reproducible across all 4 bundle seeds $\times$ 5 data seeds (20/20 cells)**.
If different bundles exhibit fundamentally different outcomes (e.g. Bundle 1 passes ceiling while Bundles 2–4 fail, or modal failure mechanisms diverge), the class is classified as **`BUNDLE_SPECIFIC_COMPONENT_FAILURE`**.

---

## 3. Empirical Attribution Results

### 3.1 Class-Level Breakdown (35 Failure Classes)

| Final Attribution Category | Class Count | Share | Description |
|:---|:---:|:---:|:---|
| **`BUNDLE_SPECIFIC_COMPONENT_FAILURE`** | **24** | 68.6% | Divergent failure across bundles (e.g. Seed 1 succeeds while Seeds 2–4 fail) |
| **`SHORT_SEQUENCE_CAPACITY_DEFICIT`** | **7** | 20.0% | Intrinsic collapse of `SORT` on $L \in [3, 5]$ across all 4 bundles |
| **`ARGUMENT_HANDLING`** | **4** | 11.4% | Causal gap collapse on length-1 intermediate (`*->BIND->COUNT`) |
| **Total Failure Classes** | **35** | 100.0% | Full failure registry systematically attributed |

### 3.2 Detailed Attribution Registry

#### 1. Short-Sequence Capacity Deficit (7 Classes)
All 7 classes share the exact motif `SORT` immediately downstream of `SELECT`:
- `NEGATE->SELECT->SORT`
- `SELECT->SORT->BIND`
- `SELECT->SORT->NEGATE`
- `SELECT->SORT->REVERSE`
- `SELECT->SORT->SELECT`
- `SELECT->SORT->SHIFT`
- `SHIFT->SELECT->SORT`

*Causal Mechanism:* When `SELECT` contracts sequence length to $L \in [3, 5]$, `SORT` collapses across **all 4 bundles** in standalone length-matched control ($\text{EM}_C \le 0.133$). Even under Condition (B) (diagnostic reset with clean ground truth intermediate tokens re-encoded), `SORT` achieves $\text{EM}_B \le 0.14$. The positional attention weights learned during Phase A.1 curriculum training ($L \in [6, 10]$) cannot extrapolate downward to short sequences.

#### 2. Argument Handling on Contracted Length (4 Classes)
All 4 classes share the motif `*->BIND->COUNT`:
- `REVERSE->BIND->COUNT`
- `SELECT->BIND->COUNT`
- `SHIFT->BIND->COUNT`
- `SORT->BIND->COUNT`

*Causal Mechanism:* `BIND` outputs a single token ($L = 1$). When `COUNT` executes on an input of length 1, `COUNT` achieves high accuracy regardless of the `target` argument, because counting a single token under categorical projection collapses the counterfactual contrast ($\text{EM}_{\text{wr\_arg}} \approx \text{EM}_{\text{corr}}$), yielding subthreshold causal gap ($\Delta_{\text{causal}} < 0.20$).

#### 3. Bundle-Specific Component Failure (24 Classes)
The remaining 24 classes fail due to parameter divergence across reconstructed bundles:
- **`*->SELECT->SHIFT` & `SELECT->SHIFT->*` (6 classes):** Bundle 1's `SHIFT` primitive successfully generalizes to $L \in [3, 5]$ ($\text{EM} = 1.000$), allowing Bundle 1 to achieve near-ceiling execution. In contrast, Bundles 2, 3, and 4 collapse on short shifts ($\text{EM}_C \le 0.367$).
- **`*->COUNT->NEGATE` & `*->BIND->NEGATE` (4 classes):** Bundle 1 executes `NEGATE` on length 1 with $\text{EM} = 0.88 - 1.00$, while Bundles 2, 3, 4 collapse to $\text{EM} \le 0.10$ on length 1.
- **Chained Shifts (`SHIFT->SHIFT->SHIFT`, `REVERSE->SHIFT->SHIFT`) (2 classes):** Bundle 4 exhibits slight degradation in `SHIFT` accuracy ($\text{EM} = 0.867$), causing 3-step chained shifts to drop below the 0.95 threshold on Bundle 4 while Bundles 1 and 3 achieve 1.000.
- **Double Selects (`*->SELECT->SELECT`, `SELECT->SELECT->*`) (12 classes):** Repeated selection contracts length to $L \le 3$, producing severe bundle-dependent attention loss.

### 3.3 Cell-Level Distribution across All 700 Evaluated Cells

```
Short-Sequence Capacity Deficit : [====================] 330 cells (47.1%)
Argument Handling Failure        : [===============     ] 250 cells (35.7%)
Passed Cells                     : [=====               ]  79 cells (11.3%)
Upstream Error Accumulation      : [==                  ]  26 cells  (3.7%)
Hidden-State Interface Failure   : [=                   ]  15 cells  (2.1%)
```

---

## 4. Architectural Decision & Amendment of ADR-0166

1. **Evaluation of the All-Cell Gate:**
   Applying the strict all-cell gate without averaging, only **19 out of 60 canonical classes** pass all 20 cells (all 4 bundles and 5 data seeds). 41 classes fail at least one cell.
2. **Amendment of ADR-0166:**
   ADR-0166 previously concluded:
   > "confirms depth <= 3 closure for all 41 length-adequate classes... qualifies ADR-0165's unconditional closure claim (ADR0165_QUALIFIED_BY_FULL_REGISTRY_AUDIT)."
   
   Based on NRQ-007's exhaustive 1200-cell re-aggregation and causal attribution:
   - **Amended Decision:** **`ADR0166_QUALIFIED_BY_STEPWISE_CAUSAL_ATTRIBUTION` (`QUALIFIED`)**
   - **Correction of Registry Counts:** The length-adequate panel consists of 29 classes (not 41); 8 length-adequate classes fail; exactly 35 classes fail at mean level, and 41 classes fail at the all-cell gate.
   - **Causal Mechanism Established:** Failures are not monolithic length collapses. They divide into 7 reproducible short-sequence capacity deficit classes (`SORT` on $L < 6$), 4 argument handling classes on length-1 intermediates, and 24 bundle-divergent component failures.
   - **Reaffirmation:** Program line closure remains firmly confirmed. No further search, tuning, or retraining is authorized.

---

## 5. Primary Artifacts

- Review Document: `docs/research/STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md`
- Reaggregation Summary: `runs/nrq007_causal_attribution/reaggregation_summary.json`
- Attribution Report: `runs/nrq007_causal_attribution/nrq007_attribution_report.json`
- Evaluation Module: `src/apc/evaluation/nrq007_stepwise_causal_attribution.py`
- CLI Runner: `scripts/nrq007_stepwise_causal_attribution.py`
- Test Suite: `tests/test_nrq007_stepwise_causal_attribution.py`
