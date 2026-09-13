# NRQ-003 — Exact-Depth-3 Irreducible Composition Search Benchmark: Prerequisite Audit and Model Adequacy Gate

**Document ID:** `DOC-NRQ-003-PREREQUISITE-AUDIT`  
**Date:** 2026-09-13  
**Status:** Completed prerequisite verification; `decision: BLOCKED_BY_MODEL_ADEQUACY`  
**Task Type:** Empirical benchmark prerequisite audit & stop gate enforcement  
**Task Execution:** `BLOCKED_BY_MODEL_ADEQUACY` (depth-3 search execution barred without interpretation per prerequisite contract)  
**Predecessor Tasks:** NRQ-001 (`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`, ADR-0161), NRQ-002 (`NO_COUNTEREXAMPLE_CONSTRUCTED`, ADR-0162)  

---

## 0. Task Context and Prerequisite Contract

Task NRQ-003 was proposed to benchmark exact-depth-3 irreducible composition search against the Bounded-Resource Corollary formulated in NRQ-002 (ADR-0162), comparing heuristic beam search against exhaustive search (584 candidates) across 5 fixed seeds (0, 1, 2, 3, 4) with support $N=32$.

However, the task specification established an explicit, fail-closed prerequisite gate:

> *"First validate the frozen core/primitive-bank bundle provenance and require the depth-2 positive controls to reproduce; if that prerequisite fails, record NRQ-003 as invalid/blocked by model adequacy without interpreting depth-3 search."*

Furthermore, the task rules strictly forbid unauthorized training, relation registration, or architectural modification: existing frozen core and primitive bank checkpoints must be used as-is without retraining.

---

## 1. Frozen Bundle Provenance Audit

An exhaustive audit of the frozen core checkpoints (`runs/phase_a1_shift_compact_structural_probe/seed_*/shared_encoder.pt`) and primitive bank checkpoints (`runs/phase_a1_composition_library_benchmark/seed_*/primitive_bank.pt`) was conducted.

### 1.1 Checkpoint Inspection Matrix

| Seed | Core Path | Core SHA256 | `token_emb` Shape | Expected Shape | Bank SHA256 | Provenance Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | `runs/.../seed_0/shared_encoder.pt` | `739e383b...` | `[44, 192]` | `[44, 192]` | `8fc10033...` | **BROKEN** (Latent distribution mismatch) |
| **1** | `runs/.../seed_1/shared_encoder.pt` | `1518d334...` | `[36, 192]` | `[44, 192]` | `c87ae13a...` | **BROKEN** (`size mismatch` RuntimeError) |
| **2** | `runs/.../seed_2/shared_encoder.pt` | `682ba287...` | `[36, 192]` | `[44, 192]` | `36fc80fd...` | **BROKEN** (`size mismatch` RuntimeError) |
| **3** | `runs/.../seed_3/shared_encoder.pt` | `676b9cc1...` | `[36, 192]` | `[44, 192]` | `00912ee9...` | **BROKEN** (`size mismatch` RuntimeError) |
| **4** | `runs/.../seed_4/shared_encoder.pt` | `6cbfd9dd...` | `[36, 192]` | `[44, 192]` | `69fa7e2d...` | **BROKEN** (`size mismatch` RuntimeError) |

### 1.2 Incompatibility Root Cause

1. **Vocabulary Schema Drift (Seeds 1–4):**  
   The current repository architecture (`SharedCoreTokens` / `build_shared_encoder_architecture`) requires a token embedding table of shape `[44, 192]`, reflecting 10 base tokens plus 34 structural and extended operation tokens. Checkpoints for seeds 1, 2, 3, and 4 preserve an earlier 36-token vocabulary (`[36, 192]`). Loading these weights raises:
   ```text
   RuntimeError: Error(s) in loading state_dict for SharedCoreModel:
       size mismatch for token_emb.weight: copying a param with shape torch.Size([36, 192]) from checkpoint, the shape in current model is torch.Size([44, 192])
   ```
   The runner framework in `_get_or_train_frozen_shared_core` catches `RuntimeError` and silently initiates an unauthorized 16,000-step pretraining loop (`train_probe_encoder`). Under the frozen-bundle constraint, executing retraining is strictly prohibited.

2. **Bundle Latent De-synchronization (Seed 0):**  
   While seed 0 possesses a 44-token embedding table, timestamp inspection confirms `shared_encoder.pt` was rewritten on 2026-09-13 13:44:21, whereas `primitive_bank.pt` dates from 2026-09-04 15:29:16. The primitive bank was trained on top of a different latent representation, destroying the functional coupling between encoder and primitives.

---

## 2. Depth-2 Positive Controls Reproduction Audit (Seed 0)

To determine whether the frozen bundle for seed 0 could nevertheless execute the positive controls, the Task A1-B004 depth-2 composition benchmark was executed without retraining across the 6 historical canonical recipes:
- `SHIFT -> SELECT`
- `REVERSE -> COUNT`
- `COPY -> SORT`
- `NEGATE -> SELECT`
- `SHIFT -> BIND`
- `REVERSE -> SORT`

### 2.1 Measured Results

| Composition Recipe | Oracle EM | Recovered EM | Functional Agreement | Threshold Met (EM $\ge 0.85$, Agree $\ge 0.99$) |
|:---|:---:|:---:|:---:|:---:|
| `SHIFT -> SELECT` | 0.0100 (1.0%) | 0.0100 (1.0%) | 1.0000 (100.0%) | **FAIL** |
| `REVERSE -> COUNT` | 0.2660 (26.6%) | 0.4280 (42.8%) | 0.1080 (10.8%) | **FAIL** |
| `COPY -> SORT` | 0.0220 (2.2%) | 0.1000 (10.0%) | 0.0440 (4.4%) | **FAIL** |
| `NEGATE -> SELECT` | 0.0380 (3.8%) | 0.0380 (3.8%) | 1.0000 (100.0%) | **FAIL** |
| `SHIFT -> BIND` | 0.3640 (36.4%) | 0.4220 (42.2%) | 0.2740 (27.4%) | **FAIL** |
| `REVERSE -> SORT` | 0.0080 (0.8%) | 0.1020 (10.2%) | 0.0040 (0.4%) | **FAIL** |
| **Overall Mean** | **0.1180 (11.8%)** | **0.1833 (18.3%)** | **0.4050 (40.5%)** | **FAIL (0 / 6 Passed)** |

*Historical A1-B004 baseline for comparison:* Mean Recovered EM = **0.9962 (99.62%)**, Functional Agreement = **0.9993 (99.93%)**.

### 2.2 Functional Collapse and Model Inadequacy

Crucially, **oracle exact match collapses to 0.1180 (11.8%)**. Even when provided with the exact ground-truth recipe, the composite execution fails in 88.2% of evaluation instances. This proves conclusively that the failure is not an algorithmic failure of heuristic beam search or candidate pruning, but a fundamental defect in the underlying neural substrate (**Model Inadequacy**).

---

## 3. Enforcement of Stop Gate

The prerequisite contract mandates:
> *"if that prerequisite fails, record NRQ-003 as invalid/blocked by model adequacy without interpreting depth-3 search."*

1. **Prerequisite Check:** **`FAIL`** (both bundle provenance and positive control reproduction failed).
2. **Execution Bar:** Evaluating depth-3 recipes under a corrupted core/bank bundle would produce uninterpretable noise, falsely attributing execution substrate collapse to combinatorial search intractability.
3. **Integrity Rule:** In accordance with scientific integrity and the pre-registered protocol, depth-3 search execution and interpretation are **strictly barred**.

---

## 4. Decision and Summary

- **Task Decision:** **`BLOCKED_BY_MODEL_ADEQUACY`**
- **Prerequisite Status:** **`FAIL`**
- **Depth-3 Search Interpretation:** **`NONE (BARRED)`**
- **ADR Reference:** `ADR-0163`
- **Audit Record:** `docs/research/NRQ003_REVIEW_RECORD.json`
- **Verification Suite:** `tests/test_nrq003_prerequisite_audit.py`

Any future composition search benchmark will require a newly authorized bundle retraining and alignment cycle under an explicitly approved charter.
