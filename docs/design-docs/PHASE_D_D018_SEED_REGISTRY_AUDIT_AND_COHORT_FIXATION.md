# Design Contract — D-018 Seed Registry Audit and Confirmatory Cohort Fixation

**Date:** 2026-09-15
**Status:** Static audit complete, mechanically verified, cohort fixed pre-result. No model, no
data, no training was touched to produce this document (AST/JSON reads only, identical in kind to
D-005's own audit, `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`).

## 1. Why D-013's seed-40-44 cohort cannot be reused as this task's confirmatory cohort

Per task instruction, D-013's cohort (seeds 40-44) is used in
`PHASE_D_D018_REVERSE_SHORT_SEQUENCE_REPAIR_PILOT_PREREGISTRATION.md` section 1 **only as
exploratory evidence** (the NRQ-007 per-step numbers cited there were measured on the old
reconstructed bundles seed 1-4, and the D-013/D-017 numbers were measured on seed 40-44 — neither
this task observes as new evidence). A genuinely confirmatory test of H-D2 requires a cohort that
has never been evaluated against any REVERSE-related hypothesis before this task locks its design.
Seeds 40-44 fail that bar: they are the exact cohort D-013 trained/evaluated (`SELECT->SORT->REVERSE`
was itself one of D-013's own registered target-panel classes) and D-014/D-015/D-016/D-017 have
since analyzed them repeatedly. Reusing them here would make "confirmatory" evidence partly
observed before this design was fixed.

## 2. Full existing-registration table (identical to D-005's, plus one addition)

| Use | Model seeds | Source |
|---|---|---|
| Original sealed / NRQ reconstructed bundles | `0-4` (bundle `1-4`; seed 0 lost, ADR-0164) | `SEALED_GATE_SEEDS`; `NRQ005-008_REVIEW_RECORD.json` |
| Model Bundle Recovery development / existing cohort | `10-14` | `DEFAULT_DEV_SEEDS` / `RECOVERY_DEV_SEEDS` |
| Phase B validation split | `15-19` | `NEW_VALIDATION_SEEDS` |
| Phase B re-gate sealed split | `20-24` | `DEFAULT_REGATE_SEEDS` |
| Phase B sealed V2 / D-001 rejected candidate | `30-34` | `NEW_SEALED_V2_SEEDS`; permanently forbidden to Phase D |
| **Phase D five-model cohort (D-005/D-013, now already-observed)** | **`40-44`** | `PHASE_D_D005_SEED_REGISTRY.json`; D-013 execution (ADR-0181); D-017 audit restricts to exploratory-only use |
| NRQ-005~007 historical data-seed axis | `101-105` | `NRQ005-007_REVIEW_RECORD.json` |
| NRQ-008 historical data-seed axis | `201-220` | `NRQ008_REVIEW_RECORD.json` |
| D-001 pilot evaluation-seed axis | `301-305` | `PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md` section 7; reused by D-013-D-016 |

Row 6 (`40-44`) is the one addition relative to D-005's own table — D-005 could not have listed it
(it was D-005's own output). This task adds it explicitly so a mechanical audit can enforce
non-reuse, not merely a prose statement.

**Phase D D-018 confirmatory cohort model seeds (fixed before any result): `50, 51, 52, 53, 54`.**

## 3. Mechanical verification (reusing the existing, unmodified D-005 audit function)

`src/apc/evaluation/phase_d_seed_registry.py: audit_phase_d_seed_registry` is a generic,
already-implemented, fail-closed static auditor: it reads literal seed constants by AST (no
import), reads historical provenance JSON records, and checks a candidate cohort for collisions
against both. It is not modified by this task. Two small new JSON artifacts let it check row 6
above, which its original invocation (hardcoded to `PHASE_D_D005_SEED_REGISTRY.json`) predates:

- `docs/phase_d/PHASE_D_D013_COHORT_AND_EVAL_SEEDS_PROVENANCE.json` — re-expresses
  already-recorded values (D-005's own `candidate_cohort.model_seeds` and D-017's
  `provenance.evaluation_seeds_D013_target_panel`) in the exact field names
  (`bundle_seeds_evaluated`/`data_seeds_evaluated`) the audit function's schema expects. No new
  fact is introduced; every value is copied verbatim and cited to its source.
- `docs/phase_d/PHASE_D_D018_SEED_REGISTRY.json` — D-005's registry, unmodified in its 6 source
  registrations and 4 NRQ historical-provenance entries, plus the new row-6 entry above, and a new
  `candidate_cohort.model_seeds = [50,51,52,53,54]`.

Invocation (`audit_phase_d_seed_registry(Path("docs/phase_d/PHASE_D_D018_SEED_REGISTRY.json"))`,
same function, no code change), executed 2026-09-15:

```
status: PASS
candidate_model_seeds: [50, 51, 52, 53, 54]
forbidden_model_seeds: [0,1,2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,32,33,34]
historical_bundle_model_seeds: [1,2,3,4,40,41,42,43,44]
historical_data_seeds: [101,102,103,104,105,201..220,301,302,303,304,305]
sealed_access: 0
```

No collision between `{50,51,52,53,54}` and any of `forbidden_model_seeds`,
`historical_bundle_model_seeds`, or `historical_data_seeds`. The candidate is therefore both
**unused** (appears in no prior registration or provenance record checked) and **non-sealed**
(absent from every `sealed_model_seed`-role entry).

**Manual check (outside this function's scope, since it only checks model seeds):** the new
evaluation seeds `401-405` fixed in the pilot preregistration section 10 were checked by inspection
against every value in the audit output above plus `301-305` — no collision.

## 4. Construction procedure and budget (reused unchanged from the D-001/D-005 cohort contract)

The new cohort is built by the **identical** procedure `PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`
section 3 already fixes for seeds 40-44 (`model_bundle_recovery.py: _build_stage_graph`, run as a
fresh build — no restore candidate exists for these seeds either), applied to seeds `50-54` instead.
Nothing in that contract's stage graph, per-primitive step counts, or router-calibration step count
is changed. The budget table (that contract's section 7) is reused unchanged with the same
substitution: Core pretrain 16,000 steps/model (`core_train_steps`, fallback path, no checkpoint
exists for these seeds), 4 parameterized canonical primitives x 6,000 = 24,000, 4 parameter-free
canonical primitives (COPY/REVERSE/SORT/NEGATE) x 3,000 = 12,000, 2 Branch-B novel primitives x
3,000 = 6,000, SHIFT-dedicated repair (existing R3-009 recipe, its own budget, unchanged), router
calibration 400. Per-model non-core total: 42,400 steps; 5-model total: 212,000 steps + 80,000 Core
steps (16,000 x 5) — identical numbers to the 40-44 cohort's own contract, since only the seed
values differ. This is a **separate** budget from the repair/evaluation budget fixed in the pilot
preregistration document section 9, exactly as D-001's cohort-construction budget was kept separate
from its repair budget.

The new cohort's storage namespace is fixed as `runs/phase_d_d018_five_model_cohort/seed_{50..54}/`
— independent of `runs/phase_d_d013_five_model_cohort/` and every other existing namespace; no
existing file is overwritten. Strict fresh-load verification (five `load_bundle(...,
mode="nominal")` calls plus `assert_distinct_model_identities`) is registered identically to the
40-44 cohort's own section 6.

## 5. What this document does not do

This document does not build the cohort, does not run any optimizer step, and does not authorize
training on its own — it only fixes and mechanically verifies the seed values and reuses an
existing, unmodified construction procedure by reference. Authorization is recorded in the Phase D
charter's H-D2 section and its corresponding ADR (`docs/DECISIONS_PHASE_D.md`), not here.
