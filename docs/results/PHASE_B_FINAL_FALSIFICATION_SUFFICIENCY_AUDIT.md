# PHASE-B-FINAL — Falsification Sufficiency & Research-Termination Audit

**Date:** 2026-09-13<br>
**Task:** PHASE-B-FINAL<br>
**Result:** `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`<br>
**Execution type:** artifact-only audit; no new research execution

## Decision

The present Phase-B architecture does not support its load-bearing claim: safe hard-negative retrieval and the resulting open-world/semantic-task-inference lifecycle. The conclusion is negative and terminated for this registered architecture and evidence boundary.

This is not a claim that APC in general, a different architecture, or a future independently authorized relation registry is impossible. B1's explicit-TaskSpec lifecycle PASS remains historical evidence, but it is insufficient to overcome the B2 and G1 stops below.

## Audit boundary and integrity ledger

Read-only sources were the ADR-0074--0147 decision ledgers, the four active execution plans, and the existing development artifacts listed below. No checkpoint, cache, or manifest was modified.

| Prohibited action | Count/status |
|---|---:|
| New training or optimizer update | 0 |
| New seed or initialization | 0 |
| New relation family or repair candidate | 0 |
| Candidate selection / child-bundle write | 0 |
| RG3 / REC-005 / G4 / G5 execution | 0 |
| Sealed data or sealed model-output access | 0 |

### ADR evidence ledger

| ADR range | Evidence role | Result carried into this audit |
|---|---|---|
| ADR-0074 | B1 explicit-TaskSpec lifecycle | PASS, but limited to explicit TaskSpec. |
| ADR-0075--0081 | B2, re-gate, and second diagnosis | B2 and sealed re-gate FAIL; diagnosis did not authorize a bypass. |
| ADR-0082--0091 | reproducibility, adequacy, local repairs, integration | G0/G2/G3 infrastructure/control progress; G1 relation insufficiency and G4 development integration FAIL remain. |
| ADR-0092--0105 | bundle recovery through early MIRROR investigation | RG0--RG2 PASS, RG3 FAIL; no all-init recoverable candidate. |
| ADR-0106--0125 | downstream causal/robustness interventions | no stable, sufficiently learnable repair; safety-preserving interventions remained sub-threshold. |
| ADR-0126--0133 | measurement correction and scorer-target tests | EM FAIL preserved; global scale, unique QK/position target, and single-component repair stopped. |
| ADR-0134--0137 | CD-DPCA derivation, structural validation, pilot | representation/serialization pass, but I01 pilot terminates at sequence EM 0.793945 < 0.95. |
| ADR-0138--0140 | failure localization and gradient diagnosis | position-4 attractor, aliasing credit dilution, and late saturation identified. |
| ADR-0141 | single-init causal warm-start | I01 sequence EM 0.985352 and length-10 EM 1.0; a pilot only. |
| ADR-0143--0146 | multi-init replication and identifiability | qualification 1/5; causal co-improvement 2/5; no oracle-free generic initialization or CE-only repair is identifiable. |
| ADR-0147 | G1 strict-holdout feasibility | one clean component < two required in validation and sealed_v2; `G1_RELATION_TRANSFER_STOP`. |

ADR-0142 is process guidance and does not add scientific evidence. The ADR number sequence is otherwise complete: 74 entries, ADR-0074 through ADR-0147 inclusive.

### Existing artifact ledger

Namespace inventory (read-only; temporary pytest directories and `bundles`/`staging` excluded):

- 18 restart scientific namespaces (`mirror_score_scale_precheck`, `rec004ac_metric_v2`, and REC-004AE--AT);
- 11 Post-D2 scientific namespaces (R3-001--010 plus R3-002R);
- 34 model-bundle-recovery namespaces (REC-001--004Z).

| Artifact | Evidence used | Integrity/result |
|---|---|---|
| `runs/phase_b_restart/rec004al/run_001/summary.json` | CD-DPCA I01 pilot | 0.793945 sequence EM (813/1024), below 0.95; Correct/Wrong/None causal gap 0.793945; no candidate/bundle/RG3. |
| `runs/phase_b_restart/rec004aq/run_001/summary.json` | fixed I01 warm-start causal pilot | 0.985352 sequence EM; length-10 1.0; still no candidate/bundle/RG3. |
| `runs/phase_b_restart/rec004am/run_001/summary.json` | preregistered five-init qualification | 1/5 passed; mean EM 0.950195; floor-consistency FAIL. |
| `runs/phase_b_restart/rec004ar/run_001/summary.json` | five-init matched-baseline causal replication | 2/5 simultaneous improvements; overall EM mean delta +0.018164, median 0.0, range [-0.106445, +0.191406]. |
| `runs/phase_b_restart/rec004as/run_001/summary.json` | 130 archived states / 40 cells per init | initialization × early-data interaction; no uniquely identified generic repair; sealed access 0. |
| `runs/phase_b_restart/rec004at/run_001/summary.json` | CE-only formulation identifiability | no unique permitted formulation; no new training/seed/candidate; sealed access 0. |
| `runs/phase_b_restart/mirror_score_scale_precheck/run_001/summary.json` | frozen-endpoint sensitivity | all 4 preregistered scales fail; historical and O1 parity preserved; optimizer updates 0. |
| `runs/phase_b_restart/rec004ae/run_004/summary.json` | causal localization/ablation | routing recovers under oracle but QK and position ablations do not identify one target. |
| `runs/phase_b_restart/rec004af/run_005/summary.json` | matched endpoint counterfactuals | source/baseline parity; no unique QK or position target; optimizer updates 0. |
| `runs/phase_b_b2_post_d2/r3_002r_single_family_g1/run_002/{protocol,preregistration,gradient_isolation,relation_coupling_graph}.json` | G1 feasibility and isolation | one coupled component; strict gradient isolation PASS; sealed model outputs inspected 0; G1 STOP. |

## Seven required evaluations

| Evaluation | Finding | Sufficiency judgement |
|---|---|---|
| 1. Primary hypothesis vs baseline | B1 works only with explicit TaskSpec. B2/re-gate failed. CD-DPCA's I01 improvement is not a qualifying replacement: 1/5 five-init qualification and 2/5 matched co-improvement. | The load-bearing Phase-B claim is not supported. |
| 2. Alternative explanations | Strict bundle/load checks passed before later failures. Frozen scale, QK/position, score-component, stability, and functional-role alternatives were tested; source and prediction parity were recorded. | Stale bundle, a metric-only error, simple global scale, or one isolated component do not explain away the failure. |
| 3. Causal ablation | Correct/Wrong/None controls were decisive in the CD-DPCA pilot (Correct causal gap 0.793945; Wrong/None 0.0). Oracle routing recovered the fixed endpoint's direct errors, while intervention ablations did not isolate a legal repair target. | Mechanism dependence exists, but does not meet viability/safety qualification. |
| 4. Five-seed reproducibility | Qualification: 1/5. Matched causal replication: 2/5 co-improve; I02 and I03 have negative interference. | FAIL against the preregistered 5/5 qualification and >=4/5 causal-superiority rules. |
| 5. Failure analysis | 89.4% of I01 length-10 errors localized at output position 4; 13.1% aliased stratum had 12.4x destructive-gradient dominance; archived 5-init audit identifies interaction, not a generic initial-state defect. | Failure is characterized sufficiently to reject post-hoc generic fixes; it is not repaired. |
| 6. Sensitivity and robustness | All four frozen score scales fail. Paired QK/position tests and non-degenerate position transport do not support a unique scorer target. Safety-preserving downstream alternatives are sub-threshold. | The negative result is robust to the registered finite alternatives, not a claim over untested designs. |
| 7. Unresolved uncertainty | No relation-transfer result is available because G1 lacks a second independent group; no full bundle cohort/integration/sealed B2 v2 evaluation exists. | These are explicit limits, not missing PASSes; below, each is classified by dependency. |

## Unexecuted-gate classification

| Gate/work | Classification | Evidence-based reason |
|---|---|---|
| Candidate adoption, child bundle, RG3 | Upstream STOP: scientifically unnecessary and impossible under the current contract | REC-004AM fails qualification; REC-004AR/AS/AT do not supply an authorized, unique repair. |
| REC-005 / RG4 | Upstream STOP: scientifically unnecessary and impossible under the current contract | Requires a qualified candidate and RG3. |
| REC-006--008 / RG5--RG6 / G4 | Upstream STOP: scientifically unnecessary and impossible under the current contract | Requires a coherent recovered bundle/cohort; none is eligible. |
| R3-011 and R3-012 / G5 | Upstream STOP: scientifically unnecessary and impossible under the current contract | G1 relation transfer STOP and G4 block remain; sealed evaluation cannot bypass them. |
| B-C006--014, including B3--B6 | Upstream STOP: scientifically unnecessary and impossible under the current contract | Parent Phase-B plan blocks B-C006 onward until B2_PROTOCOL_V2/G5 passes. |

No conclusion-changing, independently executable experiment remains in the existing contracts under the requested prohibitions. A new relation family, repair hypothesis, training run, seed, or sealed access would be a new research scope, not an independent continuation or a permitted way around these STOPs.

## Preserved terminal state

`candidate_selected: null`; `child_bundle: null`; `bundle_write: false`; `RG3: NOT_EXECUTED`; `REC-005: BLOCKED`; G4/G5 blocked; sealed access remains zero. Historical artifacts and negative results are preserved. The corresponding ADR is ADR-0148 and the restart plan is updated in section 10.

## Closeout authority — ADR-0149

PHASE-B-CLOSEOUT fixes the authoritative final claim classification in the
[closeout evidence ledger](PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md), with its
[freeze manifest](PHASE_B_CLOSEOUT_FREEZE_MANIFEST.json) and
[verification record](PHASE_B_CLOSEOUT_AUDIT.json).
This audit and ADR-0148 remain preserved evidence; the closeout ledger extends
their classification without changing their negative conclusion or measurements.
Phase B is `CLOSED_ARCHIVED`; G1=`STOP`, G4/G5=`BLOCKED`, RG3=`NOT_EXECUTED`,
REC-005=`BLOCKED`, candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`.
Closeout sealed-data/model-output access=0; protected sealed_v2 evaluation is unexecuted.
Historical sealed re-gate results above remain historical, not never-opened data.
The [independent Phase C charter](../research/PHASE_C_RESEARCH_CHARTER.md) is
`READY_FOR_REVIEW_NOT_APPROVED`, not authorization to restart research.
