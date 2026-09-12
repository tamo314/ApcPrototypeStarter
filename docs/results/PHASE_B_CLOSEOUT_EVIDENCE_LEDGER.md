# PHASE-B-CLOSEOUT — Frozen evidence ledger and lessons

Date: 2026-09-13. Decision: `PHASE_B_CLOSED_NEXT_RESEARCH_CHARTER_READY`.
This is the authoritative final claim classification for ADR-0074 through ADR-0148,
extending the preserved [termination audit](PHASE_B_FINAL_FALSIFICATION_SUFFICIENCY_AUDIT.md).
It does not replace any historical measurement or ADR-0148's conclusion.
The independent [Phase C charter](../research/PHASE_C_RESEARCH_CHARTER.md) is ready
for review, not approved for research execution.

## Terminal state

| Field | Final value |
|---|---|
| Phase B scientific decision | `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE` |
| Administrative state | `CLOSED_ARCHIVED` |
| candidate_selected | `null` |
| child_bundle | `null` |
| bundle_write | `false` |
| RG3 | `NOT_EXECUTED` (final candidate recheck); original REC-004 RG3 FAIL remains historical |
| REC-005 | `BLOCKED` |
| G1 | `STOP` (`G1_RELATION_TRANSFER_STOP`) |
| G4 / G5 | `BLOCKED` / `BLOCKED` (historical development G4 FAIL preserved) |
| Closeout sealed-data / model-output access | `0` / `0` |
| Protected sealed_v2 model evaluation | `NOT_EXECUTED`; seal remains unopened |

RG3, REC-005, REC-006--008, R3-011, R3-012, and B-C006 onward are archived
**non-executions due to upstream STOP**, not unfinished backlog. No Phase-B
experiment remains queued. Earlier permission to continue Phase B is exhausted
by its scientific terminal state. No new charter can reopen or promote its runs.

The zero-access statement is scoped: historical B1, B2, and B2 re-gate reports
already include their then-authorized sealed results (ADR-0074/0075/0079/0080).
They are not relabeled never-opened or clean future holdouts. This closeout reads
their tracked ADR descriptions only; it does not reopen those run outputs.
The protected Post-D2 sealed_v2 remains unevaluated, as recorded in ADR-0147/0148.

## Evidence classification

Classification applies to a **claim**, not to an entire task. A task can support
an invariant while falsifying a performance hypothesis. Mechanistically explained
means a bounded cause has intervention/diagnostic evidence; it does not mean a
repair succeeded or every failure has that cause. Falsified refers to a registered
criterion under its measured conditions, never to APC's universal impossibility.
Every ADR in 0074--0148 is covered below; process entries add no performance evidence.
Exact source locations, individual ADR hashes and run references are in the
[freeze manifest](PHASE_B_CLOSEOUT_FREEZE_MANIFEST.json).

### Supported

| Claim and boundary | Evidence |
|---|---|
| Explicit-TaskSpec unseen-family lifecycle works in the B1 setting: novel EM 0.9766, recurrence EM 0.9719, zero recurrence adaptation/temp capacity and legacy regression. This does not establish implicit semantic task inference. | ADR-0074 |
| Original B2 retained top-5 recall and rejected wrong calls in that benchmark. Development ranking/argument and sequential-adequacy repairs passed their local contracts, without proving transfer or full functional safety. | ADR-0075, 0077--0078; safety qualification in ADR-0080 |
| Deterministic generation, explicit adequacy vocabulary, finite-look verifier and safe uncertainty handling passed their scoped infrastructure/control gates G0/G2/G3. | ADR-0082, 0084, 0086 |
| Local implementation improvements exist: COUNT/BIND scoping (no dev confusion left to fix), SELECT train/inference formula correction, BIND rare-value coverage on development; safe SHIFT replacement/rollback works independently of its performance FAIL. | ADR-0085, 0087--0090 |
| Dependency inventory, strict immutable-bundle loading and complete explicit 16-operation build plan pass RG0--RG2; original REC-004 fresh-load parity works despite RG3 performance FAIL. SciPy dependency requalification is an environment result only. | ADR-0092--0095, 0107 |
| Three incremental operations meet seed-10 calibration floors; padding/batching controls are valid; explicit position bias and more training improve measured EM, although all-init adoption fails. Clean-v2 confirms individual earlier passes without erasing the original overlap disclosure. | ADR-0096, 0098--0104 |
| Joint downstream protection and role/residual separation preserve oracle compatibility in their fixed settings; this is stability evidence, not adequate normal execution. | ADR-0118--0119, 0122--0125 |
| Corrected coordinates and pinned saved vocabulary preserve original forward/EM evidence. | ADR-0126, 0128 |
| CD-DPCA satisfies the finite routing representability/structural contract and strict serialization/fresh-load checks. Representational sufficiency is restricted to the validated finite domain; it is not learnability or arbitrary-length transfer. | ADR-0134--0136 |
| CD-DPCA execution is causally dependent on the primitive: AL Correct EM 0.793945, Wrong/None 0.0. The fixed AQ I01 warm-start can succeed: EM 0.985352, length-10 EM 1.0. | ADR-0137, 0141; no override of ADR-0143 |
| Relation-scoped CE blocks held-out key gradients, updates and optimizer state on the fixed throwaway-router probe while retaining in-scope gradients. | ADR-0147; not production transfer evidence |
| The registered current architecture has enough negative evidence to terminate. Process guidance/replanning is administrative support only. | ADR-0148; process-only ADR-0127, 0142 |

### Falsified

| Registered claim that failed | Evidence and scope |
|---|---|
| Full B2 hard-negative safety/ranking and repaired sealed re-gate sufficiency. | ADR-0075, 0079; original N=128 top-1 L2/L3/L4 = 0.866/0.662/0.504, false plastic 0.0375 > 0.02. Local development PASS does not rescue either FAIL. |
| Existing relation registry and original full-class-CE exposure suffice for strict relation transfer. | ADR-0083, 0147: latest validation 0/2, sealed_v2 1/2 independent clean components. Isolation PASS does not fix the count. |
| SHIFT local repair and combined development integration meet the registered gates. | ADR-0090 (3/5 replacement, local FAIL), 0091 (G4 FAIL). |
| Original recovered bundle meets non-SHIFT execution floors; finite budget/schedule/position-bias/extension alternatives produce an all-init MIRROR candidate. | ADR-0095--0099, 0102--0104. Partial gains and individual endpoint passes are not adoption. |
| Fixed score-only or content-prep-release continuation adequately repairs I03; fixed shared-gradient aggregation explains its terminal failure. | ADR-0111--0112, 0114; bounded negative tests, not a proof that content adaptation or interference never matters. |
| Attention drift alone causes the old downstream collapse; fixed trust radius, stability metrics or compact residual interventions supply the registered adequate repair/gate. | ADR-0117, 0119--0125. ADR-0116's temporal ordering does not establish causation after the clamp test. |
| Global score rescaling or position-only transport identifies an adequate scorer repair. | ADR-0129 (4/4 scales fail), 0132 (transport EM=0). |
| Baseline CD-DPCA I01 reaches terminal viability; fixed warm-start meets all-init qualification or broad causal superiority. | ADR-0137: 0.793945 < 0.95; ADR-0143: 1/5 versus required 5/5; ADR-0144: 2/5 co-improve versus required >=4/5, I02/I03 adverse. |

### Mechanistically explained

| Bounded failure mechanism | Evidence |
|---|---|
| Original L4 failure was argument resolution; original false-plastic cells were support-estimation variance. This attribution is specific to that original set. Later seed-24 SHIFT revealed true primitive inadequacy and asymmetric early acceptance, exposing a safety-metric blind spot. | ADR-0076, 0080--0081 |
| Relation coupling and CE-denominator exposure invalidate naive relation counting/holdout; scoped CE can isolate keys. | ADR-0083, 0147 |
| SELECT formula mismatch, BIND rare-value coverage, stale Core/bank pairing and missing build paths explain distinct local defects. | ADR-0088--0089, 0091--0094 |
| Old MIRROR failures vary by init and checkpoint: oracle attention is sufficient for some states, insufficient/harmful for others; downstream compatibility changes over time. FFN/value interactions and joint CVOF drift are needed to explain later collapse. | ADR-0100--0106, 0108--0110, 0115--0118; one-step fidelity does not imply trajectory stability. |
| Corrected endpoint errors are routing-dependent, but QK/position ablations and matched controls cannot isolate a single repair target; the old decomposition fails the joint identifiability conditions. | ADR-0126--0133 |
| CD-DPCA I01 error concentration, destructive alias-stratum gradients and terminal softmax starvation explain the measured false attractor. | ADR-0138--0140: 89.4% of length-10 errors at position 4; alias stratum 27/206 (13.1%), 12.4x destructive-gradient dominance. |
| Matched warm-start effects depend on initialization and early data jointly; step-0 geometry alone and a generic stream rule do not identify a repair. | ADR-0144--0145: 130 archived states; 40 cells/init; early sign changes 10/13/18/19/12 and basin divergences 6/9/5/5/9. |
| In the fixed token-alias collision, identical allowed token-output CE observables admit opposite semantic routing directions. They do not identify a unique oracle-free CE-only repair under AT's information/search boundary. | ADR-0146: key0/key7 gradient ratio about 192; conditional full-cell formulation test not executed because no formula was identified. |

### Unresolved

| Unproved question | Why Phase B cannot answer it |
|---|---|
| Which single legal scorer component, generic initialization constraint or CE-only optimization formulation repairs the failure? | ADR-0113, 0130--0133, 0145--0146 identify no such target/formula. Non-identification is not an impossibility theorem over all optimizers or population-level information. |
| Do local development repairs generalize to truly independent relations and a coherent five-model lifecycle? | ADR-0083, 0085, 0087--0091, 0147--0148: relation and bundle prerequisites fail. Five primitive initializations on one Core are not five independent models. |
| Would RG3 recheck, REC-005--008, G4/G5 or B3--B6 pass with a different qualifying design? | No eligible current candidate exists. The unexecuted results are unknown and archived due to upstream STOP, not tasks to resume. |
| Can additional lawful information or a different learning contract resolve ambiguous routing robustly? | Neither selected nor tested in Phase B. This is the separate charter's question. |

### Out of scope

APC's general impossibility, arbitrary architecture/optimizer families, arbitrary
lengths and real-world language capability, pretrained LMs/RL, and universal
scaling claims were not evaluated. B3--B6 belonged to the planned parent claim
but remain **unresolved and unexecuted**, not retroactively outside its intended
scope. A future design's result cannot relabel the current architecture PASS.

## Consolidated failure chain

1. The old score decomposition did not identify a single-component repair target (AH; ADR-0133).
2. CD-DPCA supplied finite representational sufficiency and structural separation (AI--AK; ADR-0134--0136).
3. After baseline AL failed, AQ showed success is possible for I01 (ADR-0137, 0141).
4. The fixed recipe did not reproduce stably across five inits: AM passed only 1/5 (ADR-0143).
5. AN--AP explained part of this failure family through token aliasing and softmax saturation; the evidence is localized, not a universal cause for every init (ADR-0138--0140).
6. Matched AR showed warm-start's initialization-dependent mixed effect, including harm (ADR-0144).
7. AS could not identify a generic initialization-geometry-only repair (ADR-0145).
8. Its matched states and early-gradient records identified initialization × early-data interaction as the supported account of basin susceptibility (ADR-0145).
9. AT's collision showed that allowed token-output CE observables alone cannot choose the correct routing direction in that ambiguous observation (ADR-0146).
10. No single CE-only repair formulation followed within the registered boundary. Candidate qualification stayed failed and the independent G1 count stayed insufficient; ADR-0148 therefore terminated the current architecture.

This is a causal synthesis, not a new experiment or an assertion that the studies
were executed in this numbered order. In particular, AN--AP preceded AQ/AM, and
old-scorer CVOF failures must not be conflated with CD-DPCA's later mechanism.

## Assumptions that must change

| Requirement | Consequence for the independent charter |
|---|---|
| Training-signal identifiability | Explicitly identify what breaks routing symmetry: a separate lawful signal, task/data constraints, or a latent-routing treatment with an identifiable observation contract. No alternative is adopted here. A new latent alone adds no information; removing all evaluation collisions would evade the proposed ambiguity question. |
| Initialization robustness | Put multi-init stability in the primary contract before implementation; require all registered inits, worst-cell performance and non-regression. A successful seed or mean above threshold cannot substitute. |
| Relation transfer | Demonstrate at least two independent clean components in each validation and sealed partition before training; audit alias, learned-parameter coupling and full-loss/optimizer exposure. A new seed, inverse relation or renamed member adds no independence. |
| Sealed prerequisites | Retain a coherent candidate, multi-init qualification, RG3 equivalent, G1 and G4 equivalents, immutable fresh-load evidence, and a separately frozen sealed protocol. A diagnosis is not a gate PASS. |

## Reusable assets and exclusions

| Reuse candidate, subject to new-scope validation | Boundary |
|---|---|
| Immutable bundle/hash audits and strict fresh-load validation | Reuse infrastructure; do not inherit Phase-B candidate eligibility or relabel exposed data. |
| Correct/Wrong/None and Wrong-argument controls | Keep task-blind content, primitive necessity and no decoder bypass. |
| Multi-init qualification, side-effect and sealed-access audits | Preserve all-init and no-selection rules; qualify independent model axes separately. |
| Routing, gradient directional-derivative and failure-localization diagnostics | Oracle coordinate maps stay evaluation-only, never loss/init/sampler/selection input. |
| ADR/execution-plan gate structure and capacity accounting | Separate resident/active/temporary parameters; log fixed budgets, failures and provenance. |

Do not carry over current Phase-B scorer assumptions, CE-only routing
identifiability, uniform warm-start as a generic repair, successful-seed selection,
post-hoc candidate promotion, or current G1 relation-family sufficiency.
Historical checkpoints are evidence, not adopted Phase-C parents. Any later
reuse needs independent lineage/exposure qualification; this charter adopts none.

## Freeze and verification contract

The [freeze manifest](PHASE_B_CLOSEOUT_FREEZE_MANIFEST.json) fixes all 75 ADR
entries, source document identities at the pre-closeout commit, explicit historical
run-reference existence, and SHA-256 identities of the selected development
reports/protocols/manifests. All 130 AL/AM/AR checkpoint states named in the
existing AS source manifest were also rehashed as raw bytes and matched their
recorded digests; no checkpoint is loaded or evaluated here.
The [audit record](PHASE_B_CLOSEOUT_AUDIT.json) records verification and scope.

The run audit confirms 114 concrete references; nine historical template paths
are explicitly classified as templates rather than fabricated artifact paths.
Three pre-existing supplied-attachment links in the archived REC-004D contract
are missing: `REC004C_ADR0098_SUPPLIED.md`, `REC004C_TASK_SUPPLIED.md`, and
`MODEL_BUNDLE_CONTRACT_SUPPLIED.md` under `docs/research/evidence/`. They are
disclosed historical provenance gaps, not passing links and not newly recreated
evidence. ADR-0098, the original REC-004C task and the bundle contract still exist
in their canonical documents; this audit does not claim they reconstruct the
missing attachments. Current closeout links and decision-index anchors are
verified separately. No scientific conclusion relies on those missing attachments.

The local git commit makes the ledger/charter and these manifests a content-addressed
snapshot. Future corrections must be appended with a new ADR and new snapshot;
never edit this frozen interpretation in place. Existing qualified and failed runs,
including preliminary G1 run_001 and qualified run_002, remain in their original
gitignored locations. Hashes detect later changes to the covered files; local git
and manifests do not provide a physically write-once backup of all run binaries.
Neither absence of a reference nor a missing artifact may be repaired by fabrication.

Required audit: local links/anchors, ADR existence/coverage, explicit run-reference
existence, before/after development evidence hashes, historical ADR preservation,
terminal status and queue consistency, and `git diff --check`. Only documentation
and audit artifacts change. pytest/ruff/mypy are not rerun; no implementation
changed. Architecture/optimizer implementation, training, candidate creation,
dataset generation, relation registration, seed execution, pilot and sealed
evaluation are all zero. Charter readiness is an administrative PASS; Phase-B
recovery and research gates retain their recorded FAIL/STOP/BLOCKED statuses.
