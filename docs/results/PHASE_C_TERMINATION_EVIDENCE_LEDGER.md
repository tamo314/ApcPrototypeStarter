# PHASE-C-TERMINATION — Frozen evidence ledger

Date: 2026-09-13. Decision: `PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`.
This is the authoritative final claim classification for ADR-0150 through ADR-0159,
extending the [C-D001AA termination audit](../phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md).
It does not replace, rewrite, or renumber any of ADR-0150–ADR-0159; it classifies which of
their conclusions remain load-bearing given the retractions those ADRs already record.

## Terminal state

| Field | Final value |
|---|---|
| Phase C scientific decision | `PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT` |
| Charter administrative state | `TERMINATED_CURRENT_CHARTER` |
| Architecture selected | Never — `NOT_SELECTED`, permanently barred at every audit step |
| First experiment defined | Never — `NOT_DEFINED` |
| Research execution | Never authorized — `NOT_AUTHORIZED` at every point in the chain |
| candidate_selected | `null` |
| C-D002 (architecture derivation) | `NOT_AUTHORIZED` / permanently barred |
| H-C1-Residual | `RETRACTED` (ADR-0159) |
| Relation inventory | validation 1/2, sealed 1/2 clean independent components (ADR-0150; never remedied) |
| Termination sealed-data / model-output access | `0` / `0` |
| Sealed data | Never opened at any point in Phase C |

No Phase C experiment was ever queued, run, or partially executed. This closure ends a
pre-execution design/audit chain, not an experimental program — Phase C never advanced past
its own design-review gate (C-D002) at any point.

## Evidence classification

Classification applies to a **claim**, not to an entire ADR. An ADR's stoppage rationale can
be superseded while its underlying measurement remains current evidence. "Retracted" means
the claim's own successor ADR explicitly withdrew it — not that APC in general, or every
oracle-free contract, was disproven. Full reasoning and direct quotations are in the
[C-D001AA audit](../phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md) §2.

### Current

| Claim and boundary | Evidence |
|---|---|
| Contract v1 (pure positional/content-decoupled routing geometry) is falsified via the World A/B collision-cell counterexample under opaque task ID. | ADR-0151 |
| Lawful compositional descriptors carrying an explicit, general tie-break policy (FIRST/LAST/LEFTMOST/RIGHTMOST/procedural composition) provably separate adversarial worlds that require divergent routing coordinates (Theorem 3). | ADR-0153 |
| The 5-dimensional non-circular oracle-supervision criterion (provenance, example specificity, relation specificity, inference-time availability, counterfactual invariance) is the current oracle-boundary source of truth; it classifies FIRST/LAST-style descriptors as strictly non-oracle (0/5). | ADR-0155, validated adversarially by ADR-0156, still invoked unmodified by ADR-0159 |
| Descriptor-Only Deterministic Baseline $B_{\text{det}}$ (0 parameters) satisfies 100% of routing/execution floors with $H(Z\mid X,D)=0.0$ bits, trivializing H-C1's estimand under Contract v1.1. | ADR-0157 |
| Embedding-Aware Deterministic Baseline $B_{\text{det\_emb}}$ (0 parameters) dominates under invertible/discrete representations (100% ceiling); degradation under lossy/colliding representations (Controls 3–4) is an identifiability limit, not a learnability limit. | ADR-0158 |
| Blind codebook/manifold grounding under permutation ($S_V$)/orthogonal ($O(d)$) symmetry is either informationally unidentifiable (no/1/partial anchor) or deterministic-baseline-dominated (full anchor); the Impossibility–Dominance Dilemma leaves no third regime; `H-C1-Residual` is retracted. | ADR-0159 |
| Relation inventory: validation has 1/2, sealed has 1/2 required independent clean components. | ADR-0150 (never re-audited or contradicted) |

### Superseded

| Claim | Superseded by | Boundary |
|---|---|---|
| `RELATION_INVENTORY_FEASIBILITY_STOP` as the *primary* blocking rationale for Phase C. | ADR-0151 (`ROUTING_IDENTIFIABILITY_STOP`) | The underlying relation-count *measurement* (1/2, 1/2) is not superseded — only its status as the primary/sole blocker is. |
| Training Information Contract v1.1's practical value as a route to a non-trivial H-C1 result. | ADR-0157 | The *validation methodology* that produced Contract v1.1 (ADR-0156's adversarial mixed-control test of the 5-dim criterion) remains sound; only the contract's payload is a scientific dead end once $B_{\text{det}}$ dominates it. |

### Retracted

| Claim | Retracted by | Reason |
|---|---|---|
| "`ROUTING_IDENTIFIABILITY_STOP` is a quantifier-complete impossibility theorem across all permitted task-side observables" (including compositional descriptors). | ADR-0153 | Theorem 3 constructs lawful descriptors that do separate the adversarial worlds; ADR-0153 states the quantifier-complete claim "is mathematically refuted and retracted." |
| "Deterministic $z$-computability of a descriptor implies oracle-equivalent supervision" (and the resulting universal STOP / Contract v1.1 rejection). | ADR-0155 | Diagnosed as circular: "any formal deterministic task semantics ... produc[es] an unfalsifiable circular criterion" under ADR-0154's own standard. ADR-0155 replaces it with the 5-dimensional criterion. |

### Restricted

| Claim | Restriction |
|---|---|
| Non-identifiability under permitted oracle-free observables (ADR-0151/ADR-0152's opaque-ID and finite-output-support-set sub-findings). | Restricted by ADR-0153 to opaque task identifiers, finite output-labeled support sets under token supervision alone, and underspecified descriptors lacking tie-break semantics — not fully specified lawful descriptors, which are separately identifiable (and separately shown trivial by ADR-0157, not unidentifiable). |

### Historical evidence only

None of ADR-0150–ADR-0159 is classified purely `HISTORICAL_EVIDENCE_ONLY` — every one contributes either a current finding or an explicitly superseded/retracted claim that shaped a later current finding. All ten remain load-bearing to the final closure and are frozen, not archived as inert.

## Consolidated falsification chain

1. C-D001 derives Contract v1 and separately finds the relation inventory insufficient (ADR-0150).
2. C-D001R mathematically falsifies Contract v1 itself via the World A/B collision-cell counterexample, correcting the primary stop rationale to identifiability (ADR-0151).
3. C-D001S over-generalizes this into a quantifier-complete impossibility claim across all permitted observables (ADR-0152).
4. C-D001T repairs the circularity in that claim and proves lawful descriptors with tie-break semantics *do* separate the worlds, retracting ADR-0152's overreach (ADR-0153).
5. C-D001U counters that any such separating descriptor is oracle-equivalent because it is deterministically computable, rejecting Contract v1.1 (ADR-0154).
6. C-D001V diagnoses ADR-0154's criterion as circular, replaces it with a 5-dimensional non-circular test, and shows FIRST/LAST-style descriptors score 0/5 (non-oracle) — retracting ADR-0154's universal stop (ADR-0155).
7. C-D001W adversarially stress-tests the 5-dimensional criterion against seven mixed controls, upholds it, and derives candidate Contract v1.1 plus a deterministic-baseline requirement (ADR-0156).
8. C-D001X builds the descriptor-only deterministic baseline $B_{\text{det}}$, proves it solves Contract v1.1 completely with 0 parameters, and shows this trivializes H-C1's estimand rather than validating it (ADR-0157).
9. C-D001Y extends the same argument to continuous/embedding representations via $B_{\text{det\_emb}}$, closing "continuous grounding" as a non-trivial residual (ADR-0158).
10. C-D001Z formalizes the last remaining residual (blind codebook/manifold discovery) as a group-symmetry problem, proves the Impossibility–Dominance Dilemma, shows the non-trivial residual set is empty, and retracts `H-C1-Residual` (ADR-0159).
11. C-D001AA audits this entire chain for falsification sufficiency, confirms no in-charter conclusion-changing task remains, and formally terminates the current charter (ADR-0160, this closure).

This is a causal/logical synthesis of an already-completed pre-execution design-review chain, not a new experiment. No architecture was ever implemented, no training ever ran, and no sealed data was ever touched at any step 1–11.

## Assumptions and scope that must not be overgeneralized

Per the [C-D001AA audit](../phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md) §3, the impossibility findings above are proven only for:

- opaque-ID / token-only supervision;
- lawful fully specified descriptors, with and without tie-break policy;
- discrete deterministic reduction;
- known/invertible continuous representations (dense linear maps, bounded perturbations);
- blind codebook/manifold representations under permutation ($S_V$) and orthogonal ($O(d)$) group symmetry, across no/1/partial/full anchor regimes.

They explicitly do **not** establish: APC's impossibility in general; futility of every
oracle-free task-information contract; impossibility of nonlinear, stochastic, interactive,
or future task formulations; or impossibility of routing-based architectures in general.
Phase C never reached architecture derivation (C-D002 stayed barred throughout), so this
closure is an estimand/identifiability finding, independent of Phase B's separate,
already-closed architecture-level negative result (ADR-0148).

## Out of scope

APC's general impossibility, arbitrary representation-learning families outside the
permutation/orthogonal group-symmetry model, nonlinear/stochastic/interactive task
formulations, pretrained LMs/RL, and any Phase D hypothesis were **not evaluated** by Phase C
and are not addressed by this closure. A future `NEXT-RESEARCH-QUESTION REVIEW` is the
correct venue for such a question, starting from whether a deterministic-baseline-resistant,
information-theoretically identifiable estimand exists at all.

## Reusable assets and exclusions

| Reuse candidate, subject to new-scope validation | Boundary |
|---|---|
| The 5-dimensional non-circular oracle-supervision criterion (ADR-0155/0156) | Reuse as a general oracle-boundary test; do not assume it was calibrated against representation classes beyond §"Assumptions and scope" above. |
| Descriptor-Only and Embedding-Aware deterministic baselines ($B_{\text{det}}$, $B_{\text{det\_emb}}$) | Reuse as ceiling controls for any future routing-identifiability estimand; do not assume they cover nonlinear or non-invertible representations without re-derivation. |
| The Impossibility–Dominance Dilemma's proof pattern (entropy-vs-anchor-information monotonicity) | Reuse as a template for auditing a new estimand's non-triviality *before* proposing architecture; do not assume its group-symmetry premises transfer to a different representation model without restating them. |

Do not carry forward H-C1 itself, Contract v1/v1.1, or `H-C1-Residual` as live hypotheses —
all three are closed. Do not treat the relation-inventory deficit (ADR-0150) as the sole
remaining blocker; it is independent of, and does not cure, the estimand closure above.

## Freeze and verification contract

[`PHASE_C_TERMINATION_AUDIT.json`](PHASE_C_TERMINATION_AUDIT.json) records ADR-0150–ADR-0160
existence/consistency checks, artifact existence for all eleven `docs/phase_c/artifacts/*.json`
files, the four verification test files, the two baseline implementation files, cross-document
status-string consistency, and `git diff --check`. This closure changed documentation only;
`pytest`/`ruff`/`mypy` were not rerun because no implementation file changed (AGENTS.md
documentation-only exception). Architecture/optimizer implementation, training, candidate
creation, dataset generation, relation registration, seed execution, pilot, and sealed
evaluation are all zero for this closure, as they were for every ADR-0150–ADR-0159 task before it.

Future corrections must be appended with a new ADR; never edit this ledger's interpretation
of ADR-0150–ADR-0159 in place. ADR-0150 through ADR-0159 remain at their original byte content.
