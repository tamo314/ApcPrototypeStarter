# Experiment Plan — A1-R005E Representation / Operator Isolation

## 1. Goal

Determine whether parameterized primitive failure is primarily limited by:
- frozen task-blind representation,
- primitive/operator computation class,
- or interface/factorization.

## 2. D-E1 — Raw representation audit

Use the exact frozen content encoder checkpoint(s) used in the failed retry.

Train small diagnostic probes:
1. token identity from each `h_content[j]`,
2. absolute position,
3. full content-sequence reconstruction,
4. BIND key/value role,
5. optional adjacency/pair probe.

Suggested diagnostic thresholds:
- token identity >=0.98 desirable,
- position >=0.98 desirable,
- full input reconstruction >=0.95 sequence exact desirable.

These are diagnostics, not APC gates.

## 3. D-E2 — Oracle latent operator

Frozen content encoder; no learned primitive routing.

Targets:
- SHIFT/SELECT/BIND token accuracy >=0.98,
- SHIFT/SELECT/BIND exact match >=0.90,
- COUNT exact match >=0.90 using documented oracle-match aggregation diagnostic.

If this fails, investigate decoder/state interface before operator learning.

## 4. D-E3 — Frozen high-capacity upper-bound operator

Frozen content encoder.

Use expressive argument-conditioned cross-position model.

Minimum 5 seeds for branch claims.

Arms:
- Correct argument,
- effectful Wrong argument,
- None/no argument where meaningful.

Targets per operation:
- Correct exact >=0.90,
- SHIFT/SELECT token >=0.98,
- effectful Wrong argument <=0.30 exact where comparable,
- causal gap >=0.50.

The operator may exceed APC primitive parameter limits.

## 5. D-E4 — Compact cross-position operator probe

Run only if D-E3 passes substantially.

Use a deliberately small operator, e.g. one cross-attention block with small projections.

Compare against:
- historical conditioned low-rank primitive,
- high-capacity upper bound.

Positive evidence:
- materially above historical Correct performance,
- materially larger argument causal gap,
- substantial fraction of upper-bound performance,
- primitive-scale persistent size.

## 6. D-E5 — Joint task-blind representation control

Run if D-E3 fails or remains severely limited.

Jointly train:
`task-blind content encoder + strong argument-conditioned operator`

Constraints:
- content encoder sees content only,
- task/argument reaches operator only,
- same content under different tasks yields same content representation,
- no target leakage.

If joint succeeds and frozen fails -> Representation branch.
If joint also fails -> Interface branch.

## 7. Branch criteria

### Operator branch
- oracle latent operator passes,
- frozen high-cap upper bound passes >=0.90 on all/most operations,
- compact operator materially exceeds historical primitive.

### Representation branch
- frozen upper bound fails,
- joint task-blind training reaches >=0.90,
- representation diagnostics improve after joint training.

### Interface branch
- oracle latent operator fails badly, or
- joint task-blind strong model also fails.

## 8. Seed policy

Development: 1–2 seeds.
Decision evidence: >=5 seeds for D-E3 and D-E5 claims.

## 9. No efficiency claim

Upper-bound models are diagnostic only. Do not treat their parameter count as APC evidence.

## 10. Final output

Create:
`docs/results/A1_R005E_DIAGNOSTIC_RESULT.md`

It must fill the decision matrix and recommend the next branch without implementing it.
