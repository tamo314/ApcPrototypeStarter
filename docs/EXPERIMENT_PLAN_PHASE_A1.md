# Phase A.1 Experiment Plan

## 1. Research question

Can the APC loop work when representation, composition, novelty, routing, plastic learning, consolidation, and reuse are isolated and introduced one at a time?

## 2. Hypotheses and gates

### H1 — Stable systematic generalization

A Stable Core trained on procedurally generated known operations executes those operations on unseen content.

Gate: unseen-content K exact match >= 0.95 across >=5 seeds. If this fails, stop downstream Phase A.1 work.

### H2 — Composition without expansion

With oracle primitive routing/search, unseen compositions of known primitives are solved without Plastic Workspace allocation.

Targets: C exact match >=0.90; C expansion rate <=0.10.

### H3 — Selective plasticity with oracle novelty

Ground-truth N triggers expansion; K/C do not.

Targets: K/C expansion <=0.10; N expansion >=0.90.

### H4 — Residual learning

Residual-only plastic learning matches full-task plastic performance with lower new persistent capacity.

Target: final performance within 2 percentage points with lower persistent added parameters.

### H5 — Functional consolidation

An overcomplete workspace for a compressibility-controlled N task is distilled into a materially smaller primitive.

Targets:

- functional agreement >=0.99 on held-out probe states,
- task performance retention >=0.95 of temporary solution,
- consolidated/temporary parameter ratio <=0.50.

Stretch: candidate rank <=2x ground-truth rank.

### H6 — Recurrence reuse

Oracle recurrence routing: R exact match >=0.95 with approximately zero new adaptation steps.

Retrieval recurrence: R exact match >=0.90 and >=90% reuse without new consolidation.

### H7 — Learned routing

Oracle-required primitive appears in selected top-k >=0.95 and recurrence reuse >=0.90.

### H8 — Learned novelty

K/C-vs-N AUROC or predeclared equivalent >=0.90 while achieving K/C expansion <=10% and N expansion >=90%.

### H9 — True sparse execution

Non-selected primitive forward calls = 0; active primitive count matches selected IDs; active primitive capacity is materially below resident primitive capacity as the bank grows.

### H10 — Full sequential loop

After H1-H9 pass, run repeated K/C/N/R sequences with >=5 seeds and measure bounded growth, retention, reuse, compression, and true sparse compute.

## 3. Task universe

- **K:** known primitive, unseen content.
- **C:** unseen ordered composition of known primitives.
- **N:** genuinely absent primitive; controlled N families have hidden low-rank ground truth.
- **R:** recurrence of a previously learned N operation with fresh content.

## 4. Anti-memorization controls

For H1-H3:

- online procedural generation,
- symbol permutation,
- unseen evaluation content,
- examples/tokens seen reported,
- fixed datasets only as diagnostics.

## 5. Oracle Ladder

| Stage | Novelty | Routing | Consolidation | Purpose |
|---|---|---|---|---|
| O0 | Oracle | Oracle | Oracle/simple | verify environment/execution semantics |
| O1 | Oracle | Oracle | Learned | isolate consolidation |
| O2 | Oracle | Retrieval | Learned | isolate recurrence retrieval |
| O3 | Oracle | Learned | Learned | isolate router |
| O4 | Learned | Learned | Learned | full Phase A.1 |

Never skip a failed stage.

## 6. Baselines

Retain Phase A baselines where meaningful, but use corrected accounting. Add:

- fixed dense Stable Core,
- oracle primitive execution,
- retrieval-routed sparse system,
- grow-only residual workspace,
- full-task plastic + consolidation,
- residual plastic + consolidation,
- full APC Phase A.1.

All comparisons must state training-step/example budgets.

## 7. Seed policy

- debugging: 1-2 seeds,
- gate: >=5 seeds,
- milestone claim: prefer 10 seeds when practical.

Report mean, standard deviation, and per-seed values.

## 8. Scale policy

Do not use model scale to rescue a failed mechanism until online generation, symbol permutation, and the relevant oracle control are verified. Larger scale is confirmation, not a substitute for isolation.

## 9. Optional grokking diagnostic

Run the old fixed-dataset setup much longer (for example checkpoints around 50k/100k/300k steps with regularization). This is diagnostic only and does not replace H1.

## 10. Stop conditions

Stop dependent work if:

- H1 remains near chance under online randomized training,
- oracle C cannot execute the supplied decomposition,
- oracle N cannot be learned by Plastic Workspace,
- controlled low-rank N cannot be functionally consolidated,
- oracle recurrence cannot reproduce the consolidated behavior,
- true sparse execution is numerically incorrect.

## 11. Required metrics

Per event:

- exact match/task score,
- expansion decision,
- adaptation steps,
- selected primitive IDs,
- recipe ID,
- routing entropy,
- best-existing residual loss,
- resident total parameters,
- resident primitive parameters,
- active total parameters,
- active primitive parameters,
- temporary peak,
- compression ratio,
- functional agreement,
- recurrence reuse flag,
- wall-clock.

Sequence-level:

- backward transfer,
- max forgetting,
- expansion-cycle count,
- recurrence-without-expansion count,
- lifetime train steps/examples,
- primitive growth,
- recipe growth.

## 12. Completion

Phase A.1 is complete only when H1 passes, O0-O3 pass, functional consolidation passes on controlled tasks, recurrence reuse works, learned routing/novelty pass, true sparse compute is verified, and the full loop runs at >=5 seeds.
