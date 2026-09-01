# Codex Task Queue — Phase A.1 Post-Correction

Execute exactly one task at a time.

## A1-R001 — Task-blind content encoder path

Goal: make primitive input independent of task specification.

Work:
1. add content-only encoding path,
2. reuse Stable Core weights if practical,
3. exclude task tokens from content encoding,
4. keep task-only encoding for `z_task`,
5. preserve A1-C004 shared-core solver as baseline.

Acceptance:
- same content + different task specs => invariant primitive input state,
- all 8 operation families covered,
- no task token in content-encoder input.

STOP GATE.

---

## A1-R002 — Decoder leakage control

Goal: prevent task information from bypassing primitive execution.

Work:
1. audit decoder inputs,
2. remove task-conditioned bypasses in causal mode,
3. add no-primitive/identity mode,
4. evaluate decoder-only task performance.

Acceptance:
- no-primitive path materially below future Correct target,
- tests prove task-spec tensors are absent except documented formatting-only info.

STOP GATE if no-primitive stays near full accuracy.

---

## A1-R003 — Parameter-free oracle primitive benchmark

Operations:
COPY / NEGATE / COMPARE / ACCUMULATE.

Work:
1. pre-register one neural primitive family per operation,
2. freeze Stable Core,
3. train primitives,
4. use oracle family selection,
5. evaluate Correct / Wrong / None,
6. run >=5 seeds.

Acceptance:
- Correct >=0.95,
- Wrong <=0.30,
- None <=0.30,
- causal gap >=0.50,
- selected-only execution remains true.

STOP GATE.

---

## A1-R004 — Argument conditioning module

Goal: make neural primitives consume `PrimitiveCall.arguments`.

Work:
- typed encoders for SHIFT.amount, SELECT.indices, COUNT.target, BIND.query_key,
- condition transform on argument embedding,
- shared family weights across values.

Acceptance:
- changing argument changes output,
- same family handles >=3 values,
- persistent family count unchanged,
- invalid/missing arguments tested.

---

## A1-R005 — Parameterized oracle primitive benchmark

Operations:
SHIFT / SELECT / COUNT / BIND.

Evaluate:
1. correct family + correct argument,
2. correct family + wrong argument,
3. wrong family,
4. no primitive.

Acceptance:
- Correct >=0.90,
- Correct materially exceeds controls,
- family count stays constant across argument values.

STOP GATE.

---

## A1-R006 — Unified oracle primitive benchmark

Run all eight through:
task-only encoder -> oracle PrimitiveCall -> task-blind content encoder -> selected primitive -> decoder.

Acceptance:
- overall Correct >=0.95,
- parameter-free ops >=0.95,
- parameterized ops >=0.90,
- aggregate Wrong/None materially low.

---

## A1-R007 — Composition Library execution

Goal: execute ordered recipes of existing PrimitiveCalls.

Work:
- add/complete CompositionLibrary,
- execute sequential primitive calls,
- support oracle recipes,
- do not re-run task-conditioned Stable Core between steps.

Acceptance:
- designated C >=0.90,
- no Plastic Workspace allocation.

STOP GATE.

---

## A1-R008 — Composition search baseline

Goal: recover recipes without oracle identity.

Use exhaustive/beam search on small bank.

Suggested:
- depth <=3,
- beam 8-32.

Acceptance:
- designated C >=0.85,
- no expansion,
- recovered recipe matches or functionally equals oracle recipe.

---

## A1-R009 — Oracle novelty controller v2

Use ground-truth K/C/N/R for transitions.

Acceptance:
- K/C expansion <=10%,
- N expansion >=90%,
- R first attempts reuse.

---

## A1-R010 — Residual Plastic Workspace v2

Goal: learn only computation not explained by current bank.

Work:
1. find best recipe,
2. freeze stable pieces,
3. train temporary residual,
4. retain full-task plastic control.

Acceptance:
- controlled N >=0.95 after adaptation,
- residual formulation validated,
- no persistent install yet.

---

## A1-R011 — Compressibility-controlled N generator

Create hidden low-rank novel operations.

Suggested:
- target rank 4,
- overcomplete temporary workspace.

Acceptance:
- temporary workspace learns held-out examples,
- ground-truth weights never leak to learner/controller.

---

## A1-R012 — Functional consolidation v2

Work:
1. collect `(h, delta)` probes,
2. fit rank/capacity sweep,
3. choose smallest passing candidate,
4. shadow validate.

Acceptance:
- agreement >=0.99,
- task retention >=0.95,
- permanent/temporary <=0.50.

STOP GATE.

---

## A1-R013 — Oracle recurrence

Acceptance:
- score >=0.95,
- near-zero adaptation,
- >=90% R events avoid new consolidation.

STOP GATE if this fails.

---

## A1-R014 — Retrieval recurrence baseline

Store task-key prototypes and retrieve by similarity.

Acceptance:
- oracle item retrieved >=95%,
- recurrence >=0.90,
- reuse without new cycle >=90%.

---

## A1-R015 — Learned router

Train from `z_task`; oracle only for supervision/evaluation.

Acceptance:
- oracle primitive in top-k >=0.95,
- recurrence reuse >=0.90,
- no severe bank-size collapse.

STOP GATE.

---

## A1-R016 — Residual novelty baseline

Signals:
- best-recipe residual loss,
- retrieval confidence,
- task-key distance.

Acceptance:
- K/C-vs-N AUROC >=0.90,
- threshold supports K/C expansion <=10%, N >=90%.

---

## A1-R017 — Gradient/subspace novelty diagnostic

Optional if A1-R016 is already robust.

Estimate:
`N_grad = ||g_perp|| / ||g||`.

Adopt only if discrimination benefit justifies compute overhead.

---

## A1-R018 — Learned novelty controller

Acceptance across >=5 seeds:
- K/C expansion <=10%,
- N expansion >=90%.

STOP GATE if this fails.

---

## A1-R019 — Sparse bank scaling check

Evaluate bank sizes such as 8 / 16 / 32 / 64 / 128 where practical.

Report:
- resident primitive params,
- active primitive params,
- primitive forward call count,
- FLOPs estimate/measurement,
- latency.

Acceptance:
- non-selected forward calls = 0,
- active capacity follows top-k rather than bank size.

---

## A1-R020 — Full sequential closed-loop benchmark

Use repeated K/C/N/R.

Run >=5 seeds.

Report:
- performance,
- transitions,
- expansions,
- reuse,
- forgetting,
- resident/active capacity,
- temporary peak,
- compression,
- lifetime train examples/steps,
- wall-clock.

Acceptance:
- prior gates remain valid,
- recurrence reuse materially exceeds Phase A,
- K/C rarely expand,
- N expands,
- active primitive compute is sparse,
- growth below grow-only control.

---

## A1-R021 — Baselines and ablations

Baselines:
- A1-C004 dense shared-core solver,
- no-primitive causal path,
- oracle causal primitive system,
- retrieval sparse system,
- grow-only residual,
- full-task plastic,
- residual plastic,
- full APC.

Ablations:
- no task-blind content path,
- no argument conditioning,
- no recipe library,
- no residual-only learning,
- no functional rank sweep.

Report compute/data/capacity differences explicitly.

---

## A1-R022 — Final Phase A.1 audit

Output:
`docs/results/PHASE_A1_RESULT_FINAL.md`

Include:
1. Phase A history,
2. A1-006 correction context,
3. H1a/H1b/H1c,
4. primitive causal gates,
5. composition,
6. plasticity,
7. consolidation,
8. recurrence,
9. routing,
10. novelty,
11. sparse scaling,
12. baselines,
13. seed robustness,
14. stop conditions,
15. what remains unproven.

Audit only. No new features.
