# Codex Task Queue — Phase A.1

Execute exactly one task at a time. Every task ends with changed files, tests, experiment commands, run-artifact paths, acceptance criteria pass/fail, assumptions, and required ADRs.

If a task contains **STOP GATE**, do not start later dependent tasks when it fails.

---

## A1-001 — Reproducibility and accounting repair

**Goal:** fix Phase A measurement issues before new science.

**Work:** make editable install match the declared Python requirement; constrain/document PyTorch version sufficiently to avoid silent test drift; include Stable Core in resident totals; preserve primitive-only metrics separately; add report-schema tests.

**Accept:** install works; all existing tests pass; tests prove `resident_total = stable_core + persistent_bank`; old Phase A run files remain unchanged.

---

## A1-002 — True top-k primitive execution

**Goal:** make routing sparse in actual computation.

**Work:** refactor `apc.core.execution.apply_bank` or current equivalent to gather and execute only selected primitive IDs; support batched top-k; instrument primitive forward calls; correct active counts.

**Accept:** non-selected forward calls are zero; selecting all primitives reproduces dense-equivalent output; active primitive count equals selected capacity; resident and active counts differ when bank size > top-k.

**STOP GATE** if sparse execution is numerically incorrect.

---

## A1-003 — Online procedural generator

**Goal:** remove finite-dataset lookup as the main solution.

**Work:** generate fresh examples every step/batch; deterministic under `(seed, step, split)`; preserve interpreter truth; expose K/C/N/R and decomposition metadata to evaluation/oracle code only.

**Accept:** reproducible same-seed generation, fresh content across steps, exact interpreter outputs, no fixed finite train set required.

---

## A1-004 — Symbol permutation anti-shortcut control

**Goal:** prevent stable token IDs from encoding semantics.

**Work:** per-batch/episode symbol permutation; log permutation identity; test semantic invariance.

**Accept:** same abstract task appears under multiple token mappings; decoded interpreter result is invariant; learned paths do not receive the inverse mapping as a shortcut.

---

## A1-005 — Task/content factorized Stable Core

**Goal:** expose separate task and content states.

**Work:** add an API conceptually like `encoded.task_state` (`z_task`) and `encoded.content_state` (`h_content`). Routing/novelty uses `z_task`; primitives transform `h_content`.

**Accept:** interfaces/shapes tested; states are independently probeable/loggable; Phase A compatibility preserved where practical.

---

## A1-006 — Stable Core systematic-generalization gate

**Goal:** prove known-operation generalization before APC machinery.

**Work:** train K-only on online generated + permuted data; evaluate unseen content; >=5 seeds; no plastic/consolidation path.

**Accept:** mean unseen-content exact match >=0.95 with all seeds reported.

**STOP GATE:** if failed, investigate representation/training objective only.

---

Before Task A1-007, complete every mandatory task in
`docs/CODEX_TASKS_PHASE_A1_CORRECTION.md`.

---

## A1-007 — Oracle primitive routing

**Goal:** verify primitive execution independently of learned routing.

**Work:** environment supplies oracle primitive IDs to the oracle controller; learned router fully bypassed.

**Accept:** oracle-routed K >=0.95; selected IDs exactly match metadata.

---

## A1-008 — Composition Library + oracle composition/search

**Goal:** separate unseen recipes from new primitives.

**Work:** add `CompositionLibrary`; ordered recipes; multi-step execution; exhaustive/beam search for small banks; add C tasks.

**Accept:** oracle C >=0.90; designated search benchmark recovers a valid recipe; successful C allocates no Plastic Workspace.

**STOP GATE** if oracle C fails.

---

## A1-009 — Oracle novelty controller

**Goal:** temporarily remove novelty detection from the experiment.

**Work:** ground-truth K/C/N/R controls transitions: K/C no expansion, N plastic, R reuse.

**Accept:** K/C expansion <=10%; N expansion >=90%; deterministic transition logs.

---

## A1-010 — Residual Plastic Workspace

**Goal:** learn only computation not explained by existing recipes.

**Work:** compute best existing solution; train workspace on residual; retain full-task plastic control; log existing vs residual contribution.

**Accept:** designated N reaches >=0.95 after adaptation; composed residual output is numerically tested; no persistence yet.

---

## A1-011 — Compressibility-controlled novel operations

**Goal:** make the existence of a compact target known to evaluation.

**Work:** hidden low-rank ground-truth transforms; weights never exposed to learner; metadata records ground-truth rank; deliberately overcomplete temporary capacity (initial suggestion: GT rank 4 vs 16 rank-4 temporary transforms or comparable capacity).

**Accept:** workspace learns held-out generated content; no weight leakage; ground-truth complexity recorded.

---

## A1-012 — Functional consolidation rank sweep

**Goal:** distill the temporary function rather than prune temporary weights.

**Work:** collect `(h, delta_h)` probes; fit candidates at increasing ranks; choose smallest passing functional/task thresholds; log agreement-vs-rank curve; then run shadow validation.

**Accept on controlled N:** functional agreement >=0.99; performance retention >=0.95; candidate/temporary parameter ratio <=0.50.

**STOP GATE** if a known compact target cannot be recovered materially below temporary capacity.

---

## A1-013 — Oracle recurrence reuse

**Goal:** prove consolidated primitive generalization independently of routing.

**Work:** reintroduce learned N with fresh content; force its installed primitive/recipe; disable new allocation unless oracle reuse itself fails.

**Accept:** R >=0.95; approximately zero new adaptation steps; >=90% of R events require no new cycle.

**STOP GATE:** failure here means routing is not the root cause.

---

## A1-014 — Retrieval routing baseline

**Goal:** add an interpretable bridge between oracle and learned routing.

**Work:** store task-key prototype(s); similarity retrieval; report retrieval accuracy as bank grows.

**Accept:** R >=0.90; reuse without new cycle >=90%; retrieved item matches oracle item >=0.95 on labeled eval.

---

## A1-015 — Learned router reintroduction

**Goal:** test learned routing after retrieval works.

**Work:** train router from `z_task`; labels only for supervision/evaluation; log top-k inclusion and bank-size curves.

**Accept:** oracle-required primitive in selected top-k >=0.95; recurrence reuse >=0.90; no severe monotonic collapse across Phase A.1 bank sizes.

**STOP GATE** if recurrence routing collapses.

---

## A1-016 — Residual computational novelty baseline

**Goal:** define novelty as existing-computation insufficiency.

**Work:** combine best-existing residual loss, retrieval confidence, and optional task-key distance; compare against old error+entropy baseline.

**Accept:** K/C-vs-N discrimination >=0.90 AUROC or predeclared equivalent; threshold achieves K/C expansion <=10% and N >=90% on held-out data.

---

## A1-017 — Gradient/subspace novelty experiment (optional)

**Goal:** test whether gradient novelty improves C-vs-N separation.

**Work:** compact task-gradient representations; basis for existing tasks/primitives; estimate `||g_perp|| / ||g||`; report compute overhead.

**Accept:** adopt only if it materially improves discrimination/trade-off over A1-016. May be skipped if A1-016 is robustly sufficient.

---

## A1-018 — Learned novelty controller

**Goal:** replace oracle novelty with the best learned signal.

**Work:** integrate novelty metric with hysteresis and current controller.

**Accept across >=5 seeds:** K/C expansion <=10%; N expansion >=90%.

**STOP GATE** if failed.

---

## A1-019 — Full Phase A.1 sequential benchmark

**Goal:** run the complete learned K/C/N/R loop.

**Report:** accuracy, expansion cycles, reuse, forgetting, resident total/primitive params, active total/primitive params, temporary peak, compression, lifetime train steps/examples, and wall-clock.

**Accept:** H1-H9 gates already pass; recurrence reuse materially exceeds Phase A; K/C normally avoid expansion; N expands; active primitive compute is truly sparse; controlled consolidation remains <=0.50; growth below grow-only control. Run >=5 seeds.

---

## A1-020 — Corrected baselines and ablations

**Baselines:** fixed dense; oracle primitive; retrieval sparse; grow-only residual; full-task plastic+consolidation; residual plastic+consolidation; full APC A.1.

**Ablations:** no symbol permutation; no task/content split; no residual learning; no recipe library; no retrieval baseline; no functional rank sweep.

**Accept:** comparisons use corrected resident totals and explicitly state data/compute budgets.

---

## A1-021 — Phase A.1 final audit

**Goal:** scientific verdict only; no new features.

Create `docs/results/PHASE_A1_RESULT.md` with: scope, verdict, implementation summary, reproducibility, accounting, stop-condition audit, hypothesis verdicts, baselines, seed robustness, failure cases, recommendations.

---

## Optional D-A1 — Grokking duration diagnostic

Run the old fixed-dataset Stable Core much longer with suitable regularization/checkpoints (for example 50k/100k/300k). Log train/test exact match, loss, weight norm, and relevant entropy metrics.

This does not replace A1-006 and must not be used to bypass the online-generalization gate.
