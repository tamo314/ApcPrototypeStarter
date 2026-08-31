# Phase A Result — Synthetic Closed-Loop Proof of Concept

**Status:** Phase A implementation complete (Milestones A0-A10 / Tasks 001-013); this document is the Task 014 review and closes the phase.

**Scope of evidence:** all measured results below come from CPU-only, single-machine, mostly single-seed (seed 0, plus two extra seeds run for this review) dev-tier runs with small models (64-192 hidden dim, 2-4 layers) on a tiny synthetic vocabulary (6-10 tokens). No milestone-scale (>=5 seed) or GPU run has been executed. Every number quoted here names its source run under `runs/` (gitignored, but reproducible via the commands in `README.md`) so it can be re-derived.

## 1. Verdict

**The mechanical closed loop works. The scientific hypotheses it was built to test are not supported by the evidence gathered so far, at the scale actually tested.**

Concretely:
- `STABLE -> SEARCH -> PLASTIC -> CONSOLIDATE -> SHADOW -> STABLE` executes end to end, repeatedly, with correct capacity bookkeeping, correct shadow-validation gating (failing shadow preserves temporary capacity; passing shadow installs-and-releases atomically), and correct separation of temporary vs. persistent capacity in code. Task 012's literal acceptance criterion ("at least two learn/consolidate/release cycles in one task stream") is satisfied many times over (7-10 cycles per run, see section 4).
- But per `docs/exec-plans/active/PHASE_A.md`'s own "Stop conditions", at least three of the six listed conditions are triggered by the actual measured numbers (section 3.3), robustly across three different seeds (section 6). `PHASE_A.md` is explicit about what this means: *"Pause architecture expansion and investigate ... These are useful negative results; do not hide them by increasing model size prematurely."* This document is that pause-and-report step; Task 013 (baselines) was implemented before this review connected the dots explicitly, which is itself one of this review's findings (section 3.6).
- The root cause, established in `docs/DECISIONS.md` ADR-0006 before this review, is that the Task 003 dense core does not generalize known-operation execution to unseen token content at any scale tried (up to d_model=192) — well below the project's own stated "comfortable" range (`docs/HARDWARE_ENVIRONMENT.md`: 10M-100M parameters; `docs/design-docs/ARCHITECTURE.md` section 3: 10M-60M). Whether the negative result would persist at that larger, doc-recommended scale is an open question this review did not (and could not, within the "no new features" scope of Task 014) resolve.

None of this means APC is falsified as an architecture — it means Phase A has not yet demonstrated it working, and the honest next step per the project's own plan is to investigate the stop conditions (most plausibly: retest at meaningfully larger model/data scale) before adding more scaffolding on top of the current closed loop.

## 2. What was built (Milestones A0-A10)

| Milestone | Task(s) | Delivered |
|---|---|---|
| A0 Bootstrap | 001 | Package skeleton, pytest/ruff/mypy config, seed/system-info utilities |
| A1 Synthetic environment | 002 | Deterministic interpreter + 8 known operations + composition/novel splits |
| A2 Fixed dense baseline | 003 | `DecoderOnlyTransformer`, smoke training, checkpoint reload |
| A3 Primitive bank + router | 004, 005 | `PrimitiveBank`, top-k `Router` with entropy/usage logging |
| A4 Composition benchmark | 006 | K vs. C oracle-labeled evaluation |
| A5 Plastic workspace | 007 | `PlasticWorkspace` + `Allocator` presets |
| A6 Novelty + controller | 008 | Error+entropy novelty score, hysteretic finite-state `Controller` |
| A7 Consolidation | 009, 010 | `SortOp`/`ReverseOp` novel operations; distillation into a candidate primitive |
| A8 Shadow validation | 011 | Compare-then-release, atomic install, bounded retry |
| A9 Sequential benchmark | 012 | `apc.core.execution` (the first real model/bank/workspace wiring) + `sequential_benchmark` runner |
| A10 Baselines | 013 | B0 fixed dense, B1 fixed sparse, B2 grow-only, B3 grow+replay, alongside B4 (APC) |

328 tests pass; `ruff check .` and `mypy src/apc` are clean as of this review's starting commit (`02bbde2`).

## 3. Audit findings

### 3.1 Reproducibility

- **Every run correctly records what `AGENTS.md`'s "Experiment discipline" section requires**: git commit, config, seed, wall-clock, persistent/temporary/active parameter counts, task accuracy, retention, and compression ratio are all present in `system.json`/`config.yaml`/`report.json`/`summary.json` for every script in this repo.
- **The documented installation path does not work on the machine this project has actually been built on.** `pyproject.toml` declares `requires-python = ">=3.12"`; the installed interpreter is Python 3.10.11. `pip install -e .` fails with `ERROR: Package 'apc' requires a different Python: 3.10.11 not in '>=3.12'`. Every task in this repository (001 through 013) has been implemented and verified by prefixing commands with `PYTHONPATH=src` instead of installing the package — a workaround that happens to work because `pytest` already sets `pythonpath = ["src"]`, but the README's own literal documented commands (`python scripts/....py`) have never actually been run in their stated form on this machine. This is a live, unresolved reproducibility gap, not a historical one — re-verified during this review (2026-08-31).
- **Bit-exact reproducibility is not guaranteed and is not claimed.** `apc.utils.seed.set_seed`'s `deterministic_algorithms` flag defaults to `False`; two full runs of the same seeded config can produce matching qualitative metrics (e.g. cycle count) without byte-identical `to_dict()` output, consistent with ordinary CPU floating-point non-determinism from thread scheduling. This was established during Task 012 and re-confirmed by nothing in this review contradicting it.
- **All of Phase A's headline experiments were run at a single seed (0) until this review.** `docs/EXPERIMENT_PLAN.md` section 7 asks for >=5 seeds before treating a result as a milestone claim ("if compute permits" — and here it clearly does: one dev-tier run takes under two minutes). As part of this review I ran two additional seeds (1, 2) of the Task 012 sequential benchmark under the same config; results in section 6 below. This is evidence toward, not a substitute for, the >=5-seed requirement.
- **No run in this repository has ever executed on the target GPU.** `system.json` for every sequential/baseline run correctly reports `"cuda_available": true` (an RTX 5060 Ti is present) but `"peak_vram_bytes": 0`, because `apc.evaluation.sequential_benchmark`/`apc.evaluation.baselines` never move the model off CPU (unlike `apc.core.train.run_smoke_training`, which does support a `device` config). This is a reasonable choice for fast iteration at this model scale, but it means Phase A's results have never been validated against `docs/HARDWARE_ENVIRONMENT.md`'s stated target environment.

### 3.2 Capacity accounting

- **"Resident persistent parameters" (`docs/design-docs/ARCHITECTURE.md` section 12: "stable core + persistent primitives") currently excludes the stable core everywhere except in B0.** `EventReport.persistent_param_count`, `SequentialBenchmarkReport.persistent_parameter_count_final`, and `BaselineReport.persistent_parameter_count_final` for B1-B4 all report `PrimitiveBank`/grown-transform totals only; the frozen dense core's own parameter count (70016 in the dev-tier model config) is never added in. B0 (which has no separate primitive bank) reports `model.num_parameters()` directly, so it alone reflects the full resident cost. This makes the cross-baseline "final persistent capacity" comparison misleading as currently plotted (`apc.evaluation.baseline_plots.plot_persistent_growth_comparison`): B0's bar already includes its whole model; B1-B4's bars sit on top of an equal, unlabeled 70016-parameter base that isn't shown. Correcting for this (`runs/phase_a_baselines/summary.json`, dev-tier, seed 0):

  | Baseline | Reported "persistent" | + shared stable core (70016) | True resident persistent |
  |---|---:|---:|---:|
  | B0 | 70016 | (already included) | 70016 |
  | B1 | 1792 | +70016 | 71808 |
  | B2 | 20480 | +70016 | 90496 |
  | B3 | 20480 | +70016 | 90496 |
  | B4 | 17920 | +70016 | 87936 |

  On the corrected basis, **APC (B4) is larger than the fixed-dense baseline (B0), not smaller** — the "bounded growth" story only holds when B4 is compared against the uncontrolled grow-only baselines (B2/B3), which is the comparison H4 is actually about, but does not hold against B0 the way the current bar chart visually implies.

- **"Active parameters per inference step" is, as implemented, numerically identical to "persistent parameters" for every router-based path, in every single event of every run measured.** Verified directly: in `runs/phase_a_sequential/report.json`, `persistent_param_count == active_param_count` for all 10 events, with no exception. Root cause: `apc.core.execution.apply_bank` computes **every enabled bank primitive's** residual delta unconditionally for every forward pass — the router's top-k selection only zeroes out the *weight* of non-selected primitives in the summed output, it does not skip computing them. `EventReport.active_param_count` is then derived from `bank.active_parameter_count(stable_candidate_ids(bank))`, i.e. "every enabled candidate," never the router's actual per-example `selected_ids`. `docs/design-docs/ARCHITECTURE.md` section 12 is explicit that these three sizes must be tracked *separately* and that "the core research claim depends on not conflating these" — as it stands, Phase A has never actually exercised or measured true sparse compute, despite sparse execution being the architecture's central premise (section 1). This is not a router bug (top-k selection and weighting are correct, see `test_primitives_router.py`) and not a bank/workspace accounting-method bug (`bank.active_parameter_count`/`workspace.active_parameter_count` are correctly unit-tested at the method level in `test_primitives_bank.py`/`test_plastic_workspace.py`) — it is a wiring/integration gap between the two. B0's and B2/B3's `active_param_count == persistent_param_count` is, by contrast, *correct* for their architectures (B0 has no sparsity at all; B2/B3 deliberately apply every grown transform unconditionally, by ADR-0011's own design) — the gap is specific to the router-based paths (B1, B4).
- **Temporary peak parameter tracking is correct**: verified as a true point-in-time peak (one allocator-preset batch's size, e.g. 2048 for `SMALL`), not a cumulative sum across the whole run, matching section 12's definition.
- **Consolidation's compression margin is real but thin, and close to a stop-condition boundary.** Every one of the 10 cycles in the seed-0 dev-tier run compresses `4 transforms x rank 4` (2048 parameters) down to one `rank 14` candidate (1792 parameters) — a compression ratio of 0.875 (the candidate keeps 87.5% of the temporary budget). This sits close to the `PHASE_A.md` stop condition "consolidation repeatedly needs almost all temporary parameters." The likely mechanistic reason: `docs/design-docs/ARCHITECTURE.md` section 10 step 4 ("cluster transforms by functional similarity if more than one remains") was explicitly scoped out of Task 010, so consolidation always distills the combined delta of independently-trained, non-redundant temporary transforms with nothing removing redundancy first — there is little redundancy to remove, so a large fraction of the allocated rank is genuinely needed.

### 3.3 Result validity — and cross-reference to `PHASE_A.md`'s stop conditions

The sequential benchmark's own saved output already answers several of Phase A's headline questions, and the answer is a negative result on three of `PHASE_A.md`'s six explicit stop conditions, robust across seeds (see section 6 for the multi-seed table):

| Stop condition (`PHASE_A.md`) | Triggered? | Evidence |
|---|---|---|
| No distinction between novel composition and novel operation is measurable | **Yes** | `generalization_gap_known_vs_composition` and `novelty_gap_composition_vs_operation` are exactly `0.0` in every seed run (0, 1, 2) |
| Consolidation repeatedly needs almost all temporary parameters | **Borderline** | 87.5% of temporary budget retained every cycle (section 3.2) |
| New primitives are not reused on recurrence | **Yes** | `num_reused_without_new_cycle: 0` in every seed run (0 of 2 `R` events reused an existing primitive) |
| Controller expands on most familiar tasks | **Yes** | 7-10 of 10 events triggered a full cycle in every seed run, including plain `K` events (section 6's per-event table) |
| Retention requires replay buffers so large they dominate the system | No | `replay_buffer_max_events=6` stays small relative to the run |
| Dynamic model is consistently worse than all static baselines at equal compute | Not established either way | Compute is not equal across baselines (section 3.4); on the numbers available, B4 beats B0 on retention but at ~3.4x the compute, and beats the true "dynamic, no consolidation" baselines (B2/B3) on both retention and (corrected) persistent size |

The first three are unambiguous, not edge cases requiring interpretation. `PHASE_A.md`'s instruction for this situation is explicit: pause and investigate rather than proceed. Task 013 (baselines) was implemented after Task 012 produced these numbers, without an explicit pause-and-investigate step tied to this stop-condition list — see 3.6.

### 3.4 Baseline fairness

- **Data fairness holds by construction.** `apc.environments.generator.TaskGenerator.generate` derives its own RNG per `(seed, split)` via SHA-256 (`_derive_seed`), independent of any global RNG state; `apc.evaluation.stream.build_task_generators`/`EventExamplePool` are the exact functions all five baselines call with the same `SequentialBenchmarkConfig` fields. Every baseline sees byte-identical per-event examples regardless of what else happens in its own control flow. Verified by code inspection, not just by construction-intent.
- **Model init/pretraining fairness holds up to floating-point noise.** `set_seed(config.seed)` runs before any baseline-specific object construction that consumes additional RNG (e.g. B1's `Router`/`PrimitiveBank`, constructed only after the shared dense model's own weights are already initialized); pretraining itself has no additional randomness (dropout=0, deterministic AdamW). All five baselines reported `pretrain_exact_match: 0.0` in the real dev-tier run — consistent with, though not independently decisive proof of, identical pretraining.
- **Budget fairness is partial by explicit design, not by oversight, but the resulting compute ratio is large and should be stated whenever a cross-baseline comparison is quoted.** Every baseline's per-event training shares the same plateau-stopping budget (`apc.evaluation.stream.train_until_plateau`, reading `SequentialBenchmarkConfig.plastic`/`controller`), and B1's one seed cycle reuses the same consolidation/shadow config B4 uses per cycle. But total compute spent differs by over an order of magnitude across the run actually measured (`runs/phase_a_baselines/summary.json`, dev-tier, seed 0): B1=1300, B0=6300, B2=B3=12200, B4=21300 train-steps. AGENTS.md: *"Never compare runs with materially different data budgets without saying so."* The comparison plots (`apc.evaluation.baseline_plots`) do surface the compute numbers alongside the retention numbers, but no prose in the existing README/script output states the ratio next to the retention claim — this review's section 7 states it explicitly for the numbers quoted here, and any future presentation of these results should keep doing so.
- The persistent-parameter conflation in 3.2 affects fairness too: it makes B1-B4 look uniformly smaller relative to B0 than they actually are on a true resident-parameter basis.

### 3.5 Failure cases

- Router-selection fidelity degrading as the bank grows (ADR-0009) is the most likely mechanistic explanation for the 0/2 reuse failures in 3.3: shadow validation genuinely passes at consolidation time (task-score-ratio 0.95-0.99), but the router increasingly mis-routes or under-weights the correct existing primitive for its own recurring content as more primitives accumulate on an already low-information hidden-state signal (ADR-0006's root cause).
- B1's one-time seed cycle (ADR-0010) can fail shadow validation and leave the fixed bank empty; this is handled without crashing and is honestly reflected in the report (`tests/test_baselines.py` deliberately does not assume the seed cycle passes).
- The bounded-shadow-retry-then-give-up path (Task 011/012, `gave_up=True`, capacity preserved) is exercised by a dedicated unit test (`test_failing_shadow_preserves_temporary_capacity_and_retries`) but has never been triggered by an actual Phase A run with realistic (non-deliberately-unreachable) thresholds — every real run's cycles passed shadow validation on the first or a subsequent ordinary attempt. Its correctness currently rests on the unit test alone, not on any observed real run.

### 3.6 Documentation drift

- **Scale drift, undocumented.** `docs/design-docs/ARCHITECTURE.md` section 3 recommends 192-384 hidden size / 10M-60M parameters as the "recommended first scale"; `docs/HARDWARE_ENVIRONMENT.md` calls 10M-100M "comfortable". No experiment in this repository has reached that range: Task 003's smoke config (192d, 4 layer, ~1.8M params) is the closest attempt and is still roughly 5-30x below the recommended floor; Tasks 012/013's dev-tier configs (64d, 2 layer) are one to two further orders of magnitude smaller. This is a defensible, explicit engineering trade-off for iteration speed (AGENTS.md: "prefer the smallest implementation that can falsify the current hypothesis") but neither doc was updated to note that Phase A's actual measured numbers come from far below their own stated targets. A reader of `ARCHITECTURE.md`/`HARDWARE_ENVIRONMENT.md` alone would reasonably expect Phase A's headline numbers to come from a 10M+ parameter model; they do not, at any point in this repository's history.
- **The Python version mismatch (3.1 above) is itself undocumented drift**: nothing in `docs/DECISIONS.md` or `README.md` notes that the actual development interpreter differs from `pyproject.toml`'s declared `requires-python`.
- **ADR-0006/ADR-0009 describe the generalization/router-degradation findings accurately but never cross-reference `PHASE_A.md`'s own stop-condition list**, which describes overlapping symptoms from the plan's perspective (section 3.3 above draws that connection for the first time). Future ADRs describing a measured limitation should check whether it also names one of `PHASE_A.md`'s stop conditions, and say so.
- `apc.evaluation.sequential_benchmark_plots.plot_lifetime_compute_proxy`'s docstring/title referenced "Task 013" as future work; this was corrected in the same commit that landed Task 013 and is not an open item.

## 4. Hypothesis-by-hypothesis verdict (`docs/EXPERIMENT_PLAN.md` section 1)

| Hypothesis | Verdict | Basis |
|---|---|---|
| H1 Selective expansion (expand more for N than for held-out C) | **Not supported** | `generalization_gap`/`novelty_gap` are exactly 0.0 in every seed; 7-10/10 events expand regardless of label |
| H2 Consolidation (compress while retaining performance) | **Weakly supported** | Real if modest compression (0.875 ratio) every cycle; shadow's task-score-ratio gate (>=0.95) is met by construction of what "passed" means, so this is closer to "the gate works as designed" than "compression is aggressive" |
| H3 Reuse (consolidated primitive reduces adaptation cost on recurrence) | **Not supported** | 0/2 recurrence events reused an existing primitive without a new cycle, in every seed |
| H4 Bounded persistent growth (less than grow-only, less than a fixed-dense baseline) | **Partially supported** | Holds against the correct comparison (grow-only B2/B3: 87936 < 90496 corrected) but not against B0 once the shared stable core is counted (87936 > 70016) |
| H5 Retention (better than naive fine-tuning) | **Supported in relative terms only** | Mean backward transfer -0.094 (B4) vs. -0.794 (B0) vs. -0.294 (B2) vs. -0.156 (B3) — but B4 still forgets substantially in absolute terms (max forgetting 0.9375) and spends ~3.4x B0's compute to get there |

## 5. Baseline comparison (dev-tier, seed 0; `runs/phase_a_baselines/`)

| Baseline | Reported persistent | True resident persistent (+70016 core) | Total train steps | Max forgetting | Mean backward transfer |
|---|---:|---:|---:|---:|---:|
| B0 fixed dense | 70016 | 70016 | 6300 | 1.000 | -0.794 |
| B1 fixed sparse | 1792 | 71808 | 1300 | 0.000 | 0.000 |
| B2 grow-only | 20480 | 90496 | 12200 | 1.000 | -0.294 |
| B3 grow+replay | 20480 | 90496 | 12200 | 1.000 | -0.156 |
| B4 APC | 17920 | 87936 | 21300 | 0.9375 | -0.094 |

B1's zero forgetting is vacuous, not a genuine strength: per ADR-0006/section 3.3, the seed-and-freeze bank cannot solve fresh content any better than chance, so "nothing changes" is trivially true rather than evidence of good retention of a working capability. B3 vs. B2 is the one comparison in this table that isolates a single mechanism cleanly (replay, holding growth policy fixed) and shows the expected direction: replay measurably reduces mean backward transfer (-0.156 vs. -0.294) without eliminating it.

## 6. Seed-robustness check (added by this review)

Re-ran the Task 012 sequential benchmark at seeds 1 and 2 under the same dev-tier config (`configs/phase_a_sequential.yaml`), in addition to the existing seed-0 result:

| Seed | Cycles (of 10 events) | Reused on recurrence | Max forgetting | Mean backward transfer | Persistent (bank only) | Generalization gap | Novelty gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0/2 | 0.9375 | -0.094 | 17920 | 0.0 | 0.0 |
| 1 | 7 | 0/2 | 0.875 | -0.088 | 12544 | 0.0 | 0.0 |
| 2 | 9 | 0/2 | 1.000 | -0.100 | 16128 | 0.0 | 0.0 |

Per-event cycle pattern (`K C N K C N K C R R`, `True` = a full learn/consolidate/release cycle occurred):

- seed 0: `True True True True True True True True True True`
- seed 1: `True False False True True False True True True True`
- seed 2: `True True True True True False True True True True`

The generalization/novelty gaps are exactly `0.0` and reuse is exactly `0/2` in **all three** seeds — this is not a seed-0 artifact. The cycle count varies (7-10) but plain `K` events still expand in 8 of 9 (seed1, seed2) `K`-event instances across the two extra seeds, confirming "controller expands on most familiar tasks" is a robust property of this configuration, not noise. Three seeds is short of `EXPERIMENT_PLAN.md` section 7's >=5-seed bar for a milestone claim, but it is enough to rule out "unlucky seed 0" as the explanation.

## 7. Recommendations (not actioned by this review — Task 014 is audit-only)

1. Investigate the triggered stop conditions before adding further architecture (per `PHASE_A.md`'s own instruction), most plausibly by retesting ADR-0006's generalization finding at the project's own stated "comfortable" scale (10M+ parameters) before concluding the negative result is scale-independent.
2. Fix the `requires-python`/interpreter mismatch (either relax the declared requirement to match what has actually been tested, or install Python 3.12 and re-verify) so the documented installation path in `README.md` actually works.
3. Either wire genuine top-k-restricted compute into `apc.core.execution.apply_bank` or relabel/re-derive `active_param_count` so it stops being numerically identical to `persistent_param_count` — as it stands, the metric `docs/design-docs/ARCHITECTURE.md` calls load-bearing for the "core research claim" carries no information beyond persistent count.
4. Include the frozen stable core in every reported "persistent parameter count" (or clearly rename the existing fields to "primitive-only persistent parameters" everywhere they are surfaced: reports, plots, and docs) so cross-baseline comparisons are not accidentally apples-to-oranges.
5. Run >=5 seeds before treating any Phase A number as a milestone claim, per `docs/EXPERIMENT_PLAN.md` section 7.

## 8. Addendum: recommendation 1 followed up (model-scale axis ruled out)

Recommendation 1 above (retest ADR-0006 at recommended scale) was carried out immediately after this review, as a root-cause investigation task. Two environment fixes were made first: a Python 3.12 virtual environment (both native Windows and WSL2/Ubuntu, via `uv`) now satisfies `pyproject.toml`'s `requires-python` and installs `apc` in editable mode successfully for the first time (`.venv`, `.venv-wsl`, both gitignored) -- addressing recommendation 2 for these two environments specifically (`pyproject.toml` itself is unchanged). The WSL environment additionally has working CUDA (RTX 5060 Ti), the first GPU-backed run in this repository's history.

Three new configs (`configs/phase_a_scale_check_{low,mid,high}.yaml`: 384/512/640 hidden size x 8 layers, 14.2M/25.3M/39.4M parameters, spanning `ARCHITECTURE.md` section 3's full 10M-60M recommended range) were trained from scratch on GPU and evaluated exactly as ADR-0006 originally measured (`scripts/composition_benchmark.py`, held-out `test`-split exact match vs. training exact match). Result, recorded in full as `docs/DECISIONS.md` ADR-0013: held-out exact match stays flat at chance (0.008-0.016) across a 22x parameter range, while training exact match is a perfect 1.0 at every scale. **Model scale, on its own, is ruled out as the explanation for ADR-0006's finding.**

This leaves training-data quantity/coverage and training duration ("grokking"-style late-phase generalization) as the leading untested candidate causes; see ADR-0013's consequence section. This addendum does not change section 1's verdict -- it removes one specific caveat from it (scale) rather than reversing it.

A side finding surfaced while building the two new Python 3.12 environments, unrelated to the scale question but relevant to reproducibility (section 3.1): `tests/test_sequential_benchmark.py::test_two_events_each_complete_a_learn_consolidate_release_cycle` passes reliably under `torch==2.11.0` (the version the original Python 3.10 environment happened to have) and fails reliably under `torch==2.13.0` (the version `pip install -e .` naturally resolves to today, since `pyproject.toml` only requires `torch>=2.7`) -- confirmed 3/3 each way on Windows. This is deferred (by request) to be fixed alongside the test itself after the current root-cause investigation, not re-litigated here.

## 9. Addendum: recommendation-1 follow-up's next hypothesis (training-data quantity ruled out up to 4096 examples)

With model scale ruled out (ADR-0013), the root-cause investigation continued into the next candidate cause: whether the 256-example training set was simply too small/undiverse for the model to find anything beyond a per-example lookup table. `configs/phase_a_data_scale_{1024,4096}.yaml` reran the same held-out-generalization measurement at 4x and 16x the original example count. Result, recorded in full as `docs/DECISIONS.md` ADR-0014: held-out exact match stays flat at chance (0.008-0.016) across a 16x increase in training examples, while training exact match remains a perfect 1.0 throughout. **Training-data quantity, on its own, is ruled out as the explanation for ADR-0006's finding, over the 256-4096 example range tested.**

Pushing the data axis further hit a practical wall rather than a scientific one: `apc.core.train.run_smoke_training` collates the entire dataset into a single batch reused every gradient step, so activation memory scales directly with `num_examples` independent of model size. The first 4096-example attempt (at the 14.2M-parameter model used for the 256/1024 points) overflowed the 16GB GPU's VRAM and was killed after ~4 hours stuck thrashing at step 900/1500; rerunning at the original 1.79M-parameter model dimensions fit comfortably and completed in ~4 minutes. Testing 16384+ examples under this same training loop would need either a further-shrunk model (risking insufficient capacity to fit the training set at all) or implementing minibatching/gradient accumulation in the training loop itself -- a real code change, not just a new config, and not undertaken here.

This narrows the untested candidate causes from ADR-0013 to one: training duration / "grokking"-style late-phase generalization. Loss is already fully converged (~1e-4) after only 1500 steps at every combination of scale and data quantity tried so far in this investigation, so none of these runs can distinguish "no generalizing solution exists for this task representation" from "one exists but was never reached because memorization is the lower-loss optimum found first."
