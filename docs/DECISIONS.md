# Architecture Decision Log

Use this file for short decisions discovered during implementation. Do not rewrite history; append entries.

## ADR-0001 — Validate the learning loop before using a language model

**Status:** Accepted

**Decision:** Phase A uses synthetic symbolic tasks rather than a pretrained LLM.

**Reason:** The first scientific question is whether selective expansion, consolidation, release and reuse can form a stable closed loop. LLM scale would confound failures in routing, novelty detection, consolidation, continual learning and optimization.

**Consequence:** Early results demonstrate architecture behavior, not language capability.

---

## ADR-0002 — Use low-rank residual transforms as the first primitive type

**Status:** Accepted

**Decision:** Persistent and temporary primitives share a low-rank residual transform interface.

**Reason:** Low-rank transforms are cheap, differentiable, easy to count, easy to compress and compatible with sparse routing.

**Consequence:** Phase A may fail on operations needing richer computation. If so, add a richer primitive type only after proving the limitation experimentally.

---

## ADR-0003 — Rule-based meta-controller before learned controller

**Status:** Accepted

**Decision:** Use a finite-state controller with configurable thresholds and hysteresis in Phase A.

**Reason:** A learned RL controller would make failure attribution substantially harder.

**Consequence:** Controller optimality is not a Phase A claim.

---

## ADR-0004 — Stable primitives are frozen in the strict Phase A experiment

**Status:** Accepted

**Decision:** New learning occurs in temporary capacity. Existing persistent primitives do not update in the primary experiment.

**Reason:** This creates a clear forgetting boundary and makes consolidation effects measurable.

**Consequence:** Later phases must test controlled metaplastic updates because a permanently frozen bank may eventually become inefficient.

---

## ADR-0005 — Resource release requires shadow validation

**Status:** Accepted

**Decision:** Temporary capacity remains available until the consolidated candidate passes current-task and prior-task validation.

**Reason:** Immediate deletion can hide lossy consolidation and produce irreversible failures.

**Consequence:** Peak memory temporarily contains both candidate and temporary solutions.

---

## ADR-0006 — The Task 003 dense core does not generalize to unseen token content

**Status:** Accepted

**Decision:** Task 012's sequential benchmark evaluates novelty, the PLASTIC accuracy gate, and shadow validation's "current task" signal against the same fixed example set an event trains on, rather than a held-out split of fresh instances of the same task.

**Reason:** Measured while building Task 012: the fixed dense baseline (Task 003) does not generalize known-operation execution to token content it did not train on, at every data/model scale tried (16-512 training examples; d_model 32-192; 1-4 layers; 200-1500 steps) -- held-out exact match stays at chance (~0.02), no better than an untrained network, while train-set exact match reaches ~1.0. This is visible in the pre-existing `runs/phase_a_smoke` checkpoint too: its `composition_benchmark.json` shows `known` (held-out `test` split) exact match of 0.016, statistically indistinguishable from `novel_composition`. No task before 012 measured this, because Task 003's acceptance criterion is memorization of a fixed set and Tasks 006/009's benchmarks only require correct reporting mechanics, not a small generalization gap.

**Consequence:** A `K`/`C` event in the sequential benchmark can legitimately escalate through PLASTIC if the core cannot even fit its own small example set, which weakens (but does not fabricate around) H1's selective-expansion story within a single run; whether K/C reliably resolve via STABLE/SEARCH depends on future work improving core generalization (larger pretraining budgets, relative positional encoding, or a curriculum), not on anything Task 012 can fix by construction. Report generalization/novelty gaps as measured, including near-zero or negative ones.

---

## ADR-0007 — A permanent "null" routing candidate, calibrated after consolidation

**Status:** Accepted

**Decision:** The router's candidate set always includes a permanent, unassigned `NULL_PRIMITIVE_ID` alongside real bank primitives (`apc.core.execution`); after a primitive is consolidated, a short supervised step (`_calibrate_router` in `apc.evaluation.sequential_benchmark`) fits the router's query projection and every registered key so the primitive's own consolidation-batch hidden states route to it and replay-buffer hidden states route to null.

**Reason:** `apc.primitives.router.Router` normalizes scores via softmax over the candidate set; with only one or two real primitives in the bank, that softmax is either trivially 1.0 (one candidate: nothing to compare against) or an arbitrary, untrained split -- either way a persistent primitive would apply itself to every input regardless of relevance, corrupting known/composition tasks once anything is consolidated. Nothing in Tasks 005-011 trained the router's keys at all, since no prior task wired a real model through the bank.

**Consequence:** Selective reuse is only as good as this calibration step's data (the primitive's own consolidation batch vs. whatever is in the replay buffer at the time); it is a lightweight, explicitly-scoped addition for Task 012, not a general router-training procedure, and its failure mode (a recurrence event not being routed to the right primitive) shows up honestly as an unreused recurrence requiring a fresh PLASTIC cycle rather than as a crash or a hidden fabrication.

---

## ADR-0008 — Sequential benchmark stream drops Milestone A9's mixed known+novel composition event

**Status:** Accepted

**Decision:** `apc.evaluation.sequential_benchmark.default_task_stream` uses two independent novel operations (`SORT`, `REVERSE`) each followed by its own recurrence, instead of Milestone A9's illustrative "novel composition using X + old primitives" event.

**Reason:** `apc.environments.generator.TaskGenerator`'s `novel_operation` split is depth-1 only by construction (Task 009), so a novel operation can never be composed with known operations into a multi-step chain without new environment support. Building that support is an environment change, out of scope for Task 012.

**Consequence:** The stream cannot demonstrate a novel primitive being reused *inside* a new composition, only reused standalone (via `R` events). `docs/exec-plans/active/PHASE_A.md` Milestone A9's exact example stream is not literally reproduced; a future task should extend the generator before attempting that event.

---

## ADR-0009 — Router-selection fidelity degrades as the bank grows, tracing back to ADR-0006

**Status:** Accepted

**Decision:** No additional mechanism was added to compensate; Task 012 reports post-cycle token-level exact match and retention as measured, including the degradation, rather than engineering around it.

**Reason:** Measured while building Task 012's sequential benchmark: with the tuned defaults (`candidate_rank=14`, `max_retention_degradation=0.08`), every event in a 10-event dev-tier run completed a full learn/consolidate/release cycle with shadow validation genuinely passing (hidden-state task-score-ratio 0.95-0.99). But post-cycle token-level exact match, measured through the bank/router, was strong for the first consolidated primitive (0.9) and fell toward 0 for most later ones despite equally-passing shadow reports. The router's calibration step (`_calibrate_router`) classifies raw core hidden states per primitive, and per ADR-0006 those hidden states are not well-differentiated for content the core never generalized to; as more primitives (and anchors) accumulate, the classification problem the router must solve gets harder on an already low-information signal, so it increasingly mis-routes or under-weights the correct primitive for its own task.

**Consequence:** In this run's configuration, "at least two learn/consolidate/release cycles" (Task 012's stated acceptance) is satisfied on the shadow-validation definition of a cycle, but H3 (reuse reduces adaptation cost) and H5 (retention) are not demonstrated cleanly -- `SequentialBenchmarkReport.retention`/`num_reused_without_new_cycle` will often show large forgetting and zero measured reuse under these settings. This is one symptom of ADR-0006's root cause, not a separate bug in the router or consolidation code; improving it requires improving core generalization (or moving routing to a representation that survives it), which is future work, not a Task 012 fix.

---

## ADR-0010 — B1's fixed primitive bank is populated by one unconditional seed cycle, not left empty

**Status:** Accepted

**Decision:** `apc.evaluation.baselines.B1Runner` ("fixed sparse", Task 013) runs exactly one learn -> consolidate -> shadow cycle on a slice of known-operation data immediately after pretraining -- reusing `apc.consolidation.distill.consolidate`, `apc.consolidation.shadow.run_shadow_validation`, and `apc.evaluation.stream.calibrate_router`, the same functions B4's own cycle uses -- then freezes the bank and router for the rest of the run. The controller/novelty machinery is never invoked again regardless of event label. If that one shadow validation fails, B1 proceeds with whatever ended up in the bank (possibly still empty) rather than retrying.

**Reason:** `docs/EXPERIMENT_PLAN.md` section 5 defines B1 as "stable core + fixed primitive bank + router, no expansion" without specifying where the fixed bank's content comes from. A bank left empty routes over zero real candidates, which `apc.core.execution.apply_bank` treats as a no-op -- B1 would then be numerically identical to B0 (minus routing overhead), which is not an informative comparison point. Running one unconditional cycle gives B1 genuine (if never-updated) sparse capacity to contrast against B4's repeatedly-updated capacity.

**Consequence:** B1's reported behavior depends on whether that one seed cycle happens to pass shadow validation, which per ADR-0006/ADR-0009 is not guaranteed at these scales. `apc.evaluation.baselines` tests this baseline's "no expansion, ever" property structurally (identical persistent parameter count across every event, zero training steps per event) rather than asserting the seed cycle passes, since the pass/fail outcome itself is not the property Task 013 needs B1 to demonstrate.

---

## ADR-0011 — B2/B3 grow-only baselines bypass the primitive bank and router entirely

**Status:** Accepted

**Decision:** `apc.evaluation.baselines._GrowRunner` (B2, B3) does not use `apc.primitives.bank.PrimitiveBank` or `apc.primitives.router.Router`. Each event allocates a fresh batch of low-rank transforms (same `Allocator`/preset as B4), trains them, then freezes and permanently appends them to a plain `list[Primitive]` (`self.grown`) that is summed unconditionally into every forward pass via `apc.core.execution.sum_primitive_deltas` -- no routing, no gating, no consolidation. Growth happens for every event unconditionally, with no novelty-gated decision of whether to grow.

**Reason:** A "grow-only" baseline in the continual-learning literature (e.g. Progressive Neural Networks) adds an unconditionally-active block of capacity per task; there is no sparse-selection decision to make; that decision is exactly what APC's router/consolidation loop exists to make learnable in the first place. Wiring B2/B3's ever-growing, never-compressed capacity through a `Router` would reintroduce the same small-bank discriminability problem ADR-0009 documents, into a baseline whose entire point is architectural simplicity, and unconditional growth (rather than novelty-gated growth) is what makes the "bounded persistent growth" comparison (H4) meaningful -- B2/B3 have no mechanism to grow selectively, so their persistent parameter count should visibly outgrow B4's over the same stream, and does (measured on the `configs/phase_a_sequential.yaml` dev-tier stream: B2/B3 end at 20480 persistent parameters after 10 events, versus B4's 17920).

**Consequence:** B2/B3's "active parameters per inference step equals persistent parameters" (every grown transform is always active, never gated) is itself a reportable architectural property contrasting directly with B4's sparse routing. B3's replay loss (see ADR-0012) is the only mechanism available to reduce forgetting in this design; measured on the same dev-tier stream, B3's mean backward transfer (-0.156) is closer to zero than B2's (-0.294), i.e. replay measurably helps without eliminating forgetting, which is the expected shape of the H5 comparison.

---

## ADR-0012 — B3's replay loss uses a dedicated config field, not `ConsolidationConfig.replay_weight`

**Status:** Accepted

**Decision:** `apc.evaluation.baselines.BaselineConfig.replay_weight` is a new, baseline-only field, separate from `apc.consolidation.distill.ConsolidationConfig.replay_weight`.

**Reason:** `ConsolidationConfig.replay_weight` scales a consolidation-distillation loss: a candidate primitive learning to reproduce a frozen teacher's *delta* on replay hidden states. B3 has no distillation step at all -- it adds a plain cross-entropy replay loss directly onto the raw task loss its newly grown transforms are trained with. These are different losses over different targets; reusing the same config field for both would be a silent semantic overload of one number meaning two different things depending on which baseline reads it.

**Consequence:** A config file that wants to tune B3's replay strength sets a new top-level `baseline_replay_weight` key (see `apc.evaluation.baselines.baseline_config_from_dict`), separate from the `consolidation.replay_weight` key that affects B4 (and B1's one seed cycle, per ADR-0010).
