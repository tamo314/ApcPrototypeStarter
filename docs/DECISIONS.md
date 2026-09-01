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

---

## ADR-0013 — ADR-0006's generalization failure is not a model-scale artifact

**Status:** Accepted

**Decision:** No further Phase A time is spent re-testing whether a bigger dense core (within or above `docs/design-docs/ARCHITECTURE.md` section 3's "recommended first scale") fixes held-out generalization. Model scale is ruled out as the explanation; investigation should move to other candidate causes (training data quantity/coverage, training duration/"grokking"-style late-phase generalization, or task representation) before trying yet another scale point.

**Reason:** ADR-0006 measured no generalization at every scale tried during Task 012 (d_model 32-192, up to ~1.8M parameters) -- below `ARCHITECTURE.md` section 3's own "recommended first scale" of 10M-60M parameters and `docs/HARDWARE_ENVIRONMENT.md`'s "10M-100M comfortable" range, leaving open whether the negative result was simply a too-small-model artifact (`docs/exec-plans/completed/PHASE_A_RESULT.md` recommendation 1). This was retested directly: three new configs (`configs/phase_a_scale_check_{low,mid,high}.yaml`, 384/512/640 hidden size x 8 layers, 14.2M/25.3M/39.4M parameters -- spanning the full recommended range) were each trained from scratch (seed 0, 256 examples, 1500 steps, first real GPU run in this repository's history via the WSL Python 3.12 + CUDA environment built for this investigation) and evaluated with `scripts/composition_benchmark.py`:

| Config | Parameters | Train exact match | Held-out `known` (test) exact match | `novel_composition` exact match |
|---|---:|---:|---:|---:|
| Original smoke (192d, 4L) | 1.79M | 1.0 | 0.0156 | 0.0156 |
| `phase_a_scale_check_low` (384d, 8L) | 14.2M | 1.0 | 0.0078 | 0.0313 |
| `phase_a_scale_check_mid` (512d, 8L) | 25.3M | 1.0 | 0.0078 | 0.0234 |
| `phase_a_scale_check_high` (640d, 8L) | 39.4M | 1.0 | 0.0156 | 0.0156 |

Across a 22x parameter range spanning the entire recommended scale, held-out exact match stays flat at chance (0.008-0.016) while training exact match is a perfect 1.0 in every case -- the model always has more than enough capacity to memorize its (small) training set, and more capacity does not measurably help it do anything else with unseen token content.

**Consequence:** `docs/exec-plans/completed/PHASE_A_RESULT.md` recommendation 1 ("retest at the project's own comfortable scale before concluding the negative result is scale-independent") is now satisfied for the *model-parameter-count* axis specifically; the open caveat about scale no longer applies to that axis. The more likely remaining explanations are training-data quantity/coverage (256-512 examples is a vanishingly small fraction of the possible token sequences at this vocabulary/length range) and training duration (loss was already fully converged, ~1e-4 to 1e-5, after 1500 steps at every scale tried -- these runs cannot distinguish "no generalizing solution exists" from "a generalizing solution exists but was never reached because memorization is the lower-loss optimum found first," the latter being the standard "grokking" phenomenon in the literature, which typically requires an order of magnitude or more additional training steps with weight decay to resolve). Neither has been tested yet.

---

## ADR-0014 — ADR-0006's generalization failure is not a training-data-quantity artifact (at least up to 4096 examples)

**Status:** Accepted

**Decision:** No further Phase A time is spent scaling `data.num_examples` alone, within a single-batch (whole-dataset-every-gradient-step) training loop, as a way to fix held-out generalization. Training-data quantity/coverage is ruled out as the explanation over the range actually tested (256-4096 examples, a 16x span); investigation should move to training duration ("grokking") or task representation before trying yet more raw examples under this training regime.

**Reason:** ADR-0013 left training-data quantity/coverage as a leading untested candidate cause of ADR-0006's finding. This was tested directly: `configs/phase_a_data_scale_{1024,4096}.yaml` reran the same held-out-generalization measurement at 4x and 16x the original 256-example count (seed 0, 1500 steps, `scripts/composition_benchmark.py` with `known_split=test`, 128 known + 128 novel-composition examples evaluated):

| `num_examples` | Model | Parameters | Train exact match | Held-out `known` (test) exact match | `novel_composition` exact match |
|---:|---|---:|---:|---:|---:|
| 256 | `phase_a_scale_check_low` (384d, 8L) | 14.2M | 1.0 | 0.0078 | 0.0313 |
| 1024 | same (384d, 8L) | 14.2M | 1.0 | 0.0156 | 0.0313 |
| 4096 | original smoke dims (192d, 4L) | 1.79M | 1.0 | 0.0156 | 0.0469 |

(The 4096 row uses the smaller model because this training loop collates the *entire* dataset into one batch and reruns it every step, so activation memory scales directly with `num_examples`; 4096 examples at the 14.2M model's width overflowed the 16GB card -- observed as a 67x per-step slowdown from VRAM-overflow thrashing, not real compute growth, killed after ~4h stuck at step 900/1500 -- while 4096 at 1.79M params fit comfortably (4.9GB peak). ADR-0013 already established held-out exact match is flat across this exact parameter range at fixed data, so the model-size change does not confound this row.)

Across a 16x increase in training examples, held-out exact match stays flat at chance (0.008-0.016, same range as ADR-0013) and training exact match remains a perfect 1.0 throughout -- more data did not produce any measurable movement toward generalization, at any point along the range tested.

**Consequence:** Training-data quantity/coverage, on its own, is ruled out as the explanation for ADR-0006's finding, *for the 256-4096 example range*. Pushing further (e.g. 16384+ examples) under this exact training loop is not straightforward: the whole-dataset-every-step design means VRAM scales with `num_examples` regardless of model size, and doing so would require either implementing minibatching/gradient accumulation in `apc.core.train.run_smoke_training` (a real code change, not just a new config) or shrinking the model further, which risks losing enough capacity to fit even the training set. The two remaining untested candidate causes from ADR-0013 narrow to one: training duration / "grokking"-style late-phase generalization (loss is already fully converged, ~1e-4, after only 1500 steps at every data scale tried here too -- these runs still cannot distinguish "no generalizing solution exists" from "one exists but was never reached").

---

## ADR-0015 — `encode_split`'s `task_state` is a parameter-free view, not a learned projection

**Status:** Accepted

**Decision:** `apc.core.model.DecoderOnlyTransformer.encode_split` (Task A1-005) derives `task_state` from `content_state` via `task_head`, a `LayerNorm(elementwise_affine=False)` -- zero learnable parameters -- rather than a learned `nn.Linear` projection.

**Reason:** A first implementation used a learned `nn.Linear(d_model, d_model)` for `task_head`. It broke two pre-existing, passing tests: `tests/test_core_checkpoint.py::test_save_and_load_checkpoint_restores_weights_exactly` (its zero-initialized bias is identical across two differently-seeded models, since nothing in the current pipeline calls `encode_split`/`use_split_state=True` during training, so the layer never receives gradient and stays at its untouched init) and `tests/test_sequential_benchmark.py::test_two_events_each_complete_a_learn_consolidate_release_cycle` (merely *constructing* a new randomly-initialized submodule consumes extra draws from this codebase's one shared global `torch` RNG stream, per `apc.utils.seed.set_seed`; every random draw made anywhere else in the same process *after* model construction -- primitive bank/router initialization, data sampling, plastic-workspace training -- shifts by exactly that amount, silently changing an already-seeded integration test's step-by-step trajectory even though the new layer's output is never read). This is general: any new `nn.Parameter`-bearing submodule added to `DecoderOnlyTransformer.__init__`, used or not, perturbs every later seeded random draw in the process under this codebase's RNG model. A parameter-free transform reads `content_state` without registering a `nn.Parameter`, so it consumes zero RNG draws and leaves both tests (and everything downstream of model construction) untouched.

**Consequence:** `task_state` is currently a deterministic, non-learned function of `content_state` (mean/variance-normalized per position) -- distinct-valued (see `tests/test_core_model.py::test_encode_split_task_state_is_independently_probeable`) but not yet a separate *learned* representation. `PHASE_A1_ARCHITECTURE_DELTA.md` section 4 explicitly leaves the implementation flexible for this reason; the low-information-content-state generalization constraint it states is a gate for a later milestone that actually routes/detects novelty on `task_state`, not this task's acceptance criterion. Whichever future task first makes `task_state` a genuinely learned, separate representation (e.g. a learned router in A1-015) will reintroduce a new `nn.Parameter`-bearing submodule to the model; ADR-0016's setup-phase re-seeding means that, done inside `_SequentialBenchmarkRunner`/`_BaseRunner`'s existing construction, it will *not* require retuning seed-sensitive event-loop tests the way this task's first attempt did -- it only would if the new parameters are trained inside the setup phase itself (before the re-seed point), which would need the re-seed moved after that training.

---

## ADR-0016 — Benchmark/baseline runners re-anchor the global RNG stream after setup-phase construction

**Status:** Accepted

**Decision:** `apc.evaluation.sequential_benchmark._SequentialBenchmarkRunner.__init__` and `apc.evaluation.baselines._BaseRunner.__init__` (plus `B1Runner.__init__`, which constructs additional state after calling `super().__init__`) call `apc.utils.seed.set_seed(seed)` a second time, immediately after all setup-phase construction (model, primitive bank, router's null key, plastic workspace, allocator, controller, novelty estimator) finishes and before returning. Any `__init__` override that constructs further RNG-consuming state after calling its parent's `__init__` must add its own trailing re-seed for the same reason (`B1Runner` does; `_GrowRunner`/`B2Runner`/`B3Runner` do not need one because `Allocator.__init__` and their own added state do not consume the global RNG at construction).

**Reason:** ADR-0015's first (rejected) implementation surfaced this while debugging why an unused, never-forward-called `nn.Linear` broke `tests/test_sequential_benchmark.py::test_two_events_each_complete_a_learn_consolidate_release_cycle`, an integration test that asserts a specific emergent outcome ("at least 2 learn-consolidate-release cycles complete within a bounded step budget") purely by pinning a seed and relying on bit-exact reproducibility of the entire multi-stage training trajectory that follows. Before this ADR, `__init__` called `set_seed(seed)` exactly once at the top, then constructed several *conceptually independent* pieces (model, then bank, then router, then workspace, ...) that nonetheless all drew from the same shared global `torch` RNG stream in sequence. This meant the exact random-number trajectory the event loop's training/allocation/consolidation logic saw was accidentally coupled to *how many parameters the model, or any other piece built during setup, happened to have* -- a completely unrelated implementation detail. `apc.environments.generator.TaskGenerator` already avoids this failure mode correctly, by deriving its own independent `random.Random` instances per seed/label (`_derive_seed`, see `generator.py`) rather than sharing the global `torch` stream; `apc.consolidation.distill.consolidate` also already re-seeds locally and deliberately right before constructing its candidate primitive. The runners' `__init__` methods were the one place in the active Phase A.1 path that hadn't adopted either pattern.

**Consequence:** Re-seeding here changed the exact numeric trajectory of every existing runner-driven test relative to before this ADR (verified: all pre-existing tests, including the cycle-count test, still pass after the change -- see `tests/test_sequential_benchmark.py::test_runner_init_reanchors_rng_regardless_of_setup_phase_model_size` and the matching test in `tests/test_baselines.py`, both added as regression coverage for this property specifically). Going forward, adding, removing, or resizing anything constructed during a runner's setup phase (model layers, bank/router/workspace scaffolding) cannot silently alter the event loop's random trajectory, so long as it happens *before* the trailing re-seed and does not itself need to be trained during setup. `AGENTS.md`'s "Use deterministic seeds wherever practical" principle is elaborated with this pattern so future code (new runners, or new setup-phase construction in existing ones) follows it by default rather than rediscovering this failure mode.

---

## ADR-0017 — Four of the eight known operations are not learnable as a function of the presented input alone

**Status:** Accepted

**Decision:** `apc.evaluation.stable_core_generalization` (Task A1-006, the Stable Core systematic-generalization gate) trains and evaluates only `apc.environments.operations.DETERMINISTIC_OPERATION_NAMES` (`COPY`, `NEGATE`, `COMPARE`, `ACCUMULATE`) rather than the full `KNOWN_OPERATION_NAMES` (which also includes `SELECT`, `COUNT`, `SHIFT`, `BIND`). `KNOWN_OPERATION_NAMES` itself is unchanged and every other Phase A/A.1 consumer (composition benchmark, sequential benchmark, baselines) keeps using the full eight-operation curriculum.

**Reason:** Discovered while designing Task A1-006's training/eval loop. `apc.environments.generator.TaskGenerator._generate` samples each operation's parameters via `Operation.sample_params(rng, sequence, vocab_size)` (`SelectOp`'s `indices`, `CountOp`'s `target`, `ShiftOp`'s `amount`, `BindOp`'s `query_key`) from the *same* per-example `rng` stream that also drew the input content -- but those parameters are stored only in the latent `Program`/`OperationGraph` (oracle-only metadata per the generator's own docstring) and are never included in `Example.input_tokens`. `apc.core.data.encode_example` presents the model with `[BOS] input... [SEP] target... [EOS]` and nothing else. For `SELECT`/`COUNT`/`SHIFT`/`BIND`, the target is therefore a function of `(input, hidden_parameter)`, not of `input` alone: e.g. `COUNT`'s target token depends on a `target` value drawn uniformly from the whole vocabulary, independent of the input content, so two examples with identical presented input but different sampled `target` correctly have different presented targets. No model -- regardless of capacity, training regime, online generation, or symbol permutation -- can recover a hidden, input-independent parameter from the input alone; unseen-content exact match for these four operations is upper-bounded by chance/guessing, not by representation quality. `COPY`, `NEGATE`, `COMPARE`, and `ACCUMULATE` have no such parameter (`sample_params` always returns `{}`, see `tests/test_operations.py::test_deterministic_operations_sample_params_is_always_empty`): their target is a pure, deterministic function of the presented input, so they are the only members of `KNOWN_OPERATION_NAMES` for which "does the Stable Core generalize this known operation to unseen content" is even a well-posed question.

**Consequence:** A1-006's gate is scoped to a 4-operation curriculum, not all 8; a mixed-operation Stable Core gate as originally implied by "known operations" would be structurally incapable of reaching the >=0.95 acceptance threshold regardless of architecture or training quality, making any failure uninterpretable (task ill-posedness vs. representation/training failure -- exactly the ambiguity `docs/AGENTS_PHASE_A1_ADDENDUM.md`'s "investigate representation/training objective only" instruction for a failed STOP GATE presumes is *not* present). This is an environment-design gap, not a Task A1-006 defect: `apc.evaluation.composition_benchmark`/`novel_operation_benchmark`/`sequential_benchmark` still mix parameterized and deterministic known operations, so their own reported `K`/`C` exact-match numbers (Task 006/009/012, and ADR-0006/0009/0013/0014's negative-generalization findings) are also confounded by this same non-identifiability for any example whose sampled operation is `SELECT`/`COUNT`/`SHIFT`/`BIND` -- those prior findings remain valid as measured (near-chance generalization was never claimed to be *entirely* explained by task well-posedness), but their true achievable ceiling was never 1.0 to begin with. A future task should decide, for the parameterized operations, either (a) encode the sampled parameter into the presented input (e.g. a query token before `[SEP]`), turning them into well-posed input-to-output tasks, or (b) permanently exclude them from any generalization-ceiling claim and keep them only for capacity/routing-oriented benchmarks (composition, novelty, sequential) where a hidden parameter does not undermine the property being measured (e.g. whether capacity expands). Neither is in scope for A1-006 itself.

---

## ADR-0018 — Symbol permutation must be shared per batch/episode, not drawn fresh per example

**Status:** Accepted

**Decision:** `apc.environments.generator.TaskGenerator._generate` now draws exactly one `SymbolPermutation` per `generate`/`generate_online` call (shared by every example returned from that call) when `permute_symbols=True`, instead of one independent permutation per example. This corrects the implementation to match `docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md` section 3's own stated design ("per-batch or per-episode symbol permutation"), which Task A1-004's original implementation did not actually follow (it derived a fresh permutation keyed by each example's loop index).

**Reason:** Discovered empirically while building Task A1-006's gate. With `apc.evaluation.stable_core_generalization.DETERMINISTIC_OPERATION_NAMES` (`COPY`, `NEGATE`, `COMPARE`, `ACCUMULATE`; ADR-0017), a single-operation, zero-ambiguity, 6000-step training run per operation gave: `COPY` unseen exact match 1.0 (loss ~2.8e-5), but `NEGATE` 0.0 (loss 1.29), `COMPARE` 0.0 (loss 1.11), `ACCUMULATE` 0.0 (loss 1.80) -- all three stuck far above zero loss with *zero* measurable exact match despite no operation-identity ambiguity at all (see ADR-0017's issue; this ablation used one operation at a time). The cause is permutation, not capacity or training budget: `COPY`'s target at position `i` is exactly the input token at position `i`, so it is *value-blind* -- correct regardless of how tokens are relabeled, since relabeling commutes with the identity function. `NEGATE` (`vocab_size - 1 - x`), `COMPARE` (`a < b` / `a == b` / `a > b`), and `ACCUMULATE` (running sum mod `vocab_size`) all depend on the *canonical numeric value* of a token, not merely its identity/position. Under a permutation resampled independently for every example, a given presented token id encodes a different canonical value in every example, with no side channel revealing which -- so the presented-space relationship between input and target for these operations is effectively a fresh, unrelated random relabeling on every single training example, carrying no learnable cross-example regularity. Only operations that commute with an arbitrary bijective relabeling of the vocabulary (`f(π(x)) = π(f(x))` for every permutation `π`) -- i.e. purely positional/structural operations like `COPY`, `SHIFT`, `SELECT`, `REVERSE` -- survive independent per-example permutation; anything depending on order or arithmetic over token values cannot. Sharing one permutation across an entire batch (all `n` examples from one `generate_online` call, i.e. one training step) fixes this: the model can learn a permutation-*equivariant* circuit (e.g. genuine numeric comparison/arithmetic over presented ids) that generalizes across the many different permutations seen over the course of training, while a fixed token id still never survives an entire training run unchanged (preserving A1-004's original anti-memorization intent -- see the updated `apc.environments.generator` module docstring). Verified safe: grepped the full repository for `permute_symbols` before making this change -- only the generator itself, its own tests, and the new Task A1-006 module/config reference it, so no other Phase A/A.1 benchmark (composition, novel-operation, sequential, baselines) currently enables it, and every pre-existing permutation test in `tests/test_generator.py` still passes unmodified after the fix (none of them asserted per-example independence as a property).

**Consequence:** Task A1-004's own acceptance criteria (bijection validity, decode-invariance, per-`(seed, step, split)` determinism, semantic invariance of interpreter truth) are unaffected -- all still hold with a shared-per-call permutation, and the affected commit's tests all pass unchanged. `Example.symbol_permutation` is now the *same* value across every example returned by one `generate`/`generate_online` call rather than a distinct one per example; nothing observed reading that field individually per example changes shape or type. Two new regression tests (`test_permute_symbols_is_shared_across_every_example_in_one_call`, `test_permute_symbols_changes_across_different_generate_calls`) lock in "shared within a call, different across calls" going forward. This was a pre-A1-006 latent defect in already-"accepted" A1-004 code, not a new deviation introduced by A1-006; historical Phase A run artifacts are unaffected since none of them used `permute_symbols=True`.

---

## ADR-0019 — Symbol permutation makes value/order-dependent operations unrecoverable on held-out content, even shared per batch

**Status:** Accepted

**Decision:** `apc.evaluation.stable_core_generalization.StableCoreGateConfig.permute_symbols` defaults to `False`, not `True`. The gate's primary variant (`configs/phase_a1_stable_core_gate.yaml`) tests all four `DETERMINISTIC_OPERATION_NAMES` with permutation off; a secondary variant (`configs/phase_a1_stable_core_gate_permuted_copy_only.yaml`) tests `COPY` only with `permute_symbols=True`, satisfying `docs/AGENTS_PHASE_A1_ADDENDUM.md`'s "must use ... token/symbol permutation" instruction on the one operation for which it is actually meaningful. Both variants are run and reported (a deliberate choice, confirmed with the user given it revises the recommendation behind ADR-0018).

**Reason:** After ADR-0018's per-batch permutation fix, a single-operation (`NEGATE`, zero cross-operation ambiguity), zero-hidden-parameter, 30000-step run under `permute_symbols=True` still reached only 0.0 unseen exact match, with loss still steadily falling (2.02 -> 0.77, no plateau) -- i.e. *not* a training-budget problem, since the identical configuration with `permute_symbols=False` reaches 1.0 exact match (loss ~6e-5) in 6000 steps. The reason is structural, not empirical: at evaluation time, a held-out example's specific relabeling (which `SymbolPermutation` was used) is never part of the presented input (by design -- it is oracle-only metadata, see `apc.environments.permutation`'s module docstring) and cannot be inferred from that one example's own content for any token value that does not repeat within it (a decoder-only Transformer processes one example at a time, with no cross-example information at inference). The continuously falling *loss* despite near-zero *exact match* is consistent with the model learning only what generalizes across every possible relabeling -- primarily copy-on-repeat / induction-style behavior for token values that happen to reappear within the same example -- while any "first occurrence" position within a held-out example remains fundamentally unresolvable, and exact match requires every position correct. `COPY` is the exception because its output equals its input position-for-position: a *value-blind* operation commutes with any relabeling (`f(π(x)) = π(f(x))` when `f` is the identity), so it needs no information about which permutation is active. `NEGATE`, `COMPARE`, and `ACCUMULATE` all depend on the actual numeric value (arithmetic complement, order comparison, running sum) and do not commute with an arbitrary permutation, so they cannot be solved by a model that never observes which permutation was used, independent of model capacity or training duration.

**Consequence:** No known-operation gate variant that both (a) uses per-example-unknown `permute_symbols=True` and (b) includes a value/order-dependent operation can pass the >=0.95 threshold, ever -- this is not a candidate for a future "just train longer / bigger" fix; fixing it for real would require an architecture change out of scope for A1-006 (e.g. giving the model a few in-context demonstration pairs under the same permutation before asking it to generalize). The gate therefore reports two honestly-scoped variants instead of one conflated number: the primary, more informative variant (`permute_symbols=False`, all four operations) is the one `docs/results/PHASE_A1_RESULT.md`-style reporting should treat as the H1 verdict; the secondary variant (`permute_symbols=True`, `COPY` only) demonstrates the addendum's symbol-permutation mandate is satisfied where it is actually meaningful, and should not be read as a broader claim about generalization under permutation. Every result reported under either variant must state which variant it is (this ADR's terms "primary"/"secondary" are the intended shorthand for run artifacts and future references).

---

## ADR-0020 — Operations sharing an output-length signature are mutually non-identifiable when pooled in one model

**Status:** Accepted

**Decision:** `apc.evaluation.stable_core_generalization.run_stable_core_gate_grid` trains and evaluates one single-operation `TaskGenerator`/model per operation (never a pool of multiple operations in one `TaskGenerator`/model), then aggregates results across the resulting operation x seed grid. `run_stable_core_gate`/`run_stable_core_gate_multi_seed` remain general-purpose (a caller may still pass a multi-operation `operation_names` pool if a future task genuinely wants that), but the gate's own entry point (`run_stable_core_gate_grid`, and `scripts/stable_core_generalization_gate.py`) always uses the per-operation grid form.

**Reason:** Measured directly while scoping A1-006 (before discovering ADR-0019's permutation issue): even with `permute_symbols=False` and zero hidden parameters, training one model on all four `DETERMINISTIC_OPERATION_NAMES` pooled together plateaus at ~0.25-0.35 unseen exact match after 10000 steps (loss 0.16-0.19, essentially converged, not still improving) -- far below the 0.95 gate and *not* explained by ADR-0017's or ADR-0019's findings (no hidden parameters here, no permutation here). The cause: `apc.environments.generator.TaskGenerator._generate` samples which operation applies to a given example *uniformly at random*, independent of input content, and this choice is never part of the presented input (only latent `Program`/`OperationGraph` metadata carries it). `COPY`, `NEGATE`, and `ACCUMULATE` all preserve input length; only `COMPARE` differs (`input_length - 1`). A model can use output-length as a structural cue to recognize "this example needs `COMPARE`" (and indeed learns it near-perfectly, contributing ~1/4 of the pooled exact match), but has no analogous cue to distinguish `COPY` from `NEGATE` from `ACCUMULATE` for any single example -- the three are pairwise indistinguishable from input content alone, so the model can do no better than an uninformed guess among them, capping the pooled result near "the identifiable fraction of the pool, solved well" plus "the rest, near chance." This is structurally the same category of problem as ADR-0017 (a value the model needs is never revealed to it) one level up: there, it was a hidden operation *parameter*; here, it is the hidden *choice of operation itself*.

**Consequence:** Any future Phase A.1 K-gate (or gate-adjacent benchmark) that mixes multiple known operations into one shared model/`TaskGenerator` pool risks measuring this identifiability ceiling rather than genuine representation/training quality, unless every pooled operation is pairwise distinguishable by some structural cue available in the presented input (rare, and not something to rely on by construction). Running one operation at a time and aggregating is the general-purpose fix and is what A1-006 adopts; this does not change `apc.evaluation.composition_benchmark`/`novel_operation_benchmark`/`sequential_benchmark`, which already report per-label (`K`/`C`/`N`/`R`) results rather than one pooled number, so they are not directly affected by this specific failure mode, though ADR-0017's parameter-identifiability caveat still applies to their `K`/`C` pools whenever a sampled operation happens to be `SELECT`/`COUNT`/`SHIFT`/`BIND`.

---

## ADR-0021 — H1b passes: an explicit task segment resolves ADR-0017/ADR-0020's non-identifiability, one shared model, all eight known operations

**Status:** Accepted

**Decision:** `apc.evaluation.shared_core_generalization` (Task A1-C004, the Shared-Core systematic-generalization gate, STOP GATE) is recorded as **PASS**. `docs/exec-plans/active/PHASE_A1.md`'s H1b ("shared-core conditional systematic generalization") is confirmed: `runs/phase_a1_shared_core_gate/report.json`/`summary.json` (config `configs/phase_a1_shared_core_gate.yaml`, 5 seeds `0-4`, RTX 5060 Ti, git commit `e693c34`) show a single `apc.core.model.DecoderOnlyTransformer` (192d, 4 layer, ~1.80M params), trained on `apc.environments.generator.build_mixed_operation_generator`'s online mixed stream over all eight `KNOWN_OPERATION_NAMES` with the model-visible task segment included (`include_task_spec=True`, Task A1-C003), reaches:

- overall mean unseen exact match **0.9946** (threshold 0.95, stdev 0.0030, min-over-seeds 0.9922, `meets_seed_policy=true`);
- every operation's mean **>=0.984** (threshold 0.90) -- `COPY` 0.9996, `SELECT` 0.9946, `COMPARE` 0.9989, `COUNT` 0.9863, `SHIFT` 0.9842 (lowest), `BIND` 0.9981, `NEGATE` 0.9977, `ACCUMULATE` 0.9982;
- zero `(seed, operation)` pairs below the stricter 0.85 per-run floor from `docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md` section 2 (`low_outlier_seed_operations: []` for every seed).

The paired negative control (identical config/architecture/`SharedCoreTokens` vocabulary, `include_task_spec=False`) reaches only **0.139** mean overall exact match (stdev 0.008) -- a **0.856** gap, far exceeding the predeclared 0.20 `MATERIAL_UNDERPERFORMANCE_MARGIN`. All three `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` A1-C004 acceptance criteria are satisfied; A1-C004's own `run_shared_core_gate_h1b(...).passed` is `True`.

**Reason:** This is the intended resolution of ADR-0017 (`SELECT`/`COUNT`/`SHIFT`/`BIND`'s hidden per-instance parameter) and ADR-0020 (mutual non-identifiability of which operation applies, even among parameter-free operations) -- both root-caused to "a value the model needs is never revealed to it." Tasks A1-C001 (`apc.environments.task_spec.TaskSpec`) and A1-C003 (`apc.core.tokens.SharedCoreTokens`/`encode_task_spec`, rendering `[TASK_START] op arg... [TASK_END]` between `BOS` and the content input) fixed exactly that: operation identity and every previously-hidden argument are now part of the presented input. The negative control isolates this as the causal variable, not an incidental effect of "one shared model + mixed online data" alone: it shares every other field (seed, architecture, vocabulary, training budget) and its training curves (`runs/phase_a1_shared_core_gate/negative_control/seed_*/metrics.jsonl`) show a stable plateau from step ~2000 onward (loss oscillating ~0.42-0.53, exact match ~0.09-0.18 for the whole 30000-step budget, not still improving) -- consistent with ADR-0020's mechanism, not a training-budget shortfall. Per-operation negative-control means show the expected structure: `COPY` (value-blind, needs no task information to execute correctly) still reaches 0.77 -- close to its ADR-0020 single-mechanism finding -- while every operation whose correct output depends on which operation was requested and/or a hidden argument (`SELECT` 0.0004, `COMPARE` 0.028, `COUNT` 0.0024, `SHIFT` 0.097, `BIND` 0.016, `NEGATE` 0.015) collapses far below chance-adjacent baselines; `ACCUMULATE` (0.175) is the one mild outlier, plausibly because a running-sum's first output position is position-invariant regardless of which operation is guessed. This is consistent with, not contradicting, ADR-0020's ~0.25-0.35 pooled ceiling measured on a 4-operation, all-parameter-free pool: an 8-operation pool that also includes four ADR-0017 hidden-parameter operations is strictly harder to guess correctly without the task segment, so a lower negative-control ceiling here is expected.

**Consequence:** Per `docs/exec-plans/active/PHASE_A1_CORRECTION.md` section 4 ("Routing-readiness criterion"), A1-007 is unblocked *from H1b's perspective specifically* -- H1b passing is one of that section's four conditions, not all of them. `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` still requires A1-C005 (probe whether `z_task` actually carries usable task information -- H1b passing says the *model* can condition on the task segment somewhere in its computation, not that `encode_split`'s specific `z_task`/`h_content` split routes it there usefully), A1-C006 (parameterized `PrimitiveCall` abstraction), and A1-C007 (oracle `PrimitiveCall` routing adapter) before A1-007 itself may begin; A1-C008's audit is the task that should record the combined H1a+H1b+probe verdict formally. `docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`'s historical-interpretation instruction is satisfied as written: A1-006 remains H1a (per-operation systematic generalization, unmodified), and this ADR records H1b (shared-core conditional systematic generalization) as now also passed, not a replacement or reinterpretation of A1-006's own result. No STOP-discipline investigation is triggered (`docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`'s "If the shared-core gate fails" branch does not apply); the only open, non-blocking observation is `ACCUMULATE`'s comparatively higher negative-control score (0.175 vs. <0.10 for most other hidden-dependency operations), which does not affect this gate's own pass/fail and is left for a future task to investigate only if it becomes load-bearing for a later conclusion.
