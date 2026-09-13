# Phase D / Task D-001 — SORT-Only Local Repair Pilot Preregistration

**Document ID:** `DOC-PHASE-D-D001-SORT-REPAIR-PREREGISTRATION`
**Date:** 2026-09-13
**Status:** Design/preregistration complete (Task D-001). **`training_execution: AUTHORIZED`**, as
seed-amended by Task D-005
([ADR-0172](../DECISIONS_PHASE_D.md#adr-0172-d-005-phase-d-cohort-seed-amendment--complete-static-registry-audit-and-replacement-authorization)),
strictly scoped to executing exactly the recipe fixed in section 4 below, on exactly the seed-`40-44`
cohort, evaluated only against the comparison conditions and panels registered in sections 5 and
7-8. No sweep, no additional seed, no other primitive, and no candidate/bundle promotion is
authorized. This document fixes a single repair recipe, its comparison conditions, its acceptance
criteria, and its resource budget before any execution. No optimizer step, no model initialization,
and no candidate construction was performed to produce this document or by the D-003 approval
review.

**Execution attempt STOP-GATE-FAILed (Task D-004,
[ADR-0171](../DECISIONS_PHASE_D.md#adr-0171-d-004-sort-only-repair-confirmation-execution-stop-gate-fails-on-the-data-boundary-prerequisite-seeds-30-34-collide-with-the-sealed-v2-partition)):**
before building the seed-`30-34` cohort this recipe depends on, D-004 found that those five seeds
collide with `src/apc/evaluation/relation_split_protocol.py`'s `NEW_SEALED_V2_SEEDS` — a sealed
model-seed partition Phase B task B-C005R3-002 reserved for the not-yet-executed R3-011/R3-012
sealed-gate pathway. No cohort was built, no repair step ran, and no evaluation was performed; every
number in this document (recipe, comparison conditions, sample sizes, acceptance criteria, budget)
is unchanged. D-005 has now authorized the statically verified seed-`40-44` cohort; seed-`30-34`
remains forbidden to this charter.
**Depends on:** `docs/design-docs/PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` (input-domain derivation, ownership boundary, dependency hashes), `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md` (target/regression/causal-control panels), `docs/design-docs/PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` (model cohort this pilot runs against).

---

## 1. The single repair hypothesis (fixed, not a sweep)

> Keeping SORT's existing architecture unchanged, standalone training of SORT alone — over the
> valid input-length set derived from the composition execution contract
> (`{3,4,5,6,7,8,9,10}`, `PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` section 3) — recovers
> short-sequence SORT execution and the 7 target compositions (`PHASE_D_D001_TARGET_PANEL_MANIFEST.md`
> section 2), without degrading SORT's existing length-adequate behavior or any other primitive's
> behavior.

This is the **only** hypothesis this task registers. No alternative architecture, no COUNT repair,
no other primitive's repair, and no joint multi-primitive repair is proposed or authorized here.

## 2. Update scope and ownership (shared vs. non-shared, confirmed from source)

| Component | Shared or SORT-owned | Frozen during repair | Source |
|---|---|:-:|---|
| `SharedContentEncoder` (Core) | Shared across all 8 canonical + 2 Branch-B primitives | **Yes** | `core.model.eval()`, `param.requires_grad_(False)` (`learned_routing_benchmark.py:566-568`) |
| SORT's `CrossPositionPrimitive` instance | **SORT-owned.** Bank stores primitives in an `nn.ModuleDict` keyed by `str(primitive_id)`; SORT's slice is `_primitives.<sort_id>.*`, structurally disjoint from every other primitive's slice | **No — this is the only unfrozen module** | `src/apc/utils/model_bundle.py: primitive_state_dict` (lines 294-306); `src/apc/primitives/bank.py` `ModuleDict` primitive storage |
| SELECT / COUNT / BIND / SHIFT / COPY / REVERSE / NEGATE primitives | Each independently owned, disjoint `ModuleDict` slice | **Yes** | Same isolation mechanism as above; per-primitive `PrimitiveBank.freeze(primitive_id)` (`bank.py:235`) |
| Router | Shared | **Yes** | Not touched by this recipe: SORT is invoked directly via `op_to_id["SORT"]` / `CompositionRecipe`, not through router inference |
| Argument scorer | Shared | **Yes** | SORT has no argument (`required_argument_names = frozenset()`), so the argument scorer is never consulted for SORT and is untouched by construction |
| Vocabulary / token schema | Shared | **Yes** | Unmodified; `vocabulary.schema_hash` must be identical before/after |

**Mechanical enforcement:** `PrimitiveBank.freeze(primitive_id)` / `unfreeze(primitive_id)`
(`bank.py:235,238`) already exists and gives per-primitive freeze granularity. The repair recipe
calls `bank.freeze_all()` then `bank.unfreeze(op_to_id["SORT"])`, and asserts
`core.model.training is False` and every non-SORT primitive's `requires_grad` is `False` before
the first optimizer step.

**Verification (not diagnosis):** before and after the repair run, recompute
`canonical_state_hash` (`model_bundle.py:202-217`) for the Core and for every non-SORT primitive's
sliced `state_dict`. All of these hashes must be bitwise identical before/after. Any mismatch is a
contract violation, not a repair outcome, and halts the pilot.

## 3. Architecture (unchanged)

SORT is built as `CrossPositionPrimitive` with
`CrossPositionPrimitiveConfig(operation="SORT", d_operator=32, n_head=4, d_operator_ff=64,
vocab_size=10, max_sequence_length=32, arg_dim=16)`
(`src/apc/evaluation/unified_oracle_causal_benchmark.py:458-472`). This recipe does not change
`d_operator`, `n_head`, `d_operator_ff`, or any other config field. No new layer, residual branch,
position bias, or argument encoder is added. The trainable parameter count is whatever
`CrossPositionPrimitive`'s own module tree yields for this config (documented order of magnitude:
~18k-21k parameters, `src/apc/primitives/primitive.py` docstring); the exact count is read from
the live module's `num_parameters()` at execution time, not fabricated here.

## 4. Training recipe (single, fixed — no simultaneous architecture/optimizer/LR/loss/sampling sweep)

All hyperparameters below are the **existing, unmodified** defaults from
`_train_single_primitive` (`src/apc/evaluation/unified_oracle_causal_benchmark.py:493-601`), the
function that already trains SORT (and every other bank primitive) over the frozen Core. This
recipe changes exactly one thing relative to that existing function: the length-sampling
distribution (item 4.2 below). Nothing else is modified.

### 4.1 Optimizer, scheduler, loss, gradient clipping (inherited, fixed)

| Field | Value | Source |
|---|---|---|
| Optimizer | `torch.optim.AdamW(sort_primitive.parameters(), lr=0.0008, weight_decay=0.0001)` | `UnifiedBenchmarkConfig.operator_lr=0.0008`, `operator_weight_decay=0.0001` (`:164-165`); instantiated at `:506-510` |
| Scheduler | `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)` | `:511-513` |
| Loss | `torch.nn.functional.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=IGNORE_INDEX)` — ordinary per-token cross-entropy on the token-output target. No routing label, no teacher-attention loss, no composition-intermediate supervision, no position-specific correction term. | `:590-592`; `IGNORE_INDEX=-100` (`src/apc/core/data.py:36`) |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(sort_primitive.parameters(), 1.0)` | `UnifiedBenchmarkConfig.operator_grad_clip=1.0` (`:166`), applied at `:594-595` |
| Batch size | 32 examples/step | `:539` (`for _ in range(32)`) |
| Step budget | **6,000 steps/model** (registered ceiling, see section 8) | This document (see rationale below) |

**Step-budget rationale:** SORT's own historical training budgets are 3,000 steps
(`bank_train_steps // 2`, the recipe that actually built the reconstructed bundles seed 1-4 used
by NRQ-005/006/007/008; `learned_routing_benchmark.py:630-632`) or 4,000 steps
(`UnifiedBenchmarkConfig.parameter_free_train_steps`, a different recipe series; `:163`). This
pilot registers 6,000 — roughly 2x either historical figure — as a fixed ceiling to give headroom
for the newly added short-length curriculum, without authorizing a sweep across step counts. If
6,000 is insufficient, that is a recipe **failure**, not grounds to silently raise the budget
mid-run (section 9).

### 4.2 Length-sampling rule (single fixed rule, no curriculum, no adaptive oversampling)

At every training step, the 32 example sequence lengths are drawn i.i.d. **uniformly** from the
valid length set derived in `PHASE_D_COMPOSITION_EXECUTION_CONTRACT.md` section 3:

```
L ~ Uniform{3, 4, 5, 6, 7, 8, 9, 10}     (8 equally likely lengths, p = 1/8 each)
```

Each token within a sampled sequence remains i.i.d. `Uniform{0,...,9}` (`vocab_size=10`), matching
the existing non-parameterized generation branch (`:540-543`). This is the **only** sampling rule
registered. It is fixed before any training run and does not change based on interim accuracy,
per-length loss, or which cells previously failed. No curriculum ordering (e.g. short-to-long) and
no adaptive oversampling of failing lengths is authorized.

## 5. Comparison conditions

| Condition | Definition | Update rule |
|---|---|---|
| `FROZEN_PARENT` | The same parent bundle (one of the 5 cohort models), no update of any kind. | Zero optimizer steps. |
| `LOCAL_SORT_REPAIR` | The **only** repair candidate: starts from the same parent bundle, executes exactly the recipe in section 4, updating only SORT's own parameters (section 2). | Exactly the recipe above; no alternative recipe is run for comparison. |
| `SYMBOLIC_REFERENCE` | The deterministic ground-truth function `SortOp.apply` (`tuple(sorted(sequence))`, `src/apc/environments/operations.py:279-303`) evaluated directly on tokens. Used only to certify panel/label correctness and as the causal-control panel's ground truth (section 4 of the panel manifest) — **never** as a training signal, a fallback executor, or an input to the learner. | No learning; a pure function. Never receives or emits routing labels, intermediate ground truth, or any signal into `LOCAL_SORT_REPAIR`'s forward/backward pass. |

`LOCAL_SORT_REPAIR` is deliberately the sole repair candidate — no seed sweep, no architecture
variant, no COUNT/BIND repair, and no switch to a different primitive is part of this comparison.

### 5.1 Explicit-recipe evaluation (not a search benchmark)

Primary evaluation supplies the operation sequence and arguments explicitly as runtime input
(`CompositionRecipe` / `calls_per_example` passed to `execute_composition_recipe`,
`composition.py:194-299`), exactly as NRQ-005/006/007/008 already do. This does **not** evaluate
composition-search/discovery ability. An auxiliary search-based measurement (using the existing,
unmodified `apc.primitives.composition_search` / `apc.evaluation.composition_search_benchmark`
machinery) may optionally be run **before and after** the repair with the **identical search
procedure and budget** in both cases (same beam width / candidate cap / support budget), but a
search-based number is never substituted for the primary explicit-recipe metric, and no new search
algorithm is introduced.

### 5.2 Information-boundary flags (required for every `LOCAL_SORT_REPAIR` run, training and eval)

| Flag | Required value | How it is structurally guaranteed |
|---|:-:|---|
| `task_conditioned_core` | `false` | Core is frozen and never receives task/operation/argument tokens (`_encode_initial_content` / `_encode_intermediate_tokens` in `composition.py` encode content-only sequences; `h_content = f(content)` invariant unchanged) |
| `oracle_coordinate_input` | `false` | SORT has no argument (`required_argument_names = frozenset()`); no `PrimitiveCall` routing/argument oracle metadata is passed to the learner at any point |
| `ground_truth_intermediate_injection` | `false` | Training and the primary success metric use only the normal continuous path (`execute_composition_recipe`'s argmax-then-re-encode loop); NRQ-007's diagnostic-reset (condition B) mechanism is never invoked for `LOCAL_SORT_REPAIR` |
| `symbolic_execution_fallback` | `false` | `SYMBOLIC_REFERENCE` never substitutes for a `LOCAL_SORT_REPAIR` forward pass; it is an evaluator-side artifact only |
| `non_target_parameter_updates` | `0` | Verified by the before/after hash comparison in section 2; a nonzero value is a contract violation, reported as such, and voids the run |

A diagnostic-reset (ground-truth intermediate) evaluation may still be run **separately** for
root-cause bookkeeping (as NRQ-007 already does), but its result is never mixed into the normal
execution success rate, per the composition execution contract section 1/8.

## 6. Staged candidate handling (repair output is not a promoted bundle)

`LOCAL_SORT_REPAIR` writes to a **new namespace** per model
(`runs/phase_d_d010_sort_repair/seed_{40..44}/candidate/`), never overwriting the parent bundle's
files. The candidate is evaluated exactly like a shadow-validation candidate: it must pass every
acceptance criterion in section 8 before it is anything more than a recorded, evaluated candidate.
Per the task's charter-level status (section 10 of this pilot and the Phase D charter),
`bundle_promotion` remains `NOT_AUTHORIZED` regardless of the pilot's outcome — this task does not
authorize adopting the candidate into any production or shared-cache path.

## 7. Sample sizes and interval-estimation procedure (fixed before execution)

Fixed length. `vocab_size = 10`, so short lengths admit **exhaustive** (full-population)
enumeration; longer lengths do not. The two regimes are treated differently and are not mixed:

| Evaluation | Length(s) | Regime | N | Notes |
|---|---|---|---:|---|
| Standalone SORT, target lengths | `L=3` | Exhaustive | 1,000 (= 10^3, all sequences) | Report exact population proportion. No CI (not a sample estimate). |
| Standalone SORT, target lengths | `L=4` | Exhaustive | 10,000 (= 10^4) | Same. |
| Standalone SORT, target lengths | `L=5` | Exhaustive | 100,000 (= 10^5) | Same. Feasible in a single batched forward pass (primitive ~20k params); not run in this task. |
| Standalone SORT, regression lengths | `L in {6,...,10}` | i.i.d. sample | 2,000/length, 3 fixed eval seeds (`301,302,303`) = 6,000/length | Wilson 95% score interval on the aggregate proportion. |
| Target composition panel (7 classes) | composed, `L in {3,4,5}` | i.i.d. sample | 1,000/class/model, 5 fixed eval seeds (`301-305`) | Wilson 95% CI. Full enumeration is infeasible for depth-3 composition (SELECT's index-subset choice multiplies the space) even though the token domain itself is small. |
| Composition regression (11 classes) | composed, `L in {6,...,10}` | i.i.d. sample | 500/class/model, 5 fixed eval seeds (`301-305`) | Wilson 95% CI. |
| Non-SORT canary (8 classes) | composed | i.i.d. sample | 500/class/model, 5 fixed eval seeds (`301-305`) | Wilson 95% CI. |
| Causal control (Correct/Wrong-family/None) | `L in {3,4,5}` and `L in {6,...,10}`, separately | i.i.d. sample | 1,000/arm/length-group/model, seeds `301-305` | Wilson 95% CI on each arm; causal gap computed from the point estimates. |

Eval seeds `301-305` are fixed now, chosen to avoid every registered model-seed namespace (model
seeds `0-4`,`10-14`,`15-19`,`20-24`,`30-34`,`40-44`; data seeds `101-105`,`201-220`) — see
`PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` section 2 for the full existing-seed table.

**On train/test disjointness at small `L`:** at `L=3,4` the full input population is small enough
that i.i.d. training exposure will, in expectation, cover most or all of it; a sequence-level
train/test partition is not feasible at this population size and this task does not require one.
The claim this pilot can support at these lengths is therefore scoped to **"SORT executes its
fully-specified deterministic operation correctly over its declared valid input domain,"** not to
unseen-content generalization beyond that domain — consistent with the claim-scope boundary in
section 9. This is not a weakening of the acceptance floor; it is a statement of what exact-match
against a deterministic, closed-form target function (`sorted()`) actually measures at small
population sizes.

## 8. Acceptance criteria (numeric, fixed before execution)

All criteria are evaluated **per model** and reported for **every** registered run; a mean across
the 5 models never substitutes for a per-model or per-cell floor (AGENTS.md "no hidden fallback";
task instruction "平均値で失敗モデルや失敗セルを救済しない").

1. **Target recovery:** for every one of the 5 cohort models, sequence EM `>= 0.95` on:
   - standalone SORT at `L=3,4,5` (exhaustive proportion), and
   - each of the 7 target composition classes (i.i.d. sample, Wilson CI lower bound `>= 0.95` is
     not required — the point estimate must clear `0.95`; the CI is reported for precision, not as
     an additional gate, matching the point-estimate floors used throughout NRQ-006/007).
2. **Existing-capability preservation:** for every model, on every regression/canary panel
   (section 3.1-3.3 of the panel manifest):
   - degradation `<= 1.0` percentage point relative to the `FROZEN_PARENT` baseline measured under
     the identical procedure, **and**
   - the panel's own pre-existing acceptance floor is still met (e.g. standalone `L in {6,...,10}`
     stays `>= 0.95`; composition/canary classes stay at the all-cell-gate pass they hold today).
3. **Causal control:** for every model, at both length groups (`{3,4,5}` and `{6,...,10}`):
   `correct_exact_match >= 0.95`; `causal_gap >= 0.50`; `none_exact_match <= 0.06`
   (`natural_baseline["SORT"] + 0.05`).
4. **Invariance:** Core, all 7 non-SORT primitives, router, and argument-scorer
   `canonical_state_hash` unchanged before/after (`Δ = 0`, exact hash match, section 2).
5. **Strict fresh-load parity:** after saving the candidate, a fresh process reloads it via
   `load_bundle(..., mode="diagnostic")` and reproduces every metric above to `Δ = 0` (matching the
   two-independent-process bitwise-reproducibility convention already used by NRQ-006/007/008).
6. **Full reporting:** every one of the 5 models' full result set (all panels, all criteria) is
   recorded, whether it individually passes or fails. A model that fails is reported as a failing
   model, not omitted or averaged away.

Secondary (recorded, not gating): updated parameter count (from `num_parameters()` on the live
SORT module), total training examples consumed, wall-clock time, and peak VRAM/RAM. This pilot
does not claim superiority over any full-cohort retraining alternative — no such comparison is
run.

## 9. Resource budget (repair + evaluation only — parent-cohort construction budget is separate)

This budget is **not** the parent-cohort construction budget
(`PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md` section 7, which is materially larger and
covers building the 5 base models this pilot starts from).

| Item | Per model | 5 models |
|---|---:|---:|
| Repair optimizer steps (registered ceiling) | 6,000 | 30,000 |
| Training examples consumed | 6,000 x 32 = 192,000 | 960,000 |
| Resident parameters touched (unfrozen) | SORT's own `CrossPositionPrimitive` only (~18k-21k, exact count read from `num_parameters()` at execution time) | same, x5 (independent copies, no sharing) |
| Active parameters during repair | = resident (single primitive; nothing else receives a gradient) | same |
| Temporary/plastic parameters | 0 (direct update of an existing persistent bank slot in a staged namespace, section 6 — not a plastic-workspace residual) | 0 |
| Wall-clock [ESTIMATE, not measured] | <= ~5 minutes, assuming <= 50ms/step (conservative, given SORT's primitive is ~3-4 orders of magnitude smaller than the "30-100M model: straightforward" comfort tier in `docs/HARDWARE_ENVIRONMENT.md`) | <= ~25 minutes |
| Peak GPU VRAM [ESTIMATE] | well within the 16GB budget; batch 32, sequence length <= 10, ~20k-parameter module | same |
| Peak RAM [ESTIMATE] | negligible versus the 64GB budget | same |
| Evaluation forward passes (section 7 totals) | ~1,000+10,000+100,000 (exhaustive) + 6,000x5 (standalone regression) + 1,000x7 (target composition) + 500x11 (composition regression) + 500x8 (canary) + 1,000x3x2 (causal control, two length groups) ≈ 152,000 per model | ≈ 760,000 total |

Wall-clock/VRAM/RAM figures are explicitly **planning-upper-bounds**, not measurements; no timing
run was executed to produce this task's deliverables. At execution time these must be replaced
with recorded measurements per AGENTS.md's evidence requirements ("Record ... hardware/time/memory
... for meaningful runs").

## 10. Success and failure claim scope (fixed regardless of outcome)

**If the pilot passes every criterion in section 8:** the only claim authorized is **local repair
success for the registered target panel** (the 7 classes and the `{3,4,5}` standalone domain, on
the specific 5-model cohort registered in
`PHASE_D_FIVE_MODEL_COHORT_CONSTRUCTION_CONTRACT.md`). This is explicitly **not**:
- a claim of composition-execution guarantee across the full 60-class/1200-cell registry,
- a claim about unknown-relation transfer (this pilot's target panel is pre-existing development
  knowledge, not an unseen relation — section 2 of the panel manifest),
- a recovery or reversal of any Phase B/C terminal status (`CLOSED_ARCHIVED` /
  `TERMINATED_CURRENT_CHARTER` remain unmodified — see the Phase D charter),
- a production/candidate-bundle adoption decision (`bundle_promotion` remains `NOT_AUTHORIZED`,
  section 6),
- a claim that single-primitive training generalizes to end-to-end composition training — no
  end-to-end training over composed sequences was performed by this recipe; only standalone
  single-primitive training was performed, and composition behavior is measured, not trained.

**If the pilot fails any criterion:** the failure is recorded as **a failure of this exact
registered recipe** (this optimizer/LR/scheduler/loss/sampling/step-budget combination, on this
cohort). It does **not** automatically authorize: additional training steps beyond the 6,000/model
ceiling, additional seeds beyond the 5 registered, an alternative architecture, a switch to
repairing COUNT or any other primitive, or any other follow-up experiment. A follow-up requires a
new, separately authorized task. The failure is also not extended into a general claim about APC's
feasibility.

**G1/sealed status:** the pre-existing independent-relation-count deficit (G1) and sealed-partition
boundary from Phase B (`docs/DECISIONS_PHASE_B.md`, `docs/DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md`)
are preserved unmodified. This pilot does not evaluate unknown-relation transfer and does not
touch sealed data (`sealed_access: 0`).

## 11. Authorization statement

This document is a preregistration. **No optimizer step, model initialization, candidate
construction, or sealed-data access was performed to produce it, nor by Task D-003's subsequent
approval review.** `training_execution: AUTHORIZED` as seed-amended by Task D-005
([ADR-0172](../DECISIONS_PHASE_D.md#adr-0172-d-005-phase-d-cohort-seed-amendment--complete-static-registry-audit-and-replacement-authorization)),
strictly scoped to the exact recipe, cohort, comparison conditions, and panels this document fixes
— any deviation from those fixed values requires a new, separately recorded authorization.
`bundle_promotion` remains `NOT_AUTHORIZED` and `sealed_access` remains `0` regardless of this
authorization or of the pilot's eventual outcome (section 6, section 10).
