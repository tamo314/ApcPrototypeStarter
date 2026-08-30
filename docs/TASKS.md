# Task Queue

Use these as issue-sized prompts. Complete them in order unless an earlier task reveals an architecture problem.

## Task 001 — Bootstrap the repository

**Goal:** Create the Python package skeleton and development tooling described in `README.md` and `AGENTS.md`.

**Required files:**
- `pyproject.toml`;
- `src/apc/...` package directories;
- `tests/`;
- `configs/`;
- `scripts/`;
- `.gitignore`.

**Constraints:**
- Python 3.12;
- PyTorch only as required ML framework;
- pytest, ruff, mypy for dev tooling;
- no model implementation yet beyond an import smoke test.

**Acceptance:** documented verification commands pass.

---

## Task 002 — Implement deterministic symbolic task engine

**Goal:** Implement the reference interpreter and data generator for the Phase A known operations.

Read `docs/exec-plans/active/PHASE_A.md`, Milestone A1.

**Acceptance:**
- deterministic by seed;
- generated sample includes latent operation graph metadata;
- interpreter tests cover every primitive operation;
- no neural-model dependency.

---

## Task 003 — Implement fixed dense baseline

**Goal:** Add the smallest Transformer baseline that can learn initial synthetic tasks.

**Constraints:**
- configurable model size;
- CPU smoke configuration;
- CUDA mixed precision only inside training code;
- checkpoint + metrics output.

**Acceptance:** smoke training overfits a tiny deterministic dataset; reload reproduces outputs.

---

## Task 004 — Implement low-rank primitive module and bank

**Goal:** Add `Primitive`, `PrimitiveBank`, accounting, status and usage metrics.

Do not add dynamic expansion yet.

**Acceptance:** tests prove total/persistent/active counts and freeze behavior.

---

## Task 005 — Implement top-k primitive router

**Goal:** Route a hidden state to at most k primitives and expose scores/entropy/IDs.

**Acceptance:** deterministic tests for top-k; gradients flow only through intended selected path/weights according to implementation choice; usage logging works.

---

## Task 006 — Build the composition benchmark

**Goal:** Define train and held-out composition splits of known operations.

**Acceptance:** evaluator can label examples K/C; model/controller does not see these oracle labels; benchmark report includes composition generalization.

---

## Task 007 — Implement plastic workspace

**Goal:** Add temporary low-rank capacity and an allocator with small/medium/large presets.

**Acceptance:** temporary parameter count changes without changing persistent count; frozen persistent weights remain unchanged after a plastic training step.

---

## Task 008 — Implement novelty signals and state controller

**Goal:** Implement error + router entropy first, finite-state transitions and hysteresis.

**Acceptance:** transition log includes trigger metrics and capacity changes; unit tests cover no-oscillation behavior.

Do not implement RL.

---

## Task 009 — Add novel-operation task family

**Goal:** Add at least one operation not expressible under the configured known-composition budget.

**Acceptance:** static composition baseline fails materially more often than on held-out compositions; oracle metadata marks N but remains hidden from controller.

---

## Task 010 — Implement first consolidation path

**Goal:** Distill active temporary transforms into lower-rank candidate primitive(s).

**Acceptance:** candidate is smaller; reports task imitation score and prior-task replay score; temporary module is not deleted.

---

## Task 011 — Implement shadow validation and release

**Goal:** Compare candidate vs temporary solution and release temporary capacity only after configured thresholds pass.

**Acceptance:** failing shadow test preserves temporary capacity; passing test installs candidate and releases temporary capacity atomically.

---

## Task 012 — Sequential Phase A benchmark

**Goal:** Run at least two learn/consolidate/release cycles in one task stream.

**Acceptance:** produce the metrics and plots required by `docs/EXPERIMENT_PLAN.md`; no unsupported research claims in the report.

---

## Task 013 — Add baselines

Implement B0–B4 from `docs/EXPERIMENT_PLAN.md` using shared data, evaluation and budgets where possible.

---

## Task 014 — Phase A review

Do not add features. Audit:
- reproducibility;
- capacity accounting;
- result validity;
- baseline fairness;
- failure cases;
- documentation drift.

Write `docs/exec-plans/completed/PHASE_A_RESULT.md` only after measured results exist.
