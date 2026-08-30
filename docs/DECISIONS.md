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
