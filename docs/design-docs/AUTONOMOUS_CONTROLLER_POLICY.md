# Autonomous Controller Policy

## 1. Objective

The controller should autonomously decide whether a task can be solved by one existing primitive, a composition of existing primitives, or new plastic computation.

It should not infer novelty from registry membership or an unseen operation name.

## 2. Runtime evidence

For a small support set `S={(x,y)}`, compute evidence such as:

### Direct evidence

- best primitive support exact match,
- best primitive token accuracy,
- best primitive loss,
- top-1 / top-2 score margin,
- router confidence.

### Composition evidence

- best recipe support exact match,
- best recipe loss,
- recipe depth,
- direct-vs-composition improvement.

### Retrieval evidence

- recurrence key/prototype similarity,
- distance to known task representations,
- router margin.

The primary adequacy signal is **functional performance on support examples**.

## 3. Runtime action

```text
best direct candidate
        |
        | sufficient
        v
  DIRECT_REUSE

        | insufficient
        v
composition search
        |
        | sufficient
        v
     COMPOSE

        | insufficient
        v
  PLASTIC_SEARCH
```

## 4. K/C/N/R semantics

### K — Known

Existing primitive is sufficient. Expected action: `DIRECT_REUSE`.

### C — Composition

No single primitive is sufficient, but an existing composition is sufficient. Expected action: `COMPOSE`. No bank expansion.

### N — Novel

Neither primitive nor allowed composition is sufficient. Expected action: `PLASTIC_SEARCH`. On successful consolidation, bank grows exactly once.

### R — Recurrence

A previously novel task returns after consolidation. Expected action: `DIRECT_REUSE`. No adaptation, no bank growth, no reconsolidation.

## 5. Learned controller

Train a small controller on **evidence features**, not oracle operation identity.

Oracle K/C/N/R labels may be used as supervision/evaluation labels, but not runtime inputs.

Suggested outputs:

- direct probability,
- compose probability,
- plastic probability.

## 6. Primary controller metrics

Report:

- K false-plastic rate,
- C false-plastic rate,
- C direct-misroute rate,
- N missed-novelty rate,
- N plastic-trigger rate,
- R recurrence-reuse rate,
- overall action accuracy.

Primary targets:

- K false plastic <=10%
- C false plastic <=10%
- N plastic trigger >=90%
- R direct reuse >=90%
- composition action accuracy >=85%
- K/C-vs-N AUROC >=0.90

## 7. Controller leakage prohibition

Do not use:

- oracle operation labels,
- `example.program`,
- hidden oracle metadata,
- "is this op registered?" as novelty truth,
- target from held-out evaluation.

Support-set targets are allowed because the controller is explicitly an adaptation/search controller.

## 8. Threshold policy

Thresholds are calibrated on development tasks/seeds only. Freeze them before final >=5-seed evaluation.

Do not retune thresholds on final runs.

## 9. Failure interpretation

### Direct false positive

Controller selects an existing primitive that is inadequate. This is a missed novelty/composition error.

### False plastic

Controller expands despite adequate existing computation. This is a lifetime-capacity efficiency failure.

### Recurrence plasticity

Controller relearns a consolidated task. This is a persistent-memory/retrieval failure.
