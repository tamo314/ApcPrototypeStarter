# Experiment Plan

## 1. Primary hypotheses

### H1 — Selective expansion
The system allocates new trainable capacity substantially more often for genuinely novel operations than for held-out compositions of known operations.

### H2 — Consolidation
A temporary learned solution can be compressed into smaller persistent primitive capacity while retaining most of its performance.

### H3 — Reuse
A consolidated primitive reduces adaptation cost when the learned operation recurs or appears inside a new composition.

### H4 — Bounded persistent growth
Across a structured task stream, persistent parameter growth is lower than cumulative temporary capacity allocation and lower than an expansion-without-consolidation baseline.

### H5 — Retention
The closed loop preserves earlier tasks better than unconstrained fine-tuning of a comparable dynamic model.

## 2. Experimental unit

One run is defined by:
- model configuration;
- task-stream configuration;
- seed;
- compute budget;
- controller thresholds;
- consolidation thresholds.

Never aggregate results across materially different task streams without stratification.

## 3. Task-stream construction

Each stream should contain four event types:

- `K`: familiar operation/task;
- `C`: novel composition of familiar operations;
- `N`: genuinely novel operation;
- `R`: recurrence of a previously learned novel operation.

Example:

`K K C N K C R N C R`

The generator must expose ground-truth event type to the evaluator, but the controller must not receive the label.

## 4. Metrics

### Performance
- exact match;
- token/sequence loss;
- adaptation steps to threshold.

### Continual learning
For task i after learning task t:
- current performance;
- retained performance on all prior tasks;
- maximum forgetting per task;
- mean backward transfer.

### Capacity
- stable-core parameters;
- persistent primitive parameters;
- temporary allocated parameters;
- peak resident parameters;
- active parameters per inference step.

### Consolidation
- temporary-to-persistent compression ratio;
- temporary vs candidate score ratio;
- shadow disagreement;
- fraction of temporary transforms discarded;
- candidate reuse count.

### Compute
- train steps;
- wall-clock;
- peak CUDA memory;
- measured tokens/s or examples/s;
- analytical FLOP proxy by module;
- cumulative/lifetime compute across the stream.

### Routing
- top-k usage distribution;
- router entropy;
- primitive utilization Gini/imbalance metric;
- reuse across task families.

## 5. Baselines

### B0 Fixed dense
A capacity-matched or compute-matched small Transformer with no primitives.

### B1 Fixed sparse
Stable core + fixed primitive bank + router, no expansion.

### B2 Grow-only
Dynamic plastic capacity becomes permanent; no consolidation/pruning.

### B3 Grow + replay
Dynamic expansion with replay but no primitive compression.

### B4 APC
Full finite-state loop.

Where feasible compare both:
- same initial persistent size;
- similar cumulative training compute.

## 6. Ablations

Run after the full loop works:
- remove SEARCH;
- error-only novelty;
- remove router-entropy signal;
- remove replay;
- remove shadow validation;
- consolidate without functional similarity tests;
- allow stable primitive plasticity;
- remove hysteresis.

## 7. Statistical plan

During development: 1–3 seeds.

For milestone claims: target >=5 seeds if run time remains practical on the local machine.

Report:
- mean;
- standard deviation;
- individual-seed points when plots are produced.

Do not over-interpret tiny differences on synthetic tasks.

## 8. Hardware-aware run tiers

### Smoke
- < 2 minutes preferred;
- tiny model/data;
- CPU-capable where possible.

### Dev
- tens of minutes acceptable;
- one GPU;
- 1–3 seeds.

### Milestone
- longer single-GPU runs;
- explicit command and config;
- never triggered by default tests.

### Sweep
Only after a stable baseline exists. Keep search spaces small and log failed/OOM configs.

## 9. Required plots for Phase A report

- performance over task stream;
- persistent vs temporary parameter count over time;
- controller state over time;
- adaptation steps per event type K/C/N/R;
- primitive reuse heatmap by task family;
- forgetting per task;
- lifetime compute proxy by baseline;
- consolidation compression ratio per novel operation.

## 10. Falsification criteria

The hypothesis is weakened if, under matched conditions:
- novel compositions require expansion almost as often as novel operations;
- consolidated primitives do not reduce recurrence adaptation cost;
- persistent capacity approaches grow-only baseline size;
- consolidation causes large forgetting;
- routing overhead removes most compute savings;
- fixed sparse or fixed dense baselines dominate both performance and lifetime compute.
