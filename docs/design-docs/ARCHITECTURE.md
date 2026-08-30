# APC Architecture

## 1. Objective

APC tests a resource-elastic continual-learning architecture in which persistent computation remains sparse and relatively small, while additional trainable capacity exists only while a genuinely novel operation is being learned.

The architecture separates five concerns that conventional dense training largely entangles:

1. **execution** — use already consolidated skills;
2. **composition** — recombine existing skills without changing persistent weights;
3. **novelty detection** — decide whether existing computation is insufficient;
4. **plastic learning** — temporarily add capacity and learn;
5. **consolidation** — compress useful learned computation, validate it, then release temporary capacity.

## 2. State machine

```text
                 existing composition succeeds
        +---------------------------------------------+
        |                                             |
        v                                             |
      STABLE ---> SEARCH ---> PLASTIC ---> CONSOLIDATE ---> SHADOW
        ^          |             |                          |
        |          |             | learn                   | validate
        |          +-------------+                          |
        |        composition found                          |
        +---------------------------------------------------+
                           release temporary capacity
```

### STABLE
Normal sparse execution. Stable core and primitive bank are used. Persistent primitives are frozen in Phase A.

### SEARCH
Try alternate routing/composition of existing primitives. SEARCH must be cheaper than allocating new trainable capacity.

### PLASTIC
Allocate temporary trainable transforms. Train them while protecting the stable system.

### CONSOLIDATE
Compress the useful temporary computation into one or more candidate persistent primitives.

### SHADOW
Run temporary and consolidated solutions on held-out/replay inputs. Only release temporary capacity if equivalence/retention criteria pass.

## 3. Stable core

Phase A uses a small Transformer or recurrent Transformer-like sequence model. Its responsibilities are deliberately limited:

- encode task/input tokens into a working hidden state;
- expose insertion points for primitive transforms;
- host routing signals;
- decode the final output.

The core should be frozen after initial pretraining/baseline training in the first closed-loop experiments. This isolates whether capability growth can occur through primitives rather than core drift.

Recommended first scale:

- hidden size: 192–384;
- layers: 4–8;
- context length: 64–256 synthetic tokens;
- total parameters: roughly 10M–60M.

Do not target the upper end until the closed loop works at smaller scale.

## 4. Primitive representation

The first implementation should use low-rank residual transforms:

`P_i(h) = h + s_i(h) * B_i A_i h`

where:

- `A_i: d -> r`;
- `B_i: r -> d`;
- `r` is a small rank (default 8);
- `s_i(h)` is a router/gating scalar.

This representation is intentionally simple because it is:

- cheap to store;
- cheap to execute;
- differentiable;
- easy to merge/approximate;
- easy to attribute in capacity accounting;
- similar in spirit to sparse low-rank transform dictionaries.

A primitive record stores:

```text
id
A, B
router key / metadata
status: candidate | stable | archived
usage_count
created_at_task
utility_ema
stability_score
```

Human-readable semantic labels are optional diagnostics, never training targets.

## 5. Primitive bank

The bank owns persistent primitives and exposes:

- append candidate;
- activate/deactivate;
- top-k retrieval;
- persistent parameter accounting;
- active parameter accounting;
- usage statistics;
- merge/archive hooks.

Phase A should start with dozens, not millions, of primitive slots. The objective is to validate behavior, not PEER-scale retrieval.

## 6. Router

The router maps the current working hidden state and optional task context to primitive scores.

Initial version:

- learned query projection;
- learned key per primitive;
- cosine or dot-product score;
- top-k hard selection;
- soft weights inside selected top-k.

The router must expose:

- selected IDs;
- score distribution;
- entropy;
- per-primitive usage.

Router entropy is one novelty signal but must not be the only one.

## 7. Plastic workspace

The plastic workspace contains temporary low-rank transforms with the same execution interface as persistent primitives.

Key invariant:

> Temporary capacity is not part of the persistent primitive bank until consolidation and shadow validation succeed.

The allocator adds capacity in small increments, e.g.:

```text
small:  4 transforms, rank 4
medium: 8 transforms, rank 8
large: 16 transforms, rank 8 or 16
```

Exact values are experimental configuration, not architecture constants.

During Phase A, only temporary transforms and optionally their local router parameters are trained during PLASTIC.

## 8. Novelty estimator

Novelty means **computational insufficiency**, not merely unusual input.

Initial signals:

- task loss/error (`E`);
- predictive uncertainty (`U`), initially ensemble/dropout approximation if needed;
- router entropy (`H`);
- gradient novelty (`G`), measured against a rolling low-dimensional gradient basis or simpler proxy.

A heuristic normalized score can start as:

`N = alpha*E + beta*U + gamma*H + delta*G`

Do not train a learned meta-controller before these signals are observable and logged.

## 9. Controller

Phase A uses a deterministic finite-state controller with hysteresis.

Example thresholds (placeholders, to be calibrated):

- enter SEARCH when performance falls below expected stable range;
- enter PLASTIC only after bounded search fails;
- leave PLASTIC when improvement-per-compute plateaus and validation accuracy exceeds a minimum;
- enter STABLE only after shadow validation and retention checks pass.

Use separate enter/exit thresholds to avoid mode oscillation.

## 10. Consolidation

Consolidation is a structured compression pipeline, not generic model pruning.

Suggested first implementation:

1. measure activity of temporary transforms;
2. drop inactive transforms;
3. collect input/output pairs at each active temporary transform over replay + current-task data;
4. cluster transforms by functional similarity if more than one remains;
5. fit a smaller low-rank transform to reproduce the combined residual delta;
6. distill candidate transform(s) on held-out data;
7. test causal necessity by disabling candidates;
8. register candidates for SHADOW.

The compression objective should include both current-task imitation and prior-task retention.

## 11. Shadow validation

For a configured number of batches, compare:

- temporary solution output;
- consolidated candidate output;
- ground truth;
- prior-task replay performance.

Release criteria should include:

- candidate reaches >= 95% of temporary solution task score (initial target);
- prior-task degradation <= 2 percentage points (initial target);
- no invariant failures;
- candidate persistent parameter cost is materially smaller than temporary peak capacity.

The thresholds are hypotheses and must remain configurable.

## 12. Resource accounting

Track three sizes separately:

1. **resident persistent parameters** — stable core + persistent primitives;
2. **temporary peak parameters** — extra plastic capacity at peak;
3. **active parameters** — parameters actually involved in one forward step.

The core research claim depends on not conflating these.

Also track FLOP proxies and actual wall-clock/VRAM. Parameter count alone is not sufficient.

## 13. Phase A synthetic task universe

Use symbolic token sequences with executable ground truth.

Task families must distinguish:

### Known primitive tasks
Operations seen during initial training.
Examples: COPY, SELECT, COMPARE, COUNT, SHIFT, BIND, NEGATE, ACCUMULATE.

### Novel composition tasks
New chains of known operations. These should be solvable without expansion.

### Novel operation tasks
Operations that cannot be expressed efficiently by the trained primitive set under the allowed composition depth. Examples may include SORT, MODULAR_ADD, or TRANSITIVE_CLOSURE.

### Recurrence tasks
Previously learned operations reappear later. These test reuse and forgetting.

The environment generator must know the latent operation graph so evaluation can distinguish new composition from genuinely new operation.

## 14. Long-term extensions (not Phase A)

Only after the closed loop works:

- learned/constrained-RL controller;
- different plasticity timescales;
- tiny attention primitives;
- memory/computation separation;
- larger primitive retrieval (product-key style);
- pretrained 0.5B–2B language-model core;
- LoRA/QLoRA adaptation;
- mechanistic circuit extraction from dense teachers;
- CPU-hosted primitive bank and prefetch;
- generated primitives via hypernetworks.
