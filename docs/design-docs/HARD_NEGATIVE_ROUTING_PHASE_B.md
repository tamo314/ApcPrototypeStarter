# Design — Hard-Negative Routing for Phase B

## Goal

Test retrieval capacity against semantically competitive alternatives rather than only easy orthogonal distractors.

---

## 1. Difficulty ladder

### L0 — Orthogonal

Retain the existing easy control.

### L1 — Random score-space

Random competitors live in the same score/key space but are not explicitly pushed near the target.

### L2 — Near-neighbor

Construct competitor keys with controlled similarity to a target query/key.

The generator must record the achieved similarity/distance.

### L3 — Semantically related learned primitive

Use keys from real learned primitives whose task semantics are related enough to compete.

Do not fabricate semantic status from only vector proximity.

### L4 — Confusable family/argument variant

Challenge discrimination within a family or between closely related task specifications.

Do not multiply persistent primitive families per argument just to construct this level.

---

## 2. Separate retrieval from acceptance

Runtime should conceptually log:

```text
task representation
    ↓
candidate proposal / ranking
    ↓
candidate set
    ↓
functional verification
    ↓
accepted computation or rejection
    ↓
controller action
```

Record the first failure point.

---

## 3. Metrics

For each `(N, level)`:

- target rank;
- top-1;
- top-k;
- score margin;
- target/competitor similarity;
- candidate-set coverage;
- wrong candidate accepted?;
- false reuse;
- false plastic;
- final EM;
- primitive forward calls.

Top-1 alone is not sufficient.

---

## 4. Safety invariant

Functional verification should make the system conservative under uncertainty.

A hard negative may rank high, but if its executed function fails the support/verification examples, it should not be accepted as adequate.

Therefore distinguish:

```text
retrieval_error
functional_acceptance_error
controller_error
execution_error
```

---

## 5. Scaling

Primary bank sizes:

`16, 32, 64, 128`

Do not call N=128 "128 semantic skills" unless every entry is genuinely learned semantic knowledge.

Report the composition of the bank:

- real semantic primitives;
- consolidated primitives;
- hard-negative learned competitors;
- synthetic/distractor entries.

---

## 6. Implementation constraint

Hard-negative generation belongs to evaluation/benchmark infrastructure unless a runtime component genuinely needs it.

Do not contaminate production routing code with oracle difficulty labels.
