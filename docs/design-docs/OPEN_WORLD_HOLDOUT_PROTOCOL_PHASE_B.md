# Design — Open-World Holdout Protocol for Phase B

## Purpose

Prevent the Phase B unseen-family result from becoming another closed-universe tuning result.

The holdout is a protocol property, not merely a train/test split.

---

## 1. Family states

Every operation family has exactly one research status:

```text
DEVELOPMENT
SEALED_EVALUATION
RETIRED_FROM_SEALED
```

### DEVELOPMENT

May be used for:

- implementation debugging;
- architecture choice;
- threshold choice;
- plastic budget choice;
- search-budget choice.

Cannot support unseen-family generalization claims.

### SEALED_EVALUATION

May be executed for a declared gate.

Its result may not be used to tune the mechanisms evaluated by that gate.

### RETIRED_FROM_SEALED

A former sealed family that influenced a design change.

Keep its results, but it no longer counts as unseen-family evidence.

---

## 2. Required metadata

Each family definition should expose evaluation-only metadata:

- family ID;
- status;
- structural dependency type;
- output shape rule;
- argument schema;
- generator version;
- date/status transition if persisted in docs.

Do not pass status or family ID to model/controller inference.

---

## 3. Recommended structural criterion

The primary sealed family should differ in dependency structure from the families that shaped Phase A.2.

A recommended form is a same-length local-neighborhood conditional transform.

Why same-length:

- avoids changing decoder/output protocol;
- isolates computation novelty from output-format novelty.

Why local-neighborhood:

- differs from pure tokenwise transforms;
- differs from fixed permutation;
- differs from gather/count/keyed retrieval;
- requires cross-position conditional computation.

This is a design proposal and must still pass empirical novelty-validity checks.

---

## 4. Novelty-validity precheck

For each candidate sealed operation:

1. run direct primitive search;
2. run allowed composition search;
3. use the same functional adequacy threshold as the frozen controller;
4. if adequate, do not label the operation novel.

Record:

```text
best_direct_EM
best_direct_loss
best_composition_EM
best_composition_loss
best_recipe
adequate_by_existing_library
```

A "surprising composition success" is a positive APC result, but it invalidates that operation as a novel-family test.

---

## 5. Identifiability precheck

Before task inference:

- explicit TaskSpec must uniquely specify the intended operation/arguments.

Before few-shot inference:

- demonstrate uniqueness under the candidate task set.

Before controlled language:

- confirm the instruction grammar is semantically unambiguous.

The benchmark should produce an explicit ambiguity reason instead of silently returning a target.

---

## 6. Seal invalidation

The sealed result becomes non-final if any of the following is changed because of its performance:

- controller decision threshold;
- adequacy threshold;
- router architecture;
- router embedding dimension;
- hard-negative proposal mechanism;
- direct candidate budget;
- composition beam/budget;
- compact plastic capacity;
- fallback trigger;
- training-step budget;
- consolidation acceptance threshold.

When this occurs:

1. save the original failed result;
2. add an ADR;
3. mark the family `RETIRED_FROM_SEALED`;
4. designate another family;
5. do not erase the earlier result.

---

## 7. Suggested tests

- status cannot be model-visible;
- sealed and development registries are disjoint;
- deterministic seed produces identical examples;
- changing family status changes protocol metadata only, not task examples;
- novelty-validity precheck catches composition-solvable operations;
- leakage detector rejects an input containing oracle family/operation ID in no-ID modes.
