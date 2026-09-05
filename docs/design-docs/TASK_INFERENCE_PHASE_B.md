# Design — Task Inference for Phase B

## Purpose

Replace explicit operation identity with progressively weaker task evidence while keeping the downstream APC controller and causal content path interpretable.

---

## 1. Core boundary

Task Inference produces task-side state.

It must not perform the content transformation itself.

Conceptually:

```text
task observation
    ↓
TaskInference
    ↓
z_task
    ↓
router / controller / PrimitiveCall

content
    ↓
task-blind Content Encoder
    ↓
h_content
    ↓
selected primitive / composition
    ↓
output
```

Load-bearing invariant:

`h_content = f(content)`

---

## 2. TaskObservation modalities

A common structure should represent one of:

### Explicit TaskSpec

Upper bound only.

### Structured descriptor

Semantic fields/tokens without canonical operation ID.

### Demonstrations

Input-output pairs.

### Controlled natural-language instruction

Generated instruction text with semantic template splits.

The runtime interface should not expose oracle family/operation labels in no-ID modes.

---

## 3. Demonstration-role separation

Use:

```text
inference_examples
verification_examples
query_examples
```

### inference_examples

Consumed by TaskInference.

### verification_examples

Consumed by direct/composition functional adequacy checks.

### query_examples

Consumed only for reported final performance.

This separation prevents a task inference model from receiving the same targets later claimed as held-out functional evaluation.

---

## 4. Structured descriptor stage

Purpose:

- remove explicit registry ID;
- keep semantics easy and deterministic;
- validate the interface before few-shot inference.

Use held-out templates.

A canonical enum such as `operation=SHIFT` is forbidden in primary no-ID input.

---

## 5. Few-shot stage

TaskInference must aggregate a set of demonstrations.

Important properties:

- permutation-invariant aggregation over demonstration order where practical;
- support for variable shot count;
- explicit mask/padding behavior;
- no query-target access;
- identifiability audit.

The architecture should be the smallest model adequate for the gate.

Do not introduce a pretrained LM.

---

## 6. Controlled language stage

Instruction generation should separate semantics from a single canonical wording.

Recommended splits:

```text
train templates
dev templates
held-out test templates
```

Optional:

```text
held-out lexical aliases
```

The operation enum token remains absent.

The claim is controlled semantic instruction inference, not unrestricted language understanding.

---

## 7. Evaluation-only task identity

Ground-truth operation/family identity may be retained for:

- supervision during training when the experiment allows;
- metric calculation;
- confusion matrices;
- failure localization.

It may not be injected into runtime inference in a no-ID condition.

---

## 8. Failure localization

Every integrated episode should be able to distinguish:

```text
task inference failed
candidate retrieval failed
functional verification failed
controller decision failed
primitive execution failed
plastic learning failed
consolidation failed
recurrence failed
```

Do not collapse all of these into "Task Inference accuracy."

---

## 9. Negative controls

At minimum:

- task observation removed;
- task observation mismatched to episode;
- demonstration outputs shuffled;
- wrong-task demonstrations;
- language instruction mismatched;
- explicit TaskSpec upper bound.

A task inference component is not causally supported if matched and mismatched observations perform similarly.

---

## 10. Recommended tests

- no operation ID in serialized primary inputs;
- no K/C/N/R label in model inputs;
- content representation identical across task observations for identical content;
- demonstration order invariance if the chosen aggregator claims it;
- query targets unavailable to TaskInference;
- ambiguity detector catches collided tasks;
- held-out language templates are disjoint from training templates;
- task observation negative controls materially change task-side state while leaving content state unchanged.
