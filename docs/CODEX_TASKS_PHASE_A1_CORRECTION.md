# Codex Task Queue — Phase A.1 Correction

Run these tasks after A1-006 and before A1-007.

Do not renumber the existing A1 tasks.

---

## Task A1-C001 — Explicit task specification schema

### Goal

Make every output-changing task variable model-visible.

### Work

1. Define a task specification representation for the synthetic environment.
2. Encode operation identity.
3. Encode required arguments:
   - SELECT index,
   - COUNT target,
   - SHIFT amount,
   - BIND key/spec.
4. Keep oracle metadata separate from model-visible task specification.
5. Add interpreter/generator tests.

### Acceptance

- two examples with identical content but different task specification can correctly have different targets,
- task specification fully determines all previously hidden operation parameters,
- no model-facing path reads oracle-only fields.

---

## Task A1-C002 — Mixed-operation online generator

### Goal

Generate one identifiable mixed-operation training stream.

### Work

1. Extend online generation to sample all eligible known operations.
2. Emit explicit task specification with every example.
3. Keep fresh procedural content.
4. Use one shared vocabulary/encoding.
5. Primary run uses `permute_symbols=False`.

### Acceptance

- all included operations appear in one stream,
- same content can be paired with multiple operations,
- outputs remain deterministic from visible task spec + content.

---

## Task A1-C003 — Shared-core input encoding

### Goal

Allow the existing Stable Core to consume task/control and content information in one model.

### Work

1. Define input layout or structured encoder interface.
2. Ensure `encode_split()` exposes:
   - `z_task`,
   - `h_content`.
3. Do not train separate models per operation.
4. Add logging/probe hooks.

### Acceptance

- one model forward handles all known operations,
- task/content states are available separately,
- existing per-operation A1-006 path remains reproducible.

---

## Task A1-C004 — Shared-core systematic-generalization gate

### Goal

Test H1b.

### Work

1. Train one shared model on the mixed online task stream.
2. Evaluate fresh unseen content.
3. Run >=5 seeds.
4. Report per-operation and overall exact match.
5. Add a negative-control run with task specification removed or masked.

### Acceptance

- overall mean >=0.95,
- every operation mean >=0.90,
- negative control materially underperforms the explicit-task model.

### STOP GATE

If this fails, do not start A1-007.
Investigate task encoding / Stable Core training only.

---

## Task A1-C005 — Task/content representation probes

### Goal

Verify that the routing state actually contains task information.

### Work

Freeze a trained shared Stable Core and fit lightweight probes.

Required:

1. operation ID from `z_task`,
2. operation argument from `z_task` where applicable,
3. content feature(s) from `h_content`.

Optional:

- operation from `h_content`,
- content from `z_task`.

### Acceptance

- operation-from-z_task >=0.95,
- applicable argument decoding >=0.90,
- content probe is sufficient to support the operation family.

### STOP GATE

If `z_task` does not contain reliable task information, do not begin learned routing work.

---

## Task A1-C006 — Parameterized PrimitiveCall abstraction

### Goal

Represent primitive selection and arguments separately.

### Work

1. Add a `PrimitiveCall` abstraction.
2. Support at least:
   - SHIFT(amount),
   - SELECT(index),
   - COUNT(target),
   - BIND(key/spec).
3. Preserve parameter-free calls for COPY/NEGATE/COMPARE/ACCUMULATE.
4. Avoid creating one persistent primitive per argument value.
5. Add serialization/reporting support if needed.

### Acceptance

- one primitive family executes at least 3 distinct argument values correctly,
- persistent primitive count does not increase with argument value count,
- tests cover invalid/missing arguments.

---

## Task A1-C007 — Oracle PrimitiveCall routing adapter

### Goal

Prepare A1-007 so the oracle reveals a complete executable call, not only primitive ID.

### Work

1. Convert environment oracle metadata into `PrimitiveCall`.
2. Ensure the learned path never receives oracle calls.
3. Update A1-007 evaluation API to accept an oracle call provider.
4. Add tests for parameterized and parameter-free operations.

### Acceptance

- oracle call fully determines the primitive execution,
- hidden-parameter failures from A1-006 cannot recur in oracle mode,
- A1-007 is now unblocked.

---

## Task A1-C008 — Correction audit / ADR

### Goal

Record the scientific interpretation before continuing.

### Work

Create an ADR or short progress report summarizing:

- H1a A1-006 result,
- H1b shared-core result,
- probe result,
- parameterized primitive decision,
- changes to A1-007 assumptions,
- whether the next task is unblocked.

### Rule

No new architecture in this task.

Audit only.
