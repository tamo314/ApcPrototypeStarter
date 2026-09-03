# Shared Queryable Representation

## 1. Target property

APC requires a stronger property than per-operation task-blind encoders:

`one shared representation -> many sparse reusable operators`

## 2. Target architecture

```text
                         ┌─> Compact SHIFT(amount)
                         ├─> Compact SELECT(indices)
content -> Shared Encoder├─> Compact COUNT(target)
                         └─> Compact BIND(key)
```

The shared encoder sees only content.

Operation identity is used only outside the encoder to select the diagnostic operator.

## 3. Desired factorization

`h = E_shared(content)`

`y = P_operation(h, argument)`

Forbidden:

`h = E_operation(content)`

or any task/argument-conditioned encoder path.

## 4. Shared-vs-specialized score

Let:

- `M_shared(op)` = Shared Encoder + Compact Operator result
- `M_specialized(op)` = E-006A per-operation Joint + Compact result

Define:

`R_shared(op) = M_shared(op) / max(eps, M_specialized(op))`

for Correct exact and causal gap.

Suggested interpretation:

- `R_shared >= 0.90`: shared representation retains most specialized benefit
- `0.70 <= R_shared < 0.90`: partial sharing / mixed
- `R_shared < 0.70`: strong specialization dependence

## 5. Multi-task interference

Shared training can fail because:
1. a single representation cannot satisfy all operations, or
2. optimization balance causes interference.

Measure:
- per-operation loss curves,
- operation sampling frequency,
- shared encoder gradient norms by operation if practical,
- optional pairwise gradient cosine.

Do not add gradient surgery inside the main gate.

## 6. Readout policy

Preferred first gate:
- one shared encoder,
- per-operation compact operator,
- per-operation readout.

Do not add shared readout as an extra confound.

## 7. SHIFT

If sharing works broadly but SHIFT remains weak, treat the residual as likely operator-inductive-bias mismatch for modular position arithmetic.

Do not modify SHIFT operator inside the main shared gate.

## 8. COUNT/BIND None arm

Do not silently change the historical `None <= 0.30` threshold.

Measure explicit argument-blind baselines and use them to recommend a future baseline-relative criterion.

## 9. Desired conclusion

Answer:

> Is one shared task-blind encoder sufficient to support multiple compact operators?
