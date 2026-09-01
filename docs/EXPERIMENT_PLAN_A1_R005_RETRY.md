# Experiment Plan — A1-R005 Diagnostic and Retry

## 1. Goal

Determine why parameterized primitive causality failed, then rerun H2c only after the failing mechanism is isolated.

## 2. Existing result is frozen

Historical A1-R005:

- Correct: 0.308
- Wrong argument: 0.255
- Wrong family: 0.028
- None: 0.143
- causal gap: 0.053

Do not overwrite these artifacts.

## 3. D1 — Existing-run metric expansion

No retraining.

For each operation and seed report:
- Correct exact match,
- Wrong-argument exact match,
- Wrong-family exact match,
- None exact match,
- Correct token accuracy if recoverable,
- Wrong-argument token accuracy if recoverable,
- output length,
- causal gap.

Add `argument_effect_rate`: fraction where wrong argument changes the interpreter target.

Also report wrong-argument metrics restricted to effectful examples.

## 4. D2 — Argument representation injectivity

Verify whether distinct semantic arguments collapse to identical encodings.

Required for SELECT:
- order reversal,
- distinct same-length sequences,
- repeated indices if legal.

Semantically distinct ordered SELECT arguments must not be structurally forced to identical representations.

## 5. D3 — COUNT counterfactual benchmark

Use COUNT only.

Training groups contain same content with several target arguments whose correct counts differ.

Gate:
- Correct exact >=0.90
- effectful Wrong argument <=0.30
- causal gap >=0.50
- >=5 seeds

If this fails, do not move to unified retry.

## 6. D4 — Training/capacity sweep

Only after D3 setup is identifiable.

Staged sweep:
- primitive steps baseline / 2x / 4x,
- primitive rank 8 / 16,
- arg_dim baseline / 2x.

Choose the smallest robust passing configuration.

## 7. D5 — Conditioning architecture comparison

Only if additive conditioning remains weak.

Compare:
- V0 additive,
- V1 FiLM/gated multiplicative,
- optional V2 basis modulation.

Select by causal gap at comparable persistent size, not Correct accuracy alone.

## 8. D6 — BIND gate

Counterfactual same-content/multiple-key groups.

Target:
- Correct >=0.90
- effectful Wrong argument <=0.30
- causal gap >=0.50

## 9. D7 — SHIFT gate

Report:
- exact match,
- token accuracy,
- length-stratified metrics,
- effectful Wrong-argument metrics.

Initial target:
- Correct exact >=0.85
- Correct token accuracy >=0.95
- effectful Wrong-argument token accuracy materially lower.

If low-rank conditioning fails despite strong single-token results, test tiny cross-attention primitive.

## 10. D8 — SELECT gate

Order-preserving argument encoder is mandatory.

Report exact match, token accuracy, and length-stratified metrics.

Initial target:
- Correct exact >=0.85
- Correct token accuracy >=0.95
- effectful Wrong-argument performance materially lower.

## 11. Final R005 retry

All four parameterized operations, >=5 seeds.

Targets:
- aggregate Correct exact >=0.90
- each operation Correct >=0.85
- aggregate effectful Wrong argument <=0.30
- Wrong family <=0.30
- None <=0.30
- aggregate causal gap >=0.50
- one persistent family per operation independent of argument values

Only this gate unblocks A1-R006.

## 12. Stop conditions

Stop and write an ADR if:
- COUNT cannot learn argument causality under counterfactual training,
- SELECT encoding remains non-injective,
- more steps/rank improve Correct but not Correct-vs-Wrong gap,
- passing requires one family per argument value,
- tiny cross-attention cannot solve effectful argument controls.

## 13. Interpretation

Outcome A: counterfactual training fixes R005 -> training shortcut/identifiability issue.

Outcome B: FiLM/basis modulation fixes it -> additive conditioning too weak.

Outcome C: cross-attention fixes SHIFT/SELECT only -> evidence for heterogeneous primitive classes.

Outcome D: controlled COUNT still fails -> parameterized primitive design needs deeper revision.
