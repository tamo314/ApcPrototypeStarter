# NRQ-002 — Constructive Falsification Experiment: Lawful-Disambiguation Dichotomy

**Document ID:** `DOC-NRQ-002-CONSTRUCTIVE-FALSIFICATION`
**Date:** 2026-09-13
**Status:** Completed; `decision: NO_COUNTEREXAMPLE_CONSTRUCTED` (sub-finding:
`BOUNDED_RESOURCE_LOOPHOLE_CLOSED_FOR_CURRENT_REGISTRY`)
**Task Type:** Non-experimental constructive/mathematical falsification attempt, exactly one
occurrence, per the task's own instruction ("実施する… 1件だけ"). Not a Phase C task (Phase C is
`TERMINATED_CURRENT_CHARTER`, ADR-0160) and not a Phase D charter. This is a direct, targeted
follow-on to NRQ-001 (`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`, ADR-0161): where NRQ-001 *surveyed*
candidate estimands against the Lawful-Disambiguation Dichotomy it derived, this task *actively
tries to construct a counterexample* to that dichotomy, specifically targeting the one gap NRQ-001
itself flagged as open ("the one honest loophole") and did not fully close for arbitrary finite
compositions.
**Research execution:** `NOT_AUTHORIZED` (unchanged; this task authorizes none).

---

## 0. Task instruction and scope boundary

The task instruction (translated from the user's original Japanese) is:

> Conduct exactly one constructive falsification experiment for the Lawful-Disambiguation
> Dichotomy. Without training or sealed access, without giving the baseline task identity or the
> corresponding ground-truth `apply`, and with finite compute/description-length constraints fixed
> in advance, explore whether a finite composite estimand can be constructed that simultaneously
> satisfies (1) $H(Z\mid\mathcal O)=0$, (2) oracle-free, (3) relation-transfer compatible, and
> (4) the strongest lawful deterministic baseline scores below 0.95. If a counterexample is
> obtained, it refutes NRQ-001's generalization and program-line closure. If none is obtained,
> record the search space and the reasons for impossibility, and re-audit the termination decision.

This is a read-and-reason task. It does not run code, train a model, register a relation,
construct a dataset, or open sealed data — consistent with the task's own "training・sealed
accessなしで" constraint and with how NRQ-001 and every Phase-C `C-D001*` task were executed
(pre-execution mathematical/design review, not an experimental run).

| Prohibited action | Count |
|---|---:|
| Training / optimizer update | 0 |
| Model initialization / seed draw | 0 |
| Dataset / relation generation or registration | 0 |
| Architecture / primitive implementation | 0 |
| Candidate construction / bundle write | 0 |
| GPU experiment time | 0 s |
| Sealed partition access (input/label/output) | 0 |
| Ground-truth `apply` handed to any constructed baseline | 0 |
| Task-identity descriptor handed to any constructed baseline | 0 |

Sources read (beyond the ones NRQ-001 already read, which this task inherits): NRQ-001's own
review document and JSON record, `docs/DECISIONS_PHASE_C.md` ADR-0161, `src/apc/primitives/
composition_search.py` (the actual A1-B004 beam-search implementation), the A1-B004/A1-B005/A2-C004
ADR entries in `docs/DECISIONS_A1_R005E_DIAGNOSTIC.md` and `docs/DECISIONS_PHASE_A2.md`, and
`docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md` §5 (relation-unit definition). No
sealed artifact was opened.

---

## 1. The four admission criteria, formalized (task's own numbering)

A constructed estimand $E$ is a **falsifying counterexample** to the Lawful-Disambiguation
Dichotomy only if it survives all four filters simultaneously:

1. **Identifiability.** $H(Z \mid \mathcal O) = 0$ under the permitted, oracle-free observable set
   $\mathcal O$ — $Z$ is a deterministic function of $\mathcal O$, not merely of hidden task state.
2. **Oracle-free.** Scores 0/5 on the ADR-0155/0156 five-dimensional oracle-supervision criterion.
   Concretely for this task: the constructed baseline receives neither the task/relation identity
   nor the ground-truth `apply` that generated $Z$ — only the same lawful, oracle-free observables
   a learner would see (inputs, structural/positional statistics, declared non-oracle descriptors).
3. **Relation-transfer compatible.** Per NRQ-001 criterion 4's operational meaning: the estimand's
   design does not need to weaken, bypass, or silently redefine the standing >=2-clean-component
   (validation) / >=2-clean-component (sealed) relation-transfer requirement. An estimand that
   invokes no held-out-*relation* generalization claim at all trivially satisfies this criterion
   (nothing to transfer, nothing bypassed); an estimand that *does* invoke one inherits the
   existing, twice-confirmed 2-of-4 deficit (ADR-0147, ADR-0150) and fails outright.
4. **Non-triviality under a pre-fixed finite compute/description-length budget.** The **strongest
   lawful deterministic baseline** — any zero-or-bounded-parameter, non-oracle, non-learned
   procedure allowed the same finite compute and description-length budget fixed *before* the
   estimand is examined — scores below 0.95 on the relevant metric.

These map onto NRQ-001's five criteria as: (1)=NRQ-001 criterion 2, (2)=criterion 3,
(3)=criterion 4, (4)=criterion 1. The task does not restate NRQ-001's criterion 5 ("actually tests
core separation / conclusion-changing") explicitly, but it is inherited implicitly: resolving
whether the dichotomy has a counterexample is itself conclusion-changing for the program-line
closure claim (ADR-0161), so no separate filter is needed here — a genuine counterexample would
automatically qualify.

$$E \text{ is a counterexample} \iff E \text{ passes criteria 1--4 simultaneously.}$$

---

## 2. What NRQ-001 already established, and the specific gap this task targets

NRQ-001 §3 proved the **Lawful-Disambiguation Dichotomy**: for any discrete $Z$ that is a
deterministic function of frozen task semantics, and any lawful oracle-free observable set
$\mathcal O$ (linear, nonlinear, invertible, lossy, or interactive), exactly one of

- (A) $H(Z\mid\mathcal O) > 0$ (unidentifiable — fails this task's criterion 1), or
- (B) $H(Z\mid\mathcal O) = 0$, hence $Z=g(\mathcal O)$ for a lawfully computable $g$, which "is
  exactly the class of function an evaluator is already permitted to hard-code as a
  zero-learned-parameter baseline" (fails this task's criterion 4)

holds. NRQ-001 §3 itself flagged the one place this argument has a genuine gap:

> "**The one honest loophole, and why it is closed under current scope.** Case B assumes $g$ is
> *tractable to state* as a baseline. If $g$ were true but intractable to write down in closed
> form, a learned approximation could be non-trivial in a practical (not information-theoretic)
> sense. This loophole does not apply to any currently registered APC relation: [...] $g =
> \zeta_{\mathcal T}$ already [is] exactly known and computable for every relation used in this
> project [...] A relation whose ground truth were intentionally intractable to state would
> violate this invariant before an estimand could even be posed, and would itself be a new-primitive
> scope change (§4, candidate N2), not a minimal experiment on the current registry."

NRQ-001 closed this loophole only for the case of **a single relation's own ground truth being
intractable**. It did not examine whether a **finite composition of several already-tractable,
already-registered primitives** could reopen the same loophole — i.e., whether stacking simple,
individually-tractable deterministic operations could produce an aggregate $Z$ that is still
$H(Z\mid\mathcal O)=0$ (composition of deterministic functions is deterministic) but whose
disambiguating function $g$ becomes intractable to *compute within a pre-fixed finite budget*,
even though each component is individually easy. This is precisely the task's own phrase "有限合成
estimand" (a finite **composite** estimand) and is the gap this task's constructive attempt targets.

This is a genuinely different question from anything NRQ-001's six-candidate survey (N1–N6)
covered — none of N1–N6 concerned composition-recipe recovery or computational-resource bounds as
the axis of non-triviality. It is therefore examined here as a fresh construction attempt, not
re-litigated from NRQ-001's table.

---

## 3. Construction attempts

Four candidate finite composite estimands were actively constructed and checked against criteria
1–4. Each is a genuine attempt to open the loophole, not a restatement of an already-excluded case.

### Attempt A — Composition-recipe identification under recipe-space combinatorial growth

**Construction.** Let $Z$ = the identity of a length-$k$ composition recipe
$(\text{op}_1, \dots, \text{op}_k)$ drawn from the resident primitive bank of size $m$, applied
consistently to all examples in an episode. The recipe is *not* given to the baseline as a
descriptor (satisfies criterion 2's "no task identity"); it must be inferred purely from a small
set of oracle-free (input, output) pairs sharing the same recipe (cross-example structural
consistency — lawful per ADR-0155/0156). If $K$ examples uniquely pin down one recipe among $m^k$
candidates (a pigeonhole/uniqueness argument), $H(Z\mid\mathcal O)=0$ holds by construction
(criterion 1 passes). The hoped-for counterexample requires that a deterministic enumerate-and-check
baseline, bounded to a small pre-fixed compute budget, cannot search all $m^k$ candidates before
the budget is exhausted, forcing baseline accuracy below 0.95, while some other lawful procedure
(left unspecified) still succeeds.

**Result: closed, empirically.** This exact construction — recover a composition recipe from a
small oracle-free adaptation set, with **zero access to `example.oracle_metadata`,
`example.program`, or `step.operation`** (verified directly in `src/apc/primitives/
composition_search.py`'s own docstring: "Strict Invariants Enforced: 1. Zero oracle primitive
identity") — is not hypothetical; it is `Task A1-B004` (ADR-0049), already executed and already
measured. At the scale actually used ($m=8$ resident primitives, `max_depth` up to 3, giving at
most $8+8^2+8^3=584$ raw candidates before any pruning), a **heuristic beam search** — a
zero-learned-parameter, deterministic, oracle-free procedure, exactly the class of "strongest
lawful deterministic baseline" this criterion asks about — achieves **99.62% mean exact match**
(threshold was $\ge 85\%$) and **99.93% functional agreement**, from an adaptation set of only
$N=32$ examples, with strictly zero bank expansion. The baseline **exceeds 0.95**, failing this
task's criterion 4 directly and empirically, not merely by the abstract dichotomy argument.
Structural pruning (rejecting length-invalid candidates before evaluation) is why the effective
search cost stays far below the raw $m^k$ bound — most of the $584$ raw candidates are eliminated
in $O(1)$ per candidate by output-length constraints alone, so even the *unpruned* bound never
approached any realistic compute ceiling at this scale.

### Attempt B — Pushing recipe-space size past any single pre-fixed budget

**Construction.** Attempt A failed because $m^k$ stayed small ($584$). Could $k$ or $m$ be
increased, within the "finite composition" framing, until $m^k$ genuinely exceeds any compute
budget one would plausibly pre-register (e.g., $>10^{12}$ enumerations)?

**Result: closed, by the project's own hardware/scope invariants.** The largest primitive-bank
scale ever exercised in this repository is $N=128$ (Task A2-C004, ADR-0065, "Bank Competition
Robustness and Compute Scaling to N=128"), and no task has run composition search at depth beyond
3. Even at that (never-jointly-tested) combination, $128^3 \approx 2.1\times10^6$ — still trivially
enumerable in seconds on CPU, let alone within an "hours-scale milestone experiment" budget this
project's own `AGENTS.md` already contemplates ("multi-hour sweeps belong only to explicit
milestone experiments"). To reach $m^k>10^{12}$ with $m\le 128$ requires $k\ge 6$; no acceptance
criterion, benchmark, or design document in this repository composes more than 3–4 primitive calls
in one recipe, and `AGENTS.md` explicitly forbids the kind of open-ended architecture/hyperparameter
search that would be needed to justify introducing such a depth as a *default* experiment on a
single RTX 5060 Ti / 64 GB workstation. Manufacturing $m^k$ large enough to defeat a fixed budget
would therefore require an **out-of-scope escalation of scale** (banned by the same "single
workstation, no distributed training, no architecture search" invariant already governing this
project), not a "minimal" finite composite estimand on the current registry. This closes Attempt B
on scope grounds, independent of Attempt C/D's information-theoretic argument below.

### Attempt C — Argument-space combinatorial blow-up (same idea, applied to arguments instead of recipe steps)

**Construction.** Instead of composing operations, fix one operation but let $Z$ depend on a
combination of several discrete arguments (e.g., a `BIND` query key and a `SELECT` index set drawn
jointly from a larger domain), hoping the joint argument space is large enough to defeat a fixed
enumeration budget while remaining, in principle, oracle-free identifiable from a handful of paired
examples.

**Result: closed, by the project's own measured argument domains.** ADR-0031 (A1-R005D-002)
directly audited every registered argument encoder's real legal domain and found them small and
**injective** by construction: `SHIFT.amount` over `range(10)`, `COUNT.target`/`BIND.query_key`
over `range(10)` (vocab size), bucketed into $\le 32$ discrete embedding rows. Joint argument spaces
built by combining a handful of such slots (e.g., two 10-valued arguments) reach at most
$10\times10=100$ combinations — far below any threshold that would strain a fixed enumeration
budget, and already the scale at which ADR-0031's audit checked "every pair of distinct values"
exhaustively (100 pairs for `SHIFT`, 45+45 for `COUNT`/`BIND`) as a matter of routine, non-milestone
verification. As in Attempt B, reaching genuine intractability would require enlarging argument
domains far past what any registered operation, generator default, or acceptance criterion in this
repository uses — again an out-of-scope scale escalation, not a finite composite estimand on the
existing registry.

### Attempt D — Deliberately hard (cryptographic/NP-hard-style) composition target

**Construction.** Abandon the current primitive set's simple algebraic structure and posit a
composition whose recipe-identification problem is *designed* to be hard to search but easy to
verify (a planted-solution / one-way-function-style construction: e.g., a hash-like scrambling
composed from several primitives such that finding the recipe from examples requires an
exponential search, but checking a candidate recipe against one example is $O(1)$).

**Result: closed, on the same grounds as NRQ-001's candidate N2.** None of the eight currently
registered primitives (`SHIFT`, `SELECT`, `COUNT`, `BIND`, `MIRROR_HALVES`, `NEIGHBOR_MAX`, plus
the two SHIFT-adjacent compact operators from Branch B) has any known cryptographic one-wayness or
average-case hardness property — they are simple permutation/selection/counting/binding operations
over small finite domains, chosen specifically for their tractability and human-auditable semantics
(`docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`'s design intent). Manufacturing a genuinely hard
verifier/prover gap would require introducing a **new primitive family** with deliberately
intractable structure. This is not a composition of the existing registry; it is a new-primitive
scope change, dispositionally identical to NRQ-001's candidate N2 (stochastic ground truth): it is
not shown impossible in the abstract, but it requires (i) unauthorized new-primitive registration
beyond a "minimal" experiment on the current registry, and (ii) — independently — inherits N2's own
criterion-3 failure, because a deliberately novel primitive family is itself an unregistered
relation, and the relation inventory remains short by the same twice-confirmed 2-of-4 margin
(ADR-0147, ADR-0150) that has not changed since NRQ-001. Attempt D is excluded on scope (criterion
2's "no new relation/oracle construction this task can authorize") before criterion 4 is even
reachable.

### Attempt summary table

| ID | Construction | Criterion 1 ($H=0$) | Criterion 2 (oracle-free) | Criterion 3 (relation-transfer) | Criterion 4 (baseline < 0.95) | Verdict |
|---|---|---|---|---|---|---|
| A | Recipe-space combinatorics at tested scale ($m=8$, depth $\le 3$) | Passes | Passes | Trivially compatible (no relation-transfer claim needed) | **Fails — measured 99.62% EM, empirically baseline-dominant** (A1-B004) | **Closed (empirical)** |
| B | Recipe-space pushed past any fixed budget ($m^k$ large) | Would pass in principle | Would pass in principle | Trivially compatible | Unreachable within scope — requires banned scale escalation | **Closed (scope)** |
| C | Argument-space combinatorics | Would pass in principle | Would pass in principle | Trivially compatible | Unreachable — measured domains ($\le 100$ joint combos) stay enumerable | **Closed (measured domain size)** |
| D | Deliberately hard (cryptographic-style) composition | Unclear | **Fails — requires a new, unregistered, potentially-hard-to-verify-as-lawful primitive** | Independently fails (same 2-of-4 deficit as N2) | Untested — moot | **Closed (scope, same disposition as N2)** |

No fifth construction strategy was identified that escapes both Attempt B/C's measured-scale
closure and Attempt D's scope closure while also passing Attempt A's empirical baseline-dominance
finding.

---

## 4. The Bounded-Resource Corollary (why this class of loophole stays closed in general)

Beyond the four concrete attempts, a general argument closes the entire family of "combinatorial
tractability gap" constructions for the currently registered primitive class, not just the four
instances tried:

1. **Structural pruning defeats naive combinatorial blow-up for these primitives specifically.**
   Every registered primitive carries simple, checkable structural constraints (output length as a
   function of input length and arguments; legal argument domains). As `composition_search.py`
   demonstrates, these constraints prune most of the raw $m^k$ or domain-product search space in
   $O(1)$ per candidate *before* any expensive evaluation, because the primitives were designed to
   be simple and compositional (not adversarially entangled). This is not an incidental
   implementation detail; it follows from `AGENTS.md`'s own invariant that primitives are lawful,
   auditable, deterministic operations, not obfuscated ones.
2. **No free lunch without exploitable structure.** If a composed target genuinely had no
   exploitable structure (uniformly hard, no compressible pattern), no learner — including a
   trained neural network — could generalize from a finite training/adaptation set either; the
   problem would be uninformative about APC's architecture (fails to be conclusion-changing, the
   same disposition NRQ-001 gave candidate N5/N6). If a composed target *does* have exploitable
   structure (which is what would let a hypothetical learner succeed where a naive baseline fails),
   that same structure is, by definition, available to a **structured** lawful deterministic
   procedure (dynamic programming, constraint propagation, or the same beam-search style already in
   this codebase) — not only to gradient-based learning. Nothing about `SHIFT`/`SELECT`/`COUNT`/
   `BIND`/`MIRROR_HALVES`/`NEIGHBOR_MAX` is known to admit a statistical-query-style separation
   (structure exploitable by gradient descent but provably not by any efficient deterministic
   search) — that is a property of specific cryptographic/parity-style constructions (Attempt D),
   not of this project's simple symbolic primitives.
3. **Reaching genuine intractability requires an out-of-scope change.** Every attempt to force a
   real gap between "$H(Z\mid\mathcal O)=0$" and "computable within a pre-fixed budget" required
   either (a) scale far beyond anything this repository's hardware/scope invariants permit as a
   default experiment (Attempts B/C), or (b) a new primitive type with deliberately non-lawful,
   hard-to-audit structure (Attempt D) — which independently fails the relation-transfer criterion
   via the same unremedied inventory deficit NRQ-001 already found (ADR-0147/ADR-0150, unchanged
   since that review, confirmed unchanged as of this task).

**Corollary.** For any finite composition of the currently registered, non-cryptographic,
structurally-pruneable APC primitives, evaluated at any scale this repository has actually
exercised or could exercise within its own stated hardware/scope invariants, the
Lawful-Disambiguation Dichotomy's Case B holds without the tractability caveat: $g$ remains
tractable to state (and, per Attempt A, is already empirically stated and baseline-dominant). This
extends NRQ-001's closure of the loophole from "a single relation's ground truth" to "any finite
composition of registered relations at in-scope depth/domain size."

---

## 5. Relation-transfer criterion — re-confirmed unchanged

None of Attempts A–D that reached criterion 3 needed to invoke a genuine held-out-relation claim
(each stayed within the existing registered primitive set, using cross-example structural
consistency within already-known relations rather than transfer to an unseen one), so criterion 3
was satisfied trivially for A/B/C. Attempt D would have needed one (a new primitive is an
unregistered relation) and failed it for the same reason NRQ-001's N2 did. The underlying inventory
fact is unchanged and was re-checked, not re-derived, for this task:

| Partition | Required independent clean components | Available | Deficit |
|---|---:|---:|---:|
| Validation | >= 2 | 1 (`L3:CYCLE_FOUR-SHIFT`) | 1 |
| Sealed | >= 2 | 1 (`family:sealed_local_neighborhood`) | 1 |

No relation-registration work has occurred between ADR-0161 (2026-09-13) and this task (same day).

---

## 6. Decision

$$\mathbf{DECISION:\quad NO\_COUNTEREXAMPLE\_CONSTRUCTED}$$

No constructed estimand survives all four criteria simultaneously. Attempt A is closed by direct
empirical evidence already in this repository (A1-B004, 99.62% EM, oracle-free, zero bank
expansion) — the single most concrete test this task could have hoped to run turns out to already
have been run, with the opposite of the hoped-for result. Attempts B and C are closed because
manufacturing the required combinatorial scale would itself violate this project's own hardware/
scope invariants. Attempt D is closed on the same scope/inventory grounds as NRQ-001's candidate
N2. The Bounded-Resource Corollary (§4) generalizes this closure to the whole family of
"tractability gap via finite composition" constructions, not just the four instances tried.

Per the task's own branching instruction, because no counterexample was constructed, §7 records
the search space and reasons for impossibility and re-audits the termination decision.

---

## 7. Re-audit of the termination decision

**What this task adds.** NRQ-001 (ADR-0161) confirmed `PROGRAM_LINE_CLOSURE_CONFIRMED` for the
oracle-free task/relation-inference research line, while explicitly flagging one open gap (the
tractability loophole) as closed only for single-relation ground truth, not proven closed for
arbitrary finite compositions. This task closes that specific, previously-open gap: no finite
composition of the current registry, at any in-scope scale, reopens the dichotomy's Case B into a
practically non-trivial regime. This is a **strengthening** of ADR-0161's finding, not a
re-opening — it removes a caveat NRQ-001 itself had left unproven for compositions, rather than
reversing anything NRQ-001 concluded.

**What does not change.**

| Component | Status | Changed by this task? |
|---|---|---|
| Core separation, explicit TaskSpec (Phase A/A.1/A.2) | `VALIDATED` | No |
| Semantic task-inference architecture (Phase B) | `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE` | No |
| Routing-identifiability estimand (Phase C, H-C1) | `TERMINATED_CURRENT_CHARTER` | No |
| Oracle-free task/relation-inference research line | `PROGRAM_LINE_CLOSURE_CONFIRMED` (ADR-0161) | **Reinforced** — the composition/tractability gap is now explicitly closed, not merely un-examined |
| Relation inventory for any future relation-transfer claim | `STRUCTURALLY_INSUFFICIENT` (2 of 4 required) | No — re-confirmed unchanged |
| APC in general / future differently-scoped premises (e.g., a genuinely new hard primitive, N2/Attempt-D style) | `NOT_EVALUATED` | No — still explicitly out of scope, still not shown impossible in the abstract |

**Termination decision: unchanged and reinforced.** `PROGRAM_LINE_CLOSURE_CONFIRMED` stands.
No new research question, hypothesis, architecture, relation family, or execution authorization is
proposed or granted by this task. As with NRQ-001, this is explicitly **not** a claim of APC's
general impossibility, and does not touch Phase A/A.1/A.2's positive core-separation evidence.

---

## 8. Integrity and consistency checks performed

- Confirmed A1-B004's cited figures (bank size 8, `max_depth` support up to 3, 99.62% EM, 99.93%
  functional agreement, zero bank expansion) directly against `src/apc/primitives/
  composition_search.py`'s source and `docs/DECISIONS_A1_R005E_DIAGNOSTIC.md`'s ADR-0049 text —
  not re-derived, not re-run.
- Confirmed A2-C004's $N=128$ figure and ADR-0031's argument-domain sizes directly against
  `docs/DECISIONS_PHASE_A2.md` (ADR-0065) and `docs/DECISIONS_A1_R005E_DIAGNOSTIC.md` (ADR-0031).
- Confirmed the relation-inventory deficit numbers are unchanged from ADR-0147/ADR-0150/ADR-0161
  (same day, no relation-registration work performed in between).
- Confirmed no ADR text (ADR-0001–ADR-0161) is rewritten, renumbered, or deleted by this task.
- Confirmed the highest existing ADR number in the repository is ADR-0161 before assigning
  ADR-0162 to this task's decision record.
- Confirmed this document does not relax, add to, or remove any charter threshold, floor,
  oracle-boundary criterion, initialization count, or relation requirement.
- This is a documentation-only change; per `AGENTS.md`'s documentation-only exception, `pytest`,
  `ruff`, and `mypy` were not re-run (no file under `src/`, `tests/`, or `configs/` changed).

---

## 9. Consequences and documentation updated

- `docs/DECISIONS_PHASE_C.md` — ADR-0162 appended (this task's decision record). Filed here per
  `docs/DECISIONS.md`'s standing "append to the latest record file" instruction (same file ADR-0161
  used).
- `docs/DECISIONS.md` — index row added for ADR-0162, appended to the existing NRQ bridging
  section.
- `docs/TASKS.md` — NRQ-002 completion recorded; no task queued.
- `README.md` — one-line NRQ-002 status added alongside the existing NRQ-001 line.
- `docs/research/NRQ002_REVIEW_RECORD.json` — machine-readable integrity/decision record.

No code, test, config, or run artifact is touched by this task.
