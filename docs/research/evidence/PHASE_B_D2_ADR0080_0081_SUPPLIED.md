

**## ADR-0080: SHIFT Seed-24 Adequacy Audit — Primitive Inadequacy, Asymmetric-Rule Premature Acceptance, and a Benchmark-Generation Nondeterminism Bug**

**\*\*Date:\*\*** 2026-09-06

**\*\*Status:\*\*** Accepted (Task B-C005D2-005 Complete; diagnostic-only, no repair authorized)

**\*\*Affects:\*\*** \`src/apc/evaluation/adequacy\_reference\_audit.py\`, \`scripts/run\_phase\_b\_b2\_adequacy\_reference\_audit.py\`, \`configs/phase\_b\_b2\_adequacy\_reference\_audit.yaml\`, \`tests/test\_adequacy\_reference\_audit.py\`, \`docs/DECISIONS.md\`, \`docs/DECISIONS\_PHASE\_B.md\`, \`docs/CODEX\_TASKS\_PHASE\_B\_B2\_SECOND\_DIAGNOSTIC.md\`

**\*\*Run Artifacts:\*\*** \`runs/phase\_b\_b2\_second\_diagnostic/\` (\`shift\_seed24\_adequacy\_audit.json\`, \`reference\_adequacy\_summary.json\`, \`adequacy\_reference\_audit\_config.yaml\`, \`adequacy\_reference\_audit\_protocol.json\`, \`adequacy\_reference\_audit\_system.json\`, \`adequacy\_reference\_audit\_console.log\`)

**### Context**

ADR-0079 (B-C005G) reported false plastic \`1.000\` in every sealed seed-24 / SHIFT cell (all five hard-negative levels) at bank sizes 16, 32, and 128, while bank size 64 accepted reuse. ADR-0076 had earlier attributed all false-plastic outcomes in the *\*original, unrelated\** B-C005 sealed gate (seeds 0–4) to "100% finite-support estimator variance" acting on a genuinely adequate candidate. Task B-C005D2-005 tests whether that same attribution explains the new seed-24/SHIFT cells, using a >= 1024-example independent reference batch, the installed \`SequentialAdequacyVerifier\`'s full step-by-step trace, and a diagnostic-only symmetric counterfactual rule (Wilson lower bound >= 0.95 → accept, upper bound < 0.95 → reject, otherwise uncertain).

**### Evidence**

1\. **\*\*Reference adequacy\*\*** (\`n=1024\` held-out examples per bank size, never used for support/query/training, executed through the same installed SHIFT primitive): the primitive at \`model\_seed=4\` (\`24 % 5\`) measured \`reference\_EM\` in \`[0.9033, 0.9248]\` across all four bank-size reconstructions, with the Wilson 95% CI upper bound never exceeding \`0.940\`. Zero of four cells were \`reference\_adequate\`. This falsifies "finite-support variance acting on an adequate candidate" for this seed/operation: **\*\*TRUE\_PRIMITIVE\_INADEQUACY\*\***.

2\. **\*\*Reclassification\*\*** (D2-005.3) of the four sealed cells against this reference: two (bank 32, 64) were \`FUNCTIONALLY\_JUSTIFIED\_PLASTIC\` (the verifier correctly rejected a genuinely inadequate candidate); the remaining two (bank 16, 128) were **\*\*\`UNSAFE\_REUSE\`\*\*** — the verifier *\*accepted\** the same genuinely inadequate candidate after a single \`n=32\` support draw (\`31/32\` and \`32/32\` respectively) whose point estimate happened to clear \`0.95\`, despite a Wilson lower bound (\`0.843\`, \`0.893\`) far below threshold. Zero cells were \`TRUE\_FALSE\_PLASTIC\`.

3\. In both \`UNSAFE\_REUSE\` cells, the diagnostic-only symmetric rule returns \`UNCERTAIN\` rather than \`ACCEPT\` on the identical evidence (D2-005.4), directly implicating the installed rule's asymmetric early-accept condition (empirical EM >= threshold alone, with no confidence-interval confirmation) as the proximate mechanism: **\*\*SEQUENTIAL\_RULE\_BIAS\*\***, specifically in the premature-accept direction. The rule's forced-reject-at-max-support behavior (observed for bank 32/64 at \`n=96\`–\`128\`) is a separate, deliberate, conservative tie-break and is not evidence of bias.

4\. \`verify\_installed\_rule\_matches\_spec\` confirms \`SequentialAdequacyVerifier\` implements exactly its documented specification (early accept \`EM >= threshold\`; early reject Wilson \`upper < threshold\`; otherwise gather evidence; forced decision at \`max\_support\`) across 10 synthetic \`(successes, trials)\` cases — the asymmetry is a property of the *\*specified\** rule itself, not an implementation defect in the verifier.

5\. **\*\*Unrelated to the SHIFT-specific finding, but discovered while reconstructing this cell:\*\*** repeated runs of the identical config and seeds produced *\*different\** per-bank-size accept/reject outcomes across separate \`python\` process invocations, despite \`deterministic\_algorithms=True\`. Root cause isolated to \`generate\_benchmark\_examples\` (\`src/apc/evaluation/recurrence\_benchmark.py\`, duplicated in \`consolidation\_benchmark.py\`), whose RNG seed includes \`hash(operation) % 10000\` — Python randomizes string hashing per process by default, and no run script in this repository pins \`PYTHONHASHSEED\`. Pinning \`PYTHONHASHSEED=0\` made two independent full reruns of this audit bit-for-bit identical, confirming the diagnosis. This affects every Phase B benchmark that calls \`generate\_benchmark\_examples\`, including the archived B-C005/B-C005D/R1/R2/B-C005G runs, whose exact per-seed examples were therefore never guaranteed reproducible from a fresh process.

**### Consequences**

\- ADR-0076's attribution that false plastic was "100% finite-support estimator variance" is **\*\*retrospectively qualified\*\***: it remains the correct attribution for the original B-C005 sealed cells it measured (seeds 0–4; not reinterpreted here), but does **\*\*not\*\*** generalize to the new sealed seed-24/SHIFT cells, where the installed primitive is genuinely inadequate (\`TRUE\_PRIMITIVE\_INADEQUACY\`) and where two of four bank-size draws instead exhibit the opposite failure mode (\`SEQUENTIAL\_RULE\_BIAS\`, premature-accept direction). ADR-0076 itself is not rewritten.

\- A previously invisible safety gap is newly disclosed: the installed \`false\_functional\_acceptance\_rate\` metric (\`0.000\` throughout B-C005G) only tracks acceptance of a *\*wrong competitor's\** arguments and cannot detect acceptance of the *\*correct\** family/arguments backed by a primitive that is itself below the 0.95 functional bar. B-C005G's "wrong functional acceptance = 0.000 — PASS" must not be read as ruling out this failure mode.

\- Per AGENTS.md's forbidden-conclusions guidance, this ADR does **\*\*not\*\*** recommend lowering the adequacy threshold (the primitive is genuinely below it) and does **\*\*not\*\*** recommend enlarging the router (retrieval/ranking remained perfect top-1 at every level and bank size for this operation; the failure is in primitive execution accuracy and verifier acceptance policy).

\- No repair is authorized by this ADR. \`B-C005D2-006\` must incorporate this evidence into its integrated causal-diagnosis table before any next-repair recommendation.

\- The \`generate\_benchmark\_examples\` hash-seed nondeterminism (evidence item 5) is a separate, repository-wide reproducibility bug, disclosed for the user's attention. It is outside B-C005D2's diagnostic scope and touches infrastructure shared by many historical results, so it is **\*\*not\*\*** fixed by this task; recommended as a small, dedicated follow-up (e.g., replacing \`hash(operation)\` with a deterministic hash, or pinning \`PYTHONHASHSEED\` in run scripts).

**---**

**## ADR-0081: Integrated Causal Diagnosis and Next-Repair Decision Gate (Second Diagnostic Phase Complete)**

**\*\*Date:\*\*** 2026-09-06

**\*\*Status:\*\*** Accepted (Task B-C005D2-006 Complete; diagnostic-only, no repair authorized — STOP per \`docs/CODEX\_TASKS\_PHASE\_B\_B2\_SECOND\_DIAGNOSTIC.md\` Section 9)

**\*\*Affects:\*\*** \`src/apc/evaluation/integrated\_causal\_diagnosis.py\`, \`scripts/run\_phase\_b\_b2\_integrated\_causal\_diagnosis.py\`, \`configs/phase\_b\_b2\_integrated\_causal\_diagnosis.yaml\`, \`tests/test\_integrated\_causal\_diagnosis.py\`, \`docs/DECISIONS.md\`, \`docs/DECISIONS\_PHASE\_B.md\`, \`docs/CODEX\_TASKS\_PHASE\_B\_B2\_SECOND\_DIAGNOSTIC.md\`

**\*\*Run Artifacts:\*\*** \`runs/phase\_b\_b2\_second\_diagnostic/\` (\`final\_causal\_diagnosis.json\`, \`integrated\_causal\_diagnosis\_config.yaml\`, \`integrated\_causal\_diagnosis\_protocol.json\`, \`integrated\_causal\_diagnosis\_system.json\`)

**### Context**

\`B-C005D2-001\` through \`B-C005D2-005\` each diagnosed one mechanism of the \`B-C005G\` sealed re-gate failure in isolation. Task \`B-C005D2-006\` is the final task of the second diagnostic phase: combine that evidence into one causal findings table and exactly one recommended next research action (or \`UNRESOLVED\`), without implementing any repair. Every row of the table below is read directly back from the five prior tasks' own JSON artifacts (\`summary.json\`, \`representation\_stage\_summary.json\`, \`semantic\_relation\_summary.json\`, \`l4\_argument\_breakdown.json\`, \`shift\_seed24\_adequacy\_audit.json\`) — none is recomputed or asserted from intuition, per the task doc's own D2-006.1 rule. A runtime guard (\`\_assert\_no\_forbidden\_conclusions\`) additionally checks the output never contains the two conclusions Section 2 of the task doc explicitly forbids.

**### Findings table**

\| Mechanism | Evidence | Verdict | Confidence | Next action |

\|---|---|---|---|---|

\| L2 retrieval | D2-005's SHIFT/seed-24 reconstruction: \`primitive\_call\_top1 = 1.0\` at L0/L1/L2 for every bank size; matches ADR-0079's reported \`L0-L2 top1 = 1.000\` | NO\_FAILURE | HIGH | none |

\| L3 task representation | D2-002 \`z\_probe\_accuracy = 1.0\` at the failing \`regate\_sealed/R2\_frozen\_post\_repair\` cell | NO\_FAILURE | HIGH | none |

\| L3 query projection | D2-002 \`q\_probe\_accuracy = 1.0\` at the same cell; \`query\_proj\` is frozen and identical across R0/R1/R2 | NO\_FAILURE | HIGH | none |

\| L3 key/scoring | D2-002 per-relation verdicts at the focus cell: \`COUNT->BIND\` and \`BIND->COUNT\` = \`KEY\_SCORING\_BOTTLENECK\` (\`control\_a\` top1 0.40/0.628 vs \`z\`/\`q\` probes both 1.0); \`SHIFT->CYCLE\_FOUR\` = \`NO\_FAILURE\`; \`SELECT->BIND\` = \`UNRESOLVED\` (shuffled-control artifact) | KEY\_SCORING\_BOTTLENECK (scoped to \`COUNT<->BIND\`) | MEDIUM | scope any future key-scoring repair to \`COUNT->BIND\`/\`BIND->COUNT\` only |

\| L3 relation split | D2-003: \`cross\_relation\_spread\_at\_focus\_cell = 0.878\`; difficulty-matched \`matched\_gap = 0.366\` stays close to \`raw\_gap = 0.370\` | SEMANTIC\_RELATION\_HOLDOUT\_REQUIRED | HIGH | define development/validation/sealed relation sets before any next repair training |

\| L4 family routing | D2-004: \`family\_top1 = 1.0\` for SHIFT/SELECT/COUNT/BIND at the focus cell | NO\_FAILURE | HIGH | none |

\| L4 argument resolution | D2-004 \`failure\_classification\`: SHIFT/COUNT \`NO\_FAILURE\`; BIND \`ARGUMENT\_SCORER\_GENERALIZATION\_FAILURE\`; SELECT \`ARGUMENT\_ENCODING\_FAILURE\` | MIXED | HIGH | scope any future argument-scorer repair to \`SELECT\`/\`BIND\` only |

\| adequacy estimator | D2-005: \`implementation\_matches\_spec = true\` (10/10 synthetic cases), but \`SEQUENTIAL\_RULE\_BIAS\` present in \`overall\_classification.labels\` (2/4 \`UNSAFE\_REUSE\` cells) | SEQUENTIAL\_RULE\_BIAS | HIGH | confirm asymmetric early-accept before deploying any confidence-based verifier change (Option E) |

\| installed SHIFT adequacy | D2-005: \`reference\_EM\` in \`[0.9033, 0.9248]\` across all 4 bank sizes; Wilson upper bound never exceeds \`0.940\` | TRUE\_PRIMITIVE\_INADEQUACY | HIGH | primitive functional-generalization repair for this seed/operation, not the controller (Option F) |

\| false-plastic metric | D2-005: \`n\_true\_false\_plastic = 0\`, \`n\_unsafe\_reuse = 2\` — the metric tracks only plastic-when-should-reuse and has no signal for reuse-when-should-go-plastic | METRIC\_MISCLASSIFICATION | HIGH | adequacy metric/protocol repair (Option E) |

D2-001's own \`representativeness\_verdict\` (\`NON\_REPRESENTATIVE\`, flag \`DEVELOPMENT\_DIFFICULTY\_MISMATCH\`) is reproduced in the artifact for completeness but is not itself a table row, since it describes the development/sealed split rather than one runtime mechanism; its consequence is folded into the "L3 relation split" row and the recommendation below.

**### Recommendation**

Evaluating D2-006.2's six allowed options against the table above: **\*\*A\*\*** (query-projection repair) and **\*\*B\*\*** (task-representation repair) are not evidence-supported (both \`z\_task\` and \`query\_proj\` retain full separating information everywhere); **\*\*C\*\*** (semantic-relation holdout redesign), **\*\*D\*\*** (argument-scorer repair, scoped to SELECT/BIND), **\*\*E\*\*** (adequacy metric/protocol repair), and **\*\*F\*\*** (primitive functional-generalization repair for the installed SHIFT candidate) are all evidence-supported.

**\*\*Primary recommended next action: Option C — semantic-relation holdout redesign.\*\*** This is recommended as the single gating action, not merely one option among equals, because it precedes every other supported option: D2-003 found genuine model-generalization failure even after matching development and sealed examples on relation and geometric difficulty (\`MODEL\_GENERALIZATION\_FAILURE\_AFTER\_MATCHING\`), meaning any future repair — key-scoring for \`COUNT<->BIND\`, argument-scorer changes for SELECT/BIND, or SHIFT primitive retraining — trained under the current seed-only development/sealed split risks reproducing the exact develops-fine/fails-sealed pattern this entire D2 phase exists to diagnose. Options D, E, and F remain evidence-supported and are recorded in \`final\_causal\_diagnosis.json\` as pending actions, but are not authorized by this ADR.

Per D2-006.3, this diagnosis does not conclude the router needs to be larger (the key-scoring bottleneck is scoped to one specific relation pair, not a capacity claim) and does not conclude the adequacy threshold should be lower (the installed SHIFT primitive is genuinely below it).

**### Consequences**

\- The second diagnostic phase (\`B-C005D2-001\` through \`B-C005D2-006\`) is complete. Per the task doc's Section 9 STOP condition, no repair, threshold change, \`B-C006\`, or Task Inference work is authorized by this ADR or any prior D2 task. \`B-C006\` and all dependent Task Inference work remain blocked pending an explicit user instruction naming the next repair task.

\- All historical ADRs (0075–0080) remain in force; none is reinterpreted or overwritten by this diagnosis.

\- Any future repair task must first define development/validation/sealed relation sets (Option C) before training, per this ADR's recommendation; once that protocol change is in place, Options D, E, and F become candidate follow-on repair tasks, each scoped to the specific operations/mechanisms this diagnosis identified rather than applied uniformly.