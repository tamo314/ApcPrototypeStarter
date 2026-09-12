> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# I03 Shared Content-Prep Release / Score–Value Conflict Pilot

Task ID: `B-C005REC-004P`. This is a single I03 mechanism-repair training
pilot after REC-004O's negative score-only result. It authorizes neither a
candidate, child bundle, RG3 query, sealed evaluation, REC-005, nor an
architecture repair.

Starting from REC-004D's complete I03 `P_LENGTH_POSITION_BIAS` state at
cumulative step 6000, run exactly steps 6001–12000 with the historical
optimizer, scheduler, RNG and deterministic training stream. The only change
from REC-004O is that `content_in_proj.*` and
`content_position_embedding.*` (`CONTENT_PREP`) train with Q/K rows and the
position-bias MLP. V rows remain bit-identical, including every row-shaped
AdamW state. The shared scalar AdamW step may advance because Q/K legitimately
update. All other primitive tensors, Core, other primitives, router,
ArgumentScorer and shared cache are protected.

Before output observation, lock two new, mutually disjoint development sets:
`content_prep_release_validation_v1` (1024 normal examples) and
`content_prep_release_length10_confirmation_v1` (512 length-10 examples).
They must be disjoint from steps 1–12000, the complete prior REC-004 lineage,
including REC-004O's sets, and sealed data.

Evaluate I03@6000 (`EARLY`), official REC-004O `run_003` score-only@12000
(`SCORE_ONLY`), historical joint@12000 (`JOINT`), and this arm (`CP_SCORE`)
on both sets with J0 and O1. The primary success gate requires CP_SCORE J0
overall and length-10 validation EM, confirmation EM, and both O1 EMs to be
at least 0.95, plus source/data/freeze contracts. Its label is
`SHARED_CONTENT_PREP_RELEASE_SUPPORTED_WITH_COMPATIBILITY`.

If that gate misses, classify in order: score–value conflict when CP_SCORE
improves J0 length-10 EM by at least 0.10 over SCORE_ONLY on both new sets and
either O1 EM is below 0.95; helpful-but-insufficient when the same improvement
holds and both O1 EMs pass; otherwise no support. The effect floor is
diagnostic only. The terminal step 12000 is the sole decision point.
