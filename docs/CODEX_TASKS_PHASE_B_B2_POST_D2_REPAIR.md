> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Tasks — Phase B / B2 Post-D2 Repair

> **現在地・実行順の正本:** [Phase B 再開計画](exec-plans/active/PHASE_B_RESTART.md)（2026-09-12、ADR-0127）。
> 本文は既存の仕様・作成当時の状態を保持する。現在の進捗と今回の継続指示は再開計画を参照。

**Task IDs：B-C005R3-001～B-C005R3-012**  
**状態：実装前の指示書。B-C006 / Task Inferenceはblocked。**

> **状態更新（2026-09-07）：** `R3-001`～`R3-010`は実行済み。`R3-010`のG4は
> `DEVELOPMENT_INTEGRATION_FAIL`（ADR-0091）。`R3-011`以降は、復旧分岐
> `B-C005REC-001`～`B-C005REC-008`（[タスク](CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)、
> ADR-0092）がartifact復元／再構築判断を完了するまでblocked。

## 0. 使用方法と共通規則

実行するのはユーザーが指定した一タスクだけ。タスク間の自動継続は禁止。最初は **B-C005R3-001**。

本系列はADR-0080/0081後の修正分岐であり、Phase Bを終了して新Phaseへ移る指示ではない。[実行計画](exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md)、[実験計画](EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md)、[エージェント規則](AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md)を先に読む。

### Source of truth

報告された事実：[S1](research/evidence/PHASE_B_D2_ADR0080_0081_SUPPLIED.md)。現コードの実在・関数契約・checkpointを最初のタスクで確認する。報告に記載されたパスは確認対象であり、未確認のAPI仕様を捏造しない。

### 全タスク共通の完了条件

- focused testsとrepo標準検証の実行結果を保存する。未実行をPASSとしない。
- scientific milestoneは5個以上の固有model checkpointで評価し、data seedの繰返しとは区別する。
- 変更許可した機構以外のstate hashを比較し、freezeを検査する。
- 各指標の分子/分母、missing・UNRESOLVED、比較可能性、run artifact pathを記載する。
- failed Gateは保存・ADR追記・依存停止。過去FAILを修正しない。
- GPUの長い実験はmilestone commandのみ。unit testsはCPUで実行可能にする。

数値基準は[実験計画](EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md)を正とする。本文の要約と不一致があれば実行前に停止して解消する。

---

## B-C005R3-001 — Reproducible Benchmark Generation & Artifact Inventory

### 目的

ADR-0080で報告されたhash-seed非決定性を、モデル変更とは独立に修正する。次の研究課題Option Cより前に、同一入力で比較できる土台を作る。

### 読むもの・対象

- `src/apc/evaluation/recurrence_benchmark.py`
- `src/apc/evaluation/consolidation_benchmark.py`
- `src/apc/evaluation/adequacy_reference_audit.py`
- 各generatorを呼ぶPhase B scripts/configs、seed utilities、関連テスト
- [再現性設計](design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md)の§2–3

### 実装

1. 本パックをdocsへ導入。root AGENTS.mdとPhase B task/planへ修正分岐を追記。次ADR番号は実repoから取得する。
2. 乱数導出のcall graphを調べる。`hash(operation)`の2箇所と関連経路を、一つのversion付きdeterministic utilityへ集約する。無関係な全体refactorは禁止。
3. `generator_version`、各roleのdataset hash、モデルとデータの別seed、生成recipeを保存する。
4. 新versionのdefaultを明示し、legacy configの暗黙v2化を禁止。旧runはそのまま保存する。
5. B-C005 / D / R1 / R2 / G / D2で必要なcheckpoint・生入力・JSONの実在を台帳にする。ないものは `UNAVAILABLE`。同じファイルを名前違いで独立モデルと数えない。

### テスト

異なるPYTHONHASHSEEDの4 subprocess、正逆cell走査、単体再開、worker数変更、重複wrapper一致、namespace分離、v1/v2識別を確認する。GPU tensor再現と入力byte再現は別のテストにする。

### 禁止

router・ArgumentScorer・verifier・controller・primitiveの修正、重み更新、threshold変更、全過去Phaseの再学習は禁止。`PYTHONHASHSEED=0`を設定しただけで修正完了にしない。

### 成果物・境界

`runs/phase_b_b2_post_d2/r3_001_reproducibility/` に `artifact_inventory.json`、`generator_audit.json`、`subprocess_comparison.json`、通常run metadataを出す。**G0 PASS**後に次が実行可能。入手不能な過去入力は報告するが、その欠落を理由に捏造せず、新比較が正しく組める範囲を示す。

---

## B-C005R3-002 — Semantic-Relation Split & Exposure Protocol（Option C）

### 目的

ADR-0081が求めるdevelopment / validation / sealed relation setsを、次の学習より先に定義する。

### 実装

1. 実際のhard-negative builder・bank registryからrelation catalogueを作る。COUNT↔BINDと、その他の実在関係を列挙し、観測済み/未評価を記録する。
2. directed relationの逆向きと機能aliasをまとめたgroupでsplitを作る。seedだけの分離へ戻さない。
3. `targeted_repair_regression` と `repair_relation_transfer` を別suiteにする。既知の失敗pairを修正学習した結果は前者で評価する。
4. lossのCE denominator・replay・negative miningを含めた新training exposureを監査する。単にtop-level datasetを分けただけで関係holdoutと主張しない。
5. profile用dev fixtureを凍結親checkpointで測り、COUNT↔BINDやSELECT/BINDの失敗を本当に含むかを確認する。失敗例だけを最終評価から抜き取らない。
6. sealed partitionのmembership/recipeは固定するが、そのmodel出力はまだ測らない。difficulty binとrelation別quotaを事前登録する。

### 禁止

ここでは重み更新・loss変更・candidate scorer変更をしない。real semantic relationが不足したとき、random score attackをL3と呼び替えない。

### 必須テスト

逆relation・aliasリーク、seed modulo alias、held-out pairがCE/replayへ入る違反、history exposureのUNKNOWN、sealed access guardをテストする。

### 成果物・境界

`relation_catalog.json`、`relation_split.json`、`exposure_manifest.json`、`development_representativeness.json`。**G1**を満たさない場合は、学習を始めず `PROTOCOL_INSUFFICIENT_RELATIONS` 等で停止する。

---

## B-C005R3-003 — Functional Metrics v2 & Statistical Contract（Option E / 仕様のみ）

### 目的

「wrong competitorの受理」と「機能不足の正しい候補の受理」を別に測る。verifier変更前に、受理・未確定・候補不足・library scopeの定義を固定する。

### 実装

1. [Adequacy v2設計](design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md)のreference state、runtime verdict、action envelopeを型/schemaへする。
2. offline reference evaluatorとmetric aggregatorを追加/拡張する。runtimeにはまだ新verifierを接続しない。
3. `REF_ADEQUATE / REF_INADEQUATE / REF_UNRESOLVED` をconfidence boundsで分類する。point estimateだけの二分に戻さない。
4. 旧known-task false plasticを残し、新しいinadequate-call acceptance、same-identity unsafe acceptance、avoidable plasticity、coverageを併記する。
5. finite-look exact boundsの誤り配分・support schedule・reference budget・candidate数をCPUで検査する。512例でも境界付近は未確定になり得ることを表にする。
6. nominal performance suiteとdegraded-candidate safety stressを分ける。nominalから不十分候補を結果に基づいて除外しない。

### 必須テスト

empty denominator→null、reference未確定→未確定、abstention→unconditional EMで不正解、正ID/argsでもreference不足ならunsafe、機能同値wrong-IDの扱い、未評価recipeがある場合に「library全体不足」と断言しないこと。

### 成果物・境界

`metrics_schema_v2.json`、`statistical_contract.json`、`metric_fixture_results.json`、`budget_feasibility.json`。**G2 PASS**後に契約を凍結する。旧Gateの定義変更は、新versionとADRとして事前記録する。

---

## B-C005R3-004 — Paired Frozen Baseline & Comparison Repair

### 目的

生成バグを除いた同一入力で、修正の出発点を確定する。第二診断を丸ごと繰り返さず、次の修正判断に必要な対照に限る。

### 実装・評価

- 入手可能な親checkpoint、R1後checkpoint、旧verifierを、固定v2 fixtureで評価する。
- retrieval比較では同じtask・logical candidate graph・synthetic competitor値を固定。modelに属するkey変更は各conditionの明示的変更として記録する。
- adequacy比較では同じcandidate出力streamで、fixed32/64/128、旧asymmetric、新contractのoffline判定を比較する。新runtimeはまだ接続しない。
- COUNT→BIND、BIND→COUNTの低ranking、SELECT/BIND argument failure、対象SHIFTのreference不足を個別に報告する。再現しないものは `NOT_REPRODUCED_ON_V2` とし、元結果を消さない。
- SELECT→BINDのshuffled-control問題は、label/split/metric配線とprobe leakageを一度だけ確認する。修復してもmeasurement bugの修正だけであり、routerを同時変更しない。未確定ならUNRESOLVEDを保持する。

### 判定

ここは性能PASSを要求しない。**比較の正当性がPASSであること**、各対照の適用可能性が明示されることを要求する。個別故障が新fixtureで再現しなければ、その機構の学習は自動開始せず、後続タスクを `NEEDS_SCOPE_REVIEW` にする。

### 成果物

`paired_baseline.json`、`comparison_manifest.json`、`failure_reproduction_matrix.json`、`select_bind_control_status.json`。元sealedのexact replayとv2 reconstructionを別欄にする。

---

## B-C005R3-005 — Safe Bounded Verification（Option E / 実装）

### 目的

ADR-0080で観測された点推定による早期受理を除き、機能不足candidateの受理を防ぐ。router、arguments、primitiveを同時に改善しない。

### 対象

`src/apc/meta/adequacy_verifier.py`、現在のhard-negative/adequacy runner、対応テスト。既存クラスとの互換を保ち、新policyをversion明示で追加する。

### 実装

- R3-003で凍結したfinite-look exact-bound verifierを実装。lower boundでACCEPT、upper boundでREJECT、残りUNCERTAIN。
- 最大budgetのUNCERTAINを強制ACCEPT/REJECTに変えない。薄いaction envelopeでNEEDS_MORE_EVIDENCEを返す。
- 既存3-action controllerのweightsは固定。unsafeなreuse命令の実行をwrapperで遮断する。未確定を0点として13-dimensional evidenceへ入れない。
- direct失敗からcomposition確認を飛ばしてplasticへ行かない。B2 direct fixtureでcompositionを評価していないなら、そこでは「候補verification失敗」とだけ記録する。
- 旧不十分SHIFTをstress対象に残す。まだSHIFTを再学習してはならない。

### 比較

同一stream上でold asymmetric、fixed32/64/128、新verifierを比較する。独立Bernoulli CPU契約テストと、real neural candidateのvalidation stressを分ける。

### 成果物・境界

`verifier_operating_characteristics.json`、`same_identity_safety.json`、`verification_trace.jsonl`、`cost_breakdown.json`。**G3 PASS**が必要。対象SHIFTを安全に棄却/未確定化できたことを、SHIFTの機能修正成功と呼ばない。

---

## B-C005R3-006 — COUNT↔BIND Key/Scoring Repair

### 目的

情報が存在するz/qを作り直さず、ADR-0081で局所化されたCOUNT↔BINDのscoringを修正する。

### 修正前の必須確認

keyとphysical ID対応、normalization、key norm、学習対象parameter、人工的score offset、relation exposureを検査する。これらのbugと学習objectiveの不足を同時に修正しない。

### 実装範囲

- Task Encoder、`query_proj`、ArgumentScorer、verifier、全primitiveは凍結。
- COUNT/BINDに関係するkey/scoringだけを変更可能とする。pairを見たらoracle nameで正解へ分岐する処理は禁止。
- 変更方式はdevelopmentの該当pairと独立validationで一つ選ぶ。巨大router、全bankの自由な再学習は禁止。
- balanced replayでは正常taskを対照として守る。held-out関係を新CE lossへ混入させない。
- frozen baseline、通常objective対照、対象修正版を同一v2 fixtureで比較。十分なdeltaがない場合も、そのまま報告する。

### Gate

COUNT→BINDとBIND→COUNTを別々に評価し、両方向ともvalidation top-1>=0.95、top-5>=0.99、正常routing低下<=1 pp。maskや未選択primitive forward=0。SELECT→BINDの改善を要求してscopeを広げない。

### 成果物

`count_bind_scoring_repair.json`、pair別margin、checkpoint hashes、frozen-state audit、loss exposure log。新sealedには触れない。

---

## B-C005R3-007 — SELECT Argument-Encoding Repair

### 目的

ADR-0081の `ARGUMENT_ENCODING_FAILURE` を、実際のargument schemaに即して修正する。BIND scorerは同時に変更しない。

### 実装

1. 現SELECT schemaが集合、順序付き列、mask等のどれかをコードとunit testsから確定する。報告にないroot causeを先に決めつけない。
2. valid/invalid値、padding、mask、serialization、round-trip、canonicalization、size情報の伝達を検査する。
3. 見つかったcontract violationを最小差分で修正する。変更encodingに依存するSELECT専用headは必要な範囲だけdevelopmentで再学習可能。
4. 意味を保つ並び替えのみ同値として扱う。出力順に意味があるSELECTを勝手にset化しない。
5. SHIFT/COUNT/BIND paths、family scoring、verifier、primitive weightsは凍結。

### 指標・Gate

full argument exact match、schemaに応じたelement precision/recall、set/cardinality等の層別値を報告する。primaryはfull-call accuracyでありelement平均への置換は禁止。validation full argument accuracy>=0.95、L4 full call top-1>=0.90、他operation低下<=1 pp。

### 成果物

`select_encoding_contract.json`、`select_argument_repair.json`、round-trip/property tests、schema migration notes。現contractに欠陥が見つからずscorer学習だけが原因なら、無関係encoding変更をせずscope reviewで停止する。

---

## B-C005R3-008 — BIND Argument-Scorer Generalization Repair

### 目的

BIND専用のscorer generalizationを修正する。R3-007のSELECT修正は固定する。

### 実装

- BINDのargument value/structure、wrong-key距離、頻度、content-length条件ごとの失敗をdevelopmentで確認する。
- oracle TaskSpec argumentを直接渡す経路は評価用上限とし、primaryをoracle lookupへ置換しない。
- BIND compatibility/headとその損失・samplingだけを変更可能にする。global lambdaで他familyのscoreを一律に変更しない。
- dev-onlyのvalue/structure coverageを整え、validation argument条件を使ってtrain例を選ばない。
- family ranking、z/q、SELECT encoding、verifier、primitive bankは凍結。

### Gate

validation BIND argument accuracy>=0.95、full call top-1>=0.90、family top-1>=0.98、top-5>=0.99。他argument pathsの低下<=1 pp。candidate-per-argumentのpersistent複製=0。

### 成果物

`bind_argument_repair.json`、value/structure別結果、calibration、parent/output checkpoint、freeze audit。返した数値がlogitなのか確率なのかを明記し、根拠なくconfidence保証と呼ばない。

---

## B-C005R3-009 — SHIFT Functional-Generalization Repair & Safe Versioned Replacement

### 目的

正しいcallでも機能不足だったSHIFT実装を修正する。controllerの閾値を緩めて解決したことにしない。

### 実装

1. 対象親checkpointとSHIFT primitiveを明示する。旧model_seed=4は診断履歴として残すが、旧sealed入力は学習に使わない。
2. 現SHIFTの長さ・amount・wrap-around等の合法条件をdevelopmentで層化し、コンパクトな別candidateを学習する。既存compact予算をmanifestから取得し、無制限のstep/capacity追加をしない。
3. Stable Core、他primitive、routing/scoring、argument heads、verifierは凍結。generatorのoracle実装をruntime primitiveへ差し替えない。
4. 現candidateとnew candidateを独立shadow setで比較。reference/queryは学習・early stoppingへ渡さない。
5. commitは同じphysical familyの**versioned replacement**とする。新argument値ごとにbankを増やさない。
6. bank transactionでversionを上げ、recipe/cache/reference cacheをweight/schema hashで無効化する。失敗時rollback、成功commit後だけworkspace release。旧checkpointはread-onlyで残す。
7. fresh processで新bankをloadし、adaptation=0、temporary params=0で再評価する。

### Gate

validationのSHIFT mean query EM>=0.99、全modelのREF_ADEQUATE、旧SHIFT条件と他taskのEM低下<=1 pp。既知対象のreuse性能が不足する場合はGate FAILとする。new physical primitive countは増やさず、成功transactionは1回、workspace leak=0、fresh-runtime recurrenceの再学習=0。

`>=0.99` は今回の利用可能性を評価する新規目標であり、過去実績の主張ではない。原理的adequacy thresholdの0.95を変更する意味でもない。

### 成果物

`shift_generalization_repair.json`、length×argument breakdown、shadow/ref/query separation、bank transaction log、fresh-runtime report。旧不十分SHIFTはG5 stress suiteから除去しない。

---

## B-C005R3-010 — Paired Integration & Frozen Legacy Regression

### 目的

局所修正を順番に統合し、相互作用・旧機能劣化・安全性・availabilityを同じ入力で確認する。ここで新lossやthresholdを調整しない。

### 最低限の条件

```text
C0: v2入力 + 旧runtime
C1: C0 + new verifier
C2: C1 + COUNT↔BIND key/scoring
C3: C2 + SELECT encoding
C4: C3 + BIND scorer
C5: C4 + SHIFT version replacement
```

同じ親checkpointと入力を用いたcheckpoint lineageを保存する。変更点が二つ以上同時に入る比較はcausal deltaと呼ばない。必要なon/offアブレーションはdevelopment validationだけで行い、sealedは触らない。

### 評価

- nominal B2 matrix、same-identity inadequacy stress、COUNT↔BIND、SELECT/BIND、fresh-runtime SHIFT。
- 現在通過済みのlegacy K/C/N/R短縮回帰stream。Rは対応するNのcommit後だけ。
- UNCERTAIN→不必要なworkspace確保がないこと、wrong/不十分候補→ACCEPTがないこと、既存compositionを確認すること。
- parameter、calls、supportの実数、conditional/unconditional EM、coverage、median/p95 latency、peak VRAM。

### 成果物・Gate

`integration_matrix.json`、`legacy_regression.json`、`safety_and_availability.json`、`compute_accounting.json`。**G4**を満たすこと。ここで失敗したら原因機構の新タスクを別途設計し、自動で戻って複数修正を混ぜない。

---

> **B-C005R3-011 is blocked as of 2026-09-07 (ADR-0091/ADR-0092).** Do not
> start it until the `B-C005REC-00N` model bundle recovery branch
> (`docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`) reports a
> restored/rebuilt coherent parent cohort and an explicit next user
> instruction authorizes resuming R3-011. The task text below is preserved
> unchanged as the pre-block specification.

## B-C005R3-011 — B2 Protocol v2 Seal & Preflight

### 目的

新Gateを測る前に、評価対象・指標・予算・主張範囲を封印する。

### 実装

- R3-002の未評価sealed relation membershipを再確認。学習露出があればseal invalidとする。
- 完成model5個以上のhash、全component version、generator/split/data hash、loss exposure、metric schema、verifier error budget、H/search scope、sample countsをmanifestに固定する。
- nominal/stress/transfer suiteごとにmetric分母とquotaを確定。元B2矩形と新relation拡張の行数は別記する。
- 旧Gateの必須top-1/top-kを残し、unsafe reuse等の新指標を追加する。旧結果とは非同一protocolであることを宣言する。
- 実数値を見ないCPU/synthetic preflightで、欠損artifact、dataset reuse、test-time fitting、budget超過、post-seal mutationを検出する。
- `gate_runner` はtraining APIにアクセスしない。state hashをmatrix前後に検査する。

### STOP

このタスクではsealed出力を測らない。seal manifestを保存してSTOP。次のR3-012は別の明示指示で実行する。

---

## B-C005R3-012 — New Sealed B2 Gate v2 & Handoff

### 目的

新relation条件・新safety基準で修正結果を評価する唯一の正式再Gate。

### 実行

- R3-011のhash lockと同一のartifactだけを使用。
- nominal matrix、repair-relation transfer、same-identity inadequacy stress、必要なfresh-runtime検証を実行。
- per-model-seed、operation、relation、argument structure、bank size、difficulty別に報告する。
- legacy指標、新metric、分子/分母、UNRESOLVED、coverage、raw失敗例を保存。
- modelを選び直す、難しいrelationを外す、reference予算を成績に応じて増やすことは禁止。

### G5後の動作

**PASS:** 新ADRに `B2_PROTOCOL_V2_PASS` と記録し、旧B-C005/GはFAILのまま残す。B-C006を実行可能にするが、自動着手しない。

**FAIL / UNRESOLVED / RESOURCE_BLOCKED:** 原因と未測定範囲を保存し、B-C006はblockedのまま。sealed dataを再学習に回さない。追加修正は新しいユーザー指示を待つ。

### 最終成果物

`gate_result.json`、`report.md`、全matrix、confusion/uncertainty tables、failure traces、source/model/data manifest、旧Gateとの比較上の制限、新ADR。reportの冒頭に全Gate statusとB-C006の可否を書く。

---

## 推奨ファイル配置とCLI契約

既存moduleを優先する。下記の新moduleは同等の抽象化がない場合だけ追加する。

```text
src/apc/utils/seed_derivation.py                  # new if needed
src/apc/evaluation/recurrence_benchmark.py        # existing; inspect
src/apc/evaluation/consolidation_benchmark.py     # existing; inspect
src/apc/evaluation/semantic_relation_audit.py     # reported existing; inspect
src/apc/evaluation/adequacy_reference_audit.py    # existing; extend
src/apc/evaluation/functional_metrics_v2.py       # new if needed
src/apc/meta/adequacy_verifier.py                 # existing; versioned policy
src/apc/primitives/routing_losses.py             # existing; scoped change
src/apc/primitives/argument_scoring.py            # existing; scoped change
src/apc/evaluation/post_d2_repair_benchmark.py    # thin runner if needed
scripts/run_phase_b_b2_post_d2_repair.py          # explicit one-task dispatch
configs/phase_b_b2_post_d2_<task_slug>.yaml
tests/test_b2_<mechanism>.py
runs/phase_b_b2_post_d2/<task_slug>/<run_id>/
```

CLIを新設する場合の推奨形（まだ実装されていない）：

```bash
python scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-001 --config configs/phase_b_b2_post_d2_reproducibility.yaml --run-dir runs/phase_b_b2_post_d2/r3_001_reproducibility/run_001
```

`--all`や暗黙のnext-task実行は禁止。既存CLIが同じ目的を満たす場合は薄いdispatcherを重複実装せず、実commandをtask完了報告へ記載する。
